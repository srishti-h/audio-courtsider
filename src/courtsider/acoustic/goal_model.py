"""Learned audio goal detector: python -m courtsider.acoustic.goal_model [--epochs 6]

A causal temporal CNN over broadcast log-mel features (crowd + commentator), trained on the
SoccerNet train split to output P(goal in the last WINDOW seconds) 50 times a second. Causal means
each output only sees audio that has already arrived, so it runs live. The decision threshold is
picked on the validation split; detection metrics are reported on test, and per-half probability
tracks for test matches are saved for the market simulation.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from courtsider.acoustic.extract import FPS, HOP, SR, feature_path
from courtsider.paths import MODELS, PROCESSED, RESULTS

WINDOW = 6.0  # target: a goal happened in the last 6 s
BASELINE_S = 30.0
CROP_S = 40.0


def frame_times(n: int) -> np.ndarray:
    return (np.arange(n) * HOP + 1024) / SR


def normalise(mel: np.ndarray) -> np.ndarray:
    """Per band: subtract a causal 30 s exponential moving average (broadcast level / mixing), keep raw too."""
    x = mel.astype(np.float32)
    alpha = 1.0 / (BASELINE_S * FPS)
    ema = np.empty_like(x)
    ema[0] = x[0]
    for i in range(1, len(x)):  # vectorised over bands; ~0.3 s per half
        ema[i] = ema[i - 1] + alpha * (x[i] - ema[i - 1])
    return np.concatenate([(x - ema) / 2.0, (x + 8.0) / 6.0], axis=1)


def labels(n: int, goal_times: np.ndarray) -> np.ndarray:
    t = frame_times(n)
    y = np.zeros(n, dtype=np.float32)
    for g in goal_times:
        y[(t >= g) & (t <= g + WINDOW)] = 1.0
    return y


class CausalTCN(torch.nn.Module):
    def __init__(self, n_in: int = 128, ch: int = 64, dilations=(1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.inp = torch.nn.Conv1d(n_in, ch, 1)
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(ch, ch, 3, dilation=d) for d in dilations])
        self.norms = torch.nn.ModuleList([torch.nn.GroupNorm(8, ch) for _ in dilations])
        self.dilations = dilations
        self.out = torch.nn.Conv1d(ch, 1, 1)

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * sum(self.dilations)

    def forward(self, x):  # x: (B, T, C) -> logits (B, T)
        h = self.inp(x.transpose(1, 2))
        for conv, norm, d in zip(self.convs, self.norms, self.dilations, strict=True):
            r = torch.nn.functional.pad(h, (2 * d, 0))  # left padding only: causal
            h = h + torch.nn.functional.gelu(norm(conv(r)))
        return self.out(h)[:, 0]


def load_split(split: str) -> list[dict]:
    games = pd.read_parquet(PROCESSED / "games.parquet")
    events = pd.read_parquet(PROCESSED / "events.parquet")
    goals = events[events.event == "goal"]
    out = []
    for gid in games[games.split == split].game_id:
        for half in (1, 2):
            p = feature_path(gid, half)
            if not p.exists() or p.stat().st_size < 1_000_000:  # missing or truncated broadcast
                continue
            g = np.sort(goals[(goals.game_id == gid) & (goals.half == half)].t.to_numpy())
            g = g[np.concatenate([[True], np.diff(g) > 5])] if len(g) else g  # duplicate labels
            out.append(dict(game_id=gid, half=half, path=p, goals=g))
    return out


def _device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train(halves: list[dict], epochs: int, seed: int = 0) -> CausalTCN:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = _device()
    data = []
    for h in halves:
        x = normalise(np.load(h["path"]))
        if len(x) >= int(CROP_S * FPS):  # a few broadcasts are truncated to seconds of audio
            data.append((x, labels(len(x), h["goals"]), h["goals"]))
    print(f"training on {len(data)} halves", flush=True)
    model = CausalTCN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    crop, rf = int(CROP_S * FPS), model.receptive_field
    pos_weight = torch.tensor(8.0, device=dev)
    steps_per_epoch = 400
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * steps_per_epoch)
    for ep in range(epochs):
        tot = 0.0
        for _ in range(steps_per_epoch):
            xb, yb = [], []
            for _ in range(32):
                x, y, g = data[rng.integers(len(data))]
                if len(g) and rng.random() < 0.5:  # half the crops contain a goal
                    c = int(rng.choice(g) * FPS) - int(rng.uniform(5, CROP_S - 10) * FPS)
                else:
                    c = int(rng.integers(0, max(1, len(x) - crop)))
                c = int(np.clip(c, 0, max(0, len(x) - crop)))
                xb.append(x[c : c + crop])
                yb.append(y[c : c + crop])
            xb = torch.tensor(np.stack(xb), device=dev)
            yb = torch.tensor(np.stack(yb), device=dev)
            logits = model(xb)[:, rf:]  # skip the warm-up frames without full context
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb[:, rf:], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        print(f"epoch {ep + 1}/{epochs} loss {tot / steps_per_epoch:.4f}", flush=True)
    return model


@torch.no_grad()
def predict(model: CausalTCN, path) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    x = normalise(np.load(path))
    p = torch.sigmoid(model(torch.tensor(x[None], device=_device()))[0]).float().cpu().numpy()
    return frame_times(len(x)), p


def detect(t: np.ndarray, p: np.ndarray, th: float, refractory: float = 20.0) -> np.ndarray:
    above = p >= th
    out, last = [], -1e9
    for i in np.flatnonzero(above & ~np.concatenate([[False], above[:-1]])):
        if t[i] - last >= refractory:
            out.append(t[i])
            last = t[i]
    return np.array(out)


def evaluate(tracks: list[tuple[dict, np.ndarray, np.ndarray]], th: float) -> dict:
    delays, misses, fa, minutes = [], 0, 0, 0.0
    for h, t, p in tracks:
        dets = detect(t, p, th)
        used = np.zeros(len(dets), bool)
        for g in h["goals"]:
            ok = np.flatnonzero((dets >= g - 2) & (dets <= g + 30) & ~used)
            if len(ok):
                used[ok[0]] = True
                delays.append(float(dets[ok[0]] - g))
            else:
                misses += 1
        fa += int((~used).sum())
        minutes += t[-1] / 60
    n = len(delays) + misses
    return dict(
        threshold=th,
        goals=n,
        recall=round(len(delays) / max(1, n), 3),
        precision=round(len(delays) / max(1, len(delays) + fa), 3),
        delay_p50=round(float(np.median(delays)), 2) if delays else None,
        delay_p25=round(float(np.percentile(delays, 25)), 2) if delays else None,
        false_alarms_per_min=round(fa / max(minutes, 1e-9), 4),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--radio", action="store_true", help="add logged live-radio features (data.radio) to training")
    ap.add_argument("--min-radio-goals", type=int, default=30, help="with --radio: skip retraining below this")
    args = ap.parse_args()
    tr, va, te = load_split("train"), load_split("valid"), load_split("test")
    radio_tr, radio_te = [], []
    if args.radio:
        from courtsider.data import radio

        radio_tr, radio_te = radio.audio_halves("train"), radio.audio_halves("test")
        n = sum(len(h["goals"]) for h in radio_tr)
        if n < args.min_radio_goals:
            print(
                f"only {n} labelled radio goals with features (need {args.min_radio_goals}); keeping the current model"
            )
            return
        tr = tr + radio_tr
    print(f"halves: train {len(tr)} valid {len(va)} test {len(te)}", flush=True)
    model = train(tr, args.epochs)
    MODELS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODELS / "audio_goal_tcn.pt")
    va_raw = [(h, *predict(model, h["path"])) for h in va]
    # isotonic calibration on validation frames: P(goal in the last WINDOW s | network output)
    from sklearn.isotonic import IsotonicRegression

    xs = np.concatenate([p[::5] for _, _, p in va_raw])
    ys = np.concatenate([labels(len(p), h["goals"])[::5] for h, _, p in va_raw])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(xs, ys)
    import pickle

    (MODELS / "audio_goal_calibrator.pkl").write_bytes(pickle.dumps(iso))
    va_tracks = [(h, t, iso.predict(p)) for h, t, p in va_raw]
    grid = [evaluate(va_tracks, th) for th in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]
    f1 = lambda r: 2 * r["precision"] * r["recall"] / max(1e-9, r["precision"] + r["recall"])  # noqa: E731
    best = max(grid, key=f1)
    te_tracks = [(h, t, iso.predict(p)) for h, t, p in ((h, *predict(model, h["path"])) for h in te)]
    test = {str(th): evaluate(te_tracks, th) for th in sorted({0.1, 0.2, 0.3, 0.5, best["threshold"]})}
    report = dict(
        threshold_selected_on_valid=best["threshold"],
        valid_grid=grid,
        test=test,
        test_at_selected=evaluate(te_tracks, best["threshold"]),
        window_s=WINDOW,
        receptive_field_s=round(model.receptive_field / FPS, 2),
        radio_train_goals=sum(len(h["goals"]) for h in radio_tr),
    )
    if radio_te:  # 2026 radio, held out by time
        rt = [(h, t, iso.predict(p)) for h, t, p in ((h, *predict(model, h["path"])) for h in radio_te)]
        report["radio_2026_test"] = evaluate(rt, best["threshold"])
    (RESULTS / "audio_goal.json").write_text(json.dumps(report, indent=2))
    # probability tracks for the market simulation (test matches), downsampled to 10 Hz
    rows = []
    for h, t, p in te_tracks:
        rows.append(
            pd.DataFrame(dict(game_id=h["game_id"], half=h["half"], t=t[::5], p_goal=p[::5].astype(np.float32)))
        )
    pd.concat(rows).to_parquet(PROCESSED / "audio_goal_tracks.parquet")
    print(json.dumps(report["test_at_selected"], indent=1))


if __name__ == "__main__":
    main()
