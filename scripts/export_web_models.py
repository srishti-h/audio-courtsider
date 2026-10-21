"""Export the models for in-browser Live mode: python scripts/export_web_models.py

web/public/models/
  text/model.onnx (fp16) + tokenizer files    the event/team classifier (models/text_extractor)
  audio/model.onnx + mel.bin                   the causal audio goal detector and its log-mel filterbank
  meta.json                                    labels, thresholds, calibrator curve, feature settings, self-test

Each export is checked against PyTorch before it is written.
"""

from __future__ import annotations

import json
import pickle
import shutil

import librosa
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.transformers.float16 import convert_float_to_float16

from courtsider.acoustic.extract import HOP, N_MELS, SR, log_mel
from courtsider.acoustic.goal_model import BASELINE_S, WINDOW, CausalTCN, normalise
from courtsider.data.taxonomy import EVENT_TYPES
from courtsider.paths import MODELS, RESULTS, ROOT
from courtsider.text.dataset import TEAM_CLASSES
from courtsider.text.models import TransformerModel

OUT = ROOT / "web" / "public" / "models"
SAMPLES = [
    "Barcelona vs Real Madrid | Pedri on the ball, he looks up || GOAL! Lewandowski scores for Barcelona!",
    "Arsenal vs Chelsea | a quiet spell in midfield || throw-in to Chelsea near the halfway line",
    "Sevilla vs Betis | the referee reaches for his pocket || straight red card for the Betis defender",
]


def selftest_signal(seconds: float = 3.0) -> np.ndarray:
    """Deterministic test audio the browser can rebuild exactly (no random numbers)."""
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 3000 * t) * (t % 1.0)).astype(np.float32)


def export_text() -> None:
    m = TransformerModel.load(MODELS / "text_extractor")
    net = m.net.cpu().eval()
    out = OUT / "text"
    out.mkdir(parents=True, exist_ok=True)
    enc = m.tok(SAMPLES, padding=True, truncation=True, max_length=m.max_len, return_tensors="pt")
    raw = out / "model_fp32.onnx"
    torch.onnx.export(
        net,
        (enc["input_ids"], enc["attention_mask"]),
        raw,
        input_names=["input_ids", "attention_mask"],
        output_names=["events", "team"],
        dynamic_axes={k: {0: "batch", 1: "seq"} for k in ("input_ids", "attention_mask")}
        | {"events": {0: "batch"}, "team": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    # fp16, not int8: dynamic int8 moved probabilities by up to 0.16 (0.07 per-channel) on the samples;
    # fp16 stays within 0.001 at 163 MB
    onnx.save(convert_float_to_float16(onnx.load(raw), keep_io_types=True), out / "model.onnx")
    raw.unlink()
    for f in raw.parent.glob("model_fp32*"):  # external-data sidecar, if any
        f.unlink()
    m.tok.save_pretrained(out)
    with torch.no_grad():
        ev, tm = net(enc["input_ids"], enc["attention_mask"])
    sess = ort.InferenceSession(str(out / "model.onnx"))
    oev, otm = sess.run(None, {k: enc[k].numpy() for k in ("input_ids", "attention_mask")})
    gap = np.abs(torch.sigmoid(ev).numpy() - 1 / (1 + np.exp(-oev))).max()
    assert gap < 0.01, f"fp16 text model drifts from PyTorch by {gap:.3f}"
    print(f"text model: {(out / 'model.onnx').stat().st_size / 1e6:.0f} MB, max prob gap vs PyTorch {gap:.4f}")


def export_audio() -> dict:
    net = CausalTCN()
    net.load_state_dict(torch.load(MODELS / "audio_goal_tcn.pt", map_location="cpu"))
    net.eval()
    out = OUT / "audio"
    out.mkdir(parents=True, exist_ok=True)
    x = torch.tensor(normalise(log_mel(selftest_signal()))[None])
    torch.onnx.export(
        net,
        (x,),
        out / "model.onnx",
        input_names=["mel"],
        output_names=["logits"],
        dynamic_axes={"mel": {1: "frames"}, "logits": {1: "frames"}},
        opset_version=17,
        dynamo=False,
    )
    with torch.no_grad():
        ref = net(x).numpy()
    got = ort.InferenceSession(str(out / "model.onnx")).run(None, {"mel": x.numpy()})[0]
    assert np.abs(ref - got).max() < 1e-3
    fb = librosa.filters.mel(sr=SR, n_fft=1024, n_mels=N_MELS, fmax=8000).astype(np.float32)
    (out / "mel.bin").write_bytes(fb.tobytes())
    iso = pickle.loads((MODELS / "audio_goal_calibrator.pkl").read_bytes())
    report = json.loads((RESULTS / "audio_goal.json").read_text())
    mel = log_mel(selftest_signal()).astype(np.float32)
    print(f"audio model: {(out / 'model.onnx').stat().st_size / 1e3:.0f} kB")
    return dict(
        sr=SR,
        n_fft=1024,
        hop=HOP,
        n_mels=N_MELS,
        baseline_s=BASELINE_S,
        window_s=WINDOW,
        threshold=report["threshold_selected_on_valid"],
        calibrator=dict(x=iso.X_thresholds_.tolist(), y=iso.y_thresholds_.tolist()),
        # the browser rebuilds selftest_signal() and must match these log-mel frames and the last logit
        selftest=dict(
            frames=[0, 50, 100], logmel=[mel[i].tolist() for i in (0, 50, 100)], last_logit=float(ref[0, -1])
        ),
    )


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    export_text()
    meta = dict(event_types=EVENT_TYPES, team_classes=list(TEAM_CLASSES), max_len=96, audio=export_audio())
    (OUT / "meta.json").write_text(json.dumps(meta))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
