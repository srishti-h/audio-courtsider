"""Three text event extractors of increasing capacity, sharing a predict_proba interface."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from courtsider.data.taxonomy import EVENT_TYPES
from courtsider.text.dataset import TEAM_CLASSES

# ---- 1. keyword rules -------------------------------------------------------------------

KEYWORDS = {
    "goal": r"\bgoa+l+\b|\bscores?\b|\bin the net\b|\bback of the net\b",
    "shot": r"\bshot\b|\bshoots?\b|\bsaves?d?\b|\bwide\b|\bover the bar\b|\bpost\b",
    "corner": r"\bcorner",
    "yellow_card": r"\byellow\b|\bbooked\b|\bbooking\b|\bcaution",
    "red_card": r"\bred card\b|\bsent off\b|\bsending off\b|\bsecond yellow\b",
    "penalty": r"\bpenalt|\bspot kick\b",
    "offside": r"\boffside\b|\bflag is up\b",
    "substitution": r"\bsubstitut|\bcomes on\b|\bcoming on\b|\breplaces?\b",
}
_KW = {k: re.compile(v, re.I) for k, v in KEYWORDS.items()}


def current_segment(text: str) -> str:
    return text.split("||", 1)[-1]


class KeywordModel:
    name = "keywords"

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        cur = [current_segment(t) for t in texts]
        return np.array([[1.0 if _KW[e].search(c) else 0.0 for e in EVENT_TYPES] for c in cur])


# ---- 2. TF-IDF + logistic regression ----------------------------------------------------


class TfidfModel:
    name = "tfidf_lr"

    def __init__(self) -> None:
        self.vec_cur = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)
        self.vec_ctx = TfidfVectorizer(ngram_range=(1, 1), min_df=3, max_features=50_000, sublinear_tf=True)
        self.clfs: list[LogisticRegression] = []

    def _features(self, texts: list[str], fit: bool = False):
        from scipy.sparse import hstack

        cur = [current_segment(t) for t in texts]
        ctx = [t.split("|", 1)[-1] for t in texts]
        if fit:
            return hstack([self.vec_cur.fit_transform(cur), self.vec_ctx.fit_transform(ctx)]).tocsr()
        return hstack([self.vec_cur.transform(cur), self.vec_ctx.transform(ctx)]).tocsr()

    def fit(self, texts: list[str], y: np.ndarray) -> TfidfModel:
        X = self._features(texts, fit=True)
        self.clfs = [LogisticRegression(C=4.0, max_iter=2000).fit(X, y[:, k]) for k in range(y.shape[1])]
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self._features(texts)
        return np.column_stack([c.predict_proba(X)[:, 1] for c in self.clfs])


# ---- 3. fine-tuned transformer ----------------------------------------------------------


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class TransformerExtractor(torch.nn.Module):
    """Shared encoder with a multi-label event head and a 3-way team head."""

    def __init__(self, backbone: str = "distilroberta-base"):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(backbone)
        h = self.encoder.config.hidden_size
        self.event_head = torch.nn.Linear(h, len(EVENT_TYPES))
        self.team_head = torch.nn.Linear(h, len(TEAM_CLASSES))

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.event_head(x), self.team_head(x)


class TransformerModel:
    name = "distilroberta"

    def __init__(self, backbone: str = "distilroberta-base", max_len: int = 96):
        from transformers import AutoTokenizer

        self.backbone, self.max_len = backbone, max_len
        self.tok = AutoTokenizer.from_pretrained(backbone)
        self.tok.truncation_side = "left"  # keep the current segment, drop the oldest context
        self.net = TransformerExtractor(backbone).to(device())

    def _encode(self, texts: list[str]):
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        return enc["input_ids"].to(device()), enc["attention_mask"].to(device())

    def fit(
        self,
        texts: list[str],
        y: np.ndarray,
        team: np.ndarray,
        epochs: int = 1,
        bs: int = 64,
        lr: float = 3e-5,
        log_every: int = 200,
    ) -> TransformerModel:
        dev = device()
        prior = y.mean(0)
        pos_weight = torch.tensor(
            np.clip((1 - prior) / np.maximum(prior, 1e-4), 1, 20) ** 0.5, dtype=torch.float32, device=dev
        )
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        ce = torch.nn.CrossEntropyLoss()
        team_idx = np.array([TEAM_CLASSES.index(t) for t in team])
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * int(np.ceil(len(texts) / bs))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.06)
        self.net.train()
        rng = np.random.default_rng(0)
        step, t0 = 0, time.time()
        for _ in range(epochs):
            order = rng.permutation(len(texts))
            for i in range(0, len(order), bs):
                b = order[i : i + bs]
                ids, mask = self._encode([texts[j] for j in b])
                ev_logits, team_logits = self.net(ids, mask)
                loss = bce(ev_logits, torch.tensor(y[b], dtype=torch.float32, device=dev))
                has_team = torch.tensor(team_idx[b] != 2, device=dev)
                if has_team.any():
                    tb = torch.tensor(team_idx[b], device=dev)
                    loss = loss + 0.5 * ce(team_logits[has_team], tb[has_team])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                sched.step()
                step += 1
                if step % log_every == 0:
                    rate = (time.time() - t0) / step
                    print(f"step {step}/{steps} loss {loss.item():.4f} {rate:.3f}s/step", flush=True)
        return self

    @torch.no_grad()
    def predict(self, texts: list[str], bs: int = 256) -> tuple[np.ndarray, np.ndarray]:
        self.net.eval()
        ev, tm = [], []
        for i in range(0, len(texts), bs):
            ids, mask = self._encode(texts[i : i + bs])
            e, t = self.net(ids, mask)
            ev.append(torch.sigmoid(e).float().cpu().numpy())
            tm.append(torch.softmax(t, -1).float().cpu().numpy())
        return np.concatenate(ev), np.concatenate(tm)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.predict(texts)[0]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "weights.pt")
        self.tok.save_pretrained(path)
        (path / "backbone.txt").write_text(f"{self.backbone}\n{self.max_len}\n")

    @classmethod
    def load(cls, path: Path) -> TransformerModel:
        backbone, max_len = (path / "backbone.txt").read_text().split()
        m = cls(backbone, int(max_len))
        m.net.load_state_dict(torch.load(path / "weights.pt", map_location=device()))
        m.net.eval()
        return m
