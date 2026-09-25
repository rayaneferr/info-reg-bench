"""Accuracy, NLL, entropy, Expected Calibration Error and reliability bins."""
from __future__ import annotations

from itertools import pairwise

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Expected Calibration Error (Guo et al. 2017) + per-bin stats for reliability diagrams."""
    conf = probs.max(axis=-1)
    pred = probs.argmax(axis=-1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total, bins = 0.0, []
    for lo, hi in pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            bins.append({"lo": lo, "hi": hi, "count": 0, "acc": None, "conf": None})
            continue
        acc_b, conf_b = correct[m].mean(), conf[m].mean()
        total += m.mean() * abs(acc_b - conf_b)
        bins.append({"lo": float(lo), "hi": float(hi), "count": int(m.sum()),
                     "acc": float(acc_b), "conf": float(conf_b)})
    return float(total), bins


def summarize(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    probs = softmax(logits)
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    nll = float(-logp[np.arange(len(labels)), labels].mean())
    acc = float((probs.argmax(-1) == labels).mean())
    ent = float(-(probs * logp).sum(-1).mean())
    e, bins = ece(probs, labels, n_bins)
    return {"accuracy": acc, "nll": nll, "entropy": ent, "ece": e,
            "mean_confidence": float(probs.max(-1).mean()), "reliability": bins}


def mnli_to_hans(logits: np.ndarray) -> np.ndarray:
    """Collapse 3-way MNLI probs into HANS 2-way: [entailment, neutral+contradiction]."""
    p = softmax(logits)
    p2 = np.stack([p[:, 0], p[:, 1] + p[:, 2]], axis=-1)
    return np.log(np.clip(p2, 1e-12, 1.0))
