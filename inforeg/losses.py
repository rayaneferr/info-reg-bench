"""Explicit regularizers, all expressed as  L = CE + R(theta, p, z).

none / weight_decay : R = 0 (weight decay lives in the optimizer -> classic *non-informational* explicit reg)
label_smoothing     : CE against (1-eps) one-hot + eps/K uniform
                      == KL(u || p) up to constants   (Pereyra et al. 2017, Sec. 2)
confidence_penalty  : R = -beta * H(p)                (penalize low-entropy outputs; KL(p || u) direction)
vib                 : R =  beta * KL(q(z|x) || N(0,I)) (variational bound on I(X;Z); Alemi et al. 2017,
                                                       Mahabadi et al. 2021)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import Config


def compute_loss(outputs: dict, labels: torch.Tensor, cfg: Config):
    logits = outputs["logits"]
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    entropy = -(p * logp).sum(dim=-1).mean()

    if cfg.method == "label_smoothing":
        ce = F.cross_entropy(logits, labels, label_smoothing=cfg.epsilon)
    else:
        ce = F.cross_entropy(logits, labels)

    reg = logits.new_zeros(())
    if cfg.method == "confidence_penalty":
        reg = -cfg.beta * entropy
    elif cfg.method == "vib":
        reg = cfg.beta * outputs["kl"]

    loss = ce + reg
    stats = {
        "loss": loss.item(), "ce": ce.item(), "reg": reg.item(),
        "entropy": entropy.item(), "kl": float(outputs["kl"].item()),
    }
    return loss, stats
