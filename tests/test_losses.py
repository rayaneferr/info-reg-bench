"""Unit tests that need no model download: closed-form KL, KL duality of LS / CP, metrics."""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from inforeg.config import Config
from inforeg.losses import compute_loss
from inforeg.metrics import ece, mnli_to_hans, softmax, summarize


def _out(logits, kl=0.0):
    return {"logits": logits, "kl": torch.tensor(kl)}


def test_none_is_plain_ce():
    logits = torch.randn(8, 3); y = torch.randint(0, 3, (8,))
    loss, st = compute_loss(_out(logits), y, Config(method="none"))
    assert torch.isclose(loss, F.cross_entropy(logits, y))
    assert st["reg"] == 0.0


def test_confidence_penalty_equals_kl_to_uniform_up_to_constant():
    """-beta*H(p) = beta*KL(p||u) - beta*log K"""
    torch.manual_seed(0)
    logits = torch.randn(16, 3); y = torch.randint(0, 3, (16,)); beta = 0.3
    loss, _ = compute_loss(_out(logits), y, Config(method="confidence_penalty", beta=beta))
    p = logits.softmax(-1); u = torch.full_like(p, 1 / 3)
    kl_pu = (p * (p.log() - u.log())).sum(-1).mean()
    expected = F.cross_entropy(logits, y) + beta * kl_pu - beta * math.log(3)
    assert torch.isclose(loss, expected, atol=1e-6)


def test_label_smoothing_equals_kl_from_uniform_up_to_constant():
    """CE_ls = (1-eps)*CE + eps*(H(u) + KL(u||p))"""
    torch.manual_seed(0)
    logits = torch.randn(16, 3); y = torch.randint(0, 3, (16,)); eps = 0.1
    loss, _ = compute_loss(_out(logits), y, Config(method="label_smoothing", epsilon=eps))
    logp = logits.log_softmax(-1); u = torch.full_like(logp, 1 / 3)
    kl_up = (u * (u.log() - logp)).sum(-1).mean()
    expected = (1 - eps) * F.cross_entropy(logits, y) + eps * (math.log(3) + kl_up)
    assert torch.isclose(loss, expected, atol=1e-6)


def test_vib_adds_beta_times_kl():
    logits = torch.randn(4, 3); y = torch.tensor([0, 1, 2, 0])
    loss, st = compute_loss(_out(logits, kl=5.0), y, Config(method="vib", beta=0.01))
    assert torch.isclose(loss, F.cross_entropy(logits, y) + 0.05)
    assert st["kl"] == 5.0


def test_gaussian_kl_closed_form_is_zero_at_prior():
    mu = torch.zeros(4, 8); logvar = torch.zeros(4, 8)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).sum(-1).mean()
    assert kl.item() == 0.0


def test_ece_is_zero_when_perfectly_calibrated():
    # half the samples at confidence .5 with 50% accuracy, half at 1.0 always right
    probs = np.array([[0.5, 0.5, 0.0]] * 100 + [[1.0, 0.0, 0.0]] * 100)
    labels = np.array([0] * 50 + [1] * 50 + [0] * 100)
    e, _ = ece(probs, labels, n_bins=10)
    assert e < 1e-9


def test_mnli_to_hans_merges_neutral_and_contradiction():
    logits = np.log(np.array([[0.2, 0.3, 0.5]]))
    p2 = softmax(mnli_to_hans(logits))
    assert np.allclose(p2, [[0.2, 0.8]])


def test_summarize_keys():
    rng = np.random.default_rng(0)
    s = summarize(rng.standard_normal((20, 3)), rng.integers(0, 3, 20))
    assert {"accuracy", "nll", "entropy", "ece", "reliability"} <= s.keys()


def test_run_name_encodes_hyperparameters():
    assert Config(method="vib", beta=1e-3, n_train=1000).run_name == "vib_n1000_b0.001_s0"
    assert Config(method="label_smoothing", epsilon=0.1).run_name == "label_smoothing_n1000_e0.1_s0"
    with pytest.raises(ValueError, match="method must be one of"):
        Config(method="nope")
