"""Classifier heads on a tiny random backbone — no model download."""
from types import SimpleNamespace

import pytest
import torch

from inforeg.model import Classifier

H, K, Z = 16, 3, 4


class FakeBackbone(torch.nn.Module):
    """last_hidden_state[b, t] = t (constant over features), so pooling is readable from the output."""

    def forward(self, input_ids, attention_mask):
        b, t = input_ids.shape
        h = torch.arange(t, dtype=torch.float32).view(1, t, 1).expand(b, t, H)
        return SimpleNamespace(last_hidden_state=h)


def _batch():
    ids = torch.ones(2, 5, dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])   # right padding
    return ids, mask


def test_pool_takes_last_non_pad_token():
    model = Classifier(FakeBackbone(), H, K)
    pooled = model.pool(*_batch())
    assert pooled[:, 0].tolist() == [2.0, 4.0]


def test_linear_head_has_no_kl():
    torch.manual_seed(0)
    out = Classifier(FakeBackbone(), H, K).eval()(*_batch())
    assert out["logits"].shape == (2, K)
    assert out["kl"].item() == 0.0


@pytest.fixture
def vib():
    torch.manual_seed(0)
    return Classifier(FakeBackbone(), H, K, head="vib", z_dim=Z)


def test_vib_kl_matches_closed_form(vib):
    out = vib.eval()(*_batch())
    mu, logvar = vib.enc(out["pooled"]).chunk(2, dim=-1)
    expected = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).sum(-1).mean()
    assert torch.isclose(out["kl"], expected)
    assert out["kl"].item() >= 0.0


def test_vib_is_deterministic_in_eval_and_stochastic_in_train(vib):
    batch = _batch()
    vib.eval()
    assert torch.equal(vib(*batch)["logits"], vib(*batch)["logits"])
    vib.train()
    assert not torch.equal(vib(*batch)["logits"], vib(*batch)["logits"])


def test_vib_gradients_reach_the_encoder(vib):
    out = vib.train()(*_batch())
    (out["logits"].sum() + out["kl"]).backward()
    assert all(p.grad is not None for p in vib.enc.parameters())


def test_unknown_head_raises():
    with pytest.raises(ValueError, match="mlp"):
        Classifier(FakeBackbone(), H, K, head="mlp")
