"""Tests for the V5 pretraining stack (Mamba encoder + multi-task pretraining)."""

import numpy as np
import pytest
import torch

from agents.ecg_mamba_encoder import EcgMambaEncoder, MambaBlock
from agents.ecg_pretrain import EcgPretrainModel, augment
from agents.ecg_resnet_encoder import EcgMorphologyEncoder


# --- Mamba encoder ----------------------------------------------------------

def test_shapes_and_param_budget():
    m = EcgMambaEncoder()
    r = EcgMorphologyEncoder(in_channels=12, final_embedding_size=256)
    pm = sum(p.numel() for p in m.parameters())
    pr = sum(p.numel() for p in r.parameters())
    x = torch.randn(2, 12, 5000)
    assert m(x).shape == (2, 256)
    assert r(x).shape == (2, 256)
    # fair A/B: within ~15% of each other, both ~4M
    assert abs(pm - pr) / max(pm, pr) < 0.15
    assert 3e6 < pm < 5e6


def test_reverse_block_scans_the_other_direction():
    torch.manual_seed(0)
    fwd = MambaBlock(d_model=32, expand=2, d_state=4, dt_rank=4, reverse=False)
    rev = MambaBlock(d_model=32, expand=2, d_state=4, dt_rank=4, reverse=True)
    rev.load_state_dict(fwd.state_dict())        # identical weights
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        yf, yr = fwd(x), rev(x)
    assert not torch.allclose(yf, yr, atol=1e-5)
    # a causal check: with a forward scan, early outputs cannot depend on late
    # inputs. Perturb the LAST timestep; the FIRST output must not move
    # (conv padding is causal-left in this implementation).
    x2 = x.clone()
    x2[:, -1] = 10.0
    with torch.no_grad():
        assert torch.allclose(fwd(x)[:, 0], fwd(x2)[:, 0], atol=1e-4)
        assert not torch.allclose(rev(x)[:, 0], rev(x2)[:, 0], atol=1e-4)


def test_seed_determinism():
    torch.manual_seed(7)
    a = EcgMambaEncoder()
    torch.manual_seed(7)
    b = EcgMambaEncoder()
    x = torch.randn(1, 12, 5000)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-6)


def test_checkpointed_training_grads_match_plain():
    """Checkpointing is a memory optimisation -- gradients must be identical."""
    import copy
    torch.manual_seed(1)
    a = EcgMambaEncoder(n_blocks=2, d_model=32, dt_rank=4, d_state=4)
    b = copy.deepcopy(a)
    b.use_checkpoint = False
    x = torch.randn(2, 12, 5000)
    a.train(); b.train()
    a(x).pow(2).mean().backward()
    b(x).pow(2).mean().backward()
    ga = a.blocks[0].in_proj.weight.grad
    gb = b.blocks[0].in_proj.weight.grad
    assert torch.allclose(ga, gb, atol=1e-5)


# --- Pretraining pieces -----------------------------------------------------

def test_pretrain_model_heads_and_embed():
    torch.manual_seed(0)
    model = EcgPretrainModel(EcgMorphologyEncoder(12, 256), n_flags=15, n_values=9)
    x = torch.randn(3, 12, 5000)
    logits, preds = model(x)
    assert logits.shape == (3, 15) and preds.shape == (3, 9)
    assert model.embed(x).shape == (3, 256)


def test_augment_preserves_shape_and_is_bounded():
    rng = np.random.default_rng(0)
    x = torch.randn(8, 12, 5000)
    y = augment(x.clone(), rng)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    # amplitude ops only: the augmented signal cannot exceed a generous bound
    assert y.abs().max() < x.abs().max() * 2.0 + 1.0


def test_augment_never_timewarps():
    """Two impulses 1000 samples apart must stay exactly 1000 apart (shift is a
    rigid roll) -- time-warping would corrupt the RR/QT regression targets."""
    rng = np.random.default_rng(3)
    x = torch.zeros(4, 12, 5000)
    x[..., 2000] = 5.0
    x[..., 3000] = 5.0
    y = augment(x.clone(), rng)
    for b in range(4):
        for c in range(12):
            lead = y[b, c].abs()
            if lead.max() < 1.0:          # this lead was dropped
                continue
            top2 = torch.topk(lead, 2).indices.sort().values
            assert int(top2[1] - top2[0]) == 1000, "impulse spacing changed"
