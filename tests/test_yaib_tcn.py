"""YAIB's TCN, ported as a vitals agent: depth rule, causality, receptive field.

The live engine scores a provisional row from a window of RF-1 committed rows,
so the port must be causal AND its receptive field must be exactly what
``receptive_field`` claims -- checked here by perturbing the input one step
inside and one step outside the field.
"""

import numpy as np
import pytest
import torch

from agents.vitals_tcn import N_FEATURES
from agents.yaib_tcn import (
    YaibTCNNet,
    receptive_field,
    sample_config,
    yaib_levels,
)


@pytest.mark.parametrize("k, levels", [(2, 4), (3, 3), (4, 2), (8, 2), (12, 1), (128, 1)])
def test_depth_follows_yaibs_rule_at_horizon_24(k, levels):
    # ceil(log(24/2) / log k), TemporalConvNet's own formula
    assert yaib_levels(k) == levels
    assert YaibTCNNet(kernel_size=k, num_channels=8).n_levels == levels


def _net(k, c=16):
    torch.manual_seed(0)
    return YaibTCNNet(n_features=N_FEATURES, num_channels=c, kernel_size=k).eval()


@pytest.mark.parametrize("k", [2, 3, 5])
def test_yaib_tcn_is_causal(k):
    net = _net(k)
    x = torch.randn(1, N_FEATURES, 70)
    y = x.clone()
    y[:, :, 40:] = torch.randn(1, N_FEATURES, 30)          # rewrite the future
    with torch.no_grad():
        a, ea = net(x)
        b, eb = net(y)
    assert torch.allclose(a[:, :40], b[:, :40], atol=1e-6)
    assert torch.allclose(ea[:, :40], eb[:, :40], atol=1e-6)


@pytest.mark.parametrize("k", [2, 3, 5])
def test_receptive_field_is_exact(k):
    net = _net(k)
    rf = net.receptive_field
    assert rf == receptive_field(k, net.n_levels)
    T = rf + 20
    x = torch.randn(1, N_FEATURES, T)
    with torch.no_grad():
        base = net(x)[0][0, -1]
        outside = x.clone()
        outside[:, :, T - rf - 1] += 5.0                      # one step beyond the field
        inside = x.clone()
        inside[:, :, T - rf] += 5.0                           # the oldest step inside it
        assert torch.allclose(net(outside)[0][0, -1], base, atol=1e-6)
        assert not torch.allclose(net(inside)[0][0, -1], base, atol=1e-6)


def test_agent_interface_matches_the_vitals_tcn():
    net = _net(2, c=24)
    logits, emb = net(torch.randn(3, N_FEATURES, 11))
    assert logits.shape == (3, 11) and emb.shape == (3, 11, 24)
    assert net.latent_dim == 24
    assert (emb >= 0).all()          # YAIB's blocks end in ReLU


def test_search_space_draws_stay_inside_yaibs_bounds():
    rng = np.random.default_rng(0)
    for _ in range(200):
        c = sample_config(rng)
        assert 1e-6 <= c["lr"] <= 3e-4
        assert 32 <= c["num_channels"] <= 256
        assert 2 <= c["kernel_size"] <= 128
        assert 0.0 <= c["dropout"] <= 0.4
