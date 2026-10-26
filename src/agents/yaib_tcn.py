"""The vitals agent with YAIB's TCN in place of Sentinel's.

YAIB (van de Water et al., ICLR 2024; arXiv:2306.05109) ships a temporal
convolutional network as one of its deep-learning baselines. This module ports
it so the vitals agent can be swapped and the change measured, both standalone
and inside the flagship (scripts/yaib_tcn_swap.py).

What is copied from YAIB (``rvandewater/YAIB`` @ development,
``icu_benchmarks/models/dl_models/{tcn,layers}.py`` and
``configs/prediction_models/TCN.gin`` + ``common/DLCommon.gin``):

* **Architecture**, verbatim: ``TemporalBlock`` = two weight-normed causal
  convolutions (symmetric padding then ``Chomp1d``), ReLU + dropout after
  each, a 1x1 residual projection when the width changes, ReLU on the sum.
  Dilation ``2**i`` at level ``i``; depth from ``max_seq_length`` and the
  kernel exactly as ``TemporalConvNet`` computes it, with
  ``max_seq_length = HORIZON = 24`` from ``configs/tasks/BinaryClassification.gin``.
* **Training recipe**: Adam with ``weight_decay = 1e-6``; class-"balanced"
  loss weights (``train_common.weight = "balanced"``); batch 64; up to 50
  epochs; early stopping on validation loss with patience 10 and
  ``min_delta = 1e-4``.
* **Search space**: lr log-uniform 1e-6..3e-4, channels log 32..256, kernel
  log 2..128, dropout 0..0.4.

What is NOT copied, on purpose, so the comparison isolates the network:

* **Inputs.** The same 19 channels as Sentinel's vitals agent (five vitals,
  their ages, GCS, the two flags, missingness masks), z-scored on train --
  not YAIB's 52 features. Swapping both would confound "better network" with
  "more data".
* **Tuning budget.** YAIB runs 30 Bayesian calls with 5-fold CV; the swap
  script runs a seeded random search over the same space on the val split.

One YAIB quirk is reproduced rather than fixed: ``init_weights`` writes
N(0, 0.01) into ``conv.weight.data``, but with ``torch.nn.utils.weight_norm``
that tensor is recomputed from ``weight_g``/``weight_v`` on every forward, so
the write is a no-op for the two convolutions (it does reach the residual 1x1).
Using the same API keeps the port faithful, quirk included.

The network is causal (left context only, no normalisation across time), so
the live engine's windowed peek applies unchanged with
:func:`receptive_field`.
"""

import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.vitals_tcn import (
    N_FEATURES,
    VitalsTCNAgent,
    build_sequences,
    collate,
    fit_normalizer,
    make_batches,
)

YAIB_HORIZON = 24                      # configs/tasks/BinaryClassification.gin

# YAIB's TCN search space (configs/prediction_models/TCN.gin).
SEARCH_SPACE = {
    "lr": (1e-6, 3e-4, "log"),
    "num_channels": (32, 256, "log-int"),
    "kernel_size": (2, 128, "log-int"),
    "dropout": (0.0, 0.4, "uniform"),
}


def _weight_norm(module):
    # The same deprecated API YAIB calls; the FutureWarning is noise here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return nn.utils.weight_norm(module, dim=None)


class Chomp1d(nn.Module):
    """From TCN original paper https://github.com/locuslab/TCN"""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """YAIB's ``TemporalBlock``, verbatim apart from the warning filter."""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 dropout=0.2):
        super().__init__()
        self.conv1 = _weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding,
                                            dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = _weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding,
                                            dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = (nn.Conv1d(n_inputs, n_outputs, 1)
                           if n_inputs != n_outputs else None)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


def yaib_levels(kernel_size, max_seq_length=YAIB_HORIZON):
    """``TemporalConvNet``'s depth rule: ceil(log(max_seq_length/2) / log k)."""
    return int(np.ceil(np.log(max_seq_length / 2) / np.log(kernel_size)))


def receptive_field(kernel_size, n_levels):
    """Two convolutions per block at dilation 2**i: 1 + 2(k-1)(2**L - 1)."""
    return 1 + 2 * (kernel_size - 1) * (2 ** n_levels - 1)


class YaibTCNNet(nn.Module):
    """YAIB's ``TemporalConvNet`` with Sentinel's agent interface.

    Returns ``(logits (B, T), embedding (B, T, C))`` like
    :class:`agents.vitals_tcn.VitalsTCNNet`, so the agent plumbing (batching,
    emission, calibration) and the live peek are shared. The embedding is
    YAIB's pre-logit representation; one logit replaces YAIB's two-class head
    (equivalent under the balanced cross-entropy used to train it).
    """

    def __init__(self, n_features=N_FEATURES, num_channels=64, kernel_size=2,
                 dropout=0.0, max_seq_length=YAIB_HORIZON):
        super().__init__()
        n_levels = yaib_levels(kernel_size, max_seq_length)
        channels = [num_channels] * n_levels
        layers = []
        for i in range(n_levels):
            dilation = 2 ** i
            layers.append(TemporalBlock(n_features if i == 0 else channels[i - 1],
                                        channels[i], kernel_size, stride=1,
                                        dilation=dilation,
                                        padding=(kernel_size - 1) * dilation,
                                        dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.logit = nn.Linear(channels[-1], 1)
        self.latent_dim = channels[-1]
        self.n_levels = n_levels
        self.receptive_field = receptive_field(kernel_size, n_levels)

    def forward(self, x):                      # x: (B, F, T), channel-first
        emb = self.network(x).transpose(1, 2)  # (B, T, C)
        return self.logit(emb).squeeze(-1), emb


class YaibTCNAgent(VitalsTCNAgent):
    """The vitals agent, YAIB's network and training recipe, Sentinel's inputs."""

    name = "vitals_yaib"

    def __init__(self, num_channels=64, kernel_size=2, dropout=0.0, lr=1e-4,
                 weight_decay=1e-6, batch_size=64, max_epochs=50, patience=10,
                 min_delta=1e-4, seed=42, device=None, verbose=True):
        super().__init__(lr=lr, batch_size=batch_size, max_epochs=max_epochs,
                         patience=patience, seed=seed, device=device, verbose=verbose)
        self.hparams = dict(num_channels=int(num_channels), kernel_size=int(kernel_size),
                            dropout=float(dropout))
        self.weight_decay, self.min_delta = weight_decay, min_delta
        self.latent_dim = int(num_channels)
        self.history = []

    def _weighted_loss(self, logits, y, mask, w_pos, w_neg):
        per = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        w = torch.where(y > 0.5, w_pos, w_neg) * mask
        return (per * w).sum() / mask.sum()

    @torch.no_grad()
    def _val_loss(self, val_seq, w_pos, w_neg):
        self.net.eval()
        total, seen = 0.0, 0.0
        for idx in make_batches(val_seq, self.batch_size):
            X, y, mask = collate(val_seq, idx, self.device)
            logits, _ = self.net(X)
            n = float(mask.sum())
            total += float(self._weighted_loss(logits, y, mask, w_pos, w_neg)) * n
            seen += n
        return total / seen

    def _fit_model(self, train_df, val_df):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.stats = fit_normalizer(train_df)
        train_seq = build_sequences(train_df, self.stats)
        val_seq = build_sequences(val_df, self.stats)
        self.net = YaibTCNNet(n_features=N_FEATURES, **self.hparams).to(self.device)

        # train_common.weight = "balanced": w_c = n / (2 n_c), from train labels.
        y_tr = np.concatenate([s["y"] for s in train_seq])
        n, n_pos = len(y_tr), float(y_tr.sum())
        w_pos = torch.tensor(n / (2.0 * n_pos), device=self.device)
        w_neg = torch.tensor(n / (2.0 * (n - n_pos)), device=self.device)

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        rng = np.random.default_rng(self.seed)
        best, best_state, stale = math.inf, None, 0
        for epoch in range(1, self.max_epochs + 1):
            self.net.train()
            for idx in make_batches(train_seq, self.batch_size, shuffle=True, rng=rng):
                X, y, mask = collate(train_seq, idx, self.device)
                logits, _ = self.net(X)
                loss = self._weighted_loss(logits, y, mask, w_pos, w_neg)
                opt.zero_grad()
                loss.backward()
                opt.step()
            vl = self._val_loss(val_seq, w_pos, w_neg)
            self.history.append(vl)
            if self.verbose:
                print(f"  epoch {epoch}: val loss {vl:.5f}")
            if vl < best - self.min_delta:
                best, stale = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val_loss = best
        self.epochs_run = len(self.history)
        return self


def sample_config(rng):
    """One draw from YAIB's TCN search space."""
    def log_uniform(lo, hi):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return {"lr": log_uniform(*SEARCH_SPACE["lr"][:2]),
            "num_channels": int(round(log_uniform(*SEARCH_SPACE["num_channels"][:2]))),
            "kernel_size": int(round(log_uniform(*SEARCH_SPACE["kernel_size"][:2]))),
            "dropout": float(rng.uniform(*SEARCH_SPACE["dropout"][:2]))}
