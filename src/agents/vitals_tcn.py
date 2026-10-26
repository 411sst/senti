"""Agent #1: the vitals TCN -- the anchor agent.

Reads the hourly vitals channels from the MIMIC cohort and emits, per
patient-hour, a latent embedding and a calibrated deterioration risk (the
:class:`~agents.base.Agent` contract).

Why a TCN rather than per-hour tabular features: the project's claim is about
reasoning over **trajectory**. A dilated causal convolution sees a window of
recent history at every hour, so the embedding encodes *how the patient has been
moving*, not just where they are now -- which is exactly what the orchestrator
is meant to fuse.

**Causality is enforced twice over.**

1. The cohort's feature columns are already strictly-before-``t`` (LOCF then
   shift, in :mod:`mimic_cohort`).
2. Every convolution here is *causal*: the input is left-padded only, so the
   output at hour ``t`` is a function of inputs at hours ``<= t`` and never of
   later ones. ``tests/test_agents.py`` verifies this empirically by rewriting
   the future and checking the past does not move.

The model is deliberately small (~30k parameters, 3 dilated blocks) to stay
inside free-tier compute, per CLAUDE.md.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import KEY_COLUMNS, Agent, AgentOutput
from config import SEED
from mimic_cohort import VITALS

LABEL = "label_24h"

# Channels the agent reads. Continuous channels are z-scored on train
# statistics; each also contributes a "was this missing?" mask channel, so the
# network can tell an imputed value from a real one.
CONTINUOUS = VITALS + [f"{v}_hrs_since" for v in VITALS] + ["gcs"]
BINARY = ["gcs_unable", "on_oxygen"]
MASKED = VITALS + ["gcs"]          # channels that get a missingness indicator
N_FEATURES = len(CONTINUOUS) + len(BINARY) + len(MASKED)


def pick_device(prefer_mps=False):
    """Choose a training device. CUDA if present, else CPU.

    Apple's MPS backend is deliberately NOT the default: it crashes with a hard
    Metal assertion ("Weights tensor and ndArray input channel mismatch") partway
    through training on this model, and because that is an assertion rather than
    a Python exception it cannot be caught or retried. Individual forward and
    backward passes succeed, so the bug is data-shape dependent and not worth
    chasing for a ~30k-parameter network. Pass ``prefer_mps=True`` to opt in.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- Model -----------------------------------------------------------------

class CausalConv1d(nn.Module):
    """1D convolution padded on the LEFT only.

    This is what makes the network causal: with ``(kernel-1)*dilation`` zeros
    prepended and none appended, output position ``t`` reads input positions
    ``t-(k-1)d .. t`` -- strictly the present and past.
    """

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):                      # x: (B, C, T)
        if self.left_pad:
            x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the CHANNEL axis only, independently at each timestep.

    Normalisation is a sneaky way to break causality: anything that pools
    statistics across time (``BatchNorm1d``, or ``GroupNorm`` over ``(C, T)``)
    lets a future timestep shift the normalised value of a past one, even when
    every convolution is causal. Normalising each timestep over its own channels
    keeps the time axis untouched. ``tests/test_agents.py::test_network_is_causal``
    is what caught this.
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):                      # (B, C, T)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TCNBlock(nn.Module):
    """Two causal convolutions at one dilation, with a residual connection."""

    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = ChannelLayerNorm(channels)
        self.norm2 = ChannelLayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.relu(self.norm1(self.conv1(x))))
        h = self.drop(F.relu(self.norm2(self.conv2(h))))
        return x + h


class VitalsTCNNet(nn.Module):
    """Dilated causal TCN. Returns per-hour logits AND the penultimate embedding.

    With kernel 3 and dilations (1, 2, 4) over two convolutions per block the
    receptive field is 29 hours -- roughly a day of trajectory behind every
    prediction.
    """

    def __init__(self, n_features=N_FEATURES, channels=32, kernel_size=3,
                 dilations=(1, 2, 4), dropout=0.1):
        super().__init__()
        self.project = CausalConv1d(n_features, channels, 1)
        self.blocks = nn.ModuleList(
            TCNBlock(channels, kernel_size, d, dropout) for d in dilations
        )
        self.head = nn.Linear(channels, 1)
        self.latent_dim = channels

    def forward(self, x):                      # x: (B, F, T)
        h = self.project(x)
        for block in self.blocks:
            h = block(h)
        embedding = h.transpose(1, 2)          # (B, T, C) <- the latent vector
        logits = self.head(embedding).squeeze(-1)   # (B, T)
        return logits, embedding


# --- Sequence plumbing -----------------------------------------------------

def fit_normalizer(train_df):
    """Per-channel mean/std from the TRAIN split only."""
    stats = {}
    for col in CONTINUOUS:
        v = pd.to_numeric(train_df[col], errors="coerce").to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        mean = float(v.mean()) if v.size else 0.0
        std = float(v.std()) if v.size else 1.0
        stats[col] = (mean, std if std > 1e-6 else 1.0)
    return stats


def encode_features(df, stats):
    """Build the (N, F) feature matrix: z-scored channels + missingness masks."""
    parts = []
    for col in CONTINUOUS:
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mean, std = stats[col]
        z = (v - mean) / std
        parts.append(np.nan_to_num(z, nan=0.0))   # missing -> the channel mean
    for col in BINARY:
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        parts.append(np.nan_to_num(v, nan=0.0))
    for col in MASKED:
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        parts.append(np.isnan(v).astype(float))   # 1 = this value was missing
    return np.stack(parts, axis=1).astype(np.float32)


def build_sequences(df, stats, with_labels=True):
    """Split the frame into per-stay sequences, ordered by hour.

    Returns a list of dicts with the stay's feature matrix, hours, and labels.
    Sequence order follows the order stays appear in ``df``.
    """
    frame = df.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    X = encode_features(frame, stats)
    y = (frame[LABEL].to_numpy(dtype=float) if with_labels
         else np.zeros(len(frame), dtype=float))
    stay = frame["stay_id"].to_numpy()
    hour = frame["hour"].to_numpy()

    # Contiguous runs of the same stay_id (frame is sorted, so runs are stays).
    boundaries = np.flatnonzero(np.diff(stay)) + 1
    sequences = []
    for lo, hi in zip(np.r_[0, boundaries], np.r_[boundaries, len(frame)]):
        sequences.append({
            "stay_id": stay[lo], "hours": hour[lo:hi],
            "X": X[lo:hi], "y": y[lo:hi],
        })
    return sequences


def make_batches(sequences, batch_size, shuffle=False, rng=None):
    """Length-bucketed batches, so padding waste stays small."""
    order = np.argsort([len(s["hours"]) for s in sequences], kind="stable")
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        (rng or np.random.default_rng(SEED)).shuffle(batches)
    return batches


def collate(sequences, idx, device):
    """Pad a batch to its longest sequence and return a validity mask."""
    items = [sequences[i] for i in idx]
    lengths = [len(s["hours"]) for s in items]
    T, B = max(lengths), len(items)
    X = np.zeros((B, items[0]["X"].shape[1], T), dtype=np.float32)
    y = np.zeros((B, T), dtype=np.float32)
    mask = np.zeros((B, T), dtype=np.float32)
    for i, s in enumerate(items):
        n = len(s["hours"])
        X[i, :, :n] = s["X"].T
        y[i, :n] = s["y"]
        mask[i, :n] = 1.0
    return (torch.from_numpy(X).to(device),
            torch.from_numpy(y).to(device),
            torch.from_numpy(mask).to(device))


# --- The agent -------------------------------------------------------------

class VitalsTCNAgent(Agent):
    """Vitals trajectory agent: latent embedding + calibrated risk per hour."""

    name = "vitals_tcn"

    def __init__(self, channels=32, kernel_size=3, dilations=(1, 2, 4),
                 dropout=0.1, lr=1e-3, batch_size=64, max_epochs=8, patience=2,
                 seed=SEED, device=None, verbose=True):
        super().__init__()
        self.hparams = dict(channels=channels, kernel_size=kernel_size,
                            dilations=dilations, dropout=dropout)
        self.lr, self.batch_size = lr, batch_size
        self.max_epochs, self.patience = max_epochs, patience
        self.seed, self.verbose = seed, verbose
        self.device = device or pick_device()
        self.latent_dim = channels
        self.net = None
        self.stats = None

    # -- training ---------------------------------------------------------
    def _fit_model(self, train_df, val_df):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if not self.warm_start:                  # fine-tuning keeps source stats
            self.stats = fit_normalizer(train_df)
        train_seq = build_sequences(train_df, self.stats)
        val_seq = build_sequences(val_df, self.stats)

        if not self.warm_start:
            self.net = VitalsTCNNet(n_features=N_FEATURES, **self.hparams).to(self.device)
        n_params = sum(p.numel() for p in self.net.parameters())
        if self.verbose:
            print(f"[{self.name}] {n_params:,} params on {self.device}; "
                  f"{len(train_seq):,} train stays, {len(val_seq):,} val stays")

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.seed)
        from metrics import auprc          # local import avoids a cycle

        def val_auprc_now():
            logits, y = self._raw_logits(val_seq)
            return auprc(y, logits)

        best_auprc, best_state = self._initial_best(val_auprc_now)
        stale = 0

        for epoch in range(1, self.max_epochs + 1):
            self.net.train()
            total, seen = 0.0, 0
            for idx in make_batches(train_seq, self.batch_size, shuffle=True, rng=rng):
                X, y, mask = collate(train_seq, idx, self.device)
                logits, _ = self.net(X)
                # Mean BCE over real (non-padded) hours only.
                loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none")
                        * mask).sum() / mask.sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                n_valid = float(mask.sum())
                total += float(loss.detach()) * n_valid
                seen += n_valid

            val_logits, val_y = self._raw_logits(val_seq)
            val_auprc = auprc(val_y, val_logits)
            if self.verbose:
                print(f"  epoch {epoch}: train loss {total/seen:.4f} | "
                      f"val AUPRC {val_auprc:.4f}")

            if val_auprc > best_auprc:
                best_auprc, stale = val_auprc, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    if self.verbose:
                        print(f"  early stop (no val gain for {self.patience} epochs)")
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        if self.verbose:
            print(f"[{self.name}] best val AUPRC {best_auprc:.4f}")
        # Calibration is NOT done here -- Agent.fit() applies isotonic-on-val
        # after this returns, so every agent calibrates identically.
        return self

    # -- inference --------------------------------------------------------
    @torch.no_grad()
    def _forward(self, sequences, want_embedding=False):
        """Run the net over sequences; return per-hour logits (+ embeddings)."""
        self.net.eval()
        logits_out, emb_out, keys = [], [], []
        for idx in make_batches(sequences, self.batch_size):
            X, _, mask = collate(sequences, idx, self.device)
            logits, emb = self.net(X)
            logits, emb, mask = logits.cpu().numpy(), emb.cpu().numpy(), mask.cpu().numpy()
            for row, i in enumerate(idx):
                n = int(mask[row].sum())
                logits_out.append(logits[row, :n])
                if want_embedding:
                    emb_out.append(emb[row, :n])
                seq = sequences[i]
                keys.append(np.stack([np.full(n, seq["stay_id"]), seq["hours"]], axis=1))
        return (np.concatenate(logits_out),
                np.concatenate(emb_out) if want_embedding else None,
                np.concatenate(keys))

    def _raw_logits(self, sequences):
        logits, _, keys = self._forward(sequences)
        y = np.concatenate([
            sequences[i]["y"] for idx in make_batches(sequences, self.batch_size)
            for i in idx
        ])
        return logits, y

    def _emit_raw(self, df):
        """``(keys, latent, raw_probability)`` sorted by ``(stay_id, hour)``.

        Inference batches are length-bucketed for speed, so the network's
        natural output order is by sequence length; re-sorting here means
        callers (and the base class's calibrator) can rely on a stable order
        instead of that internal detail.
        """
        if self.net is None:
            raise RuntimeError("agent is not fitted -- call fit() first")
        sequences = build_sequences(df, self.stats, with_labels=False)
        logits, emb, keys = self._forward(sequences, want_embedding=True)
        order = np.lexsort((keys[:, 1], keys[:, 0]))     # by stay_id, then hour
        logits, emb, keys = logits[order], emb[order], keys[order]
        raw = 1.0 / (1.0 + np.exp(-logits))              # uncalibrated
        return (pd.DataFrame(keys, columns=KEY_COLUMNS),
                emb.astype(np.float32), raw)


