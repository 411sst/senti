"""The orchestrator: latent fusion + trajectory reasoning over agent signals.

This is the project's contribution. Naive multi-agent systems make agents talk
in **text**, which is lossy: a per-modality model's internal state gets
compressed into a sentence, and everything not expressible in that sentence is
discarded. Here the agents pass their **latent vectors** directly -- no natural
language anywhere in the pipeline -- and the orchestrator reasons over how those
latents *move over time*.

Two mechanisms, both deliberately small (~25k parameters, free-tier compute):

1. **Latent fusion (across agents, per hour).** Each agent's latent is
   projected to a common width and combined by a learned attention over agents.
   Attention rather than plain concatenation buys one thing that matters for
   the write-up: the weights say *which agent is driving the risk at this hour*,
   and they are inspectable per patient-hour.
2. **Trajectory reasoning (across time).** The fused per-hour vectors go through
   a dilated **causal** TCN, so the risk at hour ``t`` is a function of how the
   agents' signals have been developing over the preceding ~29 hours, not just
   where they sit now.

**Leakage.** Two independent guarantees stack here. Every agent latent at hour
``t`` was already built from strictly-before-``t`` data, and the temporal stack
is causal (left-padded convolutions, per-timestep normalisation). The attention
is over *agents within one hour*, never over time, so it cannot see forward
either. ``tests/test_orchestrator.py::test_orchestrator_is_causal`` verifies the
whole stack empirically by rewriting the future and checking the past is
unmoved.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import KEY_COLUMNS, Agent, AgentOutput
from agents.vitals_tcn import TCNBlock, collate, make_batches
from config import SEED

LABEL = "label_24h"
DEFAULT_AGENTS = ("vitals_tcn", "news2", "labs")
EMB_DIR = "data/cache/embeddings"


# --- Loading the agents' cached emissions ----------------------------------

class AgentBank:
    """Cached agent latents + risks, addressable by ``(stay_id, hour)``.

    Agents emit on matched keys, so their rows line up one-for-one. This class
    loads each agent's cache, *verifies* that alignment rather than assuming it,
    and concatenates them into a single feature matrix laid out as

        [ agent_1 latent | ... | agent_n latent | agent_1..n calibrated risk ]
    """

    def __init__(self, keys, matrix, latent_dims, agent_names):
        self.keys = keys
        self.matrix = matrix
        self.latent_dims = latent_dims
        self.agent_names = list(agent_names)
        self._index = pd.MultiIndex.from_arrays(
            [keys["stay_id"].to_numpy(), keys["hour"].to_numpy()]
        )

    @property
    def n_features(self):
        return self.matrix.shape[1]

    @classmethod
    def load(cls, splits=("train", "val", "test"), agent_names=DEFAULT_AGENTS,
             emb_dir=EMB_DIR):
        from pathlib import Path
        emb_dir = Path(emb_dir)
        per_split = []
        for split in splits:
            outputs = {
                name: AgentOutput.load(emb_dir / f"{name}_{split}.npz")
                for name in agent_names
            }
            reference = outputs[agent_names[0]].keys
            for name, out in outputs.items():
                if not out.keys.equals(reference):
                    raise ValueError(
                        f"agent '{name}' emitted different keys than "
                        f"'{agent_names[0]}' on split '{split}' -- agents must "
                        f"emit row-for-row on matched (stay_id, hour)"
                    )
            block = np.concatenate(
                [outputs[n].latent for n in agent_names]
                + [np.stack([outputs[n].risk for n in agent_names], axis=1)],
                axis=1,
            ).astype(np.float32)
            per_split.append((reference, block))

        keys = pd.concat([k for k, _ in per_split], ignore_index=True)
        matrix = np.concatenate([m for _, m in per_split], axis=0)
        latent_dims = {
            n: AgentOutput.load(emb_dir / f"{n}_{splits[0]}.npz").latent.shape[1]
            for n in agent_names
        }
        return cls(keys, matrix, latent_dims, agent_names)

    def lookup(self, df):
        """Feature rows for ``df``'s ``(stay_id, hour)``, in sorted key order."""
        ordered = df.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        want = pd.MultiIndex.from_arrays(
            [ordered["stay_id"].to_numpy(), ordered["hour"].to_numpy()]
        )
        pos = self._index.get_indexer(want)
        if (pos < 0).any():
            raise ValueError(
                f"{int((pos < 0).sum())} cohort rows have no cached agent "
                f"emission -- re-run the agent training scripts"
            )
        return self.matrix[pos], ordered[list(KEY_COLUMNS)].copy()

    def split_features(self, x):
        """Split a feature tensor into per-agent latents and the risk block."""
        out, offset = {}, 0
        for name in self.agent_names:
            dim = self.latent_dims[name]
            out[name] = x[..., offset:offset + dim]
            offset += dim
        return out, x[..., offset:offset + len(self.agent_names)]


# --- Model -----------------------------------------------------------------

class LatentFusionNet(nn.Module):
    """Attention over agents, then a causal TCN over time."""

    def __init__(self, latent_dims, agent_names, d_model=32, kernel_size=3,
                 dilations=(1, 2, 4), dropout=0.1):
        super().__init__()
        self.agent_names = list(agent_names)
        self.project = nn.ModuleDict({
            name: nn.Linear(latent_dims[name], d_model) for name in agent_names
        })
        # Scores one scalar per agent per hour -> softmax over agents.
        self.attend = nn.Linear(d_model, 1)
        # The agents' calibrated risks enter as a small extra signal alongside
        # the fused latents (they are what a text-passing system would send).
        self.risk_project = nn.Linear(len(agent_names), d_model)
        self.blocks = nn.ModuleList(
            TCNBlock(d_model, kernel_size, d, dropout) for d in dilations
        )
        self.head = nn.Linear(d_model, 1)
        self.latent_dim = d_model

    def forward(self, latents, risks):
        """``latents``: {name: (B, T, dim)}; ``risks``: (B, T, A)."""
        projected = torch.stack(
            [self.project[name](latents[name]) for name in self.agent_names], dim=2
        )                                                  # (B, T, A, d)
        weights = torch.softmax(
            self.attend(torch.tanh(projected)).squeeze(-1), dim=-1
        )                                                  # (B, T, A)
        fused = (weights.unsqueeze(-1) * projected).sum(dim=2)   # (B, T, d)
        fused = fused + self.risk_project(risks)

        h = fused.transpose(1, 2)                          # (B, d, T)
        for block in self.blocks:                          # causal over time
            h = block(h)
        embedding = h.transpose(1, 2)                      # (B, T, d)
        return self.head(embedding).squeeze(-1), embedding, weights


# --- Orchestrator ----------------------------------------------------------

class Orchestrator(Agent):
    """Fuses agent latents into one calibrated hourly risk.

    Subclasses :class:`~agents.base.Agent` so it inherits exactly the same
    isotonic-on-val calibration as the agents it consumes -- the fused risk is
    therefore directly comparable to theirs.
    """

    name = "orchestrator"

    def __init__(self, bank=None, d_model=32, kernel_size=3, dilations=(1, 2, 4),
                 dropout=0.1, lr=1e-3, batch_size=64, max_epochs=8, patience=2,
                 seed=SEED, verbose=True):
        super().__init__()
        self.hparams = dict(d_model=d_model, kernel_size=kernel_size,
                            dilations=dilations, dropout=dropout)
        self.lr, self.batch_size = lr, batch_size
        self.max_epochs, self.patience = max_epochs, patience
        self.seed, self.verbose = seed, verbose
        self.latent_dim = d_model
        self.bank = bank
        self.net = None
        self.device = torch.device("cpu")     # ~25k params; CPU is plenty

    def _ensure_bank(self):
        if self.bank is None:
            self.bank = AgentBank.load()
        return self.bank

    def _sequences(self, df, with_labels=True):
        """Per-stay sequences of fused agent features."""
        bank = self._ensure_bank()
        X, keys = bank.lookup(df)
        ordered = df.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        y = (ordered[LABEL].to_numpy(dtype=float) if with_labels
             else np.zeros(len(ordered), dtype=float))

        stay = keys["stay_id"].to_numpy()
        hour = keys["hour"].to_numpy()
        boundaries = np.flatnonzero(np.diff(stay)) + 1
        sequences = []
        for lo, hi in zip(np.r_[0, boundaries], np.r_[boundaries, len(stay)]):
            sequences.append({"stay_id": stay[lo], "hours": hour[lo:hi],
                              "X": X[lo:hi], "y": y[lo:hi]})
        return sequences

    def _split_batch(self, X):
        """(B, F, T) tensor -> per-agent latents (B, T, dim) + risks (B, T, A)."""
        x = X.transpose(1, 2)                              # (B, T, F)
        return self._ensure_bank().split_features(x)

    # -- training ---------------------------------------------------------
    def _fit_model(self, train_df, val_df):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        bank = self._ensure_bank()

        train_seq = self._sequences(train_df)
        val_seq = self._sequences(val_df)
        self.net = LatentFusionNet(bank.latent_dims, bank.agent_names,
                                   **self.hparams).to(self.device)
        if self.verbose:
            n = sum(p.numel() for p in self.net.parameters())
            print(f"[{self.name}] {n:,} params fusing {bank.agent_names}; "
                  f"{len(train_seq):,} train stays")

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.seed)
        best, best_state, stale = -np.inf, None, 0
        from metrics import auprc

        for epoch in range(1, self.max_epochs + 1):
            self.net.train()
            total, seen = 0.0, 0.0
            for idx in make_batches(train_seq, self.batch_size, shuffle=True, rng=rng):
                X, y, mask = collate(train_seq, idx, self.device)
                latents, risks = self._split_batch(X)
                logits, _, _ = self.net(latents, risks)
                loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none")
                        * mask).sum() / mask.sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                n_valid = float(mask.sum())
                total += float(loss.detach()) * n_valid
                seen += n_valid

            scores, y_val = self._score_sequences(val_seq)
            val_auprc = auprc(y_val, scores)
            if self.verbose:
                print(f"  epoch {epoch}: train loss {total/seen:.4f} | "
                      f"val AUPRC {val_auprc:.4f}")
            if val_auprc > best:
                best, stale = val_auprc, 0
                best_state = {k: v.detach().clone()
                              for k, v in self.net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    if self.verbose:
                        print(f"  early stop (no val gain for {self.patience} epochs)")
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        if self.verbose:
            print(f"[{self.name}] best val AUPRC {best:.4f}")
        return self

    @torch.no_grad()
    def _forward(self, sequences, want_extras=False):
        self.net.eval()
        logits_out, emb_out, attn_out, keys = [], [], [], []
        for idx in make_batches(sequences, self.batch_size):
            X, _, mask = collate(sequences, idx, self.device)
            latents, risks = self._split_batch(X)
            logits, emb, weights = self.net(latents, risks)
            logits = logits.numpy()
            emb, weights, mask = emb.numpy(), weights.numpy(), mask.numpy()
            for row, i in enumerate(idx):
                n = int(mask[row].sum())
                logits_out.append(logits[row, :n])
                if want_extras:
                    emb_out.append(emb[row, :n])
                    attn_out.append(weights[row, :n])
                seq = sequences[i]
                keys.append(np.stack([np.full(n, seq["stay_id"]), seq["hours"]], axis=1))
        return (np.concatenate(logits_out),
                np.concatenate(emb_out) if want_extras else None,
                np.concatenate(attn_out) if want_extras else None,
                np.concatenate(keys))

    def _score_sequences(self, sequences):
        logits, _, _, _ = self._forward(sequences)
        y = np.concatenate([sequences[i]["y"]
                            for idx in make_batches(sequences, self.batch_size)
                            for i in idx])
        return 1.0 / (1.0 + np.exp(-logits)), y

    # -- emission ---------------------------------------------------------
    def _emit_raw(self, df):
        if self.net is None:
            raise RuntimeError("orchestrator is not fitted -- call fit() first")
        sequences = self._sequences(df, with_labels=False)
        logits, emb, _, keys = self._forward(sequences, want_extras=True)
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        logits, emb, keys = logits[order], emb[order], keys[order]
        raw = 1.0 / (1.0 + np.exp(-logits))
        return (pd.DataFrame(keys, columns=KEY_COLUMNS),
                emb.astype(np.float32), raw)

    def attention_weights(self, df):
        """Per-hour attention over agents, sorted by ``(stay_id, hour)``.

        Returns ``(keys, weights)`` where ``weights[i, a]`` is how much agent
        ``a`` drove the fused risk at row ``i``. This is what makes the fusion
        inspectable instead of a black box.
        """
        sequences = self._sequences(df, with_labels=False)
        _, _, weights, keys = self._forward(sequences, want_extras=True)
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        return (pd.DataFrame(keys[order], columns=KEY_COLUMNS), weights[order])
