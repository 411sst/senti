"""Agent #5: the notes agent (radiology reports).

A small head over a **frozen** Bio_ClinicalBERT embedding of the most recent
radiology report available before the prediction hour.

**Why radiology only.** MIMIC-IV-Note contains exactly two note types:
discharge summaries and radiology reports. There are no nursing or physician
progress notes (MIMIC-III had them; MIMIC-IV does not). Discharge summaries are
written a median of **57 hours after ICU discharge** and narrate the outcome
being predicted, so they are excluded outright -- the discharge table is never
read anywhere in this pipeline. What remains is radiology.

**Availability, not acquisition.** Notes are keyed by
``GREATEST(storetime, charttime)`` -- when the report entered the record, not
when the image was taken. The median gap is 136 minutes (p75 303), so keying on
charttime would hand the model a radiologist's impression two hours before
anyone could have read it.

**Presence-leakage.** Hours with a recent report are event-enriched (5.09% vs
4.18%) because imaging is ordered on deteriorating patients. As with the ECG
agent, every no-note hour receives the *same* learned vector and no other
varying input, so the head is mathematically forced to emit a single constant
there -- making presence-separation auditable rather than hidden, and testable
via :func:`presence_only_auprc`.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import KEY_COLUMNS, Agent
from agents.note_encoder import EMBED_DIM
from config import SEED
from metrics import auprc
# The assembler lives in the torch-free modality_features module so the fair
# GBM baselines can use it next to XGBoost (torch and XGBoost cannot share a
# process on macOS -- see tests/conftest.py). Re-exported here unchanged, under
# this module's historical names.
from modality_features import (  # noqa: F401  (re-exported public names)
    NOTE_EMBED_CACHE as EMBED_CACHE,
    NOTE_INDEX_PATH as NOTE_INDEX,
    NOTE_STALENESS_HOURS as STALENESS_HOURS,
    assemble_note_features,
)

LABEL = "label_24h"


def presence_only_auprc(y, has_note):
    """AUPRC from the note-presence indicator alone -- the floor to beat."""
    return auprc(y, has_note)


class NoteHead(nn.Module):
    """Small head over a frozen 768-dim note embedding.

    ``no_note_embedding`` is a learned vector meaning "no report available".
    Substituted wholesale on uncovered hours, where staleness is also zeroed, so
    every such row presents an identical input and receives one constant risk.
    """

    def __init__(self, embed_dim=EMBED_DIM, hidden=128, latent=32, dropout=0.2):
        super().__init__()
        self.no_note_embedding = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.no_note_embedding, std=0.02)
        self.body = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, latent), nn.ReLU(),
        )
        self.head = nn.Linear(latent, 1)
        self.latent_dim = latent

    def forward(self, embedding, has_note, age):
        mask = has_note.unsqueeze(-1)
        x = mask * embedding + (1.0 - mask) * self.no_note_embedding
        x = torch.cat([x, (age * has_note).unsqueeze(-1)], dim=-1)
        latent = self.body(x)
        return self.head(latent).squeeze(-1), latent


class NotesAgent(Agent):
    """Frozen Bio_ClinicalBERT note embedding -> latent + calibrated risk."""

    name = "notes"

    def __init__(self, hidden=128, latent=32, dropout=0.2, lr=1e-3,
                 batch_size=4096, max_epochs=15, patience=3, seed=SEED,
                 embed_cache=EMBED_CACHE, note_index=NOTE_INDEX,
                 staleness=STALENESS_HOURS, verbose=True):
        super().__init__()
        self.hparams = dict(hidden=hidden, latent=latent, dropout=dropout)
        self.lr, self.batch_size = lr, batch_size
        self.max_epochs, self.patience = max_epochs, patience
        self.seed, self.verbose = seed, verbose
        self.latent_dim = latent
        self.staleness = staleness
        self.embed_cache, self.note_index_path = embed_cache, note_index
        self.net, self.stats, self._cache = None, None, None
        self.device = torch.device("cpu")

    def _load_cache(self):
        if self._cache is None:
            z = np.load(self.embed_cache, allow_pickle=True)
            self._cache = (z["embeddings"], z["note_ids"],
                           pd.read_parquet(self.note_index_path))
        return self._cache

    def _features(self, df):
        emb, ids, index = self._load_cache()
        X, has, age, keys = assemble_note_features(
            df, index, emb, ids, staleness=self.staleness)
        if self.stats is not None:
            mean, std = self.stats
            X = np.where(has[:, None] > 0, (X - mean) / std, 0.0).astype(np.float32)
        return X, has, age, keys

    def _labels(self, df, keys):
        ordered = df.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        if not (ordered["stay_id"].to_numpy() == keys["stay_id"].to_numpy()).all():
            raise ValueError("note features are misaligned with the cohort rows")
        return ordered[LABEL].to_numpy(dtype=float)

    def _fit_model(self, train_df, val_df):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        emb, ids, index = self._load_cache()
        X0, has0, _, _ = assemble_note_features(train_df, index, emb, ids,
                                                staleness=self.staleness)
        covered = X0[has0 > 0]
        mean = covered.mean(axis=0) if len(covered) else np.zeros(EMBED_DIM)
        std = covered.std(axis=0) if len(covered) else np.ones(EMBED_DIM)
        self.stats = (mean.astype(np.float32),
                      np.where(std > 1e-6, std, 1.0).astype(np.float32))

        Xtr, htr, atr, ktr = self._features(train_df)
        ytr = self._labels(train_df, ktr)
        Xva, hva, ava, kva = self._features(val_df)
        yva = self._labels(val_df, kva)

        self.net = NoteHead(**self.hparams).to(self.device)
        if self.verbose:
            n = sum(p.numel() for p in self.net.parameters())
            print(f"[{self.name}] {n:,} head params (encoder frozen); "
                  f"{100*htr.mean():.1f}% of train hours have a note "
                  f"within {self.staleness}h")

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        Xt, ht, at = (torch.from_numpy(v) for v in (Xtr, htr, atr))
        yt = torch.from_numpy(ytr.astype(np.float32))
        rng = np.random.default_rng(self.seed)
        best, best_state, stale = -np.inf, None, 0

        for epoch in range(1, self.max_epochs + 1):
            self.net.train()
            order = rng.permutation(len(Xt))
            total = 0.0
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                logits, _ = self.net(Xt[idx], ht[idx], at[idx])
                loss = F.binary_cross_entropy_with_logits(logits, yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)

            scores = self._scores(Xva, hva, ava)
            # Select on the NOTE-AVAILABLE subset: that is the honest verdict
            # and it cannot be gamed by learning who gets imaging ordered.
            cov = hva > 0
            sel = auprc(yva[cov], scores[cov]) if yva[cov].sum() > 0 else auprc(yva, scores)
            if self.verbose:
                print(f"  epoch {epoch}: loss {total/len(order):.4f} | "
                      f"val AUPRC(note-available) {sel:.4f}")
            if sel > best:
                best, stale, best_state = sel, 0, {
                    k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    if self.verbose:
                        print(f"  early stop (no gain for {self.patience} epochs)")
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        if self.verbose:
            print(f"[{self.name}] best val AUPRC (note-available) {best:.4f}")
        return self

    @torch.no_grad()
    def _scores(self, X, has, age):
        self.net.eval()
        out = []
        for i in range(0, len(X), self.batch_size):
            sl = slice(i, i + self.batch_size)
            logits, _ = self.net(torch.from_numpy(X[sl]), torch.from_numpy(has[sl]),
                                 torch.from_numpy(age[sl]))
            out.append(logits.numpy())
        return 1.0 / (1.0 + np.exp(-np.concatenate(out)))

    @torch.no_grad()
    def _emit_raw(self, df):
        if self.net is None:
            raise RuntimeError("agent is not fitted -- call fit() first")
        self.net.eval()
        X, has, age, keys = self._features(df)
        logits, latents = [], []
        for i in range(0, len(X), self.batch_size):
            sl = slice(i, i + self.batch_size)
            lo, lat = self.net(torch.from_numpy(X[sl]), torch.from_numpy(has[sl]),
                               torch.from_numpy(age[sl]))
            logits.append(lo.numpy())
            latents.append(lat.numpy())
        raw = 1.0 / (1.0 + np.exp(-np.concatenate(logits)))
        return keys, np.concatenate(latents).astype(np.float32), raw

    def has_note(self, df):
        """Boolean mask: is a note available within the staleness bound?"""
        return self._features(df)[1] > 0
