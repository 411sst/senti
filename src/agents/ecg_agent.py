"""Agent #4: the ECG agent.

A small head over a **frozen** ECG-FM embedding. The encoder is never trained;
its 768-dim outputs are extracted once and cached (see :mod:`agents.ecg_encoder`),
so retraining the head costs nothing.

**Sparse by nature.** Only ~42% of cohort patient-hours have an ECG recorded
within the 72h staleness bound. That is not a defect to be imputed away -- it is
the clinical reality that ECGs are ordered, not monitored -- so missingness is a
first-class case here.

**The presence-leakage trap.** Hours *with* a recent ECG are mildly
event-enriched (5.17% vs 4.74% overall) because sicker patients get ECGs
ordered. A model that simply learned "ECG present -> higher risk" would post a
respectable full-cohort AUPRC while knowing nothing about cardiac physiology.
Two structural choices make that shortcut visible rather than hidden:

1. Every no-ECG hour receives the *same* learned vector and no other input, so
   the head is mathematically incapable of ranking within the no-ECG group -- it
   emits one constant there. Any full-cohort AUPRC above chance therefore
   decomposes cleanly into "presence separation" plus "within-ECG signal".
2. ``presence_only_auprc`` computes what a presence indicator ALONE would score.
   The agent must beat that, and the ECG-available-subset AUPRC is the honest
   verdict on the modality.

Both are asserted in ``tests/test_ecg_agent.py``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import KEY_COLUMNS, Agent
from agents.ecg_encoder import EMBED_DIM
from config import SEED
from metrics import auprc

LABEL = "label_24h"
EMBED_CACHE = "data/cache/ecg_embeddings.npz"
ECG_INDEX = "data/cache/ecg_index.parquet"
STALENESS_HOURS = 72


# --- Leakage-safe assembly -------------------------------------------------

def assemble_ecg_features(cohort, ecg_index, embeddings, study_ids,
                          staleness=STALENESS_HOURS):
    """Per patient-hour: the most recent ECG embedding from strictly before t.

    ``merge_asof(direction="backward", allow_exact_matches=False)`` is the whole
    leakage guarantee -- an ECG recorded *at* hour ``t`` is excluded, only
    earlier ones qualify. Anything older than ``staleness`` hours is dropped, so
    a month-old ECG is treated as no ECG rather than as current cardiac state.

    Returns ``(embeddings (N, 768), has_ecg (N,), age_hours (N,), keys)`` with
    rows in ``(stay_id, hour)`` order. Rows without a usable ECG get a zero
    embedding and ``has_ecg == 0``; the model substitutes its learned vector.
    """
    lookup = {int(s): i for i, s in enumerate(study_ids)}

    left = cohort[["stay_id", "hour"]].copy()
    left["stay_id"] = left.stay_id.astype("int64")
    left["hour"] = left.hour.astype("float64")
    right = ecg_index[["stay_id", "ecg_hour", "study_id"]].copy()
    right["stay_id"] = right.stay_id.astype("int64")
    right["ecg_hour"] = right.ecg_hour.astype("float64")

    merged = pd.merge_asof(
        left.sort_values("hour").reset_index(drop=True),
        right.sort_values("ecg_hour").reset_index(drop=True),
        left_on="hour", right_on="ecg_hour", by="stay_id",
        direction="backward", allow_exact_matches=False,
    ).sort_values(["stay_id", "hour"]).reset_index(drop=True)

    age = (merged["hour"] - merged["ecg_hour"]).to_numpy(dtype=float)
    usable = merged["study_id"].notna().to_numpy() & (age <= staleness)

    out = np.zeros((len(merged), embeddings.shape[1]), dtype=np.float32)
    has = np.zeros(len(merged), dtype=np.float32)
    for i in np.flatnonzero(usable):
        idx = lookup.get(int(merged["study_id"].iloc[i]))
        if idx is None:          # study not downloaded / unreadable
            continue
        out[i] = embeddings[idx]
        has[i] = 1.0
    age = np.where(has > 0, age, 0.0)
    keys = merged[["stay_id", "hour"]].astype("int64")
    return out, has, age.astype(np.float32), keys


def presence_only_auprc(y, has_ecg):
    """AUPRC obtainable from the ECG presence indicator alone.

    The floor the agent must clear. If the agent's full-cohort AUPRC merely
    matches this, it has learned who gets an ECG ordered, not what the ECG says.
    """
    return auprc(y, has_ecg)


# --- Model -----------------------------------------------------------------

class EcgHead(nn.Module):
    """Small head over a frozen 768-dim embedding; penultimate layer is the latent.

    ``no_ecg_embedding`` is a learned vector standing for "no ECG information
    available". It is substituted wholesale for the (zeroed) embedding on
    uncovered hours, and since those hours supply no other varying input, the
    head necessarily returns a single constant risk for all of them.
    """

    def __init__(self, embed_dim=EMBED_DIM, hidden=128, latent=32, dropout=0.2):
        super().__init__()
        self.no_ecg_embedding = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.no_ecg_embedding, std=0.02)
        self.body = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, latent), nn.ReLU(),
        )
        self.head = nn.Linear(latent, 1)
        self.latent_dim = latent

    def forward(self, embedding, has_ecg, age):
        # Where no ECG is available, replace the embedding entirely with the
        # learned "no ECG" vector -- not a zero vector, which the head would
        # otherwise have to interpret as a real (degenerate) ECG.
        mask = has_ecg.unsqueeze(-1)
        x = mask * embedding + (1.0 - mask) * self.no_ecg_embedding
        # Staleness is only meaningful when an ECG exists; zeroed otherwise, so
        # every no-ECG row presents an identical input vector.
        x = torch.cat([x, (age * has_ecg).unsqueeze(-1)], dim=-1)
        latent = self.body(x)
        return self.head(latent).squeeze(-1), latent


class EcgAgent(Agent):
    """Frozen ECG-FM embedding -> latent vector + calibrated risk."""

    name = "ecg"

    def __init__(self, hidden=128, latent=32, dropout=0.2, lr=1e-3,
                 batch_size=4096, max_epochs=15, patience=3, seed=SEED,
                 embed_cache=EMBED_CACHE, ecg_index=ECG_INDEX,
                 staleness=STALENESS_HOURS, verbose=True):
        super().__init__()
        self.hparams = dict(hidden=hidden, latent=latent, dropout=dropout)
        self.lr, self.batch_size = lr, batch_size
        self.max_epochs, self.patience = max_epochs, patience
        self.seed, self.verbose = seed, verbose
        self.latent_dim = latent
        self.staleness = staleness
        self.embed_cache, self.ecg_index_path = embed_cache, ecg_index
        self.net, self.stats = None, None
        self._cache = None
        self.device = torch.device("cpu")

    # -- data -------------------------------------------------------------
    def _load_cache(self):
        if self._cache is None:
            z = np.load(self.embed_cache)
            self._cache = (z["embeddings"], z["study_ids"],
                           pd.read_parquet(self.ecg_index_path))
        return self._cache

    def _features(self, df):
        emb, study_ids, index = self._load_cache()
        X, has, age, keys = assemble_ecg_features(
            df, index, emb, study_ids, staleness=self.staleness)
        if self.stats is not None:
            mean, std = self.stats
            X = np.where(has[:, None] > 0, (X - mean) / std, 0.0).astype(np.float32)
        return X, has, age, keys

    def _labels(self, df, keys):
        ordered = df.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        if not (ordered["stay_id"].to_numpy() == keys["stay_id"].to_numpy()).all():
            raise ValueError("ECG features are misaligned with the cohort rows")
        return ordered[LABEL].to_numpy(dtype=float)

    # -- training ---------------------------------------------------------
    def _fit_model(self, train_df, val_df):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Normalise using ONLY hours that actually have an ECG, on train only.
        emb, study_ids, index = self._load_cache()
        X0, has0, _, _ = assemble_ecg_features(train_df, index, emb, study_ids,
                                               staleness=self.staleness)
        covered = X0[has0 > 0]
        dim = emb.shape[1]                     # width of whatever cache is loaded
        mean = covered.mean(axis=0) if len(covered) else np.zeros(dim)
        std = covered.std(axis=0) if len(covered) else np.ones(dim)
        self.stats = (mean.astype(np.float32), np.where(std > 1e-6, std, 1.0).astype(np.float32))

        Xtr, htr, atr, ktr = self._features(train_df)
        ytr = self._labels(train_df, ktr)
        Xva, hva, ava, kva = self._features(val_df)
        yva = self._labels(val_df, kva)

        self.net = EcgHead(embed_dim=dim, **self.hparams).to(self.device)
        if self.verbose:
            n = sum(p.numel() for p in self.net.parameters())
            print(f"[{self.name}] {n:,} head params (encoder frozen); "
                  f"{100*htr.mean():.1f}% of train hours have an ECG "
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
            # Select on the ECG-AVAILABLE subset: that is what the modality is
            # actually being judged on, and it cannot be gamed by presence.
            cov = hva > 0
            sel = auprc(yva[cov], scores[cov]) if yva[cov].sum() > 0 else auprc(yva, scores)
            if self.verbose:
                print(f"  epoch {epoch}: loss {total/len(order):.4f} | "
                      f"val AUPRC(ECG-available) {sel:.4f}")
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
            print(f"[{self.name}] best val AUPRC (ECG-available) {best:.4f}")
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

    # -- emission ---------------------------------------------------------
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

    def has_ecg(self, df):
        """Boolean mask: does this row have an ECG within the staleness bound?"""
        return self._features(df)[1] > 0
