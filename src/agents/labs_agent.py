"""Agent #3: the labs agent.

Reads the hourly lab panel (24 analytes, binned by result-availability time)
and emits a latent embedding plus a calibrated deterioration risk.

**Leakage.** Two separate guards, because labs are the easiest modality to leak:

1. *Availability, not collection.* The extract bins each result by
   ``storetime`` -- when it appeared in the system -- not ``charttime``, when
   the blood was drawn. The median gap is 70 minutes, so binning by charttime
   would let the model read results before they existed. See
   ``scripts/mimic_labs.sql``.
2. *Strictly before t.* As with vitals, values are carried forward within a
   stay and then shifted one hour, so the row for hour ``t`` sees only results
   available at hours ``< t``. Carry-forward is bounded (labs go stale), and
   the lab window opens 24h before ICU admission so ED bloods count as history.

**Why an MLP rather than a GBM.** A GBM would score at least as well on tabular
data, but the agent contract requires a *latent vector*, and gradient-boosted
trees have no penultimate layer to expose (leaf indices are high-dimensional and
not a smooth embedding). A small MLP gives a natural 32-dim representation that
matches the vitals-TCN's latent width, which is what the orchestrator will fuse.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import KEY_COLUMNS, Agent
from config import SEED

LABEL = "label_24h"
LABS_PATH = "data/cache/mimic_labs_hourly.parquet"

# The 24-analyte panel, in the order the extract produces them.
LAB_COLUMNS = [
    "lab_creatinine", "lab_bun", "lab_sodium", "lab_potassium", "lab_chloride",
    "lab_bicarbonate", "lab_anion_gap", "lab_glucose", "lab_calcium",
    "lab_magnesium", "lab_phosphate", "lab_hematocrit", "lab_hemoglobin",
    "lab_wbc", "lab_platelets", "lab_rdw", "lab_alt", "lab_ast",
    "lab_bilirubin", "lab_albumin", "lab_inr", "lab_ptt", "lab_lactate",
    "lab_ph",
]

# How long a lab result stays informative. Chemistry and haematology are drawn
# roughly daily in the ICU, so a 24h bound keeps the usual cadence while
# stopping a week-old creatinine from being read as the current one -- the same
# staleness discipline applied to vitals in mimic_cohort.py.
LAB_MAX_STALENESS_HOURS = 24

# Labs are drawn from 24h before ICU admission (ED bloods are real history).
LAB_LOOKBACK_HOURS = 24


def load_labs(path=LABS_PATH):
    return pd.read_parquet(path)


def build_lab_features(cohort, labs, max_staleness=LAB_MAX_STALENESS_HOURS):
    """Leakage-safe lab features aligned to the cohort's (stay_id, hour) rows.

    Builds a dense hourly grid per stay running from ``-LAB_LOOKBACK_HOURS`` to
    the stay's last prediction hour, forward-fills each analyte along it, shifts
    one hour so the present is excluded, drops values older than
    ``max_staleness``, and finally selects the cohort's rows.

    The grid has to extend before hour 0 because the cohort starts at hour 6 and
    a lab drawn in the ED must still be visible there.

    Returns a frame with one row per cohort row: the carried lab values plus a
    ``<lab>_hrs_since`` staleness channel for each.
    """
    cohort_keys = cohort[["stay_id", "hour"]].drop_duplicates()
    span = cohort_keys.groupby("stay_id")["hour"].max().reset_index(name="max_hour")

    # Dense grid: one row per (stay, hour) from -lookback to the stay's end.
    grid = span.assign(
        hour=[np.arange(-LAB_LOOKBACK_HOURS, int(m) + 1) for m in span["max_hour"]]
    ).explode("hour")[["stay_id", "hour"]]
    grid["hour"] = grid["hour"].astype(int)

    labs = labs[labs["hour"].between(-LAB_LOOKBACK_HOURS, int(span["max_hour"].max()))]
    merged = grid.merge(labs, on=["stay_id", "hour"], how="left")
    merged = merged.sort_values(["stay_id", "hour"]).reset_index(drop=True)

    stay, hour = merged["stay_id"], merged["hour"]
    out = merged[["stay_id", "hour"]].copy()
    for col in LAB_COLUMNS:
        observed = merged[col] if col in merged else pd.Series(np.nan, index=merged.index)
        value = observed.groupby(stay, sort=False).ffill()
        value = value.groupby(stay, sort=False).shift(1)          # strictly before t
        last = hour.where(observed.notna()).groupby(stay, sort=False).ffill()
        last = last.groupby(stay, sort=False).shift(1)
        staleness = hour - last
        out[col] = value.where(staleness <= max_staleness).to_numpy()
        out[f"{col}_hrs_since"] = staleness.where(staleness <= max_staleness).to_numpy()

    # Keep only the cohort's prediction hours, in cohort key order.
    return cohort_keys.merge(out, on=["stay_id", "hour"], how="left") \
                      .sort_values(["stay_id", "hour"]).reset_index(drop=True)


FEATURE_COLUMNS = LAB_COLUMNS + [f"{c}_hrs_since" for c in LAB_COLUMNS]
# Every analyte also contributes a "was this ever measured?" mask.
N_FEATURES = len(FEATURE_COLUMNS) + len(LAB_COLUMNS)


class LabsMLP(nn.Module):
    """Small MLP; the penultimate layer is the agent's latent vector."""

    def __init__(self, n_features, hidden=64, latent=32, dropout=0.2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, latent), nn.ReLU(),
        )
        self.head = nn.Linear(latent, 1)
        self.latent_dim = latent

    def forward(self, x):
        embedding = self.body(x)                       # (N, latent)
        return self.head(embedding).squeeze(-1), embedding


class LabsAgent(Agent):
    """Lab-panel agent: 32-dim latent + calibrated risk per patient-hour."""

    name = "labs"

    def __init__(self, hidden=64, latent=32, dropout=0.2, lr=1e-3,
                 batch_size=4096, max_epochs=12, patience=2, seed=SEED,
                 labs=None, verbose=True):
        super().__init__()
        self.hparams = dict(hidden=hidden, latent=latent, dropout=dropout)
        self.lr, self.batch_size = lr, batch_size
        self.max_epochs, self.patience = max_epochs, patience
        self.seed, self.verbose = seed, verbose
        self.latent_dim = latent
        self.labs = labs if labs is not None else None
        self.net, self.stats = None, None
        self.device = torch.device("cpu")   # tiny model; CPU is plenty

    # -- features ---------------------------------------------------------
    def _features(self, df):
        """(N, F) matrix for ``df``'s rows, sorted by (stay_id, hour)."""
        if self.labs is None:
            self.labs = load_labs()
        feats = build_lab_features(df, self.labs)
        parts = []
        for col in FEATURE_COLUMNS:
            v = feats[col].to_numpy(dtype=float)
            mean, std = self.stats[col]
            parts.append(np.nan_to_num((v - mean) / std, nan=0.0))
        for col in LAB_COLUMNS:                      # missingness masks
            parts.append(np.isnan(feats[col].to_numpy(dtype=float)).astype(float))
        return np.stack(parts, axis=1).astype(np.float32), feats[["stay_id", "hour"]]

    def _fit_normalizer(self, df):
        if self.labs is None:
            self.labs = load_labs()
        feats = build_lab_features(df, self.labs)
        self.stats = {}
        for col in FEATURE_COLUMNS:
            v = feats[col].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            mean = float(v.mean()) if v.size else 0.0
            std = float(v.std()) if v.size else 1.0
            self.stats[col] = (mean, std if std > 1e-6 else 1.0)

    # -- training ---------------------------------------------------------
    def _fit_model(self, train_df, val_df):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._fit_normalizer(train_df)                # TRAIN statistics only

        X_train, keys_train = self._features(train_df)
        y_train = self._labels_in_key_order(train_df, keys_train)
        X_val, keys_val = self._features(val_df)
        y_val = self._labels_in_key_order(val_df, keys_val)

        self.net = LabsMLP(n_features=X_train.shape[1], **self.hparams).to(self.device)
        if self.verbose:
            n = sum(p.numel() for p in self.net.parameters())
            print(f"[{self.name}] {n:,} params; {len(X_train):,} train rows, "
                  f"{X_train.shape[1]} features")

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        Xt = torch.from_numpy(X_train)
        yt = torch.from_numpy(y_train.astype(np.float32))
        rng = np.random.default_rng(self.seed)
        best, best_state, stale = -np.inf, None, 0

        from metrics import auprc                      # local import avoids a cycle
        for epoch in range(1, self.max_epochs + 1):
            self.net.train()
            order = rng.permutation(len(Xt))
            total = 0.0
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                logits, _ = self.net(Xt[idx])
                loss = F.binary_cross_entropy_with_logits(logits, yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)

            val_scores = self._scores(X_val)
            val_auprc = auprc(y_val, val_scores)
            if self.verbose:
                print(f"  epoch {epoch}: train loss {total/len(order):.4f} | "
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

    def _labels_in_key_order(self, df, keys):
        ordered = df.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        if not (ordered["stay_id"].to_numpy() == keys["stay_id"].to_numpy()).all():
            raise ValueError("lab features are misaligned with the cohort rows")
        return ordered[LABEL].to_numpy(dtype=float)

    @torch.no_grad()
    def _scores(self, X):
        self.net.eval()
        out = []
        for i in range(0, len(X), self.batch_size):
            logits, _ = self.net(torch.from_numpy(X[i:i + self.batch_size]))
            out.append(logits.numpy())
        return 1.0 / (1.0 + np.exp(-np.concatenate(out)))

    # -- emission ---------------------------------------------------------
    @torch.no_grad()
    def _emit_raw(self, df):
        if self.net is None:
            raise RuntimeError("agent is not fitted -- call fit() first")
        self.net.eval()
        X, keys = self._features(df)
        logits, embeddings = [], []
        for i in range(0, len(X), self.batch_size):
            lo, em = self.net(torch.from_numpy(X[i:i + self.batch_size]))
            logits.append(lo.numpy())
            embeddings.append(em.numpy())
        logits = np.concatenate(logits)
        raw = 1.0 / (1.0 + np.exp(-logits))
        return (keys.reset_index(drop=True),
                np.concatenate(embeddings).astype(np.float32), raw)
