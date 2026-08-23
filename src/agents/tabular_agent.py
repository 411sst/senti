"""Agents #6 and #7: measurement cadence and urine/fluid balance.

Two thin instances of one generic per-hour tabular MLP agent (the labs-agent
recipe, parameterized by a feature builder):

* :class:`CadenceAgent` -- CLINICIAN BEHAVIOUR as a modality: how often labs
  are drawn, vitals are charted, orders are placed. The construct is
  RCT-validated (CONCERN, Nature Medicine 2025); Sentinel's contribution is
  the separable-agent framing with paired earliness attribution. Expected to
  be clinician-loaded: big lead on vent/vaso, little on death -- which makes
  it the built-in positive control for the clinician-vs-physiology
  decomposition.
* :class:`FluidAgent` -- urine output and fluid balance: oliguria precedes
  hypotension/vasopressors by hours, and renal trajectory speaks to the
  orchestrator's weakest per-event component (death, AUPRC 0.073).

Leakage: both builders bin events by action/charting time and shift one hour
(strictly-before-t; see modality_features.py); counts fill zero because
observed silence is real information, unlike lab values.
"""

import numpy as np
import torch

from agents.base import KEY_COLUMNS, Agent
from agents.labs_agent import LabsMLP
from config import SEED
from metrics import auprc
from modality_features import (  # noqa: F401  (re-exported public names)
    CADENCE_FEATURES,
    CADENCE_PATH,
    FLUID_FEATURES,
    FLUIDS_PATH,
    RESP_FEATURES,
    RESP_PATH,
    build_cadence_features,
    build_fluid_features,
    build_resp_features,
    load_cadence,
    load_fluids,
    load_resp,
)

LABEL = "label_24h"


class TabularMLPAgent(Agent):
    """Per-hour tabular MLP agent: 32-dim latent + calibrated risk.

    Subclasses set ``name``/``FEATURES`` and implement ``_feature_frame``.
    Same training mechanics as the labs agent; no NaN masks because every
    feature here is a count/window statistic that is always defined.
    """

    FEATURES = []

    def __init__(self, hidden=64, latent=32, dropout=0.2, lr=1e-3,
                 batch_size=4096, max_epochs=12, patience=2, seed=SEED,
                 source=None, verbose=True, device=None):
        super().__init__()
        self.hparams = dict(hidden=hidden, latent=latent, dropout=dropout)
        self.lr, self.batch_size = lr, batch_size
        self.max_epochs, self.patience = max_epochs, patience
        self.seed, self.verbose = seed, verbose
        self.latent_dim = latent
        self.source = source          # the raw hourly extract (lazy-loaded)
        self.net, self.stats = None, None
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # -- subclass hooks ----------------------------------------------------
    def _load_source(self):
        raise NotImplementedError

    def _feature_frame(self, df):
        raise NotImplementedError

    # -- features ----------------------------------------------------------
    def _features(self, df):
        feats = self._feature_frame(df)
        parts = []
        for col in self.FEATURES:
            v = feats[col].to_numpy(dtype=float)
            mean, std = self.stats[col]
            parts.append(np.nan_to_num((v - mean) / std, nan=0.0))
        return (np.stack(parts, axis=1).astype(np.float32),
                feats[["stay_id", "hour"]])

    def _fit_normalizer(self, df):
        feats = self._feature_frame(df)
        self.stats = {}
        for col in self.FEATURES:
            v = feats[col].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            mean = float(v.mean()) if v.size else 0.0
            std = float(v.std()) if v.size else 1.0
            self.stats[col] = (mean, std if std > 1e-2 else 1.0)

    # -- training (labs-agent mechanics) -----------------------------------
    def _fit_model(self, train_df, val_df):
        import torch.nn.functional as F

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._fit_normalizer(train_df)               # TRAIN statistics only

        X_train, keys_train = self._features(train_df)
        y_train = self._labels_in_key_order(train_df, keys_train)
        X_val, keys_val = self._features(val_df)
        y_val = self._labels_in_key_order(val_df, keys_val)

        self.net = LabsMLP(n_features=X_train.shape[1],
                           **self.hparams).to(self.device)
        if self.verbose:
            n = sum(p.numel() for p in self.net.parameters())
            print(f"[{self.name}] {n:,} params; {len(X_train):,} train rows, "
                  f"{X_train.shape[1]} features")

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        Xt = torch.from_numpy(X_train).to(self.device)
        yt = torch.from_numpy(y_train.astype(np.float32)).to(self.device)
        rng = np.random.default_rng(self.seed)
        best, best_state, stale = -np.inf, None, 0
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
            val_auprc = auprc(y_val, self._scores(X_val))
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
                        print(f"  early stop (no val gain for "
                              f"{self.patience} epochs)")
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        if self.verbose:
            print(f"[{self.name}] best val AUPRC {best:.4f}")
        return self

    def _labels_in_key_order(self, df, keys):
        ordered = df.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        if not (ordered["stay_id"].to_numpy()
                == keys["stay_id"].to_numpy()).all():
            raise ValueError(f"{self.name} features misaligned with cohort")
        return ordered[LABEL].to_numpy(dtype=float)

    @torch.no_grad()
    def _scores(self, X):
        self.net.eval()
        out = []
        for i in range(0, len(X), self.batch_size):
            xb = torch.from_numpy(X[i:i + self.batch_size]).to(self.device)
            logits, _ = self.net(xb)
            out.append(logits.cpu().numpy())
        return 1.0 / (1.0 + np.exp(-np.concatenate(out)))

    @torch.no_grad()
    def _emit_raw(self, df):
        if self.net is None:
            raise RuntimeError("agent is not fitted -- call fit() first")
        self.net.eval()
        X, keys = self._features(df)
        logits, embeddings = [], []
        for i in range(0, len(X), self.batch_size):
            xb = torch.from_numpy(X[i:i + self.batch_size]).to(self.device)
            lo, em = self.net(xb)
            logits.append(lo.cpu().numpy())
            embeddings.append(em.cpu().numpy())
        logits = np.concatenate(logits)
        raw = 1.0 / (1.0 + np.exp(-logits))
        return (keys.reset_index(drop=True),
                np.concatenate(embeddings).astype(np.float32), raw)


class CadenceAgent(TabularMLPAgent):
    name = "cadence"
    FEATURES = CADENCE_FEATURES

    def _feature_frame(self, df):
        if self.source is None:
            self.source = load_cadence()
        return build_cadence_features(df, self.source)


class FluidAgent(TabularMLPAgent):
    name = "fluid"
    FEATURES = FLUID_FEATURES

    def _feature_frame(self, df):
        if self.source is None:
            self.source = load_fluids()
        return build_fluid_features(df, self.source)


class RespAgent(TabularMLPAgent):
    """Respiratory-support trajectory. NEAR-LABEL: ships with the
    ``max_level=4`` cap-at-HFNC sensitivity variant (drops the NIV rung one
    step from the vent label) and is reported per event component."""

    name = "resp"
    FEATURES = RESP_FEATURES

    def __init__(self, max_level=5, **kwargs):
        super().__init__(**kwargs)
        self.max_level = max_level
        if max_level < 5:
            self.name = f"resp_cap{max_level}"

    def _feature_frame(self, df):
        if self.source is None:
            self.source = load_resp()
        return build_resp_features(df, self.source,
                                   max_level=self.max_level)
