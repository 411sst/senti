"""The agent interface every per-modality agent implements.

The research claim this project defends is that agents should pass **latent
vectors**, not text. So the contract is deliberately narrow: given a cohort
frame, an agent emits, for each patient-hour, exactly two things --

1. ``latent``  -- a fixed-width embedding from its penultimate layer, and
2. ``risk``    -- a **calibrated** probability of deterioration.

Every agent (vitals, labs, ECG, notes) implements this same interface, so the
orchestrator can fuse their latents without knowing anything about the modality
underneath.

Two rules hold for all agents:

* **Leakage.** An agent may only read data recorded strictly before the
  prediction hour. The cohort frames are already built that way
  (:mod:`mimic_cohort`); a sequence agent must additionally be *causal* along
  time -- see ``tests/test_agents.py``, which perturbs the future and asserts
  the past does not move.
* **Calibration.** Risk is calibrated on the **val** split only -- never train
  (the model already fit it) and never test (that would be leakage). This is
  implemented once, here in :class:`Agent`, so every agent inherits identical
  isotonic-on-val calibration rather than each re-inventing it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# Cohort columns that identify a prediction point. Every emission is aligned to
# these so latents from different agents can be joined row-for-row.
KEY_COLUMNS = ["stay_id", "hour"]


@dataclass
class AgentOutput:
    """What an agent emits for a set of patient-hours.

    ``keys`` carries ``stay_id``/``hour`` in the same row order as ``latent``
    and ``risk``, so downstream fusion can align agents by key rather than by
    assuming a shared ordering.
    """

    keys: pd.DataFrame          # (N, 2) stay_id + hour
    latent: np.ndarray          # (N, D) penultimate-layer embedding
    risk: np.ndarray            # (N,)   calibrated P(deterioration)

    def __post_init__(self):
        n = len(self.keys)
        if self.latent.shape[0] != n or self.risk.shape[0] != n:
            raise ValueError(
                f"length mismatch: keys={n}, latent={self.latent.shape[0]}, "
                f"risk={self.risk.shape[0]}"
            )

    @property
    def latent_dim(self):
        return int(self.latent.shape[1])

    def to_frame(self):
        """Keys + risk as a DataFrame (latents stay as a dense array)."""
        out = self.keys.copy()
        out["risk"] = self.risk
        return out

    def save(self, path):
        """Cache to a single ``.npz`` so fusion can load it instantly.

        CLAUDE.md rule #5: encoders must never re-run unnecessarily.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            stay_id=self.keys["stay_id"].to_numpy(),
            hour=self.keys["hour"].to_numpy(),
            latent=self.latent.astype(np.float32),
            risk=self.risk.astype(np.float32),
        )
        return path

    @classmethod
    def load(cls, path):
        z = np.load(path)
        keys = pd.DataFrame({"stay_id": z["stay_id"], "hour": z["hour"]})
        return cls(keys=keys, latent=z["latent"], risk=z["risk"])


class Agent(ABC):
    """Base class for a per-modality agent.

    A subclass supplies two things -- how to fit its model, and how to produce
    raw (uncalibrated) scores plus latents:

    * :meth:`_fit_model` -- train on ``train_df`` (``val_df`` is available for
      early stopping only).
    * :meth:`_emit_raw`  -- return ``(keys, latent, raw_score)`` for a frame,
      ordered by ``(stay_id, hour)``.

    Everything else -- calibration, the public :meth:`fit` / :meth:`emit`, and
    caching -- is inherited, so labs, ECG and notes agents get the same
    calibration protocol for free and it can only be got wrong in one place.
    """

    name = "agent"
    latent_dim = 0
    label_column = "label_24h"
    warm_start = False

    def __init__(self):
        self.calibrator = None
        self.calibration_summary = None
        # Set only inside :meth:`finetune`. A subclass that supports warm
        # starts checks it in ``_fit_model`` to keep its existing ``net`` and
        # normaliser instead of building fresh ones.
        self.warm_start = False

    # -- subclass responsibilities ----------------------------------------
    @abstractmethod
    def _fit_model(self, train_df, val_df):
        """Fit the underlying model. ``val_df`` is for early stopping only."""

    @abstractmethod
    def _emit_raw(self, df):
        """Return ``(keys, latent, raw_score)`` ordered by ``(stay_id, hour)``.

        ``raw_score`` is UNcalibrated and need not be a probability -- any
        score where "higher means higher risk" is fine (the NEWS2 agent emits
        0-20). Isotonic regression is rank-based, so :meth:`_fit_calibrator`
        maps whatever scale the agent uses onto a genuine probability.
        """

    # -- inherited behaviour ----------------------------------------------
    def fit(self, train_df, val_df):
        """Fit the model on train, then calibrate on val. Returns ``self``."""
        self._fit_model(train_df, val_df)
        self._fit_calibrator(val_df)
        return self

    def finetune(self, train_df, val_df, lr=None, max_epochs=None):
        """Continue training a fitted agent on a new cohort, then recalibrate.

        Used for site adaptation (MIMIC -> eICU). The network starts from its
        current weights and the normaliser fitted on the SOURCE train split is
        kept: the weights were learned against that z-scoring, so refitting it
        on the target would move every input under a frozen first layer. The
        isotonic calibrator is refit on the target ``val_df`` as usual.

        Early stopping is seeded with the incoming weights' own val score, so
        if no epoch beats the pretrained net on target val, the pretrained net
        is what comes back.
        """
        if getattr(self, "net", None) is None:
            raise RuntimeError(f"{self.name} has no fitted net to fine-tune")
        saved = (self.lr, self.max_epochs)
        if lr is not None:
            self.lr = lr
        if max_epochs is not None:
            self.max_epochs = max_epochs
        self.warm_start = True
        try:
            return self.fit(train_df, val_df)
        finally:
            self.warm_start = False
            self.lr, self.max_epochs = saved

    def _initial_best(self, val_score):
        """``(best, best_state)`` to seed an early-stopping loop.

        From scratch there is nothing to beat. When fine-tuning, the incoming
        weights are the candidate to beat, scored by ``val_score()``.
        """
        if not self.warm_start:
            return -np.inf, None
        state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        score = val_score()
        if getattr(self, "verbose", False):
            print(f"  epoch 0 (pretrained): val AUPRC {score:.4f}")
        return score, state

    def _fit_calibrator(self, val_df):
        """Fit isotonic regression mapping raw score -> P(deterioration).

        **Val only.** Calibrating on train would reuse data the model already
        fit (flattering ECE); calibrating on test would be outright leakage.
        Isotonic is monotone, so it changes probabilities without touching any
        ranking metric -- AUPRC before and after is identical by construction.
        """
        keys, _, raw = self._emit_raw(val_df)
        y = self._aligned_labels(val_df, keys)
        self.calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw, y)
        self.calibration_summary = {
            "split_rows": int(len(y)), "positives": int(y.sum()),
        }
        return self.calibrator

    def _aligned_labels(self, df, keys):
        """Labels for ``df`` in the same row order as ``keys``."""
        ordered = df.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        if not (ordered["stay_id"].to_numpy() == keys["stay_id"].to_numpy()).all():
            raise ValueError("label alignment failed: stay_id order differs")
        return ordered[self.label_column].to_numpy(dtype=float)

    def calibrate(self, raw_score):
        """Map raw scores through the fitted calibrator."""
        if self.calibrator is None:
            raise RuntimeError(f"{self.name} is not calibrated -- call fit() first")
        return np.asarray(self.calibrator.transform(raw_score), dtype=float)

    def emit(self, df):
        """Latent vector + CALIBRATED risk for every row of ``df``."""
        keys, latent, raw = self._emit_raw(df)
        return AgentOutput(keys=keys, latent=latent,
                           risk=self.calibrate(raw).astype(np.float32))

    def raw_risk(self, df):
        """UNcalibrated probabilities, same row order as :meth:`emit`.

        Kept so agents can report raw vs calibrated ECE side by side.
        """
        return self._emit_raw(df)[2]

    def emit_and_cache(self, df, path):
        """Emit and write the result to ``path`` (see :meth:`AgentOutput.save`)."""
        out = self.emit(df)
        out.save(path)
        return out
