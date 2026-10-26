"""Agent V4: ``ecg_meas`` -- machine-read ECG measurements, no waveforms.

The cheapest serious use of the ECG modality: the cart's own machine read
(conduction intervals + diagnostic statements, featurized in
:mod:`agents.ecg_meas_features`) through the **existing** frozen-embedding
machinery. This is a thin :class:`agents.ecg_agent.EcgAgent` subclass -- the
49-wide feature vector plays the role of the "embedding", so the presence
guard (constant risk on no-ECG hours), staleness input, covered-train
z-scoring, early stopping and isotonic-on-val calibration are all inherited,
tested behaviour.

Coverage is the full indexed set (~42% of cohort hours) because the
measurements live in BigQuery for every study -- no waveform download needed.
A study missing from the feature npz (no measurement row / fully corrupt) hits
the inherited cache-lookup miss and its hours become no-ECG, exactly like a
study that was never downloaded.

``drop_features`` powers the ``--no-rate`` ablation: heart-rate-carrying
features are redundant with the vitals agent's live HR, so the honest
standalone story needs a variant without them (see the training script).
"""

import numpy as np
import pandas as pd

from agents.ecg_agent import EcgAgent
from agents.ecg_meas_features import RATE_FEATURES  # noqa: F401  (re-export)

MEAS_CACHE = "data/cache/ecg_meas_features.npz"


class EcgMeasAgent(EcgAgent):
    """Machine-measurement features -> latent + calibrated risk per hour."""

    name = "ecg_meas"

    def __init__(self, embed_cache=MEAS_CACHE, drop_features=(), **kwargs):
        super().__init__(embed_cache=embed_cache, **kwargs)
        self.drop_features = tuple(drop_features)

    def _load_cache(self):
        if self._cache is None:
            z = np.load(self.embed_cache)
            feats, names = z["embeddings"], [str(n) for n in z["feature_names"]]
            if self.drop_features:
                keep = [i for i, n in enumerate(names) if n not in self.drop_features]
                dropped = len(names) - len(keep)
                if dropped != len(self.drop_features) and self.verbose:
                    print(f"[{self.name}] warning: asked to drop "
                          f"{len(self.drop_features)} features, matched {dropped}")
                feats = feats[:, keep]
            self._cache = (feats.astype(np.float32), z["study_ids"],
                           pd.read_parquet(self.ecg_index_path))
        return self._cache
