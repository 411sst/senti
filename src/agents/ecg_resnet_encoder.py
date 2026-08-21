"""Trainable 1D-ResNet ECG encoder (ported from the teammate's stonecap model).

Where :mod:`agents.ecg_encoder` wraps a **frozen** foundation model (ECG-FM),
this module is the *task-trained* alternative: a from-scratch 1D-ResNet
(``ECG_Morphology_Encoder``) that is optimised end-to-end on the real
``label_24h`` cohort. It is the encoder behind the ``ecg_resnet`` (V2) and
``ecg_resnet_film`` (V3) agents.

Three things are reused verbatim from the ECG-FM path so the two encoders are a
fair swap and can be scored on identical rows:

* ``read_ecg`` (:mod:`agents.ecg_encoder`) -- the WFDB -> ``(12, 5000)`` loader.
* the leakage-safe "most recent ECG strictly before hour t, within 72h" merge --
  factored here as :func:`assemble_ecg_study_ids` and shared by V1/V2/V3.
* the presence-guard contract (a learned ``no_ecg`` vector; one constant risk on
  every no-ECG hour) -- implemented in :mod:`agents.ecg_resnet_agent`.

The port differs from stonecap in exactly the ways the integration requires:
``in_channels=12`` (12-lead MIMIC-IV-ECG, not synthetic single-lead), per-lead
train-split standardisation (a trainable net *wants* z-scored input, unlike
ECG-FM which needs raw millivolts), and the multimodal FiLM/vitals/aux heads are
dropped -- one modality per agent is the whole thesis.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from agents.ecg_encoder import EXPECTED_SAMPLES, N_LEADS, read_ecg

STALENESS_HOURS = 72
EMBED_DIM = 256                 # ECG_Morphology_Encoder projection_head width
WAVEFORM_DIR = "data/ecg_waveforms"
ECG_INDEX = "data/cache/ecg_index.parquet"


# --- Leakage-safe assembly (shared with V1/V2/V3) --------------------------

def assemble_ecg_study_ids(cohort, ecg_index, staleness=STALENESS_HOURS):
    """Per patient-hour: the study_id of the most recent ECG from strictly before t.

    Identical merge logic to :func:`agents.ecg_agent.assemble_ecg_features` --
    ``merge_asof(direction="backward", allow_exact_matches=False)`` plus the
    staleness bound -- but it returns the *study id* rather than a pre-extracted
    embedding, so a trainable encoder can fetch the waveform itself. Factoring it
    here means V1 (embeddings) and V2/V3 (waveforms) provably agree on which rows
    are ECG-covered.

    Returns ``(study_id, ecg_age, keys)`` in ``(stay_id, hour)`` order, where
    ``study_id`` is ``NaN`` for hours with no usable prior ECG and ``ecg_age`` is
    hours since that ECG (0.0 where there is none).
    """
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
    study = merged["study_id"].to_numpy(dtype=float)          # NaN where no ECG
    stale = ~(age <= staleness)
    study = np.where(stale, np.nan, study)
    age = np.where(np.isnan(study), 0.0, age)
    keys = merged[["stay_id", "hour"]].astype("int64")
    return study, age.astype(np.float32), keys


# --- Waveform store --------------------------------------------------------

class WaveformStore:
    """Serves standardised ``(12, 5000)`` waveforms for the studies on disk.

    Only a subset of the index's studies are downloaded at any time (the fetch
    is staged), so this is the single source of truth for "which studies does V2
    actually cover?" -- a row whose most-recent ECG study is absent here is
    treated as no-ECG, exactly as V1 treats a missing embedding.

    Waveforms are preloaded into RAM as float16 when they fit (``~53 KB``/study
    -> ~0.9 GB for the 5k-stay subset, ~5.3 GB for the full covered set); above
    ``max_ram_gb`` it falls back to reading each record on demand. Per-lead
    standardisation stats are fit on the TRAIN covered studies only.
    """

    def __init__(self, waveform_dir=WAVEFORM_DIR, max_ram_gb=12.0, verbose=True):
        self.dir = Path(waveform_dir)
        self.max_ram_gb = max_ram_gb
        self.verbose = verbose
        self.index = {}             # study_id -> row in self._data (or in on-disk set)
        self._data = None           # (S, 12, 5000) float16, or None if lazy
        self._lazy_paths = {}       # study_id -> record path stem (lazy mode)
        self._cache = {}            # lazy read cache
        self.mean = np.zeros((N_LEADS, 1), np.float32)
        self.std = np.ones((N_LEADS, 1), np.float32)

    @classmethod
    def from_arrays(cls, study_arrays, **kw):
        """Build a store directly from ``{study_id: (12, 5000) array}``.

        For unit tests and smoke runs that need coverage without downloading
        waveforms; the standardiser stays at mean 0 / std 1 unless fit.
        """
        store = cls(verbose=False, **kw)
        ids = sorted(int(s) for s in study_arrays)
        data = np.zeros((len(ids), N_LEADS, EXPECTED_SAMPLES), np.float16)
        for i, s in enumerate(ids):
            data[i] = np.asarray(study_arrays[s], np.float16)
            store.index[s] = i
        store._data = data
        return store

    def available_studies(self):
        """Set of study_ids that have both .hea and .dat on disk."""
        heas = {p.stem for p in self.dir.glob("*.hea")}
        dats = {p.stem for p in self.dir.glob("*.dat")}
        return {int(s) for s in (heas & dats) if s.isdigit()}

    def load(self, study_ids):
        """Preload (or index for lazy read) the requested studies."""
        want = sorted({int(s) for s in study_ids} & self.available_studies())
        est_gb = len(want) * N_LEADS * EXPECTED_SAMPLES * 2 / 1e9
        lazy = est_gb > self.max_ram_gb
        if self.verbose:
            mode = "lazy on-disk" if lazy else "preload"
            print(f"[wave] {len(want):,} studies on disk ({est_gb:.1f} GB float16) -> {mode}")

        self.index = {}
        if lazy:
            self._data = None
            self._lazy_paths = {s: str(self.dir / str(s)) for s in want}
            self.index = {s: s for s in want}
            return self

        data = np.zeros((len(want), N_LEADS, EXPECTED_SAMPLES), dtype=np.float16)
        keep = []
        for s in want:
            sig = read_ecg(self.dir / str(s))        # (12, 5000) or None
            if sig is None:
                continue
            data[len(keep)] = sig.astype(np.float16)
            self.index[s] = len(keep)
            keep.append(s)
        self._data = data[:len(keep)]
        if self.verbose and len(keep) != len(want):
            print(f"[wave] {len(want) - len(keep):,} records unreadable, dropped")
        return self

    def has(self, study_id):
        return int(study_id) in self.index

    def _raw(self, study_id):
        """Unstandardised (12, 5000) float32 for one study."""
        s = int(study_id)
        if self._data is not None:
            return self._data[self.index[s]].astype(np.float32)
        if s not in self._cache:
            sig = read_ecg(self._lazy_paths[s])
            self._cache[s] = (np.zeros((N_LEADS, EXPECTED_SAMPLES), np.float32)
                              if sig is None else sig)
        return self._cache[s]

    def fit_standardizer(self, study_ids):
        """Per-lead mean/std over the given (train-covered) studies.

        Accumulates per-lead sum and sum-of-squares in float64 (one pass over the
        preloaded studies), so no large stack is ever materialised.
        """
        present = [int(s) for s in {int(x) for x in study_ids} if self.has(s)]
        if not present:
            return self
        ssum = np.zeros(N_LEADS, np.float64)
        ssq = np.zeros(N_LEADS, np.float64)
        count = 0
        for s in present:
            x = self._raw(s).astype(np.float64)       # (12, 5000)
            ssum += x.sum(axis=1)
            ssq += (x * x).sum(axis=1)
            count += x.shape[1]
        mean = ssum / count
        std = np.sqrt(np.maximum(ssq / count - mean**2, 0.0))
        self.mean = mean.astype(np.float32).reshape(N_LEADS, 1)
        self.std = np.where(std > 1e-6, std, 1.0).astype(np.float32).reshape(N_LEADS, 1)
        return self

    def batch(self, study_ids, device):
        """Standardised ``(B, 12, 5000)`` float32 tensor for the given studies."""
        arr = np.empty((len(study_ids), N_LEADS, EXPECTED_SAMPLES), dtype=np.float32)
        for i, s in enumerate(study_ids):
            arr[i] = (self._raw(s) - self.mean) / self.std
        return torch.from_numpy(arr).to(device)


# --- The ResNet (ported: ECG_Morphology_Encoder + ResBlock1D) --------------

class ResBlock1D(nn.Module):
    """Standard 1D residual block: (Conv-BN-GELU)-(Conv-BN) + projected shortcut."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.gelu(out + identity)


class EcgMorphologyEncoder(nn.Module):
    """1D-ResNet waveform -> ``final_embedding_size`` vector.

    Ported from stonecap's ``ECG_Morphology_Encoder`` with ``in_channels``
    generalised to 12 for MIMIC-IV-ECG. ``AdaptiveAvgPool1d(1)`` makes it
    length-agnostic, so the 5000-sample (10 s @ 500 Hz) input works unchanged.
    """

    def __init__(self, in_channels=N_LEADS, final_embedding_size=EMBED_DIM):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=3, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.projection_head = nn.Linear(512, final_embedding_size)

    @staticmethod
    def _make_layer(in_c, out_c, stride):
        return nn.Sequential(ResBlock1D(in_c, out_c, stride), ResBlock1D(out_c, out_c, 1))

    def forward(self, x):                              # (B, 12, 5000)
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = torch.flatten(self.adaptive_pool(x), 1)    # (B, 512)
        return self.projection_head(x)                 # (B, 256)


# --- FiLM(age) conditioning (V3 only) --------------------------------------

class AgeMetadataEncoder(nn.Module):
    """Patient age -> 128-d context vector (stonecap's MetadataEncoder, age-only).

    The condition branch of the original is dropped: V3's hypothesis is narrowly
    "does age improve *waveform* interpretation?", and per-patient comorbidity
    baselines belong to vitals/labs/orchestrator, not the ECG no-ECG floor.
    """

    def __init__(self, embed_dim=128):
        super().__init__()
        self.age_encoder = nn.Sequential(nn.Linear(1, 32), nn.GELU())
        self.context_combiner = nn.Sequential(nn.Linear(32, embed_dim), nn.GELU())

    def forward(self, age):                            # age: (B, 1) in years
        return self.context_combiner(self.age_encoder(age / 100.0))


class FiLMGenerator(nn.Module):
    """context -> (gamma, beta), initialised to the identity (gamma=1, beta=0).

    stonecap initialised ``gamma_head.weight`` to ones (so FiLM did *not* start
    at identity); here it starts as identity -- gamma bias 1, all else 0 -- so V3
    is provably equal to V2 at initialisation and only diverges as age-modulation
    is actually learned.
    """

    def __init__(self, context_dim=128, target_feature_dim=EMBED_DIM):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(context_dim, 256), nn.GELU())
        self.gamma_head = nn.Linear(256, target_feature_dim)
        self.beta_head = nn.Linear(256, target_feature_dim)
        nn.init.zeros_(self.gamma_head.weight); nn.init.ones_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight); nn.init.zeros_(self.beta_head.bias)

    def forward(self, context):
        hidden = self.network(context)
        return self.gamma_head(hidden), self.beta_head(hidden)
