"""Agents V2/V3: a task-trained 1D-ResNet ECG agent (+ optional FiLM-on-age).

These are the trainable-encoder counterparts to :class:`agents.ecg_agent.EcgAgent`
(V1, frozen ECG-FM). They obey the identical agent contract -- a latent per hour
plus an isotonic-on-val calibrated risk -- and the identical presence guard: a
learned ``no_ecg_embedding`` is substituted on every hour without a usable prior
ECG, so those hours emit **one constant risk** and any full-cohort AUPRC above
that decomposes into presence-separation plus genuine waveform signal.

Two variants share one implementation:

* ``ecg_resnet`` (V2)      -- the ResNet trained end-to-end on ``label_24h``.
* ``ecg_resnet_film`` (V3) -- V2 plus FiLM conditioning on patient **age**,
  applied to the ECG-present pathway *before* the no-ECG substitution so the
  single-constant guarantee survives (age never leaks into the no-ECG floor).

**Why a bespoke training loop.** Unlike V1 (a tiny head over cached embeddings),
here the encoder is trained, so each step must run raw waveforms through the
ResNet. Because one ECG is the "most recent" for up to 72 hourly rows, we encode
each unique study *once* per step and scatter its embedding to that study's
covered rows (a "study-batched two-stream" sampler: an ECG stream + a quota of
no-ECG rows), which is the difference between a tractable and an intractable run.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import KEY_COLUMNS, Agent
from agents.ecg_resnet_encoder import (
    ECG_INDEX,
    EMBED_DIM,
    STALENESS_HOURS,
    WAVEFORM_DIR,
    AgeMetadataEncoder,
    EcgMorphologyEncoder,
    FiLMGenerator,
    WaveformStore,
    assemble_ecg_study_ids,
)
from agents.ecg_encoder import N_LEADS
from config import SEED
from metrics import auprc

LABEL = "label_24h"
PATIENT_AGE = "data/cache/patient_age.parquet"


# --- Model -----------------------------------------------------------------

class EcgResnetNet(nn.Module):
    """ResNet encoder + presence-guarded head; optional FiLM(age) modulation.

    ``encode`` and ``head_from_feat`` are split so the (expensive) encoder can be
    run once per unique study and its embedding scattered to that study's rows.
    """

    def __init__(self, embed_dim=EMBED_DIM, hidden=128, latent=32, dropout=0.2,
                 in_channels=N_LEADS, use_film=False):
        super().__init__()
        self.encoder = EcgMorphologyEncoder(in_channels, embed_dim)
        self.no_ecg_embedding = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.no_ecg_embedding, std=0.02)
        self.use_film = use_film
        if use_film:
            self.metadata_encoder = AgeMetadataEncoder(embed_dim=128)
            self.film_generator = FiLMGenerator(context_dim=128, target_feature_dim=embed_dim)
        self.body = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, latent), nn.ReLU(),
        )
        self.head = nn.Linear(latent, 1)
        self.latent_dim = latent
        self.embed_dim = embed_dim

    def encode(self, waveforms):                       # (B, 12, 5000) -> (B, D)
        return self.encoder(waveforms)

    def head_from_feat(self, feat, has_ecg, ecg_age, patient_age=None):
        """Presence-guarded head over per-row ResNet embeddings.

        ``feat`` is the raw ResNet embedding for covered rows (any placeholder is
        fine for no-ECG rows -- it is masked out). FiLM, when enabled, modulates
        ``feat`` *before* the substitution, so it only ever touches the
        ECG-present pathway and no-ECG rows keep the single unmodulated vector.
        """
        if self.use_film:
            if patient_age is None:
                raise ValueError("FiLM variant requires patient_age")
            gamma, beta = self.film_generator(self.metadata_encoder(patient_age.unsqueeze(-1)))
            feat = gamma * feat + beta
        mask = has_ecg.unsqueeze(-1)
        x = mask * feat + (1.0 - mask) * self.no_ecg_embedding
        # Staleness is only meaningful when an ECG exists; zeroed otherwise, so
        # every no-ECG row presents an identical input vector.
        x = torch.cat([x, (ecg_age * has_ecg).unsqueeze(-1)], dim=-1)
        latent = self.body(x)
        return self.head(latent).squeeze(-1), latent


# --- Agent -----------------------------------------------------------------

class EcgResnetAgent(Agent):
    """V2: 1D-ResNet trained end-to-end on the ECG waveforms -> latent + risk."""

    name = "ecg_resnet"
    use_film = False

    def __init__(self, hidden=128, latent=32, dropout=0.2, lr=1e-3, weight_decay=1e-4,
                 n_studies=32, rows_per_study=4, noecg_ratio=1.0, encode_batch=256,
                 max_epochs=12, patience=3, seed=SEED, device=None,
                 waveform_dir=WAVEFORM_DIR, ecg_index=ECG_INDEX,
                 staleness=STALENESS_HOURS, max_ram_gb=12.0, verbose=True):
        super().__init__()
        self.hparams = dict(hidden=hidden, latent=latent, dropout=dropout)
        self.lr, self.weight_decay = lr, weight_decay
        self.n_studies, self.rows_per_study = n_studies, rows_per_study
        self.noecg_ratio, self.encode_batch = noecg_ratio, encode_batch
        self.max_epochs, self.patience = max_epochs, patience
        self.seed, self.verbose = seed, verbose
        self.latent_dim = latent
        self.staleness = staleness
        self.ecg_index_path = ecg_index
        self.device = device or (torch.device("cuda") if torch.cuda.is_available()
                                 else torch.device("cpu"))
        self.store = WaveformStore(waveform_dir, max_ram_gb=max_ram_gb, verbose=verbose)
        self._index = None
        self.net = None

    # -- data -------------------------------------------------------------
    def _load_index(self):
        if self._index is None:
            self._index = pd.read_parquet(self.ecg_index_path)
        return self._index

    def _patient_age(self, df, keys):
        """Per-row patient age aligned to ``keys`` (zeros for the non-FiLM V2)."""
        return np.zeros(len(keys), dtype=np.float32)

    def _align(self, df):
        """Map each row to its covered study (store index) + staleness + age.

        Returns a dict with everything the loops need, in ``(stay_id, hour)``
        order: ``sidx`` (store row index, -1 if no usable/on-disk ECG), ``has``,
        ``ecg_age``, ``age`` (patient age), ``keys``.
        """
        study, ecg_age, keys = assemble_ecg_study_ids(
            df, self._load_index(), staleness=self.staleness)
        n = len(keys)
        sidx = np.full(n, -1, dtype=np.int64)
        has = np.zeros(n, dtype=np.float32)
        for i in np.flatnonzero(~np.isnan(study)):
            s = int(study[i])
            if self.store.has(s):
                sidx[i] = self.store.index[s]
                has[i] = 1.0
        ecg_age = np.where(has > 0, ecg_age, 0.0).astype(np.float32)
        age = self._patient_age(df, keys)
        return dict(sidx=sidx, has=has, ecg_age=ecg_age, age=age, keys=keys)

    def _labels(self, df, keys):
        ordered = df.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        if not (ordered["stay_id"].to_numpy() == keys["stay_id"].to_numpy()).all():
            raise ValueError("ECG features are misaligned with the cohort rows")
        return ordered[LABEL].to_numpy(dtype=float)

    def _make_net(self):
        return EcgResnetNet(embed_dim=EMBED_DIM, in_channels=N_LEADS,
                            use_film=self.use_film, **self.hparams).to(self.device)

    # -- training ---------------------------------------------------------
    def _fit_model(self, train_df, val_df):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        # Store: which studies are on disk, preloaded (presence drives _align).
        # A pre-populated store (tests/smoke via WaveformStore.from_arrays) is
        # used as-is rather than reloaded from disk.
        idx = self._load_index()
        if not self.store.index:
            self.store.load(idx.study_id.unique())
        tr = self._align(train_df)
        # Standardise on TRAIN-covered studies ONLY (leakage discipline): recover
        # their study_ids from the store's index (store row -> study_id).
        inv = {row: sid for sid, row in self.store.index.items()}
        covered_train_studies = {inv[int(r)] for r in set(tr["sidx"][tr["has"] > 0])}
        self.store.fit_standardizer(covered_train_studies)

        ytr = self._labels(train_df, tr["keys"])
        va = self._align(val_df)
        yva = self._labels(val_df, va["keys"])

        self.net = self._make_net()
        if self.verbose:
            n_params = sum(p.numel() for p in self.net.parameters())
            print(f"[{self.name}] {n_params:,} params on {self.device} "
                  f"(FiLM={self.use_film}); {100*tr['has'].mean():.1f}% of train "
                  f"hours ECG-covered, {int(tr['has'].sum()):,} covered rows over "
                  f"{len(set(tr['sidx'][tr['has']>0])):,} studies")

        # Group covered train rows by study_id for the two-stream sampler
        # (store.batch/encode work in study-id space; inv maps store row -> id).
        cov_rows = np.flatnonzero(tr["has"] > 0)
        by_study = {}
        for r in cov_rows:
            by_study.setdefault(inv[int(tr["sidx"][r])], []).append(r)
        studies = np.array(sorted(by_study), dtype=np.int64)
        by_study = {s: np.array(v) for s, v in by_study.items()}
        noecg_rows = np.flatnonzero(tr["has"] == 0)

        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        best, best_state, stale = -np.inf, None, 0

        for epoch in range(1, self.max_epochs + 1):
            self.net.train()
            order = rng.permutation(len(studies))
            total, seen = 0.0, 0
            for b in range(0, len(order), self.n_studies):
                batch_studies = studies[order[b:b + self.n_studies]]
                feats = self.net.encode(self.store.batch(batch_studies, self.device))
                # ECG stream: up to rows_per_study covered rows per study.
                rows, feat_rows = [], []
                for j, s in enumerate(batch_studies):
                    cand = by_study[int(s)]
                    take = cand if len(cand) <= self.rows_per_study else \
                        rng.choice(cand, self.rows_per_study, replace=False)
                    rows.extend(take.tolist())
                    feat_rows.append(feats[j].expand(len(take), -1))
                feat_ecg = torch.cat(feat_rows, dim=0)
                n_ecg = len(rows)
                # No-ECG stream: a quota drawn fresh each step.
                n_no = int(self.noecg_ratio * n_ecg)
                no_rows = (rng.choice(noecg_rows, min(n_no, len(noecg_rows)),
                                      replace=False) if len(noecg_rows) else np.array([], int))
                feat = torch.cat(
                    [feat_ecg, torch.zeros(len(no_rows), self.net.embed_dim,
                                           device=self.device)], dim=0)
                sel = np.array(rows + no_rows.tolist(), dtype=np.int64)
                has = torch.from_numpy(tr["has"][sel]).to(self.device)
                ecg_age = torch.from_numpy(tr["ecg_age"][sel]).to(self.device)
                age = torch.from_numpy(tr["age"][sel]).to(self.device)
                y = torch.from_numpy(ytr[sel].astype(np.float32)).to(self.device)

                logits, _ = self.net.head_from_feat(feat, has, ecg_age, age)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss.detach()) * len(sel); seen += len(sel)

            scores = self._scores(va)
            cov = va["has"] > 0
            sel_auprc = (auprc(yva[cov], scores[cov]) if yva[cov].sum() > 0
                         else auprc(yva, scores))
            if self.verbose:
                print(f"  epoch {epoch}: loss {total/max(seen,1):.4f} | "
                      f"val AUPRC(ECG-available) {sel_auprc:.4f}")
            if sel_auprc > best:
                best, stale = sel_auprc, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
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

    # -- inference --------------------------------------------------------
    @torch.no_grad()
    def _encode_all(self):
        """Embed every on-disk study once; returns (S, D) tensor on device."""
        self.net.eval()
        n = len(self.store.index)
        feats = torch.zeros(n, self.net.embed_dim, device=self.device)
        studies = np.array(sorted(self.store.index, key=lambda s: self.store.index[s]),
                           dtype=np.int64)
        for i in range(0, len(studies), self.encode_batch):
            chunk = studies[i:i + self.encode_batch]
            feats[i:i + len(chunk)] = self.net.encode(self.store.batch(chunk, self.device))
        return feats

    @torch.no_grad()
    def _forward_rows(self, aln, feats_all):
        """Per-row logits + latents given precomputed study embeddings."""
        self.net.eval()
        n = len(aln["keys"])
        raw = np.empty(n, np.float32)
        lat = np.empty((n, self.latent_dim), np.float32)
        for i in range(0, n, self.encode_batch * 8):
            sl = slice(i, i + self.encode_batch * 8)
            sidx = aln["sidx"][sl]
            feat = torch.zeros(len(sidx), self.net.embed_dim, device=self.device)
            cov = sidx >= 0
            if cov.any():
                dst = torch.from_numpy(np.flatnonzero(cov)).to(self.device)
                src = feats_all[torch.from_numpy(sidx[cov]).to(self.device)]
                feat.index_copy_(0, dst, src)
            has = torch.from_numpy(aln["has"][sl]).to(self.device)
            ecg_age = torch.from_numpy(aln["ecg_age"][sl]).to(self.device)
            age = torch.from_numpy(aln["age"][sl]).to(self.device)
            logits, latent = self.net.head_from_feat(feat, has, ecg_age, age)
            raw[sl] = torch.sigmoid(logits).cpu().numpy()
            lat[sl] = latent.cpu().numpy()
        return raw, lat

    def _scores(self, aln):
        """Calibration-free probabilities for an aligned frame (val early-stop)."""
        feats = self._encode_all()
        return self._forward_rows(aln, feats)[0]

    @torch.no_grad()
    def _emit_raw(self, df):
        if self.net is None:
            raise RuntimeError("agent is not fitted -- call fit() first")
        aln = self._align(df)
        feats = self._encode_all()
        raw, lat = self._forward_rows(aln, feats)
        return aln["keys"], lat.astype(np.float32), raw

    def has_ecg(self, df):
        """Boolean mask: does this row have an on-disk ECG within the bound?"""
        return self._align(df)["has"] > 0


class EcgResnetFilmAgent(EcgResnetAgent):
    """V3: V2 + FiLM conditioning on patient age (ECG-present pathway only)."""

    name = "ecg_resnet_film"
    use_film = True

    def __init__(self, *args, patient_age=PATIENT_AGE, **kwargs):
        super().__init__(*args, **kwargs)
        self.patient_age_path = patient_age
        self._age_lookup = None

    def _load_age(self):
        if self._age_lookup is None:
            age = pd.read_parquet(self.patient_age_path)
            self._age_lookup = dict(zip(age.patient_id.astype("int64"),
                                        age.anchor_age.astype("float32")))
        return self._age_lookup

    def _patient_age(self, df, keys):
        """Patient age per row, aligned to ``keys`` (stay -> patient_id map)."""
        ordered = df.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        lut = self._load_age()
        pid = ordered["patient_id"].astype("int64").to_numpy()
        med = float(np.median(list(lut.values()))) if lut else 0.0
        return np.array([lut.get(int(p), med) for p in pid], dtype=np.float32)
