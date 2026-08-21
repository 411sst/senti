"""Agent H1: the gated-residual ECG omni-agent (machine read + waveform).

The bake-off (results_ecg.md) established: the cart's machine read is the only
fusion-positive ECG source; frozen waveform embeddings transfer ≈ nothing on
their own; and sparse extra channels *hurt* the orchestrator. H1 therefore
fuses everything ECG **inside one agent**, with the machine read as the
backbone and the waveform as a gated residual:

    latent = meas_body([meas ⊕ Δmeas ⊕ age ⊕ staleness])
             + gate · wave_body(Linear(founder_embedding))
    risk   = head(latent)

* ``gate`` is a scalar parameter initialised to 0, so at initialisation H1 IS
  the structured-features model; "H1 − V4" cleanly isolates the waveform's
  incremental value.
* **Δmeas is the novel block**: change versus the patient's PREVIOUS ECG in the
  same stay — QTc drift, HR trend, axis shift, new-onset AFib/ischemia flags,
  time between carts. Serial-ECG change is clinically canonical for
  deterioration and unused by every other variant and baseline.
* **Three presence states** extend the tested guard: no ECG -> learned
  ``no_ecg`` vector (ONE constant risk, auditability preserved); measured-only
  -> real features + learned ``no_wave`` slot; measured+downloaded -> both.
  ``has_wave`` implies ``has_meas``'s ECG exists, so no-ECG rows have both
  slots substituted => the constant survives the hybrid.

Leakage: the current-ECG pick is the shared strictly-before-t merge; the
"previous" ECG is the predecessor by ``ecg_hour`` within the same stay, so it
is strictly older than an ECG that is itself strictly before t.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import KEY_COLUMNS, Agent
from agents.ecg_resnet_encoder import assemble_ecg_study_ids
from config import SEED
from metrics import auprc

LABEL = "label_24h"
MEAS_CACHE = "data/cache/ecg_meas_features.npz"
WAVE_CACHE = "data/cache/ecgfounder_embeddings.npz"
ECG_INDEX = "data/cache/ecg_index.parquet"
PATIENT_AGE = "data/cache/patient_age.parquet"
STALENESS_HOURS = 72

# Δmeas: numeric drifts + new-onset flags computed against the previous ECG of
# the SAME stay. All zero (with has_prev=0) when there is no previous ECG.
DELTA_NUMERIC = ("hr", "qtc_bazett", "qrs_dur", "qt", "qrs_axis")
DELTA_ONSET = ("afib", "acute_ischemia", "st_t_abnormal", "aflutter", "vtach")
DELTA_NAMES = ([f"d_{n}" for n in DELTA_NUMERIC]
               + [f"new_{n}" for n in DELTA_ONSET]
               + ["hours_since_prev", "log1p_n_prior", "has_prev"])


def build_delta_features(index, meas_feats, meas_names, meas_lookup):
    """Per (stay_id, study_id): change features vs the stay's previous ECG.

    ``index`` rows are (stay_id, ecg_hour, study_id); the previous ECG is the
    row with the largest strictly-smaller ``ecg_hour`` within the stay. Returns
    ``{(stay_id, study_id): (len(DELTA_NAMES),) float32}``.
    """
    col = {n: meas_names.index(n) for n in DELTA_NUMERIC + DELTA_ONSET}
    valid_col = {n: meas_names.index(f"{n}_valid") for n in DELTA_NUMERIC}
    out = {}
    idx = index.sort_values(["stay_id", "ecg_hour"])
    for stay, grp in idx.groupby("stay_id", sort=False):
        prev_sid, prev_hour, n_prior = None, None, 0
        for _, row in grp.iterrows():
            sid = int(row.study_id)
            feats = np.zeros(len(DELTA_NAMES), dtype=np.float32)
            cur = meas_lookup.get(sid)
            prev = meas_lookup.get(prev_sid) if prev_sid is not None else None
            if cur is not None and prev is not None:
                fc, fp = meas_feats[cur], meas_feats[prev]
                for i, n in enumerate(DELTA_NUMERIC):
                    # drift only when BOTH readings are valid
                    if fc[valid_col[n]] > 0 and fp[valid_col[n]] > 0:
                        d = fc[col[n]] - fp[col[n]]
                        if n == "qrs_axis":              # wrap axis difference
                            d = (d + 180.0) % 360.0 - 180.0
                        feats[i] = d
                for j, n in enumerate(DELTA_ONSET):
                    feats[len(DELTA_NUMERIC) + j] = float(
                        fc[col[n]] > 0 and fp[col[n]] == 0)   # new onset
                feats[-3] = float(row.ecg_hour - prev_hour)   # hours since prev
                feats[-2] = np.log1p(n_prior)
                feats[-1] = 1.0
            out[(int(stay), sid)] = feats
            prev_sid, prev_hour = sid, float(row.ecg_hour)
            n_prior += 1
    return out


# --- Model -----------------------------------------------------------------

class EcgHybridNet(nn.Module):
    """Presence-guarded gated-residual head over meas(+Δ) and a waveform slot."""

    def __init__(self, meas_dim, wave_dim=1024, wave_proj=64, hidden=128,
                 latent=32, dropout=0.2, latent_mode="sum", use_film=False):
        """``latent_mode``:

        * ``"sum"`` (H1/H2): latent = meas_body + gate·wave_body; one head.
        * ``"concat_aux"`` (H5): risk comes from the meas latent ONLY (so the
          agent's risk is provably V4-equivalent in structure), the wave latent
          is trained by an auxiliary head and CONCATENATED into the emitted
          latent -- exploiting the mamba finding that latents can carry
          fusable information even when they don't improve the risk ranking.

        ``use_film`` (H6): FiLM-modulate the measurement features with patient
        age -- gamma(age)*meas + beta(age) -- so age conditions the
        *interpretation* of each measurement (multiplicative interactions like
        age-dependent QTc/HR norms) rather than only shifting risk additively.
        Applied to the ECG-present branch only, before the no-ECG substitution,
        so the single-constant audit survives (V3 rationale).
        """
        super().__init__()
        self.latent_mode = latent_mode
        self.use_film = use_film
        if use_film:
            from agents.ecg_resnet_encoder import AgeMetadataEncoder, FiLMGenerator
            self.metadata_encoder = AgeMetadataEncoder(embed_dim=64)
            self.film_generator = FiLMGenerator(context_dim=64,
                                                target_feature_dim=meas_dim)
        self.no_ecg_embedding = nn.Parameter(torch.zeros(meas_dim))
        nn.init.normal_(self.no_ecg_embedding, std=0.02)
        self.no_wave_embedding = nn.Parameter(torch.zeros(wave_proj))
        nn.init.normal_(self.no_wave_embedding, std=0.02)
        self.wave_proj = nn.Linear(wave_dim, wave_proj)
        # meas backbone: [meas_aug ⊕ staleness ⊕ age]
        self.meas_body = nn.Sequential(
            nn.Linear(meas_dim + 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, latent), nn.ReLU(),
        )
        # waveform residual arm (over the guarded wave slot)
        self.wave_body = nn.Sequential(
            nn.Linear(wave_proj, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, latent),
        )
        self.gate = nn.Parameter(torch.zeros(1))     # residual OFF at init
        self.head = nn.Linear(latent, 1)
        self.aux_head = nn.Linear(latent, 1)         # trains wave arm in concat_aux
        self.latent_dim = latent if latent_mode == "sum" else 2 * latent

    def forward(self, meas, has_meas, wave, has_wave, ecg_age, age):
        m = has_meas.unsqueeze(-1)
        if self.use_film:
            # age arrives already /100 (agent convention); the V3 encoder
            # divides by 100 itself, so rescale back to years here.
            gamma, beta = self.film_generator(
                self.metadata_encoder((age * 100.0).unsqueeze(-1)))
            meas = gamma * meas + beta          # present branch only (masked next)
        x = m * meas + (1.0 - m) * self.no_ecg_embedding
        # Age (like staleness) is gated by ECG presence: age-related BASELINE
        # risk belongs to the orchestrator/other agents, and an ungated age
        # would break the "no-ECG hours emit ONE constant" audit (V3 rationale).
        x = torch.cat([x, (ecg_age * has_meas).unsqueeze(-1),
                       (age * has_meas).unsqueeze(-1)], dim=-1)
        w = has_wave.unsqueeze(-1)
        wslot = w * self.wave_proj(wave) + (1.0 - w) * self.no_wave_embedding
        meas_lat = self.meas_body(x)
        wave_lat = self.wave_body(wslot)
        if self.latent_mode == "sum":
            latent = meas_lat + self.gate * wave_lat
            return self.head(latent).squeeze(-1), latent
        # concat_aux: risk reads the meas latent only; the wave latent is
        # trained through aux_head (see agent loss) and only travels onward
        # inside the emitted latent.
        latent = torch.cat([meas_lat, wave_lat], dim=-1)
        return self.head(meas_lat).squeeze(-1), latent

    def aux_logits(self, wave_lat):
        return self.aux_head(wave_lat).squeeze(-1)


# --- Agent -----------------------------------------------------------------

class EcgHybridAgent(Agent):
    """H1: machine read + Δmeas + gated waveform residual -> latent + risk."""

    name = "ecg_hybrid"

    def __init__(self, hidden=128, latent=32, wave_proj=64, dropout=0.2,
                 lr=1e-3, batch_size=4096, max_epochs=15, patience=3,
                 seed=SEED, meas_cache=MEAS_CACHE, wave_cache=WAVE_CACHE,
                 ecg_index=ECG_INDEX, patient_age=PATIENT_AGE,
                 staleness=STALENESS_HOURS, use_delta=True, use_wave=True,
                 normalize_wave=False, latent_mode="sum", use_film=False,
                 name=None, verbose=True):
        super().__init__()
        # Ablation switches (H3): use_delta=False zeroes the Δmeas block;
        # use_wave=False marks every row waveform-absent (the arm collapses to
        # its learned constant and the gate has nothing to modulate).
        # normalize_wave: z-score the aux-slot matrix column-wise over the
        # cached studies -- needed when the slot holds raw-scale features (HRV)
        # rather than ~unit-scale embeddings.
        self.use_delta, self.use_wave = use_delta, use_wave
        self.normalize_wave = normalize_wave
        self.latent_mode = latent_mode
        self.use_film = use_film
        if name is not None:
            self.name = name
        self.hparams = dict(hidden=hidden, latent=latent, wave_proj=wave_proj,
                            dropout=dropout)
        self.lr, self.batch_size = lr, batch_size
        self.max_epochs, self.patience = max_epochs, patience
        self.seed, self.verbose = seed, verbose
        self.latent_dim = latent
        self.staleness = staleness
        self.paths = dict(meas=meas_cache, wave=wave_cache, index=ecg_index,
                          age=patient_age)
        self.net, self.stats = None, None
        self._cache = None
        self.device = torch.device("cpu")

    # -- data -------------------------------------------------------------
    def _load_cache(self):
        """(meas_feats, meas_lookup, meas_names, wave_emb, wave_lookup, index,
        delta_map, age_lookup) -- injectable by tests via self._cache."""
        if self._cache is None:
            zm = np.load(self.paths["meas"])
            meas = zm["embeddings"].astype(np.float32)
            names = [str(n) for n in zm["feature_names"]]
            mlook = {int(s): i for i, s in enumerate(zm["study_ids"])}
            zw = np.load(self.paths["wave"])
            wave = zw["embeddings"].astype(np.float32)
            wlook = {int(s): i for i, s in enumerate(zw["study_ids"])}
            index = pd.read_parquet(self.paths["index"])
            delta = build_delta_features(index, meas, names, mlook)
            adf = pd.read_parquet(self.paths["age"])
            alook = dict(zip(adf.patient_id.astype("int64"),
                             adf.anchor_age.astype("float32")))
            self._cache = (meas, mlook, names, wave, wlook, index, delta, alook)
        return self._cache

    def _features(self, df):
        """Row-aligned arrays for the head; wave kept as an INDEX (gathered per
        batch -- a per-row (N, 1024) matrix would be ~7 GB on the full train
        split)."""
        meas, mlook, names, wave, wlook, index, delta, alook = self._load_cache()
        study, ecg_age, keys = assemble_ecg_study_ids(
            df, index, staleness=self.staleness)
        n = len(keys)
        meas_dim = meas.shape[1] + len(DELTA_NAMES)
        X = np.zeros((n, meas_dim), dtype=np.float32)
        has_meas = np.zeros(n, dtype=np.float32)
        wave_idx = np.full(n, -1, dtype=np.int64)
        stay = keys["stay_id"].to_numpy()
        for i in np.flatnonzero(~np.isnan(study)):
            sid = int(study[i])
            mi = mlook.get(sid)
            if mi is None:
                continue                       # no measurements -> no ECG
            X[i, :meas.shape[1]] = meas[mi]
            if self.use_delta:
                X[i, meas.shape[1]:] = delta.get((int(stay[i]), sid), 0.0)
            has_meas[i] = 1.0
            if self.use_wave:
                wi = wlook.get(sid)
                if wi is not None:
                    wave_idx[i] = wi
        ecg_age = np.where(has_meas > 0, ecg_age, 0.0).astype(np.float32)

        ordered = df.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        med = float(np.median(list(alook.values()))) if alook else 0.0
        age = np.array([alook.get(int(p), med)
                        for p in ordered["patient_id"].astype("int64")],
                       dtype=np.float32) / 100.0
        if self.stats is not None:
            mean, std = self.stats
            X = np.where(has_meas[:, None] > 0, (X - mean) / std, 0.0
                         ).astype(np.float32)
        return X, has_meas, wave_idx, ecg_age, age, keys

    def _labels(self, df, keys):
        ordered = df.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        if not (ordered["stay_id"].to_numpy() == keys["stay_id"].to_numpy()).all():
            raise ValueError("hybrid features misaligned with cohort rows")
        return ordered[LABEL].to_numpy(dtype=float)

    def _wave_matrix(self):
        wave = self._load_cache()[3]
        if self.normalize_wave:
            mean, std = wave.mean(axis=0), wave.std(axis=0)
            wave = (wave - mean) / np.where(std > 1e-6, std, 1.0)
        return torch.from_numpy(wave.astype(np.float32))

    def _batch(self, X, has_meas, wave_idx, ecg_age, age, sl, wave_all):
        """Tensors for rows ``sl``; waveform embeddings gathered on the fly."""
        wi = wave_idx[sl]
        wave = torch.zeros(len(wi), wave_all.shape[1])
        cov = wi >= 0
        if cov.any():
            wave[np.flatnonzero(cov)] = wave_all[wi[cov]]
        return (torch.from_numpy(X[sl]), torch.from_numpy(has_meas[sl]),
                wave, torch.from_numpy((wi >= 0).astype(np.float32)),
                torch.from_numpy(ecg_age[sl]), torch.from_numpy(age[sl]))

    # -- training ---------------------------------------------------------
    def _fit_model(self, train_df, val_df):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        Xtr, htr, wtr, atr, gtr, ktr = self._features(train_df)
        covered = Xtr[htr > 0]
        mean = covered.mean(axis=0) if len(covered) else np.zeros(Xtr.shape[1])
        std = covered.std(axis=0) if len(covered) else np.ones(Xtr.shape[1])
        self.stats = (mean.astype(np.float32),
                      np.where(std > 1e-6, std, 1.0).astype(np.float32))
        Xtr, htr, wtr, atr, gtr, ktr = self._features(train_df)
        ytr = self._labels(train_df, ktr)
        Xva, hva, wva, ava, gva, kva = self._features(val_df)
        yva = self._labels(val_df, kva)
        wave_all = self._wave_matrix()

        self.net = EcgHybridNet(meas_dim=Xtr.shape[1],
                                wave_dim=wave_all.shape[1],
                                latent_mode=self.latent_mode,
                                use_film=self.use_film,
                                **self.hparams).to(self.device)
        self.latent_dim = self.net.latent_dim
        if self.verbose:
            n = sum(p.numel() for p in self.net.parameters())
            print(f"[{self.name}] {n:,} params; meas coverage "
                  f"{100*htr.mean():.1f}% of train hours, waveform on "
                  f"{100*(wtr>=0).mean():.1f}%")

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        yt = torch.from_numpy(ytr.astype(np.float32))
        rng = np.random.default_rng(self.seed)
        best, best_state, stale = -np.inf, None, 0

        for epoch in range(1, self.max_epochs + 1):
            self.net.train()
            order = rng.permutation(len(Xtr))
            total = 0.0
            for i in range(0, len(order), self.batch_size):
                sl = order[i:i + self.batch_size]
                mb = self._batch(Xtr, htr, wtr, atr, gtr, sl, wave_all)
                logits, lat = self.net(*mb)
                loss = F.binary_cross_entropy_with_logits(logits, yt[sl])
                if self.latent_mode == "concat_aux":
                    # auxiliary head trains the wave latent on the label; the
                    # risk itself never reads it.
                    half = lat.shape[1] // 2
                    loss = loss + 0.5 * F.binary_cross_entropy_with_logits(
                        self.net.aux_logits(lat[:, half:]), yt[sl])
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss.detach()) * len(sl)

            scores = self._scores(Xva, hva, wva, ava, gva, wave_all)
            cov = hva > 0
            sel = auprc(yva[cov], scores[cov]) if yva[cov].sum() > 0 else \
                auprc(yva, scores)
            if self.verbose:
                print(f"  epoch {epoch}: loss {total/len(order):.4f} | "
                      f"val AUPRC(ECG-available) {sel:.4f} | "
                      f"gate {float(self.net.gate):+.4f}")
            if sel > best:
                best, stale = sel, 0
                best_state = {k: v.detach().clone()
                              for k, v in self.net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    if self.verbose:
                        print(f"  early stop (no gain for {self.patience} epochs)")
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        if self.verbose:
            print(f"[{self.name}] best val AUPRC (ECG-available) {best:.4f} | "
                  f"final gate {float(self.net.gate):+.4f}")
        return self

    @torch.no_grad()
    def _scores(self, X, has, widx, ecg_age, age, wave_all):
        self.net.eval()
        out = []
        for i in range(0, len(X), self.batch_size):
            sl = np.arange(i, min(i + self.batch_size, len(X)))
            logits, _ = self.net(*self._batch(X, has, widx, ecg_age, age, sl,
                                              wave_all))
            out.append(logits.numpy())
        return 1.0 / (1.0 + np.exp(-np.concatenate(out)))

    # -- emission ---------------------------------------------------------
    @torch.no_grad()
    def _emit_raw(self, df):
        if self.net is None:
            raise RuntimeError("agent is not fitted -- call fit() first")
        self.net.eval()
        X, has, widx, ecg_age, age, keys = self._features(df)
        wave_all = self._wave_matrix()
        logits, latents = [], []
        for i in range(0, len(X), self.batch_size):
            sl = np.arange(i, min(i + self.batch_size, len(X)))
            lo, lat = self.net(*self._batch(X, has, widx, ecg_age, age, sl,
                                            wave_all))
            logits.append(lo.numpy())
            latents.append(lat.numpy())
        raw = 1.0 / (1.0 + np.exp(-np.concatenate(logits)))
        return keys, np.concatenate(latents).astype(np.float32), raw

    def has_ecg(self, df):
        return self._features(df)[1] > 0

    def has_wave(self, df):
        return self._features(df)[2] >= 0
