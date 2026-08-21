"""Supervised pretraining of ECG waveform encoders on machine-read targets (V5).

V2 showed that 4M-parameter encoders cannot learn from ``label_24h`` alone (one
weak, noisy bit per study -> instant overfit). Here the encoder first learns
real electrophysiology: for every study the ECG cart already provides ~20 dense
targets -- diagnostic statement flags (AFib, blocks, ectopy, ischemia, ...) and
conduction intervals (HR, PR, QRS, QT/QTc, axes) -- featurized by
:mod:`agents.ecg_meas_features`. The encoder is trained multi-task on those,
then **frozen**; the small ``label_24h`` head runs over cached embeddings via
the existing :class:`agents.ecg_agent.EcgAgent` machinery, exactly like V1.

Split discipline: pretraining sees TRAIN-split patients' studies only, with a
patient-grouped 90/10 carve-out for early stopping -- never the real val split
(it later early-stops and calibrates the label head). Val/test studies are
touched exactly once: frozen embedding extraction.

Augmentation is restricted to transforms that leave the regression targets
true: amplitude/per-lead scaling, noise, lead dropout, time shift. Anything
that warps time (resampling, stretching) would silently corrupt RR/QT targets
and is banned. MixUp/CutMix are excluded (they hurt on ECG per the ECG-Mamba
literature).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.ecg_meas_features import STATEMENT_PATTERNS
from agents.ecg_resnet_encoder import EcgMorphologyEncoder, WaveformStore
from agents.ecg_mamba_encoder import EcgMambaEncoder
from config import SEED

MEAS_CACHE = "data/cache/ecg_meas_features.npz"
ECG_INDEX = "data/cache/ecg_index.parquet"
COHORT = "data/cache/mimic_cohort_hourly.parquet"

# Regression targets: machine-read numerics the encoder must recover from the
# waveform. Rate/rhythm and conduction/repolarisation both represented.
REGRESSION_TARGETS = ["hr", "pr", "qrs_dur", "qt", "qtc_bazett", "p_dur",
                      "p_axis", "qrs_axis", "t_axis"]
# Classification targets: every statement flag EXCEPT acquisition-quality ones
# (lead-dropout augmentation would contradict a "no quality problem" label).
EXCLUDED_FLAGS = ("quality_warning", "has_urgent")
MIN_POSITIVES = 50

BACKBONES = {
    "resnet": lambda: EcgMorphologyEncoder(in_channels=12, final_embedding_size=256),
    "mamba": lambda: EcgMambaEncoder(in_channels=12, final_embedding_size=256),
}


# --- Targets ----------------------------------------------------------------

def study_splits(index_path=ECG_INDEX, cohort_path=COHORT):
    """study_id -> (split, patient_id) via its stay(s); patient-level splits
    guarantee each study maps to exactly one split."""
    idx = pd.read_parquet(index_path)[["stay_id", "study_id"]]
    coh = (pd.read_parquet(cohort_path, columns=["stay_id", "patient_id", "split"])
           .drop_duplicates("stay_id"))
    j = idx.merge(coh, on="stay_id").drop_duplicates("study_id")
    return (j.set_index("study_id")["split"].to_dict(),
            j.set_index("study_id")["patient_id"].to_dict())


class PretrainTargets:
    """Per-study targets pulled out of the ecg_meas feature cache.

    ``flags`` (S, K) float32 {0,1}; ``values`` (S, M) z-scored; ``valid`` (S, M)
    mask. Z-stats are fit on the PRETRAIN-TRAIN studies only and stored for the
    checkpoint.
    """

    def __init__(self, meas_cache=MEAS_CACHE):
        z = np.load(meas_cache)
        self.names = [str(n) for n in z["feature_names"]]
        self.X = z["embeddings"]
        self.lookup = {int(s): i for i, s in enumerate(z["study_ids"])}
        self.flag_names = [n for n in STATEMENT_PATTERNS if n not in EXCLUDED_FLAGS]
        self.zstats = None

    def _cols(self, names):
        return np.array([self.names.index(n) for n in names])

    def select(self, study_ids):
        rows = np.array([self.lookup[int(s)] for s in study_ids])
        flags = self.X[np.ix_(rows, self._cols(self.flag_names))]
        values = self.X[np.ix_(rows, self._cols(REGRESSION_TARGETS))]
        valid = self.X[np.ix_(rows, self._cols(
            [f"{n}_valid" for n in REGRESSION_TARGETS]))]
        return flags.astype(np.float32), values.astype(np.float32), valid.astype(np.float32)

    def has(self, study_id):
        return int(study_id) in self.lookup

    def fit_zstats(self, values, valid):
        """Masked per-target mean/std from pretrain-train values."""
        mean = np.array([values[valid[:, j] > 0, j].mean() if valid[:, j].sum()
                         else 0.0 for j in range(values.shape[1])])
        std = np.array([values[valid[:, j] > 0, j].std() if valid[:, j].sum()
                        else 1.0 for j in range(values.shape[1])])
        self.zstats = (mean.astype(np.float32),
                       np.where(std > 1e-6, std, 1.0).astype(np.float32))
        return self.zstats

    def zscore(self, values):
        mean, std = self.zstats
        return (values - mean) / std


def prune_rare_flags(targets, flags_train):
    """Keep flags with >= MIN_POSITIVES in the pretrain-train set; print census."""
    counts = flags_train.sum(axis=0).astype(int)
    keep = [i for i, c in enumerate(counts) if c >= MIN_POSITIVES]
    dropped = [(targets.flag_names[i], int(counts[i]))
               for i in range(len(counts)) if i not in keep]
    targets.flag_names = [targets.flag_names[i] for i in keep]
    return keep, dropped


# --- Model ------------------------------------------------------------------

class EcgPretrainModel(nn.Module):
    """Backbone -> 256-d embedding -> shared trunk -> (flags logits, values).

    The heads are discarded at extraction time; ``embed()`` is the deliverable.
    """

    def __init__(self, backbone, n_flags, n_values, embed_dim=256):
        super().__init__()
        self.backbone = backbone
        self.trunk = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.cls_head = nn.Linear(embed_dim, n_flags)
        self.reg_head = nn.Linear(embed_dim, n_values)

    def embed(self, x):
        return self.backbone(x)

    def forward(self, x):
        h = self.trunk(self.embed(x))
        return self.cls_head(h), self.reg_head(h)


# --- Augmentation -----------------------------------------------------------

def augment(x, rng):
    """Target-preserving augmentation of a (B, 12, 5000) batch, in place-ish.

    Amplitude ops leave intervals untouched; time SHIFT (crop-pad) moves the
    window but not beat spacing. No time-warping -- it would corrupt RR/QT.
    """
    B, C, L = x.shape
    dev = x.device
    x = x * torch.as_tensor(rng.uniform(0.8, 1.25, (B, 1, 1)),
                            dtype=x.dtype, device=dev)
    x = x * torch.as_tensor(rng.uniform(0.95, 1.05, (B, C, 1)),
                            dtype=x.dtype, device=dev)
    x = x + 0.02 * x.std(dim=-1, keepdim=True) * torch.randn_like(x)
    drop = torch.as_tensor(rng.random((B, C, 1)) < 0.2 / 12 * 2,
                           device=dev)                    # ~0-2 leads expected
    x = x.masked_fill(drop, 0.0)
    shift = int(rng.integers(-250, 251))                  # +-0.5 s @ 500 Hz
    if shift:
        x = torch.roll(x, shifts=shift, dims=-1)
        if shift > 0:
            x[..., :shift] = 0.0
        else:
            x[..., shift:] = 0.0
    return x


# --- Training loop ----------------------------------------------------------

def pretrain(backbone_name, store=None, targets=None, epochs=100, patience=7,
             batch_size=64, lr=3e-4, weight_decay=1e-4, seed=SEED,
             device=None, verbose=True, out_path=None):
    """Pretrain one backbone; returns (model, history). Saves checkpoint if
    ``out_path`` given: backbone weights + store stats + z-stats + flag names."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or (torch.device("cuda") if torch.cuda.is_available()
                        else torch.device("cpu"))

    split_of, patient_of = study_splits()
    targets = targets or PretrainTargets()
    store = store or WaveformStore(verbose=verbose)
    if not store.index:
        store.load(list(split_of))

    # Pretrain pool: downloaded TRAIN-split studies with measurement targets.
    pool = sorted(s for s in store.index
                  if split_of.get(s) == "train" and targets.has(s))
    patients = np.array([patient_of[s] for s in pool])
    uniq = np.unique(patients)
    val_pat = set(rng.choice(uniq, max(1, int(0.1 * len(uniq))), replace=False))
    tr_ids = [s for s, p in zip(pool, patients) if p not in val_pat]
    va_ids = [s for s, p in zip(pool, patients) if p in val_pat]
    if verbose:
        print(f"[pretrain:{backbone_name}] {len(tr_ids):,} train / "
              f"{len(va_ids):,} val studies ({len(uniq):,} patients)")

    store.fit_standardizer(tr_ids)

    # Select BOTH splits before pruning: prune_rare_flags shrinks
    # targets.flag_names, so select() afterwards would double-subset.
    flags_tr, values_tr, valid_tr = targets.select(tr_ids)
    flags_va, values_va, valid_va = targets.select(va_ids)
    keep, dropped = prune_rare_flags(targets, flags_tr)
    if verbose and dropped:
        print(f"  dropped rare flags (<{MIN_POSITIVES} pos): {dropped}")
    flags_tr, flags_va = flags_tr[:, keep], flags_va[:, keep]
    targets.fit_zstats(values_tr, valid_tr)
    zvals_tr, zvals_va = targets.zscore(values_tr), targets.zscore(values_va)

    pos = flags_tr.sum(axis=0)
    pos_weight = torch.as_tensor(
        np.clip((len(flags_tr) - pos) / np.maximum(pos, 1), 1.0, 20.0),
        dtype=torch.float32, device=device)

    model = EcgPretrainModel(BACKBONES[backbone_name](), n_flags=len(keep),
                             n_values=len(REGRESSION_TARGETS)).to(device)
    if verbose:
        n = sum(p.numel() for p in model.backbone.parameters())
        print(f"  backbone params: {n:,} | flags: {len(keep)} | "
              f"regression targets: {len(REGRESSION_TARGETS)} | device {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    to = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    ftr, ztr, mtr = to(flags_tr), to(zvals_tr), to(valid_tr)
    fva, zva, mva = to(flags_va), to(zvals_va), to(valid_va)

    def epoch_pass(ids, flags, zvals, masks, train):
        model.train(train)
        total, bces, regs = 0.0, 0.0, 0.0
        order = rng.permutation(len(ids)) if train else np.arange(len(ids))
        val_logits = []
        for i in range(0, len(order), batch_size):
            sel = order[i:i + batch_size]
            x = store.batch([ids[j] for j in sel], device)
            if train:
                x = augment(x, rng)
            with torch.set_grad_enabled(train):
                logits, preds = model(x)
                bce = F.binary_cross_entropy_with_logits(
                    logits, flags[sel], pos_weight=pos_weight)
                m = masks[sel]
                reg = (F.smooth_l1_loss(preds, zvals[sel], reduction="none")
                       * m).sum() / m.sum().clamp(min=1.0)
                loss = bce + reg
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            else:
                val_logits.append(logits.detach().cpu().numpy())
            total += float(loss) * len(sel)
            bces += float(bce) * len(sel)
            regs += float(reg) * len(sel)
        n = len(order)
        return total / n, bces / n, regs / n, val_logits

    def macro_auroc(logits):
        from metrics import auroc
        scores = np.concatenate(logits)
        out = []
        for k in range(scores.shape[1]):
            yk = flags_va[:, k]
            if 5 <= yk.sum() <= len(yk) - 5:
                out.append(auroc(yk.astype(float), scores[:, k]))
        return float(np.mean(out)) if out else float("nan")

    best, best_state, stale, history = np.inf, None, 0, []
    for ep in range(1, epochs + 1):
        tr_loss, tr_bce, tr_reg, _ = epoch_pass(tr_ids, ftr, ztr, mtr, True)
        va_loss, va_bce, va_reg, vl = epoch_pass(va_ids, fva, zva, mva, False)
        auc = macro_auroc(vl)
        sched.step()
        history.append(dict(epoch=ep, train=tr_loss, val=va_loss, auroc=auc))
        if verbose:
            print(f"  epoch {ep:3d}: train {tr_loss:.4f} (bce {tr_bce:.3f} reg {tr_reg:.3f})"
                  f" | val {va_loss:.4f} | val macro-AUROC {auc:.3f}")
        if va_loss < best:
            best, stale = va_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f"  early stop (no val gain for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "backbone": backbone_name,
            "state_dict": model.state_dict(),
            "store_mean": store.mean, "store_std": store.std,
            "zstats": targets.zstats, "flag_names": targets.flag_names,
            "history": history,
        }, out_path)
        if verbose:
            print(f"[pretrain:{backbone_name}] saved -> {out_path}")
    return model, history
