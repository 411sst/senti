"""Bundle 3: LatentCast -- alarm on the forecasted future of the latent bank.

Today the system alarms on the CURRENT fused risk. The channel decomposition
(channel_arms.py) showed the orchestrator's earliness lives in the latent
trajectory; the most direct route to MORE lead time is to extrapolate that
trajectory: forecast the agent-latent bank k hours ahead, push the forecast
through the FROZEN fusion head, and alarm on predicted-future risk at the
usual budgets. The paradigm is proven in 2026 video anticipation (TAP-JEPA,
FLaRA: predict future latents, score with a frozen head); it has no clinical
instantiation with a lead-time metric.

**Substitution semantics (the design decision, settled).** For each stay we
build a forecast sequence ``F_k(s, t)`` = the bank row at ``t+k`` as predicted
from true rows ``<= t`` (a unidirectional GRU, so this holds by construction),
then run the frozen causal TCN over the WHOLE forecast sequence and read its
output at ``t``.

* Leakage-free: every forecast row at position tau is a function of true data
  ``<= tau``; the TCN is causal, so its output at ``t`` reads only forecast
  rows ``<= t`` -- nothing recorded after ``t`` can reach the score at ``t``.
* Semantically right: the forecast sequence approximates the true bank shifted
  left by k, so the frozen head's output at ``t`` approximates the
  orchestrator's risk at ``t+k`` -- exactly "risk k hours from now, as best
  estimable at t". The alternative (true rows to ``t-1`` + one forecast row at
  ``t``) hands the TCN a temporally inconsistent sequence and estimates
  nothing. The oracle test in tests/test_latentcast.py pins this: a perfect
  forecaster makes ``risk_hat_k(t)`` equal the parent's risk at ``t+k``.

Scores are evaluated RAW: AUPRC and volume-matched thresholding are invariant
to monotone recalibration (fair_capability docstring), so the per-horizon
isotonic would change no number reported here.

Controls (both mandatory, both graded on the SAME standard label_24h race):

* ``control-shifted-gbm`` -- retrain the FAIR+TRAJ GBM with a widened label
  (event within 24+k h). If this matches LatentCast, the forecast is just
  label widening.
* ``control-raw-news2`` -- forecast the 5 banded vitals with the same GRU
  recipe and recompute NEWS2 on the forecast. If this matches LatentCast,
  forecasting did not need to happen in latent space. (The forecast is dense,
  so this arm never suffers NEWS2's missing-as-zero degradation -- a bias in
  the control's favour, i.e. conservative.)

    python src/latentcast.py train             [--epochs 20] [--device cuda]
    python src/latentcast.py eval              [--reps 2000]
    python src/latentcast.py control-shifted-gbm  [--device cuda] [--reps 2000]
    python src/latentcast.py control-raw-news2 [--epochs 20] [--device cuda]
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from baselines_mimic import load_splits
from config import SEED
from fair_capability import (frozen_thresholds, paired_first_alarm_diffs,
                             summarize_paired)
from fusion_common import FusionContext
from metrics import auprc
from mimic_cohort import VITALS, compute_news2
from orchestrator.fusion import AgentBank, Orchestrator, config_agents

HORIZONS = (2, 4, 6, 8, 12)
LABEL = "label_24h"
CKPT_BANK = "models/latentcast/gru_bank.pt"
CKPT_VITALS = "models/latentcast/gru_vitals.pt"
ORCH_CKPT = "models/orchestrator/meas_s{seed}.pt"
EVAL_CACHE = "data/cache/latentcast_scores.npz"
SHIFTED_CACHE = "data/cache/latentcast_shifted_gbm.npz"
RAWNEWS2_CACHE = "data/cache/latentcast_rawnews2.npz"
EXCEED_CACHE = "data/cache/latentcast_exceed.npz"
ENSEMBLE_SEEDS = (42, 1, 2, 3, 4)


def _ens_ckpt(seed):
    return (CKPT_BANK if seed == 42
            else f"models/latentcast/gru_bank_s{seed}.pt")


# --- Model -------------------------------------------------------------------

class LatentForecaster(nn.Module):
    """Unidirectional GRU trunk + one direct head per horizon.

    Direct multi-horizon (no autoregressive rollout): each head reads the
    hidden state at ``t`` and predicts the full feature row at ``t+k``, so
    forecast error never compounds across steps. Unidirectionality is the
    leakage guarantee -- output at ``t`` is a function of input rows ``<= t``.
    """

    def __init__(self, n_features, hidden=128, horizons=HORIZONS):
        super().__init__()
        self.horizons = tuple(horizons)
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.heads = nn.ModuleDict(
            {str(k): nn.Linear(hidden, n_features) for k in self.horizons})

    def forward(self, x):
        """``x``: (B, T, F) -> {k: (B, T, F) prediction of row t+k}."""
        h, _ = self.gru(x)
        return {k: self.heads[str(k)](h) for k in self.horizons}


# --- Sequences ---------------------------------------------------------------

def bank_sequences(bank, df):
    """Per-stay feature sequences for ``df``'s rows, sorted by (stay, hour)."""
    X, keys = bank.lookup(df)
    stay = keys["stay_id"].to_numpy()
    hour = keys["hour"].to_numpy()
    bounds = np.flatnonzero(np.diff(stay)) + 1
    seqs = []
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(stay)]):
        seqs.append({"stay_id": stay[lo], "hours": hour[lo:hi], "X": X[lo:hi]})
    return seqs, keys


def collate_forecast(seqs, idx, horizons, device):
    """Padded inputs + per-horizon targets/masks; targets never cross stays.

    ``mask_k[i, t]`` is 1 exactly when row ``t+k`` exists within stay ``i`` --
    the batcher structurally cannot pair ``t`` with another stay's future.
    """
    items = [seqs[i] for i in idx]
    lens = [len(s["hours"]) for s in items]
    B, T, Fdim = len(items), max(lens), items[0]["X"].shape[1]
    x = np.zeros((B, T, Fdim), dtype=np.float32)
    targets = {k: np.zeros((B, T, Fdim), dtype=np.float32) for k in horizons}
    masks = {k: np.zeros((B, T), dtype=np.float32) for k in horizons}
    for i, s in enumerate(items):
        n = lens[i]
        x[i, :n] = s["X"]
        for k in horizons:
            if n > k:
                targets[k][i, :n - k] = s["X"][k:]
                masks[k][i, :n - k] = 1.0
    to = lambda a: torch.from_numpy(a).to(device)
    return (to(x), {k: to(v) for k, v in targets.items()},
            {k: to(v) for k, v in masks.items()})


def _standardize_stats(train_seqs):
    """Per-column mean/std over TRAIN rows only (stored in the checkpoint).

    The std floor is 1e-2, not epsilon: presence-guard columns (e.g. the
    ecg_meas risk, one constant on every no-ECG hour) have std ~4e-6, and
    dividing by that turns the covered hours into z-scores of ~1e5 --
    which detonated training with NaN on the first attempt.
    """
    X = np.concatenate([s["X"] for s in train_seqs])
    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), 1e-2)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_stats(seqs, mean, std, clip=20.0):
    for s in seqs:
        s["X"] = np.clip((s["X"] - mean) / std, -clip, clip)


def _forecast_loss(model, seqs, batch_size, device, rng=None, opt=None):
    """One pass (train if ``opt``): masked SmoothL1 summed over horizons."""
    from agents.vitals_tcn import make_batches
    total, seen = 0.0, 0.0
    training = opt is not None
    model.train(training)
    with torch.set_grad_enabled(training):
        for idx in make_batches(seqs, batch_size, shuffle=training, rng=rng):
            x, targets, masks = collate_forecast(seqs, idx, model.horizons,
                                                 device)
            preds = model(x)
            loss, weight = 0.0, 0.0
            for k in model.horizons:
                per = nn.functional.smooth_l1_loss(
                    preds[k], targets[k], reduction="none").mean(dim=-1)
                loss = loss + (per * masks[k]).sum()
                weight = weight + masks[k].sum()
            loss = loss / weight.clamp(min=1.0)
            if training:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            total += float(loss.detach()) * float(weight)
            seen += float(weight)
    return total / seen


def train_forecaster(features="bank", epochs=20, batch_size=32, hidden=128,
                     lr=1e-3, seed=SEED, device=None, patience=3,
                     ckpt_path=None):
    """Fit the forecaster on TRAIN stays, early-stop on val forecast loss.

    Early stopping reads no labels (pure reconstruction loss), so the full val
    split is fair game here; labels only ever enter downstream via V_fit
    calibrators and V_freeze thresholds.
    """
    device = torch.device(device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    splits = load_splits()

    if features == "bank":
        bank = AgentBank.load(agent_names=config_agents("core-3+meas"))
        train_seqs, _ = bank_sequences(bank, splits["train"])
        val_seqs, _ = bank_sequences(bank, splits["val"])
        ckpt_path = ckpt_path or CKPT_BANK
    else:                                   # the 5 banded vitals (raw arm)
        train_seqs, _ = vitals_sequences(splits["train"])
        val_seqs, _ = vitals_sequences(splits["val"])
        ckpt_path = ckpt_path or CKPT_VITALS

    mean, std = _standardize_stats(train_seqs)
    _apply_stats(train_seqs, mean, std)
    _apply_stats(val_seqs, mean, std)

    n_features = train_seqs[0]["X"].shape[1]
    model = LatentForecaster(n_features, hidden=hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[latentcast] {features} forecaster: {n_features} features, "
          f"{n_params:,} params, {len(train_seqs):,} train stays, {device}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    best, best_state, stale = np.inf, None, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = _forecast_loss(model, train_seqs, batch_size, device,
                            rng=rng, opt=opt)
        va = _forecast_loss(model, val_seqs, batch_size, device)
        print(f"  epoch {epoch}: train {tr:.4f} | val {va:.4f} "
              f"({time.time() - t0:.0f}s)")
        if va < best:
            best, stale = va, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop (no val gain for {patience} epochs)")
                break
    if best_state is None:
        raise RuntimeError("forecaster never produced a finite val loss")
    model.load_state_dict(best_state)

    path = Path(ckpt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in
                               model.state_dict().items()},
                "features": features, "n_features": n_features,
                "hidden": hidden, "horizons": list(HORIZONS),
                "mean": mean, "std": std, "seed": seed,
                "best_val_loss": float(best)}, path)
    print(f"[latentcast] best val loss {best:.4f} -> {path}")
    return model, (mean, std)


def load_forecaster(path, device="cpu"):
    z = torch.load(path, map_location="cpu", weights_only=False)
    model = LatentForecaster(z["n_features"], hidden=z["hidden"],
                             horizons=tuple(z["horizons"])).to(device)
    model.load_state_dict(z["state_dict"])
    model.eval()
    return model, (z["mean"], z["std"]), z


# --- Forecast banks + the frozen head ---------------------------------------

@torch.no_grad()
def forecast_matrix(model, seqs, k, device, batch_size=64):
    """Stacked ``F_k`` rows for every sequence position, original row order."""
    from agents.vitals_tcn import make_batches
    model.eval()
    out = [None] * len(seqs)
    for idx in make_batches(seqs, batch_size):
        x, _, _ = collate_forecast(seqs, idx, model.horizons, device)
        preds = model(x)[k].cpu().numpy()
        for row, i in enumerate(idx):
            out[i] = preds[row, :len(seqs[i]["hours"])]
    return np.concatenate(out)


def build_forecast_bank(bank, df, model, stats, k, device,
                        risk_cols_from=None):
    """AgentBank whose row at ``(s, t)`` holds the predicted row for ``t+k``.

    Identical keys to the real bank's rows for ``df`` -- with_bank + the
    frozen head consume it with zero orchestrator changes. Risk columns are
    clamped to [0, 1] after inverse standardisation.
    """
    seqs, keys = bank_sequences(bank, df)
    mean, std = stats
    _apply_stats(seqs, mean, std)
    pred = forecast_matrix(model, seqs, k, device)
    pred = pred * std + mean
    n_risk = len(bank.agent_names)
    pred[:, -n_risk:] = np.clip(pred[:, -n_risk:], 0.0, 1.0)
    return AgentBank(keys, pred.astype(np.float32), bank.latent_dims,
                     bank.agent_names)


def run_eval(seeds=None, reps=2000, device=None, out_cache=EVAL_CACHE):
    """Score every (horizon, orchestrator seed) cell through the harness."""
    from explain.faithfulness import with_bank

    device = torch.device(device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    splits = load_splits()
    ctx = FusionContext(splits)
    val_df, test_df = splits["val"], splits["test"]
    bank = AgentBank.load(agent_names=config_agents("core-3+meas"))
    model, stats, meta_ck = load_forecaster(CKPT_BANK, device=device)

    fbanks = {}
    for k in HORIZONS:
        t0 = time.time()
        fbanks[k] = {"val": build_forecast_bank(bank, val_df, model, stats,
                                                k, device),
                     "test": build_forecast_bank(bank, test_df, model, stats,
                                                 k, device)}
        print(f"[latentcast] forecast banks k={k} ({time.time() - t0:.0f}s)")

    seeds = seeds or [42, 1, 2]
    results, payload = {}, {
        "stay_id_test": ctx.test_ordered["stay_id"].to_numpy(),
        "hour_test": ctx.test_ordered["hour"].to_numpy(),
    }
    for seed in seeds:
        orch = Orchestrator.load_checkpoint(ORCH_CKPT.format(seed=seed), bank)

        # k=0 sanity: checkpoint + real bank must reproduce the cached
        # emissions -- validates the whole substitution pipeline.
        cached = np.load(
            f"data/cache/embeddings/orchestrator-meas-s{seed}_val.npz")
        drift = float(np.abs(orch.emit(val_df).risk - cached["risk"]).max())
        if drift > 1e-5:
            raise RuntimeError(f"seed {seed}: reloaded checkpoint drifts from "
                               f"cached emissions (max |diff| {drift:.2e})")
        print(f"[seed {seed}] k=0 pass-through OK (drift {drift:.1e})")

        results[seed] = {}
        for k in HORIZONS:
            rv = with_bank(orch, fbanks[k]["val"]).raw_risk(val_df)
            rt = with_bank(orch, fbanks[k]["test"]).raw_risk(test_df)
            r = ctx.evaluate_rung(rv, rt, reps=reps)
            results[seed][k] = r
            payload[f"val_k{k}_s{seed}"] = rv.astype(np.float32)
            payload[f"test_k{k}_s{seed}"] = rt.astype(np.float32)
            leads = {b: r["budgets"][b]["median"] for b in r["budgets"]}
            print(f"[seed {seed}] k={k}: test AUPRC {r['auprc']:.4f}, "
                  f"median lead {leads}")

    payload["meta_json"] = np.array(json.dumps(_jsonable(
        {"horizons": list(HORIZONS), "seeds": seeds, "reps": reps,
         "forecaster_val_loss": meta_ck["best_val_loss"],
         "results": results})))
    Path(out_cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_cache, **payload)
    print(f"[cache] -> {out_cache}")
    return results


# --- Control 1: horizon-shifted-label GBM ------------------------------------

def shifted_label(frame, extra_hours):
    """Event within ``24 + extra_hours`` -- the label-widening control.

    Verified equivalent to the shipped ``label_24h`` at ``extra_hours=0``
    (bit-for-bit on all 2,460,983 cohort rows); the assert in
    :func:`run_shifted_gbm` keeps that guarantee alive.
    """
    ev = frame["event_h"].to_numpy(dtype=float)
    h = frame["hour"].to_numpy(dtype=float)
    return (~np.isnan(ev) & (ev > h) & (ev <= h + 24 + extra_hours)
            ).astype(float)


def run_shifted_gbm(reps=2000, device="cuda", out_cache=SHIFTED_CACHE):
    """One FAIR+TRAJ fit per horizon on the widened label, standard-label race.

    Hyperparameters are pinned to the recorded FAIR+TRAJ winner (no re-tuning
    grid); early stopping watches the WIDENED label on val -- the label shift
    is a training-time intervention only, the harness race never changes.
    """
    from baselines_fair import SCORE_CACHE, build_design, variant_columns
    from modality_features import load_labs

    splits = load_splits()
    ctx = FusionContext(splits)
    for frame in (splits["train"], splits["val"], splits["test"]):
        ordered = frame.sort_values(["stay_id", "hour"])
        if not (shifted_label(ordered, 0)
                == ordered[LABEL].to_numpy(dtype=float)).all():
            raise AssertionError("shifted_label(extra=0) != label_24h -- "
                                 "the event_h derivation no longer holds")

    meta = json.loads(str(np.load(SCORE_CACHE, allow_pickle=True)["meta_json"]))
    rec = meta["records"]["FAIR+TRAJ"]
    cols = variant_columns()["FAIR+TRAJ"]
    labs = load_labs()
    print("[shifted-gbm] building design matrices once...")
    X_tr, _, _ = build_design(splits["train"], labs, None)
    X_va, _, _ = build_design(splits["val"], labs, None)
    X_te, _, _ = build_design(splits["test"], labs, None)
    tr_ordered = splits["train"].sort_values(["stay_id", "hour"])
    va_ordered = splits["val"].sort_values(["stay_id", "hour"])

    import xgboost as xgb
    results, payload = {}, {
        "stay_id_test": ctx.test_ordered["stay_id"].to_numpy(),
        "hour_test": ctx.test_ordered["hour"].to_numpy(),
    }
    for k in HORIZONS:
        y_tr = shifted_label(tr_ordered, k)
        y_va = shifted_label(va_ordered, k)
        t0 = time.time()
        model = xgb.XGBClassifier(
            n_estimators=3000, learning_rate=0.05,
            max_depth=rec["max_depth"], min_child_weight=10,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=rec["scale_pos_weight"], random_state=SEED,
            n_jobs=-1, tree_method="hist", eval_metric="aucpr",
            early_stopping_rounds=50, device=device)
        model.fit(X_tr[cols], y_tr, eval_set=[(X_va[cols], y_va)],
                  verbose=False)
        rv = model.predict_proba(X_va[cols])[:, 1].astype(float)
        rt = model.predict_proba(X_te[cols])[:, 1].astype(float)
        r = ctx.evaluate_rung(rv, rt, reps=reps)
        results[k] = r
        payload[f"val_k{k}"] = rv.astype(np.float32)
        payload[f"test_k{k}"] = rt.astype(np.float32)
        print(f"[shifted-gbm] k={k}: {model.best_iteration + 1} trees, "
              f"test AUPRC {r['auprc']:.4f} ({time.time() - t0:.0f}s)")

    payload["meta_json"] = np.array(json.dumps(_jsonable(
        {"horizons": list(HORIZONS), "reps": reps, "record": rec,
         "results": results})))
    np.savez_compressed(out_cache, **payload)
    print(f"[cache] -> {out_cache}")
    return results


# --- Control 2: forecast raw vitals, recompute NEWS2 -------------------------

def vitals_sequences(df):
    """Per-stay sequences of the 5 banded vitals + missingness indicators."""
    ordered = df.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    vit = ordered[list(VITALS)].to_numpy(dtype=np.float32)
    isnan = np.isnan(vit).astype(np.float32)
    X = np.concatenate([np.nan_to_num(vit), isnan], axis=1)
    stay = ordered["stay_id"].to_numpy()
    bounds = np.flatnonzero(np.diff(stay)) + 1
    seqs = []
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(stay)]):
        seqs.append({"stay_id": stay[lo],
                     "hours": ordered["hour"].to_numpy()[lo:hi],
                     "X": X[lo:hi]})
    return seqs, ordered


def run_raw_news2(reps=2000, epochs=20, device=None,
                  out_cache=RAWNEWS2_CACHE):
    """Forecast vitals -> NEWS2-on-forecast -> the same harness."""
    device = torch.device(device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if not Path(CKPT_VITALS).exists():
        train_forecaster(features="vitals", epochs=epochs, device=device)
    model, stats, _ = load_forecaster(CKPT_VITALS, device=device)

    splits = load_splits()
    ctx = FusionContext(splits)
    n_vit = len(VITALS)
    results, payload = {}, {
        "stay_id_test": ctx.test_ordered["stay_id"].to_numpy(),
        "hour_test": ctx.test_ordered["hour"].to_numpy(),
    }
    for split_name, df in (("val", splits["val"]), ("test", splits["test"])):
        seqs, ordered = vitals_sequences(df)
        mean, std = stats
        _apply_stats(seqs, mean, std)
        for k in HORIZONS:
            pred = forecast_matrix(model, seqs, k, device) * std + mean
            frame = pd.DataFrame(pred[:, :n_vit], columns=list(VITALS))
            # consciousness/oxygen persist from t: the control forecasts the
            # banded vitals only, everything else is last-known.
            frame["is_alert"] = ordered["is_alert"].to_numpy()
            frame["on_oxygen"] = ordered["on_oxygen"].to_numpy()
            score, _, _ = compute_news2(frame)
            results.setdefault(k, {})[split_name] = score.astype(float)
    out = {}
    for k in HORIZONS:
        r = ctx.evaluate_rung(results[k]["val"], results[k]["test"],
                              reps=reps)
        out[k] = r
        payload[f"val_k{k}"] = results[k]["val"].astype(np.float32)
        payload[f"test_k{k}"] = results[k]["test"].astype(np.float32)
        print(f"[raw-news2] k={k}: test AUPRC {r['auprc']:.4f}, "
              f"lead@2% {r['budgets'][0.02]['median']:+.1f}")

    payload["meta_json"] = np.array(json.dumps(_jsonable(
        {"horizons": list(HORIZONS), "reps": reps, "results": out})))
    np.savez_compressed(out_cache, **payload)
    print(f"[cache] -> {out_cache}")
    return out


# --- Exceedance-probability ensemble (the lead-extension attack) -------------
#
# Plain LatentCast failed because the SmoothL1-optimal forecast is a
# conditional MEAN: regression-to-the-mean smooths exactly the sharp risk
# spikes that cross thresholds early. The exceedance design preserves spike
# mass twice over: max over horizons <= K ("will it cross at any point in the
# next K hours"), and a VOTE over independently-seeded forecasters (a spike
# any member predicts still moves the score). tau is the PARENT's own
# volume-matched alarm threshold, so "exceedance" means "predicted to alarm".

def within_horizon_max(hats_by_k, ks):
    """Elementwise max of the per-horizon forecast risks for horizons <= K."""
    return np.max(np.stack([hats_by_k[k] for k in ks]), axis=0)


def exceedance_score(within_members, tau):
    """Vote fraction over ensemble members + a sub-vote-step tiebreak.

    Votes live on a lattice of 1/E steps; the mean-forecast tiebreak (1e-6
    scale, far below the 1/E step) makes the score strictly ordered inside
    each vote level so quantile volume matching behaves (the isotonic-plateau
    lesson from fusion_ladder applied here by construction).

    Returns ``(score, votes, mean_within)`` so callers can compose without
    un-mixing the tiebreak.
    """
    stack = np.stack(within_members)
    votes = (stack >= tau).mean(axis=0)
    mean_within = np.clip(stack.mean(axis=0), 0.0, 1.0)
    return votes + 1e-6 * mean_within, votes, mean_within


def or_score(votes, mean_within, parent_raw, tau):
    """Alarm now OR predicted-to-alarm: never later than the parent.

    The parent's own crossing counts as a full vote (1.0, the lattice
    maximum), so at any threshold inside the vote range every parent alarm
    fires and exceedance votes can only ADD earlier ones.
    """
    crossed = (parent_raw >= tau).astype(float)
    tie = 1e-6 * np.clip(np.maximum(mean_within, parent_raw), 0.0, 1.0)
    return np.maximum(votes, crossed) + tie


def run_exceed(orch_seeds=(42, 1, 2), windows=(2, 4, 6, 8), reps=2000,
               device=None, out_cache=EXCEED_CACHE):
    """Score EXCEED-K and PARENT∨EXCEED-K through the paired harness."""
    from explain.faithfulness import with_bank
    from fusion_common import load_orch_seeds

    device = torch.device(device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    splits = load_splits()
    ctx = FusionContext(splits)
    bank = AgentBank.load(agent_names=config_agents("core-3+meas"))
    _, parent, _ = load_orch_seeds(ctx.val_ordered, ctx.test_ordered)
    ks_needed = sorted({k for K in windows for k in HORIZONS if k <= K})

    missing = [e for e in ENSEMBLE_SEEDS if not Path(_ens_ckpt(e)).exists()]
    if missing:
        raise FileNotFoundError(
            f"train the forecaster ensemble first (missing seeds {missing}): "
            f"python src/latentcast.py train --seed <s>")

    orchs = {s: Orchestrator.load_checkpoint(ORCH_CKPT.format(seed=s), bank)
             for s in orch_seeds}
    hats = {s: {e: {} for e in ENSEMBLE_SEEDS} for s in orch_seeds}
    for e in ENSEMBLE_SEEDS:
        model, stats, _ = load_forecaster(_ens_ckpt(e), device=device)
        for k in ks_needed:
            t0 = time.time()
            fb = {name: build_forecast_bank(bank, df, model, stats, k, device)
                  for name, df in (("val", splits["val"]),
                                   ("test", splits["test"]))}
            for s in orch_seeds:
                hats[s][e][k] = {
                    name: with_bank(orchs[s], fb[name]).raw_risk(df)
                    for name, df in (("val", splits["val"]),
                                     ("test", splits["test"]))}
            print(f"[exceed] member {e} k={k} scored through "
                  f"{len(orch_seeds)} frozen heads "
                  f"({time.time() - t0:.0f}s)", flush=True)
        del model

    nf = ctx.news2_val[ctx.freeze_mask]
    af = ctx.alertable_val[ctx.freeze_mask]
    results, payload = {}, {
        "stay_id_test": ctx.test_ordered["stay_id"].to_numpy(),
        "hour_test": ctx.test_ordered["hour"].to_numpy(),
    }
    for s in orch_seeds:
        results[s] = {}
        for K in windows:
            ks = [k for k in HORIZONS if k <= K]
            within = {name: [within_horizon_max(
                {k: hats[s][e][k][name] for k in ks}, ks)
                for e in ENSEMBLE_SEEDS] for name in ("val", "test")}
            mean_within_test = np.mean(np.stack(within["test"]), axis=0)
            row = {"auprc_mean_within": float(auprc(ctx.y_test,
                                                    mean_within_test)),
                   "budgets": {}}
            for budget in (0.01, 0.02, 0.05):
                _, tau, _ = frozen_thresholds(
                    nf, parent[s]["val"][ctx.freeze_mask], af, budget)
                sc = {name: exceedance_score(within[name], tau)
                      for name in ("val", "test")}
                orr = {name: or_score(sc[name][1], sc[name][2],
                                      parent[s][name], tau)
                       for name in ("val", "test")}
                cell = {}
                for rung, sv, st in (
                        ("EXCEED", sc["val"][0], sc["test"][0]),
                        ("OR", orr["val"], orr["test"])):
                    thr_n2, thr_m, _ = frozen_thresholds(
                        nf, sv[ctx.freeze_mask], af, budget)
                    diffs = paired_first_alarm_diffs(
                        ctx.test_ordered, ctx.news2_test, st, thr_n2, thr_m)
                    r = summarize_paired(diffs, ctx.stay_to_patient,
                                         reps=reps)
                    r["model_rate"] = float(
                        (st[ctx.alertable_test] >= thr_m).mean())
                    cell[rung] = r
                row["budgets"][budget] = cell
            results[s][K] = row
            b2 = row["budgets"][0.02]
            print(f"[seed {s}] K<={K}: AUPRC(mean-within) "
                  f"{row['auprc_mean_within']:.4f} | @2% EXCEED lead "
                  f"{b2['EXCEED']['median']:+.1f} (win "
                  f"{100*b2['EXCEED']['win_rate']:.0f}%, n "
                  f"{b2['EXCEED']['n_pair']}) | OR lead "
                  f"{b2['OR']['median']:+.1f} (win "
                  f"{100*b2['OR']['win_rate']:.0f}%, n {b2['OR']['n_pair']})",
                  flush=True)

    payload["meta_json"] = np.array(json.dumps(_jsonable(
        {"orch_seeds": list(orch_seeds), "windows": list(windows),
         "ensemble_seeds": list(ENSEMBLE_SEEDS), "reps": reps,
         "results": results})))
    Path(out_cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_cache, **payload)
    print(f"[cache] -> {out_cache}")
    return results


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("train", "eval", "control-shifted-gbm",
                                     "control-raw-news2", "exceed"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seeds", default="42,1,2",
                    help="orchestrator (frozen-head) seeds")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="train: forecaster seed (ensemble members)")
    ap.add_argument("--windows", default="2,4,6,8",
                    help="exceed: within-K horizon windows")
    ap.add_argument("--device", default=None,
                    help="cuda/cpu (default: cuda if available)")
    ap.add_argument("--out", default=None,
                    help="exceed: output npz override (parallel per-seed runs)")
    args = ap.parse_args()
    if args.mode == "train":
        train_forecaster(features="bank", epochs=args.epochs,
                         device=args.device, seed=args.seed,
                         ckpt_path=_ens_ckpt(args.seed))
    elif args.mode == "eval":
        run_eval(seeds=[int(s) for s in args.seeds.split(",")],
                 reps=args.reps, device=args.device)
    elif args.mode == "control-shifted-gbm":
        run_shifted_gbm(reps=args.reps, device=args.device or "cuda")
    elif args.mode == "control-raw-news2":
        run_raw_news2(reps=args.reps, epochs=args.epochs, device=args.device)
    else:
        run_exceed(orch_seeds=[int(s) for s in args.seeds.split(",")],
                   windows=tuple(int(k) for k in args.windows.split(",")),
                   reps=args.reps, device=args.device,
                   out_cache=args.out or EXCEED_CACHE)
