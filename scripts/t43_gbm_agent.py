"""T43 preparation: the all-features GBM as Sentinel's GBM agent.

Builds the agent's emissions with the shipped GBM agent's construction
(``scripts/orch_gbm_agent.py``), swapping FAIR+TRAJ (145 columns) for the T30
all-features GBM (262 columns):

* train rows: out of fold, two patient-level halves drawn exactly as the
  shipped agent draws them (``gbm_agent_oof.patient_halves``). Each fold model
  uses T30's winning settings (depth 6, no class weight) with the tuner's
  fixed ones (lr 0.05, min_child_weight 10, subsample and colsample 0.8, up to
  3,000 trees, early stopping after 50 rounds on validation AUPRC), seed 42,
  GPU ``hist``.
* validation and test rows: T30's seed-42 fit on all of train, read from
  ``allfeat_gbm_scores.npz`` (the fit T31 reproduced exactly).
* latent = log-odds of the raw probability; risk = isotonic calibration fit on
  V_fit. The shipped agent fits its calibrator on the full validation split;
  T43 follows the paper-test rule (calibrate on V_fit), a stated difference.

Outputs, all new paths:
* ``data/cache/embeddings_t43/``: ``gbm_{train,val,test}.npz`` plus links to
  the seven unchanged agents' shipped emissions, so ``AgentBank.load`` reads the
  T43 bank from one directory and the shipped ``embeddings/`` is never written.
* ``models/t43/gbm_agent_fold{0,1}.ubj``: the two fold boosters.
* ``data/cache/t43_gbm_agent.json``: aggregates only.

No test-split metric is computed here.

    python scripts/t43_gbm_agent.py agent [--device cuda]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

SHIPPED_EMB = REPO / "data" / "cache" / "embeddings"
T43_EMB = REPO / "data" / "cache" / "embeddings_t43"
MODELS = REPO / "models" / "t43"
ALLFEAT_CACHE = REPO / "data" / "cache" / "allfeat_gbm_scores.npz"
OUT_JSON = REPO / "data" / "cache" / "t43_gbm_agent.json"
OTHER_AGENTS = ("vitals_tcn", "news2", "labs", "ecg_meas", "cadence", "fluid",
                "resp")
SPLITS = ("train", "val", "test")


def _keys_equal(a_stay, a_hour, frame):
    return (np.array_equal(np.asarray(a_stay), frame["stay_id"].to_numpy())
            and np.array_equal(np.asarray(a_hour), frame["hour"].to_numpy()))


def stage_agent(device):
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression

    from agents.base import AgentOutput
    from allfeat_gbm import build_allfeat_design, column_sets, load_sources
    from baselines_mimic import load_splits
    from fusion_common import FusionContext
    from gbm_agent_oof import logit, oof_scores, patient_halves
    from metrics import auprc, auroc

    t0 = time.time()
    z = np.load(ALLFEAT_CACHE, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    rec = meta["records"]["allfeat"]
    if (rec["max_depth"], rec["scale_pos_weight"]) != (6, 1.0):
        raise SystemExit(f"unexpected T30 winner {rec}")

    splits = load_splits()
    ctx = FusionContext(splits)
    train_o = splits["train"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    val_o, test_o = ctx.val_ordered, ctx.test_ordered
    if not (_keys_equal(z["val_stay_id"], z["val_hour"], val_o)
            and _keys_equal(z["test_stay_id"], z["test_hour"], test_o)):
        raise SystemExit("T30 score cache rows do not match the cohort order")
    y_val = val_o["label_24h"].to_numpy(dtype=float)
    raw = {"val": z["rawval_allfeat_s42"].astype(float),
           "test": z["rawtest_allfeat_s42"].astype(float)}
    check = auprc(y_val, raw["val"])
    if abs(check - rec["val_auprc"]) > 1e-9:
        raise SystemExit(f"cached val AUPRC {check} != recorded {rec['val_auprc']}")

    print("[t43] building the 262-column design for train and val...", flush=True)
    sources = load_sources()
    cols = column_sets(sources["ecg"]["names"])["allfeat"]
    X_tr, y_tr, keys_tr = build_allfeat_design(splits["train"], sources)
    X_va, y_va, keys_va = build_allfeat_design(splits["val"], sources)
    if not (_keys_equal(keys_tr["stay_id"], keys_tr["hour"], train_o)
            and _keys_equal(keys_va["stay_id"], keys_va["hour"], val_o)):
        raise SystemExit("design rows do not match the cohort order")
    y_tr = np.asarray(y_tr, dtype=float)
    if not np.array_equal(y_tr, train_o["label_24h"].to_numpy(dtype=float)):
        raise SystemExit("design label is not the cohort's label_24h")
    X_tr, X_va = X_tr[cols], X_va[cols]
    d_val = xgb.DMatrix(X_va, label=y_val)
    params = {"objective": "binary:logistic", "eval_metric": "aucpr",
              "eta": 0.05, "max_depth": rec["max_depth"],
              "min_child_weight": 10, "subsample": 0.8,
              "colsample_bytree": 0.8,
              "scale_pos_weight": rec["scale_pos_weight"],
              "seed": 42, "tree_method": "hist", "device": device}
    fold_info = []

    def fit(Xr, yr):
        t = time.time()
        b = xgb.train(params, xgb.DMatrix(Xr, label=yr), num_boost_round=3000,
                      evals=[(d_val, "val")], early_stopping_rounds=50,
                      verbose_eval=False)
        fold_info.append({"rows": int(len(yr)), "trees": int(b.best_iteration + 1),
                          "val_auprc_es": float(b.best_score),
                          "seconds": round(time.time() - t, 1)})
        print(f"[t43] fold fit: {len(yr):,} rows, {b.best_iteration + 1} trees, "
              f"val AUPRC {b.best_score:.4f} ({time.time() - t:.0f}s)", flush=True)
        return b

    def predict(b, Xr):
        return b.predict(xgb.DMatrix(Xr),
                         iteration_range=(0, b.best_iteration + 1)).astype(float)

    pid = train_o["patient_id"].to_numpy()
    raw["train"], boosters = oof_scores(X_tr, y_tr, pid, fit, predict)
    halves = patient_halves(pid)

    iso = IsotonicRegression(out_of_bounds="clip").fit(
        raw["val"][ctx.fit_mask], y_val[ctx.fit_mask])
    T43_EMB.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    frames = {"train": train_o, "val": val_o, "test": test_o}
    for split in SPLITS:
        out_path = T43_EMB / f"gbm_{split}.npz"
        if out_path.exists():
            raise SystemExit(f"{out_path} exists; new outputs go to new files")
        AgentOutput(keys=frames[split][["stay_id", "hour"]].copy(),
                    latent=logit(raw[split]).astype(np.float32)[:, None],
                    risk=iso.transform(raw[split]).astype(np.float32)
                    ).save(out_path)
    for name in OTHER_AGENTS:
        for split in SPLITS:
            link = T43_EMB / f"{name}_{split}.npz"
            if not link.exists():
                link.symlink_to(SHIPPED_EMB / f"{name}_{split}.npz")
    for k, b in enumerate(boosters):
        b.save_model(MODELS / f"gbm_agent_fold{k}.ubj")

    shipped_tr = np.load(SHIPPED_EMB / "gbm_train.npz")
    y_shipped = train_o["label_24h"].to_numpy(dtype=float)
    summary = {
        "stage": "agent",
        "run_seconds": round(time.time() - t0, 1),
        "columns": len(cols),
        "columns_sha256": hashlib.sha256("|".join(cols).encode()).hexdigest(),
        "t30_winner": rec,
        "folds": fold_info,
        "fold_patients": [len(h) for h in halves],
        "train_rows": int(len(train_o)),
        "train_oof_auprc": float(auprc(y_tr, raw["train"])),
        "train_oof_auroc": float(auroc(y_tr, raw["train"])),
        "shipped_agent_train_oof_auprc": float(
            auprc(y_shipped, shipped_tr["risk"].astype(float))),
        "val_auprc_cached_fit": float(check),
        "calibrator": "isotonic on V_fit",
        "xgboost": xgb.__version__,
        "device": device,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["agent"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.stage == "agent":
        stage_agent(args.device)


if __name__ == "__main__":
    main()
