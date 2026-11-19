"""T44-T49 preparation: save the GBM agent's two out-of-fold boosters.

The shipped GBM agent (``scripts/orch_gbm_agent.py``) scores TRAINING rows out
of fold: two patient halves (fold rng 42), each scored by a FAIR+TRAJ booster
fitted on the other half and early-stopped on validation. The boosters were
never saved. Heads trained on new training rows (T45, T46, T49) need the same
out-of-fold scores on rows our cohort leaves out, so this refits the two
boosters with the unchanged code and saves them.

Gate: on the cohort's training rows the refit boosters must reproduce the
shipped ``data/cache/embeddings/gbm_train.npz`` latents exactly; otherwise the
script stops and saves nothing.

    OMP_NUM_THREADS=1 python scripts/t4x_gbm_agent_folds.py

Writes ``models/t4x/gbm_agent_fold{0,1}.ubj`` and
``data/cache/t4x/gbm_agent_folds.json`` (aggregates). Never overwrites.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

MODELS = REPO / "models" / "t4x"
OUT = REPO / "data" / "cache" / "t4x" / "gbm_agent_folds.json"
SHIPPED = REPO / "data" / "cache" / "embeddings" / "gbm_train.npz"


def main():
    from agents.base import AgentOutput
    from baselines_fair import build_design, variant_columns
    from baselines_mimic import load_splits
    from distill_lead import fit_gbm_student, gbm_predict
    from gbm_agent_oof import logit, patient_halves
    from modality_features import load_labs

    paths = [MODELS / f"gbm_agent_fold{i}.ubj" for i in (0, 1)]
    if any(p.exists() for p in paths) or OUT.exists():
        raise SystemExit("refusing to overwrite existing fold boosters or report")
    t0 = time.time()
    splits = load_splits()
    labs = load_labs()
    cols = variant_columns()["FAIR+TRAJ"]
    train = splits["train"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    val = splits["val"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    X_tr = build_design(splits["train"], labs, None)[0][cols]
    X_va = build_design(splits["val"], labs, None)[0][cols]
    y_tr = train["label_24h"].to_numpy(dtype=float)
    y_va = val["label_24h"].to_numpy(dtype=float)
    pid = train["patient_id"].to_numpy()

    oof = np.full(len(train), np.nan)
    boosters, folds = [], []
    for i, held in enumerate(patient_halves(pid)):
        mask = np.isin(pid, list(held))
        b = fit_gbm_student(X_tr[~mask], y_tr[~mask], np.ones(int((~mask).sum())),
                            X_va, y_va, device="cuda")
        oof[mask] = gbm_predict(b, X_tr[mask])
        boosters.append(b)
        folds.append({"fold": i, "held_out_patients": len(held),
                      "held_out_rows": int(mask.sum()),
                      "best_iteration": int(b.best_iteration)})
    shipped = AgentOutput.load(SHIPPED)
    keys_ok = bool((shipped.keys["stay_id"].to_numpy() == train["stay_id"].to_numpy()).all()
                   and (shipped.keys["hour"].to_numpy() == train["hour"].to_numpy()).all())
    mine = logit(oof).astype(np.float32)
    max_diff = float(np.abs(mine - shipped.latent[:, 0]).max())
    exact = bool(keys_ok and np.array_equal(mine, shipped.latent[:, 0]))
    report = {"folds": folds, "columns": len(cols), "train_rows": int(len(train)),
              "keys_match_shipped": keys_ok, "latent_max_abs_diff": max_diff,
              "reproduces_shipped_exactly": exact, "seconds": round(time.time() - t0, 1)}
    print(json.dumps(report, indent=2))
    if not exact:
        raise SystemExit("GATE: refit folds do not reproduce the shipped GBM agent; nothing saved")

    MODELS.mkdir(parents=True, exist_ok=True)
    for b, p in zip(boosters, paths):
        b.save_model(p)
    # reloaded boosters must give the same scores (score them on the GPU)
    import xgboost as xgb
    for i, (held, p) in enumerate(zip(patient_halves(pid), paths)):
        mask = np.isin(pid, list(held))
        b = xgb.Booster()
        b.load_model(p)
        b.set_param({"device": "cuda"})   # fitted on the GPU; CPU predictions differ by ~1e-5
        if not np.array_equal(gbm_predict(b, X_tr[mask]), oof[mask]):
            raise SystemExit(f"GATE: reloaded fold {i} differs")
    report["reloaded_identical"] = True
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"[done] {paths[0].name}, {paths[1].name} -> {MODELS.relative_to(REPO)}")


if __name__ == "__main__":
    main()
