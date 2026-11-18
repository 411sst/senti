"""T40 reuse mode: the FAIR+TRAJ stage with the committed tuned settings.

The rebuild's FAIR+TRAJ stage (``src/baselines_fair.py --variants FAIR+TRAJ``)
fits a six-cell grid (depth {4, 6, 8} x class weight {1, rebalanced}) and keeps
the best cell on validation AUPRC. In T36 that grid was 19.3 of 30.5 minutes.
The GBM agent downstream only needs the winning cell's settings and scores.

This runs the same stage with the grid replaced by the one cell the published
model selected (``results_fusion.md`` l.189: depth 4, class weight 1.0). The
cell is fitted exactly as ``tune_gbm`` fits it (same XGBoost parameters, same
early stopping on validation AUPRC), and everything else in
``baselines_fair.run`` is unchanged: the GBM13 row, the cached-score format
and the 10-resample bootstrap. ``src/baselines_fair.py`` is not edited; its
tuner is swapped in this process only.

Run ONLY inside an isolated clone, as a stage of ``scripts/t40_rebuild.py``.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

COMMITTED = {"max_depth": 4, "scale_pos_weight": 1.0}   # results_fusion.md l.189


def fixed_cell(max_depth, scale_pos_weight):
    """A drop-in for ``baselines_fair.tune_gbm`` that fits one grid cell."""
    def tune(X_tr, y_tr, X_va, y_va, seed=None, quick=False, verbose=True,
             device=None):
        import xgboost as xgb

        from config import SEED
        from metrics import auprc

        extra = {"device": device} if device else {}
        model = xgb.XGBClassifier(
            n_estimators=3000, learning_rate=0.05, max_depth=max_depth,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=SEED if seed is None else seed, n_jobs=-1,
            tree_method="hist", eval_metric="aucpr",
            early_stopping_rounds=50, **extra,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        score = auprc(y_va, model.predict_proba(X_va)[:, 1])
        if verbose:
            print(f"    depth={max_depth} spw={scale_pos_weight:>6.1f}: "
                  f"val AUPRC {score:.4f} ({model.best_iteration + 1} trees) "
                  f"[committed setting, no grid]")
        return model, {"max_depth": max_depth,
                       "scale_pos_weight": round(scale_pos_weight, 2),
                       "n_trees": int(model.best_iteration + 1),
                       "val_auprc": float(score),
                       "grid": "skipped: committed winner reused"}
    return tune


def main():
    os.chdir(REPO)
    from h1_rebuild import require_isolated
    require_isolated()
    import baselines_fair
    baselines_fair.tune_gbm = fixed_cell(**COMMITTED)
    baselines_fair.run(reps=10, variants=("FAIR+TRAJ",), device=None)


if __name__ == "__main__":
    main()
