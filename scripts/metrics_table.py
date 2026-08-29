"""Both discrimination metrics for every model the program cached.

The results tables were built around AUPRC (the right headline at 4.7%
prevalence, where AUROC is optimistic and compresses differences). AUROC is
reported alongside it because it is what most of the EWS literature quotes,
so a reader comparing Sentinel to published numbers needs both.

Point estimates for every cached model; patient-clustered bootstrap CIs on
both metrics for the headline set (parents + flagships), reusing
``baselines_fair.clustered_bootstrap``.

    python scripts/metrics_table.py [--reps 1000] [--json out.json]
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

CACHE = REPO / "data" / "cache"
SEEDS = (42, 1, 2, 3, 4)  # flagship chain default; a cache's own meta_json seed list wins

# (label, npz, key-template) -- {s} expands over SEEDS for multi-seed rows.
SPEC = [
    ("NEWS2", "fair_gbm_scores", "raw_NEWS2"),
    ("GBM13 (vitals-only)", "fair_gbm_scores", "raw_GBM13"),
    ("FAIR (61 feat)", "fair_gbm_scores", "raw_FAIR"),
    ("FAIR+TRAJ (145 feat)", "fair_gbm_scores", "raw_FAIR+TRAJ"),
    ("ORCH core-3 (published)", "fair_gbm_scores", "raw_ORCH"),
    ("ORCH core-3+meas", "orch_meas_seeds", "raw_s{s}_test"),
    ("MAX", "fusion_ladder_scores", "test_MAX_s{s}"),
    ("LOGIT3", "fusion_ladder_scores", "test_LOGIT3_s{s}"),
    ("CONSTRAINED", "fusion_ladder_scores", "test_CONSTRAINED_s{s}"),
    ("CONSTR-1H", "fusion_ladder_scores", "test_CONSTR_1H_s{s}"),
    ("GATED", "fusion_ladder_scores", "test_GATED_s{s}"),
    ("ORACLE-GATE (bound)", "fusion_ladder_scores", "test_ORACLE_GATE_s{s}"),
    ("GBM x TLS H=6", "distill_grid", "test_gbm_tls_6_sNone"),
    ("GBM-distill lam=0.75", "distill_grid", "test_gbm_distill_0p75_s42"),
    ("ORCH-distill lam=0.5", "distill_grid", "test_orch_distill_0p5_s42"),
    ("ORCH-distill lam=0.75", "distill_grid", "test_orch_distill_0p75_s1"),
    ("LatentCast k=2", "latentcast_scores", "test_k2_s{s}"),
    ("LatentCast k=4", "latentcast_scores", "test_k4_s{s}"),
    ("LatentCast k=6", "latentcast_scores", "test_k6_s{s}"),
    ("LatentCast k=8", "latentcast_scores", "test_k8_s{s}"),
    ("LatentCast k=12", "latentcast_scores", "test_k12_s{s}"),
    ("shifted-GBM k=2", "latentcast_shifted_gbm", "test_k2"),
    ("shifted-GBM k=12", "latentcast_shifted_gbm", "test_k12"),
    ("raw-NEWS2 forecast k=2", "latentcast_rawnews2", "test_k2"),
    ("raw-NEWS2 forecast k=12", "latentcast_rawnews2", "test_k12"),
    ("ORCH+GBM (5th agent)", "orch_gbm_agent", "test_s{s}"),
    ("GBM +RISKS", "gbm_agents_scores_main", "test_+RISKS"),
    ("GBM +RISKTRAJ", "gbm_agents_scores_main", "test_+RISKTRAJ"),
    ("GBM +LATENT", "gbm_agents_scores_main", "test_+LATENT"),
    ("GBM +ORCH (OOF)", "gbm_agents_scores_main", "test_+ORCH_s{s}"),
    ("GBM +ORCHONLY (clean)", "gbm_agents_scores", "test_+ORCHONLY_s{s}"),
    ("meas+cadence", "new_agents_fusion", "test_meas+cadence_s{s}"),
    ("meas+fluid", "new_agents_fusion", "test_meas+fluid_s{s}"),
    ("meas+both", "new_agents_fusion", "test_meas+both_s{s}"),
    ("gbm+cadence", "new_agents_fusion", "test_gbm+cadence_s{s}"),
    ("gbm+fluid", "new_agents_fusion", "test_gbm+fluid_s{s}"),
    ("ALL-7 (gbm+both)", "new_agents_fusion", "test_gbm+both_s{s}"),
    ("ALL-8 (all7+resp)", "resp_fusion", "test_all7+resp_s{s}"),
    ("ALL-8 cap-at-HFNC", "resp_fusion", "test_all7+respcap4_s{s}"),
    ("TWO-HEAD ALL-8", "two_head", "test_s{s}"),
    ("  |- D-head (death only)", "two_head", "dhead_test_s{s}"),
    # Literature-anchored GBMs (T10) -- specifications taken from published
    # papers rather than chosen here; see docs/references.md.
    ("lit: Sadanandan XGB", "lit_gbm_scores", "test_sadanandan"),
    ("lit: YAIB LGBM", "lit_gbm_scores", "test_yaib"),
    ("lit: eCART-style GBM", "lit_gbm_scores", "test_ecart"),
]

# Standalone agents come from their emission caches.
AGENT_ROWS = [("cadence agent", "cadence"), ("fluid agent", "fluid"),
              ("resp agent", "resp"), ("gbm agent", "gbm"),
              ("vitals_tcn agent", "vitals_tcn"), ("labs agent", "labs"),
              ("ecg_meas agent", "ecg_meas")]

# Rows that get patient-clustered CIs on both metrics.
HEADLINE = {"NEWS2", "FAIR+TRAJ (145 feat)", "ORCH core-3+meas", "LOGIT3",
            "CONSTR-1H", "ORCH+GBM (5th agent)", "ALL-7 (gbm+both)",
            "ALL-8 (all7+resp)", "TWO-HEAD ALL-8", "  |- D-head (death only)",
            "lit: Sadanandan XGB", "lit: YAIB LGBM", "lit: eCART-style GBM"}


def cache_seeds(z):
    """Seeds a cache carries: its own meta_json list when it has one (the
    two-head flagship carries ten since T18), else the flagship-chain default."""
    if "meta_json" in z.files:
        try:
            seeds = json.loads(str(z["meta_json"])).get("seeds")
            if seeds:
                return tuple(int(s) for s in seeds)
        except (ValueError, TypeError):
            pass
    return SEEDS


def load_scores(name, key):
    z = np.load(CACHE / f"{name}.npz", allow_pickle=True)
    if "{s}" not in key:
        return {None: z[key].astype(float)}
    return {s: z[key.format(s=s)].astype(float) for s in cache_seeds(z)
            if key.format(s=s) in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--json", default=str(CACHE / "metrics_table.json"))
    args = ap.parse_args()

    from agents.base import AgentOutput
    from baselines_fair import bootstrap_indices
    from baselines_mimic import load_splits
    from metrics import auprc, auroc

    splits = load_splits()
    test = splits["test"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    y = test["label_24h"].to_numpy(dtype=float)
    y_death = test["label_24h_death"].to_numpy(dtype=float)
    patients = test["patient_id"].to_numpy()

    rows, headline_scores = [], {}
    for label, npz, key in SPEC:
        try:
            per_seed = load_scores(npz, key)
        except (FileNotFoundError, KeyError) as exc:
            print(f"[skip] {label}: {exc}")
            continue
        target = y_death if "D-head" in label else y
        pr = [auprc(target, s) for s in per_seed.values()]
        ro = [auroc(target, s) for s in per_seed.values()]
        rows.append({"model": label, "n_seeds": len(per_seed),
                     "auprc": float(np.mean(pr)), "auroc": float(np.mean(ro)),
                     "auprc_seeds": [float(v) for v in pr],
                     "auroc_seeds": [float(v) for v in ro],
                     "target": "death" if "D-head" in label else "composite"})
        if label in HEADLINE:
            headline_scores[label] = list(per_seed.values())[0]

    for label, agent in AGENT_ROWS:
        path = CACHE / "embeddings" / f"{agent}_test.npz"
        if not path.exists():
            continue
        s = AgentOutput.load(path).risk.astype(float)
        rows.append({"model": label, "n_seeds": 1,
                     "auprc": float(auprc(y, s)), "auroc": float(auroc(y, s)),
                     "auprc_seeds": [float(auprc(y, s))],
                     "auroc_seeds": [float(auroc(y, s))],
                     "target": "composite"})

    # Patient-clustered CIs for the headline set (seed 42 / single run each).
    print(f"[bootstrap] {args.reps} patient resamples on "
          f"{len(headline_scores)} headline models...")
    draws = {k: {"auprc": [], "auroc": []} for k in headline_scores}
    for idx in bootstrap_indices(patients, args.reps):
        yb, ydb = y[idx], y_death[idx]
        if yb.sum() == 0:
            continue
        for k, s in headline_scores.items():
            t = ydb if "D-head" in k else yb
            draws[k]["auprc"].append(auprc(t, s[idx]))
            draws[k]["auroc"].append(auroc(t, s[idx]))
    cis = {k: {m: [float(np.percentile(v[m], 2.5)),
                   float(np.percentile(v[m], 97.5))]
               for m in ("auprc", "auroc")} for k, v in draws.items()}
    for r in rows:
        if r["model"] in cis:
            r["ci"] = cis[r["model"]]

    hdr = f"  {'model':<26}{'seeds':>6}{'AUPRC':>8}{'AUROC':>8}   95% CI (AUPRC / AUROC)"
    print("\nDISCRIMINATION -- test split, 361,229 hours, prevalence 4.74%")
    print(hdr)
    print("  " + "-" * (len(hdr) + 12))
    for r in rows:
        ci = ""
        if "ci" in r:
            a, u = r["ci"]["auprc"], r["ci"]["auroc"]
            ci = (f"   [{a[0]:.3f}, {a[1]:.3f}] / [{u[0]:.3f}, {u[1]:.3f}]")
        star = " *" if r["target"] == "death" else ""
        print(f"  {r['model']:<26}{r['n_seeds']:>6}{r['auprc']:>8.4f}"
              f"{r['auroc']:>8.4f}{ci}{star}")
    print("  * scored against label_24h_death, not the composite")

    Path(args.json).write_text(json.dumps(rows, indent=1))
    print(f"\n[cache] -> {args.json}")


if __name__ == "__main__":
    main()
