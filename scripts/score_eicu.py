"""Zero-shot external validation: score a MIMIC-trained stack on eICU-CRD.

    python scripts/score_eicu.py [--agents vitals_tcn news2 labs] [--epochs 8]

Nothing here is trained, calibrated or thresholded on eICU. Agents are fit on
MIMIC train and isotonic-calibrated on MIMIC val; the fusion is fit the same
way; eICU is then scored through the frozen result. eICU is a pure held-out
set in the strongest sense available -- a different set of 208 hospitals.

WHY THIS SCRIPT RE-FITS RATHER THAN LOADING. Only the orchestrator checkpoints
are persisted as weights; the per-modality agents cache *emissions*, not
models. Re-fitting on MIMIC with the audited seed reproduces them, and the
reproduction is checked against the shipped caches before any eICU number is
believed (--check-repro, on by default).

WHY IT WRITES TO ITS OWN EMBEDDING DIRECTORY. The shipped caches under
data/cache/embeddings/ are what the entire results corpus is paired against.
This script never writes there; it uses data/cache/embeddings_eicu/ so a
transfer experiment can never silently move a published number.

REPORTING NOTE. AUPRC is prevalence-dependent and the two cohorts differ
(4.74% vs 3.38% positive hours), so raw AUPRC is not comparable across them.
Every table below reports lift = AUPRC / prevalence alongside it, and AUROC,
which is prevalence-free.
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from baselines_mimic import load_splits  # noqa: E402
from metrics import auprc, auroc, ece  # noqa: E402

LABEL = "label_24h"
VITALS = ["heart_rate", "resp_rate", "spo2", "sbp", "temperature"]
EICU_COHORT = REPO / "data" / "cache" / "eicu_cohort_hourly.parquet"
EICU_LABS = REPO / "data" / "cache" / "eicu_labs_hourly.parquet"
SHIPPED_EMB = REPO / "data" / "cache" / "embeddings"
EMB_DIR = REPO / "data" / "cache" / "embeddings_eicu"


def build_agent(name, epochs):
    """Construct one untrained agent by name."""
    if name == "vitals_tcn":
        from agents.vitals_tcn import VitalsTCNAgent
        return VitalsTCNAgent(max_epochs=epochs)
    if name == "news2":
        from agents.news2_agent import News2Agent
        return News2Agent()
    if name == "labs":
        from agents.labs_agent import LabsAgent
        return LabsAgent(max_epochs=epochs)
    raise KeyError(f"agent {name!r} has no eICU feature mapping yet")


def point_at_eicu(name, agent):
    """Repoint an agent's auxiliary feature source at its eICU twin.

    Only the DATA source moves. Weights, normalisation statistics and the
    isotonic calibrator all stay as fitted on MIMIC -- repointing those would
    turn a transfer test into a refit.
    """
    if name == "labs":
        from modality_features import load_labs
        agent.labs = load_labs(EICU_LABS)


def restore_mimic(name, agent):
    if name == "labs":
        from modality_features import load_labs
        agent.labs = load_labs()


def row(tag, y, r, extra=""):
    p = float(y.mean())
    a = auprc(y, r)
    print(f"  {tag:<30} n={len(y):>9,}  prev {100*p:5.2f}%  "
          f"AUPRC {a:.4f}  lift {a/p:5.2f}  AUROC {auroc(y, r):.4f}  "
          f"ECE {ece(y, r):.4f}{extra}")
    return dict(tag=tag, n=len(y), prev=p, auprc=a, lift=a / p,
                auroc=auroc(y, r), ece=ece(y, r))


def fresh_mask(df, bound):
    m = np.ones(len(df), dtype=bool)
    for c in VITALS:
        m &= df[f"{c}_hrs_since"].to_numpy() <= bound
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=["vitals_tcn", "news2", "labs"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--fusion-epochs", type=int, default=8)
    ap.add_argument("--no-check-repro", action="store_true")
    args = ap.parse_args()

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    eicu = pd.read_parquet(EICU_COHORT)

    # A collision would make eICU stays inherit MIMIC ECG/labs rows silently.
    mimic_ids = set(pd.concat([train, val, test])["stay_id"].unique())
    clash = mimic_ids & set(eicu["stay_id"].unique())
    assert not clash, f"{len(clash)} eICU/MIMIC stay_id collisions"
    print(f"[data] MIMIC test {len(test):,} h | eICU {len(eicu):,} h "
          f"({eicu.stay_id.nunique():,} stays, {eicu.hospitalid.nunique()} hospitals)")

    # ---- agents: fit on MIMIC, emit on MIMIC and eICU --------------------
    for name in args.agents:
        t0 = time.time()
        agent = build_agent(name, args.epochs)
        agent.fit(train, val)
        for split, df in (("train", train), ("val", val), ("test", test)):
            agent.emit_and_cache(df, EMB_DIR / f"{name}_{split}.npz")
        if not args.no_check_repro and (SHIPPED_EMB / f"{name}_test.npz").exists():
            from agents.base import AgentOutput
            mine = AgentOutput.load(EMB_DIR / f"{name}_test.npz")
            ship = AgentOutput.load(SHIPPED_EMB / f"{name}_test.npz")
            if mine.keys.equals(ship.keys):
                r = float(np.corrcoef(mine.risk, ship.risk)[0, 1])
                d = float(np.abs(mine.risk - ship.risk).max())
                print(f"[repro] {name}: risk corr vs shipped cache {r:.4f}, max|delta| {d:.4f}")
            else:
                print(f"[repro] {name}: key sets differ from shipped cache; skipped")
        point_at_eicu(name, agent)
        agent.emit_and_cache(eicu, EMB_DIR / f"{name}_eicu.npz")
        restore_mimic(name, agent)
        print(f"[agent] {name} done in {time.time()-t0:.0f}s")

    # ---- fusion: fit on MIMIC, score MIMIC test and eICU -----------------
    from orchestrator.fusion import AgentBank, Orchestrator
    bank = AgentBank.load(splits=("train", "val", "test", "eicu"),
                          agent_names=tuple(args.agents), emb_dir=EMB_DIR)
    orch = Orchestrator(bank=bank, max_epochs=args.fusion_epochs)
    t0 = time.time()
    orch.fit(train, val)
    print(f"[fusion] {'+'.join(args.agents)} fit in {time.time()-t0:.0f}s")

    def scored(df):
        out = orch.emit(df)
        srt = df.sort_values(["stay_id", "hour"]).reset_index(drop=True)
        assert (out.keys["stay_id"].to_numpy() == srt["stay_id"].to_numpy()).all()
        return srt, out.risk

    ts, tr = scored(test)
    es, er = scored(eicu)
    np.savez_compressed(REPO / "data" / "cache" / "eicu_fusion_scores.npz",
                        stay_id=es.stay_id.to_numpy(), hour=es.hour.to_numpy(),
                        risk=er, agents="+".join(args.agents))

    print(f"\n=== fusion ({'+'.join(args.agents)}) pooled ===")
    row("MIMIC test", ts[LABEL].to_numpy(float), tr)
    row("eICU (208 hospitals)", es[LABEL].to_numpy(float), er)

    print("\n=== matched charting density (all 5 vitals fresh) ===")
    for b in (12, 6, 3, 2):
        mm, me = fresh_mask(ts, b), fresh_mask(es, b)
        row(f"MIMIC  fresh <= {b}h", ts.loc[mm, LABEL].to_numpy(float), tr[mm])
        row(f"eICU   fresh <= {b}h", es.loc[me, LABEL].to_numpy(float), er[me])

    print("\n=== per-hospital (>=20k hours, >=50 positive hours) ===")
    tmp = es[["hospitalid", LABEL]].copy()
    tmp["r"] = er
    rows = []
    for h, g in tmp.groupby("hospitalid"):
        y = g[LABEL].to_numpy(float)
        if len(g) >= 20000 and y.sum() >= 50:
            rows.append((len(g), auroc(y, g.r.to_numpy()),
                         auprc(y, g.r.to_numpy()) / y.mean()))
    d = pd.DataFrame(rows, columns=["n", "auroc", "lift"])
    print(f"  {len(d)} hospitals | AUROC p10 {d.auroc.quantile(.1):.3f} "
          f"med {d.auroc.median():.3f} p90 {d.auroc.quantile(.9):.3f}")
    print(f"  {'':22} lift p10 {d.lift.quantile(.1):.2f} "
          f"med {d.lift.median():.2f} p90 {d.lift.quantile(.9):.2f}")


if __name__ == "__main__":
    main()
