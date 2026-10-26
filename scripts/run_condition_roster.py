"""Condition -> agent roster: which situation correlates best to which agents (T13).

The clinician's actual meaning of "standby", clarified: if the patient's
situation is (say) sepsis, then particular agents — their guess was vitals and
fluids — may be the ones that detect the anomaly. Units answer WHERE the
patient is; this answers WHAT IS WRONG with them. Conditions are any-position
ICD flags, so unlike units they OVERLAP (a septic patient can also be in renal
failure) and are not a partition — each condition is analyzed as its own
subgroup and no threshold refitting happens here.

**Deployment honesty:** these flags are discharge-coded. They are a valid
retrospective label of what the stay was, and therefore valid for asking
"which agents carry the signal in sepsis stays" — but they can never be a
real-time selector input. The real-time form of this standby already exists:
the orchestrator's attention reads the same evidence the flags summarize,
hour by hour, from the agents themselves.

    python scripts/run_condition_roster.py [--reps 1000]
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
import pandas as pd

OUT_JSON = REPO / "data" / "cache" / "condition_roster.json"

CONDITIONS = [
    ("sepsis",       "dx_sepsis",    "sepsis / septicemia"),
    ("resp failure", "dx_respfail",  "acute respiratory failure"),
    ("pneumonia",    "dx_pneumonia", "pneumonia"),
    ("AKI",          "dx_aki",       "acute kidney injury"),
    ("heart failure","dx_hf",        "heart failure"),
    ("GI bleed",     "dx_gibleed",   "gastrointestinal haemorrhage"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--budget", type=float, default=0.02)
    args = ap.parse_args()

    from baselines_mimic import load_splits
    from fair_capability import frozen_thresholds, paired_first_alarm_diffs, summarize_paired
    from fusion_common import FusionContext
    from icu_strata import load_stay_meta
    from metrics import auprc
    from run_standby import load_agent_risks
    from standby import FLAGSHIP_AGENTS

    splits = load_splits()
    ctx = FusionContext(splits)
    meta = load_stay_meta().set_index("stay_id")
    risks = load_agent_risks(ctx)

    th = np.load(REPO / "data/cache/two_head.npz")
    assert (th["stay_id_test"] == ctx.test_ordered["stay_id"].to_numpy()).all()
    s_val, s_test = th["val_s42"].astype(float), th["test_s42"].astype(float)
    thr_n2, thr_m, _ = frozen_thresholds(
        ctx.news2_val[ctx.freeze_mask], s_val[ctx.freeze_mask],
        ctx.alertable_val[ctx.freeze_mask], args.budget)

    test_stays = ctx.test_ordered["stay_id"].to_numpy()
    y = ctx.y_test

    print("=" * 96)
    print("T13 — WHICH SITUATION CORRELATES BEST TO WHICH AGENTS "
          "(standalone lift per condition, test split)")
    print("=" * 96)
    print("condition flags are any-position ICD — retrospective stay labels, "
          "overlapping, never a real-time input\n")

    agents = list(FLAGSHIP_AGENTS)
    hdr = (f"{'condition':<14}{'stays':>7}{'prev':>7}"
           + "".join(f"{a[:9]:>11}" for a in agents) + f"{'  race':>14}")
    print(hdr); print("-" * len(hdr))

    out = {}
    for label, flag, desc in CONDITIONS:
        stay_has = meta[flag].astype("float").fillna(0).astype(bool)
        sel = pd.Series(test_stays).map(stay_has).to_numpy()
        n_stays = int(pd.unique(test_stays[sel]).size)
        ys = y[sel]
        prev = float(ys.mean())
        rec = {"desc": desc, "test_stays": n_stays, "hours": int(sel.sum()),
               "prevalence": prev, "agents": {}}

        cells, lifts = "", {}
        if ys.sum() >= 25:
            for a in agents:
                v = float(auprc(ys, risks[a][sel]))
                lifts[a] = v / prev
                rec["agents"][a] = {"auprc": v, "lift": lifts[a]}
            top3 = set(sorted(lifts, key=lambda k: -lifts[k])[:3])
            cells = "".join(f"{lifts[a]:>10.1f}{'*' if a in top3 else ' '}"
                            for a in agents)
            rec["top3"] = sorted(top3, key=lambda k: -lifts[k])
        else:
            cells = "   under-powered".ljust(11 * len(agents))
            rec["underpowered"] = True

        # flagship race within the condition subgroup, global thresholds
        sub = ctx.test_ordered[sel]
        diffs = paired_first_alarm_diffs(sub, ctx.news2_test[sel], s_test[sel],
                                         thr_n2, thr_m)
        if len(diffs) >= 20:
            r = summarize_paired(diffs, ctx.stay_to_patient, reps=args.reps)
            race = f"{r['median']:+.1f}h/{r['win_rate']:.0%}({r['n_pair']})"
            rec["race"] = {"median": r["median"], "win_rate": r["win_rate"],
                           "n_pair": r["n_pair"]}
        else:
            race = f"--({len(diffs)})"
            rec["race"] = {"n_pair": int(len(diffs))}
        print(f"{label:<14}{n_stays:>7,}{prev:>7.1%}{cells}{race:>14}")
        out[label] = rec

    print("\n  * top-3 for that condition (lift = AUPRC / prevalence; race = "
          "flagship vs NEWS2 at the global 2% budget)")

    # The clinician's specific guess, answered directly.
    sep = out.get("sepsis", {})
    if "top3" in sep:
        guess = {"vitals_tcn", "fluid"}
        hit = guess & set(sep["top3"])
        print(f"\n  clinician's guess for sepsis (vitals + fluids): top-3 is "
              f"{', '.join(sep['top3'])} — "
              + (f"{len(hit)}/2 of the guess confirmed" if hit else
                 "neither in the top-3"))
        fl = sep["agents"]
        print(f"    fluid lift in sepsis {fl['fluid']['lift']:.1f} vs its "
              f"cohort-wide 1.7-1.8 — "
              + ("elevated" if fl["fluid"]["lift"] > 2.0 else "not elevated"))

    OUT_JSON.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
