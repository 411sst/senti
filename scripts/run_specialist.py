"""Specialist Sentinels: tailor the model to an ICU scenario (T14).

The doctors' proposal, stated precisely: build Sentinel variants specialized to
cardio / neuro / respiratory / urology / burns, each using the best-suited
agents ("standby agents" = the full bank stays cached; the specialist activates
its roster). T12 already gave every unit a local OPERATING POINT; this script
asks the remaining question — does specializing the MODEL add anything beyond
specializing the threshold?

Three arms per specialty, all raced under specialty-local thresholds
(frozen on the specialty's own V_freeze — the T12 standby points):

    generalist        the global two-head flagship, scores as deployed
    specialist-all8   two-head retrained ONLY on the specialty's stays,
                      full 8-agent bank
    specialist-roster two-head retrained on the specialty's stays, bank
                      restricted to the top-4 agents by standalone lift on
                      the specialty's V_freeze half (selection never sees test)

Urology and burns are reported as unbuildable, with the numbers that make the
call (197 / 128 stays in the whole cohort; 36 / 21 in test): a specialist
cannot be trained, evaluated, or honestly claimed for them from MIMIC-IV.

    python scripts/run_specialist.py [--specialty cardio,neuro,respiratory]
                                     [--seeds 42,1,2] [--epochs 8]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import pandas as pd

OUT_DIR = REPO / "data" / "cache"
UNBUILDABLE = {"urology": (197, 36), "burns": (128, 21)}
ROSTER_K = 4


def specialty_splits(splits, strata, cat):
    keep = set(strata.loc[strata["clinician_category"] == cat, "stay_id"])
    out = {k: df[df["stay_id"].isin(keep)] for k, df in splits.items()}
    for k, df in out.items():
        if not len(df):
            raise ValueError(f"{cat}: empty {k} split")
    return out


def freeze_roster(ctx_s, full_val_ordered, k=ROSTER_K):
    """Top-k agents by standalone lift on the specialty's V_freeze rows.

    Emissions are cached aligned to the FULL val split, so they are masked down
    to the specialty rows by (stay_id, hour) position. Selection uses V_freeze
    only — the same val-selection optimism the repo already accepts.
    """
    from metrics import auprc
    from standby import FLAGSHIP_AGENTS

    full_keys = pd.MultiIndex.from_arrays(
        [full_val_ordered["stay_id"].to_numpy(),
         full_val_ordered["hour"].to_numpy()])
    spec_val = ctx_s.val_ordered
    want = pd.MultiIndex.from_arrays(
        [spec_val["stay_id"].to_numpy(), spec_val["hour"].to_numpy()])
    pos = full_keys.get_indexer(want)
    assert (pos >= 0).all(), "specialty val rows missing from full val"

    yf = ctx_s.y_val[ctx_s.freeze_mask]
    prev = float(yf.mean())
    lifts = {}
    for name in FLAGSHIP_AGENTS:
        z = np.load(OUT_DIR / "embeddings" / f"{name}_val.npz")
        risk = z["risk"].astype(float)[pos][ctx_s.freeze_mask]
        lifts[name] = float(auprc(yf, risk)) / max(prev, 1e-9)
    roster = sorted(lifts, key=lambda n: -lifts[n])[:k]
    return roster, lifts


def fit_two_head(bank, ssp, ctx_s, seed, epochs):
    """Two-head on the given bank/splits; returns calibrated val/test scores."""
    from decompose import treatment_only_label
    from fusion_ladder import _strictify
    from sklearn.isotonic import IsotonicRegression
    from two_head import fit_head

    train_o = ssp["train"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    val_o = ctx_s.val_ordered
    labels = {
        "T": ({"train": treatment_only_label(train_o),
               "val": treatment_only_label(val_o)}),
        "D": ({"train": train_o["label_24h_death"].to_numpy(dtype=float),
               "val": val_o["label_24h_death"].to_numpy(dtype=float)}),
    }
    heads = {}
    for name, y in labels.items():
        head = fit_head(bank, ssp, y["train"], y["val"], train_o, val_o,
                        seed, epochs)
        raw_val = head.raw_risk(ssp["val"])
        raw_test = head.raw_risk(ssp["test"])
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            raw_val[ctx_s.fit_mask], y["val"][ctx_s.fit_mask])
        heads[name] = {"val": _strictify(iso.transform(raw_val), raw_val),
                       "test": _strictify(iso.transform(raw_test), raw_test)}
    s_val = 1.0 - (1.0 - heads["T"]["val"]) * (1.0 - heads["D"]["val"])
    s_test = 1.0 - (1.0 - heads["T"]["test"]) * (1.0 - heads["D"]["test"])
    return s_val, s_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specialty", default="cardio,neuro,respiratory")
    ap.add_argument("--seeds", default="42,1,2")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    from baselines_mimic import load_splits
    from fusion_common import FusionContext, delta_ci, paired_auprc_draws
    from icu_strata import assign_strata
    from metrics import auroc
    from orchestrator.fusion import AgentBank
    from run_icu_strata import stay_frame
    from standby import FLAGSHIP_AGENTS

    for cat, (n_all, n_test) in UNBUILDABLE.items():
        print(f"[unbuildable] {cat}: {n_all} cohort stays / {n_test} in test — "
              f"no specialist can be trained or evaluated from MIMIC-IV")

    splits = load_splits()
    strata = assign_strata(stay_frame(splits))
    full_val_ordered = splits["val"].sort_values(
        ["stay_id", "hour"]).reset_index(drop=True)
    zg = np.load(OUT_DIR / "two_head.npz")

    for cat in args.specialty.split(","):
        cat = cat.strip()
        t0 = time.time()
        ssp = specialty_splits(splits, strata, cat)
        ctx_s = FusionContext(ssp)
        n_tr = ssp["train"]["stay_id"].nunique()
        print(f"\n=== {cat}: train {len(ssp['train']):,} rows / {n_tr:,} stays | "
              f"test {len(ssp['test']):,} rows, prev {ctx_s.y_test.mean():.1%} ===",
              flush=True)

        roster, lifts = freeze_roster(ctx_s, full_val_ordered)
        print(f"[roster] V_freeze lift: "
              + " ".join(f"{n}={v:.1f}" for n, v in
                         sorted(lifts.items(), key=lambda kv: -kv[1]))
              + f" -> top-{ROSTER_K}: {roster}", flush=True)

        # generalist scores restricted to the specialty rows (key-mapped)
        spec_test_keys = pd.MultiIndex.from_arrays(
            [ctx_s.test_ordered["stay_id"].to_numpy(),
             ctx_s.test_ordered["hour"].to_numpy()])
        full_test_keys = pd.MultiIndex.from_arrays(
            [zg["stay_id_test"], zg["hour_test"]])
        tpos = full_test_keys.get_indexer(spec_test_keys)
        assert (tpos >= 0).all()
        full_val_keys = pd.MultiIndex.from_arrays(
            [full_val_ordered["stay_id"].to_numpy(),
             full_val_ordered["hour"].to_numpy()])
        spec_val_keys = pd.MultiIndex.from_arrays(
            [ctx_s.val_ordered["stay_id"].to_numpy(),
             ctx_s.val_ordered["hour"].to_numpy()])
        vpos = full_val_keys.get_indexer(spec_val_keys)
        assert (vpos >= 0).all()

        bank8 = AgentBank.load(agent_names=tuple(FLAGSHIP_AGENTS))
        bankR = AgentBank.load(agent_names=tuple(roster))

        payload = {"stay_id_test": ctx_s.test_ordered["stay_id"].to_numpy(),
                   "hour_test": ctx_s.test_ordered["hour"].to_numpy()}
        results = {}
        for seed in seeds:
            arms = {"generalist": (zg[f"val_s{seed}"].astype(float)[vpos],
                                   zg[f"test_s{seed}"].astype(float)[tpos])}
            arms["specialist-all8"] = fit_two_head(bank8, ssp, ctx_s,
                                                   seed, args.epochs)
            arms["specialist-roster"] = fit_two_head(bankR, ssp, ctx_s,
                                                     seed, args.epochs)

            row = {}
            draws = paired_auprc_draws(
                ctx_s.y_test, {k: v[1] for k, v in arms.items()},
                ctx_s.patient_test, reps=args.reps)
            for name, (s_val, s_test) in arms.items():
                r = ctx_s.evaluate_rung(s_val, s_test, reps=args.reps)
                r["auroc"] = float(auroc(ctx_s.y_test, s_test))
                if name != "generalist":
                    r["delta_vs_generalist_ci"] = delta_ci(
                        draws[name], draws["generalist"])
                row[name] = r
                b = r["budgets"][0.02]
                extra = ("" if name == "generalist" else
                         f" | dAUPRC vs gen [{r['delta_vs_generalist_ci'][0]:+.4f},"
                         f"{r['delta_vs_generalist_ci'][1]:+.4f}]")
                print(f"[{cat} s{seed}] {name:<18} AUPRC {r['auprc']:.4f} "
                      f"AUROC {r['auroc']:.4f} | lead@2% {b['median']:+.1f}h "
                      f"win {b['win_rate']:.0%} (n={b['n_pair']}){extra}",
                      flush=True)
                payload[f"val_{name}_s{seed}"] = arms[name][0]
                payload[f"test_{name}_s{seed}"] = arms[name][1]
            results[seed] = row

        payload["meta_json"] = np.array(json.dumps(
            {"specialty": cat, "roster": roster, "lifts": lifts,
             "seeds": seeds, "results": results}, default=float))
        out = OUT_DIR / f"specialist_{cat}.npz"
        np.savez_compressed(out, **payload)
        print(f"[{cat}] done in {(time.time()-t0)/60:.1f} min -> {out}",
              flush=True)


if __name__ == "__main__":
    main()
