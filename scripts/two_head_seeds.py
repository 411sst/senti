"""T18 -- ten-seed consistency of the timing edge, and the seed ensemble.

Five seeds left the flagship's lead@2% "positive in 3/5, null in two". This
rung reads the two-head cache at every seed it carries and reports, without
retraining anything:

  1. per seed, the anatomy of the paired race: wins : ties : losses, median,
     mean, 10th/90th percentiles, and the share of the model being earlier
     among DISCORDANT stays (the median sits on a ~20% tie atom, the mean is
     tail-dominated; the discordant share is the statistic that survives both);
  2. seed-level inference: mean and t-interval over seeds for each statistic,
     and how many seeds would be needed to pin the mean lead to +/-1 h;
  3. the pooled race (every seed's paired stays stacked, patient-clustered CI);
  4. the seed ensemble -- the mean of the seeds' calibrated composite scores,
     thresholds re-frozen on V_freeze -- graded like every rung (AUPRC/AUROC,
     paired delta vs FAIR+TRAJ, race at 1/2/5%, component race), plus
     prefix ensembles (k = 1..K in seed order) and leave-one-out ensembles as
     stability diagnostics of that one configuration;
  5. the tail: among stays where one system was >= 12 h earlier, how often
     the earlier system was simply at threshold from the first scored hour.

The shipped ensemble is "all seeds in the cache"; the prefix and
leave-one-out rows are diagnostics, nothing is selected on test.

    python scripts/two_head_seeds.py [--budget 0.02] [--reps 2000]
                                     [--cache data/cache/two_head.npz]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

OUT_JSON = REPO / "data" / "cache" / "two_head_seeds.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(REPO / "data/cache/two_head.npz"))
    ap.add_argument("--budget", type=float, default=0.02)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--auprc-reps", type=int, default=1000)
    ap.add_argument("--tail-cutoff", type=float, default=12.0)
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    from baselines_mimic import load_splits
    from decompose import component_auprc, component_race
    from fair_capability import (frozen_thresholds, paired_first_alarm_diffs,
                                 summarize_paired)
    from fusion_common import (FusionContext, delta_ci, load_fair_scores,
                               paired_auprc_draws)
    from metrics import auprc, auroc
    from seed_lead import (ensemble_scores, race_anatomy, seed_level_ci,
                           seeds_needed, summarize_tail, tail_diagnosis)

    t0 = time.time()
    ctx = FusionContext(load_splits())
    z = np.load(args.cache, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    seeds = [int(s) for s in meta["seeds"]]
    assert np.array_equal(z["stay_id_test"], ctx.test_ordered["stay_id"].to_numpy())
    nf, af = ctx.news2_val[ctx.freeze_mask], ctx.alertable_val[ctx.freeze_mask]
    budget = args.budget

    def race(s_val, s_test, reps):
        thr_n2, thr_m, _ = frozen_thresholds(nf, s_val[ctx.freeze_mask], af, budget)
        diffs = paired_first_alarm_diffs(ctx.test_ordered, ctx.news2_test, s_test,
                                         thr_n2, thr_m)
        r = race_anatomy(diffs["diff"])
        r.update({k: v for k, v in summarize_paired(
            diffs, ctx.stay_to_patient, reps=reps).items()
            if k in ("median_ci", "win_rate", "win_rate_ci")})
        r["thr_model"] = thr_m
        r["thr_news2"] = thr_n2
        r["model_rate"] = float((s_test[ctx.alertable_test] >= thr_m).mean())
        return r, diffs

    # 1. per seed ---------------------------------------------------------------
    sv = np.stack([z[f"val_s{s}"].astype(float) for s in seeds])
    st = np.stack([z[f"test_s{s}"].astype(float) for s in seeds])
    per_seed, pooled, tails = {}, [], {}
    for i, s in enumerate(seeds):
        r, diffs = race(sv[i], st[i], args.reps)
        r["auprc"] = float(auprc(ctx.y_test, st[i]))
        r["auroc"] = float(auroc(ctx.y_test, st[i]))
        tail = tail_diagnosis(ctx.test_ordered, ctx.news2_test, st[i],
                              r["thr_news2"], r["thr_model"], diffs,
                              cutoff=args.tail_cutoff)
        r["tail"] = summarize_tail(tail)
        per_seed[str(s)] = r
        pooled.append(diffs)
    print(f"[{len(seeds)} seeds {seeds}] loaded + raced in {time.time()-t0:.0f}s")

    # 2. seed-level inference ---------------------------------------------------
    level = {}
    for stat in ("median", "mean", "win_given_discordant", "auprc"):
        m, ci, sd = seed_level_ci([per_seed[str(s)][stat] for s in seeds])
        level[stat] = {"mean": m, "ci": ci, "sd": sd}
    level["n_median_positive"] = sum(per_seed[str(s)]["median"] > 0 for s in seeds)
    level["n_wins_exceed_losses"] = sum(
        per_seed[str(s)]["wins"] > per_seed[str(s)]["losses"] for s in seeds)
    level["n_sign_p_below_05"] = sum(per_seed[str(s)]["sign_p"] < 0.05 for s in seeds)
    level["seeds_for_pm1h_mean"] = seeds_needed(level["mean"]["sd"], 1.0)
    level["seeds_for_pm05h_median"] = seeds_needed(level["median"]["sd"], 0.5)

    # 3. pooled -----------------------------------------------------------------
    pooled_df = pd.concat(pooled, ignore_index=True)
    pooled_r = race_anatomy(pooled_df["diff"])
    pooled_r.update({k: v for k, v in summarize_paired(
        pooled_df, ctx.stay_to_patient, reps=args.reps).items()
        if k in ("median_ci", "win_rate", "win_rate_ci")})

    # 4. the seed ensemble ------------------------------------------------------
    _, fair_test = load_fair_scores(ctx.val_ordered, ctx.test_ordered)
    ens_val, ens_test = ensemble_scores(sv), ensemble_scores(st)
    ens = ctx.evaluate_rung(ens_val, ens_test, reps=args.reps)
    ens["auroc"] = float(auroc(ctx.y_test, ens_test))
    ens["race"] = component_race(ctx, ens_val, ens_test, reps=args.reps)
    ens["auprc_components"] = component_auprc(ctx, ens_test)
    draws = paired_auprc_draws(ctx.y_test, {"ens": ens_test, "fair": fair_test},
                               ctx.patient_test, reps=args.auprc_reps)
    ens["delta_vs_fair_ci"] = delta_ci(draws["ens"], draws["fair"])
    ens_r, ens_diffs = race(ens_val, ens_test, args.reps)
    ens["anatomy"] = ens_r
    tail = tail_diagnosis(ctx.test_ordered, ctx.news2_test, ens_test,
                          ens_r["thr_news2"], ens_r["thr_model"], ens_diffs,
                          cutoff=args.tail_cutoff)
    ens["tail"] = summarize_tail(tail)
    med_r, _ = race(ensemble_scores(sv, "median"), ensemble_scores(st, "median"),
                    args.reps)
    ens["median_rule_anatomy"] = med_r
    print(f"[ensemble of {len(seeds)}] AUPRC {ens['auprc']:.4f} AUROC {ens['auroc']:.4f} "
          f"vs FAIR [{ens['delta_vs_fair_ci'][0]:+.4f},{ens['delta_vs_fair_ci'][1]:+.4f}] "
          f"| lead@{budget:.0%} median {ens_r['median']:+.1f} CI "
          f"[{ens_r['median_ci'][0]:+.1f},{ens_r['median_ci'][1]:+.1f}] "
          f"win|disc {ens_r['win_given_discordant']:.3f} "
          f"({time.time()-t0:.0f}s)")

    prefix = {}
    for k in range(1, len(seeds) + 1):
        r, _ = race(ensemble_scores(sv[:k]), ensemble_scores(st[:k]), args.reps)
        r["auprc"] = float(auprc(ctx.y_test, ensemble_scores(st[:k])))
        prefix[str(k)] = r
    loo = {}
    for i, s in enumerate(seeds):
        keep = [j for j in range(len(seeds)) if j != i]
        r, _ = race(ensemble_scores(sv[keep]), ensemble_scores(st[keep]), args.reps)
        loo[str(s)] = r

    out = {
        "seeds": seeds, "budget": budget, "reps": args.reps,
        "cache": str(args.cache), "tail_cutoff": args.tail_cutoff,
        "per_seed": per_seed, "seed_level": level, "pooled": pooled_r,
        "ensemble": ens, "prefix_ensembles": prefix, "leave_one_out": loo,
    }
    Path(args.out).write_text(json.dumps(_jsonable(out), indent=1))
    print(f"[report] -> {args.out}  ({time.time()-t0:.0f}s)")
    print_markdown(out)


def print_markdown(o):
    seeds, b = o["seeds"], o["budget"]
    print(f"\n| seed | AUPRC | wins : ties : losses | median | mean | p10 / p90 | earlier when they differ | sign p |")
    print("|---|---|---|---|---|---|---|---|")
    for s in seeds:
        r = o["per_seed"][str(s)]
        print(f"| {s} | {r['auprc']:.4f} | {r['wins']} : {r['ties']} : {r['losses']} "
              f"| {r['median']:+.1f} [{r['median_ci'][0]:+.0f}, {r['median_ci'][1]:+.0f}] "
              f"| {r['mean']:+.1f} | {r['p10']:+.0f} / {r['p90']:+.0f} "
              f"| {100*r['win_given_discordant']:.0f}% | {r['sign_p']:.1e} |")
    L = o["seed_level"]
    print(f"\nseed-level ({len(seeds)} seeds): median {L['median']['mean']:+.2f} h "
          f"[{L['median']['ci'][0]:+.2f}, {L['median']['ci'][1]:+.2f}] SD {L['median']['sd']:.2f}; "
          f"mean {L['mean']['mean']:+.2f} h [{L['mean']['ci'][0]:+.2f}, {L['mean']['ci'][1]:+.2f}] "
          f"SD {L['mean']['sd']:.2f}; earlier-when-differ {100*L['win_given_discordant']['mean']:.1f}% "
          f"[{100*L['win_given_discordant']['ci'][0]:.1f}, {100*L['win_given_discordant']['ci'][1]:.1f}]; "
          f"AUPRC {L['auprc']['mean']:.4f} SD {L['auprc']['sd']:.4f}")
    print(f"  median > 0 in {L['n_median_positive']}/{len(seeds)}; wins > losses in "
          f"{L['n_wins_exceed_losses']}/{len(seeds)}; sign p < 0.05 in {L['n_sign_p_below_05']}/{len(seeds)}; "
          f"seeds needed for +/-1 h on the mean: {L['seeds_for_pm1h_mean']}; "
          f"for +/-0.5 h on the median: {L['seeds_for_pm05h_median']}")
    P = o["pooled"]
    print(f"pooled: n {P['n']} median {P['median']:+.1f} [{P['median_ci'][0]:+.1f}, {P['median_ci'][1]:+.1f}] "
          f"mean {P['mean']:+.2f} earlier-when-differ {100*P['win_given_discordant']:.1f}% "
          f"win {100*P['win_rate']:.1f}% [{100*P['win_rate_ci'][0]:.1f}, {100*P['win_rate_ci'][1]:.1f}]")
    E, A = o["ensemble"], o["ensemble"]["anatomy"]
    print(f"\nensemble ({len(seeds)} seeds, mean rule): AUPRC {E['auprc']:.4f} AUROC {E['auroc']:.4f} "
          f"vs FAIR+TRAJ [{E['delta_vs_fair_ci'][0]:+.4f}, {E['delta_vs_fair_ci'][1]:+.4f}] "
          f"death AUPRC {E['auprc_components']['death']:.3f}")
    for bud, r in E["budgets"].items():
        print(f"  lead@{float(bud):.0%}: n {r['n_pair']} median {r['median']:+.1f} "
              f"[{r['median_ci'][0]:+.1f}, {r['median_ci'][1]:+.1f}] win {100*r['win_rate']:.0f}% "
              f"[{100*r['win_rate_ci'][0]:.0f}, {100*r['win_rate_ci'][1]:.0f}]")
    print(f"  @{b:.0%} anatomy: {A['wins']} : {A['ties']} : {A['losses']}, mean {A['mean']:+.2f}, "
          f"earlier-when-differ {100*A['win_given_discordant']:.1f}%, sign p {A['sign_p']:.1e}; "
          f"median rule: median {E['median_rule_anatomy']['median']:+.1f} "
          f"earlier-when-differ {100*E['median_rule_anatomy']['win_given_discordant']:.1f}%")
    for arm in ("treatment", "death"):
        r = E["race"][b][arm] if b in E["race"] else E["race"][str(b)][arm]
        print(f"  {arm}-stay lead@{b:.0%}: n {r['n_pair']} median {r['median']:+.1f} "
              f"[{r['median_ci'][0]:+.1f}, {r['median_ci'][1]:+.1f}] win {100*r['win_rate']:.0f}%")
    print("\nprefix ensembles (k seeds in cache order):")
    for k, r in o["prefix_ensembles"].items():
        print(f"  k={int(k):>2}: AUPRC {r['auprc']:.4f} median {r['median']:+.1f} "
              f"[{r['median_ci'][0]:+.1f}, {r['median_ci'][1]:+.1f}] "
              f"earlier-when-differ {100*r['win_given_discordant']:.1f}% mean {r['mean']:+.2f}")
    print("leave-one-out ensembles:")
    for s, r in o["leave_one_out"].items():
        print(f"  without {s:>2}: median {r['median']:+.1f} [{r['median_ci'][0]:+.1f}, {r['median_ci'][1]:+.1f}] "
              f"earlier-when-differ {100*r['win_given_discordant']:.1f}% mean {r['mean']:+.2f}")
    print(f"\ntail (|diff| >= {o['tail_cutoff']:.0f} h): per seed, share of the EARLIER system already at "
          f"threshold on the first scored hour")
    for s in seeds:
        t = o["per_seed"][str(s)]["tail"]
        print(f"  seed {s:>2}: losses {t['loss']['n']:>3} ({100*t['loss']['share_from_first_hour']:.0f}% standing NEWS2) "
              f"| wins {t['win']['n']:>3} ({100*t['win']['share_from_first_hour']:.0f}% standing model)")
    t = E["tail"]
    print(f"  ensemble: losses {t['loss']['n']} ({100*t['loss']['share_from_first_hour']:.0f}% standing NEWS2) "
          f"| wins {t['win']['n']} ({100*t['win']['share_from_first_hour']:.0f}% standing model)")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


if __name__ == "__main__":
    main()
