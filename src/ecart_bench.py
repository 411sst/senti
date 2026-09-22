"""T19 -- the eCART bench: the flagship vs the strongest published GBM,
with NEWS2 demoted from score-to-beat to clinical anchor.

Why this rung exists. The eCARTv5 validation paper (Churpek 2025, Crit Care
Explor; ward population) picks each score's "commonly used" thresholds post
hoc and measures lead time as time-from-first-crossing-anywhere-in-the-
admission -- unpaired, volume-unmatched, standing alarms included. Its own
numbers show the ranking/earliness dissociation this repo keeps finding:
eCARTv5 beats NEWS by 6.8 AUROC points and, at the moderate-risk threshold,
warns no earlier (median 16 h vs NEWS's 17 h). Meanwhile racing NEWS2 at a
volume-matched threshold of ~11 is statistically necessary (the ward bands
>=5 / >=7 cover 45% / 23% of ICU hours -- volumes no ICU can staff) but
rhetorically fragile ("a patient at 11 is already being seen"). This rung
removes the dependency: the reference row becomes the eCART-style GBM
itself, raced head-to-head at matched alert volume, and NEWS2 appears only
as a cohort-relative severity anchor whose bands are derived from this
cohort's own score distribution rather than imported from the wards.

What it computes (nothing is trained; caches only):

1. Direct race, per seed and for the seed-mean ensemble: the two-head
   flagship vs ``lit:ecart`` at the same V_freeze-frozen alert volume --
   anatomy (wins : ties : losses), medians with patient-clustered CIs, the
   discordant-share statistic, the standing-alarm tail diagnosis, and the
   paired AUPRC delta.
2. Cohort-relative NEWS2 bands: the smallest integer thresholds whose
   V_freeze alert volume is <= 25% / 10% / 5% (this cohort's own ~P75 /
   P90 / P95). Per band: test volume, non-event-stay burden, and the
   volume-matched race -- with the standing-alarm share that shows where
   the race stops measuring detection.
3. Anchor stats at the 2% NEWS2-matched operating point: how many of the
   model's catches fire before the stay ever enters each band, how many
   caught stays never enter it pre-event, the alarm-to-band gap, and the
   NEWS2 level of the final pre-event hours.

    python src/ecart_bench.py [--budget 0.02] [--reps 2000] [--auprc-reps 500]
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")

OUT_JSON = "data/cache/ecart_bench.json"
LIT_CACHE = "data/cache/lit_gbm_scores.npz"
TWO_HEAD_CACHE = "data/cache/two_head.npz"
BAND_TAIL_SHARES = (0.25, 0.10, 0.05)


# --- Pure helpers (tested in tests/test_ecart_bench.py) -----------------------

def cohort_bands(news2, alertable=None, tail_shares=BAND_TAIL_SHARES):
    """Cohort-relative severity bands from the score's own distribution.

    The RCP >=5 / >=7 bands were calibrated on general wards, where the
    average patient scores near 0; in this ICU cohort the mean alertable
    hour already sits at ~4.4, so importing them wholesale yields alarm
    volumes no ICU can staff. Instead each band is the smallest integer
    threshold whose alert volume on the given hours is <= the tail share
    (25% / 10% / 5% ~ the cohort's own P75 / P90 / P95). Derived from val's
    freeze half in the harness, like every other frozen operating point.
    """
    n2 = np.asarray(news2, dtype=float)
    if alertable is not None:
        n2 = n2[np.asarray(alertable, dtype=bool)]
    out = []
    for share in tail_shares:
        thr = 0
        while (n2 >= thr).mean() > share:
            thr += 1
        out.append({"tail_share": float(share),
                    "percentile": int(round(100 * (1 - share))),
                    "threshold": int(thr),
                    "val_volume": float((n2 >= thr).mean())})
    return out


def first_alarm_hours(test, scores, thr):
    """First hour each stay's score reaches ``thr`` (stays that never do
    are absent from the result)."""
    tmp = test[["stay_id", "hour"]].copy()
    tmp["s"] = np.asarray(scores, dtype=float)
    return tmp[tmp["s"] >= thr].groupby("stay_id")["hour"].min()


def standing_alarm_share(test, scores, thr, stay_ids):
    """Among ``stay_ids`` whose score ever reaches ``thr``: the share whose
    first crossing is the stay's FIRST scored hour -- a standing alarm
    (chart position), not a detection."""
    fa = first_alarm_hours(test, scores, thr).reindex(stay_ids).dropna()
    if not len(fa):
        return float("nan")
    h0 = test.groupby("stay_id")["hour"].min().reindex(fa.index)
    return float((fa == h0).mean())


def band_anchor_stats(test, s_model, thr_model, band_thr):
    """The model's catches, positioned against one NEWS2 band.

    Catches are event stays whose first model alarm lands at or before the
    event (the paired-race eligibility rule). Reported against the band:
    the share of first alarms firing before the stay has EVER reached the
    band, the share of caught stays that never enter the band at any
    pre-event hour, and the alarm-to-first-crossing gap where a crossing
    exists.
    """
    ev = test.groupby("stay_id")["event_h"].first()
    caught = first_alarm_hours(test, s_model, thr_model).reindex(ev.index).dropna()
    caught = caught[caught <= ev.reindex(caught.index)]
    fa_band = first_alarm_hours(test, test["news2"], band_thr).reindex(caught.index)
    ev_c = ev.reindex(caught.index)
    gap = (fa_band - caught).dropna()
    gap = gap[gap > 0]
    n = len(caught)
    nan = float("nan")
    return {
        "band_thr": float(band_thr),
        "n_caught": int(n),
        "share_before_stay_ever_in_band":
            float((fa_band.isna() | (caught < fa_band)).mean()) if n else nan,
        "share_never_in_band_pre_event":
            float((fa_band.isna() | (fa_band >= ev_c)).mean()) if n else nan,
        "n_crossing_later": int(len(gap)),
        "gap_median_h": float(gap.median()) if len(gap) else nan,
        "gap_iqr_h": ((float(gap.quantile(0.25)), float(gap.quantile(0.75)))
                      if len(gap) else (nan, nan)),
    }


def pre_event_news2(test, window):
    """NEWS2 level over the final ``window`` pre-event hours of event stays."""
    evt = test[test["event_h"].notna()]
    pre = evt[(evt["hour"] < evt["event_h"])
              & (evt["hour"] >= evt["event_h"] - window)]
    return {"window_h": int(window),
            "n_hours": int(len(pre)),
            "mean": float(pre["news2"].mean()) if len(pre) else float("nan"),
            "median": float(pre["news2"].median()) if len(pre) else float("nan")}


# --- The rung -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=TWO_HEAD_CACHE)
    ap.add_argument("--lit-cache", default=LIT_CACHE)
    ap.add_argument("--budget", type=float, default=0.02)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--auprc-reps", type=int, default=500)
    ap.add_argument("--tail-cutoff", type=float, default=12.0)
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()

    from baselines_mimic import load_splits
    from fair_capability import (BUDGETS, frozen_thresholds,
                                 paired_first_alarm_diffs, summarize_paired)
    from fusion_common import FusionContext, delta_ci, paired_auprc_draws
    from metrics import auprc, auroc
    from seed_lead import (ensemble_scores, race_anatomy, seed_level_ci,
                           summarize_tail, tail_diagnosis)

    t0 = time.time()
    ctx = FusionContext(load_splits())
    z = np.load(args.cache, allow_pickle=True)
    seeds = [int(s) for s in json.loads(str(z["meta_json"]))["seeds"]]
    assert np.array_equal(z["stay_id_test"],
                          ctx.test_ordered["stay_id"].to_numpy())

    lit = np.load(args.lit_cache, allow_pickle=True)
    for name, ordered in (("val", ctx.val_ordered), ("test", ctx.test_ordered)):
        if not ((lit[f"{name}_stay_id"] == ordered["stay_id"].to_numpy()).all()
                and (lit[f"{name}_hour"] == ordered["hour"].to_numpy()).all()):
            raise ValueError(f"{args.lit_cache} {name} rows do not align with "
                             "the cohort -- re-run src/lit_gbm.py")
    g_val = lit["val_ecart"].astype(float)
    g_test = lit["test_ecart"].astype(float)

    fm = ctx.freeze_mask
    af = ctx.alertable_val[fm]
    gf = g_val[fm]

    def gbm_race(s_val, s_test, budget, reps):
        """Flagship vs eCART-style GBM, both frozen to the same V_freeze
        alert volume; diff > 0 = flagship's first alarm earlier."""
        thr_g, thr_m, matched = frozen_thresholds(gf, s_val[fm], af, budget)
        diffs = paired_first_alarm_diffs(ctx.test_ordered, g_test, s_test,
                                         thr_g, thr_m)
        r = race_anatomy(diffs["diff"])
        r.update({k: v for k, v in summarize_paired(
            diffs, ctx.stay_to_patient, reps=reps).items()
            if k in ("median_ci", "win_rate", "win_rate_ci")})
        r.update({"thr_ecart": thr_g, "thr_model": thr_m, "matched": matched,
                  "ecart_rate": float((g_test[ctx.alertable_test] >= thr_g).mean()),
                  "model_rate": float((s_test[ctx.alertable_test] >= thr_m).mean())})
        tail = tail_diagnosis(ctx.test_ordered, g_test, s_test, thr_g, thr_m,
                              diffs, cutoff=args.tail_cutoff)
        r["tail"] = summarize_tail(tail)
        return r, diffs

    # 1. direct race, per seed -------------------------------------------------
    sv = np.stack([z[f"val_s{s}"].astype(float) for s in seeds])
    st = np.stack([z[f"test_s{s}"].astype(float) for s in seeds])
    per_seed, pooled = {}, []
    for i, s in enumerate(seeds):
        r, diffs = gbm_race(sv[i], st[i], args.budget, args.reps)
        r["auprc"] = float(auprc(ctx.y_test, st[i]))
        per_seed[str(s)] = r
        pooled.append(diffs)
    print(f"[{len(seeds)} seeds raced vs lit:ecart @{args.budget:.0%}] "
          f"{time.time()-t0:.0f}s")

    pooled_df = pd.concat(pooled, ignore_index=True)
    pooled_r = race_anatomy(pooled_df["diff"])
    pooled_r.update({k: v for k, v in summarize_paired(
        pooled_df, ctx.stay_to_patient, reps=args.reps).items()
        if k in ("median_ci", "win_rate", "win_rate_ci")})

    ens_val, ens_test = ensemble_scores(sv), ensemble_scores(st)
    ens = {"auprc": float(auprc(ctx.y_test, ens_test)),
           "auroc": float(auroc(ctx.y_test, ens_test)),
           "budgets": {}}
    for b in BUDGETS:
        r, _ = gbm_race(ens_val, ens_test, b, args.reps)
        ens["budgets"][str(b)] = r

    # one clustered resampling loop serves every AUPRC delta
    draws = paired_auprc_draws(
        ctx.y_test,
        {"ecart": g_test, "ens": ens_test,
         **{f"s{s}": st[i] for i, s in enumerate(seeds)}},
        ctx.patient_test, reps=args.auprc_reps)
    ecart_auprc = float(auprc(ctx.y_test, g_test))
    for s in seeds:
        per_seed[str(s)]["delta_auprc_vs_ecart"] = \
            per_seed[str(s)]["auprc"] - ecart_auprc
        per_seed[str(s)]["delta_auprc_ci"] = delta_ci(draws[f"s{s}"],
                                                      draws["ecart"])
    ens["delta_auprc_vs_ecart"] = ens["auprc"] - ecart_auprc
    ens["delta_auprc_ci"] = delta_ci(draws["ens"], draws["ecart"])
    print(f"[AUPRC deltas, {args.auprc_reps} clustered reps] "
          f"{time.time()-t0:.0f}s")

    level = {}
    for stat in ("median", "mean", "win_given_discordant", "auprc",
                 "delta_auprc_vs_ecart"):
        m, ci, sd = seed_level_ci([per_seed[str(s)][stat] for s in seeds])
        level[stat] = {"mean": m, "ci": ci, "sd": sd}
    level["n_wins_exceed_losses"] = sum(
        per_seed[str(s)]["wins"] > per_seed[str(s)]["losses"] for s in seeds)
    level["n_sign_p_below_05"] = sum(
        per_seed[str(s)]["sign_p"] < 0.05 for s in seeds)
    level["n_delta_ci_clear"] = sum(
        per_seed[str(s)]["delta_auprc_ci"][0] > 0 for s in seeds)

    # 2. cohort-relative NEWS2 bands -------------------------------------------
    i42 = seeds.index(42)
    bands = cohort_bands(ctx.news2_val[fm], af)
    n2_test_alert = ctx.news2_test[ctx.alertable_test]
    ev_stay = ctx.test_ordered.groupby("stay_id")["event_h"].first()
    nonevent_ids = ev_stay[ev_stay.isna()].index
    for band in bands:
        thr = float(band["threshold"])
        band["test_volume"] = float((n2_test_alert >= thr).mean())
        fa_ne = first_alarm_hours(ctx.test_ordered, ctx.news2_test, thr)
        band["nonevent_stay_burden"] = \
            float(fa_ne.reindex(nonevent_ids).notna().mean())
        band["races"] = {}
        for name, (s_val, s_test) in (("ensemble", (ens_val, ens_test)),
                                      ("s42", (sv[i42], st[i42]))):
            thr_m = float(np.quantile(s_val[fm][af], 1.0 - band["val_volume"]))
            diffs = paired_first_alarm_diffs(ctx.test_ordered, ctx.news2_test,
                                             s_test, thr, thr_m)
            r = race_anatomy(diffs["diff"])
            r.update({k: v for k, v in summarize_paired(
                diffs, ctx.stay_to_patient, reps=args.reps).items()
                if k in ("median_ci", "win_rate", "win_rate_ci")})
            r["thr_model"] = thr_m
            r["news2_standing_share"] = standing_alarm_share(
                ctx.test_ordered, ctx.news2_test, thr, diffs["stay_id"])
            band["races"][name] = r
    print(f"[bands {[b['threshold'] for b in bands]}] {time.time()-t0:.0f}s")

    # 3. anchors at the 2% NEWS2-matched operating point -----------------------
    nf = ctx.news2_val[fm]
    anchor_thrs = sorted({5.0, 7.0} | {float(b["threshold"]) for b in bands})
    anchors = {}
    for name, (s_val, s_test) in (("s42", (sv[i42], st[i42])),
                                  ("ensemble", (ens_val, ens_test))):
        _, thr_pub, matched_pub = frozen_thresholds(nf, s_val[fm], af, 0.02)
        anchors[name] = {
            "thr_model": thr_pub, "matched": matched_pub,
            "bands": [band_anchor_stats(ctx.test_ordered, s_test, thr_pub, t)
                      for t in anchor_thrs]}
    pre_evt = [pre_event_news2(ctx.test_ordered, w) for w in (4, 6)]

    out = {
        "seeds": seeds, "budget": args.budget, "reps": args.reps,
        "auprc_reps": args.auprc_reps, "tail_cutoff": args.tail_cutoff,
        "caches": {"two_head": args.cache, "lit_gbm": args.lit_cache},
        "ecart_auprc": ecart_auprc,
        "per_seed": per_seed, "seed_level": level, "pooled": pooled_r,
        "ensemble": ens, "bands": bands, "anchors": anchors,
        "pre_event_news2": pre_evt,
    }
    with open(args.out, "w") as fh:
        json.dump(_jsonable(out), fh, indent=1)
    print(f"[report] -> {args.out}  ({time.time()-t0:.0f}s)")
    print_markdown(out)


def print_markdown(o):
    seeds, b = o["seeds"], o["budget"]
    print(f"\n| seed | AUPRC | delta vs eCART [95% CI] | wins : ties : losses "
          f"| median | earlier when they differ | sign p |")
    print("|---|---|---|---|---|---|---|")
    for s in seeds:
        r = o["per_seed"][str(s)]
        print(f"| {s} | {r['auprc']:.4f} | {r['delta_auprc_vs_ecart']:+.4f} "
              f"[{r['delta_auprc_ci'][0]:+.4f}, {r['delta_auprc_ci'][1]:+.4f}] "
              f"| {r['wins']} : {r['ties']} : {r['losses']} "
              f"| {r['median']:+.1f} [{r['median_ci'][0]:+.0f}, {r['median_ci'][1]:+.0f}] "
              f"| {100*r['win_given_discordant']:.0f}% | {r['sign_p']:.1e} |")
    L = o["seed_level"]
    print(f"\nseed-level ({len(seeds)} seeds): delta AUPRC "
          f"{L['delta_auprc_vs_ecart']['mean']:+.4f} "
          f"[{L['delta_auprc_vs_ecart']['ci'][0]:+.4f}, "
          f"{L['delta_auprc_vs_ecart']['ci'][1]:+.4f}] "
          f"(per-seed CI clear of 0 in {L['n_delta_ci_clear']}/{len(seeds)}); "
          f"median lead {L['median']['mean']:+.2f} h "
          f"[{L['median']['ci'][0]:+.2f}, {L['median']['ci'][1]:+.2f}]; "
          f"earlier-when-differ {100*L['win_given_discordant']['mean']:.1f}% "
          f"[{100*L['win_given_discordant']['ci'][0]:.1f}, "
          f"{100*L['win_given_discordant']['ci'][1]:.1f}]; "
          f"wins > losses in {L['n_wins_exceed_losses']}/{len(seeds)}, "
          f"sign p < 0.05 in {L['n_sign_p_below_05']}/{len(seeds)}")
    P = o["pooled"]
    print(f"pooled: n {P['n']} median {P['median']:+.1f} "
          f"[{P['median_ci'][0]:+.1f}, {P['median_ci'][1]:+.1f}] "
          f"earlier-when-differ {100*P['win_given_discordant']:.1f}% "
          f"sign p {P['sign_p']:.1e}")
    E = o["ensemble"]
    print(f"\nensemble: AUPRC {E['auprc']:.4f} (eCART {o['ecart_auprc']:.4f}, "
          f"delta {E['delta_auprc_vs_ecart']:+.4f} "
          f"[{E['delta_auprc_ci'][0]:+.4f}, {E['delta_auprc_ci'][1]:+.4f}])")
    for bud, r in E["budgets"].items():
        t = r["tail"]
        print(f"  @{float(bud):.0%}: {r['wins']} : {r['ties']} : {r['losses']}, "
              f"median {r['median']:+.1f} [{r['median_ci'][0]:+.1f}, "
              f"{r['median_ci'][1]:+.1f}], earlier-when-differ "
              f"{100*r['win_given_discordant']:.1f}%, sign p {r['sign_p']:.1e}; "
              f"tail>= {o['tail_cutoff']:.0f}h: losses {t['loss']['n']} "
              f"({100*t['loss']['share_from_first_hour']:.0f}% standing eCART) "
              f"wins {t['win']['n']} "
              f"({100*t['win']['share_from_first_hour']:.0f}% standing model)")
    print(f"\n| band | thr | freeze vol | test vol | non-event burden "
          f"| ens median | earlier-when-differ | NEWS2 standing |")
    print("|---|---|---|---|---|---|---|---|")
    for band in o["bands"]:
        r = band["races"]["ensemble"]
        print(f"| ~P{band['percentile']} | >={band['threshold']} "
              f"| {band['val_volume']:.1%} | {band['test_volume']:.1%} "
              f"| {band['nonevent_stay_burden']:.1%} "
              f"| {r['median']:+.1f} [{r['median_ci'][0]:+.0f}, {r['median_ci'][1]:+.0f}] "
              f"| {100*r['win_given_discordant']:.0f}% (n {r['n']}) "
              f"| {100*r['news2_standing_share']:.0f}% |")
    print(f"\nanchors at the 2% NEWS2-matched operating point:")
    for name, a in o["anchors"].items():
        rows = "; ".join(
            f">={x['band_thr']:.0f}: before-band {100*x['share_before_stay_ever_in_band']:.0f}%, "
            f"never-pre-event {100*x['share_never_in_band_pre_event']:.0f}%, "
            f"gap {x['gap_median_h']:.1f}h (n {x['n_crossing_later']})"
            for x in a["bands"])
        print(f"  {name} (thr {a['thr_model']:.4f}, n_caught "
              f"{a['bands'][0]['n_caught']}): {rows}")
    for w in o["pre_event_news2"]:
        print(f"NEWS2 final {w['window_h']}h pre-event: mean {w['mean']:.2f}, "
              f"median {w['median']:.1f}")


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
