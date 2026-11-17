"""Operating-point results for the paper: budget sweep + volume-matched NEWS2 race.

Two results the paper needs that no committed write-up holds yet:

1. **Budget sweep.** Share of deteriorating stays caught at seven alarm budgets
   (0.5-10% of patient-hours), for the two-head flagship over all ten seeds and
   for the three published-spec comparators. Each model's threshold is its own
   (1 - b) quantile on the V_freeze half (the demo bundle's sweep rule), and the
   REALIZED test alarm rate is reported next to every catch rate. Also reports
   mid-stay catches (first alarm after the stay's first scored hour), because
   first-hour flags inflate catch rates, and catches by an alarm AFTER the
   first scored hour (first-hour alarms ignored), the fair version of that check.

2. **Volume-matched lead over NEWS2.** At a nominal 2% budget NEWS2 fires on
   ~2.3% of test hours while the flagship, frozen at NEWS2's V_freeze volume,
   fires on ~2.8% -- so the published lead-vs-NEWS2 races compare unequal
   alarm volumes. Here the flagship's threshold is re-set to fire on exactly
   NEWS2's realized test volume, then the paired first-alarm race is re-run.
   Test labels are never read for thresholding, but test SCORES set the
   matched threshold, so this is a diagnostic for the volume confound (as in
   T27), not a deployable protocol. The protocol (V_freeze-frozen) race is
   reported alongside for comparison. The same equal-volume race is run against
   each published comparator at the 2% NEWS2-matched volume.

Inputs (both gitignored, under data/cache/): two_head.npz, lit_gbm_scores.npz.
Output: aggregate-only JSON at data/cache/paper_operating_points.json.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/paper_operating_points.py [--reps 1000]
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from baselines_mimic import load_splits  # noqa: E402
from demo_bundle import first_alarm_table  # noqa: E402
from fair_capability import (  # noqa: E402
    frozen_thresholds,
    paired_first_alarm_diffs,
    summarize_paired,
)
from fusion_common import FusionContext  # noqa: E402

CACHE = REPO / "data" / "cache"
OUT = CACHE / "paper_operating_points.json"
SWEEP_BUDGETS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
RACE_BUDGETS = (0.01, 0.02, 0.05)
COMPARATORS = ("ecart", "yaib", "sadanandan")

# Published values this script must reproduce before anything new is trusted
# (seed 42; demo bundle h2h block and the "Sentinel in Numbers" sweep).
EXPECTED_MATCHED_CATCH = {"sentinel": 0.6545, "ecart": 0.5357, "yaib": 0.5182}
EXPECTED_SWEEP_S42 = {0.005: 0.3468, 0.10: 0.8682}


# --- loading -----------------------------------------------------------------

def load_scores(ctx):
    """Flagship scores per seed and comparator scores, alignment-checked."""
    z = np.load(CACHE / "two_head.npz", allow_pickle=False)
    if not (z["stay_id_test"] == ctx.test_ordered["stay_id"].to_numpy()).all():
        raise RuntimeError("two_head.npz is not aligned with the test rows")
    seeds = sorted(int(k.split("_s")[1]) for k in z.files if k.startswith("test_s"))
    flagship = {s: {"val": z[f"val_s{s}"].astype(float),
                    "test": z[f"test_s{s}"].astype(float)} for s in seeds}

    lit = np.load(CACHE / "lit_gbm_scores.npz", allow_pickle=False)
    for split, frame in (("val", ctx.val_ordered), ("test", ctx.test_ordered)):
        if not (lit[f"{split}_stay_id"] == frame["stay_id"].to_numpy()).all():
            raise RuntimeError(f"lit_gbm_scores.npz is not aligned with {split}")
    comps = {n: {"val": lit[f"val_{n}"].astype(float),
                 "test": lit[f"test_{n}"].astype(float)} for n in COMPARATORS}
    return seeds, flagship, comps


# --- thresholds ----------------------------------------------------------------

def freeze_quantile(ctx, val_scores, share):
    """Threshold that fires on ``share`` of V_freeze's alertable hours."""
    fm = ctx.freeze_mask
    v = val_scores[fm][ctx.alertable_val[fm]]
    return float(np.quantile(v, 1.0 - share))


def test_rate(ctx, test_scores, thr):
    return float((test_scores[ctx.alertable_test] >= thr).mean())


# --- catch rates -------------------------------------------------------------

def catch_stats(table, name):
    """Catch, early (>= 6 h), mid-stay and quiet-stay false-alarm shares."""
    ev = table[table["event_h"].notna()]
    qt = table[table["event_h"].isna()]
    fa = ev[f"fa_{name}"]
    caught = fa.notna() & (fa <= ev["event_h"])
    return {
        "catch": float(caught.mean()),
        "catch_6h": float((fa.notna() & (fa <= ev["event_h"] - 6.0)).mean()),
        "catch_midstay": float((caught & (fa > ev["h0"])).mean()),
        "quiet_fp": float(qt[f"fa_{name}"].notna().mean()),
        "n_event_stays": int(len(ev)),
    }


def after_first_hour_catch(ctx, scores, thr):
    """Share of event stays alarmed at some hour after the stay's first scored
    hour and at or before the event. First-hour alarms are ignored, so a model
    that flags many stays at hour 6 gets no credit for those flags here."""
    t = ctx.test_ordered
    keep = (t["hour"] > t.groupby("stay_id")["hour"].transform("min")).to_numpy()
    table = first_alarm_table(t[keep].reset_index(drop=True),
                              {"m": np.asarray(scores)[keep]}, {"m": thr})
    ev = table[table["event_h"].notna()]
    return float((ev["fa_m"].notna() & (ev["fa_m"] <= ev["event_h"])).mean())


def sweep(ctx, seeds, flagship, comps):
    """Catch rates at each budget: flagship per seed, comparators once."""
    rows = []
    for b in SWEEP_BUDGETS:
        comp_thr = {n: freeze_quantile(ctx, comps[n]["val"], b) for n in COMPARATORS}
        row = {"budget": b, "comparators": {}, "sentinel_by_seed": {}}
        table = first_alarm_table(
            ctx.test_ordered, {n: comps[n]["test"] for n in COMPARATORS}, comp_thr)
        for n in COMPARATORS:
            row["comparators"][n] = {**catch_stats(table, n),
                                     "catch_after_h0": after_first_hour_catch(ctx, comps[n]["test"], comp_thr[n]),
                                     "test_alarm_rate": test_rate(ctx, comps[n]["test"], comp_thr[n])}
        for s in seeds:
            thr = freeze_quantile(ctx, flagship[s]["val"], b)
            t = first_alarm_table(ctx.test_ordered, {"sentinel": flagship[s]["test"]},
                                  {"sentinel": thr})
            row["sentinel_by_seed"][s] = {**catch_stats(t, "sentinel"),
                                          "catch_after_h0": after_first_hour_catch(ctx, flagship[s]["test"], thr),
                                          "test_alarm_rate": test_rate(ctx, flagship[s]["test"], thr)}
        rows.append(row)
    return rows


def matched_volume_check(ctx, flagship, comps):
    """Reproduce the published seed-42 head-to-head at NEWS2's V_freeze volume."""
    fm = ctx.freeze_mask
    _, thr_m, matched = frozen_thresholds(
        ctx.news2_val[fm], flagship[42]["val"][fm], ctx.alertable_val[fm], 0.02)
    thr = {"sentinel": thr_m,
           **{n: freeze_quantile(ctx, comps[n]["val"], matched) for n in ("ecart", "yaib")}}
    scores = {"sentinel": flagship[42]["test"],
              **{n: comps[n]["test"] for n in ("ecart", "yaib")}}
    table = first_alarm_table(ctx.test_ordered, scores, thr)
    return {n: catch_stats(table, n)["catch"] for n in thr}, matched


# --- NEWS2 race -----------------------------------------------------------------

def race(ctx, opp_test, s_test, thr_opp, thr_m, reps):
    """Paired first-alarm race; diff > 0 means the flagship alarmed first."""
    diffs = paired_first_alarm_diffs(ctx.test_ordered, opp_test, s_test,
                                     thr_opp, thr_m)
    out = summarize_paired(diffs, ctx.stay_to_patient, reps=reps)
    d = diffs["diff"].to_numpy(dtype=float)
    disc = int((d != 0).sum())
    out["earlier_when_differ"] = float((d > 0).sum() / disc) if disc else float("nan")
    return out


def news2_races(ctx, seeds, flagship, reps):
    """Per seed and budget: the protocol race and the volume-matched race."""
    fm = ctx.freeze_mask
    results = []
    for b in RACE_BUDGETS:
        for s in seeds:
            thr_n2, thr_m, _ = frozen_thresholds(
                ctx.news2_val[fm], flagship[s]["val"][fm], ctx.alertable_val[fm], b)
            s_test = flagship[s]["test"]
            n2_rate = test_rate(ctx, ctx.news2_test, thr_n2)
            # Flagship threshold that fires on exactly NEWS2's realized test volume.
            thr_vm = float(np.quantile(s_test[ctx.alertable_test], 1.0 - n2_rate))
            results.append({
                "budget": b, "seed": s, "news2_threshold": thr_n2,
                "news2_test_rate": n2_rate,
                "protocol": {"model_test_rate": test_rate(ctx, s_test, thr_m),
                             **race(ctx, ctx.news2_test, s_test, thr_n2, thr_m, reps)},
                "volume_matched": {"model_test_rate": test_rate(ctx, s_test, thr_vm),
                                   **race(ctx, ctx.news2_test, s_test, thr_n2, thr_vm, reps)},
            })
    return results


def comparator_races(ctx, seeds, flagship, comps, reps, budget=0.02):
    """Race the flagship against each comparator at the NEWS2-matched volume:
    protocol thresholds (both frozen on V_freeze) and equal realized test volume."""
    fm = ctx.freeze_mask
    results = []
    for n in COMPARATORS:
        for s in seeds:
            _, thr_m, matched = frozen_thresholds(
                ctx.news2_val[fm], flagship[s]["val"][fm], ctx.alertable_val[fm], budget)
            thr_c = freeze_quantile(ctx, comps[n]["val"], matched)
            c_rate = test_rate(ctx, comps[n]["test"], thr_c)
            s_test = flagship[s]["test"]
            thr_vm = float(np.quantile(s_test[ctx.alertable_test], 1.0 - c_rate))
            results.append({
                "comparator": n, "seed": s, "comparator_test_rate": c_rate,
                "protocol": {"model_test_rate": test_rate(ctx, s_test, thr_m),
                             **race(ctx, comps[n]["test"], s_test, thr_c, thr_m, reps)},
                "volume_matched": {"model_test_rate": test_rate(ctx, s_test, thr_vm),
                                   **race(ctx, comps[n]["test"], s_test, thr_c, thr_vm, reps)},
            })
    return results


def print_races(races, key, label):
    print(f"\n{label}: first-when-differ (mean over seeds) | seeds with sign test p<0.05")
    groups = sorted({r[key] for r in races}, key=str)
    for g in groups:
        for tag in ("protocol", "volume_matched"):
            r = [x[tag] for x in races if x[key] == g]
            sig = sum(1 for x in r if x["sign_p"] < 0.05 and x["earlier_when_differ"] > 0.5)
            print(f"  {str(g):<10} {tag:<15} model alarm {np.mean([x['model_test_rate'] for x in r]):.2%} | "
                  f"median lead {np.mean([x['median'] for x in r]):+.2f} h | "
                  f"first-when-differ {np.mean([x['earlier_when_differ'] for x in r]):.0%} | {sig}/{len(r)}")


# --- early catches (T33 / L1) ----------------------------------------------------

EARLY_HOURS = 6.0
EARLY_OUT = CACHE / "paper_early_catches.json"


def stay_first_alarms(test_ordered, scores, thr, skip_first_hour=False):
    """One row per EVENT stay: patient, event hour, first alarm at or before it.

    ``first_alarm`` is NaN when the stay is not caught. With
    ``skip_first_hour`` an alarm at the stay's first scored hour does not count
    (the stay can still be caught by a later one).
    """
    t = test_ordered
    crossed = np.asarray(scores, dtype=float) >= thr
    hour = t["hour"].to_numpy(dtype=float)
    if skip_first_hour:
        crossed &= hour > t.groupby("stay_id")["hour"].transform("min").to_numpy(dtype=float)
    frame = pd.DataFrame({"stay_id": t["stay_id"].to_numpy(),
                          "patient_id": t["patient_id"].to_numpy(),
                          "event_h": t["event_h"].to_numpy(dtype=float),
                          "fa": np.where(crossed, hour, np.nan)})
    g = frame.groupby("stay_id", sort=True).agg(
        patient_id=("patient_id", "first"), event_h=("event_h", "first"),
        fa=("fa", "min")).reset_index()
    g = g[g["event_h"].notna()].reset_index(drop=True)
    g["first_alarm"] = g["fa"].where(g["fa"] <= g["event_h"])
    return g.drop(columns="fa")


def early_stats(table):
    """Catch, >= 6 h catch and warning time (median, IQR) among caught stays."""
    caught = table["first_alarm"].notna().to_numpy()
    warn = (table["event_h"] - table["first_alarm"]).to_numpy(dtype=float)
    early = caught & (warn >= EARLY_HOURS)
    w = warn[caught]
    return {"n_event_stays": int(len(table)), "catch": float(caught.mean()),
            "catch_6h": float(early.mean()),
            "warning_median": float(np.median(w)) if w.size else float("nan"),
            "warning_q25": float(np.percentile(w, 25)) if w.size else float("nan"),
            "warning_q75": float(np.percentile(w, 75)) if w.size else float("nan")}


def early_flags(table):
    warn = (table["event_h"] - table["first_alarm"]).to_numpy(dtype=float)
    return (table["first_alarm"].notna().to_numpy() & (warn >= EARLY_HOURS)).astype(float)


def early_catches(ctx, seeds, flagship, comps, reps):
    """>= 6 h catches and warning times, protocol and equal realized volume.

    Comparators sit at their own V_freeze (1 - b) quantile. The flagship is
    reported at its own protocol threshold and, as the volume diagnostic,
    re-thresholded on TEST score quantiles to fire on exactly each
    comparator's realized test volume. Paired CIs resample patients, shared by
    every model at a budget; the pooled difference is the ten-seed mean.
    """
    from baselines_fair import bootstrap_indices
    from config import SEED

    t = ctx.test_ordered
    out = []
    for b in SWEEP_BUDGETS:
        row = {"budget": b, "comparators": {}, "flagship_protocol": {},
               "flagship_equal_volume": {}, "paired_6h_equal_volume": {}}
        for skip in (False, True):
            tag = "no_first_hour" if skip else "all_alarms"
            flags_c, flags_f, comp_rates = {}, {}, {}
            for n, c in comps.items():
                thr = freeze_quantile(ctx, c["val"], b)
                comp_rates[n] = test_rate(ctx, c["test"], thr)
                tab = stay_first_alarms(t, c["test"], thr, skip)
                row["comparators"].setdefault(n, {})[tag] = {
                    **early_stats(tab), "test_alarm_rate": comp_rates[n]}
                flags_c[n] = early_flags(tab)
                patients = tab["patient_id"].to_numpy()
            prot, eqv = [], {n: [] for n in comps}
            for s in seeds:
                f = flagship[s]
                thr = freeze_quantile(ctx, f["val"], b)
                prot.append({**early_stats(stay_first_alarms(t, f["test"], thr, skip)),
                             "test_alarm_rate": test_rate(ctx, f["test"], thr)})
                for n in comps:
                    thr_vm = float(np.quantile(f["test"][ctx.alertable_test],
                                               1.0 - comp_rates[n]))
                    tab = stay_first_alarms(t, f["test"], thr_vm, skip)
                    eqv[n].append({**early_stats(tab),
                                   "test_alarm_rate": test_rate(ctx, f["test"], thr_vm)})
                    flags_f[(n, s)] = early_flags(tab)
            keys = ("catch", "catch_6h", "warning_median", "warning_q25",
                    "warning_q75", "test_alarm_rate")
            row["flagship_protocol"][tag] = {
                **{k: float(np.mean([p[k] for p in prot])) for k in keys},
                "catch_6h_min": float(min(p["catch_6h"] for p in prot)),
                "catch_6h_max": float(max(p["catch_6h"] for p in prot))}
            row["flagship_equal_volume"][tag] = {
                n: {**{k: float(np.mean([p[k] for p in eqv[n]])) for k in keys},
                    "catch_6h_min": float(min(p["catch_6h"] for p in eqv[n])),
                    "seeds_ahead_6h": int(sum(p["catch_6h"] > row["comparators"][n][tag]["catch_6h"]
                                              for p in eqv[n]))}
                for n in comps}
            # Paired >= 6 h catch difference at equal volume, pooled over seeds.
            diffs = {n: [] for n in comps}
            for idx in bootstrap_indices(patients, reps, SEED):
                for n in comps:
                    base = flags_c[n][idx].mean()
                    diffs[n].append(np.mean([flags_f[(n, s)][idx].mean() - base
                                             for s in seeds]))
            row["paired_6h_equal_volume"][tag] = {
                n: {"delta": float(np.mean([flags_f[(n, s)].mean() for s in seeds])
                                   - flags_c[n].mean()),
                    "ci": [float(np.percentile(diffs[n], 2.5)),
                           float(np.percentile(diffs[n], 97.5))]}
                for n in comps}
        out.append(row)
        best = max(comps, key=lambda n: row["comparators"][n]["all_alarms"]["catch_6h"])
        e = row["flagship_equal_volume"]["all_alarms"][best]
        d = row["paired_6h_equal_volume"]["all_alarms"][best]
        print(f"[early] {b:>5.1%}: best comparator {best} >=6h "
              f"{row['comparators'][best]['all_alarms']['catch_6h']:.1%} | flagship at its "
              f"volume {e['catch_6h']:.1%} | d {d['delta']:+.1%} "
              f"[{d['ci'][0]:+.1%}, {d['ci'][1]:+.1%}] | warning median "
              f"{e['warning_median']:.1f} h vs "
              f"{row['comparators'][best]['all_alarms']['warning_median']:.1f} h")
    return out


def early_verdict(rows, names):
    """Pass rule (T33): at equal realized volume the flagship's >= 6 h catch
    beats every comparator in ``names`` at every budget (ten-seed mean), and
    against the best of them the pooled paired CI is above zero at 1/2/5%."""
    beats, ci_ok = {}, {}
    for r in rows:
        b = r["budget"]
        beats[b] = all(r["flagship_equal_volume"]["all_alarms"][n]["catch_6h"]
                       > r["comparators"][n]["all_alarms"]["catch_6h"] for n in names)
        if b in RACE_BUDGETS:
            best = max(names, key=lambda n: r["comparators"][n]["all_alarms"]["catch_6h"])
            ci_ok[b] = {"best": best,
                        "ci_above_zero": r["paired_6h_equal_volume"]["all_alarms"][best]["ci"][0] > 0}
    passed = all(beats.values()) and all(v["ci_above_zero"] for v in ci_ok.values())
    return {"beats_every_budget": beats, "ci_at_race_budgets": ci_ok, "pass": passed}


def load_allfeat(ctx):
    """The T30 all-features GBM (seed 42), alignment-checked, if cached."""
    path = CACHE / "allfeat_gbm_scores.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    for split, frame in (("val", ctx.val_ordered), ("test", ctx.test_ordered)):
        if not (z[f"{split}_stay_id"] == frame["stay_id"].to_numpy()).all():
            raise RuntimeError(f"allfeat_gbm_scores.npz is not aligned with {split}")
    return {"val": z["val_allfeat_s42"].astype(float),
            "test": z["test_allfeat_s42"].astype(float)}


# --- reporting -----------------------------------------------------------------

def print_sweep(rows):
    print("\nBudget sweep: catch % (realized test alarm %); Sentinel = mean over seeds [min, max]")
    for r in rows:
        sent = list(r["sentinel_by_seed"].values())
        c = [x["catch"] for x in sent]
        rate = np.mean([x["test_alarm_rate"] for x in sent])
        mid = np.mean([x["catch_after_h0"] for x in sent])
        comps = "  ".join(
            f"{n} {v['catch']:.1%} ({v['test_alarm_rate']:.2%}) after-h0 {v['catch_after_h0']:.1%}"
            for n, v in r["comparators"].items())
        print(f"  {r['budget']:>5.1%}: Sentinel {np.mean(c):.1%} [{min(c):.1%}, {max(c):.1%}] "
              f"({rate:.2%}) after-h0 {mid:.1%} | {comps}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--early", action="store_true",
                    help="T33: >= 6 h catches and warning times (after the gates); "
                         "writes data/cache/paper_early_catches.json")
    args = ap.parse_args()

    ctx = FusionContext(load_splits())
    seeds, flagship, comps = load_scores(ctx)
    print(f"[load] flagship seeds {seeds}; comparators {list(COMPARATORS)}")

    # Gate: reproduce published numbers first.
    got, matched = matched_volume_check(ctx, flagship, comps)
    print(f"[gate] matched volume {matched:.4f}; catch " +
          ", ".join(f"{n} {v:.4f} (expect {EXPECTED_MATCHED_CATCH[n]})" for n, v in got.items()))
    for n, v in got.items():
        if abs(v - EXPECTED_MATCHED_CATCH[n]) > 5e-4:
            raise RuntimeError(f"head-to-head does not reproduce for {n}: {v:.4f}")

    rows = sweep(ctx, seeds, flagship, comps)
    for b, want in EXPECTED_SWEEP_S42.items():
        got_b = next(r for r in rows if r["budget"] == b)["sentinel_by_seed"][42]["catch"]
        print(f"[gate] sweep s42 at {b:.1%}: {got_b:.4f} (expect {want})")
        if abs(got_b - want) > 5e-4:
            raise RuntimeError(f"sweep does not reproduce at {b}: {got_b:.4f}")
    print_sweep(rows)

    if args.early:
        from demo_bundle import assert_aggregate_only
        allfeat = load_allfeat(ctx)
        early_comps = dict(comps)
        if allfeat is not None:
            early_comps["allfeat"] = allfeat
        early = early_catches(ctx, seeds, flagship, early_comps, args.reps)
        payload = {"test": "T33", "seeds": seeds, "reps": args.reps,
                   "early_hours": EARLY_HOURS, "rows": early,
                   "verdict_published": early_verdict(early, COMPARATORS),
                   "verdict_with_allfeat": (early_verdict(early, (*COMPARATORS, "allfeat"))
                                            if allfeat is not None else None),
                   "note": "aggregate-only; equal-volume rows set the flagship "
                           "threshold from test score quantiles (diagnostic)"}
        payload = json.loads(json.dumps(payload, default=float))
        assert_aggregate_only(payload)
        EARLY_OUT.write_text(json.dumps(payload, indent=1))
        for k in ("verdict_published", "verdict_with_allfeat"):
            if payload[k] is not None:
                print(f"[verdict] {k}: {'PASS' if payload[k]['pass'] else 'FAIL'} "
                      f"{payload[k]['ci_at_race_budgets']}")
        print(f"[saved] {EARLY_OUT}")
        return

    races = news2_races(ctx, seeds, flagship, args.reps)
    print_races(races, "budget", "Lead over NEWS2")
    comp_races = comparator_races(ctx, seeds, flagship, comps, args.reps)
    print_races(comp_races, "comparator", "Race vs comparators at the 2% NEWS2-matched volume")

    payload = {"seeds": seeds, "sweep": rows, "news2_races": races,
               "comparator_races": comp_races,
               "matched_volume_check": {"matched": matched, "catch": got},
               "note": "aggregate-only; volume-matched race thresholds use test "
                       "score quantiles (diagnostic, not protocol)"}
    OUT.write_text(json.dumps(payload, indent=1, default=float))
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
