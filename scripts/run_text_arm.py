"""Run the text-passing ablation and compare it against the latent orchestrator.

    python scripts/run_text_arm.py --stays 4 --smoke      # tiny end-to-end check
    python scripts/run_text_arm.py --stays 150            # real subset

Pipeline: rebuild the three agents' calibrated risks and NEWS2 sub-scores ->
render deterministic templated summaries -> ask a local Llama-3.1-8B for a fused
risk per patient-hour (cached, resumable) -> isotonic-calibrate on val -> report
against the latent orchestrator and NEWS2 on identical rows.

Sampling is by STAY, not by row, so trajectories stay intact and the lead-time
analysis remains valid. Event-bearing stays are kept preferentially in the val
slice so the isotonic calibrator has positives to work with.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

from agents.base import AgentOutput  # noqa: E402
from agents.labs_agent import build_lab_features, load_labs  # noqa: E402
from baselines_mimic import (  # noqa: E402
    _false_alarm_stay_rate, _stay_series, load_splits, paired_first_alarm,
)
from metrics import auprc, auroc, ece, lead_time  # noqa: E402
from mimic_cohort import EVENT_TYPES  # noqa: E402
from text_arm.llm_orchestrator import DEFAULT_MODEL, generate_risks  # noqa: E402
from text_arm.summaries import build_summaries  # noqa: E402

LABEL = "label_24h"
EMB = REPO / "data" / "cache" / "embeddings"
CACHE = REPO / "data" / "cache" / "text_arm_risks.jsonl"


def load_agent_risks(split):
    """Each agent's calibrated risk + the NEWS2 sub-score latent, by key."""
    outs = {n: AgentOutput.load(EMB / f"{n}_{split}.npz")
            for n in ("vitals_tcn", "news2", "labs")}
    keys = outs["vitals_tcn"].keys
    risks = {n: o.risk.astype(float) for n, o in outs.items()}
    return keys, risks, outs["news2"].latent


def sample_stays(df, n_stays, seed=42):
    """Sample whole stays, keeping every event-bearing stay we can.

    Events are what both AUPRC and lead-time depend on, and at 4.74% prevalence
    a naive row sample would leave too few. Sampling whole stays also keeps each
    trajectory contiguous, which the LLM window and the lead-time metric need.
    """
    rng = np.random.default_rng(seed)
    per_stay = df.groupby("stay_id")[LABEL].max()
    event_stays = per_stay[per_stay == 1].index.to_numpy()
    quiet_stays = per_stay[per_stay == 0].index.to_numpy()
    rng.shuffle(event_stays)
    rng.shuffle(quiet_stays)

    # keep the natural stay-level event rate so prevalence is not distorted
    frac_event = len(event_stays) / max(len(per_stay), 1)
    n_event = min(len(event_stays), max(1, int(round(n_stays * frac_event))))
    n_quiet = min(len(quiet_stays), n_stays - n_event)
    chosen = np.concatenate([event_stays[:n_event], quiet_stays[:n_quiet]])
    return df[df.stay_id.isin(set(chosen))].copy()


def build_text_frame(df, split):
    """Templated summaries for every row of ``df`` (a slice of ``split``)."""
    keys, risks, news2_latent = load_agent_risks(split)
    index = pd.MultiIndex.from_arrays([keys.stay_id, keys.hour])
    ordered = df.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    want = pd.MultiIndex.from_arrays([ordered.stay_id, ordered.hour])
    pos = index.get_indexer(want)
    if (pos < 0).any():
        raise ValueError("rows missing from the agent caches")

    lab_feats = build_lab_features(ordered[["stay_id", "hour"]], load_labs())
    return build_summaries(
        ordered, lab_feats, news2_latent[pos],
        {n: risks[n][pos] for n in risks},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stays", type=int, default=150,
                    help="test-split stays to evaluate on")
    ap.add_argument("--val-stays", type=int, default=60,
                    help="val stays used to fit the isotonic calibrator")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run: print prompts and stop early")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap LLM calls this run (checkpointing resumes later)")
    args = ap.parse_args()

    splits = load_splits()
    val_sample = sample_stays(splits["val"], args.val_stays, seed=7)
    test_sample = sample_stays(splits["test"], args.stays, seed=42)

    print(f"[subset] val : {val_sample.stay_id.nunique():,} stays, "
          f"{len(val_sample):,} patient-hours, "
          f"{100*val_sample[LABEL].mean():.2f}% positive")
    print(f"[subset] test: {test_sample.stay_id.nunique():,} stays, "
          f"{len(test_sample):,} patient-hours, "
          f"{100*test_sample[LABEL].mean():.2f}% positive "
          f"({int(test_sample[LABEL].sum()):,} positive hours)")

    t0 = time.time()
    val_text = build_text_frame(val_sample, "val")
    test_text = build_text_frame(test_sample, "test")
    print(f"[text] rendered {len(val_text)+len(test_text):,} summaries "
          f"in {time.time()-t0:.1f}s")

    if args.smoke:
        from text_arm.llm_orchestrator import build_prompt, windows_for
        print("\n--- example rendered summaries ---")
        for col in ("vitals_text", "labs_text", "news2_text"):
            print(f"  {test_text.iloc[-1][col]}")
        print("\n--- example prompt (last row) ---")
        _, hour, window = list(windows_for(test_text))[-1]
        print(build_prompt(window))
        print("---")

    # --- generate (cached + checkpointed) --------------------------------
    t0 = time.time()
    val_cache = generate_risks(val_text, CACHE, model=args.model, limit=args.limit)
    test_cache = generate_risks(test_text, CACHE, model=args.model, limit=args.limit)
    elapsed = time.time() - t0
    total_rows = len(val_text) + len(test_text)
    print(f"[llm] {elapsed:.0f}s  ({elapsed/max(total_rows,1):.2f}s per patient-hour)")

    val_keys = list(zip(val_text.stay_id, val_text.hour))
    test_keys = list(zip(test_text.stay_id, test_text.hour))
    val_raw = val_cache.risks_for(val_keys)
    test_raw = test_cache.risks_for(test_keys)

    done = np.isfinite(test_raw).sum()
    print(f"[llm] test rows with a usable risk: {done:,}/{len(test_raw):,}")
    if done < len(test_raw):
        print("      (incomplete -- re-run to resume; reporting on what exists)")

    # --- evaluate on rows the LLM actually scored ------------------------
    ok = np.isfinite(test_raw)
    test_ordered = test_sample.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    evaluated = test_ordered[ok].reset_index(drop=True)
    y = evaluated[LABEL].to_numpy(dtype=float)
    text_raw = test_raw[ok]
    if len(np.unique(y)) < 2:
        print("\n[stop] the evaluated subset has only one class -- widen --stays")
        return

    val_ok = np.isfinite(val_raw)
    val_y = val_sample.sort_values(["stay_id", "hour"]).reset_index(drop=True)[
        LABEL].to_numpy(dtype=float)[val_ok]
    if len(np.unique(val_y)) > 1:
        cal = IsotonicRegression(out_of_bounds="clip").fit(val_raw[val_ok], val_y)
        text_cal = cal.transform(text_raw)
    else:
        print("[warn] val subset has one class; skipping calibration")
        text_cal = text_raw

    # comparators on the SAME rows
    latent = AgentOutput.load(EMB / "orchestrator_test.npz")
    lat_idx = pd.MultiIndex.from_arrays([latent.keys.stay_id, latent.keys.hour])
    pos = lat_idx.get_indexer(
        pd.MultiIndex.from_arrays([evaluated.stay_id, evaluated.hour]))
    latent_risk = latent.risk[pos].astype(float)
    news2 = evaluated["news2"].to_numpy(dtype=float)

    types = list(EVENT_TYPES)
    print(f"\n{'='*96}")
    print(f"TEXT-PASSING ABLATION vs LATENT FUSION -- {len(y):,} patient-hours, "
          f"{evaluated.stay_id.nunique():,} stays, {100*y.mean():.2f}% positive")
    print(f"model: {args.model} (local), window=4h, temperature=0")
    print("="*96)
    hdr = (f"  {'arm':<22}{'AUPRC raw':>10}{'AUPRC cal':>10}{'AUROC':>8}"
           f"{'ECE raw':>9}{'ECE cal':>9}   " + "".join(f"{t:>8}" for t in types))
    print(hdr); print("  " + "-"*(len(hdr)-2))
    def per_type_auprc(scores):
        """AUPRC per event type; nan when a type has no positives in the subset."""
        out = {}
        for t in types:
            yt = evaluated[f"{LABEL}_{t}"].to_numpy(dtype=float)
            out[t] = auprc(yt, scores) if yt.sum() > 0 else float("nan")
        return out

    rows = [("TEXT (LLM fusion)", text_raw, text_cal),
            ("LATENT (orchestrator)", latent_risk, latent_risk),
            ("NEWS2", news2, news2)]
    for name, raw, calv in rows:
        per_type = per_type_auprc(raw)
        print(f"  {name:<22}{auprc(y, raw):>10.3f}{auprc(y, calv):>10.3f}"
              f"{auroc(y, raw):>8.3f}"
              f"{ece(y, np.clip(raw, 0, 1)):>9.4f}{ece(y, np.clip(calv, 0, 1)):>9.4f}   "
              + "".join(f"{per_type[t]:>8.3f}" for t in types))
    print(f"  {'(random)':<22}{y.mean():>10.3f}{y.mean():>10.3f}{0.5:>8.3f}")

    # --- lead-time + paired comparison -----------------------------------
    series = {n: _stay_series(evaluated, s) for n, s in
              (("TEXT", text_raw), ("LATENT", latent_risk), ("NEWS2", news2))}
    print("\n  Lead-time at matched alert budgets:")
    sh = (f"  {'budget':>7}  {'arm':<7}{'actual':>8}{'caught':>8}{'falarm':>8}"
          f"{'n':>6}    lead p25/p50/p75 (h)")
    print(sh); print("  " + "-"*(len(sh)-2))
    for rate in (0.01, 0.02, 0.05):
        for name, s in (("NEWS2", news2), ("TEXT", text_raw), ("LATENT", latent_risk)):
            thr = float(np.quantile(s, 1.0 - rate))
            r = lead_time(*series[name][:2], thr, times=series[name][2])
            band = ("n/a" if r["n_caught"] == 0 else
                    f"{r['lead_p25']:>4.0f} /{r['median_lead_time']:>4.0f} /"
                    f"{r['lead_p75']:>4.0f}")
            print(f"  {f'{rate*100:.0f}%':>7}  {name:<7}"
                  f"{100*float((s>=thr).mean()):>7.1f}%{r['caught_in_time_rate']:>8.2f}"
                  f"{_false_alarm_stay_rate(series[name], thr):>8.2f}"
                  f"{r['n_caught']:>6}    {band}")

    print("\n  PAIRED first-alarm, volume-matched (+ = second arm earlier):")
    ph = (f"  {'budget':>7}  {'A vs B':<16}{'n_pair':>7}{'median':>8}{'B<':>6}"
          f"{'A<':>6}{'tie':>5}{'Bwin':>7}{'sign_p':>10}")
    print(ph); print("  " + "-"*(len(ph)-2))
    for rate in (0.01, 0.02, 0.05):
        for a_name, a, b_name, b in (("NEWS2", news2, "TEXT", text_raw),
                                     ("NEWS2", news2, "LATENT", latent_risk),
                                     ("TEXT", text_raw, "LATENT", latent_risk)):
            thr_a = float(np.quantile(a, 1.0 - rate))
            matched = float((a >= thr_a).mean())
            thr_b = float(np.quantile(b, 1.0 - matched))
            r = paired_first_alarm(evaluated, a, b, thr_a, thr_b)
            if r["n_paired"] == 0:
                continue
            print(f"  {f'{rate*100:.0f}%':>7}  {a_name+' vs '+b_name:<16}"
                  f"{r['n_paired']:>7}{r['median_diff']:>8.1f}"
                  f"{r['n_gbm_earlier']:>6}{r['n_news2_earlier']:>6}{r['n_tie']:>5}"
                  f"{100*r['frac_gbm_earlier']:>6.0f}%{r['sign_p']:>10.2g}")


if __name__ == "__main__":
    main()
