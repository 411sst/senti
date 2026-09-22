"""Build the demo-dashboard bundle: one-time precompute, CPU except ollama.

    python scripts/build_demo_bundle.py                  # full build (~10 min)
    python scripts/build_demo_bundle.py --skip-llm       # no ollama needed
    python scripts/build_demo_bundle.py --refit          # rebuild the checkpoint
    python scripts/build_demo_bundle.py --pick-index 1   # swap the curated stays

Outputs, both under the gitignored data/ tree:

    data/cache/demo/bundle.json   patient-level -- rendered on this laptop only
    data/cache/demo/cohort.json   aggregate-only -- may feed the hosted leave-behind

plus models/orchestrator/all8_s42.pt the first time (the single-head ALL-8
refit that supplies attention weights; the two-head composite has no
checkpoint, but its test scores are cached and are THE risk the demo plots).

The fusion refit and scoring run at one CPU thread (house rule: CPU refits
drift with thread count); only the explanation cards touch the GPU, via
ollama. Stay ids are printed to the console and written to bundle.json only.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)                       # the loaders default to repo-relative paths
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from baselines_mimic import load_splits  # noqa: E402
from demo_bundle import (  # noqa: E402
    COMPARATOR_LABELS, COMPARATORS, assert_aggregate_only, budget_sweep_row,
    dumps_strict, first_alarm_hour, first_alarm_table, h2h_summary, lead_stats,
    select_demo_stays, stay_summary, t_head_from_composite, validate_bundle,
)
from explain.drivers import build_contexts  # noqa: E402
from explain.faithfulness import column_spans, restrict_bank  # noqa: E402
from explain.rationales import (  # noqa: E402
    SYSTEM_PROMPT, RationaleCache, build_prompt, generate_rationales,
)
from fair_capability import frozen_thresholds  # noqa: E402
from fusion_common import FusionContext  # noqa: E402
from icu_strata import assign_strata  # noqa: E402
from mimic_cohort import NEWS2_COMPONENTS, RAW_PATH, VITALS  # noqa: E402
from orchestrator.fusion import ORCHESTRATOR_CONFIGS, AgentBank, Orchestrator  # noqa: E402
from run_icu_strata import stay_frame  # noqa: E402
from standby import burden_spread  # noqa: E402

CACHE = REPO / "data" / "cache"
OUT_DIR = CACHE / "demo"
CKPT = REPO / "models" / "orchestrator" / "all8_s42.pt"
AGENTS = tuple(ORCHESTRATOR_CONFIGS["all-8"])
LLM = "gemma4:12b"
BUDGET = 0.02
SEED = 42

SCATTER_NAMES = {
    "NEWS2": "NEWS2",
    "FAIR+TRAJ (ours)": "Our 145-feature GBM",
    "two-head flagship s42": "Sentinel two-head",
    "lit:sadanandan": "Sadanandan 2026 XGB (published spec)",
    "lit:ecart": "eCART-style GBM (published spec)",
    "lit:yaib": "YAIB LightGBM (published spec)",
}

CAVEATS = [
    "The timing edge is seed-sensitive: lead at the 2% budget is +1.0 / +2.0 / "
    "+1.5 / 0.0 / 0.0 h across five training seeds (positive in 3 of 5). The "
    "ranking edge over the equal-information GBM holds on 5 of 5, CI clear of zero.",
    "Death arm: on death stays with no documented care-limitation decision "
    "(chart or discharge note) the lead is +0.5 h at 50% win — null. Early "
    "death-stay alarms largely track care patterns of decisions already made.",
    "Externally validated, retrospective: trained on MIMIC-IV (Beth Israel "
    "Deaconess) and scored zero-shot on eICU-CRD — 208 hospitals, 5.97M "
    "patient-hours — keeping 80% of its lift and beating all three published "
    "specs where charting density is matched (results_fusion.md T22, seed 42). "
    "Calibration does not transfer: thresholds must be re-frozen locally. "
    "NEWS2 is as charted; no outcome evidence — the lead is an alarm-time "
    "difference, not a mortality difference.",
]


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True,
                              cwd=REPO).stdout.strip()
    except Exception:
        return "unknown"


def load_or_fit(bank, splits, refit):
    if CKPT.exists() and not refit:
        orch = Orchestrator.load_checkpoint(CKPT, bank)
        print(f"[fusion] loaded {CKPT.relative_to(REPO)}")
        return orch
    t0 = time.perf_counter()
    orch = Orchestrator(bank=bank, max_epochs=8, verbose=False, seed=SEED)
    orch.fit(splits["train"], splits["val"])
    orch.save_checkpoint(CKPT)
    print(f"[fusion] refit single-head ALL-8 s{SEED} in {time.perf_counter() - t0:.0f}s "
          f"-> {CKPT.relative_to(REPO)}")
    return orch


def observed_vitals(stay_ids):
    raw = pd.read_parquet(RAW_PATH, columns=["stay_id", "hour"] + VITALS,
                          filters=[("stay_id", "in", [int(s) for s in stay_ids])])
    for v in VITALS:                     # temperature is a Decimal column
        raw[v] = pd.to_numeric(raw[v], errors="coerce").astype(float)
    return raw


def _aligned(keys, sub):
    return ((keys["stay_id"].to_numpy().astype(np.int64) == sub["stay_id"].to_numpy().astype(np.int64)).all()
            and (keys["hour"].to_numpy().astype(np.int64) == sub["hour"].to_numpy().astype(np.int64)).all())


def build_stay(role, sid, ctx, z, bank, orch, strata, unit_thresholds, raw, summary,
               thr, lit_test, thr_comp):
    te = ctx.test_ordered
    mask = (te["stay_id"] == sid).to_numpy()
    sub = te[mask].reset_index(drop=True)
    idx = np.flatnonzero(mask)
    hours = sub["hour"].to_numpy(dtype=int)

    comp_risk = {n: lit_test[n][idx] for n in COMPARATORS}
    comp_fa = {}
    for n in COMPARATORS:
        fa = first_alarm_hour(hours, comp_risk[n], thr_comp[n])
        comp_fa[n] = None if np.isnan(fa) else int(fa)

    comp = z["test_s42"][idx].astype(float)
    d = z["dhead_test_s42"][idx].astype(float)
    t = t_head_from_composite(comp, d)

    bank_sub = restrict_bank(bank, [sid])
    X, keys = bank_sub.lookup(sub)
    if not _aligned(keys, sub):
        raise RuntimeError(f"bank lookup misaligned for stay role {role}")
    spans, risk_offset = column_spans(bank_sub)
    lo, hi = spans["news2"]
    subscores = X[:, lo:hi].astype(float)
    agent_risks = {a: X[:, risk_offset + i].astype(float)
                   for i, a in enumerate(bank_sub.agent_names)}

    attn_keys, attn = orch.attention_weights(sub)
    if not _aligned(attn_keys, sub):
        raise RuntimeError(f"attention misaligned for stay role {role}")
    single = orch.emit(sub)
    if not _aligned(single.keys, sub):
        raise RuntimeError(f"single-head emission misaligned for stay role {role}")

    contexts = build_contexts(sub, sub, subscores, agent_risks, attn, AGENTS)
    obs = sub[["stay_id", "hour"]].merge(raw, on=["stay_id", "hour"], how="left")

    row = summary.set_index("stay_id").loc[sid]
    fa_m = None if pd.isna(row["fa_model"]) else int(row["fa_model"])
    fa_n = None if pd.isna(row["fa_news2"]) else int(row["fa_news2"])
    lead = (fa_n - fa_m) if (fa_m is not None and fa_n is not None) else None
    event_h = None if pd.isna(row["event_h"]) else float(row["event_h"])
    unit = str(strata.set_index("stay_id").loc[sid, "unit_group"])
    uthr = unit_thresholds.get(unit, unit_thresholds["__global__"])

    return {
        "role": role, "stay_id": int(sid), "unit_group": unit,
        "unit_thresholds": {k: uthr[k] for k in
                            ("thr_model_budget", "thr_model", "thr_news2", "source")},
        "event_type": row["event_type"] if event_h is not None else None,
        "event_h": event_h,
        "first_alarm": {"model": fa_m, "news2": fa_n,
                        "lead_h": None if lead is None else float(lead)},
        "hours": hours.tolist(),
        "risk": {"composite": comp, "t_head": t, "d_head": d,
                 "single_head": single.risk.astype(float)},
        "news2": {"total": sub["news2"].to_numpy(dtype=float),
                  "subscores": {c: subscores[:, k] for k, c in enumerate(NEWS2_COMPONENTS)}},
        "vitals_observed": {v: obs[v].to_numpy(dtype=float) for v in VITALS},
        "vitals_model_view": {v: sub[v].to_numpy(dtype=float) for v in VITALS},
        "vitals_hrs_since": {v: sub[f"{v}_hrs_since"].to_numpy(dtype=float) for v in VITALS},
        "agent_risk": {a: agent_risks[a] for a in AGENTS},
        "attention": {a: attn[:, i].astype(float) for i, a in enumerate(AGENTS)},
        "alarm": {"model": (comp >= thr["model"]).tolist(),
                  "news2": (sub["news2"].to_numpy(dtype=float) >= thr["news2"]).tolist()},
        "comp_risk": comp_risk,
        "comp_alarm": {n: (comp_risk[n] >= thr_comp[n]).tolist() for n in COMPARATORS},
        "comp_first_alarm": comp_fa,
        "contexts": {str(c["hour"]): c for c in contexts},
        "cards": {},
        "_contexts_list": contexts,
    }


def check_prompt_pin(stays):
    """Overlap with the T17 cache must render byte-identical prompts."""
    path = CACHE / "rationales_all_8_gemma4_12b.jsonl"
    if not path.exists():
        return
    t17 = RationaleCache(path)
    hits, bad = 0, 0
    for stay in stays:
        for c in stay["_contexts_list"]:
            rec = t17.entries.get((c["stay_id"], c["hour"]))
            if rec is None:
                continue
            hits += 1
            bad += int(rec["prompt_chars"] != len(build_prompt(c)))
    print(f"[pin] {hits} patient-hours overlap the T17 card cache; "
          f"{bad} prompt-length mismatches" + (" -- REFIT DRIFT" if bad else ""))


def footprint(orch, ctx, first_stay_sub, runner, contexts):
    params = sum(p.numel() for p in orch.net.parameters())
    state_bytes = sum(p.numel() * p.element_size() for p in orch.net.parameters())
    t0 = time.perf_counter()
    orch.emit(ctx.test_ordered)
    batch = time.perf_counter() - t0
    n = len(ctx.test_ordered)
    laps = []
    for _ in range(3):
        t0 = time.perf_counter()
        orch.emit(first_stay_sub)
        laps.append(time.perf_counter() - t0)
    out = {
        "n_params_per_head": int(params),
        "n_params_two_head": int(2 * params),
        "state_dict_bytes": int(state_bytes),
        "checkpoint_bytes": int(CKPT.stat().st_size),
        "torch_threads": int(torch.get_num_threads()),
        "batch_wall_s": float(batch),
        "batch_patient_hours": int(n),
        "us_per_patient_hour": float(batch / n * 1e6),
        "patient_hours_per_s": float(n / batch),
        "single_stay_ms": float(np.median(laps) * 1e3),
        "single_stay_hours": int(len(first_stay_sub)),
        "batch_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "llm_model": LLM, "llm_card_s_mean": None, "llm_cards_timed": 0,
        "llm_device": "local GPU via ollama",
    }
    if runner is not None:
        if llm_loop["n"] >= 20:               # the generation loop itself, warm
            out["llm_card_s_mean"] = float(llm_loop["s"] / llm_loop["n"])
            out["llm_cards_timed"] = int(llm_loop["n"])
        else:                                 # cache was warm: time a few calls
            laps = []
            for c in contexts[:5]:
                t0 = time.perf_counter()
                runner.generate(build_prompt(c))
                laps.append(time.perf_counter() - t0)
            out["llm_card_s_mean"] = float(np.mean(laps))
            out["llm_cards_timed"] = len(laps)
    return out


llm_loop = {"n": 0, "s": 0.0}


def cohort_json(ctx, z, meta, thr, fp, h2h):
    metrics = json.load(open(CACHE / "metrics_table.json"))
    by_name = {r["model"].strip(): r for r in metrics}
    two = by_name["TWO-HEAD ALL-8"]
    dhead = by_name["|- D-head (death only)"]
    fair = by_name["FAIR+TRAJ (145 feat)"]
    news2 = by_name["NEWS2"]
    seeds = [int(s) for s in meta["seeds"]]
    res = meta["results"]
    leads = [res[str(s)]["budgets"]["0.02"]["median"] for s in seeds]

    lit = json.load(open(CACHE / "lit_gbm_report.json"))["our_protocol"]
    standby = json.load(open(CACHE / "standby_report.json"))
    units = standby["units"]

    cohort = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "headline": {
            "n_test_hours": int(len(ctx.test_ordered)),
            "n_stays": int(ctx.test_ordered["stay_id"].nunique()),
            "prevalence_pct": float(100 * ctx.y_test.mean()),
            "auprc_mean": two["auprc"], "auroc_mean": two["auroc"],
            "auprc_seeds": two["auprc_seeds"], "auroc_seeds": two["auroc_seeds"],
            "auprc_ci": two["ci"]["auprc"],
            "dhead_death_auprc_mean": dhead["auprc"],
            "dhead_death_auroc_mean": dhead["auroc"],
            "fair_traj": {"auprc": fair["auprc"], "auroc": fair["auroc"]},
            "news2": {"auprc": news2["auprc"], "auroc": news2["auroc"]},
            "lead_2pct_mean_h": float(np.mean(leads)),
            "lead_positive_seeds": f"{sum(l > 0 for l in leads)}/{len(leads)}",
            "win_rate_mean": float(np.mean(
                [res[str(s)]["budgets"]["0.02"]["win_rate"] for s in seeds])),
        },
        "thresholds": {"budget": BUDGET, "thr_news2": thr["news2"],
                       "thr_model": thr["model"], "matched": thr["matched"]},
        "scatter": [
            {"name": SCATTER_NAMES.get(k, k), "auprc": v["auprc"], "auroc": v["auroc"],
             "median_lead_h": v["median_lead_h"], "win_rate": v["win_rate"],
             "n_pair": v["n_pair"], "model_alarm_rate": v["model_alarm_rate"],
             "is_flagship": k == "two-head flagship s42"}
            for k, v in lit.items()],
        "seeds": [
            {"seed": s, "auprc": res[str(s)]["auprc"],
             "auroc": two["auroc_seeds"][i],
             "dhead_death_auprc": res[str(s)]["d_head_death_auprc"],
             "lead_median_h": res[str(s)]["budgets"]["0.02"]["median"],
             "lead_median_ci": res[str(s)]["budgets"]["0.02"]["median_ci"],
             "win_rate": res[str(s)]["budgets"]["0.02"]["win_rate"],
             "win_rate_ci": res[str(s)]["budgets"]["0.02"]["win_rate_ci"],
             "n_pair": res[str(s)]["budgets"]["0.02"]["n_pair"],
             "delta_vs_fair_ci": res[str(s)]["delta_vs_fair_ci"]}
            for i, s in enumerate(seeds)],
        "burden": {
            "budget": standby["budget"],
            "spread_global_x": burden_spread(units, "alarm_global"),
            "spread_standby_x": burden_spread(units, "alarm_standby"),
            "units": [
                {"unit": u, "hours": r["hours"], "prevalence": r["prevalence"],
                 "source": r["source"], "alarm_global": r["alarm_global"],
                 "alarm_standby": r["alarm_standby"],
                 "thr_model_budget": standby["thresholds"][u]["thr_model_budget"],
                 "race_global": {k: r["race_global"].get(k) for k in ("n_pair", "median", "win_rate")},
                 "race_standby": {k: r["race_standby"].get(k) for k in ("n_pair", "median", "win_rate")}}
                for u, r in sorted(units.items(), key=lambda kv: -kv[1]["hours"])],
        },
        "footprint": fp,
        "h2h": h2h,
        "caveats": CAVEATS,
    }
    assert_aggregate_only(cohort)
    return cohort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--with-death", action="store_true")
    ap.add_argument("--pick-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_all = time.perf_counter()
    splits = load_splits()
    ctx = FusionContext(splits)
    strata = assign_strata(stay_frame(splits))
    bank = AgentBank.load(agent_names=list(AGENTS))
    print(f"[bank] all-8 loaded: {bank.matrix.shape}")
    orch = load_or_fit(bank, splits, args.refit)

    z = np.load(CACHE / "two_head.npz")
    if not ((z["stay_id_test"] == ctx.test_ordered["stay_id"].to_numpy()).all()
            and (z["hour_test"] == ctx.test_ordered["hour"].to_numpy()).all()):
        raise RuntimeError("two_head.npz does not align with the cohort test rows")
    meta = json.loads(str(z["meta_json"]))

    zl = np.load(CACHE / "lit_gbm_scores.npz")
    if not ((zl["test_stay_id"] == ctx.test_ordered["stay_id"].to_numpy()).all()
            and (zl["test_hour"] == ctx.test_ordered["hour"].to_numpy()).all()):
        raise RuntimeError("lit_gbm_scores.npz does not align with the cohort test rows")
    if not ((zl["val_stay_id"] == ctx.val_ordered["stay_id"].to_numpy()).all()
            and (zl["val_hour"] == ctx.val_ordered["hour"].to_numpy()).all()):
        raise RuntimeError("lit_gbm_scores.npz does not align with the cohort val rows")

    fm = ctx.freeze_mask
    thr_n2, thr_m, matched = frozen_thresholds(
        ctx.news2_val[fm], z["val_s42"].astype(float)[fm], ctx.alertable_val[fm], BUDGET)
    standby = json.load(open(CACHE / "standby_report.json"))
    g = standby["thresholds"]["__global__"]
    if abs(g["thr_model"] - thr_m) > 1e-9 or abs(g["thr_news2"] - thr_n2) > 1e-9:
        raise RuntimeError("demo thresholds differ from the published standby report")
    thr = {"model": thr_m, "news2": thr_n2, "matched": matched}
    flag_test = z["test_s42"].astype(float)
    print(f"[thresholds] budget {BUDGET:.0%}: NEWS2 >= {thr_n2:g}, model >= {thr_m:.6f} "
          f"(matched {matched:.4f}); test rates model "
          f"{(flag_test >= thr_m).mean():.4f} / NEWS2 {(ctx.news2_test >= thr_n2).mean():.4f}")

    # Comparators frozen on the same V_freeze half at the same matched volume,
    # so every model on the head-to-head tab spends the same alarm budget.
    av = ctx.alertable_val[fm]
    thr_comp = {n: float(np.quantile(zl[f"val_{n}"].astype(float)[fm][av], 1.0 - matched))
                for n in COMPARATORS}
    lit_test = {n: zl[f"test_{n}"].astype(float) for n in COMPARATORS}
    for n in COMPARATORS:
        print(f"[thresholds] {n} >= {thr_comp[n]:.6f}; test rate "
              f"{(lit_test[n] >= thr_comp[n]).mean():.4f}")

    summary = stay_summary(ctx.test_ordered, flag_test, ctx.news2_test, thr_m, thr_n2)

    h2h_table = first_alarm_table(
        ctx.test_ordered,
        {"sentinel": flag_test, **lit_test},
        {"sentinel": thr_m, **thr_comp})
    h2h = h2h_summary(h2h_table, "sentinel", list(COMPARATORS))
    h2h.update({
        "budget": BUDGET, "matched": matched, "labels": dict(COMPARATOR_LABELS),
        "thresholds": {"sentinel": thr_m, **thr_comp},
        "alarm_rate_test": {
            "sentinel": float((flag_test >= thr_m).mean()),
            **{n: float((lit_test[n] >= thr_comp[n]).mean()) for n in COMPARATORS}},
        "caveat": ("published feature specs retrained on our cohort, "
                   "not the commercial products"),
    })
    # Ten-seed lead-time consistency: the flagship refrozen per seed at the
    # matched volume. The GBM baselines are deterministic single fits, so only
    # the flagship's warning depth and the discordant race vary by seed.
    from scipy.stats import binomtest
    seed_list = sorted({int(k.split("_s")[1]) for k in z.files
                        if k.startswith("test_s")})
    per_seed, earlier = [], {n: [] for n in COMPARATORS}
    for s_ in seed_list:
        thr_s = float(np.quantile(z[f"val_s{s_}"].astype(float)[fm][av],
                                  1.0 - matched))
        tab_s = first_alarm_table(
            ctx.test_ordered,
            {"sentinel": z[f"test_s{s_}"].astype(float), **lit_test},
            {"sentinel": thr_s, **thr_comp})
        ls = lead_stats(tab_s, "sentinel")
        per_seed.append({"seed": s_, "n": ls["n"], "median_h": ls["median_h"],
                         "mean_h": ls["mean_h"]})
        ev_s = tab_s[tab_s["event_h"].notna()]
        c_sent = ev_s["fa_sentinel"].notna() & (ev_s["fa_sentinel"] <= ev_s["event_h"])
        for n in COMPARATORS:
            c_opp = ev_s[f"fa_{n}"].notna() & (ev_s[f"fa_{n}"] <= ev_s["event_h"])
            d = (ev_s[f"fa_{n}"] - ev_s["fa_sentinel"])[c_sent & c_opp]
            nz = d[d != 0]
            p = binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else 1.0
            earlier[n].append({
                "seed": s_, "n_differ": int(len(nz)),
                "earlier_pct": float((nz > 0).mean()) if len(nz) else 0.0,
                "sig": bool(p < 0.05)})
    meds = [r["median_h"] for r in per_seed]
    h2h["seeds"] = {
        "n_seeds": len(seed_list),
        "sentinel_lead": per_seed,
        "lead_median_h": {"mean": float(np.mean(meds)), "min": float(min(meds)),
                          "max": float(max(meds))},
        "lead_mean_h": {"mean": float(np.mean([r["mean_h"] for r in per_seed]))},
        "earlier": {n: {
            "per_seed": earlier[n],
            "mean_pct": float(np.mean([r["earlier_pct"] for r in earlier[n]])),
            "min_pct": float(min(r["earlier_pct"] for r in earlier[n])),
            "max_pct": float(max(r["earlier_pct"] for r in earlier[n])),
            "sig_seeds": int(sum(r["sig"] for r in earlier[n]))}
            for n in COMPARATORS},
    }
    print(f"[seeds] flagship lead median {np.mean(meds):.2f} h across "
          f"{len(seed_list)} seeds (range {min(meds):.2f}-{max(meds):.2f}); "
          f"earlier-when-differ "
          + "; ".join(f"{n} {h2h['seeds']['earlier'][n]['mean_pct']:.1%} "
                      f"(sig {h2h['seeds']['earlier'][n]['sig_seeds']}/{len(seed_list)})"
                      for n in COMPARATORS))

    # Budget sweep: the same frozen-threshold head-to-head at seven nominal
    # budgets plus the demo's own operating point (the matched volume), so the
    # "does 2% matter?" question is answered on screen.
    sweep_budgets = sorted({0.005, 0.01, 0.02, round(matched, 6), 0.03, 0.05,
                            0.075, 0.10})
    sweep_names = ["sentinel", *COMPARATORS]
    sweep_val = {"sentinel": z["val_s42"].astype(float),
                 **{n: zl[f"val_{n}"].astype(float) for n in COMPARATORS}}
    sweep_test = {"sentinel": flag_test, **lit_test}
    sweep_rows = []
    for b in sweep_budgets:
        thr_b = {n: float(np.quantile(sweep_val[n][fm][av], 1.0 - b))
                 for n in sweep_names}
        row = budget_sweep_row(
            first_alarm_table(ctx.test_ordered, sweep_test, thr_b), sweep_names)
        sweep_rows.append({"budget": float(b), **row})
    h2h["sweep"] = {"models": sweep_names, "operating": matched,
                    "rows": sweep_rows}
    print("[sweep] catch at " + "; ".join(
        f"{r['budget']:.1%}: " + "/".join(f"{r[n]['catch']:.0%}" for n in sweep_names)
        for r in sweep_rows))

    f_any = h2h["funnel"]["any"]
    print("[h2h] catch " + " / ".join(f"{m} {f_any[m]['pct']:.1%}" for m in h2h["models"])
          + "; race " + "; ".join(
              f"vs {r['opponent']}: {r['flagship_first']}-{r['tie']}-{r['opponent_first']}"
              f" (only {r['only_flagship']}:{r['only_opponent']})" for r in h2h["race"])
          + "; quiet FP " + " / ".join(
              f"{h2h['cost']['quiet_fp_rate'][m]:.1%}" for m in h2h["models"]))
    chosen = select_demo_stays(summary, seed=args.seed, pick_index=args.pick_index,
                               with_death=args.with_death)
    print("[stays] " + ", ".join(f"{role}={sid}" for role, sid in chosen.items()))

    raw = observed_vitals(list(chosen.values()))
    stays = [build_stay(role, sid, ctx, z, bank, orch, strata, standby["thresholds"],
                        raw, summary, thr, lit_test, thr_comp)
             for role, sid in chosen.items()]
    for s in stays:
        fa = s["first_alarm"]
        comps = " / ".join(f"{n} {s['comp_first_alarm'][n]}" for n in COMPARATORS)
        print(f"  {s['role']:<18} {s['unit_group']:<11} hours {s['hours'][0]}-{s['hours'][-1]}"
              f"  event {s['event_type']} @ {s['event_h']}  first alarm model {fa['model']}"
              f" / NEWS2 {fa['news2']} / {comps}  lead {fa['lead_h']}")
    check_prompt_pin(stays)

    contexts = [c for s in stays for c in s["_contexts_list"]]
    runner = None
    if not args.skip_llm:
        from text_arm.llm_orchestrator import OllamaRunner
        runner = OllamaRunner(model=LLM, system=SYSTEM_PROMPT, num_predict=90,
                              think=False)
        cache_path = OUT_DIR / "rationales_demo_gemma4_12b.jsonl"
        n_before = len(RationaleCache(cache_path))
        t0 = time.perf_counter()
        cache = generate_rationales(contexts, cache_path, model=LLM, runner=runner)
        llm_loop.update({"n": len(cache) - n_before, "s": time.perf_counter() - t0})
        texts = cache.texts_for(contexts)
        pos = 0
        for s in stays:
            for c in s["_contexts_list"]:
                s["cards"][str(c["hour"])] = texts[pos]
                pos += 1
        empty = sum(1 for t in texts if not t)
        print(f"[cards] {len(texts)} patient-hours, {empty} empty")

    fp = footprint(orch, ctx, ctx.test_ordered[ctx.test_ordered["stay_id"] == stays[0]["stay_id"]],
                   runner, contexts)
    print(f"[footprint] {fp['n_params_per_head']:,} params/head; "
          f"{fp['us_per_patient_hour']:.1f} us per patient-hour at "
          f"{fp['torch_threads']} thread; single stay {fp['single_stay_ms']:.1f} ms; "
          f"card {fp['llm_card_s_mean']} s")

    for s in stays:
        del s["_contexts_list"]
    bundle = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(), "seed": args.seed,
        "agents": list(AGENTS), "news2_components": list(NEWS2_COMPONENTS),
        "vitals": list(VITALS), "llm": LLM,
        "thresholds": {"budget": BUDGET, "thr_news2": thr_n2, "thr_model": thr_m,
                       "matched": matched,
                       **{f"thr_{n}": thr_comp[n] for n in COMPARATORS},
                       "source": "two_head.npz val_s42 on the V_freeze half",
                       "comparator_source": ("lit_gbm_scores.npz val on the "
                                             "V_freeze half at the matched volume")},
        "comparators": list(COMPARATORS),
        "comparator_labels": dict(COMPARATOR_LABELS),
        "risk_source": "two-head ALL-8 composite, seed 42 (data/cache/two_head.npz)",
        "attention_source": ("single-head ALL-8 refit, seed 42 "
                             "(models/orchestrator/all8_s42.pt); the composite's "
                             "shared substrate, as in T17"),
        "stays": stays,
    }
    bundle = json.loads(dumps_strict(bundle))
    validate_bundle(bundle)
    (OUT_DIR / "bundle.json").write_text(dumps_strict(bundle))

    cohort = cohort_json(ctx, z, meta, thr, fp, h2h)
    (OUT_DIR / "cohort.json").write_text(dumps_strict(cohort))
    print(f"[saved] {OUT_DIR / 'bundle.json'} and cohort.json in "
          f"{time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()
