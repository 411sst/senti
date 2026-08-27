"""T16 -- goals-of-care extraction over discharge notes, validated then applied.

    python scripts/run_goc_extraction.py --phase validate   # both models vs charted truth
    python scripts/run_goc_extraction.py --phase apply      # winner over uncharted stays
    python scripts/run_goc_extraction.py --phase report     # re-partitioned T11 lead table
    python scripts/run_goc_extraction.py --phase all
    python scripts/run_goc_extraction.py --phase validate --limit 40   # smoke

Audit-only (see src/goc_notes.py): verdicts partition evaluation tables,
never feed a model. The 1,029 charted death stays are the free validation set
-- both local models are scored against the charted field there, and only the
empirically better one is trusted on the uncharted stays. Cross-model
contradiction (yes vs no) downgrades to "unsure" rather than picking a side.

The report phase reads two_head.npz s42 scores -- run it only after the
5-seed chain regeneration has finished and the seed-42 regression pin passed.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from goc_notes import (  # noqa: E402
    GOC_NOTES_PATH, SYSTEM_PROMPT, GocCache, adjudicate, death_stays,
    extract_snippets, goc_fingerprint,
)
from text_arm.llm_orchestrator import OllamaRunner  # noqa: E402

OUT_JSON = REPO / "data" / "cache" / "goc_report.json"
# Chosen empirically (probed 2026-08-27, ollama 0.30.10): gemma4:12b/e4b are
# REASONING models here -- with a capped num_predict the whole budget goes to
# the hidden think block and the response comes back empty (and thinking left
# on, 12b did not finish deliberating in 2,048 tokens). With think=False,
# gemma4:12b answers cleanly in ~40 tokens, so all three families read;
# deepseek-r1:14b also works but spends ~8x the tokens per verdict. The
# verdict source is still picked by validation kappa, three-way.
DEFAULT_MODELS = "qwen2.5-coder:14b,llama3.1:8b,gemma4:12b"
SNIPPET_CAP = 8
BUDGET = 0.02


def model_tag(model):
    return re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_")


def stay_snippets(notes):
    """{stay_id: [snippet, ...]} over all of a stay's notes, capped jointly."""
    out = {}
    for stay_id, g in notes.sort_values("charttime").groupby("stay_id"):
        snips = []
        for text in g["text"]:
            snips.extend(extract_snippets(text, cap=SNIPPET_CAP - len(snips)))
            if len(snips) >= SNIPPET_CAP:
                break
        out[int(stay_id)] = snips
    return out


def run_model(model, stay_ids, snippets, limit=None):
    """Verdict per stay for one model, resumable via the per-model cache."""
    cache = GocCache(REPO / "data" / "cache" / f"goc_verdicts_{model_tag(model)}.jsonl",
                     goc_fingerprint(model))
    runner = OllamaRunner(model=model, system=SYSTEM_PROMPT,
                          num_predict=200, num_ctx=8192, timeout=180,
                          think=False)
    todo = [s for s in stay_ids if s in snippets and s not in cache]
    if limit is not None:
        todo = todo[:limit]
    llm_todo = [s for s in todo if snippets[s]]
    print(f"[{model}] {len(cache):,} cached · {len(todo):,} to do "
          f"({len(todo) - len(llm_todo):,} with no GoC text -> no LLM call)")
    done = 0
    for s in todo:
        if snippets[s]:
            verdict = adjudicate(runner, snippets[s])
        else:
            verdict = {"any_limitation": "no", "cmo": "no",
                       "evidence_quote": "", "no_goc_text": True}
        cache.add(s, verdict, n_snippets=len(snippets[s]))
        done += 1
        if done % 25 == 0:
            print(f"  {done:,}/{len(todo):,}", flush=True)
    return cache


def kappa(truth, pred):
    """Cohen's kappa for two binary vectors (no sklearn dependency needed)."""
    truth, pred = np.asarray(truth, bool), np.asarray(pred, bool)
    po = float((truth == pred).mean())
    pe = (truth.mean() * pred.mean()
          + (1 - truth.mean()) * (1 - pred.mean()))
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def validate(models, deaths, snippets, limit=None):
    """Both models against the charted field on charted death stays."""
    charted = deaths[deaths["has_code_status"]]
    with_note = charted[charted["stay_id"].isin(snippets)]
    print(f"\nVALIDATE -- {len(charted):,} charted death stays, "
          f"{len(with_note):,} with a discharge note "
          f"(ever_limited base rate {with_note['ever_limited'].mean():.1%}, "
          f"limit-before-death {with_note['limit_before_event'].mean():.1%})")
    out = {}
    for model in models:
        cache = run_model(model, with_note["stay_id"].tolist(), snippets, limit)
        rows = with_note[with_note["stay_id"].map(lambda s: s in cache)]
        v = rows["stay_id"].map(lambda s: cache.get(s)["any_limitation"])
        sure = (v != "unsure").to_numpy()
        pred = (v == "yes").to_numpy()
        m = {
            "n_scored": int(len(rows)),
            "unsure_rate": float((~sure).mean()) if len(rows) else None,
            "parse_failed": int(sum(bool(cache.get(s).get("parse_failed"))
                                    for s in rows["stay_id"])),
            "no_goc_text": int(sum(bool(cache.get(s).get("no_goc_text"))
                                   for s in rows["stay_id"])),
        }
        if sure.sum():
            # Truth: was a limitation EVER charted. But the charted extract is
            # ICU chartevents only, so it is authoritative ONLY for in-ICU
            # deaths -- for post-ICU deaths the terminal decision is made on
            # the ward and the chart is structurally blind. Metrics are
            # therefore stratified; the winner is picked on the in-ICU
            # stratum, where disagreement means extractor error.
            strata = {"": np.ones(len(rows), bool),
                      "_in_icu": ~rows["post_icu_death"].to_numpy(bool),
                      "_post_icu": rows["post_icu_death"].to_numpy(bool)}
            t_all = rows["ever_limited"].to_numpy(bool)
            for name, sel in strata.items():
                k = sure & sel
                if not k.sum():
                    continue
                t, p = t_all[k], pred[k]
                m.update({
                    f"n{name}": int(k.sum()),
                    f"accuracy{name}": float((t == p).mean()),
                    f"sensitivity{name}": float(p[t].mean()) if t.any() else None,
                    f"specificity{name}": float((~p[~t]).mean()) if (~t).any() else None,
                    f"kappa{name}": kappa(t, p),
                })
        out[model] = m
        print(f"  {model}: " + json.dumps(m))
    if len(models) >= 2 and all(out[m].get("n_scored") for m in models):
        caches = {m: GocCache(REPO / "data/cache" /
                              f"goc_verdicts_{model_tag(m)}.jsonl",
                              goc_fingerprint(m)) for m in models}
        pairs = {}
        for i, ma in enumerate(models):
            for mb in models[i + 1:]:
                a, b = caches[ma], caches[mb]
                common = [s for s in with_note["stay_id"] if s in a and s in b]
                if common:
                    rate = float(np.mean(
                        [a.get(s)["any_limitation"] == b.get(s)["any_limitation"]
                         for s in common]))
                    pairs[f"{ma} vs {mb}"] = {"n": len(common), "rate": rate}
        out["cross_model_agreement"] = pairs or None
    def _kappa(m):                       # 0.0 is a real kappa, not "missing"
        k = out[m].get("kappa_in_icu", out[m].get("kappa"))
        return -9.0 if k is None else k
    ranked = sorted(models, key=_kappa, reverse=True)
    out["ranking"] = ranked
    out["winner"] = ranked[0]
    print(f"  winner by in-ICU kappa: {out['winner']}")
    return out


def apply_uncharted(models, ranking, deaths, snippets, limit=None):
    """Top-ranked reader's verdict on uncharted stays; a contradiction from
    the runner-up (yes vs no) downgrades to unsure. All listed models run, so
    the full three-way comparison is cached for the report."""
    winner = ranking[0]
    unch = deaths[~deaths["has_code_status"]]
    with_note = unch[unch["stay_id"].isin(snippets)]
    print(f"\nAPPLY -- {len(unch):,} uncharted death stays, "
          f"{len(with_note):,} with a discharge note")
    caches = {m: run_model(m, with_note["stay_id"].tolist(), snippets, limit)
              for m in models}
    second = next((m for m in ranking[1:] if m in caches), None)
    final, counts = {}, {"yes": 0, "no": 0, "unsure": 0, "no_note": 0}
    for s in unch["stay_id"]:
        s = int(s)
        if s not in snippets or s not in caches[winner]:
            final[s] = "no_note"
        else:
            w = caches[winner].get(s)["any_limitation"]
            if second and s in caches[second]:
                o = caches[second].get(s)["any_limitation"]
                if {w, o} == {"yes", "no"}:
                    w = "unsure"           # contradiction: don't pick a side
            final[s] = w
        counts[final[s]] += 1
    print("  " + json.dumps(counts))
    return final, counts


def lead_with_notes(deaths, final_verdicts, budget=BUDGET, reps=2000):
    """The T11 death-lead race, re-partitioned with note-informed status.

    Same globally frozen thresholds and the same paired race as
    run_code_status.lead_by_decision -- only the SUMMARY cells change, so the
    "all death stays" row must reproduce the T11 composite.
    """
    from baselines_mimic import load_splits
    from fair_capability import (frozen_thresholds, paired_first_alarm_diffs,
                                 summarize_paired)
    from fusion_common import FusionContext

    ctx = FusionContext(load_splits())
    th = np.load(REPO / "data/cache/two_head.npz")
    if not (th["stay_id_test"] == ctx.test_ordered["stay_id"].to_numpy()).all():
        raise ValueError("two_head cache does not align with the cohort")
    s_val, s_test = th["val_s42"].astype(float), th["test_s42"].astype(float)
    thr_n2, thr_m, _ = frozen_thresholds(
        ctx.news2_val[ctx.freeze_mask], s_val[ctx.freeze_mask],
        ctx.alertable_val[ctx.freeze_mask], budget)
    diffs = paired_first_alarm_diffs(ctx.test_ordered, ctx.news2_test, s_test,
                                     thr_n2, thr_m)

    idx = deaths.set_index("stay_id")
    d = diffs[diffs["stay_id"].isin(idx.index)].copy()
    d["charted"] = d["stay_id"].map(idx["has_code_status"]).fillna(False)
    d["decided_charted"] = d["stay_id"].map(idx["limit_before_event"]).fillna(False)
    d["note"] = d["stay_id"].map(lambda s: final_verdicts.get(int(s), "n/a"))

    unch = ~d["charted"]
    cells = {
        "all death stays": d,
        "charted, decision before death": d[d["decided_charted"]],
        "charted, no limitation": d[d["charted"] & ~d["decided_charted"]],
        "uncharted, note: limitation": d[unch & (d["note"] == "yes")],
        "uncharted, note: no limitation": d[unch & (d["note"] == "no")],
        "uncharted, unsure or no note": d[unch & d["note"].isin(["unsure", "no_note"])],
        "EXCLUDING all decided": d[~d["decided_charted"] & ~(unch & (d["note"] == "yes"))],
    }
    print(f"\nREPORT -- budget {budget:.0%}, thresholds frozen globally · "
          f"{len(d)} jointly-caught death stays in the test split")
    print(f"  {'subgroup':<36}{'n':>5}{'median lead':>13}{'win':>7}")
    print("  " + "-" * 61)
    out = {}
    for name, cell in cells.items():
        if len(cell) == 0:
            print(f"  {name:<36}{0:>5}{'--':>13}{'--':>7}")
            out[name] = {"n_pair": 0}
            continue
        r = summarize_paired(cell, ctx.stay_to_patient, reps=min(reps, 1000))
        flag = "  *" if len(cell) < 20 else ""
        print(f"  {name:<36}{r['n_pair']:>5}{r['median']:>12.1f}h"
              f"{r['win_rate']:>7.0%}{flag}")
        out[name] = {"n_pair": r["n_pair"], "median": r["median"],
                     "win_rate": r["win_rate"], "median_ci": r.get("median_ci")}
    print("  * fewer than 20 pairs -- reported for completeness, not evidence")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--phase", choices=["validate", "apply", "report", "all"],
                    default="all")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test cap")
    ap.add_argument("--json", default=str(OUT_JSON))
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    notes = pd.read_parquet(REPO / GOC_NOTES_PATH)
    deaths = death_stays(cohort_path=str(REPO / "data/cache/mimic_cohort_hourly.parquet"))
    snippets = stay_snippets(notes)
    no_note = int((~deaths["stay_id"].isin(snippets)).sum())
    print(f"{len(deaths):,} death stays · {len(snippets):,} with a discharge "
          f"note · {no_note:,} without")

    report = json.loads(Path(args.json).read_text()) if Path(args.json).exists() else {}
    report["populations"] = {
        "death_stays": int(len(deaths)),
        "charted": int(deaths["has_code_status"].sum()),
        "uncharted": int((~deaths["has_code_status"]).sum()),
        "without_note": no_note,
        "post_icu_death": int(deaths["post_icu_death"].sum()),
        "uncharted_post_icu": int((~deaths["has_code_status"]
                                   & deaths["post_icu_death"]).sum()),
    }

    if args.phase in ("validate", "all"):
        report["validation"] = validate(models, deaths, snippets, args.limit)
    if args.phase in ("apply", "all"):
        val = report.get("validation") or {}
        ranking = val.get("ranking") or ([val["winner"]] if val.get("winner")
                                         else None)
        if not ranking:
            raise SystemExit("run --phase validate first (no winner recorded)")
        final, counts = apply_uncharted(models, ranking, deaths, snippets,
                                        args.limit)
        report["apply"] = {"winner": ranking[0], "counts": counts,
                           "verdicts": {str(k): v for k, v in final.items()}}
    if args.phase in ("report", "all"):
        verdicts = {int(k): v for k, v in
                    (report.get("apply", {}).get("verdicts") or {}).items()}
        if not verdicts:
            raise SystemExit("run --phase apply first (no verdicts recorded)")
        report["lead_with_notes"] = lead_with_notes(deaths, verdicts)

    Path(args.json).write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[saved] {args.json}")


if __name__ == "__main__":
    main()
