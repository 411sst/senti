"""Rationales for orchestrator alerts + the automatic faithfulness benchmark.

    python scripts/run_explainability.py --smoke                 # prompts only
    python scripts/run_explainability.py --n-alerts 30 --n-null 30
    python scripts/run_explainability.py --config all-8 --model gemma4:12b \
        --n-alerts 100 --n-null 100 --json data/cache/explain_report_all8_gemma.json
    python scripts/run_explainability.py --config all-8 --model medgemma:4b \
        --prompt-variant menu-fewshot ...   # enumerated menu + worked examples

Part 1 -- generate a one-sentence rationale from the structured drivers the
orchestrator already exposes (NEWS2 sub-scores, per-agent calibrated risk,
attention over agents, 4h vital trend), using a local ollama model
(``--model``; the T17 bench winner is gemma4:12b). ``--prompt-variant
menu-fewshot`` appends an enumerated candidate menu and three worked examples
taken from another reader's cached plain-prompt cards: it lifts any reader,
but it saturates the menu-served columns and inflates the shuffled baseline,
so the plain prompt is the reported bench (results_fusion.md T17).

Part 2 -- score those rationales on three offline faithfulness metrics that use
the model's own structure as ground truth, and score two anti-gaming baselines
alongside them. The baselines are the point: a metric that ranks a shuffled or
generic sentence as highly as a real one is measuring fluency, not faithfulness.

Sampling is at most ONE row per stay. That is not cosmetic -- it means each
row's ablation is confined to its own stay (so a whole arm is one forward pass)
and each row is its own cross-validation group (so the surrogate cannot see two
hours of the same patient across a fold boundary).
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

from agents.base import AgentOutput  # noqa: E402
from baselines_mimic import load_splits  # noqa: E402
from config import SEED  # noqa: E402
from explain.drivers import (  # noqa: E402
    AGENT_LABELS, build_contexts, driver_vocabulary, named_agents,
    named_components, named_drivers, top_components,
)
from explain.faithfulness import (  # noqa: E402
    attention_baseline, attention_consistency, bootstrap_auroc_gap,
    bootstrap_gap, column_reference, column_spans, comprehensiveness,
    component_consistency, raw_risk_at, receptive_field_hours, restrict_bank,
    simulatability, sufficiency, summarise_consistency, with_bank,
)
from explain.rationales import (  # noqa: E402
    MENU_SYSTEM_PROMPT, SYSTEM_PROMPT, RationaleCache, build_menu_prompt,
    build_prompt, examples_for, few_shot_pool, generate_rationales,
    generic_rationales, prompt_fingerprint, scrub_leak_words,
    shuffle_rationales,
)
from orchestrator.fusion import AgentBank, Orchestrator, resolve_config  # noqa: E402
from text_arm.llm_orchestrator import DEFAULT_MODEL  # noqa: E402
from text_arm.summaries import TREND_LAG_HOURS  # noqa: E402

LABEL = "label_24h"
EMB = REPO / "data" / "cache" / "embeddings"


def cache_path(config, model, variant="plain"):
    """One rationale cache per (config, model, prompt variant).

    The cache key is (stay_id, hour) and its fingerprint covers only the
    SYSTEM PROMPT -- so a single shared file would silently serve sentences
    generated under a different agent set or a different model. (The
    pre-T17 single-config cache was data/cache/rationales.jsonl.) The plain
    variant keeps the T17 path; other variants get their own file.
    """
    import re
    tag = re.sub(r"[^A-Za-z0-9]+", "_", f"{config}_{model}").strip("_")
    if variant != "plain":
        tag += "__" + re.sub(r"[^A-Za-z0-9]+", "_", variant).strip("_")
    return REPO / "data" / "cache" / f"rationales_{tag}.jsonl"


# --- picking the patient-hours to explain ----------------------------------

def sample_rows(test, risk, threshold, n_alerts, n_null, seed=SEED):
    """One fired hour and one quiet hour from disjoint stays.

    **Eligibility.** Every stay begins at hour 6 (the stabilisation window) and
    that first row is always all-NaN, because nothing precedes it. So an hour
    can only carry a 4h trend if it sits at least ``TREND_LAG_HOURS + 1`` past
    the stay's start; earlier hours render every vital as "no 4h history".

    That matters more than it looks. 91% of *first* alarms fire by hour 10, so
    "explain the first alert" would mean explaining a decision with no
    trajectory behind it -- a poor showcase for a trajectory model, and worse, a
    confound: if fired rows systematically lacked trends while quiet rows had
    them, the simulatability surrogate would learn that structural artifact
    instead of the drivers. The filter is therefore applied to BOTH groups, and
    for a firing stay we take its first *eligible* hour over threshold.

    Quiet rows come from stays that never fire at all, so the two groups share
    no stay and the surrogate's folds stay clean.
    """
    ordered = test.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    start = ordered.groupby("stay_id")["hour"].transform("min")
    ordered = ordered.assign(
        _risk=risk,
        _fired=risk >= threshold,
        _eligible=ordered["hour"] >= start + TREND_LAG_HOURS + 1,
    )
    ever_fires = ordered.groupby("stay_id")["_fired"].transform("max")

    rng = np.random.default_rng(seed)
    fired_pool = ordered[ordered["_fired"] & ordered["_eligible"]]
    quiet_pool = ordered[~ever_fires & ordered["_eligible"]]
    fired_stays = np.asarray(fired_pool.stay_id.unique())
    quiet_stays = np.asarray(quiet_pool.stay_id.unique())
    rng.shuffle(fired_stays)
    rng.shuffle(quiet_stays)
    print(f"[sample] eligible pool: {len(fired_stays):,} firing stays "
          f"(of {int(ordered[ordered['_fired']].stay_id.nunique()):,} that ever "
          f"fire), {len(quiet_stays):,} quiet stays")

    picks = []
    for stay in fired_stays[:n_alerts]:
        picks.append(int(fired_pool[fired_pool.stay_id == stay].index[0]))
    for stay in quiet_stays[:n_null]:
        candidates = quiet_pool[quiet_pool.stay_id == stay].index.to_numpy()
        picks.append(int(rng.choice(candidates)))

    return ordered.loc[picks].sort_values(
        ["stay_id", "hour"]).reset_index(drop=True)


def verify_shipped(orch, test, config):
    """Confirm we are explaining the model that was shipped, not a re-roll.

    ``Orchestrator`` has no weight checkpoint -- it is refit here from the same
    seed and the same cached latents. Deterministic, so it should reproduce the
    cached emissions exactly; this checks rather than assumes.
    """
    path = EMB / f"orchestrator-{config}_test.npz"
    if not path.exists():
        print(f"[verify] no cached emission at {path.name} -- skipping")
        return
    cached = AgentOutput.load(path)
    mine = orch.emit(test)
    if not cached.keys.equals(mine.keys):
        print("[verify] **key mismatch** against the shipped cache")
        return
    diff = np.abs(cached.risk.astype(float) - mine.risk.astype(float))
    status = "matches" if diff.max() < 1e-6 else "**DIFFERS from**"
    print(f"[verify] refit orchestrator {status} the shipped "
          f"{config} cache (max |diff| {diff.max():.2e}, "
          f"corr {np.corrcoef(cached.risk, mine.risk)[0, 1]:.6f})")


# --- reporting -------------------------------------------------------------

def show_examples(rows, contexts, texts, driver_sets, base_risk, n=6):
    print(f"\n{'=' * 100}")
    print(f"EXAMPLE RATIONALES ({n} of {len(texts)})")
    print("=" * 100)
    for i in range(min(n, len(texts))):
        context, row = contexts[i], rows.iloc[i]
        attn = ", ".join(f"{AGENT_LABELS.get(a, a)} {v:.2f}"
                         for a, v in sorted(context["attention"].items(),
                                            key=lambda kv: -kv[1]))
        tops = top_components(context) or ["none"]
        state = "FIRED" if row["_fired"] else "quiet"
        print(f"\n  stay {context['stay_id']} hour {context['hour']}  "
              f"fused risk {base_risk[i]:.3f}  {state}  "
              f"label_24h={int(row[LABEL])}")
        print(f"    NEWS2 {context['news2_total']:.0f}  "
              f"top sub-scores: {', '.join(tops)}")
        print(f"    attention: {attn}")
        print(f"    > {texts[i] or '(empty)'}")
        print(f"    named: {', '.join(sorted(driver_sets[i])) or '(nothing)'}")


def arm_table(results, order):
    print(f"\n{'=' * 100}")
    print("FAITHFULNESS -- computed on FIRED alerts only "
          "(what the metric is about: was the alert explained?)")
    print("=" * 100)
    hdr = (f"  {'arm':<12}{'comp@t':>9}{'comp@win':>10}{'rel@win':>9}"
           f"{'suff':>8}   {'attn rho':>9}{'attn@1':>8}{'attnC rho':>11}"
           f"{'attnC@1':>9}   {'n2 rho':>8}{'n2@1':>7}{'halluc':>9}"
           f"   {'named':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name in order:
        r = results[name]
        print(f"  {name:<12}{r['comp_at_t']:>9.4f}{r['comp_window']:>10.4f}"
              f"{r['rel_window']:>8.1%}{r['suff']:>8.4f}   "
              f"{r['attn']['rho']:>9.3f}{r['attn']['top1_hit']:>8.2f}"
              f"{r['attn_c']['rho']:>11.3f}{r['attn_c']['top1_hit']:>9.2f}   "
              f"{r['n2']['rho']:>8.3f}{r['n2']['top1_hit']:>7.2f}"
              f"{r['n2']['unsupported']:>9.2f}   "
              f"{r['attn']['n_named']:>6.2f}")
    print("\n  comp@t    mean risk drop when the named drivers are removed at hour t")
    print("  comp@win  ... removed across the whole 29h causal receptive field "
          "(the headline)")
    print("  rel@win   the same drop as a fraction of the row's base risk")
    print("  suff      drop when everything EXCEPT the named drivers is removed "
          "(lower = better)")
    print("  attn rho  Spearman(named agents, raw attention); attn@1 = names the "
          "top-attention agent")
    print("  attnC     the same against attention CENTRED per agent -- the raw "
          "version is saturated,")
    print("            because labs is the argmax on 99.7% of hours so 'always "
          "name labs' wins it")
    print("  n2 rho    Spearman(named NEWS2 parameters, sub-scores); n2@1 = "
          "names the top sub-score")
    print("  halluc    fraction of named NEWS2 parameters whose sub-score is "
          "actually ZERO")
    print("  named     mean number of the 4 agents the sentence names")


def gap_table(results, baselines=("RANDOM", "GENERIC")):
    """Is FAITHFUL actually ahead of each baseline, or is the gap noise?

    Paired bootstrap over rows. One row per stay, so this is the same
    stay-level resample the rest of the project's confidence intervals use.
    """
    print(f"\n{'=' * 100}")
    print("PAIRED BOOTSTRAP -- FAITHFUL minus baseline (2000 resamples of the "
          "scored rows)")
    print("=" * 100)
    hdr = (f"  {'vs':<9}{'metric':<16}{'gap':>9}{'95% CI':>22}{'P(>0)':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    faithful = results["FAITHFUL"]
    metrics = [("comp@win", "rows_comp", np.nanmean, +1),
               ("sufficiency", "rows_suff", np.nanmean, -1),
               ("n2 top-1", "rows_n2_top1", np.nanmean, +1),
               ("attn rho", "rows_attn_rho", np.nanmean, +1),
               ("hallucination", "rows_halluc", np.nanmean, -1)]
    for baseline in baselines:
        other = results[baseline]
        for label, key, statistic, direction in metrics:
            gap = bootstrap_gap(faithful[key], other[key], statistic=statistic)
            better = "P(better)" if direction > 0 else "P(worse) "
            p = gap["p_gt0"] if direction > 0 else 1 - gap["p_gt0"]
            print(f"  {baseline:<9}{label:<16}{gap['mean']:>+9.4f}"
                  f"   [{gap['lo']:>+8.4f}, {gap['hi']:>+8.4f}]{p:>8.2f}")
        sim_a, sim_b = faithful["sim"], other["sim"]
        if "proba" in sim_a and "proba" in sim_b:
            gap = bootstrap_auroc_gap(sim_a["y"], sim_a["proba"], sim_b["proba"])
            print(f"  {baseline:<9}{'sim AUROC':<16}{gap['mean']:>+9.4f}"
                  f"   [{gap['lo']:>+8.4f}, {gap['hi']:>+8.4f}]"
                  f"{gap['p_gt0']:>8.2f}")
        print()
    print("  Sign convention: a POSITIVE gap is better for comp/n2/attn, and "
          "the P column\n  always reads as 'probability FAITHFUL is the better "
          "arm' (sufficiency and\n  hallucination are inverted, since lower is "
          "better for both).")


def sim_table(results, order):
    print(f"\n{'=' * 100}")
    print("SIMULATABILITY (LAS-style) -- surrogate predicts FIRED vs quiet "
          "from the sentence alone")
    print("=" * 100)
    hdr = (f"  {'arm':<12}{'AUROC':>8}{'accuracy':>10}{'majority':>10}   "
           f"{'AUROC*':>8}{'accuracy*':>11}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name in order:
        s, t = results[name]["sim"], results[name]["sim_scrubbed"]
        print(f"  {name:<12}{s['auroc']:>8.3f}{s['accuracy']:>10.3f}"
              f"{s['majority']:>10.3f}   {t['auroc']:>8.3f}{t['accuracy']:>11.3f}")
    print("\n  * numerals and risk-level words removed, so the surrogate can only "
          "read NAMED DRIVERS.")
    print("  The generator is never told the fused risk or whether the alert "
          "fired, so any skill\n  here comes from the drivers the sentence "
          "chose to name.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-alerts", type=int, default=30)
    ap.add_argument("--n-null", type=int, default=30)
    ap.add_argument("--budget", type=float, default=0.05,
                    help="alert budget: fraction of test hours over threshold")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--config", default=None,
                    help="orchestrator config to explain (e.g. all-8); "
                         "default: the primary config")
    ap.add_argument("--json", default=None,
                    help="dump the metric table to this JSON path")
    ap.add_argument("--smoke", action="store_true",
                    help="print a rendered prompt and stop before the LLM")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--prompt-variant", choices=["plain", "menu-fewshot"],
                    default="plain",
                    help="menu-fewshot: enumerated candidate menu + worked "
                         "examples taken from another reader's cached cards")
    ap.add_argument("--fewshot-from", default="gemma4:12b",
                    help="reader whose plain-variant cards supply the examples")
    ap.add_argument("--fewshot-k", type=int, default=3)
    args = ap.parse_args()

    config, agent_names = (resolve_config(preferred=args.config, emb_dir=EMB)
                           if args.config else resolve_config(emb_dir=EMB))
    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    print(f"[config] {config} -> {agent_names}")

    t0 = time.time()
    bank = AgentBank.load(agent_names=agent_names)
    orch = Orchestrator(bank=bank, max_epochs=args.epochs, verbose=False)
    orch.fit(train, val)
    print(f"[fit] orchestrator refit in {time.time() - t0:.0f}s")
    if not args.no_verify:
        verify_shipped(orch, test, config)

    # --- pick the hours to explain --------------------------------------
    risk = orch.raw_risk(test)
    threshold = float(np.quantile(risk, 1.0 - args.budget))
    rows = sample_rows(test, risk, threshold, args.n_alerts, args.n_null)
    stays = rows.stay_id.unique()
    sub = test[test.stay_id.isin(set(stays))].copy()
    n_fired = int(rows["_fired"].sum())
    print(f"[sample] {len(rows)} rows from {len(stays)} disjoint stays "
          f"({n_fired} fired, {len(rows) - n_fired} quiet) "
          f"| threshold {threshold:.4f} at a {100 * args.budget:.0f}% budget "
          f"| {len(sub):,} patient-hours of trajectory")

    # --- structured drivers, straight out of the fusion's own input ------
    bank_sub = restrict_bank(bank, stays)
    X, keys = bank_sub.lookup(rows)
    if not keys.equals(rows[["stay_id", "hour"]].reset_index(drop=True)):
        raise RuntimeError("bank lookup returned a different row order")
    spans, risk_offset = column_spans(bank_sub)
    if "news2" not in spans:
        raise RuntimeError("this explainer needs the NEWS2 agent in the config")
    lo, hi = spans["news2"]
    subscores = X[:, lo:hi]
    agent_risks = {a: X[:, risk_offset + i].astype(float)
                   for i, a in enumerate(bank_sub.agent_names)}

    attn_keys, attn = orch.attention_weights(sub)
    pos = pd.MultiIndex.from_arrays(
        [attn_keys.stay_id.to_numpy(), attn_keys.hour.to_numpy()]
    ).get_indexer(pd.MultiIndex.from_arrays(
        [rows.stay_id.to_numpy(), rows.hour.to_numpy()]))
    contexts = build_contexts(rows, sub, subscores, agent_risks, attn[pos],
                              agent_names)
    fired = rows["_fired"].to_numpy()

    # --- prompt variant ---------------------------------------------------
    system, prompt_fn, fingerprint = SYSTEM_PROMPT, build_prompt, None
    if args.prompt_variant == "menu-fewshot":
        exemplars = RationaleCache(cache_path(config, args.fewshot_from))
        pool = few_shot_pool(contexts, exemplars.texts_for(contexts),
                             agent_names, fired)
        if not pool:
            raise SystemExit(f"no faithful exemplar cards cached for "
                             f"{args.fewshot_from} under config {config}")
        print(f"[fewshot] {len(pool)} exemplar cards from {args.fewshot_from}: "
              + ", ".join(f"{c['stay_id']}/{c['hour']}" for c, _ in pool))
        system = MENU_SYSTEM_PROMPT
        prompt_fn = lambda c: build_menu_prompt(  # noqa: E731
            c, examples_for(pool, c, args.fewshot_k))
        # the pool is part of the instructions: a changed pool must regenerate
        fingerprint = prompt_fingerprint(
            MENU_SYSTEM_PROMPT + "\n" + "\n".join(t for _, t in pool))

    if args.smoke:
        print(f"\n--- system prompt ---\n{system}")
        for context in contexts[:2]:
            print(f"\n--- prompt: stay {context['stay_id']} hour {context['hour']} ---")
            print(prompt_fn(context))
        print("\n[smoke] stopping before the LLM.")
        return

    # --- Part 1: generate -------------------------------------------------
    t0 = time.time()
    from text_arm.llm_orchestrator import OllamaRunner
    # think=False: a reasoning model (gemma4) otherwise spends the whole
    # num_predict budget deliberating and returns an empty sentence; harmless
    # for non-reasoning models.
    runner = OllamaRunner(model=args.model, system=system,
                          num_predict=90, think=False)
    cache = generate_rationales(
        contexts, cache_path(config, args.model, args.prompt_variant),
        model=args.model, runner=runner, prompt_fn=prompt_fn, system=system,
        fingerprint=fingerprint)
    texts = cache.texts_for(contexts)
    elapsed = time.time() - t0
    print(f"[rationale] {elapsed:.0f}s "
          f"({elapsed / max(len(contexts), 1):.2f}s per patient-hour)")

    # --- the three arms ---------------------------------------------------
    arms = {
        "FAITHFUL": texts,
        "RANDOM": shuffle_rationales(texts, seed=SEED),
        "GENERIC": generic_rationales(len(texts)),
    }

    # metrics 1 and 2 are about the alert, so they run on fired rows only
    fired_rows = rows[fired].reset_index(drop=True)
    fired_contexts = [c for c, f in zip(contexts, fired) if f]
    base_all = raw_risk_at(with_bank(orch, bank_sub), sub, rows)
    base_fired = base_all[fired]
    vocabulary = driver_vocabulary(agent_names)
    window = receptive_field_hours()
    reference = column_reference(bank, train)
    attn_base = attention_baseline(fired_contexts, agent_names)

    results = {}
    for name, arm_texts in arms.items():
        arm_fired = [t for t, f in zip(arm_texts, fired) if f]
        driver_sets = [named_drivers(t, agent_names) for t in arm_fired]
        named_agent_sets = [named_agents(t, agent_names) for t in arm_fired]

        comp = comprehensiveness(orch, bank_sub, sub, fired_rows, driver_sets,
                                 base_risk=base_fired, window=window,
                                 reference=reference)
        suff = sufficiency(orch, bank_sub, sub, fired_rows, driver_sets,
                           vocabulary, base_risk=base_fired, window=window,
                           reference=reference)
        attn_rows = attention_consistency(fired_contexts, named_agent_sets,
                                          agent_names)
        centred_rows = attention_consistency(fired_contexts, named_agent_sets,
                                             agent_names, baseline=attn_base)
        n2_rows = component_consistency(
            fired_contexts, [named_components(t) for t in arm_fired])

        results[name] = {
            "comp_at_t": float(np.mean(comp["drop_at_t"])),
            "comp_window": float(np.mean(comp["drop_window"])),
            "rel_window": float(np.nanmean(comp["rel_window"])),
            "suff": float(np.mean(suff["drop"])),
            "attn": summarise_consistency(attn_rows),
            "attn_c": summarise_consistency(centred_rows),
            "n2": summarise_consistency(n2_rows),
            "sim": simulatability(arm_texts, fired.astype(int)),
            "sim_scrubbed": simulatability([scrub_leak_words(t) for t in arm_texts],
                                           fired.astype(int)),
            "driver_sets": driver_sets,
            # per-row values, kept so the gaps between arms can be bootstrapped
            "rows_comp": comp["drop_window"],
            "rows_suff": suff["drop"],
            "rows_n2_top1": np.array([r["top1_hit"] for r in n2_rows]),
            "rows_attn_rho": np.array([r["rho"] for r in attn_rows]),
            "rows_halluc": np.array([r["unsupported"] for r in n2_rows]),
        }

    order = ["FAITHFUL", "RANDOM", "GENERIC"]
    show_examples(rows, contexts, texts,
                  [named_drivers(t, agent_names) for t in texts], base_all)
    print(f"\n  base fused risk on fired rows: mean {base_fired.mean():.3f}, "
          f"min {base_fired.min():.3f}, max {base_fired.max():.3f}  "
          f"({len(fired_rows)} rows, ablation window {window}h)")
    print(f"  ablation reference: train-split column mean "
          f"(zeroing is NOT neutral for this model -- see column_reference)")
    print("  mean attention over the scored rows: "
          + ", ".join(f"{AGENT_LABELS.get(a, a)} {w:.3f}"
                      for a, w in zip(agent_names, attn_base)))
    arm_table(results, order)
    sim_table(results, order)
    gap_table(results)

    if args.json:
        import json

        def clean(v):
            """JSON-safe: numpy arrays -> lists, numpy scalars -> floats."""
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [clean(x) for x in v]
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            return v

        payload = {"config": config, "model": args.model,
                   "prompt_variant": args.prompt_variant,
                   "fewshot_from": (args.fewshot_from
                                    if args.prompt_variant != "plain" else None),
                   "agents": list(agent_names),
                   "n_rows": len(rows), "n_fired": int(fired.sum()),
                   "arms": {name: clean({k: v for k, v in r.items()
                                         if not k.startswith("rows_")
                                         and k != "driver_sets"})
                            for name, r in results.items()}}
        Path(args.json).write_text(json.dumps(payload, indent=2, default=float))
        print(f"\n[saved] {args.json}")


if __name__ == "__main__":
    main()
