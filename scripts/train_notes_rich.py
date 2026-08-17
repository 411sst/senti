"""Does a WINDOW of radiology reports beat the single most recent one?

    python scripts/train_notes_rich.py                     # standalone sweep
    python scripts/train_notes_rich.py --fusion            # ... and in the orchestrator

Four arms, everything else held identical (same cohort, same three-way split,
same frozen Bio_ClinicalBERT embeddings, same isotonic-on-val calibration,
seed 42):

    notes (shipped)   the single most recent report -- the incumbent
    recent            same information, new code path -- the CONTROL, so any
                      difference below is attributable to aggregation and not
                      to the rewrite
    mean              mean embedding over the window
    rich              recent + mean + (recent - mean of older) -- the delta
                      block is exactly zero when only one report exists, so any
                      gain it buys comes from having more than one

Reported three ways, because the honest verdict differs between them:
standalone over ALL hours (dominated by coverage), over the NOTE-AVAILABLE
subset (what the text can actually be judged on), and inside the fusion (does
it survive contact with the strong structured agents?).
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

from agents.notes_agent import NotesAgent, presence_only_auprc  # noqa: E402
from agents.notes_rich_agent import RichNotesAgent, count_only_auprc  # noqa: E402
from baselines_mimic import load_splits  # noqa: E402
from metrics import auprc, auroc, ece  # noqa: E402
from mimic_cohort import EVENT_TYPES  # noqa: E402

LABEL = "label_24h"
EMB = REPO / "data" / "cache" / "embeddings"


def evaluate(name, risk, covered, test_sorted, types):
    y = test_sorted[LABEL].to_numpy(dtype=float)
    per_type = {t: auprc(test_sorted[f"{LABEL}_{t}"].to_numpy()[covered],
                         risk[covered]) for t in types}
    return {
        "name": name,
        "coverage": float(covered.mean()),
        "auprc_all": auprc(y, risk),
        "auroc_all": auroc(y, risk),
        "ece": ece(y, np.clip(risk, 0, 1)),
        "auprc_cov": auprc(y[covered], risk[covered]),
        "auroc_cov": auroc(y[covered], risk[covered]),
        "per_type": per_type,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--window", type=float, default=72.0)
    ap.add_argument("--fusion", action="store_true")
    ap.add_argument("--orch-epochs", type=int, default=8)
    args = ap.parse_args()

    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    test_sorted = test.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    y = test_sorted[LABEL].to_numpy(dtype=float)
    types = list(EVENT_TYPES)

    results, emissions = [], {}

    # -- incumbent: the shipped single-note agent -------------------------
    t0 = time.time()
    shipped = NotesAgent(max_epochs=args.epochs, verbose=False).fit(train, val)
    risk = shipped.emit(test).risk.astype(float)
    covered = shipped.has_note(test)
    results.append(evaluate("notes (shipped)", risk, covered, test_sorted, types))
    presence_floor = presence_only_auprc(y[covered], covered[covered].astype(float))
    print(f"[fit] notes (shipped)   {time.time() - t0:>5.0f}s  "
          f"coverage {100 * covered.mean():.1f}%")

    # -- windowed variants -------------------------------------------------
    for aggregation in ("recent", "mean", "rich"):
        t0 = time.time()
        agent = RichNotesAgent(aggregation=aggregation, window=args.window,
                               max_epochs=args.epochs, verbose=False)
        agent.fit(train, val)
        out = agent.emit(test)
        cov, counts = agent.coverage(test)
        results.append(evaluate(f"window {aggregation}", out.risk.astype(float),
                                cov, test_sorted, types))
        emissions[aggregation] = (agent, out, cov, counts)
        print(f"[fit] window {aggregation:<10} {time.time() - t0:>5.0f}s  "
              f"coverage {100 * cov.mean():.1f}%  "
              f"{counts[cov].mean():.1f} reports per covered hour")

    _, _, cov, counts = emissions["rich"]
    count_floor = count_only_auprc(y[cov], counts[cov])

    # -- table --------------------------------------------------------------
    print(f"\n{'=' * 104}")
    print(f"STANDALONE -- test {len(y):,} patient-hours, prevalence "
          f"{100 * y.mean():.2f}% (note-available {100 * y[cov].mean():.2f}%)")
    print("=" * 104)
    hdr = (f"  {'arm':<18}{'cover':>7}{'AUPRC all':>11}{'AUROC all':>11}"
           f"{'ECE':>8}   {'AUPRC cov':>11}{'AUROC cov':>11}   "
           + "".join(f"{t:>9}" for t in types))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        print(f"  {r['name']:<18}{100 * r['coverage']:>6.1f}%"
              f"{r['auprc_all']:>11.4f}{r['auroc_all']:>11.4f}{r['ece']:>8.4f}   "
              f"{r['auprc_cov']:>11.4f}{r['auroc_cov']:>11.4f}   "
              + "".join(f"{r['per_type'][t]:>9.4f}" for t in types))
    print(f"\n  Floors on the note-available subset -- what a model could score "
          f"WITHOUT reading anything:")
    print(f"    prevalence (random)            {y[cov].mean():.4f}")
    print(f"    note COUNT alone               {count_floor:.4f}   "
          f"<- clinician-suspicion proxy (CLAUDE.md rule #4); "
          f"count features are OFF in the model")
    print(f"    note PRESENCE alone (all hours) {presence_floor:.4f}")

    if not args.fusion:
        return

    # -- fusion -------------------------------------------------------------
    from orchestrator.fusion import AgentBank, Orchestrator

    agent, _, _, _ = emissions["rich"]
    for split_name, df in (("train", train), ("val", val), ("test", test)):
        agent.emit_and_cache(df, EMB / f"notes_rich_{split_name}.npz")
    print(f"\n[cache] notes_rich_*.npz written")

    core = ("vitals_tcn", "news2", "labs")
    configs = {
        "core-3": core,
        "core-3 + notes": core + ("notes",),
        "core-3 + notes_rich": core + ("notes_rich",),
    }
    fused = {}
    for name, agents in configs.items():
        t0 = time.time()
        bank = AgentBank.load(agent_names=agents)
        orch = Orchestrator(bank=bank, max_epochs=args.orch_epochs, verbose=False)
        orch.fit(train, val)
        raw = orch.raw_risk(test)
        _, attn = orch.attention_weights(test)
        fused[name] = {"raw": raw, "agents": agents,
                       "attn": dict(zip(agents, attn.mean(axis=0)))}
        print(f"[fit] {name:<22} {time.time() - t0:>5.0f}s")

    print(f"\n{'=' * 104}")
    print("IN FUSION -- does the richer text survive the strong structured agents?")
    print("=" * 104)
    hdr = (f"  {'config':<22}{'AUPRC':>10}{'AUROC':>9}   "
           + "".join(f"{t:>9}" for t in types) + "    notes attention")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, r in fused.items():
        note_key = [a for a in r["agents"] if a.startswith("notes")]
        attn = f"{r['attn'][note_key[0]]:.3f}" if note_key else "--"
        per_type = {t: auprc(test_sorted[f"{LABEL}_{t}"].to_numpy(), r["raw"])
                    for t in types}
        print(f"  {name:<22}{auprc(y, r['raw']):>10.4f}{auroc(y, r['raw']):>9.4f}   "
              + "".join(f"{per_type[t]:>9.4f}" for t in types)
              + f"    {attn:>14}")

    # -- is any fusion difference bigger than noise? ------------------------
    print(f"\n  Paired stay-level bootstrap vs core-3 (1000 resamples):")
    rng = np.random.default_rng(0)
    stays = test_sorted["stay_id"].to_numpy()
    uniq = np.unique(stays)
    idx = {s: np.flatnonzero(stays == s) for s in uniq}
    base = fused["core-3"]["raw"]
    for name in ("core-3 + notes", "core-3 + notes_rich"):
        deltas = []
        for _ in range(1000):
            pick = rng.choice(uniq, len(uniq), replace=True)
            ii = np.concatenate([idx[s] for s in pick])
            if y[ii].sum() < 5:
                continue
            deltas.append(auprc(y[ii], fused[name]["raw"][ii])
                          - auprc(y[ii], base[ii]))
        deltas = np.array(deltas)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        print(f"    {name:<22} AUPRC {deltas.mean():+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  P(>0)={100 * (deltas > 0).mean():>3.0f}%")

    # and the direct head-to-head that answers the question
    a = fused["core-3 + notes_rich"]["raw"]
    b = fused["core-3 + notes"]["raw"]
    deltas = []
    for _ in range(1000):
        pick = rng.choice(uniq, len(uniq), replace=True)
        ii = np.concatenate([idx[s] for s in pick])
        if y[ii].sum() < 5:
            continue
        deltas.append(auprc(y[ii], a[ii]) - auprc(y[ii], b[ii]))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"    {'notes_rich MINUS notes':<22} AUPRC {deltas.mean():+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}]  P(>0)={100 * (deltas > 0).mean():>3.0f}%"
          f"   <- the question")


if __name__ == "__main__":
    main()
