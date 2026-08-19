"""Does the notes agent PREDICT deterioration, or read a report that DESCRIBES it?

    python scripts/audit_note_contamination.py
    python scripts/audit_note_contamination.py --blackout 6

Radiology reports are the one modality that can narrate the outcome. A chest
film filed shortly before intubation may say "endotracheal tube tip 4 cm above
the carina"; a film before a line-dependent vasopressor start may describe the
central line going in. Every such note passes the pipeline's timing rules --
it genuinely was available before hour ``t`` -- so no leakage audit that only
checks *timing* can catch it. This one checks **content**.

Three questions, in increasing strength:

1. **Textual.** Do outcome-describing terms get more frequent as a note sits
   closer to the event, and does the term family track the event TYPE? Rising
   frequency alone proves little (ICU patients get daily tube-position films);
   frequency that spikes near the event *and* matches the event type is the
   signature of description rather than prediction.
2. **Signal.** Is the agent's discrimination concentrated on exactly those
   patient-hours whose note sits close to the event?
3. **Blackout.** Censor every note filed within K hours before an event, refit,
   and see what survives.

**Reading the blackout honestly.** It uses event times to remove data, so it is
an audit-only counterfactual and never a deployable model. It is also
asymmetric evidence: if performance holds, contamination is ruled out and that
is a strong conclusion. If performance drops, it is ambiguous -- the censored
notes may have been describing the event, or may have been legitimately warning
of it, which is the thing we want the agent to do. Check 1 is what
disambiguates.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from baselines_mimic import load_splits  # noqa: E402
from metrics import auprc, auroc  # noqa: E402

LABEL = "label_24h"
CACHE = REPO / "data" / "cache"
COHORT = CACHE / "mimic_cohort_hourly.parquet"
NOTE_INDEX = CACHE / "note_index.parquet"
NOTE_TEXTS = CACHE / "note_texts.parquet"

# Vocabulary tied to each arm of the composite outcome. The point is not that
# these words are forbidden -- they appear in routine ICU imaging all the time --
# but whether their rate climbs as the note approaches the event, and whether
# the family that climbs matches the event that happened.
EVENT_TERMS = {
    "vent": ("endotracheal", "et tube", "ett", "intubat", "extubat",
             "tracheostomy", "trach tube", "ventilator", "mechanical ventilation",
             "endotracheal tube"),
    "vaso": ("central venous", "central line", "swan-ganz", "swan ganz",
             "introducer", "cordis", "picc", "vasopressor", "pressor",
             "right atrial catheter"),
    "death": ("cardiac arrest", "cardiopulmonary arrest", "cpr",
              "resuscitat", "expired", "postmortem", "post-mortem",
              "code blue", "asystole"),
}
DETERIORATION_TERMS = (
    "worsening", "interval worsening", "new consolidation", "pulmonary edema",
    "increasing opacit", "new opacit", "interval increase", "progression of",
    "new effusion", "increased effusion",
)

# The sharp version of the question. A device term on its own proves nothing --
# an ICU patient already on a ventilator generates a tube-position film every
# morning, and "ETT tip 4 cm above the carina, unchanged" is routine
# surveillance, not a description of anything happening. What would betray
# description of the *event* is NEW-placement language: the report saying the
# device has just appeared. These patterns look for that, within 60 characters
# of a device term so that "new opacity" does not count.
PLACEMENT_PATTERNS = (
    r"(?:new|newly|interval)\s+(?:\w+\s+){0,3}(?:placement|placed|insertion|inserted)",
    r"interval\s+(?:\w+\s+){0,2}(?:intubation|placement)",
    r"has\s+been\s+(?:placed|inserted|intubated)",
    r"now\s+(?:intubated|in\s+place|present)",
    r"(?:new|newly)\s+(?:endotracheal|ett|tracheostomy|central|picc|swan)",
    r"(?:endotracheal|ett|tracheostomy|central line|picc)\s+(?:\w+\s+){0,3}"
    r"(?:newly\s+)?(?:placed|inserted)",
)

LEAD_BINS = [0, 3, 6, 12, 24, 48, np.inf]
LEAD_LABELS = ["0-3h", "3-6h", "6-12h", "12-24h", "24-48h", ">48h"]


def compile_terms(terms):
    """Case-insensitive alternation with WORD BOUNDARIES on each term.

    Without the boundaries, short terms fire inside ordinary words -- "ett"
    matched "better" and "settled", inflating every vent-term count. ``\\b``
    anchors to the term's own edges, so multi-word phrases keep matching as
    phrases.
    """
    return re.compile(
        "|".join(rf"\b{re.escape(t)}\b" for t in terms), re.IGNORECASE)


def stay_events(cohort):
    """One row per stay: its event hour and type (NaN where there is none)."""
    return (cohort.groupby("stay_id")[["event_h", "event_type"]].first()
            .reset_index())


def note_lead_frame(cohort, note_index):
    """Every note, tagged with how long BEFORE its stay's event it was filed."""
    events = stay_events(cohort)
    notes = note_index.merge(events, on="stay_id", how="inner")
    notes["lead"] = notes["event_h"] - notes["note_hour"]
    # notes filed after the event are irrelevant -- no prediction hour can see
    # them without the event already having happened
    notes["bucket"] = pd.cut(notes["lead"], LEAD_BINS, labels=LEAD_LABELS,
                             right=True)
    return notes


# ------------------------------------------------------------------ 1 ------

def check_textual(cohort, note_index, texts):
    print(f"\n{'=' * 100}")
    print("1. TEXTUAL -- do reports start describing the outcome as it approaches?")
    print("=" * 100)

    notes = note_lead_frame(cohort, note_index).merge(
        texts[["note_id", "text"]], on="note_id", how="inner")
    has_event = notes["event_h"].notna()
    with_event = notes[has_event & notes["bucket"].notna()]

    # control: notes from stays that never deteriorate
    control = notes[~has_event]
    patterns = {k: compile_terms(v) for k, v in EVENT_TERMS.items()}
    patterns["deterioration"] = compile_terms(DETERIORATION_TERMS)
    # the sharp one: language saying a device has JUST appeared
    patterns["NEW-placement"] = re.compile("|".join(PLACEMENT_PATTERNS),
                                           re.IGNORECASE)

    print(f"  {len(with_event):,} pre-event notes across "
          f"{with_event.stay_id.nunique():,} stays; "
          f"{len(control):,} control notes from event-free stays\n")

    hdr = (f"  {'lead before event':<20}{'notes':>9}   "
           + "".join(f"{k:>16}" for k in patterns))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label in LEAD_LABELS:
        sub = with_event[with_event["bucket"] == label]
        if sub.empty:
            continue
        rates = [f"{100 * sub.text.str.contains(p).mean():>15.1f}%"
                 for p in patterns.values()]
        print(f"  {label:<20}{len(sub):>9,}   " + "".join(rates))
    rates = [f"{100 * control.text.str.contains(p).mean():>15.1f}%"
             for p in patterns.values()]
    print(f"  {'(no-event stays)':<20}{len(control):>9,}   " + "".join(rates))

    # does the term family that spikes match the event that happened?
    print("\n  Term rate by EVENT TYPE, notes filed 0-6h before the event")
    print("  (contamination shows as a DIAGONAL: vent events -> vent terms)")
    columns = list(EVENT_TERMS) + ["NEW-placement"]
    hdr = f"    {'event type':<14}{'window':<10}{'notes':>8}   " + "".join(
        f"{k:>18}" for k in columns)
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for event_type in EVENT_TERMS:
        for window, sub in (("0-6h", with_event[with_event["lead"] <= 6]),
                            (">24h", with_event[with_event["lead"] > 24])):
            sub = sub[sub["event_type"] == event_type]
            if sub.empty:
                continue
            cells = "".join(
                f"{100 * sub.text.str.contains(patterns[k]).mean():>17.1f}%"
                for k in columns)
            print(f"    {event_type:<14}{window:<10}{len(sub):>8,}   {cells}")
    cells = "".join(f"{100 * control.text.str.contains(patterns[k]).mean():>17.1f}%"
                    for k in columns)
    print(f"    {'(no event)':<14}{'--':<10}{len(control):>8,}   {cells}")


# ------------------------------------------------------------------ 2 ------

def check_signal(test, risk, index_counts, note_to_event):
    print(f"\n{'=' * 100}")
    print("2. SIGNAL -- is the agent's discrimination concentrated near the event?")
    print("=" * 100)
    y = test[LABEL].to_numpy(dtype=float)
    covered = index_counts > 0
    negatives = (y == 0) & covered
    print(f"  {int(negatives.sum()):,} covered negative hours form the common "
          f"comparison pool\n")

    bucket = pd.cut(note_to_event, LEAD_BINS, labels=LEAD_LABELS, right=True)
    hdr = (f"  {'note filed before event':<26}{'pos hours':>11}"
           f"{'mean risk':>11}{'AUROC vs pool':>15}{'AUPRC vs pool':>15}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label in LEAD_LABELS:
        # pd.cut over an ndarray yields a Categorical, so the comparison is
        # already an ndarray -- no .to_numpy() to call
        sel = np.asarray(bucket == label) & (y == 1) & covered
        if sel.sum() < 20:
            continue
        keep = sel | negatives
        print(f"  {label:<26}{int(sel.sum()):>11,}{risk[sel].mean():>11.4f}"
              f"{auroc(y[keep], risk[keep]):>15.4f}"
              f"{auprc(y[keep], risk[keep]):>15.4f}")
    both = (y == 1) & covered
    keep = both | negatives
    print(f"  {'ALL covered positives':<26}{int(both.sum()):>11,}"
          f"{risk[both].mean():>11.4f}{auroc(y[keep], risk[keep]):>15.4f}"
          f"{auprc(y[keep], risk[keep]):>15.4f}")
    print(f"  {'covered negatives':<26}{int(negatives.sum()):>11,}"
          f"{risk[negatives].mean():>11.4f}")
    print("\n  Read the AUROC column, not the AUPRC one. Each bucket is scored "
          "against the SAME\n  208k-negative pool, so AUPRC is driven by how "
          "many positives the bucket happens\n  to hold and is not comparable "
          "across rows; AUROC is rank-based and prevalence-\n  independent, so "
          "it is. Contamination would show as risk and AUROC rising as the\n"
          "  note gets closer to the event.")


# ------------------------------------------------------------------ 3 ------

def censored_index(note_index, cohort, blackout):
    """Note index with every report filed within ``blackout`` h of an event removed."""
    events = stay_events(cohort)
    merged = note_index.merge(events, on="stay_id", how="left")
    lead = merged["event_h"] - merged["note_hour"]
    drop = merged["event_h"].notna() & (lead >= 0) & (lead <= blackout)
    kept = note_index[~drop.to_numpy()].reset_index(drop=True)
    return kept, int(drop.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregation", default="rich",
                    choices=["recent", "mean", "rich"])
    ap.add_argument("--blackout", type=float, default=6.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--skip-blackout", action="store_true")
    ap.add_argument("--textual-only", action="store_true",
                    help="check 1 only -- needs no fitted model")
    args = ap.parse_args()

    from agents.notes_rich_agent import RichNotesAgent

    cohort = pd.read_parquet(COHORT)
    note_index = pd.read_parquet(NOTE_INDEX)
    texts = pd.read_parquet(NOTE_TEXTS)
    cohort["stay_id"] = cohort.stay_id.astype("int64")
    note_index["stay_id"] = note_index.stay_id.astype("int64")

    check_textual(cohort, note_index, texts)
    if args.textual_only:
        return

    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    test_sorted = test.sort_values(["stay_id", "hour"]).reset_index(drop=True)

    t0 = time.time()
    agent = RichNotesAgent(aggregation=args.aggregation, max_epochs=args.epochs,
                           verbose=False).fit(train, val)
    print(f"\n[fit] {args.aggregation} agent in {time.time() - t0:.0f}s")

    out = agent.emit(test)
    risk = out.risk.astype(float)
    index, _ = agent._index_for(test)
    counts = index.counts

    # how long before the event was the note this hour is looking at?
    events = stay_events(cohort).set_index("stay_id")
    most_recent_hour = np.full(len(test_sorted), np.nan)
    covered = counts > 0
    most_recent_hour[covered] = (test_sorted["hour"].to_numpy()[covered]
                                 - index.ages[index.offsets[:-1][covered]])
    event_h = test_sorted["stay_id"].map(events["event_h"]).to_numpy(dtype=float)
    note_to_event = event_h - most_recent_hour

    check_signal(test_sorted, risk, counts, note_to_event)

    if args.skip_blackout:
        return
    # -------------------------------------------------------------- 3 ------
    print(f"\n{'=' * 100}")
    print(f"3. BLACKOUT -- censor every note filed within {args.blackout:g}h "
          f"before an event, then refit")
    print("=" * 100)
    kept, dropped = censored_index(note_index, cohort, args.blackout)
    path = CACHE / "note_index_blackout.parquet"
    kept.to_parquet(path, index=False)
    print(f"  removed {dropped:,} of {len(note_index):,} notes "
          f"({100 * dropped / len(note_index):.2f}%)")

    t0 = time.time()
    censored_agent = RichNotesAgent(
        aggregation=args.aggregation, max_epochs=args.epochs, verbose=False,
        note_index=str(path)).fit(train, val)
    print(f"  refit in {time.time() - t0:.0f}s")

    y = test_sorted[LABEL].to_numpy(dtype=float)
    c_risk = censored_agent.emit(test).risk.astype(float)
    c_counts = censored_agent._index_for(test)[0].counts

    types = ["vent", "vaso", "death"]
    hdr = (f"  {'arm':<20}{'coverage':>10}{'AUPRC all':>11}"
           f"{'note-available':>16}   " + "".join(f"{t:>10}" for t in types))
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))
    rows = {}
    for name, r, cnt in (("full", risk, counts),
                         (f"blackout {args.blackout:g}h", c_risk, c_counts)):
        cov = cnt > 0
        # per-type AUPRC on note-available hours: check 1 said the risk is
        # concentrated in the vaso arm, so a type-blind number would hide it
        per_type = {t: auprc(test_sorted[f"{LABEL}_{t}"].to_numpy()[cov], r[cov])
                    for t in types}
        rows[name] = per_type
        print(f"  {name:<20}{100 * cov.mean():>9.1f}%{auprc(y, r):>11.4f}"
              f"{auprc(y[cov], r[cov]):>16.4f}   "
              + "".join(f"{per_type[t]:>10.4f}" for t in types))
    cov = counts > 0
    print(f"  {'(prevalence)':<20}{'':>10}{y.mean():>11.4f}"
          f"{y[cov].mean():>16.4f}   "
          + "".join(f"{test_sorted[f'{LABEL}_{t}'].to_numpy()[cov].mean():>10.4f}"
                    for t in types))
    before, after = rows["full"], rows[f"blackout {args.blackout:g}h"]
    print(f"  {'change':<20}{'':>10}{'':>11}{'':>16}   "
          + "".join(f"{after[t] - before[t]:>+10.4f}" for t in types))

    # Is the drop bigger than resampling noise? Paired stay-level bootstrap on
    # the SAME rows, so the two arms differ only in which notes they could see.
    print(f"\n  Paired stay-level bootstrap on the blackout delta "
          f"(1000 resamples, note-available rows):")
    stays = test_sorted["stay_id"].to_numpy()[cov]
    uniq = np.unique(stays)
    where = {s: np.flatnonzero(stays == s) for s in uniq}
    y_cov, full_cov, black_cov = y[cov], risk[cov], c_risk[cov]
    per_type_cov = {t: test_sorted[f"{LABEL}_{t}"].to_numpy()[cov] for t in types}
    rng = np.random.default_rng(0)
    picks = [np.concatenate([where[s] for s in rng.choice(uniq, len(uniq),
                                                          replace=True)])
             for _ in range(1000)]

    def report(label, target):
        deltas = np.array([auprc(target[ii], black_cov[ii])
                           - auprc(target[ii], full_cov[ii])
                           for ii in picks if target[ii].sum() >= 5])
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        verdict = "drop is real" if hi < 0 else "consistent with no change"
        print(f"    {label:<18}{deltas.mean():>+9.4f}  "
              f"[{lo:>+8.4f}, {hi:>+8.4f}]  P(drop)={100*(deltas<0).mean():>3.0f}%"
              f"   {verdict}")

    report("composite", y_cov)
    for t in types:
        report(t, per_type_cov[t])
    print("\n  Holding up under blackout rules contamination OUT. A drop is "
          "ambiguous:\n  the censored notes may have described the event, or "
          "may have legitimately\n  warned of it -- check 1 is what tells the "
          "two apart.")


if __name__ == "__main__":
    main()
