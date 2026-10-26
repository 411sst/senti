"""Does the rationale describe what the orchestrator actually did?

Three offline metrics, no clinicians, no human annotation. Each one uses the
model's **own structure** as ground truth, which is what makes the benchmark
automatic and reproducible:

1. :func:`comprehensiveness` -- switch off the drivers the sentence names and
   see whether the fused risk falls. A faithful rationale names things the model
   depended on, so removing them should hurt. (ERASER-style; here "remove"
   means zeroing the corresponding latent block or NEWS2 sub-score, which for a
   sub-score is exactly the clinical counterfactual "this parameter is normal".)
   :func:`sufficiency` is the complement -- keep *only* the named drivers -- and
   it is what stops "name every driver" from being a winning strategy.
2. :func:`attention_consistency` / :func:`component_consistency` -- rank the
   drivers the sentence names against the attention weights and sub-scores the
   model actually produced at that hour.
3. :func:`simulatability` -- LAS-style: can a small surrogate recover the
   model's alert decision from the rationale text alone?

None of these is meaningful without the anti-gaming controls in
:mod:`explain.rationales`. A metric that scores a shuffled or generic sentence
as highly as a real one is measuring fluency, not faithfulness, so every metric
here is reported for all three arms.

**Ablation is causal-safe by construction.** Zeroing a driver at hours ``<= t``
can only move outputs at hours ``>= t``; we read the risk at ``t``. Sampling at
most one row per stay means every row's perturbation is confined to its own
stay, so an entire arm's ablation is a single forward pass.
"""

import copy

import numpy as np
import pandas as pd

from agents.base import KEY_COLUMNS
from config import SEED
from mimic_cohort import NEWS2_COMPONENTS
from orchestrator.fusion import AgentBank


def receptive_field_hours(kernel_size=3, dilations=(1, 2, 4), convs_per_block=2):
    """How many hours of history feed the fused risk at one hour.

    Two causal convolutions per block, each adding ``(k-1) * dilation`` of
    history, plus the current hour: 29 for the shipped settings.
    """
    return 1 + sum(convs_per_block * (kernel_size - 1) * d for d in dilations)


# --- addressing drivers as columns of the bank matrix ----------------------

def column_spans(bank):
    """``({agent: (lo, hi)}, risk_offset)`` for the bank's feature layout.

    Mirrors :meth:`AgentBank.split_features`, but as integer slices so we can
    zero them in the numpy matrix rather than in a torch tensor.
    """
    spans, offset = {}, 0
    for name in bank.agent_names:
        dim = bank.latent_dims[name]
        spans[name] = (offset, offset + dim)
        offset += dim
    return spans, offset


def driver_columns(bank, driver):
    """Which matrix columns a driver id owns, or ``None`` if it is not present.

    ``agent:X`` owns X's whole latent block plus its calibrated-risk entry --
    removing an agent means removing everything it contributed. ``news2:C``
    owns the single sub-score column for parameter ``C``.
    """
    spans, risk_offset = column_spans(bank)
    kind, _, what = driver.partition(":")
    if kind == "agent":
        if what not in spans:
            return None
        lo, hi = spans[what]
        return list(range(lo, hi)) + [risk_offset + bank.agent_names.index(what)]
    if kind == "news2":
        if "news2" not in spans or what not in NEWS2_COMPONENTS:
            return None
        lo, _ = spans["news2"]
        return [lo + NEWS2_COMPONENTS.index(what)]
    raise ValueError(f"unknown driver id {driver!r}")


def column_reference(bank, df):
    """Per-column reference values -- the "no information" input for ablation.

    **Zero is not neutral here, and assuming it was gave nonsense.** Measured on
    the shipped model: replacing every driver with zero moves the mean fused
    risk on fired alerts from 0.499 *up* to 0.763, because the agents' latents
    are nowhere near zero-centred (median |train column mean| 0.47, max 4.10).
    An all-zero vector is an extreme out-of-distribution point, so "comprehensive-
    ness by zeroing" measures distance from that point rather than importance.

    Replacing the same drivers with their **train-split column means** moves the
    risk to 0.031 -- essentially the cohort base rate, which is exactly what an
    uninformative input should produce. So the mean is the default reference and
    zero is kept only for the literal ERASER-style form.
    """
    X, _ = bank.lookup(df)
    return X.mean(axis=0).astype(np.float32)


def restrict_bank(bank, stay_ids):
    """A bank holding only the given stays -- so ablation copies stay small.

    The orchestrator is fitted against the full bank; this is purely an
    analysis-time view. ``latent_dims`` and ``agent_names`` are preserved, which
    is all the network reads.
    """
    keep = bank.keys["stay_id"].isin(set(stay_ids)).to_numpy()
    return AgentBank(bank.keys[keep].reset_index(drop=True),
                     bank.matrix[keep], bank.latent_dims, bank.agent_names)


def ablated_bank(bank, targets, reference=None):
    """Copy of ``bank`` with each target's drivers replaced on its own hours.

    ``targets`` is a list of ``{"stay_id", "hours", "drivers"}``. Hours absent
    from the bank (before the start of a stay) are simply skipped.

    ``reference`` is the per-column "no information" value from
    :func:`column_reference`; ``None`` falls back to zero, which is the literal
    ERASER form but is *not* neutral for this model -- see that function.
    """
    matrix = bank.matrix.copy()
    index = pd.MultiIndex.from_arrays([bank.keys["stay_id"].to_numpy(),
                                       bank.keys["hour"].to_numpy()])
    for target in targets:
        if not target["drivers"]:
            continue
        hours = np.asarray(target["hours"])
        want = pd.MultiIndex.from_arrays(
            [np.full(len(hours), target["stay_id"]), hours])
        pos = index.get_indexer(want)
        pos = pos[pos >= 0]
        if len(pos) == 0:
            continue
        columns = sorted({c for d in target["drivers"]
                          for c in (driver_columns(bank, d) or [])})
        if columns:
            fill = (0.0 if reference is None
                    else np.asarray(reference, dtype=np.float32)[columns])
            matrix[np.ix_(pos, columns)] = fill
    return AgentBank(bank.keys, matrix, bank.latent_dims, bank.agent_names)


def with_bank(orch, bank):
    """Shallow clone of ``orch`` reading a different bank.

    Shares the trained network and the fitted calibrator -- we are perturbing
    the *input*, never the model.
    """
    clone = copy.copy(orch)
    clone.bank = bank
    return clone


# --- reading the fused risk at specific rows -------------------------------

def raw_risk_at(orch, df, rows):
    """Uncalibrated fused probability at ``rows``' ``(stay_id, hour)``.

    Raw, not calibrated, on purpose: isotonic regression is a step function, so
    a real change in the model's score can be flattened to exactly zero by the
    calibrator. Comprehensiveness measured on calibrated risk would silently
    understate itself.
    """
    ordered = df.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    risk = orch.raw_risk(df)
    if len(risk) != len(ordered):
        raise ValueError(f"emitted {len(risk)} risks for {len(ordered)} rows")
    index = pd.MultiIndex.from_arrays([ordered["stay_id"].to_numpy(),
                                       ordered["hour"].to_numpy()])
    want = pd.MultiIndex.from_arrays([rows["stay_id"].to_numpy(),
                                      rows["hour"].to_numpy()])
    pos = index.get_indexer(want)
    if (pos < 0).any():
        raise ValueError("sampled rows are not all inside the frame being scored")
    return risk[pos]


def _targets(rows, driver_sets, mode, window):
    """Ablation targets for one arm: which hours of which stay lose which drivers."""
    out = []
    for (_, row), drivers in zip(rows.iterrows(), driver_sets):
        hour = int(row["hour"])
        hours = (np.array([hour]) if mode == "at_t"
                 else np.arange(hour - window + 1, hour + 1))
        out.append({"stay_id": int(row["stay_id"]), "hours": hours,
                    "drivers": set(drivers)})
    return out


def comprehensiveness(orch, bank, sub, rows, driver_sets, base_risk=None,
                      window=None, reference=None):
    """Risk drop when the named drivers are removed from the model's input.

    Reported two ways, because a trajectory model can recover a driver it lost
    at hour ``t`` from the hours just before it:

    ``at_t``
        remove the drivers at hour ``t`` only. A lower bound -- autocorrelation
        lets the TCN reconstruct much of what was deleted.
    ``window``
        remove them across the whole causal receptive field ending at ``t``.
        This is the honest "the model could not have seen this driver"
        condition and is the headline number.

    "Remove" means replace with ``reference`` (see :func:`column_reference`).
    """
    window = window or receptive_field_hours()
    if base_risk is None:
        base_risk = raw_risk_at(with_bank(orch, bank), sub, rows)

    out = {"base": np.asarray(base_risk, dtype=float)}
    for mode in ("at_t", "window"):
        targets = _targets(rows, driver_sets, mode, window)
        perturbed = ablated_bank(bank, targets, reference=reference)
        out[mode] = raw_risk_at(with_bank(orch, perturbed), sub, rows)
        out[f"drop_{mode}"] = out["base"] - out[mode]
    # relative drop keeps rows comparable across very different base risks
    with np.errstate(divide="ignore", invalid="ignore"):
        out["rel_window"] = np.where(out["base"] > 0,
                                     out["drop_window"] / out["base"], np.nan)
    return out


def sufficiency(orch, bank, sub, rows, driver_sets, vocabulary, base_risk=None,
                window=None, reference=None):
    """Risk drop when everything EXCEPT the named drivers is removed.

    The guard against gaming comprehensiveness by naming every driver: a
    rationale that names everything scores a perfect comprehensiveness and a
    terrible sufficiency, because "keep only what I named" then keeps
    everything and removes nothing informative. A rationale that names the few
    drivers that matter scores well on both. **Lower is better**: a small drop
    means the named drivers alone reproduce the decision.
    """
    window = window or receptive_field_hours()
    if base_risk is None:
        base_risk = raw_risk_at(with_bank(orch, bank), sub, rows)
    complements = [set(vocabulary) - set(named) for named in driver_sets]
    targets = _targets(rows, complements, "window", window)
    perturbed = ablated_bank(bank, targets, reference=reference)
    kept = raw_risk_at(with_bank(orch, perturbed), sub, rows)
    return {"kept": kept, "drop": np.asarray(base_risk, dtype=float) - kept}


# --- metric 2: does the sentence name what the model weighted? -------------

def _spearman(mask, importance):
    """Spearman rho, or nan when the rationale named all or none of the items."""
    from scipy.stats import spearmanr
    if mask.sum() == 0 or mask.sum() == len(mask):
        return float("nan")
    if np.allclose(importance, importance[0]):
        return float("nan")          # nothing to rank against
    return float(spearmanr(mask, importance).statistic)


def _consistency(importance, mask, labels):
    named = mask.astype(bool)
    return {
        "rho": _spearman(mask, importance),
        "top1_hit": (float("nan") if np.allclose(importance, importance[0])
                     else float(bool(named[int(np.argmax(importance))]))),
        "mean_named": float(importance[named].mean()) if named.any() else float("nan"),
        "mean_unnamed": (float(importance[~named].mean()) if (~named).any()
                         else float("nan")),
        "n_named": int(named.sum()),
        "labels": labels,
    }


def attention_baseline(contexts, agent_names):
    """Mean attention per agent across the rows being scored."""
    stacked = np.array([[c["attention"][a] for a in agent_names]
                        for c in contexts], dtype=float)
    return stacked.mean(axis=0)


def attention_consistency(contexts, named_agent_sets, agent_names,
                          baseline=None):
    """Per row: do the agents named match the attention the fusion gave them?

    ``baseline`` subtracts each agent's mean weight before ranking, and on this
    model that is the only version with any discriminative power. The fitted
    attention head is close to constant -- ``labs`` is the argmax on 99.7% of
    test hours and every agent's weight has sd 0.02-0.04 -- so against raw
    weights "always name labs" is a winning strategy and a shuffled rationale
    scores the same as a real one. Centring asks the sharper question: does the
    sentence name the agent that is unusually high *for this hour*?
    """
    rows = []
    for context, named in zip(contexts, named_agent_sets):
        importance = np.array([context["attention"][a] for a in agent_names])
        if baseline is not None:
            importance = importance - np.asarray(baseline, dtype=float)
        mask = np.array([1.0 if a in named else 0.0 for a in agent_names])
        rows.append(_consistency(importance, mask, agent_names))
    return rows


def component_consistency(contexts, named_component_sets):
    """Per row: do the NEWS2 parameters named match the highest sub-scores?

    Also records ``unsupported`` -- the fraction of parameters the sentence
    named whose sub-score at this hour is actually **zero**, i.e. normal. That
    is a direct hallucination rate: NEWS2 sub-scores are handed to the model
    verbatim in the prompt, so naming a zero-scoring parameter is inventing a
    finding rather than misreading one.
    """
    rows = []
    for context, named in zip(contexts, named_component_sets):
        importance = np.array([context["subscores"][c] for c in NEWS2_COMPONENTS])
        mask = np.array([1.0 if c in named else 0.0 for c in NEWS2_COMPONENTS])
        row = _consistency(importance, mask, NEWS2_COMPONENTS)
        chosen = importance[mask.astype(bool)]
        row["unsupported"] = (float((chosen == 0).mean()) if len(chosen)
                              else float("nan"))
        rows.append(row)
    return rows


def summarise_consistency(rows):
    """Mean over rows, ignoring the ones where a correlation is undefined."""
    def mean(key):
        values = np.array([r[key] for r in rows], dtype=float)
        values = values[np.isfinite(values)]
        return float(values.mean()) if len(values) else float("nan")

    finite_rho = sum(1 for r in rows if np.isfinite(r["rho"]))
    out = {"rho": mean("rho"), "top1_hit": mean("top1_hit"),
           "mean_named": mean("mean_named"), "mean_unnamed": mean("mean_unnamed"),
           "n_named": mean("n_named"), "n_rho": finite_rho, "n_rows": len(rows)}
    if rows and "unsupported" in rows[0]:
        out["unsupported"] = mean("unsupported")
    return out


# --- metric 3: can a surrogate recover the decision from the text? ---------

def simulatability(texts, alerts, seed=SEED, n_splits=5):
    """LAS-style: out-of-fold skill of a text-only surrogate predicting the alert.

    A TF-IDF + logistic-regression surrogate is deliberately weak. The question
    is not "can a strong model reverse-engineer the orchestrator" but "does the
    sentence carry the decision signal at all"; a small surrogate answers that
    without being able to memorise the cohort.

    **Leakage control.** The generator never saw the fused risk or the alert
    flag (see :mod:`explain.rationales`), so the only path from text back to the
    decision is through the drivers the sentence names. Sampling one row per
    stay makes every row its own group, so cross-validation cannot put two
    hours of the same patient on both sides of a fold.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    from metrics import auroc

    y = np.asarray(alerts, dtype=int)
    majority = float(max(y.mean(), 1 - y.mean()))
    blank = {"auroc": float("nan"), "accuracy": float("nan"),
             "majority": majority, "n": len(y)}
    if len(np.unique(y)) < 2 or not any(t.strip() for t in texts):
        return blank

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
    folds = StratifiedKFold(n_splits=min(n_splits, int(min(np.bincount(y)))),
                            shuffle=True, random_state=seed)
    if folds.get_n_splits() < 2:
        return blank
    try:
        proba = cross_val_predict(pipeline, list(texts), y, cv=folds,
                                  method="predict_proba")[:, 1]
    except ValueError:
        return blank              # e.g. every rationale scrubbed to nothing
    return {"auroc": float(auroc(y, proba)),
            "accuracy": float(((proba >= 0.5).astype(int) == y).mean()),
            "majority": majority, "n": len(y), "proba": proba, "y": y}


# --- is a gap between two arms bigger than noise? --------------------------

def bootstrap_gap(a, b, statistic=np.nanmean, reps=2000, seed=SEED):
    """Paired bootstrap CI for ``statistic(a) - statistic(b)`` over rows.

    Sampling is one row per stay, so resampling rows *is* the stay-level
    resample the rest of the project's confidence intervals use. Paired --
    both arms are recomputed on the same resampled rows, because they are
    scored on identical patient-hours and only the sentence differs.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, len(a), len(a))
        diffs[i] = statistic(a[idx]) - statistic(b[idx])
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {"mean": float(np.nanmean(diffs)), "lo": float(lo), "hi": float(hi),
            "p_gt0": float(np.nanmean(diffs > 0))}


def bootstrap_auroc_gap(y, proba_a, proba_b, reps=2000, seed=SEED):
    """Paired bootstrap CI for the AUROC gap between two surrogates.

    Both arms' out-of-fold predictions are scored on the same resampled rows,
    so the interval reflects the difference between the rationales rather than
    two independent estimates of surrogate skill.
    """
    from metrics import auroc
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(reps):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(auroc(y[idx], np.asarray(proba_a)[idx])
                     - auroc(y[idx], np.asarray(proba_b)[idx]))
    if not diffs:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_gt0": float("nan")}
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean": float(diffs.mean()), "lo": float(lo), "hi": float(hi),
            "p_gt0": float((diffs > 0).mean())}
