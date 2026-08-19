"""The controlled channel experiment: latent vs text vs risk-only.

The thesis says text is a lossy inter-agent channel. This experiment isolates
exactly that. Three arms, identical in agents (core-3), information source,
temporal window (the same ~29h causal-TCN receptive field), fusion
architecture, training protocol, seed, and isotonic-on-val calibration --
differing ONLY in what each agent hands the fusion every hour:

    LATENT     the agent's latent vector (32/7/32 dims) -- the shipped system
               (scores are the canonical core-3 orchestrator cache)
    TEXT       the agent's state verbalized by the deterministic templates in
               text_arm/summaries.py, embedded by frozen Bio_ClinicalBERT, and
               linearly compressed to TEXT_DIM dims per agent (PCA fit on
               train-split sentences only)
    RISK-ONLY  nothing but the calibrated scalar risk -- the information a
               majority-vote system carries

Every arm also receives the numeric risks through the same ``risk_project``
pathway, so the arm deltas isolate the STATE channel alone. TEXT's channel is
4x wider than LATENT's (128 vs 32 dims) -- deliberately generous: if latent
still wins, channel width is not the excuse. The verbalization quantizes
values to clinically realistic precision; that loss is not a bug in the arm,
it IS the property of the channel being measured.

Naive multi-agent rows for the same table: MEAN-RISK (soft vote over the three
calibrated risks) and HARD-VOTE (each agent votes at its own val-quantile
threshold; the score is the vote count 0-3). The discreteness analysis counts
each channel's distinct messages and each score's distinct output values, and
reports the achieved alert volume when thresholded at 1/2/5% budgets.

    python src/channel_arms.py [--reps 1000] [--device auto|cpu|mps]
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

from baselines_fair import clustered_bootstrap
from baselines_mimic import load_splits
from config import SEED
from metrics import auprc, auroc, ece_adaptive
from modality_features import build_lab_features, load_labs
from orchestrator.fusion import AgentBank, Orchestrator
from text_arm.summaries import build_summaries

LABEL = "label_24h"
CORE3 = ("vitals_tcn", "news2", "labs")
TEXT_DIM = 128
PCA_FIT_CAP = 200_000
EMB_CACHE = "data/cache/channel_text_embeddings.npz"
SCORE_CACHE = "data/cache/channel_arms_scores.npz"
ORCH_TEST_NPZ = "data/cache/embeddings/orchestrator_test.npz"
LLM_LEGACY_CACHE = "data/cache/text_arm_risks.jsonl"
TEXT_COLUMNS = ("vitals_text", "news2_text", "labs_text")
BUDGETS = (0.01, 0.02, 0.05)


# --- Summaries for every row --------------------------------------------------

# Verbal precision for the encoded-text channel: the values a clinician would
# actually say out loud. Exact-precision sentences produce 2.2M distinct
# vitals messages (nearly one per row) -- unencodable on CPU and clinically
# meaningless precision; rounding is part of what "verbalized" means.
VITAL_PRECISION = {"heart_rate": 5.0, "sbp": 5.0, "resp_rate": 2.0,
                   "spo2": 1.0, "temperature": 0.5}


def _coarsen_vitals(df):
    out = df.copy()
    for col, step in VITAL_PRECISION.items():
        out[col] = np.round(out[col] / step) * step
    return out


def _coarsen_labs(df):
    """Round every lab value to 2 significant figures."""
    out = df.copy()
    for col in out.columns:
        if col.startswith("lab_") and not col.endswith("_hrs_since"):
            v = out[col].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                mag = np.where(v != 0, np.floor(np.log10(np.abs(v))), 0.0)
            factor = np.power(10.0, mag - 1)
            out[col] = np.where(np.isfinite(v) & (v != 0),
                                np.round(v / factor) * factor, v)
    return out


def split_summaries(splits, bank):
    """The three agents' sentences for every row of every split.

    Reuses the text arm's deterministic templates: each sentence is a pure
    function of the agent's own before-t inputs and calibrated risk, so the
    text channel carries the same evidence the latent channel compresses --
    written at verbal precision (see VITAL_PRECISION) and without the
    heart-rate range detail, both to keep the message space encodable.
    """
    labs = load_labs()
    out = {}
    for name, df in splits.items():
        ordered = df.sort_values(["stay_id", "hour"]).reset_index(drop=True)
        X, _ = bank.lookup(ordered)
        latents, risks = bank.split_features(X)
        lab_feats = build_lab_features(ordered[["stay_id", "hour"]], labs)
        out[name] = build_summaries(
            _coarsen_vitals(ordered), _coarsen_labs(lab_feats),
            latents["news2"],
            {a: risks[:, i] for i, a in enumerate(bank.agent_names)},
            include_ranges=False)
    return out


# --- Sentence encoding (frozen ClinicalBERT -> PCA) ----------------------------

def unique_sentences(summaries):
    """All distinct sentences per text column, plus per-split code arrays."""
    uniques, codes = {}, {}
    for col in TEXT_COLUMNS:
        values = pd.concat([s[col] for s in summaries.values()],
                           ignore_index=True)
        cats = pd.Categorical(values)
        uniques[col] = np.asarray(cats.categories)
        offsets = np.cumsum([0] + [len(s) for s in summaries.values()])
        codes[col] = {
            name: cats.codes[offsets[i]:offsets[i + 1]]
            for i, name in enumerate(summaries)
        }
    return uniques, codes


def _fingerprint(uniques):
    h = hashlib.sha256()
    for col in TEXT_COLUMNS:
        for s in uniques[col]:
            h.update(s.encode())
        h.update(b"|")
    from agents.note_encoder import MODEL_NAME
    h.update(f"{MODEL_NAME}|{TEXT_DIM}|{PCA_FIT_CAP}|{SEED}|coarse-v1".encode())
    return h.hexdigest()[:16]


def encode_uniques(uniques, train_codes, device="cpu", batch_size=256,
                   cache_path=EMB_CACHE):
    """PCA-compressed embeddings of every unique sentence, cached by content.

    Single pass per column: every unique sentence is encoded once (raw 768-dim
    kept in memory -- the coarsened message space is small enough), the PCA is
    fit on a seeded, capped subset of TRAIN-appearing sentences only (val/test
    text never shapes the axes), then everything is projected.
    """
    fingerprint = _fingerprint(uniques)
    cache = Path(cache_path)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if str(z["fingerprint"]) == fingerprint:
            print(f"[encode] cache hit ({fingerprint})")
            return {col: z[f"proj_{col}"] for col in TEXT_COLUMNS}, \
                {col: float(z[f"evr_{col}"]) for col in TEXT_COLUMNS}

    from agents.note_encoder import embed_texts, load_encoder
    tokenizer, model = load_encoder(device=device)
    rng = np.random.default_rng(SEED)

    projected, explained = {}, {}
    for col in TEXT_COLUMNS:
        sentences = list(uniques[col])
        train_idx = np.unique(train_codes[col])
        train_idx = train_idx[train_idx >= 0]
        if len(train_idx) > PCA_FIT_CAP:
            train_idx = rng.choice(train_idx, PCA_FIT_CAP, replace=False)
        print(f"[encode] {col}: {len(sentences):,} unique sentences "
              f"(PCA fit on {len(train_idx):,} train-appearing)", flush=True)

        raw = np.empty((len(sentences), 768), dtype=np.float32)
        import time
        t0 = time.time()
        for i in range(0, len(sentences), batch_size):
            raw[i:i + batch_size] = embed_texts(
                tokenizer, model, sentences[i:i + batch_size], device=device)
            if i and i % (batch_size * 100) == 0:
                rate = i / (time.time() - t0)
                print(f"  {i:,}/{len(sentences):,} ({rate:,.0f}/s, "
                      f"~{(len(sentences) - i) / rate / 60:.0f} min left)",
                      flush=True)

        pca = PCA(n_components=min(TEXT_DIM, len(train_idx)),
                  random_state=SEED, svd_solver="randomized").fit(raw[train_idx])
        explained[col] = float(pca.explained_variance_ratio_.sum())
        projected[col] = pca.transform(raw).astype(np.float32)
        del raw
        print(f"[encode] {col}: PCA-{projected[col].shape[1]} keeps "
              f"{100 * explained[col]:.1f}% of embedding variance", flush=True)

    payload = {"fingerprint": np.array(fingerprint)}
    for col in TEXT_COLUMNS:
        payload[f"proj_{col}"] = projected[col]
        payload[f"evr_{col}"] = np.array(explained[col])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **payload)
    return projected, explained


# --- Arm banks ------------------------------------------------------------------

def _keys_and_risks(bank, splits):
    """Per-split (keys, risks) from the latent bank, in sorted row order."""
    out = {}
    for name, df in splits.items():
        ordered = df.sort_values(["stay_id", "hour"]).reset_index(drop=True)
        X, keys = bank.lookup(ordered)
        _, risks = bank.split_features(X)
        out[name] = (keys, risks.astype(np.float32))
    return out


def build_text_bank(bank, splits, summaries, projected, codes):
    """AgentBank whose 'latents' are the PCA-compressed sentence embeddings."""
    kr = _keys_and_risks(bank, splits)
    all_keys, blocks = [], []
    for name in splits:
        keys, risks = kr[name]
        per_agent = [projected[col][codes[col][name]] for col in TEXT_COLUMNS]
        blocks.append(np.concatenate(per_agent + [risks], axis=1))
        all_keys.append(keys)
    matrix = np.concatenate(blocks, axis=0).astype(np.float32)
    keys = pd.concat(all_keys, ignore_index=True)
    dims = {a: projected[c].shape[1]
            for a, c in zip(("vitals_tcn", "news2", "labs"),
                            TEXT_COLUMNS)}
    return AgentBank(keys, matrix, dims, list(CORE3))


def build_risk_bank(bank, splits):
    """AgentBank whose 'latent' per agent is its own calibrated risk (dim 1)."""
    kr = _keys_and_risks(bank, splits)
    all_keys, blocks = [], []
    for name in splits:
        keys, risks = kr[name]
        blocks.append(np.concatenate(
            [risks[:, [i]] for i in range(risks.shape[1])] + [risks], axis=1))
        all_keys.append(keys)
    matrix = np.concatenate(blocks, axis=0).astype(np.float32)
    keys = pd.concat(all_keys, ignore_index=True)
    return AgentBank(keys, matrix, {a: 1 for a in CORE3}, list(CORE3))


# --- Naive multi-agent scores ------------------------------------------------------

def hard_vote_scores(risks_val, risks_test, budget):
    """Vote count 0-3: each agent votes at its OWN val-quantile threshold."""
    thresholds = np.quantile(risks_val, 1.0 - budget, axis=0)
    return (risks_test >= thresholds).sum(axis=1).astype(float), thresholds


# --- Discreteness ---------------------------------------------------------------------

def achieved_rates(scores, budgets=BUDGETS):
    """Alert volume actually achieved when thresholding at each budget's
    quantile -- discrete scores overshoot (the NEWS2 integer problem)."""
    out = {}
    for b in budgets:
        thr = np.quantile(scores, 1.0 - b)
        out[b] = float((scores >= thr).mean())
    return out


def reparse_legacy_llm(path=LLM_LEGACY_CACHE):
    """Re-parse the legacy LLM replies with the FIXED parser.

    Returns (n rows, distinct risks, n changed by the 0.9-bug fix), or None if
    the cache is absent. Output-discreteness evidence for the LLM text channel;
    provenance: pre-h6fix cohort, llama3.1:8b, buggy parser at generation time.
    """
    from text_arm.llm_orchestrator import parse_risk
    p = Path(path)
    if not p.exists():
        return None
    old_risks, new_risks = [], []
    with p.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "reply" not in rec:
                continue
            old_risks.append(np.nan if rec.get("risk") is None else rec["risk"])
            new_risks.append(parse_risk(rec["reply"]))
    old_risks, new_risks = np.array(old_risks), np.array(new_risks)
    ok = ~np.isnan(new_risks)
    changed = int((ok & ~np.isnan(old_risks)
                   & (np.abs(new_risks - old_risks) > 1e-9)).sum())
    return {"n": len(new_risks), "distinct": len(np.unique(new_risks[ok])),
            "changed_by_fix": changed}


# --- The experiment ----------------------------------------------------------------------

def run(reps=1000, device="auto"):
    if device == "auto":
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    splits = load_splits()
    splits = {k: splits[k] for k in ("train", "val", "test")}
    test_ordered = splits["test"].sort_values(
        ["stay_id", "hour"]).reset_index(drop=True)
    y = test_ordered[LABEL].to_numpy(dtype=float)
    y_val = splits["val"].sort_values(
        ["stay_id", "hour"])[LABEL].to_numpy(dtype=float)

    print("[bank] loading core-3 latents + risks...")
    bank = AgentBank.load(agent_names=list(CORE3))
    kr = _keys_and_risks(bank, splits)
    risks_val, risks_test = kr["val"][1], kr["test"][1]

    print("[text] building deterministic agent sentences...")
    summaries = split_summaries(splits, bank)
    uniques, codes = unique_sentences(summaries)
    message_counts = {
        col: {name: int(len(np.unique(codes[col][name])))
              for name in splits} | {"total": int(len(uniques[col]))}
        for col in TEXT_COLUMNS
    }

    projected, explained = encode_uniques(uniques, codes_train := {
        col: codes[col]["train"] for col in TEXT_COLUMNS
    }, device=device)

    # ---- train the two new arms (LATENT = canonical core-3 cache) -----------
    z = np.load(ORCH_TEST_NPZ)
    if not ((z["stay_id"] == test_ordered["stay_id"].to_numpy()).all()
            and (z["hour"] == test_ordered["hour"].to_numpy()).all()):
        raise ValueError("canonical orchestrator cache misaligned with test")
    scores = {"LATENT": z["risk"].astype(float)}
    calibrated = {"LATENT": scores["LATENT"]}

    arms = {"TEXT": build_text_bank(bank, splits, summaries, projected, codes),
            "RISK-ONLY": build_risk_bank(bank, splits)}
    for arm, arm_bank in arms.items():
        print(f"[{arm}] training the same fusion on the {arm} channel...")
        orch = Orchestrator(bank=arm_bank, verbose=True)
        orch.fit(splits["train"], splits["val"])
        scores[arm] = orch.raw_risk(splits["test"]).astype(float)
        calibrated[arm] = orch.emit(splits["test"]).risk.astype(float)

    # ---- naive multi-agent rows ---------------------------------------------
    scores["MEAN-RISK"] = risks_test.mean(axis=1).astype(float)
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        risks_val.mean(axis=1), y_val)
    calibrated["MEAN-RISK"] = iso.transform(scores["MEAN-RISK"])
    vote5, _ = hard_vote_scores(risks_val, risks_test, 0.05)
    vote5_val, _ = hard_vote_scores(risks_val, risks_val, 0.05)
    scores["HARD-VOTE@5%"] = vote5
    calibrated["HARD-VOTE@5%"] = IsotonicRegression(out_of_bounds="clip").fit(
        vote5_val, y_val).transform(vote5)

    print(f"[bootstrap] {reps} patient-level resamples...")
    boots = clustered_bootstrap(y, scores,
                                test_ordered["patient_id"].to_numpy(),
                                reference="LATENT", reps=reps)

    _report(y, scores, calibrated, boots, message_counts, explained,
            risks_val, risks_test)
    _save(SCORE_CACHE, test_ordered, y, scores, calibrated, boots,
          message_counts, explained, reps)
    return scores, boots


def _report(y, scores, calibrated, boots, message_counts, explained,
            risks_val, risks_test):
    print(f"\nCHANNEL EXPERIMENT -- same agents, information, fusion, and "
          f"protocol; only the channel differs\n({len(y):,} test hours; "
          f"patient-clustered 95% CIs)\n")
    hdr = (f"  {'arm':<14}{'AUPRC':>8}  {'95% CI':<17}{'AUROC':>7}  "
           f"{'95% CI':<17}{'ECE-em':>8}{'ΔAUPRC vs LATENT':>20}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name in ("LATENT", "TEXT", "RISK-ONLY", "MEAN-RISK", "HARD-VOTE@5%"):
        b = boots[name]
        delta = b["delta_auprc_vs_ref"]
        dtxt = ("(reference)" if delta is None else
                f"{auprc(y, scores['LATENT']) - auprc(y, scores[name]):+.4f} "
                f"[{delta[0]:+.3f}, {delta[1]:+.3f}]")
        print(f"  {name:<14}{auprc(y, scores[name]):>8.3f}  "
              f"[{b['auprc_ci'][0]:.3f}, {b['auprc_ci'][1]:.3f}]"
              f"{auroc(y, scores[name]):>7.3f}  "
              f"[{b['auroc_ci'][0]:.3f}, {b['auroc_ci'][1]:.3f}]"
              f"{ece_adaptive(y, calibrated[name]):>8.4f}{dtxt:>20}")
    print("  (deltas are LATENT minus arm: positive = latent channel ahead)")

    # ---- discreteness ---------------------------------------------------
    print("\nDISCRETENESS -- channel cardinality and output thresholdability\n")
    print("  Distinct messages per agent channel (TEXT arm input):")
    for col, counts in message_counts.items():
        print(f"    {col:<13} train {counts['train']:>9,}  val "
              f"{counts['val']:>9,}  test {counts['test']:>9,}  "
              f"total {counts['total']:>9,}  "
              f"(PCA keeps {100 * explained[col]:.1f}%)")
    print("\n  Distinct output values and achieved alert volume at each budget:")
    print(f"    {'score':<14}{'distinct':>10}   target->achieved")
    for name in ("LATENT", "TEXT", "RISK-ONLY", "MEAN-RISK", "HARD-VOTE@5%"):
        s = scores[name]
        rates = achieved_rates(s)
        r = "  ".join(f"{100*b:.0f}%->{100*v:.1f}%" for b, v in rates.items())
        print(f"    {name:<14}{len(np.unique(s)):>10,}   {r}")
    for b in BUDGETS:
        votes, thr = hard_vote_scores(risks_val, risks_test, b)
        rate_ge1 = float((votes >= 1).mean())
        rate_ge2 = float((votes >= 2).mean())
        print(f"    HARD-VOTE@{100*b:.0f}%: 4 values; alarm at >=2 votes fires "
              f"on {100*rate_ge2:.1f}% of hours (>=1: {100*rate_ge1:.1f}%) -- "
              f"budget cannot be tuned between steps")

    legacy = reparse_legacy_llm()
    if legacy:
        print(f"\n  LLM text channel (legacy llama3.1:8b cache, "
              f"{legacy['n']:,} generations): {legacy['distinct']} distinct "
              f"risk values ever produced;\n  the parse_risk fix changes "
              f"{legacy['changed_by_fix']:,} cached rows "
              f"('0.9'-style replies were read as 0.009).")


def _save(path, test_ordered, y, scores, calibrated, boots, message_counts,
          explained, reps):
    meta = {"seed": SEED, "reps": reps, "text_dim": TEXT_DIM,
            "message_counts": message_counts, "pca_explained": explained,
            "boots": boots}
    payload = {"stay_id": test_ordered["stay_id"].to_numpy(),
               "hour": test_ordered["hour"].to_numpy(),
               "patient_id": test_ordered["patient_id"].to_numpy(),
               "y": y, "meta_json": np.array(json.dumps(meta))}
    for name, s in scores.items():
        payload[f"raw_{name}"] = s
    for name, s in calibrated.items():
        payload[f"cal_{name}"] = s
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"\n[cache] channel-arm scores + provenance -> {path}")


def _arm_val_and_test_scores(splits, device="cpu"):
    """Raw val+test scores for TEXT and RISK-ONLY, cached back into SCORE_CACHE.

    The main run cached only test scores; the timing race needs val scores to
    freeze thresholds. Retraining under the same seed and protocol reproduces
    the identical models (verified against the cached test scores); the encode
    cache makes this cheap. Once the ``rawval_*`` keys exist this function is a
    pure cache read.
    """
    z = dict(np.load(SCORE_CACHE, allow_pickle=True))
    if all(f"rawval_{arm}" in z for arm in ("TEXT", "RISK-ONLY")):
        return {arm: (z[f"rawval_{arm}"].astype(float),
                      z[f"raw_{arm}"].astype(float))
                for arm in ("TEXT", "RISK-ONLY")}

    print("[timing] re-deriving arm val scores (encode cache makes this cheap)")
    # Reproduce the ORIGINAL run's threading exactly: there, load_encoder()
    # had pinned torch to 8 threads before the arms trained, and CPU reduction
    # order changes with thread count -- unpinned, early stopping lands on a
    # different epoch and the refit drifts visibly from the cached scores.
    # (The audit's determinism finding, demonstrated live.)
    import torch
    torch.set_num_threads(8)
    bank = AgentBank.load(agent_names=list(CORE3))
    summaries = split_summaries(splits, bank)
    uniques, codes = unique_sentences(summaries)
    projected, _ = encode_uniques(
        uniques, {col: codes[col]["train"] for col in TEXT_COLUMNS},
        device=device)
    arms = {"TEXT": build_text_bank(bank, splits, summaries, projected, codes),
            "RISK-ONLY": build_risk_bank(bank, splits)}
    out = {}
    for arm, arm_bank in arms.items():
        print(f"[timing] retraining {arm} for val emissions...")
        orch = Orchestrator(bank=arm_bank, verbose=False)
        orch.fit(splits["train"], splits["val"])
        raw_te = orch.raw_risk(splits["test"]).astype(float)
        drift = float(np.abs(raw_te - z[f"raw_{arm}"]).max())
        if drift > 1e-6:
            print(f"[timing] WARNING: {arm} refit drifts from cache "
                  f"(max |diff| {drift:.2e}); keeping cached test scores")
        raw_va = orch.raw_risk(splits["val"]).astype(float)
        z[f"rawval_{arm}"] = raw_va
        out[arm] = (raw_va, z[f"raw_{arm}"].astype(float))
    np.savez_compressed(SCORE_CACHE, **z)
    print(f"[timing] val scores cached back into {SCORE_CACHE}")
    return out


def timing_leg(reps=2000, device="cpu"):
    """Which channel buys the earliness? Paired first-alarm vs NEWS2 per arm.

    Discrimination alone cannot separate "latents matter" from
    "trajectory-over-risks matters". This race can: RISK-ONLY shares the causal
    TCN but carries no latent; MEAN-RISK carries the same risks with NO
    temporal model at all. Val-frozen thresholds, matched to NEWS2's achieved
    val volume, patient-clustered CIs -- the exact protocol of the shipped
    orchestrator's timing headline.
    """
    from fair_capability import (frozen_thresholds, load_orch,
                                 paired_first_alarm_diffs, summarize_paired)

    splits = load_splits()
    splits = {k: splits[k] for k in ("train", "val", "test")}
    val_ordered = splits["val"].sort_values(
        ["stay_id", "hour"]).reset_index(drop=True)
    test_ordered = splits["test"].sort_values(
        ["stay_id", "hour"]).reset_index(drop=True)
    news2_val = val_ordered["news2"].to_numpy(dtype=float)
    news2_test = test_ordered["news2"].to_numpy(dtype=float)
    stay_to_patient = dict(zip(test_ordered["stay_id"],
                               test_ordered["patient_id"]))
    all_alertable = np.ones(len(news2_val), dtype=bool)

    bank = AgentBank.load(agent_names=list(CORE3))
    kr = _keys_and_risks(bank, splits)
    arm_scores = {"LATENT": (load_orch(splits["val"], "val"),
                             load_orch(splits["test"], "test"))}
    arm_scores.update(_arm_val_and_test_scores(splits, device=device))
    arm_scores["MEAN-RISK"] = (kr["val"][1].mean(axis=1).astype(float),
                               kr["test"][1].mean(axis=1).astype(float))

    print(f"\nTIMING BY CHANNEL -- paired first-alarm vs NEWS2, thresholds "
          f"frozen on val\n(diff > 0 = arm fires earlier; {reps}-rep "
          f"patient-clustered CIs)\n")
    hdr = (f"  {'arm':<11}{'budget':>7}{'n2%':>6}{'arm%':>6}{'n_pair':>7}"
           f"{'median':>7}  {'95% CI':<14}{'win':>5}  {'95% CI':<14}{'ties':>5}"
           f"{'sign_p':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for arm, (s_val, s_test) in arm_scores.items():
        for budget in BUDGETS:
            thr_n2, thr_arm, _ = frozen_thresholds(news2_val, s_val,
                                                   all_alertable, budget)
            diffs = paired_first_alarm_diffs(test_ordered, news2_test, s_test,
                                             thr_n2, thr_arm)
            r = summarize_paired(diffs, stay_to_patient, reps=reps)
            print(f"  {arm:<11}{f'{budget*100:.0f}%':>7}"
                  f"{100*float((news2_test >= thr_n2).mean()):>5.1f}%"
                  f"{100*float((s_test >= thr_arm).mean()):>5.1f}%"
                  f"{r['n_pair']:>7}{r['median']:>7.1f}  "
                  f"[{r['median_ci'][0]:>4.1f}, {r['median_ci'][1]:>4.1f}]"
                  f"{100*r['win_rate']:>4.0f}%  "
                  f"[{100*r['win_rate_ci'][0]:>3.0f}%, "
                  f"{100*r['win_rate_ci'][1]:>3.0f}%]"
                  f"{r['ties']:>5}{r['sign_p']:>10.2g}")
    print("\n  LATENT = canonical core-3 cache; TEXT / RISK-ONLY retrained "
          "identically; MEAN-RISK\n  has NO temporal model -- if RISK-ONLY "
          "wins timing and MEAN-RISK does not, the\n  mechanism is the causal "
          "TCN over the risk trajectory, not the channel.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    ap.add_argument("--timing", action="store_true",
                    help="run the paired lead-time race per arm (after `run`)")
    args = ap.parse_args()
    if args.timing:
        timing_leg(reps=max(args.reps, 2000),
                   device="cpu" if args.device == "auto" else args.device)
    else:
        run(reps=args.reps, device=args.device)
