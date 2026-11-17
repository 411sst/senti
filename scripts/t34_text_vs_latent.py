"""T34 (D4): text against latents on the flagship.

Two arms through the identical two-head pipeline (``two_head.fit_head`` per
head, isotonic on V_fit, ``_strictify``, noisy-OR), differing ONLY in what
each of the eight agents hands the fusion every hour:

    LATENT    the agent's latent vector (the shipped flagship, two_head.npz)
    TEXT-128  the agent's state written as a templated sentence, embedded by
              frozen Bio_ClinicalBERT (run locally), PCA-compressed per agent
              to at most 128 dims (fit on train-appearing sentences only)

Both arms also receive the eight calibrated risks through the same shared risk
pathway. Sentences: vitals, labs and NEWS2 from ``text_arm/summaries.py`` at
the August verbal precision (``channel_arms.split_summaries``); ECG, cadence,
fluid, resp and GBM from ``text_arm/agent_templates.py``. Sentences are
deduplicated before encoding and the embeddings are cached with a fingerprint.

    .venv/bin/python scripts/t34_text_vs_latent.py --stage sentences
    .venv/bin/python scripts/t34_text_vs_latent.py --stage encode --device cuda
    OMP_NUM_THREADS=1 .venv/bin/python scripts/t34_text_vs_latent.py --stage heads
    OMP_NUM_THREADS=1 .venv/bin/python scripts/t34_text_vs_latent.py --stage eval

Gate: LATENT seed 42, fit by this script's worker, must equal two_head.npz.
Outputs (all new files): data/cache/t34_text_codes.npz,
data/cache/t34_text_embeddings.npz, data/cache/t34_heads/, and the aggregate
report data/cache/t34_text_vs_latent.json.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

CACHE = REPO / "data" / "cache"
CODES = CACHE / "t34_text_codes.npz"
EMB = CACHE / "t34_text_embeddings.npz"
JOB_DIR = CACHE / "t34_heads"
OUT = CACHE / "t34_text_vs_latent.json"
SHIPPED = CACHE / "two_head.npz"
RISK_T27 = CACHE / "channel_flagship"
SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
RISK_SEEDS = (42, 1, 2)
SPLITS = ("train", "val", "test")
TEXT_DIM = 128
PCA_FIT_CAP = 200_000
EPOCHS = 8
REPS = 1000
RACE_BUDGETS = (0.01, 0.02, 0.05)
AUGUST_COLS = {"vitals_tcn": "vitals_text", "labs": "labs_text", "news2": "news2_text"}
CODE_FILES = ["src/text_arm/summaries.py", "src/text_arm/agent_templates.py",
              "src/channel_arms.py", "src/allfeat_gbm.py", "src/modality_features.py"]
HEAD_CODE = ["scripts/two_head.py", "src/distill_lead.py", "src/orchestrator/fusion.py",
             "src/fusion_common.py", "src/fusion_ladder.py", "src/decompose.py",
             "src/agents/base.py"]


def log(msg):
    print(msg, flush=True)


def _sha_files(paths):
    return {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def _bank_stats():
    from orchestrator.fusion import EMB_DIR
    from two_head import ALL8
    return [[f"{a}_{s}", (Path(EMB_DIR) / f"{a}_{s}.npz").stat().st_size,
             (Path(EMB_DIR) / f"{a}_{s}.npz").stat().st_mtime_ns]
            for a in ALL8 for s in SPLITS]


# --- stage: sentences ------------------------------------------------------------

def stage_sentences():
    from allfeat_gbm import build_allfeat_design, load_sources
    from baselines_mimic import load_splits
    from channel_arms import split_summaries
    from orchestrator.fusion import AgentBank
    from text_arm.agent_templates import TEMPLATE_VERSION, TEMPLATES, sentences
    from two_head import ALL8

    if CODES.exists():
        raise SystemExit(f"{CODES.name} exists; sentences are frozen once built")
    t0 = time.time()
    splits = load_splits()
    bank = AgentBank.load(agent_names=ALL8)
    log("[sentences] vitals, labs, NEWS2 (August templates, verbal precision)...")
    august = split_summaries(splits, bank)
    log(f"[sentences] August templates done ({time.time() - t0:.0f} s)")
    sources = load_sources()
    per_split = {}
    for s in SPLITS:
        X, _, keys = build_allfeat_design(splits[s], sources)
        ordered = splits[s].sort_values(["stay_id", "hour"]).reset_index(drop=True)
        Xb, bkeys = bank.lookup(ordered)
        _, risks = bank.split_features(Xb)
        if not ((bkeys["stay_id"].to_numpy() == keys["stay_id"].to_numpy()).all()
                and (august[s]["stay_id"].to_numpy() == keys["stay_id"].to_numpy()).all()
                and (august[s]["hour"].to_numpy() == keys["hour"].to_numpy()).all()):
            raise SystemExit(f"{s}: sentence rows are misaligned with the bank")
        per_split[s] = {}
        for agent in TEMPLATES:
            r = risks[:, list(bank.agent_names).index(agent)]
            uniq, codes = sentences(agent, X, r)
            per_split[s][agent] = np.asarray(uniq, dtype=object)[codes]
        for agent, col in AUGUST_COLS.items():
            per_split[s][agent] = august[s][col].to_numpy(dtype=object)
        log(f"[sentences] {s}: {len(keys):,} rows ({time.time() - t0:.0f} s)")
        del X
    payload = {}
    counts = {}
    for agent in ALL8:
        allv = np.concatenate([per_split[s][agent] for s in SPLITS]).astype(str)
        uniq, inv = np.unique(allv, return_inverse=True)
        offsets = np.cumsum([0] + [len(per_split[s][agent]) for s in SPLITS])
        payload[f"uniq_{agent}"] = uniq
        for i, s in enumerate(SPLITS):
            payload[f"codes_{agent}_{s}"] = inv[offsets[i]:offsets[i + 1]].astype(np.int32)
        counts[agent] = int(len(uniq))
        log(f"[sentences] {agent}: {len(uniq):,} distinct sentences")
    spec = {"version": TEMPLATE_VERSION, "code": _sha_files(CODE_FILES), "banks": _bank_stats()}
    payload["fingerprint"] = np.array(hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()).hexdigest())
    payload["counts_json"] = np.array(json.dumps(counts))
    np.savez_compressed(CODES, **payload)
    log(f"[saved] {CODES.name} ({time.time() - t0:.0f} s)")


# --- stage: encode -------------------------------------------------------------------

def _emb_fingerprint(z):
    from agents.note_encoder import MODEL_NAME
    from two_head import ALL8
    h = hashlib.sha256()
    for agent in ALL8:
        for s in z[f"uniq_{agent}"]:
            h.update(str(s).encode())
        h.update(b"|")
    h.update(f"{MODEL_NAME}|{TEXT_DIM}|{PCA_FIT_CAP}|{str(z['fingerprint'])}".encode())
    return h.hexdigest()


def stage_encode(device, batch_size):
    from agents.note_encoder import embed_texts, load_encoder
    from config import SEED
    from sklearn.decomposition import PCA
    from two_head import ALL8

    z = np.load(CODES, allow_pickle=False)
    fp = _emb_fingerprint(z)
    if EMB.exists():
        if str(np.load(EMB)["fingerprint"]) == fp:
            log("[encode] cache is current")
            return
        raise SystemExit(f"{EMB.name} exists with a different fingerprint")
    tokenizer, model = load_encoder(device=device)
    rng = np.random.default_rng(SEED)
    payload, meta = {"fingerprint": np.array(fp)}, {}
    for agent in ALL8:
        uniq = [str(s) for s in z[f"uniq_{agent}"]]
        train_idx = np.unique(z[f"codes_{agent}_train"])
        if len(train_idx) > PCA_FIT_CAP:
            train_idx = np.sort(rng.choice(train_idx, PCA_FIT_CAP, replace=False))
        t0 = time.time()
        raw = np.empty((len(uniq), 768), dtype=np.float32)
        for i in range(0, len(uniq), batch_size):
            raw[i:i + batch_size] = embed_texts(tokenizer, model, uniq[i:i + batch_size],
                                                device=device)
        # August's settings (randomized PCA, seeded, train-appearing sentences
        # only). An agent with very few distinct sentences keeps all but one
        # axis, since centred points span one dimension fewer than their count.
        n_comp = int(max(1, min(TEXT_DIM, len(train_idx) - 1)))
        pca = PCA(n_components=n_comp, random_state=SEED,
                  svd_solver="randomized" if n_comp < 64 or len(train_idx) > 256 else "full")
        pca.fit(raw[train_idx])
        payload[f"proj_{agent}"] = pca.transform(raw).astype(np.float32)
        meta[agent] = {"distinct": len(uniq), "train_distinct": int(len(train_idx)),
                       "dims": n_comp,
                       "explained": float(pca.explained_variance_ratio_.sum()),
                       "encode_seconds": round(time.time() - t0, 1)}
        log(f"[encode] {agent}: {meta[agent]}")
    payload["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(EMB, **payload)
    log(f"[saved] {EMB.name}")


# --- stage: heads ---------------------------------------------------------------------

def text_bank(splits):
    """ALL-8 AgentBank whose per-agent latent is that agent's sentence embedding."""
    import pandas as pd
    from orchestrator.fusion import AgentBank
    from two_head import ALL8
    latent = AgentBank.load(agent_names=ALL8)
    zc, ze = np.load(CODES), np.load(EMB)
    keys, blocks = [], []
    for s in SPLITS:
        ordered = splits[s].sort_values(["stay_id", "hour"]).reset_index(drop=True)
        X, k = latent.lookup(ordered)
        _, risks = latent.split_features(X)
        per = [ze[f"proj_{a}"][zc[f"codes_{a}_{s}"]] for a in ALL8]
        blocks.append(np.concatenate(per + [risks.astype(np.float32)], axis=1))
        keys.append(k)
        del X
    dims = {a: int(ze[f"proj_{a}"].shape[1]) for a in ALL8}
    del latent
    return AgentBank(pd.concat(keys, ignore_index=True),
                     np.concatenate(blocks, axis=0).astype(np.float32), dims, list(ALL8))


def head_fingerprint(arm, seed):
    import torch
    spec = {"arm": arm, "seed": seed, "epochs": EPOCHS, "torch": torch.__version__,
            "code": _sha_files(HEAD_CODE), "banks": _bank_stats()}
    if arm == "text":
        spec["emb"] = str(np.load(EMB)["fingerprint"])
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def worker(arm, seed):
    import torch
    torch.set_num_threads(1)
    from baselines_mimic import load_splits
    from decompose import treatment_only_label
    from fusion_common import FusionContext
    from fusion_ladder import _strictify
    from orchestrator.fusion import AgentBank
    from sklearn.isotonic import IsotonicRegression
    from two_head import ALL8, fit_head

    out_path = JOB_DIR / f"{arm}_s{seed}.npz"
    if out_path.exists():
        return
    t0 = time.time()
    splits = load_splits()
    ctx = FusionContext(splits)
    train_o = splits["train"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    val_o = ctx.val_ordered
    y_death = {"train": train_o["label_24h_death"].to_numpy(dtype=float),
               "val": val_o["label_24h_death"].to_numpy(dtype=float)}
    y_treat = {"train": treatment_only_label(train_o), "val": treatment_only_label(val_o)}
    bank = AgentBank.load(agent_names=ALL8) if arm == "latent" else text_bank(splits)
    log(f"[worker] {arm} s{seed}: bank {bank.n_features} features ({time.time() - t0:.0f} s)")
    heads = {}
    for name, y in (("T", y_treat), ("D", y_death)):
        head = fit_head(bank, splits, y["train"], y["val"], train_o, val_o, seed, EPOCHS)
        raw_val = head.raw_risk(splits["val"])
        raw_test = head.raw_risk(splits["test"])
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            raw_val[ctx.fit_mask], y["val"][ctx.fit_mask])
        heads[name] = {"val": _strictify(iso.transform(raw_val), raw_val),
                       "test": _strictify(iso.transform(raw_test), raw_test)}
    s_val = 1.0 - (1.0 - heads["T"]["val"]) * (1.0 - heads["D"]["val"])
    s_test = 1.0 - (1.0 - heads["T"]["test"]) * (1.0 - heads["D"]["test"])
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, val=s_val, test=s_test,
                        stay_id_test=ctx.test_ordered["stay_id"].to_numpy(),
                        fingerprint=np.array(head_fingerprint(arm, seed)))
    tmp.replace(out_path)
    log(f"[worker] {arm} s{seed} done in {time.time() - t0:.0f} s")


def run_jobs(jobs, parallel):
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    pending = [j for j in jobs if not (JOB_DIR / f"{j[0]}_s{j[1]}.npz").exists()]
    running, t0 = [], time.time()
    while pending or running:
        while pending and len(running) < parallel:
            arm, seed = pending.pop(0)
            lf = open(JOB_DIR / f"{arm}_s{seed}.log", "w")
            p = subprocess.Popen([sys.executable, __file__, "--worker", "--arm", arm,
                                  "--seed", str(seed)], env=env, stdout=lf,
                                 stderr=subprocess.STDOUT)
            running.append((p, arm, seed, lf))
        time.sleep(5)
        for item in list(running):
            p, arm, seed, lf = item
            if p.poll() is None:
                continue
            lf.close()
            running.remove(item)
            if p.returncode != 0:
                for q, *_ in running:
                    q.terminate()
                raise SystemExit(f"{arm} seed {seed} failed; see its log in {JOB_DIR}")
            log(f"[heads] {arm} seed {seed} done ({time.time() - t0:.0f} s elapsed)")


def gate_latent():
    z, c = np.load(JOB_DIR / "latent_s42.npz"), np.load(SHIPPED)
    return {s: {"identical": bool(np.array_equal(z[s], c[f"{s}_s42"])),
                "max_abs_diff": float(np.abs(z[s] - c[f"{s}_s42"]).max())}
            for s in ("val", "test")}


def stage_heads(parallel):
    run_jobs([("latent", 42)], 1)
    g = gate_latent()
    log(f"[gate] LATENT seed 42 vs two_head.npz: {g}")
    if not all(v["identical"] for v in g.values()):
        raise SystemExit("LATENT seed 42 does not reproduce two_head.npz; stopping")
    run_jobs([("text", s) for s in SEEDS], parallel)


# --- stage: eval -------------------------------------------------------------------------

def _pooled_race(ctx, pairs, reps, seed):
    """Stack per-seed paired first-alarm diffs; first-when-differ with a
    patient-clustered CI (patients resampled with every seed's stays)."""
    import pandas as pd
    from baselines_fair import bootstrap_indices
    d = pd.concat(pairs, ignore_index=True)
    diff = d["diff"].to_numpy(dtype=float)
    pid = d["stay_id"].map(ctx.stay_to_patient).to_numpy()
    wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
    rates = []
    for idx in bootstrap_indices(pid, reps, seed):
        w, lo = (diff[idx] > 0).sum(), (diff[idx] < 0).sum()
        if w + lo:
            rates.append(w / (w + lo))
    return {"n_pair": int(len(diff)), "latent_first": wins, "text_first": losses,
            "ties": int(len(diff) - wins - losses),
            "latent_first_when_differ": float(wins / (wins + losses)) if wins + losses else float("nan"),
            "ci": [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))],
            "median_diff_h": float(np.median(diff)) if len(diff) else float("nan")}


def stage_eval(workers):
    from baselines_mimic import load_splits
    from config import SEED
    from demo_bundle import assert_aggregate_only, first_alarm_table
    from fair_capability import paired_first_alarm_diffs
    from fusion_common import FusionContext
    from metrics import auprc, auroc
    from paper_operating_points import freeze_quantile, test_rate
    from paper_stats import metric_draws, paired_delta, pooled_delta

    t0 = time.time()
    ctx = FusionContext(load_splits())
    gate = gate_latent()
    log(f"[gate] LATENT seed 42 vs two_head.npz: {gate}")
    if not all(v["identical"] for v in gate.values()):
        raise SystemExit("gate failed")
    zt = np.load(SHIPPED)
    latent = {s: {"val": zt[f"val_s{s}"].astype(float), "test": zt[f"test_s{s}"].astype(float)}
              for s in SEEDS}
    text = {}
    for s in SEEDS:
        z = np.load(JOB_DIR / f"text_s{s}.npz")
        if str(z["fingerprint"]) != head_fingerprint("text", s):
            raise SystemExit(f"text seed {s}: fingerprint does not match the inputs")
        if not (z["stay_id_test"] == ctx.test_ordered["stay_id"].to_numpy()).all():
            raise SystemExit(f"text seed {s} is not aligned with test")
        text[s] = {"val": z["val"].astype(float), "test": z["test"].astype(float)}
    risk = {}
    for s in RISK_SEEDS:
        z = np.load(RISK_T27 / f"risk_s{s}.npz")
        assert (z["stay_id_test"] == ctx.test_ordered["stay_id"].to_numpy()).all()
        risk[s] = {"val": z["val"].astype(float), "test": z["test"].astype(float)}

    y = ctx.y_test
    named = {**{f"latent_s{s}": latent[s]["test"] for s in SEEDS},
             **{f"text_s{s}": text[s]["test"] for s in SEEDS},
             **{f"risk_s{s}": risk[s]["test"] for s in RISK_SEEDS}}
    targets = {"auprc": ("auprc", y), "auroc": ("auroc", y)}
    fns = {"auprc": auprc, "auroc": auroc}
    points = {t: {n: float(fns[t](y, v)) for n, v in named.items()} for t in targets}
    draws = metric_draws(targets, named, ctx.patient_test, reps=REPS, seed=SEED, workers=workers)
    ranking = {}
    for t in targets:
        per = {s: paired_delta(points[t][f"latent_s{s}"], points[t][f"text_s{s}"],
                               draws[t][f"latent_s{s}"], draws[t][f"text_s{s}"]) for s in SEEDS}
        ranking[t] = {
            "points": points[t],
            "latent_minus_text": {
                "pooled": pooled_delta([points[t][f"latent_s{s}"] for s in SEEDS],
                                       [points[t][f"text_s{s}"] for s in SEEDS],
                                       [draws[t][f"latent_s{s}"] for s in SEEDS],
                                       [draws[t][f"text_s{s}"] for s in SEEDS]),
                "per_seed": per,
                "seeds_ci_above_zero": int(sum(p["ci_above_zero"] for p in per.values())),
                "seeds_ci_below_zero": int(sum(p["ci"][1] < 0 for p in per.values()))},
            "text_minus_risk_t27_seeds": pooled_delta(
                [points[t][f"text_s{s}"] for s in RISK_SEEDS],
                [points[t][f"risk_s{s}"] for s in RISK_SEEDS],
                [draws[t][f"text_s{s}"] for s in RISK_SEEDS],
                [draws[t][f"risk_s{s}"] for s in RISK_SEEDS]),
            "latent_minus_risk_t27_seeds": pooled_delta(
                [points[t][f"latent_s{s}"] for s in RISK_SEEDS],
                [points[t][f"risk_s{s}"] for s in RISK_SEEDS],
                [draws[t][f"latent_s{s}"] for s in RISK_SEEDS],
                [draws[t][f"risk_s{s}"] for s in RISK_SEEDS]),
        }
    p = ranking["auprc"]["latent_minus_text"]["pooled"]
    log(f"[ranking] LATENT - TEXT pooled dAUPRC {p['delta']:+.4f} [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]")

    timing = {}
    for b in RACE_BUDGETS:
        timing[str(b)] = {}
        for rule in ("equal_volume", "protocol"):
            pairs, cells = [], {}
            for s in SEEDS:
                L, T = latent[s], text[s]
                if rule == "equal_volume":
                    # Diagnostic: each arm at the same quantile of its own TEST scores.
                    thr_l = float(np.quantile(L["test"][ctx.alertable_test], 1.0 - b))
                    thr_t = float(np.quantile(T["test"][ctx.alertable_test], 1.0 - b))
                else:
                    thr_l, thr_t = freeze_quantile(ctx, L["val"], b), freeze_quantile(ctx, T["val"], b)
                diffs = paired_first_alarm_diffs(ctx.test_ordered, T["test"], L["test"], thr_t, thr_l)
                pairs.append(diffs)
                # Same warm-up mask as the race (paired_first_alarm_diffs).
                masked = {k: np.where(ctx.alertable_test, v["test"], -np.inf)
                          for k, v in (("l", L), ("t", T))}
                tab = first_alarm_table(ctx.test_ordered, masked, {"l": thr_l, "t": thr_t})
                ev = tab[tab["event_h"].notna()]
                cl = ev["fa_l"].notna() & (ev["fa_l"] <= ev["event_h"])
                ct = ev["fa_t"].notna() & (ev["fa_t"] <= ev["event_h"])
                d = diffs["diff"].to_numpy(dtype=float)
                cells[s] = {"n_pair": int(len(d)), "latent_first": int((d > 0).sum()),
                            "text_first": int((d < 0).sum()),
                            "only_latent": int((cl & ~ct).sum()), "only_text": int((ct & ~cl).sum()),
                            "latent_rate": test_rate(ctx, L["test"], thr_l),
                            "text_rate": test_rate(ctx, T["test"], thr_t)}
            timing[str(b)][rule] = {"pooled": _pooled_race(ctx, pairs, REPS, SEED),
                                    "per_seed": cells}
            r = timing[str(b)][rule]["pooled"]
            log(f"[timing] {b:.0%} {rule:<12} LATENT first when they differ "
                f"{r['latent_first_when_differ']:.1%} [{r['ci'][0]:.1%}, {r['ci'][1]:.1%}] "
                f"(n {r['n_pair']}, ties {r['ties']})")

    lo, hi = ranking["auprc"]["latent_minus_text"]["pooled"]["ci"]
    rank_verdict = "latent better" if lo > 0 else ("text better" if hi < 0 else "tie")
    lo2, hi2 = timing["0.02"]["equal_volume"]["pooled"]["ci"]
    time_verdict = ("latent earlier" if lo2 > 0.5 else
                    ("text earlier" if hi2 < 0.5 else "no difference"))
    zc, ze = np.load(CODES), np.load(EMB)
    payload = {
        "test": "T34", "seeds": list(SEEDS), "risk_seeds": list(RISK_SEEDS), "reps": REPS,
        "gate": gate, "sentences": json.loads(str(zc["counts_json"])),
        "encoding": json.loads(str(ze["meta_json"])),
        "ranking": ranking, "timing": timing,
        "verdict": {"ranking": rank_verdict, "timing": time_verdict},
        "rules": {"ranking": "pooled dAUPRC (LATENT - TEXT) CI above zero: latent better; "
                             "below zero: text better; else tie",
                  "timing": "2% equal-volume direct race, pooled over seeds: LATENT-first "
                            "share of discordant stays, CI above 50%: latent earlier; below "
                            "50%: text earlier; else no difference"},
        "note": "aggregate-only; equal-volume thresholds use test score quantiles (diagnostic)",
    }
    payload = json.loads(json.dumps(payload, default=float))
    assert_aggregate_only(payload)
    OUT.write_text(json.dumps(payload, indent=1))
    log(f"[verdict] ranking: {rank_verdict}; timing: {time_verdict}")
    log(f"[saved] {OUT} ({time.time() - t0:.0f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("sentences", "encode", "heads", "eval"))
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--arm", choices=("latent", "text"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if args.worker:
        worker(args.arm, args.seed)
    elif args.stage == "sentences":
        stage_sentences()
    elif args.stage == "encode":
        stage_encode(args.device, args.batch_size)
    elif args.stage == "heads":
        stage_heads(args.parallel)
    elif args.stage == "eval":
        stage_eval(args.workers)
    else:
        ap.error("give --stage or --worker")


if __name__ == "__main__":
    main()
