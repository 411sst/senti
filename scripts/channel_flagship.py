"""T27 -- the channel experiment, re-run on the shipped flagship.

    OMP_NUM_THREADS=1 python scripts/channel_flagship.py          # all arms, seeds 42 1 2
    python scripts/channel_flagship.py --worker --arm latent --seed 42   # (internal)

The paper's central claim is that agents should hand the fusion LATENT VECTORS
rather than text or bare risk scores. src/channel_arms.py tested that on the
old 3-agent orchestrator; this re-runs the cheap half of it on the shipped
two-head ALL-8 flagship. The question: does the latent channel add anything
over the agents' calibrated risks alone?

Arms, all on the ALL-8 bank exactly as scripts/two_head.py builds it
(vitals_tcn, news2, labs, ecg_meas, gbm, cadence, fluid, resp; shipped caches):

    LATENT     the two-head flagship itself: two_head.fit_head per head,
               isotonic on V_fit, _strictify, noisy-OR -- imported, not copied
    RISK-ONLY  the same two-head pipeline, but each agent's latent replaced by
               its own 1-d calibrated risk (src/channel_arms.build_risk_bank's
               design, rebuilt for eight agents). Every agent's risk still also
               reaches the fusion through the shared risk pathway, so the two
               arms differ ONLY in the per-agent state channel.
    MEAN-RISK  the mean of the eight calibrated risks; nothing trained
    HARD-VOTE  each agent votes when its risk reaches its own 2% quantile on
               V_freeze; the score is the vote count, 0-8; nothing trained

Protocol is two_head.py's: train on train, early-stop each head on its own
label, isotonic on V_fit with the rank tiebreak, noisy-OR, thresholds frozen on
V_freeze through FusionContext.evaluate_rung (1/2/5% budgets, paired
first-alarm race vs NEWS2, patient-clustered medians), 1000-rep
patient-clustered bootstrap for AUPRC with ONE resample shared by every score.
Trained arms at seeds 42, 1, 2; the untrained arms have no seed.

Gate: LATENT at seed 42 must reproduce data/cache/two_head.npz test_s42 and
val_s42 element for element. If it does not, the run stops before anything
else is trusted. Heads run in worker processes, CPU, one thread each, so every
fit is bit-reproducible. Scores go to the NEW file
data/cache/channel_flagship.npz (per-job files under
data/cache/channel_flagship/); no existing cache is written. Aggregates only.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

JOB_DIR = REPO / "data" / "cache" / "channel_flagship"
OUT = REPO / "data" / "cache" / "channel_flagship.npz"
REPORT = REPO / "data" / "cache" / "channel_flagship.json"
SHIPPED = REPO / "data" / "cache" / "two_head.npz"
SEEDS = (42, 1, 2)
TRAINED = ("latent", "risk")
VOTE_BUDGET = 0.02


def log(msg):
    print(msg, flush=True)


# --- banks -------------------------------------------------------------------

def latent_bank():
    from orchestrator.fusion import AgentBank
    from two_head import ALL8
    return AgentBank.load(agent_names=ALL8)          # two_head.py's exact call


def risk_bank(bank, splits):
    """ALL-8 AgentBank whose per-agent 'latent' is that agent's calibrated risk.

    channel_arms.build_risk_bank hard-codes the core-3; same construction for
    eight. The risk block is appended unchanged, so the fusion's shared risk
    pathway sees exactly what it sees in LATENT.
    """
    from orchestrator.fusion import AgentBank
    keys, blocks = [], []
    for name in ("train", "val", "test"):
        ordered = splits[name].sort_values(["stay_id", "hour"]).reset_index(drop=True)
        X, k = bank.lookup(ordered)
        _, risks = bank.split_features(X)
        risks = risks.astype(np.float32)
        blocks.append(np.concatenate([risks[:, [i]] for i in range(risks.shape[1])]
                                     + [risks], axis=1))
        keys.append(k)
    return AgentBank(pd.concat(keys, ignore_index=True),
                     np.concatenate(blocks, axis=0).astype(np.float32),
                     {a: 1 for a in bank.agent_names}, list(bank.agent_names))


def risk_blocks(bank, ctx):
    """(val, test) calibrated-risk matrices (N, 8), in sorted row order."""
    out = []
    for frame in (ctx.val_ordered, ctx.test_ordered):
        X, _ = bank.lookup(frame)
        out.append(bank.split_features(X)[1].astype(float))
    return out


# --- worker: one trained arm at one seed --------------------------------------

def worker(arm, seed):
    """two_head.py's per-seed loop, verbatim in substance, for one arm."""
    import torch
    torch.set_num_threads(1)
    from baselines_mimic import load_splits
    from decompose import treatment_only_label
    from fusion_common import FusionContext
    from fusion_ladder import _strictify
    from sklearn.isotonic import IsotonicRegression
    from two_head import fit_head

    out_path = JOB_DIR / f"{arm}_s{seed}.npz"
    if out_path.exists():
        return
    splits = load_splits()
    ctx = FusionContext(splits)
    train_o = splits["train"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    val_o = ctx.val_ordered
    y_death = {"train": train_o["label_24h_death"].to_numpy(dtype=float),
               "val": val_o["label_24h_death"].to_numpy(dtype=float)}
    y_treat = {"train": treatment_only_label(train_o),
               "val": treatment_only_label(val_o)}
    bank = latent_bank()
    if arm == "risk":
        bank = risk_bank(bank, splits)
    heads = {}
    for name, y in (("T", y_treat), ("D", y_death)):
        head = fit_head(bank, splits, y["train"], y["val"], train_o, val_o, seed, 8)
        raw_val = head.raw_risk(splits["val"])
        raw_test = head.raw_risk(splits["test"])
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            raw_val[ctx.fit_mask], y["val"][ctx.fit_mask])
        heads[name] = {"val": _strictify(iso.transform(raw_val), raw_val),
                       "test": _strictify(iso.transform(raw_test), raw_test)}
    s_val = 1.0 - (1.0 - heads["T"]["val"]) * (1.0 - heads["D"]["val"])
    s_test = 1.0 - (1.0 - heads["T"]["test"]) * (1.0 - heads["D"]["test"])
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, val=s_val, test=s_test,
                        stay_id_test=ctx.test_ordered["stay_id"].to_numpy(),
                        hour_test=ctx.test_ordered["hour"].to_numpy())


def run_jobs(jobs, parallel):
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    pending = [j for j in jobs if not (JOB_DIR / f"{j[0]}_s{j[1]}.npz").exists()]
    running, t0 = [], time.time()
    JOB_DIR.mkdir(parents=True, exist_ok=True)
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
                raise RuntimeError(f"{arm} seed {seed} failed; see {JOB_DIR / f'{arm}_s{seed}.log'}")
            log(f"[heads] {arm:<6} seed {seed:>2} done ({time.time()-t0:.0f}s elapsed)")


def gate():
    """LATENT s42 must equal the cached flagship, element for element."""
    z, c = np.load(JOB_DIR / "latent_s42.npz"), np.load(SHIPPED)
    out = {}
    for split in ("val", "test"):
        a, b = z[split], c[f"{split}_s42"]
        out[split] = {"identical": bool(np.array_equal(a, b)),
                      "max_abs_diff": float(np.abs(a - b).max())}
    return out


# --- evaluation ---------------------------------------------------------------

def evaluate(reps=1000):
    from baselines_mimic import load_splits
    from fusion_common import FusionContext, delta_ci, paired_auprc_draws
    from metrics import auprc, auroc
    splits = load_splits()
    ctx = FusionContext(splits)
    bank = latent_bank()
    r_val, r_test = risk_blocks(bank, ctx)
    del bank

    scores = {}
    for arm in TRAINED:
        for s in SEEDS:
            z = np.load(JOB_DIR / f"{arm}_s{s}.npz")
            assert (z["stay_id_test"] == ctx.test_ordered["stay_id"].to_numpy()).all()
            scores[f"{arm}_s{s}"] = (z["val"].astype(float), z["test"].astype(float))
    scores["mean_risk"] = (r_val.mean(axis=1), r_test.mean(axis=1))
    vote_thr = np.quantile(r_val[ctx.freeze_mask], 1.0 - VOTE_BUDGET, axis=0)
    scores["hard_vote"] = ((r_val >= vote_thr).sum(axis=1).astype(float),
                           (r_test >= vote_thr).sum(axis=1).astype(float))

    t0 = time.time()
    draws = paired_auprc_draws(ctx.y_test, {k: v[1] for k, v in scores.items()},
                               ctx.patient_test, reps=reps)
    log(f"[eval] {reps} shared patient-clustered resamples x {len(scores)} scores "
        f"in {time.time()-t0:.0f}s")

    rows = {}
    for key, (sv, st) in scores.items():
        t1 = time.time()
        rung = ctx.evaluate_rung(sv, st, reps=reps)
        lo, hi = np.percentile(draws[key], [2.5, 97.5])
        row = {"auprc": rung["auprc"], "auprc_ci": [float(lo), float(hi)],
               "auroc": float(auroc(ctx.y_test, st)),
               "distinct_test_values": int(len(np.unique(st))),
               "budgets": {str(b): {k: (list(v) if isinstance(v, tuple) else v)
                                    for k, v in r.items()}
                           for b, r in rung["budgets"].items()}}
        # The paired delta vs LATENT at the same seed; the untrained arms are
        # compared with each LATENT seed and with the seed-averaged LATENT.
        seed = key.rsplit("_s", 1)[1] if key.startswith(TRAINED) else None
        if key.startswith("risk"):
            row["delta_vs_latent"] = list(delta_ci(draws[key], draws[f"latent_s{seed}"]))
        elif not key.startswith("latent"):
            lat = np.mean([draws[f"latent_s{s}"] for s in SEEDS], axis=0)
            row["delta_vs_latent_mean"] = list(delta_ci(draws[key], lat))
        rows[key] = row
        b2 = row["budgets"]["0.02"]
        log(f"  {key:<10} AUPRC {row['auprc']:.4f} [{lo:.4f}, {hi:.4f}]  AUROC "
            f"{row['auroc']:.4f}  lead@2% {b2['median']:+.1f} h (win {100*b2['win_rate']:.0f}%, "
            f"n {b2['n_pair']})  test alarm rate @2% {100*b2['model_rate']:.2f}%  "
            f"distinct {row['distinct_test_values']:,}  [{time.time()-t1:.0f}s]")

    # Pooled RISK-ONLY - LATENT over the three seeds: the seed-averaged delta
    # inside each shared resample, so the pairing survives the averaging.
    pooled = np.mean([draws[f"risk_s{s}"] - draws[f"latent_s{s}"] for s in SEEDS], axis=0)
    summary = {
        "latent_auprc_mean": float(np.mean([rows[f"latent_s{s}"]["auprc"] for s in SEEDS])),
        "risk_auprc_mean": float(np.mean([rows[f"risk_s{s}"]["auprc"] for s in SEEDS])),
        "latent_auroc_mean": float(np.mean([rows[f"latent_s{s}"]["auroc"] for s in SEEDS])),
        "risk_auroc_mean": float(np.mean([rows[f"risk_s{s}"]["auroc"] for s in SEEDS])),
        "risk_minus_latent_pooled": [float(pooled.mean()),
                                     float(np.percentile(pooled, 2.5)),
                                     float(np.percentile(pooled, 97.5))],
        "vote_thresholds_freeze_2pct": [float(t) for t in vote_thr],
    }
    np.savez_compressed(OUT, stay_id_test=ctx.test_ordered["stay_id"].to_numpy(),
                        hour_test=ctx.test_ordered["hour"].to_numpy(),
                        **{f"{k}_val": v[0] for k, v in scores.items()},
                        **{f"{k}_test": v[1] for k, v in scores.items()})
    return {"rows": rows, "summary": summary, "reps": reps, "seeds": list(SEEDS)}


def direct_race(reps=1000, budgets=(0.01, 0.02, 0.05)):
    """LATENT vs RISK-ONLY head to head, same seed, paired first alarms.

    The NEWS2 races above leave a confound: at the same val-matched threshold
    LATENT fires a little more on test than RISK-ONLY, so part of its lead
    over NEWS2 could be volume. Two threshold rules, both reported:

    val-matched   each arm at the same quantile of its OWN V_freeze scores --
                  the deployable rule; test volumes may still differ slightly
    test-matched  each arm at the same quantile of its own TEST scores, so
                  both fire on exactly the same share of test hours. Test
                  labels are never read, but test scores set the threshold:
                  a diagnostic for the volume confound, not a protocol.

    ``fair_capability.paired_first_alarm_diffs`` with RISK-ONLY in the NEWS2
    slot: diff = first alarm RISK - first alarm LATENT, so positive means
    LATENT fired first. Jointly caught stays only, as everywhere; the stays
    only one arm catches are counted alongside.
    """
    from baselines_mimic import load_splits
    from fair_capability import paired_first_alarm_diffs, summarize_paired
    from fusion_common import FusionContext
    ctx = FusionContext(load_splits())
    z = np.load(OUT)
    test = ctx.test_ordered
    event = test.groupby("stay_id")["event_h"].first()

    def caught(scores, thr):
        hit = test.loc[scores >= thr, ["stay_id", "hour"]]
        fa = hit.groupby("stay_id")["hour"].min().reindex(event.index)
        return event.notna() & fa.notna() & (fa <= event)

    rows = []
    for s in SEEDS:
        lat = {k: z[f"latent_s{s}_{k}"].astype(float) for k in ("val", "test")}
        rsk = {k: z[f"risk_s{s}_{k}"].astype(float) for k in ("val", "test")}
        for b in budgets:
            for rule in ("val-matched", "test-matched"):
                if rule == "val-matched":
                    t_l = float(np.quantile(lat["val"][ctx.freeze_mask], 1 - b))
                    t_r = float(np.quantile(rsk["val"][ctx.freeze_mask], 1 - b))
                else:
                    t_l = float(np.quantile(lat["test"], 1 - b))
                    t_r = float(np.quantile(rsk["test"], 1 - b))
                diffs = paired_first_alarm_diffs(test, rsk["test"], lat["test"], t_r, t_l)
                r = summarize_paired(diffs, ctx.stay_to_patient, reps=reps)
                c_l, c_r = caught(lat["test"], t_l), caught(rsk["test"], t_r)
                rows.append({
                    "seed": s, "budget": b, "rule": rule,
                    "n_pair": r["n_pair"], "median_h": r["median"],
                    "median_ci": list(r["median_ci"]), "latent_first": r["win_rate"],
                    "latent_first_ci": list(r["win_rate_ci"]), "ties": r["ties"],
                    "sign_p": r["sign_p"],                     # discordant pairs
                    "risk_first": float((diffs["diff"] < 0).mean()) if len(diffs) else float("nan"),
                    "rate_latent": float((lat["test"] >= t_l).mean()),
                    "rate_risk": float((rsk["test"] >= t_r).mean()),
                    "caught_latent": int(c_l.sum()), "caught_risk": int(c_r.sum()),
                    "only_latent": int((c_l & ~c_r).sum()), "only_risk": int((c_r & ~c_l).sum())})
                x = rows[-1]
                log(f"  [race] s{s:<2} {100*b:.0f}% {rule:<12} LATENT first {100*x['latent_first']:.0f}% "
                    f"[{100*x['latent_first_ci'][0]:.0f}, {100*x['latent_first_ci'][1]:.0f}], RISK first "
                    f"{100*x['risk_first']:.0f}% (sign p {x['sign_p']:.1g}), median {x['median_h']:+.1f} h "
                    f"on {x['n_pair']} pairs | "
                    f"test rates {100*x['rate_latent']:.2f}% vs {100*x['rate_risk']:.2f}% | "
                    f"caught {x['caught_latent']} vs {x['caught_risk']} (only {x['only_latent']} vs {x['only_risk']})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--race-only", action="store_true",
                    help="run only the direct LATENT vs RISK-ONLY race, from the "
                         "cached scores in data/cache/channel_flagship.npz")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--arm", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(args.arm, args.seed)
        return
    if args.race_only:
        res = json.loads(REPORT.read_text())
        res["direct_race"] = direct_race(args.reps)
        REPORT.write_text(json.dumps(res, indent=1))
        log(f"[done] direct race -> {REPORT.relative_to(REPO)}")
        return

    t0 = time.time()
    # Phase A carries the gate: nothing else is worth fitting if LATENT s42 is
    # not the cached flagship.
    run_jobs([("latent", 42), ("risk", 42), ("risk", 1)], args.parallel)
    g = gate()
    log(f"[gate] LATENT s42 vs two_head.npz: val identical {g['val']['identical']} "
        f"(max|d| {g['val']['max_abs_diff']:.3g}), test identical {g['test']['identical']} "
        f"(max|d| {g['test']['max_abs_diff']:.3g})")
    if not (g["val"]["identical"] and g["test"]["identical"]):
        REPORT.write_text(json.dumps({"gate": g, "stopped": True}, indent=1))
        raise SystemExit("[gate] FAILED -- LATENT does not reproduce the cached "
                         "flagship; stopping before any comparison.")
    run_jobs([("latent", 1), ("latent", 2), ("risk", 2)], args.parallel)
    res = evaluate(args.reps)
    res["gate"] = g
    res["direct_race"] = direct_race(args.reps)
    REPORT.write_text(json.dumps(res, indent=1))
    s = res["summary"]
    lo_hi = s["risk_minus_latent_pooled"]
    log(f"\nLATENT mean AUPRC {s['latent_auprc_mean']:.4f} | RISK-ONLY {s['risk_auprc_mean']:.4f} "
        f"| pooled RISK - LATENT {lo_hi[0]:+.4f} [{lo_hi[1]:+.4f}, {lo_hi[2]:+.4f}]")
    log(f"[done] {time.time()-t0:.0f}s -> {OUT.relative_to(REPO)}, {REPORT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
