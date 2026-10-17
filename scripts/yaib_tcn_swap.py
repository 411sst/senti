"""T26 -- swap Sentinel's vitals TCN for YAIB's TCN: standalone and in the flagship.

    python scripts/yaib_tcn_swap.py                  # search + 10 seeds + heads + eval
    python scripts/yaib_tcn_swap.py --seeds 42       # one seed only
    python scripts/yaib_tcn_swap.py --heads --seed S --arch A   # worker (internal)

Arms, paired by seed:

    current  Sentinel's vitals TCN (src/agents/vitals_tcn.py, shipped settings)
    yaib     YAIB's TemporalConvNet + training recipe (src/agents/yaib_tcn.py),
             hyperparameters from a seeded random search over YAIB's own space

Both arms read the same 19 vitals channels, so the only thing that changes is
the network and its training recipe. For every seed ``s`` both agents are fit
at ``s``; the flagship is the two-head ALL-8 composite with the vitals slot
filled by that arm's agent and the other seven agents taken unchanged from the
shipped caches; both heads are fit at ``s`` on CPU, one thread, as
scripts/two_head.py does. Seed noise therefore hits both arms identically and
the per-seed difference is the architecture's.

Search: 16 configs drawn with seed 42 from YAIB's space, each trained with
YAIB's recipe at seed 42 and ranked by val AUPRC of the agent alone (the repo's
selection rule for every tuned model; YAIB's tuner minimises val loss, which is
reported alongside). Val only -- test is never touched until the final table.

Outputs (gitignored): data/cache/yaib_swap/ -- per-seed emissions, head scores,
search.json, results.json. Aggregates only.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = REPO / "data" / "cache" / "yaib_swap"
SHIPPED_EMB = REPO / "data" / "cache" / "embeddings"
OTHERS = ("news2", "labs", "ecg_meas", "gbm", "cadence", "fluid", "resp")
ARMS = {"current": "vitals_tcn", "yaib": "vitals_yaib"}
SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)       # the T18 ten-seed set
N_SEARCH = 16
SPLITS = ("train", "val", "test")


def log(msg):
    print(msg, flush=True)


def seed_dir(seed):
    return OUT / f"s{seed}"


def _labels(df):
    return df.sort_values(["stay_id", "hour"])["label_24h"].to_numpy(float)


# --- 1. search over YAIB's space (val only) ---------------------------------

def search(splits):
    from agents.yaib_tcn import YaibTCNAgent, sample_config
    from metrics import auprc
    path = OUT / "search.json"
    if path.exists():
        return json.loads(path.read_text())
    rng = np.random.default_rng(42)
    configs = [sample_config(rng) for _ in range(N_SEARCH)]
    y_val = _labels(splits["val"])
    rows = []
    for i, cfg in enumerate(configs):
        t0 = time.time()
        agent = YaibTCNAgent(**cfg, seed=42, verbose=False)
        agent._fit_model(splits["train"], splits["val"])
        raw = agent.raw_risk(splits["val"])
        r = {**cfg, "val_auprc": float(auprc(y_val, raw)),
             "val_loss": float(agent.best_val_loss), "epochs": agent.epochs_run,
             "levels": agent.net.n_levels, "receptive_field": agent.net.receptive_field,
             "params": int(sum(p.numel() for p in agent.net.parameters())),
             "seconds": round(time.time() - t0, 1)}
        rows.append(r)
        log(f"[search {i+1:>2}/{N_SEARCH}] lr {cfg['lr']:.2e} ch {cfg['num_channels']:>3} "
            f"k {cfg['kernel_size']:>3} drop {cfg['dropout']:.2f} | RF {r['receptive_field']:>3} "
            f"| {r['params']:>9,} params | {r['epochs']:>2} ep | val AUPRC {r['val_auprc']:.4f} "
            f"loss {r['val_loss']:.4f} | {r['seconds']:.0f}s")
    best = max(rows, key=lambda r: r["val_auprc"])
    by_loss = min(rows, key=lambda r: r["val_loss"])
    out = {"rows": rows, "best": best, "best_by_val_loss": by_loss}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


# --- 2. agents per seed --------------------------------------------------------

def fit_agents(splits, best, seeds):
    from agents.vitals_tcn import VitalsTCNAgent
    from agents.yaib_tcn import YaibTCNAgent
    cfg = {k: best[k] for k in ("lr", "num_channels", "kernel_size", "dropout")}
    for s in seeds:
        d = seed_dir(s)
        d.mkdir(parents=True, exist_ok=True)
        for name in OTHERS:
            for sp in SPLITS:
                link = d / f"{name}_{sp}.npz"
                if not link.exists():
                    link.symlink_to(SHIPPED_EMB / f"{name}_{sp}.npz")
        for arm, name in ARMS.items():
            if all((d / f"{name}_{sp}.npz").exists() for sp in SPLITS):
                continue
            t0 = time.time()
            agent = (VitalsTCNAgent(max_epochs=8, seed=s, verbose=False) if arm == "current"
                     else YaibTCNAgent(**cfg, seed=s, verbose=False))
            agent.fit(splits["train"], splits["val"])
            for sp in SPLITS:
                agent.emit_and_cache(splits[sp], d / f"{name}_{sp}.npz")
            log(f"[agent] seed {s} {arm:<7} fit + emitted in {time.time()-t0:.0f}s")


# --- 3. heads (worker process: CPU, one thread) --------------------------------

def heads_worker(seed, arm):
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    from baselines_mimic import load_splits
    from decompose import treatment_only_label
    from fusion_common import FusionContext
    from fusion_ladder import _strictify
    from orchestrator.fusion import AgentBank
    from sklearn.isotonic import IsotonicRegression
    from two_head import fit_head

    out_path = seed_dir(seed) / f"heads_{arm}.npz"
    if out_path.exists():
        return
    splits = load_splits()
    ctx = FusionContext(splits)
    train_o = splits["train"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    names = (ARMS[arm],) + OTHERS
    bank = AgentBank.load(splits=SPLITS, agent_names=names, emb_dir=seed_dir(seed))
    targets = {"T": (treatment_only_label(train_o), treatment_only_label(ctx.val_ordered)),
               "D": (train_o["label_24h_death"].to_numpy(float),
                     ctx.val_ordered["label_24h_death"].to_numpy(float))}
    heads = {}
    for h, (y_tr, y_va) in targets.items():
        head = fit_head(bank, splits, y_tr, y_va, train_o, ctx.val_ordered, seed, 8)
        raw_val, raw_test = head.raw_risk(splits["val"]), head.raw_risk(splits["test"])
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw_val[ctx.fit_mask],
                                                           y_va[ctx.fit_mask])
        heads[h] = {k: np.clip(_strictify(iso.transform(r), r), 0.0, 1.0)
                    for k, r in (("val", raw_val), ("test", raw_test))}
    s = {k: 1.0 - (1.0 - heads["T"][k]) * (1.0 - heads["D"][k]) for k in ("val", "test")}
    np.savez_compressed(out_path, val=s["val"], test=s["test"],
                        stay_id_test=ctx.test_ordered["stay_id"].to_numpy(),
                        hour_test=ctx.test_ordered["hour"].to_numpy())


def run_heads(seeds, parallel):
    jobs = [(s, arm) for s in seeds for arm in ARMS
            if not (seed_dir(s) / f"heads_{arm}.npz").exists()]
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    running = []
    t0 = time.time()
    while jobs or running:
        while jobs and len(running) < parallel:
            s, arm = jobs.pop(0)
            logf = open(seed_dir(s) / f"heads_{arm}.log", "w")
            p = subprocess.Popen([sys.executable, __file__, "--heads", "--seed", str(s),
                                  "--arch", arm], env=env, stdout=logf, stderr=subprocess.STDOUT)
            running.append((p, s, arm, logf))
        time.sleep(5)
        for item in list(running):
            p, s, arm, logf = item
            if p.poll() is not None:
                logf.close()
                running.remove(item)
                if p.returncode != 0:
                    raise RuntimeError(f"head job seed {s} {arm} failed; see "
                                       f"{seed_dir(s) / f'heads_{arm}.log'}")
                log(f"[heads] seed {s} {arm:<7} done ({time.time()-t0:.0f}s elapsed)")


# --- 4. evaluation -------------------------------------------------------------

def evaluate(splits, seeds, search_out, reps=300):
    from agents.base import AgentOutput
    from fusion_common import FusionContext, delta_ci, paired_auprc_draws
    from metrics import auprc, auroc
    ctx = FusionContext(splits)
    y = ctx.y_test
    scores, agent_risk = {}, {}
    for s in seeds:
        for arm, name in ARMS.items():
            z = np.load(seed_dir(s) / f"heads_{arm}.npz")
            assert (z["stay_id_test"] == ctx.test_ordered["stay_id"].to_numpy()).all()
            scores[(s, arm)] = (z["val"], z["test"])
            e = AgentOutput.load(seed_dir(s) / f"{name}_test.npz")
            assert (e.keys["stay_id"].to_numpy() == ctx.test_ordered["stay_id"].to_numpy()).all()
            agent_risk[(s, arm)] = e.risk.astype(float)

    model_arrays = {f"f_{s}_{a}": scores[(s, a)][1] for s in seeds for a in ARMS}
    model_arrays.update({f"a_{s}_{a}": agent_risk[(s, a)] for s in seeds for a in ARMS})
    t0 = time.time()
    draws = paired_auprc_draws(y, model_arrays, ctx.patient_test, reps=reps)
    log(f"[eval] {reps} patient-clustered resamples x {len(model_arrays)} models "
        f"in {time.time()-t0:.0f}s")

    per_seed = []
    for s in seeds:
        r = {"seed": s}
        for a in ARMS:
            r[f"flagship_auprc_{a}"] = float(auprc(y, scores[(s, a)][1]))
            r[f"flagship_auroc_{a}"] = float(auroc(y, scores[(s, a)][1]))
            r[f"agent_auprc_{a}"] = float(auprc(y, agent_risk[(s, a)]))
            r[f"agent_auroc_{a}"] = float(auroc(y, agent_risk[(s, a)]))
        r["flagship_delta"] = r["flagship_auprc_yaib"] - r["flagship_auprc_current"]
        r["flagship_delta_ci"] = list(delta_ci(draws[f"f_{s}_yaib"], draws[f"f_{s}_current"]))
        r["agent_delta"] = r["agent_auprc_yaib"] - r["agent_auprc_current"]
        r["agent_delta_ci"] = list(delta_ci(draws[f"a_{s}_yaib"], draws[f"a_{s}_current"]))
        # Lead at the 2% budget vs NEWS2, the repo's timing harness, per arm.
        for a in ARMS:
            b = ctx.evaluate_rung(*scores[(s, a)], reps=200)["budgets"][0.02]
            r[f"lead2_median_{a}"] = b["median"]
            r[f"lead2_win_{a}"] = b["win_rate"]
        per_seed.append(r)
        log(f"[seed {s:>2}] agent AUPRC current {r['agent_auprc_current']:.4f} -> YAIB "
            f"{r['agent_auprc_yaib']:.4f} (d {r['agent_delta']:+.4f} "
            f"[{r['agent_delta_ci'][0]:+.4f}, {r['agent_delta_ci'][1]:+.4f}]) | flagship "
            f"{r['flagship_auprc_current']:.4f} -> {r['flagship_auprc_yaib']:.4f} "
            f"(d {r['flagship_delta']:+.4f} [{r['flagship_delta_ci'][0]:+.4f}, "
            f"{r['flagship_delta_ci'][1]:+.4f}]) | lead@2% {r['lead2_median_current']:+.1f} -> "
            f"{r['lead2_median_yaib']:+.1f} h")

    def summary(key):
        v = np.array([r[key] for r in per_seed])
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max())}

    def clear(key, sign):
        return int(sum((r[key][0] > 0) if sign > 0 else (r[key][1] < 0) for r in per_seed))

    # Pooled over seeds: the seed-averaged delta per resample. Every model was
    # scored on the SAME patient resamples, so averaging within a resample keeps
    # the pairing; this is the test a consistent small effect needs, which
    # per-seed intervals each too wide to clear cannot give.
    pooled = {}
    for k, pre in (("flagship", "f"), ("agent", "a")):
        d = np.mean([draws[f"{pre}_{s}_yaib"] - draws[f"{pre}_{s}_current"]
                     for s in seeds], axis=0)
        pooled[k] = {"mean_delta": float(np.mean([r[f"{k}_delta"] for r in per_seed])),
                     "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                     "seeds_negative": int(sum(r[f"{k}_delta"] < 0 for r in per_seed))}

    res = {"seeds": list(seeds), "search": search_out["best"], "pooled": pooled,
           "search_best_by_val_loss": search_out["best_by_val_loss"],
           "per_seed": per_seed, "reps": reps,
           "summary": {k: summary(k) for k in (
               "agent_auprc_current", "agent_auprc_yaib", "agent_delta",
               "flagship_auprc_current", "flagship_auprc_yaib", "flagship_delta",
               "lead2_median_current", "lead2_median_yaib")},
           "ci_clear": {"agent_yaib_better": clear("agent_delta_ci", +1),
                        "agent_yaib_worse": clear("agent_delta_ci", -1),
                        "flagship_yaib_better": clear("flagship_delta_ci", +1),
                        "flagship_yaib_worse": clear("flagship_delta_ci", -1)}}
    # The seed-42 current arm should sit on the shipped flagship.
    shipped = np.load(REPO / "data" / "cache" / "two_head.npz")
    if 42 in seeds:
        res["shipped_s42_auprc"] = float(auprc(y, shipped["test_s42"]))
    return res


def report(res):
    s, c = res["summary"], res["ci_clear"]
    n = len(res["seeds"])
    b = res["search"]
    log(f"\nYAIB config (best val AUPRC of {N_SEARCH}): lr {b['lr']:.2e}, channels "
        f"{b['num_channels']}, kernel {b['kernel_size']}, dropout {b['dropout']:.2f} -> "
        f"{b['levels']} levels, RF {b['receptive_field']} h, {b['params']:,} params")
    log(f"agent alone, test AUPRC over {n} seeds: current {s['agent_auprc_current']['mean']:.4f} "
        f"(sd {s['agent_auprc_current']['sd']:.4f}) vs YAIB {s['agent_auprc_yaib']['mean']:.4f} "
        f"(sd {s['agent_auprc_yaib']['sd']:.4f}); delta {s['agent_delta']['mean']:+.4f}; "
        f"CI-clear better {c['agent_yaib_better']}/{n}, worse {c['agent_yaib_worse']}/{n}")
    log(f"flagship, test AUPRC over {n} seeds: current {s['flagship_auprc_current']['mean']:.4f} "
        f"(sd {s['flagship_auprc_current']['sd']:.4f}) vs YAIB {s['flagship_auprc_yaib']['mean']:.4f} "
        f"(sd {s['flagship_auprc_yaib']['sd']:.4f}); delta {s['flagship_delta']['mean']:+.4f}; "
        f"CI-clear better {c['flagship_yaib_better']}/{n}, worse {c['flagship_yaib_worse']}/{n}")
    for k in ("agent", "flagship"):
        pl = res["pooled"][k]
        log(f"pooled over seeds, {k}: mean delta {pl['mean_delta']:+.4f}, 95% CI "
            f"[{pl['ci'][0]:+.4f}, {pl['ci'][1]:+.4f}], YAIB lower on "
            f"{pl['seeds_negative']}/{n} seeds")
    log(f"lead@2% median, mean over seeds: current {s['lead2_median_current']['mean']:+.2f} h "
        f"vs YAIB {s['lead2_median_yaib']['mean']:+.2f} h")
    if "shipped_s42_auprc" in res:
        r42 = next(r for r in res["per_seed"] if r["seed"] == 42)
        log(f"check: seed-42 current arm {r42['flagship_auprc_current']:.4f} vs shipped "
            f"flagship {res['shipped_s42_auprc']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--parallel", type=int, default=5,
                    help="concurrent head jobs (~3 GB RAM each)")
    ap.add_argument("--agents-only", action="store_true",
                    help="stop after the search and agent fits")
    ap.add_argument("--heads", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--arch", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.heads:
        heads_worker(args.seed, args.arch)
        return

    from baselines_mimic import load_splits
    seeds = [int(x) for x in args.seeds.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    t0 = time.time()
    so = search(splits)
    log(f"[search] best by val AUPRC: {so['best']} ({time.time()-t0:.0f}s)")
    fit_agents(splits, so["best"], seeds)
    if args.agents_only:
        return
    run_heads(seeds, args.parallel)
    res = evaluate(splits, seeds, so)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    report(res)
    log(f"[done] {time.time()-t0:.0f}s -> {(OUT / 'results.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
