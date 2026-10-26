"""Single-node CPU-only benchmark: what Sentinel needs on one commodity box.

    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/bench_single_node.py
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/bench_single_node.py \
        --threads 1,2,4,8 --llm-cards 3 --llm-models medgemma:4b,gemma4:12b

Answers the deployment question directly: can the whole system run on a single
compute node with no discrete GPU? Three measurements, all with the GPU hidden:

1. PREDICTION PATH -- every agent's forward pass plus the two-head fusion, at
   several CPU thread counts, in the two shapes that matter operationally:
   scoring the current hour for one patient (streaming, with the receptive
   field as context) and rescoring a whole stay.
2. SERVING MEMORY -- peak RSS of a process that loads the models and scores
   patients, WITHOUT the cached agent bank. The bank is a batch-evaluation
   convenience (2.46M rows held in RAM); a bedside deployment never holds it,
   and quoting its footprint as a requirement overstates the ask by ~100x.
3. EXPLANATION CARDS -- ollama with ``num_gpu: 0``, i.e. the language model
   forced onto the CPU. Cards are generated only when an alarm fires, so this
   is a per-alarm cost, not a per-hour one, and no prediction depends on it.

Agent weights are not needed for a compute benchmark (parameter count and
shape fix the FLOPs, values do not), so the agent nets are instantiated from
the deployed architectures; the fusion uses the real checkpoint when present.
No patient data is read: the LLM prompt is built from a synthetic context with
the real schema.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from agents.ecg_agent import EcgHead  # noqa: E402
from agents.labs_agent import LabsMLP  # noqa: E402
from agents.vitals_tcn import N_FEATURES as VITALS_FEATURES  # noqa: E402
from agents.vitals_tcn import VitalsTCNNet  # noqa: E402

CKPT = REPO / "models" / "orchestrator" / "all8_s42.pt"
OUT = REPO / "data" / "cache" / "bench_single_node.json"

# (name, n_features, hidden) for the plain MLP agents
MLP_AGENTS = [("labs", 24, 64), ("cadence", 32, 64), ("fluid", 13, 64),
              ("resp", 12, 64)]
ECG_FEATURES = 51
RECEPTIVE_FIELD = 29          # hours of context the TCN stack needs
STAY_HOURS = 34               # a representative stay length


IS_MACOS = sys.platform == "darwin"

# getrusage reports peak RSS in KILOBYTES on Linux but in BYTES on macOS.
# Dividing by the wrong constant silently reports memory off by ~1000x.
RSS_TO_MB = (1024 * 1024) if IS_MACOS else 1024


def cpu_model():
    if IS_MACOS:
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return platform.machine() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def timed(fn, reps, warmup=5):
    """Median and p95 wall time in ms over ``reps`` calls."""
    for _ in range(warmup):
        fn()
    laps = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        laps.append((time.perf_counter() - t0) * 1e3)
    a = np.asarray(laps)
    return float(np.median(a)), float(np.percentile(a, 95))


def build_agents():
    """The deployed agent architectures, in eval mode on CPU."""
    nets = {}
    nets["vitals_tcn"] = VitalsTCNNet(n_features=VITALS_FEATURES)
    nets["ecg_meas"] = EcgHead(embed_dim=ECG_FEATURES)
    for name, nf, hidden in MLP_AGENTS:
        nets[name] = LabsMLP(nf, hidden=hidden)
    for net in nets.values():
        net.eval()
    return nets


def load_fusion():
    """The two-head fusion net; real checkpoint when available."""
    from orchestrator.fusion import LatentFusionNet
    latent_dims = {"vitals_tcn": 32, "labs": 32, "ecg_meas": 32, "cadence": 32,
                   "fluid": 32, "resp": 32, "news2": 7, "gbm": 1}
    names, source = list(latent_dims), "architecture only"
    if CKPT.exists():
        blob = torch.load(CKPT, map_location="cpu", weights_only=False)
        names = list(blob["agent_names"])
        latent_dims = {n: int(d) for n, d in dict(blob["latent_dims"]).items()}
        net = LatentFusionNet(latent_dims, names, **dict(blob.get("hparams", {})))
        try:
            net.load_state_dict(blob["state_dict"])
            source = f"checkpoint {CKPT.relative_to(REPO)} (real weights)"
        except Exception as exc:
            source = f"architecture only ({type(exc).__name__})"
    else:
        net = LatentFusionNet(latent_dims, names)
    net.eval()
    return net, names, latent_dims, source


def agent_inputs(nets, hours):
    """One patient, ``hours`` of context, per agent."""
    x = {}
    for name, net in nets.items():
        if name == "vitals_tcn":
            x[name] = torch.randn(1, VITALS_FEATURES, hours)   # (B, F, T)
        elif name == "ecg_meas":        # (embedding, has_ecg, age)
            x[name] = (torch.randn(hours, ECG_FEATURES),
                       torch.ones(hours), torch.rand(hours))
        else:
            nf = next(f for n, f, _ in MLP_AGENTS if n == name)
            x[name] = torch.randn(hours, nf)
    return x


def fusion_inputs(latent_dims, names, hours):
    latents = {n: torch.randn(1, hours, d) for n, d in latent_dims.items()}
    risks = torch.rand(1, hours, len(names))
    return latents, risks


def measure_path(nets, fusion, names, latent_dims, hours, reps):
    """Per-component and end-to-end latency for one scoring call."""
    xa = agent_inputs(nets, hours)
    latents, risks = fusion_inputs(latent_dims, names, hours)
    per = {}
    with torch.inference_mode():
        for name, net in nets.items():
            arg = xa[name]
            call = ((lambda net=net, a=arg: net(*a)) if isinstance(arg, tuple)
                    else (lambda net=net, a=arg: net(a)))
            per[name] = timed(call, reps)
        per["fusion_two_head"] = timed(
            lambda: (fusion(latents, risks), fusion(latents, risks)), reps)

        def end_to_end():
            for name, net in nets.items():
                arg = xa[name]
                net(*arg) if isinstance(arg, tuple) else net(arg)
            fusion(latents, risks)          # treatment head
            fusion(latents, risks)          # mortality head
        per["END-TO-END"] = timed(end_to_end, reps)
    return per


def measure_throughput(nets, fusion, names, latent_dims, batch, hours, reps=20):
    """Batched throughput: many patients scored in one call (the backfill path)."""
    xa = {}
    for name in nets:
        if name == "vitals_tcn":
            xa[name] = torch.randn(batch, VITALS_FEATURES, hours)
        elif name == "ecg_meas":
            xa[name] = (torch.randn(batch * hours, ECG_FEATURES),
                        torch.ones(batch * hours), torch.rand(batch * hours))
        else:
            nf = next(f for n, f, _ in MLP_AGENTS if n == name)
            xa[name] = torch.randn(batch * hours, nf)
    latents = {n: torch.randn(batch, hours, d) for n, d in latent_dims.items()}
    risks = torch.rand(batch, hours, len(names))

    def run():
        for name, net in nets.items():
            arg = xa[name]
            net(*arg) if isinstance(arg, tuple) else net(arg)
        fusion(latents, risks)
        fusion(latents, risks)

    with torch.inference_mode():
        med, _ = timed(run, reps, warmup=3)
    return batch * hours / (med / 1e3)


def llm_cards(models, n_cards, timeout=600):
    """Explanation cards with ollama forced onto the CPU (num_gpu=0)."""
    from explain.rationales import SYSTEM_PROMPT, build_prompt
    agents = ["vitals_tcn", "news2", "labs", "ecg_meas", "gbm", "cadence",
              "fluid", "resp"]
    components = ["resp_rate", "spo2", "temperature", "sbp", "heart_rate",
                  "oxygen", "consciousness"]
    vitals = ["heart_rate", "resp_rate", "spo2", "sbp", "temperature"]
    rng = np.random.default_rng(42)
    context = {                       # synthetic: real schema, invented values
        "stay_id": 7, "hour": 23, "news2_total": 9.0,
        "subscores": {c: float(v) for c, v in zip(components, [3, 1, 0, 2, 1, 2, 0])},
        "agent_risks": {a: float(r) for a, r in zip(agents, rng.uniform(.05, .45, 8))},
        "attention": {a: float(w) for a, w in
                      zip(agents, rng.dirichlet(np.ones(8)))},
        "trends": {v: {"now": float(n), "before": float(b), "word": w}
                   for v, n, b, w in zip(vitals, [116, 38, 96, 107, 35.9],
                                         [107, 35, 96, 102, 36.0],
                                         ["rising", "rising", "stable",
                                          "rising", "falling"])},
    }
    prompt = build_prompt(context)
    out = {}
    for model in models:
        laps, toks = [], []
        for _ in range(n_cards):
            body = json.dumps({
                "model": model, "prompt": prompt, "system": SYSTEM_PROMPT,
                "stream": False, "think": False,
                "options": {"temperature": 0.0, "num_predict": 90,
                            "top_p": 1.0, "seed": 42, "num_gpu": 0},
            })
            t0 = time.perf_counter()
            r = subprocess.run(
                ["curl", "-s", "--max-time", str(timeout),
                 "http://localhost:11434/api/generate",
                 "-H", "Content-Type: application/json", "-d", body],
                capture_output=True, text=True, timeout=timeout + 30)
            dt = time.perf_counter() - t0
            if r.returncode != 0 or not r.stdout:
                continue
            try:
                resp = json.loads(r.stdout)
            except json.JSONDecodeError:
                continue
            laps.append(dt)
            if resp.get("eval_count") and resp.get("eval_duration"):
                toks.append(resp["eval_count"] / (resp["eval_duration"] / 1e9))
        if laps:
            out[model] = {"n": len(laps), "median_s": float(np.median(laps)),
                          "mean_s": float(np.mean(laps)),
                          "tokens_per_s": float(np.median(toks)) if toks else None,
                          "prompt_chars": len(prompt)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="1,2,4,8")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--llm-cards", type=int, default=3)
    ap.add_argument("--llm-models", default="medgemma:4b,gemma4:12b")
    ap.add_argument("--skip-llm", action="store_true")
    args = ap.parse_args()

    gpu_hidden = not torch.cuda.is_available()
    print("=" * 78)
    print("SENTINEL — SINGLE-NODE CPU-ONLY BENCHMARK")
    print("=" * 78)
    print(f"host CPU     : {cpu_model()}")
    print(f"cores        : {os.cpu_count()} logical")
    print(f"torch        : {torch.__version__}   CUDA visible: {not gpu_hidden}")
    print(f"python       : {platform.python_version()}  ({platform.machine()})")
    if IS_MACOS:
        print("note         : macOS — peak RSS read as bytes; torch+xgboost share "
              "an OpenMP runtime here, so keep OMP_NUM_THREADS=1 for any run that "
              "loads both (see scripts/results_table.py).")

    rss_base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / RSS_TO_MB
    nets = build_agents()
    fusion, names, latent_dims, source = load_fusion()
    rss_loaded = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / RSS_TO_MB

    agent_params = {n: sum(p.numel() for p in m.parameters()) for n, m in nets.items()}
    fusion_params = sum(p.numel() for p in fusion.parameters())
    total = sum(agent_params.values()) + 2 * fusion_params
    print(f"\nfusion weights: {source}")
    print("\n-- parameters ---------------------------------------------------")
    for n, p in sorted(agent_params.items(), key=lambda kv: -kv[1]):
        print(f"  {n:16s} {p:>9,}")
    print(f"  {'news2':16s} {0:>9,}   (deterministic lookup)")
    print(f"  {'fusion x2':16s} {2 * fusion_params:>9,}")
    print(f"  {'TOTAL (neural)':16s} {total:>9,}")

    results = {}
    thread_list = [int(t) for t in args.threads.split(",") if int(t) <= os.cpu_count()]
    for shape, hours in (("one patient-hour (streaming, 29 h context)", RECEPTIVE_FIELD),
                         (f"one stay rescored ({STAY_HOURS} h)", STAY_HOURS)):
        print(f"\n-- {shape} ---")
        print(f"  {'threads':>7}  {'median ms':>10}  {'p95 ms':>8}  {'patient-hours/s':>16}")
        for t in thread_list:
            torch.set_num_threads(t)
            per = measure_path(nets, fusion, names, latent_dims, hours, args.reps)
            med, p95 = per["END-TO-END"]
            rate = (hours if "stay" in shape else 1) / (med / 1e3)
            print(f"  {t:>7}  {med:>10.3f}  {p95:>8.3f}  {rate:>16,.0f}")
            results.setdefault(shape, {})[t] = {
                "end_to_end_median_ms": med, "end_to_end_p95_ms": p95,
                "components_median_ms": {k: v[0] for k, v in per.items()},
            }
        torch.set_num_threads(1)

    print(f"\n-- batched throughput (backfill / catch-up path) ----------------")
    torch.set_num_threads(1)
    tput = {}
    for batch in (64, 256):
        tput[batch] = measure_throughput(nets, fusion, names, latent_dims,
                                         batch, RECEPTIVE_FIELD)
        print(f"  {batch:>4d} patients x 29 h, 1 thread : "
              f"{tput[batch]:>12,.0f} patient-hours/s")

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / RSS_TO_MB
    print(f"\n-- serving memory ----------------------------------------------")
    print(f"  python + torch baseline                : {rss_base:,.0f} MB")
    print(f"  after loading every model              : {rss_loaded:,.0f} MB "
          f"(models add {rss_loaded - rss_base:,.1f} MB)")
    print(f"  peak RSS through the whole benchmark   : {rss_mb:,.0f} MB")
    print(f"  fusion checkpoint on disk              : "
          f"{CKPT.stat().st_size / 1024:,.0f} KB" if CKPT.exists() else "")

    one_hour_med = results[
        "one patient-hour (streaming, 29 h context)"][thread_list[0]]["end_to_end_median_ms"]
    for beds in (20, 100, 500):
        busy = beds * one_hour_med / 1e3 / 3600 * 100
        print(f"  {beds:>3d}-bed unit, hourly scoring, 1 thread : "
              f"{beds * one_hour_med:,.0f} ms of compute per hour "
              f"({busy:.4f}% duty cycle)")

    llm = {}
    if not args.skip_llm:
        models = [m for m in args.llm_models.split(",") if m]
        print(f"\n-- explanation cards, LLM forced to CPU (num_gpu=0) ------------")
        print("  (generated only when an alarm fires; no prediction depends on them)")
        llm = llm_cards(models, args.llm_cards)
        for model, r in llm.items():
            tps = f"{r['tokens_per_s']:.1f} tok/s" if r["tokens_per_s"] else "n/a"
            print(f"  {model:16s} {r['median_s']:>7.1f} s/card (median of "
                  f"{r['n']})   {tps}")
        if not llm:
            print("  no models responded — is ollama running?")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cpu": cpu_model(), "logical_cores": os.cpu_count(),
        "torch": torch.__version__, "cuda_visible": not gpu_hidden,
        "fusion_weights": source,
        "params": {**agent_params, "news2": 0, "fusion_per_head": fusion_params,
                   "total_neural": total},
        "latency": results, "throughput_ph_per_s": tput,
        "rss_mb": {"baseline": rss_base, "models_loaded": rss_loaded, "peak": rss_mb},
        "checkpoint_bytes": CKPT.stat().st_size if CKPT.exists() else None,
        "llm_cpu": llm,
    }, indent=1))
    print(f"\n[saved] {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
