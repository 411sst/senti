"""T40: rebuild the flagship on an ordinary laptop, CPU only, and compare with T36.

A cross-platform (Windows, Linux, macOS) replacement for ``scripts/h1_rebuild.sh``
with the same seven stages, the same seed (42) and one OpenMP thread. Run it
ONLY inside an isolated clone: ``data/cache`` must be a real directory that
starts with nothing but the seven copied input files and an empty
``embeddings/``, because several stages overwrite cached agent outputs.

    python scripts/t40_rebuild.py check     # isolation, input checksums, machine facts
    python scripts/t40_rebuild.py grid      # T36's seven stages, FAIR+TRAJ grid-tuned
    python scripts/t40_rebuild.py reuse     # FAIR+TRAJ with the committed winner, then
                                            # the GBM agent and both heads again
    python scripts/t40_rebuild.py bundle    # t40/result.json: aggregates only

Developer only, on the T36 machine (writes docs/t40_reference.json):

    python scripts/t40_rebuild.py reference --clone ~/code/sentinel-h1

Each stage runs as a child process with ``OMP_NUM_THREADS=1``, the GPU hidden
and UTF-8 output. Per stage the runner records wall time, peak memory (exact
from ``wait4`` on Linux and macOS, the peak working set on Windows), the
lowest free RAM seen, and swap activity (bytes swapped out on Linux, pagefile
growth on Windows). It then hashes every score array the flagship is built
from, so the comparison with T36 needs no score file to leave either machine.

Only ``t40/result.json`` is meant to leave the laptop. It holds aggregates and
sha256 digests only; the runner refuses to write it if it contains anything
id-shaped, the user name or the host name.
"""

import argparse
import getpass
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

CACHE = REPO / "data" / "cache"
EMB = CACHE / "embeddings"
HEADS = REPO / "h1" / "two_head_s42.npz"
FAIR = CACHE / "fair_gbm_scores.npz"
KEEP = CACHE / "t40_grid_keep"
OUT_DIR = REPO / "t40"
LOGS = REPO / "logs"
REFERENCE = REPO / "docs" / "t40_reference.json"

INPUT_FILES = ("ecg_index.parquet", "ecg_machine_measurements.parquet",
               "mimic_cadence_hourly.parquet", "mimic_cohort_hourly.parquet",
               "mimic_fluids_hourly.parquet", "mimic_labs_hourly.parquet",
               "mimic_resp_hourly.parquet")
AGENTS = ("vitals_tcn", "news2", "labs", "ecg_meas", "gbm", "cadence", "fluid", "resp")
SPLITS = ("train", "val", "test")
HEAD_ARRAYS = ("val_s42", "test_s42", "dhead_test_s42")
FAIR_ARRAYS = ("rawval_FAIR+TRAJ", "raw_FAIR+TRAJ")

STAGES = {
    "ecg_features": ["scripts/build_ecg_meas_features.py"],
    "core3_agents": ["scripts/results_table.py"],
    "ecg_meas_agent": ["scripts/train_ecg_meas_agent.py"],
    "fair_gbm": ["src/baselines_fair.py", "--variants", "FAIR+TRAJ",
                 "--device", "cpu", "--reps", "10"],
    "fair_reuse": ["scripts/t40_fair_reuse.py"],
    "gbm_agent": ["scripts/h1_rebuild.py", "gbm-agent"],
    "tab_agents": ["scripts/train_new_agents.py", "--agent", "cadence,fluid,resp"],
    "heads": ["scripts/h1_rebuild.py", "heads"],
}
GRID = ("ecg_features", "core3_agents", "ecg_meas_agent", "fair_gbm",
        "gbm_agent", "tab_agents", "heads")            # T36's order
REUSE = ("fair_reuse", "gbm_agent", "heads")
SHARED = ("ecg_features", "core3_agents", "ecg_meas_agent", "tab_agents")

SAMPLE_SECONDS = 0.5
WINDOWS_PAGEFILE_TOLERANCE = 256 * 2 ** 20      # bytes of pagefile growth
MIN_FREE_RAM_SHARE = 0.05


# --- hashing ---------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(a):
    """sha256 of an array's dtype, shape and bytes (C order)."""
    import numpy as np
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def score_digests(cache_dir, heads_path):
    """Digests of every score array the flagship is built from.

    Agent risk and latent per split, the FAIR+TRAJ raw scores, and the two
    heads. Key arrays (stay ids, hours) are never hashed or read out.
    """
    import numpy as np
    cache_dir = Path(cache_dir)
    out = {}
    for agent in AGENTS:
        for split in SPLITS:
            path = cache_dir / "embeddings" / f"{agent}_{split}.npz"
            if path.exists():
                z = np.load(path, allow_pickle=False)
                for k in ("risk", "latent"):
                    out[f"agent.{agent}.{split}.{k}"] = array_digest(z[k])
    fair = cache_dir / "fair_gbm_scores.npz"
    if fair.exists():
        z = np.load(fair, allow_pickle=True)
        for k in FAIR_ARRAYS:
            if k in z.files:
                out[f"fair.{k}"] = array_digest(z[k])
    if Path(heads_path).exists():
        z = np.load(heads_path, allow_pickle=False)
        for k in HEAD_ARRAYS:
            out[f"heads.{k}"] = array_digest(z[k])
    return out


def compare_digests(got, ref):
    """Which arrays match the reference bit for bit, by group."""
    names = sorted(set(got) | set(ref))
    same = [n for n in names if n in got and n in ref and got[n] == ref[n]]
    diff = [n for n in names if n in got and n in ref and got[n] != ref[n]]
    missing = [n for n in names if n not in got]
    extra = [n for n in names if n not in ref]
    flagship = [n for n in ref if not n.startswith("fair.")]
    return {"n_reference": len(ref), "identical": same, "different": diff,
            "missing": missing, "not_in_reference": extra,
            "flagship_all_identical": all(n in same for n in flagship)}


# --- isolation and inputs ------------------------------------------------------

def check_isolated(repo, fresh):
    """Refuse to run anywhere but an isolated clone.

    ``data/cache`` must be a real directory with no symlinks. With ``fresh``
    it must hold only the seven inputs and an empty ``embeddings/``, so the
    stages cannot overwrite anything a working repository depends on.
    """
    cache = Path(repo) / "data" / "cache"
    if (Path(repo) / "data").is_symlink() or cache.is_symlink():
        raise SystemExit("data/cache is a symlink: T40 runs only in an isolated clone")
    if not cache.is_dir():
        raise SystemExit("data/cache does not exist: copy the seven inputs in first")
    if any(p.is_symlink() for p in cache.iterdir()):
        raise SystemExit("data/cache holds symlinks: copy the inputs instead")
    if fresh:
        emb = cache / "embeddings"
        if emb.exists() and any(emb.iterdir()):
            raise SystemExit("data/cache/embeddings is not empty: start from a clean clone")
        extra = sorted(p.name for p in cache.iterdir()
                       if p.name not in INPUT_FILES and p.name != "embeddings")
        if extra:
            raise SystemExit(f"data/cache holds files other than the inputs: {extra}; "
                             "start from a clean clone")
        missing = [n for n in INPUT_FILES if not (cache / n).exists()]
        if missing:
            raise SystemExit(f"inputs missing from data/cache: {missing}")


def verify_inputs(cache_dir, expected):
    bad = {n: "missing" for n in INPUT_FILES if not (Path(cache_dir) / n).exists()}
    for n in INPUT_FILES:
        if n not in bad and sha256_file(Path(cache_dir) / n) != expected[n]:
            bad[n] = "checksum differs"
    if bad:
        raise SystemExit(f"inputs do not match T36's: {bad}")
    return {"inputs_match_t36": True, "n_inputs": len(INPUT_FILES)}


def dir_bytes(path, exclude=()):
    total = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if Path(root, d) not in {Path(e) for e in exclude}]
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


# --- machine facts -------------------------------------------------------------

def cpu_name():
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except Exception:
            return platform.processor() or "unknown"
    if sys.platform == "darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return platform.machine() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def os_name():
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/etc/os-release").read_text().splitlines():
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip('"') + f" (kernel {platform.release()})"
        except OSError:
            pass
    if sys.platform == "win32":
        return f"Windows {platform.release()} (build {platform.version()})"
    if sys.platform == "darwin":
        return f"macOS {platform.mac_ver()[0]}"
    return f"{platform.system()} {platform.release()}"


def _version(module):
    try:
        return __import__(module).__version__
    except Exception:
        return None


def ollama_version():
    try:
        r = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=20)
        return (r.stdout or r.stderr).strip().splitlines()[-1][:80] or None
    except Exception:
        return None


def git_state():
    """The commit the clone is on, and whether tracked files are unmodified."""
    def git(*args):
        r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    try:
        status = git("status", "--porcelain", "--untracked-files=no")
        return {"git_commit": git("rev-parse", "HEAD"),
                "git_tracked_files_clean": status == "" if status is not None else None}
    except Exception:
        return {"git_commit": None, "git_tracked_files_clean": None}


def machine_facts():
    """CPU, cores, RAM, OS and library versions; no user, host or path names."""
    import psutil
    import torch
    return {
        **git_state(),
        "cpu": cpu_name(), "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1),
        "os": os_name(), "python": platform.python_version(),
        "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "cuda_visible": bool(torch.cuda.is_available()),
        "numpy": _version("numpy"), "pandas": _version("pandas"),
        "sklearn": _version("sklearn"), "xgboost": _version("xgboost"),
        "lightgbm": _version("lightgbm"), "scipy": _version("scipy"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "ollama": ollama_version(),
    }


def private_strings():
    """Names that must never appear in an output file."""
    names = set()
    for get in (platform.node, getpass.getuser, lambda: Path.home().name):
        try:
            v = (get() or "").strip()
        except Exception:
            v = ""
        if len(v) >= 3:
            names.add(v.lower())
    names.add(str(REPO).lower())
    return names


def _without_digests(tree):
    """The payload minus sha256 digest maps (pure hex, no names), so a short
    user name made of hex letters cannot match inside a digest by chance."""
    if isinstance(tree, dict):
        return {k: _without_digests(v) for k, v in tree.items()
                if k not in ("digests", "t36_digests")}
    if isinstance(tree, list):
        return [_without_digests(v) for v in tree]
    return tree


def assert_shareable(payload, private=None):
    """Aggregates only, and no user, host or path names."""
    from demo_bundle import assert_aggregate_only
    assert_aggregate_only(payload)
    text = json.dumps(_without_digests(payload)).lower()
    for name in (private_strings() if private is None else private):
        if name and name in text:
            raise SystemExit("output contains a user, host or path name; not writing it")
    return True


# --- stage runner --------------------------------------------------------------

def stage_env():
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
                "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"})
    return env


def run_stage(argv, log_path, env=None, cwd=None, interval=SAMPLE_SECONDS):
    """Run ``python argv...``; return wall time, peak memory and swap activity."""
    import psutil
    swap0 = psutil.swap_memory()
    vm0 = psutil.virtual_memory()
    stats = {"peak_tree_rss_bytes": 0, "peak_process_bytes": 0,
             "min_available_bytes": vm0.available, "max_swap_used_bytes": swap0.used}
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, *argv], cwd=cwd or REPO,
                                env=env or stage_env(), stdout=log,
                                stderr=subprocess.STDOUT)
        root = psutil.Process(proc.pid)
        exact_peak = None
        while True:
            if hasattr(os, "wait4"):
                pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
                if pid == proc.pid:
                    proc.returncode = os.waitstatus_to_exitcode(status)
                    exact_peak = ru.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
                    break
            elif proc.poll() is not None:
                break
            _sample(root, stats)
            time.sleep(interval)
    wall = time.perf_counter() - t0
    swap1 = psutil.swap_memory()
    if exact_peak is not None:
        stats["peak_process_bytes"] = max(stats["peak_process_bytes"], exact_peak)
        stats["peak_source"] = "wait4 ru_maxrss"
    else:
        stats["peak_source"] = "sampled peak working set"
    stats.update({
        "exit_code": proc.returncode, "wall_seconds": round(wall, 2),
        "swap_out_bytes": (swap1.sout - swap0.sout) if sys.platform != "win32" else None,
        "swap_used_growth_bytes": max(0, stats["max_swap_used_bytes"] - swap0.used),
    })
    return stats


def _sample(root, stats):
    import psutil
    try:
        procs = [root] + root.children(recursive=True)
    except psutil.Error:
        return
    tree = 0
    for p in procs:
        try:
            mi = p.memory_info()
        except psutil.Error:
            continue
        tree += mi.rss
        stats["peak_process_bytes"] = max(stats["peak_process_bytes"], mi.rss,
                                          getattr(mi, "peak_wset", 0))
    stats["peak_tree_rss_bytes"] = max(stats["peak_tree_rss_bytes"], tree)
    vm = psutil.virtual_memory()
    stats["min_available_bytes"] = min(stats["min_available_bytes"], vm.available)
    stats["max_swap_used_bytes"] = max(stats["max_swap_used_bytes"], psutil.swap_memory().used)


def no_swap(stage_stats, total_ram_bytes, windows=None):
    """Did every stage fit in RAM without swapping?

    POSIX: nothing was swapped out while a stage ran. Windows (no swap-out
    counter): the pagefile grew by at most 256 MB. Either way the free RAM
    never fell below 5% of the total.
    """
    windows = (sys.platform == "win32") if windows is None else windows
    for s in stage_stats.values():
        if s["min_available_bytes"] < MIN_FREE_RAM_SHARE * total_ram_bytes:
            return False
        if windows:
            if s["swap_used_growth_bytes"] > WINDOWS_PAGEFILE_TOLERANCE:
                return False
        elif s["swap_out_bytes"]:
            return False
    return True


def run_stages(names, mode):
    stages = {}
    for name in names:
        print(f"[{datetime.now():%H:%M:%S}] >>> {mode}/{name}", flush=True)
        st = run_stage(STAGES[name], LOGS / f"t40_{mode}_{name}.log")
        stages[name] = st
        print(f"[{datetime.now():%H:%M:%S}] {'ok ' if st['exit_code'] == 0 else 'FAILED'} "
              f"{name}: {st['wall_seconds'] / 60:.1f} min, peak "
              f"{st['peak_process_bytes'] / 2 ** 30:.2f} GiB", flush=True)
        if st["exit_code"] != 0:
            raise SystemExit(f"stage {name} failed (exit {st['exit_code']}); "
                             f"see logs/t40_{mode}_{name}.log")
    return stages


# --- metrics -------------------------------------------------------------------

def head_metrics(heads_path):
    """Test AUPRC and AUROC of the two-head score, and the D head on death."""
    import numpy as np

    from baselines_mimic import load_splits
    from fusion_common import FusionContext
    from metrics import auprc, auroc
    ctx = FusionContext(load_splits())
    z = np.load(heads_path, allow_pickle=False)
    if not (z["stay_id_test"] == ctx.test_ordered["stay_id"].to_numpy()).all():
        raise SystemExit("heads file is not aligned with the test rows")
    y_death = ctx.test_ordered["label_24h_death"].to_numpy(dtype=float)
    s = z["test_s42"].astype(float)
    return {"test_auprc": float(auprc(ctx.y_test, s)),
            "test_auroc": float(auroc(ctx.y_test, s)),
            "dhead_death_auprc": float(auprc(y_death, z["dhead_test_s42"].astype(float)))}


# --- commands ------------------------------------------------------------------

def _load_reference():
    if not REFERENCE.exists():
        raise SystemExit(f"{REFERENCE.relative_to(REPO)} is missing")
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def _write(name, payload):
    payload = json.loads(json.dumps(payload, default=float))
    assert_shareable(payload)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / name).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"[saved] t40/{name}")
    return payload


def cmd_check():
    os.environ.update({"OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""})
    ref = _load_reference()
    check_isolated(REPO, fresh=True)
    facts = machine_facts()
    facts.update(verify_inputs(CACHE, ref["inputs_sha256"]))
    facts["checked_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write("machine.json", facts)
    print(json.dumps(facts, indent=1))


def _finish(mode, stages, ref, extra=None):
    import psutil
    digests = score_digests(CACHE, HEADS)
    cmp = compare_digests(digests, ref["t36_digests"])
    total = psutil.virtual_memory().total
    payload = {
        "mode": mode, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stages": stages,
        "stage_wall_seconds_total": round(sum(s["wall_seconds"] for s in stages.values()), 1),
        "peak_process_gib": round(max(s["peak_process_bytes"] for s in stages.values()) / 2 ** 30, 2),
        "fits_in_ram_without_swapping": no_swap(stages, total),
        "metrics": head_metrics(HEADS), "digests": digests, "vs_t36": cmp,
        **(extra or {}),
    }
    print(f"[{mode}] flagship arrays identical to T36: {cmp['flagship_all_identical']} "
          f"({len(cmp['identical'])} identical, {len(cmp['different'])} different, "
          f"{len(cmp['missing'])} missing); test AUPRC {payload['metrics']['test_auprc']:.4f}")
    return _write(f"{mode}.json", payload)


def cmd_grid():
    ref = _load_reference()
    check_isolated(REPO, fresh=True)
    verify_inputs(CACHE, ref["inputs_sha256"])
    disk_inputs = dir_bytes(CACHE)
    stages = run_stages(GRID, "grid")
    disk_after = dir_bytes(CACHE)
    KEEP.mkdir(exist_ok=True)              # what reuse mode overwrites, kept locally
    for p in [FAIR, HEADS] + [EMB / f"gbm_{s}.npz" for s in SPLITS]:
        shutil.copy2(p, KEEP / p.name)
    _finish("grid", stages, ref, {"disk_inputs_bytes": disk_inputs,
                                  "disk_after_bytes": disk_after})


def cmd_reuse():
    ref = _load_reference()
    if not (OUT_DIR / "grid.json").exists():
        raise SystemExit("run `grid` first: reuse mode re-runs only the stages after it")
    check_isolated(REPO, fresh=False)
    grid = json.loads((OUT_DIR / "grid.json").read_text(encoding="utf-8"))
    stages = run_stages(REUSE, "reuse")
    shared = {n: grid["stages"][n]["wall_seconds"] for n in SHARED}
    total = sum(shared.values()) + sum(s["wall_seconds"] for s in stages.values())
    _finish("reuse", stages, ref, {
        "rebuild_wall_seconds_with_shared_stages": round(total, 1),
        "shared_stage_seconds_from_grid": shared,
        "disk_after_bytes": dir_bytes(CACHE, exclude=(KEEP,)),
        "committed_setting": {"max_depth": 4, "scale_pos_weight": 1.0,
                              "source": "results_fusion.md l.189"}})


def cmd_bundle():
    ref = _load_reference()
    parts = {}
    for name in ("machine", "grid", "reuse"):
        p = OUT_DIR / f"{name}.json"
        parts[name] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    bench = CACHE / "bench_single_node.json"
    parts["bench"] = json.loads(bench.read_text(encoding="utf-8")) if bench.exists() else None
    parts["disk_env_bytes"] = dir_bytes(sys.prefix)
    parts["reference"] = {"t36_metrics": ref["t36_metrics"],
                          "published_ten_seed_auprc": ref["published_ten_seed_auprc"]}
    g = parts["grid"]
    if g:
        parts["verdict_t40"] = {
            "pass": g["vs_t36"]["flagship_all_identical"],
            "rule": "grid mode: every agent and head score array identical to T36's CPU rebuild",
            "n_different": len(g["vs_t36"]["different"]),
        }
    _write("result.json", parts)
    print("send only t40/result.json")


def cmd_reference(clone):
    """Digests and metrics of T36's rebuild, read from its isolated clone."""
    clone = Path(clone).expanduser()
    inputs = {}
    for line in (clone / "h1" / "inputs.sha256").read_text().splitlines():
        digest, name = line.split()
        inputs[name] = digest
    if set(inputs) != set(INPUT_FILES):
        raise SystemExit("T36 input list differs from INPUT_FILES")
    stages = {}
    for line in (clone / "h1" / "stages.tsv").read_text().splitlines():
        name, wall, peak_kb = line.split("\t")
        stages[name] = {"wall_seconds": float(wall), "peak_process_bytes": int(peak_kb) * 1024}
    t30 = json.loads((CACHE / "t30_allfeat_gbm.json").read_text())
    pub = [v for k, v in t30["ranking"]["auprc"]["points"].items() if k.startswith("flagship_s")]
    payload = {
        "source": "T36 CPU rebuild (isolated clone on the Core Ultra 9 275HX)",
        "inputs_sha256": inputs,
        "t36_digests": score_digests(clone / "data" / "cache", clone / "h1" / "two_head_s42.npz"),
        "t36_metrics": head_metrics(clone / "h1" / "two_head_s42.npz"),
        "t36_stages": stages,
        "t36_machine": json.loads((clone / "h1" / "machine.json").read_text()),
        "published_ten_seed_auprc": {"mean": sum(pub) / len(pub), "min": min(pub),
                                     "max": max(pub), "n": len(pub)},
    }
    payload["t36_machine"].pop("kernel", None)
    payload = json.loads(json.dumps(payload, default=float))
    assert_shareable(payload)
    REFERENCE.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"[saved] {REFERENCE.relative_to(REPO)}: {len(payload['t36_digests'])} digests; "
          f"T36 test AUPRC {payload['t36_metrics']['test_auprc']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("check", "grid", "reuse", "bundle", "reference"))
    ap.add_argument("--clone", help="T36's isolated clone (reference only)")
    args = ap.parse_args()
    os.chdir(REPO)
    os.environ.update({"OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONUTF8": "1"})
    if args.what == "reference":
        if not args.clone:
            raise SystemExit("--clone is required")
        cmd_reference(args.clone)
    else:
        {"check": cmd_check, "grid": cmd_grid, "reuse": cmd_reuse,
         "bundle": cmd_bundle}[args.what]()


if __name__ == "__main__":
    main()
