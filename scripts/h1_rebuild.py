"""H1 (T36): rebuild the flagship on this laptop, CPU only, and compare.

Helpers for ``scripts/h1_rebuild.sh``. They are run ONLY inside an isolated
clone: ``data/cache`` must be a real directory holding copied inputs, because
the agent stages overwrite cached agent outputs (``scripts/results_table.py``
is the reason this exists; never run it in a working repository).

    python scripts/h1_rebuild.py gbm-agent     # the GBM pseudo-agent cache (CPU)
    python scripts/h1_rebuild.py heads         # both heads at seed 42 -> h1/two_head_s42.npz
    python scripts/h1_rebuild.py machine       # CPU / RAM / OS / torch (no user or host names)
    python scripts/h1_rebuild.py compare --shipped /path/to/published/data/cache

``compare`` reads the published caches read-only and writes h1/compare.json
with aggregates only.
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

OUT_DIR = REPO / "h1"
HEADS = OUT_DIR / "two_head_s42.npz"
SPLITS = ("train", "val", "test")


def require_isolated():
    cache = REPO / "data" / "cache"
    if cache.is_symlink() or (REPO / "data").is_symlink():
        raise SystemExit("data/cache is a symlink: H1 runs only in an isolated clone")
    if any(p.is_symlink() for p in cache.iterdir()):
        raise SystemExit("data/cache holds symlinks: copy the inputs instead")


def gbm_agent():
    require_isolated()
    from baselines_mimic import load_splits
    from orch_gbm_agent import build_gbm_agent_cache
    build_gbm_agent_cache(load_splits(), device="cpu")


def heads():
    """``two_head.py``'s per-seed loop at seed 42, as in its gated replicas."""
    require_isolated()
    import torch
    torch.set_num_threads(1)
    from baselines_mimic import load_splits
    from decompose import treatment_only_label
    from fusion_common import FusionContext
    from fusion_ladder import _strictify
    from orchestrator.fusion import AgentBank
    from sklearn.isotonic import IsotonicRegression
    from two_head import ALL8, fit_head

    splits = load_splits()
    ctx = FusionContext(splits)
    train_o = splits["train"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    val_o = ctx.val_ordered
    y_death = {"train": train_o["label_24h_death"].to_numpy(dtype=float),
               "val": val_o["label_24h_death"].to_numpy(dtype=float)}
    y_treat = {"train": treatment_only_label(train_o), "val": treatment_only_label(val_o)}
    bank = AgentBank.load(agent_names=ALL8)
    out = {}
    for name, y in (("T", y_treat), ("D", y_death)):
        head = fit_head(bank, splits, y["train"], y["val"], train_o, val_o, 42, 8)
        raw_val = head.raw_risk(splits["val"])
        raw_test = head.raw_risk(splits["test"])
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            raw_val[ctx.fit_mask], y["val"][ctx.fit_mask])
        out[name] = {"val": _strictify(iso.transform(raw_val), raw_val),
                     "test": _strictify(iso.transform(raw_test), raw_test)}
    OUT_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        HEADS,
        val_s42=1.0 - (1.0 - out["T"]["val"]) * (1.0 - out["D"]["val"]),
        test_s42=1.0 - (1.0 - out["T"]["test"]) * (1.0 - out["D"]["test"]),
        dhead_test_s42=out["D"]["test"],
        stay_id_test=ctx.test_ordered["stay_id"].to_numpy())


def machine():
    """Hardware and software facts, with no user, host or path names."""
    import torch
    cpu = ""
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    mem_kb = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                  if line.startswith("MemTotal"))
    os_name = ""
    for line in Path("/etc/os-release").read_text().splitlines():
        if line.startswith("PRETTY_NAME="):
            os_name = line.split("=", 1)[1].strip('"')
    info = {"cpu": cpu, "logical_cpus": os.cpu_count(),
            "ram_gb": round(mem_kb / 1024 ** 2, 1), "os": os_name,
            "kernel": platform.release(), "python": platform.python_version(),
            "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
            "cuda_visible": torch.cuda.is_available(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "machine.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def _diff(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return {"same_shape": False}
    return {"identical": bool(np.array_equal(a, b)), "max_abs_diff": float(np.abs(a - b).max()),
            "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 1 else float("nan")}


def compare(shipped):
    from agents.base import AgentOutput
    from baselines_mimic import load_splits
    from demo_bundle import assert_aggregate_only
    from fusion_common import FusionContext
    from metrics import auprc, auroc
    from two_head import ALL8

    shipped = Path(shipped)
    mine_emb, ship_emb = REPO / "data" / "cache" / "embeddings", shipped / "embeddings"
    agents = {}
    for a in ALL8:
        agents[a] = {}
        for s in SPLITS:
            m, p = AgentOutput.load(mine_emb / f"{a}_{s}.npz"), AgentOutput.load(ship_emb / f"{a}_{s}.npz")
            if not m.keys.equals(p.keys):
                agents[a][s] = {"keys_match": False}
                continue
            agents[a][s] = {"keys_match": True, "risk": _diff(m.risk, p.risk),
                            "latent": _diff(m.latent, p.latent)}
        t = agents[a]["test"]
        print(f"[agent] {a:<10} test risk identical {t['risk'].get('identical')} "
              f"max|d| {t['risk'].get('max_abs_diff', float('nan')):.3g} corr {t['risk'].get('corr', float('nan')):.5f}")
    ctx = FusionContext(load_splits())
    mine, ship = np.load(HEADS), np.load(shipped / "two_head.npz")
    y = ctx.y_test
    heads = {s: _diff(mine[f"{s}_s42"], ship[f"{s}_s42"]) for s in ("val", "test")}
    result = {"agents": agents, "heads": heads,
              "auprc_rebuilt": float(auprc(y, mine["test_s42"])),
              "auprc_shipped": float(auprc(y, ship["test_s42"])),
              "auroc_rebuilt": float(auroc(y, mine["test_s42"])),
              "auroc_shipped": float(auroc(y, ship["test_s42"]))}
    result["pass"] = bool(heads["val"].get("identical") and heads["test"].get("identical"))
    assert_aggregate_only(result)
    (OUT_DIR / "compare.json").write_text(json.dumps(result, indent=1))
    print(f"[heads] test identical {heads['test'].get('identical')} max|d| "
          f"{heads['test'].get('max_abs_diff', float('nan')):.3g}; AUPRC rebuilt "
          f"{result['auprc_rebuilt']:.4f} vs shipped {result['auprc_shipped']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("gbm-agent", "heads", "machine", "compare"))
    ap.add_argument("--shipped")
    args = ap.parse_args()
    t0 = time.time()
    if args.what == "gbm-agent":
        gbm_agent()
    elif args.what == "heads":
        heads()
    elif args.what == "machine":
        machine()
    else:
        if not args.shipped:
            ap.error("compare needs --shipped")
        compare(args.shipped)
    print(f"[done] {args.what} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
