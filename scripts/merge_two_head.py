"""Merge per-batch two-head caches into data/cache/two_head.npz.

Seed batches run as separate single-thread processes (``two_head.py --out``)
so ten seeds finish in the wall-clock of four. This merges them, checks the
row keys agree, and -- when a reference cache is given -- verifies that every
seed present in both reproduces bit-for-bit (the regression pin: same code,
same seed, same scores) before anything is overwritten.

    python scripts/merge_two_head.py --parts data/cache/two_head_part_*.npz \
        --reference data/cache/backup_pre10seed_20260829/two_head.npz \
        [--out data/cache/two_head.npz] [--seed-order 42,1,2,3,4,5,6,7,8,9]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ARRAYS = ("val_s{s}", "test_s{s}", "dhead_test_s{s}")


def load(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    return z, meta


def merge_parts(paths, out, reference=None, seed_order=None, allow_drift=False):
    """Merge the caches at ``paths`` into ``out``; returns the seed order.

    Raises ``ValueError`` on disagreeing row keys, a duplicated seed, a
    ``seed_order`` that does not match, or (unless ``allow_drift``) a seed that
    fails to reproduce the ``reference`` cache bit-for-bit.
    """
    parts = [load(p) for p in paths]
    z0, meta0 = parts[0]
    payload = {"stay_id_test": z0["stay_id_test"], "hour_test": z0["hour_test"]}
    results, seen = {}, []
    for path, (z, meta) in zip(paths, parts):
        for key in ("stay_id_test", "hour_test"):
            if not np.array_equal(z[key], payload[key]):
                raise ValueError(f"{path}: {key} differs from {paths[0]}")
        for s in meta["seeds"]:
            if s in seen:
                raise ValueError(f"seed {s} appears twice (again in {path})")
            seen.append(s)
            for tmpl in ARRAYS:
                payload[tmpl.format(s=s)] = z[tmpl.format(s=s)]
            results[str(s)] = meta["results"][str(s)]
    order = list(seed_order) if seed_order else seen
    if sorted(order) != sorted(seen):
        raise ValueError(f"seed order {order} does not match the seeds found {seen}")

    if reference:
        zr, mr = load(reference)
        drift = []
        for s in order:
            if s not in mr["seeds"]:
                continue
            same = all(np.array_equal(zr[t.format(s=s)], payload[t.format(s=s)])
                       for t in ARRAYS)
            same_meta = _comparable(mr["results"][str(s)]) == \
                _comparable(results[str(s)])
            print(f"[pin] seed {s}: arrays {'identical' if same else 'DIFFER'}, "
                  f"results {'identical' if same_meta else 'DIFFER'}")
            if not (same and same_meta):
                drift.append(s)
        if drift and not allow_drift:
            raise ValueError(f"regression pin failed for seeds {drift}; not writing")

    payload["meta_json"] = np.array(json.dumps({
        "seeds": order, "reps": meta0["reps"], "agents": meta0["agents"],
        "results": {str(s): results[str(s)] for s in order},
        "merged_from": [str(p) for p in paths],
    }))
    np.savez_compressed(out, **payload)
    print(f"[cache] {len(order)} seeds {order} -> {out}")
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--out", default=str(REPO / "data/cache/two_head.npz"))
    ap.add_argument("--reference", default=None,
                    help="cache whose overlapping seeds must reproduce exactly")
    ap.add_argument("--seed-order", default=None,
                    help="comma list; default = order of appearance")
    ap.add_argument("--allow-drift", action="store_true",
                    help="write even if a shared seed differs from the reference")
    args = ap.parse_args()
    order = ([int(s) for s in args.seed_order.split(",")]
             if args.seed_order else None)
    try:
        merge_parts(args.parts, args.out, reference=args.reference,
                    seed_order=order, allow_drift=args.allow_drift)
    except ValueError as exc:
        sys.exit(str(exc))


def _comparable(r):
    """Drop the one field the guard may legitimately change (missing parent)."""
    r = json.loads(json.dumps(r))
    r.pop("delta_vs_all8_ci", None)
    return r


if __name__ == "__main__":
    main()
