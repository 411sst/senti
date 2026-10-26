"""The two-head batch merge: keys must agree, seeds must not repeat, and any
seed shared with the reference cache must reproduce bit-for-bit."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "merge_two_head", Path(__file__).resolve().parents[1] / "scripts" / "merge_two_head.py")
merge_two_head = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge_two_head)


def _part(path, seeds, rows=5, shift=0.0, hour_shift=0):
    payload = {"stay_id_test": np.arange(rows) + 39_999_990,
               "hour_test": np.arange(rows) + hour_shift}
    results = {}
    for s in seeds:
        rng = np.random.default_rng(s)
        payload[f"val_s{s}"] = rng.random(rows) + shift
        payload[f"test_s{s}"] = rng.random(rows) + shift
        payload[f"dhead_test_s{s}"] = rng.random(rows)
        results[str(s)] = {"auprc": 0.3 + s / 1000, "delta_vs_all8_ci": [-0.01, 0.01]}
    payload["meta_json"] = np.array(json.dumps(
        {"seeds": list(seeds), "reps": 2, "agents": ["a"], "results": results}))
    np.savez_compressed(path, **payload)
    return path


def test_merge_keeps_order_checks_pin_and_tolerates_missing_parent(tmp_path):
    a = _part(tmp_path / "a.npz", [42, 1])
    b = _part(tmp_path / "b.npz", [5])
    ref = _part(tmp_path / "ref.npz", [42, 1, 2])
    # the guard may null the parent column on a re-run; that must not trip the pin
    z = np.load(ref, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    meta["results"]["42"]["delta_vs_all8_ci"] = None
    ref2 = tmp_path / "ref2.npz"
    np.savez_compressed(ref2, **{k: z[k] for k in z.files if k != "meta_json"},
                        meta_json=np.array(json.dumps(meta)))
    out = tmp_path / "out.npz"
    order = merge_two_head.merge_parts([str(a), str(b)], str(out), reference=str(ref2),
                                       seed_order=[42, 5, 1])
    assert order == [42, 5, 1]
    m = json.loads(str(np.load(out, allow_pickle=True)["meta_json"]))
    assert m["seeds"] == [42, 5, 1] and list(m["results"]) == ["42", "5", "1"]


def test_merge_rejects_drift_duplicates_and_key_mismatch(tmp_path):
    a = _part(tmp_path / "a.npz", [42])
    drifted = _part(tmp_path / "ref.npz", [42], shift=1e-9)
    with pytest.raises(ValueError, match="regression pin"):
        merge_two_head.merge_parts([str(a)], str(tmp_path / "o.npz"), reference=str(drifted))
    merge_two_head.merge_parts([str(a)], str(tmp_path / "o.npz"), reference=str(drifted),
                               allow_drift=True)
    dup = _part(tmp_path / "dup.npz", [42])
    with pytest.raises(ValueError, match="twice"):
        merge_two_head.merge_parts([str(a), str(dup)], str(tmp_path / "o.npz"))
    other_rows = _part(tmp_path / "rows.npz", [7], hour_shift=1)
    with pytest.raises(ValueError, match="hour_test"):
        merge_two_head.merge_parts([str(a), str(other_rows)], str(tmp_path / "o.npz"))
    with pytest.raises(ValueError, match="does not match"):
        merge_two_head.merge_parts([str(a)], str(tmp_path / "o.npz"), seed_order=[42, 1])
