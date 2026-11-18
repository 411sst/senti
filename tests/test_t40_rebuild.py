"""Tests for the T40 runner (scripts/t40_rebuild.py), its reuse-mode FAIR+TRAJ
stage (scripts/t40_fair_reuse.py) and the Windows-safe benchmark paths.

Synthetic data only; no cohort file is read.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import t40_rebuild as t40  # noqa: E402


# --- hashing -------------------------------------------------------------------

def test_array_digest_depends_on_values_dtype_and_shape_only():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert t40.array_digest(a) == t40.array_digest(a.copy())
    assert t40.array_digest(np.asfortranarray(a)) == t40.array_digest(a)   # layout-free
    assert t40.array_digest(a) != t40.array_digest(a.astype(np.float64))
    assert t40.array_digest(a) != t40.array_digest(a.reshape(4, 3))
    b = a.copy()
    b[1, 1] = np.nextafter(b[1, 1], np.float32(100))                       # one ulp
    assert t40.array_digest(a) != t40.array_digest(b)


def _write_cache(root, scale=1.0):
    emb = root / "data" / "cache" / "embeddings"
    emb.mkdir(parents=True)
    for agent in t40.AGENTS:
        for split in t40.SPLITS:
            np.savez(emb / f"{agent}_{split}.npz", stay_id=np.arange(3) + 101,
                     hour=np.arange(3), risk=np.full(3, 0.1 * scale, np.float32),
                     latent=np.ones((3, 2), np.float32))
    np.savez(root / "data" / "cache" / "fair_gbm_scores.npz",
             **{"raw_FAIR+TRAJ": np.full(3, 0.2), "rawval_FAIR+TRAJ": np.full(3, 0.3)})
    (root / "h1").mkdir()
    np.savez(root / "h1" / "two_head_s42.npz", val_s42=np.full(3, 0.4),
             test_s42=np.full(3, 0.5 * scale), dhead_test_s42=np.full(3, 0.6),
             stay_id_test=np.arange(3) + 101)


def test_score_digests_cover_agents_fair_and_heads_but_never_keys(tmp_path):
    _write_cache(tmp_path)
    d = t40.score_digests(tmp_path / "data" / "cache", tmp_path / "h1" / "two_head_s42.npz")
    assert len(d) == len(t40.AGENTS) * len(t40.SPLITS) * 2 + 2 + 3
    assert not any("stay_id" in k or "hour" in k for k in d)


def test_compare_digests_reports_each_group(tmp_path):
    _write_cache(tmp_path / "a")
    _write_cache(tmp_path / "b", scale=2.0)          # changes risk and test_s42
    ref = t40.score_digests(tmp_path / "a/data/cache", tmp_path / "a/h1/two_head_s42.npz")
    same = t40.compare_digests(dict(ref), ref)
    assert same["flagship_all_identical"] and not same["different"]
    got = t40.score_digests(tmp_path / "b/data/cache", tmp_path / "b/h1/two_head_s42.npz")
    cmp = t40.compare_digests(got, ref)
    assert not cmp["flagship_all_identical"]
    assert "heads.test_s42" in cmp["different"] and "agent.gbm.test.risk" in cmp["different"]
    assert "agent.gbm.test.latent" in cmp["identical"]
    got.pop("heads.val_s42")
    assert "heads.val_s42" in t40.compare_digests(got, ref)["missing"]


def test_fair_differences_alone_do_not_fail_the_flagship_comparison():
    ref = {"agent.gbm.test.risk": "a", "heads.test_s42": "b", "fair.raw_FAIR+TRAJ": "c"}
    got = dict(ref, **{"fair.raw_FAIR+TRAJ": "x"})
    assert t40.compare_digests(got, ref)["flagship_all_identical"]


# --- isolation -----------------------------------------------------------------

def _clone(root):
    cache = root / "data" / "cache"
    (cache / "embeddings").mkdir(parents=True)
    for name in t40.INPUT_FILES:
        (cache / name).write_bytes(name.encode())
    return cache


def test_check_isolated_accepts_a_fresh_clone(tmp_path):
    _clone(tmp_path)
    t40.check_isolated(tmp_path, fresh=True)


def test_check_isolated_refuses_extra_cache_files_and_used_embeddings(tmp_path):
    cache = _clone(tmp_path)
    (cache / "two_head.npz").write_bytes(b"x")
    with pytest.raises(SystemExit, match="other than the inputs"):
        t40.check_isolated(tmp_path, fresh=True)
    t40.check_isolated(tmp_path, fresh=False)        # allowed after the grid run
    (cache / "two_head.npz").unlink()
    (cache / "embeddings" / "gbm_test.npz").write_bytes(b"x")
    with pytest.raises(SystemExit, match="not empty"):
        t40.check_isolated(tmp_path, fresh=True)


def test_check_isolated_refuses_a_symlinked_cache(tmp_path):
    real = _clone(tmp_path / "real")
    (tmp_path / "clone" / "data").mkdir(parents=True)
    (tmp_path / "clone" / "data" / "cache").symlink_to(real)
    with pytest.raises(SystemExit, match="symlink"):
        t40.check_isolated(tmp_path / "clone", fresh=True)


def test_verify_inputs_names_the_file_that_differs(tmp_path):
    cache = _clone(tmp_path)
    expected = {n: t40.sha256_file(cache / n) for n in t40.INPUT_FILES}
    assert t40.verify_inputs(cache, expected)["inputs_match_t36"]
    (cache / t40.INPUT_FILES[0]).write_bytes(b"changed")
    with pytest.raises(SystemExit, match="checksum differs"):
        t40.verify_inputs(cache, expected)


# --- what leaves the laptop ----------------------------------------------------

def test_assert_shareable_refuses_ids_and_private_names():
    private = {"hostxyz", "userxyz"}
    assert t40.assert_shareable({"cpu": "Intel", "n": 7}, private)
    stay_shaped = 3 * 10 ** 7 + 7      # built at runtime: no id-shaped literal in the repo
    with pytest.raises(ValueError):
        t40.assert_shareable({"n": stay_shaped}, private)
    with pytest.raises(SystemExit, match="user, host or path"):
        t40.assert_shareable({"os": "Windows 11 on HOSTXYZ"}, private)


def test_assert_shareable_ignores_private_strings_inside_digests():
    # A short hex-looking name can occur inside a sha256 by chance.
    assert t40.assert_shareable({"digests": {"a": "00beef00"}}, {"beef"})


def test_no_swap_rules_per_platform():
    gib = 2 ** 30
    ok = {"s": {"min_available_bytes": 4 * gib, "swap_out_bytes": 0,
                "swap_used_growth_bytes": 0}}
    assert t40.no_swap(ok, 16 * gib, windows=False)
    swapped = {"s": dict(ok["s"], swap_out_bytes=4096)}
    assert not t40.no_swap(swapped, 16 * gib, windows=False)
    pagefile = {"s": dict(ok["s"], swap_out_bytes=None, swap_used_growth_bytes=gib)}
    assert not t40.no_swap(pagefile, 16 * gib, windows=True)
    tight = {"s": dict(ok["s"], min_available_bytes=gib // 2)}
    assert not t40.no_swap(tight, 16 * gib, windows=False)


def test_run_stage_measures_a_child_process(tmp_path):
    pytest.importorskip("psutil")
    script = tmp_path / "child.py"
    script.write_text("import time\nb = bytearray(80 * 2**20)\nfor i in range(0, len(b), 4096): b[i] = 1\n"
                      "time.sleep(1.2)\nprint('done')\n")
    st = t40.run_stage([str(script)], tmp_path / "log.txt", cwd=tmp_path, interval=0.1)
    assert st["exit_code"] == 0 and st["wall_seconds"] >= 1.2
    assert st["peak_process_bytes"] >= 70 * 2 ** 20
    assert (tmp_path / "log.txt").read_text().strip() == "done"


# --- reuse mode ---------------------------------------------------------------

class _FakeXGB:
    calls = []

    def __init__(self, **kw):
        self.kw = kw
        _FakeXGB.calls.append(kw)
        self.best_iteration = 9

    def fit(self, X, y, eval_set=None, verbose=False):
        return self

    def predict_proba(self, X):
        p = np.linspace(0.1, 0.9, len(X))
        return np.c_[1 - p, p]


def test_fair_reuse_fits_exactly_the_grid_cell_it_reuses(monkeypatch):
    import xgboost

    import baselines_fair
    import t40_fair_reuse
    monkeypatch.setattr(xgboost, "XGBClassifier", _FakeXGB)
    X = np.zeros((20, 3))
    y = np.r_[np.zeros(10), np.ones(10)]
    _FakeXGB.calls = []
    baselines_fair.tune_gbm(X, y, X, y, verbose=False)
    grid = {(c["max_depth"], c["scale_pos_weight"]): c for c in _FakeXGB.calls}
    _FakeXGB.calls = []
    tune = t40_fair_reuse.fixed_cell(**t40_fair_reuse.COMMITTED)
    _, record = tune(X, y, X, y, verbose=False)
    assert len(_FakeXGB.calls) == 1
    assert _FakeXGB.calls[0] == grid[(4, 1.0)]
    assert record["max_depth"] == 4 and record["scale_pos_weight"] == 1.0
    assert record["n_trees"] == 10


# --- benchmark on Windows -----------------------------------------------------

def test_bench_does_not_time_an_ollama_error_as_a_card(monkeypatch):
    import bench_single_node as bench

    class R:
        def __init__(self, body):
            self.returncode, self.stdout = 0, json.dumps(body)

    def fake_run(argv, **kw):
        body = json.loads(argv[argv.index("-d") + 1])
        if body["model"] == "big":
            return R({"error": "model requires more system memory (9.1 GiB) than is available"})
        return R({"response": "card text", "eval_count": 90, "eval_duration": 9e9})

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    out = bench.llm_cards(["small", "big"], n_cards=2)
    assert out["small"]["n"] == 2 and out["small"]["tokens_per_s"] == pytest.approx(10.0)
    assert out["big"]["n"] == 0 and "more system memory" in out["big"]["skipped"]


def test_bench_peak_memory_has_a_path_without_the_resource_module(monkeypatch):
    import bench_single_node as bench
    assert bench.peak_rss_mb() > 0
    psutil = pytest.importorskip("psutil")

    class MI:
        peak_wset = 512 * 2 ** 20

    class P:
        def memory_info(self):
            return MI()

    monkeypatch.setattr(bench, "resource", None)
    monkeypatch.setattr(psutil, "Process", P)
    assert bench.peak_rss_mb() == pytest.approx(512.0)
