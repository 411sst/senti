"""Empirical alignment / leakage audit of the orchestrator's data flow.

    python scripts/audit_alignment.py

``tests/test_alignment_audit.py`` pins these invariants on synthetic data where
every value encodes its own key. This script is the companion check against the
REAL artifacts on disk -- the cohort Parquet and the cached agent emissions --
because those get regenerated whenever an agent is refit, and a stale or
mismatched cache would not show up in a unit test.

Re-run it after anything that rewrites data/cache/embeddings/.

Four questions, in order:
  1. do the three agents' outputs join on identical (stay_id, hour), with the
     right label attached?
  2. does the trajectory stack at hour t see only hours <= t?
  3. was each agent's vector for hour t built only from data before t?
  4. is the label at t the FUTURE 24h deterioration label, correctly aligned?
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from agents.base import AgentOutput  # noqa: E402
from agents.labs_agent import build_lab_features, load_labs  # noqa: E402
from orchestrator.fusion import (  # noqa: E402
    AgentBank, Orchestrator, resolve_config,
)

COHORT = REPO / "data" / "cache" / "mimic_cohort_hourly.parquet"
RAW = REPO / "data" / "cache" / "mimic_raw_hourly.parquet"
EMB = REPO / "data" / "cache" / "embeddings"

# Derived from the SHIPPED configuration rather than hardcoded: the audit must
# follow whatever the orchestrator actually fuses, or it silently stops checking
# a modality (this bit once -- the list said 3 agents while the bank loaded 4).
AGENTS = resolve_config(emb_dir=EMB)[1]

failures = []


def check(label, ok, detail=""):
    status = "PASS" if ok else "**FAIL**"
    print(f"  [{status}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(label)
    return ok


def section(n, title):
    print(f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}")


def check_freshness():
    """The orchestrator cache must be newer than every agent cache it derives from.

    The orchestrator consumes the agents' cached latents, so refitting an agent
    invalidates the orchestrator's emissions. This actually happened: a results
    run refit all three agents but did not re-cache the orchestrator, leaving a
    file trained on superseded latents -- so the ablation's LATENT column and the
    results table's ORCHESTRATOR row came from two different training runs. The
    numbers barely moved, which is precisely why nothing caught it.
    """
    section(0, "ARTIFACT FRESHNESS: orchestrator cache newer than its inputs")
    for split in ("train", "val", "test"):
        orch_path = EMB / f"orchestrator_{split}.npz"
        if not orch_path.exists():
            check(f"{split}: orchestrator cache exists", False, str(orch_path))
            continue
        orch_mtime = orch_path.stat().st_mtime
        stale_against = []
        for agent in AGENTS:
            agent_path = EMB / f"{agent}_{split}.npz"
            if agent_path.exists() and agent_path.stat().st_mtime > orch_mtime:
                age = (agent_path.stat().st_mtime - orch_mtime) / 60
                stale_against.append(f"{agent} (+{age:.0f}min)")
        check(f"{split}: orchestrator cache is newer than all agent caches",
              not stale_against,
              "stale vs " + ", ".join(stale_against) if stale_against else "")


def main():
    cohort = pd.read_parquet(COHORT)
    bank = AgentBank.load()

    check_freshness()

    # ---------------------------------------------------------------- 1 ---
    section(1, "CROSS-AGENT JOIN: same (stay_id, hour) everywhere, right label")
    check("cohort has no duplicate (stay_id, hour)",
          cohort.duplicated(subset=["stay_id", "hour"]).sum() == 0,
          f"{len(cohort):,} rows")
    check("bank index is unique (a duplicate would mis-join silently)",
          bank._index.is_unique)
    check("bank covers the cohort exactly",
          len(bank.keys) == len(cohort)
          and set(map(tuple, bank.keys.to_numpy()))
          == set(map(tuple, cohort[["stay_id", "hour"]].to_numpy())))

    for split in ("train", "val", "test"):
        outs = {n: AgentOutput.load(EMB / f"{n}_{split}.npz") for n in AGENTS}
        ref = outs[AGENTS[0]].keys
        check(f"{split}: all agents emit identical keys",
              all(o.keys.equals(ref) for o in outs.values()), f"n={len(ref):,}")
        check(f"{split}: emissions are (stay_id, hour)-sorted",
              ref.equals(ref.sort_values(["stay_id", "hour"]).reset_index(drop=True)))

    # trace individual rows back through the join
    test = cohort[cohort.split == "test"]
    sample = test.sample(8, random_state=0)
    X, keys = bank.lookup(sample)
    src = {n: AgentOutput.load(EMB / f"{n}_test.npz") for n in AGENTS}
    idx = {n: pd.MultiIndex.from_arrays([o.keys.stay_id, o.keys.hour])
           for n, o in src.items()}
    joined = keys.merge(sample, on=["stay_id", "hour"], how="left")

    traced, n2_ok = True, True
    for i, row in joined.iterrows():
        sid, hr = int(row.stay_id), int(row.hour)
        offset = 0
        for n in bank.agent_names:
            dim = bank.latent_dims[n]
            pos = idx[n].get_loc((sid, hr))
            traced &= np.array_equal(X[i, offset:offset + dim], src[n].latent[pos])
            offset += dim
        traced &= np.allclose(
            X[i, offset:offset + len(AGENTS)],
            [src[n].risk[idx[n].get_loc((sid, hr))] for n in bank.agent_names])
        # independent cross-check: the NEWS2 latent is 7 sub-scores that must sum
        # to the cohort's news2 column, which is computed in a separate code path
        n2_ok &= abs(src["news2"].latent[idx["news2"].get_loc((sid, hr))].sum()
                     - float(row.news2)) < 1e-6
    check("traced rows: every agent block matches its own cache", traced,
          f"{len(joined)} patient-hours")
    check("NEWS2 latent sums to the cohort's news2 column (independent path)", n2_ok)

    # ---------------------------------------------------------------- 2 ---
    section(2, "TRAJECTORY STACK: hour t uses only hours <= t")
    orch = Orchestrator(bank=bank, d_model=8, dilations=(1, 2), verbose=False)
    orch.net = None
    from orchestrator.fusion import LatentFusionNet
    torch.manual_seed(0)
    net = LatentFusionNet(bank.latent_dims, bank.agent_names,
                          d_model=8, dilations=(1, 2, 4)).eval()
    T = 40
    latents = {n: torch.randn(1, T, d) for n, d in bank.latent_dims.items()}
    risks = torch.rand(1, T, len(AGENTS))
    with torch.no_grad():
        base, base_emb, base_attn = net(latents, risks)
        past_ok, registers = True, True
        for cut in (5, 12, 25):
            tampered = {k: v.clone() for k, v in latents.items()}
            for v in tampered.values():
                v[:, cut:, :] = 99.0
            tr = risks.clone()
            tr[:, cut:, :] = 0.0
            out, emb, attn = net(tampered, tr)
            past_ok &= (torch.allclose(base[:, :cut], out[:, :cut], atol=1e-6)
                        and torch.allclose(base_emb[:, :cut], emb[:, :cut], atol=1e-6)
                        and torch.allclose(base_attn[:, :cut], attn[:, :cut], atol=1e-6))
            registers &= not torch.allclose(base[:, cut:], out[:, cut:], atol=1e-6)
    check("rewriting hours >= t leaves every output before t bit-identical", past_ok,
          "cuts at 5/12/25")
    check("the perturbation does register after the cut (test can fail)", registers)

    gaps = (cohort.sort_values(["stay_id", "hour"]).groupby("stay_id")["hour"]
            .apply(lambda s: int((s.diff().dropna() != 1).sum())))
    check("every stay's hours are consecutive (TCN position == real hour offset)",
          int((gaps > 0).sum()) == 0, f"{len(gaps):,} stays")

    # label alignment inside the sequence builder
    stays = test.stay_id.drop_duplicates().head(40)
    sub = test[test.stay_id.isin(stays)]
    seqs = orch._sequences(sub)
    truth = sub.set_index(["stay_id", "hour"])["label_24h"]
    bad = sum(int(truth.loc[(s["stay_id"], hr)]) != int(yv)
              for s in seqs for hr, yv in zip(s["hours"], s["y"]))
    check("y[i] matches the label of keys[i] inside each sequence", bad == 0,
          f"{sum(len(s['hours']) for s in seqs):,} rows")
    check("hours ascend within every sequence",
          all((np.diff(s["hours"]) > 0).all() for s in seqs if len(s["hours"]) > 1))

    # ---------------------------------------------------------------- 3 ---
    section(3, "AGENT-LEVEL LEAKAGE: each vector for hour t built from data before t")
    raw = pd.read_parquet(RAW).rename(columns={"subject_id": "patient_id"})
    raw["temperature"] = pd.to_numeric(raw["temperature"], errors="coerce")
    probe = cohort.stay_id.drop_duplicates().head(60).tolist()
    r = raw[raw.stay_id.isin(probe)]
    c = cohort[cohort.stay_id.isin(probe)].sort_values(["stay_id", "hour"])

    # NOTE: the source Parquets are NOT stored in (stay_id, hour) order -- tens of
    # thousands of stays have out-of-order rows. The pipeline sorts before doing
    # anything temporal (build_before_t_features / build_lab_features both call
    # sort_values first); this audit must too, or `.iloc[-1]` picks an arbitrary
    # row rather than the most recent one and reports phantom violations.
    viol = leaks = checked = 0
    for sid, cg in c.groupby("stay_id"):
        rg = r[r.stay_id == sid].set_index("hour")
        for col in ("heart_rate", "resp_rate", "spo2", "sbp"):
            obs = rg[col].dropna().sort_index()          # chronological
            for hr, val in zip(cg["hour"], cg[col]):
                checked += 1
                past = obs[obs.index < hr]                    # STRICTLY before t
                expect = past.iloc[-1] if len(past) else np.nan
                if len(past) and (hr - past.index[-1]) > 12:  # staleness bound
                    expect = np.nan
                if np.isclose(val, expect, equal_nan=True):
                    continue
                viol += 1
                at_t = obs[obs.index <= hr]
                if len(at_t) and np.isclose(val, at_t.iloc[-1]):
                    leaks += 1
    check("vitals at hour t == last raw observation strictly BEFORE t",
          viol == 0, f"{checked:,} cells, {viol} violations, {leaks} using hour t")

    labs = load_labs()
    lsub = labs[labs.stay_id.isin(probe)]
    feat = build_lab_features(c[["stay_id", "hour"]], labs).set_index(["stay_id", "hour"])
    lviol = lchecked = 0
    for sid in probe[:25]:
        lg = lsub[lsub.stay_id == sid].set_index("hour")
        if lg.empty:
            continue
        for col in ("lab_creatinine", "lab_wbc", "lab_lactate"):
            obs = lg[col].dropna().sort_index()          # chronological (see above)
            for hr in c[c.stay_id == sid]["hour"]:
                lchecked += 1
                val = feat.loc[(sid, hr), col]
                past = obs[obs.index < hr]
                expect = past.iloc[-1] if len(past) else np.nan
                if len(past) and (hr - past.index[-1]) > 24:
                    expect = np.nan
                if not np.isclose(val, expect, equal_nan=True):
                    lviol += 1
    check("labs at hour t == last result AVAILABLE (storetime) before t",
          lviol == 0, f"{lchecked:,} cells, {lviol} violations")

    # the orchestrator must not be able to bypass the agents
    before = orch._sequences(sub)
    scrambled = sub.copy()
    rng = np.random.default_rng(0)
    raw_cols = [col for col in scrambled.columns
                if col not in ("stay_id", "hour", "label_24h")
                and scrambled[col].dtype.kind in "fi"]
    for col in raw_cols:
        scrambled[col] = rng.normal(size=len(scrambled)) * 1e6
    after = orch._sequences(scrambled)
    check("scrambling raw cohort columns leaves the fused input unchanged",
          all(np.array_equal(a["X"], b["X"]) for a, b in zip(before, after)),
          f"{len(raw_cols)} columns scrambled")

    # ---------------------------------------------------------------- 4 ---
    section(4, "LABEL TIMING: label at t is the FUTURE (t, t+24] window")
    h = cohort["hour"].to_numpy(dtype=float)
    ev = cohort["event_h"].to_numpy(dtype=float)
    lab = cohort["label_24h"].to_numpy(dtype=int)
    forward = (~np.isnan(ev) & (ev > h) & (ev <= h + 24)).astype(int)
    check("stored label == recomputed (t, t+24] on every row",
          int((lab != forward).sum()) == 0, f"{len(lab):,} rows")

    shifted = (~np.isnan(ev) & (ev > h - 1) & (ev <= h + 23)).astype(int)
    backward = (~np.isnan(ev) & (ev >= h - 24) & (ev < h)).astype(int)
    check("label is NOT the off-by-one window (t-1, t+23]",
          int((lab != shifted).sum()) > 0, f"{int((lab != shifted).sum()):,} rows differ")
    check("label is NOT backward-looking [t-24, t)",
          int((lab != backward).sum()) > 0, f"{int((lab != backward).sum()):,} rows differ")
    gap = (ev - h)[~np.isnan(ev)]
    check("every event lies strictly AFTER its prediction hour (censoring holds)",
          float(gap.min()) > 0, f"min gap {gap.min():.3f}h")

    # ---------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    if failures:
        print(f"AUDIT FAILED -- {len(failures)} check(s) misaligned:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("AUDIT CLEAN -- no misalignment, no leakage found.")
    print(f"{'=' * 78}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
