"""Run the NEWS2 tool-agent and report it in the standard agent table.

    python scripts/run_news2_agent.py

Nothing is trained -- NEWS2 is a fixed spec -- but the agent is still
isotonic-calibrated on val, and its latents are cached like any other agent's.
The AUPRC and ECE here should reproduce the NEWS2 row of the existing baseline
table; that match is the check the agent is wired up correctly.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from agents.news2_agent import News2Agent  # noqa: E402
from baselines_mimic import load_splits  # noqa: E402
from metrics import auprc, ece  # noqa: E402
from mimic_cohort import EVENT_TYPES  # noqa: E402

LABEL = "label_24h"
EMB_DIR = REPO / "data" / "cache" / "embeddings"

# The NEWS2 row of the baseline table (src/baselines_mimic.py), for comparison.
BASELINE = {"auprc": 0.108, "ece_raw": 0.1623, "ece_cal": 0.0014,
            "vent": 0.059, "vaso": 0.041, "death": 0.062}


def main():
    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]

    agent = News2Agent().fit(train, val)

    for name, df in (("train", train), ("val", val), ("test", test)):
        out = agent.emit_and_cache(df, EMB_DIR / f"{agent.name}_{name}.npz")
        print(f"[cache] {name}: {out.latent.shape} latents -> "
              f"{EMB_DIR / f'{agent.name}_{name}.npz'}")

    test_sorted = test.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    out = agent.emit(test)
    assert (out.keys["stay_id"].to_numpy() == test_sorted["stay_id"].to_numpy()).all()

    y = test_sorted[LABEL].to_numpy()
    risk_cal = out.risk
    raw_score = agent.raw_risk(test)          # 0-20
    raw_prob = agent.raw_probability(test)    # 0-20 rescaled to [0,1]

    # Wiring check: the agent must reproduce the cohort's stored NEWS2 column.
    stored = test_sorted["news2"].to_numpy(dtype=float)
    max_diff = float(abs(raw_score - stored).max())
    print(f"\n[check] agent NEWS2 vs cohort 'news2' column: max abs diff {max_diff:g} "
          f"({'MATCH' if max_diff == 0 else 'MISMATCH'})")

    cal = agent.calibration_summary
    print(f"\nNEWS2 tool-agent on test ({len(y):,} patient-hours, "
          f"{test_sorted['stay_id'].nunique():,} stays, "
          f"prevalence {100 * y.mean():.2f}%)")
    print(f"  latent dim {out.latent_dim} = {agent.component_names}")
    print(f"  isotonic calibration fit on val ({cal['split_rows']:,} rows, "
          f"{cal['positives']:,} positive)\n")

    print(f"  {'model':<18}{'AUPRC raw':>11}{'AUPRC cal':>11}{'ECE raw':>10}{'ECE cal':>10}")
    print(f"  {'NEWS2 agent':<18}{auprc(y, raw_score):>11.3f}{auprc(y, risk_cal):>11.3f}"
          f"{ece(y, raw_prob):>10.4f}{ece(y, risk_cal):>10.4f}")
    print(f"  {'(NEWS2 baseline)':<18}{BASELINE['auprc']:>11.3f}{BASELINE['auprc']:>11.3f}"
          f"{BASELINE['ece_raw']:>10.4f}{BASELINE['ece_cal']:>10.4f}")
    print(f"  {'(vitals-TCN)':<18}{0.161:>11.3f}{0.156:>11.3f}{0.0519:>10.4f}{0.0023:>10.4f}")
    print(f"  {'(GBM)':<18}{0.170:>11.3f}{0.170:>11.3f}{0.0036:>10.4f}{0.0016:>10.4f}")
    print(f"  {'(random)':<18}{y.mean():>11.3f}{y.mean():>11.3f}")

    print("\n  AUPRC by event type:")
    types = list(EVENT_TYPES)
    print("    " + f"{'model':<18}" + "".join(f"{t:>10}" for t in types))
    per_type = {t: auprc(test_sorted[f"{LABEL}_{t}"].to_numpy(), raw_score) for t in types}
    print("    " + f"{'NEWS2 agent':<18}" + "".join(f"{per_type[t]:>10.3f}" for t in types))
    print("    " + f"{'(NEWS2 baseline)':<18}"
          + "".join(f"{BASELINE[t]:>10.3f}" for t in types))

    print("\n  Mean sub-score contribution (the agent's latent, averaged):")
    for name, mean in zip(agent.component_names, out.latent.mean(axis=0)):
        print(f"    {name:<14}{mean:>7.3f}")


if __name__ == "__main__":
    main()
