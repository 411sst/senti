"""Ultra-short HRV + beat features per downloaded ECG study (H2 / meas++).

    python scripts/build_ecg_hrv_features.py [--force]

From each 10 s 12-lead strip: R-peaks on lead II (fallback lead I) via
NeuroKit2, then the subset of HRV measures that validity studies show are
usable on 10 s recordings (RMSSD r~0.9 vs 5-min reference; SDNN marginal;
frequency-domain invalid and therefore excluded):

    hr_nk, rmssd, sd1, pnn50, cvnn, n_beats, r_amp_cv

Critical-care literature ties depressed RMSSD/SDNN to ICU mortality, which is
exactly the deterioration-adjacent signal the cart's machine read does NOT
provide (machine_measurements has single-beat intervals, no variability).

Writes ``data/cache/ecg_hrv_features.npz`` (``embeddings`` (S, 7),
``study_ids``, ``feature_names``) -- the standard guarded-slot cache format.
Studies where peak detection fails (<5 beats) are dropped -> the consuming
agent treats them as HRV-absent.
"""

import argparse
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from agents.ecg_encoder import SAMPLE_RATE, read_ecg  # noqa: E402

WAVE_DIR = REPO / "data" / "ecg_waveforms"
OUT_PATH = REPO / "data" / "cache" / "ecg_hrv_features.npz"
FEATURES = ["hr_nk", "rmssd", "sd1", "pnn50", "cvnn", "n_beats", "r_amp_cv"]


def hrv_features(sig):
    """(12, 5000) float32 -> (7,) features or None if peaks can't be found."""
    import neurokit2 as nk

    for lead in (1, 0):                          # lead II, fallback lead I
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clean = nk.ecg_clean(sig[lead], sampling_rate=SAMPLE_RATE)
                _, info = nk.ecg_peaks(clean, sampling_rate=SAMPLE_RATE)
            peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
        except Exception:
            continue
        if len(peaks) < 5:
            continue
        rr = np.diff(peaks) / SAMPLE_RATE * 1000.0          # ms
        rr = rr[(rr > 200) & (rr < 3000)]
        if len(rr) < 4:
            continue
        diff = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(diff ** 2)))
        feats = np.array([
            60000.0 / rr.mean(),                            # hr_nk
            rmssd,
            rmssd / np.sqrt(2.0),                           # sd1
            100.0 * float((np.abs(diff) > 50).mean()),      # pnn50
            float(rr.std() / rr.mean()),                    # cvnn
            float(len(peaks)),                              # n_beats
            float(np.std(clean[peaks]) /
                  (np.abs(np.mean(clean[peaks])) + 1e-6)),  # r_amp_cv
        ], dtype=np.float32)
        if np.isfinite(feats).all():
            return feats
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if OUT_PATH.exists() and not args.force:
        print(f"[skip] {OUT_PATH} exists (use --force)")
        return

    studies = sorted(int(p.stem) for p in WAVE_DIR.glob("*.dat")
                     if p.stem.isdigit())
    print(f"[scan] {len(studies):,} studies")
    ids, rows, failed = [], [], 0
    t0 = time.time()
    for k, sid in enumerate(studies):
        sig = read_ecg(WAVE_DIR / str(sid))
        f = hrv_features(sig) if sig is not None else None
        if f is None:
            failed += 1
            continue
        ids.append(sid)
        rows.append(f)
        if k % 1000 == 0:
            rate = (k + 1) / max(time.time() - t0, 1e-9)
            print(f"  {k+1:,}/{len(studies):,} ({rate:.0f}/s, "
                  f"ETA {(len(studies)-k)/max(rate,1e-9)/60:.1f} min)", flush=True)

    X = np.stack(rows)
    np.savez_compressed(OUT_PATH, embeddings=X.astype(np.float32),
                        study_ids=np.array(ids, dtype=np.int64),
                        feature_names=np.array(FEATURES))
    print(f"[done] {X.shape} -> {OUT_PATH} | {failed:,} studies without "
          f"detectable beats ({100*failed/len(studies):.1f}%) "
          f"in {(time.time()-t0)/60:.1f} min")
    med = np.median(X, axis=0)
    print("  medians: " + " ".join(f"{n}={m:.1f}" for n, m in zip(FEATURES, med)))


if __name__ == "__main__":
    main()
