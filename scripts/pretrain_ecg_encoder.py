"""Pretrain an ECG waveform encoder on machine-measurement targets (V5).

    python scripts/pretrain_ecg_encoder.py --backbone resnet
    python scripts/pretrain_ecg_encoder.py --backbone mamba

Trains on TRAIN-split downloaded studies only (patient-grouped 90/10 carve-out
for early stopping -- the real val split is never touched here) and saves the
checkpoint to models/ecg_pretrain/{backbone}.pt.

GATE 3: the printed pretrain-val macro-AUROC must clearly beat 0.5, and the
loss curve must not peak at epoch 1 (the V2 failure signature). If the encoder
cannot even learn machine-readable structure at this data scale, no label head
will fix it -- stop V5.
"""

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from agents.ecg_pretrain import BACKBONES, pretrain  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(BACKBONES), required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    out = REPO / "models" / "ecg_pretrain" / f"{args.backbone}.pt"
    t0 = time.time()
    _, history = pretrain(args.backbone, epochs=args.epochs,
                          batch_size=args.batch_size, out_path=out)
    best = min(history, key=lambda h: h["val"])
    print(f"\n[done] {args.backbone}: {len(history)} epochs in "
          f"{(time.time()-t0)/60:.0f} min | best val loss {best['val']:.4f} "
          f"@ epoch {best['epoch']} | macro-AUROC there {best['auroc']:.3f}")
    print(f"GATE 3: macro-AUROC {'PASS' if best['auroc'] > 0.6 else 'CHECK'} "
          f"(must clearly beat 0.5); best epoch {best['epoch']} "
          f"({'not ' if best['epoch'] > 1 else ''}epoch-1-peaked)")


if __name__ == "__main__":
    main()
