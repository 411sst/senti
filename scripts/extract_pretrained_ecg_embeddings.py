"""Extract frozen embeddings from a pretrained ECG encoder (V5).

    python scripts/extract_pretrained_ecg_embeddings.py --backbone resnet
    python scripts/extract_pretrained_ecg_embeddings.py --backbone mamba

Loads models/ecg_pretrain/{backbone}.pt, freezes it, and embeds every
downloaded study once -> data/cache/ecg_{resnet_pre|mamba}_embeddings.npz with
keys ``embeddings`` (S, 256) and ``study_ids`` (S,) -- the exact format the
frozen-embedding EcgAgent machinery consumes. Standardization uses the
pretrain-train per-lead stats persisted in the checkpoint (val/test studies are
touched exactly once, here).
"""

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from agents.ecg_pretrain import BACKBONES, EcgPretrainModel  # noqa: E402
from agents.ecg_resnet_encoder import WaveformStore  # noqa: E402

OUT_NAME = {"resnet": "ecg_resnet_pre_embeddings.npz",
            "mamba": "ecg_mamba_embeddings.npz"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(BACKBONES), required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    ckpt_path = REPO / "models" / "ecg_pretrain" / f"{args.backbone}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EcgPretrainModel(BACKBONES[args.backbone](),
                             n_flags=len(ckpt["flag_names"]),
                             n_values=len(ckpt["zstats"][0]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    store = WaveformStore()
    store.load(store.available_studies())
    store.mean, store.std = ckpt["store_mean"], ckpt["store_std"]
    studies = np.array(sorted(store.index), dtype=np.int64)
    print(f"[extract:{args.backbone}] {len(studies):,} studies on {device}")

    out = np.zeros((len(studies), 256), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(studies), args.batch_size):
            chunk = studies[i:i + args.batch_size]
            out[i:i + len(chunk)] = model.embed(
                store.batch(chunk, device)).cpu().numpy()

    out_path = REPO / "data" / "cache" / OUT_NAME[args.backbone]
    np.savez_compressed(out_path, embeddings=out, study_ids=studies)
    print(f"[done] {out.shape} -> {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
