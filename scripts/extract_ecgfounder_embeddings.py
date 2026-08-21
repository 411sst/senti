"""Extract frozen ECGFounder embeddings for every downloaded study (V6).

    python scripts/extract_ecgfounder_embeddings.py

Writes ``data/cache/ecgfounder_embeddings.npz`` (``embeddings`` (S, 1024),
``study_ids``) -- the standard frozen-embedding cache format. One-time; the
encoder never runs again after this (CLAUDE.md rule #5).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from agents.ecgfounder_encoder import EMBED_DIM, embed_batch, load_encoder  # noqa: E402
from agents.ecg_encoder import read_ecg  # noqa: E402

WAVE_DIR = REPO / "data" / "ecg_waveforms"
OUT_PATH = REPO / "data" / "cache" / "ecgfounder_embeddings.npz"
BATCH = 128


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_encoder(device=device)
    studies = sorted(int(p.stem) for p in WAVE_DIR.glob("*.dat") if p.stem.isdigit())
    print(f"[extract:ecgfounder] {len(studies):,} studies on {device}")

    t0 = time.time()
    kept_ids, chunks, batch = [], [], []
    for sid in studies:
        sig = read_ecg(WAVE_DIR / str(sid))
        if sig is None:
            continue
        batch.append(sig)
        kept_ids.append(sid)
        if len(batch) == BATCH:
            chunks.append(embed_batch(model, batch, device))
            batch = []
    if batch:
        chunks.append(embed_batch(model, batch, device))

    emb = np.concatenate(chunks) if chunks else np.zeros((0, EMBED_DIM), np.float32)
    assert emb.shape == (len(kept_ids), EMBED_DIM)
    np.savez_compressed(OUT_PATH, embeddings=emb,
                        study_ids=np.array(kept_ids, dtype=np.int64))
    print(f"[done] {emb.shape} -> {OUT_PATH} "
          f"({OUT_PATH.stat().st_size/1e6:.1f} MB) in {(time.time()-t0)/60:.1f} min; "
          f"{len(studies) - len(kept_ids)} unreadable skipped")


if __name__ == "__main__":
    main()
