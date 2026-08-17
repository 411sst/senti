"""Encode every needed radiology note once, with Bio_ClinicalBERT (frozen).

    python scripts/encode_notes.py

Writes ``data/cache/note_embeddings.npz`` (note_ids + (N, 768) float32).
Checkpoints every few thousand notes so an interruption resumes rather than
restarts, and skips anything already encoded.

Notes are length-sorted before batching: padding is per-batch, so grouping
similar lengths together avoids wasting compute on padding tokens.
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agents.note_encoder import EMBED_DIM, embed_texts, load_encoder  # noqa: E402

TEXTS = REPO / "data" / "cache" / "note_texts.parquet"
OUT = REPO / "data" / "cache" / "note_embeddings.npz"
BATCH = 16
CHECKPOINT_EVERY = 4000


def save(done):
    ids = np.array(sorted(done), dtype=object)
    mat = np.stack([done[i] for i in ids]).astype(np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, note_ids=ids.astype(str), embeddings=mat)
    return len(ids)


def main():
    df = pd.read_parquet(TEXTS)
    print(f"[load] {len(df):,} note texts")

    done = {}
    if OUT.exists():
        z = np.load(OUT, allow_pickle=True)
        # Pull both arrays out ONCE. Indexing z["embeddings"] inside the
        # comprehension re-decompresses the whole compressed array on every
        # iteration -- with 90k notes cached that is ~25 TB of redundant
        # decompression and a fresh 277 MB allocation each time, which hangs the
        # resume and gets the process OOM-killed. It went unnoticed because the
        # first run started from an empty cache and never took this branch.
        cached_ids, cached_matrix = z["note_ids"], z["embeddings"]
        done = {str(n): cached_matrix[i] for i, n in enumerate(cached_ids)}
        print(f"[resume] {len(done):,} already encoded")

    todo = df[~df.note_id.astype(str).isin(done)].copy()
    if todo.empty:
        print("[done] cache is current")
        return
    # length-sort: per-batch padding means similar lengths batch efficiently
    todo["_len"] = todo.text.str.len()
    todo = todo.sort_values("_len").reset_index(drop=True)
    print(f"[todo] {len(todo):,} to encode")

    tokenizer, model = load_encoder()
    t0, since_ckpt = time.time(), 0
    for i in range(0, len(todo), BATCH):
        chunk = todo.iloc[i:i + BATCH]
        vecs = embed_texts(tokenizer, model, chunk.text.tolist())
        for note_id, vec in zip(chunk.note_id.astype(str), vecs):
            done[note_id] = vec
        since_ckpt += len(chunk)
        if since_ckpt >= CHECKPOINT_EVERY:
            save(done)
            since_ckpt = 0
            n = i + len(chunk)
            rate = n / max(time.time() - t0, 1e-9)
            print(f"  {n:,}/{len(todo):,}  ({rate:.0f}/s, "
                  f"ETA {(len(todo)-n)/max(rate,1e-9)/60:.0f} min)", flush=True)

    total = save(done)
    print(f"[done] {total:,} embeddings x {EMBED_DIM}d -> {OUT} "
          f"({OUT.stat().st_size/1e6:.0f} MB) in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
