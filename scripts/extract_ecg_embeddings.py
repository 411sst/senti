"""Extract frozen ECG-FM embeddings for every downloaded waveform, once.

    python scripts/extract_ecg_embeddings.py

Reads every record in ``data/ecg_waveforms/`` and writes a single
``data/cache/ecg_embeddings.npz`` holding ``study_ids`` and a
``(N, 768)`` float32 matrix. The encoder is frozen and never re-run: this is the
point of a foundation model used as a feature extractor (CLAUDE.md rule #5).

Resumable -- studies already present in the cache are skipped, so adding more
waveforms later only encodes the new ones.
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from agents.ecg_encoder import EMBED_DIM, embed_batch, load_encoder, read_ecg  # noqa: E402

WAVE_DIR = REPO / "data" / "ecg_waveforms"
OUT = REPO / "data" / "cache" / "ecg_embeddings.npz"
BATCH = 16


def main():
    records = sorted(p.with_suffix("") for p in WAVE_DIR.glob("*.hea"))
    print(f"[scan] {len(records):,} records on disk")

    done = {}
    if OUT.exists():
        z = np.load(OUT)
        done = {int(s): z["embeddings"][i] for i, s in enumerate(z["study_ids"])}
        print(f"[cache] {len(done):,} already encoded")

    todo = [r for r in records if int(r.name) not in done]
    print(f"[todo] {len(todo):,} to encode")
    if not todo:
        print("[done] cache is current")
        return

    model = load_encoder()
    t0, unreadable = time.time(), 0
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        signals, ids = [], []
        for rec in batch:
            sig = read_ecg(rec)
            if sig is None:
                unreadable += 1
                continue
            signals.append(sig)
            ids.append(int(rec.name))
        if signals:
            for sid, vec in zip(ids, embed_batch(model, signals)):
                done[sid] = vec
        if (i // BATCH) % 10 == 0:
            n = i + len(batch)
            rate = n / max(time.time() - t0, 1e-9)
            print(f"  {n:,}/{len(todo):,}  ({rate:.1f}/s, "
                  f"ETA {(len(todo)-n)/max(rate,1e-9)/60:.1f} min)", flush=True)

    study_ids = np.array(sorted(done), dtype=np.int64)
    embeddings = np.stack([done[s] for s in study_ids]).astype(np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, study_ids=study_ids, embeddings=embeddings)
    print(f"[done] {len(study_ids):,} embeddings x {embeddings.shape[1]}d "
          f"-> {OUT} ({OUT.stat().st_size/1e6:.0f} MB) in "
          f"{(time.time()-t0)/60:.1f} min; {unreadable} unreadable")
    assert embeddings.shape[1] == EMBED_DIM


if __name__ == "__main__":
    main()
