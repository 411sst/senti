"""Frozen ECG-FM encoder: 12-lead waveform -> 768-dim embedding.

ECG-FM (``wav2vec2_cmsc``, 12 transformer layers, 768-dim) is used strictly as a
**frozen feature extractor** -- we never train or fine-tune it. Embeddings are
extracted once and cached to disk, so the encoder runs exactly once per ECG no
matter how many times the agent is retrained (CLAUDE.md rule #5).

Preprocessing matches the checkpoint's own task config: 12 leads in the standard
order, 500 Hz, and ``normalize=False`` (the checkpoint was trained on unnormalised
millivolt signals, so we must not z-score here).

The encoder emits one vector per time-step; we mean-pool over time to get a
single per-ECG embedding, which is the standard way to use a wav2vec2-style
encoder as a fixed feature extractor.
"""

from pathlib import Path

import numpy as np
import torch

CHECKPOINT = "models/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt"
EMBED_DIM = 768          # encoder_embed_dim from the checkpoint config
N_LEADS = 12
SAMPLE_RATE = 500        # Hz, as recorded in MIMIC-IV-ECG
DURATION_S = 10
EXPECTED_SAMPLES = SAMPLE_RATE * DURATION_S      # 5000


def load_encoder(checkpoint=CHECKPOINT, device="cpu"):
    """Load ECG-FM frozen, in eval mode, with gradients disabled."""
    from fairseq_signals.models import build_model_from_checkpoint

    model = build_model_from_checkpoint(Path(checkpoint))
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)     # frozen: never trained, never fine-tuned
    return model


def read_ecg(record_path):
    """Read a WFDB record as a (12, 5000) float32 array in the canonical lead order.

    Returns ``None`` if the record is unreadable or has the wrong shape, so a
    handful of corrupt files cannot abort a long extraction run.
    """
    import wfdb

    try:
        rec = wfdb.rdrecord(str(record_path))
    except Exception:
        return None
    signal = np.asarray(rec.p_signal, dtype=np.float32)      # (samples, leads)
    if signal.ndim != 2 or signal.shape[1] != N_LEADS:
        return None
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    signal = signal.T                                        # (leads, samples)

    # Pad or trim to exactly 10 s so every embedding sees the same window.
    if signal.shape[1] < EXPECTED_SAMPLES:
        pad = EXPECTED_SAMPLES - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad)), mode="constant")
    elif signal.shape[1] > EXPECTED_SAMPLES:
        signal = signal[:, :EXPECTED_SAMPLES]
    return signal


@torch.no_grad()
def embed_batch(model, signals, device="cpu"):
    """Mean-pooled 768-dim embeddings for a list of (12, 5000) arrays.

    Uses ``extract_features(...)["x"]`` -- the post-transformer encoder output,
    ``(B, 312, 768)``. Two nearby tensors are the wrong ones and both look
    plausible enough to grab by accident:

    * ``forward()["x"]`` is the wav2vec2 **contrastive logits**, shaped
      ``(num_negatives+1, B, T')`` and containing ``inf``. Pooling it yields
      garbage of the wrong dimensionality.
    * ``extract_features(...)["features"]`` is the 256-dim convolutional output
      *before* the transformer, so it discards the model's contextual reasoning.

    312 time-steps is 5000 samples downsampled by the conv stack's stride of 16;
    we mean-pool over them for one fixed vector per ECG.
    """
    x = torch.from_numpy(np.stack(signals)).to(device)        # (B, 12, 5000)
    out = model.extract_features(source=x, padding_mask=None)
    features = out["x"]                                       # (B, 312, 768)
    return features.mean(dim=1).cpu().numpy().astype(np.float32)
