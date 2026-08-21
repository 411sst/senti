"""Frozen ECGFounder encoder: 12-lead waveform -> 1024-dim embedding (V6).

ECGFounder (Net1D, 33.8M params; NEJM AI 2025) was trained *supervised* on
10.77M ECGs / 150 diagnoses (Harvard-Emory ECG DB) and is the top-ranked frozen
12-lead feature extractor in the independent ICLR-2026 FM benchmark -- ahead of
ECG-FM, which only competes fine-tuned. Its native input (12-lead, 10 s,
500 Hz) matches MIMIC-IV-ECG exactly, so :func:`agents.ecg_encoder.read_ecg`
output feeds it with no resampling.

Used strictly as a **frozen** extractor, mirroring the ECG-FM path: embeddings
are extracted once, cached, and the small label head trains over the cache.

Preprocessing replicates the authors' ``dataset.py``: one **global** z-score
per record -- scalar mean/std over the whole (12, 5000) array, NOT per-lead --
``(x - mean) / (std + 1e-8)``.

The embedding is ``deep_features``: the temporal mean-pool right before the
classification head (``return_features=True`` in Net1D). The pretrained
``dense`` layer (their 150 diagnoses) is dropped at load.

Checkpoint: ``models/ecgfounder/12_lead_ECGFounder.pth`` from HF
``PKUDigitalHealth/ECGFounder`` (MIT).
"""

import numpy as np
import torch

from agents.net1d import Net1D

CHECKPOINT = "models/ecgfounder/12_lead_ECGFounder.pth"
EMBED_DIM = 1024


def load_encoder(checkpoint=CHECKPOINT, device="cpu"):
    """ECGFounder frozen, in eval mode, gradients disabled."""
    model = Net1D(
        in_channels=12, base_filters=64, ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16, stride=2, groups_width=16,
        verbose=False, use_bn=False, use_do=False,
        n_classes=1,                        # dummy; dense is not used
        return_features=True,
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k: v for k, v in state.items() if not k.startswith("dense.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    # only the (replaced) dense head may be missing; anything else is a bug
    bad = [k for k in missing if not k.startswith("dense.")]
    if bad or unexpected:
        raise RuntimeError(f"ECGFounder load mismatch: missing={bad}, "
                           f"unexpected={list(unexpected)}")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def embed_batch(model, signals, device="cpu"):
    """1024-dim ``deep_features`` for a list of (12, 5000) arrays.

    Applies the authors' per-record GLOBAL z-score before the forward pass.
    """
    x = np.stack(signals).astype(np.float32)              # (B, 12, 5000)
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True)
    x = (x - mean) / (std + 1e-8)
    _, feats = model(torch.from_numpy(x).to(device))      # (B, 1024)
    return feats.cpu().numpy().astype(np.float32)
