"""Frozen Bio_ClinicalBERT encoder: radiology report text -> 768-dim embedding.

Used strictly as a **frozen feature extractor** -- never fine-tuned. Embeddings
are computed once and cached, so retraining the agent's head costs nothing
(CLAUDE.md rule #5).

Pooling is a mask-aware mean over token states rather than the ``[CLS]`` vector.
BERT's ``[CLS]`` is only meaningful when a model has been trained with a
sentence-level objective on the target task; for a frozen off-the-shelf encoder,
mean-pooling over real (non-padding) tokens is the stronger and more standard
sentence representation.
"""

import numpy as np
import torch

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
EMBED_DIM = 768
MAX_TOKENS = 512      # BERT's limit; radiology reports have a median of ~190 tokens


def load_encoder(name=MODEL_NAME, device="cpu", threads=8):
    """Load the tokenizer + model frozen and in eval mode."""
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)          # frozen: never trained
    return tokenizer, model


@torch.no_grad()
def embed_texts(tokenizer, model, texts, device="cpu"):
    """Mask-aware mean-pooled embeddings for a list of note texts."""
    encoded = tokenizer(list(texts), truncation=True, max_length=MAX_TOKENS,
                        padding=True, return_tensors="pt").to(device)
    hidden = model(**encoded).last_hidden_state          # (B, T, 768)
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    # Mean over REAL tokens only -- padding must not dilute the representation.
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled.cpu().numpy().astype(np.float32)
