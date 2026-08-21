"""Agents V5: frozen pretrained waveform encoders -> label_24h heads.

Two thin :class:`agents.ecg_agent.EcgAgent` subclasses differing only in name
and embedding-cache path. The encoders (1D-ResNet and CNN->Mamba, pretrained on
machine-measurement targets by :mod:`agents.ecg_pretrain`) are frozen; their
embeddings are extracted once by ``scripts/extract_pretrained_ecg_embeddings.py``
and everything else -- presence guard, staleness, z-scoring, early stopping,
calibration, caching -- is the inherited, tested V1 machinery.
"""

from agents.ecg_agent import EcgAgent


class EcgResnetPreAgent(EcgAgent):
    """V5a: pretrained-then-frozen 1D-ResNet embeddings."""

    name = "ecg_resnet_pre"

    def __init__(self, embed_cache="data/cache/ecg_resnet_pre_embeddings.npz",
                 **kwargs):
        super().__init__(embed_cache=embed_cache, **kwargs)


class EcgMambaAgent(EcgAgent):
    """V5b: pretrained-then-frozen CNN->Mamba embeddings."""

    name = "ecg_mamba"

    def __init__(self, embed_cache="data/cache/ecg_mamba_embeddings.npz",
                 **kwargs):
        super().__init__(embed_cache=embed_cache, **kwargs)


class EcgFounderAgent(EcgAgent):
    """V6: frozen ECGFounder (Net1D, 1024-d) embeddings -- see
    :mod:`agents.ecgfounder_encoder`. Not pretrained by us; included here
    because it consumes the identical frozen-embedding machinery."""

    name = "ecgfounder"

    def __init__(self, embed_cache="data/cache/ecgfounder_embeddings.npz",
                 **kwargs):
        super().__init__(embed_cache=embed_cache, **kwargs)
