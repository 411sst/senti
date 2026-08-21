"""CNN-stem -> Mamba-blocks ECG encoder, pure PyTorch (V5 backbone B).

A selective-state-space alternative to the 1D-ResNet, sized to the same ~4M
parameter budget for a fair A/B under identical pretraining. Implemented from
the Mamba recurrence directly -- **no ``mamba-ssm`` dependency**: its fused CUDA
kernels require fragile source builds on this machine's Blackwell (sm_120)
GPU, and after the CNN stem the sequence is only ~250 tokens, so a naive
sequential scan is entirely affordable.

Design choices (see the plan):

* **Stem** downsamples 5000 samples -> 250 tokens x d_model=256, so the SSM
  reasons over beat-scale tokens rather than raw samples.
* **Bidirectionality via alternating scan direction per block.** ECG windows
  are entirely in the past at prediction time, so non-causality is fine -- and
  morphology (P before QRS before T) benefits from both directions. Alternating
  flips costs zero extra parameters, keeping the budget honest vs the ResNet.
* **S4D-real initialisation** for A and the standard softplus-inverse dt bias
  init, which is what makes small SSMs trainable from scratch.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.ecg_encoder import N_LEADS

EMBED_DIM = 256


class MambaBlock(nn.Module):
    """One selective-SSM block (Mamba recurrence, naive scan).

    Input/output ``(B, L, d_model)``. ``reverse=True`` scans right-to-left
    (implemented by flipping the sequence on the way in and out).
    """

    def __init__(self, d_model=256, expand=2, d_state=16, dt_rank=16,
                 d_conv=4, reverse=False):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner, self.d_state, self.reverse = d_inner, d_state, reverse

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner,
                                padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # S4D-real: A_n = -n for n = 1..d_state, shared shape per channel.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).expand(d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A.contiguous()))
        self.D = nn.Parameter(torch.ones(d_inner))

        # dt bias: softplus(bias) ~ U[1e-3, 1e-1] log-uniform (standard init).
        dt = torch.exp(torch.rand(d_inner)
                       * (math.log(1e-1) - math.log(1e-3)) + math.log(1e-3))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, u):                                  # (B, L, D)
        if self.reverse:
            u = u.flip(1)
        B_, L, _ = u.shape
        x, z = self.in_proj(u).chunk(2, dim=-1)            # (B, L, d_inner) x2
        x = self.conv1d(x.transpose(1, 2))[..., :L].transpose(1, 2)
        x = F.silu(x)

        dt_bc = self.x_proj(x)                             # (B, L, dt_rank + 2*ds)
        dt, Bm, Cm = torch.split(
            dt_bc, [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))               # (B, L, d_inner)

        A = -torch.exp(self.A_log)                         # (d_inner, ds)
        # Discretize per step INSIDE the loop: materialising deltaA/deltaB_x as
        # (B, L, d_inner, d_state) up front costs ~0.5 GB each at B=64 and OOMs
        # an 11 GB GPU once autograd keeps them for backward.
        h = x.new_zeros(B_, self.d_inner, self.d_state)
        ys = []
        for t in range(L):                                 # naive scan, L ~ 250
            dt_t = delta[:, t].unsqueeze(-1)               # (B, d_inner, 1)
            h = (torch.exp(dt_t * A) * h
                 + dt_t * Bm[:, t].unsqueeze(1) * x[:, t].unsqueeze(-1))
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1)) # (B, d_inner)
        y = torch.stack(ys, dim=1) + self.D * x            # (B, L, d_inner)

        out = self.out_proj(y * F.silu(z))
        return out.flip(1) if self.reverse else out


class EcgMambaEncoder(nn.Module):
    """12-lead waveform (B, 12, 5000) -> ``final_embedding_size`` vector."""

    def __init__(self, in_channels=N_LEADS, final_embedding_size=EMBED_DIM,
                 d_model=256, n_blocks=8, expand=2, d_state=16, dt_rank=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=9, stride=5, padding=4, bias=False),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )                                                  # 5000 -> 250 tokens
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_blocks))
        self.blocks = nn.ModuleList(
            MambaBlock(d_model, expand, d_state, dt_rank, reverse=bool(i % 2))
            for i in range(n_blocks)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.projection_head = nn.Linear(d_model, final_embedding_size)
        self.use_checkpoint = True        # toggleable for tests/debugging

    def forward(self, x):                                  # (B, 12, 5000)
        h = self.stem(x).transpose(1, 2)                   # (B, 250, d_model)
        for norm, block in zip(self.norms, self.blocks):
            if self.training and self.use_checkpoint and h.requires_grad:
                # The scan's per-step intermediates dominate training memory;
                # checkpointing recomputes them in backward (~2x block compute,
                # ~8x less activation memory) so batch 64 fits an 11 GB GPU.
                h = h + torch.utils.checkpoint.checkpoint(
                    lambda hh, n=norm, b=block: b(n(hh)), h, use_reentrant=False)
            else:
                h = h + block(norm(h))                     # pre-LN residual
        h = self.final_norm(h).mean(dim=1)                 # mean-pool tokens
        return self.projection_head(h)                     # (B, 256)
