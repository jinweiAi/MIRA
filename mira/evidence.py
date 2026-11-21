"""Low-rank evidence projection and receiving-slot value adaptation."""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .routing import normalization_dtype

class EvidenceProjection(nn.Module):
    """A_b = up_b @ down: shared compression plus block-specific expansion."""
    def __init__(self, hidden_dim: int, value_dim: int, depth: int, rank: int = 128):
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.ModuleList(nn.Linear(rank, value_dim, bias=False) for _ in range(depth))
        self.gate_logit = nn.Parameter(torch.full((depth,), -4.59511985013459))

    def forward(self, block: int, mixed: Tensor, original: Tensor):
        # original [B,N,dV]; mixed [B,E,N,d]
        return original[:, None] + self.gate_logit[block].sigmoid() * self.up[block](self.down(mixed))


class ExpertValueAdapter(nn.Module):
    """Slot-specific nonlinear transform plus a block/slot residual scale."""
    def __init__(self, experts: int, value_dim: int, depth: int, rank: int = 32,
                 eps: float = 1e-6):
        super().__init__()
        self.down = nn.Parameter(torch.randn(experts, rank, value_dim) * .02)
        self.up = nn.Parameter(torch.randn(experts, value_dim, rank) * .02)
        self.norm_weight = nn.Parameter(torch.ones(depth, value_dim))
        self.gamma = nn.Parameter(torch.full((depth, experts), .01))
        self.eps = eps

    def forward(self, block: int, values: Tensor):
        # values [B,E,N,dV]
        work = values.to(normalization_dtype(values))
        normed = (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.eps)).to(values.dtype)
        normed = normed * self.norm_weight[block]
        low = torch.einsum("bend,erd->benr", normed, self.down)
        delta = torch.einsum("benr,edr->bend", F.silu(low), self.up)
        return delta + self.gamma[block][None, :, None, None] * values, delta
