"""Current-state spatial routing and per-position visual-layer mixing."""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F

@dataclass
class Routes:
    edge: Tensor       # bool [B,E,N]
    prior: Tensor      # [B,E,N], zero outside selected edges; mask separately
    logits: Tensor
    probabilities: Tensor


def normalization_dtype(x: Tensor):
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def selected_log_prior(logits: Tensor, edge: Tensor, scale: float = 1.0,
                       eps: float = 0.0) -> Tensor:
    """Normalize over final axis, with a data-dependent selected count.

    Every row must contain at least one selected element. Logits outside the
    selected set have zero local gradient. With exactly one selection, all
    normalized scores are constants, so the routing prior has zero gradient.
    """
    counts = edge.sum(-1, keepdim=True)
    if torch.any(counts == 0):
        raise ValueError("Every routing row must have a nonempty selected set")
    if eps < 0:
        raise ValueError("eps must be nonnegative")
    work = logits.to(normalization_dtype(logits)).masked_fill(~edge, -torch.inf)
    logp = F.log_softmax(work, dim=-1)
    log_prior = counts.log() + logp
    if eps:
        log_prior = torch.logaddexp(log_prior, log_prior.new_tensor(math.log(eps)))
    return torch.where(edge, scale * log_prior, torch.zeros_like(logp))


def route_expert_to_position(logits: Tensor, valid: Tensor, k: int | Tensor,
                             scale: float = 1.0) -> Routes:
    """Proposed direction: logits [B,E,N], k positions for each expert.

    valid [B,N]. k is scalar or [B], allowing different image sizes per sample.
    Set k=valid.sum(-1) for a dense warm-up. Padded positions are never selected.
    """
    b, e, n = logits.shape
    if valid.shape != (b, n):
        raise ValueError("valid shape mismatch")
    count = valid.sum(-1)
    k = torch.as_tensor(k, device=logits.device).expand(b)
    if torch.any(k < 1) or torch.any(k > count):
        raise ValueError("k must be between 1 and each sample's valid length")
    ranked = logits.masked_fill(~valid[:, None, :], -torch.inf).argsort(-1, descending=True)
    chosen_in_sorted_order = (torch.arange(n, device=logits.device)[None, None, :]
                              < k[:, None, None]).expand(b, e, n)
    edge = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, ranked, chosen_in_sorted_order)
    prior = selected_log_prior(logits, edge, scale)
    probs = logits.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
    return Routes(edge, prior, logits, probs)


class StateSpatialRouter(nn.Module):
    def __init__(self, hidden_dim: int, route_dim: int = 128):
        super().__init__()
        self.query = nn.Linear(2 * hidden_dim, route_dim, bias=False)
        self.key = nn.Linear(hidden_dim, route_dim, bias=False)
        self.route_dim = route_dim

    def forward(self, z: Tensor, question: Tensor, bank: Tensor,
                valid: Tensor, k: int | Tensor, temperature: float = 1.0):
        # z [B,E,d]; question [B,d]; bank [B,N,M,d]
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        d = z.shape[-1]
        descriptor = F.layer_norm(bank.mean(-2), (d,))
        condition = torch.cat((F.layer_norm(z, (d,)),
                               question[:, None].expand_as(z)), dim=-1)
        score = torch.einsum("ber,bnr->ben", self.query(condition), self.key(descriptor))
        score = score / (math.sqrt(self.route_dim) * temperature)
        return route_expert_to_position(score, valid, k)


class LayerRouter(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, route_dim: int = 128):
        super().__init__()
        self.query = nn.Linear(2 * hidden_dim, route_dim, bias=False)
        self.key = nn.Linear(hidden_dim, route_dim, bias=False)
        self.layer_identity = nn.Parameter(torch.randn(num_layers, route_dim) * .02)
        self.route_dim = route_dim

    def forward(self, z: Tensor, question: Tensor, bank: Tensor,
                state_conditioned: bool = True, temperature: float = 1.0,
                position_indices: Tensor | None = None, materialize: bool = True):
        # Reference keeps the original full mixture by default. The decoder
        # can gather projected keys and mix compressed Values without creating
        # an unused [B,E,N,d] hidden-width intermediate.
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        d = z.shape[-1]
        state = F.layer_norm(z, (d,)) if state_conditioned else torch.zeros_like(z)
        condition = torch.cat((state, question[:, None].expand_as(z)), -1)
        q = self.query(condition)
        key = self.key(bank) + self.layer_identity
        if position_indices is None:
            logits = torch.einsum("ber,bnmr->benm", q, key)
        else:
            b, e, _ = z.shape
            if (position_indices.ndim != 3 or position_indices.shape[:2] != (b, e)
                    or position_indices.dtype != torch.long
                    or position_indices.device != bank.device
                    or position_indices.shape[-1] == 0
                    or bool((position_indices < 0).any() or (position_indices >= bank.shape[1]).any())):
                raise ValueError("position_indices must be valid long [B,E,K]")
            selected_key = key[:, None].expand(-1, e, -1, -1, -1).gather(
                2, position_indices[..., None, None].expand(-1, -1, -1, key.shape[-2], key.shape[-1]))
            logits = torch.einsum("ber,bekmr->bekm", q, selected_key)
        logits = logits / (math.sqrt(self.route_dim) * temperature)
        weights = logits.to(normalization_dtype(logits)).softmax(-1).to(bank.dtype)
        mixed = None
        if materialize:
            if position_indices is None:
                mixed = torch.einsum("benm,bnmd->bend", weights, bank)
            else:
                selected_bank = bank[:, None].expand(-1, z.shape[1], -1, -1, -1).gather(
                    2, position_indices[..., None, None].expand(-1, -1, -1, bank.shape[-2], d))
                mixed = torch.einsum("bekm,bekmd->bekd", weights, selected_bank)
        return mixed, weights, logits
