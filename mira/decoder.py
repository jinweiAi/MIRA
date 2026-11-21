"""Qwen2.5-VL decoder integration; only latent query outputs receive evidence."""
from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from torch.utils.checkpoint import checkpoint
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb, repeat_kv,
)
from .routing import Routes, StateSpatialRouter, LayerRouter
from .evidence import EvidenceProjection, ExpertValueAdapter


@dataclass(frozen=True)
class MiraReadingContext:
    bank: Tensor             # frozen [1,N,M,d], same spatial order at all M layers
    visual_indices: Tensor   # [N], physical positions in the decoder sequence
    expert_indices: Tensor   # [E], physical positions of the parallel latent slots
    question: Tensor         # frozen [1,d], mean original question-token embeddings
    allowed: Tensor          # bool [T,T], causal Stage-I/II topology; True = allowed
    temperature: float = 1.0
    spatial_temperature: float = 1.0
    spatial_prior_scale: float = 1.0
    spatial_k: int | None = None


class MIRAStaticDecoder(nn.Module):
    """Evidence changes latent query outputs; shared K/V tensors stay intact."""
    def __init__(self, qwen_model: nn.Module, num_candidates: int = 32,
                 route_dim: int = 128, projection_rank: int = 128,
                 adapter_rank: int = 32, num_experts: int = 8):
        super().__init__()
        self.qwen_model = qwen_model
        cfg = qwen_model.config
        if cfg._attn_implementation != "eager" or cfg.attention_dropout != 0:
            raise ValueError("Use Qwen eager attention with attention_dropout=0.")
        self.num_candidates = num_candidates
        self.value_dim = cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads)
        self.layer_router = LayerRouter(cfg.hidden_size, num_candidates, route_dim)
        self.spatial_router = StateSpatialRouter(cfg.hidden_size, route_dim)
        self.evidence = EvidenceProjection(cfg.hidden_size, self.value_dim,
                                           len(qwen_model.layers), projection_rank)
        self.esva = ExpertValueAdapter(num_experts, self.value_dim,
                                      len(qwen_model.layers), rank=adapter_rank)

    def _layer(self, hidden: Tensor, compressed: Tensor, *, layer: nn.Module,
               block: int, ctx: MiraReadingContext, vi: Tensor, ei: Tensor,
               bias: Tensor, position_embeddings):
        # 1. Route from BLOCK-INPUT latent states to visual positions.
        z = hidden.index_select(1, ei).to(self.layer_router.query.weight.dtype)
        k_count = ctx.spatial_k or min(vi.numel(), max(2, (vi.numel() + 1) // 2))
        routes = self.spatial_router(
            z, ctx.question.to(z.dtype), ctx.bank.to(z.dtype),
            torch.ones((1, vi.numel()), dtype=torch.bool, device=hidden.device),
            k_count, temperature=ctx.spatial_temperature)
        routes = Routes(routes.edge, routes.prior * ctx.spatial_prior_scale,
                        routes.logits, routes.probabilities)
        selected = routes.edge.nonzero(as_tuple=True)[-1].reshape(1, ei.numel(), k_count)
        bias = bias.clone()
        spatial_bias = routes.prior[0].to(hidden.dtype).masked_fill(~routes.edge[0], -torch.inf)
        bias[0, 0, ei[:, None], vi[None, :]] = spatial_bias

        # 2. Soft layer selection for EACH selected position and receiving latent.
        _, pi, _ = self.layer_router(
            z, ctx.question.to(z.dtype),
            ctx.bank.to(self.layer_router.key.weight.dtype),
            state_conditioned=True, temperature=ctx.temperature,
            position_indices=selected, materialize=False)
        selected_low = compressed[:, None].expand(-1, ei.numel(), -1, -1, -1).gather(
            2, selected[..., None, None].expand(-1, -1, -1,
                                               compressed.shape[-2], compressed.shape[-1]))
        mixed_low = torch.einsum("bekm,bekmr->bekr", pi, selected_low)
        delta = self.evidence.up[block](mixed_low)
        delta = delta * self.evidence.gate_logit[block].sigmoid()

        # 3. Original Qwen Q/K/V and attention, with the latent-to-visual mask/prior.
        residual = hidden
        normed = layer.input_layernorm(hidden)
        attn = layer.self_attn
        b, t, _ = hidden.shape
        q = attn.q_proj(normed).view(b, t, -1, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(normed).view(b, t, -1, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(normed).view(b, t, -1, attn.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin,
                                             attn.rope_scaling["mrope_section"])
        kr = repeat_kv(k, attn.num_key_value_groups)
        vr = repeat_kv(v, attn.num_key_value_groups)
        scores = torch.matmul(q, kr.transpose(2, 3)) / math.sqrt(attn.head_dim)
        scores = scores + bias[..., :kr.shape[-2]]
        if q.dtype == torch.float16:
            scores = torch.where(torch.isinf(scores), torch.zeros_like(scores), scores)
        alpha = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(alpha, vr).transpose(1, 2).contiguous().reshape(b, t, -1)

        # 4. Slot-specific Value transformation after adding routed evidence.
        visual_alpha = alpha.index_select(2, ei).index_select(3, vi)
        visual_alpha = visual_alpha.gather(3, selected[:, None].expand(-1, alpha.shape[1], -1, -1))
        count = pi.shape[2]
        visual_alpha = visual_alpha.permute(0, 2, 1, 3).reshape(
            b, ei.numel(), attn.num_key_value_heads, attn.num_key_value_groups, count)
        original_visual = v.transpose(1, 2).index_select(1, vi)
        original_visual = original_visual[:, None].expand(-1, ei.numel(), -1, -1, -1)
        original_visual = original_visual.gather(
            2, selected[..., None, None].expand(-1, -1, -1,
                                               attn.num_key_value_heads, attn.head_dim))
        adapter_dtype = self.esva.down.dtype
        injected = original_visual.to(adapter_dtype) + delta.to(adapter_dtype).reshape(
            b, ei.numel(), count, attn.num_key_value_heads, attn.head_dim)
        adapted, _ = self.esva(block, injected.reshape(b, ei.numel(), count, self.value_dim))
        adapted = adapted.reshape_as(injected)
        difference = (delta.to(adapter_dtype).reshape_as(injected)
                      + (adapted - injected)).to(v.dtype)

        # 5. Same alpha weights aggregate the replacement-minus-original Value.
        #    Add only to latent query rows, then apply native o_proj/residual/MLP.
        correction = torch.einsum("behgn,benhd->behgd", visual_alpha, difference).reshape(
            b, ei.numel(), attn.num_heads * attn.head_dim)
        output = output.index_add(1, ei, correction)
        hidden = residual + attn.o_proj(output)
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return hidden

    def forward(self, inputs_embeds: Tensor, position_ids: Tensor,
                context: MiraReadingContext, *, gradient_checkpointing=False) -> Tensor:
        # Prefix sequence: original image/text context + 8 learned latents
        # + assistant header + teacher-forced/generated answer prefix.
        hidden = inputs_embeds
        ctx = context
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("MIRA supports one unpadded example per forward pass.")
        vi, ei = ctx.visual_indices, ctx.expert_indices
        if vi.ndim != 1 or ei.ndim != 1 or vi.dtype != torch.long or ei.dtype != torch.long:
            raise ValueError("Indices must be one-dimensional torch.long tensors.")
        if not vi.numel() or ei.numel() != self.esva.down.shape[0]:
            raise ValueError("Nonempty visual positions and matching latent count required.")
        if vi.unique().numel() != vi.numel() or ei.unique().numel() != ei.numel() or torch.isin(vi, ei).any():
            raise ValueError("Visual/latent indices must be unique and disjoint.")
        if ctx.bank.shape != (1, vi.numel(), self.num_candidates, hidden.shape[-1]):
            raise ValueError("bank must be [1,N,M,d].")
        if ctx.bank.requires_grad or ctx.question.requires_grad:
            raise ValueError("bank and original-question embeddings must be detached.")
        if ctx.question.shape != (1, hidden.shape[-1]):
            raise ValueError("question must be [1,d].")
        if position_ids.shape != (3, 1, hidden.shape[1]):
            raise ValueError("Use explicit Qwen M-RoPE position_ids [3,1,T].")
        allowed = ctx.allowed
        if allowed.dtype != torch.bool or allowed.shape != (hidden.shape[1], hidden.shape[1]):
            raise ValueError("allowed must be boolean [T,T].")
        if not allowed.any(-1).all() or allowed.triu(1).any():
            raise ValueError("Topology must be causal with no all-masked rows.")
        if not allowed[ei[:, None], vi[None, :]].all():
            raise ValueError("All candidate latent-to-visual edges must exist before Top-K.")
        if ctx.spatial_k is not None and not 1 <= ctx.spatial_k <= vi.numel():
            raise ValueError("spatial_k must be between 1 and N.")
        bias = hidden.new_zeros(allowed.shape).masked_fill(~allowed, -torch.inf)[None, None]
        position_embeddings = self.qwen_model.rotary_emb(hidden, position_ids)
        compressed = self.evidence.down(ctx.bank.to(self.layer_router.key.weight.dtype))
        for block, layer in enumerate(self.qwen_model.layers):
            fn = partial(self._layer, layer=layer, block=block, ctx=ctx, vi=vi, ei=ei,
                         bias=bias, position_embeddings=position_embeddings)
            hidden = (checkpoint(fn, hidden, compressed, use_reentrant=False)
                      if gradient_checkpointing and self.training else fn(hidden, compressed))
        return self.qwen_model.norm(hidden)
