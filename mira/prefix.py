"""Parallel latent insertion, answer labels, and Qwen multimodal positions.

The processor image-token layout and assistant header are preserved.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ChatSegments:
    """Tokenized context and assistant header from one Qwen chat prompt."""

    context_input_ids: Tensor
    header_input_ids: Tensor
    image_grid_thw: Tensor
    image_token_id: int


@dataclass
class LatentPrefix:
    """The decoder inputs and routing metadata for one unpadded example."""

    input_ids: Tensor
    inputs_embeds: Tensor
    labels: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    mrope_position_deltas: Tensor
    group_ids: tuple[str, ...]
    visual_indices: Tensor
    latent_indices: Tensor
    expert_indices: Tensor
    summary_indices: Tensor
    readout_indices: Tensor
    answer_indices: Tensor
    question_mask: Tensor
    image_grid_thw: Tensor


def tokenize_chat_segments(processor, messages, images) -> ChatSegments:
    """Tokenize context and the original assistant header without duplicating it.

    ``messages`` must contain the user image/text content.  The context prompt
    ends after the user's turn; the full prompt adds Qwen's assistant header.
    The function verifies that the full token sequence is a strict extension of
    the context sequence, so latent insertion cannot silently move a token.
    """
    context_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    context = processor(
        text=[context_text], images=images, padding=True, return_tensors="pt"
    )
    full = processor(
        text=[full_text], images=images, padding=True, return_tensors="pt"
    )
    context_ids = context["input_ids"][0]
    full_ids = full["input_ids"][0]
    if full_ids.shape[0] <= context_ids.shape[0]:
        raise AssertionError("assistant generation header was not added")
    if not torch.equal(full_ids[: context_ids.shape[0]], context_ids):
        raise AssertionError("chat template header is not a token-level suffix")
    if not torch.equal(context["image_grid_thw"], full["image_grid_thw"]):
        raise AssertionError("context/full image grids differ")
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is None or image_token_id < 0:
        raise AssertionError("processor tokenizer has no <|image_pad|> token")
    return ChatSegments(
        context_input_ids=context_ids,
        header_input_ids=full_ids[context_ids.shape[0] :],
        image_grid_thw=full["image_grid_thw"],
        image_token_id=int(image_token_id),
    )


def _one_dim(name: str, value: Tensor, dtype: torch.dtype | None = None) -> Tensor:
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1:
        raise ValueError(f"{name} must be a rank-1 tensor, got {tuple(value.shape)}")
    if dtype is not None:
        value = value.to(dtype=dtype)
    return value


def question_mask_from_text(tokenizer, context_input_ids: Tensor, question_text: str) -> Tensor:
    """Locate a unique exact question token span for the single-turn input.

    Fail closed if tokenization merges across boundaries or the question occurs
    more than once. This conservative helper is not a general multi-turn data
    collator: that collator will need offsets from the rendered chat template.
    """
    context = _one_dim("context_input_ids", context_input_ids, torch.long)
    ids = tokenizer(question_text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("question text must not be empty")
    if set(ids) & set(tokenizer.all_special_ids):
        raise ValueError("question text must not contain special tokens")
    needle = context.new_tensor(ids)
    starts = []
    if needle.numel() <= context.numel():
        windows = context.unfold(0, needle.numel(), 1)
        starts = torch.where((windows == needle).all(-1))[0].tolist()
    if len(starts) != 1:
        raise ValueError("question must match one exact context token span")
    mask = torch.zeros_like(context, dtype=torch.bool)
    mask[starts[0] : starts[0] + len(ids)] = True
    return mask


def assemble_latent_prefix(
    model,
    segments: ChatSegments,
    visual_embeds: Tensor,
    latent_embeds: Tensor,
    answer_ids: Tensor,
    question_mask: Tensor,
    latent_placeholder_id: int,
) -> LatentPrefix:
    """Build one decoder example with frozen visual and trainable latent inputs.

    Args:
        model: ``Qwen2_5_VLForConditionalGeneration`` or a compatible tiny model.
        segments: context/header token IDs from :func:`tokenize_chat_segments`.
        visual_embeds: flattened merged visual embeddings, ``[N, d]``.
        latent_embeds: independent trainable visual expert embeddings, ``[E, d]``.
        answer_ids: answer token IDs including the desired EOS token.
        question_mask: boolean mask over *context* tokens only; answer/header,
            image tokens and latents are excluded by construction.
        latent_placeholder_id: ordinary text token used only for logical IDs
            and position calculation; its embedding is replaced by latents.

    The initial implementation intentionally supports one unpadded example.
    Padding and ragged bank packing are added after this sequence-level contract
    is tested against the real Qwen processor.
    """
    context_ids = _one_dim("context_input_ids", segments.context_input_ids, torch.long)
    header_ids = _one_dim("header_input_ids", segments.header_input_ids, torch.long)
    answers = _one_dim("answer_ids", answer_ids, torch.long)
    question = _one_dim("question_mask", question_mask).bool()
    if question.shape != context_ids.shape:
        raise ValueError("question_mask must cover context_input_ids exactly")
    if question[(context_ids == segments.image_token_id)].any():
        raise ValueError("question_mask cannot include image tokens")
    if latent_embeds.ndim != 2:
        raise ValueError("latent_embeds must have shape [num_latents, hidden]")
    if visual_embeds.ndim != 2:
        raise ValueError("visual_embeds must have shape [num_visual_tokens, hidden]")
    image_positions = torch.where(context_ids == segments.image_token_id)[0]
    if image_positions.numel() != visual_embeds.shape[0]:
        raise ValueError(
            f"image token count {image_positions.numel()} != visual rows "
            f"{visual_embeds.shape[0]}"
        )
    if latent_embeds.shape[0] < 2:
        raise ValueError("latent_embeds must contain at least two visual experts")
    if header_ids.numel() == 0:
        raise ValueError("assistant readout header cannot be empty")

    config = model.config
    forbidden = {
        int(config.image_token_id),
        int(config.video_token_id),
        int(config.vision_start_token_id),
        int(config.vision_end_token_id),
    }
    if int(latent_placeholder_id) in forbidden:
        raise ValueError("latent placeholder cannot be a vision special token")
    if any(question[context_ids == token].any() for token in forbidden):
        raise ValueError("question_mask cannot include vision special tokens")
    grid = segments.image_grid_thw
    merge_size = config.vision_config.spatial_merge_size
    if grid.ndim != 2 or grid.shape[1] != 3 or not (grid > 0).all():
        raise ValueError("image_grid_thw must have positive shape [images, 3]")
    if (grid[:, 1:] % merge_size).any():
        raise ValueError("image grid must be divisible by spatial_merge_size")
    if int((grid.prod(-1) // merge_size**2).sum()) != visual_embeds.shape[0]:
        raise ValueError("image grid token count does not match visual rows")

    device = latent_embeds.device
    hidden = latent_embeds.shape[-1]
    embedding = model.get_input_embeddings()
    if embedding.weight.shape[-1] != hidden:
        raise ValueError("latent hidden size does not match decoder embeddings")
    if visual_embeds.shape[-1] != hidden:
        raise ValueError("visual hidden size does not match decoder embeddings")
    context_ids = context_ids.to(device=device)
    header_ids = header_ids.to(device=device)
    answers = answers.to(device=device)
    visual_embeds = visual_embeds.to(device=device, dtype=latent_embeds.dtype)
    image_positions = image_positions.to(device=device)
    context_embeds = embedding(context_ids[None])[0].to(dtype=latent_embeds.dtype)
    context_embeds = context_embeds.clone()
    context_embeds[image_positions] = visual_embeds

    num_experts = latent_embeds.shape[0]
    latent_ids = torch.full(
        (latent_embeds.shape[0],),
        int(latent_placeholder_id),
        device=device,
        dtype=torch.long,
    )
    suffix_ids = torch.cat((header_ids, answers))
    suffix_embeds = embedding(suffix_ids[None])[0].to(dtype=latent_embeds.dtype)
    input_ids = torch.cat((context_ids, latent_ids, suffix_ids))
    inputs_embeds = torch.cat((context_embeds, latent_embeds, suffix_embeds), dim=0)

    context_len = context_ids.shape[0]
    latent_len = latent_embeds.shape[0]
    header_len = header_ids.shape[0]
    answer_len = answers.shape[0]
    latent_start = context_len
    header_start = latent_start + latent_len
    answer_start = header_start + header_len
    labels = torch.full_like(input_ids, -100)
    labels[answer_start:] = answers
    attention_mask = torch.ones_like(input_ids)
    position_ids, mrope_deltas = model.get_rope_index(
        input_ids=input_ids[None],
        image_grid_thw=segments.image_grid_thw.to(device=device),
        attention_mask=attention_mask[None],
    )
    groups = tuple(
        ("V" if int(token) == segments.image_token_id else "P")
        for token in context_ids.tolist()
    ) + ("E",) * num_experts + ("R",) * header_len + ("A",) * answer_len
    if len(groups) != input_ids.shape[0]:
        raise AssertionError("group/input length mismatch")
    all_indices = torch.arange(input_ids.shape[0], device=device)
    visual_indices = all_indices[:context_len][context_ids == segments.image_token_id]
    latent_indices = all_indices[latent_start:header_start]
    expert_indices = latent_indices
    summary_indices = latent_indices.new_empty((0,))
    readout_indices = all_indices[header_start:answer_start]
    answer_indices = all_indices[answer_start:]
    # Keep the question mask aligned with the complete decoder sequence.  The
    # question lives only in the original context; inserted latents, the
    # assistant readout header, and answer tokens must not enter question
    # pooling or any question-conditioned router.
    question_mask_full = torch.zeros_like(input_ids, dtype=torch.bool)
    question_mask_full[:context_len] = question.to(device=device)
    return LatentPrefix(
        input_ids=input_ids[None],
        inputs_embeds=inputs_embeds[None],
        labels=labels[None],
        attention_mask=attention_mask[None],
        position_ids=position_ids,
        mrope_position_deltas=mrope_deltas,
        group_ids=groups,
        visual_indices=visual_indices,
        latent_indices=latent_indices,
        expert_indices=expert_indices,
        summary_indices=summary_indices,
        readout_indices=readout_indices,
        answer_indices=answer_indices,
        question_mask=question_mask_full[None],
        image_grid_thw=segments.image_grid_thw.to(device=device),
    )
