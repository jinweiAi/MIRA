from dataclasses import replace
import json
import random
import numpy as np
import pytest
import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
from mira.topology import build_access_mask
from mira.prefix import ChatSegments, assemble_latent_prefix
from mira.routing import route_expert_to_position, LayerRouter
from mira.visual_bank import extract_visual_bank
from mira.training import (MiraTrainConfig, build_training_components, make_optimizer,
    build_scheduler, forward_prefix, save_checkpoint, load_checkpoint)
from mira.losses import answer_cross_entropy
from mira.evaluation import score
from mira.inference import append_token
from mira.data import read_manifest

def tiny_model():
    vision = Qwen2_5_VLVisionConfig(depth=2, hidden_size=32, intermediate_size=64,
        num_heads=4, in_channels=3, patch_size=2, spatial_merge_size=2,
        temporal_patch_size=2, window_size=8, out_hidden_size=32,
        fullatt_block_indexes=[1])
    config = Qwen2_5_VLConfig(vocab_size=200, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        vision_config=vision.to_dict(), image_token_id=90, video_token_id=91,
        vision_start_token_id=92, vision_end_token_id=93,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
        attention_dropout=0.0)
    config._attn_implementation = "eager"
    return Qwen2_5_VLForConditionalGeneration(config).eval()

def components():
    torch.manual_seed(41)
    model = tiny_model()
    cfg = MiraTrainConfig(model="tiny", manifest="tiny.jsonl", device="cpu", precision="fp32",
        lora_rank=2, lora_alpha=4, lora_dropout=0.0, route_dim=8,
        projection_rank=8, esva_rank=4, num_experts=4, layer_indices=(0, 1),
        num_candidates=2, save_every_steps=0)
    decoder, latents, params, _ = build_training_components(model, cfg)
    decoder.train()
    return model, decoder, latents, params, cfg

def prefix_for(model, latents, *, answer=True):
    bank = torch.randn(4, 2, 32)
    seg = ChatSegments(torch.tensor([92, 90, 90, 90, 90, 93, 20, 21]),
                       torch.tensor([30, 31]), torch.tensor([[1, 4, 4]]), 90)
    prefix = assemble_latent_prefix(model, seg, bank[:, -1], latents.value,
        torch.tensor([40, 99] if answer else [], dtype=torch.long),
        torch.tensor([0, 0, 0, 0, 0, 0, 1, 1]), 10)
    return prefix, bank[None]

@pytest.mark.parametrize("stage", [1, 2])
def test_parallel_topology(stage):
    groups = ("P", "V", "V", "P", "E", "E", "R", "A")
    mask = build_access_mask(groups, stage)
    assert not mask.triu(1).any() and mask.any(-1).all()
    assert mask[4, 1:3].all() and mask[5, 1:3].all()
    assert not mask[5, 4]  # Independent parallel latent slots.
    assert mask[5, 5] and mask[7, 4:6].all()
    assert bool(mask[7, 1]) == (stage == 2)
    assert bool(mask[3, 1]) == (stage == 2)

def test_selected_prior_has_task_gradient_including_dense_case():
    for k in [2, 4]:
        logits = torch.randn(1, 3, 4, requires_grad=True)
        routes = route_expert_to_position(logits, torch.ones(1, 4, dtype=torch.bool), k)
        assert (routes.edge.sum(-1) == k).all()
        (routes.prior * torch.arange(4.)).sum().backward()
        assert logits.grad.abs().sum() > 0
        assert torch.count_nonzero(logits.grad[~routes.edge]) == 0
        assert torch.count_nonzero(routes.prior[~routes.edge]) == 0

def test_layer_mixture_normalized_per_position():
    router = LayerRouter(16, 4, 8)
    bank = torch.randn(1, 5, 4, 16)
    z = torch.randn(1, 3, 16, requires_grad=True)
    idx = torch.tensor([[[0, 3], [1, 4], [0, 2]]])
    mixed, pi, _ = router(z, torch.randn(1, 16), bank, position_indices=idx)
    assert mixed.shape == (1, 3, 2, 16)
    torch.testing.assert_close(pi.sum(-1), torch.ones(1, 3, 2))
    mixed.square().sum().backward()
    assert z.grad.abs().sum() > 0

@pytest.mark.parametrize("stage", [1, 2])
def test_tiny_answer_ce_reaches_all_core_modules(stage):
    model, decoder, latents, params, cfg = components()
    prefix, bank = prefix_for(model, latents)
    logits = forward_prefix(model, decoder, prefix, bank, stage=stage)
    loss = answer_cross_entropy(logits, prefix.labels)
    loss.backward()
    assert torch.isfinite(loss)
    for group in [decoder.spatial_router, decoder.layer_router, decoder.evidence, decoder.esva, latents]:
        grads = [p.grad for p in group.parameters() if p.grad is not None]
        assert grads and sum(float(g.abs().sum()) for g in grads) > 0
    assert any(p.grad is not None for n, p in model.named_parameters() if "lora_" in n)
    assert all(p.grad is None for n, p in model.named_parameters() if "lora_" not in n)
    assert len({id(p) for p in params}) == len(params)

def test_gradient_checkpointing_preserves_output_and_gradients():
    model, decoder, latents, params, cfg = components()
    prefix, bank = prefix_for(model, latents)
    a = forward_prefix(model, decoder, prefix, bank, stage=1)
    answer_cross_entropy(a, prefix.labels).backward()
    grads = [None if p.grad is None else p.grad.clone() for p in params]
    for p in params:
        p.grad = None
    prefix2, _ = prefix_for(model, latents)
    prefix2 = replace(prefix2, inputs_embeds=prefix.inputs_embeds.detach().clone().index_copy(
        1, prefix.expert_indices, latents.value[None]))
    b = forward_prefix(model, decoder, prefix2, bank, stage=1, gradient_checkpointing=True)
    answer_cross_entropy(b, prefix2.labels).backward()
    torch.testing.assert_close(a, b)
    for expected, p in zip(grads, params):
        if expected is not None:
            torch.testing.assert_close(expected, p.grad)

def test_checkpoint_roundtrip_and_stage_transfer(tmp_path):
    model, decoder, latents, params, cfg = components()
    opt = make_optimizer(params, cfg)
    sched = build_scheduler(opt, 4, cfg.warmup_ratio)
    prefix, bank = prefix_for(model, latents)
    answer_cross_entropy(forward_prefix(model, decoder, prefix, bank, stage=1), prefix.labels).backward()
    opt.step(); sched.step(); opt.zero_grad()
    weights = [p.detach().clone() for p in params]
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, decoder=decoder, latents=latents, optimizer=opt, scheduler=sched,
        config=cfg, epoch=0, global_step=1, row_index=4, manifest_sha256="abc", model_revision="tiny")
    draws = (random.random(), np.random.rand(), torch.rand(2))
    with torch.no_grad():
        for p in params:
            p.add_(1)
    state = load_checkpoint(path, decoder=decoder, latents=latents, optimizer=opt, scheduler=sched,
        manifest_sha256="abc", model_revision="tiny", config=cfg)
    assert state["global_step"] == 1
    assert random.random() == draws[0] and np.random.rand() == draws[1]
    torch.testing.assert_close(torch.rand(2), draws[2])
    for expected, p in zip(weights, params):
        torch.testing.assert_close(expected, p)
    newopt = make_optimizer(params, cfg)
    newsched = build_scheduler(newopt, 4, cfg.warmup_ratio)
    result = load_checkpoint(path, decoder=decoder, latents=latents, optimizer=newopt,
        scheduler=newsched, manifest_sha256="abc", model_revision="tiny",
        config=replace(cfg, stage=2, resume_mode="stage_transfer"), resume_mode="stage_transfer")
    assert result["global_step"] == 0 and result["optimizer_reset"]
    assert len(newopt.state) == 0
    with pytest.raises(ValueError, match="manifest"):
        load_checkpoint(path, decoder=decoder, latents=latents, optimizer=opt, scheduler=sched,
            manifest_sha256="changed", model_revision="tiny", config=cfg)

def test_visual_bank_last_layer_and_hook_cleanup():
    model = tiny_model()
    pixels, grid = torch.randn(16, 24), torch.tensor([[1, 4, 4]])
    original, bank, lengths = extract_visual_bank(model.visual, pixels, grid, (0, 1))
    assert bank.shape == (4, 2, 32) and lengths.tolist() == [4]
    torch.testing.assert_close(bank[:, -1], original)
    assert not bank.requires_grad
    assert all(not b._forward_hooks for b in model.visual.blocks)

def test_answer_free_prefix_and_generation_positions():
    model, decoder, latents, _, _ = components()
    prefix, bank = prefix_for(model, latents, answer=False)
    assert "A" not in prefix.group_ids and prefix.answer_indices.numel() == 0
    updated = append_token(model, prefix, 40)
    assert updated.group_ids[-1] == "A"
    assert not updated.question_mask[0, -1] and updated.labels[0, -1] == -100
    torch.testing.assert_close(updated.position_ids[:, :, -1], prefix.position_ids[:, :, -1] + 1)
    logits = forward_prefix(model, decoder, updated, bank, stage=2, last_only=True)
    assert logits.shape == (1, 1, 200)

def test_manifest_validation_and_unparsed_denominator(tmp_path):
    path = tmp_path / "eval.jsonl"
    rows = [{"id": "1", "images": ["x.png"], "question": "Question?", "answer": "B",
             "options": [{"label": "A", "text": "Apple"}, {"label": "B", "text": "Banana"}]},
            {"id": "2", "images": ["x.png"], "question": "Question?", "answer": "A",
             "options": [{"label": "A", "text": "Apple"}, {"label": "B", "text": "Banana"}]}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    loaded = read_manifest(path, check_images=False)
    assert loaded[0]["images"][0] == str(tmp_path / "x.png")
    result, audit = score(loaded, [{"id": "1", "prediction": "The answer is B."},
                                 {"id": "2", "prediction": "I am unsure."}])
    assert result["micro"]["n"] == 2
    assert result["micro"]["normalized_accuracy"] == 50.0
    assert result["micro"]["unparsed"] == 1
    with pytest.raises(ValueError, match="coverage"):
        score(loaded, [{"id": "1", "prediction": "B"}])
    path.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest(path, check_images=False)
