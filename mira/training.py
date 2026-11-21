"""Two-stage Full MIRA training and legacy-compatible trainable checkpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, inject_adapter_in_model

from .checkpoint import atomic_torch_save, append_json_record
from .topology import build_access_mask
from .losses import answer_cross_entropy
from .decoder import MiraReadingContext, MIRAStaticDecoder
from .prefix import (
    LatentPrefix,
    assemble_latent_prefix,
    question_mask_from_text,
    tokenize_chat_segments,
)
from .visual_bank import extract_visual_bank


from .data import read_manifest

@dataclass
class MiraTrainConfig:
    model: str
    manifest: str
    output_dir: str = "runs/stage1"
    stage: int = 1
    epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation: int = 4
    max_pixels: int = 401408
    spatial_routing: bool = True
    esva: bool = True
    esva_state_conditioned: bool = False
    gradient_checkpointing: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    route_dim: int = 128
    projection_rank: int = 128
    esva_rank: int = 32
    num_candidates: int = 32
    num_experts: int = 8
    layer_indices: tuple[int, ...] = tuple(range(32))
    latent_std: float = 0.02
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    seed: int = 20261016
    device: str = "cuda:0"
    precision: str = "bf16"
    resume: str | None = None
    resume_mode: str = "exact"
    save_every_steps: int = 250
    limit: int | None = None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_decoder_lora(model: nn.Module, config: MiraTrainConfig) -> list[str]:
    decoder = getattr(model, "model", None)
    if decoder is None:
        raise TypeError("Qwen model must expose model decoder")
    suffixes = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    inject_adapter_in_model(
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=suffixes,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        decoder,
        adapter_name="default",
        low_cpu_mem_usage=False,
    )
    names = [name for name, parameter in decoder.named_parameters() if "lora_" in name]
    matched = {name.split(".lora_", 1)[0].split(".")[-1] for name in names}
    if not names or matched != set(suffixes):
        raise AssertionError(f"unexpected LoRA targets: {sorted(matched)}")
    if any(name.startswith(("visual.", "lm_head", "embed_tokens")) for name in names):
        raise AssertionError("LoRA entered a forbidden module")
    return names


class SharedLatents(nn.Module):
    """One persistent visual/summary latent table shared by every manifest row."""

    def __init__(self, hidden_size: int, count: int, std: float, *, device, dtype):
        super().__init__()
        if count < 2 or count % 2:
            raise ValueError("latent count must be an even integer >= 2")
        self.value = nn.Parameter(torch.randn(count, hidden_size, device=device, dtype=dtype) * std)

    def forward(self) -> Tensor:
        return self.value


def freeze_and_collect_trainable_parameters(
    model: nn.Module, decoder: MIRAStaticDecoder, latents: SharedLatents
):
    """Freeze original Qwen weights and return one deduplicated optimizer registry."""
    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.requires_grad_(True)
    for name, parameter in decoder.named_parameters():
        parameter.requires_grad_(not name.startswith("qwen_model.") or "lora_" in name)
    latents.value.requires_grad_(True)
    parameters, names, seen = [], [], set()
    for name, parameter in decoder.named_parameters():
        if parameter.requires_grad and id(parameter) not in seen:
            seen.add(id(parameter))
            parameters.append(parameter)
            names.append(name)
    if id(latents.value) not in seen:
        parameters.append(latents.value)
        names.append("shared_latents.value")
    forbidden = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (name.startswith("visual.") or name.startswith("lm_head") or name.startswith("model.embed_tokens"))
    ]
    if forbidden:
        raise AssertionError(f"forbidden trainable base parameters: {forbidden[:5]}")
    if not any("lora_" in name for name in names):
        raise AssertionError("LoRA is absent from optimizer registry")
    if not any(name.startswith("layer_router.") for name in names):
        raise AssertionError("layer router is absent from optimizer registry")
    return parameters, {
        "count": sum(parameter.numel() for parameter in parameters),
        "parameters": names,
        "lora_count": sum(
            parameter.numel()
            for name, parameter in decoder.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ),
        "shared_latent_count": latents.value.numel(),
    }


def build_training_components(model: nn.Module, config: MiraTrainConfig):
    apply_decoder_lora(model, config)
    device = torch.device(config.device)
    decoder = MIRAStaticDecoder(
        model.model,
        num_candidates=len(config.layer_indices),
        route_dim=config.route_dim,
        projection_rank=config.projection_rank,
        adapter_rank=config.esva_rank,
        num_experts=config.num_experts,
    ).to(device)
    latents = SharedLatents(
        model.config.hidden_size,
        config.num_experts,
        config.latent_std,
        device=device,
        dtype=model.get_input_embeddings().weight.dtype,
    )
    parameters, report = freeze_and_collect_trainable_parameters(model, decoder, latents)
    return decoder, latents, parameters, report


def make_optimizer(parameters: list[nn.Parameter], config: MiraTrainConfig) -> AdamW:
    return AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.999), eps=1e-8)


def build_scheduler(optimizer: Optimizer, total_steps: int, warmup_ratio: float) -> LambdaLR:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_ratio <= 1:
        raise ValueError("warmup_ratio must be in [0, 1]")
    warmup = int(round(total_steps * warmup_ratio))

    def rule(step: int) -> float:
        if warmup and step < warmup:
            return max(1e-8, (step + 1) / warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1, max(0, progress))))

    return LambdaLR(optimizer, rule)


def _states(decoder, latents):
    mira, lora = {}, {}
    for name, parameter in decoder.named_parameters():
        if not parameter.requires_grad:
            continue
        key = name[len("qwen_model.") :] if name.startswith("qwen_model.") else name
        (lora if "lora_" in name else mira)[key] = parameter.detach().cpu().clone()
    return mira, lora, latents.value.detach().cpu().clone()


def _restore_states(decoder, latents, payload):
    mira, lora = payload["mira_state_dict"], payload["lora_state_dict"]
    found_mira, found_lora = set(), set()
    with torch.no_grad():
        for name, parameter in decoder.named_parameters():
            if not parameter.requires_grad:
                continue
            key = name[len("qwen_model.") :] if name.startswith("qwen_model.") else name
            source = lora if "lora_" in name else mira
            if key not in source:
                raise KeyError(f"checkpoint missing {key}")
            if parameter.shape != source[key].shape:
                raise ValueError(f"checkpoint shape mismatch: {key}")
            parameter.copy_(source[key].to(parameter.device, parameter.dtype))
            (found_lora if "lora_" in name else found_mira).add(key)
        if set(mira) != found_mira or set(lora) != found_lora:
            raise KeyError("checkpoint has unexpected trainable parameters")
        if latents.value.shape != payload["latent"].shape:
            raise ValueError("checkpoint latent shape mismatch")
        latents.value.copy_(payload["latent"].to(latents.value.device, latents.value.dtype))


def _rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path,
    *,
    decoder,
    latents,
    optimizer,
    scheduler,
    config,
    epoch,
    global_step,
    row_index,
    manifest_sha256,
    model_revision,
):
    mira, lora, latent = _states(decoder, latents)
    payload = {
        "schema_version": 1,
        "mira_state_dict": mira,
        "lora_state_dict": lora,
        "latent": latent,
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "row_index": row_index,
        "rng": _rng_state(),
        "manifest_sha256": manifest_sha256,
        "model_revision": model_revision,
        "config": asdict(config),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    atomic_torch_save(payload, path)


def load_checkpoint(
    path,
    *,
    decoder,
    latents,
    optimizer,
    scheduler,
    manifest_sha256,
    model_revision,
    config: MiraTrainConfig | None = None,
    resume_mode: str = "exact",
):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema")
    if payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("checkpoint manifest hash mismatch")
    if payload.get("model_revision") != model_revision:
        raise ValueError("checkpoint model revision mismatch")
    if resume_mode not in ("exact", "stage_transfer"):
        raise ValueError("resume_mode must be exact or stage_transfer")
    if config is not None:
        saved_config = dict(payload.get("config", {}))
        current_config = asdict(config)
        for key in ("output_dir", "resume", "resume_mode"):
            saved_config.pop(key, None)
            current_config.pop(key, None)
        if resume_mode == "stage_transfer":
            if saved_config.get("stage") != 1 or current_config.get("stage") != 2:
                raise ValueError("stage_transfer only supports checkpoint stage 1 to current stage 2")
            saved_config.pop("stage", None)
            current_config.pop("stage", None)
            # Stage 2 may intentionally use a longer continuation budget than
            # Stage 1. The optimizer is reset for transfer, so the epoch count
            # is a current-run setting rather than a checkpoint invariant.
            saved_config.pop("epochs", None)
            current_config.pop("epochs", None)
        if saved_config != current_config:
            differing = sorted(
                key for key in set(saved_config) | set(current_config)
                if saved_config.get(key) != current_config.get(key)
            )
            raise ValueError(f"checkpoint training config mismatch: {differing}")
    _restore_states(decoder, latents, payload)
    if resume_mode == "exact":
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        _restore_rng(payload["rng"])
    else:
        return {"epoch": 0, "global_step": 0, "row_index": 0,
                "parent_epoch": payload["epoch"],
                "parent_global_step": payload["global_step"],
                "parent_row_index": payload["row_index"],
                "optimizer_reset": True}
    return {key: payload[key] for key in ("epoch", "global_step", "row_index")}


def answer_token_ids(tokenizer, answer: str) -> list[int]:
    ids = list(tokenizer(str(answer), add_special_tokens=False)["input_ids"])
    if tokenizer.eos_token_id is not None and (not ids or ids[-1] != tokenizer.eos_token_id):
        ids.append(int(tokenizer.eos_token_id))
    if not ids:
        raise ValueError("empty answer tokenization")
    return ids


def prepare_row(model, processor, row, latents, *, device, max_pixels, layer_indices,
                include_answer=True):
    """Encode image(s), then insert parallel latents before the assistant header.

    include_answer=False never reads row['answer']; evaluation labels cannot
    enter tokenization, question pooling, the bank, or the decoder prefix.
    """
    processor.image_processor.max_pixels = max_pixels
    images = []
    for path in row["images"]:
        with Image.open(path) as opened:
            images.append(opened.convert("RGB"))
    content = row.get("_content") or ([{"type": "image", "image": "placeholder"} for _ in images]
        + [{"type": "text", "text": row["question"]}])
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs = processor(text=[prompt], images=images, padding=True, return_tensors="pt")
    dtype = next(model.visual.parameters()).dtype
    pixels = image_inputs["pixel_values"].to(device=device, dtype=dtype)
    grid = image_inputs["image_grid_thw"].to(device)
    _, bank, _ = extract_visual_bank(model.visual, pixels, grid, layer_indices)
    segments = tokenize_chat_segments(processor, messages, images)
    texts = [item["text"] for item in content if item["type"] == "text" and item["text"].strip()]
    qmask = torch.zeros_like(segments.context_input_ids, dtype=torch.bool)
    for text in texts:
        qmask |= question_mask_from_text(processor.tokenizer, segments.context_input_ids, text)
    placeholder = processor.tokenizer("latent", add_special_tokens=False)["input_ids"]
    if len(placeholder) != 1:
        raise ValueError("latent placeholder must tokenize as exactly one token")
    answers = answer_token_ids(processor.tokenizer, row["answer"]) if include_answer else []
    prefix = assemble_latent_prefix(
        model=model, segments=segments, visual_embeds=bank[:, -1],
        latent_embeds=latents.value,
        answer_ids=torch.tensor(answers, device=device, dtype=torch.long),
        question_mask=qmask, latent_placeholder_id=int(placeholder[0]))
    return prefix, bank.unsqueeze(0).detach()


def forward_prefix(model, decoder, prefix, bank, *, stage, gradient_checkpointing=False,
                   last_only=False):
    if bank.ndim == 3:
        bank = bank.unsqueeze(0)
    qvalues = prefix.inputs_embeds[0][prefix.question_mask[0]]
    if not qvalues.numel():
        raise ValueError("question mask is empty")
    context = MiraReadingContext(
        bank=bank.detach(), visual_indices=prefix.visual_indices,
        expert_indices=prefix.expert_indices, question=qvalues.mean(0, keepdim=True).detach(),
        allowed=build_access_mask(prefix.group_ids, stage, device=prefix.inputs_embeds.device))
    hidden = decoder(prefix.inputs_embeds, prefix.position_ids, context,
                     gradient_checkpointing=gradient_checkpointing)
    return model.lm_head(hidden[:, -1:] if last_only else hidden)


def load_qwen_model(config):
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    dtype = torch.bfloat16 if config.precision == "bf16" and device.type == "cuda" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model, torch_dtype=dtype, device_map={"": str(device)},
        attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(config.model, use_fast=False)
    processor.image_processor.max_pixels = config.max_pixels
    return model, processor


def validate_config(config):
    if not config.spatial_routing or not config.esva or config.esva_state_conditioned:
        raise ValueError("This release implements Full MIRA: spatial routing and static ESVA enabled")
    if config.stage not in (1, 2) or config.epochs < 1 or config.gradient_accumulation < 1:
        raise ValueError("Invalid stage, epoch count or gradient accumulation")
    if config.lr <= 0 or config.weight_decay < 0 or config.max_pixels < 1:
        raise ValueError("Invalid optimizer or image budget")
    if config.precision not in ("bf16", "fp32") or not 0 <= config.warmup_ratio <= 1:
        raise ValueError("Invalid precision or warmup ratio")
    if config.save_every_steps < 0 or config.max_grad_norm <= 0:
        raise ValueError("Invalid save interval or gradient clipping")
    if (tuple(config.layer_indices) != tuple(range(32)) or config.num_candidates != 32):
        raise ValueError("Full MIRA uses the ordered V1-V32 visual bank")
    if config.num_experts < 2 or config.num_experts % 2:
        raise ValueError("num_experts must be an even integer >= 2")


def run_training(model, processor, rows, decoder, latents, optimizer, scheduler, *,
                 config, manifest_sha256, model_revision,
                 start_epoch=0, start_row_index=0, start_step=0):
    device = torch.device(config.device)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    global_step = start_step
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    for epoch in range(start_epoch, config.epochs):
        model.eval()
        decoder.train()
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        first = start_row_index if epoch == start_epoch else 0
        for index in range(first, len(rows)):
            started = time.monotonic()
            prefix, bank = prepare_row(model, processor, rows[index], latents,
                device=device, max_pixels=config.max_pixels, layer_indices=config.layer_indices)
            logits = forward_prefix(model, decoder, prefix, bank, stage=config.stage,
                                    gradient_checkpointing=config.gradient_checkpointing)
            loss = answer_cross_entropy(logits, prefix.labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite answer loss at row {index}")
            (loss / config.gradient_accumulation).backward()
            pending += 1
            did_step = pending == config.gradient_accumulation or index == len(rows) - 1
            if did_step:
                if pending < config.gradient_accumulation:
                    for parameter in parameters:
                        if parameter.grad is not None:
                            parameter.grad.mul_(config.gradient_accumulation / pending)
                torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                global_step += 1
                if config.save_every_steps and global_step % config.save_every_steps == 0:
                    save_checkpoint(output / "latest.pt", decoder=decoder, latents=latents,
                        optimizer=optimizer, scheduler=scheduler, config=config, epoch=epoch,
                        global_step=global_step, row_index=index + 1,
                        manifest_sha256=manifest_sha256, model_revision=model_revision)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            record = {"id": rows[index]["id"], "epoch": epoch, "global_step": global_step,
                      "loss": float(loss.detach().float()), "seconds": time.monotonic() - started}
            append_json_record(record, output / "metrics.jsonl")
            if did_step and (global_step == 1 or global_step % 25 == 0):
                print(json.dumps(record), flush=True)
            del prefix, bank, logits, loss
        start_row_index = 0
    save_checkpoint(output / "final.pt", decoder=decoder, latents=latents,
        optimizer=optimizer, scheduler=scheduler, config=config, epoch=config.epochs,
        global_step=global_step, row_index=len(rows), manifest_sha256=manifest_sha256,
        model_revision=model_revision)
    return global_step


def main():
    import argparse
    from dataclasses import fields
    p = argparse.ArgumentParser(description="Train Full MIRA with answer/EOS supervision")
    p.add_argument("--config", required=True)
    for key in ("model", "manifest", "output-dir", "resume", "device"):
        p.add_argument("--" + key)
    p.add_argument("--resume-mode", choices=("exact", "stage_transfer"))
    args = p.parse_args()
    values = json.loads(Path(args.config).read_text())
    known = {f.name for f in fields(MiraTrainConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    values.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})
    if "layer_indices" in values:
        values["layer_indices"] = tuple(values["layer_indices"])
    config = MiraTrainConfig(**values)
    validate_config(config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    rows = read_manifest(config.manifest, training=True, limit=config.limit)
    model, processor = load_qwen_model(config)
    decoder, latents, parameters, report = build_training_components(model, config)
    optimizer = make_optimizer(parameters, config)
    total_steps = math.ceil(len(rows) / config.gradient_accumulation) * config.epochs
    scheduler = build_scheduler(optimizer, total_steps, config.warmup_ratio)
    manifest_hash = sha256_file(config.manifest)
    model_revision = Path(config.model).name
    start = {"epoch": 0, "row_index": 0, "global_step": 0}
    if config.resume:
        start = load_checkpoint(config.resume, decoder=decoder, latents=latents,
            optimizer=optimizer, scheduler=scheduler, manifest_sha256=manifest_hash,
            model_revision=model_revision, config=config, resume_mode=config.resume_mode)
    elif config.resume_mode == "stage_transfer" or config.stage == 2:
        raise ValueError("Stage II requires --resume stage1/final.pt --resume-mode stage_transfer")
    if not config.resume or config.resume_mode == "stage_transfer":
        training_seed = config.seed + 10000 * config.stage
        random.seed(training_seed)
        np.random.seed(training_seed)
        torch.manual_seed(training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(training_seed)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps({
        "config": asdict(config), "manifest_sha256": manifest_hash,
        "model_revision": model_revision, "row_count": len(rows),
        "optimizer_updates_this_stage": total_steps, "trainable": report,
    }, indent=2) + "\n")
    steps = run_training(model, processor, rows, decoder, latents, optimizer, scheduler,
        config=config, manifest_sha256=manifest_hash, model_revision=model_revision,
        start_epoch=int(start["epoch"]), start_row_index=int(start["row_index"]),
        start_step=int(start["global_step"]))
    print(json.dumps({"status": "complete", "updates": steps, "checkpoint": str(output / "final.pt")}))
if __name__ == "__main__":
    main()
