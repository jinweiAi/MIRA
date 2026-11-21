"""Answer-free, deterministic full-prefix autoregressive decoding."""
from __future__ import annotations
import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import time
import torch
from .data import read_manifest
from .training import (MiraTrainConfig, _restore_states, build_training_components,
                       forward_prefix, load_qwen_model, prepare_row, validate_config)

def load_mira(checkpoint, model=None, device="cuda:0", max_pixels=None):
    # Only load checkpoints from a trusted source: schema v1 includes Python RNG state.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    keys = {f.name for f in fields(MiraTrainConfig)}
    cfg = {k: v for k, v in payload["config"].items() if k in keys}
    cfg.update(model=model or cfg["model"], device=device, gradient_checkpointing=False)
    if max_pixels is not None:
        cfg["max_pixels"] = max_pixels
    if "layer_indices" in cfg:
        cfg["layer_indices"] = tuple(cfg["layer_indices"])
    config = MiraTrainConfig(**cfg)
    validate_config(config)
    backbone, processor = load_qwen_model(config)
    decoder, latents, _, _ = build_training_components(backbone, config)
    _restore_states(decoder, latents, payload)
    backbone.eval()
    decoder.eval()
    latents.eval()
    return backbone, processor, decoder, latents, config

def append_token(model, prefix, token):
    new = prefix.input_ids.new_tensor([[token]])
    length = prefix.input_ids.shape[1]
    return replace(prefix,
        input_ids=torch.cat((prefix.input_ids, new), dim=1),
        inputs_embeds=torch.cat((prefix.inputs_embeds,
            model.get_input_embeddings()(new).to(prefix.inputs_embeds.dtype)), dim=1),
        labels=torch.cat((prefix.labels, prefix.labels.new_full((1, 1), -100)), dim=1),
        attention_mask=torch.cat((prefix.attention_mask, prefix.attention_mask.new_ones((1, 1))), dim=1),
        position_ids=torch.cat((prefix.position_ids, prefix.position_ids[:, :, -1:] + 1), dim=2),
        question_mask=torch.cat((prefix.question_mask, prefix.question_mask.new_zeros((1, 1))), dim=1),
        group_ids=prefix.group_ids + ("A",),
        answer_indices=torch.cat((prefix.answer_indices,
                                  prefix.answer_indices.new_tensor([length]))))

@torch.inference_mode()
def generate(model, processor, decoder, latents, config, row, max_new_tokens=16):
    if isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    device = torch.device(config.device)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    prefix, bank = prepare_row(model, processor, row, latents, device=device,
        max_pixels=config.max_pixels, layer_indices=config.layer_indices, include_answer=False)
    if prefix.answer_indices.numel() or "A" in prefix.group_ids:
        raise AssertionError("Generation prefix must contain no answer tokens")
    stop = set()
    for value in (processor.tokenizer.eos_token_id, model.generation_config.eos_token_id):
        stop.update(value if isinstance(value, list) else [] if value is None else [value])
    tokens, first_token_seconds = [], None
    for _ in range(max_new_tokens):
        logits = forward_prefix(model, decoder, prefix, bank, stage=2, last_only=True)
        token = int(logits[0, -1].argmax())
        tokens.append(token)
        if first_token_seconds is None:
            sync()
            first_token_seconds = time.monotonic() - started
        if token in stop:
            break
        prefix = append_token(model, prefix, token)
    sync()
    ended = tokens[-1] in stop
    return {
        "prediction": processor.tokenizer.decode(tokens, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False),
        "tokens": tokens, "answer_tokens": len(tokens), "ended_eos": ended,
        "truncated": not ended and len(tokens) == max_new_tokens,
        "ttft_seconds": first_token_seconds, "seconds": time.monotonic() - started,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30
                           if device.type == "cuda" else None,
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", help="Local Qwen directory or Hugging Face model ID")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--max-pixels", type=int)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {output}")
    rows = read_manifest(args.manifest, limit=args.limit)
    model, processor, decoder, latents, config = load_mira(
        args.checkpoint, args.model, args.device, args.max_pixels)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", buffering=1) as handle:
        for i, row in enumerate(rows):
            result = generate(model, processor, decoder, latents, config, row, args.max_new_tokens)
            result["id"] = row["id"]
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(json.dumps({"completed": i + 1, "total": len(rows), "id": row["id"],
                              "prediction": result["prediction"]}), flush=True)
    output.with_suffix(output.suffix + ".config.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "model": config.model, "stage": 2,
        "max_new_tokens": args.max_new_tokens, "max_pixels": config.max_pixels,
        "do_sample": False, "num_beams": 1, "kv_cache": False,
        "timing_includes": ["image preprocessing", "visual encoding", "visual bank",
                            "latent prefix", "answer generation"],
    }, indent=2) + "\n")
if __name__ == "__main__":
    main()
