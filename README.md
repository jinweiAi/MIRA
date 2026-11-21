# MIRA

**Latent-conditioned visual reading for Qwen2.5-VL.**

MIRA equips parallel latent slots with three connected operations: selecting
visual positions, choosing a layer-specific representation for each selected
position, and integrating that evidence through a slot-specific value adapter.
Both explicit routers use the current latent state and the question embedding.

This repository contains the core implementation, two-stage training, answer-free
inference, and evaluation entry points. Training runs, analysis notebooks, reports,
figures, benchmark outputs, datasets, and checkpoints are kept outside the repository.

## Method

```text
Image -> frozen visual encoder -> aligned V1-V32 visual bank
                                      |
Question + current latent state ------+-> spatial Top-K + selected-score prior
                                      |
                                      +-> per-position layer mixture
                                      |
                        low-rank evidence projection
                                      |
                       native visual Value + evidence
                                      |
                         slot-specific value adapter
                                      |
                 correction to that latent's attention output
```

The visual bank applies the frozen Qwen merger to each visual block's output and
restores the same spatial ordering. MIRA replaces the visual Value contribution
for latent query rows; it leaves the shared backbone K/V tensors intact. The
native Qwen attention, residual path, and MLP continue to update the latent states.

- **Spatial selection:** each latent reads
  `min(N, max(2, ceil(N / 2)))` visual positions. The selected-score log prior is
  retained in the attention computation and in the training graph.
- **Layer reading:** a separate soft mixture over all 32 visual layers is computed
  for every selected position and receiving latent.
- **Evidence writing:** a shared down-projection and block-specific up-projections
  inject evidence into native visual Values, followed by the receiving slot's
  adapter. Only the latent query output receives the resulting correction.
- **Parallel slots:** the eight latent slots have independent learned initial
  vectors. They do not attend to one another.
- **Two-stage training:** Stage I prevents prompt/readout/answer tokens from
  directly reading visual tokens, establishing the latent path. Stage II restores
  the native causal visual path and retains MIRA. Both stages use answer/EOS
  cross-entropy, without auxiliary routing or diversity losses.

## Installation

The integration targets **Python 3.10, PyTorch 2.6.0, Transformers 4.51.3,
PEFT 0.15.2**, and Qwen2.5-VL-7B-Instruct. It uses eager attention and supports
one unpadded example per forward pass, with gradient accumulation for training.

```bash
git clone https://github.com/jinweiAi/MIRA.git
cd MIRA
python -m venv .venv
source .venv/bin/activate
# Install the PyTorch 2.6.0 / torchvision 0.21.0 wheels for your CUDA environment.
pip install -e ".[test,eval]"
pytest -q
```

The backbone is loaded from `Qwen/Qwen2.5-VL-7B-Instruct` or a local model
directory. Base-model weights and datasets must be obtained separately.

## Data interface

Provide JSONL with one example per line. Relative image paths are resolved against
the manifest directory. The question must already include the benchmark's exact
prompt, options, and answer-format instruction; the code does not silently rewrite it.

```json
{"id": "example-001", "images": ["images/example.jpg"], "question": "What fruit is shown?\nA. Apple\nB. Banana\nAnswer with the option letter.", "answer": "B", "options": [{"label": "A", "text": "Apple"}, {"label": "B", "text": "Banana"}], "category": "perception"}
```

`answer` is required for training and scoring, and is optional for inference.
Inference never reads it. `options` and `category` are used only for scoring.
Training follows the manifest order; shuffling/sampling should be fixed when
creating the manifest. The released recipe expects the same static 10,000-example
manifest in both stages. The original manifest and image files are not bundled;
using a different manifest is a new training run and does not guarantee the
paper's checkpoint or benchmark results.

```bash
mira-prepare-data --input data/raw_train.jsonl --output data/train.jsonl --training
```

## Training

The configuration files use values checked against the retained Full MIRA
checkpoint:

| Setting | Value |
| --- | --- |
| Trainable components | Decoder LoRA, latent initial vectors, routers, evidence projection, adapters |
| Frozen components | Original backbone weights, visual encoder/merger, visual bank, pooled original question embeddings |
| Latent slots / visual layers | 8 / 32 |
| Spatial reading ratio | 50%, with at least 2 positions when available |
| Router width / projection rank / adapter rank | 128 / 128 / 32 |
| Spatial and layer temperatures / prior scale | 1.0 / 1.0 / 1.0 |
| LoRA | rank 16, alpha 32, dropout 0.05, no bias |
| LoRA targets | q/k/v/o projections and gate/up/down MLP projections, decoder only |
| Optimizer | AdamW, learning rate 1e-4, weight decay 0.01, betas (0.9, 0.999), eps 1e-8 |
| Schedule | 5% warmup, cosine decay, gradient clipping 1.0 |
| Per-device microbatch / gradient accumulation | 1 / 4 |
| Image budget | max_pixels = 401408 |
| Precision | BF16 backbone and latents; FP32 added modules |
| Each stage | 1 pass over 10,000 rows = 2,500 optimizer updates |
| Stage transition | Transfer trainable weights; reset optimizer and scheduler |
| Final checkpoint | End of Stage II, with Stage-II inference topology |

```bash
CUDA_VISIBLE_DEVICES=0 mira-train --config configs/train/stage1.json
CUDA_VISIBLE_DEVICES=0 mira-train --config configs/train/stage2.json
```

Override paths explicitly when needed:

```bash
mira-train --config configs/train/stage1.json \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --manifest /path/to/train.jsonl --output-dir runs/stage1
mira-train --config configs/train/stage2.json \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --manifest /path/to/train.jsonl --output-dir runs/stage2 \
  --resume runs/stage1/final.pt --resume-mode stage_transfer
```

Use the same model and manifest path/settings across stages. To resume an
interrupted stage, pass `--resume runs/stage1/latest.pt --resume-mode exact`
with the matching stage configuration. Exact resume checks the manifest hash
and configuration and restores optimizer, scheduler, row cursor, and RNG state.
Intermediate checkpoints are written only at optimizer-step boundaries.
`gradient_checkpointing` is configurable; the reference configuration leaves it
disabled. Changing it is a memory-saving option, not a claim of bitwise
reproducibility.

## Inference

```bash
mira-infer --checkpoint runs/stage2/final.pt \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --manifest data/test.jsonl --output outputs/predictions.jsonl \
  --max-new-tokens 16
```

Generation is greedy (`do_sample=False`, one beam), always uses Stage II, and
recomputes the complete prefix at each answer token. This transparent
implementation does **not** use a dynamic-routing KV cache. Set the answer-token
budget to match the benchmark; 16 is a multiple-choice example, not a universal
evaluation setting. There is no fixed-total-length truncation that reduces the
answer budget when the latent prefix changes.

Each record includes the generated text and IDs, TTFT, end-to-end time, peak
allocated GPU memory, answer length, and truncation status. Timings include image
preprocessing, visual encoding, bank construction, latent computation and answer
generation; they exclude model loading and input JSONL reading. Peak allocated
memory includes the loaded model and per-example tensors, rather than CUDA
reserved memory. Hardware, image budget, and execution load must match for cost
comparisons.

The loader accepts the retained static Full MIRA checkpoint schema
(`schema_version=1`, MIRA weights, decoder LoRA, and latent table). Checkpoints
are not included in Git. Only load trusted checkpoints because this schema also
stores Python RNG/optimizer objects.

## Evaluation

For a prepared multiple-choice manifest:

```bash
mira-evaluate --manifest data/test.jsonl \
  --predictions outputs/predictions.jsonl --output outputs/scores.json
```

The scorer reports strict-letter and deterministic format-normalized accuracy,
plus category counts and a per-example parsing audit. Normalization accepts only
the generated text and option texts; it receives no gold labels. Unresolved
outputs remain incorrect in the full denominator. The normalized parser is the
frozen `mmstar_normalize_v4_20261105` implementation; keep raw and normalized
scores distinguishable when comparing methods.

The local scorer is **MCQ micro accuracy**, not a universal replacement for
official benchmark protocols (for example, paired MMVP scoring or MME subtotals).
Use the optional **VLMEvalKit** adapter for its native dataset preparation,
prompts, and evaluation:

```bash
git clone https://github.com/open-compass/VLMEvalKit.git /path/to/VLMEvalKit
# Install VLMEvalKit's dependencies in this environment following its README.
python scripts/eval_vlmeval.py --vlmeval-root /path/to/VLMEvalKit \
  --checkpoint runs/stage2/final.pt --model Qwen/Qwen2.5-VL-7B-Instruct \
  --data MMStar --work-dir outputs/vlmeval
```

The adapter preserves the image/text order supplied by VLMEvalKit. Select a
VLMEvalKit revision and benchmark protocol explicitly for comparisons. Its dataset
identifier for SEED-Bench-2-Plus is `SEEDBench2_Plus`. BLINK full-benchmark micro
accuracy must cover all 14 tasks; an average over selected categories must be
labeled as a subset result. The adapter is optional and not required for core
training or the local scorer.

## Code map

| Path | Responsibility |
| --- | --- |
| `mira/routing.py` | Latent-conditioned position router and per-position layer mixture |
| `mira/evidence.py` | Low-rank evidence projection and slot-specific value adapter |
| `mira/decoder.py` | Qwen decoder integration and latent-only output correction |
| `mira/visual_bank.py` | Frozen, aligned multi-layer visual features |
| `mira/prefix.py`, `mira/topology.py` | Prefix assembly, M-RoPE positions, two-stage masks |
| `mira/training.py`, `mira/checkpoint.py` | LoRA, training, state transfer and atomic saves |
| `mira/inference.py` | Answer-free greedy generation and cost records |
| `mira/evaluation.py`, `mira/answer_parser.py` | MCQ scoring and frozen output normalization |
| `mira/vlmeval.py` | Optional VLMEvalKit model adapter |
| `configs/train/` | The two reference training configurations |
| `scripts/` | Thin runnable entry points |
| `tests/` | CPU regression tests for the core contracts |

## Release scope and license

This is a code release. Model weights, the original training sample manifest,
datasets, and paper metadata are not included. CPU tests cover the implementation
contracts; they do not replace full-size training or benchmark reproduction.
No new benchmark result is claimed by this repository.

The code is licensed under Apache-2.0. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependencies and attribution.
