# DiffusionGemma

[← back to model index](README.md)

## Status snapshot

| Field | Status |
|---|---|
| GGUF architecture keys | `diffusion-gemma`, `diffusion_gemma` |
| Source class | [`DiffusionGemmaModel`](../../TensorSharp.Models/Models/DiffusionGemma/DiffusionGemmaModel.cs) |
| Sampler | [`DiffusionGemmaSampler`](../../TensorSharp.Models/Models/DiffusionGemma/DiffusionGemmaSampler.cs) |
| Modalities | Text only |
| Thinking / tools | Not supported |
| Generation mode | Block text diffusion, not autoregressive token decode |
| CLI support | `TensorSharp.Cli` detects `DiffusionGemmaModel` and uses diffusion run mode |
| Server support | Web UI chat stream with live denoising previews; Ollama/OpenAI compatibility paths remain autoregressive chat paths |
| Continuous batching | Dedicated [`DiffusionBatchScheduler`](../../TensorSharp.Server/DiffusionBatchScheduler.cs), admitted at block boundaries |

## 1. Origin and intent

DiffusionGemma is a block text-diffusion language model built on a Gemma-4-style
Mixture-of-Experts backbone. It is not the same runtime contract as the
autoregressive `gemma4` model:

- `Forward(int[] tokens)` intentionally throws. Generation must go through
  `DiffusionGemmaSampler`.
- Each denoising step runs over a concatenated `[prompt | canvas]` sequence.
- The prompt side is causal and never attends to the canvas.
- The canvas side is bidirectional over the prompt and canvas.
- The emitted block is the current deterministic argmax canvas, refined over
  multiple denoising steps.

The GGUF file must report `general.architecture=diffusion-gemma` or
`diffusion_gemma`; `ModelBase.Create()` routes those keys to
`DiffusionGemmaModel`.

## 2. Forward graph

The model exposes two execution regimes.

The unified correctness path is `ForwardCanvas(tokens, promptLen)`:

```text
[prompt tokens | canvas tokens]
  -> region-aware embedding scale
  -> prompt/canvas attention masks
  -> N Gemma-style transformer layers
       - local/global QK-norm attention
       - dense gated-GELU MLP
       - top-k MoE experts
       - prompt encoder scale / canvas decoder scale
  -> output norm
  -> tied lm-head
  -> final logit softcap
  -> canvas logits
```

The optimized GPU path splits each block into a prompt prefill plus repeated
canvas decodes:

1. `PrefillPrompt(promptTokens)` computes the prompt K/V once.
2. `DecodeCanvas(canvasTokens, scBuffer, scUse, prevTempInv)` reuses prompt K/V
   for every denoising step.
3. The sampler accepts low-entropy positions, re-noises the rest, and repeats.

Prompt-KV caching is enabled on device-glue backends and bypassed on the pure
CPU path.

## 3. Sampler contract

`DiffusionEbParams` controls generation:

| Parameter | Default | Meaning |
|---|---:|---|
| `MaxDenoisingSteps` | 48 | Maximum refinement steps per canvas block |
| `TMin` / `TMax` | 0.4 / 0.8 | Temperature schedule from late to early denoising |
| `EntropyBound` | 0.1 | Cumulative mutual-information bound for accepted positions |
| `StabilityThreshold` | 1 | How many stable argmax steps are required before early stop |
| `ConfidenceThreshold` | 0.005 | Mean entropy threshold for early stop |
| `Seed` | 0 | Deterministic sampler seed |
| `MaxBlocks` | 1 | Number of block-autoregressive canvas blocks |

The CLI maps this through:

```bash
./TensorSharp.Cli --model diffusion-gemma.gguf --input prompt.txt --backend ggml_metal \
  --max-tokens 256 --diffusion-steps 48 --diffusion-seed 0 --diffusion-blocks 1
```

When `--diffusion-blocks` is `0`, the CLI derives the number of blocks from
`--max-tokens` and `diffusion.canvas_length`.

## 4. Architecture details

DiffusionGemma reuses many Gemma-4 backbone choices:

- NeoX RoPE with separate local/global dimensions.
- Five local sliding-window layers followed by one global layer pattern.
- Per-head Q/K RMSNorm and unweighted V RMSNorm.
- Global layers can omit `attn_v.weight`, using raw K as V.
- Dense gated-GELU MLP plus 128-expert top-8 MoE.
- Tied embeddings / lm-head and final logit softcapping.

Diffusion-specific metadata includes:

| Key | Meaning |
|---|---|
| `diffusion.canvas_length` | Number of canvas positions denoised per block, default 256 |
| `tokenizer.ggml.mask_token_id` | Mask token id used by warmup and fallback paths |
| `<arch>.attention.sliding_window_pattern` | Local/global layer pattern |
| `<arch>.attention.head_count_kv` | Per-layer KV head counts |
| `<arch>.expert_count` / `<arch>.expert_used_count` | MoE expert count and active top-k |

## 5. Acceleration status

Current optimized paths include:

- Prompt-KV cache for GPU backends.
- Self-conditioning enabled by default; disable with `DIFFUSION_NO_SC=1`.
- GGML fused decode layer, fused whole-model decode, and fused lm-head tail.
- MLX K-quant affine repacking for DiffusionGemma's multi-row canvas workload.
- Block-boundary continuous batching in `TensorSharp.Server` through
  `DiffusionBatchScheduler`.

Important toggles:

| Variable | Effect |
|---|---|
| `DIFFUSION_STEPS` | Server-side denoising steps per block, default 48 |
| `DIFFUSION_MAX_BATCH` | Server diffusion scheduler max active requests, default 2 |
| `DIFFUSION_NO_PKV=1` | Disable prompt-KV caching on device-glue backends |
| `DIFFUSION_NO_SC=1` | Disable self-conditioning |
| `DIFFUSION_SC_TOPK` | Experimental self-conditioning top-K cutoff, default 32 |
| `DIFFUSION_BATCHED_FORWARD=1` | Use true batched canvas decode instead of time-sliced fused single-canvas decode |
| `DIFFUSION_NO_FUSED_DECODE=1` | Disable GGML fused whole-model diffusion decode |
| `DIFFUSION_NO_FUSED_LMHEAD_TAIL=1` | Disable fused output-norm + lm-head + softcap tail |
| `DIFFUSION_LMHEAD_BATCH_CAP_MB` | Cap transient batched lm-head logits memory, default 300 MB |
| `DIFFUSION_PROFILE=1` / `DIFFUSION_STEPTIME=1` / `DIFFUSION_FUSED_DEBUG=1` | Development timing and fused-kernel debug diagnostics |

## 6. Server behavior

When the Web UI hosts a DiffusionGemma GGUF:

- `/api/chat` takes the diffusion path.
- The stream emits `replace` events rather than token append events, because
  every denoising step refines the whole current canvas.
- A final replacement is emitted before the `done` event.
- Concurrent requests share one background diffusion scheduler and are admitted
  between blocks.

The Ollama and OpenAI compatibility adapters still use append-oriented response
shapes through `ChatStreamWithMetricsAsync`. They can surface the final
DiffusionGemma text, but the live denoising previews and `replace` frames are
Web UI-only.

## 7. Test coverage

[`DiffusionGemmaTests`](../../InferenceWeb.Tests/DiffusionGemmaTests.cs) is
opt-in on real GGUFs via `TS_TEST_MODEL_DIR`. It covers:

- `ForwardCanvas` finite-logit correctness.
- End-to-end EntropyBound generation.
- Prompt-KV equivalence and speed probes.
- Regression guards for repeated-token output and device-memory retention.
- Batched decode equivalence and two-request generation through the scheduler
  style used by the server.

## 8. Remaining work

- Add dedicated API examples once Ollama/OpenAI adapters grow a diffusion-aware
  compatibility surface.
- Promote true batched canvas decode only if it wins on target GPUs; today the
  fused single-canvas path can be faster when one canvas already saturates the
  GPU.
- Fold more diffusion scheduler metrics into `/api/queue/status` if operators
  need per-diffusion-batch visibility.
