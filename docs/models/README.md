# Model Architecture Cards

[English](README.md) | [中文](README_zh-cn.md)

This folder is the canonical, per-model reference for every architecture that
TensorSharp can run. Each card is a self-contained brief: it walks an engineer
or researcher from "I have never heard of this model" all the way to "I can
explain the forward graph and reproduce the inference path in TensorSharp." If
you only need a top-level pointer, use the table below; otherwise jump into the
individual cards.

## What every card contains

Each card follows the same shape so you can diff architectures cleanly:

1. **Origin and intent** — who designed the model, what the GGUF arch keys are,
   and which capabilities (modalities, thinking, tools) it exposes.
2. **Model architecture** — the high-level block diagram, layer counts, and any
   per-layer heterogeneity.
3. **Forward graph** — the exact ordered list of ops a single token (decode), a
   multi-token sequence (prefill), or a diffusion denoising step flows through,
   including residuals and normalizations.
4. **Components** — every sub-block (attention, FFN/SSM, routing, normalization,
   RoPE flavor, vision/audio encoder) explained in detail with the math that
   governs it.
5. **Parameters and settings** — the GGUF metadata keys, weight tensor naming
   convention, and dtype expectations.
6. **TensorSharp implementation** — pointers to the C# source files, the
   instantiation order, the cache layout, and the way the model plugs into
   `ModelBase` / `Ops` / native GGML kernels.
7. **Prefill optimization** — chunking, fused per-layer kernels, parallelization,
   cross-layer caches.
8. **Decode optimization** — fused single-call kernels, pre-resolved weight
   pointers, batched MoE, in-place kernels, cache reuse.
9. **Memory and KV cache strategy** — circular vs. linear caches, mmap-backed
   weights, pre-allocated decode buffers.
10. **Multimodal pipeline** — how images / audio / video are processed,
    encoded, and injected into the language model.
11. **Output / chat template** — protocol parser, stop tokens, thinking / tool
    formats.
12. **Optimization opportunities** — work that has not been done yet but that
    we know would unlock more performance or capability.

## Verified start lane

The verified native GGML family/path tier is Gemma 4 E4B Q8_0; the
recommended public artifact is
[ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF).
Run it on `ggml_cuda`, `ggml_metal`, or `ggml_vulkan`; this lane exercises
fused native kernels. See the
[Gemma 4 card](gemma4.md#verified-gemma-4-e4b-native-ggml-fast-path).
Its matching `mmproj` is optional for text and required for image, video, or
audio input.

For a continuous learning path through that example—from tensor foundations to
a complete multimodal inference engine—use Zhongkai Fu's
[From Tensors to Tokens book guide](../BOOK.md), or
[view the paperback on Amazon](https://www.amazon.com/dp/B0H9P44QZZ).

## Implementation matrix

| Architecture | Card | Verified download (HF) | Source class | GGUF keys | Modalities | Reasoning | Tools | Batched / paged forward | Notable acceleration |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | [deepseek41.md](deepseek41.md) | [vcruz305/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF/tree/8e0c4de3cb6519bfc11ed69dc87184b457a57bb5), seven Q2_K shards plus prepared Engram sidecar; [validation status](../deepseek41_validation.md) | `DeepSeek4Model` with a dedicated native V4.1 graph, plus the pure-C# `DeepSeek4CpuExecutor`, which implements that same V4.1 graph for `--backend cpu` | `deepseek41` | Text; image and video through the separately prepared vision companion (`--mmproj`; it is native, so it needs a ggml backend — `LoadVisionEncoder` throws on `--backend cpu`), audio rejected with HTTP 400 | Reference-format renderer | Spaced DSML; model-level validation tracked separately | Per-sequence slots; per-slot decode fallback | `ggml_cuda`, layer placement, CPU MoE, and experimental routed-MoE TP; attention/distributed TP and V4.1 DSpark remain unimplemented. `--backend ggml_cpu` (the same native graph on scalar ggml kernels) and `--backend cpu` (the pure-C# executor, with no ggml, no native library and no GPU) are correctness and portability paths, not serving paths |
| DeepSeek V4 Flash | [deepseek4.md](deepseek4.md) | [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) (multi-shard per quant directory; point `--model` at the `-00001-of-` shard). DSpark drafters: [MODEL_DOWNLOADS.md](../../MODEL_DOWNLOADS.md#dspark-drafters) | `DeepSeek4Model` (+ `DeepSeek4CudaExecutor`, `DeepSeek4CpuExecutor`) | `deepseek4` | Text | Yes | Yes (DSML markup) | Native per-sequence slots (`DeepSeek4Model.PerSeqCache.cs`) rather than `IBatchedPagedModel` — servable with continuous batching through the same engine | Three whole-model executors (direct CUDA, native ggml, pure C#), automatic layer split across every visible GPU, on-device compressed KV state (SWA ring + CSA/HCA + lightning indexer), shape-signature graph cache replaying a captured CUDA graph, fused decode index-gather over `[ring \| top-512]` K, and DSpark block speculative decoding (1.3–1.4× decode) |
| Qwen 3.8 Flash Next | [qwen38-flash-next.md](qwen38-flash-next.md) | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (multi-shard; point `--model` at the `-00001-of-` shard) | `Qwen4ExpModel` (whole-token fused graph on GGML) | `qwen4exp` | Text + image | Yes | Yes | Per-sequence state holders (`SupportsPerSequenceFusedForward`): per-request KV + GDN + PLE state, round-robin fused decode | One captured graph per token incl. in-graph PLE and fused LM head; IMRoPE vision; KV reuse across turns (extend-only); multi-GPU **layer split** across every visible GPU (`--tp N`: contiguous whole layers per GPU — a capacity feature, not tensor parallelism, since `qwen4exp` shards no weights; byte-identical output, `TS_Q4E_LAYER_SPLIT` overrides the balance) |
| GLM 5.x | [glm.md](glm.md) | [unsloth/GLM-5.2-GGUF](https://huggingface.co/unsloth/GLM-5.2-GGUF), [unsloth/GLM-5.3-GGUF](https://huggingface.co/unsloth/GLM-5.3-GGUF) (UD-Q2_K_XL is seven shards, 236.4 GiB), [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) (multi-shard per quant directory; point `--model` at the `-00001-of-` shard) | `GlmDsaModel` (+ the native `ggml_ops_glm_dsa.cpp` whole-model executor); GLM-5.3 (the same 79-block `glm-dsa` shape as 5.2, 256 experts, text only) loads on the GLM-5.2 path with no new code and no new flag — see the [card](glm.md#glm-53-glm-dsa) | `glm-dsa` (GLM-5.2 and GLM-5.3), `glm5next` (5.3-Flash) — one architecture descriptor serves all three, Flash vs non-Flash being a single boolean on `general.architecture` | Text (5.2 and 5.3 — the 5.3 repo publishes no mmproj at any quant, and `LoadVisionEncoder` warns and ignores an `--mmproj` on `glm-dsa` instead of failing the run); text + image (5.3-Flash via `mmproj`) | Yes | Yes (XML tool calls) | Native per-sequence slots (`TSGgml_GlmSlotAlloc`) rather than `IBatchedPagedModel` — servable with continuous batching through the same engine | Native whole-model ggml executor and a pure-C# per-op reference (GLM 5.2, GLM-5.3 and GLM-5.3-Flash all run on `--backend cpu`, but it is a reference implementation to A/B against rather than bit-parity: on GLM-5.3-Flash the prefill-logit cosine is 0.9567 against `ggml_cpu` and the greedy token differs — see the [card](glm.md#glm-53-flash-on---backend-cpu)), automatic layer split across every visible GPU **or** Megatron tensor parallelism (`--tp N`: column/row-parallel heads, every routed expert split row-wise), `--cpu-moe` host-resident experts served straight from the GGUF mapping, MLA weight absorption with a 576-wide cache row, DSA lightning indexer with a selection reused across 57 of 78 layers, and a shape-keyed graph cache replaying a captured CUDA graph; on GLM-5.3 the `blk.78` NextN block drafts with `--spec` on the default layer split (no `--tp`), because it ships no LM head of its own and borrows the trunk LM head, which `--tp` splits column-wise |
| Gemma 4 | [gemma4.md](gemma4.md) | E4B Q8_0 is the verified native-GGML family/path tier; [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) is the recommended public artifact | `Gemma4Model` | `gemma4` (`gemma4-assistant` / `gemma4_assistant` load only as the MTP draft) | Text, image, video, audio | Yes | Yes | **Default** (toggle off with `TS_GEMMA4_BATCHED=0`) | Single-graph fused decode (all layers in one GGML dispatch), fused whole-model prefill/verify with in-kernel PLE + shared-KV handling, chunked prefill, circular SWA cache, and MoE variants. Batched path matches legacy logits within FP noise (`Gemma4BatchedForwardTests`); reaches ~1.5× legacy at batch=8 and ~1.6× at 4×800-token prompts. |
| DiffusionGemma | [diffusiongemma.md](diffusiongemma.md) | [unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) | `DiffusionGemmaModel` + `DiffusionGemmaSampler` | `diffusion-gemma`, `diffusion_gemma` | Text | No | No | Separate Web UI `DiffusionBatchScheduler`; not an autoregressive `IBatchedPagedModel` path | EntropyBound block denoising over `[prompt \| canvas]`, prompt-KV caching on GPU backends, self-conditioning, fused GGML whole-model diffusion decode and fused lm-head tail |
| Qwen-Image-Edit | [qwenimage.md](qwenimage.md) | [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF) (DiT; VAE / text-encoder companions in the card) | `QwenImageModel` (+ `QwenImagePipeline`) | `qwen_image`, `qwen-image` | Image edit (image+text → image) | No | No | None — `Forward()` throws; editing runs through `EditImage()` and edits are serialized | 60-block MMDiT diffusion (FlowMatch-Euler, true-CFG, reference-latent concat), CUDA-graph-captured whole-DiT forward (~2.9x per forward), optional Lightning distillation LoRA as a runtime side-path (`--qwen-image-lora`: 60 DiT forwards -> 4-8), default flash attention, CFG-batching, opt-in EasyCache / First-Block-Cache denoise caches, fused Qwen2.5-VL conditioning encoders and fused whole-VAE graph, VRAM-aware area clamp |
| Qwen 3.5 / 3.6 family | [qwen35.md](qwen35.md) | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF); NextN MTP: [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) (base-repo Qwen3.6 GGUFs strip the NextN block and silently fall back to standard decode) | `Qwen35Model` | `qwen35`, `qwen35moe`, `qwen3next` | Text, image | Yes | Yes | **Default** (toggle off with `TS_QWEN35_BATCHED=0` or `--no-continuous-batching`). Per-slot recurrent-state pool + optional native GatedDeltaNet kernel (`TS_QWEN35_BATCHED_GDN_NATIVE=1`) | Hybrid FullAttention + GatedDeltaNet recurrent, fused attention layer decode, fused prefill attention, fused output-projection + FFN, fused output-projection + norm + router, batched MoE (routed + shared + residual in a single kernel), fused vision encoder blocks |
| Bonsai Q1_0 | [bonsai.md](bonsai.md) | Local sideload only: exact `Bonsai-8B-Q1_0.gguf` and `Bonsai-27B-Q1_0.gguf` byte lengths and SHA-256 pins are in the card; their GGUFs declare no publisher URL or license | `Qwen3Model` (8B), `Qwen35Model` (27B) | `qwen3`, `qwen35` | Text | 27B yes; 8B fixed empty think block | Yes | Qwen 3 paged interface when cache geometry permits; Qwen 3.5 per-slot KV + recurrent state by default | 8B whole-model native prefill and persistent Metal decode; 27B whole-model hybrid prefill/verify and persistent decode, both with a measured 512-token Metal chunk |
| GPT OSS | [gptoss.md](gptoss.md) | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) | `GptOssModel` | `gptoss`, `gpt-oss` | Text | Yes (always) | Yes | **Default** (toggle off with `TS_GPTOSS_BATCHED=0`). Per-head attention sinks via `TSGgml_PagedAttentionForwardWithSinks` (or `TS_GPTOSS_PAGED_ATTN_MANAGED=1` for the C# fallback). 100% greedy match vs legacy in `GptOssBatchedCorrectnessTests`. | Stacked MoE prefill kernel (mul_mat_id + add_id + swiglu_oai), attention sinks, MXFP4 expert weights |
| Nemotron-H | [nemotron.md](nemotron.md) | [bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF); Omni: [unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF) (+ `mmproj-BF16.gguf` for image); Nemotron 3.5 Lightning: [unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) (+ an optional DSpark drafter for `--draft-model`) | `NemotronModel` | `nemotron_h`, `nemotron_h_moe` | Text, image (Omni-class) | Yes | Yes | **Default** (toggle off with `TS_NEMOTRON_BATCHED=0`). Per-slot Mamba2 conv + SSM state pool; optional native batched Mamba2 step (`TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1`). 100% greedy match vs legacy; up to 3.95× tps at batch=3 on Apple M4 Pro. | Mamba2 + attention + MoE FFN hybrid stack, batched GPU MoE, RADIO/v2_vl image encoder, Parakeet audio preprocessor (audio inference needs a Parakeet mmproj the GGUF distributions do not ship) |
| Mistral 3 | [mistral3.md](mistral3.md) | [bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF) | `Mistral3Model` | `mistral3` | Text, image | No | No | **Default** — reference IBatchedPagedModel implementation. End-to-end validated on Ministral-3-14B; native paged-attention kernel is ~21% faster than the legacy per-seq path on long context. | YaRN-corrected RoPE with position-dependent Q scaling, fused QKV / gate_up, Pixtral vision encoder |
| Hunyuan Dense | [hunyuan-dense.md](hunyuan-dense.md) | Tencent dense Hunyuan GGUFs declaring `hunyuan-dense`, e.g. the Hy-MT2 releases (`tencent/Hy-MT2-1.8B` supplies the reference chat template) | [`HunyuanDenseModel`](../../TensorSharp.Models/Models/HunyuanDense/HunyuanDenseModel.cs) | `hunyuan-dense` | Text | No | No (the protocol renders neither tool declarations nor tool results) | Generic per-op forward | None yet — QKV and gate/up fusion at load, single device, no fused whole-model graph, no TP or layer split |
| Muse-Glimmer | [muse-glimmer.md](muse-glimmer.md) | [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) (`Muse-Glimmer-30B-*.gguf` + `mmproj-Muse-Glimmer-30B-*.gguf`; DFlash drafter `dflash-kquant.gguf` in the same repo) | `MuseGlimmerModel` | `muse-glimmer`, `muse_glimmer` | Text, image | Yes | Yes | No (legacy per-seq) | Interleaved SWA with NoPE full layers, attention output gate, 4 RMSNorms/layer (post-norms at eps 1e-8), logit scale + tanh softcap, sparse-window 2D-RoPE ViT with 2x2 pixel shuffle, optional DFlash block drafter (`--draft-model`, lossless), **tensor parallelism** (`--tp 2` on GGML CUDA/Vulkan — 2 KV heads cap the degree at 2) |
| MiniMax-H3 | [minimax-h3.md](minimax-h3.md) | Denoisers (separate checkpoints, not settings): `minimax_h3_fl2va_pruned-Q4_K.gguf` (text + keyframes) and `minimax_h3_ref2va_pruned-Q4_K.gguf` (text + references), plus the shared Qwen3-VL-32B text encoder `qwen3vl_32b_minimax_h3-Q4_K_M.gguf`, all from [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF); `minimax_h3_video_vae_fp16.safetensors` and `minimax_h3_audio_vae_fp32.safetensors` (omit for silent video) from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3). The text-encoder GGUF ships no tokenizer — put `vocab.json` and `merges.txt` from [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/processor) beside it, or point `TS_VIDEO_TOKENIZER` at them | `MiniMaxH3Model` (+ `MiniMaxH3Pipeline`) | `minimax-h3`, `minimax_h3` — **but neither published GGUF carries any metadata at all**, so `ModelBase.Create()` recognises H3 from its tensor table (`MiniMaxH3Architecture.DetectFromTensors`) instead of an architecture string | Video **+ native 32 kHz stereo audio** out, generated together in one packed latent (text → video, image → video, first/last frame, reference → video) | No | No | None — `Forward()` throws; generation runs through `GenerateVideo()` | Native whole-network ggml graphs, one graph per network, weights bound resident straight from the GGUF/safetensors mmap; CFG-distilled, so `--cfg 1.0` at 4–8 steps is the operating point (M5 Pro / Metal, 22 frames, 8 steps: 2.4× faster than stable-diffusion.cpp at 256×256 and 1.7× at 640×384); learned AdaLN curve table instead of a timestep MLP; 3-axis continuous-float RoPE putting video and audio on one timeline; video VAE decode chunked at 5 latent frames and tiled at 256 px as a correctness requirement rather than an optimization; FP16-safe `h3_attend` that pre-scales V by a power of two derived from the key count, which is what keeps a 107-frame clip finite |
| Wan video | [wan.md](wan.md) | Base: [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF), [QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF), [city96/Wan2.1-T2V-14B-gguf](https://huggingface.co/city96/Wan2.1-T2V-14B-gguf). **Step-distilled (25× less denoising work, same flags):** [hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF](https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF), [jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF](https://huggingface.co/jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF). (+ UMT5-XXL encoder and video VAE, see the card) | `WanVideoModel` (+ `WanVideoPipeline`) | `wan`, `wan2.1`, `wan2.2` | Video out (text -> video, image -> video) | No | No | None - `Forward()` throws; generation runs through `GenerateVideo()` and is serialized | Step-distilled checkpoints auto-detected from the DiT file name (100 DiT passes -> 4; M5 Pro 1088x832x121f: 3 h 30 m -> 17 m 30 s), one resident-weight ggml graph per denoise step (CUDA-graph-captured, flash attention over F16 keys/values -- 2.02x at 27 k tokens, per-token-timestep modulation for TI2V i2v), `--cfg-cache-stride` guidance reuse (1.30x / 1.43x on base checkpoints), causal 3D video VAE encode and decode each as a single graph with the convs on MPSGraph on Metal (VAE decode 1.99x), A14B's two 14B experts hot-swapped at the timestep boundary, stagewise VRAM handoff (TE -> DiT -> VAE), memory-sized im2col budget and 720p decode tiling |

## Backend notes

Model code is intentionally backend-agnostic. `ModelBase` selects tensor
storage through `BackendType` and the registered execution plan, then delegates
the actual ops to the backend that owns those allocators:

| Backend type | Package | Notes |
|---|---|---|
| `Cpu` | `TensorSharp.Core` | Pure managed tensors with SIMD/managed quantized fast paths (RMSNorm, RoPE, softmax, fused activations, GEMM, dequant). The quantized matmuls run on a persistent worker pool (`TensorSharp.Models/CpuWorkerPool.cs`) rather than a per-matmul `Parallel.For`, which is what lets decode scale past a handful of cores; it is sized at half the usable CPUs on purpose, and `TS_CPU_*` tunes it. Quantized weights are bound zero-copy from the GGUF mapping — the same file-backed binding the GGML backends already used — instead of being copied into fresh anonymous memory at load (`TS_DIRECT_QUANT_WEIGHTS=0` restores the old expand-to-F32 behaviour for an A/B). `ManagedQuantizedOps` also covers `IQ2_XS` and `IQ4_XS` (managed dequantizers plus entry into the CPU quantized-storage matrix, so they stay quantized instead of expanding to F32 at load), and has direct `IQ2_XS x Q8_K` and `IQ3_XXS x Q8_K` dot kernels with AVX2 paths. Families whose generation path bypasses the ggml graph (Wan, Qwen-Image, MiniMax-H3) share the direct primitives in `TensorSharp.Models/Direct/`. |
| `Cuda` | `TensorSharp.Backends.Cuda` | Direct CUDA Driver-API allocator and storage, cuBLAS GEMM, PTX kernels for hot ops (RMSNorm, softmax, RoPE/RoPEEx, SDPA, GQA prefill/decode, causal mask, gather/concat, activation fusions), native quantized matmul / get_rows for supported quant types, CPU fallback for ops that are not yet implemented. |
| `Mlx` | `TensorSharp.Backends.MLX` | Apple Silicon `mlx-c` bridge with quantized / fused / compiled kernels, async worker dispatch, MoE expert offload, and a CPU fallback layer. Requires `libmlxc`. |
| `GgmlCpu` / `GgmlMetal` / `GgmlCuda` / `GgmlVulkan` | `TensorSharp.Backends.GGML` + `TensorSharp.GGML.Native` | Native ggml bridge with quantized graph dispatch and platform backends. mmap-backed quantized weights are bound zero-copy through host-pointer buffers. Includes the paged-attention kernel (`TSGgml_PagedAttentionForward`, plus the GPT OSS sinks variant) that powers the batched / paged execution path. |

When a card mentions a fused GGML kernel (for example `Qwen35AttentionLayerDecode`,
`Gemma4LayerPrefill`, or `MoEExpertsSwiGLUResidual`), the kernel is compiled from
`TensorSharp.GGML.Native/ggml_ops_*.cpp` and exposed through
`TensorSharp.Backends.GGML/GgmlBasicOps.cs`. The native bridge is the place to
look when a fused path engages on GGML CPU / Metal / CUDA but not on the pure
managed CPU or direct CUDA backends.

## Continuous batching & paged KV cache

All autoregressive architectures listed above run through the shared
`InferenceEngine` + `ContinuousBatchScheduler` + `BatchExecutor` stack documented
in [`docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md`](../PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md).
Models that implement `IBatchedPagedModel.ForwardBatch` execute one batched
forward per scheduler step (with `slotMapping`-based K/V scatter into a
shared paged buffer and per-sequence attention via the native paged kernel);
the others run through the per-sequence KV-swap fallback inside the same engine.
DiffusionGemma does not support autoregressive `Forward()`, so it uses
`DiffusionGemmaSampler` and the server-side `DiffusionBatchScheduler` instead.
Qwen-Image-Edit is likewise not autoregressive: `Forward()` throws, editing runs
through `QwenImageModel.EditImage()` over a FlowMatch-Euler diffusion loop, and
concurrent edits are serialized (the diffusion nets are not thread-safe).
The opt-in env vars are summarised in the matrix above and in the project root
README.

Solo (non-concurrent) sequences on architectures that ship a multi-token-prediction
draft head — Qwen 3.6 (embedded NextN block) and Gemma 4 (separate `gemma4-assistant`
draft GGUF) — can additionally run lossless MTP speculative decoding through the same
engine (`--spec` opts in Qwen 3.6's embedded NextN block; for Gemma 4, naming the
draft GGUF on `--draft-model` enables speculation by itself. Both flags are accepted
on **both** hosts, since `TensorSharp.Cli` and
`TensorSharp.Server` share one flag parser; the `TS_SPEC_*` / legacy `TS_MTP_*` env
vars work too). The shared draft / verify /
rollback core is
`SpeculativeExecution`; per-architecture mechanics are in the Qwen 3.5/3.6 (§12)
and Gemma 4 (§12) cards.

DeepSeek V4 plugs a *block* drafter into that same core: its DSpark support module
ships as a separate GGUF loaded with `--draft-model` (on both the CLI and the server)
and proposes a whole block of tokens per step instead of one at a time. Because the
drafter's weights must be counted by the layer split, it is passed to
`ModelBase.Create()` at load time rather than attached afterwards. See the
[DeepSeek V4 card](deepseek4.md#dspark-speculative-decoding).

## Architecture comparison

| Feature | DeepSeek V4 | Gemma 4 | DiffusionGemma | Qwen 3.5 / 3.6 family | GPT OSS | Nemotron-H | Mistral 3 | Muse-Glimmer |
|---|---|---|---|---|---|---|---|---|
| Layer type | MoE (256 routed experts, top-6 + 1 shared) | Dense / MoE | Gemma-4-derived MoE encoder/decoder | Hybrid (Attn + Recurrent) ± MoE | MoE | Hybrid (Mamba2 + Attn + FFN, dense or MoE) | Dense | Dense (52 layers, 32 Q / 2 KV heads) |
| Attention | Raw SWA-128 + compressed CSA 4:1 / HCA 128:1 (lightning-indexer top-512 on CSA layers) | SWA + Global | Region-aware prompt/canvas attention | Full GQA + Sigmoid Gate | Full + Sinks | Full GQA (no RoPE) | Full GQA | Interleaved SWA-2048 + full NoPE layers (39 + 13), sigmoid attention output gate |
| FFN activation | SwiGLU with a per-layer clamp | GeGLU | Dense GeGLU + top-8 MoE | SwiGLU | SiLUAlphaLimit (clamped GLU) | ReLU² | SwiGLU | SwiGLU |
| RoPE variant | Interleaved-pair + YaRN; separate raw and compress bases, inverted after attention | NeoX + proportional / partial | NeoX, local/global bases | NeoX / MRoPE | NeoX + YaRN | None | GPT-J + YaRN | ggml NORM (interleaved pairs) on the SWA layers only; the full layers are NoPE |
| QK-norm | Q only (per-head RMS) | Yes | Yes | Yes | No | No | No | Yes (per-head; the Q norm carries the folded qk_scale_factor) |
| V-norm | No | Yes (unweighted) | Yes (unweighted) | No | No | No | No | No |
| Bias in projections | No (router selection bias only) | No | No | No | Yes (all linear) | No | No | No |
| Per-layer scaling | No (per-layer swiglu clamp and compress ratio instead) | Yes | Encoder / decoder scalars | No | No | No | No | No (logit scale 0.19612 + tanh softcap 20.0 on the output instead) |
| Per-Layer Embedding (PLE) | No | Yes | No | No | No | No | No | No |
| KV sharing | Yes (one shared 512-dim K=V head for all queries) | Yes (tail layers) | Prompt-KV cache across denoising steps | No | No | No | No | No |
| Attention sinks | Yes | No | No | No | Yes | No | No | No |
| Circular KV cache | Yes (raw SWA-128 ring) | Yes (SWA layers) | No autoregressive KV | No | No | No | No | Yes (SWA ring on the GPU backends; `TS_MUSE_GLIMMER_SWA_RING=0` disables) |
| SSM / recurrent layers | No (4-stream hyper-connections replace the plain residual) | No | No | Yes (GatedDeltaNet) | No | Yes (Mamba2) | No | No |
| Shared experts | Yes | No | No | Yes (qwen35moe / qwen3next) | No | Yes (optional) | No | No (dense FFN) |
| Latent bottleneck FFN | No (LoRA-factored Q / output projections instead) | No | No | No | No | Yes (optional) | No | No |
| Position-dependent Q scaling | No | No | No | No | No | No | Yes (with YaRN) | No |
| Vision | No | Yes | No | Yes | No | Yes (Omni) | Yes (Pixtral) | Yes (sparse-window 2D-RoPE ViT with 2×2 pixel shuffle) |
| Audio | No | Yes | No | No | No | No — image-only Omni (Parakeet log-mel preprocessing exists, but inference needs an audio mmproj that is not shipped) | No | No |
| Video | No | Yes | No | No | No | No | No | No |
| Thinking | Yes | Yes | No | Yes | Yes (always) | Yes | No | Yes (`assistant to=self` channel) |
| Tool calling | Yes (DSML markup) | Yes | No | Yes | Yes | Yes | No | Yes (ATEM XML markup) |
| MTP / NextN speculative decoding | DSpark block drafter (separate GGUF via `--draft-model`) | Yes (separate `gemma4-assistant` draft GGUF) | No | Yes on Qwen 3.6 (embedded NextN block) | No | No | No | DFlash block drafter (separate 5-layer GGUF via `--draft-model`, lossless) |
| Fused QKV | n/a (LoRA-factored Q, single shared K=V head) | Yes | Yes | Mixed (full attention layers split, recurrent layers fuse a 5-way pack) | Yes | Yes | Yes | No |
| Fused single-graph decode | Yes (whole-model executor, one graph per ubatch, CUDA-graph replayed) | Yes (Gemma4ModelDecode) | Yes (DiffusionModelDecode + lm-head tail) | Per-layer fused (Qwen35AttentionLayerDecode, FusedOutProjFFN, FusedOutProjNormRouter) | Per-layer | Per-layer / batched MoE | No | Yes (persistent whole-model decode graph on GGML CUDA / Vulkan / Metal / CPU) |
| Fused single-graph prefill | Yes (same whole-model executor, chunked ubatches) | Yes (whole-model NativeGemma4ModelVerify + per-layer Gemma4LayerPrefill fallback) | Prompt-KV prefill cache | Yes (FusedPrefillAttention, FusedOutProjFFN, MoE prefill) | Yes (MoE prefill via mul_mat_id) | No | No | Yes (same fused kernel, chunked with on-device causal+SWA band masks) |
| Batched GPU MoE | Yes (grouped expert kernels) | Yes for all-MoE variants (fused whole-model MoE decode/verify); mixed dense+MoE pending | Fused per-canvas MoE; concurrent requests batched by diffusion scheduler | Yes (routed + shared + residual fused) | Yes (stacked weight slabs) | Yes | n/a | n/a (dense FFN) |
| Fused vision encoder | n/a | Standard | n/a | Yes (FusedVisionAttention + FusedVisionMLP) | n/a | Standard (RADIO ViT) | Standard (Pixtral) | Yes (fused vision block + flash attention on CUDA) |
| Output parser | `DeepSeek4OutputParser` | `Gemma4OutputParser` | `PassthroughOutputParser` | `Qwen35OutputParser` | `HarmonyOutputParser` (always required) | `ChatMlOutputParser` | `PassthroughOutputParser` | `MuseGlimmerOutputParser` |

## Adding a new architecture

When you add a new model:

1. Create `TensorSharp.Models/Models/<Name>/<Name>Model.cs` inheriting
   `ModelBase`.
2. In the constructor: read GGUF metadata via `_gguf.GetXxx()`, call
   `ParseBaseConfig()` and `ParseTokenizer()`, call `LoadWeights()`, fuse
   weights, then initialize caches.
3. Implement `Forward(int[] tokens) → float[]` for autoregressive models:
   embedding → optional multimodal injection → transformer blocks → final norm
   → LM head → logit copy. For diffusion models, document the alternate sampler
   entry point and make unsupported autoregressive paths explicit.
4. Implement `ResetKVCache()` and `Dispose()`. Implement `TruncateKVCache()`
   when KV-cache reuse is supported.
5. Declare the architecture plug-in in
   `TensorSharp.Models/Models/<Name>/<Name>Architecture.cs` -- a
   `ModelArchitectureDescriptor` with the GGUF `general.architecture` aliases,
   the factory, and anything non-default (multi-GPU mode and why, mmproj
   companion file hints, a tensor-based detector for metadata-free GGUFs,
   process-wide native tunables) -- then add ONE line for it to
   `TensorSharp.Models/Architecture/BuiltInArchitectures.cs`. There is no switch
   to extend: `ModelBase.Create()` resolves through `ModelArchitectureRegistry`.
6. If the model is multimodal, implement the capability interfaces on it:
   `IVisionCapableModel` (load the tower, receive an embedding span) and
   `IMultimodalPromptExpander` (expand your own placeholders), plus
   `IAudioCapableModel` / `IAudioEncoderLoader` / `IMRoPEPositionSink` as
   applicable. `ModelMultimodalInjector` owns all the generic bookkeeping and
   names no model types, so nothing there needs editing.
7. If the model has its own chat format, add ONE `ChatProtocol` entry to
   `TensorSharp.Runtime/ChatProtocolRegistry.cs`. That single record carries the
   renderer, whether to bypass the GGUF's Jinja template, the media placeholder
   tokens, the `IOutputParser` (implemented in
   `TensorSharp.Runtime/OutputParser.cs`), whether that parser is mandatory,
   where a structured-output grammar may arm, the KV-cache generation suffix
   that keeps multi-turn prefix reuse working, and video-frame capping.
8. Add a card under `docs/models/<name>.md` (and `<name>_zh-cn.md` if you want
   bilingual coverage), update this README's matrix, and link the card from
   the project root README.
9. Update `TensorSharp.Server/testdata/` capability gates if the model exposes
   new modalities, thinking, or tool capabilities.
