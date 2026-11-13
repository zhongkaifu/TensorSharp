# Detailed Model Capabilities and Implementation

[English](capability-reference.md) | [中文](capability-reference_zh-cn.md)

[← Choose a model by task](README.md) · [Project status](../PROJECT_STATUS.md) · [Backend and feature support](../supported_models.md)

The detailed material below was moved intact from the model index, retaining architecture, implementation, validation limits and recorded numbers. “Yes” and “Default” in these tables describe the stated code or architecture capability; availability still depends on the checkpoint, companions, backend and validation scope in its card. Use the index's [support-label definitions](README.md#reading-support-status) to distinguish these layers. Maintainers adding a family should update both the selection index and this detailed matrix.

<!-- preserved-index-detail:start -->
This folder is the canonical, per-model reference for every architecture that
TensorSharp can run. Each card is a self-contained brief: it walks an engineer
or researcher from "I have never heard of this model" all the way to "I can
explain the forward graph and reproduce the inference path in TensorSharp." If
you only need a top-level pointer, use the table below; otherwise jump into the
individual cards.

For EmbeddingGemma 2 and BERT/XLM-R sentence encoders (Snowflake Arctic Embed L v2.0 and MiniLM), see the [embedding model guide](../embeddings.md) for downloads, the forward graph, and APIs.

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
| DeepSeek V4.1 Flash | [deepseek41.md](deepseek41.md) | [repaired smalinin Q2_K/Q5_K checkpoint](https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/tree/d1de55c19f95172c882906cc83c0e55932d26a63/Q2_K-Q5), ten shards / 312.349 GiB with embedded Engram; [validation status](../deepseek41_validation.md) | `DeepSeek4Model` with a dedicated native V4.1 graph, plus the pure-C# `DeepSeek4CpuExecutor`, which implements that same V4.1 graph for `--backend cpu` | `deepseek41` | Text; image and video through the separately prepared vision companion (`--mmproj`; it is native, so it needs a ggml backend — `LoadVisionEncoder` throws on `--backend cpu`), audio rejected with HTTP 400 | Reference-format renderer | Spaced DSML; model-level validation tracked separately | Per-sequence slots; token-batched fused CUDA decode with per-slot fallbacks | `ggml_cuda`, layer placement, CPU MoE, and experimental routed-MoE TP; attention/distributed TP remain unimplemented, and V4.1 DSpark is experimental (`--draft-model` loads a `deepseek41-dspark` drafter on `ggml_cuda` / `ggml_cpu` only; initial text/image HTTP probes with trained weights passed using two-GPU layer split on `ggml_cuda`; broad quality and throughput remain unqualified). `--backend ggml_cpu` (the same native graph on scalar ggml kernels) and `--backend cpu` (the pure-C# executor, with no ggml, no native library and no GPU) are correctness and portability paths, not serving paths |
| DeepSeek V4 Flash | [deepseek4.md](deepseek4.md) | [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) (multi-shard per quant directory; point `--model` at the `-00001-of-` shard). DSpark drafters: [MODEL_DOWNLOADS.md](../../MODEL_DOWNLOADS.md#dspark-drafters) | `DeepSeek4Model` (+ `DeepSeek4CudaExecutor`, `DeepSeek4CpuExecutor`) | `deepseek4` | Text | Yes | Yes (DSML markup) | Native per-sequence slots (`DeepSeek4Model.PerSeqCache.cs`) rather than `IBatchedPagedModel` — servable with continuous batching through the same engine | Three whole-model executors (direct CUDA, native ggml, pure C#), whole-layer placement across N local GPUs with `--layer-split N` (default: one device), on-device compressed KV state (SWA ring + CSA/HCA + lightning indexer), shape-signature graph cache replaying a captured CUDA graph, fused decode index-gather over `[ring \| top-512]` K, and DSpark block speculative decoding (1.3–1.4× decode) |
| Qwen 3.8 Flash Next | [qwen38-flash-next.md](qwen38-flash-next.md) | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (multi-shard; point `--model` at the `-00001-of-` shard) | `Qwen4ExpRuntimeModel` (shared decoder runtime: GGML backends and cpu) and `Qwen4ExpCudaModel` / `Qwen4ExpCudaExecutor` (direct CUDA) | `qwen4exp` | Text + image + video (`video_url`) | Yes | Yes | Slots on the shared runtime (every backend but cuda); the direct engine's slots on cuda | One graph per step (hyper-connections, PLE, QSA dense to the indexer budget then block-selecting, MoE with host seams and tiers), Metal prompt kernels as device operations; IMRoPE vision with QSA over media positions; prefix cache across media spans; MTP head (`--draft-model mtp-*.gguf`) and DFlash2 speculation; `--tp N` as per-rank graphs on ggml_cuda / ggml_vulkan and `--layer-split N` (their hybrid refused); see the model card for validation limits |
| GLM 5.x | [glm.md](glm.md) | [unsloth/GLM-5.2-GGUF](https://huggingface.co/unsloth/GLM-5.2-GGUF), [unsloth/GLM-5.3-GGUF](https://huggingface.co/unsloth/GLM-5.3-GGUF) (UD-Q2_K_XL is seven shards, 236.4 GiB), [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) (multi-shard per quant directory; point `--model` at the `-00001-of-` shard) | `GlmDsaModel` (+ the native `ggml_ops_glm_dsa.cpp` whole-model executor); GLM-5.3 (the same 79-block `glm-dsa` shape as 5.2, 256 experts, text only) loads on the GLM-5.2 path with no new code and no new flag — see the [card](glm.md#glm-53-glm-dsa) | `glm-dsa` (GLM-5.2 and GLM-5.3), `glm5next` (5.3-Flash) — one architecture descriptor serves all three, Flash vs non-Flash being a single boolean on `general.architecture` | Text (5.2 and 5.3 — the 5.3 repo publishes no mmproj at any quant, and `LoadVisionEncoder` warns and ignores an `--mmproj` on `glm-dsa` instead of failing the run); text + image (5.3-Flash via `mmproj`) | Yes | Yes (XML tool calls) | Native per-sequence slots (`TSGgml_GlmSlotAlloc`) rather than `IBatchedPagedModel` — servable with continuous batching through the same engine | Native whole-model ggml executor and a pure-C# per-op reference (GLM 5.2, GLM-5.3 and GLM-5.3-Flash all run on `--backend cpu`, but it is a reference implementation to A/B against rather than bit-parity: on GLM-5.3-Flash the prefill-logit cosine is 0.9567 against `ggml_cpu` and the greedy token differs — see the [card](glm.md#glm-53-flash-on---backend-cpu)), automatic whole-layer placement with `--layer-split N` **or** Megatron tensor parallelism (`--tp N`: column/row-parallel heads, every routed expert split row-wise), `--cpu-moe` host-resident experts served straight from the GGUF mapping, MLA weight absorption with a 576-wide cache row, DSA lightning indexer with a selection reused across 57 of 78 layers, and a shape-keyed graph cache replaying a captured CUDA graph; on GLM-5.3 the `blk.78` NextN block drafts with `--spec` with single-device or explicit `--layer-split N` placement (no active tensor parallelism), because it ships no LM head of its own and borrows the trunk LM head, which `--tp` splits column-wise |
| Gemma 4 | [gemma4.md](gemma4.md) | E4B Q8_0 is the verified native-GGML family/path tier; [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) is the recommended public artifact | `Gemma4Model` | `gemma4` (`gemma4-assistant` / `gemma4_assistant` load only as the MTP draft) | Text, image, video, audio | Yes | Yes | **Default** | Single-graph fused decode (all layers in one GGML dispatch), fused whole-model prefill/verify with in-kernel PLE + shared-KV handling, chunked prefill, circular SWA cache, and MoE variants. Batched path matches per-seq logits within FP noise (`Gemma4BatchedForwardTests`); reaches ~1.5× the per-seq path at batch=8 and ~1.6× at 4×800-token prompts. Concurrent decode runs the token-batched fused kernel (PLE + KV-donor + SWA wrap, so E2B/E4B qualify): short-horizon streams token-for-token equal to round-robin (long greedy runs can flip a low-margin token, as with any batched GEMM), 2.2× aggregate decode at 4 concurrent on E4B Q8_0 / A40. |
| DiffusionGemma | [diffusiongemma.md](diffusiongemma.md) | [unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF); vision tower: `model-00011-of-00011.safetensors` (2.84 GB) from [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) | `DiffusionGemmaModel` + `DiffusionGemmaSampler` | `diffusion-gemma`, `diffusion_gemma` | Text + image (Gemma 4's `gemma4v` vision tower, from an mmproj GGUF or the raw HF `model-00011-of-00011.safetensors` shard); ordinary chat refuses audio and `video_url` (Web UI video arrives as plain image frames); [Jev](jev.md) adds documents, sampled video and configured ASR transcripts | No (not prompted; a thought block the model writes anyway is returned only with `think: true`) | No (refused with HTTP 400) | Separate Web UI `DiffusionBatchScheduler`; not an autoregressive `IBatchedPagedModel` path | EntropyBound block denoising over `[prompt \| canvas]`, prompt-KV caching on the GPU backends and the pure-C# `cpu` backend (host K/V, batched MoE), self-conditioning, fused GGML whole-model diffusion decode and fused lm-head tail; the same model serves the [Jev](jev.md) decision API (`POST /v1/systemone`) |
| Qwen-Image-2.1 | [qwenimage21.md](qwenimage21.md) | [Abiray/Qwen-Image-2.1-GGUF](https://huggingface.co/Abiray/Qwen-Image-2.1-GGUF); a metadata-free Unsloth Q8_0 GGUF is recognised from its tensor layout (its `mmproj-BF16.gguf` needs `--qwen-image-mmproj`) | `QwenImageModel` | `qwen_image`, `qwen-image` | Text-to-image, reference-image and masked local edit | No | No | Serialized diffusion requests | Qwen3-VL-8B conditioning, dedicated 2.1 VAE, FlowMatch-Euler, unmerged LoRA plug-ins (`--lora`), a default-on prefix KV cache for the text and reference-image tokens (`TS_QWEN21_PREFIX_CACHE=0` disables), and diffusion-transformer tensor parallelism (`--tp` on `ggml_cuda` / `ggml_vulkan`; the encoders and VAE stay on GPU 0). Runs on the GGML backends and on the pure-C# `cpu` backend, where the DiT, text encoder, vision encoder and VAE are managed code (single process; `--tp` refused) |
| Qwen 3.5 / 3.6 family | [qwen35.md](qwen35.md) | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF); NextN MTP: [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) (base-repo Qwen3.6 GGUFs strip the NextN block and silently fall back to standard decode) | `Qwen35Model` | `qwen35`, `qwen35moe`, `qwen3next` | Text, image | Yes | Yes | **Default**. Per-slot recurrent-state pool | Hybrid FullAttention + GatedDeltaNet recurrent, fused attention layer decode, fused prefill attention, fused output-projection + FFN, fused output-projection + norm + router, batched MoE (routed + shared + residual in a single kernel), fused vision encoder blocks |
| Bonsai2 27B | [bonsai2.md](bonsai2.md) | Local `Ternary-Bonsai-2-27B-PQ2_0.gguf` / `PTQ1_0.gguf`; exact hashes in the card | `Qwen35RuntimeModel` | `qwen35` plus `prism.hadamard.*` | Text, image with companion projector | Yes | Yes | Qwen 3.5 per-sequence KV/recurrent state; single-device GGML | Lossless publisher-format repacking and TensorSharp-owned signed Hadamard transforms; on `ggml_metal` the 27B dense hybrid geometry it shares with Qwen3.8-27B prefills in 512-token chunks by default (`TS_PREFILL_CHUNK` overrides); see validation scope in the card |
| GPT OSS | [gptoss.md](gptoss.md) | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) | `GptOssModel` | `gptoss`, `gpt-oss` | Text | Yes (always) | Yes | **Default**. Per-head attention sinks via `TSGgml_PagedAttentionForwardWithSinks` (a C# fallback on non-GGML backends). 100% greedy match vs the per-seq path in `GptOssBatchedCorrectnessTests`. | Stacked MoE prefill kernel (mul_mat_id + add_id + swiglu_oai), attention sinks, MXFP4 expert weights |
| Nemotron-H | [nemotron.md](nemotron.md) | [bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF); Omni: [unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF) (+ `mmproj-BF16.gguf` for image); Nemotron 3.5 Lightning: [unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) | `NemotronModel` | `nemotron_h`, `nemotron_h_moe`, `nemotron_h_omni` | Text, image (Omni-class); audio only with a companion GGUF carrying the Parakeet tower | Yes | Yes | **Default**. Per-slot Mamba2 conv + SSM state pool; optional native batched Mamba2 step (`TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1`). 100% greedy match vs the per-seq path; up to 3.95× tps at batch=3 on Apple M4 Pro. | Mamba2 + attention + MoE FFN hybrid stack, batched GPU MoE, RADIO/v2_vl image encoder, Parakeet/FastConformer audio tower (it runs only when a companion GGUF carrying it is loaded through `--mmproj` or `TS_NEMOTRON_AUDIO_MMPROJ`; the public GGUF distributions ship none) |
| Mistral 3 | [mistral3.md](mistral3.md) | [bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF) | `Mistral3Model` | `mistral3` (plus `llama`-labelled Mistral Small 3.x files) | Text, image | No | No | **Default** — the shared decoder runtime's batched forward (every placement). End-to-end validated on Ministral-3-14B; native paged-attention kernel is ~21% faster than the per-seq path on long context. | YaRN-corrected RoPE with position-dependent Q scaling, fused QKV / gate_up, Pixtral vision encoder |
| Hunyuan Dense | [hunyuan-dense.md](hunyuan-dense.md) | Tencent dense Hunyuan GGUFs declaring `hunyuan-dense`, e.g. the Hy-MT2 releases (`tencent/Hy-MT2-1.8B` supplies the reference chat template) | [`HunyuanDenseModel`](../../TensorSharp.Models/Models/HunyuanDense/HunyuanDenseModel.cs) | `hunyuan-dense` | Text | No | No (the protocol renders neither tool declarations nor tool results) | **Default** (`IBatchedPagedModel` over the generic per-op kernels; a `q8_0` / `q4_0` KV cache selects the K/V-snapshot swap path) | Shared decoder runtime: QKV and gate/up fusion at load, `--tp`, `--layer-split` and their hybrid with batched serving in every placement; no fused whole-model graph yet |
| Muse Spark | [muse-spark.md](muse-spark.md) | [unsloth/Muse-Spark-Open-816B-A42B-GGUF](https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-GGUF) | `MuseSparkModel` | `muse-spark`, `muse_spark` | Text, image with matching Spark projector; no audio | Yes | Yes | **Default** on `ggml_cuda`/`ggml_vulkan` (native executor: batched decode and batched text prefill over per-sequence slots); per-op path: packed paged text with F32 KV | Native whole-model executor (one GGML graph per micro-batch over every GPU and the CPU), per-stack expert placement, NUMA-local host experts, sliding-window KV rings, prefix checkpoints shared across conversations, n-gram speculation. Pipelined layer-split prefill and native tensor parallelism (`--tp N` on `ggml_cuda`). On 10x A40 with UD-IQ3_XXS: layer split prefills 2.07x / 1.63x faster than llama.cpp (4K / 16K) at equal decode; `--tp 10` decodes 1.40x llama.cpp's `-sm tensor` ([report](../perf/muse-spark-vs-llama-cpp.md)). |
| MiniMax-M3.1-Flash | [minimax-m3.md](minimax-m3.md) | [unsloth/MiniMax-M3.1-Flash-GGUF](https://huggingface.co/unsloth/MiniMax-M3.1-Flash-GGUF) `UD-Q4_K_XL/` (seven shards, 232.3 GiB; point `--model` at the `-00001-of-` shard). The repository has no mmproj or DSpark drafter yet: both are converted from [MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) ([card](minimax-m3.md#make-the-vision-projector-and-the-dspark-drafter)) | `MiniMaxM3Model` over a native whole-model executor | `minimax-m3`, `minimax_m3` | Text, image with the converted `minimax_m3` projector, video as sampled frames; audio refused | Yes (adaptive; `think` forces it on, no off switch) | Yes (namespaced XML; grammar-enforced `tool_choice` required / named) | **Default** (native executor: batched decode and batched text prefill over per-sequence F16 slots) | Native whole-model executor on `ggml_cuda` (`ggml_cpu` as a correctness path; other backends refused): MiniMax Sparse Attention on every layer, exact dense below 2048 and TensorSharp CUDA MSA kernels above, NVFP4 experts with per-expert scales, pipelined layer-split prefill, host experts (`--n-cpu-moe`, `TS_MINIMAX_M3_CPU_MOE=auto`), native tensor parallelism (`--tp N` on ggml's meta backend, attention and caches mirrored, FFN split), prefix checkpoints, DSpark block drafting (layer split). On 8x A40 against llama.cpp: 32K-token server prefill 6.43x, decode at 4K depth 1.24x, DSpark decode 1.25x ([measurements](../benchmarks.md#minimax-m31-flash-on-eight-a40-gpus)) |
| Muse-Glimmer | [muse-glimmer.md](muse-glimmer.md) | [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) (`Muse-Glimmer-30B-*.gguf` + `mmproj-Muse-Glimmer-30B-*.gguf`; DFlash drafter `dflash-kquant.gguf` in the same repo) | `MuseGlimmerModel` | `muse-glimmer`, `muse_glimmer` | Text, image | Yes | Yes | No (per-seq KV swap) | Interleaved SWA with NoPE full layers, attention output gate, 4 RMSNorms/layer (post-norms at eps 1e-8), logit scale + tanh softcap, sparse-window 2D-RoPE ViT with 2x2 pixel shuffle, optional DFlash block drafter (`--draft-model`, lossless), **tensor parallelism** (`--tp 2` on GGML CUDA/Vulkan — 2 KV heads cap the degree at 2) |
| MiniMax-H3 | [minimax-h3.md](minimax-h3.md) | The single TensorSharp config and the TensorAgent catalog download both denoisers; chat and Video Studio switch modes without reloading shared components. Denoisers (separate checkpoints, not settings): `minimax_h3_fl2va_pruned-Q4_K.gguf` (text + keyframes) and `minimax_h3_ref2va_pruned-Q4_K.gguf` (text + references), plus the shared Qwen3-VL-32B text encoder `qwen3vl_32b_minimax_h3-Q4_K_M.gguf`, all from [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF); `minimax_h3_video_vae_fp16.safetensors` and `minimax_h3_audio_vae_fp32.safetensors` (omit for silent video) from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3). The config also fetches the tokenizer files (`vocab.json`, `merges.txt`, `tokenizer_config.json`) from [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/processor); manual installs may use `TS_VIDEO_TOKENIZER` | `MiniMaxH3Model` (+ `MiniMaxH3Pipeline`) | `minimax-h3`, `minimax_h3` — **but neither published GGUF carries any metadata at all**, so `ModelBase.Create()` recognises H3 from its tensor table (`MiniMaxH3Architecture.DetectFromTensors`) instead of an architecture string | Video **+ native 32 kHz stereo audio** out, generated together in one packed latent (text → video, image → video, first/last frame, reference → video) | No | No | None — `Forward()` throws; generation runs through `GenerateVideo()` | Native graph paths on `ggml_cpu`/`ggml_cuda`/`ggml_metal` and direct paths on `cpu`/`cuda`/`mlx`; corrected component gates exercised on all six. Direct CUDA has independent same-weight full 50-block F32 and matched-input one-step stereo waveform evidence. H3-only bounded quantized projections and BF16-weight projections default to pedantic F32. Latest two-round 20-step before/after process checks preserve exact RGB/audio, with 5.0% lower Metal latency, 12.5% lower MLX latency and 19.0% lower direct CUDA latency at equal precision; the 1.3% native CUDA difference is not a meaningful gain. Whole-process timings include loading/codecs and uncontrolled caches; component and full-generation evidence are separate in the card. Earlier Metal 2.4×/1.7× generation claims are superseded by no-RoPE text refinement and corrected audio carrier/endpoint math. CFG 1, learned AdaLN, continuous three-axis trunk RoPE, tiled/chunked video VAE and scoped stage residency. |
| LTX 2.5 Distilled | [ltx-2.5.md](ltx-2.5.md) | Pinned UD-Q4_K_XL denoiser, custom Gemma 4 with projections and embedded tokenizer, convolutional video VAE, x2 latent upscaler and audio VAE from the shipped preset (43.08 GB); optional gated Ingredients IC-LoRA adds 1.31 GB | `Ltx25Model` + `Ltx25Pipeline` | `ltx-2.5` (+ aliases/tensor detection) | Text/image→video + 48 kHz stereo audio | No | No | None; `GenerateVideo()` serializes generation | Native CUDA/MLX; BF16 prompt boundaries, the publisher's two-stage recipe (eight ancestral Euler steps at half size, x2 latent upscale, three refinement steps at CFG 1, full 48-block AV sampling), streamed prompt/VAE weights, retained denoiser and bounded prompt caches; experimental 48 GB tier; first/last/timed image keyframes and optional Ingredients identity sheets; no reference-video/audio or multi-GPU path |
| Wan video | [wan.md](wan.md) | Base: [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF), [QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF), [city96/Wan2.1-T2V-14B-gguf](https://huggingface.co/city96/Wan2.1-T2V-14B-gguf). **Step-distilled (25× less denoising work, same flags):** [hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF](https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF), [jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF](https://huggingface.co/jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF). (+ UMT5-XXL encoder and video VAE, see the card) | `WanVideoModel` (+ `WanVideoPipeline`) | `wan`, `wan2.1`, `wan2.2` | Video out (text -> video, image -> video) | No | No | None - `Forward()` throws; generation runs through `GenerateVideo()` and is serialized | Step-distilled checkpoints auto-detected from the DiT file name (100 DiT passes -> 4; M5 Pro 1088x832x121f: 3 h 30 m -> 17 m 30 s), one resident-weight ggml graph per denoise step (CUDA-graph-captured, flash attention over F16 keys/values -- 2.02x at 27 k tokens, per-token-timestep modulation for TI2V i2v), `--cfg-cache-stride` guidance reuse (1.30x / 1.43x on base checkpoints), causal 3D video VAE encode and decode each as a single graph with the convs on MPSGraph on Metal (VAE decode 1.99x), A14B's two 14B experts hot-swapped at the timestep boundary, stagewise VRAM handoff (TE -> DiT -> VAE), memory-sized im2col budget and 720p decode tiling |

## Backend notes

Model code is intentionally backend-agnostic. `ModelBase` selects tensor
storage through `BackendType` and the registered execution plan, then delegates
the actual ops to the backend that owns those allocators:

| Backend type | Package | Notes |
|---|---|---|
| `Cpu` | `TensorSharp.Core` | Pure managed tensors with SIMD/managed quantized fast paths (RMSNorm, RoPE, softmax, fused activations, GEMM, dequant). The quantized matmuls run on a persistent worker pool (`TensorSharp.Models/CpuWorkerPool.cs`) rather than a per-matmul `Parallel.For`, which is what lets decode scale past a handful of cores; it is sized at half the usable CPUs on purpose, and `TS_CPU_*` tunes it. Quantized weights are bound zero-copy from the GGUF mapping — the same file-backed binding the GGML backends already used — instead of being copied into fresh anonymous memory at load. `ManagedQuantizedOps` also covers `IQ2_XS` and `IQ4_XS` (managed dequantizers plus entry into the CPU quantized-storage matrix, so they stay quantized instead of expanding to F32 at load), and has direct `IQ2_XS x Q8_K` and `IQ3_XXS x Q8_K` dot kernels with AVX2 paths. Families whose generation path bypasses the ggml graph (Wan, MiniMax-H3) share the direct primitives in `TensorSharp.Models/Direct/`. |
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
Prompt-prefix reuse on every autoregressive family in the matrix goes through the
engine's Radix prefix cache, which is on by default (`TS_SCHED_PREFIX_CACHE=0` or
`--no-prefix-cache` disables reuse).
DiffusionGemma does not support autoregressive `Forward()`, so it uses
`DiffusionGemmaSampler` and the server-side `DiffusionBatchScheduler` instead.
Its [Jev decision API](jev.md) reads typed answer probabilities directly from a
seeded canvas with one denoising step and a selected-label output projection, over
a state that may include images, uploaded documents, sampled video frames and
configured ASR transcripts as well as text.
Qwen-Image-2.1 is likewise not autoregressive: `Forward()` throws, generation runs
through `QwenImageModel.GenerateImage()` and editing through
`QwenImageModel.EditImage()` over a FlowMatch-Euler diffusion loop, and concurrent
requests are serialized (the diffusion nets are not thread-safe).
The opt-in env vars are summarised in the matrix above and in the project root
README.

Solo (non-concurrent) sequences on architectures that ship a multi-token-prediction
draft head — Qwen 3.6 and Qwen 3.8-27B (embedded NextN block), GLM 5.2 and GLM-5.3
(embedded NextN block, single-device or layer-split placement only), Gemma 4 (separate `gemma4-assistant`
draft GGUF) and Qwen 3.8 Flash Next (separate shared MTP head GGUF, GGML backends
only) — can additionally run lossless MTP speculative decoding through the same
engine (`--spec` opts in the embedded NextN blocks; for Gemma 4 and Qwen 3.8
Flash Next, naming the draft GGUF on `--draft-model` enables speculation by itself. Both flags are accepted
on **both** hosts, since `TensorSharp.Cli` and
`TensorSharp.Server.Host` share one flag parser (`SpeculativeCliFlags`); the `TS_SPEC_*` env
vars work too). The shared draft / verify /
rollback core is
`SpeculativeExecution`; per-architecture mechanics are in the Qwen 3.5/3.6 (§12)
and Gemma 4 (§12) cards.

DeepSeek V4 plugs a *block* drafter into that same core: its DSpark support module
ships as a separate GGUF loaded with `--draft-model` (on both the CLI and the server)
and proposes a whole block of tokens per step instead of one at a time. Because the
drafter's weights must be counted by the layer split, it is passed to
`ModelBase.Create()` at load time rather than attached afterwards. See the
[DeepSeek V4 card](deepseek4.md#dspark-speculative-decoding). DeepSeek V4.1 accepts
only a `deepseek41-dspark` drafter, on `ggml_cuda` or `ggml_cpu`; that path is
experimental; initial text/image HTTP probes with trained weights passed using two-GPU layer split on `ggml_cuda`; broad quality and throughput remain unqualified.
Muse-Glimmer (DFlash / DFlash2) and Qwen 3.8-27B (DFlash2) also take block drafters
through `--draft-model`. MiniMax-M3.1 takes its DSpark drafter (a `dflash` GGUF with a Markov head) the same way;
its native executor loads it with the target on a layer split and refuses it under `--tp`
([MiniMax-M3.1 card](minimax-m3.md#dspark-speculative-decoding)).

A weight-free n-gram (prompt-lookup) speculator, selected with `--spec-type ngram`,
needs no drafter and runs on the Qwen 3.5 family, Gemma 4, GLM 5.x, Qwen 3.8
Flash Next, Muse Spark (its only speculator) and MiniMax-M3.1 (layer split only), and on DeepSeek V4 / V4.1 and Muse-Glimmer only while their drafter is
loaded. It does not run on GPT OSS, Mistral 3 or Hunyuan Dense,
and Nemotron-H refuses every speculator; see
[speculative decoding](../speculative_decoding.md).

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
| Vision | No | Yes | Yes (Gemma 4's `gemma4v` tower, from an mmproj GGUF or the HF safetensors shard) | Yes | No | Yes (Omni) | Yes (Pixtral) | Yes (sparse-window 2D-RoPE ViT with 2×2 pixel shuffle) |
| Audio | No | Yes | No (refused) | No | No | Only with a companion GGUF carrying the Parakeet tower (`--mmproj` or `TS_NEMOTRON_AUDIO_MMPROJ`); the public Omni GGUFs ship none | No | No |
| Video | No | Yes | No | No | No | No | No | No |
| Thinking | Yes | Yes | No (not prompted) | Yes | Yes (always) | Yes | No | Yes (`assistant to=self` channel) |
| Tool calling | Yes (DSML markup) | Yes | No (refused) | Yes | Yes | Yes | No | Yes (ATEM XML markup) |
| MTP / NextN speculative decoding | DSpark block drafter (separate GGUF via `--draft-model`) | Yes (separate `gemma4-assistant` draft GGUF) | No | Yes: embedded NextN on Qwen 3.6 and Qwen 3.8-27B (`--spec`); a DFlash2 block drafter via `--draft-model` on Qwen 3.8-27B | No | No | No | DFlash block drafter (separate 5-layer GGUF via `--draft-model`, lossless) |
| Fused QKV | n/a (LoRA-factored Q, single shared K=V head) | Yes | Yes | Mixed (full attention layers split, recurrent layers fuse a 5-way pack) | Yes | Yes | Yes | No |
| Fused single-graph decode | Yes (whole-model executor, one graph per ubatch, CUDA-graph replayed) | Yes (Gemma4ModelDecode) | Yes (DiffusionModelDecode + lm-head tail) | Per-layer fused (Qwen35AttentionLayerDecode, FusedOutProjFFN, FusedOutProjNormRouter) | Per-layer | Per-layer / batched MoE | No | Yes (persistent whole-model decode graph on GGML CUDA / Vulkan / Metal / CPU) |
| Fused single-graph prefill | Yes (same whole-model executor, chunked ubatches) | Yes (whole-model NativeGemma4ModelVerify + per-layer Gemma4LayerPrefill fallback) | Prompt-KV prefill cache | Yes (FusedPrefillAttention, FusedOutProjFFN, MoE prefill) | Yes (MoE prefill via mul_mat_id) | No | No | Yes (same fused kernel, chunked with on-device causal+SWA band masks) |
| Batched GPU MoE | Yes (grouped expert kernels) | Yes for all-MoE variants (fused whole-model MoE decode/verify); mixed dense+MoE pending | Fused per-canvas MoE; concurrent requests batched by diffusion scheduler | Yes (routed + shared + residual fused) | Yes (stacked weight slabs) | Yes | n/a | n/a (dense FFN) |
| Fused vision encoder | n/a | Standard | Standard (Gemma 4's) | Yes (FusedVisionAttention + FusedVisionMLP) | n/a | Standard (RADIO ViT) | Standard (Pixtral) | Yes (fused vision block + flash attention on CUDA) |
| Output parser | `DeepSeek4OutputParser` (always required) | `Gemma4OutputParser` (always required) | `Gemma4OutputParser` (always required) | `Qwen35OutputParser` | `HarmonyOutputParser` (always required) | `ChatMlOutputParser` | `PassthroughOutputParser` | `MuseGlimmerOutputParser` (always required) |

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
9. Update the capability gates (`_detect_capabilities` in
   `TensorSharp.Server.Host/testdata/test_multiturn.py`) if the model exposes new
   modalities, thinking, or tool capabilities.
<!-- preserved-index-detail:end -->
