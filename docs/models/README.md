# Choose a Model and Explore Its Architecture

[English](README.md) | [中文](README_zh-cn.md)

Start with the output you need, then check the model's inputs, weights and backend requirements. This page is a selection index; each family card retains its downloads, forward graph, implementation and validation details. Check source capabilities, passing measurements and released-package availability separately.

## Start with your task

| What you want to do | Next step |
|---|---|
| Run a first local text model | [Getting started](../getting_started.md) → [Gemma 4 E4B native-GGML lane](gemma4.md#verified-gemma-4-e4b-native-ggml-fast-path); add the matching projector for media input. |
| Use images, documents or tools in a chat | Choose a text family below, then check its modality, tool-protocol and backend limits. |
| Search text, code or media | [Embedding guide](../embeddings.md): Snowflake, MiniLM and EmbeddingGemma 2 output vectors; media support depends on the encoder. |
| Generate or edit images | [Qwen-Image-2.1](qwenimage21.md), with its matching files and sampling recipe; [MiniMax-H3-Image](minimax-h3-image.md) for text to image up to ~16 MP. |
| Generate video, with or without audio | [LTX 2.5](ltx-2.5.md), [MiniMax-H3](minimax-h3.md) or [Wan](wan.md); compare the outputs below. |
| Get files for a model you have chosen | [Model downloads](../../MODEL_DOWNLOADS.md) and its card; check the named quantization, memory and companion files. |

## Reading support status

- **Implemented:** a code path exists; device and checkpoint coverage is stated in the card.
- **Conditional:** a particular variant, backend, companion file or setting is required.
- **Experimental:** the feature or model has explicit implementation or validation limits.
- **Fixture-only:** checks use small synthetic or component fixtures; they do not qualify a full checkpoint.
- **Measured:** evidence applies to the recorded checkpoint, hardware, workload and build.
- **Unsupported:** the feature is refused or unimplemented.

Media in the input column is content the model receives; the output column distinguishes text, decisions, images, video and vectors. Media usually requires the companion files named in the card. The table does not certify every device.

## Text, multimodal input and decisions

| Family / card | Inputs | Output / task | Conditions and evidence |
|---|---|---|---|
| [Gemma 4](gemma4.md) | Text, image, video, audio | Text | Conditional: matching projector for media. Start with the E4B Q8_0 native-GGML lane. |
| [Qwen 3.5 / 3.6 / 3.8 27B](qwen35.md) | Text, image | Text | Conditional: NextN needs a GGUF retaining its draft block; check variant and backend. |
| [Qwen 3.8 Flash Next](qwen38-flash-next.md) | Text, image, video | Text | Conditional: memory placement, tensor parallelism and the separate MTP head depend on quantization/backend. |
| [Bonsai2 27B](bonsai2.md) | Text, image with projector | Text | Conditional: single-device GGML only. Recorded Metal and bounded CPU checks do not qualify other devices. |
| [GPT OSS](gptoss.md) | Text | Text | Implemented: thinking mode and tools; check the card for backend and cache requirements. |
| [Nemotron-H / Omni](nemotron.md) | Text; image on Omni; conditional audio | Text | Conditional: audio needs a Parakeet companion absent from public GGUFs. Speculation is unsupported. |
| [Mistral 3](mistral3.md) | Text, image | Text | Unsupported: explicit thinking mode and tools. Check the named checkpoint/backend coverage. |
| [Hunyuan Dense](hunyuan-dense.md) | Text | Text | Implemented: single-device text path; tools, tensor parallelism and layer split are unsupported. |
| [GLM 5.x](glm.md) | Text; image only on 5.3-Flash | Text | Conditional: 5.2/5.3 are text-only. NextN excludes active tensor parallelism; the CPU reference has numerical limits. |
| [DeepSeek V4 Flash](deepseek4.md) | Text | Text | Conditional: check memory and placement; optional DSpark speculation requires a matching drafter. |
| [DeepSeek V4.1 Flash / 1017](deepseek41.md) | Text; conditional image/video | Text | Experimental: routed-MoE TP/DSpark. Pure-C# CPU coverage is fixture-only; default 1017 YOCO prefill is approximate. |
| [Muse Spark](muse-spark.md) | Text, image with Spark projector | Text | Conditional: multi-GPU or large host memory. Recorded CUDA measurements do not qualify every backend; audio is unsupported. |
| [MiniMax-M3.1-Flash](minimax-m3.md) | Text, image with a converted projector | Text | Conditional: `ggml_cuda` and several GPUs (232 GiB UD-Q4_K_XL); the projector and DSpark drafter must be converted. Recorded 8x A40 measurements do not qualify other backends; audio is unsupported. |
| [Muse-Glimmer](muse-glimmer.md) | Text, image | Text | Conditional: projector for images, ATEM for tools, separate DFlash drafter for optional speculation; check backend limits. |
| [DiffusionGemma](diffusiongemma.md) | Text, image | Text | Unsupported: chat audio, video_url and tools. Web UI video is supplied as sampled images. |
| [Jev (DiffusionGemma)](jev.md) | Text, images, extracted documents/frames/transcripts | Typed decisions | Conditional: sampled video and configured ASR transcripts; see ingestion and probability limits. |

DeepSeek 1017 defaults to approximate YOCO prefill; read its [quality and benchmark conditions](deepseek41.md#unsloths-deepseek-v41-flash-1017-ggufs) before choosing exact settings. Published Nemotron Omni companions omit the Parakeet audio tower. Jev's document, video and audio ingestion produces extracted content, sampled frames or transcripts; its card records the coverage limits.

## Image and video output

| Family / card | Inputs | Output / task | Conditions and evidence |
|---|---|---|---|
| [Qwen-Image-2.1](qwenimage21.md) | Text, optional reference/mask images | Images / image edits | Conditional: matching encoder/VAE and sampling recipe; component coverage varies by backend. |
| [MiniMax-H3-Image](minimax-h3-image.md) | Text | Images, ~4 to 20 MP | Conditional: ggml backends only (`ggml_cuda` measured); six distilled steps, guidance 1; editing not implemented. |
| [LTX 2.5 Distilled](ltx-2.5.md) | Text, image keyframes; optional identity sheets | Video + 48 kHz stereo | Experimental: current two-stage recipe and companions; Vulkan and multi-GPU are unsupported. |
| [MiniMax-H3](minimax-h3.md) | Text, keyframes or reference media by checkpoint | Video + 32 kHz stereo | Conditional: distinct FL2VA/Ref2VA checkpoints. Component gates and whole-generation coverage differ. |
| [Wan 2.1 / 2.2](wan.md) | Text, image by variant | Video (no audio) | Conditional: variant-specific encoders/VAE, sampler and step schedule; multi-GPU is unsupported. |

## Detailed reference and architecture development

<a id="what-every-card-contains"></a>
<a id="verified-start-lane"></a>
<a id="implementation-matrix"></a>
<a id="backend-notes"></a>
<a id="continuous-batching--paged-kv-cache"></a>
<a id="architecture-comparison"></a>
<a id="adding-a-new-architecture"></a>

The full [implementation matrix](capability-reference.md#implementation-matrix), [backend notes](capability-reference.md#backend-notes), [batching and cache details](capability-reference.md#continuous-batching--paged-kv-cache), [architecture comparison](capability-reference.md#architecture-comparison) and [new-architecture steps](capability-reference.md#adding-a-new-architecture) now live in the detailed reference in this directory. The original detailed material is retained there, while family cards remain the entry point for a single model.

For speed comparisons, read the [latest scoped benchmarks](../benchmarks.md). Before deployment, check [project status](../PROJECT_STATUS.md) and [backend/feature support](../supported_models.md). For a continuous path from tensors to inference, see the [companion book guide](../BOOK.md).
