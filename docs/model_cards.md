# Model Architecture Cards

[English](model_cards.md) | [中文](model_cards_zh-cn.md)

> This file has been split. Each model now has a dedicated card under
> [`docs/models/`](models/README.md), and there is a per-model 中文 version
> alongside each English card.

Use the index below to jump straight into a specific architecture, or read
[`docs/models/README.md`](models/README.md) for the full implementation
matrix and feature comparison.

Most cards begin with download pointers — checked Hugging Face repositories
for most families, plus exact file sizes and SHA-256 hashes for the Bonsai and
Bonsai2 files — exact example filenames, and copy/paste
`TensorSharp.Cli` and `TensorSharp.Server.Host` commands. The DeepSeek V4, GLM
and MiniMax-H3 cards give their repositories further down, and the Muse-Glimmer
and Hunyuan Dense cards link none; the
[implementation matrix](models/README.md#implementation-matrix) lists
Muse-Glimmer's. The verified quick-start lane (see the repository
[README](../README.md#quick-start)) is TensorSharp's Gemma 4 E4B Q8_0 native
GGML family/path tier: use `gemma-4-E4B-it-Q8_0.gguf` from the recommended
public
[ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)
with `ggml_cuda`, `ggml_metal`, or `ggml_vulkan`; see the
[Gemma 4 card](models/gemma4.md#verified-gemma-4-e4b-native-ggml-fast-path).
The matching `mmproj` is optional for text and required for image, video, or
audio input; no particular public-file checksum is asserted as the benchmark
input.

| Architecture | GGUF arch keys | What it does | English card | 中文卡片 |
|---|---|---|---|---|
| DeepSeek V4 Flash | `deepseek4` | Sparse-MoE text model with compressed attention; DSpark block speculative decoding via a separate `--draft-model` GGUF | [models/deepseek4.md](models/deepseek4.md) | [models/deepseek4_zh-cn.md](models/deepseek4_zh-cn.md) |
| DeepSeek V4.1 Flash | `deepseek41` | Sparse-MoE text model on a dedicated native V4.1 graph, with Engram tables read straight from the GGUF and an optional image/video vision companion; serves on `ggml_cuda`, with `ggml_cpu` and `cpu` as correctness paths; an experimental `deepseek41-dspark` drafter loads via `--draft-model` | [models/deepseek41.md](models/deepseek41.md) | [models/deepseek41_zh-cn.md](models/deepseek41_zh-cn.md) |
| GLM 5.x | `glm-dsa`, `glm_dsa`, `glm5next` | MoE text models with MLA and the DSA lightning indexer (GLM-5.2, GLM-5.3), with embedded NextN drafting under `--spec`; GLM-5.3-Flash (`glm5next`) is a KDA/MLA hybrid with image input | [models/glm.md](models/glm.md) | [models/glm_zh-cn.md](models/glm_zh-cn.md) |
| Gemma 4 | `gemma4` | Dense and MoE text + image + video + audio chat, thinking, tools; MTP speculative decoding with a separate `gemma4-assistant` draft GGUF | [models/gemma4.md](models/gemma4.md) | [models/gemma4_zh-cn.md](models/gemma4_zh-cn.md) |
| DiffusionGemma | `diffusion-gemma`, `diffusion_gemma` | Text **diffusion** generation — an EntropyBound denoising sampler instead of autoregressive decode — with image input through the Gemma-4 vision tower (audio and `video_url` video are refused; a Web UI video reaches it only as plain image frames). The same model serves the [Jev](models/jev.md) typed-decision API (`POST /v1/systemone`), which also accepts documents, sampled video frames and speech through a configured ASR companion | [models/diffusiongemma.md](models/diffusiongemma.md) | [models/diffusiongemma_zh-cn.md](models/diffusiongemma_zh-cn.md) |
| Qwen 3.5 / 3.6 family | `qwen35`, `qwen35moe`, `qwen3next` | Hybrid full-attention + GatedDeltaNet text + image chat, dense or MoE; Qwen 3.6 embeds a NextN draft block for speculative decoding | [models/qwen35.md](models/qwen35.md) | [models/qwen35_zh-cn.md](models/qwen35_zh-cn.md) |
| Qwen 3.8 Flash Next | `qwen4exp` | Hybrid MoE text + image + video chat with thinking and tools, run as one fused graph per token on GGML with a multi-GPU layer split; per-token MTP speculative decoding with a separate shared MTP head via `--draft-model` | [models/qwen38-flash-next.md](models/qwen38-flash-next.md) | [models/qwen38-flash-next_zh-cn.md](models/qwen38-flash-next_zh-cn.md) |
| Bonsai Q1_0 | `qwen3` (8B), `qwen35` (27B) | Two hash-pinned Q1_0 files, downloaded from pinned prism-ml Hugging Face revisions, that share a name but not an architecture: a dense Qwen 3 decoder and a dense Qwen 3.5 hybrid. Text only; TensorAgent keeps them sideload-only | [models/bonsai.md](models/bonsai.md) | [models/bonsai_zh-cn.md](models/bonsai_zh-cn.md) |
| Bonsai2 27B | `qwen35` plus `prism.hadamard.*` | Dense Qwen 3.5 hybrid with PRISM signed Hadamard rotations and PQ2_0 / PTQ1_0 low-bit weights, transcoded to Q2_0 at load; text + image with a companion projector; single-device GGML backends only | [models/bonsai2.md](models/bonsai2.md) | [models/bonsai2_zh-cn.md](models/bonsai2_zh-cn.md) |
| GPT OSS | `gptoss`, `gpt-oss` | MXFP4 MoE text model with attention sinks and Harmony thinking/tools | [models/gptoss.md](models/gptoss.md) | [models/gptoss_zh-cn.md](models/gptoss_zh-cn.md) |
| Nemotron-H | `nemotron_h`, `nemotron_h_moe`, `nemotron_h_omni` | Hybrid Mamba2 SSM + attention + (MoE) FFN text model; the Omni checkpoints add image input, and audio with a companion GGUF carrying the Parakeet tower. Speculative decoding (including Nemotron 3.5 Lightning's DSpark drafter) is refused because the trunk's verify and decode kernels disagree | [models/nemotron.md](models/nemotron.md) | [models/nemotron_zh-cn.md](models/nemotron_zh-cn.md) |
| Mistral 3 | `mistral3` (and `llama`-labelled Mistral Small 3.x files) | Dense text + image chat with YaRN-corrected RoPE and the Pixtral vision encoder | [models/mistral3.md](models/mistral3.md) | [models/mistral3_zh-cn.md](models/mistral3_zh-cn.md) |
| Hunyuan Dense | `hunyuan-dense` | Tencent's dense Hunyuan decoders, such as the Hy-MT2 releases; text only, single device, no tools and no thinking | [models/hunyuan-dense.md](models/hunyuan-dense.md) | [models/hunyuan-dense_zh-cn.md](models/hunyuan-dense_zh-cn.md) |
| Muse-Glimmer | `muse-glimmer`, `muse_glimmer` | Interleaved-SWA text + image chat with thinking and ATEM tools; DFlash / DFlash2 block speculative decoding via a separate `--draft-model` GGUF | [models/muse-glimmer.md](models/muse-glimmer.md) | [models/muse-glimmer_zh-cn.md](models/muse-glimmer_zh-cn.md) |
| Qwen-Image-2.1 | `qwen_image`, `qwen-image` | Text-to-image and reference-image editing; Qwen3-VL-8B encoder and dedicated 2.1 VAE; unmerged LoRA plug-ins, a default-on prefix KV cache, and diffusion-transformer tensor parallelism on GGML CUDA / Vulkan | [models/qwenimage21.md](models/qwenimage21.md) | [models/qwenimage21_zh-cn.md](models/qwenimage21_zh-cn.md) |
| MiniMax-H3 | `minimax-h3`, `minimax_h3` | **Joint audio-video generation** — prompt (+ optional keyframes or references) → video **and native 32 kHz stereo audio generated together in one packed latent**, by a single diffusion transformer. Text-to-video, image-to-video, first/last frame and reference-to-video, all CFG-free at 4-8 steps | [models/minimax-h3.md](models/minimax-h3.md) | [models/minimax-h3_zh-cn.md](models/minimax-h3_zh-cn.md) |
| Wan video | `wan`, `wan2.1`, `wan2.2` | **Video generation, video only** — prompt (+ optional first frame) → H.264 MP4, Wan 2.1 T2V and Wan 2.2 TI2V-5B / A14B; a step-distilled checkpoint turns the 100-DiT-pass recipe into 4 | [models/wan.md](models/wan.md) | [models/wan_zh-cn.md](models/wan_zh-cn.md) |

The Qwen 3 / Qwen 2 / Qwen 2.5-VL plug-in (`qwen3`, `qwen2`, `qwen2vl`,
`qwen2_vl`; text only) also loads but has no card of its own; see the note under
the [implementation matrix](models/README.md#implementation-matrix).

Each card walks an engineer or researcher from "I have never heard of this
model" to "I can explain the forward graph and reproduce the inference path
in TensorSharp", covering:

1. Checked downloads and runnable CLI/server commands
2. Origin and intent (provider, GGUF arch keys, modalities, thinking, tools)
3. Model architecture (high-level block diagram, layer counts, per-layer heterogeneity)
4. Forward graph (the exact ordered list of ops, per-token decode and multi-token prefill)
5. Components in detail (attention, FFN/SSM, routing, normalization, RoPE flavor, vision/audio encoder)
6. Parameters and settings (GGUF metadata keys, weight tensor naming, dtype expectations)
7. TensorSharp implementation walkthrough
8. Prefill optimization
9. Decode optimization
10. Memory and KV cache strategy
11. Multimodal pipeline (when applicable)
12. Output parser and chat template
13. Optimization opportunities

When adding a new architecture, follow the checklist in
[`docs/models/README.md`](models/README.md#adding-a-new-architecture).
