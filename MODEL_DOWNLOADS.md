# Model Downloads (GGUF)
[English](MODEL_DOWNLOADS.md) | [中文](MODEL_DOWNLOADS_zh-cn.md)

> Part of the [TensorSharp](README.md) documentation. See also the [per-model architecture cards](docs/models/README.md).


TensorSharp loads models in GGUF format. Below are verified Hugging Face repos for every supported architecture, including the multimodal-projector (mmproj) and MTP-draft companion files each family uses. Pick a quantization that fits your hardware (Q4_K_M / UD-Q4_K_XL for low memory, Q8_0 for higher quality, etc.). Rows marked *optional* are the speed artifacts — step-distilled checkpoints and speculative-decoding drafters. Nothing breaks without them, but they are usually the difference between minutes and hours, so skim them before you start a long download.

| Architecture | Model | GGUF Download |
|---|---|---|
| Embedding encoder (`bert` / XLM-R) | Snowflake Arctic Embed L v2.0, Q8_0, 1024 dimensions | [fisher046/snowflake-arctic-embed-l-v2.0-Q8_0-GGUF](https://huggingface.co/fisher046/snowflake-arctic-embed-l-v2.0-Q8_0-GGUF), file `snowflake-arctic-embed-l-v2.0-q8_0.gguf`; about 635 MB; use `--embeddings`. Pinned revisions, checksums, and examples: [guide](docs/embeddings.md) |
| Embedding encoder (`bert`) | all-MiniLM-L6-v2, Q8_0, 384 dimensions | [second-state/All-MiniLM-L6-v2-Embedding-GGUF](https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF), file `all-MiniLM-L6-v2-Q8_0.gguf`; about 25 MB; use `--embeddings` |
| Gemma 4 verified native tier | gemma-4-E4B-it Q8_0 | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) — recommended public artifact `gemma-4-E4B-it-Q8_0.gguf`; lower-memory Q4_K_M is also available; mmproj `mmproj-gemma-4-E4B-it-Q8_0.gguf` is in the same repo |
| Gemma 4 | gemma-4-12B-it (QAT) | [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) — mmproj `mmproj-BF16.gguf` and MTP draft `mtp-gemma-4-12B-it.gguf` in the same repo |
| Gemma 4 | gemma-4-26B-A4B-it (MoE, QAT) | [unsloth/gemma-4-26B-A4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF) — mmproj `mmproj-BF16.gguf` and MTP draft `mtp-gemma-4-26B-A4B-it.gguf` in the same repo |
| Gemma 4 | gemma-4-26B-A4B-it (MoE) | [ggml-org/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF) — mmproj files in the same repo |
| Gemma 4 | gemma-4-31B-it | [ggml-org/gemma-4-31B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF) — mmproj files in the same repo |
| Gemma 4 | `gemma4-assistant` MTP drafts (optional — speculative decoding) | [AtomicChat/gemma-4-E4B-it-assistant-GGUF](https://huggingface.co/AtomicChat/gemma-4-E4B-it-assistant-GGUF) (E4B) and [AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF](https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF) (26B-A4B) — load via `--draft-model`, which enables speculation by itself; pair each draft with its matching target size |
| Qwen 3.5 / 3.6 family | Qwen3.5-9B | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) — mmproj `mmproj-F16.gguf` in the same repo |
| Qwen 3.5 / 3.6 family | Qwen3.5-35B-A3B (MoE) | [ggml-org/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF) — mmproj `mmproj-Qwen3.5-35B-A3B-Q8_0.gguf` in the same repo |
| Qwen 3.5 / 3.6 family | Qwen3.6-35B-A3B (MoE, embedded NextN MTP) | [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) — these GGUFs retain the NextN block for `--spec`; mmproj `mmproj-F16.gguf` in the same repo. The base repo [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) ships the same file names with NextN stripped — those load fine but silently fall back to standard decode |
| Qwen 3.5 / 3.6 family | Qwen3.8-27B (dense hybrid, embedded NextN MTP, image-capable) | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) — e.g. `Qwen3.8-27B-UD-Q4_K_XL.gguf`, the file [`config/agent-qwen3.8-27b.json`](config/agent-qwen3.8-27b.json) pins by commit and SHA-256; it keeps the NextN block for `--spec`, and mmproj `mmproj-BF16.gguf` is in the same repo (name it with `--mmproj`). It runs on the same `Qwen35Model` path as Qwen 3.5. Optional speed artifact: the DFlash2 block drafter [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF), loaded with `--draft-model` (when attached it replaces the NextN block); what it gains depends on the workload, see §12.4 of [qwen35.md](docs/models/qwen35.md) |
| Qwen 3.8 Flash Next | Qwen3.8-Flash-Next (hybrid MoE, image-capable) | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) — one subdirectory per quant (`UD-Q2_K_XL/`, …), each a multi-shard set; point `--model` at the `-00001-of-` shard. `mmproj-BF16.gguf` (same repo) enables image input, multi-image prompts and multi-turn image sessions included: the CLI finds it beside the model, the server needs it named with `--mmproj`. `general.architecture` = `qwen4exp`. On a multi-GPU box `--tp N` runs a **layer split** — whole layers per GPU, the same (and only) multi-GPU mode llama.cpp offers this architecture — which buys capacity, not speed; see [USAGE.md](USAGE.md#tensor-parallelism--distributed-inference). Optional speed artifact: the shared MTP head `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (take the `-shared-` file, not the other `MTP/mtp-*` heads beside it: the loader accepts only a shared-target NextN block; 2,786,568,256 bytes; the validation package in `eng/validation/qwen38_mtp_followup/` records it from this repo at revision `38bb39ee97821de2c9009abb7e93950eec396e66`), a single GGUF loaded with `--draft-model` on a GGML backend. Measured on UD-Q2_K_XL over three A40s, it pays on copy-heavy output (about 1.7x on a 192-token code-copy stream with the exact verify-row kernels added on 2026-09-17: 83.2 against 49.1 tok/s plain; 1.75-1.96x before that change) but made 512-token prose decode slower (44-46 against 52 tok/s, measured before that change); see [qwen38-flash-next.md](docs/models/qwen38-flash-next.md#speculative-decoding-with-the-shared-mtp-head) |
| Qwen 3 / Qwen 2 / Qwen 2.5-VL | `qwen3`, `qwen2`, `qwen2vl` checkpoints (text only) | No repository is pinned here: any GGUF whose `general.architecture` is `qwen3`, `qwen2`, `qwen2vl` or `qwen2_vl` loads through `Qwen3Model`. Text only: no projector is loaded, so a Qwen 2.5-VL file chats without its vision tower. No drafter, and n-gram speculation does not run on it either. Bonsai-8B below is a `qwen3` file. See [docs/models/README.md](docs/models/README.md) |
| Bonsai Q1_0 | Bonsai-8B / Bonsai-27B | Publisher files with the exact documented hashes are available from [prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf/tree/48516770dd04643643e9f9019a2a349cf26c5dbd) and [prism-ml/Bonsai-27B-gguf](https://huggingface.co/prism-ml/Bonsai-27B-gguf/tree/f10afb355f104535e3e3e98cf7ab7795c72bd292); both publisher cards declare Apache-2.0. Download the `Q1_0` file and verify its SHA-256. The files use different architectures (`qwen3` for 8B, `qwen35` for 27B); exact revisions, download commands and existing measurements are in [bonsai.md](docs/models/bonsai.md). TensorAgent's catalog still uses manual import. |
| Bonsai2 | Ternary-Bonsai-2-27B (PQ2_0 / PTQ1_0, image-capable) | No download repository is recorded here: bring `Ternary-Bonsai-2-27B-PQ2_0.gguf` (7.21 GB) or `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5.95 GB) and check it against the SHA-256 in [bonsai2.md](docs/models/bonsai2.md). Both declare `general.architecture` = `qwen35` and carry PRISM signed-Hadamard metadata (`prism.hadamard.*`) with the custom tensor types PQ2_0 / PTQ1_0, which TensorSharp transcodes losslessly to GGML Q2_0 at load (about 6% / 29% larger in memory than the stored payload). The companion projectors `Ternary-Bonsai-2-27B-mmproj-BF16.gguf` / `-mmproj-Q8_0.gguf` enable image input: the CLI finds them beside the model, the server takes `--mmproj`. Single-device GGML backends only (`cpu`, `cuda`, `mlx` and `--tp` are refused); end-to-end validation so far is on Metal (M5 Pro) only. Not in TensorAgent's catalog |
| GPT OSS | gpt-oss-20b (MoE) | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) — `gpt-oss-20b-MXFP4.gguf` (note the uppercase `MXFP4`), text only, no companion files |
| Nemotron-H | Nemotron-H-8B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF) |
| Nemotron-H | Nemotron-H-47B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF) |
| Nemotron-H | Nemotron 3 Nano Omni 30B-A3B (image-capable) | [unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF) — mmproj `mmproj-BF16.gguf` (same repo) is required for image input. Audio input is refused (HTTP 400 / CLI error) unless a separately converted audio companion GGUF is loaded: these GGUFs ship no audio tower, see [nemotron.md §4.6-4.7](docs/models/nemotron.md) |
| Nemotron 3.5 | Nemotron-3.5-Lightning-30B-A3B (hybrid 23 Mamba-2 + 23 MoE + 6 attention) | [unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) — e.g. `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MXFP4_MOE.gguf` (MoE experts MXFP4, ~17 GB); `general.architecture` = `nemotron_h_moe`. Smaller/other quants: [ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) (BF16/Q4_0/Q8_0 + separate MTP GGUFs). The DSpark drafter in the next row is listed for reference only: speculation is refused on this trunk |
| Nemotron 3.5 | DSpark speculative drafter — **not used** | [magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark-GGUF](https://huggingface.co/magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark-GGUF) (the llama.cpp DFlash export of the official DSpark module). TensorSharp refuses speculative decoding on Nemotron-H, so `--draft-model` does not attach it: the trunk's verify and decode kernels disagree and a speculative stream would differ from plain decoding (see [speculative_decoding.md](docs/speculative_decoding.md#nemotron-h-refuses-speculation)) |
| Mistral 3 | Mistral-Small-3.1-24B-Instruct-2503 | [bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF) — Pixtral mmproj `mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf` in the same repo |
| Hunyuan Dense | Tencent dense Hunyuan checkpoints (`hunyuan-dense`) | Any GGUF whose `general.architecture` is `hunyuan-dense`, such as the Hy-MT2 releases (`tencent/Hy-MT2-1.8B` supplies the reference chat template). Text only, single device, no projector and no drafter. See [hunyuan-dense.md](docs/models/hunyuan-dense.md) |
| Muse-Glimmer | Muse-Glimmer-30B (dense, image-capable) | [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) — e.g. `Muse-Glimmer-30B-UD-Q4_K_XL.gguf` or `Muse-Glimmer-30B-Q8_0.gguf`; `general.architecture` = `muse-glimmer` / `muse_glimmer`. Image input requires `mmproj-Muse-Glimmer-30B-Q8_0.gguf` (same repo): the CLI finds it beside the model when you pass `--image`, and the server needs it named with `--mmproj`. Optional speed artifacts: the DFlash block drafter `dflash-kquant.gguf` (same repo) or the newer DFlash2 drafter [z-lab/Muse-Glimmer-30B-DFlash2-GGUF](https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2-GGUF) (prefer `-Q4_K_M` on a 16 GB card — see the note on drafter size in [speculative_decoding.md](docs/speculative_decoding.md#what-to-expect)), loaded with `--draft-model` for lossless speculative decoding — pass no sampler flags, it needs plain greedy |
| DeepSeek V4.1 | DeepSeek-V4.1-Flash (`deepseek41`, 384 routed experts) | [vcruz305/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF/tree/58d8ac86298fdf85a2440defee08b1abcad32e45) at revision `58d8ac86298fdf85a2440defee08b1abcad32e45` — seven Q2_K shards (246.35 GiB, mixed Q2_K/Q3_K tensors) kept in one directory; point `--model` at the first shard. Engram weights and hash constants are already embedded in the GGUF; no Engram preparation or separate Engram file is needed. `eng/dsv41-prepare-vision.py` builds the optional ~970 MB vision companion that `--mmproj` needs for images and video. `ggml_cuda` is the serving backend; `ggml_cpu` and `cpu` are correctness and portability paths rather than serving ones — `--backend cpu` runs the whole V4.1 graph on the pure-C# `DeepSeek4CpuExecutor`, with no ggml, no native library and no GPU; it reads the embedded Engram metadata, and the vision companion does not follow it there (`LoadVisionEncoder` throws), so images and video are not available on that backend. V4 drafters are rejected; a `deepseek41-dspark` drafter is accepted through `--draft-model` on `ggml_cuda` / `ggml_cpu` as an experimental path, validated only on synthetic test fixtures — no trained V4.1 drafter has been measured. Full recipe and checkpoint hashes: [deepseek41.md](docs/models/deepseek41.md) |
| DeepSeek V4 | DeepSeek-V4-Flash-0731 (284B MoE) | [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) — one subdirectory per quant (`UD-Q8_K_XL/`, `UD-IQ4_XS/`, `UD-IQ1_S/`, …), each a multi-shard set; point `--model` at the `-00001-of-` shard. Text only |
| GLM 5.x | GLM-5.2 (744B-A40B MoE, embedded NextN MTP) | [unsloth/GLM-5.2-GGUF](https://huggingface.co/unsloth/GLM-5.2-GGUF) — one subdirectory per quant (`UD-Q4_K_XL/`, `UD-IQ2_XXS/`, …), each a multi-shard set; point `--model` at the `-00001-of-` shard. **Text only** — GLM-5.3-Flash, two rows down, is the one that takes images; the GLM-5.3 row in between is text-only as well. These GGUFs already carry the NextN block for `--spec` — unlike Qwen 3.6 there is no separate MTP repo to pick |
| GLM 5.x | GLM-5.3 (`glm-dsa`, 256 routed experts, text only) | [unsloth/GLM-5.3-GGUF](https://huggingface.co/unsloth/GLM-5.3-GGUF) — one subdirectory per quant (`UD-Q2_K_XL/`, …), each a multi-shard set; point `--model` at the `-00001-of-` shard. `general.architecture` = `glm-dsa`, and the block shape matches GLM-5.2 (79 blocks — 78 trunk plus one NextN — 256 routed experts top-8, MLA with the lightning indexer, rope base 8e6), so it loads on the existing GLM-5.2 path with nothing new to enable. **Text only** — this repo publishes no mmproj at all, unlike the Flash one below. It does carry the NextN block for `--spec`, but `blk.78` ships no `nextn.shared_head_head.weight` of its own, so the draft block borrows the trunk's LM head — which is column-parallel under `--tp N`. The loader refuses to draft from one rank's strip of the vocabulary and says so on stderr, so `--spec` is only engaged when you run **without** `--tp`, i.e. on the default layer split across every visible GPU |
| GLM 5.x | GLM-5.3-Flash (320B, 288 routed experts, text + image) | [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) — one subdirectory per quant (`UD-Q2_K_XL/`, …), each a multi-shard set; point `--model` at the `-00001-of-` shard. `general.architecture` = `glm5next`, and it loads through the same native executor as GLM-5.2. Unlike 5.2 it **takes images**: `mmproj-BF16.gguf` (the GLM-OCR ViT, same repo) enables `--image`, multi-image prompts and multi-turn image sessions. Its NextN block is not wired up yet, so there is no `--spec` here. Omitting `--tp` uses the default layer split across every visible GPU; on GGML GPU backends, `--tp N` selects native local/single-process tensor parallelism |
| DeepSeek V4 | DSpark speculative drafters (optional — speed only) | see [DSpark drafters](#dspark-drafters) below — a separate GGUF loaded with `--draft-model` for ~1.3-1.4x decode |
| DiffusionGemma | diffusiongemma-26B-A4B-it | [unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) (`general.architecture` = `diffusion-gemma`); a smaller Q3_K_M, which unsloth does not publish, is in [DevQuasar/google.diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/DevQuasar/google.diffusiongemma-26B-A4B-it-GGUF). Every published GGUF is text-only. For image input also download the Gemma-4 vision tower, which TensorSharp reads straight from the upstream shard `model-00011-of-00011.safetensors` (2.84 GB) in [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it), and pass it with `--mmproj` (there is no auto-detection for this family). Audio is refused, and there is no video path: an OpenAI `video_url` part is refused, and a video uploaded in the Web UI reaches the model only as extracted frames, treated as plain images. The `config/diffusiongemma-26b-a4b-*.json` files fetch both; see [diffusiongemma.md](docs/models/diffusiongemma.md) |
| Qwen-Image-2.1 | Diffusion transformer (the `--model` GGUF) | [Abiray/Qwen-Image-2.1-GGUF](https://huggingface.co/Abiray/Qwen-Image-2.1-GGUF) — `qwen_image_2.1_Q4_K_M.gguf`. [`config/qwen-image-2.1.json`](config/qwen-image-2.1.json) downloads this file and the three companions below with pinned revisions and SHA-256 checksums (four files, about 10.29 GiB). Unsloth's metadata-free `qwen-image-2.1-Q8_0.gguf` (zero GGUF metadata, mixed BF16/F32/Q8_0 tensors) also loads: the architecture is recognised from the tensor layout, including the `model.diffusion_model.` prefix, and its `mmproj-BF16.gguf` does not match the companion scan, so pass it with `--qwen-image-mmproj`. Earlier Qwen-Image / Qwen-Image-Edit checkpoints are refused. See [qwenimage21.md](docs/models/qwenimage21.md) |
| Qwen-Image-2.1 | Dedicated 2.1 VAE (required) | `vae/qwen_image_2.1_vae_bf16.safetensors` from [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/vae) — place next to the DiT or point `--qwen-image-vae` / `TS_QWEN_IMAGE_VAE` at it |
| Qwen-Image-2.1 | Qwen3-VL-8B text encoder (required) | `Qwen3VL-8B-Instruct-Q4_K_M.gguf` from [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) — place next to the DiT or set `--qwen-image-vl` / `TS_QWEN_IMAGE_TE` |
| Qwen-Image-2.1 | Vision encoder for editing | `mmproj-Qwen3VL-8B-Instruct-F16.gguf` from the same Qwen3-VL repository — place next to the DiT or set `--qwen-image-mmproj` / `TS_QWEN_IMAGE_MMPROJ` |
| Qwen-Image-2.1 | LoRA plug-ins (optional) | Twelve plug-ins in [`config/lora/`](config/lora/), loaded with `--lora`; each downloads its `.safetensors` from Hugging Face with a pinned revision and SHA-256 on first use, into `qwen-image-2.1/loras/` under the model root (Fun-Acc also fetches the `pdd_config.json` it forwards). Step-distillation adapters (Viggle Turbo, Pruna 8/5-step, Alibaba PAI Fun-Acc 4-step) and style / editing LoRAs; several are non-commercial. See [USAGE.md](USAGE.md#qwen-image-21-lora-plug-ins) |
| MiniMax-H3 audio+video | denoiser (the `--model` GGUF) | **Two separate checkpoints, not settings** — which one you load decides what conditioning it accepts. [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF): `minimax_h3_fl2va_pruned-Q4_K.gguf` (10.64 GiB) for text / image-to-video / first-and-last-frame, or `minimax_h3_ref2va_pruned-Q4_K.gguf` (10.60 GiB) for identity/appearance references. Also Q8_0 (19.97 GiB) down to Q2_K (6.26 GiB). H3 is CFG-distilled: **pass `--cfg 1.0`** and 4-8 steps. The GGUFs carry **no metadata at all**, so TensorSharp identifies them by their tensors, and the partition off the file name — keep `fl2va` / `ref2va` in it if you rename or requantize. Both checkpoints share the three networks below, so adding the second one later costs only its own ~10.6 GiB |
| MiniMax-H3 audio+video | Qwen3-VL-32B text encoder (required) | Same repo: `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` (16.97 GiB), or `-Q2_K_M.gguf` (12.20 GiB) to pair with the two smallest denoisers. Truncated to 50 layers with the final norm removed. Freed before the denoise starts. **It ships no tokenizer** — also download `vocab.json` and `merges.txt` from [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/processor) and put them beside it (or set `TS_VIDEO_TOKENIZER`) |
| MiniMax-H3 audio+video | `vocab.json` + `merges.txt` (required) | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/processor) — the Qwen2 byte-level BPE pair the encoder GGUF omits, and the one thing a config cannot auto-download for you (auto-download fills in options that are flags; the tokenizer is not one). `curl -L -o models/vocab.json https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/42ed227ee7df40d41602854ae760620d6eb651fe/processor/vocab.json` and the same for `merges.txt` |
| MiniMax-H3 audio+video | video VAE (required) | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) — `minimax_h3_video_vae_fp16.safetensors` (5.21 GB). 16x spatial / 4x temporal, with a pure-transformer decoder. Place next to the denoiser or set `--video-vae` |
| MiniMax-H3 audio+video | audio VAE (optional) | Same folder — `minimax_h3_audio_vae_fp32.safetensors` (0.61 GB). Decodes the jointly generated audio latent to 32 kHz stereo, written as a sidecar `.wav`. **Omit it and you still get video**, just silent. Set with `--audio-vae` |
| Wan video generation | **Step-distilled DiT (start here)** | **The single biggest speed lever — pick this unless you are reproducing a reference sample.** A distilled checkpoint runs 4 denoise passes instead of the official recipe's 100 for the same video: measured on M5 Pro / `ggml_metal` at 1088×832×121 frames, **17 m 30 s** end to end versus **3 h 30 m** on the base checkpoint, same request, no other flag changed. TI2V-5B: [hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF](https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF) — `Wan2_2-TI2V-5B-Turbo-Q8_0.gguf` (5.40 GB), also Q6_K (4.22 GB), Q5_K_M (3.82 GB), Q4_K_M (3.44 GB), down to Q2_K (1.86 GB). **Mind the `Wan2_2` underscore** — copying the base repo's `Wan2.2` spelling into `hf download` 404s. I2V-A14B: [jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF](https://huggingface.co/jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF) — Lightning already merged into both experts; download `high_noise/wan2.2_i2v_A14b_high_noise_lightx2v_4step-Q4_K_M.gguf` **and** `low_noise/wan2.2_i2v_A14b_low_noise_lightx2v_4step-Q4_K_M.gguf` (9.66 GB each; Q8_0 15.42 GB, Q2_K 5.31 GB) under one `--local-dir` and point `--model` at either — the sibling expert is found automatically. Secondary: [Green-Sky/FastWan2.2-TI2V-5B-FullAttn-GGUF](https://huggingface.co/Green-Sky/FastWan2.2-TI2V-5B-FullAttn-GGUF) (`FastWan2.2-TI2V-5B-q8_0.gguf`, 5.41 GB). **No flag is needed**: TensorSharp reads the DiT file name for `turbo` / `distill` / `lightning` / `lightx2v` / `fastwan` / `-dmd` or an explicit `<N>steps` (1-16), switches to that step count with guidance off, and prints `step-distilled checkpoint detected -> N steps, guidance off` on load; `--diffusion-steps` / `--cfg` override it. The Turbo and A14B distilled repos ship no VAE and no text encoder — take those from the two rows below |
| Wan video generation | Base DiT (the `--model` GGUF) | The full official recipe (50 steps × 2 CFG passes = 100 DiT passes) — use it when you need to match a reference sample; otherwise prefer the distilled row above. Wan 2.2 text/image-to-video: [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF) (`Wan2.2-TI2V-5B-Q8_0.gguf` 5.40 GB or `Wan2.2-TI2V-5B-Q4_K_M.gguf` 3.43 GB; bundles `VAE/Wan2.2_VAE.safetensors`), [QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF) or [QuantStack/Wan2.2-T2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF) (both `HighNoise/` **and** `LowNoise/` experts are required; each repo bundles `VAE/Wan2.1_VAE.safetensors`); Wan 2.1 text-to-video: [samuelchristlie/Wan2.1-T2V-1.3B-GGUF](https://huggingface.co/samuelchristlie/Wan2.1-T2V-1.3B-GGUF) (`Wan2.1-T2V-1.3B-Q8_0.gguf` / `-F16.gguf`) or [city96/Wan2.1-T2V-14B-gguf](https://huggingface.co/city96/Wan2.1-T2V-14B-gguf) (lowercase names, e.g. `wan2.1-t2v-14b-Q8_0.gguf`) — neither 2.1 repo ships a VAE or encoder. `general.architecture` = `wan` / `wan2.1` / `wan2.2`. See [docs/models/wan.md](docs/models/wan.md) |
| Wan video generation | UMT5-XXL text encoder (required, every Wan checkpoint) | [city96/umt5-xxl-encoder-gguf](https://huggingface.co/city96/umt5-xxl-encoder-gguf) — `umt5-xxl-encoder-Q8_0.gguf` (6.04 GB), or `umt5-xxl-encoder-Q5_K_M.gguf` (4.15 GB) / `umt5-xxl-encoder-Q4_K_M.gguf` (3.66 GB) for tighter memory. Turns the prompt into conditioning and is freed before the denoise starts. Place next to the DiT or set `--video-text-encoder` / `TS_WAN_TE` |
| Wan video generation | video VAE (required) | Decodes latents to frames — **which one is decided by the DiT**, not by you: TI2V-5B needs [`Wan2.2_VAE.safetensors`](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/tree/main/VAE) (bundled in the TI2V-5B repo), Wan 2.1 and A14B need `Wan2.1_VAE.safetensors` — bundled as `VAE/Wan2.1_VAE.safetensors` in both QuantStack A14B repos, or standalone as [`wan_2.1_vae.safetensors`](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors). The distilled repos above ship no VAE, so pair them with the matching file from here. Place next to the DiT (a `VAE/` subfolder works) or set `--video-vae` / `TS_WAN_VAE` |

### DSpark drafters

[DSpark](docs/models/deepseek4.md#dspark-speculative-decoding) is DeepSeek's block
speculative-decoding drafter. TensorSharp runs it for **DeepSeek V4** on both GPU engines
(`--backend cuda` and `--backend ggml_cuda`); the drafter is a separate GGUF passed with
`--draft-model`, and greedy output is unchanged because the trunk verifies every block.

Pick ONE of these — all three load as-is (the loader accepts each publisher's tensor/metadata
spelling). Drafters read the trunk's hidden states, so a drafter built from the **same
checkpoint revision** as your model accepts more often:

| Drafter | Size | For | Notes |
|---|---|---|---|
| [bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF](https://huggingface.co/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF) | 7.0 GB | `DSpark-drafter-Q2K-Q8-0731.gguf` for the **0731** release (a non-0731 build is in the same repo) | Q2_K experts + Q8_0 dense; 66% acceptance on `ggml_cuda` (120-token sample), 66-87% per turn in a five-turn chat |
| [sakamakismile/DeepSeek-V4-Flash-DSpark-support-ds4-GGUF](https://huggingface.co/sakamakismile/DeepSeek-V4-Flash-DSpark-support-ds4-GGUF) | 5.6 GB | the pre-0731 `DeepSeek-V4-Flash` release | Smallest, and still ~69% acceptance against the 0731 trunk; fastest of the three on the direct-CUDA engine because its weights are re-read every speculative step |
| [alessandrobologna/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF](https://huggingface.co/alessandrobologna/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF) | 10.9 GB | the **0731** release | MXFP4 experts (lossless repack of the checkpoint's FP4); highest acceptance on `ggml_cuda` (68%), but only 65% on the direct-CUDA engine, where it was slower than the 5.6 GB drafter (30.9 against 34.0 tok/s); most VRAM — it displaces about a whole trunk layer per GPU |

Or build one from any DeepSeek V4 checkpoint that ships the module (only its three `mtp.*`
shards are downloaded, ~11 GB): see
[Getting a drafter](docs/models/deepseek4.md#getting-a-drafter) and
`eng/dsv4-dspark-to-gguf.py`.

**DSpark drafters for Gemma 4 are NOT supported yet.** DeepSeek also released
DSpark drafters for Gemma 4, and community GGUF conversions exist, but they are a
different drafter design — a 5-layer transformer stack with an `fc` fusion over five target
layers (`general.architecture` = `dspark` or `dflash`, `block_size` 7), not DeepSeek V4's
three hyper-connection blocks (`mtp.*`). TensorSharp rejects them against a DeepSeek V4 target
with a clear message rather than mis-loading them. Listed here so you know what exists upstream:

> That 5-layer `fc`-fusion design **is** implemented for Muse-Glimmer — see
> [DFlash speculative decoding](docs/models/muse-glimmer.md#3-dflash-speculative-decoding).
> The drafters below are not wired up because each one needs its target model to expose the
> per-layer input residuals its encoder consumes, and `Gemma4Model` does not. Today
> `MuseGlimmerModel` and `Qwen35Model` (the DFlash/DFlash2 drafters for Qwen 3.5 / 3.8) do;
> `NemotronModel` taps them too, but speculation is refused on that trunk.

| Backbone | Official checkpoint (safetensors) | Community GGUF |
|---|---|---|
| Gemma-4-12B | [deepseek-ai/dspark_gemma4_12b_block7](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7) | [ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF](https://huggingface.co/ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF) (1.9 GB), [williamliao/dspark_gemma4_12b-GGUF](https://huggingface.co/williamliao/dspark_gemma4_12b-GGUF) (IQ4_XS…F16) |
| Gemma-4-26B-A4B | — | [williamliao/dspark_gemma4_26b-a4b-it-GGUF](https://huggingface.co/williamliao/dspark_gemma4_26b-a4b-it-GGUF) (1.2-3.8 GB) |
| Gemma-4-31B | — | [williamliao/dspark_gemma4_31b-it-GGUF](https://huggingface.co/williamliao/dspark_gemma4_31b-it-GGUF) (3.3-11 GB) |

Gemma 4 does have a supported speculative path today — the `gemma4-assistant` MTP drafts in
the table above, via `--draft-model` — and Qwen 3.6, GLM 5.2 and GLM-5.3 have their
embedded NextN blocks (GLM-5.3 drafts only on the default layer split, i.e. without `--tp`).
Those are different drafters from DSpark.

### Download & Run — per-model quick reference

These commands run from the repository root. First install the [.NET 10 SDK](DEVELOPMENT.md#install-the-net-10-sdk) for your platform and run `dotnet build TensorSharp.slnx -c Release`; a runtime-only installation cannot build the binaries used below.

The `hf download` commands need the Hugging Face CLI (`pip install -U huggingface_hub`) and drop every file into `./models`. Reminders that apply to all blocks: the CLI reads its one-shot prompt from a **file** via `--input` (`--prompt` is the Qwen-Image-2.1 image prompt and the video-generation prompt, MiniMax-H3 and Wan alike), samples **greedily** by default, and generates only 100 tokens unless you raise `--max-tokens`; the server always listens on **http://localhost:5000**. Swap `--backend ggml_cuda` for the backend that fits your hardware (see [Pick a Backend](README.md#pick-a-backend)). Create a prompt file first:

```bash
echo "Give me three facts about the Moon." > prompt.txt
```

**DeepSeek V4.1 Flash** — 384 routed experts, served on `ggml_cuda`, uses the Engram metadata embedded in the GGUF ([vcruz305/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF/tree/58d8ac86298fdf85a2440defee08b1abcad32e45))

```bash
# 246 GiB of Q2_K weights across seven shards. Warming host-mapped Engram tables
# uses ~60 GiB of host page cache; tables placed on GPUs do not need that warm.
python3 -m venv /tmp/dsv41-tools
/tmp/dsv41-tools/bin/python -m pip install huggingface_hub
/tmp/dsv41-tools/bin/hf download vcruz305/DeepSeek-V4.1-Flash-GGUF \
    --revision 58d8ac86298fdf85a2440defee08b1abcad32e45 \
    --include "DeepSeek-V4.1-Flash-Q2_K-*.gguf" --local-dir models/deepseek41-q2

# Q4_K_M instead of Q2_K: 415 GiB across eleven shards. Its two Engram tables are
# 51.5 GiB each. On 8x46 GB they stay host mappings and routed experts need CPU
# offload -- the loader prints the --n-cpu-moe N it wants.
#   --include "DeepSeek-V4.1-Flash-Q4_K_M-*.gguf" --local-dir models/deepseek41-q4
# Both quantizations include their Engram constants; no Engram preparation is needed.

# Optional: the ~970 MB vision companion that --mmproj needs for images and video
/tmp/dsv41-tools/bin/python -m pip install numpy==2.0.2 gguf
/tmp/dsv41-tools/bin/python eng/dsv41-prepare-vision.py models/deepseek41-q2 \
    --parent-model models/deepseek41-q2/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf \
    --repository deepseek-ai/DeepSeek-V4.1-Flash \
    --revision dba1be0a40aa45a94ad051997016db3960a90277

dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model models/deepseek41-q2/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf \
    --mmproj models/deepseek41-q2/deepseek41.vision.gguf \
    --backend ggml_cuda --tp 8 --port 5000
```

`--tp N` here selects a **layer split** across N GPUs, not tensor parallelism; `TS_DSV41_TP=N`
opts into the experimental routed-MoE TP, which has measured slower than the split so far.
Add `--n-cpu-moe N` when the weights and context do not fit. Python is needed for
the Hugging Face download CLI and optional vision preparation; inference needs no Python.
For text-only serving, omit the vision commands and `--mmproj`.

`--backend cpu` is not the `ggml_cpu` of the swap-the-backend note above: it runs the whole
V4.1 graph on the pure-C# `DeepSeek4CpuExecutor` — no ggml, no native library, no GPU, so it
runs anywhere .NET runs — and it is a correctness and portability path, not a serving path;
no throughput, load time or resident footprint has ever been measured for a full checkpoint
on it. The CPU executor reads Engram directly from the GGUF. `--mmproj` is not
available there at all (the vision companion is a native ggml component and
`LoadVisionEncoder` throws), so no images and no video, and distributed TP groups, any draft
model or `TS_DSV4_DSPARK`, `TS_DSV41_TP` other than `0` and `TS_DSV41_ENGRAM_DEVICE` other
than `0` are refused before a weight is read. There is also no multi-turn KV prefix reuse and
no per-sequence slots — every diverging turn re-prefills and concurrent requests serialize.

**DeepSeek V4 Flash** — 284B MoE, text only, DSpark speculative decoding ([unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF))

```bash
# ~160 GB of weights: needs several GPUs (layer-split automatically) plus ~7 GB for the drafter
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF --include "UD-Q8_K_XL/*" --local-dir models
hf download bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF DSpark-drafter-Q2K-Q8-0731.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model models/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf \
    --backend ggml_cuda --draft-model models/DSpark-drafter-Q2K-Q8-0731.gguf \
    --input prompt.txt --max-tokens 200 --temperature 0
```

Drop `--draft-model` for plain decode. Speculation is lossless under any sampler, because each verify row is drawn with the run's own sampler;
`--temperature 0` here only makes the output reproducible; `--spec-pmin` tunes how far each block is drafted.

**GLM 5.x** — GLM-5.3 (`glm-dsa`), 256 routed experts, text only, embedded NextN block ([unsloth/GLM-5.3-GGUF](https://huggingface.co/unsloth/GLM-5.3-GGUF))

```bash
# UD-Q2_K_XL is seven shards, 236.4 GiB -- it wants a box whose *combined* VRAM clears
# that plus the KV cache; the measured configuration is eight 46 GB A40s, layer-split
hf download unsloth/GLM-5.3-GGUF --include "UD-Q2_K_XL/*" --local-dir models

dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model models/UD-Q2_K_XL/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf \
    --backend ggml_cuda --spec --port 5000
```

There is no `--mmproj` here: this repo publishes no vision tower at any quant, and an
`--mmproj` on a `glm-dsa` model is warned about and ignored rather than refused, so it is a
text-only run either way. `--spec` has to be on the command line before load, and it is
engaged on the **default layer split** shown above — omitting `--tp` uses every visible GPU.
`--tp N` is accepted on the GGML GPU backends but is local / single-process only
(`--tp-node-id` / `--tp-peers` are refused for the whole GLM family before the model is built)
and replicates the KV cache on every rank; it is not a validated configuration for GLM-5.3,
and under it the loader declines to draft — the trunk LM head that `blk.78` borrows is
column-parallel — and serves standard decode. GLM-5.2 and GLM-5.3-Flash download the same
way from their own repos in the table above. Details: [glm.md](docs/models/glm.md#glm-53-glm-dsa).

**Gemma 4** — text + image/video/audio, thinking, tools, MTP ([ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF))

```bash
hf download ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q8_0.gguf --local-dir models
hf download ggml-org/gemma-4-E4B-it-GGUF mmproj-gemma-4-E4B-it-Q8_0.gguf --local-dir models
hf download AtomicChat/gemma-4-E4B-it-assistant-GGUF gemma-4-E4B-it-assistant.Q8_0.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gemma-4-E4B-it-Q8_0.gguf --mmproj models/mmproj-gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/gemma-4-E4B-it-Q8_0.gguf --mmproj models/mmproj-gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --draft-model models/gemma-4-E4B-it-assistant.Q8_0.gguf
```

(The third download and the `--draft-model` flag are optional — they enable MTP speculative decoding, which works on either host; add the same flag to the CLI line to use it there.)

**Qwen 3.5 / 3.6 family** — text + image, thinking, tools, NextN MTP on 3.6 ([unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF))

```bash
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf --local-dir models
hf download unsloth/Qwen3.5-9B-GGUF mmproj-F16.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/Qwen3.5-9B-UD-Q4_K_XL.gguf --mmproj models/mmproj-F16.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/Qwen3.5-9B-UD-Q4_K_XL.gguf --mmproj models/mmproj-F16.gguf --backend ggml_cuda
```

Qwen 3.6 NextN speculative decoding (either host; download from the **-MTP-** repo — base-repo GGUFs strip the NextN block and silently fall back to standard decode):

```bash
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --local-dir models

dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --backend ggml_cuda --spec
```

**GPT OSS** — text, thinking (always on), tools ([ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF))

```bash
hf download ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gpt-oss-20b-MXFP4.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/gpt-oss-20b-MXFP4.gguf --backend ggml_cuda
```

**Nemotron-H** — text, thinking, tools; image on the Omni distribution ([bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF))

```bash
hf download bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --backend ggml_cuda
```

For image input use the Omni distribution instead: `NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf` + `mmproj-BF16.gguf` from [unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF). Audio input is refused unless an audio companion GGUF converted from NVIDIA's checkpoint is loaded: the GGUFs ship no Parakeet/FastConformer audio tower (see [nemotron.md §4.6-4.7](docs/models/nemotron.md)).

**Hunyuan Dense** — Tencent dense Hunyuan / Hy-MT2, text only, single device

No repository is pinned here: any GGUF whose `general.architecture` is `hunyuan-dense` loads,
and the architecture is selected from that key rather than from the file name.

```bash
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model models/<your-hunyuan-dense>.gguf \
    --backend ggml_cuda --input prompt.txt --max-tokens 200
```

No projector, no drafter, and extra GPUs stay idle — startup says so rather than using them.

**Mistral 3** — text + image (Pixtral) ([bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF))

```bash
hf download bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --local-dir models
hf download bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --mmproj models/mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --mmproj models/mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf --backend ggml_cuda
```

**DiffusionGemma** — block text-diffusion, text + image input ([unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF))

```bash
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF diffusiongemma-26B-A4B-it-Q4_K_M.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --input prompt.txt --max-tokens 256 --diffusion-steps 48 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --backend ggml_cuda
```

(The Web UI streams live denoising previews for DiffusionGemma; the compat APIs return the final text.)

For image input, add the vision tower: every published GGUF is text-only, so TensorSharp reads the Gemma-4 tower straight from the upstream Hugging Face shard. Name it with `--mmproj` on either host (this family has no projector auto-detection). Audio is refused, and there is no video path: an OpenAI `video_url` part is refused, and a video uploaded in the Web UI reaches the model only as extracted frames, treated as plain images.

```bash
hf download google/diffusiongemma-26B-A4B-it model-00011-of-00011.safetensors --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --mmproj models/model-00011-of-00011.safetensors --image photo.png --input prompt.txt --max-tokens 256 --backend ggml_cuda
```

**Qwen-Image-2.1** — prompt → image, or prompt + one or more reference images → edited image; needs the DiT, the dedicated 2.1 VAE and the Qwen3-VL-8B text encoder, plus its mmproj for editing ([Abiray/Qwen-Image-2.1-GGUF](https://huggingface.co/Abiray/Qwen-Image-2.1-GGUF))

The shortest route is the ready-made config: it pins revisions and SHA-256 checksums and downloads whatever is missing (four files, about 10.29 GiB) into `$TENSORSHARP_MODELS/qwen-image-2.1/`, or `models/qwen-image-2.1/` when `TENSORSHARP_MODELS` is unset. It selects `ggml_metal`; append `--backend ggml_cuda` on an NVIDIA machine.

```bash
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1.json \
  --prompt 'A small orange cat beside a blue ceramic vase, soft daylight, detailed photograph' \
  --width 2048 --height 2048 --diffusion-steps 40 --cfg 1 \
  --diffusion-seed 42 --output generated.png
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1.json \
  --image generated.png \
  --prompt 'Change the blue vase to a red vase. Preserve the cat, lighting and composition.' \
  --width 2048 --height 2048 --diffusion-steps 40 --cfg 1 \
  --diffusion-seed 42 --output edited.png
dotnet run --project TensorSharp.Server.Host -c Release --no-build -- \
  --config config/qwen-image-2.1.json --host 127.0.0.1 --port 5000
```

To fetch the files yourself instead:

```bash
hf download Abiray/Qwen-Image-2.1-GGUF qwen_image_2.1_Q4_K_M.gguf --local-dir models
hf download Comfy-Org/Qwen-Image-2.1 vae/qwen_image_2.1_vae_bf16.safetensors --local-dir models
hf download Qwen/Qwen3-VL-8B-Instruct-GGUF Qwen3VL-8B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-8B-Instruct-F16.gguf --local-dir models

dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/qwen_image_2.1_Q4_K_M.gguf --qwen-image-vae models/vae/qwen_image_2.1_vae_bf16.safetensors --qwen-image-vl models/Qwen3VL-8B-Instruct-Q4_K_M.gguf --qwen-image-mmproj models/mmproj-Qwen3VL-8B-Instruct-F16.gguf --backend ggml_cuda
```

(In the Web UI, a prompt without an attachment generates an image; attach one or more images to edit. Omitted settings select 2048×2048, 40 Euler steps and CFG 1; `--width 1024 --height 1024` is the faster draft size, and on the server `--width` / `--height` given together change that default. See [qwenimage21.md](docs/models/qwenimage21.md).)

Step-distillation LoRA plug-ins (optional) cut the 40 steps to 4–8. Drop `--diffusion-steps 40 --cfg 1` from the CLI commands above and add `--lora config/lora/qwen-image-2.1-viggle-turbo.json` (6 steps by default, CFG 1) or `--lora config/lora/qwen-image-2.1-pruna-8step.json` (8 steps): the adapter downloads on first use and its recipe supplies the steps and CFG. An explicit `--diffusion-steps` / `--cfg` overrides the recipe, and a step count the recipe has no schedule for is refused. The server takes the same `--lora` at startup. All twelve plug-ins are listed in [config/README.md](config/README.md#qwen-image-21-lora-plug-ins-lora).

**MiniMax-H3 audio+video generation** — prompt (+ optional keyframes or references) → H.264 MP4 **and native 32 kHz stereo audio, generated together in one packed latent** ([unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF))

Four networks cooperate here, so the shortest route is a ready-made config — it names all four
and downloads whatever is missing (~35.5 GB on the first run):

```bash
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --config config/minimax-h3-fl2va.json
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --config config/minimax-h3-fl2va.json \
    --prompt "a red fox trotting through falling snow, cinematic" --output fox.mp4
```

`config/minimax-h3-ref2va.json` is the other checkpoint: up to nine identity and appearance
references — stills, clips, soundtracks — for a brand-new scene rather than frames the clip has to
reproduce. FL2VA and Ref2VA are **separate checkpoints, not a setting**, and asking one for the
other's conditioning fails with a message naming the file you actually need. Only the denoiser
differs between the two configs (~35.4 GB there), so the three networks below are shared and the
second config downloads just its own DiT. See
[config/README.md](config/README.md#video-generation-with-sound-minimax-h3). Files land wherever
`TENSORSHARP_MODELS` points, or in the `models/` folder at the repository root (gitignored).

One pair is not automated either way: the text-encoder GGUF carries no tokenizer, and auto-download
can only fill in options that are flags.

```bash
curl -L -o models/vocab.json https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/42ed227ee7df40d41602854ae760620d6eb651fe/processor/vocab.json
curl -L -o models/merges.txt https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/42ed227ee7df40d41602854ae760620d6eb651fe/processor/merges.txt
```

The manual route is below.

```bash
# FL2VA is the text / image-to-video / first-and-last-frame checkpoint; swap in
# minimax_h3_ref2va_pruned-Q4_K.gguf for reference conditioning. Both VAEs are mirrored
# in unsloth/MiniMax-H3-GGUF's own vae/ folder if Comfy-Org is slow.
hf download unsloth/MiniMax-H3-GGUF minimax_h3_fl2va_pruned-Q4_K.gguf --local-dir models
hf download unsloth/MiniMax-H3-GGUF qwen3vl_32b_minimax_h3-Q4_K_M.gguf --local-dir models
hf download Comfy-Org/MiniMax-H3 vae/minimax_h3_video_vae_fp16.safetensors --local-dir models
hf download Comfy-Org/MiniMax-H3 vae/minimax_h3_audio_vae_fp32.safetensors --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model models/minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_cuda \
    --video-text-encoder models/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
    --video-vae models/vae/minimax_h3_video_vae_fp16.safetensors \
    --audio-vae models/vae/minimax_h3_audio_vae_fp32.safetensors \
    --prompt "a red fox trotting through falling snow, cinematic" \
    --output fox.mp4 --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model models/minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_cuda \
    --video-text-encoder models/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
    --video-vae models/vae/minimax_h3_video_vae_fp16.safetensors \
    --audio-vae models/vae/minimax_h3_audio_vae_fp32.safetensors \
    --video-width 640 --video-height 384 --video-steps 20 --video-frames 22
```

That CLI run writes `fox.mp4` **and `fox.wav`**: the soundtrack is a sidecar, never muxed in,
because muxing needs an encoder that may not be installed. Put them together with
`ffmpeg -i fox.mp4 -i fox.wav -c:v copy -c:a aac fox_with_audio.mp4`. With everything in one folder the three
companion flags can be dropped: the denoiser's directory and its parent are scanned recursively,
subfolders included. Drop the audio VAE, or pass `--no-audio`, and you still get video — just silent.

H3 is CFG-distilled, so **`--cfg 1.0` is required** and anything higher is refused outright; the
pipeline's own default is 20 steps and 4-8 is the fast operating point, at the cost of some
chromatic fringing around moving subjects that is gone by ~20. Width and height round up to a
multiple of 32, the frame count snaps to the `17k+5` grid (5, 22, 39, 56, 73, 90 …) and fps is
pinned to 24 whatever you ask for. On the server the step count is spelled `--video-steps` and
there is no `--cfg` at all, which is why the shipped configs set neither.

For conditioning, `--image first.png` animates that picture as the first frame; adding
`--end-image last.png --video-mode fl2v` interpolates between the two; and on the Ref2VA checkpoint
`--ref-image` (repeatable, up to nine), `--ref-video`, `--ref-video-audio` and `--ref-audio` carry
identity and appearance into a new scene instead. Measured on an M5 Pro over Metal at 22 frames and
8 steps with the same seed, H3 runs **2.4x** faster than stable-diffusion.cpp at 256×256
(49.3 s → 20.9 s) and **1.7x** at 640×384 (108.5 s → 63.1 s). See
[docs/models/minimax-h3.md](docs/models/minimax-h3.md).

**Wan video generation** — prompt (+ optional first-frame image) → H.264 MP4, video only; needs the DiT + video VAE + UMT5-XXL text encoder ([hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF](https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF))

Wan needs three separate networks, so here too the shortest route is a
ready-made config — it names all three and downloads whatever is missing:

```bash
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --config config/wan-video-ti2v-5b-turbo.json
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --config config/wan-video-ti2v-5b-turbo.json \
    --prompt "a cute fluffy orange cat walking through a sunny garden" --output cat.mp4
```

`config/wan-video-ti2v-5b.json` is the undistilled 50-step variant and
`config/wan-video-i2v-a14b.json` the two-expert 14B image-to-video model; see
[config/README.md](config/README.md#video-generation-video-only-wan). Files land wherever
`TENSORSHARP_MODELS` points, or in the `models/` folder at the repository root (gitignored).
The manual route is below.

```bash
# The step-distilled Turbo DiT: 4 denoise passes instead of 100, detected from the file name.
# Note the Wan2_2 underscore in the Turbo file name; the VAE and encoder come from the base repos.
hf download hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF Wan2_2-TI2V-5B-Turbo-Q8_0.gguf --local-dir models
hf download QuantStack/Wan2.2-TI2V-5B-GGUF VAE/Wan2.2_VAE.safetensors --local-dir models
hf download city96/umt5-xxl-encoder-gguf umt5-xxl-encoder-Q8_0.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model models/Wan2_2-TI2V-5B-Turbo-Q8_0.gguf --backend ggml_cuda \
    --video-vae models/VAE/Wan2.2_VAE.safetensors --video-text-encoder models/umt5-xxl-encoder-Q8_0.gguf \
    --prompt "a cute fluffy orange cat walking through a sunny garden with flowers" \
    --output cat.mp4 --width 832 --height 480 --video-frames 81
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model models/Wan2_2-TI2V-5B-Turbo-Q8_0.gguf --backend ggml_cuda \
    --video-vae models/VAE/Wan2.2_VAE.safetensors --video-text-encoder models/umt5-xxl-encoder-Q8_0.gguf \
    --video-frames 121 --fps 24
```

The console prints `step-distilled checkpoint detected -> 4 steps, guidance off` on load — that
line is how you confirm you are on the fast path. Swapping only the `--model` path for the base
`Wan2.2-TI2V-5B-Q8_0.gguf` runs the official 50-step + CFG recipe instead: the same 1088×832×121-frame
request measured 3 h 30 m there against 17 m 30 s here (M5 Pro, `ggml_metal`). Add `--image first_frame.png`
for image-to-video, or attach an image in the Web UI (it becomes the first frame); on the server
`--video-frames` / `--fps` are defaults that a request can override. Wan does not run on
`--backend mlx`; use `ggml_cuda`, `ggml_metal`, `ggml_vulkan`, `ggml_cpu`, `cuda` or `cpu`.

If all three files sit in one folder (a `VAE/` subfolder counts) the `--video-vae` / `--video-text-encoder` flags
can be dropped — they are resolved automatically. For the two-expert A14B models download **both**
experts under the same `--local-dir` and point `--model` at either one:

```bash
hf download jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF high_noise/wan2.2_i2v_A14b_high_noise_lightx2v_4step-Q4_K_M.gguf --local-dir models
hf download jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF low_noise/wan2.2_i2v_A14b_low_noise_lightx2v_4step-Q4_K_M.gguf --local-dir models
hf download QuantStack/Wan2.2-I2V-A14B-GGUF VAE/Wan2.1_VAE.safetensors --local-dir models
hf download city96/umt5-xxl-encoder-gguf umt5-xxl-encoder-Q8_0.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model models/high_noise/wan2.2_i2v_A14b_high_noise_lightx2v_4step-Q4_K_M.gguf \
    --backend ggml_cuda --video-vae models/VAE/Wan2.1_VAE.safetensors \
    --video-text-encoder models/umt5-xxl-encoder-Q8_0.gguf \
    --prompt "the ship sails into the storm, waves crashing" --image ship.jpg --output ship.mp4
```
