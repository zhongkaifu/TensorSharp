# Model Downloads (GGUF)
[English](MODEL_DOWNLOADS.md) | [中文](MODEL_DOWNLOADS_zh-cn.md)

> Part of the [TensorSharp](README.md) documentation. See also the [per-model architecture cards](docs/models/README.md).


TensorSharp loads models in GGUF format. Below are verified Hugging Face repos for every supported architecture, including the multimodal-projector (mmproj) and MTP-draft companion files each family uses. Pick a quantization that fits your hardware (Q4_K_M / UD-Q4_K_XL for low memory, Q8_0 for higher quality, etc.). Rows marked *optional* are the speed artifacts — step-distilled checkpoints, distillation LoRAs and speculative-decoding drafters. Nothing breaks without them, but they are usually the difference between minutes and hours, so skim them before you start a long download.

| Architecture | Model | GGUF Download |
|---|---|---|
| Gemma 4 verified native tier | gemma-4-E4B-it Q8_0 | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) — recommended public artifact `gemma-4-E4B-it-Q8_0.gguf`; lower-memory Q4_K_M is also available; mmproj `mmproj-gemma-4-E4B-it-Q8_0.gguf` is in the same repo |
| Gemma 4 | gemma-4-12B-it (QAT) | [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) — mmproj `mmproj-BF16.gguf` and MTP draft `mtp-gemma-4-12B-it.gguf` in the same repo |
| Gemma 4 | gemma-4-26B-A4B-it (MoE, QAT) | [unsloth/gemma-4-26B-A4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF) — mmproj `mmproj-BF16.gguf` and MTP draft `mtp-gemma-4-26B-A4B-it.gguf` in the same repo |
| Gemma 4 | gemma-4-26B-A4B-it (MoE) | [ggml-org/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF) — mmproj files in the same repo |
| Gemma 4 | gemma-4-31B-it | [ggml-org/gemma-4-31B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF) — mmproj files in the same repo |
| Gemma 4 | `gemma4-assistant` MTP drafts (optional — speculative decoding) | [AtomicChat/gemma-4-E4B-it-assistant-GGUF](https://huggingface.co/AtomicChat/gemma-4-E4B-it-assistant-GGUF) (E4B) and [AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF](https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF) (26B-A4B) — load via `--draft-model`, which enables speculation by itself; pair each draft with its matching target size |
| Qwen 3.5 / 3.6 family | Qwen3.5-9B | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) — mmproj `mmproj-F16.gguf` in the same repo |
| Qwen 3.5 / 3.6 family | Qwen3.5-35B-A3B (MoE) | [ggml-org/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF) — mmproj `mmproj-Qwen3.5-35B-A3B-Q8_0.gguf` in the same repo |
| Qwen 3.5 / 3.6 family | Qwen3.6-35B-A3B (MoE, embedded NextN MTP) | [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) — these GGUFs retain the NextN block for the server's `--spec`; mmproj `mmproj-F16.gguf` in the same repo. The base repo [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) ships the same file names with NextN stripped — those load fine but silently fall back to standard decode |
| Qwen 3.8 Flash Next | Qwen3.8-Flash-Next (hybrid MoE, image-capable) | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) — one subdirectory per quant (`UD-Q2_K_XL/`, …), each a multi-shard set; point `--model` at the `-00001-of-` shard. `mmproj-BF16.gguf` beside the model enables image input, multi-image prompts and multi-turn image sessions included. `general.architecture` = `qwen4exp`. On a multi-GPU box `--tp N` runs a **layer split** — whole layers per GPU, the same (and only) multi-GPU mode llama.cpp offers this architecture — which buys capacity, not speed; see [USAGE.md](USAGE.md#tensor-parallelism--distributed-inference) |
| GPT OSS | gpt-oss-20b (MoE) | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) — `gpt-oss-20b-MXFP4.gguf` (note the uppercase `MXFP4`), text only, no companion files |
| Nemotron-H | Nemotron-H-8B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF) |
| Nemotron-H | Nemotron-H-47B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF) |
| Nemotron-H | Nemotron 3 Nano Omni 30B-A3B (image-capable) | [unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF) — mmproj `mmproj-BF16.gguf` (same repo) is required for image input. Audio is preprocessed only: real audio inference needs a Parakeet audio mmproj these GGUFs do not ship |
| Nemotron 3.5 | Nemotron-3.5-Lightning-30B-A3B (hybrid 23 Mamba-2 + 23 MoE + 6 attention) | [unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) — e.g. `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MXFP4_MOE.gguf` (MoE experts MXFP4, ~17 GB); `general.architecture` = `nemotron_h_moe`. Smaller/other quants: [ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) (BF16/Q4_0/Q8_0 + separate MTP GGUFs). Optional speed artifact: the DSpark drafter below |
| Nemotron 3.5 | DSpark speculative drafter (optional — speed only) | [magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark-GGUF](https://huggingface.co/magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark-GGUF) — the llama.cpp DFlash export of the official `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark` module (6 SWA layers, Markov head r512, attention sinks); loaded with `--draft-model` for block (DSpark) speculative decoding. Rebuild it from the official safetensors with `eng/nemotron-dspark-to-gguf.py` |
| Mistral 3 | Mistral-Small-3.1-24B-Instruct-2503 | [bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF) — Pixtral mmproj `mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf` in the same repo |
| Muse-Glimmer | Muse-Glimmer-30B (dense, image-capable) | [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) — e.g. `Muse-Glimmer-30B-UD-Q4_K_XL.gguf` or `Muse-Glimmer-30B-Q8_0.gguf`; `general.architecture` = `muse-glimmer` / `muse_glimmer`. Image input requires `mmproj-Muse-Glimmer-30B-Q8_0.gguf` (same repo) passed **explicitly** with `--mmproj` — this is the one family with no mmproj auto-detection. Optional speed artifacts: the DFlash block drafter `dflash-kquant.gguf` (same repo) or the newer DFlash2 drafter [z-lab/Muse-Glimmer-30B-DFlash2-GGUF](https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2-GGUF) (prefer `-Q4_K_M` on a 16 GB card — see the note on drafter size in [speculative_decoding.md](docs/speculative_decoding.md#what-to-expect)), loaded with `--draft-model` for lossless speculative decoding — pass no sampler flags, it needs plain greedy |
| DeepSeek V4 | DeepSeek-V4-Flash-0731 (284B MoE) | [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) — one subdirectory per quant (`UD-Q8_K_XL/`, `UD-IQ4_XS/`, `UD-IQ1_S/`, …), each a multi-shard set; point `--model` at the `-00001-of-` shard. Text only |
| GLM 5.x | GLM-5.2 (744B-A40B MoE, embedded NextN MTP) | [unsloth/GLM-5.2-GGUF](https://huggingface.co/unsloth/GLM-5.2-GGUF) — one subdirectory per quant (`UD-Q4_K_XL/`, `UD-IQ2_XXS/`, …), each a multi-shard set; point `--model` at the `-00001-of-` shard. **Text only** — GLM-5.3-Flash in the next row is the one that takes images. These GGUFs already carry the NextN block for the server's `--spec` — unlike Qwen 3.6 there is no separate MTP repo to pick |
| GLM 5.x | GLM-5.3-Flash (320B, 288 routed experts, text + image) | [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) — one subdirectory per quant (`UD-Q2_K_XL/`, …), each a multi-shard set; point `--model` at the `-00001-of-` shard. `general.architecture` = `glm5next`, and it loads through the same native executor as GLM-5.2. Unlike 5.2 it **takes images**: `mmproj-BF16.gguf` (the GLM-OCR ViT, same repo) enables `--image`, multi-image prompts and multi-turn image sessions. Its NextN block is not wired up yet, so there is no `--spec` here. Omitting `--tp` uses the default layer split across every visible GPU; on GGML GPU backends, `--tp N` selects native local/single-process tensor parallelism |
| DeepSeek V4 | DSpark speculative drafters (optional — speed only) | see [DSpark drafters](#dspark-drafters) below — a separate GGUF loaded with `--draft-model` for ~1.3-1.4x decode |
| DiffusionGemma | diffusiongemma-26B-A4B-it | [unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) (`general.architecture` = `diffusion-gemma`) |
| Qwen-Image-Edit | MMDiT DiT (the `--model` GGUF) | [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF) (e.g. `qwen-image-edit-2511-Q4_K_M.gguf`; `general.architecture` = `qwen_image`) |
| Qwen-Image-Edit | Qwen-Image VAE (required) | `VAE/Qwen_Image-VAE.safetensors` from [QuantStack/Qwen-Image-Edit-GGUF](https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF) — place next to the DiT or point `--qwen-image-vae` / `TS_QWEN_IMAGE_VAE` at it (the `.safetensors` VAE loads directly) |
| Qwen-Image-Edit | Qwen2.5-VL-7B text encoder (required) | [unsloth/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF) — place next to the DiT or set `--qwen-image-vl` / `TS_QWEN_IMAGE_TE` |
| Qwen-Image-Edit | Vision mmproj (optional) | `mmproj-BF16.gguf` from [unsloth/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF) — image-grounded conditioning via `--qwen-image-mmproj` / `TS_QWEN_IMAGE_MMPROJ` |
| Qwen-Image-Edit | Lightning LoRA (optional, 4/8-step) | [lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) (`Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`) — `--qwen-image-lora` / `TS_QWEN_IMAGE_LORA`; auto-switches to the LoRA's step count and CFG 1.0 |
| MiniMax-H3 audio+video | denoiser (the `--model` GGUF) | **Two separate checkpoints, not settings** — which one you load decides what conditioning it accepts. [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF): `minimax_h3_fl2va_pruned-Q4_K.gguf` (10.64 GiB) for text / image-to-video / first-and-last-frame, or `minimax_h3_ref2va_pruned-Q4_K.gguf` (10.60 GiB) for identity/appearance references. Also Q8_0 (19.97 GiB) down to Q2_K (6.26 GiB). H3 is CFG-distilled: **pass `--cfg 1.0`** and 4-8 steps. The GGUFs carry **no metadata at all**, so TensorSharp identifies them by their tensors, and the partition off the file name — keep `fl2va` / `ref2va` in it if you rename or requantize. Both checkpoints share the three networks below, so adding the second one later costs only its own ~10.6 GiB |
| MiniMax-H3 audio+video | Qwen3-VL-32B text encoder (required) | Same repo: `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` (16.97 GiB), or `-Q2_K_M.gguf` (12.20 GiB) to pair with the two smallest denoisers. Truncated to 50 layers with the final norm removed. Freed before the denoise starts. **It ships no tokenizer** — also download `vocab.json` and `merges.txt` from [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/processor) and put them beside it (or set `TS_VIDEO_TOKENIZER`) |
| MiniMax-H3 audio+video | `vocab.json` + `merges.txt` (required) | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/processor) — the Qwen2 byte-level BPE pair the encoder GGUF omits, and the one thing a config cannot auto-download for you (auto-download fills in options that are flags; the tokenizer is not one). `curl -L -o models/vocab.json https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/processor/vocab.json` and the same for `merges.txt` |
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
| [bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF](https://huggingface.co/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF) | 7.0 GB | `DSpark-drafter-Q2K-Q8-0731.gguf` for the **0731** release (a non-0731 build is in the same repo) | Q2_K experts + Q8_0 dense; measured 71% acceptance |
| [sakamakismile/DeepSeek-V4-Flash-DSpark-support-ds4-GGUF](https://huggingface.co/sakamakismile/DeepSeek-V4-Flash-DSpark-support-ds4-GGUF) | 5.6 GB | the pre-0731 `DeepSeek-V4-Flash` release | Smallest, and still ~69% acceptance against the 0731 trunk; fastest of the three on the direct-CUDA engine because its weights are re-read every speculative step |
| [alessandrobologna/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF](https://huggingface.co/alessandrobologna/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF) | 10.9 GB | the **0731** release | MXFP4 experts (lossless repack of the checkpoint's FP4); highest acceptance measured (68%), most VRAM — it displaces about a whole trunk layer per GPU |

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
> per-layer input residuals its encoder consumes; only `MuseGlimmerModel` does so today.

| Backbone | Official checkpoint (safetensors) | Community GGUF |
|---|---|---|
| Gemma-4-12B | [deepseek-ai/dspark_gemma4_12b_block7](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7) | [ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF](https://huggingface.co/ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF) (1.9 GB), [williamliao/dspark_gemma4_12b-GGUF](https://huggingface.co/williamliao/dspark_gemma4_12b-GGUF) (IQ4_XS…F16) |
| Gemma-4-26B-A4B | — | [williamliao/dspark_gemma4_26b-a4b-it-GGUF](https://huggingface.co/williamliao/dspark_gemma4_26b-a4b-it-GGUF) (1.2-3.8 GB) |
| Gemma-4-31B | — | [williamliao/dspark_gemma4_31b-it-GGUF](https://huggingface.co/williamliao/dspark_gemma4_31b-it-GGUF) (3.3-11 GB) |

Gemma 4 does have a supported speculative path today — the `gemma4-assistant` MTP drafts in
the table above, via `--draft-model` — and Qwen 3.6 and GLM 5.2 have their
embedded NextN blocks. Those are different drafters from DSpark.

### Download & Run — per-model quick reference

These commands run from the repository root. First install the [.NET 10 SDK](DEVELOPMENT.md#install-the-net-10-sdk) for your platform and run `dotnet build TensorSharp.slnx -c Release`; a runtime-only installation cannot build the binaries used below.

The `hf download` commands need the Hugging Face CLI (`pip install -U huggingface_hub`) and drop every file into `./models`. Reminders that apply to all blocks: the CLI reads its one-shot prompt from a **file** via `--input` (`--prompt` is the Qwen-Image-Edit edit instruction and the video-generation prompt, MiniMax-H3 and Wan alike), samples **greedily** by default, and generates only 100 tokens unless you raise `--max-tokens`; the server always listens on **http://localhost:5000**. Swap `--backend ggml_cuda` for the backend that fits your hardware (see [Pick a Backend](README.md#pick-a-backend)). Create a prompt file first:

```bash
echo "Give me three facts about the Moon." > prompt.txt
```

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

Drop `--draft-model` for plain decode. Speculation needs greedy sampling (`--temperature 0`);
`--spec-pmin` tunes how far each block is drafted.

**Gemma 4** — text + image/video/audio, thinking, tools, MTP ([ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF))

```bash
hf download ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q8_0.gguf --local-dir models
hf download ggml-org/gemma-4-E4B-it-GGUF mmproj-gemma-4-E4B-it-Q8_0.gguf --local-dir models
hf download AtomicChat/gemma-4-E4B-it-assistant-GGUF gemma-4-E4B-it-assistant.Q8_0.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gemma-4-E4B-it-Q8_0.gguf --mmproj models/mmproj-gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/gemma-4-E4B-it-Q8_0.gguf --mmproj models/mmproj-gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --draft-model models/gemma-4-E4B-it-assistant.Q8_0.gguf
```

(The third download and the `--draft-model` flag are optional — they enable MTP speculative decoding, a server-only feature.)

**Qwen 3.5 / 3.6 family** — text + image, thinking, tools, NextN MTP on 3.6 ([unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF))

```bash
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf --local-dir models
hf download unsloth/Qwen3.5-9B-GGUF mmproj-F16.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/Qwen3.5-9B-UD-Q4_K_XL.gguf --mmproj models/mmproj-F16.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/Qwen3.5-9B-UD-Q4_K_XL.gguf --mmproj models/mmproj-F16.gguf --backend ggml_cuda
```

Qwen 3.6 NextN speculative decoding (server-only; download from the **-MTP-** repo — base-repo GGUFs strip the NextN block and silently fall back to standard decode):

```bash
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --local-dir models

dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --backend ggml_cuda --spec
```

**GPT OSS** — text, thinking (always on), tools ([ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF))

```bash
hf download ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gpt-oss-20b-MXFP4.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/gpt-oss-20b-MXFP4.gguf --backend ggml_cuda
```

**Nemotron-H** — text, thinking, tools; image on the Omni distribution ([bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF))

```bash
hf download bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --backend ggml_cuda
```

For image input use the Omni distribution instead: `NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf` + `mmproj-BF16.gguf` from [unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF). Audio is not functional (it needs a Parakeet audio mmproj the GGUFs do not ship).

**Mistral 3** — text + image (Pixtral) ([bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF))

```bash
hf download bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --local-dir models
hf download bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --mmproj models/mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --mmproj models/mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf --backend ggml_cuda
```

**DiffusionGemma** — block text-diffusion ([unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF))

```bash
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF diffusiongemma-26B-A4B-it-Q4_K_M.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --input prompt.txt --max-tokens 256 --diffusion-steps 48 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --backend ggml_cuda
```

(The Web UI streams live denoising previews for DiffusionGemma; the compat APIs return the final text.)

**Qwen-Image-Edit** — image + prompt → edited image; needs the DiT + VAE + text encoder, Lightning LoRA optional ([unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF))

```bash
hf download unsloth/Qwen-Image-Edit-2511-GGUF qwen-image-edit-2511-Q4_K_M.gguf --local-dir models
hf download QuantStack/Qwen-Image-Edit-GGUF VAE/Qwen_Image-VAE.safetensors --local-dir models
hf download unsloth/Qwen2.5-VL-7B-Instruct-GGUF Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf --local-dir models
hf download lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/qwen-image-edit-2511-Q4_K_M.gguf --image input.png --prompt "Make the sky a dramatic sunset." --output edited.png --qwen-image-vae models/VAE/Qwen_Image-VAE.safetensors --qwen-image-vl models/Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf --qwen-image-lora models/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/qwen-image-edit-2511-Q4_K_M.gguf --qwen-image-vae models/VAE/Qwen_Image-VAE.safetensors --qwen-image-vl models/Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf --qwen-image-lora models/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors --backend ggml_cuda
```

(In the Web UI, attach an image and type the edit instruction. The Lightning LoRA download and `--qwen-image-lora` flag are optional — they cut the denoise to 4 steps at CFG 1.0.)

**MiniMax-H3 audio+video generation** — prompt (+ optional keyframes or references) → H.264 MP4 **and native 32 kHz stereo audio, generated together in one packed latent** ([unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF))

Four networks cooperate here, so the shortest route is a ready-made config — it names all four
and downloads whatever is missing (~33.5 GB on the first run):

```bash
TensorSharp.Server --config config/minimax-h3-fl2va.json
TensorSharp.Cli    --config config/minimax-h3-fl2va.json \
    --prompt "a red fox trotting through falling snow, cinematic" --output fox.mp4
```

`config/minimax-h3-ref2va.json` is the other checkpoint: up to nine identity and appearance
references — stills, clips, soundtracks — for a brand-new scene rather than frames the clip has to
reproduce. FL2VA and Ref2VA are **separate checkpoints, not a setting**, and asking one for the
other's conditioning fails with a message naming the file you actually need. Only the denoiser
differs between the two configs (~33.4 GB there), so the three networks below are shared and the
second config downloads just its own DiT. See
[config/README.md](config/README.md#video-generation-with-sound-minimax-h3). Files land wherever
`TENSORSHARP_MODELS` points, or in `models/` next to the repository.

One pair is not automated either way: the text-encoder GGUF carries no tokenizer, and auto-download
can only fill in options that are flags.

```bash
curl -L -o models/vocab.json https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/processor/vocab.json
curl -L -o models/merges.txt https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/processor/merges.txt
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
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll \
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
TensorSharp.Server --config config/wan-video-ti2v-5b-turbo.json
TensorSharp.Cli    --config config/wan-video-ti2v-5b-turbo.json \
    --prompt "a cute fluffy orange cat walking through a sunny garden" --output cat.mp4
```

`config/wan-video-ti2v-5b.json` is the undistilled 50-step variant and
`config/wan-video-i2v-a14b.json` the two-expert 14B image-to-video model; see
[config/README.md](config/README.md#video-generation-video-only-wan). Files land wherever
`TENSORSHARP_MODELS` points, or in `models/` next to the repository. The manual
route is below.

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
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll \
    --model models/Wan2_2-TI2V-5B-Turbo-Q8_0.gguf --backend ggml_cuda \
    --video-vae models/VAE/Wan2.2_VAE.safetensors --video-text-encoder models/umt5-xxl-encoder-Q8_0.gguf \
    --video-frames 121 --fps 24
```

The console prints `step-distilled checkpoint detected -> 4 steps, guidance off` on load — that
line is how you confirm you are on the fast path. Swapping only the `--model` path for the base
`Wan2.2-TI2V-5B-Q8_0.gguf` runs the official 50-step + CFG recipe instead: the same 1088×832×121-frame
request measured 3 h 30 m there against 17 m 30 s here (M5 Pro, `ggml_metal`). Add `--image first_frame.png`
for image-to-video, or attach an image in the Web UI (it becomes the first frame); on the server
`--video-frames` / `--fps` are defaults that a request can override. Wan is the one family that does
not run on `--backend mlx`; use `ggml_cuda`, `ggml_metal`, `ggml_vulkan`, `ggml_cpu`, `cuda` or `cpu`.

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
