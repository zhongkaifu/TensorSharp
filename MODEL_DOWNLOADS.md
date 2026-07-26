# Model Downloads (GGUF)
[English](MODEL_DOWNLOADS.md) | [中文](MODEL_DOWNLOADS_zh-cn.md)

> Part of the [TensorSharp](README.md) documentation. See also the [per-model architecture cards](docs/models/README.md).


TensorSharp loads models in GGUF format. Below are verified Hugging Face repos for every supported architecture, including the multimodal-projector (mmproj) and MTP-draft companion files each family uses. Pick a quantization that fits your hardware (Q4_K_M / UD-Q4_K_XL for low memory, Q8_0 for higher quality, etc.).

| Architecture | Model | GGUF Download |
|---|---|---|
| Gemma 4 verified native tier | gemma-4-E4B-it Q8_0 | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) — recommended public artifact `gemma-4-E4B-it-Q8_0.gguf`; lower-memory Q4_K_M is also available; mmproj `mmproj-gemma-4-E4B-it-Q8_0.gguf` is in the same repo |
| Gemma 4 | gemma-4-12B-it (QAT) | [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) — mmproj `mmproj-BF16.gguf` and MTP draft `mtp-gemma-4-12B-it.gguf` in the same repo |
| Gemma 4 | gemma-4-26B-A4B-it (MoE, QAT) | [unsloth/gemma-4-26B-A4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF) — mmproj `mmproj-BF16.gguf` and MTP draft `mtp-gemma-4-26B-A4B-it.gguf` in the same repo |
| Gemma 4 | gemma-4-26B-A4B-it (MoE) | [ggml-org/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF) — mmproj files in the same repo |
| Gemma 4 | gemma-4-31B-it | [ggml-org/gemma-4-31B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF) — mmproj files in the same repo |
| Gemma 4 | `gemma4-assistant` MTP drafts | [AtomicChat/gemma-4-E4B-it-assistant-GGUF](https://huggingface.co/AtomicChat/gemma-4-E4B-it-assistant-GGUF) (E4B) and [AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF](https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF) (26B-A4B) — load via the server's `--mtp-spec --mtp-draft-model`; pair each draft with its matching target size |
| Gemma 3 | gemma-3-4b-it | [ggml-org/gemma-3-4b-it-GGUF](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF) — mmproj `mmproj-model-f16.gguf` in the same repo. The official QAT repo [google/gemma-3-4b-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf) is gated (requires HF login + accepting Google's Gemma license) |
| Qwen 3 | Qwen3-4B | [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF) (text only — no companion files) |
| Qwen 3.5 / 3.6 family | Qwen3.5-9B | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) — mmproj `mmproj-F16.gguf` in the same repo |
| Qwen 3.5 / 3.6 family | Qwen3.5-35B-A3B (MoE) | [ggml-org/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF) — mmproj `mmproj-Qwen3.5-35B-A3B-Q8_0.gguf` in the same repo |
| Qwen 3.5 / 3.6 family | Qwen3.6-35B-A3B (MoE, embedded NextN MTP) | [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) — these GGUFs retain the NextN block for the server's `--mtp-spec`; mmproj `mmproj-F16.gguf` in the same repo. The base repo [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) ships the same file names with NextN stripped — those load fine but silently fall back to standard decode |
| GPT OSS | gpt-oss-20b (MoE) | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) (`gpt-oss-20b-mxfp4.gguf`, text only) |
| Nemotron-H | Nemotron-H-8B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF) |
| Nemotron-H | Nemotron-H-47B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF) |
| Nemotron-H | Nemotron 3 Nano Omni 30B-A3B (image-capable) | [unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF) — mmproj `mmproj-BF16.gguf` (same repo) is required for image input. Audio is preprocessed only: real audio inference needs a Parakeet audio mmproj these GGUFs do not ship |
| Mistral 3 | Mistral-Small-3.1-24B-Instruct-2503 | [bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF) — Pixtral mmproj `mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf` in the same repo |
| DiffusionGemma | diffusiongemma-26B-A4B-it | [unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) (`general.architecture` = `diffusion-gemma`) |
| Qwen-Image-Edit | MMDiT DiT (the `--model` GGUF) | [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF) (e.g. `qwen-image-edit-2511-Q4_K_M.gguf`; `general.architecture` = `qwen_image`) |
| Qwen-Image-Edit | Qwen-Image VAE (required) | `VAE/Qwen_Image-VAE.safetensors` from [QuantStack/Qwen-Image-Edit-GGUF](https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF) — place next to the DiT or point `--qwen-image-vae` / `TS_QWEN_IMAGE_VAE` at it (the `.safetensors` VAE loads directly) |
| Qwen-Image-Edit | Qwen2.5-VL-7B text encoder (required) | [unsloth/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF) — place next to the DiT or set `--qwen-image-vl` / `TS_QWEN_IMAGE_TE` |
| Qwen-Image-Edit | Vision mmproj (optional) | `mmproj-BF16.gguf` from [unsloth/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF) — image-grounded conditioning via `--qwen-image-mmproj` / `TS_QWEN_IMAGE_MMPROJ` |
| Qwen-Image-Edit | Lightning LoRA (optional, 4/8-step) | [lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) (`Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`) — `--qwen-image-lora` / `TS_QWEN_IMAGE_LORA`; auto-switches to the LoRA's step count and CFG 1.0 |

### Download & Run — per-model quick reference

These commands run from the repository root. First install the [.NET 10 SDK](DEVELOPMENT.md#install-the-net-10-sdk) for your platform and run `dotnet build TensorSharp.slnx -c Release`; a runtime-only installation cannot build the binaries used below.

The `hf download` commands need the Hugging Face CLI (`pip install -U huggingface_hub`) and drop every file into `./models`. Reminders that apply to all blocks: the CLI reads its one-shot prompt from a **file** via `--input` (`--prompt` is exclusively the Qwen-Image-Edit edit instruction), samples **greedily** by default, and generates only 100 tokens unless you raise `--max-tokens`; the server always listens on **http://localhost:5000**. Swap `--backend ggml_cuda` for the backend that fits your hardware (see [Pick a Backend](README.md#pick-a-backend)). Create a prompt file first:

```bash
echo "Give me three facts about the Moon." > prompt.txt
```

**Gemma 4** — text + image/video/audio, thinking, tools, MTP ([ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF))

```bash
hf download ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q8_0.gguf --local-dir models
hf download ggml-org/gemma-4-E4B-it-GGUF mmproj-gemma-4-E4B-it-Q8_0.gguf --local-dir models
hf download AtomicChat/gemma-4-E4B-it-assistant-GGUF gemma-4-E4B-it-assistant.Q8_0.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gemma-4-E4B-it-Q8_0.gguf --mmproj models/mmproj-gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/gemma-4-E4B-it-Q8_0.gguf --mmproj models/mmproj-gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --mtp-spec --mtp-draft-model models/gemma-4-E4B-it-assistant.Q8_0.gguf
```

(The third download and the `--mtp-spec --mtp-draft-model` pair are optional — they enable MTP speculative decoding, a server-only feature.)

**Gemma 3** — text + image ([ggml-org/gemma-3-4b-it-GGUF](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF); the official [google/gemma-3-4b-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf) is gated: HF login + Gemma license)

```bash
hf download ggml-org/gemma-3-4b-it-GGUF gemma-3-4b-it-Q4_K_M.gguf --local-dir models
hf download ggml-org/gemma-3-4b-it-GGUF mmproj-model-f16.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gemma-3-4b-it-Q4_K_M.gguf --mmproj models/mmproj-model-f16.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/gemma-3-4b-it-Q4_K_M.gguf --mmproj models/mmproj-model-f16.gguf --backend ggml_cuda
```

**Qwen 3** — text, thinking, tools ([Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF))

```bash
hf download Qwen/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/Qwen3-4B-Q4_K_M.gguf --input prompt.txt --max-tokens 300 --think --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/Qwen3-4B-Q4_K_M.gguf --backend ggml_cuda
```

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

dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --backend ggml_cuda --mtp-spec
```

**GPT OSS** — text, thinking (always on), tools ([ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF))

```bash
hf download ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-mxfp4.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gpt-oss-20b-mxfp4.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/gpt-oss-20b-mxfp4.gguf --backend ggml_cuda
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

