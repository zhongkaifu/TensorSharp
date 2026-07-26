# 模型下载（GGUF）
[English](MODEL_DOWNLOADS.md) | [中文](MODEL_DOWNLOADS_zh-cn.md)

> [TensorSharp](README_zh-cn.md) 文档的一部分。另见[各模型架构卡片](docs/models/README_zh-cn.md)。


TensorSharp 使用 GGUF 格式模型文件。以下是各架构对应的已核对 Hugging Face 下载入口与伴随文件。请根据硬件条件选择合适的量化版本（Q4_K_M / UD-Q4_K_XL 适合低内存，Q8_0 适合更高质量等）。

| 架构 | 模型 | GGUF 下载 |
|---|---|---|
| Gemma 4 已验证原生规格 | gemma-4-E4B-it Q8_0 | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)；推荐公开文件为 `gemma-4-E4B-it-Q8_0.gguf`，另有低内存 Q4_K_M；同仓库投影器为 `mmproj-gemma-4-E4B-it-Q8_0.gguf` |
| Gemma 4 | 12B / 26B-A4B QAT | [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) / [unsloth/gemma-4-26B-A4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF)；同仓库含 `mmproj-BF16.gguf` 与匹配的 MTP draft |
| Gemma 4 | 31B / 26B-A4B | [ggml-org/gemma-4-31B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF) / [ggml-org/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF)；同仓库含 mmproj |
| Gemma 4 | E4B / 26B-A4B MTP draft | [AtomicChat E4B assistant](https://huggingface.co/AtomicChat/gemma-4-E4B-it-assistant-GGUF) / [AtomicChat 26B assistant](https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF)；仅与匹配尺寸的目标配对 |
| Gemma 3 | gemma-3-4b-it | 非 gated 的 [ggml-org/gemma-3-4b-it-GGUF](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF)，投影器 `mmproj-model-f16.gguf`；官方 [Google QAT 仓库](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf)需要登录并接受许可证 |
| Qwen 3 | Qwen3-4B | [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF)，如 `Qwen3-4B-Q4_K_M.gguf` |
| Qwen 3.5 | Qwen3.5-9B | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)，投影器 `mmproj-F16.gguf` |
| Qwen 3.5 | Qwen3.5-35B-A3B | [ggml-org/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF)，投影器 `mmproj-Qwen3.5-35B-A3B-Q8_0.gguf` |
| Qwen 3.6 | Qwen3.6-35B-A3B（保留 NextN） | [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)，投影器 `mmproj-F16.gguf`；基础仓库会剥离 NextN 块 |
| GPT OSS | gpt-oss-20b（MoE） | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF)，文件 `gpt-oss-20b-mxfp4.gguf` |
| Nemotron-H | Nemotron-H-8B / 47B Reasoning | [8B](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF) / [47B](https://huggingface.co/bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF) |
| Nemotron-H | Nemotron 3 Nano Omni 30B-A3B | [unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF)，图像输入需 `mmproj-BF16.gguf`；仓库未附真实音频推理需要的 Parakeet mmproj |
| Mistral 3 | Mistral-Small-3.1-24B-Instruct | [bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF)，Pixtral 投影器 `mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf` |
| DiffusionGemma | diffusiongemma-26B-A4B-it | [unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF)，如 `diffusiongemma-26B-A4B-it-Q4_K_M.gguf` |
| Qwen-Image-Edit | MMDiT DiT（必需） | [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF)，如 `qwen-image-edit-2511-Q4_K_M.gguf` |
| Qwen-Image-Edit | VAE + Qwen2.5-VL（必需） | [QuantStack VAE](https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF) 中的 `VAE/Qwen_Image-VAE.safetensors` + [unsloth/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF) |
| Qwen-Image-Edit | Lightning LoRA（可选） | [lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)，如 4-step `.safetensors`；通过 `--qwen-image-lora` 加载 |

### 按模型下载并运行

以下命令从仓库根目录运行；请先按平台安装完整的 [.NET 10 SDK](DEVELOPMENT_zh-cn.md#安装-net-10-sdk)，再执行 `dotnet build TensorSharp.slnx -c Release`。仅安装 Runtime 无法构建下方使用的二进制文件。`hf` 来自 Hugging Face CLI（`pip install -U huggingface_hub`）。单次文本提示词必须通过 `--input` 文件传入，`--prompt` 仅用于 Qwen-Image-Edit。按硬件把示例的 `ggml_cuda` 换成 `ggml_metal`、`ggml_vulkan` 或 `ggml_cpu`。

```bash
echo "列出三条关于月球的事实。" > prompt.txt
```

**Gemma 4**（文本 + 图像/视频/音频、思维链、工具、可选 MTP）：

```bash
hf download ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q8_0.gguf --local-dir models
hf download ggml-org/gemma-4-E4B-it-GGUF mmproj-gemma-4-E4B-it-Q8_0.gguf --local-dir models
hf download AtomicChat/gemma-4-E4B-it-assistant-GGUF gemma-4-E4B-it-assistant.Q8_0.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gemma-4-E4B-it-Q8_0.gguf --mmproj models/mmproj-gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/gemma-4-E4B-it-Q8_0.gguf --mmproj models/mmproj-gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --mtp-spec --mtp-draft-model models/gemma-4-E4B-it-assistant.Q8_0.gguf
```

第三个下载与两项 MTP 参数可省略。

**Gemma 3**（文本 + 图像；下方非 gated 仓库）：

```bash
hf download ggml-org/gemma-3-4b-it-GGUF gemma-3-4b-it-Q4_K_M.gguf --local-dir models
hf download ggml-org/gemma-3-4b-it-GGUF mmproj-model-f16.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gemma-3-4b-it-Q4_K_M.gguf --mmproj models/mmproj-model-f16.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/gemma-3-4b-it-Q4_K_M.gguf --mmproj models/mmproj-model-f16.gguf --backend ggml_cuda
```

**Qwen 3**（文本、思维链、工具）：

```bash
hf download Qwen/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/Qwen3-4B-Q4_K_M.gguf --input prompt.txt --think --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/Qwen3-4B-Q4_K_M.gguf --backend ggml_cuda
```

**Qwen 3.5 / 3.6**（图像、思维链、工具；3.6 可用 NextN）：

```bash
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf --local-dir models
hf download unsloth/Qwen3.5-9B-GGUF mmproj-F16.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/Qwen3.5-9B-UD-Q4_K_XL.gguf --mmproj models/mmproj-F16.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/Qwen3.5-9B-UD-Q4_K_XL.gguf --mmproj models/mmproj-F16.gguf --backend ggml_cuda

# 3.6 必须从保留 NextN 块的 -MTP- 仓库下载
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --local-dir models
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --backend ggml_cuda --mtp-spec
```

**GPT OSS**（文本、始终思考、工具）：

```bash
hf download ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-mxfp4.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gpt-oss-20b-mxfp4.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/gpt-oss-20b-mxfp4.gguf --backend ggml_cuda
```

**Nemotron-H**（文本、思维链、工具）：

```bash
hf download bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --backend ggml_cuda
```

Omni 图像版本需另下同仓库 `mmproj-BF16.gguf`；当前发行版没有真实音频推理所需的 Parakeet audio mmproj。

**Mistral 3**（文本 + 图像）：

```bash
hf download bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --local-dir models
hf download bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --mmproj models/mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --mmproj models/mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf --backend ggml_cuda
```

**DiffusionGemma**（块文本扩散）：

```bash
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF diffusiongemma-26B-A4B-it-Q4_K_M.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --input prompt.txt --max-tokens 256 --diffusion-steps 48 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --backend ggml_cuda
```

**Qwen-Image-Edit**（DiT + VAE + 文本编码器；Lightning LoRA 可选）：

```bash
hf download unsloth/Qwen-Image-Edit-2511-GGUF qwen-image-edit-2511-Q4_K_M.gguf --local-dir models
hf download QuantStack/Qwen-Image-Edit-GGUF VAE/Qwen_Image-VAE.safetensors --local-dir models
hf download unsloth/Qwen2.5-VL-7B-Instruct-GGUF Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf --local-dir models
hf download lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/qwen-image-edit-2511-Q4_K_M.gguf --image input.png --prompt "把天空改成壮丽的日落。" --output edited.png --qwen-image-vae models/VAE/Qwen_Image-VAE.safetensors --qwen-image-vl models/Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf --qwen-image-lora models/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/qwen-image-edit-2511-Q4_K_M.gguf --qwen-image-vae models/VAE/Qwen_Image-VAE.safetensors --qwen-image-vl models/Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf --qwen-image-lora models/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors --backend ggml_cuda
```

