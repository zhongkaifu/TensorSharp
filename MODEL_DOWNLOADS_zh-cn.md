# 模型下载（GGUF）
[English](MODEL_DOWNLOADS.md) | [中文](MODEL_DOWNLOADS_zh-cn.md)

> [TensorSharp](README_zh-cn.md) 文档的一部分。另见[各模型架构卡片](docs/models/README_zh-cn.md)。


TensorSharp 使用 GGUF 格式模型文件。以下是各架构对应的已核对 Hugging Face 下载入口与伴随文件。请根据硬件条件选择合适的量化版本（Q4_K_M / UD-Q4_K_XL 适合低内存，Q8_0 适合更高质量等）。标注“可选”的条目是提速用的产物——步数蒸馏 checkpoint、蒸馏 LoRA、推测解码 draft 模型。不下载也能跑通，但它们往往就是“几分钟”和“几小时”的差别，动手前请先扫一眼。

| 架构 | 模型 | GGUF 下载 |
|---|---|---|
| Gemma 4 已验证原生规格 | gemma-4-E4B-it Q8_0 | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)；推荐公开文件为 `gemma-4-E4B-it-Q8_0.gguf`，另有低内存 Q4_K_M；同仓库投影器为 `mmproj-gemma-4-E4B-it-Q8_0.gguf` |
| Gemma 4 | 12B / 26B-A4B QAT | [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) / [unsloth/gemma-4-26B-A4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF)；同仓库含 `mmproj-BF16.gguf`，以及匹配的 MTP draft（`mtp-gemma-4-12B-it.gguf` / `mtp-gemma-4-26B-A4B-it.gguf`，可选，仅用于推测解码） |
| Gemma 4 | 31B / 26B-A4B | [ggml-org/gemma-4-31B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF) / [ggml-org/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF)；同仓库含 mmproj |
| Gemma 4 | E4B / 26B-A4B MTP draft（可选，仅用于推测解码） | [AtomicChat E4B assistant](https://huggingface.co/AtomicChat/gemma-4-E4B-it-assistant-GGUF) / [AtomicChat 26B assistant](https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF)；仅与匹配尺寸的目标配对 |
| Qwen 3.5 | Qwen3.5-9B | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)，投影器 `mmproj-F16.gguf` |
| Qwen 3.5 | Qwen3.5-35B-A3B | [ggml-org/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF)，投影器 `mmproj-Qwen3.5-35B-A3B-Q8_0.gguf` |
| Qwen 3.6 | Qwen3.6-35B-A3B（保留 NextN） | [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)，投影器 `mmproj-F16.gguf`。**注意不要下载基础仓库** [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)：它的文件名完全相同，但剥离了 NextN 块，`--spec` 会静默回落到普通解码 |
| Qwen 3.8 Flash Next | Qwen3.8-Flash-Next（混合 MoE，支持图像） | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)；每种量化一个子目录（`UD-Q2_K_XL/` 等），均为多分片，`--model` 指向 `-00001-of-` 分片。模型旁的 `mmproj-BF16.gguf` 启用图像输入，多图提示与多轮图像会话都可用。`general.architecture` 为 `qwen4exp`。多卡机器上 `--tp N` 走的是**按层切分**——整层落在单卡，也是 llama.cpp 对这个架构唯一提供的多卡模式——买到的是容量而不是速度，见 [USAGE_zh-cn.md](USAGE_zh-cn.md#张量并行与分布式推理) |
| Bonsai Q1_0 | Bonsai-8B / Bonsai-27B | **无下载——仅限本地旁加载。** 两个 GGUF 都没有声明发布方仓库 URL 或 license，因此 TensorSharp 只钉住确切的字节长度与 SHA-256，而不替它们编造来源；没有可 `hf download` 的东西。请把 `--model` 指向你已经拥有、并且有权使用的文件。这两个文件名字相同但架构不同（8B 为 `qwen3`，27B 为 `qwen35`）；钉住的取值、命令与实测数据见 [bonsai_zh-cn.md](docs/models/bonsai_zh-cn.md) |
| GPT OSS | gpt-oss-20b（MoE） | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF)，文件 `gpt-oss-20b-MXFP4.gguf`（注意 `MXFP4` 为大写）；纯文本，无伴随文件 |
| Nemotron-H | Nemotron-H-8B / 47B Reasoning | [8B](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF) / [47B](https://huggingface.co/bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF) |
| Nemotron-H | Nemotron 3 Nano Omni 30B-A3B | [unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF)，图像输入需 `mmproj-BF16.gguf`；仓库未附真实音频推理需要的 Parakeet mmproj |
| Nemotron 3.5 | Nemotron-3.5-Lightning-30B-A3B（23 Mamba-2 + 23 MoE + 6 注意力的混合架构） | [unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF)，例如 `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MXFP4_MOE.gguf`（MoE 专家为 MXFP4，约 17 GB）；`general.architecture` = `nemotron_h_moe`。更小 / 其他量化见 [ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF)（BF16/Q4_0/Q8_0 与独立的 MTP GGUF）。可选的提速产物见下一行的 DSpark drafter |
| Nemotron 3.5 | DSpark 投机 drafter（可选，仅提速） | [magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark-GGUF](https://huggingface.co/magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark-GGUF)，是官方 `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark` 模块的 llama.cpp DFlash 导出版（6 层 SWA、Markov head r512、attention sinks）；用 `--draft-model` 加载即可启用块级（DSpark）投机解码。也可用 `eng/nemotron-dspark-to-gguf.py` 从官方 safetensors 重新构建 |
| Mistral 3 | Mistral-Small-3.1-24B-Instruct | [bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF)，Pixtral 投影器 `mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf` |
| Hunyuan Dense | 腾讯稠密 Hunyuan 检查点（`hunyuan-dense`） | 任何 `general.architecture` 为 `hunyuan-dense` 的 GGUF 均可加载，例如 Hy-MT2 系列（参考对话模板取自 `tencent/Hy-MT2-1.8B`）。仅文本、单设备，没有投影器也没有草稿模型。见 [hunyuan-dense](docs/models/hunyuan-dense_zh-cn.md) |
| Muse-Glimmer | Muse-Glimmer-30B（稠密，支持图像） | [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF)，如 `Muse-Glimmer-30B-UD-Q4_K_XL.gguf` 或 `Muse-Glimmer-30B-Q8_0.gguf`；`general.architecture` 为 `muse-glimmer` / `muse_glimmer`。图像输入需同仓库的 `mmproj-Muse-Glimmer-30B-Q8_0.gguf`，且必须**显式**用 `--mmproj` 指定——这是唯一没有 mmproj 自动探测的系列。可选提速产物：同仓库的 DFlash 分块 draft `dflash-kquant.gguf`，用 `--draft-model` 加载即可无损推测解码——不要传任何采样参数，它只在纯贪心下生效 |
| DeepSeek V4.1 | DeepSeek-V4.1-Flash（`deepseek41`，384 个路由专家） | [vcruz305/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF/tree/8e0c4de3cb6519bfc11ed69dc87184b457a57bb5)，固定 revision `8e0c4de3cb6519bfc11ed69dc87184b457a57bb5`——七个 Q2_K 分片（246.35 GiB，张量类型混合 Q2_K/Q3_K）需放在同一目录，`--model` 指向第一个分片。该文件**不能单独运行**：`eng/dsv41-prepare.py` 会依据官方 [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) 的 `config.json` / `tokenizer.json` 在分片旁生成由分词器派生的 Engram sidecar；`eng/dsv41-prepare-vision.py` 生成约 970 MB 的可选视觉伴随文件，图像与视频经 `--mmproj` 使用。服务后端为 `ggml_cuda`；`ggml_cpu` 与 `cpu` 是正确性与可移植性路径，而非服务路径——`--backend cpu` 用纯 C# 执行器 `DeepSeek4CpuExecutor` 跑完整的 V4.1 计算图，不依赖 ggml、原生库与 GPU；它同样必须准备 Engram sidecar，而视觉伴随文件不会跟到这个后端上（`LoadVisionEncoder` 会抛异常），因此该后端上没有图像也没有视频。V4 的草稿模型会被拒绝。完整流程与校验哈希见 [deepseek41](docs/models/deepseek41_zh-cn.md) |
| DeepSeek V4 | DeepSeek-V4-Flash-0731（284B MoE） | [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)；每种量化一个子目录（`UD-Q8_K_XL/`、`UD-IQ4_XS/` 等），均为多分片，`--model` 指向 `-00001-of-` 分片。仅文本 |
| GLM 5.x | GLM-5.2（744B-A40B MoE，内嵌 NextN MTP） | [unsloth/GLM-5.2-GGUF](https://huggingface.co/unsloth/GLM-5.2-GGUF)；每种量化一个子目录（`UD-Q4_K_XL/`、`UD-IQ2_XXS/` 等），均为多分片，`--model` 指向 `-00001-of-` 分片。**仅文本**——再往下两行的 GLM-5.3-Flash 才是支持图像的那个；中间那行的 GLM-5.3 同样仅文本。这些 GGUF 已带有服务端 `--spec` 所需的 NextN 块——与 Qwen 3.6 不同，不存在需要挑选的独立 MTP 仓库 |
| GLM 5.x | GLM-5.3（`glm-dsa`，256 个路由专家，仅文本） | [unsloth/GLM-5.3-GGUF](https://huggingface.co/unsloth/GLM-5.3-GGUF)；每种量化一个子目录（`UD-Q2_K_XL/` 等），均为多分片，`--model` 指向 `-00001-of-` 分片。`general.architecture` 为 `glm-dsa`，层结构与 GLM-5.2 一致（79 个 block —— 78 层主干加 1 个 NextN ——256 个路由专家 top-8、带 lightning indexer 的 MLA、rope base 8e6），因此直接走现有的 GLM-5.2 路径，无需额外开关。**仅文本**——与下面的 Flash 仓库不同，这个仓库完全没有发布 mmproj。它确实带着供 `--spec` 使用的 NextN 块，但 `blk.78` 没有自己的 `nextn.shared_head_head.weight`，draft 块只能借用主干的 LM head——而在 `--tp N` 下该 head 是按列切分的。加载器拒绝用某个 rank 上的词表切片来 draft，并在 stderr 上明说，所以只有**不带** `--tp`（即默认按层切分到所有可见 GPU）运行时 `--spec` 才会真正生效 |
| GLM 5.x | GLM-5.3-Flash（320B，288 个路由专家，文本 + 图像） | [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)；每种量化一个子目录（`UD-Q2_K_XL/` 等），均为多分片，`--model` 指向 `-00001-of-` 分片。`general.architecture` 为 `glm5next`，与 GLM-5.2 走同一个原生执行器。与 5.2 不同，它**支持图像**：同仓库的 `mmproj-BF16.gguf`（GLM-OCR ViT）启用 `--image`、多图提示与多轮图像会话。它的 NextN 块尚未接入，因此这里没有 `--spec`。不传 `--tp` 时默认按层切分到所有可见 GPU；在 GGML GPU 后端上，传入 `--tp N` 则选择仅支持本地单进程的原生张量并行 |
| DeepSeek V4 | DSpark 推测解码 draft（可选，仅提速） | 见下方 [DSpark draft 模型](#dspark-draft-模型)，用 `--draft-model` 加载，解码约 1.3-1.4 倍 |
| DiffusionGemma | diffusiongemma-26B-A4B-it | [unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF)，如 `diffusiongemma-26B-A4B-it-Q4_K_M.gguf` |
| Qwen-Image-Edit | MMDiT DiT（必需） | [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF)，如 `qwen-image-edit-2511-Q4_K_M.gguf` |
| Qwen-Image-Edit | VAE + Qwen2.5-VL（必需） | [QuantStack VAE](https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF) 中的 `VAE/Qwen_Image-VAE.safetensors` + [unsloth/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF) |
| Qwen-Image-Edit | 视觉 mmproj（可选） | [unsloth/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF) 中的 `mmproj-BF16.gguf`，用 `--qwen-image-mmproj` / `TS_QWEN_IMAGE_MMPROJ` 加载，可让编辑指令参考源图内容 |
| Qwen-Image-Edit | Lightning LoRA（可选，4/8 步） | [lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)，文件 `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`（0.850 GB）；用 `--qwen-image-lora` / `TS_QWEN_IMAGE_LORA` 加载，会自动按文件名里的步数把采样默认值切到该步数 + CFG 1.0（基础默认为 30 步、CFG 2.5） |
| MiniMax-H3 音视频生成 | 去噪器（`--model` GGUF） | **两个独立的 checkpoint，不是开关**——加载哪一个决定了它接受什么条件输入。[unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF)：`minimax_h3_fl2va_pruned-Q4_K.gguf`（10.64 GiB）用于文生视频 / 图生视频 / 首尾帧，`minimax_h3_ref2va_pruned-Q4_K.gguf`（10.60 GiB）用于身份与外观参考。另有 Q8_0（19.97 GiB）到 Q2_K（6.26 GiB）。H3 是 CFG 蒸馏模型：**必须传 `--cfg 1.0`**，步数取 4-8。这些 GGUF **完全没有元数据**，TensorSharp 靠张量表识别它们，并从文件名读出分区——重命名或重新量化时请保留 `fl2va` / `ref2va`。两个 checkpoint 共用下面三个网络，所以事后再加另一个只需下它自己的约 10.6 GiB |
| MiniMax-H3 音视频生成 | Qwen3-VL-32B 文本编码器（必需） | 同仓库：`qwen3vl_32b_minimax_h3-Q4_K_M.gguf`（16.97 GiB），或 `-Q2_K_M.gguf`（12.20 GiB）以搭配最小的那几个去噪器。截断到 50 层并去掉最后的 norm，去噪开始前即从显存释放。**它不含分词器**——还需要下一行那两个文件 |
| MiniMax-H3 音视频生成 | `vocab.json` + `merges.txt`（必需） | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/processor)——编码器 GGUF 缺的那对 Qwen2 字节级 BPE 文件，也是配置文件唯一无法替你自动下载的东西（自动下载只能补齐“是参数”的条目，而分词器不是）。放在编码器旁边，或用 `TS_VIDEO_TOKENIZER` 指向存放它们的目录 |
| MiniMax-H3 音视频生成 | 视频 VAE（必需） | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) 里的 `minimax_h3_video_vae_fp16.safetensors`（5.21 GB）。空间 16 倍 / 时间 4 倍，解码器是纯 Transformer。放在去噪器旁或用 `--video-vae` 指定 |
| MiniMax-H3 音视频生成 | 音频 VAE（可选） | 同一目录下的 `minimax_h3_audio_vae_fp32.safetensors`（0.61 GB）。把联合生成的音频 latent 解码成 32 kHz 立体声，作为旁挂 `.wav` 写在视频边上。**不下载也照样出视频**，只是没有声音。用 `--audio-vae` 指定 |
| Wan 视频生成 | **步数蒸馏 DiT（首选）** | **这是最大的提速手段——除非要复现参考样例，都应该用它。**蒸馏 checkpoint 生成同一段视频只跑 4 次去噪，而官方配方要跑 100 次：在 M5 Pro / `ggml_metal` 上以 1088×832×121 帧实测，端到端 **17 分 30 秒**，而基础 checkpoint 是 **3 小时 30 分**——同一个请求，其他参数一律不变。TI2V-5B：[hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF](https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF)，文件 `Wan2_2-TI2V-5B-Turbo-Q8_0.gguf`（5.40 GB），另有 Q6_K（4.22 GB）、Q5_K_M（3.82 GB）、Q4_K_M（3.44 GB），最小到 Q2_K（1.86 GB）。**注意文件名里是 `Wan2_2` 下划线**，照抄基础仓库的 `Wan2.2` 写法会 404。I2V-A14B：[jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF](https://huggingface.co/jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF)，Lightning 已合并进两个专家；需同时下载 `high_noise/wan2.2_i2v_A14b_high_noise_lightx2v_4step-Q4_K_M.gguf` **和** `low_noise/wan2.2_i2v_A14b_low_noise_lightx2v_4step-Q4_K_M.gguf`（各 9.66 GB；Q8_0 15.42 GB，Q2_K 5.31 GB），放在同一个 `--local-dir` 下，`--model` 指向任意一个即可，另一个专家会自动找到。备选：[Green-Sky/FastWan2.2-TI2V-5B-FullAttn-GGUF](https://huggingface.co/Green-Sky/FastWan2.2-TI2V-5B-FullAttn-GGUF)（`FastWan2.2-TI2V-5B-q8_0.gguf`，5.41 GB）。**无需任何参数**：TensorSharp 读取 DiT 文件名，命中 `turbo` / `distill` / `lightning` / `lightx2v` / `fastwan` / `-dmd` 或显式的 `<N>steps`（1-16）即切换到该步数并关闭 guidance，加载时打印 `step-distilled checkpoint detected -> N steps, guidance off`；`--diffusion-steps` / `--cfg` 可覆盖。Turbo 与 A14B 蒸馏仓库都不含 VAE 和文本编码器，请从下面两行获取 |
| Wan 视频生成 | 基础 DiT（`--model` GGUF） | 完整官方配方（50 步 × 2 次 CFG = 100 次 DiT 前向）——需要对齐参考样例时才用，否则优先用上一行的蒸馏版本。Wan 2.2 文/图生视频：[QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF)（`Wan2.2-TI2V-5B-Q8_0.gguf` 5.40 GB 或 `Wan2.2-TI2V-5B-Q4_K_M.gguf` 3.43 GB，仓库自带 `VAE/Wan2.2_VAE.safetensors`）、[QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF) 或 [QuantStack/Wan2.2-T2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF)（`HighNoise/` 与 `LowNoise/` 两个专家缺一不可，两个仓库都自带 `VAE/Wan2.1_VAE.safetensors`）；Wan 2.1 文生视频：[samuelchristlie/Wan2.1-T2V-1.3B-GGUF](https://huggingface.co/samuelchristlie/Wan2.1-T2V-1.3B-GGUF)（`Wan2.1-T2V-1.3B-Q8_0.gguf` / `-F16.gguf`）或 [city96/Wan2.1-T2V-14B-gguf](https://huggingface.co/city96/Wan2.1-T2V-14B-gguf)（文件名为小写，如 `wan2.1-t2v-14b-Q8_0.gguf`）——这两个 2.1 仓库都不含 VAE 和编码器。`general.architecture` 为 `wan` / `wan2.1` / `wan2.2`。参见 [docs/models/wan.md](docs/models/wan.md) |
| Wan 视频生成 | UMT5-XXL 文本编码器（必需，所有 Wan checkpoint 都要） | [city96/umt5-xxl-encoder-gguf](https://huggingface.co/city96/umt5-xxl-encoder-gguf)：`umt5-xxl-encoder-Q8_0.gguf`（6.04 GB），内存紧张可用 `umt5-xxl-encoder-Q5_K_M.gguf`（4.15 GB）/ `umt5-xxl-encoder-Q4_K_M.gguf`（3.66 GB）。负责把提示词编码成条件向量，去噪开始前即从显存释放。放在 DiT 旁或用 `--video-text-encoder` / `TS_WAN_TE` 指定 |
| Wan 视频生成 | 视频 VAE（必需） | 把 latent 解码成画面——**用哪个由 DiT 自己决定**，不是由你选：TI2V-5B 需要 [`Wan2.2_VAE.safetensors`](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/tree/main/VAE)（TI2V-5B 仓库自带），Wan 2.1 与 A14B 需要 `Wan2.1_VAE.safetensors`——两个 QuantStack A14B 仓库里就有 `VAE/Wan2.1_VAE.safetensors`，也可单独下载 [`wan_2.1_vae.safetensors`](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors)。上面的蒸馏仓库都不含 VAE，请从这里配一个对应的。放在 DiT 旁（`VAE/` 子目录亦可）或用 `--video-vae` / `TS_WAN_VAE` 指定 |

### DSpark draft 模型

[DSpark](docs/models/deepseek4.md#dspark-speculative-decoding) 是 DeepSeek 的分块推测解码
draft 模型。TensorSharp 目前在 **DeepSeek V4** 上支持它，两个 GPU 引擎均可用
（`--backend cuda` 与 `--backend ggml_cuda`）：draft 是独立的 GGUF，用 `--draft-model`
加载；由主干逐块验证，贪心输出不变。

以下三个任选其一，均可直接加载（加载器兼容各发布者的张量/元数据命名）。draft 读取主干的
隐藏状态，因此与模型**同一 checkpoint 版本**的 draft 接受率更高：

| Draft | 大小 | 适配 | 说明 |
|---|---|---|---|
| [bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF](https://huggingface.co/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF) | 7.0 GB | **0731** 版本请用 `DSpark-drafter-Q2K-Q8-0731.gguf`（同仓库另有非 0731 版本） | Q2_K 专家 + Q8_0 稠密层；实测接受率 71% |
| [sakamakismile/DeepSeek-V4-Flash-DSpark-support-ds4-GGUF](https://huggingface.co/sakamakismile/DeepSeek-V4-Flash-DSpark-support-ds4-GGUF) | 5.6 GB | 0731 之前的 `DeepSeek-V4-Flash` | 体积最小；即使搭配 0731 主干也最快（接受率 69%）——draft 的带宽开销比精度更重要 |
| [alessandrobologna/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF](https://huggingface.co/alessandrobologna/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF) | 10.9 GB | **0731** 版本 | MXFP4 专家（对 checkpoint FP4 的无损重排）；精度最高、显存占用最大 |

也可以从任何带该模块的 DeepSeek V4 checkpoint 自行转换（只需下载其三个 `mtp.*` 分片，约
11 GB）：见 [Getting a drafter](docs/models/deepseek4.md#getting-a-drafter) 与
`eng/dsv4-dspark-to-gguf.py`。

**Gemma 4 的 DSpark draft 暂不支持。** DeepSeek 也发布了 Gemma 4 的 DSpark
draft，社区亦有 GGUF 转换，但它们是另一种 draft 结构：5 层 Transformer + 对五个目标层做
`fc` 融合（`general.architecture` 为 `dspark` 或 `dflash`，block_size 7），而非 DeepSeek V4
的三个超连接块（`mtp.*`）。TensorSharp 会明确报错而不会错误加载。这里列出以便了解上游现状：

> 这套 5 层 `fc` 融合结构**已经**在 Muse-Glimmer 上实现——见
> [DFlash 投机解码](docs/models/muse-glimmer_zh-cn.md#3-dflash-投机解码)。
> 下表这些 draft 没有接入，是因为它们的编码器需要目标模型暴露逐层输入残差，
> 目前只有 `MuseGlimmerModel` 做到了这一点。

| 主干 | 官方 checkpoint（safetensors） | 社区 GGUF |
|---|---|---|
| Gemma-4-12B | [deepseek-ai/dspark_gemma4_12b_block7](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7) | [ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF](https://huggingface.co/ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF)（1.9 GB）、[williamliao/dspark_gemma4_12b-GGUF](https://huggingface.co/williamliao/dspark_gemma4_12b-GGUF) |
| Gemma-4-26B-A4B | — | [williamliao/dspark_gemma4_26b-a4b-it-GGUF](https://huggingface.co/williamliao/dspark_gemma4_26b-a4b-it-GGUF) |
| Gemma-4-31B | — | [williamliao/dspark_gemma4_31b-it-GGUF](https://huggingface.co/williamliao/dspark_gemma4_31b-it-GGUF) |

Gemma 4 目前已有可用的推测解码路径：上表中的 `gemma4-assistant` MTP draft（
`--draft-model`）；Qwen 3.6、GLM 5.2 与 GLM-5.3 则内置 NextN 块（GLM-5.3 只在默认按层切分、
即不传 `--tp` 时才会真正 draft）。它们与 DSpark 是不同的 draft。

### 按模型下载并运行

以下命令从仓库根目录运行；请先按平台安装完整的 [.NET 10 SDK](DEVELOPMENT_zh-cn.md#安装-net-10-sdk)，再执行 `dotnet build TensorSharp.slnx -c Release`。仅安装 Runtime 无法构建下方使用的二进制文件。`hf` 来自 Hugging Face CLI（`pip install -U huggingface_hub`），所有文件都会下载到 `./models`。通用提示：单次文本提示词通过 `--input` 文件传入（`--prompt` 用于 Qwen-Image-Edit 的编辑指令，以及视频生成——MiniMax-H3 与 Wan——的提示词）；CLI 默认贪心采样，且不加 `--max-tokens` 时只生成 100 个 token；服务端固定监听 **http://localhost:5000**。按硬件把示例中的 `ggml_cuda` 换成 `ggml_metal`、`ggml_vulkan` 或 `ggml_cpu`（见 [选择后端](README_zh-cn.md#选择后端)）。

```bash
echo "列出三条关于月球的事实。" > prompt.txt
```

**DeepSeek V4.1 Flash**（384 个路由专家，服务后端为 `ggml_cuda`，必须先准备 Engram sidecar）：

```bash
# 七个分片共 246 GiB Q2_K 权重，Engram 预热还需要约 60 GiB 主机页缓存
python3 -m venv /tmp/dsv41-tools
/tmp/dsv41-tools/bin/python -m pip install numpy==2.0.2 tokenizers==0.22.2 huggingface_hub gguf
hf download vcruz305/DeepSeek-V4.1-Flash-GGUF \
    --revision 8e0c4de3cb6519bfc11ed69dc87184b457a57bb5 \
    --include "DeepSeek-V4.1-Flash-Q2_K-*.gguf" --local-dir models/deepseek41-q2

# 必需：由官方 config/tokenizer 派生的 Engram sidecar
/tmp/dsv41-tools/bin/python eng/dsv41-prepare.py models/deepseek41-q2 \
    --repo deepseek-ai/DeepSeek-V4.1-Flash \
    --revision dba1be0a40aa45a94ad051997016db3960a90277

# 也可以改用 Q4_K_M：十一个分片共 415 GiB。它的两张 Engram 表各 51.5 GiB，
# 只能留在主机内存映射中，因此在 8x46 GB 上必须把路由专家卸载到 CPU，
# 加载器会打印它需要的 --n-cpu-moe N。
#   --include "DeepSeek-V4.1-Flash-Q4_K_M-*.gguf" --local-dir models/deepseek41-q4
# 下面的 sidecar 步骤对每一种量化都是必需的：社区 GGUF 仓库都不附带
# deepseek41.engram.bin。

# 可选：约 970 MB 的视觉伴随文件，--mmproj 用它启用图像与视频
/tmp/dsv41-tools/bin/python eng/dsv41-prepare-vision.py models/deepseek41-q2 \
    --repository deepseek-ai/DeepSeek-V4.1-Flash \
    --revision dba1be0a40aa45a94ad051997016db3960a90277

dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model models/deepseek41-q2/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf \
    --mmproj models/deepseek41-q2/deepseek41.vision.gguf \
    --backend ggml_cuda --tp 8 --port 5000
```

这里的 `--tp N` 表示在 N 张 GPU 上**按层切分**，不是张量并行；`TS_DSV41_TP=N` 才会打开实验性的
routed-MoE TP，而它目前实测比按层切分更慢。权重与上下文放不下时加 `--n-cpu-moe N`。
Python 只用于准备 sidecar，推理阶段不需要。

`--backend cpu` 不是上面“换后端”提示里的 `ggml_cpu`：它用纯 C# 执行器 `DeepSeek4CpuExecutor`
跑完整的 V4.1 计算图——不依赖 ggml、原生库与 GPU，凡是 .NET 能跑的地方它都能跑——定位是正确性
与可移植性路径，而非服务路径；整份 checkpoint 在它上面的吞吐、加载时间与常驻内存都从未实测过。
上面的 `deepseek41.engram.bin` sidecar 依然必需。`--mmproj` 在这里完全不可用（视觉伴随文件是
原生 ggml 组件，`LoadVisionEncoder` 会抛异常），所以没有图像也没有视频；分布式 TP 组、任何草稿
模型或 `TS_DSV4_DSPARK`、非 `0` 的 `TS_DSV41_TP`、非 `0` 的 `TS_DSV41_ENGRAM_DEVICE`，都会在
读取任何权重之前被拒绝。它也没有多轮 KV 前缀复用、没有按序列的 slot——每次分叉的对话都要重新
prefill，并发请求只能串行。

**DeepSeek V4 Flash**（284B MoE，纯文本，支持 DSpark 推测解码）：

```bash
# 约 160 GB 权重：需要多张 GPU（自动按层切分），draft 另需约 7 GB
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF --include "UD-Q8_K_XL/*" --local-dir models
hf download bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF DSpark-drafter-Q2K-Q8-0731.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model models/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf \
    --backend ggml_cuda --draft-model models/DSpark-drafter-Q2K-Q8-0731.gguf \
    --input prompt.txt --max-tokens 200 --temperature 0
```

去掉 `--draft-model` 即为普通解码。CLI 上的推测解码要求纯贪心采样（`--temperature 0`）；
`--spec-pmin` 控制每个块草拟到多深。

**GLM 5.x**（GLM-5.3，`glm-dsa`，256 个路由专家，仅文本，内嵌 NextN 块）：

```bash
# UD-Q2_K_XL 是七个分片、236.4 GiB——需要一台*合计*显存能装下它再加 KV cache 的机器；
# 实测配置为八张 46 GB A40，按层切分
hf download unsloth/GLM-5.3-GGUF --include "UD-Q2_K_XL/*" --local-dir models

dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model models/UD-Q2_K_XL/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf \
    --backend ggml_cuda --spec --port 5000
```

这里没有 `--mmproj`：该仓库在任何量化下都没有发布 mmproj，而给 `glm-dsa` 模型传 `--mmproj`
只会告警并被忽略，不会报错退出，所以无论如何都是仅文本的运行。`--spec` 必须在加载之前就写在
命令行上，并且只在上面这种**默认按层切分**下生效——不传 `--tp` 时会用上所有可见 GPU。
`--tp N` 在 GGML GPU 后端上可以接受，但仅限本地单进程（`--tp-node-id` / `--tp-peers` 对整个
GLM 系列都会在构建模型之前被拒绝），而且会在每个 rank 上复制一份 KV cache；它对 GLM-5.3 并不是
已验证的配置，并且在它之下加载器会放弃 draft——`blk.78` 借用的主干 LM head 是按列切分的——
转为普通解码。GLM-5.2 与 GLM-5.3-Flash 的下载方式相同，仓库见上表。详见
[glm](docs/models/glm_zh-cn.md#glm-53glm-dsa)。

**Gemma 4**（文本 + 图像/视频/音频、思维链、工具、可选 MTP）：

```bash
hf download ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q8_0.gguf --local-dir models
hf download ggml-org/gemma-4-E4B-it-GGUF mmproj-gemma-4-E4B-it-Q8_0.gguf --local-dir models
hf download AtomicChat/gemma-4-E4B-it-assistant-GGUF gemma-4-E4B-it-assistant.Q8_0.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gemma-4-E4B-it-Q8_0.gguf --mmproj models/mmproj-gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/gemma-4-E4B-it-Q8_0.gguf --mmproj models/mmproj-gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --draft-model models/gemma-4-E4B-it-assistant.Q8_0.gguf
```

第三个下载与 `--draft-model` 参数可省略。

**Qwen 3.5 / 3.6**（图像、思维链、工具；3.6 可用 NextN）：

```bash
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf --local-dir models
hf download unsloth/Qwen3.5-9B-GGUF mmproj-F16.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/Qwen3.5-9B-UD-Q4_K_XL.gguf --mmproj models/mmproj-F16.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/Qwen3.5-9B-UD-Q4_K_XL.gguf --mmproj models/mmproj-F16.gguf --backend ggml_cuda

# 3.6 必须从保留 NextN 块的 -MTP- 仓库下载
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --local-dir models
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --backend ggml_cuda --spec
```

**GPT OSS**（文本、始终思考、工具）：

```bash
hf download ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gpt-oss-20b-MXFP4.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/gpt-oss-20b-MXFP4.gguf --backend ggml_cuda
```

**Nemotron-H**（文本、思维链、工具）：

```bash
hf download bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --backend ggml_cuda
```

图像输入请改用 Omni 发行版：从 [unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF) 下载 `NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf` 与 `mmproj-BF16.gguf`；当前发行版没有真实音频推理所需的 Parakeet audio mmproj。

**Hunyuan Dense**（腾讯稠密 Hunyuan / Hy-MT2，仅文本，单设备）：

这里不固定任何仓库：只要 GGUF 的 `general.architecture` 是 `hunyuan-dense` 就能加载，
架构由该键决定，与文件名无关。

```bash
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model models/<your-hunyuan-dense>.gguf \
    --backend ggml_cuda --input prompt.txt --max-tokens 200
```

没有投影器，没有草稿模型；多余的 GPU 会闲置，启动时会明确提示而不是悄悄占用。

**Mistral 3**（文本 + 图像）：

```bash
hf download bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --local-dir models
hf download bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --mmproj models/mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --mmproj models/mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf --backend ggml_cuda
```

**DiffusionGemma**（块文本扩散）：

```bash
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF diffusiongemma-26B-A4B-it-Q4_K_M.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --input prompt.txt --max-tokens 256 --diffusion-steps 48 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --backend ggml_cuda
```

（Web UI 会实时流式展示 DiffusionGemma 的去噪过程；兼容 API 只返回最终文本。）

**Qwen-Image-Edit**（DiT + VAE + 文本编码器；Lightning LoRA 可选）：

```bash
hf download unsloth/Qwen-Image-Edit-2511-GGUF qwen-image-edit-2511-Q4_K_M.gguf --local-dir models
hf download QuantStack/Qwen-Image-Edit-GGUF VAE/Qwen_Image-VAE.safetensors --local-dir models
hf download unsloth/Qwen2.5-VL-7B-Instruct-GGUF Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf --local-dir models
hf download lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/qwen-image-edit-2511-Q4_K_M.gguf --image input.png --prompt "把天空改成壮丽的日落。" --output edited.png --qwen-image-vae models/VAE/Qwen_Image-VAE.safetensors --qwen-image-vl models/Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf --qwen-image-lora models/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/qwen-image-edit-2511-Q4_K_M.gguf --qwen-image-vae models/VAE/Qwen_Image-VAE.safetensors --qwen-image-vl models/Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf --qwen-image-lora models/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors --backend ggml_cuda
```

（在 Web UI 里上传图片并输入编辑指令即可。Lightning LoRA 的下载与 `--qwen-image-lora` 参数是可选的——加上后去噪降到 4 步、CFG 1.0。）

**MiniMax-H3 音视频生成**（提示词 + 可选关键帧或参考 → H.264 MP4，**外加原生 32 kHz 立体声音频，在同一个打包 latent 里一起生成**）：

这里有四个网络协同工作，所以最短路径是现成的配置文件——它把四个都写好了，缺什么下什么（首次运行约 33.5 GB）：

```bash
TensorSharp.Server --config config/minimax-h3-fl2va.json
TensorSharp.Cli    --config config/minimax-h3-fl2va.json \
    --prompt "a red fox trotting through falling snow, cinematic" --output fox.mp4
```

`config/minimax-h3-ref2va.json` 是另一个 checkpoint：最多九个身份与外观参考——静态图、片段、音轨——
用于一个全新的镜头，而不是必须被复现的帧。FL2VA 与 Ref2VA 是**两个独立的 checkpoint，不是一个开关**，
向其中一个索要另一个的条件输入会直接报错，并在错误信息里点名你真正需要的那个文件。两份配置只有去噪器
不同（那边约 33.4 GB），下面三个网络是共用的，所以第二份配置只会下它自己的 DiT。参见
[config/README.md](config/README.md#video-generation-with-sound-minimax-h3)。文件会落到
`TENSORSHARP_MODELS` 指向的位置，或仓库旁边的 `models/`。

有一对文件无论走哪条路都不会自动下载：文本编码器的 GGUF 不含分词器，而自动下载只能补齐“是参数”的条目。

```bash
curl -L -o models/vocab.json https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/processor/vocab.json
curl -L -o models/merges.txt https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/processor/merges.txt
```

手动路线如下。

```bash
# FL2VA 是文生视频 / 图生视频 / 首尾帧的 checkpoint；需要参考条件时换成
# minimax_h3_ref2va_pruned-Q4_K.gguf。两个 VAE 在 unsloth/MiniMax-H3-GGUF 自己的
# vae/ 目录下也有镜像，Comfy-Org 慢的时候可以换过去。
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

上面这条 CLI 命令会写出 `fox.mp4` **和 `fox.wav`**：音轨是旁挂文件，从不混流进 MP4，因为混流需要一个
不一定装了的编码器——用 `ffmpeg -i fox.mp4 -i fox.wav -c:v copy -c:a aac fox_with_audio.mp4` 合并即可。
所有文件都在同一个目录下时，三个伴随文件参数都可以省略：去噪器所在目录及其上一级会被递归扫描，子目录也算。
不下载音频 VAE，或者加上 `--no-audio`，仍然会出视频——只是没有声音。

H3 是 CFG 蒸馏模型，**必须传 `--cfg 1.0`**，更高的值会被直接拒绝；管线自身的默认是 20 步，4-8 步是快速
工作点，代价是运动主体边缘会有一些彩色条纹，到 ~20 步就消失了。宽高向上取整到 32 的倍数，帧数对齐到
`17k+5` 网格（5、22、39、56、73、90……），fps 无论你传什么都被钉死在 24。服务端的步数参数叫
`--video-steps`，而且根本没有 `--cfg`——这正是随附配置两个都不设的原因。

条件输入方面：`--image first.png` 把这张图作为首帧动起来；再加上
`--end-image last.png --video-mode fl2v` 就是在首尾两帧之间插值；而在 Ref2VA checkpoint 上，`--ref-image`（可重复，最多九个）、
`--ref-video`、`--ref-video-audio` 与 `--ref-audio` 则是把身份与外观带进一个全新的镜头。在 M5 Pro 的
Metal 上以 22 帧、8 步、相同随机种子实测，H3 比 stable-diffusion.cpp 在 256×256 下快 **2.4 倍**
（49.3 秒 → 20.9 秒），在 640×384 下快 **1.7 倍**（108.5 秒 → 63.1 秒）。参见
[docs/models/minimax-h3_zh-cn.md](docs/models/minimax-h3_zh-cn.md)。

**Wan 视频生成**（提示词 + 可选首帧图片 → H.264 MP4，仅视频；需要 DiT + 视频 VAE + UMT5-XXL 文本编码器）：

Wan 同样需要三个独立的网络，所以这里最省事的路子依然是现成的配置文件——它把三者
一并列出，缺哪个就下载哪个：

```bash
TensorSharp.Server --config config/wan-video-ti2v-5b-turbo.json
TensorSharp.Cli    --config config/wan-video-ti2v-5b-turbo.json \
    --prompt "a cute fluffy orange cat walking through a sunny garden" --output cat.mp4
```

`config/wan-video-ti2v-5b.json` 是未蒸馏的 50 步版本，`config/wan-video-i2v-a14b.json`
是双专家的 14B 图生视频模型；详见
[config/README.md](config/README.md#video-generation-video-only-wan)。文件会落到
`TENSORSHARP_MODELS` 指向的位置，或仓库旁边的 `models/`。手动下载的方式在下面。

```bash
# 步数蒸馏的 Turbo DiT：只跑 4 次去噪而不是 100 次，由文件名自动识别。
# 注意 Turbo 文件名里的 Wan2_2 下划线；VAE 和文本编码器仍需从基础仓库获取。
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

加载时控制台会打印 `step-distilled checkpoint detected -> 4 steps, guidance off`——看到这行就说明走的是快路径。
只把 `--model` 换成基础的 `Wan2.2-TI2V-5B-Q8_0.gguf`，就会按官方 50 步 + CFG 配方运行：同一个
1088×832×121 帧的请求实测为 3 小时 30 分，而这里是 17 分 30 秒（M5 Pro，`ggml_metal`）。
加 `--image first_frame.png` 即为图生视频，Web UI 里上传图片也一样（该图作为首帧）；服务端的
`--video-frames` / `--fps` 只是默认值，单个请求可以覆盖。Wan 是唯一不支持 `--backend mlx` 的系列，
请使用 `ggml_cuda`、`ggml_metal`、`ggml_vulkan`、`ggml_cpu`、`cuda` 或 `cpu`。

三个文件放在同一个目录下时（`VAE/` 子目录也算），`--video-vae` / `--video-text-encoder` 可以省略，会自动解析。
双专家的 A14B 模型需要**同时**下载两个专家到同一个 `--local-dir`，`--model` 指向其中任意一个：

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
