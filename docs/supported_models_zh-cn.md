# 支持的模型

[English](supported_models.md) | [中文](supported_models_zh-cn.md)

> [TensorSharp](../README_zh-cn.md) 文档的一部分。每个家族都有端到端的[模型卡片](models/README_zh-cn.md)，下载命令见[模型下载](../MODEL_DOWNLOADS_zh-cn.md)。

<a id="已验证模型"></a>

## 已实现模型与验证范围

以下家族均已实现。各模型卡片列出实际测试的检查点、后端与设备路径，以及尚存的限制；支持某个家族不表示已验证每种量化或设备。请选择该家族推荐的、适配硬件容量的检查点和量化。更多尺寸与投影器文件见 [模型下载](../MODEL_DOWNLOADS_zh-cn.md)。

| 家族 | 示例模型（GGUF） | 图像 / 视频 / 音频 | 思维链 | 工具 | 卡片 |
|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | [修复版 DeepSeek-V4.1-Flash Q2_K/Q5_K](https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/tree/d1de55c19f95172c882906cc83c0e55932d26a63/Q2_K-Q5)（十个分片，312.349 GiB，内嵌 Engram；有限检查与历史 Q2_K/Q4_K_M 结果分开记录，服务路径为 `ggml_cuda`；`ggml_cpu` 是仍能加载视觉伴随文件的正确性与可移植性路径，`cuda` 与纯 C# `cpu` 执行器则是仅文本的） | ✅（视觉伴随文件） / ✅（视觉伴随文件） / — | ✅ | ✅ | [deepseek41](models/deepseek41_zh-cn.md) |
| DeepSeek V4 Flash | [DeepSeek-V4-Flash-0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)（284B MoE，分片 GGUF） | — / — / — | ✅ | ✅ | [deepseek4](models/deepseek4_zh-cn.md) |
| GLM 5.x | [GLM-5.2](https://huggingface.co/unsloth/GLM-5.2-GGUF)（744B-A40B MoE，分片 GGUF）、[GLM-5.3](https://huggingface.co/unsloth/GLM-5.3-GGUF)（256 个路由专家，仅文本；每个量化档一个子目录，UD-Q2_K_XL 为 7 个分片、236.4 GiB——`--model` 指向 `-00001-of-00007` 那一片）、[GLM-5.3-Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)（320B MoE，分片 GGUF，+ mmproj） | ✅（仅 5.3-Flash；5.2 与 5.3 均仅文本） / — / — | ✅ | ✅ | [glm](models/glm_zh-cn.md) |
| Qwen 3.8 Flash Next | [Qwen3.8-Flash-Next](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)（GDN + 注意力混合 MoE，512 专家，分片 GGUF，+ mmproj） | ✅ / ✅（`video_url`） / — | ✅ | ✅ | [qwen38-flash-next](models/qwen38-flash-next_zh-cn.md) |
| Gemma 4 | [gemma-4-E4B-it](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)（另有 12B、31B、26B-A4B MoE） | ✅ / ✅ / ✅ | ✅ | ✅ | [gemma4](models/gemma4_zh-cn.md) |
| Qwen 3.5 / 3.6 | [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)（另有 35B-A3B MoE、Qwen3.8-27B） | ✅ / — / — | ✅ | ✅ | [qwen35](models/qwen35_zh-cn.md) |
| Qwen3-VL | [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF)（MoE，激活 3B；也支持稠密 `qwen3vl` 检查点，需含 DeepStack 合并器的 mmproj） | ✅ / ✅（`video_url`） / — | ✅（Thinking 检查点） | ✅ | [qwen3vl](models/qwen3vl_zh-cn.md) |
| Bonsai2 | 本地哈希钉住的 `Ternary-Bonsai-2-27B-PQ2_0.gguf` / `-PTQ1_0.gguf`（采用 PRISM 带符号 Hadamard 变换的稠密 Qwen 3.5 混合，+ mmproj）；仅支持单设备 GGML 后端，已在 Metal 上验证 | ✅ / — / — | ✅ | ✅ | [bonsai2](models/bonsai2_zh-cn.md) |
| GPT OSS | [gpt-oss-20b](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF)（MoE） | — / — / — | ✅ | ✅ | [gptoss](models/gptoss_zh-cn.md) |
| Nemotron-H | [Nemotron-H-8B](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF)（另有 47B、Omni） | ✅（Omni） / — / — | ✅ | ✅ | [nemotron](models/nemotron_zh-cn.md) |
| Mistral 3 | [Mistral-Small-3.1-24B](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF) | ✅ / — / — | — | — | [mistral3](models/mistral3_zh-cn.md) |
| Hunyuan Dense | 腾讯稠密 Hunyuan GGUF（`hunyuan-dense`），例如 Hy-MT2 系列 | — / — / — | — | — | [hunyuan-dense](models/hunyuan-dense_zh-cn.md) |
| LFM3 / LFM3-Omni | [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF)（短卷积 + 注意力混合）、[LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF)（+ 含视觉与音频编码器的 mmproj）；聊天协议已与 llama.cpp 逐字节核对，Omni 的图像 / 视频 / 音频输入已与 llama.cpp 端到端核对（Metal、MLX、CPU）；文本与媒体输入已在 NVIDIA A40 上的 ggml_cuda、直连 CUDA 与 ggml_vulkan 上与 llama.cpp 核对 | Omni：图像 / 视频（从文件以 2 fps 读取并带音轨） / 音频输入与语音输出（需投影器） | ✅（1.7B；Omni 不做推理） | ✅ | [lfm3](models/lfm3_zh-cn.md) |
| DiffusionGemma | [diffusiongemma-26B-A4B-it](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF)（视觉塔取自上游 safetensors 分片） | ✅ / — / — | —（提示中不启用） | — | [diffusiongemma](models/diffusiongemma_zh-cn.md) |
| Muse-Glimmer | [Muse-Glimmer-30B](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF)（+ mmproj） | ✅ / — / — | ✅ | ✅ | [muse-glimmer](models/muse-glimmer_zh-cn.md) |
| Muse Spark | [Muse-Spark-Open-816B-A42B](https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-GGUF)（816B-A42B MoE，每个量化档一个子目录，+ `mmproj-Q8_0.gguf`）；原生整模型执行器是 `ggml_cuda` / `ggml_vulkan` 上的默认路径，也运行 `ggml_cuda` 上的 `--tp N`；完整模型测量用 UD-IQ3_XXS 量化在十块 CUDA A40 GPU 上以按层切分和张量并行完成（全部专家在显存中） | ✅ / — / — | ✅ | ✅ | [muse-spark](models/muse-spark_zh-cn.md) |
| MiniMax-M3.1-Flash | [MiniMax-M3.1-Flash](https://huggingface.co/unsloth/MiniMax-M3.1-Flash-GGUF)（428B-A23B MoE，`UD-Q4_K_XL/` 为七个分片 / 232.3 GiB；mmproj 与 DSpark 草稿器需从 Hugging Face 发布版转换，见卡片）；仅 `ggml_cuda`（`ggml_cpu` 为正确性路径），完整模型测量在八块 CUDA A40 GPU 上以按层切分、张量并行与 DSpark 投机完成 | ✅ / ✅（抽样帧，未测量） / — | ✅ | ✅ | [minimax-m3](models/minimax-m3_zh-cn.md) |
| Qwen-Image-2.1 | [Qwen-Image-2.1 GGUF](https://huggingface.co/Abiray/Qwen-Image-2.1-GGUF)（DiT + 专用 2.1 VAE + Qwen3-VL-8B）；Unsloth 不带元数据的 Q8_0 DiT 也可加载，靠张量表识别（图像编辑需用 `--qwen-image-mmproj` 指定其 `mmproj-BF16.gguf`） | 🖼️ 文本→图像、参考图编辑与遮罩局部编辑；RGBA；LoRA 插件（`--lora`，含 4–8 步蒸馏） | — | — | [qwenimage21](models/qwenimage21_zh-cn.md) |
| MiniMax-H3 音视频 | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF)（去噪器 + Qwen3-VL-32B 文本编码器）+ [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)（视频 VAE + 音频 VAE） | 🎬🔊 文本→视频、图像→视频、首尾帧、参考（图像/片段/音轨）→视频，**带立体声音频** | — | — | [minimax-h3](models/minimax-h3_zh-cn.md) |
| LTX 2.5 Distilled（实验性） | [Abiray/LTX-2.5-Distilled-GGUF](https://huggingface.co/Abiray/LTX-2.5-Distilled-GGUF) + 专用 [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) 编码器、VAEs 与 x2 潜变量上采样器 | 🎬🔊 文本/图像→视频 + 48 kHz 立体声音频；两阶段流程，CUDA/MLX 路径 | — | — | [ltx-2.5](models/ltx-2.5_zh-cn.md) |
| Wan 2.1 / 2.2 视频 | [Wan2.2-TI2V-5B](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF)（另有 [T2V-A14B](https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF)、[I2V-A14B](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF)、[Wan2.1-T2V-14B](https://huggingface.co/city96/Wan2.1-T2V-14B-gguf)）+ UMT5-XXL + 视频 VAE · 快速路径：[TI2V-5B-Turbo](https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF)（4 步，DiT 前向次数减少 25×） | 🎬 文本→视频、图像→视频 | — | — | [wan](models/wan_zh-cn.md) |

## 支持的模型架构

| 架构 | GGUF 架构标识 | 示例模型 | 多模态 | 思维链 | 工具调用 | MTP 投机 | 卡片 |
|---|---|---|---|---|---|---|---|
| BERT / XLM-R 嵌入 | `bert` | Snowflake Arctic Embed L v2.0、all-MiniLM-L6-v2 | 文本 → 向量 | — | — | — | [嵌入指南](embeddings_zh-cn.md) |
| EmbeddingGemma 2 | `gemma-embedding2` | Q8_0 / BF16 文本 GGUF，可选 Gemma 4 媒体投影器 | 文本/代码；配 `--mmproj` 的图像、音频、抽样视频；8192 token，归一化 768/512/256/128 维；六条后端路径实现，设备验证与待完成性能范围见指南 | — | — | — | [嵌入指南](embeddings_zh-cn.md) |
| DeepSeek V4.1 Flash | `deepseek41` | DeepSeek-V4.1-Flash（40 层，384 个路由专家 top-6 加一个共享专家，四条残差流与延迟 hyper-connection 混合，Engram n-gram 特征，声明 1M 上下文） | 文本；配合视觉伴随文件（`--mmproj`：准备好的 `deepseek41.vision.gguf` 或 llama.cpp 的 `deepseek41v` mmproj）支持图像与视频，音频请求被拒绝 | 支持 | 支持（带空格的 DSML，受语法约束） | 实验性：可在 `ggml_cuda`/`ggml_cpu` 上通过 `--draft-model` 加载 `deepseek41-dspark` 或 unsloth 的 `dflash` 草稿模型；unsloth 1017 的 Q8_0 草稿器在代码上接受率 80%，摘要任务 decode 在 `--layer-split 5` 下提速 1.31 倍、在 `--tp 5` 下提速 1.55 倍（五张 A40）；V4 的草稿模型会被拒绝 | [deepseek41](models/deepseek41_zh-cn.md) |
| DeepSeek V4 Flash | `deepseek4` | DeepSeek-V4-Flash（284B MoE，256 专家，压缩稀疏注意力，1M 上下文） | 仅文本 | 支持 | 支持（DSML） | 支持（DSpark 块级草稿，独立 GGUF） | [deepseek4](models/deepseek4_zh-cn.md) |
| GLM 5.x | `glm-dsa`、`glm_dsa`、`glm5next` | GLM-5.2（744B-A40B MoE，256 专家，MLA + DeepSeek 稀疏注意力，1M 上下文）、[GLM-5.3](models/glm_zh-cn.md#glm-53glm-dsa)（与 5.2 完全相同的 79 层 `glm-dsa` 形态——78 层主干加 1 个 NextN，256 个路由专家 top-8 外加 1 个共享专家，带 lightning indexer 的 MLA，rope base 8e6——因此直接走 GLM-5.2 的加载路径，既不需要新代码也不需要新开关；仅文本）、GLM-5.3-Flash（320B MoE，288 专家，KDA 线性注意力 + NoPE MLA 与池化索引器） | 仅文本（5.2 与 5.3）、图像（5.3-Flash） | 支持 | 支持（XML 工具调用） | GLM-5.2 与 GLM-5.3 支持（内嵌 NextN 块；5.3 上投机在单设备或显式 `--layer-split N` 模式下生效，不启用张量并行） | [glm](models/glm_zh-cn.md) |
| Qwen 3.8 Flash Next | `qwen4exp` | Qwen3.8-Flash-Next（混合 MoE，512 专家 / 激活 10 个，48 层中 36 层为 GatedDeltaNet 并与 QSA 索引的全注意力层交错，PLE n-gram 块，×4 超连接） | 图像、视频（`video_url`） | 支持 | 支持（Qwen XML / JSON 工具调用） | 支持（共享 MTP 头，独立 GGUF，经 `--draft-model` 加载；可在其可加载的所有后端上使用） | [qwen38-flash-next](models/qwen38-flash-next_zh-cn.md) |
| Gemma 4 | `gemma4` | gemma-4-E4B、gemma-4-12B、gemma-4-31B、gemma-4-26B-A4B（MoE） | 图像、视频、音频 | 支持 | 支持 | 支持（独立草稿 GGUF） | [gemma4](models/gemma4_zh-cn.md) |
| Qwen 3.5 / 3.6 family | `qwen35`, `qwen35moe`, `qwen3next` | Qwen3.5-9B（混合 Attn+递归）、Qwen3.5/3.6-35B-A3B（MoE）、Qwen3.8-27B（稠密混合） | 图像 | 支持 | 支持 | 支持：Qwen 3.6 与 Qwen 3.8 27B 内嵌 NextN（`--spec`）；Qwen 3.8 27B 另可经 `--draft-model` 加载 DFlash2 块级草稿（独立 GGUF） | [qwen35](models/qwen35_zh-cn.md) |
| Qwen3-VL | `qwen3vl`, `qwen3vlmoe` | Qwen3-VL-30B-A3B（MoE，128 专家 / 激活 8）、稠密 Qwen3-VL（交错 M-RoPE，DeepStack 注入前三层） | 图像、视频（`video_url`） | Thinking 检查点 | 支持（Hermes JSON） | 无权重 n-gram（`--spec --spec-type ngram`） | [qwen3vl](models/qwen3vl_zh-cn.md) |
| Bonsai2（Qwen 家族） | 带 `prism.hadamard.*` 元数据与 PQ2_0 / PTQ1_0 张量的 `qwen35` | Ternary-Bonsai-2-27B PQ2_0 / PTQ1_0（64 层稠密 Qwen 3.5 混合；加载时无损重打包为 GGML Q2_0；仅支持单设备 GGML 后端） | 图像（伴随 mmproj） | 支持 | 支持 | — | [bonsai2](models/bonsai2_zh-cn.md) |
| GPT OSS | `gptoss`, `gpt-oss` | gpt-oss-20b（MoE） | 仅文本 | 支持（始终） | 支持 | — | [gptoss](models/gptoss_zh-cn.md) |
| Nemotron-H | `nemotron_h`, `nemotron_h_moe`, `nemotron_h_omni` | Nemotron-H-8B/47B（混合 SSM-Transformer，MoE）、Nemotron 3 Nano Omni、Nemotron 3.5 Lightning 30B-A3B（23 Mamba-2 + 23 MoE + 6 注意力） | 图像（Omni）；音频仅在加载自行转换的 Parakeet 音频伴随 GGUF（`--mmproj` 或 `TS_NEMOTRON_AUDIO_MMPROJ`）时可用，否则拒绝 | 支持 | 支持 | 不支持（拒绝：verify 与 decode 内核不同，投机会改变输出） | [nemotron](models/nemotron_zh-cn.md) |
| Mistral 3 | `mistral3`；以及[标记为 `llama` 的 Mistral Small 3.x 文件](models/mistral3_zh-cn.md#标记为-llama-的文件)（Tekken 分词器，含 `[INST]`/`[SYSTEM_PROMPT]` 控制 token） | Mistral-Small-3.1-24B-Instruct | 图像 | 不支持 | 不支持 | — | [mistral3](models/mistral3_zh-cn.md) |
| Hunyuan Dense | `hunyuan-dense` | 腾讯稠密 Hunyuan 解码器，例如 Hy-MT2（GQA，per-head QK-norm 在 NeoX RoPE **之后**，SwiGLU） | 仅文本 | 不支持 | 不支持 | — | [hunyuan-dense](models/hunyuan-dense_zh-cn.md) |
| LFM3 / LFM3-Omni | `lfm3` | LFM3-1.7B（24 层：13 个门控短卷积层 + 11 个 GQA 注意力层，其中 5 个为 256 滑动窗口，注意力输出带 sigmoid 门）；LFM3-Omni-2B（同一解码器 + 一个 mmproj：`lfm3vl` 视觉、`lfm3a` 音频、`lfm3a_gen` 语音生成器） | Omni：图像、视频（2 fps 帧对并带音轨）、音频输入；语音输出（TTS 与语音聊天回复） | 支持（1.7B；其模板默认开启，TensorSharp 请求默认关闭） | 支持（Qwen3-Coder XML） | —（没有公开的 LFM3 草稿器） | [lfm3](models/lfm3_zh-cn.md) |
| Muse-Glimmer | `muse-glimmer`、`muse_glimmer` | Muse-Glimmer-30B（交错滑动窗口 + NoPE 全注意力层，注意力输出门控） | 图像 | 支持 | 支持（ATEM） | 支持（DFlash 块级草稿，独立 GGUF；n-gram 无需草稿模型） | [muse-glimmer](models/muse-glimmer_zh-cn.md) |
| Muse Spark | `muse-spark`、`muse_spark` | Muse-Spark-Open-816B-A42B（62 层：46 个滑动窗口层 + 16 个 NoPE 全局层；256 个路由专家 top-8，sigmoid 路由，专家在 4096 宽的潜空间中计算） | 图像（需配套的 Spark mmproj）；不支持音频 | 支持 | 支持（ATEM） | 无训练草稿器；可用无权重的 n-gram（`--spec --spec-type ngram`） | [muse-spark](models/muse-spark_zh-cn.md) |
| MiniMax-M3.1 | `minimax-m3`、`minimax_m3` | MiniMax-M3.1-Flash（60 层：3 个稠密层 + 57 个 MoE 层，128 个路由专家 top-4 加 1 个共享专家；每层都是 MiniMax 稀疏注意力，位置 2048 之前精确稠密；带逐专家缩放的 NVFP4 专家） | 图像（转换得到的 `minimax_m3` mmproj），视频以抽样帧输入；拒绝音频 | 支持（自适应；`think` 可强制开启） | 支持（命名空间 XML，`tool_choice` 由语法强制） | DSpark 块级草稿器（独立 `dflash` GGUF，通过 `--draft-model`，仅按层切分）；n-gram 通过 `--spec --spec-type ngram` | [minimax-m3](models/minimax-m3_zh-cn.md) |
| DiffusionGemma | `diffusion-gemma`、`diffusion_gemma` | diffusion-gemma 文本扩散 GGUF | 聊天支持图像；[Jev](models/jev_zh-cn.md) 还支持文档、抽样视频帧和已配置 ASR 配套服务的语音转录 | 不支持（提示中不启用） | 不支持（会被拒绝） | — | [diffusiongemma](models/diffusiongemma_zh-cn.md) |
| Qwen-Image-2.1 | `qwen_image`、`qwen-image`（通过张量键识别 2.1；更早的 Qwen-Image / Edit-2511 checkpoint 会在加载时被拒绝） | Qwen-Image-2.1 DiT GGUF（+ 专用 2.1 VAE 与 Qwen3-VL-8B） | 文本→图像、参考图编辑与遮罩局部编辑，RGBA 输出；GGML 与纯 C# `cpu` 路径；LoRA 插件；前缀 KV 缓存默认开启；DiT 张量并行（`--tp`，GGML CUDA/Vulkan；Vulkan 上实测双卡比单卡更慢） | 不支持 | 不支持 | — | [qwenimage21](models/qwenimage21_zh-cn.md) |
| MiniMax-H3 | `minimax-h3`、`minimax_h3`（官方发布的 GGUF 完全没有元数据，因此靠张量表识别） | MiniMax-H3 FL2VA / Ref2VA（193 亿参数的打包音视频 DiT + Qwen3-VL-32B 文本编码器、视频 VAE、音频 VAE） | 视频输出 **+ 32 kHz 立体声音频**（文本→视频、图像→视频、首尾帧、参考→视频） | 不支持 | 不支持 | — | [minimax-h3](models/minimax-h3_zh-cn.md) |
| LTX 2.5 Distilled | `ltx-2.5`、`ltx`、`ltxv`、`ltx2`、`ltx-2`、`ltx_2_5`；无元数据时按张量识别 | 完整 48 块 AV 去噪器 + 专用 Gemma 4 编码器/投影 + x2 潜变量上采样器 + 卷积视频/音频 VAEs | 文本/图像→视频 + 48 kHz 立体声音频；半尺寸八步 ancestral Euler、x2 上采样、全尺寸三步细化，CFG 1；首尾/定时关键帧和可选 Ingredients 身份参考；实验性 | 不支持 | 不支持 | — | [ltx-2.5](models/ltx-2.5_zh-cn.md) |
| Wan 视频 | `wan`、`wan2.1`、`wan2.2` | Wan 2.1 T2V 1.3B/14B、Wan 2.2 TI2V-5B、Wan 2.2 A14B T2V/I2V（双专家） | 视频输出（文本→视频、图像→视频） | 不支持 | 不支持 | — | [wan](models/wan_zh-cn.md) |

各架构的端到端文档（前向图、组件、参数、prefill/decode 优化）见[按模型架构卡片](models/README_zh-cn.md)。
