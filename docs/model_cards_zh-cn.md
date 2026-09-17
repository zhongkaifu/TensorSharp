# 模型架构卡片

[English](model_cards.md) | [中文](model_cards_zh-cn.md)

> 本文件已被拆分。每个模型现在都在 [`docs/models/`](models/README_zh-cn.md)
> 下有独立卡片，每张英文卡片旁边都有对应的中文版本。

通过下表可以直接跳到具体架构，或阅读
[`docs/models/README_zh-cn.md`](models/README_zh-cn.md) 了解完整的实现矩阵
与特性对比。

每张当前卡片开头都列有已核对的 Hugging Face 下载入口、精确示例文件名，以及可复制的
`TensorSharp.Cli` / `TensorSharp.Server` 命令。已验证的快速上手路径（见仓库
[README](../README_zh-cn.md#快速开始)）是 TensorSharp 的 Gemma 4 E4B Q8_0
原生 GGML 家族 / 路径层级：使用推荐公开
[ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)
中的 `gemma-4-E4B-it-Q8_0.gguf`，后端选择 `ggml_cuda`、`ggml_metal` 或
`ggml_vulkan`；详见 [Gemma 4 卡片](models/gemma4_zh-cn.md#已验证的-gemma-4-e4b-原生-ggml-快速路径)。
纯文本不需要 `mmproj`；图像、视频或音频输入需要匹配的 `mmproj`。这里不声称基准输入
对应某个公开文件的特定校验和。

| 架构 | GGUF 架构标识 | 功能 | 中文卡片 | English card |
|---|---|---|---|---|
| DeepSeek V4 Flash | `deepseek4` | 带压缩注意力的稀疏 MoE 文本模型；通过独立 `--draft-model` GGUF 支持 DSpark 块级投机解码 | [models/deepseek4_zh-cn.md](models/deepseek4_zh-cn.md) | [models/deepseek4.md](models/deepseek4.md) |
| Gemma 4 | `gemma4` | 稠密与 MoE 的文本 + 图像 + 视频 + 音频对话，支持思维链与工具调用；通过独立 `gemma4-assistant` 草稿 GGUF 支持 MTP 投机解码 | [models/gemma4_zh-cn.md](models/gemma4_zh-cn.md) | [models/gemma4.md](models/gemma4.md) |
| DiffusionGemma | `diffusion-gemma`、`diffusion_gemma` | 文本**扩散**生成——用 EntropyBound 去噪采样器取代自回归 decode | [models/diffusiongemma_zh-cn.md](models/diffusiongemma_zh-cn.md) | [models/diffusiongemma.md](models/diffusiongemma.md) |
| Qwen 3.5 / 3.6 family | `qwen35`、`qwen35moe`、`qwen3next` | 全注意力 + GatedDeltaNet 混合的文本 + 图像对话，稠密或 MoE；Qwen 3.6 内嵌 NextN 草稿块用于投机解码 | [models/qwen35_zh-cn.md](models/qwen35_zh-cn.md) | [models/qwen35.md](models/qwen35.md) |
| Bonsai Q1_0 | `qwen3`（8B）、`qwen35`（27B） | 两个哈希钉住、仅支持本地旁加载的 Q1_0 文件：名字相同但架构不同——一个是稠密 Qwen 3 解码器，一个是稠密 Qwen 3.5 混合架构。仅文本 | [models/bonsai_zh-cn.md](models/bonsai_zh-cn.md) | [models/bonsai.md](models/bonsai.md) |
| GPT OSS | `gptoss`、`gpt-oss` | 带 attention sinks 的 MXFP4 MoE 文本模型，使用 Harmony 思维链 / 工具格式 | [models/gptoss_zh-cn.md](models/gptoss_zh-cn.md) | [models/gptoss.md](models/gptoss.md) |
| Nemotron-H | `nemotron_h`、`nemotron_h_moe` | Mamba2 SSM + 注意力 +（MoE）FFN 混合文本模型；Omni 版本增加图像输入。投机解码（包括 Nemotron 3.5 Lightning 的 DSpark 草稿器）被拒绝，因为主干的 verify 与 decode 内核结果不一致 | [models/nemotron_zh-cn.md](models/nemotron_zh-cn.md) | [models/nemotron.md](models/nemotron.md) |
| Mistral 3 | `mistral3` | 稠密文本 + 图像对话，YaRN 校正 RoPE 与 Pixtral 视觉编码器 | [models/mistral3_zh-cn.md](models/mistral3_zh-cn.md) | [models/mistral3.md](models/mistral3.md) |
| Muse-Glimmer | `muse-glimmer`、`muse_glimmer` | 交错滑动窗口的文本 + 图像对话，支持思维链与 ATEM 工具调用；通过独立 `--draft-model` GGUF 支持 DFlash 块级投机解码 | [models/muse-glimmer_zh-cn.md](models/muse-glimmer_zh-cn.md) | [models/muse-glimmer.md](models/muse-glimmer.md) |
| Qwen-Image-Edit | `qwen_image`、`qwen-image` | **图像编辑**——提示词 + 输入图像 → 编辑后的图像，走 60 块 MMDiT 扩散循环；Lightning LoRA 可把 60 次 DiT 前向降到 4–8 次 | [models/qwenimage_zh-cn.md](models/qwenimage_zh-cn.md) | [models/qwenimage.md](models/qwenimage.md) |
| MiniMax-H3 | `minimax-h3`、`minimax_h3` | **音视频联合生成**——提示词（可选关键帧或参考图）→ 视频 **+ 原生 32 kHz 立体声音频**，由同一个扩散 Transformer 在一份打包潜变量里一起生成；支持文生视频、图生视频（照片作为首帧）、首尾帧、参考生视频，全部在 CFG-free 的 4–8 步下运行 | [models/minimax-h3_zh-cn.md](models/minimax-h3_zh-cn.md) | [models/minimax-h3.md](models/minimax-h3.md) |
| Wan 视频 | `wan`、`wan2.1`、`wan2.2` | **视频生成，仅视频**——提示词（可选首帧图）→ H.264 MP4，涵盖 Wan 2.1 T2V 与 Wan 2.2 TI2V-5B / A14B；换用步数蒸馏检查点可把 100 次 DiT 前向的官方配方降到 4 次 | [models/wan_zh-cn.md](models/wan_zh-cn.md) | [models/wan.md](models/wan.md) |

每张卡片会把工程师或研究员从“从未听说过这个模型”带到“可以解释它的前向计算图，
并能在 TensorSharp 中复现推理路径”，统一覆盖：

1. 已核对的下载入口与可运行 CLI / Server 命令
2. 来源与目标（提供方、GGUF 架构标识、模态、思维链、工具调用）
3. 模型架构（顶层模块图、层数、每层异构性）
4. 前向计算图（per-token decode 与多 token prefill 中算子的精确顺序）
5. 组件细节（attention、FFN/SSM、routing、normalization、RoPE 变体、视觉/音频编码器）
6. 参数与配置（GGUF 元数据 key、权重张量命名、dtype 要求）
7. TensorSharp 实现走读
8. Prefill 优化
9. Decode 优化
10. 内存与 KV cache 策略
11. 多模态管线（如适用）
12. 输出解析器与聊天模板
13. 优化机会

新增架构时，请按
[`docs/models/README_zh-cn.md`](models/README_zh-cn.md#新增模型架构) 中的清单操作。
