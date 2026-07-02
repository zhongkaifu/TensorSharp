# 模型架构卡片

[English](model_cards.md) | [中文](model_cards_zh-cn.md)

> 本文件已被拆分。每个模型现在都在 [`docs/models/`](models/README_zh-cn.md)
> 下有独立卡片，每张英文卡片旁边都有对应的中文版本。

通过下表可以直接跳到具体架构，或阅读
[`docs/models/README_zh-cn.md`](models/README_zh-cn.md) 了解完整的实现矩阵
与特性对比。

| 架构 | 中文卡片 | English card |
|---|---|---|
| Gemma 3 | [models/gemma3_zh-cn.md](models/gemma3_zh-cn.md) | [models/gemma3.md](models/gemma3.md) |
| Gemma 4 | [models/gemma4_zh-cn.md](models/gemma4_zh-cn.md) | [models/gemma4.md](models/gemma4.md) |
| DiffusionGemma | [models/diffusiongemma_zh-cn.md](models/diffusiongemma_zh-cn.md) | [models/diffusiongemma.md](models/diffusiongemma.md) |
| Qwen 3 | [models/qwen3_zh-cn.md](models/qwen3_zh-cn.md) | [models/qwen3.md](models/qwen3.md) |
| Qwen 3.5 / 3.6 family | [models/qwen35_zh-cn.md](models/qwen35_zh-cn.md) | [models/qwen35.md](models/qwen35.md) |
| GPT OSS | [models/gptoss_zh-cn.md](models/gptoss_zh-cn.md) | [models/gptoss.md](models/gptoss.md) |
| Nemotron-H | [models/nemotron_zh-cn.md](models/nemotron_zh-cn.md) | [models/nemotron.md](models/nemotron.md) |
| Mistral 3 | [models/mistral3_zh-cn.md](models/mistral3_zh-cn.md) | [models/mistral3.md](models/mistral3.md) |
| Qwen-Image-Edit | [models/qwenimage_zh-cn.md](models/qwenimage_zh-cn.md) | [models/qwenimage.md](models/qwenimage.md) |

每张卡片会把工程师或研究员从“从未听说过这个模型”带到“可以解释它的前向计算图，
并能在 TensorSharp 中复现推理路径”，统一覆盖：

1. 来源与目标（提供方、GGUF 架构标识、模态、思维链、工具调用）
2. 模型架构（顶层模块图、层数、每层异构性）
3. 前向计算图（per-token decode 与多 token prefill 中算子的精确顺序）
4. 组件细节（attention、FFN/SSM、routing、normalization、RoPE 变体、视觉/音频编码器）
5. 参数与配置（GGUF 元数据 key、权重张量命名、dtype 要求）
6. TensorSharp 实现走读
7. Prefill 优化
8. Decode 优化
9. 内存与 KV cache 策略
10. 多模态管线（如适用）
11. 输出解析器与聊天模板
12. 优化机会

新增架构时，请按
[`docs/models/README_zh-cn.md`](models/README_zh-cn.md#新增模型架构) 中的清单操作。
