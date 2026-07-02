# Model Architecture Cards

[English](model_cards.md) | [中文](model_cards_zh-cn.md)

> This file has been split. Each model now has a dedicated card under
> [`docs/models/`](models/README.md), and there is a per-model 中文 version
> alongside each English card.

Use the index below to jump straight into a specific architecture, or read
[`docs/models/README.md`](models/README.md) for the full implementation
matrix and feature comparison.

| Architecture | English card | 中文卡片 |
|---|---|---|
| Gemma 3 | [models/gemma3.md](models/gemma3.md) | [models/gemma3_zh-cn.md](models/gemma3_zh-cn.md) |
| Gemma 4 | [models/gemma4.md](models/gemma4.md) | [models/gemma4_zh-cn.md](models/gemma4_zh-cn.md) |
| DiffusionGemma | [models/diffusiongemma.md](models/diffusiongemma.md) | [models/diffusiongemma_zh-cn.md](models/diffusiongemma_zh-cn.md) |
| Qwen 3 | [models/qwen3.md](models/qwen3.md) | [models/qwen3_zh-cn.md](models/qwen3_zh-cn.md) |
| Qwen 3.5 / 3.6 family | [models/qwen35.md](models/qwen35.md) | [models/qwen35_zh-cn.md](models/qwen35_zh-cn.md) |
| GPT OSS | [models/gptoss.md](models/gptoss.md) | [models/gptoss_zh-cn.md](models/gptoss_zh-cn.md) |
| Nemotron-H | [models/nemotron.md](models/nemotron.md) | [models/nemotron_zh-cn.md](models/nemotron_zh-cn.md) |
| Mistral 3 | [models/mistral3.md](models/mistral3.md) | [models/mistral3_zh-cn.md](models/mistral3_zh-cn.md) |
| Qwen-Image-Edit | [models/qwenimage.md](models/qwenimage.md) | [models/qwenimage_zh-cn.md](models/qwenimage_zh-cn.md) |

Each card walks an engineer or researcher from "I have never heard of this
model" to "I can explain the forward graph and reproduce the inference path
in TensorSharp", covering:

1. Origin and intent (provider, GGUF arch keys, modalities, thinking, tools)
2. Model architecture (high-level block diagram, layer counts, per-layer heterogeneity)
3. Forward graph (the exact ordered list of ops, per-token decode and multi-token prefill)
4. Components in detail (attention, FFN/SSM, routing, normalization, RoPE flavor, vision/audio encoder)
5. Parameters and settings (GGUF metadata keys, weight tensor naming, dtype expectations)
6. TensorSharp implementation walkthrough
7. Prefill optimization
8. Decode optimization
9. Memory and KV cache strategy
10. Multimodal pipeline (when applicable)
11. Output parser and chat template
12. Optimization opportunities

When adding a new architecture, follow the checklist in
[`docs/models/README.md`](models/README.md#adding-a-new-architecture).
