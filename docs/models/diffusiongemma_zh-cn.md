# DiffusionGemma

[← 返回模型索引](README_zh-cn.md)

## 状态快照

| 字段 | 状态 |
|---|---|
| GGUF 架构标识 | `diffusion-gemma`、`diffusion_gemma` |
| 模型类 | [`DiffusionGemmaModel`](../../TensorSharp.Models/Models/DiffusionGemma/DiffusionGemmaModel.cs) |
| 采样器 | [`DiffusionGemmaSampler`](../../TensorSharp.Models/Models/DiffusionGemma/DiffusionGemmaSampler.cs) |
| 模态 | 仅文本 |
| 思维链 / 工具调用 | 不支持 |
| 生成方式 | 分块文本扩散，不是自回归 token decode |
| CLI 支持 | `TensorSharp.Cli` 检测到 `DiffusionGemmaModel` 后进入 diffusion 运行模式 |
| 服务端支持 | Web UI chat stream 带实时去噪预览；Ollama/OpenAI 兼容路径仍是自回归 chat 路径 |
| 连续批处理 | 独立的 [`DiffusionBatchScheduler`](../../TensorSharp.Server/DiffusionBatchScheduler.cs)，在 block 边界接纳请求 |

## 1. 来源与目标

DiffusionGemma 是基于 Gemma-4 风格 Mixture-of-Experts backbone 的分块文本扩散语言模型。
它和自回归 `gemma4` 的运行时契约不同：

- `Forward(int[] tokens)` 会主动抛错；生成必须通过 `DiffusionGemmaSampler`。
- 每个去噪步骤在拼接后的 `[prompt | canvas]` 序列上运行。
- prompt 区域是因果的，并且不会看见 canvas。
- canvas 区域对 prompt + canvas 做双向注意力。
- 每个 block 输出当前确定性的 argmax canvas，并在多步去噪中逐步收敛。

GGUF 必须报告 `general.architecture=diffusion-gemma` 或 `diffusion_gemma`；
`ModelBase.Create()` 会把这两个 key 路由到 `DiffusionGemmaModel`。

## 2. 前向计算图

模型暴露两种执行路径。

统一正确性路径是 `ForwardCanvas(tokens, promptLen)`：

```text
[prompt tokens | canvas tokens]
  -> 区域感知 embedding scale
  -> prompt/canvas attention mask
  -> N 个 Gemma 风格 transformer 层
       - local/global QK-norm attention
       - dense gated-GELU MLP
       - top-k MoE experts
       - prompt encoder scale / canvas decoder scale
  -> output norm
  -> tied lm-head
  -> final logit softcap
  -> canvas logits
```

GPU 优化路径会把每个 block 拆成 prompt prefill 与多次 canvas decode：

1. `PrefillPrompt(promptTokens)` 只计算一次 prompt K/V。
2. `DecodeCanvas(canvasTokens, scBuffer, scUse, prevTempInv)` 在每个去噪步复用 prompt K/V。
3. 采样器接受低熵位置，重新噪声化其余位置，然后继续迭代。

Prompt-KV 缓存在 device-glue 后端启用；纯 CPU 路径会使用统一 forward。

## 3. 采样器契约

`DiffusionEbParams` 控制生成：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `MaxDenoisingSteps` | 48 | 每个 canvas block 最大去噪步数 |
| `TMin` / `TMax` | 0.4 / 0.8 | 从后期到前期的温度调度 |
| `EntropyBound` | 0.1 | 被接受位置的累计互信息上界 |
| `StabilityThreshold` | 1 | 早停前 argmax canvas 需要稳定的步数 |
| `ConfidenceThreshold` | 0.005 | 早停使用的平均熵阈值 |
| `Seed` | 0 | 确定性采样种子 |
| `MaxBlocks` | 1 | block-autoregressive canvas 数量 |

CLI 对应：

```bash
./TensorSharp.Cli --model diffusion-gemma.gguf --input prompt.txt --backend ggml_metal \
  --max-tokens 256 --diffusion-steps 48 --diffusion-seed 0 --diffusion-blocks 1
```

当 `--diffusion-blocks` 为 `0` 时，CLI 会根据 `--max-tokens` 与
`diffusion.canvas_length` 推导 block 数。

## 4. 架构细节

DiffusionGemma 复用了许多 Gemma-4 backbone 设计：

- NeoX RoPE，并区分 local/global 维度。
- 五个 local sliding-window 层加一个 global 层的循环模式。
- 每头 Q/K RMSNorm 与无权重 V RMSNorm。
- Global 层可以没有 `attn_v.weight`，此时 V 使用原始 K 投影。
- Dense gated-GELU MLP 加 128-expert top-8 MoE。
- 共享 embedding / lm-head，并带 final logit softcapping。

Diffusion 专属元数据：

| Key | 含义 |
|---|---|
| `diffusion.canvas_length` | 每个 block 去噪的 canvas 位置数，默认 256 |
| `tokenizer.ggml.mask_token_id` | warmup 与回退路径使用的 mask token id |
| `<arch>.attention.sliding_window_pattern` | local/global 层模式 |
| `<arch>.attention.head_count_kv` | 每层 KV head 数 |
| `<arch>.expert_count` / `<arch>.expert_used_count` | MoE expert 总数与 active top-k |

## 5. 加速状态

当前优化路径包括：

- GPU 后端的 prompt-KV 缓存。
- 默认启用 self-conditioning；可用 `DIFFUSION_NO_SC=1` 关闭。
- GGML 融合 decode layer、融合整模型 decode、融合 lm-head tail。
- 针对 DiffusionGemma 多行 canvas 工作负载的 MLX K-quant affine repack。
- `TensorSharp.Server` 中通过 `DiffusionBatchScheduler` 做 block 边界连续批处理。

重要开关：

| 变量 | 作用 |
|---|---|
| `DIFFUSION_STEPS` | 服务端每个 block 的去噪步数，默认 48 |
| `DIFFUSION_MAX_BATCH` | 服务端 diffusion scheduler 最大活跃请求数，默认 2 |
| `DIFFUSION_NO_PKV=1` | 关闭 device-glue 后端上的 prompt-KV 缓存 |
| `DIFFUSION_NO_SC=1` | 关闭 self-conditioning |
| `DIFFUSION_SC_TOPK` | 实验用 self-conditioning top-K 截断，默认 32 |
| `DIFFUSION_BATCHED_FORWARD=1` | 使用真正的批处理 canvas decode，而不是按时间片执行融合单 canvas decode |
| `DIFFUSION_NO_FUSED_DECODE=1` | 关闭 GGML 融合整模型 diffusion decode |
| `DIFFUSION_NO_FUSED_LMHEAD_TAIL=1` | 关闭融合 output-norm + lm-head + softcap tail |
| `DIFFUSION_LMHEAD_BATCH_CAP_MB` | 临时批处理 lm-head logits 的内存上限，默认 300 MB |
| `DIFFUSION_PROFILE=1` / `DIFFUSION_STEPTIME=1` / `DIFFUSION_FUSED_DEBUG=1` | 开发用计时与融合 kernel 调试诊断 |

## 6. 服务端行为

当 Web UI 承载 DiffusionGemma GGUF 时：

- `/api/chat` 会进入 diffusion 路径。
- 流式输出发送 `replace` 事件而不是 token append，因为每一步都会重新修正整个 canvas。
- 在 `done` 事件前会先发送最终定稿 replacement。
- 并发请求共享一个后台 diffusion scheduler，并在 block 之间被接纳。

Ollama 与 OpenAI 兼容适配器仍通过 `ChatStreamWithMetricsAsync` 使用 append-oriented
响应形状。它们可以返回 DiffusionGemma 的最终文本，但实时去噪预览与 `replace`
帧只在 Web UI 中提供。

## 7. 测试覆盖

[`DiffusionGemmaTests`](../../InferenceWeb.Tests/DiffusionGemmaTests.cs) 通过
`TS_TEST_MODEL_DIR` 指向真实 GGUF 后按需运行，覆盖：

- `ForwardCanvas` 有限 logits 正确性。
- 端到端 EntropyBound 生成。
- Prompt-KV 等价性与速度探针。
- 重复 token 输出与 device memory 留存的回归保护。
- 与服务端调度风格一致的批处理 decode 等价性和双请求生成。

## 8. 后续工作

- 等 Ollama/OpenAI 适配层有 diffusion-aware 兼容面后，再补充专门 API 示例。
- 只有在目标 GPU 上胜过当前路径时，才把真正批处理 canvas decode 提升为默认；目前单个 canvas 已经能吃满 GPU 时，融合单 canvas 路径可能更快。
- 如运维需要更细粒度可见性，可把更多 diffusion scheduler 指标接入 `/api/queue/status`。
