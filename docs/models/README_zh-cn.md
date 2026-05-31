# 模型架构卡片

[English](README.md) | [中文](README_zh-cn.md)

本目录是 TensorSharp 支持的每一种模型架构的权威分卡参考。每张卡片都是一份独立的简报：从「我从未听说过这个模型」走到「我能解释它的前向计算图，并能在 TensorSharp 中复现它的推理路径」。如果你只需要顶层指引，请使用下方的总览表；否则请直接阅读对应的卡片。

## 每张卡片的内容

为了便于横向对比，所有卡片都采用相同的章节结构：

1. **来源与目标** —— 模型作者、GGUF 架构标识，以及它支持的能力（多模态、思维链、工具调用）。
2. **模型架构** —— 顶层模块图、层数与每层异构性。
3. **前向计算图** —— 单 token（decode）和多 token（prefill）依次经过的算子序列，包含 residual 与各处 normalization。
4. **组件细节** —— 每一个子模块（attention、FFN/SSM、routing、normalization、RoPE 变体、视觉/音频编码器）的数学描述。
5. **参数与配置** —— GGUF 元数据 key、权重张量命名约定、dtype 要求。
6. **TensorSharp 实现** —— C# 源码定位、初始化顺序、缓存布局、模型如何接入 `ModelBase` / `Ops` / 原生 GGML kernel。
7. **Prefill 优化** —— 分块、融合 per-layer kernel、并行化、跨层缓存。
8. **Decode 优化** —— 融合单调用 kernel、预解析权重指针、批量 MoE、in-place kernel、缓存复用。
9. **内存与 KV cache 策略** —— 环形缓存 vs 线性缓存、mmap 权重、预分配 decode 缓冲。
10. **多模态管线** —— 图像 / 音频 / 视频如何被处理、编码并注入到语言模型。
11. **输出 / 聊天模板** —— 协议解析器、停止 token、思维链 / 工具调用格式。
12. **优化机会** —— 已知尚未实现但能进一步释放性能或能力的工作。

## 实现矩阵

| 架构 | 卡片 | 模型类 | GGUF keys | 模态 | 思维链 | 工具调用 | 批处理 / 分页前向 | 主要加速路径 |
|---|---|---|---|---|---|---|---|---|
| Gemma 3 | [gemma3.md](gemma3.md) | `Gemma3Model` | `gemma3` | 文本、图像 | 否 | 否 | 否（仅旧单序列路径） | SWA / 全局注意力交替、GeGLU FFN、QK-norm、V-norm |
| Gemma 4 | [gemma4.md](gemma4.md) | `Gemma4Model` | `gemma4` | 文本、图像、视频、音频 | 是 | 是 | **默认启用**（可用 `TS_GEMMA4_BATCHED=0` 关闭） | 整模型融合 decode（一次 GGML 调度）、融合 per-layer prefill、分块 prefill、SWA 环形缓存、PLE、KV 共享、MoE 变体。批处理路径与旧路径 logits 在 FP 噪声内一致（`Gemma4BatchedForwardTests`）；batch=8 短 prompt 达 ~1.5×，4×800-token prompt 达 ~1.6×。 |
| Qwen 3 | [qwen3.md](qwen3.md) | `Qwen3Model` | `qwen3` | 文本 | 是 | 是 | 参考实现（`Qwen3Model.BatchedForward.cs`）—— 当提供基础 Qwen3 GGUF 时由 `Qwen3BatchedForwardTests` 验证 | 整模型原生 decode，权重指针在加载时预解析 |
| Qwen 3.5 / 3.6 family | [qwen35.md](qwen35.md) | `Qwen35Model` | `qwen35`、`qwen35moe`、`qwen3next` | 文本、图像 | 是 | 是 | **默认启用**（`TS_QWEN35_BATCHED=0` 或 `--no-continuous-batching` 可关闭）。带每槽位的递归状态池，可选原生 GatedDeltaNet 内核（`TS_QWEN35_BATCHED_GDN_NATIVE=1`）。 | 全注意力 + GatedDeltaNet 递归混合、融合 attention 层 decode、融合 prefill attention、融合输出投影 + FFN、融合输出投影 + norm + router、批量 MoE（routed + shared + residual 一次完成）、融合视觉编码器 |
| GPT OSS | [gptoss.md](gptoss.md) | `GptOssModel` | `gptoss`、`gpt-oss` | 文本 | 是（始终启用） | 是 | **默认启用**（`TS_GPTOSS_BATCHED=0` 可关闭）。通过 `TSGgml_PagedAttentionForwardWithSinks` 处理每头 attention sinks（或 `TS_GPTOSS_PAGED_ATTN_MANAGED=1` 使用 C# fallback）。在 `GptOssBatchedCorrectnessTests` 中与旧路径 100% 贪心一致。 | Stacked MoE prefill kernel（mul_mat_id + add_id + swiglu_oai）、attention sinks、MXFP4 专家权重 |
| Nemotron-H | [nemotron.md](nemotron.md) | `NemotronModel` | `nemotron_h`、`nemotron_h_moe` | 文本、图像（Omni 版） | 是 | 是 | **默认启用**（`TS_NEMOTRON_BATCHED=0` 可关闭）。带每槽位 Mamba2 conv + SSM 状态池，可选原生批处理 Mamba2 步（`TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1`）。与旧路径 100% 贪心一致；Apple M4 Pro 上 batch=3 最高可达 3.95× tps。 | Mamba2 + 注意力 + MoE FFN 混合堆栈、批量 GPU MoE、可选 Parakeet 音频前端、RADIO/v2_vl 图像编码器 |
| Mistral 3 | [mistral3.md](mistral3.md) | `Mistral3Model` | `mistral3` | 文本、图像 | 否 | 否 | **默认启用** —— `IBatchedPagedModel` 的参考实现。在 Ministral-3-14B 上完成端到端验证；原生分页注意力内核在长上下文下比旧的单序列路径快 ~21%。 | YaRN 校正 RoPE 与位置相关 Q 缩放、融合 QKV / gate_up、Pixtral 视觉编码器 |

## 后端说明

模型代码尽量保持后端无关。`ModelBase` 通过 `BackendType` 与执行计划选择张量存储，再把算子分发给已注册的后端实现：

| 后端类型 | 包 | 说明 |
|---|---|---|
| `Cpu` | `TensorSharp.Core` | 纯托管张量，附带 SIMD / 托管量化快路径（RMSNorm、RoPE、softmax、融合激活、GEMM、dequant）。 |
| `Cuda` | `TensorSharp.Backends.Cuda` | Direct CUDA Driver-API 分配器与存储、cuBLAS GEMM、热点算子的 PTX 内核（RMSNorm、softmax、RoPE/RoPEEx、SDPA、GQA prefill/decode、causal mask、gather/concat、融合激活）、受支持量化类型的原生 quant matmul / get_rows，未实现的算子回退到 CPU。 |
| `Mlx` | `TensorSharp.Backends.MLX` | Apple Silicon `mlx-c` 桥接，含量化 / 融合 / 编译内核、异步 worker 派发、MoE 专家 offload，以及 CPU 回退层。依赖 `libmlxc`。 |
| `GgmlCpu` / `GgmlMetal` / `GgmlCuda` | `TensorSharp.Backends.GGML` + `TensorSharp.GGML.Native` | 原生 ggml 桥接，包括量化计算图调度与平台后端；mmap 量化权重通过 host 指针缓冲零拷贝绑定。还包含驱动批处理 / 分页执行路径的分页注意力内核（`TSGgml_PagedAttentionForward`，含 GPT OSS sinks 变体）。 |

凡是卡片中提到融合 GGML kernel（例如 `Qwen35AttentionLayerDecode`、`Gemma4LayerPrefill`、`MoEExpertsSwiGLUResidual`），其源码都在 `TensorSharp.GGML.Native/ggml_ops_*.cpp`，并通过 `TensorSharp.Backends.GGML/GgmlBasicOps.cs` 暴露给托管侧。如果某个融合路径只在 GGML CPU / Metal / CUDA 上启用而在纯托管 CPU 或 direct CUDA 上没有启用，请到原生桥侧查看。

## 连续批处理 & 分页 KV 缓存

上表所列的全部架构都会经过共享的 `InferenceEngine` + `ContinuousBatchScheduler` + `BatchExecutor` 栈，详情见 [`docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md`](../PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md)。实现了 `IBatchedPagedModel.ForwardBatch` 的模型会在每个调度步骤中执行一次批处理前向（使用基于 `slotMapping` 的 K/V 写入与共享分页缓冲，并通过原生分页内核做按序列注意力）；其余模型则在同一引擎内沿用按序列 KV 交换。各模型的启用方式见上方实现矩阵以及项目根 README。

## 架构对比

| 特性 | Gemma 3 | Gemma 4 | Qwen 3 | Qwen 3.5 / 3.6 family | GPT OSS | Nemotron-H | Mistral 3 |
|---|---|---|---|---|---|---|---|
| 层类型 | 密集 | 密集 / MoE | 密集 | 混合（注意力 + 递归）± MoE | MoE | 混合（Mamba2 + 注意力 + MoE FFN） | 密集 |
| 注意力 | SWA + 全局 | SWA + 全局 | 全 GQA | 全 GQA + Sigmoid Gate | 全 + Sinks | 全 GQA（无 RoPE） | 全 GQA |
| FFN 激活 | GeGLU | GeGLU | SwiGLU | SwiGLU | SiLUAlphaLimit（带 clamp 的 GLU） | ReLU² | SwiGLU |
| RoPE 类型 | NeoX（双 base） | NeoX + 比例 / 部分 | NeoX | NeoX / MRoPE | NeoX + YaRN | 无 | GPT-J + YaRN |
| QK-norm | 是 | 是 | 是 | 是 | 否 | 否 | 否 |
| V-norm | 否 | 是（无权重） | 否 | 否 | 否 | 否 | 否 |
| 投影偏置 | 无 | 无 | 无 | 无 | 全部都有 | 无 | 无 |
| 每层缩放 | 否 | 是 | 否 | 否 | 否 | 否 | 否 |
| Per-Layer Embedding (PLE) | 否 | 是 | 否 | 否 | 否 | 否 | 否 |
| KV 共享 | 否 | 是（尾部若干层） | 否 | 否 | 否 | 否 | 否 |
| Attention sinks | 否 | 否 | 否 | 否 | 是 | 否 | 否 |
| 环形 KV cache | 否 | 是（SWA 层） | 否 | 否 | 否 | 否 | 否 |
| SSM / 递归层 | 否 | 否 | 否 | 是（GatedDeltaNet） | 否 | 是（Mamba2） | 否 |
| 共享专家 | 否 | 否 | 否 | 是（qwen35moe / qwen3next） | 否 | 是（可选） | 否 |
| Latent bottleneck FFN | 否 | 否 | 否 | 否 | 否 | 是（可选） | 否 |
| 位置相关 Q 缩放 | 否 | 否 | 否 | 否 | 否 | 否 | 是（与 YaRN 配合） |
| 视觉 | 是 | 是 | 否 | 是 | 否 | 是（Omni） | 是（Pixtral） |
| 音频 | 否 | 是 | 否 | 否 | 否 | 是（Parakeet，需 mmproj） | 否 |
| 视频 | 否 | 是 | 否 | 否 | 否 | 否 | 否 |
| 思维链 | 否 | 是 | 是 | 是 | 是（始终启用） | 是 | 否 |
| 工具调用 | 否 | 是 | 是 | 是 | 是 | 是 | 否 |
| 融合 QKV | 否 | 是 | 是 | 混合（attention 层拆开，递归层融合 5 路） | 是 | 是 | 是 |
| 融合单调用 decode | 否 | 是（Gemma4ModelDecode） | 是（TransformerModelDecode，原生循环） | per-layer 融合（Qwen35AttentionLayerDecode、FusedOutProjFFN、FusedOutProjNormRouter） | per-layer | per-layer / 批量 MoE | 否 |
| 融合单调用 prefill | 否 | 是（Gemma4LayerPrefill，密集层） | 否 | 是（FusedPrefillAttention、FusedOutProjFFN、MoE prefill） | 是（MoE prefill via mul_mat_id） | 否 | 否 |
| 批量 GPU MoE | n/a | 待实现 | n/a | 是（routed + shared + residual 融合） | 是（stacked weight slabs） | 是 | n/a |
| 融合视觉编码器 | n/a | 标准 | n/a | 是（FusedVisionAttention + FusedVisionMLP） | n/a | 标准（RADIO ViT） | 标准（Pixtral） |
| 输出解析器 | `PassthroughOutputParser` | `Gemma4OutputParser` | `Qwen3OutputParser` | `Qwen35OutputParser` | `HarmonyOutputParser`（始终启用） | `Qwen3OutputParser` | `PassthroughOutputParser` |

## 新增模型架构

要在 TensorSharp 中加入新的模型架构：

1. 在 `TensorSharp.Models/Models/<Name>/<Name>Model.cs` 创建类并继承 `ModelBase`。
2. 构造函数中：通过 `_gguf.GetXxx()` 读取 GGUF 元数据，调用 `ParseBaseConfig()` 与 `ParseTokenizer()`，调用 `LoadWeights()`，融合权重，然后初始化缓存。
3. 实现 `Forward(int[] tokens) → float[]`：embedding → 可选多模态注入 → transformer 层 → final norm → LM head → logits 拷贝。
4. 实现 `ResetKVCache()` 与 `Dispose()`；如支持 KV 缓存复用则实现 `TruncateKVCache()`。
5. 在 `TensorSharp.Models/ModelBase.cs` 的 `ModelBase.Create()` switch 中注册。
6. 如果模型有非标准的输出格式，在 `TensorSharp.Runtime/OutputParser.cs` 中实现 `IOutputParser`，并在 `OutputParserFactory.Create()` 注册。
7. 如果模型使用了新的聊天模板，在 `TensorSharp.Runtime/ChatTemplate.cs` / `Jinja2Template.cs` 中加入支持。
8. 在 `docs/models/<name>.md`（与 `<name>_zh-cn.md` 如果你打算双语覆盖）下新增卡片，更新本 README 的实现矩阵，并从项目根 README 链入卡片。
9. 如果模型涉及新的模态、思维链或工具能力，更新 `TensorSharp.Server/testdata/` 的能力门控。
