# TensorSharp 中的分页注意力与连续批处理

[English](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md) | [中文](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)

本文是 TensorSharp 当前 vLLM 风格分页 KV cache、基于块哈希的前缀共享、
以及迭代级连续批处理的实现参考。服务端默认通过这套引擎执行推理；旧的
单请求 FIFO 队列对象只作为队列状态 / 事件形状的 no-op 兼容 shim 保留。

## 当前状态

| 范围 | 状态 |
|---|---|
| 服务端引擎 | `TensorSharp.Server` 为当前加载模型持有一个 `InferenceEngineHost`。`ChatGenerationPipeline` 将渲染后的 prompt 提交给引擎，并从 `InferenceRequestHandle` 流式读取 token。 |
| 调度器 | `ContinuousBatchScheduler` 负责接纳等待请求、在块压力下抢占运行中的序列、应用每步 token 预算，并按内容哈希共享完整前缀块。 |
| KV 存储 | `BlockPool`、`BlockTable`、`PagedKvStorage`、`BlockHashIndex` 持有固定大小物理块，包含引用计数、LRU 空闲顺序与内容寻址查找。 |
| 批处理执行 | 实现 `IBatchedPagedModel.ForwardBatch` 的模型会把本轮所有序列打包到一次模型调用中，显式传入 `positions`、`slotMapping`、`queryStartLoc` 与每序列 block table。 |
| 回退执行 | 模型或某些功能组合无法批处理时会抛出 `NotSupportedException`；`BatchExecutor` 会在同一引擎内回退到隔离的按序列 KV-swap 路径。 |
| 原生注意力 | `TSGgml_PagedAttentionForward` 在 C++ 中聚合分页 K/V 并派发 `ggml_flash_attn_ext`；GPT OSS 使用 `TSGgml_PagedAttentionForwardWithSinks`。 |
| 队列 API | `InferenceQueue` 是 no-op 兼容层。`/api/queue/status` 与队列位置事件形状保留给依赖这些字段的客户端，不再承担请求串行化。 |
| 扩散模型 | DiffusionGemma 不进入这套自回归 `ForwardBatch` 契约。CLI 生成使用 `DiffusionGemmaSampler`；Web UI 使用 `DiffusionBatchScheduler` 在 block 边界批处理去噪工作。 |

## 分层架构

```text
Adapters (Web UI / Ollama / OpenAI)
        |
        v
ChatGenerationPipeline
  - 渲染 prompt
  - 准备多模态 embedding
  - 提交 SequenceState
  - 流式返回 InferenceRequestHandle token
        |
        v
InferenceEngine
  - worker thread
  - submit / abort API
  - completion future
        |
        +--> ContinuousBatchScheduler
        |      - waiting / running 集合
        |      - token 与序列预算
        |      - 块分配 / 抢占
        |      - 前缀块采纳
        |
        +--> BatchExecutor
               - 可用时调用 ForwardBatch
               - 否则按序列交换 KV 块
               - 采样 decode token
               - 捕获新写满的 KV 块
        |
        v
BlockPool + PagedKvStorage + BlockHashIndex
```

### 核心组件

| 组件 | 文件 | 作用 |
|---|---|---|
| `KvBlock` | `TensorSharp.Runtime/Paged/KvBlock.cs` | 物理块元数据、引用计数、哈希信息。 |
| `BlockPool` | `TensorSharp.Runtime/Paged/BlockPool.cs` | 分配、释放、引用计数与淘汰物理块。 |
| `BlockTable` | `TensorSharp.Runtime/Paged/BlockTable.cs` | 将每个序列的逻辑块映射到物理块。 |
| `PagedKvStorage` | `TensorSharp.Runtime/Paged/PagedKvStorage.cs` | 按物理 block id 索引的字节 slab。 |
| `BlockHashIndex` | `TensorSharp.Runtime/Paged/BlockHashIndex.cs` | 用内容哈希查找可复用前缀块。 |
| `PagedKvBatchOps` | `TensorSharp.Runtime/Paged/PagedKvBatchOps.cs` | 批处理 K/V scatter 与每序列最后 token gather。 |
| `ManagedPagedAttention` | `TensorSharp.Runtime/Paged/ManagedPagedAttention.cs` | 纯 C# 分页注意力正确性回退。 |
| `TensorPagedAttention` | `TensorSharp.Models/Paged/TensorPagedAttention.cs` | 基于 Tensor 算子的分页注意力回退。 |
| `SequenceState` | `TensorSharp.Runtime/Scheduling/SequenceState.cs` | 每请求的状态、token、块、logits 与采样信息。 |
| `ContinuousBatchScheduler` | `TensorSharp.Runtime/Scheduling/ContinuousBatchScheduler.cs` | 带前缀缓存与抢占的迭代级调度器。 |
| `BatchExecutor` | `TensorSharp.Runtime/Scheduling/BatchExecutor.cs` | 执行调度结果、采样并捕获 KV 块。 |
| `InferenceEngine` | `TensorSharp.Runtime/Scheduling/InferenceEngine.cs` | worker loop 与公开 submit/abort 接口。 |
| `InferenceEngineHost` | `TensorSharp.Server/InferenceEngineHost.cs` | 服务端按模型注册的引擎单例。 |

## 请求流程

1. 协议适配器构造归一化的聊天请求。
2. `ChatGenerationPipeline` 渲染 prompt，解析采样参数，并准备图像 / 音频 / 视频 embedding。
3. Pipeline 创建 `SequenceState` 并调用 `InferenceEngine.SubmitRequest`。
4. 引擎 worker 向 `ContinuousBatchScheduler` 请求下一步工作。
5. 调度器在 token 与序列预算允许时接纳等待序列。分配新块前，它会在 `BlockHashIndex` 中查找完整 prompt 块，命中时直接复用共享块。
6. 块池压力较大时，调度器可以抢占优先级较低的运行序列，提交其完整块、释放剩余块，并重新排入等待队列。
7. `BatchExecutor` 执行本步工作。模型与功能组合支持时使用 `ForwardBatch`，否则使用按序列 KV-swap 回退。
8. 引擎把采样 token 发给 request handle，检查 EOS / max-tokens / abort 状态，并释放已完成序列的块。

前缀采纳会保留至少一个 prompt token 重新送入模型。这样即使可见前缀已经全部
命中块哈希缓存，也能产生新的 logits 用于采样。

## Batched Forward 契约

`IBatchedPagedModel.ForwardBatch(BatchedForwardContext ctx)` 接收一个紧凑的批描述：

| 字段 | 含义 |
|---|---|
| `Sequences` | 按输出顺序排列的调度序列。 |
| `InputTokens` | 本步所有序列拼接后的 prefill 或 decode token。 |
| `Positions` | 每个 token 的绝对位置。 |
| `QueryStartLoc` | `InputTokens` 的前缀和偏移，长度为 `numSeqs + 1`。 |
| `SlotMapping` | 每个 token 的分页写入槽：`blockId * blockSize + offset`。 |
| `BlockTables` | 每个序列在分页注意力中使用的物理 block table。 |

模型在拼接 token 轴上批量执行 embedding、projection、norm、FFN/MoE 与最终
logits；使用 `SlotMapping` 将新的 K/V 写入分页缓冲；注意力阶段再按每序列
block table 读取 K/V。返回值是每个序列一份 logits，顺序与 `ctx.Sequences` 一致。

## 执行路径

### 批处理路径

批处理路径是多请求推理的快路径。它避免了 K/V 所有权交换，并在所有调度 token
上摊薄线性投影开销。当前大多数批处理移植在 GGML 后端使用原生分页注意力：

| 内核 | 范围 | 说明 |
|---|---|---|
| `TSGgml_PagedAttentionForward` | 标准因果 / 滑窗注意力 | C++ K/V 聚合加 `ggml_flash_attn_ext`。Mistral 3 与大多数 GGML 分页注意力层默认使用。 |
| `TSGgml_PagedAttentionForwardWithSinks` | GPT OSS attention sinks | 将每头可学习 sink logit 加入 softmax 分母。 |
| `TensorPagedAttention.Forward` | Tensor 算子回退 | 使用 Tensor gather、批量 matmul 与 softmax，适合 A/B 测试。 |
| `ManagedPagedAttention.Forward` | 纯 C# 回退 | online-softmax 实现，用于正确性与未支持后端回退。 |

`TS_PAGED_ATTN_KERNEL=native|tensor|managed` 选择 Mistral 3 的派发路径。
GPT OSS 可用 `TS_GPTOSS_PAGED_ATTN_MANAGED=1` 强制走托管 sinks 路径。

### 按序列回退路径

回退路径仍运行在 `InferenceEngine` 内部；它不再是服务端外层并发原语。它会把
一个序列的 K/V 状态临时安装到旧模型 cache，调用 `model.Forward(tokens)`，捕获
写满的块，然后切换到下一个序列。这样旧路径或功能受限路径在移植到真正批处理
计算之前仍保持正确。

## 模型状态

| 模型家族 | 批处理 / 分页状态 | 关闭 / 子开关 |
|---|---|---|
| Mistral 3 | 默认 `ForwardBatch` 路径。使用分页 K/V、YaRN 感知位置、原生分页注意力，并在 prompt 准备后注入视觉 embedding。已在 Ministral-3-14B 上验证；长上下文原生分页注意力比旧按序列 GGML 路径快约 21%。 | `TS_PAGED_ATTN_KERNEL` 选择 `native`、`tensor` 或 `managed`。 |
| Gemma 4 | 密集文本负载默认走批处理路径，覆盖逐层 SWA / 全局注意力、可变 head dim、PLE、KV donor 层别名。当前回退场景包括待注入多模态 embedding、MoE 层与块量化 KV cache。 | `TS_GEMMA4_BATCHED=0` 强制按序列回退。 |
| Qwen 3 | attention-only 参考批处理移植，包含分页 K/V、逐 token RoPE position 与最后 token gather。提供基础 Qwen 3 GGUF 时可运行可选测试自验证。 | 无模型专属关闭开关；全局 `TS_SCHED_DISABLE_BATCHED=1` 强制回退。 |
| Qwen 3.5 / 3.6 family | 默认批处理路径。支持 FullAttention 层、通过每槽位状态池处理 GatedDeltaNet 递归层、MoE 变体、视觉注入与多模态 RoPE 表。 | `TS_QWEN35_BATCHED=0`；`TS_QWEN35_BATCHED_GDN_NATIVE=1` 启用原生批处理 GDN 内核。 |
| GPT OSS | 默认批处理路径。支持 Q/K/V/O bias、YaRN RoPE、滑窗层、attention sinks、MXFP4 MoE expert 与原生 sinks 注意力。已与旧路径做贪心正确性验证；性能仍主要受逐层图构建限制。 | `TS_GPTOSS_BATCHED=0`；`TS_GPTOSS_PAGED_ATTN_MANAGED=1`。 |
| Nemotron-H | 默认批处理路径。Attention 层使用分页 K/V；Mamba2 层使用每槽位 conv/SSM 状态池；MoE 层使用批处理 expert 内核；准备好的图像 / 音频 embedding 可注入到批处理 hidden state。 | `TS_NEMOTRON_BATCHED=0`；`TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1` 启用原生批处理 Mamba2 step。 |
| Gemma 3 | 尚无真正 `ForwardBatch` 移植；通过引擎的按序列回退执行。 | 仅全局回退。 |
| DiffusionGemma | 独立文本扩散路径。`Forward(int[] tokens)` 刻意不支持；生成会迭代去噪固定长度 canvas block。Web UI 请求共享 `DiffusionBatchScheduler`，在 block 之间接纳并发请求，并可选择批处理活跃 canvas。 | `DIFFUSION_STEPS`、`DIFFUSION_MAX_BATCH`、`DIFFUSION_BATCHED_FORWARD`；`DIFFUSION_NO_FUSED_DECODE=1` 关闭 GGML 融合整模型 diffusion decode。 |

## 测试覆盖

| 范围 | 测试 |
|---|---|
| 调度器 / 块池 | `ContinuousBatchSchedulerTests`、`PagedKvCacheTests`、`PagedKvCacheCodecTests` |
| 批处理执行原语 | `BatchedExecutorTests`，覆盖托管分页注意力正确性与多序列 logits 路由 |
| 按模型正确性 | `Qwen35BatchedCorrectnessTests`、`Mistral3BatchedForwardTests`、`Gemma4BatchedForwardTests`、`GptOssBatchedCorrectnessTests`、`NemotronBatchedCorrectnessTests`、可选 `Qwen3BatchedForwardTests` |
| 按模型性能探针 | `Gemma4BatchedPerfBench`、`Qwen35BatchedPerfBench`、`GptOssBatchedPerfBench`、`NemotronBatchedPerfBench` |
| DiffusionGemma 路径 | `DiffusionGemmaTests` 覆盖去噪、prompt-KV 缓存与批处理生成探针 |
| 端到端引擎行为 | 通过 `TS_TEST_MODEL_DIR` 指向真实 GGUF 后运行的 `EngineParallelInferenceTests` |
| 服务端参数翻译 | `ServerOptionsBuilderTests` 覆盖 `--continuous-batching`、`--no-continuous-batching` 与分页 KV 兼容参数 |

## 配置

| 变量 | 默认 | 作用 |
|---|---|---|
| `TS_SCHED_DISABLE_BATCHED` | `0` | 设为 `1` 后，即使模型实现了 `IBatchedPagedModel` 也强制按序列 KV-swap 回退。 |
| `TS_SCHED_MAX_BATCHED_TOKENS` | `4096` | 每步 token 预算。 |
| `TS_SCHED_MAX_RUNNING_SEQS` | `16` | 最大同时执行序列数。 |
| `TS_SCHED_PREFILL_CHUNK` | `1024` | 每步最多调度的 prefill token 数。 |
| `TS_SCHED_NUM_BLOCKS` | `256` | 引擎块池物理块数。 |
| `TS_SCHED_BLOCK_SIZE` | `256` | 每块 token 数。 |
| `TS_SCHED_PREFIX_CACHE` | `1` | 设为 `0` 关闭块哈希前缀复用。 |
| `TS_SCHED_DECODE_QUANTUM` | block size | 在偏回退路径中，允许切换序列前的 decode token 数。 |
| `TS_BATCHED_N1_FAST_PATH` | `0` | 将符合条件的单序列步骤也走批处理路径，用于 A/B 测试。 |
| `TS_KV_PAGED_QUANT_BITS` | `0` | 可选 TurboQuant 分页 KV 块编码位数（`4` 或 `8`）；带递归状态的模型可能回退到 passthrough。 |
| `DIFFUSION_STEPS` | `48` | Web UI DiffusionGemma 每个 block 的去噪步数；与自回归调度器的 step 预算无关。 |
| `DIFFUSION_MAX_BATCH` | `2` | diffusion scheduler 中同时活跃的 DiffusionGemma Web UI 请求数上限。 |
| `DIFFUSION_BATCHED_FORWARD` | `0` | 对活跃 DiffusionGemma canvas 启用真正的批处理 decode；默认更偏向融合单 canvas 路径。 |

服务端 CLI 别名：

```bash
--continuous-batching      # 默认，设置 TS_SCHED_DISABLE_BATCHED=0
--no-continuous-batching   # 设置 TS_SCHED_DISABLE_BATCHED=1
--paged-batching           # --continuous-batching 的别名
--no-paged-batching        # --no-continuous-batching 的别名
```

旧的 `--paged-kv*` 参数只为已移除的独立按会话分页 KV 管理器保留兼容。当前
服务端请求 KV 状态由 `InferenceEngine` 持有。

## 后续工作

- 为整批注意力构建一个原生 GGML 图，而不是每个序列一个小图。这会降低大量短序列场景下的 launch / compile 开销。
- 在后端支持且收益明确时，将 K/V 聚合从 CPU memcpy 迁移到 GPU 侧 `ggml_get_rows` 或等价 indexed gather。
- 补齐 Gemma 4 对 MoE 变体、多模态待注入 embedding、块量化 KV cache 的批处理覆盖。
- 根据实际运维需要，决定是否把 DiffusionGemma scheduler 指标接入
  `/api/queue/status` 或单独的 diffusion 端点。
- 将准备好的多模态 embedding 列表从模型级可变状态迁移到 `SequenceState`，使多模态 prompt 准备也能完全并行，而不是提交前串行化。
- 当客户端不再依赖旧字段后，移除队列位置兼容事件。
