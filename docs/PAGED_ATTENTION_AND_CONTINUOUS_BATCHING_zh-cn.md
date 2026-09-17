# TensorSharp 中的分页注意力与连续批处理

[English](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md) | [中文](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)

本文是 TensorSharp 当前 vLLM 风格分页 KV cache、基于块哈希的前缀共享、
以及迭代级连续批处理的实现参考。服务端默认通过这套引擎执行推理；旧的
单请求 FIFO 队列对象只作为队列状态 / 事件形状的 no-op 兼容 shim 保留。

请把本文当作实现参考，而不是对所有路径的性能承诺。通用分页 K/V 池目前驻留在
主机内存中，仍是瓶颈；受支持的模型 / 后端组合则可以改走设备驻留的 token 批量
融合 decode。因此吞吐取决于实际选中且被模型接受的执行路径。
参见[并发实测表现](#并发实测表现)。

## 当前状态

| 范围 | 状态 |
|---|---|
| 服务端引擎 | `TensorSharp.Server` 为当前加载模型持有一个 `InferenceEngineHost`。`ChatGenerationPipeline` 将渲染后的 prompt 提交给引擎，并从 `InferenceRequestHandle` 流式读取 token。 |
| 调度器 | `ContinuousBatchScheduler` 负责接纳等待请求、在块压力下抢占运行中的序列、应用每步 token 预算，并按内容哈希共享完整前缀块。 |
| KV 存储 | `BlockPool`、`BlockTable`、`PagedKvStorage`、`BlockHashIndex` 持有固定大小物理块，包含引用计数、LRU 空闲顺序与内容寻址查找。块字节存放在**托管主机内存**中。 |
| 批处理执行 | 实现 `IBatchedPagedModel.ForwardBatch` 的模型会把本轮所有序列打包到一次模型调用中，显式传入 `positions`、`slotMapping`、`queryStartLoc` 与每序列 block table。 |
| 回退执行 | 路径选择集中在 `ExecutionPlanner`：模型+后端能力（`ExecutionCapabilities`）、运维覆盖（`ExecutionOptions`）与每步请求特征共同产出 `ExecutionPlan`（选中路径、回退链、被拒原因）。模型仍可对某个具体 batch 抛出 `NotSupportedException`，该步会落入计划中的下一个候选，最终止于按序列 KV-swap 路径。 |
| 原生注意力 | `TSGgml_PagedAttentionForward` 在 C++ 中聚合分页 K/V 并派发 `ggml_flash_attn_ext`；GPT OSS 使用 `TSGgml_PagedAttentionForwardWithSinks`。 |
| 投机解码 | 可选的 MTP / NextN 草稿头加速单序列（无并发）请求。`BatchExecutor` 为实现了 `IBatchedSpeculativeTarget` 的模型（Qwen 3.6 内嵌 NextN；Gemma 4 独立 `gemma4-assistant` 草稿 GGUF）驱动共享的 `SpeculativeExecution` 起草 / 验证 / 回滚核心。默认关闭；服务端 `--spec`。详见 [投机解码（MTP / NextN）](#投机解码mtp--nextn)。 |
| 并发吞吐 | 通用的主机驻留 `BatchedPaged` 路径本身不能扩展合计 decode 吞吐；Gemma 4 的一次测量在约 **69 tok/s** 处饱和。模型实现 `TryForwardBatchedFusedDecode` 时，per-sequence-fused 路径会先尝试用一张设备图处理活跃 decode 子集，仅在该批次不满足条件或被拒绝时才按请求回退。参见[并发实测表现](#并发实测表现)。 |
| 设备驻留分页池 | **已实现，但未接入。** `TSGgml_PagedKvPool*`（`TensorSharp.GGML.Native/ggml_ops_paged_kv_pool.cpp`）与其托管封装 `DevicePagedKvCache`（`TensorSharp.Models/Paged/DevicePagedKvCache.cs`）都已存在，但没有任何模型或 executor 调用它们，因此它还不是已交付的功能。 |
| 队列 API | `InferenceQueue` 是 no-op 兼容层。`/api/queue/status` 与队列位置事件形状保留给依赖这些字段的客户端，不再承担请求串行化。 |
| 扩散模型 | DiffusionGemma 不进入这套自回归 `ForwardBatch` 契约。CLI 生成使用 `DiffusionGemmaSampler`；Web UI 使用 `DiffusionBatchScheduler` 在 block 边界批处理去噪工作。 |

## 并发实测表现

下列结果与具体路径、模型相关：它们解释了通用的主机驻留分页路径为何仍然较慢，
并不代表成功进入 token 批量融合 decode 的模型 / 后端组合。

- **实测的主机分页路径没有把并发转化为总吞吐。** 在 gemma-4-E4B / 1x Blackwell
  上通过服务端 chat 接口实测，无论同时有多少序列在跑，`BatchedPaged` 路径都会
  在约 **69 tok/s** 处饱和。
- **原因在于 K/V 放在哪里。** `PagedKvStorage` 是托管主机内存，因此批处理路径在
  每一步的每一层都要把序列历史从主机内存聚合出来并推过总线。即便是原生内核也
  只对 Q 与 OUT 做零拷贝——`ggml_ops_paged_attention.cpp` 里就写着 "K and V are
  still passed as host scratch arrays (the caller gathers …)"。相对按序列 fused
  decode，这大约是 **7.7 倍的每 token 开销**，而且随总历史长度增长。
- **默认并发路径由能力决定。** 对声明了 `SupportsPerSequenceFusedForward` 的模型，
  `ExecutionPlanner` 在 N >= 2 时选择 `PerSequenceFused`。executor 会先把满足条件的
  decode 子集交给 `TryForwardBatchedFusedDecode`，让一张图摊薄权重读取（包括 Qwen
  3.5/3.6 在 GGML CUDA/Metal 上的 slot-stable arena）；模型拒绝时，同一步再回退为
  N 个相互隔离的 fused 前向。
- **确实有效的部分。** 迭代级调度、块哈希前缀共享、抢占、按序列原生 slot、
  per-request fused holder，以及按能力启用的 token 批量融合 decode。批次被拒绝时
  仍保持正确性与公平性，但吞吐可能回到 round-robin 上限。
- **已实现但未接入。** 设备驻留的分页 K/V 池已经存在
  （`TensorSharp.GGML.Native/ggml_ops_paged_kv_pool.cpp`，托管封装为
  `TensorSharp.Models/Paged/DevicePagedKvCache.cs`）：池本身是后端张量，一步的
  K/V 用 `ggml_set_rows` 写入，序列历史在注意力图内用 `ggml_get_rows` 在设备侧
  聚合。**但没有任何模型或 executor 调用它。** 它还不是已交付的功能，也没有改变
  上面任何一个数字。

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
               - 执行 ExecutionPlanner 选中的路径
               - 按序列 fused / 批处理 ForwardBatch /
                 按序列 KV-swap 回退
               - 采样 decode token
               - 捕获新写满的 KV 块
        |
        v
BlockPool + PagedKvStorage + BlockHashIndex   (托管主机内存)
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
| `DevicePagedKvCache` | `TensorSharp.Models/Paged/DevicePagedKvCache.cs` | 基于 `TSGgml_PagedKvPool*` 的设备驻留分页 K/V 池。**已实现，但未接入任何模型**——目前没有任何地方构造它。 |
| `SequenceState` | `TensorSharp.Runtime/Scheduling/SequenceState.cs` | 每请求的状态、token、块、logits 与采样信息。 |
| `ContinuousBatchScheduler` | `TensorSharp.Runtime/Scheduling/ContinuousBatchScheduler.cs` | 带前缀缓存与抢占的迭代级调度器。 |
| `BatchExecutor` | `TensorSharp.Runtime/Scheduling/BatchExecutor.cs` | 执行计划中的步骤、采样并捕获 KV 块。 |
| `ExecutionPlanner` | `TensorSharp.Runtime/Scheduling/ExecutionPlanner.cs` | 纯函数路径选择：能力 + 覆盖 + 步特征 → `ExecutionPlan`。 |
| `ExecutionCapabilities` | `TensorSharp.Runtime/Scheduling/ExecutionCapabilities.cs` | 已加载模型 × 后端组合的声明式能力快照。 |
| `ExecutionOptions` | `TensorSharp.Runtime/Scheduling/ExecutionOptions.cs` | executor 级 `TS_*` 覆盖的结构化快照（唯一读取处）。 |
| `InferenceEngine` | `TensorSharp.Runtime/Scheduling/InferenceEngine.cs` | worker loop 与公开 submit/abort 接口。 |
| `InferenceEngineHost` | `TensorSharp.Server/InferenceEngineHost.cs` | 服务端按模型注册的引擎单例。 |

## 请求流程

1. 协议适配器构造归一化的聊天请求。
2. `ChatGenerationPipeline` 渲染 prompt，解析采样参数，并准备图像 / 音频 / 视频 embedding。
3. Pipeline 创建 `SequenceState` 并调用 `InferenceEngine.SubmitRequest`。
4. 引擎 worker 向 `ContinuousBatchScheduler` 请求下一步工作。
5. 调度器在 token 与序列预算允许时接纳等待序列。分配新块前，它会在 `BlockHashIndex` 中查找完整 prompt 块，命中时直接复用共享块。
6. 块池压力较大时，调度器可以抢占优先级较低的运行序列，提交其完整块、释放剩余块，并重新排入等待队列。
7. `BatchExecutor` 执行本步工作。它向 `ExecutionPlanner` 请求本步的 `ExecutionPlan`，并运行第一个接受该步的候选路径（见 [执行规划](#执行规划capability-model)）。
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

## 执行规划（Capability Model）

路径组合（批处理 / 回退、融合 / 逐算子、多模态 / 文本、投机 / 标准、按模型
opt-out、`TS_*` 覆盖）的数量已经超出零散 `if` 链可维护的范围，因此路径选择
收敛为一个纯函数：

```text
ExecutionCapabilities（模型 × 后端，声明式）
        +
ExecutionOptions     （运维 TS_* 覆盖，集中读取）
        +
SchedulerConfig      （引擎配置，如 --spec）
        +
ExecutionStepFeatures（本步请求特征：N、待注入多模态、
        |             KV 驻留位置、fused cache 驻留、是否需换主）
        v
ExecutionPlanner.PlanStep(...)
        |
        v
ExecutionPlan
  - 选中路径 + 有序回退链
  - Rejections：每条本可选择但未选择的路径及其原因
```

要点：

- **声明式能力，而非异常探测。** 模型通过 `IBatchedPagedModel` 的 getter
  （`BatchedForwardAvailable`、`SupportsBatchedMultimodal`、
  `SupportsPerSequenceFusedForward`、`SupportsLinearKVMigration` 等）与 MTP
  接口（`HasMtp`、`SpeculationProfitable`、`SupportsBatchedSpecTrunk`）声明
  自身能力，`ExecutionCapabilities.FromModel` 每步做一次快照。像
  `TS_QWEN35_BATCHED=0` 这类按模型 opt-out 现在会体现为
  `BatchedForwardAvailable=false`，planner 会提前绕开批处理路径；
  `ForwardBatch` 抛 `NotSupportedException` 仅保留为针对单个 batch 的拒绝，
  不再是路由机制。
- **候选有序且安全。** 可拒绝的候选（`SpecBatchedTrunk` 的武装/连续性门、
  `BatchedPaged` 的迁移失败/模型拒绝）会落入下一个候选；每个计划都以不可
  拒绝的路径收尾。`ExecutionPlannerTests` 扫描能力/特征空间验证该不变量。
- **可观测性。** `InferenceEngine` 启动时输出一次 capability 报告（哪些路径
  静态可用、不可用的原因）；`BatchExecutor` 在决策变化时（如并发切换）记录
  计划——选中路径、回退链、被拒原因——"这个请求为什么没走快路径" 从考古
  变成日志事实。
- **路径种类**（`ExecutionPathKind`）：`SpecBatchedTrunk`、`SpecPerSequence`、
  `PerSequenceFused`、`MixedMultimodalSplit`、`SingleSequenceFused`（N=1 快速
  路径）、`BatchedPaged`、`PerSequence`。

## 执行路径

### 批处理路径

批处理路径把本轮所有序列打包进一次前向：它避免了 K/V 所有权交换，并在所有调度
token 上摊薄线性投影开销。它是唯一能在整批上摊薄权重读取的路径，但它的 K/V 池
驻留在主机内存，所以这份摊薄目前并没有变成吞吐（见
[并发实测表现](#并发实测表现)）。对声明了按序列 fused 前向的模型，planner 在
N >= 2 时改选 `PerSequenceFused`，因此 `BatchedPaged` 实际服务的是没有 fused
前向的模型——或显式设置 `TS_PER_SEQ_FUSED=0` 做 A/B 时。当前大多数批处理移植在
GGML 后端使用原生分页注意力：

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

### 投机解码（MTP / NextN）

当设置服务端 `--spec` 参数（环境变量 `TS_SPEC=1`，旧名 `TS_MTP_SPEC=1`）时，`BatchExecutor` 会为实现了
`IBatchedSpeculativeTarget` 的模型对**单序列（无并发）**请求运行可选的多 token 预测
投机路径。每步流程：

1. **起草。** 模型的草稿头最多提议 `TS_MTP_DRAFT`（默认 `8`）个未来 token，遇到第一个
   草稿置信度低于 `TS_MTP_PMIN`（默认 `0.15`）的 token 即停止。起草由该请求自己的采样器
   （temperature、top-k/p、重复/存在/频率惩罚）驱动，使投机与标准 decode 的产出保持一致。
2. **验证。** 主干用一次批量前向验证所有起草 token，同一个采样器接受最长匹配前缀。由于验证
   会重新推导每个被提交的 token，输出与标准 decode **完全一致**；投机只改变所需的前向次数。
3. **回滚。** 部分接受时，超出已接受前缀的 KV（及任何递归状态）在下一步前被回滚。

两种草稿头形态共用 `SpeculativeExecution` 核心：

| 模型 | 草稿头 | 被拒时的状态 |
|---|---|---|
| Qwen 3.6 | 主干 GGUF 中内嵌的 NextN 块（`{arch}.nextn_predict_layers`）；无需额外文件。`--draft-model` 被忽略。 | GatedDeltaNet 递归状态快照 / 恢复（CUDA 上在设备侧完成）。 |
| Gemma 4 | 通过 `--draft-model` 加载的独立 EAGLE 风格 `gemma4-assistant` GGUF；草稿层读取**目标**最后一个 local / global 层的 KV（自身无 K/V）。 | 仅注意力 KV 位置回退——草稿器在给定 `(token, h)` 时无状态。 |

投机仅在有收益处启用（`SpeculationProfitable`）：ggml 后端（融合多 token 验证 + 草稿步
内核）与纯 C# `cuda` 后端（完全驻留 GPU 的逐算子验证 / 草稿）。在 CPU / GGML CPU / MLX 上验证
跟不上，因此引擎走标准 decode。并发批次从不投机——当有多个序列在运行时，每个序列都走普通的
批处理 / 回退步骤。Gemma 4 草稿 GGUF 不匹配或不完整会在服务端启动时立即失败
（`SpeculationStartupValidation`）。

## 模型状态

| 模型家族 | 批处理 / 分页状态 | 关闭 / 子开关 |
|---|---|---|
| Mistral 3 | 默认 `ForwardBatch` 路径。使用分页 K/V、YaRN 感知位置、原生分页注意力，并在 prompt 准备后注入视觉 embedding。已在 Ministral-3-14B 上验证；长上下文原生分页注意力比旧按序列 GGML 路径快约 21%。 | `TS_PAGED_ATTN_KERNEL` 选择 `native`、`tensor` 或 `managed`。 |
| Gemma 4 | 密集文本负载默认走批处理路径，覆盖逐层 SWA / 全局注意力、可变 head dim、PLE、KV donor 层别名。当前回退场景包括待注入多模态 embedding、MoE 层与块量化 KV cache。已完成请求的 request-owned fused K/V holder 可被保留，用于精确前缀续接。并发（N>=2）的 decode 步骤跑 token 批量融合内核（`TSGgml_Gemma4ModelDecodeBatchedEx`：一张图、每序列一个 token、权重只读一次），它覆盖 per-layer embedding、KV-donor 层与已回绕的 SWA 环，因此 E2B/E4B 不再回退到轮询。可选地通过独立 `gemma4-assistant` 草稿 GGUF 做 MTP 投机解码。 | `TS_GEMMA4_BATCHED=0` 强制按序列回退；`TS_RETAINED_FUSED_CACHE=0` 关闭 retained-holder 续接。服务端只需 `--draft-model` 即可启用投机（显式 `--no-spec` 可否决）；`TS_GMTP_*` 为草稿路径 A/B 开关。 |
| Qwen 3.5 / 3.6 family | 默认批处理路径。支持 FullAttention 层、通过每槽位状态池处理 GatedDeltaNet 递归层、MoE 变体、视觉注入与多模态 RoPE 表。其 request-owned fused holder 会把 attention K/V 与匹配的 GDN 递归状态保存在一起；正常结束的 holder 可被保留并重新绑定，用于精确前缀续接。Qwen 3.6 还通过其内嵌 NextN 块支持 MTP 投机解码（GDN 递归状态快照 / 回滚）。 | `TS_QWEN35_BATCHED=0`；`TS_QWEN35_BATCHED_GDN_NATIVE=1` 启用原生批处理 GDN 内核；`TS_RETAINED_FUSED_CACHE=0` 关闭 retained-holder 续接；服务端 `--spec` 在 Qwen 3.6 上启用投机。 |
| GPT OSS | 默认批处理路径。支持 Q/K/V/O bias、YaRN RoPE、滑窗层、attention sinks、MXFP4 MoE expert 与原生 sinks 注意力。已与旧路径做贪心正确性验证；性能仍主要受逐层图构建限制。 | `TS_GPTOSS_BATCHED=0`；`TS_GPTOSS_PAGED_ATTN_MANAGED=1`。 |
| Nemotron-H | 默认批处理路径。Attention 层使用分页 K/V；Mamba2 层使用每槽位 conv/SSM 状态池；MoE 层使用批处理 expert 内核；准备好的图像 / 音频 embedding 可注入到批处理 hidden state。 | `TS_NEMOTRON_BATCHED=0`；`TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1` 启用原生批处理 Mamba2 step。 |
| GLM 5.x | 没有 `ForwardBatch`：MLA 每个 token 只存一行压缩表示，DSA indexer 又要对同一段连续历史打分，没有分页 KV 布局可批。并发改由原生**序列 slot** 承担（`TSGgml_GlmSlotAlloc` / `SetActiveSlot` / `SlotFree`）——绑定请求只是切换活动 slot，不搬运 KV 字节，每个 slot 的计算图独立缓存与捕获。在此之上默认启用批量融合 decode（一张图、每序列一个 token，整批只读一遍权重）：4 个并发请求时合计 decode 提速 1.81×。批处理会改变 GEMM 形状，而 2 bit MoE 可能把这点差别放大成不同的专家选择。 | `TS_BATCHED_FUSED_DECODE=0` 关闭批量 decode；`TS_GLM_BATCHED_DECODE=0` 让原生侧拒绝它。 |
| DiffusionGemma | 独立文本扩散路径。`Forward(int[] tokens)` 刻意不支持；生成会迭代去噪固定长度 canvas block。Web UI 请求共享 `DiffusionBatchScheduler`，在 block 之间接纳并发请求，并可选择批处理活跃 canvas。 | `DIFFUSION_STEPS`、`DIFFUSION_MAX_BATCH`、`DIFFUSION_BATCHED_FORWARD`；`DIFFUSION_NO_FUSED_DECODE=1` 关闭 GGML 融合整模型 diffusion decode。 |

### 保留 fused holder 的续接

这与共享的分页前缀缓存不同。模型可能无法从按字节保存的分页快照重建完整续接
状态，但仍可拥有自包含的 per-request fused holder。模型声明
`SupportsRetainedFusedCache` 后，executor 可以把正常结束的 holder 保存在一个小型
LRU 中；后续请求精确扩展已记录的 token 前缀时，再把该 holder 重新绑定给新请求。
Gemma 4 保留其环形 attention K/V；Qwen 3.5/3.6 则把 attention K/V 与匹配的
GatedDeltaNet 递归状态作为一个混合 holder 一起保留。未声明该能力的模型会忽略这组设置。
带作用域的会话在主（N=1）缓存上结束的请求，在 fused 步骤接管模型时也会以同样方式保留，因此不会因为
另一个会话插在它两轮之间到达而丢失自己的状态。

### 跨请求的提示复用：会话作用域与媒体身份

所有跨请求复用路径——live cache 续接、保留的 holder、共享前缀检查点和池化块——都遵守两条规则。

**会话作用域。** 每个 `SequenceState` 携带一个 `CacheScope`（不透明的哈希）以及它的公开边界
`SharedPrefixTokens`（开头的 system/developer 消息加工具声明）。其他作用域产生的状态只能复用到
这个公开前缀为止，而且只能通过被克隆的共享前缀检查点：绝不会采纳、回退进入或移走另一个会话
的保留 holder，绝不会续接它的 live cache，公开前缀之后的池化块在哈希中带有作用域。带作用域的
请求也不会克隆比自己公开前缀更长的检查点。作用域由 chat 层给出：

| 请求 | 作用域 |
|---|---|
| 带 `sessionId` 的 Web UI / TensorAgent | 会话及其新会话纪元（`newChat:true` 开始一个新纪元） |
| OpenAI Chat / Responses、Ollama chat、不带 `sessionId` 的 Web UI | 请求历史证明自己所延续的会话：它最后一条 assistant 消息是本服务器生成并发出的回合（见下文）；否则是一个全新的作用域 |
| Skills / 代码工具循环的各轮 | 启动该循环的客户端回合的作用域 |
| 不设置作用域的引擎调用方（基准测试、CLI） | 无作用域，与所有作用域匹配（行为不变） |

chat 层的原始 token 拼接遵循同样的身份。每个生成的回合都以其之前的客户端可见历史的内容哈希链
（角色、内容、工具调用、按内容计的媒体）为键记录下来，同时记录原始输出 token 以及当时发给客户端
的内容（解析后的正文和工具调用，或原始文本）。之后的 assistant 消息只有在（忽略空白后）等于这份
已发出的内容时才会用记录的 token 渲染；客户端自己编写或修改过的 assistant 消息按其自身文本渲染。
在此之前，无状态 API 共享同一份跟踪历史，会把另一个客户端生成的回合拼接到本客户端自己的消息上。
并发的会话也不再相互覆盖记录。

**媒体身份。** 每张图片、视频帧（对）和音频片段都以其字节的 SHA-256 标识。Base64 附件（OpenAI
`image_url`、Responses `input_image`、Ollama `images`、音频）以 `<sha256>.<ext>` 存储且只写一次，
因此客户端每轮重发同一张图片只保留一个文件。视觉与音频嵌入缓存以该内容 id 为键，受
`TS_MM_EMBEDDING_CACHE_MB` 约束并按最近最少使用淘汰，已准备好的提示仍引用的条目不会被淘汰。
请求以位置区间的形式携带其媒体（`SequenceState.MediaSpans`）；当缓存前缀内的每个区间都是同一
位置上的相同内容时，该前缀可以复用，复用长度会被截到它将切断的任何区间的起点。因此第一张图片
之前的文本总是可以复用。池化块哈希只把区间 id 混入包含该区间的块（并通过父链带入其后的所有块），
而不混入之前的块。

Qwen 3.5/3.6 声明 `SupportsReuseAcrossMediaSpan = false`：它们的 M-RoPE 提示位置在图片之后被压缩，
但 decode 使用绝对 token 下标，holder 也不记录 rope 偏移，因此越过图片续接缓存得到的状态与重新
prefill 不同。在 decode 使用压缩位置之前，它们的复用止于第一个媒体区间；Gemma 4 使用绝对位置，
可以越过图片续接。

在复用前缀*之后*预填充图片是另一回事。Gemma 4 的融合 prefill 只在起始位置 0 输出图片的双向掩码，
因此这样的分块走较慢的逐算子路径。在滑动窗口之内，该路径与冷启动 prefill 逐 token 一致（但更慢：
E4B/Metal 上一个复用 179 token 的 457 token 图片回合首 token 用时 1.25 s，而不是 0.66 s）。一旦提示
超出窗口就不再一致，所以 `IModelArchitecture.CanPrefillMediaAfterReusedPrefix` 让这样的回合不复用、
从零走融合路径 prefill；其后的文本回合仍会越过图片续接缓存。

在 Gemma 4 上，不超过 `MaxReusablePrefixTokens`（滑动窗口）个 token 的回合现在也会续接 live cache；
之前这类回合落到池化路径，只能返回整块的 256 token。已回绕环上的回退依旧被拒绝。

准入日志会写明服务该请求的来源——`the model's live KV cache of this conversation`、
`a shared-prefix checkpoint (public, N tokens)`、`a retained holder of this conversation` 或
`pooled prefix-cache blocks`——带 token 数和截断哈希形式的作用域；Debug 级别下一行
`blocked by scope` 报告另一个会话的状态在公开前缀之后还匹配了多少 token。

## 测试覆盖

| 范围 | 测试 |
|---|---|
| 调度器 / 块池 | `ContinuousBatchSchedulerTests`、`PagedKvCacheTests`、`PagedKvCacheCodecTests` |
| 批处理执行原语 | `BatchedExecutorTests`，覆盖托管分页注意力正确性与多序列 logits 路由；`RetainedFusedCacheTests` 覆盖按能力启用的 holder 保留 / 重新绑定与 LRU 清理、会话作用域隔离（含随机交错的性质测试）与按位置的媒体检查 |
| 跨请求隔离与媒体身份 | `ModelServiceRawTokenHistoryTests` 与 `ToolTranscriptSpliceTests`（按内容校验的原始 token 拼接）、`PooledPrefixScopeAndMediaTests`、`ContentAddressedMediaTests` |
| 按模型正确性 | `Qwen35BatchedCorrectnessTests`、`Mistral3BatchedForwardTests`、`Gemma4BatchedForwardTests`、`GptOssBatchedCorrectnessTests`、`NemotronBatchedCorrectnessTests` |
| MTP 投机解码 | `SpeculativeExecutionTests`（起草 / 验证 / 回滚核心）、可选端到端 `Qwen36SpeculativeTests`（`TS_MTP_E2E=1`）与 `Gemma4SpeculativeTests`（`TS_GMTP_E2E=1`），需真实 GGUF |
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
| `TS_SCHED_PREFILL_CHUNK` | `256` | 存在活跃 decode 时每请求的 prefill 上限；仅 prefill 的步骤会公平分配完整 token 预算。服务端参数：`--prefill-chunk-size N`。 |
| `TS_SCHED_SOLO_PREFILL_CHUNK` | `8192` | solo（无争用）请求的每步 prefill 上限——以大分块把 prompt 送入融合整图 prefill 路径。受 `TS_SCHED_MAX_BATCHED_TOKENS` 约束。 |
| `TS_SCHED_NUM_BLOCKS` | `256` | 引擎块池物理块数。 |
| `TS_SCHED_BLOCK_SIZE` | `256` | 每块 token 数。 |
| `TS_SCHED_PREFIX_CACHE` | `1` | 设为 `0` 关闭准入时的全部提示复用：池化块、live cache 续接、保留的 holder 和共享前缀检查点。 |
| `TS_SCHED_STOP_REPETITION` | `1` | 设为 `0` 时，陷入重复循环的生成会继续跑到 token 上限，而不是以 `repetition` 结束原因停止。 |
| `TS_SCHED_DECODE_QUANTUM` | `256` | 在偏回退路径中，允许切换序列前的 decode token 数。 |
| `TS_BATCHED_N1_FAST_PATH` | `1` | solo 单序列步骤走融合 N=1 快速路径 decode；设为 `0` 可强制这些步骤走完全批处理路径（A/B 测试）。 |
| `TS_PER_SEQ_FUSED` | `1` | fused 能力模型上的并发（N≥2）序列走 per-request 融合 Forward；设为 `0` 强制走逐算子批处理分页路径（A/B 测试）。 |
| `TS_BATCHED_FUSED_DECODE` | `1` | `0` 在 per-seq fused 路径内关闭真正的 token 批量融合 decode（一张图同时 decode 全部 N 个序列）。 |
| `TS_GEMMA4_BATCHED_CAPS` | 原生探测 | 覆盖 Gemma 4 token 批量内核报告的能力位（1 PLE、2 KV donor、4 SWA 回绕）；`0` 强制 v1 门控，PLE / 共享 KV / 已回绕 SWA 的模型（E2B/E4B）改为轮询 decode（A/B 测试）。 |
| `TS_RETAINED_FUSED_CACHE` | `1` | 对声明支持的模型，保留已完成请求的 request-owned fused holder，用于精确前缀续接；`0` 关闭（限 VRAM / A/B）。支持的 holder 包括 Gemma 4 K/V，以及 Qwen 3.5/3.6 的 attention K/V 与 GDN 递归状态。 |
| `TS_RETAINED_FUSED_CACHE_MAX` | `4` | 保留 fused holder 的 LRU 预算（每个 holder 都会占用模型完整的 per-request 续接状态）。 |
| `TS_PREFIX_CHECKPOINTS` | `1` | 在共享提示前缀结束处（由 chat 层在请求上标记的边界）对模型完整状态做检查点，并让每个新会话从其副本开始（Gemma 4、Qwen 3.5/3.6）。`0` 关闭。 |
| `TS_PREFIX_CHECKPOINTS_MAX` | `2` | 同时保留多少个不同共享前缀的检查点（LRU）。 |
| `TS_MM_EMBEDDING_CACHE_MB` | `512` | 视觉/音频嵌入缓存的字节预算，缓存以媒体内容（SHA-256）为键；超出后淘汰没有被已准备提示引用的最近最少使用条目。 |
| `TS_KV_INITIAL_TOKENS` | `0` | 缓存创建时、任何请求声明预算之前分配的 K/V token 数；`0` 沿用引擎策略（显式 `MAX_CONTEXT` 时为整个窗口）。缓存仍按需增长。 |
| `TS_KV_GENERATION_RESERVE_MAX` | `0` | 请求预先保留的 K/V（prompt + max_new_tokens）中生成部分的上限；`0` = 不限制。超过上限后缓存按需增长。 |
| `TS_KV_HOLDER_POOL_MAX` | `64` | 模型最多可停放多少个已释放的 per-request holder 以待复用；停放期间每个都占用其完整 K/V 分配。 |
| `TS_KV_PAGED_QUANT_BITS` | `0` | 可选 TurboQuant 分页 KV 块编码位数（`2`、`4` 或 `8`）；带递归状态的模型可能回退到 passthrough。 |
| `TS_MTP_SPEC` | `0` | `1` 为单序列启用 MTP / NextN 投机解码（服务端 `--spec`）。 |
| `TS_MTP_DRAFT` | `8` | 每个投机步最多起草的 token 数（服务端 `--spec-draft`）。 |
| `TS_MTP_PMIN` | `0.15` | 保留草稿 token 所需的最低草稿置信度（服务端 `--spec-pmin`；`0` 表示从不设阈）。 |
| `TS_MTP_DRAFT_MODEL` | 无 | Gemma 4 独立 `gemma4-assistant` 草稿 GGUF 路径（服务端 `--draft-model`）；Qwen 3.6 忽略。 |
| `TS_GMTP_NO_FUSED` / `TS_GMTP_NO_FAST_ROLLBACK` / `TS_GMTP_BATCHED_TRUNK` | 关闭 | Gemma 4 草稿路径 A/B 开关（关闭融合验证/草稿内核；恢复保留前缀回滚；用批量主干代替线性主干）。 |
| `DIFFUSION_STEPS` | `48` | Web UI DiffusionGemma 每个 block 的去噪步数；与自回归调度器的 step 预算无关。 |
| `DIFFUSION_MAX_BATCH` | `2` | diffusion scheduler 中同时活跃的 DiffusionGemma Web UI 请求数上限。 |
| `DIFFUSION_BATCHED_FORWARD` | `0` | 对活跃 DiffusionGemma canvas 启用真正的批处理 decode；默认更偏向融合单 canvas 路径。 |

宿主可以给引擎挂上一个 `IPrefixCheckpointStore`（`InferenceEngine.PrefixCheckpointStore`，
或 TensorSharp.Chat 中的 `InferenceEngineHost.PrefixCheckpointStore`），让共享前缀检查点
在进程之外保留：执行器在接纳一个共享前缀尚无内存中检查点的请求时读取已保存的检查点，并在
取得检查点的那一刻写入一份；字节格式由模型家族自己负责（`IBatchedPagedModel.TryExportRetainedCache`
/ `TryImportRetainedCache`，Qwen 3.5/3.6 与 Gemma 4 已实现）并在读入时校验。TensorAgent 的
`PrefixCheckpointFileStore` 是基于文件的实现，它让启动后的第一条消息和第二条一样快。


服务端 CLI 别名：

```bash
--continuous-batching      # 默认，设置 TS_SCHED_DISABLE_BATCHED=0
--no-continuous-batching   # 设置 TS_SCHED_DISABLE_BATCHED=1
--paged-batching           # --continuous-batching 的别名
--no-paged-batching        # --no-continuous-batching 的别名
--prefill-chunk-size N     # 设置 TS_SCHED_PREFILL_CHUNK
```

旧的 `--paged-kv*` 参数只为已移除的独立按会话分页 KV 管理器保留兼容。当前
服务端请求 KV 状态由 `InferenceEngine` 持有。

与分页 TurboQuant 编解码器（`TS_KV_PAGED_QUANT_BITS`）相互独立，KV cache 本身也可以用
`--kv-cache-dtype <f32|f16|q8_0|q4_0>`（或 `KV_CACHE_DTYPE` 环境变量）以更低精度存储，
CLI 与服务端均支持。默认按模型自动选择（模型权重低于 F32 时为 `f16`，否则为 `f32`）；
块量化档位（`q8_0`、`q4_0`）要求原生 GGML
flash-attention 路径，`q4_0`（约为 f32 的 1/7）面向 KV cache 主导内存占用的
128K–256K 超长上下文。

## 后续工作

- 为整批注意力构建一个原生 GGML 图，而不是每个序列一个小图。这会降低大量短序列场景下的 launch / compile 开销。
- 把设备驻留的分页 K/V 池接入某个模型。池本身已经实现（`ggml_ops_paged_kv_pool.cpp`
  加 `DevicePagedKvCache`），用 `ggml_set_rows` 写入与设备侧 `ggml_get_rows` 读取
  取代主机聚合，但还没有任何模型把分页 K/V 分配到那里，所以批处理路径仍然保持
  主机驻留的 cache 与实测上限。这是并发能换来任何吞吐的前提。
- 补齐 Gemma 4 对 MoE 变体、多模态待注入 embedding、块量化 KV cache 的批处理覆盖。
- 根据实际运维需要，决定是否把 DiffusionGemma scheduler 指标接入
  `/api/queue/status` 或单独的 diffusion 端点。
- 将准备好的多模态 embedding 列表从模型级可变状态迁移到 `SequenceState`，使多模态 prompt 准备也能完全并行，而不是提交前串行化。
- 当客户端不再依赖旧字段后，移除队列位置兼容事件。
