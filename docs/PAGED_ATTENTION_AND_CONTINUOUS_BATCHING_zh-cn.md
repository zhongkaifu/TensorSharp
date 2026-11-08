# TensorSharp 中的分页注意力与连续批处理

[English](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md) | [中文](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)

本文是 TensorSharp 当前 vLLM 风格分页 KV cache、Radix 前缀缓存以及迭代级连续批处理的实现参考。服务端通过这套引擎执行推理。

请把本文当作实现参考，而不是对所有路径的性能承诺。通用分页 K/V 池目前驻留在
主机内存中，仍是瓶颈；受支持的模型 / 后端组合则可以改走设备驻留的 token 批量
融合 decode。因此吞吐取决于实际选中且被模型接受的执行路径。
参见[并发实测表现](#并发实测表现)。

## 当前状态

| 范围 | 状态 |
|---|---|
| 服务端引擎 | `ModelService`（TensorSharp.Chat，服务端与 TensorAgent 共用）为每个加载的模型持有一个 `InferenceEngineHost`。`ChatGenerationPipeline` 将渲染后的 prompt 提交给引擎，并从 `InferenceRequestHandle` 流式读取 token。 |
| 调度器 | `ContinuousBatchScheduler` 负责接纳等待请求、在块压力下抢占运行中的序列、应用每步 token 预算，并从前缀缓存中为每个被接纳的 prompt 采纳可复用前缀。 |
| KV 存储 | `BlockPool`、`BlockTable`、`PagedKvStorage` 持有固定大小物理块，包含引用计数与 LRU 空闲顺序。块字节存放在**托管主机内存**中。 |
| 前缀缓存 | Radix 树（`PrefixTree`，由 `PrefixCacheCoordinator` 驱动）是服务端、CLI 生成与 TensorAgent 的默认前缀缓存：可复用页面、保留的会话状态与共享前缀检查点都在同一个索引里。`TS_SCHED_PREFIX_CACHE=0` 或 `--no-prefix-cache` 关闭复用。见 [Radix 前缀缓存](#radix-前缀缓存)。 |
| 批处理执行 | 实现 `IBatchedPagedModel.ForwardBatch` 的模型会把本轮所有序列打包到一次模型调用中，显式传入 `positions`、`slotMapping`、`queryStartLoc` 与每序列 block table。 |
| 回退执行 | 路径选择集中在 `ExecutionPlanner`：模型+后端能力（`ExecutionCapabilities`）、运维覆盖（`ExecutionOptions`）与每步请求特征共同产出 `ExecutionPlan`（选中路径、回退链、被拒原因）。模型仍可对某个具体 batch 抛出 `NotSupportedException`，该步会落入计划中的下一个候选，最终止于按序列 KV-swap 路径。 |
| 原生注意力 | `TSGgml_PagedAttentionForward` 在 C++ 中聚合分页 K/V 并派发 `ggml_flash_attn_ext`；GPT OSS 使用 `TSGgml_PagedAttentionForwardWithSinks`。 |
| 投机解码 | 为单序列（无并发）请求提供可选的投机解码：学习型草稿头（内嵌 NextN / MTP、Gemma 4 独立的 `gemma4-assistant` GGUF、Qwen 3.8 Flash Next 的共享 MTP 头）、块级草稿器（DSpark、DFlash）以及无需权重的 n-gram 投机器。`BatchExecutor` 为实现了 `ISpeculativeTarget` 的模型驱动共享的 `SpeculativeExecution` 起草 / 验证 / 回滚核心。默认关闭；CLI 与服务端用 `--spec` 或 `--draft-model` 开启。详见 [投机解码（MTP / NextN）](#投机解码mtp--nextn)。 |
| 并发吞吐 | 通用的主机驻留 `BatchedPaged` 路径本身不能扩展合计 decode 吞吐；Gemma 4 的一次测量在约 **69 tok/s** 处饱和。模型实现 `TryForwardBatchedFusedDecode` 时，per-sequence-fused 路径会先尝试用一张设备图处理活跃 decode 子集，仅在该批次不满足条件或被拒绝时才按请求回退。参见[并发实测表现](#并发实测表现)。 |
| 队列 API | `/api/queue/status` 报告引擎的实时负载：正在处理的请求、等待批处理槽位的请求与已完成的请求。引擎之外没有任何地方把请求串行化。 |
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
  3.5/3.6/3.8 在 GGML CUDA/Metal 上的 slot-stable arena——在 CUDA 上它也支持 K/IQ 量化的
  token embedding——以及 Gemma 4 支持按序列缓存容量的内核）；模型拒绝时，同一步再回退为
  N 个相互隔离的 fused 前向；服务端会记录第一次成功的批次和第一次被拒绝的原因。
- **确实有效的部分。** 迭代级调度、前缀共享（默认为 Radix 树）、抢占、按序列原生 slot、
  per-request fused holder，以及按能力启用的 token 批量融合 decode。批次被拒绝时
  仍保持正确性与公平性，但吞吐可能回到 round-robin 上限。

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
        |      - 前缀采纳（Radix 树）
        |
        +--> BatchExecutor
               - 执行 ExecutionPlanner 选中的路径
               - 按序列 fused / 批处理 ForwardBatch /
                 按序列 KV-swap 回退
               - 采样 decode token
               - 捕获新写满的 KV 块
        |
        v
BlockPool + PagedKvStorage   (托管主机内存)
```

### 核心组件

| 组件 | 文件 | 作用 |
|---|---|---|
| `KvBlock` | `TensorSharp.Runtime/Paged/KvBlock.cs` | 物理块元数据与引用计数。 |
| `BlockPool` | `TensorSharp.Runtime/Paged/BlockPool.cs` | 分配、释放、引用计数与淘汰物理块。 |
| `BlockTable` | `TensorSharp.Runtime/Paged/BlockTable.cs` | 将每个序列的逻辑块映射到物理块。 |
| `PagedKvStorage` | `TensorSharp.Runtime/Paged/PagedKvStorage.cs` | 按物理 block id 索引的字节 slab。 |
| `PrefixTree` / `PrefixCacheCoordinator` | `TensorSharp.Runtime/Scheduling/PrefixCache/` | Radix 前缀缓存：基于 prompt token 与媒体身份的 radix 键、作用域、预算，以及每个节点可以从中续接的负载。 |
| `IPrefixCacheModel` | `TensorSharp.Runtime/Scheduling/PrefixCache/IPrefixCacheModel.cs` | 按家族的契约：模型声明自己的复用能力，只实现状态操作（物化、捕获、释放）。 |
| `PrefixCheckpointFileStore` | `TensorSharp.Runtime/Scheduling/PrefixCheckpointFileStore.cs` | 基于文件的 `IPrefixCheckpointStore`，让共享前缀检查点跨启动保留（服务端与 TensorAgent）。 |
| `PagedKvBatchOps` | `TensorSharp.Runtime/Paged/PagedKvBatchOps.cs` | 批处理 K/V scatter 与每序列最后 token gather。 |
| `ManagedPagedAttention` | `TensorSharp.Runtime/Paged/ManagedPagedAttention.cs` | 纯 C# 分页注意力正确性回退。 |
| `SequenceState` | `TensorSharp.Runtime/Scheduling/SequenceState.cs` | 每请求的状态、token、块、logits 与采样信息。 |
| `ContinuousBatchScheduler` | `TensorSharp.Runtime/Scheduling/ContinuousBatchScheduler.cs` | 带前缀缓存与抢占的迭代级调度器。 |
| `ISoloPrefillBudgetModel` | `TensorSharp.Runtime/Scheduling/ISoloPrefillBudgetModel.cs` | 可选的模型契约：由模型决定单个请求的 prefill 步长（启用流水化 prefill 的 Muse Spark），除非设置了 `TS_SCHED_SOLO_PREFILL_CHUNK`。 |
| `BatchExecutor` | `TensorSharp.Runtime/Scheduling/BatchExecutor.cs` | 执行计划中的步骤、采样并捕获 KV 块。 |
| `ExecutionPlanner` | `TensorSharp.Runtime/Scheduling/ExecutionPlanner.cs` | 纯函数路径选择：能力 + 覆盖 + 步特征 → `ExecutionPlan`。 |
| `ExecutionCapabilities` | `TensorSharp.Runtime/Scheduling/ExecutionCapabilities.cs` | 已加载模型 × 后端组合的声明式能力快照。 |
| `ExecutionOptions` | `TensorSharp.Runtime/Scheduling/ExecutionOptions.cs` | executor 级 `TS_*` 覆盖的结构化快照（唯一读取处）。 |
| `InferenceEngine` | `TensorSharp.Runtime/Scheduling/InferenceEngine.cs` | worker loop 与公开 submit/abort 接口。 |
| `InferenceEngineHost` | `TensorSharp.Chat/InferenceEngineHost.cs` | 由 `ModelService` 持有的按模型引擎宿主，服务端与 TensorAgent 共用。 |

## 请求流程

1. 协议适配器构造归一化的聊天请求。
2. `ChatGenerationPipeline` 渲染 prompt，解析采样参数，并准备图像 / 音频 / 视频 embedding。
3. Pipeline 创建 `SequenceState` 并调用 `InferenceEngine.SubmitRequest`。
4. 引擎 worker 向 `ContinuousBatchScheduler` 请求下一步工作。
5. 调度器在 token 与序列预算允许时接纳等待序列。分配新块前，它会向 Radix 前缀缓存查询该 prompt 最长的可复用前缀（页面、保留的会话状态或共享前缀检查点）并采纳它。
   只有当空闲块在扣除运行中请求尚需分配的 prompt 块之后，仍能容纳等待请求的整段 prompt 时，才会接纳它；否则它留在队列中，直到有请求结束（单独一个请求总会被接纳）。
6. 块池压力较大时（decode 增长不预留），调度器可以抢占排名低于需要块的那个序列的运行序列（优先级更低，或同优先级但提交更晚），提交其完整块、释放剩余块，并重新排入等待队列。序列绝不抢占比它更早的序列：它会等待一步，因此块池满时按从旧到新的顺序排空，而不是让长 prefill 彼此抢占形成活锁。
7. `BatchExecutor` 执行本步工作。它向 `ExecutionPlanner` 请求本步的 `ExecutionPlan`，并运行第一个接受该步的候选路径（见 [执行规划](#执行规划capability-model)）。
8. 引擎把采样 token 发给 request handle，检查 EOS / max-tokens / abort 状态，并释放已完成序列的块。

前缀采纳会保留至少一个 prompt token 重新送入模型。这样即使可见前缀已经全部
命中前缀缓存，也能产生新的 logits 用于采样。

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
  `SupportsPerSequenceFusedForward`、`SupportsLinearKVMigration` 等）与投机解码
  接口（`HasDraftHead`、`SpeculationProfitable`、`SupportsBatchedSpecTrunk`）声明
  自身能力，`ExecutionCapabilities.FromModel` 每步做一次快照。模型的静态限制
  （带 MoE 层或块量化 KV cache 的 Gemma 4、张量并行）会体现为
  `BatchedForwardAvailable=false`，planner 会提前绕开批处理路径；
  `ForwardBatch` 抛 `NotSupportedException` 仅保留为针对单个 batch 的拒绝，
  不再是路由机制。`--no-continuous-batching`（`TS_SCHED_DISABLE_BATCHED=1`）会关闭
  所有批处理路径。
- **候选有序且安全。** 可拒绝的候选（`BatchedPaged` 的迁移失败/模型拒绝）会落入
  下一个候选；每个计划都以不可
  拒绝的路径收尾。`ExecutionPlannerTests` 扫描能力/特征空间验证该不变量。
- **可观测性。** `InferenceEngine` 启动时输出一次 capability 报告（哪些路径
  静态可用、不可用的原因）；`BatchExecutor` 在决策变化时（如并发切换）记录
  计划——选中路径、回退链、被拒原因——"这个请求为什么没走快路径" 从考古
  变成日志事实。
- **路径种类**（`ExecutionPathKind`）：`SpeculativePerSequence`、
  `PerSequenceFused`、`MixedMultimodalSplit`、`SingleSequenceFused`（N=1 快速
  路径）、`BatchedPaged`、`PerSequence`。

## 执行路径

### 批处理路径

批处理路径把本轮所有序列打包进一次前向：它避免了 K/V 所有权交换，并在所有调度
token 上摊薄线性投影开销。它是唯一能在整批上摊薄权重读取的路径，但它的 K/V 池
驻留在主机内存，所以这份摊薄目前并没有变成吞吐（见
[并发实测表现](#并发实测表现)）。对声明了按序列 fused 前向的模型，planner 在
N >= 2 时改选 `PerSequenceFused`，因此 `BatchedPaged` 实际服务的是没有 fused
前向的模型——或在 `TS_PER_SEQ_FUSED=0` 下服务所有模型，此时它们的 K/V 放在共享的主机块池而不是按请求的设备端 holder。当前大多数批处理移植在
GGML 后端使用原生分页注意力：

| 内核 | 范围 | 说明 |
|---|---|---|
| `TSGgml_PagedAttentionForward` | 标准因果 / 滑窗注意力 | C++ K/V 聚合加 `ggml_flash_attn_ext`。Mistral 3 与大多数 GGML 分页注意力层默认使用。后端没有 flash kernel 的 head 大小（ggml-cuda：40/64/72/80/96/112/128/256 以及 grouped-query 的 192/320/512/576 之外的任何大小）改为以显式注意力运行并警告一次，而不是在 `fattn.cu` 中 abort。 |
| `TSGgml_PagedAttentionForwardWithSinks` | GPT OSS attention sinks | 将每头可学习 sink logit 加入 softmax 分母。 |
| `ManagedPagedAttention.Forward` | 纯 C# 回退 | online-softmax 实现，用于正确性与未支持后端回退。 |

Mistral 3 与 Hunyuan Dense 在 GGML 后端上使用原生内核，其他后端使用托管实现。
GPT OSS 在 GGML 后端上使用原生 sinks kernel，其他后端使用托管实现。

### 并发下的输出一致性

同样的批次必须逐位给出同样的 logits；不同的批次不必。哪些请求共处一步、每个请求
前向多少 token，决定了跑哪些内核：ggml-cuda 对 MXFP4 的
量化 `mul_mat_id` 在 Turing 与 Ampere 上不超过 7 个 token 时走 MMVQ（Volta、Ada 与
Blackwell 上为 8 个），超过时走 MMQ；而独自先到达的序列会
先走单序列 fused 路径。这些内核只在浮点近似平局（near-tie）的范围内一致，贪心解码会
把一个近似平局变成另一段续写。因此，到达顺序没有固定的并发轮次，即使没有任何缺陷，
不同运行之间也可能产生不同的 token。调度出相同批次的两次运行则不可能不同。

区分这两种情况的方法：

- `TS_CB_DEBUG=1` 为引擎的每一步打印一行 `[cb] step#N <path>`，列出本步调度的每个
  请求（`id:P|D fwd=<token 数> computed=<本步之后的 token 数>`），以及它留下的 logits
  指纹：前两个 token、二者的差值（margin）和整行的哈希。逐步对比两次运行：如果在第一个
  哈希不同之前批次组成已经不同，属于近似平局一类；批次组成相同而哈希不同，就是缺陷。
- `AgentTurnBench --conc-gate` 让引擎的 compute gate 保持关闭，直到整轮并发请求都已
  入队，于是每次运行都以相同的批次接纳这一轮。这些行会记录 `ArrivalOrderFixed: true`，
  `compare.py` 要求它们的 token 完全相同。没有该标记的并发行，token 差异只作为信息
  报告，除非传入 `--require-concurrent-identity`。

**实测（2026-09-17，gpt-oss-20b MXFP4，1x A40，`ggml_cuda`，`TS_PER_SEQ_FUSED=0`，
`AgentTurnBench --conc 1,4,8`）。** 4 个并发贪心请求在不同运行之间于第 3-26 个输出
token 处分叉，连同一进程内的预热轮与测量轮之间也会分叉；在这些运行中 8 个并发请求则没有。步骤跟踪
显示两次运行的批次完全相同，而在第一个解码步，同一 token 的 logits 相差数个单位
（41.28 对 38.84），所有 margin 仍然很大。这是缺陷，不是近似平局。独立 MoE 内核
（`TSGgml_MoEFFNPrefillSwiGLUQuantF32`，批处理分页路径用它计算 GPT OSS 的专家）把每个
专家的 bias 作为图的叶子张量上传到可复用的计算缓冲区。分配器在 gate bias 的 `add_id`
之后释放了它，并把 SwiGLU 激活放在同一块内存上。ggml-cuda 在 1-7 个 token 时把
`{mul_mat_id, add_id, mul_mat_id, add_id, swiglu_oai}` 融合成一个 MMVQ 内核。该内核
在写激活的同时读取 bias，而它的内存重叠检查会跳过叶子张量，因为 llama.cpp 的 bias
都是权重。于是内核覆盖了它仍在读取的 bias。在这块 Ampere 卡上 8 个 token 走不做融合的 MMQ，这就是 8 请求
轮次保持稳定的原因；在 Ada 或 Blackwell 上 8 个 token 仍走 MMVQ，同样会受影响。现在图构建器用分配器的 output 标志固定住所有小的上传参数（ids、
路由权重、bias、post-norm 权重）。CTest `moe-fused-bias-alias-cuda`
（`GgmlOpsMoeFusedBiasAliasTest`）在 1、4、7 个 token 下把该内核与精确的主机计算结果
比较。修复前 4 个和 7 个 token 的误差分别高达 644 和 1118（容差为 21 和 25），且重复运行
结果不一致（1 个 token 的情况通过）；修复后与 CPU 后端一致。`moe-fused-bias-alias-metal`
在 Metal 上做同样的检查；Metal 不融合这条算子链，修复前也能通过。

修复后，在 `--conc-gate` 下，三轮运行在 1、4、8 请求轮次的每一步都给出逐位相同的
logits。不加 gate 时，三轮中仍有一轮改变了 8 请求轮次的输出：它的第一个请求在另外七个
到达之前被单独调度到单序列 fused 路径上。那一步的 logits 已经不同（42.94 对 42.86），
随后的 argmax 翻转发生在 0.011-0.11 的 margin 上。这属于近似平局一类，`compare.py`
只报告而不判失败。

### 按序列回退路径

回退路径仍运行在 `InferenceEngine` 内部；它不再是服务端外层并发原语。它会把
一个序列的 K/V 状态临时安装到旧模型 cache，调用 `model.Forward(tokens)`，捕获
写满的块，然后切换到下一个序列。这样旧路径或功能受限路径在移植到真正批处理
计算之前仍保持正确。

### 投机解码（MTP / NextN）

开启投机解码时——CLI 或服务端的 `--spec`（环境变量 `TS_SPEC=1`），或者
`--draft-model`（除非给出 `--no-spec`，它本身就会开启投机）——`BatchExecutor` 会为实现了
`ISpeculativeTarget` 的模型对**单序列（无并发）**请求运行共享的 `SpeculativeExecution` 起草 / 验证 /
回滚核心，主干是模型的线性缓存。每步流程：

1. **起草。** `--spec-type` 选中的投机器（默认 `auto`，即使用检查点自带的草稿器，没有草稿器时拒绝；
   `ngram` 不需要草稿器）最多提议
   `--spec-draft` 个 token（`TS_SPEC_DRAFT`；默认 `8`）。除非显式设置窗口，
   模型偏好的窗口会取代它：Qwen 3.5 家族的递归主干、GLM-5.3-Flash 与 Qwen 3.8 Flash Next 为 3，
   ggml 后端上的 Gemma 4 为 7（Metal 上 IQ4_XS 权重更短）。n-gram 另有偏好：`ggml_metal` 上矩阵以
   IQ4_XS 为主的稠密（非 MoE）Qwen 3.5 家族递归主干（非 TP）会把它放宽到 12（13 行验证）。`--spec-pmin`（`TS_SPEC_PMIN`）
   是草稿置信度门限，默认值按算法而定：逐 token 草稿头为 `0.15`，块级草稿器为 `0.35`，
   n-gram 为 `0`。
2. **验证。** 主干用一次批量前向验证所有起草 token。该请求自己的采样器（temperature、top-k/p、
   重复/存在/频率惩罚）对每一行采样，采到的 token 与草稿一致时草稿即被接受。每个输出 token 都采自主干的
   某一行，所以投机只改变一条输出流需要的前向次数，而不改变它可能包含的 token。不过多行验证与单行 decode
   是不同的内核，因此贪心输出只在 logits 近似平局处可能与标准 decode 不同（见
   [What greedy parity delivers](speculative_decoding.md#what-greedy-parity-delivers)，英文）。
3. **回滚。** 部分接受时，超出已接受前缀的 KV（及任何递归状态）在下一步前被回滚。

三种算法共用 `SpeculativeExecution` 核心，设计见
[Speculative Decoding in TensorSharp](speculative_decoding.md)（英文）：

| 算法 | 草稿器 |
|---|---|
| `draft-head` | 逐 token 草稿头。主干 GGUF 中内嵌的 NextN / MTP（`{arch}.nextn_predict_layers`：Qwen 3.6、Qwen 3.8 27B、GLM 5.2、GLM-5.3）；通过 `--draft-model` 加载的 Gemma 4 独立 EAGLE 风格 `gemma4-assistant` GGUF，其草稿层读取**目标**最后一个 local / global 层的 KV（自身无 K/V）；通过 `--draft-model` 加载的 Qwen 3.8 Flash Next 共享 MTP 头 GGUF，它保留自己的 K/V，因此只对从位置 0 开始 prefill 的请求投机。 |
| `block` | 通过 `--draft-model` 加载的块级草稿器：DeepSeek V4 DSpark，以及用于 Muse-Glimmer 与 Qwen 3.5 家族的 DFlash / DFlash2。DeepSeek V4.1 DSpark 是实验性的：`ggml_cuda` / `ggml_cpu` 上的 `deepseek41-dspark` 草稿器，训练模型已在 `ggml_cuda` 双 GPU 按层切分下通过初步文本/图像 HTTP 检查；尚不构成通用质量或吞吐验证。 |
| `ngram` | 无需权重：在序列自身的 token 上做后缀匹配。 |

草稿被拒时，Qwen 3.5 家族主干恢复其 GatedDeltaNet 递归状态（GGML 融合验证为每一行保留快照，
`ggml_metal` 上超过 8 行的验证只保留最后三份，并在设备上提交被接受的那一份；更早的拒绝则恢复并重新前向
已接受前缀）；Gemma 4 回退注意力 KV 位置，并放回被验证覆盖的滑动窗口行（其草稿器在给定 `(token, h)`
时无状态）；GLM-5.3-Flash 恢复其 KDA 状态快照（原生执行器上为设备副本），再重新前向已接受的前缀。

投机只在模型声明有收益（`SpeculationProfitable`）时启用，这由各模型自己决定：Qwen 3.5/3.6/3.8 与
GLM 5.2 / GLM-5.3 在所有后端上；Gemma 4 在 ggml 后端（含 `ggml_cpu`）与纯 C# `cuda` 后端上；
Qwen 3.8 Flash Next 在其 GGML token 计算图路径上；GLM-5.3-Flash 在其 KDA 回滚可用时；Muse Spark 在 GGML 后端上（仅 n-gram），其原生 `--tp` 除外；DeepSeek V4 / V4.1
与 Muse-Glimmer 只在加载了各自草稿器时。Nemotron-H 直接拒绝投机（`SpeculationRefusal`）。GPT OSS、Mistral 3 与
Hunyuan Dense 没有实现投机主干，对任何算法（包括 n-gram）都走标准 decode。并发批次从不投机——当有多个
序列在运行时，每个序列都走普通的批处理 / 回退步骤。无法挂载的 `--draft-model`（草稿器不匹配或不完整，
或模型拒绝投机）会在服务端启动时立即失败（`SpeculationStartupValidation`）。`--spec` 不会关闭 Radix
前缀缓存；Qwen 3.5 家族的 NextN 头会在复用前缀之后重启自己的私有缓存（见
[Arming after a reused KV prefix](speculative_decoding.md#arming-after-a-reused-kv-prefix)，英文）。

## 模型状态

`--no-continuous-batching`（`TS_SCHED_DISABLE_BATCHED=1`）会让所有模型都走按序列路径。

| 模型家族 | 批处理 / 分页状态 | 子开关 |
|---|---|---|
| Mistral 3 | 默认 `ForwardBatch` 路径。使用分页 K/V、YaRN 感知位置、原生分页注意力，并在 prompt 准备后注入视觉 embedding。已在 Ministral-3-14B 上验证；长上下文原生分页注意力比旧按序列 GGML 路径快约 21%。 | `TS_PAGED_ATTN_KERNEL` 选择 `native`、`tensor` 或 `managed`。 |
| Gemma 4 | 密集文本负载默认走批处理路径，覆盖逐层 SWA / 全局注意力、可变 head dim、PLE、KV donor 层别名。当前回退场景包括待注入多模态 embedding、MoE 层与块量化 KV cache。已完成请求的 request-owned fused K/V holder 可被保留，用于精确前缀续接。并发（N>=2）的 decode 步骤跑 token 批量融合内核（`TSGgml_Gemma4ModelDecodeBatched`：一张图、每序列一个 token、权重只读一次），它覆盖 per-layer embedding、KV-donor 层、已回绕的 SWA 环，以及每个序列不同的缓存容量（保留前缀的克隆比新建缓存小），因此 E2B/E4B 不再回退到轮询。MoE 检查点仍走轮询，除非设置 `TS_BATCHED_FUSED_MOE=1`。可选地通过独立 `gemma4-assistant` 草稿 GGUF 做 MTP 投机解码。 | `TS_RETAINED_FUSED_CACHE_MAX=0` 关闭 retained-holder 续接。只需 `--draft-model` 即可启用投机（显式 `--no-spec` 可否决）。 |
| Qwen 3.5 / 3.6 / 3.8 family | 默认批处理路径。支持 FullAttention 层、通过每槽位状态池处理 GatedDeltaNet 递归层、MoE 变体、视觉注入与多模态 RoPE 表。其 request-owned fused holder 会把 attention K/V 与匹配的 GDN 递归状态保存在一起；正常结束的 holder 可被保留并重新绑定，用于精确前缀续接。在 `ggml_cuda` / `ggml_metal` 上，并发 decode 步骤跑 slot-stable arena 计算图（每序列一个 token，权重只读一次）；在 CUDA 上它也支持 token embedding 为 K/IQ 量化的检查点（`TSGgml_Qwen35ArenaDecodeBatchedHidden`）。Qwen 3.6 与 Qwen 3.8 27B 通过内嵌 NextN 块做投机（GDN 递归状态快照 / 回滚）；DFlash / DFlash2 草稿器与 n-gram 适用于整个家族。 | `TS_RETAINED_FUSED_CACHE_MAX=0` 关闭 retained-holder 续接；`--spec` 启用投机。 |
| GPT OSS | 默认批处理路径。支持 Q/K/V/O bias、YaRN RoPE、滑窗层、attention sinks、MXFP4 MoE expert 与原生 sinks 注意力。已与按序列路径做贪心正确性验证；性能仍主要受逐层图构建限制。在 GGML 后端上（非 TP），并发 decode 改走 per-request fused holder 与 token 批量融合 decode，在 `ggml_cuda` / `ggml_vulkan` 上使用 slot-stable arena。每个结束请求的 holder 会作为它所属会话的终态保留下来并交给下一轮，因此并行的会话各自续接自己的状态（此前，与其他会话并行运行过的会话在下一轮什么都复用不了）。不走 arena 时（Metal），token 批量 decode 的计算图上下文超过了原生上下文池的缓冲区，每个并发步骤都退回到一次只处理一个序列；现在八个并行会话用时 26.8 秒，而不是 45.2 秒。 | — |
| Nemotron-H | 默认批处理路径。Attention 层使用分页 K/V；Mamba2 层使用每槽位 conv/SSM 状态池；MoE 层使用批处理 expert 内核；准备好的图像 / 音频 embedding 可注入到批处理 hidden state。页面无法在会话中途恢复递归状态，因此结束的批处理序列的 Mamba2 slot 会和它的池块一起作为会话终态保留，并以精确长度交给下一轮：八个并行会话从什么都复用不了变为每一轮都复用上一轮，完成时间缩短 12%。 | `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1` 启用原生批处理 Mamba2 step。 |
| GLM 5.x | 没有 `ForwardBatch`：MLA 每个 token 只存一行压缩表示，DSA indexer 又要对同一段连续历史打分，没有分页 KV 布局可批。并发改由原生**序列 slot** 承担（`TSGgml_GlmSlotAlloc` / `SetActiveSlot` / `SlotFree`）——绑定请求只是切换活动 slot，不搬运 KV 字节，每个 slot 的计算图独立缓存与捕获。在此之上默认启用批量融合 decode（一张图、每序列一个 token，整批只读一遍权重）：4 个并发请求时合计 decode 提速 1.81×。批处理会改变 GEMM 形状，而 2 bit MoE 可能把这点差别放大成不同的专家选择。 | `TS_BATCHED_FUSED_DECODE=0` 关闭批量 decode。 |
| Hunyuan Dense | 默认 `ForwardBatch` 路径，使用 F32 分页缓冲。块量化（`q8_0` / `q4_0`）KV cache 会保留 KV 快照换入路径，该路径能精确处理这些 dtype。 | — |
| Muse-Glimmer | 没有 `ForwardBatch`（它不是 `IBatchedPagedModel`）：并发请求走按序列 KV 换入回退路径，把每个序列的 K/V 快照到主机内存，`--tp` 下同样如此。属于 Radix 页面家族（主机 slab 页面）。只在加载了 DFlash 草稿器（`--draft-model`）时投机。 | 没有批处理开关；`TS_MUSE_GLIMMER_*` 是内核 A/B 开关（见 [Muse-Glimmer 模型卡](models/muse-glimmer_zh-cn.md#7-环境变量)）。 |
| Muse Spark | 在 `ggml_cuda` / `ggml_vulkan` 上，原生整模型执行器用自己的**序列 slot** 处理并发（每个 slot 在各层所在 GPU 上持有 F16 K/V，`ggml_cuda` 上 `--tp N` 时则按 KV 组切分到各 GPU；46 个滑动窗口层使用环形缓冲）：所有运行中请求的 decode 步共享一张带按序列掩码的计算图，同时到达的文本提示在一张图中 prefill，图像嵌入在图内替换占位行。完成的 slot 会被保留并移交给延续该会话的请求（可先回退），模板前导与共享系统提示会被写入一个检查点 slot，再克隆进新会话。按层切分在 CUDA GPU 间流水化 prefill 时，单个 prompt 按模型选择的步长 prefill（8,192 token；见下文 `TS_SCHED_SOLO_PREFILL_CHUNK`）。逐算子路径（`TS_MUSE_SPARK_NATIVE=0`，以及 `ggml_vulkan` 上的 `--tp`）属于 Radix 页面家族；它的打包分页 `ForwardBatch` 需要 F32 KV 且只接受纯文本请求。 | `TS_MUSE_SPARK_NATIVE=0` 选择逐算子路径；`TS_MUSE_SPARK_PLAN_SLOTS` 在放置规划中预留 slot；`TS_MUSE_SPARK_PIPELINE=0` 关闭流水化 prefill（见 [Muse Spark 卡片](models/muse-spark_zh-cn.md#选项)）。 |
| DeepSeek V4 / V4.1 | 没有 `ForwardBatch`：压缩注意力缓存没有分页布局。并发由原生执行器的序列 slot 承担（纯 C# `cpu` 与直连 CUDA `cuda` 执行器保持串行）。默认启用的 token 批量融合 decode 每步只读一遍权重，最多 16 个序列，更多时分窗口执行；加载了 DSpark 草稿器时不启用。V4.1 会把已结束会话的 slot 保留给它的下一轮（加载了 DSpark 草稿器时不保留）。 | `TS_BATCHED_FUSED_DECODE=0`；V4.1 保留 slot 的预算由 `TS_DSV41_RETAINED_CACHE_MB`（默认 2048）决定。 |
| Qwen 3.8 Flash Next（`qwen4exp`） | 没有 `ForwardBatch`：并发通过 GGML 融合 span 路径上的按序列状态 holder 实现，逐个序列 decode（没有 token 批量融合 decode）。已结束的会话会被保留，共享提示前缀会被做成检查点并克隆到新会话中，仅限精确前缀。`--layer-split N` 是按层切分（qwen4exp 没有张量并行模式），保留与检查点在按层切分下都可用。共享 MTP 头（`--draft-model`）只对从位置 0 开始 prefill 的单序列请求投机。 | `TS_Q4E_RETAINED_CACHE_MB` 是保留与检查点共用的预算（默认取实测内存余量的一半；`0` 不保留任何状态）。 |
| DiffusionGemma | 独立文本扩散路径。`Forward(int[] tokens)` 刻意不支持；生成会迭代去噪固定长度 canvas block。Web UI 请求共享 `DiffusionBatchScheduler`，在 block 之间接纳并发请求，并可选择批处理活跃 canvas。 | `DIFFUSION_STEPS`、`DIFFUSION_MAX_BATCH`、`DIFFUSION_BATCHED_FORWARD`；`DIFFUSION_NO_FUSED_DECODE=1` 关闭 GGML 融合整模型 diffusion decode。 |

### Radix 前缀缓存

跨请求的提示复用由一个索引统一负责，即 Radix 树（`TensorSharp.Runtime/Scheduling/PrefixCache/` 中的
`PrefixTree`，由引擎线程上的 `PrefixCacheCoordinator` 驱动）。它服务于服务端、CLI 生成与 TensorAgent。
请求的 radix 键是它的 prompt token，每个媒体区间以其内容身份为键；每个
节点记录请求可以从这里续接的状态：

- **页面**：主机 slab 上的 KV 快照或模型自己的分页块，用于没有续接 holder 的家族（GPT OSS、
  Mistral 3、Hunyuan Dense、Muse-Glimmer、Nemotron-H）。
- **终态（end state）**：模型自己持有的续接状态。Gemma 4、Qwen 3.5 / 3.6 / 3.8 家族（`qwen35`、
  `qwen35moe`、`qwen3next`）与 Qwen 3.8 Flash Next 可以复制它，因此已结束的会话会被保留给下一轮，
  共享公开前缀末尾的状态会被做成检查点并克隆到每个新会话中。前提是它们运行按请求的 fused holder：
  GGML 后端上的 Gemma 4、`ggml_cuda` / `ggml_metal` 上且非 TP 的 Qwen 3.5 家族、GGML token-span 路径上的
  Flash Next，且 `TS_PER_SEQ_FUSED=0` 时一律没有；其他情况下只续接主缓存。
  GLM 5.x（原生执行器）与 DeepSeek V4.1 无法复制原生 slot：只能把它整体交给延续该会话的那一轮（只在
  `TS_RETAINED_FUSED_CACHE_MAX` 不为 0 时；V4.1 在加载了 DSpark 草稿器时不交出），也不做共享前缀检查点。

预算沿用已有的开关：`TS_PREFIX_CHECKPOINTS_MAX` 约束公共检查点，`TS_RETAINED_FUSED_CACHE_MAX` 约束保留的
按会话终态（未设置时取调度器的运行中请求上限且至少为 4，使每个并行运行的请求都保有自己会话的状态；
字节数由显存而不是这个数量约束），家族自己的字节上限（`TS_Q4E_RETAINED_CACHE_MB`、`TS_DSV41_RETAINED_CACHE_MB`）仍然生效。复用要求
渲染出的 token 完全相同（包括工具 schema、聊天模板与思考设置），止于每个模型可续接的边界与显式的缓存断点，
并且总会留下至少一个 prompt token 去计算。下文的会话作用域规则照常适用，媒体身份也一样（区间以内容为键，
复用长度不会切断区间）。复用能否越过媒体区间取决于各家族的前缀缓存能力：Gemma 4、Qwen 3.5 / 3.6 / 3.8
家族与 GPT OSS 可以越过；其他树家族（Mistral 3、Nemotron-H、Muse-Glimmer、Hunyuan Dense、
Qwen 3.8 Flash Next、GLM 5.x、DeepSeek V4 / V4.1）的复用止于第一张图片、视频帧或音频片段。DiffusionGemma
与图像/视频模型不使用它。

`TS_SCHED_PREFIX_CACHE=0` 关闭复用；`--no-prefix-cache`（CLI 与服务端）会设置它，并同时跳过共享 system /
工具前缀的启动预热，以及服务端的检查点文件。`--spec` 不会关闭前缀缓存。（Radix 之前的块哈希共享及其各自独立的
复用路径已被移除；设置 `TS_PREFIX_CACHE_MODE` 会报错。）

### 保留的终态

模型可能无法从按字节保存的分页快照重建完整续接状态，但仍可拥有自包含的 per-request fused holder。
模型声明 `SupportsRetainedFusedCache` 后，正常结束的 holder 会成为其会话在 Radix 树中的终态；同一会话的
后续请求精确扩展已记录的 token 前缀时，再把它重新绑定给新请求。Gemma 4 保留其环形 attention K/V；
Qwen 3.5/3.6/3.8 则把 attention K/V 与匹配的 GatedDeltaNet 递归状态作为一个混合 holder 一起保留；
Qwen 3.8 Flash Next 保留其整个按序列 holder（预算为 `TS_Q4E_RETAINED_CACHE_MB`）；DeepSeek V4.1 保留原生
slot（加载了 DSpark 草稿器时不保留）。`TS_RETAINED_FUSED_CACHE_MAX=0` 关闭保留，也包括 GLM 5.x 交出的原生 slot
（原生执行器，一个 slot）。带作用域的会话在主（N=1）缓存上结束的请求，在 fused 步骤接管模型时也会以同样方式
保留，因此不会因为另一个会话插在它两轮之间到达而丢失自己的状态。

### 跨请求的提示复用：会话作用域与媒体身份

所有跨请求复用——live 主缓存、保留的终态、共享前缀检查点和页面——都遵守两条规则。

**会话作用域。** 每个 `SequenceState` 携带一个 `CacheScope`（不透明的哈希）以及它的公开边界
`SharedPrefixTokens`（开头的 system/developer 消息加工具声明）。其他作用域产生的状态只能复用到
这个公开前缀为止：通过被克隆的共享前缀检查点，或者把 live cache 回退到恰好这个前缀（这是没有检查点的
模型——例如 DeepSeek V4.1——唯一的公开复用；新请求的预填充本来也会覆盖这个缓存，存在检查点时仍优先使用
检查点）。绝不会采纳、回退进入或移走另一个会话的保留 holder，绝不会越过公开前缀续接它的 live cache，
公开前缀之后的页面属于其作用域（Radix 树把作用域记在节点上）。带作用域的
请求也不会克隆比自己公开前缀更长的检查点。作用域由 chat 层给出：

| 请求 | 作用域 |
|---|---|
| 带 `sessionId` 的 Web UI / TensorAgent | 会话及其新会话纪元（`newChat:true` 开始一个新纪元）；把会话绑定到已保存对话的宿主（`WebUiChatService.BindSessionConversation`，TensorAgent 为它打开的每个会话都会调用）改用该对话，因此重新打开一个聊天会延续它自己的缓存状态 |
| OpenAI Chat / Responses、Ollama chat、不带 `sessionId` 的 Web UI | 请求历史证明自己所延续的会话：它最后一条 assistant 消息是本服务器生成并只发给该会话的回合（见下文）；否则（包括两个会话在相同历史之后收到了相同回合的情况，例如对常见开场白的贪心回复）是一个全新的作用域 |
| Skills / 代码工具循环的各轮 | 启动该循环的客户端回合的作用域 |
| CLI | 每个会话一个作用域；`/new` 与 `/reset` 开始新的作用域，每个 JSONL 会话也各有自己的作用域 |
| 不设置作用域的引擎调用方（基准测试） | 每个请求一个全新作用域，因此只共享声明的公开前缀 |

chat 层的原始 token 拼接遵循同样的身份。每个生成的回合都以其之前的客户端可见历史的内容哈希链
（角色、内容、工具调用、按内容计的媒体与附件文件）为键记录下来，同时记录原始输出 token 以及当时发给客户端
的内容（解析后的正文和工具调用，或原始文本）。之后的 assistant 消息只有在（忽略空白后）等于这份
已发出的内容时才会用记录的 token 渲染；客户端自己编写或修改过的 assistant 消息按其自身文本渲染。
在此之前，无状态 API 共享同一份跟踪历史，会把另一个客户端生成的回合拼接到本客户端自己的消息上。
并发的会话也不再相互覆盖记录。

对无状态请求而言这是"以内容为证"，存在一个残余风险：一个请求只要重现了某会话的**任意**一个较早的
生成回合（不只是最新一个），就会延续该会话的作用域，包括其后续回合留下的状态；而确定性（贪心）
回复可以在本服务器之外复现。这样的请求只会复用它自己发来的 token，但 `cached_tokens` 会反映其提示
与该会话后续回合匹配到多远：页面家族上是整页，Gemma 4 上是 holder 末尾的少数 token，
在支持精确原生回退的模型（DeepSeek V4.1）上更远。需要严格隔离的客户端应使用 Web UI / TensorAgent
的 `sessionId`；无状态 API 目前还没有按请求的缓存键。

**媒体身份。** 每张图片、视频帧（对）和音频片段都以其字节的 SHA-256 标识。Base64 附件（OpenAI
`image_url`、Responses `input_image`、Ollama `images`、音频）以 `<sha256>.<ext>` 存储且只写一次，
因此客户端每轮重发同一张图片只保留一个文件。视觉与音频嵌入缓存以该内容 id 为键，受
`TS_MM_EMBEDDING_CACHE_MB` 约束并按最近最少使用淘汰，已准备好的提示仍引用的条目不会被淘汰。
请求以位置区间的形式携带其媒体（`SequenceState.MediaSpans`）；当缓存前缀内的每个区间都是同一
位置上的相同内容时，该前缀可以复用，复用长度会被截到它将切断的任何区间的起点。因此第一张图片
之前的文本总是可以复用：radix 键在每个区间的位置上记录其内容 id，因此只差一张图片的两个提示会共享
图片之前的全部内容。

无法精确越过媒体续接缓存的模型声明 `SupportsReuseAcrossMediaSpan = false`，此时复用止于第一个媒体区间。
Radix 树从各家族的前缀缓存能力中读取这条规则，只有 Gemma 4、Qwen 3.5/3.6/3.8 家族与 GPT OSS 能越过媒体区间续接（见
[Radix 前缀缓存](#radix-前缀缓存)）。Gemma 4 使用绝对位置。Qwen 3.5/3.6 的 M-RoPE 提示位置在图片
之后被压缩，位置表之外的每个 token（decode、投机 verify、文本续接）都按其 KV 下标加上该序列的
M-RoPE 偏移（delta）旋转，而每个 holder、检查点和检查点文件（格式版本 2）都保存这个 delta；因此后续
回合越过图片续接缓存，并与重新 prefill 一致（仅差后端 decode 与 prefill 内核之间的数值差异；见 [Qwen 3.5 模型卡](models/qwen35_zh-cn.md)：在 Metal
上，图片之后的 Web UI 回合复用 98% 的提示，首 token 用时 0.13 s，而不是约 1.1 s）。在此修复之前
Qwen 3.5/3.6 声明为 `false`，因为 decode 使用绝对下标。

在复用前缀*之后*预填充图片是另一回事。无法精确做到这一点的模型让
`IModelArchitecture.CanPrefillMediaAfterReusedPrefix` 返回 false，这样的回合便不复用公共前缀之后的内容。
公共前缀本身仍从共享前缀检查点克隆：启用检查点时每次 prefill 都会在该边界切分，所以无论是否复用，媒体
都在它之后预填充；没有公共前缀的回合从零 prefill。目前发布的模型都不返回 false。Gemma 4 过去在超出
滑动窗口时返回 false，直到它的融合 prefill 能在任意起始位置应用图片的双向掩码、逐算子路径能把掩码
映射到已回绕的窗口上；现在图片回合会复用会话的文本，并在融合图上预填充图片（E4B/Metal：一个复用
179 token 的 457 token 图片回合首 token 用时 0.57 s，逐算子路径为 1.25 s，冷启动为 0.64 s；一个超出窗口的
889 token 回合复用 611 token，首 token 0.62 s，不复用时为 0.85 到 0.90 s）。详见
[Gemma 4 模型卡](models/gemma4_zh-cn.md#复用前缀之后的图片与音频回合)。

在 Gemma 4 上，不超过 `MaxReusablePrefixTokens`（滑动窗口）个 token 的回合也会续接 live cache。
已回绕环上的回退依旧被拒绝。

准入日志会写明服务该请求的来源——`radix pages`、`radix end state`、`radix batched end state` 或
`radix primary cache`——并带上 token 数；请求什么都没有复用时，写出某个来源拒绝的第一个具体原因。

## 测试覆盖

| 范围 | 测试 |
|---|---|
| 调度器 / 块池 | `ContinuousBatchSchedulerTests`、`PagedKvCacheTests`、`PagedKvCacheCodecTests` |
| Radix 前缀缓存 | `InferenceWeb.Tests/PrefixCache/`：`PrefixTree*Tests`、`RadixKeyTests`、`ResumabilityRulesTests`、`PrefixCacheContractConformanceTests` 与 `PrefixCacheFamilyCoverageTests`（按家族的契约）、`RadixPagedEngineTests` / `RadixHolderEngineTests`（引擎集成）、`TreeTraceHarnessTests`（`Category=PrefixCacheProperty`）；`PrefixCacheStartupTests`；受模型门控的 `Qwen35MtpPrefixCacheTests`（复用前缀之后的 NextN 投机） |
| 批处理执行原语 | `BatchedExecutorTests`，覆盖托管分页注意力正确性与多序列 logits 路由；`InjectShortfallAccountingTests`（页面注入提前停止即视为未命中，绝不留下部分状态） |
| 跨请求隔离与媒体身份 | `ModelServiceRawTokenHistoryTests` 与 `ToolTranscriptSpliceTests`（按内容校验的原始 token 拼接）、`ContentAddressedMediaTests`；`Gemma4MediaAfterReusedPrefixExactnessTests`（受模型门控：复用前缀之后的图片或音频回合对比冷启动 prefill）与 `Gemma4SoftTokenMaskTests` |
| 越过媒体复用（Qwen 3.5 M-RoPE） | `Qwen35MRopeReferencePositionTests`（与 SGLang `get_rope_index` 夹具比较位置），需显式启用的 `Qwen35ImageFollowUpExactnessTests`（真实权重下图片之后复用与冷启动对比，单请求与并发，检查点文件往返） |
| 按模型正确性 | `Qwen35BatchedCorrectnessTests`、`Mistral3BatchedForwardTests`、`Gemma4BatchedForwardTests`、`GptOssBatchedCorrectnessTests`、`NemotronBatchedCorrectnessTests` |
| 后端融合下的批处理 MoE 内核 | 原生 CTest `moe-fused-bias-alias-cpu` / `moe-fused-bias-alias-cuda` / `moe-fused-bias-alias-metal`（`GgmlOpsMoeFusedBiasAliasTest`）：带每专家 bias 的独立 MoE FFN 内核与精确主机计算结果对比，1、4、7 个 token，重复执行 |
| 投机解码 | `SpeculativeExecutionTests`（起草 / 验证 / 回滚核心）、`SpeculatorRegistryTests`、`NGramSpeculatorTests`、`SpeculationCostGovernorTests`、`NemotronSpeculationRefusalTests`、可选端到端 `Qwen36SpeculativeTests`（`TS_TEST_QWEN36_SPEC=1`）与 `Gemma4SpeculativeTests`（`TS_GMTP_E2E=1`），需真实 GGUF |
| 按模型性能探针 | `Gemma4BatchedPerfBench`、`Qwen35BatchedPerfBench`、`GptOssBatchedPerfBench`、`NemotronBatchedPerfBench` |
| DiffusionGemma 路径 | `DiffusionGemmaTests` 覆盖去噪、prompt-KV 缓存与批处理生成探针 |
| 端到端引擎行为 | 通过 `TS_TEST_MODEL_DIR` 指向真实 GGUF 后运行的 `EngineParallelInferenceTests` |
| 服务端参数翻译 | `ServerOptionsBuilderTests` 覆盖 `--continuous-batching`、`--no-continuous-batching` 与 `--no-prefix-cache`（以及 `--spec` 保持 Radix 开启） |

## 配置

| 变量 | 默认 | 作用 |
|---|---|---|
| `TS_SCHED_DISABLE_BATCHED` | `0` | 设为 `1` 后，即使模型实现了 `IBatchedPagedModel` 也强制按序列 KV-swap 回退。 |
| `TS_SCHED_MAX_BATCHED_TOKENS` | `4096` | 每步 token 预算。 |
| `TS_SCHED_MAX_RUNNING_SEQS` | `16` | 最大同时执行序列数。 |
| `TS_SCHED_PREFILL_CHUNK` | `256` | 存在活跃 decode 时每请求的 prefill 上限；仅 prefill 的步骤会公平分配完整 token 预算。服务端参数：`--prefill-chunk-size N`。 |
| `TS_SCHED_SOLO_PREFILL_CHUNK` | `8192` | solo（无争用）请求的每步 prefill 上限——以大分块把 prompt 送入融合整图 prefill 路径。受 `TS_SCHED_MAX_BATCHED_TOKENS` 约束。有自己单请求步长的模型（`ISoloPrefillBudgetModel`）——启用流水化 prefill 的 Muse Spark：8192 token，不超过 2048 的余量并入前一步——在未设置该变量时自行决定步长，且步长可超过 `TS_SCHED_MAX_BATCHED_TOKENS`；设置了该变量则按此处所述生效，并关闭模型自己的步长。只设置 `TS_SCHED_MAX_BATCHED_TOKENS` 不会关闭它。引擎日志会说明当前生效的是哪条规则。 |
| `TS_SCHED_NUM_BLOCKS` | `256` | 引擎块池物理块数。 |
| `TS_SCHED_BLOCK_SIZE` | 只靠页面复用的家族（Hunyuan Dense、Mistral 3、Muse-Glimmer）为 `16`，其余为 `256` | 每块 token 数。只靠整页做跨请求复用的家族按这个粒度复用：块大小为 256 时，短于一块的一轮什么都复用不了。 |
| `TS_SCHED_PREFIX_CACHE` | `1` | 设为 `0` 关闭准入时的全部提示复用：Radix 的页面、终态与检查点。`--no-prefix-cache`（CLI 与服务端）会设置它。 |
| `TS_SCHED_STOP_REPETITION` | `1` | 设为 `0` 时，陷入重复循环的生成会继续跑到 token 上限，而不是以 `repetition` 结束原因停止；重复模板行的思考也不再被闭合。 |
| `TS_SCHED_DECODE_QUANTUM` | `256` | 在偏回退路径中，允许切换序列前的 decode token 数。 |
| `TS_PER_SEQ_FUSED` | `1` | fused 能力模型上的并发（N≥2）序列走 per-request 融合 Forward，每个序列有自己的设备端 K/V holder；设为 `0` 改走逐算子批处理分页路径，其 K/V 位于共享的主机块池（设备内存更少、decode 速率更低、没有保留的 holder）。 |
| `TS_BATCHED_FUSED_DECODE` | `1` | `0` 在 per-seq fused 路径内关闭真正的 token 批量融合 decode（一张图同时 decode 全部 N 个序列）。 |
| `TS_BATCHED_FUSED_MOE` | `0` | `1` 允许 Gemma 4 MoE 检查点走 token 批量融合 decode；默认关闭，因为在实测的 16 GB 显卡上它并不比轮询快。 |
| `TS_RETAINED_FUSED_CACHE_MAX` | 运行中请求上限，至少 `4` | 对声明支持保留 holder 的模型，为下一轮保留多少个已结束会话的终态：Gemma 4 K/V、Qwen 3.5/3.6/3.8 的 attention K/V 与 GDN 递归状态、Qwen 3.8 Flash Next 的按序列 holder、DeepSeek V4.1 的原生 slot，以及 GLM 5.x 交出的原生 slot。每个都会占用模型完整的 per-request 续接状态。未设置时跟随 `TS_SCHED_MAX_RUNNING_SEQS`，使每个并行运行的请求都保有自己的终态；`0` 关闭保留。 |
| `TS_PREFIX_CHECKPOINTS_MAX` | `4` | Radix 树在共享提示前缀结束处（由 chat 层在请求上标记的边界）保留多少个模型完整状态的检查点；每个新会话从其中一个的副本开始。适用于能复制自身状态的模型（GGML 后端上的 Gemma 4；`ggml_cuda` / `ggml_metal` 上的 Qwen 3.5/3.6/3.8，非 TP；GGML token-span 路径上的 Qwen 3.8 Flash Next，包括 `--layer-split N` 按层切分时）。一个提示在每个声明的边界（系统指令末尾与整个共享前缀末尾）各发布一个检查点，因此 4 可覆盖同时预热两种思考模式的主机；`0` 关闭检查点。 |
| `TS_MM_EMBEDDING_CACHE_MB` | 主机内存的 1/64，512-4096 MB | 视觉/音频嵌入缓存的字节预算，缓存以媒体内容（SHA-256）为键；超出后淘汰没有被已准备提示引用的最近最少使用条目。 |
| `TS_KV_INITIAL_TOKENS` | `0` | 缓存创建时、任何请求声明预算之前分配的 K/V token 数；`0` 沿用引擎策略（显式 `MAX_CONTEXT` 时为整个窗口）。缓存仍按需增长。 |
| `TS_KV_GENERATION_RESERVE_MAX` | `0` | 请求预先保留的 K/V（prompt + max_new_tokens）中生成部分的上限；`0` = 不限制。超过上限后缓存按需增长。 |
| `TS_KV_HOLDER_POOL_MAX` | `64` | 模型最多可停放多少个已释放的 per-request holder 以待复用；停放期间每个都占用其完整 K/V 分配。 |
| `TS_SPEC` | `0` | `1` 为单序列启用投机解码（`--spec`）。 |
| `TS_SPEC_TYPE` | `auto` | 投机算法：`auto`、`draft-head`、`block` 或 `ngram`（`--spec-type`）。 |
| `TS_SPEC_DRAFT` | `8` | 每个投机步最多起草的 token 数，1-64（`--spec-draft`）；模型偏好的窗口会取代默认值（通常更窄；`ggml_metal` 上以 IQ4_XS 为主的稠密 Qwen 3.5 家族主干在 n-gram 下为 12）。 |
| `TS_SPEC_PMIN` | 按算法 | 草稿置信度门限（`--spec-pmin`）：逐 token 草稿头为 `0.15`，块级草稿器为 `0.35`，n-gram 为 `0`；`0` 表示从不设阈。 |
| `TS_SPEC_DRAFT_MODEL` | 无 | 以独立 GGUF 发布的草稿器路径（`--draft-model`）：Gemma 4 的 `gemma4-assistant`、Qwen 3.8 Flash Next 的共享 MTP 头、DSpark 或 DFlash 草稿器。架构从文件中读取。 |
| `DIFFUSION_STEPS` | `48` | Web UI DiffusionGemma 每个 block 的去噪步数；与自回归调度器的 step 预算无关。 |
| `DIFFUSION_MAX_BATCH` | `2` | diffusion scheduler 中同时活跃的 DiffusionGemma Web UI 请求数上限。 |
| `DIFFUSION_BATCHED_FORWARD` | `0` | 对活跃 DiffusionGemma canvas 启用真正的批处理 decode；默认更偏向融合单 canvas 路径。 |

宿主可以给引擎挂上一个 `IPrefixCheckpointStore`（`InferenceEngine.PrefixCheckpointStore`，
或 TensorSharp.Chat 中的 `InferenceEngineHost.PrefixCheckpointStore`），让共享前缀检查点
在进程之外保留：执行器在接纳一个共享前缀尚无内存中检查点的请求时读取已保存的检查点，并在
取得检查点的那一刻写入一份；字节格式由模型家族自己负责（`IBatchedPagedModel.TryExportRetainedCache`
/ `TryImportRetainedCache`，Qwen 3.5/3.6 与 Gemma 4 已实现）并在读入时校验。`PrefixCheckpointFileStore`
（`TensorSharp.Runtime.Scheduling`）是基于文件的实现，每个模型最多保留两个文件，它让启动后的第一条消息和第二条
一样快。服务端为启动模型挂上它，文件放在二进制旁的 `prefix-cache/<模型文件名>-<哈希>/` 下（可用
`TENSORSHARP_PREFIX_CACHE_DIR` 移走，用 `--no-prefix-cache` 关闭），并在开始接受请求之前预热共享提示；
TensorAgent 为每个模型在应用缓存目录下挂一个。CLI 只在内存中保留检查点。


服务端 CLI 别名：

```bash
--continuous-batching      # 默认，设置 TS_SCHED_DISABLE_BATCHED=0
--no-continuous-batching   # 设置 TS_SCHED_DISABLE_BATCHED=1
--prefill-chunk-size N     # 设置 TS_SCHED_PREFILL_CHUNK
```

请求 KV 状态由 `InferenceEngine` 持有。KV cache 可以用
`--kv-cache-dtype <f32|f16|q8_0|q4_0>`（或 `KV_CACHE_DTYPE` 环境变量）以更低精度存储，
CLI 与服务端均支持。默认按模型自动选择（模型权重低于 F32 时为 `f16`，否则为 `f32`）；
块量化档位（`q8_0`、`q4_0`）要求原生 GGML
flash-attention 路径，`q4_0`（约为 f32 的 1/7）面向 KV cache 主导内存占用的
128K–256K 超长上下文。Qwen3.8-Flash-Next 的直接引擎（`--backend cuda`）有自己的 `q8_0`
注意力内核，默认使用 `q8_0`，`q4_0` 时保持 `f16`（见 `docs/models/qwen38-flash-next_zh-cn.md`）。

## 后续工作

- 为整批注意力构建一个原生 GGML 图，而不是每个序列一个小图。这会降低大量短序列场景下的 launch / compile 开销。
- 把分页 K/V 池移到设备上：一步的 K/V 用 `ggml_set_rows` 写入，每个序列的历史在注意力图内
  用 `ggml_get_rows` 聚合，取代主机聚合。在那之前，批处理路径仍然保持主机驻留的 cache 与实测上限。
  这是通用 `BatchedPaged` 路径（没有 token 批量融合 decode 的模型）在并发下提升合计吞吐的前提。
- 补齐 Gemma 4 对 MoE 变体、多模态待注入 embedding、块量化 KV cache 的批处理覆盖。
- 根据实际运维需要，决定是否把 DiffusionGemma scheduler 指标接入
  `/api/queue/status` 或单独的 diffusion 端点。
- 将准备好的多模态 embedding 列表从模型级可变状态迁移到 `SequenceState`，使多模态 prompt 准备也能完全并行，而不是提交前串行化。
