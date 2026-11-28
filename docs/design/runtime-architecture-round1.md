# 统一运行时与 Tensor Executors 架构改进：首轮实施记录

**日期：** 2026-11-28
**基线：** `main` @ `75e17d9`（含 2026-11-28 合并的多 GPU layer split / TP 更新）
**分支：** `claude/sleepy-heisenberg-pqokd1`
**对应计划：** 《TensorSharp 统一运行时与 Tensor Executors 架构改进设计及实施计划》（2026-11-28），D01–D09

本文记录该计划首轮的实际实施内容、每项设计（D01–D09）的结论（完成 / 部分完成 / 延期）与理由，以及验证覆盖范围和限制。
计划第八节建议的首轮交付是 D01、D03、D04；本轮完成了这三项，并完成了 D02 的资源所有者层及其首批迁移。其余各项给出延期理由和后续入口。

---

## 1. 结论一览

| 设计 | 优先级 | 结论 | 本轮交付 |
|---|---|---|---|
| D01 统一执行契约独立成立 | P0 | **完成** | `ISequenceStateStore`；`EngineServability`；统一 step 的状态绑定（契约绑定 + 旧接口适配器）；按契约释放；V4.1 native slot 由声明决定；`ExecutionPlanner.PlanAdmission` |
| D02 资源生命周期由所有者管理 | P0/P1 | **部分完成** | `DeviceResidency` / `ResidentTensorLease`；CUDA/MLX 注册常驻权重释放器；`NetworkWeights`、LFM3（depthformer、codec decoder、audio encoder）、LTX 2.5、MiniMax-H3 的释放迁移 |
| D03 Graph 协议单一来源 | P0 | **完成** | `graph-ops.json` + `eng/graph-schema` 生成器；C#/C++ 三份定义生成；建图期结构校验；ABI hash 握手；slot 常量与 typed emitter |
| D04 配置作用域与生效时机 | P0 | **完成（核心路径）** | step 级 `ExecutionOptionsSource`；模型级 `MoeCpuOffloadPolicy`、`ModelPlacementOptions`、holder pool 快照；进程级 `ProcessSettings` 冲突报告；CLI/Server 不再通过环境变量传递 TP/layer split |
| D05 BackendSession | P1 | 延期 | 进程级约束已由 `ProcessSettings` 显式化（GGML backend 类型、MLX TF32） |
| D06 共享 decoder 组件化 / 权重描述 | P1 | 延期 | — |
| D07 数学语义与后端策略分离 | P1 | 延期 | — |
| D08 TP×PP 原生子通信组 | P1 | 延期 | — |
| D09 多模态扩展边界 | P1/P2 | 延期 | — |

---

## 2. D03：Graph 协议单一来源（完成）

**问题。** `GraphOp`（Core）、`GgmlGraphOp`（GGML managed bridge）、`TsgGraphOp`（native）三处手工同步；参数位置（`ip[3]`、`fp[2]`）在 managed executor、native builder 和各家族 emitter 中各自解释；参数缺失只在 executor 读越界时暴露。

**实现。**

- 唯一定义：[`TensorSharp.Core/Graph/Schema/graph-ops.json`](../../TensorSharp.Core/Graph/Schema/graph-ops.json)。每个 op 记录 code、名称、native 名称、操作数列表（可选项以 `?` 标记、可变尾部给出上限）、`ip`/`fp` 槽位（`Name=default`）及必需个数，以及文档。
- 生成器：[`eng/graph-schema`](../../eng/graph-schema/README.md)（`dotnet run --project eng/graph-schema`，`-- --check` 只检查）。生成：
  - `TensorSharp.Core/Graph/GraphOp.g.cs`：`GraphOp`、`GraphOpSchema`（结构表、`Check`/`Validate`、`Hash`、`Version`）、`GraphOpParams`（槽位常量与默认值）、`GraphOpEmit`（`IGraphOpEmitter` 上的 typed emitter）。
  - `TensorSharp.Backends.GGML/GgmlGraphOp.g.cs`：`GgmlGraphOp`。
  - `TensorSharp.GGML.Native/tsg_graph_ops.g.h`：`TsgGraphOp`、`signature()`/`check()`、`kHash`/`kVersion`、`slots::<op>::IP_*`/`FP_*`/`SRC_*`。
- 结构校验在建图时进行：`GraphProgramBuilder.Op`（记录与 hash-only 两种模式）和 native `TSGgml_GraphOp` 都按同一张表拒绝操作数或参数个数不符的 op。运行时表示仍是紧凑的整数/浮点数组，没有新增每 token 的对象分配或动态派发（每个 op 一次数组下标查表）。
- ABI 握手：native 导出 `TSGgml_GraphSchemaHash`/`TSGgml_GraphSchemaVersion`；GGML backend 初始化（`GgmlNative.EnsureAvailable`）和每个 `GgmlGraph` 构造前比较 managed/native 的 hash，不一致或旧库缺少该导出时报告需要重新构建 `TensorSharp.GGML.Native`。两个新导出已加入 TensorAgent Release 链接清单。
- 迁移：native builder 与 `TensorGraphExecutor` 中 RoPE、RoPE multi、flash attention、view、KV window 的参数读取改用槽位常量；共享 decoder 的 RoPE/KV window emitter 与 `GraphEmitter.FlashAttention` 改用 typed emitter，产生的 `ip`/`fp` 与原手写版本逐字节相同（签名不变，计划缓存不受影响）。

**验证。** `GraphOpSchemaTests`：生成文件与 schema 一致（过期即失败）、hash 只覆盖结构不覆盖文档、schema 自身的一致性检查、建图期拒绝、native 拒绝、typed emitter 与手写 op 的 program 与签名逐字节一致、native/managed 握手。全量 CPU 测试与 A40 实模型 A/B 见第 8 节。

---

## 3. D01：统一执行契约独立成立（完成）

**问题。** 只实现 `IUnifiedBatchModel` 的模型无法被 host 接入（`InferenceEngineHost` 只接受 snapshot 或 `IBatchedPagedModel`）；统一 step 中把模型转换为 `IBatchedPagedModel` 去绑定 holder、`RestorePrimaryCache()`；请求状态通过旧的 `OnSequenceReleased` 释放；`StandardDecoderModel.SupportsUnifiedBatching` 沿用旧能力；`PrefixCacheCoordinator.PredictRoute()` 重新读配置并检查接口；InferenceEngine 用架构名 `deepseek41` 识别 native slot。

**实现。**

| 契约 | 职责 | 位置 |
|---|---|---|
| 执行能力 | `IUnifiedBatchModel`：batched forward、greedy/row IO、media | `TensorSharp.Runtime/Scheduling/IUnifiedBatchModel.cs` |
| 序列状态存储 | `ISequenceStateStore`：`EnsureSequenceSlot`（创建/预留/增长）、`HasSequenceState`、`TryRewindSequenceState`、`ReleaseSequenceState`、`StateStorage`（slots / native slots） | `ISequenceStateStore.cs`；`IUnifiedBatchModel : ISequenceStateStore` |
| 状态复用能力（可选） | `IPrefixCacheModel`（copy/donate/materialize/checkpoint） | 既有 |
| 推测执行能力（可选） | `IBatchedSpeculativeTarget` | 既有 |

- **Host 接入**：`EngineServability.ContractOf/CanServe/RefusalReason` 是所有 host 共用的判断（Unified、BatchedPaged、KvStateSnapshot、None）；`InferenceEngineHost.TryGetEngine` 与 `ChatGenerationPipeline` 的报错改用它。
- **统一 step 的状态绑定**（`BatchExecutor.UnifiedState.cs`）：执行器构造后一次性选定绑定方式——
  - `StoreStateBinding`（只实现新契约的模型）：计划中的回退通过 `TryRewindSequenceState`，回退失败则释放状态并从头重算；没有状态的 live-cache 声明被撤销；`EnsureSequenceSlot` 返回的 token 数少于已计算数时，在该请求自己的 slot 上重新 forward 丢失的尾部（`RecomputeLostTailUnified`），多于时回退。
  - `LegacyHolderBinding`（仍实现 `IBatchedPagedModel` 的模型，即共享 decoder）：保持原 bind → inject（paged block 前缀注入）→ restore 顺序的适配器。
  统一分支本身不再做 `IBatchedPagedModel` 类型转换。
- **释放**：`InferenceEngine` 在模型有状态存储时通过 `ReleaseSequenceState` 释放，否则走旧 hook。
- **共享 decoder**：`SupportsUnifiedBatching` 成为独立能力（`_cacheK != null && LayerSplitGroup == null`），旧的 `SupportsPerSequenceFusedForward` 由它派生；在 slots 上实现状态存储成员。
- **native slot**：`DeepSeek41Model` 声明 `UsesNativeSequenceSlots`；引擎读取声明（或 `ISequenceStateStore.StateStorage == NativeSlots`），不再比较架构名。物理 KV 存储与调度预留预算的区别写入 `SequenceStateStorage` 的文档。
- **Admission plan 与 step plan**：`ExecutionPlanner.PlanAdmission` 是唯一的准入路由决策（slot 或 primary、可恢复哪类 end state、miss 时是否改走 batched paged 路由）；`PrefixCacheCoordinator` 的路由预测、构造期 end-state 路由、miss 后重规划和持有状态判断都消费它。step plan 仍按每步的 batch 成员和动态条件变化。

**验证。** `UnifiedContractStandaloneTests`：只实现统一契约（可选 `IPrefixCacheModel`）的 oracle 模型通过提交、并发 batched forward（多个请求同一 forward）、精确前缀复用（与冷启动 greedy 输出一致）、取消、forward 失败、首步前状态丢失后的尾部重算和释放；并通过 `InferenceEngineHost` 创建引擎并运行请求（基线下 host 会拒绝该模型）。`AdmissionPlanTests` 检查所有能力 × 开关组合下，step plan 读取状态的路由与 admission 放置的一致。旧模型经适配器运行：全量测试与 A40 实模型（共享 decoder 的 Gemma 4、Qwen 3.5/3.6）见第 8 节。

**保留。** `IBatchedPagedModel` 仍有消费者（DeepSeek V4/V4.1、GLM-DSA、MiniMax-M3、Muse Spark 以及共享 decoder 的 paged-block 前缀注入），本轮不删除。共享 decoder 内部仍以 primary/active 字段实现 slot 绑定，这是模型内部细节，已不再泄漏到统一分支。

---

## 4. D04：配置的作用域与生效时机（核心路径完成）

| 作用域 | 内容 | 本轮实现 | 生效方式 |
|---|---|---|---|
| Process | 只能初始化一次的原生设置 | `ProcessSettings.Claim/TryClaim`：GGML backend 类型（第二种类型在调用 native 前明确拒绝，说明由谁先初始化）；MLX `MLX_ENABLE_TF32`（首个 MLX allocator 声明实际精度，LTX 2.5 需要不同值时报告冲突）；TP 的 `GGML_CUDA_DISABLE_GRAPHS` 由 loader 按解析后的 placement 在 backend 初始化前设置 | 首次声明即固定；冲突报告 |
| Model instance | TP/layer split/设备列表/多节点；MoE CPU offload；holder pool 上限 | `ModelPlacementOptions`（`ModelBase.Placement`）；`MoeCpuOffloadPolicy`（`ModelBase.MoeOffload`，权重驻留、expert placement 和 FFN 派发都读它）；共享 decoder 的 `_holderPoolMax` | 加载时形成不可变快照；之后修改进程默认值只影响下一次加载 |
| Engine step | `ExecutionOptions`（batched/per-seq 开关、保留与 checkpoint 预算、KV 预留等） | `ExecutionOptionsSource`：每个 step 边界取一份快照（带版本号），该步的 admission、planner、executor 和 prefix cache coordinator 读同一份；`SchedulerConfig.ExecutionOptions` 传入显式配置；`InferenceEngine.UpdateExecutionOptions` 是 step 边界生效的 engine command | 未显式给出时兼容地在 step 边界读取环境变量 |
| Request | sampling、adapter | 既有 | 请求创建时 |

**输入渠道与优先级。** CLI 参数 > 进程环境变量 > 默认值（`ModelParallelismOptions.Parse` 逐项解析，`Placement` 给出结果）。宿主只在入口解析一次：

- CLI：不再调用 `ModelParallelismOptions.ApplyEnvironment()`；`ModelBase.Create(path, backend, placement, tpGroup, draft)` 接收解析结果；交互式 reload 沿用已加载模型的 placement 快照。
- Server.Host：`ServerOptionsBuilder.ResolveModelParallelism` 解析；`ModelService.Placement` 交给每次加载，分布式 group factory 读同一份选项，日志与 OpenTelemetry resource 也读它。`ApplyTensorParallelCliFlags` 保留给仍需写环境变量的集成。
- `ModelBase.Create` 的显式 placement 重载不读取环境变量；旧重载为兼容保留，从环境读取一次后转交新重载（修复了“显式传入 1 无法覆盖环境里的 2”的歧义，宿主改用新重载）。

**行为变化。** MoE offload 与 holder pool 上限在模型加载后不再跟随进程默认值变化（`MiniMaxM3ModelTests.ShouldPrefaultSkipsOnlyOffloadedExperts` 已改为在加载前配置，并断言加载后的修改不影响已加载模型）。

**未迁移（列入后续）。** Server 其余以环境变量为通道的 CLI applier（`--gpu-device`→`TS_GGML_VULKAN_DEVICE`、`--prompt-attention`、`--diffusion-products`、`--kv-cache-dtype`、`--spec*`、`TS_SCHED_*` 等）、native 侧直接 `getenv` 的调优项（如 `TS_CPU_MOE_THREADS`）以及 `LayerSplitPlan` 的 driver stage 仍按原方式工作；它们按同一模式迁移即可。

---

## 5. D02：资源所有者层（部分完成）

- [`DeviceResidency`](../../TensorSharp.Core/Graph/DeviceResidency.cs)：backend 在模块加载时注册其常驻权重缓存的释放方式（CUDA、MLX 按 cache key）；模型用 `ReleaseWeightCopies`（常驻权重 + 绑定 host 指针的 executor 副本）、`ReleaseTensor`（先失效再释放）或 `ResidentTensorLease`（释放顺序由 lease 保证、只释放一次）释放自己拥有的资源。不会为查询而创建 executor，也不引入设备同步。
- 迁移：`NetworkWeights.Release`（原先按 CUDA/MLX/其他分支）、LFM3 depthformer/codec decoder/audio encoder、LTX 2.5 denoiser、MiniMax-H3 denoiser 的释放代码不再区分 backend 类型。
- 验证：`DeviceResidencyTests`（路由与顺序的单元测试；`[CudaFact]` 的地址复用测试：权重驻留 → 通过 `DeviceResidency` 释放 → 同一地址写入新值 → 再次上传得到新值）。
- 未完成：storage generation、view 借用关系、异步工作与 captured graph 的有效期追踪；其余模型（Qwen-Image、MiniMax-H3-Image、Gemma 4 vision/audio、embedding encoder、`ModelBase` 权重释放等）的迁移。

---

## 6. 延期项与理由

| 设计 | 理由 | 后续入口 |
|---|---|---|
| D05 BackendSession | 需要改造 native bridge 的全局状态（`g_backend_type`、`g_device_states`、异步状态）为 session handle，并移动 `ModelBase` 构造中的后端创建；影响所有家族和三种 GGML 设备，需在多 GPU、Metal、Vulkan 上验证，本环境只有单 A40 | `ProcessSettings` 已把“一进程一种 GGML backend”显式化；`ModelPlacementOptions` 已把设备选择从 `ModelBase` 的环境读取中移出 |
| D06 decoder 组件化 / 权重描述 | `StandardDecoderModel`（31 个文件、约 2 万行）的状态拆分需要逐组件迁移并在 Gemma 4、Qwen 3.5 dense/MoE 上做数值验证，一轮内无法安全完成 | D01 的 `ISequenceStateStore` 是 sequence state store 组件的外部边界 |
| D07 语义与策略分离 | 需要先定义 shape-aware capability 查询与 resolved execution plan，并迁移 `GraphWideTokenRoutes` 等调优路线；与 D06 的权重描述相互依赖 | D03 的 schema 已把参数语义与 executor 解释分开 |
| D08 原生子通信组 | 需要 NCCL/ggml 子组通信与多 GPU 验证；本环境单 GPU，无法验证正确性和性能 | — |
| D09 多模态扩展边界 | 需要把家族方法移出 `ModelMultimodalInjector` 并为每个多模态家族做 span/position/prefix identity 验证 | — |

---

## 7. 模块化指标（计划 7.3）

| 指标 | 本轮状态 |
|---|---|
| 新统一执行路径依赖旧接口 | 已消除；旧路径由 `LegacyHolderBinding` 承接 |
| 同一执行事实的权威来源 | 准入路由：`PlanAdmission`；op 协议：`graph-ops.json`；可服务性：`EngineServability` |
| 模型中的手工设备缓存失效 | 已迁移的家族由 `DeviceResidency` 收拢；其余待迁移 |
| Graph 协议维护源 | 一份 schema |
| step 配置读取 | 同一步一份快照（`ExecutionOptionsSource`） |
| Bespoke / known gaps 清单 | 未改变（`ServingContractConformanceTests` 照常通过） |

---

## 8. 验证

### 8.1 环境与方法

- **本地（云端容器，4 核 CPU，无 GPU）**：.NET 10.0.115；ggml `v1.1.1`（`bb877c0`，上游未修改，`eng/fetch-ggml.sh` 获取）；`libGgmlOps.so` 以 ggml-cpu 构建。运行 `InferenceWeb.Tests` 全量（ggml-cpu 固定为测试进程的 GGML backend；Metal、MLX、Vulkan、CUDA 以及需要真实权重的测试被跳过，不计为通过）。
- **GPU（RunPod，1× NVIDIA A40 48 GB，驱动 595.112.07 / CUDA 13.2，cgroup 约 7.6 个 CPU）**：同一 ggml `v1.1.1`，`build-linux.sh --cuda --no-vulkan --cuda-arch=86-real`。基线 `75e17d9` 与分支各自构建 Release 二进制，之后只运行预构建的二进制，按 ABBA 顺序交替，两次运行之间空闲 10 s。
- **模型**：`gemma-4-E4B-it-Q8_0`（ggml-org）、`Qwen3.5-9B-Q4_K_M`（unsloth）、`Qwen3.6-35B-A3B-UD-Q4_K_M`（unsloth，MoE），均在 `ggml_cuda` 上运行。
- **测量**：CLI greedy 生成 160 token（`--temperature 0`）比较完整文本；CLI `--benchmark --bench-prefill 1024 --bench-decode 128`（prefill 取每轮最好值、decode 取每轮平均值，再对 ABBA 轮次取均值）；`benchmarks/ConcurrentDecodeBench`（`InferenceEngine` 路径，即共享 decoder 在统一 step 上的服务路径；N=1 与 N=4，prompt 256、生成 96，每个并发请求与单独运行的 greedy 输出逐 token 比较）。

### 8.2 第一轮 A/B（分支 = D03 + D01，提交 `5645002`）

| 模型 | greedy 文本 | CLI prefill tok/s 基线→分支 | CLI decode tok/s 基线→分支 | 引擎 N=1 tok/s | 引擎 N=4 合计 tok/s | 引擎正确性 |
|---|---|---|---|---|---|---|
| gemma-4-E4B-it-Q8_0 | 完全相同 | 8632 → 8913 | 85.8 → 86.3 | 89.5 → 89.5 | 287.4 → 295.1 | 全部与单独运行一致 |
| Qwen3.5-9B-Q4_K_M | 完全相同 | 4327 → 4101 | 90.6 → 90.4 | 93.5 → 93.4 | 264.4 → 264.2 | 全部与单独运行一致 |
| Qwen3.6-35B-A3B-UD-Q4_K_M | 完全相同 | 4963 → 4930 | 108.0 → 108.4 | 见第二轮 | 见第二轮 | 见第二轮 |

单次 1024-token prefill 约 120–280 ms，轮间波动可达 15%（Qwen3.5 基线第二轮 3642、分支第一轮 3319/3002/3843，分支第二轮 4316/4308/4360，基线最好 4301），因此第一轮 prefill 均值的差异属于噪声范围；第二轮增加到每轮 5 次。
