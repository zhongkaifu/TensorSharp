# 环境变量 x 功能矩阵

[English](env_var_feature_matrix.md) | [中文](env_var_feature_matrix_zh-cn.md)

本文是 [`TensorSharp.TestMatrix`](../TensorSharp.TestMatrix/README_zh-cn.md)
使用的运行时开关参考。它只覆盖会真实影响推理正确性、吞吐、内存占用或模型路由
的高影响环境变量。

代码侧的事实来源是
[`TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs`](../TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs)。
默认 sweep 列表配置在
[`TensorSharp.TestMatrix/Defaults/matrix-config.json`](../TensorSharp.TestMatrix/Defaults/matrix-config.json)。

## TestMatrix 如何使用本文

- 每个适用的 `(model, backend, feature)` cell 都会先运行一个**baseline**：
  不强制设置任何 sweep 变量。
- 对每个被选中的环境变量，运行器会为每个列出的值创建一个 case，并只把该变量
  传给 `TensorSharp.Cli` 子进程。
- 每个子进程启动前，会清理继承来的 `TS_*`、`GDN_*`、`QWEN35_*`、
  `FUSED_*`、`KV_CACHE_DTYPE`、`MAX_CONTEXT`、`MAX_TOKENS`、
  `VIDEO_MAX_FRAMES`、`VIDEO_SAMPLE_FPS`，确保矩阵值是权威输入。
- `--env-vars none` 会关闭 sweep case。如果配置文件中的 `default_env_vars`
  为空，且 CLI 没有覆盖它，运行器会使用全部已注册的 `EnvVarMatrix.All` 项——每项仍只作用于它适用的组合。

下表中的“运行时 baseline”表示变量未设置时的行为。“默认 sweep”表示当前默认
配置是否会扫描该变量，而不是所有已注册变量。

DiffusionGemma 当前不属于已注册的 TestMatrix 功能目录：还没有 diffusion prompt
类型，没有 diffusion 专属 env sweep，运行器也不会清理继承来的 `DIFFUSION_*`
变量。要把 diffusion 结果纳入标准矩阵，请先显式配置模型，并新增对应 feature
与 env-var 注册。

## 连续批处理 / 批处理前向

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE` | Nemotron-H | 原生批处理 Mamba2 step | 关闭 | `0`, `1` | 否 |
| `TS_PER_SEQ_FUSED` | fused 能力模型（GGML 后端上的 Gemma 4、GPT OSS 与 Qwen 3.8 Flash Next；`ggml_cuda` / `ggml_metal` 上的 Qwen 3.5/3.6/3.8；各自原生执行器上的 DeepSeek V4 / V4.1 与 GLM 5.x） | 并发（N>=2）序列走 per-request 融合 Forward，每个序列有自己的设备端 K/V holder；`0` 改走逐算子批处理分页路径，其 K/V 位于共享的主机块池（设备内存更少、decode 速率更低、没有保留的 holder） | 启用 | 未注册 | 否 |
| `TS_BATCHED_FUSED_DECODE` | 具备 token 批量融合 decode 的模型（Gemma 4、Qwen 3.5/3.6/3.8、GPT OSS、GLM 5.x、DeepSeek V4 / V4.1） | per-seq fused 路径内的真正 token 批量融合 decode（一张图跑全部 N 个序列）。在 GLM 5.x 上 4 个并发请求可得合计 1.81× decode；在 DeepSeek V4.1 Flash 的 Q4_K_M 上为 2.0×（24.3 → 48.9 tok/s），上限来自路由——每个 token 各自从 384 个专家里挑 6 个。批处理会改变 GEMM 形状，2 bit MoE 可能把这点差别放大成不同的专家选择；设为 `0` 可做串行路径 A/B。 | 开启 | 未注册 | 否 |
| `TS_BATCHED_FUSED_MOE` | Gemma 4 MoE | `1` 允许 Gemma 4 MoE 检查点走 token 批量融合 decode。默认关闭：它的可捕获计算图加上 KV holder 占满了 16 GB 显卡，在那里也不比轮询快 | 关闭 | 未注册 | 否 |
| `TS_RETAINED_FUSED_CACHE_MAX` | 具有可保留 request-owned fused holder 的模型（Gemma 4；Qwen 3.5/3.6/3.8；Qwen 3.8 Flash Next；未加载 DSpark 草稿器时的 DeepSeek V4.1；原生执行器上的 GLM 5.x，每个会话交出一个原生 slot） | 为下一轮的精确前缀续接保留多少个已结束会话的终态（holder、原生 slot）；Qwen holder 同时包含 attention K/V 与匹配的 GatedDeltaNet 递归状态。未设置时跟随 `TS_SCHED_MAX_RUNNING_SEQS`（至少 4），使每个可以并行运行的请求都保有自己的终态；字节数由显存而不是这个数量约束。`0` 关闭保留 | 运行中请求上限（16），至少 `4` | 不适用 | 否 |
| `TS_PREFIX_CHECKPOINTS_MAX` | GGML 后端上的 Gemma 4；`ggml_cuda` / `ggml_metal` / `mlx` 上的 Qwen 3.5/3.6/3.8，即它运行按请求 holder 的后端（TP 下不支持）；GGML token-span 路径上的 Qwen 3.8 Flash Next（包括 `--layer-split N` 按层切分时）。需要开启 `TS_PER_SEQ_FUSED` | 在每个会话共享的提示前缀（系统提示、工具、技能）结束处对模型完整状态做检查点，新会话从其副本继续，只需重新预填自己的消息。取值是同时保留多少个不同共享前缀的检查点（每个占用一份前缀的 K/V，Qwen 还包含递归状态）；`0` 关闭检查点。它是 Radix 前缀缓存保留的公共检查点预算；一个提示在每个边界（系统指令、完整共享前缀）各发布一个检查点，因此 4 可覆盖同时预热两种思考模式的主机 | `4` | 不适用 | 否 |
| `TS_KV_INITIAL_TOKENS` | 通过 `ModelBase.ResolveInitialCacheAllocationLength` 确定缓存大小的模型家族（Qwen 3.5/3.6、Gemma 4、GPT-OSS 等 ModelBase 家族；不含自行确定大小的 DeepSeek V4 / GLM 5.x） | 缓存创建时（加载时的主缓存、每个 per-request holder）在任何请求声明预算之前分配的 K/V token 数；`0` 沿用引擎策略（显式 `MAX_CONTEXT` 时为整个窗口，否则为后端默认值；Qwen 3.8 Flash Next 在 Metal 或单个 ggml_cuda 设备上、以及 Qwen 3.5/3.6/3.8 在单个 ggml_cuda 设备上的 per-request holder 无论如何都从 2048 起步，并且设备放不下时新的并发请求会等待）。缓存仍按需增长。内存受限设备把它设小，因为每个保留的 holder 都按此大小付费，主机副本与设备镜像各一份 | `0` | 不适用 | 否 |
| `TS_KV_GENERATION_RESERVE_MAX` | 全部 | 请求预先保留的 K/V 中生成部分的上限（prompt + max_new_tokens）；回复上限不小于窗口时否则每个请求都会保留整个窗口。超过上限后缓存按需增长。`0` = 不限制 | `0` | 不适用 | 否 |
| `TS_KV_HOLDER_POOL_MAX` | 具有 per-request fused holder 的模型（Qwen 3.5/3.6/3.8、Gemma 4、GPT-OSS） | 已释放的 holder 最多可停放多少个以待复用而不是释放；每个停放的 holder 都占用其完整 K/V 分配 | `64` | 不适用 | 否 |
| `TS_SCHED_DISABLE_BATCHED` | 全部 | `1` 让所有模型都走按序列 KV-swap 路径，投机解码也一样（`--no-continuous-batching`） | 关闭 | `0`, `1` | 是 |
| `TS_SCHED_PREFIX_CACHE` | 全部自回归模型 | `0` 关闭准入时的全部提示复用（Radix 前缀缓存：页面、保留的终态与公共检查点在同一个索引里）。`--no-prefix-cache`（CLI 与服务端）会设置它，并同时跳过共享提示的启动预热，以及服务端的检查点文件 | 启用（`1`） | 未注册 | 否 |
| `TS_SCHED_MAX_BATCHED_TOKENS` / `TS_SCHED_MAX_RUNNING_SEQS` / `TS_SCHED_PREFILL_CHUNK` / `TS_SCHED_SOLO_PREFILL_CHUNK` / `TS_SCHED_NUM_BLOCKS` / `TS_SCHED_BLOCK_SIZE` / `TS_SCHED_DECODE_QUANTUM` | 全部 | 调度器预算：每步 token 数、同时运行的序列数、有 decode 在跑时的 prefill 分块（服务端 `--prefill-chunk-size`）、solo prefill 分块、块池块数、每块 token 数（只靠页面做跨请求复用的家族为 16，其余为 256）、decode 时间片。有自己单请求步长的模型——启用流水化 prefill 的 Muse Spark：8192——在未设置 `TS_SCHED_SOLO_PREFILL_CHUNK` 时自行决定步长；设置了该变量则按文档生效并关闭模型自己的步长，只设置 `TS_SCHED_MAX_BATCHED_TOKENS` 不会关闭它。见[配置表](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md#配置) | `4096` / `16` / `256` / `8192` / `256` / `16` 或 `256` / `256` | 未注册 | 否 |
| `TS_SCHED_STOP_REPETITION` | 全部 | `0` 时陷入循环的生成会一直跑到 token 上限，而不是以 `repetition` 结束原因停止；重复模板行的思考也不再被闭合 | 启用（`1`） | 未注册 | 否 |

本节的 executor 级开关（`TS_SCHED_DISABLE_BATCHED`、`TS_PER_SEQ_FUSED`、
`TS_BATCHED_FUSED_DECODE`，以及保留 holder、检查点与 `TS_KV_*` 预算）通过 `ExecutionOptions.FromEnvironment()` 统一读取，由 `ExecutionPlanner` 消费
（见 `docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md` 的"执行规划"一节）；
`TS_SCHED_*` 由 `SchedulerConfig.FromEnvironment()` 读取；
MoE 开关由模型或其原生内核读取。

TensorAgent 在每次加载模型之前写入自己的取值（`EngineMemoryPolicy`）：
`TS_KV_INITIAL_TOKENS=2048`、`TS_KV_GENERATION_RESERVE_MAX=1024`、
`TS_KV_HOLDER_POOL_MAX=0`、`TS_RETAINED_FUSED_CACHE_MAX=1`，以及来自目录条目或用户
设置的 `MAX_CONTEXT` / `KV_CACHE_DTYPE`。iOS 应用（`TensorAgent.Maui`）在未设置时还会
默认 `TS_SCHED_SOLO_PREFILL_CHUNK=1024`、`TS_PREFIX_CHECKPOINTS_MAX=1` 与
`GGML_METAL_NO_RESIDENCY=1`。

## KV Cache / 上下文

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `KV_CACHE_DTYPE` | 除 DeepSeek V4 / V4.1 执行器之外的全部：它们的 cache 固定为 F16，由自己的注意力、gather 与压缩器内核直接读取，`q8_0` / `q4_0` 会在**加载时被拒绝**并给出原因（打开检查点之前抛 `NotSupportedException`），`f32` 会被告知并按 `f16` 报告 | KV cache 元素类型 | 自动（随模型对齐：模型权重低于 F32 时为 `f16`，否则为 `f32`） | `f32`, `f16`, `q8_0`（运行时还接受 `q4_0`，不参与 sweep） | 是 |
| `TS_N_CPU_MOE` | MoE 模型 | 前 N 层的路由专家留在系统内存：decode 时在主机上做乘法，prefill 时流式送到加速器上跑一整张图。在 `ggml_metal` 上未设置时，若专家放不下，Qwen 3.8 Flash Next（`qwen4exp`）会按 Metal 工作集与内存自行规划：每层都在主机接缝处调度，并保留按路由画像排序的加速器专家层与显式读取的窗口槽位（见 `TS_MOE_HOT_TIER`） | 关闭（`0`） | `0`, `16`, `all` | 否（已为 GGML GPU 后端与 MoE 家族注册，但不在默认配置的列表中） |
| `TS_CPU_MOE` | MoE 模型 | 卸载所有层的路由专家（等价于 `TS_N_CPU_MOE=all`） | 关闭 | 未注册 | 否 |
| `TS_CPU_MOE_THREADS` | MoE 模型 | 主机端专家 matmul 的工作线程数。可用 CPU（硬件线程数按亲和性掩码与 cgroup CPU 配额收敛后）超过 8 个时，默认取其一半，上限 64：decode 侧的 matmul 只有一个 token 宽，超过几十个线程后每多一个线程只是多一个屏障参与者（在双路 Xeon 上实测 192 线程比 32 线程慢 7 倍）。较小的主机几乎用满全部 CPU。DeepSeek V4 / V4.1 则默认使用全部可用 CPU（见下文） | 可用数 ≤2 时为 1，≤8 时为可用数−1，否则 min(可用数/2, 64) | - | 否 |
| `TS_HOST_MOE_DEVICE_MIN_BATCH` | 启用卸载的 MoE 模型 | 达到或超过该 batch 大小时，被卸载的层改为在加速器上计算、专家权重流式送入，而不是在主机上算。`0` 恢复纯主机卸载 | `128` | 未注册 | 否 |
| `TS_HOST_MOE_PIN` | 启用卸载的 MoE 模型 | 把被卸载的专家区间页锁定（`cudaHostRegister`），使流式 prefill 走 DMA 而不是经驱动中转（PCIe 5.0 上 9.3 → 55.6 GB/s）。`0` 对所有架构关闭。DeepSeek V4 / V4.1 的默认值是例外：它们的加载器在任何批大小下都在 CPU 后端上计算被卸载的专家，没有任何流式传输，因此只有 `1` 时才锁页（七卡 A40 通道上锁定 48.2 GiB 让加载多花 20.4 s，并使这些页面无法被回收） | 启用；DeepSeek V4 / V4.1 为关闭 | 未注册 | 否 |
| `TS_HOST_MOE_PIN_MAX_MB` | 启用卸载的 MoE 模型 | 页锁定专家区间的预算 | cgroup / 主机内存上限的 60% | - | 否 |
| `TS_HOST_MOE_DECODE` | 启用卸载的 MoE 模型 | `0` 让单 token 的被卸载层改走 ggml CPU 计算图，而不是 TensorSharp 的 decode 内核（仍用 ggml 自己的 CPU 点积，但线程组每层只唤醒一次；在 M5 Pro 上跑 Qwen3.8-Flash-Next 时每层 0.6 → 0.32 ms）。用于 A/B 对比 | 启用 | 未注册 | 否 |
| `TS_HOST_MOE_TIMING` | 启用卸载的 MoE 模型 | 诊断：`1` 主机侧每次调用的准备与 matmul 时间，以及流式 prefill 的字节数、传输速率与 GPU 时间；`2` 每个权重的拷贝速率（会同步，改变所测的东西）；`3` 每个 decode pass（按同一宽度的 32 次统计）及每个 prompt pass 的加速器分段、主机专家与上传时间；`4` 单 token 内核的各阶段与工作线程的唤醒 | 关闭 | 未注册 | 否 |
| `TS_HOST_MOE_EXPERT_FILTER` | 启用卸载的 MoE 模型 | 只流式传输该 batch 实际路由到的专家，并合并成连续区间 | 启用 | 未注册 | 否 |
| `TS_MOE_HOT_TIER` | 专家放不下时在 `ggml_metal` 或单卡 `ggml_cuda` 上运行的 Qwen 3.8 Flash Next（`qwen4exp`） | `0` 保留旧方案（整层把末尾若干层的专家常驻 GPU）。默认：每层的路由专家都在主机接缝处调度，每层保留一块紧凑的加速器专家层：按路由画像排序的高频专家占固定槽位；在统一内存上另有一块按需显式读取填充的窗口槽位，由加速器计算路由，主机只负责 id 重映射和读取未命中的专家（独立显存上，固定槽位以外的路由在主机 decode 内核上计算）。若尚无覆盖该模型的画像：Metal 上本次加载只用窗口槽位（与有画像时相差 3-5%）并记录画像；CUDA 上本次保留层计划，并记录所有层（包括常驻层）的路由，之后的加载即可分层。在 `ggml_metal` 上，若在与分层相同的内存上限下整层方案仍能让至少 85% 的层常驻（主机接缝更少），则改用整层方案；`1` 仍使用分层 | 启用 | 未注册 | 否 |
| `TS_Q4E_EXPERT_VRAM_GB` | 在 `--backend cuda`（直接 CUDA 引擎）单卡上运行的 Qwen 3.8 Flash Next | 路由专家 VRAM 槽位的预算，强制启用专家驻留（`0` 让所有专家都留在锁页主机内存）。未设置时，仅当专家放不下时才规划驻留：按路由画像最热的专家占用 VRAM 槽位，只有其余专家被复制进锁页主机内存，由选中它们的调用暂存进 VRAM，使 decode 步骤仍是一张捕获的计算图 | 按空闲显存规划 | - | 否 |
| `TS_Q4E_EXPERT_READERS` | 同上 | 加载时把没有 VRAM 槽位的专家复制进锁页主机内存、把槽位专家复制进 VRAM 的并行读取线程数 | `16` | - | 否 |
| `TS_Q4E_HOST_EXPERTS` | 同上，启用专家驻留时 | `0` 把每个未命中的专家暂存进 VRAM，而不是在主机上计算。默认：最多 8 行的 decode 步骤或验证窗口把没有 VRAM 槽位的专家交给 CPU（Strata 的 doorbell：GPU 把该层的输入和选择结果发布到主机映射内存，在同一张捕获的计算图内等待主机的求和结果，同时自己计算驻留的专家）；prompt 仍走暂存。主机用 ggml 的 CPU 内核求和，因此有未命中的步骤与全驻留引擎在最后几位上不同。RTX 5090、UD-Q2_K_XL：70.6/47.9/64.5 对暂存的 52.0/38.7/45.7 tok/s（code/prose/list）；ISTA `Q2_0` 为 94.6/65.6/97.3（ggml_cuda 分层 82.6/46.4/88.4）。主机工作线程组使用流等待和 doorbell 服务线程之外剩下的 CPU（`--cpu-moe-threads` 可覆盖） | 开启 | - | 否 |
| `TS_Q4E_HOST_EXPERT_ROWS` | 同上，启用主机专家时 | 不超过该行数（1-16，默认 8）的步骤把缺失专家交给 CPU 计算；更宽的步骤改为经总线暂存。`1` 让验证窗口不经过 CPU；`16` 让四个序列各 4 行窗口的批量投机验证同样走 CPU 路径（需要本树构建的原生库，其主机内核支持 16 行）。在 RTX 5090 上挂 MTP 头时，4 行窗口仍为 27-30 ms，与默认相同。 |
| `TS_Q4E_HOST_EXPERTS_PIN` | 同上，启用专家驻留时 | 没有显存 slot 的专家在主机上的存放方式。未设置时：若它们与锁页 KV 主机副本在主机可用内存（Linux 为受内存 cgroup 限制的 MemAvailable；Windows 为可用物理内存）中放得下并留出 4 GiB，则使用锁页副本；否则直接从模型文件的内存映射读取：可被操作系统换出的分页内存，因此在放不下锁页副本的机器上也能加载（16 GB RTX 3080 Laptop、32 GB 内存上的 ISTA `Q2_0`：25.1 GiB 主机专家，可用 16.7 GiB），速度取决于操作系统提供这些页面的速度。映射模式下，不超过 16 行的步骤都把缺失专家交给 CPU，prompt 由主机把缺失专家复制到 GPU，GPU 分担缺失专家（`TS_Q4E_HYBRID_MISSES`）与 prompt 预取关闭；输出与锁页模式逐位一致。锁页加载耗尽主机内存时回退到映射。`1` 总是锁页（无法锁页时失败），`0` 总是映射。 | 未设置 | - | 否 |
| `TS_Q4E_HYBRID_MISSES` | `--backend cuda` 上启用主机专家的 Qwen 3.8 Flash Next（decode 步与验证窗口） | 一步中由 GPU 经总线复制并自行计算的不同缺失专家数，其余仍由 CPU 计算（0 = 全部缺失专家交给 CPU；最多占专家暂存区的一半）。doorbell 内核规划 GPU 的份额并把它从主机份额中去掉；缺失专家全部由 GPU 接手的层不再敲 doorbell。GPU 上的专家按全常驻方式计算，因此输出会变化（在主机计算的容差内，且更接近全常驻；接手全部缺失专家时与全常驻逐位一致）。RTX 5090、ISTA `Q2_0`、MTP、长生成：0 -> 208-217 tok/s，2（odd）214-226，2（`first`）210-232；4 行窗口的 CPU 专家 41 -> 20（odd）/ 6（`first`）。自 2026-11-08 起默认 4（配合 `first`）：4 行窗口验证 11.4-13.0 -> 10.7-12.1 ms。 | `4` | - | 否 |
| `TS_Q4E_HYBRID_SPLIT` | 同上，配合 `TS_Q4E_HYBRID_MISSES` | GPU 接手哪些缺失专家：默认 `odd`，只接编号为奇数的专家，因此无论一次调用还选了什么，同一专家总在同一侧（份额足够时，验证行与其自身的 decode 步计算一致）；`first`，按 slot 顺序取最前面的（接得更多，但一行的划分取决于窗口；自 2026-11-08 起为默认）。 | `first` | - | 否 |
| `TS_Q4E_COLLAPSE_Q81` | `--backend cuda` 上的 Qwen 3.8 Flash Next，decode 步与验证窗口 | `0` 让每个半层的混合输入单独启动一次量化。默认：hyper-connection 的 collapse 同时写出注意力、GDN、共享专家和路由专家投影读取的 q8_1 块（数值相同；每个半层、每次路由专家调用少一次启动）。 | 开启 | - | 否 |
| `TS_Q4E_KV_RESIDENT` | `--backend cuda` 上使用 Q8_0 KV 缓存的 Qwen 3.8 Flash Next | 每个 QSA 注意力层留在显存中的 K/V 单元数（Strata 的 `--kv-resident`）；整个上下文的权威 K/V 放在锁页、设备映射的主机内存中，decode 步与验证窗口在捕获的步骤内把要读的页面调入显存、缺页经总线复制，prompt 读取从主机副本填充的暂存池；原始索引键改为环形缓冲。至少容纳一次调用的页面（16 行 x 2,051 单元；Flash Next 为 32,960 单元）。输出与常驻缓存逐位一致；腾出的显存给专家 slot。`0` 让每层全部常驻。未设置时：路由专家无法全部放入显存（专家驻留）时为 32768，能放下时每层全部常驻（自 2026-11-08 起为默认）。 | 未设置：专家驻留时 `32768`，否则 `0` | - | 否 |
| `TS_Q4E_LANES` | 专家驻留下 `--backend cuda` 上的 Qwen 3.8 Flash Next，以及 `ggml_cuda` | 驻留规划在第一个序列之外再为多少个序列（1-16）预留显存（缓存与草稿 KV），使并发请求拥有自己的 slot、以一次批量步解码，而不是排队等待；每条 lane 占用专家 slot（挂载 MTP 头时 131K 上下文的一条 lane：常驻约 2.6 GB，配合 `TS_Q4E_KV_RESIDENT` 约 0.9 GB）。在 `ggml_cuda` 上，专家层级为第一个之外的每条 lane 预留 8192 行 K/V 及该序列的状态（约 0.5 GB）；并发请求的缓存从 2048 行开始并随请求增长，放不下（需留 1 GB 余量）的请求无论此设置如何都会等待正在运行的请求结束。 | `1` | - | 否 |
| `TS_Q4E_BATCH_GRAPHS` | 并发请求时 `--backend cuda` 上的 Qwen 3.8 Flash Next | `0` 让批量步（多个序列的行在一次前向中：批量 decode 步与批量投机验证）不经 CUDA graph 捕获运行。捕获时每个 GPU 以一张 graph 重放一步（包括层切分），以各序列的 slot 与 QSA 区段、行数和输出头形式为键；同一键第三次运行时才捕获，slot 无法分配或增长时已捕获的步骤会让出显存。捕获的步骤与未捕获的结果逐位相同；需要暂存缺失专家的步骤不捕获。 | 开 | - | 否 |
| `TS_Q4E_BATCH_GRAPHS_MAX` | 同上 | 最多保留的已捕获批量步数量；超出时先丢弃最久未用的。 | `64` | - | 否 |
| `TS_Q4E_SLOT_TOKENS` | `--backend cuda` 上的 Qwen 3.8 Flash Next | 序列 slot 的初始容量（向上取整到完整的 QSA 块）；随序列增长倍增直到上下文长度，无法增长时先释放最早保留的对话的 slot。处于常驻 KV 窗口（`TS_Q4E_KV_RESIDENT`）下的层不增长：其窗口与主机副本已覆盖整个上下文。 | `32768` | - | 否 |
| `TS_Q4E_REWIND_CHECKPOINT` | 同上 | `0` 关闭每个 slot 在每次多 token 前向结束时对递归状态的检查点；有了它，只与上一轮回复部分匹配的新一轮可以回退到匹配处并重放该 slot 自己的 token。 | 开 | - | 否 |
| `TS_Q4E_PLACEMENT_SLOTS` | 使用 `--layer-split` 时 `--backend cuda` 上的 Qwen 3.8 Flash Next | 层切分在权重之外为多少个 slot 的缓存做均衡（尽量容纳）；`1` 只为一个均衡。 | `16` | - | 否 |
| `TS_Q4E_PERF_SAMPLE` | `--backend cuda` 上的 Qwen 3.8 Flash Next | 每 N 次前向中有一次不经捕获运行并带阶段计时，按阶段、步骤类型与行数记录 `tensorsharp.engine.stage.duration`；`TS_Q4E_PERF` 为 2 或以上时关闭。 | `0` | - | 否 |
| `TS_Q4E_PERF_SAMPLE_MIN_MS` | 同一采样剖析 | 两次采样前向之间的最短间隔；`0` 表示每第 `TS_Q4E_PERF_SAMPLE` 次前向都采样。采样前向不经捕获运行，每个阶段一个 GPU 事件，耗时是已捕获步骤的数倍，在层切分下还要先等另一条解码通道的步骤完成：两条通道、16 路并发时每 200 次前向采样一次约为每两秒一次，损失 5-7% 的 token（715 对 757 tok/s）。解码通道的步骤现在也会被采样 | `10000` | - | 否 |
| `TS_Q4E_MTP_RESUME` | 挂载 MTP 头（`--draft-model`）时 `--backend cuda` 上的 Qwen 3.8 Flash Next | `0` 让 MTP 块在出现它从未重放过的位置后不重启自己的缓存：投机只用于提示词从位置 0 起经由其上下文预填的请求，因此并发请求（彼此并排预填）、复用前缀的轮次以及经历过普通解码间隔的请求都按普通方式解码。开启（默认）时，该块在间隔末尾重启，这些请求都会投机，并发请求在一次批量验证中完成。 | 开 | - | 否 |
| `TS_Q4E_BATCHED_VERIFY_ROWS` | 并发请求且挂载 MTP 头时 `--backend cuda` 上的 Qwen 3.8 Flash Next | 一次批量投机验证步保持的行数（2-16）；各序列分摊这些行作为各自的窗口（不超过单序列窗口），每个序列不足两行时改为在一次批量步中普通解码。不同序列的行很少共享专家，验证成本几乎与行数成正比：取 8 时，两个序列各验证 4 行，三或四个序列各 2 行。 | `8` | - | 否 |
| `TS_Q4E_DECODE_LANES` | 带层切分且批量步骤已捕获的 Qwen 3.8 Flash Next（`qwen4exp`）直连 CUDA 引擎 | `0` 拒绝第二个在途的批量解码步骤（调度器随后每次只解码一个批量步骤）。开启时，两条解码通道各有自己的固定内存 token、位置、边界、PLE 行、状态表和 logits，各通道的图分开键控，一条通道的步骤只等待它自己的上一步；其他所有公共调用都会先完成在途的通道 | 开 | 未注册 | 否 |
| `TS_Q4E_PROMPT_PIPELINE` | 带层切分的 Qwen 3.8 Flash Next（`qwen4exp`）直连 CUDA 引擎 | `0` 等待每个提示词分块完成。开启时，后面还有提示词的分块（调度器通过 `IModelArchitecture.ForwardPrompt(tokens, morePromptFollows)` 告知；不含结束于检查点边界的分块）不运行 LM 头，发出后留在最后一个设备上继续计算，第一个设备随即开始同一序列的下一个分块，按流内顺序衔接，如同一次调用中的第二个 ubatch；其他任何调用都先等它完成。内核与顺序相同，因此答案相同（两块 H100 NVL，Q8_0，`--layer-split 2`：4 万 token 的提示词 6,229-6,262 tok/s，原为 4,535-4,597） | 开 | 未注册 | 否 |
| `TS_Q4E_PLE_GATHER_THREADS` | Qwen 3.8 Flash Next（`qwen4exp`）直连 CUDA 引擎 | 在主机上收集提示词 ubatch 的 PLE 行所用的线程数（每行只取决于自身 token 和历史；每线程至少 64 行，因此解码步骤仍为单线程）。`1` 保持单线程（4,096 个 token：37 ms，16 线程为 5 ms） | min(16, CPU 数的一半) | 未注册 | 否 |
| `TS_Q4E_MTP_EXPERTS` | 挂载共享 MTP 头（`--draft-model`）的 Qwen 3.8 Flash Next，GGML 后端和 `--backend cuda` | `q2_0`（在 GGML 后端上也可用 `q4_0`）在加载时用 ggml 的量化器重新量化该头的 512 个路由专家，与 Strata 发布其草稿层的方式相同。共享头的 Q8_0 专家从 2,550 MiB 降到 675 MiB，腾出的空间留给目标模型的专家分层或显存槽位；在 5090 上，常驻显存的路由从 95.0% 升到 96.8%。只影响草稿：每个 token 都由验证决定，长生成的接受率保持在 85-87%。`q8_0` 或 `file`（默认）保留文件中的专家。 |
| `TS_Q4E_UBATCH` | `--backend cuda` 上的 Qwen 3.8 Flash Next | 每次前向的 prompt 行数（自 2026-11-08 起默认 1024，之前为 512：在 RTX 5090 上用 ISTA `Q2_0`，4K TTFT 从 1.29 降到 1.07 s，16K 从 5.05 降到 4.17 s，代价是少 345 个专家槽位，长生成在噪声之内）。启用专家驻留时，每个缺失专家每个 ubatch 暂存一次，所以 `2048` 把 4K prompt 从 2.70 s 降到 1.75 s，16K 从 11.4 s 降到 7.5 s。它的 scratch 占用专家槽位：普通 decode 从 103 降到 98 tok/s。 |
| `TS_Q4E_PROMPT_GROUP` | 单 GPU、启用专家驻留时 `--backend cuda` 上的 Qwen 3.8 Flash Next | 按层读取的 prompt 行数（默认 4096，向下取整到整数个 ubatch；`0` 保持按 ubatch 顺序）。每一层先跑完组内所有 ubatch 再进入下一层，并在这些 ubatch 之间保留已暂存的专家，因此缺失专家每层每组只经总线传输一次，而不是每 512 行 ubatch 一次（Strata 以 8192 token 的块读取 prompt 也是这个原因）。结果与按 ubatch 顺序逐位一致。组的流（4096 行时 160 MiB）占用专家槽位。RTX 5090、ISTA `Q2_0`、MTP：4K TTFT 2.65 -> 1.85 s，16K 11.1 -> 7.5 s | `4096` | - | 否 |
| `TS_Q4E_GROUP_F16` | `--backend cuda` 上的 Qwen 3.8 Flash Next，分组读取的 prompt（`TS_Q4E_PROMPT_GROUP`，专家驻留下的单 GPU） | `0` 让组中每个 ubatch 都重新把这一层的稠密权重反量化为 F16。默认：一层的第一个 ubatch 把每个量化投影反量化到组的 F16 arena（最大一层投影的 F16 形式：ISTA 包为 194 MiB；驻留规划为它预留空间），组内其余 ubatch 复用该副本，因此 prompt 的结果逐位相同。RTX 5090、ISTA `Q2_0`：显存中少 147 个专家；4K TTFT 0.55 -> 0.51 s，16K 2.38 -> 2.27 s | 开启 | - | 否 |
| `TS_Q4E_GROUP_EXPERTS` | `--backend cuda` 上的 Qwen 3.8 Flash Next，分组读取的 prompt（`TS_Q4E_PROMPT_GROUP`），Q2_0 专家走 int8 路线（`TS_Q2_0_EXPERTS_INT8`） | `0` 让组内每个 ubatch 自己运行路由专家的 gate/up。默认：每个 ubatch 暂存它缺失的专家，并把选择结果、int8 激活、共享专家输出与 scatter logits 留给组；这一层最后一个 ubatch 之后，gate/up 在整个组上运行一次（每个专家的 gate 与 up 权重每层每组只读取一次），每个 ubatch 的 down 通过组的计划读取其 slot 的 H。结果逐位相同。缓冲区（4,096 行时 85 MiB）从专家 slot 中划出。组的最后一个 ubatch 不超过 16 行时保持 ubatch 顺序。RTX 5090、ISTA `Q2_0`：显存中少 64 个专家；4K TTFT 0.53 -> 0.50 s，16K 2.27 -> 2.21 s | 开启 | - | 否 |
| `TS_Q4E_PROMPT_PREFETCH` | 启用专家驻留、按层读取 prompt 时 `--backend cuda` 上的 Qwen 3.8 Flash Next | `0` 在 prompt 的 stream 上按 ubatch 暂存一层的缺失专家。默认：暂存区分成两半，一层在存放其专家的那一半上运行时，复制引擎把下一层的所有主机专家拷入另一半（每个矩阵一次 2D 复制，计划由主机写入），适用于每对主机专家都能放进一半的相邻层。prompt 结果逐位一致。RTX 5090、ISTA `Q2_0`（每层最多 228 个主机专家，每半 256 个）：4K TTFT 1.61-1.69 -> 1.44-1.46 s，16K 6.76-6.81 -> 5.84-5.87 s。按 ubatch 暂存所用的复制内核在这里更慢（4K 为 1.82 s）：它的 block 占着 SM 等待总线 | 开启 | - | 否 |
| `TS_Q4E_Q2_0_PROMPT_GEMM` | `--backend cuda` 上的 Qwen 3.8 Flash Next（ISTA 的 `Q2_0` 稠密投影） | `0` 让 `Q2_0` 的 prompt 行（64 行及以上）保持 warp-per-row 内核，该内核每一行都要重读一次权重。默认：为本次调用把权重反量化为 F16 副本并使用张量核 GEMM，即共享路径对其他格式所用的 prompt 方式。RTX 5090、ISTA `Q2_0`：4K TTFT 1.85 -> 1.69 s，16K 7.4-7.5 -> 6.7-6.9 s | 开启 | - | 否 |
| `TS_Q4E_ATTN_SPLITS` | `--backend cuda` 上的 Qwen 3.8 Flash Next | decode 规模的行把注意力单元格拆分到多少个 block（flash decoding；`1` 或 `0` 保持每个头每行一个 block）。超过 256 个单元格的行把它们分给这些 block，最后完成的 block 合并各部分结果；更短的行整行计算，与之前逐位一致。拆分只取决于该行的单元格数，所以验证窗口中的行仍然算出与各自 decode 步相同的结果。MTP 头（从不捕获）按自己的上下文决定拆分数。在此之前，RTX 5090 上 4 行验证窗口的注意力在 4K 上下文耗时 2.4 ms，16K 时 4.9 ms（`TS_Q4E_PERF=3`） | `16` | - | 否 |
| `TS_Q4E_ATTN_MMA` | `--backend cuda` 上的 Qwen 3.8 Flash Next，sm_80 及更新 | `0` 让 prompt 大小的注意力保持每个（头，行）一个 block 的内核。默认：每个（行，KV 头）一个四 warp 的 block，把该组的 query 头当作一个 16 行的 tensor core tile 一起计算，每个 16 个 cell 的块的 key 和 value 只在共享内存中展开为 F16 一次（`mma.sync`，F16 操作数，F32 累加，在线 softmax）。decode 步和验证窗口保持原内核。操作数舍入到 F16，所以 prompt 的 logits 的变化与 `q8_0` 缓存相对 F16 的变化相当。RTX 5090、ISTA `Q2_0`：4K TTFT 1.38-1.44 -> 1.23-1.29 s，16K 5.79-5.86 -> 4.99-5.08 s。为 compute_75 构建的模块（提交的基线）没有 tensor core 实现；此时引擎提示一次并保留旧内核 | 开启 | - | 否 |
| `TS_Q4E_QSA_SCORES_MMA` | `--backend cuda` 上的 Qwen 3.8 Flash Next，prompt 的 QSA 层（超过 16 行），sm_80 及更新 | `0` 让 prompt 的块分数保持每个块一个 warp 的内核。默认：移植自 Strata 的 prompt 打分。每个 block 在 `mma.sync` 上把 16 行的四个 indexer 头与 64 个块的 key 相乘，两个操作数都拆成 F16 高、低两半（q.k = qh.kh + ql.kh + qh.kl，F32 累加），并按头的顺序累加各头的 relu。分数与双精度结果相差不超过 sum\|q.k\| 的 4.6e-7，测试的每一行保留的块都相同。decode 步、验证窗口和批量步保持 warp 内核。RTX 5090、ISTA `Q2_0`：16K TTFT 2.47 -> 2.38 s。为 compute_75 构建的模块（提交的基线）没有 tensor core 实现，此时运行 warp 内核 | 开启 | - | 否 |
| `TS_Q4E_GDN_CHUNKED` | `--backend cuda` 上的 Qwen 3.8 Flash Next，单个序列 128 行及以上的 prompt | `0` 保持逐 token 的门控 DeltaNet 扫描（KdaScan）。默认：移植自 Strata 的分块形式。以 FP32 为每个 32 token 的块构建 WY 因子，再由 16 个 key 头 x 8 个列 tile 的扫描在共享内存中跨块携带状态（第一个长 prompt 时分配约 25 MB scratch）。输出相对变化 3e-7（求和顺序不同）。每个 block 需要 93,696 B 的 opt-in 共享内存，并需要至少 128 个 SM（一波完成）。不满足条件时、图捕获期间，以及 decode 步、验证窗口和批量步，运行 KdaScan。RTX 5090、ISTA `Q2_0`：4K TTFT 0.79 -> 0.70 s，16K 3.35 -> 3.02 s | 开启 | - | 否 |
| `TS_Q2_0_EXPERTS_INT8` | `--backend cuda` 上的 Q2_0 路由专家（ISTA 的 Qwen 3.8 Flash Next 包），超过 16 行的 prompt ubatch，sm_80 及更新 | `0` 保持 F16 tensor core 内核（权重展开为 F16，F32 累加）。默认：移植自 Strata 的 int8 内核。code 作为 int8 `mma.m16n8k32` 操作数，与每 32 个值舍入为 int8 的激活（F32 scale）相乘；SwiGLU 与 down 乘积的 int8 输入保留在寄存器中；通过驻留表就地读取 GGUF 块。激活的舍入方式与 decode 内核的 q8_1 行（ggml-cuda 的 MMQ）相同：相对以双精度计算的专家块，误差为 9.8e-3，q8_1 内核为 9.7e-3，F16 为 1.9e-4。RTX 5090：1,024 行的 512 专家层 1.64 -> 0.85 ms；4K TTFT 0.70 -> 0.55 s，16K 3.03 -> 2.47 s。为 compute_75 构建的模块（提交的基线）没有 int8 实现，此时运行 F16 内核 | 开启 | - | 否 |
| `TS_DENSE_FAST` | 直接 CUDA 引擎中 Q8_0、Q2_0、Q3_K、Q4_K、Q5_K、Q6_K、IQ4_XS 或 IQ4_NL 格式的 decode 大小稠密投影（1-16 行：decode 步、验证窗口、批量步） | `0` 让它们回到逐行内核（Q3_K/Q6_K/Q8_0 回到共享的 q8_1 内核）。默认：每个权重单元只解码一次（IQ4 查表在寄存器中用 `__byte_perm` 完成），与每个激活行做点积；按形状每个输出行用 1、2 或 4 个 warp；一行的和只取决于形状，所以验证窗口中的一行与它自己的 decode 步逐位一致。不超过 8 行的窗口运行与其宽度相同的内核（自 2026-11-09 起 1-8 每个宽度都有；此前 3 行运行 4 行内核，5-7 行运行 8 行内核：Flash Next 上 5 行的验证窗口 12.4 -> 11.3 ms，`--spec-draft 4` 的长生成 221-231 -> 234-236 tok/s）。RTX 5090，4 行，冷权重：Flash Next 的 2560 -> 10240 IQ4_XS 投影 33.4 -> 11.1 us，Q3_K 32.8 -> 15.4 us，2560 -> 151643 的 Q5_K LM head 298 -> 208 us。自 2026-11-09 起，4 行及以上的窗口把各行的 warp 部分和合并为一次 reduce-scatter，使用同一个 xor butterfly（结果逐位相同）：Flash Next 的 4 行验证窗口 10.02-10.03 -> 9.89-9.94 ms | 开启 | - | 否 |
| `TS_DSV4_GEMV_BLOCK` | 直接 CUDA 引擎中 256 个以上输出的 F32/BF16 GEMV（MoE router） | `0` 保持每个输出一个 warp。默认：每个输出一个 256 线程的 block，各 warp 的和按顺序相加（BF16 与 F32 形式结果逐位一致）。Flash Next 的 512 x 2560 BF16 router，4 行：RTX 5090 上 12.1 -> 4.4 us | 开启 | - | 否 |
| `TS_DSV4_SELECT_KEYS` | 直接 CUDA 引擎中不超过 512 个专家的 MoE top-k 选择（Flash Next、DeepSeek V4/V4.1 与 GLM 的非 hash 层） | `0` 保持浮点 argmax 的 warp 选择。默认：每个排序分数变成一个 64 位键（保序的分数位在上，专家编号的补码在下），每次选取是每步两次 shuffle 的键最大值，胜出的 lane 用树形归约重取自己的最佳值，lane k 保存第 k 个选择。选择与权重逐位相同。RTX 5090，Flash Next（512 个专家，top 10），graph 捕获：每层 4.83 -> 3.11 us；长生成 4 行验证窗口 10.10 -> 10.03 ms，237-238 -> 240 tok/s，输出完全一致 | 开启 | - | 否 |
| `TS_MOE_GATEUP_HALF` | 直接 CUDA 引擎中 decode 大小的专家 gate/up 投影 | `0` 保持每个 gate/up 行一个 warp。默认：每行 16 个 lane、每个 warp 两行（Q2_0 行按 32 值单元）。Qwen3.8 Flash Next 一个 decode 步的十个 Q2_0 专家：RTX 5090 上 22.7 -> 4.4 us；4 行窗口的 40 个 slot 44 -> 23 us | 开启 | - | 否 |
| `TS_Q4E_MTP_KV_ONLY` | `--backend cuda` 上带 MTP 头的 Qwen 3.8 Flash Next | `0` 让 MTP 层对追赶（catch-up）的每一行完整运行。默认：追赶的行只写入自己的 K/V，层的其余部分（注意力、专家、输出头）只在最后一行起草时为它运行（Strata 的 `record_front` / `record_rest`）。一个 16K prompt 的追赶原先耗时 320 ms，主要是覆盖整个 prompt 的注意力和每一行的专家；现在约 45 ms。只影响草稿：每个 token 都由验证决定 | 开启 | - | 否 |
| `TS_Q4E_MTP_CHAIN` | `--backend cuda` 上带 MTP 头、单 GPU、无采样惩罚的 Qwen 3.8 Flash Next | `0` 每个草稿 token 调用一次头部，草稿 logits 返回主机。默认：一个窗口的草稿作为设备上的一条链运行（`IDraftHead.DraftChain`）：每一步的首个最大值 token 及其 top-10 置信度由挑选内核给出，并在没有主机往返的情况下送入下一步，逐 token 门限在主机上截断窗口。RTX 5090、ISTA `Q2_0`：每个窗口的起草时间从 2.1-2.2 ms 降到 1.9-2.1 ms；token 与逐步调用相同 | 开启 | - | 否 |
| `TS_Q4E_MTP_CHAIN_SPLIT` | `--backend cuda --layer-split 2` 及以上、带 MTP 头的 Qwen 3.8 Flash Next | `0` 让 token 嵌入只保留在第一块 GPU 上：草稿在那里嵌入后经主机交给 MTP 块所在的 GPU，每个 token 调用一次头部，MTP 块也无法在自己的设备上跟随提示词（`TS_Q4E_MTP_FOLLOW`）。开启时，最后一块 GPU 在 MTP 块旁保留第二份 token 嵌入（Q8_0 为 0.67 GiB），一个窗口的草稿在该设备上成链运行，追赶行也在该设备上嵌入 | ON | 未注册 | 否 |
| `TS_Q4E_MTP_FOLLOW` | `--backend cuda` 上带 MTP 头的 Qwen 3.8 Flash Next（提示词按 ubatch 顺序运行：专家全部驻留，或层切分） | `0` 把投机请求提示词每一行的输出流交给主机，再由主机交回给 MTP 块做追赶（每个 token 来回各 40 KB，前向为此等待）。开启时，MTP 块在自己的设备上随每个 ubatch 完成而接收单独请求的提示词（`IDraftHead.DraftSelfCatchUp`）：把每一行的 token 与前一行的流拼接并写入自己的 K/V；提示词走普通路径（层切分下各块保持在途），第一个草稿从这些行继续。两块 H100 NVL、Q8_0、`--layer-split 2`：19K token 的提示词 6,670 对 4,070 tok/s，40K 的 6,066 对 3,819，草稿接受率不变（80-85%） | ON | 未注册 | 否 |
| `TS_Q4E_MTP_PROMPT` | 同上 | `0` 不让 MTP 块以提示词为条件：所有提示词按普通方式读取，MTP 块在第一个解码步骤重启，与并发请求或复用前缀的请求相同（19K token 提示词之后接受率 67-72%，对 82-90%） | ON | 未注册 | 否 |
| `TS_Q4E_PIPELINE_OVER_SPEC` | `--backend cuda --layer-split 2` 及以上、带 MTP 头的 Qwen 3.8 Flash Next | `0` 让两个及以上请求优先走批量投机验证，草稿头随之关闭流水线解码组（与之前相同）。开启时，单独的请求进行投机，两个及以上的请求作为两个在途的组普通解码（`IPipelinedBatchedDecode.PipelinedDecodeOutrunsSpeculation`）：两块 H100 NVL、Q8_0、贪心，2 / 4 / 8 / 16 / 32 路 241 / 389 / 560 / 771 / 1,019 tok/s，对批量验证的 209 / 273 / 363 / 470 / 449；单路两者均为 178；token 相同 | ON | 未注册 | 否 |
| `TS_Q4E_MTP_D2D` | `--backend cuda` 上带 MTP 头、按组读取提示词的 Qwen 3.8 Flash Next | `0` 让每个追赶行的主干流都从主机上传。默认：提示词之后的 MTP 追赶从设备上最后一个提示词组的缓冲区读取该组各位置的流（与主机副本中的值相同，因此草稿逐位一致）。RTX 5090：4K 或 16K 提示词之后 4,000 行的追赶 22.3 -> 6.0 ms；decode 4K 189-199 -> 198-211 tok/s，16K 231-233 -> 247 tok/s | 开启 | - | 否 |
| `TS_Q4E_DRAFT_PICK_SPLIT` | `--backend cuda` 上带 MTP 草稿链的 Qwen 3.8 Flash Next | `0` 在一个 block 中对整个词表挑选草稿步的 token（RTX 5090 上每步 115 us）。默认：每 1,024 个 logits 一个 block 保留其片段中最大的十个，再由一个 block 合并（25 us）；token 和置信度逐位一致 | 开启 | - | 否 |
| `TS_Q4E_DRAFT_VOCAB` | `--backend cuda` 上带 MTP 草稿链的 Qwen 3.8 Flash Next | 草稿 head 计算的 token id 范围 `[0, N)`；`0` 计算整个词表。默认 151643，即 Qwen2 的基础词表（之后的 id 是后来加入的多语言 token 和特殊 token：英文/中文文本中占 0.1-0.2%，纯中文 0.01%）。草稿只是提议，验证窗口对所有 token 打分，所以答案不变（不依赖历史的运行中开关前后文本完全一致），接受率变化在噪声之内（英文 58.3 vs 58.4%，中文 55.3 vs 56.0%，代码 85.9 vs 85.2%）。草稿步的 LM head 只读 61% 的行：RTX 5090 上约 270 -> 约 165 us；长生成 202-222 -> 214-234 tok/s | 151643 | - | 否 |
| `TS_Q4E_GREEDY_VERIFY` | `--backend cuda` 上使用普通贪心采样器的 Qwen 3.8 Flash Next 投机解码 | `0` 把每个验证行的 logits 返回主机。默认：每行的首个最大值 token 由捕获的验证窗口内的 argmax 内核给出，窗口的 logits（每行 1 MB）不再经过总线；token 相同 | 开启 | - | 否 |
| `TS_MOE_TIER_ADAPT_PROMPT_SWAPS` | `--backend cuda` 上启用自适应驻留的 Qwen 3.8 Flash Next | prompt（超过 8 行）之后一轮最多交换多少个专家（默认 64；`0` 关闭从 prompt 学习）。在设备上统计 prompt 的专家选择，按 128 行的分量计入，并在回答 decode 之前让它最常选中的主机专家与最少选中的显存专家互换。RTX 5090、ISTA `Q2_0`、MTP：16K prompt 之后每个 4 行窗口交给 CPU 的专家从 156-200 个降到 64-136 个，decode 从 132-153 升到 152-164 tok/s；4K prompt 之后基本不变。文本也会取决于之前的 prompt | `64` | 未注册 | 否 |
| `TS_MOE_TIER_ADAPT_ASYNC` | `--backend cuda` 上启用自适应驻留和主机专家的 Qwen 3.8 Flash Next | `0` 让每轮自适应在 GPU 空闲时复制专家（原先的阻塞交换，每轮 8 个）。默认：一轮的复制在独立的 stream 上分三步完成，每步相隔一个前向边界（两个专家先复制到暂存槽、表切换到副本、副本复制到最终位置、表切回），decode 步或验证窗口与之并行，一轮最多可以交换 254 个（Strata 的重叠自适应层；`TS_MOE_TIER_ADAPT_SWAPS` 自 2026-11-10 起默认 256，此前为 64）。RTX 5090、ISTA `Q2_0`、MTP：4K prompt 之后每个验证窗口的 CPU 专家从 115-171 降到 31-146，16K 之后从 69-98 降到 26-33；长生成 154-182 -> 174-179，4K decode 135-160 -> 137-188，16K 167-184 -> 183-198 tok/s | 开启 | 未注册 | 否 |
| `TS_MOE_ROUTE_TAIL_SKIP` | 带主机专家的 MoE 层：启用专家驻留时 `--backend cuda` 上的 Qwen 3.8 Flash Next，以及 `ggml_metal` / `ggml_cuda` 上各模型的加速器专家分层；解码步与验证窗口（最多 8 行） | Strata 的 route-tail skip（`STRATA_ROUTE_TAIL_SKIP=7`，其默认值）：在 decode 步或验证窗口中，窗口每一行都以该 0 起始排名或更低排名（7：Flash Next 10 个中的第 8 到第 10 个）路由到的缺失专家（没有显存槽位）既不由 CPU 计算也不等待；它的路由不贡献任何值，权重不重新归一化，因此回答成为近似结果。所有专家都驻留、至少 `TS_MOE_TAIL_SKIP_MAX_CACHED` 百分比的专家有槽位、或处理 prompt 时不启用。`0` 计算每条路由。在 `--backend cuda` 上，门铃内核据此决定一层是否需要唤醒主机，主机服务器据此处理被唤醒层的槽位。RTX 5090、ISTA `Q2_0`、MTP，开启对比 `0`（同一构建、同一客户端）：长生成 153-188 对 140-175 tok/s，4K 128-176 对 119-144，16K 163-183 对 150-170；4K prompt 之后每个验证窗口交给 CPU 的专家 128-140 个对 185-242 个。 在 GGML 分层上，被跳过的专家也不读入窗口槽位：Flash Next、ISTA `Q2_0`、M5 Pro、DFlash2，交替的关/开测量：每次解码的窗口读取 88-114 -> 33，短对话 22.4-24.6 -> 23.5-25.9 tok/s，长生成均值 42.1 -> 43.4 tok/s。依用户 2026-11-08 的决定默认开启。 | `7` | 未注册 | 否 |
| `TS_MOE_TAIL_SKIP_MAX_CACHED` | 同上 | 至少这个百分比的路由专家拥有显存或分层槽位时关闭 route-tail skip（Strata 的规则：缓存缺失多时跳过才划算）。32 GB RTX 5090 上的 Flash Next ISTA `Q2_0` 驻留 69% | `90` | 未注册 | 否 |
| `TS_Q4E_HC_Q8` | GGML 后端和 `--backend cuda` 上的 Qwen 3.8 Flash Next | `1` 把超连接混合器的 BF16 down/up 矩阵按 Q8_0 读取，加载时转换一次（Strata 的默认读取方式，`STRATA_HC_Q8`）：每个 token 从 1.26 GB 降到 0.67 GB。会改变数值（Strata：相对 BF16 的平均 KL 为 0.0007）。M5 Pro、ISTA Q2_0、普通 decode：22.8/16.3/23.1 提升到 24.1/16.8/24.5 tok/s（code/prose/list），4 个 prompt 中 3 个文本完全相同；Metal 的专家规划会计入这些 Q8_0 副本，因为它们与文件页面同时驻留。RTX 5090、`--backend cuda`：94.6/65.6/97.3 提升到 106.0/70.4/109.1 tok/s。已把这些矩阵存为 Q8_0 的文件（Unsloth 的 UD 包）不受影响 | 关闭 | - | 否 |
| `TS_METAL_MOE` | `ggml_metal` 上的 Qwen 3.8 Flash Next，专家分层（1-8 行的 decode 步或验证窗口） | 默认开启；`0` 让一层加速器分层持有的路由仍用 ggml-metal 的 MUL_MAT_ID。开启时每个 MoE seam 用 TensorSharp 的分组专家内核（移植自 Strata 的 MIT Metal 内核，`TensorSharp.GGML.Native/tsm/`）计算这些路由：每个不同槽位的 gate/up/down 行对窗口里路由到它的所有行只读一次，在 ggml-metal 自己的命令队列上编码。与 MUL_MAT_ID 相差 1e-6（相对），与双精度参考相差 2e-7。M5 Pro、ISTA `Q2_0`：4 行验证窗口 69.9 -> 59.8 ms，文本不变 | 开启 | - | 否 |
| `TS_METAL_MOE_TIMING` | `TS_METAL_MOE` | `1` 每 96 层打印一次分组专家内核每层的 GPU 时间（诊断用） | 关闭 | - | 否 |
| `TS_METAL_SEAM_EVENTS` | `TS_METAL_MOE`（分层的 decode 步或验证窗口） | 默认开启；`0` 让每个分层 seam 提交并等待。开启时一次前向的各段提前排入 ggml-metal 的队列，每个 seam 通过共享事件交接（主机读取路由、计算没有槽位的路由、写入计划并放行 GPU）；logits 逐位一致。M5 Pro、ISTA `Q2_0`，配合 `TS_METAL_KEEPALIVE`：1 行前向 38 -> 33 ms | 开启 | - | 否 |
| `TS_METAL_KEEPALIVE` | `TS_METAL_SEAM_EVENTS` | 默认开启；`0` 关闭。分层前向运行期间，第二个 Metal 队列上的短自旋内核让 GPU 保持清醒，seam 握手约 83 us 而不是约 190 us（M5 Pro） | 开启 | - | 否 |
| `TS_METAL_SEAM_PROFILE` | `TS_METAL_SEAM_EVENTS` | `1` 每 32 次前向按 GPU 时钟打印 ggml 各段与 seam（其内核与等待主机）的时间（诊断用） | 关闭 | - | 否 |
| `TS_Q4E_PLE_RAM` | 在 `--backend cuda` 上运行的 Qwen 3.8 Flash Next | `1` 在加载后把 PLE n-gram 表读入进程内存（Strata 的 `--ple-io ram`），`0` 继续从文件映射读取行。未设置时：进程仍可占用的内存不少于表大小加 6 GB 时放入内存（Linux；MemAvailable 受内存 cgroup 上限约束，因此容器的限额、其 tmpfs 文件和引擎锁页的专家都会计入）。从映射读取时，页面未缓存的行每个 token 都要访问一次存储：在网络文件系统上，decode 线程会因此等待，GPU 空闲 | 自动 | - | 否 |
| `TS_MOE_HOT_TIER_GB` | 同上 | 固定（按画像排序）槽位的字节预算，单位 GB | 旧方案会常驻的字节数（48 GiB Mac 上为 14.4 GB） | - | 否 |
| `TS_MOE_WINDOW_SHARE` | 同上，统一内存（`ggml_metal`） | 分层预算中分给窗口槽位（最近最少使用，随文本路由填充）而不是按 profile 排名的固定槽位的比例，0 到 0.9。显式设置 `TS_MOE_HOT_TIER_GB` 时保留其自己的划分。ISTA `Q2_0`、48 GB M5 Pro：窗口 0.3 -> 3.1 GB 让 4 行验证一次从 52.8-59.0 降到 49.4-51.0 ms（窗口读取 14-19 次而不是 26-47 次，没有主机路由） | 0.12 | - | 否 |
| `TS_HOST_MOE_IO_CACHE_GB` | 启用卸载的 MoE 模型，Apple | 显式读取专家（并行 `pread`，`F_NOCACHE`）所用的内存：有专家层时 7/8 作为窗口槽位，其余作为溢出路由的主机缓存；否则作为主机计算层读取的主机缓存。同时让 prefill 用 `pread` 直接读入加速器的暂存内存。模型放得进内存时默认关闭（页缓存即可）；`0` 改回读映射 | 扣除主机保留与所有常驻内容后剩余内存的 3/4 | - | 否 |
| `TS_HOST_MOE_STAGE_PREFETCH` | `ggml_metal` 上启用卸载的 MoE 模型，提示规模的批次 | 默认开启；`0` 在每个卸载层的 MoE 运行时才暂存其专家。开启时，加速器计算当前层期间，在线程上把下一个卸载层的专家整体暂存（分层复制与显式读取）到一个缓冲区，其 MoE 直接绑定；logits 逐位一致。Qwen 3.8 Flash Next、ISTA `Q2_0`、M5 Pro：关键路径上的暂存每 30 层约 600 -> 24 ms，每个 4096 token 的块快 0.5-0.7 s | 开启 | - | 否 |
| `TS_HOST_MOE_TIER_MAX_ROWS` | 带加速器专家分层的 MoE 模型（Qwen 3.8 Flash Next） | 分层仍在其槽位（固定槽位，加上把未命中专家读入窗口槽位）与主机之间拆分计算的最宽跨度；更宽的跨度把整层流式送入暂存区。窗口槽位位于主机可写内存（统一内存）时默认 2048。否则（独立显卡的分层没有窗口，宽跨度的未命中路由在主机上算）为 seam 开始流式传输的宽度减一（默认 127，即 `TS_HOST_MOE_DEVICE_MIN_BATCH` - 1），且不小于 64：既不拆分也不流式的跨度曾把每条路由都放在主机上算（RTX 5090 上 ggml_cuda 的 Flash Next：拆分止于 64 时，76 token 的提示首 token 需 1.1-1.4 s，120 token 需 2.2 s；拆分到 127 时为 0.55-0.77 s；四个并发的 77 token 请求从 5.2-6.1 s 降到约 1-1.7 s）。ISTA `Q2_0`、M5 Pro，冷启动 TTFT：88-473 token 的提示从流式 1.35-1.4 s 降到拆分 0.33-0.76 s；900 / 1770 / 3500 token 从 1.76-2.0 / 2.7 / 5.2 s 降到 1.33-1.55 / 2.55 / 5.1-5.3 s | 2048 / 127 | 未注册 | 否 |
| `TS_MOE_TIER_PREFETCH` | 带窗口的加速器专家分层的 MoE 模型（`ggml_metal` 上的 Qwen 3.8 Flash Next） | Strata 的预读：在每个解码/验证接缝处，用下一层的路由器作用于本层的 FFN 输入，预测下一层可能缺失的专家；每行前 N 个在 GPU 计算时于后台读入下一层分层的窗口槽，下一个接缝先等待它们完成。结果精确（槽中是专家的原始字节）。M5 Pro、ISTA `Q2_0`：1-5 行时同步窗口读取减少 45-70%，每次前向的主机时间减少 40-60%；该处前向受 GPU 限制，吞吐量不变。`0` 关闭 | `6` | 未注册 | 否 |
| `TS_METAL_SEAM_N_CB` | `ggml_metal` 上带加速器专家分层的 MoE 模型（Qwen 3.8 Flash Next 的解码/验证前向） | ggml-metal 编码接缝前向每个分段时额外使用的 command buffer 数。分段已提前两段入队，第二个编码线程起不到隐藏作用；而 ggml-metal 在约 120 个节点的分段第 64 个节点处的切分恰好落在 GDN 层的 delta net 与其状态写回之间，GDN cache 融合无法跨越该切分。其他图保持 ggml-metal 自身设置（1）。M5 Pro、ISTA `Q2_0`：单行前向减少 1.9 ms；验证宽度与 4K prompt 不变 | `0` | 未注册 | 否 |
| `TS_METAL_MOE_MM` | `ggml_metal` 上带 Q2_0 专家、支持 Metal 4 tensor API 的 MoE 模型（M5 上 Qwen 3.8 Flash Next 的 prompt） | prompt 分块的路由专家（流式层）改用 Strata 的分组专家 GEMM，替代三个 `MUL_MAT_ID`：按专家把 (token, 专家) 条目排序成 16 行的 tile，gate + up + SwiGLU 在矩阵单元上一次完成，再做 down，最后按路由顺序对每个 token 加权求和。激活与 SwiGLU 乘积按矩阵单元的输入取 fp16（相对 double 参考 2-3e-4），贪心文本在近似平局处可能不同。其临时缓冲（4096 行分块约 0.8 GB）在每个 prompt pass 结束后释放。M5 Pro、ISTA `Q2_0`：4K 分块的专家耗时 1.66-1.70 -> 1.28-1.30 s；long_prompt_4k TTFT 5.74/5.58 -> 5.18/5.01 s，long_prompt_16k 22.9/22.7 -> 20.9/20.9 s；解码不变。`0` 保留 `MUL_MAT_ID` | 开启 | 未注册 | 否 |
| `TS_METAL_MOE_MM_SEAM` | `ggml_metal` 上带 Q2_0 专家、支持 Metal 4 tensor API、提示规模的卸载层走 `TS_METAL_MOE_MM` 的 MoE 模型（M5 上 Qwen 3.8 Flash Next 的 prompt） | 默认开启；`0` 保留往返路径。开启时，专家栈已由暂存预取备好的层在 seam 处直接计算专家：从 span 图中读取路由的 id 与权重和 FFN 输入，并就地写入 MoE 输出。原路径先下载输入，再上传到它自己的流式图，下载输出后再作为 MoE 输出上传：每层激活四次拷贝，4096 行时约 4 ms。同样的内核、同样的字节：结果逐位相同。M5 Pro、ISTA `Q2_0` + DFlash2：一个 4K pass 4.28-4.42 -> 3.79-4.10 s；long_prompt_4k TTFT 5.14/4.86 -> 4.45/4.33 s，long_prompt_16k 19.7/19.6 -> 18.4/17.9 s | 开启 | 未注册 | 否 |
| `TS_DFLASH_METAL_FUSE` | `ggml_metal` 上的 DFlash2 草稿模型 | 草稿模型的分组动态卷积在每个 kernel 中处理所有 tap（一次收集移位行、一次构建卷积核、一次乘积、对各 tap 用 ADD 链求和），不再每个 tap 一条链：乘积与求和顺序相同，草稿结果逐位一致。3 个 tap 时每次草稿少约 44 个 kernel；M5 Pro 上的 Qwen3.8 Flash Next DFlash2：草稿 6.62 -> 6.47 ms。`0` 保留逐 tap 的链 | 开启 | 未注册 | 否 |
| `TS_DFLASH_DRAFT_VOCAB` | 融合的 DFlash / DFlash2 草稿模型（不含 DSpine） | 一个 token id 文件（小端 int32，升序；即 Strata 的 `data/draft_vocab*.bin` 格式），草稿只对其中的 token 打分，而不是整个词表。加载时把目标模型的 LM head 行和 selector 表按子集顺序收集一次，草稿输出的 id 再映射回原 token。目标模型验证每个 token，所以输出仍然精确；但子集之外的 token 永远不会被起草。在 Flash Next（ISTA Q2_0 + DFlash2，48 GiB Mac）上，Strata 默认的 106K 子集把 8 行草稿从 5.0-5.4 ms 降到 3.8 ms。decode 变化：英文长生成 +4.5%，4K 提示 +0.9%，中文长生成 -6.1%（接受率 51% -> 46-49%），代码 -1.9%。Strata 的 40K 英文子集表现不稳定 | 未设置（完整词表） | 未注册 | 否 |
| `TS_DFLASH_RING_BLOCK` | 融合的 DFlash / DFlash2 草稿模型（单路与批量草稿，所有后端） | 把草稿块的 key 和 value 写入草稿模型环形缓存中该块自己的槽位（`position % ring_rows`），注意力只读环形缓存，不再在每一层把整个环形缓存与草稿块拼接。环形缓存在滑动窗口之外留有一个块加一个锚点的余量，块槽位原先只存放任何查询都看不到的位置；已提交位置的注入会覆盖这些行。草稿结果与拼接方式在浮点舍入范围内一致（选出的 token 相同，活跃的环形缓存行逐位一致；`GgmlOpsDFlashRingBlockTest`）。M5 Pro 上的 Qwen3.8 Flash Next DFlash2：草稿 6.47-6.53 -> 5.39-5.43 ms，long_generation 47.5-47.6 -> 48.1-48.3 tok/s。`0` 保留拼接方式 | 开启 | 未注册 | 否 |
| `TS_METAL_DF_TAIL` | `ggml_metal` 上目标模型输出头为 q5_K、选择器码本为 q4_K（rank 256）且 top-16 的 DFlash2 草稿模型（Qwen3.8 Flash Next） | 默认开启；`0` 保留 ggml 的算子。开启时草稿计算图止于最终行与选择器的隐藏投影，其余部分由 TensorSharp 的 kernel（`tsm/kernels/tsm_df_tail.metal`）在一个命令缓冲区内完成：在矩阵单元上对候选行计算输出头（直接读取 ggml 的 q5_K 块，激活为 fp16）、分两遍求每行的 top-16（参照 Strata 的 `df2_topk_part` / `df2_topk_merge`）、并用一个 kernel 计算选择器的整个转移网格。候选与网格与 ggml 路径一致，仅差输出头激活的 fp16 舍入（`TsmDfTailTests`：与基于 ggml 反量化的双精度参考相差 1e-7 以内）。M5 Pro，Flash Next ISTA Q2_0 + DFlash2：输出头 2.47 -> 1.96 ms，top-k 约 0.4 -> 0.14 ms；草稿 5.37-5.42 -> 4.95-4.99 ms，long_generation 48.1-48.3 -> 48.7-49.1 tok/s，输出与接受率一致 | 开启 | 未注册 | 否 |
| `TS_METAL_DF_TAIL_PROF` | `TS_METAL_DF_TAIL` | `1` 每 64 次草稿打印一次草稿尾部的加速器时间；`2` 把各阶段分别放进独立的命令缓冲区运行并分别打印（xprep、输出头、top-k、网格）（诊断用） | 关闭 | - | 否 |
| `TS_PLE_EXPLICIT_READ` | Qwen 3.8 Flash Next，PLE n-gram 表由启用显式专家 I/O 的 GGUF 提供 | 默认开启；`0` 让提示规模批次的行通过映射读取（每行一次缺页）。开启时 1024 行以上的 gather 在原生 I/O 线程组上每行一次 `pread`；logits 逐位一致。M5 Pro：一个 4096 token 块的 65K 行 1.1-3.1 s -> 0.53 s | 开启 | - | 否 |
| `TS_PLE_IO_THREADS` | 同上 | PLE 行读取同时进行的读请求数，使用独立线程组（接缝线程组仍按 `TS_HOST_MOE_IO_THREADS`：每次调用都会唤醒整个线程组）。每次读取在 SSD 上约等待 90 us，吞吐由并发读数决定：M5 Pro 上 4K prompt 的 65K 行在 12 个时 506 ms，32 个时 272 ms（64 个时 250 ms）；4K TTFT 减少 0.23 s，16K TTFT 24.2 -> 22.5 s | `32` | - | 否 |
| `TS_METAL_GDN_PROMPT` | `ggml_metal` 上的 Qwen 3.8 Flash Next，提示规模的 span（`TS_METAL_PROMPT_OPS_MIN_ROWS`） | 默认开启；`0` 保留 ggml 的算子。开启时，GDN 层的递推（卷积 + SiLU、q/k L2 归一化、门控、状态扫描、带门控的归一化）在两个图分段之间由移植自 Strata `gdn_prompt.metal` 的 TensorSharp 内核运行，替代 ggml 的转置、拼接、卷积、q/k/v 复制、头重复、gated delta net 和归一化；状态保持 ggml 的布局。仅舍入差异（测试夹具上 6.4e-5）。M5 Pro、ISTA `Q2_0`：一个 4096 token 块的计算 7.6 -> 6.3 s | 开启 | - | 否 |
| `TS_METAL_HC_PROMPT` | `ggml_metal` 上的 Qwen 3.8 Flash Next，提示规模的 span | 默认开启；`0` 保留 ggml 的算子。开启时，每个半层的超连接混合（sigmoid 门、乘积、流均值）和散射（`res + b * 2 sigmoid(inject / hc)`）各由 TensorSharp 内核一遍完成，替代 ggml 在 `[hc * n, T]` 上的算子链。仅舍入差异（测试夹具上 5.3e-5）。M5 Pro：配合 GDN 内核，一个 4096 token 块的计算 6.3 -> 5.2 s | 开启 | - | 否 |
| `TS_METAL_HC_READ` | `ggml_metal` 上支持 Metal 4 tensor API（M5）的 Qwen 3.8 Flash Next，提示规模的 span（`TS_METAL_HC_PROMPT` 开启时） | 默认开启；`0` 保留 ggml 的 norm 与矩阵乘及单独的混合遍。开启时，每个半层的超连接读取由 TensorSharp 内核一遍完成（参照 Strata 的 `gr_norm_h` / `gemm_f16` / `gemm_hc_up`）：各流的 norm 只以 fp16 写一次，down 投影与 inject 权重合成一个 fp16 GEMM，up 投影的收尾直接完成流混合，因此既不写 F32 的 norm 输出，也不写 up 投影的输出。权重（F32、F16、BF16 或 Q8_0）每个分块反量化为 fp16。fp16 操作数：相对 double 参考 3.1e-4 以内；贪心文本在近似平局处可能不同。M5 Pro、ISTA `Q2_0` + DFlash2：4K 分块的稠密部分 3.09-3.40 -> 2.72-3.02 s；long_prompt_4k TTFT 5.42/5.39 -> 4.71/4.71 s，long_prompt_16k 21.4/20.9 -> 19.4/19.4 s；解码不变 | 开启 | - | 否 |
| `TS_METAL_QSA_SELECT` | `ggml_metal` 上的 Qwen 3.8 Flash Next，窗口宽于 QSA 层 top-k 的提示规模 span | 默认开启；`0` 保留 ggml 的算子。开启时，QSA 层的 cell 选择（各头 relu 分数之和加每块的 bias，然后每个 token 选出允许的最佳 `top_k + ratio - 1` 个 cell）由 TensorSharp 内核以逐 token 基数选择完成并直接写出注意力掩码，替代 ggml 把每个 cell 的分数收集到 `[cells, T]`、两次转置、掩码的转换与相加以及对每一行的 top-k。所选 cell 相同，只有截断处并列的 cell 按最小下标优先选取（ggml 的 top-k 按原子操作的顺序选取） | 开启 | - | 否 |
| `TS_METAL_QSA_ATTN` | `ggml_metal` 上的 Qwen 3.8 Flash Next，使用 QSA 选择内核、F16 KV 缓存、窗口达到 `TS_METAL_QSA_ATTN_MIN_CELLS` 的提示规模 span | 默认开启；`0` 保留 ggml 对所选掩码的 flash attention。开启时，QSA 层的注意力只读取每个 token 所选的 cell（选择内核把它们写成升序列表）：每个（token，kv 头）一个线程组，该 kv 头的各查询头作为矩阵乘的行，cell 每次收集 32 个。其开销与窗口宽度无关，ggml 的开销随窗口增长。与 ggml 在相同 cell 上相比仅有舍入差异（测试夹具上 4.7e-5） | 开启 | - | 否 |
| `TS_METAL_QSA_ATTN_MIN_CELLS` | `ggml_metal` 上的 Qwen 3.8 Flash Next | 只读所选 cell 的注意力从多宽的窗口（cell 数）起运行：tensor API 内核默认 8192，simdgroup 内核默认 12288。M5 Pro，一个 4096 token 块的 12 个 QSA 层，相对 ggml 的 flash attention：tensor API 在 4K 窗口 +0.18 s，8K -0.20 s，12K -0.57 s，16K -0.97 s；simdgroup +0.40 / +0.20 / -0.23 / -0.62 s | 8192 / 12288 | - | 否 |
| `TS_METAL_QSA_ATTN_MM` | `ggml_metal` 上的 Qwen 3.8 Flash Next，具备 tensor API 的 Metal 4 设备（M5、M6、A19、A20；`TS_METAL_TENSOR_ENABLE=1` 可在其他 Metal 4 设备上尝试） | 默认开启；`0` 让只读所选 cell 的注意力运行在 simdgroup 内核上。开启时它运行在 Metal 4 tensor API 上（MetalPerformancePrimitives 的 `matmul2d`，采用 ggml tensor flash attention 的结构，所选 cell 的 key 与 value 收集到线程组内存），并编译为独立的库，使没有该 API 的设备或 SDK 不影响其他内核：16K 窗口下 4096 token 块的一层 79 -> 42 ms（M5 Pro） | 开启 | - | 否 |
| `TS_METAL_TENSOR_ENABLE` | TensorSharp 的 Metal tensor API 内核 | `1` 在早于 M5 的 Metal 4 设备上尝试 tensor API 内核（ggml 的规则：从 M5 起才划算） | - | - | 否 |
| `TS_METAL_PROMPT_OPS_MIN_ROWS` | `ggml_metal` 上的 Qwen 3.8 Flash Next | span 从多少行起运行上述提示内核（默认 64）。更少的行以及验证宽度的 span 使用 ggml 的算子 | 64 | - | 否 |
| `TS_HOST_MOE_STREAM_PREAD` | 启用卸载的 MoE 模型，Apple | `0` 让 prefill 流式传输专家时改回缺页读入映射再拷贝，而不是直接读入 CPU 可见的加速器内存 | 启用 | 未注册 | 否 |
| `TS_HOST_MOE_IO_THREADS` | 同上 | 显式 I/O 线程组的读取线程数（调用之间休眠） | `12` | - | 否 |
| `TS_HOST_MOE_IO_WIRE` | 同上 | `0` 让显式读取的专家缓存保持可换页。macOS 上默认（iOS 不变）：缓存槽位被锁定在内存中（`mlock`），与 Strata 锁定其 arena 的做法相同。专家权重几乎无法压缩（1.2:1），macOS 压缩一个槽位只能腾出很少内存，之后每次命中都要先解压；锁定后 macOS 改为从文件缓存和其他程序可压缩的内存中腾出空间。DeepSeek V4.1 UD-IQ2_XXS、48 GiB Mac 且开着其他应用：不锁定时 21 GB 缓存中有 15-17 GB 在压缩器里，decode 每 token 1,000-1,300 ms；锁定后仅 0.15 GB，453 ms | 启用 | 未注册 | 否 |
| `TENSORSHARP_MOE_PROFILE_DIR` / `TENSORSHARP_MOE_PROFILE` | 专家会到达主机接缝的 MoE 模型 | 路由画像（每层专家被选中的次数，跨运行累加）的存放位置：按模型分文件的目录，或一个显式文件（可来自同一模型的其他量化版本） | `$XDG_CACHE_HOME/tensorsharp/moe-routing` 或 `~/.cache/tensorsharp/moe-routing` | - | 否 |
| `TS_MOE_TIER_STATS` | 同上 | 诊断：`1` 在模型卸载时打印专家槽位的命中、窗口读取（及其中由批量 verify 在后台完成的次数）与溢出次数、主机缓存的计数以及自适应槽位的替换次数；`ggml_metal` 上的 DeepSeek V4.1 则打印 CPU lane RAM 缓存的命中与未命中 | 关闭 | 未注册 | 否 |
| `TS_MOE_TIER_ADAPT` | 独立显存（`ggml_cuda`）上 Qwen 3.8 Flash Next 的专家分层，以及 `--backend cuda` 上启用主机专家的专家驻留 | 默认开启；`0` 让每层的固定槽位保持路由 profile 决定的位置。开启时每层的固定槽位跟随当前对话（Strata 的自适应专家缓存）：seam 为每个专家维护衰减的路由计数，每 32 次调用把最多两个明显更热的主机端专家换入最冷的槽位，在段与段之间从已预热的映射复制。在 `--backend cuda` 上，门铃内核在设备上统计每层的专家选择；每 `TS_MOE_TIER_ADAPT_EVERY` 次前向（默认 4，decode 步、MTP 验证窗口和批量步都算）最多进行 `TS_MOE_TIER_ADAPT_SWAPS` 次交换（与 decode 重叠时默认 256，即一轮暂存槽位能承载的数量，Flash Next 为 254，见 `TS_MOE_TIER_ADAPT_ASYNC`；阻塞时默认 8；在各层之间按收益从大到小）。每次交换把一层中最常被选的主机专家与最少被选的 VRAM 专家互换位置；槽位里的字节换到主机专家的打包位置，内存占用与规划一致。RTX 5090、UD-Q2_K_XL、固定 profile：普通 decode 56-62 对 47-53 tok/s；ISTA `Q2_0` 加 MTP：每个 4 行验证窗口交给 CPU 的专家从 229 降到 123，长生成从 115-125 升到 132-167 tok/s。路由的数值取决于它的位置，因此贪心文本取决于之前生成的内容；需要可复现输出时设为 `0`。统一内存上的分层本来就按需填充窗口槽位 | 开启 | 未注册 | 否 |
| `TS_HOST_MOE_OVERLAP` | 带分层主机 seam 的 MoE 模型 | `0` 在主机算完冷路由之后才运行 seam 中加速器分层的路由。默认：先异步启动热路由的图片段，主机同时计算冷路由；结果逐位一致（RTX 5090：52 对 50 tok/s） | 启用 | 未注册 | 否 |
| `TS_HOST_WARM_WAIT` | Qwen 3.8 Flash Next 加载 | `1` 在首个请求之前等待后台对 PLE 表和主机端专家（按路由频率从高到低，限于 RAM 余量）的预热完成，而不是边读边服务 | 关闭 | 未注册 | 否 |
| `TS_DSV41_HOST_LANE` | 专家卸载到主机的 DeepSeek V4.1：`ggml_metal`，以及主机专家超过进程可常驻内存（非 Apple 平台为 cgroup 上限）60% 时的 `ggml_cuda` | `0` 让主机专家重新以 ggml CPU 算子运行；`1` 在独立显卡上强制使用 lane。默认：路由专家用 TensorSharp 的主机 decode 内核计算；专家通过显式 `pread` 读入 RAM 缓存（`TS_HOST_MOE_IO_CACHE_GB`，默认为扣除保留量以及统一内存上已上传权重后剩余 RAM 的 3/4；Linux 上 RAM 取 cgroup 上限，并以 `POSIX_FADV_RANDOM` 读取文件），而不是对 157 GB 的映射触发缺页。不超过 8 个 token 的 decode 与 verify 步骤始终走这条路径；有缓存时 prefill 也走它，按 8 行分块，前一块计算时读取下一块的专家。48 GiB M 系列 Mac、UD-IQ2_XXS：每 token 416-446 对 759 ms | 启用 | 未注册 | 否 |
| `TS_DSV41_LOOKAHEAD` | 同上，且启用 RAM 缓存 | `0` 关闭 Strata 的 lookahead：每个 lane 层把下一层的路由器（F32，或 ISTA-DASLab 文件使用的 BF16）应用到自己的 FFN 输入上，在 GPU 运行下一层注意力时把选中的专家读入缓存（预读的切片中 77.9% 被用到） | 启用 | 未注册 | 否 |
| `MAX_CONTEXT` | 长文本 / 上传文本 | 硬上下文上限。设置了就是硬性要求：缓存放得下就照办，放不下就带着数字拒绝。不设置时，GGUF 宣称的长度只是上限，加载器会按设备真正装得下的量来定——GLM-5.2 宣称 1M token，那是约 93 GiB 的 KV | 模型默认值（是上限而非承诺） | `4096`, `8192`, `16384` | 是 |

## Prefill / Decode 调优

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_PREFILL_CHUNK` | 在 GPT OSS、Qwen 3.5 / 3.6 family 长上下文功能上 sweep；运行时 Gemma 4、Nemotron-H 以及所有共享解码运行时家族（Mistral 3、Hunyuan Dense）也会读取 | 分块 prefill 大小 | 架构默认值 | `256`, `512`, `1024` | 是 |
| `TS_GGML_ASYNC_COMPUTE` | GGML 后端 | 异步 compute 提交 | `ggml_metal` 上启用（`0` 关闭），其他 GGML 后端关闭 | `0`, `1` | 是 |

## 多模态

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `VIDEO_SAMPLE_FPS` | 视频功能 | 按时间抽帧的每秒帧数 | `1` | `1`, `2` | 是 |
| `VIDEO_MAX_FRAMES` | 视频功能 | 抽取视频帧上限 | 不限制 | `8`, `16` | 是 |
| `TS_NEMOTRON_IMAGE_MAX_TILES` | Nemotron-H 图像功能 | 最大图像 tile 数 | 架构默认值 | `4`, `8`, `12` | 是 |

## MLX 专属

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_MLX_HALF_MATMUL_MIN_ROWS` | MLX affine 量化 matmul | scales 为 F16 的 matmul 从多少行起使用 F16 激活；`0` 使所有 matmul 保持 F32 | `1`（全部） | `0`, `1`, `32` | 否 |
| `TS_MLX_Q6K_MATVEC_MAX_ROWS` | MLX 上的精确 Q6_K matmul | 行数不超过该值时，Q6_K 以矩阵-向量乘运行（移植自 ggml-metal 的 `kernel_mul_mv_q6_K_f32`）；更多行且不超过 `TS_MLX_IQ_FEWROW_MAX_ROWS` 时走少行核函数，再多则把 Q6_K 反量化为 F16（每片不超过 256 MB）并用 MLX 的 GEMM 相乘 | `4` | 整数 >= 0 | 否 |
| `TS_MLX_IQ4XS_MATVEC_MAX_ROWS` | MLX 上的 IQ4_XS matmul | 行数不超过该值时，IQ4_XS 以矩阵-向量乘运行（移植自 ggml-metal 的 `kernel_mul_mv_iq4_xs_f32`）；更多行且不超过 `TS_MLX_IQ_FEWROW_MAX_ROWS` 时走少行核函数，再多则把 IQ4_XS 反量化为 F16（每片不超过 256 MB）并用 MLX 的 GEMM 相乘 | `4` | 整数 >= 0 | 否 |
| `TS_MLX_IQ_FEWROW_MAX_ROWS` | MLX 上原始 i-quant / K-quant 的 matmul（IQ2_XXS、IQ2_XS、IQ2_S、IQ3_XXS、IQ3_S、IQ1_S、IQ1_M、IQ4_XS、Q2_K、Q3_K、Q6_K） | 少行核函数接受的最大行数（从 2 行起），用于推测解码的验证窗口和较短的提示词或提示词分块。8 行以内（以及没有张量 API 时的所有行数）：ggml-metal 的少行 MMA 布局，threadgroup 内各 simdgroup 切分 K，每个 lane 把自己的权重只解码一次、直接放进 simdgroup 矩阵寄存器（F32）；Qwen3.8-27B-UD-IQ3_XXS 在 M5 Pro 上 8 行前向 128 -> 79 ms（ggml_metal 87）。9 行起（`TS_MLX_FEWROW_TENSOR` 生效时）同样切分 K，乘法改在张量 API 上做（F16 操作数，F32 求和）：16 行 151 -> 82 ms（ggml_metal 153），32 行 599 -> 104（291），512 行 1377 -> 1205（1289）。`0` 让这些行数改走旧的 simdgroup 与逐输出核函数 | 有张量 API 时 `1024`，否则 `31` | 整数 >= 0 | 否 |
| `TS_MLX_FEWROW_TENSOR` | 上面的少行 matmul 在 9 行起、支持 Metal 4 张量 API 的 Apple GPU 上（M5 及以后；MLX 0.32+） | 改在 GPU 核心的矩阵单元上运行：8 行以上 F32 simdgroup 乘法与反量化 GEMM 每次调用都要复制一份 F16 权重，这两者都用不到矩阵单元。每个 simdgroup 把 32 个权重行的一个 64 值块解码进自己的 tile，再与 16、32 或 64 行输入相乘。启动后构建并运行一次核函数来判定；没有该 API 的设备或 SDK 继续使用 simdgroup 核函数（以及 31/32 行的默认值）。`0` 在所有设备上都不用它 | 启用（`1`） | `0`、`1` | 否 |
| `TS_MLX_GDN_ROWS_FUSED` | MLX 上 Qwen 3.5-3.8 的 GatedDeltaNet，在前向需要每一行的递归状态时（推测验证窗口） | 用一个 kernel 跑完整个窗口的递归，并边算边写出每一行的状态，而不是每行调用一次递归；部分接受的验证会回滚到这些逐行状态。`0` 恢复为每行一次调用 | 启用（`1`） | `0`、`1` | 否 |
| `TS_MLX_IQ_DEQUANT_GEMM_MIN_ROWS` | 同上原始类型（Q6_K 与 IQ4_XS 用各自的开关） | 从该行数起把权重反量化为 F16（分片）并用 MLX 的 GEMM 相乘，而非少行核函数：Qwen3.8-27B-UD-IQ3_XXS 2.3K token 预填充 74 s -> 6 s；2048 行时比张量 API 少行核函数快 5%，1024 行时持平。`0` 保留 simdgroup 核函数 | `TS_MLX_IQ_FEWROW_MAX_ROWS` 加一（有张量 API 时 `1025`，否则 `32`） | 整数 >= 0 | 否 |
| `TS_MLX_AFFINE_FEWROW_MAX_ROWS` | MLX 仿射权重（Q4_K、Q8_0、Q4_0/Q4_1 以及 MLX 原生 4 位和 8 位文件，F16 缩放） | 这类 matmul 走上面少行核函数的最大行数（从 2 行起）：按所在组的缩放与偏置解码打包的值，而不是用 MLX 的 `quantized_matmul`（在 M5 Pro 上行数很少时它只有约 120 GB/s）。Qwen3.8-27B 的 DFlash2 草稿器（Q4_K）：每次起草 17.6 -> 13.4 ms，长生成 27-28 -> 29-32 tok/s；Gemma 4 E2B Q8_0：8 行 28 -> 22 ms，128 行 87 -> 68（256 行起 MLX 更快）。`0` 让所有行数都交给 MLX | 有张量 API 时 `128`，否则 `8` | 整数 >= 0 | 否 |

## 矩阵外的纯 C# CPU 后端变量

这些用于调节 `--backend cpu` 背后的常驻工作线程池、量化权重处理与托管 SIMD 内核。它们是真实的
运行时开关，但没有注册进 `EnvVarMatrix.All`，默认的 TestMatrix 配置也不会扫它们。取 `0` 的开关会在
同一个二进制里恢复旧的代码路径，用于 A/B；正确输出不依赖其中任何一个。下文引用的实测数据如无特别说明，
均来自一台 i7-11800H（8 核 16 线程、AVX-512、32 GB）。

**运行哪套指令集。** 这个后端的所有手写内核都从同一个判定（`TensorSharp.Cpu.CpuIsa`）取得指令集，
因此同一台主机不会出现某类内核用 AVX-512、另一类用 AVX2 的情况：CPU 具备 AVX-512 F/BW/DQ 且运行时对
`Vector512` 做了硬件加速时用 AVX-512（在某些 512 位向量更慢的处理器上，以及 `DOTNET_EnableAVX512=0` 时，
JIT 不启用它），否则用 AVX2+FMA；没有 AVX2 时用可移植的 Vector128 / 标量代码（包括 ARM64，量化 matmul
在那里仍走逐行路径）。

**测试 AVX2 路径。** `TS_CPU_DISABLE_AVX512=1` 让手写的 AVX-512 内核在 AVX-512 主机上以其 AVX2 形式运行。
它不会收窄 .NET 运行时本身（`TensorPrimitives`、普通拷贝、`Vector512.IsHardwareAccelerated`）。要模拟只有
AVX2 的主机，请用 `DOTNET_EnableAVX512=0` 启动进程；.NET 10 会忽略旧的 `DOTNET_EnableAVX512F=0`。
`DOTNET_EnableAVX2=0`（或 `DOTNET_EnableHWIntrinsic=0`）只留下可移植档；在那里，比较 GEMM 内核的单元测试
会报告为跳过，而不是在没有可比对象的情况下算作通过。AVX2 内核只在 AVX-512 机器上以这种方式跑过，
没有在只有 AVX2 的硬件上跑过。

**仍然是原生的部分。** 使用 `--backend cpu` 时，模型计算不加载任何原生库（没有 GgmlOps，也没有 CUDA）。
模型计算之外：桌面平台上图像文件的读写经由 Magick.NET（一个原生 ImageMagick 构建）；服务端启动时会探测
GGML 与 CUDA 后端，以列出这台机器能运行什么，与 `--backend` 无关。这两点都早于这些内核。

| 环境变量 | 适用范围 | 功能影响 | 运行时默认 | 扫描取值 | 默认是否扫描 |
|---|---|---|---|---|---|
| `TS_CPU_THREADS` | `cpu` 后端（100% 纯 C#） | 常驻工作线程池的宽度。这个池运行托管 matmul；Core 的 CPU 内核（`CpuStorage` 张量上的 F32 SGEMM、逐元素、norm、softmax、RoPE，经由 `TensorSharp.Models` 在模块加载时装上的 `CpuParallel` 钩子）；DiffusionGemma 与 Qwen-Image-2.1 的 Transformer 内核；以及 Qwen-Image 的文本编码器与视觉塔。Qwen-Image 的 VAE 另有一个更宽的专用池（`TS_CPU_GEMM_THREADS`）。8 个以上 CPU 时默认取可用数的**一半**，刻意不是全部。这个宽度是在线程池只运行量化 matmul、CPU 路径其余部分仍使用 ThreadPool 时调出来的：池内线程在两次任务之间自旋，占满每个核心会把那部分工作饿死。在 122 个 CPU 的配额上实测，每格为两次交替运行（prefill / decode tok/s）：关闭池 21.7,21.0 / 2.0,2.4；32 线程 24.9,24.1 / 4.9,5.0；48 线程 25.6,28.5 / 5.4,6.0；61 线程 24.2,24.9 / 6.3,5.9；122 线程 13.5 / 4.8。122 线程时只有 prefill 回退，解码仍优于关闭池的基线。在所有工作都上池之后重新测量，8 核 16 线程的 i7-11800H 上 16 线程对比默认的 8 线程，各跑两次：Qwen-Image-2.1 256x256、2 步 15.7 / 16.5 s 对 16.9 / 17.1 s（Transformer 步更快，9.2-9.9 s 对 11.0-11.1 s；文本与视觉编码阶段（含加载文本编码器）更慢，3.4-3.6 s 对 3.1 s；单看文本编码器前向（`QwenImageStagesBench`，37 token 的提示），首次调用 1.65-2.26 s 对 1.24-1.26 s，之后 0.72-1.42 s 对 0.68-0.72 s，输出完全相同）；DiffusionGemma 对新提示的 Jev 读取平均 1.80 s 对 1.54-1.62 s；对已缓存提示的读取 209-223 ms 对 218-231 ms。没有明显收益，因此默认值不变 | 8 核及以下取全部核心，否则 max(8, 可用数/2) | 未注册 | 否 |
| `TS_CPU_POOL` | `cpu` 后端 | `0` 回退到 ThreadPool 的 `Parallel.For`，用于负担不起专用自旋线程的主机，也便于在同一个二进制里做 A/B。它覆盖所有经 `CpuWorkers`、量化 matmul 的 `RunParallelBlocks` 或 Core 的 `CpuParallel` 钩子（跳过其绑定）分派并行任务的托管内核：量化 GEMM 与浮点面板 GEMM、Core 的 CPU 内核、DiffusionGemma 与 Qwen-Image-2.1 的 Transformer 内核，以及 Qwen-Image 的 VAE、文本编码器与视觉塔背后的 packed GEMM（VAE 的宽线程池变为宽度上限为 `TS_CPU_GEMM_THREADS` 的 `Parallel.For`）。这些内核的结果与任务切分无关，因此两种方式输出相同。仍在线程池上的：Direct 视频网络的行循环（`DirectOps`、MiniMax-H3 自己的循环）；DeepSeek V4 的执行器有自己的线程（`TS_DSV4_THREADS`） | 启用 | 未注册 | 否 |
| `TS_CPU_SPIN` | `cpu` 后端 | 池内线程挂起前的自旋次数。在这个宽度下挂起才是最贵的部分（唤醒 N 个线程的开销超过它们要分到的那约 60 微秒工作量），因此默认自旋次数足够多，使稳态下根本不会挂起：同一个模型在 256 时实测 0.1 tok/s，在 4096 时是 7.0 | `4096` | 未注册 | 否 |
| `TS_CPU_TASK_BYTES` / `TS_CPU_TASKS_PER_WORKER` | `cpu` 后端 | 单次托管 matmul 的切分方式：每个工作项对应多少字节权重，以及每个线程最多分到几个工作项。是按**工作量**而不是线程数来定的——旧的按线程数缩放的规则在 122 线程时会为一次 matmul 造出 1024 个极小任务，并且超过 8 线程后就不再有加速 | `131072` / `4` | 未注册 | 否 |
| `TS_CPU_QGEMM_MIN_ROWS` | `cpu` 上的量化 matmul（诊断） | 行数更少的调用与批任务改走逐行路径。不设置时任何行数都走 GEMM，这使某一行的结果与同一次调用里有多少行无关（decode、投机验证、连续批处理与 MoE 批得到相同的比特）；大于 1 的值会放弃这一点 | 未设置（1） | 未注册 | 否 |
| `TS_CPU_QGEMM_TASK_MACS` / `TS_CPU_QGEMM_L2_BYTES` | `cpu` 上的量化 matmul（调优） | 每个并行任务的最少乘加次数（单核约 50 微秒），以及一个行块最多容纳的激活字节数；每解码一对列就要重读一次这个行块，所以它必须留在 L2 里 | `1048576` / `524288` | 未注册 | 否 |
| `TS_CPU_QGEMM_VERIFY` | `cpu` 上的量化 matmul（诊断） | `1` 让每次 GEMM 再经逐行路径算一遍，并把迄今最大的相对差打印到 stderr，用真实模型的权重与激活检查内核。很慢 | 关闭 | 未注册 | 否 |
| `TS_CPU_SGEMM_KERNEL` | `cpu` 上的 F32 matmul（`Ops.Addmm` / `AddmmBatch`、Direct 视频网络的 GEMM） | 固定微内核：`avx512`、`avx2wide`（利用 32 个 EVEX 寄存器的 8x24，仅限 AVX-512 硬件）、`avx2` 或 `portable`。不受支持的选择回落到默认值 | 支持的最宽内核 | 未注册 | 否 |
| `TS_CPU_SGEMM_KC` / `TS_CPU_SGEMM_MC` / `TS_CPU_SGEMM_NC` | `cpu` 上的 F32 matmul（调优） | 缓存分块覆盖；MC 与 NC 向上取整到寄存器分块 | 256 / 144 / 1024（AVX-512、AVX2） | 未注册 | 否 |
| `TS_CPU_SGEMM_DOT_MAXN` | `cpu` 上的 F32 matmul | 走窄 dot 路径的最大 N（N 很小，且 A 的行与 B 的列沿 K 连续）；`0` 关闭该路径 | 64（AVX-512、AVX2），40（`avx2wide`） | 未注册 | 否 |
| `TS_CPU_DISABLE_AVX512` | 纯 C# CPU 路径里所有手写的 AVX-512 内核（量化 GEMM 及其量化器、逐行 Q4_0 / Q8_0 点积、SGEMM、逐元素、DiffusionGemma 注意力、Qwen-Image DiT / VAE / 文本编码器 / 视觉内核） | `1` 让它们以 AVX2 形式运行，从而能在 AVX-512 主机上测试 AVX2 路径（见上文"运行哪套指令集"与"测试 AVX2 路径"）。它们都经由同一个判定读取这个开关，不会有的保留 AVX-512 而有的放弃 | 关闭 | 未注册 | 否 |

## 矩阵外的 DiffusionGemma 变量

这些变量是真实运行时开关，但目前未注册到 `EnvVarMatrix.All`，也不在默认
TestMatrix 配置中 sweep。

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `DIFFUSION_STEPS` | DiffusionGemma Web UI | 服务端路径每个 block 的去噪步数 | `48` | 未注册 | 否 |
| `DIFFUSION_MAX_BATCH` | DiffusionGemma Web UI | `DiffusionBatchScheduler` 的最大活跃请求数 | `2` | 未注册 | 否 |
| `DIFFUSION_BATCHED_FORWARD` | DiffusionGemma | 真正批处理 canvas decode vs 按时间片执行融合单 canvas decode | 关闭 | 未注册 | 否 |
| `DIFFUSION_NO_PKV` | DiffusionGemma | 关闭 prompt-KV 缓存（device-glue 后端与 `cpu`）：之后每次读取、每个去噪步都走统一的 `[prompt\|canvas]` 前向 | 关闭 | 未注册 | 否 |
| `DIFFUSION_CPU_ATTN_FAST` | `cpu` 上的 DiffusionGemma | `1` 选择 FMA 注意力分块（硬件加速时用 Vector512）与向量化 softmax。默认内核精确复现旧的算术，因为最后一个比特的变化就可能翻转 128 选 8 的专家路由；在 Jev 与聊天的提示长度下，注意力只占一次前向的不到百分之一 | 关闭 | 未注册 | 否 |
| `DIFFUSION_CPU_MOE_CHUNK` | `cpu` 上的 DiffusionGemma | 每次批量 MoE 处理的 token 数；限制按路由收集的暂存区大小（否则 4k token 的 prefill 要占约 1 GB 的路由行） | `512` | 未注册 | 否 |
| `DIFFUSION_NO_SC` / `DIFFUSION_SC_TOPK` | DiffusionGemma | self-conditioning 开关与实验 top-K 截断 | 启用 / `32` | 未注册 | 否 |
| `DIFFUSION_NO_FUSED_DECODE` / `DIFFUSION_NO_FUSED_LMHEAD_TAIL` | GGML 后端上的 DiffusionGemma | 关闭融合整模型 diffusion decode 或融合 lm-head tail | 关闭 | 未注册 | 否 |
| `DIFFUSION_LMHEAD_BATCH_CAP_MB` | DiffusionGemma | 回退到按序列 lm-head 前的临时 logits 内存上限 | `300` | 未注册 | 否 |
| `DIFFUSION_VRAM_HEADROOM_MB` | ggml_cuda 上的 DiffusionGemma | 预加载权重之外保留的 VRAM 余量（计算缓冲、device copy） | `2048` | 未注册 | 否 |
| `DIFFUSION_DEVICE_COPY_BUDGET_MB` | ggml_cuda 上的 DiffusionGemma | 模型放不进 VRAM 时 device-copy 缓存的上限（prompt K/V、mask、激活） | `768` | 未注册 | 否 |
| `DIFFUSION_SEGMENTED_DECODE` | ggml_cuda 上的 DiffusionGemma | 强制开启（`1`）/关闭（`0`）逐层融合 decode；模型放不进 VRAM 时自动启用 | 自动 | 未注册 | 否 |
| `DIFFUSION_PIN_STREAMED` | ggml_cuda 上的 DiffusionGemma | 把流式（非常驻）权重复制到页锁定内存以 DMA 速度上传（消耗 RAM） | 关闭 | 未注册 | 否 |
| `DIFFUSION_IMAGE_BIDIRECTIONAL` | DiffusionGemma 图像输入 | `0` 让图像 soft token 区间内的注意力在滑动窗口层上改为普通因果，而不是双向 | 启用 | 未注册 | 否 |
| `DIFFUSION_FUSED_PREFILL_ATTN` | GGML 后端上的 DiffusionGemma | 融合的提示 prefill 注意力内核。在 `ggml_cuda` 上默认开启（`0` 恢复逐算子参考路径）；在其他 GGML 后端上用 `1` 显式启用。带图像的提示始终走逐算子掩码路径 | `ggml_cuda` 上启用，其余关闭 | 未注册 | 否 |
| `DIFFUSION_NO_DEVICE_SAMPLE` / `DIFFUSION_DEVICE_SAMPLE_FORCE` | ggml_cuda 上的 DiffusionGemma | `1` 关闭设备端采样（在设备 logits 上完成 argmax、熵、采样与 self-conditioning top-K）；`FORCE=1` 在分段 decode 下也保持开启，否则那里会因实测更慢而跳过 | 模型放得进 VRAM 时启用设备端采样 | 未注册 | 否 |
| `DIFFUSION_ASYNC_COMPUTE` | `ggml_metal` 上的 DiffusionGemma | `1` 保持异步计算开启。该模型会关掉它，因为 Metal 的延迟同步对"主机写入后接设备内核"没有屏障，会破坏逐算子 prefill 并丢失提示内容；强行打开是不安全的，只用于 A/B 这次同步的代价 | 关闭 | 未注册 | 否 |

## 矩阵外的 Qwen-Image-2.1 开关

这些变量调节 CLI 与服务端上的 Qwen-Image-2.1 生成与编辑。矩阵功能目录没有图像生成功能，因此它们都没有
注册进 `EnvVarMatrix.All`。在纯 C# 的 `cpu` 后端上，未指定尺寸的请求使用 1 百万像素的自动面积（1024x1024，
编辑时沿用第一张参考图的宽高比），而不是原生的 2048x2048；`ggml_cpu` 与 GPU 后端保持 2048x2048。实测数据与其余诊断开关见 [Qwen-Image-2.1 卡片](models/qwenimage21_zh-cn.md)。

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_QWEN21_EDIT_NOISE` | Qwen-Image-2.1 编辑 | `references`（或不设置）：编辑的初始噪声取自由参考图（8 位像素的 SHA-256）决定的 Philox 流，因此在相同种子与尺寸下也不会从画出源图的那份噪声重新开始。`seed`：编辑改用该种子的文生图噪声（与 stable-diffusion.cpp 相同），用于噪声对齐的对比，每次这样的编辑都会打印一行。其他取值会使请求失败。文生图噪声在两种设置下相同 | `references` | 未注册 | 否 |
| `TS_QWEN21_PREFIX_CACHE` | Qwen-Image-2.1 DiT | `0`（或 `false` / `off` / `no`）关闭前缀 KV 缓存：它在第一个去噪步保存文本与参考图像的 key / value，之后每一步复用 | 启用 | 未注册 | 否 |
| `TS_QWEN21_PREFIX_CACHE_TYPE` | 同上 | 缓存前缀的存储类型：`auto`（注意力实际读取的类型，输出与不缓存时一致）、`f16`、`f32`、`q8_0` 或 `q8_0_v`（8 bit 设置会对保存的前缀做舍入）。即使缓存关闭，拼错的值也会报错 | `auto` | 未注册 | 否 |
| `TS_QWEN21_PREFIX_CACHE_MAX_MIB` | 同上 | 单个缓存的上限（MiB），叠加在"缓存最多使用设备空闲内存一半"的规则之上。放不下的缓存会带警告被拒绝，该请求每一步都重新计算前缀 | 未设置 | 未注册 | 否 |
| `TS_QWEN21_GRAPH_REUSE` | `ggml_cuda` / `ggml_metal` 上的 Qwen-Image-2.1 DiT | 在去噪步之间保留计算图，ggml-cuda 因此可以把它们捕获为 CUDA graph；`0` 为每次预测重建计算图。CPU 与 Vulkan 总是构建临时计算图 | 启用 | 未注册 | 否 |
| `TS_QWEN21_FLASH` / `TS_QWEN21_PAD_MASK` | Qwen-Image-2.1 DiT | 诊断开关：`TS_QWEN21_FLASH=0` 关闭 flash attention；在 `ggml_cuda` 上，`TS_QWEN21_PAD_MASK=1` 恢复填充掩码注意力路径（其他后端忽略它） | 启用 / 关闭 | 未注册 | 否 |
| `TS_QWEN21_CUBLAS_MATMUL` / `TS_QWEN21_CUDNN_ATTENTION` | `ggml_cuda` 上的 Qwen-Image-2.1 DiT | `TS_QWEN21_CUBLAS_MATMUL=0` 让宽的 Q4_K / Q6_K 投影留在 ggml 的 MMQ 内核，而不用 cuBLAS BF16 GEMM（需要 sm_80 或更新；`TS_QWEN21_CUBLAS_MIN_ROWS` 为 token 阈值，默认 256）。`TS_QWEN21_CUDNN_ATTENTION=0` 让无掩码注意力留在 ggml 的 flash attention 内核，而不用 cuDNN 的融合 SDPA | 启用 / 启用 | 未注册 | 否 |
| `TS_QWEN21_VAE_FUSED` | Qwen-Image-2.1 VAE | 整 VAE 计算图，CUDA 与 Metal 上默认使用。`0` 选择逐卷积路径；`1` 在其他 GGML 后端上强制使用，但 Vulkan 除外，那里会被忽略并给出警告 | CUDA / Metal 上启用 | 未注册 | 否 |
| `TS_QWEN21_VISION_FUSED` | `ggml_cuda` 上的 Qwen-Image-2.1 视觉编码器 | `0` 恢复此前的逐块视觉路径，用于 A/B | 启用 | 未注册 | 否 |
| `TS_QWEN21_CPU_MATMUL` | `cpu` 上的 Qwen-Image-2.1 DiT | 不设置（或为 `q8`）时，量化投影像 ggml-cpu 那样把激活量化成 Q8_K / Q8_0，并在 `ManagedQuantizedOps` 的多行整数 GEMM 上计算：有了这些内核它是更快的路径（带缓存的单步在 256x256 时 4.0 s 对 7.2-9.0 s，512x512 时 17.6-21.7 s 对 29-38 s），图像也更接近 ggml_cpu（512x512、Pruna 5 步 LoRA：PSNR 32.9 dB 对 31.4）。`f32`（之前的默认值）改用 F32 激活乘以反量化的权重分块：更慢，但数值更稳定——输入 latent 相对变化 1e-6 时，整数流水线的速度场变化约 2.5e-2（相对 L2，ggml-cpu 自己约 1e-2），F32 的只变化约 1e-5 | 整数（Q8）激活 | 未注册 | 否 |
| `TS_QWEN21_CPU_PROFILE` | 同上 | `1` 每次前向打印一行各阶段耗时 | 关闭 | 未注册 | 否 |
| `TS_QWEN21_CPU_GATHER_V` / `TS_QWEN21_CPU_MLP_ROWS` / `TS_QWEN21_CPU_DEPTH` | 同上（调优） | 注意力从按头排列的副本读取每个头的 value（`0` 直接读取按 token 排列的 V）；MLP 每块的行数，用来限制其 `[rows, 2 * ff]` 激活的大小；dot 分块一趟的深度（16 的倍数） | 启用 / `1024` / `1024` | 未注册 | 否 |
| `TS_QWEN21_CPU_GELU_FP16` / `TS_QWEN21_CPU_ROUND_ACTIVATIONS` | 同上（对齐用） | 为 A/B 对比复现 ggml-cpu 的舍入：它的 F16 GELU 表，以及输入 F16/BF16 `img_in` / `txt_in` 权重前按 BF16 舍入。默认是 F32 的 tanh GELU（与 CUDA、Metal 的算法相同）和 F32 输入 | 关闭 / 关闭 | 未注册 | 否 |
| `TS_QWEN_VAE_PROFILE` | `cpu` 上的 Qwen-Image-2.1 VAE | `1` 打印一次编码或解码中各类算子（conv、norm、add、attention、resample、权重打包）的耗时 | 关闭 | 未注册 | 否 |
| `TS_QWEN_IMAGE_CPU_MEMORY_CHECK` | `cpu` 上的 Qwen-Image-2.1 | 在开始任何工作之前，估计峰值（VAE 解码：每个输出像素 2304 字节再加约 1.8 GiB；去噪：每个 token 128 KiB 再加 2.5 GiB 与映射的 Transformer）超过机器内存的尺寸会被拒绝，并给出能放下的最大方形尺寸；只超过当前空闲内存的尺寸会得到一条警告。`0` 跳过拒绝 | 启用 | 未注册 | 否 |
| `TS_CPU_GEMM_THREADS` | `cpu` 上的 Qwen-Image-2.1 VAE | 托管 VAE 专用线程池的宽度。默认每个逻辑 CPU 一个线程：卷积受 FMA 限制，每核两个 SMT 线程能让唯一的 512 位 FMA 端口更忙（512x512 解码 7.9 -> 7.0 s）。文本编码器与视觉塔仍在共享池上，那里多出的自旋线程得不偿失。`TS_CPU_POOL=0` 时 VAE 改用宽度上限为此值的 `Parallel.For`，不再使用专用池 | 逻辑 CPU 数，最多 64（限制在 1-512） | 未注册 | 否 |
| `TS_CPU_GEMM_KC` / `TS_CPU_GEMM_NT` | `cpu` 上的 Qwen-Image packed GEMM（VAE、文本编码器、视觉塔） | packed GEMM 的 K 分块与 N 分块，仅用于调优；结果与它们无关 | `256` / `256`（限制在 16-4096 / 32-2048） | 未注册 | 否 |
| `TS_QWEN_TE_CPU_MATMUL` | `cpu` 上的 Qwen-Image-2.1 文本编码器（Qwen3-VL-8B） | 不设置时，投影像 ggml-cpu 那样把激活量化到 8 位，并在多行整数 GEMM 上计算：37 token 的默认提示用时 0.76-0.88 s，packed F32 GEMM 为 1.8-1.9 s（ggml_cpu 约 1.4 s）。`f32` 改用反量化权重分块上的 packed F32 GEMM（激活为精确的 F32：每个投影与双精度参照相差约 1e-6，8 位路径约 4e-3）。 | 整数（Q8）激活 | 未注册 | 否 |
| `TS_QWEN_TE_PROFILE` | 同上 | `1` 每次前向打印逐算子路径的耗时拆分（linear / attention / norm） | 关闭 | 未注册 | 否 |

在 `cpu` 上，上述前缀 KV 缓存设置作用于托管缓存，它位于主机内存中："空闲内存"指未被使用的物理内存。
只对 GGML 有效的开关（`TS_QWEN21_GRAPH_REUSE`、`TS_QWEN21_FLASH`、`TS_QWEN21_PAD_MASK`、
`TS_QWEN21_VAE_FUSED`、`TS_QWEN21_VISION_FUSED`）在那里不起作用。

## 矩阵外的投机解码变量

这些变量控制 `TensorSharp.Cli` 与 `TensorSharp.Server` 中可选的投机解码路径
（Qwen 3.6、Qwen 3.8 27B、GLM 5.2 与 GLM-5.3 内嵌的 NextN 块；Gemma 4 独立的 `gemma4-assistant`
草稿 GGUF；Qwen 3.8 Flash Next 的共享 MTP 头 GGUF；DeepSeek V4 DSpark，以及用于 Muse-Glimmer 与
Qwen 3.5 家族的 DFlash / DFlash2 块级草稿器；实验性的 DeepSeek V4.1 DSpark 路径，训练模型已在 `ggml_cuda` 双 GPU 按层切分下通过初步文本/图像 HTTP 检查；尚不构成通用质量或吞吐验证；
以及无需权重的 n-gram 投机器）。投机仅对单序列（无并发）请求生效，且只在模型声明有收益时启用，这由各
模型自己决定：Qwen 3.5/3.6/3.8 与 GLM 5.2 / GLM-5.3 在所有后端上，GLM-5.3-Flash（仅 n-gram）在其 KDA
回滚可用时，Gemma 4 在 ggml 后端与 `cuda` 上，Qwen 3.8 Flash Next 在其 GGML token 计算图路径上，DeepSeek V4 / V4.1 与 Muse-Glimmer 只在加载了各自草稿器时。Nemotron-H
拒绝一切投机器，GPT OSS、Mistral 3 与 Hunyuan Dense 没有投机主干，连 n-gram 也不会运行。
它们未注册在 `EnvVarMatrix.All` 中，也不在默认 TestMatrix 配置里扫描——矩阵特性目录目前
没有投机解码特性，请用显式运行来验证这些变量。

每个开关只有一个 `TS_SPEC_*` 名字，glm-dsa 的**原生**加载器也读取它们：在模型加载过程中
从 C++ 侧读取 `TS_SPEC_DRAFT`（据此确定图缓存大小），并在托管侧读取 `TS_SPEC`，决定要不要把
多出来的一整层 256 专家 decoder 调进显存。所有这些也都可以通过两个宿主上的 `--spec*` 参数与
`--draft-model` 设置。旧的 `--mtp-*` 参数（以及 `--spec-draft-model`、`--spec-draft-n-max`、
`--spec-draft-conf-min`）和旧的 `TS_MTP_*` 环境变量都已被移除，现在会在启动时报错并给出替代名字。

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_SPEC` | Qwen 3.5/3.6/3.8、GLM 5.2、GLM-5.3、GLM-5.3-Flash（仅 n-gram）、Gemma 4、Qwen 3.8 Flash Next、DeepSeek V4 / V4.1、Muse-Glimmer、Muse Spark（仅 n-gram，其原生 `--tp` 下不启用）、MiniMax-M3.1（DSpark 或 n-gram，仅按层切分）（CLI + 服务端） | 为单序列启用投机解码 | 关闭（`0`） | 未注册 | 否 |
| `TS_SPEC_TYPE` | 同上全部 | 投机算法：`auto` \| `draft-head` \| `block` \| `ngram` | `auto` | 未注册 | 否 |
| `TS_SPEC_DRAFT` | 同上全部 | 每个投机步最多起草的 token 数（1-64） | `8` | 未注册 | 否 |
| `TS_SPEC_PMIN` | 同上全部 | 草稿置信度门限；含义随算法而定 | 按算法（`0.15` / `0.35` / `0`） | 未注册 | 否 |
| `TS_SPEC_DRAFT_MODEL` | 草稿器以独立 GGUF 发布的模型（CLI + 服务端） | 独立草稿器的路径，按文件的架构识别：Gemma 4 的 `gemma4-assistant`、Qwen 3.8 Flash Next 的共享 MTP 头、DeepSeek V4 DSpark（或实验性的 V4.1 `deepseek41-dspark`）草稿器，DFlash / DFlash2 草稿器（Muse-Glimmer、Qwen 3.5 家族），或 MiniMax-M3.1 的 DSpark `dflash` 草稿器（只由其原生执行器在按层切分时加载）。由 `--draft-model` 设置，且除非给出 `--no-spec`，它本身就会启用投机 | 无 | 未注册 | 否 |

这些开关背后的设计——把模型架构、投机算法与投机器权重拆成三层——记录在
[Speculative Decoding in TensorSharp](speculative_decoding.md)（英文）。

## 矩阵外的 Muse-Glimmer 与 DFlash 开关

Muse-Glimmer 的融合整模型内核与它的 DFlash 块级草稿模型各有一个 A/B 开关，另外还有
长上下文的尺寸开关。这些都没有注册进 `EnvVarMatrix.All`。完整清单（含逐层追踪开关）见
[Muse-Glimmer 卡片](models/muse-glimmer_zh-cn.md#7-环境变量)。

| 环境变量 | 适用范围 | 功能影响 | 运行时基线 | 扫描取值 | 默认是否扫描 |
|---|---|---|---|---|---|
| `TS_MUSE_GLIMMER_PREFILL_CHUNK` | Muse-Glimmer | 每次 prefill 前向的 token 数；`0` 关闭分块 | `2048` | 未注册 | 否 |
| `TS_MUSE_GLIMMER_SWA_RING` | Muse-Glimmer（融合） | 把 39 个滑动窗口层按 `pad(n_swa + chunk + 1, 256)` 行做环，而不是所有层都按完整上下文分配 | 开 | 未注册 | 否 |
| `TS_MUSE_GLIMMER_SWA_ROWS` | Muse-Glimmer（融合） | 覆盖 SWA 环的行数（诊断用） | 自动 | 未注册 | 否 |
| `TS_DFLASH_PREFILL_CHUNK` | 任意 DFlash 草稿器 | 每次投机 prefill 前向的 token 数（驱动的是**主干**，不只是草稿器） | `1024`，并受草稿器环形缓冲与主干自身窗口的限制 | 未注册 | 否 |
| `TS_DFLASH_SELECTOR` | DFlash2 草稿器 | `0` 改为按逐位置 argmax 起草，而不走候选格（仅用于归因分析——权重本来就是带着它训练的） | 开 | 未注册 | 否 |
| `TS_DFLASH_CONV` | DFlash2 草稿器 | `0` 去掉分组动态卷积（同上，仅用于归因分析） | 开 | 未注册 | 否 |
| `TS_DFLASH_SELECTOR_DEBUG` | DFlash2 草稿器（逐算子路径） | `1` 打印前几个 block 的候选格归因：一元项分布、转移项分布，以及这次游走是否离开了一元 argmax | 关 | 未注册 | 否 |
| `TS_DSPINE_DUMP` | DSpine 草稿器 | 一个目录：把前四次草稿的输入（环里保存的每个上下文特征行及其位置、block）和输出（候选、候选格、链）写成 `draftN.bin`，供 `eng/dspine-reference.py real` 用检查点重新计算（诊断用） | 未设置 | 未注册 | 否 |
| `TS_DSPINE_BF16_POINTS` | DSpine 草稿器 | `1` 在参考实现舍入的位置把代码桥的激活舍入为 BF16（raw、归一化行、读投影、码、打包投影、消息、写入增量、写入后的行、下一层归一化），而不是保持 F32——用来衡量 TensorSharp 的 F32 激活带来的差别（诊断用） | 关 | 未注册 | 否 |
| `TS_PREFIX_RELEASE_DEBUG` | Qwen 3.5 / 3.8 前缀缓存 | `1` 记录保留/规划/整理决策、降级（`[prefix-demote]`）及其主机同步（`[prefix-sync]`） | 关 | 未注册 | 否 |
| `TS_GGML_HOST_POOL_MB` | GGML CUDA 主机内存分配 | 保留任意大小已释放主机块以供复用的字节预算（新映射首次访问会产生缺页）；`0` 仅保留按数量限制的小块池 | `min(8 GiB, 内存/16)` | 未注册 | 否 |
| `TS_BATCHED_SPEC_LOG` | 批量推测解码 | `1` 打印每个批量步骤保留的 token 数以及草稿 / 验证 / 提交 / 草稿模型提交的毫秒数（`[batched-spec]`） | 关 | 未注册 | 否 |
| `TS_SCHED_STEP_LOG` | 连续批处理执行器 | `1` 每步打印一行：距上一步的空闲间隔、服务路径耗时毫秒数，以及每个序列的预填充/解码 token 数（`[step]`） | 关 | 未注册 | 否 |
| `TS_SCHED_PARALLEL_SAMPLE` | 连续批处理执行器 | `0` 让批量解码步骤逐行依次采样。开启时，四行及以上的步骤并行采样各行，每个序列用自己的采样器（自己的随机数生成器和缓冲区），因此抽样结果不变；在 Qwen 248K 词表上做一次全词表 top-k，每行在主机上约 0.3 ms | 开 | 未注册 | 否 |
| `TS_SCHED_PIPELINE_DECODE` | 连续批处理执行器，能同时保持两个解码组在途的模型（`IPipelinedBatchedDecode`：`--backend cuda --layer-split 2` 及以上的 Qwen 3.8 Flash Next） | `0` 让每一步的序列作为一个批量步骤解码。开启时，只含两个或更多解码序列的步骤把它们分成两组，每组的步骤发出后不等待：其 logits 在该组下一步采样时才读取，因此第一块 GPU 运行一组的层时，最后一块 GPU 运行另一组的层。每个序列在解码期间保持所在的组；含 prompt 分块、推测解码或其他工作的步骤会先完成在途的组。两块 H100 NVL、Q8_0、贪心：4 / 8 / 16 路 327 / 457 / 582 tok/s，对比 236 / 309 / 377；生成的 token 相同 | 开 | 未注册 | 否 |
| `TS_GPU_LOCK_YIELD_MS` | 连续批处理执行器 | 当请求线程已在等待模型 GPU 锁（提示的媒体准备）时，执行步骤在自己获取该锁之前最多等待的毫秒数；`0` 表示步骤优先获取 | `100` | 未注册 | 否 |
| `TS_PREFIX_ASYNC_SAVE` | 前缀检查点持久化 | `0` 在引擎工作线程上写共享前缀检查点文件（导出、写入并 fsync），而不是在后台写；后台保存需要 32 GB 主机内存 | 开 | 未注册 | 否 |
| `TS_SCHED_PREFILL_FCFS` | 连续批处理调度器 | `1` 在没有解码进行时按先到先得把步骤预算分给提示，而不是平均分配 | 关 | 未注册 | 否 |
| `TS_SCHED_MIXED_PREFILL_TOKENS` / `TS_SCHED_MIXED_PREFILL_FCFS` | 连续批处理调度器 | 同时有解码的步骤里所有预填充序列合计可携带的提示 token 数（`0`：只受每序列分块上限约束）；FCFS 把这份预算留给最早的提示。变量未设置时模型可自行给出上限（Qwen 3.5 / 3.8 直连 CUDA 引擎：`1024`） | 模型给出的上限，否则无 / 关 | 未注册 | 否 |
| `TS_SCHED_MIXED_STEP` / `TS_Q35_MIXED_STEP` | Qwen 3.5 / 3.8 直连 CUDA 引擎 | `0` 把同一步的提示分块与批量投机验证分成两次前向，而不是一次混合前向 | 开 | 未注册 | 否 |
| `TS_Q35_MIXED_FP4_VERIFY` | Qwen 3.5 / 3.8 直连 CUDA 引擎 | `0` 让不超过 8 行的验证窗口和提示尾部单独走 q8_1 GEMV（8 位激活），而不是随混合前向走 FP4 GEMM（更宽的批量验证一直如此） | 开 | 未注册 | 否 |
| `TS_Q35_NATIVE` | `--backend cuda` 上的 Qwen 3.5 / 3.8 稠密 NVFP4 | `0` 保留托管路径而不用直连 CUDA 引擎；没有 sm_12x 或未用 nvcc 编出其内核时引擎也会让位 | 开 | 未注册 | 否 |
| `TS_Q35_HOST_CACHE_MB` | Qwen 3.5 / 3.8 直连 CUDA 引擎 | 移出显存的保留前缀槽位所用的锁页主机内存（`0` 关闭主机层） | `min(可用内存 / 2, 32 GiB)` | 未注册 | 否 |
| `TS_Q35_UBATCH` | Qwen 3.5 / 3.8 直连 CUDA 引擎 | 每次前向的行数（提示分块的微批） | `2048` | 未注册 | 否 |
| `TS_Q35_ATTN_PREFILL` / `TS_Q35_GEMM_SPLITK` | Qwen 3.5 / 3.8 直连 CUDA 引擎 | 诊断用：`0` 恢复旧的预填充注意力内核 / 让每个 FP4 GEMM 只做一遍 K（求和顺序与行数无关） | 开 / 开 | 未注册 | 否 |
| `TS_QWEN_VISION_MERGED` | Qwen 3.5 / 3.8 视觉编码器 | `0` 在融合编码器图之后以独立算子运行 post-LN 空间合并器，而不是放在图内（两者结果逐位一致） | 开 | 未注册 | 否 |
| `TS_QWEN_VISION_F16` | Qwen 3.5 / 3.8 视觉编码器（CUDA） | `1` 让合并编码器使用 F16 线性权重（F32 累加）：常驻权重内存减半，相对 F32 的相对 L2 约 0.002-0.003，未测得加速 | 关 | 未注册 | 否 |
| `TS_QWEN_VISION_DUMP` | Qwen 3.5 / 3.8 视觉编码器 | 将每张图像的投影嵌入（int32 个数 + float32 值）写入该目录，供 `eng/validation/compare-vision-embeddings.py` 比较 | 未设置 | 未注册 | 否 |
| `TS_QWEN_VISION_ATTN` | Qwen 3.5 / 3.8 视觉编码器（CUDA sm_80+） | 视觉注意力内核：`mma`（张量核心，F16 操作数，F32 分数/softmax/输出；相对精确 F32 的精度与分解路径相同，编码器耗时约减半）、`ggml`（分解的 TF32 矩阵乘 + softmax）、`f32`（精确 F32 内核，较慢） | `mma` | 未注册 | 否 |
| `TS_QWEN_VISION_F32` | Qwen 3.5 / 3.8 视觉编码器（CUDA） | `1` 选择精确 F32 注意力内核（等同 `TS_QWEN_VISION_ATTN=f32`） | 关 | 未注册 | 否 |
| `TS_SPEC_ADAPTIVE` | 投机解码（所有草稿器） | `0` 关闭成本调节器，于是起草不再与普通 baseline 做对比、也永远不会被暂停。用于 A/B 测量：调节器每一轮的 baseline 步骤都是普通 decode，它们并不免费 | 开 | 未注册 | 否 |
| `TS_SPEC_TURNS` | 并发解码、目标模型没有批量验证时的投机解码 | `0` 从不轮流投机，`1` 在目标模型支持时总是轮流。轮流：一个序列在自己的 holder 上做单序列投机步（最多 64 步），其他序列等待，然后由最久未被服务的序列接手，而不是每个序列在批量步中各解一个 token。未设置时：由目标模型决定（`ISpeculativeTarget.SpecTurnsPreferred`；目前没有模型偏好轮流——`ggml_metal` 上的 Qwen 3.8 Flash Next 在其 arena 把注意力半层按槽位批量运行之前偏好轮流：M5 Pro、ISTA `Q2_0`，parallel_2 批量 38.4 tok/s，轮流 37.0-37.9；parallel_4 批量 44.7-44.9，轮流 38.1-39.7） | 关闭（由模型决定） | 未注册 | 否 |
| `TS_GGML_LOG_DEBUG` | GGML 后端 | `1` 把 ggml 的 DEBUG 日志通道透传出来而不是丢弃。它承载 CUDA 后端的 "CUDA graph warmup complete" / "reset" 这两行，而这是唯一能看出一张图是否真的被 CUDA graph 捕获的途径 | 关 | 未注册 | 否 |

## 矩阵外的 Muse Spark 开关

Muse Spark 原生整模型执行器的选择与容量规划由以下开关控制，均未在 `EnvVarMatrix.All`
中注册。完整列表（包括逐算子路径的可选优化）见 [Muse Spark 卡片](models/muse-spark_zh-cn.md)。

| 环境变量 | 适用范围 | 功能影响 | 运行时基线 | 扫描取值 | 默认是否扫描 |
|---|---|---|---|---|---|
| `TS_MUSE_SPARK_NATIVE` | Muse Spark | `0` 选择逐算子路径；`1` 让 `ggml_cpu` 也运行原生执行器。`ggml_cuda` 上的 `--tp N` 由原生执行器通过 ggml 的 meta 后端在前 N 张 CUDA GPU 上做张量并行（用 `CUDA_VISIBLE_DEVICES` 选择；`TENSORSHARP_TP_DEVICES` 会被拒绝）；`0` 保留逐算子 TP 路径，`ggml_vulkan` 上的 `--tp` 总是走该路径 | `ggml_cuda` / `ggml_vulkan` 上开启 | 未注册 | 否 |
| `TS_MUSE_SPARK_CPU_MOE` | Muse Spark（原生） | `auto` 逐个放置专家栈，尽可能多地留在 GPU 上、让主机上的字节尽可能少；覆盖 `--n-cpu-moe`。原生 `--tp` 下所有专家都留在显存中 | 关闭 | 未注册 | 否 |
| `TS_MUSE_SPARK_PLAN_SLOTS` | Muse Spark（原生） | 放置规划中预留的滑动窗口环数；设为并发数再加一（留给共享前缀检查点） | `1` | 未注册 | 否 |
| `TS_MUSE_SPARK_UBATCH` | Muse Spark（原生） | prefill 微批行数；放不下的计算图会把该次调用拆成减半的微批 | `2048` | 未注册 | 否 |
| `TS_MUSE_SPARK_DEV0_RESERVE_MB` | Muse Spark（原生） | 在 GPU 0 上额外预留的内存，例如给视觉编码器 | `0` | 未注册 | 否 |
| `TS_MUSE_SPARK_HOST_NUMA` | Muse Spark（原生） | `0` 让主机专家保留在 GGUF 映射中，而不是复制到 NUMA 本地并由绑核工作线程计算 | 有两个及以上 NUMA 节点时开启 | 未注册 | 否 |
| `TS_MUSE_SPARK_WARM_POOLS` | Muse Spark（原生，GPU） | `0` 跳过加载时的满尺寸 prefill；该 prefill 让 ggml-cuda 的内核 scratch 池在 slot 占用内存之前先长到峰值；启用流水化 prefill 时，加载还会跑一次短的流水化 prefill（`0` 同样跳过它），建好流水线的调度器并确认它能生效 | 有 GPU 时开启 | 未注册 | 否 |
| `TS_MUSE_SPARK_PIPELINE` | Muse Spark（原生，按层切分） | 流水化 prefill：一次 prompt 调用被切成若干微批，GPU k 处理一个微批时 GPU k+1 处理前一个。未设置 = 自动（两张及以上 GPU、全部为 CUDA 系、没有主机专家、未设置 `TS_MUSE_SPARK_PROFILE`、不是 `--tp` 时开启）；`1` 强制开启（多 GPU 的非 CUDA 后端与 `--tp` 下会拒绝加载）；`0` 关闭；其他取值拒绝加载。加载日志会打印 `pipelined prefill: on (...)` 或 `off (原因)` | 自动 | 未注册 | 否 |
| `TS_MUSE_SPARK_PIPELINE_UBATCH` | Muse Spark（原生，流水化 prefill） | 流水化 prefill 的固定微批行数，取值为 [1, `TS_MUSE_SPARK_UBATCH`] 内的整数，其他值拒绝加载。未设置 = 自适应：约为每次调用的四分之一，介于 512 行与 prefill ubatch 之间；长上下文的掩码超出暂存预算时会更小（最少 128 行） | 自适应 | 未注册 | 否 |

## 矩阵外的 MiniMax-M3.1 开关

MiniMax-M3.1-Flash（`minimax-m3`）只在其原生整模型执行器上运行（`ggml_cuda`；`ggml_cpu` 为正确性路径），
下表列出执行器及其托管胶水代码读取的全部 `TS_MINIMAX_M3_*` 设置，均未在 `EnvVarMatrix.All` 中注册。
托管侧把 `_LAYER_SPLIT`、`_UBATCH`、`_THREADS` 与 `_CPU_MOE` 作为加载参数交给执行器，并转发其余变量，
因此在 .NET 进程内设置的值也能生效。放置参数本身
（`--layer-split N`、`--tp N`、`--n-cpu-moe N`、`--draft-model`）见 [MiniMax-M3.1 卡片](models/minimax-m3_zh-cn.md#运行)。

| 环境变量 | 适用范围 | 功能影响 | 运行时基线 | 扫描取值 | 默认是否扫描 |
|---|---|---|---|---|---|
| `TS_MINIMAX_M3_MSA` | MiniMax-M3.1 | 注意力路径。`auto`：最后一个位置低于 2048 的微批运行稠密 flash attention（在那里与稀疏结果完全一致），其余微批在 CUDA GPU 上用 TensorSharp 的 MSA 内核、在其他设备上用上游 ggml 算子运行稀疏注意力（`--tp` 下始终用 ggml 算子）。`dense`：处处稠密（调试用；从位置 2048 起改变结果）。`ggml`：每个微批都通过上游 ggml 算子运行稀疏注意力。`kernel`：每个微批都用 TensorSharp 的内核（纯 CPU 加载时用其 CPU 参考实现）；无法运行它们的 GPU 上以及 `--tp` 下拒绝。其他取值拒绝加载 | `auto` | 未注册 | 否 |
| `TS_MINIMAX_M3_MSA_CHUNK_MB` | MiniMax-M3.1（ggml 路径上的稀疏提示段） | 稀疏提示段带掩码注意力的单个 token 分块的字节预算；同时计入每张 GPU 的放置预留。必须是正的 MiB 数，否则拒绝加载 | `256` | 未注册 | 否 |
| `TS_MINIMAX_M3_INDEXER_F16` | MiniMax-M3.1 | 为 decode 形状的图（至多 8 行）使用 F32 索引器投影的 F16 副本，使每步 decode 读取的索引器字节减半；提示仍读取 F32 原件。`0` 只保留 F32 | 有 GPU 时开启，纯 CPU 加载时关闭 | 未注册 | 否 |
| `TS_MINIMAX_M3_FUSE_QKV` | MiniMax-M3.1 | 把类型相同的 Q、K、V 投影以及索引器的 q、k 投影各拼成一个矩阵（一次权重读取、一次启动）。`0` 保持分开 | 开启 | 未注册 | 否 |
| `TS_MINIMAX_M3_LAYER_SPLIT` | MiniMax-M3.1（按层切分） | 显式给出每张 GPU 的层数：逗号分隔的非负数，总和为 60，每张 GPU 至多一项；列表更长时会提高 GPU 数。格式错误或放不下时拒绝加载。与 `--tp` 同用会被拒绝 | 自动放置 | 未注册 | 否 |
| `TS_MINIMAX_M3_UBATCH` | MiniMax-M3.1 | prefill 微批行数；也限定一次投机验证（草稿加锚点）的大小 | `2048` | 未注册 | 否 |
| `TS_MINIMAX_M3_THREADS` | MiniMax-M3.1 | 纯 GPU 加载时 CPU 后端的线程数。纯 CPU 加载时使用可用的并行度；GPU 旁有主机专家时使用其一半（其 decode 受内存带宽限制，另一半让服务进程自己的线程不与专家工作线程争用核心）；`--cpu-moe-threads N` / `TS_CPU_MOE_THREADS` 覆盖两者 | 可用并行度（有主机专家时为一半） | 未注册 | 否 |
| `TS_MINIMAX_M3_CPU_MOE` | MiniMax-M3.1（按层切分） | `auto` 只把 GPU 放不下的路由专家栈（gate、up、down 分别放置）留在系统内存，并覆盖 `--n-cpu-moe`。它不会限制上下文（而是把专家移到主机），因此请同时设置 `MAX_CONTEXT`。`--tp` 下所有专家都留在显存 | 关闭 | 未注册 | 否 |
| `TS_MINIMAX_M3_PLAN_CTX_TOKENS` | MiniMax-M3.1 | 放置规划为所有槽位预留的 K/V token 数；应设为上下文乘以（并发请求数加一，留给共享前缀检查点）。不是正的 token 数时会给出提示并改用默认值 | 一个完整上下文 | 未注册 | 否 |
| `TS_MINIMAX_M3_VRAM_RESERVE_MB` | MiniMax-M3.1 | 放置规划在每张 GPU 上为计算缓冲与 ggml-cuda 暂存池保留的内存；稀疏分块预算以及 `--tp` 下的归约缓冲另计 | `2048` + ubatch / 4 | 未注册 | 否 |
| `TS_MINIMAX_M3_DEV0_RESERVE_MB` | MiniMax-M3.1 | 在 GPU 0 上额外保留的内存，例如留给视觉编码器 | `0` | 未注册 | 否 |
| `TS_MINIMAX_M3_KV_PAD` | MiniMax-M3.1 | 构建计算图时注意力窗口的粒度（行）：decode 在这么多个位置内复用同一图形状。会向上取整到 128 行 MSA 块的倍数，并给出警告 | `256` | 未注册 | 否 |
| `TS_MINIMAX_M3_BD_KV_PAD` | MiniMax-M3.1（批量 decode） | 批量 decode 图的窗口粒度：N 个序列跨越窗口边界的频率是单序列的 N 倍；保持为 `TS_MINIMAX_M3_KV_PAD` 的倍数 | 2 × `TS_MINIMAX_M3_KV_PAD` | 未注册 | 否 |
| `TS_MINIMAX_M3_SLOT_INIT_ROWS` | MiniMax-M3.1 | 序列槽位的初始 K/V 行数（按 KV 粒度取整，至多为上下文长度）；按需翻倍 | `4096` | 未注册 | 否 |
| `TS_MINIMAX_M3_SLOT_HEADROOM_MB` | MiniMax-M3.1 | 新槽位或 KV 增长必须在每张 GPU 上留出的空闲内存 | `256` | 未注册 | 否 |
| `TS_MINIMAX_M3_ADMIT_GRAPH_MB` | MiniMax-M3.1 | 新槽位或 KV 增长在余量之外最多为计算图保留的内存，避免一次长 prefill 的图让之后每个槽位都去驱逐保留的前缀状态 | `512` | 未注册 | 否 |
| `TS_MINIMAX_M3_GRAPH_CACHE` | MiniMax-M3.1 | 按形状缓存的 LRU 中保留的已构建图数量。`--tp` 下只保留一张图，大于 1 的取值会拒绝加载 | `12` | 未注册 | 否 |
| `TS_MINIMAX_M3_NODES_PER_LAYER` | MiniMax-M3.1（诊断） | 覆盖执行器根据图中各段推导出的每层图节点预算 | 推导值 | 未注册 | 否 |
| `TS_MINIMAX_M3_PIPELINE` | MiniMax-M3.1（按层切分） | 流水线 prefill：一次提示调用被切成多个微批，GPU k 处理一个时 GPU k+1 处理前一个。未设置 = 自动（两张及以上 GPU、全部为 CUDA 系、没有主机专家、未设置 `TS_MINIMAX_M3_PROFILE`、不是 `--tp` 时开启）；`1` 强制开启（多 GPU 的非 CUDA 后端与 `--tp` 下拒绝）；`0` 关闭；其他取值拒绝加载。加载日志打印 `pipelined prefill: on (...)` 或 `off (原因)` | 自动 | 未注册 | 否 |
| `TS_MINIMAX_M3_PIPELINE_UBATCH` | MiniMax-M3.1（流水线 prefill） | 流水线 prefill 的固定微批行数，为 [1, `TS_MINIMAX_M3_UBATCH`] 内的整数；其他取值拒绝加载。未设置 = 自适应：选取使流水线估计耗时最小的微批数（八张 A40 上 2,048 token 为 2 个、4,096 为 3 个、8,192 为 4 个），大小在 256 行到 prefill ubatch 之间，长上下文的掩码超过 96 MiB 暂存预算时更小（最小 128 行） | 自适应 | 未注册 | 否 |
| `TS_MINIMAX_M3_WARM_POOLS` | MiniMax-M3.1（GPU） | `0` 跳过加载时的满尺寸 prefill；该 prefill 让 ggml-cuda 的内核暂存池在槽位占用内存之前先长到峰值；启用流水线 prefill 时，加载还会运行一次短的流水线 prefill（`0` 同样跳过），建好流水线的调度器并确认它能生效 | 有 GPU 时开启 | 未注册 | 否 |
| `TS_MINIMAX_M3_HOST_NUMA` | MiniMax-M3.1（主机专家，Linux） | 有两个及以上 NUMA 节点时，把每个主机专家栈复制到读取它的绑核工作线程旁边的内存；`0` 从 GGUF 映射提供 | 两个及以上 NUMA 节点时开启 | 未注册 | 否 |
| `TS_MINIMAX_M3_NUMA_BATCH_GB` | MiniMax-M3.1（NUMA 主机专家） | NUMA 复制每批读取的主机专家栈 GiB 数，读完后丢弃该批的文件页 | `16` | 未注册 | 否 |
| `TS_MINIMAX_M3_HOST_MOE_TOKENS` | MiniMax-M3.1（主机专家） | 2 到 N 个 token 的批次通过 TensorSharp 的静态行内核（使用 ggml 的量化与点积）运行主机专家；`0` 保留 ggml 的 `mul_mat_id` | `16` | 未注册 | 否 |
| `TS_MINIMAX_M3_CPU_REPACK` | MiniMax-M3.1（主机专家） | `1` 把主机专家栈重排为 ggml 的 CPU_REPACK 布局，而不是从映射提供；仅在未做 NUMA 放置时生效。`--tp` 下拒绝 | 关闭 | 未注册 | 否 |
| `TS_MINIMAX_M3_OP_OFFLOAD` | MiniMax-M3.1 | 允许 ggml 调度器为大批次把主机权重拷贝到 GPU | 仅在没有主机专家时开启 | 未注册 | 否 |
| `TS_MINIMAX_M3_WARM_HOST` | MiniMax-M3.1（主机专家） | 当主机映射的专家权重加 8 GiB 不超过主机内存额度（cgroup 限制）时，在加载时预取这些页；否则日志说明跳过了预取。`0` 从不预取 | 开启 | 未注册 | 否 |
| `TS_MINIMAX_M3_LOAD_CHUNK_MB` | MiniMax-M3.1 | 权重上传的分块大小 | `64` | 未注册 | 否 |
| `TS_MINIMAX_M3_LOAD_THREADS` | MiniMax-M3.1 | 权重上传的读取线程数（也是主机预取的线程数） | `16` | 未注册 | 否 |
| `TS_MINIMAX_M3_DROP_UPLOAD_CACHE` | MiniMax-M3.1 | 每个分块上传到设备后释放其页缓存（Linux 用 `posix_fadvise`，macOS 用 `F_NOCACHE`；其他平台给出警告且不生效）。`0` 保留页缓存 | 开启 | 未注册 | 否 |
| `TS_MINIMAX_M3_DSPARK_GRAPHS` | MiniMax-M3.1 DSpark 草稿器 | 保留的草稿图数量（每个序列槽位一张，按最近使用淘汰） | `4` | 未注册 | 否 |
| `TS_MINIMAX_M3_EXEC_DEBUG` | MiniMax-M3.1（诊断） | `1` 为每次改变状态的执行器调用打印一行 stderr（调度器请求了什么、活动槽位的头位置与耗时），另含槽位增长、图构建与各阶段主机耗时 | 关闭 | 未注册 | 否 |
| `TS_MINIMAX_M3_PROFILE` | MiniMax-M3.1（诊断） | `N` > 0 时同步计时每个图节点，并每 N 次图计算打印一次按后端、按算子的汇总。会使 CUDA graph 失效、关闭流水线 prefill，`--tp` 下拒绝 | `0` | 未注册 | 否 |

## 矩阵外的 DeepSeek V4 / V4.1 开关

这些变量配置 DeepSeek 的整模型执行器。V4 有三套（Direct CUDA、原生 ggml、纯 C#）；
V4.1 的服务路径是一套原生 `ggml_cuda` 计算图，另有 `ggml_cpu` 作为标量正确性通道，
以及自己的纯 C# 与 Direct CUDA 执行器作为可移植性通道。它们都没有
注册进 `EnvVarMatrix.All`，因此默认的 TestMatrix 扫描不会覆盖。完整背景见
[V4 卡片](models/deepseek4_zh-cn.md)与 [V4.1 卡片](models/deepseek41_zh-cn.md)。

| 变量 | 适用范围 | 作用 | 默认值 | 在矩阵中 |
|---|---|---|---|---|
| `TS_DSV4_UBATCH` | V4 与 V4.1 | Prefill 微批宽度。不设置时，V4.1 在 ggml GPU 后端上由原生加载器在 1024、512、256 中选择：取所需路由专家 CPU 层数不多于 256（或显式 `--n-cpu-moe`）的最宽者，并记录为 `[dsv4] prefill ubatch: N (auto; ...)`。驻留 GPU 的路由专家层每个分块的耗时在各宽度下相近，因此越宽每个 prefill token 越便宜。任何显式的正整数都原样使用并关闭自动选择；`256` 恢复此前固定的 V4.1 默认值 | V4.1：ggml GPU 后端上自动，CPU 执行器与 direct CUDA 为 `256`；V4：`1024`（纯 C# 执行器为 `512`） | 否 |
| `TS_DSV4_THREADS` | V4 与 V4.1 | 纯 GPU 加载时的原生线程池。CPU 专家卸载改用探测到的可用并行度，由 `--cpu-moe-threads N` / `TS_CPU_MOE_THREADS` 设定。在纯 C# 的 `--backend cpu` 执行器上，它设定的是该执行器自己的工作线程池，默认取 `ProcessorCount` 而不是 min(核数, 32) | min(核数, 32) | 否 |
| `TS_DSV4_PERF` | V4 与 V4.1 | `1` 打印分阶段耗时 | 关 | 否 |
| `TS_DSV4_VRAM_RESERVE_MB` / `TS_DSV4_GRAPH_CACHE` / `TS_DSV4_LOAD_THREADS` / `TS_DSV4_LOAD_CHUNK_MB` / `TS_DSV4_MOE_MMAP` | V4 与 V4.1 | 放置余量、计算图缓存深度、权重加载并行度，以及驻留主机的专家是否直接在 GGUF 映射上就地相乘 | 见各卡片 | 否 |
| `TS_DSV41_RETAINED_CACHE_MB` | V4.1，原生执行器 | 为已结束会话的下一轮保留原生槽位的预算（保留始终开启，只在加载了 DSpark 草稿器时不生效）；不是正数的值保持默认 | `2048` | 否 |
| `TS_DSV41_TP_HOST_TOKENS` | V4.1 CUDA routed-MoE TP，仅主机驱动节点（`TS_DSV4_FUSED=0`） | 默认的 `--tp` 路径在融合 CUDA 后端上把每次 rank 交换都经 pinned 内存放在各 GPU 自己的队列上完成，不读取此变量。主机驱动节点上：`0`–`4096` 的整数：不超过此 token 数的批次使用精确 pinned 主机激活拼接，更大批次使用可用的私有 F32 NCCL。`0` 强制使用可用的设备拼接。自动阈值仅在 NCCL 选择 `NCCL_P2P_DISABLE=1` 时取 `16`，依据六张 A40 的配对实测；其他配置取 `0`。`TS_GGML_TP_F32_NCCL=0` 仍使全部批次回退到主机。 | 自动 | 否 |
| `TS_DSV41_ENGRAM_DEVICE` | V4.1（ggml 后端与 `--backend cuda`） | `1` 要求 Engram 表驻留 GPU，放不下就失败；`0` 强制主机映射，需要与 CPU oracle 逐位一致时也用它。不设置则自动且保守：只要不会因此逼出路由专家的 CPU 卸载，就放在 GPU 上 | 自动（放得下就驻留 GPU） | 否 |
| `TS_DSV41_ENGRAM_WARM` | V4.1，仅主机映射 | 读入 Engram 表页，使查表变成一次内存读取而不是一次存储往返。未设置时在模型开始服务后于**后台**预热（默认）；`1` 与此前一致在加载期间同步预热；`0` 从不预热。八卡 A40、Q4_K_M 上 prefill 从 200–252 提升到 452–492 tok/s，decode 从 23–26 提升到 31–33 tok/s；在默认的 GPU 驻留路径上无意义。代价是表本身的主机页缓存（Q2_K 60 GiB，Q4_K_M 103 GiB）以及读表的时间：七卡 A40 通道上同步形式用旧的逐页遍历读 103 GiB 花了 311.3 s；`pread` 预热（`TS_DSV4_WARM_PREAD`）在该存储上实测 2.24–2.54 GiB/s，即约 41–46 s 的读取 | 后台预热 | 否 |
| `TS_DSV4_GRAPH_CACHE_HEADROOM_MB` | V4 与 V4.1 | 图缓存必须为下一张图留出的设备内存，叠加在它持有的最大条目之上。会逐个释放最久未使用的条目直到满足。`0` 回到纯条目数上限，而四个并发的 10.8k token prefill 曾因此耗尽显存 | `1024` | 否 |
| `TS_DSV41_ENGRAM_THREADS` | V4.1，仅主机映射 | 常驻查表工作线程数，`1`–`32`。单个 token 在每张表上要取 24 行互不相关的数据，串行读意味着串行缺页 | min(16, 硬件线程数) | 否 |
| `TS_DSV41_ENGRAM_RANDOM` | V4.1，Linux 上的主机映射 | 对映射的 Engram 区间给出随机访问建议。`0` 关闭，`1` 强制 | 自动 | 否 |
| `TS_DSV4_LOAD_CONTIGUOUS` | V4 与 V4.1 | `0` 让权重加载器退回到从共享游标分发分块任务，每个读取线程会以 `线程数 x 分块` 的步幅跨越文件，而不是读一段连续区间。保留它只是为了能对连续读取的默认行为做 A/B：在 MooseFS 挂载上实测慢 2.5 倍（八卡 A40 加载 Q4_K_M 发行版 363–382 s，对比 144–155 s） | 开 | 否 |
| `TS_DSV4_WARM_PREAD` | V4 与 V4.1 主机映射；V4.1 TP 路由权重加载 | 控制主机专家预取（`--n-cpu-moe`）、同步/后台 Engram 预热，以及 V4.1 TP 在上传 GPU 前对当前层路由 gate/up/down 源权重的预读。未设置或 `1` 使用有界并行 `pread`，线程数由 `TS_DSV4_LOAD_THREADS` 决定；跳过 `mincore` 报告已驻留的区间，读取错误包含分片与偏移。TP 在打开当前层的源映射后仅预读该层，不预读整个检查点，也不通过此路径预热 Engram 表。`0` 关闭 TP 源预读，并恢复既有主机专家/Engram 逐页触碰遍历（预取为 256 MiB 段，Engram 为 8 MiB 块）。历史七卡 A40 的 8 GiB 区间测试中，`pread` 为 2.24–2.54 GiB/s，逐页遍历为 0.62–0.74 GiB/s；这不是 TP 模型加载提速的测量，也不保证冷存储性能 | 开 | 否 |
| `TS_DSV4_LOAD_DROP_CACHE` | V4 与 V4.1 的普通逐层上传 | 每个权重分块上传到设备后是否释放它的页缓存。V4.1 的专用 TP 路由专家上传路径不使用此设置，已上传的源数据页仍保留在可回收的文件缓存中。未设置时自动判断：当上传字节数加上主机映射的权重（专家、Engram 表）再加 8 GiB 超过主机额度（cgroup 上限）时释放，否则保留，额度未知时也保留，因此纯 GPU 驻留的检查点重新加载时仍是热的；加载时会打印这一判断及三个数值。`1` 总是释放，`0` 从不释放（此前的默认）。七卡 A40 通道（上传 263.0 GiB + 映射 151.2 GiB，对比 326.9 GiB）会释放。在 MooseFS 挂载上释放每个已驻留的 64 MiB 分块耗时 5.9–7.3 ms，也无法让上传本身变快；预期收益在其后的预取与 Engram 预热上 | 自动 | 否 |
| `TS_DSV41_REWIND_CHECKPOINT` | V4.1，原生执行器 | `0` 去掉逐槽位的回退检查点（在每个 prompt 边界对原始滑动窗口环与压缩器状态环做的影子拷贝）。没有它，部分 KV 复用最多只能回退到活动环还覆盖的位置——发布的 checkpoint 上是 385 个位置——多轮思考对话因此会重新 prefill。每个序列槽位约占 21 MiB 显存 | 开 | 否 |
| `TS_DSV41_SPARSE_FA` | V4.1 | 稀疏 prefill attention。在自有 F32 CUDA 路径上**默认开启**，作用于超过 8 个 query、至少 2,048 个 key 的调用：每个 query 只关注它的滑动窗口加索引器选中的行（最多 640 个 key），而不是全部 key。CUDA 用 MQA 内核执行这类调用，每个 key 分块只加载一次、供四个头共用：A40 上 1,024 个 query 恒为 16.1–16.3 ms，而分块（tiled）在 2,560 个 key 时为 22.5 ms、6,656 个 key 时为 63.8 ms（512 × 8,960：8.0 ms，分块 108 ms），与 F32 参考的相对 L2 偏差在 5.3e-7 以内；decode、DSpark verify 与低于阈值的提示词保持稠密内核、逐位不变。`0` 恢复分块 prefill。`1` 另外让 ggml flash attention 路径（非 CUDA GPU、CPU 后端）在单个 query 或至少 16,384 个 key 时使用其掩码压缩内核，该内核有已记录的 F16 差异 | 自有 CUDA 路径默认开；ggml flash attention 默认关 | 否 |
| `TS_DSV41_COMPACT_RAW_GATHER` | V4.1 | `1` 为原始滑动窗口选择紧凑 gather。同样需显式开启，同样有浮点差异 | 关 | 否 |
| `TS_DSV41_MQA_DECODE` | V4.1 自有 F32 CUDA attention | V4.1 的 64 个头共用一个宽 512 的 F16 K=V 头，因此 CUDA 用 MQA 内核执行这类 attention：一个 block 每个 key 分块只加载一次、供四个头共用，可见 key 分到四个流上处理，被掩码的 key 根本不加载。decode 级调用（1–8 个 query）与压缩 prefill 都使用它，全程 F32，跨调用宽度仍逐位一致。A40 上一层 decode 实测 171 -> 42 us（稀疏 prefill 512 × 8,960 个 key：34.4 -> 8.0 ms）。`0` 恢复逐头内核，用于 A/B 与诊断 | 开 | 否 |
| `TS_DSV41_KV_MIRRORS` | V4.1 多 GPU（层切分，以及 attention 仍按层放置的 `--tp`） | 在每个读取共享压缩 / 索引器 cache 的 GPU 上各保留一份副本（每步只复制源层新写入的行），使层读取本 GPU 的副本，而不是经主机把选中的行或整个可见 cache 跨 PCIe 拉过来。两种方式 logits 逐位相同；5x A40 上 2K 上下文的 decode 从 32.6 提到 34.5 tok/s。无法分配的副本会被跳过并给出警告。`0` 跨 GPU 读取 | 开（多 GPU） | 否 |
| `TS_DSV41_YOCO_KEEP` | V4.1 prefill（ggml 执行器：ggml_cuda、ggml_cpu） | 有界重放，参照 DeepInfer 的 YOCO prefill 截断。**默认开启**，取 DeepInfer 的 140。从第一个 ratio 为 1 的层（40 层中的第 20 层）起，每一层读取的都是第 20 层由其 attention 输入写入的压缩 cache，因此一次多 token 调用只让这些层处理最后 N 个位置，每个重放位置的滑动窗口被截到重放范围之内，所以 prefill 是**近似**的：`0` 让每一层处理每个位置；其他值至少为滑动窗口 + 8（136）。decode 步、DSpark 验证、含图像 token 的调用与单 token 的 prefill 微批始终让每一层处理每个位置（`TS_DSV4_UBATCH=1` 时默认值提示重放不生效，显式设值则拒绝加载）。滑动窗口宽于 132 的检查点默认至少保留窗口 + 8。回退到一次长调用被跳过的位置（比其末尾早 13 个 token 以上，且在其起点之后）会重新 prefill 而不是复用 cache，并记录一行日志。服务端按每次 prefill 调用（即长提示的每个分块，最多 4,096 个 token，并发时更少）分别应用。布局无法重放的检查点（第 20 层之后自带压缩 cache 的层、位于第 20 层或更深层的 Engram 表、索引器候选源早于第 20 层、DSpark 草稿器读取第 20 层或更早的层）在默认值下以一条提示改走精确 prefill，显式设值则拒绝加载。`--backend cuda` 与 `--backend cpu` 未实现它：各提示一次，并拒绝显式的非零值。5x A40（UD-IQ2_XXS，随机 token）上取 140：`--layer-split 5` 时 2,048 个 token 的 prefill 由 463.5 提到 844.3 tok/s，8,192 个 token 由 444.3 提到 906.2；`--tp 5` 时分别由 562.9 提到 1,046.7、由 540.8 提到 1,082.7。wikitext-2 4,608 token 窗口（prefill 2,304 个，其余逐 token 打分，10 段）：PPL 2.4820 -> 2.4970（+0.6%），相对 llama.cpp 的平均 KLD 0.0160 -> 0.0172，集中在提示之后最先生成的 token 上（前 128 个为 0.026，精确 prefill 为 0.015） | `140`（开） | 否 |
| `TS_DSV41_ALLOW_NON_CUDA_GPU` | V4.1 | `1` 允许 `ggml_vulkan` / `ggml_metal`：普通计算图跑在 GPU 上，只有架构专属算子回退到 CPU 后端，每次都要一次主机往返。之所以需要显式开启，是因为它解除的那道拒绝原本挡住的是*静默*回退 | 关 | 否 |
| `TS_DSV41_VISION_FA` / `TS_DSV41_VISION_BF16_GEMM` | V4.1 视觉伴随文件 | `TS_DSV41_VISION_FA=1` 让图像编码器使用 F16 中间量的 flash attention（真实图像上的特征差异更大）；`TS_DSV41_VISION_BF16_GEMM=0` 选择诊断用的 F32 提升矩阵路径 | 稠密 F32 注意力，BF16 GEMM + F32 累加 | 否 |
| `TS_DSV41_TRACE_DIR` / `TS_DSV41_VISION_TRACE_DIR` | V4.1（诊断） | 文本计算图与视觉编码器的张量转储目录。两者都会保留中间张量并增加设备传输——跑基准时保持不设置 | 未设置 | 否 |
| `TS_DSV4_CPU_TRACE_DIR` / `TS_DSV4_CUDA_TRACE_DIR` | V4.1（诊断） | 纯 C# 与 Direct CUDA V4.1 执行器的同类逐张量转储，采用同一种文件命名（`p<pos>_blk<NN>_<stage>.f32`，Direct 引擎选中的行与专家为 `.i32`）；`eng/validation/compare-dsv4-traces.py` 可逐张量比对其中两份，或其中一份与 `TS_DSV41_TRACE_DIR`（其 `v41.<NN>.<stage>` 命名会被映射到前者），定位第一个发散的张量。转储只是把各阶段复制出来：logits 逐位不变 | 未设置 | 否 |

## 矩阵外的 GLM 5.x（`glm-dsa`）开关

这些变量配置 GLM 5.x（`glm-dsa`）执行器——`ggml_cuda` / `ggml_vulkan` /
`ggml_cpu` / `ggml_metal` 使用的原生整模型 ggml 路径，以及 `cpu` 与 `cuda` 使用的
托管逐算子路径。它们都未注册在 `EnvVarMatrix.All` 中，默认的 TestMatrix 扫描不会
覆盖；完整清单与背景见 [GLM 卡片](models/glm_zh-cn.md#环境变量)。张量并行相关的两个
开关（`TS_GLM_TP_SHARD`、`TS_GLM_TP_OVERSUBSCRIBE`）列在下面的 TP 表里。

| 变量 | 适用范围 | 作用 | 基线 | 扫描取值 | 在矩阵中 |
|---|---|---|---|---|---|
| `TS_GLM_NATIVE` | GLM 5.x | `0` 在 GGML 后端上改走托管逐算子路径而非原生整模型图——正是用来对照两条路径是否一致的 A/B | `1`（原生） | `0`, `1` | 否 |
| `TS_GLM_UBATCH` | GLM 5.x | Prefill 微批。显存允许时 `2048` 在长提示上更快：3x RTX PRO 6000 上 pp2048 为 1145.8，对比 918.9 t/s | `1024` | `512`, `1024`, `2048` | 否 |
| `TS_GLM_THREADS` | `ggml_cpu` 上的 GLM 5.x | CPU 后端线程数；开启 `--n-cpu-moe` / `--cpu-moe` 或没有 GPU 时改为全部可用 CPU，`--cpu-moe-threads`（其后是继承来的 `TS_CPU_MOE_THREADS`）可覆盖两者 | min(核数, 32) | — | 否 |
| `TS_GLM_OP_OFFLOAD` | GGML 上的 GLM 5.x | 调度器的 op-offload；一旦有任何层的专家驻留主机就会自动关闭 | 自动 | `0`, `1` | 否 |
| `TS_GLM_VRAM_RESERVE_MB` | GGML 上的 GLM 5.x | 按层切分在开始放层之前，为计算缓冲在每张卡上预留的余量 | `3072` | — | 否 |
| `TS_GLM_GRAPH_CACHE` | GGML 上的 GLM 5.x | 缓存多少张已构建且已分配的计算图，使相同形状可以直接重放而不必重建 | `8` | — | 否 |
| `TS_GLM_NODES_PER_LAYER` | GGML 上的 GLM 5.x | 每 rank 每层的计算图节点预算 | `256` | — | 否 |
| `TS_GLM_MOE_MMAP` | 带 `--n-cpu-moe` 的 GLM 5.x | `0` 把驻留主机的专家拷进私有缓冲，而不是在 GGUF 映射上就地做乘法 | `1`（映射） | `0`, `1` | 否 |
| `TS_GLM_LOAD_THREADS` / `TS_GLM_LOAD_CHUNK_MB` | GLM 5.x | 权重加载的并行度与分块大小——16 个读线程跨 6 个分片，在页缓存预热的情况下约 37 秒读入 218 GiB（5.9 GiB/s） | `16` / `64` | — | 否 |
| `TS_GLM_TRACE` | GLM 5.x（诊断） | 指定层列表（或 `all`）按 `llama-eval-callback` 的排版打印逐层激活和，用于与 llama.cpp 对拍 | 未设置 | — | 否 |
| `TS_GLM_BD_DEBUG` | GLM 5.x（诊断） | `1` 逐步叙述每次批处理解码：参与的是哪些槽位、计算图是复用还是重建、跑到了哪一步 | `0` | `0`, `1` | 否 |
| `TS_GLM_DEBUG` / `TS_GLM_DEBUG_LAYERS` | 托管逐算子路径上的 GLM 5.x（`cpu` / `cuda` / `TS_GLM_NATIVE=0`，诊断） | 逐层激活追踪：打印每个具名中间张量的形状、求和与前几个值，标签与 `llama-eval-callback` 对齐，便于逐标签对拍。`TS_GLM_DEBUG=1` 只追踪第 0 层，`TS_GLM_DEBUG_LAYERS` 接受层列表。原生执行器请改用 `TS_GLM_TRACE` | 未设置 | — | 否 |

## 矩阵外的张量并行 / 分布式推理变量

这些变量配置张量并行（把单个模型切分到多张 GPU）以及基于点对点 TCP 网格的多节点
分布式 TP。它们未注册在 `EnvVarMatrix.All` 中，也不在默认 TestMatrix 配置里扫描
——TP 需要多张 GPU，而标准的单 GPU 测试环境无法覆盖。TP 可运行在直连 `cuda` 后端
以及 GGML CUDA / Vulkan 后端（`ggml_cuda`、`ggml_vulkan`）上。
`TENSORSHARP_TP_DEGREE`、`TENSORSHARP_TP_NODE_ID` 与 `TENSORSHARP_TP_PEERS` 也可
通过 `TensorSharp.Cli` 与 `TensorSharp.Server` 的 `--tp`、`--tp-node-id`、
`--tp-peers` 参数设置。

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TENSORSHARP_TP_DEGREE` | 全部自回归模型；`cuda`、`ggml_cuda`、`ggml_vulkan` 后端 | 把模型切分到本机多少张 GPU（Megatron-LM 列/行并行） | `1`（单 GPU） | 未注册 | 否 |
| `TENSORSHARP_LAYER_SPLIT_DEGREE` | 支持按层切分的架构 | 本地整层放置的 GPU 数，等价于 `--layer-split N`；与 TP 参数互斥 | `1` | 未注册 | 否 |
| `TENSORSHARP_LAYER_SPLIT_DEVICES` | Qwen 3.8 Flash Next 的共享 GGML 按层执行器 | 逗号分隔的设备序号，例如 `0,2`；独立于 `TENSORSHARP_TP_DEVICES`；原生 GLM/DeepSeek 使用 `CUDA_VISIBLE_DEVICES` | `0..N-1` | 未注册 | 否 |
| `TENSORSHARP_TP_DEVICES` | GGML 后端上的本地 TP | 各 rank 使用的 GPU 序号（逗号分隔，例如 `0,2`）。Muse Spark 的原生 `--tp` 使用前 N 张 CUDA GPU，设置了该变量时拒绝加载；那里请用 `CUDA_VISIBLE_DEVICES` | `0..tp-1` | 未注册 | 否 |
| `TENSORSHARP_TP_NODE_ID` | 支持分布式 TP 或层切分的架构和后端 | 本节点的 0 起始编号；必须与 `TENSORSHARP_TP_PEERS` 一起设置 | 未设置（关闭） | 未注册 | 否 |
| `TENSORSHARP_TP_PEERS` | 支持分布式 TP 或层切分的架构和后端 | 所有节点的 `host:port` 列表（逗号分隔）；必须与 `TENSORSHARP_TP_NODE_ID` 一起设置；IPv6 使用 `[address]:port` | 未设置（关闭） | 未注册 | 否 |
| `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS` | 分布式 TP 和层切分 | 连接重试、入站连接与网格握手的总期限 | `120` 秒 | 未注册 | 否 |
| `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS` | 分布式 TP 和层切分 | peer 套接字读写无数据超时；心跳可保持空闲控制等待 | `300` 秒 | 未注册 | 否 |
| `TENSORSHARP_TP_OPERATION_TIMEOUT_SECONDS` | 分布式 TP 和层切分 | 完整数据帧的期限，包含仅收到心跳和分段传输的时间；空闲控制等待和加载屏障除外 | 接收超时 | 未注册 | 否 |
| `TENSORSHARP_TP_SOCKET_BUFFER_BYTES` | 分布式 TP 和层切分 | 可选的套接字收发缓冲区大小，至少 4096 字节；应根据实际网络测量调优 | 操作系统默认值 | 未注册 | 否 |
| `TENSORSHARP_TP_DISABLE_P2P` | 本地 TP，`cuda` 后端 | `1` 表示所有跨 GPU 传输一律经主机中转，不使用 CUDA 点对点 DMA（与 A16 vGPU 等无 P2P 硬件一致） | 关闭（通过 DMA 自检的设备对使用 P2P） | 未注册 | 否 |
| `TENSORSHARP_TP_HOST_ALLREDUCE` | 本地 TP，`cuda` 后端 | `1` 表示本地 AllReduce 走主机内存（设备→主机、求和、主机→设备）而非设备到设备路径——诊断兜底 | 关闭（设备到设备） | 未注册 | 否 |
| `TS_GGML_TP_PARALLEL` | 本地 TP，GGML 后端 | `0` 表示顺序而非并发地驱动各 rank（诊断用） | 开启（并发 rank 工作线程） | 未注册 | 否 |
| `TS_GGML_TP_FUSED_MATMUL` | 已移除 | 原为 `1` 表示由单个线程提交两个 rank 的线性层（从不是默认值；在 Qwen 3.5 35B 上实测慢 2.3×）。现在设置它会使主机带着建议退出 | — | 已移除（硬错误） | 否 |
| `TS_GGML_TP_DEVICE_AR_THRESHOLD` | 本地 TP，GGML 后端 | 超过该元素数量时 AllReduce 走设备集合通信，否则在主机内存中归约 | `262144` | 未注册 | 否 |
| `TS_GGML_F32_RESIDENT` | GGML 后端 | `0` 表示每次调用重新绑定 F32 线性层权重，而不是常驻设备（诊断用） | 开启（常驻设备） | 未注册 | 否 |
| `TS_GEMMA4_TP_FUSED_MOE` | GGML 上 TP 下的 Gemma 4 MoE | `0` 表示从融合的整模 MoE 主干（专家内部 Megatron 切分）回退到逐算子的整专家路径 | 开启（融合主干） | 未注册 | 否 |
| `TS_GLM_TP_SHARD` | GGML 上 TP 下的 GLM 5.x | 切分哪一半：`1` 注意力头，`2` 路由专家，`3` 两者都切。路由专家是在每个专家内部按行切分，而不是按专家 id 分配，因为 `ggml_mul_mat_id` 要求同一 token 选中的专家 id 互不相同 | `3`（两者） | `1`, `2`, `3` | 否 |
| `TS_GLM_TP_OVERSUBSCRIBE` | GGML 上 TP 下的 GLM 5.x | `1` 允许多个 rank 共享一张 GPU，用于在单卡机器上验证切分的正确性 | `0`（一 rank 一卡） | `0`, `1` | 否 |
| `TS_Q4E_Q80_PAIR` | Qwen 3.8 Flash Next（`qwen4exp`）直连 CUDA 引擎的 Q8_0 解码投影 | `0` 让解码步骤的 Q8_0 投影（1 到 16 行）回到“解码一次”稠密核（`TS_DENSE_FAST`）。开启时（sm_80 及以上 GPU 且加载 sm_80 及以上代码）改用成对矩阵向量乘：同一种算术由三个核实现，即 int8 张量核、每个块四条 lane 的 dp4a 孪生核（1 到 4 行）和每个块一条 lane 的 dp4a 核（1 到 8 行），按形状和行数选择最快的一个（`CudaKernels.Q80PairKernelFor`），因此批量中的每一行仍与该序列单独解码时完全一致。在 H100 NVL 上按该模型的形状测得，一个解码步骤的 Q8_0 投影在 1 / 2 / 4 / 8 行时耗时 3.7 / 4.4 / 5.7 / 7.5 ms，稠密核为 4.3 / 4.6 / 5.8 / 9.1 ms。它与稠密核的求和顺序不同，所以结果在最后几位上有差异。自 2026-11-09 起，两类权重在任意行数下都保留稠密核：`TS_Q4E_HC_Q8` 为超连接 down 混合器（10240 -> 320）生成的 Q8_0 副本，以及 MTP 块的 Q8_0 权重。在 RTX 5090 上（于 lane 核加入之前测得），成对矩阵向量乘从三行起处理这些权重要慢约 2 倍；MTP 块只处理 1-5 行。原生 Q8_0 检查点的主干投影与 up 混合器仍用成对矩阵向量乘。每个权重在任意行数下只走一种核 | 开 | 未注册 | 否 |
| `TS_Q80_PAIR_SWAP` | Qwen 3.8 Flash Next（`qwen4exp`）直连 CUDA 引擎的 Q8_0 解码投影，sm_80 及以上 | `0` 不做打包，成对矩阵向量乘保留基于 GGUF 布局的三个核。开启时，成对核处理的每个稠密 Q8_0 权重在加载时得到一份打包副本（再占一份权重大小；两块 H100 NVL 中最后一块上约 2.4 GB），调优可以为其选用交换操作数的张量核，结果逐位相同：在 H100 NVL 上 8 行时比其他核快 1.1-2.0 倍。 |
| `TS_CUDA_PDL` | sm_90 及以上并加载 sm_90 代码时的直连 CUDA 核（`tensorsharp_kernels`、`dsv4`、`glm`、`q4e` 模块） | `0` 普通启动所有核。开启时使用可编程依赖启动（每个核在第一条指令处等待前一个网格完成），输出不变。两块 H100 NVL 上 Flash Next 单流快 21%。 |
| `TS_Q4E_ATTN_GQA` | Qwen 3.8 Flash Next（`qwen4exp`）直连 CUDA 解码注意力 | `0` 保持每个查询头一个块。开启时一个块处理同一 KV 头的四个查询头，每个缓存单元的 K/V 只加载一次（批量步骤 4 行及以上、单序列调用 8 行及以上），结果逐位相同。 |
| `TS_Q4E_BATCHED_ATTN` | Qwen 3.8 Flash Next（`qwen4exp`）直连 CUDA 批量解码 | `0` 让批量步骤的注意力逐个序列发出。开启时每个核对所有行只启动一次，每行的缓存来自指针表，结果逐位相同。 |
| `TS_Q4E_EXPERT_VEC` | Qwen 3.8 Flash Next（`qwen4exp`）直连 CUDA 引擎的 Q8_0 解码专家 | `0` 让 Q8_0 解码专家回到共享解码核（gate/up 每行十六个 lane，较短的 down 行八个 lane）。开启时一个 warp 用每 32 值块四个 lane 读取一行权重（与稠密矩阵向量乘相同）；无论行数多少，每个 slot 的计算都相同，但求和顺序与共享核不同；通过驻留表换入的专家与常驻专家计算相同 | 开 | 未注册 | 否 |
| `TS_MOE_Q80_LANES` | Qwen 3.8 Flash Next（`qwen4exp`）直连 CUDA 引擎的 Q8_0 解码专家（`TS_Q4E_EXPERT_VEC` 开启时） | `0` 让每个权重行在任何 slot 数下都由一个 warp 计算。开启时，一行由八个 lane 计算（一个 warp 四行），每个 lane 把整个 34 字节的块按对齐的 64 位字读取：down 投影在任何 slot 数下都如此，gate/up 从 40 个 slot 起（十个专家的四个解码行），且一个 slot 的 gate 行和 up 行在同一组 lane 中。求和及其顺序与每行一个 warp 的内核相同，因此结果不变。H100 NVL，2560 <-> 640，10 / 40 / 160 个 slot 每层 us：down 20.2 / 46.7 / 188.0 -> 13.4 / 30.4 / 111.6，gate/up 17.4 / 56.1 / 232.7 -> 17.4 / 52.0 / 186.3 | 开 | 未注册 | 否 |
| `TS_MOE_GATEUP_Q80_PAIR` | Qwen 3.8 Flash Next（`qwen4exp`）直连 CUDA 引擎的 Q8_0 解码专家（`TS_Q4E_EXPERT_VEC` 开启时） | `0` 让每个 slot 的 gate 行和 up 行在任何 slot 数下都由各自的 warp 计算。开启时，一个 slot 的 gate 行和 up 行一起计算，激活块只加载一次：从 40 个 slot 起每行八个 lane（`TS_MOE_Q80_LANES`），关闭它时从 32 个 slot 起（十个专家的四个解码行）在同一个 warp 中；每行的求和与之前相同，因此结果不变。H100 NVL，2560 -> 640，两行共用一个 warp：160 个 slot 每层 200 对 233 us，40 个 slot 51 对 56 | 开 | 未注册 | 否 |
| `TS_KDA_SCAN_R4` | Qwen 3.8 Flash Next 与 GLM 5.x 直连 CUDA 引擎（门控 delta 递推） | `1` 让所有扫描每个 warp 处理四行状态。未设置时，预填充 ubatch 的扫描每个 warp 处理一行状态（在每行按顺序走完 ubatch 时可容纳四倍的 warp），并预取下一行的输入；解码步骤仍每个 warp 四行。两者对每个状态元素的计算相同（逐位一致） | 关 | 未注册 | 否 |
| `GGML_CUDA_ALLREDUCE` | 本地 TP，`ggml_cuda` | `nccl` / `internal` / `none` —— 直接透传给 ggml 的集合通信选择；显式设置同时会跳过启动前探测 | 自动（构建时能找到 NCCL 且通过探测就用 NCCL） | 未注册 | 否 |
| `TS_GGML_TP_CUDA_GRAPHS` | 本地 TP，`ggml_cuda` | `0` 关闭多 GPU 运行下的 CUDA graph 捕获。TP 下默认**开启**捕获：一个张量并行 token 是几十次按 rank 的小提交，重放的代价远低于重新下发（4×A40：Qwen3.5-9B tp4 88 → 128.5 tok/s，Qwen3.5-35B-A3B tp2 71.3 → 104.1）。历史上曾因捕获污染的隐患而禁用，那个隐患已不再成立——ggml 用 `cudaStreamCaptureModeRelaxed` 捕获。这个 opt-out 会在第一次后端调用之前翻译成原生的 `GGML_CUDA_DISABLE_GRAPHS`，因为 ggml 会在首次使用时锁定该值 | 开启捕获 | 未注册 | 否 |
| `TS_GGML_TP_AR_PROBE` | 本地 TP，`ggml_cuda` | `0` 跳过两项启动前探测；`force` 忽略缓存的判定（`~/.cache/tensorsharp/tp-collective-probe`）重新探测。模型加载前，进程组会检查两件事：所宣称的设备对之间 peer copy 是否真的把数据送到，以及一次小型 NCCL AllReduce 能否端到端完成——一些云主机声称支持 P2P 但数据永远送不到，NCCL 的第一次集合通信随后会让每块 GPU 永远空转。peer 检查失败时会保留 NCCL 但拿掉它的 peer 传输（`NCCL_P2P_DISABLE=1`），这正是超过 2 张 GPU 时仍能保住设备集合通信的原因 | 探测开启，判定按 驱动/NCCL/GPU 组合缓存 | 未注册 | 否 |
| `TS_GGML_TP_AR_PROBE_MS` | 本地 TP，`ggml_cuda` | 每项探测（先 peer copy，后 AllReduce）的完成期限，超时即判定该传输不可用；集合通信随后在 2 张 GPU 时回退到钉页主机内存的 `internal` 管线，更多卡时回退到主机归约。`0` 关闭探测 | `10000` 毫秒 | 未注册 | 否 |
| `TS_GGML_TP_F32_NCCL` | 对精度敏感的本地 TP，`ggml_cuda` | `0` 禁用 TensorSharp 自有的 F32 NCCL 路径，供诊断使用。对精度敏感的通用计划回退到保持 F32 的分块归约或主机归约；DeepSeek V4.1 回退到主机行收集。默认使用不压缩的 NCCL 操作，其中 DeepSeek 不等宽激活分片使用逐位保真的 AllGather，并遵守显式的 `GGML_CUDA_ALLREDUCE=internal/none`。不改变普通 TP 计划。 | NCCL 可用时开启 | 未注册 | 否 |
| `GGML_CUDA_AR_BF16_THRESHOLD` | 本地 TP，`ggml_cuda` | ggml 在多大载荷以上把 F32 集合通信转成 BF16；TensorSharp 把 ggml 的默认值提高到 1 MB，使 decode 规模的归约保持精确 | `1 MB`（由 `TSGgml_TensorParallelInit` 设置） | 未注册 | 否 |

## 矩阵外的 Redis 共享状态变量

这个变量配置可选的 Redis 状态，它未注册在 `EnvVarMatrix.All` 中，`--redis-url` 会设置它。Redis 不保存任何 KV 状态。

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_RESPONSES_STORE_REDIS_URL` | 仅服务端 | Responses API 存储的 Redis 连接串；设置后取代内存存储 | 未设置（关闭） | 未注册 | 否 |

## 矩阵外的遥测（OpenTelemetry）变量

这些变量配置 `TensorSharp.Server.Host` 与 `TensorSharp.Cli` 可选的 OpenTelemetry 导出（trace、指标和
日志）。它们改变的是哪些数据离开进程，而不是模型的计算结果。它们未注册在 `EnvVarMatrix.All` 中，也不参与
sweep。runner 也不会清除它们，所以 runner 自身环境中设置的值会传到每个 `TensorSharp.Cli` 子进程，由子进程
各自导出遥测。两个宿主上的 `--otel-endpoint <url>` 都会设置 `OTEL_EXPORTER_OTLP_ENDPOINT`。开关类变量接受
`1`、`true`、`yes` 或 `on`。完整说明（包括记录哪些 trace、哪些内容会被脱敏）见
[遥测（OpenTelemetry）](../USAGE_zh-cn.md#遥测opentelemetry)。

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | 服务端、CLI | 开启遥测，并通过 OTLP 把三种信号发往这个绝对 `http://` / `https://` URL；`http/protobuf` 下 SDK 会追加 `/v1/traces`、`/v1/metrics`、`/v1/logs`。其他形式是配置错误（退出码 1） | 未设置（关闭） | 未注册 | 否 |
| `OTEL_EXPORTER_OTLP_{TRACES,METRICS,LOGS}_ENDPOINT` | 服务端、CLI | 只为该信号走 OTLP，URL 按原样使用；会开启遥测 | 未设置 | 未注册 | 否 |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | 服务端、CLI | `grpc` 或 `http/protobuf`；有信号通过 OTLP 导出时，其他取值是配置错误，否则不读取该变量 | `grpc` | 未注册 | 否 |
| `OTEL_EXPORTER_OTLP_HEADERS` | 服务端、CLI | 每次 OTLP 导出附带的 `key=value,...` 请求头（例如 Aspire 看板的 `x-otlp-api-key=<key>`）；从不写入日志 | 无 | 未注册 | 否 |
| `OTEL_EXPORTER_OTLP_TIMEOUT` | 服务端、CLI | OTLP 导出超时（毫秒），由 SDK 读取 | `10000` | 未注册 | 否 |
| `OTEL_EXPORTER_OTLP_{TRACES,METRICS,LOGS}_{HEADERS,PROTOCOL,TIMEOUT}` | 服务端、CLI | 按信号覆盖上面三项，只在该信号通过 OTLP 导出时读取；此时协议无效或超时不是正整数毫秒是配置错误 | 沿用通用设置 | 未注册 | 否 |
| `OTEL_{TRACES,METRICS,LOGS}_EXPORTER` | 服务端、CLI | 单种信号除 Application Insights 之外的去处，逗号分隔：`otlp`、`console`、`none`。`otlp` 或 `console` 会开启遥测；`none` 让该信号在所有去处（包括 Application Insights）关闭；无法识别的名称会被忽略并给出警告 | 设置了端点或 `TENSORSHARP_OTEL` 时走 OTLP | 未注册 | 否 |
| `OTEL_SDK_DISABLED` | 服务端、CLI | `true` 时无论其他设置如何都保持关闭 | 未设置 | 未注册 | 否 |
| `OTEL_SERVICE_NAME` | 服务端、CLI | 所有信号上的服务名（在 Application Insights 中为 `cloud_RoleName`） | `tensorsharp-server` / `tensorsharp-cli` | 未注册 | 否 |
| `OTEL_RESOURCE_ATTRIBUTES` | 服务端、CLI | 额外的 resource 属性，优先于 TensorSharp 自带的属性（`service.instance.id` = `<机器名>:<pid>`、`host.name`、`process.pid`、`tensorsharp.tp.*`） | 无 | 未注册 | 否 |
| `OTEL_TRACES_SAMPLER` / `OTEL_TRACES_SAMPLER_ARG` | 服务端、CLI | trace 采样。OTLP 下由 SDK 读取；Application Insights 接收 trace 时映射到它的采样器，由它为所有 trace 导出器做决定（`traceidratio` / `microsoft.fixed_percentage` 取 0-1 的比例，`microsoft.rate_limited` 取每秒 trace 数） | `parentbased_always_on`（保留每条 trace） | 未注册 | 否 |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | 服务端、CLI | 开启遥测，并在进程内把每种已开启的信号发往 Application Insights；从不写入日志 | 未设置 | 未注册 | 否 |
| `TENSORSHARP_AZURE_MONITOR_AAD` | 服务端、CLI（需同时设置连接串） | 用 Entra ID（`DefaultAzureCredential`）代替连接串中的密钥向 Application Insights 认证 | 关闭 | 未注册 | 否 |
| `TENSORSHARP_OTEL` | 服务端、CLI | 开启遥测，OTLP 发往默认端点（`http://localhost:4317`；`http/protobuf` 时为 `http://localhost:4318`） | 关闭 | 未注册 | 否 |
| `TENSORSHARP_OTEL_LOG_LEVEL` | 服务端、CLI | 导出日志的最低级别（`Trace` ... `Critical`、`None`）；控制台和文件日志保持各自的级别。遥测开启时，其他取值是配置错误 | 与控制台 / 文件日志相同 | 未注册 | 否 |
| `Logging__OpenTelemetry__LogLevel__Default` | 仅服务端 | 通过 ASP.NET Core 配置设置同一级别（也可按类别设置）；与 `TENSORSHARP_OTEL_LOG_LEVEL` 二选一 | 与控制台 / 文件日志相同 | 未注册 | 否 |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | 服务端、CLI | `true` 时导出内容：不脱敏的日志、聊天 span 上的提示与输出消息、视频提示词、工具参数与结果、命令、上传路径 | 关闭（内容脱敏） | 未注册 | 否 |
| `TENSORSHARP_OTEL_TRACE_STEPS` | 服务端、CLI | 每个引擎步骤一个 `inference.step` span，并链接到它运行的请求；用于剖析单次运行 | 关闭 | 未注册 | 否 |

## 矩阵外的通用运行时开关

这些变量是真实运行时开关，但目前未注册到 `EnvVarMatrix.All`，也不在默认
TestMatrix 配置中 sweep。

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TENSORSHARP_UPLOAD_DIR` | `TensorSharp.Server` | 上传媒体与抽取出的视频帧所在目录；不可变部署时设为应用目录之外的绝对路径 | 服务端二进制旁的 `uploads` | 未注册 | 否 |
| `TS_UPLOAD_MAX_MB` / `TS_UPLOAD_QUOTA_MB` / `TS_UPLOAD_TTL_HOURS` | `TensorSharp.Server`（也可用 `--upload-max-mb`、`--upload-quota-mb`、`--upload-ttl-hours`） | 客户端上传的单文件上限（超过返回 HTTP 413）、上传目录的总预算（用尽时返回 HTTP 507），以及该目录中文件被删除前的存留时长。Jev 内联图片同样受这些限制约束 | `500` / 关 / 关 | 未注册 | 否 |
| `TS_LORAS_DIR` | `TensorSharp.Server`（也可用 `--loras-dir`，命令行优先） | 文本模型 LoRA 适配器目录（llama.cpp 的 `.gguf` 适配器，或张量名为 `<tensor>.lora_a` / `.lora_b` 的 `.safetensors`）。顶层的每个适配器都在启动时、模型加载之后加载；`POST /v1/loras` 可以再加载，但只能从该目录内、以相对路径给出；最多同时保留 32 个。不是已存在目录的路径会导致启动失败。空值视为未设置；嵌入服务忽略它 | 未设置（没有适配器；`POST /v1/loras` 返回 403） | 未注册 | 否 |
| `TENSORSHARP_PREFIX_CACHE_DIR` | `TensorSharp.Server` | 共享前缀检查点文件的根目录，使重启后不必再为共享提示做 prefill；每个模型有自己的 `<模型文件名>-<哈希>` 子目录，最多两个文件。`--no-prefix-cache` 关闭持久化。TensorAgent 把自己的检查点放在应用缓存目录下；CLI 只在内存中保留检查点 | 服务端二进制旁的 `prefix-cache` | 未注册 | 否 |
| `TS_GEMMA4_AUDIO_GGML_CUDA_PEDANTIC` | `ggml_cuda` 上的 Gemma 4 / EmbeddingGemma 2 音频图（进程启动时读取） | 未设置：conformer、注意力、深度卷积与 SSCP 图的 F32 矩阵乘在 TensorSharp 自有的 pedantic cuBLAS 句柄上计算。ggml-cuda 自己的句柄使用 TF32 tensor-op 数学模式，曾使 conformer 固件偏离双精度参照 5.8e-4。Q8_0 投影仍用 ggml 整数内核。`1` 另把 Q8_0 投影换成稠密 F32 投影（EmbeddingGemma 2 已验证的默认）。`0` 保留 ggml 的 TF32 cuBLAS 乘法，并关闭 EmbeddingGemma 2 的已验证 F32 策略 | pedantic F32 乘法 | 未注册 | 否 |
| `TS_H3_CUDA_F32_PRODUCTS` | `ggml_cuda` 上的 MiniMax-H3 DiT 前向（每个图读取） | `1`：DiT 前向的权重矩阵乘（F32/F16/BF16 权重与 LoRA 插件因子，含其行视图）改在 TensorSharp 自有的 pedantic cuBLAS 句柄上以 F32 GEMM 计算，而不用 ggml-cuda 的 TF32 张量核 cuBLAS；量化权重保留其内核，注意力自身的乘法保留 ggml 的。在全 F32 的 LoRA 一致性夹具上，原生步对直接 CPU 路径的底噪由 6.08e-4 降到 8.07e-6（只改 LoRA 乘法时仍为 6.08e-4）。没有 F32 路径的乘法、或无法创建句柄的设备，会在 stderr 上点名一次并在 ggml-cuda 上运行。真实检查点上的开销未测 | 未设置（ggml-cuda 的 TF32/BF16 乘法） | 未注册 | 否 |
| `TS_NO_MULTI_AGENT` | `TensorSharp.Server` 聊天端点 | 取 `0` 以外的任何值都会关闭子代理委派，等同 `--no-multi-agent` | 未设置（委派开启） | 未注册 | 否 |
| `TS_THINKING_BUDGET` | 开启思考的聊天轮次（`TensorSharp.Server` 与交互式 CLI） | 思考块最多可用的 token 数。有训练过的闭合 token 的系列（DeepSeek V4.1、Nemotron 3.5 / Omni、Qwen 3.5-3.8、GLM 5.x）随后闭合思考，并在 `max_tokens` 内作答；Qwen 和 GLM 会先写入 Qwen 的交接句。其他系列以结束原因 `thinking_budget` 停止该轮。无论上限如何，有闭合 token 的系列在模型于思考中抽到停止 token，或思考重复模板行时（`ReasoningLoopDetector`，32 行），都会闭合思考而不是结束该轮；仍以只有思考、没有答案结束的轮次报告 `thinking_budget`（issue #338）。`0` 只取消上限 | `max_tokens` 不小于 512 时为其 75%，否则无 | 未注册 | 否 |
| `TS_JEV_MAX_BODY_MB` / `TS_JEV_MAX_CANVAS` / `TS_JEV_MAX_PENDING` | `POST /v1/systemone`（Jev，DiffusionGemma） | 请求体上限（MiB，1-64）、每个问题分块的答案 canvas 宽度（token，8-4096，同时受检查点限制），以及同时可接纳的请求数（1-1024），超出后端点返回 HTTP 529。超出范围的值会报错而不是被截断 | `8` / `64` / `32` | 未注册 | 否 |
| `TS_NEMOTRON_AUDIO_MMPROJ` | 带 `--mmproj` 的 Nemotron-H | 改从该音频配套 GGUF（NVIDIA `sound_encoder.*` / `sound_projection.*` 张量）而不是 `--mmproj` 文件加载 Parakeet 音频塔，从而可以同时使用视觉 mmproj 与音频配套文件。除非该文件的张量校验通过，音频仍被拒绝（HTTP 400）；见 `docs/models/nemotron_zh-cn.md` §4.7 | 未设置（`--mmproj` 含这些张量时从中读取音频塔） | 未注册 | 否 |
| `TS_PDF_MAX_PAGES` | PDF 文档输入（CLI `--pdf`、服务端 `/api/upload`） | 文本提取与页面图像渲染读取的 PDF 页数上限 | `0`（全部页面） | 未注册 | 否 |
| `TS_GGUF_PREFAULT` / `TS_GGUF_PREFAULT_THREADS` / `TS_GGUF_PREFAULT_RESIDENT` | 通过 `GgufReader` 加载模型 | `PREFAULT=0` 跳过并行页缓存预热。所有分片所选张量总量共享进程内存预算一半的上限，跳过 Qwen 的稀疏 PLE 表。所选张量超过上限时，模型仍可预热加载时立即读取的子集：默认是路由专家（`*_exps*`）以外的全部张量，比内存大的 MoE 之后只读取其中一部分专家（geesefs 上的 Qwen3.8-Flash-Next：3.7 GiB 用 13 秒，原先是 137 秒的串行缺页）。所有读取都经过每个文件 32 个句柄的池，在读取任何张量数据之前打开一次，并在文件生命周期内保留：geesefs 这类 FUSE 挂载每次 `open()` 都会丢弃该文件的全部页缓存。读取线程数不随 CPU 配额变化（它们主要在等存储）；`THREADS` 最多 32。Linux 检查页驻留状态，避免复制已缓存的数据；`RESIDENT=0` 强制读取，供 A/B 测量。其他平台常规读取所选范围，iOS/tvOS 跳过预热。该预算不代表当前主机空闲内存。 | 开，32，开启驻留检查 | 未注册 | 否 |
| `TS_DUMP_LOGITS` | 所有模型、所有后端，CLI 与服务端 | 把**决定第一个生成 token 的 logits** 以原始 float32（小端）写入该路径，每个进程只写一次：即**整个提示词处理完之后的最后一行**，不论提示词走的是哪条路径——一次前向还是分成多块（前缀缓存检查点、递归块边界、分块上限）、逐序列 / fused / 批量 prefill、`--tp` / `--layer-split`、模型自带的引擎（q4e、GLM、DeepSeek V4 等），还是 CLI `--benchmark`（`Forward`，或带 `--bench-chunked` 时的 `ForwardRefill`）。它在引擎为请求采样第一个 token 的地方取得（以及 benchmark / `--test` 的提示词前向之后），因此**绝不会**是预热前向、decode 前向或预热请求：在预热中提交的请求（`--warmup-runs`、服务端与 TensorAgent 的前缀缓存预热、各 benchmark 的预热轮）在整个生命周期内都被排除——即使被取消的预热仍留在引擎里也一样——而与预热同时提交的真实请求不受影响。并发请求时，最先采样的那个请求会被导出，因此请用单请求的运行来对比。旁边的 `<路径>.json` 记录向量的 SHA-256、提示词长度与 token id 指纹、媒体片段数量与 LoRA 适配器（不对媒体**内容**做指纹）、前缀缓存复用量、每一次 prefill 前向所属的调度计划路径（只是标签而非实际执行的内核；CLI 的直接循环对整段提示词只记一步，因此不要据此断言前向次数），以及第一个 token。进程启动时（CLI 启动、每个推理引擎创建时）会删掉这两个文件，因此没跑到第一个 token 的运行不会留下旧文件。`eng/validation/compare-parallel-logits.py` 会拒绝比较：sidecar 指向不同提示词、媒体数量或适配器的两个向量；与旁边向量不符的 sidecar（过期 sidecar：哈希、长度或 argmax 不一致）；无法解析的 sidecar；加 `--require-sidecar` 时也拒绝没有 sidecar 的向量。（此前导出的是第一次非预热的 `ModelBase.Forward`：只要提示词用了不止一次前向，它就是提示词的**某一块**——若提示词根本不经过 `Forward`，则是第一步 decode——两种配置可能比较的是不同的计算。）这样就能用 logit 向量而不是生成文本来比较两个后端——贪心解码会把一次几乎打平的比分变成一句明显不同的话。DiffusionGemma 不写（没有自回归的第一个 token） | 未设置（不导出） | 未注册 | 否 |
| `TS_CUDA_QMM_F16GEMM_MIN_ROWS` | 直连 `cuda` 后端 | 激活行数达到该阈值的量化矩阵乘会把权重一次性反量化为 F16 并走张量核心 cuBLAS GEMM（ggml 风格的 prefill 路线），替代分块量化内核 | `32` | 未注册 | 否 |
| `TS_CUDA_QMM_F16GEMM_MAX_MB` | 直连 `cuda` 后端 | F16 权重暂存区上限（MB）；超过上限的权重（如 LM head）继续使用量化内核 | `768` | 未注册 | 否 |
| `TS_CUDA_Q80_VEC_MIN_OUT` | 直连 `cuda` 后端 | Q8_0 dp4a 矩阵-向量乘的最小输出宽度（诊断开关） | `0` | 未注册 | 否 |
| `TS_CUDA_Q80_MMQ_MAX_ROWS` | 直连 `cuda` 后端 | 激活行数在 32 到该值之间的 Q8_0 矩阵乘直接在原始 Q8_0 块上执行 int8 张量核心 GEMM（mma.m16n8k32，ggml MMQ 风格）；超过该值后 F16 GEMM 路线更优（MMQ 的权重扫描次数随 ceil(rows/128) 增长） | `512` | 未注册 | 否 |
| `TS_CUDA_PREFILL_GRAPH_MAX` | 直连 `cuda` 后端 | 缓存的 prefill + decode CUDA graph 数量（LRU 淘汰；每个 graph 固定持有其捕获时使用的内存池块）。Qwen 3.5 / 3.6 纯文本 prefill 与 decode 会把逐算子层循环捕获为 graph 并重放（结果逐位一致；捕获失败时回退普通路径） | `4` | 未注册 | 否 |
| `TS_CUDA_PREFILL_GRAPH_LOG` | 直连 `cuda` 后端 | 打印 graph 捕获/重放/中止事件（`1`） | 关闭 | 未注册 | 否 |
| `TENSORSHARP_CUDA_POOL_LARGE_MB` | 直连 `cuda` 后端 | 全局大块（≥ 2 MB）显存缓存预算；让 prefill 级激活保持池化，避免每层重复 cuMemAlloc/cuMemFree（每次释放都会同步设备） | 显存的四分之一，至少 `1024` | 未注册 | 否 |
| `TS_CUDA_PROFILE` | 直连 `cuda` 后端 | 退出时打印 CPU 回退算子与主机↔设备同步计数（`1`），含调用点归因（`2`） | 关闭 | 未注册 | 否 |

## 矩阵外的知识库配置

这些宿主设置不在 TestMatrix 中注册或扫描；它们配置[个人知识库](knowledge-system_zh-cn.md)，
不改变聊天模型的数值后端。启动宿主前设置；提取器或嵌入身份变化时需重新索引。
独立服务端通过 `TS_KNOWLEDGE_DIRECTORY` 启用；TensorAgent 使用自己的持久化资料库目录。

| 环境变量 | 适用范围 | 功能影响 | 默认值 | 扫描值 | 默认扫描 |
|---|---|---|---|---|---|
| `TS_KNOWLEDGE_DIRECTORY` | 独立服务端 | 启用本地资料库与 `/api/knowledge`；聊天仍显式选择 `knowledgeCollections` | 未设置（关闭） | 未注册 | 否 |
| `TS_KNOWLEDGE_EMBEDDING_MODEL` | 知识运行时 | 独立于聊天模型的本地句向量 GGUF | 未设置（仅 BM25） | 未注册 | 否 |
| `TS_KNOWLEDGE_EMBEDDING_BACKEND` | 知识编码器 | 编码器后端 | `CPU` | 未注册 | 否 |
| `TS_KNOWLEDGE_QUERY_PREFIX` / `TS_KNOWLEDGE_DOCUMENT_PREFIX` | 知识编码器 | 模型要求的前缀；更改会改变向量空间身份 | 均为空 | 未注册 | 否 |
| `TS_KNOWLEDGE_RERANK` | 已选知识的聊天 | `0` 关闭一次有界模型相关性排序；无效输出或超时保留检索顺序 | 有聊天模型且至少三个候选时启用 | 未注册 | 否 |
| `TS_KNOWLEDGE_PDF_OCR` | PDF 知识提取 | `auto`、`apple`、`tesseract` 或 `none`；`none` 保留可选择文本 | `auto` | 未注册 | 否 |
| `TS_KNOWLEDGE_PDFTOPPM_PATH` / `TS_KNOWLEDGE_TESSERACT_PATH` | 可移植 PDF OCR | 显式工具绝对路径；退出 TensorAgent 托管安装 | 查找本地工具或使用提供的托管安装器 | 未注册 | 否 |
| `TS_KNOWLEDGE_TESSDATA_DIRECTORY` | 可移植 PDF OCR | 显式语言数据目录；退出托管安装 | 提取器配置 | 未注册 | 否 |
| `TS_KNOWLEDGE_OCR_LANGUAGES` | 可移植 PDF OCR | 已安装的 Tesseract 语言名 | 托管 `eng+chi_sim`，可选 `eng+chi_tra` | 未注册 | 否 |

独立 Windows/Linux 宿主需要已安装工具或显式提供 `PdfOcrProvisioner`；
仅设置 `auto` 不会准备 TensorAgent 的私有 OCR 运行时。
平台、媒体模型与验证限制见[知识库指南](knowledge-system_zh-cn.md)。

## 功能覆盖

功能目录位于
[`TensorSharp.TestMatrix/Matrix/FeatureCatalog.cs`](../TensorSharp.TestMatrix/Matrix/FeatureCatalog.cs)。
当前功能集合如下：

| 功能 | 驱动方式 | 能力门控 |
|---|---|---|
| `pp512` | `--benchmark --bench-prefill 512 --bench-decode 0` | 所有模型 |
| `pp2048` | `--benchmark --bench-prefill 2048 --bench-decode 0` | 所有模型 |
| `tg128` | `--benchmark --bench-prefill 32 --bench-decode 128` | 所有模型 |
| `short_text` | `--input prompts/short_text.txt --max-tokens 64` | 所有模型 |
| `long_text` | `--input prompts/long_text.txt --max-tokens 64` | 所有模型 |
| `uploaded_text` | `--input prompts/upload_text.txt --max-tokens 64` | 所有模型 |
| `multi_turn` | `--multi-turn-jsonl multi_turn/three_turn.jsonl` | 所有模型 |
| `tools` | `--tools tools/weather_tools.json` | 矩阵能力标记为支持工具调用的模型 |
| `thinking` | `--think` | 矩阵能力标记为支持思维链的模型 |
| `image` | `--image media/apple.png --mmproj ...` | 图像模型且有 mmproj |
| `audio` | `--audio media/sample.mp3 --mmproj ...` | 音频模型且有 mmproj |
| `video` | `--video media/sample.mp4 --mmproj ...` | 视频模型且有 mmproj |

默认语义检查刻意保持较弱，用于捕获灾难性回归。相关功能会检查 `blue`、
`paged`、`08:01:12`、`alex` + `teal`、`get_current_weather` + `tokyo`、
`10:38` 与 `apple`。音频与视频没有默认期望子串，因为样例媒体由运行环境提供。

## 过滤规则

运行前会过滤组合爆炸：

1. 后端可用性：CUDA 与 Vulkan 后端在 macOS 跳过（macOS 上的 GPU 后端是 Metal）；MLX 需要 Apple Silicon；GGML Metal 需要 macOS。
2. 模型能力：当发现或配置的模型不支持图像 / 音频 / 视频 / 工具 / 思维链时，对应功能跳过。
3. 投影器可用性：多模态功能需要 mmproj 路径。
4. 环境变量适用性：每个 `EnvVarSpec.AppliesTo` 决定该变量是否对当前 `(model, backend, feature)` cell 有意义。

## 更新矩阵

新增高影响环境变量时：

1. 在 [`TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs`](../TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs) 注册一个 `EnvVarSpec`。
2. 如果它应进入默认 sweep，把它加入
   [`Defaults/matrix-config.json`](../TensorSharp.TestMatrix/Defaults/matrix-config.json)
   的 `default_env_vars`。
3. 更新本文和英文版本中的对应行。
4. 如果该变量改变功能适用性，同步更新
   [`FeatureCatalog.cs`](../TensorSharp.TestMatrix/Matrix/FeatureCatalog.cs)
   或模型发现的能力推断。
