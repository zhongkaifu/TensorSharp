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
| `TS_RETAINED_FUSED_CACHE_MAX` | 具有可保留 request-owned fused holder 的模型（Gemma 4；Qwen 3.5/3.6/3.8；Qwen 3.8 Flash Next，其字节预算为 `TS_Q4E_RETAINED_CACHE_MB`；未加载 DSpark 草稿器时的 DeepSeek V4.1；原生执行器上的 GLM 5.x，每个会话交出一个原生 slot） | 为下一轮的精确前缀续接保留多少个已结束会话的终态（holder、原生 slot）；Qwen holder 同时包含 attention K/V 与匹配的 GatedDeltaNet 递归状态。未设置时跟随 `TS_SCHED_MAX_RUNNING_SEQS`（至少 4），使每个可以并行运行的请求都保有自己的终态；字节数由显存而不是这个数量约束。`0` 关闭保留 | 运行中请求上限（16），至少 `4` | 不适用 | 否 |
| `TS_PREFIX_CHECKPOINTS_MAX` | GGML 后端上的 Gemma 4；`ggml_cuda` / `ggml_metal` / `mlx` 上的 Qwen 3.5/3.6/3.8，即它运行按请求 holder 的后端（TP 下不支持）；GGML token-span 路径上的 Qwen 3.8 Flash Next（包括 `--layer-split N` 按层切分时）。需要开启 `TS_PER_SEQ_FUSED` | 在每个会话共享的提示前缀（系统提示、工具、技能）结束处对模型完整状态做检查点，新会话从其副本继续，只需重新预填自己的消息。取值是同时保留多少个不同共享前缀的检查点（每个占用一份前缀的 K/V，Qwen 还包含递归状态）；`0` 关闭检查点。它是 Radix 前缀缓存保留的公共检查点预算；一个提示在每个边界（系统指令、完整共享前缀）各发布一个检查点，因此 4 可覆盖同时预热两种思考模式的主机 | `4` | 不适用 | 否 |
| `TS_KV_INITIAL_TOKENS` | 通过 `ModelBase.ResolveInitialCacheAllocationLength` 确定缓存大小的模型家族（Qwen 3.5/3.6、Gemma 4、GPT-OSS 等 ModelBase 家族；不含自行确定大小的 DeepSeek V4 / GLM 5.x） | 缓存创建时（加载时的主缓存、每个 per-request holder）在任何请求声明预算之前分配的 K/V token 数；`0` 沿用引擎策略（显式 `MAX_CONTEXT` 时为整个窗口，否则为后端默认值）。缓存仍按需增长。内存受限设备把它设小，因为每个保留的 holder 都按此大小付费，主机副本与设备镜像各一份 | `0` | 不适用 | 否 |
| `TS_KV_GENERATION_RESERVE_MAX` | 全部 | 请求预先保留的 K/V 中生成部分的上限（prompt + max_new_tokens）；回复上限不小于窗口时否则每个请求都会保留整个窗口。超过上限后缓存按需增长。`0` = 不限制 | `0` | 不适用 | 否 |
| `TS_KV_HOLDER_POOL_MAX` | 具有 per-request fused holder 的模型（Qwen 3.5/3.6/3.8、Gemma 4、GPT-OSS） | 已释放的 holder 最多可停放多少个以待复用而不是释放；每个停放的 holder 都占用其完整 K/V 分配 | `64` | 不适用 | 否 |
| `TS_SCHED_DISABLE_BATCHED` | 全部 | `1` 让所有模型都走按序列 KV-swap 路径，投机解码也一样（`--no-continuous-batching`） | 关闭 | `0`, `1` | 是 |
| `TS_SCHED_PREFIX_CACHE` | 全部自回归模型 | `0` 关闭准入时的全部提示复用（Radix 前缀缓存：页面、保留的终态与公共检查点在同一个索引里）。`--no-prefix-cache`（CLI 与服务端）会设置它，并同时跳过共享提示的启动预热，以及服务端的检查点文件 | 启用（`1`） | 未注册 | 否 |
| `TS_SCHED_MAX_BATCHED_TOKENS` / `TS_SCHED_MAX_RUNNING_SEQS` / `TS_SCHED_PREFILL_CHUNK` / `TS_SCHED_SOLO_PREFILL_CHUNK` / `TS_SCHED_NUM_BLOCKS` / `TS_SCHED_BLOCK_SIZE` / `TS_SCHED_DECODE_QUANTUM` | 全部 | 调度器预算：每步 token 数、同时运行的序列数、有 decode 在跑时的 prefill 分块（服务端 `--prefill-chunk-size`）、solo prefill 分块、块池块数、每块 token 数（只靠页面做跨请求复用的家族为 16，其余为 256）、decode 时间片。有自己单请求步长的模型——启用流水化 prefill 的 Muse Spark：8192——在未设置 `TS_SCHED_SOLO_PREFILL_CHUNK` 时自行决定步长；设置了该变量则按文档生效并关闭模型自己的步长，只设置 `TS_SCHED_MAX_BATCHED_TOKENS` 不会关闭它。见[配置表](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md#配置) | `4096` / `16` / `256` / `8192` / `256` / `16` 或 `256` / `256` | 未注册 | 否 |
| `TS_SCHED_STOP_REPETITION` | 全部 | `0` 时陷入循环的生成会一直跑到 token 上限，而不是以 `repetition` 结束原因停止 | 启用（`1`） | 未注册 | 否 |

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
| `TS_N_CPU_MOE` | MoE 模型 | 前 N 层的路由专家留在系统内存：decode 时在主机上做乘法，prefill 时流式送到加速器上跑一整张图。在 `ggml_metal` 上未设置时，若专家放不下，Qwen 3.8 Flash Next（`qwen4exp`）会按 Metal 工作集与内存自行规划（48 GiB 的 Mac 上 48 层中有 15 层的专家留在 GPU 上） | 关闭（`0`） | `0`, `16`, `all` | 否（已为 GGML GPU 后端与 MoE 家族注册，但不在默认配置的列表中） |
| `TS_CPU_MOE` | MoE 模型 | 卸载所有层的路由专家（等价于 `TS_N_CPU_MOE=all`） | 关闭 | 未注册 | 否 |
| `TS_CPU_MOE_THREADS` | MoE 模型 | 主机端专家 matmul 的工作线程数。可用 CPU（硬件线程数按亲和性掩码与 cgroup CPU 配额收敛后）超过 8 个时，默认取其一半，上限 64：decode 侧的 matmul 只有一个 token 宽，超过几十个线程后每多一个线程只是多一个屏障参与者（在双路 Xeon 上实测 192 线程比 32 线程慢 7 倍）。较小的主机几乎用满全部 CPU。DeepSeek V4 / V4.1 则默认使用全部可用 CPU（见下文） | 可用数 ≤2 时为 1，≤8 时为可用数−1，否则 min(可用数/2, 64) | - | 否 |
| `TS_HOST_MOE_DEVICE_MIN_BATCH` | 启用卸载的 MoE 模型 | 达到或超过该 batch 大小时，被卸载的层改为在加速器上计算、专家权重流式送入，而不是在主机上算。`0` 恢复纯主机卸载 | `128` | 未注册 | 否 |
| `TS_HOST_MOE_PIN` | 启用卸载的 MoE 模型 | 把被卸载的专家区间页锁定（`cudaHostRegister`），使流式 prefill 走 DMA 而不是经驱动中转（PCIe 5.0 上 9.3 → 55.6 GB/s）。`0` 对所有架构关闭。DeepSeek V4 / V4.1 的默认值是例外：它们的加载器在任何批大小下都在 CPU 后端上计算被卸载的专家，没有任何流式传输，因此只有 `1` 时才锁页（七卡 A40 通道上锁定 48.2 GiB 让加载多花 20.4 s，并使这些页面无法被回收） | 启用；DeepSeek V4 / V4.1 为关闭 | 未注册 | 否 |
| `TS_HOST_MOE_PIN_MAX_MB` | 启用卸载的 MoE 模型 | 页锁定专家区间的预算 | cgroup / 主机内存上限的 60% | - | 否 |
| `TS_HOST_MOE_DECODE` | 启用卸载的 MoE 模型 | `0` 让单 token 的被卸载层改走 ggml CPU 计算图，而不是 TensorSharp 的 decode 内核（仍用 ggml 自己的 CPU 点积，但线程组每层只唤醒一次；在 M5 Pro 上跑 Qwen3.8-Flash-Next 时每层 0.6 → 0.32 ms）。用于 A/B 对比 | 启用 | 未注册 | 否 |
| `TS_HOST_MOE_TIMING` | 启用卸载的 MoE 模型 | 诊断：`1` 主机侧每次调用的准备与 matmul 时间，以及流式 prefill 的字节数、传输速率与 GPU 时间；`2` 每个权重的拷贝速率（会同步，改变所测的东西）；`3` 每个 decode pass 的加速器分段、主机专家与上传时间；`4` 单 token 内核的各阶段与工作线程的唤醒 | 关闭 | 未注册 | 否 |
| `TS_HOST_MOE_EXPERT_FILTER` | 启用卸载的 MoE 模型 | 只流式传输该 batch 实际路由到的专家，并合并成连续区间 | 启用 | 未注册 | 否 |
| `MAX_CONTEXT` | 长文本 / 上传文本 | 硬上下文上限。设置了就是硬性要求：缓存放得下就照办，放不下就带着数字拒绝。不设置时，GGUF 宣称的长度只是上限，加载器会按设备真正装得下的量来定——GLM-5.2 宣称 1M token，那是约 93 GiB 的 KV | 模型默认值（是上限而非承诺） | `4096`, `8192`, `16384` | 是 |

## Prefill / Decode 调优

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_PREFILL_CHUNK` | 在 GPT OSS、Qwen 3.5 / 3.6 family 长上下文功能上 sweep；运行时 Gemma 4、Nemotron-H、Mistral 3 也会读取 | 分块 prefill 大小 | 架构默认值 | `256`, `512`, `1024` | 是 |
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
| `TS_MLX_BATCHED_MOE_DECODE` | MLX 上的 Qwen 3.5 / 3.6 MoE | 在堆叠的专家权重上每种 gate/up/down 一次批处理 dispatch，而不是按 expert dispatch；`0` 走按专家顺序执行并省去堆叠副本（内存紧张的机器） | 启用 | `0`, `1` | 是 |
| `TS_MLX_HALF_MATMUL_MIN_ROWS` | MLX affine 量化 matmul | scales 为 F16 的 matmul 从多少行起使用 F16 激活；`0` 使所有 matmul 保持 F32 | `1`（全部） | `0`, `1`, `32` | 否 |
| `TS_MLX_Q6K_MATVEC_MAX_ROWS` | MLX 上的精确 Q6_K matmul | 行数不超过该值时，Q6_K 以矩阵-向量乘运行（移植自 ggml-metal 的 `kernel_mul_mv_q6_K_f32`）；更多行时把 Q6_K 反量化为 F16（每片不超过 256 MB）并用 MLX 的 GEMM 相乘 | `4` | 整数 >= 0 | 否 |
| `TS_MLX_IQ4XS_MATVEC_MAX_ROWS` | MLX 上的 IQ4_XS matmul | 行数不超过该值时，IQ4_XS 以矩阵-向量乘运行（移植自 ggml-metal 的 `kernel_mul_mv_iq4_xs_f32`）；更多行时把 IQ4_XS 反量化为 F16（每片不超过 256 MB）并用 MLX 的 GEMM 相乘 | `4` | 整数 >= 0 | 否 |

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
| `TS_Q35_VERIFY_SNAPSHOTS` | Qwen 3.5 / 3.8 投机验证 | `0` 回退为先保存验证前的递归状态副本、再对已接受前缀重新前向，而不是每行保留一份快照 | 开 | 未注册 | 否 |
| `TS_QAB_BOUND` | Qwen 3.5 / 3.8 批量解码（CUDA） | `0` 使 N>=2 解码退回到 arena 自有的 KV/状态平面（加入时拷贝、退出时回写、窗口变化时重新填充）。默认让每个槽位直接绑定其 holder 自身常驻的 K/V 与递归状态副本，即单序列解码使用的同一缓冲区 | 开 | 未注册 | 否 |
| `TS_Q35_DEVICE_MASK` | Qwen 3.5 / 3.8 预填充验证（CUDA） | `0` 改为在主机上填充因果掩码并上传（窗口 x 分块个 half，21K 上下文下 4K 分块约 170 MB），而不是在图内生成 | 开 | 未注册 | 否 |
| `TS_Q35_VERIFY_PACK` | Qwen 3.5 / 3.8 验证（CUDA、Metal） | `0` 让每个持久验证图使用未打包缓冲区，而不是按生命周期打包的 gallocr 缓冲区 | 开 | 未注册 | 否 |
| `TS_Q35_VERIFY_PERSIST_KV_STRIDE` | Qwen 3.5 / 3.8 投机验证 | 持久验证图的注意力窗口步长（行）；步长越大，重建/重新捕获越少 | `2048` | 未注册 | 否 |
| `TS_Q35_GPU_STATE_COMMIT` | Qwen 3.5 / 3.8 投机验证（CUDA、Metal） | `0` 通过主机提交已接受的递归状态快照，而不是设备内拷贝 | 开 | 未注册 | 否 |
| `TS_Q35_VERIFY_LOG` | Qwen 3.5 / 3.8 | `1` 打印每次融合验证/预填充调用（起点、行数、logits、捕获、状态链接、耗时） | 关 | 未注册 | 否 |
| `TS_QWEN35_DEVICE_HOLDER_COPY` | Qwen 3.5 / 3.8 前缀缓存（CUDA） | `0` 使检查点、克隆和 KV 扩容经主机内存拷贝 K/V，而不是设备到设备 | 开 | 未注册 | 否 |
| `TS_QWEN35_DEVICE_RETAIN_RESERVE_MB` | Qwen 3.5 / 3.8 前缀缓存（CUDA） | 空闲保留 holder 的设备 K/V 必须留出的空闲显存；低于该值时最久未用者被降级到主机 | `max(4 GiB, 显存/8)` | 未注册 | 否 |
| `TS_QWEN35_IDLE_KV_SYNC` | Qwen 3.5 / 3.8 前缀缓存（CUDA） | `0` 停止在显存吃紧之前于后台把空闲保留 holder 的设备 K/V 下载到主机，之后的降级会重新在请求关键路径上同步（每 GB 约 150 ms，而非约 5 ms） | 开 | 未注册 | 否 |
| `TS_PREFIX_RELEASE_DEBUG` | Qwen 3.5 / 3.8 前缀缓存 | `1` 记录保留/规划/整理决策、降级（`[prefix-demote]`）及其主机同步（`[prefix-sync]`） | 关 | 未注册 | 否 |
| `TS_GGML_HOST_POOL_MB` | GGML CUDA 主机内存分配 | 保留任意大小已释放主机块以供复用的字节预算（新映射首次访问会产生缺页）；`0` 仅保留按数量限制的小块池 | `min(8 GiB, 内存/16)` | 未注册 | 否 |
| `TS_QWEN35_BATCHED_SPEC` | Qwen 3.5 / 3.8 批量推测解码（CUDA，DFlash 草稿模型） | `0` 让并发解码请求在批量解码中每步只解一个 token，而不是在一次前向中验证每个请求的草稿窗口 | 开 | 未注册 | 否 |
| `TS_QWEN35_BATCHED_SPEC_WINDOW` | Qwen 3.5 / 3.8 批量推测解码 | 每个请求最宽的验证窗口（锚点加草稿），2..8；当每个请求经置信度筛选后的草稿都不超过四行时，该步只验证四行 | `8` | 未注册 | 否 |
| `TS_QWEN35_SOLO_SPEC_DRAFT` | Qwen 3.5 / 3.8 推测解码（单个请求） | 单个请求每步验证最多使用的草稿数 | `3` | 未注册 | 否 |
| `TS_QWEN35_BATCHED_DRAFT` | Qwen 3.5 / 3.8 批量推测解码 | `0` 为每个请求单独运行 DFlash 草稿图，而不是把所有请求的草稿块放进一个图 | 开 | 未注册 | 否 |
| `TS_BATCHED_SPEC_LOG` | 批量推测解码 | `1` 打印每个批量步骤保留的 token 数以及草稿 / 验证 / 提交 / 草稿模型提交的毫秒数（`[batched-spec]`） | 关 | 未注册 | 否 |
| `TS_SCHED_STEP_LOG` | 连续批处理执行器 | `1` 每步打印一行：距上一步的空闲间隔、服务路径耗时毫秒数，以及每个序列的预填充/解码 token 数（`[step]`） | 关 | 未注册 | 否 |
| `TS_GPU_LOCK_YIELD_MS` | 连续批处理执行器 | 当请求线程已在等待模型 GPU 锁（提示的媒体准备）时，执行步骤在自己获取该锁之前最多等待的毫秒数；`0` 表示步骤优先获取 | `100` | 未注册 | 否 |
| `TS_QWEN35_BACKGROUND_CAPTURE` | Qwen 3.5 / 3.8 前缀缓存（CUDA） | `0` 让主机路径的前缀检查点同步下载请求的 K/V 行，而不是在后台流中直接写入检查点 | 开 | 未注册 | 否 |
| `TS_QWEN35_DEVICE_STATE_HANDOFF` | Qwen 3.5 / 3.8 每请求缓存（CUDA） | `0` 让验证器的循环状态在每次易手时经主机内存搬运（每次切换持有者下载 151 MB、每次恢复再上传、位置 0 上传主机零值），而不是在设备上直接移入/移出各请求自己的设备副本 | 开 | 未注册 | 否 |
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
| `TS_DSV4_ASYNC_SEND` | V4.1 `--backend cuda` 分布式层切分，发送节点 | `1` 在下一个 ubatch 的首个 GPU 的层排队后再发送本 prefill ubatch 的边界数据，使节点内各 GPU 重叠处理相邻 ubatch；`2` 在下一个 ubatch 自己发送时才发送；未设置或 `0` 同步发送（S9）。一次 forward 的最后一个 ubatch 总在 forward 返回前发出。只影响时序：各模式下 logits 逐位一致。可选：从未在每个 stage 各用自己的 GPU 时运行过，因此它的收益以及跨两个 CUDA 上下文的事件等待都未测量 | `0` | 否 |
| `TS_DSV4_VRAM_RESERVE_MB` / `TS_DSV4_GRAPH_CACHE` / `TS_DSV4_LOAD_THREADS` / `TS_DSV4_LOAD_CHUNK_MB` / `TS_DSV4_MOE_MMAP` | V4 与 V4.1 | 放置余量、计算图缓存深度、权重加载并行度，以及驻留主机的专家是否直接在 GGUF 映射上就地相乘 | 见各卡片 | 否 |
| `TS_DSV41_RETAINED_CACHE_MB` | V4.1，原生执行器 | 为已结束会话的下一轮保留原生槽位的预算（保留始终开启，只在加载了 DSpark 草稿器时不生效）；不是正数的值保持默认 | `2048` | 否 |
| `TS_DSV41_TP_HOST_TOKENS` | V4.1 CUDA routed-MoE TP，仅主机驱动节点（`TS_DSV4_FUSED=0`） | 默认的 `--tp` 路径在融合 CUDA 后端上把每次 rank 交换都经 pinned 内存放在各 GPU 自己的队列上完成，不读取此变量。主机驱动节点上：`0`–`4096` 的整数：不超过此 token 数的批次使用精确 pinned 主机激活拼接，更大批次使用可用的私有 F32 NCCL。`0` 强制使用可用的设备拼接。自动阈值仅在 NCCL 选择 `NCCL_P2P_DISABLE=1` 时取 `16`，依据六张 A40 的配对实测；其他配置取 `0`。`TS_GGML_TP_F32_NCCL=0` 仍使全部批次回退到主机。 | 自动 | 否 |
| `TS_DSV41_ENGRAM_DEVICE` | V4.1（ggml 后端与 `--backend cuda`） | `1` 要求 Engram 表驻留 GPU，放不下就失败；`0` 强制主机映射，需要与 CPU oracle 逐位一致时也用它。不设置则自动且保守：只要不会因此逼出路由专家的 CPU 卸载，就放在 GPU 上 | 自动（放得下就驻留 GPU） | 否 |
| `TS_DSV41_ENGRAM_WARM` | V4.1，仅主机映射 | 读入 Engram 表页，使查表变成一次内存读取而不是一次存储往返。未设置时在模型开始服务后于**后台**预热（默认）；`1` 与此前一致在加载期间同步预热；`0` 从不预热。八卡 A40、Q4_K_M 上 prefill 从 200–252 提升到 452–492 tok/s，decode 从 23–26 提升到 31–33 tok/s；在默认的 GPU 驻留路径上无意义。代价是表本身的主机页缓存（Q2_K 60 GiB，Q4_K_M 103 GiB）以及读表的时间：七卡 A40 通道上同步形式用旧的逐页遍历读 103 GiB 花了 311.3 s；`pread` 预热（`TS_DSV4_WARM_PREAD`）在该存储上实测 2.24–2.54 GiB/s，即约 41–46 s 的读取 | 后台预热 | 否 |
| `TS_DSV4_GRAPH_CACHE_HEADROOM_MB` | V4 与 V4.1 | 图缓存必须为下一张图留出的设备内存，叠加在它持有的最大条目之上。会逐个释放最久未使用的条目直到满足。`0` 回到纯条目数上限，而四个并发的 10.8k token prefill 曾因此耗尽显存 | `1024` | 否 |
| `TS_DSV41_ENGRAM_THREADS` | V4.1，仅主机映射 | 常驻查表工作线程数，`1`–`32`。单个 token 在每张表上要取 24 行互不相关的数据，串行读意味着串行缺页。在 `--backend cuda` 上：按 (token, 列) 同时读取的行数（含发起读取的线程），在读取器自己的线程上读取（从不使用 .NET 线程池），预读共用这些线程，因此读取器在途的行数不会更多；`1` 精确恢复 S9 的读取方式（token 之间并行、列串行），并在未显式设置时关闭访问建议 | min(16, 硬件线程数) | 否 |
| `TS_DSV41_ENGRAM_RANDOM` | V4.1，Linux 上的主机映射 | 对映射的 Engram 区间给出随机访问建议。`0` 关闭，`1` 强制 | 自动 | 否 |
| `TS_DSV41_ENGRAM_PREFETCH` | V4.1 `--backend cuda`，主机映射 | 每次前向开始时，由读取器的线程按层的顺序预先读取本节点各表在这次前向中要取的所有行，同时主机继续下发各层；它只对历史的副本做哈希，只读不写，且取行总是优先。reset、失败的前向与页缓存逐出会停止它。`1` 开启，未设置或 `0` 关闭。可选：它在冷的网络存储上的收益尚未在完整模型上测量 | 关 | 否 |
| `TS_DSV41_ENGRAM_OWNER` | V4.1 `--backend cuda` 跨节点层切分，主机映射 | 每张 Engram 表由哪个节点持有（映射、预热、取行）。未设置或 `local`：由运行该表所在层的节点持有（S9）。`balance`：按表序号依次分给当前持有表字节最少的节点，相同时留给读它的节点（1017 在 2x2 上：节点 0 持有第 14 层的切片，节点 1 持有第 1 层的，因此每个节点的表都能在其 cgroup 中保持预热）。由别的节点持有的表在每次前向开始时以行号加原始量化行的形式传过来；两种方式结果逐位相同。属于配置摘要的一部分，按含义比较（未设置等于 `local`）。在双节点实测之前 `balance` 保持为可选 | `local` | 否 |
| `TS_DSV41_ENGRAM_WARM_THREADS` | V4.1 `--backend cuda`，主机映射 | Engram 预热的线程数（每个线程用自己的句柄 `pread`），`1`–`64`。只预热本节点持有的表，只读 `mincore` 报告不在内存中的页，且只有在 cgroup（上限减去匿名与 shmem 内存再减 8 GiB）和 `MemAvailable` 都放得下时才预热 | `16` | 否 |
| `TS_DSV4_LOAD_CONTIGUOUS` | V4 与 V4.1 | `0` 让权重加载器退回到从共享游标分发分块任务，每个读取线程会以 `线程数 x 分块` 的步幅跨越文件，而不是读一段连续区间。保留它只是为了能对连续读取的默认行为做 A/B：在 MooseFS 挂载上实测慢 2.5 倍（八卡 A40 加载 Q4_K_M 发行版 363–382 s，对比 144–155 s） | 开 | 否 |
| `TS_DSV4_WARM_PREAD` | V4 与 V4.1 主机映射；V4.1 TP 路由权重加载 | 控制主机专家预取（`--n-cpu-moe`）、同步/后台 Engram 预热，以及 V4.1 TP 在上传 GPU 前对当前层路由 gate/up/down 源权重的预读。未设置或 `1` 使用有界并行 `pread`，线程数由 `TS_DSV4_LOAD_THREADS` 决定；跳过 `mincore` 报告已驻留的区间，读取错误包含分片与偏移。TP 在打开当前层的源映射后仅预读该层，不预读整个检查点，也不通过此路径预热 Engram 表。`0` 关闭 TP 源预读，并恢复既有主机专家/Engram 逐页触碰遍历（预取为 256 MiB 段，Engram 为 8 MiB 块）。历史七卡 A40 的 8 GiB 区间测试中，`pread` 为 2.24–2.54 GiB/s，逐页遍历为 0.62–0.74 GiB/s；这不是 TP 模型加载提速的测量，也不保证冷存储性能 | 开 | 否 |
| `TS_DSV4_LOAD_DROP_CACHE` | V4 与 V4.1 的普通逐层上传；直连引擎（`--backend cuda`）使用同一规则，其上传量为本节点的权重，主机映射量为本节点持有的 Engram 表（仅 Linux；位于 tmpfs 的分片不处理） | 每个权重分块上传到设备后是否释放它的页缓存。V4.1 的专用 TP 路由专家上传路径不使用此设置，已上传的源数据页仍保留在可回收的文件缓存中。未设置时自动判断：当上传字节数加上主机映射的权重（专家、Engram 表）再加 8 GiB 超过主机额度（cgroup 上限）时释放，否则保留，额度未知时也保留，因此纯 GPU 驻留的检查点重新加载时仍是热的；加载时会打印这一判断及三个数值。`1` 总是释放，`0` 从不释放（此前的默认）。七卡 A40 通道（上传 263.0 GiB + 映射 151.2 GiB，对比 326.9 GiB）会释放。在 MooseFS 挂载上释放每个已驻留的 64 MiB 分块耗时 5.9–7.3 ms，也无法让上传本身变快；预期收益在其后的预取与 Engram 预热上 | 自动 | 否 |
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
| `TS_GGML_TP_FUSED_MATMUL` | 本地 TP，GGML 后端 | `1` 表示由单个线程提交两个 rank 的线性层；每次调用都要为每个 rank 分配设备缓冲，在 Qwen 3.5 35B 上实测慢 2.3× | 关闭（通用按 rank 路径） | 未注册 | 否 |
| `TS_GGML_TP_DEVICE_AR_THRESHOLD` | 本地 TP，GGML 后端 | 超过该元素数量时 AllReduce 走设备集合通信，否则在主机内存中归约 | `262144` | 未注册 | 否 |
| `TS_GGML_F32_RESIDENT` | GGML 后端 | `0` 表示每次调用重新绑定 F32 线性层权重，而不是常驻设备（诊断用） | 开启（常驻设备） | 未注册 | 否 |
| `TS_GEMMA4_TP_FUSED_MOE` | GGML 上 TP 下的 Gemma 4 MoE | `0` 表示从融合的整模 MoE 主干（专家内部 Megatron 切分）回退到逐算子的整专家路径 | 开启（融合主干） | 未注册 | 否 |
| `TS_GLM_TP_SHARD` | GGML 上 TP 下的 GLM 5.x | 切分哪一半：`1` 注意力头，`2` 路由专家，`3` 两者都切。路由专家是在每个专家内部按行切分，而不是按专家 id 分配，因为 `ggml_mul_mat_id` 要求同一 token 选中的专家 id 互不相同 | `3`（两者） | `1`, `2`, `3` | 否 |
| `TS_GLM_TP_OVERSUBSCRIBE` | GGML 上 TP 下的 GLM 5.x | `1` 允许多个 rank 共享一张 GPU，用于在单卡机器上验证切分的正确性 | `0`（一 rank 一卡） | `0`, `1` | 否 |
| `TS_Q4E_LAYER_SPLIT` | `--layer-split N` 下按层切分的 Qwen 3.8 Flash Next（`qwen4exp`） | 直接指定每张 GPU 分到的层数（逗号分隔，例如 `20,28`），取代自动的显存均衡；给出无法满足的值时会直接抛错，而不是静默忽略。这个架构上的 `--layer-split N` 是按层切分而非张量并行——`--tp N` 是独立的 FFN 通道切分模式，不使用此层分配覆盖值 | 自动（按各设备空闲显存装箱） | 未注册 | 否 |
| `TS_Q4E_RETAINED_CACHE_MB` | Qwen 3.8 Flash Next（`qwen4exp`）保留复用 | 保留会话与共享前缀检查点共用的预算（MiB），受实测内存余量限制。未设置时预算为实测余量的一半（与 Qwen 3.5 对空闲 holder 的规则相同），只有无法测得余量时才用 4096；原先固定的 4096 默认值在 4x A40 张量切分上只能容纳四个并发 1.3 GB 会话中的三个。Radix 前缀缓存负责淘汰，放不下的 holder 会被拒绝（只报告一次）。`0` 或无法解析的值拒绝所有保留 | 实测余量的一半（无法测得时为 `4096`） | 未注册 | 否 |
| `GGML_CUDA_ALLREDUCE` | 本地 TP，`ggml_cuda` | `nccl` / `internal` / `none` —— 直接透传给 ggml 的集合通信选择；显式设置同时会跳过启动前探测 | 自动（构建时能找到 NCCL 且通过探测就用 NCCL） | 未注册 | 否 |
| `TS_GGML_TP_CUDA_GRAPHS` | 本地 TP，`ggml_cuda` | `0` 关闭多 GPU 运行下的 CUDA graph 捕获。TP 下默认**开启**捕获：一个张量并行 token 是几十次按 rank 的小提交，重放的代价远低于重新下发（4×A40：Qwen3.5-9B tp4 88 → 128.5 tok/s，Qwen3.5-35B-A3B tp2 71.3 → 104.1）。历史上曾因捕获污染的隐患而禁用，那个隐患已不再成立——ggml 用 `cudaStreamCaptureModeRelaxed` 捕获。这个 opt-out 会在第一次后端调用之前翻译成原生的 `GGML_CUDA_DISABLE_GRAPHS`，因为 ggml 会在首次使用时锁定该值 | 开启捕获 | 未注册 | 否 |
| `TS_GGML_TP_AR_PROBE` | 本地 TP，`ggml_cuda` | `0` 跳过两项启动前探测；`force` 忽略缓存的判定（`~/.cache/tensorsharp/tp-collective-probe`）重新探测。模型加载前，进程组会检查两件事：所宣称的设备对之间 peer copy 是否真的把数据送到，以及一次小型 NCCL AllReduce 能否端到端完成——一些云主机声称支持 P2P 但数据永远送不到，NCCL 的第一次集合通信随后会让每块 GPU 永远空转。peer 检查失败时会保留 NCCL 但拿掉它的 peer 传输（`NCCL_P2P_DISABLE=1`），这正是超过 2 张 GPU 时仍能保住设备集合通信的原因 | 探测开启，判定按 驱动/NCCL/GPU 组合缓存 | 未注册 | 否 |
| `TS_GGML_TP_AR_PROBE_MS` | 本地 TP，`ggml_cuda` | 每项探测（先 peer copy，后 AllReduce）的完成期限，超时即判定该传输不可用；集合通信随后在 2 张 GPU 时回退到钉页主机内存的 `internal` 管线，更多卡时回退到主机归约。`0` 关闭探测 | `10000` 毫秒 | 未注册 | 否 |
| `TS_GGML_TP_F32_NCCL` | 对精度敏感的本地 TP，`ggml_cuda` | `0` 禁用 TensorSharp 自有的 F32 NCCL 路径，供诊断使用。对精度敏感的通用计划回退到保持 F32 的分块归约或主机归约；DeepSeek V4.1 回退到主机行收集。默认使用不压缩的 NCCL 操作，其中 DeepSeek 不等宽激活分片使用逐位保真的 AllGather，并遵守显式的 `GGML_CUDA_ALLREDUCE=internal/none`。不改变普通 TP 计划。 | NCCL 可用时开启 | 未注册 | 否 |
| `GGML_CUDA_AR_BF16_THRESHOLD` | 本地 TP，`ggml_cuda` | ggml 在多大载荷以上把 F32 集合通信转成 BF16；TensorSharp 把 ggml 的默认值提高到 1 MB，使 decode 规模的归约保持精确 | `1 MB`（由 `TSGgml_TensorParallelInit` 设置） | 未注册 | 否 |
| `TS_QWEN35_LAYER_TRACE` | Qwen 3.5/3.6 | `1` 打印首次前向的逐层残差流摘要，单卡与 TP 两条路径都会输出（诊断用） | 关闭 | 未注册 | 否 |

## 矩阵外的 Redis 共享状态变量

这个变量配置可选的 Redis 状态，它未注册在 `EnvVarMatrix.All` 中，`--redis-url` 会设置它。Redis 不保存任何 KV 状态。

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_RESPONSES_STORE_REDIS_URL` | 仅服务端 | Responses API 存储的 Redis 连接串；设置后取代内存存储 | 未设置（关闭） | 未注册 | 否 |

## 矩阵外的通用运行时开关

这些变量是真实运行时开关，但目前未注册到 `EnvVarMatrix.All`，也不在默认
TestMatrix 配置中 sweep。

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TENSORSHARP_UPLOAD_DIR` | `TensorSharp.Server` | 上传媒体与抽取出的视频帧所在目录；不可变部署时设为应用目录之外的绝对路径 | 服务端二进制旁的 `uploads` | 未注册 | 否 |
| `TS_UPLOAD_MAX_MB` / `TS_UPLOAD_QUOTA_MB` / `TS_UPLOAD_TTL_HOURS` | `TensorSharp.Server`（也可用 `--upload-max-mb`、`--upload-quota-mb`、`--upload-ttl-hours`） | 客户端上传的单文件上限（超过返回 HTTP 413）、上传目录的总预算（用尽时返回 HTTP 507），以及该目录中文件被删除前的存留时长。Jev 内联图片同样受这些限制约束 | `500` / 关 / 关 | 未注册 | 否 |
| `TS_LORAS_DIR` | `TensorSharp.Server`（也可用 `--loras-dir`，命令行优先） | 文本模型 LoRA 适配器目录（llama.cpp 的 `.gguf` 适配器，或张量名为 `<tensor>.lora_a` / `.lora_b` 的 `.safetensors`）。顶层的每个适配器都在启动时、模型加载之后加载；`POST /v1/loras` 可以再加载，但只能从该目录内、以相对路径给出；最多同时保留 32 个。不是已存在目录的路径会导致启动失败。空值视为未设置；嵌入服务忽略它 | 未设置（没有适配器；`POST /v1/loras` 返回 403） | 未注册 | 否 |
| `TENSORSHARP_PREFIX_CACHE_DIR` | `TensorSharp.Server` | 共享前缀检查点文件的根目录，使重启后不必再为共享提示做 prefill；每个模型有自己的 `<模型文件名>-<哈希>` 子目录，最多两个文件。`--no-prefix-cache` 关闭持久化。TensorAgent 把自己的检查点放在应用缓存目录下；CLI 只在内存中保留检查点 | 服务端二进制旁的 `prefix-cache` | 未注册 | 否 |
| `TS_NO_MULTI_AGENT` | `TensorSharp.Server` 聊天端点 | 取 `0` 以外的任何值都会关闭子代理委派，等同 `--no-multi-agent` | 未设置（委派开启） | 未注册 | 否 |
| `TS_JEV_MAX_BODY_MB` / `TS_JEV_MAX_CANVAS` / `TS_JEV_MAX_PENDING` | `POST /v1/systemone`（Jev，DiffusionGemma） | 请求体上限（MiB，1-64）、每个问题分块的答案 canvas 宽度（token，8-4096，同时受检查点限制），以及同时可接纳的请求数（1-1024），超出后端点返回 HTTP 529。超出范围的值会报错而不是被截断 | `8` / `64` / `32` | 未注册 | 否 |
| `TS_NEMOTRON_AUDIO_MMPROJ` | 带 `--mmproj` 的 Nemotron-H | 改从该音频配套 GGUF（NVIDIA `sound_encoder.*` / `sound_projection.*` 张量）而不是 `--mmproj` 文件加载 Parakeet 音频塔，从而可以同时使用视觉 mmproj 与音频配套文件。除非该文件的张量校验通过，音频仍被拒绝（HTTP 400）；见 `docs/models/nemotron_zh-cn.md` §4.7 | 未设置（`--mmproj` 含这些张量时从中读取音频塔） | 未注册 | 否 |
| `TS_PDF_MAX_PAGES` | PDF 文档输入（CLI `--pdf`、服务端 `/api/upload`） | 文本提取与页面图像渲染读取的 PDF 页数上限 | `0`（全部页面） | 未注册 | 否 |
| `TS_GGUF_PREFAULT` / `TS_GGUF_PREFAULT_THREADS` / `TS_GGUF_PREFAULT_RESIDENT` | 通过 `GgufReader` 加载模型 | `PREFAULT=0` 跳过并行页缓存预热；线程数不超过处理器数量。所有分片所选张量总量共享进程内存预算一半的上限，跳过 Qwen 的稀疏 PLE 表。Linux 检查页驻留状态，避免复制已缓存的数据；`RESIDENT=0` 强制读取，供 A/B 测量。其他平台常规读取所选范围，iOS/tvOS 跳过预热。该预算不代表当前主机空闲内存。 | 开，`min(16, 核心数)`，开启驻留检查 | 未注册 | 否 |
| `TS_DUMP_LOGITS` | 所有模型、所有后端，CLI 与服务端 | 把**决定第一个生成 token 的 logits** 以原始 float32（小端）写入该路径，每个进程只写一次：即**整个提示词处理完之后的最后一行**，不论提示词走的是哪条路径——一次前向还是分成多块（前缀缓存检查点、递归块边界、分块上限）、逐序列 / fused / 批量 prefill、`--tp` / `--layer-split`、模型自带的引擎（q4e、GLM、DeepSeek V4 等），还是 CLI `--benchmark`（`Forward`，或带 `--bench-chunked` 时的 `ForwardRefill`）。它在引擎为请求采样第一个 token 的地方取得（以及 benchmark / `--test` 的提示词前向之后），因此**绝不会**是预热前向、decode 前向或预热请求：在预热中提交的请求（`--warmup-runs`、服务端与 TensorAgent 的前缀缓存预热、各 benchmark 的预热轮）在整个生命周期内都被排除——即使被取消的预热仍留在引擎里也一样——而与预热同时提交的真实请求不受影响。并发请求时，最先采样的那个请求会被导出，因此请用单请求的运行来对比。旁边的 `<路径>.json` 记录向量的 SHA-256、提示词长度与 token id 指纹、媒体片段数量与 LoRA 适配器（不对媒体**内容**做指纹）、前缀缓存复用量、每一次 prefill 前向所属的调度计划路径（只是标签而非实际执行的内核；CLI 的直接循环对整段提示词只记一步，因此不要据此断言前向次数），以及第一个 token。进程启动时（CLI 启动、每个推理引擎创建时）会删掉这两个文件，因此没跑到第一个 token 的运行不会留下旧文件。`eng/validation/compare-parallel-logits.py` 会拒绝比较：sidecar 指向不同提示词、媒体数量或适配器的两个向量；与旁边向量不符的 sidecar（过期 sidecar：哈希、长度或 argmax 不一致）；无法解析的 sidecar；加 `--require-sidecar` 时也拒绝没有 sidecar 的向量。（此前导出的是第一次非预热的 `ModelBase.Forward`：只要提示词用了不止一次前向，它就是提示词的**某一块**——若提示词根本不经过 `Forward`，则是第一步 decode——两种配置可能比较的是不同的计算。）这样就能用 logit 向量而不是生成文本来比较两个后端——贪心解码会把一次几乎打平的比分变成一句明显不同的话。DiffusionGemma 不写（没有自回归的第一个 token） | 未设置（不导出） | 未注册 | 否 |
| `TS_FUSED_QKNORM_ROPE` | 直连 `cuda` 后端上的 Qwen 3.5 / 3.6 纯文本 prefill | 融合 QK-Norm + NeoX-RoPE CUDA 内核；`0` 回退到分离的 norm + RoPE 算子（多模态 MRoPE 与其他后端始终走分离路径） | 启用 | 未注册 | 否 |
| `TS_CUDA_QMM_F16GEMM_MIN_ROWS` | 直连 `cuda` 后端 | 激活行数达到该阈值的量化矩阵乘会把权重一次性反量化为 F16 并走张量核心 cuBLAS GEMM（ggml 风格的 prefill 路线），替代分块量化内核 | `32` | 未注册 | 否 |
| `TS_CUDA_QMM_F16GEMM_MAX_MB` | 直连 `cuda` 后端 | F16 权重暂存区上限（MB）；超过上限的权重（如 LM head）继续使用量化内核 | `768` | 未注册 | 否 |
| `TS_CUDA_Q80_VEC_MIN_OUT` | 直连 `cuda` 后端 | Q8_0 dp4a 矩阵-向量乘的最小输出宽度（诊断开关） | `0` | 未注册 | 否 |
| `TS_CUDA_Q80_MMQ_MAX_ROWS` | 直连 `cuda` 后端 | 激活行数在 32 到该值之间的 Q8_0 矩阵乘直接在原始 Q8_0 块上执行 int8 张量核心 GEMM（mma.m16n8k32，ggml MMQ 风格）；超过该值后 F16 GEMM 路线更优（MMQ 的权重扫描次数随 ceil(rows/128) 增长） | `512` | 未注册 | 否 |
| `TS_CUDA_PREFILL_GRAPH_MAX` | 直连 `cuda` 后端 | 缓存的 prefill + decode CUDA graph 数量（LRU 淘汰；每个 graph 固定持有其捕获时使用的内存池块）。Qwen 3.5 / 3.6 纯文本 prefill 与 decode 会把逐算子层循环捕获为 graph 并重放（结果逐位一致；捕获失败时回退普通路径） | `4` | 未注册 | 否 |
| `TS_CUDA_PREFILL_GRAPH_LOG` | 直连 `cuda` 后端 | 打印 graph 捕获/重放/中止事件（`1`） | 关闭 | 未注册 | 否 |
| `TENSORSHARP_CUDA_POOL_LARGE_MB` | 直连 `cuda` 后端 | 全局大块（≥ 2 MB）显存缓存预算；让 prefill 级激活保持池化，避免每层重复 cuMemAlloc/cuMemFree | `1024` | 未注册 | 否 |
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
