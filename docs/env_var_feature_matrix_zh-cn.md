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
  为空，且 CLI 没有覆盖它，运行器会使用全部已注册的 `EnvVarMatrix.All` 项。

下表中的“运行时 baseline”表示变量未设置时的行为。“默认 sweep”表示当前默认
配置是否会扫描该变量，而不是所有已注册变量。

DiffusionGemma 当前不属于已注册的 TestMatrix 功能目录：还没有 diffusion prompt
类型，没有 diffusion 专属 env sweep，运行器也不会清理继承来的 `DIFFUSION_*`
变量。要把 diffusion 结果纳入标准矩阵，请先显式配置模型，并新增对应 feature
与 env-var 注册。

## 连续批处理 / 批处理前向

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_GPTOSS_BATCHED` | GPT OSS | 批处理分页前向 vs 按序列回退 | 启用 | `0`, `1` | 是 |
| `TS_QWEN35_BATCHED` | Qwen 3.5 / 3.6 family、`qwen3next` | 批处理分页前向 vs 按序列回退 | 启用 | `0`, `1` | 是 |
| `TS_QWEN35_BATCHED_GDN_NATIVE` | Qwen 3.5 / 3.6 family、`qwen3next` | 原生批处理 GatedDeltaNet 内核 | 关闭 | `0`, `1` | 否 |
| `TS_NEMOTRON_BATCHED` | Nemotron-H | 批处理分页前向 vs 按序列回退 | 启用 | `0`, `1` | 是 |
| `TS_GEMMA4_BATCHED` | Gemma 4 | 批处理分页前向 vs 按序列回退 | 启用 | `0`, `1` | 是 |
| `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE` | Nemotron-H | 原生批处理 Mamba2 step | 关闭 | `0`, `1` | 否 |
| `TS_BATCHED_N1_FAST_PATH` | 全部 | solo 序列走融合 N=1 快速路径 decode；`0` 强制这些步骤走完全批处理路径 | 启用 | `0`, `1` | 是 |
| `TS_PER_SEQ_FUSED` | fused 能力模型（Gemma 4、Qwen 3.5/3.6、DeepSeek V4、GLM 5.x） | 并发（N>=2）序列走 per-request 融合 Forward；`0` 强制逐算子批处理分页路径 | 启用 | `0`, `1` | 否 |
| `TS_BATCHED_FUSED_DECODE` | fused 能力模型 | per-seq fused 路径内的真正 token 批量融合 decode（一张图跑全部 N 个序列）。在 GLM 5.x 上 4 个并发请求可得合计 1.81× decode；在 DeepSeek V4.1 Flash 的 Q4_K_M 上为 2.0×（24.3 → 48.9 tok/s），上限来自路由——每个 token 各自从 384 个专家里挑 6 个。批处理会改变 GEMM 形状，2 bit MoE 可能把这点差别放大成不同的专家选择；设为 `0` 可做串行路径 A/B。 | 开启 | `0`, `1` | 否 |
| `TS_RETAINED_FUSED_CACHE` | 具有可保留 request-owned fused holder 的模型（Gemma 4、Qwen 3.5/3.6） | 保留已完成请求的 holder，用于精确前缀续接；Qwen holder 同时包含 attention K/V 与匹配的 GatedDeltaNet 递归状态 | 启用 | `0`, `1` | 否 |
| `TS_RETAINED_FUSED_CACHE_MAX` | 具有可保留 request-owned fused holder 的模型 | 保留 holder 的 LRU 预算（限制 VRAM；适用时包含递归状态） | `4` | 不适用 | 否 |
| `TS_PREFIX_CHECKPOINTS` | Gemma 4；Qwen 3.5/3.6（GGML 后端） | 在每个会话共享的提示前缀（系统提示、工具、技能）结束处对模型完整状态做检查点，新会话从其副本继续，只需重新预填自己的消息 | 开 | `0`、`1` | 否 |
| `TS_PREFIX_CHECKPOINTS_MAX` | 同上 | 同时保留多少个不同共享前缀的检查点（每个占用一份前缀的 K/V，Qwen 还包含递归状态） | `2` | 不适用 | 否 |
| `TS_KV_INITIAL_TOKENS` | 通过 `ModelBase.ResolveInitialCacheAllocationLength` 确定缓存大小的模型家族（Qwen 3.5/3.6、Gemma 4、GPT-OSS 等 ModelBase 家族；不含自行确定大小的 DeepSeek V4 / GLM 5.x） | 缓存创建时（加载时的主缓存、每个 per-request holder）在任何请求声明预算之前分配的 K/V token 数；`0` 沿用引擎策略（显式 `MAX_CONTEXT` 时为整个窗口，否则为后端默认值）。缓存仍按需增长。内存受限设备把它设小，因为每个保留的 holder 都按此大小付费，主机副本与设备镜像各一份 | `0` | 不适用 | 否 |
| `TS_KV_GENERATION_RESERVE_MAX` | 全部 | 请求预先保留的 K/V 中生成部分的上限（prompt + max_new_tokens）；回复上限不小于窗口时否则每个请求都会保留整个窗口。超过上限后缓存按需增长。`0` = 不限制 | `0` | 不适用 | 否 |
| `TS_KV_HOLDER_POOL_MAX` | 具有 per-request fused holder 的模型（Qwen 3.5/3.6、GPT-OSS） | 已释放的 holder 最多可停放多少个以待复用而不是释放；每个停放的 holder 都占用其完整 K/V 分配 | `64` | 不适用 | 否 |
| `TS_SCHED_DISABLE_BATCHED` | 全部 | 全局按序列 KV-swap 回退 | 关闭 | `0`, `1` | 是 |

本节所有 executor 级开关都通过 `ExecutionOptions.FromEnvironment()` 统一读取，
由 `ExecutionPlanner` 消费（见 `docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md`
的"执行规划"一节）；按模型的 `TS_*_BATCHED` opt-out 则体现为模型声明的
`BatchedForwardAvailable` 能力。

## KV Cache / 上下文

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `KV_CACHE_DTYPE` | 全部 | KV cache 元素类型 | 自动（随模型对齐：模型权重低于 F32 时为 `f16`，否则为 `f32`） | `f32`, `f16`, `q8_0`（运行时还接受 `q4_0`，不参与 sweep） | 是 |
| `TS_KV_PAGED_QUANT_BITS` | 全部分页 KV 模型（不含 `glm-dsa`：MLA 每个 token 只存一行 576 宽的压缩行，DSA 索引器打分的也是这段连续历史，没有可供量化的分页块布局） | TurboQuant 分页 KV 块编解码器（2 bit 使用 affine 的 min+scale 布局） | 关闭（`0`） | `0`, `4`, `8`（运行时还接受 `2`，不参与 sweep） | 是 |
| `TS_N_CPU_MOE` | MoE 模型 | 前 N 层的路由专家留在系统内存：decode 时在主机上做乘法，prefill 时流式送到加速器上跑一整张图 | 关闭（`0`） | `0`, `16`, `all` | 是（GGML 后端、MoE 家族） |
| `TS_CPU_MOE` | MoE 模型 | 卸载所有层的路由专家（等价于 `TS_N_CPU_MOE=all`） | 关闭 | `0`, `1` | 否 |
| `TS_CPU_MOE_THREADS` | MoE 模型 | 主机端专家 matmul 的工作线程数。默认是可用 CPU 并行度（硬件线程数按亲和性掩码与 cgroup CPU 配额收敛后）的一半，上限 64：decode 侧的 matmul 只有一个 token 宽，超过几十个线程后每多一个线程只是多一个屏障参与者（在双路 Xeon 上实测 192 线程比 32 线程慢 7 倍） | min(可用数/2, 64) | - | 否 |
| `TS_HOST_MOE_DEVICE_MIN_BATCH` | 启用卸载的 MoE 模型 | 达到或超过该 batch 大小时，被卸载的层改为在加速器上计算、专家权重流式送入，而不是在主机上算。`0` 恢复纯主机卸载 | `128` | `0`, `32`, `128` | 否 |
| `TS_HOST_MOE_PIN` | 启用卸载的 MoE 模型 | 把被卸载的专家区间页锁定（`cudaHostRegister`），使流式 prefill 走 DMA 而不是经驱动中转（PCIe 5.0 上 9.3 → 55.6 GB/s） | 启用 | `0`, `1` | 否 |
| `TS_HOST_MOE_PIN_MAX_MB` | 启用卸载的 MoE 模型 | 页锁定专家区间的预算 | cgroup / 主机内存上限的 60% | - | 否 |
| `TS_HOST_MOE_EXPERT_FILTER` | 启用卸载的 MoE 模型 | 只流式传输该 batch 实际路由到的专家，并合并成连续区间 | 启用 | `0`, `1` | 否 |
| `MAX_CONTEXT` | 长文本 / 上传文本 | 硬上下文上限。设置了就是硬性要求：缓存放得下就照办，放不下就带着数字拒绝。不设置时，GGUF 宣称的长度只是上限，加载器会按设备真正装得下的量来定——GLM-5.2 宣称 1M token，那是约 93 GiB 的 KV | 模型默认值（是上限而非承诺） | `4096`, `8192`, `16384` | 是 |

## Prefill / Decode 调优

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_PREFILL_CHUNK` | 在 GPT OSS、Qwen 3.5 / 3.6 family 长上下文功能上 sweep；运行时 Gemma 4、Nemotron-H、Mistral 3 也会读取 | 分块 prefill 大小 | 架构默认值 | `256`, `512`, `1024` | 是 |
| `GDN_DISABLE_CHUNKED_PREFILL` | `qwen3next` | 关闭 GDN 分块 prefill | 关闭 | `0`, `1` | 否 |
| `TS_GGML_ASYNC_COMPUTE` | GGML 后端 | 异步 compute 提交 | `ggml_metal` 上启用（`0` 关闭），其他 GGML 后端关闭 | `0`, `1` | 是 |
| `TS_QWEN35_FD_PERSIST` | GGML GPU 后端上的 Qwen 3.5 / 3.6 family | 保留并重放整模型单 token decode 图 | 启用 | `0`, `1` | 否 |
| `TS_GPTOSS_MODEL_DECODE` | GGML 后端上的 GPT OSS | 每个 decode token 用**一张图**跑完整个 transformer（所有层 + MoE + 最终 norm + LM head）；`0` 回退到逐层融合内核 | 启用 | `0`, `1` | 否 |
| `TS_GPTOSS_FD_PERSIST` | `ggml_cuda` / `ggml_vulkan` 上的 GPT OSS | 保留并重放那张整模型 decode 图（padded KV 窗口 + `set_rows`），这正是 ggml-cuda 能捕获它的前提 | 启用 | `0`, `1` | 否 |
| `TS_NEMOTRON_FLASH_DECODE` | GGML 后端上的 Nemotron-H | 在常驻 KV cache 上做设备端单 token 注意力；`0` 恢复主机侧的 C# decode 注意力 | 启用 | `0`, `1` | 否 |
| `TS_QWEN35_METAL_GDN_INPLACE_STATE` | 单设备 `ggml_metal` 上的 Qwen 3.5 / 3.6 family | 让 K=1 GatedDeltaNet 输出与递归状态共享存储，消除逐层状态复制 | 启用 | `0`, `1` | 否 |
| `TS_QWEN35_METAL_TOKEN_INPUT` | `ggml_metal` 上的 Qwen 3.5 / 3.6 family | 在 decode 图内直接从量化表读取 token embedding | 启用 | `0`, `1` | 否 |
| `TS_QWEN35_METAL_KV_CPY` | `ggml_metal` 上的 Qwen 3.5 / 3.6 family | 通过可移动 `CPY` view 追加 K/V，而不是索引 scatter | 启用 | `0`, `1` | 否 |
| `TS_QWEN35_METAL_ASYNC_SUBMIT` | `ggml_metal` 上的 Qwen 3.5 / 3.6 family | decode 和 logits 回读提交后只同步一次 | 启用 | `0`, `1` | 否 |

## 多模态

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `VIDEO_SAMPLE_FPS` | 视频功能 | 按时间抽帧的每秒帧数 | `1` | `1`, `2` | 是 |
| `VIDEO_MAX_FRAMES` | 视频功能 | 抽取视频帧上限 | 不限制 | `8`, `16` | 是 |
| `TS_NEMOTRON_IMAGE_MAX_TILES` | Nemotron-H 图像功能 | 最大图像 tile 数 | 架构默认值 | `4`, `8`, `12` | 是 |

## MLX 专属

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_MLX_BATCHED_MOE_DECODE` | MLX 上的 Qwen 3.5 / 3.6 MoE | 每种 gate/up/down 一次批处理 dispatch，而不是按 expert dispatch | 启用 | `0`, `1` | 是 |
| `TS_MLX_DEVICE_ROUTER` | MLX 上的 Qwen 3.5 / 3.6 MoE | 满足前置条件时在 device 上执行 top-K + softmax router | 启用，且会自动回退 | `0`, `1` | 是 |
| `TS_MLX_PIPELINED_DECODE` | MLX decode 功能 | 模型支持时使用 device-side argmax 的流水化贪心 decode | 满足条件时启用 | `0`, `1` | 是 |
| `TS_MLX_DEVICE_KV_COPY` | MLX | Device 侧 KV scatter | 启用 | `0`, `1` | 否 |
| `TS_MLX_QWEN35_GDN_PACKED_KERNELS` | MLX 上的 Qwen 3.5 / 3.6 family | Packed GDN kernel | 关闭 | `0`, `1` | 是 |

## 矩阵外的纯 C# CPU 后端变量

这些用于调节 `--backend cpu` 背后的常驻工作线程池与量化权重处理。它们是真实的
运行时开关，但没有注册进 `EnvVarMatrix.All`，默认的 TestMatrix 配置也不会扫它们。

| 环境变量 | 适用范围 | 功能影响 | 运行时默认 | 扫描取值 | 默认是否扫描 |
|---|---|---|---|---|---|
| `TS_CPU_THREADS` | `cpu` 后端（100% 纯 C#） | 运行托管 matmul 的常驻工作线程池宽度。默认是可用 CPU 数的**一半**，刻意不是全部：CPU 路径的其余部分仍然使用 ThreadPool，而池内线程在两次任务之间自旋，占满每个核心会把那部分工作饿死。在 122 个 CPU 的配额上实测，每格为两次交替运行（prefill / decode tok/s）：关闭池 21.7,21.0 / 2.0,2.4；32 线程 24.9,24.1 / 4.9,5.0；48 线程 25.6,28.5 / 5.4,6.0；61 线程 24.2,24.9 / 6.3,5.9；122 线程 13.5 / 4.8。122 线程时只有 prefill 回退，解码仍优于关闭池的基线 | 8 核及以下取全部核心，否则 max(8, 可用数/2) | 未注册 | 否 |
| `TS_CPU_POOL` | `cpu` 后端 | `0` 回退到引入线程池之前的行为——按线程数切块的 ThreadPool `Parallel.For`——便于在同一个二进制里做 A/B | 启用 | 未注册 | 否 |
| `TS_CPU_SPIN` | `cpu` 后端 | 池内线程挂起前的自旋次数。在这个宽度下挂起才是最贵的部分（唤醒 N 个线程的开销超过它们要分到的那约 60 微秒工作量），因此默认自旋次数足够多，使稳态下根本不会挂起：同一个模型在 256 时实测 0.1 tok/s，在 4096 时是 7.0 | `4096` | 未注册 | 否 |
| `TS_CPU_TASK_BYTES` / `TS_CPU_TASKS_PER_WORKER` | `cpu` 后端 | 单次托管 matmul 的切分方式：每个工作项对应多少字节权重，以及每个线程最多分到几个工作项。是按**工作量**而不是线程数来定的——旧的按线程数缩放的规则在 122 线程时会为一次 matmul 造出 1024 个极小任务，并且超过 8 线程后就不再有加速 | `131072` / `4` | 未注册 | 否 |

## 矩阵外的 DiffusionGemma 变量

这些变量是真实运行时开关，但目前未注册到 `EnvVarMatrix.All`，也不在默认
TestMatrix 配置中 sweep。

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `DIFFUSION_STEPS` | DiffusionGemma Web UI | 服务端路径每个 block 的去噪步数 | `48` | 未注册 | 否 |
| `DIFFUSION_MAX_BATCH` | DiffusionGemma Web UI | `DiffusionBatchScheduler` 的最大活跃请求数 | `2` | 未注册 | 否 |
| `DIFFUSION_BATCHED_FORWARD` | DiffusionGemma | 真正批处理 canvas decode vs 按时间片执行融合单 canvas decode | 关闭 | 未注册 | 否 |
| `DIFFUSION_NO_PKV` | DiffusionGemma | 关闭 device-glue 后端上的 prompt-KV 缓存 | 关闭 | 未注册 | 否 |
| `DIFFUSION_NO_SC` / `DIFFUSION_SC_TOPK` | DiffusionGemma | self-conditioning 开关与实验 top-K 截断 | 启用 / `32` | 未注册 | 否 |
| `DIFFUSION_NO_FUSED_DECODE` / `DIFFUSION_NO_FUSED_LMHEAD_TAIL` | GGML 后端上的 DiffusionGemma | 关闭融合整模型 diffusion decode 或融合 lm-head tail | 关闭 | 未注册 | 否 |
| `DIFFUSION_LMHEAD_BATCH_CAP_MB` | DiffusionGemma | 回退到按序列 lm-head 前的临时 logits 内存上限 | `300` | 未注册 | 否 |
| `DIFFUSION_VRAM_HEADROOM_MB` | ggml_cuda 上的 DiffusionGemma | 预加载权重之外保留的 VRAM 余量（计算缓冲、device copy） | `2048` | 未注册 | 否 |
| `DIFFUSION_DEVICE_COPY_BUDGET_MB` | ggml_cuda 上的 DiffusionGemma | 模型放不进 VRAM 时 device-copy 缓存的上限（prompt K/V、mask、激活） | `768` | 未注册 | 否 |
| `DIFFUSION_SEGMENTED_DECODE` | ggml_cuda 上的 DiffusionGemma | 强制开启（`1`）/关闭（`0`）逐层融合 decode；模型放不进 VRAM 时自动启用 | 自动 | 未注册 | 否 |
| `DIFFUSION_PIN_STREAMED` | ggml_cuda 上的 DiffusionGemma | 把流式（非常驻）权重复制到页锁定内存以 DMA 速度上传（消耗 RAM） | 关闭 | 未注册 | 否 |

## 矩阵外的投机解码变量

这些变量控制 `TensorSharp.Cli` 与 `TensorSharp.Server` 中可选的投机解码路径
（Qwen 3.6 / GLM 5.2 内嵌的 NextN 块；Gemma 4 独立的 `gemma4-assistant` 草稿 GGUF；
DeepSeek V4 DSpark 与 Muse-Glimmer DFlash 这类块级草稿器；以及无需权重的 n-gram 投机器）。
投机仅对单序列（无并发）请求生效，且只在有收益处启用（ggml 后端与纯 C# 的 `cuda` 后端）。
它们未注册在 `EnvVarMatrix.All` 中，也不在默认 TestMatrix 配置里扫描——矩阵特性目录目前
没有投机解码特性，请用显式运行来验证这些变量。

每个开关都有当前的 `TS_SPEC_*` 写法和旧的 `TS_MTP_*` 写法。应用某个参数时，两个宿主都会
**同时**导出这两种写法，读取端也两种都认：glm-dsa 的**原生**加载器是在模型加载过程中从
C++ 侧读取 `TS_MTP_SPEC` 与 `TS_MTP_DRAFT` 的（它据此决定要不要把多出来的一整层 256 专家
decoder 调进显存，并据此确定图缓存大小），所以这两个名字是一份跨语言契约，不能简单改名。
所有这些也都可以通过两个宿主上的 `--spec*` 参数（或其 `--mtp-*` 别名）设置。

| 环境变量 | 旧写法 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|---|
| `TS_SPEC` | `TS_MTP_SPEC` | Qwen 3.5/3.6、GLM 5.2、Gemma 4、DeepSeek V4、Muse-Glimmer（CLI + 服务端） | 为单序列启用投机解码 | 关闭（`0`） | 未注册 | 否 |
| `TS_SPEC_TYPE` | — | 同上全部 | 投机算法：`auto` \| `draft-head` \| `block` \| `ngram` | `auto` | 未注册 | 否 |
| `TS_SPEC_DRAFT` | `TS_MTP_DRAFT` | 同上全部 | 每个投机步最多起草的 token 数（1-64） | `8` | 未注册 | 否 |
| `TS_SPEC_PMIN` | `TS_MTP_PMIN` | 同上全部 | 草稿置信度门限；含义随算法而定 | 按算法（`0.15` / `0.35` / `0`） | 未注册 | 否 |
| `TS_SPEC_DRAFT_MODEL` | `TS_MTP_DRAFT_MODEL` | Gemma 4（CLI + 服务端） | 独立 `gemma4-assistant` 草稿 GGUF 路径 | 无 | 未注册 | 否 |
| `TS_GLM_MTP` | — | GLM 5.2 | 强制开启（`1`）或关闭（`0`）NextN 块，双向覆盖 `TS_SPEC`/`TS_MTP_SPEC` | 未设置 | 未注册 | 否 |
| `TS_GMTP_NO_FUSED` | — | ggml 后端上的 Gemma 4 | 关闭融合多 token 验证 / 草稿步内核（逐算子回退） | 关闭 | 未注册 | 否 |
| `TS_GMTP_NO_FAST_ROLLBACK` | — | Gemma 4 | 部分接受时恢复保留前缀回滚，而非稠密快速回滚 | 关闭 | 未注册 | 否 |
| `TS_GMTP_BATCHED_TRUNK` | — | Gemma 4 | 验证主干走批处理分页路径，而非线性主干 | 关闭 | 未注册 | 否 |

这些开关背后的设计——把模型架构、投机算法与投机器权重拆成三层——记录在
[Speculative Decoding in TensorSharp](speculative_decoding.md)（英文）。

## 矩阵外的 Muse-Glimmer 与 DFlash 开关

Muse-Glimmer 的融合整模型内核与它的 DFlash 块级草稿模型各有一个 A/B 开关，另外还有
长上下文的尺寸开关。这些都没有注册进 `EnvVarMatrix.All`。完整清单（含逐层追踪开关）见
[Muse-Glimmer 卡片](models/muse-glimmer_zh-cn.md#7-环境变量)。

| 环境变量 | 适用范围 | 功能影响 | 运行时基线 | 扫描取值 | 默认是否扫描 |
|---|---|---|---|---|---|
| `TS_MUSE_GLIMMER_FUSED` | GGML CUDA / Vulkan 上的 Muse-Glimmer | 融合整模型图 vs 逐算子路径 | 开 | 未注册 | 否 |
| `TS_MUSE_GLIMMER_PERSIST` | Muse-Glimmer（融合） | 持久、可被 CUDA 图捕获的计算图 vs 每次调用重建 | 开 | 未注册 | 否 |
| `TS_MUSE_GLIMMER_INGRAPH_EMBED` | Muse-Glimmer（融合） | 在图内完成 embedding gather + 无权重输入 norm（LM head 未绑定时是负收益） | 自动（仅绑定时开启） | 未注册 | 否 |
| `TS_MUSE_GLIMMER_PREFILL_CHUNK` | Muse-Glimmer | 每次 prefill 前向的 token 数；`0` 关闭分块 | `2048` | 未注册 | 否 |
| `TS_MUSE_GLIMMER_SWA_RING` | Muse-Glimmer（融合） | 把 39 个滑动窗口层按 `pad(n_swa + chunk + 1, 256)` 行做环，而不是所有层都按完整上下文分配 | 开 | 未注册 | 否 |
| `TS_MUSE_GLIMMER_SWA_ROWS` | Muse-Glimmer（融合） | 覆盖 SWA 环的行数（诊断用） | 自动 | 未注册 | 否 |
| `TS_MUSE_GLIMMER_VENC_F32` | Muse-Glimmer 视觉塔 | 把塔反量化为 F32（约 7.4 GB），而不是把 GGUF 量化直接喂给 `AddmmQuant` | 关 | 未注册 | 否 |
| `TS_MUSE_GLIMMER_VENC_FUSED` | CUDA 上的 Muse-Glimmer 视觉塔 | 融合视觉块 / flash-attention 路径 | 开 | 未注册 | 否 |
| `TS_MUSE_GLIMMER_DFLASH` | Muse-Glimmer | DFlash 草稿模型 GGUF 路径（等同 CLI 的 `--draft-model`） | 无 | 未注册 | 否 |
| `TS_QWEN35_DFLASH` | Qwen 3.5 / 3.8 | DFlash / DFlash2 草稿模型 GGUF 路径（等同 CLI 的 `--draft-model`） | 无 | 未注册 | 否 |
| `TS_DFLASH_FUSED` | 任意 DFlash 草稿器 | 融合的 `TSGgml_DFlashInject` / `TSGgml_DFlashDraftBlock` 图 vs 逐算子草稿模型 | 开 | 未注册 | 否 |
| `TS_DFLASH_PERSIST` | 任意 DFlash 草稿器 | 重放持久草稿图，而不是每步重建 | 开 | 未注册 | 否 |
| `TS_DFLASH_PREFILL_CHUNK` | 任意 DFlash 草稿器 | 每次投机 prefill 前向的 token 数（驱动的是**主干**，不只是草稿器） | `1024`，并受草稿器环形缓冲与主干自身窗口的限制 | 未注册 | 否 |
| `TS_DFLASH_SELECTOR` | DFlash2 草稿器 | `0` 改为按逐位置 argmax 起草，而不走候选格（仅用于归因分析——权重本来就是带着它训练的） | 开 | 未注册 | 否 |
| `TS_DFLASH_CONV` | DFlash2 草稿器 | `0` 去掉分组动态卷积（同上，仅用于归因分析） | 开 | 未注册 | 否 |
| `TS_DFLASH_SELECTOR_DEBUG` | DFlash2 草稿器（逐算子路径） | `1` 打印前几个 block 的候选格归因：一元项分布、转移项分布，以及这次游走是否离开了一元 argmax | 关 | 未注册 | 否 |
| `TS_Q35_VERIFY_SNAPSHOTS` | Qwen 3.5 / 3.8 投机验证 | `0` 回退为先保存验证前的递归状态副本、再对已接受前缀重新前向，而不是每行保留一份快照 | 开 | 未注册 | 否 |
| `TS_Q35_VERIFY_DEFER_STATE` | Qwen 3.5 / 3.8 投机验证 | `0` 在每次持久化调用后都把窗口末尾的递归状态下载回主机，而不是留在设备上等待 slot 提交；它与快照可以分开测，因为它同样覆盖投机会话中穿插的单行步骤 | 开 | 未注册 | 否 |
| `TS_Q35_VERIFY_STRIDED_VIEWS` | Qwen 3.5 / 3.8 投机验证 | `0` 关闭 CUDA 与 Metal 上连续跨步的 KV view，回退为按 head 的 `set_rows` 写入 | 开 | 未注册 | 否 |
| `TS_QWEN35_SPEC_DEVICE_STATE` | Qwen 3.5 / 3.8 投机解码，Metal 与 CUDA | `0` 让 gated-delta-net 的递归状态在每个投机步前后都排空到主机镜像再上传，而不是留在设备上。在 `ggml_cuda` 的 n-gram 场景中它本身值 1.67x，因此这是应急开关而非调优旋钮 | 开 | 未注册 | 否 |
| `TS_QWEN35_VERIFY_RESIDENT` | Qwen 3.5 / 3.8 投机验证 | `1` 让 conv 与 delta 状态在验证期间常驻设备，省去每次调用约 60 MB 的搬运。**目前会产生错误的输出流**——常驻调用会就地更新状态，被拒绝的草稿无处回滚。仅供测量时显式开启，见[投机解码](speculative_decoding.md) | 关 | 未注册 | 否 |
| `TS_Q35_MTP_DRAFT_PERSIST` | Qwen 3.5 / 3.8 MTP 草稿图 | `1` 允许单层 MTP 草稿图使用持久化 / 重放缓存。默认关闭：这张图曾在 CUDA graph 捕获重放时死锁，保留这个开关是为了在新版 ggml 上重新验证。收益约 1% | 关 | 未注册 | 否 |
| `TS_MTP_FOLD_CATCHUP` | Qwen 3.5 / 3.6 NextN/MTP 投机 | `0` 把草稿头的 catch-up 与第一个草稿步拆成两次调用，而不是折叠成对 `n_accepted + 1` 行的一次前向（llama.cpp draft-mtp 的形状）。收益约 4-5% | 开 | 未注册 | 否 |
| `TS_SPEC_ADAPTIVE` | 投机解码（所有草稿器） | `0` 关闭成本调节器，于是起草不再与普通 baseline 做对比、也永远不会被暂停。用于 A/B 测量：调节器每一轮的 baseline 步骤都是普通 decode，它们并不免费 | 开 | 未注册 | 否 |
| `TS_GGML_LOG_DEBUG` | GGML 后端 | `1` 把 ggml 的 DEBUG 日志通道透传出来而不是丢弃。它承载 CUDA 后端的 "CUDA graph warmup complete" / "reset" 这两行，而这是唯一能看出一张图是否真的被 CUDA graph 捕获的途径 | 关 | 未注册 | 否 |

## 矩阵外的 DeepSeek V4 / V4.1 开关

这些变量配置 DeepSeek 的整模型执行器。V4 有三套（Direct CUDA、原生 ggml、纯 C#）；
V4.1 的服务路径是一套原生 `ggml_cuda` 计算图，另有 `ggml_cpu` 作为标量正确性通道，
以及自己的纯 C# 与 Direct CUDA 执行器作为可移植性通道。它们都没有
注册进 `EnvVarMatrix.All`，因此默认的 TestMatrix 扫描不会覆盖。完整背景见
[V4 卡片](models/deepseek4_zh-cn.md)与 [V4.1 卡片](models/deepseek41_zh-cn.md)。

| 变量 | 适用范围 | 作用 | 默认值 | 在矩阵中 |
|---|---|---|---|---|
| `TS_DSV4_NGPU` | V4 与 V4.1 | 按层切分把整层铺到几张 GPU 上——对这两个架构来说，`--tp N` 设置的就是它。`0` 表示使用所有可见设备 | `0`（全部可见） | 否 |
| `TS_DSV4_UBATCH` | V4 与 V4.1 | Prefill 微批宽度 | 保守配置为 `256`；八卡 A40 的实测 V4.1 配置用 `1024` | 否 |
| `TS_DSV4_THREADS` | V4 与 V4.1 | 纯 GPU 加载时的原生线程池。CPU 专家卸载改用探测到的可用并行度，由 `--cpu-moe-threads N` / `TS_CPU_MOE_THREADS` 设定。在纯 C# 的 `--backend cpu` 执行器上，它设定的是该执行器自己的工作线程池，默认取 `ProcessorCount` 而不是 min(核数, 32) | min(核数, 32) | 否 |
| `TS_DSV4_PERF` | V4 与 V4.1 | `1` 打印分阶段耗时 | 关 | 否 |
| `TS_DSV4_VRAM_RESERVE_MB` / `TS_DSV4_GRAPH_CACHE` / `TS_DSV4_LOAD_THREADS` / `TS_DSV4_LOAD_CHUNK_MB` / `TS_DSV4_MOE_MMAP` | V4 与 V4.1 | 放置余量、计算图缓存深度、权重加载并行度，以及驻留主机的专家是否直接在 GGUF 映射上就地相乘 | 见各卡片 | 否 |
| `TS_DSV4_DSPARK` | 仅 V4 | DSpark 草稿 GGUF，等价于 `--draft-model`。V4.1 会拒绝 V4 的草稿模型——它没有 DSpark 路径 | 未设置 | 否 |
| `TS_DSV41_TP` | V4.1 | `0` 关闭；`2`–`8` 打开**实验性 routed-MoE 张量并行**，且必须与 `--tp` / `TS_DSV4_NGPU` 选中的 GPU 数一致。gate/up 沿 FFN 中间维切分，down 沿输入维切分，partial 经主机中转的 F32 缓冲归约。首次完整 Q2_K 实测比按层切分更慢 | `0` | 否 |
| `TS_DSV41_ENGRAM_DEVICE` | V4.1 | `1` 要求 Engram 表驻留 GPU，放不下就失败；`0` 强制主机映射，需要与 CPU oracle 逐位一致时也用它。不设置则自动且保守：只要不会因此逼出路由专家的 CPU 卸载，就放在 GPU 上 | 自动（放得下就驻留 GPU） | 否 |
| `TS_DSV41_ENGRAM_WARM` | V4.1，仅主机映射 | 读入 Engram 表页，使查表变成一次内存读取而不是一次存储往返。未设置时在模型开始服务后于**后台**预热（默认）；`1` 与此前一致在加载期间同步预热；`0` 从不预热。八卡 A40、Q4_K_M 上 prefill 从 200–252 提升到 452–492 tok/s，decode 从 23–26 提升到 31–33 tok/s；在默认的 GPU 驻留路径上无意义。代价是表本身的主机页缓存（Q2_K 60 GiB，Q4_K_M 103 GiB） | 后台预热 | 否 |
| `TS_DSV4_GRAPH_CACHE_HEADROOM_MB` | V4 与 V4.1 | 图缓存必须为下一张图留出的设备内存，叠加在它持有的最大条目之上。会逐个释放最久未使用的条目直到满足。`0` 回到纯条目数上限，而四个并发的 10.8k token prefill 曾因此耗尽显存 | `1024` | 否 |
| `TS_DSV41_ENGRAM_THREADS` | V4.1，仅主机映射 | 常驻查表工作线程数，`1`–`32`。单个 token 在每张表上要取 24 行互不相关的数据，串行读意味着串行缺页 | min(16, 硬件线程数) | 否 |
| `TS_DSV41_ENGRAM_RANDOM` | V4.1，Linux 上的主机映射 | 对映射的 Engram 区间给出随机访问建议。`0` 关闭，`1` 强制 | 自动 | 否 |
| `TS_DSV41_ENGRAM_SIDECAR` | V4.1 | 显式指定已准备好的 `deepseek41.engram.bin`，而不是使用分片旁边那个 | 与 GGUF 同目录 | 否 |
| `TS_DSV41_SPARSE_FA` | V4.1 | `1` 选择稀疏 flash attention。需显式开启，与稠密路径存在已记录的浮点差异 | 关 | 否 |
| `TS_DSV41_COMPACT_RAW_GATHER` | V4.1 | `1` 为原始滑动窗口选择紧凑 gather。同样需显式开启，同样有浮点差异 | 关 | 否 |
| `TS_DSV41_ALLOW_NON_CUDA_GPU` | V4.1 | `1` 允许 `ggml_vulkan` / `ggml_metal`：普通计算图跑在 GPU 上，只有架构专属算子回退到 CPU 后端，每次都要一次主机往返。之所以需要显式开启，是因为它解除的那道拒绝原本挡住的是*静默*回退 | 关 | 否 |
| `TS_DSV41_VISION_FA` / `TS_DSV41_VISION_BF16_GEMM` | V4.1 视觉伴随文件 | `TS_DSV41_VISION_FA=1` 让图像编码器使用 F16 中间量的 flash attention（真实图像上的特征差异更大）；`TS_DSV41_VISION_BF16_GEMM=0` 选择诊断用的 F32 提升矩阵路径 | 稠密 F32 注意力，BF16 GEMM + F32 累加 | 否 |
| `TS_DSV41_TRACE_DIR` / `TS_DSV41_VISION_TRACE_DIR` | V4.1（诊断） | 文本计算图与视觉编码器的张量转储目录。两者都会保留中间张量并增加设备传输——跑基准时保持不设置 | 未设置 | 否 |
| `TS_DSV4_CPU_TRACE_DIR` / `TS_DSV4_CUDA_TRACE_DIR` | V4.1（诊断） | 纯 C# 与 Direct CUDA V4.1 执行器的同类逐张量转储，文件命名与 `TS_DSV41_TRACE_DIR` 一致，因此可以在两个后端之间逐张量比对，定位第一个发散的张量 | 未设置 | 否 |

## 矩阵外的 GLM 5.x（`glm-dsa`）开关

这些变量配置 GLM 5.x（`glm-dsa`）执行器——`ggml_cuda` / `ggml_vulkan` /
`ggml_cpu` / `ggml_metal` 使用的原生整模型 ggml 路径，以及 `cpu` 与 `cuda` 使用的
托管逐算子路径。它们都未注册在 `EnvVarMatrix.All` 中，默认的 TestMatrix 扫描不会
覆盖；完整清单与背景见 [GLM 卡片](models/glm_zh-cn.md#环境变量)。张量并行相关的三个
开关（`TS_GLM_TP_SHARD`、`TS_GLM_TP_OVERSUBSCRIBE`、`TS_GLM_TP_FUSED`）列在下面的 TP 表里。

| 变量 | 适用范围 | 作用 | 基线 | 扫描取值 | 在矩阵中 |
|---|---|---|---|---|---|
| `TS_GLM_NATIVE` | GLM 5.x | `0` 在 GGML 后端上改走托管逐算子路径而非原生整模型图——正是用来对照两条路径是否一致的 A/B | `1`（原生） | `0`, `1` | 否 |
| `TS_GLM_NGPU` | GGML 上的 GLM 5.x | 按层切分把主干层摊到多少张 GPU 上 | `0`（全部可见 GPU） | `1`, `2`, `3` | 否 |
| `TS_GLM_UBATCH` | GLM 5.x | Prefill 微批。显存允许时 `2048` 在长提示上更快：3x RTX PRO 6000 上 pp2048 为 1145.8，对比 918.9 t/s | `1024` | `512`, `1024`, `2048` | 否 |
| `TS_GLM_THREADS` | `ggml_cpu` 上的 GLM 5.x | CPU 后端线程数（路由专家的 matmul 另用 `--cpu-moe-threads`） | min(核数, 32) | — | 否 |
| `TS_GLM_FA` | GLM 5.x | `0` 关闭 flash attention，退回显式的 `soft_max` 链路 | `1`（flash） | `0`, `1` | 否 |
| `TS_GLM_FUSED_LID` | GLM 5.x | `0` 用基本算子拼出 DSA lightning indexer，而不是用融合的 `ggml_lightning_indexer` | `1`（融合） | `0`, `1` | 否 |
| `TS_GLM_TOPK` | GLM 5.x | `0` 越过索引器 top-k 做稠密注意力——用于对照稀疏选择本身，不是生产设置 | `1`（稀疏） | `0`, `1` | 否 |
| `TS_GLM_OP_OFFLOAD` | GGML 上的 GLM 5.x | 调度器的 op-offload；一旦有任何层的专家驻留主机就会自动关闭 | 自动 | `0`, `1` | 否 |
| `TS_GLM_HC_NATIVE` | GLM 5.3-Flash | `0` 把 Sinkhorn 超连接的 pre/post 算子拆成批量 mul_mat，而不是用融合的 `ggml_dsv4_hc_*` 内核（A/B；后端没有对应内核时会自动拆解） | 探测决定 | `0`, `1` | 否 |
| `TS_GLM_VENC_FUSED` | GLM 5.3-Flash 视觉 | `0` 用托管算子逐块跑 GLM-OCR ViT，而不是走整图原生编码器（`TSGgml_GlmVisionEncoderF32`） | `1`（融合） | `0`, `1` | 否 |
| `TS_GLM_VRAM_RESERVE_MB` | GGML 上的 GLM 5.x | 按层切分在开始放层之前，为计算缓冲在每张卡上预留的余量 | `3072` | — | 否 |
| `TS_GLM_GRAPH_CACHE` | GGML 上的 GLM 5.x | 缓存多少张已构建且已分配的计算图，使相同形状可以直接重放而不必重建 | `8` | — | 否 |
| `TS_GLM_NODES_PER_LAYER` | GGML 上的 GLM 5.x | 每 rank 每层的计算图节点预算 | `256` | — | 否 |
| `TS_GLM_MOE_MMAP` | 带 `--n-cpu-moe` 的 GLM 5.x | `0` 把驻留主机的专家拷进私有缓冲，而不是在 GGUF 映射上就地做乘法 | `1`（映射） | `0`, `1` | 否 |
| `TS_GLM_BATCHED_DECODE` | GLM 5.x | `0` 让原生侧拒绝一切批处理解码，即便全局批处理融合 decode 已启用，也强制走按序列的槽位路径 | `1`（接受） | `0`, `1` | 否 |
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
| `TENSORSHARP_TP_DEVICES` | GGML 后端上的本地 TP | 各 rank 使用的 GPU 序号（逗号分隔，例如 `0,2`） | `0..tp-1` | 未注册 | 否 |
| `TENSORSHARP_TP_NODE_ID` | 全部自回归模型；`cuda`、`ggml_cuda`、`ggml_vulkan` 后端 | 多节点分布式 TP 中本节点的 0 起始编号；必须与 `TENSORSHARP_TP_PEERS` 一起设置 | 未设置（关闭） | 未注册 | 否 |
| `TENSORSHARP_TP_PEERS` | 全部自回归模型；`cuda`、`ggml_cuda`、`ggml_vulkan` 后端 | 分布式 TP 集群中所有节点的 `host:port` 列表（逗号分隔）；必须与 `TENSORSHARP_TP_NODE_ID` 一起设置 | 未设置（关闭） | 未注册 | 否 |
| `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS` | 仅分布式 TP | 各节点向 peer 重试连接多久后放弃 | `120` 秒 | 未注册 | 否 |
| `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS` | 仅分布式 TP | peer 套接字的单次接收超时；卡住的 peer 会让集合通信失败而不是一直挂起 | `300` 秒 | 未注册 | 否 |
| `TENSORSHARP_TP_DISABLE_P2P` | 本地 TP，`cuda` 后端 | `1` 表示所有跨 GPU 传输一律经主机中转，不使用 CUDA 点对点 DMA（与 A16 vGPU 等无 P2P 硬件一致） | 关闭（通过 DMA 自检的设备对使用 P2P） | 未注册 | 否 |
| `TENSORSHARP_TP_HOST_ALLREDUCE` | 本地 TP，`cuda` 后端 | `1` 表示本地 AllReduce 走主机内存（设备→主机、求和、主机→设备）而非设备到设备路径——诊断兜底 | 关闭（设备到设备） | 未注册 | 否 |
| `TS_GGML_TP_PARALLEL` | 本地 TP，GGML 后端 | `0` 表示顺序而非并发地驱动各 rank（诊断用） | 开启（并发 rank 工作线程） | 未注册 | 否 |
| `TS_GGML_TP_FUSED_MATMUL` | 本地 TP，GGML 后端 | `1` 表示由单个线程提交两个 rank 的线性层；每次调用都要为每个 rank 分配设备缓冲，在 Qwen 3.5 35B 上实测慢 2.3× | 关闭（通用按 rank 路径） | 未注册 | 否 |
| `TS_GGML_TP_DEVICE_AR_THRESHOLD` | 本地 TP，GGML 后端 | 超过该元素数量时 AllReduce 走设备集合通信，否则在主机内存中归约 | `262144` | 未注册 | 否 |
| `TS_GGML_F32_RESIDENT` | GGML 后端 | `0` 表示每次调用重新绑定 F32 线性层权重，而不是常驻设备（诊断用） | 开启（常驻设备） | 未注册 | 否 |
| `TS_GEMMA4_TP_FUSED_MOE` | GGML 上 TP 下的 Gemma 4 MoE | `0` 表示从融合的整模 MoE 主干（专家内部 Megatron 切分）回退到逐算子的整专家路径 | 开启（融合主干） | 未注册 | 否 |
| `TS_GLM_TP_SHARD` | GGML 上 TP 下的 GLM 5.x | 切分哪一半：`1` 注意力头，`2` 路由专家，`3` 两者都切。路由专家是在每个专家内部按行切分，而不是按专家 id 分配，因为 `ggml_mul_mat_id` 要求同一 token 选中的专家 id 互不相同 | `3`（两者） | `1`, `2`, `3` | 否 |
| `TS_GLM_TP_OVERSUBSCRIBE` | GGML 上 TP 下的 GLM 5.x | `1` 允许多个 rank 共享一张 GPU，用于在单卡机器上验证切分的正确性 | `0`（一 rank 一卡） | `0`, `1` | 否 |
| `TS_GLM_TP_FUSED` | GGML 上 GLM-5.3-Flash 的本地 TP | `0` 强制使用组合调度器诊断回退，而不是并发提交按 rank 分段计算图。CPU MoE、张量 tracing、部分 `TS_GLM_TP_SHARD` 切分、rank 超额共享 GPU，或后端缺少原生超连接内核时也会自动回退 | 自动（满足条件时分段） | `0`, `1` | 否 |
| `TS_Q4E_LAYER_SPLIT` | `--tp N` 下按层切分的 Qwen 3.8 Flash Next（`qwen4exp`） | 直接指定每张 GPU 分到的层数（逗号分隔，例如 `20,28`），取代自动的显存均衡；给出无法满足的值时会直接抛错，而不是静默忽略。这个架构上的 `--tp N` 是按层切分而非张量并行——`qwen4exp` 不切分任何权重 | 自动（按各设备空闲显存装箱） | 未注册 | 否 |
| `GGML_CUDA_ALLREDUCE` | 本地 TP，`ggml_cuda` | `nccl` / `internal` / `none` —— 直接透传给 ggml 的集合通信选择；显式设置同时会跳过启动前探测 | 自动（构建时能找到 NCCL 且通过探测就用 NCCL） | 未注册 | 否 |
| `TS_GGML_TP_CUDA_GRAPHS` | 本地 TP，`ggml_cuda` | `0` 关闭多 GPU 运行下的 CUDA graph 捕获。TP 下默认**开启**捕获：一个张量并行 token 是几十次按 rank 的小提交，重放的代价远低于重新下发（4×A40：Qwen3.5-9B tp4 88 → 128.5 tok/s，Qwen3.5-35B-A3B tp2 71.3 → 104.1）。历史上曾因捕获污染的隐患而禁用，那个隐患已不再成立——ggml 用 `cudaStreamCaptureModeRelaxed` 捕获。这个 opt-out 会在第一次后端调用之前翻译成原生的 `GGML_CUDA_DISABLE_GRAPHS`，因为 ggml 会在首次使用时锁定该值 | 开启捕获 | 未注册 | 否 |
| `TS_GGML_TP_AR_PROBE` | 本地 TP，`ggml_cuda` | `0` 跳过两项启动前探测；`force` 忽略缓存的判定（`~/.cache/tensorsharp/tp-collective-probe`）重新探测。模型加载前，进程组会检查两件事：所宣称的设备对之间 peer copy 是否真的把数据送到，以及一次小型 NCCL AllReduce 能否端到端完成——一些云主机声称支持 P2P 但数据永远送不到，NCCL 的第一次集合通信随后会让每块 GPU 永远空转。peer 检查失败时会保留 NCCL 但拿掉它的 peer 传输（`NCCL_P2P_DISABLE=1`），这正是超过 2 张 GPU 时仍能保住设备集合通信的原因 | 探测开启，判定按 驱动/NCCL/GPU 组合缓存 | 未注册 | 否 |
| `TS_GGML_TP_AR_PROBE_MS` | 本地 TP，`ggml_cuda` | 每项探测（先 peer copy，后 AllReduce）的完成期限，超时即判定该传输不可用；集合通信随后在 2 张 GPU 时回退到钉页主机内存的 `internal` 管线，更多卡时回退到主机归约。`0` 关闭探测 | `10000` 毫秒 | 未注册 | 否 |
| `GGML_CUDA_AR_BF16_THRESHOLD` | 本地 TP，`ggml_cuda` | ggml 在多大载荷以上把 F32 集合通信转成 BF16；TensorSharp 把 ggml 的默认值提高到 1 MB，使 decode 规模的归约保持精确 | `1 MB`（由 `TSGgml_TensorParallelInit` 设置） | 未注册 | 否 |
| `TS_QWEN35_LAYER_TRACE` | Qwen 3.5/3.6 | `1` 打印首次前向的逐层残差流摘要，单卡与 TP 两条路径都会输出（诊断用） | 关闭 | 未注册 | 否 |

## 矩阵外的 Redis 共享状态变量

这些变量为 `TensorSharp.Server` 配置可选的 Redis 共享状态：用于跨会话复用的共享 KV
缓存层，以及 Redis 支持的 OpenAI Responses API 存储。它们未注册在 `EnvVarMatrix.All`
中。`TS_KV_CACHE_REDIS_URL` 也可通过 `--redis-url` 或 `--paged-kv-redis-url` 设置；
`TS_KV_CACHE_REDIS_TTL_MINUTES` 对应 `--paged-kv-redis-ttl`；
`TS_RESPONSES_STORE_REDIS_URL` 对应 `--redis-url`。

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_KV_CACHE_REDIS_URL` | 仅服务端 | 共享 KV 缓存层的 Redis 连接串；设置后 KV 块持久化到 Redis 以便跨会话复用 | 未设置（关闭） | 未注册 | 否 |
| `TS_KV_CACHE_REDIS_TTL_MINUTES` | 仅服务端 | Redis KV 缓存条目的 TTL（分钟）；`0` = 不过期 | `1440`（24 小时） | 未注册 | 否 |
| `TS_RESPONSES_STORE_REDIS_URL` | 仅服务端 | Responses API 存储的 Redis 连接串；设置后取代内存存储 | 未设置（关闭） | 未注册 | 否 |

## 矩阵外的通用运行时开关

这些变量是真实运行时开关，但目前未注册到 `EnvVarMatrix.All`，也不在默认
TestMatrix 配置中 sweep。

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_PDF_MAX_PAGES` | PDF 文档输入（CLI `--pdf`、服务端 `/api/upload`） | 文本提取与页面图像渲染读取的 PDF 页数上限 | `0`（全部页面） | 未注册 | 否 |
| `TS_GGUF_PREFAULT` / `TS_GGUF_PREFAULT_THREADS` | 模型加载，所有走 `GgufReader` 的架构 | `0` 跳过加载器读文件之前的并行页缓存预热；threads 变量设定它的并发流数。加载路径本身只用一到两条流读文件，因此冷加载被单流带宽卡住——在 MooseFS 卷上单流约 440 MB/s，8-16 流约 1.8 GB/s。该预热在 iOS 上、以及文件大于可用内存一半时会自行跳过，文件已在页缓存中时则是空操作 | 开，`min(16, 核心数)` | 未注册 | 否 |
| `TS_DIRECT_QUANT_WEIGHTS` | `cpu` 后端上的 direct 视频网络（Wan、MiniMax-H3） | `0` 改回在加载时把每个量化权重一次性展开成 F32 再走普通 GEMM，而不是保持 GGUF 存储类型直接参与乘法。展开会占用 4 倍权重内存，每次前向也要多读 4 倍字节；保留该开关是为了在同一个二进制里 A/B 比较两者的数值漂移 | 启用（权重保持量化） | 未注册 | 否 |
| `TS_DUMP_LOGITS` | 所有模型、所有后端 | 把**第一次真实前向**的 logits 以原始 float32 一次性写入该路径。它会刻意**跳过预热前向**：`WarmUpKernels` 在真实提示词之前会自己跑一次丢弃用的 decode 和 prefill，导出那几次等于在一个无意义的 token 上比较两个执行器，而不是在比较模型。这样就能用 logit 向量而不是生成文本来比较两个后端——贪心解码会把一次几乎打平的比分变成一句明显不同的话 | 未设置（不导出） | 未注册 | 否 |
| `TS_FUSED_QKNORM_ROPE` | 直连 `cuda` 后端上的 Qwen 3.5 / 3.6 纯文本 prefill | 融合 QK-Norm + NeoX-RoPE CUDA 内核；`0` 回退到分离的 norm + RoPE 算子（多模态 MRoPE 与其他后端始终走分离路径） | 启用 | 未注册 | 否 |
| `TS_CUDA_QMM_F16GEMM` | 直连 `cuda` 后端，激活行数 ≥ `TS_CUDA_QMM_F16GEMM_MIN_ROWS` 的量化矩阵乘 | 将权重一次性反量化为 F16 并走张量核心 cuBLAS GEMM（ggml 风格的 prefill 路线），替代分块量化内核；`0` 回退到量化内核 | 启用 | 未注册 | 否 |
| `TS_CUDA_QMM_F16GEMM_MIN_ROWS` | 直连 `cuda` 后端 | F16 GEMM 路线的激活行数阈值 | `32` | 未注册 | 否 |
| `TS_CUDA_QMM_F16GEMM_MAX_MB` | 直连 `cuda` 后端 | F16 权重暂存区上限（MB）；超过上限的权重（如 LM head）继续使用量化内核 | `768` | 未注册 | 否 |
| `TS_CUDA_Q80_VEC` | 直连 `cuda` 后端，Q8_0 单行（decode）矩阵乘 | 对 q8_1 量化后的激活行执行每 warp 一列的 dp4a 矩阵-向量乘（类似 ggml `mul_mat_vec_q`）；`0` 回退到精确 FP32 反量化内核 | 启用 | 未注册 | 否 |
| `TS_CUDA_Q80_VEC_MIN_OUT` | 直连 `cuda` 后端 | Q8_0 dp4a 矩阵-向量乘的最小输出宽度（诊断开关） | `0` | 未注册 | 否 |
| `TS_CUDA_Q80_MMQ` | 直连 `cuda` 后端，激活行数在 32..`TS_CUDA_Q80_MMQ_MAX_ROWS` 的 Q8_0 矩阵乘 | 直接在原始 Q8_0 块上执行 int8 张量核心 GEMM（mma.m16n8k32，ggml MMQ 风格），替代反量化+cuBLAS F16 路线；`0` 回退到 F16 GEMM | 启用 | 未注册 | 否 |
| `TS_CUDA_Q80_MMQ_MAX_ROWS` | 直连 `cuda` 后端 | 行数超过该阈值后 F16 GEMM 路线更优（MMQ 的权重扫描次数随 ceil(rows/128) 增长） | `512` | 未注册 | 否 |
| `TS_CUDA_Q80_MMQ2` | 直连 `cuda` 后端 | MMQ GEMM 的 cp.async 暂存变体（拆分的 q8_1 激活暂存 + 原始权重窗口以 cp.async 异步拷贝到共享内存；在 inDim % 256 == 0 时启用，结果逐位一致，prefill 约快 18%）；`0` 固定使用寄存器预取的 MMQ 内核 | 启用 | 未注册 | 否 |
| `TS_CUDA_GDN_PREFILL_SPLIT` | 直连 `cuda` 后端上的 Qwen 3.5 / 3.6 GDN prefill | 三阶段无同步 GDN prefill（并行卷积/归一化 → 寄存器驻留行扫描 → 并行 RMS+门控）；`0` 固定使用旧的单内核逐 token 路径 | 启用（seqLen ≥ 8 且 headKDim = 128） | 未注册 | 否 |
| `TS_CUDA_PREFILL_GRAPH` | 直连 `cuda` 后端上的 Qwen 3.5 / 3.6 纯文本多 token prefill | 在同一 (seqLen, startPos, 缓存标识) 形状第二次运行时把逐算子 prefill 层循环捕获为 CUDA graph，之后以单次 `cuGraphLaunch` 重放（结果逐位一致；捕获失败时自动回退普通路径）；`0` 关闭全部 CUDA graph 捕获（包括 decode graph） | 启用 | 未注册 | 否 |
| `TS_CUDA_DECODE_GRAPH` | 直连 `cuda` 后端上的 Qwen 3.5 / 3.6 纯文本 decode | 把逐算子 decode 步骤（seqLen = 1）捕获为 CUDA graph 并逐 token 重放；位置相关的值（注意力长度、KV 写入槽位、GDN 卷积环索引、RoPE 位置）由内核从一块以锁页主机内存刷新的设备参数块中读取，因此在 KV 缓存扩容之前一个 graph 可服务所有位置（结果逐位一致；捕获失败时自动回退普通路径）；`0` 关闭 | 启用 | 未注册 | 否 |
| `TS_CUDA_PREFILL_GRAPH_MAX` | 直连 `cuda` 后端 | 缓存的 prefill + decode graph 数量（LRU 淘汰；每个 graph 固定持有其捕获时使用的内存池块） | `4` | 未注册 | 否 |
| `TS_CUDA_PREFILL_GRAPH_LOG` | 直连 `cuda` 后端 | 打印 graph 捕获/重放/中止事件（`1`） | 关闭 | 未注册 | 否 |
| `TENSORSHARP_CUDA_POOL_LARGE_MB` | 直连 `cuda` 后端 | 全局大块（≥ 2 MB）显存缓存预算；让 prefill 级激活保持池化，避免每层重复 cuMemAlloc/cuMemFree | `1024` | 未注册 | 否 |
| `TS_CUDA_PROFILE` | 直连 `cuda` 后端 | 退出时打印 CPU 回退算子与主机↔设备同步计数（`1`），含调用点归因（`2`） | 关闭 | 未注册 | 否 |

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
