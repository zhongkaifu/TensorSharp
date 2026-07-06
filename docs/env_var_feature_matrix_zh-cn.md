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
| `TS_BATCHED_N1_FAST_PATH` | 全部 | 让符合条件的 N=1 步骤也走批处理调度器 | 关闭 | `0`, `1` | 是 |
| `TS_SCHED_DISABLE_BATCHED` | 全部 | 全局按序列 KV-swap 回退 | 关闭 | `0`, `1` | 是 |

## KV Cache / 上下文

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `KV_CACHE_DTYPE` | 全部 | KV cache 元素类型 | `f32` | `f32`, `f16`, `q8_0`（运行时还接受 `q4_0`，不参与 sweep） | 是 |
| `TS_KV_PAGED_QUANT_BITS` | 全部 | TurboQuant 分页 KV 块编解码器 | 关闭（`0`） | `0`, `4`, `8` | 是 |
| `MAX_CONTEXT` | 长文本 / 上传文本 | 硬上下文上限 | 模型默认值 | `4096`, `8192`, `16384` | 是 |

## Prefill / Decode 调优

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_PREFILL_CHUNK` | 在 GPT OSS、Qwen 3.5 / 3.6 family 长上下文功能上 sweep；运行时 Gemma 4、Nemotron-H、Mistral 3、Qwen 3 也会读取 | 分块 prefill 大小 | 架构默认值 | `256`, `512`, `1024` | 是 |
| `GDN_DISABLE_CHUNKED_PREFILL` | `qwen3next` | 关闭 GDN 分块 prefill | 关闭 | `0`, `1` | 否 |
| `TS_GGML_ASYNC_COMPUTE` | GGML 后端 | 异步 compute 提交 | 关闭 | `0`, `1` | 是 |

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
| `DIFFUSION_PROFILE` / `DIFFUSION_STEPTIME` / `DIFFUSION_FUSED_DEBUG` | DiffusionGemma | 开发用计时与融合 kernel 调试诊断 | 关闭 | 未注册 | 否 |

## 矩阵外的 MTP / 投机解码变量

这些变量控制 `TensorSharp.Server` 中可选的 MTP / NextN 投机解码路径（Qwen 3.6 内嵌
NextN 块；Gemma 4 独立 `gemma4-assistant` 草稿 GGUF）。投机仅对单序列（无并发）请求
生效，且只在有收益处（ggml 后端与纯 C# `cuda` 后端）启用。它们未注册在
`EnvVarMatrix.All` 中，也不在默认 TestMatrix 配置里扫描——矩阵特性目录目前没有投机解码
特性，请用显式服务端运行来验证这些变量。`TS_MTP_*` 也可由 `--mtp-*` 服务端 CLI 参数设置。

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_MTP_SPEC` | Qwen 3.6、Gemma 4（服务端） | 为单序列启用投机解码 | 关闭（`0`） | 未注册 | 否 |
| `TS_MTP_DRAFT` | Qwen 3.6、Gemma 4（服务端） | 每个投机步最多起草的 token 数 | `8` | 未注册 | 否 |
| `TS_MTP_PMIN` | Qwen 3.6、Gemma 4（服务端） | 保留草稿 token 所需最低置信度 | `0.75` | 未注册 | 否 |
| `TS_MTP_DRAFT_MODEL` | Gemma 4（服务端） | 独立 `gemma4-assistant` 草稿 GGUF 路径 | 无 | 未注册 | 否 |
| `TS_GMTP_NO_FUSED` | ggml 后端上的 Gemma 4 | 关闭融合多 token 验证 / 草稿步内核（逐算子回退） | 关闭 | 未注册 | 否 |
| `TS_GMTP_NO_FAST_ROLLBACK` | Gemma 4 | 部分接受时恢复保留前缀回滚，而非稠密快速回滚 | 关闭 | 未注册 | 否 |
| `TS_GMTP_BATCHED_TRUNK` | Gemma 4 | 验证主干走批量分页路径，而非线性主干 | 关闭 | 未注册 | 否 |

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
