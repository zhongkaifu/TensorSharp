# TensorSharp 项目状态

本页集中放置仓库级状态，以及不适合塞进 README 的较长说明。

## 当前方向

TensorSharp 是面向 GGUF 模型的原生 .NET 10 推理引擎。当前源码包含 CLI、服务端/Web UI、兼容 HTTP API、AgentHost，以及 TensorAgent iOS/iPadOS 应用。AgentHost 与 TensorAgent 属于以源码为先的能力，最新标签版不一定已经包含它们。

### 最新加入的架构

上一个发布标签之后又落地了两个系列，两者都带着值得先了解的限制。

- **DeepSeek V4.1 Flash（`deepseek41`）**——专用的原生 V4.1 计算图，外加可选的视觉
  伴随文件。服务后端是 `ggml_cuda`；`ggml_cpu` 用同一套图跑标量回退实现，`cpu` 则运行
  纯 C# 的 V4.1 执行器，两者都是正确性与可移植性通道；`cuda` 用 Direct CUDA 引擎自己的
  内核运行 V4.1，但尚无数值门禁；`ggml_vulkan` 与 `ggml_metal` 需要
  `TS_DSV41_ALLOW_NON_CUDA_GPU=1`；`mlx` 会拒绝该检查点。每一种量化都必须先准备好
  Engram sidecar 才能运行——社区 GGUF 仓库都不附带，需用 `eng/dsv41-prepare.py` 生成。
  Q2_K 与 Q4_K_M 均已测试；在 Q4_K_M 上两张 Engram 表各 51.5 GiB，只能留在主机内存映射
  中，因此在 8x46 GB 上必须把路由专家卸载到 CPU（见
  [量化报告](validation/deepseek41-quants/README.md)）。多 GPU 指的是按层切分；routed-MoE 张量并行藏在 `TS_DSV41_TP`
  之后，实测比按层切分更慢。并发请求各有独立槽位，但目前回退到逐槽前向，因此并发还不
  等于批处理的 GPU 吞吐；也没有 V4.1 的 DSpark。哪些已实测、哪些明确未验证，都记录在
  [验证报告](deepseek41_validation.md)与[模型卡片](models/deepseek41.md)中。
- **Hunyuan Dense（`hunyuan-dense`）**——腾讯的稠密 Hunyuan 解码器，加入之后官方
  Hy-MT2 GGUF 才不会因架构未注册而加载失败。第一版：仅文本、单设备、走通用 per-op
  路径，没有工具调用也没有思考模式。见[模型卡片](models/hunyuan-dense_zh-cn.md)。

GLM-5.3 不在上面这份清单里，是因为它不需要新架构：非 Flash 版与 GLM-5.2 是同一套
`glm-dsa` 块形态——79 块（78 层主干加一个 NextN）、256 个路由专家 top-8 外加一个共享
专家、带 lightning indexer 的 MLA、rope base 8e6——因此无需新代码也无需新标志，直接
走 GLM-5.2 的加载路径。它仅文本（[unsloth/GLM-5.3-GGUF](https://huggingface.co/unsloth/GLM-5.3-GGUF)
在任何量化档都没有发布 mmproj，`LoadVisionEncoder` 在 `glm-dsa` 上遇到 `--mmproj` 只会
告警并忽略，而不是让这次运行失败）；`--spec` 只在不传 `--tp`（即默认按层切分）时生效，
因为 `blk.78` 的 NextN 块没有自己的 LM head，只能借用主干的 LM head，而 `--tp` 会把它
按列切开；UD-Q2_K_XL 为 236.4 GiB、分成七个分片。在 8x A40 46 GB 上与 llama.cpp 实测
对比（10,531 token 的 prompt、300 个 decode token、三次取中位数、按整层放置）：decode
打平（20.48 对 20.28 t/s），加载这份 236.4 GiB 检查点快 2.9×（264 秒对 753 秒），而
TTFT 更慢（41.9 秒对 29.0 秒）；见[跨引擎报告](validation/cross-engine-2026-09/README.md)
与[GLM 卡片](models/glm_zh-cn.md#glm-53glm-dsa)。

### TensorAgent 与 iOS

TensorAgent 是使用 .NET MAUI 构建的 iOS/iPadOS 应用，在设备本地运行 TensorSharp 引擎。它把原生 GGML 作为 iOS `.xcframework` 链接进来，在真机上使用 `ggml_metal`，并与 CLI、服务端共享与宿主无关的聊天流水线（`TensorSharp.Chat`）。iOS 目标通过 `TensorSharpIosTargets=true` 启用；它不是独立的数值后端，也不是远程推理服务。

应用包含本地模型下载、保存会话、附件、听写、Agent Skills 以及有界的进程内智能体工具。由于 iOS 不支持 ASP.NET Core runtime hosting，也不允许运行子进程，TensorAgent 使用进程内 loopback server，以及由运行时提供的 shell/Python/JavaScript 集成。它与桌面端共享的是 API 而不是页面：应用自带手机版 UI，绑定同一套 `WebUiChatService` 与 `SkillsService` 路由。

手机带来三条桌面端没有的约束，它们塑造了当前实现：

- **回答进行到一半时屏幕被切走。** 生成过程归宿主侧的管理器所有，而不属于 WebView——视图离开窗口后 WebKit 会挂起该页面。切换应用时这一轮继续进行，返回时页面重新挂接上去。
- **否则每次启动的第一条消息都要为整段共享前缀买单。** 所有会话共享的那段提示词末尾的模型状态，会按模型持久化到磁盘（`IPrefixCheckpointStore` / `PrefixCheckpointFileStore`），并在准入阶段恢复，因此一次启动的首条消息只需付出恢复的代价，而不是完整预填充。
- **iOS jetsam 杀进程时既无栈也无消息。** `EngineMemoryPolicy` 按 jetsam 实际计费的口径来设定引擎的常驻占用——被 wire 住的文件页算在设备头上而不是进程头上——收到内存警告时则释放那些只为下一次请求提速的部分。

构建、模拟器、真机、打包与测试详见 [TensorAgent README](../TensorAgent/README.md)，上述三点的实测数据也在其中。

## 让它跑得更快

简要顺序如下：

1. Wan 选择 step-distilled checkpoint；Qwen-Image-Edit 选择 Lightning LoRA。
2. 后端匹配硬件：NVIDIA 用 `ggml_cuda`，Apple Silicon 和 iOS 用 `ggml_metal`，原生 CPU 用 `ggml_cpu`。
3. 调高级开关前，先降低分辨率、帧数或扩散步数。MiniMax-H3 的实用配置是 `--cfg 1.0` 与 4–8 步。
4. 文本任务只有在模型和负载适合时，再尝试投机解码、CPU MoE offload 或 `--tp N`。

测量数据与限制条件见[引擎对比报告](engine_comparison_report.md)、[Apple Silicon 上 ggml_metal 对比 llama.cpp](perf/metal-vs-llama-cpp.md)、[模型卡片](models/README_zh-cn.md)、[功能说明](../FEATURES_zh-cn.md)和[环境变量功能矩阵](env_var_feature_matrix_zh-cn.md)。

## 详细内容

- [快速开始](../README_zh-cn.md#快速开始)——首次运行与后端选择。
- [计算后端](../USAGE_zh-cn.md#计算后端)——能力、构建要求与回退。
- [Agent Skills 与智能体工作](agent_skills.md)——技能、工具、工作区与安全。
- [TensorAgent](../TensorAgent/README.md)——iOS 应用架构与验证。
- [开发指南](../DEVELOPMENT_zh-cn.md)——项目分层与原生库构建。
