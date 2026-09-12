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

## 状态矩阵

各个方面的实际情况，包含 README 概要里略去的各架构例外。

| 范围 | 状态 |
|---|---|
| 模型家族 | DeepSeek V4 Flash（`deepseek4`）、DeepSeek V4.1 Flash（`deepseek41`）、GLM 5.x（`glm-dsa`、`glm5next`）、Gemma 4、DiffusionGemma、Qwen 3.5/3.6-family（`qwen35`、`qwen35moe`、`qwen3next`）、Qwen 3.8 Flash Next（`qwen4exp`）、GPT OSS、Nemotron-H（含 Nemotron 3 Nano Omni 与 Nemotron 3.5 Lightning，`nemotron_h_moe`）、Mistral 3、Hunyuan Dense（`hunyuan-dense`）、Muse-Glimmer（`muse-glimmer`、`muse_glimmer`）。图像编辑通过 Qwen-Image-Edit（`qwen_image`、`qwen-image` MMDiT）；音视频联合生成通过 MiniMax-H3（`minimax-h3`、`minimax_h3`），纯视频生成通过 Wan 2.1 / 2.2（`wan`、`wan2.1`、`wan2.2`）。 |
| 推理宿主 | CLI、交互式 REPL、ASP.NET Core Web UI、Ollama 风格 API、OpenAI Chat Completions 风格 API 与 OpenAI Responses 风格 API。 |
| iOS 应用 | TensorAgent 支持 iOS/iPadOS，将 GGML 作为 iOS `.xcframework` 链接，并在真机上使用 `ggml_metal`。它共享与宿主无关的聊天流水线（`TensorSharp.Chat`），但通过进程内 loopback 宿主提供自己的手机版页面——iOS 既没有 ASP.NET Core 运行时包，也不能启动子进程。生成过程在应用离开屏幕后仍然继续；共享提示词前缀的 checkpoint 会按模型持久化，使每次启动的第一条消息只需一次恢复而不必完整预填充（在 iPhone 17 Pro Max 上以 Qwen3.5 9B 实测：原本 54 秒的冷启动首条消息，变成 1.2 秒预热加约 0.6 秒的首条消息）；引擎的内存策略也按 iOS jetsam 实际计费的口径来设定。详见 [TensorAgent](../TensorAgent/README.md)。 |
| 后端 | 纯 C# CPU、Direct CUDA/cuBLAS（`cuda`）、MLX Metal（`mlx`）、GGML CPU、GGML Metal、GGML CUDA、GGML Vulkan。DeepSeek V4 另有三套专属的整模型执行器——Direct CUDA、原生 ggml 与纯 C# CPU——都会把权重按层切分到所有可见 GPU（`--tp N` / `TS_DSV4_NGPU` 限定卡数）。DeepSeek V4.1 的服务路径是 `ggml_cuda`；`ggml_cpu` 用同一套原生计算图跑标量回退实现，`cpu` 则是纯 C# 的 V4.1 执行器，两者都是正确性与可移植性通道，而非服务通道。`cuda` 用 Direct CUDA 引擎自己的内核运行 V4.1（不经过 ggml），目前还没有数值门禁。`ggml_vulkan` / `ggml_metal` 需要 `TS_DSV41_ALLOW_NON_CUDA_GPU=1`；`mlx` 会直接拒绝该检查点，而不会把 V4.1 的权重塞进并未实现它的计算图。视频家族中，Wan 是对后端有限制的那一个：它可运行于各 GGML 后端以及 Direct `cuda` / 纯 C# `cpu` 后端，但不支持 MLX。 |
| 多模态 | Gemma 4 图像/视频/音频；Qwen 3.5-family、Qwen 3.8 Flash Next、GLM-5.3-Flash、Mistral 3、Nemotron-H Omni、Muse-Glimmer 图像输入；PDF（CLI `--pdf` + Web UI）。媒体*输出*：Qwen-Image-Edit（图像）、MiniMax-H3（H.264 MP4 **外加一份 32 kHz 立体声 `.wav` 旁挂文件**，两者在同一份打包潜变量里一起生成），以及 Wan 2.1 / 2.2（仅 H.264 MP4 视频，文本→视频与图像→视频）。 |
| 连续批处理 | vLLM 风格分页 KV 缓存、基于内容哈希的前缀共享、共享前缀 checkpoint（所有会话共享的那段提示词末尾的状态会被克隆进每个新会话，因此新会话只需重新 prefill 自己的那条消息；适用于 GGML 上的 Gemma 4 与 Qwen 3.5/3.6，宿主还可通过 `IPrefixCheckpointStore` 让它跨进程重启存活）、迭代级调度器（默认启用，`--no-continuous-batching` 关闭）。分页池常驻主机内存，因此它买到的是内存效率与前缀复用，而不是随并发增长的吞吐。DeepSeek V4 与 GLM 5.x 在同一引擎上通过各自原生的 per-sequence slot 提供服务——压缩后的 MLA 每 token 只有一行缓存，没有可分页的布局——GLM 的批处理融合解码默认启用（设置 `TS_BATCHED_FUSED_DECODE=0` 可切回串行融合 decode；4 路并发下总吞吐 1.81 倍）。Qwen 3.8 Flash Next 出于同样的原因使用逐序列状态持有者——它的 GatedDeltaNet、PLE 与索引器状态同样没有可分页的布局。 |
| 投机解码 | Qwen 3.6、GLM 5.2 与 GLM-5.3（三者均内嵌于 checkpoint——GLM-5.3 的 `blk.78` NextN 块本身是完整的，但它没有自己的 LM head，因此投机只在默认的按层切分下生效，即不传 `--tp` 时）以及 Gemma 4（独立草稿 GGUF，通过 `--draft-model` 加载）的 MTP / NextN 草稿头；DeepSeek V4 的 DSpark 块级起草（仅 `cuda` / `ggml_cuda`）、Muse-Glimmer 与 Qwen 3.8 的 DFlash / DFlash2 块级起草，以及 Nemotron 3.5 Lightning 的 DSpark Markov 头块级起草——它是第一个作为块级起草目标的递归（Mamba-2）主干——这些都通过 `--draft-model` 加载独立的草稿 GGUF；此外还有一个不需要任何草稿权重的 n-gram（prompt-lookup）投机器，用 `--spec-type ngram` 选择，因而在任何检查点上都能用。每个输出 token 都取自主干的一行 logits，并由本次运行自身配置的采样器抽出，因此输出流与普通 decode 产生的完全相同。默认关闭；内嵌草稿头在 CLI 与服务端两端均以 `--spec` 启用，而对以独立 GGUF 发布的草稿器，传入 `--draft-model` 本身即可启用投机。 |
| 张量并行 | Direct `cuda` 后端与 GGML CUDA / Vulkan 后端上的 Megatron-LM 列/行并行 TP（`--tp N` / `TENSORSHARP_TP_DEGREE`，CLI 与服务端均支持）；通过点对点 TCP 的多节点分布式 TP（`--tp-node-id` / `--tp-peers`），采用分层 AllReduce，CUDA P2P 不可用时自动回退到主机中转。覆盖全部自回归架构；GGML 上 Gemma 4 与 Qwen 3.5/3.6 使用 MoE 专家并行与融合的按 rank decode/prefill 计算图。GLM 5.x 默认按层切分，但 GGML GPU 后端上的 `--tp N` 会为 GLM 5.2、GLM-5.3 与 GLM-5.3-Flash 选择仅支持本地单进程的原生 TP（GLM-5.3-Flash 会切 KDA / MLA head 与路由专家隐藏行）；整个 GLM 家族都会在构建模型之前硬拒 `--tp-node-id` / `--tp-peers`，而在 GLM-5.3 上 `--tp N` 只是一个被接受的模式，而非已验证的配置——它会按 rank 复制 MLA 与索引器缓存，占用随之成倍增长、能装下的上下文随之缩短，并且该 checkpoint 上从未跑过 `--tp 1` 以上的配置（`--tp 8` 折算下来每个 rank 需要 41.7 GiB，放不进 46 GB 的卡）。本身不切分权重的架构把同一个 `--tp N` 当作按层切分——每张 GPU 拿一段连续的整层，DeepSeek V4 与 V4.1 即是如此（V4.1 上还可用 `TS_DSV41_TP=N` 打开实验性的 routed-MoE 张量并行，目前实测比按层切分更慢）；Qwen 3.8 Flash Next（`qwen4exp`）上可用 `TS_Q4E_LAYER_SPLIT=20,28` 覆盖自动均衡，遇到无法满足的切分会直接报错而不是静默忽略。启动时会打印实际采用的模式与每张 GPU 的层数/字节分配；两种模式都不支持的架构会在 stderr 上明说，并改用单卡运行。可选 Redis 支撑的 KV 缓存与 Responses API 存储。 |
| Agent Skills | 技能目录来自 `--skills-dir`（或二进制旁的 `skills` 目录），也可通过 `POST /api/skills` 在运行期安装。在 `/v1/chat/completions`、`/v1/responses`、`/api/chat/ollama`（Ollama）与 `/api/chat`（Web UI）上用 `"skills": [...]` 按请求选中，CLI 上用 `--skill`。支持完整工具闭环的家族只接收元数据，并通过进程内应答的 `skills_list` / `skills_read` 激活说明；调用方自己的工具仍照常回传。脚本执行（`skills_run`）默认关闭。Mistral 3 以及没有可解析工具协议的家族（包括 `qwen4exp`）改为内联选中技能正文，且不提供技能 / 代码工具。 |
| 智能体代码工作 | 可选的 `--code-exec` 在同一个有界“模型→工具”循环里提供 `read_file`、`edit_file`、`write_file`、`shell` 与原子 `apply_patch`。Web UI / CLI 聊天保留会话工作区；每个 OpenAI / Ollama 请求只在内部修复轮次间保留一个私有工作区，响应后删除。生成文件可作为产物下载。这是单模型的进程内循环；TensorSharp 目前不提供多智能体委派或逐命令审批工作流。 |
| 沙箱与权限 | 代码执行和技能脚本默认关闭。macOS 使用 Seatbelt，Linux 需要 `bwrap` 0.12.0+；`required` 模式在无法隔离时拒绝运行。Windows 代码执行必须显式传入 `--code-exec-unconfined`，Windows 技能脚本则需以 `--skills-sandbox preferred` 明确接受仅 job-object 的限制。技能脚本联网、代码联网与宿主代办装包是三个独立开关。 |
| 服务端模型范围 | 通过 `--model` 显式托管单个 GGUF；可通过 `--mmproj` 显式指定投影器；不扫描目录。 |
| 可观测性 | 结构化每轮日志、队列状态，以及 Web UI / Ollama / OpenAI 中的 KV 缓存复用指标。 |

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
