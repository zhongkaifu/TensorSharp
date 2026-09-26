# TensorSharp 项目状态

本页集中放置仓库级状态，以及不适合塞进 README 的较长说明。

## 当前方向

TensorSharp 是面向 GGUF 模型的原生 .NET 10 推理引擎。当前源码包含 CLI、服务端/Web UI、兼容 HTTP API、AgentHost，以及 TensorAgent iOS/iPadOS 应用。AgentHost 随 CLI 与服务端归档一同发布，也作为 `TensorSharp.AgentHost` NuGet 包发布；TensorAgent 只能从源码构建，因为没有任何发布工作流会构建这个 iOS 应用。`v2026.09.01` 标签之后合入的改动（其中包括 Qwen-Image-2.1、Bonsai2、DiffusionGemma 图像输入与 Jev API、子智能体、Playwright 浏览器技能以及 GB10 发布归档）在下一个标签之前只存在于源码构建中；其中会改变现有配置行为的改动列在[发布说明](#发布说明自上个标签以来的行为变化)中。

### 嵌入模型与服务

当前源码新增 GGUF BERT/XLM-R 句向量编码器，覆盖 Snowflake Arctic Embed L v2.0 Q8_0 与 all-MiniLM-L6-v2 Q8_0。`--embeddings` 启动独立的常驻编码器，提供 OpenAI `/v1/embeddings`、Ollama `/api/embed` 与旧版 `/api/embeddings`。后端为 100% 纯 C# CPU（`cpu`）与原生 GGML CPU（`ggml_cpu`）、Metal、CUDA；验证范围和复现实验见[嵌入指南](embeddings_zh-cn.md)。旧版发布归档不一定包含这项功能。

### 最新加入的架构

最新加入的两个架构系列（均已包含在 `v2026.09.01` 标签中）都带着值得先了解的限制。

- **DeepSeek V4.1 Flash（`deepseek41`）**——专用的原生 V4.1 计算图，外加可选的视觉
  伴随文件。服务后端是 `ggml_cuda`；`ggml_cpu` 用同一套图跑标量回退实现，`cpu` 则运行
  纯 C# 的 V4.1 执行器，两者都是正确性与可移植性通道；`cuda` 用 Direct CUDA 引擎自己的
  内核运行 V4.1，但尚无数值门禁；`ggml_vulkan` 与 `ggml_metal` 需要
  `TS_DSV41_ALLOW_NON_CUDA_GPU=1`；`mlx` 会拒绝该检查点。当前 vcruz305 GGUF 已包含
  Engram 权重与哈希常量，TensorSharp 直接读取，无需生成或提供单独的 Engram 文件（这是该标签之后的变化，见
  [发布说明](#发布说明自上个标签以来的行为变化)）。
  Q2_K 与 Q4_K_M 的历史测试与当前文件的验证分别记录；Q4_K_M 的两张 Engram 表各 51.5 GiB。
  在 8x46 GB 上，这些表留在主机内存映射中，路由专家需要卸载到 CPU（见
  [Q4_K_M 吞吐报告](perf/deepseek41-q4km-throughput.md)；量化报告
  `docs/validation/deepseek41-quants/README.md`（本地验证记录，未提交到 Git））。多 GPU 指的是按层切分；routed-MoE 张量并行藏在 `TS_DSV41_TP`
  之后，实测比按层切分更慢。并发请求各有独立槽位，但目前回退到逐槽前向，因此并发还不
  等于批处理的 GPU 吞吐。V4.1 的 DSpark 属于实验性功能：`--draft-model` 只能在 `ggml_cuda`
  或 `ggml_cpu` 上加载 `deepseek41-dspark` 草稿器，仅在合成测试夹具上验证过，尚未实测任何
  训练好的草稿器。多轮对话会复用 KV 前缀：普通聊天会丢弃推理内容，使渲染结果在上一轮 assistant
  头之后一个 token 处分叉，原生执行器回退到该位置，而不是重新 prefill 整段对话；由于生成回答会让
  原始滑动窗口环回绕，这需要为每个槽位保存该环的检查点。哪些已实测、哪些明确未验证，都记录在
  [验证报告](deepseek41_validation.md)与[模型卡片](models/deepseek41_zh-cn.md)中。
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
TTFT 更慢（41.9 秒对 29.0 秒）；见[GLM 卡片](models/glm_zh-cn.md#glm-53glm-dsa)。完整的跨引擎报告位于
`docs/validation/cross-engine-2026-09/README.md`（本地验证记录，未提交到 Git）。

Bonsai2 27B 同样不需要新的架构字符串。它的 `Ternary-Bonsai-2-27B-PQ2_0.gguf` 与
`-PTQ1_0.gguf` 声明的是 `qwen35`，靠 PRISM 的 `prism.hadamard.*` 元数据以及 PQ2_0 / PTQ1_0
张量被识别；TensorSharp 在加载时把这些权重转码为 GGML Q2_0，并套上自己实现的带符号
Hadamard 变换。它需要单设备 GGML 后端（`cpu`、`cuda`、`mlx` 与 `--tp` 都会被拒绝），图像
输入通过配套的投影器。验证覆盖 Metal（M5 Pro）上的功能冒烟测试，外加一次简短的 GGML CPU
后端 PQ2 检查；CUDA、Vulkan 与 iOS 均未验证。之后修复了一个仅在 CUDA 上出现的 Hadamard 缺陷，
也有 CUDA 对比脚本，但尚未提交任何 CUDA 结果。见
[Bonsai2 卡片](models/bonsai2_zh-cn.md)。

### 发布说明：自上个标签以来的行为变化

`v2026.09.01` 标签之后合入、会改变现有配置行为的改动：

- **Qwen-Image：只加载 Qwen-Image-2.1。** 更早的 Qwen-Image 与 Qwen-Image-Edit 检查点（例如
  Qwen-Image-Edit-2511）会在加载时被拒绝（退出码 2）。`--qwen-image-lora` 与 `--offload-cpu` 现在是
  硬错误，并会说明改用什么做法（LoRA 插件用 `--lora`；内存不足时减小 `--width` / `--height`）；`TS_QWEN_IMAGE_LORA` 会被拒绝；LoRA 插件改用 `--lora` / `--lora-scale` /
  `--lora-config` 加载。见 [Qwen-Image-2.1 卡片](models/qwenimage21_zh-cn.md)。
- **服务端聊天路径默认启用子智能体。** 在 `/v1/chat/completions`、`/v1/responses`、`/api/chat/ollama`
  与 Web UI 的 `/api/chat` 上，支持工具的模型即使没有技能、也没有 `--code-exec`，也会拿到五个协调工具
  （`spawn_agent`、`wait_agent`、`send_input`、`close_agent`、`list_agents`）。用 `--no-multi-agent`、
  `TS_NO_MULTI_AGENT` 或请求中的 `"multi_agent": false` 可关闭委派。见[子智能体](multi_agent.md)。
- **代码工具：`apply_patch` 负责修改，`write_file` 负责新建。** 不再向模型声明 `edit_file`，也不再声明
  `write_file` 的 `overwrite` 选项；因此 `write_file` 遇到已存在的路径会拒绝，并提示模型改用 `apply_patch`
  （模型若仍自行传入 `overwrite: true`，宿主依然会照做）。
- **DeepSeek V4.1：Engram 改为从 GGUF 读取。** 按分词器生成的 `deepseek41.engram.bin` 旁挂文件，以及生成它的
  `eng/dsv41-prepare.py`，均已移除。TensorSharp 读取当前 vcruz305 发布版（revision
  `58d8ac86298fdf85a2440defee08b1abcad32e45`）中内嵌的 Engram 哈希常量，缺少这些常量的 GGUF 会在加载时被拒绝
  （"Missing or invalid DeepSeek V4.1 GGUF metadata"）。`v2026.09.01` 的流程使用 revision
  `8e0c4de3cb6519bfc11ed69dc87184b457a57bb5` 加旁挂文件，因此旧的下载需要升级到当前发布版。见
  [V4.1 卡片](models/deepseek41_zh-cn.md)。
- **Qwen 3.8 Flash Next 的工具调用可以解析了。** `qwen4exp` 现在返回结构化工具调用，因此会拿到技能、代码工具与
  子智能体，而不再只是内联收到选中的技能正文。见
  [Qwen 3.8 Flash Next 卡片](models/qwen38-flash-next_zh-cn.md#工具调用与-agent-工作流)。
- **随仓库提供的模型配置会下载缺失文件。** `config/` 下的 Gemma 4、Qwen 3.5 / 3.6、Qwen 3.8-27B 智能体、GPT OSS
  与 DiffusionGemma 配置现在把每个模型文件写成带 Hugging Face URL 与 SHA-256 的下载条目，`config/*.json` 中的每个
  下载条目都固定到完整的 commit。命令行（或后出现的 `--config` 文件）设置了的单值参数，会在解析之前丢弃文件中的
  对应条目，因此自己传入 `--model` 或 `--mmproj none` 就会跳过这次下载，并在 stderr 上打印一行 `[config]`；
  `--stop`、`--lora`、`--skills-dir` 这类可重复参数则追加到文件的值之后。在服务端，这也让命令行上的
  `--gpu-device` 或 `--kv-cache-dtype` 优先于文件中的值（以前并非如此）。见
  [配置说明](../config/README.md#auto-download)。
- **CLI 的拼写与服务端一致。** `--penalty-last-n` 已移除：CLI 的惩罚窗口改为 `--repeat-last-n`，与服务端及请求字段
  `repeat_last_n` 同名，因此配置里的 `repeat-last-n` 键（CLI 过去会丢掉它）现在在 CLI 上也生效；在任一宿主上或作为
  配置键使用 `--penalty-last-n` 都会以退出码 1 结束，并打印 `Configuration error: --penalty-last-n was removed: …`。
  别名 `--paged-kv-cache` / `--no-paged-kv-cache` 也以同样方式移除（请用 `--paged-kv` / `--no-paged-kv`）。
  已列出的 CLI 选项现在也不区分大小写并接受 `--option=value`；带了值的开关或后面没有值的取值参数属于配置错误；
  CLI 也接受 `--mmproj none`。
- **对不起作用的参数给出启动警告。** 没有 `--spec` 或 `--draft-model`（且未设置 `TS_SPEC`）时给出的
  `--spec-type`、`--spec-draft` 或 `--spec-pmin`，会在两个宿主上都打印一条说明投机保持关闭的警告。在服务端，
  独立分页 KV 参数（`--paged-kv*`、`--paged-kv-redis-*`）会打印一条说明它们不生效的警告。DiffusionGemma、Wan 与
  MiniMax-H3 收到 `--tp N` 时会打印警告并只用一张 GPU 运行，分布式组则在加载时被拒绝（退出码 2）。
- **默认技能根目录。** 未指定 `--skills-dir` / `TS_SKILLS_DIR` 时，技能来自从工作目录向上直到 Git 根目录
  之间所有已存在的 `.agents/skills` 目录（由近及远），然后是二进制旁的 `skills` 目录。服务端上该目录同时是
  `POST /api/skills` 的上传目录，因此总是最先扫描——排在 `.agents/skills` 根目录之前，即使指定了
  `--skills-dir` / `TS_SKILLS_DIR` 也会保留——同名技能以它为准；CLI 没有上传目录。`.sh` 技能脚本用
  `bash` 运行。
- **Qwen 3.6 / Qwen 3.8-27B 上 `--spec` 在前缀复用之后仍会投机。** 内嵌 NextN 头过去遇到复用了前缀的序列
  （聊天的下一轮、预热或从磁盘恢复的共享前缀、Radix 命中）就不再启用，因此在 `--spec` 下只有第一轮会投机。
  现在它在这段空缺之后只重启自己的草稿缓存，每一轮都能重新启用投机（适用于默认融合 verify 路径上的单独请求），而主干保留复用的前缀、prefill 走正常路径。
  `--spec` 不会关闭 Radix 缓存；`--no-prefix-cache` 仍会关闭它。见[投机解码](speculative_decoding.md)。

以下改动已包含在 `v2026.09.01` 中，从 `v3.4.0.0` 或更早版本升级时需要留意：

- **默认启用 Radix 前缀缓存。** 提示词复用改走引擎的 Radix（基数树）缓存（`TS_PREFIX_CACHE_MODE=tree`）；
  `TS_PREFIX_CACHE_MODE=legacy` 恢复旧的块哈希共享，`--no-prefix-cache` / `TS_SCHED_PREFIX_CACHE=0` 关闭复用。
- **Qwen 3.5/3.6：图片之后的输出会改变。** 图片之后的 token 现在在 Qwen-VL 压缩后的 M-RoPE 位置上
  生成（KV 下标加上按序列保存的 rope 偏移，与 HF、SGLang 一致），而不是在绝对 KV 下标上，因此对图片
  提示的回复与之前的构建不同，并与正确实现一致。图片之后的后续回合重新复用缓存（Metal 上复用 98%
  的提示，首 token 0.13 s，而不是约 1.1 s）。已保存的 Qwen 3.5 共享前缀检查点文件升级到格式版本 2；
  版本 1 的文件会被忽略并给出警告，然后重新写入。原生库必须与托管代码一起重新构建。详见
  [Qwen 3.5 模型卡](models/qwen35_zh-cn.md#图片之后的位置m-rope-偏移delta)。

### TensorAgent 与 iOS

TensorAgent 是使用 .NET MAUI 构建的 iOS/iPadOS 应用，在设备本地运行 TensorSharp 引擎。它把原生 GGML 作为 iOS `.xcframework` 链接进来，在真机上使用 `ggml_metal`，并与 CLI、服务端共享与宿主无关的聊天流水线（`TensorSharp.Chat`）。iOS 目标通过 `TensorSharpIosTargets=true` 启用；它不是独立的数值后端，也不是远程推理服务。

应用包含本地模型下载、保存会话、附件、听写、Agent Skills、有界的进程内智能体工具，以及“Ask TensorAgent”共享扩展——它把其他应用共享过来的文本、链接、网页、图片、影片、音频、PDF 与文档变成一条尚未发送的聊天草稿。由于 iOS 不支持 ASP.NET Core runtime hosting，也不允许运行子进程，TensorAgent 使用进程内 loopback server，以及由运行时提供的 shell/Python/JavaScript 集成。它与桌面端共享的是 API 而不是页面：应用自带手机版 UI，绑定同一套 `WebUiChatService` 与 `SkillsService` 路由。

手机带来三条桌面端没有的约束，它们塑造了当前实现：

- **回答进行到一半时屏幕被切走。** 生成过程归宿主侧的管理器所有，而不属于 WebView——视图离开窗口后 WebKit 会挂起该页面。用户在应用内切换页面时这一轮继续生成；切换到其他应用时这一轮不会丢失，但 iOS 禁止后台 GPU 计算，因此应用不在前台时生成会暂停，回到前台后继续，页面重新挂接到这一轮。
- **否则每次启动的第一条消息都要为整段共享前缀买单。** 所有会话共享的那段提示词末尾的模型状态，会按模型持久化到磁盘（`IPrefixCheckpointStore` / `PrefixCheckpointFileStore`），并在准入阶段恢复，因此一次启动的首条消息只需付出恢复的代价，而不是完整预填充。
- **iOS jetsam 杀进程时既无栈也无消息。** `EngineMemoryPolicy` 按 jetsam 实际计费的口径来设定引擎的常驻占用——被 wire 住的文件页算在设备头上而不是进程头上——收到内存警告时则释放那些只为下一次请求提速的部分。

其他当前行为：

- **设备。** 支持 iPhone 与 iPad，仅 arm64，需要 iOS 17.0 或更高版本。应用采用单窗口的 UIKit scene 生命周期，由此修复了用 iOS 27 SDK 构建时的启动崩溃；已记录的真机运行是在 iOS 26.6.1 上。
- **投机解码默认开启**（与桌面宿主不同）：Gemma 4 E4B 与 12B 条目在草稿头已下载时使用草稿头，否则应用选择 n-gram 投机（Bonsai 8B 是 `qwen3` 模型，没有投机路径）。修改该设置会从下一条回复起作用于正在运行的引擎；启动环境中设置的 `TS_SPEC` / `TS_SPEC_TYPE` 仍然优先。
- **子智能体有开关。** 设置 > Sandbox >“Sub-agents”开启时（默认开启，使用宿主默认上限），支持工具的对话会拿到委派工具；关闭后从下一条消息起不再声明这些工具与协调提示，效果与服务端的 `--no-multi-agent` 相同。手机页面不显示子智能体面板，也没有发布任何设备端实测数据。
- **桌面宿主。** 当 `TensorAgent.Core` 运行在 macOS、Linux 或 Windows 上时，`AgentExecutionMode.Auto` 以 OS 沙箱下的原生进程运行代码，而不是嵌入式解释器；设置了 `networkHosts` 允许列表时，它会拒绝需要联网的启动，因为进程沙箱无法强制执行该列表。
- **CI。** PR CI 通过 `InferenceWeb.Tests` 检查应用的项目文件、Info.plist、entitlements、共享扩展与原生导出清单，但不运行 `TensorAgent.Tests`，也没有任何工作流构建 iOS 应用。

构建、模拟器、真机、打包与测试详见 [TensorAgent README](../TensorAgent/README.md)，上述三点的实测数据也在其中。

## 状态矩阵

各个方面的实际情况，包含 README 概要里略去的各架构例外。

| 范围 | 状态 |
|---|---|
| 嵌入模型 | GGUF BERT/XLM-R：Snowflake Arctic Embed L v2.0、all-MiniLM-L6-v2；纯 C# CPU 与原生 GGML CPU/Metal/CUDA，独立 `--embeddings` 服务，OpenAI/Ollama 单条与批量 API。上下文、分词、质量及实测范围见[指南](embeddings_zh-cn.md)。 |
| 模型家族 | DeepSeek V4 Flash（`deepseek4`）、DeepSeek V4.1 Flash（`deepseek41`）、GLM 5.x（`glm-dsa`、`glm_dsa`、`glm5next`）、Gemma 4、DiffusionGemma、Qwen 3.5/3.6-family（`qwen35`、`qwen35moe`、`qwen3next`）、Qwen 3.8 Flash Next（`qwen4exp`）、Qwen 3 / Qwen 2 插件（`qwen3`、`qwen2`、`qwen2vl`、`qwen2_vl`；仅文本）、Bonsai Q1_0（8B 走 `qwen3`，27B 走 `qwen35`）与 Bonsai2 27B（带 PRISM `prism.hadamard.*` 元数据和 PQ2_0 / PTQ1_0 张量的 `qwen35` 文件）、GPT OSS、Nemotron-H（含 Nemotron 3 Nano Omni 与 Nemotron 3.5 Lightning；`nemotron_h`、`nemotron_h_moe`、`nemotron_h_omni`）、Mistral 3（`mistral3`，以及标为 `llama` 的 Mistral Small 3.x 文件）、Hunyuan Dense（`hunyuan-dense`）、Muse-Glimmer（`muse-glimmer`、`muse_glimmer`）。文生图与图像编辑通过 Qwen-Image-2.1（`qwen_image`、`qwen-image`）；音视频联合生成通过 MiniMax-H3（`minimax-h3`、`minimax_h3`），纯视频生成通过 Wan 2.1 / 2.2（`wan`、`wan2.1`、`wan2.2`）。 |
| 推理宿主 | CLI、交互式 REPL、ASP.NET Core Web UI、Ollama 风格 API、OpenAI Chat Completions 风格 API、OpenAI Responses 风格 API，以及 Jev 类型化决策 API（`POST /v1/systemone`，由已加载的 DiffusionGemma 模型提供，状态可以是文本或最多 8 张图像；见 [Jev](models/jev_zh-cn.md)）。 |
| iOS 应用 | TensorAgent 支持 iOS/iPadOS，将 GGML 作为 iOS `.xcframework` 链接，并在真机上使用 `ggml_metal`。它共享与宿主无关的聊天流水线（`TensorSharp.Chat`），但通过进程内 loopback 宿主提供自己的手机版页面——iOS 既没有 ASP.NET Core 运行时包，也不能启动子进程。它支持 iPhone 与 iPad（arm64，iOS 17.0 或更高版本）。应用离开屏幕后这一轮不会丢失（应用不在前台时生成暂停，回到前台后继续）；投机解码与子智能体委派默认开启；共享提示词前缀的 checkpoint 会按模型持久化，使每次启动的第一条消息只需一次恢复而不必完整预填充（在 iPhone 17 Pro Max 上以 Qwen3.5 9B 实测：原本 54 秒的冷启动首条消息，变成 1.2 秒预热加约 0.6 秒的首条消息）；引擎的内存策略也按 iOS jetsam 实际计费的口径来设定。详见 [TensorAgent](../TensorAgent/README.md)。 |
| 后端 | 纯 C# CPU、Direct CUDA/cuBLAS（`cuda`）、MLX Metal（`mlx`）、GGML CPU、GGML Metal、GGML CUDA、GGML Vulkan。DeepSeek V4 另有三套专属的整模型执行器——Direct CUDA、原生 ggml 与纯 C# CPU——都会把权重按层切分到所有可见 GPU（`--tp N` / `TS_DSV4_NGPU` 限定卡数）。DeepSeek V4.1 的服务路径是 `ggml_cuda`；`ggml_cpu` 用同一套原生计算图跑标量回退实现，`cpu` 则是纯 C# 的 V4.1 执行器，两者都是正确性与可移植性通道，而非服务通道。`cuda` 用 Direct CUDA 引擎自己的内核运行 V4.1（不经过 ggml），目前还没有数值门禁。`ggml_vulkan` / `ggml_metal` 需要 `TS_DSV41_ALLOW_NON_CUDA_GPU=1`；`mlx` 会直接拒绝该检查点，而不会把 V4.1 的权重塞进并未实现它的计算图。视频家族中，Wan 是对后端有限制的那一个：它可运行于各 GGML 后端以及 Direct `cuda` / 纯 C# `cpu` 后端，但不支持 MLX。Qwen-Image-2.1 只能运行在 GGML 后端上，Bonsai2 需要单设备 GGML 后端。 |
| 发布构建与 CI | 打标签的发布会构建自包含的 CLI 与服务端归档：Windows x64（CPU/CUDA）、Linux x64（CPU/CUDA）与 macOS arm64。自 `v2026.09.01` 之后，发布工作流还会构建面向 NVIDIA GB10 / DGX Spark 的实验性 `linux-arm64-cuda13-GB10` 归档（CUDA 13、SM121a），它在没有 GPU 的托管 ARM64 runner 上用 Docker 构建；第一个带有该归档的标签发布将是下一个标签。已记录的 GB10 真机冒烟数据是历史数据，早于上游重新集成，不能为当前代码背书（[GB10 构建](../DEVELOPMENT_zh-cn.md#gb10--dgx-spark-构建容器实验性)）。PR CI 在 x64 与 ARM64 Linux 上运行 `InferenceWeb.Tests` 的 CPU 正确性测试与 GB10 容器门禁检查；发往 `main` 的 PR 还会在自托管 CUDA runner 上运行一次引擎对比冒烟测试（Gemma 4 12B，TensorSharp 对比 llama.cpp，`test-matrix.yml`）。PR CI 不运行 `TensorAgent.Tests`，也不构建 iOS 应用。打标签的发布还会推送 `eng/verify-packages.ps1` 所列的 NuGet 包（含 `TensorSharp.AgentHost`）。 |
| 多模态 | Gemma 4 图像/视频/音频；Qwen 3.5-family（含带配套投影器的 Bonsai2）、Qwen 3.8 Flash Next、GLM-5.3-Flash、Mistral 3、Nemotron-H Omni、Muse-Glimmer、DiffusionGemma 图像输入；Qwen 3.8 Flash Next 的 `video_url` 视频；DeepSeek V4.1 通过视觉伴随文件支持图像与视频；Nemotron-H 只有加载了携带 Parakeet 音频塔的伴随 GGUF 时才支持音频（公开 GGUF 都不附带）；DiffusionGemma 拒绝音频与 `video_url` 视频（Web UI 上传的视频只会以逐帧普通图像的形式送入模型）；PDF（CLI `--pdf` + Web UI）。媒体*输出*：Qwen-Image-2.1（图像，可通过 `--lora` 加载 LoRA 插件，并对文本与参考图 token 默认启用前缀 KV 缓存）、MiniMax-H3（H.264 MP4 **外加一份 32 kHz 立体声 `.wav` 旁挂文件**，两者在同一份打包潜变量里一起生成），以及 Wan 2.1 / 2.2（仅 H.264 MP4 视频，文本→视频与图像→视频）。 |
| 连续批处理 | vLLM 风格分页 KV 缓存、默认启用的 Radix（基数树）前缀缓存（`TS_PREFIX_CACHE_MODE=tree`；`legacy` 选择旧的基于内容哈希的前缀共享，`--no-prefix-cache` / `TS_SCHED_PREFIX_CACHE=0` 关闭全部复用；`--spec` 不会关闭它），覆盖 Qwen 3.5-family、Gemma 4、GLM 5.x、Qwen 3.8 Flash Next、DeepSeek V4 / V4.1、Qwen 3、GPT OSS、Mistral 3、Hunyuan Dense、Muse-Glimmer 与 Nemotron-H，但不包括 DiffusionGemma 与媒体生成模型；共享前缀 checkpoint（所有会话共享的那段提示词末尾的状态会被克隆进每个新会话，因此新会话只需重新 prefill 自己的那条消息；适用于 GGML 上的 Gemma 4、Qwen 3.5/3.6 与 Qwen 3.8 Flash Next，宿主还可通过 `IPrefixCheckpointStore` 让 Gemma 4 或 Qwen 3.5-family 的 checkpoint 跨进程重启存活）、迭代级调度器（默认启用，`--no-continuous-batching` 关闭）。分页池常驻主机内存，因此它买到的是内存效率与前缀复用，而不是随并发增长的吞吐。DeepSeek V4 与 GLM 5.x 在同一引擎上通过各自原生的 per-sequence slot 提供服务——压缩后的 MLA 每 token 只有一行缓存，没有可分页的布局——GLM 的批处理融合解码默认启用（设置 `TS_BATCHED_FUSED_DECODE=0` 可切回串行融合 decode；4 路并发下总吞吐 1.81 倍）。Qwen 3.8 Flash Next 出于同样的原因使用逐序列状态持有者——它的 GatedDeltaNet、PLE 与索引器状态同样没有可分页的布局。 |
| 投机解码 | Qwen 3.6、Qwen 3.8-27B、GLM 5.2 与 GLM-5.3（均内嵌于 checkpoint——GLM-5.3 的 `blk.78` NextN 块本身是完整的，但它没有自己的 LM head，因此投机只在默认的按层切分下生效，即不传 `--tp` 时）、Gemma 4（独立草稿 GGUF，通过 `--draft-model` 加载）以及 Qwen 3.8 Flash Next（独立的共享 MTP head GGUF，通过 `--draft-model` 加载，仅限 GGML 后端）的 MTP / NextN 草稿头；DeepSeek V4 的 DSpark 块级起草（仅 `cuda` / `ggml_cuda`）、DeepSeek V4.1 的实验性 `deepseek41-dspark` 草稿器（仅 `ggml_cuda` / `ggml_cpu`，仅在合成测试夹具上验证过，尚未实测任何训练好的草稿器）、Muse-Glimmer 与 Qwen 3.8-27B 的 DFlash / DFlash2 块级起草（Nemotron-H 拒绝投机解码：它的 verify 与 decode 内核结果不一致，投机输出会与普通解码不同）——这些都通过 `--draft-model` 加载独立的草稿 GGUF；此外还有一个不需要任何草稿权重的 n-gram（prompt-lookup）投机器，用 `--spec-type ngram` 选择，适用于 Qwen 3.5 家族、Gemma 4、GLM 5.x 与 Qwen 3.8 Flash Next。GPT OSS、Mistral 3、Qwen 3 / Qwen 2（含 Bonsai 8B）与 Hunyuan Dense 没有投机路径，因此无法使用它；DeepSeek V4 / V4.1 与 Muse-Glimmer 在没有加载各自草稿器时也无法使用。每个输出 token 都取自主干的一行 logits，并由本次运行自身配置的采样器抽出，因此输出流与普通 decode 产生的完全相同。CLI 与服务端默认关闭（TensorAgent 默认开启）；内嵌草稿头在 CLI 与服务端两端均以 `--spec` 启用，而对以独立 GGUF 发布的草稿器，传入 `--draft-model` 本身即可启用投机。 |
| 张量并行 | Direct `cuda` 后端与 GGML CUDA / Vulkan 后端上的 Megatron-LM 列/行并行 TP（`--tp N` / `TENSORSHARP_TP_DEGREE`，CLI 与服务端均支持）；通过点对点 TCP 的多节点分布式 TP（`--tp-node-id` / `--tp-peers`），采用分层 AllReduce，CUDA P2P 不可用时自动回退到主机中转。覆盖大多数自回归架构（Hunyuan Dense 只用单设备，Bonsai2 的 PRISM 变换需要单设备 GGML 后端因而拒绝 `--tp`，DiffusionGemma、Wan 与 MiniMax-H3 在单设备上加载并警告 `--tp` 被忽略）；GGML 上 Gemma 4 与 Qwen 3.5/3.6 使用 MoE 专家并行与融合的按 rank decode/prefill 计算图。GLM 5.x 默认按层切分，但 GGML GPU 后端上的 `--tp N` 会为 GLM 5.2、GLM-5.3 与 GLM-5.3-Flash 选择仅支持本地单进程的原生 TP（GLM-5.3-Flash 会切 KDA / MLA head 与路由专家隐藏行）；整个 GLM 家族都会在构建模型之前硬拒 `--tp-node-id` / `--tp-peers`，而在 GLM-5.3 上 `--tp N` 只是一个被接受的模式，而非已验证的配置——它会按 rank 复制 MLA 与索引器缓存，占用随之成倍增长、能装下的上下文随之缩短，并且该 checkpoint 上从未跑过 `--tp 1` 以上的配置（`--tp 8` 折算下来每个 rank 需要 41.7 GiB，放不进 46 GB 的卡）。本身不切分权重的架构把同一个 `--tp N` 当作按层切分——每张 GPU 拿一段连续的整层，DeepSeek V4 与 V4.1 即是如此（V4.1 上还可用 `TS_DSV41_TP=N` 打开实验性的 routed-MoE 张量并行，目前实测比按层切分更慢）；Qwen 3.8 Flash Next（`qwen4exp`）上可用 `TS_Q4E_LAYER_SPLIT=20,28` 覆盖自动均衡，遇到无法满足的切分会直接报错而不是静默忽略。启动时会打印实际采用的模式与每张 GPU 的层数/字节分配；两种模式都不支持的架构会在 stderr 上明说，并改用单卡运行。Qwen-Image-2.1 在 `ggml_cuda` / `ggml_vulkan` 上用 `--tp` 只切分扩散 Transformer（GPU 数须能整除它的 32 个注意力头，且仅限单机）；文本编码器、视觉编码器与 VAE 留在 GPU 0 上，Vulkan 上的 TP 实测比单卡更慢。`--redis-url` 为 Responses API 存储提供后端；服务端会接受 Redis KV 层（`--redis-url` 中的 KV 部分、`--paged-kv-redis-url` / `--paged-kv-redis-ttl`）以及 `--paged-kv*` 标志，但只把它们写进日志，因为分层的 RAM/SSD/Redis `PagedKvCacheManager` 只由 CLI 的 `--paged-bench` 使用。 |
| Agent Skills | 技能目录来自 `--skills-dir` / `TS_SKILLS_DIR`；两者都未设置时，来自从工作目录向上直到 Git 根目录之间所有已存在的 `.agents/skills` 目录（由近及远），然后是二进制旁的 `skills` 目录。服务端上该目录同时是 `POST /api/skills` 的运行期安装目录：它总是最先扫描（排在 `.agents/skills` 根目录之前，即使指定了 `--skills-dir` / `TS_SKILLS_DIR` 也会保留），同名技能以它为准；CLI 没有安装目录。在 `/v1/chat/completions`、`/v1/responses`、`/api/chat/ollama`（Ollama）与 `/api/chat`（Web UI）上用 `"skills": [...]` 按请求选中，CLI 上用 `--skill`。支持完整工具闭环的家族（包括 Qwen 3.8 Flash Next，`qwen4exp`）只接收元数据，并通过进程内应答的 `skills_list` / `skills_read` 激活说明；调用方自己的工具仍照常回传。脚本执行（`skills_run`）默认关闭，需传入 `--skills-allow-exec`；`.sh` 脚本用 `bash` 运行。Mistral 3 以及没有可解析工具协议的家族改为内联选中技能正文，且不提供技能 / 代码工具。Playwright 浏览器技能（`TensorAgent/skills/playwright`）通过 `skills_run` 驱动浏览器，而不是屏幕或鼠标工具；它需要 Node.js/npm（宿主已安装的，或在开启联网后由智能体装进会话的 `$HOME/.local/bin`），并显式打开脚本、联网与装包开关，只在 macOS arm64 上验证过；TensorAgent 的 iOS 应用包不包含它，因为在 iOS 上无法运行（[指南](playwright_agent.md)）。 |
| 智能体代码工作 | 可选的 `--code-exec` 在同一个有界“模型→工具”循环里提供 `read_file`、原子 `apply_patch`（修改已有文件一律用它）、用于新建文件的 `write_file`（遇到已存在的路径会拒绝，除非模型传入未声明的 `overwrite: true`）与 `shell`。Web UI / CLI 聊天保留会话工作区；每个 OpenAI / Ollama 请求只在内部修复轮次间保留一个私有工作区，响应后删除。生成文件可作为产物下载。“模型→工具”循环在进程内由同一个已加载模型驱动，而在 CLI 与服务端上，shell 命令与技能脚本以子进程运行（隔离方式见“沙箱与权限”一行）；子智能体（见下一行）是唯一的委派方式，也没有逐命令审批工作流。 |
| 子智能体 | 在 `/v1/chat/completions`、`/v1/responses`、`/api/chat/ollama` 与 Web UI 的 `/api/chat` 上，以及 TensorAgent 中（设置里有“Sub-agents”开关，默认开启；手机页面不显示子智能体面板），默认提供有界、由模型决定的委派；CLI 没有子智能体。凡是会渲染工具声明且有工具解析器的家族都可使用——Mistral 3、Hunyuan Dense 与 DiffusionGemma 除外——不设模型规模门槛。子智能体运行在同一个已加载模型上，默认只读（`explorer` 与 `reviewer` 始终只读；`worker` 只有在传入 `--agents-allow-worker-tools` 时才获得父智能体的可写工具）；同一请求树内的宿主工具调用逐个执行；每个子智能体复制而非共享 KV 状态，并从父智能体已设置检查点的共享前缀恢复。用 `--no-multi-agent`、`TS_NO_MULTI_AGENT` 或请求中的 `"multi_agent": false` 关闭，用 `--agents-max-*` / `--agents-timeout` 设定上限。尚未发布任何延迟、质量或委派率的实测数据。[指南](multi_agent.md)。 |
| 沙箱与权限 | 代码执行和技能脚本默认关闭。macOS 使用 Seatbelt，Linux 需要 `bwrap` 0.12.0+；`required` 模式在无法隔离时拒绝运行。Windows 代码执行必须显式传入 `--code-exec-unconfined`，Windows 技能脚本则需以 `--skills-sandbox preferred` 明确接受仅 job-object 的限制。技能脚本联网、代码联网与宿主代办装包是三个独立开关。 |
| 服务端模型范围 | 通过 `--model` 显式托管单个 GGUF；可通过 `--mmproj` 显式指定投影器；不扫描目录。 |
| 可观测性 | 结构化每轮日志、队列状态，以及 Web UI / Ollama / OpenAI 中的 KV 缓存复用指标。 |

## 让它跑得更快

简要顺序如下：

1. Wan 选择 step-distilled checkpoint。Qwen-Image-2.1 出草图时用 `--width 1024 --height 1024` 或 `--diffusion-steps 25`，或使用 `config/lora/` 中的步数蒸馏 LoRA 插件（4–8 步）。
2. 后端匹配硬件：NVIDIA 用 `ggml_cuda`，Apple Silicon 和 iOS 用 `ggml_metal`，原生 CPU 用 `ggml_cpu`。
3. 调高级开关前，先降低分辨率、帧数或扩散步数。MiniMax-H3 的实用配置是 `--cfg 1.0` 与 4–8 步。
4. 文本任务只有在模型和负载适合时，再尝试投机解码、CPU MoE offload 或 `--tp N`。

测量数据与限制条件见[引擎对比报告](engine_comparison_report.md)、[Apple Silicon 上 ggml_metal 对比 llama.cpp](perf/metal-vs-llama-cpp.md)、[模型卡片](models/README_zh-cn.md)、[功能说明](../FEATURES_zh-cn.md)和[环境变量功能矩阵](env_var_feature_matrix_zh-cn.md)。

## 详细内容

- [快速开始](../README_zh-cn.md#快速开始)——首次运行与后端选择。
- [计算后端](../USAGE_zh-cn.md#计算后端)——能力、构建要求与回退。
- [Agent Skills 与智能体工作](agent_skills.md)——技能、工具、工作区与安全。
- [子智能体](multi_agent.md)——委派工具、角色、限额与 KV 前缀共享。
- [TensorAgent](../TensorAgent/README.md)——iOS 应用架构与验证。
- [开发指南](../DEVELOPMENT_zh-cn.md)——项目分层与原生库构建。
