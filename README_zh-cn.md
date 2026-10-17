# TensorSharp

<p align="center">
  <img src="imgs/banner_1.png" alt="TensorSharp logo" width="320">
</p>

[English](README.md) | [中文](README_zh-cn.md)

**TensorSharp 是在本地运行 GGUF 模型的 .NET 10 推理引擎。** 在 Windows、macOS 或 Linux 上，通过 CLI、浏览器聊天或兼容 Ollama/OpenAI 的 API 使用，也可直接集成到自己的 .NET 应用中。可选择纯托管 C# CPU 内核或原生 CUDA、Metal、Vulkan 后端，具体支持取决于模型。

当前源码涵盖文本与推理、多模态输入、嵌入、图像生成与编辑、带音频的视频、个人知识库，以及带代码工具的 Agent Skills。TensorSharp 也为 iPhone、iPad、Mac 和 Windows 上的本地应用 [TensorAgent](TensorAgent/README.md) 提供引擎。可直接从下文开始，或查看[项目状态](docs/PROJECT_STATUS_zh-cn.md)中的能力与验证限制；LTX 2.5、EmbeddingGemma 2 与原生技能发现需要 `v2026.10.11` 标签之后的源码构建。

## 亮点功能

- **本地模型，多种入口。** 同一引擎用于 CLI、浏览器聊天及兼容 Ollama/OpenAI 的 API，支持纯托管 CPU 与原生加速后端。
- **文本与多模态模型。** 支持稠密与 MoE GGUF 模型、推理、图像 / 音频输入和文档问答。各模型家族的能力见[支持的模型](docs/supported_models_zh-cn.md)。
- **嵌入与媒体生成。** [EmbeddingGemma 2](docs/embeddings_zh-cn.md) 提供文本 / 代码嵌入及可选的图像、音频和视频输入；[Qwen-Image-2.1 与 Turbo](docs/models/qwenimage21_zh-cn.md) 支持图像生成与蒙版编辑；[LTX 2.5 Distilled](docs/models/ltx-2.5_zh-cn.md)（48 kHz）与 [MiniMax-H3](docs/models/minimax-h3_zh-cn.md)（32 kHz）支持视频与立体声音频联合生成。
- **高效服务。** 连续批处理与分页、Radix 前缀共享 KV 缓存默认开启；支持的模型还可使用投机解码与多 GPU 放置。详见[功能说明](FEATURES_zh-cn.md)。
- **Agent Skills 与代码工具。** `TensorSharp.AgentHost` 提供文件、shell 和文档工作流。[17 个随附技能](docs/agent_skills.md)（英文）涵盖研究、文档、TDD、设计访谈与交接；开启联网后，`find-skills` 可搜索、检查并安装更多技能。服务端与 TensorAgent 聊天还支持[有界子智能体委派](docs/multi_agent.md)（英文），子智能体拥有私有工作区并默认只读。
- **个人知识库。** [上传文件或嵌套目录，导入已保存聊天及其附件和产物](docs/knowledge-system_zh-cn.md)。按聊天选择集合，检索带页码 / 行号 / 时间位置的引用片段、预览原件，或让代码智能体打开完整数据集。默认使用本地 BM25，可选 GGUF 嵌入提供语义搜索；兼容的媒体模型可索引图像、音频和采样视频。提取状态与警告显示覆盖边界。
- **生成中追加指令。** CLI、服务端 Web UI 和 TensorAgent 在推理或回答时都可提交新指令；后续回答使用对话、已有部分输出与新指导。仍可单独停止当前生成。
- **有记录的性能对比。** 与 `llama.cpp` 的测试使用相同 GGUF 文件和硬件，结果对应所测的模型、后端及工作负载。详见[性能数据](docs/benchmarks_zh-cn.md)。
- **TensorAgent 应用。** 在手机和桌面进行本地聊天、使用附件与技能并保存工作。详见 [Mac / Windows 安装指南](docs/tensoragent_desktop_zh-cn.md)与[应用验证范围](TensorAgent/README.md)。

## 快速开始

### TensorSharp CLI 与服务端

[Releases 页面](https://github.com/zhongkaifu/TensorSharp/releases)提供自包含的 Windows x64（CPU/CUDA）、Linux x64（CPU/CUDA）与 macOS arm64 CLI / Server 归档。

从源码构建需要完整的 **.NET 10 SDK**（[各平台安装方法](docs/getting_started_zh-cn.md#安装与首次运行)）、`git`、`curl`、[CMake](https://cmake.org/download/) 3.20+，以及所选 GPU 的工具链。然后运行已验证的 [Gemma 4 E4B](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q8_0.gguf) 模型（7.48 GiB）。在 Windows + NVIDIA GPU 上（PowerShell）：

```powershell
git clone https://github.com/zhongkaifu/TensorSharp.git; Set-Location TensorSharp
New-Item -ItemType Directory -Force models | Out-Null
curl.exe -L --fail "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf?download=true" -o models\gemma-4-E4B-it-Q8_0.gguf
'用一句话回答：TensorSharp 是什么？' | Set-Content prompt.txt
$env:TENSORSHARP_GGML_NATIVE_ENABLE_CUDA = 'ON'
dotnet run --project TensorSharp.Cli -c Release -p:TensorSharpSkipMlxNative=true -- --model models\gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --max-tokens 128 --backend ggml_cuda
```

在其他机器上换用对应后端（见[选择后端](#选择后端)）：

- **macOS（Apple Silicon）：** 去掉 CUDA 环境变量，使用 `--backend ggml_metal`。
- **Linux + NVIDIA：** 在 `dotnet run` 前加 `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON`，使用 `--backend ggml_cuda`。
- **AMD / Intel / NVIDIA Vulkan：** 设置 `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON`，使用 `--backend ggml_vulkan`。

将同一模型作为服务托管：浏览器聊天在 <http://localhost:5000>，另有兼容 Ollama 与 OpenAI 的 API。

```bash
dotnet run --project TensorSharp.Server.Host -c Release -p:TensorSharpSkipMlxNative=true -- --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --max-tokens 512
```

> 服务端监听 `0.0.0.0:5000`，没有内置鉴权或 TLS——请置于防火墙之后，或使用带鉴权的 HTTPS 反向代理。

### 选择后端

| 你的硬件 | 后端 |
|---|---|
| Apple Silicon（Mac） | `--backend ggml_metal` |
| Windows / Linux + NVIDIA GPU | `--backend ggml_cuda` |
| Windows / Linux + AMD / Intel / NVIDIA GPU | `--backend ggml_vulkan` |
| 无 GPU | `--backend ggml_cpu`（原生算子），或 `--backend cpu`（纯 C#，无原生依赖） |

其余内容见[快速上手指南](docs/getting_started_zh-cn.md)：各平台的 SDK 安装、多 GPU 与多节点运行、NVIDIA DGX Spark、多模态输入、嵌入服务，以及如何跑得更快。所有参数见 [CLI](USAGE_zh-cn.md#控制台应用) 与 [Server](USAGE_zh-cn.md#web-应用) 参考，两个程序也都可以用 `--help` 打印。

`dotnet build TensorSharp.slnx` 还会构建当前平台可用的 TensorAgent 桌面目标，以及 Apple Silicon 上的 iOS 模拟器目标；所选 SDK 须具备相应 MAUI 工作负载，且原生引擎与 Python 文件已准备好。缺少前置条件时，对应应用目标会被跳过并显示警告；见 [TensorAgent 构建说明](TensorAgent/README.md#build-and-run)。

### TensorAgent 桌面版：下载、安装、聊天

**[下载最新发布](https://github.com/zhongkaifu/TensorSharp/releases/latest)**，展开 **Assets**，选择以 `tensoragent-desktop-` 开头的文件：

| 平台 | 下载与安装 |
|---|---|
| **macOS 14+，Apple Silicon** | `tensoragent-desktop-<version>-osx-arm64.dmg`：打开并把 TensorAgent 拖到“应用程序”。也提供 PKG 安装程序和 ZIP。 |
| **Windows x64** | `tensoragent-desktop-<version>-win-x64-cpu.msi`，兼容 NVIDIA GPU / 驱动时选择 `win-x64-cuda.msi`：安装后从“开始”打开 TensorAgent。也提供 ZIP。 |

应用包含 .NET 运行库与原生引擎；Windows 如缺少 [WebView2 Evergreen Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/)，需另外安装。Python / Node 是技能所需的可选工具。当前内置模型至少需要 **12 GB 系统内存**，权重在应用内另外下载。打开 **☰ → 模型 → 下载 → 使用**，即可输入消息。[桌面版用户指南](docs/tensoragent_desktop_zh-cn.md)包含安装包校验、未签名应用提示、首次配置、附件、技能、更新与故障排查。历史发布可能没有桌面资源，需等待更新后的工作流运行发布；iPhone / iPad 仍使用[源码构建](TensorAgent/README.md#build-and-run)。

## 支持的模型家族一览

- **文本、推理与多模态 LLM：** [DeepSeek V4 Flash](docs/models/deepseek4_zh-cn.md) / [V4.1 Flash](docs/models/deepseek41_zh-cn.md)、[GLM 5.x](docs/models/glm_zh-cn.md)、[Gemma 4](docs/models/gemma4_zh-cn.md)、[Qwen 3.5 / 3.6 / 3.8 27B](docs/models/qwen35_zh-cn.md)、[Qwen 3.8 Flash Next](docs/models/qwen38-flash-next_zh-cn.md)、[Bonsai2](docs/models/bonsai2_zh-cn.md)（Qwen 家族）、[GPT OSS](docs/models/gptoss_zh-cn.md)、[Nemotron-H](docs/models/nemotron_zh-cn.md)、[Mistral 3](docs/models/mistral3_zh-cn.md)、[Hunyuan Dense](docs/models/hunyuan-dense_zh-cn.md) 与 [Muse-Glimmer](docs/models/muse-glimmer_zh-cn.md)。
- **文本扩散：** [DiffusionGemma](docs/models/diffusiongemma_zh-cn.md)，包含 `/v1/systemone` 上的 [Jev 类型化判定](docs/models/jev_zh-cn.md)（支持文本、图像、上传文档、抽样视频帧，以及通过已配置 ASR 配套服务得到的语音转录）。
- **图像生成/编辑与视频生成：** [Qwen-Image-2.1](docs/models/qwenimage21_zh-cn.md)、[LTX 2.5 Distilled（视频 + 48 kHz 立体声音频）](docs/models/ltx-2.5_zh-cn.md)、[MiniMax-H3（视频 + 立体声音频）](docs/models/minimax-h3_zh-cn.md) 与 [Wan 2.1 / 2.2](docs/models/wan_zh-cn.md)。
- **嵌入：** [EmbeddingGemma 2](docs/embeddings_zh-cn.md)，包括文本 / 代码及可选的图像、音频、视频输入；以及 Snowflake Arctic Embed L v2.0、all-MiniLM-L6-v2 等 BERT / XLM-R 编码器。

各模型的后端、模态、功能支持与验证覆盖范围不同，详见[支持的模型](docs/supported_models_zh-cn.md)各表、[模型卡片](docs/models/README_zh-cn.md)及[嵌入指南](docs/embeddings_zh-cn.md)。

TensorAgent 当前目录包含 39 个条目，按可搜索的模型家族分组，并支持内存档位与已下载过滤；其中包括实验性 48 GB 内存档位的 LTX 2.5 Distilled、4 位和 8 位 Qwen-Image-2.1 Turbo、十四款图像 LoRA 插件，以及两个 MiniMax-H3 Turbo 提速插件。扩展目录包含实验性条目，其内存档位与应用 / 设备行为尚未全部实测。Qwen3.8 Flash Next 使用 SSD 支持的权重路径；Q2 包已在 48 GB Mac 上测量，IQ1_M 另有 Windows 目录宿主验证。多 GPU 的 `--layer-split` 与 `--tp` 是独立选项，支持范围和性能取决于架构与量化格式。验证范围见 [TensorAgent 说明](TensorAgent/README.md#what-has-not-been-verified)（英文）与模型卡片。更新后的 Release Binaries 工作流会在新发布中提供桌面安装包，历史发布可能不包含这些资源。

## 实际运行效果

同一个引擎，四种用法，每张都是真实运行的原样截图。

<table>
  <tr>
    <td align="center" width="50%"><img src="website/assets/screenshots/tensorsharp-cli.png" alt="终端中的 TensorSharp.Cli：Gemma 4 E4B 的交互式聊天，读取本 README 并回答相关问题" width="250"><br><b>TensorSharp.Cli</b><br>在终端里运行模型</td>
    <td align="center" width="50%"><img src="website/assets/screenshots/tensorsharp-webui.png" alt="TensorSharp Web UI：Qwen3.8 27B 编写并运行 Python 脚本比较两种房贷" width="400"><br><b>TensorSharp.Server.Host</b><br>Web UI 聊天与兼容 Ollama/OpenAI 的 API</td>
  </tr>
  <tr>
    <td align="center"><img src="website/assets/screenshots/tensoragent-iphone.png" alt="iPhone 17 Pro 模拟器中的 TensorAgent：Gemma 4 E2B 使用 ggml_cpu，并在应用内运行 Python 脚本换算食谱" width="140"><br><b>TensorAgent · iPhone 模拟器</b><br>CPU 推理与应用内 Python；此处为模拟器截图</td>
    <td align="center"><img src="website/assets/screenshots/tensoragent-mac.png" alt="Mac 上的 TensorAgent：Qwen-Image 2.1 按文字指令将 TensorSharp 横幅背景改为蓝色星空，可比较原图并再次编辑" width="400"><br><b>Mac 上的 TensorAgent</b><br>Mac 应用中保存的 Qwen-Image 2.1 编辑结果</td>
  </tr>
</table>

每次运行具体做了什么，见[实际运行截图](docs/showcase_zh-cn.md)。

## 性能数据

TensorSharp 与 `llama.cpp` 在同一块 NVIDIA RTX 3080 Laptop GPU（16 GB）上运行相同的 GGUF 文件，两个引擎均分别在 GGML CUDA 与 Vulkan 构建上测量。下表为在相同后端上 TensorSharp 相对 llama.cpp 的加速比（几何平均、单流、贪心采样、关闭 MTP）；大于 1.0× 表示 TensorSharp 更快。

| 模型 | 后端 | decode | prefill | TTFT |
|---|---|---:|---:|---:|
| Gemma 4 E4B it（Q8_0，dense 多模态） | CUDA | 1.02× | **1.28×** | **1.27×** |
| Gemma 4 E4B it（Q8_0，dense 多模态） | Vulkan | 1.00× | 1.05× | 1.03× |
| Gemma 4 12B it（QAT UD-Q4_K_XL，dense） | CUDA | 1.04× | **1.17×** | **1.16×** |
| Gemma 4 12B it（QAT UD-Q4_K_XL，dense） | Vulkan | **1.21×** | 1.04× | 1.03× |
| Qwen 3.6 35B-A3B（UD-IQ2_XXS，MoE） | CUDA | 0.98× | **1.28×** | **1.27×** |
| Qwen 3.6 35B-A3B（UD-IQ2_XXS，MoE） | Vulkan | 0.87× | 1.04× | 1.03× |
| Qwen 3.6 27B（UD-IQ2_XXS，dense） | CUDA | **1.07×** | 0.96× | 0.95× |
| Qwen 3.6 27B（UD-IQ2_XXS，dense） | Vulkan | 1.02× | 0.85× | 0.84× |

这些数字的含义、如何复现，以及放不进这块 GPU 的大模型的正面对比，见[性能数据](docs/benchmarks_zh-cn.md)。

## 文档

初次使用？上面几节足以让你跑起来。其余均为详细参考：

| 文档 | 内容 |
|---|---|
| [TensorAgent 桌面版用户指南](docs/tensoragent_desktop_zh-cn.md) | Mac / Windows 下载、DMG / PKG / MSI / ZIP 安装、模型配置、首次聊天、附件、技能、更新与故障排查 |
| [TensorSharp 与 TensorAgent 书籍指南](docs/BOOK_zh-cn.md) | 《Building LLM Inference Engines and Agentic Runtimes from Scratch》与《From Tensors to Tokens》：书籍介绍、Amazon 链接与仓库伴读路线 |
| [快速上手指南](docs/getting_started_zh-cn.md) | 完整的首次运行指南：各平台的 .NET SDK、所有后端、多 GPU 与多节点运行、NVIDIA DGX Spark、嵌入服务、后端选择与提速 |
| [支持的模型](docs/supported_models_zh-cn.md) | 已实现的模型家族与实际验证范围：示例 GGUF、模态、思维链、工具与投机解码 |
| [性能数据](docs/benchmarks_zh-cn.md) | 在相同 GPU 与相同文件上对比 llama.cpp，以及更大模型的正面对比 |
| [实际运行截图](docs/showcase_zh-cn.md) | CLI、Web UI，以及 iPhone 与 Mac 上的 TensorAgent 的实际运行截图，及每次运行做了什么 |
| [模型下载](MODEL_DOWNLOADS_zh-cn.md) | 各模型 `huggingface-cli` 下载 + 运行速查（量化档位、投影器、伴随文件） |
| [使用方法](USAGE_zh-cn.md) | 完整 CLI 参考（选项、交互式 REPL、JSONL 批处理）、服务端托管、日志、HTTP API 示例、后端与环境变量矩阵 |
| [功能特性](FEATURES_zh-cn.md) | 连续批处理、投机解码、工具调用、思维链、多模态、MoE、KV 编解码等深入说明 |
| [配置文件](config/README.md) | 把参数写进可复用的 JSON 文件，支持 `${变量}` 与模型自动下载 |
| [开发](DEVELOPMENT_zh-cn.md) | 前置要求、构建原生 GGML/MLX 库、仓库结构、包分层、内部架构与测试工具 |
| [按模型架构卡片](docs/models/README_zh-cn.md) | 各架构端到端文档（前向图、组件、参数、prefill/decode 优化） |
| [分页注意力 & 连续批处理](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md) | vLLM 风格的分页 KV 缓存、前缀共享与迭代级调度器 |
| [Agent Skills 与智能体工作](docs/agent_skills.md)（英文） | `SKILL.md` 格式、渐进式披露与其预算、进程内工具循环、沙箱化代码执行、工作区与产物、路径 / ZIP / 执行安全模型，以及 HTTP 与 C# 两套接口 |
| [多智能体](docs/multi_agent.md)（英文） | 自动任务委派、隔离的子上下文、并发与权限限制、服务端控制项，以及可复现的评测 |
| [个人知识库](docs/knowledge-system_zh-cn.md) | 集合、目录和已保存聊天导入、文档 / PDF / 媒体提取、本地检索、引用预览、数据集工具、API 与验证边界 |
| [浏览器自动化技能（Playwright）](docs/playwright_agent.md)（英文） | 运行内置的 `playwright` 技能（经 `skills_run` 通过 `@playwright/cli` 驱动浏览器）：所需参数、macOS 上的 Chromium 沙箱配置、账号交接，以及 TensorAgent 桌面托管（不支持 iOS） |
| [投机解码](docs/speculative_decoding.md)（英文） | 三层设计（模型适配层 / 算法 / 草稿权重）、已内置的 `auto` / `draft-head` / `block` / `ngram` 四种算法，以及新增一种算法需要写什么 |
| [环境变量功能矩阵](docs/env_var_feature_matrix_zh-cn.md) | 哪些高影响运行时开关影响哪些模型、后端与提示类型 |
| [引擎对比报告](docs/engine_comparison_report.md) | TensorSharp 对比 llama.cpp 的完整逐场景表格 |
| [ggml_metal 对比 llama.cpp](docs/perf/metal-vs-llama-cpp.md) | Apple Silicon 上 prefill / decode 的正面对比，找到的四处计算图构建差距，以及每一处的实际收益 |
| [测试 / 基准矩阵运行器](TensorSharp.TestMatrix/README_zh-cn.md) | 扫描 model × backend × feature × env-var 组合并生成回归报告 |
| [服务端 API 示例](TensorSharp.Server.Host/API_EXAMPLES_zh-cn.md) | 完整的 curl 与 Python 示例 |

## 当前状态

仍在活跃开发中，源码树领先于已发布的包。

| 范围 | 当前情况 |
|---|---|
| 模型 | 十余个自回归家族，另有文本扩散、图像生成与编辑，以及带音频的视频生成。见[支持的模型](docs/supported_models_zh-cn.md)。 |
| 推理宿主 | CLI、Web UI、兼容 Ollama 与 OpenAI 的 API，以及 iPhone、iPad、Mac 和 Windows 上的 TensorAgent 应用。发布工作流打包 Mac / Windows 桌面版，历史发布可能没有对应资源；iPhone / iPad 使用源码构建。 |
| 后端 | 纯 C# CPU、Direct CUDA/cuBLAS、MLX Metal，以及 GGML CPU/Metal/CUDA/Vulkan，各架构另有例外。 |
| 服务能力 | 带共享前缀缓存的连续批处理、投机解码、张量并行、结构化输出与工具调用。 |
| 智能体能力 | Agent Skills、沙箱化的文件与 shell 工具，以及有界的子智能体。见 [Agent Skills](docs/agent_skills.md)（英文）与[多智能体](docs/multi_agent.md)（英文）。 |
| 知识库 | 本地文档、数据集与媒体集合；TensorAgent 支持目录和已保存聊天导入、每聊天作用域、词法 / 可选语义检索、来源预览与完整数据集智能体工具。媒体提取与 OCR 有明确的覆盖边界。见[个人知识库](docs/knowledge-system_zh-cn.md)。 |
| TensorAgent | 38 项目录条目，支持家族搜索与过滤；保存聊天与产物、个人知识库、生成中追加指令、蒙版编辑、8 步 Turbo、十四款图像 LoRA 与两个 MiniMax-H3 Turbo 插件、八种界面语言和持久保存的文本轮次统计。扩展目录条目仍属实验性；iOS 媒体生成及 Windows 图像 / 音频 / 视频生成仍未验证。 |

逐项细节（哪个架构跑在哪个后端上、各家族分别支持哪些特性，以及已知限制）见[状态矩阵](docs/PROJECT_STATUS_zh-cn.md#状态矩阵)。

## 作者

Zhongkai Fu

## 许可证

详见 [LICENSE](LICENSE)。

## 配合书籍学习

| Qwen 推理与智能体运行时 | Gemma 4 与多模态推理 |
|---|---|
| <a href="https://www.amazon.com/dp/B0HJQ4VQ31"><img src="website/assets/building-llm-inference-engines-cover.jpg" alt="Building LLM Inference Engines and Agentic Runtimes from Scratch: Qwen Dense and MoE Models with TensorSharp and TensorAgent" width="190"></a> | <a href="https://www.amazon.com/dp/B0H9P44QZZ"><img src="website/assets/from-tensors-to-tokens-cover.jpg" alt="From Tensors to Tokens: Building a Multimodal LLM Inference Engine from Scratch with TensorSharp and Gemma 4 E4B" width="190"></a> |
| **[Building LLM Inference Engines and Agentic Runtimes from Scratch: Qwen Dense and MoE Models with TensorSharp and TensorAgent](https://www.amazon.com/dp/B0HJQ4VQ31)** | **[From Tensors to Tokens: Building a Multimodal LLM Inference Engine from Scratch with TensorSharp and Gemma 4 E4B](https://www.amazon.com/dp/B0H9P44QZZ)** |
| 使用 C# 构建 Qwen 稠密/MoE 推理与受控智能体工作流。从张量、分词、注意力、专家路由、量化和缓存，逐步走向 GPU 加速、多模态执行、工具、技能、沙箱代码执行，以及 TensorSharp 和 TensorAgent 的桌面与移动端部署。 | 以 Gemma 4 E4B 为例，用 C#/.NET 构建多模态推理引擎。从张量、GGUF 模型加载、量化与分词，走向文本、图像、视频和音频执行，并结合 TensorSharp 源码理解正确性检查与服务优化。 |
| **[在 Amazon 购买](https://www.amazon.com/dp/B0HJQ4VQ31)** | **[在 Amazon 购买](https://www.amazon.com/dp/B0H9P44QZZ)** |

**[查看两本书的介绍与仓库伴读路线](docs/BOOK_zh-cn.md)**

<p align="center"><a href="https://buymeacoffee.com/zhongkaifu"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>TensorSharp/TensorAgent is free. If you like it, a coffee keeps the work on it going.</sub></p>
