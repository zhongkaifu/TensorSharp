/* 中文搜索索引（TensorSharp 维基）。每条：
   t = 标题, p = 页面标签, u = url(+锚点，英文形式，app.js 会本地化为 _zh-cn), s = 摘要, k = 额外关键词。 */
window.SEARCH_INDEX_ZH = [
  { t: "TensorSharp 是什么", p: "首页", u: "index.html", s: "面向 GGUF 模型的原生 .NET 大模型推理引擎 —— CLI、Web 服务器，以及兼容 Ollama/OpenAI 的 API。", k: "intro 简介 概览 关于 llm gguf csharp dotnet" },
  { t: "约 30 秒快速开始", p: "首页", u: "index.html#quickstart", s: "克隆、构建、下载模型，并流式输出你的第一个回复。", k: "begin 开始 教程 第一次 运行 hello" },
  { t: "TensorSharp 适合谁", p: "首页", u: "index.html#audience", s: "面向开发者、管理者、CTO、学生与采用本地大模型推理的企业的价值。", k: "audience 受众 商业价值 高管 销售 市场 学生" },

  { t: "概览与架构", p: "概览", u: "overview.html#architecture", s: "分层系统：Core 张量、Runtime、Models、后端、Server、CLI。", k: "design 设计 分层 架构 工作原理" },
  { t: "项目结构", p: "概览", u: "overview.html#structure", s: "仓库布局与每个项目/包的角色。", k: "folders 目录 模块" },
  { t: "当前状态与能力", p: "概览", u: "overview.html#status", s: "模型家族、推理宿主、后端、多模态、批处理与可观测性。", k: "support 支持 矩阵 成熟度" },
  { t: "商业价值", p: "概览", u: "overview.html#business", s: "组织为何运行 TensorSharp：隐私、成本、掌控、私有部署、无按 token 计费。", k: "cost roi 隐私 私有部署 manager cto ceo" },

  { t: "功能目录", p: "功能特性", u: "features.html#catalog", s: "多架构、多模态、思考、工具调用、量化、连续批处理、推测解码。", k: "capabilities 能力 列表 全部" },

  { t: "快速开始", p: "快速上手", u: "getting-started.html", s: "环境要求、构建、下载模型，并运行 CLI 或服务器。", k: "install 安装 setup 开始" },
  { t: "环境要求", p: "快速上手", u: "getting-started.html#prerequisites", s: ".NET 10 SDK、git、CMake，以及可选的 CUDA / Metal 工具链。", k: "requirements dotnet 10 cmake cuda xcode 依赖" },
  { t: "构建解决方案", p: "快速上手", u: "getting-started.html#build", s: "dotnet build TensorSharp.slnx -c Release。原生 GGML 库在首次构建时编译。", k: "compile 编译 make native ggml" },
  { t: "下载 GGUF 模型", p: "快速上手", u: "getting-started.html#download", s: "从 Hugging Face 获取模型，例如 gemma-4-E4B-it Q8_0。", k: "huggingface 权重 gguf 量化" },
  { t: "首次运行", p: "快速上手", u: "getting-started.html#first-run", s: "一次性生成、交互式聊天，或启动服务器。", k: "hello world 示例 运行" },

  { t: "选择后端", p: "后端", u: "backends.html#table", s: "选择 ggml_metal（Mac）、ggml_cuda（NVIDIA）或 cpu/ggml_cpu（可移植）。", k: "choose 选择 硬件 gpu 哪个后端" },
  { t: "GGML CUDA 后端", p: "后端", u: "backends.html#ggml", s: "--backend ggml_cuda —— Windows/Linux 上测试最充分的 NVIDIA 路径。", k: "nvidia gpu cuda windows linux" },
  { t: "GGML Metal 后端", p: "后端", u: "backends.html#ggml", s: "--backend ggml_metal —— macOS / Apple Silicon 默认。", k: "apple silicon mac metal 苹果" },
  { t: "MLX 后端", p: "后端", u: "backends.html#mlx", s: "--backend mlx —— 基于 mlx-c 的 Apple Silicon GPU 路径。", k: "apple metal mlx-c 苹果" },
  { t: "直接 CUDA 后端", p: "后端", u: "backends.html#cuda", s: "--backend cuda —— 直接 CUDA Driver API + cuBLAS + PTX 内核（纯 C#）。", k: "nvidia ptx cublas 实验" },
  { t: "纯 C# CPU 后端", p: "后端", u: "backends.html#cpu", s: "--backend cpu —— 可移植、无原生依赖；ggml_cpu 提供原生 CPU 内核。", k: "可移植 调试 无 gpu" },
  { t: "构建原生 GGML / MLX 库", p: "后端", u: "backends.html#native-build", s: "build-windows.ps1 / build-linux.sh / build-macos.sh 与 CUDA 架构检测。", k: "compile 编译 native cmake cuda arch" },

  { t: "支持的模型", p: "模型", u: "models.html#table", s: "Gemma 3/4、Qwen 3 / 3.5 / 3.6、GPT OSS、Nemotron-H、Mistral 3、DiffusionGemma。", k: "architectures 架构 家族 gemma qwen gptoss nemotron mistral" },
  { t: "模型下载（GGUF）", p: "模型", u: "models.html#downloads", s: "每个受支持架构的 Hugging Face 链接。", k: "huggingface 权重 下载" },
  { t: "多模态：图像、视频、音频", p: "模型", u: "models.html#multimodal", s: "Gemma 4 支持图像/视频/音频；Gemma 3、Qwen 3.5-family、Mistral 3、Nemotron-H Omni 支持图像。", k: "vision 视觉 图像 音频 视频 mmproj 投影器" },
  { t: "思考 / 推理模式", p: "模型", u: "models.html#thinking", s: "Qwen、Gemma 4、GPT OSS、Nemotron-H 带 think 标签的结构化思维链。", k: "reasoning 推理 思维链 think cot" },
  { t: "工具调用 / 函数调用", p: "模型", u: "models.html#toolcalling", s: "模型调用用户自定义工具；跨三种 API 风格的多轮。", k: "functions 函数 工具 agent 智能体" },

  { t: "CLI 示例", p: "CLI", u: "cli.html#examples", s: "文本、图像、视频、音频、思考、工具、批处理 JSONL、基准测试。", k: "command line 命令行 控制台 示例" },
  { t: "CLI 选项参考", p: "CLI", u: "cli.html#options", s: "--model、--input、--backend、--max-tokens、采样参数等。", k: "flags 参数 选项" },
  { t: "交互式 REPL 命令", p: "CLI", u: "cli.html#repl", s: "斜杠命令：/reset、/system、/think、/model、/backend、/image、/temp。", k: "chat repl 交互 斜杠命令" },
  { t: "批处理（JSONL）", p: "CLI", u: "cli.html#jsonl", s: "用 --input-jsonl 从 JSONL 文件运行大量提示词。", k: "batch 批处理 jsonl 多轮" },
  { t: "DiffusionGemma 生成", p: "CLI", u: "cli.html#diffusion", s: "--diffusion-steps、--diffusion-seed、--diffusion-blocks 文本扩散。", k: "diffusion 扩散 去噪 entropybound" },

  { t: "启动服务器", p: "服务器", u: "server.html#start", s: "./TensorSharp.Server --model model.gguf --backend ggml_metal —— 提供 http://localhost:5000。", k: "web 服务器 托管 运行 端口" },
  { t: "服务器选项", p: "服务器", u: "server.html#options", s: "--model、--mmproj、--backend、--max-tokens、默认采样、批处理参数。", k: "flags 参数 选项" },
  { t: "Web UI 功能", p: "服务器", u: "server.html#webui", s: "浏览器聊天：多轮、上传、思考开关、工具调用、流式、消息编辑。", k: "browser 浏览器 聊天机器人 ui 前端" },
  { t: "服务器环境变量", p: "服务器", u: "server.html#env", s: "BACKEND、MAX_TOKENS、PORT、调度器与 MLX 调参。", k: "env 环境变量 配置 TS_SCHED" },
  { t: "连续批处理调参", p: "服务器", u: "server.html#tunables", s: "用于分页 KV 块、运行序列、prefill 分块的 TS_SCHED_* 旋钮。", k: "scheduler 调度器 分页 kv 调优" },

  { t: "兼容 Ollama 的 API", p: "HTTP API", u: "http-api.html#ollama", s: "/api/generate、/api/chat/ollama、/api/tags、/api/show，附 curl 示例。", k: "ollama rest 端点 curl generate chat" },
  { t: "兼容 OpenAI 的 API", p: "HTTP API", u: "http-api.html#openai", s: "/v1/chat/completions 与 /v1/models —— OpenAI 客户端即插即用。", k: "openai chat completions v1 兼容" },
  { t: "Web UI SSE 协议", p: "HTTP API", u: "http-api.html#webui-sse", s: "/api/chat 服务器发送事件、会话、KV 复用字段。", k: "sse 流式 事件 会话" },
  { t: "结构化输出（JSON schema）", p: "HTTP API", u: "http-api.html#structured", s: "response_format text / json_object / json_schema，带严格校验。", k: "json 模式 结构化 schema response_format" },
  { t: "通过 HTTP 进行工具调用", p: "HTTP API", u: "http-api.html#tools", s: "发送 tools 数组；从响应解析结构化 tool_calls。", k: "functions 函数 工具 api" },
  { t: "采样参数", p: "HTTP API", u: "http-api.html#sampling", s: "temperature、top_k、top_p、min_p、惩罚、seed、stop、num_predict/max_tokens。", k: "options 温度 top_p seed stop 采样" },
  { t: "Python 客户端示例", p: "HTTP API", u: "http-api.html#python", s: "requests 与 openai SDK，流式与结构化输出。", k: "python requests openai sdk 客户端" },

  { t: "从 C# 使用 TensorSharp", p: "C# 库", u: "code-api.html#quickstart", s: "ModelBase.Create、Tokenizer.Encode、Forward、Sample —— 代码中的解码循环。", k: "library 库 代码 嵌入 nuget dotnet api" },
  { t: "NuGet 包", p: "C# 库", u: "code-api.html#packages", s: "TensorSharp.Core、.Runtime、.Models、.Backends.*、.Server、.Cli。", k: "nuget 包 依赖 命名空间" },
  { t: "SamplingConfig", p: "C# 库", u: "code-api.html#sampling", s: "Temperature、TopK、TopP、MinP、惩罚、Seed、StopSequences、MaxTokens。", k: "采样 配置 类 属性" },
  { t: "公共命名空间", p: "C# 库", u: "code-api.html#namespaces", s: "TensorSharp、TensorSharp.Runtime、TensorSharp.Models 与后端命名空间。", k: "namespace 命名空间 类型 api" },

  { t: "连续批处理与分页 KV 缓存", p: "高级", u: "advanced.html#continuous-batching", s: "vLLM 式分页 KV 池、块哈希前缀共享、迭代级调度器。", k: "vllm 分页注意力 调度器 批处理" },
  { t: "分页注意力", p: "高级", u: "advanced.html#paged-kv", s: "原生 TSGgml_PagedAttentionForward 在 Metal/CUDA 上驱动 ggml_flash_attn_ext。", k: "flash 闪存注意力 kv 块" },
  { t: "MTP / NextN 推测解码", p: "高级", u: "advanced.html#mtp", s: "草稿头提出 token；主干一次批量前向验证。--mtp-spec。", k: "speculative 推测 解码 草稿 eagle nextn qwen gemma" },
  { t: "性能优化", p: "高级", u: "advanced.html#perf", s: "融合 GPU decode/prefill、原生量化计算、批量 MoE、KV 前缀复用。", k: "fused 融合 内核 速度 优化 gpu" },
  { t: "内存优化", p: "高级", u: "advanced.html#memory", s: "零拷贝 mmap 权重、最佳匹配池、SSD KV 溢出、KV 编解码器。", k: "memory 内存 mmap 占用 ram" },
  { t: "DiffusionGemma 文本扩散", p: "高级", u: "advanced.html#diffusion", s: "在 Gemma-4 MoE 主干上的块式 EntropyBound 去噪。", k: "diffusion 扩散 去噪 文本生成" },

  { t: "同台对比 vs llama.cpp", p: "基准测试", u: "benchmarks.html#head-to-head", s: "纯 .NET 的 TensorSharp 在每个模型上的 prefill 都快于 llama.cpp（几何平均 1.18×–1.88×），相同 GGUF + GPU 下 decode 持平：26B-A4B MoE 领跑，达 1.88× prefill / 1.69× TTFT；结构化输出 prefill 5.89×；多轮最高 1.93×。", k: "llama.cpp 对比 更快 加速 几何平均 moe prefill ttft 多轮 json 结构化输出 decode 持平 vs versus" },
  { t: "基准测试", p: "基准测试", u: "benchmarks.html#head-to-head", s: "对比 llama.cpp 的同台评测与跨引擎推理矩阵。", k: "performance 性能 数字 吞吐 每秒 token" },
  { t: "跨引擎矩阵", p: "基准测试", u: "benchmarks.html#cross-engine", s: "在相同 GGUF 文件上对比 TensorSharp vs llama.cpp 与 Ollama。", k: "llama.cpp ollama 对比" },
  { t: "测试", p: "基准测试", u: "benchmarks.html#testing", s: "xUnit 单元测试、服务器集成测试与推理矩阵运行器。", k: "tests 测试 xunit 集成 ci" },

  { t: "CLI 参数参考", p: "API 参考", u: "api-reference.html#cli-flags", s: "TensorSharp.Cli 命令行选项完整表。", k: "reference 参考 参数 cli 选项 表" },
  { t: "服务器参数参考", p: "API 参考", u: "api-reference.html#server-flags", s: "TensorSharp.Server 命令行选项完整表。", k: "reference 参考 参数 服务器 选项 表" },
  { t: "环境变量", p: "API 参考", u: "api-reference.html#env-vars", s: "运行时、调度器、MTP、MLX 与扩散环境变量。", k: "env 环境变量 TS_ 参考" },
  { t: "HTTP 端点", p: "API 参考", u: "api-reference.html#endpoints", s: "Ollama、OpenAI、Web UI 与工具端点集于一表。", k: "rest 端点 路由 api 参考" },
  { t: "采样参数参考", p: "API 参考", u: "api-reference.html#sampling-params", s: "Ollama 风格与 OpenAI 风格的采样字段与默认值。", k: "采样 参考 温度 默认" },
  { t: "C# 公共 API", p: "API 参考", u: "api-reference.html#csharp", s: "ModelBase、SamplingConfig、ITokenizer、BackendType 与关键接口。", k: "csharp 类 接口 api 参考" },
  { t: "REPL 命令参考", p: "API 参考", u: "api-reference.html#repl-commands", s: "按类别分组的全部交互式斜杠命令。", k: "repl 斜杠命令 参考" },

  { t: "术语表", p: "术语表", u: "glossary.html#terms", s: "GGUF、量化、token、KV 缓存、MoE、prefill、decode、TTFT 解释。", k: "definitions 定义 术语 初学者 概念" },
  { t: "FAQ", p: "术语表", u: "glossary.html#faq", s: "常见问题：选哪个模型、哪个后端、是否需要 GPU、隐私、许可证。", k: "questions 问题 帮助 故障排查" },
  { t: "GGUF 是什么？", p: "术语表", u: "glossary.html#terms", s: "TensorSharp 加载的单文件模型格式，含量化权重与元数据。", k: "gguf 格式 量化" },
  { t: "量化是什么？", p: "术语表", u: "glossary.html#terms", s: "把权重压缩到更少位数（Q4_K_M、Q8_0），以适配内存并更快运行。", k: "量化 q4 q8 位" },
  { t: "KV 缓存是什么？", p: "术语表", u: "glossary.html#terms", s: "存储的注意力键/值，使多轮与长上下文更高效。", k: "kv 缓存 注意力 复用" }
];
