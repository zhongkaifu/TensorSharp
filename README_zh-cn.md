# TensorSharp

<p align="center">
  <img src="imgs/banner_1.png" alt="TensorSharp logo" width="320">
</p>

[English](README.md) | [中文](README_zh-cn.md)

一个用于在本地运行 GGUF 语言模型的 C# 推理引擎，覆盖自回归 LLM 与 DiffusionGemma 风格的文本扩散模型。TensorSharp 提供控制台应用、基于 Web 的聊天界面，以及兼容 Ollama/OpenAI 的 HTTP API 以便程序化调用。

## 快速开始

下载模型后，大约 30 秒即可看到流式输出。

**1. 前置要求** —— [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)、`git`，以及（可选）GPU 工具链：NVIDIA → CUDA Toolkit 12.x；Apple Silicon → Xcode 命令行工具（Metal 已内置）。完整列表见 [前置要求](#前置要求)。

**2. 克隆并构建** —— 首次构建会自动编译原生 GGML 库。

```bash
git clone https://github.com/zhongkaifu/TensorSharp.git
cd TensorSharp
dotnet build TensorSharp.slnx -c Release
```

**3. 下载模型** —— 推荐从体积小、测试充分的 Gemma-4-E4B（Q8_0）开始：[ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)。更多选择见 [已验证模型](#已验证模型)。

**4. 运行** —— 按你的硬件选择 `--backend`（参见 [选择后端](#选择后端)）：

```bash
# 单次生成
echo "用一句话解释 Mixture-of-Experts。" > prompt.txt

./TensorSharp.Cli --model gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --backend ggml_metal   # macOS
./TensorSharp.Cli --model gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --backend ggml_cuda    # Windows/Linux + NVIDIA
./TensorSharp.Cli --model gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --backend cpu          # 可移植 / 调试

# 交互式聊天（REPL）
./TensorSharp.Cli --model gemma-4-E4B-it-Q8_0.gguf -i --backend ggml_metal
```

想要浏览器界面和 HTTP API？改为启动服务端：

```bash
./TensorSharp.Server --model gemma-4-E4B-it-Q8_0.gguf --backend ggml_metal
# 打开 http://localhost:5000 —— 同时提供 Ollama 与 OpenAI 兼容端点
```

构建完成后，CLI 可执行文件位于 `TensorSharp.Cli/bin/...`，服务端位于 `TensorSharp.Server/bin/...`。完整参数：[CLI 用法](#控制台应用) · [服务端用法](#web-应用)。

## 选择后端

不确定用哪个后端？从这里开始。每个后端对尚未实现的算子都会回退到 CPU，因此所有后端的输出都正确。

| 你的硬件 | 推荐后端 | 标志 | 说明 |
|---|---|---|---|
| **Apple Silicon（Mac）** | GGML Metal | `--backend ggml_metal` | macOS 默认。`--backend mlx` 是另一条 Apple Silicon GPU 路径。 |
| **Windows / Linux + NVIDIA GPU** | GGML CUDA | `--backend ggml_cuda` | 测试最充分的 NVIDIA 路径。`--backend cuda` 是用于实验的 Direct PTX/cuBLAS 后端。 |
| **无 GPU / 可移植 / 调试** | 纯 C# CPU | `--backend cpu` | 无原生依赖。需要更快的 CPU 推理可用 `--backend ggml_cpu`（原生算子）。 |

每个后端的完整说明及其加速范围见 [计算后端](#计算后端)。

## 已验证模型

以下架构均已实现，并由测试 / 基准矩阵覆盖。请选择适配你硬件的量化（如低内存用 Q4_K_M、更高质量用 Q8_0）。更多尺寸与多模态投影器文件见下方「模型下载」章节，或参阅 [按模型架构卡片](docs/models/README_zh-cn.md)。

| 家族 | 示例模型（GGUF） | 图像 / 视频 / 音频 | 思维链 | 工具 | 卡片 |
|---|---|---|---|---|---|
| Gemma 4 | [gemma-4-E4B-it](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)（另有 31B、26B-A4B MoE） | ✅ / ✅ / ✅ | ✅ | ✅ | [gemma4](docs/models/gemma4_zh-cn.md) |
| Qwen 3.5 / 3.6 | [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)（另有 35B-A3B MoE） | ✅ / — / — | ✅ | ✅ | [qwen35](docs/models/qwen35_zh-cn.md) |
| Qwen 3 | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-GGUF) | — / — / — | ✅ | ✅ | [qwen3](docs/models/qwen3_zh-cn.md) |
| GPT OSS | [gpt-oss-20b](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF)（MoE） | — / — / — | ✅ | ✅ | [gptoss](docs/models/gptoss_zh-cn.md) |
| Nemotron-H | [Nemotron-H-8B](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF)（另有 47B、Omni） | ✅（Omni） / — / — | ✅ | ✅ | [nemotron](docs/models/nemotron_zh-cn.md) |
| Mistral 3 | [Mistral-Small-3.1-24B](https://huggingface.co/bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF) | ✅ / — / — | — | — | [mistral3](docs/models/mistral3_zh-cn.md) |
| Gemma 3 | [gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf) | ✅ / — / — | — | — | [gemma3](docs/models/gemma3_zh-cn.md) |
| DiffusionGemma | diffusion-gemma GGUF | — / — / — | — | — | [diffusiongemma](docs/models/diffusiongemma_zh-cn.md) |

## 亮点功能

- **连续批处理 & 分页 KV 缓存** —— vLLM 风格的分页 KV 池，支持基于内容哈希的前缀共享与迭代级调度器，服务端默认启用。→ [深入文档](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)
- **DiffusionGemma 文本扩散** —— 基于 Gemma-4 派生 MoE backbone 的分块 EntropyBound 去噪生成，提供 CLI 扩散参数与 Web UI 实时去噪预览。→ [DiffusionGemma 卡片](docs/models/diffusiongemma_zh-cn.md)
- **多模态** —— 图像 / 视频 / 音频输入（Gemma 4）；图像输入（Gemma 3、Qwen 3.5-family、Mistral 3、Nemotron-H Omni）。→ [多模态支持](#多模态支持)
- **工具调用 / 函数调用** —— 三种 API 风格均支持多轮工具调用，输出解析与架构无关。→ [工具调用](#工具调用--函数调用)
- **思维链 / 推理模式** —— Qwen 3、Qwen 3.5/3.6-family、Gemma 4、GPT OSS、Nemotron-H 支持结构化思维链。→ [思维链模式](#思维链--推理模式)
- **Ollama 与 OpenAI 兼容 API** —— 现有工具可直接接入的端点，外加浏览器聊天 UI。→ [HTTP API](#http-api)
- **原生量化计算** —— Q4_K_M / Q8_0 / MXFP4 / IQ2_XXS 等量化权重直接参与 matmul，无需反量化为 FP32。

---

以下均为详细参考文档。初次使用？上面五个章节足以让你跑起来。

## 文档导航

| 从这里开始 | 适合你想要... |
|---|---|
| [快速构建与使用](#构建) | 构建解决方案、编译原生 GGML 桥接库、运行 CLI 或 Server |
| [支持的模型架构](#支持的模型架构) | 查看已实现的 GGUF 架构标识、多模态、思维链与工具调用能力 |
| [计算后端](#计算后端) | 在纯 C# CPU、Direct CUDA/cuBLAS、MLX Metal、GGML CPU、GGML Metal、GGML CUDA 之间选择 |
| [HTTP API](#http-api) | 使用 Ollama 兼容、OpenAI 兼容或 Web UI SSE 端点 |
| [按模型架构卡片](docs/models/README_zh-cn.md) | 阅读单个架构的端到端文档（来源、前向计算图、组件、参数，以及 TensorSharp 如何实现并优化 prefill 与 decode） |
| [分页注意力 & 连续批处理](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md) | 了解 vLLM 风格的分页 KV 缓存、前缀共享与迭代级调度器 |
| [推理基准矩阵](docs/inference_benchmark_matrix_zh-cn.md) | 在文本 / 多模态负载与 KV cache dtype 上对比 TensorSharp、llama.cpp 与 Ollama |
| [环境变量功能矩阵](docs/env_var_feature_matrix_zh-cn.md) | 查看高影响运行时开关会影响哪些模型、后端与提示类型 |
| [测试 / 基准矩阵运行器](TensorSharp.TestMatrix/README_zh-cn.md) | 扫描 model × backend × feature × env-var 组合并生成回归报告 |
| [服务端 API 示例](TensorSharp.Server/API_EXAMPLES_zh-cn.md) | 复制完整的 curl 与 Python 示例 |
| [服务端集成测试](TensorSharp.Server/testdata/README_zh-cn.md) | 针对运行中的服务验证公开 API 契约 |

## 当前状态

| 范围 | 状态 |
|---|---|
| 模型家族 | Gemma 3/4、DiffusionGemma、Qwen 3、Qwen 3.5/3.6-family GGUF（`qwen35`、`qwen35moe`、`qwen3next`）、GPT OSS、Nemotron-H（含 Nemotron 3 Nano Omni）、Mistral 3 |
| 推理宿主 | CLI、交互式 REPL、ASP.NET Core Web UI、Ollama 风格 API、OpenAI Chat Completions 风格 API。DiffusionGemma 当前走 CLI 扩散运行模式与 Web UI 去噪流。 |
| 后端 | 纯 C# CPU、Direct CUDA/cuBLAS（`cuda`）、MLX Metal（`mlx`）、GGML CPU、GGML Metal、GGML CUDA |
| 多模态 | Gemma 4 支持图像/视频/音频；Gemma 3、Qwen 3.5-family、Mistral 3、Nemotron-H Omni 支持图像输入 |
| 连续批处理 | vLLM 风格的分页 KV 缓存、跨请求基于内容哈希的前缀共享、迭代级调度器（默认启用，可通过 `--no-continuous-batching` 关闭） |
| 服务端模型范围 | 通过 `--model` 显式托管单个 GGUF；可通过 `--mmproj` 显式指定投影器；不再扫描目录 |
| 可观测性 | 结构化每轮日志、队列状态，以及 Web UI / Ollama / OpenAI 响应中的 KV 缓存复用指标 |
| 测试 / 评测 | `TensorSharp.TestMatrix` 可按模型、后端、功能与环境变量组合做矩阵扫描，并与每类主机的基线做回归对比 |

## 功能特性

- **多架构支持** —— Gemma 4、Gemma 3、DiffusionGemma、Qwen 3、Qwen 3.5/3.6-family、GPT OSS、Nemotron-H、Mistral 3
- **多模态推理** —— 图像、视频和音频输入（Gemma 4）；图像输入（Gemma 3 / Qwen 3.5-family / Mistral 3 / Nemotron-H Omni）
- **思维链 / 推理模式** —— 通过 `<think>` / `<|channel>thought` / `<|channel>analysis` 标签输出结构化的思维链推理（Qwen 3、Qwen 3.5/3.6-family、Gemma 4、GPT OSS、Nemotron-H）
- **工具调用 / 函数调用** —— 模型可调用用户定义的工具；所有三种 API 风格均支持多轮工具调用对话
- **量化模型支持** —— 加载 Q4_K_M、Q8_0、F16、MXFP4 等量化格式的 GGUF 文件；执行原生量化矩阵乘法（matmul），无需反量化到 FP32，并且纯 C# CPU 后端在加载大型 GGUF 时也会保持量化权重压缩状态
- **GPU 加速** —— 通过 GGML 支持 Apple Metal（macOS）和 GGML CUDA（Windows/Linux + NVIDIA），并提供 Direct CUDA/cuBLAS 后端（含 PTX 内核与未覆盖算子的 CPU 回退），以及面向 Apple Silicon 的 MLX 后端（mlx-c / Metal）
- **优化后的纯 C# CPU 后端** —— 为 GEMM、RMSNorm、RoPE、softmax、融合激活等推理热点路径提供托管快速路径和 SIMD 内核
- **连续批处理 & 分页 KV 缓存** —— vLLM 风格的分页 KV 块池，跨请求的块级哈希前缀共享，迭代级调度器（可在批内动态加入/抢占序列），可选的 SSD 冷层用于超大 KV 工作集，原生融合分页注意力内核（`TSGgml_PagedAttentionForward`，在 Metal/CUDA 上驱动 `ggml_flash_attn_ext`）。`TensorSharp.Server` 默认启用，可用 `--no-continuous-batching` 关闭。详见 [docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)。
- **批处理 / 并行推理** —— 已为 Mistral 3、Gemma 4、GPT OSS、Qwen 3、Qwen 3.5/3.6-family、Nemotron-H 全部默认启用 `IBatchedPagedModel.ForwardBatch`，能在一次前向传播中打包 N 个序列，使用 `slotMapping` 进行分页 K/V 写入，并通过原生内核做按序列注意力。每个模型都提供 `TS_<FAMILY>_BATCHED=0` 兜底开关（如 `TS_GEMMA4_BATCHED=0`、`TS_QWEN35_BATCHED=0`、`TS_GPTOSS_BATCHED=0`、`TS_NEMOTRON_BATCHED=0`），可强制回到按序列 KV-swap 路径用于 A/B 对比或回归排查。
- **兼容 Ollama 与 OpenAI API** —— 可作为现有工具链的即插即用替代端点
- **可配置采样** —— temperature、top-k、top-p、min-p、重复/存在/频率惩罚、seed、停止序列
- **聊天模板** —— 从 GGUF 元数据自动加载（Jinja2），并为不同架构提供硬编码回退模板
- **推理引擎** —— `TensorSharp.Server` 中的新 `InferenceEngine`（工作线程调度器 + 分页块池）取代了旧的单请求 FIFO 队列。旧队列对象现在只是状态 / 事件形状的兼容 shim；引擎本身已经处理并发。
- **批处理** —— 控制台应用支持 JSONL 输入，并内置用于测量 prefill / decode 吞吐的推理基准
- **流式输出** —— 按 token 输出（Web 通过 SSE，控制台通过 stdout），并支持中断/停止正在生成的请求
- **文本扩散生成** —— DiffusionGemma 使用 EntropyBound 迭代去噪采样器，而不是自回归 `Forward()`。CLI 提供 `--diffusion-steps`、`--diffusion-seed` 与 `--diffusion-blocks`；Web UI 使用整条消息 `replace` 事件展示实时去噪预览，并通过 `DiffusionBatchScheduler` 批处理并发扩散请求。
- **混合 SSM-Transformer** —— Nemotron-H 在单个模型中混合 Mamba2 SSM 层、纯注意力层和 MoE FFN 层；Mamba2 步现在同时提供单序列原生内核与批处理原生内核（`TSGgml_NemotronMamba2BatchedStepF32`，NEON SIMD + GCD 并行）。
- **混合注意力-递归网络** —— Qwen 3.5/3.6-family 在同一模型中混合全注意力层与 GatedDeltaNet 递归层；批处理路径下递归运行状态保存在每槽位的递归状态池中
- **专家混合（MoE）** —— 支持 Gemma 4 MoE 变体（例如 gemma-4-26B-A4B）、GPT OSS MoE（例如 gpt-oss-20b）、Qwen 3.5/3.6-family MoE（`qwen35moe` / `qwen3next` 变体，例如 Qwen3.5-35B-A3B）以及 Nemotron-H MoE FFN 层
- **批量 GPU MoE** —— Qwen 3.5/3.6-family 与 Nemotron-H 在 decode 时通过单次融合的 GGML 计算图调度处理所有被选中的专家（Qwen 3.5-family 还包括可选的 shared expert 与残差加法），消除每个专家的 CPU-GPU 往返
- **KV 缓存编解码器** —— 通过 `IKvBlockCodec` 接口插件化；内置的 TurboQuant（Q4 / Q8）编解码器可在分页块上启用，由 `--paged-kv-quant-bits` 控制
- **消息编辑** —— 在 Web 聊天界面中编辑或删除历史消息，并从该位置重新生成回复
- **文本/图像/音频/视频上传** —— Web 界面支持最大 500 MB 的文件上传，对超大文本会按 token 预算自动截断
- **每轮可观测性** —— 结构化日志会完整保留用户输入与模型原始输出（包括 `<think>` 思维链和最终结果），并记录 KV 缓存命中率。同样的命中率指标通过所有 API 透出：Ollama 的 `prompt_cache_hit_tokens` / `prompt_cache_hit_ratio`、OpenAI 的 `usage.prompt_tokens_details.cached_tokens`，以及 Web UI SSE `done` 事件中的 `promptTokens` / `kvReusedTokens` / `kvReusePercent`

## 支持的模型架构

| 架构 | GGUF 架构标识 | 示例模型 | 多模态 | 思维链 | 工具调用 | 卡片 |
|---|---|---|---|---|---|---|
| Gemma 4 | `gemma4` | gemma-4-E4B、gemma-4-31B、gemma-4-26B-A4B（MoE） | 图像、视频、音频 | 支持 | 支持 | [gemma4_zh-cn.md](docs/models/gemma4_zh-cn.md) |
| Gemma 3 | `gemma3` | gemma-3-4b | 图像 | 不支持 | 不支持 | [gemma3_zh-cn.md](docs/models/gemma3_zh-cn.md) |
| Qwen 3 | `qwen3` | Qwen3-4B | 仅文本 | 支持 | 支持 | [qwen3_zh-cn.md](docs/models/qwen3_zh-cn.md) |
| Qwen 3.5 / 3.6 family | `qwen35`, `qwen35moe`, `qwen3next` | Qwen3.5-9B（混合 Attn+递归）、Qwen3.5-35B-A3B / Qwen3.6-35B-A3B（MoE） | 图像 | 支持 | 支持 | [qwen35_zh-cn.md](docs/models/qwen35_zh-cn.md) |
| GPT OSS | `gptoss`, `gpt-oss` | gpt-oss-20b（MoE） | 仅文本 | 支持 | 支持 | [gptoss_zh-cn.md](docs/models/gptoss_zh-cn.md) |
| Nemotron-H | `nemotron_h`, `nemotron_h_moe` | Nemotron-H-8B、Nemotron-H-47B（混合 SSM-Transformer，MoE）、Nemotron 3 Nano Omni（图像） | 图像（Omni） | 支持 | 支持 | [nemotron_zh-cn.md](docs/models/nemotron_zh-cn.md) |
| Mistral 3 | `mistral3` | Mistral-Small-3.1-24B-Instruct | 图像 | 不支持 | 不支持 | [mistral3_zh-cn.md](docs/models/mistral3_zh-cn.md) |
| DiffusionGemma | `diffusion-gemma`, `diffusion_gemma` | diffusion-gemma 文本扩散 GGUF | 仅文本 | 不支持 | 不支持 | [diffusiongemma_zh-cn.md](docs/models/diffusiongemma_zh-cn.md) |

各架构的端到端文档见[按模型架构卡片](docs/models/README_zh-cn.md)（来源、前向计算图、组件、参数、权重命名，以及 TensorSharp 如何实现并优化 prefill 与 decode）。

## 模型下载（GGUF）

TensorSharp 使用 GGUF 格式模型文件。以下是各架构对应的 Hugging Face 下载链接。请根据硬件条件选择合适的量化版本（Q4_K_M 适合低内存，Q8_0 适合更高质量等）。

| 架构 | 模型 | GGUF 下载 |
|---|---|---|
| Gemma 4 | gemma-4-E4B-it | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) |
| Gemma 4 | gemma-4-31B-it | [ggml-org/gemma-4-31B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF) |
| Gemma 4 | gemma-4-26B-A4B-it（MoE） | [ggml-org/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF) |
| Gemma 4 | gemma-4-mmproj（多模态投影器） | 包含在上述 GGUF 仓库中 |
| Gemma 3 | gemma-3-4b-it | [google/gemma-3-4b-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf) |
| Qwen 3 | Qwen3-4B | [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF) |
| Qwen 3.5 / 3.6 family | Qwen3.5-9B | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) |
| Qwen 3.5 / 3.6 family | Qwen3.5-35B-A3B | [ggml-org/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF) |
| GPT OSS | gpt-oss-20b（MoE） | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) |
| Nemotron-H | Nemotron-H-8B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF) |
| Nemotron-H | Nemotron-H-47B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF) |
| Mistral 3 | Mistral-Small-3.1-24B-Instruct | [bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF) |
| Mistral 3 | mistral3-mmproj（Pixtral 视觉投影器） | [bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF) |
| DiffusionGemma | diffusion-gemma 文本扩散 GGUF | 使用 `general.architecture` 为 `diffusion-gemma` 或 `diffusion_gemma` 的 GGUF 文件 |

## 计算后端

| 后端 | 参数 | 适合场景 | 说明 |
|---|---|---|---|
| Direct CUDA/cuBLAS | `--backend cuda` | NVIDIA 推理与实验 | 通过 CUDA Driver API、cuBLAS GEMM、常用 Float32 PTX 内核（fill、unary、binary、ternary、activations、RMSNorm、softmax、RoPE/RoPEEx、SDPA、GQA prefill/decode、causal mask、gather/concat），以及受支持 GGUF 量化类型的原生量化 matmul/get-rows 加速推理；未实现的算子会回退到 CPU，同时保持张量语义。 |
| MLX Metal | `--backend mlx` | Apple Silicon（GGML Metal 之外的另一选择） | 基于 [mlx-c](https://github.com/ml-explore/mlx-c) 的 GPU 加速路径。实现了原生量化算子（Q4_K_M、Q8_0、Q5_K、Q6_K、IQ2_XXS、IQ4_XS、IQ4_NL、MXFP4 等无需反量化到 FP32）、融合 decode / prefill Metal kernel（融合 QKV 预处理、融合 gate+up+SiLUMul MoE、融合多维 KV 写入）、编译图 kernel、定期 `async_eval` 让 GPU/CPU 工作重叠的异步 worker 派发、用堆叠权重 slab 的批处理 MoE 解码、MoE 专家 offload、通过 `mlock(2)` 把 GGUF mmap 钉在物理内存、按宿主机派生的分配器上限（`TS_MLX_MEMORY_LIMIT_MB` / `TS_MLX_CACHE_LIMIT_MB` / `TS_MLX_WIRED_LIMIT_MB`），并对未实现的算子提供 CPU 回退。依赖 `libmlxc`（可通过 `TensorSharp.Backends.MLX/build-native-macos.sh` 在本地编译，或用 `TENSORSHARP_MLX_LIBRARY` / `TENSORSHARP_MLX_LIBRARY_DIR` 指定路径）。 |
| GGML Metal | `--backend ggml_metal` | Apple Silicon（macOS 默认） | 通过 Apple Metal 进行 GPU 加速。量化权重通过 host 指针缓冲区从 GGUF 文件零拷贝映射到 Metal command buffer，常驻内存接近模型在磁盘上的大小。 |
| GGML CUDA | `--backend ggml_cuda` | 通过 ggml 使用 NVIDIA 推理 | 通过 GGML CUDA 在 Windows 或 Linux + NVIDIA GPU 上进行加速。量化权重在加载时一次性上传到设备显存，之后释放主机端拷贝。 |
| GGML CPU | `--backend ggml_cpu` | 原生 CPU 内核 | 使用原生 GGML 与优化内核进行 CPU 推理。量化权重以零拷贝方式从 GGUF 文件映射。 |
| 纯 C# CPU | `--backend cpu` | 可移植性与调试 | 无原生依赖的可移植 CPU 推理。 |

## 项目结构

```text
TensorSharp/
├── TensorSharp.Core/            # 核心张量库（Tensor、Ops、内存、设备抽象，含 CPU SIMD/托管量化内核）
├── TensorSharp.Runtime/         # GGUF、分词器、模板、采样、协议解析
│   ├── Paged/                   # 分页 KV 缓存原语（BlockPool、BlockTable、KvBlock、BlockHashIndex、PagedKvStorage、PagedKvBatchOps、ManagedPagedAttention）
│   ├── Scheduling/              # 连续批处理引擎（InferenceEngine、BatchExecutor、ContinuousBatchScheduler、SequenceState、SchedulerConfig/Output、InferenceRequestHandle）
│   ├── PagedKvCacheManager.cs   # 单会话分页 KV 管理（块分配、前缀复用）
│   ├── PagedKvBlockStore.cs     # 带可选 SSD 溢出的 RAM/磁盘分级分页块存储
│   ├── SsdKvBlockTier.cs        # 分页块的 SSD 冷层
│   ├── TurboQuantKvCodec.cs     # 实现 IKvBlockCodec 的量化 KV 块编解码器（Q4 / Q8）
│   ├── PrefillChunking.cs       # SWA / 超长 prompt 使用的分块 prefill 辅助
│   ├── KvBlockHash.cs           # 内容寻址的块哈希，用于跨请求前缀复用
│   └── Logging/                 # JSON-line 文件日志器 + 每轮遥测
├── TensorSharp.Models/          # 模型架构实现与多模态编码/注入
│   ├── Models/<Family>/         # 每个架构一个目录（DiffusionGemma、Gemma3、Gemma4、GptOss、Mistral3、Nemotron、Qwen3、Qwen35）
│   │   ├── <Family>Model.cs                # 旧的单序列 ModelBase 实现
│   │   └── <Family>Model.BatchedForward.cs # IBatchedPagedModel.ForwardBatch —— 批处理/分页路径（Mistral3、Gemma4、GptOss、Qwen35、Nemotron、Qwen3）
│   ├── Paged/                   # 张量侧的分页注意力辅助（TensorPagedAttention）
│   ├── KvBlockTransfer.cs       # 跨序列的 KV 块 extract/inject 辅助
│   └── ModelMultimodalInjector.cs # 视觉 / 音频 / 视频嵌入注入
├── TensorSharp.Backends.GGML/   # GGML 后端绑定（通过原生库支持 Metal/CUDA/CPU）
├── TensorSharp.Backends.Cuda/   # Direct CUDA 后端（CUDA Driver API、cuBLAS、PTX 内核）
├── TensorSharp.Backends.MLX/    # Apple Silicon MLX 后端（mlx-c / Metal），原生桥接由 `build-native-macos.sh` 编译
├── TensorSharp.GGML.Native/     # 到 ggml 的原生 C++ 桥接（构建 libGgmlOps，拆分为多个专注源文件）
│   ├── ggml_ops_core.cpp                  # 元素级、归约、基础 shape 操作
│   ├── ggml_ops_elementwise.cpp           # 元素级 / 激活融合
│   ├── ggml_ops_matmul.cpp                # GEMM / 量化 matmul
│   ├── ggml_ops_fused.cpp                 # 跨域融合的每层内核
│   ├── ggml_ops_norm_attn.cpp             # Norm + 注意力融合
│   ├── ggml_ops_transformer.cpp           # 完整层的融合 Transformer 内核（decode + prefill）
│   ├── ggml_ops_moe.cpp                   # 专家混合前向 / 融合路由
│   ├── ggml_ops_gated_delta_net.cpp       # Qwen 3.5/3.6 GatedDeltaNet 内核（按序列 + 批处理）
│   ├── ggml_ops_mamba2.cpp                # Nemotron Mamba2 内核（按序列 + 批处理 SIMD）
│   ├── ggml_ops_paged_attention.cpp       # 分页注意力原生内核（驱动 ggml_flash_attn_ext + sinks 变体）
│   ├── ggml_ops_diffusion.cpp             # DiffusionGemma 融合 decode-layer / 整模型 / lm-head 内核
│   ├── ggml_ops_training.cpp              # 仅训练用内核（运行时不使用）
│   └── tests/                              # 原生单元 + 烟雾测试
├── TensorSharp.Server/          # Web 聊天 + API 服务（ASP.NET Core）
│   ├── Program.cs               # 精简启动：DI 注册、中间件、端点映射、paged-KV + 连续批处理 CLI 翻译
│   ├── ModelService.cs          # 保持服务端推理公共 API 稳定的门面，持有 InferenceEngineHost
│   ├── ModelLifecycleService.cs # 模型加载/释放与后端选择（CPU / CUDA / MLX / GGML CPU/Metal/CUDA）
│   ├── InferenceEngineHost.cs   # DI 注册的单模型 InferenceEngine 单例（连续批处理入口）
│   ├── ChatGenerationPipeline.cs # Prompt 渲染，将请求提交到 InferenceEngine，流式返回 token，处理 stop
│   ├── InferenceTelemetry.cs    # Prompt/eval 计时、TTFT、tokens/sec、完整输入/输出日志
│   ├── ChatHistoryPreparer.cs   # 历史归一化、raw token 拼接、多模态顺序辅助
│   ├── ChatSession.cs           # 单会话历史跟踪与 assistant raw token
│   ├── SessionManager.cs        # 线程安全的会话注册（默认会话 + 每个 UI Tab 的会话）
│   ├── InferenceQueue.cs        # 向后兼容的队列状态接口（并发由引擎本身处理）
│   ├── BackendCatalog.cs        # 可用计算后端的发现（CPU / CUDA / MLX / GGML*）
│   ├── TextUploadHelper.cs      # 按 token 预算截断的文本上传辅助
│   ├── WebUiChatPolicy.cs       # Web UI 聊天请求合法性校验
│   ├── OpenAIResponseFormatParser.cs  # OpenAI response_format（json_object / json_schema）解析
│   ├── Hosting/                 # 启动期相关：选项装配（ServerOptionsBuilder）、后端选择、日志、wwwroot 解析、paged-KV / 连续批处理 CLI 翻译
│   ├── RequestParsers/          # JSON 请求解析（采样配置、聊天消息、工具函数）
│   ├── ResponseSerializers/     # 各协议响应形状构造（Ollama / OpenAI / Web UI）
│   ├── StreamingWriters/        # SSE 与 NDJSON 线协议辅助
│   ├── ProtocolAdapters/        # 各协议的请求处理器（WebUiAdapter、OllamaAdapter、OpenAIChatAdapter）
│   ├── Endpoints/               # ASP.NET Core 路由映射（每协议一个扩展方法）
│   ├── Logging/                 # 请求日志中间件 + 低噪声路径支持
│   ├── wwwroot/index.html       # 聊天界面
│   ├── testdata/                # 集成测试套件（bash + Python）
│   └── API_EXAMPLES.md          # 详细 API 文档
├── TensorSharp.Cli/             # CLI 应用（单次生成、交互式 REPL、JSONL 批处理、基准）
├── TensorSharp.TestMatrix/      # 测试 / 基准矩阵运行器、默认提示、环境变量扫描与主机基线
├── InferenceWeb.Tests/          # xUnit 单元测试，覆盖算子、KV 缓存、分页调度器、批处理模型正确性以及 Web/服务辅助逻辑
├── AdvUtils/                    # 工具库（日志）
├── docs/                        # 开发者参考文档
│   ├── models/                  # 按模型架构卡片（每个模型一份 .md，中英双语）
│   ├── PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md  # 分页 KV 缓存、前缀共享、调度器、按模型批处理状态
│   ├── env_var_feature_matrix.md  # TestMatrix 使用的运行时开关 × 模型/后端/功能覆盖矩阵
│   └── inference_benchmark_matrix.md  # 跨引擎吞吐矩阵（TensorSharp vs llama.cpp vs Ollama）
├── benchmarks/                  # 可重现的基准脚本
│   └── inference_matrix/        # 驱动脚本、modelfiles、prompts、每格原始 JSON 结果
└── ExternalProjects/            # ggml/ 在构建时从 github.com/ggml-org/ggml 克隆（不纳入版本控制）
```

## NuGet 包分层

现在仓库按包边界拆成独立层，使用者可以只引用真正需要的部分。

| 项目 | NuGet 包 | 对外 namespace | 职责 |
|---|---|---|---|
| `TensorSharp.Core` | `TensorSharp.Core` | `TensorSharp` | Tensor 原语、Ops、分配器、存储与设备抽象 |
| `TensorSharp.Runtime` | `TensorSharp.Runtime` | `TensorSharp.Runtime` | GGUF 解析、分词器、Prompt 渲染、采样、输出协议解析、分页 KV 缓存、连续批处理调度器 |
| `TensorSharp.Models` | `TensorSharp.Models` | `TensorSharp.Models` | `ModelBase`、各模型架构、多模态编码器、批处理 / 分页前向、模型侧执行辅助 |
| `TensorSharp.Backends.GGML` | `TensorSharp.Backends.GGML` | `TensorSharp.GGML` | GGML 执行后端与原生互操作 |
| `TensorSharp.Backends.Cuda` | `TensorSharp.Backends.Cuda` | `TensorSharp.Cuda` | Direct CUDA 分配器、存储、cuBLAS GEMM、PTX 内核和量化 CUDA 算子 |
| `TensorSharp.Backends.MLX` | `TensorSharp.Backends.MLX` | `TensorSharp.MLX` | Apple Silicon MLX 后端（mlx-c / Metal），含量化 / 融合 / 编译内核与 MoE 专家 offload |
| `TensorSharp.Server` | `TensorSharp.Server` | `TensorSharp.Server` | ASP.NET Core 服务、OpenAI/Ollama 适配层、推理引擎宿主与 Web UI |
| `TensorSharp.Cli` | `TensorSharp.Cli` | `TensorSharp.Cli` | 控制台宿主、调试工具与 JSONL 批处理 |

这样的拆分让引擎使用者不必带上 Web 依赖，也能把 API 层改动和核心运行时隔离开，并让后续 benchmark / eval harness 更容易独立发布。

发布前可验证包元数据与 README 依赖边界：

```powershell
pwsh ./eng/verify-packages.ps1
```

该验证会对上表 7 个公开包运行 `dotnet pack`，并在 `AdvUtils` 等内部依赖泄漏到 `.nuspec`，或 TensorSharp 包依赖了上表之外的分层时失败。

## 前置要求

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- **`git` 与网络访问：** GGML/CUDA 原生构建会在首次构建时从 [github.com/ggml-org/ggml](https://github.com/ggml-org/ggml) 克隆 ggml 源码到 `ExternalProjects/ggml/`（参见 `eng/fetch-ggml.sh` / `eng/fetch-ggml.ps1`）。克隆默认跟踪 ggml 的默认分支（`master`）；可用 `TENSORSHARP_GGML_GIT_REF` 指定其他引用，或在克隆完成后设置 `TENSORSHARP_GGML_NO_UPDATE=1` 跳过网络更新（用于离线重建）
- **macOS（Metal 后端）：** 用于构建原生 GGML 库的 CMake 3.20+ 与 Xcode 命令行工具；若需使用 MLX 后端，还需通过 `bash TensorSharp.Backends.MLX/build-native-macos.sh` 从 `TensorSharp.Backends.MLX/Native/` 构建 `libmlxc`
- **Windows（GGML CPU / CUDA 后端）：** CMake 3.20+ 与 Visual Studio 2022 C++ 构建工具；若使用 `ggml_cuda` 或 `cuda`，还需要 NVIDIA 驱动和带 cuBLAS 的 CUDA Toolkit 12.x 或其他兼容版本
- **Linux（GGML CPU / CUDA 后端）：** CMake 3.20+；若使用 `ggml_cuda` 或 `cuda`，还需要 NVIDIA 驱动和带 cuBLAS 的 CUDA Toolkit 12.x 或其他兼容版本
- GGUF 模型文件（例如来自 [Hugging Face](https://huggingface.co)）

## 构建

### 构建整个解决方案

```bash
dotnet build TensorSharp.slnx
```

### 构建单独应用

```bash
# 控制台应用
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj

# Web 应用
dotnet build TensorSharp.Server/TensorSharp.Server.csproj
```

### 构建原生 GGML 库

如果原生库不存在，首次执行 `dotnet build` 时会自动构建。也可以手动构建：

```bash
cd TensorSharp.GGML.Native
```

macOS：

```bash
bash build-macos.sh
```

Linux（仅 CPU）：

```bash
bash build-linux.sh
```

Linux（启用 GGML_CUDA）：

```bash
bash build-linux.sh --cuda
```

Windows（仅 CPU）：

```powershell
.\build-windows.ps1 --no-cuda
```

Windows（启用 GGML_CUDA）：

```powershell
.\build-windows.ps1 --cuda
```

在 Windows 和 Linux 上，原生构建脚本会自动检测可见 NVIDIA GPU 的 compute capability，并把一个精简的 `CMAKE_CUDA_ARCHITECTURES` 列表传给 ggml-cuda（例如在 RTX 3080 上为 `86-real`），从而显著降低 CUDA 构建时间。原生构建默认还会以受控的并行任务数运行，避免 `nvcc` 拖慢普通开发机器。

如需覆盖自动检测到的架构列表或默认的并行度，可使用以下任一方式：

```bash
TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES='86-real;89-real' bash build-linux.sh --cuda
bash build-linux.sh --cuda --cuda-arch='86-real;89-real'
TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL=2 bash build-linux.sh --cuda
```

```powershell
$env:TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES='86-real;89-real'; .\build-windows.ps1 --cuda
.\build-windows.ps1 --cuda --cuda-arch='86-real;89-real'
$env:TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL=2; .\build-windows.ps1 --cuda
```

也可以在 `dotnet build` 时通过环境变量请求 CUDA 版本的原生库：

```bash
TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release
```

```powershell
$env:TENSORSHARP_GGML_NATIVE_ENABLE_CUDA='ON'; dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release
```

在 macOS 上会生成带 Metal GPU 支持的 `libGgmlOps.dylib`。在 Windows 和 Linux 上，原生脚本会保留已有的 CUDA 构建，并在检测到 CUDA 工具链时自动启用 GGML_CUDA；也可以通过 `build-windows.ps1 --cuda`、`build-linux.sh --cuda` 或 `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON` 显式启用。构建产物会自动复制到应用输出目录。

Direct `cuda` 后端由托管 C# 代码和 PTX 内核组成。执行 `dotnet build` 时，`TensorSharp.Backends.Cuda` 会在检测到 `nvcc` 后把 `native/kernels/*.cu` 编译到 `native/ptx/*.ptx`；如果缺少 `nvcc`，构建会继续，PTX 覆盖的算子会使用 CPU 回退。cuBLAS GEMM 仍要求运行时能够找到 CUDA 运行库。

### 构建原生 MLX 库（仅 macOS）

MLX 后端依赖 `libmlxc`（[MLX](https://github.com/ml-explore/mlx) 的 C 绑定）。仓库在 `TensorSharp.Backends.MLX/Native/MLX_C_VERSION` 中固定了已知可用的 `mlx-c` tag，并提供一个辅助脚本来获取和构建：

```bash
bash TensorSharp.Backends.MLX/build-native-macos.sh
```

脚本会把生成的库（`libmlxc.dylib`、`libmlx.dylib` 以及任何后端依赖）写入 `TensorSharp.Backends.MLX/Native/dist/`。运行时后端会优先在应用目录下查找；也可以使用 `TENSORSHARP_MLX_LIBRARY=<libmlxc.dylib 路径>` 或 `TENSORSHARP_MLX_LIBRARY_DIR=<包含 libmlxc 的目录>` 指定自定义安装位置。如果找不到对应库，后端会报告不可用，启动时 `--backend mlx` 会被拒绝。

## 使用方法

### 控制台应用

```bash
cd TensorSharp.Cli/bin

# 文本推理
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --output result.txt \
    --max-tokens 200 --backend ggml_metal

# Windows/Linux + NVIDIA GPU 文本推理
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --output result.txt \
    --max-tokens 200 --backend ggml_cuda

# 交互式逐轮对话（REPL），支持 KV 缓存复用与斜杠命令
./TensorSharp.Cli --model <model.gguf> --backend ggml_metal --interactive
./TensorSharp.Cli --model <model.gguf> --backend ggml_metal -i \
    --system "你是一名简洁的助手。" --temperature 0.7 --top-p 0.9 --think

# 图像推理（Gemma 3/4，Qwen 3.5-family）
./TensorSharp.Cli --model <model.gguf> --image photo.png --backend ggml_metal

# 视频推理（Gemma 4）
./TensorSharp.Cli --model <model.gguf> --video clip.mp4 --backend ggml_metal

# 音频推理（Gemma 4）
./TensorSharp.Cli --model <model.gguf> --audio speech.wav --backend ggml_metal

# DiffusionGemma 文本扩散生成
./TensorSharp.Cli --model <diffusion-gemma.gguf> --input prompt.txt --backend ggml_metal \
    --max-tokens 256 --diffusion-steps 48 --diffusion-seed 0

# 思维链 / 推理模式
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --backend ggml_metal --think

# 工具调用
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --tools tools.json

# 使用采样参数
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.2 --seed 42

# 批处理（JSONL）
./TensorSharp.Cli --model <model.gguf> --input-jsonl requests.jsonl \
    --output results.txt --backend ggml_metal

# 多轮对话模拟（含 KV 缓存复用，模拟 Web UI 行为）
./TensorSharp.Cli --model <model.gguf> --multi-turn-jsonl chat.jsonl \
    --backend ggml_metal --max-tokens 200

# 吞吐基准测试：N 次最优运行的 prefill 和 decode 计时
./TensorSharp.Cli --model <model.gguf> --backend ggml_metal \
    --benchmark --bench-prefill 256 --bench-decode 128 --bench-runs 3

# KV 缓存复用基准：在多轮对话中比较启用与禁用缓存的 prefill 时延
# （以一个 8 轮的对话为例，对比有缓存与强制重置的 prefill 延迟差异）
./TensorSharp.Cli --model <model.gguf> --backend ggml_metal \
    --bench-kvcache --bench-kv-turns 4 --max-tokens 64

# 仅查看渲染后的 prompt 和分词结果（不运行推理）
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --dump-prompt

# 对目录下每个 *.gguf 文件，对比硬编码回退模板与 GGUF 内置 Jinja2 模板
# （在适配新架构时尤其有用）
./TensorSharp.Cli --test-templates ~/models
```

**命令行参数：**

| 参数 | 说明 |
|---|---|
| `--model <path>` | GGUF 模型文件路径（必填） |
| `--input <path>` | 包含用户提示词的文本文件 |
| `--input-jsonl <path>` | JSONL 批量请求文件（每行一个 JSON） |
| `--multi-turn-jsonl <path>` | 用于多轮对话模拟（含 KV 缓存复用）的 JSONL 文件 |
| `--output <path>` | 将生成文本写入该文件 |
| `--image <path>` | 用于视觉推理的图像文件 |
| `--video <path>` | 用于视频推理的视频文件 |
| `--audio <path>` | 音频文件（WAV、MP3、OGG）用于音频推理 |
| `--mmproj <path>` | 多模态投影器 GGUF 文件路径 |
| `--max-tokens <N>` | 最大生成 token 数（默认：100） |
| `--backend <type>` | 计算后端：`cpu`、`cuda`、`mlx`、`ggml_cpu`、`ggml_metal` 或 `ggml_cuda` |
| `--kv-cache-dtype <type>` | KV 缓存精度：`f32`（默认）、`f16` 或 `q8_0`。量化 / 半精度 KV 缓存以微小数值漂移换取内存节省；详见 [`docs/inference_benchmark_matrix_zh-cn.md`](docs/inference_benchmark_matrix_zh-cn.md)。 |
| `--interactive` / `-i` | 进入交互式 REPL 聊天会话（逐轮输入/输出），支持 KV 缓存复用、斜杠命令、运行时热切换 模型/后端/投影器、文件附件（图像、音频、视频、文本）以及实时调整采样参数。完整命令列表见下文「**交互式 REPL 命令**」一节 |
| `--system <text>` | 用于初始化交互式会话的系统提示词（在 REPL 中可用 `/system` 覆盖） |
| `--system-file <path>` | 从 UTF-8 文本文件读取初始系统提示词（`--system` 的替代写法） |
| `--think` | 启用思维链/推理模式 |
| `--tools <path>` | 包含工具/函数定义的 JSON 文件 |
| `--temperature <f>` | 采样温度（0 = 贪心） |
| `--top-k <N>` | Top-K 过滤（0 = 关闭） |
| `--top-p <f>` | Nucleus 采样阈值（1.0 = 关闭） |
| `--min-p <f>` | 最小概率过滤（0 = 关闭） |
| `--repeat-penalty <f>` | 重复惩罚（1.0 = 无） |
| `--presence-penalty <f>` | 存在惩罚（0 = 关闭） |
| `--frequency-penalty <f>` | 频率惩罚（0 = 关闭） |
| `--seed <N>` | 随机种子（-1 = 非确定性） |
| `--stop <string>` | 停止序列（可重复指定） |
| `--dump-prompt` | 仅渲染 prompt 与分词后退出（不进行推理） |
| `--benchmark` | 运行合成的 prefill / decode 吞吐基准 |
| `--bench-prefill <N>` | 合成 prefill 的 token 长度（默认：32） |
| `--bench-decode <N>` | 合成 decode 的 token 长度（默认：64） |
| `--bench-runs <N>` | 基准运行次数；输出最佳与平均结果（默认：1） |
| `--bench-kvcache` | 运行多轮 KV 缓存复用基准（对比启用缓存与强制重置时的 prefill 延迟） |
| `--bench-kv-turns <N>` | `--bench-kvcache` 使用的对话轮数（默认：4，最多 8） |
| `--bench-chunked` | 运行分块 prefill 微基准（Gemma 4） |
| `--warmup-runs <N>` | 在对真实文本 / 多模态 prompt 计时前丢弃的前向次数（默认：0） |
| `--test-chunked-prefill` | 运行分块 prefill 正确性检查（对比分块与非分块 logits） |
| `--correct-prefill <N>` | `--test-chunked-prefill` 使用的 prompt 长度 |
| `--correct-decode <N>` | `--test-chunked-prefill` 使用的 decode 长度 |
| `--diffusion-steps <N>` | DiffusionGemma 每个 block 的去噪步数（默认：48） |
| `--diffusion-seed <N>` | DiffusionGemma 确定性采样种子（默认：0） |
| `--diffusion-blocks <N>` | DiffusionGemma block-autoregressive canvas 数量。`0` 表示根据 `--max-tokens` 与模型 canvas 长度推导。 |
| `--test` | 运行内置的分词器、Qwen3 聊天模板与 ollama 对比测试 |
| `--test-templates <dir>` | 对 `<dir>` 下的每个 *.gguf 校验硬编码模板与 GGUF Jinja2 模板的一致性 |
| `--log-level <lvl>` | 控制台与文件日志级别：`trace`、`debug`、`info`、`warning`、`error`、`critical`、`off` |
| `--log-dir <path>` | JSON-line 文件日志的写入目录（默认：`<binDir>/logs`） |
| `--log-file <0\|1>` | 关闭（`0`）或开启（`1`）文件日志（默认：开启） |
| `--log-console <0\|1>` | 关闭（`0`）或开启（`1`）控制台日志（默认：开启） |

如果把多模态投影器文件放在模型文件同目录并使用可识别命名（例如 `gemma-4-mmproj-F16.gguf`），系统会自动检测。

**JSONL 输入格式：**

每行是一个 JSON 对象，包含 `messages`、可选 `prompt` 和可选采样参数：

```json
{"id": "q1", "messages": [{"role": "user", "content": "What is 2+3?"}], "max_tokens": 50}
{"id": "q2", "messages": [{"role": "user", "content": "Write a haiku."}], "max_tokens": 100, "temperature": 0.8}
```

**交互式 REPL 命令：**

通过 `--interactive` / `-i` 启动后，可使用斜杠命令驱动当前会话。在 REPL 中输入 `/help`（或 `/?`）可查看相同的命令列表。任何不以 `/` 开头的输入都会被视为一轮用户消息。

每轮提示符前的状态行会汇总当前状态——模型、后端、架构、上下文长度、投影器、对话深度，以及为下一轮排队的附件数量（例如 `[turn 3 (2 attachments pending)]> `）。生成过程中按 Ctrl+C 可中断当前回复；在提示符处按 Ctrl+C 可退出。

会话控制：

| 命令 | 说明 |
|---|---|
| `/help`、`/?` | 显示全部交互命令 |
| `/exit`、`/quit` | 退出当前会话 |
| `/reset`、`/new` | 清空对话历史与 KV 缓存 |
| `/history` | 打印对话历史 |
| `/save <文件>` | 将当前对话追加写入 UTF-8 文件 |
| `/system <文本>` | 设置系统提示词（参数为空表示清空），并重置 KV 缓存 |
| `/think on\|off` | 切换思维链/推理模式（仅对支持的模型生效） |
| `/multiline on\|off` | 切换多行输入（在单独一行输入 `.` 结束消息） |

模型与运行时：

| 命令 | 说明 |
|---|---|
| `/info`、`/status` | 显示当前加载的模型、后端、架构、上下文/词表大小、投影器、对话深度与待发送附件 |
| `/model <路径>` | 在当前后端上加载另一个 `.gguf` 模型（会重置会话） |
| `/backend <名称>` | 用其他后端重新加载当前模型：`cpu`、`cuda`、`mlx`、`ggml_cpu`、`ggml_metal`、`ggml_cuda` |
| `/mmproj <路径>` | 为当前模型加载（或替换）多模态投影器。别名：`/projector` |

采样（实时生效，跨多轮持久化）：

| 命令 | 说明 |
|---|---|
| `/sampling`、`/show` | 打印当前采样配置 |
| `/max <N>` | 单次回复最大 token 数 |
| `/temp <float>` | 采样温度（0 = 贪心） |
| `/topk <int>` | Top-K 过滤（0 = 关闭） |
| `/topp <float>` | Top-P / Nucleus 阈值（1.0 = 关闭） |
| `/minp <float>` | Min-P 过滤（0 = 关闭） |
| `/repeat <float>` | 重复惩罚（1.0 = 关闭） |
| `/presence <float>` | 存在惩罚 |
| `/frequency <float>` | 频率惩罚 |
| `/seed <int>` | 随机种子（-1 = 非确定性） |
| `/stop <文本>` | 追加一条停止序列 |
| `/clearstop` | 清空所有停止序列 |

附件上传（排队到下一轮，发送后自动清空）：

| 命令 | 说明 |
|---|---|
| `/image <路径>`、`/img <路径>` | 附加一张图像（仅对视觉模型有效） |
| `/audio <路径>` | 附加一个音频文件（Gemma 4） |
| `/video <路径>`、`/vid <路径>` | 附加视频，自动抽取关键帧（Gemma 4） |
| `/text <路径>`、`/file <路径>`、`/txt <路径>` | 将 UTF-8 文本/Markdown/CSV/代码文件内联到下一轮提示词中（超大文件会按 token 预算自动截断） |
| `/clearattach` | 清空尚未发送的图像/音频/视频/文本附件 |

路径支持单引号或双引号，因此可以直接在 macOS 上从 Finder 拖拽文件到终端。多模态命令需要先加载多模态投影器——在启动时通过 `--mmproj` 指定，或在 REPL 中用 `/mmproj <路径>` 加载。

### Web 应用

```bash
cd TensorSharp.Server/bin

# 通过 --model 指定要托管的模型
./TensorSharp.Server --model ./models/model.gguf --backend ggml_metal

# Linux + NVIDIA GPU
./TensorSharp.Server --model ./models/model.gguf --backend ggml_cuda

# 多模态模型：同时显式指定投影器
./TensorSharp.Server --model ./models/model.gguf --mmproj ./models/mmproj.gguf --backend ggml_cuda

# 配置服务端默认采样参数（仅在请求未自行覆盖时生效）
./TensorSharp.Server --model ./models/model.gguf --backend ggml_metal \
    --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.1 \
    --presence-penalty 0.0 --frequency-penalty 0.0 --seed 42 \
    --stop "</s>" --stop "<|endoftext|>"
```

在浏览器中打开 `http://localhost:5000`。Web 界面支持：

- 多轮聊天
- 每个浏览器 Tab 独立的会话：每个 Tab 拥有自己的对话历史；KV block 由推理引擎统一管理
- 通过 `--model` 显式托管单个 GGUF 模型
- 在需要时通过 `--mmproj` 显式托管多模态投影器
- 上传图像、视频和音频进行多模态推理（最大 500 MB）
- 思维链/推理模式切换
- 带函数定义的工具调用
- 通过 Server-Sent Events 进行流式 token 生成
- 承载 `diffusion-gemma` GGUF 时展示 DiffusionGemma 去噪预览（每一步替换整条 assistant 消息，最终再发出定稿）
- 向后兼容的队列状态事件（实际并发由推理引擎处理）
- 消息编辑和删除，支持从对话中任意位置重新生成
- 自由滚动：在生成过程中可向上滚动查看历史消息；只要重新滚回底部，新内容会继续自动跟随

使用 `--model` 选择要托管的 GGUF 文件，使用 `--mmproj` 选择要托管的投影器文件。`TensorSharp.Server` 不再扫描 `MODEL_DIR`。

**服务命令行参数：**

| 参数 | 说明 |
|---|---|
| `--model <path>` | 需要托管的 GGUF 文件（推理时必填；如未指定，服务仍可启动，但 `/api/models/load` 会报告未加载模型） |
| `--mmproj <path>` | 多模态投影器 GGUF（仅给文件名时按模型目录解析；传 `none` 可显式禁用）。需要先指定 `--model`。 |
| `--backend <type>` | 默认计算后端：`cpu`、`cuda`、`mlx`、`ggml_cpu`、`ggml_metal` 或 `ggml_cuda` |
| `--max-tokens <N>` | 当请求未携带 max-tokens 时使用的默认上限（默认：`20000`） |
| `--temperature <f>` | 当请求未提供时使用的默认采样温度（`0` = 贪心） |
| `--top-k <N>` | 当请求未提供时使用的默认 Top-K 过滤（`0` = 关闭） |
| `--top-p <f>` | 当请求未提供时使用的默认 Nucleus 采样阈值（`1.0` = 关闭） |
| `--min-p <f>` | 当请求未提供时使用的默认 min-p 过滤（`0` = 关闭） |
| `--repeat-penalty <f>` | 当请求未提供时使用的默认重复惩罚（`1.0` = 无） |
| `--presence-penalty <f>` | 当请求未提供时使用的默认存在惩罚（`0` = 关闭） |
| `--frequency-penalty <f>` | 当请求未提供时使用的默认频率惩罚（`0` = 关闭） |
| `--seed <N>` | 当请求未提供时使用的默认随机种子（`-1` = 非确定性） |
| `--stop <string>` | 默认停止序列（可重复指定）。请求体里的 `stop`/`stop_sequences` 会**完全替换**默认列表，而不是与之合并。 |
| `--continuous-batching` / `--no-continuous-batching` | 启用（默认）或关闭迭代级分页批处理。启用时服务会在批内动态加入 / 抢占序列，并在实现了 `IBatchedPagedModel` 的模型上将多个序列打包到一次前向中执行。`--no-continuous-batching` 会让所有模型回退到按序列 KV 交换。别名：`--paged-batching` / `--no-paged-batching`。 |
| `--mtp-spec` / `--no-mtp-spec` | 在带有多 token 预测草稿头（Qwen3.6 NextN/MTP）的模型上启用投机解码（默认关闭）。仅对单序列（无并发）请求生效：草稿头每步最多提议 `--mtp-draft` 个 token，主干网络用一次批量前向完成验证；起草与验证均由该请求自己的采样器（含惩罚项）驱动。环境变量：`TS_MTP_SPEC`。 |
| `--mtp-draft <N>` | 每个投机步最多起草的 token 数（默认 `8`）。环境变量：`TS_MTP_DRAFT`。 |
| `--mtp-pmin <f>` | 草稿 token 被保留所需的最低置信度，取值 `(0, 1]`；遇到第一个低置信 token 即停止起草（默认 `0.75`）。环境变量：`TS_MTP_PMIN`。 |
| `--paged-kv` / `--no-paged-kv` | 已移除的按会话分页 KV 管理器的兼容参数。当前服务端 KV 状态由引擎持有；请使用连续批处理 / `TS_SCHED_*` 开关调节引擎。别名：`--paged-kv-cache` / `--no-paged-kv-cache`。 |
| `--paged-kv-block-size <N>` | 旧的独立分页 KV 块大小。当前引擎使用 `TS_SCHED_BLOCK_SIZE`。 |
| `--paged-kv-ram-mb <N>` | 旧的独立分页 KV RAM 层上限。 |
| `--paged-kv-ssd-dir <dir>` | 旧的独立分页 KV SSD 冷层目录。 |
| `--paged-kv-ssd-mb <N>` | 旧的独立分页 KV SSD 上限。 |
| `--paged-kv-quant-bits <0\|4\|8>` | 旧的独立分页 KV 块量化（TurboQuantKvCodec）。 |

请求 JSON 中的字段（如 `temperature`、`top_p`、`top_k`、`min_p`、
`repeat_penalty`、`presence_penalty`、`frequency_penalty`、`seed`、
`stop`/`stop_sequences`）始终优先于上述服务端默认值；这些默认值仅
用于填充客户端未指定的字段。

**运行时环境变量：**

| 变量 | 说明 |
|---|---|
| `BACKEND` | 未传 `--backend` 时使用的默认计算后端（`cpu`、`cuda`、`mlx`、`ggml_cpu`、`ggml_metal` 或 `ggml_cuda`；默认：macOS 为 `ggml_metal`，其他平台为 `ggml_cpu`） |
| `MAX_TOKENS` | 当 `--max-tokens` 与请求级上限均未指定时使用的默认生成长度（默认：`20000`） |
| `MAX_TEXT_FILE_CHARS` | 在没有可用分词器时，对纯文本上传按字符数截断的上限（默认：`8000`） |
| `VIDEO_SAMPLE_FPS` | 视频提示词每秒抽取的帧数；基于时间的抽帧（默认：`1`） |
| `VIDEO_MAX_FRAMES` | 抽取视频帧数量的可选上限（超出时均匀降采样）；未设置或为 `0` 表示不限制（默认：不限制） |
| `PORT` / `ASPNETCORE_URLS` | 当前会被 `Program.cs` 中固定的 `http://0.0.0.0:5000` 监听地址覆盖；Docker Space 镜像会在构建时用 `APP_PORT` 改写该常量。 |
| `TENSORSHARP_TEMPERATURE` | `--temperature` 与请求体均未指定时的默认采样温度 |
| `TENSORSHARP_TOP_K` | `--top-k` 与请求体均未指定时的默认 Top-K |
| `TENSORSHARP_TOP_P` | `--top-p` 与请求体均未指定时的默认 Top-P |
| `TENSORSHARP_MIN_P` | `--min-p` 与请求体均未指定时的默认 min-P |
| `TENSORSHARP_REPEAT_PENALTY` | `--repeat-penalty` 与请求体均未指定时的默认重复惩罚 |
| `TENSORSHARP_PRESENCE_PENALTY` | `--presence-penalty` 与请求体均未指定时的默认存在惩罚 |
| `TENSORSHARP_FREQUENCY_PENALTY` | `--frequency-penalty` 与请求体均未指定时的默认频率惩罚 |
| `TENSORSHARP_SEED` | `--seed` 与请求体均未指定时的默认随机种子 |
| `TENSORSHARP_LOG_LEVEL` | 控制台与文件日志的最低输出级别：`Trace`、`Debug`、`Information`、`Warning`、`Error`、`Critical`（默认：`Information`）。`TensorSharp.Cli` 同样识别该变量。 |
| `TENSORSHARP_LOG_DIR` | JSON-line 文件日志的写入目录（默认：`<binDir>/logs`）。`TensorSharp.Cli` 同样识别该变量。 |
| `TENSORSHARP_LOG_FILE` | 设为 `0` 可关闭文件日志，仅保留控制台输出（默认：开启）。`TensorSharp.Cli` 同样识别该变量。 |
| `DIFFUSION_STEPS` | 服务端 DiffusionGemma 每个 block 的去噪步数（默认：`48`；CLI 对应 `--diffusion-steps`） |
| `DIFFUSION_MAX_BATCH` | Web UI 扩散调度器可批处理的最大并发 DiffusionGemma 请求数（默认：`2`） |

**分页 KV 缓存 & 连续批处理可调参数（进程 / 模型启动时读取）**

下述变量也可以通过 `--paged-kv*` / `--continuous-batching` CLI 参数设置（它们会被翻译为对应的环境变量）：

| 变量 | 说明 |
|---|---|
| `TS_KV_PAGED_CACHE` | 旧的独立 `PagedKvCacheManager` 兼容开关；当前 `TensorSharp.Server` 的请求 KV 状态由引擎持有。CLI 快捷方式是 `--paged-kv` / `--no-paged-kv`。 |
| `TS_KV_BLOCK_SIZE` | 旧的独立分页 KV 块大小。当前引擎使用 `TS_SCHED_BLOCK_SIZE`。 |
| `TS_KV_CACHE_MAX_RAM_MB` | 旧的独立分页 KV RAM 层上限。 |
| `TS_KV_CACHE_SSD_DIR` | 旧的独立分页 KV SSD 冷层目录。 |
| `TS_KV_CACHE_MAX_SSD_MB` | 旧的独立分页 KV SSD 上限。 |
| `TS_KV_PAGED_QUANT_BITS` | 旧的独立分页 KV 块量化位数（`0` = 透传，`4`，或 `8`）。 |
| `TS_SCHED_DISABLE_BATCHED` | `1` 会即使模型实现了 `IBatchedPagedModel`，也强制回退到按序列 KV 交换。CLI 快捷方式是 `--no-continuous-batching`。 |
| `TS_SCHED_MAX_BATCHED_TOKENS` | 调度器每步 token 预算（默认：`4096`）。 |
| `TS_SCHED_MAX_RUNNING_SEQS` | 同时在执行的最大序列数（默认：`16`）。 |
| `TS_SCHED_PREFILL_CHUNK` | 每步最大 prefill token 数（默认：`1024`）。 |
| `TS_SCHED_NUM_BLOCKS` | 引擎块池的物理块数（默认：`256`）。 |
| `TS_SCHED_BLOCK_SIZE` | 引擎侧每块的 token 数（默认：`256`）。 |
| `TS_SCHED_PREFIX_CACHE` | `0` 关闭跨请求的块级哈希前缀共享。 |
| `TS_SCHED_DECODE_QUANTUM` | 在允许切换序列前的 token 数（默认与 block size 相同）。 |
| `TS_QWEN35_BATCHED` | 设为 `0` 强制 Qwen 3.5/3.6 走旧的按序列 KV-swap 路径（默认走批处理 / 分页）。`--no-continuous-batching` 也会隐式关闭。 |
| `TS_QWEN35_BATCHED_GDN_NATIVE` | 在 Qwen 3.5/3.6 批处理路径中使用原生批处理 GatedDeltaNet 内核。 |
| `TS_GEMMA4_BATCHED` | 设为 `0` 可强制 Gemma 4 走旧的单序列 KV 交换路径（默认走批处理 / 分页）。 |
| `TS_GPTOSS_BATCHED` | 设为 `0` 强制 GPT OSS 走旧的按序列 KV-swap 路径（默认走批处理 / 分页）。 |
| `TS_GPTOSS_PAGED_ATTN_MANAGED` | 在 GPT OSS 批处理路径中使用托管 (C#) 的带 sinks 分页注意力内核。 |
| `TS_NEMOTRON_BATCHED` | 设为 `0` 强制 Nemotron-H 走旧的按序列 KV-swap 路径（默认走批处理 / 分页）。 |
| `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE` | 在 Nemotron-H 批处理路径中使用原生 Mamba2 批处理步骤内核。 |
| `TS_PAGED_ATTN_KERNEL` | `Mistral3Model.BatchedForward` 选择的分页注意力派发内核：`native`（默认）、`tensor`（基于 C# Tensor）或 `managed`（纯 C# 标量）。 |
| `TS_MLX_PIPELINED_DECODE` | 默认 `1`，当请求为贪心采样、没有 stop 序列且模型支持 device-side argmax / 下一 token embedding 查找时，在 MLX 后端启用流水化贪心 decode。设为 `0` 可关闭。仅 CLI。 |
| `TS_MLX_MLOCK_GGUF` | 默认 `1`，通过 `mlock(2)` 把 GGUF mmap 区域钉在物理内存，避免前向之间被换出。设为 `0` 关闭（适用于进程 `memlock` rlimit 太低、或希望让 OS 自行管理分页的情况）。仅 MLX 后端。 |
| `TS_MLX_FUSED_KV_WRITE` | 默认 `1`，使用单次多维 `slice_update` 写入每个 token 的 KV block。设为 `0` 回退到按 head 的循环（A/B 测试 / 隔离回归用）。 |
| `TS_MLX_BATCHED_MOE_DECODE` | 默认 `1`，将 Qwen 3.5/3.6 MoE 解码时每专家的 K 次 dispatch 合并为每种（gate / up / down）一次批处理 dispatch。在显存紧张的机器上可设为 `0` 关闭（可节省堆叠权重 slab 带来的近一倍权重显存占用）。 |
| `TS_MLX_MOE_FUSED_GATE_UP_SILU` | 默认 `1`，把批处理 MoE 解码的 gate matmul + up matmul + SiLUMul 融合到一个 Metal kernel。设为 `0` 用于和旧的 3-dispatch 路径做 A/B 对比。 |
| `TS_MLX_DEVICE_ROUTER` | 默认 `1`，让 MoE router 的 top-K + softmax 留在 device 上，避免每个 MoE 层一次主机同步（在 Qwen3.6-35B-A3B 上约能节省每 token ~60 次同步）。设为 `0` 可关闭；不满足前置条件时会自动回退到 host routing。 |
| `TS_MLX_LOG_MEMORY_POLICY` | 默认 `1`，加载时打印一行 MLX 内存策略信息（wired limit、GGUF mlock 状态、分配器上限等）。设为 `0` 静默。 |
| `TS_MLX_MEMORY_LIMIT_MB` / `TS_MLX_CACHE_LIMIT_MB` / `TS_MLX_WIRED_LIMIT_MB` | 覆盖 MLX 分配器硬上限 / 空闲缓冲池上限 / wired 缓冲上限（兆字节）。默认值会根据宿主机统一内存大小派生。 |
| `TS_MLX_EVAL_EVERY_N_LAYERS` / `TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS` | 解码时定期触发 `mlx_async_eval` 的层间隔，用于让 GPU 计算和宿主端排队重叠。Gemma 4 通过 `TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS` 默认每 4 层一次；Qwen 3 / Qwen 3.5 / Nemotron-H 通过 `TS_MLX_EVAL_EVERY_N_LAYERS` 默认每 16 层一次。支持处可设为 `0` 关闭。 |
| `TENSORSHARP_MLX_LIBRARY` / `TENSORSHARP_MLX_LIBRARY_DIR` | 覆盖 `--backend mlx` 时 `libmlxc` 的搜索路径。 |

**DiffusionGemma 专属调优变量**

| 变量 | 说明 |
|---|---|
| `DIFFUSION_NO_SC` | 设为 `1` 关闭 self-conditioning。默认开启。 |
| `DIFFUSION_SC_TOPK` | 实验用 self-conditioning top-K 截断（默认：`32`）。 |
| `DIFFUSION_NO_PKV` | 设为 `1` 关闭 device-glue 后端上的 prompt-KV 缓存。支持处默认开启。 |
| `DIFFUSION_NO_FUSED_DECODE` | 设为 `1` 关闭 GGML 融合整模型 diffusion decode，回退到逐算子 / 逐层 diffusion decode。 |
| `DIFFUSION_NO_FUSED_LMHEAD_TAIL` | 设为 `1` 关闭融合 output-norm + lm-head + softcap 尾部。 |
| `DIFFUSION_BATCHED_FORWARD` | 设为 `1` 后，对活跃 diffusion canvas 使用真正的 `DecodeCanvasBatched`；默认按请求时间片执行更快的融合单 canvas 路径。 |
| `DIFFUSION_LMHEAD_BATCH_CAP_MB` | diffusion lm-head logits 批处理内存上限，超过后回退到按序列 lm-head（默认：`300`）。 |
| `DIFFUSION_PROFILE` / `DIFFUSION_STEPTIME` / `DIFFUSION_FUSED_DEBUG` | 开发诊断用的 diffusion 分段计时与融合 kernel 调试日志开关。 |

采样参数的优先级（从高到低）：

1. API 请求 JSON 中的字段（如 `temperature`、`top_p`、`stop`）。
2. 服务端命令行参数（如 `--temperature`、`--top-p`、`--stop`）。
3. 上面列出的 `TENSORSHARP_*` 环境变量。
4. `SamplingConfig` 内置默认值（`temperature=1.0`、`top_k=0`、`top_p=1.0`、`min_p=0`、`repeat_penalty=1.0`、存在/频率惩罚均为 `0`、`seed=-1`、无停止序列）。

### 功能 × 环境变量矩阵

每个主要功能由哪些环境变量（以及对应的 CLI 参数）控制的速查矩阵。**加粗**的变量是该功能的开关；其余的是该功能默认启用后的调优参数。

#### 连续批处理 & 分页 KV 缓存

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| 连续批处理引擎（`InferenceEngine` + 调度器） | 在 `TensorSharp.Server` 中默认启用 | `TS_SCHED_DISABLE_BATCHED=1` 强制按序列回退 | `--no-continuous-batching` / `--continuous-batching` |
| 旧的按会话分页 KV 管理器 | 已从服务端请求路径移除 | `TS_KV_PAGED_CACHE`（`0` / `1`）、`TS_KV_BLOCK_SIZE` 仅为兼容 / 独立测试保留 | `--paged-kv` / `--no-paged-kv`、`--paged-kv-block-size N` |
| 旧的分页 KV SSD 冷层溢出 | 关闭 | `TS_KV_CACHE_MAX_RAM_MB`、`TS_KV_CACHE_SSD_DIR`、`TS_KV_CACHE_MAX_SSD_MB` | `--paged-kv-ram-mb`、`--paged-kv-ssd-dir`、`--paged-kv-ssd-mb` |
| 旧的分页 KV 块量化（TurboQuantKvCodec） | 关闭（`0` = 透传） | `TS_KV_PAGED_QUANT_BITS`（`0` / `4` / `8`） | `--paged-kv-quant-bits` |
| 跨请求的块级哈希前缀共享 | 启用 | `TS_SCHED_PREFIX_CACHE=0` 关闭 | — |
| 调度器调优（每步 token 预算、最大同时序列数、prefill 分块、块池大小、decode quantum） | 引擎默认 | `TS_SCHED_MAX_BATCHED_TOKENS`、`TS_SCHED_MAX_RUNNING_SEQS`、`TS_SCHED_PREFILL_CHUNK`、`TS_SCHED_NUM_BLOCKS`、`TS_SCHED_BLOCK_SIZE`、`TS_SCHED_DECODE_QUANTUM` | — |

#### 按模型的批处理 / 分页前向（`IBatchedPagedModel.ForwardBatch`）

| 模型 | 默认状态 | 切换默认的环境变量 | 原生内核子开关 |
|---|---|---|---|
| Mistral 3 | 启用 | — | `TS_PAGED_ATTN_KERNEL` = `native`（默认）/ `tensor` / `managed` |
| Gemma 4 | 启用 | `TS_GEMMA4_BATCHED=0` 强制走旧的按序列路径 | — |
| Qwen 3 | 启用（参考移植） | — | — |
| Qwen 3.5 / 3.6 系列 | 启用 | `TS_QWEN35_BATCHED=0` 强制走旧的按序列路径（或 `--no-continuous-batching`） | `TS_QWEN35_BATCHED_GDN_NATIVE=1` 启用原生批处理 GDN 内核；`FUSED_ATTN_LAYER_MIN_SEQ_LEN=N` 覆盖融合注意力启用阈值（默认 4096） |
| GPT OSS | 启用 | `TS_GPTOSS_BATCHED=0` 强制走旧的按序列路径 | `TS_GPTOSS_PAGED_ATTN_MANAGED=1` 强制使用托管 (C#) sinks softmax，而非原生带 sinks 的分页注意力内核 |
| Nemotron-H | 启用 | `TS_NEMOTRON_BATCHED=0` 强制走旧的按序列路径 | `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1` 启用原生批处理 Mamba2 步（NEON SIMD + GCD 并行） |
| Gemma 3 | 未实现（走按序列回退） | — | — |
| DiffusionGemma | Web UI 路径使用独立 diffusion 调度器；不是 `IBatchedPagedModel` 自回归路径 | `DIFFUSION_MAX_BATCH`、`DIFFUSION_STEPS` | `DIFFUSION_BATCHED_FORWARD=1` 启用真正的批处理 canvas decode；GGML 融合 decode 默认开启，可用 `DIFFUSION_NO_FUSED_DECODE=1` 关闭 |

#### 后端

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| 默认计算后端 | `ggml_metal`（macOS）、`ggml_cpu`（Windows/Linux） | `BACKEND` | `--backend` |
| MLX 后端库查找 | 优先探测应用目录 | `TENSORSHARP_MLX_LIBRARY`（`libmlxc` 完整路径）、`TENSORSHARP_MLX_LIBRARY_DIR`（目录） | — |
| MLX 流水化贪心 decode（仅 CLI） | 满足条件时启用 | `TS_MLX_PIPELINED_DECODE=0` 关闭 | — |
| 使用 `mlock(2)` 钉住 GGUF mmap，使权重常驻 | 启用 | `TS_MLX_MLOCK_GGUF=0` 关闭 | — |
| MLX 融合多维 KV 写入（每个 cache block 单次 `slice_update`） | 启用 | `TS_MLX_FUSED_KV_WRITE=0` 回退到按 head 循环 | — |
| MLX 批处理 MoE 解码（Qwen 3.5/3.6 MoE） | 启用 | `TS_MLX_BATCHED_MOE_DECODE=0` 走旧的按专家路径 | — |
| MLX MoE gate+up+SiLUMul 融合 Metal kernel | 启用 | `TS_MLX_MOE_FUSED_GATE_UP_SILU=0` 走旧的 3-dispatch | — |
| MLX 设备端 MoE router top-K + softmax | 满足条件时启用 | `TS_MLX_DEVICE_ROUTER=0` 关闭 | — |
| MLX 解码层边界 `async_eval` 间隔 | Gemma 4：每 4 层；Qwen / Nemotron：每 16 层 | `TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS=N` 或 `TS_MLX_EVAL_EVERY_N_LAYERS=N`（支持处 `0` = 关闭） | — |
| MLX 分配器上限（内存 / 缓存 / wired buffer） | 按宿主机派生 | `TS_MLX_MEMORY_LIMIT_MB`、`TS_MLX_CACHE_LIMIT_MB`、`TS_MLX_WIRED_LIMIT_MB` | — |
| 加载时打印一行 MLX 内存策略信息 | 启用 | `TS_MLX_LOG_MEMORY_POLICY=0` 静默 | — |

#### 采样默认值（仅服务端）

这些变量用于填充请求体未提供的字段；请求 JSON 字段始终优先于 CLI 参数，CLI 参数优先于环境变量。

| 采样字段 | 环境变量 | CLI 等价参数 |
|---|---|---|
| `temperature` | `TENSORSHARP_TEMPERATURE` | `--temperature` |
| `top_k` | `TENSORSHARP_TOP_K` | `--top-k` |
| `top_p` | `TENSORSHARP_TOP_P` | `--top-p` |
| `min_p` | `TENSORSHARP_MIN_P` | `--min-p` |
| `repeat_penalty` | `TENSORSHARP_REPEAT_PENALTY` | `--repeat-penalty` |
| `presence_penalty` | `TENSORSHARP_PRESENCE_PENALTY` | `--presence-penalty` |
| `frequency_penalty` | `TENSORSHARP_FREQUENCY_PENALTY` | `--frequency-penalty` |
| `seed` | `TENSORSHARP_SEED` | `--seed` |
| 最大 token 数 | `MAX_TOKENS` | `--max-tokens` |
| 停止序列 | —（仅 CLI / 请求体支持） | `--stop`（可重复） |

#### 服务托管与上传（仅服务端）

| 功能 | 默认 | 环境变量 |
|---|---|---|
| ASP.NET Core 监听 | `http://0.0.0.0:5000` | 固定在 `Program.cs`；Docker Space 镜像用 `APP_PORT` 构建参数改写 |
| 没有分词器时的纯文本上传字符上限 | 8000 字符 | `MAX_TEXT_FILE_CHARS` |
| 视频帧抽取 | 1 fps（基于时间，不限制） | `VIDEO_SAMPLE_FPS`、`VIDEO_MAX_FRAMES` |
| DiffusionGemma Web UI 去噪 | 48 步，最大 batch 2 | `DIFFUSION_STEPS`、`DIFFUSION_MAX_BATCH` |

#### 日志（服务端 + CLI）

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| 控制台 + 文件日志最低级别 | `Information` | `TENSORSHARP_LOG_LEVEL` | `--log-level` |
| 文件日志输出目录 | `<binDir>/logs` | `TENSORSHARP_LOG_DIR` | `--log-dir` |
| 文件日志开关 | 启用 | `TENSORSHARP_LOG_FILE=0` 关闭 | `--log-file 0\|1` |
| 控制台日志开关 | 启用 | — | `--log-console 0\|1`（仅 CLI） |

#### 原生构建（仅编译期）

下列变量由 `build-linux.sh` / `build-windows.ps1` / `dotnet build` 时自动构建 `TensorSharp.GGML.Native` 的脚本读取，不在运行时生效。

| 功能 | 默认 | 环境变量 | 构建脚本参数 |
|---|---|---|---|
| 在原生构建中启用 GGML CUDA | 根据工具链自动检测 | `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON` | `--cuda` / `--no-cuda` |
| 精简 `CMAKE_CUDA_ARCHITECTURES` 列表 | 根据可见 GPU 自动检测 | `TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES` | `--cuda-arch='86-real;89-real'` |
| 原生构建并行度上限 | 保守自动上限 | `TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL` | — |

### 服务端日志

每一轮 chat / generate 请求开始与结束时，服务都会打出一条结构化的
Information 级别日志。只需对日志文件做一次 grep 即可还原完整的请求—响应审计
轨迹，无需重放任何流量。

| 事件 ID | 触发位置 | 字段 |
|---|---|---|
| `ChatStarted`（1500） | `chat.start`、`generate.start` 以及各协议的请求横幅 | 采样配置、消息与附件计数、`userInput=`（完整的最近一条用户消息）、`fullInput=`（本轮请求中**全部**消息的 JSON 数组：system 提示 + 历史 user/assistant + 本轮新消息 + 附件计数），`/api/generate` 则为完整 prompt |
| `ChatCompleted`（1502） | `chat.complete`、`generate.complete` | token 数、KV 缓存复用（`kvReused`、`kvReusePercent`）、TTFT、耗时、吞吐、终止原因，以及完整的模型原始输出（思维链 + 结果） |
| `ChatAborted`（1503） | 客户端中途断开 | 已生成的部分输出、当时的 KV 复用占比 |
| `KvCacheReusePlan`（1510） | 每次前缀复用判断 | `Debug` 级的细粒度分支信息（精确匹配 / 部分复用 / 完整重置） |
| `HttpRequestStarted/Completed`（1100/1101） | 每个 HTTP 请求 | method、path、远端 IP、状态码、耗时；`/api/queue/status` 被降级到 `Debug`，避免 UI 高频轮询淹没每轮日志 |

模型原始输出会保留 `<think>...</think>`、`<|channel|>analysis` 等内联标记，因
此单条日志即可看到推理过程与最终用户可见结果。配合 `chat.start` 上的
`fullInput=` 字段，每一轮的请求输入和模型原始输出都可以**只**通过日志文件完
整复现。长上传或长思维链可能产生数 KB 的日志；如需控制噪声可调高级别（例如
`TENSORSHARP_LOG_LEVEL=Warning`），仍会保留启动横幅与错误日志。

`fullInput` 字段示例（为便于阅读做了缩进，实际日志为单行）：

```json
[
  {"role":"system","content":"你是一个有帮助的助手。"},
  {"role":"user","content":"世界上最高的山是哪座？"},
  {"role":"assistant","content":"珠穆朗玛峰。"},
  {"role":"user","content":"它有多高？","images":1}
]
```

同样的 KV 缓存复用统计会通过所有 API 透出：

- **Web UI SSE**（`POST /api/chat`） —— `done` 事件携带 `promptTokens`、`kvReusedTokens`、`kvReusePercent`。
- **Ollama NDJSON**（`POST /api/generate`、`POST /api/chat/ollama`） —— 流式末尾 chunk 与非流式响应均携带 `prompt_cache_hit_tokens`（int）和 `prompt_cache_hit_ratio`（0..1）。
- **OpenAI**（`POST /v1/chat/completions`） —— `usage` 块携带 `prompt_tokens_details.cached_tokens`，与 OpenAI 标准扩展一致，现有 SDK 可直接读取。

Web UI 中每条助手消息下方的统计行也会展示命中率（例如 `187 tokens · 2.1s · 87.2 tok/s · KV 420/512 (82%)`）。

### HTTP API

TensorSharp.Server 暴露三种 API 风格。完整文档及 curl/Python 示例见 [API_EXAMPLES.md](TensorSharp.Server/API_EXAMPLES.md)。

**兼容 Ollama 的 API：**

```bash
# 列出模型
curl http://localhost:5000/api/tags

# 文本生成
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "prompt": "Hello!", "stream": false}'

# 聊天
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "Hi"}], "stream": false}'

# 启用思维链模式的聊天
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "计算 17*23"}], "think": true, "stream": false}'

# 带工具调用的聊天
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "天气怎么样？"}], "tools": [{"function": {"name": "get_weather", "description": "获取当前天气", "parameters": {"properties": {"city": {"type": "string"}}, "required": ["city"]}}}], "stream": false}'
```

**兼容 OpenAI 的 API：**

```bash
# Chat completions
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'

# 结构化输出（OpenAI response_format）
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it-Q8_0.gguf",
    "messages": [{"role": "user", "content": "从“Paris, France”中提取城市与国家。"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "location_extraction",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
            "confidence": {"type": ["string", "null"]}
          },
          "required": ["city", "country", "confidence"],
          "additionalProperties": false
        }
      }
    }
  }'
```

**OpenAI Python SDK：**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="gemma-4-E4B-it-Q8_0.gguf",
    messages=[{"role": "user", "content": "What is 2+3?"}],
    max_tokens=50
)
print(response.choices[0].message.content)
```

**队列状态：**

```bash
curl http://localhost:5000/api/queue/status
# {"busy":false,"pending_requests":0,"total_processed":42}
```

## 思维链 / 推理模式

支持思维链模式的模型（Qwen 3、Qwen 3.5/3.6-family、Gemma 4、GPT OSS、Nemotron-H）可以在生成最终答案之前产出结构化的思维链推理内容。思维内容与主要回复分开，客户端可选择显示或隐藏。

- **Qwen 3 / Qwen 3.5/3.6-family / Nemotron-H：** 使用 `<think>...</think>` 标签
- **Gemma 4：** 使用 `<|channel>thought\n...<channel|>` 标签
- **GPT OSS：** 使用 Harmony 格式，以 `<|channel|>analysis` 标记思维过程，以 `<|channel|>final` 标记最终回复

通过 `--think`（控制台）、`"think": true`（Ollama API）或 Web 界面中的思维链开关启用。

## 工具调用 / 函数调用

模型可以调用用户定义的工具并参与多轮工具调用对话。将工具定义为 JSON 格式，通过 `--tools`（控制台）或 API 中的 `tools` 参数传入。

各架构使用各自的工具调用格式：

- **Qwen 3 / Qwen 3.5/3.6-family / Nemotron-H：** `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- **Gemma 4：** `<|tool_call>call:function_name{args}<tool_call|>`
- **GPT OSS（Harmony）：** 工具以 TypeScript namespace 形式声明在 developer 消息中，调用通过 commentary channel 输出：`<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}<|call|>`

输出解析器（`OutputParser.cs`）会自动从模型原始输出中提取工具调用，与架构无关。

## 多模态支持

### Gemma 4

Gemma 4 模型支持图像、视频和音频输入。将多模态投影器（`gemma-4-mmproj-F16.gguf`）放在与模型文件相同目录即可自动加载。

- **图像：** PNG、JPEG、HEIC/HEIF
- **视频：** MP4（使用 OpenCV 以 1 fps 基于时间抽帧；可通过 `VIDEO_SAMPLE_FPS` / `VIDEO_MAX_FRAMES` 调整）
- **音频：** WAV（16kHz 单声道）、MP3、OGG Vorbis

### Gemma 3

Gemma 3 支持 PNG、JPEG 与 HEIC/HEIF 图像输入。将其多模态投影器（`mmproj-gemma3-4b-f16.gguf`）放在模型文件相同目录即可自动加载。

### Qwen 3.5 / 3.6 family

所有 Qwen 3.5/3.6-family 变体（`qwen35`、`qwen35moe` 与 `qwen3next`）共用同一个 `Qwen35Model` 实现。图像输入通过支持动态分辨率的 `Qwen35VisionEncoder` 处理；将投影器（`Qwen3.5-mmproj-F16.gguf`）放到模型 GGUF 同一目录下即可自动加载。MoE 变体（例如 Qwen3.5-35B-A3B，以及使用同一架构标识的 Qwen3.6-35B-A3B GGUF）在 decode 时还会启用融合的 `MoEExpertsSwiGLUResidual` GGML 内核，将所有被选中的专家、可选的 shared expert 与残差加法合并到一次 GPU 计算图调度中执行。

### Mistral 3

Mistral 3 通过 Pixtral 视觉编码器支持图像输入。将多模态投影器（`mistral3-mmproj.gguf`）放在与模型文件相同目录即可自动加载。

- **图像：** PNG、JPEG、HEIC/HEIF

### Nemotron-H（Omni 发行版）

Nemotron Omni 发行版加入了 RADIO / v2_vl ViT 图像编码器。通过 `--mmproj` 传入对应的多模态投影器（例如 `nvidia_Nemotron-H-Omni-mmproj.gguf`）即可启用；语言模型 GGUF 不变。图像 token 在 `<image>` 占位符处插入，并由多模态注入器自动展开为 `<img>` + N 个 tile token + `</img>`。

- **图像：** PNG、JPEG、HEIC/HEIF
- **音频：** 聊天模板会为每个上传的音频文件发出一个 `<so_embedding>` token，CLI 仍会运行 Parakeet 风格 log-mel 预处理器以验证管线，但真正的音频推理需要尚未在公开 GGUF 中发布的 Parakeet 音频 mmproj。

## 架构说明

TensorSharp 采用分层系统结构：

1. **TensorSharp.Core** 提供核心 `Tensor` 类型、存储抽象和可扩展的操作注册表（`Ops`）。CPU 实现使用 `System.Numerics.Vectors` 进行 SIMD 加速。

2. **TensorSharp.Runtime** 负责运行时契约与通用服务：GGUF 解析、分词（SentencePiece / BPE）、聊天模板渲染、可配置 token 采样、输出解析、分页 KV 缓存（`Runtime/Paged/*`）、连续批处理调度器 / 引擎（`Runtime/Scheduling/*`）、`IKvBlockCodec` 接口及其 `TurboQuantKvCodec` Q4/Q8 实现，以及 `IModelArchitecture`、`IBatchedPagedModel`、`IPromptRenderer`、`IOutputProtocolParser`、`IMultimodalInjector`、`IKVCachePolicy`、`IBackendExecutionPlan` 等抽象。

3. **TensorSharp.Models** 实现 `ModelBase` 以及各具体模型架构和多模态辅助组件（Gemma 3/4、DiffusionGemma、Qwen 3/3.5、GPT OSS、Nemotron-H、Mistral 3）。自回归架构提供旧的单序列前向，多数架构还提供面向连续批处理的 `IBatchedPagedModel.ForwardBatch` 实现（`<Family>Model.BatchedForward.cs`）。DiffusionGemma 刻意不同：它不支持 `Forward()`，生成必须通过 `DiffusionGemmaSampler` 在固定长度 canvas 上迭代去噪。模型通过 `ModelBase.Create()` 加载，并依据 GGUF 元数据自动识别架构。

4. **TensorSharp.Backends.GGML** 通过原生 C++ 桥接库（`libGgmlOps` / `GgmlOps.dll`）注册同名操作的加速实现，并链接 [ggml](https://github.com/ggml-org/ggml)。在 macOS 上可提供 Metal GPU 计算，在 Windows/Linux 上可启用面向 NVIDIA GPU 的 GGML CUDA。除原生量化 matmul（Q4_K_M、Q8_0 等，无需反量化到 FP32）外，还提供分页注意力（`TSGgml_PagedAttentionForward`，含 / 不含注意力 sinks 两种版本）以及架构特定的批处理内核（Mamba2、GatedDeltaNet）。

5. **TensorSharp.Backends.Cuda** 是 Direct CUDA 路径。它使用 CUDA Driver API 管理设备、上下文与存储，用 cuBLAS 执行 Float32 GEMM，用 PTX 内核覆盖热点标量与 Transformer 辅助算子，并对尚未实现的原生内核使用 CPU 回退。

6. **TensorSharp.Backends.MLX** 是 Apple Silicon 上的 MLX 路径。它封装 [mlx-c](https://github.com/ml-explore/mlx-c)（`libmlxc`），提供分配器、存储、异步 worker 派发、量化 / 融合 / 编译内核、MoE 专家 offload，以及对未实现算子的 CPU 回退层。

7. **TensorSharp.Server** 是 HTTP / 应用层，提供兼容 Ollama 与 OpenAI 的 REST API、浏览器聊天 UI、上传处理；其中 `InferenceEngineHost` 持有自回归模型的连续批处理引擎，`DiffusionBatchScheduler` 处理 DiffusionGemma 的 Web UI 轮次，旧的队列状态接口保留作为向后兼容。

8. **TensorSharp.Cli** 是控制台 / 应用层，用于本地 prompt 运行、多模态实验、prompt 检查、JSONL 批处理、交互式 REPL 与内置的 prefill / decode 基准。

### 性能优化

下表是跨架构汇总；[`docs/models/`](docs/models/README_zh-cn.md) 里每个模型卡片会在上下文中走一遍同样的内核，包含具体派发的 GGML 图与触发融合路径的条件。

- **融合 GPU decode**（Gemma 4）：在 Metal 上将所有 Transformer 层合并为单次 GGML 计算图调度，将每个 token 的 CPU-GPU 往返从数百次降低到一次。相较逐算子调度约提升 2.6 倍。
- **融合 GPU prefill**（Gemma 4）：对于密集（非 MoE、非 KV 共享、无 PLE/多模态）层，`Gemma4LayerPrefill` 将整个 Transformer 块（RMSNorm + QKV + QK-norm + RoPE + 注意力 + 输出投影 + post-attn norm + GeGLU FFN + post-FFN norm + 残差 + 层缩放因子）合并为 prefill 期间每层一次的 GGML 计算图调度，将融合方法从单 token decode 扩展到多 token prefill。
- **分块 prefill**（Gemma 4）：长提示被拆分为有界的分块（2 倍滑动窗口，最大 2048 tokens），以避免 SWA 层上 O(n²) 的注意力分数张量。分块在纯文本（无多模态嵌入）时自动应用，确保每个分块在 SWA 窗口预算内。
- **整模型原生 decode**（Qwen 3）：所有 Transformer 层在一次原生调用（`TransformerModelDecode`）中完成，每层权重指针在加载阶段预解析并缓存，从 decode 热点路径中移除托管循环开销。
- **融合 Qwen 3.5/3.6-family attention 层 decode**：单次 GGML 计算图为每个 FullAttention 层完成 RMSNorm + 融合 QKV + Q/gate 反交错 + 每头 QK norm + RoPE + KV 缓存追加 + flash attention + sigmoid 门控混合 + 输出投影 + 残差加法。替换了原本每层 ~2 次独立 GGML 调用与 ~6 个小型 CPU/GPU 同步点。当缓存序列长度超过 4096 token 时启用（可通过 `FUSED_ATTN_LAYER_MIN_SEQ_LEN=N` 覆盖）。
- **融合 prefill 注意力**（Qwen 3.5/3.6-family）：`FusedPrefillAttention` 将 Q*K^T、因果掩码、softmax 和 *V 合并为 prefill 期间每个注意力层一次的 GGML 计算图调度，消除了每个注意力层约 5 次独立的 C# 到 GGML 往返。同时支持初始 prefill 和带有已有 KV 缓存条目的续接。
- **融合输出投影 + FFN**（Qwen 3.5/3.6-family）：对于 FullAttention 和 GatedDeltaNet 中的 dense FFN 层，`FusedOutProjFFN` 将输出投影、残差加法、post-attention RMSNorm 以及完整的 SwiGLU FFN（gate_up matmul + SiLU + down matmul + 残差加法）合并为单次 GGML 计算图调度，将每层 2 次 GPU 往返减少为 1 次。
- **融合输出投影 + 归一化 + 路由器**（Qwen 3.5/3.6-family MoE）：`FusedOutProjNormRouter` 将 GatedDeltaNet 输出投影、残差加法、post-attention RMSNorm 和 MoE 路由器投影合并为一次调度。预计算的路由器 logits 随后由批量 MoE 内核直接消费，消除了每个 MoE 层的独立路由器调度。
- **融合视觉编码器**（Qwen 3.5/3.6-family）：`FusedVisionAttention` 将 LayerNorm + QKV + 偏置 + 2D RoPE + 缩放点积注意力 + 输出投影 + 偏置 + 残差合并为一次 GGML 计算图调度（~8 个算子 → 1）。`FusedVisionMLP` 将 LayerNorm + up + 偏置 + GELU + down + 偏置 + 残差合并为一次调度（7 个算子 → 1）。两者结合将每个编码器块的 GPU 往返从约 15 次减少到 2 次。
- **融合权重投影**：Q/K/V 投影融合为单次 QKV matmul；gate 与 up 投影融合为单次 gate_up matmul。
- **原生量化计算**：量化权重（Q4_K_M、Q6_K、Q8_0、IQ2_XXS、MXFP4 等）直接参与 matmul，无需展开为 FP32，节省内存与带宽。批量 `AddmmQuantBatch` 内核可在一次调度内完成对同一量化权重块的多个子矩阵 matmul。
- **Direct CUDA 内核**：`cuda` 后端加速 fill/copy、unary ops、融合激活、RMSNorm、softmax、index select、因果掩码、RoPE/RoPEEx、cuBLAS GEMM，以及受支持的量化 matmul/get-rows；未覆盖算子会安全回退。
- **批量 GPU MoE**：`MoEExpertsSwiGLUResidual`（Qwen 3.5/3.6-family）和 `MoEExpertsForward`（Nemotron-H）将每个 MoE 层中所有被选中的专家——以及 Qwen 3.5/3.6-family 中可选的 shared expert 与残差加法——合并为一次 GGML 计算图调度。
- **基于 GEMM 的视觉 patch embedding**（Qwen 3.5/3.6-family）：将 patch embedding 重构为并行 im2col + 矩阵乘法，把单线程标量五重嵌套循环替换为可在 GPU 上加速的 matmul。
- **并行化 Q/gate 反交错**（Qwen 3.5/3.6-family）：FullAttention prefill 中的 Q + sigmoid-gate 反交错按 token 并行化，长 prompt 时可随 CPU 核心数线性扩展。
- **优化后的纯 C# CPU 路径**：托管 GEMM 快速路径和连续 Float32 内核加速了 decode、softmax、RMSNorm、RoPE、融合激活等热点路径，同时在 CPU 加载时保持量化 GGUF 权重压缩状态。
- **环形 KV 缓存**：滑动窗口注意力层使用固定大小环形缓冲区，使内存占用不随序列长度增长。
- **KV 缓存前缀复用**：多轮对话会复用各轮之间最长的匹配 token 前缀。对 SWA 模型，截断会自动按滑动窗口大小回退，使后缀部分可以重建 SWA 上下文。
- **分页 KV 缓存 & 块哈希前缀共享**：连续批处理引擎把 KV 切分成固定大小的块，对每个写满的块做内容哈希，并在并发 / 历史请求间共享。尚未实现 `IBatchedPagedModel` 的模型仍会走同一引擎内隔离的按序列 KV-swap 回退路径。
- **原生分页注意力内核**：`TSGgml_PagedAttentionForward`（及面向 GPT OSS 的 `WithSinks` 变体）在 C++ 中按序列从分页缓冲区聚合 K/V，按序列构建小型 GGML 图，并派发 `ggml_flash_attn_ext`——也就是旧的单序列路径所使用的同一融合 Metal/CUDA flash 注意力内核。在 Ministral-3-14B 长上下文（4×~800 tokens）上比旧的按序列 GGML 路径**快 ~21%**。
- **批处理 / 分页前向**：Mistral 3、Gemma 4、GPT OSS、Qwen 3.5/3.6（含 GatedDeltaNet 递归状态池）、Nemotron-H（含 Mamba2 递归状态池 + 原生批处理 Mamba2 内核）把 N 个序列打包到一次 `ForwardBatch` 调用中，每层执行一次批处理线性投影 matmul，通过 `slotMapping` 写入分页 K/V，并通过原生内核做按序列注意力。Gemma 4 批处理路径在 batch=8 短 prompt 下达到 **1.5×** 旧吞吐，在 4×800-token prompt 下达到 **1.6×**；Nemotron-H Mamba2 批处理在 Apple M4 Pro 上 batch=3 时达到 **3.95×**。详见 [docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)。
- **DiffusionGemma prompt-KV 缓存与融合去噪**：GPU 后端会在每个 block 中只对 `[prompt | canvas]` 的 prompt 部分预填充一次 K/V，并在去噪多步中复用；GGML 后端默认使用融合整模型 diffusion decode 与融合 lm-head tail。Web UI 通过 `DiffusionBatchScheduler` 在 block 边界批处理并发 diffusion 请求。
- **内核预热**：CLI 和 Server 在启动时运行一次微型前向传播，以预编译 GPU 内核（Metal pipeline state、CUDA JIT）并预热内存池，避免首次推理请求的冷启动延迟。
- **Prefill 缓存**（Gemma 4、Qwen 3.5/3.6-family）：逐 forward 传播的 SWA 掩码缓存（Gemma 4）、跨全局层的 NeoX RoPE cos/sin 查找表缓存（Gemma 4）、以及跨层的 RoPE 位置张量缓存（Gemma 4、Qwen 3.5/3.6-family），消除了 prefill 期间的冗余重复计算。
- **原地 QK RMSNorm**（Qwen 3.5/3.6-family）：逐头 QK 归一化通过 `View` 原地执行，避免了每层每个 Q/K 的一次张量分配与拷贝。

### 内存优化

- **零拷贝文件映射量化权重**（Direct CUDA、GGML CUDA、GGML Metal、GGML CPU）：GGUF 模型文件以内存映射方式打开，量化张量通过 host 指针缓冲区直接绑定到原生算子。这样省去了之前每张张量从磁盘复制到新分配原生堆缓冲区的过程——这一过程在 Apple Silicon 上会让大型量化模型的常驻内存几乎翻倍。例如，`Qwen3.5-35B-A3B-IQ2_XXS`（约 10 GB GGUF）在 Metal 后端的实际工作内存峰值从约 17 GB 降至约 7 GB。映射文件由操作系统的页缓存管理，必要时可换出，且在 Apple Silicon（统一内存）上不会带来推理性能损失。
- **最佳匹配内存池**：GGML 主机分配器使用 best-fit 而非 first-fit 在已池化块中检索可重用空间，避免把大块草稿内存交给小型中间张量请求，从而把工作集严格控制在合理范围内。
- **有界池保留量**：集成 GPU / CPU 内存池现在将单个保留块上限设为 64 MB，整池上限设为 32 块。结合 mmap 后的权重，可在快速复用短生命中间张量的同时限制峰值常驻内存。
- **高内存效率模型加载**：大张量直接流式加载到原生内存，避免中间托管分配。F32 权重与 norm 仍按需加载；量化权重在受支持的后端上通过 mmap 方式绑定。
- **可选 SSD 溢出的分页 KV 块池**：`PagedKvBlockStore` 保留了 RAM / SSD 分层块存储能力（`TS_KV_CACHE_MAX_RAM_MB`、`TS_KV_CACHE_SSD_DIR`、`TS_KV_CACHE_MAX_SSD_MB`），主要服务独立分页 KV 组件与后续扩展；服务端请求路径的活跃块由每个引擎的 `BlockPool` 统一管理。
- **KV 块编解码器**：`TurboQuantKvCodec`（Q4 或 Q8）可压缩分页块，以微小精度成本换取每块带宽与内存占用减半 / 减为四分之一。带递归状态的模型会自动回退到 passthrough。

## 性能数据

### 内部回归基线

在 `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf`（磁盘约 10 GB；256 个路由专家，每 token 激活 8 个；12 个全注意力层 + 30 个 GatedDeltaNet 循环层）上，Apple M4 Pro（24 GB 统一内存）的参考结果：

| 指标 | 优化前（`v1` 基线） | 优化后（当前分支） | 变化 |
|---|---|---|---|
| 进程峰值内存占用 | ~17 GB | **~8 GB** | **-52%** |
| TensorSharp.Server 加载后常驻内存 | ~20 GB | **~8 GB** | **-60%** |
| Decode 吞吐（warm，prefill 256 / decode 64，M4 Pro） | ~3.8 tok/s | **~10.8 tok/s** | **+2.85x** |
| Decode 延迟（warm，prefill 256 / decode 64，M4 Pro） | ~264 ms/token | **~92 ms/token** | **-65%** |

复现命令：

```bash
./TensorSharp.Cli --model Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf --backend ggml_metal \
    --benchmark --bench-prefill 256 --bench-decode 64 --bench-runs 3
```

内存节省主要来自不再把 GGUF 模型文件复制到独立的原生堆缓冲区（现在文件是以 mmap 方式零拷贝绑定到 Metal command buffer 中）。Decode 吞吐量提升在很大程度上也是消除约 10 GB 重复工作集的副效应——这部分重复占用此前会在仅有 24 GB 或更少物理内存的机器上触发系统级内存压力。

### 跨引擎推理矩阵

在相同磁盘 GGUF 文件上对 TensorSharp、llama.cpp 与 Ollama 做苹果对苹果的对比（当前覆盖 Gemma 4 E4B Q8_0，包含文本 / 合成 prefill / 图像 / 音频 / 视频任务，并对 `f32`、`f16`、`q8_0` 三种 KV cache dtype 做扫描），见 [`docs/inference_benchmark_matrix_zh-cn.md`](docs/inference_benchmark_matrix_zh-cn.md)。驱动脚本位于 `benchmarks/inference_matrix/scripts/`；运行矩阵后，每格原始 JSON 会生成到 `benchmarks/inference_matrix/results/`。

## 测试

### 单元测试（xUnit）

`InferenceWeb.Tests` 覆盖无需启动服务的进程内行为：托管量化算子、可用 CUDA 设备上的 Direct CUDA 后端内核、可用 MLX 时的 MLX 后端内核、分页 KV 缓存调度（`ContinuousBatchSchedulerTests`、`PagedKvCacheTests`、`PagedKvCacheCodecTests`）、批处理执行器正确性（`BatchedExecutorTests`）、按模型批处理前向与旧路径的一致性（`Qwen35BatchedCorrectnessTests`、`Mistral3BatchedForwardTests`、`Gemma4BatchedForwardTests`、`GptOssBatchedCorrectnessTests`、`NemotronBatchedCorrectnessTests`）、DiffusionGemma 去噪 / prompt-KV / 批处理生成探针（`DiffusionGemmaTests`）、按模型批处理性能微基准（`*BatchedPerfBench.cs`）、`TurboQuantKvCodec` 编解码往返、prefill 分块、KV 缓存策略、KV 缓存 Prompt 渲染与多轮集成、聊天会话与 SessionManager 隔离、ModelService 历史跟踪、请求日志中间件与文件日志 Provider、图像预处理、媒体辅助逻辑、结构化输出校验、文本上传辅助、ModelService 上传日志、Web UI 聊天策略、模型上下文长度解析、可用后端发现，以及服务器 CLI 选项构造（`ServerOptionsBuilderTests`）。

```bash
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj
```

### 服务端集成测试

TensorSharp.Server 的集成测试位于 `TensorSharp.Server/testdata/`。测试覆盖所有三种 API 风格（Web UI SSE、Ollama、OpenAI）、多轮对话、思维链模式、工具调用、结构化输出、队列状态兼容、并发请求和中断支持。架构特定能力（思维链、工具调用）会自动检测，当前模型不支持时会自动跳过。

```bash
# 先启动 TensorSharp.Server，然后运行：
python3 TensorSharp.Server/testdata/test_multiturn.py
# 或
bash TensorSharp.Server/testdata/test_multiturn.sh
```

完整测试矩阵见 [TensorSharp.Server/testdata/README.md](TensorSharp.Server/testdata/README.md)。

### 推理矩阵运行器

`TensorSharp.TestMatrix` 是更大的 CLI 驱动覆盖工具，用于长时间模型 / 后端验证。它会发现 GGUF 文件，过滤不可用后端与不受支持的提示类型，运行 baseline 与环境变量 sweep，用每个 cell 一个 JSON 的形式保存结果，生成汇总 Markdown 报告，并可按需与每类主机的基线做回归对比。

```bash
dotnet build TensorSharp.TestMatrix/TensorSharp.TestMatrix.csproj -c Release
dotnet run --project TensorSharp.TestMatrix -c Release -- --dry-run
```

当前运行器契约见 [TensorSharp.TestMatrix/README_zh-cn.md](TensorSharp.TestMatrix/README_zh-cn.md) 与 [docs/env_var_feature_matrix_zh-cn.md](docs/env_var_feature_matrix_zh-cn.md)。

## 作者

Zhongkai Fu

## 许可证

详见 [LICENSE](LICENSE)。
