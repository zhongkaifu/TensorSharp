# TensorSharp 与 TensorAgent 书籍

[← 返回 TensorSharp](../README_zh-cn.md) | [English](BOOK.md) | [中文](BOOK_zh-cn.md)

## Building LLM Inference Engines and Agentic Runtimes from Scratch: Qwen Dense and MoE Models with TensorSharp and TensorAgent

<p align="center">
  <a href="https://www.amazon.com/dp/B0HJQ4VQ31">
    <img src="../website/assets/building-llm-inference-engines-cover.jpg" alt="Building LLM Inference Engines and Agentic Runtimes from Scratch: Qwen Dense and MoE Models with TensorSharp and TensorAgent" width="280">
  </a>
</p>

**Building LLM Inference Engines and Agentic Runtimes from Scratch** 提供一条 C# 实践路线：从张量、分词和模型加载，走向 Qwen 稠密/MoE 推理与受控智能体工作流。书中将注意力、专家路由、量化、缓存、GPU 执行、批处理、投机解码、分布式与多模态推理，与工具、技能、沙箱代码执行及 TensorSharp 和 TensorAgent 的桌面与移动端部署联系起来。

建议配合源码阅读，从 TensorSharp 的模型执行到 TensorAgent 的智能体体验，理解这些部分如何协作。下方配套文档提供当前实现细节及项目运行指南。

**[在 Amazon 购买本书](https://www.amazon.com/dp/B0HJQ4VQ31)**

### 配合仓库阅读

| 阅读目标 | 实时更新的配套内容 |
|---|---|
| 运行 TensorSharp | [TensorSharp 快速开始](../README_zh-cn.md#快速开始) |
| 查看 Qwen 模型实现 | [模型架构卡片](models/README_zh-cn.md) |
| 理解推理引擎架构 | [开发与架构指南](../DEVELOPMENT_zh-cn.md) |
| 了解智能体工具与运行时行为 | [Agent Skills 与智能体工作](agent_skills.md)（英文） |
| 构建并运行 TensorAgent | [TensorAgent 指南](../TensorAgent/README.md)（英文） |

---

## From Tensors to Tokens: Building a Multimodal LLM Inference Engine from Scratch with TensorSharp and Gemma 4 E4B

<p align="center">
  <a href="https://www.amazon.com/dp/B0H9P44QZZ">
    <img src="../website/assets/from-tensors-to-tokens-cover.jpg" alt="From Tensors to Tokens: Building a Multimodal LLM Inference Engine from Scratch with TensorSharp and Gemma 4 E4B" width="280">
  </a>
</p>

<p align="center"><strong><a href="https://www.amazon.com/dp/B0H9P44QZZ">在 Amazon 购买平装本</a></strong></p>

### 贯穿真实推理引擎的学习路线

TensorSharp 的参考文档按主题组织，方便快速查询模型、后端、命令或优化。**From Tensors to Tokens** 提供与之互补的叙事路线：以 TensorSharp 和 Gemma 4 E4B 为具体示例，从张量基础一路走向可运行的多模态 LLM 推理引擎。

这本书适合不满足于只会运行命令、还希望理解各部分如何协作的读者。你可以沿着输入到张量与 token、模型执行与生成，再到 CLI、浏览器界面和兼容 HTTP API 的完整数据路径学习；同时打开本仓库，查看实现、运行示例，并继续阅读项目的最新文档。

> 本书正文为英语。

### 出版信息

| 项目 | 信息 |
|---|---|
| 作者 | Zhongkai Fu |
| 装帧 | 平装本 |
| 页数 | 128 页 |
| 语言 | 英语 |
| 出版日期 | 2026 年 7 月 18 日 |
| ISBN-13 | 979-8187878703 |
| 示例模型 | Gemma 4 E4B |
| 配套项目 | TensorSharp |

### 这条学习路线连接什么

- **从张量基础到模型运算** —— 建立对推理所需数据结构与计算过程的实用理解。
- **从 token 到生成** —— 串起模型输入、embedding、前向执行、采样与自回归输出。
- **从文本到多模态推理** —— 以 Gemma 4 E4B 为例，将文本、图像、视频和音频路径放进同一个引擎。
- **从架构到加速** —— 把模型实现与 CPU/GPU 后端、GGUF 权重、量化执行和性能路径联系起来。
- **从引擎到应用** —— 理解同一套运行时如何形成命令行工具、浏览器体验，以及兼容 Ollama/OpenAI 的服务。

### 配合仓库阅读

| 阅读目标 | 实时更新的配套内容 |
|---|---|
| 运行示例模型 | [TensorSharp 快速开始](../README_zh-cn.md#快速开始) |
| 选择并下载 Gemma 4 E4B 文件 | [模型下载](../MODEL_DOWNLOADS_zh-cn.md) |
| 端到端查看模型实现 | [Gemma 4 架构卡片](models/gemma4_zh-cn.md) |
| 理解张量、包分层与引擎架构 | [开发与架构指南](../DEVELOPMENT_zh-cn.md) |
| 深入多模态、批处理、工具调用与推理功能 | [功能指南](../FEATURES_zh-cn.md) |
| 使用 CLI、服务器与兼容 API | [使用指南](../USAGE_zh-cn.md) |

仓库会在出版后继续演进；最新支持的模型、参数、后端和基准结果请以当前参考文档为准。书籍则提供一条紧凑、连贯的路线，帮助你理解这些概念如何组装成完整系统。

### 适合哪些读者

- 希望从调用托管 AI API 进一步走向引擎内部的 .NET / C# 开发者。
- 想通过具体项目系统学习 LLM 推理的学生与自学者。
- 希望连接模型架构与系统实现的机器学习工程师。
- 正在评估可理解、可本地运行推理技术栈的技术负责人。

### 获取本书

准备好走完从张量到 token 的完整路径了吗？

**[在 Amazon 查看 From Tensors to Tokens](https://www.amazon.com/dp/B0H9P44QZZ)**
