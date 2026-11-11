# 选择模型与阅读架构卡片

[English](README.md) | [中文](README_zh-cn.md)

先选择要得到的输出，再检查模型所接受的输入、权重与后端要求。下面是选择索引；每个家族的下载、前向图、实现与验证细节仍在各自卡片中。源码能力、某个测试通过和发布包可用性需要分别核对。

## 按任务开始

| 你要完成的任务 | 下一步 |
|---|---|
| 首次运行本地文本模型 | [快速上手](../getting_started_zh-cn.md) → [Gemma 4 E4B 原生 GGML 路径](gemma4_zh-cn.md#已验证的-gemma-4-e4b-原生-ggml-快速路径)；媒体输入另需匹配投影器。 |
| 在聊天中使用图片、文档或工具 | 从下方文本家族选择；再核对卡片的模态、工具协议与后端限制。 |
| 搜索文本、代码或媒体 | [嵌入指南](../embeddings_zh-cn.md)：Snowflake、MiniLM 与 EmbeddingGemma 2 输出向量；媒体支持取决于编码器。 |
| 生成或编辑图像 | [Qwen-Image-2.1](qwenimage21_zh-cn.md) 的匹配文件与采样配方。 |
| 生成视频，或同时生成音频 | [LTX 2.5](ltx-2.5_zh-cn.md)、[MiniMax-H3](minimax-h3_zh-cn.md) 或 [Wan](wan_zh-cn.md)；比较下表的输出区别。 |
| 已经选定模型，需要文件 | [模型下载](../../MODEL_DOWNLOADS_zh-cn.md)与该模型卡片；按卡片核对量化、内存及伴随文件。 |

## 如何阅读支持状态

- **已实现**：代码路径存在；具体设备与检查点覆盖见卡片。
- **条件支持**：需要指定变体、后端、伴随文件或设置。
- **实验性**：该功能或模型明确存在实现或验证限制。
- **夹具级验证**：基于小型合成 / 组件夹具；不能等同完整检查点的验证。
- **已测量**：只覆盖记录中的检查点、硬件、工作负载与构建。
- **不支持**：功能被拒绝或尚未实现。

输入列中的媒体是模型接收的内容；输出列区分文本、决策、图像、视频和向量。媒体通常需要卡片指定的伴随文件。表格不是跨设备的统一验证结论。

## 文本、多模态输入与决策

| 模型家族 / 卡片 | 输入 | 输出 / 任务 | 条件与证据 |
|---|---|---|---|
| [Gemma 4](gemma4_zh-cn.md) | 文本、图像、视频、音频 | 文本 | 条件支持：媒体输入需要匹配投影器。可从 E4B Q8_0 原生 GGML 路径开始。 |
| [Qwen 3.5 / 3.6 / 3.8 27B](qwen35_zh-cn.md) | 文本、图像 | 文本 | 条件支持：NextN 需要保留草稿块的 GGUF；核对变体与后端。 |
| [Qwen 3.8 Flash Next](qwen38-flash-next_zh-cn.md) | 文本、图像、视频 | 文本 | 条件支持：内存放置、张量并行及独立 MTP 头取决于量化与后端。 |
| [Bonsai2 27B](bonsai2_zh-cn.md) | 文本；图像需投影器 | 文本 | 条件支持：仅单设备 GGML。已记录的 Metal 与有限 CPU 检查不能覆盖其他设备。 |
| [GPT OSS](gptoss_zh-cn.md) | 文本 | 文本 | 已实现：显式思考模式与工具；后端与缓存要求见卡片。 |
| [Nemotron-H / Omni](nemotron_zh-cn.md) | 文本；Omni 图像；有条件的音频 | 文本 | 条件支持：音频需要公开 GGUF 未附带的 Parakeet 伴随文件。不支持投机解码。 |
| [Mistral 3](mistral3_zh-cn.md) | 文本、图像 | 文本 | 不支持：显式思考模式与工具。验证范围以具体检查点与后端为准。 |
| [Hunyuan Dense](hunyuan-dense_zh-cn.md) | 文本 | 文本 | 已实现：单设备文本路径；不支持工具、张量并行与层切分。 |
| [GLM 5.x](glm_zh-cn.md) | 文本；仅 5.3-Flash 支持图像 | 文本 | 条件支持：5.2/5.3 仅文本。NextN 不支持启用张量并行的运行；CPU 参考路径有数值限制。 |
| [DeepSeek V4 Flash](deepseek4_zh-cn.md) | 文本 | 文本 | 条件支持：核对内存与放置；可选的 DSpark 投机解码需要匹配草稿器。 |
| [DeepSeek V4.1 Flash / 1017](deepseek41_zh-cn.md) | 文本；有条件的图像与视频 | 文本 | 实验性：路由 MoE TP / DSpark。纯 C# CPU 为夹具级验证；1017 默认 YOCO prefill 是近似计算。 |
| [Muse Spark](muse-spark_zh-cn.md) | 文本；图像需 Spark 投影器 | 文本 | 条件支持：需要多 GPU 或大容量主机内存。已记录 CUDA 测量不能覆盖所有后端；不支持音频。 |
| [MiniMax-M3.1-Flash](minimax-m3_zh-cn.md) | 文本；图像需自行转换的投影器 | 文本 | 条件支持：需要 `ggml_cuda` 与多张 GPU（UD-Q4_K_XL 为 232 GiB）；投影器与 DSpark 草稿器需自行转换。已记录的 8x A40 测量不能覆盖其他后端；不支持音频。 |
| [Muse-Glimmer](muse-glimmer_zh-cn.md) | 文本、图像 | 文本 | 条件支持：图像需投影器，工具使用 ATEM，可选投机解码需独立 DFlash 草稿器；核对后端限制。 |
| [DiffusionGemma](diffusiongemma_zh-cn.md) | 文本、图像 | 文本 | 不支持：聊天音频、video_url 与工具。Web UI 视频以抽取图像提供。 |
| [Jev (DiffusionGemma)](jev_zh-cn.md) | 文本、图像、抽取的文档 / 帧 / 转录 | 类型化决策 | 条件支持：抽样视频帧及已配置 ASR 的转录；核对导入与概率语义限制。 |

DeepSeek 1017 的默认 YOCO prefill 是近似计算；选择精确设置前请阅读[质量与基准条件](deepseek41_zh-cn.md#unsloth-的-deepseek-v41-flash-1017-gguf)。公开的 Nemotron Omni 伴随文件没有 Parakeet 音频塔。Jev 的文档、视频与音频导入会变成抽取内容、抽样帧或转录文本，覆盖限制见其卡片。

## 图像与视频输出

| 模型家族 / 卡片 | 输入 | 输出 / 任务 | 条件与证据 |
|---|---|---|---|
| [Qwen-Image-2.1](qwenimage21_zh-cn.md) | 文本、可选参考 / 蒙版图像 | 图像 / 图像编辑 | 条件支持：匹配编码器 / VAE 与采样配方；组件验证覆盖随后端变化。 |
| [LTX 2.5 Distilled](ltx-2.5_zh-cn.md) | 文本、图像关键帧；可选身份参考图 | 视频 + 48 kHz 立体声 | 实验性：使用当前两阶段配方与伴随模型；不支持 Vulkan 与多 GPU。 |
| [MiniMax-H3-Image](minimax-h3-image_zh-cn.md) | 文本 | 图像，约 4 至 20 MP | 条件支持：仅 ggml 后端（已在 `ggml_cuda` 上实测）；6 步蒸馏、guidance 1；尚未实现编辑。 |
| [MiniMax-H3](minimax-h3_zh-cn.md) | 文本、关键帧或由检查点决定的参考媒体 | 视频 + 32 kHz 立体声 | 条件支持：FL2VA / Ref2VA 是不同检查点。组件门控与完整生成验证范围不同。 |
| [Wan 2.1 / 2.2](wan_zh-cn.md) | 文本；图像取决于变体 | 视频（无音频） | 条件支持：按变体匹配编码器 / VAE、采样器与步数；不支持多 GPU。 |

## 详细参考与架构开发

<a id="每张卡片的内容"></a>
<a id="已验证的起步路径"></a>
<a id="实现矩阵"></a>
<a id="后端说明"></a>
<a id="连续批处理--分页-kv-缓存"></a>
<a id="架构对比"></a>
<a id="新增模型架构"></a>

完整的[实现矩阵](capability-reference_zh-cn.md#实现矩阵)、[后端说明](capability-reference_zh-cn.md#后端说明)、[批处理与缓存](capability-reference_zh-cn.md#连续批处理--分页-kv-缓存)、[架构对比](capability-reference_zh-cn.md#架构对比)及[新增架构步骤](capability-reference_zh-cn.md#新增模型架构)已移到同目录的详细参考。原有详细内容保留在那里，模型卡片仍是单个家族的阅读入口。

比较速度时请阅读[最新且限定范围的基准](../benchmarks_zh-cn.md)；部署前核对[项目状态](../PROJECT_STATUS_zh-cn.md)与[后端 / 功能支持](../supported_models_zh-cn.md)。如需从张量到推理的连续学习路径，可阅读[配套书籍指南](../BOOK_zh-cn.md)。
