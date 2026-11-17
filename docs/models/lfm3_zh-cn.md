# Liquid LFM3 与 LFM3-Omni（`lfm3`）

[← 返回模型索引](README_zh-cn.md) | [English](lfm3.md)

Liquid AI 的 LFM3 检查点声明 `general.architecture = lfm3`：**LFM3-1.7B** 是支持推理与工具调用的文本模型；
**LFM3-Omni-2B** 使用同一解码器，外加一个多模态投影器，可输入图像、视频与音频并能说话（投影器中的语音生成器与编解码器：
见[语音输出](#语音输出)）。两者共用一套聊天协议，由 GGUF 自带的模板区分。

| 属性 | 值 |
|---|---|
| GGUF 架构 | `lfm3`（两个检查点相同） |
| 模型类 | 共享解码运行时上的 `Lfm3RuntimeModel`（`TensorSharp.Models/Models/Lfm3/`） |
| 聊天协议 | [`ChatProtocolRegistry`](../../TensorSharp.Runtime/ChatProtocolRegistry.cs) 中的 `lfm3`，后备渲染器 [`ChatTemplate.Lfm3.cs`](../../TensorSharp.Runtime/ChatTemplate.Lfm3.cs) |
| 参考模板 | GGUF 的 `tokenizer.chat_template`（与 `LiquidAI/LFM3-1.7B` / `LFM3-Omni-2B` 的 `chat_template.jinja` 逐字节相同） |
| 模态 | 1.7B：文本。Omni：图像、视频、音频输入（需投影器），文本输出 |
| 思考 | 1.7B：支持（`<think>`，默认值见[推理](#推理)）。Omni：不支持 |
| 工具调用 | 支持：Qwen3-Coder XML（`<tool_call><function=…><parameter=…>`） |
| 现成配置 | [`config/lfm3-1.7b.json`](../../config/lfm3-1.7b.json)、[`config/lfm3-1.7b-nvfp4.json`](../../config/lfm3-1.7b-nvfp4.json)、[`config/lfm3-omni-2b.json`](../../config/lfm3-omni-2b.json) |
| TensorAgent | `lfm3-1.7b-q4km`、`lfm3-omni-2b-q4km`（实验性，12 GB 档） |
| 许可证 | LFM Open License v1.0 |

## 验证状态

截至 2026-11-16（llama.cpp 参考版本 `d3c3e242c`）：

- **已验证：** 聊天协议。32 个 OpenAI 请求体（普通对话、系统提示、带推理的多轮、工具声明、单次与并行工具轮次、
  带类型的参数、工具轮次中的推理、Unicode，以及 Omni 的图像 / 音频 / 图像+音频 / 历史轮次）经 TensorSharp 的
  OpenAI 消息与工具解析器处理后，无论走 GGUF 模板还是后备渲染器，渲染结果都与 llama-server `/apply-template`
  **逐字节一致**（`InferenceWeb.Tests/Lfm3ChatProtocolTests.cs`）。回复解析使用 llama.cpp 对两个 Q4_K_M GGUF 的
  真实贪心输出，以 1 到 4096 字符的分块流式验证。固定文件的 SHA-256 与 Hugging Face LFS 对象 id 一致。
- **尚未验证：** 任何后端上的实际生成、Omni 的视觉与音频编码器、设备上的 TensorAgent 应用，以及所有 CUDA 路径
  （没有可用的 NVIDIA GPU；`config/lfm3-1.7b-nvfp4.json` 从未运行过）。尚未与 llama.cpp、vLLM、SGLang 对比速度。

## 运行

```bash
echo "What is 17*23?" > prompt.txt
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --config config/lfm3-1.7b.json --think --input prompt.txt
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --config config/lfm3-omni-2b.json
```

配置会在首次使用时把固定版本的文件下载到 `${TENSORSHARP_MODELS:-../models}/lfm3/`。

| 文件 | 字节数 | SHA-256 | 仓库 @ 版本 |
|---|---|---|---|
| `LFM3-1.7B-Q4_K_M.gguf` | 1,099,627,808 | `3793e1e6…b05c93c4` | `LiquidAI/LFM3-1.7B-GGUF@cd0c74c5` |
| `LFM3-1.7B-NVFP4.gguf` | 1,280,019,776 | `462c2379…59a67bec` | `LiquidAI/LFM3-1.7B-GGUF@cd0c74c5` |
| `LFM3-Omni-2B-Q4_K_M.gguf` | 1,099,630,080 | `9f8d30e9…cc9893e2` | `LiquidAI/LFM3-Omni-2B-GGUF@d508b88f` |
| `mmproj-LFM3-Omni-2B-Q8_0.gguf` | 506,842,528 | `8a25f9d1…9c6b27b1` | `LiquidAI/LFM3-Omni-2B-GGUF@d508b88f` |

完整哈希见配置文件，与这些版本中文件的 LFS 对象 id 一致。仓库还提供 Q5_K_M、Q6_K、Q8_0、BF16、F16 以及经量化感知蒸馏的
`QAD-Q4_0`（1.7B），Q4_0/Q8_0/BF16/F16（Omni）和 mmproj BF16/F16/Q4_0/Q4_K_M，元数据相同，加载方式相同。

## 架构

24 层，残差宽度 2,048，每层都有稠密 SwiGLU FFN（7,680）。`lfm3.attention.head_count_kv` 是逐层数组：KV 头数为 0 的
13 层是**短卷积**层（`lfm3.shortconv.l_cache = 7`：带门控的因果逐通道卷积，每个序列一份 6 列 F32 状态），其余 11 层
是 GQA 注意力（16 个查询头、4 个 128 维 KV 头、逐头 Q/K RMSNorm、NeoX RoPE、sigmoid 输出门）。其中
`sliding_window_pattern` 标记第 4、8、12、16、20 层为滑动窗口（窗口 256，RoPE 基 10,000）；第 2、6、10、14、18、22 层
关注完整上下文（RoPE 基 1,000,000）。GGUF 声明的窗口为 262,144（1.7B）与 128,000（Omni）token，可用 `MAX_CONTEXT` 限制。

F16 下每 token 的 K/V：所有注意力层都使用线性缓存时为 11 层 × 4 头 × 128 × 2 × 2 = 22,528 字节；五个滑动窗口层改用
固定 256 行环形缓存（llama.cpp 的 iSWA 缓存）后为 12,288 字节。短卷积状态按序列固定且不能回退，因此前缀只能精确复用
（与 Qwen 3.5 系列的递归状态规则相同）。

## 聊天模板

GGUF 模板按原样渲染：TensorSharp 的 Jinja 引擎把 `{% generation %}` 视为透明，`bos_token` 渲染为空，因为模板要求的
那一个 `<|startoftext|>` 由分词器添加（GGUF 中 `add_bos_token` 为 false）。没有模板的 GGUF 由 `RenderLfm3` 生成相同文本。

### 推理

- 生成提示以 `<think>`（开启思考）或 `<think></think>`（关闭）结尾，不带换行。模型先写推理，再写 `</think>` 和答案。
- **默认值：** 1.7B 模板默认 `enable_thinking` 为 true，Omni 模板默认为 false；但 TensorSharp 在所有宿主上默认关闭
  请求的思考开关（与 Qwen 3.5 相同）。要使用 1.7B 模型卡的默认模式，请发送 `"think": true` 或
  `chat_template_kwargs.enable_thinking: true`（CLI 用 `--think`）；与默认开启思考的 llama.cpp、vLLM、SGLang 对比时务必固定该值。
- 早先的轮次重新渲染为 `<think></think>` + 答案，推理被丢弃（模板的 `preserve_thinking` 默认关闭；llama-server 默认用
  `--reasoning-preserve` 打开它，因此客户端发送 `reasoning_content` 时，它的多轮提示与 HF、TensorSharp 不同）。TensorSharp
  自己的对话会把过去轮次生成的 token 原样拼回，使下一轮精确延续 KV 缓存。客户端为当前轮次的工具轮次发送的推理会被保留，
  与模板一致。
- `</think>`（124902）是单个 token：思考预算在此结束推理块并继续生成答案，`response_format` 语法也在其后生效。
- LFM3-Omni 不做推理：开启思考时它会立即关闭推理块。

### 工具调用

模型输出 Qwen3-Coder XML，并行调用之间以空行分隔。`Qwen35OutputParser` 负责解析，与 llama.cpp 为该模板使用的
`qwen3-coder` 解析器一致：推理在 `</think>` 处结束，或在推理过程中写出 `<tool_call>` 时结束；声明为字符串的参数保留原文，
其他声明类型按 JSON 解析。工具结果渲染为 `<|im_start|>tool\n<tool_response>\n…\n</tool_response>`，只依赖工具消息本身，
因此工具轮次生成的 token 会被拼回，下一轮延续实时缓存。接受 `tool_choice: required`，但不用语法强制（与 Qwen 3.5 相同）。

工具声明经由 TensorSharp 共用的工具构建器，保留每个参数的 `type`、`description` 与 `enum`，但不保留嵌套 schema（`items`、
嵌套 `properties`、默认值）：这是所有模型族共有的限制，而 llama.cpp 与 vLLM 会原样打印客户端的 schema。

### 媒体占位符

LFM3-Omni 在多模态展开器运行之前的占位符约定（同一条消息的媒体总是排在文本之前）：

| 输入 | 提示文本 | 展开器替换 |
|---|---|---|
| 每张图像 | `<\|vision_start\|><\|image\|><\|vision_end\|>` | 把 `<\|image\|>`（125030）替换为编码器输出的 N 行 |
| 一段视频（抽样帧） | `<\|vision_start\|><\|video\|><\|vision_end\|>` | 把 `<\|video\|>`（125029）替换为每对帧的 `"<t.t seconds>"` + `<\|vision_start\|>` + 各行 + `<\|vision_end\|>`（+ 该对帧的音轨行） |
| 每段音频 | `<\|vision_start\|><\|audio\|><\|vision_end\|>` | 把 `<\|audio\|>`（125031）替换为编码器输出的各行 |

这些就是模板为 `image`、`video`、`audio` 内容片段渲染的片段，最终布局与 llama.cpp 的 mtmd（Omni 音频同样包在视觉标记中）
以及 HF（成对标记中放 N 个 `<|image|>`）一致。OpenAI `video_url` 片段会被抽样为带时间的帧并限制帧数（`CapsVideoFrames`）；
llama.cpp 对 Omni 以 2 fps 抽样，帧数限制在 [4, 384]。

### 停止 token

`<|im_end|>`（GGUF 的 EOS）与 `<|endoftext|>`，与 llama.cpp 的词表一致。Omni 的 `generation_config.json` 还列出
`<|text_end|>`（125014）；llama.cpp 只用它结束语音与交错生成，在 LFM3-Omni-2B-Q4_K_M 的 12 次贪心文本对话中它从未出现，
因此文本对话不以它停止。

## 采样

| 模型 | 模型卡 / GGUF `general.sampling` | 配置 |
|---|---|---|
| LFM3-1.7B | temperature 0.1、top_k 50、repetition_penalty 1.1（数学/理工：temperature 1.0、top_p 0.95、top_k 20） | 模型卡通用预设，top-p 1、min-p 0、`repeat-last-n -1` |
| LFM3-Omni-2B | 贪心，repetition_penalty 1.0 | temperature 0、top-k 1、repeat-penalty 1 |

服务端不读取 `general.sampling`，因此配置文件固定了这些值（CLI 在对话时会读取它）。TensorSharp 的重复惩罚只回看最近
`repeat_last_n` 个**生成的** token；llama-server 还计入提示词最后 `repeat_last_n` 个 token，vLLM 与 HF 计入整个提示词和输出，
SGLang 计入全部输出。`repeat-last-n -1` 与 SGLang 一致；与其他引擎比较时请使用 repeat penalty 1.0 并贪心解码。

## 语音输出

LFM3-Omni 的 mmproj 中带有语音生成器（`a.gen.code.*`：3 层双向 depthformer，用 3 次 MaskGIT 把 LM 的最终隐藏行变成
一帧 12 个编码）与 LiquidCodec 解码器（`a.gen.wav.*`：帧经主机端 ISTFT 变成 24 kHz PCM）。加载 mmproj（`--mmproj`）
即开启语音；组件为 `TensorSharp.Models.Lfm3.Lfm3SpeechOutput`，在 ggml-metal 上与 llama.cpp 逐位一致。

**语音回复如何运行。** 一个请求的各个槽由驱动器决定（`Lfm3SpeechDriver`，即 llama.cpp 的 `step_gen` 调度）：文本槽从
logits 采样 token；语音槽用上一位置的最终隐藏行采样一帧，排入其 80 ms PCM，并把该帧的反馈行（12 个码本嵌入之和）作为
下一个输入行解码。运行时的槽步骤可以用嵌入行代替 token，并返回最后一行的最终隐藏行；某步没有请求需要 logits 时跳过
128000 x 2048 的 LM head，因此语音请求与文本请求在每个执行器上同批运行。模式：`audio`（`<|audio_start|>`）、
`interleaved`（`<|mixed_start|>`：6 个文本 token，然后 12 帧，……，直到 `<|text_end|>`；文本即语音的转写），以及
TTS（GGUF 的 `tokenizer.chat_template.tts`，在语音结束帧停止）。对话回复在语音之后继续以文本进行（与模型卡的
`generate` 和 llama.cpp 的 realtime 服务一致）；TTS 回复在语音结束时停止（与 llama-tts 一致）。

用法：CLI `--tts` / `--speech-out`（配合 `--speech-mode`、`--speech-temperature`、`--voice`、`--language`、`--seed`）、
`POST /v1/audio/speech`，以及带 `modalities` `["text","audio"]` 或 `["audio"]` 的 `/v1/chat/completions` /
`/v1/responses`——见 [USAGE_zh-cn.md → 语音输出](../../USAGE_zh-cn.md#语音输出lfm3-omni)。音色为 `female` 与 `male`
（写在系统提示中，没有说话人嵌入）；未知音色（包括 OpenAI 默认的 `alloy`）返回 400 并列出可用音色。语音对话回复的系统提示
加上 llama.cpp 的 realtime 句子（“Speak in English using a female voice.”），交错回复再加上模型卡的
“Respond with interleaved text and audio.”。

LFM3-Omni 的文本回复永不采样媒体占位符（`<|audio|>`、`<|audio_out|>`、`<|image|>`、`<|video|>`），纯文本回复也不会开启
语音（禁止 `<|audio_start|>`、`<|mixed_start|>`），与 HF 的 `generate` 一致；llama.cpp 在普通对话中两者都不抑制。（有被抑制
token 的采样器不是纯 argmax，因此 LFM3-Omni 的文本回复不使用直接 CUDA 的设备端贪心选择。）

**验证（2026-11-17，M5 Pro，ggml-metal，Q8_0 LM + Q8_0 mmproj，llama.cpp `d3c3e242c`）：**

- 回复驱动器使用 llama.cpp 在一次 llama-tts 运行中的隐藏行时，重现 llama-tts 的 WAV，误差在 1 LSB 以内（130,560 个采样中
  20 个不同），并在同一帧 68 结束语音。
- 端到端：TensorSharp 的 TTS 使用 llama-tts 的种子时，前 2-7 帧的编码与 llama-tts 相同，之后分叉：LM 的隐藏行与
  llama.cpp 相差 2-7e-4（相对 L2），某帧的采样翻转。语音温度为 0（argmax）时，5 段文本中有 2 段整句与 llama-tts 相差
  不超过 1 LSB（72 帧与 86 帧）。
- 可懂度（通过 llama.cpp 运行 LFM3-Omni 自身的 ASR，`eng/lfm3/speech-asr.py`）：5 段文本（美式英语、英式英语、法语、
  中文）x 5 个种子，TensorSharp 的 WER/CER 为 0.00%（25 段中 25 段完全正确），llama-tts 为 1.38%（25 段中 22 段）。
- 交错回复：每条英文回复的语音经 ASR 后与其转写一致（6/6，WER 0%），与用同样方式驱动的 llama.cpp realtime 路径相同
  （`eng/lfm3/speech-chat-ref.cpp`）。两个引擎的中文交错语音都不可懂（模型的限制；中文 TTS 是可懂的）。
- 纯语音回复（`modalities ["audio"]`，与 llama.cpp realtime 的 AUDIO 模式一样以 `<|audio_start|>` 开始）：LFM3-Omni
  会把用户的消息朗读出来而不是回答，TensorSharp 与 llama.cpp 相同（8/8 个问题，ASR 结果即问题本身）；没有音色句时语音不可懂。
  需要语音回答时请用 `["text","audio"]`。
- TTS 回复说话期间处理的文本请求保持其贪心回复不变；引擎上的 TTS 与独立合成逐位相同。
- 速度（流式 `/v1/audio/speech`，同一客户端交替请求 TensorSharp 与 llama-server，在共享且高负载的 Mac 上）：Metal 上两者持平——
  首段音频 88-114 ms 对 119-141 ms，实时倍率 1.83x 对 1.38-1.92x（负载 36-95；空闲机器上两者都快 3-5 倍）。纯语音步骤跳过 LM
  head（该步 -15%）。MLX 约为 Metal 的一半速度（每次图运行的固定开销）。

限制：没有 `/v1/realtime`；之后历史中的语音助手回合按其转写渲染（OpenAI 的 `audio.id` 在一小时内映射到它），而不是按音频帧；
语音请求不保留在前缀缓存中；语音与 CUDA 未在 NVIDIA GPU 上测试。

## 基准测试

在 `/v1/chat/completions`（以及 `/v1/responses`）上设置 `ignore_eos: true`，会越过结束 token 精确生成 `max_tokens` 个 token，
符合 vLLM 与 SGLang 基准客户端的预期（`vllm bench serve --ignore-eos`，SGLang 默认开启）。在每个引擎上都要固定 `think` 与采样参数。

## 当前限制

- 语音输出没有实时端点（`/v1/realtime`），历史中的语音回合按其转写重新渲染（见[语音输出](#语音输出)）。
- 不提取视频的音轨（CLI 与 Web UI 只传递帧）。
- 发给 LFM3-1.7B 或未加载投影器的 Omni 的音频尚未被拒绝；多模态集成必须拒绝它，而不是丢弃它。
- `tool_choice: required` 不由语法强制；嵌套工具 schema 会被展平。
- 在滑动窗口环形缓存落地之前，TensorAgent 目录按每 token 22,528 字节 K/V 计算；QAD-Q4_0 文件尚未列入目录，等待与 Q4_K_M
  的质量与速度测量。
