# Qwen3-VL（`qwen3vl`、`qwen3vlmoe`）

[← 返回模型索引](README_zh-cn.md) | [English](qwen3vl.md)

TensorSharp 支持 Qwen 的 Qwen3-VL 视觉语言模型：llama.cpp 转换器标为 `qwen3vl` 的稠密检查点，以及标为
`qwen3vlmoe` 的混合专家检查点，例如
[Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF)。支持文本、图像和视频
（`video_url`）。

该系列建立在共享解码运行时之上
（[`StandardDecoderModel`](../../TensorSharp.Models/SharedDecoder/StandardDecoderModel.cs)；见
[统一执行](../design/unified-execution.md)），因此无需额外代码即可获得连续批处理、前缀复用、张量并行（`--tp T`）、
整层放置（`--layer-split P`）及二者混合，以及把路由专家放在主机上（`--n-cpu-moe`、`--cpu-moe`）。它自己的部分是
Q/K 的处理、交错 M-RoPE 分段和 DeepStack。

| 属性 | 值 |
|---|---|
| GGUF 架构 | `qwen3vl`（稠密）、`qwen3vlmoe`（混合专家） |
| 模型类 | [`Qwen3VLModel`](../../TensorSharp.Models/Models/Qwen3VL/Qwen3VLModel.cs) |
| 架构插件 | [`Qwen3VLArchitecture`](../../TensorSharp.Models/Models/Qwen3VL/Qwen3VLArchitecture.cs) |
| 视觉塔 | [`Qwen35VisionEncoder`](../../TensorSharp.Models/Models/Qwen35/Qwen35VisionEncoder.cs)（Qwen 3.5 / 3.6 也使用的 Qwen3-VL 视觉塔），带 DeepStack 合并器 |
| 投影器 | 模型的 `mmproj`（`clip.projector_type = qwen3vl_merger`）；在模型旁按 `mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf`、`mmproj-F16.gguf` 等名称查找，或用 `--mmproj` 指定 |
| 模态 | 文本、图像、视频（`video_url`，采样为带时间的帧） |
| 思维链 | 仅 Thinking 检查点（其模板会打开 `<think>`）；Instruct 检查点没有推理块 |
| 工具调用 | 模板声明的 Hermes `<tool_call>` JSON |
| 批处理 / 分页前向 | 来自共享运行时 |
| 投机解码 | 仅无权重 n-gram（`--spec --spec-type ngram`）；没有公开的草稿模型 |
| 后端 | 文本支持所有后端；媒体提示以整图前向运行（DeepStack），所有后端都具备 |

## 运行

```bash
dotnet run --project TensorSharp.Server.Host -- \
    --model models/Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf \
    --mmproj models/mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf \
    --backend ggml_cuda --port 5000
```

视频以 OpenAI `video_url` 部分传入（data URL，或服务器可以获取的 URL）：

```json
{"role": "user", "content": [
  {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,...", "fps": 2, "max_frames": 64}},
  {"type": "text", "text": "What happens in this clip?"}]}
```

`VIDEO_SAMPLE_FPS=2`（以及 `VIDEO_MAX_FRAMES=64`）让服务器在请求未指定时按 Qwen3-VL 处理器的方式采样：每秒 2 帧，
两两合并（见[视频](#视频)）。

## 架构

文本图与 llama.cpp 的 `qwen3vl` / `qwen3vl-moe` 图以及 Hugging Face 的 `Qwen3VLMoeTextModel` 一致：

```text
tokens -> embedding (media rows: the tower's main merger)
  -> N x [ RMSNorm -> Q, K, V
            -> per-head Q/K RMSNorm -> NeoX RoPE over the whole head
               (interleaved M-RoPE: rope.dimension_sections (T, H, W) for media rows)
            -> causal GQA -> output projection -> residual
          RMSNorm -> SwiGLU (dense) or top-K routed experts (renormalized, no shared expert) -> residual
          after layer i < n_deepstack_layers: + DeepStack tap i at the media rows ]
  -> RMSNorm -> LM head
```

Qwen3-VL-30B-A3B：48 层，hidden 2048，32 个查询头和 4 个 KV 头（每头 128），128 个路由专家（top 8，宽 768），
分段 `[24, 20, 20, 0]`，三个 DeepStack 层。

### DeepStack

视觉塔的第 8、16、24 个 block（`clip.vision.is_deepstack_layers`）各接一个额外的合并器（`v.deepstack.N.*`）：
每个合并器取其 block 之后的残差流，把每 2 x 2 个 patch 合并为一行 4,608 宽的向量，对这一行做归一化（最终合并器是先
归一化再合并），再经过 fc1、精确 GELU 和 fc2 投影到语言模型的宽度。语言模型在第 i 个解码层之后，把第 i 个 tap 加到
媒体行的残差流上。

- 视觉塔对每个媒体行返回 `[main | tap 0 | tap 1 | tap 2]`（`ConcatenateDeepStack`），在 GGML 后端上与各 block
  在同一张合并图中计算（`TSGgml_Qwen35VisionEncoderMergedBatchF32`）。
- `Qwen3VLModel.SetVisionEmbeddings` 把主体部分交给运行时的媒体行，并保留各 tap，锚定到注入其媒体的那次前向的 KV
  索引（`StandardDecoderModel.OnMediaInjected`），因此前缀缓存截断或拆分的提示也能保持对齐。
- 每个图步骤把它所含行的 tap 作为每个 tap 层的一个逐次输入相加（文本行为零），在 `EmitBlockOutput` 中完成。tap 尚未
  消费时无法以图运行的步骤会直接失败，而不是在没有 tap 的情况下作答。
- `TS_QWEN3VL_DEEPSTACK=0` 去掉 tap，作为诊断用的 A/B。

### 视觉塔

Qwen3-VL 的视觉塔就是 Qwen 3.5 的：27 个宽 1152 的 block，16 个宽 72 的头，patch 16 带两个时间切片，2 x 2 空间
合并。其 block 使用 tanh GELU，合并器（最终合并器和 DeepStack 合并器）使用精确 GELU。图像与 Qwen 3.5 一样，在处理器的
像素预算内缩放到 32 的倍数。

### 视频

`video_url` 部分会被采样为带时间的帧（部分中的 `fps` / `max_frames`，否则用 `VIDEO_SAMPLE_FPS` /
`VIDEO_MAX_FRAMES`），并按 Qwen3-VL 处理器的布局渲染
（[`QwenVideoFrames`](../../TensorSharp.Runtime/QwenVideoFrames.cs)）：相距不超过 0.575 s 的帧共用一个时间 patch，
每个 patch 是一个 `<t seconds><|vision_start|><|video_pad|>...<|vision_end|>` 块，整段视频按处理器的 25,165,824
像素预算统一缩放。TensorSharp 还会在视频前用一行文本写出采样帧的时间，Hugging Face 处理器没有这一行。

一段视频的帧对一起编码（`Qwen35VisionEncoder.EncodeVideoPairs`）：帧并行解码，在 GGML 后端上合并图每次最多堆叠
`TS_QWEN_VISION_BATCH_PATCHES`（8,192）个 patch 的帧对，attention 仍限于每对之内。在 GGML GPU 后端上加载投影器时会
编码一对小的空白帧（`WarmUp`，约 0.4 s），这样第一段视频不必承担视觉塔一次性的设备初始化（在 CUDA 上与语言模型同时
运行时约 10 s）；`TS_QWEN_VISION_WARMUP=0` 跳过它。

在 OpenCvSharp 原生库无法加载的地方（并非每个平台都提供该库），改用 `ffmpeg` 和 `ffprobe` 可执行文件读帧
（[`FfmpegVideoDecoder`](../../TensorSharp.Models/Media/Desktop/FfmpegVideoDecoder.cs)；`TS_FFMPEG` / `TS_FFPROBE`，
或 `PATH`）。

## 聊天模板

`qwen3vl` [聊天协议](../../TensorSharp.Runtime/ChatProtocolRegistry.cs)按原样渲染 GGUF 自带的模板。它与 Qwen 3.5 的
区别在推理块：Instruct 模板没有推理块，因此生成提示完全按模板渲染的样子以 `<|im_start|>assistant\n` 结尾
（`KeepsGenerationPromptWhitespace`），只有 Thinking 模板的 `<think>\n` 会在缓存的助手轮次前恢复。工具调用是 Hermes
`<tool_call>` JSON。

## 性能

一块 CUDA GPU，`ggml_cuda` 上运行 Qwen3VL-30B-A3B-Instruct Q4_K_M 和 F16 投影器，对照 vLLM 0.32 运行 QuantTrio 的 AWQ
检查点（4-bit 语言模型、BF16 视觉塔、FP8 KV 缓存）。三段 20-37 s 的监控摄像头视频，每秒采样 2 帧（64 帧，提示
11.9K-12.5K token），一次一个请求，使用检查点的采样参数（temperature 0.7、top_p 0.8、top_k 20）：

| | TensorSharp | vLLM |
|---|---|---|
| 首个 token，未见过的视频 | 7.9-8.0 s（启动后的第一段视频为 10.1 s） | 4.2-4.7 s |
| 首个 token，同一视频的另一个问题（TensorSharp 复用已采样的帧和视觉塔输出，vLLM 复用解码后的帧） | 4.3-4.7 s | 3.0-3.4 s |
| 12K token 视频提示之后的解码 | 64-66 tok/s | 78 tok/s |
| 短文本提示之后的解码 | 98 tok/s | — |

一段新视频的 12.5K token 提示中，帧提取约 1.2 s，帧解码 0.3 s，视觉塔 2.2 s（34 对、每对 1,400 个 patch；F32 线性层
时为 3.3 s），预填充 4-5 s。预填充用的是 ggml 的 kernel：同一 GGUF 上 11.3K token 的文本提示，llama.cpp b12030 需要 2.9 s，
TensorSharp 需要 3.5 s，vLLM 的 AWQ kernel 更快。针对这些视频的 15 个简短问题，对照事先根据画面写好的答案，TensorSharp
答对 11 个、部分正确 2 个、答错 2 个，vLLM 分别为 10、3、2。

`top_k` 0（不限制）会在主机上对整个 151,936 token 的词表采样，解码降到约 42 tok/s；vLLM 默认使用检查点的
`top_k` 20。
