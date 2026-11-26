# DiffusionGemma

[← 返回模型索引](README_zh-cn.md) | [English](diffusiongemma.md)

## 状态快照

| 字段 | 状态 |
|---|---|
| GGUF 架构标识 | `diffusion-gemma`、`diffusion_gemma` |
| 模型类 | [`DiffusionGemmaModel`](../../TensorSharp.Models/Models/DiffusionGemma/DiffusionGemmaModel.cs) |
| 采样器 | [`DiffusionGemmaSampler`](../../TensorSharp.Models/Models/DiffusionGemma/DiffusionGemmaSampler.cs) |
| 模态 | 原生文本 + **图像**；[Jev](jev_zh-cn.md) 另支持文档提取、抽样视频帧，以及已配置 ASR 配套服务的语音转录 |
| 思维链 / 工具调用 | 不提示思考：提示词始终以关闭思考的方式渲染。模型自行写出的思维块会被解析剥离，仅在 `"think": true` 时作为推理返回；若 canvas 只有思维块，这段文本会作为答案返回（见 §6）。tools/tool_choice 以 HTTP 400 拒绝 |
| 生成方式 | 分块文本扩散，不是自回归 token decode |
| CLI 支持 | `TensorSharp.Cli` 检测到 `DiffusionGemmaModel` 后进入 diffusion 运行模式 |
| 服务端支持 | Web UI chat stream 带实时去噪预览；Ollama/OpenAI 兼容端点使用 append-oriented 响应形状，只返回最终文本（没有去噪预览） |
| 连续批处理 | 独立的 [`DiffusionBatchScheduler`](../../TensorSharp.Chat/DiffusionBatchScheduler.cs)，在 block 边界接纳请求 |

## 下载

已验证的 GGUF 下载指引：

| 模型 | HF 仓库 | 推荐文件 | 备注 |
|---|---|---|---|
| diffusiongemma-26B-A4B-it | [unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) | `diffusiongemma-26B-A4B-it-Q4_K_M.gguf`（16.807 GB）；另有 `Q5_K_M`、`Q6_K`、`Q8_0`、`BF16` | GGUF `general.architecture` = `diffusion-gemma`。官方上游权重：[google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) |

`Q4_K_M` 是 unsloth 仓库中最小的量化版本。显存更紧张时，
`config/diffusiongemma-26b-a4b-q3.json` 固定使用 DevQuasar 的 `Q3_K_M`（约 13.3 GB；
unsloth 没有发布 Q3_K_M）。它的元数据与 unsloth 文件一致，只多了 TensorSharp 不读取的
`diffusion.eb_*` 采样提示。

**图像输入还需要视觉塔，而任何 GGUF 里都没有它。** 该检查点已发布的 GGUF 全部
是纯文本的 —— 转换过程丢弃了视觉塔，官方也从未发布过 mmproj。上游权重里确实有
这座塔：它的全部 356 个张量都在 11 分片 BF16 检查点的某一个 2.8 GB 分片中，
TensorSharp 直接加载该分片（无需转换）：

| 文件 | HF 仓库 | 大小 |
|---|---|---|
| `model-00011-of-00011.safetensors` | [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) | 2.84 GB |

```bash
hf download google/diffusiongemma-26B-A4B-it model-00011-of-00011.safetensors --local-dir models
```

也可以让配置在首次使用时自动获取 —— `config/diffusiongemma-26b-a4b-q4.json`、
`config/diffusiongemma-26b-a4b-q3.json` 和 `config/jev-diffusiongemma-q4.json` 都在
`mmproj` 下声明了它并带 SHA-256，只下载一次，之后复用。若要只跑纯文本且不下载该分片，
在命令行上追加 `--mmproj none`：命令行上的 `--mmproj` 会在解析之前丢弃配置中的条目，因此
该分片永远不会被下载，而 `none` 表示不加载投影器——服务端与 CLI 都是如此（纯文本请求正常，
图像请求被拒绝）。

如果更想要一个普通的 mmproj GGUF，
[`eng/diffusiongemma-mmproj.py`](../../eng/diffusiongemma-mmproj.py) 只依赖 numpy（不需要 torch
或 gguf 包）就能把该分片转换成单文件的 `gemma4v` 投影器，其二维矩阵乘权重为 F16（`--dtype f32`
则保留 F32）：

```bash
python3 eng/diffusiongemma-mmproj.py --src models/model-00011-of-00011.safetensors --out models/mmproj-diffusiongemma-26B-A4B-it-F16.gguf
```

像其他投影器一样把结果传给 `--mmproj` 即可。
`Gemma4VisionOracleTests.MmprojTower_AgreesWithSafetensorsTower` 会检查它与分片的编码结果一致；
该测试需要本地 fixture 目录（`TS_DIFFUSIONGEMMA_VISION_DIR`），没有时会跳过。脚本同样面向
llama.cpp 的 clip 加载器编写，但没有记录过用 llama.cpp 运行其输出的结果。

原生音频**不受支持**，任何 projector 也补不上：上游 config 没有 `audio_config`，
权重里也没有音频塔，因此 tokenizer 从 Gemma 4 继承来的 `<|audio|>` token 背后
什么都没有。

类型化判定（`noul`、`choice`、`score`）请使用原生 [`/v1/systemone` Jev 端点](jev_zh-cn.md)：
它在一次去噪步中从带种子的 canvas 读取标签概率，使用稀疏输出投影，不解析生成的 JSON。
其状态支持上传文本/文档、通过视觉塔读取的图像与抽样视频帧，以及由配置的 ASR 配套服务提供的音频转录。
这些预处理通路详见 Jev 指南。可从
[`jev-diffusiongemma-q4.json`](../../config/jev-diffusiongemma-q4.json) 开始。

命令行下载（每个文件一行；需要先 `pip install -U huggingface_hub`）：

```bash
python -m pip install -U huggingface_hub
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF diffusiongemma-26B-A4B-it-Q4_K_M.gguf --local-dir models
```

CLI diffusion 模式（根据模型架构自动分发 —— 无需模式标志；提示词通过
`--input` 从文件读取）：

```bash
dotnet run --project TensorSharp.Cli -c Release -- --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --input prompt.txt \
  --backend ggml_cuda --max-tokens 256 --diffusion-steps 48 --diffusion-seed 0 --diffusion-blocks 1
```

若要针对图片提问，追加 `--mmproj models/model-00011-of-00011.safetensors` 与
`--image photo.png`（多张图片可重复 `--image`，按顺序传入）。CLI 不会在模型旁边查找
这个分片；没有加载视觉塔时，它会拒绝图像，而不是在看不到图像的情况下作答。

服务端（`http://localhost:5000/index.html` 的 Web UI 会流式展示实时去噪预览 ——
每一步通过 `replace` SSE 帧重绘整条消息；Ollama/OpenAI 兼容端点只返回最终文本）：

```bash
dotnet run --project TensorSharp.Server.Host -c Release -- --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --backend ggml_cuda
```

追加 `--mmproj models/model-00011-of-00011.safetensors` 即可在聊天请求中接受图像；
服务端从不自动探测投影器。

## 1. 来源与目标

DiffusionGemma 是基于 Gemma-4 风格 Mixture-of-Experts backbone 的分块文本扩散语言模型。
它和自回归 `gemma4` 的运行时契约不同：

- `Forward(int[] tokens)` 会主动抛错；生成必须通过 `DiffusionGemmaSampler`。
- 每个去噪步骤在拼接后的 `[prompt | canvas]` 序列上运行。
- prompt 区域是因果的，并且不会看见 canvas。
- canvas 区域对 prompt + canvas 做双向注意力。
- 每个 block 输出当前确定性的 argmax canvas，并在多步去噪中逐步收敛。

GGUF 必须报告 `general.architecture=diffusion-gemma` 或 `diffusion_gemma`；
`ModelBase.Create()` 会把这两个 key 路由到 `DiffusionGemmaModel`。

## 2. 前向计算图

DiffusionGemma 运行在共享解码运行时上：`DiffusionGemmaModel` 派生自
[`Gemma4DecoderModel`](../../TensorSharp.Models/Models/Gemma4/Gemma4DecoderModel.cs)，即 Gemma 4 26B-A4B
的配方（逐层的 head 宽度、V = K 的全局层、分解的 RoPE、三明治归一化、与 128 专家路由分支并列的稠密 GeGLU
分支），因此它的缓存、放置、专家卸载与计算图执行器在每个后端上都是运行时提供的。

一个 block 先做一次提示预填充，之后每个去噪步运行一次 canvas 步骤：

1. `PrefillPrompt(promptTokens)` 把提示作为因果提示写入运行时的缓存（滑动窗口层对其裁剪；图像的软 token
   只在这些层上彼此双向注意，且每张图像各自独立）。每一层的输出乘以其 `enc_layer_output_scale`。下一个 block
   的提示在此基础上延伸，因此只预填充新增的行。
2. `DecodeCanvas(canvasTokens, scBuffer, scUse, prevTempInv)`（或 `DecodeCanvasSampled`）把 canvas 作为提示之后
   的一组块行（block rows）以一张图运行：在每一层上每个 canvas 行都读取其他所有 canvas 行，并从第一个 canvas
   行的窗口起读取提示行（全局层读取全部）；每一层的输出乘以其 `layer_output_scale`。不提交任何内容：canvas 的
   K/V 写入之后会被下一批提示行覆盖的单元，因此一个 block 的每一步读到的是同一份提示。同一个 block 的各步重放
   同一张已记录的计算图。

```text
canvas tokens
  -> rms_norm(embed * sqrt(n_embd) + selfcond(上一步的 top-32))   （无权重）
  -> 30 层 Gemma 4，作为基于提示缓存的块行
  -> output norm -> 共享 lm-head -> final logit softcap
  -> canvas logits，或（设备端采样）每个位置的 argmax / 熵 / 抽样 / top-32
```

`ForwardCanvas(tokens, promptLen)` 在一次调用中接收提示与 canvas，执行同样的两步（提示是因果的，从不读取 canvas）。

## 3. 采样器契约

`DiffusionEbParams` 控制生成：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `MaxDenoisingSteps` | 48 | 每个 canvas block 最大去噪步数 |
| `TMin` / `TMax` | 0.4 / 0.8 | 从后期到前期的温度调度 |
| `EntropyBound` | 0.1 | 被接受位置的累计互信息上界 |
| `StabilityThreshold` | 1 | 早停前 argmax canvas 需要稳定的步数 |
| `ConfidenceThreshold` | 0.005 | 早停使用的平均熵阈值 |
| `Seed` | 0 | 确定性采样种子 |
| `MaxBlocks` | 1 | block-autoregressive canvas 数量 |

CLI 对应：

```bash
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --input prompt.txt --backend ggml_metal \
  --max-tokens 256 --diffusion-steps 48 --diffusion-seed 0 --diffusion-blocks 1
```

当 `--diffusion-blocks` 为 `0` 时，CLI 会根据 `--max-tokens` 与
`diffusion.canvas_length` 推导 block 数。

## 4. 架构细节

DiffusionGemma 复用了许多 Gemma-4 backbone 设计：

- NeoX RoPE，并区分 local/global 维度。
- 五个 local sliding-window 层加一个 global 层的循环模式。
- 每头 Q/K RMSNorm 与无权重 V RMSNorm。
- Global 层可以没有 `attn_v.weight`，此时 V 使用原始 K 投影。
- Dense gated-GELU MLP 加 128-expert top-8 MoE。
- 共享 embedding / lm-head，并带 final logit softcapping。

Diffusion 专属元数据：

| Key | 含义 |
|---|---|
| `diffusion.canvas_length` | 每个 block 去噪的 canvas 位置数，默认 256 |
| `tokenizer.ggml.mask_token_id` | warmup 与回退路径使用的 mask token id |
| `<arch>.attention.sliding_window_pattern` | local/global 层模式 |
| `<arch>.attention.head_count_kv` | 每层 KV head 数 |
| `<arch>.expert_count` / `<arch>.expert_used_count` | MoE expert 总数与 active top-k |

## 5. 加速状态

- 提示缓存、canvas 步骤与批量 canvas 在每个后端上运行（`ggml_cuda`、`ggml_metal`、`ggml_vulkan`、`ggml_cpu`、
  `cuda`、`mlx`、`cpu`），每一步都是该后端执行器上的一张计算图，每个 block 记录一次后重放。
- 一个 block 的提示复用上一个 block 已缓存的行，只预填充新增的行。
- `ggml_cuda` 上的设备端采样：输出头的 `[vocab, canvas]` logits 留在设备上，计算图只返回每个位置的 argmax、熵、
  多项式抽样与 top-32（下一步的自条件输入）。其他后端把 logits 读回，由主机采样。
- 自条件（检查点带有 `self_cond_*` 权重时开启）：上一步 top-32 的软嵌入在主机上由词表求和，再在该步的计算图中
  经过门控 GELU MLP。
- 放不进设备的路由专家在主机内存中运行（运行时的放置，或 `--n-cpu-moe N` / `--cpu-moe`），每个执行器上都使用
  融合的 gate|up GeGLU 专家栈。
- `TensorSharp.Server` 通过 `DiffusionBatchScheduler` 在 block 边界做连续批处理；多个请求的 canvas 可以在一步中
  一起运行（`DecodeCanvasBatched`）。

在单张 A40 上实测（`diffusiongemma-26B-A4B-it-Q4_K_M.gguf`，ggml d9fe7bff，`eng/validation/DiffusionGemmaAB`），
与被本运行时取代的旧模型对比：

| `ggml_cuda` | 旧模型 | 运行时 |
|---|---:|---:|
| 512 token 提示预填充（预热后） | 2.47–2.65 s | 0.14–0.15 s |
| canvas 步骤，主机采样（256 行，含自条件） | 325–333 ms | 145–160 ms |
| canvas 步骤，设备端采样 | 218 ms | 114–128 ms |
| 生成，三个提示（每个去噪步的毫秒数） | 268–353 | 129–153 |

| 其他后端（同一检查点与探针） | 旧模型 | 运行时 |
|---|---:|---:|
| `cuda`：512 token 提示预填充（预热后） | 1.71–1.75 s | 0.39 s |
| `cuda`：canvas 步骤，主机采样 | 1.67–1.76 s | 0.92–0.95 s |
| `cpu`（A40 虚拟机的主机）：64 token 提示预填充（预热后） | 0.65–0.70 s | 0.69–0.71 s |
| `cpu`：canvas 步骤，主机采样 | 2.79–2.89 s | 2.58–2.82 s |

`cpu` 各行是在共享主机（负载 11-13）上交替运行的两组结果。在 `cpu` 上，模型通过回收式分配器分配内存
（每一步词表宽度的中间结果原地写入，而不是每步映射再解除映射），全上下文缓存来自已清零的页，重置只清零
提示写过的行。

以旧模型自己的回答做 teacher forcing（相同的 256 token canvas），两者在每个位置上的 argmax 都一致，canvas token
的平均对数概率相同（带自条件时为 -0.0000 / -0.0000 与 -0.0003 / -0.0003）；相同提示与种子下生成的回答内容一致。

重要开关：

| 变量 | 作用 |
|---|---|
| `DIFFUSION_STEPS` | 服务端每个 block 的去噪步数，默认 48 |
| `DIFFUSION_MAX_BATCH` | 服务端 diffusion scheduler 的最大活跃请求数，默认 2 |
| `TS_VRAM_HEADROOM_MB` | 运行时专家放置为设备保留的空闲显存（默认设备显存的 1/16）；放不下的路由专家在主机内存中运行 |

旧模型的开关（`DIFFUSION_NO_PKV`、`DIFFUSION_NO_SC`、`DIFFUSION_SC_TOPK`、`DIFFUSION_BATCHED_FORWARD`、
`DIFFUSION_NO_FUSED_DECODE`、`DIFFUSION_NO_FUSED_LMHEAD_TAIL`、`DIFFUSION_LMHEAD_BATCH_CAP_MB`、
`DIFFUSION_VRAM_HEADROOM_MB`、`DIFFUSION_DEVICE_COPY_BUDGET_MB`、`DIFFUSION_SEGMENTED_DECODE`、
`DIFFUSION_PIN_STREAMED`、`DIFFUSION_FUSED_PREFILL_ATTN`、`DIFFUSION_NO_DEVICE_SAMPLE`、
`DIFFUSION_DEVICE_SAMPLE_FORCE`、`DIFFUSION_IMAGE_BIDIRECTIONAL`、`DIFFUSION_ASYNC_COMPUTE`、
`DIFFUSION_CPU_ATTN_FAST`、`DIFFUSION_CPU_MOE_CHUNK`、`DIFFUSION_CPU_LEGACY*`）已随之移除；设置其中任何一个都会
提示取代它的方式。

## 6. 服务端行为

当 Web UI 承载 DiffusionGemma GGUF 时：

- `/api/chat` 会进入 diffusion 路径。
- 流式输出发送 `replace` 事件而不是 token append，因为每一步都会重新修正整个 canvas。
- 在 `done` 事件前会先发送最终定稿 replacement。
- 并发请求共享一个后台 diffusion scheduler，并在 block 之间被接纳。
- Jev 结构化读取或新一轮对话的图像编码不必等聊天的整个 block：scheduler 会在下一次前向之前把模型交出，
  所以只需等正在进行的那一次前向。block 的输出不变。
- 每个请求都在运行时中拥有自己的提示缓存（每个 block 预填充一次，由下一个 block 延伸），每一步都在其上运行该请求的 canvas。
- 在 `cpu` 与 `ggml_cpu` 上，服务端对该模型跳过启动时的共享提示预热。DiffusionGemma 不在请求之间保留
  任何状态，而这次预热是在打开端口之前，按每种思考模式各对 256 token 的 canvas 做一次完整的 48 步去噪
  （在上文的 `cpu` 改动之前，`cpu` 上第一次就用了 252.8 s）。GPU 后端保留预热：它会在第一个请求之前记录提示与 canvas 的计算图。
- 图像回合需要用 `--mmproj` 加载视觉塔。每张图片在上下文检查之前展开为其软 token
  span；编码后的 span 归 scheduler 中的该序列所有，每当该序列的缓存预填充这些行时都会写到占位行上，因此图像请求
  与文本请求可以一起批处理。普通聊天拒绝音频，没有原生视频通路：OpenAI 的 `video_url` 会被拒绝；
  在 Web UI 中上传的视频只以抽取出的帧送入模型，每帧按普通 `<|image>` 渲染（没有
  `<|video>` 标记，也没有帧时间戳）。独立的 [Jev 端点](jev_zh-cn.md#文件文档视频与音频)支持
  有界视频抽帧，以及 ASR 配套服务的语音转录。

Ollama 与 OpenAI 兼容适配器仍通过 `ChatStreamWithMetricsAsync` 使用 append-oriented
响应形状。它们可以返回 DiffusionGemma 的最终文本，但实时去噪预览与 `replace`
帧只在 Web UI 中提供。

模型使用 Gemma 4 的 channel 语法：一张 canvas 可能以 `<|channel>thought\n` 开头，或
用一个孤立的 `<channel|>` 关闭提示里打开的思维块，然后才是答案。该架构注册为
`diffusion-gemma` 聊天协议（`Gemma4OutputParser`，始终必需，提示仍由 GGUF 模板渲染），
每一帧预览和最终文本都经过该解析器：除非请求开启推理（`"think": true` 时以
`reasoning_content` 返回），思维块会被丢弃，channel 标记永远不会到达客户端。此前
原始 canvas 被原样返回，OpenAI 的回答以字面的 `<|channel>thought` 标记开头。

该检查点经常在思维块内部作答，而且从不写出结束的 `<channel|>`。已完成的 canvas 不是被截断的
思考，因此当解析结果只有思维块、没有答案时，这段文本会作为答案返回（不会再作为推理重复返回）。
如果不这样处理，Q4_K_M 文件上 6 个一句话问题中有 3 个返回空内容。同样，若 canvas 上只有模型
自行写出的工具调用，它会以文本形式显示，而不是返回空答案；若同时还有其他答案文本，该调用会被丢弃。

工具调用会被直接拒绝：加载 DiffusionGemma 模型时，`/v1/chat/completions` 对任何带
`tools` 或 `tool_choice`（`"none"` 除外）的请求返回 HTTP 400（`{"error": ...}`，
`invalid_request_error`），因为分块扩散的一轮没有可以回填结果的工具循环。
`/v1/responses` 与 Ollama 端点 `/api/chat/ollama` 以同样方式拒绝 `tools`。内置的
skills / 代码执行工具以及子智能体协调工具也永远不会提供给该系列（协议条目声明
`RendersToolDeclarations = false`），因此 `--code-exec`、skills 发现与子智能体委派
都不会改变扩散请求的任何行为。

## 7. 测试覆盖

[`DiffusionGemmaRuntimeTests`](../../InferenceWeb.Tests/DiffusionGemmaRuntimeTests.cs) 不需要权重：基于 26B-A4B
骨干的合成 diffusion-gemma 文件（融合 gate|up 专家、V = K 全局层、32 token 窗口、16 行 canvas、自条件 MLP 与
encoder scale），覆盖 canvas 步骤的注意力（每个 canvas 行读取其他所有行；低于第一个 canvas 行窗口的提示 token
不会影响 canvas，窗口内的第一个会影响）、不提交任何内容的步骤、延伸已缓存提示与重新预填充的一致性、统一调用、
固定的 GGML 后端与托管后端的一致性、计算图采样器与主机算术的一致性、两种采样器上的生成、批量 canvas 与单独运行
的一致性、scheduler 的批量 block、主机上的路由专家，以及结构化读取与其全词表参考的一致性。它是
`SharedRuntimeConformanceTests` 中该家族的专用测试套件。

[`DiffusionGemmaTests`](../../InferenceWeb.Tests/DiffusionGemmaTests.cs) 与
[`JevStructuredReadTests`](../../InferenceWeb.Tests/JevStructuredReadTests.cs) 通过 `TS_TEST_MODEL_DIR` 在真实 GGUF 上
按需运行（`TS_TEST_BACKEND` 选择后端）：有限的 logits、端到端生成、提示缓存一致性、重复 token 与显存驻留保护、
批量 decode，以及选中标签读取与全词表读取的一致性。
[`DiffusionGemmaInterleaveTests`](../../InferenceWeb.Tests/DiffusionGemmaInterleaveTests.cs) 检查在一个 block 的前向
之间插入 Jev 读取既不改变 block 也不改变读取结果。

[`DiffusionGemmaProtocolTests`](../../InferenceWeb.Tests/DiffusionGemmaProtocolTests.cs) 不需要权重，固定
channel 解析：除非请求否则丢弃 thought 块，未闭合的 thought 块即为回答，自发的工具调用以文本呈现。

图像输入还有两个测试套件。
[`DiffusionGemmaVisionConcurrencyTests`](../../InferenceWeb.Tests/DiffusionGemmaVisionConcurrencyTests.cs) 不需要权重，
固定图像 span 归属于序列，因此两个同时进行的图像请求不会互相覆盖。
[`Gemma4VisionOracleTests`](../../InferenceWeb.Tests/Gemma4VisionOracleTests.cs)（通过 `TS_DIFFUSIONGEMMA_VISION_DIR`
按需运行）把从分片加载的视觉塔与 Hugging Face 参考实现的 NumPy 转写（`eng/diffusiongemma-vision-oracle.py`）比较。

[`eng/validation/DiffusionGemmaAB`](../../eng/validation/DiffusionGemmaAB/Program.cs) 在真实检查点上比较两个构建
（teacher forcing 的 logits、生成、计时、批量 canvas、结构化读取）；其报告应放在已忽略的 `artifacts/` 中。

## 8. 后续工作

- 当 Ollama/OpenAI 适配器提供面向扩散的兼容接口后，补充专门的 API 示例。
- `ggml_metal` 与 `ggml_vulkan` 上的设备端采样（它们的计算图带有相同的采样算子，尚未在其上验证），以及多个请求的
  canvas 在一步中的设备端采样。
- 如果运维需要按扩散批次观察，把更多 diffusion scheduler 指标并入 `/api/queue/status`。
