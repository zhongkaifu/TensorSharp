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
- **已验证（2026-11-17），Omni 媒体输入端到端**：与 llama-server（`--mmproj`，相同 GGUF 与 mmproj，贪心解码，
  完全相同的 OpenAI 请求体；见[媒体输入](#媒体输入)）对比。7 个图像提示（照片、报纸版面、流程图、收据、
  2048 行的风景图、一次两张图）、8 段 ASR 音频（6 到 47 秒）、2 个音频问答与 2 个图像多轮对话的提示 token 数相同
  （一段 MP3 例外：两边解码器的样本数相差约一行），并且这 19 个用例中在 ggml_metal 上有 18 个**回复逐 token 一致**
  （一个多轮回复在 97 个相同 token 后分叉），在 MLX 上 18 个一致（一个流程图描述在 93 个 token 后分叉）；
  在 ggml_cpu 与托管 CPU 上运行的 4 个图像与音频用例全部一致。7 段已知原文的音频中有 5 段 ASR 词错误率为 0，
  另外两段各错一个词，且两个引擎错得完全相同。视频按模型自己的方式读取（2 fps 并带音轨），布局与 llama.cpp
  相差至多一行音频（见下文）；由于两边解码出的帧不同，回复在 6 到 14 个 token 后分叉。
- **端到端（文本，2026-11-17，`eng/lfm3/e2e`）：** 服务器的 OpenAI chat completions、Responses、Web UI 接口与 CLI 由
  20 个脚本化场景检查（流式、停止串、随机种子、思考及其预算、工具调用与 `tool_choice`、结构化输出、前缀复用、取消、健壮性），
  覆盖 ggml_metal 上的 Q4_K_M、Q8_0、QAD-Q4_0、BF16、NVFP4 以及作为文本模型的 LFM3-Omni-2B，并在 MLX、ggml_cpu 与托管 CPU
  上运行子集。与同一 GGUF 上的 llama-server 对比（贪心解码、相同的渲染提示）：GSM8K（关闭思考 200 题、开启思考 50 题）在
  Q4_K_M（81.5% / 98%）与 Q8_0（86% / 98%）上得分相同，关闭思考时 92-93% 的回复文本完全一致；8K-128K 的大海捞针在两个引擎上全部
  命中且回复一致；33 条指令遵循与 35 个函数调用用例得分相同或更高，差异来自下文两项行为（关闭思考时的游离推理、
  `tool_choice`）。4 与 16 个并发对话、n-gram 推测解码（回复一致）与客户端取消均正常。长提示的预填充速度与 llama.cpp 相当
  （见下一项；早先“32K 时 3,000 对 5,400 token/s”的数字计时的是 llama-server 处理一个前半段已被上一请求留在缓存中的提示）；
  长提示之后的解码速度相当。
- **速度（2026-11-17，M5 Pro，ggml_metal，`eng/lfm3/bench`，每次运行与同一 GGUF 上的 llama.cpp `d3c3e242c` 交替进行，机器空闲）：**
  LFM3-1.7B Q4_K_M 在 128、512、2,048 个提示 token 时预填充 4,320、6,043、5,971 token/s，llama-bench 为 4,471 / 5,962 / 5,883
  （Q8_0：4,446 / 6,298 / 6,210 对 4,544 / 6,195 / 6,084）；16K 与 32K 时为 4,952 与 4,129，对 4,932 / 4,126；解码在深度 0 时
  195 token/s 对 193（4K 时 185 对 181）。经由服务器，首个 token（`max_tokens` 为 1）在 40 到 2K 提示 token 时与 llama-server
  相差 5% 以内，在 16K-32K 时晚 3-5%。在被其他任务占用的机器上，TensorSharp 的多步提示比 llama.cpp 多损失 5-10%。MLX 对 512
  个以上 token 的提示预填充快于 ggml_metal（Q4_K_M 在 512 时 6,589 token/s），但解码慢约 15%（165 token/s，来自每步的主机开销）。
  CPU 路径见下面两项 CPU 条目。
- **CUDA 与 Vulkan（2026-11-17，一块 NVIDIA A40，sm_86；同一 GPU 上 llama.cpp `d3c3e242c` 的 CUDA 与 Vulkan 构建）：** 文本解码器、
  Omni 媒体输入与语音输出均可在 ggml_cuda、直连 `cuda` 后端与 ggml_vulkan 上运行。与 llama.cpp 做教师强制对比（5 个提示 × 64 步，
  其中一个 9,294 token）：ggml_vulkan 上 320 行中 top-1 不一致为 0-1 行，CUDA 路径上最多 7 行，且全部发生在近乎打平之处——与
  llama.cpp CUDA 的距离和 llama.cpp 自身 Vulkan 运行相当，因为 ggml-cuda 与直连 CUDA 会把每个量化矩阵乘的激活重新量化为 q8_1
  （BF16：各处 0-1 行）；五种量化的困惑度均在 llama.cpp 的 0.4% 以内。服务器端到端场景、CLI 检查以及 4 与 16 个并发对话在三条路径上
  全部通过；语音输出经 ASR 往返无词错误（在 CUDA 上与 llama-tts 并非逐位一致：其采样器从相同隐藏行抽到其他码）。
- **A40 上的速度（`eng/lfm3/bench/gpu-bench.sh`，每个分支与 llama-bench 交替运行，llama-bench 取开启与关闭 flash attention 中更快者）：**
  在 Q4_K_M、Q8_0、QAD-Q4_0 与 BF16 上，ggml_cuda 在所有测试长度上与 llama.cpp CUDA 持平或更快（-1.5% 到 +17%）。Q4_K_M：在
  512 / 4K / 16K / 32K 个提示 token 时预填充 17,004 / 18,283 / 17,132 / 15,156 token/s，对 17,003 / 17,946 / 16,961 / 15,390；
  在深度 0 / 4K / 16K / 32K 时解码 347 / 337 / 309 / 280 token/s，对 334 / 327 / 299 / 268（Q8_0、QAD-Q4_0 与 BF16 同样领先；NVFP4
  的提示慢 9-11%，解码快 3-5%）。直连 `cuda` 后端在 Q4_K_M 上解码比 llama.cpp 低 16-19%（277 / 264 / 246 / 225），预填充约为其
  60%：其量化提示矩阵乘每一步都要把所有权重反量化为 F16，而 ggml-cuda 直接在量化块上计算。ggml_vulkan 对比 llama.cpp 的 Vulkan
  构建：Q4_K_M 预填充 10,119 / 10,430 / 9,258 / 7,749 token/s，对 11,021 / 10,833 / 10,600 / 9,633，解码 272 / 251 / 242 / 217，
  对 295 / 283 / 254 / 226（落后 4-20%，主要是步与步之间的主机时间）；BF16 与 NVFP4 持平。
- **A40 上的服务（`eng/lfm3/bench/serve-vs.sh`：所有引擎使用同一客户端、随机提示、关闭前缀缓存；BF16 对比 vLLM 0.34 与 SGLang，
  后者的短卷积在这块 GPU 上走 torch 回退）：** 单个请求时，ggml_cuda 解码更快（每 token 6.7 ms，对 7.3 与 7.2 ms），128 token
  提示的首 token 更早（26 ms，对 56 与 43 ms），但 4K token 提示的首 token 更晚（275 ms，对 204 与 192 ms）。并发时服务器落后：
  4 / 16 / 64 个请求时每 token 9.4 / 17.4 / 51 ms，对 vLLM 的 7.6 / 8.1 / 11.0 ms，64 个请求时输出吞吐约为其五分之一（1,203 对
  5,506 token/s），原因是批处理步骤中每个序列的注意力与短卷积仍是各自独立的算子。服务器占用 4-6 GB 显存（ggml_cuda），而 vLLM 与
  SGLang 预留约 40 GB。在 Q4_K_M 上对比 llama-server（CUDA 构建，槽位数与请求数相同），两条 TensorSharp 路径在 1、4、16 个
  并发请求时解码都更快（ggml_cuda 每 token 3.0 / 5.8 / 14.1 ms，对 3.8 / 7.0 / 20.1 ms）。
- **CPU 路径的质量（2026-11-18，M5 Pro，CPU 优化之后）：** 在相同 GGUF 上把 ggml_cpu 与托管 CPU 执行器同 llama.cpp
  （`d3c3e242c`，仅 CPU 构建）的 CPU 后端对比。CPU 上的矩阵乘会把激活值舍入到 8 位，因此两个只在求和顺序上不同、都正确的
  CPU 实现，其 logits 会相差 2-4%：llama.cpp 自己的不同 CPU 内核之间就是如此（320 行教师强制对比中相对 L2 中位数为
  1.7-3.8e-2）。沿 llama.cpp 的生成历史做教师强制（5 个提示，14 到 9,294 个 token，各 64 步），ggml_cpu 落在这一噪声底的
  1.03-1.16 倍，托管 CPU 为 1.17-1.37 倍（与 llama.cpp 关闭 llamafile 与重排的内核变体比较相对 L2 中位数）：Q8_0 与
  QAD-Q4_0 上为 1.8 与 2.0-2.1e-2（变体 1.7e-2），Q4_K_M 上为 4.4e-2 与 4.7e-2（变体 3.8e-2），NVFP4 上为 1.9e-2 与
  2.4e-2（变体 1.8e-2，ggml_cpu 的 5 处翻转中有 4 处变体自己也翻转）。每个文件
  320 行中 top-1 差异 ggml_cpu 为 1-5 处、托管 CPU 为 0-9 处，均位于近似平局处（llama.cpp 自己的前两名 logit 相差不到
  0.5），只有 Q4_K_M 的两行例外，而 llama.cpp 自己的内核变体在这两行上也翻转或相差 18%。两条路径离 llama.cpp Metal 运行的
  距离与 llama.cpp 自己的 CPU 运行相同（量化文件上相差在 7% 以内）。ForcedLogitProbe 的通过门槛（相对 L2 1e-3，即 GPU
  路径的标准）在每次 CPU 运行上都不通过，llama.cpp 的变体也一样。BF16 在两条路径上都不再舍入激活值：与 llama.cpp CPU 运行相差 4e-3，与其 Metal
  运行相差 4e-4，比 llama.cpp 自己的 CPU 运行离 Metal 近十倍。提示的最后一层只对输出行运行前馈（与 llama.cpp 的图相同），
  因此在 CPU 路径上该行的矩阵乘走单行内核：其 logits 与整宽矩阵乘相比最多相差 8.3e-7（相对 L2），之后的各行不变；在 42 个
  测量用例中有 14 个出现差异，top-1 全部相同（ggml_cpu 的 Q4_K_M 与托管 CPU 的 BF16 在提示最后一段 512 行分块的行数为 4
  的倍数时；ggml_cpu 的 BF16 从 64 行起；两条路径上的 Q8_0 与托管 CPU 的 Q4_K_M 从不变化）。Q4_K_M 的困惑度（32 x 1,024 token）ggml_cpu 为 26.79、托管
  CPU 为 26.75，llama.cpp CPU 为 26.79。20 个服务器场景在两条路径上均通过，8 个 CPU 语音合成测试也通过（引擎内合成与独立
  合成逐位一致）。GSM8K（前 200 题，关闭思考，Q4_K_M）ggml_cpu 得分 79.0%、托管 CPU 为 80.0%，llama.cpp CPU 为
  81.0%（配对 McNemar 检验 p = 0.50 与 0.82；Metal 上两个引擎都是 81.5%）。这次运行先在 50 题时停过一次，当时
  ggml_cpu 以 37 对 43 落后（p = 0.07；对比更早一次 llama.cpp CPU 运行时 p = 0.03）。只按回复内容计分（关闭思考时客户端
  读到的部分）时，两条路径曾都是 76.5%，llama.cpp 为 81.0%（p = 0.09）：对随后被 max_tokens 截断的回复，TensorSharp
  把多余 `</think>` 之前的文字移入了 `reasoning_content`（ggml_cpu 有 5 道、托管 CPU 有 7 道的答案只在那里）。现在这类
  回复按其流式回复的方式返回（见下文"推理"）：在 ggml_cpu 上重跑，200 道回复与上次运行逐字相同，只按内容计分为 79.0%，与
  整段回复得分一致（对 llama.cpp CPU 的 81.0%，p = 0.50）；托管 CPU 的已记录回复按同样方式重新计分为 80.0%
  （p = 0.82；未重跑）。CPU 上的贪心回复会在几百个字符内于近似平局
  处与 llama.cpp CPU 的回复分叉，llama.cpp 自己的 CPU 与 Metal 回复之间也是如此。4 个并发对话（Q4_K_M）在托管 CPU 上与逐个运行逐 token 一致；在 ggml_cpu 上会在近似平局处
  分叉（与 llama.cpp 相同），因为批量解码步经 ggml-cpu 的 GEMM 读取重排后的权重，而单条解码经其 GEMV
  （`TS_GGML_CPU_REPACK=0` 可恢复一致，但提示速度减半）。以下在优化前验证过、未重做：8K 与 32K 的大海捞针、CLI 检查、
  LFM3-Omni 的图像与音频输入（提示 token 数与 llama-server 相同）以及语音输出（回读识别的词错误率为 0）。
- **CPU 路径的速度（2026-11-18，6 个性能核 + 12 个能效核的 M5 Pro，`eng/lfm3/bench/run-matrix.sh cpu cpu-serve
  cpu-omni`）：** TensorSharp 用默认设置，llama.cpp 取其最佳（Accelerate BLAS 与纯 ggml-cpu 两种构建中较快者，提示用 18
  线程，解码用 16 线程——12 线程在深度 0 与 4K 时慢 4-11%，BF16 与 16K 时慢 21%，18 线程慢 6%），开启 flash attention，在
  空闲机器上交替 3 轮（服务测试为 2 轮）。解码取两边各 5 次运行的中位数，之前先跑一次不计时的运行（评审后重测：第一版矩阵
  拿 TensorSharp 两次运行中较快的一次对比 llama.cpp 两次的平均，掩盖了托管路径 JIT 预热的那一次）。数据表见
  [基准测试](#基准测试)。*ggml_cpu* 在每个文件上解码都快于 llama.cpp（深度 0 与 4K 时 1.03-1.16 倍，16K 时 1.60 倍，BF16
  为 3.9 倍——ggml-cpu 的 BF16 点积在 ARM 上是标量实现），512 token 及以上的提示为 0.93-1.00 倍（NVFP4 为 1.11-1.20
  倍），128 token 的提示为 0.91-0.97 倍；同一格在不同测量批次之间会变动最多 5%（Q4_K_M pp512 在六个批次中为 0.93-0.98
  倍）。服务单路请求时解码与 llama-server 一样快（每 token 6.4-6.5 对 6.5-6.6 ms），首 token 延迟长 20-75 ms（512 token
  提示 0.60-0.65 对 0.53-0.60 s），4 路并发的总吞吐为其 0.97-1.08 倍；512 x 512 图像编码耗时约为 llama.cpp 的 0.6 倍
  （热编码 0.58 倍，进程内首次编码 0.66 倍），音频编码为 0.48-0.50 倍（含 mel 前端为 0.50-0.63 倍），两个引擎都用 18
  线程；语音为 6.8-7.0 倍实时，llama.cpp 为 4.2 倍。*托管 CPU* 的 NVFP4 提示为 llama.cpp 的 1.4-2.0 倍、BF16 解码为
  3.5 倍，Q4_K_M 512 token 及以上的提示为 0.92-0.95 倍，Q8_0 与 Q4_0 的同类提示为 0.77-0.84 倍，这三个文件的 128 token
  提示为 0.54-0.70 倍（JIT 完成分层编译后为 0.70-0.88 倍），量化文件的解码为 0.70-0.86 倍（16K 时 0.94 倍），BF16 提示为
  llama.cpp Accelerate 构建的 0.44 倍。与 vLLM 0.34 的 CPU 后端相比（fp16，带其 macOS short-conv 变通；它在这里没有量化
  CPU 路径），512 token 提示：ggml_cpu 在 Q4_K_M 上首 token 0.64 s 对 1.9 s，解码每 token 6.2 对 37.9 ms（该测试工具的
  TPOT；llama-server 在其中为 6.6 ms）；BF16 上为 1.1 s 与 17.2 ms。相关开关见[环境变量矩阵](../env_var_feature_matrix_zh-cn.md#矩阵外的-ggml-cpu-后端开关)。
- **尚未验证：** 设备上的 TensorAgent 应用；CUDA 上的张量并行与多 GPU 布局（只有一块 GPU 可用）；`config/lfm3-1.7b-nvfp4.json`
  的 FP4 张量核心路径（Blackwell GPU 上的直连 CUDA：A40 没有 FP4 张量核心，NVFP4 文件在其上走的是其他内核）。vLLM 在 A40 与 Mac 的
  CPU 上都对比过，SGLang 只在 A40 上对比过（其 CPU 后端只支持 x86 AMX）。

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

F16 下每 token 的 K/V：6 个全注意力层 × 4 头 × 128 × 2 × 2 = 12,288 字节，与 llama.cpp 的 iSWA 缓存相同。五个滑动
窗口层各保留 768 个单元的环形缓存（256 键窗口加 512 行的提示步长，按 256 对齐：每个序列 7.5 MB），因此环形缓存将要回绕时，
提示按每个序列至多 512 行的步长运行；若 11 层都使用线性缓存则为每 token 22,528 字节。短卷积状态按序列固定且不能回退，
因此前缀只能精确复用（与 Qwen 3.5 系列的递归状态规则相同）。

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
- 关闭思考时 1.7B 仍常常推理：它写出推理（有时先自己写一个 `<think>`），自行关闭推理块后再作答（Q4_K_M 贪心解码下，
  200 道 GSM8K 中有 52 道、33 条指令提示中有 5 条如此；llama.cpp 生成的 token 相同）。模型卡也提示关闭思考"可能不稳定"。
  TensorSharp 渲染的关闭思考提示与 llama.cpp、Hugging Face 完全相同（200 道 GSM8K 提示及 50 道开启思考的提示 token id
  全部一致；`eng/lfm3/prompt-parity.py`），因此多余的 `</think>` 是模型自身的行为。TensorSharp 如何返回这类回复取决于
  客户端能否撤回已显示的文本，以及模型是否写完了回复：
  - 能撤回文本的流（Web UI、TensorAgent），以及模型已写完的非流式回复（OpenAI、Ollama、Responses）：4,096 字符内出现的
    `</think>` 使其之前的文本成为推理，以 `reasoning_content`（OpenAI）或 `thinking`（Ollama、Web UI）返回；Web UI
    立即显示文本，在关闭标记到来时撤回推理部分。
  - 不能撤回文本的流（OpenAI、Ollama 与 Responses 的流、CLI 交互模式）：回复在生成时即作为内容流出，`</think>` 及其前后的
    文本都包含在内，与 llama.cpp、vLLM 和 SGLang 的返回方式相同。不做任何缓冲，短答案逐 token 到达。以模型自己写出的
    `<think>` 开头的回复，该推理块作为推理流出。
  - 被截断的非流式回复（`finish_reason: length`：max_tokens、思考预算、重复保护）按其流式回复的方式返回：推理和
    `</think>` 留在内容中，也不计入 `reasoning_tokens`。此时关闭标记之后的答案并不完整，而之前的推理常常已给出结果：
    在 GSM8K（前 200 道，关闭思考，max_tokens 1,024）上，ggml_cpu 有 5 道、托管 CPU 有 7 道回复的答案只出现在关闭标记
    之前，且都在其后的答案中被截断，只读内容的客户端因此丢失了答案。无论是否经过服务器的子代理循环都是如此：对这类
    客户端，循环的每一轮都等到该轮结束后再输出。
  因此模型已写完的非流式回复与其流式回复在推理所在位置上仍可能不同；客户端在下一次请求中发回其中任一种形式，都能延续
  该对话的 KV 缓存。模型自己打开的推理块像 Gemma 4 的未请求通道一样被限制（最多 64 个 token，在换行处关闭）。
- LFM3-Omni 不做推理：开启思考时它会立即关闭推理块。

### 工具调用

模型输出 Qwen3-Coder XML，并行调用之间以空行分隔。`Qwen35OutputParser` 负责解析，与 llama.cpp 为该模板使用的
`qwen3-coder` 解析器一致：推理在 `</think>` 处结束，或在推理过程中写出 `<tool_call>` 时结束；声明为字符串的参数保留原文，
其他声明类型按 JSON 解析。工具结果渲染为 `<|im_start|>tool\n<tool_response>\n…\n</tool_response>`，只依赖工具消息本身，
因此工具轮次生成的 token 会被拼回，下一轮延续实时缓存。`tool_choice: required` 与指定函数由语法（`QwenXmlToolGrammar`）强制：
回复必须是对已声明函数（指定时只能是该函数）的合法调用，必需参数齐全，取值符合声明的类型与枚举；请求开启思考时语法在
`</think>` 之后生效。`auto` 与 `none` 不受约束（`none` 不把工具放进提示）。llama-server 对该模板不强制指定函数。

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
以及 HF（成对标记中放 N 个 `<|image|>`）一致。一条消息有哪些媒体项由同一条规则决定，渲染器与展开器共用
（[`Lfm3MediaLayout`](../../TensorSharp.Runtime/Lfm3MediaLayout.cs)）：静态图与视频按附件顺序，然后是音频。
以文件给出的视频（`ChatMessage.VideoFilePaths`）从文件读取；只有没有给出文件时，宿主抽好的帧才作为视频。

### 停止 token

`<|im_end|>`（GGUF 的 EOS）与 `<|endoftext|>`，与 llama.cpp 的词表一致。Omni 的 `generation_config.json` 还列出
`<|text_end|>`（125014）；llama.cpp 只用它结束语音与交错生成，在 LFM3-Omni-2B-Q4_K_M 的 12 次贪心文本对话中它从未出现，
因此文本对话不以它停止。

## 媒体输入

加载投影器（`--mmproj mmproj-LFM3-Omni-2B-Q8_0.gguf`；配置文件已写明）后，LFM3-Omni 接受图像、视频（含音轨）与音频。
这一个 mmproj 同时包含视觉塔（`lfm3vl`）与音频塔（`lfm3a`），两者都从它加载。

| 入口 | 图像 | 音频 | 视频 |
|---|---|---|---|
| CLI | `--image`（可重复） | `--audio` | `--video clip.mp4` |
| `/v1/chat/completions` | `image_url`（data URI） | `input_audio` / `audio_url` | `video_url`（data URI） |
| `/v1/responses` | `input_image` | `input_audio` | `input_video` `{url}` |
| `/api/chat`（Web UI、TensorAgent） | `imagePaths` | `audioPaths` | `videoFilePaths`（另带 `isVideo` 与 `imagePaths` 中的帧） |

- **视频从文件读取。** 模型像 llama.cpp 与 HF 一样自行抽帧：每秒 2 帧，按 `round(linspace)` 取下标，帧数限制在
  [4, 384]，所有帧同一尺寸，相邻两帧合成一对并标注 `<t.t seconds>`；音轨（桌面：`PATH` 中的 ffmpeg 或 `TS_FFMPEG`；
  Apple：AVFoundation）整段编码一次，再切分放在每对帧之后。因此对这个模型族，`video_url` / `input_video` 不在宿主上抽帧
  （`ChatProtocol.ReadsVideoFiles`），其 `fps` / `max_frames` 不起作用。没有音频流的视频只用画面；音轨无法解码时会给出警告，
  并只用画面。
- **他处抽好的帧**（客户端只发送帧）按时间配对：相距不超过 0.575 秒的帧合成一对，标注两者的平均时间；更稀疏的帧各自成为
  一个时间块；没有时间的帧按 `1 / VIDEO_SAMPLE_FPS` 计时。这不是模型自己的抽帧方式，也没有音轨。
- **视频上与 llama.cpp 的差异。** TensorSharp 遵循 HF：取精确的源帧（逐帧前进而非定位）、torchvision 的双三次缩放、
  解码器的平均帧率。llama.cpp 用 swscale 缩放、读取 ffmpeg 的恒定帧率输出并按标称帧率计时，因此在标称帧率与平均帧率
  不同的视频上，某个时间标签可能相差 0.1 秒，音轨可能多或少一行（`test-3.mp4`：3,221 与 3,220 个提示 token）。
- **前缀复用遵循媒体身份。** 每张图、每段音频、每对帧与每段音轨都是一个以内容（字节而非文件名）命名的片段：针对同一张图
  的追问会复用到图之后的缓存提示，换一张图则在其片段处停止复用（`Lfm3OmniModelTests`）。
- **并发。** 解码文件、缩放帧与计算音频特征时会释放模型的 GPU 锁；视频在每对帧之间让出锁，超过 30 秒的音频在分块之间
  让出，超过 4,096 个 patch（1,024 行以上）的图像分成两个程序运行，中间让出一次。
- **拒绝。** LFM3-1.7B，或未加载投影器的 Omni，会拒绝图像、视频与音频输入，并在消息中指明投影器；在请求入口允许的
  地方，拒绝发生在写入任何上传之前；CLI 输出一行并以退出码 1 结束。Omni 投影器不能加载到 LFM3-1.7B 上（模型加载拒绝，
  退出码 2），无论是用 `--mmproj` 指定还是在模型旁自动找到：LiquidAI 将 1.7B 作为纯文本模型发布。

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

历史：之后请求中送回的语音助手回合（OpenAI 的 `{"role":"assistant","audio":{"id":...}}`，保留一小时）按生成时解码的样子拼回：
开启模式的 token、文本 token，以及每个已解码帧一个 `<|audio_out|>`，每个占位符取该帧的码本求和行（包括语音结束帧），与 HF
处理器为过去语音回合的占位符所做的一致。因此其提示 token 就是该回复自己的序列（在 LFM3-Omni Q8_0 上与生成逐 token 核对）。
每段连续帧是一个以其编码为键的媒体片段。生成信息未知的 id 按其转写渲染。

限制：没有 `/v1/realtime`；语音请求不保留在前缀缓存中，之后的回合暂时还不能跨越拼回的语音回合复用缓存（未解决问题）。
在 CUDA 上，语音输出通过 ASR 往返检查（无词错误），而不是与 llama-tts 逐位比对。

## 基准测试

在 `/v1/chat/completions`（以及 `/v1/responses`）上设置 `ignore_eos: true`，会越过结束 token 精确生成 `max_tokens` 个 token，
符合 vLLM 与 SGLang 基准客户端的预期（`vllm bench serve --ignore-eos`，SGLang 默认开启）。在每个引擎上都要固定 `think` 与采样参数。

M5 Pro（6 个性能核 + 12 个能效核，2026-11-18，LFM3-1.7B）上的 CPU 路径：每秒 token 数，交替 3 轮的中位数
（`eng/lfm3/bench/run-matrix.sh cpu`）。TensorSharp 用默认设置（提示 18 线程；解码 ggml_cpu 16 线程、托管 CPU 18 线程）。
llama.cpp（`d3c3e242c`）取其最佳：Accelerate BLAS 与纯 ggml-cpu 两种构建中较快者，提示 18 线程，解码 16 线程（其最快的
线程数）。ppN 为 N 个 token 的提示，tg 为在 0 或 4,096 个 token 之后解码 128 个 token：两边各 5 次运行的中位数，之前先跑一次
不计时的运行，评审后另测 3 轮（每次运行前机器忙碌度 0-3%）。提示的同一格在不同批次之间会变动最多 5%：Q4_K_M pp512 在六个
批次中为 llama.cpp 的 0.93-0.98 倍。

| 文件 | 引擎 | pp128 | pp512 | pp2048 | pp8192 | tg | 4K 处 tg |
|---|---|---|---|---|---|---|---|
| Q4_K_M | TS ggml_cpu | 953 | 960 | 885 | 759 | **171** | **148** |
| | TS cpu（托管） | 575 | 898 | 849 | 712 | 128 | 107 |
| | llama.cpp | 1001 | 978 | 919 | 759 | 163 | 139 |
| Q8_0 | TS ggml_cpu | 1178 | 1170 | 1160 | 917 | **118** | **107** |
| | TS cpu（托管） | 880 | 1014 | 934 | 776 | 92 | 81 |
| | llama.cpp | 1255 | 1249 | 1161 | 923 | 115 | 103 |
| QAD-Q4_0 | TS ggml_cpu | 1223 | 1222 | 1158 | 933 | **190** | **164** |
| | TS cpu（托管） | 699 | 994 | 939 | 783 | 131 | 107 |
| | llama.cpp | 1292 | 1288 | 1190 | 938 | 182 | 153 |
| NVFP4 | TS ggml_cpu | 380 | 628 | 645 | 561 | **152** | **135** |
| | TS cpu（托管） | **777** | **820** | **791** | **665** | 114 | 94 |
| | llama.cpp | 399 | 566 | 537 | 488 | 133 | 116 |
| BF16 | TS ggml_cpu | 267 | 487 | 491 | - | **60** | - |
| | TS cpu（托管） | 224 | 228 | 223 | - | 55 | - |
| | llama.cpp | 274 | 520 | 508 | - | 16 | - |

16K token 上下文时 Q4_K_M 的解码为 115（ggml_cpu）、67（托管）与 71（llama.cpp）token/s（不计时运行一次后测一次，3 轮）。托管 CPU 的短提示在最初几次运行中
会随 .NET JIT 对内核的分层编译而变快；稳定状态（10 次运行）下其 pp128 在 Q4_K_M / Q8_0 / QAD-Q4_0 上为 838 / 935 / 892，
llama.cpp 为 952 / 1238 / 1270。CPU 优化之前（`feature/lfm3` 052db7ed，在同一轮中测量），ggml_cpu 在 Q4_K_M / Q8_0 /
QAD-Q4_0 / NVFP4 上的 pp512 为 498 / 605 / 508 / 361（BF16：22，取自 CPU 基线报告），托管 CPU 在五个文件上为
20 / 135 / 25 / 28 / 143。

**线程池一轮之后的托管 CPU**（同一天稍晚，`wip/lfm3-cpu-pool`；同一台与其他任务共用的机器，交替 3 轮，开始时机器忙于 5% 以上的轮次不计；写成区间的格子涵盖这一轮的 A/B 与几小时后对最终分支的复核重跑）。
解码规模的步给进程里的其他线程留出两个核心（`TS_CPU_DECODE_THREADS`，在有性能核与能效核的 Mac 上默认取全部核心减二），
提交线程在让出 CPU 之前先轮询任务的最后几个块，矩阵乘按逐渐变小的列块切分、让快核接手尾部，主机程序也让热点框架代码完成分层编译
（在 `TieredCompilation=false` 下，每个 SiLU 与 softmax 底下的 .NET 向量 `exp` 一直以预编译形式运行）。所有 logit 与之前逐位相同
（五个文件在 18、9、4 线程下的困惑度摘要；五个文件的 teacher-forcing 解码行，取默认宽度、9 线程，Q4_K_M 另取 2 线程的解码宽度）。

| | 之前 | 之后 | llama.cpp |
|---|---|---|---|
| tg128，Q4_K_M / Q8_0 / QAD-Q4_0 / NVFP4 / BF16 | 122 / 87 / 125 / 107 / 53 | 142-143 / 95-101 / 131-140 / 123-125 / 58 | 160 / 111-114 / 178-180 / 131 / 16 |
| Q4_K_M pp128 / pp512 | 880 / 924 | 897-937 / 920-939 | 896-920 / 922-946 |
| 服务，单路：解码 tok/s、TTFT（Q4_K_M，512 + 128） | 76，702 ms | 125-126，575-615 ms | 149-151，589-605 ms |
| 服务，四路：合计 tok/s、TTFT | 87，2,806 ms | 111-113，1.8-2.4 s | 105-115，2.4-2.5 s |
| TTS（Omni-2B Q8_0，`speech-bench.py`）：实时倍数、首段音频 | 3.5 倍，123 ms | 5.1 倍，70 ms | 4.1 倍，112 ms |
| 音频编码器，10 s / 30 s 片段（每个进程 15 次调用，4 轮） | 54 / 110 ms | 44 / 98 ms | 35 / 97 ms |

llama.cpp：解码为 `build-cpu-noblas -t 16`，提示为同一构建的 `-t 18`；服务为同一构建的 `llama-server`，`-t 16 -tb 18`；TTS 为 Accelerate 构建的
`llama-server`，`-t 12`（两者中较好的一个）；音频为其编码器 18 线程。托管音频编码器在前一百次调用中会随 JIT 的分层编译继续变快
（300 次调用之后 10 s 片段为 30 ms）。四路并发时两个服务器的首 token 时间都是双峰的（单轮 1.8-2.5 s，取决于四个提示是否落在同一个提示步里），因此不能据此分高下。

## 当前限制

- 语音输出没有实时端点（`/v1/realtime`），之后的回合暂时还不能跨越历史中的语音回合复用前缀缓存（见[语音输出](#语音输出)）。
- 大量并发请求的解码慢于 vLLM 与 SGLang（见上文速度一项）：批处理步骤中每个序列的注意力与短卷积都是各自独立的算子。
- 直连 `cuda` 后端在其逐元素主机回退上运行 Omni 视觉与音频编码器的部分算子（abs、exp、neg、一次填充以及编码器的部分矩阵乘
  在那里没有适用于其操作数的内核）：图像或音频提示的首个 token 比 ggml_cuda 晚。
- 不提供以实时流方式输入语音（实时端点）；音频整段编码（30 秒一块）。
- 无论客户端顺序如何，一条消息中的静态图都排在视频之前；以 `video_url` 发送的视频在它们之间没有位置。
- 提示中的嵌套工具 schema 会被展平（工具调用语法读取客户端原始 schema）。
- QAD-Q4_0 文件尚未列入 TensorAgent 目录，等待与 Q4_K_M 的质量与速度测量。
- M5 Pro 上 CPU 路径与 llama.cpp CPU 构建的差距（见[基准测试](#基准测试)）：ggml_cpu 在重排量化格式上的提示落后 0-9%，
  主要在 128-512 token（矩阵乘持平；每个 512 行的步骤多出约 25-30 ms 的提示 K/V 复制，而 llama.cpp 原地写入），其 256 token
  以下的 NVFP4 提示仍走 ggml 自己的 NVFP4 乘法，速度只有托管 CPU NEON GEMM 的一半。托管 CPU 的量化文件解码为 0.74-0.95 倍
  （见[基准测试](#基准测试)中的线程池一轮；剩下的是矩阵乘之外的每步开销：图解释器与单线程的逐元素运算约占解码步主线程时间的 20%），
  Q8_0 / Q4_0 的 512 token 及以上提示为 0.77-0.84 倍（llama.cpp 的重排 GEMM 使用 i8mm，而 .NET 10 没有暴露该指令），
  Q4_K_M 的 128-512 token 提示为 0.99-1.02 倍，BF16 提示为 llama.cpp Accelerate 构建的
  0.44 倍，音频编码器耗时为 llama.cpp 的 1.0-1.25 倍（30 s / 10 s 片段）。经 TensorSharp.Server.Host 服务时（单路，512 token
  提示），Q4_K_M 解码为 125-126 token/s，llama-server 为 149-151，首 token 时间持平（575-615 对 589-605 ms）；四路并发合计 111-113 对 105-115 token/s。
- ggml_cpu 在重排副本之外还保留自己在加载时生成的主机副本（融合的 gate|up 与重排列的注意力权重）：Q4_K_M 峰值占用约
  2.0 GB，llama-server 为 1.2 GB。`TS_GGML_CPU_REPACK=0` 恢复较小的占用，但提示速度减半。开启重排时，批量解码步经 ggml-cpu 的 GEMM 读取权重、单条解码经其 GEMV，因此并发的贪心回复可能在
  近似平局处与逐个运行的回复分叉（llama.cpp 也是如此）；托管 CPU 在 Q4_K_M 上保持一致（已测量），但其 BF16 矩阵乘
  按每四个激活行一组计算、单独一行另走一条路径，因此 BF16 回复在托管 CPU 上同样可能分叉。重排与解码线程池只在 M5 Pro 上测量过；BF16/NVFP4
  的 Accelerate 路径仅限 Apple。
