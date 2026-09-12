# DeepSeek V4.1 Flash（`deepseek41`）

[← 返回模型索引](README_zh-cn.md) | [English](deepseek41.md)

TensorSharp 为 V4.1 提供了**运行在 `ggml_cuda` 上的专用推理计算图**，并带有可选的
原生视觉编码器。它复用 DeepSeek 的整模型加载器与调度器，注意力、Engram、残差连接
与对话处理则是 V4.1 专属的。本卡片描述已实现的路径及其限制。涉及模型质量与性能的
结论需要以[验证报告](../deepseek41_validation.md)中记录的实测产物为准。
同一套计算图也能在 `--backend ggml_cpu` 上加载，但那条路径是为了在没有 GPU 时运行
和检查该架构，而不是拿来对外服务——见
[在 ggml CPU 后端上运行](#在-ggml-cpu-后端上运行)。还有第二条不需要 GPU 的路径，而且
完全不带 ggml、也不依赖任何原生库：`--backend cpu` 用纯 C# 的
`DeepSeek4CpuExecutor` 跑 V4.1，用托管代码实现同一套计算图，并对齐 PyTorch 参考实现。
两者都是正确性与可移植性通道，而不是服务通道——见[后端](#后端)。

[官方模型](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)声明的架构是
`DeepseekV41ForCausalLM`。它的文本网络有 40 层、hidden size 5120、64 个 query head
（每个 512 个分量）、384 个路由专家 top-6、一个共享专家，专家中间维为 2304，声明的
上下文长度为 1,048,576 token。V4.1 与 V4 的差异会影响每一次前向，把 GGUF 的架构名
改成 `deepseek4` 是无效的。

## 准备 Q2_K 检查点

目前处于验证状态的产物是
[vcruz305/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF/tree/8e0c4de3cb6519bfc11ed69dc87184b457a57bb5)
在 revision `8e0c4de3cb6519bfc11ed69dc87184b457a57bb5` 上发布的七分片 Q2_K 版本。
七个分片必须放在同一目录，并把第一个分片交给 TensorSharp。该发布包含混合的张量类型，
既有 Q2_K 也有 Q3_K；文件名并不意味着每个张量都是 Q2_K。七个文件合计
264,514,761,248 字节（246.35 GiB）。它们的
[整文件 SHA-256 校验记录](../validation/deepseek41/checkpoint-sha256.json)
列出了每个文件名、期望大小与对应摘要。

V4.1 还需要一个由分词器派生的小体积 Engram sidecar。已发布的 GGUF 并未包含完整的
因果 encoder-decoder 与 Engram 配置：部分 Engram 键仍用旧的 `deepseek4` 前缀，而分词器
的 padding 元数据与 Engram 哈希的 padding 也不一致。准备脚本会读取官方配置与分词器，
而不是猜测这些取值。

在存放模型的机器上，从仓库根目录执行：

```bash
python3 -m venv /workspace/dsv41-tools
/workspace/dsv41-tools/bin/python -m pip install \
  numpy==2.0.2 tokenizers==0.22.2 huggingface_hub

/workspace/dsv41-tools/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="vcruz305/DeepSeek-V4.1-Flash-GGUF",
    revision="8e0c4de3cb6519bfc11ed69dc87184b457a57bb5",
    allow_patterns=["DeepSeek-V4.1-Flash-Q2_K-*.gguf"],
    local_dir="/workspace/models/deepseek41-q2",
)
PY

/workspace/dsv41-tools/bin/python eng/dsv41-prepare.py \
  /workspace/models/deepseek41-q2 \
  --repo deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277
```

准备脚本只下载官方的 `config.json` 与 `tokenizer.json`，然后在 GGUF 分片旁边写出
`deepseek41.engram.bin` 以及溯源文件 `deepseek41.config.json`。它会核对词表大小、
压缩后词表大小与官方 Engram 布局，并记录来源与 sidecar 的 SHA-256。原生加载器会用
该布局校验 GGUF 的张量维度。准备阶段需要 Python，推理阶段不需要。
如果官方文件已经下载到本地，用 `--source-dir` 直接从本地准备。

实现期间锁定的官方文件哈希如下：

| 文件 | SHA-256 |
|---|---|
| `config.json` | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| `tokenizer.json` | `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b` |

## 准备可选的视觉伴随文件

已发布的 GGUF 转换过程丢掉了视觉塔、aligner、可学习的图像分隔符以及逐层的视觉路由
偏置。官方发布把约 970 MB 的视觉/aligner 权重放在一个独立分片里，因此可以在不下载
原始文本权重的情况下准备它们：

```bash
/workspace/dsv41-tools/bin/python -m pip install gguf
/workspace/dsv41-tools/bin/python eng/dsv41-prepare-vision.py \
  /workspace/models/deepseek41-q2 \
  --repository deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277
```

它会在文本分片与 Engram sidecar 旁边生成 `deepseek41.vision.gguf` 与
`deepseek41.vision.json`。
该脚本下载独立的视觉分片，以及分隔符/路由张量所需的少量字节范围，并保留 BF16/F32
存储。它会用 LFS 的 SHA-256 校验完整的独立视觉分片，并记录分隔符/路由权重各自的
范围哈希；它不会下载或校验完整的原始文本分片。
[伴随文件溯源记录](../validation/deepseek41/vision-companion.json)
包含全部 306 个张量、来源 revision、字节范围与输出摘要。
原生加载器在挂载前会检查父模型的分词器指纹与模型维度。要启用图像，在下文的服务命令
中加上 `--mmproj /workspace/models/deepseek41-q2/deepseek41.vision.gguf`。
没有挂载伴随文件时，图像能力保持关闭。

视觉部分默认使用稠密 F32 注意力，与官方视觉塔的注意力算术一致。在 Ampere 及更新的
NVIDIA CUDA 上，BF16 矩阵输入会保持 F32 累加与 F32 输出，直到加偏置并做 BF16 激活
舍入为止。`TS_DSV41_VISION_BF16_GEMM=0` 选择诊断用的 F32 提升矩阵路径。
`TS_DSV41_VISION_FA=1` 选择更快的、使用 F16 中间量的 flash attention；在真实图像的
参考对比中，这条路径的特征差异更大。这些选项只影响图像编码器，与文本注意力无关。
精确边界与实测取舍见
[验证报告](../deepseek41_validation.md#independent-vision-and-mixed-modality-reference)。

做数值排查时，`TS_DSV41_VISION_TRACE_DIR=/绝对路径` 会写出 F32 的 patch、block、norm
与 projector 输出。开启 trace 会保留中间张量并增加设备传输，正常推理与基准测试时请
保持不设置。

编码器使用 32 层双向 transformer、二维旋转位置、带 padding 的 3×3 空间 merge，以及
两层 projector。它输出完整的图像 span，包括可学习的起止标记与逐行的 newline embedding。
图像 token 使用视觉 MoE 偏置，抑制 Engram 注入，并在图像 span 处打断 Engram 的 n-gram。
图文混排的 span 可以跨越 prefill 微批边界。已有的视频抽帧走同一条图像路径。
CPU/CUDA 数值 fixture 与精确预处理检查均已通过。
完整检查点的 CUDA 按层切分运行通过了并发 1/4 下的 25 个图像/视频请求，以及一个长文本
之后的单独图像请求。最终的 routed-TP 配置同样通过了并发 1/4 下的全部 25 个图像/视频
请求。严格的编码器 parity 单独跟踪。
官方配置没有提供音频解码器。V4.1 会拒绝带音频的请求（包括图像+音频的混合请求），
而不是忽略音频。Chat Completions、Responses 与 Web UI 都返回 HTTP 400。
OpenAI 解析器在读取音频负载之前就拒绝音频部分（包括缺失或格式错误的音频），因此后续
的音频附件不会把先前已上传的图像留在半途状态。

OpenAI chat 端点接受包含 base64 图像 data URI 的 `image_url` 部分，支持多张图像以及
历史轮次中的图像。它的 V4.1 `video_url` 扩展会通过既有的视频解码器对 base64 的 MP4、
WebM 或 MOV 采样。例如，消息的 content 数组里可以包含：

```json
{"type":"video_url","video_url":{"url":"data:video/mp4;base64,...","fps":1,"max_frames":3}}
```

每个采样帧成为一个图像 span，并带上以帧序号除以探测到的帧率算出的源时间（秒）。
对可变帧率的片段，时间标签是近似值。
帧保持原始顺序；这是抽帧，不是原生的时序编码器。`fps` 必须大于 0 且不超过 60，
`max_frames` 取 1–64。默认值取自 `VIDEO_SAMPLE_FPS` 与正的 `VIDEO_MAX_FRAMES`，
否则为 1 fps 与 16 帧。超过上限的帧会在整个片段上均匀采样。远程 HTTP 图像/视频 URL
不会被拉取，请发送 data URI。对 V4.1 而言，`/v1/chat/completions`（`image_url`）与
`/v1/responses`（`input_image`）都会在开始流式输出或写入任何图像之前，以 HTTP 400
拒绝远程或格式错误的图像 URL 以及无效/空的图像 base64。
合法的图像 data URI 保持既有的解码与上传路径。展开所有图像之后，文本上下文预算依然
适用。
最终的 routed-TP 主机（托管侧 stage 3,651，原生 `6b3b5ab3…`）通过了全部八项
[图像拒绝检查](../validation/deepseek41/full-checkpoint/tp8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-image-input-rejections.json)
与四项 [Responses 音频拒绝检查](../validation/deepseek41/full-checkpoint/tp8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-audio-input-rejections.json)。
无论流式还是非流式请求，对远程图像 URL、格式错误的图像 base64、以及形状合法或格式
错误的音频部分，都返回 JSON 400 且不产生 SSE。更早的 CPU-offload 主机的八项图像检查
另行保留。

## 运行已实现的路径

需要安装 .NET 10 SDK、CMake、C++ 编译器，以及 `PATH` 上带 `nvcc` 的 CUDA 工具链。
从仓库根目录构建。下面的命令针对被请求的 A40 VM（CUDA 架构 8.6）；换其他 GPU 时
两处架构值都要改：

```bash
TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON \
  TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES=86 \
  bash TensorSharp.GGML.Native/build-linux.sh
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release \
  -p:CudaArch=compute_86 -p:TensorSharpSkipGgmlNative=true

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_CONTEXT=65536 \
  TS_CPU_MOE_THREADS=32 TS_DSV41_TP=0 TS_DSV4_UBATCH=256 \
  TS_DSV41_ENGRAM_WARM=1 TS_DSV41_SPARSE_FA=1 \
  TS_DSV41_COMPACT_RAW_GATHER=0 KV_CACHE_DTYPE=f16 \
  TS_SCHED_MAX_RUNNING_SEQS=4 TS_SCHED_MAX_BATCHED_TOKENS=4096 \
  TS_SCHED_PREFILL_CHUNK=256 TS_SCHED_SOLO_PREFILL_CHUNK=8192 \
  dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /workspace/models/deepseek41-q2/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf \
  --backend ggml_cuda --tp 8 --port 5000
```

宿主构建会把原生库复制到服务端 DLL 旁边。上面这条启动命令使用的是保守基准矩阵的微批
与调度器设置；验证报告里的优化配置用的是另一组参数。`TS_CPU_MOE_THREADS` 要按可用的
CPU 配额来选，并为每次运行记录下来。即便是纯 GPU 放置也要在启动环境里设置它：原生的
CPU 图工作与主机侧归约仍会影响延迟。当前 CLI 也接受 `--cpu-moe-threads N`；两者都给
时请填相同的值，因为原生加载器优先采用为正的环境变量值。

若要改用最终实测的八卡 A40 层放置配置，可在准备好视觉伴随文件后使用下面这条可选命令。
它显式设置了优化参数；上面的保守示例与各项默认值保持不变：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_CONTEXT=65536 \
  TS_DSV4_NGPU=8 TS_DSV4_UBATCH=1024 KV_CACHE_DTYPE=f16 \
  TS_CPU_MOE_THREADS=32 TS_DSV41_TP=0 \
  TS_DSV41_SPARSE_FA=1 TS_DSV41_COMPACT_RAW_GATHER=1 \
  TS_DSV41_ENGRAM_WARM=1 TS_DSV41_ENGRAM_THREADS=16 TS_DSV4_PERF=1 \
  TS_SCHED_MAX_RUNNING_SEQS=4 TS_SCHED_MAX_BATCHED_TOKENS=4096 \
  TS_SCHED_PREFILL_CHUNK=1024 TS_SCHED_SOLO_PREFILL_CHUNK=1024 \
  dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /workspace/models/deepseek41-q2/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf \
  --mmproj /workspace/models/deepseek41-q2/deepseek41.vision.gguf \
  --backend ggml_cuda --tp 8 --n-cpu-moe 0 --cpu-moe-threads 32 \
  --host 127.0.0.1 --port 5000 --max-tokens 2048
```

[实测启动记录](../validation/deepseek41/full-checkpoint/layer8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-launch.json)
保留了原始 VM 路径、二进制哈希与环境。该配置通过了 138/138 项推理用例；其吞吐与限制
记录在下文。稀疏 flash attention 与紧凑 gather 仍是可选项，并带有已记录的浮点差异。
启动与页预热不计入推理测量。

在没有显式线程设置时，纯 GPU 的 V4.1 加载使用调用方的 `TS_DSV4_THREADS`，默认上限为
32；CPU 专家卸载则改用探测到的可用 CPU 并行度。更早的原生构建收不到 CLI 的线程覆盖。
具体来说，首次实测 TP 运行中的 `--cpu-moe-threads 48` **并没有**配置 48 个原生线程：
根据加载器源码，在没有原生环境变量覆盖、也没有 VM 覆盖探测的情况下，其原生线程池推断
为默认的 32。那台原始主机没有暴露线程池宽度的读取接口，所以 32 并非在那里直接测得。
请把这个基线与后续显式设置线程数的实验区分开。

对这个架构，`--tp 8` 表示**用八张 GPU 做按层切分**。启动诊断会说明采用的放置模式。
TensorSharp 默认按可用显存分配整层。
`TS_DSV4_NGPU` 覆盖 GPU 数量。用 `CUDA_VISIBLE_DEVICES` 精确指定本次运行使用的设备。
显式设置 `TS_DSV4_NGPU=0` 表示自动选择可见设备，并把 rank 数校验推迟到原生加载器。

`TS_DSV41_TP=8` 会在这八张 GPU 上额外启用实验性的 **routed-MoE 张量并行**。该设置接受
`0`（关闭）或 `2` 到 `8` 的 rank 数，且必须等于 `--tp` 或 `TS_DSV4_NGPU` 选中的 GPU 数。
自动选择 GPU 时，原生加载器会在枚举可见设备之后再校验数量。取值非法或数量不匹配都会
报错。

在这种模式下，路由专家的 gate/up/down 矩阵沿 FFN 中间维切分，并在所有选中的 GPU 上并发
执行。partial 输出经主机中转的 F32 缓冲归约。注意力、共享专家与各类 cache 仍保持按层
放置。这是一份部分实现的张量并行：它不切分注意力，也不支持分布式张量并行组。主机传输
可能成为吞吐瓶颈，因此该选项并未证明比按层切分更快。首次完整 Q2_K 的 TP 质量/性能运行
已完成，结果比按层切分更慢；见
[实测放置配置](../deepseek41_validation.md#full-checkpoint-routed-moe-tp)。

独立数值 fixture 在 2/4/8 张 GPU 上均通过，包括量化专家分片与整模型 oracle 检查。这些
小规模 fixture 并不能说明完整的 Q2_K 检查点能装进两张或四张 A40。本卡片的 VM 示例用的
是八张；更少的卡数需要足够的 CPU 专家卸载才能装下。
它的 Engram 预热在就绪之前会占用约 60 GiB 主机页缓存。冷加载与预热时间要与热态吞吐
分开记录。

如果权重与上下文放不下，加上 `--n-cpu-moe N` 把前 N 层的路由专家留在主机上，或者用
`--cpu-moe` 卸载全部路由专家。注意力、路由与共享专家仍在 GPU 上。Engram 表始终以内存
映射方式留在主机上；每个输入批次只读取并传输选中的 embedding 行。CPU MoE 卸载与按层
切分都已实现，但它们在你所用硬件与上下文下的吞吐需要实测。与 `TS_DSV41_TP` 组合时，
被 CPU 卸载的前置层保留完整的 CPU 专家，其余层使用路由专家分片。

原生版本 `6b3b5ab3…` 显式把共享专家的 gate/up/down 投影指派到该层所在设备。这修正了
更早的调度器放置问题：在经过 CPU 卸载或 TP routed 分支之后，共享的 gate/up 计算可能被
送到 CPU。相关的
[放置与数值检查](../validation/deepseek41/shared-expert-placement/README.md)
在两张 GPU 上通过了 597/597。更早的完整检查点放置基准仍保留其原始二进制与结果。
使用修正后放置的最终 CPU-offload、routed-TP 与按层切分配置均已完成，见
[放置记录](../validation/deepseek41/final-placements/README.md)。

要做热态的 CPU 卸载基准，请在模型加载之后、开始计时请求之前，单独把被卸载的专家页读入：

```bash
/workspace/dsv41-tools/bin/python eng/dsv41-warm-experts.py \
  /workspace/models/deepseek41-q2/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf \
  --layers 4 \
  --report cpu-expert-warming.json
```

这需要 `gguf` Python 包，并会读取所选各层的全部三个路由矩阵。它的 I/O 时间要单独记录。
这些页仍可被回收；Engram 预热本身不会预热 CPU 专家权重。

### Engram 表放在哪里

默认情况下两张 Engram 表**驻留 GPU**：各自加载到拥有对应层的设备上，计算图用
`get_rows` 直接在量化表上取行。主机完全不读行，因此 CPU 上不做任何反量化，只有行号
跨越链路。启动会打印
`Engram lookup: GPU-resident tables, gathered in-graph`。

之所以能这样，是因为 TensorSharp 保持检查点自身的量化格式。一行 256 个值的 Q2_K 只有
84 字节，因此两张 3.84 亿行的表合计 60.2 GiB。vLLM 与 SGLang 把同样的行存成 FP8 值加
每 32 个元素的块 scale，一行 264 字节——这也是它们默认使用主机表加 FP8 gather kernel
的原因。

放置策略是自动且保守的：这些表会计入按层切分的装箱；如果把它们放上 GPU 会逼出任何
路由专家的 CPU 卸载，它们就改为主机映射，并在启动时说明。设置
`TS_DSV41_ENGRAM_DEVICE=0` 强制主机映射，`=1` 要求驻留 GPU、放不下就失败。
`TS_DSV41_ENGRAM_THREADS`、`TS_DSV41_ENGRAM_WARM` 与 `TS_DSV41_ENGRAM_RANDOM` 只对
主机映射有效；在 GPU 驻留路径上设置了它们，启动时会给出提示。

在八张 A40 上实测，每次都用冷提示词，因此每次都会选到此前未触及的行：

| Engram 放置 | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| 主机映射，不预热 | 207-221 | 28.0-29.2 |
| 主机映射，`TS_DSV41_ENGRAM_WARM=1` | 506-528 | 34.5-35.2 |
| 驻留 GPU（默认） | 532-541 | 35.6-35.9 |

驻留 GPU 的路径还省掉了启动时 130 秒的整表预热，以及预热路径所依赖的 60.1 GiB 主机
页缓存。在确定性检查中，三条路径在 temperature 0 下产出的文本完全一致。

那是 argmax 层面的稳定，而不是逐位一致：ggml 的 CPU 与 CUDA Q2_K 反量化计算同一个
表达式，但设备端可能把乘减合并成 FMA，从而相差最多一个 ulp。如果某次运行必须与 CPU
oracle 逐位一致，请使用 `TS_DSV41_ENGRAM_DEVICE=0`。

### 主机映射的 Engram 表

以下选项仅在表留在主机上时生效——GPU 装不下，或显式设置了
`TS_DSV41_ENGRAM_DEVICE=0` 时会发生这种情况。

在网络存储上，首次访问稀疏的 Engram 行会主导 prefill 与 decode 延迟。每个 token 在每张
表上要哈希出 24 个行号，任何不在页缓存中的行都是一次存储往返：在八卡 A40 上的网络挂载
Q4_K_M 权重上，这意味着**每个 1024-token 的 prefill 分块要花 1435–2565 毫秒做输入准备**，
而页面驻留后只需 56–81 毫秒。decode 的输入准备冷态为每 token 5–15 毫秒，热态为 0.9 毫秒。

因此只要表在主机映射上且页面能够常驻，**预热就是默认行为**。它在模型开始服务之后于独立
线程上运行，而不是在加载期间，所以启动时间不变，预热进行期间请求照常工作（只是更慢）；
启动时会打印预热的数据量，完成时再打印一次。每次模型加载都会把整份权重读过页缓存，从而
挤掉上一次运行的 Engram 页面，这就是为什么每次加载之后都要重做，而不是每台机器做一次。

预热只读这些表，不创建私有副本或 pinned 分配，页面仍可被操作系统回收。当映射的主机权重
再加 8 GiB 装不进探测到的主机/cgroup 内存额度、或 `MemAvailable` 本来就留不住这些表时，
预热会被跳过并给出诊断。

| `TS_DSV41_ENGRAM_WARM` | 行为 |
|---|---|
| 未设置（默认） | 模型开始服务后在后台预热 |
| `1` | 与此前一致，在加载期间同步预热；启动时间增加约 110–210 秒 |
| `0` | 从不预热 |

稀疏读取的映射建议（`MADV_RANDOM`）只在预热结束之后才施加，无论采用哪种预热形式：该建议
会关闭预热本身所依赖的预读。

`TS_DSV41_ENGRAM_THREADS=1..32` 控制常驻查表工作线程数，默认取 16 与硬件线程数中的
较小者。prefill 与 decode 都使用并行取行：单个 token 在每张 Engram 表上要选 24 行互不
相关的数据。设为 1 会让读取串行化。并行取行避免了 decode 期间串行的缺页等待，且不改变
embedding 数值。执行器在同一个工作池任务中交错读取两张表，工作线程完成后再逐表上传行。
staging 以 64 MiB 为单位分组；单张更大的表沿用此前的一表分配上限。

在 Linux 上，并行取行会自动为映射的 Engram 区间请求随机访问建议，以减少不必要的预读。
该提示在可选的整表预热之后应用，除共享的边界页之外，不改变其他张量区间的策略。设置
`TS_DSV41_ENGRAM_RANDOM=0` 可关闭它，`=1` 可强制开启。不设置时，单线程配置保持默认的
映射策略。不支持的平台与被 OS 拒绝的提示都是非致命的。源码对比与配对的 scratch 文件
结果见 [Engram 调查](../validation/deepseek41/cli-gpu-execution/README.md)。

### 后端

`--backend ggml_cuda` 是服务路径：只有它为这个架构的融合算子提供了内核。

`--backend ggml_cpu` 在一个 CPU 设备上运行整个模型，这些融合算子退回到标量实现。
它的存在是为了在没有 GPU 时运行和检查该架构，而不是用于服务：这个检查点每层每个 token
要从 246 GiB 中读出 384 个路由专家中的 6 个。

`TS_DSV41_ALLOW_NON_CUDA_GPU=1` 额外允许 `ggml_vulkan` 与 `ggml_metal`。在那里，普通
计算图跑在 GPU 上，只有架构专属算子落到 CPU 后端，每出现一次就要一次主机往返。它需要
显式开启而不是自动生效，因为那道拒绝原本挡住的是静默回退到"第一个被枚举出来的 GPU"，
而不是一次明确的请求；启动时会逐个列出它作用到的设备。在你的硬件上实测之前，请把它
当作可移植性与正确性通道。

`--backend cpu` 根本不是 ggml 后端：它是纯 C# 的 `DeepSeek4CpuExecutor`，不依赖任何
原生库，也不需要 GPU，并且实现了完整的 V4.1 计算图——比例 1 与比例 2 的块压缩器、
共享的压缩 cache 与索引器 cache（含 lightning indexer 的 top-k）、候选块剪枝、
Engram 表、延迟 hyper-connection 门控、共享专家，以及检查点里训练好的 cache 量化
（原始行 FP8 E4M3、索引器 MXFP4、压缩 cache NVFP4）。它以 atol=rtol=2e-5 对齐独立的
PyTorch 参考实现 `eng/dsv41-reference.py`，并要求贪心 argmax 完全一致，覆盖一次性
prefill、分块大小 1/3/5/8 与 reset——不过那是 fixture 规模的对齐，而不是在已发布权重上
的等价。Direct CUDA 引擎 `--backend cuda` 同样用自己的内核、不经 ggml 运行 V4.1，但还
没有数值门禁。和 `ggml_cpu` 一样，这两条都是正确性与可移植性通道，而不是服务通道；
细节见[在 ggml CPU 后端上运行](#在-ggml-cpu-后端上运行)的末尾。`--backend mlx` 仍然被
拒绝。

### 每张 GPU 一个后端

DeepSeek 架构专属的算子（压缩器、注意力前后处理、MoE 路由与归约、带 clamp 的 SwiGLU、
hyper-connection 门控、top-k 掩码）以 `GGML_OP_CUSTOM` 节点发出，由一个 TensorSharp
后端执行。该后端**包裹**所在 GPU 的 CUDA 后端，并在 `ggml_backend_sched` 中取而代之：
它同时声明 CUDA 设备的算子与缓冲类型以及自己的，把普通节点以图视图的形式转交 CUDA，
并在同一条流上启动融合内核。

如果改成把两个后端并列注册，计算图就会在每次后端切换处被切开，而 V4.1 的一层大约要
切十四次。一张 decode 图曾被切成 565 个平均约六个节点的 split，调度器还会在每个边界上
做一次阻塞式主机同步。包裹方式把它降到每张 GPU 一个 split：

| | 每张 decode 图的 split 数 | Decode 计算 | Decode tok/s |
|---|---:|---:|---:|
| 后端并列 | 565 / 577 | 26.6-27.1 ms | 35.6 |
| 包裹式后端 | 8 | 23.2-23.5 ms | 41.1 |

如预期，prefill 不受影响：一个 prefill split 本身就有毫秒级的工作量，逐边界同步在那里
只是噪声。设置 `TS_DSV4_FUSED=0` 可让这些算子完全退回到原生 CUDA 内核。

启动时会报告每个已初始化的计算设备，以及路由专家的 CPU 卸载层数。使用
`--backend ggml_cuda` 且不带任何 CPU 卸载选项时，全部 40 层都在 CUDA 设备上运行。
`auxiliary CPU worker pool` 这条消息描述的是调度器的主机线程池，并不表示在做纯 CPU
推理。Engram 查表仍使用主机内存，而按层切分会让相邻的层落在相邻的 GPU 上，因此仅凭
单卡利用率低并不能断定发生了 CPU 回退。`TS_DSV4_PERF=2` 报告输入准备与图计算耗时；
`TS_DSV4_PERF=3` 还会记录调度器实际的后端切换。它们是诊断模式，其日志开销会影响吞吐。
见 [CLI 执行调查](../validation/deepseek41/cli-gpu-execution/README.md)。

`TS_DSV41_SPARSE_FA=1` 为单 token 批次、或缓存 key 不少于 16,384 时启用 CUDA 掩码压缩
flash attention。它最多注意 128 个原始窗口 key 加 512 个选中的压缩 key。较短的 prefill
仍用稠密 flash attention，因为在测试用的 A40 上其共享 KV tile 更快。实测的完整检查点
配置显式启用了该选项，但它的默认值仍是关闭。这个选项减少的是注意力计算量；它并不能
消除 prefill 期间共享压缩缓存的跨 GPU 拷贝。

`TS_DSV41_COMPACT_RAW_GATHER=1` 为稀疏的单 token decode 启用原始窗口压缩。它先在持有
这些行的 GPU 上把 128 行可见的原始行 gather 起来，再搬到持有共享压缩缓存的 GPU。被
掩码的重复行把原始前缀补齐到 256 行，使合并后的 768 行 K 张量满足 CUDA 512 分量注意力
的对齐要求。物理 ring、prefill 路径以及不做 gather 的 decode 均保持不变。该选项默认
关闭。在约 8k 提示词下的限定 Q2_K 对比中，并发 1 与 4 的持续 decode 都提升了 13.7%；
严格的 flash 算术差异与完整测量设置见验证报告。

默认上下文分配上限为 65,536 token，除非提供 `MAX_CONTEXT`。`TS_DSV4_UBATCH` 控制前向
微批，V4.1 默认为 256。模型声明的窗口更大，并不意味着某个具体的 GPU 配置能分配或高效
服务那么长的上下文。

### token 批量 decode

多个序列同时 decode 时，它们的 token 会进入**同一张图**，而不是各自一张。一个 decode 步
的开销由读权重（Q4_K_M 下约每 token 9.8 GiB）以及读它们的约 2200 个小 kernel 主导，批量
化让这两项只付一次：只有触及某个序列自身状态的部分才按槽位分叉，对 V4.1 而言就是滑动
窗口环、压缩器状态、lightning indexer 的选择以及 attention 本身。Engram 查表完全不需要
分叉——它暂存的行本来就是每 token 一列，槽位不过是其中一列，只是用该槽位自己的历史来
哈希。

收益上限来自路由：每个 token 各自从 384 个专家里挑 6 个，因此槽位之间的路由专家读取并不
重叠，只有稠密投影、共享专家和输出头是共享的。八卡 A40、Q4_K_M 上的聚合 decode 吞吐：

| 并发请求数 | 串行 decode | token 批量 decode |
|---:|---:|---:|
| 1 | 22.8 | 28.9 |
| 2 | 24.8 | 39.3 |
| 4 | 24.3 | 48.9 |
| 8 | 26.5 | 48.5 |

批量化会改变 GEMM 形状，所以批量步与单独步并非逐位一致，logits 上接近平手的位置可能选出
不同的 token。单独 decode 在 6/6 条贪心提示上复现了自己的输出；批量步在 2–3/6 上与单独文本
一致。同一条提示分别以批宽 2 和批宽 4 运行——同一条代码路径，只是 GEMM 更宽——分歧比例相同，
所以改变输出的是"哪些请求恰好同处一步"，而不是各槽位的接线。需要串行路径及其确定性时设
`TS_BATCHED_FUSED_DECODE=0`。

四个各藏一个不同秘密、长度 10,836 token 的并发文档，得到 4/4 正确答案，且没有任何一份答案
包含其他槽位的秘密——这正是用来检验各槽位的环、压缩缓存与稀疏选择确实彼此独立的检查。

### 为图保留的设备内存

`TS_DSV4_VRAM_RESERVE_MB` 覆盖分层切分打包器在每张卡上留出的空余。默认值按 indexer 的
top-k 临时量、一个微批的激活以及 2 GiB 底线计价。留得太多并不是没有代价：在八卡 A40 的
Q4_K_M 上，5240 MiB 会把三层路由专家挤到主机上，而 3174 MiB 只需一层，差别是 prefill
350 → 480 tok/s。

图缓存现在同时按字节数和条目数设限。一个条目的计算缓冲区随其形状增长，而处于不同位置的
并发序列会产生很多不同形状：过去四个并发的 10.8k token prefill 会占满全部十二个条目并把
某张卡的显存用尽，而这不是可恢复的错误——ggml 的分配器在重新分配前先释放旧缓冲区，因此
一次失败的 reserve 会留下空指针，进程直接死掉，而不是让该请求失败。现在在构建新条目之前
会先释放最久未使用的条目，直到每张卡都能再放下一个与已缓存的最大条目同样大的图，外加一个
底线。`TS_DSV4_GRAPH_CACHE_HEADROOM_MB` 设定这个底线（默认 1024）；设为 `0` 回到纯条目数
上限。

原生 V4.1 请求各自拥有独立的 KV 槽位。因此调度器按"每个允许运行的请求一个上下文"来
计算它那份仅含元数据的块池。调度器默认允许 16 个运行中的请求；示例显式设置
`TS_SCHED_MAX_RUNNING_SEQS=4` 表示四个活动槽位。在上下文 65,536、块大小 256 时，它们
的自动记账容量是 1,024 个块。这既不会预分配额外的 GPU 缓存，也不会放大单个请求的上下文
上限。原生槽位在分配时仍然需要足够的设备内存。显式设置为正的 `TS_SCHED_NUM_BLOCKS`
仍是一个硬性的总量记账上限；把它设得低于活动请求的合计需求会导致提示词重算。

对四个并发 prefill，`TS_SCHED_MAX_BATCHED_TOKENS=4096` 允许在一个全 prefill 的调度步中
每个请求处理 1,024 个 token。一旦有请求开始 decode，`TS_SCHED_PREFILL_CHUNK` 就会限制
其余每个 prefill，其默认值为 256。`TS_SCHED_SOLO_PREFILL_CHUNK` 控制单独一个请求的情形，
其配置默认值为 8,192，并受批处理 token 总上限（默认 4,096）约束。
把两个 chunk 上限都显式设为 1,024，可以避免一个很短的首个请求改变其他请求的 prefill
大小，但更长的混合步骤也会拖慢正在 decode 的请求。做性能对比时，请把这些调度器设置与
`TS_DSV4_UBATCH` 一并记录。

[基准矩阵](../../benchmarks/engine_comparison/benchmark_config_deepseek41.json)
使用上述保守配置，并继承其各后端环境中未列出的设置。要复现该基线，请在启动矩阵前显式
设置 `TS_CPU_MOE_THREADS` 与 `TS_DSV41_COMPACT_RAW_GATHER=0`。若使用本卡片中的模型目录，
请把 `BENCH_DSV41_GGUF` 指向第一个分片；矩阵默认的目录名与此不同。它的文本场景不能替代
验证报告中单独的严格工具、JSON、图像/视频与推理检查。

### 在 ggml CPU 后端上运行

`--backend ggml_cpu` 选择加载器的纯 CPU 分支：只有一个 CPU 计算设备而不是枚举出的
加速器，所有层都在它上面，所有 V4.1 专属算子都走
`ggml_ops_dsv4_fused_cpu.cpp` 里的标量 CPU 实现而不是 CUDA 内核。启动会打印
`compute devices initialized: 1 CPU device(s)` 与
`routed-expert placement: all 40 layer(s) on the explicitly selected CPU device`。

**这是一条正确性与可移植性通道，不是服务通道。**每解码一个 token，都要在 40 层里
各读出 384 个路由专家中的 6 个，来源是 246 GiB 的 Q2_K 检查点，而且跑在通用核心上。
那些参考内核每个节点只跑一个 worker（V4.1 的 quantize、candidate-score 与
candidate-mask 内核是例外），而[每张 GPU 一个后端](#每张-gpu-一个后端)里那个包裹式
后端是 CUDA 对象，在这里根本没有编译进来，所以那一节的数字在这里一个都不适用。
请把这个后端当作"在没有 CUDA 设备的地方运行该架构"的手段——用来检查改动、用来把
CUDA 结果与主机结果对照，或者在一台本来无法承载该模型的机器上把它跑起来。不要把它
挂到服务端点后面，也不要把它当成吞吐数字来引用：本卡片没有给出任何 CPU 吞吐，因为
从未测过。

除 macOS 之外，服务端的默认后端就是 `ggml_cpu`，所以在有 GPU 的机器上省略
`--backend` 就会选到这条路径。以前这种情况会直接拒绝并提示 `ggml_cuda`；现在它可以
加载，并且在读取任何权重之前先在 stderr 上说明一次
（`[dsv41] --backend ggml_cpu: DeepSeek V4.1 will run on ONE CPU device...`）。
如果你在一台有 GPU 的机器上看到这一行，那你想要的是 `--backend ggml_cuda`。

CPU 路径确实有一致性证据，但它是逐算子的，而不是整检查点的。验证报告的
[fixture 对比](../deepseek41_validation.md#independent-numerical-reference)
记录了在本地 CPU 上、以 `atol=rtol=2e-5` 对独立 oracle 的 41/41 项逐元素检查，最大
绝对误差 5.1633e-6，贪心 token 41/41 一致。这覆盖了融合算子与 fixture 规模的计算图。
完整检查点的 CPU 运行尚未测量，因此真实权重上的 CPU/CUDA parity 在此并未确立。

```bash
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /models/deepseek41-q2/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf \
  --backend ggml_cpu --port 5000
```

准备步骤没有变化：由分词器派生的 `deepseek41.engram.bin` sidecar 在 CPU 后端上与在
CUDA 上一样是必需的，缺少它时会在读取任何权重之前拒绝加载。

`TS_DSV4_THREADS` 设置计算线程数，默认上限为 32。那个上限是为 GPU 运行选的——在那里
这些线程只做辅助性的主机工作；而在纯 CPU 运行里它们就是整个引擎，所以请把它设为本次
运行真正可以使用的核心数。`TS_DSV4_UBATCH`（默认 256）与 `MAX_CONTEXT`（默认 65,536）
的行为与 CUDA 上一致，只不过在这里消耗的是主机内存而不是显存。

那些以 GPU 命名的选项在这里的行为如下：

- `TS_DSV41_TP` 是把路由专家的维度切分到多张 GPU 上。与 `ggml_cpu` 组合时，会在打开
  检查点之前被拒绝，而不是被忽略。
- `TS_DSV4_NGPU` 选择枚举多少张 GPU。这里没有 GPU 可枚举，所以加载器根本不会读它；
  它既不是错误，也不是拿到多于一个 CPU 设备的办法。
- `--tp N` 找不到第二个设备来切分层，因此多 GPU 门控会把它降级为单设备并给出警告。
  那条警告是为 GPU 主机写的，写的是 "Running on ONE GPU"；在这个后端上请读作"一个
  CPU 设备"。
- Engram 表保持主机映射：上文描述的 GPU 驻留放置需要有设备可放。因此
  `TS_DSV41_ENGRAM_DEVICE=1` 会在打开检查点之前被拒绝，而不是被接受后忽略；除 `0`
  与 `1` 以外的取值同样如此；`=0` 指的正是这条路径本来就在用的主机映射，会被接受。
  由于表是主机映射的，[主机映射的 Engram 选项](#主机映射的-engram-表)——
  `TS_DSV41_ENGRAM_WARM`、`TS_DSV41_ENGRAM_THREADS`、`TS_DSV41_ENGRAM_RANDOM`——
  在这个后端上全部有效。
- `TS_DSV4_VRAM_RESERVE_MB` 会在往这个唯一设备上装层之前，从它的空闲内存里扣除。
  这里那个设备就是主机，所以它 2048 MiB 的默认值会保留 2 GiB 系统内存；而模型装不下
  时加载器的拒绝信息是按显存措辞的（"not enough VRAM ... Re-run with
  `--n-cpu-moe N`"）。那条建议本身仍然正确——见下文。
- `TS_DSV41_COMPACT_RAW_GATHER` 是在 CUDA 上测的，默认关闭；它改的是计算图而不是内核，
  因此在这里也能触发，但这条路径上开启它的效果从未测过。保持关闭。
- `TS_DSV41_SPARSE_FA` 在这里没有任何作用。它设置的是 flash attention 的 `n_kv_max`
  边界，而 ggml 的 CPU flash-attention 内核只读取算子的前三个参数
  （`ggml-cpu/ops.cpp`），从不读那一个。设置它是静默无效，而不是变慢或出错；这里没有
  可供开启的掩码压缩 CPU 内核。

以下来自阅读加载器源码而非实测：层权重分配在所选的计算设备上，而这条路径上那就是主机，
因此它们是私有的匿名分配，而不是可回收的 GGUF 映射。这也是在开始一次漫长加载之前值得
做的一个决定——`--cpu-moe`（或 `--n-cpu-moe N`）会把这些层的路由专家移进加载器的主机
上下文，而后者*确实*由 GGUF 映射提供，于是内核可以回收这些页面，而不是把进程 OOM 掉。
当权重加上本上下文的各类 cache 装不下时，加载器建议的也正是这一条，尽管它嘴上说的是
"VRAM"。纯 CPU 完整检查点加载的常驻内存占用与加载时间都尚未测量。

视觉伴随文件会跟随文本模型落到同一个后端上。在这条路径存在之前，它是用硬编码的
`CUDA` 加载的——那会把一张 GPU 拉进一次明确要求纯 CPU 的运行里；现在，没有 ggml 注册
名的后端会按名字被拒绝，而不是被硬着头皮尝试。

`--backend cpu` 是纯 C# 执行器（`DeepSeek4CpuExecutor`），它如今除 V4 之外也实现了
V4.1 的计算图：比例 1 与比例 2 的压缩器、共享的压缩 cache 与索引器 cache、候选剪枝、
Engram 表、延迟 hyper-connection 门控，以及训练好的 cache 量化。
`InferenceWeb.Tests.Dsv41CpuExecutorTests` 以 atol=rtol=2e-5 把它对齐到
`eng/dsv41-reference.py`，覆盖一次性 prefill、分块大小 1/3/5/8 与 reset。那道门禁是
fixture 规模的——一个五层、hidden 256、16 token 的 F32 合成模型——因此它确立的是与参考
实现在架构层面的一致，而不是在已发布的 246 GiB Q2_K 权重上的等价；而且不设置
`TS_DSV41_FIXTURE_DIR` 指向 fixture 目录时，这些测试会静默返回，什么都不检查。与
`--backend ggml_cpu` 不同，它不接受视觉伴随文件：那个编码器是原生 ggml 组件，在这里
`LoadVisionEncoder` 会抛异常，所以图像与视频输入不可用。原生加载器的 Engram 与注意力
开关——`TS_DSV41_ENGRAM_WARM`、`_THREADS`、`_RANDOM`、`_SIDECAR`、
`TS_DSV41_SPARSE_FA`、`TS_DSV41_COMPACT_RAW_GATHER`——在这里全部无效，但准备好的
`deepseek41.engram.bin` sidecar 仍然是必需的；`TS_DSV4_THREADS` 在这个后端上默认取
`ProcessorCount`，而不是 min(核数, 32)；`TS_DSV4_CPU_TRACE_DIR` 写出的逐张量文件与
`eng/dsv41-reference.py --output` 写出的同名，于是两个目录可以逐张量对拍。和
`--backend ggml_cpu` 一样，它是正确性与可移植性通道，而不是服务通道：完整检查点在它
上面的吞吐、加载时间与常驻内存占用都没有测过。

Direct CUDA 引擎 `--backend cuda` 也用自己的内核、不经 ggml 运行 V4.1。它还没有数值
门禁——已经验证了什么、还有什么挡着，见
[CUDA 后端说明](../validation/deepseek41-cuda-backend/README.md)。`--backend mlx`
仍然被拒绝。

## 前向计算图与状态

原生计算图使用四条残差流以及 V4.1 的延迟 hyper-connection 混合。第 1 层与第 14 层会
加入由确定性 token n-gram 哈希选出的 Engram 特征。token 归一化与桶布局来自准备好的
sidecar；各序列槽位保留各自的 token 历史。

每个注意力块都包含一个 128 token 的原始滑动窗口。前两层没有压缩注意力，接下来的 18 层
压缩比为 2，最后 20 层为 1。压缩 KV 源与索引器选择按官方的因果 encoder-decoder 拓扑
共享。query 投影、cache 量化、逆 RoPE 与分组输出 LoRA 都遵循 V4.1 的计算图。索引选择
使用 lightning indexer 与候选块过滤。MoE 使用共享专家加上归一化后的选中路由专家输出。

该实现复用了原生的量化矩阵乘、`mul_mat_id`、注意力、hyper-connection 内核、
per-sequence 槽位与计算图缓存。V4.1 的激活量化与候选过滤有专门的算子。它不调用纯 C#
或直接 CUDA 的 V4 执行器。

有用的源码位置：

- [架构门控](../../TensorSharp.Models/Models/DeepSeek4/DeepSeek41Architecture.cs)
  与[托管驱动](../../TensorSharp.Models/Models/DeepSeek4/DeepSeek4Model.cs)。
- [原生加载器与调度器](../../TensorSharp.GGML.Native/ggml_ops_deepseek4.cpp)
  与 [V4.1 计算图](../../TensorSharp.GGML.Native/ggml_ops_deepseek41.inc)。
- [Engram 哈希与 sidecar 读取](../../TensorSharp.GGML.Native/dsv41_engram.h)
  与[准备脚本](../../eng/dsv41-prepare.py)。
- [对话渲染器](../../TensorSharp.Runtime/ChatTemplate.DeepSeek41.cs)
  与[输出解析器](../../TensorSharp.Runtime/DeepSeek41OutputParser.cs)。

## 对话、工具与 JSON

V4.1 使用显式 BOS 与 `<｜System｜>` 框架，以及带空格的 DSML 标签，例如
`<｜DSML｜ calls>` 与 `<｜DSML｜ invoke name="tool">`。V4 不带空格的 DSML 格式与之
不兼容。渲染器处理 system、user、developer、assistant 与 tool 历史；并行工具结果按其
来源调用 ID 重新排序。字符串型工具参数保留空白字符，未完成的 invoke 不会被派发。
在完整的 DSML invoke 内部，解析器还接受在 Q2_K 上观察到的朴素 `<parameter name="...">`
与 `</parameter>` 变体。无法识别或格式错误的参数标记会被拒绝，而不是被转换成空参数
或残缺参数。OpenAI chat 端点在渲染下一轮时会保留传入的工具调用、推理内容与工具结果 ID。

V4.1 的 OpenAI chat 端点用请求级的 DSML 语法约束已声明的工具调用。`tool_choice: "auto"`
让普通回答不受约束，只有在模型开启 calls 块之后才激活。开启思考时，激活还会等到
`</think>` 之后，这样推理中被引用的工具语法不会启动一次调用。`required` 要求必须调用
一个客户端声明的函数；指定函数名则把该次调用限定为该函数。`none` 阻止产生工具调用，
`parallel_tool_calls: false` 把一个 calls 块限制为一次 invoke。内部的 skill 轮次会拿到
全新的语法状态。其他模型系列保持各自既有的策略行为。

工具语法会强制约束已声明的函数名、必填参数、可省略参数、原始类型、原始类型的
enum/const，以及递归定义的对象/数组。它按 schema 顺序发出参数与对象属性。没有声明
属性的嵌套开放对象接受任意 JSON map。对于声明了属性的对象，即使 schema 允许额外键，
生成时也只发出这些声明过的属性。没有声明属性的函数参数 schema 保留"无参函数"的约定。
带类型的 `additionalProperties` schema 不受支持，会被显式拒绝。声明的整数 `enum`/`const`
取值必须是有符号 64 位范围内的 JSON 整数字面量；其他编码或取值在生成之前就被拒绝。
普通数值参数保持既有的 Int64-或-double 解析行为；下文的无损参数保证针对的是字符串，
而不是任意精度的 JSON 数字。
不支持的断言——包括类型联合、schema 组合子、pattern、数值范围与字符串/数组长度限制
——都会在生成之前返回 HTTP 400。这一受支持子集适用于工具参数；JSON 响应 schema 使用
另一套既有的编译器。

普通字符串可以使用训练过的原始 `string="true"` 表示。包含 DSML 保留分隔符的字符串仍可
通过 `string="false"` 加 JSON 字符串来表达：诸如 `<` 这样的 JSON 转义能在不闭合
外围工具标记的前提下保留精确的解码值。当已解析的调用被渲染进后续工具历史时（包括嵌套
JSON 中的字符串与键），同样的保护依然有效。普通历史的格式化保持不变。原始字符串还保留
了 `<param`、`</param`、`<invoke` 与 `</invoke` 这几个标签族，使得写错的工具标签无法把
后续回复整段吞成参数文本。包含这些前缀的字面字符串使用同一套无损 JSON 替代方案；普通
的 XML（如 `<x>`）与比较符号仍是合法的原始文本。严格解析器保持不变。语法测试确立的是
语法与参数往返；完整检查点上的工具选择与准确率单独测量，并保留未加约束时的失败基线。

推理渲染器使用参考实现默认的 effort 50。普通对话会丢弃过去的推理内容；启用工具的对话
则保留。缓存的原始 assistant token 无法覆盖这条历史策略。开启思考时，JSON 语法约束在
`</think>` 之后开始；否则从第一个输出 token 就开始。提示词与解析器测试确立的是格式
兼容性，而不是模型层面的工具选择、推理质量或 JSON 任务准确率。

由于 V4.1 的协议声明了这种延迟触发的语法，Chat Completions 端点允许在开启思考的同时
使用 `response_format`。这个组合要求启用 JSON 语法约束；`TS_JSON_GRAMMAR=0` 会被拒绝。
若要在一次工具往返之后再请求 JSON 最终答案，请保留工具历史与工具目录，并发送
`tool_choice: "none"`。进行中的工具生成与 `response_format` 仍然互斥。校验只看 assistant
的 content 通道；只有推理内容不算最终答案。
这些工具策略与"思考 + JSON"保证适用于 `/v1/chat/completions`。既有的 `/v1/responses`
接口不支持同样的 V4.1 工具历史往返或"推理加 JSON"组合。

对 V4.1 而言，达到 `TS_THINKING_BUDGET` 会发出训练过的 `</think>` token，并在原有
`max_tokens` 限制之内继续写最终答案。当请求的输出额度不少于 512 token 时，默认预算为
75%。额度更小时没有自动的思考预算；显式设置为正的 `TS_THINKING_BUDGET` 仍然生效。
设为 `0` 关闭该预算。收尾 token 本身也占一个输出 token。这一转换有托管层测试覆盖；
完整检查点上的思考工作流在下文单独测量。
在这条 V4.1 策略生效期间，如果推理进入被检测到的循环，重复守卫也会请求同一个正常的
收尾转换。最终答案中的重复仍会停止生成。关闭请求级或调度器级的重复守卫会同时关闭这个
提前转换；取消、EOS 与原有的输出上限仍具有优先级。

## 当前限制与张量并行工作

- `ggml_cuda` 是 V4.1 的服务后端。`ggml_cpu` 用标量 CPU 实现加载同一套原生计算图，
  作为正确性与可移植性通道，没有实测吞吐；见
  [在 ggml CPU 后端上运行](#在-ggml-cpu-后端上运行)。`cpu` 运行纯 C# 的 V4.1 执行器，
  已按 2e-5 对齐 PyTorch 参考实现；`cuda` 用 Direct CUDA 引擎自己的内核运行 V4.1，
  但尚无数值门禁——两者都是正确性与可移植性通道，而不是服务通道。`mlx` 会在读取权重
  之前失败，而不会把 V4.1 的权重塞进并未实现它的计算图。
- 多 GPU 执行默认按整层放置。`TS_DSV41_TP` 启用实验性的 routed-MoE 张量并行，归约经
  主机中转。注意力张量并行与分布式组尚未实现。
- 并发请求拥有隔离的序列槽位。V4.1 目前回退到逐槽前向调用，而不是 V4 的融合按 token
  批处理计算图，因此并发并不意味着批处理的 GPU 吞吐。
- V4.1 的 DSpark 投机解码尚未实现；V4 的草稿模型会被拒绝。
- 图像/视频输入需要单独准备的视觉伴随文件。编码器与图文计算图有 CPU/CUDA fixture 覆盖，
  并在按层放置与 routed TP 下做过完整检查点的媒体检查。真实图像的 BF16 特征对比超出了
  小规模 fixture 的逐元素容差；见验证报告。目前没有经过验证的音频推理路径。
- 要做同权重对比，需要一个兼容的 llama.cpp V4.1 推理运行时。上面链接的 GGUF 仓库的补丁
  只增加了转换支持。缺少参考实现并不能确立质量或性能的对等。
- 完整检查点的数值 smoke 能产出预期 token，但未通过严格的 F32 输入 oracle 对比（相对
  L2 0.146216，最大绝对误差 2.708920）。量化激活的算术与该参考不同；
  [保留的分阶段分析](../validation/deepseek41/smoke18-reference/README.md)
  未能完全归因最终的差异。贪心结果一致不等于严格的数值一致。

把张量并行扩展到注意力，需要 rank 局部的计算图、权重分片与 cache 状态，并在进入下一个
非线性残差算子之前对注意力输出做一次归约。现有的 GLM 支持
（[ggml_ops_glm_dsa.cpp](../../TensorSharp.GGML.Native/ggml_ops_glm_dsa.cpp)）提供了可复用
的块对齐权重切分，以及带设备集合通信的 rank 局部计算图执行。切分 query head 与分组输出
投影时，V4.1 的八个输出组应保持完整；单个 KV head 与共享的索引器状态可以先做复制。

已实现的专家切分使用不等宽的块对齐分区。2304 的中间维包含九个 256 元素的 K-quant 块：
两个 rank 可取 1280 与 1024，四个 rank 可取 768、512、512、512。gate/up 张量对每个专家
都沿该中间维切分；down 张量沿其输入维切分，随后经主机中转归约。共享专家与 CPU 卸载的
输出只计一次。当前的 ggml CUDA 不再暴露旧的 split-buffer 接口；这条路径显式构建并执行
rank 局部的专家计算图。

短提示、长提示、JSON、工具往返、agent 工作流、并发、放置以及既有模型回归检查的完整
流程，见[验证协议](../deepseek41_validation.md)。未支持的场景保持明确的未验证状态。
