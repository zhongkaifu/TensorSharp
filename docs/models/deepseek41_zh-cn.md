# DeepSeek V4.1 Flash（`deepseek41`）

> **多 GPU 模式选择：** 整层放置使用 `--layer-split N`，支持的张量并行使用 `--tp N`。未配置两种模式时默认单设备。下面的历史命令与测量早于这项默认值变更；多 GPU 启动请加 `--layer-split N`。按层切分仅限单节点。

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

## 下载修复后的 Q2_K/Q5_K 检查点

新下载请使用
[smalinin/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/tree/d1de55c19f95172c882906cc83c0e55932d26a63/Q2_K-Q5)
的 `Q2_K-Q5/`，固定 revision 为 `d1de55c19f95172c882906cc83c0e55932d26a63`。
十个分片放在同一目录，并把第一个分片交给 TensorSharp。文件共 **335,382,014,624 字节
（312.349 GiB）**：主干与专家混合使用 Q2_K/Q3_K，Engram 表为 Q5_K，80 个 mHC
矩阵保留 F32，四个 Engram 门控张量保留 BF16。

原始 `vcruz305` 七分片 Q2_K 把这 84 个敏感张量也量化成了 Q2_K，因此不再推荐用于
质量验证。[发布者的修复报告](https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/blob/2c525d63b9ba5319185c93637f00d70fea55b44f/Q2_K/Q2_REPAIR_REPORT.md)
说明了受影响的张量。给旧分片补充 Engram 元数据不能修复张量精度。

已检查修复版全部 1,046 个张量的名称与形状、敏感张量的存储类型，以及分词器/Engram
元数据的一致性。下载的十个分片均已通过完整文件 SHA-256 校验。`ggml_cuda` 上
`--layer-split 2` 与实验性路由专家 TP（`--tp 2`）的有限普通/DSpark
HTTP 检查均已通过。四个服务各完成三个文本检查与一个图像 OCR/颜色检查，均以 EOS
结束并正常退出。另行进行的普通/DSpark 文本与图像配对检查，在两种模式下均逐一匹配全部
24 个 token ID 与 `max_tokens` 结束原因，DSpark 实际参与解码且进程正常退出。这些有限
续写与 HTTP 的 EOS 检查分开记录。两个图像配对中 DSpark 都更慢。简短的重复文本预热对照
也已完成：每种模式一次预热、三次计时，每对输入 69 token、输出 24 token，token ID 与
结束原因完全一致，DSpark 实际参与解码且进程正常退出。
**这些小样本、受磁盘换页限制的检查不构成通用质量、性能、提速、跨节点执行或完整模型 TP 验证。**
下方历史结果不能作为这套量化的验证结果。本地证据保存在
`docs/validation/model-matrix-20260927/deepseek41/`（不提交）。每次重新下载仍须按下方
哈希校验文件。

**GGUF 已包含 Engram。** TensorSharp 直接读取 GGUF 元数据中的 token 映射、哈希乘数、
桶质数与偏移以及 padding ID，并使用同一检查点中的 Engram 权重张量。文本推理无需生成
Engram、无需单独的 Engram 文件，也无需另行下载分词器或配置文件。

加载器会对照 Engram 张量维度校验内嵌布局，并拒绝缺失或无效的常量；不再回退读取旧的
附属文件，也不再提供 Engram 路径覆盖。Engram 表的设备放置与页预热仍作用于学习到的
权重，不会生成哈希常量。

在存放模型的机器上，从仓库根目录执行：

```bash
python3 -m venv /workspace/dsv41-tools
/workspace/dsv41-tools/bin/python -m pip install huggingface_hub
/workspace/dsv41-tools/bin/hf download smalinin/DeepSeek-V4.1-Flash-GGUF \
  --revision d1de55c19f95172c882906cc83c0e55932d26a63 \
  --include "Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-*.gguf" \
  --local-dir /workspace/models/deepseek41-q2-q5
```

`hf download` 会保留 `Q2_K-Q5/` 子目录。推理前须校验每个完整文件；以下 SHA-256
来自固定 revision 的发布者 LFS 标识。此操作会读取全部 312.349 GiB，请在计时测试之外执行：

```bash
(cd /workspace/models/deepseek41-q2-q5/Q2_K-Q5 && sha256sum --check - <<'SHA256'
8126b49dfcfde02cb3db24b6f98d56b031f0a34eaca2509fc7b2b9d362cf0ac8  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf
22bb293aee509a348ce32a739e006fa41f2348c6bcfafa3be76a3ee079eadf96  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00002-of-00010.gguf
db894848b4f14d42c39e18faa907c737cd4850f2fcb9deff9d5ebd86f580aaf7  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00003-of-00010.gguf
655a3400f2c092d6e3c11b8b18bf319b29563e59b960337115266574321953e3  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00004-of-00010.gguf
4b9378c6819d1130517e8719026b34e5257f1f73bd1d50cf6f30243982100bac  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00005-of-00010.gguf
04f161084d82032c65247c02e6169784a757be7db9e068b0baba6833125b6bb8  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00006-of-00010.gguf
b1b1bf3cfbbc7388ce49b5ced69c42a13c7c5c5d3d609ac86902897a08d3c88a  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00007-of-00010.gguf
fae7f35123ae3557034a541507bb9fc24fb62c2e16ff8441c32e8e0477743d5d  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00008-of-00010.gguf
781fd69e9dd17c09676865830523a2d077a97544d6a004429aab95d2570f8534  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00009-of-00010.gguf
fa5affb1f971cd6e7effad5684b73780472b486a46330cd7c5776f87c38fea97  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00010-of-00010.gguf
SHA256
)
```

任一哈希不匹配时不要继续推理。`eng/dsv41-verify-download.py` 是原七分片布局的历史
校验器，不能校验这套十分片文件。两张 Q5_K Engram 表共约 125.889 GiB；在 RAM 限额
57.74 GiB 的主机上，即使不计 CPU 专家也无法全部驻留。此类主机应使用
`TS_DSV41_ENGRAM_DEVICE=0 TS_DSV41_ENGRAM_WARM=0`，并记录分页开销。CPU 专家
卸载层数须依据实际设备容量选择，同时为可选草稿器与视觉编码器预留空间。

**历史基准的来源：** 本卡片中的早期结果与整文件 SHA-256 记录
`docs/validation/deepseek41/checkpoint-sha256.json`（本地验证记录，未提交到 Git）对应
原 `vcruz305` 发布的 revision `8e0c4de3cb6519bfc11ed69dc87184b457a57bb5` 及其旧版
第一个分片。较晚的七分片示例使用 `58d8ac86298fdf85a2440defee08b1abcad32e45`；
旧 Q4_K_M 的放置记录也须单独看待。保留的历史启动命令指向这些旧文件，而非修复版。
不能把它们的哈希、246.35 GiB 大小、约 60 GiB Engram 预热、质量结果或吞吐率归给
Q2_K-Q5。每次新验证都须记录 revision 与全部分片哈希。

## 准备可选的视觉伴随文件

已发布的 GGUF 转换过程丢掉了视觉塔、aligner、可学习的图像分隔符以及逐层的视觉路由
偏置。官方发布把约 970 MB 的视觉/aligner 权重放在一个独立分片里，因此可以在不下载
原始文本权重的情况下准备它们：

```bash
/workspace/dsv41-tools/bin/python -m pip install numpy==2.0.2 gguf
/workspace/dsv41-tools/bin/python eng/dsv41-prepare-vision.py \
  /workspace/models/deepseek41-q2-q5/Q2_K-Q5 \
  --parent-model /workspace/models/deepseek41-q2-q5/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf \
  --repository deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277
```

它会在文本分片旁边生成 `deepseek41.vision.gguf` 与
`deepseek41.vision.json`。`--parent-model` 提供 GGUF 的分词器指纹；分词器一致时，
已有的视觉伴随文件仍然兼容。
该脚本下载独立的视觉分片，以及分隔符/路由张量所需的少量字节范围，并保留 BF16/F32
存储。它会用 LFS 的 SHA-256 校验完整的独立视觉分片，并记录分隔符/路由权重各自的
范围哈希；它不会下载或校验完整的原始文本分片。
伴随文件溯源记录 `docs/validation/deepseek41/vision-companion.json`（本地验证记录，未提交到 Git）
包含全部 306 个张量、来源 revision、字节范围与输出摘要。
原生加载器在挂载前会检查父模型的分词器指纹与模型维度。要启用图像，在下文的服务命令
中加上 `--mmproj /workspace/models/deepseek41-q2-q5/Q2_K-Q5/deepseek41.vision.gguf`。
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
图像拒绝检查 `docs/validation/deepseek41/full-checkpoint/tp8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-image-input-rejections.json`
与四项 Responses 音频拒绝检查 `docs/validation/deepseek41/full-checkpoint/tp8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-audio-input-rejections.json`（本地验证记录，未提交到 Git）。
无论流式还是非流式请求，对远程图像 URL、格式错误的图像 base64、以及形状合法或格式
错误的音频部分，都返回 JSON 400 且不产生 SSE。更早的 CPU-offload 主机的八项图像检查
另行保留。

## 准备可选的 DSpark 伴随文件

V4.1 需要 `deepseek41-dspark` 产物，不能使用 V4 草稿模型。以下官方 revision 将 DSpark
放在分片 44–46 中（7,933,129,808 字节），无需下载原始文本模型：

```bash
/workspace/dsv41-tools/bin/hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277 \
  --include config.json model.safetensors.index.json \
    model-00044-of-00048.safetensors model-00045-of-00048.safetensors \
    model-00046-of-00048.safetensors \
  --local-dir /workspace/models/deepseek41-source/DeepSeek-V4.1-Flash

(cd /workspace/models/deepseek41-source/DeepSeek-V4.1-Flash && sha256sum --check - <<'SHA256'
8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879  config.json
74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8  model.safetensors.index.json
9a6b39fb88a2510487a8efaef77aa7864e8061f6b62c95a0f010e9dd538f3b05  model-00044-of-00048.safetensors
0cc9d5f6ca3a2158ccc63ce2c70c76aeda8177d54913340481af566680329eb5  model-00045-of-00048.safetensors
e625902027b9d23d416f8818c665fab4704e0b96dc1bc778321601b700475a9d  model-00046-of-00048.safetensors
SHA256
)
```

五项校验全部通过后，再转换伴随文件：

```bash
/workspace/dsv41-tools/bin/python -m pip install numpy==2.0.2
mkdir -p /workspace/models/deepseek41-dspark
/workspace/dsv41-tools/bin/python eng/dsv4-dspark-to-gguf.py \
  --checkpoint /workspace/models/deepseek41-source/DeepSeek-V4.1-Flash \
  --expert-type mxfp4 \
  --out /workspace/models/deepseek41-dspark/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf
```

已审计的转换产物包含 81 个张量（包括三个视觉路由偏置），大小为 7,940,628,416 字节。
来源标识与转换校验已经完成。真实 DSpark 草稿器已在 `ggml_cuda` 双 GPU 按层切分与
实验性路由专家 TP 下通过初步文本/图像 HTTP 检查，图像回答完整结束，服务正常退出。
大量磁盘换页下的通用质量与吞吐仍未获验证。另行进行的 24-token 文本/图像配对检查，
在两种模式下均匹配普通解码的 token ID 与 `max_tokens` 结束原因，DSpark 实际参与解码
且进程正常退出；这些结果仅涵盖有限续写，不声明提速、跨节点执行或完整模型 TP 通过。
在使用修复版十个文本分片的启动命令中，加上
`--draft-model /workspace/models/deepseek41-dspark/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf --spec`。
对比普通解码时仍加载同一伴随文件，改用 `--no-spec`。草稿器驻留在输出头所在 GPU，
会影响设备放置；视觉编码器还需要额外空间。

## 运行已实现的路径

需要安装 .NET 10 SDK、CMake、C++ 编译器，以及 `PATH` 上带 `nvcc` 的 CUDA 工具链。
从仓库根目录构建。此示例使用 A40 的 CUDA 架构 8.6；换其他 GPU 时，两处架构值都要改。
命令指向修复后的下载；仍须在实际硬件上验证容量与推理行为：

```bash
TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON \
  TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES=86 \
  bash TensorSharp.GGML.Native/build-linux.sh
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release \
  -p:CudaArch=compute_86 -p:TensorSharpSkipGgmlNative=true

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_CONTEXT=65536 \
  TS_CPU_MOE_THREADS=32 TS_DSV4_UBATCH=256 \
  TS_DSV41_ENGRAM_WARM=0 \
  TS_DSV41_COMPACT_RAW_GATHER=0 KV_CACHE_DTYPE=f16 \
  TS_SCHED_MAX_RUNNING_SEQS=4 TS_SCHED_MAX_BATCHED_TOKENS=4096 \
  TS_SCHED_PREFILL_CHUNK=256 TS_SCHED_SOLO_PREFILL_CHUNK=8192 \
  dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /workspace/models/deepseek41-q2-q5/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf \
  --backend ggml_cuda --layer-split 8 --port 5000
```

宿主构建会把原生库复制到服务端 DLL 旁边。上面是显式指定微批与调度器设置的启动示例，
并非修复版检查点的实测配置；验证报告中的历史配置使用不同权重，部分参数也不同。
稀疏 prefill attention 不需要任何
开关：它在这条路径上默认开启，`TS_DSV41_SPARSE_FA=0` 可将其关闭。`TS_DSV4_UBATCH=256`
固定该示例的宽度；不设置则由加载器自行选择（见[后端](#后端)）。`TS_CPU_MOE_THREADS` 要按可用的
CPU 配额来选，并为每次运行记录下来。即便是纯 GPU 放置也要在启动环境里设置它：原生的
CPU 图工作与主机侧归约仍会影响延迟。当前 CLI 也接受 `--cpu-moe-threads N`；两者都给
时请填相同的值，因为原生加载器优先采用为正的环境变量值。

**仅用于复现历史记录：** 以下八卡 A40 启动命令保留了原七分片检查点的路径与实测设置，
不是推荐的新下载方案，也不是 Q2_K-Q5 的测量结果：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_CONTEXT=65536 \
  TS_DSV4_UBATCH=1024 KV_CACHE_DTYPE=f16 TS_CPU_MOE_THREADS=32 \
  TS_DSV41_SPARSE_FA=1 TS_DSV41_COMPACT_RAW_GATHER=1 \
  TS_DSV41_ENGRAM_WARM=1 TS_DSV41_ENGRAM_THREADS=16 TS_DSV4_PERF=1 \
  TS_SCHED_MAX_RUNNING_SEQS=4 TS_SCHED_MAX_BATCHED_TOKENS=4096 \
  TS_SCHED_PREFILL_CHUNK=1024 TS_SCHED_SOLO_PREFILL_CHUNK=1024 \
  dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /workspace/models/deepseek41-q2/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf \
  --mmproj /workspace/models/deepseek41-q2/deepseek41.vision.gguf \
  --backend ggml_cuda --layer-split 8 --n-cpu-moe 0 --cpu-moe-threads 32 \
  --host 127.0.0.1 --port 5000 --max-tokens 2048
```

实测启动记录 `docs/validation/deepseek41/full-checkpoint/layer8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-launch.json`（本地验证记录，未提交到 Git）
保留了原始 VM 路径、二进制哈希与环境。该配置通过了 138/138 项推理用例；其吞吐与限制
记录在下文。紧凑 gather 仍是可选项，并带有已记录的浮点差异。该记录中的
`TS_DSV41_SPARSE_FA=1` 选择的是 ggml 的掩码压缩 flash-attention 内核：这些结果是在最初的
V4.1 支持（`3347b06b`）上记录的，当时还没有 TensorSharp 自有的 F32 attention，而后者的
稀疏 prefill 现在是默认行为（见[后端](#后端)）。因此其 19.985/80.240 s 的长提示词耗时并不
是对当前 attention 路径的测量。
启动与页预热不计入推理测量。

在没有显式线程设置时，纯 GPU 的 V4.1 加载使用调用方的 `TS_DSV4_THREADS`，默认上限为
32；CPU 专家卸载则改用探测到的可用 CPU 并行度。更早的原生构建收不到 CLI 的线程覆盖。
具体来说，首次实测 TP 运行中的 `--cpu-moe-threads 48` **并没有**配置 48 个原生线程：
根据加载器源码，在没有原生环境变量覆盖、也没有 VM 覆盖探测的情况下，其原生线程池推断
为默认的 32。那台原始主机没有暴露线程池宽度的读取接口，所以 32 并非在那里直接测得。
请把这个基线与后续显式设置线程数的实验区分开。

`--layer-split 8` 表示**用八张 GPU 做按层切分**。启动诊断会说明采用的放置模式。
TensorSharp 默认按可用显存分配整层。用 `CUDA_VISIBLE_DEVICES` 精确指定本次运行使用的设备。

`--tp 8 --backend ggml_cuda` 会在这八张 GPU 上启用 **routed-MoE 张量并行**；`--tp` 支持 2 到 8 的每个度数（含 3、5、6）。
该度数作为按模型参数传给原生加载器，且必须等于它选中的 GPU 数。`--tp` 与 `--layer-split` 不能组合。

在这种模式下，路由专家的 gate/up 矩阵沿 FFN 中间维切分，down 矩阵按输出行切分，并在所有
选中的 GPU 上并发执行。先按 F32 精度拼接完整 SwiGLU 激活，再计算 down 的各输出行，最后
拼接互不重叠的输出。这保留完整的点积归约次序，避免舍入差异被后续激活量化放大。注意力、共享专家与各类 cache 仍保持按层
放置。这是一份部分实现的张量并行：它不切分注意力，也不支持分布式张量并行组。主机传输
可能成为吞吐瓶颈，因此该选项并未证明比按层切分更快。下面通过校验的完整检查点运行
比按层切分更慢；另见
[实测放置配置](../deepseek41_validation.md#full-checkpoint-routed-moe-tp)。

TP 每层上传之前，按 `TS_DSV4_LOAD_THREADS`（默认 16，每线程 64 MiB 临时缓冲）
顺序读取当前层三个路由专家张量的文件范围，跳过已驻留的页面，避免 rank 跨行读取时触发
大量网络文件缺页。只准备当前层，不预读整个检查点；`TS_DSV4_WARM_PREAD=0` 保留此阶段
原有的直接映射读取。

2026 年 9 月 29 日，在六张 A40 的 VM 上以同一构建、同一 TP6 放置，按
`TS_DSV4_WARM_PREAD=0,1,1,0` 的模型启动顺序测得加载时间（不含内核预热）为
**546.26、149.61、157.35、510.88 秒**。关闭/开启预读的中位数分别为
**528.57/153.48 秒**，描述性比值为 **3.444×**。每次启动前，都要求 120 个路由专家
张量中 183.25 GiB 的完整页面在客户端内核页缓存中的驻留数为零。配置、检查点、源码、
原生库及托管运行时均保持一致；129,280 个首轮预填充 logits 和单 token 贪心检查全部
逐位相同，进程均正常退出。第四次缓存准备起初残留 135 页，因而没有启动模型；另行记录
的续跑在第一次尝试中满足同样的零驻留条件，提供了第四个观测值，原失败记录仍保留。
这是每种设置两次启动、途中中断的对照，并非连续的 ABBA 试验。部分边界页及 MooseFS
用户态、网络和服务端缓存未受控，因此不能把该比值称为冷存储或通用加速比。

当 NCCL 选择 `NCCL_P2P_DISABLE=1` 时，不超过 16 个 token 的批次使用可复用 pinned
主机激活缓冲进行拼接，更大批次使用私有 F32 NCCL 拼接。该阈值来自六张 PCIe A40 的
配对实测，不会用于启用 P2P 的配置。`TS_DSV41_TP_HOST_TOKENS=0` 强制使用可用的设备
拼接，便于对比；0 到 4096 的整数设置主机拼接阈值。`TS_GGML_TP_F32_NCCL=0` 对所有
批次使用主机回退。两条路径均保持 F32 位模式。设备拼接沿 rank 的 CUDA stream 顺序
执行，省去 gate/up 后单独等待；异常返回前仍等待每个 rank 完成。所选放置与传输方式的
整模型吞吐仍需实测。

2026 年 9 月 29 日 UTC 的验证使用六张 PCIe A40、修复版十分片 EngramQ5/Q2_K 检查点、
4096 上下文、F16 KV、关闭预热的主机 Engram 表，以及 `NCCL_P2P_DISABLE=1`。
每种放置启动一次进程，执行五轮相同固定输入：512 个 prefill token、128 个 decode token。
候选原生库 `473ee64d…` 通过运行时文件完整性检查；ggml 上游保持 `353b63b4…` 且未修改。
三次运行的完整 128-token 非计时贪心序列均与原始按层切分基线相同。

| 放置 | 首轮 prefill / decode，tok/s | 第 2–5 轮中位数 prefill / decode，tok/s | 全五轮 prefill 范围，tok/s | 全五轮 decode 范围，tok/s |
|---|---:|---:|---:|---:|
| 原始 `--layer-split 6` | 53.3 / 13.1 | 363.05 / 30.20 | 53.3–502.6 | 13.1–30.3 |
| `--tp 6`，主机自适应阈值 16 | 19.7 / 9.8 | 209.75 / 19.95 | 19.7–280.7 | 9.8–26.2 |
| `--tp 6`，强制设备拼接，阈值 0 | 37.7 / 10.3 | 196.75 / 18.20 | 37.7–302.0 | 10.3–22.8 |

在这台 VM 和此工作负载上，吞吐优先时使用 `--layer-split 6`。自适应传输的 decode
中位数是强制设备传输的 1.096 倍，但轮次波动明显，且每种配置只启动一次，不能据此
声称稳定或普遍加速。两次 TP 运行的采样 SM 时钟均保持 1740 MHz、P0 状态；后续轮次
未记录到 major fault。其余计时波动尚未归因。

计时期间，TP 的容器内存约为 255 GiB，按层切分约为 70 GiB。其中文件缓存约为
252 / 68 GiB，匿名内存约为 1.6–1.7 / 0.8–0.9 GiB。TP 上传临时缓冲有界，但已上传的
路由权重页面仍保留在可回收的文件缓存中：`TS_DSV4_LOAD_DROP_CACHE` 目前仅作用于普通按层
上传，不作用于私有 TP 上传路径。因此，显存足够并不代表主机内存限制也足够。
观测到的 255 GiB 也不代表最低内存要求。这几次顺序加载的源页面驻留状态不同，不能作为
受控的加载延迟对比。

当前按输出行切分的实现通过了两张和六张 A40 的数值 fixture，包括双卡 F32、BF16、F16、
Q2_K、Q3_K、Q4_K、Q6_K，以及六卡真实修复版检查点的专家权重。CPU fixture 覆盖 2 到 8 个
rank；验证 VM 只有六张 GPU，未执行七卡与八卡 CUDA 测试。最终 `473ee64d…` 运行时
使用修复版十分片 Q2_K/Q5 检查点通过六卡 HTTP 验证：16 个文本响应与未修改的按层
切分基线完全相同，四个严格工具场景、三个图像场景全部通过，129,280 个首轮 prefill
logits 逐位相同。DSpark 与 ngram 在文本、图像场景都保持全部 96 个贪心 token 及结束
原因一致。草稿/接受/校验步数分别为：两个 ngram 场景均 49/49/10，文本 DSpark
83/73/19，图像 DSpark 40/22/13；DSpark 分别实际执行七次与九次回滚。HTTP 与投机
测试均正常退出，运行时文件未改变。这些计数证明实际执行了投机路径，不代表投机
吞吐加速。这些检查不能说明完整检查点能装进两张或四张 A40。
当历史七分片 Q2_K 的 Engram 表使用主机映射时，同步预热会在就绪之前占用约 60 GiB 主机页缓存；
驻留 GPU 的表跳过这一步。冷加载与预热时间要与热态吞吐分开记录。

在 CUDA 上，量化 gate/up 与 down 输出行分片走 TensorSharp 自有的量化分片 kernel
（`ggml_ops_matmul_quant_strip.cuh`、`tsg_matmul_id_quant_pair`）：它只读取本 rank 的权重分片，
但保留未切分发射的 stream-k 划分与归约顺序，因此每个分片的 gate/up 行与完整张量逐位相同。
在此之前，ggml 的批量 MMQ 路径对分片的 F32 求和分组与未切分发射不同，down 投影的 Q8 激活
重新量化又把它放大成检查点形状 Q2_K/Q3_K 在 16 token 时的失败（相对 L2 `3.9e-5`，完整权重
容差为 `1e-5`）。`GgmlOpsDsv41TpTest` 仍以严格的完整权重参考及其原始容差为通过标准，同时
记录同设备按分区求值的结果，`--cuda 1 --quant-strip-only` 检查 gate/up 逐位相等以及 scratch
增长/失败恢复。非对齐的分片形状仍走 ggml 路径。已记录的微基准中窄分片每次 MoE 调用慢
15-36%；这描述的是此前切分 down 归约维的实现，并非当前的两次拼接实现；见
`docs/validation/qualification-2026-09-16/numerical-tp-chosen-r1/README.md`（本地验证记录，未提交到 Git）。

在 `ggml_cuda` 上，未指定卸载策略时，原生容量规划器根据各卡可用显存、上下文和工作区，
默认选择能放下的最少主机专家层数；全部能放下时不卸载。显式 `--n-cpu-moe 0` 要求全 GPU
驻留，正数指定前 N 层，`--cpu-moe` 卸载全部路由专家。其他后端仍需显式开启。
注意力、路由与共享专家仍在 GPU 上。这是加载期的单模型规划，不是统一请求预算；
主机映射仍可能超过 RAM 限制并产生换页，加载日志会说明。
[Engram 表的放置](#engram-表放在哪里)单独选择；使用主机映射时，每个输入批次只读取并
传输选中的 embedding 行。CPU MoE 卸载与按层切分都已实现，但它们在你所用硬件与上下文下的
吞吐需要实测。与 `--tp N` 组合时，
被 CPU 卸载的前置层保留完整的 CPU 专家，其余层使用路由专家分片。

原生版本 `6b3b5ab3…` 显式把共享专家的 gate/up/down 投影指派到该层所在设备。这修正了
更早的调度器放置问题：在经过 CPU 卸载或 TP routed 分支之后，共享的 gate/up 计算可能被
送到 CPU。相关的放置与数值检查
（`docs/validation/deepseek41/shared-expert-placement/README.md`，本地验证记录，未提交到 Git）
在两张 GPU 上通过了 597/597。更早的完整检查点放置基准仍保留其原始二进制与结果。
使用修正后放置的最终 CPU-offload、routed-TP 与按层切分配置均已完成，见放置记录
`docs/validation/deepseek41/final-placements/README.md`（本地验证记录，未提交到 Git）。

要做热态的 CPU 卸载基准，请在模型加载之后、开始计时请求之前，单独把被卸载的专家页读入：

```bash
/workspace/dsv41-tools/bin/python eng/dsv41-warm-experts.py \
  /workspace/models/deepseek41-q2-q5/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf \
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
84 字节，因此两张 3.84 亿行的表合计 60.1 GiB。vLLM 与 SGLang 把同样的行存成 FP8 值加
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

预热用 `pread` 以 64 MiB 为块读取每张表，每个读取线程（`TS_DSV4_LOAD_THREADS`，默认 16）
负责表中一段连续区间，并跳过 `mincore` 报告页面已全部驻留的块，因此表仍在缓存中时重新
加载几乎不读任何数据。过去的做法是每 4 KiB 页触碰映射的一个字节，而在网络文件系统上每一次
这样的缺页都是一次受挂载点预读上限约束的同步读（A40 虚拟机的 MooseFS 挂载为 128 KiB）：
七卡 A40 通道同步预热 103 GiB 的 Q4_K_M 表用了 311.3 秒。在同一台虚拟机上实测，对一段已
驱逐的 8 GiB 表区间，`pread` 为 2.38–2.54 GiB/s，逐页触碰为 0.63–0.69 GiB/s；测量方法见
[加载时间](#加载时间)。`TS_DSV4_WARM_PREAD=0` 让下面两种形式都恢复逐页触碰。同步模式下的
读取错误会让加载失败并给出文件与偏移；后台模式下只打印一行日志，模型照常服务。

| `TS_DSV41_ENGRAM_WARM` | 行为 |
|---|---|
| 未设置（默认） | 模型开始服务后在后台预热；完成时打印 `[dsv41] warmed ... Engram pages in ...s (background, ...)` |
| `1` | 与此前一致，在加载期间同步预热；启动时间增加的就是读表的时间（七卡 A40 通道上逐页触碰 103 GiB 用了 311.3 秒；按实测的 2.24–2.54 GiB/s `pread` 速率计算约为 41–46 秒的读取） |
| `0` | 从不预热 |

稀疏读取的映射建议（`MADV_RANDOM`）只在预热结束之后才施加，无论采用哪种预热形式：该建议
会关闭预热本身所依赖的预读。`pread` 预热只把页面放进页缓存，并不把它们映射进进程；此后
查表第一次触碰某一行是一次次缺页（minor fault），而不是一次存储读取。在这样预热过的表上，
经 `MADV_RANDOM` 映射随机读取 2,000 个 144 字节的行，平均 0.0037–0.0056 毫秒，没有主缺页。

`TS_DSV41_ENGRAM_THREADS=1..32` 控制常驻查表工作线程数，默认取 16 与硬件线程数中的
较小者。prefill 与 decode 都使用并行取行：单个 token 在每张 Engram 表上要选 24 行互不
相关的数据。设为 1 会让读取串行化。并行取行避免了 decode 期间串行的缺页等待，且不改变
embedding 数值。执行器在同一个工作池任务中交错读取两张表，工作线程完成后再逐表上传行。
staging 以 64 MiB 为单位分组；单张更大的表沿用此前的一表分配上限。

在 Linux 上，并行取行会自动为映射的 Engram 区间请求随机访问建议，以减少不必要的预读。
该提示在可选的整表预热之后应用，除共享的边界页之外，不改变其他张量区间的策略。设置
`TS_DSV41_ENGRAM_RANDOM=0` 可关闭它，`=1` 可强制开启。不设置时，单线程配置保持默认的
映射策略。不支持的平台与被 OS 拒绝的提示都是非致命的。源码对比与配对的 scratch 文件
结果见 Engram 调查 `docs/validation/deepseek41/cli-gpu-execution/README.md`（本地验证记录，未提交到 Git）。

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

在 `ggml_cuda` 上，V4.1 的 attention 运行 TensorSharp 自有的 F32 内核（ggml 的 CUDA flash
attention 会把 Q 与 softmax 权重收窄到 F16，而 cache 量化可能放大丢失的这些位）。**prefill
默认是稀疏的**，前提是调用既宽又长：超过 8 个 query、至少 8,192 个 key（原始窗口环加上可见的
压缩行）的分块，每个 query 只经由掩码压缩内核关注自己的滑动窗口与索引器选中的行，最多
128 + 512 个 key。其余情况保持稠密内核：单 token decode 与每次 DSpark verify（6 行）使用
分 key 内核，因此 verify 提交的 cache 行仍与 decode 完全一致；更短的 prefill 使用分块
（tiled）内核，因此 attention 始终低于 8,192 个 key 的提示词逐位不变。可见 key 多于上限的行
会退回全行扫描，所以这个上限不会丢掉任何 key。

在单张 A40 上实测（`GgmlOpsCudaAttentionPrecisionTest --benchmark-dsv41-prefill 512 33536 64 5`，
它通过生产环境的门控与该变量选择内核）：64 个头、512 个 query、33,536 个 key 时，稀疏每次
**34.1–34.3 ms**，分块为 **1,547–1,549 ms**。与分解的 F32 参考相比，稀疏内核的最大绝对误差为
1.1e-7（相对 L2 7.4e-7），分块内核为 8.9e-8（4.8e-7）。稀疏 query 也与同一调用中的其他 query
无关——query 0 单独计算、在 9 个 query 中、在 512 个 query 中逐位相同——因此提示词的结果不取决于
prefill 如何分块。`TS_DSV41_SPARSE_FA=0` 恢复分块 prefill。

这个自有门控（超过 8 个 query、至少 8,192 个 key、F32 压缩内核）与 ggml flash-attention
内核的门控不同，后者由非自有的 attention 路径使用（非 CUDA GPU、CPU 后端）。在那里
`TS_DSV41_SPARSE_FA=1` 仍然只在单个 query 或至少 16,384 个 key 时启用 ggml 的掩码压缩 flash
attention，其 F16 运算与 CPU oracle 的相对 L2 实测最高 7.8e-4；该提示保持显式开启。稀疏
attention 减少的是 attention 计算量，并不消除 prefill 期间共享压缩 cache 在 GPU 之间的拷贝。

默认上下文分配上限为 65,536 个 token，除非提供 `MAX_CONTEXT`。`TS_DSV4_UBATCH` 控制前向
微批。不设置时，V4.1 在 ggml GPU 后端上由加载器选择：1024、512 或 256 中所需路由专家 CPU
层数不多于 256 的最宽者，记录为 `[dsv4] prefill ubatch: N (auto; ...)`（见
[为图保留的设备内存](#为图保留的设备内存)）。CPU 执行器与 direct CUDA 引擎保持 256。任何
显式值都原样使用；`TS_DSV4_UBATCH=256` 恢复此前固定的默认值。模型声明的窗口更大，并不意味着
某个具体的 GPU 配置能分配或高效服务那么长的上下文。

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
推理。Engram 仅在表使用主机映射时才从主机查表。按层切分会让相邻的层落在相邻的 GPU 上，因此仅凭
单卡利用率低并不能断定发生了 CPU 回退。`TS_DSV4_PERF=2` 报告输入准备与图计算耗时；
`TS_DSV4_PERF=3` 还会记录调度器实际的后端切换。它们是诊断模式，其日志开销会影响吞吐。
见 CLI 执行调查 `docs/validation/deepseek41/cli-gpu-execution/README.md`（本地验证记录，未提交到 Git）。

`TS_DSV41_COMPACT_RAW_GATHER=1` 为稀疏的单 token decode 启用原始窗口压缩。它先在持有
这些行的 GPU 上把 128 行可见的原始行 gather 起来，再搬到持有共享压缩缓存的 GPU。被
掩码的重复行把原始前缀补齐到 256 行，使合并后的 768 行 K 张量满足 CUDA 512 分量注意力
的对齐要求。物理 ring、prefill 路径以及不做 gather 的 decode 均保持不变。该选项默认
关闭。在约 8k 提示词下的限定 Q2_K 对比中，并发 1 与 4 的持续 decode 都提升了 13.7%；
严格的 flash 算术差异与完整测量设置见验证报告。

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

这个保留量与原始滑动窗口环都随 prefill 微批增长，所以在 `TS_DSV4_UBATCH` 未设置时，加载器
会为每个候选宽度分别计算一次切分。上下文 65,536 时，256 / 512 / 1024 的默认保留量为每张卡
2,318 / 2,588 / 3,128 MiB，窗口环为 512 / 768 / 1,280 行。它选择所需路由专家 CPU 层数不多于
256（或本次运行本就要付出的显式 `--n-cpu-moe`）、且不会把驻留 GPU 的 Engram 表挪回主机的
最宽候选。更宽的分块让每个 prefill token 更便宜：一层 Q4_K_M 形状的路由专家层
（`GgmlOpsDsv4MoeWidthBench`，均匀 top-6 路由，单张 A40）驻留 GPU 时在 256 / 512 / 1024 个
token 下每个分块耗时 35.3–35.7 / 36.9–37.2 / 38.9–39.1 ms，1024 时每 token 便宜 3.6 倍；在 32 个
主机线程上为 112.7–117.9 / 185.0–189.1 / 349.9–353.3 ms（每 token 0.44–0.46 对 0.34 ms）。但多一层
主机专家的代价要由每个 decode token 承担，所以从不做这种交换。日志行会给出宽度，并在放弃更宽宽度时说明原因。低于 256
所需层数的 `--n-cpu-moe` 会被拒绝并给出该数字（`Re-run with --n-cpu-moe N`）。2048 不在候选
之列：保留量只在 1024 上验证过（一次 57,424 token 的 prefill 在 3,072 MiB 保留量下峰值时仍剩
1,522 MiB），而 ggml 的设备 OOM 会直接结束进程。所选宽度通过 `TSGgml_Dsv4UBatch` 导出，推测
解码的 prefill 按同一宽度分块。

用 Q4_K_M 发行版的张量大小，以及它在七卡 A40 上的运行所反推的每卡预算（加载后空闲内存加上
加载放置的字节，每卡 45,091–45,123 MiB）来计算：上下文 65,536 时三种宽度都需要 6 层路由
专家 CPU 卸载，因此加载器选择 1024；上下文 131,072 时同样的预算下每种宽度也都是 6 层。这些
是计算出的方案（`GgmlOpsDsv4UbatchPlanTest`），不是实测加载。

图缓存现在同时按字节数和条目数设限。一个条目的计算缓冲区随其形状增长，而处于不同位置的
并发序列会产生很多不同形状：过去四个并发的 10.8k token prefill 会占满全部十二个条目并把
某张卡的显存用尽，而这不是可恢复的错误——ggml 的分配器在重新分配前先释放旧缓冲区，因此
一次失败的 reserve 会留下空指针，进程直接死掉，而不是让该请求失败。现在在构建新条目之前
会先释放最久未使用的条目，直到每张卡都能再放下一个与已缓存的最大条目同样大的图，外加一个
底线。`TS_DSV4_GRAPH_CACHE_HEADROOM_MB` 设定这个底线（默认 1024）；设为 `0` 回到纯条目数
上限。

### 多轮 KV 复用

第二轮渲染出的提示词并不是第一轮缓存的延续。普通对话会丢弃上一轮 assistant 的推理
内容，因此渲染结果恰好在上一轮 `<｜Assistant｜>` 之后一个 token 处与缓存分叉：缓存里
是 `<think>`，渲染出的是 `</think>`。分叉点之前的内容仍然一致，而从第三轮起，那就是
上一轮的**整个**提示词。

因此原生执行器支持部分复用：`TSGgml_Dsv4Truncate` 把槽位的 head 回退到匹配前缀的末尾，
只前向新的后缀，于是每轮的 prefill 取决于最新的回答，而不是整段对话。两个条件约束它
（见 `dsv41_truncate.h`）：

* **对齐。** 目标位置必须是最大压缩比（已发布检查点为 2）的整数倍，使任何压缩块都不
  跨越新的 head；调用方把复用长度向下对齐，最多损失一个 token。
* **深度。** 原始滑动窗口存放在 `pad64(n_swa + n_ubatch, 256)` 个位置的环里（已发布
  检查点为 512，窗口 128），所以从当前状态最多只能回退 385 个位置。生成回答会让 head
  远远越过下一轮需要的提示词边界，因此每个槽位都保留一份**回退检查点**：在每次多
  token 前向结束（即提示词边界）时，对两个按模寻址的环（原始窗口与压缩器状态）做影子
  拷贝，decode 步刻意不移动它。它每层每个槽位占 `n_embd_head x ring_raw x 2` 字节
  （已发布检查点约 21 MiB）；`TS_DSV41_REWIND_CHECKPOINT=0` 将其关闭，此后超出活动环的
  回退会被拒绝。

在八张 A40、Q4_K_M 上实测（`--n-cpu-moe 2`，贪心，提示词后接两轮 `continue`，
`TS_KV_DEBUG=1`），时间是 2026-09-11，当时 CLI 仍自行规划复用（`KVCache.PlanReuse`，
`TS_KV_DEBUG` 打印的正是它）。自 2026-09-17 起 CLI 与服务器都走引擎的 Radix 前缀缓存
（见下文），`TS_KV_DEBUG` 在那里不再打印任何内容。分叉点正好落在策略预期的位置——两轮中
缓存在那里都是 token 128821（`<think>`），而渲染结果是 128822（`</think>`），恰在
`<｜Assistant｜>` 之后一个 token：

| 轮次 | 提示词 token | 匹配前缀 | 计划 | prefill |
|---:|---:|---:|---|---:|
| 1 | 38 | 0（冷启动） | Reset | 970 ms |
| 2 | 2,056 | 37，对齐到 36 | PartialReuse | 8,399 ms |
| 3 | 5,841 | 2,055，对齐到 2,054 | PartialReuse | 17,058 ms |

第 2 轮只能收回一个常量——system 块、问题与 assistant 头——因为该点之后的缓存是带着
推理内容的第一轮回答，而提示词里已经没有它了。第 3 轮收回第 2 轮的整个提示词，此后的
占比会继续增长：匹配前缀随对话增长，而需要重新前向的后缀始终只有一个回答那么长。两次
回退都远超活动环（第 3 轮为 6,529 个位置），由检查点提供。

回退被拒绝是正常结果而不是错误：调用方会 reset 并重新 prefill。普通 `deepseek4` 完全
不做截断（它的压缩器块相互重叠，对齐 head 不足以保证正确），V4.1 的 direct CUDA 与纯 C#
执行器也不做（它们没有检查点）。关闭 `--think` 时这一切都不需要：没有推理内容被丢弃，
渲染结果就是缓存的纯追加，复用无需回退。

跨请求时，这种复用由 Radix 前缀缓存驱动——它是默认模式，CLI 与服务器走的都是这条路径。
结束的一轮作为模型的主缓存（primary）常驻。下一轮思考模式的对话把它回退到整段上一轮回答
之前来保留它，树在准入时询问模型槽位能否回退到那么远（`CanRewindPrimary`，读取
`TSGgml_Dsv4SlotCanReuse`：活动环或提示词边界检查点）。被拒绝就完整 prefill，准入日志行会
说明原因，例如 `Radix prompt reuse for …: 0/2056 tokens; 2056 token(s) to prefill
(rewinding the cached conversation is declined by the model).` 放置方式不影响规划：
`--tp`、`--layer-split` 与单卡的规划相同。复用之后至少前向两个提示词 token 的一轮会留下
新的检查点（能力字段 `MinTailPrefillTokens`），因此重新生成的一轮不会让它之后那一轮失去复用。

以上成立的前提是：这一轮单独运行，且紧随其后的是同一会话的下一轮——CLI 正是这样运行的。
与其他请求重叠过的一轮运行在按请求分配的槽位上，结束时即被释放；而引擎为任何其他请求执行的
第一步都会丢弃常驻的主缓存。因此在会话相互重叠的服务器上，除非上一轮结束时没有其他请求在运行、
且在它之前没有其他请求被接纳，思考模式的一轮会重新 prefill 它的提示词。

2026-09-29 之前，树的捐赠（donation）规则拒绝任何超过 16 个 token 的回退，这一次也不例外，
因此第一轮之后的每一轮思考对话都没有任何复用（CLI 中显示 `kvPlan=Prefill`；最早在
`--tp 6` 下被报告）。该规则的用意是把较深的缓存状态留给之后的请求，而不是交给一个只共享
其开头的请求。但拒绝并不能保住主缓存——引擎执行的下一步无论如何都会丢弃它——所以该规则
不再约束主缓存。`DeepSeek41ThinkingTurnReuseTests` 用真实的 V4.1 对话模板让这样一段对话
走完引擎。匹配前缀按构造与上表相同，但上表尚未在真实模型上通过引擎重新测量。

已结束请求的原生槽位也会被保留，使多个会话（包括相互重叠的会话）
都能继续而无需完整重新 prefill。一个被保留的槽位至多服务之后的一个请求；树允许它**自己的**
会话在槽位能做到的范围内越过 16 个 token 的规则回退它（`CanMaterialize`，同样的检查点判断）：
思考模式的一轮总要回退越过上一轮回答，按那条规则被保留的槽位将无法服务任何思考轮次。其他会话
永远到达不了带作用域的槽位，因此无法取走它。若原生侧拒绝保留某个槽位（预算或设备余量不足），
结束的那一轮不会被复用，也不会被登记——2026-09-29 之前，失败的尝试会留下一个已被清空却仍被
登记的主缓存，精确延续的请求随后会从它解码。能够回退的
原生执行器始终开启保留，只在加载了 DSpark 草稿器时不生效；保留槽位的预算由
`TS_DSV41_RETAINED_CACHE_MB`（默认 2048）决定，不是正数的值保持默认。2026-09-29 在 6x A40（`--tp 6`，
Q2_K）上用四个重叠的三轮 Web UI 会话（`eng/validation/parallel-multiturn-webui.py`）实测：没有保留时
每一轮复用 0 个 token，因为每个结束的槽位在该会话下一轮到来之前就已释放；有保留时每个会话每一轮都
复用了自己上一轮的提示——开启思考时，第 2 轮复用 167-210 个 token 中的 54-62 个（被丢弃思考内容
之前的部分会重新渲染），第 3 轮复用第 2 轮的整段提示；关闭思考时分别为 175-200 中的 171-196 与
307-437 中的 287-417。两种模式下八个会话全部通过答案检查。

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

### 加载时间

权重由一组读取线程（`TS_DSV4_LOAD_THREADS`，默认 16）按 `TS_DSV4_LOAD_CHUNK_MB` 大小的
分块（默认 64）流式上传到 GPU。每个线程只走任务列表中的**一段连续区间**。在网络文件系统上，
这一点比加载器的其他任何细节都重要：任务按（分片，文件偏移）排序，如果像这个加载器过去那样
从共享游标分发，每个文件描述符读完一个分块就要跳过 `线程数 x 分块`，默认参数下是 1 GiB。
预读是按描述符进行的，因此十六条读流没有一条是顺序的。

在八卡 A40、MooseFS 挂载上的 Q4_K_M 发行版上冷态实测（每次运行前把全部 414 GiB 分片逐出
页缓存，两种顺序各交替两次）：

| 任务顺序 | 权重上传，294.8 GiB | 模型总加载时间 |
|---|---:|---:|
| 每线程一段连续区间 | **141–147 s**（2.0–2.1 GiB/s） | **144–155 s** |
| 共享游标（`TS_DSV4_LOAD_CONTIGUOUS=0`） | 360–377 s（0.78–0.82 GiB/s） | 363–382 s |

**2.5 倍。**`TensorSharp.Runtime/GgufReader.cs:330` 为托管侧 GGUF 预取记录了同样的结论
（"~3x slower on MooseFS"）。区间按**字节数**而不是任务数切分，因为一个张量的最后一个分块
不是满块；读完自己区间的线程从进度最落后区间的**尾部**窃取任务，让被窃取者继续向前读。

有三件事看起来像是解法，但在这台机器上实测都不是：

* **页锁定 staging 缓冲区。** 主机到设备的拷贝只占 87 s 线程时间，而 `fread` 占 5,539 s——
  只是加载工作量的 1.6%。pinning 能让拷贝本身快几倍，但对整个加载几乎没有影响。
* **更多读取线程。** 在这个文件系统上吞吐量并不随线程数单调增长；
  `TensorSharp.Backends.Cuda/Dsv4/Dsv4CudaEngine.cs:744` 记录了 16 线程 2.4 GB/s，
  96 线程只有 1.0 GB/s。
* **对主机专家预取使用 `MADV_WILLNEED`。** 29.7 s 与 31.3 s，而普通的逐页缺页遍历平均
  27.7 s，也就是说并没有更好。

上传之后的两趟读取同样用 `pread`。驻留主机的专家（`--n-cpu-moe`）预取与 Engram 预热（见
[主机映射的 Engram 表](#主机映射的-engram-表)）把各自的张量合并成文件区间，按字节切成每个
线程一段连续区间（`TS_DSV4_LOAD_THREADS`），每个线程用自己的描述符按 64 MiB 块读取，并跳过
`mincore` 报告页面已全部驻留的块。它们过去是每 4 KiB 页触碰映射的一个字节。在网络文件系统上，
每一次这样的缺页都是一次受挂载点预读（`read_ahead_kb`，A40 虚拟机上为 128 KiB）约束的同步读，
所以七卡 A40 通道（`--n-cpu-moe 6`，Q4_K_M）预取 48.2 GiB 专家花了 129.7 s，预热 103 GiB
Engram 表花了 311.3 s。

在那台虚拟机上用 `GgmlOpsDsv4FileWarmBench`（在 Linux 上随原生测试一起构建）实测：16 线程，
8 GiB 区间，每组测量前逐出并确认 `mincore` = 0，重复三到五次，各组交替进行：

| 读取 8 GiB | Engram 表，分片 00002 @ 20 GiB | 专家，分片 00003 @ 9,002,135,936 |
|---|---:|---:|
| `pread`（默认） | 2.38–2.54 GiB/s | 2.24–2.47 GiB/s |
| 预取的逐页遍历，256 MiB 段（`TS_DSV4_WARM_PREAD=0`） | 0.62–0.66 GiB/s | 0.68–0.74 GiB/s |
| Engram 的逐页遍历，8 MiB 块（`TS_DSV4_WARM_PREAD=0`） | 0.63–0.69 GiB/s | 0.65–0.68 GiB/s |
| 同一区间再次预热，已驻留（所有块均跳过） | 64–159 GiB/s | 139–187 GiB/s |

`pread` 只填充页缓存，并不填充进程的页表，而逐页遍历两者都做了；第一次 prefill 又会密集地
读取专家。因此预取在每个块进入缓存后还会逐页读一个字节：在已驻留的页面上，这趟遍历在 16
线程下耗时 0.004–0.006 s/GiB，而对同样区间调用 `madvise(MADV_POPULATE_READ)` 需要
0.019–0.023 s/GiB。主机映射的 Engram 表不预先填充进程页表，因为每次只读取其中几行。

在这个挂载点上，预读提示无法替代读取。对一段已逐出的 8 GiB 区间调用 `MADV_WILLNEED`、
`POSIX_FADV_WILLNEED` 或 `readahead(2)`，十秒后都只有 128 KiB（0.0015%）驻留。不要再尝试。

预取或同步 Engram 预热中的读取错误会让加载失败，并给出分片与偏移。`TS_DSV4_WARM_PREAD=0`
原样恢复两种逐页遍历。

除非设置 `TS_HOST_MOE_PIN=1`，被卸载的专家不会被页锁定。被卸载层的路由专家的每个节点都被
指定到 CPU 后端（`build_moe_host`），而 `ggml_backend_sched` 从不覆盖这种指定，因此它的
op-offload 规则永远不会把这些权重流式送到 GPU：跨总线的只有 `[n_embd, n_tokens]` 的激活，
锁定的专家页从来不是 DMA 的来源。在七卡 A40 通道（`--n-cpu-moe 6`）上，锁定 48.2 GiB 让加载
多花 20.4 s，并让这些页面在同样承载页缓存的 cgroup 中无法被回收。现在加载时会打印一行，
说明被卸载的专家保持可分页。`TS_HOST_MOE_PIN=1` 恢复页锁定以及
`page-locked ... GiB of host experts` 日志；`TS_HOST_MOE_PIN=0` 仍对所有架构关闭锁页。
其他 MoE 架构的 prefill 确实会流式传输被卸载的专家，默认仍然锁页。

加载器不会再读已上传的分块，但上传之后紧接着就要读主机映射的权重，并在整个运行期间从页缓存
提供它们，而页缓存计入 cgroup。因此默认情况下，只有当上传字节数加上主机映射的权重再加 8 GiB
超过主机额度（cgroup 上限）时，每个分块上传到设备后才释放它的页缓存；否则保留，额度未知时也
保留。加载时会打印这一判断及其数值：

```text
[dsv4] load page cache: dropping each uploaded chunk's page cache (automatic: 263.0 GiB upload + 151.2 GiB host-mapped + 8.0 GiB headroom exceeds the 326.9 GiB allowance; TS_DSV4_LOAD_DROP_CACHE=0 overrides)
```

这正是七卡 A40 通道的数值：414 GiB 的读取装进 326.9 GiB 的 cgroup。它的加载日志中专家预取为
0.37 GiB/s、Engram 预热为 0.33 GiB/s，大约只有同样的逐页遍历在该虚拟机上、cgroup 约一半占用
时实测速率（见上表）的一半。释放无法让上传本身变快——上传已经以存储速率读取——而且在该挂载上
每个已驻留的 64 MiB 分块要花 5.9–7.3 ms（`GgmlOpsDsv4FileWarmBench --drop-cost`，263 GiB
约合 25–30 s 线程时间）。预期收益在上传之后的阶段，尚未在完整加载上测量；检验方法是用默认值的
冷加载对比 `TS_DSV4_LOAD_DROP_CACHE=0` 的冷加载。规则之所以保持有条件，是因为每次都释放会让
完全驻留在 GPU 上的检查点每次重新加载都变冷。`TS_DSV4_LOAD_DROP_CACHE=0` 从不释放（此前的
默认），`=1` 总是释放。在八卡 A40 机器上，`=1` 没有改变上传的读取线程时间（5,374 s 对比
5,539 s，处于运行间波动之内），加载结束时页缓存约 39 GiB 而不是约 330 GiB。

自己计时时有一点要注意：如果机器的页缓存已经塞满了这份检查点，加载可能比从空缓存开始**更慢**，
因为 cgroup 在第一次读取之前就已到达上限，之后的每次读取都要与回收竞争。请同条件比较——先把
分片逐出。

### 在 ggml CPU 后端上运行

`--backend ggml_cpu` 选择加载器的纯 CPU 分支：只有一个 CPU 计算设备而不是枚举出的
加速器，所有层都在它上面，所有 V4.1 专属算子都走
`ggml_ops_dsv4_fused_cpu.cpp` 里的标量 CPU 实现而不是 CUDA 内核。启动会打印
`compute devices initialized: 1 CPU device(s)` 与
`routed-expert placement: all 40 layer(s) on the explicitly selected CPU device`。

**这是一条正确性与可移植性通道，不是服务通道。**每解码一个 token，都要在 40 层里
各读出所选检查点 384 个路由专家中的 6 个，而且跑在通用核心上。
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
  --model /models/deepseek41-q2-q5/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf \
  --backend ggml_cpu --port 5000
```

CPU 后端与 CUDA 一样，直接读取内嵌的 Engram 元数据。下载当前 GGUF 分片即可，无需
另行准备 Engram。

`TS_DSV4_THREADS` 设置计算线程数，默认上限为 32。那个上限是为 GPU 运行选的——在那里
这些线程只做辅助性的主机工作；而在纯 CPU 运行里它们就是整个引擎，所以请把它设为本次
运行真正可以使用的核心数。`TS_DSV4_UBATCH`（这里未设置时固定为 256，而 `ggml_cuda`
会自动选择宽度）与 `MAX_CONTEXT`（默认 65,536）在这里消耗的都是主机内存而不是显存。

那些以 GPU 命名的选项在这里的行为如下：

- `--tp N` 是把路由专家的维度切分到多张 GPU 上。与 `ggml_cpu` 组合时，会在打开
  检查点之前被拒绝，而不是被忽略。
- CPU 后端会拒绝 `--tp N` 与 `--layer-split N` 的多 GPU 请求。
  那条警告是为 GPU 主机写的，写的是 "Running on ONE GPU"；在这个后端上请读作"一个
  CPU 设备"。
- Engram 表保持主机映射：上文描述的 GPU 驻留放置需要有设备可放。因此
  `TS_DSV41_ENGRAM_DEVICE=1` 会在打开检查点之前被拒绝，而不是被接受后忽略；除 `0`
  与 `1` 以外的取值同样如此；`=0` 指的正是这条路径本来就在用的主机映射，会被接受。
  由于表是主机映射的，[主机映射的 Engram 选项](#主机映射的-engram-表)——
  `TS_DSV41_ENGRAM_WARM`、`TS_DSV41_ENGRAM_THREADS`、`TS_DSV41_ENGRAM_RANDOM`——
  在这个后端上全部有效。
- `TS_DSV4_VRAM_RESERVE_MB` 会在往这个唯一设备上装层之前，从它的空闲内存里扣除。
  这里那个设备就是主机，所以它的默认保留量（至少 2 GiB；在这条路径使用的 256 ubatch、
  65,536 上下文下约 2,318 MiB，见[为图保留的设备内存](#为图保留的设备内存)）是从系统
  内存中扣除的；而模型装不下
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
开关——`TS_DSV41_ENGRAM_WARM`、`_THREADS`、`_RANDOM`、
`TS_DSV41_SPARSE_FA`、`TS_DSV41_COMPACT_RAW_GATHER`——在这里全部无效。Engram 配置直接来自 GGUF；
`TS_DSV4_THREADS` 在这个后端上默认取
`ProcessorCount`，而不是 min(核数, 32)；`TS_DSV4_CPU_TRACE_DIR` 写出的逐张量文件与
`eng/dsv41-reference.py --output` 写出的同名，于是两个目录可以逐张量对拍。和
`--backend ggml_cpu` 一样，它是正确性与可移植性通道，而不是服务通道：完整检查点在它
上面的吞吐、加载时间与常驻内存占用都没有测过。

Direct CUDA 引擎 `--backend cuda` 也用自己的内核、不经 ggml 运行 V4.1。它具有逐序列槽位、
批量解码、会话状态保留，并通过 `--layer-split N` 在本地按整层放置；路由专家 `--tp N` 仍是
`ggml_cuda` 模式。源码包含针对性的数值约束：
[`Dsv4ExpertKernelTests`](../../InferenceWeb.Tests/Dsv4ExpertKernelTests.cs)
把合成量化专家投影与上游反量化结果对比；
[`Dsv41CudaSlotTests`](../../InferenceWeb.Tests/Dsv41CudaSlotTests.cs)
把槽位隔离、批量解码、环形缓存回退与保留状态，和同一 Direct 引擎的全新运行对比，容差为最大
logit 的 1e-5。这些是内核与状态一致性检查，不是独立的完整模型数值参考或完整检查点质量验证。
模型/设备门控测试需要对应夹具与 CUDA 硬件，不可用的场景会跳过。已经验证了什么、还有什么挡着，见 CUDA 后端说明
`docs/validation/deepseek41-cuda-backend/README.md`（本地验证记录，未提交到 Git）。`--backend mlx`
仍然被拒绝。

## 前向计算图与状态

原生计算图使用四条残差流以及 V4.1 的延迟 hyper-connection 混合。第 1 层与第 14 层会
加入由确定性 token n-gram 哈希选出的 Engram 特征。token 归一化与桶布局来自 GGUF 元数据；
各序列槽位保留各自的 token 历史。

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
- [Engram 元数据加载器](../../TensorSharp.GGML.Native/dsv41_engram_gguf.h)
  与[哈希与配置校验](../../TensorSharp.GGML.Native/dsv41_engram.h)。
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
通过 `string="false"` 加 JSON 字符串来表达：诸如 `\u003c` 这样的 JSON 转义能在不闭合
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
使用 `response_format`。这个组合要求启用 JSON 语法约束。
若要在一次工具往返之后再请求 JSON 最终答案，请保留工具历史与工具目录，并发送
`tool_choice: "none"`。进行中的工具生成与 `response_format` 仍然互斥。校验只看 assistant
的 content 通道；只有推理内容不算最终答案。
这些工具策略与"思考 + JSON"保证适用于 `/v1/chat/completions`。既有的 `/v1/responses`
接口不支持同样的 V4.1 工具历史往返或"推理加 JSON"组合。

由于 V4.1 会渲染工具声明并解析 DSML 调用，它同样可以使用 skills、代码工具
（`--code-exec`），以及服务端的[子智能体委派](../multi_agent.md)（在对话路径上默认开启）。
V4.1 没有发布任何委派相关的实测结果。

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
  它具有针对内核/槽位的约束，但没有独立的完整检查点数值门禁——两者都是正确性与可移植性通道，而不是服务通道。`mlx` 会在读取权重
  之前失败，而不会把 V4.1 的权重塞进并未实现它的计算图。
- 默认只使用一张 GPU。`--layer-split N` 选择整层放置；`--tp N` 启用 routed-MoE 张量并行，
  使用 F32 激活与输出拼接。注意力张量并行与分布式组尚未实现。
- 并发请求拥有隔离的序列槽位。在带 CUDA 融合后端的原生执行器上（`--backend ggml_cuda`），
  它们的 decode 步作为一张按 token 批处理的计算图执行（见[token 批量 decode](#token-批量-decode)）。
  `--backend ggml_cpu`、`TS_DSV4_FUSED=0`、加载了 DSpark 草稿器或设置
  `TS_BATCHED_FUSED_DECODE=0` 时，仍走逐槽前向调用。
- V4.1 的 DSpark 投机解码属于实验性功能。加载器只在 `ggml_cuda` 与 `ggml_cpu` 上接受
  `deepseek41-dspark` 草稿器（`--draft-model`），在其他执行器上拒绝，
  V4 的草稿模型也会被拒绝。合成测试（`DeepSeek41DsparkIntegrationTests`）以及真实草稿器在
  `ggml_cuda` 双 GPU 按层切分与实验性路由专家 TP 下的初步文本/图像 HTTP 检查已通过；
  大量磁盘换页下的通用质量与吞吐仍未获验证。另行进行的 24-token 文本/图像配对检查，
  在两种模式下均匹配普通解码的 token ID 与 `max_tokens` 结束原因，DSpark 实际参与解码
  且进程正常退出。加载草稿器期间，按 token
  批量 decode 与保留缓存都不生效。没有草稿器时，`--spec`（包括 `--spec-type ngram`）
  只提供普通解码。
- K/V cache 在每个执行器上都是 F16，`KV_CACHE_DTYPE=q8_0` / `q4_0` 会在**加载时被拒绝**
  （`DeepSeek41Architecture.ValidateLoad` 在打开检查点之前抛 `NotSupportedException`）；
  显式的 `f32` 会在 stderr 上被告知并按 `f16` 报告。以前它是被静默接受的：原生计算图照样
  分配 F16 cache，而 `KvCacheDtype` 却报告 `q8_0`。这些 cache 不是共享家族交给 ggml
  flash attention 的逐层 K/V 张量（后者在其向量内核的 64/128 宽 head 尺寸下能读
  q8_0/q4_0 K/V）：它们是 MLA 潜变量行（K 兼作 V，宽度为潜变量宽度，CUDA flash-attention
  内核只接受 F16），存放在滑动窗口环、压缩行与索引器行、回退检查点影子以及 DSpark 草稿环
  里，由 `TSG_DSV4_FUSED_ATTN_PREP` / `TSG_DSV4_FUSED_COMPRESS` 写入，由
  `TSG_DSV4_FUSED_KGATHER`、compact / TP gather 和检查点复制以 F16 行直接读取，没有任何
  反量化步骤 —— ggml CUDA、ggml CPU、Direct CUDA 与纯 C# 执行器都是如此。要量化它们就得
  把上述每个内核重新定型；何况检查点自带的训练时 cache 量化（原始行 FP8 E4M3、索引器行
  MXFP4、压缩行 NVFP4）已在每次 F16 存储之前施加，q8_0 块并不会进一步缩小 cache 的信息量。
  上面的启动示例传 `KV_CACHE_DTYPE=f16`，这是唯一什么都不改变的取值。
- 图像/视频输入需要单独准备的视觉伴随文件。编码器与图文计算图有 CPU/CUDA fixture 覆盖，
  并在按层放置与 routed TP 下做过完整检查点的媒体检查。真实图像的 BF16 特征对比超出了
  小规模 fixture 的逐元素容差；见验证报告。目前没有经过验证的音频推理路径。
- 要做同权重对比，需要一个兼容的 llama.cpp V4.1 推理运行时。上面链接的 GGUF 仓库的补丁
  只增加了转换支持。缺少参考实现并不能确立质量或性能的对等。
- 完整检查点的数值 smoke 能产出预期 token，但未通过严格的 F32 输入 oracle 对比（相对
  L2 0.146216，最大绝对误差 2.708920）。量化激活的算术与该参考不同；
  保留的分阶段分析（`docs/validation/deepseek41/smoke18-reference/README.md`，
  本地验证记录，未提交到 Git）未能完全归因最终的差异。贪心结果一致不等于严格的数值一致。

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

最新的托管阶段**在本地与指定 VM 上都通过了 3,651/3,651 项测试**，没有跳过，其中包括在
两台主机上都运行的 84 项音频/图像/API 专项检查。覆盖范围包括工具历史分隔符的往返、合法的
部分 Unicode token、畸形 UTF-8 的拒绝，以及在开始流式输出之前对图像/音频请求的校验。
之前的图像验证阶段在两台主机上都通过了 3,635/3,635 项测试，其中包括本地 91 项专项检查。
每个 Runtime DLL 都与其已验证的 3,597 项测试的原始 tag-family 序列化阶段保持一致。最初的
匹配放置基线使用的是更早的 3,566 阶段主机；那些测量以及中间 3,582 阶段的检查都仍然保留。
推理测试工具另外在本地与 VM 上通过了 33 项单元测试；其当前范围列在
[验证报告](../deepseek41_validation.md)中。
第一次图像验证的 VM 尝试暴露了一个测试准入竞争
（`docs/validation/deepseek41/retained-cache-admission/README.md`，本地验证记录，未提交到
Git）；同步化后的测试类与完整测试通道在两台主机上都通过，原有断言保持不变。
确切的命令、排除项、计数器、哈希以及保留下来的间歇性测试失败
（`docs/validation/deepseek41/managed-correctness/README.md`，本地验证记录，未提交到 Git）
与完整检查点的质量和性能结果分开记录。

最终的按层与 routed-TP 配置使用原生 `6b3b5ab3…` 与托管侧 stage 3,651。
按层切分通过了 **138/138 项推理用例**；routed TP 通过了 **129/130**。
按层计划还包含八个并发长上下文用例：

| 场景组 | 按层切分 | Routed TP |
|---|---:|---:|
| 短提示、JSON、schema、历史与默认并行的工具工作流 | 30/30 | 29/30 |
| required、named、none、串行与并行工具策略 | 30/30 | 30/30 |
| 思考工作流 | 4/4 | 4/4 |
| 非流式工作流 | 4/4 | 4/4 |
| 图像与抽样的视频帧 | 25/25 | 25/25 |
| 中文与 Unicode JSON | 10/10 | 10/10 |
| 单独配置的串行工具工作流 | 10/10 | 10/10 |
| 持续 decode | 15/15 | 15/15 |
| 约 8k 与 32k 输入 token 的长文检索 | 2/2 | 2/2 |
| 每种长上下文尺寸下的四个并发请求 | 8/8 | 不在该配置中 |

在八卡 A40 VM 上，按层配置的持续 decode 中位数为单请求 **34.83 tokens/s**、并发 4 时
**每请求 8.46 tokens/s**，每个被测请求恰好生成 512 个 token。首 token 时间在 7,706 个输入
token 时为 **19.985 s**，在 30,585 个时为 **80.240 s**。
并发长上下文阶段记录了 153,187 个原生 prefill token（含预热），KV 池抢占为零。
decode 结果（`layer8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-steady.json`）、单请求长上下文结果
（`layer8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-long.json`）与并发长上下文统计
（`layer8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-long-parallel-runs.json`）都在 `docs/validation/deepseek41/full-checkpoint/`
下（本地验证记录，未提交到 Git），保留了测量范围与计时证据。

之前失败的 named-thinking 与 thinking-agent 用例在两个最终配置中都已通过。在 routed TP 下，
仍有一个默认并行的 agent 请求在拿到发票结果之前，就与 `read_invoice` 一起发出了参数全为
零占位值的 `calculate_total`。严格检查器拒绝了它；该组因此从第一次 TP 配置的 30/30 变为
29/30。串行策略单独成功并不能抵消这次失败。一次本地的提示词/语法/解析器诊断
（`docs/validation/deepseek41/parallel-tool-dependency/README.md`，本地验证记录，未提交到
Git）保留了模型发出的调用，没有发现任何迫使多出这次调用或产生零值的缺陷。最终的四层
CPU 卸载配置另外通过了 28/30 个默认并行质量用例，保留了两次过早发出依赖调用的失败。
按层配置的 30/30 结果并不能抵消另外两种放置的失败。原生代码与启动设置也有变化，因此
这些结果既不能单独归因于语法改动，也不能证明质量的全面提升。上文的十二项 HTTP 输入
拒绝检查与推理计划相互独立。确切的报告与其余对比记录在最终放置记录
`docs/validation/deepseek41/final-placements/README.md`（本地验证记录，未提交到 Git）中。

既有模型的检查同样保留了回归。最终的 75 用例对比
（`docs/validation/deepseek41/existing-model-regressions/final3651-native6b3/README.md`，
本地验证记录，未提交到 Git）通过了 39/75 个用例，并且在那一轮中相对其配对参考没有引入新的
失败；单独的 Unicode JSON 覆盖通过了 15/15。随后重复进行的 JSON 对比
（`docs/validation/deepseek41/json-performance/completed-r2/README.md`，本地验证记录，
未提交到 Git）暴露了同一请求上的又一次失败，并记录到 Qwen3.5 的首 token 延迟变慢，
尽管其短回答的 decode 更快。这些结果与前一轮“没有引入新失败”的观察分开记录，并不能证明
完全不存在回归。

Qwen3.5 较短的交替对照（`docs/validation/deepseek41/json-performance/qwen35-alternating/README.md`）同样
显示最终延迟变慢。之后一次 72 请求的对照
（`docs/validation/deepseek41/json-performance/qwen35-solo72/README.md`；以上两项均为本地
验证记录，未提交到 Git）固定了原生库，所有回答都通过，也没有复现这次变慢。这些诊断没有
促成任何生产代码修复；不一致的结果及其局限仍记录在验证报告中。

## 按路由读取 CPU 专家（Linux 实验路径）

`TS_DSV4_HOST_EXPERT_READ=1` 为 GPU 执行器卸载到 CPU 的专家开启有界并行读取；
未设置或 `0` 保持原路径，其他值明确拒绝。仅适用于 Linux 的连续 GGUF 主机映射。
它在当前层路由产生后，读取选中专家的 gate/up/down 原始字节，仍由原来的 ggml
算子读取 mmap 并计算，不改变量化、路由或归约顺序。它不预测下一层专家，也不是
跨层异步计算流水线。

TensorSharp 自有 CPU 后端在原有 `MUL_MAT_ID` 执行前准备已登记的专家映射，不增加
路由或读取节点。若路由由 CPU 产生，先执行依赖的图前缀；gate/up/down 使用相同
路由张量时共享一次准备。读取在计算线程组之外执行，所有算术和其他自定义算子
仍交给未修改的 upstream CPU 后端。

文件描述符和 I/O 线程池在模型生命周期内复用。工作线程受 CPU affinity/cgroup、
请求线程配置和 16 线程上限约束；暂存按主机实际内存 allowance 缩放，最大 64 MiB，
单次读取最大 4 MiB。配置共享预算的 `hostPools` 时，这些暂存也须先取得 RAM 额度，
模型释放物理缓冲后才归还；不包含 mmap/OS 页缓存及所有运行时内存。

驻留页跳过读取。准备阶段的多 token 输入检查驻留，单 token 输入可复用专家提示，
最多八次层调用后重查。连续 32 个单 token CPU 图没有实际读取或 major fault 时，
可直接提交整个原始 CPU 图。进程 major-fault 计数变化会恢复准备并失效专家提示；
多 token 图也恢复准备。读取已开启时，`TS_DSV4_HOST_EXPERT_HOT_BYPASS=0` 可关闭
该策略（默认为 `1`，其他值拒绝）。阈值仍是实验性启发式，尚非经过标定的存储成本模型。
这些提示不是锁页或驻留保证，OS 提前回收时原 mmap 仍可正常缺页读取。
读取失败会拒绝此次 Forward，需要重新加载模型；不会使用不完整暂存
充当权重。默认开关尚未改变，网络存储的冷热对照与局限见
[统一内存设计中的验证记录](../design/unified-memory.zh-CN.md)。
