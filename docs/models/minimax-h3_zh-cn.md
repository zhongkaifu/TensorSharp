# MiniMax-H3

MiniMax-H3 **同时生成视频与原生立体声音频**，最长 15 秒、24 fps、32 kHz 立体声。
它不是"视频模型外挂音频"：同一个扩散 Transformer 在**一条 token 序列**里对打包
在一起的"视频 + 音频"潜变量做去噪，因此声音是模型输出的一部分，而不是事后配上去的。

TensorSharp 用**七张**原生 ggml 整图运行它——文本编码、视觉编码、DiT 前向，以及两个
VAE 各自的编码与解码——权重直接从 GGUF / safetensors 的 mmap 常驻绑定。对外的入口是
`MiniMaxH3Model.GenerateVideo(prompt, VideoGenerationParams)`，位于共用的
`IVideoGenerationModel` 接缝之后，因此 CLI 与服务端用同一条路径驱动 H3 和 Wan，
而不必逐个判断具体模型类型。

```sh
TensorSharp.Cli --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --prompt "a red fox trotting through falling snow, cinematic" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output fox.mp4
```

这会写出 `fox.mp4`，以及带生成音轨的 `fox.wav`。

## 支持状态

| 组件 | 状态 |
|---|---|
| 文生视频（`t2v`） | 可用 |
| 图生视频（`i2v`）——让照片动起来 | 可用 |
| 首尾帧（`fl2v`） | 可用 |
| 参考生视频（`ref`）——参考图、参考视频、参考音频 | 可用 |
| 联合立体声音频 | 可用 |
| 视频 VAE 编码 + 解码 | 可用（分块） |
| 超过 22 帧的片段 | 可用（VAE 每次解码 5 个潜变量帧） |
| `--backend cpu`（100% 纯 C#） | 可用：t2v、i2v、fl2v 与参考条件 |

### 纯 C# 后端

`--backend cpu` 运行 H3 时完全不使用 ggml：走的是共用的 `TensorSharp.Models/Direct`
基础算子之上的 `MiniMaxH3Direct{DiT,TextEncoder,VideoVae,AudioVae}`。这覆盖了整条
文生视频路径——提示词编码、去噪、视频 VAE 解码，以及音频声码器。

视觉塔、因果 3-D 视频 VAE 编码器和音频 VAE 编码器也一并移植了，因此每一种条件路径都能用：
`i2v`、`fl2v`，以及参考图、参考视频与参考音频。所有计算都在主机上以 F32 进行，所以要有
比 `ggml_cuda` 慢的心理准备；它存在的意义是正确性、可移植性，以及没有加速器的机器。
256x160x5f 的单步 `t2v` 实测 69 秒，`ggml_cpu` 为 14 秒；同一个 `i2v` 实测 70 秒，
`ggml_cpu` 为 176 秒——因为托管的 3-D 编码比 GGML 那一版便宜得多（单次采样，方向性别读太多）。

在相同输入下与 GGML 路径对比（256x160、5 帧、单步、固定 `--diffusion-seed`）：

| 阶段 | 余弦相似度 | 相对 RMS |
|---|---|---|
| 文本编码器（64 层、32B Q4_K_M） | 0.99999899 | 0.147% |
| DiT，音频速度场 | 0.9994631 | 3.31% |
| DiT，视频速度场 | 0.9975865 | 7.60% |
| i2v（视觉塔 + 3-D VAE 编码 + DiT） | 0.9998974 | - |
| 参考音频（音频 VAE 编码 + DiT） | 0.9994101 | 4.32% |
| 参考图（视觉塔 + DiT） | 0.9985535 | - |
| 视觉塔输出本身 | 0.9999189 | 1.28% |

**视觉塔是唯一一个残差比 GGML 自身内部差异更大、而不是更小的阶段。** 直接测塔的输出：
托管路径对 GGML 的 flash 内核相对残差是 1.28%，对它的显式 softmax 回退是 1.40%，而
GGML 这两个内核彼此的差异是 0.99%。关掉 flash 反而让托管路径的一致性**变差**，所以
F16 的 K/V 转换并不能解释它。在 737k 个元素上余弦达到 0.9999，说明这座塔在结构上是对的
——一座错的塔不会落在 1 附近——但它的残差大约是对照组的 1.4 倍而不是更低，而且没有再继续
追下去。在把精度敏感的参考图条件交给托管路径之前，值得先知道这一点。

在成片上，以对 GGML 路径的 PSNR 衡量：t2v 31.95 dB、i2v 34.87 dB、参考音频 35.27 dB。
让这些数字可读的对照组是 **GGML 与它自己**：它的 flash 内核对 `TS_H3_NO_FLASH=1` 的渲染
只有 28.17 dB——也就是说，每一条托管路径与 GGML 的一致程度，都高于 GGML 自己那两个注意力
内核之间的一致程度。

主干与 GGML 不是逐比特一致的，也不可能是，因为去噪器会放大极小的差异。`TS_H3_DIT_LAYERS`
在**两条路径上**同时截断主干，把速度场的比较变成一条误差随深度变化的曲线：

| 主干深度 | 1 | 10 | 25 | 30 | 35 | 40 | 44 | 47 | 50 |
|---|---|---|---|---|---|---|---|---|---|
| 1 - 余弦 | 1.4e-6 | 3.8e-6 | 9.4e-6 | 2.8e-5 | 2.8e-4 | 1.15e-3 | 1.5e-4 | 2.0e-4 | 1.26e-3 |

前 25 层两者一致到约 1e-5，也就是说实现是对的。再往后差异开始放大，而且是**非单调**的
——深度 40 处比 44 处还大。正是这个形状排除了 bug：如果某一层引入了错误，那么从该层往后
每个深度上曲线都不会下降。这条主干的后半段单纯就是增益很高。

作为尺度参照，GGML 与**它自己**的分歧比它与托管路径的分歧还大。在全深度下，`h3_attend`
的 flash 内核对它自己的显式 softmax 回退（`TS_H3_NO_FLASH=1`）是 1 - 余弦 = 2.97e-3，
而托管路径对两者中任意一个都是 1.26e-3。文本编码器实际深度更浅、两边也都没有这类内核，
一致度达到 1e-6——正是这一点说明共用的机制（量化矩阵乘、RMSNorm、rotate-half RoPE、
GQA 注意力、SwiGLU）是对的。

`TS_H3_DUMP_TE`、`TS_H3_DUMP_VEL_V` 与 `TS_H3_DUMP_VEL_A` 把这些张量写到磁盘，使两条
路径可以在**同一次**前向上比较，而 `TS_H3_DIT_LAYERS` 截断主干，把这种比较变成一条曲线。
不要拿成片来比：采样器会把第一步的差异不断放大，所以几步渲染出来的结果说明不了实现是否
正确。另外注意 `--seed` 是**采样**种子，视频生成的噪声来自 `--diffusion-seed`；不设它就
去比较两次运行，实际上是在悄悄比较两份不同的噪声。

### 长片段与 FP16 注意力上限

H3 在**一条**打包序列上做双向注意力，因此它的 key 数量是整个片段而不是一个窗口：
640x384 的请求在 22 帧时是 2364 个打包 token，到 107 帧就是 8646 个。ggml 的 flash
attention 内核用 FP16 累加 softmax 分子，只留了三个比特的余量，所以这个 key 数量本身
就是一份数值预算——而 H3 正是会把它花光的模型：检查点里没有 `v_norm`，value 流带着与
`h3_mm` 下一步所防范的同样无界的幅度。不做处理时，107 帧的片段在第一个去噪步就溢出成
NaN，输出每个像素全黑、每个音频采样被削平，而同一请求在 73 帧下却能正常生成。

`h3_attend` 现在会按 key 数量取一个 2 的幂对 V 预缩放，并在输出端还原——因为注意力对 V
是线性的，所以这一步是精确的——这样无论片段多长，累加器看到的有效 key 数都有界。采样器
也不再把发散的结果写出去：出现非有限速度时会直接报错并指出是哪一步，而不是保存一个全黑
文件。设置 `TS_H3_TRACE=1` 可以打印每一步的潜变量与速度幅度。

## 性能

两台机器，两个结论。两组数据都是 22 帧、8 步、相同随机种子，对比
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) 在**该机器上**的
最佳配置，两个引擎用的是同一批权重文件。两组数字分开列出，并且刻意**不做平均**——
这台 CUDA 机器面对这套模型是内存吃紧的，Metal 那台不是，两者的分歧正来自这里。

### M5 Pro / Metal

| 分辨率 | stable-diffusion.cpp | TensorSharp | |
|---|---|---|---|
| 256×256 | 49.3 秒 | **20.9 秒** | 快 2.4 倍 |
| 640×384 | 108.5 秒 | **63.1 秒** | 快 1.7 倍 |

### RTX 3080 Laptop 16 GB / CUDA

发布版构建，跑在 `ggml_cuda` 上，三次取最好。驱动 566.36、CUDA 12.6、i7-11800H、
31.7 GB 内存、Windows 11、.NET 10.0.204。

| 分辨率 | stable-diffusion.cpp | TensorSharp | |
|---|---|---|---|
| 256×256 | **37.8 秒** | 43.6 秒 | sd.cpp 快 1.15 倍 |
| 640×384 | **59.8 秒** | 63.7 秒 | sd.cpp 快 1.07 倍 |

| 引擎 | 显存峰值（640×384） |
|---|---|
| TensorSharp | 15 780 MiB |
| stable-diffusion.cpp | 12 035 MiB |

显卡共 16 384 MiB。stable-diffusion.cpp 用的是 `97d2990`（ggml `8e800ce`），按
`SD_CUDA=ON`、arch 86 从源码重新编译，运行参数为
`--auto-fit --stream-layers --diffusion-fa --rng cpu`——它默认的 `--offload-to-cpu`
路径在这台机器上**根本跑不了**这个模型：它会试图把 17.7 GB 钉进只有 12.3 GB 空闲的内存。

### 两组数字为什么不一致

**单个去噪步**上，TensorSharp 在 CUDA 上同样领先——按 8 步与 16 步的斜率折算是
3.325 秒对 3.338 秒，逐步直接测量是 3.00–3.11 秒。它在 CUDA 上输掉的完全是**固定的启动
开销**；而在剩下 3.9 秒的差距里，约有 3 秒根本不是推理：TensorSharp 编码的是 H.264，
sd.cpp 只是把 MJPEG + PCM 写进 AVI，另外还有 .NET 进程相对原生二进制的启动时间。

两组硬件之所以给出不同结论，是因为这台 CUDA 机器是刻意挑来“难为”引擎的：16 GB 显存、
31.7 GB 内存，面对一套约 35.5 GB 的模型——权重装不下，页缓存也装不下，于是启动开销成了
主导项；换成一张装得下模型的卡就不会这样。两组数字都是真实的，究竟哪一组描述的是你的
运行，取决于机器有多少内存，所以引用任何一组之前请先看清硬件标注。

### 本轮带来的提升——RTX 3080 Laptop 16 GB / CUDA

同一台机器、同一负载，[四个网络之间的显存调度](#四个网络之间的显存调度)一节所述工作的
前后对比：

| 分辨率 | 之前 | 之后 | |
|---|---|---|---|
| 256×256 | 67.2 秒 | **43.6 秒** | 1.54 倍 |
| 640×384 | 89.0 秒 | **63.7 秒** | 1.40 倍 |

## 四个网络之间的显存调度

四个网络依次运行，所以显存峰值是 `max(...)` 而不是总和——但前提是每一个都在下一个到来
之前真正被交还，而且把下一个搬上显卡本身不能成为瓶颈。在 16 GB 的卡上，这两件事都不是
免费的。下面所有数据都测自上文那台 RTX 3080 Laptop 16 GB；`TS_H3_PHASE=1` 会打印这些
数字所来自的逐阶段耗时。

### 视频 VAE 加载之前先释放去噪器

去噪器在 Q4_K 下约 10.6 GB，视频 VAE 另有约 5.2 GB。两者同时常驻，在 16 GB 的卡上就是
15.8 GB 的权重——还没算任何一次激活。而 Windows/WDDM **不会**让这次分配失败：它静默地
用共享主机内存兜住溢出的部分，于是整个解码都跑在 PCIe 速度上。实测中，解码窗口全程钉在
16 384 MiB 显卡的 16 041 MiB 上。

现在，当 VAE 无法与去噪器并存时，会先把已经用完的去噪器的设备常驻交还回去。解码期间的
显存峰值降到约 5 600 MiB，在 640x384 下值 **22 秒**。这是一个上限而不是策略变更：判断
依据是当前真实的空闲显存，所以显存充裕的卡仍然保持去噪器常驻、行为与以前完全一致；非
CUDA 的 GGML 后端一律不动——统一内存设备没有什么可交还的。代价只是**下一次**请求要重新
上传 DiT，而它的权重是仍留在内存里的 mmap GGUF 页。同门的 Qwen-Image 流水线早就这么做了。

### 去噪器文件在上传之前先被顺序读入

权重是以指针形式绑定到 mmap 的 GGUF 上的，因此第一次上传会**在主机到设备的拷贝内部**
把每一页从磁盘缺页调入——这正是驱动最糟的情形：

| 路径 | 速率 | 10.6 GB 需要 |
|---|---|---|
| H2D，源为锁页主机内存 | 19.09 GB/s | 0.56 秒 |
| H2D，源为已驻留的可分页内存 | 5.97 GB/s | 1.78 秒 |
| 普通顺序读该文件 | 2.70 GB/s | 3.93 秒 |
| **H2D，边拷贝边缺页调入** | **0.91 GB/s** | **11.6 秒** |

所以现在会先把去噪器文件顺序通读一遍。有两个细节才让它划算：

* **开始得足够早。** 文本主干一产出隐状态就发起这次读取，于是它与编码器的拆卸并行——
  交还 551 个逐张量设备缓冲是驱动的活，读取是磁盘的活，两者互不争抢。
* **不在上传之前 join。** 读取与上传按同样的顺序走这个文件，而读取是两者中更快的那个，
  所以让它继续跑就能一直领先，拷贝落到的都是已经就位的页。先 join 反而会把一整趟读取
  串在一整趟拷贝前面。

第一个去噪步从 14.87 秒降到约 10.2 秒，去噪器的权重搬运从 11.8 秒降到 8.25 秒。
开与不开，输出**逐字节一致**。

修复之前已经确认这段启动开销与 token 数无关——256x256 是 11.77 秒、640x384 是
11.55 秒——正是这一点把它定性为权重搬运而不是构图。模式用 `TS_H3_PREFAULT` 选择，
见[环境变量](#环境变量)。

## 需要的文件

四个网络协同工作，因此一次运行需要四个文件。两个去噪器是**两个独立的检查点，而不是
开关**——你加载哪个，决定了它接受哪种条件输入。

| 文件 | 大小 | 来源 |
|---|---|---|
| `minimax_h3_fl2va_pruned-Q4_K.gguf` | 10.64 GiB | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF)：文本 + 关键帧 |
| `minimax_h3_ref2va_pruned-Q4_K.gguf` | 10.60 GiB | 同一仓库：文本 + 参考 |
| `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | 16.97 GiB | 同一仓库：文本编码器，两者共用（换成 `-Q2_K_M.gguf`，12.20 GiB，可搭配更小的去噪器） |
| `minimax_h3_video_vae_fp16.safetensors` | 5.21 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)（unsloth 仓库的 `vae/` 下也有镜像） |
| `minimax_h3_audio_vae_fp32.safetensors` | 0.61 GB | 同上；不给则输出无声视频 |

一套约 35.5 GB（Ref2VA 为 35.4 GB）。每个网络都是依次加载并**释放**的，所以显存峰值是 `max(...)` 而不是
总和；之后再补第二个去噪器只多花它自身约 10.6 GiB，因为编码器和两个 VAE 是共用的。

伴随文件会在去噪器旁自动解析，也可用 `--video-vae`、`--video-text-encoder`、
`--audio-vae` 指定。

仓库里有两个配置文件，首次运行会把这些文件全部下载下来；两者只有 `model` 一项不同，
所以文本编码器和两个 VAE 只会下载一次：

```sh
TensorSharp.Server.Host --config config/minimax-h3-fl2va.json # 关键帧
TensorSharp.Cli         --config config/minimax-h3-ref2va.json \
  --ref-image person.png --prompt "…" --output out.mp4        # 参考
```

两者都写了 `"backend": "ggml_cuda"`（命令行上的 `--backend` 优先），并设定宽 640、
高 384、22 帧、24 fps。两者都**故意不设**步数与 guidance：一个随仓库分发的配置必须在
两个宿主上都能解析，而两者对步数的拼法不同（服务端是 `--video-steps N`，CLI 是
`--diffusion-steps N`），并且服务端根本没有 `--cfg`——因此这里用的是模型自己的默认值。

> **文本编码器的 GGUF 不带分词器**，而这恰恰是配置文件唯一没法替你下载的东西：
> 自动下载只能补齐那些**以命令行选项形式存在**的文件，而分词器不是选项。请把
> [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/processor)
> 里的 `vocab.json` 和 `merges.txt` 放到它旁边，或用 `TS_VIDEO_TOKENIZER` 指向它们。
> 两个已发布的 GGUF **完全没有元数据**，因此 TensorSharp 靠张量而不是架构字符串来识别
> H3（文件里确实写了架构名时，接受 `minimax-h3` / `minimax_h3`）。

> **必须用 `--cfg 1.0`。** H3 是 CFG 蒸馏模型，超过 1.0 质量会明显劣化，
> TensorSharp 会直接拒绝。4–8 步是常用工作点，所以迭代很快。

## 条件模式

一张图对 H3 来说可以有两种**完全不同**的含义，而且它们用**不同的检查点**。
`--video-mode` 用来显式指定；不指定时会根据你传入的内容自动推断。

| 你想要什么 | 模式 | 检查点 |
|---|---|---|
| "让这张照片动起来" | `i2v` | FL2VA |
| "从照片 A 变到照片 B" | `fl2v` | FL2VA |
| "用这个人物，生成全新场景" | `ref` | Ref2VA |
| "参考这个产品/人物，但换机位、背景和构图" | `ref` | Ref2VA |
| 只有文本 | `t2v` | 都可以 |

当前生效的是哪一个分区，是从**文件名**里读出来的——不区分大小写地看有没有 `ref2va`，
所以改名或重新量化时要保留这个子串。用一个检查点去要另一个检查点的模式，不会静默地
丢掉输入：请求会直接失败，并在错误信息里指出该改成加载哪个文件。关键帧与具名参考出现
在同一个请求里，在两个检查点上都会被直接拒绝。

### 文生视频

```sh
TensorSharp.Cli --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --prompt "a red fox trotting through falling snow, cinematic" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output fox.mp4
```

### 图生视频——让照片动起来

关键帧会被**两次**用作条件，两者缺一不可：VAE 编码出的潜变量把它钉在所锚定的那一帧上，
同一张图还会经 Qwen3-VL 视觉塔进入提示词，为整个片段描述画面内容。只有潜变量的话条件会
衰减——片段开头贴合图片，一两秒后就跑偏了；这在 22 帧时看不出来，到 124 帧就很明显。

图片会成为**第一帧**，提示词决定后续发生什么。

```sh
TensorSharp.Cli --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --image portrait.jpg \
  --prompt "the person turns toward the camera and smiles, subtle handheld motion" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output animated.mp4
```

加 `--video-mode i2v` 可显式声明。图片会适配到生成画布；宽高比与 `--width`/`--height`
不一致时会被居中裁剪（运行时会打印一行 `[h3] conditioning image … centre-cropping to fit`
提示），想保留整张图就不要指定尺寸。

### 首尾帧

两端都被钉住，模型补出中间的运动。

```sh
TensorSharp.Cli --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --image start.png --end-image end.png \
  --prompt "a slow cinematic push-in" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output morph.mp4
```

只给 `--end-image` 也可以：片段会结束在那一帧。

### 参考生视频——保留主体，其余全部重来

Ref2VA 把图片当作身份与外观的**参考**，而不是画面帧。第一帧完全不必与它相似：
人物、产品或物体的特征会保留下来，而机位、背景和构图完全由提示词决定。

这需要 **Ref2VA** 检查点——`i2v`/`fl2v` 与 `ref` 是两个不同的文件，不是一个开关。

```sh
TensorSharp.Cli --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-image person.jpg \
  --prompt "the same woman sits at a table in a sunlit cafe by a window, drinking coffee, wide shot" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output cafe.mp4
```

多次传入 `--ref-image` 可以给出多张参考（最多九张），例如一个人物加上她手中的产品：

```sh
TensorSharp.Cli --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-image person.jpg --ref-image bottle.png \
  --prompt "she holds the bottle up to the light on a rooftop at golden hour, slow orbit" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output rooftop.mp4
```

注意事项：

* 参考图只会被**缩小**到生成画面的面积，并保持自身的宽高比，不会被拉伸；
  输出画布也**不会**取自参考图，请用 `--width`/`--height` 明确指定。
* **每张参考图都是同一条打包序列里的额外 token**，所以开销是线性的、也容易预估：
  一张 640x384 的参考是 240 个 token，576x448 的是 252 个。在 RTX 3080 Laptop 上、
  针对 640x384 的 22 帧片段实测，一个去噪步从不带参考的 4.37 秒升到带八张参考的
  9.38 秒——每张参考每步约 626 毫秒，从一张到八张都是这个斜率。
* **超过大约四张参考之后，主导耗时的是 Qwen3-VL 那一趟，而不是去噪器。** 每张参考会
  往提示词里加约 250 个视觉占位 token，而这段提示词要过全部 50 层：两张参考是 548
  token 的提示词，八张是 2086，于是文本条件从约 65 秒涨到约 447 秒，而整个 8 步去噪
  才约 75 秒。先想办法减少参考数量、提高参考质量，再考虑减少步数。
* TensorSharp 把参考图上限定为九张。参考实现本身没有上限；这个上限存在，是因为打包
  序列是不带 mask 全量注意的，它的长度既是时间预算也是数值预算。
* 在送进语言模型之前，参考会**按你传入的顺序编号**——静态图是 `<Picture 1>`、
  `<Picture 2>`……，视频片段是 `<Video 1>`……，音频是 `<Audio 1>`……——所以提示词里
  可以直接点名："`<Picture 2>` 里的那件夹克"。自带声音的参考视频会被当成**两个**
  条目呈现，除了 `<Video n>` 还会占用一个 `<Audio n>` 编号。
* 提示词里要写清画面中有谁。参考图提供身份，镜头内容仍然要靠提示词描述，
  否则会得到一个画得很好、但主体缺席的场景。
* 参考也可以是**视频片段**或**音频**，不限于静态图——见下文。

> **在 Ref2VA 检查点上，普通的 `--image` 会被当作参考图。** 因此只会"上传一张图"的
> 客户端（包括 Web UI）无需任何额外字段即可使用 Ref2VA。想写明确可以加 `--video-mode ref`。

### 参考视频与参考音频

`--ref-video` 接受视频**文件**或一个帧目录，`--ref-audio` 接受 WAV/MP3/Ogg。
视频自带的声音要用 `--ref-video-audio` 单独给出，按位置与 `--ref-video` 配对——
容器里的音轨无法通过帧解码器读取，所以两者是分开的输入。

```sh
TensorSharp.Cli --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-video walk.mp4 --ref-video-audio walk.wav \
  --prompt "the same woman walks along a beach at sunset, wide shot, waves behind her" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output beach.mp4
```

不带画面的独立音频同样可以作为参考：

```sh
TensorSharp.Cli --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-image singer.jpg --ref-audio song.wav \
  --prompt "she performs on a small club stage under a single spotlight" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output stage.mp4
```

参考视频会经过这些处理：

* 重采样到 H3 自己的 **24 fps**，并向下对齐到 17k+5 帧网格，且不超过本次生成的帧数。
  24 fps 下不足 5 帧的片段会被拒绝。
* 画布由它自身的宽高比在 768 px 附近推出，再按面积封顶并对齐到 32。比这更**小**的
  源保持自身尺寸：放大并不会产生可供参考的细节，代价却很大——448x320 的片段若放大到
  1088x768，会带来 5712 个条件 token，而被生成的片段本身只需要 1680 个。
* 语言模型以 **2 fps** 看它，每次看两帧，并标注该对所处的时间；去噪器则以潜变量的
  形式看到全部内容。
* 音频会重采样到音频 VAE 的 32 kHz 立体声，并截断到生成片段的时长。

> **参考视频是 H3 最昂贵的输入。** 一个 22 帧 448x320 的参考会在输出本身所需的
> 1680 个 token 之外再加 980 个条件 token，而且 VAE 需要先把这 22 帧全部编码。
> 在 M5 Pro 上实测：编码参考约 14 秒，之后每个去噪步约 9 秒（不带参考时约 5 秒）。

## 如何获得高质量

### 先把主体放进画面——这一点比其余所有设置加起来都重要

**主体在画面中占多大比例，比下面任何参数都更决定成败。** 模型只能按给定的分辨率
渲染它拿到的东西；如果条件图里一张脸只有 40 像素宽，再多的步数、再大的输出尺寸
也补不回来。

用一张拍摄公园通行证的照片实测（证件上印着两个很小的人像），提示词、种子、20 步
全部相同：

| 输入 | 输出中的人脸 |
|---|---|
| 原样照片，640x384 | 约 40 像素宽——糊成一团，认不出 |
| 裁到两个人物，768x480 | 约 250 像素宽——头发、皮肤纹理、领针都清晰可辨 |

除了裁剪之外什么都没变。如果你在意的是人脸，请先把源图裁到人脸再生成。

> **条件图永远不会被拉伸。** 当画布宽高比与图片不一致时，图片会被居中裁剪以适配，
> 并在运行日志中说明。否则一张 4:3 的照片被强行放进 640x384（5:3）会被横向压缩 25%，
> 而关键帧**就是**第一帧，这个变形会被后面每一帧继承。不指定 `--width`/`--height`
> 可以直接采用图片自身的宽高比；想控制取景就自己裁剪源图。

有两个设置起决定性作用，其余的相比之下都是零头。

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--width` / `--height` | 640×384；有条件图时按该面积取图片的宽高比 | **影响最大的一项。** 人脸需要像素：256×256 下一张脸只有几十像素宽，无论其他参数怎么调都会糊、都会变形。 |
| `--diffusion-steps` | 20 | 消除运动主体周围的彩色边缘。8 步是快速档，约 20 步干净，超过 30 提升有限。耗时大致线性增长。 |
| `--diffusion-seed` | 随机 | 有些种子构图就是更好，是最省事的重试手段。 |
| `--flow-shift` | 12 | 改变时间表把步数花在哪里，通常不需要动。 |
| 量化等级 | — | 去噪器用 `-Q8_0` 优于 `-Q4_K`，文本编码器用 `Q4_K_M` 优于 `Q2_K_M`，显存够就用更高的。 |

`--cfg` **不是**质量参数：H3 是 CFG 蒸馏模型，只接受 1.0。同理 `--negative-prompt`
也不起作用——根本没有无条件分支可以用来做反向引导；`--cfg-cache-stride` 缓存的正是
那一趟无条件计算，因此同样没有可缓存的东西。`--sampler` 是 Wan 家族的开关，H3 跑的是
自己的 flow-match 时间表，会忽略它。

同一个图生视频请求下实测（M5 Pro、`ggml_metal`、seed 42、22 帧）：

| 设置 | 耗时 | 结果 |
|---|---|---|
| 256×256、8 步 | 26 秒 | 模糊，人脸无法辨认 |
| 640×384、8 步 | 79 秒 | 清晰，但运动的手周围有彩色边缘 |
| 640×384、24 步 | 212 秒 | 干净 |
| 576×448（图片宽高比）、20 步 | 187 秒 | 最好——没有任何拉伸 |

> **在服务端，尺寸是启动参数。** Web UI 只发送提示词和图片，请求会继承服务端默认值。
> 启动时加上 `--video-width 640 --video-height 384 --video-steps 20 --video-frames 22`
> （或者宽高干脆不加，让每个请求按自己的图片取宽高比；只给宽高中的一个时，H3 会按
> 条件图的宽高比推出另一个）。`--width` / `--height` 是别名，`--video-mode` 可以为
> 只提供一种模式的部署固定条件模式。服务端**没有 `--cfg`**——也没什么可设的，
> 因为 H3 自己就把它锁在 1.0。

> **强行指定与图片不匹配的尺寸会裁掉画面。** 4:3 的照片被放进 640×384 会损失约 20% 的高度，
> 由上下两边分摊。做图生视频时不指定宽高，就会按图片的宽高比来。

> **本次修订改变了音轨输出。** 去噪器输出的是白化后的音频潜变量，而解码器需要 VAE
> 自身的尺度；此前缺少了反白化这一步，因此每一条生成的音轨都是从一个约小 1.9 倍的
> 潜变量解码出来的。它只表现为音量偏低和音色略有偏差，不会是静音或噪声。已对照参考
> 实现对同一潜变量的解码验证：余弦 0.99998、增益 1.0000 倍（此前为 0.81 倍）。

## 尺寸与长度

* 宽高向**上**取整到 32 的倍数。默认 640×384；有条件图时按该面积取图片的宽高比。
* 帧数向上对齐到 **17k+5** 网格——5、22、39、56、73、90、107、124……这来自视频 VAE
  的时间分块，不是随意规定的：每个 5 潜变量帧的分块产出 17 个像素帧，另加 5 帧的
  起始段。默认 22。
* fps 固定为 24，传别的值会被覆盖。
* **任意网格长度的片段都能正确解码**：VAE 每次跑 5 个潜变量帧，带 2 帧前瞻，并对接缝
  做交叉淡化，与参考实现一致。反过来把长片段一次性解码，细节会被逐步冲淡——对着条件
  照片测量，第 0 帧的相关度从 22 帧时的 0.97 掉到 90 帧时的 0.86——所以分块是正确性
  要求，而不是优化。

> **长片段会保留条件图，但不一定保留取景。** 关键帧钉住的是片段的**开头**。在 124 帧
> （5.2 秒）里，一个描述了不同镜头的提示词会占上风：把"两个人争吵的中景、手持镜头"
> 用在一张**拍摄证件的照片**上，镜头会一路推进到证件里，大约第 60 帧就变成了那个中景。
> 换成不要求运镜的提示词，同样 124 帧的片段全程与源照片保持 0.94–0.97 的相关度。
> 想让照片一直在画面里，就在提示词里说明，或者生成更短的片段。

## HTTP API

三个路由共用同一个解析器：`POST /api/video-generate`、
`POST /api/video-generate/stream`（请求体相同，以 SSE 推送进度），以及 OpenAI 风格的
`POST /v1/videos/generations`。

```sh
curl -s localhost:5000/api/video-generate -H 'content-type: application/json' -d '{
  "prompt": "a red fox trotting through falling snow, cinematic",
  "width": 640, "height": 384, "frames": 22, "steps": 8, "cfg": 1.0,
  "imagePath": "card.jpeg", "videoMode": "i2v"
}'
```

参考条件走同一个路由，只是换成 Ref2VA 检查点：

```sh
curl -s localhost:5000/api/video-generate -H 'content-type: application/json' -d '{
  "prompt": "the same woman sits at a table in a sunlit cafe by a window, wide shot",
  "width": 640, "height": 384, "frames": 22, "steps": 20, "cfg": 1.0,
  "referenceImages": ["person.jpg", "bottle.png"], "videoMode": "ref"
}'
```

返回 `{ ok, url, audioUrl, width, height, frames, fps, seed, codec, elapsedSeconds }`。
模型没有产出音轨时 `audioUrl` 为 null。完整字段为 `prompt`、`width`、`height`、`frames`、
`steps`、`cfg`、`fps`、`seed`、`flowShift`、`imagePath`（或内联 base64 的 `image`）、
`videoMode`、`generateAudio`、`endImage`、`referenceImages`、`referenceVideos`、
`referenceAudios`、`referenceVideoAudios`；Wan 的 `negativePrompt`、`sampler`、
`cfgCacheStride` 和 `cfg2` 字段会被接受，但 H3 会忽略它们。其中大多数
**只接受 camelCase**——只有 `videoMode`、`generateAudio`、`endImage` 以及四个 `reference*`
列表同时接受 snake_case 拼法（`video_mode`、`generate_audio`、`end_image`、
`reference_images` 等），两种拼法同时出现时以 camelCase 为准。写成 `image_path` 或
`flow_shift` 会被静默忽略而不是报错，因此一律用 camelCase 最稳妥。`referenceVideoAudios`
与 `referenceVideos` **按下标**配对，和 `--ref-video-audio` 的按位置配对是同一条规则。`imagePath` / `endImage` /
`referenceImages` 引用的是之前通过 `/api/upload` 上传的文件；上传目录之外的路径会被拒绝。
`/v1/videos/generations` 接受同样的字段（可用 snake_case 的同样是那七个：`video_mode`、
`reference_images` 等），返回 `audio_url`。

模型侧的拒绝——检查点与模式不匹配，或关键帧与参考同时出现——会以 **400 并带上模型自己
的错误信息**返回，而不是一个笼统的 500，因此该改成加载哪个文件就写在响应体里。

`GET /api/models` 会返回一个 `video` 对象（非视频模型为 null），其中包含 `family`
（`minimax-h3`）、`supportsAudio`、`supportsImageConditioning`、
`supportsEndImageConditioning`、`supportsReferenceConditioning` 和 `maxReferenceImages`。
Web UI 正是据此决定要不要提供首帧、尾帧或最多九张参考，而不是去匹配架构字符串。

音频写成独立的 `.wav` 而不是封装进 MP4，因为封装需要一个未必安装的编码器。合并：

```sh
ffmpeg -i fox.mp4 -i fox.wav -c:v copy -c:a aac fox_with_audio.mp4
```

`--no-audio`（对应 `"generateAudio": false`）会完全跳过音频解码，省下音频 VAE 的时间和
内存；纯视频模型会忽略它。

## 架构要点

这些都记在这里，是因为检查点本身没有任何元数据，无从推断。以下内容与
stable-diffusion.cpp、上游参考 PyTorch 实现、以及 GGUF 张量表三方交叉核对过。

### 扩散 Transformer

50 层、约 193 亿参数，单流：文本、条件帧、目标音频、目标视频在**同一条序列**里做
全双向注意力，**没有 cross-attention**。

| | |
|---|---|
| hidden | 5376 |
| 注意力 | 56 头 × 128 = **内部 7168**，比 5376 的残差流更宽 |
| MLP | SwiGLU，gate 在前 |
| patch | (t, h, w) = (1, 2, 2)；视频 24 通道 → 96 维 patch |
| 音频 | 32 通道，每个潜变量帧、每个立体声通道各一个 token |

一个 token 内部的 96 个视频数值是**通道优先、patch 其次**（`c*4 + ky*2 + kx`）。
反过来排列统计特征完全一样，却会悄悄打乱每一个 token。

**AdaLN** 用的是学习到的曲线表，而不是时间步 MLP：`adaln_t_table` 形状 `[8, 1025]`，
在时间步处插值，每层一个 `Linear(8 → 96768)` 为 **3 个模态**（0=视觉、1=文本、2=音频）
各产生 6 个调制向量。一段 token 取 `timestep*3 + modality` 这一行。

**RoPE** 是三轴的，`inv_freq` 是学习到的 16 项，因此 128 个头维中有 96 维参与旋转。
位置是**连续浮点数**：时间轴以音频潜变量为单位（1/40 秒），使两条流共享同一条时间线；
空间轴按画面的几何平均尺度归一化。

### 视频 / 音频双 shift

视频要 shift 12，音频要 shift 3。H3 没有跑两个采样器，而是让采样器留在视频时间表上、
由模型做换算：音频流按"同一底层时间下 shift-3 时间表会有的 sigma"来加条件，
其速度再乘以 `d(sigma_a)/d(sigma_v)`，于是共享的 Euler 步会沿音频自己的轨迹积分。
两条时间表都从 1 开始、到 0 结束，所以两条流同时落地。`--flow-shift` 只影响视频流。

### 文本编码器

Qwen3-VL-32B 截断到 50 层并**去掉最后的 norm**——DiT 取的是未归一化的第 50 层隐状态，
其中数量级上万的离群激活是模型的一部分，不是 bug。**没有 chat template**：
提示词就是原始文本，套模板会让每个隐状态都发生偏移。

### 视频 VAE

空间 16 倍、时间 4 倍。解码器是**纯 36 层 Transformer**，没有任何反卷积。
每个潜变量体素是一个 token，最后一个 `Linear(2048 → 3072)` 加 depth-to-space
一步完成全部上采样。

解码**按 256 px 分块，而且这是正确性要求而非优化**：解码器的 RoPE 坐标会按拿到的
范围做长度归一化，只有在它训练时的那个范围内才成立。把 640×384 一次性解码，
即使每个分块单独都完全正确，整体也会出现明显的错切。

编码器是因果 3D CNN；但图像条件只需要单帧，而因果 padding 补的是两帧**零**，
于是每个卷积核只有最后一个时间切片起作用，整个网络**精确地**退化成 2D 卷积。

### 音频 VAE

DAC/BigVGAN，32 潜变量通道 → 32 kHz 立体声。上采样倍率 {5,5,2,2,2,2,2} 相乘为 800，
而 32000/800 正好是 40 Hz 的潜变量帧率。使用无混叠 snake 激活：每个非线性都被夹在
"2 倍上采样 / 激活 / 2 倍下采样"之间，避免把高频折回带内。

与视频潜变量不同，音频潜变量**原样**送进解码器——**不**应用 `latents_mean` /
`latents_std`。应用它们会让幅度小约 15 倍，得到一条频谱看似合理、实际几乎听不见的音轨。

## 环境变量

| 变量 | 作用 |
|---|---|
| `TS_H3_PREFAULT` | 在去噪器第一次上传之前顺序读入其文件。`0` 关闭，`1` 串行（读完再上传），`2` 与文本条件阶段重叠，`3` **默认**——与上传流水线并行 |
| `TS_H3_PREFAULT_THREADS` | 该预读使用的读取流数量，默认 `1` |
| `TS_H3_PHASE` | `1` = 打印逐阶段耗时——编码器 open / 主干 / 拆卸、预读、每一个去噪步、VAE open / 解码。原有的单行汇总只告诉你哪个阶段慢，这个告诉你慢在它的哪一半 |
| `TS_H3_TE_GROUP` | `<n>` = 把 50 层文本编码器主干按每组 `n` 层运行，每组跑完就交还该组的设备副本。**默认关闭**；任何不小于层数的 `n` 都精确复现原来那次整段单调用 |
| `TS_H3_TRACE` | `1` = 打印每个去噪步的潜变量与速度幅度 |

**为什么默认是 `TS_H3_PREFAULT=3`。** 模式 3 让读取在上传进行的同时继续跑。它之所以
胜出，是因为读取与上传按同样的顺序走这个文件，而读取是两者中更快的那个，于是它一直
领先，拷贝拿到的页都已经就位。模式 1 先 join，等于把一整趟读取串在一整趟拷贝前面。
模式 2 开始得更早，与文本条件阶段重叠，结果反而更差：文本编码器会把它自己的 17 GB
从同一个页缓存里流过去，正好把预读刚放进去的页挤掉。保留模式 2，是因为在内存足够同时
容纳两个文件的主机上，这个结论会反过来；模式 0 用来做 A/B——无论哪种模式输出都一致。

**为什么只用一个读取流。** 这与 `GgufReader.PrefaultFileCache` 用十六个流的做法正好
相反，而且是刻意的：那个预热跑在其他一切工作之前，整台机器归它用；这个预读却与编码器
拆卸、以及它要帮忙的那次上传同时进行。640x384 下三次取最好：1 个流 63.9 秒、4 个流
64.9 秒、16 个流 66.6 秒，而且十六个流时单是编码器拆卸就从 2.2 秒涨到 4.0 秒。只有在
存储不是争抢资源的主机上才值得调高。

**为什么 `TS_H3_TE_GROUP` 默认关闭。** 在 16 GB 的卡上，17 GB 的 Qwen3-VL 主干确实会
溢出到共享主机内存，分组也确实消除了这次溢出——640x384 下设备占用峰值从 16 041 MiB 降到
12 981 MiB，输出逐比特一致——但它仍然**慢 3 秒**。原因是这条主干是一次性的 prefill，
提示词很短（实测是十个 token），因此每个权重只被读一次，溢出的那约 1.3 GB 只付一次
PCIe 传输；分组并不能让这件事更便宜——它照样要搬完全部 17 GB——却额外多出每组一次的
分配 / 失效开销。只有当权重被逐步反复读取时溢出才会累积，那说的是去噪器，不是这条主干。
保留它，是为了显存而不是时间成为瓶颈的场景——更小的卡，或者另有进程要用这块显存。

### 试过但没有效果的做法

都测自 RTX 3080 Laptop 16 GB，记在这里，免得有人再跑一遍：

| 尝试 | 结果 |
|---|---|
| 文本编码器分层分组（`TS_H3_TE_GROUP`） | 消除了编码器自身的溢出，16 041 → 12 981 MiB，逐比特一致——但慢 3 秒。因此默认关闭 |
| 对视频 VAE 也做预读 | 纯亏：多花 1.9 秒读取，解码没有变化（9.4 秒对 9.1 秒）。4.85 GB 的体量并不受缺页约束，它的解码是计算瓶颈 |
| 多流预读（`TS_H3_PREFAULT_THREADS`） | 在这里更差：1 / 4 / 16 个流分别是 63.9 / 64.9 / 66.6 秒（640x384，三次取最好），而且单是编码器拆卸就从 2.2 秒涨到 4.0 秒 |
| 把预读与文本条件阶段重叠（`TS_H3_PREFAULT=2`） | 不如与上传流水线并行：编码器会把刚放进去的页挤掉 |
| 关掉 CUDA graphs | 没有影响：65.6 秒对 65.8 秒 |
| `TS_GGML_ASYNC_COMPUTE=1` | 没有影响：67.4 秒对 66.2 秒。它管的是图的提交，不是权重上传 |
| 用 ffmpeg 替换系统自带的 H.264 编码器 | 更慢：98.2 秒。因此保留原来的旁路编码器 |

## 数值验证

每个网络都是对着**参考实现**校验的，而不是自己跟自己比。测试基准来自上游 PyTorch
以及 stable-diffusion.cpp 自己的中间张量。

| 组件 | 结果 |
|---|---|
| 文本编码器（全部 50 层） | 对比 sd.cpp 的条件输出 cos 0.999999 |
| DiT 单步（全部 50 层） | 视频 cos 0.9983、音频 cos 0.9998 |
| 视频 VAE 解码 | cos 1.000000 |
| 视频 VAE 编码 | cos 1.000000 |
| 音频 VAE 解码 | cos 0.999995 |

英文版与完整命令见 [minimax-h3.md](minimax-h3.md)。
