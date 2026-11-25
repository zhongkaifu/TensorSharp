# MiniMax-H3-Image

[English](minimax-h3-image.md) | [中文](minimax-h3-image_zh-cn.md)

## 许可与法律声明

> **重要 —— MiniMax-H3-Image 受独立的第三方许可约束。** 在通过 TensorSharp 下载、运行、部署、修改、再分发或使用 MiniMax-H3-Image 及其输出之前，您必须阅读并遵守 [MiniMax-H3-Image Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE)、其可接受使用政策以及所有适用的法律法规。
>
> 该社区许可明确将**美国、欧盟、英国和大韩民国（韩国）**排除在许可地域之外，不授权在这些地域内的受覆盖使用。当您及您的关联方年化收入超过 100 万美元，或向第三方提供托管服务时，还需取得 MiniMax 的单独书面授权；许可也禁止用其输出改进其他 AI 模型。完整条款及其变更以官方许可为准。
>
> TensorSharp 是独立的推理引擎。它与 MiniMax-H3-Image 的技术兼容性**不**授予任何许可、不豁免任何限制，也不意味着 MiniMax 的授权或认可。**您须自行判断预期用途是否被允许，并自行承担因使用或违规而产生的一切风险、后果、索赔、处罚、损害及其他责任。在适用法律允许的最大范围内，TensorSharp 及其作者、贡献者和维护者不对您使用 MiniMax-H3-Image、其输出或任何违反其许可或适用法律的行为承担任何责任。**本声明不凌驾于 MiniMax 许可之上，也不在法律禁止的情况下限制责任。

MiniMax-H3-Image 是 MiniMax 由 MiniMax-H3 视频模型衍生的文生图模型：同一个 33B
单流 DiT（50 个块，hidden 5376，56 个 128 维头）去掉了音频流，使用同一个
Qwen3-VL-32B 文本编码器（第 50 层隐藏状态），并配一个新的图像 VAE，其 2-D ViT 解码器
读取 H3 的潜空间。它被蒸馏为 **6 步 Euler、guidance 1**：最后一层为检查点自身 sigma
网格的每一步各带一个输出头，且不支持负面提示词。适用范围约 4 MP 至 20 MP，约 16 MP
时效果最好（4096x4096、4736x3552、5024x3360、5472x3072、6272x2688）。

TensorSharp 在其共享扩散运行时（`TensorSharp.Models.SharedDiffusion`：MiniMax-H3-Image
系列的文本编码器及其视觉塔、去噪器与 VAE 编码器和解码器，由同一条流水线按阶段运行）上运行文生图与
图像编辑（最多 16 张参考图，见[图像编辑](#图像编辑)），`ggml_cuda`、`ggml_metal` 与 `ggml_cpu` 均可
端到端运行。已用真实权重在 `ggml_cuda`（H100 与 A40）和 `ggml_metal`（M5 Pro）上实测；`ggml_cpu`
运行相同的程序（已在小型合成检查点上验证；使用真实权重时每张图需要数十分钟）。编辑可以用遮罩限定在 `<Picture 1>` 的
一个选区内（见[遮罩编辑](#遮罩编辑)），聊天宿主可以跨轮次连续编辑（见[多轮对话](#多轮对话)）。

| 组件 | 文件 | 来源 |
|---|---|---|
| 扩散 Transformer（bf16，官方发布格式） | `minimax_h3_image_pruned_bf16.safetensors`，40.2 GB | [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/diffusion_models) |
| 文本编码器（与 MiniMax-H3 共用） | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf`，16.97 GiB，以及 `vocab.json`、`merges.txt`、`tokenizer_config.json` | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF)、[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/processor) |
| 图像 VAE | `minimax_h3_image_vae_fp16.safetensors`，1.17 GB | [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/vae) |

更小的去噪器（Unsloth 对同一裁剪 DiT 制作的 GGUF）搭配同一个文本编码器和 VAE 运行：各自的实测画质、速度
和配置见[量化版本](#量化版本)，TensorAgent 应用列出了其中两个（见 [TensorAgent](#tensoragent)）。

去噪器直接从 safetensors 文件按原始张量名读取，无需转换为 GGUF。包含相同张量的
GGUF（用于更小的量化）同样可以加载，依据张量识别（没有 `audio_patch_proj` 的 H3 DiT）。
这也包括 ComfyUI-GGUF 转换工具生成的 GGUF，例如
[unsloth/MiniMax-H3-Image-GGUF](https://huggingface.co/unsloth/MiniMax-H3-Image-GGUF)：
它们声明 `general.architecture = hyvid`，这个标签并不指向本模型，因此由张量决定。
ComfyUI 自己的量化重打包（`*_int8_convrot`、`*_w4a8`）以 ggml 没有对应类型的格式存储
投影，会被拒绝，并在提示中给出 bf16 文件名。

## 运行

```bash
TensorSharp.Cli --config config/minimax-h3-image.json \
  --prompt "雪夜庭院里一盏发光的红纸灯笼，一块写着“冬至”的木牌" \
  --width 4096 --height 4096 --diffusion-seed 7 --output lantern.png

TensorSharp.Server.Host --config config/minimax-h3-image.json
```

[`config/minimax-h3-image.json`](../../config/minimax-h3-image.json) 在首次使用时下载这三个文件
（约 58 GB），并固定其版本和 SHA-256。文本编码器与
[`config/minimax-h3-fl2va.json`](../../config/minimax-h3-fl2va.json) 下载的是同一个文件，两个配置共用。
文本编码器和 VAE 会在去噪器旁查找，也可通过 MiniMax-H3 的伴随参数
`--video-text-encoder` 与 `--video-vae`（`TS_VIDEO_TEXT_ENCODER`、`TS_VIDEO_VAE`）指定。

服务器在图像路由（`/api/image-generate`、`/api/image-edit` 及其 `/stream` 形式）上提供它，
与 Qwen-Image-2.1 相同；也以 SGLang 的格式提供 OpenAI 形式的 `/v1/images/generations` 与
`/v1/images/edits`（MiniMax 官方的 `examples/t2i.py` 与 `examples/edit.py` 无需修改即可使用，见
[API 示例](../../TensorSharp.Server.Host/API_EXAMPLES_zh-cn.md)）。ComfyUI 节点包为 **TensorSharp Image** 节点提供启动配置
`minimax-h3-image`（`examples/comfyui/profiles/`），以及使用 int8 投影的
`minimax-h3-image-int8`（`--diffusion-products int8`，见[性能](#性能)）。

请求参数：

- **尺寸**：未指定时为 2048x2048（ComfyUI 模板的默认值）；边长须为 32 的倍数。
- **步数**：默认 6，即蒸馏步数。其他步数也能运行（跨越多个网格区间的一步取这些区间
  输出头的 dt 加权平均，与 ComfyUI 一致），但不是检查点训练时的设置。
- **Guidance**：固定为 1。`cfg` 大于 1 或负面提示词会被拒绝。
- **种子**：初始潜变量按 ComfyUI KSampler 的方式生成（CPU `torch.randn`，
  `torch.manual_seed(seed)`），因此相同种子在两者中从相同的潜变量出发。

## 量化版本

[unsloth/MiniMax-H3-Image-GGUF](https://huggingface.co/unsloth/MiniMax-H3-Image-GGUF/tree/0d25c7349e007529691bbc0f4426b579fce7de01)
（`0d25c734`）对 Comfy-Org 裁剪后的 DiT 做了量化；GGUF 与 bf16 文件的加载方式相同，搭配同一个文本编码器
和 VAE。下列四种 DiT 各有一个配置，均固定版本与 SHA-256，其余文件共用（路径相同）：

| DiT | 文件，字节数 | 配置 | 实测可运行于 |
|---|---|---|---|
| bf16 | `minimax_h3_image_pruned_bf16.safetensors`，40,234,657,162 | [`minimax-h3-image.json`](../../config/minimax-h3-image.json) | A40 46 GB（至 4096x4096）、H100；48 GB Mac 不行 |
| Q8_0 | `minimax_h3_image_pruned-Q8_0.gguf`，22,026,760,928 | [`minimax-h3-image-q8.json`](../../config/minimax-h3-image-q8.json) | A40 46 GB、M5 Pro 48 GB；20 GB 显卡上被拒绝 |
| Q5_K_M | `minimax_h3_image_pruned-Q5_K_M.gguf`，14,599,107,296 | [`minimax-h3-image-q5km.json`](../../config/minimax-h3-image-q5km.json) | RTX A4500 20 GB（至 3840x3840） |
| Q4_K_M | `minimax_h3_image_pruned-Q4_K_M.gguf`，12,721,119,968 | [`minimax-h3-image-q4km.json`](../../config/minimax-h3-image-q4km.json) | RTX A4500 20 GB（至 4480x4480）、A40、M5 Pro |

**画质。** 相同提示词、种子和尺寸（2048x2048、6 步）下与 bf16 DiT 对比：四个提示词（人像、带中英文
文字的招牌与横幅、风景、水彩插画），8 位 RGB 图片的 PSNR 与 SSIM。

| DiT | RTX A4500（`ggml_cuda`）对 `ggml_cpu` 上的 bf16，四个提示词的平均 | A40（`ggml_cuda`）对 A40 上的 bf16：人像 / 文字 / 风景 / 插画 |
|---|---|---|
| Q8_0 | 放不进 20 GB | 27.7 / 22.9 / 28.4 / 17.6 dB |
| Q6_K | 21.84 dB / 0.840 | |
| Q5_K_M | 21.81 dB / 0.835 | |
| UD-Q4_K_XL | 19.65 dB / 0.791 | |
| Q4_K_M | 17.29 dB / 0.725 | 18.9 / 16.2 / 21.6 / 13.3 dB |
| Q3_K_M | 17.06 dB / 0.703 | |

6 步蒸馏采样器会把单步里的微小差异放大成构图的移动，所以即使是最好的量化也只在 17-28 dB：同一 CPU
后端上 Q8_0 对 bf16 为 26.7 dB，同一份 Q6_K 权重在 CUDA 与 CPU 之间也只有 29.3 dB。前几档量化之间
1-2 dB 的差异在此范围之内；降到 Q4_K_M 和 Q3_K_M 的 4-5 dB 则不是。所有图片都连贯且符合提示词。Q8_0
在四张图上都保持了 bf16 的构图；Q5_K_M 和 Q6_K 的人像几乎与 bf16 相同，横幅文字干净；Q4_K_M 会改变
面孔和物体，并在文字提示词中多出一块文字错乱的招牌；Q3_K_M 把横幅缩小，并多出一行错乱的文字。

**20 GB 显卡上的速度与内存。** 一张 RTX A4500（`ggml_cuda`，sm_86），2048x2048，6 步，运行时默认设置
（在这张卡上去噪前会释放文本编码器）。每步为三个全新进程中第 2-6 步的中位数；CLI 一张图为整个进程的时间；
服务端热态一张图为第二、三个请求（每次都要重新上传 DiT，它放不进文本编码器旁边）；峰值为整个进程的峰值，
含 CUDA 上下文。

| DiT | 每步 | CLI 一张图 | 服务端热态一张图 | 峰值 | 可运行的最大正方形 |
|---|---:|---:|---:|---:|---|
| Q3_K_M | 3.68 s | 30.4 s | 29.5 s（共享主机负载偏高时） | 15.2 GiB | 4480x4480 |
| Q4_K_M | 3.32 s | 28.6 s | 24.8 s | 15.2 GiB | 4480x4480（18.9 GiB） |
| UD-Q4_K_XL | 3.66 s | 31.5 s | 26.4 s | 15.6 GiB | 4096x4096 |
| Q5_K_M | 3.88 s | 32.0 s | 27.6 s | 16.6 GiB | 3840x3840（19.0 GiB） |
| Q6_K | 4.22 s | 34.5 s | 30.2 s | 17.4 GiB | 3328x3328（19.2 GiB） |
| Q8_0 | 被拒绝 | | | | 无：仅 DiT 就有 22.0 GB |

相同种子下每张图都与早先的原生路径逐位一致；显卡放不下的尺寸会在任何阶段运行之前被拒绝（见[内存](#内存)）。
A40 和 M5 Pro 的耗时见[性能](#性能)。

**如何选择。** 20-24 GB 显卡：Q5_K_M（放得下的最接近 bf16 的一档，文字干净），需要 16-20 MP 输出时用
Q4_K_M。32 GB Mac：Q4_K_M（最大的阶段是文本编码器，连同工作内存 18.9 GB；Q8_0 的去噪超出 32 GB Mac
的 GPU 工作集）。46-48 GB 显卡或 48 GB Mac：Q8_0。80 GB 及以上：bf16。

## 图像编辑

```bash
TensorSharp.Cli --config config/minimax-h3-image.json \
  --image studio.png --image poster.png --width 2048 --height 2048 --diffusion-seed 1234 \
  --prompt "Hang <Picture 2> as a framed print on the wall behind the two hosts of <Picture 1>." \
  --output edit.png
```

一次编辑最多接受 16 张参考图（重复 `--image`；服务器上为 `imagePaths` 或各个 `image` 部分）。提示词按给出的顺序
称它们为 `<Picture 1>`、`<Picture 2>`……，并原样使用。TensorSharp 遵循 MiniMax 官方流程（diffusers main d5df05ed
的实现），每张参考图都以两种方式参与条件：

- **副本。** 每张图按 EXIF 方向摆正后以 RGB 读取；某边小于 32 像素会被拒绝。*VAE 副本*在短边为 1024 至 2048
  像素时保持原图，否则缩放到较近的边界，两边都取 32 的倍数；*视觉副本*缩放到短边恰为 1024，再按 Qwen3-VL
  处理器的规则对齐（边长为 32 的倍数）。两者都使用 8 位的 Pillow Lanczos（处理器自身的一步用 Pillow 双三次），
  逐字节一致。参考图从不按输出尺寸缩放。
- **文本编码器。** 每张图依次为标签 `Picture i: `、其视觉块（`<|vision_start|>`、每个合并后的 32x32 块一个
  `<|image_pad|>`、`<|vision_end|>`），最后是提示词；标签与提示词的行标为文本，视觉块标为图像。视觉塔的输出替换
  这些占位 token，其 DeepStack 分支在前三层之后注入。1024x1024 的视觉副本有 1,024 行，多张图会使序列很长，
  超过约 2,000 行时，主干按查询行分块计算注意力分数（分数相同，每块不超过 1 GiB）。
- **VAE 编码器。** VAE 副本（裁到完整的 16 像素格）经图像 VAE 的编码器（单帧的 H3 因果 3-D CNN），以 256 像素、
  重叠至少 64 像素的分块处理；各块的矩按 diffusers 的方式融合成后验。从后验采样（舍入到 float16、裁到完整的 2x2
  块、归一化）得到参考图的潜变量，加噪到 t = 0.999 后在每一步保持不变。
- **打包序列。** `[文本 | 参考图 | 目标图]`：第 k 张参考图在时间轴上位于文本之后的第 k 个位置，使用其自身按宽高比
  归一化的网格；目标图位于最后一张参考图之后。文本与目标图的时间步为 t = 1 - sigma，参考图为 0.999。
- **种子。** 每个请求一个 CPU torch 生成器，按 MiniMax 的顺序依次抽取：每张参考图的后验采样、目标图的噪声、
  每张参考图的噪声。因此编辑从不复用源图自身的噪声。
- **尺寸。** `--width`/`--height`（32 的倍数，最多 2000 万像素）；`--keep-source-size` 返回 `<Picture 1>` 的精确尺寸；
  否则按 `<Picture 1>` 的宽高比取请求的面积（服务器默认 2048x2048 的面积），CLI 上则取其 VAE 副本的尺寸。小于
  2048x2048 的编辑在同宽高比的更大画布上采样（diffusers 的 `min_generation_area`），再用 Lanczos 缩放回来；
  加速插件自带的下限会取代它（pDMD 为 1024 像素：其编辑按请求的尺寸采样，与其 diffusers 流水线一致）。
  文生图保持自己的尺寸规则（没有下限）；遮罩编辑的选区另有规则（见下文）。
- **拒绝。** `cfg` 大于 1、负面提示词、超过 16 张参考图、某边小于 32 像素；以及在任何组件运行之前，参考图使文本编码或
  去噪一步所需显存超过设备空闲显存的编辑（消息给出行数；减少参考图或使用更小的参考图即可）。这是每个请求（包括文生图）
  都经过的同一项检查：一步只要放得进空闲显存就会运行，即使规划显示它占用了部分预留余量。

每张参考图都会增加两处行数：文本中约 1,024 行的视觉块，以及 DiT 中 (VAE 副本 / 32)² 行（1024x1024 副本为 1,024 行，
2048x2048 为 4,096 行）。16 张 1024x1024 参考图加 2048x2048 的目标图约为 37,000 行 DiT，是文生图一步的九倍；
注意力随行数平方增长。

### 遮罩编辑

```bash
TensorSharp.Cli --config config/minimax-h3-image.json \
  --image shop.png --mask banner.png --keep-source-size --diffusion-seed 42 \
  --prompt "In <Picture 1>, change the painted banner above the door so it reads \"SPRING EQUINOX CAFE\"." \
  --output shop-edit.png
```

遮罩把编辑限定在被编辑图片 `<Picture 1>` 的一个选区内；其他图片仍是参考图。MiniMax 的官方流程没有遮罩（其 README：
"not yet supported: mask-based inpainting"），因此这是 TensorSharp 自己的方法，与其 Qwen-Image 路径相同，参数与字段也相同：

- **遮罩**与 `<Picture 1>` 尺寸完全相同。`--mask-mode grayscale`（默认）：白色编辑、黑色保留、灰色按比例混合；`alpha`：
  透明编辑、不透明保留。`--mask-invert` 反转二者，`--mask-feather N` 把边缘向内柔化 N 个源像素（不会越出选区），
  `--mask-crop` 与 `--mask-crop-padding N`（默认 64）只采样选区的外接框及其周围这么多的上下文。服务器接受 `maskPath`
  （一个上传文件）、`maskMode`、`maskInvert`、`maskFeather`、`maskCrop` 与 `maskCropPadding`，或 multipart `/api/image-edit`
  的 `mask` 文件；OpenAI 的 `/v1/images/edits` 接受 `mask` 部分，按 [OpenAI 的约定](../../TensorSharp.Server.Host/API_EXAMPLES_zh-cn.md)读取。
- **尺寸。** 结果始终是 `<Picture 1>` 自己的画布。整张图（或裁剪的外接框）按不带遮罩的同一编辑所用画布的密度采样（上述规则），
  但绝不超过其自身分辨率，编辑的下限也不会抬高它：整张图的潜变量遮罩、去噪与解码至多与要合成进去的像素一样大，并向上取整到
  32 像素网格。比该画布更大的图片仍以低于其自身分辨率的尺寸采样，解码后用 Lanczos 缩放回来。（此前这里同样应用 2048x2048 的
  下限：512x512 图片的遮罩编辑会以 2048x2048 采样，是其像素的 16 倍。）
- **裁剪的下限。** 小于 `<Picture 1>` 的裁剪按其自身宽高比至少以 2048x2048 像素（编辑自身的下限）采样，但不超过整张图像素数的
  4 倍；加速插件的下限不改变它。大于 4 MP 的裁剪按其自身尺寸采样；外延边距覆盖到所有边缘的裁剪就是整张图，按其自身尺寸采样。
  1024x1024 图片的 448x512 裁剪以 1920x2176 采样，与加上上限之前相同。此前在 A40 上试过更小的下限（Q4_K_M，在这个裁剪里画红色花瓶，
  种子 7、11、23）：

  | 裁剪的采样尺寸 | 总耗时 | 每步 | 峰值 | 花瓶 |
  |---|---|---|---|---|
  | 自身尺寸 448x512 | 16.9–17.5 s | 1.71 s | 32.5 GB | 平淡；一个种子画出了多余的白色形状 |
  | 1 MP，960x1088 | 19.9–20.4 s | 2.1 s | 32.8 GB | 干净，但平淡 |
  | 4 MP，1920x2176（现在的下限） | 33.5–34.6 s | 3.9 s | 33.8 GB | 两个种子画出了浮雕或彩绘纹样 |

  4 MP 下三张 PNG 与加上上限之前逐字节相同。峰值主要来自 32B 文本编码器。小图片的裁剪也会提高到 4 MP：512x512 图片的一个裁剪
  以 1472x2848 采样，是整张图像素的 16 倍，耗时 42.9 s（每步 5.1 s，35.0 GB），对比 1 MP 下的 25.3 s（352x704，每步 2.7 s）。
  这个种子在 4 MP 下基本保留了原图的花瓶，只加上了一个瓶颈；1 MP 下则画出了整个瓶子。
- **潜变量。** 遮罩按每个 16x16 格中的最大权重映射到潜变量网格，因此细笔画仍可编辑。`<Picture 1>`（或其裁剪）在采样画布上
  编码为后验均值，并与参考图一样归一化（参考图副本恰为该尺寸时直接复用，否则多一次 VAE 编码）。每一步之后，保留的格子
  被重置为该潜变量按下一个 sigma、用请求自身的噪声加噪后的值，因此最后一步恰好落在它上面：被编辑的区域在去噪时能看到
  周围，周围保持图片自己的像素。`<Picture 1>` 仍是参考图，模型能看到它需要融合的内容。
- **合成。** 解码结果缩放到选区后按遮罩权重混入 `<Picture 1>`：权重为 0 的每个像素都是源图自己的字节（包括 alpha），权重为 1
  的像素来自编辑结果。空遮罩不运行模型，直接返回 `<Picture 1>`；全选遮罩等同于在源画布上的无遮罩编辑。
- **与 pDMD 一起使用。** 遮罩与 pDMD 插件可以组合：它的两步与六步一样在每步之后重置保留的格子，合成方式也相同，
  因此保留的像素仍是源图自己的字节；选区按上面的规则采样（至多为其自身分辨率，裁剪至少为裁剪的下限）。在加上这一上限之前，以插件的下限在 M5 Pro 上验证过（Q8_0，用方框遮罩把杯子换成玻璃茶壶）：
  2,960,883 个保留像素无一不同；每步 34.9 s（六步遮罩编辑为 33.6 s），整次编辑 99 s。

开销与无遮罩编辑相同；当 `<Picture 1>` 的参考副本不是采样画布时（2048x2048 的图片是，1024x1024 的不是），多一次 VAE 编码。
选区按画出的样子保留：围绕物体的矩形取反后会保留整个矩形（包括背景），与新的周围环境之间形成硬边；沿物体轮廓画出的选区或羽化
可以让它融合。

M5 Pro 48 GB，`ggml_metal`，Q8_0，2048x2048 图片，一个进程，6 步（每步为第 2 至 6 步的中位数；进程中第一次编辑为冷启动，其余为热）：

| 遮罩编辑 | 选区 | 文本 | 视觉 | VAE 编码 | 每步 | 解码 | 编辑 | 峰值（已分配） |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 改写招牌（"SPRING EQUINOX CAFE"） | 横幅，10.0% | 3.81 s | 1.24 s | 16.94 s（121） | 33.6 s | 2.86 s | 232.2 s | 30.6 GB |
| 把杯子换成玻璃茶壶 | 杯子外接框，29.4% | 6.44 s | 1.15 s | 16.95 s（121） | 33.6 s | 2.71 s | 229.9 s | 30.6 GB |
| 取反：杯子外接框以外的一切 | 70.6% | 6.27 s | 1.16 s | 16.93 s（121） | 33.6 s | 2.69 s | 229.6 s | 30.6 GB |

同一张图不带遮罩的开销相同（S5 的文字编辑：每步 33.8 s，231.0 s，30.7 GB）：参考副本就是采样画布，不会多编码任何内容。
每个未选中的像素都是源图自己的字节（每张图 2,960,883 至 3,774,333 个，没有一个不同）。

### 多轮对话

TensorAgent 的聊天（其 `/api/chat` 轮次）会把每条消息与之前的图片对照理解，与 Qwen-Image-2.1 相同；服务器的 Web UI 则把每条消息
原样发给图像接口（附带的照片就是要编辑的图，"Edit again" 会恢复该轮的照片、选区与文字）。在 TensorAgent 中：

- 没有附图的后续消息修改聊天中最新的图片。Qwen-Image 会用自身的语言模型判断后续消息指的是哪张图；MiniMax-H3-Image 的编码器
  没有输出头来回答这个问题，因此该轮修改最新的一张，页面会说明这是推测（另一种理解，即生成新图，一键即可）。
- 消息附带的照片就是它要编辑的（第一张为 `<Picture 1>`，其余为参考图，选区属于第一张）。用户指明要修改聊天中较早的某张图
  （在页面上选择的理解）的消息，会以附带的照片作为参考图编辑那张图：`[图片, ...附图]`，选区属于那张图。
- 每次编辑都保持源图尺寸，每轮请求模型自身的 2048x2048 面积（TensorAgent 的另一个图像模型 Qwen-Image-2.1 仍为 1024x1024）。
- 页面通过 `/api/models` 的 `image` 块（`family`、`supportsEdit`、`supportsMask`、`minimumArea`）识别图像模型，因此编辑区域
  （Edit again、Select area、多张附图）对 MiniMax-H3-Image 与对 Qwen-Image 一样可用。

## 运行方式

- **阶段。** 每张图分三个阶段运行：文本（编码器）、去噪（DiT 的 6 步）与解码（VAE）。各组件
  放得下时，流水线在阶段之间与多张图之间都把它们保留在设备上；放不下时在阶段之间释放其中一个
  （见[内存](#内存)）。
- **提示词。** 原始提示词 token，不套聊天模板，经过 MiniMax-H3 的 Qwen3-VL 主干（即 MiniMax-H3
  视频所用的原生编码器）；条件是第 50 层的隐藏状态。
- **打包序列。** `[文本 | 目标图像]`，使用 H3 的三轴 RoPE：第 i 个文本 token 位于 (i, 0, 0)，
  图像的 2x2 patch 在时间轴上紧随其后，空间网格按宽高比归一化。文本与图像都在
  t = 1 - sigma 上运行，没有 flow shift。
- **调度。** ComfyUI 模板在检查点网格上使用 KSampler 的 `simple` 调度器
  （`MiniMaxH3ImageModelSampling`）；TensorSharp 以 float32 复现这些 sigma
  （6 步时为 1、0.96765、0.88771、0.75、0.53364、0.23261、0）。
- **PDD 输出头。** `final_layer.video_out` 叠放了 6 个头；从网格点 k 到 k+1 的一步使用
  头 0 加头 k（第一步只用头 0）。组合后的头按区间构建一次，作为该步的输入传入，因此 6 步共用
  同一个已编译的步骤。
- **DiT。** 每步一个计算图程序：token refiner、50 个块与最后一层，不含音频流。在 `ggml_cuda` 上，
  调制、门控残差、注意力（cuDNN）与 BF16 的 SwiGLU-in-GEMM 以融合阶段运行，宽矩阵乘走 cuBLAS
  （或 int8）；其他后端用 ggml 自身的内核运行同一程序。该程序与 MiniMax-H3 原生 DiT 构建的计算图
  逐节点相同，因此速度场逐位一致。
- **VAE。** 图像解码器（24 个 1024 宽的块、16 个头、在每个头的一半维度上对 (h, w) 做 RoPE、
  16x16 RGB patch）在至少重叠 64 像素的 256 像素瓦片上解码，并完全按 ComfyUI 的
  `MiniMaxH3ImageVAE` 方式融合进画布。一行瓦片（至多 64 个）是一个计算图程序；在 `ggml_cuda` 上
  这一行的注意力是一次覆盖整行的 cuDNN 调用。任何一行在所有后端上都能解码：解码器的计算图按程序
  的节点数分配，而先前的原生解码在 `ggml_metal` 与 `ggml_cpu` 上每次调用超过 4 个瓦片就会中止
  （`GGML_ASSERT(cgraph->n_nodes < cgraph->size)`，2048x2048 及以上的图都会触发）。
  `TS_H3_IMAGE_VAE_BATCH` 是那个解码器的变通办法，已随它一起移除（阶段 S10）；设置了它，主机会拒绝启动。
- **唯一路径。** 运行时是 MiniMax-H3-Image 的唯一路径：先前的原生模型（`MiniMaxH3ImageModel`、它的解码器
  以及每步一个计算图的 DiT 前向）在阶段 S10 删除之前一直是测试的参照。它在合成测试数据上的输出已按后端冻结
  （`InferenceWeb.Tests/Fixtures/MiniMaxH3Image/LegacyGoldens/`），运行时以此为准：在 `ggml_metal`（M5 Pro）与
  `ggml_cuda`（RTX PRO 4500）上逐字节一致，其他环境在舍入误差范围内。在 ARM64 的 `ggml_cpu` 上，运行时采用 main
  引入的 CPU 乘法（为 i8mm 与点积内核重排的权重、以未舍入激活计算的 BF16 乘法），先前的模型没有采用：在测试数据上
  与其相差至多最大值的 2.5e-3（Q4_K），属舍入差异。
- **乘法精度。** `--diffusion-products bf16|int8|int8-all`（配置键 `diffusion-products`；CLI 与服务器）选择
  `ggml_cuda` 上宽矩阵乘的算术：`bf16`（默认）按原样运行（BF16 与 Q4_K 走 cuBLAS BF16，其他量化走 ggml 内核）；
  `int8` 让 DiT 的投影在 int8 Tensor Core 上运行（见下文）；`int8-all` 再加上 VAE 解码器的 F16 投影。它取代
  `TS_H3_INT8_MATMUL`（`1` 即 `int8`，`all` 即 `int8-all`），主机现在会按名称拒绝该旧变量。其他后端照旧按原样运行，
  主机会提示一次。

## 内存

流水线按设备预算为每张图做规划：`ggml_cuda` 上为空闲显存减去余量（显卡容量的 1/16，至少
512 MiB）；`ggml_metal` 上为 GPU 推荐工作集减去已分配量，再减去同样的余量（统一内存上，超出工作集的
命令缓冲会直接失败）。文本编码器、去噪器、VAE 及其工作内存放得下时，所有组件在阶段之间与多张图之间
都保持驻留；放不下时，在下一阶段开始前释放它不用的组件：丢弃其设备副本，权重仍映射在主机上，之后的
阶段再从内存或页缓存重新上传。`ggml_cpu` 保留全部组件。决策按尺寸打印一次，例如：

```
[diffusion-residency] 2048x2048, 0 reference(s): before Denoise: release the conditioner (peak 24.7 GB, budget 37.7 GB)
```

实测（一张图期间设备的峰值占用，含 CUDA 上下文；Metal 上为 GPU 已分配量与系统的 wired 内存）：

| 设备 | DiT | 尺寸 | 规划 | 峰值 |
|---|---|---|---|---:|
| A40 46 GB | Q4_K_M | 2048x2048 | 全部驻留 | 30.0 GiB |
| A40 46 GB | Q4_K_M | 4096x4096 | 全部驻留 | 33.9 GiB |
| A40 46 GB | bf16 | 2048x2048 | 去噪前释放文本编码器（0.8 s 内归还 15.6 GB） | 39.9 GiB |
| A40 46 GB | bf16 | 4096x4096 | 同上，并在解码前释放 DiT 的工作区 | 42.6 GiB |
| M5 Pro 48 GB | Q8_0 | 2048x2048 | 去噪前释放文本编码器；从第二张图起，文本阶段前还释放 DiT | 已分配 24.3 GiB，wired 29.9 GB |
| M5 Pro 48 GB | Q4_K_M | 512x512 | 全部驻留 | wired 33.1 GB |
| RTX A4500 20 GB | Q4_K_M | 2048x2048 | 去噪前释放文本编码器 | 15.2 GiB |
| RTX A4500 20 GB | Q4_K_M | 4480x4480 | 同上，并在解码前释放 DiT 的工作区 | 18.9 GiB |
| RTX A4500 20 GB | Q6_K | 3328x3328 | 同上 | 19.2 GiB |

若不做这些释放，bf16 在 A40 上与文本编码器放不下（约 60 GB），Q8_0 在 48 GB 的 Mac 上也放不下：
先前的路径保留文本编码器，在加载 DiT 时于 wired 43.5 GB 处被 wired 内存看门狗终止。A40 上的 bf16
每张后续图片都要重新上传文本编码器与 DiT（在该虚拟机上从磁盘读取约 4 s 与 15 s，其页缓存放不下两者）。
bf16 在 48 GB 的 Mac 上完全放不下（40.2 GB 权重对 40.2 GB 工作集）：请在那里使用 Q8_0。H100 级别的
显卡可保留全部组件，16 MP 的 bf16 也不例外。

设备放不下的图片会在任何阶段运行之前被拒绝：即使释放其他所有组件，其最耗显存的阶段（该组件的权重加上
工作内存）所需仍超过设备的空闲显存时，CLI 以退出码 `2` 结束并输出 `error: request refused: <原因>`，
服务端返回 `507`，模型保持加载，继续处理下一张图。否则该阶段会在 ggml-cuda 内核内部耗尽显存并结束进程。
在 RTX A4500（20 GiB，扣除 CUDA 上下文后空闲 20.8 GB）上能运行的最大正方形尺寸：Q3_K_M 与 Q4_K_M 为 4480x4480（模型上限），
UD-Q4_K_XL 为 4096x4096，Q5_K_M 为 3840x3840，Q6_K 为 3328x3328；超出这些的下一档实测尺寸
（UD-Q4_K_XL 4480、Q5_K_M 4096、Q6_K 3584）此前在去噪阶段内部失败，现在会被拒绝。Q8_0 在该卡上任何
尺寸都会被拒绝：仅其 DiT 就有 22.0 GB。

## LoRA 插件

MiniMax-H3-Image 通过各宿主的 `--lora`、`--lora-scale` 与 `--lora-config`（用法见
[USAGE](../../USAGE_zh-cn.md#minimax-h3-image-lora-插件)）以及 `TS_LORAS` 接受
`minimax-h3-image-lora` 类型的 LoRA 适配器：任意数量的风格适配器（各有自己的强度），外加至多一个
加速适配器，其配置固定采样方式。

| 插件 | 来源 | 类型 | 命名 | 秩 / alpha |
|---|---|---|---|---|
| [`minimax-h3-image-pdmd-2step.json`](../../config/lora/minimax-h3-image-pdmd-2step.json) | linoyts/MiniMax-H3-Image-pDMD-2steps @c07d96e5 | 加速：在 `[1, 12/13, 0]` 上 2 步，`by-step` 输出头 | diffusers PEFT | 128 / 128 |
| [`minimax-h3-image-pdmd-2step-comfy.json`](../../config/lora/minimax-h3-image-pdmd-2step-comfy.json) | akhaliq/minimax-h3-image-pdmd-2step @ef1e3b9a（ComfyUI 转换版） | 加速：同样的 sigma，`grid-interval` 输出头 | ComfyUI 原生，块对角融合 qkv | qkv 为 384 / 384，其余 128 / 128 |
| [`minimax-h3-image-seedreamstyle.json`](../../config/lora/minimax-h3-image-seedreamstyle.json) | Nebsh/Seedreamstyle-H3-6K @5e1b48cd | 风格（触发词 `seedreamstyle.`） | kohya | 16 / 16 |

两个 pDMD 文件是两次不同的训练：linoyts 的与 akhaliq（pdmd2026）的因子互不相关。三种命名格式之一的
裸 `.safetensors` 按风格适配器处理。

**更新的应用方式。** 不合并：在适配器所指的投影上计算 `y = W x + up (down x)`，基础权重保持原样存储。
diffusers 的 `to_q`、`to_k`、`to_v` 落在 `attn.qkv_proj` 堆叠的三等分上，其 `ff.net.0.proj` 的 `lora_B`
两半从 `[value; gate]` 交换为检查点的 `[gate; value]`；ComfyUI 与 kohya 的命名就是检查点本身的布局
（块对角的融合 qkv 项会精确拆分到三等分上）。强度 × alpha / rank 以 F32 折入 `up`；每个秩分量平衡后
以 F16 存储（`ggml_cpu` 上为 F32）。同一投影上的多个适配器沿秩拼接，因此无论有几个，每个槽只做一次
收缩与一次扩张。在 `ggml_cuda` 上，宽乘积（Q4_K、Q6_K 或 BF16 权重，1,024 行及以上）在自己的 cuBLAS
GEMM 内完成更新：`[W | up] · [x ; down x]`，深度 K + R（BF16 权重在 K < 2 × 行数时复制进这些行，
否则各部分的扩张累加到它的乘积上）。在 `ggml_metal` 上，Q8_0、F16 或 F32 权重的宽乘积用 ggml 自己的算子
以同样的形式构建（每步把权重与 up 拼接，权重本身不被修改），前提是这样搬运的字节比在乘积旁边加上更新更少：
4 MP 时为 qkv、out 与 fc1，fc2 不在此列。其他情况在 ggml 上运行该区域的乘积与加法。请求换用另一组适配器时，会先丢弃上一组的已编译步骤与设备副本。

**pDMD 的输出头。** by-step 输出头在第 1 步使用 head 0，第 2 步使用 head 0 加 head 1：TensorSharp 在单个
网格区间 `[i, i + 1)` 上组合出的输出头与 diffusers 的 `pdd_proj_out[i]`（MiniMaxAI 的
`pdd_fused_heads.safetensors`）一致，相对误差 3.8e-10，六个头、每一行都检查过。

**验证**（`eng/validation/MiniMaxH3ImageLoraRows`、`eng/validation/MiniMaxH3ImageAB`）：

- *与独立合并逐行比对。* pDMD（linoyts）、akhaliq 的 pDMD（diffusers 与 ComfyUI 两个文件）、
  Seedreamstyle（kohya）、waryusya（ComfyUI 命名，无 alpha）与 domokun（ComfyUI，带 `.alpha` 的融合 qkv）：
  第 0、25、49 块及精炼器第 0 块中适配器涉及的每个投影都按后端的实际方式运行，并在抽样行上与 NumPy 的
  `(W + scale · B A) x` 比较；适配器通过 HTTP 范围请求从 Hub 读取，按已发布的转换规则映射，而不是用
  TensorSharp 的代码。`y` 的相对误差在 A40 的 Q4_K_M 上为 3.6e-3，bf16 上为 2.1e-3，M5 Pro 的 Q8_0
  （`ggml_metal`，在乘积旁边时为 1,024 行，在乘积内时为 4,096 行）上为 6.4e-4，即基础乘积自身的舍入；
  单看更新（同一区域用全零 up 运行后相减）分别为 5.8e-3、5.7e-3，以及在乘积旁边 1.3e-3、在乘积内 5.4e-3
  （其 up 舍入为 Q8_0）。akhaliq 的两个文件给出逐位相同的更新，NumPy 与
  TensorSharp 中都是如此。
- *没有适配器就没有变化。* 2048x2048 的肖像与支持 LoRA 之前逐位相同：A40 上的 Q4_K_M 与 bf16、M5 Pro
  上的 Q8_0。
- *每步开销*（2048x2048，热图片中除每张图片第一步之外所有步的中位数；A40 上 bf16 图片的第一步会重新
  上传权重）：

  | 每步，2048x2048 | 无适配器 | pDMD（rank 128） | Seedreamstyle（rank 16） |
  |---|---:|---:|---:|
  | A40，Q4_K_M | 2.188 s | 2.087 s（−4.6%） | 2.045 s（−6.5%） |
  | A40，bf16 | 1.907 s | 2.104 s（+10.3%） | 2.062 s（+8.1%） |
  | M5 Pro，Q8_0 | 12.53 s | 12.94 s（+3.3%） | 12.65 s（+0.9%） |

  在 Q4_K_M 上，带适配器的一步比不带的更快：K 拼接 GEMM 同时把 Q4_K_M 文件中保留为 Q6_K 的投影（一半块的
  qkv 与 fc2）放到 cuBLAS BF16 上，而普通路径用 ggml 自己的内核运行它们；同一路径上 rank 128 比 rank 16
  多 2.1%。在 bf16 上开销几乎与秩无关：它来自每次把权重复制进 K + R 行（每个投影 4 × K × N 字节，每步约
  71 GB），量化权重的反量化本来就要付出这笔开销。把复制放到第二个流上、以及先写扩张再以 beta 1 加上乘积，
  两种做法都更慢（pDMD 分别为 2.149 s 与 2.166 s）；要消除它，需要一个从各自指针读取 LoRA 列的 GEMM。这是已知的不足，留待后续处理。
  在 M5 Pro 上，乘积旁边的算子让 pDMD 多花 10.8%、Seedreamstyle 多花 8.2%；把更新放进乘积内后，两者同时使用
  多花 5.7%。
- *pDMD 端到端。* A40 Q4_K_M 上一张热的 2048x2048 图片用时 5.1 s，对比 14.1 s（2.8×），进程的第一张图片
  13.0 s 对比 21.0 s；bf16 上 24.5 s 对比 34.4 s（1.4×：46 GB 的 A40 每张图片都会重新上传 bf16 权重）；
  M5 Pro 上 30.8 s 对比 79.8 s（2.6×）。
- *图片。* 验证集的 8 个提示词，2048x2048，pDMD 分别在 A40（Q4_K_M、bf16）与 M5 Pro（Q8_0）上，与 6 步
  图片对比：主体与构图相同，清晰干净；小字受损（茶馆的英文招牌变成乱码，而「冬至」与时刻表保持正确），
  少数细节在 6 步图片正确的地方出错（产品图的杯子多了一个把手，野餐桌下的狗变模糊）。diffusers 自己的流水线（`diffusers_pdmd.py`：
  官方 bf16 transformer 并把 LoRA 融合进权重，输入 TensorSharp 的文本嵌入、相同的种子与噪声）画出同一张肖像，
  姿势、帽子与窗户都相同，细节有差别：与 TensorSharp 的 bf16 图片 PSNR 为 17.2 dB，而 6 步图片与它相差 14.0 dB。
  ComfyUI 转换版本（akhaliq，pdmd2026 的训练）主体画得不错，但会在皮肤与平坦区域
  叠上一层细密的网纹：无论用它随附的网格区间输出头还是 by-step 输出头都如此（两者都在 M5 Pro 上试过）；它自己卡片上的
  图片出自 Pruna 的 P-H3-Image（4B 文本编码器），而不是官方模型。它的更新与其 diffusers 版本逐位相同，因此这是该适配器
  在这套模型上的表现，而不是加载问题。生成图片请使用 `minimax-h3-image-pdmd-2step.json`。

## 性能

### A40 与 M5 Pro

一张 A40 46 GB（`ggml_cuda`，sm_86，共享虚拟机）与一台 M5 Pro 48 GB（`ggml_metal`），验证集的提示词
（40 至 160 个 token），6 步。“文本”为编码；“每步”为第 2 至 6 步的中位数；“整张图”为整个请求。进程中的
第一张图冷读文件（A40 上事先把文件逐出页缓存）并包含上传；之后的图为热启动。

| A40，2048x2048 | 文本 | 每步 | 解码 | 整张图 |
|---|---:|---:|---:|---:|
| Q4_K_M，第一张 | 3.25 s | 2.23 s | 1.22 s | 21.0 s |
| Q4_K_M，后续 | 0.08 s | 2.24 s | 0.69 s | 14.3 s |
| Q4_K_M，`--diffusion-products int8`，后续 | 0.08 s | 1.88 s | 0.68 s | 12.1 s |
| Q8_0，第一张（早先的原生路径，释放文本编码器；运行时生成的图片相同） | 3.6-4.0 s | 2.39 s | 1.18-1.21 s | 24.1-24.8 s |
| bf16，第一张（去噪前释放文本编码器） | 3.07 s | 1.92 s | 1.20 s | 29.7 s |
| bf16，后续（每次都从磁盘重新加载两者） | 4.27 s | 1.91 s | 0.73 s | 34.2 s |

| A40，4096x4096 | 文本 | 每步 | 解码 | 整张图 |
|---|---:|---:|---:|---:|
| Q4_K_M，第一张 | 3.25 s | 11.42 s | 2.91 s | 77.8 s |
| Q4_K_M，后续 | 0.07 s | 11.47 s | 2.40 s | 71.5 s |
| Q4_K_M，`--diffusion-products int8`，后续 | 0.07 s | 9.96 s | 2.48 s | 62.5 s |
| bf16，第一张 | 3.36 s | 10.36 s | 3.09 s | 82.6 s |

| M5 Pro，2048x2048 | 文本 | 每步 | 解码 | 整张图 |
|---|---:|---:|---:|---:|
| Q8_0，第一张 | 1.71 s | 12.82 s | 3.45 s | 82.6 s |
| Q8_0，后续 | 1.70 s | 12.67 s | 2.72 s | 81.1 s |

在同一台 Mac 上，TensorAgent 宿主（`eng/validation/TensorAgentHost`，即应用的 `AgentAppHost`）用 Q4_K_M 条目、
所有组件保持驻留，根据聊天消息生成：一张 2048x2048 图片（应用每轮的面积）用时 89.1 s（文本 3.08 s，每步
13.64 s，解码 3.93 s）；启用 pDMD 插件后，一张 1024x1024 图片用时 7.8 s（2 步，每步 3.47 s），对它的遮罩编辑用时
30.6 s（2 步，每步 10.6 s；未选中的像素全部保持不变）。峰值 39.2 GB wired，进程峰值 footprint 4.38 GB。

在相同机器上与先前的原生路径交替进程对比：没有任何阶段慢超过 1.4%，多数持平或更快（解码最多快 13%；
第一张图的文本阶段快 11% 至 37%，因为编码器文件现在在该阶段开始前就已映射），设备峰值内存相差不超过
0.5%（16 MP 的 bf16 低 1.3%），并且每张图都逐位相同：A40 上 2048x2048 的 Q4_K_M、int8 乘法的 Q4_K_M
与 bf16 以及 M5 Pro 上的 Q8_0 各 8 个提示词，4096x4096 的 Q4_K_M 与 int8 各 2 个，4096x4096 的 bf16
1 个。在 M5 Pro 上，先前的路径必须释放文本编码器并把解码器限制为每次调用 4 个瓦片才能运行。

### 图像编辑

一张 A40 46 GB（`ggml_cuda`）与一台 M5 Pro 48 GB（`ggml_metal`，Q8_0），2048x2048 编辑，6 步，每次编辑一个进程、冷读文件
（A40 上先从页缓存中驱逐），因此这些是首次编辑的数据。文本为不含视觉塔的编码；视觉为视觉塔处理所有图片；VAE 编码为所有
参考图的分块；每步为第 2 至 6 步的中位数。1、4、16 张参考图均为 1024x1024 图片（每张 1,024 行 DiT 与约 1,024 行文本）；
文字编辑使用一张 2048x2048 参考图（4,096 行），组合使用两张（或两张 1024x1024），MiniMax 示例为 `examples/edit.py`
的图片与提示词，输出 2048x2048。

| A40，Q4_K_M | 行数（文本 + 参考图 + 目标） | 文本 | 视觉 | VAE 编码 | 每步 | 解码 | 编辑 | 峰值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 张参考图 | 1,076 + 1,024 + 4,096 | 3.19 s | 1.12 s | 0.93 s（25 块） | 3.44 s | 1.15 s | 33.0 s | 32.8 GiB |
| 4 张参考图 | 4,189 + 4,096 + 4,096 | 8.47 s | 1.98 s | 2.63 s（100） | 9.35 s | 1.13 s | 74.2 s | 36.1 GiB |
| 16 张参考图 | 16,635 + 16,384 + 4,096 | 38.88 s | 6.15 s | 9.91 s（400） | 49.15 s | 1.54 s | 356.3 s | 42.5 GiB |
| 文字编辑（1 张 2048x2048） | 1,091 + 4,096 + 4,096 | 4.07 s | 0.93 s | 3.05 s（121） | 5.48 s | 1.24 s | 46.0 s | 33.7 GiB |
| 组合（2 张 2048x2048） | 2,125 + 8,192 + 4,096 | 5.44 s | 1.53 s | 6.60 s（242） | 9.62 s | 1.19 s | 76.3 s | 36.1 GiB |
| 组合（2 张 1024x1024） | 2,125 + 2,048 + 4,096 | 5.63 s | 1.30 s | 1.38 s（50） | 4.87 s | 1.15 s | 42.5 s | 34.3 GiB |
| MiniMax 示例（1 张 1024x1024） | 1,459 + 1,024 + 4,096 | 4.82 s | 0.98 s | 0.80 s（25） | 3.70 s | 1.09 s | 33.6 s | 33.2 GiB |

| A40，bf16 | 行数 | 文本 | 视觉 | VAE 编码 | 每步 | 解码 | 编辑 | 峰值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 张参考图 | 1,076 + 1,024 + 4,096 | 4.43 s | 1.03 s | 0.75 s（25） | 3.01 s | 1.19 s | 41.1 s | 41.0 GiB |
| 4 张参考图 | 4,189 + 4,096 + 4,096 | 8.37 s | 2.20 s | 2.77 s（100） | 8.64 s | 1.36 s | 83.2 s | 41.8 GiB |
| 16 张参考图 | 16,635 + 16,384 + 4,096 | 运行前被拒绝：一步约需 52.1 GB，而预算为 44.4 GB | | | | | | |
| 文字编辑（1 张 2048x2048） | 1,091 + 4,096 + 4,096 | 4.64 s | 0.94 s | 3.14 s（121） | 4.89 s | 1.14 s | 55.0 s | 40.4 GiB |
| 组合（2 张 2048x2048） | 2,125 + 8,192 + 4,096 | 运行前被拒绝：一步约需 44.9 GB，而预算为 44.4 GB | | | | | | |
| 组合（2 张 1024x1024） | 2,125 + 2,048 + 4,096 | 5.94 s | 1.32 s | 1.37 s（50） | 4.29 s | 1.17 s | 51.4 s | 41.1 GiB |
| MiniMax 示例（1 张 1024x1024） | 1,459 + 1,024 + 4,096 | 4.96 s | 1.05 s | 0.76 s（25） | 3.20 s | 1.21 s | 42.2 s | 41.1 GiB |

bf16 的第一步包含 40 GB DiT 的上传（与文生图相同，文本编码器在此之前被释放），编辑时间已计入。同一张卡上 2048x2048
文生图的一步为 2.2 s（Q4_K_M）与 1.9 s（bf16）（[见上文](#a40-与-m5-pro)）；一步的开销随行数增长，注意力随行数平方增长。
16 张 1024x1024 参考图在 A40 上使用 Q4_K_M 时所有组件都可常驻（规划器只在去噪前释放文本编码器的工作区）；使用 bf16 时，
同样的编辑会在运行前被行数预算拒绝，消息会给出其行数；两张 2048x2048 参考图的组合同样被拒绝（两张 1024x1024 的可以运行）。
编辑的注意力分数超过 1 GiB（约 2,000 行）时，文本编码器按查询行分块计算：4 张参考图时每块 1,001 行，16 张时 252 行。

| M5 Pro，Q8_0 | 行数 | 文本 | 视觉 | VAE 编码 | 每步 | 解码 | 编辑 | 峰值（已分配） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 张参考图 | 1,076 + 1,024 + 4,096 | 3.59 s | 1.01 s | 3.54 s（25） | 19.98 s | 3.74 s | 134.3 s | 28.3 GB |
| 4 张参考图 | 4,189 + 4,096 + 4,096 | 19.63 s | 3.91 s | 14.11 s（100） | 51.55 s | 3.83 s | 352.4 s | 33.0 GB |
| 16 张参考图 | 16,635 + 16,384 + 4,096 | 运行前被拒绝：一步约需 49.8 GB，而预算为 37.7 GB | | | | | | |
| 文字编辑（1 张 2048x2048） | 1,091 + 4,096 + 4,096 | 5.18 s | 1.25 s | 16.91 s（121） | 33.77 s | 3.81 s | 231.0 s | 30.7 GB |
| 组合（2 张 2048x2048） | 2,125 + 8,192 + 4,096 | 10.60 s | 2.15 s | 33.90 s（242） | 62.57 s | 3.78 s | 427.4 s | 34.5 GB |
| MiniMax 示例（1 张 1024x1024） | 1,459 + 1,024 + 4,096 | 7.77 s | 1.25 s | 3.52 s（25） | 21.47 s | 3.81 s | 146.3 s | 28.4 GB |

上表中的拒绝来自 S5 阶段，当时按规划的预算（空闲显存减去预留余量）检查编辑。此后，编辑与其他所有请求一样按设备的空闲显存检查，
且编辑一步的估算对每个打包行多计两行隐藏宽度的 F32（按上面的 A40 峰值校准：bf16 加 4 张参考图实测为原估算的 1.08 倍，
Q4_K_M 加 16 张参考图比其规划高出 1.1 GB）。按这项检查，bf16 加 16 张参考图仍需约 53.7 GB（A40 为 48.0 GB），Q8_0 加 16 张约需
51.4 GB（M5 Pro 为 40.2 GB），两者都被拒绝；bf16 组合两张 2048x2048 参考图约需 45.5 GB，不再被拒绝，此后尚未实测。

### H100

下面的 H100 数据是在先前的原生路径上测得的；运行时执行的是相同的计算图（见上文）。

一张 H100 NVL 94 GB（`ggml_cuda`，sm_90），热启动（模型已驻留，同尺寸的第二张图），
相同提示词（156 个 token）与种子，6 步。ComfyUI 为 0.45.0，运行其自带模板
（“MiniMax-H3-Image: Text to Image”：剪枝 INT8 DiT、INT8 Qwen3-VL-32B 编码器、FP16 VAE）；
TensorSharp 读取 bf16 DiT 与 Q4_K_M 编码器。ComfyUI 的采样时间取自其进度条；TensorSharp 的
各阶段取自模型打印的 `[h3-image]` 行。

| 2048x2048（4 MP） | 文本 | 去噪，6 步 | VAE 解码 | ComfyUI 任务 |
|---|---:|---:|---:|---:|
| ComfyUI 模板 | | 2.6 s（0.43 s/步） | | 3.97 s |
| TensorSharp，bf16 投影 | 0.10 s | 3.09 s（0.51 s/步） | 0.28 s（121 个瓦片） | 4.38 s |
| TensorSharp，`--diffusion-products int8`（`minimax-h3-image-int8`） | 0.09 s | 2.55 s（0.43 s/步） | 0.27 s | 3.88 s |

| 4096x4096（16 MP） | 文本 | 去噪，6 步 | VAE 解码 | ComfyUI 任务 |
|---|---:|---:|---:|---:|
| ComfyUI 模板 | | 14.7 s（2.45 s/步） | | 19.9 s |
| TensorSharp，bf16 投影 | 0.09 s | 15.8 s（2.63 s/步） | 0.95 s（441 个瓦片） | 19.6 s |
| TensorSharp，`--diffusion-products int8`（`minimax-h3-image-int8`） | 0.09 s | 13.1 s（2.18 s/步） | 0.9 s | 16.9 s |

“任务”指经由 TensorSharp Image 节点的整个 ComfyUI prompt：TensorSharp 的生成及其 PNG 编码
（4 MP 时不到 0.1 s，16 MP 时 0.1 至 0.4 s）、节点的下载与解码，以及 ComfyUI 的 SaveImage
（纯 ComfyUI 任务同样需要，16 MP 的 PNG 为 2.0 s）。服务器的第一个任务（权重来自页缓存）在 4 MP
时 int8 为 9.1 s、bf16 为 9.5 s，ComfyUI 的第一个任务为 55.1 s。

- **投影。** BF16 DiT 的宽矩阵乘（qkv、out、fc1、fc2）以原样权重在 cuBLAS 上运行；
  `--diffusion-products int8`（阶段 S10 改名之前以 `TS_H3_INT8_MATMUL=1` 测得，同一路径）每次调用时把每一行权重和
  每个 token 舍入为 int8，在 int8 Tensor Core 上相乘，
  与 ComfyUI INT8 检查点属同一精度档次。它是可选项，因为会改变图像：与同一种子的 bf16 图像相比，
  PSNR 为 17.5 至 19.8 dB（构图相同，细节不同）。fc2 直接读取 fc1 的 [gate | value] 输出，在收窄
  （或舍入）输入时完成 SwiGLU，省去了 SwiGLU 对 [14336, token 数] 张量的单独一遍（16 MP 时每块 1.5 ms）。
- **注意力。** cuDNN 的 SDPA 在 16,540 个 token 上运行 56 x 128 的注意力，每块 18 ms（含准备与回宽为
  21 ms），与同一 GPU 上 PyTorch 的 cuDNN 后端相当（其 FlashAttention-2 为 30 ms），约占 16 MP 每步的一半。
- **VAE。** 解码器用一个程序处理一整行 256 像素瓦片，这一行的注意力也以一个批次在一次 cuDNN
  调用中完成，而不是每个瓦片一次调用加一次拼接：16 MP 时解码器耗时 0.7 s，逐瓦片注意力为 1.29 s，
  逐瓦片调用为 1.84 s。
- **PNG。** 所有媒体提供者都用托管编解码器写 PNG：Up 滤波，按约 1 MiB 的行分段，以 zlib 4 级并行压缩
  （每段一个 IDAT 块，即 pigz 的做法）。一张 16 MP 图像的编码约 0.1 s，19.9 MB；ImageMagick 在默认的
  7 级下需要 8.3 s，1 级为 0.9 s（同一张图 21.3 MB）；从解码结束到文件写完，由 1.6 s 变为 0.1 至 0.4 s。
- **预览。** TensorSharp Image 节点对生成的图像只预览长边至多 1024 像素的缩小版；ComfyUI 会把预览写成完整
  PNG，16 MP 时需要 1.1 s。节点的 `image` 输出仍是完整图像。
- **画面而非像素。** 两个引擎从相同的潜变量出发（噪声与 ComfyUI 一致），16 MP 时构图相同，但
  逐像素并不一致：Q4_K_M 编码器对 INT8、bf16 或逐次 int8 对 ComfyUI 的 INT8 权重与内核，这些差异
  在 6 步蒸馏采样中被放大（两个引擎之间 PSNR 为 9 至 12 dB；4 MP 时构图可能不同）。两者都遵循了
  提示词，并正确写出了日历上的“十一月 11”。

## TensorAgent

TensorAgent 应用在模型页的 MiniMax H3 下列出两个条目：

| 条目 | DiT | 下载 | 起始内存档 |
|---|---|---|---|
| `minimax-h3-image-q4km` | Q4_K_M | 12,721,119,968 字节的 DiT + 18,218,065,024 字节的文本编码器 + 1,168,505,312 字节的 VAE + `vocab.json`、`merges.txt`、`tokenizer_config.json`（4.46 MB）；共 32.1 GB | 32 GB 系统内存 |
| `minimax-h3-image-q8-0` | Q8_0 | 22,026,760,928 字节的 DiT + 同样的配套文件；共 41.4 GB | 48 GB |

这些文件就是 `minimax-h3-image-q4km.json` 与 `minimax-h3-image-q8.json` 的文件，版本与 SHA-256 相同。
文本编码器和分词器文件与 MiniMax-H3 视频条目的逐字节相同：已装有其中一个条目时，另一个直接链接这些文件，
不会再下载 18.2 GB。加载模型后，一条消息就是对图片的描述；附上照片时则是编辑，提示词按顺序称它们为
`<Picture 1>`、`<Picture 2>`……，与 Qwen-Image 2.1 相同。

- **内存档。** Q4_K_M 最大的阶段是文本编码器（18.2 GB 权重和 0.6 GB 工作内存）；当 GPU 工作集放不下两者时
  （例如 32 GB Mac 约 22.9 GB），流水线会在去噪前释放它。Q8_0 的去噪在 2048x2048 时需要 22.0 GB 权重和
  3.1 GB 工作内存，超出这个工作集，因此从 48 GB 起提供（在 48 GB 的 M5 Pro 上实测 29.9 GB wired）。
- **不标记为实验性。** 它们相对 bf16 DiT 的画质已经实测（见[量化版本](#量化版本)），并写在每个条目的说明里：
  Q4_K_M 是四档中差距最大的一档，可能多出零散的文字；Q8_0 最接近。
- **图片与插件。** 每轮请求模型自己的 2048x2048 面积（见[多轮对话](#多轮对话)）。LoRA 面板为两个条目提供一个插件：
  pDMD 2-step（linoyts @c07d96e5，社区许可）；启用后一张图只需 2 步，该轮请求它蒸馏时的 1024x1024。它是同一许可下的
  模型衍生物，只有在接受该许可之后才会下载：为任一条目接受过当前文本（即该条目自身下载时询问的那次）即可覆盖它；没有
  接受记录时，下载在获取任何内容之前就被拒绝（`POST /api/agent/loras/{id}/download` 返回 `409`，带 `licenseRequired`、
  需要为之接受的条目及其条款；拒绝该条目的许可会撤销这一覆盖）。其余
  MiniMax-H3-Image 插件配置（akhaliq 的 ComfyUI 转换版、Seedreamstyle 风格插件）仅供 CLI 与服务端使用（决策 D7）。
- **许可接受。** MiniMax-H3-Image 的许可限制了谁可以在哪里使用它，因此在条目第一次下载之前，模型页会显示其
  条款：界面语言的摘要（不包括美国、欧盟、英国和大韩民国的地域限制；年收入超过 100 万美元或提供托管服务时须
  获得 MiniMax 的书面授权；商业产品须显示“MiniMax-H3-Image”；使用限制；转交时须附许可副本和 NOTICE 文件；
  适用香港法律），然后是许可原文（逐字照录）以及 Hugging Face 上的链接。**接受并下载**才会开始下载；
  **拒绝**或关闭页面则什么也不会开始。
  - 选择保存在应用数据文件夹（不是缓存）中的 `license-acceptances.json`，按条目和许可版本分别记录：版本是
    应用随附文本的 SHA-256（`0ec3bb5e...`，即 MiniMaxAI/MiniMax-H3-Image 的 `8ced6006`；Unsloth 仓库里是
    MiniMax 在 2026-11-11 修改之前的版本，仅授权链接不同）。每个条目各自询问；目录随附新文本时会再次询问，
    并说明条款已更改；拒绝会撤回对当前文本的接受。
  - 无论由谁发起，下载管理器都会拒绝条款未被接受的条目：模型页的按钮、应用回到前台时的续传、桌面宿主的 HTTP
    路由。在桌面宿主上，`POST /api/agent/catalog/{id}/download` 返回 `409`，带 `licenseRequired` 和条款；
    `GET /api/agent/catalog/{id}/license` 返回条款及原文；`POST /api/agent/catalog/{id}/license` 带
    `{"revision": "...", "accept": true}` 记录接受（版本不是当前版本时返回 `409`），带 `"accept": false`
    记录拒绝（同时停止该条目正在进行的下载）。`/api/agent/catalog` 为每个条目给出 `licenseTerms`（不需要
    接受的条目为 null）。
  - 其他所有条目照旧下载，不会询问。

## 尚未实现

- 让 `ggml_cuda` 上 bf16 权重的 LoRA 与量化权重一样便宜：目前每步多花约 10%（每次把权重复制进 K + R 行）；
  解决办法是一个从各自指针读取 LoRA 列的 GEMM。
- LoRA 合并（更新始终以不合并方式运行）、DoRA、LoCon、LoHa 与 LoKr 适配器，以及 PEFT 配置中按模块的
  alpha 或秩模式。
- 该模型的聊天规划提问（后续消息修改哪张图）：其文本编码器没有输出头，因此后续消息修改最新的图片。
- 对保留行的遮罩感知调制（把它们按干净状态调制，如 ComfyUI 的 H3 视频那样）：设计为默认关闭的实验，尚未实现或测量。
- 提示词扩写器（MiniMaxAI/MiniMax-H3-Image-PE）。ComfyUI 的文生图模板默认不启用它。
- `mlx`、`cuda` 与 `cpu`（纯 C#）后端。
