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
运行相同的程序（已在小型合成检查点上验证；使用真实权重时每张图需要数十分钟）。遮罩编辑尚未实现。

| 组件 | 文件 | 来源 |
|---|---|---|
| 扩散 Transformer（bf16，官方发布格式） | `minimax_h3_image_pruned_bf16.safetensors`，40.2 GB | [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/diffusion_models) |
| 文本编码器（与 MiniMax-H3 共用） | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf`，16.97 GiB，以及 `vocab.json`、`merges.txt`、`tokenizer_config.json` | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF)、[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/processor) |
| 图像 VAE | `minimax_h3_image_vae_fp16.safetensors`，1.17 GB | [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/vae) |

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
`minimax-h3-image-int8`（`TS_H3_INT8_MATMUL=1`，见[性能](#性能)）。

请求参数：

- **尺寸**：未指定时为 2048x2048（ComfyUI 模板的默认值）；边长须为 32 的倍数。
- **步数**：默认 6，即蒸馏步数。其他步数也能运行（跨越多个网格区间的一步取这些区间
  输出头的 dt 加权平均，与 ComfyUI 一致），但不是检查点训练时的设置。
- **Guidance**：固定为 1。`cfg` 大于 1 或负面提示词会被拒绝。
- **种子**：初始潜变量按 ComfyUI KSampler 的方式生成（CPU `torch.randn`，
  `torch.manual_seed(seed)`），因此相同种子在两者中从相同的潜变量出发。

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
  2048x2048 的编辑在同宽高比的更大画布上采样（diffusers 的 `min_generation_area`），再用 Lanczos 缩放回来。
  文生图保持自己的尺寸规则（没有下限）。
- **拒绝。** `cfg` 大于 1、负面提示词、遮罩、超过 16 张参考图、某边小于 32 像素，以及参考图使序列超出设备显存的
  编辑（消息给出行数；减少参考图或使用更小的参考图即可）。

每张参考图都会增加两处行数：文本中约 1,024 行的视觉块，以及 DiT 中 (VAE 副本 / 32)² 行（1024x1024 副本为 1,024 行，
2048x2048 为 4,096 行）。16 张 1024x1024 参考图加 2048x2048 的目标图约为 37,000 行 DiT，是文生图一步的九倍；
注意力随行数平方增长。

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
  `TS_H3_IMAGE_VAE_BATCH` 是那个解码器的变通办法，现已不再读取。

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

若不做这些释放，bf16 在 A40 上与文本编码器放不下（约 60 GB），Q8_0 在 48 GB 的 Mac 上也放不下：
先前的路径保留文本编码器，在加载 DiT 时于 wired 43.5 GB 处被 wired 内存看门狗终止。A40 上的 bf16
每张后续图片都要重新上传文本编码器与 DiT（在该虚拟机上从磁盘读取约 4 s 与 15 s，其页缓存放不下两者）。
bf16 在 48 GB 的 Mac 上完全放不下（40.2 GB 权重对 40.2 GB 工作集）：请在那里使用 Q8_0。H100 级别的
显卡可保留全部组件，16 MP 的 bf16 也不例外。

## 性能

### A40 与 M5 Pro

一张 A40 46 GB（`ggml_cuda`，sm_86，共享虚拟机）与一台 M5 Pro 48 GB（`ggml_metal`），验证集的提示词
（40 至 160 个 token），6 步。“文本”为编码；“每步”为第 2 至 6 步的中位数；“整张图”为整个请求。进程中的
第一张图冷读文件（A40 上事先把文件逐出页缓存）并包含上传；之后的图为热启动。

| A40，2048x2048 | 文本 | 每步 | 解码 | 整张图 |
|---|---:|---:|---:|---:|
| Q4_K_M，第一张 | 3.25 s | 2.23 s | 1.22 s | 21.0 s |
| Q4_K_M，后续 | 0.08 s | 2.24 s | 0.69 s | 14.3 s |
| Q4_K_M，`TS_H3_INT8_MATMUL=1`，后续 | 0.08 s | 1.88 s | 0.68 s | 12.1 s |
| bf16，第一张（去噪前释放文本编码器） | 3.07 s | 1.92 s | 1.20 s | 29.7 s |
| bf16，后续（每次都从磁盘重新加载两者） | 4.27 s | 1.91 s | 0.73 s | 34.2 s |

| A40，4096x4096 | 文本 | 每步 | 解码 | 整张图 |
|---|---:|---:|---:|---:|
| Q4_K_M，第一张 | 3.25 s | 11.42 s | 2.91 s | 77.8 s |
| Q4_K_M，后续 | 0.07 s | 11.47 s | 2.40 s | 71.5 s |
| Q4_K_M，`TS_H3_INT8_MATMUL=1`，后续 | 0.07 s | 9.96 s | 2.48 s | 62.5 s |
| bf16，第一张 | 3.36 s | 10.36 s | 3.09 s | 82.6 s |

| M5 Pro，2048x2048 | 文本 | 每步 | 解码 | 整张图 |
|---|---:|---:|---:|---:|
| Q8_0，第一张 | 1.71 s | 12.82 s | 3.45 s | 82.6 s |
| Q8_0，后续 | 1.70 s | 12.67 s | 2.72 s | 81.1 s |

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
| TensorSharp，`TS_H3_INT8_MATMUL=1`（`minimax-h3-image-int8`） | 0.09 s | 2.55 s（0.43 s/步） | 0.27 s | 3.88 s |

| 4096x4096（16 MP） | 文本 | 去噪，6 步 | VAE 解码 | ComfyUI 任务 |
|---|---:|---:|---:|---:|
| ComfyUI 模板 | | 14.7 s（2.45 s/步） | | 19.9 s |
| TensorSharp，bf16 投影 | 0.09 s | 15.8 s（2.63 s/步） | 0.95 s（441 个瓦片） | 19.6 s |
| TensorSharp，`TS_H3_INT8_MATMUL=1`（`minimax-h3-image-int8`） | 0.09 s | 13.1 s（2.18 s/步） | 0.9 s | 16.9 s |

“任务”指经由 TensorSharp Image 节点的整个 ComfyUI prompt：TensorSharp 的生成及其 PNG 编码
（4 MP 时不到 0.1 s，16 MP 时 0.1 至 0.4 s）、节点的下载与解码，以及 ComfyUI 的 SaveImage
（纯 ComfyUI 任务同样需要，16 MP 的 PNG 为 2.0 s）。服务器的第一个任务（权重来自页缓存）在 4 MP
时 int8 为 9.1 s、bf16 为 9.5 s，ComfyUI 的第一个任务为 55.1 s。

- **投影。** BF16 DiT 的宽矩阵乘（qkv、out、fc1、fc2）以原样权重在 cuBLAS 上运行；
  `TS_H3_INT8_MATMUL=1` 每次调用时把每一行权重和每个 token 舍入为 int8，在 int8 Tensor Core 上相乘，
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

## 尚未实现

- 遮罩编辑（局部重绘）以及该模型的多轮对话规划。
- 提示词扩写器（MiniMaxAI/MiniMax-H3-Image-PE）。ComfyUI 的文生图模板默认不启用它。
- `mlx`、`cuda` 与 `cpu`（纯 C#）后端。
