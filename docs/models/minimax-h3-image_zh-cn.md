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

TensorSharp 通过 `MiniMaxH3ImageModel` 在 ggml 后端上运行文生图。已在 `ggml_cuda`
（H100，见下文）上实测；`ggml_metal` 与 `ggml_cpu` 使用相同的原生计算图，但尚未实际运行。
使用参考图的编辑（`<Picture 1>` ……）尚未实现。

| 组件 | 文件 | 来源 |
|---|---|---|
| 扩散 Transformer（bf16，官方发布格式） | `minimax_h3_image_pruned_bf16.safetensors`，40.2 GB | [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/diffusion_models) |
| 文本编码器（与 MiniMax-H3 共用） | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf`，16.97 GiB，以及 `vocab.json`、`merges.txt`、`tokenizer_config.json` | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF)、[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/processor) |
| 图像 VAE | `minimax_h3_image_vae_fp16.safetensors`，1.17 GB | [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/vae) |

去噪器直接从 safetensors 文件按原始张量名读取，无需转换为 GGUF。包含相同张量的
GGUF（用于更小的量化）同样可以加载，依据张量识别（没有 `audio_patch_proj` 的 H3 DiT）。
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

服务器在图像路由（`/api/image-generate`、`/api/image-generate/stream`）上提供它，
与 Qwen-Image-2.1 相同；ComfyUI 节点包为 **TensorSharp Image** 节点提供启动配置
`minimax-h3-image`（`examples/comfyui/profiles/`），以及使用 int8 投影的
`minimax-h3-image-int8`（`TS_H3_INT8_MATMUL=1`，见[性能](#性能)）。

请求参数：

- **尺寸**：未指定时为 2048x2048（ComfyUI 模板的默认值）；边长须为 32 的倍数。
- **步数**：默认 6，即蒸馏步数。其他步数也能运行（跨越多个网格区间的一步取这些区间
  输出头的 dt 加权平均，与 ComfyUI 一致），但不是检查点训练时的设置。
- **Guidance**：固定为 1。`cfg` 大于 1 或负面提示词会被拒绝。
- **种子**：初始潜变量按 ComfyUI KSampler 的方式生成（CPU `torch.randn`，
  `torch.manual_seed(seed)`），因此相同种子在两者中从相同的潜变量出发。

## 运行方式

- **提示词。** 原始提示词 token，不套聊天模板，经过 MiniMax-H3 的 Qwen3-VL 主干；条件是
  第 50 层的隐藏状态。编码器在多次生成之间保持驻留。
- **打包序列。** `[文本 | 目标图像]`，使用 H3 的三轴 RoPE：第 i 个文本 token 位于 (i, 0, 0)，
  图像的 2x2 patch 在时间轴上紧随其后，空间网格按宽高比归一化。文本与图像都在
  t = 1 - sigma 上运行，没有 flow shift。
- **调度。** ComfyUI 模板在检查点网格上使用 KSampler 的 `simple` 调度器
  （`MiniMaxH3ImageModelSampling`）；TensorSharp 以 float32 复现这些 sigma
  （6 步时为 1、0.96765、0.88771、0.75、0.53364、0.23261、0）。
- **PDD 输出头。** `final_layer.video_out` 叠放了 6 个头；从网格点 k 到 k+1 的一步使用
  头 0 加头 k（第一步只用头 0）。组合后的头按区间构建一次，并像普通权重一样绑定。
- **DiT。** H3 原生计算图（`TSGgml_MiniMaxH3DitForward`），不含音频流，使用 cuDNN 注意力
  及融合的调制与残差阶段。
- **VAE。** 复用 H3 视频 VAE 的原生 ViT 解码，换成图像解码器的几何参数（24 个 1024 宽的块、
  16 个头、在每个头的一半维度上对 (h, w) 做 RoPE、16x16 RGB patch），在至少重叠 64 像素的
  256 像素瓦片上解码，并完全按 ComfyUI 的 `MiniMaxH3ImageVAE` 方式融合进画布。

## 性能

一张 H100 NVL 94 GB（`ggml_cuda`，sm_90），热启动（模型已驻留，同尺寸的第二张图），
相同提示词（156 个 token）与种子，6 步。ComfyUI 为 0.45.0，运行其自带模板
（“MiniMax-H3-Image: Text to Image”：剪枝 INT8 DiT、INT8 Qwen3-VL-32B 编码器、FP16 VAE）；
TensorSharp 读取 bf16 DiT 与 Q4_K_M 编码器。ComfyUI 的采样时间取自其进度条；TensorSharp 的
各阶段取自模型打印的 `[h3-image]` 行。

| 2048x2048（4 MP） | 文本 | 去噪，6 步 | VAE 解码 | ComfyUI 任务 |
|---|---:|---:|---:|---:|
| ComfyUI 模板 | | 2.6 s（0.43 s/步） | | 3.97 s |
| TensorSharp，bf16 投影 | 0.09 s | 3.17 s（0.53 s/步） | 0.47 s（121 个瓦片） | 5.28 s |
| TensorSharp，`TS_H3_INT8_MATMUL=1`（`minimax-h3-image-int8`） | 0.09 s | 2.63 s（0.44 s/步） | 0.44 s | 4.72 s |

| 4096x4096（16 MP） | 文本 | 去噪，6 步 | VAE 解码 | ComfyUI 任务 |
|---|---:|---:|---:|---:|
| ComfyUI 模板 | | 14.7 s（2.45 s/步） | | 19.9 s |
| TensorSharp，bf16 投影 | 0.09 s | 16.4 s（2.74 s/步） | 1.7 s（441 个瓦片） | 23.7 s |
| TensorSharp，`TS_H3_INT8_MATMUL=1`（`minimax-h3-image-int8`） | 0.09 s | 13.4 s（2.24 s/步） | 1.7 s | 20.7 s |

“任务”指经由 TensorSharp Image 节点的整个 ComfyUI prompt：TensorSharp 的生成及其 PNG 编码
（4 MP 时 0.45 s，16 MP 时 1.55 s）、节点的下载与解码，以及 ComfyUI 的 SaveImage（纯 ComfyUI
任务同样需要）。服务器的第一个任务（权重来自页缓存）在 4 MP 时 int8 为 9.9 s、bf16 为 16.1 s，
ComfyUI 的第一个任务为 55.1 s。

- **投影。** BF16 DiT 的宽矩阵乘（qkv、out、fc1、fc2）以原样权重在 cuBLAS 上运行；
  `TS_H3_INT8_MATMUL=1` 每次调用时把每一行权重和每个 token 舍入为 int8，在 int8 Tensor Core 上相乘，
  与 ComfyUI INT8 检查点属同一精度档次。它是可选项，因为会改变图像：与同一种子的 bf16 图像相比，
  PSNR 为 17.5 至 19.8 dB（构图相同，细节不同）。
- **VAE。** 解码器在一次原生调用中处理一整行 256 像素瓦片（`TSGgml_MiniMaxH3VitDecodeBatch`）：
  16 MP 时为 21 次调用而非 441 次，原生耗时 1.29 s，逐瓦片调用为 1.84 s。
- **PNG。** 桌面媒体提供者以 zlib 1 级加自适应滤波（ImageMagick quality 15）编码 PNG：一张 16 MP
  图像在 ImageMagick 默认的 7 级下需要 8.3 s，现在为 0.9 s，文件大 14%。
- **画面而非像素。** 两个引擎从相同的潜变量出发（噪声与 ComfyUI 一致），16 MP 时构图相同，但
  逐像素并不一致：Q4_K_M 编码器对 INT8、bf16 或逐次 int8 对 ComfyUI 的 INT8 权重与内核，这些差异
  在 6 步蒸馏采样中被放大（两个引擎之间 PSNR 为 9 至 12 dB；4 MP 时构图可能不同）。两者都遵循了
  提示词，并正确写出了日历上的“十一月 11”。

## 尚未实现

- 编辑与参考图（VAE 编码器路径以及 `Picture n:` 视觉块）。
- 提示词扩写器（MiniMaxAI/MiniMax-H3-Image-PE）。ComfyUI 的文生图模板默认不启用它。
- `mlx`、`cuda` 与 `cpu`（纯 C#）后端。
