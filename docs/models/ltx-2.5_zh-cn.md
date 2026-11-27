# LTX 2.5 Distilled

[English](ltx-2.5.md) | [中文](ltx-2.5_zh-cn.md)

TensorSharp 原生实现 LTX 2.5 Distilled 的文生视频、图像条件视频及联合生成的
**48 kHz 立体声音频**。TensorAgent 的实验条目为
**LTX 2.5 Distilled — UD-Q4_K_XL + BF16**，Download 下载五个固定基础文件，Use 激活模型。
可选 Ingredients IC-LoRA 可启用身份/外观参考；TensorAgent 默认会一并下载可选文件。
48 GB 目录档位是保守起点，并非已证明的最低要求或任意分辨率/时长保证。

## 文件与启动

[config/ltx-2.5-distilled.json](../../config/ltx-2.5-distilled.json)
按提交和 SHA-256 固定五份文件，共 **43,077,008,276 字节（43.08 GB）**：
UD-Q4_K_XL 去噪器、带投影的专用 Gemma 4 编码器、卷积视频 VAE、x2 空间潜变量上采样器、
音频 VAE。全部来自无访问限制的 [Unsloth 仓库](https://huggingface.co/unsloth/LTX-2.5-GGUF)，
固定修订为 `ac3c540c6025903784336e99b2dc179385c4c3b7`，无需 Hugging Face 账号。
伴随文件与 Lightricks 原文件逐字节相同，配置还保留受限的官方地址作为备用。

配置还声明可选的 Ingredients IC-LoRA，保存到 `${ltxRoot}/loras/`，固定于 Lightricks
修订 `12040e4091ac2008d3906a594e31a7fb1ab9d546`（1,308,787,472 字节，SHA-256
`ff873a5beada3c579a8137c7c53343916f78bcc9a529ba910143073fe8715e95`）。该仓库受限，需先接受
LTX-2.x 条款并设置 `HF_TOKEN` 或 Hugging Face token 文件。授权时，配置会在加载模型前下载；
无权限时会跳过，普通视频生成仍可使用，Video Studio 会在文件存在时自动启用身份参考。

应下载 `distilled/ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf`，直接放在 `ltx25/` 下与伴随文件夹并列。
仓库根目录的 **dev** 检查点使用另一套配方（30 步、CFG 3）；TensorSharp 拒绝文件名标注 dev 的文件，
改名无法将其转换为蒸馏检查点。TensorAgent 新条目为 `ltx-2.5-distilled-q4kxl`，
使用默认模型缓存时，启动迁移保留旧条目中完整的伴随文件与用户添加的适配器文件夹。

配置在 `${TENSORSHARP_MODELS:-../models}/ltx25` 下查找文件。
手动安装可用 `TS_VIDEO_TEXT_ENCODER`、`TS_VIDEO_VAE`、`TS_VIDEO_AUDIO_VAE`、
`TS_VIDEO_UPSCALER`（CLI/服务端 `--video-upscaler`）。
上采样器否则从去噪器旁的 `latent_upscale_models/` 加载，是必需组件。
专用编码器内嵌分词器，普通 Gemma 4 聊天检查点不能替代它。
默认使用配置指定的 **conv** 视频 VAE；扩散解码器是额外可选文件。

```sh
TENSORSHARP_MODELS=/path/to/models MLX_ENABLE_TF32=0 \
dotnet run --project TensorSharp.Cli -- \
  --config config/ltx-2.5-distilled.json --backend mlx \
  --prompt "A red toy car drives slowly along a quiet road in warm afternoon light." \
  --diffusion-seed 42 --output ltx25.mp4
```

| 后端名称 | 实际执行路径 |
|---|---|
| `cuda` / `ggml_cuda` | 扩散运行时的 CUDA tensor executor |
| `mlx` / `ggml_metal` | 扩散运行时的 MLX tensor executor |
| `cpu` / `ggml_cpu` | 扩散运行时的托管 CPU tensor executor |

不支持 Vulkan 或多 GPU。GPU 测量不证明 CPU 吞吐。
桌面 MP4 音轨需要 `PATH` 上的 FFmpeg 或 `TS_FFMPEG`，CLI 另写原始 PCM WAV；
Mac 应用通过 AVFoundation 编码。

## Unsloth 预设实测（2026-10-18）

48 GB M5 Pro、MLX、`MLX_ENABLE_TF32=0`、1024×1024×17 两阶段生成、seed 42，
同一进程的两条不同提示词分别耗时 109.7 / 107.1 秒，旧 Abiray 文件为 115.3 / 115.2 秒。
新文件的进程峰值占用为 17.6 GB，旧文件 27.0 GB。混合 Q2_K/Q3_K/BF16 权重现在保持映射，
逐次投影精确解量化到 F32；修复前一次性展开会导致 56 GB 占用。修复前后原始视频与音频逐比特相同。

以更高精度的 Unsloth Q8_0 为参考，狐狸/灯塔两条样本的 PSNR 为 26.6/30.8 dB，旧量化为
25.1/23.0 dB。这两条样本支持当前选择，但不能证明通用质量排名。
1280×704×121 样本生成耗时 513.5 秒、编码 36.2 秒，MLX 分配峰值为 20.2 GB。
CLI 配置下载与真实 TensorAgent HTTP/SSE 生成、媒体下载路径已验证；本轮未重测 CUDA、iOS
或原生 MAUI 控件。工具与限制见[验证说明](../../eng/validation/README-ltx25.md#unsloth-distilled-gguf-validation-2026-10-18)。

## 两阶段生成与限制

普通文本/图像模式采用发布方 `DistilledPipeline` 的两阶段流程：

1. 宽高是最终输出尺寸；先在一半宽高以八个公开 sigma 运行完整 48 块 AV 去噪器。
2. 必需的 x2 空间潜变量上采样器放大视频潜变量。
3. 全尺寸运行三个 refinement sigma（0.909375、0.725、0.421875），
   视频与音频重新加噪；图像引导在全尺寸重新编码并施加。

每一阶段有独立噪声，遵循发布方 +10000 / +20000 seed 偏移。
两阶段均使用 CFG 1、无 flow-shift 覆盖、ancestral Euler（`euler_ancestral`，eta 1）。
`--sampler euler` 显式选择确定性诊断，日志会说明偏离发布方配方。
其他显式步数/采样器/引导请求会被拒绝；请求 `steps: 8` 指第一阶段的八步。
身份参考 `Mode = "ref"` 使用单次全尺寸 Ingredients 流程，见后文。

默认 **1024×1024、17 帧、24 fps**；宽高必须为 64 的倍数，帧数必须为 `8k+1`。
1024 输出对应 512 首阶段，有助于人脸和手部细节；降低至 512 输出时首阶段仅 256，
不能由小尺寸执行成功推断同等细节质量。
Studio 和聊天使用相同的生成流程，但结果一致不等于手指或眼睛一定正确。
145 帧 banner 样例在修正图像预处理后，逐帧检查首秒未再发现原来的白眼异常，
采用首阶段 0.7、精修阶段 1.0 后，第 63–84 帧有明显更稳定的五指和不透明手掌；
第 56–61、88–94 帧附近仍有手指与脸部混合的遮挡错误。单独更换解码器未解决此前全强度引导的手部错误。
此结果不能泛化为所有提示词、种子和时长都已解决人体结构问题。
普通单张首图上传在自动或 `i2v` 模式采用上述 0.7/1.0 默认值；显式 `ImageConditions` 强度、
有序多图、尾帧和其他模式保持原值，各阶段会记录实际引导强度。
当前保留用户提示词，不自动使用官方 ComfyUI 模板的图像感知提示词增强。
不支持参考视频/音频、retake、HDR/EXR、生成式关键帧槽或多 GPU。
关闭音频解码仍运行训练好的联合 AV 去噪器。
原生音频可能略短于视频；导出裁掉多余音频，不给短音轨补齐。

Apple 进程建议启动时设置 `MLX_ENABLE_TF32=0`。LTX 在首次 MLX 操作前设置该默认值，
但 MLX 会在进程首次 GEMM 时缓存选项，已初始化的进程需要启动环境设置。
TensorAgent Mac 应用在启动宿主前设置该默认，显式用户覆盖仍优先。

CUDA 的 SM 8.0+ 提示词投影用 BF16 Tensor Core、F32 累加并保留 BF16 激活边界，
较旧设备用 F32。CUDA 对映射投影做有界预读，每个 reader 一个 8 MiB 缓冲；Linux 检查驻留页，
至多八路读取 1 MiB 切片。

所有后端还会在 4 GiB 窗口内按层的运行顺序提前读取权重：Gemma 编码器、提示词
connector 以及第一次 CUDA 去噪器上传。upscaler、视频解码器和音频 VAE 则在首次
使用前一个阶段读取。safetensors 与 GGUF 按名字字母序存放张量（第 0、1、10……层）。
没有预读时，冷映射在 M5 Pro SSD 上只能以 0.9 GiB/s 缺页读入，而一个顺序读取线程
可达 12 GiB/s；在 A40 pod 的网络卷上只有 0.3 GiB/s，而 16 路并发读取可达
2.3 GiB/s。已在文件缓存中的块会跳过。

当内存（或容器的内存上限）小于文本编码器、去噪器与 16 GiB 之和时，每个新提示词
编码后释放编码器的缓存页，使其不与去噪器争用；之后的新提示词会重新读取。
`TS_VIDEO_READAHEAD=0` 同时关闭预读与释放。`TS_VIDEO_READAHEAD_WORKERS`（1–32）
设置读取线程：Linux 默认 16，其他平台默认 1。条件编码日志行会报告读取的字节数、
已驻留的字节数以及是否释放缓存。
大序列自注意力在 256 MiB 分数工作区中按查询/head 分块，仍关注完整 key 序列。

## 首帧、尾帧与多图时间线

TensorAgent 或服务端上传一张图表示首帧，多张图默认是按顺序均匀分布的关键帧。
提示词描述所需动作。API 可明确控制位置和强度：

| 输入 | 含义 |
|---|---|
| `Image` / `ImageBytes` / `ImagePath` | 首帧；优先级按左到右 |
| `EndImage` / `EndImagePath` | 最后一帧 |
| `ReferenceImagePaths` | 默认按顺序均匀分布；`Mode = "ref"` 时为身份参考板 |
| `ImageConditions` | `VideoImageCondition` 列表，指定 `FrameIndex` 和 `Strength`；`-1` 为末帧 |

```csharp
var request = new VideoGenerationParams
{
    Width = 1024, Height = 1024, Frames = 17, Fps = 24, Seed = 42,
    ImageConditions = new[]
    {
        new VideoImageCondition { ImagePath = "first.png", FrameIndex = 0 },
        new VideoImageCondition { ImagePath = "middle.png", FrameIndex = 8, Strength = 0.8f },
        new VideoImageCondition { ImagePath = "last.png", FrameIndex = -1 },
    },
};
var video = model.GenerateVideo("The woman smiles and blows a kiss.", request);
```

最多 16 张图，位置必须唯一且位于片段内；强度为有限的 0–1，1 保留编码引导潜变量。
`ReferenceImagePaths` 不可与其他图像输入混用；唯一例外是 `Mode = "ref"` 下的一个第 0 帧
`ImageConditions` 引导（见[从首帧开始](#从首帧开始)）。
时间线图像按发布方 LTX 2.5 的流程预处理：先去掉底部/右侧的奇数边缘以满足偶数尺寸，
使用 libx264、CRF 18、`veryfast`、`yuv420p` 再压缩；随后对完整图像做 Float32
双线性缩放（`align_corners=False`），最后中心裁剪到各阶段尺寸，保留缩放产生的小数像素。
RGB/YUV 转换及 slice 线程设置与 PyAV 一致；不同 x264 版本或 CPU 线程数可能改变有损压缩后的像素。
增加引导会增加 token 数、内存和耗时。
第零帧替换首个潜变量帧，较后引导追加 clean token 及像素时间区间。
首个画布潜变量帧总有模型的 keyframe 标记，包括文生视频；后追加的图像 token 不带标记。
解码首帧是 VAE 重建，不保证与上传像素精确相同。
桌面时间线图像条件要求 `PATH` 或 `TS_FFMPEG` 指向包含 **libx264** 的 FFmpeg。
能力发现会检查该编码器；缺失或不支持时，在图像编码和去噪前报错，不回退到其他系统编码器。
Managed 和仅 Apple 媒体提供器尚未实现此变换，因此不声明时间线图像条件能力；
文生视频与独立的 Ingredients 身份参考路径不受影响。
`Ltx25ImagePreprocessingTests` 覆盖 CPU 缩放数值对照和可选的本机 PyAV 编解码对照。
用 `eng/validation/generate_ltx25_image_preprocessing_reference.py` 生成本机参考，
再将 `TS_LTX25_IMAGE_ORACLE` 设为其输出路径可运行编解码逐像素对照。

## 身份与外观参考：Ingredients

[Ingredients IC-LoRA](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients)
使用组合参考板来保留人物、物件和场景。将
`ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors` 放在去噪器旁的 `loras/`，
或设置 `TS_LTX25_INGREDIENTS_LORA`。授权后，随附配置会在首次加载模型前下载该文件。
需先接受受限仓库条款并设置 `HF_TOKEN` 或 Hugging Face token 文件。文件大小
1,308,787,472 字节，SHA-256 为
`ff873a5beada3c579a8137c7c53343916f78bcc9a529ba910143073fe8715e95`。
TensorAgent 默认也会下载此可选文件，并在加载后提供 **Identity references (Ingredients)**。
服务端启动时读取配置；更新后下次启动即可在首次模型加载时启用参考图。

```csharp
var request = new VideoGenerationParams
{
    Mode = "ref", Width = 768, Height = 448, Frames = 121, Fps = 24, Seed = 42,
    ReferenceImagePaths = new[] { "character.png", "outfit.png" },
};
var video = model.GenerateVideo(
    "The woman smiles, touches her fingertips to her lips and blows a kiss toward the camera.", request);
```

一张图成为整张参考板；多图按顺序组成黑底面板，各图等比放入单元格，不裁剪或拉伸。
完整静态参考视频经 VAE 编码，所有参考帧作为 clean guide token 追加；
IC-LoRA 对 AV Transformer 施加独立低秩更新，不替换生成首帧。
训练默认 **768×448、121 帧、24 fps、八步、CFG 1**，至少 121 帧。
小尺寸只是诊断，不是训练配方质量验证。

建议说明每个面板的实际位置、哪些视图为同一身份，再描述目标视频中的动作。
使用无文字、logo 或无关拼贴的干净主体图。单参考结果更可预测；
已观察到多参考照搬面板布局、改变镜头构图，修改字幕或裁掉文字也未必消除。
保留身份与遵循动作是两个独立质量要求，当前没有多人物广泛验证。
MotionTrack、Union depth/canny/pose、layout、inpainting、HDR、AlphaGen 等官方工作流
需要各自管线/适配器/解码器，不能当作通用图片参考使用。

### 从首帧开始

Ingredients 请求也可以从固定的首帧开始，即发布方 IC-LoRA 管线同时接收图像与参考条件时的组合。
Video Studio **不提供**此功能（原因见下方验证结果与[跨片段保持人物](../video-studio.md#keep-a-character-across-segments)，英文），
可通过视频 API 使用。
做法是加一个 `FrameIndex = 0` 的 `ImageConditions` 条目；`Mode = "ref"` 下 `Image` / `ImagePath`
仍是单张身份图，而不是帧。仍为单次全尺寸流程：该引导与其他图像引导一样经 CRF 18 预处理（因此需要带
libx264 的 FFmpeg），按输出尺寸只编码一次，并按其强度固定——画布第 0 个潜变量帧取干净引导潜变量，
去噪掩码为 1 − strength，不增加 token；参考板 token 照常追加在画布之后。两者占用互不重叠的 token，
所以先参考后引导与上游"先图像后参考"的顺序等价，对应上游
`ICLoraPipeline(stages=(ICLoraStageConfig(1, True, DISTILLED_SIGMA_VALUES),))(images=[(frame, 0, 1.0)], video_conditioning=[(sheet, 1.0)])`。
最多一个引导且只能在第 0 帧；参考旁的尾帧或带时间位置的引导会被拒绝。去噪日志会写出
`first-frame guide strength: frame 0=1`。该组合在强度 1 时与未修改的上游函数逐位一致
（`FrameZeroGuideBesideReferenceVideoMatchesUnchangedOfficialImagesThenReferences`）。发布方训练 Ingredients 时
`first_frame_conditioning_p` 为 0.0，训练中从未见过固定首帧，也未验证这种用法。2026-10-17
（Q4_K_M、MLX、768×448×121、seed 42）的实测中，每个从首帧开始的镜头都在第二帧切走：第 0 帧与固定首帧一致
（PSNR 27–29 dB），第 1 帧与第 0 帧相差 9.6–9.7 dB，呈现参考板的构图（竖幅参考在板上加黑边、宽幅参考铺满时均如此）。
用竖幅参考的普通 Ingredients 镜头同样复现了加黑边的参考板。

## 视频解码器与清晰度

默认卷积 VAE 为 `ltx-2.5-video-vae-conv-bf16.safetensors`。
额外的 `ltx-2.5-video-vae-bf16.safetensors` 是发布方 `NADiffusionDecoder`，
通过 `--video-diffusion-vae` / `TS_VIDEO_DIFFUSION_VAE` 在 MLX 或 CPU 上启用；
CUDA 缺少 neighborhood-attention 内核，会明确拒绝。
它是额外 1.47 GB，不属于默认预设或 TensorAgent 下载；两份文件的编码器和潜变量统计相同。

在同一 banner 图、两阶段 1024×1024×97、seed 42 样本中，扩散解码器噪点和闪烁较少，
但眼睫毛、牙齿和布料边缘更柔和。首帧对源图 PSNR/SSIM：卷积为 33.4 dB / 0.930，
扩散为 30.2 dB / 0.915；因此卷积仍为默认。这是单主体、单种子，不是普遍质量排名。

此前人脸和遮脸手掌失真涉及三项偏差：确定性 Euler、AV cross-attention 时间步倍率
未按检查点元数据取 1000，以及省略半尺寸/上采样/全尺寸细化流程。三项已修正。
M5 Pro MLX 的同图检查中，两阶段 1024×1024×121 的手掌能遮挡面部，单阶段高分辨率
仍出现透明手掌。该运行约 1,600 秒，MLX 峰值 41.5 GB；机器在连续运行后热降频，
不能当作常态延迟。更大尺寸、更多主体或种子没有由这个样本验证。
详细检查与解码器差异见[英文卡片](ltx-2.5.md#clear-faces-and-hands)。

## 执行图、缓存与最新性能工作

[`Ltx25PromptEncoder`](../../TensorSharp.Models/Models/Ltx25/Ltx25PromptEncoder.cs)
运行真实 48 层 Gemma 4 Unified，收集 49 个隐藏状态，
保留 BF16 边界、局部 global RoPE 和 padding 位置。V2 特征归一化与投影进入八个视频和
八个音频 connector；学习的 register 让完整 1024-token 上下文有效，不能直接丢掉 padding。
LTX 2.5 运行在共享的[扩散运行时](../design/diffusion-runtime.md#312-ltx-25-on-the-runtime-as-built)上：
[`Ltx25Family`](../../TensorSharp.Models/Models/Ltx25/Ltx25Family.cs) 通过 `DiffusionVideoPipeline` 完成两阶段采样，
[`Ltx25Denoiser`](../../TensorSharp.Models/Models/Ltx25/Ltx25Denoiser.cs) 每步把 48 个 AV 块作为一个图程序在后端的 tensor executor 上运行；
[`潜变量上采样器`](../../TensorSharp.Models/Models/Ltx25/Ltx25LatentUpsampler.cs)
有 1024 个中间通道、前后各四个 ResBlock、Conv2d + PixelShuffle(2)，
GroupNorm(32) 在整段片上以 Float64 累积统计。解码使用精确扩宽的 BF16 权重与 F32 激活。

模型保留映射提示词/视频文件、去噪器设备权重身份、音频转换缓存与最多四组最终提示词上下文。
提示词逐投影、视频 VAE 逐卷积流式加载，新提示词结束后释放 connector 设备权重；
`TrimIdleMemory` 释放所有保留组件。组件内存数据不代表完整生成峰值。

最新源码把保持原顺序的 BF16 舍入与 Gemma split-half RoPE 放在 CUDA/MLX 设备执行，
三角函数仍按原 `MathF` 和 BF16 舍入产生。CUDA 融合 `2 * sigmoid(logit)` head gating，
保留独立 F32 舍入边界。选择旧内核的开关已随直接去噪器删除（见[扩散运行时迁移](#扩散运行时迁移2026-11-27)）。
MLX Q5_K 以有界 CPU 并行直接打包，保留所有量化值和缩放字节；
`TS_MLX_Q5K_REPACK=0` 选择旧转换。旧可选 PTX 缺少导出时会回退，
所以成功生成本身不证明实际用了新内核，严格测试还要求 native dispatch。

额外矩形注意力批量路径在组件上通过，却未通过完整生成 RGB/双声道 PCM 等价检查，
扩展与选择器已移除。保留已接受的提示词运算和 head gating 后，
三个远端完整生成用例重新得到逐比特相同的 RGB 和两个 PCM 声道。
组件微基准与打包耗时不能换算成模型整体提速。

## 验证与复现

### 扩散运行时迁移（2026-11-27）

LTX 2.5 及其 Ingredients IC-LoRA 只运行在共享的扩散运行时上
（[设计](../design/diffusion-runtime.md#312-ltx-25-on-the-runtime-as-built)）：`Ltx25Family`
驱动视频管线的两个阶段（Ingredients 为单阶段），`Ltx25Denoiser` 每步在后端的 tensor executor 上
运行一个图程序，`Ltx25LoraSet` 作为运行时的 adapter set，以精确 F32 未合并地应用。退役的
`Ltx25Pipeline`、`Ltx25Sampling`、直接去噪器与 `Ltx25LoraAdapter` 在冻结其输出后已删除。

证据（NVIDIA A40 48 GB，驱动 595.112，CUDA 13.2，.NET 10.0.404；Intel Xeon Gold 6342，
50 GB 容器；ggml `bb877c0c` 未修改；退役路径由 `c178bfc` 以相同 ggml 与检查点构建）：

- **冻结的去噪步。** 退役去噪器在合成检查点上的输出（官方 oracle 的四个用例，各再加一个覆盖全部
  适配投影的合成 rank-2 Ingredients 适配器）按通道冻结在
  `InferenceWeb.Tests/Fixtures/Ltx25/LegacyGoldens/`。运行时在 CPU 与 CUDA 两个通道上逐位相同
  （各 16 个输出），由 `Ltx25DenoiserTests.StepsGiveTheRetiredDirectDenoisersBitsOnItsLane` 保持。
- **真实检查点单步**（生成几何下的合成潜变量与上下文；因无法获得受限适配器，使用 480 个投影、
  rank-128 BF16 的合成 Ingredients 布局适配器）：CUDA 上 512×512×17 首阶段、图像引导的逐 token
  时间步、1024×1024×17 细化、带追加关键帧引导的 384×256×49 以及 768×448×121 Ingredients 阶段全部
  逐位相同。稳态单步：820/825、897/893、2954/2954、919/931 与 21838/21919 ms（退役/运行时）。
  释放设备副本后的首步：首阶段 2.45–2.63 s 对 2.37–2.49 s，细化 4.65–4.89 s 对 4.51–4.72 s。
  CPU（两个块）两个用例逐位相同，2829/2548 与 3478/3154 ms。
- **完整片段**（`Ltx25AbProbe` 走公开模型 API，比较原始 RGB 与两个 PCM 声道）：每个用例的每一轮都
  逐位相同。用预构建二进制交替运行的墙钟时间（最终代码，各两轮）：

  | 片段（A40，`cuda`） | 退役 | 运行时 | 运行时/退役 |
  |---|---:|---:|---:|
  | t2v 1024×1024×17 | 33.86、32.43 s | 32.39、32.08 s | 0.973 |
  | i2v 1024×1024×17 | 32.26、31.87 s | 32.03、31.96 s | 0.998 |
  | 关键帧 768×512×49（三个引导） | 34.89、34.68 s | 34.68、34.61 s | 0.996 |
  | t2v 1024×576×121 | 91.32、89.98 s | 90.39、90.13 s | 0.996 |
  | Ingredients 768×448×121（两张参考） | 203.49 s | 204.03 s | 1.003 |

  各阶段（文本与图像编码、两个阶段、上采样器与解码）持平或更快。做到这一点靠三处改动：投影在首次
  乘积时上传，与此前排队的设备工作重叠，与退役去噪器一致；Philox 噪声与旋转表在多核上生成
  （位相同）；Ingredients 更新原地加到每个投影上。在这些改动之前的轮次最多慢 1.8%。
- **内存。** 单步峰值活动设备内存更低：2,063 对 2,345 MiB（细化）、6,192 对 6,684 MiB
  （Ingredients），不含权重。CUDA 内存池在该峰值处多缓存 0.3–0.4 GB 已释放的块，分配失败前会先归还，
  因此 Ingredients 阶段 nvidia-smi 读数为 23.9 对 23.4 GB；其它片段两棵树的峰值都是 18.2–22.9 GB
  （121 帧解码的瞬时峰值）。两片段进程的主机内存：匿名 3.56 对 3.61 GiB，文件映射均为 38.31 GiB。
- **CPU。** Xeon（7.65 核配额）上一个 256×256×9 片段在顺序相反的两组中都逐位相同：退役 411.7/405.3 s，
  运行时 415.5/403.9 s（两组合计 1.003×）。
- **测试。** LTX、扩散运行时、图与 LoRA 过滤集通过：CPU 1,637 个（跳过 34），CUDA 1,628 个
  （跳过 34）。跳过的是仅 MLX/Metal 的测试、GGML 后端区域测试，以及需要受限适配器、分词器参考或
  PyAV 图像参考的 oracle。

已移除（服务端与 CLI 启动时拒绝并说明替代）：`TS_LTX25_CUDA_ATTN`、`TS_LTX25_CUDA_FUSED_HEAD_GATE`、
`TS_LTX25_FUSED_ROPE`、`TS_LTX25_CUDA_TILE`、`TS_LTX25_CUDA_REQUIRE_TILE`、`TS_LTX25_CUDA_F16_DEQUANT`、
`TS_LTX25_STAGE_TIMING`、`TS_LTX25_CUDA_SOURCE_UPLOAD`，以及提示词编码器的 `TS_LTX25_CUDA_PREFAULT`、
`TS_LTX25_CUDA_PREFAULT_WORKERS`、`TS_LTX25_DEVICE_PROMPT_MATH`、`TS_LTX25_CUDA_BF16_GEMM` 与
`TS_LTX25_MLX_BF16_GEMM`。各自保留原默认行为，因此输出不变。`TS_LTX25_INGREDIENTS_LORA` 保留。
潜变量探针的 `--first-stage-image-strength` 与提示词探针按开关交替的轮次模式一并移除
（`Ltx25PromptProbe --rounds N` 重复编码）。

未验证：MLX 与 Metal 硬件（无可用 Apple 设备）；真实的受限 Ingredients 适配器（合成适配器具有相同
布局、形状与数据类型）；其它 GPU 与操作系统；逐位相同之外的感知质量（输出即同一片段）。

### 2026-10-17 的工作

2026-10-17 的工作为默认 1024×1024×17 配方（seed 42）去掉了权重加载与主机开销，
所有输出都没有改变。M5 Pro 各行使用端到端探针，用
`ltx25-compare-native-generations.py --require-bitwise` 比较原始 Float32 RGB 与两个
PCM 声道。A40 各行是交替两轮的完整 CLI 进程，四次运行的解码 RGB 与 PCM 全部相同。
M5 Pro 上的模型文件从冷缓存开始，A40 上未受控。

| 机器 / 后端 | 之前 | 之后 | 输出 |
|---|---:|---:|---|
| M5 Pro `mlx`，新进程 | 168.9 秒 | 108.0 秒 | RGB 与 PCM 逐位相同 |
| M5 Pro `mlx`，同一进程中第二个未缓存提示词 | 142.8 秒 | 110.3 秒 | RGB 与 PCM 逐位相同 |
| A40 `ggml_cuda`（direct CUDA），CLI 进程 | 163.9 秒 | 72.9 秒 | 解码 RGB 与 PCM 相同 |

改动如下：

- **按计算顺序预读权重。** M5 Pro 上文本编码在新进程中从 42.3 降到 5.5 秒，第二个提示词
  从 30.6 降到 5.2 秒；在 A40 pod 的网络卷上从 61.7–65.2 降到 13.0–13.5 秒。A40 的第一遍去噪（包含首次去噪器上传）
  从 39–44 降到 15–19 秒。
- **提示词 connector。** RoPE 表改为用相同表达式只计算一次，connector 权重按层序预读。
  A40 上 connector 从 12.3 降到 1.8 秒。
- **视频 VAE 与 upscaler 权重。** 权重转换分摊到多个核心，CUDA 主机缓冲写入前不再回读
  从未写过的设备副本。A40 上 upsample 从 9.0–11.6 降到 2.5 秒，视频解码从 12.7–15.4
  降到 6.9 秒；M5 Pro 上 upsample 从 2.3 降到 0.5 秒。
- **释放 Gemma 缓存。** 在 48 GB Mac 上，每个新提示词后释放 Gemma 的缓存页，避免 22 GiB
  过期页拖慢第一个 1024×1024 去噪步。与预读一起，M5 Pro 新进程的第一遍去噪从 64.8
  降到 43.5 秒。
- **权重转换循环。** 这些循环现在直接编译为优化代码，并使用块拷贝。

[eng/validation/README-ltx25.md](../../eng/validation/README-ltx25.md)
记录独立小夹具、实际权重、完整媒体、应用和计时的不同门禁。
图像条件验证使用未修改的 ggml
`3612c0af93f2cfc8f048295300a61187723b12ae`，原实现固定为
`be6d6d0a69d11a0a399c462ed92606692e5f8697`；干净 SD.cpp 对照为
`d637cfb935feb2254898a8f1524046c958f3df80`，自带 ggml
`d25b121ce1d6ae0c7e62ad080460d9fc1e46dd93`。

实际图像路径覆盖 Apple Silicon 和 A40 上首图、尾图、首尾图和三图条件，
以及 TensorAgent HTTP/SSE/正常聊天；真实 Release Mac Catalyst 探针另覆盖 Models 页 Use、
loopback 路由与 MP4/WAV 下载。512×512 训练编码器对官方 F32 潜变量的相对 L2 在两 GPU
上均不超过 `4.2e-6`。48 块 512×512×97 去噪器对未修改官方 F32 的相对 L2
为视频 `4.2e-5`、音频 `1.9e-6`，并非只覆盖小几何。

旧 512×512×17 同提示图生视频测量：Apple 首次/重复 126.85 / 39.81 秒，
SD.cpp 221.34 / 160.63 秒；A40 90.47 / 12.22 秒，SD.cpp 56.67 / 43.43 秒。
这些单图、单提示词/种子的模型生成数据早于当前两阶段修正，不能当作新默认配方基准；
首次加载、文件驻留、媒体编码、长片和应用仍有独立成本。
Ingredients 在两 GPU 上覆盖一图和同主体双视图的训练配方，包括实际上传与聊天；
全部 480 组 LoRA factor 与真实 rank-128 更新另有严格数值检查，
不消除已观察到的参考板复制问题。

曾有的 Linux CUDA 可选 pinned 源上传路径（单组 A40 对比快 8.45%，但从未成为默认）已随直接去噪器删除；
权重在文件缓存预读之后从映射的 checkpoint 上传。

独立夹具保存在 `InferenceWeb.Tests/Fixtures/Ltx25/`；工具在 `eng/validation/`。
生成证据在被忽略的 `docs/validation/ltx25/` 或 `artifacts/ltx25/`，不提交到 Git。
不可用设备、跳过、失败或中断场景不能计作通过；短片和个别视觉样本不证明广泛质量或性能持平。
