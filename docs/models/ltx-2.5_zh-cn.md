# LTX 2.5 Distilled

[English](ltx-2.5.md) | [中文](ltx-2.5_zh-cn.md)

TensorSharp 原生实现 LTX 2.5 Distilled 的文生视频、图像条件视频及联合生成的
**48 kHz 立体声音频**。TensorAgent 的实验条目为
**LTX 2.5 Distilled — Q4_K_M + BF16**，Download 下载五个固定文件，Use 激活模型。
48 GB 目录档位是保守起点，并非已证明的最低要求或任意分辨率/时长保证。

## 文件与启动

[config/ltx-2.5-distilled.json](../../config/ltx-2.5-distilled.json)
按提交和 SHA-256 固定五份文件，共 **44,764,412,884 字节（44.76 GB）**：
Q4_K_M 去噪器、带投影的专用 Gemma 4 编码器、卷积视频 VAE、x2 空间潜变量上采样器、
音频 VAE。官方伴随文件需要先获得发布方 Hugging Face 仓库访问许可。
可设置 `HF_TOKEN`、`HUGGING_FACE_HUB_TOKEN`，或使用 Hugging Face CLI token 缓存；
凭证只附在 HTTPS `huggingface.co` 请求上，CDN 重定向不携带凭证。

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
| `cuda` / `ggml_cuda` | TensorSharp 自有 CUDA 执行器 |
| `mlx` / `ggml_metal` | TensorSharp 自有 MLX 执行器 |
| `cpu` / `ggml_cpu` | Direct CPU 实现 |

不支持 Vulkan 或多 GPU。GPU 测量不证明 CPU 吞吐。
桌面 MP4 音轨需要 `PATH` 上的 FFmpeg 或 `TS_FFMPEG`，CLI 另写原始 PCM WAV；
Mac 应用通过 AVFoundation 编码。

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

CUDA 的 SM 8.0+ 提示词投影默认用 BF16 Tensor Core、F32 累加并保留 BF16 激活边界；
`TS_LTX25_CUDA_BF16_GEMM=0` 选择 F32 投影，较旧设备自动 F32。
CUDA 对映射投影做有界预读，每个 reader 一个 8 MiB 缓冲；Linux 检查驻留页，
至多八路读取 1 MiB 切片。`TS_LTX25_CUDA_PREFAULT=0` 关闭，
`TS_LTX25_CUDA_PREFAULT_WORKERS=1` 选择顺序读取。
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
`ReferenceImagePaths` 不可与其他图像输入混用。
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
或设置 `TS_LTX25_INGREDIENTS_LORA`。固定文件大小 1,308,787,472 字节，SHA-256 为
`ff873a5beada3c579a8137c7c53343916f78bcc9a529ba910143073fe8715e95`。
它不包含在五文件默认下载中；安装后 TensorAgent 提供 **Identity references (Ingredients)**。

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
[`Ltx25Pipeline`](../../TensorSharp.Models/Models/Ltx25/Ltx25Pipeline.cs) 负责两阶段采样；
[`潜变量上采样器`](../../TensorSharp.Models/Models/Ltx25/Ltx25LatentUpsampler.cs)
有 1024 个中间通道、前后各四个 ResBlock、Conv2d + PixelShuffle(2)，
GroupNorm(32) 在整段片上以 Float64 累积统计。解码使用精确扩宽的 BF16 权重与 F32 激活。

模型保留映射提示词/视频文件、去噪器设备权重身份、音频转换缓存与最多四组最终提示词上下文。
提示词逐投影、视频 VAE 逐卷积流式加载，新提示词结束后释放 connector 设备权重；
`TrimIdleMemory` 释放所有保留组件。组件内存数据不代表完整生成峰值。

最新源码把保持原顺序的 BF16 舍入与 Gemma split-half RoPE 放在 CUDA/MLX 设备执行，
三角函数仍按原 `MathF` 和 BF16 舍入产生；`TS_LTX25_DEVICE_PROMPT_MATH=0` 选择主机控制。
CUDA 融合 `2 * sigmoid(logit)` head gating，保留独立 F32 舍入边界，
`TS_LTX25_CUDA_FUSED_HEAD_GATE=0` 恢复分立操作。
MLX Q5_K 以有界 CPU 并行直接打包，保留所有量化值和缩放字节；
`TS_MLX_Q5K_REPACK=0` 选择旧转换。旧可选 PTX 缺少导出时会回退，
所以成功生成本身不证明实际用了新内核，严格测试还要求 native dispatch。

额外矩形注意力批量路径在组件上通过，却未通过完整生成 RGB/双声道 PCM 等价检查，
扩展与选择器已移除。保留已接受的提示词运算和 head gating 后，
三个远端完整生成用例重新得到逐比特相同的 RGB 和两个 PCM 声道。
组件微基准与打包耗时不能换算成模型整体提速。

## 验证与复现

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

Linux 可选 `TS_LTX25_CUDA_SOURCE_UPLOAD=1` 对冷映射页使用每组件两个 8 MiB pinned
缓冲；默认关闭。单组反序 A40 对比 77.015 → 70.507 秒，RGB/PCM 精确相同，
仍慢于记录的 SD.cpp 首次时间，且依赖存储/页驻留，不是通用提速结论。

独立夹具保存在 `InferenceWeb.Tests/Fixtures/Ltx25/`；工具在 `eng/validation/`。
生成证据在被忽略的 `docs/validation/ltx25/` 或 `artifacts/ltx25/`，不提交到 Git。
不可用设备、跳过、失败或中断场景不能计作通过；短片和个别视觉样本不证明广泛质量或性能持平。
