# Wan 视频生成（2.1 文生视频 · 2.2 文/图生视频）

[English](wan.md) | [中文](wan_zh-cn.md)

TensorSharp 原生运行 [Wan 2.1](https://github.com/Wan-Video/Wan2.1) 与
[Wan 2.2](https://github.com/Wan-Video/Wan2.2) 视频扩散模型：输入提示词
（Wan 2.2 模型可再加一张首帧图片），输出 H.264 MP4，`TensorSharp.Cli` 与
`TensorSharp.Server.Host`（OpenAI 风格 API + 自带 Web UI 聊天）均可驱动。

> **Wan 是只生成视频的家族。** TensorSharp 中更新的视频模型是
> [MiniMax-H3](minimax-h3_zh-cn.md)：它在**同一份打包潜变量**里把视频与
> **原生 32 kHz 立体声音频一起生成**——支持文生视频、图生视频、首尾帧与参考生视频，
> 并在 CFG-free 的 4–8 步下运行。只需要视频时仍然选 Wan，它也是拥有步数蒸馏检查点、
> 能把 100 次 DiT 前向的官方配方降到 4 次的那个家族。

支持的模型家族（按 DiT GGUF 自动识别）：

| 家族 | 潜空间 | VAE | 模式 | 说明 |
|---|---|---|---|---|
| Wan 2.1 T2V（1.3B / 14B） | 16 通道，8×8×4 | wan_2.1_vae | 文生视频 | 单 DiT |
| Wan 2.2 TI2V-5B | 48 通道，16×16×4 | Wan2.2_VAE | 文生 + **图生视频** | 稠密 5B，24 fps，可达 720p |
| Wan 2.2 A14B（T2V / I2V） | 16 通道（I2V 输入 36 通道） | wan_2.1_vae | 文生 + **图生视频** | 两个 14B 专家按时间步边界切换 |

图生视频时，上传的图片成为视频的**首帧**，文本提示词控制动作、镜头运动、
氛围与场景变化。

各网络在每次生成中协同工作，放在同一目录即可自动解析（`VAE/`、`HighNoise/`、
`LowNoise/` 等子目录亦可）：

| 组件 | 文件 | 来源 |
|---|---|---|
| DiT（`--model` GGUF，`general.architecture = wan`） | 如 `Wan2.2-TI2V-5B-Q8_0.gguf` | [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF)、[QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF)、[city96/Wan2.1-T2V-14B-gguf](https://huggingface.co/city96/Wan2.1-T2V-14B-gguf) |
| A14B 第二专家（仅 A14B） | 对应的 `…HighNoise…`/`…LowNoise…` GGUF | 同一仓库（在同级或兄弟目录中按文件名自动查找；`TS_WAN_DIT2` 可覆盖） |
| UMT5-XXL 文本编码器 | `umt5-xxl-encoder-Q8_0.gguf` | [city96/umt5-xxl-encoder-gguf](https://huggingface.co/city96/umt5-xxl-encoder-gguf)（`--video-text-encoder` / `TS_WAN_TE`） |
| Wan 2.1 视频 VAE（2.1 + A14B） | `wan_2.1_vae.safetensors` | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors)（`--video-vae` / `TS_WAN_VAE`） |
| Wan 2.2 视频 VAE（TI2V-5B） | `Wan2.2_VAE.safetensors` | [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/tree/main/VAE) 内附 |

文本编码器在去噪开始前、（图生视频时）VAE 编码器在 DiT 加载前、DiT 在 VAE 解码前
分别释放显存，因此峰值显存约为 `max(TE, DiT + 注意力, VAE)` —— TI2V-5B Q8_0
可在 16 GB GPU 上不到 8 分钟生成 81 帧 480p 图生视频；两个 A14B Q4_K_M 专家
可在同一张卡上顺序运行。

### 最快的上手方式：使用配置文件

视频生成需要多个网络协同，因此最省事的做法是使用
[`config/`](../../config/README.md#video-generation-video-only-wan) 中现成的配置文件——
它们会声明全部网络，并在首次运行时自动下载缺失的文件：

```bash
# 文生视频 + 图生视频，4 步蒸馏（首次运行约下载 12.9 GB）
TensorSharp.Server.Host --config config/wan-video-ti2v-5b-turbo.json

TensorSharp.Cli --config config/wan-video-ti2v-5b-turbo.json \
  --prompt "a red fox trotting through falling snow" --output fox.mp4

# 图生视频：图片成为首帧，提示词控制动作
TensorSharp.Cli --config config/wan-video-ti2v-5b-turbo.json \
  --image first_frame.png --prompt "she turns and smiles" --output clip.mp4
```

| 配置文件 | DiT | VAE | 文本编码器 | 支持模式 | 首次下载 |
|---|---|---|---|---|---|
| `wan-video-ti2v-5b-turbo.json` | TI2V-5B Turbo（4 步） | Wan 2.2 | UMT5-XXL | 文生 + 图生 | 约 12.9 GB（DiT 5.4 GB、UMT5 6.0 GB、Wan 2.2 VAE 1.4 GB） |
| `wan-video-ti2v-5b.json` | TI2V-5B（50 步） | Wan 2.2 | UMT5-XXL | 文生 + 图生 | 约 12.9 GB |
| `wan-video-i2v-a14b.json` | A14B 高噪 **+** 低噪 | Wan 2.1 | UMT5-XXL | 仅图生 | 约 25.6 GB |

模型存放位置由 `TENSORSHARP_MODELS` 环境变量决定；未设置时存放在仓库根目录下的
`models/` 目录（配置会相对 `config/` 解析 `../models`，即 `<仓库>/models`）。配置文件中不含任何绝对路径，因此同一个文件在 Windows、Linux 与
macOS 上都能直接使用，无需修改。

**两个 VAE 不可互换**：TI2V-5B 需要 Wan 2.2 VAE（48 通道潜空间，16×16×4 压缩），
而 A14B 与 Wan 2.1 系列需要 Wan 2.1 VAE（16 通道，8×8×4）。装错会在启动时报错，
而不是生成错误的视频。UMT5-XXL 为所有 Wan 模型共用，只需下载一次。

若要指向已有文件，使用 `--video-vae`、`--video-text-encoder`，以及（A14B）`--video-dit2`；
或把它们与 DiT 放在同一目录，由同目录扫描自动找到。

## 后端

除 MLX 外，视频生成在 TensorSharp 的每个后端上都可运行（`--backend mlx` 会在
加载时被拒绝）：

| `--backend` | 路径 | 说明 |
|---|---|---|
| `ggml_cuda` | GGML 整图内核 | 最快；常驻 DiT 图 + CUDA graph 捕获 |
| `ggml_metal` | GGML 整图内核 | Apple Silicon；VAE 卷积走 MPSGraph |
| `ggml_vulkan` | GGML 整图内核 | 通过 Vulkan 支持 AMD/Intel/NVIDIA；VAE 保持分带卷积路径（Vulkan 驱动拒绝多 GB 显存竞技场） |
| `ggml_cpu` | GGML 整图内核 | 纯 CPU 机器；慢但精确 |
| `cuda` | 直连 CUDA（`WanDirect*`） | 不依赖 ggml：量化权重常驻，走 TensorSharp 自有的 MMQ/dp4a/cuBLAS 路由、流式 online-softmax 注意力内核、channels-last im2col VAE |
| `cpu` | 直连纯 C#（`WanDirect*`） | 除托管运行时外无任何原生依赖；并行 SIMD GEMM/注意力内核 |

所有后端产生数值等价的视频（互相之间以及与 diffusers 参考实现在相同种子/嵌入下
交叉验证 —— 后端之间最终潜变量余弦相似度 ≥ 0.999，量化检查点相对 bf16 diffusers
参考约 0.994）。`cpu` / `ggml_cpu` 后端只用于功能性验证 —— 在高核心数机器上，
一段数秒的 480p 视频需要几十分钟。

参考耗时（RTX 2000 Ada 16 GB，Wan2.1-1.3B F16，832×480，33 帧，30 步 UniPC ——
官方 480p 配方）：`ggml_cuda` 12.0 s/步（总计 445 s）、`ggml_vulkan` 17.2 s/步
（625 s）、直连 `cuda` 19.3 s/步（700 s）。在较短序列上直连后端与 `ggml_cuda`
持平（TI2V-5B 480×480×9 帧：两者均为 1.1 s/步）；到 14k token 的配方时，它的
F32 注意力路径落后于 ggml 的 F16 flash attention。长序列注意力会自动走分块
cuBLAS GEMM+softmax（`TS_WAN_DIRECT_FUSED_ATTN=0` 可强制全程使用）。

## CLI

文生视频（任意家族）：

```bash
TensorSharp.Cli --model Wan2.2-TI2V-5B-Q8_0.gguf --backend ggml_cuda \
  --prompt "a cute fluffy orange cat walking through a sunny garden with flowers" \
  --output cat.mp4 --width 832 --height 480 --video-frames 49 --diffusion-seed 7
```

图生视频（Wan 2.2 模型，加 `--image`）：

```bash
TensorSharp.Cli --model Wan2.2-TI2V-5B-Q8_0.gguf --backend ggml_cuda \
  --prompt "the cat runs toward the camera, dynamic tracking shot, golden hour" \
  --image first_frame.png --output cat_run.mp4 --video-frames 81
```

```bash
# A14B：--model 指向任一专家即可，另一专家自动配对
TensorSharp.Cli --model HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf \
  --backend ggml_cuda --prompt "the ship sails into the storm, waves crashing" \
  --image ship.jpg --output ship.mp4
```

- `--image <文件>` —— 首帧条件图（PNG/JPEG/WebP/…），自动应用 EXIF 旋转。
  未显式指定 `--width/--height` 时，输出分辨率按图片纵横比适配模型原生面积
  （TI2V-5B：1280×704，即官方 720p 配方；其余：832×480）。
- `--video-frames N` —— 帧数，对齐到 `4k+1`；`--video-frames 1` 生成静态图
  （配合 `--output out.png`）。
- `--sampler unipc|euler` —— UniPC（默认）为官方 Wan 采样器，已与 diffusers 的
  `UniPCMultistepScheduler` 做数值对照验证。
- `--cfg` / `--flow-shift` / `--diffusion-steps` 默认为各模型官方配方：TI2V-5B
  cfg 5.0、shift 5.0、50 步、24 fps；A14B I2V cfg 3.5（双专家）、shift 5.0、40 步；
  A14B T2V cfg 4.0/3.0、shift 12.0；Wan 2.1 cfg 6.0、shift 8.0（1.3B 视频）或
  3.0/5.0、30 步、16 fps。
- `--negative-prompt` 默认使用官方 Wan 负面提示词。
- **步数蒸馏检查点会被自动识别**，也是目前最大的提速杠杆 —— 100 次 DiT 前向
  变成 4 次，视频内容不变。无需任何开关：它就是一个普通的 `--model` GGUF。
  详见下方[快车道](#快车道步数蒸馏检查点)。
- `--cfg-cache-stride N`（默认关闭）—— 近似提速。一次带引导的步进为
  `v = v_cond + (cfg-1)·d`，其中 `d = v_cond - v_uncond`；引导方向 `d` 在整个
  调度上的变化远慢于 `v`，因此无条件前向可以每 `N` 步只跑一次，中间复用缓存的
  `d`。50 步时，`2` 只跑 100 次前向中的 77 次（快 1.30×），`3` 跑 70 次（1.43×）。
  前三步与最后一步始终重新计算 `d`。需要与参考样本严格对齐时请关闭它。
  服务端字段：`"cfgCacheStride": 2`。
- MP4 输出优先使用 `PATH` 上的 `ffmpeg`（或 `TS_FFMPEG=<path>`，或可执行文件旁的
  `ffmpeg` 目录）—— 近无损 CRF 17 H.264。没有 ffmpeg 时回退到系统编码器
  （OpenCV），默认码率画质明显更差，CLI 会给出警告。

### 快车道：步数蒸馏检查点

这就是一段 5 秒 720p 视频花三个半小时还是十七分钟的区别，而代价只是改一下
文件路径里的一个词。

TensorSharp 读的是 DiT GGUF 的**文件名**。文件名中含有 `Turbo` / `distill` /
`Lightning` / `lightx2v` / `FastWan` / `-dmd`，或含有显式步数（`…-4steps-…`、
`…8step…`，取值 1–16 有效）时，采样默认值会切换到该步数并**关闭引导（CFG）**
—— 这正是这些检查点训练时的设定。加载时控制台会确认：

```
step-distilled checkpoint detected -> 4 steps, guidance off (--diffusion-steps / --cfg override)
```

官方 50 步 + CFG 的配方需要 50 × 2 = 100 次 DiT 前向；一个 4 步、无引导的检查点
只需 **4** 次 —— 去噪工作量只有 1/25。其余一切（配套文件、参数、端点、Web UI）
都不变，`--diffusion-steps` / `--cfg` 仍可覆盖被识别出的默认值。

可直接使用的 GGUF。蒸馏仓库只提供 DiT，VAE 与文本编码器仍从基础仓库获取：

```bash
# TI2V-5B，4 步 —— 注意文件名里是 Wan2_2（下划线）
hf download hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF Wan2_2-TI2V-5B-Turbo-Q8_0.gguf --local-dir models
hf download QuantStack/Wan2.2-TI2V-5B-GGUF VAE/Wan2.2_VAE.safetensors --local-dir models
hf download city96/umt5-xxl-encoder-gguf umt5-xxl-encoder-Q8_0.gguf --local-dir models

TensorSharp.Cli --model models/Wan2_2-TI2V-5B-Turbo-Q8_0.gguf --backend ggml_metal \
  --prompt "the cat runs toward the camera, cinematic" --image first_frame.png \
  --output cat.mp4 --video-frames 121
```

```bash
# I2V-A14B，4 步 —— 两个专家都要；--model 指向任一个，另一个按文件名自动找到
hf download jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF high_noise/wan2.2_i2v_A14b_high_noise_lightx2v_4step-Q4_K_M.gguf --local-dir models
hf download jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF low_noise/wan2.2_i2v_A14b_low_noise_lightx2v_4step-Q4_K_M.gguf --local-dir models
hf download QuantStack/Wan2.2-I2V-A14B-GGUF VAE/Wan2.1_VAE.safetensors --local-dir models
```

[hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF](https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF)
还提供 Q6_K（4.22 GB）直到 Q2_K（1.86 GB）；
[jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF](https://huggingface.co/jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF)
已把 Lightning 蒸馏合并进两个专家，提供 Q8_0（每专家 15.4 GB）直到 Q2_K（5.31 GB）。
[Green-Sky/FastWan2.2-TI2V-5B-FullAttn-GGUF](https://huggingface.co/Green-Sky/FastWan2.2-TI2V-5B-FullAttn-GGUF)
是 TI2V-5B 的另一个选择。注意
[lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning)
只发布 LoRA `.safetensors`，而 TensorSharp 没有 Wan LoRA 选项 —— 请使用上面这些
已经合并好的 GGUF。（`--lora` 只作用于 Qwen-Image-2.1：CLI 遇到 Wan 模型会拒绝它，
服务端则记录一条警告，并在不带它的情况下加载模型。）

### 获得好画质

- **分辨率是最大的杠杆。** Wan 的训练分辨率为 480p（832×480 面积）与
  720p（1280×704）；TI2V-5B 原生是 720p 模型。远低于 480p 面积时 DiT 处于
  分布外，无论多少步都会输出模糊、不稳定、偏色的视频 —— 低于约 0.3 MP 时
  管线会打印警告。请在受支持的分辨率下生成后再缩小（例如用
  `--width 480 --height 704` 而不是 320×480）。
- **把镜头描述完整。** Wan 对详细提示词（主体、动作、运镜、光线、风格）的
  跟随远好于三四个词的短语；官方演示会先用 LLM 做“提示词扩写”。过短的
  提示词让模型缺乏约束、容易漂移。
- 超过官方配方（50）继续加 `--diffusion-steps` 收益很小；优先解决分辨率
  与提示词。

## 服务器

```bash
TensorSharp.Server.Host --model Wan2.2-TI2V-5B-Q8_0.gguf --backend ggml_cuda \
  --video-frames 121 --fps 24
```

`--video-frames` 与 `--fps` 为 Web UI 和下述三个视频端点设置服务启动默认值。
Web UI 不会发送这两个字段，因此上例会以 24 fps 生成 121 帧（播放时长约 5.04 秒）。
API 请求显式提供的 `frames` 或 `fps` 会分别覆盖对应的启动默认值；这些值是默认值，
不是上限。如果启动参数和请求字段均省略，则采用模型配方：TI2V-5B 为 49 帧 / 24 fps，
其余模型为 33 帧 / 16 fps。帧数会对齐到 `4k+1`。要调整时长，建议保持模型原生
帧率并修改帧数；仅改变 FPS 会改变播放速度。`--video-width` / `--video-height`
（也可写作 `--width` / `--height`）与 `--video-steps` 同样是服务端全局默认值：请求中的
`width` / `height`（或 `size`）与 `steps` 会覆盖它们，都未设置时采用模型配方。

步数蒸馏快车道是 `--model` GGUF 自身的属性，而不是请求字段，因此它对 Web UI 和
所有端点都生效，客户端无需做任何改动 —— 把 `--model` 指向 Turbo/Lightning
检查点，其余一切照旧：

```bash
TensorSharp.Server.Host --model Wan2_2-TI2V-5B-Turbo-Q8_0.gguf --backend ggml_metal \
  --video-frames 121 --fps 24
```

这一点在浏览器里最重要：那里的请求只是一句提示词加一张图片，代价在 ETA 出现
之前完全看不见。同一个 Web UI 请求 —— 1024×768 的首帧、24 fps 下 121 帧，
解析为 1088×832、27 404 个 token —— 在基础检查点 `Wan2.2-TI2V-5B-Q8_0.gguf`
上约需 **3 小时 30 分**，在 `Wan2_2-TI2V-5B-Turbo-Q8_0.gguf` 上只需
**17 分 30 秒**（M5 Pro，`ggml_metal`；见[性能](#性能)）。如果某次运行的 ETA
看起来离谱，先到启动日志里找 `step-distilled checkpoint detected` 那一行，
再去动别的参数。

- **Web UI**（`http://localhost:5000`）：在聊天框输入提示词 —— 附带一张图片即为
  图生视频（图片成为首帧）；回复即是带实时去噪进度的生成视频。
- **API**：

```bash
curl http://localhost:5000/v1/videos/generations -H "Content-Type: application/json" -d '{
  "prompt": "猫向镜头跑来，电影感",
  "image": "data:image/png;base64,....",
  "size": "832x480", "frames": 81, "steps": 50, "seed": 7
}'
# -> { "created": ..., "data": [ { "url": "/uploads/video-....mp4" } ], "frames": 81, ... }
```

  `image` 接受 base64（可带 `data:` URL 前缀），省略即为文生视频。另有
  `POST /api/video-generate`（同样的 JSON，返回 `{ ok, url, ... }` 信封，并支持
  `imagePath` 引用已上传文件）与 `POST /api/video-generate/stream`（SSE 进度，
  Web UI 使用）。A14B 在各端点均支持 `cfg2`（低噪专家引导系数）。

## 实现说明

每个网络每次调用都作为**一张**权重常驻的 ggml 图运行
（`TensorSharp.GGML.Native/ggml_ops_wan.cpp`）：

- `TSGgml_WanT5Encode` —— 24 层 UMT5 编码器，带逐层相对注意力偏置；输入 token，
  输出隐藏状态（条件按 Wan 的语义零填充到 512 token）。
- `TSGgml_WanDitForward` —— 一步去噪的速度预测；在 CUDA 上按形状常驻，使 ggml 的
  CUDA graph 捕获可以消除启动开销；3D-RoPE 自注意力走 flash attention。对
  TI2V-5B 的图生视频，该图会在两个不同时间步上调制两段 token（条件帧的 token 位于
  t=0，即 diffusers 的 `expand_timesteps` 语义）—— 注意力仍在完整的联合序列上进行。
- `TSGgml_WanVaeDecode` —— 因果 3D 视频 VAE 解码器（两代通用），在图内按时间分块
  迭代，因果特征缓存在块之间传递；卷积在一个显存预算下以分带 im2col+GEMM 执行，
  使峰值显存有界（`TS_WAN_VAE_GEMM_MAX_MB`；CUDA 上为 384 MB —— 按可用显存推算
  实测**慢 2.2 倍**，因为 scratch 会吃掉图中其余部分所需的余量，WDDM 再把溢出部分
  换页 —— Metal 上为可用显存的 1/8、限制在 384 MB–8 GB 之间，因为它没有 PCIe 断崖；
  其他 GGML 后端为 384 MB；`0` 强制直接卷积）。在 Metal 上，这个预算只在
  `TS_WAN_VAE_MPS_CONV=0` 时生效，因为默认由 MPSGraph 整体执行每个卷积；Metal 4
  tensor API 生效时（M5 级设备上运行 14B 级模型）默认为 `0`。因果卷积的 kd 个时间
  抽头共用**一次** im2col，而不是把同样的像素各下降一遍；
  条带也按数量均分，而不是按固定高度往下走；两者都逐位一致。跨块特征缓存以 F16
  存储。Wan 2.2 解码器额外包含
  无权重的 DupUp3D 残差捷径与最后的 2×2 像素 unpatchify。超过每条带的像素预算时
  （GGML 后端随可用显存缩放：16 GB 可用时约 0.64 MP，最低 0.16 MP；直连
  `cuda`/`cpu` 后端为 0.3 MP），解码
  会进一步切成全宽横向条带，并在 8 个潜变量行的重叠区上做混合（即 diffusers 的
  `enable_tiling` 方案，相对不分块解码约 59 dB）：否则一个 720p 平面的激活加上
  因果缓存约占 12 GB 设备显存，会把 16 GB 的 WDDM 卡推入共享内存分页
  （69 帧 720p 从 33 分钟降到约 3 分钟）。`TS_WAN_VAE_TILE=0` 关闭分块。
- `TSGgml_WanVaeEncode` —— 图生视频条件背后的因果 3D VAE 编码器（两代通用）：
  1+4k 像素帧分块、带因果缓存的步长 2 空间与时间下采样、AvgDown3D 捷径，以及
  Wan 2.2 的 2×2 像素 patchify。

图生视频的条件注入严格遵循参考管线：TI2V-5B 每一步都用 VAE 编码后的图片替换
第一个潜变量帧，并把它的 token 在时间步 0 上调制；A14B 则在 16 个噪声通道之后
拼接 4 通道首帧掩码与 VAE 编码的 `[image, zeros…]` 片段（36 通道 DiT 输入），
并在 `t < boundary·1000`（I2V 0.9 / T2V 0.875）处从高噪专家切换到低噪专家，
切换时释放第一个专家占用的显存。

数值由带守卫的测试对照参考实现验证（`InferenceWeb.Tests/WanVideoOracleTests.cs`，
测试夹具由 `InferenceWeb.Tests/tools/wan22_oracle.py` 生成）：TI2V-5B 的 DiT（Q8_0）
在统一时间步与逐 token 时间步两种模式下与 diffusers 的 `WanTransformer3DModel`
余弦相似度 > 0.995；两个 VAE 编码器与 `AutoencoderKLWan` 余弦相似度 > 0.999；
Wan 2.2 VAE 解码 > 35 dB PSNR（Wan 2.1 解码：59.9 dB）；分词器在英文与中日韩
提示词上与 HuggingFace T5 的 id 完全一致。

环境变量开关：`TS_WAN_DIT_CAPTURE=0`（关闭常驻捕获的 DiT 图）、
`TS_WAN_DIT_FLASH=0`（改用材料化注意力）、`TS_WAN_DIT_KV_F16=0`（注意力 K/V 用
F32 —— 旧的、约慢 2 倍的默认值）、`TS_WAN_HEARTBEAT_S`（进度心跳间隔，默认 30，
`0` 关闭）、`TS_WAN_VAE_GEMM_MAX_MB`（im2col 预算）、`TS_WAN_DIT_FFN_CHUNK_MB`
（DiT 前馈按 token 分块的预算，默认 256 MB；`0` 关闭分块）、`TS_WAN_VAE_TAP_SHARE=0`
（每个时间抽头各做一次 im2col，而不是共用一次 —— 仅用于 A/B；共用的下降逐位一致）、
`TS_VAE_CUDNN_CONV=1`（在 CUDA 上为 VAE 卷积启用 cuDNN；默认关闭，因为它在短片段上快
约 1.2 倍、在 121 帧片段上慢约 4.5 倍 —— 每个卷积都在图外运行，主机往返次数随时间分块数
增长）、`TS_CUDNN_DIR`（要使用的 cuDNN 安装目录）、`TS_WAN_VAE_MPS_CONV=0`
（Metal 上改用 ggml 的卷积下降而非 MPSGraph）、
`TS_WAN_METAL_TENSOR_API=1|0`（强制开/关 Metal 4 tensor API；14B 级 DiT 默认开启，
更小的模型默认关闭 —— 见[下文](#metal-4-tensor-api以及不该怎么测它)）、
`TS_WAN_VAE`/`TS_WAN_TE`/`TS_WAN_DIT2`（配套文件路径）、`TS_FFMPEG`（MP4 导出用的
ffmpeg 路径）、`TS_WAN_DIT_TRACE=<file>`（逐阶段激活统计，用于调试）。

## 性能

**一句话结论：用步数蒸馏检查点。** 本页其它每一个杠杆都只值几十个百分点，
这一个值 25 倍。同一个 5 秒 720p 图生视频请求，在同一台机器、同样的参数下，
基础检查点要 **3 小时 30 分**，Turbo 检查点只要 **17 分 30 秒** ——
去哪里下载见[快车道](#快车道步数蒸馏检查点)，拆解见下面的 M5 Pro 表格。

RTX 3080 Laptop 16 GB（Windows/WDDM），ggml_cuda：

| 模型 / 工作负载 | 文本编码 | 图像编码 | 去噪 | VAE 解码 | 总计 |
|---|---|---|---|---|---|
| **Wan2_2-TI2V-5B-Turbo Q4_0** 图生视频 1088×832×121 帧、4 步（720p，5 秒） | 5.9 s | 6.1 s | 27.8 s/次前向（27 404 token） | 305 s | **429 s** |
| **Wan2.2-TI2V-5B Q8_0** 图生视频 832×480×81 帧、30 步 | 3.5 s | 8.5 s | 11.5 s/步（CFG ×2，8190 token） | 103 s | 464 s |
| Wan2.2-TI2V-5B Q8_0 文生视频 640×384×25 帧、20 步 | 3.7 s | — | 1.5 s/步 | 23 s | 65 s |
| Wan2.2-I2V-A14B Q4_K_M 480×480×9 帧、6 步（冒烟） | 3.4 s | 5 s | 7–8 s/步 + 两次专家加载 | 7.4 s | 89 s |
| Wan2.1-T2V-1.3B F16 832×480×33 帧、30 步 | 3.1 s | — | 9.4 s/步（CFG ×2） | 51 s | 337 s |

那一行 720p/121 帧在 2026-08 这轮优化之前是 **1 143 s**：DiT 图在 27 404 token 下需要
11.4 GiB，而这张卡只有 16 GB，整个去噪都跑在装不下的工作集上。对前馈按 token 分块
（`TS_WAN_DIT_FFN_CHUNK_MB`）把 1.5 GiB 的中间结果移出峰值，每次前向从 110.4 s 降到
26.1 s；限制 VAE 的 im2col scratch 并在因果抽头之间共用一次下降，把解码从 712 s 降到
305 s。两者都是精确的：DiT 与 diffusers 的余弦仍为 0.99997，解码仍为 80.2 dB PSNR，
与之前相同。

TI2V-5B 的 16×16 空间压缩使同分辨率下 DiT token 数约为 Wan 2.1 的 1/2.7 ——
它是消费级 GPU 上速度最快、质量最高的选择，也是唯一的 720p-24fps 模型。

### 长序列：开销随 token 数平方增长

token 数为 `潜变量帧数 × (h/2) × (w/2)`，而 DiT 自注意力的开销是 `O(token²)`。
超过几千 token 后，时间就花在注意力上而不是权重矩阵乘上，因此帧数和帧面积
的影响都远大于步数：

| 请求（TI2V-5B） | token | 注意力工作量 |
|---|---|---|
| 640×384×25 帧 | 1 200 | 1× |
| 832×480×81 帧 | 8 190 | 47× |
| 1088×832×121 帧（24 fps 下 5 秒，720p 级） | 27 404 | 520× |

完整的 5 秒 720p 配方确实是一个大任务：50 步 × 2 次 CFG 前向 = 100 次 DiT 前向，
每次 27k token。管线会打印 token 数、每次前向的耗时和滚动 ETA，并在一次前向
进行中每 30 秒心跳一次，让长时间运行看起来是在推进而不是卡死：

```
Wan 2.2-TI2V I2V: 1088x832x121f (31x26x34 = 27404 tokens), 50 steps, unipc, cfg 5, shift 5, seed ...
  [wan] large request: 100 DiT passes over 27404 tokens. Self-attention costs O(tokens^2) ...
  step 1/50 t=999.0 (249.0s: cond 124.6s + uncond 124.4s) — ~3h 24m left
  [wan] …denoise step 2/50 (cond pass) 60s in this pass, 315s total, ~3h 24m left
```

要让这样的请求更便宜，按效果排序：

1. **换用步数蒸馏检查点。** 这一项碾压其它所有手段：100 次 DiT 前向变成 4 次。
   见[快车道](#快车道步数蒸馏检查点) —— TensorSharp 按文件名自动识别并切换配方。
2. **减少帧数。** 121 → 61 帧大约把注意力工作量降到 1/4，VAE 解码减半；
   24 fps 下视频从 5 秒变成 2.5 秒。
3. **减小帧面积**，但不要低于 Wan 的训练分辨率 —— 低于约 0.3 MP 时模型处于
   分布外，结果会*更差*而不只是更便宜。`--width 480 --height 704` 是不错的
   竖屏目标。
4. **减少步数**（仅限基础检查点）。50 是官方 TI2V 配方；30 步观感非常接近，
   便宜 1.7×。
5. **`--cfg-cache-stride 2` 或 `3`** —— 通过在步之间复用引导方向获得
   1.30× / 1.43×（近似方法，见 CLI 一节）。对蒸馏检查点没有意义，它本来就
   不用引导。

M5 Pro（20 核 GPU，48 GB 统一内存），`ggml_metal`，Wan2.2-TI2V-5B Q8_0，
1088×832×121 帧 = 27 404 token，图生视频 —— 下文所有优化都是针对这个请求
测量的（24 fps 下 5 秒，720p 级）：

| | 基础检查点，优化前 | 基础检查点，现在 | **Turbo 检查点，现在** |
|---|---|---|---|
| DiT 前向次数 | 100（50 步 × CFG） | 100 | **4**（4 步，无引导） |
| 每次前向 | 206.2 s | **120.2 s** | **120.2 s** |
| 去噪总计 | 20 615 s | 12 020 s | **481 s** |
| VAE 解码，121 帧 | 863 s | **563 s** | **563 s** |
| **端到端** | **≈ 5 小时 58 分** | ≈ 3 小时 30 分 | **17 分 30 秒** |

这两个杠杆是独立的：单次前向约 1.7× 来自下文的注意力与 VAE 改动，前向次数
25× 来自步数蒸馏检查点。一旦蒸馏，**瓶颈就变成 VAE 解码**（约占整轮的 55%），
不再是 DiT。

之后就由帧数和帧面积决定其余部分：VAE 随输出像素线性增长，注意力随 token 数
平方增长。以下均在同一台机器、用同一个 Turbo 检查点和同一张图片测得：

| 输出 | token | 去噪 | VAE 解码 | **总计** |
|---|---|---|---|---|
| 736×544 × 81 帧（3.4 秒，480p 级） | 8 211 | 84 s | 159 s | **4 分 09 秒** |
| 736×544 × 121 帧（5 秒，480p 级） | 12 121 | 137 s | 237 s | **6 分 19 秒** |
| 1088×832 × 121 帧（5 秒，720p 级） | 27 404 | 481 s | 563 s | **17 分 30 秒** |

480p（约 0.4 MP）是 Wan *训练过*的分辨率，所以中间那几行是分布内的结果，
不是降级模式 —— 当你只有几分钟时间时，就该选它。真正开始掉画质的是低于
约 0.3 MP。

单次前向的提速由三处改动贡献。

**注意力的 K/V 用 F16。** 每个后端的 flash attention 内核都是围绕 F16 KV 缓存
设计的：ggml-metal 的 F32 变体用 `simdgroup_float8x8` 累加器，而 F16 变体用
`simdgroup_half8x8`；在一个每 8 个 query 线程组就要重新流式读一遍 K 和 V 的
内核里，F32 分块还要付双倍带宽。转成 flash attention 布局的 permute 与降精度
合并在一次 `ggml_cpy` 中完成，所以 F16 路径写出的字节数也只有旧的 `ggml_cont`
的一半。`TS_WAN_DIT_KV_F16=0` 可恢复 F32。

**VAE 解码的分带布局。** 分块现在从显存预算推出条带的*数量*再均分平面，而不是
按固定条带高度往下走、让最后一条落在步长决定的任意位置。52 个潜变量行原来是
三条 24 行的带 —— 解码 72 行才得到 52 行 —— 现在在同样的每带预算下是两条 30 行
的带（共 60 行，接缝从两条减为一条）。

**Metal 上 VAE 卷积走 MPSGraph。** ggml 把 `conv2d` 下降为 im2col + `mul_mat`，
而一次解码剖析（`WanVideoBench`，640×480×9 帧）显示**图中 44.4% 在 MUL_MAT、
另有 30.2% 在 IM2COL** —— 74.6% 都在卷积里，im2col 每个节点搬运 865 MB，速率约
57 GB/s。这个下降过程纯粹是开销：一个 3×3 卷积在 GEMM 开始之前要先材料化出 9 倍
于输入的数据。Apple 调优过的卷积在同样的形状上快得多：

| 卷积形状 | ggml im2col+GEMM | MPSGraph | |
|---|---|---|---|
| 512→512 k3 320×240 t9 | 666 ms | 108 ms | 6.2× |
| 256→256 k3 640×480 t9 | 947 ms | 110 ms | 8.6× |
| 160→160 k3 640×480 t9 | 491 ms | 47 ms | 10.4× |
| 512→512 k1 320×240 t9 | 184 ms | 14 ms | 13.9× |

MPS 能跑到约 30 TFLOP/s，而 ggml 的 Metal GEMM 只有约 4.9 —— 而且它*不*经过
ggml 的 `mul_mm`，也就是那个 Metal 4 tensor 路径会把这张图算错的内核。它既拿到
了矩阵单元的吞吐，又绕开了那个缺陷。

布局是零成本的：ggml 的激活 `[W,H,C,T]` 与卷积核 `[KW,KH,IC,OC]` 逐字节就是
MPS 的 NCHW 与 OIHW，而且这里实测 NCHW 与 NHWC 一样快，所以不需要插入任何转置。
在这条路径上 VAE 直接发出未下降的 `CONV_2D` 节点，也就顺带完全跳过了横向分带
及其前置 padding 补丁；解码图随后被分段执行，每个 `CONV_2D` 交给 MPSGraph
（用 `ggml_graph_view`，与节点剖析器相同的机制，因此对 gallocr 的缓冲复用是
安全的）。端到端实测：在 736×544×81 帧下 **VAE 解码 159 s → 80 s（1.99×）**。
数值不变 —— ggml 本来就以 F16 卷积、F32 累加，这条路径同样如此，整轮解码一致到
**93.9 dB PSNR / 最大偏差 0.128（满量程 255）**。`TS_WAN_VAE_MPS_CONV=0` 可恢复
ggml 的下降路径。

**VAE 卷积的 im2col 预算与分块阈值本身，改由设备显存推算。** 这两者过去在所有
非 CUDA 后端上都被钉死在一张 16 GB 卡的预算上。Metal 现在按可用显存推算 im2col
预算（可用显存的 1/8、上限 8 GB）；CUDA 固定为 384 MB（按可用显存推算在那里实测
慢 2.2 倍），Vulkan（与 ggml_cpu 一样）固定为 384 MB（其驱动拒绝多 GB 的竞技场）。分块阈值则在
所有 GGML 后端上都随可用显存缩放。更大的 im2col 预算把许多
小的分带 GEMM 变成少量大 GEMM：1088×832×9 帧的解码从 56.4 s 降到 49.7 s。而且
分块并不免费 —— 条带彼此重叠，解码器实际跑的潜变量行数多于平面本身，而且条带
缓冲要与整幅画布共存。把 1088×832×121 帧**整幅**解码实测既更快又更省：
**565 s vs 655 s，峰值 RSS 4.85 GB vs 5.37 GB**。小显存的卡仍然分块。

#### 实测后被否决的方案

同一轮运行中，该形状（序列 27 404、24 个头、head dim 128）下的一次自注意力：

| KV 类型 | 耗时 | 相对 F32 |
|---|---|---|
| F32（之前） | 4993 ms | 1.00× |
| **F16（现在）** | **2467 ms** | **2.02×** |
| Q8_0 | 2652 ms | 1.88× |
| Q4_0 | 2395 ms | 2.08× |

Q8_0 的 K/V 比 F16 *更慢* —— 反量化的开销盖过了省下的带宽 —— 而 Q4_0 只多买到
3%，却要付出实实在在的精度代价，所以 F16 是最优点而不是折中。材料化的分块注意力
（`mul_mat` + `soft_max_ext`，与 flash attention 不同，它*可以*用 Metal 4 tensor
API）只有 1.08×：在这个规模上注意力受限于分数矩阵的访存流量，而不是 GEMM 速率。
把 DiT 权重从 Q8_0 反量化成 F16，矩阵乘只变化了不到 5%（一次注意力投影从
25.2 ms 到 26.4 ms）。

#### Metal 4 tensor API，以及不该怎么测它

这是这里最诱人的一个开关，而它对 TI2V-5B 保持**关闭**。它在 DiT 上实测 1.20×
（121.4 → 100.8 s/次前向）、在 VAE 解码上 1.66×，因为它把 `mul_mm` 路由到 Metal 4
的 tensor 操作 —— 但在 M5 / macOS 26.6 上，它同时会把 VAE 的卷积 GEMM 算错，
视频输出为一片**纯黑**。

对 A14B 等 14B 级 DiT（patch embedding 宽度 ≥ 5120），它在 M5 级 Mac 上**默认开启**，
因为那里 DiT 的收益更大：在 M5 Pro 上以 480×480×9 帧实测，A14B 图生视频为 17.1 vs
30.2 s/步（1.77×）（见 [Wan 视频与 tensor API](../../DEVELOPMENT_zh-cn.md#wan-视频与-tensor-api)）。
此时 VAE 的卷积不走 tensor API 的 `mul_mm`：默认由 MPSGraph 执行；设置
`TS_WAN_VAE_MPS_CONV=0` 时改走 ggml 的直接卷积，而不是 im2col+GEMM。
`TS_WAN_METAL_TENSOR_API=1` / `0` 可为整个进程强制开启或关闭 tensor API，覆盖按模型
规模选定的默认值。

陷阱在于：这个问题在隔离测试里复现不出来。`WanVideoBench vae-decode` 在五种潜变量
形状下跑过（包括被记录为最初全 NaN 复现场景的 32×32 布局），在 tensor API 开与关
两种情况下导出原始像素：两者一致到 **91–93 dB PSNR、最大偏差 0.18/255、零 NaN**。
基于这个证据，规避措施被移除了 —— 而紧接着的下一次完整 1088×832×121 帧生成就
产出了 121 张黑帧。这个缺陷跟随进程的分配历史（DiT 在解码之前被加载并释放），
所以一个全新进程里的解码探针会复现出错误的缓冲布局。

随后又用一个*更*贴近真实的测试台重测了一次 —— `WanVideoBench pipeline`，它在
同一个进程里按生产环境的确切形状依次跑 VAE 编码 → DiT 前向 → 释放 → VAE 解码，
包括完整的 121 帧解码。结果同样干净，而且快 1.50×（374 s vs 563 s）。可真实生成
出来的仍然是平坦的画面。唯一剩下的差别是 UMT5-XXL 文本编码器 —— 它在其它一切
之前加载并释放，是整条管线里最大的一次分配。

所以：重新启用它必须以一次完整的端到端视频为准，任何更弱的验证都从未足够。
管线现在会拒绝返回一段完全平坦的视频，并指出这个成因
（`AssertFramesAreNotDegenerate`）—— 正是它自动抓住了第二次尝试，省下了又一次
靠肉眼看 PNG 的代价。

#### 质量

F16 的 K/V 对 DiT 输出的改变，小于采样器自身对浮点重结合顺序的敏感度：

- 对照 diffusers `WanTransformer3DModel` 参考实现的一次 DiT 前向
  （`WanVideoOracleTests`）：F16 K/V 余弦 **0.999964**，F32 K/V 余弦 **0.999964**
  —— 单次前向的精度没有变化。
- 相同种子下的完整 25 帧生成：F16 与 F32 K/V 的平均 PSNR 差异为 39.08 dB。
  对照组 —— F32 K/V 且*关闭* flash attention，即同样的算术但求和顺序不同 ——
  差异为 39.40 dB。两者量级相同，说明像素差异是一条 8 步扩散轨迹因舍入差异而
  分叉的结果，而不是精度损失。

提速只在注意力占主导时出现：在 2 310 token（480×704×25 帧）下，F16 与 F32 K/V
每次前向都是 5.0 s。

Wan 2.1 对比 stable-diffusion.cpp（master-769，相同的 GGUF 与设置，33 帧 480p）：
采样阶段接近持平（281 s vs 258 s），但 sd.cpp 的 VAE 解码要材料化约 8 GB 的 3D
im2col，把一张 16 GB 的卡压进 WDDM 分页（1762 s vs **51 s**）—— TensorSharp
端到端快 6.0×。
