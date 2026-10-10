# Qwen-Image-2.1

[← 返回模型索引](README_zh-cn.md) | [English](qwenimage21.md)

Qwen-Image-2.1 通过 `QwenImageModel` 图像流水线完成文生图与图像编辑。它需要
Qwen3-VL-8B 文本编码器和专用的 2.1 VAE。原生 RGBA 输入与 PNG 输出会保留透明度；
Qwen3-VL 条件分支把参考图合成到白色背景上，而 VAE 保留 alpha 通道。更早的
Qwen-Image / Qwen-Image-Edit 检查点（例如 Qwen-Image-Edit-2511）已不再受支持，
加载时会被拒绝（退出码 2）。`--qwen-image-lora` 选项已由 `--lora` 取代（见
[LoRA 插件](#lora-插件)），`--offload-cpu` 选项已移除；无论写在命令行上还是作为配置文件
的键，这两个旧选项现在都会让 CLI 或服务端以配置错误退出，错误信息会说明应改用什么。
旧的 `TS_QWEN_IMAGE_LORA` 环境变量会在加载时被拒绝（退出码 2），并给出同样的建议：
改用 `--lora` 传入 LoRA，并取消设置该变量。

下载配置为 [`config/qwen-image-2.1.json`](../../config/qwen-image-2.1.json)。
它固定了仓库修订版本，并为新下载的文件校验 SHA-256；已缓存的文件会直接复用。
四个文件合计 11,051,668,216 字节（约 10.29 GiB）；这是下载大小，而不是推理时的
峰值内存。运行时内存还包括激活值、解码缓冲区和工作权重。更大的图像和多张参考图
都会提高内存需求。

| 组件 | 文件 | 来源 |
|---|---|---|
| 扩散 Transformer | `qwen_image_2.1_Q4_K_M.gguf` | [Abiray/Qwen-Image-2.1-GGUF](https://huggingface.co/Abiray/Qwen-Image-2.1-GGUF) |
| 专用 VAE | `qwen_image_2.1_vae_bf16.safetensors` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/vae) |
| 文本编码器 | `Qwen3VL-8B-Instruct-Q4_K_M.gguf` | [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) |
| 编辑用视觉编码器 | `mmproj-Qwen3VL-8B-Instruct-F16.gguf` | 同一个 Qwen3-VL 仓库 |

Qwen-Image-2.1-Turbo 是 Qwen 对同一模型的 8 步蒸馏版，使用这些伴随文件，搭配自己的
Transformer 与配置文件；见 [Qwen-Image-2.1-Turbo](#qwen-image-21-turbo)。

## 启动 CLI

以下命令都在 TensorSharp 仓库根目录下运行。该配置在 Apple Silicon 上选择
`ggml_metal`。在已构建 CUDA 后端的 NVIDIA 机器上，追加 `--backend ggml_cuda`；
原生 CPU 后端为 `ggml_cpu`，`--backend cpu` 则用纯 C# 运行整条流水线（见
[纯 C# CPU 后端](#纯-c-cpu-后端--backend-cpu)）。缺失的模型会在启动时自动下载。把 `TENSORSHARP_MODELS`
设为一个绝对路径即可选择存放位置，文件会放在它的 `qwen-image-2.1` 子目录中。
不设置该变量时，配置会相对 `config/` 解析 `../models`，即
`<仓库>/models/qwen-image-2.1/`。

对于下载到本工作区的模型，设置：

```bash
export TENSORSHARP_MODELS="$PWD/../models"
```

构建：

```bash
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release
```

文生图：

```bash
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1.json \
  --prompt 'A small orange cat beside a blue ceramic vase, soft daylight, detailed photograph' \
  --width 2048 --height 2048 --diffusion-steps 40 --cfg 1 \
  --diffusion-seed 42 --output generated.png
```

图像编辑：

```bash
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1.json \
  --image generated.png \
  --prompt 'Change the blue vase to a red vase. Preserve the cat, lighting and composition.' \
  --width 2048 --height 2048 --diffusion-steps 40 --cfg 1 \
  --diffusion-seed 42 --output edited.png
```

不带 `--image` 时执行生成；带一个或多个 `--image` 参数时执行编辑。重复
`--image first.png --image second.png` 可按该顺序传入多张参考图。每张参考图按命令行顺序在提示词之前
标记为 `<image1>`、`<image2>`……，提示词可以用这些标记指代图片。也可以用
`--input prompt.txt` 提供提示词。省略采样设置时使用 **40 步 Euler、CFG 1.0**，
遵循 [Qwen 推荐的无引导采样](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/qwenimage21.md)。
CFG 1 每一步只运行一次 Transformer 预测；此前的默认值 CFG 6 会同时运行正向和
负向两次预测。显式设置大于 1 的 CFG 仍会启用第二次预测，并以 `--negative-prompt`
作为它的条件（例如 `--negative-prompt 'blur, low detail'`；不指定时负向分支使用空提示词）。
在 CFG 1 下，负向提示词不起作用。

省略尺寸时，**生成默认为 2048×2048**；编辑则使用与第一张参考图宽高比一致、
像素面积大致相同的尺寸。在纯 C# 的 `cpu` 后端上，自动面积改为 1 百万像素（1024×1024），
因为在那里 2048×2048 的每一步要慢约 5 倍；`ggml_cpu` 与 GPU 后端保持 2048×2048。
要覆盖此行为，请同时设置宽度和高度，且都取 32 的倍数。
该模型支持[原生 2K 宽高比](https://github.com/QwenLM/Qwen-Image-2.1#supported-aspect-ratios)。
每张参考图以约 1 百万像素（若输出面积更小，则以输出面积）作为条件输入；把输出
提高到 2K 并不会同时把每张参考图的 VAE、视觉编码器和 Transformer 工作量翻四倍。

按面积决定尺寸的编辑会把比该面积大的图片变小、比它小的图片变大。要按图片自身尺寸编辑——从而
对编辑结果再次编辑时不缩小——请传 `--keep-source-size`（API 请求中为 `keepSourceSize: true`）：
输出为第一张图的精确宽高。采样面积约为图片自身面积——至少 1 百万像素（参考图的条件面积），至多该编辑
原本使用的面积——宽高比近似保持（二者都按 32 像素网格取整）；与源图尺寸不同时，结果再缩放回源图尺寸。它不能与显式宽高
同时使用，并且需要输入图。带遮罩的编辑（见下文）无需此参数即保持源图尺寸。服务端 Web UI 与
TensorAgent 的每次编辑都会发送它。初始噪声按编辑实际采样的尺寸生成（见[种子与编辑](#种子与编辑)）。

调度现在遵循
[官方调度器配置](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/scheduler/scheduler_config.json)：
使用指数动态偏移（序列长度/偏移锚点为 256/0.5 与 8192/0.9），随后做终端拉伸到
0.02，并以最后一个 Euler 步走到零。它取代了此前源自 Flux 的 4096/1.15 调度，
因此在这一修正之后，相同的种子可能生成不同的图像。

需要更快的草图时，指定 `--width 1024 --height 1024`。2K 正方形的潜在图像 token
数是 1K 正方形的四倍，注意力计算也更多。`--diffusion-steps 25 --cfg 1` 是一个
可选的更快配置，官方 ComfyUI
[生成](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json)
与[编辑](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_image_edit.json)
工作流使用的就是它。默认仍为 40 步；更少的步数是质量与速度之间的权衡，并不意味着
图像质量相当。

CFG 1 带来的提速直接来自已发布的 2.1 检查点。步数蒸馏 LoRA 插件可以把步数进一步降到
4–8 次 Transformer 前向；见 [LoRA 插件](#lora-插件)。

快速冒烟测试可以用 256×256、一步。这样的运行只验证加载和端到端数据通路，
并不能说明默认 2048×2048/40 步设置下的图像质量或性能。

### 种子与编辑

`--diffusion-seed`（API 中为 `seed`，默认 0）是生成初始噪声的 Philox 生成器的密钥。
文生图噪声只由种子和输出尺寸决定，与 stable-diffusion.cpp 的 `--rng cuda` 相同，
因此相同的请求得到相同的图像。

编辑的噪声还取决于参考图。TensorSharp 对传入的图片计算哈希（按顺序对 8 位 RGB 值，
以及存在不透明度不为满的像素时的 alpha，计算 SHA-256），并用这个哈希选出一条单独的
Philox 流来生成噪声；流水线的 `Qwen-Image-2.1: …` 日志行以 `(edit noise stream <id>)` 显示它。否则，
用画出某张图的种子和尺寸去编辑它时，会从恰好生成这张图的那份噪声开始，
Qwen-Image-2.1 可能照原样重描这张图而不执行指令。各个主机的默认值正会走到这一步：
请求不指定时种子为 0，编辑又按源图尺寸采样——自动尺寸在相同面积下沿用源图宽高比，而
`keepSourceSize`（Web UI 与 TensorAgent 会发送）把 1024×1024 的图仍按 1024×1024 采样。一只在种子 0 下画出的狐狸（1024×1024，
CFG 1），在种子 0 下要求“change the background to a sandy beach with the ocean behind
the fox”时，在 12 步、40 步以及使用加速 LoRA 时都变成过度锐化、仍在雪地里的图；
在种子 5 下画出并编辑的狐狸也一样，而同一只狐狸换一个种子编辑就得到了海滩。
这种重描是大概率而非必然：在 CFG 6 下从源图自身种子的噪声出发给茶壶换色依然成功
（见英文模型卡的历史对比）。

- 相同的图片、提示词、种子与设置得到相同的编辑结果；换种子得到另一种构图，与以前一样。
  改写提示词不改变噪声。
- Qwen-Image-2.1-Turbo 的编辑同样如此：流取决于图片，而不取决于检查点。
- 噪声按编辑采样的尺寸生成。使用 `keepSourceSize` 时，这是[上文](#启动-cli)所述的采样尺寸，
  两者不同时并非源图自身的尺寸；结果随后再缩放回源图尺寸。
- 在同一种子下再次编辑结果图时，噪声是新的，因为图片变了。
- 遮罩编辑使用整张源图的流，而不是选区或其裁剪的流，因此同一张图上的所有选区共用该图的流。
- 流取决于解码后的像素，内存中的图片与它保存的 PNG 哈希相同。TensorSharp 写出的每个 PNG，
  以及任何不带颜色信息的 PNG，在每个主机上都解码出相同的字节。PNG 嵌入了色彩配置文件时
  （macOS 与 iPhone 的截图都是如此），或者它没有 alpha 通道而 `gAMA` 或 `cHRM` 块描述的是
  其他色彩空间时，TensorAgent 应用会把它转换为 sRGB，CLI 和服务端则保留存储的值
  （`eng/validation/apple-png-decode-check.py` 会列出哪些 PNG 属于这种情况）。这样的 PNG 与
  JPEG 或 HEIC 照片（在应用中的解码可能相差几个色阶）一样，在应用中得到的流与 CLI 和服务端
  不同，于是得到另一种构图。跨主机比较这类图片的编辑时，请设置 `TS_QWEN21_EDIT_NOISE=seed`。
- `TS_QWEN21_EDIT_NOISE=seed` 让编辑使用该种子的文生图噪声（与 stable-diffusion.cpp 和
  diffusers 相同）：用于与这些引擎做噪声对齐的对比，以及复现这一改动之前的编辑结果。每次
  这样的编辑都会打印 `TS_QWEN21_EDIT_NOISE=seed: this edit starts from the seed's
  text-to-image noise …`。默认值为 `references`；其他取值会使请求失败。
- 唯一的开销是哈希。在 M5 Pro 上用 CoreCLR，Release 构建对 1 百万像素参考图耗时 2.6–3.9 ms，
  对 1200 万像素照片耗时 29–44 ms（同时哈希半透明 alpha 平面时取上限）；Debug 构建分别为
  16–26 ms 与 177–298 ms。以上为 `eng/validation/QwenImage21EditNoiseCost` 两次运行的范围
  （它的 `-p:ModelsDir` 可测量在别处构建的 TensorSharp.Models，例如 CLI 的）。文生图不做哈希。
  Apple 应用运行在 Mono 上，逐像素循环还要更慢；这一点未测量。

#### 实测效果

Apple M5 Pro（48 GiB，macOS 27.0），`ggml_metal`，未修改的 ggml
`ffa4e8b80930029a35991f94e7c8a93cd67730ab`。Q4_K_M 检查点用 12 步，Turbo AD-Q4_K 用它的 8 步，
均为 1024×1024、CFG 1，每张图一个全新的 CLI 进程。“之前”指同一构建去掉按参考图选择的编辑噪声。

- 文生图没有变化：下文的茶壶提示词在种子 0 和 1 下，以及 Turbo 在种子 0 下画的“A red fox
  sitting in deep snow in a winter forest, photograph.”，前后两个构建的 PNG SHA-256 相同。
- 重描消失了。种子 0 画出的狐狸在种子 0 下按“change the background to a sandy beach with
  the ocean behind the fox”编辑，之前得到过度锐化、仍在雪林中的副本；现在狐狸坐在沙滩上，
  身后是大海。用 `--keep-source-size` 代替 `--width 1024 --height 1024` 时，编辑同样按
  1024×1024 采样，得到相同的 PNG；把这张图以 multipart 上传发给服务端的 `/api/image-edit`，
  带 `keepSourceSize`、不带 `seed` 与尺寸（即 Web UI 的发送方式），结果也相同；两者打印的
  噪声流都一样。设置 `TS_QWEN21_EDIT_NOISE=seed` 时，该编辑的 PNG 与之前的 SHA-256 相同，
  遮罩编辑（下文种子 0 的换色）也一样。
- Turbo 的表现相同。它在种子 0 下画的狐狸，在种子 0 下按海滩指令并带 `--keep-source-size`
  编辑，之前得到过度锐化的副本：狐狸身后仍是森林，雪地变成了龟裂的地面；现在狐狸坐在海滩上，
  身后是海浪。种子模式同样复现了之前的 PNG。
- 连续编辑会执行指令：把两个检查点得到的海滩图在种子 0 下再按“put a red knitted scarf
  around the fox's neck”编辑，狐狸都戴上了围巾，海滩保持不变。
- 局部编辑更好地保留了画面。用“A red ceramic teapot on a wooden table, soft daylight,
  product photograph.”在种子 0、1、2 下各画一张，再在各自的种子下按“Change the red teapot
  to cobalt blue. Keep its shape, lighting, the wooden table and background unchanged.”
  换色，一次不带选区，一次带围住茶壶的选区。两种噪声都把茶壶变成了蓝色（蓝色像素比例相差
  不超过 0.003）。从源图自身的噪声出发时，茶壶成了颗粒感很重的深蓝色，桌面和背景被过度锐化；
  从按参考图选择的噪声出发时，它们与源图保持接近。去掉茶壶及其周围 16 像素后与源图比较的
  PSNR（`eng/validation/qwen-image21-edit-fidelity.py`）：

  | 换色 | 种子 0 | 种子 1 | 种子 2 |
  |---|---:|---:|---:|
  | 整张图，按参考图选择的噪声 | 25.4 dB | 29.5 dB | 30.9 dB |
  | 整张图，`TS_QWEN21_EDIT_NOISE=seed` | 20.2 dB | 17.0 dB | 16.9 dB |
  | 选区内，按参考图选择的噪声 | 29.8 dB | 26.4 dB | 29.0 dB |
  | 选区内，`TS_QWEN21_EDIT_NOISE=seed` | 21.8 dB | 19.3 dB | 19.4 dB |

  每次遮罩编辑中，选区外的像素都与源图完全一致。
- 耗时在运行间波动范围内相同：两种设置下以及改动之前，每次编辑都是 135–139 s，Turbo 每次编辑 98 s。

图片、哈希与分析保存在被忽略的 `artifacts/editnoise4/` 与 `artifacts/editnoise5/` 中（本地验证
证据，未提交）。未测量：CUDA、Vulkan 与 `--tp`、`cpu` 后端、TensorAgent 应用（它们把同样的图片
交给同一条流水线，但运行在 Mono 上；它们的 PNG 解码只在 macOS 上用其 Swift 复刻核对过，未在 iOS
上核对）、Turbo 上的遮罩与多参考图编辑，以及 JPEG 或 HEIC 源图。

## 启动 TensorSharp.Server.Host

```bash
dotnet run --project TensorSharp.Server.Host -c Release --no-build -- \
  --config config/qwen-image-2.1.json --host 127.0.0.1 --port 5000
```

打开 `http://127.0.0.1:5000`。不带附件的提示词会生成图像；附加一张或多张图像即可
编辑。页面会显示去噪进度和输出文件的下载链接。由于扩散流水线共享可变的工作状态，
图像操作会串行执行。

重新构建主机后，可以直接传入已有的 Unsloth 下载文件：

```bash
TensorSharp.Server.Host/bin/TensorSharp.Server.Host \
  --model ~/work/models/qwen-image-2.1-unsloth/qwen-image-2.1-Q8_0.gguf \
  --qwen-image-vae ~/work/models/qwen-image-2.1-unsloth/qwen_image_2.1_vae_bf16.safetensors \
  --qwen-image-vl ~/work/models/qwen-image-2.1-unsloth/Qwen3-VL-8B-Instruct-Q4_K_M.gguf \
  --qwen-image-mmproj ~/work/models/qwen-image-2.1-unsloth/mmproj-BF16.gguf \
  --backend ggml_metal
```

加载器通过张量布局识别不带元数据的扩散 GGUF，包括带 `model.diffusion_model.`
前缀的文件。专用 2.1 VAE 同时接受原始 Wan 命名和带空间卷积核的 Diffusers 命名；
适配器保留存储的权重，无需转换模型文件。

### HTTP API

文生图 JSON API：

```bash
curl --fail-with-body http://127.0.0.1:5000/api/image-generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"An orange cat beside a blue ceramic vase, soft daylight","width":2048,"height":2048,"steps":40,"cfg":1,"seed":42}'
```

响应为 `{ "ok": true, "url": "...", "width": 2048, "height": 2048,
"elapsedSeconds": ... }`。请从同一台服务器下载返回的 URL。

Multipart 图像编辑：

```bash
curl --fail-with-body http://127.0.0.1:5000/api/image-edit \
  -F 'image=@generated.png' \
  -F 'prompt=Change the blue vase to red. Preserve the cat and composition.' \
  -F 'width=2048' -F 'height=2048' -F 'steps=40' -F 'cfg=1' -F 'seed=42'
```

重复 `image` 部分即可传入多张参考图。也可以先把文件上传到 `/api/upload`，再向
`/api/image-edit` 发送包含 `imagePaths`（或旧的 `imagePath`）的 JSON。两个图像端点都
接受 `negativePrompt`、`targetArea`、`width`、`height`、`steps`、`cfg` 和 `seed`（默认 0；
编辑的噪声还取决于参考图，见[种子与编辑](#种子与编辑)）。`targetArea` 控制自动尺寸选择；显式尺寸优先。省略 `width`、`height`、`targetArea`、
`steps` 和 `cfg` 时使用上文的模型默认值。`targetArea: 1048576` 选择约 1K 的输出，
同时保留自动宽高比选择。编辑端点还接受 `keepSourceSize`（JSON 中为 `true`，multipart 中为
`-F 'keepSourceSize=true'`）：结果保持第一张图的精确尺寸，并按上文所述在 `targetArea`
（或默认面积）之内采样。与 `width`/`height` 同时发送，或发送到 `/api/image-generate` 时，
会以 400 拒绝。

启动服务端时传入 `--width` 与 `--height` 会改变这个默认尺寸。主机把它们发布为
`TS_QWEN_IMAGE_WIDTH` / `TS_QWEN_IMAGE_HEIGHT`，之后凡是既没有 `width`/`height`、也没有显式
`targetArea` 的图像请求都使用该尺寸，包括不发送尺寸的 Web UI 请求；此时编辑也不再沿用第一张参考图的
宽高比——带 `keepSourceSize` 的编辑（Web UI 的每次编辑）除外：它只使用该尺寸的面积，按源图宽高比采样，
并保持源图尺寸。请求自己设置了 `targetArea` 时保留它自己的几何设置。默认尺寸需要两个参数都设置。不是 32 倍数的
值会向下取整到 32 的倍数（最小 32），并打印一次 `[qwen-image] WARNING: … render at WxH instead. Reported
once.`；只设置其一、或值无法解析或为负数时，默认尺寸会被忽略（同样只警告一次），继续使用自动尺寸。
Qwen-Image 服务端在这两种情况下都会在启动时警告，但不会拒绝任何东西。请求本身设置的 `width` / `height`
仍必须是 32 的正整数倍。在服务端，这两个参数同时也是 `--video-width` / `--video-height` 的别名。

需要进度时，配合 `curl -N` 使用 JSON 路由 `/api/image-generate/stream` 和
`/api/image-edit/stream`。它们发出 SSE `data:` 帧，包含 `imageGenerate: true` 或
`imageEdit: true`、`step` 与 `total`，并可能附带预览 `image` data URL。终止帧包含
`done: true` 以及最终的 `url`、尺寸和耗时秒数，或者包含 `error`。聊天补全路由不会
运行这个扩散模型。现有的 `/api/image-edit` 请求仍至少需要一张参考图；生成有自己的
端点。预览由当前流预测估计出的干净潜变量解码而来。

## Qwen-Image-2.1-Turbo

[Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) 是 Qwen 对同一个
7B Transformer 的加速检查点：蒸馏为 **8 步 Euler、CFG 1**，使用一个固定调度。它的 VAE、
Qwen3-VL-8B 文本编码器与视觉投影器与上面的 2.1 文件逐字节相同。TensorSharp 运行
[AtomicChat 的 GGUF](https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF)（修订版本
`bb25d06`），这些文件不含元数据：

| 文件 | 大小 | 相对 BF16 的 LPIPS（模型卡） | 用途 |
|---|---:|---:|---|
| `Qwen-Image-2.1-Turbo-AD-Q4_K.gguf` | 4,201,694,944 字节 | 0.147 | [`config/qwen-image-2.1-turbo.json`](../../config/qwen-image-2.1-turbo.json) 的快速默认 |
| `Qwen-Image-2.1-Turbo-Q8_0.gguf` | 7,591,554,784 字节 | 0.037 | 最接近全精度的 Transformer |

模型卡中的其他文件（AD-Q6_K、AD-Q5_K、AD-Q3_K、AD-Q2_K、BF16）张量名相同；这里只运行了上面两个。
模型卡还测量了文本编码器：Q4_K_M 编码器让图片偏离 BF16 编码器的程度（LPIPS 约 0.17）与
AD-Q4_K 偏离 BF16 Transformer 的程度相当，Q8_0 编码器为 0.037。配置文件沿用上面 2.1 布局中的 Q4_K_M
编码器及其发布文件名，因此在同一目录中存放两个模型时只需下载一次。

```bash
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1-turbo.json \
  --prompt 'a neon sign that reads "OPEN LATE", rainy night' \
  --width 1024 --height 1024 --diffusion-seed 42 --output turbo.png
dotnet run --project TensorSharp.Server.Host -c Release --no-build -- \
  --config config/qwen-image-2.1-turbo.json --host 127.0.0.1 --port 5000
```

### 声明检查点

Turbo 的张量名与形状与基础检查点相同，GGUF 又没有元数据，文件本身无法区分二者，所以由宿主声明：

- CLI 与服务端用 `--qwen-image-variant turbo`（或 `base`），配置文件用
  `"qwen-image-variant": "turbo"`；Turbo 配置文件已经写好。宿主通过 `TS_QWEN_IMAGE_VARIANT`
  交给模型，进程内调用方也可以在构造 `QwenImageModel` 之前设置它（其 `Variant` 属性报告结果）。
  未知的参数值在启动时是配置错误（退出码 1），未知的 `TS_QWEN_IMAGE_VARIANT` 会拒绝加载（退出码 2）。
- TensorAgent 的 Turbo 目录条目自行声明。

加载时打印 `variant = turbo (declared)`。不声明时会读取文件名：文件名含单词 `turbo` 即按 Turbo
处理（与 Wan 识别步数蒸馏检查点的方式相同），加载时打印
`variant = turbo, ASSUMED from the word "turbo" in the file name: ...`。这只是猜测：合并进基础
检查点的 Viggle 步数蒸馏 LoRA 以同样的文件名发布（Abiray/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-GGUF
提供的 `qwen_image_2.1_turbo_Q4_K_M.gguf` 与 Abiray/Qwen-Image-2.1-Turbo-GGUF 中的完全同名），
这样的合并需要 Viggle 的调度而不是 Turbo 的，请为它声明 `base`。其他文件名一律视为基础检查点。
服务端的声明与伴随文件路径一样，作用于它加载的每个 Qwen-Image 模型。这一声明对其他模型
没有意义：CLI 遇到其他模型时拒绝 `--qwen-image-variant`（退出码 1），服务端则照常加载该模型、
不使用声明，并记录一条警告，与 `--lora` 相同。

### 采样

Turbo 使用其
[`model_index.json`](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/model_index.json)
中的 `sample_sigmas`，再加上最后的 0：

```text
[1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0]
```

它的调度器 shift 为 1.0 且不做动态偏移，因此在任何分辨率下都原样使用这些值：diffusers 就是这样
加载的，AtomicChat 模型卡也把这组值作为 `--sigmas` 交给 stable-diffusion.cpp。运行时会在去噪前记录：

```text
Qwen-Image-2.1 Turbo: 1024x1024, 8 steps, CFG 1, seed 42, 0 reference(s)
  [qwen21] Turbo schedule (model_index.json sample_sigmas): 8 steps on fixed sigmas [1, 0.97845, 0.95418, 0.92663, 0.89508, 0.84515, 0.70453, 0.41457, 0]
```

- 默认 8 步、CFG 1。显式的 `--diffusion-steps 8`（请求中的 `steps: 8`）可以接受，其他步数会被拒绝，
  报错会写出支持的步数。Qwen 的模型卡说明只设置步数不会覆盖保存的调度，且没有评估过其他调度，
  所以 TensorSharp 不会把这 8 个值重新采样成别的步数。
- 显式的 `--cfg` 大于 1（请求中的 `cfg`）时仍会加上负向前向，与基础检查点相同；Turbo 是为 CFG 1 蒸馏的。
- 时间步以 F32 送入 Transformer，与基础检查点和 stable-diffusion.cpp 相同。
- 编辑、`--keep-source-size` 与前缀 KV 缓存的行为与基础检查点相同（见下方验证，以及 TensorAgent
  在 Turbo 上的规划测试）；遮罩与多张参考图走同一代码路径，但未在 Turbo 上运行过。编辑的噪声
  在这里同样取决于图片（见[种子与编辑](#种子与编辑)）。

### Turbo 上的 LoRA 插件

Turbo 本身已经过步数蒸馏。带采样配方的插件（Viggle Turbo、Pruna 8 步与 5 步、Fun-Acc 的 PDD
包，或任何带 `"sampling"` 的配置）会在读取张量之前被拒绝：`LoRA plug-in refused: LoRA '...'
carries a sampling recipe (...): it is a step-distillation plug-in, and this transformer is
Qwen-Image-2.1-Turbo (declared), which is step-distilled already and samples on its own 8-step
schedule.` CLI 与服务端都会拒绝加载（退出码 2），报错还会写明 Turbo 是如何识别的。
不带配方的插件可以加载。不带插件配置直接传入的 LoRA `.safetensors` 也没有配方，引擎无法区分加速
适配器与风格适配器；请传入 `config/lora/` 中的插件。

已用真实图片在 Turbo 上验证：通过 `config/lora/` 中的插件以强度 0.7 使用 Film Stills 与
Grainscape，各自与不加插件的同一提示词、同一种子对比，种子为 42 和 7（1024×1024、AD-Q4_K；
[`qwen-image21-turbo.py`](../../eng/validation/qwen-image21-turbo.py) 的 `--suite style` 可复现，
其种子 42 的图片与先前一次运行逐字节相同）。两个插件都会施加各自的风格，也都会重画部分场景，
Film Stills 改动更多：

- Film Stills 把清晰的夜间电车站场景变成更柔和、颗粒更明显、色调更暖的电影剧照。种子 42 下它还
  去掉了电车和长椅，女子坐在一团看不清的暗色物体上；种子 7 下她仍坐在候车亭的长椅上，但身后的
  街道被重画，原来轨道的位置变成了一棵树和一排店面。
- Grainscape 把清晨清澈的山间湖泊变成朦胧、低饱和的胶片质感。湖面、小船和山大致留在原来的位置；
  岩壁和山脊线被重画，种子 42 下小船还多了一副桨。
- 重画场景来自插件本身，而不是 Turbo：在基础检查点上以 40 步（Q4_K_M、种子 42）运行，Film Stills
  把同一个电车站场景整个换掉了：没有电车，视角更低，街道两旁是树。
- 每个插件每步约多 5.4%（8.28–8.31 秒对 7.86–7.88 秒）。

TensorAgent 的 Turbo 条目只提供这两个插件。编辑类插件与 Quality Fix 尚未在 Turbo
上试过，Object Remover 只在基础模型的 40 步下有效。

### 与 stable-diffusion.cpp 的一致性

2026-10-09 在 Apple M5 Pro（48 GB，`ggml_metal`）上对比 TensorSharp 与 stable-diffusion.cpp
`f89d9b1`（当天的 master，包含修正自定义 sigma 的 `c150a6b`；以 Metal 和它自带的 ggml 子模块
`d25b121` 构建）。TensorSharp 使用未修改的上游 ggml `ffa4e8b`。两个引擎使用同一个 Transformer
GGUF、Q4_K_M 文本编码器、BF16 VAE 与 F16 mmproj，同样的提示词、种子 42 与 Philox 噪声
（sd.cpp 使用 `--rng cuda`），Euler、CFG 1，以及公布的 8 个 sigma（sd.cpp 通过 `--sigmas` 传入）。
编辑用的参考图是 Turbo 模型卡中的游艇草图，预先合成到白底并缩放到 1728×608 的条件尺寸，因此两个
引擎都不再缩放它。PSNR 比较两个引擎输出的 PNG，每一对也都用肉眼核对过。每步秒数取稳态（第 2 步起）。
这次编辑运行于 TensorSharp 按参考图选择编辑噪声之前，因此两个引擎都从该种子的噪声开始；要复现它需要
`TS_QWEN21_EDIT_NOISE=seed`，基准脚本在同时运行两个引擎时会自动设置。

| 提示词 | Transformer | 尺寸 | PSNR | 每步秒数 TensorSharp / sd.cpp | 总耗时 TensorSharp / sd.cpp |
|---|---|---|---:|---:|---:|
| 模型卡的 `a neon sign that reads "OPEN LATE", rainy night` | AD-Q4_K | 1024×1024 | 57.4 dB | 7.79 / 8.76 | 70.3 / 92.8 s |
| 毛笔字写着“清风茶社”的茶馆招牌（中文提示词） | AD-Q4_K | 1024×1024 | 44.7 dB | 7.78 / 8.77 | 69.4 / 79.9 s |
| 老渔夫的特写肖像照片 | AD-Q4_K | 1024×1024 | 59.6 dB | 7.79 / 8.76 | 69.5 / 79.4 s |
| 编辑：把模型卡的游艇草图变成照片（完整提示词） | AD-Q4_K | 1728×608 | 39.3 dB | 7.97 / 11.16 | 87.2 / 114.3 s |
| 霓虹灯招牌 | Q8_0 | 1024×1024 | 59.5 dB | 7.62 / 8.63 | 69.0 / 79.1 s |

- 每一对都是同一张图：霓虹灯招牌红色的 OPEN、蓝色的 LATE；茶馆招牌写出了四个字，两个引擎都把
  “风”写成繁体“風”；肖像连皮肤纹理都一致；编辑后的游艇保留了草图中的桅杆、驾驶室、船中部的四扇窗、
  两艘救生艇、黄色烟囱，以及五个和九个一排的舷窗。
- 剩下的差异来自舍入：两个引擎的求和顺序不同。编辑的差异最大，因为参考图还要经过两个引擎各自的
  视觉编码器与 VAE 编码器。sd.cpp 的第一次运行（霓虹灯招牌）包含从冷页缓存读取文件的时间。
- 8 位与 4 位 Transformer 画出同一场景但细节不同：TensorSharp 的两张霓虹灯招牌相差 19.1 dB，与模型卡
  中 AD-Q4_K 相对 BF16 的 20.8 dB 相当。
- TensorSharp 在 Metal 上的输出是确定的：重复运行得到逐字节相同的 PNG。

模型卡的示例图使用 `--rng cpu` 与 BF16 文本编码器，无法在这里逐像素复现；上面的对比以 sd.cpp 本身为参考。

### 性能

Apple M5 Pro（48 GB），`ggml_metal`，1024×1024，霓虹灯招牌提示词，种子 42，Q4_K_M 文本编码器。
每种配置在两个全新进程中各运行一次，按 A-B-C-D-D-C-B-A 顺序、每次间隔 20 秒，表中为中位数（同一配置
两次运行相差不到 0.4%）。每步秒数取稳态；第一步还要保存前缀 KV 缓存，约多 0.2 秒。各配置的 VAE 解码
均为 5.1–5.3 秒，文本编码 0.3 秒。

| 配置 | 步数 | 每步秒数 | 去噪 | 总耗时 | 相对 40 步 |
|---|---:|---:|---:|---:|---:|
| Qwen-Image-2.1 Q4_K_M | 40 | 7.83 | 313.3 s | 319.5 s | 1.0× |
| Qwen-Image-2.1 Q4_K_M + Viggle Turbo r128 | 6 | 8.30 | 50.0 s | 56.3 s | 5.7× |
| Turbo AD-Q4_K | 8 | 7.86 | 63.1 s | 69.3 s | 4.6× |
| Turbo Q8_0 | 8 | 7.62 | 61.2 s | 67.6 s | 4.7× |

- Turbo 每步的开销与基础检查点相同：Transformer 一样，AD-Q4_K 的混合量化（首尾各四个块的注意力为
  Q5_K、其余为 Q4_K，调制为 Q8_0，时间嵌入、输出头与文本输入为 BF16）与 Q4_K_M（注意力 V 为 Q6_K）
  相差不到 0.4%。BF16 张量只作用于两行（时间嵌入）或提示词的 token（`txt_in`），而不是 4,096 个图像
  token，因此它们的类型看不出影响。
- Q8_0 每步比 AD-Q4_K 快 3%：4,096 个图像 token 时矩阵乘法受计算而非带宽限制，矩阵内核解包 Q8_0
  块比 K-quant 块便宜。质量更高的文件也更快，代价是多 3.4 GB 权重。
- Viggle Turbo 的 6 步最先完成：它每步多 6%（不合并的适配器），但少两步。Turbo 是 Qwen 自己的蒸馏，
  不需要适配器。
- 在 1248×832 编辑时（`/usr/bin/time -l`）三种 Transformer 的峰值内存占用均为 14.35–14.37 GB：
  Transformer 从文件映射，峰值来自它前后的阶段。TensorAgent 的档位在此基础上再加上权重。

可用 [`eng/validation/qwen-image21-turbo.py`](../../eng/validation/qwen-image21-turbo.py)
（`--suite parity,perf,footprint`）复现，它调用
[`qwen-image21-bench.py`](../../eng/validation/qwen-image21-bench.py) `--variant turbo`。

### Turbo 的限制

- Turbo 只运行 8 步。模型卡的测量与这里的测量都在约 1 百万像素；Qwen 推荐约 2 百万像素（默认的 2048×2048）。
- 变体靠声明。未声明且文件名为 `...turbo...` 的文件按 Turbo 处理，对同名发布的步数蒸馏合并模型来说是错的。
- Turbo 的其他量化版本未在这里运行。

## 用遮罩精确编辑局部区域

提供与第一张输入图尺寸完全相同的遮罩即可编辑选定区域。默认**白色编辑、黑色保留**，灰色在
编辑结果与原图之间混合。透明遮罩可用 `maskMode: "alpha"` 或 `--mask-mode alpha`：透明像素编辑，
不透明像素保留。原图自身的透明度与选区遮罩分开处理。

输出保留第一张输入图的原始尺寸，并在每个未选中像素处保留其解码后的 RGB 与 alpha 值。
去噪期间会约束受保护的潜变量，最后把生成区域合成到原图上。其他输入图仍作为参考图。
空选区直接返回原图，不运行扩散；全选则编辑整个画布。

```bash
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1.json \
  --image photo.png --mask selection.png --mask-mode grayscale \
  --prompt 'Change the selected vase to red ceramic' \
  --mask-feather 8 --mask-crop --mask-crop-padding 96 \
  --width 1024 --height 1024 --diffusion-seed 42 --output edited.png
```

`--mask-feather` 以原图像素为单位向内柔化边缘（0–1024，默认 0），让未选中的像素保持受保护。
`--mask-invert` 反转选区。`--mask-crop` 只处理选区及周围上下文，再把它放回原始画布；
`--mask-crop-padding` 设置上下文的原图像素范围（0–16384，默认 64）。裁切是可选的，小选区可以
使用较小的内部 `--width` / `--height` 来减少推理工作。这两个参数设置名义上的整幅采样分辨率；
裁切模式会按裁切区域占原图的比例缩放，并向上取整到 32 像素网格。保存的图像仍使用原图尺寸。
质量与速度的取舍取决于选区、上下文、内部尺寸和采样步数。

Multipart API：

```bash
curl --fail-with-body http://127.0.0.1:5000/api/image-edit \
  -F 'image=@photo.png' -F 'mask=@selection.png' \
  -F 'maskMode=grayscale' -F 'maskFeather=8' \
  -F 'maskCrop=true' -F 'maskCropPadding=96' \
  -F 'prompt=Change the selected vase to red ceramic' \
  -F 'width=1024' -F 'height=1024' -F 'seed=42'
```

JSON 与 SSE 请求先通过 `/api/upload` 上传两份文件，再发送服务端返回的文件名：

```json
{
  "imagePaths": ["uploaded-photo.png"],
  "maskPath": "uploaded-selection.png",
  "maskMode": "grayscale",
  "maskInvert": false,
  "maskFeather": 8,
  "maskCrop": true,
  "maskCropPadding": 96,
  "prompt": "Change the selected vase to red ceramic",
  "width": 1024,
  "height": 1024,
  "seed": 42
}
```

这些字段适用于 `/api/image-edit` 与 `/api/image-edit/stream`，包括 TensorAgent 的共享图像编辑
服务。遮罩引用必须位于上传目录内。遮罩需要输入图和 Qwen-Image-2.1；尺寸不一致、不支持的模式、
无效数字字段或多个 multipart 遮罩都会在推理前被拒绝。只传遮罩选项而不传遮罩也会被拒绝。
与其他编辑错误一样，流式请求在终止的 `{done,error}` 帧中报告请求错误。

在 Server Chat 与 TensorAgent（桌面和移动端）中，附加一张或多张照片，在任意照片上选择
**Select area**（选择区域）。涂抹或擦除选区，缩放和平移查看细节，然后选择 **Use selection**
（使用选区）并描述修改。保存选区会把该照片设为 **Editing target**（编辑目标），移到输入图的
第一位，其他照片仍按原顺序作为参考图。每张照片都保留自己的选区，但当前编辑只发送编辑目标的
遮罩。取消操作或选区上传失败不会改变原有目标。每轮生成一张编辑结果。
移除编辑目标时，其余照片的选区仍保留；重新打开并保存其中一张即可激活下一次局部编辑。

撤销/重做与反转作用于选区。**Edit again**（再次编辑）恢复原始照片、已保存选区、编辑目标及
提示词；对比按钮在编辑目标的原图和结果之间切换。TensorAgent 也会随会话保存选区。
遮罩与参考图分开上传。每张解码后的 SSE 预览与最终图一样，包含受保护的原图像素。

HEIC/HEIF 上传保留小缩略图，以及用于涂抹和重新打开选区的独立全分辨率 PNG；模型仍使用原始照片。
浏览器编辑器最多支持 1600 万像素及单边 8192 像素。更大的 HEIC/HEIF 照片会显示选区尺寸限制，
不会在缩小后的缩略图上涂抹遮罩。

遮罩由 TensorSharp 自有的采样和合成代码执行，不会给 Qwen 的条件输入增加标注图或专用遮罩通道。
精确保留选区外的像素不保证模型在选区内遵循每条指令。裁切有助于隔离物体，但也会移除周围上下文；
需要该上下文时请关闭裁切。

可复用验证工具：

- `eng/validation/QwenImageMaskBench`：不加载权重，测量合成输入准备、潜变量重注入、合成耗时、
  分配以及标量一致性。
- `eng/validation/qwen-image21-mask-bench.py`：真实 CLI 或 HTTP/SSE 运行，检查像素保留、对比裁切，
  并明确记录测量限制。
- `eng/validation/validate-image-mask-editor.py`：桌面与触摸模拟的浏览器交互、导出遮罩几何形状，
  以及撤销/重做的精确回归检查。
- `eng/validation/validate-image-mask-live.py`：真实 Server Chat 上传、选区、推理、结果与复用，
  以及 multipart/错误处理检查。
- `eng/validation/tensoragent-mask-bench.py`：真实 TensorAgent 宿主聊天流程。

各工具用 `--help` 查看参数（C# 基准的参数说明位于 `Program.cs`）。报告、日志和截图保存在被忽略的
`docs/validation/` 或 `artifacts/` 中。浏览器触摸模拟不能证明原生手机上的行为；CPU 遮罩微基准
不测量完整模型推理或 GPU 提速。

GGML 后端上的 2.1 扩散 Transformer 运行完整 GGML 图并常驻量化权重；`cpu` 则对同样的文件映射
权重执行托管前向。两者都没有权重流式模式。可用内存不足时先减小尺寸。CUDA 与 Vulkan 曾在
NVIDIA A40 上运行；下文记录各项测量的具体范围。

## 纯 C# CPU 后端（`--backend cpu`）

`--backend cpu` 用托管 C# 运行整条流水线：扩散 Transformer、Qwen3-VL 文本编码器、编辑用的视觉编码器、
VAE、LoRA 插件与前缀 KV 缓存。不构建任何 GGML 计算图，流水线也不调用原生 GgmlOps 库；在这个后端上，
除非进程里有其他代码加载过该库，CLI 退出时也会跳过 GGML 的清理。权重按存储类型直接从内存映射的
GGUF 与 safetensors 文件读取。此前该模型拒绝 `cpu`，必须使用 GGML 后端。模型计算之外仍有两处原生部分，
与之前相同：桌面平台上图像文件的读写经由 Magick.NET；服务端启动时会探测 GGML 与 CUDA 后端。

```bash
TENSORSHARP_MODELS="$PWD/models" dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1.json --backend cpu \
  --lora config/lora/qwen-image-2.1-pruna-5step.json \
  --prompt 'A small orange cat beside a blue ceramic vase, soft daylight, detailed photograph' \
  --width 512 --height 512 --diffusion-seed 42 --output cat-cpu.png
```

服务端同样接受 `--backend cpu`。在这个后端上，未指定尺寸的请求使用 1 百万像素的自动面积（1024×1024，
编辑时为与第一张参考图宽高比一致的同等面积），而不是 2048×2048——后者的每个 Transformer 步要慢约 5 倍；
显式尺寸、显式的 `targetArea` 以及服务端的 `--width` / `--height` 不受影响，运行时会打印所选的尺寸。
`ggml_cpu` 保持原生的 2048×2048 默认值。

各部分的实现：

- **Transformer**（`QwenImage21ManagedDiT`）：按原生计算图的算子、以相同顺序、从相同的权重描述符计算。
  量化投影像 ggml-cpu 那样把激活量化成 Q8_K / Q8_0，并在托管的多行整数 GEMM 上计算；注意力是分块的
  flash attention；LoRA 因子（堆叠的 shrink、DoRA 行缩放）与 GGML 后端一样以不合并的方式应用。
  `TS_QWEN21_CPU_MATMUL=f32`（之前的默认值）改用 F32 激活乘以反量化的权重分块：更慢，但数值更稳定。
- **前缀 KV 缓存**：位于主机内存的托管缓存，类型相同（`auto`/`f32` 保存注意力实际读取的值，因此
  使用缓存的步骤与不使用缓存时逐比特一致；另有 `f16`、`q8_0`、`q8_0_v`），规则也相同：最多使用
  空闲物理内存的一半，并受 `TS_QWEN21_PREFIX_CACHE_MAX_MIB` 限制。
- **文本编码器**：投影同样在多行整数 GEMM 上计算（8 位激活；`TS_QWEN_TE_CPU_MATMUL=f32` 改用反量化
  Q4_K / Q6_K 分块上的 packed F32 GEMM），注意力是托管的因果分组查询注意力。
- **视觉编码器**（编辑）：packed GEMM 线性层与托管多头注意力。每个线性层权重打包之后即释放其 F32 副本，
  因此视觉塔只保留一份权重：在分阶段基准中编码一张 512×512 参考图的提交内存峰值为 3.2 GB，保留两份时为
  5.4 GB，输出逐比特相同。
- **VAE**：每个卷积都是隐式 im2col 的 packed SGEMM，权重每层只打包一次；解码器的 2x 上采样折叠进
  读取它的卷积；它运行在每个逻辑 CPU 一个线程的专用池上（`TS_CPU_GEMM_THREADS`；`TS_CPU_POOL=0` 时改为
  该宽度的 `Parallel.For`）。

它们都有 AVX-512 与 AVX2 内核，由整个后端统一的指令集判定选择（`TS_CPU_DISABLE_AVX512=1` 选择 AVX2），
以及可移植回退；环境变量矩阵
列出了[全部开关](../env_var_feature_matrix_zh-cn.md#矩阵外的-qwen-image-21-开关)，包括恢复各个旧阶段的
`0` / `scalar` 设置。

### 在 8 核笔记本上的实测

i7-11800H（8 核 16 线程、AVX-512）、32 GB、Windows。DiT 为 `qwen_image_2.1_Q4_K_M.gguf`，文本编码器为
`Qwen3VL-8B-Instruct-Q4_K_M.gguf`，VAE 为 BF16；文生图、CFG 1、种子 42，使用默认的整数（Q8）投影。
`ggml_cpu` 走原生代码，这些改动没有触及它。PSNR / SSIM 把 `cpu` 的图像与 `ggml_cpu` 的图像对比。

| 运行 | 阶段 | `cpu` | `ggml_cpu` |
|---|---|---:|---:|
| 256×256、2 步（两次运行） | 文本与视觉编码 | 3.1 s | 4.1–4.2 s |
| | 去噪，2 步 | 11.0–11.1 s | 20.8–21.4 s |
| | VAE 解码 | 2.8–2.9 s | 10.8–11.2 s |
| | 总计 | **16.9–17.1 s** | 36.1–36.4 s |
| | PSNR / SSIM | 37.7 dB / 0.984 | 参照 |
| 512×512、Pruna 5 步 LoRA | 稳态单步（前缀已缓存） | 17.6–21.7 s | 43.9–61.4 s |
| | 总计 | **118 s** | 295.7 s |
| | PSNR / SSIM | 32.9 dB / 0.972 | 参照 |

使用 `TS_QWEN21_CPU_MATMUL=f32`（F32 激活乘以反量化的权重分块，之前的默认值）时，同样的 256×256 运行用时
20.4 s，PSNR / SSIM 为 42.0 dB / 0.984；512×512 用时 169.6 s（每步 30.7–31.6 s），为 31.4 dB / 0.96。
单看文本编码器，37 token 的默认提示在整数投影下用时 0.76–0.88 s，`TS_QWEN_TE_CPU_MATMUL=f32` 下为
1.8–1.9 s，`ggml_cpu` 约 1.4 s。另有两次更大的运行是在整数路径成为默认值之前、用 F32 Transformer 测的：
1024×1024、Pruna 5 步生成在 `cpu` 上用时 756 s，`ggml_cpu` 为 1101 s；512×512 编辑为 224 s 对 353 s。

两个后端产生的像素并不相同，而且两者都不是参照答案：托管 Transformer 像 ggml-cpu 那样量化激活，但求和
顺序不同，并且在 ggml-cpu 使用 F16 GELU 表与 BF16 舍入输入的地方保持 F32（`TS_QWEN21_CPU_GELU_FP16` /
`TS_QWEN21_CPU_ROUND_ACTIVATIONS` 可复现这两点），每次重新量化都可能翻转一次舍入。在单次前向上
（`benchmarks/QwenImageDiTBench`，256×256），托管结果的速度场与 ggml-cpu 的余弦相似度在 sigma 1 时为
0.99993，在 sigma 0.02 时为 0.99938（F32 路径为 0.99994 与 0.9978）；而仅仅把时间步相对改变 1e-4，
ggml-cpu 自己的速度场就会变化 4.7e-2（余弦 0.9989）。`cpu` 上的编辑还由托管 Transformer 的单元测试
（编辑布局、多张参考图）与分阶段基准（`benchmarks/QwenImageStagesBench`）覆盖。在分阶段基准中，一张
1024×1024 参考图经过视觉编码器用时 12–13 s，256×256 的参考图用时 0.72 s，`ggml_cpu` 为 5.0 s。

### `cpu` 上的限制

- **张量并行仅限 GPU。** `--backend cpu` 配合 `--tp N` 会在加载时被拒绝（退出码 2），提示信息会指向
  `ggml_cuda` / `ggml_vulkan`；托管流水线在单个进程内运行。
- **内存。** DiT 与文本编码器的权重是文件映射的（Q4_K_M DiT 4.2 GB，Q4_K_M 文本编码器 5.0 GB），但激活、
  前缀缓存与 VAE 特征图都是普通进程内存。VAE 在每张特征图最后一次被读取后即释放它：1024×1024、Pruna 5 步
  生成的提交内存峰值为 3.7 GiB（工作集 8.0 GiB），2048×2048 的 VAE 编码加解码为 10.5 GiB。在开始任何工作
  之前，估计峰值超过机器内存的尺寸会被拒绝，并给出能放下的最大方形尺寸（`TS_QWEN_IMAGE_CPU_MEMORY_CHECK=0`
  可跳过）；只超过当前空闲内存的尺寸会得到一条警告。
- **速度。** 在上面这台 8 核笔记本上，512×512 每步 17.6–21.7 s；2K 方图的图像 token 是 512×512 的 16 倍，
  注意力的增长还要更快，这也是这个后端的自动尺寸为 1024×1024 的原因。需要交互速度时，请使用步数蒸馏
  LoRA 并选小尺寸。
- 只对 GGML 有效的开关（`TS_QWEN21_GRAPH_REUSE`、`TS_QWEN21_FLASH`、`TS_QWEN21_PAD_MASK`、
  `TS_QWEN21_VAE_FUSED`、`TS_QWEN21_VISION_FUSED`）在 `cpu` 上不起作用。

## LoRA 插件

TensorSharp 在运行时把 LoRA 适配器应用到 2.1 扩散 Transformer 上：风格与编辑 LoRA、
DoRA，以及把默认 40 步换成 4–8 次 Transformer 前向的步数蒸馏适配器。在 CLI 或服务端用
`--lora` 传入，重复该参数即可叠加多个。`--lora-scale` 设置前一个 `--lora` 的强度，
`--lora-config` 指定它的伴随配置。[`config/lora/`](../../config/lora/) 中的插件会在首次
使用时下载并校验权重哈希，并带有适配器的强度与采样配方：

```bash
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1.json \
  --lora config/lora/qwen-image-2.1-viggle-turbo.json \
  --prompt 'A small orange cat beside a blue ceramic vase, soft daylight, detailed photograph' \
  --width 1024 --height 1024 --diffusion-seed 42 --output turbo.png
```

参数说明、插件配置格式以及十二个随附插件的列表见
[USAGE_zh-cn.md](../../USAGE_zh-cn.md#qwen-image-21-lora-插件)。

### 采样配方

步数蒸馏 LoRA 是针对某一个调度训练的，因此它的插件记录了这个调度。随附的配方遵循各适配器的
模型卡：

- **Viggle Turbo**（默认 6 步，支持 4–8 步，CFG 1）。原始节点（例如 6 步时为
  `[1, 0.9375, 0.875, 0.75, 0.5, 0.25]`）经过检查点随分辨率变化的指数偏移（锚点为
  256/0.5 与 8192/0.9），但不经过基础调度器的末端拉伸，最后接一个 0（`"shift": "dynamic"`）。
- **Pruna 8 步与 5 步**（CFG 1）。原样使用模型卡给出的 sigma，不做偏移，最后接一个 0
  （`"shift": "none"`）。
- **Fun-Acc 4 步**（CFG 1）。在任何分辨率下都原样使用 PDD 包在 `pdd_config.json` 中训练好的
  网格。时间步像参考实现的钩子那样经过 bf16 取整，第 *i* 步使用第 *i* 个输出头。

显式设置优先于配方，配方又优先于模型默认值（40 步、CFG 1）：CLI 上是 `--diffusion-steps` /
`--cfg`，服务端是请求中的 `steps` / `cfg`（`0` 或省略表示使用配方）。配方没有调度的步数会被
拒绝，报错会列出支持的步数；PDD 包每个训练步有一个输出头，只能按训练时的步数运行。两个都带
配方的插件不能叠加；[Qwen-Image-2.1-Turbo](#turbo-上的-lora-插件) 的检查点有自己的调度，
带配方的插件在它上面会被拒绝。风格与编辑插件不带配方，沿用检查点自己的调度。运行会在去噪前记录解析出的
配方及其 sigma。

### 支持的格式

- **张量命名**：diffusers / PEFT（`transformer.` 前缀、`lora_A` / `lora_B`、
  `lora_A.default.weight` 这类适配器槽名）、ComfyUI 与 ai-toolkit（`diffusion_model.`）、
  DiffSynth / ModelScope（无前缀）、kohya
  （`lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight`），以及带或不带 `.weight`
  的 `lora_down` / `lora_up`。
- **alpha**：来自逐模块的 `.alpha` 张量、diffusers 保存时嵌入 safetensors 元数据的 PEFT 配置
  （`lora_adapter_metadata`）、`adapter_config.json`（`lora_alpha`、`alpha_pattern`、
  `use_rslora`）；都没有时 alpha 等于 rank。kohya 的 `ss_network_alpha` 是训练元数据，会被忽略，
  与 ComfyUI 和 diffusers 一致（去掉逐模块 `.alpha` 张量的转换工具已把 alpha 折入因子）。显式配置优先于
  文件自带的元数据；即使另外指定了只含配方的配置，PEFT 目录中的 `adapter_config.json`
  仍会提供 alpha。Pruna 文件记录的是 rank 64 下 alpha 128，因此实际缩放为 2；假定
  alpha = rank 的加载器只会应用一半的适配器。
- **DoRA** 的 `dora_scale` 幅值，采用 ComfyUI 在输出轴上的语义：幅值除以检查点自身权重
  （从 GGUF 反量化）的行范数。
- **VideoX-Fun PDD 包**：替换 `proj_out` 的逐步输出头、替换的归一化增益，以及低秩增量。
  包的 `pdd_config.json` 从权重旁读取，或由 `--lora-config` 指定；只支持
  `pdd_block_size` 为 1（每步一个输出头）。
- **一维 `.diff` 张量**，作用于归一化增益（`txt_in.text_norm`、`attn.norm_q`、
  `attn.norm_k`）。
- **拆开的 MLP 投影。** diffusers 中分开的 `img_mlp.gate_layer` 与 `img_mlp.proj` 是检查点
  融合 `img_mlp.gate_up` 的 gate 半与 up 半（gate 在前），加载器会把它们放到对应位置。
- **PEFT 目录。** `adapter_model.safetensors` 旁若有 `adapter_config.json`，无需
  `--lora-config` 即会读取该配置。

以下内容会被拒绝，并给出点名张量的报错：LoKr、LoHa 与 LoCon 中间因子；文本编码器 LoRA；
偏置项与 `diff_b`（该 Transformer 没有偏置）；二维全权重差值，以及 PDD 包之外的完整 `.weight`
值；卷积 LoRA；包含多个 PEFT 适配器的文件；为其他模型制作的 LoRA，例如双流的 20B Qwen-Image
（`add_q_proj`、`txt_mlp` 或 `img_mod` 模块），或任何形状与 2.1 投影不匹配的因子。文件中的
每个张量要么被应用，要么加载失败；不会静默跳过任何张量，因为只应用了一部分的 LoRA 就不是
所要求的那个适配器了。

### 更新如何应用

基础权重按原样保持量化，每个被适配的投影计算 `y = W x + B (A x)`。步数蒸馏 LoRA 对权重的
改动约为 0.1–0.5%，与 Q8_0 自身的舍入步长相当，因此反量化、加上增量再重新量化会丢掉其中
大部分。在 Pruna 适配器的第 0 块上，这样合并后留下的增量与预期增量的余弦相似度只有 0.07。

- 强度、alpha / rank 以及 DoRA 幅值都以 F32 折入 `B`。随后对每个 rank 分量重新平衡，使其
  `A` 行与 `B` 列的范数相等（乘积不变），再以 F16 存储因子。某个值会溢出 F16 的组保持 F32。
- 同一投影上的多个 LoRA 沿 rank 拼接，因此无论插件有多少个，计算图对每个投影只做一次收缩
  与一次扩展。
- rank 用零填充：Metal 上填到 64 的倍数（其 simdgroup 矩阵内核要求 K ≥ 64），其他后端填到
  16 的倍数。
- Q、K、V 读取同一个输入，因此它们的 `A` 因子堆叠在一起，由一次收缩同时服务三者；gate 与 up
  两半也以同样方式共享。
- 被适配的投影包括图像与文本输入、时间步嵌入、调制、`norm_out`、`proj_out`，以及 32 个块中
  每块的 Q、K、V、注意力输出、gate、up 与 down。

这在模型支持的所有后端上运行：`ggml_metal`、`ggml_cuda`、`ggml_vulkan`、`ggml_cpu` 以及纯 C# 的 `cpu`。
加载时会记录插件数量与打包后因子的大小（`Qwen-Image-2.1 LoRA: N plug-in(s), applied
unmerged (... MiB of factors, ...)`），再为每个文件输出一行，列出更新数、rank 与缩放。

**前缀 KV 缓存。** 缓存保持开启。第一步让整个序列经过已适配的 Transformer，因此保存的文本与
参考图 K/V 已包含 LoRA 的作用。保留的计算图与保存的前缀以适配器的因子缓冲区为键，因此用
其他因子（或不带因子）构建的图与前缀永远不会被复用。

**张量并行。** 在 `--tp N` 下，列并行投影（Q、K、V、gate、up）让每张 GPU 取得与自己输出
切片对应的 `B` 行（以及任何 DoRA 行缩放）。行并行投影（`to_out`、`img_mlp.out`）让每张 GPU
取得与自己输入切片对应的 `A` 列，每张 GPU 在 all-reduce 之前加上自己那部分 LoRA 项，
all-reduce 把各部分相加即为完整更新。复制的投影保留完整因子。

### LoRA 性能

2026-09-25 与 stable-diffusion.cpp `19bbbca` 对比，两者均使用未修改的 ggml `353b63b`。每次运行都是
1024×1024 文生图（茶壶提示词、种子 42、CFG 1、Euler），每个引擎在全新进程中运行，两次运行之间有冷却。
两个引擎使用相同的 F32 sigma 向量。每个配置运行两次，两个引擎轮流先跑，表中为中位数。每步秒数为
稳态步（不含第一步）。PSNR 比较两个引擎的输出图像。没有步数蒸馏 LoRA 时，6 或 8 步本来就会得到
模糊的图像（模型默认 40 步），因此“无 LoRA”与 Film Stills（风格 LoRA）两行只衡量速度，不衡量质量。

Apple M5 Pro，48 GB，`ggml_metal`：

| 配置 | 步数 | 每步秒数 TensorSharp | 每步秒数 sd.cpp | 去噪加速 | 总耗时 TensorSharp / sd.cpp | 总加速 | PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 无 LoRA | 6 | 7.52 | 8.59 | 1.17× | 51.8 / 61.7 s | 1.19× | 63.1 dB |
| Viggle Turbo r128 | 6 | 7.94 | 9.40 | 1.22× | 54.5 / 67.0 s | 1.23× | 60.4 dB |
| Viggle Turbo r256 | 6 | 8.03 | 9.52 | 1.23× | 55.8 / 68.7 s | 1.23× | 57.5 dB |
| Pruna 8 步 | 8 | 7.91 | 9.39 | 1.21× | 69.7 / 85.5 s | 1.23× | 53.4 dB |
| Film Stills，强度 0.7 | 8 | 7.92 | 9.43 | 1.21× | 69.8 / 85.8 s | 1.23× | 63.9 dB |

NVIDIA RTX 4000 Ada，20 GB，驱动 580，`ggml_cuda`：

| 配置 | 步数 | 每步秒数 TensorSharp | 每步秒数 sd.cpp | 去噪加速 | 总耗时 TensorSharp / sd.cpp | 总加速 | PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 无 LoRA | 6 | 1.57 | 1.85 | 1.16× | 20.0 / 26.5 s | 1.33× | 46.2 dB |
| Viggle Turbo r128 | 6 | 1.94 | 2.53 | 1.28× | 23.3 / 32.6 s | 1.40× | 35.2 dB |
| Viggle Turbo r256 | 6 | 2.00 | 2.55 | 1.25× | 24.5 / 32.7 s | 1.33× | 36.8 dB |
| Pruna 8 步 | 8 | 1.94 | 2.53 | 1.29× | 26.6 / 37.6 s | 1.41× | 35.3 dB |
| Film Stills，强度 0.7 | 8 | 1.92 | 2.51 | 1.30× | 26.0 / 37.3 s | 1.43× | 47.5 dB |

- **每步开销。** LoRA 在 Metal 上使每步增加 5–7%（sd.cpp：9–11%），在 CUDA 上增加 22–27%
  （sd.cpp：36–38%）。开销几乎与 rank 无关（rank 64、128、256 相差不到 3%）。它来自对每个被适配
  投影输出的额外一遍读写：低秩乘积写出一个与该输出同样大小的 F32 张量，随后的加法再把它读回。
  GPU 越快，基础步越快结束，这一遍所占比例就越大。ggml 没有累加到目标张量的矩阵乘法，因此不修改
  ggml 就无法把这一遍并入基础投影。在 CUDA 上强制 cuBLAS 使用 F16 计算
  （`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16`）反而使每步慢 5%，因此 F32 乘积并不是瓶颈。
- **加载。** 启动时并行读取、缩放并打包因子：M5 Pro 上 0.1–0.8 s，CUDA 机器上 0.2–1.3 s；
  结果与串行加载逐位一致。
- **VAE 解码。** 整个 VAE 的融合计算图现在在 Metal 与 CUDA 上都是默认路径。它在 M5 Pro 上用
  5.0 s 解码 1024×1024（逐卷积路径需要 13.0 s，sd.cpp 需要 7.1 s）。2048×2048 时用 24.9 s
  代替 85.3 s，内存峰值占用为 45 GB 而非 64 GB。它从不在 Vulkan 上运行，因为解码器中的 F16
  协作矩阵操作数会溢出；在 Vulkan 上设置 `TS_QWEN21_VAE_FUSED=1` 会打印警告并改为逐卷积解码。
  `TS_QWEN21_VAE_FUSED=0` 在任何后端上选择逐卷积路径。
- **为什么 CUDA 上图像差异更大。** 在这台 CUDA 机器上，sd.cpp 把 Transformer 留在 GPU 上，整图
  解码时显存不足，于是改用 256×256 分块重试（11.5–13.2 s），它的总耗时包含这次重试。
  TensorSharp 先释放 Transformer 的权重，再用 4.0–4.1 s 完成解码。两个引擎在 CUDA 上还都以 TF32
  执行 F32 矩阵乘法。因此 CUDA 上两个引擎的图像不如 Metal 上接近。这些图像经过目视检查，
  内容一致。

测量使用 [`eng/validation/qwen-image21-bench.py`](../../eng/validation/qwen-image21-bench.py)，
通过 `--lora`、`--lora-config` 以及 `--sigma-nodes`/`--sigma-shift` 指定配方调度。
sd.cpp 忽略 `lora_adapter_metadata` 中的 alpha，因此给它的 Pruna 倍率为 2（`--sd-lora-multiplier 2`）。

### 服务端与 C# API

服务端在启动时加载 `--lora` 插件组，并把它应用到每个生成与编辑请求；其 HTTP 请求不能选择
插件。请求中的 `steps` 与 `cfg` 仍会覆盖插件的配方。在进程内，
`QwenImageModel.SetLoras(IReadOnlyList<LoraSpec>)` 为之后的请求替换插件组（空列表表示
移除）。新插件组会立即针对 Transformer 校验，失败时保留原来的插件组。

按图片选择插件的宿主把插件组传给 `WebUiChatService` 的 `ImageGenerateStreamAsync`、
`ImageEditStreamAsync` 或 `ImageEditAsync(body, loras, ct)`：插件组在与生成相同的锁内替换，
所以排队等候的图片使用它请求时的插件组；插件组未变时没有开销（请传绝对路径，模型按此记录
插件组）。TensorAgent 的 Mac 应用即如此：它从自己的固定目录提供
[USAGE_zh-cn.md 表中](../../USAGE_zh-cn.md#qwen-image-21-lora-插件)的十二个插件，并把用户的选择
应用到每张图片，包括它的编辑路由（见 [TensorAgent 的 README](../../TensorAgent/README.md)）。

### 限制

- 插件只作用于 Qwen-Image-2.1；CLI 遇到其他模型时拒绝 `--lora`，服务端则记录一条警告，
  并在不带插件的情况下加载该模型。
- 每次运行只能有一个插件带采样配方，而带 sigma 的配方只能以它定义的步数运行。
  Qwen-Image-2.1-Turbo 不接受任何带配方的插件。
- Qwen-Image-2.1-Fix 作者的工作流还使用了 APG、FreSca 以及 CFG 3 下的 `seeds_2` 采样器，
  TensorSharp 没有实现这些；DoRA 本身会被精确应用。
- Pruna 适配器在 1K 下训练。Fun-Acc 在 2048×2048 下训练，其模型卡指出小而密的文字以及部分
  编辑效果弱于 40 步的教师模型。

## 加速与内存

在 GGML 后端上，2.1 扩散 Transformer 以常驻的量化权重运行完整的 GGML 图；在 `cpu` 上，它对同样的
文件映射权重运行托管前向（见[纯 C# CPU 后端](#纯-c-cpu-后端--backend-cpu)）。两者都没有权重流式加载
模式。可用内存不足时，请先使用较小的尺寸。CUDA 与 Vulkan 已在 NVIDIA A40 上验证，
各项测量见[英文版模型卡](qwenimage21.md#prefix-kv-cache)。

在 CPU、Metal 与 CUDA 上，图像段注意力按每段精确的 K/V 长度计算，不再构建稠密的填充
掩码（CUDA 上 `TS_QWEN21_PAD_MASK=1` 可恢复填充掩码作对比诊断，见
[`docs/perf/qwen-image21-cuda.md`](../perf/qwen-image21-cuda.md)）；文本的因果掩码保持
不变，`ggml_vulkan` 仍构建填充的图像掩码。

### 前缀 KV 缓存

Qwen-Image-2.1 用 `t = 0` 那一行调制文本与参考图 token，并且其块因果注意力从不
让它们关注正在生成的图像（检查点的 `causal_condition`）。因此它们的隐藏状态，
以及每个块的 K 和 V，在每个去噪步都完全相同。TensorSharp 实现了官方的
[前缀 KV 缓存](https://github.com/QwenLM/Qwen-Image-2.1#prefix-kv-cache)：
第一步运行整个序列，把每个块前缀部分经过 RoPE 之后的 K 和 V 存到设备上；之后
每一步只计算目标图像的 token，并对"已存前缀 + 目标"做注意力。CFG 运行为每个
分支各保留一份缓存。去噪结束、VAE 解码之前释放缓存。每一步的日志行以
`prefix=extract` 或 `prefix=cached` 结尾（缓存放不下时为 `prefix=declined`，见下文）。

缓存默认开启；`TS_QWEN21_PREFIX_CACHE=0` 关闭它。默认情况下它存的正是注意力内核
读取的类型（Metal 与 CUDA flash attention 为 F16，其他为 F32），所以缓存步复现
不用缓存时的计算：在 Metal 上，同一个种子在开启缓存、关闭缓存以及加入缓存之前的
构建中得到逐字节相同的 PNG。已对生成、单参考图与双参考图编辑、以及带两份缓存的
CFG 4 做过验证。diffusers 的 `use_kv_cache` 文档指出其 PyTorch 实现在两种设置下
不能逐位复现图像；TensorSharp 的计算图可以。

代价是内存：F16 下每个前缀 token、每个 CFG 分支 512 KiB（32 个块、K 与 V、
4096 个 2 字节的值）。提示词只有几十到几百个 token；每张约 1 百万像素的参考图
增加 4,096 个 token，约 2 GiB。一份缓存最多使用设备报告的空闲内存的一半，
`TS_QWEN21_PREFIX_CACHE_MAX_MIB` 可以进一步限制。放不下的缓存会被拒绝并在
stderr 上给出警告，该请求像以前一样每一步都重算前缀。

`TS_QWEN21_PREFIX_CACHE_TYPE` 选择存储类型：`auto`（默认，与不用缓存完全一致）、
`f16`、`f32`、`q8_0`（K 与 V 均为 Q8_0，每个 token 约 272 KiB）和 `q8_0_v`
（仅 V 为 Q8_0，约 392 KiB）。两种 8 位设置对应 vLLM-Omni 的 `fp8` 与 `fp8_v`
前缀缓存，但使用每 32 个值一个缩放的 ggml Q8_0 块，而不是每个 token 与头一个
缩放的 FP8 E4M3；每一步都会把前缀转换回注意力类型，因此目标自身的 K/V 保持精确。
它们对输出质量的实测影响见[英文版模型卡](qwenimage21.md#prefix-kv-cache)。

实测（默认存储，输出 PNG 与不用缓存时逐字节相同）：在 Apple M5 Pro（`ggml_metal`）上，
1024² 单参考图编辑每步从 17.97 秒降至 9.65–10.38 秒，双参考图从 33.32 秒降至 11.46 秒；
在 NVIDIA A40（`ggml_cuda`）上分别从 2.547 秒降至 1.328 秒、从 4.106 秒降至 1.538 秒。
文生图的前缀只有提示词，只快 1–3%；2048² 时目标 token 占主导，编辑只快 13–17%。

### CUDA Graph 与张量并行

上游 ggml-cuda 在同一张图连续两次执行不变后把它捕获为 CUDA Graph 并重放。缓存步
的计算图在各步之间保留（`TS_QWEN21_GRAPH_REUSE=1`，默认），输入与缓存缓冲区
固定，因此从一次请求的第三步起，每个去噪步都是一次图重放——这就是 vLLM-Omni
CUDA Graph 解码的效果，无需另写捕获代码。与 vLLM-Omni 一样，存储前缀的第一步
不被捕获。在 A40 上统计 CUDA 运行时调用确认了这一点：10 步请求只捕获 1 次、
重放 8 次，双卡请求捕获 130 次（每卡 65 段）、重放 1,040 次；开启与关闭 CUDA Graph
的输出 PNG 相同。由于每步的内核都很大，收益很小：256×256 单卡每步快 1.5%，1024²
无可测差异。

在 `ggml_cuda` 或 `ggml_vulkan` 上使用 `--tp N` 时（`cpu` 会在加载时拒绝它），扩散 Transformer 按
Megatron 方式切分到 N 张 GPU：每张卡持有 32/N 个完整注意力头与 12,288/N 个 MLP
列，其余投影与所有归一化权重复制；每个块的两个行并行乘积在 GPU 之间求和
（ggml-cuda 有集合通信时在设备上完成，否则经由主机内存）。每张卡缓存自己那些头
的前缀。N 必须整除 32 个注意力头——单机上受 ggml 16 设备上限约束，即 2、4、8 或 16——而且切分每个权重时都不能拆开它的量化块，加载时会按权重类型逐一检查。实测只覆盖 2 卡。文本编码器、
视觉编码器与 VAE 留在第一张卡上；不支持多节点组。在两张 A40（NCCL 经共享内存传输）上，1024² 文生图每步
从 1.107 秒降至 0.823 秒（1.34×），1024² 编辑从 1.326 秒降至 0.934 秒（1.42×），2048²
文生图与编辑分别为 1.54× 与 1.57×。`ggml_vulkan` 经主机内存归约，双卡反而慢 14%。
TP 改变了部分和的相加顺序，输出与单卡不逐位相同；这种舍入差异会沿去噪轨迹累积，
构图与质量相同但细节可能不同（与单卡相比 PSNR 34–52 dB）。

Vulkan VAE：ggml-vulkan 通过 F16 协作矩阵运算 F32 矩阵，而 VAE 的部分激活超过
65,504，因此此前在 `ggml_vulkan` 下 VAE 卷积一直回退到 CPU（512×512 解码超过 8 分钟）。
现在原生 F32 卷积在 Vulkan 上先把输入按 2 的整数次幂缩放到 32,768 以内，再把 F32 结果
还原，VAE 因而在 Vulkan 设备上运行：512×512 端到端从 683.5 秒降至 48.7 秒，与 F32 CPU
参考相比相对 L2 误差为 0.14%。

FP8 权重：TensorSharp 加载块量化的 GGUF 权重；ggml 没有 FP8 E4M3 张量类型，8 位
权重配置就是 Q8_0 GGUF。vLLM-Omni 的 FP8 工作中适用于这里的是 8 位前缀存储，
见上文。

## 验证记录

Unsloth Q8_0 验证、完整模型性能测量、原生优化验证、与 stable-diffusion.cpp 的
历史对比以及复现命令，请参阅[英文版模型卡](qwenimage21.md)。其中的编辑对比是在编辑噪声
改为由参考图决定之前完成的；要与 stable-diffusion.cpp 的编辑噪声对齐，现在需要
`TS_QWEN21_EDIT_NOISE=seed`（基准脚本的 `--edit-noise seed`，两个引擎都运行时的默认值）。
