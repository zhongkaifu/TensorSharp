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

## 启动 CLI

以下命令都在 TensorSharp 仓库根目录下运行。该配置在 Apple Silicon 上选择
`ggml_metal`。在已构建 CUDA 后端的 NVIDIA 机器上，追加 `--backend ggml_cuda`；
原生 CPU 后端为 `ggml_cpu`。缺失的模型会在启动时自动下载。把 `TENSORSHARP_MODELS`
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
像素面积大致相同的尺寸。要覆盖此行为，请同时设置宽度和高度，且都取 32 的倍数。
该模型支持[原生 2K 宽高比](https://github.com/QwenLM/Qwen-Image-2.1#supported-aspect-ratios)。
每张参考图以约 1 百万像素（若输出面积更小，则以输出面积）作为条件输入；把输出
提高到 2K 并不会同时把每张参考图的 VAE、视觉编码器和 Transformer 工作量翻四倍。

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
接受 `negativePrompt`、`targetArea`、`width`、`height`、`steps`、`cfg` 和 `seed`。
`targetArea` 控制自动尺寸选择；显式尺寸优先。省略 `width`、`height`、`targetArea`、
`steps` 和 `cfg` 时使用上文的模型默认值。`targetArea: 1048576` 选择约 1K 的输出，
同时保留自动宽高比选择。

启动服务端时传入 `--width` 与 `--height` 会改变这个默认尺寸。主机把它们发布为
`TS_QWEN_IMAGE_WIDTH` / `TS_QWEN_IMAGE_HEIGHT`，之后凡是既没有 `width`/`height`、也没有显式
`targetArea` 的图像请求都使用该尺寸，包括不发送尺寸的 Web UI 请求；此时编辑也不再沿用第一张参考图的
宽高比。请求自己设置了 `targetArea` 时保留它自己的几何设置。默认尺寸需要两个参数都设置。不是 32 倍数的
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
配方的插件不能叠加。风格与编辑插件不带配方，沿用检查点自己的调度。运行会在去噪前记录解析出的
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

这在模型支持的所有后端上运行：`ggml_metal`、`ggml_cuda`、`ggml_vulkan` 与 `ggml_cpu`。
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

服务端在启动时加载 `--lora` 插件组，并把它应用到每个生成与编辑请求；尚未实现按请求选择
LoRA。请求中的 `steps` 与 `cfg` 仍会覆盖插件的配方。在进程内，
`QwenImageModel.SetLoras(IReadOnlyList<LoraSpec>)` 为之后的请求替换插件组（空列表表示
移除）。新插件组会立即针对 Transformer 校验，失败时保留原来的插件组。

### 限制

- 插件只作用于 Qwen-Image-2.1；CLI 遇到其他模型时拒绝 `--lora`，服务端则记录一条警告，
  并在不带插件的情况下加载该模型。
- 每次运行只能有一个插件带采样配方，而带 sigma 的配方只能以它定义的步数运行。
- Qwen-Image-2.1-Fix 作者的工作流还使用了 APG、FreSca 以及 CFG 3 下的 `seeds_2` 采样器，
  TensorSharp 没有实现这些；DoRA 本身会被精确应用。
- Pruna 适配器在 1K 下训练。Fun-Acc 在 2048×2048 下训练，其模型卡指出小而密的文字以及部分
  编辑效果弱于 40 步的教师模型。

## 加速与内存

2.1 扩散 Transformer 以常驻的量化权重运行完整的 GGML 图；没有 CPU 权重流式加载
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

在 `ggml_cuda` 或 `ggml_vulkan` 上使用 `--tp N` 时，扩散 Transformer 按
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
历史对比以及复现命令，请参阅[英文版模型卡](qwenimage21.md)。
