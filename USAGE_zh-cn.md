# 使用方法
[English](USAGE.md) | [中文](USAGE_zh-cn.md)

> [TensorSharp](README_zh-cn.md) 文档的一部分。快速开始命令见 [README](README_zh-cn.md#快速开始)；配置文件见 [config/README.md](config/README.md)。

## 计算后端

| 后端 | 参数 | 适合场景 | 说明 |
|---|---|---|---|
| Direct CUDA/cuBLAS | `--backend cuda` | NVIDIA 推理与实验 | 通过 CUDA Driver API、cuBLAS GEMM、常用 Float32 PTX 内核（fill、unary、binary、ternary、activations、RMSNorm、softmax、RoPE/RoPEEx、SDPA、GQA prefill/decode、causal mask、gather/concat），以及受支持 GGUF 量化类型的原生量化 matmul/get-rows 加速推理；未实现的算子会回退到 CPU，同时保持张量语义。 |
| MLX Metal | `--backend mlx` | Apple Silicon（GGML Metal 之外的另一选择） | 基于 [mlx-c](https://github.com/ml-explore/mlx-c) 的 GPU 加速路径。实现了原生量化算子（Q4_K_M、Q8_0、Q5_K、Q6_K、IQ2_XXS、IQ4_XS、IQ4_NL、MXFP4 等无需反量化到 FP32）、融合 decode / prefill Metal kernel（融合 QKV 预处理、融合 gate+up+SiLUMul MoE、融合多维 KV 写入）、编译图 kernel、定期 `async_eval` 让 GPU/CPU 工作重叠的异步 worker 派发、用堆叠权重 slab 的批处理 MoE 解码、MoE 专家 offload、通过 `mlock(2)` 把 GGUF mmap 钉在物理内存、按宿主机派生的分配器上限（`TS_MLX_MEMORY_LIMIT_MB` / `TS_MLX_CACHE_LIMIT_MB` / `TS_MLX_WIRED_LIMIT_MB`），并对未实现的算子提供 CPU 回退。依赖 `libmlxc`（可通过 `TensorSharp.Backends.MLX/build-native-macos.sh` 在本地编译，或用 `TENSORSHARP_MLX_LIBRARY` / `TENSORSHARP_MLX_LIBRARY_DIR` 指定路径）。 |
| GGML Metal | `--backend ggml_metal` | Apple Silicon（macOS 默认） | 通过 Apple Metal 进行 GPU 加速。量化权重通过 host 指针缓冲区从 GGUF 文件零拷贝映射到 Metal command buffer，常驻内存接近模型在磁盘上的大小。 |
| GGML CUDA | `--backend ggml_cuda` | 通过 ggml 使用 NVIDIA 推理 | 通过 GGML CUDA 在 Windows 或 Linux + NVIDIA GPU 上进行加速。量化权重在加载时一次性上传到设备显存，之后释放主机端拷贝。 |
| GGML Vulkan | `--backend ggml_vulkan` | 通过 ggml 的厂商无关 GPU 推理 | 通过 GGML Vulkan 在 Windows 或 Linux 上加速——支持带 Vulkan 1.3 驱动的 AMD、Intel 与 NVIDIA GPU，驱动支持时使用 cooperative-matrix（KHR coopmat / NV coopmat2）着色器。权重与 GGML CUDA 一样常驻显存，并复用同样的融合整模型 decode/prefill 图。机器有 Vulkan 运行时（已安装 loader）时原生构建会自动启用；未安装 Vulkan SDK 或发行版开发包时，构建会通过 `eng/fetch-vulkan-toolchain.ps1` / `eng/fetch-vulkan-toolchain.sh` 自动下载便携工具链（headers、glslc、SPIRV-Headers，Windows 上还有 loader 导入库）。用 `--no-vulkan`（或 `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF`）退出。 |
| GGML CPU | `--backend ggml_cpu` | 原生 CPU 内核 | 使用原生 GGML 与优化内核进行 CPU 推理。量化权重以零拷贝方式从 GGUF 文件映射。 |
| 纯 C# CPU | `--backend cpu` | 可移植性与调试 | 无原生依赖的可移植 CPU 推理。托管矩阵乘跑在一个常驻的"自旋后挂起"工作线程池上，默认宽度是可用核心数的一半（`TS_CPU_THREADS` 及下文其余 `TS_CPU_*` 开关）；在 direct 视频网络（Wan、MiniMax-H3）上，量化权重直接以 GGUF 存储类型参与乘法，而不再在加载时展开成 F32（`TS_DIRECT_QUANT_WEIGHTS=0` 恢复展开）。DeepSeek V4.1 Flash 与下文的 DeepSeek V4 Flash 一样，在这里也走自己的整模型执行器——100% 纯 C# 的 `DeepSeek4CpuExecutor`，它是正确性与可移植性路径，而非服务路径，其计算宽度来自 `TS_DSV4_THREADS`（这个后端上默认取全部处理器）而不是 `TS_CPU_THREADS`。 |

**DeepSeek V4 Flash 是上表的例外。** 它那套 284B 的压缩稀疏注意力 MoE 结构不走通用的逐算子路径，而是使用三套专属的整模型执行器之一：Direct CUDA 引擎（`--backend cuda`）、原生 ggml 执行器（`--backend ggml_cuda` / `ggml_vulkan`），以及 100% 纯 C# 的 CPU 执行器（`--backend cpu`，直接从内存映射的 GGUF 分片提供量化权重）。三者都会把权重按层切分到所有可见 GPU（CPU 路径则从映射分片流式读取），因此远大于单卡显存的模型依然跑得起来。详见 [DeepSeek V4 卡片](docs/models/deepseek4_zh-cn.md)。

**DeepSeek V4.1 Flash（`deepseek41`）另有一套自己的托管整模型执行器。** `--backend ggml_cuda` 是它的服务路径，而 `--backend cpu` 会把整张 V4.1 计算图跑在 100% 纯 C# 的 `DeepSeek4CpuExecutor` 上——没有 ggml、没有原生库、也没有 GPU，因此 .NET 能跑的地方它都能跑：ratio-1 与 ratio-2 的块压缩器、共享的 compressed 与 indexer cache（含 lightning indexer 的 top-k）、候选块剪枝、Engram 表、延迟超连接门控、共享专家，以及检查点自带的训练后 cache 量化（raw 行为 FP8 E4M3、indexer 为 MXFP4、compressed 为 NVFP4）。它由独立的 PyTorch 参照实现 `eng/dsv41-reference.py` 在一个 5 层 F32 fixture 上以 `atol=rtol=2e-5` 把关，并在一次性 prefill、分块大小 1/3/5/8 与 reset 上要求贪心 argmax 完全一致；它是**正确性与可移植性路径，而非服务路径**——从未测量过 V4.1 完整检查点在这条路径上的吞吐、加载时间或常驻内存。`deepseek41.engram.bin` sidecar 依然是必需的；图像与视频在这里不可用，因为视觉伴随件是原生 ggml 组件，`--mmproj` 会抛异常；草稿模型 / `TS_DSV4_DSPARK`、分布式 TP 组、非 `0` 的 `TS_DSV41_TP` 与非 `0` 的 `TS_DSV41_ENGRAM_DEVICE`，都会在读取任何权重之前被拒绝。`--backend ggml_cpu` 是另一条路径——在标量 ggml CPU 内核上跑原生计算图。详见 [DeepSeek V4.1 卡片](docs/models/deepseek41_zh-cn.md)。

**GLM 5.x（`glm-dsa`）是另一个例外。** 它那套 744B-A40B 的 MLA + 稀疏注意力 MoE 在 `--backend ggml_cuda` / `ggml_vulkan` / `ggml_cpu` / `ggml_metal` 上走原生整模型 ggml 执行器，在 `--backend cpu`（100% 托管，无原生依赖）与 `--backend cuda` 上走托管逐算子路径；在 GGML 后端上设 `TS_GLM_NATIVE=0` 可切到托管路径做 A/B 对照。**GLM-5.3（非 Flash）与 GLM-5.2 是同一套 `glm-dsa` 块结构**——79 个 block（78 层主干 + 一个 NextN）、256 个路由专家 top-8 外加一个共享专家、带 lightning indexer 的 MLA、rope base 8e6——因此它就走 GLM-5.2 那条路径、同样这几个后端，不需要新代码也不需要新参数；它是**仅文本**，而且是两重意义上的仅文本：仓库在任何量化档都没有发布 mmproj，而在 `glm-dsa` 上给出 `--mmproj` 只会收到一条警告并被忽略，而不是让这次运行失败。分片 GGUF 由 `GgufFile` 自己处理——`--model` 指向该量化目录里 `-00001-of-` 那一片即可（GLM-5.2 是 6 分片，GLM-5.3 的 UD-Q2_K_XL 是 7 分片、236.4 GiB）。MLX 跑不了它。详见 [GLM 卡片](docs/models/glm_zh-cn.md)，两者的差别见其中的 [GLM-5.3 专节](docs/models/glm_zh-cn.md#glm-53glm-dsa)。



## 配置文件（CLI + Server）

`TensorSharp.Cli` 与 `TensorSharp.Server` 都可以通过 `--config` 从一个 JSON 文件读取参数，
以替代（或补充）冗长的命令行：

```bash
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --config config/server-basic.json
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll       --config config/cli-basic.json
```

**命令行参数始终优先。** 先应用文件中的值，命令行上再次给出的参数会覆盖它们——因此可以在多台机器上
复用同一个文件，只覆盖需要变化的部分（`--config config/server-basic.json --backend ggml_cpu`）。
`--config` 可重复出现以叠加多个文件；后出现的文件优先。

键名与下文列出的长选项名相同（可带或不带前缀 `--`）。允许注释（`//`、`/* */`）与尾随逗号。

| JSON 值 | 展开为 | 示例 |
|---|---|---|
| 字符串 / 数字 | `--key value` | `"max-tokens": 4096` → `--max-tokens 4096` |
| `true` | 裸开关 `--key` | `"continuous-batching": true` → `--continuous-batching` |
| `false` / `null` | 什么都不加（要关闭请用取反键，如 `"no-continuous-batching": true`） | |
| 数组 | 重复的标志 | `"stop": ["</s>", "<\|eot\|>"]` → `--stop </s> --stop <\|eot\|>` |
| 对象 | 可下载文件（见下） | `{ "path": "...", "urls": ["..."] }` |

**变量。** 在 `"variables"` 中定义一次共享值，用 `${name}` 在任意字符串值中引用。未定义的 `${name}`
会回退到同名环境变量，变量之间也可以互相引用。可以定义任意多个根路径——位于不同目录的模型各用一个。

```json
{
  "variables": { "modelRoot": "C:/models" },
  "backend": "ggml_cuda",
  "model": "${modelRoot}/Qwen3.5-9B-Q8_0.gguf",
  "mmproj": "${modelRoot}/Qwen3.5-mmproj-F16.gguf"
}
```

**自动下载。** 任何文件参数都可以写成带本地 `path` 与一个或多个 `urls` 的对象，而非普通字符串。
若 `path` 不存在，则从第一个可用 URL 下载（镜像按顺序尝试），保存到该路径，之后每次运行复用；
下载进度打印到 stderr。可选的 `sha256` 用于校验刚下载的文件。

```json
{
  "backend": "ggml_cuda",
  "model": {
    "path": "C:/models/Qwen3.5-9B-Q8_0.gguf",
    "urls": [ "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q8_0.gguf" ]
  }
}
```

开箱即用的示例见 [`config/`](config/)（`cli-basic.json`、`server-basic.json`、`variables.json`、
`auto-download.json`、`qwen-image-edit.json`）——每个都使用真实、公开、无需授权的 URL，
因此在全新机器上也能直接运行。完整说明见 [`config/README.md`](config/README.md)。

## 控制台应用

不带任何参数运行 `TensorSharp.Cli` 会打印完整的参数参考——逐项列出说明、默认值、
取值范围与示例——并在日志与模型机制启动之前退出；`--help`（以及 `-h`、`-?`、`/?`）
效果相同。

```bash
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --help
```

```bash
# 文本推理
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --output result.txt \
    --max-tokens 200 --backend ggml_metal

# Windows/Linux + NVIDIA GPU 文本推理
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --output result.txt \
    --max-tokens 200 --backend ggml_cuda

# 交互式逐轮对话（REPL），支持 KV 缓存复用与斜杠命令
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_metal --interactive
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_metal -i \
    --system "你是一名简洁的助手。" --temperature 0.7 --top-p 0.9 --think

# 图像推理（Gemma 4，Qwen 3.5-family）
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --image photo.png --backend ggml_metal

# 视频推理（Gemma 4）
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --video clip.mp4 --backend ggml_metal

# 音频推理（Gemma 4）
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --audio speech.wav --backend ggml_metal

# PDF 文档输入：文字型 PDF 会提取文本层并内联进提示词；扫描型 PDF 会转成页面
# 图像，需要视觉模型（--mmproj 或内置视觉编码器）。--input 提供针对文档的指令。
echo "Summarize the key findings of this paper." > question.txt
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --pdf paper.pdf --input question.txt \
    --max-tokens 300 --backend ggml_metal

# DiffusionGemma 文本扩散生成
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <diffusion-gemma.gguf> --input prompt.txt --backend ggml_metal \
    --max-tokens 256 --diffusion-steps 48 --diffusion-seed 0

# Qwen-Image-Edit 图像编辑（提示词 + 输入图像 -> 编辑后的图像）
# VAE + Qwen2.5-VL 文本编码器伴随文件会在 DiT GGUF 旁解析
# （或用 --qwen-image-vae / --qwen-image-vl / --qwen-image-mmproj 指定）。
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <qwen-image-edit-DiT.gguf> --image input.png \
    --prompt "Make the sky a dramatic sunset." --output edited.png \
    --backend ggml_cuda --diffusion-steps 30 --cfg 2.5 --diffusion-seed 0

# MiniMax-H3 带声音的视频生成（提示词 -> H.264 MP4，外加一个 32 kHz 立体声 .wav
# 旁挂文件）。一个扩散 Transformer 在同一条 token 序列上对打包好的“视频+音频”
# 潜变量一起去噪，因此音轨是模型输出本身，而不是事后配上去的。Qwen3-VL-32B 文本
# 编码器、视频 VAE 与音频 VAE 会在 DiT GGUF 旁解析（或用 --video-text-encoder /
# --video-vae / --audio-vae 指定）。H3 是 CFG 蒸馏模型：必须 `--cfg 1.0`，默认 20
# 步，4-8 步是快速档。详见下文“音视频生成（MiniMax-H3）”与
# docs/models/minimax-h3_zh-cn.md。
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <minimax_h3_fl2va_pruned-Q4_K.gguf> \
    --prompt "a red fox trotting through falling snow, cinematic" --output fox.mp4 \
    --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
    --backend ggml_cuda

# MiniMax-H3 的条件输入：在 fl2va 检查点上 --image 就是第一帧，--end-image 钉住末
# 帧；在 ref2va 检查点上，同一张图片是用于全新场景的身份参考（--ref-image，最多
# 九个，另有 --ref-video / --ref-video-audio / --ref-audio）。含义不明确时用
# --video-mode t2v|i2v|fl2v|ref 显式声明。
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <minimax_h3_fl2va_pruned-Q4_K.gguf> \
    --image start.png --end-image end.png --video-mode fl2v \
    --prompt "a slow cinematic push-in" --output morph.mp4 \
    --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
    --backend ggml_cuda

# 同一件事也可以交给随仓库提供的配置文件，首次运行会自动下载全部四个网络
# （config/minimax-h3-ref2va.json 对应参考检查点）。
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --config config/minimax-h3-fl2va.json \
    --prompt "a red fox trotting through falling snow, cinematic" --output fox.mp4

# Wan 视频生成，仅视频（提示词 -> H.264 MP4）。UMT5-XXL 文本编码器 GGUF 与视频 VAE
# 伴随文件会在 DiT GGUF 旁解析（或用 --video-text-encoder / --video-vae 指定）。
# Wan 2.1 T2V、Wan 2.2 TI2V-5B 与 Wan 2.2 A14B（两个专家）都会自动识别；
# 步数蒸馏（Turbo / Lightning / FastWan）检查点同样自动识别——同一段视频只需
# 4 次 DiT 前向而不是 100 次。详见下文“视频生成（Wan）”与 docs/models/wan_zh-cn.md。
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <Wan2.2-TI2V-5B.gguf> \
    --prompt "a lovely cat walking through a garden" --output cat.mp4 \
    --width 832 --height 480 --video-frames 49 --backend ggml_cuda \
    --diffusion-seed 7

# Wan 2.2 图生视频：--image 提供首帧，提示词控制运动、镜头与场景变化
# （TI2V-5B 或 I2V-A14B 检查点）。
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <Wan2.2-TI2V-5B.gguf> \
    --prompt "the cat runs toward the camera, cinematic tracking shot" \
    --image first_frame.png --output cat_run.mp4 --backend ggml_cuda

# 思维链 / 推理模式
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend ggml_metal --think

# 工具调用
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --tools tools.json

# Agent Skills（智能体技能）：注册一个技能目录，并为本次运行选中其中一个。前期只有
# 每个技能的一行描述会占用上下文——SKILL.md 正文与所需的参考文件由模型通过内置的
# skills_list / skills_read 工具按需自取，而这两个工具由 TensorSharp 在进程内应答。
# --list-skills 打印注册表（名称、描述、文件、告警）后退出。
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --skills-dir ~/skills --skill pdf

# 使用采样参数
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.2 --seed 42

# DSpark 块级投机解码（DeepSeek V4）：--model 指向第一个分片，--draft-model 指向
# DSpark 草稿器 GGUF。每个输出 token 仍然取自主干的某一行，因此贪心运行逐字节不变、
# 带采样的运行同分布——下面的 --temperature 0 只是为了让对比完全一致。
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <DeepSeek-V4-Flash-...-00001-of-00005.gguf> \
    --backend ggml_cuda --draft-model <DSpark-drafter.gguf> \
    --input prompt.txt --max-tokens 200 --temperature 0

# 批处理（JSONL）
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input-jsonl requests.jsonl \
    --output results.txt --backend ggml_metal

# 多轮对话模拟（含 KV 缓存复用，模拟 Web UI 行为）
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --multi-turn-jsonl chat.jsonl \
    --backend ggml_metal --max-tokens 200

# 吞吐基准测试：N 次最优运行的 prefill 和 decode 计时
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_metal \
    --benchmark --bench-prefill 256 --bench-decode 128 --bench-runs 3

# KV 缓存复用基准：在多轮对话中比较启用与禁用缓存的 prefill 时延
# （以一个 8 轮的对话为例，对比有缓存与强制重置的 prefill 延迟差异）
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_metal \
    --bench-kvcache --bench-kv-turns 4 --max-tokens 64

# 仅查看渲染后的 prompt 和分词结果（不运行推理）
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --dump-prompt

# 对目录下每个 *.gguf 文件，对比硬编码回退模板与 GGUF 内置 Jinja2 模板
# （在适配新架构时尤其有用）
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --test-templates ~/models

# 张量并行：在单个进程内把模型切分到 2 张 GPU 上
# （--backend cuda、ggml_cuda 或 ggml_vulkan）
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend cuda --tp 2

# 分布式张量并行：2 节点 × 每节点 2 GPU（共 4 GPU）
# 节点 0：
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2 \
    --tp-node-id 0 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"
# 节点 1：
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2 \
    --tp-node-id 1 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"
```

**命令行参数：**

| 参数 | 说明 |
|---|---|
| `--model <path>` | GGUF 模型文件路径（必填） |
| `--input <path>` | 包含用户提示词的文本文件 |
| `--input-jsonl <path>` | JSONL 批量请求文件（每行一个 JSON） |
| `--multi-turn-jsonl <path>` | 用于多轮对话模拟（含 KV 缓存复用）的 JSONL 文件 |
| `--output <path>` | 将生成文本写入该文件 |
| `--image <path>` | 用于视觉推理的图像文件 |
| `--video <path>` | 用于视频推理的视频文件 |
| `--audio <path>` | 音频文件（WAV、MP3、OGG）用于音频推理 |
| `--pdf <path>` | PDF 文档输入（单次推理模式）。文字型 PDF 会提取并内联完整文本层（页数上限由 `TS_PDF_MAX_PAGES` 控制）；扫描型 PDF 会栅格化为页面图像，并需要视觉模型（`--mmproj` 或内置视觉编码器）。`--input` 文本作为针对文档的指令。 |
| `--mmproj <path>` | 多模态投影器 GGUF 文件路径 |
| `--max-tokens <N>` | 最大生成 token 数（默认：100） |
| `--backend <type>` | 计算后端：`cpu`、`cuda`、`mlx`、`ggml_cpu`、`ggml_metal`、`ggml_cuda` 或 `ggml_vulkan` |
| `--gpu-device <N>` | `ggml_vulkan` 后端使用的 Vulkan 设备索引，用于多 GPU 主机（例如同时装有 Intel 集成显卡和 NVIDIA 独立显卡的机器）。默认使用设备 0；可用 `--list-gpus` 查看索引。也可通过环境变量 `TS_GGML_VULKAN_DEVICE` 设置。 |
| `--list-gpus` | 列出 ggml-vulkan 可见的 Vulkan 设备（索引 + 显卡名称）后退出 |
| `--n-cpu-moe <N>` / `-ncmoe <N>` | 把前 N 层的路由 MoE 权重留在系统内存里并在 CPU 上做乘法；注意力、norm、路由器与始终活跃的共享专家仍留在加速器上（等价于 llama.cpp 的 `--n-cpu-moe`）。正是它让 35B-A3B 这类 MoE 能与长上下文 KV 缓存一起塞进 12–16 GB 的显卡。传 `all` 表示所有层。默认：所有架构（含 DeepSeek V4 与 GLM 5.x）都是 0 —— 装不下的模型会在加载时被拒绝并告知需要卸载多少层，而不会被悄悄卸载（环境变量 `TS_N_CPU_MOE`）。详见[混合专家 CPU 卸载](#混合专家-cpu-卸载--n-cpu-moe)。 |
| `--cpu-moe` / `-cmoe` | `--n-cpu-moe all` 的简写。默认：关闭（环境变量 `TS_CPU_MOE`）。 |
| `--cpu-moe-threads <N>` | 主机侧专家矩阵乘的工作线程数。默认：在核数多于 8 的主机上取本进程实际可用 CPU 并行度（`hardware_concurrency`，再受调度亲和性掩码与 cgroup CPU 配额约束）的**一半**，低于 8 时取「全部减一」。另一半并非浪费——加速器提交线程，以及 `TensorSharp.Server` 里的 Kestrel 与调度器，同样需要可被调度，而 .NET 自己的线程池是按机器 CPU 数而不是 cgroup 配额来定的。把它设到接近配额是悬崖而不是缓坡：在 95 CPU 配额下，托管的 26B MoE 在 64 线程时实测 20.7 tok/s，71 线程时只剩 8.2。独占机器上可以调高（环境变量 `TS_CPU_MOE_THREADS`）。 |
| `--kv-cache-dtype <type>` | KV 缓存精度：`f32`（默认）、`f16`、`q8_0` 或 `q4_0`。量化 / 半精度 KV 缓存以微小数值漂移换取内存节省；`q4_0`（约 0.56 字节/元素，约为 f32 的 1/7）是最激进的档位，面向 KV 缓存占主导内存的超长（128K–256K）上下文。块量化缓存（`q8_0`/`q4_0`）需要原生 GGML flash 路径。 |
| `--tp <N>` | 多卡度 —— 单个进程内把模型摊到几张 GPU 上（默认：`1`）。到底走哪一种多卡模式由架构决定，而不是由你决定：实现了张量并行的走**张量并行**（在层*内部*切权重），Qwen 3.8 Flash Next（`qwen4exp`）与 DeepSeek V4 走**按层切分**（整层落在单卡 —— 买的是容量，不是速度）。GLM 5.x 不传此参数时按层切分；在 GGML GPU 后端上，传入它则为 GLM-5.2、GLM-5.3 与 GLM-5.3-Flash 一律选择原生本地单进程 TP（在 GLM-5.3 上这只是一个被接受的模式，而不是已验证的配置——各 rank 会复制 cache——而且只要 `--tp N>1`，`--spec` 就会被拒绝，投机只在默认的按层切分下生效）。两种模式都不支持的架构会在 stderr 上明确说明并只用一张卡。需要 `--backend cuda`、`ggml_cuda` 或 `ggml_vulkan`。详见[张量并行与分布式推理](#张量并行与分布式推理)。 |
| `--tp-node-id <N>` | 多节点分布式张量并行中本节点的 0 起始编号。必须与 `--tp-peers` 一起使用。 |
| `--tp-peers <list>` | 集群中所有节点的 `host:port` 列表（逗号分隔，例如 `192.168.1.10:9500,192.168.1.11:9500`）。所有节点必须使用完全相同的列表。必须与 `--tp-node-id` 一起使用。 |
| `--interactive` / `-i` / `--chat` | 进入交互式 REPL 聊天会话（逐轮输入/输出），支持 KV 缓存复用、斜杠命令、运行时热切换 模型/后端/投影器、文件附件（图像、音频、视频、文本）以及实时调整采样参数。完整命令列表见下文「**交互式 REPL 命令**」一节 |
| `--no-prefix-cache` | 不在第一条消息之前先把提示词中共享的那部分前向一遍。默认情况下，交互式会话在启动时就会前向它的 system 块、工具声明与技能目录，于是第一条消息可以接着它们继续，而不必重新 prefill（在一个 agent 配置上实测 18.5s → 0.3s）；代价是会话需要这么久才就绪。它与 `--warmup-runs` 无关 |
| `--system <text>` | 用于初始化交互式会话的系统提示词（在 REPL 中可用 `/system` 覆盖） |
| `--system-file <path>` | 从 UTF-8 文本文件读取初始系统提示词（`--system` 的替代写法） |
| `--think` | 启用思维链/推理模式。所有系列（含 GLM 5.x）都是按需开启：不加时 GLM 的模板会把推理块立刻闭合（`<think></think>`），模型直接作答；加上后提示里会带上 `Reasoning Effort: Max`，并留下一个未闭合的 `<think>` 交给模型自己收尾。REPL 里用 `/think on\|off` 切换。 |
| `--tools <path>` | 包含工具/函数定义的 JSON 文件。线格式因系列而异，解析器按架构选取——GLM 5.x 发出的是 XML（`<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`，每个参数一个元素，非纯字符串的值以 `tojson` 编码）而不是 JSON 主体，服务端会把它解析回常规的 OpenAI 工具调用字段，因此客户端看到的仍是标准形状。 |
| `--skills-dir <path>` | 扫描 Agent Skills 的目录（可以是装着若干 `SKILL.md` 的文件夹，也可以是单个技能目录）。可重复；按给出的顺序扫描，最深三层。不传时使用二进制文件旁的 `skills` 目录，不存在则创建。传入的路径不存在会在启动时报错并点名该参数。环境变量：`TS_SKILLS_DIR`（以路径分隔符分隔的列表）。 |
| `--skill <name>` | 为本次运行选中一个技能，名称取自其 `SKILL.md`（也就是它的目录名）。可重复。对支持工具调用的模型族，选中只会展示技能元数据并把内置技能工具的可达范围限定到该技能，**不会**把 `SKILL.md` 正文直接写入提示词。只有模型族或请求无法使用工具声明时（包括结构化输出请求），才会在预算允许时内联正文作为回退。 |
| `--list-skills` | 打印技能注册表——名称、描述、来源、随包文件、大小，以及任何加载告警或错误——然后退出。 |
| `--no-skills` | 彻底关闭 Agent Skills：不扫描、不注入提示词块、不提供工具。环境变量：`TS_NO_SKILLS`（非 `0` 即视为开启）。 |
| `--skills-no-discovery` | 不向模型展示未被选中的技能。不加时，所有已注册技能的名称与描述都会列出，好让模型自己发现你没想到要点名的那个；加上后，本次运行只看得到 `--skill` 选中的技能。 |
| `--skills-allow-exec` | 允许模型通过 `skills_run` 运行选中技能自带的脚本。**默认关闭；开启后应把每个脚本都当作不受信任代码。**TensorSharp 仅调用已知解释器（`.py`、`.js`/`.mjs`、`.sh`、`.bash`），不经过 shell，并会清理环境、限制运行时间与输出、以只读方式挂载技能目录；工作目录是当前 Web/CLI 会话或 OpenAI/Ollama 请求的工作区（未开启代码执行时则是单次调用 scratch 目录）。默认的 `--skills-sandbox required` 会增加操作系统隔离，无法隔离时拒绝执行；除非另传 `--skills-allow-network`，否则网络仍被禁止。环境变量：`TS_SKILLS_ALLOW_EXEC`（非 `0` 即视为开启）。 |
| `--skills-max-rounds <n>` | 模型在必须作答前可以取用技能内容——或运行、查看并修正代码——的次数，范围 1-64（默认：`8`；开启 `--code-exec` 时为 `24`，因为写程序、运行它并按回溯修正它比读文件需要更多轮次；此处显式设定的值一律按原样使用）。每一轮都是一次完整生成，因此这一项限制的是那种反复把文件名叫错的模型所能造成的开销。预算用尽时会在对话里告知模型，让它用已读到的内容作答，并说明已经完成了哪些部分。环境变量：`TS_SKILLS_MAX_ROUNDS`。 |
| `--skills-sandbox <off\|preferred\|required>` | 对技能脚本要求多强的操作系统级隔离。`required`（默认）在没有安全沙箱的宿主机上拒绝运行；`preferred` 表示显式接受较弱隔离；`off` 只保留解释器、环境、时限与输出限制。macOS 使用 `sandbox-exec`；Linux 要求 `bwrap` 0.12.0 或更高版本（旧版存在已知的沙箱搭建阶段符号链接逃逸）。Windows Job Object 只能限制进程树，不能限制文件系统或网络，因此 `required` 会拒绝，必须显式选 `preferred` 才接受这种较弱隔离。环境变量：`TS_SKILLS_SANDBOX`。 |
| `--skills-allow-network` | 允许沙箱内的技能脚本联网。默认禁止。环境变量：`TS_SKILLS_ALLOW_NETWORK`。 |
| `--code-exec` | 向模型提供 `shell`、`read_file`、`edit_file`、`write_file` 与 `apply_patch` 五个工具。`shell` 执行真正的命令行并返回退出码与合并后的 stdout/stderr；文件工具由宿主完成精确读取、替换、整文件写入与原子补丁。五个工具在内部 Agent 轮次中共享同一工作区：Web/CLI 按聊天会话保留；每个 OpenAI Chat、OpenAI Responses 或 Ollama HTTP 请求则获得一个私有工作区，响应结束后删除。只有没有提供工作区的底层直接调用者才只会得到 `shell`。所有内置工具均由宿主进程内应答，不会交回 API 客户端。默认关闭。环境变量：`TS_CODE_EXEC`（非 `0` 即视为开启）。 |
| `--code-exec-allow-install` | 允许模型把 pip / npm 包装进当前 Web/CLI 会话或 HTTP 请求工作区，让后续命令与技能脚本在该工作区寿命内都能 import；需要 `--code-exec`。这项权限**不会**把套接字交给模型生成的命令。宿主会从识别到的安装命令中读出工具名与包名、完成校验，再用自己构造的参数向量代为安装，只接受预编译 wheel 并禁止安装脚本；随后用 `true` 或 `false` 替换原安装命令，保留 `&&`、`\|\|`、管道和循环的语义。在 `pip install x && python y.py` 中，`y.py` 遵循命令的联网策略：默认离线，只有另行传入 `--code-exec-allow-network` 才可不受限地访问主机网络。会改变软件源的参数、URL 依赖以及宿主无法代为执行的安装器都会被拒绝；`-r requirements.txt` 会逐行读取并校验。环境变量：`TS_CODE_EXEC_ALLOW_INSTALL`。 |
| `--code-exec-allow-network` | 给予每一条模型生成的命令不受限的宿主 IP 网络访问；生成的代码可以解析 DNS、抓取 URL、跟随重定向、调用远程 API、访问局域网/回环服务并打开 IP 监听套接字。**默认关闭**，且需要 `--code-exec`。macOS 与 Linux 上的写入及主目录读取约束仍保持生效。Linux 还通过 PID 命名空间约束后代进程；macOS 子进程会继承 Seatbelt，普通进程组也会被清理，但主动脱离进程组的子进程可能在请求结束后继续运行，每次工具结果都会明确报告这一限制。macOS 仍拒绝常见的 `/private/tmp/com.apple.launchd*` 路径套接字（但保留系统运行时所需的 Mach lookup 与 DNS 必需的精确 mDNSResponder 路径套接字），Linux 仍隐藏常见的 `/run` 端点，但本地 Unix IPC 并非完整隔离边界：macOS 为兼容性保留共享临时目录内的 Unix IPC，Linux 的宿主网络命名空间可能暴露抽象套接字以及 `/run` 之外的路径名套接字。其他宿主可读文件及 IP 服务因而可能被访问并外传；远程提示词注入与不可信下载也是额外风险。只有不含凭据的宿主 HTTP/SOCKS 代理设置会在此模式下传入。宿主会把不超过 16 MiB 的自定义 CA 包只读取一次，并仅将验证过的公开证书复制到每会话只读快照，因此源路径及其相邻数据不会暴露；认证代理需使用宿主侧无凭据转发器。包名与安装域名允许列表只约束宿主识别并代办的安装；不受限的生成代码仍可直接抓取或执行制品。它与装包权限、以及只控制 `skills_run` 的 `--skills-allow-network` 相互独立。Windows 仍须传入 `--code-exec-unconfined`。环境变量：`TS_CODE_EXEC_ALLOW_NETWORK`（非 `0` 即视为开启）。 |
| `--code-exec-packages <list>` | 把安装限制在这些包名之内，逗号分隔；其余一律拒绝，并告诉模型允许哪些名字（默认：留空，即不限包名——且无论如何单次安装最多 16 个包）。匹配的是去掉版本后的裸名字，因此模型钉住的版本（`numpy==2.1.0`）仍然匹配名单里的 `numpy`。工具接口变成 shell 时它曾被退休，因为模型自己敲 `pip install` 时，可以用包名白名单看不见的方式写出同一个请求；如今它回来了，因为安装重新由**宿主**执行：宿主是把包名从模型的命令里读出来，而不是去运行那条命令，安装器的参数向量由宿主自己拼出，所以无论请求写成 `pip`、`pip3`、`python -m pip` 还是一个 requirements 文件，这份名单都同样生效。只有配合 `--code-exec-allow-install` 才有意义。启用 `--code-exec-allow-network` 后，该名单不是安全边界：生成代码可绕过宿主安装器，直接抓取或执行制品。 |
| `--code-exec-install-index <url>` | 让宿主代办的安装使用运维者指定的软件包索引，而不是安装器默认源；模型自己写出的 `--index-url` 仍会被拒绝。索引主机会自动加入安装出口允许列表；镜像若从另一个主机下载文件，还需把该主机列入 `--code-exec-install-domains`。需要 `--code-exec-allow-install`。默认：未设置。环境变量：`TS_CODE_EXEC_INSTALL_INDEX`。 |
| `--code-exec-install-domains <list>` | **宿主代为执行的装包操作**可以到达的主机，逗号分隔，可用精确名称或 `*.suffix` 通配（默认：`pypi.org,files.pythonhosted.org,registry.npmjs.org`；空值关闭钉选）。macOS Seatbelt 能把安装器钉在单个 loopback 端口上，因此它必须走一次性的 CONNECT 代理且只能访问这些主机；bubblewrap 无法钉住端口，Linux 则依赖宿主构造的参数向量、默认软件源与只装 wheel 的策略。传入 `--code-exec-allow-network` 后，普通生成命令获得的是不受限主机网络访问，不受此安装域名列表约束。环境变量：`TS_CODE_EXEC_INSTALL_DOMAINS`。 |
| `--code-exec-timeout <seconds>` | 单条命令被杀死前最多可运行的秒数（默认 `120`，而不是旧程序运行器的 30 秒，因为现在装包、跑构建、跑测试用的都是同一个 shell 命令）。单次调用可以用 `timeout_ms` 要求更短或更长，上限 10 分钟；超时的命令会被停掉，但它在此之前打印的一切仍会交给模型——丢掉输出的超时只会让它闭着眼睛把命令再跑一遍。 |
| `--code-exec-temperature <number>` | 代码执行轮次可选的采样温度，范围 `0` 到 `2`。只改动仍处于 TensorSharp 内置默认值的采样项，因此客户端或运维者显式指定的温度仍优先；负值清除此覆盖。即使未设置温度，代码轮次也会去掉内置的聊天重复惩罚。默认：未设置。 |
| `--code-exec-shell <path\|name>` | 宿主自己的选择不合适或找不到时，用哪个 shell 来执行命令。默认：macOS 与 Linux 上是 `bash`，其次 `sh`；Windows 上是 PowerShell 7（`pwsh`），其次 Windows PowerShell——在 Windows 上，PATH 里那个裸的 `bash` 会被**故意**拒绝，因为它是 WSL 启动器，经由它执行会把命令送进一个 Linux 虚拟机，而 job object 在 Windows 这一侧握住的只是那个启动器。要在那里用真正的 bash（Git Bash、MSYS2），就用这个参数指到它。模型会被告知自己正在敲的是哪一种方言，因此这一项也会改变它工具说明里的示例。 |
| `--code-exec-max-output <bytes>` | 一条命令的输出保留并展示给模型的字节数（默认 `32768`）。放不下的部分是从**中间**丢弃的，头尾都保留：构建或测试跑到最后的那一段才是失败所在，而只保留开头的截断丢掉的恰恰是最需要的那一段。 |
| `--code-exec-unconfined` | 即使操作系统无法隔离，也照样运行模型生成的命令。CLI 与服务端都接受这一显式逃生开关。Windows 必须使用它，因为 job object 只能限制进程树，无法约束文件系统或网络；命令因而会获得本进程账户对两者的访问权限，不受 macOS / Linux 上那个较窄的联网开关约束。不要在不受信任用户可以访问的服务端上启用。 |

**shell 是怎么被用的，又能碰到什么。** 模型的全部工作都走这一行命令：用
heredoc 写出文件、运行它、grep 它、读自己的回溯再把它改掉——这也正是开启本
功能后 `--skills-max-rounds` 会从 8 升到 24 的原因。Web/CLI 在整个聊天会话内保留
一个文件工作区；每个 OpenAI Chat、OpenAI Responses 或 Ollama HTTP 请求则在其内部
工具轮次间保留一个私有工作区，并在响应结束后删除。开启代码执行后，技能脚本共享
该工作区，因此文件与宿主安装的软件包会在相应会话或请求寿命内可用。每次 shell
调用仍是全新的受限进程；不要依赖 virtualenv 激活状态、PATH 修改或常驻 shell 在调用
之间延续。联网规则属于宿主而不属于模型：命令默认离线，只有运维者显式传入
`--code-exec-allow-network`（或设置 `TS_CODE_EXEC_ALLOW_NETWORK`），每一条模型生成
的命令才会获得不受限的宿主 IP 网络访问，包括局域网/回环服务与 IP 监听套接字。
macOS 与 Linux 上的写入及主目录读取约束仍然生效。Linux 还通过 PID 命名空间约束
后代进程；macOS 子进程会继承 Seatbelt，普通进程组也会被清理，但主动脱离进程组的
子进程可能在请求结束后继续运行，每次工具结果都会明确报告这一限制。macOS 仍拒绝常见的
`/private/tmp/com.apple.launchd*` 路径套接字（但保留系统运行时所需的 Mach lookup 与 DNS 必需的精确 mDNSResponder 路径套接字），
Linux 仍隐藏常见的 `/run` 端点，但本地 Unix IPC 并非完整隔离边界：macOS
为兼容性保留共享临时目录内的 Unix IPC，Linux 的宿主网络命名空间可能暴露抽象
套接字以及 `/run` 之外的路径名套接字。生成代码仍可外传其能读取的其他宿主数据。
它与只控制 `skills_run` 的 `--skills-allow-network` 相互独立；
`--code-exec-allow-install` 也不会隐式打开命令网络。安装仍由宿主亲自执行——命令行
会被拆成各个简单命令（引号与 heredoc 正文都被正确对待），每一条被
识别出来的安装命令都会被读出工具名与包名，再由宿主用自己拼出的参数向量去调用
安装器；那条命令随后会在命令行里被替换掉——装成功换成 `true`，装失败换成
`false`——因此模型写在它周围的 `&&`、管道与循环体仍然分毫不差地是原来的意思：
安装失败时 `&&` 后面的东西不会运行，而 `||` 后面的后备方案会运行。无论哪种情况，
失败原因都会被报回来。把套接字交给安装命令是先前的做法，
它有两个无法同时堵上的窟窿：命令行出自模型之手，于是 `--index-url` 决定了用哪个
软件源；而套接字属于整行命令，于是与安装同处一行的任何东西都共享它能到达的
范围——在沙箱无法把出口钉死在代理上的宿主机上，那个范围就是整个互联网。
每次结果还会说明本机上有哪些项**未被约束**。已退休的 `--code-exec-languages`
会在启动时被按名字拒绝，并且没有替代品可指：shell 能够到达 PATH 上的每一个
解释器，而 PATH 必须包含 `/bin` 与 `/usr/bin` 这个 shell 才跑得起来，所以
`--code-exec` 现在改为如实报告本机有哪些解释器，而不是假装能把它们挡在门外——
于是旧脚本得到的是这条报错，而不是一个被悄悄忽略的设置。

| 参数 | 说明 |
|---|---|
| `--spec` / `--no-spec` | 启用投机解码（默认关闭）。`--spec` 是内嵌在主干检查点里的草稿器（GLM 5.2 的 NextN 块、GLM-5.3 的同款，以及 Qwen 3.6 的，无需额外下载任何文件）的显式开关——因为加载它们要把额外的权重调入显存；以独立 GGUF 发布的草稿器只需 `--draft-model` 即可启用，显式的 `--no-spec` 则对两者都是否决。草稿头每步最多起草 `--spec-draft` 个 token，主干用一次批量前向完成验证；每个输出 token 仍然取自主干的某一行，因此得到的 token 流与普通 decode 本该产生的完全一致（贪心配置下为 argmax，带采样器时同分布），这纯粹是一条加速路径。在所有单序列路径（`--input`、`--input-jsonl`、`--multi-turn-jsonl`、`--interactive`）上生效。**必须在模型加载之前就出现在命令行上**：对 glm-dsa 而言，正是它告诉原生加载器把约 3 GiB 的 NextN 层调入显存，而这一层要与 KV 缓存争抢上下文长度所依据的那块内存。若权重的草稿块复用主干的 LM head（GLM 5.2 与 GLM-5.3 即是如此），在 `--tp N>1` 下会被拒绝——在它们上面，投机只在默认的按层切分（不传 `--tp` 时）下生效。环境变量：`TS_SPEC`（旧写法 `TS_MTP_SPEC` 仍被 glm-dsa 的原生加载器读取；glm-dsa 还认 `TS_GLM_MTP=1`/`0`，它会覆盖上述两者，便于 A/B 对比）。 |
| `--spec-type <name>` | 投机**算法**：`auto`（默认，使用检查点自带的草稿器）、`draft-head`、`block` 或 `ngram`。`ngram` 不需要任何训练权重，对所有模型都能用——它的做法是在上文里找最近几个 token 曾经出现过的位置，把当时紧随其后的内容拿来当草稿，因此凡是答案大量引用输入的场景都很强（摘要、改写、翻译、重复性的结构化输出、Agent 循环），其他场景则退回普通 decode。在 Qwen3.5-9B（Q8_0、`ggml_metal`、M5 Pro）这个完全不带草稿头的检查点上实测 45.2 tok/s，对比普通解码的 31.4 tok/s（1.44x），输出逐字节一致。环境变量：`TS_SPEC_TYPE`。参见 [TensorSharp 的投机解码](docs/speculative_decoding.md)。 |
| `--spec-draft <N>` | 每个投机步最多起草的 token 数（取值 1-64，默认 `8`）。它同时决定加载时原生计算图缓存的大小，因此请与 `--spec` 一并显式传入，而不要依赖默认值。块级草稿器还会把它夹到自己训练时的块大小以内，且在那里默认即为该块大小：DSpark 为 5，Muse-Glimmer 的 DFlash 为 15，Qwen 3.8 的 DFlash2 为 7。在**递归**主干（Qwen 3.5/3.8 的 GatedDeltaNet 层）上，窄窗口远比宽窗口划算：它同时限制验证宽度与回滚重算，`--spec-draft 3` 在 Qwen3.8-27B 上比默认值快 1.6 倍。另外，在 GGML 后端上对 Qwen 3.5 显式传入 8 或更大是一个已知的**正确性**缺陷——验证批达到九行时会偏离普通贪心解码——因此在那里请保持 7 或更小；见[投机解码](docs/speculative_decoding.md)。环境变量：`TS_SPEC_DRAFT`（或 `TS_MTP_DRAFT`）。 |
| `--spec-pmin <f>` | 草稿置信度门限，取值 `[0, 1]`；遇到第一个低于该值的 token 即停止起草，`0` 表示从不设门限。这个数字*意味着什么*由算法自己决定，因此各算法带各自的默认值：逐 token 草稿头为 `0.15`（其 top-10 logits 上的 top-1 概率），块级草稿器为 `0.35`——后者的门限是**累积**前缀概率，即置信度头各位置估计值的乘积，因此同一个数字要严格得多（调低会起草更远、回滚更多，调高则更早退回普通 decode），n-gram 为 `0`（在那里它转而缩放所需的匹配长度）。环境变量：`TS_SPEC_PMIN`（或 `TS_MTP_PMIN`）。 |
| `--draft-model <path>` | 投机解码草稿 GGUF，适用于所有以独立文件发布的草稿器——DeepSeek V4 的 DSpark 支持模块（见 [DeepSeek V4](docs/models/deepseek4_zh-cn.md#dspark-投机解码)）、Muse-Glimmer 的 DFlash 与 Qwen 3.8 的 DFlash2 块级草稿器（见 [Muse-Glimmer](docs/models/muse-glimmer_zh-cn.md#3-dflash-投机解码)，环境变量 `TS_MUSE_GLIMMER_DFLASH`），以及 Gemma 4 的 `gemma4-assistant` 逐 token 草稿头。文件自己的 `general.architecture` 决定它如何加载——你从不需要挑选机制。在这里给出文件本身就会启用投机：不需要再加 `--spec`，显式的 `--no-spec` 则会否决它。草稿模型的隐藏维度必须与目标一致（12B 目标配它自己的 12B 草稿，而不是 26B-A4B 那个）；草稿 GGUF 不匹配、缺失或不完整会在启动时立即失败。Qwen 3.6、GLM 5.2 与 GLM-5.3 把 NextN 块内嵌在主干 GGUF 里，不需要这个参数——它们用 `--spec`。块级草稿器每步起草一整块 token，主干用一次批量前向验证。每个输出 token 仍然取自主干的某一行——贪心配置下用 argmax，否则用本次运行自己的采样器——因此两种情况下输出流都保持不变。块级起草在所有单序列路径（`--input`、`--multi-turn-jsonl`、`--interactive`）上生效，需要 `--backend cuda` 或 `--backend ggml_cuda`。环境变量：`TS_SPEC_DRAFT_MODEL`、`TS_DSV4_DSPARK`。 |
| `--temperature <f>` | 采样温度（0 = 贪心） |
| `--top-k <N>` | Top-K 过滤（0 = 关闭） |
| `--top-p <f>` | Nucleus 采样阈值（1.0 = 关闭） |
| `--min-p <f>` | 最小概率过滤（0 = 关闭） |
| `--repeat-penalty <f>` | 重复惩罚（1.0 = 无） |
| `--penalty-last-n <N>` | 重复 / presence / frequency 惩罚回看多少个最近的 token。`0` 关闭历史惩罚，`-1` 使用全部历史（默认：64） |
| `--presence-penalty <f>` | 存在惩罚（0 = 关闭） |
| `--frequency-penalty <f>` | 频率惩罚（0 = 关闭） |
| `--seed <N>` | **文本**采样的随机种子（-1 = 非确定性）。图像与视频生成的噪声改由 `--diffusion-seed` 决定。 |
| `--stop <string>` | 停止序列（可重复指定） |
| `--dump-prompt` | 仅渲染 prompt 与分词后退出（不进行推理） |
| `--benchmark` | 运行合成的 prefill / decode 吞吐基准 |
| `--bench-prefill <N>` | 合成 prefill 的 token 长度（默认：32） |
| `--bench-decode <N>` | 合成 decode 的 token 长度（默认：64） |
| `--bench-runs <N>` | 基准运行次数；输出最佳与平均结果（默认：1） |
| `--bench-kvcache` | 运行多轮 KV 缓存复用基准（对比启用缓存与强制重置时的 prefill 延迟） |
| `--bench-kv-turns <N>` | `--bench-kvcache` 使用的对话轮数（默认：4，最多 8） |
| `--bench-chunked` | 运行分块 prefill 微基准（Gemma 4） |
| `--bench-fixed-tokens` | 喂入预先确定的 decode token 流并计时，跳过主机侧贪心采样，便于与 llama-bench 对比。仍会报告一条不计时的贪心正确性链 |
| `--paged-bench` | 跨会话分页 KV 缓存基准：先付一次完整 prefill，再测量第二个同前缀请求能省回多少 |
| `--paged-bench-prompt <N>` | `--paged-bench` 的提示词长度（token，默认：2048） |
| `--paged-bench-trials <N>` | `--paged-bench` 的试验次数（默认：3） |
| `--warmup-runs <N>` | 在对真实文本 / 多模态 prompt 计时前丢弃的前向次数（默认：0） |
| `--test-chunked-prefill` | 运行分块 prefill 正确性检查（对比分块与非分块 logits） |
| `--correct-prefill <N>` | `--test-chunked-prefill` 使用的 prompt 长度 |
| `--correct-decode <N>` | `--test-chunked-prefill` 使用的 decode 长度 |
| `--diffusion-steps <N>` | DiffusionGemma 每个 block 的去噪步数（默认：48）。对 Qwen-Image-Edit 则是 FlowMatch-Euler 步数——省略时自动选择（30，或已加载 Lightning LoRA 的步数）。 |
| `--diffusion-seed <N>` | 扩散路径的噪声种子：DiffusionGemma 的确定性采样器（默认：0）、Qwen-Image-Edit，以及视频生成（Wan、MiniMax-H3）——视频不传时每次运行都会取一个新的随机种子。决定一段视频长什么样的是这个种子，`--seed` 是文本采样种子，对它没有影响。 |
| `--diffusion-blocks <N>` | DiffusionGemma block-autoregressive canvas 数量。`0` 表示根据 `--max-tokens` 与模型 canvas 长度推导。 |
| `--image <path>` | Qwen-Image-Edit 的输入图像（也是多模态聊天的图像输入）。在 `qwen_image` DiT GGUF 上触发图像编辑模式所必需。 |
| `--prompt <text>` | Qwen-Image-Edit 编辑指令（省略时回退到 `--input` 文件内容）。 |
| `--output <path>` | Qwen-Image-Edit 输出 PNG 路径（默认：`edited.png`）。 |
| `--cfg <F>` | Qwen-Image-Edit true-CFG 引导尺度（`<= 1` 关闭负向分支）。省略时自动选择：2.5（Qwen-Image-Edit-2511 的推荐值；4.0 会过度引导并扭曲人脸），加载 Lightning LoRA 时为 1.0。步数与种子复用 `--diffusion-steps` / `--diffusion-seed`。在 MiniMax-H3 上唯一可接受的取值是 `1.0`（也是它的默认值）：该检查点是 CFG 蒸馏的，更高的值会被直接拒绝，而不是照跑然后出劣化结果。`TensorSharp.Server` 根本没有 `--cfg` 参数——但请求体里仍然可以带 `cfg`。 |
| `--qwen-image-vae <path>` | 覆盖解析到的 Qwen-Image VAE 伴随文件（`.gguf` 或 `.safetensors`）。 |
| `--qwen-image-vl <path>` | 覆盖解析到的 Qwen2.5-VL-7B 文本编码器 GGUF。 |
| `--qwen-image-mmproj <path>` | 覆盖解析到的 Qwen2.5-VL mmproj（视觉接地）GGUF。 |
| `--qwen-image-lora <path>` | Qwen-Image-Edit 的 Lightning 蒸馏 LoRA（`.safetensors`）。它以运行期 F32 旁路的形式接在每个目标投影旁（`y = W_quant·x + b + (alpha/rank)·up·(down·x)`），量化基权重原样保留——**不会**被合并进权重。步数从文件名自动推导（例如 4 或 8），并把 CFG 切换为 1.0、时间步 shift 固定为 3，于是默认的 30 步 × 2 次 CFG 前向（60 次 DiT 前向）变成 4–8 次。它需要整模型或融合逐块的 CUDA 前向路径；在没有该旁路的路径上会直接报错而不是输出噪声。环境变量：`TS_QWEN_IMAGE_LORA`。 |
| `--offload-cpu` | 从内存流式读取 DiT 权重，而不是常驻显存：每步更慢，但小显存卡也能做原生约 1 MP 的编辑。默认：自动——只有当目标分辨率与常驻权重放不下时才会自动启用 |
| `--width <px>` / `--height <px>` | Qwen-Image-Edit 与视频生成的输出尺寸。默认 `0` —— 自动（Qwen-Image-Edit：源图尺寸，按 VRAM 钳制；MiniMax-H3：640×384，有条件图时按该面积取图片宽高比，并向上取整到 32 的倍数；Wan：按输入图的宽高比取模型原生面积，TI2V-5B 为 1280×704，其余为 832×480）。 |
| `--video-frames <N>` | 视频帧数，会对齐到模型自己的时间网格（Wan 为 `4k+1`；MiniMax-H3 为 `17k+5` —— 5、22、39、56、73、90…）。默认：33；Wan2.2-TI2V 为 49，MiniMax-H3 为 22。`1` 生成一张静态图（配合 `--output out.png`）。 |
| `--fps <N>` | 保存的 MP4 的播放帧率（默认：16；Wan2.2-TI2V 为 24）。以固定帧率训练的模型（MiniMax-H3，24 fps）会覆盖任何其他取值。 |
| `--flow-shift <F>` | FlowMatch 时间步 shift（默认：模型官方配方 —— Wan 2.2 为 5.0，A14B T2V 为 12.0，Wan 2.1 为 8.0/3.0/5.0，MiniMax-H3 为 12.0）。在带联合音频流的模型上，该 shift 只作用于视频流。 |
| `--sampler <name>` | 采样器：`unipc`（Wan 官方采样器，Wan 默认）或 `euler`。这是 Wan 家族的开关——MiniMax-H3 走自己的 flow-match 调度，不读取它。 |
| `--negative-prompt <text>` | 负向提示词（默认：模型官方的负向提示词）。在 `--cfg 1.0` 下不生效，因为不跑无条件分支——所以对只接受 `--cfg 1.0` 的 MiniMax-H3 完全没有作用。 |
| _（步数蒸馏检查点）_ | 按 DiT 文件名自动识别（`Turbo`、`distill`、`Lightning`、`lightx2v`、`FastWan`、`-dmd`，或显式的 `…-4steps-…` / `…8step…`，N 取 1–16）：管线切换到该步数并关闭引导，把官方 50 步 × CFG 配方的 100 次 DiT 前向变成 4 次。这是整个 TensorSharp 里最大的提速手段——详见下文 **[视频生成（Wan）](#视频生成wan)**。`--diffusion-steps` / `--cfg` 可覆盖它。 |
| `--cfg-cache-stride <N>` | Wan 引导缓存：每 `N` 步只跑一次无条件 CFG 前向，其余步复用缓存的引导方向（默认关闭——每步都跑两次前向）。`2` 约快 1.30×，`3` 约快 1.43×；属于近似，需要严格对齐参考样本时请关闭。在 `--cfg 1.0` 下不起作用，因此对 MiniMax-H3 无效。 |
| `--video-vae <path>` | 覆盖解析到的视频 VAE（`wan_2.1_vae.safetensors` / `Wan2.2_VAE.safetensors`；MiniMax-H3 用 `minimax_h3_video_vae_fp16.safetensors`）。环境变量：`TS_VIDEO_VAE`（同时兼容 `TS_WAN_VAE`）。 |
| `--video-text-encoder <path>` | 覆盖解析到的文本编码器 GGUF（Wan 用 UMT5-XXL，MiniMax-H3 用 Qwen3-VL-32B）。亦可写作 `--video-te`。环境变量：`TS_VIDEO_TEXT_ENCODER`（同时兼容 `TS_WAN_TE`）。 |
| `--video-dit2 <path>` | 双专家模型的第二个扩散专家（Wan 2.2 A14B 的 high/low-noise 搭档）。两者同目录时按文件名自动解析。环境变量：`TS_VIDEO_DIT2`（同时兼容 `TS_WAN_DIT2`）。 |
| `--audio-vae <path>` | 与视频联合生成音轨的模型所用的音频 VAE（`minimax_h3_audio_vae_fp32.safetensors`）。不提供时该类模型仍能出图，只是没有音频。环境变量：`TS_VIDEO_AUDIO_VAE`。 |
| `--video-mode <mode>` | 在有多种条件模式的模型上，指定所传图片的含义。默认：按所传内容自动推断。MiniMax-H3 支持 `t2v`（纯文本）、`i2v`（图片**就是**第一帧，并让它动起来）、`fl2v`（首帧与末帧）、`ref`（图片是用于**全新**场景的身份/外观参考）。`i2v`/`fl2v` 需要 FL2VA 检查点，`ref` 需要 Ref2VA——它们是两个独立文件而不是开关，要错了会直接报错并指出该加载的另一个文件。 |
| `--end-image <file>` | 末帧条件图，适用于支持该能力的模型（MiniMax-H3 的首尾帧模式）。与 `--image` 一起使用时，片段会被引导为从首帧开始、到末帧结束。 |
| `--ref-image <file>` | 参考图，用于参考条件模型（MiniMax-H3 Ref2VA）：主体特征保留下来，而机位、背景和构图由提示词决定。最多可重复传入 9 次；在提示词中按 `<Picture 1>`、`<Picture 2>`… 引用。参考图只会被**缩小**并保持自身宽高比，输出尺寸仍由 `--width`/`--height` 决定。 |
| `--ref-video <path>` | 参考视频片段——可以是视频**文件**，也可以是帧目录；两种都会被重采样到模型自己的 24 fps 与画布。可重复传入；在提示词中按 `<Video 1>`、`<Video 2>`… 引用。 |
| `--ref-video-audio <file>` | 与**相同位置**的 `--ref-video` 配对的音轨：第一个配第一个，依此类推。之所以与 `--ref-video` 分开，是因为容器里的音轨无法通过帧解码器读取；想要一段无声的参考片段就不传它。支持 WAV、MP3 或 Ogg。 |
| `--ref-audio <file>` | 参考音频片段。可重复传入；在提示词中按 `<Audio 1>`、`<Audio 2>`… 引用。会被重采样到音频 VAE 的 32 kHz 立体声，并截断到生成片段的时长。 |
| `--no-audio` | 对与视频联合生成音轨的模型（MiniMax-H3）跳过音频解码，省下音频 VAE 的时间与显存。纯视频模型会忽略该开关。 |
| _（参数改名）_ | 视频生成不再只有 Wan，因此 `--wan-vae`、`--wan-te`、`--wan-dit2` 改名为 `--video-vae`、`--video-text-encoder`、`--video-dit2`。旧写法在命令行、服务端以及配置文件键名中依然全部兼容，已有配置无需改动。 |
| `--test` | 运行内置的分词器、ChatML 模板与 Ollama 对比测试 |
| `--test-templates <dir>` | 对 `<dir>` 下的每个 *.gguf 校验硬编码模板与 GGUF Jinja2 模板的一致性 |
| `--config <path>` | 从 JSON 配置文件读取参数（命令行参数会覆盖它）。支持 `${变量}` 与通过 `{ "path": ..., "urls": [...] }` 自动下载模型。可重复。见[配置文件](#配置文件cli--server)。 |
| `--log-level <lvl>` | 控制台与文件日志级别：`trace`、`debug`、`info`、`warning`、`error`、`critical`、`off` |
| `--log-dir <path>` | JSON-line 文件日志的写入目录（默认：`<binDir>/logs`） |
| `--log-file <0\|1>` | 关闭（`0`）或开启（`1`）文件日志（默认：开启） |
| `--log-console <0\|1>` | 关闭（`0`）或开启（`1`）控制台日志（默认：开启） |

CLI 只会自动识别少数旧式投影器文件名，而当前模型仓库经常使用不同名称。多模态运行请用 `--mmproj` 显式传入已下载文件；`TensorSharp.Server` 从不自动检测投影器。

**JSONL 输入格式：**

每行是一个 JSON 对象，包含 `messages`、可选 `prompt` 和可选采样参数：

```json
{"id": "q1", "messages": [{"role": "user", "content": "What is 2+3?"}], "max_tokens": 50}
{"id": "q2", "messages": [{"role": "user", "content": "Write a haiku."}], "max_tokens": 100, "temperature": 0.8}
```

**交互式 REPL 命令：**

通过 `--interactive` / `-i` 启动后，可使用斜杠命令驱动当前会话。在 REPL 中输入 `/help`（或 `/?`）可查看相同的命令列表。任何不以 `/` 开头的输入都会被视为一轮用户消息。

每轮提示符前的状态行会汇总当前状态——模型、后端、架构、上下文长度、投影器、对话深度，以及为下一轮排队的附件数量（例如 `[turn 3 (2 attachments pending)]> `）。生成过程中按 Ctrl+C 可中断当前回复；在提示符处按 Ctrl+C 可退出。

会话控制：

| 命令 | 说明 |
|---|---|
| `/help`、`/?` | 显示全部交互命令 |
| `/exit`、`/quit` | 退出当前会话 |
| `/reset`、`/new` | 清空对话历史与 KV 缓存 |
| `/history` | 打印对话历史 |
| `/save <文件>` | 将当前对话追加写入 UTF-8 文件 |
| `/system <文本>` | 设置系统提示词（参数为空表示清空），并重置 KV 缓存 |
| `/skills` | 列出已注册的 Agent Skills，并标出本会话中哪些处于启用状态 |
| `/skill <name>` | 为本会话开启或关闭某个技能。与 `/system` 一样会重置对话——技能文本块位于提示词最前端。 |
| `/think on\|off` | 切换思维链/推理模式（仅对支持的模型生效） |
| `/multiline on\|off` | 切换多行输入（在单独一行输入 `.` 结束消息） |

模型与运行时：

| 命令 | 说明 |
|---|---|
| `/info`、`/status` | 显示当前加载的模型、后端、架构、上下文/词表大小、投影器、对话深度与待发送附件 |
| `/model <路径>` | 在当前后端上加载另一个 `.gguf` 模型（会重置会话） |
| `/backend <名称>` | 用其他后端重新加载当前模型：`cpu`、`cuda`、`mlx`、`ggml_cpu`、`ggml_metal`、`ggml_cuda` 或 `ggml_vulkan` |
| `/mmproj <路径>` | 为当前模型加载（或替换）多模态投影器。别名：`/projector` |

采样（实时生效，跨多轮持久化）：

| 命令 | 说明 |
|---|---|
| `/sampling`、`/show` | 打印当前采样配置 |
| `/max <N>` | 单次回复最大 token 数 |
| `/temp <float>` | 采样温度（0 = 贪心） |
| `/topk <int>` | Top-K 过滤（0 = 关闭） |
| `/topp <float>` | Top-P / Nucleus 阈值（1.0 = 关闭） |
| `/minp <float>` | Min-P 过滤（0 = 关闭） |
| `/repeat <float>` | 重复惩罚（1.0 = 关闭） |
| `/presence <float>` | 存在惩罚 |
| `/frequency <float>` | 频率惩罚 |
| `/seed <int>` | 随机种子（-1 = 非确定性） |
| `/stop <文本>` | 追加一条停止序列 |
| `/clearstop` | 清空所有停止序列 |

附件上传（排队到下一轮，发送后自动清空）：

| 命令 | 说明 |
|---|---|
| `/image <路径>`、`/img <路径>` | 附加一张图像（仅对视觉模型有效） |
| `/audio <路径>` | 附加一个音频文件（Gemma 4） |
| `/video <路径>`、`/vid <路径>` | 附加视频，自动抽取关键帧（Gemma 4） |
| `/text <路径>`、`/file <路径>`、`/txt <路径>` | 将 UTF-8 文本/Markdown/CSV/代码文件的前 256 KiB 内联到下一轮提示词中 |
| `/clearattach` | 清空尚未发送的图像/音频/视频/文本附件 |

路径支持单引号或双引号，因此可以直接在 macOS 上从 Finder 拖拽文件到终端。多模态命令需要先加载多模态投影器——在启动时通过 `--mmproj` 指定，或在 REPL 中用 `/mmproj <路径>` 加载。

## Web 应用

在构建完成后，从仓库根目录运行：

```bash
# 通过 --model 指定要托管的模型
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model ./models/model.gguf --backend ggml_metal

# Linux + NVIDIA GPU
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model ./models/model.gguf --backend ggml_cuda

# 让模型生成的代码执行网页检索。联网必须单独显式开启；
# --skills-allow-network 控制的是随技能提供的脚本，并非这些 shell 命令。
./TensorSharp.Server --model ~/work/models/Qwen/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf \
    --backend ggml_metal --port 5001 --skills-allow-exec --code-exec \
    --code-exec-allow-install --code-exec-allow-network --max-tokens 256000

# 多模态模型：同时显式指定投影器
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model ./models/model.gguf --mmproj ./models/mmproj.gguf --backend ggml_cuda

# MiniMax-H3：视频与它的 32 kHz 立体声音轨一起生成。尺寸、步数和帧数都是启动参数，
# 因为 Web UI 自己不发送任何数值；640x384 是文档给出的起点。每次运行都会写出一个
# .mp4 和一个旁挂的 .wav。详见下文“音视频生成（MiniMax-H3）”。
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model ./models/minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_cuda \
    --video-width 640 --video-height 384 --video-steps 20 --video-frames 22

# 同样的托管也可以直接用随仓库提供的配置文件（config/minimax-h3-ref2va.json 换成
# 参考检查点；两者只有去噪器不同，其余三个网络不会重复下载）。
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --config config/minimax-h3-fl2va.json

# Wan 视频生成，仅视频：当 Web UI 或 API 请求未提供自己的 frames / fps 时，
# 默认以 24 fps 生成 121 帧（约五秒）。
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model ./models/Wan2.2-TI2V-5B.gguf --backend ggml_cuda \
    --video-frames 121 --fps 24
# TI2V-5B 原生面积下的 121 帧是 27k 个 DiT token，而自注意力对 token 数是平方级，
# 因此在**基础**检查点上跑完 50 步在笔记本级 GPU 上需要数小时（实测 M5 Pro 约 3 小时
# 30 分）。把 --model 换成步数蒸馏检查点后，同一个请求只需 17 分 30 秒——其他参数
# 一律不变。详见下文“视频生成（Wan）”。
# 服务端会打印每次前向的耗时与滚动 ETA，并每 30 秒发一次心跳，Web UI 两者都会显示；
# 完整成本表见 docs/models/wan_zh-cn.md。

# 配置服务端默认采样参数（仅在请求未自行覆盖时生效）
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model ./models/model.gguf --backend ggml_metal \
    --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.1 \
    --presence-penalty 0.0 --frequency-penalty 0.0 --seed 42 \
    --stop "</s>" --stop "<|endoftext|>"

# 用可复用的 JSON 文件读取以上全部参数（首次运行会自动下载模型）。
# 示例见“配置文件”一节与 config/ 目录。
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --config config/server-basic.json
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --config config/server-basic.json --backend ggml_cpu
```

在浏览器中打开 `http://localhost:5000` —— 根地址即为聊天界面（`GET /health` 是存活检查接口）。Web 界面支持：

- 多轮聊天
- 每个浏览器 Tab 独立的会话：每个 Tab 拥有自己的对话历史；KV block 由推理引擎统一管理
- 通过 `--model` 显式托管单个 GGUF 模型
- 在需要时通过 `--mmproj` 显式托管多模态投影器
- 完整上传文本和 PDF 文档，并可上传图像、视频和音频进行多模态推理（最大 500 MB）
- 思维链/推理模式切换
- 带函数定义的工具调用
- Agent Skills：可在界面上勾选服务端已注册的技能，并实时展示模型作答过程中读取过的每一个技能文件（服务端没有技能时该控件不显示）
- 通过 Server-Sent Events 进行流式 token 生成
- 承载 `diffusion-gemma` GGUF 时展示 DiffusionGemma 去噪预览（每一步替换整条 assistant 消息，最终再发出定稿）
- 向后兼容的队列状态事件（实际并发由推理引擎处理）
- 消息编辑和删除，支持从对话中任意位置重新生成
- 自由滚动：在生成过程中可向上滚动查看历史消息；只要重新滚回底部，新内容会继续自动跟随

使用 `--model` 选择要托管的 GGUF 文件，使用 `--mmproj` 选择要托管的投影器文件。`TensorSharp.Server` 不再扫描 `MODEL_DIR`。

**服务命令行参数：**

不带任何参数运行 `TensorSharp.Server` 会打印完整的参数说明（每个参数的描述、默认值和示例）后退出；`--help` 效果相同。推理必须在启动时传入 `--model`。其他参数可以启动无模型的状态进程，但 `/api/models/load` 不能选择启动时未提供的 GGUF。

| 参数 | 说明 |
|---|---|
| `--model <path>` | 需要托管的 GGUF 文件（推理时必填；如传入了其他参数但未指定该项，服务仍可启动，但 `/api/models/load` 会报告未加载模型） |
| `--mmproj <path>` | 多模态投影器 GGUF（仅给文件名时按模型目录解析；传 `none` 可显式禁用）。需要先指定 `--model`。 |
| `--backend <type>` | 默认计算后端：`cpu`、`cuda`、`mlx`、`ggml_cpu`、`ggml_metal`、`ggml_cuda` 或 `ggml_vulkan` |
| `--tp <N>` | 多卡度 —— 把托管的模型摊到本机几张 GPU 上（默认：`1`）。架构实现了张量并行就走张量并行；Qwen 3.8 Flash Next（`qwen4exp`）与 DeepSeek V4 走按层切分（整层落在单卡 —— 买的是容量，不是速度）。GLM 5.x 不传此参数时按层切分；在 GGML GPU 后端上，传入它则为 GLM-5.2、GLM-5.3 与 GLM-5.3-Flash 一律选择原生本地单进程 TP（在 GLM-5.3 上这只是一个被接受的模式，而不是已验证的配置——各 rank 会复制 cache，因此 `--tp` 会成倍放大 KV 占用）。需要 `--backend cuda`、`ggml_cuda` 或 `ggml_vulkan`。环境变量：`TENSORSHARP_TP_DEGREE`。详见[张量并行与分布式推理](#张量并行与分布式推理)。 |
| `--tp-node-id <N>` | 多节点（分布式）张量并行中本节点的 0 起始编号。服务端只能是节点 `0`（对外提供 HTTP 的 driver）；其余节点请用 `TensorSharp.Cli` 启动。必须与 `--tp-peers` 一起使用。环境变量：`TENSORSHARP_TP_NODE_ID`。 |
| `--tp-peers <list>` | 分布式 TP 集群中所有节点的 `host:port` 列表（逗号分隔，按节点 ID 排序，例如 `192.168.1.10:9500,192.168.1.11:9500`）。必须与 `--tp-node-id` 一起使用。环境变量：`TENSORSHARP_TP_PEERS`。 |
| `--gpu-device <N>` | `ggml_vulkan` 后端使用的 Vulkan 设备索引，用于多 GPU 主机（例如同时装有 Intel 集成显卡和 NVIDIA 独立显卡的机器）。默认使用设备 0；可用 `--list-gpus` 查看索引。也可通过环境变量 `TS_GGML_VULKAN_DEVICE` 设置。 |
| `--list-gpus` | 列出 ggml-vulkan 可见的 Vulkan 设备（索引 + 显卡名称）后退出 |
| `--n-cpu-moe <N>` / `-ncmoe <N>` | 把前 N 层的路由 MoE 权重留在系统内存里、在 CPU 上运行其 FFN（见上文**混合专家 CPU 卸载**）。`all` 表示卸载所有层。默认：所有架构（含 DeepSeek V4 与 GLM 5.x）都是 0；装不下的模型会在加载时被拒绝并告知需要卸载多少层。环境变量：`TS_N_CPU_MOE`。 |
| `--cpu-moe` / `-cmoe` | `--n-cpu-moe all` 的简写。默认：关闭。环境变量：`TS_CPU_MOE`。 |
| `--cpu-moe-threads <N>` | 主机侧专家矩阵乘的工作线程数。默认：在核数多于 8 的主机上取可用 CPU 并行度（`hardware_concurrency`，再受亲和性掩码与 cgroup CPU 配额约束）的一半。服务端还需要另一半来跑 Kestrel、调度器与加速器提交线程；把它设到接近配额会让吞吐直接崩塌而不是缓慢下降（95 CPU 配额下 64 线程 20.7 tok/s，71 线程只剩 8.2）。环境变量：`TS_CPU_MOE_THREADS`。 |
| `--help` | 打印参数说明后退出（不带任何参数启动服务时也会显示） |
| `--max-tokens <N>` | 最大生成 token 数：请求未携带上限时用它填充，请求要求更多时按它截断。对所有端点生效（Web UI、`/api/chat`、`/api/generate`、`/v1/chat/completions`、`/v1/responses`）。默认：`20000`，此默认值只用于填充、不做截断。环境变量：`MAX_TOKENS`。 |
| `--skills-dir <path>` | 扫描 Agent Skills 的目录。可重复；按给出的顺序扫描，最深三层。不传时使用二进制文件旁的 `skills` 目录，不存在则创建，`POST /api/skills` 上传的技能也落在那里。传入的路径不存在会在启动时报错。环境变量：`TS_SKILLS_DIR`（以路径分隔符分隔的列表）。 |
| `--skill <name>` | 让某个技能对所有未自行指定技能的请求生效。可重复。请求体里的 `skills` 数组会覆盖它。 |
| `--list-skills` | 打印技能注册表——名称、描述、文件、告警与加载错误——然后退出。 |
| `--no-skills` | 关闭 Agent Skills：`/v1/skills` 与 `/api/skills` 会报告功能已禁用，请求中的 `skills` 字段不再生效。环境变量：`TS_NO_SKILLS`（非 `0` 即视为开启）。 |
| `--skills-no-discovery` | 不向模型展示未被选中的技能，于是每个请求只看得到它自己点名的技能。按请求粒度可用 `"skills_discovery": false` 达到同样效果。 |
| `--skills-allow-exec` | 允许 `skills_run`。**默认关闭。在共享服务器上，这就是一个远程代码执行入口**——技能是别人上传的内容，而决定要不要运行其中某个脚本的，是一个正在读同一个人写的 Markdown 的模型。脚本只经已知解释器直接运行，不经过 shell；工作目录是当前 Web 会话或私有 OpenAI/Ollama 请求工作区（未开启代码执行时为单次调用 scratch 目录），技能目录只读。默认的 required 沙箱在无法提供安全操作系统隔离时拒绝执行。环境变量：`TS_SKILLS_ALLOW_EXEC`（非 `0` 即视为开启）。 |
| `--skills-max-rounds <n>` | 模型在必须作答前可以取用技能内容——或运行、查看并修正代码——的次数，范围 1-64（默认：`8`；开启 `--code-exec` 时为 `24`；此处显式设定的值按原样使用）。每一轮都是一次完整生成。环境变量：`TS_SKILLS_MAX_ROUNDS`。 |
| `--skills-sandbox <off\|preferred\|required>` | 对技能脚本要求多强的操作系统级隔离。`required`（默认）在没有安全沙箱的宿主机上拒绝运行；`preferred` 表示显式接受较弱隔离；`off` 只保留解释器、环境、时限与输出限制。macOS 使用 `sandbox-exec`；Linux 要求 `bwrap` 0.12.0 或更高版本。Windows Job Object 只能限制进程树，因此 `required` 会拒绝，必须显式选 `preferred` 才接受较弱隔离。环境变量：`TS_SKILLS_SANDBOX`。 |
| `--skills-allow-network` | 允许沙箱内的技能脚本联网。默认禁止。环境变量：`TS_SKILLS_ALLOW_NETWORK`。 |
| `--code-exec` | 向模型提供 `shell`、`read_file`、`edit_file`、`write_file` 与 `apply_patch` 工具。`shell` 执行真正的命令行并返回退出码与合并后的 stdout/stderr；文件工具由宿主完成精确修改。Web UI 每个聊天会话保留一个工作区；每个 OpenAI Chat、OpenAI Responses 与 Ollama HTTP 请求则在其内部工具轮次间使用一个私有工作区，并在响应结束后删除。macOS / Linux 上命令必须在操作系统沙箱内运行；隔离不可用时，除非运维者显式设置 `--code-exec-unconfined`，服务端会拒绝执行。Windows 始终需要这个逃生开关。联网默认另行禁止，只能用 `--code-exec-allow-network` 开启。所有内置工具都由宿主在进程内应答，不会返回给 API 客户端。默认关闭。环境变量：`TS_CODE_EXEC`（非 `0` 即视为开启）。 |
| `--code-exec-allow-install` | 允许模型把 pip / npm 包装进当前 Web 会话或 HTTP 请求工作区；需要 `--code-exec`。这项权限**不会**给模型生成的命令套接字。TensorSharp 读取并校验识别到的安装命令，用宿主构造的参数向量执行只装 wheel / 禁止脚本的安装，再把 `true` 或 `false` 代回命令行。后续生成的代码仍然离线，除非另行传入 `--code-exec-allow-network`。环境变量：`TS_CODE_EXEC_ALLOW_INSTALL`。 |
| `--code-exec-allow-network` | 给予每一条模型生成的命令不受限的宿主 IP 网络访问；生成的代码可以解析 DNS、抓取 URL、跟随重定向、调用远程 API、访问局域网/回环服务并打开 IP 监听套接字。**默认关闭**，且需要 `--code-exec`。macOS 与 Linux 上的写入及主目录读取约束仍保持生效。Linux 还通过 PID 命名空间约束后代进程；macOS 子进程会继承 Seatbelt，普通进程组也会被清理，但主动脱离进程组的子进程可能在请求结束后继续运行，每次工具结果都会明确报告这一限制。macOS 仍拒绝常见的 `/private/tmp/com.apple.launchd*` 路径套接字（但保留系统运行时所需的 Mach lookup 与 DNS 必需的精确 mDNSResponder 路径套接字），Linux 仍隐藏常见的 `/run` 端点，但本地 Unix IPC 并非完整隔离边界：macOS 为兼容性保留共享临时目录内的 Unix IPC，Linux 的宿主网络命名空间可能暴露抽象套接字以及 `/run` 之外的路径名套接字。其他宿主可读文件及 IP 服务因而可能被访问并外传；远程提示词注入与不可信下载也是额外风险。只有不含凭据的宿主 HTTP/SOCKS 代理设置会在此模式下传入。宿主会把不超过 16 MiB 的自定义 CA 包只读取一次，并仅将验证过的公开证书复制到每会话只读快照，因此源路径及其相邻数据不会暴露；认证代理需使用宿主侧无凭据转发器。包名与安装域名允许列表只约束宿主识别并代办的安装；不受限的生成代码仍可直接抓取或执行制品。它与 `--code-exec-allow-install`、`--skills-allow-network` 都相互独立。Windows 仍须传入 `--code-exec-unconfined`。环境变量：`TS_CODE_EXEC_ALLOW_NETWORK`（非 `0` 即视为开启）。 |
| `--code-exec-packages <list>` | 把安装限制在这些包名之内，逗号分隔；其余一律拒绝，并告诉模型允许哪些名字（默认：留空，即不限包名——且无论如何单次安装最多 16 个包）。匹配的是去掉版本后的裸名字，因此模型钉住的版本（`numpy==2.1.0`）仍然匹配名单里的 `numpy`。工具接口变成 shell 时它曾被退休，因为模型自己敲 `pip install` 时，可以用包名白名单看不见的方式写出同一个请求；如今它回来了，因为安装重新由**宿主**执行：宿主是把包名从模型的命令里读出来，而不是去运行那条命令，安装器的参数向量由宿主自己拼出，所以无论请求写成 `pip`、`pip3`、`python -m pip` 还是一个 requirements 文件，这份名单都同样生效。只有配合 `--code-exec-allow-install` 才有意义。启用 `--code-exec-allow-network` 后，该名单不是安全边界：生成代码可绕过宿主安装器，直接抓取或执行制品。 |
| `--code-exec-install-index <url>` | 让宿主代办的安装使用运维者指定的软件包索引；模型自己写出的改源参数仍会被拒绝。索引主机会自动加入安装出口允许列表；任何独立下载主机需列入 `--code-exec-install-domains`。需要 `--code-exec-allow-install`。默认：未设置。环境变量：`TS_CODE_EXEC_INSTALL_INDEX`。 |
| `--code-exec-install-domains <list>` | **宿主代为执行的装包操作**可以到达的主机，逗号分隔，可用精确名称或 `*.suffix` 通配（默认：`pypi.org,files.pythonhosted.org,registry.npmjs.org`；空值关闭钉选）。传入 `--code-exec-allow-network` 后，普通生成命令获得的是不受限主机网络访问，不受此安装域名列表约束。环境变量：`TS_CODE_EXEC_INSTALL_DOMAINS`。 |
| `--code-exec-timeout <seconds>` | 单条命令被杀死前最多可运行的秒数（默认 `120`，而不是旧程序运行器的 30 秒，因为现在装包、跑构建、跑测试用的都是同一个 shell 命令）。单次调用可以用 `timeout_ms` 要求更短或更长，上限 10 分钟；超时的命令会被停掉，但它在此之前打印的一切仍会交给模型——丢掉输出的超时只会让它闭着眼睛把命令再跑一遍。 |
| `--code-exec-temperature <number>` | 代码执行轮次可选的采样温度，范围 `0` 到 `2`；请求或运维者显式设置的采样仍优先，负值清除此覆盖。无论是否设置温度，代码轮次都会去掉内置的聊天重复惩罚。默认：未设置。 |
| `--code-exec-shell <path\|name>` | 宿主自己的选择不合适或找不到时，用哪个 shell 来执行命令。默认：macOS 与 Linux 上是 `bash`，其次 `sh`；Windows 上是 PowerShell 7（`pwsh`），其次 Windows PowerShell——在 Windows 上，PATH 里那个裸的 `bash` 会被**故意**拒绝，因为它是 WSL 启动器，经由它执行会把命令送进一个 Linux 虚拟机，而 job object 在 Windows 这一侧握住的只是那个启动器。要在那里用真正的 bash（Git Bash、MSYS2），就用这个参数指到它。模型会被告知自己正在敲的是哪一种方言，因此这一项也会改变它工具说明里的示例。 |
| `--code-exec-max-output <bytes>` | 一条命令的输出保留并展示给模型的字节数（默认 `32768`）。放不下的部分是从**中间**丢弃的，头尾都保留：构建或测试跑到最后的那一段才是失败所在，而只保留开头的截断丢掉的恰恰是最需要的那一段。 |
| `--code-exec-unconfined` | 即使操作系统无法隔离，也照样运行模型生成的命令。服务端与 CLI 都接受这一显式逃生开关。Windows 必须使用它，因而文件系统与网络访问都不受约束；不要在不受信任用户可以访问的服务端上启用。 |

**shell 是怎么被用的，又能碰到什么。** 模型对文件与代码的一切操作都走这一行
命令——写、运行、grep、读回溯、修正——这也正是开启本功能后
`--skills-max-rounds` 会从 8 升到 24 的原因。Web UI 每个聊天会话保留一个文件工作区；
每个 OpenAI Chat、OpenAI Responses 或 Ollama HTTP 请求则在其内部工具轮次间保留一个
私有工作区，并在响应结束后删除。文件与宿主安装的软件包会在相应寿命内可用，但每次
shell 调用仍是全新的受限进程；不要依赖 virtualenv 激活状态、PATH 修改或常驻 shell
在调用之间延续。联网规则
属于宿主而不属于模型：命令默认离线，`--code-exec-allow-network` /
`TS_CODE_EXEC_ALLOW_NETWORK` 则给予每一条生成命令不受限的主机网络访问，包括
局域网/回环服务与 IP 监听套接字。macOS 与 Linux 上的写入及主目录读取约束仍保持
生效。Linux 还通过 PID 命名空间约束后代进程；macOS 子进程会继承 Seatbelt，普通
进程组也会被清理，但主动脱离进程组的子进程可能在请求结束后继续运行，每次工具结果
都会明确报告这一限制。macOS 仍拒绝常见的 `/private/tmp/com.apple.launchd*` 路径套接字（但保留系统运行时所需的 Mach lookup 与 DNS 必需的精确 mDNSResponder 路径套接字），Linux 仍隐藏常见的 `/run` 端点，但本地
Unix IPC 并非完整隔离边界：macOS 为兼容性保留共享临时目录内的 Unix IPC，Linux
的宿主网络命名空间可能暴露抽象套接字以及 `/run` 之外的路径名套接字。其他宿主
可读数据可能被外传。`--skills-allow-network` 只控制
`skills_run`，与它相互独立；装包权限同样不会打开 shell 网络。安装仍由宿主亲自
执行——宿主从每一条被识别出来的安装命令里读出工具名与包名，用自己拼出的参数
向量去调用安装器，再把
那条命令在命令行里替换掉——装成功换成 `true`，装失败换成 `false`——于是模型写在
它周围的那些操作符仍然分毫不差地是原来的意思：`&&` 后面的东西不会运行，而 `||`
后面的后备方案会运行。无论哪种情况，失败原因都会被报回来。把套接字交给安装命令
是先前的做法，它有两个无法同时堵上的窟窿：
命令行出自模型之手，于是 `--index-url` 决定了用哪个软件源；而套接字属于整行
命令，于是与安装同处一行的任何东西都共享它能到达的范围——在沙箱无法把出口钉死
在代理上的宿主机上，那个范围就是整个互联网。已退休的 `--code-exec-languages`
会在启动时被按名字拒绝，并且没有替代品可指，因为 shell 能够到达 PATH 上的每一
个解释器；于是从旧接口沿用下来的部署脚本得到的是这条报错，而不是一个被悄悄丢掉
的设置。

| 参数 | 说明 |
|---|---|
| `--video-width <px>` | Web UI 或 API 请求未提供 `width` 时，视频生成使用的默认输出宽度；别名 `--width`。这是**服务端最主要的质量杠杆**，因为 Web UI 自己不发送尺寸——不设置的话每段视频都会用模型默认值生成。MiniMax-H3 的推荐起点是 `640`（配合 `--video-height 384`）。会向上取整到模型的网格。 |
| `--video-height <px>` | 请求未提供 `height` 时的默认输出高度；别名 `--height`。只给出宽高之一时，MiniMax-H3 会按条件图片的宽高比推出另一个，这样 4:3 的照片就不会被拉成 16:9。 |
| `--video-steps <N>` | 请求未提供 `steps` 时的默认去噪步数——分辨率之后最重要的质量/时间取舍。MiniMax-H3 自身默认 `20`；`4`–`8` 是快速工作点，`16`–`24` 明显更干净，超过 30 提升有限。 |
| `--video-mode <mode>` | 请求未提供 `videoMode` 时的默认条件模式：`t2v`（纯文本）、`i2v`（图片是第一帧并让它动起来）、`fl2v`（同时钉住首帧**与**末帧）、`ref`（图片是用于全新场景的身份/外观参考）。不设置时每个请求按其自身携带的输入推断，通常这就是你想要的；只提供一种模式的部署才需要固定它。 |
| `--video-frames <N>` | Web UI 或 API 请求未提供 `frames` 时，视频生成使用的默认输出帧数。会对齐到模型自己的时间网格——Wan 为 `4k+1`，其中以 24 fps 生成 `121` 帧约为五秒；MiniMax-H3 为 `17k+5`。未指定此参数时沿用模型回退值：通常为 `33`，Wan 2.2 TI2V-5B 为 `49`，MiniMax-H3 为 `22`。请求中显式提供的 `frames` 会覆盖此默认值。 |
| `--fps <N>` | Web UI 或 API 请求未提供 `fps` 时，生成的 MP4 使用的默认播放帧率。未指定此参数时沿用模型回退值：通常为 `16` fps，Wan 2.2 TI2V-5B 为 `24` fps。请求中显式提供的 `fps` 会覆盖此默认值；仅改变 FPS 会改变播放速度，而不会改变生成的帧数。以固定帧率训练的模型（MiniMax-H3，24 fps）会覆盖任何其他取值。 |
| `--video-vae <path>` | 覆盖解析到的视频 VAE（`wan_2.1_vae.safetensors` / `Wan2.2_VAE.safetensors`；MiniMax-H3 用 `minimax_h3_video_vae_fp16.safetensors`）。默认：在 DiT 同目录扫描，包含 `VAE/` 子目录。环境变量：`TS_VIDEO_VAE`；`--wan-vae` 仍然兼容。 |
| `--video-text-encoder <path>` | 覆盖解析到的文本编码器 GGUF（Wan 用 UMT5-XXL，MiniMax-H3 用 Qwen3-VL-32B）。亦可写作 `--video-te`。环境变量：`TS_VIDEO_TEXT_ENCODER`；`--wan-te` 仍然兼容。 |
| `--video-dit2 <path>` | 双专家模型的第二个扩散专家（Wan 2.2 A14B 中与 `--model` 配对的 high/low-noise 搭档）。两者同目录时按文件名自动解析。环境变量：`TS_VIDEO_DIT2`；`--wan-dit2` 仍然兼容。 |
| `--audio-vae <path>` | 与视频联合生成音轨的模型所用的音频 VAE（`minimax_h3_audio_vae_fp32.safetensors`）。不提供时该类模型仍能出图，只是没有音频。环境变量：`TS_VIDEO_AUDIO_VAE`。 |
| `--temperature <f>` | 采样温度（`0` = 贪心） |
| `--top-k <N>` | Top-K 过滤（`0` = 关闭） |
| `--top-p <f>` | Nucleus 采样阈值（`1.0` = 关闭） |
| `--min-p <f>` | min-p 过滤（`0` = 关闭） |
| `--repeat-penalty <f>` | 重复惩罚（`1.0` = 无） |
| `--penalty-last-n <N>` | 重复 / presence / frequency 惩罚回看的历史长度。`0` 关闭，`-1` 使用全部历史（默认：64） |
| `--presence-penalty <f>` | 存在惩罚（`0` = 关闭） |
| `--frequency-penalty <f>` | 频率惩罚（`0` = 关闭） |
| `--seed <N>` | 随机种子（`-1` = 非确定性） |
| `--stop <string>` | 停止序列（可重复指定）。在默认的 `--sampling-precedence config` 下，请求体里的 `stop`/`stop_sequences` 会与这里的列表**合并**；在 `request` 下则完全替换。 |
| `--sampling-precedence <config\|request>` | 当请求同时携带了你在上面配置过的采样参数时，以谁为准。`config`（默认）保留你的配置值 —— VS Code Copilot Chat 等客户端会把 `temperature`/`top_p` 硬编码进每一次请求，否则就会静默覆盖你的配置；你**没有**配置过的参数仍然取请求中的值。`request` 恢复“客户端优先”。环境变量：`TENSORSHARP_SAMPLING_PRECEDENCE`。 |
| `--kv-cache-dtype <type>` | 托管模型的 KV 缓存精度：`f32`、`f16`、`q8_0` 或 `q4_0`（量化缓存以微小数值漂移换取内存节省；各档位的取舍见上文 CLI 参数表）。默认：自动 —— 由后端 / 模型决定。环境变量：`KV_CACHE_DTYPE`。 |
| `--continuous-batching` / `--no-continuous-batching` | 启用（默认）或关闭迭代级分页批处理。启用时服务会在批内动态加入 / 抢占序列，并在实现了 `IBatchedPagedModel` 的模型上将多个序列打包到一次前向中执行。`--no-continuous-batching` 会让所有模型回退到按序列 KV 交换。别名：`--paged-batching` / `--no-paged-batching`。 |
| `--no-webui` | 不提供内置 Web UI；`GET /` 改为返回纯文本存活探测。所有 HTTP API 端点（含 `/uploads`）照常可用。环境变量：`TS_NO_WEBUI` |
| `--no-prefix-cache` | 不在对外服务之前准备所有会话共享的那段提示词，也不在两次启动之间保留它。默认情况下服务端会在启动时把这段提示词前向一次并保存结果，于是一个进程的第一条消息与其他消息代价相同（在一个 agent 配置上实测 21.8s → 0.7s） |
| `--prefill-chunk-size <N>` | 混合 prefill+decode 步中每个请求的 prefill 上限，使活跃输出流更频繁地轮到 GPU（默认：`256`）；仅有 prefill 时仍会公平用满设备 token 预算。环境变量：`TS_SCHED_PREFILL_CHUNK`。 |
| `--spec` / `--no-spec` | 启用投机解码（默认关闭）。`--spec` 是内嵌在主干检查点里的草稿器（Qwen 3.6、GLM 5.2 与 GLM-5.3 的 NextN 块）的显式开关——因为加载它们要把额外的权重调入显存；以独立 GGUF 发布的草稿器只需 `--draft-model` 即可启用，显式的 `--no-spec` 则对两者都是否决。仅对单序列（无并发）请求生效：草稿头每步最多提议 `--spec-draft` 个 token，主干网络用一次批量前向完成验证；起草与验证均由该请求自己的采样器（含惩罚项）驱动，输出与标准 decode 一致。仅在有收益处自动启用：Qwen 3.6 的内嵌 NextN 块在所有后端上都被认为有收益，而 Gemma 4 的独立草稿头只在各 ggml 后端与 Direct `cuda` 后端上启用；CPU / GGML CPU / MLX 走标准 decode。环境变量：`TS_SPEC`（旧写法 `TS_MTP_SPEC`）。 |
| `--spec-type <name>` | 投机算法：`auto`（默认）/ `draft-head` / `block` / `ngram`。`ngram` 不需要任何训练权重，对所有模型都能用——它在上文里找最近几个 token 曾经出现过的位置，把当时紧随其后的内容拿来当草稿，因此凡是答案大量引用输入的场景都很强。环境变量：`TS_SPEC_TYPE`。 |
| `--spec-draft <N>` | 每个投机步最多起草的 token 数（默认 `8`；块级草稿器会把它夹到自己训练时的块大小以内，在那里默认也是该块大小）。在 GGML 后端上对 Qwen 3.5 显式传入的值请保持 7 或更小——九行验证批是一个已知的正确性缺陷。环境变量：`TS_SPEC_DRAFT`（或 `TS_MTP_DRAFT`）。 |
| `--spec-pmin <f>` | 草稿置信度门限，取值 `[0, 1]`；遇到第一个低于该值的 token 即停止起草，`0` 表示从不设门限。默认值按算法选择：逐 token 草稿头为 `0.15`（其 top-10 logits 上的 top-1 概率），块级草稿器为 `0.35`——后者的门限是**累积**前缀概率，因此同一个数字要严格得多，n-gram 为 `0`。环境变量：`TS_SPEC_PMIN`（或 `TS_MTP_PMIN`）。 |
| `--draft-model <path>` | 投机解码草稿模型，适用于所有以独立文件发布的草稿器：DeepSeek V4 的 DSpark 支持 GGUF（见 [DeepSeek V4](docs/models/deepseek4_zh-cn.md#dspark-投机解码)）、Muse-Glimmer 的 DFlash 与 Qwen 3.8 的 DFlash2 块级草稿器（见 [Muse-Glimmer](docs/models/muse-glimmer_zh-cn.md)，环境变量 `TS_MUSE_GLIMMER_DFLASH`），以及 Gemma 4 的 `gemma4-assistant` 逐 token 草稿头。文件自己的 `general.architecture` 决定它如何加载——操作者从不需要挑选机制；在这里给出文件本身就会启用投机，不需要再加 `--spec`，显式的 `--no-spec` 则会否决它。草稿的隐藏维度必须与目标一致（例如 12B 目标配 12B 草稿，而非 26B-A4B 草稿）；草稿不匹配或不完整会在启动时立即失败并给出修复提示。Qwen 3.6、GLM 5.2 与 GLM-5.3 将 NextN 块内嵌在主干 GGUF 中，不需要这个参数——它们用 `--spec`。块级草稿器每步起草一整块 token，主干用一次批量前向验证，因此贪心输出保持不变；在 `cuda` 与 `ggml_cuda` 后端上对单序列请求生效。服务端的每一行验证都用该请求自己的采样器，因此可与任意采样设置组合。环境变量：`TS_SPEC_DRAFT_MODEL`（旧写法 `TS_MTP_DRAFT_MODEL`）、`TS_DSV4_DSPARK`。 |
| `--paged-kv` / `--no-paged-kv` | 已移除的按会话分页 KV 管理器的兼容参数。当前服务端 KV 状态由引擎持有；请使用连续批处理 / `TS_SCHED_*` 开关调节引擎。别名：`--paged-kv-cache` / `--no-paged-kv-cache`。 |
| `--paged-kv-block-size <N>` | 旧的独立分页 KV 块大小。当前引擎使用 `TS_SCHED_BLOCK_SIZE`。 |
| `--paged-kv-ram-mb <N>` | 旧的独立分页 KV RAM 层上限。 |
| `--paged-kv-ssd-dir <dir>` | 旧的独立分页 KV SSD 冷层目录。 |
| `--paged-kv-ssd-mb <N>` | 旧的独立分页 KV SSD 上限。 |
| `--paged-kv-quant-bits <0\|4\|8>` | 服务端接受的旧式独立分页 KV 块量化（`4`/`8` = 对称）。运行时环境变量还接受仿射 min+scale 的 `2`，CLI 则接受 `0\|2\|4\|8`。 |
| `--redis-url <url>` | 同时启用共享 KV 缓存层与 Responses API 存储的 Redis 连接串（例如 `localhost:6379`）。它会同时设置 `TS_KV_CACHE_REDIS_URL` 与 `TS_RESPONSES_STORE_REDIS_URL`。 |
| `--paged-kv-redis-url <url>` | 仅用于共享 KV 缓存层的 Redis 连接串（例如 `localhost:6379`）。环境变量：`TS_KV_CACHE_REDIS_URL`。 |
| `--paged-kv-redis-ttl <min>` | Redis KV 缓存条目的 TTL（分钟）；`0` 表示不过期（默认 `1440`，即 24 小时）。环境变量：`TS_KV_CACHE_REDIS_TTL_MINUTES`。 |

请求 JSON 中的字段（如 `temperature`、`top_p`、`top_k`、`min_p`、
`repeat_penalty`、`presence_penalty`、`frequency_penalty`、`seed`、
`stop`/`stop_sequences`）会填充所有你**没有**在上面配置过的参数。对于你
**已经**配置过的参数，默认的 `--sampling-precedence config` 保留你的取值
并忽略请求中的值 —— 很多聊天客户端会把 `temperature`/`top_p` 硬编码进每一
次调用，终端用户无从修改，因此只有服务端未配置的参数才应由请求决定。若需
要相反的行为（客户端始终优先，也就是本参数出现之前的固定行为），请使用
`--sampling-precedence request`。

**运行时环境变量：**

| 变量 | 说明 |
|---|---|
| `BACKEND` | 未传 `--backend` 时使用的默认计算后端（`cpu`、`cuda`、`mlx`、`ggml_cpu`、`ggml_metal`、`ggml_cuda` 或 `ggml_vulkan`；默认：macOS 为 `ggml_metal`，其他平台为 `ggml_cpu`） |
| `MAX_TOKENS` | 未传 `--max-tokens` 时的最大生成长度：请求未携带上限时用它填充，请求要求更多时按它截断（默认：`20000`，该默认值只填充、不截断） |
| `MAX_CONTEXT` | 要分配的上下文窗口，覆盖 GGUF 自报的长度。设了它就是**硬上限**：缓存加一整个 `n_ubatch` 的计算图装得下就照办，装不下就带着具体数字拒绝加载。不设时，自报长度只是**上界**——权重加载完之后运行时会去问各设备实际还剩多少显存，按能装下的大小定上下文，并把选中的值打印出来。GLM-5.2 自报 1,048,576 token（约 93 GiB 的 KV）；在 3x RTX PRO 6000 上按层切分选到 342,272，`--tp 3` 选到 91,136，`--n-cpu-moe 30` 选到 646,400 |
| `VIDEO_SAMPLE_FPS` | 输入视频作为多模态提示词时每秒抽取的帧数；基于时间的抽帧（默认：`1`）。它与视频生成的输出参数 `--fps` 无关 |
| `VIDEO_MAX_FRAMES` | 输入视频作为多模态提示词时抽取帧数的可选上限（超出时均匀降采样）；未设置或为 `0` 表示不限制（默认：不限制）。它与视频生成的输出参数 `--video-frames` 无关 |
| `PORT` / `HOST` | 未传 `--port` / `--host` 时的监听端口与绑定网卡（默认：`5000`、`0.0.0.0`） |
| `ASPNETCORE_URLS` | 当 `--port`、`--host`、`--urls`、`PORT`、`HOST` 均未设置时使用的完整监听 URL |
| `TENSORSHARP_TEMPERATURE` | 未传 `--temperature` 时的采样温度。它同样算作“运维方已配置”，因此在默认的 `--sampling-precedence config` 下也优先于请求体 |
| `TENSORSHARP_TOP_K` | 未传 `--top-k` 时的 Top-K（优先级规则同 `TENSORSHARP_TEMPERATURE`） |
| `TENSORSHARP_TOP_P` | 未传 `--top-p` 时的 Top-P（优先级规则同上） |
| `TENSORSHARP_MIN_P` | 未传 `--min-p` 时的 min-P（优先级规则同上） |
| `TENSORSHARP_REPEAT_PENALTY` | 未传 `--repeat-penalty` 时的重复惩罚（优先级规则同上） |
| `TENSORSHARP_PRESENCE_PENALTY` | 未传 `--presence-penalty` 时的存在惩罚（优先级规则同上） |
| `TENSORSHARP_FREQUENCY_PENALTY` | 未传 `--frequency-penalty` 时的频率惩罚（优先级规则同上） |
| `TENSORSHARP_SEED` | 未传 `--seed` 时的随机种子（优先级规则同上） |
| `TENSORSHARP_SAMPLING_PRECEDENCE` | `config`（默认）或 `request`：服务端配置的采样参数是否优先于客户端发来的值。`--sampling-precedence` 可覆盖它 |
| `TENSORSHARP_LOG_LEVEL` | 控制台与文件日志的最低输出级别：`Trace`、`Debug`、`Information`、`Warning`、`Error`、`Critical`（默认：`Information`）。`TensorSharp.Cli` 同样识别该变量。 |
| `TENSORSHARP_LOG_DIR` | JSON-line 文件日志的写入目录（默认：`<binDir>/logs`）。`TensorSharp.Cli` 同样识别该变量。 |
| `TENSORSHARP_LOG_FILE` | 设为 `0` 可关闭文件日志，仅保留控制台输出（默认：开启）。`TensorSharp.Cli` 同样识别该变量。 |
| `TENSORSHARP_TP_DEGREE` | 多卡度 —— 把模型摊到本机多少张 GPU 上（默认：`1`）。当未传 `--tp` 参数时作为 `ModelBase.Create` 的兜底来源；`TensorSharp.Cli` 与 `TensorSharp.Server` 都提供了 `--tp <N>` 参数。需要 `--backend cuda`、`ggml_cuda` 或 `ggml_vulkan`。在改走按层切分而非张量并行的架构（`qwen4exp`、DeepSeek V4）上，它只是个设备数，而不是切分度。 |
| `TENSORSHARP_TP_DEVICES` | 各 rank 使用的 GPU 序号（逗号分隔，例如 `0,2`；默认 `0..tp-1`）。用于 GGML 后端上的 TP。 |
| `TS_Q4E_LAYER_SPLIT` | Qwen 3.8 Flash Next（`qwen4exp`）多卡按层切分时每张 GPU 分到的层数（逗号分隔，例如 `20,28`），用于取代自动的显存均衡。给出无法满足的值时会直接抛错，而不是静默忽略。 |
| `TENSORSHARP_TP_NODE_ID` | 多节点分布式张量并行中本节点的 0 起始编号。必须与 `TENSORSHARP_TP_PEERS` 一起设置。 |
| `TENSORSHARP_TP_PEERS` | 分布式 TP 集群中所有节点的 `host:port` 列表（逗号分隔，例如 `192.168.1.10:9500,192.168.1.11:9500`）。必须与 `TENSORSHARP_TP_NODE_ID` 一起设置。 |
| `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS` | 各节点向 peer 发起连接的重试窗口（默认：`120` 秒）。若节点由人工或较慢的编排系统间隔较久启动，可调大该值。 |
| `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS` | 从 peer 阻塞读取的单次接收超时（默认：`300` 秒）。有了它，卡住的 peer 会让集合通信直接失败，而不是挂在操作系统的 TCP keepalive（往往 2 小时以上）上。 |
| `TENSORSHARP_TP_DISABLE_P2P` | 设为 `1` 后，所有跨 GPU 传输都经主机内存中转，不使用 CUDA 点对点 DMA。速度更慢，但可复现无 P2P 硬件（A16 vGPU 配置、部分消费级显卡）的代码路径，便于定位 P2P 相关缺陷。 |
| `TENSORSHARP_TP_HOST_ALLREDUCE` | 设为 `1` 后，本地 AllReduce 走主机内存（设备→主机、求和、主机→设备），而非设备到设备路径。这是与已知可靠的多节点归约完全一致的诊断兜底路径。 |
| `TS_KV_CACHE_REDIS_URL` | 共享 KV 缓存层的 Redis 连接串（例如 `localhost:6379`）。设置后，KV 缓存块会持久化到 Redis 以便跨会话复用。CLI：`--redis-url` 或 `--paged-kv-redis-url`。 |
| `TS_KV_CACHE_REDIS_TTL_MINUTES` | Redis KV 缓存条目的 TTL（分钟）；`0` = 不过期（默认：`1440`）。CLI：`--paged-kv-redis-ttl`。 |
| `TS_RESPONSES_STORE_REDIS_URL` | OpenAI Responses API 存储的 Redis 连接串。设置后由 `RedisResponsesStore` 取代内存存储。CLI：`--redis-url`。 |
| `DIFFUSION_STEPS` | 服务端 DiffusionGemma 每个 block 的去噪步数（默认：`48`；CLI 对应 `--diffusion-steps`） |
| `DIFFUSION_MAX_BATCH` | Web UI 扩散调度器可批处理的最大并发 DiffusionGemma 请求数（默认：`2`） |

**分页 KV 缓存 & 连续批处理可调参数（进程 / 模型启动时读取）**

下述变量也可以通过 `--paged-kv*` / `--continuous-batching` CLI 参数设置（它们会被翻译为对应的环境变量）：

| 变量 | 说明 |
|---|---|
| `TS_KV_PAGED_CACHE` | 旧的独立 `PagedKvCacheManager` 兼容开关；当前 `TensorSharp.Server` 的请求 KV 状态由引擎持有。CLI 快捷方式是 `--paged-kv` / `--no-paged-kv`。 |
| `TS_KV_BLOCK_SIZE` | 旧的独立分页 KV 块大小。当前引擎使用 `TS_SCHED_BLOCK_SIZE`。 |
| `TS_KV_CACHE_MAX_RAM_MB` | 旧的独立分页 KV RAM 层上限。 |
| `TS_KV_CACHE_SSD_DIR` | 旧的独立分页 KV SSD 冷层目录。 |
| `TS_KV_CACHE_MAX_SSD_MB` | 旧的独立分页 KV SSD 上限。 |
| `TS_KV_PAGED_QUANT_BITS` | 旧的独立分页 KV 块量化位数（`0` = 透传，`2` = 仿射，`4`，或 `8`）。 |
| `TS_SCHED_DISABLE_BATCHED` | `1` 会即使模型实现了 `IBatchedPagedModel`，也强制回退到按序列 KV 交换。CLI 快捷方式是 `--no-continuous-batching`。 |
| `TS_SCHED_MAX_BATCHED_TOKENS` | 调度器每步 token 预算（默认：`4096`）。 |
| `TS_SCHED_MAX_RUNNING_SEQS` | 同时在执行的最大序列数（默认：`16`）。 |
| `TS_SCHED_PREFILL_CHUNK` | 混合 prefill+decode 步中每请求的 prefill 上限（默认：`256`）；仅 prefill 的步骤会公平用满批 token 预算。 |
| `TS_SCHED_SOLO_PREFILL_CHUNK` | SOLO（无争用）prompt 全新部分（start_pos = 0）的 prefill 分块大小——单个无争用请求会以大分块走融合 prefill 路径（默认：`8192`）。 |
| `TS_SCHED_NUM_BLOCKS` | 引擎块池的物理块数（默认：`256`）。 |
| `TS_SCHED_BLOCK_SIZE` | 引擎侧每块的 token 数（默认：`256`）。 |
| `TS_SCHED_PREFIX_CACHE` | `0` 关闭跨请求的块级哈希前缀共享。 |
| `TS_SCHED_STOP_REPETITION` | `0` 允许陷入重复循环的生成继续跑到 token 上限，而不是被提前结束。 |
| `TS_SCHED_DECODE_QUANTUM` | 在允许切换序列前的 token 数（默认与 block size 相同）。 |
| `TS_RETAINED_FUSED_CACHE` | `1`（默认）在请求结束后保留其融合 holder，使前缀完全一致的续写不必重新 prefill；仅对声明支持的模型有效（Gemma 4 的 K/V；Qwen 3.5/3.6 的注意力 K/V 加 GatedDeltaNet 递归状态）。`0` 关闭（用于限制显存或做 A/B）。 |
| `TS_RETAINED_FUSED_CACHE_MAX` | 保留的融合 holder 的 LRU 预算（默认 `4`）；每个都钉住一份完整的按请求续写状态。 |
| `TS_PREFIX_CHECKPOINTS` | `1`（默认）在所有会话共享的那段提示词末尾——系统提示词、工具、技能——对模型状态做 checkpoint，并让每个**新**会话从它的副本开始，因此新会话只需重新 prefill 自己的那条消息。适用于 GGML 后端上的 Gemma 4 与 Qwen 3.5/3.6。`0` 关闭。 |
| `TS_PREFIX_CHECKPOINTS_MAX` | 同时保留 checkpoint 的不同共享前缀数量，按 LRU 淘汰（默认 `2`）。每个都持有该前缀的一份 K/V，Qwen 上还包括递归状态。 |
| `TS_KV_INITIAL_TOKENS` | 缓存创建时（加载时的主缓存，以及每个按请求的 holder）在任何请求声明预算之前先分配多少 token 的 K/V。`0`（默认）沿用引擎策略：显式设置 `MAX_CONTEXT` 时取整个窗口，否则取后端默认值。缓存仍会按需增长，因此内存受限的设备应把它设小——每个保留的 holder 都要按这个大小付费，主机副本与设备镜像各一份。 |
| `TS_KV_GENERATION_RESERVE_MAX` | 限制单个请求预先保留的 K/V 中属于生成的那部分（提示词 + `max_new_tokens`）。不设它时，回复长度上限达到或超过上下文窗口，就会让每个请求预留整个窗口。超过上限后缓存按需增长。`0`（默认）表示不限制。 |
| `TS_KV_HOLDER_POOL_MAX` | 释放后的按请求 holder 最多可停放多少个以备复用，而不是直接释放（默认 `64`）。每个停放中的 holder 都占着它完整的 K/V 分配。 |
| `TS_QWEN35_BATCHED` | 设为 `0` 强制 Qwen 3.5/3.6 走旧的按序列 KV-swap 路径（默认走批处理 / 分页）。`--no-continuous-batching` 也会隐式关闭。 |
| `TS_QWEN35_BATCHED_GDN_NATIVE` | 在 Qwen 3.5/3.6 批处理路径中使用原生批处理 GatedDeltaNet 内核。 |
| `TS_GEMMA4_BATCHED` | 设为 `0` 可强制 Gemma 4 走旧的单序列 KV 交换路径（默认走批处理 / 分页）。 |
| `TS_GPTOSS_BATCHED` | 设为 `0` 强制 GPT OSS 走旧的按序列 KV-swap 路径（默认走批处理 / 分页）。 |
| `TS_GPTOSS_PAGED_ATTN_MANAGED` | 在 GPT OSS 批处理路径中使用托管 (C#) 的带 sinks 分页注意力内核。 |
| `TS_NEMOTRON_BATCHED` | 设为 `0` 强制 Nemotron-H 走旧的按序列 KV-swap 路径（默认走批处理 / 分页）。 |
| `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE` | 在 Nemotron-H 批处理路径中使用原生 Mamba2 批处理步骤内核。 |
| `TS_PAGED_ATTN_KERNEL` | `Mistral3Model.BatchedForward` 选择的分页注意力派发内核：`native`（默认）、`tensor`（基于 C# Tensor）或 `managed`（纯 C# 标量）。 |
| `TS_MLX_PIPELINED_DECODE` | 默认 `1`，当请求为贪心采样、没有 stop 序列且模型支持 device-side argmax / 下一 token embedding 查找时，在 MLX 后端启用流水化贪心 decode。设为 `0` 可关闭。仅 CLI。 |
| `TS_MLX_MLOCK_GGUF` | 默认 `1`，通过 `mlock(2)` 把 GGUF mmap 区域钉在物理内存，避免前向之间被换出。设为 `0` 关闭（适用于进程 `memlock` rlimit 太低、或希望让 OS 自行管理分页的情况）。仅 MLX 后端。 |
| `TS_MLX_FUSED_KV_WRITE` | 默认 `1`，使用单次多维 `slice_update` 写入每个 token 的 KV block。设为 `0` 回退到按 head 的循环（A/B 测试 / 隔离回归用）。 |
| `TS_MLX_BATCHED_MOE_DECODE` | 默认 `1`，将 Qwen 3.5/3.6 MoE 解码时每专家的 K 次 dispatch 合并为每种（gate / up / down）一次批处理 dispatch。在显存紧张的机器上可设为 `0` 关闭（可节省堆叠权重 slab 带来的近一倍权重显存占用）。 |
| `TS_MLX_MOE_FUSED_GATE_UP_SILU` | 默认 `1`，把批处理 MoE 解码的 gate matmul + up matmul + SiLUMul 融合到一个 Metal kernel。设为 `0` 用于和旧的 3-dispatch 路径做 A/B 对比。 |
| `TS_MLX_DEVICE_ROUTER` | 默认 `1`，让 MoE router 的 top-K + softmax 留在 device 上，避免每个 MoE 层一次主机同步（在 Qwen3.6-35B-A3B 上约能节省每 token ~60 次同步）。设为 `0` 可关闭；不满足前置条件时会自动回退到 host routing。 |
| `TS_MLX_MEMORY_LIMIT_MB` / `TS_MLX_CACHE_LIMIT_MB` / `TS_MLX_WIRED_LIMIT_MB` | 覆盖 MLX 分配器硬上限 / 空闲缓冲池上限 / wired 缓冲上限（兆字节）。默认值会根据宿主机统一内存大小派生。 |
| `TS_MLX_EVAL_EVERY_N_LAYERS` / `TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS` | 解码时定期触发 `mlx_async_eval` 的层间隔，用于让 GPU 计算和宿主端排队重叠。Gemma 4 通过 `TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS` 默认每 4 层一次；Qwen 3.5 与 Nemotron-H 通过 `TS_MLX_EVAL_EVERY_N_LAYERS` 默认每 16 层一次。支持处可设为 `0` 关闭。 |
| `TENSORSHARP_MLX_LIBRARY` / `TENSORSHARP_MLX_LIBRARY_DIR` | 覆盖 `--backend mlx` 时 `libmlxc` 的搜索路径。 |

**MTP / 投机解码调优变量**

这些变量控制可选的投机解码路径（见 [投机解码](FEATURES_zh-cn.md#投机解码)与[设计文档](docs/speculative_decoding.md)）。`TS_SPEC_*` 为通用开关（也可由 `--spec*` CLI 参数设置）；旧写法 `TS_MTP_*` 同样被接受，并且会被一起写入，因为 glm-dsa 的**原生**加载器是在加载时从 C++ 里读 `TS_MTP_SPEC` / `TS_MTP_DRAFT` 的。`TS_GMTP_*` 为 Gemma 4 草稿路径 A/B 开关。

| 变量 | 说明 |
|---|---|
| `TS_SPEC` *（旧写法 `TS_MTP_SPEC`）* | `1` 为单序列启用投机解码（默认 `0`）。CLI：`--spec` / `--no-spec`。 |
| `TS_SPEC_TYPE` | 投机算法：`auto`（默认）/ `draft-head` / `block` / `ngram`。CLI：`--spec-type`。 |
| `TS_SPEC_DRAFT` *（旧写法 `TS_MTP_DRAFT`）* | 每个投机步最多起草的 token 数（默认 `8`）。CLI：`--spec-draft`。 |
| `TS_SPEC_PMIN` *（旧写法 `TS_MTP_PMIN`）* | 草稿置信度门限，取值 `[0, 1]`，`0` 表示从不设门限（默认按算法：逐 token 草稿头 `0.15`，块级 `0.35`，n-gram `0`）。CLI：`--spec-pmin`。 |
| `TS_SPEC_DRAFT_MODEL` *（旧写法 `TS_MTP_DRAFT_MODEL`）* | Gemma 4 独立 `gemma4-assistant` 草稿 GGUF 路径。CLI：`--draft-model`。Qwen 3.6（内嵌 NextN）忽略此项。 |
| `TS_GMTP_NO_FUSED` | `1` 关闭 Gemma 4 融合多 token 验证 / 草稿步 GGML 内核，回退到逐算子路径（ggml 后端上的 A/B 测试）。 |
| `TS_GMTP_NO_FAST_ROLLBACK` | `1` 恢复保留前缀的回滚路径，而非部分接受时使用的稠密精确匹配快速回滚。 |
| `TS_GMTP_BATCHED_TRUNK` | `1` 让 Gemma 4 验证主干走批量分页路径；默认对单序列投机使用更快的线性主干。 |

**DiffusionGemma 专属调优变量**

| 变量 | 说明 |
|---|---|
| `DIFFUSION_NO_SC` | 设为 `1` 关闭 self-conditioning。默认开启。 |
| `DIFFUSION_SC_TOPK` | 实验用 self-conditioning top-K 截断（默认：`32`）。 |
| `DIFFUSION_NO_PKV` | 设为 `1` 关闭 device-glue 后端上的 prompt-KV 缓存。支持处默认开启。 |
| `DIFFUSION_NO_FUSED_DECODE` | 设为 `1` 关闭 GGML 融合整模型 diffusion decode，回退到逐算子 / 逐层 diffusion decode。 |
| `DIFFUSION_NO_FUSED_LMHEAD_TAIL` | 设为 `1` 关闭融合 output-norm + lm-head + softcap 尾部。 |
| `DIFFUSION_BATCHED_FORWARD` | 设为 `1` 后，对活跃 diffusion canvas 使用真正的 `DecodeCanvasBatched`；默认按请求时间片执行更快的融合单 canvas 路径。 |
| `DIFFUSION_LMHEAD_BATCH_CAP_MB` | diffusion lm-head logits 批处理内存上限，超过后回退到按序列 lm-head（默认：`300`）。 |

采样参数的优先级（从高到低，对应默认的 `--sampling-precedence config`）：

1. 服务端命令行参数 / 配置文件键（如 `--temperature`、`--top-p`、`--stop`）。
2. 上面列出的 `TENSORSHARP_*` 环境变量。
3. API 请求 JSON 中的字段（如 `temperature`、`top_p`、`stop`）—— 仅对第 1、2 步未设置的参数生效。
4. `SamplingConfig` 内置默认值（`temperature=0.8`、`top_k=40`、`top_p=0.9`、`min_p=0`、`repeat_penalty=1.1`、`repeat_last_n=64`、存在/频率惩罚均为 `0`、`seed=-1`、无停止序列）。

使用 `--sampling-precedence request` 时，第 1~3 步互换：请求中出现的参数优先
于服务端参数与环境变量，其余参数仍由服务端填充。无论哪种模式，服务端 `--stop`
在 `config` 下始终生效（与请求的列表合并），在 `request` 下则被请求替换。

## 音视频生成（MiniMax-H3）

MiniMax-H3 **同时生成视频与原生 32 kHz 立体声音轨**——一个扩散 Transformer 在同一条
token 序列上对打包好的“视频+音频”潜变量一起去噪，因此音频是模型输出的一部分，而不是
事后配上去的。最长 15 秒、24 fps。它是 CFG 蒸馏模型，因此必须用 `--cfg 1.0`；管线默认
20 步去噪，4–8 步是快速工作点。在 M5 Pro / Metal、22 帧、8 步、相同随机种子下实测：
256×256 比 stable-diffusion.cpp 快 2.4×（49.3 s → **20.9 s**），640×384 快 1.7×
（108.5 s → **63.1 s**）。

这个对比取决于硬件，在显存吃紧的卡上会反过来。在 16 GB 的 RTX 3080 Laptop / CUDA 上
（同样 22 帧、8 步、三次取最好），端到端反而是 stable-diffusion.cpp 更快：256×256 快
1.15×（37.8 s 对 43.6 s），640×384 快 1.07×（59.8 s 对 63.7 s）。但按**单去噪步**算，
TensorSharp 在这张卡上依然领先（按 8 步与 16 步的斜率折算为 3.325 s 对 3.338 s）；输的
是固定的启动开销，而且剩下 3.9 s 差距里约有 3 s 根本不是推理——一头是 H.264 编码
（stable-diffusion.cpp 写的是 AVI 里的 MJPEG+PCM），另一头是 .NET 进程启动对上原生
可执行文件。那台机器只有 16 GB 显存与 31.7 GB 内存，要面对 33.5 GB 的模型集合，权重和
页缓存都塞不下，于是启动开销主导了墙钟时间；显存峰值 TensorSharp 为 15 780 MiB，
stable-diffusion.cpp 为 12 035 MiB（整卡 16 384 MiB）。stable-diffusion.cpp 用的是
`--auto-fit --stream-layers --diffusion-fa --rng cpu`，因为它默认的 `--offload-to-cpu`
路径在这台机器上根本跑不动这个模型——它要把 17.7 GB 钉进只剩 12.3 GB 的内存里。

需要四个网络——两个去噪器之一，加上三个共用的伴随文件。它们依次加载、依次释放，
因此显存峰值取其中最大者，而不是四者之和：

| 角色 | 文件 | 大小 |
| --- | --- | --- |
| 去噪器——关键帧（`t2v` / `i2v` / `fl2v`） | `minimax_h3_fl2va_pruned-Q4_K.gguf` | 10.64 GiB |
| 去噪器——参考（`t2v` / `ref`） | `minimax_h3_ref2va_pruned-Q4_K.gguf` | 10.60 GiB |
| 文本编码器（Qwen3-VL-32B，50 层） | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | 16.97 GiB |
| 视频 VAE（空间 16× / 时间 4×） | `minimax_h3_video_vae_fp16.safetensors` | 5.21 GB |
| 音频 VAE（32 kHz 立体声） | `minimax_h3_audio_vae_fp32.safetensors` | 0.61 GB |

加载的是哪个去噪器，是按**文件名**判断的：名字里带 `ref2va` 的走参考检查点，其余
一律走首尾帧检查点——所以改名或重新量化时必须保留 `ref2va` 这几个字。

伴随文件与去噪器同目录时会自动解析，否则用 `--video-text-encoder`、`--video-vae`、
`--audio-vae` 指定。不给音频 VAE 仍然能出视频，只是没有声音——`--no-audio` 也一样，
它跳过音频解码来省下时间与显存；在 Ref2VA 上，`--ref-audio` 走的正是同一个 VAE 的
编码器，所以去掉它连参考音轨也一并失去。文本编码器**不带分词器**，
需要把 [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/processor)
的 `vocab.json` 与 `merges.txt` 放在它旁边（或设置 `TS_VIDEO_TOKENIZER`）。

**16 GB 的卡上会看到什么。** 一旦这套权重装不下，有两件事会起作用。其一，只要去噪器
与视频 VAE 放不进同一块显存，去噪器的设备驻留会在 VAE 加载**之前**先交还——否则一个
已经用完的 10.6 GB 去噪器会和 5.2 GB 的 VAE 一起挤在 16 GB 的卡上，而 WDDM 并不会让
这次分配失败，它会用共享主机内存兜住溢出部分，于是整个解码都跑在 PCIe 的速度上（解码
期间显存峰值 16 041 MiB → 约 5 600 MiB，在 640×384 上值 22 秒）。其二，权重是以指针形式
绑定到 mmap 的 GGUF 上的，因此去噪器文件会在第一次上传之前被顺序读一遍——在文本主干
产出隐藏状态的那一刻就开始读，并且与上传流水线并行、而不是在上传前 join，因为一边拷贝
一边缺页的 H2D 在这张卡上实测只有 0.91 GB/s，而页面已驻留时是 5.97 GB/s。两者合起来，
640×384 在 16 GB 的 RTX 3080 Laptop 上从 89.0 s 降到 63.7 s（256×256：67.2 s →
43.6 s），开关前后输出逐字节一致。交还驻留以实测空闲显存为门槛，预读则以空闲物理内存
为门槛。内存不够时预读会直接让路并照常继续；它会打印 `denoiser prefault skipped (not enough free RAM)`，但**仅在 `TS_H3_PHASE=1` 下可见**——默认设置下跳过是静默的。
因此两边都宽裕的机器上行为与之前完全一样。设 `TS_H3_PHASE=1` 会打印分阶段耗时——编码器
打开 / 主干 / 拆卸、预读、每一个去噪步、VAE 打开 / 解码——在自己的卡上正是从这里看时间
花在了哪。

随仓库提供的两个配置文件把四个网络全部写了进去，这正是它们能在首次运行时自动下载到
`${modelRoot}`（FL2VA 一套约 33.5 GB，Ref2VA 约 33.4 GB）并在之后复用的原因。两者只有
去噪器不同，所以跑完一个之后再跑另一个，只会下载它自己的 DiT：

```bash
# 关键帧：文生视频、图生视频、首尾帧
tensorsharp --config config/minimax-h3-fl2va.json \
  --prompt "a red fox trotting through falling snow, cinematic" --output fox.mp4

# 参考：同一主体，全新场景
tensorsharp --config config/minimax-h3-ref2va.json \
  --ref-image person.png --ref-image jacket.png \
  --prompt "she walks through a night market, neon reflections" --output market.mp4

# 两个文件同样可以直接托管服务端
./TensorSharp.Server --config config/minimax-h3-fl2va.json
```

两个配置都固定了 `"backend": "ggml_cuda"` 以及 640×384 × 22 帧 / 24 fps；命令行上的
`--backend ggml_metal` 会覆盖后端。它们都没有设置步数与引导，因为两个宿主对步数的写法
不同（CLI 是 `--diffusion-steps`，服务端是 `--video-steps`），而服务端根本没有 `--cfg`
参数——于是模型自身的默认值生效。

**文生视频。** 输出 `fox.mp4` 和带音轨的 `fox.wav`：

```bash
tensorsharp --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --prompt "a red fox trotting through falling snow, cinematic" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output fox.mp4
```

**图生视频——让照片动起来。** 图片成为**第一帧**，提示词决定后续动作：

```bash
tensorsharp --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --image portrait.jpg \
  --prompt "the person turns toward the camera and smiles, subtle handheld motion" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output animated.mp4
```

**首尾帧。** 两端钉住，模型补出中间的运动：

```bash
tensorsharp --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --image start.png --end-image end.png --prompt "a slow cinematic push-in" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output morph.mp4
```

**模式选择。** 一张图对 H3 有两种完全不同的含义，且使用不同的检查点。
不指定时自动推断，`--video-mode` 用于显式声明。

| 你想要什么 | `--video-mode` | 检查点 |
| --- | --- | --- |
| “让这张照片动起来” | `i2v` | `minimax_h3_fl2va_pruned-*` |
| “从照片 A 变到照片 B” | `fl2v` | `minimax_h3_fl2va_pruned-*` |
| “用这个人物，生成全新场景” | `ref` | `minimax_h3_ref2va_pruned-*` |
| “参考这个产品/人物，但换机位、背景和构图” | `ref` | `minimax_h3_ref2va_pruned-*` |
| 只有文本 | `t2v` | 都可以 |

无法解释成同一件事的组合会被直接拒绝并指名原因，而不是只兑现一半：在 FL2VA 上用
`ref`、在 Ref2VA 上传关键帧、`i2v` 却没有图片、`fl2v` 却没有 `--image` 和/或
`--end-image`、`ref` 却没有任何可参考的输入、`t2v` 却传了图片，以及关键帧与具名参考
同时出现——一段“看起来像是照做了”的视频才是更糟糕的失败。每条报错都会指出该加载的
检查点或该去掉的参数。

参考条件保留主体、其余全部重来——机位、背景和构图都由提示词决定，第一帧完全
不必与参考图相似：

```bash
tensorsharp --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-image person.jpg --ref-image bottle.png \
  --prompt "she holds the bottle up to the light on a rooftop at golden hour, slow orbit" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output rooftop.mp4
```

最多九张 `--ref-image`。参考图只会被缩小并保持自身宽高比，输出画布仍然由
`--width`/`--height` 决定。在 Ref2VA 检查点上，普通的 `--image` 同样会被当作参考图，
因此只会“上传一张图”的客户端无需改动即可使用。

参考也可以是**视频片段**（`--ref-video`，视频文件或帧目录）或**音频**
（`--ref-audio`）。视频自带的声音用 `--ref-video-audio` 按位置配对单独给出，
因为容器里的音轨无法通过帧解码器读取：

```bash
tensorsharp --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-video walk.mp4 --ref-video-audio walk.wav \
  --prompt "the same woman walks along a beach at sunset, wide shot" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output beach.mp4
```

参考视频是 H3 最昂贵的输入——22 帧 448x320 的参考会在输出所需的 1680 个 token 之外
再加 980 个条件 token，编码本身还要约 14 秒。它会被重采样到 24 fps、对齐到 17k+5
网格，并以 2 fps 呈现给语言模型。

**参考不是免费的，而且真正咬人的是第二笔开销。** 在去噪器内部它是线性且便宜的：
一张 640×384 的参考是 240 个 token，在 RTX 3080 Laptop 上，640×384 的 22 帧片段
每一步去噪从「无参考」的 4.37 s 涨到「八张参考」的 9.38 s——每张参考每步约 626 ms，
从一张到八张都是这个斜率。真正翻车的是 Qwen3-VL 那一趟：每张参考会往提示词里加
约 250 个视觉占位 token，并要过完整 50 层预填充，于是两张参考是 548 token 的提示词、
八张是 2086 token，文本条件耗时从约 65 s 涨到约 447 s，而整个 8 步去噪也才约 75 s。
超过大约四张参考之后，跑的就是编码器而不是去噪器了——先想着减少参考、换更好的参考，
再想着减步数。九张的上限是 TensorSharp 自己定的：打包后的序列是不加掩码整体注意的，
它的长度既是时间预算也是数值预算。另外记得在提示词里写清楚画面里是谁——参考图提供的
是身份，镜头内容仍然要靠提示词描述，否则会得到一段渲染得很好、但主体缺席的画面。

**质量。** 有两个设置起决定性作用：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--width` / `--height` | 640×384；有图时按该面积取图片宽高比 | **影响最大。** 人脸需要像素——256×256 下无论其他参数怎么调都会糊、都会变形。 |
| `--diffusion-steps` | 20 | 消除运动主体周围的彩色边缘。8 步是快速档，约 20 步干净，超过 30 提升有限。 |
| `--diffusion-seed` | 随机 | 有些种子构图更好，最省事的重试手段。 |
| 量化等级 | — | 显存允许就用 `-Q8_0` 去噪器而不是 `-Q4_K`。 |

`--cfg` 不是质量参数（H3 只接受 1.0），`--negative-prompt` 也不起作用，因为没有无条件分支。
注意这里的种子是 `--diffusion-seed` 而不是 `--seed`：`--seed` 是文本采样种子，不影响视频；
而不传 `--diffusion-seed` 时每次运行都会取一个新的随机种子——所以要做对照的两次运行必须显式设置它。

**在服务端，尺寸是启动参数。** 浏览器发送的是提示词，加上所托管检查点通过
`/api/models` 声明支持的那些条件输入——首帧、末帧，或最多 `maxReferenceImages` 个
参考——数值类参数一个都不发，因此每段视频都继承服务端默认值：

```bash
./TensorSharp.Server --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --video-width 640 --video-height 384 --video-steps 20 --video-frames 22 --port 5001
```

`--video-mode` 可以为只提供一种模式的部署固定条件模式；不指定时每个请求按其
自身携带的输入自动推断。

不指定 `--video-width`/`--video-height` 时，每个请求会按上传图片的宽高比来，
避免把 4:3 的照片拉成 16:9。

**HTTP 接口。** 有三个端点生成视频，且都禁用了请求超时：`POST /api/video-generate`、
`POST /api/video-generate/stream`（同样的请求体，改用 SSE，逐步推送
`{ videoGen, step, total, phase, detail, elapsedSeconds, etaSeconds }`，最后以
`{ done: true, … }` 结束），以及 OpenAI 形态的 `POST /v1/videos/generations`。三者共用
同一个解析器，因此同一份请求体到处都能用：`prompt`（必填）、`width`、`height`、
`frames`、`steps`、`cfg`、`cfg2`、`seed`、`fps`、`flowShift`、`negativePrompt`、
`sampler`、`cfgCacheStride`、`videoMode`、`generateAudio`、`imagePath`（或内联 base64
的 `image`）、`endImage`、`referenceImages`、`referenceVideos`、`referenceAudios` 与
`referenceVideoAudios`——最后一个按**下标**与 `referenceVideos` 配对。为音视频联合生成
与参考条件新增的字段同时接受下划线写法（`video_mode`、`generate_audio`、`end_image`、
`reference_images`……），两种都给时以驼峰为准；更早的字段只认驼峰。所有路径字段都必须
指向此前通过 `/api/upload` 上传的文件，并被限制在上传目录之内。

`/api/video-generate` 返回
`{ ok, url, audioUrl, width, height, frames, fps, seed, codec, elapsedSeconds }`，模型
没有生成音轨时 `audioUrl` 为 null；`/v1/videos/generations` 还接受 `"832x480"` 形式的
`size`、`negative_prompt` 与 `response_format`（`url` 或 `b64_json`），返回
`{ created, data: [{ url, b64_json }], audio_url, width, height, frames, fps, seed,
codec, elapsed_seconds }`。凡是模型自己能解释清楚的请求错误——模式与检查点不匹配、
模式缺少必需输入、关键帧与参考同时出现——都会以 400 带上模型自己的报错信息返回，而
不是笼统的 500；托管的模型本身不生成视频时，返回
`The loaded model is not a video-generation model.`

托管的模型能生成视频时，`GET /api/models` 会带上一个 `video` 对象（否则为 `null`），
客户端据此就能知道该提供什么，而不必去匹配架构字符串：`family`（`minimax-h3`）、
`supportsAudio`（解析到音频 VAE 后为 true）、`supportsImageConditioning`、
`supportsEndImageConditioning`（仅 FL2VA）、`supportsReferenceConditioning` 与
`maxReferenceImages`（Ref2VA 上为 9）。Web UI 正是读这几个字段来决定要不要提供末帧或
参考图。

**尺寸。** 宽高向上取整到 32 的倍数；帧数向上对齐到 `17k+5` 网格
（5、22、39、56、73、90……），fps 固定为 24，因此片段最长 15 秒。任意网格长度都能
正确解码——视频 VAE 每次跑 5 个潜变量帧、带 2 帧前瞻，并对接缝做交叉淡化。反过来，
一次性解码长片段会让细节被逐渐冲淡（对着条件照片测量，第 0 帧的相关度从 22 帧的
0.97 掉到 90 帧的 0.86），所以分块是正确性要求，而不是优化。

**长片段一旦发散，宁可报错也不会存盘。** H3 是对整段片段做双向注意的，因此 key 的
数量*就是*片段本身——22 帧是 2364 个打包 token，107 帧是 8646 个——而一个发散的
flow-matching 速度场解码出来的文件，长度、帧率、音轨时长全都对，只是每个像素都是黑的、
每个音频采样都被削平。所以遇到非有限的速度场时，管线会让整个请求失败并指出是第几步
出的问题，而不是把那个文件写出来。设 `TS_H3_TRACE=1` 可以打印每一步的潜变量与速度场
幅值：absmax 通常在真正变成无穷之前若干步就已经离开了正常区间。

**音频**写成独立的 `.wav`，因为封装进 MP4 需要一个未必安装的编码器。合并：

```bash
ffmpeg -i fox.mp4 -i fox.wav -c:v copy -c:a aac fox_with_audio.mp4
```

**`--backend cpu` 上完全不用 ggml 也能跑完整条流程。** 提示词编码、去噪、视频 VAE 解码
与音频声码器都有纯 C# 实现，视觉塔、因果 3-D 视频 VAE 编码器与音频 VAE 编码器同样如此——
因此 `t2v`、`i2v`、`fl2v` 以及参考图 / 参考视频 / 参考音频在那里都能用。所有计算都在主机上
以 F32 进行，所以它面向的是正确性、可移植性和没有加速器的机器，而不是速度：256x160x5f 的
单步 `t2v` 实测 69 秒，`ggml_cpu` 为 14 秒；而同一个 `i2v` 实测 70 秒，`ggml_cpu` 为 176 秒（单次采样，不要过度解读这个方向）。
它与 GGML 路径的一致性，以及唯一一个残差大于 GGML 自身内部差异的阶段（视觉塔），都记在模型卡片里。

完整说明见 [docs/models/minimax-h3_zh-cn.md](docs/models/minimax-h3_zh-cn.md)。

## 视频生成（Wan）

一个 `wan` GGUF 可以把提示词——Wan 2.2 模型还可再加一张首帧图片——变成 H.264 MP4，
`TensorSharp.Cli`、服务端的三个视频端点以及 Web UI 聊天都能驱动。Wan 是**纯视频**
家族：它声明不支持音频、不支持末帧、不支持参考条件，因此 `--end-image`、`--ref-image`、
`--ref-video`、`--ref-video-audio`、`--ref-audio`、`--audio-vae` 与 `--no-audio` 实际上
都是 MiniMax-H3 的参数——想要出片自带声音，见上文
[音视频生成（MiniMax-H3）](#音视频生成minimax-h3)。完整的架构细节见
[Wan 卡片](docs/models/wan_zh-cn.md)；本节是运维视角：该下载哪个检查点，以及哪些开关
真正影响墙钟时间。

### 选哪个检查点

| 家族 | 潜空间 | 模式 | 说明 |
|---|---|---|---|
| Wan 2.2 TI2V-5B | 48 通道、16×16×4（`Wan2.2_VAE.safetensors`） | 文生视频 + 图生视频 | 稠密 5B、24 fps、原生 720p；同分辨率下 DiT token 数约为 Wan 2.1 的 1/2.7，因此在消费级 GPU 上既最快也质量最好 |
| Wan 2.2 A14B（T2V / I2V） | 16 通道（I2V 输入 36 通道），`wan_2.1_vae.safetensors` | 文生视频 + 图生视频 | 两个 14B 专家在时间步边界切换；**两个**专家 GGUF 都必须就位（同一目录，或 `HighNoise/` + `LowNoise/`） |
| Wan 2.1 T2V（1.3B / 14B） | 16 通道，`wan_2.1_vae.safetensors` | 文生视频 | 单个 DiT |

每个家族都还需要 UMT5-XXL 文本编码器（`umt5-xxl-encoder-Q8_0.gguf`）和匹配的视频 VAE。
这三个伴随文件都会从 DiT 自身所在目录解析，包括 `VAE/`、`HighNoise/`、`LowNoise/` 这类
子目录，因此一个 `--local-dir` 就够了；`--video-vae` / `--video-text-encoder`（以及第二个 A14B 专家的
`TS_WAN_DIT2`）可以覆盖搜索结果。

Wan 是唯一会直接拒绝某个后端的家族：它可以运行在 `ggml_cuda`、`ggml_vulkan`、
`ggml_metal`、`ggml_cpu`、`cuda` 与 `cpu` 上，**不支持** `--backend mlx`。`ggml_cuda`
最快（RTX 2000 Ada、Wan2.1-1.3B F16、832×480×33f、30 步：`ggml_cuda` 12.0 秒/步，
`ggml_vulkan` 17.2 秒/步，Direct `cuda` 19.3 秒/步）；`cpu` / `ggml_cpu` 仅供功能性使用。

### 提速主路径：步数蒸馏检查点

**这是整个 TensorSharp 里最大的提速手段，代价只是换一个下载。** Wan2.2-TI2V-5B 的官方配方是
50 步 × 2 次无分类器引导前向 = **100 次 DiT 前向**。步数蒸馏检查点（Turbo / Lightning /
FastWan / DMD）本身就是按无引导、少步数训练的，因此同一段视频只需 **4 次 DiT 前向**，
去噪工作量降到 1/25。

TensorSharp 从 DiT 的**文件名**识别它：不区分大小写地匹配 `turbo`、`distill`、
`lightning`、`lightx2v`、`fastwan`、`-dmd`，或显式的 `<N>steps` / `<N>_steps`（1 ≤ N ≤ 16，
显式步数优先于标记）。只有标记而没有步数时默认为 4 步。加载时控制台会打印

```
step-distilled checkpoint detected -> 4 steps, guidance off (--diffusion-steps / --cfg override)
```

因此可以从日志确认识别是否生效。这里没有任何参数——蒸馏 GGUF 就当作普通的 `--model` 传入。

在 M5 Pro（20 核 GPU、48 GB 统一内存）、`ggml_metal`、Wan2.2-TI2V-5B Q8_0、
1088×832×121f = 27 404 token、图生视频（即完整的五秒 720p 级请求）上实测：

| | 基础检查点 | **Turbo 检查点** |
|---|---|---|
| DiT 前向次数 | 100（50 步 × CFG） | **4**（4 步，无引导） |
| 每次前向 | 120.2 秒 | 120.2 秒 |
| 去噪合计 | 12 020 秒 | **481 秒** |
| VAE 解码 121 帧 | 563 秒 | 563 秒 |
| **端到端** | **约 3 小时 30 分** | **17 分 30 秒** |

这两列之间只有 `--model` 路径不同。一旦用上蒸馏检查点，瓶颈就变成 VAE 解码（约占整段
运行的 55%），而不再是 DiT。

下载方式（TI2V-5B —— 注意文件名里的版本号用的是**下划线** `Wan2_2`，与基础仓库不同）：

```bash
pip install -U huggingface_hub
hf download hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF Wan2_2-TI2V-5B-Turbo-Q8_0.gguf --local-dir models
# Turbo 仓库不含 VAE 与文本编码器——从基础仓库取
hf download QuantStack/Wan2.2-TI2V-5B-GGUF VAE/Wan2.2_VAE.safetensors --local-dir models
hf download city96/umt5-xxl-encoder-gguf umt5-xxl-encoder-Q8_0.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/Wan2_2-TI2V-5B-Turbo-Q8_0.gguf \
    --backend ggml_metal --image first_frame.png --output out.mp4 \
    --prompt "the cat runs toward the camera, cinematic tracking shot" \
    --video-frames 121 --fps 24

dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/Wan2_2-TI2V-5B-Turbo-Q8_0.gguf \
    --backend ggml_metal --video-frames 121 --fps 24
```

Wan 2.2 I2V-A14B 的即插即用蒸馏 GGUF 是
[jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF](https://huggingface.co/jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF)，
它已经把蒸馏合并进两个专家，分别放在 `high_noise/` 与 `low_noise/` 下；两个都下载，
`--model` 指向其中任意一个即可。该仓库同样不含 VAE 与文本编码器，因此
`VAE/Wan2.1_VAE.safetensors` 取自
[QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF)，
UMT5-XXL 编码器同上。

> [lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning)
> 只发布 LoRA `.safetensors`，而 TensorSharp 没有任何 Wan LoRA 参数——请选择已经把
> 蒸馏烘焙进 GGUF 的仓库。

### `--cfg-cache-stride`（仅基础检查点）

一次带引导的去噪步是 `v = v_cond + (cfg-1)·d`，其中 `d = v_cond - v_uncond`。引导方向 `d`
在整个调度上的变化远比 `v` 缓慢，因此 `--cfg-cache-stride N` 让无条件前向每 `N` 步只跑
一次，中间各步复用缓存的 `d`。在 50 步下，`2` 只跑 100 次前向中的 77 次（**1.30×**），
`3` 跑 70 次（**1.43×**）。前三步与最后一步始终重新计算 `d`。服务端 JSON 字段：
`"cfgCacheStride": 2`。

这是一种近似——需要严格对齐参考样本时请关闭；在步数蒸馏检查点上也没有意义，因为它们本来
就无引导运行（cfg ≤ 1.0 时缓存会被自动禁用）。

### 让一个大请求变便宜，按收益排序

1. **换用步数蒸馏检查点。** 100 次 DiT 前向变成 4 次，收益远超其他所有手段。
2. **减少帧数。** 121 → 61 大约把注意力工作量降到 1/4（token 数是
   `latent_frames × (h/2) × (w/2)`，自注意力是 `O(tokens²)`），VAE 解码减半。
3. **减小画面面积**——但不要低于 Wan 的训练分辨率。低于约 0.3 MP 后模型进入分布外，
   视频会变*差*而不只是变便宜，管线也会给出警告。Wan 在 480p（832×480）与
   720p（1280×704）上训练，所以请在受支持的尺寸上生成，再事后缩小。
4. **减少步数**，仅限基础检查点——30 步相对官方 50 步在观感上很接近，成本降到 1/1.7。
5. **`--cfg-cache-stride 2` 或 `3`**——1.30× / 1.43×，仅限基础检查点。

同一台 M5 Pro、同一个 Turbo 检查点与输入图，分辨率与墙钟时间的关系：

| 输出 | Token 数 | 去噪 | VAE 解码 | **合计** |
|---|---|---|---|---|
| 736×544 × 81f（3.4 秒，480p 级） | 8 211 | 84 秒 | 159 秒 | **4 分 09 秒** |
| 736×544 × 121f（5 秒，480p 级） | 12 121 | 137 秒 | 237 秒 | **6 分 19 秒** |
| 1088×832 × 121f（5 秒，720p 级） | 27 404 | 481 秒 | 563 秒 | **17 分 30 秒** |

480p（约 0.4 MP）是 Wan *训练过*的分辨率，因此前两行属于分布内，而不是降级模式——当你
只能接受几分钟时，就该选它。

### 服务端默认值

`--video-frames N` 与 `--fps N` 为 Web UI 和三个视频端点设置服务端级**默认值**，不是上限；
请求中提供的 `frames` 或 `fps` 会各自独立覆盖。两者都不提供时使用模型配方：Wan2.2-TI2V
为 49 帧 / 24 fps，其余为 33 帧 / 16 fps。帧数会对齐到 VAE 的 `4k+1` 时序网格。改变时长
时请保持模型原生 FPS 而调整帧数——只改 FPS 只会改变播放速度。

### 其他 Wan 开关

这些开关用于 A/B 与调试，除注明外都会让速度变慢。`TS_WAN_DIT_KV_F16=0` 恢复 F32 的注意力
键值（F16 是默认值，单次 27k token 自注意力快 2.02×，且没有可测的精度损失）；
`TS_WAN_VAE_MPS_CONV=0` 在 Metal 上恢复 ggml 的 im2col+GEMM 卷积下降路径（MPSGraph 是默认
值，把 736×544×81f 的 VAE 解码从 159 秒降到 80 秒）；`TS_WAN_VAE_GEMM_MAX_MB` 设置 im2col
预算，`TS_WAN_VAE_TILE=0` 关闭分块；`TS_WAN_DIT_CAPTURE=0` 关闭常驻的 CUDA 图捕获 DiT 图；
`TS_WAN_DIT_FLASH=0` 强制走物化注意力；`TS_WAN_HEARTBEAT_S` 设置进度心跳间隔（默认 30 秒，
`0` 静默）；`TS_FFMPEG` 指定用于近无损 CRF 17 H.264 导出的 `ffmpeg`。

## 混合专家 CPU 卸载（`--n-cpu-moe`）

大型 MoE 模型几乎所有权重都花在路由专家上，而每个 token 只激活其中很小一部分
——Qwen3.6-35B-A3B 有约 11 GB 权重，活跃参数却只有约 3B。`--n-cpu-moe N` 把前 N
层的路由专家留在系统内存里，注意力、各个 norm、路由器以及始终活跃的共享专家仍留
在加速器上。因此 `--n-cpu-moe` 的含义是「这些专家不驻留显存」，而不是「这些专家在
CPU 上做矩阵乘」。它们究竟在哪里被相乘，取决于批大小：

* **Decode（单 token，或任何小于 `TS_HOST_MOE_DEVICE_MIN_BATCH` 的批，默认 128）。**
  由主机做乘法，直接从 GGUF 的 mmap 零拷贝读取量化块。一个 token 只会碰到
  `n_expert_used` 个专家，因此每层只有几 MB 的内存流量，而一次上传要花掉几百 MB。
* **Prefill（真正成批时）。** 专家会为这一张图**流式**送到加速器上并在那里相乘，
  随后暂存内存被下一层复用——它们始终不会常驻。512 token 的批会把这次上传摊薄到
  批内每个 token 上，而算术正是 GPU 的用武之地。llama.cpp 的调度器也是同样的判断
  （`ggml_backend_sched` 会把权重位于主机缓冲区的算子在超过
  `op_offload_min_batch_size` 时送回 GPU）。

有三点让流式这一侧变得很快，这也是 TensorSharp 的卸载 prefill 在相同文件上比
llama.cpp 快数倍的原因：

1. **主机页被锁定。** 从可分页的 mmap 里做拷贝无法 DMA——驱动会通过一小块固定内存
   中转。对专家区间调用 `cudaHostRegister` 一次性花费约 65 ms/GiB，却把 PCIe 5.0
   x16 上的传输从 9.3 GB/s 提升到 55.6 GB/s。用 `TS_HOST_MOE_PIN=0` 关闭，用
   `TS_HOST_MOE_PIN_MAX_MB` 设上限（默认为 cgroup/主机内存上限的 60%）。
2. **只发送这一批实际路由到的专家**，并按连续区段分组——和 llama.cpp 调度器用
   已用专家位图玩的是同一个把戏。512 token 时较大的专家池只会被部分覆盖，而在投机
   验证与轻负载服务产生的小批下，这项节省相当可观。`TS_HOST_MOE_EXPERT_FILTER=0`
   恢复整栈上传。
3. **拷贝是异步下发的**，走后端自己的 stream，并且随图一次同步，而不是每层同步三次。

被卸载的层仍然留在整模型融合图里：加速器在每个卸载层的路由器之后暂停，专家被相乘
（在主机上，或用流式送达的权重在加速器上），结果再交回，之后下一段才继续运行。token
里的其他一切——注意力、共享或稠密 FFN、LM head——仍然是一次图提交。这对
Qwen3.5/3.6、Gemma 4 MoE、GPT-OSS 与 DiffusionGemma 都成立；Gemma 4 MoE 的 prefill
图也按同样方式分段，而 DiffusionGemma 的块级 decode 会一次性把整块画布位置交给主机，
因此它卸载出去的是 GEMM 而不是 matvec。Qwen3.5/3.6 的 prefill 图同样分段，并且它和
Gemma 4 MoE 在张量并行下也保留融合图（见下文 `--tp` 说明）。GPT-OSS 现在同样有整模型
融合 prefill 图，因此它的卸载 prefill 也是分段而非逐层派发。仍然没有融合图的架构
（Nemotron-H）通过各自的逐层 MoE 算子进入流式路径，且只在单设备上；在 `--tp N` 下
它们保持在主机侧，以免 N 个 rank 各自流式传输同一份未切分的专家。

在 2 × Xeon 6952P + RTX PRO 6000 Blackwell（PCIe 5.0 x16）上实测，**pp8192 /
tg128**，显存为每进程峰值。与 llama.cpp 的完整对比——两种提示长度、每个模型五个卸载
深度，含跨两张 GPU 的 DeepSeek V4 Flash——见
[docs/moe_cpu_offload_benchmark.md](docs/moe_cpu_offload_benchmark.md)：

| 模型 | 设置 | 峰值显存 | Prefill | Decode |
| --- | --- | --- | --- | --- |
| gemma-4-26B-A4B (UD-IQ4_XS) | 默认 | 16.4 GiB | 11,274 tok/s | 161 tok/s |
| | `--n-cpu-moe 8` | 15.4 GiB | 6,500 tok/s | 80 tok/s |
| | `--n-cpu-moe 16` | 13.8 GiB | 4,888 tok/s | 55 tok/s |
| | `--cpu-moe`（30 层） | 10.8 GiB | 3,072 tok/s | 40 tok/s |
| Qwen3.5-35B-A3B (UD-IQ4_XS) | 默认 | 19.4 GiB | 9,405 tok/s | 160 tok/s |
| | `--n-cpu-moe 24` | 15.1 GiB | 5,259 tok/s | 52 tok/s |
| | `--cpu-moe`（48 层） | 11.3 GiB | 3,709 tok/s | 39 tok/s |
| gpt-oss-20b (Q8_0/MXFP4) | 默认 | 12.9 GiB | 12,925 tok/s | 213 tok/s |
| | `--n-cpu-moe 12` | 9.2 GiB | 6,394 tok/s | 52 tok/s |
| | `--cpu-moe`（24 层） | 4.7 GiB | 3,798 tok/s | 28 tok/s |
| DeepSeek-V4-Flash (UD-Q8_K_XL，2 张 GPU) | 默认 | 165.2 GiB | 4,387 tok/s | 51 tok/s |
| | `--n-cpu-moe 12` | 128.7 GiB | 428 tok/s | 10 tok/s |
| | `--n-cpu-moe 24` | 77.9 GiB | 236 tok/s | 5.3 tok/s |

在三个对照模型上，每一个卸载深度的 prefill 都是 llama.cpp 的 **4.5–10.9×**，decode
是 **2.5–4.5×**。峰值显存高于 llama.cpp（常驻时 1.1×，完全卸载时最高 3.5×）——详见
报告中的显存说明。

在 gemma-4-26B-A4B 上，任何卸载深度下的贪心输出都是 **token 一致**的。

同一功能在笔记本上的小机器画面（RTX 3080 Laptop 16 GB / i7-11800H，生成 96 个
token）——注意卸载配置离这张卡的 16 GB 有多远，以及 GPT-OSS 与 DiffusionGemma 的基线
本来就越过了 WDDM 的溢出阈值，这正是卸载在那里既能减小占用又能**提速**的原因：

| 模型 | 设置 | 峰值显存 | Prefill | Decode |
| --- | --- | --- | --- | --- |
| gemma-4-26B-A4B (Q4_K_XL) | 默认 | 16.1 GB | 190 tok/s | 39.7 tok/s |
| | `--n-cpu-moe 8` | 13.1 GB | 52 tok/s | 38.6 tok/s |
| | `--n-cpu-moe 16` | 10.2 GB | 24 tok/s | 21.6 tok/s |
| | `--cpu-moe`（30 层） | 4.8 GB | 15 tok/s | 17.7 tok/s |
| gpt-oss-20b (Q8_0) | 默认 | 16.2 GB | 2.7 tok/s | 0.3 tok/s |
| | `--n-cpu-moe 12` | 14.0 GB | 58 tok/s | 25.4 tok/s |
| | `--cpu-moe`（24 层） | 2.9 GB | 29 tok/s | 12.1 tok/s |
| diffusiongemma-26B-A4B (Q4_K_M) | 默认 | 16.0 GB | — | 1709 ms/步 |
| | `--n-cpu-moe 16` | 9.8 GB | — | 2287 ms/步 |
| | `--cpu-moe`（30 层） | 3.0 GB | — | 4697 ms/步 |

（上面这些笔记本 prefill 数字早于流式专家路径；同样的配置现在要快好几倍。）

注意事项：

* **选能装下的最小 N。** 每卸载一层都要付出 decode 吞吐的代价，因此只卸载刚好够
  腾出 KV 缓存所需显存的层数。
* **按比例算，decode 付出的代价最大。** 提示处理会把专家流式送到加速器上，与常驻
  运行相差不大；decode 则受限于主机每层从内存里读取 `n_expert_used` 个专家的速度，
  吞吐就消耗在这里。
* **工作线程数上限是 64。** decode 侧的矩阵乘只有一个 token 宽——是内存受限、每个
  算子后都有一次栅栏的工作——因此超过几十个线程之后，每多一个 worker 只是多一个栅栏
  参与者，在双路主机上还会增加跨 socket 流量。在 2× Xeon 6952P（192 核 / 384 线程）
  上用 gemma-4-26B `--cpu-moe` 实测，30 个卸载层的主机矩阵乘耗时（毫秒）：8 线程 45、
  16 线程 35、32 线程 17.6、64 线程 17.3、96 线程 26、192 线程 120。可用
  `--cpu-moe-threads N` 覆盖。
* **路由仍在加速器上**，因此专家的选择与权重与完全常驻路径挑出来的完全一致。
* **整模型融合图会被保留。** Qwen3.5/3.6、Gemma 4 MoE、GPT-OSS 与 DiffusionGemma
  各自把一整个 token（DiffusionGemma 是一整块画布）作为**单张**加速器图运行，卸载不会
  放弃这一点。每个卸载层的专家矩阵乘被从图里切出去；其余的一切——注意力、norm、路由器、
  共享/稠密 FFN 与 LM head——依然融合运行，加速器只在把该层的路由专家工作交给主机时暂停。
  Gemma 4 MoE 的 **prefill** 图也按同样方式分段，因此一个提示分块仍然是一次派发，
  而主机对块内每个 token 做的是真正的 GEMM。
* Nemotron-H 没有可保留的整模型融合图——它混合 Mamba2/注意力/MoE 的 decode 本来就是
  逐层派发——因此在它上面卸载只是把每层 MoE 走主机路径。
* **可与张量并行组合。** 在 `--tp N` 下，卸载层的接缝被并入各 rank 已有的 AllReduce
  分段调度中，因此多 rank 融合图得以保留——Qwen3.5/3.6 与 Gemma 4 MoE 的 decode 与
  prefill 都保持融合。每个卸载层只在主机上、对未切分的专家栈**求值一次**：主机专家后端
  是单一线程池，把那次矩阵乘按 rank 拆开只会在它上面串行，还要各自多付一次下载。随后
  这唯一的结果会被放置到使图自身的归约落在单 GPU 数值上的位置（该层输出会喂给后续
  AllReduce 时只放 rank 0，否则每个 rank 都放）。在 2× L4 24 GB 上实测（短提示，
  prefill 512 / decode 64）：

  | 模型 | 设置 | 常驻权重（GPU0 + GPU1） | Prefill | Decode |
  | --- | --- | --- | --- | --- |
  | Qwen3.5-35B-A3B (UD-IQ4_XS) | `--tp 2` | 9397 + 8032 MB | 1942 tok/s | 76.5 tok/s |
  | | `--tp 2 --cpu-moe` | 2277 + 912 MB | 66 tok/s | 17.8 tok/s |
  | gemma-4-26B-A4B (UD-IQ4_XS) | `--tp 2` | 6835 + 6087 MB | 3062 tok/s | 59.7 tok/s |
  | | `--tp 2 --n-cpu-moe 8` | 5459 + 4711 MB | 217 tok/s | 36.5 tok/s |
  | | `--tp 2 --cpu-moe` | 1588 + 840 MB | 60 tok/s | 13.2 tok/s |

  在 Gemma 4 上，贪心输出在大多数提示下与不带卸载的同一次 `--tp N` 运行逐字节一致，
  其余提示下也能一致数百个字符，之后走向一个等价的结尾；Qwen3.5 同样能跟随常驻运行
  到一个改写点。两者都是确定性的——差异来自主机专家矩阵乘的激活量化，也就是
  `TS_HOST_MOE_VERIFY=1` 在单 GPU 上报告的那约 1–5% 的相对差异。纯 C# 的 `cuda`
  后端没有主机 MoE 接缝，因此在那里使用 `--tp N --cpu-moe` 会打印
  `[moe-offload] WARNING` 并让专家保持常驻。
* **GLM 5.x 的卸载专家直接从映射里取。** 原生 glm-dsa 执行器把留在主机上的路由专家
  保持在 GGUF 的 mmap 里就地做乘法，而不是先拷进一块私有缓冲
  （`TS_GLM_MOE_MMAP=0` 则改为拷贝），正是这一点让「卸载一个 744B 模型的 30 层」
  在加载时几乎不花时间，而不是一次 100 GB 的 memcpy。它也能与 `--tp` 叠加：被卸载
  的层保持专家完整，由 rank 0 求值。在 3x RTX PRO 6000、GLM-5.2 UD-IQ2_XXS、
  prefill 2048 / decode 64 上实测：`--n-cpu-moe 30` 为 94.7 / 16.4 tok/s，全部常驻
  时为 915.9 / 43.9。这里卸载买到的是「装得下」而不是速度——它腾出的显存把可用
  上下文从 342,272 抬到了 646,400 token。
* `TS_HOST_MOE_VERIFY=1` 会在主机链路旁同时构建 GPU 上的专家链路，并报告二者的逐层
  偏差——用于在同样能塞进显存的模型上验证接缝的诊断手段。`TS_HOST_MOE_DEBUG=1` 打印
  分段计划（节点切点）与每个接缝的激活范数。`TS_HOST_MOE_TIMING=1` 报告卸载侧的墙钟
  时间，拆分为每次调用的准备开销与主机矩阵乘——这能告诉你一次慢运行到底慢在专家 GEMM
  还是它周围的脚手架上。

## 张量并行与分布式推理

TensorSharp 支持**张量并行（TP）**——按 Megatron-LM 列/行并行范式把单个模型切分
到多张 GPU 上——以及**分布式（多节点）张量并行**，让 TP 跨越多台通过 TCP 点对点
网络互联的机器。

### TP 与按层切分——在多卡机器上到底发生了什么

模型占用多张 GPU 有两种完全不同的方式，其中只有一种是 `--tp`。

**张量并行（`--tp N`）**把*每一层*都放到*每个 rank* 上，切分的是层*内部*的权重，
于是一次 decode 每张卡只读 `1/N` 的字节，各 rank 在每个层边界做 all-reduce。下表
中标为 ✅ 的架构都支持本地 TP，而且它是**按需开启**的——不主动要求，就不会有任何张量被切开。
✅ 本身不表示已支持分布式/跨节点；仅本地的路径会在说明中标出。

**按层切分**则是把*整层*放到不同设备上并顺序执行——设备 0 算第 0..k 层，把隐状态
交给设备 1，依此类推。切点并不是简单的 `n_layer/N`：加载器会测量每张卡的空闲显存，
再做装箱以均衡「占用预算的最大比例」，因此配置不一致的一组卡也能被均匀填满。没有
集合通信、层内也不切分，因此在慢速互连上不花额外代价，但同一时刻只有一张卡在忙。它适用于所有
走自己整模型执行器的架构——**DeepSeek V4 Flash（`deepseek4`）**、**GLM 5.x（`glm-dsa` / `glm5next`，
GLM-5.2、GLM-5.3 与 GLM-5.3-Flash 都算）**与 **Qwen 3.8 Flash Next（`qwen4exp`）**。前两者**不加任何
参数时的默认行为**就是它：它们会摊到所有可见 GPU 上，因为两者都装不进单卡，`TS_DSV4_NGPU`
与 `TS_GLM_NGPU` 用来限制使用几张卡。`qwen4exp` 上则是按需开启——默认单卡，写了 `--tp N` 才切分。

在**其他所有架构**上，不加 `--tp` 就是**只用一张 GPU**。通用的逐算子路径与融合图
路径都没有自动按层切分——装不下单卡的模型会在加载时失败，而不会被悄悄摊开。（那条
会告诉你正好需要多少 `--n-cpu-moe N` 的拒绝信息，来自 DeepSeek V4 与 GLM 5.x 的
整模型加载器。）对既不支持张量并行、也没有按层切分的架构传 `--tp N`，现在会在 stderr
上明确说明并回到单卡运行，而不是让其余的卡白白闲着。

所以在一台 3 卡机器上：只写 `--backend ggml_cuda`，GLM 5.x 与 DeepSeek V4 会用满三张卡
（按层切分），Gemma 4 与 Qwen 3.8 Flash Next 只用一张；再加上 `--tp 3`，GLM 5.x 切换成
原生本地单进程张量并行（GLM-5.2、GLM-5.3 与 GLM-5.3-Flash 都如此），DeepSeek V4 只是把按层切分限制在这 3 张卡上
（那里的 `--tp N` 仅仅是个设备数，与 `TS_DSV4_NGPU` 等价），Gemma 4 则用上三张卡，
Qwen 3.8 Flash Next 得到三路按层切分。
在已测量的 GLM-5.2 运行上，这个切换只会更慢，换来的仅仅是容量——见下文的**预期效果**实测数据。

**Qwen 3.8 Flash Next（`qwen4exp`）是新加入的那个。**它的 `--tp N` 走的是按层切分：每张
GPU 拿到一段连续的整层。这不是张量并行——`qwen4exp` 不切分任何权重——而且这也是
llama.cpp 对这个架构唯一提供的多卡模式（它的 `-sm row` 会直接拒绝加载）。所以请把它
当作容量特性。在 2× A100-80GB、Qwen3.8-Flash-Next-UD-Q2_K_XL（73.4 GiB）上实测：单卡与
双卡的贪心输出**逐字节一致**（SHA-256 相同）；显存从一张卡扛下全部变成 24.2 GB + 26.2 GB，
大致各占一半；吞吐则没有变化——prefill 约 1520-1550 t/s、decode 约 56 t/s，两种跑法都一样。
作为参照，同一台机器上的 llama.cpp 单卡为 pp1536 1094 / tg128 61.2，双卡 `-sm layer` 为
1200 / 61.5——它同样是 prefill 约 +10%、decode 不变。启动时会打印实际走的是哪种模式，
以及每张卡分到的层数与字节数。

`TS_Q4E_LAYER_SPLIT=20,28` 可以用显式的每卡层数覆盖自动均衡（精神上等同于 llama.cpp 的
`--tensor-split`）。给出无法满足的值时它会直接抛错，而不是静默忽略。这个开关值得一用，
是因为自动均衡只按权重定价，看不到视觉塔——视觉塔是后加载的，会落在 GPU 0 上。

### 本地张量并行（单进程，多 GPU）

在一个进程内把模型切分到 N 张 GPU 上（Direct CUDA，或 GGML CUDA / Vulkan 后端）。
每张 GPU 持有 `1/N` 的切分权重（列并行的 QKV / gate / up，行并行的 output /
down），外加一份完整的复制权重（归一化层、词嵌入、LM head）。各 GPU 的 KV 缓存
彼此独立；每次行并行投影之后由 AllReduce（CUDA P2P 拷贝 + 逐元素加法内核）把隐
藏状态重新汇聚。

```bash
# CLI：2 GPU 张量并行
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2

# 服务端：同样的参数（也可用 TENSORSHARP_TP_DEGREE=2 环境变量）
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model <model.gguf> --backend cuda --tp 2

# 配置文件 JSON
{ "tp": 2, "backend": "cuda", "model": "<model.gguf>" }
```

### 分布式张量并行（多节点，点对点 TCP）

把 TP 扩展到多台机器。每个节点运行自己的进程、管理自己的本地 GPU；节点之间通过
带长度前缀的分帧协议在 TCP 网格上通信。AllReduce 是分层的：先在节点内做本地
P2P 归约，再由各节点代表之间走 TCP，最后广播回来——每次集合通信只有 `1/tp_local`
的数据需要穿过网络。

```bash
# 2 节点 × 每节点 2 GPU（共 4 GPU）
# 节点 0：
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2 \
    --tp-node-id 0 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

# 节点 1：
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2 \
    --tp-node-id 1 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

# 以服务端作为集群前端：服务端必须是节点 0（负责采样并对外提供 HTTP 的 driver），
# 其余每个节点都运行一个 TensorSharp.Cli worker，使用相同的模型、后端与 peer 列表。
# TENSORSHARP_TP_* 环境变量同样可用。
# 节点 0（服务端 / driver）：
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model <model.gguf> --backend cuda \
    --tp 2 --tp-node-id 0 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

# 节点 1（CLI worker）：
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda \
    --tp 2 --tp-node-id 1 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

# 配置文件 JSON（每个节点一份）
{ "tp": 2, "tp-node-id": 0, "tp-peers": "192.168.1.10:9500,192.168.1.11:9500", "backend": "cuda" }
```

所有节点必须使用完全相同的 `--tp-peers` 列表（或 `TENSORSHARP_TP_PEERS` 环境变
量），并各自使用唯一的 `--tp-node-id`（或 `TENSORSHARP_TP_NODE_ID`）。示例中的端
口（`9500`）不是默认值——必须显式指定，且在所有节点之间可达。

### 支持的架构

| 架构 | TP 状态 | 说明 |
|---|---|---|
| Mistral 3 | ✅ | 融合 / 分离 QKV，YaRN RoPE |
| Gemma 4 | ✅ | 稠密 TP + MoE。GGML 上融合的整模 MoE 主干在**每个专家内部**切分（gate/up 列并行、down 行并行），从而保留全局专家 id；`TS_GEMMA4_TP_FUSED_MOE=0` 可回退到逐算子的整专家路径。Direct CUDA 上为逐专家切分 |
| Qwen 3.5 / 3.6 family | ✅ | GatedDeltaNet SSM 按 rank 划分 V-head 归属；GGML 上为专家并行 MoE（每个 rank 持有整个专家，shared expert 仍按 Megatron 切分）与列并行 LM head，Direct CUDA 上为专家切分。`cuda` 与 `ggml_cuda` / `ggml_vulkan` 均可运行——GGML 路径使用打包的按 rank GDN 内核（`TSGgml_Qwen35GdnLayerTP`）并把循环状态常驻设备 |
| Qwen 3.8 Flash Next | 按层切分 | 不是张量并行：`--tp N` 让每张 GPU 拿到一段连续的整层，这也是 llama.cpp 对 `qwen4exp` 唯一提供的多卡模式（它的 `-sm row` 会直接拒绝加载）。买的是容量而不是速度——2× A100-80GB、Qwen3.8-Flash-Next-UD-Q2_K_XL（73.4 GiB）实测：贪心输出与单卡逐字节一致（SHA-256 相同），显存从一张卡扛下全部变成 24.2 + 26.2 GB，prefill 约 1520-1550 t/s、decode 约 56 t/s 两种跑法一致。`TS_Q4E_LAYER_SPLIT=20,28` 可手动指定每卡层数 |
| GPT OSS | ✅ | MoE 专家切分，attention sink，YaRN。`cuda` 与 GGML 后端均可运行；GGML 路径目前仍按 token 逐个遍历专家（尚未使用专家并行） |
| Nemotron-H | ✅ | Mamba2 在 rank 0 上复制计算，MoE 专家切分。GGML 上的限制与 GPT OSS 相同 |
| GLM 5.x | ✅（仅本地） | 仅限 GGML GPU 后端；GLM-5.2、GLM-5.3 与 GLM-5.3-Flash 的原生 TP 路径都只支持本地单进程（`--tp-node-id` / `--tp-peers` 对整个 GLM 家族都是硬拒绝）。GLM-5.2 切 MLA head 与每个路由专家内的隐藏行。GLM-5.3-Flash 还会切 KDA head 并让每个 rank 持有对应的递归状态；其 MLA head 与路由专家隐藏行同样切分。注意力局部输出会在每次非线性 Sinkhorn 超连接之前归约。在 GLM-5.3-Flash 满足条件的分段快路径上，路由 MoE 局部输出先归约，随后每个 rank 在本地计算并加入其复制的共享专家；超连接、池化 indexer、router、norm、稠密层与 embedding 也保持不切分并按 rank 执行，output norm / LM head 留在 rank 0。CPU MoE、tracing、部分 `TS_GLM_TP_SHARD` 切分、超额 rank 或缺少原生超连接内核时使用组合调度器回退，其中共享专家只在 rank 0 运行一次；`TS_GLM_TP_FUSED=0` 可强制该诊断回退。GLM-5.3（非 Flash）同样以本地单进程的方式接受 `--tp N`，并在每个 rank 上复制 MLA 与 indexer cache，因此 `--tp` 会成倍放大 KV 占用；它在 `--tp N>1` 下从未被实际跑过——唯一有记录的算术是一个装不下的例子：`--tp 8` 每个 rank 需要 41.7 GiB，而卡只有 46 GB——因此它只是一个被接受的模式，而不是已验证的配置；另外在那里只要 `--tp N>1`，`--spec` 就会被拒绝。不传 `--tp` 仍默认按层切分 |
| DeepSeek V4 Flash | 按层切分 | 不是张量并行：整模型执行器按每张卡的空闲显存把连续的整层装箱分配，`--tp N`（或 `TS_DSV4_NGPU`）只限制这次切分用几张卡 |
| DeepSeek V4.1 Flash | 按层切分（+ 实验性 routed-MoE TP） | 按层放置是默认路径，也是有实测数据的路径。`TS_DSV41_TP=N`（2–8，且必须等于 `--tp` / `TS_DSV4_NGPU` 选中的 GPU 数）会把路由专家的 gate/up 沿 FFN 中间维、down 沿输入维切分，partial 结果经主机中转的 F32 缓冲归约；注意力、共享专家与各类 cache 仍按层放置。切分按块对齐且不等宽（2304 的中间维是 9 个 256 元素的 K-quant 块：两 rank 为 1280+1024，四 rank 为 768+512+512+512）。首次完整 Q2_K 实测比按层切分更慢，因此仍属实验性。注意力 TP 与分布式组尚未实现 |
| Hunyuan Dense | — | 单设备：既没有 TP 也没有按层切分。启动时会在 stderr 上明说，而不是让多余的 GPU 闲置 |
| DiffusionGemma | — | 不适用（扩散模型） |
| Qwen-Image-Edit | — | 不适用（图像生成） |

### 后端支持

TP 可运行在 **Direct CUDA** 后端（`--backend cuda`）以及 **GGML CUDA / Vulkan**
后端（`--backend ggml_cuda`、`ggml_vulkan`）上；MLX 为单设备。

在 GGML 后端上，每个 rank 在自己的 GPU 上拥有独立的 ggml 后端、常驻设备的权重分
片与 KV 缓存。跨 GPU AllReduce 使用 ggml-cuda 的集合通信（构建时能找到 NCCL 就用
NCCL，否则用其 P2P 流水线）；小载荷则改在主机内存中归约——因为 GGML 的激活本来就
在主机内存里，这样反而更省。

```bash
# GGML CUDA 后端上的 2 GPU
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_cuda --tp 2

# 指定各 rank 对应的物理 GPU
TENSORSHARP_TP_DEVICES=0,2 dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model <model.gguf> --backend ggml_cuda --tp 2
```

**预期效果。** GGML 上的 TP 最初是**容量**特性——让单卡显存装不下的模型完整跑在
GPU 上——而融合的按 rank 执行（Stage 1c）让它同时成为延迟收益。现在每个 TP block
都以按 rank 融合的原生计算图运行（注意力、稠密 FFN、MoE 主干、GatedDeltaNet），
而不再逐算子下发，因此每层每个 rank 只需几次图调度，而不是数百次原生调用。

在 2× RTX 2000 Ada（各 16 GB，PCIe，无 NVLink）上测得，prefill 512 / decode 64，
单位 tok/s：

| 模型 | 单 GPU | `--tp 2` |
|---|---|---|
| Gemma 4 E4B Q8_0 | 2760 / 37.3 | 2488 / **51.7** |
| Gemma 4 26B-A4B IQ4_XS | 1845 / 48.5 | 2537 / **51.2** |
| Qwen 3.5-9B Q8_0 | 1461 / 23.1 | 399 / **24.4** |
| Qwen 3.5-35B-A3B IQ4_XS | 装不下 | **184 / 18.1** |

Decode——TP 本该受益的访存瓶颈部分——在 Gemma 4 E4B 上达到单卡的 1.39×，在
Qwen 3.5-9B 上为 1.06×；两个 Gemma 4 模型的输出与单卡逐字节一致。Prefill 受计算
约束且要承担集合通信开销，因此在单卡装得下的模型上持平或略低于单卡。
Qwen 3.5-35B 在 16 GB 卡上根本装不下，只能靠 TP 运行。完整测量数据与尚待融合的
部分见 `TENSOR_PARALLELISM_PLAN.md`（Stage 1b 与 1c）。

TP 能带来多少，取决于互连带宽以及一层里究竟有多少能拆。在 GLM-5.2 上（3x
RTX PRO 6000，PCIe，无 NVLink）它什么也带不来：`--tp 3` 实测 pp2048 505.6 /
tg64 17.6，而按层切分是 915.9 / 43.9——78 层里每一层都要对 `[6144, n_tokens]`
的隐状态做两次 all-reduce，占用的总线时间比拆分省下的算力还多。而且每个 rank 都
要各自持有一份全长缓存，能装下的上下文因此从 342,272 token 掉到 91,136；它还改变
了归约顺序，所以在 2-bit MoE 上，对着录制的 llama.cpp 金标准，按层切分复现 5/6 条
提示，而 `--tp 3` 只复现 3/6。
只在带 NVLink 的机器上、或者模型没有别的办法装下时才用它。

| 变量 | 作用 |
|---|---|
| `TENSORSHARP_TP_DEVICES` | 各 rank 使用的 GPU 序号，例如 `0,2`（默认 `0..tp-1`） |
| `TS_GGML_TP_PARALLEL=0` | 顺序而非并发地驱动各 rank（诊断用） |
| `TS_GGML_TP_FUSED_MATMUL=1` | 由单个线程提交两个 rank 的线性层（默认关闭；它每次调用都要为每个 rank 分配设备缓冲，在 Qwen 3.5 35B 上实测慢 2.3×） |
| `TS_GGML_TP_DEVICE_AR_THRESHOLD` | 超过该元素数量时 AllReduce 走设备集合通信（默认 262144） |
| `TS_Q4E_LAYER_SPLIT=20,28` | 仅 Qwen 3.8 Flash Next：用显式的每卡层数取代按层切分的自动显存均衡。给出无法满足的值时直接抛错而不是静默忽略——这个开关有用，是因为自动均衡只按权重定价，看不到后加载、会落在 GPU 0 上的视觉塔 |
| `TS_GGML_F32_RESIDENT=0` | 每次调用重新绑定 F32 线性层权重，而不是常驻设备（诊断用） |
| `TS_GEMMA4_TP_FUSED_MOE=0` | 仅 Gemma 4：从融合的整模 MoE 主干（专家内部 Megatron 切分）回退到逐算子的整专家路径。融合路径在 26B 上加载时会物化约 10.5 GB 的专家分片（约 36 秒），换来约 10× 的 decode |
| `GGML_CUDA_AR_BF16_THRESHOLD` | ggml-cuda 集合通信在多大载荷以上把 F32 转成 BF16 再归约。TensorSharp 把 ggml 的默认值（1 字节，即总是转换）提高到 1 MB，使 decode 规模的集合通信精确归约；设为 `0` 则完全禁用转换 |
| `TS_QWEN35_LAYER_TRACE=1` | 打印首次前向的逐层残差流摘要，单卡与 TP 两条路径都会输出（诊断用） |
| `GGML_CUDA_ALLREDUCE` | `nccl` / `internal` / `none`，直接透传给 ggml |

### 约束

- `numHeads`、`numKVHeads` 与 `intermediateSize` 必须能被 TP 度整除。
- 量化权重的行并行切分要求 `ne0` 能被 `tp × blockSize` 整除。
- TP 下的批处理 / 连续批处理前向目前实现于 Mistral 3；MoE 模型（Gemma 4、Qwen 3.5/3.6、GPT OSS、Nemotron-H）在 TP 下回退到按序列前向。

### 集群调优与诊断

本地 AllReduce 优先使用 CUDA 点对点（P2P）DMA。启动时，并行组会为每一对报告支持
P2P 的设备启用 peer access，随后做一次往返自检：部分拓扑（挂在某些 PCIe 交换机
后的 L4、开启 IOMMU 的主机、BAR1 窗口过小）虽然声称支持 P2P，却会静默传输损坏数
据——任何未通过自检的设备对都会被永久降级为主机中转。完全不支持 peer access 的
硬件（A16 vGPU 配置、大多数消费级显卡）则一开始就走主机内存中转。

| 变量 | 默认值 | 作用 |
|---|---|---|
| `TENSORSHARP_TP_DISABLE_P2P=1` | 关闭 | 所有跨 GPU 传输一律经主机内存中转，与无 P2P 硬件的路径完全一致。用于判断某个多 GPU 缺陷是否出在 P2P DMA 路径上。 |
| `TENSORSHARP_TP_HOST_ALLREDUCE=1` | 关闭 | 本地 AllReduce 改为设备→主机、CPU 求和、主机→设备。更慢，但与多节点归约路径完全一致，便于隔离 P2P 正确性问题。 |
| `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS=N` | `120` | 节点向 peer 重试连接的时长。节点通常由人工间隔数秒到数分钟启动，peer 的监听端可能尚未就绪；编排系统较慢时可调大。 |
| `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS=N` | `300` | peer 套接字的单次接收超时。否则卡住的 peer 会一直阻塞到操作系统 TCP keepalive 超时（往往 2 小时以上），而不是让集合通信失败。 |

启动日志会明确给出拓扑：本地并行组打印
`Tensor parallelism: N GPUs (<设备名>)`，P2P 降级会打印 `TP: P2P disabled…` 或自
检告警，每个分布式节点在网格建立完成后打印
`[TcpCommunicator] Rank r/N connected to all peers.`。

### Redis 支撑的共享状态

服务端可以选择把 KV 缓存块与 OpenAI Responses API 存储持久化到 Redis，从而实现跨
会话的 KV 复用与持久化的响应存储：

```bash
# 同时为 KV 缓存与 Responses API 启用 Redis
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model <model.gguf> --backend cuda \
    --redis-url localhost:6379

# 仅启用 KV 缓存层，TTL 设为 12 小时
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model <model.gguf> --backend cuda \
    --paged-kv-redis-url localhost:6379 --paged-kv-redis-ttl 720
```

## 功能 × 环境变量矩阵

每个主要功能由哪些环境变量（以及对应的 CLI 参数）控制的速查矩阵。**加粗**的变量是该功能的开关；其余的是该功能默认启用后的调优参数。

#### 连续批处理 & 分页 KV 缓存

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| 连续批处理引擎（`InferenceEngine` + 调度器） | 在 `TensorSharp.Server` 中默认启用 | `TS_SCHED_DISABLE_BATCHED=1` 强制按序列回退 | `--no-continuous-batching` / `--continuous-batching` |
| 旧的按会话分页 KV 管理器 | 已从服务端请求路径移除 | `TS_KV_PAGED_CACHE`（`0` / `1`）、`TS_KV_BLOCK_SIZE` 仅为兼容 / 独立测试保留 | `--paged-kv` / `--no-paged-kv`、`--paged-kv-block-size N` |
| 旧的分页 KV SSD 冷层溢出 | 关闭 | `TS_KV_CACHE_MAX_RAM_MB`、`TS_KV_CACHE_SSD_DIR`、`TS_KV_CACHE_MAX_SSD_MB` | `--paged-kv-ram-mb`、`--paged-kv-ssd-dir`、`--paged-kv-ssd-mb` |
| 旧的分页 KV 块量化（TurboQuantKvCodec） | 关闭（`0` = 透传） | `TS_KV_PAGED_QUANT_BITS`（`0` / `2` / `4` / `8`） | `--paged-kv-quant-bits` |
| 跨请求的块级哈希前缀共享 | 启用 | `TS_SCHED_PREFIX_CACHE=0` 关闭 | — |
| 调度器调优（每步 token 预算、最大同时序列数、prefill 分块、块池大小、decode quantum） | 引擎默认 | `TS_SCHED_MAX_BATCHED_TOKENS`、`TS_SCHED_MAX_RUNNING_SEQS`、`TS_SCHED_PREFILL_CHUNK`、`TS_SCHED_SOLO_PREFILL_CHUNK`、`TS_SCHED_NUM_BLOCKS`、`TS_SCHED_BLOCK_SIZE`、`TS_SCHED_DECODE_QUANTUM` | — |

#### 按模型的批处理 / 分页前向（`IBatchedPagedModel.ForwardBatch`）

| 模型 | 默认状态 | 切换默认的环境变量 | 原生内核子开关 |
|---|---|---|---|
| Mistral 3 | 启用 | — | `TS_PAGED_ATTN_KERNEL` = `native`（默认）/ `tensor` / `managed` |
| Gemma 4 | 启用 | `TS_GEMMA4_BATCHED=0` 强制走旧的按序列路径 | — |
| Qwen 3.5 / 3.6 系列 | 启用 | `TS_QWEN35_BATCHED=0` 强制走旧的按序列路径（或 `--no-continuous-batching`） | `TS_QWEN35_BATCHED_GDN_NATIVE=1` 启用原生批处理 GDN 内核；`FUSED_ATTN_LAYER_MIN_SEQ_LEN=N` 覆盖融合注意力启用阈值（默认 4096） |
| GPT OSS | 启用 | `TS_GPTOSS_BATCHED=0` 强制走旧的按序列路径 | `TS_GPTOSS_PAGED_ATTN_MANAGED=1` 强制使用托管 (C#) sinks softmax，而非原生带 sinks 的分页注意力内核 |
| Nemotron-H | 启用 | `TS_NEMOTRON_BATCHED=0` 强制走旧的按序列路径 | `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1` 启用原生批处理 Mamba2 步（NEON SIMD + GCD 并行） |
| GLM 5.x | 未实现——并发走的是原生的按序列**槽位**（每个请求拥有自己的 MLA 与索引器缓存以及自己的 `n_past`；绑定请求只是切换活跃槽位，不搬运任何 KV 字节）。MLA 每个 token 只存一行 576 宽的数据，DSA 索引器打分的也正是这段连续历史，所以这里没有可供批处理的分页 KV 布局 | — | 跨槽位的批处理融合解码默认开启（一张图、每个序列一个 token、权重只读一次）：4 路并发下总解码吞吐 1.81 倍。设置 `TS_BATCHED_FUSED_DECODE=0` 可切回串行融合 decode；`TS_GLM_BATCHED_DECODE=0` 也会让 GLM 原生侧拒绝批处理。批处理会改变 GEMM 形状，而 2-bit MoE 可能把这点差异放大成不同的专家选择。 |
| DiffusionGemma | Web UI 路径使用独立 diffusion 调度器；不是 `IBatchedPagedModel` 自回归路径 | `DIFFUSION_MAX_BATCH`、`DIFFUSION_STEPS` | `DIFFUSION_BATCHED_FORWARD=1` 启用真正的批处理 canvas decode；GGML 融合 decode 默认开启，可用 `DIFFUSION_NO_FUSED_DECODE=1` 关闭 |

#### 投机解码

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| 投机解码引擎（单序列） | 关闭 | **`TS_SPEC=1`**（旧写法 `TS_MTP_SPEC`） | `--spec` / `--no-spec` |
| 投机算法 | `auto` | `TS_SPEC_TYPE` | `--spec-type auto\|draft-head\|block\|ngram` |
| 每步最多起草 token 数 | `8` | `TS_SPEC_DRAFT`（旧写法 `TS_MTP_DRAFT`） | `--spec-draft N` |
| 草稿置信度门限 | 按算法（`0.15` / `0.35` / `0`） | `TS_SPEC_PMIN`（旧写法 `TS_MTP_PMIN`） | `--spec-pmin X` |
| Gemma 4 独立草稿 GGUF（`gemma4-assistant`） | 无 | `TS_SPEC_DRAFT_MODEL`（旧写法 `TS_MTP_DRAFT_MODEL`） | `--draft-model <path>` |
| Muse-Glimmer DFlash / DFlash2 草稿器 GGUF | 无 | `TS_MUSE_GLIMMER_DFLASH` | `--draft-model <path>` |
| Qwen 3.5 / 3.8 DFlash2 草稿器 GGUF | 无 | `TS_QWEN35_DFLASH` | `--draft-model <path>` |
| 融合 DFlash 计算图（ggml） | 开启 | `TS_DFLASH_FUSED=0` 回退到逐算子草稿器 | — |
| DFlash 投机 prefill 分块 | `1024`（受草稿器环形缓冲与主干窗口限制） | `TS_DFLASH_PREFILL_CHUNK` | — |
| DFlash2 候选选择器 | 检查点带有时开启 | `TS_DFLASH_SELECTOR=0` 改为按逐位置 argmax 起草（仅用于归因分析） | — |
| DFlash2 分组卷积 | 检查点带有时开启 | `TS_DFLASH_CONV=0` 去掉它（同上，仅用于归因分析） | — |
| Qwen 3.5/3.8 递归状态快照 | 开启 | `TS_Q35_VERIFY_SNAPSHOTS=0` 回退为先恢复验证前的状态副本、再对已接受前缀重新前向（更慢；见 [qwen35_zh-cn.md](docs/models/qwen35_zh-cn.md)） | — |
| Gemma 4 融合验证 / 草稿内核（ggml） | 开启 | `TS_GMTP_NO_FUSED=1` 回退到逐算子 | — |
| Gemma 4 部分接受时的稠密快速回滚 | 开启 | `TS_GMTP_NO_FAST_ROLLBACK=1` 恢复保留前缀回滚 | — |
| Gemma 4 验证主干路径 | 线性（单序列） | `TS_GMTP_BATCHED_TRUNK=1` 走批量分页主干 | — |

#### GLM 5.x（`glm-dsa`）

完整列表（含调试与 A/B 开关）见 [GLM 卡片](docs/models/glm_zh-cn.md#环境变量)。

| 特性 | 默认 | 环境变量 | 对应 CLI |
|---|---|---|---|
| 执行器 | 原生整模型 ggml 计算图 | `TS_GLM_NATIVE=0` 在 GGML 后端上改走托管逐算子路径 | `--backend cpu` / `cuda` 本来就是托管路径 |
| Prefill 微批 | `1024` | `TS_GLM_UBATCH=N`——在 GLM-5.2 上、显存允许时，`2048` 实测 pp2048 1145.8，对比 918.9 | — |
| 按层切分使用的 GPU 数 | 全部可见 GPU | `TS_GLM_NGPU=N` | — |
| 每张卡为计算缓冲预留的余量 | `3072` MB | `TS_GLM_VRAM_RESERVE_MB=N` | — |
| 上下文窗口 | 自报长度只作**上界**，加载后按实际空闲显存重新定档 | `MAX_CONTEXT=N` 把它变成硬上限 | —（仅环境变量） |
| 张量并行切哪一半 | `3`（头 + 路由专家） | `TS_GLM_TP_SHARD`（1 头、2 专家、3 两者都切）、`TS_GLM_TP_OVERSUBSCRIBE=1` 允许多个 rank 挤在一张卡上做测试 | `--tp N` |
| GLM-5.3-Flash TP 执行方式 | 完整本地 GPU 配置满足条件时并发提交按 rank 分段计算图 | `TS_GLM_TP_FUSED=0` 强制使用组合调度器诊断回退；CPU MoE、tracing、部分切分、超额 rank 或缺少原生超连接内核时也会自动选择该回退 | — |
| 主机端专家从 GGUF 映射直接读取 | 开启 | `TS_GLM_MOE_MMAP=0` 改为拷进私有缓冲 | 由 `--n-cpu-moe N` 决定哪些层 |
| 跨序列的批处理融合解码 | 开启 | **`TS_BATCHED_FUSED_DECODE=0`** 可关闭；`TS_GLM_BATCHED_DECODE=0` 让原生侧拒绝它 | — |
| Flash attention / 融合 lightning 索引器 | 开启 | `TS_GLM_FA=0`、`TS_GLM_FUSED_LID=0` 退回到基本算子拼装 | — |
| 缓存的已构建+已分配计算图数量 | `8` | `TS_GLM_GRAPH_CACHE=N` | — |
| 权重加载并行度 | `16` 线程 / `64` MB 分块 | `TS_GLM_LOAD_THREADS`、`TS_GLM_LOAD_CHUNK_MB` | — |

#### DeepSeek V4.1 Flash（`deepseek41`）

完整情况（含原生加载器自己的那些开关）见
[DeepSeek V4.1 卡片](docs/models/deepseek41_zh-cn.md)。

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| 执行器 | 原生整模型 ggml 计算图；`--backend ggml_cuda` 是服务路径 | — | `--backend cpu` 改用 100% 纯 C# 的 `DeepSeek4CpuExecutor` 跑完整计算图——没有 ggml、没有原生库、也没有 GPU——它是正确性与可移植性路径，而非服务路径；`--backend ggml_cpu` 又是另一条路径：在标量 ggml CPU 内核上跑原生计算图 |
| 纯 C# 执行器的计算宽度 | `--backend cpu` 上为 `Environment.ProcessorCount`，其余后端为 min(核数, 32) | `TS_DSV4_THREADS=N` | — |
| 纯 C# 执行器的数值把关 | `InferenceWeb.Tests.Dsv41CpuExecutorTests` 以独立的 PyTorch 参照实现 `eng/dsv41-reference.py` 为准，`atol=rtol=2e-5`，并要求贪心 argmax 完全一致；覆盖一次性 prefill、按 1/3/5/8 分块的 prefill（每个位置都检查）与 reset | **`TS_DSV41_FIXTURE_DIR`**——不设置它这些测试会静默返回。fixture 是一个 5 层、256 隐藏维、16 token 的 F32 合成模型，因此它确立的是与参照实现的架构一致性，而不是真实 246 GiB Q2_K 权重上的 CPU/CUDA 一致性 | — |
| 逐张量 trace（与参照实现对拍） | 关闭 | `TS_DSV4_CPU_TRACE_DIR=<dir>` 写出与 `eng/dsv41-reference.py --output` 完全相同的逐张量文件，两个目录可以逐张量 diff | — |
| Engram sidecar | `deepseek41.engram.bin` 在所有后端（含 `--backend cpu`）都是必需的 | 在 `--backend cpu` 上 `TS_DSV41_ENGRAM_DEVICE` 必须为 `0`（其他取值会在读取任何权重之前被拒绝）；`TS_DSV41_ENGRAM_WARM` / `_THREADS` / `_RANDOM` / `_SIDECAR` 仅对原生加载器有效，在这里是空操作 | — |
| 路由 MoE 张量并行 | 关闭 | `TS_DSV41_TP=N` 仅在原生路径上有效；在 `--backend cpu` 上任何非零值都会被拒绝，分布式 TP 组同样被拒 | `--tp N` |
| 仅原生加载器可用的注意力 / gather 开关 | — | `TS_DSV41_SPARSE_FA`、`TS_DSV41_COMPACT_RAW_GATHER`——在 `--backend cpu` 上是空操作 | — |
| 投机起草 | V4.1 在任何后端上都没有 | `TS_DSV4_DSPARK` 在 V4.1 上是硬拒绝，而不是 DeepSeek V4 那种「告警后继续」 | `--draft-model`（同样被拒绝） |
| 视觉伴随件 | 原生 ggml 组件 | — | 在 `--backend cpu` 上不可用，那里 `--mmproj` 会抛异常；「视觉伴随件跟随文本模型落到同一后端」说的是 `--backend ggml_cpu` |
| `--backend cpu` 上的上下文窗口 | `MAX_CONTEXT` 默认封到 `65,536`，且所有 compressed cache 行都在主机内存里，因此宣称的 1M 在这里到不了 | `MAX_CONTEXT=N` | —（仅环境变量） |
| `--backend cpu` 上的多轮 KV 前缀复用与按序列 slot | 都没有：任何分叉的新一轮都会重新 prefill，并发请求会串行化 | — | — |

#### MiniMax-H3 视频 + 音频

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| 视频 VAE 加载前先交还去噪器的设备驻留 | 两者放不进空闲显存时开启（16 GB 卡上解码峰值 16 041 → 约 5 600 MiB，640×384 上值 22 秒） | —（按实测空闲显存自动判断） | — |
| 第一次上传前预读去噪器 GGUF | `3` —— 预读与上传流水线并行 | `TS_H3_PREFAULT`：`0` 关闭，`1` 串行，`2` 与文本条件编码重叠（更差——编码器自己的 17 GB 会流过同一份页缓存，把刚放进去的页挤掉），`3` 默认 | — |
| 预读使用的读取流数 | `1` | `TS_H3_PREFAULT_THREADS=N` —— 这里流越多越慢，因为这次读取是与它要预热的拆卸和上传并发进行的（16 GB RTX 3080 Laptop、640×384、三次取最好：1 流 63.9 s，4 流 64.9 s，16 流 66.6 s） | — |
| 文本编码器主干分组运行，每组用完即释放其设备副本 | 关闭 | `TS_H3_TE_GROUP=N` 指定每组层数 —— 能消掉编码器自身的溢出（峰值 16 041 → 12 981 MiB）且逐位一致，但在 16 GB RTX 3080 Laptop 上实测慢约 3 秒：一次性 prefill 每个权重只读一遍，分组照样要搬完 17 GB，还多出分配/失效的开销 | — |
| 分阶段耗时（编码器打开 / 主干 / 拆卸、预读、每个去噪步、VAE 打开 / 解码） | 关闭 | `TS_H3_PHASE=1` | — |
| 每步的潜变量 / 速度幅值 | 关闭 | `TS_H3_TRACE=1` | — |
| 执行路径 | 七张原生整网络 ggml 计算图 | — | `--backend cpu` 改用纯 C# 跑同一条流程（t2v、i2v、fl2v 与参考条件） |
| 托管路径与 GGML 的对拍诊断 | 关闭 | `TS_H3_DUMP_TE`、`TS_H3_DUMP_VEL_V`、`TS_H3_DUMP_VEL_A`、`TS_H3_DUMP_VIS` 把这些张量写到磁盘，使两条路径可以在**同一次**前向上比较；`TS_H3_DIT_LAYERS=N` 在**两条路径上**同时截断主干，把比较变成误差随深度的曲线；`TS_H3_NO_FLASH=1` 让 GGML 走显式 softmax 注意力而不是它的 flash 内核 | — |
| 伴随网络与分词器覆盖 | 在去噪器同目录解析 | `TS_VIDEO_TEXT_ENCODER`、`TS_VIDEO_VAE`、`TS_VIDEO_AUDIO_VAE`、`TS_VIDEO_TOKENIZER` | `--video-te`、`--video-vae`、`--audio-vae` |

#### 张量并行与分布式推理

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| 本地张量并行（单进程，多 GPU） | 关闭（`1` 张 GPU） | **`TENSORSHARP_TP_DEGREE=N`** | `--tp N`（CLI 与服务端） |
| TP 各 rank 使用的 GPU 序号（GGML 后端） | `0..tp-1` | `TENSORSHARP_TP_DEVICES=0,2` | — |
| `qwen4exp` 按层切分（`--tp N`）每张卡的显式层数 | 自动显存均衡 | `TS_Q4E_LAYER_SPLIT=20,28` | — |
| 分布式 TP 节点编号（多节点） | 未设置（关闭） | **`TENSORSHARP_TP_NODE_ID=N`** | `--tp-node-id N`（CLI 与服务端；服务端必须是节点 `0`） |
| 分布式 TP peer 端点 | 未设置（关闭） | **`TENSORSHARP_TP_PEERS=host1:port1,host2:port2`** | `--tp-peers host1:port1,host2:port2`（CLI 与服务端） |
| peer 连接重试窗口（多节点） | `120` 秒 | `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS=N` | — |
| 单次接收超时（多节点） | `300` 秒 | `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS=N` | — |
| 强制跨 GPU 拷贝走主机中转 | 关闭（可用时使用 P2P） | `TENSORSHARP_TP_DISABLE_P2P=1` | — |
| 强制本地 AllReduce 走主机中转 | 关闭（设备到设备） | `TENSORSHARP_TP_HOST_ALLREDUCE=1` | — |

#### Redis 共享状态（仅服务端）

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| Redis KV 缓存层 | 关闭 | **`TS_KV_CACHE_REDIS_URL`** | `--redis-url` 或 `--paged-kv-redis-url` |
| Redis KV 缓存条目 TTL | `1440` 分钟（24 小时） | `TS_KV_CACHE_REDIS_TTL_MINUTES`（`0` = 不过期） | `--paged-kv-redis-ttl` |
| Redis Responses API 存储 | 关闭（内存） | **`TS_RESPONSES_STORE_REDIS_URL`** | `--redis-url` |

#### 后端

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| 默认计算后端 | `ggml_metal`（macOS）、`ggml_cpu`（Windows/Linux） | `BACKEND` | `--backend` |
| 首次前向 logits 导出（后端 A/B） | 关闭 | `TS_DUMP_LOGITS=<路径>` 把**第一次真实前向**的 logits 以原始 float32 一次性写入该路径，并刻意跳过预热前向（`WarmUpKernels` 会先跑一次丢弃用的 decode 和 prefill，导出那几次等于在一个无意义的 token 上比较两个执行器）。这样就能用 logit 向量而不是生成文本来比较两个后端——贪心解码会把一次几乎打平的比分变成一句明显不同的话 | — |
| MLX 后端库查找 | 优先探测应用目录 | `TENSORSHARP_MLX_LIBRARY`（`libmlxc` 完整路径）、`TENSORSHARP_MLX_LIBRARY_DIR`（目录） | — |
| MLX 流水化贪心 decode（仅 CLI） | 满足条件时启用 | `TS_MLX_PIPELINED_DECODE=0` 关闭 | — |
| 使用 `mlock(2)` 钉住 GGUF mmap，使权重常驻 | 启用 | `TS_MLX_MLOCK_GGUF=0` 关闭 | — |
| MLX 融合多维 KV 写入（每个 cache block 单次 `slice_update`） | 启用 | `TS_MLX_FUSED_KV_WRITE=0` 回退到按 head 循环 | — |
| MLX 批处理 MoE 解码（Qwen 3.5/3.6 MoE） | 启用 | `TS_MLX_BATCHED_MOE_DECODE=0` 走旧的按专家路径 | — |
| MLX MoE gate+up+SiLUMul 融合 Metal kernel | 启用 | `TS_MLX_MOE_FUSED_GATE_UP_SILU=0` 走旧的 3-dispatch | — |
| MLX 设备端 MoE router top-K + softmax | 满足条件时启用 | `TS_MLX_DEVICE_ROUTER=0` 关闭 | — |
| MLX 解码层边界 `async_eval` 间隔 | Gemma 4：每 4 层；Qwen / Nemotron：每 16 层 | `TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS=N` 或 `TS_MLX_EVAL_EVERY_N_LAYERS=N`（支持处 `0` = 关闭） | — |
| MLX 分配器上限（内存 / 缓存 / wired buffer） | 按宿主机派生 | `TS_MLX_MEMORY_LIMIT_MB`、`TS_MLX_CACHE_LIMIT_MB`、`TS_MLX_WIRED_LIMIT_MB` | — |

#### 纯 C# CPU 后端（`--backend cpu`）

托管矩阵乘跑在一个常驻的"自旋后挂起"工作线程池上，而不是每次矩阵乘一个 `Parallel.For`。
它**刻意不占满**所有核心：CPU 路径的其余部分仍然使用 ThreadPool，而池内线程在两次任务之间
自旋，占满每个核心会把那部分工作饿死。在 122 个 CPU 的配额上用 gemma-4-E4B-it-Q8_0 实测
（prefill / decode tok/s，每格为两次交替运行）：关闭池 21.7,21.0 / 2.0,2.4；32 线程 24.9,24.1 / 4.9,5.0；48 线程 25.6,28.5 / 5.4,6.0；
61 线程 24.2,24.9 / 6.3,5.9；122 线程 13.5 / 4.8——即默认宽度下 prefill 约 +15%，decode 约
2.8 倍。122 线程时只有 prefill 回退，解码仍优于关闭池的基线。

上面这些 tok/s 只属于 gemma-4-E4B-it-Q8_0 上的通用托管逐算子路径，不代表别的东西。
**DeepSeek V4.1 Flash 根本不走那条路径——与 DeepSeek V4 Flash 一样，它在这个后端上跑
自己的整模型执行器**——100% 纯 C# 的 `DeepSeek4CpuExecutor`，没有 ggml、没有原生库、
也没有 GPU，一次只跑一个序列，没有 KV 前缀复用、也没有按序列 slot——它在一个 5 层 F32
fixture 上由 PyTorch 参照实现 `eng/dsv41-reference.py` 在 `atol=rtol=2e-5` 下把关。
它是正确性与可移植性路径，而非服务路径，而且从未有人测量过 V4.1 完整检查点在这里的吞吐、
加载时间或常驻内存；它的计算宽度来自 `TS_DSV4_THREADS`，而不是 `TS_CPU_THREADS`。
参见上文的 [DeepSeek V4.1 Flash（`deepseek41`）](#deepseek-v41-flashdeepseek41)。

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| 工作线程池宽度 | 8 核及以下取全部核心；8 核以上取一半，且不低于 8 | `TS_CPU_THREADS=N` | — |
| 是否启用工作线程池 | 启用 | `TS_CPU_POOL=0` 回退到旧的 ThreadPool `Parallel.For` 行为，便于在同一个二进制里做 A/B | — |
| 工作线程挂起前的自旋次数 | `4096` | `TS_CPU_SPIN=N` —— 在这个宽度下挂起才是最贵的部分，所以默认自旋次数足够多，使稳态下根本不会挂起 | — |
| 单次托管矩阵乘的任务切分 | 每个工作项 `131072` 字节权重，每个线程最多 `4` 个工作项 | `TS_CPU_TASK_BYTES`、`TS_CPU_TASKS_PER_WORKER` —— 按**工作量**而不是线程数来切 | — |
| DeepSeek V4.1 Flash 的整模型执行器 | 纯 C#（`DeepSeek4CpuExecutor`）——在这个后端上走整模型执行器，而不是通用逐算子路径 | 它的计算宽度由 `TS_DSV4_THREADS=N` 决定（默认取全部处理器），而不是 `TS_CPU_THREADS` | — |
| direct 视频网络（Wan、MiniMax-H3）上的量化权重 | 保持 GGUF 存储类型，直接参与乘法 | `TS_DIRECT_QUANT_WEIGHTS=0` 改回加载时一次性展开成 F32 再做普通 GEMM（旧行为，权重内存为 4 倍）。Wan 在 256x160x5f、单步下，就地路径实测 80.9 秒，展开路径 121.4 秒 | — |

#### Agent Skills

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| Agent Skills | 启用（二进制文件旁的 `skills` 目录，不存在则创建） | `TS_NO_SKILLS=1` 关闭该功能 | `--no-skills` |
| 扫描的技能目录 | `<binDir>/skills` | `TS_SKILLS_DIR`（以路径分隔符分隔的列表） | `--skills-dir <path>`（可重复） |
| 预先启用的技能 | 无 | — | `--skill <name>`（可重复）；按请求则用 `"skills": [...]` |
| 向模型展示未选中的技能 | 启用 | — | `--skills-no-discovery`；按请求则用 `"skills_discovery": false` |
| `skills_run`（运行技能自带脚本） | **关闭** | **`TS_SKILLS_ALLOW_EXEC=1`** | `--skills-allow-exec` |
| 技能脚本操作系统沙箱 | `required`（无法安全隔离时拒绝） | `TS_SKILLS_SANDBOX` | `--skills-sandbox off\|preferred\|required` |
| 技能脚本网络访问 | **关闭** | `TS_SKILLS_ALLOW_NETWORK=1` | `--skills-allow-network` |
| 每轮可取用技能内容（及代码）的次数 | `8`，开启 `--code-exec` 时为 `24`（范围 1-64） | `TS_SKILLS_MAX_ROUNDS` | `--skills-max-rounds N` |
| 打印注册表后退出 | — | — | `--list-skills` |

`TS_NO_SKILLS` 与 `TS_SKILLS_ALLOW_EXEC` 只要取值不是 `0` 就算开启。两个宿主接受
完全相同的参数拼写，且配置文件的键*就是* CLI 参数（`"skills-dir": ["/srv/skills"]`），
因此同一份配置文件可以同时驱动 CLI 与服务端。完整参考见
[Agent Skills in TensorSharp](docs/agent_skills.md)（英文）。

#### 代码执行（`shell` 工具）

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| `shell`、`read_file`、`edit_file`、`write_file` 与 `apply_patch` 工具 | **关闭** | **`TS_CODE_EXEC=1`** | `--code-exec` |
| 安装软件包 | **关闭** | **`TS_CODE_EXEC_ALLOW_INSTALL=1`** | `--code-exec-allow-install` |
| 模型生成命令的不受限主机网络访问 | **关闭** | **`TS_CODE_EXEC_ALLOW_NETWORK=1`** | `--code-exec-allow-network` |
| 一次安装可以指名的包 | 不限（单次安装最多 16 个） | — | `--code-exec-packages <list>` |
| 宿主代办安装使用的软件包索引 | 安装器默认值 | `TS_CODE_EXEC_INSTALL_INDEX` | `--code-exec-install-index <url>` |
| 一次安装可以到达的主机 | `pypi.org`、`files.pythonhosted.org`、`registry.npmjs.org`（传空值 = 不做限制） | `TS_CODE_EXEC_INSTALL_DOMAINS` | `--code-exec-install-domains <list>` |
| 单条命令的时限 | `120` 秒（单次调用最多可要求 10 分钟） | — | `--code-exec-timeout N` |
| 代码轮次采样温度 | 未设置；仍会去掉内置重复惩罚 | — | `--code-exec-temperature 0..2`（负值清除） |
| shell | `bash`，其次 `sh`；Windows 上为 PowerShell | — | `--code-exec-shell <path\|name>` |
| 单条命令保留的输出 | `32768` 字节，从中间截断 | — | `--code-exec-max-output N` |
| 操作系统沙箱 | `required`（无法提供文件系统 / IP 网络隔离时拒绝运行） | — | `--code-exec-unconfined`（CLI 与服务端；Windows 必需） |

`TS_CODE_EXEC`、`TS_CODE_EXEC_ALLOW_INSTALL` 与 `TS_CODE_EXEC_ALLOW_NETWORK`
只要取值不是 `0` 就算开启。模型生成的命令默认无法使用互联网/IP 套接字，但本地
Unix IPC 并非完整隔离边界。联网开关会给予不受限的宿主 IP 网络访问，但不会移除
macOS / Linux 上的写入及主目录读取约束。Linux 还通过 PID 命名空间约束后代进程；
macOS 子进程会继承 Seatbelt，普通进程组也会被清理，但主动脱离进程组的子进程可能
在请求结束后继续运行，每次工具结果都会明确报告这一限制。macOS 仍拒绝常见的
`/private/tmp/com.apple.launchd*` 路径套接字（但保留系统运行时所需的 Mach lookup 与 DNS 必需的精确 mDNSResponder 路径套接字），
并为兼容性保留共享临时目录内的 Unix IPC；Linux 仍隐藏常见的 `/run` 端点，但其宿主
网络命名空间可能暴露抽象套接字以及 `/run` 之外的路径名套接字。装包开关仍是由宿主
代为执行的较窄能力。两个 code-exec 开关都不控制技能脚本；后者使用
`TS_SKILLS_ALLOW_NETWORK` / `--skills-allow-network`。Windows 无法提供文件系统
隔离，因此 CLI 与服务端都仍须传入 `--code-exec-unconfined`。随旧的程序式工具一同
退休的只剩
`--code-exec-languages`：它在启动时会被按名字拒绝，并且没有替代品可指，因为
shell 能够到达 PATH 上的每一个解释器——于是手上还拿着旧脚本的运维者得到的是这条
报错，而不是眼看着一个设置被忽略。

#### 采样默认值（仅服务端）

这些变量用于填充请求体未提供的字段。CLI 参数优先于环境变量；除非服务以
`--sampling-precedence request` 启动，否则通过二者之一配置的参数还会优先于请求体。

| 采样字段 | 环境变量 | CLI 等价参数 |
|---|---|---|
| `temperature` | `TENSORSHARP_TEMPERATURE` | `--temperature` |
| `top_k` | `TENSORSHARP_TOP_K` | `--top-k` |
| `top_p` | `TENSORSHARP_TOP_P` | `--top-p` |
| `min_p` | `TENSORSHARP_MIN_P` | `--min-p` |
| `repeat_penalty` | `TENSORSHARP_REPEAT_PENALTY` | `--repeat-penalty` |
| `presence_penalty` | `TENSORSHARP_PRESENCE_PENALTY` | `--presence-penalty` |
| `frequency_penalty` | `TENSORSHARP_FREQUENCY_PENALTY` | `--frequency-penalty` |
| `seed` | `TENSORSHARP_SEED` | `--seed` |
| 最大 token 数 | `MAX_TOKENS` | `--max-tokens` |
| 停止序列 | —（仅 CLI / 请求体支持） | `--stop`（可重复） |
| 采样优先级 | `TENSORSHARP_SAMPLING_PRECEDENCE` | `--sampling-precedence` |

#### 服务托管与上传（仅服务端）

| 功能 | 默认 | 环境变量 |
|---|---|---|
| ASP.NET Core 监听 | `http://0.0.0.0:5000` | `--port` / `--host` / `--urls`，其次 `PORT` / `HOST`，再次 `ASPNETCORE_URLS` |
| 文本及原生数字 PDF 上传 | 保留全部提取内容；最终渲染的提示词必须能放入已加载模型的上下文 | — |
| 视频帧抽取 | 1 fps（基于时间，不限制） | `VIDEO_SAMPLE_FPS`、`VIDEO_MAX_FRAMES` |
| DiffusionGemma Web UI 去噪 | 48 步，最大 batch 2 | `DIFFUSION_STEPS`、`DIFFUSION_MAX_BATCH` |

#### 日志（服务端 + CLI）

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| 控制台 + 文件日志最低级别 | `Information` | `TENSORSHARP_LOG_LEVEL` | `--log-level` |
| 文件日志输出目录 | `<binDir>/logs` | `TENSORSHARP_LOG_DIR` | `--log-dir` |
| 文件日志开关 | 启用 | `TENSORSHARP_LOG_FILE=0` 关闭 | `--log-file 0\|1` |
| 控制台日志开关 | 启用 | — | `--log-console 0\|1`（仅 CLI） |

#### 原生构建（仅编译期）

下列变量由 `build-linux.sh` / `build-windows.ps1` / `dotnet build` 时自动构建 `TensorSharp.GGML.Native` 的脚本读取，不在运行时生效。

| 功能 | 默认 | 环境变量 | 构建脚本参数 |
|---|---|---|---|
| 在原生构建中启用 GGML CUDA | 根据工具链自动检测 | `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON` | `--cuda` / `--no-cuda` |
| 在原生构建中启用 GGML Vulkan | 根据已安装的 Vulkan 运行时自动检测；未安装 Vulkan SDK / 开发包时自动下载便携工具链（headers、glslc、SPIRV-Headers） | `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON/OFF` | `--vulkan` / `--no-vulkan` |
| 精简 `CMAKE_CUDA_ARCHITECTURES` 列表 | 根据可见 GPU 自动检测 | `TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES` | `--cuda-arch='86-real;89-real'` |
| 原生构建并行度上限 | 使用全部 CPU，并按内存容量限制（`nvcc` 每任务约 3 GB） | `TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL` | — |
| 原生构建使用的 CMake 生成器（Windows） | 有 Ninja 时优先使用，否则用 `Visual Studio NN` | `CMAKE_GENERATOR` | `-G <生成器>` |
| 原生构建使用的 Visual Studio 安装（Windows） | 自动检测，包含被标记为"不完整"的安装 | `TENSORSHARP_VS_INSTALL_DIR` | — |

## 服务端日志

每一轮 chat / generate 请求开始与结束时，服务都会打出一条结构化的
Information 级别日志。只需对日志文件做一次 grep，即可获得紧凑的请求—响应审计
轨迹，无需重放任何流量。

| 事件 ID | 触发位置 | 字段 |
|---|---|---|
| `ChatStarted`（1500） | `chat.start`、`generate.start` 以及各协议的请求横幅 | 采样配置、消息与附件计数、`userInput=`（最近一条用户消息的有界预览），以及 `fullInput=`（本轮全部消息的 JSON 数组，包含附件路径、原始字符数，正文最多 512 个字符）。内联上传文档会替换为省略标记，但保留末尾用户指令；`/api/generate` 同样仅记录有界 prompt 预览。 |
| `ChatCompleted`（1502） | `chat.complete`、`generate.complete` | token 数、KV 缓存复用（`kvReused`、`kvReusePercent`）、TTFT、耗时、吞吐、终止原因，以及完整的模型原始输出（思维链 + 结果） |
| `ChatAborted`（1503） | 客户端中途断开 | 已生成的部分输出、当时的 KV 复用占比 |
| `KvCacheReusePlan`（1510） | 每次前缀复用判断 | `Debug` 级的细粒度分支信息（精确匹配 / 部分复用 / 完整重置） |
| `HttpRequestStarted/Completed`（1100/1101） | 每个 HTTP 请求 | method、path、远端 IP、状态码、耗时；`/api/queue/status` 被降级到 `Debug`，避免 UI 高频轮询淹没每轮日志 |

模型原始输出会保留 `<think>...</think>`、`<|channel|>analysis` 等内联标记，因
此完成日志可以同时看到推理过程与最终用户可见结果。输入正文会刻意限制长度：
无损上传的文档不会再被完整复制到日志中，从而避免额外内存/IO 峰值和意外泄露
文档内容。上传清单与 `contentChars` 仍保留可用的审计元数据；如需逐字节重放，
请另行捕获请求。设置 `TENSORSHARP_LOG_LEVEL=Warning` 可关闭逐轮信息日志。

`fullInput` 字段示例（为便于阅读做了缩进，实际日志为单行）：

```json
[
  {"role":"system","content":"你是一个有帮助的助手。","contentChars":11},
  {"role":"user","content":"世界上最高的山是哪座？","contentChars":11},
  {"role":"assistant","content":"珠穆朗玛峰。","contentChars":6},
  {"role":"user","content":"它有多高？","contentChars":5,"images":["/uploads/mountain.jpg"]}
]
```

同样的 KV 缓存复用统计会通过所有 API 透出：

- **Web UI SSE**（`POST /api/chat`） —— `done` 事件携带 `promptTokens`、`kvReusedTokens`、`kvReusePercent`。
- **Ollama NDJSON**（`POST /api/generate`、`POST /api/chat/ollama`） —— 流式末尾 chunk 与非流式响应均携带 `prompt_cache_hit_tokens`（int）和 `prompt_cache_hit_ratio`（0..1）。
- **OpenAI**（`POST /v1/chat/completions`） —— `usage` 块携带 `prompt_tokens_details.cached_tokens`，与 OpenAI 标准扩展一致，现有 SDK 可直接读取。

Web UI 中每条助手消息下方的统计行也会展示命中率（例如 `187 tokens · 2.1s · 87.2 tok/s · KV 420/512 (82%)`）。

## HTTP API

TensorSharp.Server 暴露三种 API 风格。完整文档及 curl/Python 示例见 [API_EXAMPLES.md](TensorSharp.Server.Host/API_EXAMPLES.md)。

**兼容 Ollama 的 API：**

```bash
# 列出模型
curl http://localhost:5000/api/tags

# 文本生成
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "prompt": "Hello!", "stream": false}'

# 聊天
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "Hi"}], "stream": false}'

# 启用思维链模式的聊天
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "计算 17*23"}], "think": true, "stream": false}'

# 带工具调用的聊天
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "天气怎么样？"}], "tools": [{"function": {"name": "get_weather", "description": "获取当前天气", "parameters": {"properties": {"city": {"type": "string"}}, "required": ["city"]}}}], "stream": false}'

# 带 Agent Skills 的聊天。所有聊天接口都接受 "skills"
#（/v1/chat/completions、/v1/responses、/api/chat 的 Ollama 与 Web UI 两种形态）；
# 可选的 "skills_discovery": false 会把本次请求限制在它点名的技能上。
# 模型的 skills_read 调用在服务端内部就被应答掉了，因此返回的是一条普通回复，
# 而不是一个需要客户端自己去执行的工具调用。
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "把这份对账单里的合计表格提取出来。"}], "skills": ["pdf"], "skills_discovery": false, "stream": false}'

# 列出已注册的技能；GET /v1/skills/{name} 会额外返回 SKILL.md 正文（字段名
# "instructions"）。/api/skills 还会报告加载错误以及是否接受上传；
# POST /api/skills 安装一个 .zip，DELETE 则删除一个。
curl http://localhost:5000/v1/skills
```

**兼容 OpenAI 的 API：**

```bash
# Chat completions
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'

# 结构化输出（OpenAI response_format）
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it-Q8_0.gguf",
    "messages": [{"role": "user", "content": "从“Paris, France”中提取城市与国家。"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "location_extraction",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
            "confidence": {"type": ["string", "null"]}
          },
          "required": ["city", "country", "confidence"],
          "additionalProperties": false
        }
      }
    }
  }'
```

**OpenAI Python SDK：**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="gemma-4-E4B-it-Q8_0.gguf",
    messages=[{"role": "user", "content": "What is 2+3?"}],
    max_tokens=50
)
print(response.choices[0].message.content)
```

**队列状态：**

```bash
curl http://localhost:5000/api/queue/status
# {"busy":false,"pending_requests":0,"total_processed":42}
```
