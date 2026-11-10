# MiniMax-H3

[English](minimax-h3.md) | [中文](minimax-h3_zh-cn.md)

## 许可证与法律免责声明

> **重要：MiniMax-H3 受独立的第三方许可证约束。** 在通过 TensorSharp 下载、运行、部署、修改、再分发或使用 MiniMax-H3 及其输出前，用户必须自行阅读并遵守 [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)、其中的可接受使用政策以及所有适用法律法规。
>
> 该社区许可证明确将**美国、欧盟、英国和韩国**排除在授权地域之外，**不授权**在这些地区进行受许可证约束的使用。位于排除地域的用户，应在进行需要授权的使用前向 MiniMax 取得单独许可。具体条件及后续变更请以官方许可证为准。
>
> TensorSharp 是独立的推理引擎。其对 MiniMax-H3 的技术支持**不代表**授予模型使用许可、豁免限制或获得 MiniMax 的授权或认可。**用户须自行确认使用行为是否合法合规，并自行承担使用或违反许可证产生的一切风险、后果、索赔、处罚、损害及其他责任。在适用法律允许的最大范围内，TensorSharp 及其作者、贡献者和维护者对用户使用 MiniMax-H3、其输出或违反相关许可证及法律的行为不承担任何责任。** 本声明不修改 MiniMax 的许可证，也不排除法律禁止排除的责任。

MiniMax-H3 在同一条打包扩散序列中生成视频与原生 **32 kHz 立体声音频**。
TensorSharp 通过 `MiniMaxH3Model.GenerateVideo`、CLI 和视频 HTTP API 提供该模型。
FL2VA 用于文本和关键帧；Ref2VA 用于文本和参考媒体，两者是独立检查点。

`ggml_metal`、`ggml_cuda`、`ggml_cpu` 使用 TensorSharp 自有的原生图和未修改的
上游 ggml；`mlx`、`cuda`、`cpu` 使用直接张量实现。CUDA 与 MLX 的 DiT、
文本/视觉编码器、音频 VAE 和因果 3-D 视频编码器均在所选设备上执行；
`cpu` 前向使用托管 C# 算术。实现某条路径不表示已验证每种输入或长片段。

```sh
TensorSharp.Cli --config config/minimax-h3-fl2va.json --backend ggml_metal \
  --prompt "A red fox walks through a quiet forest. Forest birds chirp." \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --diffusion-seed 42 --output fox.mp4
```

这会生成 MP4 与立体声 WAV 旁挂文件。小尺寸、少步数请求只适合作为执行冒烟，
不能由文件成功导出推断感知质量。

## 当前验证范围

最新记录为 **2026-10-15**：本地 Apple M5 Pro、48 GiB 统一内存，远端 NVIDIA A40、
46,068 MiB 显存，使用未修改的上游 ggml
`3612c0af93f2cfc8f048295300a61187723b12ae`。
最终 candidate 7 在本地通过 36 项 CPU/MLX 专项测试，在远端通过 29 项专项测试。
Direct MLX、原生 Metal、Direct CUDA 与原生 CUDA 各自的九项组件 oracle 全部通过，
没有跳过。最终完整生成保真对比在这四条 GPU 路径上复现相同的解码 RGB 与立体声音频，
包括 MLX 的图像加标准化音频 Ref2VA，以及原生 CUDA 的首尾帧条件。

| 路径 | 已记录覆盖 | 未建立的结论 |
|---|---|---|
| `ggml_metal` / `mlx` | 组件 oracle、完整生成保真、首尾帧或混合参考的专项检查 | 任意提示词、种子、时长和自然媒体质量 |
| `ggml_cuda` / `cuda` | 组件 oracle、完整生成保真；Direct CUDA 独立 50 层 F32 预测门禁 | 多步独立真值、普遍优于其他引擎 |
| `ggml_cpu` / `cpu` | 修正后的 DiT 组件门禁；托管路径另有延迟加载编/解码门禁 | 修正后的完整配对生成仍待运行 |

组件夹具分别覆盖两层文本片段、两块 DiT 片段、完整视觉塔、视频/音频 VAE 编解码，
以及五帧和十七帧因果视频编码。它们不等同于完整文本编码器或任意形状的验证。
缺失模型、不可用设备和跳过场景均不算通过。

此前文本细化错误地使用 RoPE，采样器缺少音频载体漂移项，终点还缺少
`audioShift/videoShift` 缩放。这三项已修正；更早的 Metal **2.4× / 1.7×**
生成速度和质量对比已被取代，不再作为当前性能结论。

### 独立的完整深度 F32 检查

独立 PyTorch oracle 从同一份 GGUF 解量化，禁用 TF32，使用 IEEE F32 和 math SDPA，
重算全部 50 块。最新 19-token、sigma-one 合成输入相对 L2 为：
音频 `0.0001258906`、视频 `0.0000847537`，均通过 `0.001` 严格门禁。
BF16 权重投影现默认使用 pedantic F32；旧默认分别为 `0.0070538`、`0.0046911`。
这不是自然媒体或多步生成的独立质量证明。

另一个较早的 118-token 配对输入来自未修改 SD CLI 的公开 eval callback，
覆盖 256×160×5、22 个文本 token、sigma 1。Direct CUDA 对独立 F32 的音频/视频
相对 L2 为 `0.00007429` / `0.00031502`。单步物理终点经真实立体声 VAE 解码后，
波形余弦为 `0.999999745`；两边 PCM16 化后最大差为一个 LSB。
两组输入不同，不能混成一条精度曲线，也不能据此声称八步成片相同。

## 性能与精度

最新的前后对比（2026-10-17）使用 FL2VA Q4_K、640×384×22、24 fps、20 Euler 步、
seed 42、CFG 1、flow shift 12。每台机器用 `eng/validation/minimax-h3-ab.py`
交替运行四轮，每次运行都包含加载、生成和编码。每一对的解码 RGB 与音频都完全相同。
同一台机器上两个冻结二进制由同一个未修改的 ggml 版本构建：M5 Pro 为 a05c9922，
A40 为 03f0267b（更新的上游 ggml 已无法构建 TensorSharp 的 DSV4 CUDA 内核）。
文件缓存未受控。

| 设备 / 后端 | 基线中位数 | 候选中位数 | 结果 |
|---|---:|---:|---|
| M5 Pro `ggml_metal` | 117.920 秒 | 105.697 秒 | RGB/音频精确相同，4 对；延迟降低 10.4% |
| A40 `ggml_cuda` | 69.363 秒 | 58.864 秒 | RGB/音频精确相同，4 对；延迟降低 15.1% |

结果来自两项精确改动：

- **RoPE 重写。** DiT、文本编码器、视觉塔和视频 VAE 中的 RoPE 直接从投影读取半交换的
  head，再乘以每个图只构建一次、已折入符号的正弦表；(−a)·b 与 a·(−b) 舍入完全相同。
  因此一次旋转只需 5 遍逐元素运算而不是 10 遍，q、k、v 也保持为融合投影的视图。
  Metal 每步中位数 5.44 → 4.85 秒，A40 每步中位数 1.39 → 1.31 秒。
- **`ggml_cuda` 预读。** 文本编码器与去噪器文件由 16 路并发 8 MiB 读取提前读入
  （`TS_VIDEO_READAHEAD`、`TS_H3_PREFAULT_THREADS`）。上传原先按缺页逐页预热这些文件，
  在 A40 pod 的网络卷上约 1 GB/s。有了预读，编码器主干中位数 13.3 → 6.4 秒，
  首个去噪步中位数 11.5 → 7.6 秒。

下面 2026-10-15 的前后对比使用相同配方，交替两轮，ggml 版本见上文。

| 设备 / 后端 | 基线中位数 | 候选中位数 | 结果 |
|---|---:|---:|---|
| M5 Pro `ggml_metal` | 151.246 秒 | 143.622 秒 | RGB/音频精确相同，延迟降低 5.0% |
| M5 Pro `mlx` | 143.264 秒 | 125.426 秒 | 精确相同，延迟降低 12.5% |
| A40 `ggml_cuda` | 68.029 秒 | 67.174 秒 | 精确相同；1.3% 不构成有意义提速 |
| A40 `cuda`，同精度 | 220.456 秒 | 178.667 秒 | 精确相同，延迟降低 19.0% |

Direct CUDA 对比给两个版本都显式启用 pedantic BF16 权重投影，避免把精度变化
计作速度提升。最终构建另做的保真对比不算新的一轮速度基准。
256×160×5、八步、图像加标准化音频的 Ref2VA 两轮对比为
82.037 → 43.630 秒，RGB/音频精确相同；合成短片不证明自然参考质量或长片性能。

这些改动复用连续 token 行视图与精确 RoPE 表，融合保持算术顺序的 MLX
调制、门控残差和 SwiGLU，并缓存原生 CUDA 图存储与权重绑定。
MLX 视频 VAE 的 F32 权重直接上传到自有设备数组，避免保留 9.02 GiB 重复主机权重。
固定潜变量检查中，采样到的暖进程 RSS 中位数从 23.048 降至 14.024 GiB，输出字节相同；
这是进程驻留采样，不能当作统一内存或显存峰值。

CUDA 量化投影默认以最多 64 MiB 转换暂存区运行 pedantic F32 GEMM，保留 F32
激活与输出范围。`TS_H3_CUDA_F32_BANDS=0` 选择旧 F32 内核；`1` / `tf32`
显式选择近似 TF32。BF16 稠密权重投影也默认 pedantic F32，
`TS_H3_CUDA_BF16_PEDANTIC=0` 才恢复历史近似。

实验性批量注意力通过合成 full50 检查，却未通过完整生成 RGB/音频保真门禁，
因此模型入口已移除；探针拒绝过时的 `TS_H3_CUDA_BATCHED_ATTENTION` 覆盖。
独立后端 helper 和微基准只保留作研究工具。

较早与干净 stable-diffusion.cpp 的比较使用其
`d637cfb935feb2254898a8f1524046c958f3df80`，以及它固定的 ggml
`d25b121ce1d6ae0c7e62ad080460d9fc1e46dd93`。
640×384×22、八步原生 Metal 为 70.019 对 102.262 秒；此前原生 CUDA
20 步两轮为 71.899 / 64.904 秒，对照为 145.020 / 50.238 秒。
缓存波动很大，暖对照仍更快，因此没有普遍 CUDA 整进程持平结论。
当前 A/B 和这些历史跨引擎数据不能合并。完整旧数据与限制见[英文卡片](minimax-h3.md)。

### ggml_cuda 上的长片段

480x864、158 帧的 FL2VA 片段（八步 Turbo 插件）打包为 20,511 个 token。在测量所用的 CUDA GPU 上，
仅用 ggml 时一个去噪步耗时 67.7 s：53% 在 ggml-cuda 的 MMA F16
flash attention（此形状下 16.6 TFLOP/s，cuDNN SDPA 为 80-83 TFLOP/s），22% 在 MMQ Q4_K 投影
（52 TOPS，cuBLAS BF16 GEMM 为 89 TFLOP/s），22% 在 F32 逐元素运算。ggml 保持不变；
`graph_compute_offload` 在 ggml 之外执行 H3 自身图里的选定节点，结果写入 ggml 分配器给出的缓冲区：

- **注意力**改用 cuDNN 融合 SDPA（后端描述符 `CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR`，
  cuDNN 9.13+，与 VAE 卷积一样在运行时加载），F16 输入、F32 累加（`tsg_cuda_cudnn_sdpa.cu`）。
- **宽 Q4_K 投影**每次调用展开为 BF16，再做一次 F32 累加的 cuBLAS GEMM（`tsg_cuda_q4k_gemm.cu`，
  需要 sm_80+）。BF16 保留 F32 的指数范围，因此这些矩阵乘去掉了 q8_1 溢出保护的两次缩放。
  LoRA 更新合并进展开后的权重（W + up*down，只舍入一次 BF16），不再逐投影做 shrink、expand 和整张量加法。
- **融合阶段**（`tsg_cuda_h3_fused.cu`）以 `GGML_OP_CUSTOM` 节点出现在图中：注意力前的 qkv 偏置、
  q/k RMSNorm、RoPE 与 F16 收窄（DiT 与视频 VAE），按段的 AdaLN 调制，以及带偏置与 LoRA 项的门控残差。

同一片段（同一关键帧，种子 1），新启动服务器上的完整请求：仅 ggml 678 s（步 67.7 s）→ cuDNN 注意力
484 s → cuBLAS BF16 投影 431 s → 去掉保护缩放并以原始 RGB 管道写 MP4 376 s → 融合阶段 337 s →
合并 LoRA 303 s（步 22.8 s）→ 融合视频 VAE 注意力 293 s。同一服务器的第二个片段为 294 s：去噪器常驻，
但 17 GB 文本编码器每次请求重新加载，耗时 50 s（首个请求 28 s）。ComfyUI 的
W6A8 H3 生成同一片段为 163 s（模型已加载）。输出与仅 ggml 路径并非逐位相同：解码 RGB 的 PSNR 为
30.7-31.8 dB，主要来自注意力改为 F32 累加（仅第一项改动即为 31.8 dB）；相邻改动之间为 38-47 dB。
MP4 写出改为向 ffmpeg 管道传原始 RGB24，而不是逐帧经 ImageMagick 写 PNG（像素相同；此处 ~33 s 降到 ~1 s）。

第二轮改动收紧了去噪器之外的开销：去噪最后一步后，`ggml_cuda` 上的流水线释放 DiT 图增长出的复用计算缓冲区
以及 CUDA 注意力和投影的临时缓冲区，不再让它们一直占用到下一个请求加载 17 GB 文本编码器时；在驱动以主机内存
承接超出本地显存分配的设备上，它们占用的正是编码器映射文件所需的内存，第二个请求的文本条件阶段由 50 s 降到 33 s。
fc2 直接读取 fc1 的 `[gate | value]` 输出，在收窄为 BF16 时完成 SwiGLU，GLU 节点及其 F32 结果不再进入图。
视频 VAE 解码器的投影改用 cuBLAS F16（与 ggml-cuda 自身 cuBLAS 路径的 F16 输入相同），qkv 偏置并入注意力前处理
（VAE 按头交错排列 q、k、v），w1 偏置与 SwiGLU 并入 w2 的输入收窄，out 与 w2 的偏置连同层缩放并入一个残差内核；
分块循环中的 CPU 步骤（解包与混合）并行执行。第二个请求的解码由 44.4 s 降到 32.9 s；解码 RGB 与上一版相差
46-47 dB，并行分块逐位相同。同一服务器上连续测量：首个请求 296.8 s → 277.4 s，第二个请求 283.8 s → 251.6 s。

第三轮改动（同日稍后在同一片段、同一服务器上测量）不改变默认输出，与第二轮逐字节相同。权重上传改为直接读取文件
（`ggml_ops_direct_read.cpp`，8 个线程做无缓冲、扇区对齐的读取，写入少量页锁定缓冲区并随到随传），不经过页缓存，
因此不会挤出去噪器的页面，流水线对编码器和去噪器文件的预读也随之跳过；文本编码器上传由经映射的 0.7 GB/s 提高到
2.8 GB/s，文本条件阶段由 28-33 s 降到 8-10 s（目前在 Windows 的 `ggml_cuda` 上实现，其他平台不变）。在 WDDM 下
`cudaMemGetInfo` 可能把驱动可用来承接分配的共享系统内存也算进去，`TSGgml_DeviceMemoryInfo` 现在报告适配器自身的
专用显存减去已用部分。文本编码器运行前，只有剩余显存小于编码器文件时才释放视频与音频 VAE。门控残差改为每个 token
一个线程块，分段查找展开，去噪单步由 22.4 s 降到 20.8 s。整次请求：首个 223.4 s，第二个 214.3 s。
可选的 FP8 投影（`TS_H3_FP8_MATMUL`，`1` 为 DiT，`vae` 为视频 VAE，`all` 为两者）经 cuBLASLt 计算 E4M3 操作数，
每个操作数一个 2 的幂缩放，会改变输出：DiT 投影乘积比自身舍入误差高 27-37 dB（平均 31 dB）。`TS_H3_FP8_MATMUL=1`
时单步 16.4 s，整次请求 188.8 s / 181.0 s；`vae` 与 `all` 尚未做端到端测量。

### Metal 上的长序列注意力

FL2VA 上 1280×768×107 的首尾帧片段有 34,991 个打包 token，此时每步的耗时主要在注意力。
在 M5 Pro、`ggml_metal`、未修改的 ggml `a05c9922613c6a2230dbb1d0de679519ecafb4f0`、
20 步、seed 42 下，整次运行从 11,553.8 秒降到 5,438.5 秒（2.12×），每步约 570 秒降到
263 秒；单次 flash attention 调用从 8.05 秒降到 3.24 秒：

- ggml-metal 只在 key 数为 64 的倍数时才把长 flash attention 交给 Metal 4 tensor-API
  内核。TensorSharp 用零行把 key 补到下一个 64 的倍数并加以遮罩，注意力结果不变（被遮罩
  的 key 对 softmax 和输出没有贡献）。只在 tensor API 可用时补齐；`TS_METAL_FLASH_KV_PAD=0`
  关闭补齐，供 A/B 对比。
- 遮罩每个 query 需要一行，在这一长度下为 2.45 GB；各行相同，因此 query 分块运行，共用
  一份约 4 MB 的遮罩。

随后视频 VAE 解码耗时 146–148 秒。在 `ggml_metal` 上，管线现在会在解码前释放去噪器的
权重和复用的计算缓冲区（`TS_H3_KEEP_RESIDENT=1` 保留权重）；此前 38.3 GB 的 Metal 预算
只剩 3.3 GB 空闲，解码可能以 `kIOGPUCommandBufferCallbackErrorOutOfMemory` 失败。作为
对照，MLX 融合 SDPA 在同一形状下每次调用 F32 为 4.08 秒、F16 为 1.68 秒，未被采用。
以上是同一台机器上的单次运行，按 CLI 报告的整段耗时计；改动前后的日志分别位于
`artifacts/minimax-h3/fl2v-face-investigation/` 与 `artifacts/minimax-h3/speed/hr/`
（本地验证证据，未提交）。

## 需要的文件

| 文件 | 大小 | 用途 / 来源 |
|---|---|---|
| `minimax_h3_fl2va_pruned-Q4_K.gguf` | 10.64 GiB | 文本 + 关键帧，[unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) |
| `minimax_h3_ref2va_pruned-Q4_K.gguf` | 10.60 GiB | 文本 + 参考，同仓库，选择其中一个去噪器 |
| `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | 16.97 GiB | 共用 Qwen3-VL-32B 文本/视觉编码器，同仓库 |
| `minimax_h3_video_vae_fp16.safetensors` | 5.21 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) 视频 VAE |
| `minimax_h3_audio_vae_fp32.safetensors` | 0.61 GB | 同仓库，音频 VAE；静音输出可省略 |

一个去噪器和共享组件约 35.5 GB；单个配置会下载两个去噪器与共享组件，约 46.1 GB，
视频聊天和 Studio 可在不重启的情况下切换首尾帧与参考模式。
伴随文件可放在去噪器旁，或用 `--video-text-encoder`、`--video-vae`、`--audio-vae`
指定。`config/minimax-h3-fl2va.json` 会自动下载完整网络与分词器，默认 `ggml_cuda`、22 帧、24 fps，
命令行后端优先；配置把尺寸交给模型（文生视频与参考模式 640×384，关键帧按[配方画布](#尺寸与设置)）。
服务端步数使用 `--video-steps`，CLI 使用 `--diffusion-steps`；配置不写步数与 CFG，
由模型采用默认 20 步和 CFG 1。

**配置会自动下载分词器。** 文本 GGUF 不带分词器；配置会下载并校验
`vocab.json`、`merges.txt` 和 `tokenizer_config.json`。手动安装时可将这三份文件放在编码器旁，
或设置 `TS_VIDEO_TOKENIZER`。
将 [MiniMaxAI/MiniMax-H3 的 processor 文件](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/processor)
包含这些分词器文件。TensorAgent 目录也会自动下载它们。
视觉输入需要 `tokenizer_config.json` 中的特殊标记；缺失时图片或片段参考在视觉塔前
就会被拒绝，避免默默忽略输入。纯文本或纯音频参考不需要这些视觉标记。
已发布 GGUF 没有元数据，TensorSharp 按张量表识别；声明时也接受
`minimax-h3` / `minimax_h3`。重新命名请保留 `fl2va` / `ref2va` 分区信息。

## 条件模式

| 输入 | 模式 | 检查点 |
|---|---|---|
| 文本 | `t2v` | FL2VA 或 Ref2VA |
| 首帧图像 | `i2v` | FL2VA |
| 尾帧（可带首帧） | `fl2v` | FL2VA |
| 参考图、片段、音轨 | `ref` | Ref2VA |

单个配置会下载两个分区，因此同一个加载模型可提供所有表中模式。

`--video-mode` 可显式选择，否则按输入推断。关键帧不能与命名参考混用；
分区不符时会拒绝。关键帧同时编码为视频潜变量和 Qwen3-VL 特征，按输出比例中心裁剪；
未指定尺寸时使用图像比例。Ref2VA 保持参考自身比例，不以参考改变输出画布。
`--ref-image` 最多九份，每份增加同一条注意力序列的 token 数、内存和耗时。

```sh
TensorSharp.Cli --config config/minimax-h3-fl2va.json --backend ggml_metal \
  --image start.png --end-image end.png --video-mode fl2v \
  --prompt "A slow cinematic push-in" --diffusion-seed 42 --output morph.mp4

TensorSharp.Cli --config config/minimax-h3-fl2va.json --backend ggml_metal \
  --ref-image person.jpg --ref-audio song.wav --video-mode ref \
  --prompt "She performs on a small club stage" --diffusion-seed 42 --output stage.mp4
```

`--ref-video` 接受文件或帧目录，`--ref-video-audio` 按索引配对其音轨；
独立音轨用 `--ref-audio`。视频按 24 fps 取样、对齐 `17k+5` 帧且不超过输出长度。
独立音轨重采样为 32 kHz 立体声，按输出时长裁剪或补零，再向下对齐 800-sample VAE hop。
片段配对音轨按片段自身时长处理。被检查的 SD 版本保留完整独立音轨，因此相同文件
也可能得到不同参考上下文。配对基准必须明确标准化，不能把上下文差异当成算术错误。
已有合成首尾帧及标准化图像加音频测试；自然参考、视频片段及广泛提示词质量仍未建立。

## 尺寸与设置

- 宽高向上对齐 32；文生视频与具名的 Ref2VA 参考图默认 640×384；在 Ref2VA 上，普通
  `--image`（Web UI 附加图片的方式）按该图宽高比保持这一面积。
- i2v 与 fl2v 保留关键帧的构图，因此未指定尺寸时，画布按 MiniMax 发布的配方由第一个
  关键帧（只有末帧时用末帧）决定：保持其宽高比、短边 768 像素、面积不超过 768×1344，
  每边取最接近的 32 的倍数。3:2 照片为 1152×768，2:3 竖图为 768×1152，4:3 为
  1024×768，16:9 为 1344×768，与 vllm-omni 和 sglang 一致；超出 1:4 至 4:1 时宽高比被
  夹紧，多余部分居中裁掉。视频 VAE 在每个方向下采样 16 倍：全身照放进 640×384 后人脸只有
  约 60 像素高，仅编码再解码就已经糊掉，与去噪器无关。在这类照片上测得的整帧 VAE 往返：
  640×384 为 27.6 dB，3:2 首帧给出的 1152×768 画布上为 29.5 dB，人脸（目测比较）得以保留。
  配方画布每帧的视频 token 约为 640×384 的 2.4 倍（方形）到 4.2 倍（16:9），3:2 为
  3.6 倍，耗时也会成倍增加。
- 显式给出 `--width` 与 `--height` 仍然优先，并对所有模式固定该尺寸。被居中裁剪的关键帧
  会给出保留了多少的警告；缩得比其自身配方画布更小的关键帧会给出警告并写明该尺寸。
  TensorAgent 应用在其 32 GB 档位测得配方尺寸的峰值内存之前，仍使用 640×384 面积的
  紧凑画布。
- 帧数向上对齐 5、22、39、56……（`17k+5`），默认 22。
- 固定 24 fps、CFG 1、自有 Euler flow schedule，默认 20 步；其他 CFG 会拒绝。
  [Turbo 插件](#turbo-插件)会以其蒸馏时的 8 或 4 步取代该调度。
- Wan 的负面提示词、采样器与 guidance-cache 选项不改变 H3。
- `--diffusion-seed` 决定视频噪声；`--seed` 只控制文本采样。
- 视频 VAE 按 256 px 空间分块、按时间分段；长片的有限性与感知质量仍需单独验证。

## Turbo 插件

LightX2V 的 Turbo LoRA 把 FL2VA 检查点蒸馏为 8 或 4 次 Transformer 前向，取代基础的 20 步。
TensorSharp 在 [`config/lora/`](../../config/lora/) 中提供两份插件配置，权重在首次使用时下载：

```sh
TensorSharp.Cli --config config/minimax-h3-fl2va.json \
  --lora config/lora/minimax-h3-fl2v-turbo-8step.json \
  --prompt "A red fox walks through a quiet forest. Forest birds chirp." --output fox.mp4

TensorSharp.Server.Host --config config/minimax-h3-fl2va.json \
  --lora config/lora/minimax-h3-fl2v-turbo-8step.json
```

| 配置 | 插件与权重 | 步数 | 配方来源 |
|---|---|---|---|
| `minimax-h3-fl2v-turbo-8step.json` | LightX2V Turbo 8-step（FL2V v1.0，768p），`minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors` | 8 | ModelTC/Minimax-H3-Turbo README："8 NFE, video shift 6, audio shift 3" |
| `minimax-h3-fl2v-turbo-4step.json` | LightX2V Turbo 4-step（FL2V v1.2，768p），`minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors` | 4 | **推断所得。** README 没有 v1.2 一行，文件也未记录 shift，因此沿用其 4 步 v1.0 768p 一行的 linspace 网格与 shift 6/3；在 shift A/B 或发布方给出答案之前一直标注为推断 |

两份权重来自 [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/a29e57d195f80ead07c89f89e55ecf2911e893c5)
的修订版 `a29e57d195f80ead07c89f89e55ecf2911e893c5`，各 1,383,677,808 字节，校验 SHA-256
（数值见 [MODEL_DOWNLOADS_zh-cn.md](../../MODEL_DOWNLOADS_zh-cn.md)），Apache-2.0 许可，
保存到模型根目录下的 `minimax-h3/loras/`。

### 采样契约

插件在端点包含的 linspace 网格 q_i = (N − i)/N 上做 N 步 Euler，视频 shift 6、音频 shift 3，
末尾为 0，CFG 1，强度 1。它与基础检查点 shift 12 的 stable-diffusion.cpp 调度不是同一个网格，
因此基础模型减少步数并不等于蒸馏调度。

| 步数 | 视频 sigma（shift 6） | 音频 sigma（shift 3） |
|---|---|---|
| 8 | 1, 0.97674, 0.94737, 0.90909, 0.85714, 0.78261, 0.66667, 0.46154, 0 | 1, 0.95455, 0.9, 0.83333, 0.75, 0.64286, 0.5, 0.3, 0 |
| 4 | 1, 0.94737, 0.85714, 0.66667, 0 | 1, 0.9, 0.75, 0.5, 0 |

每段片段都会记录插件、配方与两组 sigma（`[lora] sampling recipe (...)`）。

### 锁定的设置与拒绝

配方是契约而不是默认值。步数、视频 flow shift、采样器与 CFG 可以不设（0）或与插件取值相同；
其他任何值都会被拒绝并给出插件名称，绝不会被静默覆盖。

| 位置 | 检查内容 | 冲突时 |
|---|---|---|
| CLI | `--diffusion-steps`、`--flow-shift`、`--sampler`、`--cfg` | 输出一行 `Configuration error:`，退出码 1，不下载任何文件 |
| 服务端启动 | `--video-steps` | 同上，不下载任何文件 |
| 任一宿主的 `--config` | 合并后的选项，在配置文件自身的文件下载之前 | 同上；被拒绝的运行什么都不下载，包括约 35.5 GB 的整套模型 |
| 请求 | `"steps"`、`"flowShift"`、`"sampler"`（任何 H3 模型都拒绝不等于 1 的 `"cfg"`） | `/api/video-generate` 与 `/v1/videos/generations` 返回 400；`/api/video-generate/stream` 返回 `{done, error}` 事件 |
| Video Studio | 按另一配方规划的镜头 | 409 `sampling_recipe_conflict`，不创建任务（见 [Video Studio](../video-studio.md#sampling-recipes)，英文） |

命令行未指定插件时，两个宿主对通过 `TS_LORAS` 继承的插件做同样的检查。例如 8 步插件配
`--diffusion-steps 20` 会输出：

```text
Configuration error: --diffusion-steps 20 contradicts minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors: LightX2V Turbo 8-step (FL2V v1.0, 768p) samples exactly 8 steps on its own schedule. Drop --diffusion-steps or pass 8.
```

以下情况插件本身会被拒绝并说明原因：用于 Ref2VA 检查点（按文件名中的 `ref2va` 识别；不支持
Ref2VA Turbo）；以裸 `.safetensors` 传入（H3 只接受声明分区与契约的插件配置）；用于其他模型，
或把 Qwen-Image-2.1 插件用于 H3；同时使用两个 H3 插件；`--lora-scale` 为负；以
`--lora <权重> --lora-config <插件配置>` 传入而权重文件并非该配置所指的文件。

### 应用方式

- **不合并**：`y = W x + B (A x)` 叠加在量化 GGUF 权重之上，强度 × alpha / rank 折入 B，
  与发布方推理脚本经 PEFT 的用法一致（其 `--fuse-lora` 默认关闭）。两个文件都声明 rank 128
  时 alpha 为 8，即缩放 0.0625；若加载器假定 alpha = rank，会把更新放大 16 倍。
- **312 项**（624 个张量）：50 个 DiT 块与 2 个 token-refiner 块中的 q、k、v、out、fc1、fc2。
  q/k/v 更新位于 q/k RMSNorm 与 RoPE 之前，fc1 位于 SwiGLU 之前，out 与 fc2 位于各自投影之后。
  Diffusers 文件按 [value; gate] 存放 fc1，两半会交换为检查点的 [gate | value] 顺序。
- **内存。** 因子在 `ggml_metal` 与 `ggml_cuda` 上为 F16，1.38 GB（数值超出 F16 范围的组保持
  F32）；在 `ggml_cpu` 上为 F32，2.77 GB。直接 `cpu`、`mlx` 与 `cuda` 后端在首次使用时逐块构建
  精确的 F32 因子。fc1 有更新时，原生图把 fc1 基础权重拆为 gate 与 value 两半，更新不会新增
  完整尺寸的激活。
- **强度。** `--lora-scale`（默认取配置中的 1.0）乘到更新上，必须 ≥ 0；不等于 1 时日志会说明
  该调度是在强度 1 下蒸馏的。强度 0 保留所有项而上投影因子为零，因此每次前向都与基础检查点
  在同一输入上的结果相同，而片段仍按插件的调度采样，可作恒等探针。
- **画布。** 默认值不变：文生视频仍为 640×384，关键帧使用[配方画布](#尺寸与设置)。适配器在
  768p 下训练，短边不是 768 的片段会记录一行说明，照常运行。
- **后端。** 原生图在 `ggml_metal`、`ggml_cuda` 与 `ggml_cpu` 上应用适配器；直接 `cpu`、`mlx`
  与 `cuda` 后端自行应用。哪些已经运行见下文。

服务端启动时的 `--lora` 作用于每个片段，请求不能选择插件；请求不要带 `steps`，与 Web UI 一样
（见 [API 示例](../../TensorSharp.Server.Host/API_EXAMPLES_zh-cn.md#minimax-h3--视频与原生-32-khz-立体声音频)）。
[Video Studio](../video-studio.md#sampling-recipes)（英文）显示配方、固定步数并把每个任务绑定到它。
[TensorAgent 应用](../../TensorAgent/README.md)在关键帧条目上把两个插件作为提速插件提供，在聊天与
Studio 中按片段应用，且从不发送步数、flow shift 或采样器。

### Turbo 验证状态

本次改动于 2026-10-17 在 Apple M5 Pro 上、针对未修改的 ggml
`a05c9922613c6a2230dbb1d0de679519ecafb4f0` 记录：

- **契约与宿主。** 托管 CPU 测试覆盖插件配置、配方与 sigma 网格、上述在 CLI、服务端、
  `--config`、请求路由和 Studio 中的全部拒绝，以及进程测试：每次拒绝都只输出一行
  `Configuration error:`、退出码 1、不开始任何下载。
- **原生辅助函数**（`eng/validation/minimax-h3-lora.cpp`，ctest `minimax-h3-lora`）在 `cpu` 与
  `metal` 上：堆叠的 q/k/v shrink、折叠、拆分 fc1 与零上投影恒等均逐位相同，rank 0 的适配器
  构建出与原来相同的图。
- **两块合成对比**（真实代码、随机小权重）：带插件的直接 CPU 与合并了更新的 GGUF 在五种情况下
  相对误差 ≤ 1.3e-7，而不交换 fc1 两半的合并相差 0.17。原生 `ggml_cpu` 对直接 CPU：显式 softmax
  ≤ 3.0e-7，flash attention 的 F16 K/V 下 ≤ 2.05e-4。`ggml_metal`：≤ 5.0e-4，更新余弦
  ≥ 0.9999993。强度 0 在 `ggml_cpu`、`ggml_metal` 与直接 CPU 上与不加插件逐位相同。夹具的
  head 维度为 8，Metal 不会走其 flash 内核，因此该内核尚未带适配器运行。
- **真实适配器文件**（CPU）：两个文件都加载为 624 个张量、312 项，rank 128，缩放 0.0625，
  Ref2VA 会拒绝它们。第 10 块的行与发布方 ComfyUI 格式孪生文件逐位相同
  （`InferenceWeb.Tests/Fixtures/MiniMaxH3/turbo-comfy-reference-rows.json`），以真实数据而非
  TensorSharp 自身锁定了 fc1 两半交换与堆叠 q/k/v 布局。打包耗时：F32（2.77 GB）0.7 秒，
  F16（1.38 GB）0.5 秒。

在 `ggml_metal` 上用真实权重实测（每项各运行一次，属于样本而非基准）：

- **第 0 步恒等**（384×256×22，种子 42）。Turbo-8 在 `--lora-scale 0` 下的首步视频速度与基础 H3
  逐位相同；两次运行都从 σ = 1 开始。音频速度按设计不同：插件把视频 shift 从 12 改为 6，
  这会重新缩放音频载体。强度为 1 时，更新使视频速度与基础的余弦相似度为 0.943。
- **墙钟时间。** FL2VA，首尾两张关键帧，1280×768×107（34,991 个打包 token），种子 42：

  | 运行 | DiT 步数 | 每步秒数 | VAE 解码 | 总计 |
  |---|---:|---:|---:|---:|
  | 基础，注意力修复之前 | 20 | ~570 | — | 11,553.8 秒 |
  | 基础 | 20 | 262.8 | 148.0 秒 | 5,438.5 秒 |
  | Turbo-8 | 8 | 264.7 | 146.6 秒 | 2,298.7 秒 |
  | Turbo-4 | 4 | 264.7 | 146.2 秒 | 1,238.6 秒 |

  非合并适配器每步开销不到 1%。带适配器的峰值内存未测。基础的一步在 38.3 GB 的 Metal 预算中
  留出 3.3 GB 空闲，适配器另加 1.38 GB 的 F16 因子。
- **768p 关键帧画布上的质量**（同一批运行，按全尺寸帧裁剪目测，并计算整帧统计）：
  - 两个 Turbo 片段的人脸在片段中段和固定末帧附近都保持清晰、不变形，提示词要求的动作比基础更清楚。
  - 与基础 20 步相比的偏离符合蒸馏模型的预期：对基础的整帧平均 PSNR，Turbo-8 为 17.5 dB，
    Turbo-4 为 19.8 dB。作为参照，注意力修复前后的两次基础运行彼此相差 41.6 dB。
  - 差异主要来自两张差异很大的关键帧之间的过渡：基础在第 13 帧左右溶接，两个 Turbo 变体都先保持首帧再硬切。
  - Turbo 饱和度略高（平均 0.20 对 0.17），运动更多（平均帧间变化 7.3 与 5.1 对 2.0），音轨更响
    （RMS 0.21 与 0.25 对 0.15）。
- **4 步 shift A/B**（视频 shift 6 对 12，同样的关键帧与种子）。两者都干净且彼此接近：对基础 20 步
  19.8 对 19.3 dB，饱和度 0.197 对 0.199，运动 5.1 对 5.3。一个种子无法分出优劣，因此 4 步配方
  仍为推断，保持已发布 768p 配置所用的 shift 6。

尚未运行，因此未经验证：

- 两个插件在 640×384 下的文生视频，以及多个种子。
- `ggml_metal` 对直接 `mlx` 的真实权重更新一致性。
- 带适配器的 `ggml_cuda`、直接 `cuda` 与 `mlx`，TensorAgent 应用，以及 Diffusers + PEFT 参考
  速度（本地没有可运行的参考）。

日志与报告位于被忽略的 `artifacts/minimax-h3/turbo-*` 与 `artifacts/minimax-h3/lora-native/`
（本地验证证据，未提交）。

## HTTP API

`POST /api/video-generate`、`POST /api/video-generate/stream`（SSE）与
`POST /v1/videos/generations` 共用解析器。

```sh
curl -s localhost:5000/api/video-generate -H 'content-type: application/json' -d '{
  "prompt": "A red fox walks through a quiet forest. Forest birds chirp.",
  "width": 640, "height": 384, "frames": 22, "steps": 20, "cfg": 1.0,
  "imagePath": "card.jpeg", "videoMode": "i2v", "seed": 42
}'
```

主要字段有 `prompt`、`width`、`height`、`frames`、`steps`、`cfg`、`fps`、`seed`、
`flowShift`、`imagePath` 或 base64 `image`、`videoMode`、`generateAudio`、`endImage`、
`referenceImages`、`referenceVideos`、`referenceAudios`、`referenceVideoAudios`。
优先使用 camelCase；只有模式、音频开关、尾图和四个参考列表也接受 snake_case，
`image_path` / `flow_shift` 会被忽略。参考视频与音频按索引配对。
图片路径必须来自 `/api/upload`；上传目录外路径会拒绝。模型模式错误返回带具体原因的 400。

响应包含 `url`、`audioUrl`、`audioMuxed`、尺寸、帧数、fps、种子、codec、耗时等。
OpenAI 形式返回 `audio_url`，没有 mux 标记。`GET /api/models` 的 `video` 对象声明
音频、首帧、尾帧及参考能力，Web UI 据此显示输入控件。

### 音频输出

生成音轨同时保存为 WAV。桌面提供器有 FFmpeg 时会把音轨混入 MP4；
OpenCV/OS 回退只写视频，并返回 `audioMuxed: false`。
Apple AVAssetWriter（TensorAgent Mac 应用）写入 32 kHz、双声道、128 kbit/s AAC，
并返回 `audioMuxed: true`。编码格式差异也应记录在音频对比中。

```sh
ffmpeg -i fox.mp4 -i fox.wav -c:v copy -c:a aac fox_with_audio.mp4
```

`--no-audio` / `"generateAudio": false` 跳过音频解码，减少音频 VAE 时间与内存。

## 在 TensorAgent Mac 应用里

32 GB 档位提供一个 **MiniMax-H3** 条目（`minimax-h3-fl2va-q4k`），自动下载带哈希校验的
FL2VA 与 Ref2VA 去噪器、共享网络和三份分词器文件。在 Video Studio 中可选择首尾帧模式或参考模式；
切换时只更换活动去噪器，已加载的模型和共享组件保持就绪。
目录可用性不等于已完成每个模型/设备的验证；当前 Windows 与 iOS 的媒体生成未验证。
完整应用与构建说明见 [TensorAgent](../../TensorAgent/README.md)。

## 架构与内存

主要实现位于 [`MiniMaxH3Model`](../../TensorSharp.Models/Models/MiniMaxH3/MiniMaxH3Model.cs)、
[`MiniMaxH3Pipeline`](../../TensorSharp.Models/Models/MiniMaxH3/MiniMaxH3Pipeline.cs)、
[`MiniMaxH3DirectDiT`](../../TensorSharp.Models/Models/MiniMaxH3/MiniMaxH3DirectDiT.cs)
和 TensorSharp 自有的原生图
[`ggml_ops_minimax_h3.cpp`](../../TensorSharp.GGML.Native/ggml_ops_minimax_h3.cpp)。

去噪器为约 193 亿参数、50 块单流 Transformer，hidden 5376，56×128 维注意力，
SwiGLU；文本、条件、目标音频和视频共同做双向注意力。没有 cross-attention。
视频 patch 为 `1×2×2`，24 通道展平为 96 值，次序为 `c*4 + ky*2 + kx`。
AdaLN 来自学习的 `[8,1025]` 曲线表，三模态分别取调制参数。
主干使用连续浮点三轴 RoPE，音频和视频共享时间轴；两个文本 refiner 块不使用 RoPE。

视频 shift 12、音频 shift 3，采样器保存
`carrier = (sigma_video / sigma_audio) * physical_audio`；送入 DiT 前还原物理音频，
积分时包含漂移项，终点乘 `audioShift/videoShift`，再做 VAE 通道反归一化。
文本编码器是 Qwen3-VL-32B 前 50 层，去掉 final norm，直接读取原始提示词，不套聊天模板。
视频 VAE 为 16× 空间/4× 时间压缩，36 层 Transformer 解码器；256 px 分块是坐标正确性要求。
因果 CNN 编码器处理单图时有两帧前导零填充。音频 DAC/BigVGAN 使用 40 Hz、32 通道
潜变量，800 倍上采样至 32 kHz；解码前需恢复通道均值/标准差。

模型分阶段交还设备权重。Metal 默认在下一段文本编码前释放上段去噪器和 VAE，
`TS_H3_KEEP_RESIDENT=1` 保留旧行为。原生 CUDA 预读不可变待上传权重，
自动预算为物理/cgroup 内存较小值的一半，未知容量时关闭；它不同于托管去噪器文件预读。

## 环境变量

| 变量 | 作用 |
|---|---|
| `TS_H3_CUDA_F32_BANDS` | 默认 pedantic F32；`0` 旧内核，`1` / `tf32` 近似模式 |
| `TS_H3_CUDA_BF16_PEDANTIC` | BF16 稠密投影默认 pedantic F32；`0` 恢复历史近似 |
| `TS_H3_CUDA_MAPPED_VAE_WEIGHTS=1` | 可选映射 F16 VAE 权重，存在冷/暖权衡，默认仍展开 F32 |
| `TS_H3_UPLOAD_PREFAULT` | 原生 CUDA 自动半内存预算；`0` 禁用，`1` 强制 |
| `TS_H3_UPLOAD_WORKERS` | 原生 CUDA 预读 1–8 个 worker，默认 8 |
| `TS_H3_UPLOAD_PROFILE=1` | 预读/上传/图诊断，不能混入正式计时 |
| `TS_H3_CUDNN_ATTENTION=0` | `ggml_cuda`：注意力保留在 ggml 的 flash attention，而不用 cuDNN 融合 SDPA；`TS_H3_CUDNN_ATTENTION_VERBOSE=1` 打印 cuDNN 计划与回退 |
| `TS_H3_CUBLAS_MATMUL=0` | `ggml_cuda`：DiT 的宽 Q4_K 投影保留在 MMQ，而不用 cuBLAS BF16；`TS_H3_CUBLAS_MIN_ROWS`（默认 1024）为 token 阈值 |
| `TS_H3_FUSED=0` | `ggml_cuda`：构建未融合的 DiT 与 VAE 图（不融合注意力前处理、调制、门控残差与 LoRA 合并） |
| `TS_H3_PREFAULT` | 托管去噪器预读：0 关、1 串行、2 提前重叠、3 默认流水线 |
| `TS_H3_PREFAULT_THREADS` | 去噪器预读的读取线程；macOS 默认 1，Linux 默认 16（同一遍扫描中并发读取 8 MiB 块，网络卷需要这种并发） |
| `TS_VIDEO_READAHEAD=0` | 关闭 `ggml_cuda` 文本编码器预读（在上传的页预热之前并发读取）；与 LTX 2.5 共用 |
| `TS_VIDEO_READAHEAD_WORKERS` | 该预读的读取线程，1–32；Linux 默认 16，其他平台默认 1 |
| `TS_H3_TE_GROUP` | 可选原生文本主干分组，默认关闭 |
| `TS_H3_KEEP_RESIDENT=1` | Metal 片段间保留网络 |
| `TS_METAL_FLASH_KV_PAD=0` | 关闭让 `ggml_metal` 长 flash attention 走 tensor-API 内核的 key 补齐（见[上文](#metal-上的长序列注意力)），供 A/B 对比 |
| `TS_H3_PHASE=1` / `TS_H3_TRACE=1` | 阶段耗时 / 潜变量与速度幅度 |
| `TS_H3_DIT_TRACE_DIR` / `TS_H3_LINEAR_TRACE_FILE` | 会同步下载的边界诊断，排除在正式性能数据外 |

## 复现验证

[eng/validation/README-minimax-h3.md](../../eng/validation/README-minimax-h3.md)
提供冻结二进制 A/B、独立 full50、组件与完整媒体检查的命令及证据索引。
`minimax-h3-oracles.py` 拒绝缺失输入、跳过和不完整用例数；
`minimax-h3-generation.py` 记录模型/二进制哈希、依赖、实际硬件和输入，结构通过不代表质量持平。
`minimax-h3-ab.py` 对冻结前后版本要求解码 RGB/音频精确相同。
独立夹具位于 `InferenceWeb.Tests/Fixtures/`，可复用工具位于 `eng/`；
日志、报告与快照留在被忽略的 `docs/validation/` 或 `artifacts/`。
本页整理已记录覆盖，没有把文档更新当作新模型运行。
