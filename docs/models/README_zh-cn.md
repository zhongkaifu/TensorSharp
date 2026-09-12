# 模型架构卡片

[English](README.md) | [中文](README_zh-cn.md)

本目录是 TensorSharp 支持的每一种模型架构的权威分卡参考。每张卡片都是一份独立的简报：从「我从未听说过这个模型」走到「我能解释它的前向计算图，并能在 TensorSharp 中复现它的推理路径」。如果你只需要顶层指引，请使用下方的总览表；否则请直接阅读对应的卡片。

## 每张卡片的内容

为了便于横向对比，所有卡片都采用相同的章节结构：

1. **来源与目标** —— 模型作者、GGUF 架构标识，以及它支持的能力（多模态、思维链、工具调用）。
2. **模型架构** —— 顶层模块图、层数与每层异构性。
3. **前向计算图** —— 单 token（decode）、多 token（prefill）或 diffusion 去噪步骤依次经过的算子序列，包含 residual 与各处 normalization。
4. **组件细节** —— 每一个子模块（attention、FFN/SSM、routing、normalization、RoPE 变体、视觉/音频编码器）的数学描述。
5. **参数与配置** —— GGUF 元数据 key、权重张量命名约定、dtype 要求。
6. **TensorSharp 实现** —— C# 源码定位、初始化顺序、缓存布局、模型如何接入 `ModelBase` / `Ops` / 原生 GGML kernel。
7. **Prefill 优化** —— 分块、融合 per-layer kernel、并行化、跨层缓存。
8. **Decode 优化** —— 融合单调用 kernel、预解析权重指针、批量 MoE、in-place kernel、缓存复用。
9. **内存与 KV cache 策略** —— 环形缓存 vs 线性缓存、mmap 权重、预分配 decode 缓冲。
10. **多模态管线** —— 图像 / 音频 / 视频如何被处理、编码并注入到语言模型。
11. **输出 / 聊天模板** —— 协议解析器、停止 token、思维链 / 工具调用格式。
12. **优化机会** —— 已知尚未实现但能进一步释放性能或能力的工作。

## 已验证的起步路径

已验证的原生 GGML 家族 / 路径层级是 Gemma 4 E4B Q8_0；推荐的公开文件来源是
[ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)。
后端选择 `ggml_cuda`、`ggml_metal` 或 `ggml_vulkan`；这条路径会实际执行
融合原生内核。详见
[Gemma 4 卡片](gemma4_zh-cn.md#已验证的-gemma-4-e4b-原生-ggml-快速路径)。
匹配的 `mmproj` 对纯文本可选，对图像、视频或音频输入则是必需的。

如果希望沿着一条连贯路线，从张量基础一直学到完整的多模态推理引擎，请阅读
Zhongkai Fu 的 [《From Tensors to Tokens》书籍指南](../BOOK_zh-cn.md)，或
[在 Amazon 查看平装本](https://www.amazon.com/dp/B0H9P44QZZ)。

## 实现矩阵

| 架构 | 卡片 | 已验证下载（HF） | 模型类 | GGUF keys | 模态 | 思维链 | 工具调用 | 批处理 / 分页前向 | 主要加速路径 |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | [deepseek41_zh-cn.md](deepseek41_zh-cn.md) | [vcruz305/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF/tree/8e0c4de3cb6519bfc11ed69dc87184b457a57bb5)，七个 Q2_K 分片，外加预处理好的 Engram sidecar；见[验证状态](../deepseek41_validation.md) | `DeepSeek4Model` 走专用的原生 V4.1 计算图，另有纯 C# 执行器 `DeepSeek4CpuExecutor`，它把同一套 V4.1 计算图实现给了 `--backend cpu` | `deepseek41` | 文本；配合单独准备的视觉伴随文件（`--mmproj`；它是原生组件，因此需要 ggml 后端——在 `--backend cpu` 上 `LoadVisionEncoder` 会抛异常）支持图像与视频，音频请求以 HTTP 400 拒绝 | 参考格式渲染器 | 带空格的 DSML；模型层面的验证单独跟踪 | per-sequence slot；decode 走逐 slot 回退路径 | `ggml_cuda`、按层放置、CPU MoE，以及实验性的 routed-MoE TP；注意力/分布式 TP 与 V4.1 DSpark 尚未实现。`--backend ggml_cpu`（同一套原生图跑标量 ggml 内核）与 `--backend cpu`（纯 C# 执行器，不用 ggml、不依赖原生库、也不用 GPU）都是正确性与可移植性路径，而非服务路径 |
| DeepSeek V4 Flash | [deepseek4_zh-cn.md](deepseek4_zh-cn.md) | [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)（每个量化档一个子目录、均为多分片；`--model` 指向 `-00001-of-` 那一片）。DSpark 草稿器见 [MODEL_DOWNLOADS_zh-cn.md](../../MODEL_DOWNLOADS_zh-cn.md) | `DeepSeek4Model`（+ `DeepSeek4CudaExecutor`、`DeepSeek4CpuExecutor`） | `deepseek4` | 文本 | 是 | 是（DSML 标记） | 使用原生 per-sequence slot（`DeepSeek4Model.PerSeqCache.cs`）而非 `IBatchedPagedModel`——仍可通过同一引擎以连续批处理对外服务 | 三套整模型执行器（Direct CUDA、原生 ggml、纯 C#）、按层自动切分到所有可见 GPU、设备端压缩 KV 状态（SWA 环 + CSA/HCA + lightning indexer）、按形状签名的计算图缓存以重放已捕获的 CUDA 图、对 `[ring \| top-512]` K 的融合 decode index-gather，以及 DSpark 块级投机解码（decode 提速 1.3–1.4×） |
| Qwen 3.8 Flash Next | [qwen38-flash-next_zh-cn.md](qwen38-flash-next_zh-cn.md) | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)（多分片；`--model` 指向 `-00001-of-` 那一片） | `Qwen4ExpModel`（GGML 上的整 token 融合图） | `qwen4exp` | 文本 + 图像 | 是 | 是 | 逐序列状态持有者（`SupportsPerSequenceFusedForward`）：每个请求各自的 KV + GDN + PLE 状态，轮询式融合解码 | 每个 token 一张已捕获的图（含图内 PLE 与融合 LM head）、IMRoPE 视觉、跨轮次 KV 复用（只能扩展）、跨所有可见 GPU 的多 GPU **按层切分**（`--tp N`：每张 GPU 持有一段连续的完整层——这是容量特性而非张量并行，`qwen4exp` 不切分任何权重；输出逐字节一致，`TS_Q4E_LAYER_SPLIT` 可覆盖自动均衡） |
| GLM 5.x | [glm_zh-cn.md](glm_zh-cn.md) | [unsloth/GLM-5.2-GGUF](https://huggingface.co/unsloth/GLM-5.2-GGUF)、[unsloth/GLM-5.3-GGUF](https://huggingface.co/unsloth/GLM-5.3-GGUF)（UD-Q2_K_XL 为七个分片、236.4 GiB）、[unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)（每个量化档一个子目录、均为多分片；`--model` 指向 `-00001-of-` 那一片） | `GlmDsaModel`（+ 原生整模型执行器 `ggml_ops_glm_dsa.cpp`）；GLM-5.3（非 Flash）与 5.2 是同一套 79 块 `glm-dsa` 形态（256 个专家、仅文本），无需新代码也无需新标志，直接走 GLM-5.2 的加载路径——详见[卡片](glm_zh-cn.md#glm-53glm-dsa) | `glm-dsa`（GLM-5.2 与 GLM-5.3）、`glm5next`（5.3-Flash）——同一个架构描述符服务这三者，Flash 与非 Flash 只差 `general.architecture` 上的一个布尔量 | 文本（5.2 与 5.3——5.3 仓库在任何量化档都没有发布 mmproj，`LoadVisionEncoder` 在 `glm-dsa` 上遇到 `--mmproj` 只会告警并忽略，而不是让这次运行失败）；文本 + 图像（5.3-Flash，经 `mmproj`） | 是 | 是（XML 工具调用） | 使用原生 per-sequence slot（`TSGgml_GlmSlotAlloc`）而非 `IBatchedPagedModel`——仍可通过同一引擎以连续批处理对外服务 | 原生 ggml 整模型执行器加一套纯 C# 逐算子参考实现（GLM 5.2、GLM-5.3 与 GLM-5.3-Flash 都能在 `--backend cpu` 上跑，但它是用来做 A/B 的参考实现而非逐位一致：在 GLM-5.3-Flash 上与 `ggml_cpu` 对比的 prefill logits 余弦为 0.9567，贪心 token 也不同，详见[卡片](glm_zh-cn.md#--backend-cpu-上的-glm-53-flash)）、按层自动切分到所有可见 GPU **或** Megatron 张量并行（`--tp N`：head 按列/行并行，每个路由专家按行切开）、`--cpu-moe` 让主机端专家直接由 GGUF 映射提供、带权重吸收的 MLA（每 token 一行 576 宽缓存）、DSA lightning indexer（选择结果被 78 层中的 57 层复用），以及按形状索引、重放已捕获 CUDA 图的图缓存；GLM-5.3 的 `blk.78` NextN 块只在不传 `--tp` 时（默认按层切分）用 `--spec` 起草，因为它自己没有 LM head，只能借用主干的 LM head，而 `--tp` 会把这个 head 按列切开 |
| Gemma 4 | [gemma4_zh-cn.md](gemma4_zh-cn.md) | E4B Q8_0 是已验证的原生 GGML 家族 / 路径层级；[ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) 是推荐的公开文件来源 | `Gemma4Model` | `gemma4`（`gemma4-assistant` / `gemma4_assistant` 仅作为 MTP 草稿加载） | 文本、图像、视频、音频 | 是 | 是 | **默认启用**（可用 `TS_GEMMA4_BATCHED=0` 关闭） | 整模型融合 decode（一次 GGML 调度）、带内核内 PLE + 共享 KV 处理的融合整模型 prefill/verify、分块 prefill、SWA 环形缓存与 MoE 变体。批处理路径与旧路径 logits 在 FP 噪声内一致（`Gemma4BatchedForwardTests`）；batch=8 短 prompt 达 ~1.5×，4×800-token prompt 达 ~1.6×。 |
| DiffusionGemma | [diffusiongemma_zh-cn.md](diffusiongemma_zh-cn.md) | [unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) | `DiffusionGemmaModel` + `DiffusionGemmaSampler` | `diffusion-gemma`、`diffusion_gemma` | 文本 | 否 | 否 | 独立的 Web UI `DiffusionBatchScheduler`；不是自回归 `IBatchedPagedModel` 路径 | `[prompt \| canvas]` 上的 EntropyBound 分块去噪、GPU prompt-KV 缓存、self-conditioning、融合 GGML 整模型 diffusion decode 与融合 lm-head tail |
| Qwen-Image-Edit | [qwenimage_zh-cn.md](qwenimage_zh-cn.md) | [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF)（DiT；VAE / 文本编码器配套文件见卡片） | `QwenImageModel`（+ `QwenImagePipeline`） | `qwen_image`、`qwen-image` | 图像编辑（图像+文本 → 图像） | 否 | 否 | 无——`Forward()` 抛异常；编辑通过 `EditImage()` 并串行执行 | 60 块 MMDiT 扩散（FlowMatch-Euler、true-CFG、参考潜变量拼接）、CUDA 图捕获的整 DiT 前向（单次前向约 2.9×）、可选 Lightning 蒸馏 LoRA 以运行期旁路方式接入（`--qwen-image-lora`：60 次 DiT 前向降到 4–8 次）、默认 flash 注意力、CFG-batching、可选启用的 EasyCache / First-Block-Cache 去噪缓存、融合的 Qwen2.5-VL 条件编码器与融合整 VAE 图、按 VRAM 钳制面积 |
| Qwen 3.5 / 3.6 family | [qwen35_zh-cn.md](qwen35_zh-cn.md) | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)；NextN MTP：[unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)（基础仓库的 Qwen3.6 GGUF 去掉了 NextN 块，会静默回退到标准 decode） | `Qwen35Model` | `qwen35`、`qwen35moe`、`qwen3next` | 文本、图像 | 是 | 是 | **默认启用**（`TS_QWEN35_BATCHED=0` 或 `--no-continuous-batching` 可关闭）。带每槽位的递归状态池，可选原生 GatedDeltaNet 内核（`TS_QWEN35_BATCHED_GDN_NATIVE=1`）。 | 全注意力 + GatedDeltaNet 递归混合、融合 attention 层 decode、融合 prefill attention、融合输出投影 + FFN、融合输出投影 + norm + router、批量 MoE（routed + shared + residual 一次完成）、融合视觉编码器 |
| Bonsai Q1_0 | [bonsai_zh-cn.md](bonsai_zh-cn.md) | 仅本地旁加载：`Bonsai-8B-Q1_0.gguf` 与 `Bonsai-27B-Q1_0.gguf` 的确切字节数与 SHA-256 见卡片；它们的 GGUF 未声明发布方 URL 或 license | `Qwen3Model`（8B）、`Qwen35Model`（27B） | `qwen3`、`qwen35` | 文本 | 27B 是；8B 为固定的空 think 块 | 是 | 缓存几何允许时走 Qwen 3 分页接口；Qwen 3.5 默认按槽位持有 KV 与递归状态 | 8B 整模型原生 prefill 与常驻 Metal decode；27B 整模型混合 prefill/verify 与常驻 decode，两者的 Metal 分块实测均为 512 token |
| GPT OSS | [gptoss_zh-cn.md](gptoss_zh-cn.md) | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) | `GptOssModel` | `gptoss`、`gpt-oss` | 文本 | 是（始终启用） | 是 | **默认启用**（`TS_GPTOSS_BATCHED=0` 可关闭）。通过 `TSGgml_PagedAttentionForwardWithSinks` 处理每头 attention sinks（或 `TS_GPTOSS_PAGED_ATTN_MANAGED=1` 使用 C# fallback）。在 `GptOssBatchedCorrectnessTests` 中与旧路径 100% 贪心一致。 | Stacked MoE prefill kernel（mul_mat_id + add_id + swiglu_oai）、attention sinks、MXFP4 专家权重 |
| Nemotron-H | [nemotron_zh-cn.md](nemotron_zh-cn.md) | [bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF)；Omni：[unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF)（图像输入需另配 `mmproj-BF16.gguf`）；Nemotron 3.5 Lightning：[unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF)（另有可选的 DSpark 草稿器供 `--draft-model` 使用） | `NemotronModel` | `nemotron_h`、`nemotron_h_moe` | 文本、图像（Omni 版） | 是 | 是 | **默认启用**（`TS_NEMOTRON_BATCHED=0` 可关闭）。带每槽位 Mamba2 conv + SSM 状态池，可选原生批处理 Mamba2 步（`TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1`）。与旧路径 100% 贪心一致；Apple M4 Pro 上 batch=3 最高可达 3.95× tps。 | Mamba2 + 注意力 + MoE FFN 混合堆栈、批量 GPU MoE、RADIO/v2_vl 图像编码器、Parakeet 音频预处理器（音频推理需要 GGUF 发行版未附带的 Parakeet mmproj） |
| Mistral 3 | [mistral3_zh-cn.md](mistral3_zh-cn.md) | [bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF) | `Mistral3Model` | `mistral3` | 文本、图像 | 否 | 否 | **默认启用** —— `IBatchedPagedModel` 的参考实现。在 Ministral-3-14B 上完成端到端验证；原生分页注意力内核在长上下文下比旧的单序列路径快 ~21%。 | YaRN 校正 RoPE 与位置相关 Q 缩放、融合 QKV / gate_up、Pixtral 视觉编码器 |
| Hunyuan Dense | [hunyuan-dense_zh-cn.md](hunyuan-dense_zh-cn.md) | 声明 `hunyuan-dense` 的腾讯稠密 Hunyuan GGUF，例如 Hy-MT2 系列（参考对话模板取自 `tencent/Hy-MT2-1.8B`） | [`HunyuanDenseModel`](../../TensorSharp.Models/Models/HunyuanDense/HunyuanDenseModel.cs) | `hunyuan-dense` | 文本 | 否 | 否（该协议既不渲染工具声明，也不渲染工具结果） | 通用 per-op 前向 | 暂无——加载时融合 QKV 与 gate/up，单设备，无融合整模型图，无 TP 与按层切分 |
| Muse-Glimmer | [muse-glimmer_zh-cn.md](muse-glimmer_zh-cn.md) | [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF)（`Muse-Glimmer-30B-*.gguf` + `mmproj-*.gguf`；DFlash 草稿器 `dflash-kquant.gguf` 也在同一仓库） | `MuseGlimmerModel` | `muse-glimmer`、`muse_glimmer` | 文本、图像 | 是 | 是 | 否（传统单序列） | 交错滑动窗口 + NoPE 全注意力层、注意力输出门控、每层 4 个 RMSNorm（post-norm eps 1e-8）、logit 缩放 + tanh 软上限、稀疏窗口 2D-RoPE ViT（2x2 像素重排）、可选 DFlash 块级草稿（`--draft-model`，无损）、**张量并行**（GGML CUDA/Vulkan 上 `--tp 2`——2 个 KV 头将并行度上限定为 2） |
| MiniMax-H3 | [minimax-h3_zh-cn.md](minimax-h3_zh-cn.md) | 去噪器（是**两个独立检查点，而不是一个开关**）：`minimax_h3_fl2va_pruned-Q4_K.gguf`（文本 + 关键帧）与 `minimax_h3_ref2va_pruned-Q4_K.gguf`（文本 + 参考），以及两者共用的 Qwen3-VL-32B 文本编码器 `qwen3vl_32b_minimax_h3-Q4_K_M.gguf`，均来自 [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF)；`minimax_h3_video_vae_fp16.safetensors` 与 `minimax_h3_audio_vae_fp32.safetensors`（省略则输出无声视频）来自 [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)。文本编码器 GGUF 不含分词器——请把 [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/processor) 的 `vocab.json` 与 `merges.txt` 放在它旁边，或用 `TS_VIDEO_TOKENIZER` 指向它们 | `MiniMaxH3Model`（+ `MiniMaxH3Pipeline`） | `minimax-h3`、`minimax_h3` —— **但两个已发布的 GGUF 完全没有元数据**，因此 `ModelBase.Create()` 是靠张量表（`MiniMaxH3Architecture.DetectFromTensors`）而不是架构字符串来识别 H3 的 | 视频 **+ 原生 32 kHz 立体声音频**输出，两者在同一份打包潜变量里一起生成（文生视频、图生视频、首尾帧、参考生视频） | 否 | 否 | 无——`Forward()` 抛异常；生成走 `GenerateVideo()` | 原生整网络 ggml 图，每个网络一张图，权重直接从 GGUF / safetensors 的 mmap 常驻绑定；模型已做 CFG 蒸馏，因此工作点是 `--cfg 1.0` 加 4–8 步（M5 Pro / Metal，22 帧、8 步：256×256 比 stable-diffusion.cpp 快 2.4×，640×384 快 1.7×）；AdaLN 用学习得到的曲线表而非时间步 MLP；3 轴连续浮点 RoPE 把视频与音频放在同一条时间轴上；视频 VAE 解码每次 5 个潜变量帧、并按 256 px 分块——这是正确性要求而非优化；FP16 安全的 `h3_attend` 按 key 数量推导出一个 2 的幂来预缩放 V，这正是 107 帧片段不再溢出的原因 |
| Wan 视频 | [wan_zh-cn.md](wan_zh-cn.md) | 基础版：[QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF)、[QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF)、[city96/Wan2.1-T2V-14B-gguf](https://huggingface.co/city96/Wan2.1-T2V-14B-gguf)。**步数蒸馏版（去噪工作量降到 1/25，命令完全不变）：**[hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF](https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF)、[jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF](https://huggingface.co/jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF)。（另需 UMT5-XXL 编码器与视频 VAE，见卡片） | `WanVideoModel`（+ `WanVideoPipeline`） | `wan`、`wan2.1`、`wan2.2` | 视频输出（文本 → 视频、图像 → 视频） | 否 | 否 | 无——`Forward()` 抛异常；生成走 `GenerateVideo()` 且进程内串行 | 按 DiT 文件名自动识别步数蒸馏检查点（100 次 DiT 前向降到 4 次；M5 Pro 1088×832×121f：3 小时 30 分 → 17 分 30 秒）、每个去噪步一张常驻权重的 ggml 图（CUDA 图捕获、F16 键值上的 flash attention——27k token 时快 2.02×，TI2V 图生视频带 per-token 时间步调制）、`--cfg-cache-stride` 引导方向复用（基础检查点上 1.30× / 1.43×）、因果 3D 视频 VAE 编/解码各一张图且 Metal 上卷积走 MPSGraph（VAE 解码 1.99×）、A14B 的两个 14B 专家在时间步边界热切换、分阶段显存交接（TE → DiT → VAE）、按设备显存推导的 im2col 预算与 720p 分块解码 |

## 后端说明

模型代码尽量保持后端无关。`ModelBase` 通过 `BackendType` 与执行计划选择张量存储，再把算子分发给已注册的后端实现：

| 后端类型 | 包 | 说明 |
|---|---|---|
| `Cpu` | `TensorSharp.Core` | 纯托管张量，附带 SIMD / 托管量化快路径（RMSNorm、RoPE、softmax、融合激活、GEMM、dequant）。量化矩阵乘法运行在常驻工作线程池（`TensorSharp.Models/CpuWorkerPool.cs`）上，而非每次矩阵乘法都做一次 `Parallel.For`；这正是解码能扩展到多核的原因。线程池刻意只取可用 CPU 的一半，可用 `TS_CPU_*` 调节。量化权重以零拷贝方式直接绑定到 GGUF 映射上（与 GGML 后端一直采用的 file-backed 绑定一致），而不再在加载时拷贝进新分配的匿名内存（`TS_DIRECT_QUANT_WEIGHTS=0` 可恢复旧的展开成 F32 的行为以便 A/B）。`ManagedQuantizedOps` 还覆盖了 `IQ2_XS` 与 `IQ4_XS`（托管 dequantizer 加入 CPU 量化存储矩阵，因此它们在加载时保持量化而不会展开成 F32），并提供带 AVX2 路径的 `IQ2_XS x Q8_K` 与 `IQ3_XXS x Q8_K` 直接点积内核。生成路径绕过 ggml 图的模型族（Wan、Qwen-Image、MiniMax-H3）共用 `TensorSharp.Models/Direct/` 中的直接执行原语。 |
| `Cuda` | `TensorSharp.Backends.Cuda` | Direct CUDA Driver-API 分配器与存储、cuBLAS GEMM、热点算子的 PTX 内核（RMSNorm、softmax、RoPE/RoPEEx、SDPA、GQA prefill/decode、causal mask、gather/concat、融合激活）、受支持量化类型的原生 quant matmul / get_rows，未实现的算子回退到 CPU。 |
| `Mlx` | `TensorSharp.Backends.MLX` | Apple Silicon `mlx-c` 桥接，含量化 / 融合 / 编译内核、异步 worker 派发、MoE 专家 offload，以及 CPU 回退层。依赖 `libmlxc`。 |
| `GgmlCpu` / `GgmlMetal` / `GgmlCuda` / `GgmlVulkan` | `TensorSharp.Backends.GGML` + `TensorSharp.GGML.Native` | 原生 ggml 桥接，包括量化计算图调度与平台后端；mmap 量化权重通过 host 指针缓冲零拷贝绑定。还包含驱动批处理 / 分页执行路径的分页注意力内核（`TSGgml_PagedAttentionForward`，含 GPT OSS sinks 变体）。 |

凡是卡片中提到融合 GGML kernel（例如 `Qwen35AttentionLayerDecode`、`Gemma4LayerPrefill`、`MoEExpertsSwiGLUResidual`），其源码都在 `TensorSharp.GGML.Native/ggml_ops_*.cpp`，并通过 `TensorSharp.Backends.GGML/GgmlBasicOps.cs` 暴露给托管侧。如果某个融合路径只在 GGML CPU / Metal / CUDA 上启用而在纯托管 CPU 或 direct CUDA 上没有启用，请到原生桥侧查看。

## 连续批处理 & 分页 KV 缓存

上表所列的自回归架构都会经过共享的 `InferenceEngine` + `ContinuousBatchScheduler` + `BatchExecutor` 栈，详情见 [`docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md`](../PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md)。实现了 `IBatchedPagedModel.ForwardBatch` 的模型会在每个调度步骤中执行一次批处理前向（使用基于 `slotMapping` 的 K/V 写入与共享分页缓冲，并通过原生分页内核做按序列注意力）；其余模型则在同一引擎内沿用按序列 KV 交换。DiffusionGemma 不支持自回归 `Forward()`，因此改用 `DiffusionGemmaSampler` 与服务端 `DiffusionBatchScheduler`。Qwen-Image-Edit 同样非自回归：`Forward()` 抛异常，编辑通过 `QwenImageModel.EditImage()` 在 FlowMatch-Euler 扩散循环上进行，且并发编辑被串行化（扩散网络非线程安全）。各模型的启用方式见上方实现矩阵以及项目根 README。

对于自带多 token 预测草稿头的架构——Qwen 3.6（内嵌 NextN 块）与 Gemma 4（独立 `gemma4-assistant` 草稿 GGUF）——单序列（无并发）请求还可以通过同一引擎运行无损的 MTP 投机解码（Qwen 3.6 内嵌的 NextN 块用 `--spec` 显式启用；Gemma 4 只需用 `--draft-model` 指定草稿 GGUF，本身即可启用投机。两个标志 **CLI 与服务端都接受**，因为 `TensorSharp.Cli` 与 `TensorSharp.Server` 共用同一套标志解析；`TS_SPEC_*` 以及旧的 `TS_MTP_*` 环境变量同样有效）。共享的起草 / 验证 / 回滚核心是 `SpeculativeExecution`；各架构具体机制见 Qwen 3.5/3.6（§12）与 Gemma 4（§12）卡片。

DeepSeek V4 把一个**块级**草稿器接入了同一套核心：它的 DSpark 支持模块作为独立 GGUF 通过 `--draft-model` 加载（CLI 与服务端都支持），每步提议一整块 token 而不是逐个。由于草稿器的权重必须计入层切分，它在加载阶段就传给 `ModelBase.Create()`，而不是事后附加。详见 [DeepSeek V4 卡片](deepseek4_zh-cn.md#dspark-投机解码)。

## 架构对比

| 特性 | DeepSeek V4 | Gemma 4 | DiffusionGemma | Qwen 3.5 / 3.6 family | GPT OSS | Nemotron-H | Mistral 3 | Muse-Glimmer |
|---|---|---|---|---|---|---|---|---|
| 层类型 | MoE（256 个路由专家，top-6 + 1 共享） | 密集 / MoE | Gemma-4 派生 MoE encoder/decoder | 混合（注意力 + 递归）± MoE | MoE | 混合（Mamba2 + 注意力 + FFN，密集或 MoE） | 密集 | 稠密（52 层，32 Q 头 / 2 KV 头） |
| 注意力 | 原始 SWA-128 + 压缩注意力 CSA 4:1 / HCA 128:1（CSA 层用 lightning indexer 选 top-512） | SWA + 全局 | 区分 prompt/canvas 的区域感知注意力 | 全 GQA + Sigmoid Gate | 全 + Sinks | 全 GQA（无 RoPE） | 全 GQA | 交错 SWA-2048 + NoPE 全注意力层（39 + 13），带 sigmoid 注意力输出门控 |
| FFN 激活 | SwiGLU（每层带 clamp） | GeGLU | Dense GeGLU + top-8 MoE | SwiGLU | SiLUAlphaLimit（带 clamp 的 GLU） | ReLU² | SwiGLU | SwiGLU |
| RoPE 类型 | 交错成对 + YaRN；raw 与 compress 两套 base，注意力后再做逆旋转 | NeoX + 比例 / 部分 | NeoX，local/global base | NeoX / MRoPE | NeoX + YaRN | 无 | GPT-J + YaRN | ggml NORM（相邻成对交错），仅用于 SWA 层；全注意力层为 NoPE |
| QK-norm | 仅 Q（每头 RMS） | 是 | 是 | 是 | 否 | 否 | 否 | 是（每头；Q norm 权重折入了 qk_scale_factor） |
| V-norm | 否 | 是（无权重） | 是（无权重） | 否 | 否 | 否 | 否 | 否 |
| 投影偏置 | 无（仅路由选择偏置） | 无 | 无 | 无 | 全部都有 | 无 | 无 | 无 |
| 每层缩放 | 否（改为每层 swiglu clamp 与压缩比） | 是 | encoder / decoder 标量 | 否 | 否 | 否 | 否 | 否（改为输出侧 logit 缩放 0.19612 + tanh 软上限 20.0） |
| Per-Layer Embedding (PLE) | 否 | 是 | 否 | 否 | 否 | 否 | 否 | 否 |
| KV 共享 | 是（所有 query 共用一个 512 维 K=V 头） | 是（尾部若干层） | 去噪多步间复用 prompt-KV | 否 | 否 | 否 | 否 | 否 |
| Attention sinks | 是 | 否 | 否 | 否 | 是 | 否 | 否 | 否 |
| 环形 KV cache | 是（原始 SWA-128 环） | 是（SWA 层） | 无自回归 KV | 否 | 否 | 否 | 否 | 是（GPU 后端上的 SWA 环；`TS_MUSE_GLIMMER_SWA_RING=0` 关闭） |
| SSM / 递归层 | 否（用 4 路 hyper-connection 取代普通残差） | 否 | 否 | 是（GatedDeltaNet） | 否 | 是（Mamba2） | 否 | 否 |
| 共享专家 | 是 | 否 | 否 | 是（qwen35moe / qwen3next） | 否 | 是（可选） | 否 | 否（稠密 FFN） |
| Latent bottleneck FFN | 否（改为 LoRA 分解的 Q / 输出投影） | 否 | 否 | 否 | 否 | 是（可选） | 否 | 否 |
| 位置相关 Q 缩放 | 否 | 否 | 否 | 否 | 否 | 否 | 是（与 YaRN 配合） | 否 |
| 视觉 | 否 | 是 | 否 | 是 | 否 | 是（Omni） | 是（Pixtral） | 是（稀疏窗口 2D-RoPE ViT，2×2 像素重排） |
| 音频 | 否 | 是 | 否 | 否 | 否 | 否 —— Omni 仅图像（Parakeet log-mel 预处理已实现，但推理需要未随发行版提供的音频 mmproj） | 否 | 否 |
| 视频 | 否 | 是 | 否 | 否 | 否 | 否 | 否 | 否 |
| 思维链 | 是 | 是 | 否 | 是 | 是（始终启用） | 是 | 否 | 是（`assistant to=self` 通道） |
| 工具调用 | 是（DSML 标记） | 是 | 否 | 是 | 是 | 是 | 否 | 是（ATEM XML 标记） |
| MTP / NextN 投机解码 | DSpark 块级草稿器（独立 GGUF，`--draft-model`） | 是（独立 `gemma4-assistant` 草稿 GGUF） | 否 | Qwen 3.6 支持（内嵌 NextN 块） | 否 | 否 | 否 | DFlash 块级草稿器（独立 5 层 GGUF，`--draft-model`，无损） |
| 融合 QKV | n/a（LoRA 分解的 Q，单个共享 K=V 头） | 是 | 是 | 混合（attention 层拆开，递归层融合 5 路） | 是 | 是 | 是 | 否 |
| 融合单调用 decode | 是（整模型执行器，每个 ubatch 一张图，重放 CUDA 图） | 是（Gemma4ModelDecode） | 是（DiffusionModelDecode + lm-head tail） | per-layer 融合（Qwen35AttentionLayerDecode、FusedOutProjFFN、FusedOutProjNormRouter） | per-layer | per-layer / 批量 MoE | 否 | 是（GGML CUDA / Vulkan / Metal / CPU 上的常驻整模型 decode 图） |
| 融合单调用 prefill | 是（同一整模型执行器，分块 ubatch） | 是（整模型 NativeGemma4ModelVerify + 逐层 Gemma4LayerPrefill 回退） | prompt-KV prefill cache | 是（FusedPrefillAttention、FusedOutProjFFN、MoE prefill） | 是（MoE prefill via mul_mat_id） | 否 | 否 | 是（同一融合内核，分块并在设备端生成 causal+SWA 带状掩码） |
| 批量 GPU MoE | 是（分组专家内核） | 全 MoE 变体已支持（融合整模型 MoE decode/verify）；混合 dense+MoE 待实现 | 融合单 canvas MoE；并发请求由 diffusion scheduler 批处理 | 是（routed + shared + residual 融合） | 是（stacked weight slabs） | 是 | n/a | n/a（稠密 FFN） |
| 融合视觉编码器 | n/a | 标准 | n/a | 是（FusedVisionAttention + FusedVisionMLP） | n/a | 标准（RADIO ViT） | 标准（Pixtral） | 是（CUDA 上融合视觉块 + flash attention） |
| 输出解析器 | `DeepSeek4OutputParser` | `Gemma4OutputParser` | `PassthroughOutputParser` | `Qwen35OutputParser` | `HarmonyOutputParser`（始终启用） | `ChatMlOutputParser` | `PassthroughOutputParser` | `MuseGlimmerOutputParser` |

## 新增模型架构

要在 TensorSharp 中加入新的模型架构：

1. 在 `TensorSharp.Models/Models/<Name>/<Name>Model.cs` 创建类并继承 `ModelBase`。
2. 构造函数中：通过 `_gguf.GetXxx()` 读取 GGUF 元数据，调用 `ParseBaseConfig()` 与 `ParseTokenizer()`，调用 `LoadWeights()`，融合权重，然后初始化缓存。
3. 自回归模型实现 `Forward(int[] tokens) → float[]`：embedding → 可选多模态注入 → transformer 层 → final norm → LM head → logits 拷贝。扩散模型需要明确记录替代 sampler 入口，并让不支持的自回归路径显式失败。
4. 实现 `ResetKVCache()` 与 `Dispose()`；如支持 KV 缓存复用则实现 `TruncateKVCache()`。
5. 在 `TensorSharp.Models/Models/<Name>/<Name>Architecture.cs` 中声明架构插件——一个 `ModelArchitectureDescriptor`，写明 GGUF `general.architecture` 别名、工厂方法，以及所有非默认项（多卡模式及其原因、mmproj 伴随文件名提示、面向无元数据 GGUF 的张量检测器、进程级原生调优开关）——然后在 `TensorSharp.Models/Architecture/BuiltInArchitectures.cs` 里为它加**一行**。不再有需要扩展的 switch：`ModelBase.Create()` 通过 `ModelArchitectureRegistry` 解析。
6. 如果模型是多模态的，在模型上实现能力接口：`IVisionCapableModel`（加载视觉塔、接收一段 embedding）与 `IMultimodalPromptExpander`（展开自己的占位符），并按需实现 `IAudioCapableModel` / `IAudioEncoderLoader` / `IMRoPEPositionSink`。`ModelMultimodalInjector` 拥有全部通用记账逻辑且不出现任何模型类型名，因此那里无需改动。
7. 如果模型有自己的对话格式，在 `TensorSharp.Runtime/ChatProtocolRegistry.cs` 里加**一条** `ChatProtocol` 记录。这一条记录同时承载：渲染器、是否绕过 GGUF 自带的 Jinja 模板、媒体占位符 token、`IOutputParser`（实现放在 `TensorSharp.Runtime/OutputParser.cs`）、该解析器是否必须运行、结构化输出语法从何处开始生效、保证多轮前缀复用的 KV cache 生成后缀，以及视频抽帧上限。
8. 在 `docs/models/<name>.md`（与 `<name>_zh-cn.md` 如果你打算双语覆盖）下新增卡片，更新本 README 的实现矩阵，并从项目根 README 链入卡片。
9. 如果模型涉及新的模态、思维链或工具能力，更新 `TensorSharp.Server/testdata/` 的能力门控。
