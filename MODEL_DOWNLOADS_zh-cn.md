# 模型下载（GGUF）
[English](MODEL_DOWNLOADS.md) | [中文](MODEL_DOWNLOADS_zh-cn.md)

> [TensorSharp](README_zh-cn.md) 文档的一部分。另见[各模型架构卡片](docs/models/README_zh-cn.md)。


TensorSharp 使用 GGUF 格式模型文件。以下是各架构对应的已核对 Hugging Face 下载入口与伴随文件。请根据硬件条件选择合适的量化版本（Q4_K_M / UD-Q4_K_XL 适合低内存，Q8_0 适合更高质量等）。标注“可选”的条目是提速用的产物——步数蒸馏 checkpoint、推测解码 draft 模型。不下载也能跑通，但它们往往就是“几分钟”和“几小时”的差别，动手前请先扫一眼。

| 架构 | 模型 | GGUF 下载 |
|---|---|---|
| 嵌入编码器（`bert` / XLM-R） | Snowflake Arctic Embed L v2.0，Q8_0，1024 维 | [fisher046/snowflake-arctic-embed-l-v2.0-Q8_0-GGUF](https://huggingface.co/fisher046/snowflake-arctic-embed-l-v2.0-Q8_0-GGUF)，文件 `snowflake-arctic-embed-l-v2.0-q8_0.gguf`；约 635 MB；使用 `--embeddings`。固定修订版、校验和与示例见[指南](docs/embeddings_zh-cn.md) |
| 嵌入编码器（`bert`） | all-MiniLM-L6-v2，Q8_0，384 维 | [second-state/All-MiniLM-L6-v2-Embedding-GGUF](https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF)，文件 `all-MiniLM-L6-v2-Q8_0.gguf`；约 25 MB；使用 `--embeddings` |
| Gemma 4 已验证原生规格 | gemma-4-E4B-it Q8_0 | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)；推荐公开文件为 `gemma-4-E4B-it-Q8_0.gguf`，另有低内存 Q4_K_M；同仓库投影器为 `mmproj-gemma-4-E4B-it-Q8_0.gguf` |
| Gemma 4 | 12B / 26B-A4B QAT | [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) / [unsloth/gemma-4-26B-A4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF)；同仓库含 `mmproj-BF16.gguf`，以及匹配的 MTP draft（`mtp-gemma-4-12B-it.gguf` / `mtp-gemma-4-26B-A4B-it.gguf`，可选，仅用于推测解码） |
| Gemma 4 | 31B / 26B-A4B | [ggml-org/gemma-4-31B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF) / [ggml-org/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF)；同仓库含 mmproj |
| Gemma 4 | E4B / 26B-A4B MTP draft（可选，仅用于推测解码） | [AtomicChat E4B assistant](https://huggingface.co/AtomicChat/gemma-4-E4B-it-assistant-GGUF) / [AtomicChat 26B assistant](https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF)；用 `--draft-model` 加载，指定它本身就会开启推测解码；仅与匹配尺寸的目标配对 |
| Qwen 3.5 | Qwen3.5-9B | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)，投影器 `mmproj-F16.gguf` |
| Qwen 3.5 | Qwen3.5-35B-A3B | [ggml-org/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF)，投影器 `mmproj-Qwen3.5-35B-A3B-Q8_0.gguf` |
| Qwen 3.6 | Qwen3.6-35B-A3B（保留 NextN） | [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)，投影器 `mmproj-F16.gguf`。**注意不要下载基础仓库** [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)：它的文件名完全相同，但剥离了 NextN 块，`--spec` 会静默回落到普通解码 |
| Qwen 3.8 | Qwen3.8-27B（稠密混合，内嵌 NextN MTP，支持图像） | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)，如 `Qwen3.8-27B-UD-Q4_K_XL.gguf`，即 [`config/agent-qwen3.8-27b.json`](config/agent-qwen3.8-27b.json) 以提交与 SHA-256 固定的文件；它保留 `--spec` 所需的 NextN 块，同仓库有投影器 `mmproj-BF16.gguf`（需用 `--mmproj` 指定）。它与 Qwen 3.5 走同一条 `Qwen35Model` 路径。可选提速产物：DFlash2 分块 draft [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF)，用 `--draft-model` 加载（挂上后取代 NextN 块）；收益取决于负载，见 [qwen35_zh-cn.md](docs/models/qwen35_zh-cn.md) §12.4 |
| Qwen 3.8 Flash Next | Qwen3.8-Flash-Next（混合 MoE，支持图像） | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)；每种量化一个子目录（`UD-Q2_K_XL/` 等），均为多分片，`--model` 指向 `-00001-of-` 分片。`UD-Q2_K_XL/` 共三个分片、78,869,128,864 字节（78.9 GB），在 48 GB 的 Apple 芯片 Mac 上也能运行，因为 `ggml_metal` 按需从 SSD 读取 n-gram 表与大部分路由专家（实测于 M5 Pro；见模型卡片的[超出内存运行](docs/models/qwen38-flash-next_zh-cn.md#超出内存运行)）。TensorAgent 的桌面目录（48 GB 档位，仅文本）会下载这三个分片，并按固定的大小与 SHA-256（取自 HF `main` = `38bb39ee`）校验。同仓库的 `mmproj-BF16.gguf` 启用图像输入，多图提示与多轮图像会话都可用：CLI 会在模型旁边找到它，服务端需用 `--mmproj` 指定。`general.architecture` 为 `qwen4exp`。多卡机器上 `--layer-split N` 走的是**按层切分**——整层落在单卡，也是 llama.cpp 对这个架构唯一提供的多卡模式——买到的是容量而不是速度，见 [USAGE_zh-cn.md](USAGE_zh-cn.md#张量并行与分布式推理)。可选的提速产物：共享 MTP 头 `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`（选带 `-shared-` 的那个，而不是旁边其他 `MTP/mtp-*` 头：加载器只接受共享目标张量的 NextN 块；2,786,568,256 字节，取自本仓库的 revision `38bb39ee97821de2c9009abb7e93950eec396e66`），是单个 GGUF，在 GGML 后端上用 `--draft-model` 加载。在三张 A40 上以 UD-Q2_K_XL 实测，它在大量照抄的输出上有收益（采用 2026-09-17 加入的精确 verify 行 kernel 后，192 token 代码照抄流约 1.7 倍：83.2 对普通 49.1 tok/s；此前为 1.75-1.96 倍），但 512 token 散文的解码反而变慢（44-46 对 52 tok/s，测于该改动之前）；见 [qwen38-flash-next_zh-cn.md](docs/models/qwen38-flash-next_zh-cn.md#共享-mtp-头的投机解码) 当前 `ggml_cuda --tp N` 支持 UD-IQ4_XS 等合格量化的本地 FFN TP；UD-Q2_K_XL 则拒绝 TP。见[验证范围与基准](docs/models/qwen38-flash-next_zh-cn.md#多-gpu)。 |
| Bonsai2 | Ternary-Bonsai-2-27B（PQ2_0 / PTQ1_0，支持图像） | [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/b072e1d3b35a0a630cece372c2127528e0994386)，revision `b072e1d3b35a0a630cece372c2127528e0994386`（Apache-2.0）——`Ternary-Bonsai-2-27B-PQ2_0.gguf`（7.21 GB）或 `Ternary-Bonsai-2-27B-PTQ1_0.gguf`（5.95 GB）；请按 [bonsai2_zh-cn.md](docs/models/bonsai2_zh-cn.md) 中的 SHA-256 校验。两者都声明 `general.architecture` = `qwen35`，带 PRISM 带符号 Hadamard 元数据（`prism.hadamard.*`）和自定义张量类型 PQ2_0 / PTQ1_0，TensorSharp 在加载时把它们无损转码为 GGML Q2_0（在内存中比存储的数据分别多占约 6% / 29%）。同仓库的配套投影器 `Ternary-Bonsai-2-27B-mmproj-BF16.gguf` / `-mmproj-Q8_0.gguf` 启用图像输入：CLI 会在模型旁边找到它们，服务端需用 `--mmproj` 指定。仅支持单设备 GGML 后端（`cpu`、`cuda`、`mlx` 与 `--tp` 会被拒绝）；目前的端到端验证只在 Metal（M5 Pro）上完成。TensorAgent 的实验性条目从这个 revision 下载 PTQ1_0 文件，以及可选的 Q8_0 投影器，最低需要 16 GB 设备档位 |
| GPT OSS | gpt-oss-20b（MoE） | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF)，文件 `gpt-oss-20b-MXFP4.gguf`（注意 `MXFP4` 为大写）；纯文本，无伴随文件 |
| Nemotron-H | Nemotron-H-8B / 47B Reasoning | [8B](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF) / [47B](https://huggingface.co/bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF) |
| Nemotron-H | Nemotron 3 Nano Omni 30B-A3B | [unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF)，图像输入需 `mmproj-BF16.gguf`。除非加载了单独转换的音频配套 GGUF，音频输入会被拒绝（HTTP 400 / CLI 错误）：这些 GGUF 不带音频塔，见 [nemotron_zh-cn.md §4.6-4.7](docs/models/nemotron_zh-cn.md) |
| Nemotron 3.5 | Nemotron-3.5-Lightning-30B-A3B（23 Mamba-2 + 23 MoE + 6 注意力的混合架构） | [unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF)，例如 `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MXFP4_MOE.gguf`（MoE 专家为 MXFP4，约 17 GB）；`general.architecture` = `nemotron_h_moe`。更小 / 其他量化见 [ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF)（BF16/Q4_0/Q8_0 与独立的 MTP GGUF）。下一行的 DSpark drafter 仅作参考列出：这个主干拒绝投机解码 |
| Nemotron 3.5 | DSpark 投机 drafter——**不会使用** | [magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark-GGUF](https://huggingface.co/magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark-GGUF)（官方 DSpark 模块的 llama.cpp DFlash 导出版）。TensorSharp 在 Nemotron-H 上拒绝投机解码，`--draft-model` 不会挂载它：主干的 verify 与 decode 内核结果不一致，投机输出会与普通解码不同（见 [speculative_decoding.md](docs/speculative_decoding.md#nemotron-h-refuses-speculation)） |
| Mistral 3 | Mistral-Small-3.1-24B-Instruct | [bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF)，Pixtral 投影器 `mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf` |
| Hunyuan Dense | 腾讯稠密 Hunyuan 检查点（`hunyuan-dense`） | 任何 `general.architecture` 为 `hunyuan-dense` 的 GGUF 均可加载，例如 Hy-MT2 系列（参考对话模板取自 `tencent/Hy-MT2-1.8B`）。仅文本、单设备，没有投影器也没有草稿模型。见 [hunyuan-dense](docs/models/hunyuan-dense_zh-cn.md) |
| Muse-Glimmer | Muse-Glimmer-30B（稠密，支持图像） | [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF)，如 `Muse-Glimmer-30B-UD-Q4_K_XL.gguf` 或 `Muse-Glimmer-30B-Q8_0.gguf`；`general.architecture` 为 `muse-glimmer` / `muse_glimmer`。图像输入需同仓库的 `mmproj-Muse-Glimmer-30B-Q8_0.gguf`：传 `--image` 时 CLI 会在模型旁边找到它，服务端则需用 `--mmproj` 指定。可选提速产物：同仓库的 DFlash 分块 draft `dflash-kquant.gguf`，或更新的 DFlash2 draft [z-lab/Muse-Glimmer-30B-DFlash2-GGUF](https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2-GGUF)（16 GB 显卡上优先选 `-Q4_K_M`，见 [speculative_decoding.md](docs/speculative_decoding.md#what-to-expect) 中关于 draft 大小的说明），用 `--draft-model` 加载即可无损推测解码——不要传任何采样参数，它只在纯贪心下生效 |
| DeepSeek V4.1 | DeepSeek-V4.1-Flash（`deepseek41`，384 个路由专家） | [smalinin/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/tree/d1de55c19f95172c882906cc83c0e55932d26a63/Q2_K-Q5)，固定 revision `d1de55c19f95172c882906cc83c0e55932d26a63`——`Q2_K-Q5/` 下十个分片，共 335,382,014,624 字节（312.349 GiB），Engram 表为 Q5_K，并恢复 F32 mHC 与 BF16 Engram 门控张量。所有分片放在同一目录，`--model` 指向第一个分片。原七分片 vcruz Q2 存在敏感张量量化问题，不再推荐。修复版头部与十个完整文件哈希均已校验，双卡按层切分与实验性路由专家 TP 下的有限普通/DSpark 文本/图像 HTTP 检查均已通过，回答以 EOS 完成，服务正常退出。另行进行的 24-token 普通/DSpark 文本/图像配对检查，在两种模式下均匹配 token ID 与 `max_tokens` 结束原因，DSpark 实际参与解码且进程正常退出；这些有限续写与 HTTP 的 EOS 检查分开记录。两个图像配对中 DSpark 都更慢。简短的重复文本预热对照也已完成：每种模式一次预热、三次计时，每对输入 69 token、输出 24 token，token ID 与结束原因完全一致，DSpark 实际参与解码且进程正常退出。小样本与大量磁盘换页限制了这些结果，不声明通用质量/性能、提速、跨节点执行或完整模型 TP 通过。每次重新下载仍须校验哈希。GGUF 已包含 Engram 权重与哈希常量，无需生成 Engram 或单独的 Engram 文件。`eng/dsv41-prepare-vision.py` 生成约 970 MB 的可选视觉伴随文件，图像与视频经 `--mmproj` 使用。服务后端为 `ggml_cuda`；`ggml_cpu` 与 `cpu` 是正确性与可移植性路径，而非服务路径——`--backend cpu` 用纯 C# 执行器 `DeepSeek4CpuExecutor` 跑完整的 V4.1 计算图，不依赖 ggml、原生库与 GPU；它同样直接读取内嵌的 Engram 元数据，而视觉伴随文件不会跟到这个后端上（`LoadVisionEncoder` 会抛异常），因此该后端上没有图像也没有视频。V4 的草稿模型会被拒绝；`deepseek41-dspark` 草稿模型可在 `ggml_cuda` / `ggml_cpu` 上经 `--draft-model` 加载，属于实验性路径，训练模型已在 `ggml_cuda` 双 GPU 按层切分与实验性路由专家 TP 下通过初步文本/图像 HTTP 检查；尚不构成通用质量或吞吐验证。完整流程与校验哈希见 [deepseek41](docs/models/deepseek41_zh-cn.md) |
| DeepSeek V4 | DeepSeek-V4-Flash-0731（284B MoE） | [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)；每种量化一个子目录（`UD-Q8_K_XL/`、`UD-IQ4_XS/` 等），均为多分片，`--model` 指向 `-00001-of-` 分片。仅文本 |
| GLM 5.x | GLM-5.2（744B-A40B MoE，内嵌 NextN MTP） | [unsloth/GLM-5.2-GGUF](https://huggingface.co/unsloth/GLM-5.2-GGUF)；每种量化一个子目录（`UD-Q4_K_XL/`、`UD-IQ2_XXS/` 等），均为多分片，`--model` 指向 `-00001-of-` 分片。**仅文本**——再往下两行的 GLM-5.3-Flash 才是支持图像的那个；中间那行的 GLM-5.3 同样仅文本。这些 GGUF 已带有 `--spec` 所需的 NextN 块——与 Qwen 3.6 不同，不存在需要挑选的独立 MTP 仓库 |
| GLM 5.x | GLM-5.3（`glm-dsa`，256 个路由专家，仅文本） | [unsloth/GLM-5.3-GGUF](https://huggingface.co/unsloth/GLM-5.3-GGUF)；每种量化一个子目录（`UD-Q2_K_XL/` 等），均为多分片，`--model` 指向 `-00001-of-` 分片。`general.architecture` 为 `glm-dsa`，层结构与 GLM-5.2 一致（79 个 block —— 78 层主干加 1 个 NextN ——256 个路由专家 top-8、带 lightning indexer 的 MLA、rope base 8e6），因此直接走现有的 GLM-5.2 路径，无需额外开关。**仅文本**——与下面的 Flash 仓库不同，这个仓库完全没有发布 mmproj。它确实带着供 `--spec` 使用的 NextN 块，但 `blk.78` 没有自己的 `nextn.shared_head_head.weight`，draft 块只能借用主干的 LM head——而在 `--tp N` 下该 head 是按列切分的。加载器拒绝用某个 rank 上的词表切片来 draft，并在 stderr 上明说，所以 `--spec` 在**单设备或按层切分模式**下生效；用 `--layer-split N` 指定 N 张本地 GPU，不启用张量并行 |
| GLM 5.x | GLM-5.3-Flash（320B，288 个路由专家，文本 + 图像） | [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)；每种量化一个子目录（`UD-Q2_K_XL/` 等），均为多分片，`--model` 指向 `-00001-of-` 分片。`general.architecture` 为 `glm5next`，与 GLM-5.2 走同一个原生执行器。与 5.2 不同，它**支持图像**：同仓库的 `mmproj-BF16.gguf`（GLM-OCR ViT）启用 `--image`、多图提示与多轮图像会话。它的 NextN/MTP 块尚未接入；无需训练权重的 n-gram 推测使用 `--spec --spec-type ngram`。用 `--layer-split N` 将整层放到 N 张本地 GPU；在 GGML GPU 后端上，传入 `--tp N` 则选择仅支持本地单进程的原生张量并行 |
| DeepSeek V4 | DSpark 推测解码 draft（可选，仅提速） | 见下方 [DSpark draft 模型](#dspark-draft-模型)，用 `--draft-model` 加载，解码约 1.3-1.4 倍 |
| DiffusionGemma | diffusiongemma-26B-A4B-it | [unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF)，如 `diffusiongemma-26B-A4B-it-Q4_K_M.gguf`；unsloth 没有发布的更小 Q3_K_M 在 [DevQuasar/google.diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/DevQuasar/google.diffusiongemma-26B-A4B-it-GGUF)。所有已发布的 GGUF 都只含文本部分。图像输入还需下载 Gemma-4 视觉塔：TensorSharp 直接读取 [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) 中的上游分片 `model-00011-of-00011.safetensors`（2.84 GB），用 `--mmproj` 指定（这个系列没有投影器自动探测）。音频会被拒绝；没有视频路径：OpenAI 的 `video_url` 会被拒绝，Web UI 上传的视频只以抽出的帧（当作普通图像）送入模型。`config/diffusiongemma-26b-a4b-*.json` 会把两者都下载好；见 [diffusiongemma_zh-cn.md](docs/models/diffusiongemma_zh-cn.md) |
| Qwen-Image-2.1 | 扩散 Transformer（即 `--model` GGUF） | [Abiray/Qwen-Image-2.1-GGUF](https://huggingface.co/Abiray/Qwen-Image-2.1-GGUF)，文件 `qwen_image_2.1_Q4_K_M.gguf`。[`config/qwen-image-2.1.json`](config/qwen-image-2.1.json) 会以固定修订版本和 SHA-256 校验下载它以及下面三个伴随文件（共四个文件，约 10.29 GiB）。Unsloth 不含元数据的 `qwen-image-2.1-Q8_0.gguf`（GGUF 元数据为零，张量混合 BF16/F32/Q8_0）同样可以加载：架构由张量布局识别，包括 `model.diffusion_model.` 前缀；它的 `mmproj-BF16.gguf` 匹配不上伴随文件扫描，需用 `--qwen-image-mmproj` 指定。更早的 Qwen-Image / Qwen-Image-Edit checkpoint 会被拒绝。详见 [qwenimage21_zh-cn.md](docs/models/qwenimage21_zh-cn.md) |
| Qwen-Image-2.1 | 专用 2.1 VAE（必需） | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/vae) 中的 `vae/qwen_image_2.1_vae_bf16.safetensors`——放在 DiT 旁，或用 `--qwen-image-vae` / `TS_QWEN_IMAGE_VAE` 指定 |
| Qwen-Image-2.1 | Qwen3-VL-8B 文本编码器（必需） | [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) 中的 `Qwen3VL-8B-Instruct-Q4_K_M.gguf`——放在 DiT 旁，或用 `--qwen-image-vl` / `TS_QWEN_IMAGE_TE` 指定 |
| Qwen-Image-2.1 | 编辑用视觉编码器 | 同一 Qwen3-VL 仓库中的 `mmproj-Qwen3VL-8B-Instruct-F16.gguf`——放在 DiT 旁，或用 `--qwen-image-mmproj` / `TS_QWEN_IMAGE_MMPROJ` 指定 |
| Qwen-Image-2.1-Turbo | 扩散 Transformer，8 步蒸馏版（即 `--model` GGUF） | [AtomicChat/Qwen-Image-2.1-Turbo-GGUF](https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF)，文件 `Qwen-Image-2.1-Turbo-AD-Q4_K.gguf`（4.20 GB）；`Qwen-Image-2.1-Turbo-Q8_0.gguf`（7.59 GB）是最接近全精度的 Transformer（文本编码器仍是同一个 Q4_K_M）。[`config/qwen-image-2.1-turbo.json`](config/qwen-image-2.1-turbo.json) 以固定修订版本与 SHA-256 把 AD-Q4_K 下载到模型根目录下的 `qwen-image-2.1/`，VAE、文本编码器与 mmproj 使用上面列出的发布文件名，因此已存有这些文件的目录只需下载 Transformer。GGUF 不含元数据，因此配置文件声明了检查点（`--qwen-image-variant turbo`）。详见 [qwenimage21_zh-cn.md](docs/models/qwenimage21_zh-cn.md#qwen-image-21-turbo) |
| Qwen-Image-2.1 | LoRA 插件（可选） | [`config/lora/`](config/lora/) 中的十二个插件，用 `--lora` 加载；每个插件在首次使用时以固定修订版本与 SHA-256 从 Hugging Face 下载其 `.safetensors`，保存到模型根目录下的 `qwen-image-2.1/loras/`（Fun-Acc 还会下载它转交的 `pdd_config.json`）。包括步数蒸馏适配器（Viggle Turbo、Pruna 8/5 步、阿里巴巴 PAI Fun-Acc 4 步）以及风格 / 编辑 LoRA；其中数个仅限非商业用途。见 [USAGE_zh-cn.md](USAGE_zh-cn.md#qwen-image-21-lora-插件) |
| MiniMax-H3 音视频生成 | 去噪器（`--model` GGUF） | **两个独立的 checkpoint，不是开关**——加载哪一个决定了它接受什么条件输入。[unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF)：`minimax_h3_fl2va_pruned-Q4_K.gguf`（10.64 GiB）用于文生视频 / 图生视频 / 首尾帧，`minimax_h3_ref2va_pruned-Q4_K.gguf`（10.60 GiB）用于身份与外观参考。另有 Q8_0（19.97 GiB）到 Q2_K（6.26 GiB）。H3 是 CFG 蒸馏模型：**CFG 必须为 1.0**（默认值）；当前默认 20 步，4–8 步是以质量换速度的选择。这些 GGUF **完全没有元数据**，TensorSharp 靠张量表识别它们，并从文件名读出分区——重命名或重新量化时请保留 `fl2va` / `ref2va`。两个 checkpoint 共用下面三个网络，所以事后再加另一个只需下它自己的约 10.6 GiB。TensorAgent 的桌面目录（32 GB 档位）会自己下载整套带哈希校验的文件（含分词器），分为 `minimax-h3-fl2va-q4k` 与 `minimax-h3-ref2va-q4k` 两个条目；后安装的那个直接链接已有的 24.0 GB 共用文件而不重新下载，只下载自己的去噪器。见 [minimax-h3_zh-cn.md](docs/models/minimax-h3_zh-cn.md#在-tensoragent-mac-应用里) |
| MiniMax-H3 音视频生成 | Qwen3-VL-32B 文本编码器（必需） | 同仓库：`qwen3vl_32b_minimax_h3-Q4_K_M.gguf`（16.97 GiB），或 `-Q2_K_M.gguf`（12.20 GiB）以搭配最小的那几个去噪器。截断到 50 层并去掉最后的 norm，去噪开始前即从显存释放。**它不含分词器**——还需要下一行那三个文件 |
| MiniMax-H3 音视频生成 | `vocab.json` + `merges.txt` + `tokenizer_config.json`（必需） | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/processor) 的 `processor/`（三个合计 4.46 MB）——编码器 GGUF 缺的那对 Qwen2 字节级 BPE 文件，外加 `tokenizer_config.json`：只有它定义了 `<\|vision_start\|>`、`<\|image_pad\|>` 和 `<\|vision_end\|>`。文生视频与参考全是录音的请求不需要它；缺了它，带关键帧、或参考中含照片或视频片段的请求会在视觉塔运行前被拒绝（以前图片会被放到提示词的错误位置上、悄悄被忽略）。这也是配置文件唯一无法替你自动下载的东西（自动下载只能补齐“是参数”的条目，而分词器不是），不过 TensorAgent 的目录会自己下载这三个文件。放在编码器旁边，或用 `TS_VIDEO_TOKENIZER` 指向存放它们的目录 |
| MiniMax-H3 音视频生成 | 视频 VAE（必需） | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) 里的 `minimax_h3_video_vae_fp16.safetensors`（5.21 GB）。空间 16 倍 / 时间 4 倍，解码器是纯 Transformer。放在去噪器旁或用 `--video-vae` 指定 |
| MiniMax-H3 音视频生成 | 音频 VAE（可选） | 同一目录下的 `minimax_h3_audio_vae_fp32.safetensors`（0.61 GB）。把联合生成的音频 latent 解码成 32 kHz 立体声，作为旁挂 `.wav` 写在视频边上。**不下载也照样出视频**，只是没有声音。用 `--audio-vae` 指定 |
| Wan 视频生成 | **步数蒸馏 DiT（首选）** | **这是最大的提速手段——除非要复现参考样例，都应该用它。**蒸馏 checkpoint 生成同一段视频只跑 4 次去噪，而官方配方要跑 100 次：在 M5 Pro / `ggml_metal` 上以 1088×832×121 帧实测，端到端 **17 分 30 秒**，而基础 checkpoint 是 **3 小时 30 分**——同一个请求，其他参数一律不变。TI2V-5B：[hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF](https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF)，文件 `Wan2_2-TI2V-5B-Turbo-Q8_0.gguf`（5.40 GB），另有 Q6_K（4.22 GB）、Q5_K_M（3.82 GB）、Q4_K_M（3.44 GB），最小到 Q2_K（1.86 GB）。**注意文件名里是 `Wan2_2` 下划线**，照抄基础仓库的 `Wan2.2` 写法会 404。I2V-A14B：[jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF](https://huggingface.co/jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF)，Lightning 已合并进两个专家；需同时下载 `high_noise/wan2.2_i2v_A14b_high_noise_lightx2v_4step-Q4_K_M.gguf` **和** `low_noise/wan2.2_i2v_A14b_low_noise_lightx2v_4step-Q4_K_M.gguf`（各 9.66 GB；Q8_0 15.42 GB，Q2_K 5.31 GB），放在同一个 `--local-dir` 下，`--model` 指向任意一个即可，另一个专家会自动找到。备选：[Green-Sky/FastWan2.2-TI2V-5B-FullAttn-GGUF](https://huggingface.co/Green-Sky/FastWan2.2-TI2V-5B-FullAttn-GGUF)（`FastWan2.2-TI2V-5B-q8_0.gguf`，5.41 GB）。**无需任何参数**：TensorSharp 读取 DiT 文件名，命中 `turbo` / `distill` / `lightning` / `lightx2v` / `fastwan` / `-dmd` 或显式的 `<N>steps`（1-16）即切换到该步数并关闭 guidance，加载时打印 `step-distilled checkpoint detected -> N steps, guidance off`；`--diffusion-steps` / `--cfg` 可覆盖。Turbo 与 A14B 蒸馏仓库都不含 VAE 和文本编码器，请从下面两行获取 |
| Wan 视频生成 | 基础 DiT（`--model` GGUF） | 完整官方配方（50 步 × 2 次 CFG = 100 次 DiT 前向）——需要对齐参考样例时才用，否则优先用上一行的蒸馏版本。Wan 2.2 文/图生视频：[QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF)（`Wan2.2-TI2V-5B-Q8_0.gguf` 5.40 GB 或 `Wan2.2-TI2V-5B-Q4_K_M.gguf` 3.43 GB，仓库自带 `VAE/Wan2.2_VAE.safetensors`）、[QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF) 或 [QuantStack/Wan2.2-T2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF)（`HighNoise/` 与 `LowNoise/` 两个专家缺一不可，两个仓库都自带 `VAE/Wan2.1_VAE.safetensors`）；Wan 2.1 文生视频：[samuelchristlie/Wan2.1-T2V-1.3B-GGUF](https://huggingface.co/samuelchristlie/Wan2.1-T2V-1.3B-GGUF)（`Wan2.1-T2V-1.3B-Q8_0.gguf` / `-F16.gguf`）或 [city96/Wan2.1-T2V-14B-gguf](https://huggingface.co/city96/Wan2.1-T2V-14B-gguf)（文件名为小写，如 `wan2.1-t2v-14b-Q8_0.gguf`）——这两个 2.1 仓库都不含 VAE 和编码器。`general.architecture` 为 `wan` / `wan2.1` / `wan2.2`。参见 [docs/models/wan.md](docs/models/wan.md) |
| Wan 视频生成 | UMT5-XXL 文本编码器（必需，所有 Wan checkpoint 都要） | [city96/umt5-xxl-encoder-gguf](https://huggingface.co/city96/umt5-xxl-encoder-gguf)：`umt5-xxl-encoder-Q8_0.gguf`（6.04 GB），内存紧张可用 `umt5-xxl-encoder-Q5_K_M.gguf`（4.15 GB）/ `umt5-xxl-encoder-Q4_K_M.gguf`（3.66 GB）。负责把提示词编码成条件向量，去噪开始前即从显存释放。放在 DiT 旁或用 `--video-text-encoder` / `TS_VIDEO_TEXT_ENCODER` 指定 |
| Wan 视频生成 | 视频 VAE（必需） | 把 latent 解码成画面——**用哪个由 DiT 自己决定**，不是由你选：TI2V-5B 需要 [`Wan2.2_VAE.safetensors`](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/tree/main/VAE)（TI2V-5B 仓库自带），Wan 2.1 与 A14B 需要 `Wan2.1_VAE.safetensors`——两个 QuantStack A14B 仓库里就有 `VAE/Wan2.1_VAE.safetensors`，也可单独下载 [`wan_2.1_vae.safetensors`](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors)。上面的蒸馏仓库都不含 VAE，请从这里配一个对应的。放在 DiT 旁（`VAE/` 子目录亦可）或用 `--video-vae` / `TS_VIDEO_VAE` 指定 |

### DSpark draft 模型

[DSpark](docs/models/deepseek4.md#dspark-speculative-decoding) 是 DeepSeek 的分块推测解码
draft 模型。TensorSharp 目前在 **DeepSeek V4** 上支持它，两个 GPU 引擎均可用
（`--backend cuda` 与 `--backend ggml_cuda`）：draft 是独立的 GGUF，用 `--draft-model`
加载；由主干逐块验证，贪心输出不变。

以下三个任选其一，均可直接加载（加载器兼容各发布者的张量/元数据命名）。draft 读取主干的
隐藏状态，因此与模型**同一 checkpoint 版本**的 draft 接受率更高：

| Draft | 大小 | 适配 | 说明 |
|---|---|---|---|
| [bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF](https://huggingface.co/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF) | 7.0 GB | **0731** 版本请用 `DSpark-drafter-Q2K-Q8-0731.gguf`（同仓库另有非 0731 版本） | Q2_K 专家 + Q8_0 稠密层；在 `ggml_cuda` 上接受率 66%（120 token 样本），五轮对话中每轮 66%-87% |
| [sakamakismile/DeepSeek-V4-Flash-DSpark-support-ds4-GGUF](https://huggingface.co/sakamakismile/DeepSeek-V4-Flash-DSpark-support-ds4-GGUF) | 5.6 GB | 0731 之前的 `DeepSeek-V4-Flash` | 体积最小；搭配 0731 主干接受率仍约 69%；在直接 CUDA 引擎上是三者中最快的，因为它的权重在每个推测步都要重新读取 |
| [alessandrobologna/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF](https://huggingface.co/alessandrobologna/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF) | 10.9 GB | **0731** 版本 | MXFP4 专家（对 checkpoint FP4 的无损重排）；在 `ggml_cuda` 上接受率最高（68%），但在直接 CUDA 引擎上只有 65%，且比 5.6 GB 的 draft 慢（30.9 对 34.0 tok/s）；显存占用最大——每张 GPU 上大约挤掉一整层主干 |

也可以从任何带该模块的 DeepSeek V4 checkpoint 自行转换（只需下载其三个 `mtp.*` 分片，约
11 GB）：见 [Getting a drafter](docs/models/deepseek4.md#getting-a-drafter) 与
`eng/dsv4-dspark-to-gguf.py`。

**Gemma 4 的 DSpark draft 暂不支持。** DeepSeek 也发布了 Gemma 4 的 DSpark
draft，社区亦有 GGUF 转换，但它们是另一种 draft 结构：5 层 Transformer + 对五个目标层做
`fc` 融合（`general.architecture` 为 `dspark` 或 `dflash`，block_size 7），而非 DeepSeek V4
的三个超连接块（`mtp.*`）。TensorSharp 会明确报错而不会错误加载。这里列出以便了解上游现状：

> 这套 5 层 `fc` 融合结构**已经**在 Muse-Glimmer 上实现——见
> [DFlash 投机解码](docs/models/muse-glimmer_zh-cn.md#3-dflash-投机解码)。
> 下表这些 draft 没有接入，是因为它们的编码器需要目标模型暴露逐层输入残差，而 `Gemma4Model`
> 没有暴露。目前做到这一点的是 `MuseGlimmerModel` 与 `Qwen35Model`（Qwen 3.5 / 3.8 的
> DFlash/DFlash2 draft）；`NemotronModel` 也会取出这些残差，但该主干拒绝投机解码。

| 主干 | 官方 checkpoint（safetensors） | 社区 GGUF |
|---|---|---|
| Gemma-4-12B | [deepseek-ai/dspark_gemma4_12b_block7](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7) | [ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF](https://huggingface.co/ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF)（1.9 GB）、[williamliao/dspark_gemma4_12b-GGUF](https://huggingface.co/williamliao/dspark_gemma4_12b-GGUF) |
| Gemma-4-26B-A4B | — | [williamliao/dspark_gemma4_26b-a4b-it-GGUF](https://huggingface.co/williamliao/dspark_gemma4_26b-a4b-it-GGUF) |
| Gemma-4-31B | — | [williamliao/dspark_gemma4_31b-it-GGUF](https://huggingface.co/williamliao/dspark_gemma4_31b-it-GGUF) |

Gemma 4 目前已有可用的推测解码路径：上表中的 `gemma4-assistant` MTP draft（
`--draft-model`）；Qwen 3.6、GLM 5.2 与 GLM-5.3 则内置 NextN 块（GLM-5.3 在单设备或显式 `--layer-split N` 放置下真正 draft，
不启用张量并行）。它们与 DSpark 是不同的 draft。

### 按模型下载并运行

以下命令从仓库根目录运行；请先按平台安装完整的 [.NET 10 SDK](DEVELOPMENT_zh-cn.md#安装-net-10-sdk)，再执行 `dotnet build TensorSharp.slnx -c Release`。仅安装 Runtime 无法构建下方使用的二进制文件。`hf` 来自 Hugging Face CLI（`pip install -U huggingface_hub`），所有文件都会下载到 `./models`。通用提示：单次文本提示词通过 `--input` 文件传入（`--prompt` 用于 Qwen-Image-2.1 的图像提示词，以及视频生成——MiniMax-H3 与 Wan——的提示词）；CLI 默认贪心采样，且不加 `--max-tokens` 时只生成 100 个 token；服务端固定监听 **http://localhost:5000**。按硬件把示例中的 `ggml_cuda` 换成 `ggml_metal`、`ggml_vulkan` 或 `ggml_cpu`（见 [选择后端](README_zh-cn.md#选择后端)）。

```bash
echo "列出三条关于月球的事实。" > prompt.txt
```

**DeepSeek V4.1 Flash**（384 个路由专家，服务后端为 `ggml_cuda`，直接读取 GGUF 内嵌的 Engram 元数据）：

```bash
# 修复版 Q2_K/Q5_K：十个分片共 312.349 GiB。头部与十个完整文件哈希均已校验；
# 双卡按层切分与实验性路由专家 TP2 下的有限普通/DSpark 文本/图像 HTTP 检查已通过，
# 回答以 EOS 完成，服务正常退出。另行进行的 24-token 文本/图像配对检查在两种模式下均
# 匹配 token ID 与 max_tokens 结束原因，DSpark 实际参与解码且进程正常退出；这些是有限续写。
# 两个图像配对中 DSpark 都更慢。文本预热对照已完成：每种模式一次预热、三次计时，
# 每对输入 69 token、输出 24 token，token ID 与结束原因完全一致，DSpark 实际参与且正常退出。
# 小样本与大量磁盘换页下不声明通用质量/性能、提速、跨节点执行或完整模型 TP 通过。
python3 -m venv /tmp/dsv41-tools
/tmp/dsv41-tools/bin/python -m pip install huggingface_hub
/tmp/dsv41-tools/bin/hf download smalinin/DeepSeek-V4.1-Flash-GGUF \
    --revision d1de55c19f95172c882906cc83c0e55932d26a63 \
    --include "Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-*.gguf" --local-dir models/deepseek41-q2-q5

# 继续之前，按 docs/models/deepseek41_zh-cn.md 中的十项 SHA-256
# 校验全部完整文件（使用上方实际下载目录）。
# Engram 常量已内嵌，无需另行准备。

# 可选：约 970 MB 的视觉伴随文件，--mmproj 用它启用图像与视频
/tmp/dsv41-tools/bin/python -m pip install numpy==2.0.2 gguf
/tmp/dsv41-tools/bin/python eng/dsv41-prepare-vision.py models/deepseek41-q2-q5/Q2_K-Q5 \
    --parent-model models/deepseek41-q2-q5/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf \
    --repository deepseek-ai/DeepSeek-V4.1-Flash \
    --revision dba1be0a40aa45a94ad051997016db3960a90277

dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model models/deepseek41-q2-q5/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf \
    --mmproj models/deepseek41-q2-q5/Q2_K-Q5/deepseek41.vision.gguf \
    --backend ggml_cuda --layer-split 8 --port 5000
```

`--layer-split N` 选择 N 张本地 GPU 上的整层放置。实验性 routed-MoE TP 则改用
`--tp N`。历史七分片基准比按层切分更慢，但不代表修复版量化的性能。权重与上下文放不下时加 `--n-cpu-moe N`。
Python 用于 Hugging Face 下载 CLI 与可选的视觉准备；推理无需 Python。仅提供文本服务时，
省略视觉准备命令与 `--mmproj`。[模型卡片](docs/models/deepseek41_zh-cn.md#准备可选的-dspark-伴随文件)另提供固定官方版本的 V4.1 DSpark 转换步骤及其验证限制。

`--backend cpu` 不是上面“换后端”提示里的 `ggml_cpu`：它用纯 C# 执行器 `DeepSeek4CpuExecutor`
跑完整的 V4.1 计算图——不依赖 ggml、原生库与 GPU，凡是 .NET 能跑的地方它都能跑——定位是正确性
与可移植性路径，而非服务路径；整份 checkpoint 在它上面的吞吐、加载时间与常驻内存都从未实测过。
CPU 执行器直接从 GGUF 读取 Engram。`--mmproj` 在这里完全不可用（视觉伴随文件是
原生 ggml 组件，`LoadVisionEncoder` 会抛异常），所以没有图像也没有视频；分布式 TP 组、任何草稿
模型、`--tp N`、非 `0` 的 `TS_DSV41_ENGRAM_DEVICE`，都会在
读取任何权重之前被拒绝。它也没有多轮 KV 前缀复用、没有按序列的 slot——每次分叉的对话都要重新
prefill，并发请求只能串行。

**DeepSeek V4 Flash**（284B MoE，纯文本，支持 DSpark 推测解码）：

```bash
# 约 160 GB 权重：需要多张 GPU（此处使用 --layer-split 4），draft 另需约 7 GB
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF --include "UD-Q8_K_XL/*" --local-dir models
hf download bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF DSpark-drafter-Q2K-Q8-0731.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model models/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf \
    --backend ggml_cuda --layer-split 4 --draft-model models/DSpark-drafter-Q2K-Q8-0731.gguf \
    --input prompt.txt --max-tokens 200 --temperature 0
```

去掉 `--draft-model` 即为普通解码。推测解码在任何采样设置下都无损：每个 verify 行都用本次运行自己的采样器抽取；
这里的 `--temperature 0` 只是为了让输出可复现；`--spec-pmin` 控制每个块草拟到多深。

**GLM 5.x**（GLM-5.3，`glm-dsa`，256 个路由专家，仅文本，内嵌 NextN 块）：

```bash
# UD-Q2_K_XL 是七个分片、236.4 GiB——需要一台*合计*显存能装下它再加 KV cache 的机器；
# 实测配置为八张 46 GB A40，按层切分
hf download unsloth/GLM-5.3-GGUF --include "UD-Q2_K_XL/*" --local-dir models

dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model models/UD-Q2_K_XL/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf \
    --backend ggml_cuda --layer-split 8 --spec --port 5000
```

这里没有 `--mmproj`：该仓库在任何量化下都没有发布 mmproj，而给 `glm-dsa` 模型传 `--mmproj`
只会告警并被忽略，不会报错退出，所以无论如何都是仅文本的运行。`--spec` 必须在加载之前就写在
命令行上，并且在**单设备或按层切分模式**下生效——用 `--layer-split N` 选择 N 张本地 GPU。
`--tp N` 在 GGML GPU 后端上可以接受，但仅限本地单进程（`--tp-node-id` / `--tp-peers` 对整个
GLM 系列都会在构建模型之前被拒绝），而且会在每个 rank 上复制一份 KV cache；它对 GLM-5.3 并不是
已验证的配置，并且在它之下加载器会放弃 draft——`blk.78` 借用的主干 LM head 是按列切分的——
转为普通解码。GLM-5.2 与 GLM-5.3-Flash 的下载方式相同，仓库见上表。详见
[glm](docs/models/glm_zh-cn.md#glm-53glm-dsa)。

**Gemma 4**（文本 + 图像/视频/音频、思维链、工具、可选 MTP）：

```bash
hf download ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q8_0.gguf --local-dir models
hf download ggml-org/gemma-4-E4B-it-GGUF mmproj-gemma-4-E4B-it-Q8_0.gguf --local-dir models
hf download AtomicChat/gemma-4-E4B-it-assistant-GGUF gemma-4-E4B-it-assistant.Q8_0.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gemma-4-E4B-it-Q8_0.gguf --mmproj models/mmproj-gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/gemma-4-E4B-it-Q8_0.gguf --mmproj models/mmproj-gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --draft-model models/gemma-4-E4B-it-assistant.Q8_0.gguf
```

第三个下载与 `--draft-model` 参数可省略——它们启用 MTP 推测解码，两个宿主都支持；想在 CLI 上使用，给 CLI 那一行加上同一参数即可。

**Qwen 3.5 / 3.6**（图像、思维链、工具；3.6 可用 NextN）：

```bash
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf --local-dir models
hf download unsloth/Qwen3.5-9B-GGUF mmproj-F16.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/Qwen3.5-9B-UD-Q4_K_XL.gguf --mmproj models/mmproj-F16.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/Qwen3.5-9B-UD-Q4_K_XL.gguf --mmproj models/mmproj-F16.gguf --backend ggml_cuda

# 3.6 必须从保留 NextN 块的 -MTP- 仓库下载；--spec 在 CLI 与服务端上都可用
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --local-dir models
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --backend ggml_cuda --spec
```

**GPT OSS**（文本、始终思考、工具）：

```bash
hf download ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gpt-oss-20b-MXFP4.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/gpt-oss-20b-MXFP4.gguf --backend ggml_cuda
```

**Nemotron-H**（文本、思维链、工具）：

```bash
hf download bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --backend ggml_cuda
```

图像输入请改用 Omni 发行版：从 [unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF) 下载 `NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf` 与 `mmproj-BF16.gguf`；除非加载了由 NVIDIA 检查点转换得到的音频配套 GGUF，音频输入会被拒绝：这些 GGUF 不带 Parakeet/FastConformer 音频塔（见 [nemotron_zh-cn.md §4.6-4.7](docs/models/nemotron_zh-cn.md)）。

**Hunyuan Dense**（腾讯稠密 Hunyuan / Hy-MT2，仅文本，单设备）：

这里不固定任何仓库：只要 GGUF 的 `general.architecture` 是 `hunyuan-dense` 就能加载，
架构由该键决定，与文件名无关。

```bash
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model models/<your-hunyuan-dense>.gguf \
    --backend ggml_cuda --input prompt.txt --max-tokens 200
```

没有投影器，没有草稿模型；多余的 GPU 会闲置，启动时会明确提示而不是悄悄占用。

**Mistral 3**（文本 + 图像）：

```bash
hf download bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --local-dir models
hf download bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --mmproj models/mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --mmproj models/mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf --backend ggml_cuda
```

**DiffusionGemma**（块文本扩散，文本 + 图像输入）：

```bash
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF diffusiongemma-26B-A4B-it-Q4_K_M.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --input prompt.txt --max-tokens 256 --diffusion-steps 48 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --backend ggml_cuda
```

（Web UI 会实时流式展示 DiffusionGemma 的去噪过程；兼容 API 只返回最终文本。）

需要图像输入时再加上视觉塔：所有已发布的 GGUF 都只含文本部分，因此 TensorSharp 直接从上游 Hugging Face 分片读取 Gemma-4 视觉塔。两个宿主上都用 `--mmproj` 指定（这个系列没有投影器自动探测）。音频会被拒绝；没有视频路径：OpenAI 的 `video_url` 会被拒绝，Web UI 上传的视频只以抽出的帧（当作普通图像）送入模型。

```bash
hf download google/diffusiongemma-26B-A4B-it model-00011-of-00011.safetensors --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --mmproj models/model-00011-of-00011.safetensors --image photo.png --input prompt.txt --max-tokens 256 --backend ggml_cuda
```

**Qwen-Image-2.1**（提示词 → 图像，或提示词 + 一张或多张参考图 → 编辑后的图像；需要 DiT、专用 2.1 VAE 与 Qwen3-VL-8B 文本编码器，编辑时还需要它的 mmproj）：

最短路径是现成的配置文件：它固定修订版本与 SHA-256 校验，缺什么下什么（共四个文件，约 10.29 GiB），存放到 `$TENSORSHARP_MODELS/qwen-image-2.1/`；未设置 `TENSORSHARP_MODELS` 时为 `models/qwen-image-2.1/`。配置默认选择 `ggml_metal`；在 NVIDIA 机器上追加 `--backend ggml_cuda`，或用 `--backend cpu` 走纯 C# 路径（不需要 GPU；速度与内存见[模型卡](docs/models/qwenimage21_zh-cn.md#纯-c-cpu-后端--backend-cpu)）。

```bash
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1.json \
  --prompt 'A small orange cat beside a blue ceramic vase, soft daylight, detailed photograph' \
  --width 2048 --height 2048 --diffusion-steps 40 --cfg 1 \
  --diffusion-seed 42 --output generated.png
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1.json \
  --image generated.png \
  --prompt 'Change the blue vase to a red vase. Preserve the cat, lighting and composition.' \
  --width 2048 --height 2048 --diffusion-steps 40 --cfg 1 \
  --diffusion-seed 42 --output edited.png
dotnet run --project TensorSharp.Server.Host -c Release --no-build -- \
  --config config/qwen-image-2.1.json --host 127.0.0.1 --port 5000
```

局部编辑可增加 `--mask selection.png`，掩码尺寸须与第一张输入图一致，可选
`--mask-feather 8 --mask-crop`。Server Chat 与 TensorAgent 可在照片上选择 **Select area**
绘制选区，再对比结果或再次编辑。未选中的像素与源图尺寸保持不变。
见[掩码指南](docs/models/qwenimage21_zh-cn.md#用遮罩精确编辑局部区域)。

如需自己下载文件：

```bash
hf download Abiray/Qwen-Image-2.1-GGUF qwen_image_2.1_Q4_K_M.gguf --local-dir models
hf download Comfy-Org/Qwen-Image-2.1 vae/qwen_image_2.1_vae_bf16.safetensors --local-dir models
hf download Qwen/Qwen3-VL-8B-Instruct-GGUF Qwen3VL-8B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-8B-Instruct-F16.gguf --local-dir models
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/qwen_image_2.1_Q4_K_M.gguf --qwen-image-vae models/vae/qwen_image_2.1_vae_bf16.safetensors --qwen-image-vl models/Qwen3VL-8B-Instruct-Q4_K_M.gguf --qwen-image-mmproj models/mmproj-Qwen3VL-8B-Instruct-F16.gguf --backend ggml_cuda
```

（在 Web UI 里，不带附件的提示词会生成图像；附加一张或多张图像即可编辑。省略设置时为 2048×2048（纯 C# `cpu` 上为 1024×1024）、40 步 Euler、CFG 1；`--width 1024 --height 1024` 是更快的草图尺寸；服务端同时给出 `--width` / `--height` 时会改变这个默认值。详见 [qwenimage21_zh-cn.md](docs/models/qwenimage21_zh-cn.md)。）

步数蒸馏 LoRA 插件（可选）能把 40 步降到 4–8 步。去掉上面 CLI 命令里的 `--diffusion-steps 40 --cfg 1`，再加上 `--lora config/lora/qwen-image-2.1-viggle-turbo.json`（默认 6 步，CFG 1）或 `--lora config/lora/qwen-image-2.1-pruna-8step.json`（8 步）：适配器在首次使用时下载，步数与 CFG 由它的配方提供。显式的 `--diffusion-steps` / `--cfg` 会覆盖配方，配方没有对应调度的步数会被拒绝。服务端在启动时接受同样的 `--lora`。全部十二个插件见 [config/README.md](config/README.md#qwen-image-21-lora-plug-ins-lora)。

**Qwen-Image-2.1-Turbo**（同一模型蒸馏为 8 步、CFG 1；[AtomicChat/Qwen-Image-2.1-Turbo-GGUF](https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF)）：`qwen-image-2.1/` 中已有 Qwen-Image-2.1 的文件时，它的配置只需下载 4.20 GB 的 AD-Q4_K Transformer。配置声明了检查点，因此 Turbo 按公布的 8 步调度采样；其他步数和步数蒸馏 LoRA 插件会被拒绝：

```bash
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1-turbo.json \
  --prompt 'a neon sign that reads "OPEN LATE", rainy night' \
  --width 1024 --height 1024 --diffusion-seed 42 --output turbo.png
```

手动下载：`hf download AtomicChat/Qwen-Image-2.1-Turbo-GGUF Qwen-Image-2.1-Turbo-AD-Q4_K.gguf --local-dir models`，然后像上面那样传入伴随文件，并加上 `--qwen-image-variant turbo`。详见 [Turbo 一节](docs/models/qwenimage21_zh-cn.md#qwen-image-21-turbo)。

**MiniMax-H3 音视频生成**（提示词 + 可选关键帧或参考 → H.264 MP4，**外加原生 32 kHz 立体声音频，在同一个打包 latent 里一起生成**）：

这里有四个网络协同工作，所以最短路径是现成的配置文件——它把四个都写好了，缺什么下什么（首次运行约 35.5 GB）：

```bash
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --config config/minimax-h3-fl2va.json
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --config config/minimax-h3-fl2va.json \
    --prompt "a red fox trotting through falling snow, cinematic" --output fox.mp4
```

`config/minimax-h3-ref2va.json` 是另一个 checkpoint：最多九个身份与外观参考——静态图、片段、音轨——
用于一个全新的镜头，而不是必须被复现的帧。FL2VA 与 Ref2VA 是**两个独立的 checkpoint，不是一个开关**，
向其中一个索要另一个的条件输入会直接报错，并在错误信息里点名你真正需要的那个文件。两份配置只有去噪器
不同（那边约 35.4 GB），下面三个网络是共用的，所以第二份配置只会下它自己的 DiT。参见
[config/README.md](config/README.md#video-generation-with-sound-minimax-h3)。文件会落到
`TENSORSHARP_MODELS` 指向的位置，或仓库根目录下的 `models/`（已被 Git 忽略）。

分词器无论走哪条路都不会自动下载：文本编码器的 GGUF 不含分词器，而自动下载只能补齐“是参数”的条目。
三个文件都放在编码器旁边；`tokenizer_config.json` 是唯一定义视觉标记的那个，缺了它文生视频照样能跑，
但带关键帧、或参考中含照片或视频片段的请求会被拒绝：

```bash
curl -L -o models/vocab.json https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/42ed227ee7df40d41602854ae760620d6eb651fe/processor/vocab.json
curl -L -o models/merges.txt https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/42ed227ee7df40d41602854ae760620d6eb651fe/processor/merges.txt
curl -L -o models/tokenizer_config.json https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/42ed227ee7df40d41602854ae760620d6eb651fe/processor/tokenizer_config.json
```

手动路线如下。

```bash
# FL2VA 是文生视频 / 图生视频 / 首尾帧的 checkpoint；需要参考条件时换成
# minimax_h3_ref2va_pruned-Q4_K.gguf。两个 VAE 在 unsloth/MiniMax-H3-GGUF 自己的
# vae/ 目录下也有镜像，Comfy-Org 慢的时候可以换过去。
hf download unsloth/MiniMax-H3-GGUF minimax_h3_fl2va_pruned-Q4_K.gguf --local-dir models
hf download unsloth/MiniMax-H3-GGUF qwen3vl_32b_minimax_h3-Q4_K_M.gguf --local-dir models
hf download Comfy-Org/MiniMax-H3 vae/minimax_h3_video_vae_fp16.safetensors --local-dir models
hf download Comfy-Org/MiniMax-H3 vae/minimax_h3_audio_vae_fp32.safetensors --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model models/minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_cuda \
    --video-text-encoder models/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
    --video-vae models/vae/minimax_h3_video_vae_fp16.safetensors \
    --audio-vae models/vae/minimax_h3_audio_vae_fp32.safetensors \
    --prompt "a red fox trotting through falling snow, cinematic" \
    --output fox.mp4 --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model models/minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_cuda \
    --video-text-encoder models/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
    --video-vae models/vae/minimax_h3_video_vae_fp16.safetensors \
    --audio-vae models/vae/minimax_h3_audio_vae_fp32.safetensors \
    --video-width 640 --video-height 384 --video-steps 20 --video-frames 22
```

上面这条 CLI 命令会写出 `fox.mp4` **和 `fox.wav`**：音轨是旁挂文件，从不混流进 MP4，因为混流需要一个
不一定装了的编码器——用 `ffmpeg -i fox.mp4 -i fox.wav -c:v copy -c:a aac fox_with_audio.mp4` 合并即可。
所有文件都在同一个目录下时，三个伴随文件参数都可以省略：去噪器所在目录及其上一级会被递归扫描，子目录也算。
不下载音频 VAE，或者加上 `--no-audio`，仍然会出视频——只是没有声音。

H3 是 CFG 蒸馏模型，**必须传 `--cfg 1.0`**，更高的值会被直接拒绝；管线自身的默认是 20 步，4-8 步是快速
工作点，代价是运动主体边缘会有一些彩色条纹，到 ~20 步就消失了。宽高向上取整到 32 的倍数，帧数对齐到
`17k+5` 网格（5、22、39、56、73、90……），fps 无论你传什么都被钉死在 24。服务端的步数参数叫
`--video-steps`，而且根本没有 `--cfg`——这正是随附配置两个都不设的原因。

条件输入方面：`--image first.png` 把这张图作为首帧动起来；再加上
`--end-image last.png --video-mode fl2v` 就是在首尾两帧之间插值；而在 Ref2VA checkpoint 上，`--ref-image`（可重复，最多九个）、
`--ref-video`、`--ref-video-audio` 与 `--ref-audio` 则是把身份与外观带进一个全新的镜头。在 M5 Pro 的
Metal 上以 22 帧、8 步、相同随机种子实测，H3 比 stable-diffusion.cpp 在 256×256 下快 **2.4 倍**
（49.3 秒 → 20.9 秒），在 640×384 下快 **1.7 倍**（108.5 秒 → 63.1 秒）。参见
[docs/models/minimax-h3_zh-cn.md](docs/models/minimax-h3_zh-cn.md)。

**Wan 视频生成**（提示词 + 可选首帧图片 → H.264 MP4，仅视频；需要 DiT + 视频 VAE + UMT5-XXL 文本编码器）：

Wan 同样需要三个独立的网络，所以这里最省事的路子依然是现成的配置文件——它把三者
一并列出，缺哪个就下载哪个：

```bash
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --config config/wan-video-ti2v-5b-turbo.json
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --config config/wan-video-ti2v-5b-turbo.json \
    --prompt "a cute fluffy orange cat walking through a sunny garden" --output cat.mp4
```

`config/wan-video-ti2v-5b.json` 是未蒸馏的 50 步版本，`config/wan-video-i2v-a14b.json`
是双专家的 14B 图生视频模型；详见
[config/README.md](config/README.md#video-generation-video-only-wan)。文件会落到
`TENSORSHARP_MODELS` 指向的位置，或仓库根目录下的 `models/`（已被 Git 忽略）。手动下载的方式在下面。

```bash
# 步数蒸馏的 Turbo DiT：只跑 4 次去噪而不是 100 次，由文件名自动识别。
# 注意 Turbo 文件名里的 Wan2_2 下划线；VAE 和文本编码器仍需从基础仓库获取。
hf download hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF Wan2_2-TI2V-5B-Turbo-Q8_0.gguf --local-dir models
hf download QuantStack/Wan2.2-TI2V-5B-GGUF VAE/Wan2.2_VAE.safetensors --local-dir models
hf download city96/umt5-xxl-encoder-gguf umt5-xxl-encoder-Q8_0.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model models/Wan2_2-TI2V-5B-Turbo-Q8_0.gguf --backend ggml_cuda \
    --video-vae models/VAE/Wan2.2_VAE.safetensors --video-text-encoder models/umt5-xxl-encoder-Q8_0.gguf \
    --prompt "a cute fluffy orange cat walking through a sunny garden with flowers" \
    --output cat.mp4 --width 832 --height 480 --video-frames 81
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model models/Wan2_2-TI2V-5B-Turbo-Q8_0.gguf --backend ggml_cuda \
    --video-vae models/VAE/Wan2.2_VAE.safetensors --video-text-encoder models/umt5-xxl-encoder-Q8_0.gguf \
    --video-frames 121 --fps 24
```

加载时控制台会打印 `step-distilled checkpoint detected -> 4 steps, guidance off`——看到这行就说明走的是快路径。
只把 `--model` 换成基础的 `Wan2.2-TI2V-5B-Q8_0.gguf`，就会按官方 50 步 + CFG 配方运行：同一个
1088×832×121 帧的请求实测为 3 小时 30 分，而这里是 17 分 30 秒（M5 Pro，`ggml_metal`）。
加 `--image first_frame.png` 即为图生视频，Web UI 里上传图片也一样（该图作为首帧）；服务端的
`--video-frames` / `--fps` 只是默认值，单个请求可以覆盖。Wan 不支持 `--backend mlx`，
请使用 `ggml_cuda`、`ggml_metal`、`ggml_vulkan`、`ggml_cpu`、`cuda` 或 `cpu`。

三个文件放在同一个目录下时（`VAE/` 子目录也算），`--video-vae` / `--video-text-encoder` 可以省略，会自动解析。
双专家的 A14B 模型需要**同时**下载两个专家到同一个 `--local-dir`，`--model` 指向其中任意一个：

```bash
hf download jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF high_noise/wan2.2_i2v_A14b_high_noise_lightx2v_4step-Q4_K_M.gguf --local-dir models
hf download jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF low_noise/wan2.2_i2v_A14b_low_noise_lightx2v_4step-Q4_K_M.gguf --local-dir models
hf download QuantStack/Wan2.2-I2V-A14B-GGUF VAE/Wan2.1_VAE.safetensors --local-dir models
hf download city96/umt5-xxl-encoder-gguf umt5-xxl-encoder-Q8_0.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model models/high_noise/wan2.2_i2v_A14b_high_noise_lightx2v_4step-Q4_K_M.gguf \
    --backend ggml_cuda --video-vae models/VAE/Wan2.1_VAE.safetensors \
    --video-text-encoder models/umt5-xxl-encoder-Q8_0.gguf \
    --prompt "the ship sails into the storm, waves crashing" --image ship.jpg --output ship.mp4
```
