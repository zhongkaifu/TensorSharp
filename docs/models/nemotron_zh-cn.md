# Nemotron-H

[← 返回模型索引](README_zh-cn.md) | [English](nemotron.md)

| 属性 | 值 |
|---|---|
| 提供方 | NVIDIA |
| GGUF 架构标识 | `nemotron_h`、`nemotron_h_moe` |
| 模型类 | [`NemotronModel`](../../TensorSharp.Models/Models/Nemotron/NemotronModel.cs)（旧单序列路径）+ [`NemotronModel.BatchedForward.cs`](../../TensorSharp.Models/Models/Nemotron/NemotronModel.BatchedForward.cs)（`IBatchedPagedModel`） |
| 视觉编码器 | [`NemotronVisionEncoder`](../../TensorSharp.Models/Models/Nemotron/NemotronVisionEncoder.cs)（RADIO / v2_vl ViT） |
| 图像处理器 | [`NemotronImageProcessor`](../../TensorSharp.Models/Models/Nemotron/NemotronImageProcessor.cs) |
| 音频前端 | [`NemotronAudioPreprocessor`](../../TensorSharp.Models/Models/Nemotron/NemotronAudioPreprocessor.cs)（Parakeet 风格 log-mel） |
| 音频编码器 | [`NemotronAudioEncoder`](../../TensorSharp.Models/Models/Nemotron/NemotronAudioEncoder.cs)（Parakeet/FastConformer + 声音投影器；需要任何公开仓库都不提供的配套 GGUF，见 §4.7） |
| 示例模型 | Nemotron-H-8B-Reasoning-128K、Nemotron-H-47B-Reasoning-128K、Nemotron 3 Nano Omni |
| 模态 | 文本、图像（Omni 版本配合 `mmproj`）。只有加载了带 Parakeet 音频塔的配套 GGUF 时才支持音频（§4.7）；否则音频会被**拒绝**（HTTP 400 / CLI 错误，消息为 `NemotronModel.AudioInputUnsupportedMessage`）：公开的 Omni GGUF 不带音频塔，`mmproj` 里只有 RADIO 视觉塔（见 §4.6）。 |
| 思维链模式 | 是（`<think> ... </think>`） |
| 工具调用 | 是（`<tool_call>{...}</tool_call>`） |
| 批处理 / 分页前向 | **默认启用** —— 设置 `TS_NEMOTRON_BATCHED=0` 可强制走旧的按序列 KV-swap 路径用于 A/B 对比。每槽位 Mamba2 conv + SSM 状态池，注意力层使用分页 K/V。可选的原生批处理 Mamba2 步内核（`TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1`）。详见 §11。 |
| 输出解析器 | `ChatMlOutputParser` |

## 下载

已验证的 GGUF 下载指引：

| 模型 | HF 仓库 | 推荐文件 | mmproj |
|---|---|---|---|
| Nemotron-H-8B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF) | `nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf`（4.983 GB）或 `nvidia_Nemotron-H-8B-Reasoning-128K-Q8_0.gguf`（8.620 GB） | —（仅文本） |
| Nemotron-H-47B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF) | `nvidia_Nemotron-H-47B-Reasoning-128K-Q4_K_M.gguf`（28.188 GB） | —（仅文本） |
| Nemotron 3 Nano Omni 30B-A3B | [unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF) | `NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf`（23.927 GB） | `mmproj-BF16.gguf`（1.590 GB；同仓库）—— **图像输入必需** |

Omni 的 `mmproj` 只启用**图像输入**：它只包含 32 层 RADIO 视觉塔
（`v.blk.*`、`v.patch_embd`、`v.position_embd`）和 `nemotron_v2_vl` 的 MLP
投影器（`mm.model.mlp.*`），元数据只有 `clip.has_vision_encoder`，没有任何音频
项；语言模型 GGUF 也只有 `<so_embedding>` 占位 token。因此只加载这些文件时，音频附件会在
请求层被**拒绝**（HTTP 400），而不是被解码后丢弃（见 §4.6）；音频需要单独准备的配套 GGUF（§4.7）。

这些转换仓库将 NVIDIA 对应的 Nemotron 仓库标记为上游。上游模型卡使用
NVIDIA 特定条款（`other`）；两个 bartowski 转换仓库未声明许可证。再分发前请阅读 NVIDIA 基础模型条款。

命令行下载（每个文件一行；需要先 `pip install -U huggingface_hub`）：

```bash
python -m pip install -U huggingface_hub
hf download bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --local-dir models
hf download bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF nvidia_Nemotron-H-47B-Reasoning-128K-Q4_K_M.gguf --local-dir models
hf download unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf --local-dir models
hf download unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF mmproj-BF16.gguf --local-dir models
```

CLI 单次推理（只给 `--image` 而不给 `--input` 时会使用默认的描述图片提示词；
CLI 采样默认为 greedy，`--max-tokens` 默认为 100）：

```bash
dotnet run --project TensorSharp.Cli -c Release -- --model models/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf \
  --mmproj models/mmproj-BF16.gguf \
  --image photo.png --max-tokens 512 --backend ggml_cuda
```

服务端（聊天 Web UI 以及 OpenAI/Ollama 兼容 API，位于 `http://localhost:5000`）：

```bash
dotnet run --project TensorSharp.Server.Host -c Release -- --model models/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf \
  --mmproj models/mmproj-BF16.gguf --backend ggml_cuda --max-tokens 4096
```

## 1. 来源与目标

Nemotron-H 是 NVIDIA 的混合 **Mamba2 + Transformer** 系列。同一套 backbone 同时覆盖密集 `nemotron_h` 系（如 Nemotron-H-8B / 47B）与 MoE `nemotron_h_moe` 系。Omni 发布（Nemotron 3 Nano Omni）额外携带 RADIO / v2_vl 视觉编码器（通过 `mmproj` 提供）。TensorSharp 还为 Omni 系实现了 Parakeet 风格的音频预处理器及其音频塔（24 层 Parakeet/FastConformer 编码器加声音投影器，`NemotronAudioEncoder`），但这些权重不在任何公开 GGUF 里：只用公开文件时图像是唯一实际可用的额外模态，音频会被拒绝（见 §4.6）；由 NVIDIA 检查点转换得到的配套 GGUF 可以启用音频（见 §4.7）。

它的核心特征：

- **同一个堆栈中三种层类型** —— Mamba2 SSM 层、纯注意力层、纯 FFN 层（可选 MoE）。每层类型由 GGUF 元数据数组决定：`head_count_kv[l]` 与 `feed_forward_length[l]` 一起决定层类型。
- **Mamba2 SSM** —— 选择性状态空间模型，分组 head。conv 状态与 SSM 状态按层、按 token 维护。
- **没有 RoPE** —— 注意力层不带位置编码。位置信息由 SSM 状态隐式承载。
- **ReLU² FFN** 激活（`max(0, x)²`）。
- **Sigmoid 路由 MoE** —— 用 per-expert sigmoid 概率（可选加性 bias）做 expert 选择，TopK，可选归一化 / 缩放。
- **可选 latent bottleneck 与 shared expert** —— MoE FFN 可有 in / out latent bottleneck（`ffn_latent_in` / `ffn_latent_out`）和并行的 SwiGLU 共享分支（`ffn_up_shexp` / `ffn_down_shexp`）。
- **Decode CPU 卸载阈值** —— 小 decode 算子（RMSNorm、residual add、小 matmul）可以放在 CPU 跑以避开 per-dispatch GPU 开销；大 matmul（SSM in/out、attention QKV/output、LM head）留在 GPU。

## 2. 模型架构

```
                tokens (int[])
                      │
              token_embd.weight
                      │
        [可选] InjectVisionEmbeddings (image)
                      │
        ┌──── × NumLayers ───────────────────────────────┐
        │ 根据 (head_count_kv[l], feed_forward_length[l])│
        │ 选择层类型:                                    │
        │     • Mamba2:    kv == 0  AND  ff == 0         │
        │     • Attention: kv  > 0  AND  ff == 0         │
        │     • FFN:                       ff  > 0       │
        │                                                │
        │ Mamba2:                                        │
        │   RMSNorm(attn_norm) ─► ssm_in ─► z, xBC, dt  │
        │   conv1d(xBC) ─► SiLU                          │
        │   SSM scan(dt, A, B, C, state) ─► y            │
        │   SiLU(z) * GroupRMSNorm(y) ─► ssm_out         │
        │   residual += ssm_out                          │
        │                                                │
        │ Attention:                                     │
        │   RMSNorm(attn_norm) ─► QKV (融合)            │
        │   attention (无 RoPE)                          │
        │   attn_output ─► residual                      │
        │                                                │
        │ FFN (dense):                                   │
        │   RMSNorm(attn_norm) ─► up ─► ReLU² ─► down   │
        │   residual += ffn_out                          │
        │                                                │
        │ FFN (MoE):                                     │
        │   RMSNorm(attn_norm)                           │
        │   [可选 latent_in]                             │
        │   route(sigmoid(logits) + bias) ─► topK        │
        │   每个 expert:                                 │
        │       up ─► ReLU² ─► down ─► weighted Σ        │
        │   [可选 latent_out]                            │
        │   [可选 + shared_expert]                       │
        │   residual += moe_out                          │
        └────────────────────────────────────────────────┘
                      │
              RMSNorm(output_norm)
                      │
              LM head (output.weight)
                      │
                      ▼
                   logits
```

## 3. 前向计算图

每层 L 按 `_layerTypes[l]` 派发：

```
if Mamba2:
  hidden ─► RMSNorm(attn_norm)
         ─► ssm_in.weight matmul ─► 拆为 z, xBC, dt
         ─► Conv1dStep(xBC) using ssm_conv1d.weight (state in _convState[l])
         ─► SiLU
         ─► SSM scan step:
                state ← state * exp(-softplus(dt) * A) + B * x * dt
                y     ← state @ C  (+ x * D if D present)
         ─► out = SiLU(z) * GroupRMSNorm(y, ssm_norm.weight)
         ─► out matmul ssm_out.weight → o
         ─► residual = hidden + o

if Attention:
  hidden ─► RMSNorm(attn_norm)
         ─► attn_qkv.weight matmul ─► 拆 Q, K, V (per-layer head 数)
         ─► append (K, V) 到 KV cache
         ─► attention(Q, KCache, VCache, scale = _attentionScale or 1/sqrt(headDim))
         ─► attn_output.weight matmul → o
         ─► residual = hidden + o

if FFN (dense):
  hidden ─► RMSNorm(attn_norm)
         ─► ffn_up.weight matmul → x
         ─► ReluSquaredInPlace(x)            # x ← max(0, x)²
         ─► ffn_down.weight × x → o
         ─► residual = hidden + o

if FFN (MoE):
  hidden ─► RMSNorm(attn_norm)
         ─► [if HasLatentIn] ffn_latent_in × hidden → latent
         ─► logits = ffn_gate_inp × (hidden or latent)
         ─► probs  = sigmoid(logits) + exp_probs_b      # element-wise
         ─► weights, idx = topK(probs)
         ─► if expert_weights_norm: weights /= sum(weights)
         ─► weights *= expert_weights_scale
         ─► 对每个被选 expert e:
                 up   = ffn_up_exps[e]   × (hidden or latent) → x
                 ReluSquaredInPlace(x)
                 down = ffn_down_exps[e] × x
                 accum += weights[e] * down
         ─► [if HasLatentOut] accum = ffn_latent_out × accum
         ─► [if HasSharedExperts] accum += SiLU(...)·... using ffn_up_shexp / ffn_down_shexp
         ─► residual = hidden + accum
```

所有层完成后：

```
hidden ─► narrow(seq_len-1) if prefill
       ─► RMSNorm(output_norm)
       ─► output.weight matmul → logits
       ─► 拷贝到 float[VocabSize]
```

## 4. 组件细节

### 4.1 Mamba2 SSM 层

Mamba2 块是计算上最特化的组件。`ssm_in.weight` 投影输出三个 stream 的 concat：

- **z** —— gate，维度 `_ssmDInner`。
- **xBC** —— 联合的 `(x, B, C)` stream，维度 `_ssmDInner + 2 * _ssmNGroup * _ssmDState`。
- **dt** —— 选择性 timestep，维度 `_ssmNHead`。

`xBC` 接着走 conv1d 滑动窗口（kernel size `_ssmDConv`，状态存于 `_convState[l]`）。SiLU 之后 SSM scan step 计算：

```
ΔA   = exp( -softplus(dt + ssm_dt.bias) * A )           # A ∈ ℝ^{nHead}
ΔB   = (B * x * dt)                                      # B ∈ ℝ^{nGroup × dState}
state = ΔA · state + ΔB                                  # state ∈ ℝ^{dState × headDim × nHead}
y    = state · C                                         # C ∈ ℝ^{nGroup × dState}
y   += x * ssm_d                                         # 当 `ssm_d` 存在时
```

`Mamba2SSMStepSIMD` 在 decode 时用纯 SIMD 向量化的 C# 完成（一次一个 token）。`GroupRMSNorm(y, ssm_norm.weight)` 随后做 per-group 归一，再走 gate `SiLU(z) * y`。最后 `ssm_out` 投影回 hidden 维度。

### 4.2 Attention 层

- 标准 GQA。Per-layer head 数 `_layerNumHeads[l]` 与 KV head 数 `_layerNumKVHeads[l]` 从 GGUF 数组读取。
- 当 `attn_qkv.weight` 存在时使用融合 QKV，否则构造函数的 `FuseQKVWeights()` 由分散的 `attn_q/k/v.weight` 构建融合 tensor。
- Attention scale 是 `nemotron_h.attention.scale`（如设置），否则 `1/sqrt(headDim)`。
- **没有 RoPE** —— SSM 状态隐式携带位置信息，所以 attention 不需要位置编码。

### 4.3 FFN 层（dense）

`ReluSquaredInPlace(x)` 对 `x ← max(0, x)²` 做 SIMD 向量化原地实现。Dense FFN 单次 `ffn_up` matmul 后接 ReLU² 与 `ffn_down`。

### 4.4 FFN 层（MoE）

- **Routing**：`sigmoid(logits) + exp_probs_b`（per-expert 加性 bias）→ TopK → 可选归一化 → 可选全局 scale。
- **Latent bottleneck**（可选）：`ffn_latent_in` 把 hidden 投影到更小的 `latentDim`，experts 在 latent 空间运行，然后 `ffn_latent_out` 投回 hidden。
- **Shared experts**（可选）：`ffn_up_shexp` 与 `ffn_down_shexp` 构成的并行 SwiGLU 分支。
- **批量 MoE GPU 派发**：GGML 后端（Metal / CUDA）上一次 `MoEExpertsForward` 在单张 GGML 图调度中处理所有被选 expert。预缓存的 `QuantizedWeight` 引用（`_expertUpQW`、`_expertDownQW`）和预分配 `IntPtr[]` 数组（`_moeUpPtrs`、`_moeDownPtrs`）避免字典查询与 per-token 分配。

### 4.5 视觉编码器（`NemotronVisionEncoder`）

镜像 NVIDIA RADIO / CLIP 风格 ViT，对应 v2_vl projector：

- **Linear patch embedding**，可选 bias。
- **Position embedding** 存为 `[hidden, posTokens]` 网格，源 patch 网格不同时按 align-corners=false 的双线性 resize。Resize 后的 embedding 按 `(gridW, gridH)` 在 `_resizedPositionEmbeddings` 缓存。
- **Class embedding**（可选）前置到序列。
- **32 个 encoder block**（Nemotron Omni 默认）：LayerNorm → 融合 QKV self-attention → residual → LayerNorm → up linear → GELU → down linear → residual。
- encoder 之后 class token 被剥离，按 `scaleFactor` pixel-shuffle 降低空间分辨率。
- projector（RMSNorm → Linear → ReLU² → Linear）输出 LM hidden 维度的 embedding。

`NemotronImageProcessor`（镜像 `nemotronh.ImageProcessor`）：

1. RGBA 合成到白底。
2. 选择最匹配源图像比例的 tile 网格（最多 `maxTiles`），如有 min/max patches 元数据则走动态分辨率模式。运行时默认把 tiled 图像限制为 1 个 tile，以降低服务器图像聊天的首 token 延迟；设置 `TS_NEMOTRON_IMAGE_MAX_TILES=12`（或模型声明的最大值）可恢复完整分辨率 tiling。
3. Bicubic resize 到 `gridW * imageSize × gridH * imageSize`（或动态 patch 网格）。
4. 切成 `imageSize × imageSize` tile，channel-first `[C, H, W]`。
5. 多于 1 个 tile 时附加可选缩略 tile。
6. 用 mmproj 中的 CLIP mean/std 归一化每个 tile。

多模态注入器（`ModelMultimodalInjector.cs` 中的 `ProcessNemotronHistory`）把每张图分词为一个 `<image>` 占位符，展开成 `<img>` + N 个 token + `</img>`，对每个 tile 跑视觉编码器，拼接 per-tile embedding，并入队 `PreparedEmbeddingSpan`，让模型在 `Forward()` 之前把 embedding 拼回正确位置。

### 4.6 音频前端（`NemotronAudioPreprocessor`）

Parakeet 风格 log-mel 频谱提取（镜像 ollama 的 `process_audio.go`）：

- 单声道重采样到 16 kHz。
- 0.97 pre-emphasis。
- STFT：`n_fft = 512`、`hop = 160`、`win = 400`、center-padded constant。
- Slaney 风格 mel 滤波器组，128 bins，0..8 kHz。
- `log(power + 2⁻²⁴)`，对有效（非 padding）帧做 per-mel 均值 / 方差归一化。

聊天模板对每个上传音频文件发出一个 `<so_embedding>` token，但没有任何东西能填充它：把 log-mel 帧变成 embedding 的 Parakeet/FastConformer 编码塔及其后面的投影器都不在公开的 Nemotron 3 Nano Omni GGUF 里。unsloth 仓库的 `mmproj-BF16.gguf` 恰好 390 个张量 —— 32 个 `v.blk.*` 视觉块、`v.patch_embd` / `v.position_embd` / `v.class_embd` 以及三个 `mm.model.mlp.*` 投影器张量 —— 元数据 `clip.has_vision_encoder=true`，没有任何音频键；401 个张量的语言模型 GGUF 同样没有音频张量。上游 llama.cpp 对同一检查点的回答是 "This model does not support audio input"；那边的音频需要社区分支加上带音频塔的统一 `mmproj`。

因此除非从配套 GGUF 加载了音频塔（§4.7），TensorSharp 对该家族**拒绝**音频，而不是像以前那样解码音频、打印警告、然后当作没有音频继续生成（那样模型看到的是未填充的 `<so_embedding>`，只按文本作答）。一张表 `AudioInputSupport.UnsupportedReasonFor` 通过架构注册表按架构查找，覆盖全部别名（`nemotron_h`、`nemotron_h_moe`、`nemotron_h_omni`），并参考已加载模型的 `NemotronModel.IsAudioEncoderLoaded`，驱动每一个入口：

- `/v1/chat/completions` 与 `/v1/responses` 在写入任何上传之前扫描整个请求，对任何 `input_audio` / `audio_url` 部分（包括畸形的、以及排在图像之后的）返回 **400** `invalid_request_error`，消息为 `NemotronModel.AudioInputUnsupportedMessage`；只有图像的请求仍正常解析。
- Web UI 在打开流之前以同一消息返回 400。
- CLI 在模型加载完成后、解码音频或生成任何一轮之前拒绝 `--audio`，REPL 的 `/audio` 拒绝挂载该文件。
- `ModelMultimodalInjector.ProcessNemotronHistory` 对绕过上述关卡的调用者抛出带同一消息的 `NotSupportedException`。

`NemotronAudioRefusalTests` 覆盖这些关卡（有无已加载的音频塔两种情况），`NemotronOmniMmprojContractTests`（由 `TS_TEST_NEMOTRON_MMPROJ` 门控）钉住公开 mmproj 只含视觉塔的布局；如果将来某个发行版在该文件里带上了音频塔，它会最先失败。

### 4.7 音频塔（`NemotronAudioEncoder`，配套 GGUF）

`NemotronAudioEncoder` 运行 NVIDIA 的 Parakeet/FastConformer 编码器（下采样卷积、相对位置注意力、卷积块）以及把输出投影到语言模型隐藏维度的 `sound_projection` MLP，每段音频单独编码，因此相邻音频的填充既不会改变它的长度，也不会泄漏进它的双向注意力。它读取一个**配套 GGUF**：保留官方 `sound_encoder.encoder.*` / `sound_projection.*` 张量名，超参数放在 `nemotron.audio.*`（`general.architecture=nemotron_audio`）。没有任何公开仓库提供这样的文件；从 NVIDIA BF16 检查点提取它的转换脚本与证据一起归档在 [`docs/validation/qualification-2026-09-16/nemotron-audio-cpu`](../validation/qualification-2026-09-16/nemotron-audio-cpu/README.md)（`reference-scripts/prepare.py`；`prepare_f32.py` 写出同样的权重并设置 `nemotron.audio.compute_bf16=false`）。

加载看的是张量，而不是开关。`LoadProjectors` 把 `--mmproj` 路径交给两个塔：文件含 `v.*` 张量时才加载视觉编码器，含 `sound_projection.linear2.weight` 时才加载音频编码器（若 `TS_NEMOTRON_AUDIO_MMPROJ` 指定了文件，则改从该文件加载，这样视觉 mmproj 与音频配套文件可以同时使用）。编码器随后校验每个张量形状与 Parakeet 采样配置，投影宽度必须等于语言模型隐藏维度；残缺或不匹配的配套文件会以 `InvalidDataException` 使加载失败，而不是被继续使用。只有这之后 `IsAudioEncoderLoaded` 才会解除 §4.6 的拒绝，`ProcessNemotronHistory` 针对原始 prompt 规划每张图像和每段音频（`PlanNemotronMedia`，按模态保持附件顺序），把每个 `<so_embedding>` 展开为 `<so_start>` + N + `<so_end>`，并排队投影后的行。

已确立的内容（仅 CPU；见证据 README）：Parakeet mel 前端在六段音频上与独立参考一致（`NemotronAudioInputTests`）；编码器与投影器在 8 与 128 mel 的小型 fixture 上、F32 与 BF16 计算、托管与 GGML CPU 路径下都与官方 Transformers 模块一致（`NemotronAudioEncoderTests`）；两段音频的注入与切片排队保持精确的行以及各请求独立的保留（`NemotronAudioInjectorTests`）。在官方训练好的配套权重上，显式 F32 计算配置 8,064 个输出值全部一致；原始 BF16 计算配置仍然**失败**（容差不变下 3,794/8,064，第一处差异是第 0 层的 BF16 舍入边界）。语音回答质量、GPU 执行与延迟均未经过验证。

## 5. 参数与配置

| Key | 类型 | 含义 |
|---|---|---|
| `nemotron_h.ssm.conv_kernel` | uint32 | Mamba2 conv1d kernel size |
| `nemotron_h.ssm.inner_size` | uint32 | SSM inner 维（`nHead * headDim`） |
| `nemotron_h.ssm.state_size` | uint32 | per head SSM state 维 |
| `nemotron_h.ssm.time_step_rank` | uint32 | SSM head 数 |
| `nemotron_h.ssm.group_count` | uint32 | SSM group 数 |
| `nemotron_h.attention.head_count_kv` | uint32[] | Per-layer KV head 数（0 表示 Mamba2） |
| `nemotron_h.attention.head_count` | uint32[] | Per-layer Q head 数 |
| `nemotron_h.feed_forward_length` | uint32[] | Per-layer FFN size（0 表示无 FFN） |
| `nemotron_h.attention.scale` | float32 | Attention scale 因子（0 表示自动） |
| `nemotron_h.expert_count` | uint32 | MoE expert 数（0 表示 dense） |
| `nemotron_h.expert_used_count` | uint32 | 每 token 选用 expert 数 |
| `nemotron_h.expert_weights_norm` | bool | 是否将选中 expert 权重归一为和 1 |
| `nemotron_h.expert_weights_scale` | float32 | 应用到 expert 权重的全局 scale |

Omni 视觉 projector（`mmproj` GGUF）由 `NemotronVisionEncoder` 读取标准 `clip.vision.*` keys（`embedding_length`、`feed_forward_length`、`attention.head_count`、`block_count`、`image_size`、`patch_size`、`num_channels`、`projection_dim`、`projector.scale_factor`、`attention.layer_norm_epsilon`、`use_gelu`）。

## 6. 权重命名约定

```
token_embd.weight
output_norm.weight
output.weight

# Mamba2 层:
blk.{L}.attn_norm.weight
blk.{L}.ssm_in.weight       # [hidden → 2*dInner + 2*nGroup*dState + nHead]
blk.{L}.ssm_conv1d.weight   # [xBCSize, convKernel]
blk.{L}.ssm_conv1d.bias     # (可选)
blk.{L}.ssm_dt.bias         # [nHead]
blk.{L}.ssm_a               # [nHead]
blk.{L}.ssm_d               # [nHead] (可选)
blk.{L}.ssm_norm.weight     # group RMSNorm [dInner]
blk.{L}.ssm_out.weight      # [dInner → hidden]

# Attention 层:
blk.{L}.attn_norm.weight
blk.{L}.attn_qkv.weight     # 融合 Q+K+V (或独立 attn_q/k/v.weight)
blk.{L}.attn_output.weight

# FFN 层 (dense):
blk.{L}.attn_norm.weight
blk.{L}.ffn_up.weight
blk.{L}.ffn_down.weight

# FFN 层 (MoE):
blk.{L}.attn_norm.weight
blk.{L}.ffn_gate_inp.weight       # router [numExperts, hidden]
blk.{L}.exp_probs_b.bias          # router bias (可选)
blk.{L}.ffn_latent_in.weight      # latent bottleneck in (可选)
blk.{L}.ffn_latent_out.weight     # latent bottleneck out (可选)
blk.{L}.ffn_up_exps.{E}.weight
blk.{L}.ffn_down_exps.{E}.weight
blk.{L}.ffn_up_shexp.weight       # shared expert (可选)
blk.{L}.ffn_down_shexp.weight
```

## 7. TensorSharp 实现走读

构造函数：

1. `ParseBaseConfig()`。
2. 读取 SSM 维度（`_ssmDConv`、`_ssmDInner`、`_ssmDState`、`_ssmNHead`、`_ssmNGroup`，导出 `_ssmHeadDim`）。
3. 读取 MoE 计数与 bias / scale 标志。
4. 读取 per-layer 数组（`head_count_kv`、`head_count`、`feed_forward_length`），把每层归类为 Mamba2 / Attention / FFN。
5. `ParseTokenizer()`、`LoadWeights()`。
6. `FuseFFNWeights()`（为未来 dense FFN 融合占位），`FuseQKVWeights()`（仅 attention 层）。
7. `PrepareCudaQuantizedWeightsForInference()`。
8. `InitCaches(maxSeqLen)` 为 attention 层分配 KV cache，为 Mamba2 层分配 conv state 与 SSM state。
9. `InitMamba2Buffers()` 预分配 `_mamba2ConvOutBuf` / `_mamba2YBuf`。
10. `InitLayerInfo()` 预算 layer-prefix 字符串与 per-MoE-layer 标志（`HasLatentIn`、`HasSharedExperts`、`LatentDim`）。
11. `InitMoEBuffers()` 预分配 routing 与 expert 指针数组。

`Forward(int[] tokens)` 跑 per-op 托管循环。每层根据 `_layerTypes[l]` 派发到 `Mamba2Block` / `AttentionBlock` / `FFNBlock`。可选 CPU 卸载（`CPU_MATMUL_THRESHOLD`，Apple Silicon 默认关）让小 matmul 走托管 CPU 路径，即使后端是 GPU。

## 8. Prefill 优化

- **Per-layer 派发表**（`_layerPrefixes`、`_layerWeightNames`）避免热循环里的字符串拼接。
- **MoE prefill** 仍然 per token 迭代。每 token 用批量 MoE GPU kernel（`MoEExpertsForward`），所以一次派发跑完所有被选 expert，但 token 循环还是托管 C# —— 见下方优化机会。
- **Attention prefill** 走标准托管循环。Nemotron-H 还没有融合 prefill attention kernel，因为 attention 层没有 RoPE，得分张量也比较小（不需要 SWA 窗口的 machinery）。
- **Mamba2 prefill** 顺序处理 token（按 `seqLen` 循环）跑 SSM scan；分块并行扫描在优化清单上。
- **多模态 prefill** 支持按 prompt chunk 切片已准备好的图像 / 音频 embedding span，因此长图像 prompt 不再必须作为一个超大的 forward pass 执行。
- **多模态 warmup** 在加载 Nemotron `mmproj` 的服务器启动阶段运行一次小的视觉编码和 image-token prefill，把 Metal pipeline 初始化从第一个真实图像请求前移；设置 `TS_NEMOTRON_MULTIMODAL_WARMUP=0` 可关闭。

## 9. Decode 优化

### 批量 GPU MoE（`MoEExpertsForward`）

GGML 后端（Metal / CUDA）上 MoE FFN 层的全部被选 expert 在单次 GGML 图调度中处理：

- 预缓存的 `QuantizedWeight[]` 数组（`_expertUpQW[layer][e]`、`_expertDownQW[layer][e]`）—— init 时一次性填充。
- 预分配的 `IntPtr[] _moeUpPtrs` / `_moeDownPtrs` 数组，长度 `_numExpertsUsed`。
- 预分配可复用的结果 tensor（`_expertUpResult`、`_expertDownResult`、`_latentAccumTensor`、`_latentOutResult`）。

### 小算子的 CPU 卸载

decode 热路径上的小算子（RMSNorm、residual add、expert / router matmul）即使后端是 GPU 也可以通过 CPU SIMD kernel 执行。这避免了 per-dispatch GPU 开销（Metal 上每次 ~1 ms+）。大 matmul（SSM in/out、attention QKV/output、LM head）仍留在 GPU。阈值是 `CPU_MATMUL_THRESHOLD`（Apple Silicon 统一内存默认 0；分立 GPU 系统可调高）。

### SIMD ReLU² 与 bias add

`ReluSquaredInPlace` 与 `LinearForwardInto` 用 `System.Numerics.Vector<float>` 做 SIMD 向量化，dense FFN 在 CPU 上接近峰值向量吞吐。

### 预分配的 decode 缓冲

| Buffer | 大小 | 用途 |
|---|---|---|
| `_mamba2ConvOutBuf` | `dInner + 2 * nGroup * dState` | Conv1d 输出 + SiLU |
| `_mamba2YBuf` | `dInner` | SSM scan 输出 |
| `_moeProbs` / `_moeSelectionProbs` | `numExperts` | Router 概率 |
| `_moeTopExperts` / `_moeRouteW` | `numExpertsUsed` | 选中 expert 与权重 |
| `_moeLatentAccum` | `max(hiddenSize, latentDim)` | latent 空间累加器 |
| `_expertUpResult` / `_expertDownResult` | max expert 维 | 复用的 expert matmul 输出 |
| `_latentAccumTensor` / `_latentOutResult` | `latentDim` / `hiddenSize` | latent bottleneck 复用 tensor |

## 10. 内存与 KV cache 策略

- **Attention 层**：标准 KV cache `[numKVHeads, maxSeqLen, headDim]`。
- **Mamba2 层**：`_convState[layer]`（大小 `(convKernel - 1) * (dInner + 2 * nGroup * dState)` floats）与 `_ssmState[layer]`（大小 `dState * headDim * nHead` floats）。
- `ResetKVCache()` 同时清零三类缓存（KV cache、conv state、SSM state）。
- `SupportsKVCacheTruncation` 返回 **false**，因为 SSM 状态是顺序的，不能部分复用。Nemotron-H 因此不启用多轮 KV cache 复用 —— 服务器在轮次之间回退到完整 reset。

## 11. 批处理 / 分页前向（连续批处理）

Nemotron-H 实现了 `IBatchedPagedModel.ForwardBatch`
（[`NemotronModel.BatchedForward.cs`](../../TensorSharp.Models/Models/Nemotron/NemotronModel.BatchedForward.cs)），
默认通过共享 `InferenceEngine` 连续批处理栈执行
（[`docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md`](../PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md))
—— 只有走批处理路径才能真正让并发请求并行。设置 `TS_NEMOTRON_BATCHED=0`
可强制回到旧的按序列 KV-swap 路径用于 A/B 对比。

Nemotron-H 是所有批处理移植里最复杂的，因为它结合了**三种不同的层类型**
（Mamba2 SSM、纯注意力、FFN 密集 / MoE），且 Mamba2 是递归（per-sequence
状态）。批处理路径因此需要两种正交的缓存：

### 注意力层 —— 分页 K/V

- 与 Mistral 3 同构布局（`[numBlocks * blockSize * numKvHeads * headDim]`，
  每层一份）。
- 无 RoPE —— Nemotron-H 注意力层不带位置编码。
- 按序列的注意力派发使用 `ManagedPagedAttention.Forward`（纯 C# 在线
  softmax 内核）作为正确性参考；同时通过 `GgmlBasicOps` 接入了原生分页
  内核路径。
- 原生主机数组内核（`TSGgml_PagedAttentionForward`）按形状桶缓存一份计算图和
  后端缓冲，并把 K/V 填充到桶长。构建会话时现在会清零该缓冲：后端缓冲本身不保证为零，
  而 CUDA flash attention 仍会读取被 `-inf` 掩码的填充 key，因此被释放缓冲留下的
  NaN（例如一个长提示结束之后）会让下一次批处理 prefill 的每一行都变成 NaN。
  Nemotron 3.5 随后在 `TryMoEPrefillBatchedByExpert` 中抛出
  `IndexOutOfRangeException`（NaN 行没有 top-k），整个 4 序列步失败。
  现在 MoE 路由器会按层和行号报告非有限值的行。

### Mamba2 层 —— 每槽位 conv + SSM 状态池

每个序列的递归状态落在一个 slot 中，slot 由其**主块 id** 标识（与 vLLM 的
`state_indices_tensor` 对齐）：

- `_nemoSlotMamba2NativeDecodeProjected[layer][slot]` —— 每槽位 conv 环形
  缓冲。
- `_nemoSlotMamba2NativeDecodeHidden[layer][slot]` —— 每槽位 SSM 状态。
- `_nemoSlotMamba2NativeDecodeStateInitialized[layer][slot]` —— 初始化
  标志。

槽位在首次访问时分配，序列在引擎中被回收时释放。

**原生批处理 Mamba2 步内核** —— `TSGgml_NemotronMamba2BatchedStepF32`
（[`ggml_ops_mamba2.cpp`](../../TensorSharp.GGML.Native/ggml_ops_mamba2.cpp)）
—— 通过 `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1` 控制。使用 NEON SIMD + GCD
按 head 并行（结构与 Qwen 3.5 的批处理 GDN 内核一致），把 N 次 C#
`Mamba2Block` 调用替换为一次原生派发加批处理的 `ssm_in` / `ssm_out` 投
影。通过 `GgmlBasicOps.NemotronMamba` 暴露。

### FFN —— 密集与 MoE

- 密集 FFN 在批处理 token 轴上跑 token-parallel ReLU²（`up → ReLU² → down`），
  每个投影一次 matmul。
- MoE FFN 通过已有的 `MoEForward` token-parallel router + 逐 token expert
  派发执行（目前没有 Nemotron-H 专用的批处理 MoE 内核）。

### 批处理路径下的多模态

视觉与音频嵌入通过与旧 forward 相同的逐行 `InjectMultimodalEmbeddings`
路径，直接注入批处理的 `[numTokens, hidden]` tensor。`SupportsBatchedMultimodal`
在批处理路径处于启用状态时（即未设置 `TS_NEMOTRON_BATCHED=0`）返回 true。

### 已验证的正确性与吞吐

- 文本 prompt 上与旧路径**100% 贪心一致**
  （[`NemotronBatchedCorrectnessTests`](../../InferenceWeb.Tests/NemotronBatchedCorrectnessTests.cs)）。
- 多模态 prompt 的正确性已被结构性验证（在移除多模态预检拒绝后纯文本仍
  100%），但缺少本地 audio/image fixture 用于端到端验证。

**NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-IQ2_XXS（Apple M4 Pro、
GgmlMetal、进程内 legacy-vs-batched 切换；详见
[`NemotronBatchedPerfBench`](../../InferenceWeb.Tests/NemotronBatchedPerfBench.cs)）
吞吐**：

| 路径 | n=1 | n=3 | n=5 |
|---|---|---|---|
| 批处理 + 托管 Mamba2 步 | 1.94× | 3.43× | 1.67× |
| 批处理 + 原生 Mamba2 步（`*_NATIVE=1`） | 0.75× | **3.95×** | 2.93× |

`n=1` 是唯一回退：批处理脚手架开销在单序列 decode 上盖过收益。`n=2` 起批
处理路径全面胜出。`TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1` 在多批 decode 上
进一步扩大胜势，因为它把 C# Mamba2 内层循环替换为原生 NEON 内核。

移植过程中还修了一个潜伏 bug：`s_nemoBatchedOptIn` 原本是 `static readonly`，
在 class-load 时捕获环境变量 —— 测试在运行时设置 `TS_NEMOTRON_BATCHED=1`
实际无法切换路径。现在改为方法 getter（与 Qwen 3.5 的写法一致）。

### 投机解码被拒绝

Nemotron-H 不做投机解码：`--draft-model` 不会挂载 DSpark/DFlash 草稿器（Nemotron 3.5
Lightning 的 `NVFP4-DSpark` GGUF 会被识别并报告“未挂载”；服务器在 `--draft-model` 指定它时会在启动阶段失败并提示去掉该参数），`--spec` 或
`--spec-type ngram` 也只提供普通解码，并打印一次警告。原因是正确性：投机输出必须与普通贪心解码一致，
而在这个主干上，多 token verify 与单 token decode 使用不同的注意力内核（基于展开缓存的主机端注意力 vs.
flash-attention decode 内核）和不同的 MoE 内核（按专家批处理 vs. 逐 token 内核）。在 `nemotron_h_moe`
（A40，`ggml_cuda`）上实测：单行投机步与 `Forward` 的 logits 相差 0.16-1.1，verify 各行相差 0.2-0.8，
足以翻转低置信度的贪心选择（2026-09-16 验证中每个单序列 DSpark 请求都发生了偏离）。Mamba-2 的快照 / 回滚是精确的。
把注意力和 MoE 逐行运行可以让 verify 精确，但 4 行需要 116 ms，而一个 decode 步只要 28 ms（另加每个 DSpark 块 70 ms），
因此精确的 verify 无法快过普通解码。详见 [投机解码](../speculative_decoding.md#nemotron-h-refuses-speculation)。

## 12. 输出解析器与聊天模板

- `ChatMlOutputParser` 解析 `<think> ... </think>` 思维链与 `<tool_call>{...}</tool_call>` 工具调用。
- 聊天模板使用 ChatML 格式（`<|im_start|>` / `<|im_end|>`）。多模态占位符包括 `<image>`（之后展开为 `<img>` + N 个 token + `</img>`）与 `<so_embedding>`（音频）。

## 13. 优化机会

- **原生 whole-model decode** —— 旧的单序列 forward 仍跑在托管 C#。原生
  `NemotronModelDecode` 能消除单序列路径上的托管循环开销。
- **旧路径的原生 Mamba2 decode** —— `Mamba2SSMStepSIMD` 中的 SIMD 向量化
  扫描在 CPU 上已经很快，但原生 CUDA / Metal 内核能解锁完整 Mamba2 路径在
  GPU 上的执行（针对单序列路径）。批处理路径已经在
  `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1` 下提供了原生内核。
- **分块并行 SSM 扫描** —— Mamba2 prefill 顺序处理 token。分块并行扫描
  （参考 Mamba 官方 CUDA 实现）能显著降低 TTFT。
- **向量化 conv1d** —— `Mamba2Conv1dStep` 是标量循环。SIMD 或原生向量化版
  本能进一步加速 Mamba2 层。
- **Per-token MoE 批处理** —— 即使有 `MoEExpertsForward`，per-token 托管循
  环仍是外层驱动。能在单次派发处理多 token 的批量 kernel 对长 prompt 帮助
  很大。
- **音频塔** —— `NemotronAudioEncoder` 以托管 CPU 路径运行在配套 GGUF 上（§4.7）；
  公开 GGUF 仍然不带音频塔，所以除非加载了该配套文件，音频会被拒绝。待完成：训练权重上的
  BF16 计算一致性、GPU/原生编码器计算图、语音输入验收与延迟验证。
