# Gemma 4

[← 返回模型索引](README_zh-cn.md) | [English](gemma4.md)

> **想沿着一条从零构建的路线学习？** Zhongkai Fu 所著的 [From Tensors to Tokens](../BOOK_zh-cn.md) 以 Gemma 4 E4B 为例，连接张量基础、多模态模型执行与完整的 TensorSharp 推理应用。[在 Amazon 查看平装本](https://www.amazon.com/dp/B0H9P44QZZ)。

| 属性 | 值 |
|---|---|
| 提供方 | Google |
| GGUF 架构标识 | `gemma4` |
| 模型类 | 共享解码器运行时上的 [`Gemma4RuntimeModel`](../../TensorSharp.Models/Models/Gemma4/Gemma4RuntimeModel.cs)（`StandardDecoderModel`；所有执行器上的计算图程序） |
| 视觉编码器 | [`Gemma4VisionEncoder`](../../TensorSharp.Models/Models/Gemma4/Gemma4VisionEncoder.cs)（SigLIP 风格 ViT） |
| 音频编码器 | [`Gemma4AudioEncoder`](../../TensorSharp.Models/Models/Gemma4/Gemma4AudioEncoder.cs)（USM 风格 chunked transformer） |
| 音频前端 | [`Gemma4AudioPreprocessor`](../../TensorSharp.Models/Models/Gemma4/Gemma4AudioPreprocessor.cs)（16 kHz 单声道 → 128 bin log-mel） |
| 图像处理器 | [`Gemma4ImageProcessor`](../../TensorSharp.Models/Models/Gemma4/Gemma4ImageProcessor.cs) |
| 示例模型 | gemma-4-E4B（8B 等效）、gemma-4-12B、gemma-4-31B、gemma-4-26B-A4B（MoE） |
| 模态 | 文本、图像、视频（帧栈）、音频 |
| 思维链模式 | 是（`<\|channel>thought ... <channel\|>`） |
| 工具调用 | 是（`<\|tool_call>call:name{...}<tool_call\|>`） |
| 批处理 / 分页前向 | 共享运行时的连续批处理：请求在各自槽位上以批处理步骤运行（prompt 分块与 decode 行同在一步），并带前缀缓存与检查点。见 §10。 |
| MTP 投机解码 | 可选 —— 通过 `--draft-model`（`TS_SPEC_DRAFT_MODEL`）加载独立的 `gemma4-assistant` EAGLE 风格草稿 GGUF；指定该文件本身就会启用投机（显式 `--no-spec` 可否决）。该标志在**两个宿主上都可用**：`TensorSharp.Cli` 与 `TensorSharp.Server` 共用同一个 [`SpeculativeCliFlags`](../../TensorSharp.Runtime/Speculative/SpeculativeCliFlags.cs)。详在所有计算图执行器上运行。见 §12。 |
| 输出解析器 | `Gemma4OutputParser` |

## 下载

已验证的 GGUF 下载指引：

| 模型 | HF 仓库 | 推荐文件 | mmproj（图像 / 视频 / 音频） | MTP 草稿 |
|---|---|---|---|---|
| gemma-4-E4B-it | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) | `gemma-4-E4B-it-Q8_0.gguf`（8.031 GB；另有 `gemma-4-E4B-it-Q4_0.gguf`，4.588 GB） | `mmproj-gemma-4-E4B-it-Q8_0.gguf`（0.560 GB；同仓库；备选：[unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) 中的 `mmproj-F16.gguf`，0.990 GB） | 来自 [AtomicChat/gemma-4-E4B-it-assistant-GGUF](https://huggingface.co/AtomicChat/gemma-4-E4B-it-assistant-GGUF) 的 `gemma-4-E4B-it-assistant.Q8_0.gguf`（0.100 GB） |
| gemma-4-12B-it（QAT） | [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) | `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`（6.716 GB） | `mmproj-BF16.gguf`（0.175 GB；同仓库） | `mtp-gemma-4-12B-it.gguf`（0.254 GB；同仓库根目录；量化变体位于 `MTP/` 下） |
| gemma-4-26B-A4B-it（MoE，QAT） | [unsloth/gemma-4-26B-A4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF) | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`（14.249 GB） | `mmproj-BF16.gguf`（1.195 GB；同仓库） | `mtp-gemma-4-26B-A4B-it.gguf`（0.252 GB；同仓库），或来自 [AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF](https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF) 的 `gemma-4-26B-A4B-it-assistant.Q8_0.gguf`（0.462 GB） |
| gemma-4-26B-A4B-it（MoE） | [ggml-org/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF) | `gemma-4-26B-A4B-it-Q4_K_M.gguf`（16.796 GB）或 `gemma-4-26B-A4B-it-Q8_0.gguf`（26.860 GB） | `mmproj-gemma-4-26B-A4B-it-Q8_0.gguf`（0.806 GB）/ `mmproj-gemma-4-26B-A4B-it-bf16.gguf`（1.195 GB） | 与 QAT 行相同的草稿 |
| gemma-4-31B-it | [ggml-org/gemma-4-31B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF) | `gemma-4-31B-it-Q4_K_M.gguf`（18.687 GB）或 `gemma-4-31B-it-Q8_0.gguf`（32.636 GB） | `mmproj-gemma-4-31B-it-Q8_0.gguf`（0.810 GB）/ `mmproj-gemma-4-31B-it-bf16.gguf`（1.201 GB） | — |

Hugging Face 元数据将上述每一行标记为派生自对应的 Google Gemma 4 基础模型。
部分转换仓库的模型卡未声明许可证；可匿名下载不等于重新授权，
再分发前请同时阅读基础模型与转换仓库的条款。

MTP 草稿头以独立的 `gemma4-assistant` GGUF 发布，其 backbone 维度必须等于
目标模型的 hidden size —— 请始终按相同规格配对草稿与目标（E4B 草稿 ↔ E4B 目标、
12B ↔ 12B、26B-A4B ↔ 26B-A4B）。草稿不匹配会在服务端启动时快速失败（§12.2）。

命令行下载（每个文件一行；需要先 `pip install -U huggingface_hub`）：

```bash
python -m pip install -U huggingface_hub
hf download ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q8_0.gguf --local-dir models
hf download ggml-org/gemma-4-E4B-it-GGUF mmproj-gemma-4-E4B-it-Q8_0.gguf --local-dir models
hf download unsloth/gemma-4-12B-it-qat-GGUF gemma-4-12B-it-qat-UD-Q4_K_XL.gguf --local-dir models
hf download unsloth/gemma-4-12B-it-qat-GGUF mtp-gemma-4-12B-it.gguf --local-dir models
```

### 已验证的 Gemma 4 E4B 原生 GGML 快速路径

这是 TensorSharp 的快速开始路径。已验证的是 E4B Q8_0
家族与执行路径：模型推荐使用公开的
[`ggml-org/gemma-4-E4B-it-GGUF`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)
文件，完成正常原生构建后选择原生 GGML GPU 后端。这里不声称基准输入对应某个公开文件的
特定校验和。下面的可复制命令面向 Linux + NVIDIA；其他平台的后端选择见代码块之后。
全新机器请先按平台安装并验证完整的 [.NET 10 SDK](../../DEVELOPMENT_zh-cn.md#安装-net-10-sdk)；
仅安装 Runtime 无法执行构建命令。

```bash
python -m pip install -U huggingface_hub
hf download ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q8_0.gguf --local-dir models
TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON dotnet build TensorSharp.slnx -c Release -p:TensorSharpSkipMlxNative=true
printf '%s\n' '用一句简短的话回答：TensorSharp 是什么？' > prompt.txt
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gemma-4-E4B-it-Q8_0.gguf \
  --input prompt.txt --max-tokens 64 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/gemma-4-E4B-it-Q8_0.gguf \
  --backend ggml_cuda --max-tokens 128
```

Windows/Linux + NVIDIA 使用 `ggml_cuda`；Apple Silicon 使用 `ggml_metal`；
Windows/Linux 上带 Vulkan 驱动的 AMD、Intel 或 NVIDIA GPU 使用 `ggml_vulkan`。
纯文本请求不需要 `mmproj-gemma-4-E4B-it-Q8_0.gguf`；图像、视频或音频输入需要从
同一仓库下载它，并通过 `--mmproj` 传入。

每一步都以整前向计算图程序在后端执行器上运行（§7、§8）；同一宽度的 decode 步以前一步推进的方式重放。


多模态 CLI 单次推理（文本提示词通过 `--input` 从文件读取；只给 `--image` 而不给
`--input` 时会使用默认的描述图片提示词；CLI 采样默认为 greedy，
`--max-tokens` 默认为 100）：

```bash
dotnet run --project TensorSharp.Cli -c Release -- --model models/gemma-4-E4B-it-Q8_0.gguf \
  --mmproj models/mmproj-gemma-4-E4B-it-Q8_0.gguf \
  --image photo.png --max-tokens 512 --backend ggml_cuda
```

带 MTP 投机解码。`--draft-model` 由同一个
[`SpeculativeCliFlags`](../../TensorSharp.Runtime/Speculative/SpeculativeCliFlags.cs)
在**两个宿主上**解析，因此 `TensorSharp.Cli` 与 `TensorSharp.Server` 的拼写完全一致。
指定草稿 GGUF 本身就会启用投机 —— 不需要再写 `--spec`（显式 `--no-spec` 可否决）。
它必须出现在*加载*模型的那条
命令行上 —— 草稿 GGUF 在启动时挂到目标模型上，而无法激活的 `--draft-model` 会在启动时
立即失败，而不是静默地不做投机（§12.2）：

```bash
# 服务端
dotnet run --project TensorSharp.Server.Host -c Release -- --model models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  --backend ggml_cuda --draft-model models/mtp-gemma-4-12B-it.gguf

# CLI —— 投机在 --input、--input-jsonl、--multi-turn-jsonl 与 --interactive 下均会启用
dotnet run --project TensorSharp.Cli -c Release -- --model models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  --backend ggml_cuda --draft-model models/mtp-gemma-4-12B-it.gguf --input prompt.txt --max-tokens 512
```

然后打开 `http://localhost:5000` 使用聊天 UI。

## 1. 来源与目标

Gemma 4 是 TensorSharp 当前支持的功能最丰富的架构。它把 Google Gemma 系扩展到：

- **per-layer 异构性**：每一层可独立选择注意力模式（SWA / 全局）、head 维度、KV head 数，以及是否与某个早期「donor」层共享 KV。
- **Per-Layer Embedding（PLE）**：从 `per_layer_token_embd.weight` 取出的小型 side embedding 在每个 block 内混入 residual stream。
- **MoE 变体**（如 `gemma-4-26B-A4B`）：每个 block 同时跑一个密集 MLP 与一个稀疏 MoE 分支，分别经过各自的 post-norm 后求和。
- **真正的多模态**：图像、视频帧栈与音频共用同一 residual stream。

SWA mask 与 RoPE 表会跨层缓存，SWA 缓存为环形（内存与上下文长度无关），整套 transformer 在 decode 时可以一次 GGML 图调度完成。

## 2. 模型架构

```
                              tokens (int[])
                                   │
                              token_embd.weight × sqrt(hidden)
                                   │
                       [可选]  InjectVisionEmbeddings (图像 / 视频帧)
                       [可选]  InjectVisionEmbeddings (音频 embedding，同算子)
                                   │
                       [若 PLE] ComputePLE(tokens, hidden) ──► perLayerInputs
                                   │
              ┌────── × NumLayers ─────────────────────────────────────┐
              │ HeadDim, KVHeads, SWA/全局模式 per layer              │
              │  RMSNorm(attn_norm)                                    │
              │  QKV (融合，或 KV 共享层只走 Q)                         │
              │  per-head RMSNorm(Q,K) + 无权 RMSNorm(V)                │
              │  RoPE (NeoX local 或 global+比例/部分)                  │
              │  Attention（SWA 或全因果）                              │
              │  attn_output ─► RMSNorm(post_attn_norm) + 残差          │
              │                                                        │
              │  if MoE 层:                                            │
              │     RMSNorm(ffn_norm)  ─► GeGLU(密集 MLP) ─► PostNorm1  │
              │     RMSNorm(pre_ffw_norm_2) ─► MoE(route+experts)       │
              │                              ─► PostNorm2               │
              │     残差 += PostNorm(PostNorm1 + PostNorm2)             │
              │  else:                                                  │
              │     RMSNorm(ffn_norm) ─► GeGLU ─► RMSNorm(post_ffw)     │
              │     残差 += branch                                       │
              │                                                         │
              │  if PLE:                                                │
              │     残差 += proj(GELU(inp_gate(hidden)) * pleInput)     │
              │                                                         │
              │  hidden *= layer_output_scale[layer]                    │
              └─────────────────────────────────────────────────────────┘
                                   │
                              RMSNorm(output_norm)
                                   │
                              LM head (output.weight 或 tied)
                                   │
                              [可选] tanh-softcap
                                   │
                                   ▼
                                logits
```

## 3. 前向计算图

### 3.1 Per-layer（密集）

```
hidden ─► RMSNorm(attn_norm)
       ─► QKV matmul（融合权重）─► 拆为 Q [seq, qDim], K [seq, kvDim], V [seq, kvDim]
       ─► per-head RMSNorm(Q) * attn_q_norm.weight
       ─► per-head RMSNorm(K) * attn_k_norm.weight
       ─► 无权 RMSNorm(V)                              // V-norm: weight ≡ 1
       ─► RoPE(Q, K, freqs[layer])
            • local 层：标准 NeoX RoPE，全 headDim
            • global 层：NeoX RoPE 仅旋转前 _partialRotaryDims 维，使用 rope_freqs.weight 中的比例频率因子
       ─► Q ← Q * (1/sqrt(headDim))
       ─► append (K, V) 到 per-layer cache（SWA 走环形）
       ─► attention(Q, K_cache, V_cache,
                    window = slidingWindow if SWA else totalSeq)
       ─► attn_output matmul → o
       ─► RMSNorm(post_attn_norm)
       ─► residual = hidden + o
       ─► h2 = RMSNorm(ffn_norm) on residual
       ─► ffn_gate_up matmul → [gate ‖ up]
       ─► g = GELU(gate)
       ─► h3 = ffn_down × (g * up)
       ─► RMSNorm(post_ffw_norm)
       ─► residual += h3
       ─► (PLE 分支 — 见 § 4.4)
       ─► hidden = residual * layer_output_scale[layer]
```

### 3.2 Per-layer（MoE）

```
... 同上的 attention block ...

# 密集 MLP 分支
b1 = RMSNorm(ffn_norm) on residual
b1 = GeGLU(b1) using ffn_gate_up.weight + ffn_down.weight
b1 = RMSNorm(post_ffw_norm_1) on b1

# MoE 分支
b2 = RMSNorm(pre_ffw_norm_2) on residual
logits = ffn_gate_inp.weight × b2
logits = unweighted_RMSNorm(logits) * ffn_gate_inp.scale   # 学到的 scale
weights, idx = topK(softmax(logits), _numExpertsUsed)
b2 = weighted_sum_experts(SwiGLU on b2 using ffn_gate_up_exps[idx] +
                          ffn_down_exps[idx])
b2 = RMSNorm(post_ffw_norm_2) on b2

# 合并
combined = RMSNorm(post_ffw_norm)(b1 + b2)
residual += combined
```

### 3.3 Decode vs prefill

两者是同一计算图程序的不同宽度（§8）：prompt 分块注意已缓存的前缀与自身的行，decode 行在因果（或滑窗带状）掩码下注意填充后的
缓存窗口，多模态 soft token 在其前向内彼此双向注意。

## 4. 组件细节

### 4.1 per-layer 异构性

- `_slidingWindowPattern[layer]` 来自 `gemma4.attention.sliding_window_pattern`。`IsLocalLayer(layer)` 返回该项。`true` ⇒ SWA，`false` ⇒ 全因果。
- `_localHeadDim` 来自 `gemma4.attention.key_length_swa`（默认 256），`_globalHeadDim` 来自 `gemma4.attention.key_length`（默认 512）。`HeadDimForLayer(layer)` 选择具体值。
- `_numGlobalKVHeads` 来自 `gemma4.attention.global_head_count_kv`。Local 层使用 `Config.NumKVHeads`。`KVHeadsForLayer(layer)` 解析 KV head 数。
- `DetectHeadDimsFromWeights()` 在 GGUF 元数据与实际 attention 权重形状不一致时把 head dim 重新对齐。

### 4.2 KV 共享

最后 `_sharedKVLayers` 层（来自 `gemma4.attention.shared_kv_layers`）复用其他层的 KV cache。`BuildKVDonorMap()` 生成 `_kvDonorMap[layer] → donorLayer`；共享层只投影 Q，跳过 K/V matmul 与 cache 写入。共享 cache 实现：

- `_kvCacheK[shared] = _kvCacheK[donor]`（别名）。
- 分块 prefill 时，donor 当前 chunk 内刚算出的 K/V 会暂存到 `_prefillSWAKV`，让 KV 共享的 SWA 层 attend 到完整 chunk 的 K/V，而不是滚动 cache 中的不完整窗口。

### 4.3 Per-Layer Embedding（PLE）

当 `gemma4.embedding_length_per_layer_input > 0`：

```
perLayerEmbeddings = lookup(per_layer_token_embd.weight, tokens)
perLayerEmbeddings = RMSNorm(per_layer_proj_norm.weight) on perLayerEmbeddings
perLayerEmbeddings = perLayerEmbeddings × per_layer_model_proj.weight^T

# 每层内：
ple = GELU(inp_gate.weight × hidden) * extract_layer_slice(perLayerInputs, l)
ple = proj.weight × ple
ple = RMSNorm(post_norm.weight) on ple
residual += ple
```

`ComputePLE()` 在每次 forward 中跑一次完整 PLE 流水线，输出 `[seqLen, NumLayers * pleDim]`，每层在内部 `Narrow` 自己那一片。

### 4.4 RoPE 变体

- **Local 层**：标准 NeoX RoPE，全 headDim，base `_ropeLocalBase`（来自 `gemma4.rope.freq_base_swa` 或 `gemma4.rope.local.freq_base`，默认 10000）。`ApplyNeoXRoPEDecode` / `ApplyNeoXRoPEPrefill` 实现，含向量化 cos/sin 表。
- **Global 层**：partial NeoX RoPE 只作用在 headDim 的前 `_partialRotaryDims` 维（其余直通）。频率向量为标准 NeoX schedule × 来自 `rope_freqs.weight` 的 per-frequency 因子。完整 cos/sin 查表跨全局层缓存到 `_neoXRopeCos` / `_neoXRopeSin`，每个 chunk 节省 ~35M 次 `MathF.Cos`/`MathF.Sin`。

### 4.5 V-norm

V 投影后，`ApplyUnweightedRMSNorm()` 用全 1 权重张量（`_onesForVNorm`）对每个 value 向量做 RMSNorm。这是 TensorSharp 矩阵中 Gemma 4 独有的特性。

### 4.6 MoE

`HasMoE(layer)` 在 `blk.{L}.ffn_gate_inp.weight` 存在时为 true。MoE 分支：

1. `RMSNorm(pre_ffw_norm_2)` on residual。
2. router matmul → 无权 RMSNorm → 乘学到的 scale `ffn_gate_inp.scale`。
3. softmax → TopK(`_numExpertsUsed`)。
4. 每个被选中的专家：SwiGLU(`ffn_gate_up_exps.{e}.weight`, `ffn_down_exps.{e}.weight`) + 加权累加。
5. `RMSNorm(post_ffw_norm_2)`。

密集分支与 MoE 分支求和后再过 `post_ffw_norm`。

### 4.7 per-layer 输出缩放

`_layerScalars[layer]` 来自 `blk.{L}.layer_output_scale.weight`（标量 `[1]`）。每层在返回到下一层前把 hidden 输出乘以这个 scalar。

### 4.8 Logit softcap

当 `gemma4.final_logit_softcapping > 0` 时走 `tanh(logits / cap) * cap`。

### 4.9 视觉管线（图像与视频帧）

`Gemma4VisionEncoder` 是 SigLIP 风格 ViT：2D 位置 embedding、门控 GELU-tanh MLP（`gelu_pytorch_tanh`，与视觉配置声明一致；融合原生 block 与托管回退路径都使用它，早期构建用的是 QuickGELU），以及到 LM hidden 维的池化投影：先在视觉 hidden 维上做无权 RMSNorm（`embedding_pre_projection_norm`），再做线性投影。早期构建是在投影之后才做归一化；现在的顺序与上游 `Gemma4MultimodalEmbedder` 一致。

视觉塔可以从 mmproj GGUF 加载，也可以通过同一个 `--mmproj` 选项，从 HuggingFace `.safetensors` 分片加载——分片中的塔位于 `model.encoder.vision_tower.*`，投影器为 `model.encoder.embed_vision.embedding_projection.weight`（[`Gemma4VisionEncoder.Safetensors.cs`](../../TensorSharp.Models/Models/Gemma4/Gemma4VisionEncoder.Safetensors.cs)）。几何尺寸由张量形状推出；池化因子与 RMSNorm epsilon 取自分片旁 `config.json` 中宽度与该塔一致的 `vision_config`，否则使用 Gemma 4 默认值并打印一次警告。这条路径是为没有发布 mmproj 的检查点准备的（最初就是为 [DiffusionGemma](diffusiongemma_zh-cn.md) 编写的）。分片里没有音频塔，因此音频输入仍需要 mmproj GGUF。

视频帧按时间采样从 MP4 中提取（默认通过 OpenCV 每秒一帧，可用 `VIDEO_SAMPLE_FPS` 调整；`VIDEO_MAX_FRAMES` 可额外设置帧数上限），每帧独立编码，最终的 embedding 串联起来按顺序注入到连续的 `<|image>` 占位上。

静态图像的尺寸处理取决于投影器类型。`gemma4v`（SigLIP）视觉塔（例如 E4B 或 DiffusionGemma 的塔）采用参考实现 `Gemma4ImageProcessor` 的尺寸处理：把图像放大或缩小到 280 软 token 预算内、边长为 48 像素倍数（patch 16 × pooling 3）的最大画布，再用带抗锯齿的 bicubic 滤波拉伸填满，不加 letterbox 黑边。早期构建使用的是 llama.cpp smart resize 的 letterbox 移植版，会给图像补黑边；因此连同上面的激活函数与归一化修复，升级后这些视觉塔上的图像回答会发生变化。缩放后的像素与参考实现的 numpy 转写版做过比对（`Gemma4VisionOracleTests`，需要本地 fixture 目录，没有时跳过）。12B 使用的 `gemma4uv` 统一 embedder 是一条无 block 的路径（一个大 patch 卷积、若干 LayerNorm、学习的二维位置 embedding、一次 RMSNorm 与线性投影），而不是上面的 SigLIP 堆叠，它保留了经过验证的旧 letterbox 尺寸处理。视频帧使用固定的 70 软 token 预算，同样采用旧的尺寸处理。

### 4.10 音频管线

`Gemma4AudioPreprocessor` 解码 WAV / MP3 / OGG，重采样到 16 kHz 单声道，输出 128-bin log-mel（10 ms hop），并 pad 到编码器的 chunk size（12 帧 × 12 chunks）。

`Gemma4AudioEncoder` 是 chunked-attention USM 风格 transformer：

- 时间维 conv 子采样。
- per-chunk 因果 attention，past context 12 帧。
- attention logits 上 `logit_cap = 50`（与 LM logit softcap 思想一致）。
- residual 缩放因子 `0.5`。
- 最终线性投影到 LM hidden。

输出走与图像 embedding 相同的 `InjectVisionEmbeddings` 路径；从语言模型视角，audio 与 image token 是预计算 embedding 的可互换载体。

## 5. 参数与配置（GGUF 元数据）

| Key | 类型 | 含义 |
|---|---|---|
| `gemma4.attention.sliding_window_pattern` | bool[] | per-layer SWA 模式 |
| `gemma4.attention.sliding_window` | uint32 | SWA 窗口大小（默认 512） |
| `gemma4.attention.key_length` | uint32 | 全局 head dim（默认 512） |
| `gemma4.attention.key_length_swa` | uint32 | local head dim（默认 256） |
| `gemma4.attention.global_head_count_kv` | uint32 | 全局层 KV head 数 |
| `gemma4.attention.head_count_kv` | int32[] | per-layer KV head 数 |
| `gemma4.attention.shared_kv_layers` | uint32 | 末尾共享 KV 的层数 |
| `gemma4.rope.dimension_count` | uint32 | partial rotary 维数 |
| `gemma4.rope.partial_rotary_factor` | float32 | head dim 旋转比例 |
| `gemma4.rope.freq_base_swa` | float32 | local RoPE base |
| `gemma4.embedding_length_per_layer_input` | uint32 | PLE 维度（0 关闭） |
| `gemma4.expert_count` | uint32 | MoE 专家数（0 ⇒ 密集） |
| `gemma4.expert_used_count` | uint32 | TopK 路由专家数 |
| `gemma4.final_logit_softcapping` | float32 | LM head softcap |

## 6. 权重命名约定

```
token_embd.weight
output_norm.weight
output.weight                              # （可选，若 tied 到 token_embd）

blk.{L}.attn_norm.weight
blk.{L}.attn_qkv.weight                    # 融合 QKV（非共享层）
blk.{L}.attn_q.weight                      # 仅 Q（KV 共享层）
blk.{L}.attn_q_norm.weight
blk.{L}.attn_k_norm.weight
blk.{L}.attn_output.weight
blk.{L}.post_attention_norm.weight         # 或 attn_post_norm.weight
blk.{L}.ffn_norm.weight
blk.{L}.ffn_gate_up.weight                 # 融合 gate+up
blk.{L}.ffn_down.weight
blk.{L}.post_ffw_norm.weight               # 或 ffn_post_norm.weight
blk.{L}.layer_output_scale.weight          # 标量 [1]

# 仅 MoE：
blk.{L}.ffn_gate_inp.weight                # router
blk.{L}.ffn_gate_inp.scale                 # 学到的 router scale
blk.{L}.ffn_gate_up_exps.{E}.weight        # 融合的 expert gate+up
blk.{L}.ffn_down_exps.{E}.weight           # expert down
blk.{L}.pre_ffw_norm_2.weight              # MoE 输入 norm
blk.{L}.post_ffw_norm_1.weight             # 密集 MLP post-norm
blk.{L}.post_ffw_norm_2.weight             # MoE post-norm

# 仅 PLE：
per_layer_token_embd.weight
per_layer_model_proj.weight
per_layer_proj_norm.weight
blk.{L}.inp_gate.weight                    # PLE gate
blk.{L}.proj.weight                        # PLE 投影
blk.{L}.post_norm.weight                   # PLE post-norm

# 全局 RoPE：
rope_freqs.weight                          # 比例频率因子
```

## 7. 在共享解码器运行时上的实现

Gemma 4 运行在共享解码器运行时上：[`Gemma4RuntimeModel`](../../TensorSharp.Models/Models/Gemma4/Gemma4RuntimeModel.cs) 派生自
[`Gemma4DecoderModel`](../../TensorSharp.Models/Models/Gemma4/Gemma4DecoderModel.cs)（它与 [DiffusionGemma](diffusiongemma_zh-cn.md) 共享的配方），是一个
[`StandardDecoderModel`](../../TensorSharp.Models/SharedDecoder/StandardDecoderModel.cs) 家族，因此槽位、统一步骤、连续批处理、前缀缓存、
检查点、放置策略与投机解码都由运行时提供，每一步都是由后端执行器（GGML、托管 CPU、直连 CUDA、MLX）运行的整前向计算图程序。
家族通过运行时的钩子提供模型配方：

| 部分 | 位置 | 内容 |
|---|---|---|
| 逐层注意力几何 | `Gemma4DecoderModel.cs`（`LayerKeyDim`、`LayerKvHeads`、`SlidingWindow`、`AttentionScale`） | 宽 256 的滑窗 head 与宽 512 的全局 head、逐层 KV head 数、缩放为 1 |
| KV 共享 | `KvSourceLayer`；运行时的 `EmitSharedKvAttention` | 最后 `shared_kv_layers` 层读取同一步中其来源层的操作数，只投影 query |
| Q/K/V | `Gemma4DecoderModel.Graph.cs`（`EmitAttentionProjections`、`EmitQueryKey`） | 带权重的 Q/K RMSNorm、无权重的 V norm；没有 `attn_v` 的全局层取 V = K |
| 旋转位置编码 | 带 `rope_freqs` 因子的 `RotarySpec`（带因子输入的 `GraphOp.RopeExt`） | 局部层 NeoX、基数 10 000；全局层在 512 维上按比例因子旋转 |
| 前馈 | `EmitFeedForward`、`EmitRoutedExperts` | GeGLU；MoE 层：稠密分支与基于融合 `ffn_gate_up_exps` 的路由 GeGLU 专家并列，带路由缩放与逐专家输出缩放 |
| Per-layer embedding | `EmitStreamInput` / `EmitPerLayerInputs` / `EmitBlockOutput` | `per_layer_token_embd` 的 token 行加上归一化的模型投影，在每层之后经门控加入残差流 |
| 多模态 | [`Gemma4RuntimeModel.Multimodal.cs`](../../TensorSharp.Models/Models/Gemma4/Gemma4RuntimeModel.Multimodal.cs) | 编码器的 soft token 行在 `sqrt(n_embd)` 缩放之前替换占位行（缩放跳过它们），并且彼此双向注意 |
| MTP 草稿头 | [`Gemma4RuntimeModel.Assistant.cs`](../../TensorSharp.Models/Models/Gemma4/Gemma4RuntimeModel.Assistant.cs) | gemma4-assistant 草稿器（§12） |

Gemma 4 为运行时新增、所有家族可用的部分：逐层的 key/value 宽度与 KV head 数、KV 共享层、所有执行器上的旋转频率因子
（`Ops.RoPEExFactors`、CUDA kernel、带频率的 MLX `fast.rope`）、双向多模态区段、在流水线阶段之间随残差流传递的 stage carry、
对主干层缓存的只读注意力（`EmitCacheReadAttention`），以及逐 token 的家族草稿器（`FamilyTokenDrafter`）。

## 8. Prefill 与 decode

prompt 分块与 decode 行是同一个计算图程序的不同宽度。同一宽度的 decode 步以“前一步推进一个位置”的方式重放（位置、掩码、写入行
与 per-layer embedding 的 token id 原地刷新），在直连 CUDA 上以录制的 kernel 序列重放。滑窗层保留窗口加一个 prompt 步行数的环形缓存。
在直连 CUDA 上，decode 行的路由专家作为一个融合区域运行（`GraphFusion.GegluRoutedExperts`：融合 stack 的 MoE decode kernel，dp4a），
在 A40 上把 26B-A4B 的 decode 从 34 提高到 80 tok/s。

## 9. 内存与 KV cache

KV 共享层没有自己的缓存。运行时支持的所有缓存类型都可用，包括块量化类型（`q8_0`、`q4_0`；旧模型会拒绝它们并回退到 `f16`）：
E4B 在 `ggml_cuda` 上，`q8_0` 缓存在 96 个 teacher-forced 步中与 `f16` 参考的 cos 为 0.9999。路由专家可以留在主机内存
（`--n-cpu-moe` / `--cpu-moe`），所有后端都支持：GGML 的主机 MoE 段与张量执行器的专家宿主都在原生主机专家 kernel
（`TSGgml_HostExpertRowsAct`）中对融合 stack 运行 GeGLU；26B-A4B 在 `ggml_cuda` 上把 12 层专家放在主机时，与旧模型全设备运行
在 96 个 teacher-forced 步中有 95 步一致（另一步是 0.06 nat 的近似平局）。

## 10. 连续批处理、前缀缓存与检查点

请求在各自槽位上以批处理步骤运行（prompt 分块与 decode 行同在一步）；前缀缓存、保留会话、检查点及其导出 / 导入都由运行时提供
（见[分页注意力与连续批处理](../PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)）。

## 11. 与旧模型的对照验证

运行时模型取代了旧的 `Gemma4Model`，后者连同其原生整模型计算图（`TSGgml_Gemma4ModelDecode`、`...Verify`、`...DecodeBatched`、
MoE 变体与 `TSGgml_Gemma4DraftStep`）一并删除。在一张 A40 上测得（ggml 为 d9fe7bff），旧模型与运行时由同一份源码构建：

| 检查 | 结果 |
|---|---|
| Teacher-forced argmax 一致性（E4B Q8_0、12B UD-Q4_K_XL、26B-A4B UD-Q4_K_XL；ggml_cuda、cuda、ggml_cpu、cpu） | 除近似平局外每一步都一致（不一致处旧模型的最大 margin 为 0.29 nat） |
| 验证窗口与普通步对比（E4B，窗口 4，含回滚） | ggml_cuda relL2 1.43e-2（旧模型 1.42e-2）；直连 CUDA 逐位一致（旧模型 1.3e-2） |
| 经 CLI 的图片（E4B、12B、26B；ggml_cuda 与 cuda）与语音（E4B） | 与旧模型给出相同的描述与相同的转写 |
| 服务端，5 个提示、3 个并发（ggml_cuda） | E4B 102 tok/s，旧模型 90；26B-A4B 74，旧模型 56 |
| Prefill（ggml_cuda，预热后） | 旧模型的 1.5-2.4 倍 |

两种模型的贪心输出会在某个 token 的前两个 logit 差距落在后端舍入误差之内时分叉；合成夹具上冻结的旧模型 logits
（`InferenceWeb.Tests/Fixtures/Gemma4/LegacySynthetic`）在 [`Gemma4RuntimeTests`](../../InferenceWeb.Tests/Gemma4RuntimeTests.cs) 中
约束运行时与旧模型的数学一致。此处未运行：Metal、Vulkan、MLX 与多 GPU（张量并行与流水线阶段在托管执行器上用模拟设备测试）。

## 12. MTP 投机解码（gemma4-assistant 草稿头）

Gemma 4 在两个宿主上都支持为单序列（无并发）请求做**多 token
预测（MTP）投机解码**。与 Qwen 3.6 把 NextN 块内嵌在主干 GGUF 不同，Gemma 4 的草稿头
作为一个**独立的小 `gemma4-assistant` GGUF** 发布，通过 `--draft-model`
（环境变量 `TS_SPEC_DRAFT_MODEL`）加载，
并由 [`SpeculativeDraftHeadLoader`](../../TensorSharp.Models/SpeculativeDraftHeadLoader.cs)
在启动时挂到目标模型上。源码：
[`Gemma4RuntimeModel.Assistant.cs`](../../TensorSharp.Models/Models/Gemma4/Gemma4RuntimeModel.Assistant.cs)，
由共享的
[`SpeculativeExecution`](../../TensorSharp.Runtime/Speculative/SpeculativeExecution.cs)
起草 / 验证 / 回滚核心驱动 —— Gemma 4 对应可插拔 `--spec-type` 层中的 `draft-head` 算法，
`auto` 会依据 checkpoint 自动选中它。

### 12.1 EAGLE 风格递归草稿器

草稿头是 EAGLE 风格的递归草稿器（对应 llama.cpp 的 `gemma4-assistant.cpp` +
`speculative.cpp` draft-mtp）。每起草一个 token：

```
x      = target.tok_embd[token] * sqrt(n_embd_backbone)   // backbone 嵌入（例如 3840）
xh     = concat(x, h_prev)                                 // [2*backbone]
cur    = nextn.pre_projection @ xh                         // -> 草稿隐藏维（例如 1024）
for il in 0..n_nextn-1:                                    // 若干个 Gemma 风格解码块
    Q   = rope(q_norm(wq @ attn_norm(cur)))               // 草稿只计算 Q
    a   = attn(Q, target_KV)                               // 读取目标模型最后一个 local（N-2）
                                                          //   / global（N-1）层的 K/V
    cur = block_residuals(a, ffn, out_scale)
cur    = output_norm(cur)
logits = draft.tok_embd @ cur                             // 草稿自己的 LM 头
h_next = nextn.post_projection @ cur                      // -> backbone，串接下一个草稿步
```

草稿头**自身不保存任何 K/V**（没有 `wk`/`wv` 张量）：每个草稿层都查询目标模型已有的逐层
KV 缓存，并对每个起草 token 复用相同位置（递归只通过 `h` 流动）。因此在给定 `(token, h)`
时草稿器是无状态的，被拒时唯一的回滚就是注意力 KV 位置回退——无需递归状态快照（与 Qwen 3.6
的 GatedDeltaNet 主干形成对比）。

### 12.2 草稿 / 目标配对（快速失败）

草稿的输出 backbone 维度（`{arch}.embedding_length_out`）**必须等于目标的隐藏维度**——
12B 目标配 12B 草稿，而非 26B-A4B 草稿。当给了 `--draft-model` 但草稿无法激活（文件缺失、
隐藏维不匹配、或缺少必要的草稿张量）时，服务端会在启动时**立即失败**并给出修复提示
（[`SpeculationStartupValidation`](../../TensorSharp.Chat/Hosting/SpeculationStartupValidation.cs)），
而不是静默地不做投机继续运行。

### 12.3 运行时上的草稿器

assistant 是一个逐 token 的家族草稿器（[`Gemma4RuntimeModel.Assistant.cs`](../../TensorSharp.Models/Models/Gemma4/Gemma4RuntimeModel.Assistant.cs)）：
每一步是活动槽位上的一个计算图程序，其各块经 KV 共享映射，以 `EmitCacheReadAttention` 在主干已提交长度处读取主干的 donor 缓存，
因此无需追赶，草稿可在任意位置恢复。它在所有计算图执行器上运行。`--tp` 下会拒绝该草稿头（它读取完整缓存，而张量并行按 KV head
切分缓存）；n-gram 投机仍然可用。A40 上的测量（贪心，128 token，3 个草稿）：

| 模型、后端 | 普通 tok/s | 使用草稿头 | 接受率 | 旧模型使用其草稿头 |
|---|---|---|---|---|
| E4B Q8_0，ggml_cuda | 86-89 | 1.0-2.2 倍 | 22-83 % | 0.9-1.8 倍 |
| E4B Q8_0，cuda | 69-74 | 1.0-2.1 倍 | 21-83 % | 0.9-1.8 倍 |
| 12B UD-Q4_K_XL，ggml_cuda | 65-72 | 1.5-2.3 倍 | 52-86 % | 0.7-1.4 倍 |
| 26B-A4B UD-Q4_K_XL，ggml_cuda | 114-141 | 0.9-1.4 倍 | 44-89 % | 0.6-0.9 倍 |

启用方式就是 `--draft-model` 文件本身；关闭（`--no-spec`）与调参（`--spec-draft` / `--spec-pmin`）使用共享标志，`TensorSharp.Cli`
与 `TensorSharp.Server` 都由同一个 [`SpeculativeCliFlags`](../../TensorSharp.Runtime/Speculative/SpeculativeCliFlags.cs) 解析；其他算法见
[投机解码](../../FEATURES_zh-cn.md#投机解码)（`--spec-type ngram` 完全不需要草稿 GGUF）。

### 12.4 贪心一致性

每个输出 token 都取自主干的某一行，因此投机不会改变 token 所依据的前缀，只会改变计算这一行的
kernel。用 `AgentTurnBench --spec-diagnostic --spec-diagnostic-teacher-force` 实测有两点后果：

* **已修复：** 在不带逐层嵌入的稠密检查点（实测 12B；31B 走同一路径）上、ggml 与 CPU 后端下，越过滑动窗口的
  一次**全部接受**的验证，过去会把被挤出的位置写回到已提交的行上，于是下一次验证偏差 30-47 个
  logit，12B 的输出流在十个 token 内就与普通贪心分叉。现在 `SpecOnVerifyAccepted` 只对执行器会
  重新前向的窗口恢复槽位；A40 上 12B 的 spec 与检查点克隆输出流重新与普通贪心一致，spec 提示上的
  n-gram 接受率从 57% 升到 89%。
* **容差：** K+1 行验证与单行 decode 是不同的 kernel。在 `ggml_cuda` 上两者的行在 E4B 上相差
  中位数 0.5 个 logit（其 BF16 逐层嵌入投影在 ggml-cuda 中依赖 batch 形状），在 12B/26B-QAT 上为
  1.7-2.2；Metal 上为 0.003（E4B）与 0.15（12B）；`ggml_cpu` 上为 0。若某个贪心 token 的前两名
  logit 之差落在这一误差之内，它就可能不同：E4B 的 512 个散文 token 中出现了 4-5 次，每次的差都小于 0.25。

细节与完整表格见：[What greedy parity delivers](../speculative_decoding.md#what-greedy-parity-delivers)。

## 13. 输出解析器与聊天模板

`Gemma4OutputParser` 处理两种结构化包装：

- **思维链** —— `<|channel>thought ... <channel|>` 的 chain-of-thought，再跟最终答案。
- **工具调用** —— `<|tool_call>call:function_name{...args...}<tool_call|>` 块，由 `OutputParser` 解出结构化的 tool call。参数使用 Gemma 自己的语法（裸键名，字符串用 `<|"|>` 包裹），而模型经常把看起来像标识符的字符串值直接裸写——`call:read_invoice{invoice_id:INV-472}`、`{path:src/main.py}`、`{ids:[INV-1, INV-2]}`。转换器会给每个不是 JSON 数字 / `true` / `false` / `null` 的裸值加引号（数组内也一样；数字保持数字）。参数仍无法解析的调用会以原文作为 content 返回，而不是一条空消息；多调用轮次中每个调用都带自己的 `index`，流式客户端可据此配对参数增量。

聊天模板在 GGUF 没带 Jinja2 模板时回退到内置 Gemma 4 模板。

### 关闭思考时的思维通道

关闭思考时，没有任何 Gemma 4 模板会在工具结果之后预置内容：E2B/E4B 的 canonical 模板以及 12B/26B/DiffusionGemma 模板都让 prompt 停在 `<tool_response|>`，由模型继续自己的轮次（较大模型的模板只在新的 `<|turn>model` 之后预置闭合的 `<|channel>thought\n<channel|>`）。渲染器保持这个 prompt。我们在 E4B（Metal，Q8_0，带 skills 前导）上实测过在工具结果之后强行预置闭合块，结果更糟：模型不加标记地写出推理，再次关闭通道后才回答，推理因此进入了 `content`。

E4B 在该边界上的行为（2026-09-16 验证活动，B10）改为在采样阶段处理，依据协议的 `ThinkingBudgetOpenToken` / `ThinkingBudgetEndToken` / `SuppressUnopenedThinkingEndAfter`：

- **关闭思考时模型自行打开的通道会被限长。** 预算从 `<|channel>` 开始计，在超过 `max_tokens` 四分之一（最多 64 token，两倍处强制关闭）后的第一个换行处以 `<channel|>` 关闭通道。解析器隐藏思考内容，随后是答案。此前 `agentic` 的最后一轮把 256 个 token 全部花在通道里，content 为空。我们也实测过恰好在 16 或 64 token（句子中间）关闭：模型会泄漏推理或把被截断的句子续写成工具调用；而在超过 16、32、48、64 token 后的第一个换行处关闭，每次都给出了正确答案。
- **工具结果之后，没有打开通道的 `<channel|>` 会被屏蔽。** E4B 先写出答案，关闭一个从未打开的通道，然后再写一遍答案；流式响应已经发出了第一份，客户端因此收到两份。屏蔽后模型在第一份答案后结束轮次。屏蔽需要 host logits，因此这类请求不走 device-argmax 路径。屏蔽只限于该边界：其他位置的孤立关闭标记仍由解析器用来分隔推理与答案。
- 开启思考时，同一预算从模型的 `<|channel>` 开始计（若 prompt 在工具结果后预置了 `<|channel>thought\n`，则从第一个 token 开始计），并恰好在 `TS_THINKING_BUDGET` 处关闭。此前 Gemma 4 没有声明训练过的结束 token，通用的硬停止在寻找 Gemma 从不输出的 `</think>`。`TS_THINKING_BUDGET=0` 同时关闭两个上限。

`response_format` 可以与 `"think": true` 同时使用：JSON 语法在思维通道内保持休眠，在 `<channel|>` 之后启用（`ThinkingGrammarActivationTrigger`）。关闭思考时语法从第一个 token 起生效，这同时排除了 `<|channel>`。

## 13a. 张量并行与整层流水线阶段

`--tp N` 运行每 rank 的计算图：Q/K/V 与 gate/up 按列并行，注意力输出与 down 按行并行；MoE 层的融合 `[gate | up]` 专家按两半切分
（每个 rank 的切片为 `[gate_r | up_r]`，在 GGML 上是它自己的专家 stack）；`[dense | routed]` 两部分在各自的 norm 之前用一次 all-reduce
合并。每一层的 KV head 数都必须能被 N 整除。`--layer-split N` 每个阶段运行一个计算图；per-layer embedding 随残差流在阶段之间传递，
阶段规划器不会在 KV 共享层与其来源层之间开始新的阶段。张量并行的流水线阶段（混合放置）是已声明的缺口：这个只走计算图的家族
目前还没有对应的计算图。

## 14. 待办

- 直连 CUDA 上 E4B 的 decode 在 512/128 时比旧模型低约 5-10 %（64/64 时持平）：逐算子的 matvec 与 norm 启动开销；其他检查点与后端
  持平或更快。
- 新进程中 12B / 26B-A4B 在 `ggml_cuda` 上的第一次长 prefill 需要为其步骤形状编译计算图（以及惰性上传专家）；预热后的 prefill 是
  旧模型的 1.5-2.4 倍。
- 本次迁移没有在 Metal、Vulkan、MLX 与多 GPU 放置的真实硬件上运行。
