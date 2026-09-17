# Gemma 4

[← 返回模型索引](README_zh-cn.md) | [English](gemma4.md)

> **想沿着一条从零构建的路线学习？** Zhongkai Fu 所著的 [From Tensors to Tokens](../BOOK_zh-cn.md) 以 Gemma 4 E4B 为例，连接张量基础、多模态模型执行与完整的 TensorSharp 推理应用。[在 Amazon 查看平装本](https://www.amazon.com/dp/B0H9P44QZZ)。

| 属性 | 值 |
|---|---|
| 提供方 | Google |
| GGUF 架构标识 | `gemma4` |
| 模型类 | [`Gemma4Model`](../../TensorSharp.Models/Models/Gemma4/Gemma4Model.cs)（旧单序列路径）+ [`Gemma4Model.BatchedForward.cs`](../../TensorSharp.Models/Models/Gemma4/Gemma4Model.BatchedForward.cs)（`IBatchedPagedModel`） |
| 视觉编码器 | [`Gemma4VisionEncoder`](../../TensorSharp.Models/Models/Gemma4/Gemma4VisionEncoder.cs)（SigLIP 风格 ViT） |
| 音频编码器 | [`Gemma4AudioEncoder`](../../TensorSharp.Models/Models/Gemma4/Gemma4AudioEncoder.cs)（USM 风格 chunked transformer） |
| 音频前端 | [`Gemma4AudioPreprocessor`](../../TensorSharp.Models/Models/Gemma4/Gemma4AudioPreprocessor.cs)（16 kHz 单声道 → 128 bin log-mel） |
| 图像处理器 | [`Gemma4ImageProcessor`](../../TensorSharp.Models/Models/Gemma4/Gemma4ImageProcessor.cs) |
| 示例模型 | gemma-4-E4B（8B 等效）、gemma-4-12B、gemma-4-31B、gemma-4-26B-A4B（MoE） |
| 模态 | 文本、图像、视频（帧栈）、音频 |
| 思维链模式 | 是（`<\|channel>thought ... <channel\|>`） |
| 工具调用 | 是（`<\|tool_call>call:name{...}<tool_call\|>`） |
| 批处理 / 分页前向 | **默认启用** —— `IBatchedPagedModel.ForwardBatch` 处理双 head_dim、KV donor 共享、PLE 注入、SWA + 全局混合的分页 K/V 缓冲。设置 `TS_GEMMA4_BATCHED=0` 可强制回退到旧单序列 KV 交换路径。详见 §11。 |
| MTP 投机解码 | 可选 —— 通过 `--draft-model`（`TS_SPEC_DRAFT_MODEL`）加载独立的 `gemma4-assistant` EAGLE 风格草稿 GGUF；指定该文件本身就会启用投机（显式 `--no-spec` 可否决）。该标志在**两个宿主上都可用**：`TensorSharp.Cli` 与 `TensorSharp.Server` 共用同一个 [`SpeculativeCliFlags`](../../TensorSharp.Runtime/Speculative/SpeculativeCliFlags.cs)。在 ggml 后端与纯 C# `cuda` 后端上有收益。详见 §12。 |
| 输出解析器 | `Gemma4OutputParser` |

## 下载

已验证的 GGUF 下载指引：

| 模型 | HF 仓库 | 推荐文件 | mmproj（图像 / 视频 / 音频） | MTP 草稿 |
|---|---|---|---|---|
| gemma-4-E4B-it | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) | `gemma-4-E4B-it-Q8_0.gguf`（8.031 GB；另有 `gemma-4-E4B-it-Q4_K_M.gguf`，5.335 GB） | `mmproj-gemma-4-E4B-it-Q8_0.gguf`（0.560 GB；同仓库；备选：[unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) 中的 `mmproj-F16.gguf`，0.990 GB） | 来自 [AtomicChat/gemma-4-E4B-it-assistant-GGUF](https://huggingface.co/AtomicChat/gemma-4-E4B-it-assistant-GGUF) 的 `gemma-4-E4B-it-assistant.Q8_0.gguf`（0.100 GB） |
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

快速路径的依据来自三处相互独立的路由事实：

- 多 token prefill / verify 使用融合整模型 `NativeGemma4ModelVerify` 图，其中包含
  E 系列的内核内 PLE gather 与共享 KV donor 处理。
- 稠密模型的单 token decode 通过一次 GGML 图派发在
  `NativeGemma4ModelDecode` 中执行完整 transformer。
- 只有一个调度序列时，默认的 `TS_BATCHED_N1_FAST_PATH=1` 会选择线性
  `Forward()` 路径，从而进入融合整模型 decode，而不是通用批处理逐算子路径。

实测与路由说明见[引擎对比报告](../engine_comparison_report.md)、
[E4B prefill 性能记录](../perf/gemma4-prefill-cuda-graph-design.md)与
[N=1 调度器文档](../PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)。

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

- **Decode**（`seqLen == 1`）在 GGML 后端、所有层均为密集且权重均量化时：单次原生调用（`Gemma4ModelDecode`）一次 GPU 图调度处理整套堆栈，包含 PLE、per-layer head 维、环形 SWA cache、per-layer scalar。
- **Prefill**（`seqLen > 1`）：符合条件的稠密模型优先使用融合整模型 verify 图，其中包含 E 系列 PLE 与共享 KV donor 层。该入口不可用时，符合条件的稠密、非共享、无 PLE 层仍可走 per-layer `Gemma4LayerPrefill` 图，其余层回退逐算子路径；长 prompt 会分块以控制 SWA score 张量。

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

`Gemma4VisionEncoder` 是带 2D 位置 embedding、GELU-Tanh MLP 和最终线性投影的 SigLIP 风格 ViT。视频帧从 MP4 中提取（默认通过 OpenCV / SkiaSharp 抽取最多 8 帧、1 fps），每帧独立编码，最终的 embedding 串联起来按顺序注入到连续的 `<|image>` 占位上。

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

## 7. TensorSharp 实现走读

构造函数（`Gemma4Model(string ggufPath, BackendType backend)`）：

1. `ParseBaseConfig()`（通用字段）。
2. 读取 SWA 模式与窗口、双 head dim、双 KV head 数、双 RoPE base、partial rotary 维、PLE 维、共享 KV 层数、MoE 计数。根据 SWA 模式设置 `Config.UsesCircularKvCache`。
3. `BuildKVDonorMap()` —— 产生 `_kvDonorMap[layer] → donorLayer`。
4. `ParseTokenizer()`、`LoadWeights()`。
5. `_hasTiedOutput` 检测。
6. `DetectHeadDimsFromWeights()` —— 元数据不一致时修复 head dim。
7. `LoadLayerScalars()` —— 加载 `_layerScalars[NumLayers]`。
8. `FuseQKVWeights()`、`FuseGateUpWeights()`、`FuseExpertGateUpWeights()`。
9. `PrepareCudaQuantizedWeightsForInference()`。
10. `PrecomputeRoPE()`。
11. `InitKVCache(maxSeqLen)` —— SWA 层容量为 `slidingWindow`，全局层为 `maxSeqLen`；共享层 alias 到 donor。
12. `BuildGemma4DecodeArrays()` —— 把 per-layer 指针、类型、维度，以及可选的 MoE / PLE 标志打包到 `_decodeArrays` 给融合 decode kernel 用。

`Forward(int[] tokens)`：

- embedding lookup + 缩放。
- 可选的图像 / 音频注入（被注入的位置加入 `exceptPositions` 让层代码跳过这些位置上的 RoPE / cache 写入）。
- 可选 PLE 计算。
- 然后：
  - **融合 decode**（一次原生调用，见 § 9），或
  - 逐层 C# 循环，每个密集层尝试 `TryFusedLayerPrefill`。
- 最终 RMSNorm、LM head、可选 softcap、复制到 `_logitsBuffer`。

`ForwardRefill(int[] tokens)` 是 prefill-then-decode 入口：把前缀切成 `min(2 × slidingWindow, 2048)`-token 的块（`TS_PREFILL_CHUNK` 可覆盖），最后调 `Forward([lastToken])`。多模态 prompt 跳过分块（注入位置是绝对的）。

## 8. Prefill 优化

### 整模型单图 prefill（`NativeGemma4ModelVerify`）

在 ggml 后端上，普通的多 token prefill 由 MTP 验证所用的同一个融合整模型内核（§12）来执行：所有层在单次 GGML 图派发中完成，激活值常驻设备，而不是每层一张图。`CanUseWholeModelPrefillVerify()` 决定是否走该路径——仅限密集模型，包括 E 系列的内核内 PLE 与共享 KV donor 层；多模态 chunk 在任意起始位置都可通过内核的双向 span mask 走该路径（`TS_G4_MM_PREFILL=0` 让多模态退回逐算子路径；见[复用前缀之后的图片与音频回合](#复用前缀之后的图片与音频回合)）。`startPos > 0` 的 SWA 包裹 chunk 通过内核内的 swaPrev gather 留在融合路径上（`TS_G4_VERIFY_SWAPREV=0` 关闭）。全 MoE 变体（例如 26B-A4B）有对应的融合路径：`CanUseWholeModelMoEPrefillVerify()` / `TryFusedMoEModelVerify()`。设 `TS_G4_WHOLE_PREFILL=0` 可强制走逐算子分块路径做 A/B。注意，块量化（`q8_0` / `q4_0`）KV cache 的多 token prefill *必须*走该路径——逐算子回退无法遍历块量化的 cache 布局。

调度器会把 solo（无争用）prompt 以大分块喂给该路径，分块上限由 `TS_SCHED_SOLO_PREFILL_CHUNK`（默认 8192）控制。实测设计见 [`docs/perf/gemma4-prefill-cuda-graph-design.md`](../perf/gemma4-prefill-cuda-graph-design.md)。

### 复用前缀之后的图片与音频回合

图片、视频帧或音频片段以一段软 token 进入提示，这些软 token 彼此双向注意。prefill 内核用每个 chunk token 一个字节（`is_except`）表达这一点：软 token 查询还会读取本 chunk 中位于其后的软 token 键，而所有查询都因果地读取其之前的键，局部层上限定在滑动窗口之内。当一个会话把缓存续接到图片回合时，图片 chunk 从非零位置 P 开始，位于之前回合留下的文本之后。

在此之前，融合内核（密集的 `TSGgml_Gemma4ModelVerify`、MoE 的 `TSGgml_Gemma4MoEModelVerify`）只在 P = 0 时应用软 token 条件，因为它们拿键在缓冲区中的下标与 chunk 的字节比较。于是门控把 P > 0 的媒体 chunk 交给逐算子路径，这条路径更慢（E4B/Metal，一个复用 179 token 的 457 token 图片回合：首 token 1.25 s，冷启动为 0.64 s），而且环回绕之后是错的：它的掩码把缓冲区下标当作绝对位置，但超出窗口后收集的窗口从 P - 511 开始，而不是 0。Phase 0 因此只能让这样的回合不复用公共前缀之后的内容（`CanPrefillMediaAfterReusedPrefix`）。

融合内核注意的每个缓冲区都把本 chunk 作为最后 N 个真实键：P = 0 时只有 chunk 本身，全局缓存 `[0, P + N)`，未回绕的滑动窗口缓存，或者从已回绕的环中收集并前置到 chunk 之前的上一窗口。因此键在 chunk 中的下标就是它在缓冲区中的下标减去 chunk 之前的真实键数，`gemma4_mm_mask.h` 中的行构造函数应用了这个偏移。在 P = 0 以及文本 chunk 上，它们生成的行与之前相同。逐算子路径把软 token 的绝对位置平移到其键缓冲区的坐标系中（`ShiftPositions`），与张量并行的逐算子路径早已采用的做法一致；张量并行的融合 verify 现在也在任意 P 传入 chunk 掩码（其密集变体过去在 P > 0 运行媒体 chunk 时完全没有软 token 掩码）。

所以 Gemma 4 不再重写 `CanPrefillMediaAfterReusedPrefix`：图片回合在任意提示长度下都会复用会话的文本，图片 chunk 走融合图。在 E4B（Q8_0，Metal，M5 Pro）上用 IMG 会话（文本、文本、图片、文本；贪心；两轮取中位数；Web UI 一列为首 token 时间，OpenAI 一列为整个非流式请求的耗时）测得：

| 第 3 回合（图片） | 6db6dbf6 | Phase 0 | 本改动 |
|---|---|---|---|
| Web UI，457 token 提示：复用 / 首 token 时间 | 0 / 0.64 s | 179 / 1.25 s | 179 / **0.57 s** |
| OpenAI，457 token 提示：复用 / 耗时 | 0 / 1.80 s | 179 / 2.09 s | 179 / **1.46 s** |
| Web UI，889 token 提示（超出窗口）：复用 / 首 token 时间 | 0 / 0.85 s | 0 / 0.90 s | 611 / **0.62 s** |
| OpenAI，946 token 提示（超出窗口）：复用 / 耗时 | 0 / 1.86 s | 0 / 1.60 s | 668 / **1.29 s** |

这些运行中的每条回复（短会话 24 个回合、长会话 16 个回合）都与 Phase 0 头部的文本相同，也与本改动关闭前缀缓存（`TS_SCHED_PREFIX_CACHE=0`）时的文本相同，纯文本回合的复用量和耗时保持不变（`AgentTurnBench --scenarios short,long,tool`，每个构建各跑两次：每一行都与 6db6dbf6 和 Phase 0 头部相差不超过 0.5%，token 完全相同）。

在 CUDA 上（E4B Q8_0，A40，一轮），图片回合同样获得复用并且更快：Web UI 449 token 提示 1.10 s（6db6dbf6）/ 2.85 s（Phase 0）/ **0.93 s**，OpenAI 1.45 / 2.49 / **0.98 s**；超出窗口时，Web UI 893 token、复用 615：1.10 / 1.12 / **0.91 s**，OpenAI 870 token、复用 592：2.05 / 1.06 / **0.87 s**。不过在那里，任何构建上从缓存续接的回合都不与冷启动回复逐 token 相同：文本回合和图片回合都一样，6db6dbf6 也一样，复用前缀之后的贪心回复会在几十到几百个字符之后偏离冷启动回复，因为在这些内核上把同样的 token 分两个 chunk 而不是一个 chunk 预填充，在 E4B 上会让 logits 变动 0.65 到 0.8（即下文测试中的文本对照）。在提示完全相同时，Phase 0 头部的图片回合回复在短会话中与本改动相同（两者复用同样的前缀），超出窗口时则不同（Phase 0 头部从零 prefill）。

覆盖：原生测试 `gemma4-multimodal-mask-after-reused-prefix` 保留旧行构造的逐字副本，检查它们在 P = 0 和文本上不变，并检查 P > 0 的 chunk 在每种缓冲区布局下看到的内容与冷启动 prefill 中相同查询看到的完全一致（旧的起始位置门控在其 1,200 个媒体用例中失败 1,102 个）。`Gemma4MediaAfterReusedPrefixExactnessTests`（受模型门控，`TS_TEST_MODEL_DIR`）先 prefill 一个文本回合，再以复用和不复用两种方式 prefill 一个图片或音频回合，分别在窗口内外、融合与逐算子路径上运行。它的容差是同一行中测得的两个噪声估计中较大者的两倍：去掉媒体的同样切分，以及另一条路径上的冷启动媒体 prefill。在 Metal 上（E4B、E2B）两个估计都约为 0.02 logits，融合路径的贪心输出完全相同，prefill logits 相差不超过 0.026，而在 Phase 0 头部，超出窗口时图片相差 3.2、音频相差 2.0 到 7.1（逐算子路径 3.2）。在 CUDA 上（E4B、12B、26B-A4B）两个估计为 0.35 到 1.4（12B 短文本对照为 2.7），所有行都通过；Phase 0 头部的长前缀行相差 3.3 到 3.8（图片）和 2.1（音频），高于本改动在相同行上测得的容差（1.3 到 2.8）；那里的贪心输出只比较到噪声范围内的第一个近乎平局处，通常是第一两个 token。逐算子路径的贪心输出不逐 token 比较：它自身就不可复现（四次相同的冷启动 prefill 在第 16 步解码处翻转了一个近乎平局的 token）。`Gemma4SoftTokenMaskTests` 固定了两种位置换算。

有两个限制沿袭自冷启动路径。掩码标记的是软 token，而不是它们属于哪个媒体项，因此在同一 chunk 中 prefill 的两张图片会彼此双向注意。在较早回合发送过图片的会话，是在该图片自己的 chunk 中 prefill 的，那时它看不到后来的图片；而对整段历史的冷启动 prefill 让它看得到，所以两张图片之后的回复在两者之间可能不同。另外，落在 span 内部的 chunk 边界（调度器在争用时的 256 token 分块可能造成）会切断其双向注意。

### 内核内 PLE gather

per-layer embeddings（PLE）在融合 verify 图内通过对常驻的量化 `per_layer_token_embd` 表做 `ggml_get_rows` 直接收集，而不是在 C# 中计算后每个 chunk 把约 88 MB 的结果做 device→host→device 搬运。默认开启；`TS_G4_PLE_IN_KERNEL=0` 恢复上传路径。

### KV cache 预扩容（`PrepareForPrefill`）

全新 prefill 开始时，按需增长的全局 KV cache 会被预先扩容到整个 prompt 的大小（`PrepareForPrefill(totalPromptTokens)`）。在 `start_pos == 0` 时 cache 中还没有已提交的 K/V，一次性扩容到最终大小无需拷贝任何数据——从而消除了逐次翻倍扩容（每次扩容都要重新拷贝并对整个全局 cache 做 device↔host 往返，64k 时实测约 7%）。仅 GGML GPU 后端，且钳制到模型上下文长度。

### 融合 per-layer prefill（`Gemma4LayerPrefill`）

每个符合条件的层（密集、非共享 KV、当前 chunk 无 PLE 注入、所有权重均量化），`TryFusedLayerPrefill()` 调起单次 GGML 图：

1. `RMSNorm(attn_norm)`
2. 融合 QKV matmul → 拆 Q/K/V
3. per-head QK RMSNorm、V 无权 RMSNorm
4. RoPE（local NeoX 或 global proportional）
5. 因果 attention（SWA 或全因果），含 KV cache append
6. output 投影 + `RMSNorm(post_attn_norm)` + 残差
7. `RMSNorm(ffn_norm)` → ffn_gate_up matmul → `GELU(gate)*up` → ffn_down matmul
8. `RMSNorm(post_ffw_norm)` + 残差 + layer_output_scale

带 MoE、KV 共享或当前 chunk 中有 PLE 注入的层回退到逐算子托管路径。`TS_FUSED_LAYER_PREFILL=0` 关闭融合路径（用于调试 / A/B 基准）。

### 分块 prefill

`prefillChunkSize = min(2048, max(2 × slidingWindow, 1024))`。每个块走完所有层后再前进，保持 SWA score 张量小。`TS_PREFILL_CHUNK=N` 可覆盖。

### 跨层缓存

forward 内跨层复用的三类缓存：

- `_cachedSWAMaskWidths` —— 当前 `(queryLen, startPos)` 的 per-row SWA mask 宽度；`seqLen` 或 `startPos` 改变时重建。
- `_neoXRopeCos` / `_neoXRopeSin` —— 全局层 NeoX RoPE cos/sin 表，每次 forward 构建一次，跨所有全局层复用。
- `_cachedRoPEPosQ` / `_cachedRoPEPosK` —— local 层 RoPE 位置张量，跨层复用，`(seqLen, startPos)` 匹配时不重新分配。

### SWA prev-window gather

长 prompt 在 chunk 内会让 SWA cache 滚动覆盖。chunk-2 起始位置的 query 仍需要 chunk 刚刚覆盖掉的 (W − 1) 个位置。新 chunk 第一层运行前 `PrepareSwaPrevWindowsForChunk(startPos, seqLen)` 快照活跃的 SWA 窗口；SWA 层在跑 attention 前把快照拼到新计算出的 K/V 前面。

### Donor SWA-K/V 发布

存在 KV 共享 SWA 层时，donor 层把刚算出的 K/V 发布到 `_prefillSWAKV`，让共享层 attend 到完整 chunk 的 K/V，而不是滚动 cache 中的不完整窗口。

## 9. Decode 优化

### 融合整模型 decode（`Gemma4ModelDecode`）

`_canUseFusedDecode`（全密集、全量化）且 `seqLen == 1` 时，decode 路径变成单次原生调用。`BuildGemma4DecodeArrays()` 在加载时按层打包：

- 量化权重元数据（`type`、`ne0`、`ne1`、原始字节指针）。
- per-layer head 维、KV head 数、RoPE base / kind。
- KV cache 指针（GPU 上的可写 scratch）。
- layer scalar 值。
- PLE 标志 / projection 指针。

`NativeGemma4ModelDecode()` 就跑整套 stack —— embedding lookup 在 C# 里，每一层（RMSNorm + QKV + QK-norm + V-norm + RoPE + 环形 cache append + attention + output projection + post-attn norm + GeGLU FFN + post-FFN norm + 残差 + layer scalar）在 Metal / CUDA 上一次 GGML 图调度完成。这把每 token 的几百次 CPU↔GPU 往返压成一次。

`EnsureKvCacheHostSynchronized()` 桥接融合与非融合路径：当下一次 forward 不走融合（如对话中间的 prefill），先把 host KV cache 拷贝从设备同步过来。

### SWA 层环形 KV cache

SWA 层用 `CopyToCacheCircular()` 在 `pos % cacheSize` 写入新 K/V 槽位，`AttentionDecodeCircular()` 走环形读。SWA 层因此无视上下文长度只分配 `slidingWindow` 个槽位 —— 常驻内存有界。

### 并发请求的 token 批量融合 decode（`Gemma4ModelDecodeBatchedEx`）

N >= 2 个请求同时在线时，引擎不再轮询 N 个单 token 图：`Gemma4Model.TryForwardBatchedFusedDecode` 在**一个**融合图（[`ggml_ops_gemma4_batched.cpp`](../../TensorSharp.GGML.Native/ggml_ops_gemma4_batched.cpp) 中的 `TSGgml_Gemma4ModelDecodeBatchedEx`）里为每个序列各 decode 一个 token，每步每个权重只加载一次、作用于 N 个 token。decode 受带宽限制，聚合吞吐正来自这里。每个序列保留自己的 per-request KV holder；内核以 `[layer * N + seq]` 指针数组接收这些 holder，在 `[hidden, N]` 上跑 projection / FFN / LM head，并对每个序列在其自身 cache 窗口的直接视图上跑一次单行 flash-attention。

v1 内核只覆盖无 per-layer embedding、无共享 KV 层、且每个序列都还在 SWA 环内的密集模型，所以 E2B/E4B（PLE 维 256、18 个 KV-donor 层、大多数对话都会超出的 512 槽环）总是拒绝，服务器记录 "The model declined the default batched fused-decode path ... concurrency stays near 1x"。`Ex` 入口补上这三处：

- **逐行 PLE。** per-layer embedding 在图内从常驻的量化 `per_layer_token_embd` 表通过对 N 个 token id 的一次 `get_rows` 收集（再加 `per_layer_model_proj` projection、其 RMSNorm 与 `1/sqrt(2)` 合并，与单 token decode 和 `ComputePLE` 完全一致），按层以 `[ple_dim, N]` 的跨步列切片注入。图内形式不可用时（`CanGatherPleInKernel` 为 false：`get_rows` 不支持的类型或带缩放的 projection），调用方改为上传 `ComputePLE` 的行，PLE 项永远不会被丢掉。
- **KV-donor 层。** `kv_source_arr[l]` 指明第 `l` 层要 attend 的 cache 所属层。共享层只跑 Q projection，读取 donor 的逐序列窗口与 mask，不写任何东西。
- **SWA 回绕。** 序列超出环的 local 层写到 `pos % cache_size`，并把整个环平铺读取、所有槽位有效；decode 的 softmax 对 key 的排列不变，因此旋转不需要 concat。全局（线性）层仍必须容纳整个序列；轮询回退会扩容，下一步重新进入批量路径。

原生侧通过 `TSGgml_Gemma4BatchedDecodeCapabilities()`（位：PLE、KV donor、SWA wrap）报告支持范围。托管侧对已加载原生库缺少的每一位保留 v1 限制，所以旧的 `libGgmlOps`（没有探测符号）行为与以前完全一致，`TSGgml_Gemma4ModelDecodeBatched` 作为薄包装保留 v1 ABI。`TS_GEMMA4_BATCHED_CAPS=0` 可强制 v1 门控做 A/B。

CUDA-graph 捕获不变：每步的所有输入（hidden 行、position、每 (层, 序列) 的 `set_rows` 写行、每层 F16 mask、PLE token id 或上传的 PLE 行）都是用 `ggml_backend_tensor_set` 刷新的图输入，图位于自己的 context 与独占 buffer 中，重复出现的请求集合在稳定地址上重放捕获的图。MoE 批量内核（`TSGgml_Gemma4MoEModelDecodeBatched`）保持无 PLE / 无 donor 的范围。

**已验证**（[`Gemma4BatchedFusedDecodeParityTests`](../../InferenceWeb.Tests/Gemma4BatchedFusedDecodeParityTests.cs)，`TS_TEST_GGML_BACKEND=cuda`，gemma-4-E4B-it-Q8_0）：2、3、4 个并发序列（其中一个 prefill 超过 512 token 的 SWA 环）下，批量路径 12 步的贪心续写与轮询单 token decode 逐 token 一致，且每一步都跑在批量内核上。

**实测**（gemma-4-E4B-it-Q8_0、NVIDIA A40、ggml_cuda、f16 KV、prefill 分块 512、4 个运行序列；轮询列用 `TS_GEMMA4_BATCHED_CAPS=0` 强制 v1 门控，其余完全相同）：

| 负载 | 轮询（之前） | token 批量（之后） | 倍数 |
|---|---:|---:|---:|
| AgentTurnBench `conc`，1 个请求（decode tok/s） | 68.8 | 68.5 | 1.0× |
| AgentTurnBench `conc`，2 路并发（总 decode tok/s） | 65.1 | 99.0 | 1.5× |
| AgentTurnBench `conc`，4 路并发（总 decode tok/s） | 66.5 | 148.7 | 2.2× |
| `validate_inference.py` `decode`（512 个新 token），并发 1（端到端 tok/s） | 76.8 | 77.7 | 1.0× |
| `validate_inference.py` `decode`，并发 4（总端到端 tok/s） | 70.3 | 148.6 | 2.1× |
| `validate_inference.py` `decode_8k`（8k prompt），并发 1 | 63.8 | 64.3 | 1.0× |
| `validate_inference.py` `decode_8k`，并发 4 | 59.0 | 101.7 | 1.7× |

`validate_inference.py` 各行比较的是本树构建的服务器与上一提交构建的服务器；两者单路（并发 1）输出逐字节一致。这里"逐 token 一致"的确切含义：AgentTurnBench `conc` 流（`compare.py` 要求输出 token id 完全相同）与 12 步的进程内 parity 测试与轮询 decode 完全一致；但在数百个贪心 token 之后，边际很小的 token 可能翻转（`ParityHarness --batched` 256 步：每组 2/3/4 路里各有一个序列分叉），而原有的 v1 内核在 gemma-4-12B（无 PLE、无共享 KV）上同样如此 —— 批处理改变 GEMM 形状从而改变舍入，这正是 GLM 批量 decode 已记录的注意事项。经 HTTP 服务器在并发 4 下，连两次轮询运行都会不同（16 个 case 中 10 个相同），因为 prefill/decode 的交错取决于调度，所以 parity 必须在进程内判断。

## 10. 内存与 KV cache 策略

- **SWA 层**：容量 `_slidingWindow`，环形读写。
- **全局层**：容量 `maxSeqLen`，线性 append/读。
- **共享层**：alias 到 donor 的 cache，无独立分配。
- **量化权重绑定**：在 GGML CPU / Metal / CUDA 上零拷贝 mmap（GGUF 文件用 `MemoryMappedFile` + `QuantizedWeight.CreateExternalView`）。Direct CUDA 把量化数据上传到设备一次，释放 host 拷贝。

## 11. 批处理 / 分页前向（连续批处理）

Gemma 4 提供完整的 `IBatchedPagedModel.ForwardBatch` 移植
（[`Gemma4Model.BatchedForward.cs`](../../TensorSharp.Models/Models/Gemma4/Gemma4Model.BatchedForward.cs)），
通过共享 `InferenceEngine` 连续批处理栈执行
（[`docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md`](../PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md)）。
与多数批处理移植不同，Gemma 4 **默认启用**；设置 `TS_GEMMA4_BATCHED=0`
可强制回退到旧单序列 KV 交换路径，用于调试或者旧的融合单调用 decode 更快
的 batch=1 工作负载。

Gemma 4 是 TensorSharp 中最难移植到分页批处理的模型，因为它有三种引擎默认
假设跨层一致而 Gemma 4 实际不一致的异构源：

- **异构 head dim 与 KV head 数量**：local 层用 `head_dim_local`（通常
  256）和与 global 层不同的 `num_kv_heads`（`head_dim_global` 通常 512）。
  `EnsureGemma4PagedBuffers` 为每层分配各自的 `numKvHeads * headDim` 大小
  的分页 K/V 缓冲。`_g4PagedKvDimPerLayer[layer]` 记录每层维度，方便
  scatter / gather 步骤知道每层缓冲的 stride。
- **逐层 SWA 派发**：原生分页注意力内核每次调用接受一个 `sliding_window`
  参数。批处理路径会逐层计算 `IsLocalLayer(l) ? _slidingWindow : 0`，并
  把它传给 `GgmlBasicOps.PagedAttentionForward` —— 这样 local 层得到 SWA
  窗口掩码、global 层得到完整注意力 —— 都发生在同一次 `ForwardBatch` 调用
  内部。
- **KV donor 层别名**：24-41 层和更早的层共享 K/V。"receiver" 层的批处理缓
  冲是 donor 层缓冲的指针别名（`BlockPool` 内部的引用计数共享），而不是单
  独的分配。这保留了旧的 KV 共享语义，调度器无需感知别名。

其余的批处理路径机制与 Mistral 3 一致：

- **逐 token NeoX RoPE** 通过 `Ops.RoPEExWithFreqFactors` 派发，使用每层比例
  / 部分频率因子与显式 `positions[]` tensor。
- **基于 `slotMapping` 的 K/V 写入** 进入每层的分页缓冲。
  `EnsureGemma4PagedBuffers` 在扩容时拷贝已有 K/V，保证已经在调度中的序列
  保留状态。
- **Per-Layer Embedding（PLE）** 在 `ForwardBatch` 开头**整批计算一次**
  （`ComputePLE(flatTokens, hiddenStates, numTokens)`），其每层切片在 post-
  attention residual 之后被加上，与旧路径的注入点一致。
- **多模态嵌入注入** 走与旧 forward 相同的逐行 `InjectMultimodalEmbeddings`
  路径；嵌入会按正确的绝对 token 位置写入拼接后的批处理 hidden state。

**已验证的正确性**
（[`Gemma4BatchedForwardTests`](../../InferenceWeb.Tests/Gemma4BatchedForwardTests.cs)）：
- 全部 42 层（SWA + GQA + PLE + KV-donor 共享自 L24+）的逐层校验和与旧
  unfused 路径在 FP 噪声范围内一致。
- Logit cosine ≥ 0.99，与旧的非批处理路径对比。
- `EngineParallelInferenceTests.Gemma4_ThreeLongGenerationsParallel` 通过引擎
  验证多序列批处理路径。

**吞吐**（gemma-4-E4B-it-Q8_0、Apple M4 Pro、GgmlMetal —— 进程内切换
`TS_GEMMA4_BATCHED`，详见
[`Gemma4BatchedPerfBench.cs`](../../InferenceWeb.Tests/Gemma4BatchedPerfBench.cs)）：

| 工作负载 | n | Prompt token | 旧 tps | 批处理 tps | 加速 |
|---|---|---|---|---|---|
| 单序列短 prompt | 1 | 29 | 14.0 | 4.9 | **0.35×**（批处理慢） |
| 5 个并行短 prompt | 5 | 142 | 10.2 | 13.5 | **1.32×** |
| 8 个并行短 prompt | 8 | 218 | 10.0 | 15.1 | **1.51×** |
| 4 个并行长 prompt | 4 | 3 293 | 3.4 | 5.4 | **1.61×** |

加速随 batch 大小增长：batch=8 短 prompt 时分页图构建 / gather 开销已被完
全摊销。单序列是净亏，因为旧的融合单调用 decode 在没有批处理摊销的情况下
快约 3 倍 —— 这也是 `TS_GEMMA4_BATCHED=0` 存在的原因（用于 batch=1 工作
负载）。

**移植过程中修复的两个已知 bring-up bug**（现已纳入回归测试）：

1. `TSGgml_PagedAttentionForward` 在 permute Q 后没有调用 `ggml_cont`。当
   Q 是非连续 view 时，Metal 上的 `ggml_flash_attn_ext` 会静默产出错误结果。
2. `EnsurePagedBuffers` 之前在扩容时会破坏性重建 K/V 数组，把同一 batch 中
   先调度过的序列已经写入的 K/V 抹掉。第一个序列在后续序列加入后做首次
   decode 时会退化为单 token 循环。修复（grow-on-copy）也用于 Mistral 3。

## 12. MTP 投机解码（gemma4-assistant 草稿头）

Gemma 4 在两个宿主上都支持为单序列（无并发）请求做无损的**多 token
预测（MTP）投机解码**。与 Qwen 3.6 把 NextN 块内嵌在主干 GGUF 不同，Gemma 4 的草稿头
作为一个**独立的小 `gemma4-assistant` GGUF** 发布，通过 `--draft-model`
（环境变量 `TS_SPEC_DRAFT_MODEL`，旧名 `TS_MTP_DRAFT_MODEL`）加载，
并由 [`SpeculativeDraftHeadLoader`](../../TensorSharp.Models/SpeculativeDraftHeadLoader.cs)
在启动时挂到目标模型上。源码：
[`Gemma4Model.Speculative.cs`](../../TensorSharp.Models/Models/Gemma4/Gemma4Model.Speculative.cs)，
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

### 12.3 后端收益与融合内核

`SpeculationProfitable`
（[`Gemma4Model.Speculative.cs`](../../TensorSharp.Models/Models/Gemma4/Gemma4Model.Speculative.cs)）
决定是否真正启用投机：

- **ggml 后端（CUDA / Metal）** —— 运行融合单图内核：多 token 验证
  （`NativeGemma4ModelVerify`，26B-A4B MoE 用 `TryFusedMoEModelVerify`）和融合草稿步
  （`NativeGemma4DraftStep`）。部分接受时用稠密快速回滚避免重跑已保留前缀（逃生开关
  `TS_GMTP_NO_FAST_ROLLBACK=1`）。验证主干对单序列投机默认走线性路径；
  `TS_GMTP_BATCHED_TRUNK=1` 可改走批量分页主干。`TS_GMTP_NO_FUSED=1` 回退到逐算子路径用于
  A/B 测试。
- **Direct CUDA（`cuda`，纯 C#）** —— 没有融合内核，但其逐算子验证与草稿完全驻留 GPU：草稿在
  设备上读取 donor 缓存注意力，global 验证注意力对每行用 GQA decode 内核打实时缓存，global
  RoPE 用 GPU 内核——因此验证层循环零宿主端同步停顿。在散文 / 聊天负载上有收益，在低接受率的
  贪心解码上约持平。
- **CPU / MLX** —— 既无融合内核也无驻留 GPU 的逐算子路径，因此投机关闭，引擎走标准融合 decode。

26B-A4B MoE 目标还需要在 `ggml_cuda` 上修复加载期 OOM（跳过 per-expert 设备预加载）后，MTP
投机才成为净收益。

启用即传入 `--draft-model` 文件本身；关闭（`--no-spec`）与调优（`--spec-draft` /
`--spec-pmin`）使用通用参数，它们由 `TensorSharp.Cli` 与
`TensorSharp.Server` 共用的同一个
[`SpeculativeCliFlags`](../../TensorSharp.Runtime/Speculative/SpeculativeCliFlags.cs) 解析；
完整参数列表与其他算法见[投机解码](../../FEATURES_zh-cn.md#投机解码) —— 其中
`--spec-type ngram` 完全不需要草稿 GGUF，因此即便旁边没有 assistant 文件，也能在
Gemma 4 checkpoint 上运行。

## 13. 输出解析器与聊天模板

`Gemma4OutputParser` 处理两种结构化包装：

- **思维链** —— `<|channel>thought ... <channel|>` 的 chain-of-thought，再跟最终答案。
- **工具调用** —— `<|tool_call>call:function_name{...args...}<tool_call|>` 块，由 `OutputParser` 解出结构化的 tool call。参数使用 Gemma 自己的语法（裸键名，字符串用 `<|"|>` 包裹），而模型经常把看起来像标识符的字符串值直接裸写——`call:read_invoice{invoice_id:INV-472}`、`{path:src/main.py}`、`{ids:[INV-1, INV-2]}`。转换器会给每个不是 JSON 数字 / `true` / `false` / `null` 的裸值加引号（数组内也一样；数字保持数字）。参数仍无法解析的调用会以原文作为 content 返回，而不是一条空消息；多调用轮次中每个调用都带自己的 `index`，流式客户端可据此配对参数增量。

聊天模板在 GGUF 没带 Jinja2 模板时回退到内置 Gemma 4 模板。

## 13a. 张量并行

Gemma 4 在 Direct `cuda` 后端以及 GGML CUDA / Vulkan 后端上都支持 `--tp N`，并且
多模态嵌入会被注入 TP 路径，因此切分后视觉 / 音频提示不会丢失。

MoE 变体在 GGML 上需要特殊处理。整模 MoE 内核（`TSGgml_Gemma4MoEModelDecode` /
`...Verify`）在计算图内部完成路由，`ggml_top_k` 返回的是*全局*专家 id，整专家切分
无法喂给它们。因此 TP 下的 MoE 层改为在**每个专家内部**按 Megatron 方式切分：每个
rank 保留全部 128 个专家，但只持有每个专家 FFN 宽度的 `1/tp`（gate/up 沿中间维度
列并行，保持融合的 `[gate_r | up_r]` 布局；down 沿该维度按整量化块行并行）。
`sel` 保持全局，计算图与单卡一致、只是专家矩阵更窄，而专家求和的输出成为该层的第
三个行并行部分和 —— 在路由加权求和之后立即归约；由于该运算是线性的，"先求和再加
权"与"先加权再求和"等价。这些切片在加载时物化（26B 上约 10.5 GB，约 36 秒）；
`TS_GEMMA4_TP_FUSED_MOE=0` 可回退到逐算子的整专家路径。

在 2× RTX 2000 Ada 上实测（prefill 512 / decode 64，tok/s）：E4B Q8_0 由单卡的
2760 / 37.3 变为 `--tp 2` 的 2488 / **51.7**；26B-A4B IQ4_XS 由 1845 / 48.5 变为
2537 / **51.2**。两者的输出都与单卡运行**逐字节一致**。完整细节见
[`TENSOR_PARALLELISM_PLAN.md`](../../TENSOR_PARALLELISM_PLAN.md)。

## 14. 优化机会

- **混合 dense+MoE 布局的融合 MoE kernel** —— 全 MoE 变体（例如 26B-A4B）现在已经运行融合整模型 MoE decode（`TryFusedMoEModelDecode` / `TSGgml_Gemma4MoEModelDecode`）与 prefill/verify（`TryFusedMoEModelVerify`），但假想中混合密集层与 MoE 层的模型仍会回退到 per-layer 图。（下文的 expert-batched FFN 已经把未融合路径里的顺序 per-expert 派发去掉了。）
- **GPU 上的音频 prefill** —— 音频编码器的 conv 子采样仍跑在 CPU。把 conv 栈搬到 Metal / CUDA 可以降低长音频提示的 TTFT。

### 已完成

- **MTP 投机解码（gemma4-assistant 草稿）** —— 独立的 EAGLE 风格草稿 GGUF（`--draft-model`）
  加速单序列 decode（§12）。ggml 后端运行融合多 token 验证（稠密 `NativeGemma4ModelVerify`
  与 MoE `TryFusedMoEModelVerify`）和融合草稿步（`NativeGemma4DraftStep`）内核，部分接受时用
  稠密快速回滚；纯 C# `cuda` 后端运行完全驻留 GPU 的逐算子验证 / 草稿。26B-A4B MoE 目标还需要
  先修复 `ggml_cuda` 加载期 OOM（跳过 per-expert 设备预加载），投机才成为净收益。草稿 GGUF
  不匹配 / 不完整会在启动时立即失败。
- **Expert-batched FFN（GEGLU）** —— 过去 `MoEForward` 里即便按 expert 分批，decode 路径依然退化成 `num_experts_used` 次单行 matmul。现在整层都通过 `GgmlBasicOps.MoEFFNPrefill(..., MoEActivation.GEGLUSplit)` 一次派发完成：2~3 次 `ggml_mul_mat_id`（gate[+up] / down）加上融合的 `ggml_geglu_split` 激活与专家聚合，每个 MoE 层提交常数个 GGML 图 —— 与 `seq_len`、`num_experts_used` 无关。kernel 直接消费原始的 3D `ffn_gate_up_exps.weight` / `ffn_down_exps.weight`（Apple Silicon 上是 mmap 视图，Windows / Linux 上是共享 buffer，零拷贝）。`ffn_down_exps.scale` 这种 per-expert 因子在 C# 侧提前折进 routing weights，让原生 kernel 保持与激活函数解耦。层级 stacked view 不可用时（比如 F32-only 张量）会回退到原来的 batched-by-expert C# 路径。
