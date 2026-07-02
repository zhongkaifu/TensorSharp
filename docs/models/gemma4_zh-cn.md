# Gemma 4

[← 返回模型索引](README_zh-cn.md) | [English](gemma4.md)

| 属性 | 值 |
|---|---|
| 提供方 | Google |
| GGUF 架构标识 | `gemma4` |
| 模型类 | [`Gemma4Model`](../../TensorSharp.Models/Models/Gemma4/Gemma4Model.cs)（旧单序列路径）+ [`Gemma4Model.BatchedForward.cs`](../../TensorSharp.Models/Models/Gemma4/Gemma4Model.BatchedForward.cs)（`IBatchedPagedModel`） |
| 视觉编码器 | [`Gemma4VisionEncoder`](../../TensorSharp.Models/Models/Gemma4/Gemma4VisionEncoder.cs)（SigLIP 风格 ViT） |
| 音频编码器 | [`Gemma4AudioEncoder`](../../TensorSharp.Models/Models/Gemma4/Gemma4AudioEncoder.cs)（USM 风格 chunked transformer） |
| 音频前端 | [`Gemma4AudioPreprocessor`](../../TensorSharp.Models/Models/Gemma4/Gemma4AudioPreprocessor.cs)（16 kHz 单声道 → 128 bin log-mel） |
| 图像处理器 | [`Gemma4ImageProcessor`](../../TensorSharp.Models/Models/Gemma4/Gemma4ImageProcessor.cs) |
| 示例模型 | gemma-4-E4B（8B 等效）、gemma-4-31B、gemma-4-26B-A4B（MoE） |
| 模态 | 文本、图像、视频（帧栈）、音频 |
| 思维链模式 | 是（`<\|channel>thought ... <channel\|>`） |
| 工具调用 | 是（`<\|tool_call>call:name{...}<tool_call\|>`） |
| 批处理 / 分页前向 | **默认启用** —— `IBatchedPagedModel.ForwardBatch` 处理双 head_dim、KV donor 共享、PLE 注入、SWA + 全局混合的分页 K/V 缓冲。设置 `TS_GEMMA4_BATCHED=0` 可强制回退到旧单序列 KV 交换路径。详见 §11。 |
| MTP 投机解码 | 可选 —— 通过 `--mtp-draft-model`（`TS_MTP_DRAFT_MODEL`）加载独立的 `gemma4-assistant` EAGLE 风格草稿 GGUF，并用 `--mtp-spec` 启用。在 ggml 后端与纯 C# `cuda` 后端上有收益。详见 §12。 |
| 输出解析器 | `Gemma4OutputParser` |

## 1. 来源与目标

Gemma 4 是 TensorSharp 当前支持的功能最丰富的架构。它把 Google Gemma 系扩展到：

- **per-layer 异构性**：每一层可独立选择注意力模式（SWA / 全局）、head 维度、KV head 数，以及是否与某个早期「donor」层共享 KV。
- **Per-Layer Embedding（PLE）**：从 `per_layer_token_embd.weight` 取出的小型 side embedding 在每个 block 内混入 residual stream。
- **MoE 变体**（如 `gemma-4-26B-A4B`）：每个 block 同时跑一个密集 MLP 与一个稀疏 MoE 分支，分别经过各自的 post-norm 后求和。
- **真正的多模态**：图像、视频帧栈与音频共用同一 residual stream。

相对 Gemma 3，SWA mask 与 RoPE 表跨层缓存，SWA 缓存改为环形（内存与上下文长度无关），整套 transformer 在 decode 时可以一次 GGML 图调度完成。

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
- **Prefill**（`seqLen > 1`）：每个符合条件的密集层走融合 per-layer GGML 图（`Gemma4LayerPrefill`）。MoE / KV 共享 / 当前 chunk 中有 PLE 注入的层回退到逐算子托管路径。长 prompt 切成 `min(2 × slidingWindow, 2048)` 的块，控制 SWA 层 score 张量的大小。

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

与 Gemma 3 一致：当 `gemma4.final_logit_softcapping > 0` 时走 `tanh(logits / cap) * cap`。

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

在 ggml 后端上，普通的多 token prefill 由 MTP 验证所用的同一个融合整模型内核（§12）来执行：所有层在单次 GGML 图派发中完成，激活值常驻设备，而不是每层一张图。`CanUseWholeModelPrefillVerify()` 决定是否走该路径——仅限密集模型；多模态 chunk 在 `startPos == 0` 时可通过内核的双向 span mask 走该路径（`TS_G4_MM_PREFILL=0` 让多模态退回逐算子路径）。`startPos > 0` 的 SWA 包裹 chunk 通过内核内的 swaPrev gather 留在融合路径上（`TS_G4_VERIFY_SWAPREV=0` 关闭）。全 MoE 变体（例如 26B-A4B）有对应的融合路径：`CanUseWholeModelMoEPrefillVerify()` / `TryFusedMoEModelVerify()`。设 `TS_G4_WHOLE_PREFILL=0` 可强制走逐算子分块路径做 A/B。注意，块量化（`q8_0` / `q4_0`）KV cache 的多 token prefill *必须*走该路径——逐算子回退无法遍历块量化的 cache 布局。

调度器会把 solo（无争用）prompt 以大分块喂给该路径：全新分块上限由 `TS_SCHED_SOLO_PREFILL_CHUNK`（默认 8192）控制，尾部分块由 `TS_SCHED_SOLO_TAIL_PREFILL_CHUNK`（默认 2048）控制。实测设计见 [`docs/perf/gemma4-prefill-cuda-graph-design.md`](../perf/gemma4-prefill-cuda-graph-design.md)。

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
   decode 时会退化为单 token 循环。修复（grow-on-copy）也用于 Mistral 3
   和 Qwen 3。

## 12. MTP 投机解码（gemma4-assistant 草稿头）

Gemma 4 在 `TensorSharp.Server` 中支持为单序列（无并发）请求做无损的**多 token
预测（MTP）投机解码**。与 Qwen 3.6 把 NextN 块内嵌在主干 GGUF 不同，Gemma 4 的草稿头
作为一个**独立的小 `gemma4-assistant` GGUF** 发布，通过 `--mtp-draft-model`（环境变量
`TS_MTP_DRAFT_MODEL`）加载。源码：
[`Gemma4Model.Mtp.cs`](../../TensorSharp.Models/Models/Gemma4/Gemma4Model.Mtp.cs)，
由共享的
[`MtpSpeculativeExecution`](../../TensorSharp.Runtime/Scheduling/MtpSpeculativeExecution.cs)
起草 / 验证 / 回滚核心驱动。

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
12B 目标配 12B 草稿，而非 26B-A4B 草稿。当给了 `--mtp-draft-model` 但草稿无法激活（文件缺失、
隐藏维不匹配、或缺少必要的草稿张量）时，服务端会在启动时**立即失败**并给出修复提示
（[`MtpStartupValidation`](../../TensorSharp.Server/Hosting/MtpStartupValidation.cs)），
而不是静默地不做投机继续运行。

### 12.3 后端收益与融合内核

`MtpSpeculationProfitable` 决定是否真正启用投机：

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

启用 / 关闭与调优使用通用的 `--mtp-spec` / `--mtp-draft` / `--mtp-pmin` 参数；见
[README MTP 章节](../../README_zh-cn.md#mtp--nextn-投机解码)。

## 13. 输出解析器与聊天模板

`Gemma4OutputParser` 处理两种结构化包装：

- **思维链** —— `<|channel>thought ... <channel|>` 的 chain-of-thought，再跟最终答案。
- **工具调用** —— `<|tool_call>call:function_name{...args...}<tool_call|>` 块，由 `OutputParser` 解出结构化的 tool call。

聊天模板在 GGUF 没带 Jinja2 模板时回退到内置 Gemma 4 模板。

## 14. 优化机会

- **混合 dense+MoE 布局的融合 MoE kernel** —— 全 MoE 变体（例如 26B-A4B）现在已经运行融合整模型 MoE decode（`TryFusedMoEModelDecode` / `TSGgml_Gemma4MoEModelDecode`）与 prefill/verify（`TryFusedMoEModelVerify`），但假想中混合密集层与 MoE 层的模型仍会回退到 per-layer 图。（下文的 expert-batched FFN 已经把未融合路径里的顺序 per-expert 派发去掉了。）
- **GPU 上的音频 prefill** —— 音频编码器的 conv 子采样仍跑在 CPU。把 conv 栈搬到 Metal / CUDA 可以降低长音频提示的 TTFT。

### 已完成

- **MTP 投机解码（gemma4-assistant 草稿）** —— 独立的 EAGLE 风格草稿 GGUF（`--mtp-draft-model`）
  加速单序列 decode（§12）。ggml 后端运行融合多 token 验证（稠密 `NativeGemma4ModelVerify`
  与 MoE `TryFusedMoEModelVerify`）和融合草稿步（`NativeGemma4DraftStep`）内核，部分接受时用
  稠密快速回滚；纯 C# `cuda` 后端运行完全驻留 GPU 的逐算子验证 / 草稿。26B-A4B MoE 目标还需要
  先修复 `ggml_cuda` 加载期 OOM（跳过 per-expert 设备预加载），投机才成为净收益。草稿 GGUF
  不匹配 / 不完整会在启动时立即失败。
- **Expert-batched FFN（GEGLU）** —— 过去 `MoEForward` 里即便按 expert 分批，decode 路径依然退化成 `num_experts_used` 次单行 matmul。现在整层都通过 `GgmlBasicOps.MoEFFNPrefill(..., MoEActivation.GEGLUSplit)` 一次派发完成：2~3 次 `ggml_mul_mat_id`（gate[+up] / down）加上融合的 `ggml_geglu_split` 激活与专家聚合，每个 MoE 层提交常数个 GGML 图 —— 与 `seq_len`、`num_experts_used` 无关。kernel 直接消费原始的 3D `ffn_gate_up_exps.weight` / `ffn_down_exps.weight`（Apple Silicon 上是 mmap 视图，Windows / Linux 上是共享 buffer，零拷贝）。`ffn_down_exps.scale` 这种 per-expert 因子在 C# 侧提前折进 routing weights，让原生 kernel 保持与激活函数解耦。层级 stacked view 不可用时（比如 F32-only 张量）会回退到原来的 batched-by-expert C# 路径。
