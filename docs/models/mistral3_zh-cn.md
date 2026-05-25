# Mistral 3

[← 返回模型索引](README_zh-cn.md) | [English](mistral3.md)

| 属性 | 值 |
|---|---|
| 提供方 | Mistral AI |
| GGUF 架构标识 | `mistral3` |
| 模型类 | [`Mistral3Model`](../../TensorSharp.Models/Models/Mistral3/Mistral3Model.cs)（旧单序列路径）+ [`Mistral3Model.BatchedForward.cs`](../../TensorSharp.Models/Models/Mistral3/Mistral3Model.BatchedForward.cs)（`IBatchedPagedModel`） |
| 视觉编码器 | [`Mistral3VisionEncoder`](../../TensorSharp.Models/Models/Mistral3/Mistral3VisionEncoder.cs)（Pixtral） |
| 图像处理器 | [`Mistral3ImageProcessor`](../../TensorSharp.Models/Models/Mistral3/Mistral3ImageProcessor.cs) |
| 示例模型 | Mistral-Small-3.1-24B-Instruct、Ministral-3-14B-Instruct |
| 模态 | 文本、图像 |
| 思维链模式 | 否 |
| 工具调用 | 否 |
| 批处理 / 分页前向 | **默认启用** —— `IBatchedPagedModel.ForwardBatch` 的参考实现。已在 Ministral-3-14B 上完成端到端验证；原生分页注意力内核在长上下文下比旧路径快约 21%。详见 §11。 |
| 输出解析器 | `PassthroughOutputParser` |

## 1. 来源与目标

Mistral 3 是 Mistral 第三代 LLaMA 风格密集 transformer，带两个显著扩展：

- **YaRN 校正 RoPE** —— 在原 RoPE 调度与外推调度之间做 per-frequency 部分插值。能让模型在远超原始训练上下文的长度上仍然干净 attend。
- **位置相关 Q 缩放** —— 一旦位置超过原始上下文长度，Q 乘以 `1 + β · log(1 + ⌊pos / origCtx⌋)`，让 attention 幅度在长上下文上保持良态。
- **Pixtral 视觉编码器** 用于图像输入。

其余架构是标准 LLaMA / Mistral 配方：GQA、无 QK-norm、SwiGLU FFN、RMSNorm、GPT-J 风格 RoPE 配对、可选 tied LM head。

## 2. 模型架构

```
                    tokens (int[])
                          │
                    token_embd.weight
                          │
        [可选]  InjectVisionEmbeddings (image tokens)
                          │
        ┌──── × NumLayers ──────────────────────────────┐
        │  RMSNorm(attn_norm)                            │
        │  Q, K, V 投影（或融合 QKV）                    │
        │  RoPE_GPT-J(Q, K) with YaRN-corrected freqs    │
        │  Q ← Q * (1 + β · log(1 + ⌊pos / origCtx⌋))    │
        │  attention(Q, KCache, VCache) (全因果)         │
        │  attn_output ─► residual                        │
        │  RMSNorm(ffn_norm)                              │
        │  ffn_gate_up ─► [gate ‖ up]                    │
        │  SwiGLU: SiLU(gate) * up                        │
        │  ffn_down ─► residual                           │
        └────────────────────────────────────────────────┘
                          │
                    RMSNorm(output_norm)
                          │
                    LM head (output.weight 或 tied)
                          │
                          ▼
                       logits
```

## 3. 前向计算图

每层 L：

```
hidden ─► RMSNorm(attn_norm.weight, eps)
       ─► QKV (加载时融合则用 attn_qkv.weight，
                否则三次独立的 attn_q/k/v matmul)
       ─► RoPE_GPT-J(Q, K, ropeFreqs[]) at positions startPos..startPos+seqLen-1
            • GPT-J 风格: 配对相邻元素 (x[2i], x[2i+1])
            • ropeType == "yarn" 时频率经 YaRN 校正
       ─► [if _ropeOrigCtx > 0]:
            scale = 1 + _ropeScalingBeta * log(1 + floor(pos / _ropeOrigCtx))
            Q ← Q * scale
       ─► Q ← Q * (1/sqrt(headDim))
       ─► append (K, V) 到 KV cache
       ─► attention(Q, KCache, VCache)        # 标准因果 GQA
       ─► attn_output.weight matmul → o
       ─► hidden = hidden + o
       ─► h2 = RMSNorm(ffn_norm.weight, eps) on hidden
       ─► ffn_gate_up.weight matmul → [gate ‖ up]
       ─► g = SiLU(gate)
       ─► h3 = ffn_down.weight × (g * up)
       ─► hidden = hidden + h3
```

所有层完成后：

```
hidden ─► narrow(seq_len-1) if prefill
       ─► RMSNorm(output_norm.weight, eps)
       ─► matmul against output.weight (tied 时为 token_embd.weight)
       ─► 拷贝到 float[VocabSize]
```

## 4. 组件细节

### 4.1 Attention

- **GQA**，`Config.NumKVHeads < Config.NumHeads`。支持独立的 `key_length` 与 `value_length`（`_attnKeyLen`、`_attnValLen`），未设置时缺省为 `Config.HeadDim`。
- **可选融合 QKV**。`FuseQKVWeights()` 在 GGUF 分散存放 Q / K / V 时构建 `attn_qkv.weight`。Decode 用 `_layerQkvFused[l]` per-layer 选择融合 matmul 还是三次独立 matmul。
- **没有 QK-norm** —— 与 Qwen 3、Gemma 3、Gemma 4 不同，Mistral 3 跳过 Q、K 的 per-head RMSNorm。

### 4.2 RoPE —— GPT-J 风格 + YaRN

- **配对**：相邻元素 `(x[2i], x[2i+1])` 一起旋转（GPT-J 约定），与 NeoX 的 `(x[i], x[i + halfDim])` 配对相反。
- **YaRN 频率校正**：当 `mistral3.rope.scaling.type == "yarn"` 且 `mistral3.rope.scaling.original_context_length > 0` 时，`ApplyYarnFreqCorrection()` 根据每个频段属于「慢」还是「快」rotation 范围，在外推与插值频率之间做插值：

  ```
  lowFreqWavelen  = origCtx / betaSlow
  highFreqWavelen = origCtx / betaFast
  对每个 freq f:
      wavelen = 2π / f
      if wavelen < highFreqWavelen:        # 高频段: 外推
          f 不变
      elif wavelen > lowFreqWavelen:       # 低频段: 插值
          f *= 1 / scale
      else:                                # 中频段: 平滑插值
          ramp = ...                       # 两者之间线性 ramp
          f = mix(f, f / scale, ramp)
  ```

- **位置相关 Q 缩放**：`_ropeOrigCtx > 0` 时，
  ```
  q *= 1 + _ropeScalingBeta * log(1 + floor(pos / _ropeOrigCtx))
  ```
  让 attention 幅度在超过原始上下文长度后仍然良态。

### 4.3 FFN —— SwiGLU

`ffn_gate_up.weight` 拼成 `[gate ‖ up]`。`Ops.SiLUMul` 计算 `SiLU(gate) * up`，再过 `ffn_down`。无 bias。

### 4.4 Embedding 与 LM head

`token_embd.weight` 是输入 embedding。LM head 当 `output.weight` 存在时使用它，否则与 `token_embd.weight` tied（`_hasTiedOutput`）。

### 4.5 视觉管线（Pixtral）

`Mistral3VisionEncoder` 是 Pixtral 架构：

- **Conv2D patch embedding**：`v.patch_conv.weight`（形状 `[hidden, channels, patchSize, patchSize]`）+ 可选 bias。
- **RMSNorm** 在 encoder 输入。
- **2D RoPE** 空间位置编码。
- **SiLU-gated MLP transformer block**：LayerNorm + 多头 attention + residual + LayerNorm + SiLU-gated MLP + residual。
- **Spatial patch merging**：在 projector 边界把相邻 patch 合并为一个 token。
- **多模态 projector**：RMSNorm → PatchMerger → Linear → GELU → Linear，映射到 LM hidden 维。

`Mistral3ImageProcessor` 保比例 resize 图像：

1. RGBA 合成到白底。
2. 保持比例 resize，使最长边等于 `longest_edge`。
3. Pad 到 `patch_size` 的整数倍。
4. 使用 CLIP 默认 mean / std 归一化。

多模态注入器（`ModelMultimodalInjector.cs` 中的 `ProcessMistral3History`）走聊天历史，每张唯一图像跑一次 encoder，并把 `PreparedEmbeddingSpan` 入队，让 embedding 在 `Forward()` 之前落到 prompt 中正确位置。

## 5. 参数与配置

| Key | 类型 | 含义 |
|---|---|---|
| `mistral3.rope.dimension_count` | uint32 | RoPE 维数 |
| `mistral3.rope.scaling.type` | string | RoPE scaling 类型（如 `yarn`） |
| `mistral3.attention.temperature_scale` | float32 | 位置相关 Q scaling β（默认 0.1） |
| `mistral3.rope.scaling.original_context_length` | uint32 | YaRN 原始上下文长度 |
| `mistral3.rope.scaling.extrapolation_factor` | float32 | YaRN 外推因子（默认 1.0） |
| `mistral3.rope.scaling.yarn_beta_fast` | float32 | YaRN 快速 rotation 阈值（默认 32.0） |
| `mistral3.rope.scaling.yarn_beta_slow` | float32 | YaRN 慢速 rotation 阈值（默认 1.0） |
| `mistral3.rope.scaling.mscale` | float32 | YaRN mscale（默认 0） |
| `mistral3.rope.scaling.mscale_all_dim` | float32 | YaRN mscale_all_dim（默认 0） |

Pixtral 视觉 projector（`mmproj` GGUF）使用标准 `clip.vision.*` keys 与 `mm.*` keys 描述编码器维度与 projector 层。

## 6. 权重命名约定

```
token_embd.weight                          # [vocab, hidden]
blk.{L}.attn_norm.weight                   # 注意力前 RMSNorm
blk.{L}.attn_q.weight                      # Q 投影 (融合前)
blk.{L}.attn_k.weight                      # K 投影 (融合前)
blk.{L}.attn_v.weight                      # V 投影 (融合前)
blk.{L}.attn_qkv.weight                    # 融合 Q+K+V (融合后)
blk.{L}.attn_output.weight                 # 输出投影
blk.{L}.ffn_norm.weight                    # FFN 前 RMSNorm
blk.{L}.ffn_gate.weight  }                 # 融合前
blk.{L}.ffn_up.weight    }
blk.{L}.ffn_gate_up.weight                 # 融合后: [2*intermediate, hidden]
blk.{L}.ffn_down.weight                    # FFN down 投影
output_norm.weight                         # 最终 RMSNorm
output.weight                              # LM head (tied 时可选)
```

### 视觉编码器权重（Pixtral）

```
v.patch_conv.weight                        # Conv2D patch embedding [hidden, C, P, P]
v.patch_conv.bias                          # Conv2D bias (可选)
v.encoder_norm.weight                      # encoder 输入 RMSNorm
v.blk.{L}.attn_norm.weight                 # 注意力前 RMSNorm
v.blk.{L}.attn_q.weight                    # Q 投影
v.blk.{L}.attn_k.weight                    # K 投影
v.blk.{L}.attn_v.weight                    # V 投影
v.blk.{L}.attn_output.weight               # 输出投影
v.blk.{L}.ffn_norm.weight                  # FFN 前 RMSNorm
v.blk.{L}.ffn_gate.weight                  # SiLU gate
v.blk.{L}.ffn_up.weight                    # up 投影
v.blk.{L}.ffn_down.weight                  # down 投影
mm.norm.weight                             # projector RMSNorm
mm.patch_merger.merging_layer.weight       # 空间 patch merger
mm.linear_1.weight                         # projector linear 1
mm.linear_2.weight                         # projector linear 2
```

## 7. TensorSharp 实现走读

构造函数（`Mistral3Model(string ggufPath, BackendType backend)`）：

1. `ParseBaseConfig()`。
2. 从 `KeyLength` / `ValueLength` 读 `_attnKeyLen` / `_attnValLen`（缺省 `HeadDim`）。读 `_ropeDim`。
3. 读 YaRN 参数。
4. `ParseTokenizer()`、`LoadWeights()`。
5. `FuseQKVWeights()`、`FuseGateUpWeights()`。
6. `PrepareCudaQuantizedWeightsForInference()`。
7. `InitKVCache(maxSeqLen)` 为每层分配 K、V tensor，形状 `[NumKVHeads, maxSeqLen, headDim]`（独立 `_attnKeyLen` / `_attnValLen` 会被尊重）。
8. `PrecomputeConstants()` 构建：
   - per-layer 权重名数组 —— 形状取决于该层的 QKV / gate-up 是否融合。
   - RoPE 频率表 `_ropeFreqs[halfDim]`。
   - `_ropeType == "yarn"` 且 `_ropeOrigCtx > 0` 时通过 `ApplyYarnFreqCorrection()` 校正 YaRN 频率。

`Forward(int[] tokens)` 跑 per-op 托管循环：

- Embedding lookup。
- `<image_pad>` 标记位置可选视觉注入。
- 每层：RMSNorm，QKV（融合或拆开），带 YaRN 校正频率的 RoPE，可选位置相关 Q 缩放，attention，输出投影 + residual，FFN，residual。
- prefill 时 residual 在 `output_norm` 之前 narrow 到最后一个 token，让 LM head matmul 只产出一行。
- 最终 RMSNorm、LM head、拷贝到 `_logitsBuffer`。

## 8. Prefill 优化

- **融合 QKV 与 gate / up** 减少 per-layer GEMM 数。
- **末行 narrow** 在 `output_norm` 之前，让 LM head matmul 只输出 `[1, vocab]`。
- **YaRN 频率表加载时一次性构建**。Decode 路径复用同一份 `_ropeFreqs[]`。
- **后端算子** 通过 `Ops.RoPEEx` 与 `Ops.SiLUMul` 派发，GGML / CUDA 后端上以原生 kernel 运行。

Mistral 3 没有分块 prefill（只有全因果 attention，没有 SWA），没有融合 per-layer prefill kernel，也没有 whole-model 原生 decode。任何一项都能进一步压低 decode latency 与 TTFT。

## 9. Decode 优化

- 在 `PrecomputeConstants()` 中预分配的 per-layer 权重名数组（`_layerWeightNames[L][]`）让热循环零分配。
- `_layerQkvFused[]` 标志 per-layer 选择融合 / 拆开 QKV 路径，无分配。
- 预分配的 decode 位置数组与 YaRN 校正后的 `_ropeFreqs[]` 避免每 token 重新计算。
- 量化权重保留在 `_quantWeights`；matmul 通过后端的原生量化 matmul 调用。

## 10. 内存与 KV cache 策略

- 每层 K、V tensor，形状 `[NumKVHeads, maxSeqLen, headDim]`，per-head 维度使用 `_attnKeyLen` / `_attnValLen`。
- `ResetKVCache()` 全部清零并调用 `InvalidateTensorDeviceCache()` 同步 GPU 状态。
- `TruncateKVCache(int tokenCount)` 支持（服务器多轮 KV cache 复用使用）。
- Pixtral 视觉编码器在加载时把权重 dequantize 到 F32；语言模型权重当后端支持时保持量化。

## 11. 批处理 / 分页前向（连续批处理）

Mistral 3 是 TensorSharp 中 `IBatchedPagedModel.ForwardBatch` 的**参考实现**
（[`Mistral3Model.BatchedForward.cs`](../../TensorSharp.Models/Models/Mistral3/Mistral3Model.BatchedForward.cs)），
通过 [`docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md`](../PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md)
中描述的共享连续批处理栈（`InferenceEngine` + `ContinuousBatchScheduler` +
`BatchExecutor`）执行。

关键特性：

- **默认启用，无需 opt-in 环境变量。** Mistral 3 的连续批处理始终可用；服务的
  `--no-continuous-batching` 会强制所有模型（包括 Mistral 3）走旧的单序列
  KV 交换路径。
- **每层分页 K/V 缓冲区**，布局 `[numBlocks * blockSize * numKvHeads * headDim]`，
  由 `EnsurePagedBuffersAllocated` 按需扩容。"扩容"路径会把已有 K/V 拷贝到新
  缓冲区，避免已经在调度中的序列丢失状态。
- **逐层选择融合 vs 非融合 QKV**：`ForwardBatch` 根据 `_layerQkvFused[layer]`
  决定走融合的 `attn_qkv.weight` matmul 还是三次独立的 `attn_q/k/v.weight`
  matmul。两条路径都按 token 轴批量计算，每层只发起一次 GEMM。
- **逐 token YaRN RoPE**。加载时建立的 `_ropeFreqs[]` 数组通过 `Ops.RoPEEx`
  应用，并传入显式 `positions[]` 数组；逐 token 的 YaRN 位置相关 Q 缩放
  （`q *= 1 + beta * log(1 + floor(pos / orig_ctx))`）在批处理循环中执行，
  每个 token 取各自的缩放因子。
- **基于 `slotMapping` 的 K/V 写入** 把新的 K 与 V 写入层的分页缓冲区在
  `blockId * blockSize + offset` 位置。批处理路径下不再有序列间的 KV 状态
  extract / inject。
- **三种通过 `TS_PAGED_ATTN_KERNEL` 可选的分页注意力内核**：
  - `native`（默认）：`TSGgml_PagedAttentionForward` —— 在 C++ 中按序列对
    分页缓冲区做 memcpy gather，然后派发 `ggml_flash_attn_ext`（即旧单序列
    路径使用的同一融合 Metal/CUDA flash 注意力内核）。
  - `tensor`：`TensorPagedAttention.Forward` —— 基于 C# Tensor 的 gather
    加按序列的 `Ops.AddmmBatch` + `GgmlBasicOps.AttentionSoftmaxWithSinks`。
    比 `native` 慢，因为按序列重复派发 GPU。
  - `managed`：`ManagedPagedAttention.Forward` —— 纯 C# 在线 softmax 循环，
    按 `(seq, head)` 并行化。任意后端上的正确性回退。
- **视觉嵌入注入** 仍在 per-layer 循环之前（与旧 forward 路径一致）。多模态
  请求会被多模态注入器在 prompt 准备阶段串行化；纯文本请求可以并行准备。

**在真实 GGUF 上验证的一致性**
（[`Mistral3BatchedForwardTests`](../../InferenceWeb.Tests/Mistral3BatchedForwardTests.cs)）：
- `Mistral3_BatchSize1_ForwardBatchMatchesLegacyTop1` —— 6 token 探针 prompt
  上 legacy top-1 = 批处理 top-1 = 1091。
- `Mistral3_BatchSize2_DistinctSequencesProduceDistinctLogits` —— 一次批处
  理中的两个不同 prompt 产生不同的 top-1 token，证明 K/V 被正确分隔。

**Ministral-3-14B-Instruct-2512-Q4_K_M（Apple M4 Pro、GgmlMetal、4 个并行
聊天 prompt）实测吞吐**：

| 路径 | 长上下文（~800 tok/seq）墙钟 | tokens/sec | vs 旧路径 |
|---|---|---|---|
| 批处理 + 原生内核（默认） | **42.37 s** | **0.6** | **1.21×** |
| 单序列 KV 交换（旧 GGML） | 53.95 s | 0.4 | 1.0× |
| 批处理 + Tensor 内核 | 71.42 s | 0.3 | 0.76× |
| 批处理 + Managed 内核 | 111.02 s | 0.2 | 0.49× |

在长上下文下原生内核比旧 GGML 单序列路径快约 21% —— 这是整个连续批处理工作
的关键里程碑。

**前缀缓存验证**：在同一组长上下文运行中，引擎在 4 个序列间共享了 6 个完整
prompt 块（`reused=1536`、`hashedCached=3`），首次在真实 GGUF 上端到端验证
前缀缓存路径。

## 12. 输出解析器与聊天模板

- `PassthroughOutputParser` —— Mistral 3 没有思维链 / 工具调用 wire 格式。
- 聊天模板使用 Mistral 标准格式（`[INST]...[/INST]<s>...</s>`）。GGUF 缺少 Jinja2 模板时回退到内置硬编码模板。
- 图像占位符为 `<image_pad>`，`ChatTemplate.ExpandImageTokens` 把每个 `<image_pad>` 展开为对应图像所需 token 数。

## 13. 优化机会

- **原生 whole-model decode** —— 旧单序列 forward 仍是托管 C# + 后端派发
  matmul。原生单调用 decode 路径（类比 Qwen 3 的 `TransformerModelDecode`）
  能消除单序列路径上的大部分托管开销。
- **融合单调用 decode** —— `Mistral3ModelDecode` GGML kernel 能通过把
  per-layer 派发合并显著提升旧单序列路径上的 Metal / CUDA 吞吐。
- **量化视觉编码器** —— Pixtral 权重当前在加载时 dequantize 到 F32，图像繁
  重负载下显著增加内存占用。直接支持量化视觉权重能压低常驻内存并加快加载。
- **融合 prefill attention** —— 引入 Qwen 3.5 的 `FusedPrefillAttention` 思
  路能减少 GGML / CUDA 上 per-layer prefill 往返次数。
- **整批一次 ggml 计算图** —— `ForwardBatch` 当前为每个序列每层各建一张
  分页注意力计算图。把它们合并为一张 per-layer 计算图能在短序列多的批量中
  摊销编译 / 启动开销。
