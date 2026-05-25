# Qwen 3

[← 返回模型索引](README_zh-cn.md) | [English](qwen3.md)

| 属性 | 值 |
|---|---|
| 提供方 | 阿里巴巴 |
| GGUF 架构标识 | `qwen3` |
| 模型类 | [`Qwen3Model`](../../TensorSharp.Models/Models/Qwen3/Qwen3Model.cs)（旧单序列路径）+ [`Qwen3Model.BatchedForward.cs`](../../TensorSharp.Models/Models/Qwen3/Qwen3Model.BatchedForward.cs)（`IBatchedPagedModel`） |
| 示例模型 | Qwen3-4B、Qwen3-8B、Qwen3-14B、Qwen3-32B |
| 模态 | 仅文本 |
| 思维链模式 | 是（`<think> ... </think>`） |
| 工具调用 | 是（`<tool_call>{...}</tool_call>`） |
| 批处理 / 分页前向 | 参考移植 —— TensorSharp 中第一个实现 `IBatchedPagedModel.ForwardBatch` 的模型。Mistral 3 / Gemma 4 / Qwen 3.5 / Nemotron-H 的批处理移植都以它为模板。详见 §11。 |
| 输出解析器 | `Qwen3OutputParser` |

## 1. 来源与目标

Qwen 3 是阿里 Qwen 系中纯密集型基线 transformer，也是 TensorSharp 中最简洁的「全功能」架构。它有两层意义：

1. **参考实现** —— 干净的「GQA + RoPE + RMSNorm」transformer，基本只用 `ModelBase` 的通用机制，没有奇异的 per-layer 装置。
2. **吞吐基准** —— Qwen 3 拥有 TensorSharp 当前最快的「整模型」decode 路径：单次原生调用，权重指针在加载时就预解析好，逐层用 C++ 跑完。

支持思维链（`<think>...</think>`）和工具调用（`<tool_call>` JSON 块），由 `Qwen3OutputParser` 处理。

## 2. 模型架构

```
                    tokens (int[])
                          │
                    token_embd.weight
                          │
        ┌──── × NumLayers ──────────────────────────────┐
        │  RMSNorm(attn_norm)                            │
        │  attn_qkv (融合) ──► Q, K, V (GQA)             │
        │  per-head RMSNorm(Q) using attn_q_norm.weight  │
        │  per-head RMSNorm(K) using attn_k_norm.weight  │
        │  RoPE_NeoX(Q, K)                                │
        │  Q ← Q * (1/sqrt(headDim))                      │
        │  attention(Q, KCache, VCache)                   │
        │  attn_output ──► 残差                          │
        │  RMSNorm(ffn_norm)                              │
        │  ffn_gate_up (融合) ──► gate ‖ up              │
        │  SwiGLU: SiLU(gate) * up                        │
        │  ffn_down ──► 残差                              │
        └─────────────────────────────────────────────────┘
                          │
                    RMSNorm(output_norm)
                          │
                    LM head (output.weight)
                          │
                          ▼
                       logits
```

每个 block 只有两次 RMSNorm（`attn_norm`、`ffn_norm`），无后置 norm，per-layer compute 非常精简。

## 3. 前向计算图

每一层 L：

```
hidden ─► RMSNorm(attn_norm.weight, eps)
       ─► attn_qkv.weight matmul → 拆 Q [seq, qDim], K [seq, kvDim], V [seq, kvDim]
       ─► per-head RMSNorm(Q, attn_q_norm.weight)
       ─► per-head RMSNorm(K, attn_k_norm.weight)
       ─► RoPE_NeoX(Q, K, ropeFreqs[]) at positions startPos..startPos+seqLen-1
       ─► Q ← Q * (1/sqrt(headDim))
       ─► append (K, V) 到 KV cache
       ─► attention(Q, KCache, VCache)        // 全 GQA，无窗口
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
       ─► matmul against output.weight       # Qwen 3 不 tied
       ─► 拷贝到 float[VocabSize]
```

## 4. 组件细节

### 4.1 Attention

- **GQA**，`Config.NumKVHeads < Config.NumHeads`。融合 `attn_qkv.weight` 行布局 `[qDim ‖ kvDim ‖ kvDim]`，其中 `qDim = NumHeads * headDim`，`kvDim = NumKVHeads * headDim`。
- **QK-norm**：per-head RMSNorm，权重为 `attn_q_norm.weight`（大小 `headDim`）和 `attn_k_norm.weight`（大小 `headDim`），分别独立作用于每个 Q/K head。
- **RoPE**：NeoX 风格。频率在 `_ropeFreqs[halfDim]` 中预计算，使用 `Config.RopeBase` 与 `Config.RopeScale`。
  - **decode** 用内联 C# 循环，cos/sin 表 stackalloc。
  - **prefill** 调 `Ops.RoPEEx`（CPU 上走 SIMD 托管 kernel，GGML / CUDA 上走原生 kernel）。
- **没有 attention sinks、没有 SWA、没有 Yarn / partial RoPE。** 标准因果 attention。

### 4.2 FFN —— SwiGLU

`ffn_gate_up.weight` 沿行拼 `[gate ‖ up]`。matmul 之后 `Ops.SiLUMul` 一次完成 `SiLU(gate) * up`，再走 `ffn_down`。无 bias。

### 4.3 Normalization

每个 block 两次 RMSNorm（`attn_norm`、`ffn_norm`），加上 per-head QK norm 与最终 `output_norm`。Eps 来自 `Config.Eps`，通常 `1e-6`。

### 4.4 Embedding 与 LM head

`token_embd.weight` 是输入 embedding；`output.weight` 是 LM head，**不 tied**（Qwen 3 同时携带两者）。

## 5. 参数与配置

Qwen 3 仅使用标准 GGUF 元数据，没有架构专属 key（除命名空间前缀外）：

| Key | 类型 | 含义 |
|---|---|---|
| `qwen3.attention.head_count_kv` | uint32 | GQA 的 KV head 数 |
| `qwen3.rope.freq_base` | float32 | RoPE base（Qwen 3 长上下文版本约 1e6） |
| `qwen3.rope.scale` | float32 | RoPE 频率缩放分母 |

加上 `ParseBaseConfig()` 处理的标准 `general.*` keys。

## 6. 权重命名约定

```
token_embd.weight                          # [vocab, hidden]
blk.{L}.attn_norm.weight                   # 注意力前 RMSNorm
blk.{L}.attn_qkv.weight                    # 融合 QKV [qDim+kvDim+kvDim, hidden]
blk.{L}.attn_q_norm.weight                 # per-head Q RMSNorm
blk.{L}.attn_k_norm.weight                 # per-head K RMSNorm
blk.{L}.attn_output.weight                 # 输出投影
blk.{L}.ffn_norm.weight                    # FFN 前 RMSNorm
blk.{L}.ffn_gate_up.weight                 # 融合 gate+up [2*intermediate, hidden]
blk.{L}.ffn_down.weight                    # FFN down
output_norm.weight                         # 最终 RMSNorm
output.weight                              # LM head
```

`FuseQKVWeights()` 与 `FuseGateUpWeights()` 在加载时调用，即使 GGUF 分散存放 Q/K/V 或 gate/up 也会折叠为上述融合形式。

## 7. TensorSharp 实现走读

`Qwen3Model(string ggufPath, BackendType backend)` 构造函数：

1. `ParseBaseConfig()` 读 `general.*` 与 `qwen3.*` 核心字段。
2. `Config.NumKVHeads = (int)_gguf.GetUint32("qwen3.attention.head_count_kv")`。
3. `ParseTokenizer()` 构造 BPE 分词器。
4. `LoadWeights()`。
5. `FuseQKVWeights()`、`FuseGateUpWeights()`。
6. `PrepareCudaQuantizedWeightsForInference()`。
7. `InitKVCache(maxSeqLen)` 分配 `[NumKVHeads, maxSeqLen, headDim]` 的 K、V tensor。
8. `PrecomputeConstants()` 填 `_ropeFreqs[]` 并预分配 per-head decode 位置数组（`_decodeQPositions`、`_decodeKPositions`）。
9. `BuildModelDecodeArrays()` 为 **whole-model** 原生 decode 调用打包 per-layer 指针、类型、维度。
10. `DetermineNativeLayerDecodeAvailability()` 根据后端支持情况在 per-layer 与 whole-model 原生路径之间选择。

`Forward(int[] tokens)` 之后会在三条执行路径间挑一条：

- **`useNativeModelDecode`**（GGML 后端 + `seqLen == 1` + 模型 decode arrays 可用，默认走这条）：单次原生调用处理所有层。
- **`useNativeDecode`**（per-layer 原生回退）：每层走一次 `NativeTransformerLayerDecode`。
- **托管 C# 循环**：完整逐算子托管路径；用于 prefill（`seqLen > 1`）、纯 CPU 后端、direct CUDA 后端。

decode 循环之后 LM head 跑（prefill 时先 `Narrow(seq_len-1)` 让 matmul 只算 `[1, vocab]`），然后 logits 拷贝到 `_logitsBuffer`。

## 8. Prefill 优化

Qwen 3 的 prefill 走标准托管循环 + 后端分发的算子：

- 融合 QKV 与融合 gate / up 把 per-layer GEMM 数从 5 降到 3（`qkv`、`out`、`gate_up`、`down`）。
- `Ops.RoPEEx` 在 GGML / CUDA 后端是单次原生 kernel。
- `Ops.SiLUMul` 把 gate 激活与逐元素乘融合。
- `output_norm` 之前已经 last-row narrow，LM head matmul 只产出一行 logits。

没有分块 prefill 也没有融合 per-layer prefill kernel —— 模型够小，逐算子托管路径在 GGML / CUDA 上的 prefill 吞吐已经够好。增加融合 per-layer prefill kernel 是显然的下一步（见 § 12）。

## 9. Decode 优化

### Whole-model 原生 decode（`TransformerModelDecode`）

这是当前 TensorSharp 中最快的 decode 路径。`BuildModelDecodeArrays()` 在加载时把以下东西打包到 `_modelDecodeArrays`：

- 所有 per-layer 权重指针的扁平 `IntPtr[]`（norms、QKV、Q/K norms、output、FFN norm、gate+up、down）。
- 每层的量化权重元数据（`type`、`ne0`、`ne1`、原始字节数）。
- KV cache 指针，原生侧直接写入。
- RoPE 频率表。

`NativeTransformerModelDecode()` 就是单次 P/Invoke 在 C++ 里跑完整一次 token 的 forward：

- 层与层之间没有托管循环开销。
- 没有权重名字典查询。
- 层间不分配 tensor，原生侧复用 `_modelDecodeArrays` 中保存的 scratch 缓冲。

主机侧 KV cache 拷贝标记为 dirty（`_kvCacheHostDirty = true`）；下次 prefill 或 KV cache 截断会调用 `EnsureKvCacheHostSynchronized()` 把设备数据同步回来，让托管路径看到正确状态。

### Per-layer 原生 decode（`TransformerLayerDecode`）

当 whole-model arrays 不可用时回退（少见；某些量化类型尚未被原生循环支持时会发生）。每层走一次 P/Invoke，仍然比逐算子调用便宜。

### 手写 C# decode

两条原生路径都不可用时（纯 CPU 后端、direct CUDA 后端但缺少匹配的原生构建），托管 decode 循环：

- 在 init 时把 per-layer 权重名字符串缓存到 `_layerWeightNames[]`。
- 内联 RoPE 用 `stackalloc` cos/sin 表。
- 使用预分配的 `_decodeQPositions[]` / `_decodeKPositions[]`。

## 10. 内存与 KV cache 策略

- 每层 K、V tensor `[NumKVHeads, maxSeqLen, headDim]`。KV dtype 通过 `--kv-cache-dtype` 选 `f32` / `f16` / `q8_0`。
- `ResetKVCache()` 清零并清 dirty 标志。
- `TruncateKVCache(int tokenCount)` 支持（先调 `EnsureKvCacheHostSynchronized()`），服务器多轮 KV cache 复用使用此 API。
- 量化权重保留在 `_quantWeights`；matmul 走后端原生 quant matmul。

## 11. 批处理 / 分页前向（连续批处理）

`Qwen3Model.BatchedForward.cs` 是 TensorSharp 中 `IBatchedPagedModel.ForwardBatch`
的**参考实现** —— 最简单的非混合密集 transformer 移植，Mistral 3、Gemma 4、
Qwen 3.5/3.6、Nemotron-H 后续都在它的基础上扩展。它通过
[`docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md`](../PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md)
中描述的共享 `InferenceEngine` 连续批处理栈执行。

实现要点：

- **按需扩容的每层分页 K/V 缓冲**，布局
  `[numBlocks * blockSize * numKvHeads * headDim]`，由引擎的 `BlockPool`
  决定大小。`AllocateLayerBuffer(numBlocks, blockSize, numKvHeads, headDim)`
  是后续所有批处理移植都复用的辅助函数。
- **逐 token RoPE** 用 `ctx.Positions` 构建，通过 `Ops.RoPEEx` 派发（该算子
  已经接受任意 positions tensor —— 这就是 Qwen 3 批处理移植引入、其他批
  处理模型复用的内核改动）。
- **基于 `slotMapping` 的 K/V 写入** 通过
  `PagedKvBatchOps.ScatterKv(slotMapping)` 把每层的 K 与 V 写到分页缓冲中
  `blockId * blockSize + offset` 位置。
- **按序列的注意力** 通过原生分页内核 `TSGgml_PagedAttentionForward`
  （默认）或 C# 回退 `ManagedPagedAttention.Forward`。K/V 按
  `blockTables` + `seqLens` 按序列 gather。
- **末 token gather** 通过 `PagedKvBatchOps.GatherLastTokenPerSeq`，让 LM
  head 只在 `numSeqs` 行上跑，而不是 `numTokens` 行。
- **复用已有辅助** —— `ModelBase` 上的 `Embedding`、`RMSNormOp`、
  `LinearForward`、`FFN` 已经按完整 token 轴工作，因此嵌入 / 投影 / 归一化
  / FFN 都能自动批处理。

### 已验证的正确性

针对真实 base-Qwen3 GGUF 的 logit 级正确性尚未在此 tree 中验证（本地只有
Qwen 3.6 / GatedDeltaNet GGUF）。opt-in 的
[`Qwen3BatchedForwardTests`](../../InferenceWeb.Tests/Qwen3BatchedForwardTests.cs)
在把 base Qwen3 GGUF 放进 `TS_TEST_MODEL_DIR` 后会自我验证。
`EngineParallelInferenceTests` 在已有的 Qwen GGUF 上验证了引擎侧路径。

## 12. 输出解析器与聊天模板

- `Qwen3OutputParser` 抽取：
  - `<think> ... </think>` 中的思维链内容。
  - `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` 中的工具调用。
- 聊天模板使用标准 Qwen `<|im_start|>` / `<|im_end|>` 格式。GGUF 没带 Jinja2 模板时回退到内置；带了就直接渲染嵌入的 Jinja2。

## 13. 优化机会

- **GPU 融合 per-token decode** —— whole-model 原生路径已经消除托管开销，但层与层之间仍是各自的 GGML 图。把它们融合到一张图里（参考 `Gemma4ModelDecode`）能进一步压低 Metal / CUDA 上的剩余往返开销。
- **原生 prefill** —— 把 prefill 也搬到原生单次调用路径会显著降低长 prompt 的 TTFT，因为长上下文摊薄不掉逐算子托管开销。
- **在真实 base-Qwen3 GGUF 上验证批处理正确性** —— 参考测试已存在，但需
  要在测试 fixture 中放一个模型文件（或 CI 助手脚本下载），才能自动执行
  cosine 校验。
