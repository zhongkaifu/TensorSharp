# Gemma 3

[← 返回模型索引](README_zh-cn.md) | [English](gemma3.md)

| 属性 | 值 |
|---|---|
| 提供方 | Google |
| GGUF 架构标识 | `gemma3` |
| 模型类 | [`Gemma3Model`](../../TensorSharp.Models/Models/Gemma3/Gemma3Model.cs) |
| 视觉编码器 | [`Gemma3VisionEncoder`](../../TensorSharp.Models/Models/Gemma3/Gemma3VisionEncoder.cs) |
| 图像处理器 | [`Gemma3ImageProcessor`](../../TensorSharp.Models/Models/Gemma3/ImageProcessor.cs) |
| 示例模型 | gemma-3-4b、gemma-3-12b、gemma-3-27b |
| 模态 | 文本、图像 |
| 思维链模式 | 不支持 |
| 工具调用 | 不支持 |
| 批处理 / 分页前向 | 未实现。当连续批处理引擎激活时，Gemma 3 走 `BatchExecutor` 的按序列 KV 交换回退路径。详见 §11。 |
| 输出解析器 | `PassthroughOutputParser` |

## 1. 来源与目标

Gemma 3 是 Google 第三代开源 LLM，从更大的 Gemini 系蒸馏而来。架构上的关键决定是：

- **混合注意力模式** —— 每六层中有一层是全因果注意力层，其余五层使用滑动窗口注意力（SWA）。SWA 把长上下文下的 per-layer 注意力开销限定在窗口内。
- **per-head QK 归一化** —— Q 和 K 在 attention 点积前都按 head 做 RMSNorm，可以在 Gemma 3 较宽（默认 256）的 head 维度上稳定 attention 分数。
- **GeGLU FFN**，融合 `ffn_gate_up`，使用 GELU 激活。相比 SwiGLU 在表达上略强，代价是少量额外 FLOP。
- **embedding 缩放** —— 在进入第一层之前 token embedding 乘以 `sqrt(hidden_size)`，让 residual stream 的范数与 hidden size 解耦。
- **可选 logit softcap** —— `tanh(logits / cap) * cap`，避免极端 logits 过强。

Gemma 3 是图文模型：单独的 `Gemma3VisionEncoder`（从 mmproj GGUF 加载，通常是 `mmproj-gemma3-4b-f16.gguf`）输出固定长度的图像 embedding，并在 `<start_of_image>` 占位 token 处拼回 residual stream。

## 2. 模型架构

```
                              ┌──────────────────────────┐
                              │ token_embd.weight        │
        tokens (int[]) ──────►│  (× sqrt(hidden_size))   │
                              └────────────┬─────────────┘
                                           │
                              [可选] InjectVisionEmbeddings
                                           │
                                           ▼
                  ┌────────────── × NumLayers ──────────────┐
                  │  RMSNorm (attn_norm)                    │
                  │  Q, K, V 三个投影                        │
                  │  per-head RMSNorm (attn_q_norm/k_norm)  │
                  │  RoPE (NeoX，local 或 global base)       │
                  │  Q ← Q * (1/sqrt(headDim))               │
                  │  Attention（SWA 窗口或全因果）           │
                  │  attn_output                            │
                  │  RMSNorm (post_attention_norm) + 残差    │
                  │  RMSNorm (ffn_norm)                     │
                  │  GeGLU: GELU(gate) * up → down           │
                  │  RMSNorm (post_ffw_norm) + 残差          │
                  └─────────────────────────────────────────┘
                                           │
                              RMSNorm (output_norm)
                                           │
                              LM head (output.weight 或 tied)
                                           │
                              [可选] tanh-softcap
                                           │
                                           ▼
                                       logits
```

`IsGlobalLayer(layer)` 在 `(layer + 1) % 6 == 0` 时返回 true。全局层使用全因果 attention；其余五分之一为 SWA 层，仅 attend 后 `sliding_window` 个 token。

## 3. 前向计算图

decode 与 prefill 都走同一个 `TransformerBlock`。区别在于 prefill 把 `[seqLen, hidden]` 一并送进 GEMM 并做下三角因果掩码；decode 把 `[1, hidden]` 一行送进缓存的 attention 窗口。

每一层 L：

```
hidden ─► RMSNorm(attn_norm.weight, eps)
       ─► attn_q.weight ─► Q          (按 head reshape)
       ─► attn_k.weight ─► K          (按 head reshape)
       ─► attn_v.weight ─► V          (按 head reshape)
       ─► RMSNorm(attn_q_norm.weight) on Q (per head)
       ─► RMSNorm(attn_k_norm.weight) on K (per head)
       ─► RoPE_NeoX(Q, K, freqs[layer])     // local 或 global base
       ─► Q ← Q * (1/sqrt(headDim))
       ─► append K, V 到 KV cache（位置 startPos..startPos+seqLen-1）
       ─► attention(Q, KCache, VCache, window=W if SWA else totalSeq)
       ─► attn_output.weight matmul → o
       ─► RMSNorm(post_attention_norm.weight, eps) on o
       ─► hidden = hidden + o
       ─► h2 = RMSNorm(ffn_norm.weight, eps) on hidden
       ─► ffn_gate_up.weight matmul → [gate ‖ up]   (融合投影)
       ─► g = GELU(gate)
       ─► h3 = ffn_down.weight × (g * up)
       ─► RMSNorm(post_ffw_norm.weight, eps) on h3
       ─► hidden = hidden + h3
```

所有层结束后：

```
hidden ─► narrow(seq_len-1) if prefill          // LM head 只关心最后一行
       ─► RMSNorm(output_norm.weight, eps)
       ─► matmul against output.weight (或 tied 的 token_embd.weight)
       ─► [可选] tanh(logits/cap) * cap     // _finalLogitSoftcap > 0
       ─► 拷贝到 float[VocabSize] 供 sampler 使用
```

## 4. 组件细节

### 4.1 Attention

- **GQA**，独立的 `key_length` 与 `value_length`（默认 256/256）。
- **模式**：`IsGlobalLayer(layer)` 在 `(layer + 1) % 6 == 0` 时为 true。全局层全因果；其余五层只 attend 区间 `[totalSeqLen − slidingWindow, totalSeqLen)`。
- **QK 归一化**：使用 `attn_q_norm.weight` / `attn_k_norm.weight` 的 per-head RMSNorm。
- **RoPE**：NeoX 风格。Local 层使用 `_ropeFreqsLocal[]`（base 来自 `gemma3.rope.local.freq_base`，默认 10000）。Global 层使用 `_ropeFreqsGlobal[]`（base 来自 `gemma3.rope.freq_base` 除以 `_ropeScale`）。27B 变体（`NumLayers == 34`）硬编码 `_ropeScale = 8.0`。
- **decode SWA bound**：`AttentionDecodeWithWindow()` 只读取缓存中 `[max(0, totalSeqLen − slidingWindow), totalSeqLen)` 的位置。

### 4.2 FFN —— GeGLU

`ffn_gate_up.weight` 沿行方向拼 `[gate ‖ up]`，单次 matmul 同时算出两半。`Ops.GELUMul` 计算 `GELU(gate) * up`（不是 Qwen / Mistral 的 SiLU）；接着 `ffn_down.weight` matmul。

### 4.3 Normalization

每个 block 四个 RMSNorm —— `attn_norm`、`post_attention_norm`、`ffn_norm`、`post_ffw_norm` —— 加上 per-head 的 `attn_q_norm` / `attn_k_norm`。最终的 `output_norm` 跨层共享。Eps 来自 `Config.Eps`，通常 `1e-6`。

### 4.4 Embedding 与 LM head

`token_embd.weight` 也作为 LM head 复用 —— 当 `output.weight` 缺失时（`_hasTiedOutput`）。`ScaleEmbedding()` 把 embedding 乘以 `sqrt(Config.HiddenSize)`。

### 4.5 Logit softcap

当 `gemma3.final_logit_softcapping > 0`，LM head 输出走 `logits = tanh(logits / cap) * cap`；否则跳过。

### 4.6 视觉管线

`Gemma3VisionEncoder` 是 CLIP 风格的 ViT，固定输出 `TokensPerImage`（默认 256）个大小为 hidden 的 embedding。图像处理器：

1. RGBA over white 合成。
2. 缩放到 `image_size × image_size`（通常 896×896）。
3. 用编码器自带的 mean/std 做归一化。
4. 转为 NCHW float 张量。

`InjectVisionEmbeddings()` 把 `[256, hidden]` block 写到 `<start_of_image>` 占位的位置上；输入 prompt 在分词阶段已经被 `ChatTemplate.ExpandGemma3ImageTokens()` 展开，每张图片对应 1 个 start sentinel + 256 个 padding token + 1 个 end sentinel。

## 5. 参数与配置（GGUF 元数据）

| Key | 类型 | 默认 | 含义 |
|---|---|---|---|
| `gemma3.attention.sliding_window` | uint32 | 1024 | 非全局层的 SWA 窗口长度 |
| `gemma3.attention.key_length` | uint32 | 256 | per-head key 维 |
| `gemma3.attention.value_length` | uint32 | 256 | per-head value 维 |
| `gemma3.rope.local.freq_base` | float32 | 10000 | SWA 层 RoPE base |
| `gemma3.rope.freq_base` | float32 | 来自 `general` | 全局层 RoPE base |
| `gemma3.final_logit_softcapping` | float32 | 0（关闭） | LM head 的 tanh-softcap 阈值 |

加上 `ParseBaseConfig` 读取的标准 `general.*` keys。

## 6. 权重命名约定

```
token_embd.weight                          # [vocab, hidden]
blk.{L}.attn_norm.weight                   # 注意力前 RMSNorm
blk.{L}.attn_q.weight                      # Q 投影
blk.{L}.attn_k.weight                      # K 投影
blk.{L}.attn_v.weight                      # V 投影
blk.{L}.attn_q_norm.weight                 # per-head Q RMSNorm
blk.{L}.attn_k_norm.weight                 # per-head K RMSNorm
blk.{L}.attn_output.weight                 # 输出投影
blk.{L}.post_attention_norm.weight         # 注意力后 RMSNorm
blk.{L}.ffn_norm.weight                    # FFN 前 RMSNorm
blk.{L}.ffn_gate.weight }                  # 融合前
blk.{L}.ffn_up.weight   }
blk.{L}.ffn_gate_up.weight                 # 融合后：[2*intermediate, hidden]
blk.{L}.ffn_down.weight                    # FFN down
blk.{L}.post_ffw_norm.weight               # FFN 后 RMSNorm
output_norm.weight                         # 最终 RMSNorm
output.weight                              # LM head（若 tied 则缺失）
```

`Gemma3Model` 在加载时调用 `FuseGateUpWeights()` 把 `ffn_gate.weight` 与 `ffn_up.weight` 拼成 `ffn_gate_up.weight`；如果 GGUF 已经融合，调用是 no-op。

## 7. TensorSharp 实现走读

`Gemma3Model(string ggufPath, BackendType backend)` 构造函数：

1. `ParseBaseConfig()`：读取 `general.*` 并填入 `Config`。
2. 读取 Gemma 3 专属元数据，决定 head dim、SWA 窗口、双 RoPE base、softcap，并检测 27B 变体覆盖 `_ropeScale`。
3. `ParseTokenizer()`：从 GGUF 元数据构造 SentencePiece 分词器。
4. `LoadWeights()`：把 F32 norm / embedding 加载到托管 tensor，量化矩阵乘权重保存在 `_quantWeights` 字典里。
5. `_hasTiedOutput = !_weights.ContainsKey("output.weight") && !_quantWeights.ContainsKey("output.weight")`。
6. `FuseGateUpWeights()`。
7. `PrepareCudaQuantizedWeightsForInference()`：为 direct CUDA 后端上传/重排量化数据（其他后端 no-op）。
8. `PrecomputeRoPE()`：填 `_ropeFreqsLocal` / `_ropeFreqsGlobal`。
9. `InitKVCache(maxSeqLen)`：分配 `[NumKVHeads, maxSeqLen, headDim]` 的 KV cache（dtype 由 `--kv-cache-dtype` 控制：`f32` / `f16` / `q8_0`）。

`Forward(int[] tokens)` 逐层运行 `TransformerBlock`。Gemma 3 当前没有融合的整模型 GGML kernel —— 所有算子各自调度，是下方优化机会的目标之一。

## 8. Prefill 路径

Prefill 走多 token 的标准托管路径：

- **SWA mask 缓存** —— `_cachedSWAMaskWidths` 缓存当前 `(queryLen, startPos)` 的 per-row SWA mask 宽度。所有五个 SWA 层共享同一 mask，构造一次复用多次；`startPos` 改变（如多轮 KV 复用）会失效。
- **last-row narrow** —— 只有最后一行喂给 LM head，所以最后一层之后立即 `Narrow` 到 `[1, hidden]` 再走 `output_norm` 和 LM head。

## 9. Decode 路径

Gemma 3 没有融合的整模型 kernel（不像 Gemma 4），所以每个算子单独调度。Decode 热点路径依赖：

- 在初始化时预解析的 per-layer 权重名字符串（hot loop 不再做字符串插值）。
- KV cache append 走 `Narrow + Copy` 而非整块覆写。
- `AttentionDecodeWithWindow()` 把 SWA 层的 K^T Q 与 softmax 限定在后 `slidingWindow` 个位置。

GGML CUDA / Metal 后端时每个 matmul、RMSNorm、RoPE、softmax、attention 都走原生 kernel；direct CUDA 走 cuBLAS GEMM 与 PTX kernel；纯 `cpu` 后端走 `TensorSharp.Core` 的 SIMD 优化托管 kernel。

## 10. 内存与 KV cache 策略

- 每层 K、V 张量按 `maxSeqLen` 全量分配，**SWA 层与全局层使用相同容量**（SWA bound 是 decode 时的窗口操作，不是更小的分配）—— 这条在下方优化机会里。
- `ResetKVCache()` 清零所有 cache 张量，并调用 `InvalidateTensorDeviceCache()` 让 GGML / CUDA 端看到清零状态。
- `TruncateKVCache(int tokenCount)` 保留前 `tokenCount` 个位置（服务器多轮 KV 复用使用此 API）。

量化权重通过 `LoadWeights()` 后保留在 `_quantWeights`；matmul 直接走后端的原生量化 matmul，不解压到 FP32。

## 11. 批处理 / 分页前向（连续批处理）

Gemma 3 **未实现** `IBatchedPagedModel.ForwardBatch`。当连续批处理引擎激
活时（`TensorSharp.Server` 默认开启），Gemma 3 的序列走
`BatchExecutor.ExecuteStepPerSequence` —— 即按序列 KV 交换回退路径，依靠
模型的 `TryExtractKVBlock` / `TryInjectKVBlock` 协议在调度器把不同序列交
给模型时搬入搬出 KV 状态。调度器、分页块池、前缀缓存索引、请求流式输出
都仍然生效；与真正的批处理移植（Mistral 3 / Gemma 4 / Qwen 3 等）的区别
只是 Gemma 3 在每个调度步骤内一次跑一个序列，而不是把 N 个序列打包成一
次前向。

把 Gemma 3 移植到 `IBatchedPagedModel` 主要是 Gemma 4 批处理移植的镜像，
但需求更简单（无 PLE、无 KV donor 共享、无逐层 head dim 异构），同时仍需
应对相同的 SWA / 全局注意力交替派发以及环形缓存考量。这是一个可行的后续
工作，已列入 §13 优化机会。

## 12. 输出解析器与聊天模板

- `Gemma3OutputParser` 等同于 `PassthroughOutputParser` —— Gemma 3 没有思维链 / 工具调用 wire 格式。
- 聊天模板在 GGUF 没带 Jinja2 模板时回退到内置的 Gemma chat 模板。

## 13. 优化机会

- **批处理 / 分页前向移植** —— 实现 `IBatchedPagedModel.ForwardBatch` 能
  让 Gemma 3 使用与 Mistral 3 / Gemma 4 / Qwen 3 相同的连续批处理引擎路径，
  并复用原生分页注意力内核已经支持的逐层 SWA 派发。
- **融合 QKV** —— Q、K、V 仍是三个独立投影。仿照 Gemma 4 / Qwen 3 拼成单次 matmul 可以把注意力调度数减半。
- **融合整模型 decode** —— 引入类似 `Gemma4ModelDecode` 的 `Gemma3ModelDecode` 可以消除 Metal / CUDA 上的 per-op CPU/GPU 往返。
- **环形 SWA cache** —— SWA 层可以只分配 `slidingWindow` 个槽位（参考 Gemma 4 的 `CopyToCacheCircular()`）。
- **分块 prefill** —— 长 prompt 当前会为全局层物化整张 `[seqLen × seqLen]` mask；分块（参考 Gemma 4）能把 per-step 内存占用限定住。
