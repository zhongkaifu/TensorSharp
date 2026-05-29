# GPT OSS

[← 返回模型索引](README_zh-cn.md) | [English](gptoss.md)

| 属性 | 值 |
|---|---|
| 提供方 | OpenAI |
| GGUF 架构标识 | `gptoss`、`gpt-oss` |
| 模型类 | [`GptOssModel`](../../TensorSharp.Models/Models/GptOss/GptOssModel.cs)（旧单序列路径）+ [`GptOssModel.BatchedForward.cs`](../../TensorSharp.Models/Models/GptOss/GptOssModel.BatchedForward.cs)（`IBatchedPagedModel`） |
| 示例模型 | gpt-oss-20b |
| 模态 | 仅文本 |
| 思维链模式 | 是（Harmony 格式：`<\|channel>analysis ... <\|channel>final`） |
| 工具调用 | 否 |
| 批处理 / 分页前向 | **默认启用** —— 设置 `TS_GPTOSS_BATCHED=0` 可强制走旧的按序列 KV-swap 路径用于 A/B 对比。每层分页 K/V，注意力 sinks 通过原生 `TSGgml_PagedAttentionForwardWithSinks`（或托管 C# 回退 `TS_GPTOSS_PAGED_ATTN_MANAGED=1`）。详见 §11。 |
| 输出解析器 | `HarmonyOutputParser`（始终启用） |

## 1. 来源与目标

GPT OSS 是 OpenAI 的开放权重 MoE 系列。它有几个让它在 TensorSharp 中显得与众不同的设计：

- **每一层都是 MoE**，TopK 路由后只对被选 expert 做 softmax（与 Gemma 4 的「先 softmax 再 TopK」相反）。
- **Attention sinks** —— 一种 per-head 学习到的偏置，作为「虚 token」（K=0、V=0）加入 softmax，提供一个永远存在的 attention 目标。
- **每个线性投影都有 bias**（Q、K、V、attn output、gate、up、down、router）。这在 TensorSharp 支持的架构中独此一家。
- **Clamped GLU 激活** —— FFN 用类 SiLU 的激活 `x · sigmoid(α · x) · (y + 1)`，gate 被 clamp 到 `[-∞, 7]`，up 被 clamp 到 `[-7, 7]`。
- **Harmony 输出格式** —— 每条输出都被包在 `<|channel>analysis ... <|channel>final ...` 标记中。输出解析器标记为 `AlwaysRequired = true`，因为没有「关闭思维链」的模式。
- **MXFP4 expert 权重**（GGML 量化类型 39）—— gpt-oss-20b 使用的 4 位 microscaling 格式。

## 2. 模型架构

```
                tokens (int[])
                      │
              token_embd.weight
                      │
        ┌──── × NumLayers ───────────────────────────────┐
        │  RMSNorm(attn_norm)                             │
        │  Q+bias, K+bias, V+bias（无 QK-norm）            │
        │  RoPE_NeoX with YaRN scaling                     │
        │  attention with sinks                            │
        │  attn_output + bias ─► residual                  │
        │                                                  │
        │  RMSNorm(post_attention_norm)                    │
        │  router_linear+bias                              │
        │  topK 后 softmax(选中 experts)                   │
        │  per-expert: SiLUAlphaLimit(gate+bias, up+bias)  │
        │              ─► down+bias                        │
        │  expert 输出按 router 权重加权求和                │
        │  ─► residual                                     │
        └──────────────────────────────────────────────────┘
                      │
              RMSNorm(output_norm)
                      │
              LM head (output.weight)
                      │
                      ▼
                   logits
```

## 3. 前向计算图

每一层 L：

```
hidden ─► RMSNorm(attn_norm.weight, eps)
       ─► Q+bias, K+bias, V+bias（每个线性都是 matmul + per-row bias add）
       ─► RoPE_NeoX with YaRN scaling
            • origCtxLen = Config.OriginalContextLength (4096)
            • freqScale  = 1 / RopeScale
            • beta_fast  = 32, beta_slow = 1
       ─► append (K, V) 到 KV cache
       ─► attention with sinks:
            scores = Q × K^T
            scores ← 应用因果 mask（偶数层希望时附加 SWA）
            sinks  = attn_sinks.weight                   # [numHeads]
            sums   = exp(scores - max) + exp(sinks - max)
            scores = exp(scores - max) / sums            # softmax 包含 sink
            out    = scores × V
       ─► attn_output.weight matmul + bias → o
       ─► residual = hidden + o

       moeInput = (prefill 中最后一层) ? narrow(seq_len-1) : residual
       moeInput ─► RMSNorm(post_attention_norm)
                ─► router_linear + bias
                ─► topK(_numExpertsUsed)
                ─► softmax(选中权重)
                ─► 对每个被选 expert e:
                       gate = ffn_gate_up_exps[e][:, :nFf] × moeInput + bias
                       up   = ffn_gate_up_exps[e][:, nFf:] × moeInput + bias
                       gate = clamp(gate, -∞, 7)
                       up   = clamp(up,   -7, 7)
                       out_e = gate · sigmoid(α · gate) · (up + 1)
                       out_e = ffn_down_exps[e] × out_e + bias
                ─► 用 router 权重 weighted Σ out_e
                ─► residual += weighted_sum
```

所有层完成后：

```
hidden ─► narrow(seq_len-1) if prefill          # GPT OSS 在最后一层 MoE 之前就 narrow，
                                                 # 见 § 8
       ─► RMSNorm(output_norm.weight, eps)
       ─► output.weight matmul → logits
       ─► 拷贝到 float[VocabSize]
```

## 4. 组件细节

### 4.1 Attention

- **GQA**，`Config.NumKVHeads < Config.NumHeads`。
- **可选融合 QKV**（`_isQkvFused`）—— GGUF 直接给单个 `attn_qkv.weight` 时 `FuseQKVWeights()` 构建融合 tensor，否则三次线性 op 分别派发。
- **每个投影都有 bias**：在 `LinearForwardWithBias()` 实现，做完 matmul 后逐行加 bias，bias 从 `_weights[biasName]` 取。
- **没有 QK-norm**：与 Gemma / Qwen 不同，GPT OSS 跳过 per-head Q/K 归一。
- **Attention sinks**：per-head bias `attn_sinks.weight`（形状 `[numHeads]`）。在 softmax 中作为虚 token：参与 max 计算与 exp 求和，相当于多了一个永远在的 attention 目标。在 `ApplySoftmaxWithSinks()` 与 `AttentionDecodeWithSinks()` 实现。
- **RoPE**：NeoX 风格 + YaRN scaling。`Ops.RoPEEx` 调用时传入 `origCtxLen = Config.OriginalContextLength`（4096）、`freqScale = 1 / RopeScale`、`beta_fast = 32`、`beta_slow = 1`。
- **Attention pattern**：偶数层 ⇒ SWA（窗口取自 `_slidingWindow`，默认 128）；奇数层 ⇒ 全因果。*（实现注：当前代码路径下所有层都对 `totalSeqLen` 做 attention —— SWA 边界从 GGUF 元数据读到了，但 softmax 宽度尚未受其约束。这条在「优化机会」清单里。）*

### 4.2 FFN —— Clamped GLU（`SiLUAlphaLimit`）

```
gate = clamp(gate, -∞, SiluLimit)        # SiluLimit = 7.0
up   = clamp(up,   -SiluLimit, SiluLimit)
out  = gate · sigmoid(SiluAlpha · gate) · (up + 1)   # SiluAlpha = 1.702
```

`SiluAlpha` 与 `SiluLimit` 是硬编码常量。激活在 `SiLUAlphaLimitInPlace` 中以 SIMD 向量化实现，并以 `swiglu_oai` 的形式暴露给融合 MoE prefill kernel。

### 4.3 MoE 路由

- 每层都有 `_numExperts` 个 expert（gpt-oss-20b 是 32）和 TopK 路由（`_numExpertsUsed = 4`）。
- 路由：`linear(hidden) + bias → TopK → softmax(只对选中)`。这是**先 TopK 再 softmax**，与 Gemma 4 的「先 softmax 再 TopK」相反。
- Expert 权重以融合 `gate ‖ up` 行的形式存于 `ffn_gate_up_exps.{E}.weight`，对应融合 bias 在 `ffn_gate_up_exps.{E}.bias`。`FuseExpertGateUpWeights()` 在加载时完成融合。
- Expert bias 在 GGUF 中以打包 `[numExperts, biasDim]` 的形式存（`ffn_gate_exps.bias`）。`SplitExpertBiases()` 在融合前把它们拆成 per-expert `[biasDim]` tensor。

### 4.4 Stacked MoE prefill kernel（`TryMoEPrefillFused`）

GGML 后端上 prefill 时，GPT OSS 每层只派发一次 `ggml_mul_mat_id` + `ggml_add_id` + `swiglu_oai` 图，而不是按 token 枚举活跃 expert。kernel 读取：

- 原始 3D 的 `ffn_gate_exps.weight`、`ffn_up_exps.weight`、`ffn_down_exps.weight` 块（对 mmap 模型的零成本 view）。
- 一份连续 `[2 * nFf, numExperts]` f32 的 stacked gate / up bias（在 `InitMoeStackedWeights` 中由融合前捕获的 per-expert bias 一次性构建）。
- 一份连续 `[hidden, numExperts]` f32 的 stacked down bias（同样一次性构建）。

per-token 路由（TopK + softmax）在 C# 侧完成，得到的 `(token, expert)` 映射喂给 kernel。

### 4.5 分词器

GPT-4o BPE pre-tokenizer，带 `\p{N}{1,3}` 数字分组。运行时分词器在 `tokenizer.ggml.pre == "gpt-4o"` 时启用相应分支。

## 5. 参数与配置

| Key | 类型 | 含义 |
|---|---|---|
| `gptoss.expert_count` | uint32 | Expert 数（32） |
| `gptoss.expert_used_count` | uint32 | 每 token 选用 expert 数（4） |
| `gptoss.attention.sliding_window` | uint32 | SWA 窗口大小（128） |
| `gptoss.expert_feed_forward_length` | uint32 | Expert FFN 维度 |
| `gptoss.rope.scaling.original_context_length` | uint32 | YaRN 原始上下文长度（4096） |
| `tokenizer.ggml.pre` | string | Pre-tokenizer 类型（`gpt-4o`） |

## 6. 权重命名约定

```
token_embd.weight
output_norm.weight
output.weight

blk.{L}.attn_norm.weight
blk.{L}.attn_q.weight    / attn_q.bias
blk.{L}.attn_k.weight    / attn_k.bias
blk.{L}.attn_v.weight    / attn_v.bias
blk.{L}.attn_qkv.weight  / attn_qkv.bias       # 融合后（QKV 融合时）
blk.{L}.attn_output.weight / attn_output.bias
blk.{L}.attn_sinks.weight                      # per-head bias [numHeads]
blk.{L}.post_attention_norm.weight

blk.{L}.ffn_gate_inp.weight / ffn_gate_inp.bias   # router [numExperts, hidden]
blk.{L}.ffn_gate_up_exps.{E}.weight                # 融合 expert gate+up
blk.{L}.ffn_gate_up_exps.{E}.bias                  # 融合 expert gate+up bias
blk.{L}.ffn_down_exps.{E}.weight                   # expert down
blk.{L}.ffn_down_exps.{E}.bias                     # expert down bias
```

## 7. TensorSharp 实现走读

构造函数（`GptOssModel(string ggufPath, BackendType backend)`）：

1. `ParseBaseConfig()`。
2. 读 MoE 计数、SWA 窗口、expert FFN 长度、YaRN 原始上下文长度。
3. `ParseTokenizer()`。
4. `LoadWeights()`。
5. `SplitExpertBiases()` —— 拆开 GGUF 中打包的 expert bias。
6. **快照** 每个 expert 的 gate / up bias（在融合**之前**）。融合 MoE prefill kernel 需要原始 split 形状下的 bias 来构建连续 stacked bias 表。
7. `FuseExpertGateUpWeights()` —— 把每个 expert 的 gate 与 up 拼到 `ffn_gate_up_exps.{E}.weight`，并构建融合 bias。
8. `FuseQKVWeights()`。
9. `PrepareCudaQuantizedWeightsForInference()`。
10. `InitKVCache(maxSeqLen)`。
11. `PrecomputeConstants()` —— 预分配 per-layer 与 per-expert 的权重名字符串数组，热循环中没有字符串拼接。
12. `InitMoeStackedWeights(preFuseGateBias, preFuseUpBias)` —— 构建融合 MoE prefill kernel 用的 stacked bias，同时把 per-layer `StackedExpertWeights` 解析为对原始 3D `_exps.weight` 块的 view。

`Forward(int[] tokens)` 跑 per-op 托管循环：

- Embedding lookup。
- 每层：`LinearForwardWithBias` 做 Q / K / V，RoPE，`AttentionDecodeWithSinks`（prefill 走 `ApplySoftmaxWithSinks`），输出投影 + bias，MoE block（或融合 MoE prefill kernel）。
- 在 prefill 的**最后一层**，residual 在 MoE 之前 narrow 到最后一个 token —— 早退优化，跳过那些不喂给 LM head 的位置上的 MoE 计算。
- 最终 RMSNorm、LM head、拷贝到 `_logitsBuffer`。

## 8. Prefill 优化

- **最后一层 MoE 之前的 narrow**。最后一个 `TransformerBlock` 内 `moeInput` 被 narrow 到 `[1, hidden]`，让 MoE block 只算关心的那一行。所有更早层的 MoE 仍然在整个序列上算。
- **Stacked MoE prefill kernel**（`TryMoEPrefillFused`）。每层一次 `mul_mat_id + add_id + swiglu_oai` 图调度替代每 token 每 expert 多次派发，见 § 4.4。
- **预构建 per-layer 名字数组**（`_layerNames[L][]`）与 per-expert 名字数组（`_expertNames[L][E][]`）消除热循环里的字符串拼接。
- **缓存 attention sinks** 在 `_layerSinks[L][numHeads]`，softmax 不需要每步重新取。

## 9. Decode 优化

- **每步路由 buffer**（`_moeExpertCounts`、`_moeExpertOffsets`、`_moeTokenMap`、`_moeWeightMap`）跨 token 复用。
- **SIMD 向量化的 bias 加法与 SiLUAlphaLimit 激活** 在 `LinearForwardWithBias` 与 `SiLUAlphaLimitInPlace`。
- **Sinks softmax** 跑在 CPU（标量 + 可选 SIMD 求 exp-sum）。GPU 融合 sinks kernel 在「优化机会」清单上 —— 当前 sinks softmax 是 Metal / CUDA 上的慢路径。
- **MXFP4 expert 权重** 量化保留在 `_quantWeights`，matmul 由后端的量化 matmul 派发。

GPT OSS 当前**没有 whole-model 原生 decode 路径**（不像 Qwen 3），所以 decode 仍然 per-op 派发。这是最大的待优化项。

## 10. 内存与 KV cache 策略

- 每层 K、V tensor 形状 `[NumKVHeads, maxSeqLen, headDim]`。KV dtype 可配置 `f32` / `f16` / `q8_0`。
- `ResetKVCache()` 全部清零。
- Expert FFN 权重在 `ModelBase` 加载的原始 3D `ffn_gate_exps.weight` / `ffn_up_exps.weight` / `ffn_down_exps.weight` 块中。`FuseExpertGateUpWeights()` 只 dispose per-expert *view*，不动底层大缓冲。这样融合 MoE prefill kernel 仍能通过 `_layerStackedGate` / `_layerStackedUp` / `_layerStackedDown` 直接寻址原始块。

## 11. 批处理 / 分页前向（连续批处理）

GPT OSS 实现了 `IBatchedPagedModel.ForwardBatch`
（[`GptOssModel.BatchedForward.cs`](../../TensorSharp.Models/Models/GptOss/GptOssModel.BatchedForward.cs)）
并默认启用 —— 只有走批处理路径才能让并发请求真正并行（按序列回退每步至多
推进一个序列）。设置 `TS_GPTOSS_BATCHED=0` 可强制回到旧的按序列 KV-swap
路径用于 A/B 对比。该批处理移植需要在分页调度栈内保留 GPT OSS 三大架构特征
—— **注意力 sinks**、**每个投影都有 bias**、**逐层交替 SWA**：

- **每层分页 K/V**，布局 `[numBlocks * blockSize * numKvHeads * headDim]`，
  按需扩容（copy-on-write）。
- **批处理带 bias 的 QKV**；NeoX + YaRN RoPE 通过显式 `positions[]` 数组
  派发。
- **逐层 SWA 窗口** —— 旧路径同样的 local / global 交替模式按层传给分页内
  核，调度器仍把模型当作一致的整体，每层各自尊重各自的注意力视野。
- **带每头 sinks 的原生分页注意力**：
  `TSGgml_PagedAttentionForwardWithSinks`
  （[`ggml_ops_paged_attention.cpp`](../../TensorSharp.GGML.Native/ggml_ops_paged_attention.cpp)）
  把 `ggml_flash_attn_ext` 与 `add_sinks` 变体合在一起，让 sink logits 参
  与 softmax 归一化但不贡献到 V —— 与旧的 CPU sinks softmax 数值行为完全
  一致。通过 `GgmlBasicOps.PagedAttentionForwardWithSinks` 暴露。
  - 托管 C# 回退
    [`ManagedPagedAttention.ForwardWithSinks`](../../TensorSharp.Runtime/Paged/ManagedPagedAttention.cs)
    在非 GGML 后端或 `TS_GPTOSS_PAGED_ATTN_MANAGED=1` 时被选中。两者贪心
    解码 byte 级一致。
- **MoE FFN** 通过已有的 `MoEForward(numTokens)` token-parallel 路径执行；
  目前没有 GPT-OSS 专用的批处理 MoE 内核。

### 已验证的正确性与吞吐

- 在 [`GptOssBatchedCorrectnessTests`](../../InferenceWeb.Tests/GptOssBatchedCorrectnessTests.cs)
  中与旧路径**100% 贪心一致**（12/12 tokens）—— 在托管 sinks 回退与原生
  sinks 内核两条路径上都保持。
- **吞吐基本持平**（在
  [`GptOssBatchedPerfBench`](../../InferenceWeb.Tests/GptOssBatchedPerfBench.cs)
  中 n=1, 3, 5 范围 ~0.93–1.05×）。原生 sinks 内核没有改变全局，因为旧的
  单序列路径已经把每层都装进一张融合内核（`TryFusedAttnLayerPrefill`：
  RMSNorm + 融合 QKV + RoPE + KV-append + 带 sinks 的 softmax + attn +
  output proj + residual 一张 cgraph），而批处理路径每层要派 ~5 张图
  （norm、QKV、RoPE、paged-attn、output、residual）。要进一步缩小差距需
  要给 GPT OSS 写一个 fused-per-layer 的批处理内核 —— 工作量较大。

## 12. 输出解析器与聊天模板

- **Harmony 格式** 不可关闭。解析器（`HarmonyOutputParser`）标记为 `AlwaysRequired = true`，因为模型总是把输出包进 channel 标签：
  - `<|channel>analysis ...` 思维链推理。
  - `<|channel>final ...` 用户可见的回答。
- 输出解析器剥掉 `<|channel>analysis ...` 块（或在 API 中作为 `<think>` 内容暴露），把 `<|channel>final` 部分作为 assistant 消息暴露。
- **不支持工具调用** —— Harmony 格式预留了 tool channel，但开放权重的 gpt-oss-20b 不能稳定触发；API 接受 `tools` 但不会暴露 tool call。
- 聊天模板使用 GPT-4o pre-tokenizer 与 GGUF 中加载的 Harmony-aware Jinja2 模板。

## 13. 优化机会

- **融合 per-layer 批处理内核** —— 旧路径的 `TryFusedAttnLayerPrefill` 每
  层只发一张 cgraph（norm + 融合 QKV + RoPE + KV-append + 带 sinks 的
  softmax + attn + output proj + residual）。批处理路径目前每层要发 ~5
  张图。把它们合并成一张批处理 cgraph 是 GPT OSS 批处理性能最大的剩余优
  化点。
- **任何发布版都用融合 QKV** —— 当 GGUF 拆分 Q / K / V 时，每次单独 matmul
  后还是要单独 bias add。`FusedQKVWithBias` 图能把 3 次派发合 1。
- **原生 whole-model decode（旧路径）** —— 类似 Qwen 3 的
  `TransformerModelDecode`，加上 attention sinks 与 bias 支持，能消除单序
  列路径上的大部分托管开销。
- **旧路径上的 GPU 融合 sinks softmax** —— 旧的 CPU sinks softmax 是批处理
  路径下原生 `*_WithSinks` 内核的单序列对照。给旧路径写一份自定义 Metal /
  CUDA 融合内核能缩小这段差距，让单序列路径在 single-sequence 工作负载下
  保持竞争力。
- **per-expert decode 批处理** —— 即使有 stacked MoE prefill kernel，decode 仍然每 token 顺序枚举 expert。引入批量 decode 路径（类比 Qwen 3.5 的 `MoEExpertsSwiGLUResidual`）能把 `numExpertsUsed` 次派发合并为 1 次。
- **偶数层 SWA 边界生效** —— `_slidingWindow` 已经从 GGUF 读入，但当前 attention 路径还没用它来限制 softmax 宽度。把它接入 `AttentionDecodeWithSinks` 能在长上下文上降低 per-token 计算量。
