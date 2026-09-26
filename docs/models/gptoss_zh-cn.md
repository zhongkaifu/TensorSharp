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
| 工具调用 | 是（Harmony `commentary` channel —— `to=functions.NAME`）；可使用 skills、代码工具以及服务端的[子智能体委派](../multi_agent.md) |
| 投机解码 | 否 —— GPT OSS 没有投机主干，因此 `--spec`（包括无权重的 n-gram 草稿器）只提供普通解码 |
| 批处理 / 分页前向 | **默认启用。** 在不带 `--tp` 的 GGML 后端上，并发请求使用按请求的 KV holder 与按 token 批处理的融合 decode 图（`TS_PER_SEQ_FUSED=0` 关闭 holder）。其他后端走分页 `ForwardBatch` 路径：每层分页 K/V，注意力 sinks 通过原生 `TSGgml_PagedAttentionForwardWithSinks`（或托管 C# 回退 `TS_GPTOSS_PAGED_ATTN_MANAGED=1`）。`--tp` 下两条路径都不可用，并发请求走旧的按序列 KV-swap 路径。`TS_GPTOSS_BATCHED=0` 只撤下分页路径，用于 A/B 对比。详见 §11。 |
| 输出解析器 | `HarmonyOutputParser`（始终启用） |

## 下载

已验证的 GGUF 下载指引：

| 模型 | HF 仓库 | 推荐文件 | 说明 |
|---|---|---|---|
| gpt-oss-20b（MoE） | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) | `gpt-oss-20b-MXFP4.gguf`（12.110 GB） | 原生 MXFP4 expert 量化；仅文本。思维链始终开启（Harmony analysis channel），支持工具调用。官方上游：[openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)（Apache-2.0）。 |

命令行下载（每个文件一行；需要先 `pip install -U huggingface_hub`）：

```bash
python -m pip install -U huggingface_hub
hf download ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf --local-dir models
```

CLI 单次推理（文本提示词通过 `--input` 从文件读取；CLI 采样默认为 greedy，
`--max-tokens` 默认为 100）：

```bash
dotnet run --project TensorSharp.Cli -c Release -- --model models/gpt-oss-20b-MXFP4.gguf \
  --input prompt.txt --max-tokens 512 --backend ggml_cuda
```

服务端（聊天 Web UI 以及 OpenAI/Ollama 兼容 API，位于 `http://localhost:5000`）：

```bash
dotnet run --project TensorSharp.Server.Host -c Release -- --model models/gpt-oss-20b-MXFP4.gguf \
  --backend ggml_cuda --max-tokens 4096
```

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
- **Attention pattern**：偶数层 ⇒ SWA（窗口取自 `_slidingWindow`，默认 128）；奇数层 ⇒ 全因果。所有 decode / prefill 路径都会把偶数层（SWA）限制在 `_slidingWindow` 之内：托管 per-op 路径、逐层融合图与整模型融合图、以及分页路径。

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
- **Sinks softmax。** 在 GGML 后端上，下文的融合整模型 decode 图在设备上执行带 SWA 掩码与 sinks 的 flash attention 以及 `mul_mat_id` 专家。在 per-op 路径上，GGML 后端在上下文不超过 `TS_GPTOSS_FUSED_DECODE_MAX_CTX`（默认 4096）个 token 时用逐层融合内核做 decode 注意力，`cuda` 有 GPU sinks decode 内核（`CudaFusedOps.TryGqaDecodeAttentionWithSinks`），`mlx` 有对应的 Metal 内核；其余情况下 sinks softmax 跑在 CPU（标量 + 可选 SIMD 求 exp-sum）。这些路径都会把 SWA 层限制在 `_slidingWindow` 之内。
- **MXFP4 expert 权重** 量化保留在 `_quantWeights`，matmul 由后端的量化 matmul 派发。

GPT OSS 已经把**整个 decode token 作为一次 GGML 图派发**执行 —— 每一层、MoE 路由与专家、最后的 norm 与 LM head 全部在 `GptOssModel.FusedModelDecode.cs` 的同一张图里，这也是 ggml-cuda 能把它整体捕获成 CUDA graph 的原因。A40 实测：decode 从 24 → 154 tok/s，并且随上下文长度基本持平（16K 时仍有 133 tok/s），而逐层路径在同样长度下已经掉到 2.3。设 `TS_GPTOSS_MODEL_DECODE=0` 可退回 per-op 派发。

## 10. 内存与 KV cache 策略

- 每层 K、V tensor 形状 `[NumKVHeads, maxSeqLen, headDim]`。KV dtype 为 `f32` 或 `f16`；显式请求 `q8_0` / `q4_0` 时会在 stderr 上提示并降为 `f16`（两张融合图与托管 sinks 回退都读不了块量化 cache）。
- 跨请求的前缀复用以分页家族的方式走 Radix 前缀缓存（默认模式）：缓存的页加上常驻的主缓存，回退是精确的（最多 16 个 token），因为滑动窗口只是在线性 cache 上做掩码（`GptOssModel.PrefixCache.cs`）。
- `ResetKVCache()` 全部清零。
- Expert FFN 权重在 `ModelBase` 加载的原始 3D `ffn_gate_exps.weight` / `ffn_up_exps.weight` / `ffn_down_exps.weight` 块中。`FuseExpertGateUpWeights()` 只 dispose per-expert *view*，不动底层大缓冲。这样融合 MoE prefill kernel 仍能通过 `_layerStackedGate` / `_layerStackedUp` / `_layerStackedDown` 直接寻址原始块。

## 11. 批处理 / 分页前向（连续批处理）

GPT OSS 实现了 `IBatchedPagedModel.ForwardBatch`
（[`GptOssModel.BatchedForward.cs`](../../TensorSharp.Models/Models/GptOss/GptOssModel.BatchedForward.cs)）
并默认保持可用；设置 `TS_GPTOSS_BATCHED=0` 可撤下它用于 A/B 对比。在下文按请求的
holder 不适用的地方，剩下的就是旧的按序列 KV-swap 回退；holder 本身不读这个变量。

在不带 `--tp` 的 GGML 后端上，有两个及以上运行中请求的步骤并不走这条分页路径。每个
请求从自己的 KV holder 解码，holder 通过指针切换换入
（[`GptOssModel.PerSeqCache.cs`](../../TensorSharp.Models/Models/GptOss/GptOssModel.PerSeqCache.cs)），
走 §9 的融合整模型 decode 图；由两个及以上这样的请求共享的 decode 步则作为**一张**按
token 批处理的融合图执行（`GgmlBasicOps.TryGptOssModelDecodeBatched`，
[`ggml_ops_gptoss_batched.cpp`](../../TensorSharp.GGML.Native/ggml_ops_gptoss_batched.cpp)），
所有请求只读一遍权重。按 token 批处理的步骤默认开启（`TS_BATCHED_FUSED_DECODE=0` 退回
每步每个请求各跑一次融合前向）；当路由专家被卸载到 CPU 时拒绝该步，每个请求各自跑一次
融合前向；某个请求的 holder 在这一步需要扩容时，只有它退出批处理单独 decode，其余请求仍一起
批处理。非 GGML 后端（以及设置
`TS_PER_SEQ_FUSED=0` 时）走下面的分页路径。`--tp` 下 holder 与分页路径都不可用，
并发请求走旧的 KV-swap 路径。本卡片没有记录按 token 批处理路径的吞吐。

该批处理移植需要在分页调度栈内保留 GPT OSS 三大架构特征
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
- **推理无法关闭，只能缩短。** 模型总是先打开 analysis channel 再作答；唯一能控制推理长度的开关是 Harmony system 消息里的 `Reasoning: low|medium|high` 一行。OpenAI 的 `reasoning_effort` 请求字段（`low`、`medium`、`high`；其他值返回 HTTP 400）设置这一行，默认 `medium`；Responses API 的写法是 `reasoning: {"effort": ...}`。显式发送 `"think": false` 且未指定 effort 的请求按 `low` 渲染（不带 `think` 则保持 `medium`）——此前这一行硬编码为 `medium`，因此关闭思维链的 256 token 请求会把整个预算花在 analysis 上，以 `finish_reason=length` 且没有 content 结束。该等级参与共享前缀 checkpoint 的 key，Web UI 也应用同样的 `think:false → low` 映射，使启动预热同时准备两种前缀。
- **`response_format` 可以与 `think: true` 同时使用。** 两种模式下 final channel 都以同一个 `final<|message|>` 头开始，因此 JSON 语法在同一位置启用（`ThinkingGrammarActivationTrigger`），模型可以先推理再输出受约束的答案。
- **支持工具调用**（通过 Harmony `commentary` channel）。当请求包含 `tools` 时：
  - system 消息追加 “Calls to these tools must go to the commentary channel: 'functions'.”，developer 消息追加 `# Tools` 块，把每个工具声明为 TypeScript namespace（`namespace functions { type NAME = (_: { ... }) => any; }`）。
  - 模型以 `<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}<|call|>` 形式输出调用；`HarmonyOutputParser` 解析 channel 与 `to=functions.NAME` recipient，并把 JSON 参数解析为 `ToolCall`（每轮最多一个调用，与参考实现一致）。
  - 工具结果以 `<|start|>functions.NAME to=assistant<|channel|>commentary<|message|>{result}<|end|>` 形式回传。
  - **停止 token：** 工具调用以 `<|call|>`（id 200012）结束，而 gpt-oss GGUF 只把 `<|return|>`（id 200002）列为 eos。`ModelBase` 为 `gptoss`/`gpt-oss` 架构把 `<|call|>` 加入停止集合，使生成在调用后停下来以便解析。
  - 已用 gpt-oss-20b 在 [`HarmonyToolCallIntegrationTests`](../../InferenceWeb.Tests/HarmonyToolCallIntegrationTests.cs) 做端到端往返验证；格式/解析覆盖见 [`HarmonyToolCallTests`](../../InferenceWeb.Tests/HarmonyToolCallTests.cs)。
- 聊天模板使用 GPT-4o pre-tokenizer。渲染由硬编码的 `ChatTemplate.RenderHarmony` 处理（GGUF 内置的 Harmony Jinja 依赖递归宏 / `namespace()` / `strftime_now` 等轻量 Jinja2 引擎无法完全支持的特性，尤其在工具路径上），因此 gpt-oss 与 `mistral3`、`nemotron_h` 一样走硬编码渲染器。

## 13. 优化机会

- **融合 per-layer 批处理内核** —— 旧路径的 `TryFusedAttnLayerPrefill` 每
  层只发一张 cgraph（norm + 融合 QKV + RoPE + KV-append + 带 sinks 的
  softmax + attn + output proj + residual）。分页 `ForwardBatch` 路径目前每层
  仍要发 ~5 张图。GGML 后端上的并发 decode 如今改走按请求的 holder 与按 token
  批处理的融合图（§11），因此这一点只在仍然使用分页路径的地方有意义。
- **任何发布版都用融合 QKV** —— 当 GGUF 拆分 Q / K / V 时，每次单独 matmul
  后还是要单独 bias add。`FusedQKVWithBias` 图能把 3 次派发合 1。
- **把整模型 decode 铺到分页路径** —— 融合的整模型 decode 图
  （`GptOssModel.FusedModelDecode.cs`）服务单序列 decode 与按请求的 holder，
  它的按 token 批处理变体服务 GGML 后端上的并发 decode；分页 `ForwardBatch` 路径
  没有用到它，把同样的单次派发做法延伸过去，能消除该路径剩下的大部分托管开销。
- **per-op GGML 路径上的设备端 sinks 注意力（旧路径）** —— 这只与 per-op 路径（§9）
  有关。融合整模型 decode 图已经在设备上执行 sinks 注意力，`cuda` 与 `mlx` 的 per-op
  路径也是如此。设置 `TS_GPTOSS_MODEL_DECODE=0` 时，GGML 后端在上下文超过
  `TS_GPTOSS_FUSED_DECODE_MAX_CTX` 后仍会退回 CPU sinks softmax；放开该逐层内核的
  上下文上限即可补上这段差距。
- **per-expert decode 批处理（`cpu` / `cuda`）** —— GGML 后端已经把单 token MoE 作为一次 `mul_mat_id` 派发执行，`mlx` 用 `gather_qmm` 分组 GEMM，但 `cpu` 与 `cuda` 的 per-op 路径仍然每 token 顺序枚举 expert。在那里引入批量 decode 路径（类比 Qwen 3.5 的 `MoEExpertsSwiGLUResidual`）能把 `numExpertsUsed` 次派发合并为 1 次。
