# 模型架构卡片 — 开发者参考

[English](model_cards.md) | [中文](model_cards_cn.md)

TensorSharp 支持六种模型架构。本文档面向需要修改、优化或扩展模型实现的工程师。

所有模型类位于 `TensorSharp.Models/Models/<Name>/` 目录下，继承自 `ModelBase`（`TensorSharp.Models/ModelBase.cs`）。`ModelBase` 提供共享基础设施：GGUF 加载、权重存储（`_weights` 存 F32、`_quantWeights` 存量化权重）、KV 缓存辅助方法、嵌入查询、RMSNorm、线性前向、RoPE 工具函数、性能计时，以及 `Forward(int[] tokens) → float[]` 接口。

---

## Gemma 3

| 属性 | 值 |
|---|---|
| 源文件 | `TensorSharp.Models/Models/Gemma3/Gemma3Model.cs` |
| 提供方 | Google |
| GGUF 架构标识 | `gemma3` |
| 示例模型 | gemma-3-4b、gemma-3-12b、gemma-3-27b |
| 模态 | 文本、图像 |
| 思维链模式 | 不支持 |
| 工具调用 | 不支持 |
| 输出解析器 | `PassthroughOutputParser` |

### 架构概述

Gemma 3 是一个密集型 Transformer，采用交替的局部/全局注意力。

- **注意力模式**：`IsGlobalLayer(layer)` 在 `(layer + 1) % 6 == 0` 时返回 true。全局层使用全因果注意力；其他层使用滑动窗口注意力（SWA）。
- **头维度**：可配置的 `_attnKeyLen`（默认 256）和 `_attnValLen`（默认 256），从 GGUF `gemma3.attention.key_length` / `value_length` 读取。
- **RoPE**：NeoX 风格，使用独立的频率基数。局部层使用 `_ropeLocalBase`（`gemma3.rope.local.freq_base`，默认 10000）；全局层使用 `_ropeGlobalBase`（`gemma3.rope.freq_base`）并按 `1 / _ropeScale` 缩放。27B 变体（`NumLayers == 34`）的 `_ropeScale` 硬编码为 8.0。频率表在 `PrecomputeRoPE()` 中一次性预计算。
- **归一化**：每块四个 RMSNorm——`attn_norm`、`post_attention_norm`、`ffn_norm`、`post_ffw_norm`。Q 和 K 还通过 `ApplyBatchRMSNorm()` 或 `RMSNormInPlace()` 进行逐头 RMSNorm。
- **FFN**：GeGLU——`GELU(gate) * up`（通过 `Ops.GELUMul`），然后 `down` 投影。使用融合的 `ffn_gate_up.weight`。
- **嵌入**：token 嵌入按 `sqrt(hidden_size)` 缩放（`ScaleEmbedding()`）。
- **输出**：当 `output.weight` 不存在时，与 `token_embd.weight` 绑定。
- **Logit 软封顶**：`tanh(logits / cap) * cap`，当 `_finalLogitSoftcap > 0` 时应用。
- **视觉**：`Gemma3VisionEncoder` 通过 `LoadVisionEncoder(mmProjPath)` 单独加载。视觉嵌入通过 `InjectVisionEmbeddings()` 在 `<start_of_image>` 位置注入文本隐状态。

### GGUF 元数据键

| 键 | 类型 | 说明 |
|---|---|---|
| `gemma3.attention.sliding_window` | uint32 | SWA 窗口大小（默认 1024）|
| `gemma3.attention.key_length` | uint32 | Key 头维度（默认 256）|
| `gemma3.attention.value_length` | uint32 | Value 头维度（默认 256）|
| `gemma3.rope.local.freq_base` | float32 | 局部 RoPE 基数（默认 10000）|
| `gemma3.rope.freq_base` | float32 | 全局 RoPE 基数 |
| `gemma3.final_logit_softcapping` | float32 | Logit 软封顶（0 = 禁用）|

### 权重命名规范

```
token_embd.weight                         # Token 嵌入 [vocab, hidden]
blk.{L}.attn_norm.weight                  # 注意力前 RMSNorm
blk.{L}.attn_q.weight                     # Q 投影
blk.{L}.attn_k.weight                     # K 投影
blk.{L}.attn_v.weight                     # V 投影
blk.{L}.attn_q_norm.weight                # 逐头 Q RMSNorm
blk.{L}.attn_k_norm.weight                # 逐头 K RMSNorm
blk.{L}.attn_output.weight                # 输出投影
blk.{L}.post_attention_norm.weight         # 注意力后 RMSNorm
blk.{L}.ffn_norm.weight                   # FFN 前 RMSNorm
blk.{L}.ffn_gate.weight  }                # 融合前：独立 gate/up
blk.{L}.ffn_up.weight    }
blk.{L}.ffn_gate_up.weight                # 融合后：拼接 [2*intermed, hidden]
blk.{L}.ffn_down.weight                   # FFN down 投影
blk.{L}.post_ffw_norm.weight              # FFN 后 RMSNorm
output_norm.weight                        # 最终 RMSNorm
output.weight                             # LM head（绑定时可选）
```

### 前向传播流程（每 token）

```
tokens → Embedding → ScaleEmbedding(sqrt(hidden)) → [注入视觉]
For each layer L:
  hidden → RMSNorm(attn_norm)
         → Q,K,V 投影 → QK-norm → RoPE(局部或全局) → 缩放 Q
         → 注意力(SWA 或全因果) → O 投影
         → RMSNorm(post_attention_norm) → 残差相加
         → RMSNorm(ffn_norm)
         → GateUp → GELU(gate)*up → Down
         → RMSNorm(post_ffw_norm) → 残差相加
hidden → RMSNorm(output_norm) → LM head → [软封顶] → logits
```

### KV 缓存

- 形状：每层 `[numKVHeads, maxSeqLen, headDim]`。
- 所有层统一容量（SWA 和全局层相同）。SWA 限制通过 `AttentionDecodeWithWindow()` 的 `attendStart..totalSeqLen` 窗口实现。
- `ResetKVCache()` 将所有缓存填零并调用 `InvalidateTensorDeviceCache()` 同步 GPU 状态。

### 优化空间

- 尚未融合 QKV——Q、K、V 是独立投影。融合可以减半注意力调度次数。
- 无融合 GPU decode 路径——每个操作独立调度。单图方法（类似 Gemma 4）可显著提升 Metal/CUDA 吞吐量。
- SWA 层可使用环形 KV 缓存（类似 Gemma 4）以限定内存。

---

## Gemma 4

| 属性 | 值 |
|---|---|
| 源文件 | `TensorSharp.Models/Models/Gemma4/Gemma4Model.cs` |
| 提供方 | Google |
| GGUF 架构标识 | `gemma4` |
| 示例模型 | gemma-4-E4B、gemma-4-31B、gemma-4-26B-A4B（MoE）|
| 模态 | 文本、图像、视频、音频 |
| 思维链模式 | 支持（`<\|channel>thought`）|
| 工具调用 | 支持（`<\|tool_call>`）|
| 输出解析器 | `Gemma4OutputParser` |

### 架构概述

Gemma 4 是 TensorSharp 中功能最丰富的架构，具有逐层异构特性。

- **注意力模式**：`_slidingWindowPattern[]` 布尔数组来自 GGUF。`IsLocalLayer(layer)` 检查此数组。局部 → SWA，非局部 → 全因果。
- **逐层头维度**：`_localHeadDim`（`key_length_swa`，默认 256）和 `_globalHeadDim`（`key_length`，默认 512）。`HeadDimForLayer(layer)` 返回对应值。`DetectHeadDimsFromWeights()` 自动校正 GGUF 元数据与实际权重形状不匹配的情况。
- **逐层 KV 头数**：`_numGlobalKVHeads` 可能不同于 `Config.NumKVHeads`（局部层的值）。`KVHeadsForLayer(layer)` 返回正确数量。
- **KV 共享**：最后 `_sharedKVLayers` 层复用较早"捐赠者"层的 KV 缓存。`_kvDonorMap[layer]` → 捐赠者层。共享层跳过 K/V 投影和缓存写入，仅投影 Q。
- **逐层嵌入（PLE）**：当 `_pleDim > 0` 时，从 `per_layer_token_embd.weight` + `per_layer_model_proj.weight` 计算逐层输入向量，归一化后通过 `GELU(inp_gate(hidden)) * pleInput → proj → post_norm → 残差相加` 注入每个块。
- **逐层输出缩放**：`_layerScalars[layer]` 来自 `blk.{L}.layer_output_scale.weight`，在残差相加后应用。
- **RoPE**：局部层使用标准 NeoX RoPE + `_ropeLocalBase`。全局层使用部分旋转维度（`_partialRotaryDims`）和来自 `rope_freqs.weight` 的比例频率因子。
- **V 归一化**：通过 `ApplyUnweightedRMSNorm()` 使用全 1 张量对 V 投影进行无权重 RMSNorm。
- **MoE**：部分变体含 MoE 层。`HasMoE(layer)` 检查 `ffn_gate_inp.weight`。MoE 层同时运行密集 MLP 和稀疏 MoE，然后合并：`PostNorm1(MLP) + PostNorm2(MoE) → PostNorm → 残差`。路由使用无权重 RMSNorm + 学习缩放 + 线性 → softmax → TopK。
- **Logit 软封顶**：与 Gemma 3 相同。
- **视觉/音频**：`Gemma4VisionEncoder` 处理图像/视频帧；`Gemma4AudioEncoder` 通过梅尔频谱图处理音频。

### GGUF 元数据键

| 键 | 类型 | 说明 |
|---|---|---|
| `gemma4.attention.sliding_window_pattern` | bool[] | 逐层 SWA 模式 |
| `gemma4.attention.sliding_window` | uint32 | SWA 窗口大小（默认 512）|
| `gemma4.attention.key_length` | uint32 | 全局头维度（默认 512）|
| `gemma4.attention.key_length_swa` | uint32 | 局部头维度（默认 256）|
| `gemma4.attention.global_head_count_kv` | uint32 | 全局层 KV 头数 |
| `gemma4.attention.head_count_kv` | int32[] | 逐层 KV 头数 |
| `gemma4.attention.shared_kv_layers` | uint32 | 尾部 KV 共享层数 |
| `gemma4.rope.dimension_count` | uint32 | 部分旋转维度数 |
| `gemma4.rope.partial_rotary_factor` | float32 | 旋转维度占头维度的比例 |
| `gemma4.rope.freq_base_swa` | float32 | 局部 RoPE 基数 |
| `gemma4.embedding_length_per_layer_input` | uint32 | PLE 维度（0 = 禁用）|
| `gemma4.expert_count` | uint32 | MoE 专家数（0 = 密集）|
| `gemma4.expert_used_count` | uint32 | 每 token 使用的 TopK 专家数 |

### 权重命名规范

```
blk.{L}.attn_norm.weight
blk.{L}.attn_qkv.weight                   # 融合 QKV（非共享层）
blk.{L}.attn_q.weight                     # 仅 Q（KV 共享层）
blk.{L}.attn_q_norm.weight / attn_k_norm.weight
blk.{L}.attn_output.weight
blk.{L}.post_attention_norm.weight         # （或 attn_post_norm.weight）
blk.{L}.ffn_norm.weight
blk.{L}.ffn_gate_up.weight                # 融合 gate+up
blk.{L}.ffn_down.weight
blk.{L}.post_ffw_norm.weight               # （或 ffn_post_norm.weight）
blk.{L}.layer_output_scale.weight          # 标量 [1]
blk.{L}.inp_gate.weight                   # PLE 门控
blk.{L}.proj.weight                       # PLE 投影
blk.{L}.post_norm.weight                  # PLE 后归一化
# MoE 层额外包含：
blk.{L}.ffn_gate_inp.weight               # 路由器
blk.{L}.ffn_gate_inp.scale                # 路由器学习缩放
blk.{L}.ffn_gate_up_exps.{E}.weight       # 融合专家 gate+up
blk.{L}.ffn_down_exps.{E}.weight          # 专家 down
blk.{L}.pre_ffw_norm_2.weight             # MoE 输入归一化
blk.{L}.post_ffw_norm_1.weight            # 密集 MLP 后归一化
blk.{L}.post_ffw_norm_2.weight            # MoE 后归一化
rope_freqs.weight                         # 比例 RoPE 频率因子
per_layer_token_embd.weight               # PLE token 嵌入
per_layer_model_proj.weight               # PLE hidden→PLE 投影
per_layer_proj_norm.weight                # PLE 投影归一化
```

### 前向传播流程（每 token）

```
tokens → Embedding → ScaleEmbedding → [注入视觉/音频]
ComputePLE(tokens, hidden) → perLayerInputs
For each layer L:
  hidden → RMSNorm(attn_norm)
         → QKV（或仅 Q，共享层）→ QK-norm → V-norm(无权重) 
         → RoPE(局部或全局+比例) → 注意力 → O 投影
         → RMSNorm(post_attn_norm) → 残差相加
  若 MoE:
         → RMSNorm(ffn_norm) → GeGLU(密集) → PostNorm1
         → RMSNorm(pre_ffw_norm_2) → MoE(路由+专家) → PostNorm2
         → PostNorm1 + PostNorm2 → PostNorm → 残差相加
  否则:
         → RMSNorm(ffn_norm) → GeGLU → RMSNorm(post_ffw) → 残差相加
  若 PLE:
         → GELU(inp_gate(hidden)) * perLayerInput → proj → RMSNorm → 残差相加
  hidden *= layerScalar
hidden → RMSNorm(output_norm) → LM head → [软封顶] → logits
```

### KV 缓存

- **逐层容量**：SWA 层分配 `_slidingWindow` 个槽位；全局层分配 `maxSeqLen`。
- **SWA 环形缓存**：`CopyToCacheCircular()` 和 `AttentionDecodeCircular()` 使用 `pos % cacheSize`。
- **共享层**：`_kvCacheK[sharedLayer]` 指向 `_kvCacheK[donorLayer]`——无独立分配。

### 融合 GPU Decode

当所有层均为密集层（无 MoE）且所有权重为量化权重时自动启用。`BuildGemma4DecodeArrays()` 将逐层指针、类型和维度打包到数组中。`NativeGemma4ModelDecode()` 调用 `GgmlBasicOps.Gemma4ModelDecode()`——一次原生调用处理所有层、PLE、异构头维度、环形 KV 缓存和层缩放因子，在一次 GPU 调度中完成。

### 优化空间

- MoE 层会禁用融合 decode 路径。融合 MoE GPU 内核可恢复加速。
- 专家 FFN 按 token 顺序运行。跨专家批处理可提升吞吐量。

---

## Qwen 3

| 属性 | 值 |
|---|---|
| 源文件 | `TensorSharp.Models/Models/Qwen3/Qwen3Model.cs` |
| 提供方 | 阿里巴巴 |
| GGUF 架构标识 | `qwen3` |
| 示例模型 | Qwen3-4B、Qwen3-8B、Qwen3-14B、Qwen3-32B |
| 模态 | 仅文本 |
| 思维链模式 | 支持（`<think>`）|
| 工具调用 | 支持（`<tool_call>`）|
| 输出解析器 | `Qwen3OutputParser` |

### 架构概述

Qwen 3 是 TensorSharp 中最简单的密集型 Transformer，作为参考实现。

- **注意力**：GQA，`Config.NumKVHeads < Config.NumHeads`。融合 QKV 投影为单个 `attn_qkv.weight`。
- **QK-norm**：Q 和 K 的逐头 RMSNorm。
- **RoPE**：NeoX 风格。频率预计算在 `_ropeFreqs[]` 中。Decode 路径使用手动优化的 C# 循环配合 stackalloc cos/sin 表。Prefill 路径使用 `Ops.RoPEEx`。
- **FFN**：SwiGLU——`SiLU(gate) * up`（通过 `Ops.SiLUMul`），然后 `down`。使用融合的 `ffn_gate_up.weight`。
- **归一化**：每块两个 RMSNorm——`attn_norm` 和 `ffn_norm`。

### GGUF 元数据键

仅标准键：`qwen3.attention.head_count_kv`、`qwen3.rope.freq_base` 等。

### 权重命名规范

```
blk.{L}.attn_norm.weight
blk.{L}.attn_qkv.weight                   # 融合 Q+K+V [qDim+kDim+kDim, hidden]
blk.{L}.attn_q_norm.weight / attn_k_norm.weight
blk.{L}.attn_output.weight
blk.{L}.ffn_norm.weight
blk.{L}.ffn_gate_up.weight                # 融合 gate+up [2*intermed, hidden]
blk.{L}.ffn_down.weight
output_norm.weight
output.weight
```

### 前向传播流程（每 token）

```
tokens → Embedding
若 decode (seqLen==1) 且 GGML 后端:
  → NativeTransformerModelDecode（所有层在一次原生调用中完成）
否则:
  For each layer L:
    hidden → RMSNorm(attn_norm)
           → QKV(融合) → 分拆 Q,K,V → QK-norm → RoPE → 注意力 → O 投影
           → 残差相加
           → RMSNorm(ffn_norm)
           → GateUp → SiLU(gate)*up → Down
           → 残差相加
hidden → RMSNorm(output_norm) → 截取最后 token → LM head → logits
```

### 原生 Decode 路径

**逐层**（`NativeTransformerLayerDecode`）：调用 `GgmlBasicOps.TransformerLayerDecode()`，传入归一化权重、QKV、Q/K norm、输出、FFN norm、gate+up、down、KV 缓存的原始指针。在原生代码中处理一层。

**整模型**（`NativeTransformerModelDecode`）：调用 `GgmlBasicOps.TransformerModelDecode()`，传入逐层指针数组。在单次原生调用中处理所有层。由 `BuildModelDecodeArrays()` 构建，在模型加载时预解析所有权重指针。`_modelDecodeArrays` 结构缓存量化权重元数据（type、ne0、ne1、rawBytes），decode 时无需字典查找。

### 优化空间

- 原生整模型路径已是当前最快。进一步优化需要类似 Gemma 4 的 GPU 融合路径。
- Prefill 仍使用 C# 托管路径。原生 prefill 路径可加速提示处理。

---

## Qwen 3.5

| 属性 | 值 |
|---|---|
| 源文件 | `TensorSharp.Models/Models/Qwen35/Qwen35Model.cs` |
| 提供方 | 阿里巴巴 |
| GGUF 架构标识 | `qwen35`、`qwen35moe`、`qwen3next` |
| 示例模型 | Qwen3.5-9B、Qwen3.5-32B |
| 模态 | 文本、图像 |
| 思维链模式 | 支持（`<think>`）|
| 工具调用 | 支持（`<tool_call>`）|
| 输出解析器 | `Qwen35OutputParser`（继承 `Qwen3OutputParser`）|

### 架构概述

Qwen 3.5 是一个混合模型，混合使用全注意力层和 GatedDeltaNet 循环层。

- **层类型**：`_isRecurrent[layer]` 在 `(layer + 1) % _fullAttentionInterval != 0` 时为 true（默认间隔 4）。48 层、间隔 4 时：36 个循环层 + 12 个注意力层。
- **全注意力层**：与 Qwen 3 相同，但使用**门控 Q**——Q 投影输出 `2 * numHeads * headDim`。输出解交错：前半为 Q，后半为 sigmoid 门控。注意力输出乘以 `sigmoid(gate)`（`Ops.SigmoidMul`）。
- **GatedDeltaNet 循环层**：线性 SSM，包含：
  - **投影**：`attn_qkv.weight`（Q+K+V）、`attn_gate.weight`（z）、`ssm_beta.weight`、`ssm_alpha.weight`。
  - **Conv1d**：核大小来自 GGUF，在 QKV 状态的滑动窗口上应用。状态存储在 `_convState[layer][]`。
  - **SiLU 激活**应用于 conv 输出。
  - **头扩展**：若 `_numKHeads != _numVHeads`，Q 和 K 按头重复。
  - **L2 归一化**：逐头对 Q 和 K 进行 L2 归一化。
  - **循环状态更新**：`state = exp(gate) * state + sigmoid(beta) * (v - state@k) ⊗ k^T`。通过 `Ops.AddmmBatch` 实现批量秩-1 外积。
  - **输出**：`SiLU(z) * RMSNorm(state @ q)`（`Ops.SiLUMul` + `Ops.RMSNorm`）。
  - **状态形状**：每层 `[numVHeads, headVDim, headKDim]`。
- **Prefill**：循环层**逐 token 顺序处理**（遍历 `seqLen`），注意力层使用批量 prefill。
- **视觉**：`Qwen35VisionEncoder` 支持动态分辨率。视觉嵌入注入 `<|image_pad|>` 位置。

### GGUF 元数据键

| 键 | 类型 | 说明 |
|---|---|---|
| `qwen35.ssm.inner_size` | uint32 | `headVDim * numVHeads` |
| `qwen35.ssm.state_size` | uint32 | `headKDim` |
| `qwen35.ssm.group_count` | uint32 | `numKHeads` |
| `qwen35.ssm.time_step_rank` | uint32 | `numVHeads` |
| `qwen35.ssm.conv_kernel` | uint32 | Conv1d 核大小 |
| `qwen35.full_attention_interval` | uint32 | 每 N 层为全注意力（默认 4）|
| `qwen35.rope.dimension_sections` | int32[] | MRoPE 分段边界 |
| `qwen35.rope.dimension_count` | uint32 | RoPE 维度数 |

### 权重命名规范

```
# 注意力层（每 N 层）：
blk.{L}.attn_norm.weight
blk.{L}.attn_q.weight                     # [numHeads*headDim*2, hidden]（Q+gate 交错）
blk.{L}.attn_k.weight / attn_v.weight
blk.{L}.attn_q_norm.weight / attn_k_norm.weight
blk.{L}.attn_output.weight
blk.{L}.post_attention_norm.weight
blk.{L}.ffn_gate_up.weight / ffn_down.weight

# 循环层：
blk.{L}.attn_norm.weight
blk.{L}.attn_qkv.weight                   # [qkDim*2 + vDim, hidden]
blk.{L}.attn_gate.weight                  # z 门控 [ssmDInner, hidden]
blk.{L}.ssm_beta.weight                   # [numVHeads, hidden]
blk.{L}.ssm_alpha.weight                  # [numVHeads, hidden]
blk.{L}.ssm_conv1d.weight                 # [qkvDim, convKernel]
blk.{L}.ssm_dt.bias                       # dt 偏置 [numVHeads]
blk.{L}.ssm_a                             # a 参数 [numVHeads]
blk.{L}.ssm_norm.weight                   # 输出 RMSNorm
blk.{L}.ssm_out.weight                    # 输出投影
blk.{L}.post_attention_norm.weight
blk.{L}.ffn_gate_up.weight / ffn_down.weight
```

### 缓存架构

- **注意力层**：标准 KV 缓存 `[numKVHeads, maxSeqLen, headDim]`。
- **循环层**：`_convState[layer]` 浮点数组，大小 `(convKernel-1) * qkvDim`，用于 conv1d 滑动窗口；`_deltaStateTensor[layer]` 张量，形状 `[numVHeads, headVDim, headKDim]`，用于 SSM 状态。
- `ResetKVCache()` 同时将 KV 缓存和循环状态清零。

### 预分配缓冲区

为避免 GDN decode 热路径中的逐步分配，以下缓冲区在 `InitGDNBuffers()` 中一次性分配：

| 缓冲区 | 形状 | 用途 |
|---|---|---|
| `_gdnConvOutT` | [1, qkvDim] | Conv1d 输出 + SiLU |
| `_gdnKBuf` / `_gdnQBuf` | [numVHeads, headKDim] | 扩展 + L2 归一化后的 Q/K |
| `_gdnVBuf` | [numVHeads, headVDim] | V 分拆 |
| `_gdnKvMemBuf` | [numVHeads, headVDim, 1] | state@k 中间结果 |
| `_gdnCoreOutBuf` | [numVHeads, headVDim, 1] | state@q 输出 |
| `_gdnGatedOutT` | [1, ssmDInner] | 最终门控输出 |

### 优化空间

- 循环层 prefill 是顺序的（逐 token）。分块并行 prefill（同时通过 SSM 处理多个 token）可大幅提升提示吞吐量。
- 尚无原生 decode 路径。将 GDN decode 移至原生 C/CUDA 可避免托管开销。
- Conv1d 实现为标量循环。向量化 SIMD 实现可带来提升。

---

## GPT OSS

| 属性 | 值 |
|---|---|
| 源文件 | `TensorSharp.Models/Models/GptOss/GptOssModel.cs` |
| 提供方 | OpenAI |
| GGUF 架构标识 | `gptoss`、`gpt-oss` |
| 示例模型 | gpt-oss-20b |
| 模态 | 仅文本 |
| 思维链模式 | 支持（Harmony：`<\|channel>analysis` / `<\|channel>final`）|
| 工具调用 | 不支持 |
| 输出解析器 | `HarmonyOutputParser`（`AlwaysRequired = true`）|

### 架构概述

GPT OSS 是一个专家混合 Transformer，具有多项独特设计。

- **MoE 路由**：每层 `_numExperts` 个专家（32），TopK 路由（`_numExpertsUsed` = 4）。路由过程为 `linear(hidden) + bias → TopK → softmax(仅选中的)`。注意：与 Gemma 4（先 softmax 再 TopK）不同，GPT OSS 先 TopK 再 softmax（与 Ollama 实现一致）。
- **注意力沉降（Attention Sinks）**：逐头学习偏置 `attn_sinks.weight`（形状 `[numHeads]`）。作为虚拟 token 纳入 softmax：其值同时参与最大值查找和指数求和，有效添加一个始终可用的注意力目标。实现在 `ApplySoftmaxWithSinks()` 和 `AttentionDecodeWithSinks()` 中。
- **注意力模式**：偶数层 → SWA（窗口 `_slidingWindow`，默认 128），奇数层 → 全因果。注意：与 Gemma 3/4 使用可配置模式不同，GPT OSS 使用简单奇偶规则。
- **激活函数**：SiLUAlphaLimit——`x * sigmoid(alpha * x) * (y + 1)`，alpha=1.702，limit=7.0。Gate（x）截断至 `[-inf, limit]`，up（y）截断至 `[-limit, limit]`。常量 `SiluAlpha` 和 `SiluLimit` 硬编码。
- **全面偏置**：所有线性投影使用 `LinearForwardWithBias()`，在 matmul 后逐行添加偏置。偏置从 `_weights[biasName]` 查找。
- **RoPE**：NeoX 风格，Yarn 缩放。`Ops.RoPEEx` 调用时传入 `origCtxLen=Config.OriginalContextLength`（4096）、`freqScale=1/RopeScale`、`beta_fast=32`、`beta_slow=1`。
- **归一化**：每块两个 RMSNorm——`attn_norm` 和 `post_attention_norm`。
- **Prefill 优化**：仅**最后一个** Transformer 层在 MoE 前截取最后一个 token。其他所有层处理完整序列。由 `TransformerBlock()` 的 `isLastLayer` 参数控制。
- **量化**：专家权重可能为 MXFP4（GGML 类型 39）。
- **分词器**：GPT-4o BPE 预分词器，数字分组使用 `\p{N}{1,3}`。
- **聊天模板**：Harmony 格式。解析器（`HarmonyOutputParser`）设置 `AlwaysRequired = true`——即使未显式开启思维链/工具标志也始终运行，因为模型始终用通道标签包裹输出。

### GGUF 元数据键

| 键 | 类型 | 说明 |
|---|---|---|
| `gptoss.expert_count` | uint32 | 专家数量（32）|
| `gptoss.expert_used_count` | uint32 | 每 token TopK 专家数（4）|
| `gptoss.attention.sliding_window` | uint32 | SWA 窗口大小（128）|
| `gptoss.expert_feed_forward_length` | uint32 | 专家 FFN 维度 |
| `gptoss.rope.scaling.original_context_length` | uint32 | Yarn 原始上下文长度（4096）|
| `tokenizer.ggml.pre` | string | 预分词器类型（`gpt-4o`）|

### 权重命名规范

```
blk.{L}.attn_norm.weight
blk.{L}.attn_q.weight / attn_q.bias
blk.{L}.attn_k.weight / attn_k.bias
blk.{L}.attn_v.weight / attn_v.bias
blk.{L}.attn_output.weight / attn_output.bias
blk.{L}.attn_sinks.weight                  # [numHeads] 注意力沉降偏置
blk.{L}.post_attention_norm.weight
blk.{L}.ffn_gate_inp.weight / ffn_gate_inp.bias   # 路由器 [numExperts, hidden]
blk.{L}.ffn_gate_up_exps.{E}.weight        # 融合专家 gate+up
blk.{L}.ffn_gate_up_exps.{E}.bias          # 融合专家 gate+up 偏置
blk.{L}.ffn_down_exps.{E}.weight           # 专家 down
blk.{L}.ffn_down_exps.{E}.bias             # 专家 down 偏置
output_norm.weight
output.weight
```

注：GGUF 文件以打包的 `[numExperts, biasDim]` 张量（如 `ffn_gate_exps.bias`）存储偏置。`SplitExpertBiases()` 在加载时将其解包为逐专家张量。`FuseExpertGateUpWeights()` 随后将 gate+up 权重及其偏置拼接。

### 前向传播流程（每 token）

```
tokens → Embedding
For each layer L:
  hidden → RMSNorm(attn_norm)
         → Q+bias, K+bias, V+bias → RoPE(Yarn) → 注意力+Sinks → O+bias
         → 残差相加
  moeInput = hidden（若最后一层且 prefill，截取最后 token）
  moeInput → RMSNorm(post_attention_norm)
           → 路由器(linear+bias) → TopK → softmax(选中的)
           → 对每个选中专家: SiLUAlphaLimit(gate+bias, up+bias) → down+bias
           → 专家输出加权求和
           → 残差相加
hidden → RMSNorm(output_norm) → 截取最后 token → LM head → logits
```

### 优化空间

- 无融合 QKV 投影——Q、K、V 各自独立（均带偏置）。融合可减少调度次数。
- 无原生 decode 路径。整个前向传播均为托管 C#。
- 专家 FFN 按 token 顺序运行。专家批处理或稀疏内核可提升效率。
- 注意力沉降阻止使用标准 softmax 实现——自定义含沉降的 softmax 是标量 C#。SIMD 或 GPU 融合版本可提升注意力性能。

---

## Nemotron-H

| 属性 | 值 |
|---|---|
| 源文件 | `TensorSharp.Models/Models/Nemotron/NemotronModel.cs` |
| 提供方 | NVIDIA |
| GGUF 架构标识 | `nemotron_h`、`nemotron_h_moe` |
| 示例模型 | Nemotron-H-8B-Reasoning-128K、Nemotron-H-47B-Reasoning-128K |
| 模态 | 仅文本 |
| 思维链模式 | 支持（`<think>`）|
| 工具调用 | 支持（`<tool_call>`）|
| 输出解析器 | `Qwen3OutputParser` |

### 架构概述

Nemotron-H 是一个混合模型，在单个堆栈中混合三种不同的层类型：Mamba2 SSM 层、纯注意力层和纯 FFN 层（可选 MoE）。逐层类型由 GGUF 元数据数组（`head_count_kv` 和 `feed_forward_length`）决定。

- **层类型分类**：对于每层，`head_count_kv[l]` 和 `feed_forward_length[l]` 决定类型：
  - Mamba2：`head_count_kv == 0 AND feed_forward_length == 0`
  - 纯注意力：`head_count_kv > 0 AND feed_forward_length == 0`
  - 纯 FFN：`feed_forward_length > 0`
- **Mamba2 SSM 层**：带分组头的选择性状态空间模型。输入通过 `ssm_in.weight` 投影产生 z（门控）、xBC（状态输入）和 dt（时间步）。xBC 经过 conv1d 滑动窗口、SiLU 激活，然后 SSM 扫描步骤计算 `state = state * exp(-softplus(dt) * A) + B * x * dt`，输出 `y = state @ C`。最终 `SiLU(z) * GroupRMSNorm(y)` 通过 `ssm_out.weight` 投影。卷积状态和 SSM 状态在各层跨 token 维护。
- **注意力层**：标准 GQA，不使用 RoPE。逐层头数和 KV 头数从 GGUF 数组读取。融合 QKV 投影（`attn_qkv.weight`）。注意力缩放因子从 GGUF（`attention.scale`）读取，或默认为 `1/sqrt(headDim)`。
- **FFN 层**：使用 ReLU-squared 激活函数（`max(0, x)^2`），SIMD 向量化实现。启用 MoE 时，FFN 层使用基于 sigmoid 的路由，可选专家偏置（`exp_probs_b`）、top-K 选择以及可选权重归一化/缩放。支持潜空间瓶颈投影（`ffn_latent_in` / `ffn_latent_out`）和共享专家（`ffn_up_shexp` / `ffn_down_shexp`）。
- **MoE 路由**：`sigmoid(logits) + bias → TopK → normalize`。与基于 softmax 的路由不同，Nemotron-H 使用逐专家 sigmoid 概率加可选加性偏置进行专家选择。`expert_weights_norm` 将选中权重归一化为和为 1；`expert_weights_scale` 应用全局缩放因子。
- **归一化**：每块一个 RMSNorm（`attn_norm`），每层共两个 RMSNorm（块前 + 输出）。
- **聊天模板**：使用 Qwen 3 聊天模板格式（`<|im_start|>` / `<|im_end|>`）。
- **Decode 优化**：对于 decode（seqLen=1），小型操作（RMSNorm、残差相加、专家和路由器等小矩阵乘法）可在 CPU 上执行，以避免 Metal GPU 调度开销（~1ms+ 每次调度）。大型矩阵乘法（SSM in/out、注意力 QKV/output、LM head）保留在 GPU 上。

### GGUF 元数据键

| 键 | 类型 | 说明 |
|---|---|---|
| `nemotron_h.ssm.conv_kernel` | uint32 | Mamba2 conv1d 核大小 |
| `nemotron_h.ssm.inner_size` | uint32 | SSM 内部维度（nHead * headDim）|
| `nemotron_h.ssm.state_size` | uint32 | 每头 SSM 状态维度 |
| `nemotron_h.ssm.time_step_rank` | uint32 | SSM 头数 |
| `nemotron_h.ssm.group_count` | uint32 | SSM 分组数 |
| `nemotron_h.attention.head_count_kv` | uint32[] | 逐层 KV 头数（0 → Mamba2）|
| `nemotron_h.attention.head_count` | uint32[] | 逐层 Q 头数 |
| `nemotron_h.feed_forward_length` | uint32[] | 逐层 FFN 大小（0 → 无 FFN）|
| `nemotron_h.attention.scale` | float32 | 注意力缩放因子（0 = 自动）|
| `nemotron_h.expert_count` | uint32 | MoE 专家数（0 = 密集）|
| `nemotron_h.expert_used_count` | uint32 | 每 token TopK 专家数 |
| `nemotron_h.expert_weights_norm` | bool | 归一化专家权重使和为 1 |
| `nemotron_h.expert_weights_scale` | float32 | 专家权重全局缩放因子 |

### 权重命名规范

```
token_embd.weight
output_norm.weight
output.weight

# Mamba2 层：
blk.{L}.attn_norm.weight                  # 块前 RMSNorm
blk.{L}.ssm_in.weight                     # 输入投影 [hidden → 2*dInner + 2*nGroup*dState + nHead]
blk.{L}.ssm_conv1d.weight                 # Conv1d 核 [xBCSize, convKernel]
blk.{L}.ssm_conv1d.bias                   # Conv1d 偏置（可选）
blk.{L}.ssm_dt.bias                       # 时间步偏置 [nHead]
blk.{L}.ssm_a                             # A 参数 [nHead]
blk.{L}.ssm_d                             # D 参数 [nHead]（可选）
blk.{L}.ssm_norm.weight                   # 分组 RMSNorm [dInner]
blk.{L}.ssm_out.weight                    # 输出投影 [dInner → hidden]

# 注意力层：
blk.{L}.attn_norm.weight
blk.{L}.attn_qkv.weight                   # 融合 Q+K+V（或分别为 attn_q/k/v.weight）
blk.{L}.attn_output.weight

# FFN 层（密集）：
blk.{L}.attn_norm.weight
blk.{L}.ffn_up.weight                     # Up 投影
blk.{L}.ffn_down.weight                   # Down 投影

# FFN 层（MoE）：
blk.{L}.attn_norm.weight
blk.{L}.ffn_gate_inp.weight               # 路由器 [numExperts, hidden]
blk.{L}.exp_probs_b.bias                  # 路由器偏置（可选）[numExperts]
blk.{L}.ffn_latent_in.weight              # 潜空间瓶颈输入（可选）
blk.{L}.ffn_latent_out.weight             # 潜空间瓶颈输出（可选）
blk.{L}.ffn_up_exps.{E}.weight            # 专家 up 投影
blk.{L}.ffn_down_exps.{E}.weight          # 专家 down 投影
blk.{L}.ffn_up_shexp.weight               # 共享专家 up（可选）
blk.{L}.ffn_down_shexp.weight             # 共享专家 down（可选）
```

### 前向传播流程（每 token）

```
tokens → Embedding
For each layer L:
  若 Mamba2:
    hidden → RMSNorm(attn_norm)
           → ssm_in 投影 → 分拆 z, xBC, dt
           → Conv1d(xBC) → SiLU → SSM 扫描(dt, A, B, C, state) → y
           → SiLU(z) * GroupRMSNorm(y) → ssm_out 投影
           → 残差相加
  若 注意力:
    hidden → RMSNorm(attn_norm)
           → QKV(融合) → 分拆 Q,K,V → 注意力(无 RoPE) → O 投影
           → 残差相加
  若 FFN（密集）:
    hidden → RMSNorm(attn_norm)
           → Up → ReLU² → Down
           → 残差相加
  若 FFN（MoE）:
    hidden → RMSNorm(attn_norm)
           → [latent_in] → 路由器(sigmoid+bias) → TopK → 归一化
           → 对每个专家: Up → ReLU² → Down → 加权求和
           → [latent_out] → [+ 共享专家] → 残差相加
hidden → RMSNorm(output_norm) → 截取最后 token → LM head → logits
```

### 缓存架构

- **注意力层**：标准 KV 缓存 `[numKVHeads, maxSeqLen, headDim]`。
- **Mamba2 层**：`_convState[layer]` 浮点数组，大小 `(convKernel-1) * (dInner + 2*nGroup*dState)`，用于 conv1d 滑动窗口；`_ssmState[layer]` 浮点数组，大小 `dState * headDim * nHead`，用于 SSM 状态。
- `ResetKVCache()` 同时将 KV 缓存和 SSM/卷积状态清零。
- `SupportsKVCacheTruncation` 返回 `false`，因为 SSM 状态是顺序的，无法部分复用。

### 预分配缓冲区

为避免热路径中的逐步分配，以下缓冲区一次性分配：

| 缓冲区 | 大小 | 用途 |
|---|---|---|
| `_mamba2ConvOutBuf` | dInner + 2*nGroup*dState | Conv1d 输出 + SiLU |
| `_mamba2YBuf` | dInner | SSM 扫描输出 |
| `_moeProbs` / `_moeSelectionProbs` | numExperts | 路由器概率 |
| `_moeTopExperts` / `_moeRouteW` | numExpertsUsed | 选中的专家及权重 |
| `_moeLatentAccum` | max(hiddenSize, latentDim) | 潜空间累加器 |
| `_expertUpResult` / `_expertDownResult` | 最大专家维度 | 可复用的专家 matmul 输出 |
| `_latentAccumTensor` / `_latentOutResult` | latentDim / hiddenSize | 潜空间瓶颈复用张量 |

### 批量 GPU MoE

在 GGML 后端（Metal/CUDA）上运行时，decode 阶段的 MoE 专家计算被批量合并为一次 `GgmlBasicOps.MoEExpertsForward()` 调用，在单次 GPU 图调度中处理所有选中专家，避免逐专家调度开销。预缓存的 `QuantizedWeight` 引用（`_expertUpQW`、`_expertDownQW`）和预分配的指针数组（`_moeUpPtrs`、`_moeDownPtrs`）消除了热循环中的字典查找和分配。

### 优化空间

- 注意力无 RoPE——位置信息由 SSM 状态隐式跟踪，简化了注意力实现但阻止了跨会话的 KV 缓存复用。
- Mamba2 SSM 在 prefill 期间逐 token 顺序处理。分块并行 SSM 扫描可提升提示吞吐量。
- Conv1d 实现为标量循环。SIMD 或原生向量化版本可提升 SSM 层性能。
- 无原生整模型 decode 路径。将前向传播移至单次原生调用（类似 Qwen 3）可减少托管开销。
- ReLU-squared 激活已 SIMD 向量化，但专家 FFN 仍按 token 顺序运行。跨专家批处理可带来提升。

---

## 架构对比

| 特性 | Gemma 3 | Gemma 4 | Qwen 3 | Qwen 3.5 | GPT OSS | Nemotron-H |
|---|---|---|---|---|---|---|
| 层类型 | 密集 | 密集 / MoE | 密集 | 混合（注意力 + 循环）| MoE | 混合（Mamba2 + 注意力 + MoE FFN）|
| 注意力 | SWA + 全局 | SWA + 全局 | 全 GQA | 全 GQA + 门控 | 全因果 + 沉降 | 全 GQA（无 RoPE）|
| FFN 激活 | GeGLU | GeGLU | SwiGLU | SwiGLU | SiLUAlphaLimit | ReLU² |
| RoPE 变体 | NeoX（双基数）| NeoX + 比例 | NeoX | NeoX / MRoPE | NeoX + Yarn | 无 |
| QK 归一化 | 有 | 有 | 有 | 有 | 无 | 无 |
| V 归一化 | 无 | 有（无权重）| 无 | 无 | 无 | 无 |
| 投影偏置 | 无 | 无 | 无 | 无 | 有（全部）| 无 |
| 逐层缩放 | 无 | 有 | 无 | 无 | 无 | 无 |
| PLE | 无 | 有 | 无 | 无 | 无 | 无 |
| KV 共享 | 无 | 有 | 无 | 无 | 无 | 无 |
| 注意力沉降 | 无 | 无 | 无 | 无 | 有 | 无 |
| 环形 KV 缓存 | 无 | 有（SWA 层）| 无 | 无 | 无 | 无 |
| SSM（Mamba2）| 无 | 无 | 无 | 无 | 无 | 有 |
| 共享专家 | 无 | 无 | 无 | 无 | 无 | 有（可选）|
| 潜空间瓶颈 | 无 | 无 | 无 | 无 | 无 | 有（可选）|
| 视觉 | 支持 | 支持 | 不支持 | 支持 | 不支持 | 不支持 |
| 音频 | 不支持 | 支持 | 不支持 | 不支持 | 不支持 | 不支持 |
| 视频 | 不支持 | 支持 | 不支持 | 不支持 | 不支持 | 不支持 |
| 思维链 | 不支持 | 支持 | 支持 | 支持 | 支持（始终开启）| 支持 |
| 工具调用 | 不支持 | 支持 | 支持 | 支持 | 不支持 | 支持 |
| 融合 QKV | 无 | 有 | 有 | 无 | 无 | 有 |
| 融合 GPU decode | 不支持 | 支持（Metal）| 不支持 | 不支持 | 不支持 | 不支持 |
| 原生模型 decode | 不支持 | 不支持 | 支持 | 不支持 | 不支持 | 不支持 |
| 批量 GPU MoE | 不支持 | 不支持 | 不支持 | 不支持 | 不支持 | 支持 |
| 输出解析器 | 直通 | Gemma4 | Qwen3 | Qwen35 | Harmony（始终开启）| Qwen3 |

---

## 添加新模型架构

1. 创建 `TensorSharp.Models/Models/<Name>/<Name>Model.cs`，继承 `ModelBase`。
2. 在构造函数中：通过 `_gguf.GetXxx()` 读取 GGUF 元数据，调用 `ParseBaseConfig()` 和 `ParseTokenizer()`，调用 `LoadWeights()`，然后融合权重并初始化缓存。
3. 实现 `Forward(int[] tokens) → float[]`：嵌入 → Transformer 块 → 归一化 → LM head → logits 复制。
4. 实现 `ResetKVCache()` 和 `Dispose()`。
5. 在 `ModelBase.Create()` 的 switch 表达式中注册（`ModelBase.cs`）。
6. 如果模型使用非标准输出格式，在 `OutputParser.cs` 中添加 `IOutputParser` 实现，在 `OutputParserFactory.Create()` 中注册。如果模型始终用结构化标签包裹输出，设置 `AlwaysRequired = true`。
7. 如果模型使用新颖的模板格式，在 `ChatTemplate.cs` / `Jinja2Template.cs` 中添加聊天模板支持。

