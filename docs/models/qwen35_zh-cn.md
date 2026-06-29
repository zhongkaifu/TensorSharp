# Qwen 3.5 / 3.6 family

[← 返回模型索引](README_zh-cn.md) | [English](qwen35.md)

| 属性 | 值 |
|---|---|
| 提供方 | 阿里巴巴 |
| GGUF 架构标识 | `qwen35`、`qwen35moe`、`qwen3next` |
| 模型类 | [`Qwen35Model`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.cs)（旧单序列路径）+ partial [`Qwen35Model.GatedDeltaNet.cs`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.GatedDeltaNet.cs) + [`Qwen35Model.BatchedForward.cs`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.BatchedForward.cs)（`IBatchedPagedModel`） |
| 视觉编码器 | [`Qwen35VisionEncoder`](../../TensorSharp.Models/Models/Qwen35/Qwen35VisionEncoder.cs) |
| 图像处理器 | [`Qwen35ImageProcessor`](../../TensorSharp.Models/Models/Qwen35/ImageProcessor.cs) |
| 示例模型 | Qwen3.5-9B（dense hybrid）、Qwen3.5-32B、Qwen3.5-35B-A3B / Qwen3.6-35B-A3B（MoE 系列） |
| 模态 | 文本、图像 |
| 思维链模式 | 是（`<think> ... </think>`） |
| 工具调用 | 是（`<tool_call>{...}</tool_call>`） |
| 批处理 / 分页前向 | **默认启用** —— 设置 `TS_QWEN35_BATCHED=0`（或 `--no-continuous-batching`）可强制走旧的按序列 KV-swap 路径用于 A/B 对比。带每槽位 GatedDeltaNet 递归状态池与可选的原生批处理 GDN 内核（`TS_QWEN35_BATCHED_GDN_NATIVE=1`）。详见 §11。 |
| MTP 投机解码 | Qwen 3.6 —— NextN 草稿块内嵌在主干 GGUF 中（无需独立文件）；用 `--mtp-spec` 启用。部分接受时做 GDN 递归状态快照 / 回滚。在 ggml 后端与纯 C# `cuda` 后端上有收益。详见 §12。 |
| 输出解析器 | `Qwen35OutputParser`（继承 `Qwen3OutputParser`） |

## 1. 来源与目标

Qwen 3.5 / 3.6 系是阿里第一个「混合」Qwen 系列，也是 TensorSharp 中优化最深的模型：

- **混合 attention + recurrent 堆栈。** 大多数层是 GatedDeltaNet（一种 linear-attention / SSM 风格的递归层）；少量层是 FullAttention 层，按 `full_attention_interval` 间隔均匀分布。Recurrent 层在 token 间维护形如 `[numVHeads, headVDim, headKDim]` 的隐状态；attention 层维护标准 KV cache。
- **MoE FFN** 出现在 `qwen35moe` / `qwen3next` 变体（如 Qwen3.5-35B-A3B、Qwen3.6-35B-A3B）。256 个 routed expert，每 token 激活 8 个，外加可选的 always-on shared expert。
- **Sigmoid-gated FullAttention。** Q 投影输出行数翻倍（`2 × numHeads × headDim`）；后半 deinterleave 后作为 sigmoid gate，在输出投影之前与 attention 输出相乘。
- **动态分辨率视觉。** 图像走 `Qwen35VisionEncoder`，一个支持任意分辨率的 2D-RoPE ViT，通过 patch embedding + spatial merger 完成 tokenization。

三种 GGUF 架构标识（`qwen35`、`qwen35moe`、`qwen3next`）共用同一个 `Qwen35Model`。MoE 变体额外启用 § 8 / § 9 描述的融合 MoE kernel。

## 2. 模型架构

```
                tokens (int[])
                      │
              token_embd.weight
                      │
        [可选] InjectVisionEmbeddings (image)
                      │
        ┌──── × NumLayers ──────────────────────────────┐
        │ if (l + 1) % fullAttentionInterval != 0:      │
        │     GatedDeltaNet(l)         (recurrent)      │
        │ else:                                         │
        │     FullAttention(l)         (gated GQA)      │
        │                                               │
        │ residual = block_input + branch_output        │
        │ post_attention_norm                            │
        │ if MoE 层:                                    │
        │     route ─► weighted Σ SwiGLU experts        │
        │     + 可选 shared expert                      │
        │     residual += moe_out                       │
        │ else:                                         │
        │     ffn_gate_up ─► SwiGLU ─► ffn_down         │
        │     residual += ffn_out                       │
        └───────────────────────────────────────────────┘
                      │
              RMSNorm(output_norm)
                      │
              LM head (output.weight)
                      │
                      ▼
                   logits
```

举例：48 层模型，`full_attention_interval == 4` 时，36 层 GatedDeltaNet + 12 层 FullAttention；FullAttention 层为第 4、8、12、... 层。

## 3. 前向计算图

### 3.1 FullAttention 层

```
hidden ─► RMSNorm(attn_norm)
       ─► attn_qkv.weight matmul
            输出行排列为 [Q (2 * qDim) ‖ K (kvDim) ‖ V (kvDim)]
       ─► 对 Q 做 deinterleave: 偶数列为 Q, 奇数列为 sigmoid gate
       ─► per-head RMSNorm(Q, attn_q_norm.weight)
       ─► per-head RMSNorm(K, attn_k_norm.weight)
       ─► RoPE_NeoX(Q, K) at positions startPos..startPos+seqLen-1
       ─► append (K, V) 到 KV cache
       ─► attention(Q, KCache, VCache)        // 标准因果 GQA
       ─► attn_out *= sigmoid(gate)
       ─► attn_output.weight matmul → o
       ─► residual = hidden + o
       ─► RMSNorm(post_attention_norm)
       ─► (FFN 块, 见 § 3.3)
```

### 3.2 GatedDeltaNet 层（recurrent）

```
hidden ─► RMSNorm(attn_norm)
       ─► attn_qkv (融合 Q+K+V) ─► QKV (沿时间拼接)
       ─► attn_gate ─► z (sigmoid gate)
       ─► ssm_beta ─► β (per-head)
       ─► ssm_alpha ─► α (per-head)
       ─► conv1d_step(QKV) using ssm_conv1d.weight (kernel size convK)
       ─► SiLU
       ─► 沿 head/feature 维拆为 Q, K, V
       ─► if (numKHeads != numVHeads): per-head 复制 Q 与 K
       ─► per-head L2 normalize Q, K
       ─► per-token recurrent update (per V-head):
              g = exp(α)              // forget gate ∈ (0, 1]
              s = sigmoid(β)          // input gate
              kv_mem = state @ k      // [headVDim, 1]
              state = g * state + s * (v - kv_mem) ⊗ kᵀ
              y = state @ q           // [headVDim, 1]
       ─► out = SiLU(z) * RMSNorm(y, ssm_norm.weight)
       ─► out matmul ssm_out.weight → o
       ─► residual = hidden + o
       ─► RMSNorm(post_attention_norm)
       ─► (FFN 块, 见 § 3.3)
```

每层的 recurrent 状态形如 `[numVHeads, headVDim, headKDim]`，存于 `_deltaStateTensor[layer]`。Conv 状态（每层 `(convKernel - 1) × qkvDim` floats）存于 `_convState[layer]`。

### 3.3 FFN 块（dense 或 MoE）

**Dense FFN**（当 `_isMoeLayer[l] == false`）：

```
hidden ─► RMSNorm(post_attention_norm)
       ─► ffn_gate_up ─► [gate ‖ up]
       ─► g = SiLU(gate)
       ─► ffn_down × (g * up) → o
       ─► residual += o
```

**MoE FFN**（`qwen35moe` / `qwen3next`）：

```
hidden ─► RMSNorm(post_attention_norm)
       ─► logits = ffn_gate_inp.weight × hidden
       ─► weights, idx = topK(softmax(logits), _numExpertsUsed)
       ─► if _normTopKProb: weights /= sum(weights)
       ─► moe_out = Σ_e weights[e] * (
              ffn_down_exps[idx[e]] × SiLUMul(
                  ffn_gate_up_exps[idx[e]] × hidden) )
       ─► if shared expert: moe_out += SiLUMul(
              ffn_gate_up_shexp × hidden) × ffn_down_shexp
       ─► residual += moe_out
```

GGML 后端上整个 MoE 块（routing、所有被选 expert、shared expert、residual add）合并为单次 `MoEExpertsSwiGLUResidual` kernel 调度（见 § 9）。

## 4. 组件细节

### 4.1 GatedDeltaNet（linear-attention 递归层）

GatedDeltaNet 是带 per-head recurrent 状态的 delta-net 线性 attention，更新公式：

```
state' = exp(α_t) · state + sigmoid(β_t) · (v_t − state · k_t) ⊗ k_tᵀ
y_t    = state' · q_t
```

其中：
- `state ∈ ℝ^{headVDim × headKDim}`，per V-head。
- `q_t, k_t ∈ ℝ^{headKDim}`，`v_t ∈ ℝ^{headVDim}`。
- `α_t, β_t ∈ ℝ` 是由 `ssm_alpha`、`ssm_beta` 线性投影产生的 per-head per-token gate。

rank-1 outer-product 更新通过 `Ops.AddmmBatch`（在 head 维 batched）完成；`y_t = state · q_t` 之后由 RMSNorm 加 `SiLU(z_t)` gate。`Ops.SiLUMul` 与 `Ops.RMSNorm` 完成这两步。

`_convKernel` 通常为 4，conv1d 滑动窗口在 `_convState[layer][]` 中以扁平 float buffer 维护。`Conv1dStep` 用 `ssm_conv1d.weight` 在窗口上做 conv 后接 SiLU，再做 QKV split。

### 4.2 带 Sigmoid gate 的 FullAttention

`attn_qkv.weight` 行排列让单次 matmul 同时输出：

- `2 × qDim` 行，按列交错排列 Q 与 gate。
- `kvDim` 行 K。
- `kvDim` 行 V。

deinterleave 步骤（`SplitQGateInterleaved`）把 Q 写入一个 buffer、gate 写入另一个 buffer。Q、K 跑 per-head RMSNorm，RoPE 旋转 Q、K，标准因果 attention 照常进行。在输出投影之前，`Ops.SigmoidMul` 原地完成 `attn_out *= sigmoid(gate)`。

### 4.3 Mixture-of-Experts FFN

| 配置 | 来源 |
|---|---|
| Routed expert 数 `_numExperts` | `qwen35.expert_count`（256） |
| 每 token 激活专家数 `_numExpertsUsed` | `qwen35.expert_used_count`（8） |
| Routed expert FFN 宽度 `_expertFfnLength` | `qwen35.expert_feed_forward_length` |
| Shared expert FFN 宽度 `_sharedExpertFfnLength` | `qwen35.expert_shared_feed_forward_length` |
| 是否归一 TopK 权重 | `qwen35.expert_weights_norm`（Qwen 3.5+ 强制 true） |

路由：`linear(hidden) → softmax → topK(_numExpertsUsed)`。`_normTopKProb` 为 true 时被选权重重新归一为和 1。Shared expert（始终激活，不经过路由）可选与之并行相加。

### 4.4 RoPE

NeoX 风格旋转嵌入。频率在 `_ropeFreqs[halfDim]` 一次性预计算。位置张量在每个 forward pass 内跨 attention 层缓存：当 `(seqLen, startPos)` 在所有 12 个 attention 层之间一致时，复用 `_cachedRoPEPosQ` / `_cachedRoPEPosK`，省去重复分配 + 填充。

`qwen3next` GGUF 额外提供 `qwen35.rope.dimension_sections`，定义 MRoPE section 边界。存在时 RoPE 步骤使用多模态 RoPE；不存在时使用纯 NeoX RoPE。

### 4.5 视觉编码器（`Qwen35VisionEncoder`）

SigLIP 风格 ViT：

- conv 风格 patch embedding，实现为并行 im2col + matmul（替代了原来的五重嵌套标量循环）。
- 2D RoPE 位置编码。
- LayerNorm + 多头 attention + residual + LayerNorm + GELU MLP + residual block。
- Spatial patch merging（`SpatialMergeSize × SpatialMergeSize` 邻居合并为一个 token）。
- 多模态 projector（RMSNorm + Linear + GELU + Linear）映射到语言模型 hidden dim。

GGML 后端上每个 block 走两个融合 kernel（`FusedVisionAttention` 与 `FusedVisionMLP`），见 § 8。

## 5. 参数与配置

| Key | 类型 | 含义 |
|---|---|---|
| `qwen35.ssm.inner_size` | uint32 | `headVDim * numVHeads` |
| `qwen35.ssm.state_size` | uint32 | `headKDim` |
| `qwen35.ssm.group_count` | uint32 | `numKHeads` |
| `qwen35.ssm.time_step_rank` | uint32 | `numVHeads` |
| `qwen35.ssm.conv_kernel` | uint32 | Conv1d kernel size |
| `qwen35.full_attention_interval` | uint32 | 每 N 层一个 full attention（默认 4） |
| `qwen35.rope.dimension_sections` | int32[] | MRoPE section 边界（qwen3next） |
| `qwen35.rope.dimension_count` | uint32 | RoPE dim 数 |
| `qwen35.expert_count` | uint32 | Routed expert 数（0 表示 dense FFN） |
| `qwen35.expert_used_count` | uint32 | 每 token 激活的 routed expert 数 |
| `qwen35.expert_feed_forward_length` | uint32 | Routed expert FFN 宽度 |
| `qwen35.expert_shared_feed_forward_length` | uint32 | Shared expert FFN 宽度（0 表示无 shared expert） |
| `qwen35.expert_weights_norm` | bool | 是否将 TopK 权重重新归一为和 1 |

## 6. 权重命名约定

```
# FullAttention 层（每 N 层一个）：
blk.{L}.attn_norm.weight
blk.{L}.attn_q.weight                      # [numHeads*headDim*2, hidden]  (Q+gate 交错)
blk.{L}.attn_k.weight                      # [numKVHeads*headDim, hidden]
blk.{L}.attn_v.weight                      # [numKVHeads*headDim, hidden]
blk.{L}.attn_q_norm.weight
blk.{L}.attn_k_norm.weight
blk.{L}.attn_output.weight
blk.{L}.post_attention_norm.weight

# GatedDeltaNet 层：
blk.{L}.attn_norm.weight
blk.{L}.attn_qkv.weight                    # [qkDim*2 + vDim, hidden]
blk.{L}.attn_gate.weight                   # z gate [ssmDInner, hidden]
blk.{L}.ssm_beta.weight                    # [numVHeads, hidden]
blk.{L}.ssm_alpha.weight                   # [numVHeads, hidden]
blk.{L}.ssm_conv1d.weight                  # [qkvDim, convKernel]
blk.{L}.ssm_dt.bias                        # dt bias [numVHeads]
blk.{L}.ssm_a                              # a 参数 [numVHeads]
blk.{L}.ssm_norm.weight                    # 输出 RMSNorm
blk.{L}.ssm_out.weight                     # 输出投影
blk.{L}.post_attention_norm.weight

# Dense FFN（当 ffn_gate_inp.weight 缺失时）：
blk.{L}.ffn_gate_up.weight
blk.{L}.ffn_down.weight

# MoE FFN（qwen35moe / qwen3next）：
blk.{L}.ffn_gate_inp.weight                # router [numExperts, hidden]
blk.{L}.ffn_gate_up_exps.{E}.weight        # 融合的 expert gate+up
blk.{L}.ffn_down_exps.{E}.weight           # expert down
blk.{L}.ffn_gate_up_shexp.weight           # shared expert gate+up（可选）
blk.{L}.ffn_down_shexp.weight              # shared expert down（可选）
blk.{L}.ffn_gate_inp_shexp.weight          # shared expert gate（可选）

# 标准末尾权重：
output_norm.weight
output.weight
```

`FuseAttentionProjectionWeights()` 把 FullAttention 层的 Q+K+V 融合为 `attn_qkv.weight`（如 GGUF 分散存放）。`FuseRecurrentInputWeights()` 把 GatedDeltaNet 的五路递归投影（`attn_qkv`、`attn_gate`、`ssm_beta`、`ssm_alpha`、加一个空槽）融合为 `ssm_in_proj.weight`。`FuseGateUpWeights()` 融合 dense FFN 的 gate 与 up。

## 7. TensorSharp 实现走读

构造函数：

1. `ParseBaseConfig()`。
2. `ParseGdnConfig(arch)` 读取 SSM 维度（`_ssmDInner`、`_ssmDState`、`_ssmNGroup`、`_ssmDtRank`、`_convKernel`）。
3. `_fullAttentionInterval` 与每层 `_isRecurrent[]`。
4. `_mropeSections`、`_ropeDimCount`。
5. MoE 配置（`_numExperts`、`_numExpertsUsed`、`_expertFfnLength`、`_sharedExpertFfnLength`）。
6. `ParseTokenizer()`、`LoadWeights()`。
7. `FuseAttentionProjectionWeights()`、`FuseRecurrentInputWeights()`、`FuseGateUpWeights()`。
8. `DetectMoeLayers()` 遍历每个 block，设置 `_isMoeLayer[l]`、`_hasSharedExperts[l]`、`_hasSharedExpertGate[l]`。
9. `BuildLayerKeys()` 预计算热路径用到的全部权重名（`attn_norm`、`attn_qkv`、`attn_q_norm`、`attn_k_norm`、`attn_output`、`ffn_gate_up`、`ffn_down`、`ffn_gate_inp`、`ffn_gate_up_exps[e]`、`ffn_down_exps[e]`...）。
10. `InitMoeBuffers()` 预分配 router / expert 指针数组、MoE 工作 tensor、批量 MoE kernel 用的 `IntPtr[]`。
11. `PrepareCudaQuantizedWeightsForInference()`。
12. `InitCaches(initialCacheLength, maxContextLength)` 为 FullAttention 层分配 KV cache，对 recurrent 状态清零。
13. `PrecomputeRoPE()`、`InitGDNBuffers()`、`CacheRecurrentWeights()`。

`Forward(int[] tokens)` 区分 prefill（`seqLen > 1`）与 decode（`seqLen == 1`），把每层路由到下列任一路径：

- 融合 per-layer attention decode（`TryFusedAttnLayerDecode`），仅当当前已缓存序列长度超过 `FUSED_ATTN_LAYER_MIN_SEQ_LEN` 阈值（默认 4096）时启用。
- 融合 prefill attention（`FusedPrefillAttention`），用于 GGML 后端上的多 token prefill。
- 融合输出投影 + FFN（`FusedOutProjFFN`），用于 attention 与 recurrent 层带 dense FFN 的情况。
- 融合输出投影 + post-norm + router（`FusedOutProjNormRouter`），用于 recurrent 层接入 MoE。
- 批量 MoE（`MoEExpertsSwiGLUResidual`），用于 MoE FFN 的 decode。
- 否则走标准托管 C# 路径。

## 8. Prefill 优化

### 融合 prefill attention（`FusedPrefillAttention`）

单张 GGML 图完成：

1. `Q × K^T`（带 scale），处理 GQA head 广播。
2. 因果掩码（下三角，连续 prefill 时带 `maskStartPos` 偏移）。
3. Softmax。
4. Scores × V → attention 输出。

替代每 attention 层 ~5 次 C# 到 GGML 往返。同时支持初始 prefill（KV cache 为空）与连续 prefill（已存有上一轮 KV）。`inputFormat` 参数支持 `[seqLen, numHeads, headDim]` 与 `[numHeads, seqLen, headDim]` 两种布局。

### 融合输出投影 + FFN（`FusedOutProjFFN`）

对于 FullAttention 与 GatedDeltaNet 中带 dense FFN 的层，单张 GGML 图完成：output 投影 + residual add + post-attention RMSNorm + ffn_gate_up matmul + SiLU + ffn_down matmul + residual。两次 GPU 往返合并为一次。可用 `QWEN35_DISABLE_FUSED_FFN=1` A/B 关闭。

### 并行化 Q / gate deinterleave

FullAttention prefill 中 Q + sigmoid-gate 的 deinterleave 走 `Parallel.For` 跨 token 并行。每个 token 的 deinterleave 互不依赖，因此长 prompt 下随 CPU 核数线性扩展。

### 原地 QK RMSNorm

`ApplyPerHeadRMSNorm()` 通过 `View(seqLen * numHeads, headDim)` 复用同一片内存做 RMSNorm。RMSNorm 对行序无关，所以这种 reshape 对 Q、K 都安全。每层每个 Q/K 省一次 tensor 分配 + `Ops.CopyTo`。

### RoPE 位置张量缓存

`_cachedRoPEPosQ` / `_cachedRoPEPosK` 仅在 `(seqLen, startPos)` 变化时重建，跨所有 attention 层复用。

### 基于 GEMM 的视觉 patch embedding

`Qwen35VisionEncoder` 把 patch embedding 重写为并行 im2col + matmul，把单线程标量五重嵌套循环替换为 GPU 加速 matmul。配合融合视觉 block 显著加速多 tile 图像路径。

### 融合视觉编码器 block

- **`FusedVisionAttention`**：LayerNorm + QKV 投影 + bias + 2D RoPE + scaled dot-product attention + 输出投影 + bias + residual 合并为一次 GGML 图调度（~8 op → 1）。
- **`FusedVisionMLP`**：LayerNorm + up 投影 + bias + GELU + down 投影 + bias + residual 合并为一次 GGML 图调度（7 op → 1）。

合计每个视觉 block 从 ~15 次 GPU 往返降到 2 次。

### 跨层缓存与并行

- spatial merge 中 `ReorderToBlockOrder` 对 block 行做 `Parallel.For`。
- GPU 友好 QKV split：GPU 后端用 `Narrow` + `NewContiguous` 让数据保持在设备上；CPU 用并行 `Buffer.MemoryCopy` 一次扫完成 Q/K/V 拆分。

### 分块并行 GatedDeltaNet recurrent prefill

`seqLen ≥ 64` 时，GGML 后端上的 per-token 递归循环会被一次融合的分块 SSM 扫描替代（C# 端 `GatedDeltaNetChunkedPrefill` → `GgmlBasicOps.GatedDeltaNetChunked`，原生端 `ggml_ops_gated_delta_net.cpp` 中的 `TSGgml_GatedDeltaNetChunkedF32`）。整体思路与 Mamba 的 parallel-scan kernel 相同：

1. **CPU 准备阶段（按 token 并行）**：每通道 1D 卷积 + SiLU + Q/K/V/Z/α/β 打包融合在同一次 `Parallel.ForEach` 里，每个工作线程持有自己的 SiLU scratch（`ApplySiLUInPlaceScratch`）。同一次扫描里更新 conv 环状状态的最后 `convKernel - 1` 行，保证下一层看到正确的 conv state。
2. **CPU 端预先计算 gate / β-sigmoid**：极小的 `[seqLen, H]` 张量上的 `softplus(α + dt_bias)·a_log` 与 `sigmoid(β)` 直接走 `TensorPrimitives.Sigmoid` 在 CPU 完成。本来在 GPU 上跑的 4 个 op 因此被吃掉，省去两个 per-layer 常量上传。
3. **一次融合 GGML 图调度（`GatedDeltaNetChunked`）**：整个 delta-net 块（Q/K L2-norm、scale、β 乘、分块三角解 `(I − decay·k·kᵀ)⁻¹·decay·k·kᵀ` attention、跨 chunk 递归状态传播、RMSNorm、`silu(z)` 门控）跑在一张 Metal/CUDA 图上。该图按 `(T, H, D, chunkSize, eps)` 维度只构建一次，缓存在 `g_gdn_chunked_cache`；后续调用只用 `ggml_backend_tensor_set` 重新绑定输入。`ggml_solve_tri` 在 head + chunk 维度并行求解每个 chunk 的 `(I − attn_lower)·X = attn_init`。
4. **跨 chunk 仍然顺序**：第 `c+1` 个 chunk 的递归状态依赖第 `c` 个 chunk 的输出，但循环长度只有 `nC = ceil(T/64)`，串行深度从 `O(T)` 降到 `O(T/64)`，与 Mamba parallel-scan kernel 的渐进 span 一致。

可调环境变量：

- `GDN_CHUNK_PREFILL_MIN_SEQ_LEN=N` 覆盖阈值（默认：ggml_cuda 上为 2，其他后端为 6）。在 CUDA 上逐 token 路径每层的主机/设备同步开销远大于 64 padding 浪费，chunked 内核对所有多 token 批次都更快；在 Qwen3.6-27B IQ2_XXS 上实测 MTP 投机验证解码从 217 ms/token 降到 174 ms/token。单 token 解码仍走逐 token 路径。设为 `64` 恢复旧的仅长 prefill 行为，设为很大值禁用。
- `GDN_DISABLE_CHUNKED_PREFILL=1` 强制走 per-token CPU 循环；用于 A/B 对比。
- `GDN_VERIFY_CHUNKED=1` 在递归状态快照上跑 chunked 路径，恢复快照后再跑 per-token 路径，统计两条路径在 gated 输出与递归状态上的最大绝对 / 相对差。下游使用 per-token 结果，所以即使 chunked kernel 有 bug 也不会污染后续层。`PrintGdnTimingStats` 会打印验证调用次数与最大差值。

在 Apple Metal（M 系列）上、KV cache f16、每个配置 3 次 run，对 Qwen3.6-35B-A3B（UD-IQ2_XXS，30 个 GDN 层 + MoE FFN）实测：

| prefill tokens | chunked ms / call | per-token ms / call | GDN 块加速 | 总 prefill 加速 | 采样 token 一致 |
|---:|---:|---:|---:|---:|:---:|
|  128 |   3.77 |   7.32 | **1.94×** | 1.03× | ✓ |
|  512 |  12.91 |  29.98 | **2.32×** | 1.09× | ✓ |
| 2048 |  49.66 | 118.88 | **2.39×** | 1.11× | ✓ |

`GDN_VERIFY_CHUNKED=1` 在三个 prefill 长度上分别报告 max output `|Δ|` ≈ 3 × 10⁻³、max state `|Δ|` ≈ 1 × 10⁻²，warnings 全部为 0（warn 阈值是绝对 5 × 10⁻³ / 相对 5 × 10⁻²）。这是跨 30 层、对 256–2048 个 token 步乘加结果在不同求和顺序下（Metal MM 累加 vs 标量 CPU）的 FP32 噪声底，属正常范围。所有长度下两条路径的最高概率 token 完全一致。

## 9. Decode 优化

### 融合 per-layer attention decode（`Qwen35AttentionLayerDecode`）

单张 GGML 图完成整个 FullAttention 块：

1. RMSNorm(input) * `attn_norm.weight`。
2. 融合 QKV matmul → `[Q+gate (2*qDim), K (kvDim), V (kvDim)]`。
3. strided view + 连续拷贝完成 Q、gate、K、V 拆分。
4. per-head RMSNorm(Q) * `attn_q_norm.weight`，per-head RMSNorm(K) * `attn_k_norm.weight`。
5. 当前 `position` 上 NeoX 风格 RoPE 旋转 Q、K。
6. 通过 `ggml_cpy` view 把新 K、V 写回持久化 KV cache。
7. `ggml_flash_attn_ext` 在已填充 KV cache 窗口上跑（处理 GQA 广播）。
8. `attn_out *= sigmoid(gate)`。
9. 输出投影 + residual add → 通过 host 指针写回更新后的 hidden state。

替代了 1 次独立 `FusedRmsNormMatMulQuant` + ~6 次 CPU 端小算子 + 1 次独立 `FusedMatMulQuantAdd`，合并为一次融合调度。kernel 仅在 `position + 1 >= FusedAttnLayerDecodeMinSeqLen`（默认 4096，可通过 `FUSED_ATTN_LAYER_MIN_SEQ_LEN=N` 覆盖）时启用 —— GPU flash-attn 路径有固定启动开销，只有长上下文才能摊平。阈值以下保留原有 `FusedRmsNormMatMulQuant` + CPU-SIMD attention + `FusedMatMulQuantAdd` 路径。

### 融合输出投影 + norm + router（MoE recurrent decode）

GatedDeltaNet 层后接 MoE FFN 的 decode 中，`FusedOutProjNormRouter` 把 GDN 输出投影、residual add、post-attention RMSNorm 与 MoE router 投影合并为一次 GGML 图调度（3 op → 1）。预先算好的 router logits 直接喂给 `TryMoEResidualDecodeWithRouter()`，每层 MoE 不再单独发 router。

### 批量 GPU MoE（`MoEExpertsSwiGLUResidual`）

GGML 后端上 `qwen35moe` / `qwen3next` 的 decode 中，每层 MoE expert 计算合并为一次 `MoEExpertsSwiGLUResidual()`：

1. 全部被选 routed expert：`SwiGLU(hidden × gate_up_exps[e]) × down_exps[e]`，按 router 概率加权。
2. 可选 shared expert：`SwiGLU(hidden × gate_up_shexp) × down_shexp`。
3. 把 residual 加回流动 hidden state。

预缓存的 `QuantizedWeight` 引用与 `IntPtr[]` 数组（`_routedExpertGateUpQW`、`_routedExpertDownQW`、`_sharedExpertGateUpQW`、`_sharedExpertDownQW`，以及对应的 `*Ptrs` 数组）让热 decode 循环没有字典查询、没有分配。

### 预分配的 GDN decode 缓冲

在 `InitGDNBuffers()` 一次性分配：

| Buffer | 形状 | 用途 |
|---|---|---|
| `_gdnConvOutT` | `[1, qkvDim]` | Conv1d 输出 + SiLU |
| `_gdnKBuf` / `_gdnQBuf` | `[numVHeads, headKDim]` | 扩展 + L2 norm 后的 Q/K |
| `_gdnVBuf` | `[numVHeads, headVDim]` | V split |
| `_gdnKvMemBuf` | `[numVHeads, headVDim, 1]` | `state @ k` 中间结果 |
| `_gdnCoreOutBuf` | `[numVHeads, headVDim, 1]` | `state @ q` 输出 |
| `_gdnGatedOutT` | `[1, ssmDInner]` | 最终 gated 输出 |

### 预分配的 FullAttention decode 缓冲

| Buffer | 形状 | 用途 |
|---|---|---|
| `_attnDecodeQBuf` | `[1, numHeads * headDim]` | deinterleave 出来的 Q |
| `_attnDecodeGBuf` | `[1, numHeads * headDim]` | deinterleave 出来的 gate |
| `_attnDecodeOutBuf` | `[1, numHeads * headDim]` | attention 输出 |
| `_attnDecodeQkvBuf` | `[1, qFullDim + 2*kvDim]` | 复用的融合 QKV 输出 |
| `_ffnDecodeGateUpBuf` | `[1, 2 * intermediateSize]` | 复用的融合 gate+up 输出 |

## 10. 内存与 KV cache 策略

- **FullAttention 层**：标准 KV cache `[numKVHeads, maxSeqLen, headDim]` per layer。KV dtype 通过 `--kv-cache-dtype` 选 `f32` / `f16` / `q8_0`。
- **GatedDeltaNet 层**：`_convState[layer]` 是 `(convKernel - 1) * qkvDim` 的 float 数组（conv1d 滑动窗口），`_deltaStateTensor[layer]` 形如 `[numVHeads, headVDim, headKDim]`（SSM 隐状态）。
- `ResetKVCache()` 同时清零两类缓存。
- 初始 CUDA 缓存可以小于 `maxContextLength`，按需扩张（启动时打印）。

### mmap 量化权重

后端支持时（direct CUDA、GGML CUDA、Apple Silicon Metal、GGML CPU、集成显卡），加载器避免把量化 tensor 拷贝到新分配的 native 堆缓冲，而是通过 `MemoryMappedFile` + `QuantizedWeight.CreateExternalView` 直接绑定 GGUF 文件。结合 GGML 的 host-pointer buffer mapping，让 Metal command buffer 直接读 OS page cache 中的权重。`Qwen3.5-35B-A3B-IQ2_XXS`（~10 GB GGUF）在 M 系 Mac 上的常驻内存从 ~17 GB 降到 ~7 GB，且没有任何 per-token 拷贝。

## 11. 批处理 / 分页前向（连续批处理）

Qwen 3.5 / 3.6 实现了 `IBatchedPagedModel.ForwardBatch`
（[`Qwen35Model.BatchedForward.cs`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.BatchedForward.cs)）
并默认启用 —— 该路径支持 Qwen3.5 全部三种层类型（attention、GDN 递归、MoE），
也是连续批处理多请求并发所必需的。通过 `TS_QWEN35_BATCHED=0`（或向服务传入
`--no-continuous-batching`）可强制回到旧的按序列 KV-swap 路径用于 A/B 对比
或回归排查。

Qwen 3.5/3.6 是混合架构（FullAttention + GatedDeltaNet 递归层），批处理移
植需要同时管理**两种正交的缓存** —— 注意力层的分页 K/V，以及 GatedDeltaNet
层独立的**每槽位递归状态池**：

### FullAttention 层 —— 分页 K/V

- 每层分页缓冲，布局 `[numBlocks * blockSize * numKvHeads * headDim]`，
  按需扩容。
- 逐 token NeoX RoPE 通过 `Ops.RoPEEx` 派发，使用显式 positions 数组。
- 每头 QK RMSNorm 通过 `View` 原地执行（与旧路径一致）。
- Q 与 sigmoid gate 从融合的 `[2 × qDim ‖ kDim ‖ vDim]` Q 投影中按 token
  反交错（与旧 forward 同样的布局，在批处理循环中逐 token 解开）。
- 基于 `slotMapping` 的 K/V 写入到每层的分页缓冲。
- 通过 `GgmlBasicOps.PagedAttentionForward`（驱动 `ggml_flash_attn_ext` 的
  原生分页内核）按序列计算注意力。

### GatedDeltaNet 层 —— 每槽位递归状态池

递归状态是逐序列的，并在整个输入上演进。Phase 5c 引入了一个以每个序列的
**主块 id**（与 vLLM 的 `state_indices_tensor` 对齐）为 key 的每槽位状态池：

- `_q35GdnSlotConvBuf[layer][slot]` —— 每槽位 conv1d 环形缓冲
  （`float[(convKernel - 1) * qkvDim]`）。
- `_q35GdnSlotConvWriteIdx[layer][slot]` —— 每槽位环形写指针。
- `_q35GdnSlotSsmTensor[layer][slot]` —— 每槽位 SSM 状态 Tensor，形如
  `[numVHeads, headVDim, headKDim]`。
- `_q35GdnSlotInit[layer][slot]` —— 初始化标志。

槽位在首次访问时按需分配，引擎回收序列时释放。相比 Phase 2 的方式（每层
两次复制状态到 / 从 per-model scratch buffer），这种方式在每次 decode 步骤
上**每个 GDN 层每个序列**节省了大约 **2 MB SSM tensor memcpy 加几十 KB
conv memcpy**。

**原生批处理 GatedDeltaNet 内核** —— `TSGgml_GatedDeltaNetBatchedStepF32`
（[`ggml_ops_gated_delta_net.cpp`](../../TensorSharp.GGML.Native/ggml_ops_gated_delta_net.cpp)）
—— 通过 `TS_QWEN35_BATCHED_GDN_NATIVE=1` 控制。启用时
`GgmlBasicOps.GatedDeltaNetBatchedStep` 把托管的 per-token GDN 步替换为一
次原生派发，并并行更新所有 in-flight 序列的 conv + SSM 状态；关闭时（默认）
批处理 forward 通过
[`Qwen35Model.GatedDeltaNet.cs`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.GatedDeltaNet.cs)
的托管参考路径调用。

### 批处理路径下的多模态

`SupportsBatchedMultimodal` 在批处理路径处于启用状态时（即未设置 `TS_QWEN35_BATCHED=0`）返回 true。
`ForwardBatch` 会按 batch 构建一张全局 MRoPE 位置表：每个序列的 MRoPE
positions 从多模态注入器中取出，按批中全局偏移注入到批处理 hidden tensor
中，并传给每层的 RoPE 调用。视觉嵌入按 per-layer 循环之前按行注入，与旧
forward 一致。

### 已验证的正确性与吞吐

- 短 prompt 上与旧路径 100% 贪心一致
  （[`Qwen35BatchedCorrectnessTests`](../../InferenceWeb.Tests/Qwen35BatchedCorrectnessTests.cs)）。
- 在 Qwen 3.6-27B（Apple M4 Pro、GgmlMetal、进程内 legacy-vs-batched 切换）
  上 **n=3 时达到 ~1.83× tps**。

## 12. MTP / NextN 投机解码（Qwen 3.6）

Qwen 3.6 的 GGUF 自带一个 **NextN / 多 token 预测（MTP）草稿块**，`TensorSharp.Server`
用它为单序列（无并发）请求做无损投机解码。源码：
[`Qwen35Model.Mtp.cs`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.Mtp.cs)，
由共享的
[`MtpSpeculativeExecution`](../../TensorSharp.Runtime/Scheduling/MtpSpeculativeExecution.cs)
起草 / 验证 / 回滚核心驱动。与 Gemma 4 不同，这里无需独立草稿 GGUF——草稿块就内嵌在主干
文件中，因此 `--mtp-draft-model` 被忽略。

### 12.1 内嵌 NextN 块

Qwen 3.6 GGUF 在主干栈之后带有一个额外解码块（`blk.N`，`N` == 主干层数），由
`{arch}.nextn_predict_layers` 标记。它是一个标准的 Qwen 3.5 全注意力解码块（27B 上是
dense FFN，35B-A3B 上是 MoE FFN），外加四个 NextN 专属张量：

```
nextn.eh_proj          [2*hidden, hidden]  对 concat(embedding, hidden) 的输入投影
nextn.enorm            [hidden]            对 token embedding 的 RMS norm
nextn.hnorm            [hidden]            对主干 hidden state 的 RMS norm
nextn.shared_head_norm [hidden]            LM 头前的最终 norm
```

（`nextn.embed_tokens` / `nextn.shared_head_head` 可选，在标准 GGUF 中缺失，代码会回退到
主干的 token embedding / LM 头。）MTP 步在位置 `p` 消费（token `x_p`、主干 hidden
`h_{p-1}`），产出预测 `x_{p+1}` 的 logits 以及自己的 hidden state 用于串接后续草稿步——对应
llama.cpp 的 `graph_mtp` 与 vLLM 的 `Qwen3_5MultiTokenPredictor`。

### 12.2 递归状态回滚

由于 Qwen 3.6 主干混合了 GatedDeltaNet 递归层与全注意力层，部分被拒的验证批次不能简单地
截断 KV 缓存：GDN 递归状态无法就地回退。MTP 路径在每个验证批次前对 GDN 状态做快照
（`_mtpGdnSnapshot`），部分接受时恢复。在 CUDA 后端上快照在设备侧完成，避免宿主端往返。

### 12.3 收益与调优

投机默认关闭，用 `--mtp-spec`（环境变量 `TS_MTP_SPEC=1`）启用。它仅在有收益处启用——ggml
后端与纯 C# `cuda` 后端——否则引擎走标准 decode。`--mtp-draft`（默认 `8`）限制草稿窗口，
`--mtp-pmin`（默认 `0.75`）是保留 token 所需的最低草稿置信度。在 `ggml_cuda` 上，GDN 分块
prefill 内核也会加速投机验证：在 Qwen3.6-27B IQ2_XXS 上实测把 MTP 投机验证 decode 从
217 ms/token 降到 174 ms/token（见 §8，`GDN_CHUNK_PREFILL_MIN_SEQ_LEN`）。完整参数列表见
[README MTP 章节](../../README_zh-cn.md#mtp--nextn-投机解码)。

## 13. 输出解析器与聊天模板

- `Qwen35OutputParser` 继承 `Qwen3OutputParser`，wire 格式相同：`<think> ... </think>` 表示思维链，`<tool_call>{"name": "...", "arguments": {...}}</tool_call>` 表示工具调用。
- 聊天模板使用标准 Qwen `<|im_start|>` / `<|im_end|>` 格式，外加视觉占位符 `<|image_pad|>`。
- `ChatTemplate.ExpandImageTokens(inputTokens, imagePadId, tokenCounts)` 把每个 `<|image_pad|>` 占位符展开为对应图像所需 token 数；多模态注入器随后在这些位置写入编码后的 embedding，再调用 `Forward()`。

## 14. 优化机会

- **原生 GDN decode（旧路径）** —— 旧的单序列 GDN decode 当前仍在托管 C#
  中跑（带预分配缓冲与 `Ops.AddmmBatch`）。把 per-token recurrent 更新放
  进原生 C / CUDA 能消除该路径上剩余的托管开销。
- **向量化 conv1d** —— `Conv1dStep` 是标量循环。SIMD 或原生向量化版本能给
  decode 热路径带来几个百分点的提升。
- **MoE prefill 批处理** —— MoE prefill 当前逐 token 迭代。批处理的 expert
  prefill kernel（类比 decode 路径）能进一步加速 MoE 变体的长 prompt。
- **把原生批处理 GDN 提升为默认**。`TS_QWEN35_BATCHED_GDN_NATIVE=1` 控制
  的内核已经存在，但目前因为性能验证而被 opt-in。一旦 n=1 的回退闭合，它
  会成为批处理路径上的默认 GDN 派发。
