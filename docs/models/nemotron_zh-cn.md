# Nemotron-H

[← 返回模型索引](README_zh-cn.md) | [English](nemotron.md)

| 属性 | 值 |
|---|---|
| 提供方 | NVIDIA |
| GGUF 架构标识 | `nemotron_h`、`nemotron_h_moe` |
| 模型类 | [`NemotronModel`](../../TensorSharp.Models/Models/Nemotron/NemotronModel.cs) |
| 视觉编码器 | [`NemotronVisionEncoder`](../../TensorSharp.Models/Models/Nemotron/NemotronVisionEncoder.cs)（RADIO / v2_vl ViT） |
| 图像处理器 | [`NemotronImageProcessor`](../../TensorSharp.Models/Models/Nemotron/NemotronImageProcessor.cs) |
| 音频前端 | [`NemotronAudioPreprocessor`](../../TensorSharp.Models/Models/Nemotron/NemotronAudioPreprocessor.cs)（Parakeet 风格 log-mel） |
| 示例模型 | Nemotron-H-8B-Reasoning-128K、Nemotron-H-47B-Reasoning-128K、Nemotron 3 Nano Omni |
| 模态 | 文本、图像（Omni 版本配合 `mmproj`）。音频已经被预处理用于 Omni 发布版本，但推理需要一个尚未随这些 GGUF 一起发布的 Parakeet `mmproj`。 |
| 思维链模式 | 是（`<think> ... </think>`） |
| 工具调用 | 是（`<tool_call>{...}</tool_call>`） |
| 输出解析器 | `Qwen3OutputParser` |

## 1. 来源与目标

Nemotron-H 是 NVIDIA 的混合 **Mamba2 + Transformer** 系列。同一套 backbone 同时覆盖密集 `nemotron_h` 系（如 Nemotron-H-8B / 47B）与 MoE `nemotron_h_moe` 系。Omni 发布（Nemotron 3 Nano Omni）额外携带 RADIO / v2_vl 视觉编码器与 Parakeet 音频前端。

它的核心特征：

- **同一个堆栈中三种层类型** —— Mamba2 SSM 层、纯注意力层、纯 FFN 层（可选 MoE）。每层类型由 GGUF 元数据数组决定：`head_count_kv[l]` 与 `feed_forward_length[l]` 一起决定层类型。
- **Mamba2 SSM** —— 选择性状态空间模型，分组 head。conv 状态与 SSM 状态按层、按 token 维护。
- **没有 RoPE** —— 注意力层不带位置编码。位置信息由 SSM 状态隐式承载。
- **ReLU² FFN** 激活（`max(0, x)²`）。
- **Sigmoid 路由 MoE** —— 用 per-expert sigmoid 概率（可选加性 bias）做 expert 选择，TopK，可选归一化 / 缩放。
- **可选 latent bottleneck 与 shared expert** —— MoE FFN 可有 in / out latent bottleneck（`ffn_latent_in` / `ffn_latent_out`）和并行的 SwiGLU 共享分支（`ffn_up_shexp` / `ffn_down_shexp`）。
- **Decode CPU 卸载阈值** —— 小 decode 算子（RMSNorm、residual add、小 matmul）可以放在 CPU 跑以避开 per-dispatch GPU 开销；大 matmul（SSM in/out、attention QKV/output、LM head）留在 GPU。

## 2. 模型架构

```
                tokens (int[])
                      │
              token_embd.weight
                      │
        [可选] InjectVisionEmbeddings (image)
                      │
        ┌──── × NumLayers ───────────────────────────────┐
        │ 根据 (head_count_kv[l], feed_forward_length[l])│
        │ 选择层类型:                                    │
        │     • Mamba2:    kv == 0  AND  ff == 0         │
        │     • Attention: kv  > 0  AND  ff == 0         │
        │     • FFN:                       ff  > 0       │
        │                                                │
        │ Mamba2:                                        │
        │   RMSNorm(attn_norm) ─► ssm_in ─► z, xBC, dt  │
        │   conv1d(xBC) ─► SiLU                          │
        │   SSM scan(dt, A, B, C, state) ─► y            │
        │   SiLU(z) * GroupRMSNorm(y) ─► ssm_out         │
        │   residual += ssm_out                          │
        │                                                │
        │ Attention:                                     │
        │   RMSNorm(attn_norm) ─► QKV (融合)            │
        │   attention (无 RoPE)                          │
        │   attn_output ─► residual                      │
        │                                                │
        │ FFN (dense):                                   │
        │   RMSNorm(attn_norm) ─► up ─► ReLU² ─► down   │
        │   residual += ffn_out                          │
        │                                                │
        │ FFN (MoE):                                     │
        │   RMSNorm(attn_norm)                           │
        │   [可选 latent_in]                             │
        │   route(sigmoid(logits) + bias) ─► topK        │
        │   每个 expert:                                 │
        │       up ─► ReLU² ─► down ─► weighted Σ        │
        │   [可选 latent_out]                            │
        │   [可选 + shared_expert]                       │
        │   residual += moe_out                          │
        └────────────────────────────────────────────────┘
                      │
              RMSNorm(output_norm)
                      │
              LM head (output.weight)
                      │
                      ▼
                   logits
```

## 3. 前向计算图

每层 L 按 `_layerTypes[l]` 派发：

```
if Mamba2:
  hidden ─► RMSNorm(attn_norm)
         ─► ssm_in.weight matmul ─► 拆为 z, xBC, dt
         ─► Conv1dStep(xBC) using ssm_conv1d.weight (state in _convState[l])
         ─► SiLU
         ─► SSM scan step:
                state ← state * exp(-softplus(dt) * A) + B * x * dt
                y     ← state @ C  (+ x * D if D present)
         ─► out = SiLU(z) * GroupRMSNorm(y, ssm_norm.weight)
         ─► out matmul ssm_out.weight → o
         ─► residual = hidden + o

if Attention:
  hidden ─► RMSNorm(attn_norm)
         ─► attn_qkv.weight matmul ─► 拆 Q, K, V (per-layer head 数)
         ─► append (K, V) 到 KV cache
         ─► attention(Q, KCache, VCache, scale = _attentionScale or 1/sqrt(headDim))
         ─► attn_output.weight matmul → o
         ─► residual = hidden + o

if FFN (dense):
  hidden ─► RMSNorm(attn_norm)
         ─► ffn_up.weight matmul → x
         ─► ReluSquaredInPlace(x)            # x ← max(0, x)²
         ─► ffn_down.weight × x → o
         ─► residual = hidden + o

if FFN (MoE):
  hidden ─► RMSNorm(attn_norm)
         ─► [if HasLatentIn] ffn_latent_in × hidden → latent
         ─► logits = ffn_gate_inp × (hidden or latent)
         ─► probs  = sigmoid(logits) + exp_probs_b      # element-wise
         ─► weights, idx = topK(probs)
         ─► if expert_weights_norm: weights /= sum(weights)
         ─► weights *= expert_weights_scale
         ─► 对每个被选 expert e:
                 up   = ffn_up_exps[e]   × (hidden or latent) → x
                 ReluSquaredInPlace(x)
                 down = ffn_down_exps[e] × x
                 accum += weights[e] * down
         ─► [if HasLatentOut] accum = ffn_latent_out × accum
         ─► [if HasSharedExperts] accum += SiLU(...)·... using ffn_up_shexp / ffn_down_shexp
         ─► residual = hidden + accum
```

所有层完成后：

```
hidden ─► narrow(seq_len-1) if prefill
       ─► RMSNorm(output_norm)
       ─► output.weight matmul → logits
       ─► 拷贝到 float[VocabSize]
```

## 4. 组件细节

### 4.1 Mamba2 SSM 层

Mamba2 块是计算上最特化的组件。`ssm_in.weight` 投影输出三个 stream 的 concat：

- **z** —— gate，维度 `_ssmDInner`。
- **xBC** —— 联合的 `(x, B, C)` stream，维度 `_ssmDInner + 2 * _ssmNGroup * _ssmDState`。
- **dt** —— 选择性 timestep，维度 `_ssmNHead`。

`xBC` 接着走 conv1d 滑动窗口（kernel size `_ssmDConv`，状态存于 `_convState[l]`）。SiLU 之后 SSM scan step 计算：

```
ΔA   = exp( -softplus(dt + ssm_dt.bias) * A )           # A ∈ ℝ^{nHead}
ΔB   = (B * x * dt)                                      # B ∈ ℝ^{nGroup × dState}
state = ΔA · state + ΔB                                  # state ∈ ℝ^{dState × headDim × nHead}
y    = state · C                                         # C ∈ ℝ^{nGroup × dState}
y   += x * ssm_d                                         # 当 `ssm_d` 存在时
```

`Mamba2SSMStepSIMD` 在 decode 时用纯 SIMD 向量化的 C# 完成（一次一个 token）。`GroupRMSNorm(y, ssm_norm.weight)` 随后做 per-group 归一，再走 gate `SiLU(z) * y`。最后 `ssm_out` 投影回 hidden 维度。

### 4.2 Attention 层

- 标准 GQA。Per-layer head 数 `_layerNumHeads[l]` 与 KV head 数 `_layerNumKVHeads[l]` 从 GGUF 数组读取。
- 当 `attn_qkv.weight` 存在时使用融合 QKV，否则构造函数的 `FuseQKVWeights()` 由分散的 `attn_q/k/v.weight` 构建融合 tensor。
- Attention scale 是 `nemotron_h.attention.scale`（如设置），否则 `1/sqrt(headDim)`。
- **没有 RoPE** —— SSM 状态隐式携带位置信息，所以 attention 不需要位置编码。

### 4.3 FFN 层（dense）

`ReluSquaredInPlace(x)` 对 `x ← max(0, x)²` 做 SIMD 向量化原地实现。Dense FFN 单次 `ffn_up` matmul 后接 ReLU² 与 `ffn_down`。

### 4.4 FFN 层（MoE）

- **Routing**：`sigmoid(logits) + exp_probs_b`（per-expert 加性 bias）→ TopK → 可选归一化 → 可选全局 scale。
- **Latent bottleneck**（可选）：`ffn_latent_in` 把 hidden 投影到更小的 `latentDim`，experts 在 latent 空间运行，然后 `ffn_latent_out` 投回 hidden。
- **Shared experts**（可选）：`ffn_up_shexp` 与 `ffn_down_shexp` 构成的并行 SwiGLU 分支。
- **批量 MoE GPU 派发**：GGML 后端（Metal / CUDA）上一次 `MoEExpertsForward` 在单张 GGML 图调度中处理所有被选 expert。预缓存的 `QuantizedWeight` 引用（`_expertUpQW`、`_expertDownQW`）和预分配 `IntPtr[]` 数组（`_moeUpPtrs`、`_moeDownPtrs`）避免字典查询与 per-token 分配。

### 4.5 视觉编码器（`NemotronVisionEncoder`）

镜像 NVIDIA RADIO / CLIP 风格 ViT，对应 v2_vl projector：

- **Linear patch embedding**，可选 bias。
- **Position embedding** 存为 `[hidden, posTokens]` 网格，源 patch 网格不同时按 align-corners=false 的双线性 resize。Resize 后的 embedding 按 `(gridW, gridH)` 在 `_resizedPositionEmbeddings` 缓存。
- **Class embedding**（可选）前置到序列。
- **32 个 encoder block**（Nemotron Omni 默认）：LayerNorm → 融合 QKV self-attention → residual → LayerNorm → up linear → GELU → down linear → residual。
- encoder 之后 class token 被剥离，按 `scaleFactor` pixel-shuffle 降低空间分辨率。
- projector（RMSNorm → Linear → ReLU² → Linear）输出 LM hidden 维度的 embedding。

`NemotronImageProcessor`（镜像 `nemotronh.ImageProcessor`）：

1. RGBA 合成到白底。
2. 选择最匹配源图像比例的 tile 网格（最多 `maxTiles`），如有 min/max patches 元数据则走动态分辨率模式。
3. Bicubic resize 到 `gridW * imageSize × gridH * imageSize`（或动态 patch 网格）。
4. 切成 `imageSize × imageSize` tile，channel-first `[C, H, W]`。
5. 多于 1 个 tile 时附加可选缩略 tile。
6. 用 mmproj 中的 CLIP mean/std 归一化每个 tile。

多模态注入器（`ModelMultimodalInjector.cs` 中的 `ProcessNemotronHistory`）把每张图分词为一个 `<image>` 占位符，展开成 `<img>` + N 个 token + `</img>`，对每个 tile 跑视觉编码器，拼接 per-tile embedding，并入队 `PreparedEmbeddingSpan`，让模型在 `Forward()` 之前把 embedding 拼回正确位置。

### 4.6 音频前端（`NemotronAudioPreprocessor`）

Parakeet 风格 log-mel 频谱提取（镜像 ollama 的 `process_audio.go`）：

- 单声道重采样到 16 kHz。
- 0.97 pre-emphasis。
- STFT：`n_fft = 512`、`hop = 160`、`win = 400`、center-padded constant。
- Slaney 风格 mel 滤波器组，128 bins，0..8 kHz。
- `log(power + 2⁻²⁴)`，对有效（非 padding）帧做 per-mel 均值 / 方差归一化。

聊天模板对每个上传音频文件发出一个 `<so_embedding>` token 让模型「看到」该模态，但真正的音频推理依赖 Parakeet 音频 mmproj —— 当前发行的公开 Nemotron-H 或 Nemotron Omni GGUF 都不带这个。CLI 仍会预处理音频文件以验证管线。

## 5. 参数与配置

| Key | 类型 | 含义 |
|---|---|---|
| `nemotron_h.ssm.conv_kernel` | uint32 | Mamba2 conv1d kernel size |
| `nemotron_h.ssm.inner_size` | uint32 | SSM inner 维（`nHead * headDim`） |
| `nemotron_h.ssm.state_size` | uint32 | per head SSM state 维 |
| `nemotron_h.ssm.time_step_rank` | uint32 | SSM head 数 |
| `nemotron_h.ssm.group_count` | uint32 | SSM group 数 |
| `nemotron_h.attention.head_count_kv` | uint32[] | Per-layer KV head 数（0 表示 Mamba2） |
| `nemotron_h.attention.head_count` | uint32[] | Per-layer Q head 数 |
| `nemotron_h.feed_forward_length` | uint32[] | Per-layer FFN size（0 表示无 FFN） |
| `nemotron_h.attention.scale` | float32 | Attention scale 因子（0 表示自动） |
| `nemotron_h.expert_count` | uint32 | MoE expert 数（0 表示 dense） |
| `nemotron_h.expert_used_count` | uint32 | 每 token 选用 expert 数 |
| `nemotron_h.expert_weights_norm` | bool | 是否将选中 expert 权重归一为和 1 |
| `nemotron_h.expert_weights_scale` | float32 | 应用到 expert 权重的全局 scale |

Omni 视觉 projector（`mmproj` GGUF）由 `NemotronVisionEncoder` 读取标准 `clip.vision.*` keys（`embedding_length`、`feed_forward_length`、`attention.head_count`、`block_count`、`image_size`、`patch_size`、`num_channels`、`projection_dim`、`projector.scale_factor`、`attention.layer_norm_epsilon`、`use_gelu`）。

## 6. 权重命名约定

```
token_embd.weight
output_norm.weight
output.weight

# Mamba2 层:
blk.{L}.attn_norm.weight
blk.{L}.ssm_in.weight       # [hidden → 2*dInner + 2*nGroup*dState + nHead]
blk.{L}.ssm_conv1d.weight   # [xBCSize, convKernel]
blk.{L}.ssm_conv1d.bias     # (可选)
blk.{L}.ssm_dt.bias         # [nHead]
blk.{L}.ssm_a               # [nHead]
blk.{L}.ssm_d               # [nHead] (可选)
blk.{L}.ssm_norm.weight     # group RMSNorm [dInner]
blk.{L}.ssm_out.weight      # [dInner → hidden]

# Attention 层:
blk.{L}.attn_norm.weight
blk.{L}.attn_qkv.weight     # 融合 Q+K+V (或独立 attn_q/k/v.weight)
blk.{L}.attn_output.weight

# FFN 层 (dense):
blk.{L}.attn_norm.weight
blk.{L}.ffn_up.weight
blk.{L}.ffn_down.weight

# FFN 层 (MoE):
blk.{L}.attn_norm.weight
blk.{L}.ffn_gate_inp.weight       # router [numExperts, hidden]
blk.{L}.exp_probs_b.bias          # router bias (可选)
blk.{L}.ffn_latent_in.weight      # latent bottleneck in (可选)
blk.{L}.ffn_latent_out.weight     # latent bottleneck out (可选)
blk.{L}.ffn_up_exps.{E}.weight
blk.{L}.ffn_down_exps.{E}.weight
blk.{L}.ffn_up_shexp.weight       # shared expert (可选)
blk.{L}.ffn_down_shexp.weight
```

## 7. TensorSharp 实现走读

构造函数：

1. `ParseBaseConfig()`。
2. 读取 SSM 维度（`_ssmDConv`、`_ssmDInner`、`_ssmDState`、`_ssmNHead`、`_ssmNGroup`，导出 `_ssmHeadDim`）。
3. 读取 MoE 计数与 bias / scale 标志。
4. 读取 per-layer 数组（`head_count_kv`、`head_count`、`feed_forward_length`），把每层归类为 Mamba2 / Attention / FFN。
5. `ParseTokenizer()`、`LoadWeights()`。
6. `FuseFFNWeights()`（为未来 dense FFN 融合占位），`FuseQKVWeights()`（仅 attention 层）。
7. `PrepareCudaQuantizedWeightsForInference()`。
8. `InitCaches(maxSeqLen)` 为 attention 层分配 KV cache，为 Mamba2 层分配 conv state 与 SSM state。
9. `InitMamba2Buffers()` 预分配 `_mamba2ConvOutBuf` / `_mamba2YBuf`。
10. `InitLayerInfo()` 预算 layer-prefix 字符串与 per-MoE-layer 标志（`HasLatentIn`、`HasSharedExperts`、`LatentDim`）。
11. `InitMoEBuffers()` 预分配 routing 与 expert 指针数组。

`Forward(int[] tokens)` 跑 per-op 托管循环。每层根据 `_layerTypes[l]` 派发到 `Mamba2Block` / `AttentionBlock` / `FFNBlock`。可选 CPU 卸载（`CPU_MATMUL_THRESHOLD`，Apple Silicon 默认关）让小 matmul 走托管 CPU 路径，即使后端是 GPU。

## 8. Prefill 优化

- **Per-layer 派发表**（`_layerPrefixes`、`_layerWeightNames`）避免热循环里的字符串拼接。
- **MoE prefill** 仍然 per token 迭代。每 token 用批量 MoE GPU kernel（`MoEExpertsForward`），所以一次派发跑完所有被选 expert，但 token 循环还是托管 C# —— 见下方优化机会。
- **Attention prefill** 走标准托管循环。Nemotron-H 还没有融合 prefill attention kernel，因为 attention 层没有 RoPE，得分张量也比较小（不需要 SWA 窗口的 machinery）。
- **Mamba2 prefill** 顺序处理 token（按 `seqLen` 循环）跑 SSM scan；分块并行扫描在优化清单上。

## 9. Decode 优化

### 批量 GPU MoE（`MoEExpertsForward`）

GGML 后端（Metal / CUDA）上 MoE FFN 层的全部被选 expert 在单次 GGML 图调度中处理：

- 预缓存的 `QuantizedWeight[]` 数组（`_expertUpQW[layer][e]`、`_expertDownQW[layer][e]`）—— init 时一次性填充。
- 预分配的 `IntPtr[] _moeUpPtrs` / `_moeDownPtrs` 数组，长度 `_numExpertsUsed`。
- 预分配可复用的结果 tensor（`_expertUpResult`、`_expertDownResult`、`_latentAccumTensor`、`_latentOutResult`）。

### 小算子的 CPU 卸载

decode 热路径上的小算子（RMSNorm、residual add、expert / router matmul）即使后端是 GPU 也可以通过 CPU SIMD kernel 执行。这避免了 per-dispatch GPU 开销（Metal 上每次 ~1 ms+）。大 matmul（SSM in/out、attention QKV/output、LM head）仍留在 GPU。阈值是 `CPU_MATMUL_THRESHOLD`（Apple Silicon 统一内存默认 0；分立 GPU 系统可调高）。

### SIMD ReLU² 与 bias add

`ReluSquaredInPlace` 与 `LinearForwardInto` 用 `System.Numerics.Vector<float>` 做 SIMD 向量化，dense FFN 在 CPU 上接近峰值向量吞吐。

### 预分配的 decode 缓冲

| Buffer | 大小 | 用途 |
|---|---|---|
| `_mamba2ConvOutBuf` | `dInner + 2 * nGroup * dState` | Conv1d 输出 + SiLU |
| `_mamba2YBuf` | `dInner` | SSM scan 输出 |
| `_moeProbs` / `_moeSelectionProbs` | `numExperts` | Router 概率 |
| `_moeTopExperts` / `_moeRouteW` | `numExpertsUsed` | 选中 expert 与权重 |
| `_moeLatentAccum` | `max(hiddenSize, latentDim)` | latent 空间累加器 |
| `_expertUpResult` / `_expertDownResult` | max expert 维 | 复用的 expert matmul 输出 |
| `_latentAccumTensor` / `_latentOutResult` | `latentDim` / `hiddenSize` | latent bottleneck 复用 tensor |

## 10. 内存与 KV cache 策略

- **Attention 层**：标准 KV cache `[numKVHeads, maxSeqLen, headDim]`。
- **Mamba2 层**：`_convState[layer]`（大小 `(convKernel - 1) * (dInner + 2 * nGroup * dState)` floats）与 `_ssmState[layer]`（大小 `dState * headDim * nHead` floats）。
- `ResetKVCache()` 同时清零三类缓存（KV cache、conv state、SSM state）。
- `SupportsKVCacheTruncation` 返回 **false**，因为 SSM 状态是顺序的，不能部分复用。Nemotron-H 因此不启用多轮 KV cache 复用 —— 服务器在轮次之间回退到完整 reset。

## 11. 输出解析器与聊天模板

- 复用 `Qwen3OutputParser`：`<think> ... </think>` 表示思维链，`<tool_call>{...}</tool_call>` 表示工具调用。
- 聊天模板使用 Qwen 3 风格（`<|im_start|>` / `<|im_end|>`）。多模态占位符包括 `<image>`（之后展开为 `<img>` + N 个 token + `</img>`）与 `<so_embedding>`（音频）。

## 12. 优化机会

- **原生 whole-model decode** —— 当前整个 forward pass 跑在托管 C#。原生 `NemotronModelDecode`（类比 Qwen 3）能消除托管循环开销。
- **原生 Mamba2 decode** —— `Mamba2SSMStepSIMD` 中的 SIMD 向量化扫描在 CPU 上已经很快，但原生 CUDA / Metal kernel 能解锁完整 Mamba2 路径在 GPU 上的执行。
- **分块并行 SSM 扫描** —— Mamba2 prefill 顺序处理 token。分块并行扫描（参考 Mamba 官方 CUDA 实现）能显著降低 TTFT。
- **向量化 conv1d** —— `Mamba2Conv1dStep` 是标量循环。SIMD 或原生向量化版本能进一步加速 Mamba2 层。
- **Per-token MoE 批处理** —— 即使有 `MoEExpertsForward`，per-token 托管循环仍是外层驱动。能在单次派发处理多 token 的批量 kernel 对长 prompt 帮助很大。
- **音频 mmproj 支持** —— 音频前端已经接好，但推理需要尚未在当前 GGUF 中发布的 Parakeet 音频 projector。届时只需把它接入 Gemma 4 同款的 `_pendingAudioEmbeddings` 注入路径，少量代码即可。
