# Qwen 3.5 / 3.6 family

[← 返回模型索引](README_zh-cn.md) | [English](qwen35.md)

| 属性 | 值 |
|---|---|
| 提供方 | 阿里巴巴 |
| GGUF 架构标识 | `qwen35`、`qwen35moe`、`qwen3next` |
| 模型类 | 共享解码运行时上的 [`Qwen35RuntimeModel`](../../TensorSharp.Models/Models/Qwen35/Runtime/Qwen35RuntimeModel.cs)；`--backend cuda` 上的稠密 NVFP4 由 [`Qwen35CudaModel`](../../TensorSharp.Models/Models/Qwen35/Qwen35CudaModel.cs) 执行 |
| 视觉编码器 | [`Qwen35VisionEncoder`](../../TensorSharp.Models/Models/Qwen35/Qwen35VisionEncoder.cs) |
| 图像处理器 | [`Qwen35ImageProcessor`](../../TensorSharp.Models/Models/Qwen35/ImageProcessor.cs) |
| 示例模型 | Qwen3.5-9B（dense hybrid）、Qwen3.5-35B-A3B / Qwen3.6-35B-A3B（MoE 系列）、Qwen3.6-27B / Qwen3.8-27B（dense）；另有 [Bonsai2 27B](bonsai2_zh-cn.md)（PRISM PQ2_0 / PTQ1_0） |
| 模态 | 文本、图像 |
| 思维链模式 | 是（`<think> ... </think>`） |
| 工具调用 | 是（`<tool_call>{...}</tool_call>`） |
| 批处理 / 分页前向 | **默认启用** —— `--no-continuous-batching` 会强制走按序列 KV-swap 路径。带每槽位 GatedDeltaNet 递归状态池。详见 §11。 |
| MTP 投机解码 | Qwen 3.6 —— NextN 草稿块内嵌在主干 GGUF 中（无需独立文件；仅限保留 MTP 的 GGUF，见[下载](#下载)）；在**两个宿主上**都用 `--spec` 启用 —— `TensorSharp.Cli` 与 `TensorSharp.Server` 共用同一个 [`SpeculativeCliFlags`](../../TensorSharp.Runtime/Speculative/SpeculativeCliFlags.cs)。部分接受时做 GDN 递归状态快照 / 回滚。只要 GGUF 保留 NextN 块，就会对单序列（无并发）请求启用。Qwen 3.8 另外接受以独立 `--draft-model` GGUF 提供的 **DFlash2** 块草稿器（§12.4）。详见 §12。 |
| 输出解析器 | `Qwen35OutputParser`（继承 `ChatMlOutputParser`） |

## 下载

已验证的 GGUF 来源（下表中每个模型使用独立目录 —— 9B 与 3.6 两个仓库的投影器
都叫 `mmproj-F16.gguf`，因此不能共用同一目录）：

| 模型 | HF 仓库 | 推荐文件 | 视觉投影器 |
|---|---|---|---|
| Qwen3.5-9B（dense hybrid） | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | `Qwen3.5-9B-UD-Q4_K_XL.gguf`（5.966 GB）或 `Qwen3.5-9B-Q8_0.gguf`（9.528 GB） | `mmproj-F16.gguf`（0.918 GB） |
| Qwen3.5-35B-A3B（MoE） | [ggml-org/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF) | `Qwen3.5-35B-A3B-Q8_0.gguf`（36.903 GB） | `mmproj-Qwen3.5-35B-A3B-Q8_0.gguf`（0.614 GB） |
| Qwen3.6-35B-A3B 含 NextN / MTP（MoE） | [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`（22.663 GB）或 `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf`（11.819 GB） | `mmproj-F16.gguf`（0.899 GB） |
| Qwen3.8-27B NVFP4（dense hybrid，4-bit FP） | [akopytko/Qwen3.8-27B-NVFP4-GGUF](https://huggingface.co/akopytko/Qwen3.8-27B-NVFP4-GGUF) | `Qwen3.8-27B-NVFP4-MTP-N4_0.gguf`（15.716 GB，自包含 NVFP4） | `mmproj-BF16.gguf`（0.931 GB） |

> **Qwen 3.6 MTP：** NextN 草稿块（GGUF 元数据键 `nextn_predict_layers`）**只**在
> `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` 的 GGUF 中保留。基础仓库
> [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
> 中的文件（文件名相同）去掉了该块 —— 对这些文件，`--spec` 会回退到标准 decode，
> 并打印一次“该 checkpoint 不含草稿头”的警告（`--spec-type ngram` 在那里仍然可用，
> 它不需要训练好的草稿器权重）。

> **NVFP4（GGML 类型 40）：** 4-bit E2M1 数值加 UE4M3 每 16 元素子块缩放
> （每 64 个权重 36 字节，4.5 bpw），即 NVIDIA Blackwell 原生 FP4 格式。
> GGUF 有两种形态：自包含文件（所有缩放折叠进块内，如上表仓库）和从
> checkpoint 转换、带有额外 1 元素 `blk.N.<proj>.scale` 张量（HF
> `weight_scale_2`）的文件 —— TensorSharp 与 llama.cpp 一致地把它乘到投影
> 输出上（`.input_scale` 是校准元数据，推理时忽略）。在 sm_120 GPU 上，
> prefill 走 ggml-cuda 的原生 block-scaled FP4 tensor-core 路径。

转换模型卡将对应的 Qwen 官方仓库标记为基础模型；这些上游模型卡声明 Apache-2.0。

```bash
python -m pip install -U huggingface_hub
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf --local-dir models/qwen3.5-9b
hf download unsloth/Qwen3.5-9B-GGUF mmproj-F16.gguf --local-dir models/qwen3.5-9b
hf download ggml-org/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-Q8_0.gguf --local-dir models/qwen3.5-35b-a3b
hf download ggml-org/Qwen3.5-35B-A3B-GGUF mmproj-Qwen3.5-35B-A3B-Q8_0.gguf --local-dir models/qwen3.5-35b-a3b
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --local-dir models/qwen3.6-35b-a3b-mtp
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF mmproj-F16.gguf --local-dir models/qwen3.6-35b-a3b-mtp
```

CLI 单次推理（文本提示词通过 `--input` 从文件读取；采样默认为 greedy，
`--max-tokens` 默认为 100 —— 实际问答请调大）：

```bash
printf '%s\n' '描述这张图片。' > prompt.txt
dotnet run --project TensorSharp.Cli -c Release -- --model models/qwen3.5-9b/Qwen3.5-9B-UD-Q4_K_XL.gguf --mmproj models/qwen3.5-9b/mmproj-F16.gguf --input prompt.txt --image photo.png --max-tokens 512
```

MTP 演示。`--spec` 及其余 `--spec-*` 系列参数由同一个
[`SpeculativeCliFlags`](../../TensorSharp.Runtime/Speculative/SpeculativeCliFlags.cs)
在**两个宿主上**解析，因此 `TensorSharp.Cli` 与 `TensorSharp.Server` 不会在名字、
校验或默认值上出现分歧。它们必须写在*加载*
模型的那条命令行上，因为 `--spec-draft` 同时还在加载期决定原生计算图缓存的大小：

```bash
# 服务端
dotnet run --project TensorSharp.Server.Host -c Release -- --model models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --backend ggml_cuda --spec

# CLI —— 投机在 --input、--input-jsonl、--multi-turn-jsonl 与 --interactive 上生效
dotnet run --project TensorSharp.Cli -c Release -- --model models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --backend ggml_cuda --spec --spec-draft 4 --input prompt.txt --max-tokens 512
```

## 1. 来源与目标

Qwen 3.5 / 3.6 系是阿里第一个「混合」Qwen 系列，也是 TensorSharp 中优化最深的模型：

- **混合 attention + recurrent 堆栈。** 大多数层是 GatedDeltaNet（一种 linear-attention / SSM 风格的递归层）；少量层是 FullAttention 层，按 `full_attention_interval` 间隔均匀分布。Recurrent 层在 token 间维护形如 `[numVHeads, headVDim, headKDim]` 的隐状态；attention 层维护标准 KV cache。
- **MoE FFN** 出现在 `qwen35moe` / `qwen3next` 变体（如 Qwen3.5-35B-A3B、Qwen3.6-35B-A3B）。256 个 routed expert，每 token 激活 8 个，外加可选的 always-on shared expert。
- **Sigmoid-gated FullAttention。** Q 投影输出行数翻倍（`2 × numHeads × headDim`）；后半 deinterleave 后作为 sigmoid gate，在输出投影之前与 attention 输出相乘。
- **动态分辨率视觉。** 图像走 `Qwen35VisionEncoder`，一个支持任意分辨率的 2D-RoPE ViT，通过 patch embedding + spatial merger 完成 tokenization。

三种 GGUF 架构标识（`qwen35`、`qwen35moe`、`qwen3next`）共用同一个 `Qwen35RuntimeModel`（§ 7）；MoE 变体在其 FFN 中加入路由专家。

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

#### 图片之后的位置：M-RoPE 偏移（delta）

合并网格为 H x W 的图片占用 KV cache 的 H*W 行，却只占 max(H, W) 个旋转位置。注入器按 HF 与
SGLang `get_rope_index` 的方式排布提示（`ModelMultimodalInjector.LayoutQwenVLPrompt`）：区间内
token 位于 `(base, base + h, base + w)`，其后的文本从 `base + max(H, W)` 继续。位置表之外的每个
token 都位于

```
rope 位置 = KV 下标 + delta，   delta = 位置表最后一行的最大分量 + 1 - 提示长度
```

即 SGLang 的 `mrope_position_delta`（decode 位置为 `delta - 1 + seq_len`）。delta 按序列保存
（[`Qwen35RopePositions`](../../TensorSharp.Models/Models/Qwen35/Qwen35RopePositions.cs)）：

- **在任何内核运行之前确定。** `BeginRopePositions` 从 prefill 分块暂存的位置表取得它；从 KV 下标
  0 开始且没有位置表的前向会开启新历史（delta 为 0）；其他前向沿用当前序列的 delta。若一个分块的
  所有行都是位于 `KV 下标 + delta` 的文本，则丢弃位置表，改走普通的标量 RoPE 图，旋转完全相同。
- **所有路径都使用它。** 整模型 decode（`rope_pos_delta`）、融合 prefill/verify 图（标量行使用
  `rope_pos_delta`）、arena 批处理 decode（每个 holder 的 `rope_positions`）、分页批处理前向
  （来自注入器的请求 delta）、逐算子 attention、direct-CUDA 的 prefill 与 decode 图、张量并行以及
  MTP 草稿头。逐层原生 attention 内核按 KV 下标旋转，因此越过图片的序列不会走它们。
  （DFlash 草稿器保留自己的位置；它们只影响草稿被接受的数量，
  从不影响输出。）
- **与其描述的状态一起保存。** 每个请求的 holder 在换入换出、保留、重新绑定和池化过程中携带它；
  共享前缀检查点及其克隆会复制它；KV-swap 并发路径的逐块 KV 快照在末尾保存它（与递归状态一样，
  取块结束时的值；注入的块若大小不符，或其 delta 会让下一个 token 落在负位置，会在写入任何内容之前被
  拒绝，因此被拒绝后模型恰好保留该块之前的内容）；检查点文件格式（`Q5KC`）升级为**版本 2**，在行数之后写入 delta。
  版本 1 的文件没有 delta，导入时会被拒绝；引擎记录“保存的检查点与模型不符”，重新 prefill 该前缀
  并再次保存。

在此之前，decode、投机 verify 以及所有标量路径都直接按 KV 下标旋转，带来两个后果：

- **单个请求。** 对图片的回复是在模型训练时不会在该图片之后见到的位置上生成的：提示之后的每个 token
  都比参考位置多出“图片 token 数减 max(H, W)”（1253x836 的测试图片约为一千个位置）。修复之后
  **图片之后的输出会改变**：现在它们就是正确的 Qwen-VL 实现所产生的结果。在 Mac 上实测（Metal，
  Qwen3.5-9B-Q8_0，贪心，96 token），图片回合的回复在修复前（584f8f71，与 6db6dbf6 的 decode
  相同）与修复后分别为：
  - OpenAI，第 1 轮带图：`...sitting gracefully against a futuristic, glowing background filled
    with floating cubes and digital particles.` 变为 `...sitting gracefully amidst a futuristic,
    glowing digital landscape filled with floating cubes and light trails.`
  - Web UI，第 3 轮带图：`...amidst a futuristic digital environment.` 变为
    `...amidst a futuristic digital landscape.`
  - OpenAI，第 3 轮带图：`This digital artwork features an anime-style woman...` 变为
    `This image features an anime-style illustration of a young woman...`
  - Web UI，第 1 轮带图：96 token 内没有变化。
- **复用。** 保存逐序列 rotary delta 后，后续回合可从精确的缓存端点越过相同图像或视频 span 续接。
  radix 键同时校验媒体身份、span 边界与 token；改变媒体或会话 scope 后，不能使用该会话状态。
  GDN 状态仍无法回退，因此较短的部分匹配需要重新 prefill。

**验证。**

- `Qwen35MRopeReferencePositionTests` 将提示排布、delta、模型自身的分块 prefill 与 decode 位置，
  以及从头排布的后续回合，与由 SGLang `get_rope_index(model_type="qwen3_5")` 及其 decode 规则生成
  的夹具进行比较（`eng/validation/qwen35_mrope_reference`，SGLang 2733afe5），覆盖一张图、两张图、
  高与宽的网格、两对帧的视频和纯文本。若 decode 按 KV 下标（旧规则），26 个用例中有 10 个失败。
- `Qwen35ImageFollowUpExactnessTests`（需要模型，`TS_TEST_MODEL_DIR`）在 Qwen3.5-9B-Q8_0 上运行
  “文本 -> 图片 -> 文本 -> 文本”的对话。在引擎中，第 3、4 轮必须越过图片复用上一轮，每一轮的贪心
  token 都与冷启动引擎比较；旁边同时有一段文本对话在 decode（按请求的 holder 与 arena 批处理 decode）
  时同样比较。直接在模型上，把“decode 上一轮回复再 prefill 新后缀”构建出的回合与整段提示的冷启动
  prefill 逐步比较 24 步 logits，并对一段纯文本对话做同样的比较作为对照。在图片之后建立的检查点经过
  导出、导入和克隆后，decode logits 逐位相同；版本 1 文件会被拒绝。关闭 delta 时测试失败：在 Metal
  上图片对话的最差 logit 差从 0.024 升到 3.24，而纯文本对照保持 0.010；在 CUDA 上从 2.39 升到 8.11（为其对照
  0.92 的 8.8 倍），且第 4 轮出现一个并非平局的 token 变化（在测试的早期版本中，它在第 3
  轮 token 上失败：复用得到 `...there is no roof visible. The scene depicts...`，冷启动得到
  `...features...`）。
  下面的测量使用真实照片（`TS_TEST_QWEN35_IMAGE`）。未设置时测试会绘制一张合成图片：直接比较用
  448x336，并发用例用 896x672（radix 树从 32 token 起保留已完成的请求，即 `MinRetainTokens`）。

**Logit 容差。** 复用与冷启动并非逐位相同：复用回合的回复行由 decode 图写入，冷启动回合由 prefill
图写入（attention 与矩阵乘内核不同，CUDA 上是量化激活的矩阵乘，等轴时一个是 NeoX、一个是交错
M-RoPE）。因此测试检查三点：

- 词表上最差的最大 |dlogit| 不超过该后端的容差（`TS_TEST_QWEN35_LOGIT_TOLERANCE`；Metal 默认 0.1，
  其他后端 3.0）；
- 不超过纯文本对照的 4 倍（`TS_TEST_QWEN35_CONTROL_RATIO`）：图片的上下文更长，可以放大内核差异，
  但不能带来自身的误差；
- 贪心 token 相同，除非在出现差异的那一步冷启动的 top-2 差距小于该步测得的 logit 差——这是两个内核
  都可能打破的平局。

实测（24 步，第 2-4 轮）：

| 后端 | 图片对话最差 \|dlogit\| | 纯文本对照 | 比值 | token 差异 | 检查点往返 |
|---|---|---|---|---|---|
| Metal（M 系列，Qwen3.5-9B-Q8_0） | 0.024 | 0.010 | 2.3x | 无 | 0.0 |
| CUDA（A40，Qwen3.5-9B-Q8_0，mmproj F16） | 2.39 | 0.92 | 2.6x | 第 3 轮第 0 步（差距 0.034 < 0.47）与第 4 轮第 1 步（差距 0.13 < 0.29）：平局 | 0.0 |

在 CUDA 上，即使是纯文本对话，decode 与 prefill 内核也会相差约一个 logit，因此在 24 个贪心 token 内，
无论有没有图片，复用回合与冷启动回合都可能在低差距 token 上翻转。

并发测试（holder 与 arena）采用同样的平局规则。在某个请求的 token 与其冷启动引擎运行首次不同的那一步，
测试捕获冷启动 logits（直接在模型上对同一提示做冷启动 prefill，且到该步为止必须解码出相同的 token），
只有当其 top-2 差距低于该后端的 logit 容差时才接受这一差异，并且不再比较该请求的剩余部分。它的纯文本
对照以长于一个块的系统提示开头（第 1 轮 346 token）：使用一行系统提示时每个文本回合都短于一个块，因此在旧版路径下没有
回合被保留，对照什么也没复用。现在它必须复用上一轮留下的全部内容。2026-09-17 实测：

| 后端 | 图片 | 文本第 2-4 轮复用 | token 差异 |
|---|---|---|---|
| Metal（Qwen3.5-9B-Q8_0，mmproj BF16） | 内置 896x672 | 367 / 411 / 446 | 无 |
| Metal | 真实照片 | 367 / 411 / 446 | 无 |
| CUDA（A40，Qwen3.5-9B-Q8_0，mmproj F16） | 内置 896x672 | 366 / 410 / 445 | 图片第 3 轮第 0 步（差距 0.23）、图片第 4 轮第 5 步（0.017）、文本第 3 轮第 0 步（0.041）：平局 |
| CUDA | 真实照片 | 366 / 410 / 445 | 文本第 3 轮第 0 步（差距 0.041）：平局 |

引入平局规则之前，CUDA 上使用内置图片的运行在图片第 3 轮第 0 步失败。

**经过服务端**（Phase 0 的 IMG 探针：Web UI 与 OpenAI 对话，图片在第 1 或第 3 轮，另有纯文本对照；
Metal，Qwen3.5-9B-Q8_0，贪心，每轮 96 token），图片之后的每一轮现在都续接缓存：

| 场景 | 图片之后的回合 | 之前复用（截断） | 现在复用 | 之前 TTFT（Web UI）/ 总耗时（OpenAI） | 现在 |
|---|---|---|---|---|---|
| Web UI，第 1 轮带图 | 2、3、4 | 0、0、0 | 1,099 / 1,187 / 1,286（98.0-98.2%） | 1.00 / 1.10 / 1.18 s | 0.13 / 0.13 / 0.13 s |
| Web UI，第 3 轮带图 | 4 | 0 | 1,340（98.2%） | 1.20 s | 0.13 s |
| OpenAI，第 1 轮带图 | 2、3、4 | 0、0、0 | 1,099 / 1,188 / 1,286（97.9-98.2%） | 3.67 / 3.64 / 3.31 s | 2.20 / 2.39 / 1.41 s |
| OpenAI，第 3 轮带图 | 4 | 0 | 1,301（98.1%） | 2.18 s | 0.94 s |

该次运行的全部 24 条回复都与同一构建关闭提示复用（`TS_SCHED_PREFIX_CACHE=0`）时的结果相同，图片
只编码一次，纯文本对照不变。

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

`FuseAttentionProjectionWeights()` 在三个 GGUF 张量量化类型相同时，把每层的 Q+K+V 融合为 `attn_qkv.weight`；混合类型的 importance-matrix / UD 量化投影保持分开，避免展开成数 GB 的 FP32。`FuseRecurrentInputWeights()` 把 GatedDeltaNet 的五路递归投影（`attn_qkv`、`attn_gate`、`ssm_beta`、`ssm_alpha`、加一个空槽）融合为 `ssm_in_proj.weight`。`FuseGateUpWeights()` 融合 dense FFN 的 gate 与 up。

## 7. TensorSharp 实现

所有加载都经过 [`Qwen35Architecture`](../../TensorSharp.Models/Models/Qwen35/Qwen35Architecture.cs)：

| 加载 | 由谁执行 |
|---|---|
| `--backend cuda`、单 GPU、稠密 NVFP4 检查点、没有 `--tp` / `--layer-split` / 主机专家 | [`Qwen35CudaModel`](../../TensorSharp.Models/Models/Qwen35/Qwen35CudaModel.cs)，直连 CUDA 引擎（`TS_Q35_NATIVE=0` 让它让位） |
| 其余全部：每个后端（`cpu`、`ggml_cpu`、`ggml_metal`、`ggml_cuda`、`ggml_vulkan`、`mlx`、`cuda`）、单设备、`--tp`、`--layer-split`、二者混合、多节点流水线、主机专家 | 运行在[共享解码运行时](../design/unified-execution.md)上的 [`Qwen35RuntimeModel`](../../TensorSharp.Models/Models/Qwen35/Runtime/Qwen35RuntimeModel.cs) |

本家族自己的模型类（`Qwen35Model`，及其融合整模型 GGML 图、槽位稳定 arena、常驻验证图与原生内核）已于 2026-11 删除；它服务过的每种放置都由运行时接管。

`Qwen35RuntimeModel` 是本家族在运行时上的配方：

- **带门控的注意力。** `attn_q` 按头输出 `[q_h | gate_h]`；逐头 Q/K RMSNorm 后接部分 NeoX 旋转（图像之后用 M-RoPE 分段），注意力输出先乘 `sigmoid(gate)` 再进 `attn_output`。
- **Gated DeltaNet 作为递归混合层。** 输入投影（`attn_qkv`、`attn_gate`、`ssm_beta`、`ssm_alpha`）在量化类型一致时打包为一次乘积；随后是 `ssm_conv1d`、q/k L2 归一化、delta 规则（GGML 后端用 ggml 的 `ggml_gated_delta_net`，直连 CUDA、MLX 与托管 CPU 用运行时自己的内核）、带 `silu(z)` 的门控 RMSNorm 与 `ssm_out`。卷积窗口与 delta 状态按序列槽位保存（`--tp` 下按 rank 保存）。
- **FFN。** 稠密 SwiGLU（类型一致时 gate/up 融合），或带共享专家及其 sigmoid 门的路由专家。
- **图像。** [`Qwen35VisionEncoder`](../../TensorSharp.Models/Models/Qwen35/Qwen35VisionEncoder.cs) 编码图块，在图像区间注入行，之后文本位置遵循压缩后的 M-RoPE delta（[`Qwen35RopePositions`](../../TensorSharp.Models/Models/Qwen35/Qwen35RopePositions.cs)）。
- **检查点变体。** Bonsai2 的 PRISM 旋转在 GGML 后端运行（图构建器旋转每个被旋转乘积的输入；其他后端与 `--tp` 拒绝加载）。NVFP4 旁路缩放乘到各自投影上，路由专家上的除外（运行时拒绝）。

每一步都降为一张图：GGML 后端上是整次前向的 GGML 图（decode 步以新输入重放上一步的图），直连 CUDA、MLX 与托管 CPU 上是张量执行器的程序。[`Fixtures/Qwen35Legacy`](../../InferenceWeb.Tests/Fixtures/Qwen35Legacy/README.md) 中的黄金 logits 保存了被删实现的输出，`Qwen35RuntimeTests` 让运行时在各后端的舍入范围内与之一致。

## 8. Prefill

一个提示分块就是一张覆盖所有层的图。全注意力层按下标写入 K/V 行并做因果注意力（后端对该头宽有内核时用 flash attention）；递归层在分块上运行分块 delta 规则。提示分块默认 2,048 行（`TS_PREFILL_CHUNK`）。带图像的提示先运行视觉编码器并注入其行。

## 9. Decode 与后端说明

decode 每步对所有运行中的序列只跑一张图：每个序列的行读取自己槽位的 K/V 窗口与递归状态。GGML 后端上下一步就地改写位置、写入行与掩码后重放该图。ggml_metal 上 decode 行的递归原地更新其状态。

### Direct CUDA NVFP4 路线：输出层、验证 GEMM 与预填充 GEMM

NVFP4 GGUF 走 `--backend cuda`（`Q35CudaEngine`）时，每一步都要把全部权重读一遍，所以每步耗时取决于核函数能用到多少 RTX 5090 显存带宽。2026-11 替换了三个明显偏慢的核函数；每个形状都与旧核函数逐位比对过，结果完全一致，回答不变。

| 核函数 | 改前 | 改后（RTX 5090） |
|---|---|---|
| 输出层，Q6_K 248,320 x 5,120（每步目标头与 DFlash2 草稿头各一次） | 210 字节超块每次读 2 字节，每个超块一次 `__syncthreads`：0.83 TB/s，32 行 1.27 ms | 加载时重排为 64 行一块（`ts_q35_repack_q6k_tiled`，同样的字节换顺序），用 cp.async 流水线读取：32 行 0.74 ms（1.41 TB/s），8 行 0.89 -> 0.69 ms，1 行 1.06 -> 0.87 ms |
| W4A4 验证 GEMM（9-64 行）的 split-K | 切分到 CTA 覆盖所有 SM：129 块的 GDN 输入投影切 2 份、跑两波 | 取仍能一波放下的最大切分：32 行时 GDN 输入投影 38.1 -> 30.6 us，5,120 行输出投影 18.1 -> 14.3 us |
| 预填充 W4A4 GEMM（65 行以上） | cp.async 128 行块，约 830 TFLOPS | 改用 NInfer 路线的 TMA warp 专用核函数，并为 gate/up 增加 F32 SwiGLU epilogue：2,048-4,096 行时 1,230-1,390 TFLOPS |

端到端（GGUF + DFlash2 K7，短提示、400 token 回答）：4 路并发 464 -> 521 tok/s，2 路并发 284 -> 308 tok/s。预填充首 token：11K token 提示 1.51 -> 1.20 s，36K token 6.57 -> 5.38 s，4 个 3.6K token 提示同时到达 1.9-2.1 -> 1.6-1.7 s。

仍未解决：1-8 行的 W4A8 GEMV 在 8 行时只有 1.17-1.37 TB/s。保持 GEMV 求和顺序的 int8 张量核版本（每个 GEMV lane 一个累加器）需要 242 个寄存器，只有 0.3-0.9 TB/s，没有采用。

**长文档上的 2 路并发与多轮会话。** 2 路并发时 DFlash2 验证有 16 行，投影从 8 行 GEMV 切到 W4A4 GEMM，32 行块只用了一半。实测这一半并不花时间：16 行时这些 GEMM 受显存带宽限制，与 8 行 GEMV 一样快甚至更快（gate/up 65 vs 73 us，GDN 输入投影 31.5 vs 40.5 us），逐位一致的 16 行块最多只快 3%。真正的开销有三处，均已修复：

| 开销 | 改前 | 改后（RTX 5090） |
|---|---|---|
| 9 行以上的 pre-norm + NVFP4 量化（每层两次） | 每组元素都要等一次全局内存往返：5.5 us | `ts_q35_add_rmsnorm_nvfp4_5120_f32` 把整行放在寄存器里，先发出全部读取：3.5 us，逐位一致 |
| 长缓存上的短提示块（会话的新一轮、新对话的后缀） | 8 位提示注意力不切分键：127K 缓存上 47 token 的块只有 8 个 CTA，每层 5.3 ms，另加 0.65 ms 转换 band | 改用把键切分到整卡的 16 位核函数，直到 8 位 CTA 能覆盖约 40% 的 SM（约 350 行；两条路线都适用）：该轮首 token 140 -> 50 ms |
| 同一步进入、共享缓存前缀的多个对话 | 从 radix 缓存恢复的请求还没有 holder，批量预填充跳过它，后缀逐个预填充 | 执行器先绑定 holder、注入前缀（`PrebindCachedPrefixPrefills`），后缀共用一次前向：120K token 文档上 3 个这样的对话首 token 相差不到 3 ms |

### MLX：UD 量化与推测解码

M5 Pro（48 GB）实测：Qwen3.8-27B GGUF，DFlash2 Q4_K_M 草稿模型（`--spec-draft 7`），2.3K token 提示，512 token 回答；推测解码数据为同一进程的第二个请求（核函数已编译）。

- **低比特格式上 GPU。** Unsloth 的 UD-IQ3_XXS 在 10.9 GB 中有 1.2 GB 是 Q2_K、Q3_K、IQ2_XS、IQ1_S、IQ1_M，此前 MLX 没有对应核函数，走 CPU 托管反量化。现在都有了设备核函数（`MlxLowBitQuantKernels.cs`，反量化运算与 `ManagedQuantizedOps` 一致）：解码 1.8 -> 15.7 tok/s。
- **预填充。** 原始 i-quant 在 32 行以上时先把权重分片反量化为 F16，再用 MLX GEMM（`TS_MLX_IQ_DEQUANT_GEMM_MIN_ROWS`），与 Q6_K、IQ4_XS 的做法一致：2.3K token 预填充 361 s -> 6.05 s（ggml_metal 7.0 s）。
- **验证窗口。** 2-31 行走少行核函数（`TS_MLX_IQ_FEWROW_MAX_ROWS`），现在的布局与 ggml-metal 的少行 MMA 核函数相同：threadgroup 内各 simdgroup 切分 K，每个 lane 把一块中自己的 16 个权重只解码一次、直接放进 simdgroup 矩阵寄存器（F32），最后把各部分块相加一次。被替换的核函数把 64 列解码结果暂存在 threadgroup 内存里，每 64 个权重一对 barrier；即使解码几乎不花时间，8 行前向的 128 ms 里它仍占 106 ms。8 行前向 128 -> 79 ms（ggml_metal 87），2 行 159 -> 75，16 行 212 -> 151（ggml_metal 153），31 行 383 -> 279；验证窗口 155 -> 92 ms（ggml_metal 88）。IQ4_XS 与 Q6_K 解码 8 个值时只读一次缩放，不再逐值读取。
- **拒绝后不再重算。** 验证保留每一行之后的循环状态，接受行的状态直接在设备上提交，与 ggml 验证的快照做法相同；验证前快照改为引用不可变的 MLX 数组，不再复制到主机（每步 6 ms -> 0.1 ms）。此前每次拒绝都要恢复状态并重新前向接受的前缀，平均 135 ms。
- **草稿模型。** DFlash2 的分组卷积在设备上计算，滑动窗口掩码以加法施加，不再每层四五次把系数、行和分数读回主机：每次起草 51 -> 32 ms。

- **单行解码 matvec。** codebook i-quant 与低比特类型的单行核函数原先每个权重行占一个 256 线程的 threadgroup，每行都要把整行激活（K = 5,120 时 20 KB）重读一遍，最后还有一次 barrier。现在改为 ggml-metal 的 mul_mv 布局：每个 threadgroup 两个 simdgroup、各负责 4 个权重行，每个 lane 只读一次 32 个激活值供 4 行共用。普通解码 15.7 -> 20.4 tok/s（每 token 49 ms），与同一文件上的 ggml_metal（20.6）和 llama.cpp（20.5）持平。
- **短提示与更宽的窗口改用矩阵单元。** 超过 8 行后该核函数受 F32 simdgroup 乘法限制（8 行前向约 430 GFLOP），而 32 行起的反量化 GEMM 每次调用都要把整份权重写成 F16（不论行数都约 590 ms）。在 M5 上，同样的 K 切分现在改用张量 API 做乘法：每个 simdgroup 把 32 个权重行的一块解码进自己的 tile，再与 16、32 或 64 行输入相乘（`TS_MLX_FEWROW_TENSOR`，9-1024 行）。前向 16 行 151 -> 82 ms（ggml_metal 153），32 行 599 -> 104（291），128 行 723 -> 325（363），512 行 1377 -> 1205（1289）；1024 行时与反量化 GEMM 持平，更长的分块仍走 GEMM。
- **仿射权重：草稿器。** Q4_K 与 Q8_0 加载为 MLX 仿射权重，原先走 MLX 的 `quantized_matmul`：在草稿器的 7-8 行上只有约 120 GB/s。现在它们也走少行核函数，配一个仿射解码器（`TS_MLX_AFFINE_FEWROW_MAX_ROWS`，有张量 API 时最多 128 行）：一次 DFlash2 起草 17.6 -> 13.4 ms（各层 11.0 -> 9.0，248K 行的输出头 5.9 -> 3.7），目标模型自身的 Q4_K 张量也让验证窗口从 92-96 降到 89 ms。
- **寄存器数组要展开。** 这一布局第一次移植时 8 行前向要 225 ms：遍历累加器与指针数组的循环没有展开，数组落在线程内存里。把这些循环全部展开（`clang loop unroll(full)`，即 ggml 的 `FOR_UNROLL`）后为 79 ms。相比之下分块方式在 M5 Pro 上几乎无关紧要（试过的形状都是 79-81 ms，8 个权重块为 85 ms）。
- **48 GB Mac 上保留的会话。** 每个保留的会话都按整段上下文分配（32K 时约 2.5 GB），16 个就让服务占用超过 34 GB 并开始换页，所有场景都慢了 3 倍。现在保留状态受 macOS 能给出的内存约束（见[批处理文档](../PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)）：下面的完整基准始终在 18-23 GB。

服务端，贪心，`--spec-draft 7 --spec-pmin 0`，MAX_CONTEXT 32768，各跑两次（首 token 时间 / 解码速度）：

| UD-IQ3_XXS + DFlash2 | ggml_metal | MLX 改前 | MLX 现在 |
|---|---:|---:|---:|
| 长生成（74 token 提示，1,220 token） | 0.57-0.60 s / 30.1-31.3 tok/s | 1.27-1.43 s / 17.6-17.7 | 0.34-0.37 s / 31.0-31.6 |
| 4K token 提示 | 12.9 s / 35-52 | 10.6-10.7 s / 15-18 | 9.8-10.0 s / 39-41 |
| 16K token 提示 | 53.5 s / 33 | 43.1-43.3 s / 14-15 | 41.1 s / 29 |
| 多轮：首轮，之后命中缓存的轮次 | 9.2-9.3 s，0.32-0.59 s | 8.4-8.8 s，0.56-0.67 s | 6.4-8.5 s，0.12 s |
| 短对话 | 0.57 s | 0.66-1.05 s | 0.21 s |

普通解码 MLX 每 token 46.8 ms，ggml_metal 48.4 ms。16K token 提示之后的解码（56 个 token 上 29 对 33 tok/s）是唯一仍落后的数字。

## 10. 内存与 KV cache 策略

- **全注意力层**：每个槽位一份 K/V cache，`[kv_heads, capacity, head_dim]`，类型由 `--kv-cache-dtype` 决定（`f32`、`f16`、`q8_0`、`q4_0`）。槽位从小容量开始，按需增长到上下文长度；在一步即将耗尽显存前，空闲的保留槽位会先交还其设备副本。
- **递归层**：每个槽位保存卷积窗口 `(conv_kernel - 1) x qkv_dim` 与 delta 状态 `[v_heads, head_v_dim, head_k_dim]`。递归状态无法回退，因此槽位只能截断到自身长度或清零。
- **文件映射权重**：后端能直接读取时（GGML CUDA、Metal、CPU、集成 GPU），量化张量绑定 GGUF 映射而不复制。

## 11. 连续批处理与前缀复用

并发请求各用自己的槽位；一步把所有运行中请求的行合批（不同长度的提示分块与 decode 行一起）。radix 前缀缓存把一轮结束后保留的槽位用于同一会话的下一轮；共享前缀检查点（某一边界处 K/V 行与递归状态的副本）让新请求从公共前缀开始。由于状态无法回退，提示的自动检查点位于其末尾前一个 token，以便同一提示可以再次被服务。`Qwen35RuntimeTests` 检查合批槽位与单独运行一致、引擎在每种放置下的并发请求，以及检查点克隆与文件的精确续接。

## 12. 投机解码

运行时一次前向验证一整个草稿窗口，并保留窗口内每一行之后的递归状态，因此拒绝时直接提交被接受行的状态而无需重新前向（GGML 图保留逐行快照；逐算子路径保留窗口输入并重放被接受的前缀）。草稿器：

- **n-gram**（`--spec --spec-type ngram`），适用于 Qwen 3.5 家族的每个检查点。
- **NextN / MTP**（Qwen 3.6、Qwen 3.8 27B，`--spec`）：内嵌的 NextN 块作为最后一个 stage 上的草稿层加载；单独成文件的 MTP 头用 `--draft-model` 加载。
- **DFlash / DFlash2** 块草稿器（Qwen 3.8）：`--draft-model <drafter.gguf>`；草稿器读取主干捕获的隐藏行。

投机在单设备、`--tp`（按 rank 的图）以及多节点流水线（驱动节点运行最后一个 stage 并在那里起草）上都可用。`--spec-draft` 限制每步的草稿数，引擎的调速器在投机不划算时暂停它。

## 13. 输出解析器与聊天模板

- `Qwen35OutputParser` 继承 `ChatMlOutputParser`，wire 格式为：`<think> ... </think>` 表示思维链，`<tool_call>{"name": "...", "arguments": {...}}</tool_call>` 表示工具调用。
- 聊天模板使用标准 Qwen `<|im_start|>` / `<|im_end|>` 格式，外加视觉占位符 `<|image_pad|>`。
- `ChatTemplate.ExpandImageTokens(inputTokens, imagePadId, tokenCounts)` 把每个 `<|image_pad|>` 占位符展开为对应图像所需 token 数；多模态注入器随后在这些位置写入编码后的 embedding，再调用 `Forward()`。
- **思考预算。** 开启思考时，思考内容一旦达到 `TS_THINKING_BUDGET`（`max_tokens` 不小于 512 时默认为其 75%），服务端和交互式 CLI 都会闭合思考块，答案随后在原 `max_tokens` 内生成。`</think>` 是单个训练过的 token（248069），宿主会先写入 Qwen 官方发布的交接句（"Considering the limited time by the user, I have to give the solution based on the thinking directly now."），它显示在思考内容的末尾。2026-09-29 之前该系列没有闭合 token：这样的轮次会以**空答案**结束（`finish_reason` 为 `thinking_budget`），而只写一个 `</think>` 时，Qwen3.5-9B 会在答案里继续推理。Qwen3.5-9B IQ4_XS（Metal，`max_tokens` 600，两个并发的三轮会话）：修复前 0/2 通过（答案为空），修复后 2/2。
- **没有答案就结束的思考**（issue #338）。Qwen3.8-27B NVFP4 在 Aider 下有 53 轮中的 3 轮在思考中途抽到停止 token 而结束：每一轮都已陷入计数循环（`Need maybe tests expect to_string for 1:1533 -> 02:33. Good.`，接着是 `1:1534`，持续数千个 token），OpenAI 客户端收到的是空的 `stop`。对每个有训练过的闭合 token 的系列，无论 `TS_THINKING_BUDGET` 如何，都有三处改动：
  - 思考尚未闭合时抽到的停止 token 改为闭合思考——先写交接句，再写 `</think>`——模型随后作答（`ThinkingTokenBudget.CloseOnStop`，在每个抽取 token 的位置生效：主机采样、设备 argmax 暂存和每个推测验证行）；
  - 连续 32 行重复同一模板的思考（数字串折叠，每行最多改动五分之一，因此 `twenty-nine` 变为 `thirty` 仍算重复：`ReasoningLoopDetector`）以同样方式闭合。答案中的模板行不受影响；
  - 仍以只有思考、没有答案结束的轮次（没有闭合 token 的系列）报告 `thinking_budget`，客户端看到的是 `length`。

  在 Qwen3.8-27B UD-IQ3_XXS（Metal）上，预先写入思考的 issue 循环被模型继续下去，并在生成 32 行后被闭合，答案正确（`02:33`）；思考中途的替代停止 token 被闭合后，模型对 17 x 23 回答 `391`。

## 13a. 多 GPU 放置

| 请求 | 实际运行 |
|---|---|
| `--tp N` | 执行器支持按 rank 运行时（`ggml_cuda`、直连 `cuda`）用按 rank 的图：每个 rank 持有自己的头、每个投影的分片与自己的递归状态；一个 rank 拥有一组 key 头及读取它们的 value 头，因此只需归约 `ssm_out` 与 FFN 的部分和。其他情况下每个 rank 的 Gated DeltaNet 块作为一张原生图运行（`TSGgml_Qwen35GdnLayerTP`），其余逐算子 |
| `--layer-split P` | 整层 stage，每个 stage 在其设备上一张图 |
| `--tp T --layer-split P` | 每个 stage 由 T 个 rank 组成 |
| 多节点 `--layer-split` | 每个节点加载自己的 stage（可以是各自大小的 `--tp` 组）；投机在驱动节点上起草 |
| `--n-cpu-moe N` / `--cpu-moe` | 这些层的路由专家在主机上 |

每种放置下并发请求都保持合批。`SharedRuntimeConformanceTests` 与 `Qwen35RuntimeTests` 在模拟设备上把张量并行、层切分及其混合与单设备对比；GPU 测试通道在两块真实设备上做同样的检查。

## 14. 相对被删实现的现状

2026-11-12 测量，最新 `main`（`032fdf24`，`Qwen35Model`）对比运行时，两边使用相同 CLI 脚本（预热后每个提示跑两次："short" 55 token，"4K" 3.6K token，生成 128 token，greedy），两个构建交替运行。Prefill ms / decode tok/s：

| 放置 | 模型 | `main` | 运行时 |
|---|---|---|---|
| ggml_vulkan，RTX A5000 | Qwen3.5-0.8B Q8_0，short | 39-41 / 271-278 | 45-47 / 323-327 |
| | 4K | 299-313 / 126-146 | 354-359 / 276-278 |
| | Qwen3.5-9B Q8_0，short | 152-222 / 61 | 175-198 / 62 |
| | 4K | 1,777-1,911 / 53-54 | 1,808-1,825 / 63 |
| `--tp 2` ggml_cuda，2x A40（经主机传输） | Qwen3.5-9B Q8_0，short | 105-108 / 94-95 | 112-114 / 85-86 |
| | 4K | 903-934 / 91 | 977-988 / 84 |
| | Qwen3.6-35B-A3B UD-Q4_K_M，short | 127-130 / 107-112 | 164-176 / 106-108 |
| | 4K | 950-979 / 105-106 | 900-917 / 104 |
| `--tp 2` 直连 cuda，2x A40 | Qwen3.5-9B Q8_0，short | 351-376 / 23 | 239-310 / 43-45 |
| | 4K | 14,923-15,171 / 17-18 | 6,354-7,413 / 43-45 |
| 直连 cuda，RTX A5000（非 NVFP4） | Qwen3.5-0.8B Q8_0，short | 13 / 323-324 | 38-39 / 154-163 |
| | 4K | 363 / 298 | 686-687 / 158-187 |
| | Qwen3.5-9B Q8_0，short | 51 / 66 | 111-140 / 63 |
| | 4K | 1,431-1,436 / 65 | 2,007-2,019 / 63 |
| 托管 cpu，M5 Pro，64 token | Qwen3.5-0.8B Q8_0，short | 311-367 / 105-108 | 290-300 / 73-75 |

运行时在 ggml_cuda、ggml_metal、ggml_cpu 与 MLX 单设备上的数据在这些路线迁移时已测量（[unified execution](../design/unified-execution.md)）。尚未解决的差距：

- **单 GPU 直连 CUDA**（直连引擎不服务的检查点）：运行时的 CUDA 张量执行器从主机逐个启动内核，小模型受其限制（Qwen3.5-0.8B decode 约为 `main` 的一半），其提示 GEMM 也落后于被删的融合路径（9B 4K prefill +40%）。
- **托管 CPU**：Qwen3.5-0.8B decode 落后约 30%：矩阵乘耗时相同，剩余开销在执行器的逐节点工作上。
- **Qwen3.5-9B 的 `--tp 2` decode**：落后 8-10%（35B MoE 持平）。
- **Qwen3.5-0.8B 的 ggml_vulkan 提示**：慢约 15%（decode 更快）。

