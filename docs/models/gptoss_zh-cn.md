# GPT OSS

[← 返回模型索引](README_zh-cn.md) | [English](gptoss.md)

| 属性 | 值 |
|---|---|
| 提供方 | OpenAI |
| GGUF 架构标识 | `gptoss`、`gpt-oss` |
| 模型类 | [`GptOssRuntimeModel`](../../TensorSharp.Models/Models/GptOss/GptOssRuntimeModel.cs)，运行在共享解码运行时上（`StandardDecoderModel`；旋转位置编码与路由 expert 见 [`GptOssRuntimeModel.Graph.cs`](../../TensorSharp.Models/Models/GptOss/GptOssRuntimeModel.Graph.cs)，每个执行器上都运行整次前向的计算图）。旧的 `GptOssModel` 已删除（§7、§11）。 |
| 示例模型 | gpt-oss-20b |
| 模态 | 仅文本 |
| 思维链模式 | 是（Harmony 格式：`<\|channel>analysis ... <\|channel>final`） |
| 工具调用 | 是（Harmony `commentary` channel —— `to=functions.NAME`）；可使用 skills、代码工具以及服务端的[子智能体委派](../multi_agent.md) |
| 投机解码 | n-gram 草稿（`--spec`、`--spec-type ngram`）与公开的 gpt-oss-20b DFlash 块级草稿器（`--draft-model`），在可精确回退的窗口中验证。旧模型没有投机主干。详见 §8。 |
| 批处理 / 分页前向 | **默认启用**（共享运行时的批处理槽位：每个请求有自己驻留在设备上的 K/V，每个滑动窗口层一个环形缓存；多个请求的提示块与 decode 行在同一步中运行）—— `--no-continuous-batching` 会强制走按序列路径。`--tp` 与 `--layer-split` 下保持批处理。详见 §8。 |
| 专家卸载 | `--n-cpu-moe` / `--cpu-moe` 可用于单设备或 `--layer-split` 各阶段，所有后端都支持；`--tp` 下拒绝。详见 §10。 |
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
- **Clamped GLU 激活** —— FFN 用类 SiLU 的激活 `x · sigmoid(α · x) · (y + 1)`，gate 被 clamp 到 `[-∞, 7]`，`up` 值被 clamp 到 `[-7, 7]`。
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
        │  per-expert: swiglu_oai(gate+bias, up+bias)      │
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
       ─► RoPE_NeoX with Yarn scaling
            • origCtxLen = Config.OriginalContextLength (4096)
            • freqScale  = 1 / RopeScale
            • beta_fast  = 32, beta_slow = 1
       ─► append (K, V) 到 KV cache
       ─► attention with sinks:
            scores = Q × K^T
            scores ← 因果 mask（偶数层另加 128 token 的窗口）
            sinks  = attn_sinks.weight                   # [numHeads]
            sums   = exp(scores - max) + exp(sinks - max)
            scores = exp(scores - max) / sums            # softmax 包含 sink
            out    = scores × V
       ─► attn_output.weight matmul + bias → o
       ─► residual = hidden + o

       moeInput = (提示步的最后一层) ? 各序列的最后一行 : residual
       moeInput ─► RMSNorm(post_attention_norm)
                ─► router_linear + bias
                ─► 对 logits 取 topK(4)
                ─► 对这 4 个做 softmax
                ─► 对每个被选 expert e（每个投影一次 mul_mat_id + add_id）:
                       up   = ffn_up_exps[e]   × moeInput + up_bias[e]
                       gate = ffn_gate_exps[e] × moeInput + gate_bias[e]
                       h    = swiglu_oai(gate, up)          # min(gate, 7)、clamp(up, ±7)、alpha 1.702
                       out_e = ffn_down_exps[e] × h + down_bias[e]
                ─► 用 router 权重 weighted Σ out_e
                ─► residual += weighted_sum
```

所有层完成后：

```
hidden ─► 各序列的最后一行（提示步；最后一层的 MoE 已经只在这些行上运行）
       ─► RMSNorm(output_norm.weight, eps)
       ─► output.weight matmul → logits
```

## 4. 组件细节

### 4.1 Attention

- **GQA**：gpt-oss-20b 上 64 个 query 头对应 8 个 KV 头，head 维度 64。
- **每个投影都有 bias。** 在 Q、K、V 类型相同的层上，三者在加载时逐层融合为一个 `attn_qkv` 乘积，它们的 bias 也融合为一个
  `[q | k | v]` bias，在乘积之后相加。三者类型不同的层保留三个乘积，各带自己的 bias。unsloth 的重新量化文件会混用类型，
  例如 IQ4_NL 的 q/k 配 Q5_0 的 v。输出投影加上 `attn_output.bias`。
- **没有 QK-norm。**
- **Attention sinks**，每一层都有：`attn_sinks.weight`（`[numHeads]`）是每个头的 softmax 中多出的一个 logit，没有对应的
  value。它是注意力算子的第五个操作数。GGML 构建器把它交给 `ggml_flash_attn_ext_add_sinks`。张量执行器（`cuda`、`mlx`、
  `cpu`）用通用乘积处理带 sink 的层，把 sink 作为每行 softmax 中的额外一列（`TensorGraphExecutor.FlashAttentionWithSinks`）。
- **RoPE**：NeoX 风格 + YaRN（`freq_base` 150000，在 4096 的原始上下文上系数为 32，`beta_fast` 32，`beta_slow` 1），
  作为一个 `GraphRotary` spec。
- **Attention pattern**：偶数层关注 128 token 的滑动窗口，奇数层关注整个上下文。滑动窗口层的 K/V 存放在每个槽位 768 个
  单元的环形缓存中（§9）。

### 4.2 FFN —— clamped SwiGLU（`swiglu_oai`）

```
g = min(gate + gate_bias, 7)
u = clamp(up + up_bias, -7, 7)
h = g · sigmoid(1.702 · g) · (u + 1)
out = down · h + down_bias
```

它是一个计算图算子 `GraphOp.SwigluOai`（alpha 1.702，limit 7）：GGML 上是 `ggml_swiglu_oai`，张量执行器上是托管的组合实现。
逐 expert 的 bias 是 `GraphOp.AddId`（`ggml_add_id`）：`[width, experts]` bias 的第 `e` 行加到 expert `e` 的每条路由上。

### 4.3 MoE 路由

- 32 个 expert，每 token 选 4 个（gpt-oss-20b）。
- `ffn_gate_inp · x + ffn_gate_inp.bias`，取这些 logits 的前 4 个，再对这 4 个做 softmax（llama.cpp 的 `SOFTMAX_WEIGHT`）。
  这是**先 top-k 再 softmax**，与 Gemma 4 的「先 softmax 再 top-k」相反。
- expert 直接从 GGUF 的 3-D 堆叠（`ffn_gate_exps`、`ffn_up_exps`、`ffn_down_exps`，MXFP4）原地读取，并带有各自
  `[width, experts]` 的 F32 bias。
- 计算图逐个算子沿用 llama.cpp 的 `openai-moe-iswa.cpp`：up `mul_mat_id` + `add_id`，gate `mul_mat_id` + `add_id`，
  `swiglu_oai`，down `mul_mat_id` + `add_id`，乘上各路由的权重，再求和。ggml-cuda 的融合与这条链对应：top-k MoE 选择、
  `mul_mat_id` + `add_id` + GLU，以及 expert 的加权求和。
- 在直接 CUDA 上，router 与 expert 是一个区域（`GraphFusion.OaiRoutedExperts`），由 MoE 引擎（`CudaMoe`）服务：先在设备上
  做 top-k softmax，再运行分组 expert 内核，另有两个 GPT-OSS 内核负责带 bias 的 clamped SwiGLU 与 down bias
  （`ts_moe_oai_glu_bias_f32`、`ts_moe_expert_bias_add_f32`）。decode 行与 verify 窗口运行它的逐槽位内核。
- 提示步的最后一层只在 head 读取的行上运行 expert（`PrunesExpertLastLayer`），与 llama.cpp 的计算图以及旧模型的做法相同。

### 4.4 分词器

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
blk.{L}.attn_qkv.weight  / attn_qkv.bias       # 加载后生成，q/k/v 类型相同时
blk.{L}.attn_output.weight / attn_output.bias
blk.{L}.attn_sinks.weight                      # 每个头的 sink logit [numHeads]
blk.{L}.post_attention_norm.weight             # FFN 的 norm

blk.{L}.ffn_gate_inp.weight / ffn_gate_inp.bias   # router [hidden, experts] + [experts]
blk.{L}.ffn_gate_exps.weight / ffn_gate_exps.bias # [hidden, ff, experts] + [ff, experts]
blk.{L}.ffn_up_exps.weight   / ffn_up_exps.bias   # [hidden, ff, experts] + [ff, experts]
blk.{L}.ffn_down_exps.weight / ffn_down_exps.bias # [ff, hidden, experts] + [hidden, experts]
```

## 7. TensorSharp 实现：共享解码运行时

GPT-OSS 与 Gemma 4、Qwen 3.5、Nemotron-H、Mistral 3 一样运行在共享解码运行时（`StandardDecoderModel`）上。
[`GptOssRuntimeModel`](../../TensorSharp.Models/Models/GptOss/GptOssRuntimeModel.cs) 声明该家族：
- 偶数层的滑动窗口；
- sink 与带 bias 的投影，由运行时的注意力服务；
- 以 `post_attention_norm` 作为 FFN norm；
- YaRN 旋转位置编码；
- §4.3 的路由 expert（`EmitFeedForward`）。

批处理、前缀缓存、检查点、放置、投机解码与专家卸载都由运行时提供。该家族只运行整次前向的计算图（`RequiresGraphs`）：
每一步，无论是提示块、decode 步、批处理槽位还是 verify 窗口，都是后端执行器上的一张图。该执行器在 `ggml_*` 上是 GGML
图构建器，`cuda` 上是 `CudaGraphExecutor`，`mlx` 上是 `MlxGraphExecutor`，`cpu` 上是托管执行器。

迁移新增的内容全部位于 TensorSharp 自有代码中（ggml 上游未修改）：
- **计算图 IR**：张量执行器上的 `GraphOp.SwigluOai`、`GraphOp.AddId`，以及注意力算子上的 sink 操作数。
- **提示步中的 ggml-cuda 融合**：`GraphProgram.BackendFusions` 在提示步上运行 ggml-cuda 的图优化器，先声明它所需的分配依赖。
  没有它时，提示步的 24 层中只有 1 层融合了 expert 的加权求和，其余各层以一次广播乘法加三次加法运行。
- **直接 CUDA**：MoE 引擎上的 `OaiRoutedExperts` 区域（§4.3）。
- **带 bias 的主机 expert**（§10）。
- **带 bias 投影的 DFlash 草稿器**（§8）。

旧的 `GptOssModel` 在 §11 的验证之后被删除，随之删除的还有所有只为它服务的部分：
- 它的整模型 decode、按 token 批处理的 decode 与 prefill 计算图（`ggml_ops_gptoss_decode.cpp`、
  `ggml_ops_gptoss_batched.cpp`、`ggml_ops_gptoss_prefill.cpp`、`ggml_ops_gptoss_kv.h`）；
- 融合的注意力层 prefill 及其设备端 K/V 窗口（`ggml_ops_transformer_prefill.cpp`）；
- 带 sink 的分页注意力（`TSGgml_PagedAttentionForwardWithSinks`、`ManagedPagedAttention.ForwardWithSinks`）；
- MLX 的 `gather_qmm` 堆叠 expert 路径，以及 MLX 的 sink decode、`swiglu_oai` bias-gather 与带 bias 加权求和内核；
- 它从线性缓存迁移到分页缓存时使用的 K/V 主机读取器；
- 直接 CUDA 的 sink 注意力内核（`TryAttentionSoftmaxWithSinks`、`TryGqaDecodeAttentionWithSinks`、
  `TryGqaPrefillAttentionWithSinks`）、拆分的 `swiglu_oai` 内核（`TrySwiGluOaiSplit`），以及没有调用者的
  `TryGqaDecodeAttentionRaw` / `TryGqaDecodeAttentionGgmlCuda`；
- GGML 的设备端分页注意力（`TSGgml_PagedAttentionForwardDevice`、`TSGgml_PagedAttentionForwardDeviceWithSinks`；
  主机端的 `TSGgml_PagedAttentionForward` 保留）以及 MLX 的 `gather_mm` / `gather_qmm` 绑定；
- 调度器中从线性缓存到分页存储的 K/V 迁移（`SupportsLinearKVMigration`、`TryMigrateLinearKVToPaged`），以及只有它
  才会用到的 N=1 `SingleSequenceFused` 路径和 lone-page-producer 等待。单个序列现在与并发序列一样走 `BatchedPaged`，
  在每请求缓存上则走 `PerSequenceFused`。

它的开关也一并删除：`TS_GPTOSS_FUSED_DECODE_MAX_CTX`、`TS_GPTOSS_FUSED_ATTN_MAX_SEQ`、`TS_GPTOSS_PREFILL_F16` 与
`TS_GPTOSS_PREFILL_CHUNK`（提示步遵循 `TS_PREFILL_CHUNK`，环形缓存把一步限制在 512 行）。现在设置其中任何一个都会让启动
停止，并说明这一变更（`RemovedCliFlags`），与更早删除的 `TS_GPTOSS_*` 开关相同。

## 8. 服务机制

| 机制 | 后端 | 说明 |
|---|---|---|
| 连续批处理 | 全部 | 批处理槽位：每个请求有自己的 K/V（每个滑动窗口层一个 768 单元的环形缓存）。并发请求的提示块与 decode 行在同一步中运行。 |
| 前缀缓存、检查点 | 全部 | 运行时的 holder 适配器：捕获、克隆、捐赠的 holder、导出/导入。只要环形缓存仍保留后续行的窗口要读取的位置，回退整段 Harmony 回答就是精确的（§9）。 |
| `--tp N` | cuda、ggml_cuda、ggml_vulkan | 按 rank 的计算图。注意力按 head 切分（sink 随之切分），每个 expert 按宽度切分（gate/up 按列、down 按行；down bias 只加一次）。保持批处理。gpt-oss-20b 的 MXFP4 down 行由 90 个块组成，因此 `--tp 2` 能切分它们，`--tp 4` / `--tp 8` 在加载时被拒绝并说明原因。仅在模拟设备上覆盖（§11）。 |
| `--layer-split N` | cuda、ggml_cuda、ggml_vulkan | 整层阶段，保持批处理。随运行时新增。 |
| 混合（`--tp` x `--layer-split`） | 无（缺口） | 这个只用计算图的家族还没有张量并行阶段的计算图。 |
| `--n-cpu-moe` / `--cpu-moe` | 全部 | 单设备或各阶段上可用；`--tp` 下拒绝。见 §10。 |
| 投机解码 | 全部 | n-gram 草稿与 DFlash 块级草稿器（见下文），在环形缓存可精确回退的窗口中验证。 |

**DFlash 草稿器。** [z-lab/gpt-oss-20b-DFlash](https://huggingface.co/z-lab/gpt-oss-20b-DFlash)（0.8B，Apache-2.0）
通过 `--draft-model` 加载。Q5_K_M 的 GGUF 为
[EntityDeletr/gpt-oss-20b-DFlash-GGUF](https://huggingface.co/EntityDeletr/gpt-oss-20b-DFlash-GGUF) 中的 `gpt-oss-20b-DFlash.gguf`
（0.599 GB，revision `93855ebf`）：

```bash
hf download EntityDeletr/gpt-oss-20b-DFlash-GGUF gpt-oss-20b-DFlash.gguf --local-dir models
dotnet run --project TensorSharp.Cli -c Release -- --model models/gpt-oss-20b-MXFP4.gguf \
  --draft-model models/gpt-oss-20b-DFlash.gguf --backend ggml_cuda --input prompt.txt
```

它有 8 层、8 行的块以及 q/k norm。它的 Q、K、V 与输出投影带 bias，运行时的草稿器现在会读取这些 bias
（`DfAttnQBias` ... `DfOutputBias`）。它使用目标模型的 YaRN 旋转位置编码，并读取主干第 2、7、12、17 与 22 层。

在一张 A40 上实测（`ggml_cuda`，贪心，5 个 79-96 token 的提示，每个生成 128 个 token）：
- **DFlash，7 token 窗口：** 草稿接受率 60-93 %，decode 速度为普通解码的 0.84-1.26 倍。代码与算术上最快，自由文本上最慢。
- **n-gram，4 token 窗口：** 0.95-1.11 倍。
- **一致性：** 投机流会一直跟随普通贪心解码，直到 verify 窗口的内核翻转一个接近的选择
  （`SpeculativeDecodingMatchesPlainGreedy` 覆盖合成文件）。

投机代价调节器（`SpeculationCostGovernor`）把起草与普通步对比测量，在起草不划算的地方把它停下；上面的运行都经过了它。
TensorAgent 的投机设置默认使用 n-gram 草稿，现在也会在 GPT-OSS 上投机（它的目录把 gpt-oss-20b 列为实验性）。

## 9. 内存与 KV cache

- **全注意力层**（奇数层）：每个槽位的 K/V 保存整个上下文，默认 F16（`--kv-cache-dtype`）。
- **滑动窗口层**（偶数层）：每个槽位一个 R = 768 单元的环形缓存（128 token 窗口加 512 行的步长，按 256 对齐）。位置 p 位于
  单元 p mod R。短于 R 的请求把环形缓存当作线性缓存使用。一步写入单个序列的行数从不超过 512，因此更长的提示按 512 行一步
  运行。gpt-oss-20b 的 12 个环形缓存在 F16 下每个槽位占 9.4 MB，每个 token 占 12 个全注意力层的 24 KiB。旧模型的整上下文
  缓存每个 token 占 48 KiB。
- **量化 K/V**：GGML 后端会保留 `q8_0` 请求。在 gpt-oss-20b 上，它让平均 NLL 在 `ggml_cuda` 上变化 -0.0004（384 个 token：
  2.3060 -> 2.3056），在 `ggml_cpu` 上变化 +0.008（160 个 token），`ggml_cuda` 的 decode 为 158 tok/s。张量执行器的计算图
  不在块量化缓存上运行环形缓存（直接 CUDA 的 q8_0 行写入不会回绕），因此在那里 `q8_0` 请求会变为 f16 并给出提示。`q4_0`
  请求在所有后端上都变为 f16：在 `ggml_cuda` 上它每个 token 损失 0.48 nat（2.3060 -> 2.7811）。旧模型把所有块类型都变为 f16。
- **回退**：回退会精确保留后续行的窗口要读取的位置；一旦某个环形缓存覆盖了这些位置，回退就被拒绝（`CanTruncateKVCache`）。
  Harmony 在重新渲染过去的回答时会丢掉 analysis 通道，所以下一轮要回退整段上一轮回答。只要会话持有的位置少于环形缓存的
  单元数，这次回退就是精确的，其余情况由前缀缓存的检查点处理。
  `ARewindPastALongAnswerContinuesAsACacheThatNeverHeldIt` 锁定这一行为。
- **K/V 块快照**（`TryExtractKVBlock`）不提供：环形缓存不是按位置寻址的。前缀缓存改用运行时的槽位克隆与检查点。真实模型
  一致性脚本在 gpt-oss-20b 上通过（`GptOss20B_PassesTheConformanceScript`）。
- **权重**在每个执行器上都保持 GGUF 格式（ggml-org 文件中 expert 为 MXFP4，注意力与 head 为 Q8_0）。

## 10. 专家卸载（`--n-cpu-moe` / `--cpu-moe`）

被卸载层的路由 expert 留在主机内存，这一步的计算图会为它们暂停（`GraphOp.MoeHostExperts`，带 GPT-OSS 的激活与三个
bias）。router 与专家选择留在设备上。
- **GGML 后端：** 主机 MoE 段用 ggml 的 CPU 内核直接从 GGUF 映射计算它们：带 bias 的 gate 与 up 乘积、clamped SwiGLU，
  然后是带 bias 的 down 乘积。decode 行由单 token decode 团队处理；提示块由批处理的 CPU MoE 计算图处理。
- **张量执行器：** 运行时的 expert host 用同样的内核计算它们（`TSGgml_HostExpertRowsOai`，经由同一个 decode 团队）。
- **直接 CUDA** 把一步留在设备上：
  - decode 行与短提示块通过 expert doorbell（`CudaExpertDoorbell`）通知主机，主机计算它负责的路由，同时设备运行自己的
    路由。之后设备为每条路由（包括主机的）加上 down bias。
  - 一层的一次调用最多经链路把一个主机 expert 传到设备上（`OaiGpuMisses`）：一个 gpt-oss-20b expert 是 13.2 MB 的 MXFP4。
    实测 decode 速度：四条路由全部传送时 14.0 tok/s，两条时 28.7，一条时 31.6，不传送时 29.7。
  - 32 行及以上的提示块在当前层运行时通过 DMA 预取下一层的全部主机 expert（`OaiPrefetchMinRows`）：这么长的块本来就会
    选中约 98 % 的 expert。随后分组 MoE 内核在设备上运行它们。

`--tp` 下卸载在加载时被拒绝：按 rank 的计算图把 expert 保留在设备上。

在一张 A40 上实测（ggml `bb877c0c`），卸载 24 个 MoE 层中的 12 个（`TS_N_CPU_MOE=12`），容器有 7.6 个 CPU
（5 个主机 expert 线程）：

| 后端 | Prefill 512 tok/s | Decode tok/s | NLL 常驻 -> 卸载 |
|---|---|---|---|
| ggml_cuda，旧模型 -> 运行时 | 1814-2415 -> 2116-2460 | 33.3-39.5 -> 34.0-41.4 | — |
| cuda，运行时 | 551（2048 时 547） | 29.2-31.6 | 2.7706 -> 2.7700 |

范围跨两次会话。旧模型在 `cuda` 上忽略 `TS_N_CPU_MOE`：它把所有 expert 留在 GPU 上（prefill 723 tok/s、decode 51 tok/s），因此直接 CUDA 的
卸载是新增的。`cuda` 上带卸载的贪心文本一直跟随常驻运行，直到接近平局处（第一处差异的前两名差距为 0.03-0.07）。

## 11. 迁移验证

依赖：ggml `bb877c0c51f38c52ee2d5e41d58e3d0c20f3fed4`，上游且未修改。`ggml_swiglu_oai`、`ggml_add_id` 与
`ggml_flash_attn_ext_add_sinks` 是它自己的算子。CUDA PTX 基线已用 CUDA 13.0.88 重新构建，以包含两个新内核。

**合成 fixture**（`GptOssSyntheticModelBuilder`、`GptOssRuntimeTests`，41 个测试）。四种情形：MXFP4 expert、Q8_0 expert、
一个长序列情形（83 个 token 上的窗口 16），以及一个按 `--tp` 设定尺寸的情形。每种都与旧模型的 logits 对比，这些 logits
在旧模型删除前于托管后端上冻结（`Fixtures/GptOss/LegacySynthetic/managed.json`）。`ggml_cpu`、`ggml_cuda` 与直接 `cuda`
都与托管后端对比。还覆盖了：
- 分块 prefill、模拟设备上的每种放置、批处理槽位、并发请求；
- 检查点、越过长回答的回退、投机解码；
- 每个执行器上的主机 expert（直接 CUDA 上：decode 行走 doorbell，提示块在设备上暂存 expert，没有任何一步回退到执行器的主机路径）；
- 混合 Q/K/V 类型的逐层融合；
- `--tp` 下拒绝卸载。

直接 CUDA prefill 测试套件（`DirectCudaPrefillModelTests`）新增了 GPT-OSS 的 verify 行，它们与其 decode 行逐位一致。

**真实 checkpoint**（gpt-oss-20b MXFP4，一张 A40）。与旧模型的教师强制 argmax 一致率在 5 个提示的贪心步上统计，旧模型的
token 同时强制输入两者。NLL 取 README 的 448 个 token，以困惑度报告。速度来自 CLI 的基准，按 ABBA 顺序测量：

| 后端 | 与旧模型 argmax 一致 | 困惑度 旧 -> 运行时 | Decode tok/s（512 之后） | Prefill 512 tok/s | 4K 之后的 Decode tok/s | Prefill 4K tok/s |
|---|---|---|---|---|---|---|
| ggml_cuda | 551/561（接近平局，前两名差距 < 0.15） | 10.579 -> 10.546 | 141-150 -> 155-162 | 6095-6348 -> 6191-6531 | 131 -> 157.5 | 5923-6447 -> ~6440 |
| cuda | 555/561 | 10.620 -> 10.576 | 51 -> 118-120 | ~730 -> ~1750 | — | — |

- **Logits** 与 F32 托管 CPU 参考对比：旧模型与运行时同样接近（KL ≈ 0）。
- **服务端**（`TensorSharp.Server`，总 decode tok/s）：

  | 请求数 | ggml_cuda 旧 -> 运行时 | cuda 旧 -> 运行时 |
  |---|---|---|
  | 1 | 107.8 -> 140.6 | 39.4 -> 89.3 |
  | 4 个并发 | 125.0 -> 212.7 | 4.7 -> 127.7 |
  | 8 个并发 | 151.0 -> 230.4 | 4.7 -> 132.3 |

  并发输出与单独输出在接近平局的措辞选择处分叉，旧模型也是如此：旧模型的并发输出在 4 个提示中有 1 个、8 个中有 4 个与单独输出一致。
- **覆盖限制。** 以下没有运行：
  - `mlx`、`ggml_metal` 与 `ggml_vulkan`；
  - 多 GPU 的 `--tp` 与 `--layer-split`，只在模拟设备上覆盖；
  - gpt-oss-120b。
- **已知代价。**
  - gpt-oss-20b 上 `--tp 4` / `--tp 8` 被拒绝，而旧模型以 expert 并行的方式运行它们。`--layer-split N` 支持任意 N。
  - 直接 CUDA 上超过滑动窗口之后的 decode 步以计划（plan）运行，而不是重放 CUDA graph，因为录制会重放它第一个窗口的
    注意力下界。输出是精确的，但每一步都要在主机上付出启动开销（118-120 tok/s，ggml_cuda 为 155-162）。
  - 直接 CUDA 在 sink 层上的提示注意力走通用乘积（1750 tok/s，ggml_cuda 为 6400）。
  - 直接 CUDA 上被卸载的提示达到 551 tok/s，ggml_cuda 为 2460（§10）。
  - 在合成文件上，直接 CUDA 的分块 prefill 与单次前向最多相差 3.0e-2：行数与 decode 相同的块运行 decode 内核。旧模型把这一
    差距保持在 7.5e-3 以内。与 ggml_cpu 对比时，运行时保持在旧模型所受的界限之内。

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

- **直接 CUDA 上带 sink 的提示注意力。** sink 层的提示注意力走通用乘积。一个在 softmax 中包含 sink 的 flash 内核能弥补与
  ggml_cuda prefill 之间的大部分差距。旧模型的 sink 内核从未接到执行器的环形缓存上，
  已随旧模型一起删除，因此需要给支持环形缓存的 flash 内核加上 sink 操作数。
- **直接 CUDA 上超过窗口之后的录制 decode。** 把环形缓存的下界作为输入录制，就能让这些步重放。
- **直接 CUDA 上被卸载的提示。** gpt-oss-20b 2880 宽的 expert 走分组 MoE 引擎的每行一个 warp 的路线（2880 既不是 128 的倍数，
  也不是 256 的倍数）。为这个宽度提供一条 tensor-core 路线，可以同时服务被卸载与常驻的提示 expert。
- **`--tp 4` / `--tp 8`** 需要在一个 MXFP4 行的 90 个块内部切分 down 投影的行，或者一种 expert 并行的放置。
