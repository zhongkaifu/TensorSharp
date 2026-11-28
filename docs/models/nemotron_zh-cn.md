# Nemotron-H

[← 返回模型索引](README_zh-cn.md) | [English](nemotron.md)

| 属性 | 值 |
|---|---|
| 提供方 | NVIDIA |
| GGUF 架构标识 | `nemotron_h`、`nemotron_h_moe`、`nemotron_h_omni` |
| 模型类 | [`NemotronRuntimeModel`](../../TensorSharp.Models/Models/Nemotron/NemotronRuntimeModel.cs)，运行在共享解码运行时上（`StandardDecoderModel`；Mamba-2 混合器与专家见 [`NemotronRuntimeModel.Graph.cs`](../../TensorSharp.Models/Models/Nemotron/NemotronRuntimeModel.Graph.cs)，递归状态见 [`NemotronRuntimeModel.State.cs`](../../TensorSharp.Models/Models/Nemotron/NemotronRuntimeModel.State.cs)）。旧的逐算子 `NemotronModel` 已删除（§7、§11）。 |
| 视觉编码器 | [`NemotronVisionEncoder`](../../TensorSharp.Models/Models/Nemotron/NemotronVisionEncoder.cs)（RADIO / v2_vl ViT） |
| 图像处理器 | [`NemotronImageProcessor`](../../TensorSharp.Models/Models/Nemotron/NemotronImageProcessor.cs) |
| 音频前端 | [`NemotronAudioPreprocessor`](../../TensorSharp.Models/Models/Nemotron/NemotronAudioPreprocessor.cs)（Parakeet 风格 log-mel） |
| 音频编码器 | [`NemotronAudioEncoder`](../../TensorSharp.Models/Models/Nemotron/NemotronAudioEncoder.cs)（Parakeet/FastConformer + 声音投影器；需要任何公开仓库都不提供的配套 GGUF，见 §4.7） |
| 示例模型 | Nemotron-H-8B-Reasoning-128K、Nemotron-H-47B-Reasoning-128K、Nemotron 3 Nano Omni、Nemotron 3.5 Bolt 74B-A9B 与 Super VL 120B-A12B（见[下文](#nemotron-35-super-vl-与-bolt)） |
| 模态 | 文本、图像（Omni 版本配合 `mmproj`）。只有加载了带 Parakeet 音频塔的配套 GGUF 时才支持音频（§4.7）；否则音频会被**拒绝**（HTTP 400 / CLI 错误，消息为 `NemotronRuntimeModel.AudioInputUnsupportedMessage`）：公开的 Omni GGUF 不带音频塔，`mmproj` 里只有 RADIO 视觉塔（见 §4.6）。 |
| 思维链模式 | 是（`<think> ... </think>`） |
| 工具调用 | 是：Reasoning-128K 用 `<TOOLCALL>[...]</TOOLCALL>`（JSON 或 Python 风格的调用），Nemotron 3 Nano / Omni 用调用体为 XML `<function=...>` 的 `<tool_call>`（§12）；可使用 skills、代码工具以及服务端的[子智能体委派](../multi_agent.md) |
| 批处理 / 分页前向 | **默认启用**（共享运行时的批处理槽位：每个槽位的 Mamba-2 conv 窗口与 SSM 状态，注意力层使用分页 K/V）—— `--no-continuous-batching` 会强制走按序列路径。详见 §8。 |
| 投机解码 | n-gram 草稿、Nemotron 3.5 MTP 头（主干内的 NextN 块，或通过 `--draft-model` 加载的 MTP GGUF），以及公开的 Nemotron 3.5 Lightning DFlash 与 DSpark 块级草稿器（`--draft-model`），在 Mamba-2 状态可精确回退的窗口中验证。详见 §8。 |
| 专家卸载 | `--n-cpu-moe` / `--cpu-moe` 可用于单设备或 `--layer-split` 各阶段，所有后端都支持；`--tp` 下拒绝。详见 §10。 |
| 输出解析器 | `NemotronOutputParser`（一种 `ChatMlOutputParser`） |

## 下载

已验证的 GGUF 下载指引：

| 模型 | HF 仓库 | 推荐文件 | mmproj |
|---|---|---|---|
| Nemotron-H-8B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF) | `nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf`（4.983 GB）或 `nvidia_Nemotron-H-8B-Reasoning-128K-Q8_0.gguf`（8.620 GB） | —（仅文本） |
| Nemotron-H-47B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF) | `nvidia_Nemotron-H-47B-Reasoning-128K-Q4_K_M.gguf`（28.188 GB） | —（仅文本） |
| Nemotron 3 Nano Omni 30B-A3B | [unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF) | `NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf`（23.927 GB） | `mmproj-BF16.gguf`（1.590 GB；同仓库）—— **图像输入必需** |
| Nemotron 3.5 Bolt 74B-A9B | [nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF) | `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf`（22.018 GB） | `mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf`（1.734 GB；同仓库） |
| Nemotron 3.5 Lightning 30B-A3B | [ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) | `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf`（18.898 GB）。同仓库的草稿器：`mtp-...-Q4_0.gguf`（1.156 GB）、`dflash-...-NVFP4.gguf`（1.185 GB）、`dspark-...-NVFP4.gguf`（1.357 GB） | —（仅文本） |
| Nemotron 3.5 Super VL 120B-A12B | [ggml-org/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF) | `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4.gguf`（76.919 GB）或 `-Q4_0.gguf`（71.934 GB） | `mmproj-NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q8_0.gguf`（0.967 GB）或 `-BF16.gguf`（1.734 GB；同仓库） |

Omni 的 `mmproj` 只启用**图像输入**：它只包含 32 层 RADIO 视觉塔
（`v.blk.*`、`v.patch_embd`、`v.position_embd`）和 `nemotron_v2_vl` 的 MLP
投影器（`mm.model.mlp.*`），元数据只有 `clip.has_vision_encoder`，没有任何音频
项；语言模型 GGUF 也只有 `<so_embedding>` 占位 token。因此只加载这些文件时，音频附件会在
请求层被**拒绝**（HTTP 400），而不是被解码后丢弃（见 §4.6）；音频需要单独准备的配套 GGUF（§4.7）。

这些转换仓库将 NVIDIA 对应的 Nemotron 仓库标记为上游。上游模型卡使用
NVIDIA 特定条款（`other`）；两个 bartowski 转换仓库未声明许可证。再分发前请阅读 NVIDIA 基础模型条款。

命令行下载（每个文件一行；需要先 `pip install -U huggingface_hub`）：

```bash
python -m pip install -U huggingface_hub
hf download bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --local-dir models
hf download bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF nvidia_Nemotron-H-47B-Reasoning-128K-Q4_K_M.gguf --local-dir models
hf download unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf --local-dir models
hf download unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF mmproj-BF16.gguf --local-dir models
```

CLI 单次推理（只给 `--image` 而不给 `--input` 时会使用默认的描述图片提示词；
CLI 采样默认为 greedy，`--max-tokens` 默认为 100）：

```bash
dotnet run --project TensorSharp.Cli -c Release -- --model models/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf \
  --mmproj models/mmproj-BF16.gguf \
  --image photo.png --max-tokens 512 --backend ggml_cuda
```

服务端（聊天 Web UI 以及 OpenAI/Ollama 兼容 API，位于 `http://localhost:5000`）：

```bash
dotnet run --project TensorSharp.Server.Host -c Release -- --model models/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf \
  --mmproj models/mmproj-BF16.gguf --backend ggml_cuda --max-tokens 4096
```

### Nemotron 3.5 Super VL 与 Bolt

两者都是 `nemotron_h_moe` checkpoint，共 88 层：40 层 Mamba2、8 层注意力（32 个 query 头对应 2 个 KV 头，
head_dim 128）、40 层 latent MoE（hidden 4096，MoE latent 1024，带共享 expert），并都附带 RADIO 视觉
mmproj。支持的来龙去脉与验证情况见[设计说明](../design/nemotron-super-vl-bolt-support.md)。

| | Bolt 74B-A9B | Super VL 120B-A12B |
|---|---|---|
| 路由 expert | 逐层配置（"Puzzle"）：240-496 个 expert、top-9、n_ff 1536-2560 | 512 个 expert、top-22、n_ff 2688 |
| 语言模型 GGUF | `...-Bolt-74B-A9B-24GB.gguf` **22.018 GB**（IQ2_XXS expert；Mamba2 投影为 NVFP4，带 `.scale` sidecar） | `...-NVFP4.gguf` **76.919 GB**（NVFP4 expert，带逐 expert scale）；`...-Q4_0.gguf` **71.934 GB**（无 sidecar） |
| mmproj | 1.734 GB；q/k/v **分开存放**（加载时融合） | Q8_0 0.967 GB 或 BF16 1.734 GB；融合 q/k/v，另有最终 LayerNorm `v.post_ln`（在所有 block 之后应用） |

**内存。** 语言模型 GGUF 必须放进设备，并给 KV cache 和工作区留出空间：
- **Super VL NVFP4** 单设备约需 71.6 GiB（76.919 GB）权重。它在 128 GB 的 DGX Spark（GB10）上运行过，放不进
  48 GB GPU，所以 PR #282 的评审在 `ggml_cpu` 上运行它。单张 96 GB GPU 应能容纳，但未验证。
- **Bolt-24GB** 在 DGX Spark 上以 `ggml_cuda` 和 `cuda` 运行过；PR #282 的评审在一张开启 ECC 的 48 GB A40（约 45.0 GiB 可用）上运行过它。
- **视觉塔**以计算图程序在语言模型的执行器上运行（§4.5），权重以文件编码放在模型的分配器上（BF16 mmproj 约 1.7 GB）。
  它的托管副本（主机上约 3.2 GiB 的 F32）只在加载校验期间存在。

**`--tp`。** 注意力层只有 2 个 KV 头，因此只有 `--tp 2` 可用。张量并行以共享运行时的按 rank 计算图运行（§8）：每个
rank 持有一段连续的 Mamba-2 group 及读取它们的 head、一个 KV 头及其 16 个 query 头，以及每个 expert 宽度的一个切片；
router 与专家选择在每个 rank 上运行。NVFP4 scale 跟随分片（列切分的局部结果在归约之前乘上 scale；逐 expert 的 down scale
折叠进路由权重，逐 expert 的 up scale 在 ReLU^2 之前乘到每条路由的 up 乘积上）。逐层的 expert 数与 top-k（Bolt）各层独立。
这些由模拟设备上的合成 fixture 覆盖（`NemotronRuntimeTests`、`NemotronExpertScaleTests`、`NemotronPerLayerMoEConfigTests`）；
两个模型都没有记录到多 GPU 运行。

**`--n-cpu-moe` / `--cpu-moe`** 在单设备上把被卸载层的路由 expert 留在主机内存（§10）：在一张 A40 上，Bolt-24GB 卸载全部
层后峰值显存为 6.5 GB。`--tp` 下拒绝该选项。

**单张 48 GB A40 上的 Super VL**（`--cpu-moe`：每层的路由 expert 都在主机内存；ggml `bb877c0c`）。测试虚拟机的 cgroup 配额为
7.65 个 CPU（`nproc` 报告 96），它限制了主机 expert 的速度，所以这些速度只代表这台机器：

| 后端 | NLL（64 个之后的 256 个 token） | Prefill 512 | Decode | 峰值显存 |
|---|---|---|---|---|
| `ggml_cuda` | 2.1167（ppl 8.30，top-1 57.0 %） | 24.2 tok/s | 3.85 tok/s | 14.3 GB |
| `cuda` | 2.1156（ppl 8.29，top-1 56.6 %） | 6.4 tok/s | 3.38 tok/s | 17.2 GB |

运行结果：
- **贪心回答。** 五个提示（一个事实、Python 代码、三条要点、算术、一段翻译）在两个后端上都回答正确。
- **图像。** BF16 mmproj 对 640x480 测试图像的描述在两个后端上都正确：左上的红色正方形、右上的蓝色圆形、底部中间的绿色三角形，
  以及文字 "HELLO 42"。
  - 使用 `--no-skills` 时提示为 289 个 token；CLI 默认的技能目录另外增加约 2,190 个。
  - `ggml_cuda` 在 38.4 秒内 prefill 了带技能目录的 2,475 个 token，decode 为 2.9 tok/s。
  - `cuda`（下文修复之后）在 117 秒内 prefill 了 289 个 token，decode 为 2.3 tok/s。直接 CUDA 把被卸载提示块的 expert 一次一个地
    流式传输到设备上（§10）。
- **route-tail skip。** 第一次 `cuda` 运行得到 2.2100。逐 token 与托管 CPU 后端对比（32 个 token 上 2.1195；`ggml_cuda` 为
  2.1156，逐 token 平均 |delta| 0.007），直接 CUDA 从第一个 token 起就偏离（平均 |delta| 0.14）。
  - 原因：张量执行器的 expert host 对 Nemotron 的单投影 expert 应用了 route-tail skip（`TS_MOE_ROUTE_TAIL_SKIP`，排名 7），
    每个 token 的 22 条路由中有 15 条被略去；而 GGML 不为这些 expert 保留分层，会计算全部路由。
  - 修复：张量执行器不再对 Nemotron 的 ReLU^2 expert 应用该略过，并且只在 decode 与 verify 宽度（8 行）上应用它，与文档
    一致。此后 `cuda` 与 CPU 一致（平均 |delta| 0.009）。
  - `NemotronBf16ProjectionTests` 用一个合成的 top-22 模型锁定这一点。

```bash
hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf --local-dir models
hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf --local-dir models
dotnet run --project TensorSharp.Cli -c Release -- --model models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf \
  --mmproj models/mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf --image photo.png --backend ggml_cuda
```

## 1. 来源与目标

Nemotron-H 是 NVIDIA 的混合 **Mamba2 + Transformer** 系列。同一套 backbone 同时覆盖密集 `nemotron_h` 系（如 Nemotron-H-8B / 47B）与 MoE `nemotron_h_moe` 系。Omni 发布（Nemotron 3 Nano Omni）额外携带 RADIO / v2_vl 视觉编码器（通过 `mmproj` 提供）。TensorSharp 还为 Omni 系实现了 Parakeet 风格的音频预处理器及其音频塔（24 层 Parakeet/FastConformer 编码器加声音投影器，`NemotronAudioEncoder`），但这些权重不在任何公开 GGUF 里：只用公开文件时图像是唯一实际可用的额外模态，音频会被拒绝（见 §4.6）；由 NVIDIA 检查点转换得到的配套 GGUF 可以启用音频（见 §4.7）。

它的核心特征：

- **同一个堆栈中三种层类型** —— Mamba2 SSM 层、纯注意力层、纯 FFN 层（可选 MoE）。每层类型由 GGUF 元数据数组决定：`head_count_kv[l]` 与 `feed_forward_length[l]` 一起决定层类型。
- **Mamba2 SSM** —— 选择性状态空间模型，分组 head。conv 状态与 SSM 状态按层、按 token 维护。
- **没有 RoPE** —— 注意力层不带位置编码。位置信息由 SSM 状态隐式承载。
- **ReLU² FFN** 激活（`max(0, x)²`）。
- **Sigmoid 路由 MoE** —— 用 per-expert sigmoid 概率（可选加性 bias）做 expert 选择，TopK，可选归一化 / 缩放。
- **可选 latent bottleneck 与 shared expert** —— MoE FFN 可有 in / out latent bottleneck（`ffn_latent_in` / `ffn_latent_out`，Super / Bolt 的 GGUF 中名为 `ffn_latent_down` / `ffn_latent_up`）和并行的 ReLU² 共享分支（`ffn_up_shexp` / `ffn_down_shexp`）。

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
        │   GroupRMSNorm(y * SiLU(z)) ─► ssm_out         │
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

每层是一个 block（§7），在该步的计算图中降低为：

```
if Mamba2:
  hidden ─► RMSNorm(attn_norm)
         ─► ssm_in.weight matmul ─► 拆为 z, xBC, dt
         ─► ssm_conv over [conv window | xBC] with ssm_conv1d.weight (+ bias；窗口是该 slot 的状态)
         ─► SiLU
         ─► 对该 chunk 的所有行做 selective scan（ggml_ssm_scan；状态是该 slot 的）：
                state ← state * exp(-softplus(dt) * A) + B * x * dt
                y     ← state @ C  (+ x * D if D present)
         ─► out = GroupRMSNorm(y * SiLU(z), ssm_norm.weight)
         ─► out matmul ssm_out.weight → o
         ─► residual = hidden + o

if Attention:
  hidden ─► RMSNorm(attn_norm)
         ─► attn_qkv.weight matmul ─► 拆 Q, K, V (per-layer head 数)
         ─► append (K, V) 到 KV cache
         ─► attention(Q, KCache, VCache, scale = attention.scale or 1/sqrt(headDim))
         ─► attn_output.weight matmul → o
         ─► residual = hidden + o

if FFN (dense):
  hidden ─► RMSNorm(attn_norm)
         ─► ffn_up.weight matmul → x
         ─► x ← max(0, x)²
         ─► ffn_down.weight × x → o
         ─► residual = hidden + o

if FFN (MoE):
  hidden ─► RMSNorm(attn_norm)
         ─► logits = ffn_gate_inp × hidden
         ─► probs  = sigmoid(logits)
         ─► idx    = topK(probs + exp_probs_b)            # bias 只影响选择
         ─► weights = probs[idx]; if expert_weights_norm: weights /= max(sum(weights), 2^-14)
         ─► weights *= expert_weights_scale（以及每个 expert 的 NVFP4 down scale）
         ─► [if latent] ffn_latent_in × hidden → latent
         ─► 对每个被选 expert e（每个投影一次 ggml_mul_mat_id）:
                 x    = ffn_up_exps[e] × (hidden or latent)   （× 该 expert 的 NVFP4 up scale）
                 x   ← max(0, x)²
                 down = ffn_down_exps[e] × x
                 accum += weights[e] * down
         ─► [if latent] accum = ffn_latent_out × accum
         ─► [if shared expert] accum += ffn_down_shexp × max(0, ffn_up_shexp × hidden)²
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

- **z** —— gate，维度 `d_inner`（`ssm.inner_size`）。
- **xBC** —— 联合的 `(x, B, C)` stream，维度 `d_inner + 2 * groups * d_state`。
- **dt** —— 选择性 timestep，维度 `heads`（`ssm.time_step_rank`）。

`xBC` 接着走 conv1d 滑动窗口（kernel size `ssm.conv_kernel`，窗口是序列循环状态的一部分）。SiLU 之后 SSM scan 对每行计算：

```
ΔA   = exp( -softplus(dt + ssm_dt.bias) * A )           # A ∈ ℝ^{nHead}
ΔB   = (B * x * dt)                                      # B ∈ ℝ^{nGroup × dState}
state = ΔA · state + ΔB                                  # state ∈ ℝ^{dState × headDim × nHead}
y    = state · C                                         # C ∈ ℝ^{nGroup × dState}
y   += x * ssm_d                                         # 当 `ssm_d` 存在时
```

运行时把它作为一次 `GraphOp.SsmScan`，覆盖一个 chunk 的所有行（§7）。gate `SiLU(z) * y` 之后，`GroupRMSNorm(y, ssm_norm.weight)` 做 per-group 归一。最后 `ssm_out` 投影回 hidden 维度。

### 4.2 Attention 层

- 标准 GQA。Per-layer head 数与 KV head 数从 GGUF 数组读取。
- 当 `attn_qkv.weight` 存在时使用融合 QKV，否则运行时在加载时融合分散的 `attn_q/k/v.weight`。
- Attention scale 是 `nemotron_h.attention.scale`（如设置），否则 `1/sqrt(headDim)`。
- **没有 RoPE** —— SSM 状态隐式携带位置信息，所以 attention 不需要位置编码。

### 4.3 FFN 层（dense）

Dense FFN 单次 `ffn_up` matmul 后接 ReLU²（`x ← max(0, x)²`）与 `ffn_down`。

### 4.4 FFN 层（MoE）

- **Routing**：`sigmoid(logits) + exp_probs_b`（per-expert 加性选择 bias）→ top-k → 以被选 expert 未加 bias 的概率作为权重，
  设置了 `expert_weights_norm` 时除以它们的和（至少 2^-14）做归一化，再乘以 `expert_weights_scale`。所有路径（提示块、decode、
  批处理槽位、verify 窗口、每个 rank）共用同一套计算图降级，因此它们选出相同的 expert，顺序也相同。
- **Latent bottleneck**（可选）：`ffn_latent_down`（`ffn_latent_in`）把 hidden 投影到更小的 `latentDim`，experts 在 latent
  空间运行，然后 `ffn_latent_up`（`ffn_latent_out`）投回 hidden。
- **Shared expert**（可选）：由 `ffn_up_shexp` 与 `ffn_down_shexp` 构成的 ReLU^2 分支，加到路由求和上；计算图把它标记为
  执行器可以与 router 并行运行的并发分支。
- **Experts**：对堆叠的 `ffn_up_exps` / `ffn_down_exps` 使用 `ggml_mul_mat_id`（每个投影一次乘积，覆盖所有行的全部被选
  expert），并应用上文所述的 NVFP4 逐 expert scale。

### 4.5 视觉编码器（`NemotronVisionEncoder`）

镜像 NVIDIA RADIO / CLIP 风格 ViT，对应 v2_vl projector：

- **Linear patch embedding**，可选 bias。
- **Position embedding** 存为 `[hidden, posTokens]` 网格，源 patch 网格不同时按 align-corners=false 的双线性 resize。Resize 后的 embedding 按 `(gridW, gridH)` 在 `_resizedPositionEmbeddings` 缓存。
- **Class embedding**（可选）前置到序列。
- **32 个 encoder block**（Nemotron Omni 默认）：LayerNorm → 融合 QKV self-attention → residual → LayerNorm → up linear → GELU → down linear → residual。Bolt 的 mmproj 把 Q、K、V（权重和 bias）分开存为 `attn_q` / `attn_k` / `attn_v`。加载时按 Q、K、V 的行顺序把它们拼接成 `attn_qkv`，所以分开存放和融合存放的文件运行的是相同的权重。同一 block 同时有两种布局，或者分开的三元组缺了一部分，都会被拒绝。
- encoder 之后 class token 被剥离，按 `scaleFactor` pixel-shuffle 降低空间分辨率。
- projector（RMSNorm → Linear → ReLU² → Linear）输出 LM hidden 维度的 embedding。
- **加载时检查。** forward 读取的每个张量都必须存在，且与 `clip.vision.*` 元数据一致。projector 的输出宽度必须等于语言模型的 hidden size（Nano Omni 的 mmproj 投影到 2688，Bolt 和 Super VL 投影到 4096）。否则加载以 `InvalidDataException` 失败，消息中给出文件名、出问题的张量和期望的布局。带 `--mmproj` 启动的服务端或 CLI 会拒绝启动（退出码 2）。增加这项检查之前，这样的文件能加载，但之后每个图像请求都会失败。
- **最终 LayerNorm。** Super VL 的 mmproj 以 `v.post_ln`（weight 与 bias）结束视觉塔，在所有 block 之后作用于每个 token，
  与 llama.cpp 的 `build_vit` 相同；bias 可选。
- **不使用：** 视频塔的 `v.patch_embd.weight.1`（加载时跳过，打印一行日志）。
- **计算图程序。** 视觉塔与投影器在语言模型的执行器上以每个 tile 一个计算图程序运行，覆盖所有后端，包括直接 CUDA
  （`NemotronVisionEncoder.Graph.cs`）。patch 行仍在主机上打包，缩放后的位置表按网格缓存，embedding 返回到视觉塔分配器上。
  托管逐算子路径在加载时读取并校验一次权重，然后释放它们；没有执行器的分配器仍由它服务。`NemotronVisionEncoderTests`
  在训练网格和缩放后的网格上，把融合、拆分（Bolt）与 Super VL 三种布局的计算图程序与托管路径对比：托管 CPU 上相对 L2
  不超过 1e-5，直接 CUDA 上不超过 1e-3，`ggml_cpu` 与 `ggml_cuda` 上不超过 2e-3（GGML 的注意力把 K 和 V 舍入为 F16）。

`NemotronImageProcessor`（镜像 `nemotronh.ImageProcessor`）：

1. RGBA 合成到白底。
2. 选择最匹配源图像比例的 tile 网格（最多 `maxTiles`），如有 min/max patches 元数据则走动态分辨率模式。运行时默认把 tiled 图像限制为 1 个 tile，以降低服务器图像聊天的首 token 延迟；设置 `TS_NEMOTRON_IMAGE_MAX_TILES=12`（或模型声明的最大值）可恢复完整分辨率 tiling。
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

聊天模板对每个上传音频文件发出一个 `<so_embedding>` token，但没有任何东西能填充它：把 log-mel 帧变成 embedding 的 Parakeet/FastConformer 编码塔及其后面的投影器都不在公开的 Nemotron 3 Nano Omni GGUF 里。unsloth 仓库的 `mmproj-BF16.gguf` 恰好 390 个张量 —— 32 个 `v.blk.*` 视觉块、`v.patch_embd` / `v.position_embd` / `v.class_embd` 以及三个 `mm.model.mlp.*` 投影器张量 —— 元数据 `clip.has_vision_encoder=true`，没有任何音频键；401 个张量的语言模型 GGUF 同样没有音频张量。上游 llama.cpp 对同一检查点的回答是 "This model does not support audio input"；那边的音频需要社区分支加上带音频塔的统一 `mmproj`。

因此除非从配套 GGUF 加载了音频塔（§4.7），TensorSharp 对该家族**拒绝**音频，而不是像以前那样解码音频、打印警告、然后当作没有音频继续生成（那样模型看到的是未填充的 `<so_embedding>`，只按文本作答）。一张表 `AudioInputSupport.UnsupportedReasonFor` 通过架构注册表按架构查找，覆盖全部别名（`nemotron_h`、`nemotron_h_moe`、`nemotron_h_omni`），并参考已加载模型的 `NemotronRuntimeModel.IsAudioEncoderLoaded`，驱动每一个入口：

- `/v1/chat/completions` 与 `/v1/responses` 在写入任何上传之前扫描整个请求，对任何 `input_audio` / `audio_url` 部分（包括畸形的、以及排在图像之后的）返回 **400** `invalid_request_error`，消息为 `NemotronRuntimeModel.AudioInputUnsupportedMessage`；只有图像的请求仍正常解析。
- Web UI 在打开流之前以同一消息返回 400。
- CLI 在模型加载完成后、解码音频或生成任何一轮之前拒绝 `--audio`，REPL 的 `/audio` 拒绝挂载该文件。
- `ModelMultimodalInjector.ProcessNemotronHistory` 对绕过上述关卡的调用者抛出带同一消息的 `NotSupportedException`。

`NemotronAudioRefusalTests` 覆盖这些关卡（有无已加载的音频塔两种情况），`NemotronOmniMmprojContractTests`（由 `TS_TEST_NEMOTRON_MMPROJ` 门控）钉住公开 mmproj 只含视觉塔的布局；如果将来某个发行版在该文件里带上了音频塔，它会最先失败。

### 4.7 音频塔（`NemotronAudioEncoder`，配套 GGUF）

`NemotronAudioEncoder` 运行 NVIDIA 的 Parakeet/FastConformer 编码器（下采样卷积、相对位置注意力、卷积块）以及把输出投影到语言模型隐藏维度的 `sound_projection` MLP，每段音频单独编码，因此相邻音频的填充既不会改变它的长度，也不会泄漏进它的双向注意力。它读取一个**配套 GGUF**：保留官方 `sound_encoder.encoder.*` / `sound_projection.*` 张量名，超参数放在 `nemotron.audio.*`（`general.architecture=nemotron_audio`）。没有任何公开仓库提供这样的文件；从 NVIDIA BF16 检查点提取它的转换脚本保存在 [`eng/nemotron-audio`](../../eng/nemotron-audio/README.md)（`prepare.py`；`prepare_f32.py` 写出同样的权重并设置 `nemotron.audio.compute_bf16=false`）。

加载看的是张量，而不是开关。`LoadProjectors` 把 `--mmproj` 路径交给两个塔：文件含 `v.*` 张量时才加载视觉编码器，含 `sound_projection.linear2.weight` 时才加载音频编码器（若 `TS_NEMOTRON_AUDIO_MMPROJ` 指定了文件，则改从该文件加载，这样视觉 mmproj 与音频配套文件可以同时使用）。编码器随后校验每个张量形状与 Parakeet 采样配置，投影宽度必须等于语言模型隐藏维度；残缺或不匹配的配套文件会以 `InvalidDataException` 使加载失败，而不是被继续使用。只有这之后 `IsAudioEncoderLoaded` 才会解除 §4.6 的拒绝，`ProcessNemotronHistory` 针对原始 prompt 规划每张图像和每段音频（`PlanNemotronMedia`，按模态保持附件顺序），把每个 `<so_embedding>` 展开为 `<so_start>` + N + `<so_end>`，并排队投影后的行。

**两条路径，加载时选择**（`NemotronAudioEncoder.Load`）：
- **F32 计算的配套文件**（`nemotron.audio.compute_bf16=false`，即 `prepare_f32.py` 生成的文件）。编码器在语言模型的执行器上
  以每段音频一个计算图程序运行，覆盖所有后端，包括直接 CUDA（`NemotronAudioEncoder.Graph.cs`）。该程序就是托管路径的运算：
  - **下采样。** `Conv2d` 与 `DepthwiseConv2d` 各级，有效帧之后的行置零，然后是逐点乘积。
  - **相对位置注意力。** 内容分数与位置分数按头计算为乘积，相对位移是一个带步长的视图，有效帧之后的 key 被掩码。
  - **卷积模块。** GLU，然后把 'same' 深度卷积实现为零填充窗口上的 `SsmConv`，推理用 BatchNorm 折叠为缩放与平移。

  矩阵在后端有对应内核时保留 BF16 编码，向量与卷积核为 F32。embedding 返回到视觉塔分配器上（直接 CUDA 上为主机内存，与之前相同）。
- **模拟 BF16 的配套文件**（`prepare.py` 生成的文件）。它保留托管路径，像官方 BF16 编码器那样舍入每个中间结果。

已确立的内容：
- **前端。** Parakeet mel 前端在六段音频上与独立参考一致（`NemotronAudioInputTests`）。
- **编码器与参考对比。** 编码器与投影器在 8 与 128 mel 的小型 fixture 上与官方 Transformers 模块一致（`NemotronAudioEncoderTests`）：
  托管路径与 GGML CPU 上的 F32 与 BF16 计算；以及计算图程序在托管 CPU、`ggml_cpu`、`ggml_cuda` 与直接 `cuda` 上的 F32 计算，
  使用托管路径的 F32 容差。
- **注入。** 两段音频的注入与切片排队保持精确的行以及各请求独立的保留（`NemotronAudioInjectorTests`）。
- **训练好的配套权重。** 显式 F32 计算配置 8,064 个输出值全部一致。原始 BF16 计算配置仍然**失败**（容差不变下 3,794/8,064，
  第一处差异是第 0 层的 BF16 舍入边界）。
- **真实音频。** 在训练好的 F32 计算配套权重与一段 10.44 秒的 LibriSpeech 音频（131 行）上，计算图程序的编码耗时为：
  - `ggml_cuda` 上 24.4 ms，直接 `cuda` 上 48.1 ms（A40）；
  - `ggml_cpu` 上 481 ms；
  - 托管路径为 1.8-1.9 秒（测试虚拟机的 CPU 配额为 7.65 个）。

  各行与托管路径的相对 L2 在 1.8e-3 以内（BF16 权重的乘积会舍入其激活；`eng/validation/NemotronRuntimeAB --scenario audio`）。
- **语音。** 让 Nano Omni（`UD-Q4_K_XL`、BF16 mmproj，并通过 `TS_NEMOTRON_AUDIO_MMPROJ` 加载此配套文件）转写这段音频，它在
  `ggml_cuda` 和 `cuda` 上都逐字给出了参考转写："He hoped there would be stew for dinner, turnips and carrots and bruised
  potatoes and fat mutton pieces to be ladled out in thick peppered flour-fattened sauce."
  - 155 个 token 的提示在 `ggml_cuda` 上 prefill 耗时 245 ms，在 `cuda` 上耗时 617 ms。
  - CLI 使用了 `--no-skills`。在默认的技能目录下，模型用一次技能工具调用回应了请求，而没有给出转写。

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
| `nemotron_h.expert_count` | uint32 | MoE expert 数（0 表示 dense）；每层的数目取自该层堆叠的 expert 张量，Bolt 上它在 240 到 496 之间（这个标量是其中最大值） |
| `nemotron_h.expert_used_count` | uint32 或 uint32[] | 每 token 选用 expert 数；Bolt 存为逐层数组（非 FFN 层为 0），每个 FFN 层使用自己的值 |
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

## 7. TensorSharp 实现：共享解码运行时

Nemotron-H 与 Gemma 4、Qwen 3.5、Mistral 3 一样运行在共享解码运行时（`StandardDecoderModel`）上。
[`NemotronRuntimeModel`](../../TensorSharp.Models/Models/Nemotron/NemotronRuntimeModel.cs) 声明该家族的块与权重；
批处理、前缀缓存、检查点、放置、投机解码、专家卸载与 LoRA 都由运行时提供。旧的逐算子 `NemotronModel` 在 §11 的验证之后
被删除；随之删除的还有它的原生 Mamba-2 内核（`ggml_ops_mamba2.cpp`）、逐 token 的 MoE expert 内核（`TSGgml_MoEExpertsForwardF32`）、
融合 matmul+add 与单层 flash-decode 入口，以及它的开关：`TS_NEMOTRON_MAMBA2_BATCHED_NATIVE`、
`TS_NEMOTRON_MAMBA2_NATIVE_PREFILL_MIN_TOKENS`、`TS_MAMBA2_PREFILL_CACHE_MB`、`TS_NEMOTRON_ATTN_SCORE_BUDGET_MB`、
`TS_NEMOTRON_MOE_PREFILL_BATCHED_MIN_TOKENS`、`TS_NEMOTRON_MULTIMODAL_WARMUP(_TILES)` 与 `TS_TP_SNAPSHOT_TRACE`。现在设置
其中任何一个都会让启动停止，并说明这一变更（`RemovedCliFlags`）。`TS_NEMOTRON_IMAGE_MAX_TILES` 与 `TS_NEMOTRON_AUDIO_MMPROJ`
仍然有效。

**块。** 每层恰好是一个块，由逐层数组决定：Mamba-2（`head_count_kv` 与 `feed_forward_length` 都为 0）、注意力
（`head_count_kv` > 0）或前馈（`feed_forward_length` > 0）。运行时的块布局会跳过该层没有的部分。该家族只运行整次前向的
计算图（`RequiresGraphs`）：每一步，无论是提示块、decode 步、批处理槽位还是 verify 窗口，都是后端张量执行器上的一张图——
`ggml_*` 上是 GGML 图构建器，`cuda` 上是 `CudaGraphExecutor`，`mlx` 上是 `MlxGraphExecutor`，`cpu` 上是托管执行器。

**Mamba-2 混合器**（`EmitRecurrentMixer`），对这一步的每个序列：
1. `ssm_in` 拆为 `z | xBC | dt`。
2. conv 窗口（状态的最后 `conv_kernel - 1` 行）与该块的 `xBC` 行拼接后经过 `ssm_conv`，再加 conv bias 与 SiLU；窗口尾部写回。
3. `dt + dt_bias`，然后用一次 `GraphOp.SsmScan` 对该块的所有行做选择性扫描（旧模型逐 token 扫描）。GGML 运行未修改的
   上游 `ggml_ssm_scan`；直接 CUDA 运行 `ts_graph_ssm_scan_f32`（每个状态行一个 warp）；托管执行器运行 SIMD 循环。
4. `y + x * D`、`y * SiLU(z)`、分组 RMSNorm，然后 `ssm_out`。

verify 窗口保留扫描在每一行之后的状态（`ggml_ssm_scan` 的 K 个快照）以及 conv 历史。提交被接受的前缀时，把那一行的快照
拷入实时状态（`CommitRecurrentPrefix`），因此被拒绝的草稿可以精确回退，而无需重新运行前缀。

**注意力**没有位置编码（`AttentionIsPositionless`）：使用运行时的分页 K/V 与注意力内核，GGUF 设置了 `attention.scale`
时使用该值。

**前馈**（`EmitFeedForward`）：密集的 `down(relu(up x)^2)`，或 §4.4 的路由 expert（router、选择 bias、归一化并缩放的权重、
NVFP4 scale、latent bottleneck、共享 expert、逐层 expert 数与 top-k）。

**权重**在每个执行器上都保持 GGUF 格式。NVFP4 逐张量的 `.scale` sidecar 乘到其乘积上，张量并行的分片也保留它们。

## 8. 服务机制

| 机制 | 后端 | 说明 |
|---|---|---|
| 连续批处理 | 全部 | 批处理槽位：每个槽位持有一个 Mamba-2 conv 窗口与 SSM 状态，注意力层共享分页 K/V 池。并发请求以批处理步运行。 |
| 前缀缓存、检查点 | 全部 | 运行时的 holder 适配器。缓存的前缀在其检查点处携带 Mamba-2 状态，因此续接时 K/V 行与递归状态一起恢复。由 `CheckpointsContinueExactly` 覆盖。 |
| `--tp N` | cuda、ggml_cuda、ggml_vulkan | 按 rank 的计算图。Mamba-2 按连续的 group 块切分：`ssm_in` 的行块，以及每个 rank 一个 `ssm_out` 列块，因此量化行不会在块内被切开。注意力按 head 切分，每个 expert 按宽度切分。保持批处理。仅在模拟设备上覆盖（§11）。 |
| `--layer-split N` | cuda、ggml_cuda、ggml_vulkan | 整层阶段，保持批处理。 |
| 混合（`--tp` x `--layer-split`） | cuda、ggml_cuda、ggml_vulkan | 运行时的混合放置：每个整层阶段在自己的组上以按 rank 的计算图运行其层，残差流经主机从一个阶段传到下一个阶段；在 GGML 上，阶段内的部分和经主机暂存求和（上下文的集合通信跨越它的全部设备）。只在模拟设备上覆盖过。 |
| `--n-cpu-moe` / `--cpu-moe` | 全部 | 单设备或各阶段上可用；`--tp` 下拒绝。见 §10。 |
| 投机解码 | 全部 | n-gram 草稿、Nemotron 3.5 MTP 头（主干内的 NextN 块，或通过 `--draft-model` 加载的 MTP GGUF），以及 DFlash 与 DSpark 块级草稿器（`--draft-model`，见下文），在能精确回退 Mamba-2 状态的窗口中验证。 |
| 文本 LoRA | cpu、ggml_cpu、ggml_cuda、cuda | 单设备。每个被适配乘积的增量都在计算图中（`AppliesTextLoraInGraphs`）。适配目标与以前相同（见下文"文本 LoRA 适配器"）。 |
| 媒体 | 全部 | 视觉塔与音频塔的 embedding 作为排队的媒体行，写在其占位符行上（`QueueMedia`）。 |

### 文本 LoRA 适配器

Nemotron-H 在单设备上接受文本 LoRA 适配器（`--lora`、`/v1/loras`、请求中的 `lora`），`--tp` 或 `--layer-split` 下拒绝。
除路由 expert 外的所有投影都会被适配：
- Mamba2 的 `ssm_in` 与 `ssm_out`
- `attn_q/k/v`（加载时融合为 `attn_qkv`）与 `attn_output`
- dense 的 `ffn_up` 与 `ffn_down`
- 共享 expert
- latent 的 `ffn_latent_down` / `ffn_latent_up`
- router 与 LM head

作用于路由 expert（`*_exps`）的适配器会被 400 拒绝并列出其名称。增量在这一步的计算图内应用，因此 decode、批处理槽位与
verify 窗口都会应用它们；若计划中的某个权重被跳过，前向审计会抛出异常。已在合成模型上与合并权重的孪生模型对比验证
（cpu、ggml_cpu、ggml_cuda 与 cuda 上的 `TextLoraModelParityTests`；`TextLoraPlanTests`），尚未在真实 checkpoint 上验证。
详见[设计文档](../design/logprobs-lora-moe-runtime-features.md) §2.2。

### 块级草稿器（DFlash / DSpark）

Lightning 仓库在 MTP 头之外还发布了两个块级草稿器，任一个都通过 `--draft-model` 加载：

```bash
dotnet run --project TensorSharp.Cli -c Release -- --model models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf \
  --draft-model models/dspark-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.gguf --backend ggml_cuda
```

| | DFlash（`dflash-...-NVFP4.gguf`） | DSpark（`dspark-...-NVFP4.gguf`） |
|---|---|---|
| 层 | 6 层（hidden 2688，32 个查询头对 2 个 KV 头，head_dim 128）；注意力为 BF16，FFN 与 `fc` 为带 `.scale` 旁路的 NVFP4 | 同左，另有逐头 attention sink |
| 上下文 | 每层都关注全部历史（没有 `sliding_window`）；YaRN 旋转，系数 128、原始长度 8192 | 1,024 token 的滑动窗口 |
| 块 | 16 行，块内非因果；第 1 .. 15 行是草稿 | 8 行，块内因果；奖励锚点（`sample_from_anchor = false`）：第 1 .. 7 行是草稿 |
| 头 | 目标模型的 LM head，每行取 top-1 | 目标模型的 LM head 加一个秩 512 的 Markov 头：第 e 个草稿的 logits 加上 `w2 · w1[prev]`，`prev` 先是锚点，之后是上一个草稿的选择 |
| 编码器 | 主干第 2、6、20、30、42、52 层（52 是离开最后一层的流） | 同左 |

两者都以共享运行时的块级草稿计算图程序（`StandardDecoderModel.BlockDraft.cs`）在每个执行器上运行：
- **Markov 头。** 链条在设备上运行：`GetRows(w1, prev)`、`w2` 的乘积及其 scale、每个草稿的 top-1。只有草稿 id 与置信度
  会传回主机。
- **Attention sink。** 每个 sink 是注意力算子的第五个操作数。GGML 构建器把它交给 `ggml_flash_attn_ext_add_sinks`。
  张量执行器用通用乘积处理带 sink 的层，把 sink 作为每行 softmax 中的额外一列
  （`TensorGraphExecutor.FlashAttentionWithSinks`）。
- **掩码。** 当草稿器的 `attention.causal` 为真时，块掩码对块内第 i 行之前的查询关闭第 i 个槽位。
- **DFlash 的全注意力。** 它以主干上下文作为窗口，最多 8,192 个位置：环形缓冲为 8,224 行，96 MB 的 F32 K/V。上下文超过
  这个长度时，草稿只看到最近的 8,192 个位置。草稿都会被验证，所以这只可能降低接受率，不会改变输出。
- **草稿器嵌入。** 草稿器自带的 BF16 token embedding 以文件编码留在主机上。若加载为 F32 张量，会在设备上占 1.4 GB，
  却没有任何地方读取。

逐算子草稿器遵循同样的规则（YaRN、块内因果、奖励锚点的行）。融合的 GGML 草稿器既不支持 YaRN 也不支持块内因果，因此这两个
文件通过计算图程序起草，计算图不可用时逐算子起草。

在一张 A40 上运行 Lightning Q4_0，贪心解码，每个提示生成 128 个 token。每一行报告：
- 接受长度 τ，即（接受数 + verify 次数）/ verify 次数，因此每次 verify 的奖励 token 也计入；
- 相对普通解码的 decode 速度。

NVIDIA 报告的草稿长度 7 时的 τ 为 3.16（DFlash）与 3.75（DSpark）。

| 提示 | DSpark x7 ggml_cuda | DSpark x7 cuda | DFlash x7 ggml_cuda | DFlash x7 cuda | MTP x3 ggml_cuda / cuda |
|---|---|---|---|---|---|
| Fibonacci 函数（代码） | τ 5.13，1.29 倍 | τ 5.64，1.53 倍 | τ 3.54，1.10 倍 | τ 3.18，1.19 倍 | 1.16 / 1.38 倍 |
| 天空为什么是蓝色（3 条要点） | τ 2.80，0.87 倍 | τ 2.77，1.03 倍 | τ 1.73，0.74 倍 | τ 2.00，0.91 倍 | 1.13 / 1.15 倍 |
| 火车速度（算术） | τ 6.19，1.52 倍 | τ 5.46，1.58 倍 | τ 3.50，1.11 倍 | τ 3.33，1.22 倍 | 1.24 / 1.29 倍 |
| 德语翻译 | τ 2.76，0.85 倍 | τ 3.74，1.24 倍 | τ 2.17，0.83 倍 | τ 2.00，0.91 倍 | 1.12 / 1.17 倍 |

普通解码在 `ggml_cuda` 上为 140 tok/s，在 `cuda` 上为 68-71 tok/s。用 3 token 窗口时，DSpark 在 `ggml_cuda` 上把代码提示
跑到 1.35 倍、算术提示跑到 1.39 倍。

**加速的上限。** 在这个 MoE 主干上，8 行的 verify 要读取每一行的 expert：`ggml_cuda` 上约 23 ms，而一个 decode 步为
7.1 ms。稳态下一次起草耗时 4.4 ms（DSpark 的块计算图）加 0.35 ms（环形缓冲注入）。

**一致性。** 投机流会一直跟随普通贪心解码，直到 verify 窗口的内核翻转一个接近的选择。在 `ggml_cuda` 上，各流在普通
解码前两名 logits 相差 0.004-0.088 处分叉。在 `cuda` 上分叉处的差距最大为 0.21，而 MTP 头在同一个 token 处分叉，所以这
来自直接 CUDA verify 行的舍入（§11），而不是草稿器。

合成覆盖（`GraphBlockDraftTests`）：一个 DSpark 形态的草稿器（Markov 头、sink、块内因果、奖励锚点、一个位于堆栈之外的
目标层）与一个带自有 embedding 的全注意力 YaRN 草稿器，检查如下：
- 它们的计算图草稿与逐算子草稿器一致（id 相同，置信度相差不超过 1e-3，环形缓冲行相差不超过 1e-4），覆盖托管 CPU、
  `ggml_cpu`、`ggml_cuda` 与直接 `cuda`；
- 使用它们的投机解码在托管 CPU、`ggml_cuda` 与 `cuda` 上输出普通贪心解码的流。

`DSparkConfigTests` 覆盖这些导出文件的键。

限制：
- 没有运行 `mlx`、`ggml_metal` 与 `ggml_vulkan`。
- 多个请求的 DSpark 块不共享一个计算图程序：每个请求单独起草自己的块（DFlash 保留批量 lane）。
- 在直接 CUDA 上，带 sink 的层走通用注意力，而不是环形缓冲内核。

## 9. 内存

- **注意力层**：运行时的分页 K/V 池（默认 F16；`--kv-cache-dtype`）。
- **Mamba-2 层**，每个槽位：一个 `(conv_kernel - 1) x (d_inner + 2 x groups x d_state)` 个 float 的 conv 窗口，以及一个
  `heads x head_dim x d_state` 个 float 的 SSM 状态。槽位的 verify 缓冲（每一行的状态快照与 conv 历史）在它第一次投机时
  分配，窗口容量为至少 4 的 2 的幂。
- **权重**：驻留在设备上，被卸载层的路由 expert 除外（§10）。加载很快：Lightning 在 `ggml_cuda` 上 0.7 秒完成加载，旧模型
  需要 10.7 秒。

## 10. 专家卸载（`--n-cpu-moe` / `--cpu-moe`）

被卸载层的路由 expert 留在主机内存，这一步的计算图会为它们暂停。router、专家选择、latent 投影与共享 expert 留在设备上。
在 GGML 后端上，主机 MoE 段直接从 GGUF 映射计算它们；在张量执行器（`cuda`、`mlx`、`cpu`）上，由运行时的 expert host 计算。
两者都使用 ggml 的 CPU 内核：一次投影、`relu(s * up)^2`，然后 `down`，其中 `s` 是每条路由的 NVFP4 up scale。decode 或
verify 步使用单 token decode 团队（`TSGgml_HostExpertRowsReluSquared`）；GGML 提示块使用批处理的 CPU MoE 计算图。在设备
不是主机的张量执行器上，宽于 16 行的提示块会为这一步把选中的 expert 流式送到设备。

加速器分层（`--moe-hot-experts`、CUDA doorbell 的打包副本）只服务带门控的 expert，因此 Nemotron 的层完全在主机上计算。
`--tp` 下卸载在加载时被拒绝，因为按 rank 的计算图把 expert 保留在设备上。

在一张 A40 上实测（ggml `bb877c0c`），卸载全部 MoE 层（`TS_CPU_MOE=1`），与同一模型全部常驻对比：

| 模型、后端 | NLL 常驻 -> 卸载 | Decode tok/s | Prefill 512 tok/s（预热后） | 峰值显存 |
|---|---|---|---|---|
| Lightning Q4_0，ggml_cuda | 2.1954 -> 2.2013 | 143.2 -> 32.8 | 4633 -> 851 | 18.8 GB -> 3.7 GB |
| Lightning Q4_0，cuda | 2.1966 -> 2.1980 | 70 -> 24.4 | 1724 -> 27.5 | 卸载后 3.9 GB |
| Bolt-24GB，ggml_cuda | 2.0317 -> 2.0328 | 70.8 -> 16.3 | 1717 -> 608 | 卸载后 6.5 GB |

NLL 最多变化 0.006，因为主机内核会把激活量化为权重的点积类型。旧模型只在 GGML 的 `--tp` 下卸载（需要两张 GPU），因此
这里没有它的数字；在单设备上它节省不了显存。

## 11. 迁移验证

依赖：ggml `bb877c0c51f38c52ee2d5e41d58e3d0c20f3fed4`，上游且未修改。`ggml_ssm_scan` 与 `ggml_ssm_conv` 是它自己的算子；
TensorSharp 只增加了图构建器中的降级与直接 CUDA 的扫描内核。

**合成 fixture**（`NemotronSyntheticModelBuilder`、`NemotronRuntimeTests`）。四种布局：dense；带共享 expert 的 MoE；带
NVFP4 风格 sidecar 的 latent MoE；以及逐层 expert 数与 top-k。它们与旧模型的 logits 对比，这些 logits 在旧模型删除前于
托管后端上冻结（`Fixtures/Nemotron/LegacySynthetic/managed.json`）。`ggml_cpu`、`ggml_cuda` 与直接 `cuda` 都与托管后端
对比。还覆盖了：分块 prefill、模拟设备上的每种放置、批处理槽位、并发请求、检查点、投机解码、主机 expert、NVFP4 scale 与
折叠孪生模型的对比，以及 LoRA。

**真实 checkpoint**（一张 A40；与旧模型的教师强制 argmax 一致率、512 token NLL、CLI 的基准）：

| 模型、后端 | 与旧模型 argmax 一致 | NLL 旧 -> 运行时 | Decode tok/s | Prefill 512 tok/s | Prefill 4K tok/s |
|---|---|---|---|---|---|
| Nemotron-H 8B Q4_K_M，ggml_cuda | 218/219 | 2.6113 -> 2.6078 | 45 -> 104 | ~480 -> 3600 | 441 -> 4152 |
| Nemotron-H 8B Q4_K_M，cuda | 171/171 | 2.6107 -> 2.6091 | 33 -> 89.6 | 92.7 -> 2075 | 87.6 -> 2706 |
| Lightning 30B-A3B Q4_0，ggml_cuda | 242/246（接近平局） | 2.2001 -> 2.1954 | 39.6 -> 143.5 | ~300 -> 4690 | 482 -> 6885 |
| Lightning 30B-A3B Q4_0，cuda | 198/200（接近平局） | 2.1969 -> 2.1966 | 30 -> 70 | ~160 -> 1724 | 157 -> 3261 |
| Bolt 74B-A9B 24GB，ggml_cuda | 200/200（rel L2 约 0.6 %） | 2.0287 -> 2.0317 | 17.7 -> 70.8 | 131 -> 1717 | 142.7 -> 2215 |

- **多模态。** Nano Omni 配合其 BF16 mmproj，在旧模型与运行时上都给出了正确的图像回答。它 2,466 token 的图像 prefill
  从 8,367 ms 降到 798 ms，decode 从 28.4 提升到 116.7 tok/s。
- **并发。** 在服务端上，8 个并发的 Lightning 提示在旧模型上 33.3 秒完成（14.4 tok/s），在运行时上 2.4 秒完成
  （181.6 tok/s）。运行时在 8 个提示中有 6 个与单独服务时一致，旧模型为 8 个中 4 个；其余的在接近平局处分叉。
- **投机解码**（`ggml_cuda` 上的 Lightning；旧模型拒绝投机）。DFlash 与 DSpark 草稿器的测量见 §8。使用 MTP 头、3 token
  窗口时：
  - 在生成 46-96 个 token 的提示上，接受率为 64.5-85.7%，decode 为普通解码的 1.12-1.24 倍；
  - n-gram 草稿为 0.95-1.03 倍。
  - 投机流会一直跟随普通贪心解码，直到 verify 窗口的内核（以另一种批形状运行）翻转一个接近的选择。MTP 流在普通解码
    前两名 logits 相差 0.06-0.19 处分叉，n-gram 流在相差 0.04-0.06 处分叉。有一个 n-gram 流在相差 0.82 处分叉。
  - verify 窗口与普通步的 argmax 在 24/24 行上一致，有回退时为 24/24 与 23/24 行。
- **覆盖限制。** 以下没有运行：
  - `mlx`、`ggml_metal` 与 `ggml_vulkan`；
  - 多 GPU 的 `--tp` 与 `--layer-split`，只在模拟设备上覆盖；
  - 不带 `--cpu-moe` 的 Super VL（需要 96 GB 的设备）；
  - 音频配套文件，因为没有公开的。
- **已知代价。**
  - 直接 CUDA 的 MoE decode 步没有录制为 CUDA graph：Lightning 在 `cuda` 上 decode 为 70 tok/s，`ggml_cuda` 上为 143。
  - Lightning 加载后第一次 512 token prefill 要构建计算图：2,768 ms，旧模型为 1,782 ms。之后的运行为 110 ms。
  - 直接 CUDA 上被卸载的提示块逐个流式传输 expert（27.5 tok/s，§10）。
  - 直接 CUDA 的已知缺陷 D6（[README-direct-cuda-prefill](../../eng/validation/README-direct-cuda-prefill.md)）沿用了下来：在两个合成
    MoE 文件上，超过 512 行的单次前向的最后一行可能远离参考值。它已收窄到量化乘积的大行数 F16 路径。Lightning 在 `cuda` 上
    1,100 token 的单次前向与 `ggml_cuda` 相差 1.6e-2。

## 12. 输出解析器与聊天模板

- `NemotronOutputParser`（一种 `ChatMlOutputParser`）解析 `<think> ... </think>` 思维链，以及 `<TOOLCALL>...</TOOLCALL>` 或 `<tool_call>...</tool_call>` 中的工具调用。调用体可以是一个 JSON 调用对象、这类对象的 JSON 列表（Nemotron 约定）、XML 形式的 `<function=NAME>` / `<parameter=key>`，或 Python 风格的调用列表，例如 `[read_file(path="notes.txt"), shell("wc -l notes.txt")]`（`PythonicToolCalls`）。Reasoning-128K 8B Q4_K_M 无论 prompt 要求什么格式，写出的每个调用都是这种 Python 形式。Python 调用只接受字面量，位置参数绑定到声明中同一位置的参数，被 EOS 或 token 上限截断的调用体不产生调用。只有开标签后面跟着调用体时才算开始一个调用，因此推理中提到该标签时仍是推理。其他任何形状的调用体不产生调用。此前这样的调用体会让解析器抛出异常并中断流式 HTTP 响应。
- `response_format` 可以与 `"think": true` 同时使用。开启思考时 prompt 在 assistant 标记后预置 `<think>\n`（Nemotron 3.5 的 GGUF 模板也是如此），因此 JSON 语法在推理期间保持休眠，并在模型输出 `</think>` 后启用（`ThinkingGrammarActivationTrigger`）。此前该组合返回 HTTP 400。`</think>` 同时是该家族的 `ThinkingBudgetEndToken`：Nemotron 3.5 / Omni 的词表中它是单个 token，达到 `TS_THINKING_BUDGET`（输出额度不少于 512 token 时为 75%）时会输出它，受约束的答案在原有 `max_tokens` 内继续。Nemotron-H Reasoning-128K 的 GGUF 用多个 token 拼出 `</think>`，因此保留带说明的 `thinking_budget` 停止。在 Nemotron 3.5 Lightning IQ4_XS（Metal）上实测：`max_tokens` 为 256 时推理就用完了全部额度（content 为空，`finish_reason=length`）；为 1024 时 json / json_schema / json_unicode 在 c1 全部通过，json_schema / json_unicode 在 c4 为 4/4。json 在 c4 为 1/4：四个请求中有三个是在针对另一个 prompt 推理（"User says: Mars"），这是另行跟踪的批量 prefill 损坏问题。
- `</think>` 与 JSON 之间的空白属于前导部分（`GrammarConstraint.ActivateAfter(trigger, skipLeadingWhitespace: true)`，仅 `response_format` 使用）。Nemotron-H 8B Reasoning-128K 的词表把 `</think>\n\n` 切成 `</think` + `>\n\n`；JSON 根不能以换行开头，于是该 token 被屏蔽，模型改写 `</think}`，触发词始终不匹配，整个回复都停留在推理中（content 为 null，`json_schema` 返回 HTTP 422）。在 Nemotron-H 8B Q4_K_M（RTX PRO 6000，`--thinking`，c1 与 c4，重复三次，`max_tokens` 256）上实测：修复前有 10 个回复写出 `</think}`，修复后没有；json_schema 由 4/15 升至 15/15，json 由 13/15 升至 14/15。json_unicode 仍为 0/15，因为仅推理就用满 256 token；`max_tokens` 为 1024 时为 5/5（c1 + c4）。在 1024 下 json c4 为 0/4（对象缺少 `moons`，或在针对另一个 prompt 推理），json_schema c4 为 2/4（两个请求在 768 token 处触发带说明的 `thinking_budget` 停止）。
- 同一架构名下有两种轮次格式，由 GGUF 内嵌的 `tokenizer.chat_template` 决定渲染哪一种（`ChatTemplate.IsNemotronHReasoningTemplate`）：
  - **Nemotron-H 8B/47B Reasoning-128K** 训练时使用 `<SPECIAL_10>System\n{system}\n<SPECIAL_11>User\n{user}\n<SPECIAL_11>Assistant\n`（EOS 为 `<SPECIAL_11>`）。推理开关是 system prompt 中的 `{'reasoning': True}` / `{'reasoning': False}`，生成提示随之打开（`<think>\n`）或关闭（`<think></think>`）推理块。`RenderNemotronHReasoning` 根据请求的 `think` 标志加入该标记，除非 system prompt 已经带有；标记放在 system 部分的末尾，位于 system 文本和工具声明之后，因此两种思考模式在出现差异之前渲染的是同样的 system prompt 和工具。若放在开头，两者在第五个 token 就已分叉，一种模式的预热或缓存前缀对另一种模式毫无用处。官方模板没有工具语法，因此工具按 Nemotron 家族自己的约定声明，逐字沿用 NVIDIA 的 Nemotron 模板（NVIDIA-Nemotron-Nano-9B-v2、Llama-Nemotron Super v1.5）的写法：声明放在 `<AVAILABLE_TOOLS>[...]</AVAILABLE_TOOLS>` 中，随后的说明要求用 `<TOOLCALL>[...]</TOOLCALL>` 调用，并以 `<TOOL_RESPONSE>[...]</TOOL_RESPONSE>` 接收结果。过去的调用渲染回 `<TOOLCALL>[...]</TOOLCALL>`，一个轮次的所有工具结果作为一个 user 轮次返回，包裹在 `<TOOL_RESPONSE>[...]</TOOL_RESPONSE>` 中。在 8B Q4_K_M 上（两个工具任务，每组三个种子），在 16k 字符的 skills prompt 之后，模型用这段声明调用工具 12 次中有 7 次，用 Hermes JSON `<tool_call>` 声明只有 2 次；只声明工具时为 12 次中 10 次对 8 次。模型在调用之后还会接着写出一段编造的工具响应并据此作答，因此轮次在第一个完整调用处结束（`ChatProtocol.ToolCallEndsTurn`、`ToolCallTurnEnd`）。判定由该家族自己的解析器做出，所以推理中提到的标签、或字符串参数里的闭合标签都不会结束轮次。模型把一个轮次的所有调用放在一个标签内的同一个列表中，因此不会丢失调用。这些 checkpoint 过去被当作 ChatML 渲染：模型把 `<|im_start|>` 当普通文本，回答里出现 `</think>`、自编的 `<|im_start|>user` 轮次和 `<unk>` 循环。
  - 关闭思考时，Reasoning-128K checkpoint 有时仍会在已关闭的 `<think></think>` 之后推理，并在调用或答案之前自己写出 `</think>`。这主要发生在工具结果之后的工具轮次。在 8B Q4_K_M 的 98 条日志回复中，闭合之前先有 320-1878 个字符的 "Okay, the user ..."（中位数 684），工具循环越深越长。这段推理和字面的 `</think>` 过去会作为答案显示并保存。协议按模板声明这一行为（`ChatProtocol.ThinkingOffReplyMayReason`）。管线会告诉回复的每个解析器 prompt 以什么结尾，解析器把回复的开头视为待定。`</think>` 表明它是推理；完整的工具调用或回复结束表明它是答案。空的调用列表（`<TOOLCALL>[]</TOOLCALL>`，两种调用标签、所有 Nemotron-H 模板皆然）不算调用，也不产生文本。位于 Markdown 代码块或行内代码中的 `</think>` 是引用的文本，从第 2,048 个字符之后才开始的 `</think>` 也是。日志中的 98 次多余闭合都落在这个窗口内（第 95 百分位为 1,601）。1,024 个字符的窗口只覆盖 79%，漏掉的推理仍会进入被暂扣的流。这些数字由 [`eng/validation/NemotronThinkOffReplay`](../../eng/validation/NemotronThinkOffReplay/README.md) 用真实解析器回放日志中的回复测得；`eng/validation/nemotron-thinkoff-e2e.py` 对运行中的服务器做端到端检查。
  - 已知限制：这个窗口并不覆盖所有回复。在 TensorAgent 带 12 个工具的系统提示词之后，工具轮次推理得更长；在一次 TensorAgent 实机运行（8B Q4_K_M，Metal）中，39 次多余闭合有 4 次出现在第 2,212-3,064 个字符。一旦过了 2,048 个字符仍没有闭合，解析器在所有路径上（包括 Web UI 和 TensorAgent）都已判定回复是答案，之后的闭合会作为答案自身的文本保留。在那次运行中，9 个关闭思考的工具对话里有 2 个的答案中出现了推理和字面的 `</think>`。
  - Web UI 和 TensorAgent 页面会随生成流式显示待定文本。闭合出现时，主机先用 `thinking` 帧发送这段推理，再用 `replace` 帧发送剩下的答案。不能收回已发送文本的客户端（OpenAI、Ollama 和 Responses API 以及 CLI）要等文本有了定论才收到它。因此关闭思考的普通答案，第一个字符最多晚 2,048 个字符到达。在 104 条日志中的普通回复上，中位数为 229 个字符（约 3 秒），第 90 百分位为 1,401 个字符（约 22 秒），最多 2,047 个字符（按日志中的解码速度约 37 秒）。在 Metal 上通过 `/v1/chat/completions` 流式请求时，一句话的答案（190 个字符）在第一个 token 之后 2.6 秒整段到达，1,402 个字符的答案在 17 秒后到达，3,362 个字符的答案在 27 秒后先发出前 2,053 个字符，其余部分随后流式发送。开启思考的回复以及 Nemotron 3 Nano / Omni（ChatML）既不会被暂扣，也不会被收回。
  - 窗口按字符计算，因此解码越慢等待越长：2,048 个字符约为 500 个 token，在每秒 5 个 token 时（47B 或 CPU 后端）约为 100 秒。加入保活之前，被暂扣的流在这段时间内什么都不发送，代理的空闲超时（nginx 为 60 秒）会关闭连接。现在文本被暂扣期间，`/v1/chat/completions` 和 `/v1/responses` 每 15 秒发送一行 SSE 注释（`: keep-alive`），`/api/chat/ollama` 发送 message 为空的分块。技能与子智能体循环自己暂扣文本，它为每个暂扣的片段向适配器交出一个空更新，适配器才能发送保活。CLI 写入终端，不额外发送任何内容。prompt 预填充期间仍与以前一样不发送任何内容。
  - 判定按回复生成时的原样读取。转录和非流式 API 整段解析回复，页面和 SSE API 边流式边解析，现在两者的判定一致。此前 Markdown 规则读取的是显示出来的文本：在空调用列表附近，调用分支裁掉了换行，两行围栏连在一起，同一个回复在流中是推理、在转录中却是答案，于是下一轮会重新预填充整个对话。
  - 关闭思考且带 `response_format` 的回复从第一个 token 起就由语法（或首 token 列表）决定形状，因此不可能是多余的推理。它不会被当作可能的推理来解析，其 JSON 字符串里的 `</think>` 保持原样。管线不会为它宣告 prompt 结尾，OpenAI 适配器也不对它运行解析器，与该模板需要解析器之前一样。`/v1/responses` 不对其 `text.format` 施加语法（格式只在 prompt 中要求），因此它的回复仍可能以多余的推理开头，并像其他关闭思考的回复一样解析。
  - TensorAgent 随生成显示待定的回复，因此 GPU 故障之后它无法分辨已显示的答案中哪部分已有定论。这类模型的轮次会从头重新开始，而不是续写。若续写，这段推理会作为模型自己已写出的答案交还给模型，并留在答案前面。
  - 回复从哪里开始，由 prompt 的结尾决定，而不是由请求的 `think` 标志决定。调用方在 system prompt 中写入 `{'reasoning': True}` 时，即使请求关闭了思考，推理块也会打开。此时回复按推理解析，转录记录的也是缓存中实际存在的 `<think>\n`。此前整段推理被当作答案，下一轮重新渲染的历史也比缓存少一个推理块。
  - **Nemotron 3 Nano / Omni**（以及其他所有 `nemotron_h*` 模板）使用 ChatML（`<|im_start|>` / `<|im_end|>`）。多模态占位符包括 `<image>`（之后展开为 `<img>` + N 个 token + `</img>`）与 `<so_embedding>`（音频）。`RenderNemotron` 在 system 轮次中以由 `<function>` 条目组成的 `<tools>` 块声明工具，并要求以 `<tool_call><function=NAME><parameter=key>value</parameter></function></tool_call>` 形式调用；过去的调用按同一形式渲染回去，连续的工具结果在一个 user 轮次中返回，每个结果包裹在 `<tool_response>` 中。这些模型每个并行调用各写一个 `<tool_call>`，因此其轮次不会在第一个调用处结束。

## 13. 优化机会

- **录制的直接 CUDA MoE decode** —— 把 MoE decode 步作为 CUDA graph 重放（`ggml_cuda` 实际上就是这样做的），以缩小
  Lightning 上 70 对 143 tok/s 的差距。
- **单投影 expert 的主机 expert 分层** —— GGML 加速器分层与 CUDA doorbell 只接受带门控的 expert。ReLU^2 分层可以把每个
  卸载层最热的 expert 留在设备上。
- **直接 CUDA 上的批量 expert 流式传输** —— 被卸载的提示块一次只流式传输一个 expert。
- **`--tp` 配合卸载，以及张量并行 x 按层切分的混合** —— 两者都需要这个家族尚未生成的按 rank 计算图。
- **DSpark 块的开销** —— 草稿器的 BF16 注意力权重在加载时反量化为 F32（每个块在六层上要读 565 MB）；在计算图路径上
  保持 BF16 可以减半。多个请求的 DSpark 块也可以像 DFlash 的 lane 那样共享一个程序。
- **音频塔** —— F32 计算的配套文件在每个执行器上以计算图程序运行（§4.7）；公开 GGUF 仍然不带音频塔，所以除非加载了
  该配套文件，音频会被拒绝。待完成：训练权重上的 BF16 计算一致性（模拟 BF16 的配套文件保留托管路径）。
