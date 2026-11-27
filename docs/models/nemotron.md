# Nemotron-H

[← back to model index](README.md) | [中文](nemotron_zh-cn.md)

| Property | Value |
|---|---|
| Provider | NVIDIA |
| GGUF architecture keys | `nemotron_h`, `nemotron_h_moe`, `nemotron_h_omni` |
| Source class | [`NemotronRuntimeModel`](../../TensorSharp.Models/Models/Nemotron/NemotronRuntimeModel.cs) on the shared decoder runtime (`StandardDecoderModel`; its Mamba-2 mixer and experts in [`NemotronRuntimeModel.Graph.cs`](../../TensorSharp.Models/Models/Nemotron/NemotronRuntimeModel.Graph.cs), its recurrent state in [`NemotronRuntimeModel.State.cs`](../../TensorSharp.Models/Models/Nemotron/NemotronRuntimeModel.State.cs)). The legacy per-op `NemotronModel` was removed (§7, §11). |
| Vision encoder | [`NemotronVisionEncoder`](../../TensorSharp.Models/Models/Nemotron/NemotronVisionEncoder.cs) (RADIO / v2_vl ViT) |
| Image processor | [`NemotronImageProcessor`](../../TensorSharp.Models/Models/Nemotron/NemotronImageProcessor.cs) |
| Audio frontend | [`NemotronAudioPreprocessor`](../../TensorSharp.Models/Models/Nemotron/NemotronAudioPreprocessor.cs) (Parakeet-style log-mel) |
| Audio encoder | [`NemotronAudioEncoder`](../../TensorSharp.Models/Models/Nemotron/NemotronAudioEncoder.cs) (Parakeet/FastConformer + sound projector; needs a companion GGUF that no public repository ships, see §4.7) |
| Example models | Nemotron-H-8B-Reasoning-128K, Nemotron-H-47B-Reasoning-128K, Nemotron 3 Nano Omni, Nemotron 3.5 Bolt 74B-A9B and Super VL 120B-A12B (see [below](#nemotron-35-super-vl-and-bolt)) |
| Modalities | Text, image (Omni-class with `mmproj` loaded). Audio only when an audio companion GGUF carrying the Parakeet tower is loaded (§4.7); otherwise audio is **refused** (HTTP 400 / CLI error with `NemotronRuntimeModel.AudioInputUnsupportedMessage`): the public Omni GGUFs ship no audio tower, only the RADIO vision tower in the `mmproj` (see §4.6). |
| Thinking mode | Yes (`<think> ... </think>`) |
| Tool calling | Yes: Reasoning-128K `<TOOLCALL>[...]</TOOLCALL>` (JSON or Python-style calls), Nemotron 3 Nano / Omni `<tool_call>` with an XML `<function=...>` body (§12); eligible for skills, the code tools and server-side [sub-agent delegation](../multi_agent.md) |
| Batched / paged forward | **Default ON** (the shared runtime's batched slots: each slot's Mamba-2 conv window and SSM state, paged K/V for the attention layers) — `--no-continuous-batching` forces the per-sequence path. See §8. |
| Speculative decoding | n-gram drafts, the Nemotron 3.5 MTP head (the trunk's NextN block, or an MTP GGUF via `--draft-model`), and the published Nemotron 3.5 Lightning DFlash and DSpark block drafters (`--draft-model`), verified in windows whose Mamba-2 states rewind exactly. See §8. |
| Expert offload | `--n-cpu-moe` / `--cpu-moe` on one device or over `--layer-split` stages, every backend; refused with `--tp`. See §10. |
| Output parser | `NemotronOutputParser` (a `ChatMlOutputParser`) |

## Downloads

Verified GGUF pointers:

| Model | HF repo | Recommended file | mmproj |
|---|---|---|---|
| Nemotron-H-8B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF) | `nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf` (4.983 GB) or `nvidia_Nemotron-H-8B-Reasoning-128K-Q8_0.gguf` (8.620 GB) | — (text only) |
| Nemotron-H-47B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF) | `nvidia_Nemotron-H-47B-Reasoning-128K-Q4_K_M.gguf` (28.188 GB) | — (text only) |
| Nemotron 3 Nano Omni 30B-A3B | [unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF) | `NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf` (23.927 GB) | `mmproj-BF16.gguf` (1.590 GB; same repo) — **required for image input** |
| Nemotron 3.5 Bolt 74B-A9B | [nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF) | `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` (22.018 GB) | `mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` (1.734 GB; same repo) |
| Nemotron 3.5 Lightning 30B-A3B | [ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) | `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf` (18.898 GB). Drafters in the same repo: `mtp-...-Q4_0.gguf` (1.156 GB), `dflash-...-NVFP4.gguf` (1.185 GB), `dspark-...-NVFP4.gguf` (1.357 GB) | — (text only) |
| Nemotron 3.5 Super VL 120B-A12B | [ggml-org/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF) | `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4.gguf` (76.919 GB) or `-Q4_0.gguf` (71.934 GB) | `mmproj-NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q8_0.gguf` (0.967 GB) or `-BF16.gguf` (1.734 GB; same repo) |

The Omni `mmproj` enables **image input only**: it carries the 32-block RADIO
vision tower (`v.blk.*`, `v.patch_embd`, `v.position_embd`) and the
`nemotron_v2_vl` MLP projector (`mm.model.mlp.*`) with `clip.has_vision_encoder`
and nothing for audio, and the language GGUF has only the `<so_embedding>`
placeholder token. With only these files loaded an audio attachment is
therefore **refused** at the request (HTTP 400) rather than decoded and
dropped (see §4.6); audio needs a separately prepared companion GGUF (§4.7).

These conversions identify NVIDIA's corresponding Nemotron repositories as
their upstream bases. The upstream cards use NVIDIA-specific (`other`) terms;
the two bartowski conversion cards do not declare a license. Review the NVIDIA
base-model terms before redistribution.

Command-line download (one line per file; requires `pip install -U huggingface_hub`):

```bash
python -m pip install -U huggingface_hub
hf download bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --local-dir models
hf download bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF nvidia_Nemotron-H-47B-Reasoning-128K-Q4_K_M.gguf --local-dir models
hf download unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf --local-dir models
hf download unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF mmproj-BF16.gguf --local-dir models
```

CLI one-shot (with `--image` and no `--input` a default describe-the-image
prompt is used; CLI sampling defaults to greedy and `--max-tokens` defaults
to 100):

```bash
dotnet run --project TensorSharp.Cli -c Release -- --model models/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf \
  --mmproj models/mmproj-BF16.gguf \
  --image photo.png --max-tokens 512 --backend ggml_cuda
```

Server (Web UI + OpenAI/Ollama-compatible APIs on `http://localhost:5000`):

```bash
dotnet run --project TensorSharp.Server.Host -c Release -- --model models/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf \
  --mmproj models/mmproj-BF16.gguf --backend ggml_cuda --max-tokens 4096
```

### Nemotron 3.5 Super VL and Bolt

Both are `nemotron_h_moe` checkpoints of 88 layers: 40 Mamba2, 8 attention (32 query heads over 2 KV
heads, head_dim 128) and 40 latent MoE layers (hidden 4096, MoE latent 1024, shared expert). Both come
with a RADIO vision mmproj. The [design note](../design/nemotron-super-vl-bolt-support.md) records how
support was added and what was validated.

| | Bolt 74B-A9B | Super VL 120B-A12B |
|---|---|---|
| Routed experts | Per layer ("Puzzle" config): 240-496 experts, top-9, n_ff 1536-2560 | 512 experts, top-22, n_ff 2688 |
| Language GGUF | `...-Bolt-74B-A9B-24GB.gguf`, **22.018 GB**: IQ2_XXS experts, NVFP4 Mamba2 projections with `.scale` sidecars | `...-NVFP4.gguf`, **76.919 GB** (NVFP4 experts with per-expert scales); `...-Q4_0.gguf`, **71.934 GB** (no sidecars) |
| mmproj | 1.734 GB; q/k/v stored **split** (fused at load) | Q8_0 0.967 GB or BF16 1.734 GB; fused q/k/v, plus a final LayerNorm `v.post_ln` (applied after the blocks) |

**Memory.** The language GGUF has to fit on the device, with room for the KV cache and workspace:
- **Super VL NVFP4** needs about 71.6 GiB (76.919 GB) of weights on one device. It ran on a 128 GB
  DGX Spark (GB10) and does not fit a 48 GB GPU, which is why the PR #282 review ran it on
  `ggml_cpu`. One 96 GB GPU should hold it, but that was not validated.
- **Bolt-24GB** ran on `ggml_cuda` and `cuda` on the DGX Spark. In the PR #282 review it ran on a
  48 GB A40 with ECC enabled (about 45.0 GiB usable).
- **The vision tower** runs as graph programs on the language model's executor (§4.5), its weights in their
  file encoding on the model's allocator (the BF16 mmproj: about 1.7 GB). Its managed copy, about 3.2 GiB of
  F32 on the host, lives only for the load's validation.

**`--tp`.** Only `--tp 2` can apply, because the attention layers have 2 KV heads. Tensor parallelism runs as the
shared runtime's per-rank graphs (§8): each rank owns a contiguous block of Mamba-2 groups and the heads that read them, one
KV head and its 16 query heads, and a slice of every expert's width; the router and the selection run on every rank. The NVFP4 scales follow
the shards (a column shard's partial is scaled before the reduction; the per-expert down scale is folded into the routing
weights and the per-expert up scale multiplies each route's up product before ReLU^2). Per-layer expert counts and top-k
(Bolt) are each layer's own. This is covered on synthetic fixtures over simulated devices (`NemotronRuntimeTests`,
`NemotronExpertScaleTests`, `NemotronPerLayerMoEConfigTests`); no multi-GPU run of either model is recorded.

**`--n-cpu-moe` / `--cpu-moe`** keep the offloaded layers' routed experts in host memory on one device (§10): on an A40,
Bolt-24GB ran at 6.5 GB peak VRAM with every layer offloaded. The option is refused with `--tp`.

**Super VL on one 48 GB A40** (`--cpu-moe`: every layer's routed experts in host memory; ggml `bb877c0c`). The test VM's
cgroup quota of 7.65 CPUs (`nproc` reports 96) bounds the host experts, so these rates are this machine's:

| Backend | NLL, 256 tokens after 64 | Prefill 512 | Decode | Peak VRAM |
|---|---|---|---|---|
| `ggml_cuda` | 2.1167 (ppl 8.30, top-1 57.0 %) | 24.2 tok/s | 3.85 tok/s | 14.3 GB |
| `cuda` | 2.1156 (ppl 8.29, top-1 56.6 %) | 6.4 tok/s | 3.38 tok/s | 17.2 GB |

What the runs showed:
- **Greedy answers.** Five prompts (a fact, Python code, three bullets, arithmetic, a translation) were answered correctly
  on both backends.
- **Image.** The BF16 mmproj on the 640x480 test image was described correctly on both backends: a red square top
  left, a blue circle top right, a green triangle bottom centre, and the text "HELLO 42".
  - The prompt is 289 tokens with `--no-skills`. The CLI's default skill catalog adds about 2,190 more.
  - `ggml_cuda` prefilled the 2,475 tokens with the catalog in 38.4 s and decoded at 2.9 tok/s.
  - `cuda`, after the fix below, prefilled the 289 tokens in 117 s and decoded at 2.3 tok/s. Direct CUDA streams an
    offloaded prompt chunk's experts to the device one at a time (§10).
- **The route-tail skip.** The first `cuda` run scored 2.2100. Per token against the managed CPU backend (2.1195 over 32
  tokens; `ggml_cuda` 2.1156, mean per-token |delta| 0.007), direct CUDA was off from the first token (mean |delta| 0.14).
  - The cause: the tensor executors' expert host applied the route-tail skip (`TS_MOE_ROUTE_TAIL_SKIP`, rank 7) to
    Nemotron's one-projection experts. That left 15 of each token's 22 routes out, while GGML, which keeps no tier for
    these experts, computes them all.
  - The fix: the tensor executors no longer skip Nemotron's ReLU^2 experts, and they apply the skip at decode and verify
    widths only (8 rows), as documented. Since then `cuda` follows the CPU (mean |delta| 0.009).
  - `NemotronBf16ProjectionTests` pins it with a synthetic top-22 model.

```bash
hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf --local-dir models
hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf --local-dir models
dotnet run --project TensorSharp.Cli -c Release -- --model models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf \
  --mmproj models/mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf --image photo.png --backend ggml_cuda
```

## 1. Origin and intent

Nemotron-H is NVIDIA's hybrid **Mamba2 + Transformer** family. The same
backbone covers the dense `nemotron_h` line (e.g. Nemotron-H-8B / 47B) and the
MoE `nemotron_h_moe` line. The Omni distribution (Nemotron 3 Nano Omni)
additionally ships a RADIO / v2_vl vision encoder (via `mmproj`). TensorSharp
also implements the Omni line's Parakeet-style audio preprocessor and its
audio tower (a 24-layer Parakeet/FastConformer encoder plus sound projector,
`NemotronAudioEncoder`), but those weights are not in any public GGUF: with
the published files image is the only functional extra modality and audio is
refused (see §4.6); a companion GGUF converted from NVIDIA's checkpoint enables
audio (see §4.7).

The defining traits are:

- **Three layer types in one stack** — Mamba2 SSM layers, attention-only
  layers, and FFN-only layers (optionally MoE). The per-layer type is
  determined by GGUF metadata arrays: `head_count_kv[l]` and
  `feed_forward_length[l]` together select the layer type.
- **Mamba2 SSM** — selective state-space model with grouped heads. The conv
  state and the SSM state are maintained per layer and per token.
- **No RoPE** — attention layers carry no positional encoding. Position
  information is implicit in the SSM state.
- **ReLU² FFN** activation (`max(0, x)²`).
- **Sigmoid-routed MoE** — uses per-expert sigmoid probabilities with
  optional additive bias for expert selection, top-K, and renormalization /
  scaling.
- **Optional latent bottleneck and shared experts** — MoE FFN can have an
  in / out latent bottleneck (`ffn_latent_in` / `ffn_latent_out`, named
  `ffn_latent_down` / `ffn_latent_up` in the Super / Bolt GGUFs) and a
  shared ReLU² branch (`ffn_up_shexp` / `ffn_down_shexp`) added in
  parallel.

## 2. Model architecture

```
                tokens (int[])
                      │
              token_embd.weight
                      │
        [optional] InjectVisionEmbeddings (image)
                      │
        ┌──── × NumLayers ───────────────────────────────┐
        │ select layer type from (head_count_kv[l],      │
        │   feed_forward_length[l]):                     │
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
        │   RMSNorm(attn_norm) ─► QKV (fused)           │
        │   attention (no RoPE)                          │
        │   attn_output ─► residual                      │
        │                                                │
        │ FFN (dense):                                   │
        │   RMSNorm(attn_norm) ─► up ─► ReLU² ─► down   │
        │   residual += ffn_out                          │
        │                                                │
        │ FFN (MoE):                                     │
        │   RMSNorm(attn_norm)                           │
        │   [optional latent_in]                         │
        │   route(sigmoid(logits) + bias) ─► topK        │
        │   For each expert:                             │
        │       up ─► ReLU² ─► down ─► weighted Σ        │
        │   [optional latent_out]                        │
        │   [optional + shared_expert]                   │
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

## 3. Forward graph

Each layer is one block (§7), lowered into the step's graph as:

```
if Mamba2:
  hidden ─► RMSNorm(attn_norm)
         ─► ssm_in.weight matmul ─► split into z, xBC, dt
         ─► ssm_conv over [conv window | xBC] with ssm_conv1d.weight (+ bias; the window is the slot's state)
         ─► SiLU
         ─► selective scan over the chunk's rows (ggml_ssm_scan; the state is the slot's):
                state ← state * exp(-softplus(dt) * A) + B * x * dt
                y     ← state @ C  (+ x * D if D present)
         ─► out = GroupRMSNorm(y * SiLU(z), ssm_norm.weight)
         ─► out matmul ssm_out.weight → o
         ─► residual = hidden + o

if Attention:
  hidden ─► RMSNorm(attn_norm)
         ─► attn_qkv.weight matmul ─► split Q, K, V (per-layer head counts)
         ─► append (K, V) to KV cache
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
         ─► idx    = topK(probs + exp_probs_b)            # the bias steers selection only
         ─► weights = probs[idx]; if expert_weights_norm: weights /= max(sum(weights), 2^-14)
         ─► weights *= expert_weights_scale (and each expert's NVFP4 down scale)
         ─► [if latent] ffn_latent_in × hidden → latent
         ─► For each selected expert e (one ggml_mul_mat_id per projection):
                 x    = ffn_up_exps[e] × (hidden or latent)   (× the expert's NVFP4 up scale)
                 x   ← max(0, x)²
                 down = ffn_down_exps[e] × x
                 accum += weights[e] * down
         ─► [if latent] accum = ffn_latent_out × accum
         ─► [if shared expert] accum += ffn_down_shexp × max(0, ffn_up_shexp × hidden)²
         ─► residual = hidden + accum
```

After all layers:

```
hidden ─► narrow(seq_len-1) if prefill
       ─► RMSNorm(output_norm)
       ─► output.weight matmul → logits
       ─► copy to float[VocabSize]
```

## 4. Components in detail

### 4.1 Mamba2 SSM layer

The Mamba2 block is the most computationally specialized piece. The
`ssm_in.weight` projection produces a concatenation of three streams:

- **z** — gate, dim `d_inner` (`ssm.inner_size`).
- **xBC** — the joint `(x, B, C)` stream, dim `d_inner + 2 * groups * d_state`.
- **dt** — selective timestep, dim `heads` (`ssm.time_step_rank`).

`xBC` is then run through a conv1d sliding window of kernel size `ssm.conv_kernel`
(the window is part of the sequence's recurrent state). After SiLU, the SSM scan computes, per row:

```
ΔA   = exp( -softplus(dt + ssm_dt.bias) * A )           # A ∈ ℝ^{nHead}
ΔB   = (B * x * dt)                                      # B ∈ ℝ^{nGroup × dState}
state = ΔA · state + ΔB                                  # state ∈ ℝ^{dState × headDim × nHead}
y    = state · C                                         # C ∈ ℝ^{nGroup × dState}
y   += x * ssm_d                                         # if `ssm_d` is present
```

The runtime runs it as one `GraphOp.SsmScan` over all of a chunk's rows (§7).
`GroupRMSNorm(y, ssm_norm.weight)` then normalizes per group after the gate
`SiLU(z) * y`. Finally `ssm_out` projects back to the hidden dim.

### 4.2 Attention layer

- Standard GQA. Per-layer head counts and KV head counts are read from the
  GGUF arrays.
- Fused QKV when `attn_qkv.weight` exists; otherwise the runtime fuses the
  separate `attn_q/k/v.weight` tensors at load.
- Attention scale is `nemotron_h.attention.scale` when set, else
  `1/sqrt(headDim)`.
- **No RoPE** — the SSM state implicitly carries position information, so
  attention does not need positional encoding.

### 4.3 FFN layer (dense)

The dense FFN uses a single `ffn_up` matmul followed by ReLU²
(`x ← max(0, x)²`) and `ffn_down`.

### 4.4 FFN layer (MoE)

- **Routing**: `sigmoid(logits) + exp_probs_b` (per-expert additive selection bias) → top-k → the selected experts'
  unbiased probabilities as weights, renormalized by their sum (at least 2^-14) when `expert_weights_norm` is set, times
  `expert_weights_scale`. One graph lowering serves every path (prompt chunks, decode, batched slots, verify windows,
  every rank), so they pick the same experts in the same order.
- **Latent bottleneck** (optional): `ffn_latent_down` (`ffn_latent_in`) projects hidden to a smaller `latentDim`, the
  experts run in latent space, then `ffn_latent_up` (`ffn_latent_out`) projects back to hidden.
- **Shared expert** (optional): a ReLU^2 branch using `ffn_up_shexp` and `ffn_down_shexp`, added to the routed sum; the
  graph marks it a concurrent branch an executor may run beside the router.
- **Experts**: `ggml_mul_mat_id` over the stacked `ffn_up_exps` / `ffn_down_exps` (one product per projection for all
  selected experts of all rows), with the NVFP4 per-expert scales described above.

### 4.5 Vision encoder (`NemotronVisionEncoder`)

Mirrors NVIDIA's RADIO / CLIP-style ViT used by the v2_vl projector:

- **Linear patch embedding** with optional bias.
- **Position embedding** stored as a `[hidden, posTokens]` grid, bilinearly
  resized (align-corners = false) when the source patch grid differs.
  Resized embeddings are cached per `(gridW, gridH)` in
  `_resizedPositionEmbeddings`.
- **Class embedding** (optional) prepended to the sequence.
- **32 encoder blocks** (default for Nemotron Omni): LayerNorm → fused-QKV
  self-attention → residual → LayerNorm → up linear → GELU → down linear →
  residual. Bolt's mmproj stores Q, K and V (weights and biases) as separate
  `attn_q` / `attn_k` / `attn_v` tensors. The loader concatenates them into
  `attn_qkv` in Q, K, V row order, so split and fused files run the same
  weights. A block stored in both layouts, or a split triple with a part
  missing, is refused.
- After the encoder, the class tokens are stripped and a pixel-shuffle by
  `scaleFactor` reduces the spatial resolution.
- A projector (RMSNorm → Linear → ReLU² → Linear) emits embeddings in the
  LM hidden dim.
- **Checked at load.** Every tensor the forward reads must exist and match the
  `clip.vision.*` metadata. The projector's output width must equal the
  language model's hidden size (a Nano Omni mmproj projects to 2688, Bolt and
  Super VL to 4096). Otherwise the load fails with `InvalidDataException`,
  which names the file, the tensors and the expected layout. A server started
  with `--mmproj`, or the CLI, refuses to start (exit code 2). Before this
  check, such a file loaded and then failed every image request.
- **Final LayerNorm.** Super VL's mmproj ends the tower with `v.post_ln`
  (weight and bias), applied to every token after the blocks, as llama.cpp's
  `build_vit` applies it. Its bias is optional.
- **Not used:** the video tower's `v.patch_embd.weight.1` (skipped at load,
  with one log line).
- **Graph programs.** The tower and projector run as one graph program per tile
  on the language model's executor, on every backend and direct CUDA included
  (`NemotronVisionEncoder.Graph.cs`). The patch rows are packed on the host and
  the resized position table is cached per grid, as before, and the embeddings
  come back on the tower allocator. The managed per-op path loads and validates
  the weights once at load, then frees them. It still serves an allocator that
  has no executor. `NemotronVisionEncoderTests` holds the graph program to the
  managed path for the fused, split (Bolt) and Super VL layouts, on the trained
  grid and on a resized one: within 1e-5 relative L2 on the managed CPU, 1e-3
  on direct CUDA, and 2e-3 on `ggml_cpu` and `ggml_cuda`, where GGML's
  attention rounds K and V to F16.

`NemotronImageProcessor` (mirrors `nemotronh.ImageProcessor`):

1. Composite RGBA over white background.
2. Choose a tile grid that best matches the source aspect ratio (max
   `maxTiles`), or run dynamic-resolution mode if min/max patches metadata
   is present. Runtime inference caps tiled images to 1 tile by default to
   keep first-token latency practical for server image chats; set
   `TS_NEMOTRON_IMAGE_MAX_TILES=12` (or the model's advertised max) to restore
   full-resolution tiling.
3. Bicubic resize to `gridW * imageSize × gridH * imageSize` (or dynamic
   patch grid).
4. Crop into `imageSize × imageSize` tiles, channel-first `[C, H, W]`.
5. Optional thumbnail tile when more than one tile was produced.
6. Normalize each tile with the CLIP mean/std loaded from the mmproj.

The multimodal injector (`ProcessNemotronHistory` in
`ModelMultimodalInjector.cs`) tokenizes each image into one
`<image>` placeholder, expands it into `<img>` + N tokens + `</img>`, runs
the vision encoder on every tile, concatenates the per-tile embeddings, and
queues a `PreparedEmbeddingSpan` so the model splices the embeddings into the
right positions before `Forward()`.

### 4.6 Audio frontend (`NemotronAudioPreprocessor`)

Parakeet-style log-mel spectrogram extraction (mirrors ollama's
`process_audio.go`):

- Mono resample to 16 kHz.
- 0.97 pre-emphasis.
- STFT: `n_fft = 512`, `hop = 160`, `win = 400`, center-padded constant.
- Slaney-style mel filter bank with 128 bins, 0..8 kHz.
- `log(power + 2⁻²⁴)` followed by per-mel mean/var normalization across
  valid (non-padded) frames.

The chat template emits a `<so_embedding>` token for each uploaded audio
file, but nothing can fill it: the Parakeet/FastConformer tower that turns
the log-mel frames into embeddings, and the projector behind it, are not in
the public Nemotron 3 Nano Omni GGUFs. The `mmproj-BF16.gguf` from the
unsloth repository holds exactly 390 tensors - 32 `v.blk.*` vision blocks,
`v.patch_embd` / `v.position_embd` / `v.class_embd` and the three
`mm.model.mlp.*` projector tensors - with `clip.has_vision_encoder=true` and
no audio key, and the 401-tensor language GGUF has no audio tensor either.
Stock llama.cpp answers the same checkpoint with "This model does not
support audio input"; audio there needs a community fork and a unified
`mmproj` that carries the audio tower.

Unless an audio tower was loaded from a companion GGUF (§4.7), TensorSharp
therefore **refuses** audio for this family rather than decoding the clip,
warning, and generating as if none had been sent (which is what it did before:
the model saw an unfilled `<so_embedding>` and answered the text alone). One
table, `AudioInputSupport.UnsupportedReasonFor`, keyed on the architecture
through the registry so every alias (`nemotron_h`, `nemotron_h_moe`,
`nemotron_h_omni`) is covered and told whether the loaded model reports
`NemotronRuntimeModel.IsAudioEncoderLoaded`, drives every entry point:

- `/v1/chat/completions` and `/v1/responses` scan the whole request before
  writing any upload and answer **400** `invalid_request_error` with
  `NemotronRuntimeModel.AudioInputUnsupportedMessage` for any `input_audio` /
  `audio_url` part, including malformed ones and ones that follow an image;
  image-only requests still parse.
- The Web UI answers 400 with the same message before the stream opens.
- The CLI refuses `--audio` once the model is loaded, before the clip is
  decoded or a turn is generated, and `/audio` in the REPL declines to attach
  the file.
- `ModelMultimodalInjector.ProcessNemotronHistory` throws
  `NotSupportedException` with the same message for any caller that bypasses
  those gates.

`NemotronAudioRefusalTests` covers these gates (with and without a loaded
tower) and `NemotronOmniMmprojContractTests` (gated on `TS_TEST_NEMOTRON_MMPROJ`)
pins the published mmproj's vision-only layout; if a distribution ever ships
the tower in that file, it is the test that fails first.

### 4.7 Audio tower (`NemotronAudioEncoder`, companion GGUF)

`NemotronAudioEncoder` runs NVIDIA's Parakeet/FastConformer encoder
(subsampling convolutions, relative-position attention, convolution blocks)
and the `sound_projection` MLP into the language model's hidden size, one clip
at a time so a neighbouring clip's padding can neither change its length nor
leak into its bidirectional attention. It reads a **companion GGUF** that keeps
the official `sound_encoder.encoder.*` / `sound_projection.*` tensor names with
`nemotron.audio.*` hyperparameters (`general.architecture=nemotron_audio`).
No public repository ships such a file; the conversion scripts that extract it
from NVIDIA's BF16 checkpoint are in
[`eng/nemotron-audio`](../../eng/nemotron-audio/README.md)
(`prepare.py`; `prepare_f32.py` writes the same weights with
`nemotron.audio.compute_bf16=false`).

Loading follows the tensors, not a flag. `LoadProjectors` hands the `--mmproj`
path to both towers: the vision encoder loads only when the file has `v.*`
tensors, and the audio encoder only when it has `sound_projection.linear2.weight`
(or, when `TS_NEMOTRON_AUDIO_MMPROJ` names a file, from that file instead, so a
vision mmproj and an audio companion can be used together). The encoder then
validates every tensor shape and the Parakeet sampling configuration, and the
projection width must equal the language model's hidden size; a partial or
mismatched companion fails the load with `InvalidDataException` instead of
being served. Only then does `IsAudioEncoderLoaded` lift the §4.6 refusal, and
`ProcessNemotronHistory` plans every image and clip against the original prompt
(`PlanNemotronMedia`, per-modality attachment order), expands each
`<so_embedding>` to `<so_start>` + N + `<so_end>`, and queues the projected rows.

**Two paths, chosen at load** (`NemotronAudioEncoder.Load`):
- **F32-compute companion** (`nemotron.audio.compute_bf16=false`, `prepare_f32.py`'s file). The encoder runs as one
  graph program per clip on the language model's executor, on every backend and direct CUDA included
  (`NemotronAudioEncoder.Graph.cs`). The program is the managed path's arithmetic:
  - **Subsampling.** `Conv2d` and `DepthwiseConv2d` stages with the rows past the valid frames zeroed, then pointwise
    products.
  - **Relative-position attention.** Content and position scores as per-head products, the relative shift a strided view,
    the keys past the valid frames masked.
  - **Conv module.** GLU, then a 'same' depthwise conv as `SsmConv` over a zero-padded window, then the inference
    BatchNorm folded into a scale and a shift.

  The matrices keep their BF16 encoding where the backend has kernels for it, and the vectors and conv kernels are F32.
  The embeddings come back on the tower allocator (host memory on direct CUDA, as before).
- **BF16-emulating companion** (`prepare.py`'s file). It keeps the managed path, which rounds every intermediate as the
  official BF16 encoder does.

What is established:
- **Frontend.** The Parakeet mel frontend matches an independent reference on six clips (`NemotronAudioInputTests`).
- **Encoder against the reference.** The encoder and projector match the official Transformers modules on small 8- and
  128-mel fixtures (`NemotronAudioEncoderTests`): in F32 and BF16 compute on the managed path and GGML CPU; and in F32
  for the graph program on the managed CPU, `ggml_cpu`, `ggml_cuda` and direct `cuda`, at the managed path's F32
  tolerance.
- **Injection.** Two-clip injection and sliced queueing keep exact rows and independent request retention
  (`NemotronAudioInjectorTests`).
- **Trained companion.** The explicit F32-compute configuration matches all 8,064 output values. The original BF16
  compute configuration still **fails** (3,794/8,064 at an unchanged tolerance, first difference a BF16 rounding
  boundary in layer 0).
- **Real clip.** On the trained F32-compute companion and a 10.44 s LibriSpeech clip (131 rows), the graph program
  encoded in:
  - 24.4 ms on `ggml_cuda` and 48.1 ms on direct `cuda` (A40);
  - 481 ms on `ggml_cpu`;
  - against 1.8-1.9 s for the managed path, under the test VM's 7.65-CPU quota.

  The rows are within 1.8e-3 relative L2 of the managed path's (the BF16 weights' products round their activations;
  `eng/validation/NemotronRuntimeAB --scenario audio`).
- **Speech.** Nano Omni (`UD-Q4_K_XL`, the BF16 mmproj and this companion through `TS_NEMOTRON_AUDIO_MMPROJ`) was asked
  to transcribe the clip. It answered with the reference transcript word for word on `ggml_cuda` and on `cuda`: "He hoped
  there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in
  thick peppered flour-fattened sauce."
  - The 155-token prompt prefilled in 245 ms on `ggml_cuda` and 617 ms on `cuda`.
  - The CLI was run with `--no-skills`. With its default skill catalog the model answered the request with a skill tool
    call instead of the transcript.

## 5. Parameters and settings

| Key | Type | Meaning |
|---|---|---|
| `nemotron_h.ssm.conv_kernel` | uint32 | Mamba2 conv1d kernel size |
| `nemotron_h.ssm.inner_size` | uint32 | SSM inner dim (`nHead * headDim`) |
| `nemotron_h.ssm.state_size` | uint32 | SSM state dim per head |
| `nemotron_h.ssm.time_step_rank` | uint32 | Number of SSM heads |
| `nemotron_h.ssm.group_count` | uint32 | Number of SSM groups |
| `nemotron_h.attention.head_count_kv` | uint32[] | Per-layer KV head count (0 ⇒ Mamba2) |
| `nemotron_h.attention.head_count` | uint32[] | Per-layer Q head count |
| `nemotron_h.feed_forward_length` | uint32[] | Per-layer FFN size (0 ⇒ no FFN) |
| `nemotron_h.attention.scale` | float32 | Attention scale factor (0 ⇒ auto) |
| `nemotron_h.expert_count` | uint32 | MoE expert count (0 ⇒ dense); each layer's count is taken from its stacked expert tensor, which on Bolt varies from 240 to 496 (this scalar is the largest) |
| `nemotron_h.expert_used_count` | uint32 or uint32[] | TopK experts per token; Bolt stores a per-layer array (0 on non-FFN layers), and each FFN layer uses its own value |
| `nemotron_h.expert_weights_norm` | bool | Renormalize selected expert weights to sum=1 |
| `nemotron_h.expert_weights_scale` | float32 | Global scale factor applied to expert weights |

For the Omni vision projector (`mmproj` GGUF), the standard `clip.vision.*`
keys (`embedding_length`, `feed_forward_length`, `attention.head_count`,
`block_count`, `image_size`, `patch_size`, `num_channels`, `projection_dim`,
`projector.scale_factor`, `attention.layer_norm_epsilon`, `use_gelu`) are
read by `NemotronVisionEncoder`.

## 6. Weight naming convention

```
token_embd.weight
output_norm.weight
output.weight

# Mamba2 layers:
blk.{L}.attn_norm.weight
blk.{L}.ssm_in.weight       # [hidden → 2*dInner + 2*nGroup*dState + nHead]
blk.{L}.ssm_conv1d.weight   # [xBCSize, convKernel]
blk.{L}.ssm_conv1d.bias     # (optional)
blk.{L}.ssm_dt.bias         # [nHead]
blk.{L}.ssm_a               # [nHead]
blk.{L}.ssm_d               # [nHead] (optional)
blk.{L}.ssm_norm.weight     # group RMSNorm [dInner]
blk.{L}.ssm_out.weight      # [dInner → hidden]

# Attention layers:
blk.{L}.attn_norm.weight
blk.{L}.attn_qkv.weight     # fused Q+K+V (or separate attn_q/k/v.weight)
blk.{L}.attn_output.weight

# FFN layers (dense):
blk.{L}.attn_norm.weight
blk.{L}.ffn_up.weight
blk.{L}.ffn_down.weight

# FFN layers (MoE):
blk.{L}.attn_norm.weight
blk.{L}.ffn_gate_inp.weight       # router [numExperts, hidden]
blk.{L}.exp_probs_b.bias          # router bias (optional)
blk.{L}.ffn_latent_in.weight      # latent bottleneck in (optional)
blk.{L}.ffn_latent_out.weight     # latent bottleneck out (optional)
blk.{L}.ffn_up_exps.{E}.weight
blk.{L}.ffn_down_exps.{E}.weight
blk.{L}.ffn_up_shexp.weight       # shared expert (optional)
blk.{L}.ffn_down_shexp.weight
```

## 7. TensorSharp implementation: the shared decoder runtime

Nemotron-H runs on the shared decoder runtime (`StandardDecoderModel`), as Gemma 4, Qwen 3.5 and Mistral 3 do.
[`NemotronRuntimeModel`](../../TensorSharp.Models/Models/Nemotron/NemotronRuntimeModel.cs) declares the family's
blocks and weights. The runtime provides batching, the prefix cache, checkpoints, placements, speculation, expert offload
and LoRA. The legacy per-op `NemotronModel` was removed after the validation in §11. So were its native Mamba-2 kernels
(`ggml_ops_mamba2.cpp`), its per-token MoE expert kernel (`TSGgml_MoEExpertsForwardF32`), its fused matmul+add and
single-layer flash-decode entry points, and its switches: `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE`,
`TS_NEMOTRON_MAMBA2_NATIVE_PREFILL_MIN_TOKENS`, `TS_MAMBA2_PREFILL_CACHE_MB`, `TS_NEMOTRON_ATTN_SCORE_BUDGET_MB`,
`TS_NEMOTRON_MOE_PREFILL_BATCHED_MIN_TOKENS`, `TS_NEMOTRON_MULTIMODAL_WARMUP(_TILES)` and `TS_TP_SNAPSHOT_TRACE`. Setting
one now stops startup and names the change (`RemovedCliFlags`). `TS_NEMOTRON_IMAGE_MAX_TILES` and
`TS_NEMOTRON_AUDIO_MMPROJ` still apply.

**Blocks.** Each layer is exactly one block, read from the per-layer arrays: Mamba-2 (`head_count_kv` and
`feed_forward_length` both 0), attention (`head_count_kv` > 0) or feed-forward (`feed_forward_length` > 0). The runtime's
block layout skips the parts a layer does not have. The family runs whole-forward graphs only
(`RequiresGraphs`): every step, whether a prompt chunk, a decode step, batched slots or a verify window, is one graph on the
backend's tensor executor. That is the GGML graph builder on `ggml_*`, `CudaGraphExecutor` on `cuda`, `MlxGraphExecutor`
on `mlx`, and the managed executor on `cpu`.

**Mamba-2 mixer** (`EmitRecurrentMixer`), per sequence of the step:
1. `ssm_in` splits into `z | xBC | dt`.
2. The conv window (the state's last `conv_kernel - 1` rows) is concatenated with the chunk's `xBC` rows and passed
   through `ssm_conv`, plus the conv bias and SiLU. The window's tail is written back.
3. `dt + dt_bias`, then the selective scan over all of the chunk's rows in one `GraphOp.SsmScan`. The legacy model
   scanned token by token. GGML runs upstream `ggml_ssm_scan` unchanged; direct CUDA runs `ts_graph_ssm_scan_f32` (one
   warp per state row); the managed executor runs a SIMD loop.
4. `y + x * D`, `y * SiLU(z)`, the grouped RMSNorm, then `ssm_out`.

A verify window keeps the scan's state after every row (`ggml_ssm_scan`'s K snapshots) and the conv history. Committing
an accepted prefix copies that row's snapshot into the live state (`CommitRecurrentPrefix`), so a rejected draft rewinds
exactly without re-running the prefix.

**Attention** has no positional encoding (`AttentionIsPositionless`): the runtime's paged K/V and attention kernels, at
`attention.scale` when the GGUF sets it.

**Feed-forward** (`EmitFeedForward`): dense `down(relu(up x)^2)`, or the routed experts of §4.4 (router, selection bias,
renormalized and scaled weights, NVFP4 scales, latent bottleneck, shared expert, per-layer expert counts and top-k).

**Weights** stay in their GGUF formats on every executor. NVFP4 per-tensor `.scale` sidecars multiply their products,
and the tensor-parallel shards keep them.

## 8. Serving mechanisms

| Mechanism | Backends | Notes |
|---|---|---|
| Continuous batching | all | Batched slots: each slot holds a Mamba-2 conv window and SSM state, and the attention layers share the paged K/V pool. Concurrent requests run in batched steps. |
| Prefix cache, checkpoints | all | The runtime's holder adapter. A cached prefix carries the Mamba-2 states at its checkpoints, so a continuation restores K/V rows and recurrent state together. `CheckpointsContinueExactly` covers it. |
| `--tp N` | cuda, ggml_cuda, ggml_vulkan | Per-rank graphs. Mamba-2 is sharded by contiguous blocks of groups: `ssm_in` row blocks, and one `ssm_out` column block per rank, so a quantized row never splits inside a block. Attention is sharded by heads and every expert by its width. Batching is kept. Covered on simulated devices only (§11). |
| `--layer-split N` | cuda, ggml_cuda, ggml_vulkan | Whole-layer stages, batching kept. |
| Hybrid (`--tp` x `--layer-split`) | none (gap) | The graph-only family has no graphs for tensor-parallel stages yet. |
| `--n-cpu-moe` / `--cpu-moe` | all | On one device or over stages; refused with `--tp`. See §10. |
| Speculation | all | n-gram drafts, the Nemotron 3.5 MTP head (the trunk's NextN block, or an MTP GGUF via `--draft-model`), and the DFlash and DSpark block drafters (`--draft-model`, below), verified in windows that rewind the Mamba-2 states exactly. |
| Text LoRA | cpu, ggml_cpu, ggml_cuda, cuda | One device. Each adapted product's delta is in the graph (`AppliesTextLoraInGraphs`). The targets are listed below. |
| Media | all | The vision and audio towers' embeddings enter as queued media rows written over their placeholder rows (`QueueMedia`). |

### Text LoRA adapters

Nemotron-H takes text LoRA adapters (`--lora`, `/v1/loras`, a request's `lora`) on one device, and refuses them under
`--tp` or `--layer-split`. Every projection outside the routed experts is adapted:
- Mamba2 `ssm_in` and `ssm_out`
- `attn_q/k/v` (fused into `attn_qkv` at load) and `attn_output`
- the dense `ffn_up` and `ffn_down`
- the shared experts
- the latent `ffn_latent_down` / `ffn_latent_up`
- the router and the LM head

An adapter on a routed expert (`*_exps`) is refused with a 400 that names it. The deltas are applied inside the step's
graph, so decode, batched slots and verify windows all apply them, and the forward audit throws if a planned weight was
skipped. This is validated on synthetic models against a merged-weights twin (`TextLoraModelParityTests` on cpu,
ggml_cpu, ggml_cuda and cuda; `TextLoraPlanTests`), not on a real checkpoint. Details: [design
doc](../design/logprobs-lora-moe-runtime-features.md) §2.2.

### Block drafters (DFlash / DSpark)

The Lightning repository ships two block drafters beside the MTP head; either attaches with `--draft-model`:

```bash
dotnet run --project TensorSharp.Cli -c Release -- --model models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf \
  --draft-model models/dspark-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.gguf --backend ggml_cuda
```

| | DFlash (`dflash-...-NVFP4.gguf`) | DSpark (`dspark-...-NVFP4.gguf`) |
|---|---|---|
| Layers | 6 (hidden 2688, 32 query heads over 2 KV heads, head_dim 128); BF16 attention, NVFP4 FFN and `fc` with `.scale` sidecars | the same, plus per-head attention sinks |
| Context | every layer attends the whole history (no `sliding_window`); YaRN rotary, factor 128 over 8192 | a 1,024-token sliding window |
| Block | 16 rows, non-causal; rows 1 .. 15 are drafts | 8 rows, block-causal; a bonus anchor (`sample_from_anchor = false`): rows 1 .. 7 are drafts |
| Head | the target's LM head, top-1 of each row | the target's LM head plus a rank-512 Markov head: draft e's logits gain `w2 · w1[prev]`, `prev` the anchor, then each draft's own choice |
| Encoder | trunk layers 2, 6, 20, 30, 42, 52 (52 is the stream leaving the last layer) | the same |

Both run as the shared runtime's block-draft graph programs (`StandardDecoderModel.BlockDraft.cs`) on every executor:
- **Markov head.** The chain runs on the device: `GetRows(w1, prev)`, `w2`'s product with its scale, and each draft's
  top-1. Only the drafted ids and their confidences come back.
- **Attention sinks.** Each sink is the attention op's fifth operand. The GGML builder passes it to
  `ggml_flash_attn_ext_add_sinks`. The tensor executors attend a sink layer through their generic products, with the
  sink as one more column of each row's softmax (`TensorGraphExecutor.FlashAttentionWithSinks`).
- **Mask.** The block mask closes slot i to the block's queries before row i when the drafter's `attention.causal` is set.
- **DFlash's full attention.** It is served with a window of the trunk's context, at most 8,192 positions: a ring of
  8,224 rows, 96 MB of F32 K/V. Past that much context a draft sees the latest 8,192 positions. Drafts are verified, so
  this can lower acceptance but never changes the output.
- **Drafter embedding.** The drafter's own BF16 token embedding stays on the host in its file encoding. Loading it as an
  F32 tensor would have put 1.4 GB on the device, where nothing read it.

The per-op drafter follows the same rules (YaRN, the causal block, the bonus anchor's rows). The fused GGML drafter
takes neither YaRN nor a causal block, so these two files draft through the graph programs, or per op where those
decline.

Lightning Q4_0 on one A40, greedy, 128 generated tokens a prompt. Each row reports:
- the acceptance length τ, which is (accepted + verifies) / verifies, so a verify's bonus token counts;
- the decode rate relative to plain decoding.

NVIDIA reports τ of 3.16 (DFlash) and 3.75 (DSpark) at draft length 7.

| Prompt | DSpark x7 ggml_cuda | DSpark x7 cuda | DFlash x7 ggml_cuda | DFlash x7 cuda | MTP x3 ggml_cuda / cuda |
|---|---|---|---|---|---|
| Fibonacci function (code) | τ 5.13, 1.29x | τ 5.64, 1.53x | τ 3.54, 1.10x | τ 3.18, 1.19x | 1.16x / 1.38x |
| Why the sky is blue (3 bullets) | τ 2.80, 0.87x | τ 2.77, 1.03x | τ 1.73, 0.74x | τ 2.00, 0.91x | 1.13x / 1.15x |
| Train speed (arithmetic) | τ 6.19, 1.52x | τ 5.46, 1.58x | τ 3.50, 1.11x | τ 3.33, 1.22x | 1.24x / 1.29x |
| German translation | τ 2.76, 0.85x | τ 3.74, 1.24x | τ 2.17, 0.83x | τ 2.00, 0.91x | 1.12x / 1.17x |

Plain decoding ran at 140 tok/s on `ggml_cuda` and 68-71 tok/s on `cuda`. With 3-token windows, DSpark ran the code
prompt at 1.35x and the arithmetic prompt at 1.39x on `ggml_cuda`.

**What bounds the speed-up.** On this MoE trunk an 8-row verify reads the experts of every row: about 23 ms on `ggml_cuda`
against 7.1 ms for a decode step. A draft costs 4.4 ms (DSpark's block graph) plus 0.35 ms (the ring inject) at steady
state.

**Parity.** A speculative stream follows plain greedy decoding until a verify window's kernels flip a close pick. On
`ggml_cuda` the streams parted where plain decoding's top two logits were 0.004-0.088 apart. On `cuda` they parted at up to
0.21, and the MTP head parted at the same token, so this comes from the direct-CUDA verify rows' rounding (§11), not from a
drafter.

Synthetic coverage (`GraphBlockDraftTests`): a DSpark-shaped drafter (Markov head, sinks, causal block, bonus anchor, a
target past the stack) and a full-attention YaRN drafter with its own embedding are checked as follows:
- their graph drafts match the per-op drafter's (ids, confidences within 1e-3, ring rows within 1e-4) on the managed
  CPU, `ggml_cpu`, `ggml_cuda` and direct `cuda`;
- speculation with each emits plain greedy decoding's stream on the managed CPU, `ggml_cuda` and `cuda`.

`DSparkConfigTests` covers the exports' keys.

Limits:
- `mlx`, `ggml_metal` and `ggml_vulkan` were not run.
- Several requests' DSpark blocks do not share one program: each request drafts its own block (DFlash keeps the batched
  lanes).
- On direct CUDA, a sink layer takes the generic attention rather than the ring kernel.

## 9. Memory

- **Attention layers**: the runtime's paged K/V pool (F16 by default; `--kv-cache-dtype`).
- **Mamba-2 layers**, per slot: a conv window of `(conv_kernel - 1) x (d_inner + 2 x groups x d_state)` floats and an SSM
  state of `heads x head_dim x d_state` floats. A slot's verify buffers (each row's state snapshot and the conv history)
  are allocated the first time it speculates, at a power-of-two window capacity of at least 4.
- **Weights**: resident on the device, except the routed experts of offloaded layers (§10). Load is fast: Lightning
  loaded in 0.7 s on `ggml_cuda`, against 10.7 s for the legacy model.

## 10. Expert offload (`--n-cpu-moe` / `--cpu-moe`)

An offloaded layer's routed experts stay in host memory, and the step's graph pauses for them. The router, the selection,
the latent projections and the shared expert stay on the device. On the GGML backends the host MoE segment computes them
from the GGUF mapping. On the tensor executors (`cuda`, `mlx`, `cpu`) the runtime's expert host computes them. Both use
ggml's CPU kernels: one projection, `relu(s * up)^2`, then `down`, where `s` is each route's NVFP4 up scale. A decode or
verify step uses the one-token decode team (`TSGgml_HostExpertRowsReluSquared`); a GGML prompt chunk uses the batched CPU
MoE graph. On a tensor executor whose device is not the host, a prompt chunk wider than 16 rows streams its selected
experts to the device for the step.

The accelerator tiers (`--moe-hot-experts`, the CUDA doorbell's packed copies) serve gated experts only, so a Nemotron
layer is computed entirely on the host. With `--tp`, offload is refused at load, because the per-rank graphs keep their
experts on the devices.

Measured on one A40 (ggml `bb877c0c`), every MoE layer offloaded (`TS_CPU_MOE=1`), against the same model resident:

| Model, backend | NLL resident -> offloaded | Decode tok/s | Prefill 512 tok/s (warm) | Peak VRAM |
|---|---|---|---|---|
| Lightning Q4_0, ggml_cuda | 2.1954 -> 2.2013 | 143.2 -> 32.8 | 4633 -> 851 | 18.8 GB -> 3.7 GB |
| Lightning Q4_0, cuda | 2.1966 -> 2.1980 | 70 -> 24.4 | 1724 -> 27.5 | 3.9 GB offloaded |
| Bolt-24GB, ggml_cuda | 2.0317 -> 2.0328 | 70.8 -> 16.3 | 1717 -> 608 | 6.5 GB offloaded |

The NLL moves by up to 0.006 because the host kernels quantize the activations to the weights' dot types. The legacy
model offloaded only under `--tp` on GGML, which needs two GPUs, so it has no number here. On one device it saved no VRAM.

## 11. Validation of the migration

Dependency: ggml `bb877c0c51f38c52ee2d5e41d58e3d0c20f3fed4`, upstream and unchanged. `ggml_ssm_scan` and `ggml_ssm_conv`
are its own ops. TensorSharp adds only its graph builder's lowering and the direct-CUDA scan kernel.

**Synthetic fixtures** (`NemotronSyntheticModelBuilder`, `NemotronRuntimeTests`). The four layouts are dense; MoE with a
shared expert; latent MoE with NVFP4-style sidecars; and per-layer expert counts and top-k. They are held to the legacy
model's logits, frozen on the managed backend before it was deleted (`Fixtures/Nemotron/LegacySynthetic/managed.json`).
`ggml_cpu`, `ggml_cuda` and direct `cuda` are checked against the managed backend. Also covered: chunked prefill, every
placement over simulated devices, batched slots, concurrent requests, checkpoints, speculation, host experts, NVFP4
scales against folded twins, and LoRA.

**Real checkpoints** (one A40; teacher-forced argmax agreement with the legacy model, 512-token NLL, the CLI's benchmark):

| Model, backend | Argmax vs legacy | NLL legacy -> runtime | Decode tok/s | Prefill 512 tok/s | Prefill 4K tok/s |
|---|---|---|---|---|---|
| Nemotron-H 8B Q4_K_M, ggml_cuda | 218/219 | 2.6113 -> 2.6078 | 45 -> 104 | ~480 -> 3600 | 441 -> 4152 |
| Nemotron-H 8B Q4_K_M, cuda | 171/171 | 2.6107 -> 2.6091 | 33 -> 89.6 | 92.7 -> 2075 | 87.6 -> 2706 |
| Lightning 30B-A3B Q4_0, ggml_cuda | 242/246 (near ties) | 2.2001 -> 2.1954 | 39.6 -> 143.5 | ~300 -> 4690 | 482 -> 6885 |
| Lightning 30B-A3B Q4_0, cuda | 198/200 (near ties) | 2.1969 -> 2.1966 | 30 -> 70 | ~160 -> 1724 | 157 -> 3261 |
| Bolt 74B-A9B 24GB, ggml_cuda | 200/200 (rel L2 ~0.6 %) | 2.0287 -> 2.0317 | 17.7 -> 70.8 | 131 -> 1717 | 142.7 -> 2215 |

- **Multimodal.** Nano Omni with its BF16 mmproj gave correct image answers on both models. Its 2,466-token image
  prefill took 8,367 -> 798 ms, and decode went from 28.4 to 116.7 tok/s.
- **Concurrency.** On the server, 8 concurrent Lightning prompts finished in 33.3 s at 14.4 tok/s on the legacy model and
  in 2.4 s at 181.6 tok/s on the runtime. The runtime matched each prompt served alone on 6 of 8 prompts, against 4 of 8
  for the legacy model; the others parted at near ties.
- **Speculation** (Lightning on `ggml_cuda`; the legacy model refused it). The DFlash and DSpark drafters are measured in
  §8. With the MTP head over 3-token windows:
  - acceptance was 64.5-85.7 % and decode ran 1.12-1.24x as fast as plain decoding on prompts of 46-96 generated
    tokens;
  - n-gram drafts ran at 0.95-1.03x.
  - A speculative stream follows plain greedy decoding until a verify window's kernels, which run at another batch shape,
    flip a close pick. MTP streams parted where plain decoding's top two logits were 0.06-0.19 apart, and n-gram streams
    where they were 0.04-0.06 apart. One n-gram stream parted at a margin of 0.82.
  - Verify windows matched the plain steps' argmax on 24/24 rows, and on 24/24 and 23/24 rows with rewinds.
- **Coverage limits.** These were not run:
  - `mlx`, `ggml_metal` and `ggml_vulkan`;
  - multi-GPU `--tp` and `--layer-split`, which were covered on simulated devices only;
  - Super VL without `--cpu-moe` (it needs a 96 GB device);
  - an audio companion, since none is public.
- **Known costs.**
  - Direct-CUDA MoE decode steps are not recorded as CUDA graphs: Lightning decodes at 70 tok/s on `cuda` against 143
    on `ggml_cuda`.
  - Lightning's first 512-token prefill after load builds its graphs: 2,768 ms against 1,782 ms for the legacy model.
    Later runs take 110 ms.
  - Offloaded prompt chunks on direct CUDA stream their experts one at a time (27.5 tok/s, §10).
  - The direct-CUDA known defect D6 ([README-direct-cuda-prefill](../../eng/validation/README-direct-cuda-prefill.md))
    carried over: on two synthetic MoE files a single forward of more than 512 rows can end far from the references.
    It is narrowed to the large-row F16 route of the quantized products. A 1,100-token Lightning forward on `cuda`
    stays 1.6e-2 from `ggml_cuda`.

## 12. Output parser and chat template

- `NemotronOutputParser` (a `ChatMlOutputParser`) parses `<think> ... </think>`
  for chain-of-thought reasoning and tool calls in `<TOOLCALL>...</TOOLCALL>` or
  `<tool_call>...</tool_call>`. A call body may be a JSON call object, a JSON
  list of them (the Nemotron convention), the XML `<function=NAME>` /
  `<parameter=key>` form, or a Python-style list of calls such as
  `[read_file(path="notes.txt"), shell("wc -l notes.txt")]`
  (`PythonicToolCalls`). The Reasoning-128K 8B Q4_K_M wrote every call in
  that Python form, whatever format the prompt asked for. Python calls take
  literals only, a positional argument binds to the declared parameter at that
  position, and a body cut off by EOS or the token limit yields no call. An
  open tag starts a call only when a call body follows it, so reasoning that
  names the tag stays reasoning. A body of any other shape yields no call.
  Such a body used to throw from the parser and abort the streamed HTTP
  response.
- `response_format` combines with `"think": true`. Thinking on primes
  `<think>\n` after the assistant marker (the Nemotron 3.5 GGUF template does
  the same), so the JSON grammar stays dormant through the reasoning and arms
  after the model's `</think>` (`ThinkingGrammarActivationTrigger`). Before, the
  request was refused with HTTP 400. `</think>` is also the family's
  `ThinkingBudgetEndToken`: on Nemotron 3.5 / Omni, whose vocabulary has it as
  one token, reaching `TS_THINKING_BUDGET` (75% of an allowance of at least 512
  tokens) emits it and the constrained answer follows inside `max_tokens`. The
  Nemotron-H Reasoning-128K GGUFs spell `</think>` in several tokens, so they
  keep the explained `thinking_budget` stop. Measured on Nemotron 3.5 Lightning
  IQ4_XS (Metal): with `max_tokens` 256 the reasoning alone used the whole
  allowance (content empty, `finish_reason=length`); with 1024, json /
  json_schema / json_unicode passed at c1 and json_schema / json_unicode 4/4 at
  c4. json at c4 was 1/4: three of the four requests reasoned about a different
  prompt ("User says: Mars"), the batched-prefill corruption tracked separately.
- Whitespace between `</think>` and the JSON is part of the prelude
  (`GrammarConstraint.ActivateAfter(trigger, skipLeadingWhitespace: true)`, used
  by `response_format` only). The Nemotron-H 8B Reasoning-128K vocabulary spells
  `</think>\n\n` as `</think` + `>\n\n`; the JSON root cannot start with a
  line break, so that token was masked, the model wrote `</think}` instead, the
  trigger never matched and the whole reply stayed reasoning (content null,
  `json_schema` HTTP 422). Measured on Nemotron-H 8B Q4_K_M (RTX PRO 6000,
  `--thinking`, c1 and c4, three repeats, `max_tokens` 256): 10 replies wrote
  `</think}` before, none after; json_schema 4/15 -> 15/15, json 13/15 -> 14/15.
  json_unicode stayed 0/15 because the reasoning alone reaches 256 tokens; with
  `max_tokens` 1024 it is 5/5 (c1 + c4). At 1024 json c4 was 0/4 (the objects
  omit `moons` or reason about a different prompt) and json_schema c4 2/4 (two
  requests hit the explained `thinking_budget` stop at 768 tokens).
- Two turn formats ship under the same architecture name, and the GGUF's
  embedded `tokenizer.chat_template` decides which one is rendered
  (`ChatTemplate.IsNemotronHReasoningTemplate`):
  - **Nemotron-H 8B/47B Reasoning-128K** were trained on
    `<SPECIAL_10>System\n{system}\n<SPECIAL_11>User\n{user}\n<SPECIAL_11>Assistant\n`
    (EOS is `<SPECIAL_11>`). Reasoning is switched by
    `{'reasoning': True}` / `{'reasoning': False}` in the system prompt, and
    the generation prompt then opens (`<think>\n`) or closes
    (`<think></think>`) the reasoning block. `RenderNemotronHReasoning` adds
    the marker from the request's `think` flag unless the system prompt
    already carries one, and puts it at the end of the system section, after
    the system text and the tool declarations: both thinking modes then
    render the same system prompt and tools before they differ. Prepended,
    they diverged at the fifth token, and a warm-up or cached prefix of one
    mode was useless to the other. The shipped template has no tool syntax,
    so tools are declared with the Nemotron family's own convention, word for
    word as NVIDIA's Nemotron templates render it (NVIDIA-Nemotron-Nano-9B-v2,
    Llama-Nemotron Super v1.5): the declarations in
    `<AVAILABLE_TOOLS>[...]</AVAILABLE_TOOLS>`, then the instructions to call
    in `<TOOLCALL>[...]</TOOLCALL>` and to expect results in
    `<TOOL_RESPONSE>[...]</TOOL_RESPONSE>`. A past call is rendered back as
    `<TOOLCALL>[...]</TOOLCALL>`, and a turn's tool results go back as one
    user turn wrapped in `<TOOL_RESPONSE>[...]</TOOL_RESPONSE>`. On the 8B
    Q4_K_M (two tool tasks, three seeds per arm), behind a 16k-character
    skills prompt the model called a tool 7 times in 12 with this block and
    2 in 12 with the Hermes JSON `<tool_call>` block; with only the tools,
    10 in 12 against 8. After its call the model went on to write a made-up
    tool response and answer from it, so a turn ends at its first complete
    call (`ChatProtocol.ToolCallEndsTurn`, `ToolCallTurnEnd`). The family's
    own parser decides, so a tag named in the reasoning, or a close tag
    inside a string argument, does not end the turn. The model puts all of a
    turn's calls in one list inside one tag, so no call is lost. These
    checkpoints used to be rendered as ChatML: the model saw `<|im_start|>` as
    plain text and answered with `</think>`, invented `<|im_start|>user` turns
    and `<unk>` loops.
  - With thinking off the Reasoning-128K checkpoints sometimes reason anyway,
    after the closed `<think></think>`, and close the block themselves before
    the call or the answer. This happened mostly in tool rounds after a tool
    result. In 98 logged replies of the 8B Q4_K_M, the close came after 320-1878
    characters of "Okay, the user ..." (median 684), and later the deeper the
    tool loop. That reasoning and the literal `</think>` used to be
    shown, and saved, as the answer. The protocol declares this per template
    (`ChatProtocol.ThinkingOffReplyMayReason`). The pipeline tells every parser
    of the reply what the prompt ended with, and the parser holds the start of
    the reply as undecided. A `</think>` makes it reasoning. A complete tool
    call or the end of the reply makes it the answer. An empty call list
    (`<TOOLCALL>[]</TOOLCALL>`, in either call tag and in every Nemotron-H
    template) counts as no call and adds no text. A
    `</think>` inside a Markdown code block or code span is quoted text, and
    so is one that starts after the first 2,048 characters. All 98 logged
    stray closes fall inside that window (95th percentile 1,601). A
    1,024-character window covered only 79%, and the reasoning it missed still
    reached held streams. The
    replay that measures this runs the real parser over logged replies:
    [`eng/validation/NemotronThinkOffReplay`](../../eng/validation/NemotronThinkOffReplay/README.md).
    `eng/validation/nemotron-thinkoff-e2e.py` checks a running server end to
    end.
  - Known limitation: the window does not cover every reply. Behind
    TensorAgent's 12-tool system prompt the tool rounds reason longer, and in a
    live TensorAgent run (8B Q4_K_M, Metal) 4 of 39 stray closes came at
    2,212-3,064 characters. Once 2,048 characters pass without a close, the
    parser has decided the reply is the answer, on every path (the Web UI and
    TensorAgent included), so a later close is kept as the answer's own text.
    In that run the reasoning and the literal `</think>` reached the answer in
    2 of 9 thinking-off tool chats.
  - The Web UI and TensorAgent pages stream undecided text as it arrives. When
    the close comes, the host sends the reasoning as a `thinking` frame and
    then the remaining answer as a `replace` frame. Clients that cannot take
    text back (the OpenAI, Ollama and Responses APIs, and the CLI) receive the
    undecided text only once it is decided. For an ordinary thinking-off
    answer, the first character therefore arrives up to 2,048 characters
    later. Over 104 logged ordinary replies this was a median of 229
    characters (about 3 s), 90th percentile 1,401 (about 22 s) and at most
    2,047 (about 37 s at the logged decode rate). Streamed from
    `/v1/chat/completions` on Metal, a one-sentence answer (190 characters)
    arrived whole 2.6 s after its first token, a 1,402-character answer 17 s
    after it, and a 3,362-character answer sent its first 2,053 characters
    after 27 s and streamed the rest. Thinking-on
    replies and Nemotron 3 Nano / Omni (ChatML) are never held or taken back.
  - The window is counted in characters, so the wait grows as decoding slows:
    2,048 characters are about 500 tokens, which is 100 s at 5 tokens a second
    (the 47B, or the CPU backend). Before keep-alives, a held stream sent
    nothing at all in that time, and a proxy's idle timeout (nginx: 60 s)
    closed it. While text is held, `/v1/chat/completions` and `/v1/responses`
    now send an SSE comment (`: keep-alive`) every 15 s, and `/api/chat/ollama`
    sends a chunk with an empty message. The skills and sub-agent loop, which
    holds the text itself, hands the adapter an empty update for every piece it
    holds, so the adapter can send them. The CLI writes to a terminal and sends
    nothing extra. Nothing is sent while the prompt prefills, as before.
  - The decision reads the reply exactly as it was generated. The transcript and
    the non-streaming APIs parse a reply whole, while the pages and the SSE APIs
    parse it as it streams, and both now decide alike. Before, the Markdown rule
    read the text as shown: around an empty call list the call branch had
    trimmed line breaks, two fence lines ran together, and the same reply was
    reasoning in the stream but answer in the transcript, so the follow-up
    re-prefilled the conversation.
  - A `response_format` reply with thinking off is shaped by its grammar (or
    first-token list) from the first token, so it cannot be stray reasoning. It
    is not parsed as possible reasoning, and a `</think>` inside one of its JSON
    strings stays there. The pipeline announces no prompt tail for it, and the
    OpenAI adapter runs no parser over it, as before this template needed one.
    `/v1/responses` applies no grammar to its `text.format` (the format is asked
    for in the prompt), so its replies can still open with stray reasoning and
    are parsed like any other thinking-off reply.
  - TensorAgent shows an undecided reply as it arrives, so after a GPU fault it
    cannot tell which part of the shown answer is decided. A turn on such a
    model starts again from the top instead of being carried on. Carried on,
    the reasoning was handed to the model as its own answer so far and kept in
    front of the answer.
  - The tail of the prompt, not the request's `think` flag, decides where the
    reply starts. A caller's `{'reasoning': True}` in the system prompt opens
    the block even when the request has thinking off. The reply is then parsed
    as reasoning, and the transcript records the `<think>\n` the cache holds.
    Before, the whole reasoning was the answer, and the next turn re-rendered
    the history one block short of the cache.
  - **Nemotron 3 Nano / Omni** (and every other `nemotron_h*` template) use
    ChatML (`<|im_start|>` / `<|im_end|>`). Multimodal placeholders include
    `<image>` (later expanded into `<img>` + N + `</img>`) and
    `<so_embedding>` (audio). `RenderNemotron` declares tools in the system
    turn as a `<tools>` block of `<function>` entries and asks for calls as
    `<tool_call><function=NAME><parameter=key>value</parameter></function></tool_call>`;
    past calls are rendered back in that form, and consecutive tool results
    go back in one user turn, each wrapped in `<tool_response>`. These models
    write one `<tool_call>` per parallel call, so their turns do not end at
    the first call.

## 13. Optimization opportunities

- **Recorded direct-CUDA MoE decode**: replay the MoE decode step as a CUDA graph, as `ggml_cuda` effectively does,
  to close the 70 vs 143 tok/s gap on Lightning.
- **Host-expert tiers for one-projection experts**: the GGML accelerator tier and the CUDA doorbell take gated experts
  only. A ReLU^2 tier would keep each offloaded layer's hottest experts on the device.
- **Batched expert streaming on direct CUDA**: offloaded prompt chunks stream one expert at a time.
- **`--tp` with offload, and the tensor-parallel x layer-split hybrid**: both need per-rank graphs this family does not
  emit yet.
- **DSpark block cost**: the drafter's BF16 attention weights are dequantized to F32 at load (565 MB read per block
  across its six layers); keeping them BF16 on the graph path would halve that. DSpark's blocks could also share one
  program across requests, as DFlash's lanes do.
- **Audio tower**: the F32-compute companion runs as graph programs on every executor (§4.7). The public GGUFs still
  carry no tower, so audio is refused unless that companion is loaded. Open work: BF16-compute parity on the trained
  weights (the BF16-emulating companion keeps the managed path).
