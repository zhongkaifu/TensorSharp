# Inkling 2 (`inkling2`, `inkling2_loop`) on the shared decoder runtime

Status (2026-11-21): milestone 1 (graph path, one device, text); section 3 has the validation.

Thinking Machines' Inkling 2 Mini (`thinkingmachines/Inkling-2-Mini`, Apache-2.0) is a looped mixture of experts:
24 stored layers, of which layers 6-17 run three passes, so a token sees 48 layer applications; 26.4B parameters, 3.4B
active. The reference graphs are llama.cpp's `src/models/inkling2.cpp` and `src/models/inkling2-loop.cpp` (#32217,
#32694; #32704 re-shares the loop slots after the scale pass for files with per-expert scales). ggml-org's GGUFs
(`ggml-org/Inkling-2-Mini-GGUF`, Q8_0/IQ4_NL/BF16) use that layout. vLLM serves the official NVFP4 checkpoint from
`vllm/vllm-openai:inkling-2-mini` (#69028), which is the reference for real-checkpoint outputs
(`eng/inkling2/capture_vllm_reference.py`).

## 1. The model, as llama.cpp computes it

Hyperparameters (GGUF prefix `inkling2_loop.` for Mini): hidden 2304, 24 heads of 128, 8 KV heads, no RoPE,
`sliding_window` 1024 with `sliding_window_pattern` (layers 5, 11, 17, 23 global), `shortconv.l_cache` 4,
`attn_res.block_size` 6, `attention.relative_length` 16 / `relative_extent` 1024 / `relative_clip` 8,
`attention.qk_clip` 22.627, `attention.log_scale_floor` 8192 / `log_scale_alpha` 0.1, 128 experts top 8,
`expert_weights_scale` 8, `activation.situ_beta` 4 / `situ_linear_beta` 25, `final_logit_softcapping` 30,
`vocab_size` 201024 padded / `vocab_size_unpadded` 200058, `loop.start` 6 / `end` 18 / `count` 3 / `gate_rank` 64.

Per token:

- **Input:** `rms_norm(token_embd[t]) * token_embd_norm` (no norm on media embeddings).
- **tau** (global layers): `1 + alpha * log(max(1, (pos + 1) / floor))`.
- **Layer `il` at block position `ip`** (`ip = il` outside the loop; inside it, the physical layer):
  - Attention inputs: with `attn_res_query` (every layer but 0), `mix(blocks, cur, attn_res_query)` (one query on
    sliding layers, three on global: Q, K, V); otherwise `cur`. In a later loop pass each input then goes through
    `control(input, message_j)` (j = q, k, v).
  - Norms: sliding: `attn_norm` (once, or per input when messages made them differ); global: `attn_q_inp_norm`,
    `attn_k_inp_norm`, `attn_v_inp_norm`.
  - `q = clip(Wq·xq, attn_q_clip, qk_clip)` per head; `k = clip(conv(Wk·xk, attn_k_conv, dyn_k), -, qk_clip)`;
    `v = rms_norm(conv(Wv·xv, attn_v_conv, dyn_v))` per head (no weight); `r = clip(Wrel·xq, attn_rel_clip,
    relative_clip)` [16 per head]. Global: `q *= tau`, `r *= tau`. Global layers' dynamic taps
    `dyn_k = attn_k_conv_proj·xk`, `dyn_v = attn_v_conv_proj·xv` ([4 taps x 8 heads] per token).
  - `clip(x, scale, c)`: `x *= scale` (per channel, if given), then each row scaled by `min(1, c / (|x|_2 + 1e-6))`.
  - Logits: `s_ij = q_i·k_j / sqrt(128) + r_i·B[:, d_ij] - t_i` with `B = attn_rel_bias` [16, extent] padded by one
    zero column, `d_ij = pos_i - pos_j` if `0 <= d_ij < extent` else `extent` (the zero column), and the sink
    `t_i = attn_sink_proj·xq` per head; one extra logit 0 (the zero sink) joins the softmax. Sliding layers attend
    `(p - 1024, p]` and their extent is the window; global layers attend everything.
  - `attn = softmax · v`, gated: `attn *= sigmoid(attn_gate·xq)`; then `o = conv(attn_output·attn, attn_post_conv)`.
  - Residual: if `ip % 6 == 0` the layer starts a block: `blocks.push(inpSA)` and `cur = o`; else `cur = inpSA + o`.
  - MoE input: `mix(blocks, cur, ffn_res_query)`, then `control(., message_ffn)` in a later pass, `ffn_norm`.
  - Router (F32): `p = sigmoid(ffn_gate_inp·x)`; top 8 by `p + exp_probs_b`; weights `p_sel / (sum + 1e-20) * 8`.
  - Experts: `[g; u] = ffn_gate_up_exps·x` (gate is the first 1152 rows), `situ = 4 tanh(g/4) sigmoid(g) * 25 tanh(u/25)`,
    `ffn_down_exps·situ`, weighted sum; then `conv(., ffn_post_conv)`; `cur = cur + moe`.
- **conv(x, w, dyn)**: causal depthwise filter of 4 taps over this sequence's last 3 inputs and the current one
  (ggml_ssm_conv: `w[3]` takes the current token); global layers add, per head h and lag L in 0..3,
  `dyn[L, h] * x[t - L]`; result `(x + conv) / sqrt(taps * (dyn ? 2 : 1) + 1)`. The layer's conv state holds the last 3
  inputs of K (1024), V (1024), the attention output (2304) and the MoE output (2304): 6656 channels.
- **mix(blocks, cur, query)**: `S = rms_norm([blocks..., cur])` (no weight), `scores = S·query / sqrt(2304)`,
  softmax over the block axis, `out = sum softmax * S` (one output per query).
- **control(x, m)**: `g = W_down·[rms(x); rms(m)]` (rank 64), `x - tanh(W_forget·g) * x + tanh(W_gain·g) * m`.
- **Loop**: layers 0-5 once (the prelude); then three passes over 6-17, each starting from the prelude's output and
  blocks; pass p > 0 reads messages `mix(bank, bank_out, loop_msg_query)` (4 queries: q, k, v, ffn) where the bank is
  the previous pass's blocks after its first core block (the stream at layer 12) and `bank_out` its output; then
  18-23 once. Each pass keeps its own KV cache and conv state: llama.cpp numbers pass p > 0's layer `ip` as slot
  `24 + (p - 1) * 12 + (ip - 6)`.
- **Output**: `mix(blocks, cur, output_res_query)`, `output_norm`, `output`, `30 tanh(logits / 30)`, the first 200058
  columns.

## 2. On the runtime

| Piece | Where | Status |
|---|---|---|
| family, config, tokenizer (`tokenizer.ggml.pre = inkling2`) | `Models/Inkling2/` | M1 |
| loop as virtual layers 0..47 with a physical-layer map for weights | runtime: `LayerName` seam | M1 |
| attention residuals and loop bank in a widened stream | `StreamWidth`, `EmitStreamInput`, `EmitLayerPrologue`, `EmitBlockInput/Output`, `EmitStreamOutput` | M1 |
| q/k/r prep, tau, clips, sinks, V norm | attention seam (family emits its attention) | M1 |
| relative bias by key distance (linear caches and rings) | attention seam: distance index input per (cell, row) | M1 |
| conv state on attention layers (K, V, attn out, MoE out) | runtime: recurrent state beside the KV cache | M1 |
| MoE: sigmoid + bias routing, scale, fused gate_up, SITU | family-emitted MoE (graph ops) | M1 |
| softcap and padded vocabulary | head seam | M1 |
| chat protocol: renderer, reply parser, effort levels | `ChatTemplate.Inkling2.cs`, `OutputParser.Inkling2.cs`, registry | M1 |
| continuous batching: batched slots, concurrent requests through the engine | runtime slots | M1 |
| per-op path; tensor parallelism, layer split, expert offload (declared gaps in the serving contract) | runtime contract | later |
| speculative verify windows over the conv state (n-gram and scripted drafts) | the shared short-conv history and commit | M1 |
| prefix reuse across turns (a past turn re-renders as generated) | renderer: raw-token splice, tool rounds from fields | M1 |
| checkpoints: clone, file, device copies (conv windows with the attention rows) | runtime checkpoints | M1 |
| the DSpark drafter | runtime | later |
| vision tower | multimodal | later |
| fused kernels (attention with the bias on the fly, depth mixes, controllers, convolutions), NVFP4 experts | TensorSharp-owned kernels | later |

Milestone 1 is the graph path (`RequiresGraphs`) on one device: managed CPU and ggml_cpu against a double-precision
reference on synthetic F32 checkpoints, then ggml_cuda on ggml-org's Q8_0 against the vLLM reference. ggml-cuda has
no flash-attention kernel for a per-head mask (llama.cpp's #32217 added one after ggml's last sync, and ggml stays
unpatched), so attention is explicit: scores, bias, sink, softmax, values. A global layer's prompt chunk is split by
query rows so its scores stay bounded.

## 3. Validation (milestone 1)

Synthetic checkpoints (`Inkling2RuntimeTests`, `Inkling2ReferenceTests`) against a double-precision reference
(`Inkling2Reference.cs`): managed CPU and ggml_cpu agree to relL2 1e-6 on the plain stack and 1e-5..3e-5 through the
loop, for the prompt chunk and for token-at-a-time decode. ggml_cuda is looser on the synthetic loop (up to 1.6e-1),
whose random weights make the three passes amplify rounding. Greedy speculative decoding with scripted drafts (some
accepted, some rejected) emits exactly plain greedy's tokens on both synthetic models, the sliding rings wrapping.
The same suite is the family's dedicated conformance suite (`SharedRuntimeConformanceTests.DedicatedSuites`): batched
slots match each sequence alone (managed 1e-5 plain, 5e-5 looped), three requests served together through the engine
give each one's greedy stream, and a shared-prefix checkpoint continues exactly when cloned, written to a file and read
back, or copied on the device. Its fixture keeps conv windows over 4 KB a layer: the GGML graph builder holds a state
leaf under 4 KB in per-run compute memory, not in the device copy a slot copy on the device uses, so a smaller
synthetic state cloned from an exported checkpoint decoded from stale device copies on ggml_cuda (the real model's
windows are 80 KB a layer).

The real checkpoint (`Inkling2CheckpointTests`, ggml-org's Q8_0 against vLLM's NVFP4 outputs with an FP8 KV cache,
12 prompts, 209 teacher-forced positions):

| | ggml_cuda | ggml_cpu |
|---|---|---|
| top-1 agreement with vLLM | 198/209 | 195/209 |
| \|log p - vLLM\| over vLLM's top-5, median / p95 | 0.129 / 1.71 | 0.127 / 1.54 |
| 32 greedy tokens identical to vLLM's | 7/12 | 7/12 |
| last prompt logits, chunk against token-at-a-time (relL2) | 1.2e-2..8.2e-2 | 0 (4 prompts) .. 3.4e-2 |

- **CPU against CUDA:** the two backends pick the same top-1 at the last prompt position in all 12 cases, and their
  logits differ by relL2 7e-3..6.6e-2.
- **Chunk against decode:** on ggml_cpu, four prompts give bit-identical logits on the two paths, so the paths
  compute the same function. The gaps elsewhere come from kernel choice (a GEMM against a GEMV for experts that
  receive several rows) and grow through the top-8-of-128 routing over 48 layer runs, where a near-tied expert can flip.
- **The misses against vLLM:** 8 of 11 are near-ties for vLLM itself (top-1 over top-2 margin under 0.6 nats). The
  three confident ones (1.5, 1.7 and 3.5 nats) all follow a role or content marker inside the chat template
  (`<|message_user|>`, `<|content_text|>`), positions that chat training does not score. On the 8 raw-text prompts
  the top-1 agrees at 112 of 116 positions.
- **Tokenizer and template:** they reproduce vLLM's prompt ids for all 12 prompts, including the chat prompts with
  `reasoning_effort` (`Inkling2TokenizerTests`).
- **Chat renderer:** it matches the published template rendered by jinja2 (`eng/inkling2/render_template_reference.py`)
  on tool declarations, parallel tool calls, tool results, past reasoning and effort levels
  (`Inkling2ChatProtocolTests`).

Speed with the Q8_0 GGUF on ggml_cuda (`TensorSharp.Cli --benchmark`, 512-token prompt, 128 fixed tokens): 515 tok/s
prefill and 26.6 tok/s decode. Decode reads about 5 GB of weights a token (48 layer runs of 8 experts plus attention,
and the head); much of the rest of a step is graph overhead: the explicit attention and the many small mixing, clipping
and convolution ops a layer.

The server (`TensorSharp.Server.Host --backend ggml_cuda`, OpenAI chat API, `eng/inkling2/server_e2e.py`):

- The reasoning and the answer come back apart, whole and streamed, with no message framing left in either.
- A tool call comes back as an OpenAI tool call (`get_weather {"city":"Paris"}`), and its result is answered.
- A follow-up turn resumes from the end of the previous one: 3,461 of 3,473 prompt tokens reused, 178 of 189 after a
  tool round.

Every layer carries conv state, so the cache resumes only from stored end states, never from a mid-turn divergence.
A past turn must therefore re-render exactly as generated. The renderer handles both kinds of past turn:

- **A spliced turn:** a plain turn is spliced from its raw tokens (`KVCachePromptRenderer`), which carry the model's
  own message framing, so the renderer emits the placeholder bare and then `<|content_model_end_sampling|>`, the end
  of sampling that the live cache holds after them.
- **A tool round:** it re-renders from its fields (reasoning, then the calls), which reproduces the generated tokens.

Open after milestone 1:

- **Effort levels:** the server's `reasoning_effort` parser knows low/medium/high only, so OpenAI's `none` and
  `minimal` render as `low` (0.2) and `xhigh` as `high` (0.9). The template's `none` (0) and `max` (0.99) are
  reachable only by name through the renderer.
- **Speed:** fused kernels for the attention (the relative bias computed on the fly instead of gathered from a
  per-step distance index, which grows with the context), the depth mixes, the loop controllers and the convolutions;
  a decode step is ~12K ggml nodes, most of them small. The 4-bit experts of ggml-org's IQ4_NL GGUF already run.
- **Runtime contract:** the per-op path, tensor parallelism, layer split, expert offload, and the DSpark drafter.
- **Vision.**
