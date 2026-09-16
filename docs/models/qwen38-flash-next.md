# Qwen 3.8 Flash Next (`qwen4exp`)

[← back to model index](README.md)

Qwen3.8-Flash-Next is a hybrid MoE: GatedDeltaNet recurrent layers interleaved
with full-attention layers (some behind Qwen Sparse Attention's indexer), a
PLE n-gram embedding block, ×4 hyper-connection streams and a 512-expert MoE.
The GGUF architecture id is `qwen4exp`. Weights:
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
(multi-shard per quant directory; point `--model` at the `-00001-of-` shard;
`mmproj-BF16.gguf` beside the model enables image input).

## How TensorSharp runs it

On the GGML backends the whole token runs as (almost) one graph — embedding,
PLE (in-graph), all 48 layers, the final mixer and the LM head — with a
shape-keyed cache of captured graphs (`TS_Q4E_TOKEN_GRAPH=0` falls back to
per-layer fused kernels, which in turn fall back op-by-op). Vision rides the
Qwen3.5-VL tower with (T,H,W) IMRoPE positions; multi-image and multi-turn
image sessions are supported, with KV reuse across turns (the GDN recurrence
cannot rewind, so a cached prefix is reused only when the new prompt extends
it exactly).

## Video input

A video reaches the model as an OpenAI Chat Completions `video_url` content
part carrying a base64 MP4, WebM or MOV data URI (remote URLs are not
fetched):

```json
{"type":"video_url","video_url":{"url":"data:video/mp4;base64,...","fps":1,"max_frames":8}}
```

`fps` (0 < fps ≤ 60) and `max_frames` (1–64) are optional; the defaults are
`VIDEO_SAMPLE_FPS` and a positive `VIDEO_MAX_FRAMES`, otherwise 1 fps and 16
frames, and a longer clip is sampled evenly across its length. The server
decodes the clip into ordered frames, each with its source time (frame index
over the probed frame rate, approximate for variable-rate clips), and the
Qwen-VL video layout takes over from there:

- **Temporal pairs.** The tower's patch embedding has two temporal slices
  (`v.patch_embd.weight` and `.weight.1`), so consecutive frames are merged
  two at a time exactly as the Qwen-VL processor stacks them; an odd clip
  repeats its last frame to complete the final pair. Each pair is encoded on
  its own (the reference tower attends within one temporal patch only) and
  yields the same merged-patch token count as one still frame. The clip is
  resized as a whole against the Qwen3-VL video pixel budget, so every pair of
  a clip shares one grid.
- **Prompt layout.** The template renders a video part as
  `<|vision_start|><|video_pad|><|vision_end|>`; the `<|video_pad|>` becomes
  one `<t seconds><|vision_start|><|video_pad|>…<|vision_end|>` block per pair,
  `t` being the pair's mean source time with one decimal, and the template's
  own start/end tokens stay wrapped around the clip. Two `video_url` parts in
  one message render as two clips. Still images in the same message keep their
  `<|image_pad|>` spans, in attachment order.
- **Positions.** Each pair is positioned like a still image whose (T, H, W)
  coordinates start at the running position of that pair — the Qwen3-VL
  `get_rope_index` rule, which splits a video grid into per-pair entries — so
  consecutive pairs carry strictly increasing temporal M-RoPE ids, the label
  text between them advances the stream, and the text after the clip resumes
  past the last pair's grid. The QSA indexer's position history, MTP
  draft catch-up and the post-clip rotary/cache gap all record those same
  coordinates, so speculation and retained prefixes agree with the target.

What this is not: frames are sampled, not decoded by a temporal encoder, and
the frame times are the sampled source times rather than a re-timed 2 fps
stream. Frames uploaded through the Web UI carry no source time and stay
still images, one span each, as before. The full-checkpoint check is
`benchmarks/engine_comparison/validate_deepseek41_media.py` with the
`video_order` / `video_timestamp` scenarios against a served Qwen3.8 with its
`mmproj-BF16.gguf`.

## Continuous batching

Concurrent requests are served through **per-sequence state holders**: each
in-flight request owns its attention KV + QSA indexer caches, its GDN conv +
delta-net state, its PLE conv history and n-gram window, and its pinned kernel
descriptors. The native kernel keys its device-resident recurrent state by the
holder's host seed pointers and its cached graphs by the descriptor addresses,
so switching requests is a reference swap — no state download/upload, no graph
rebuild — and each sequence decodes through its own captured single-graph
fused decode. The engine round-robins sequences per step
(`SupportsPerSequenceFusedForward`); a fused N-way batched decode is a future
optimization.

## Multi-GPU

`--tp N` on `qwen4exp` runs a **layer split**: each GPU holds a contiguous run
of whole layers. It is not tensor parallelism — `qwen4exp` shards no weights —
and it is the same (and only) multi-GPU mode llama.cpp offers this architecture
(`-sm row` refuses to load it). It is a capacity feature, not a speed feature:
it is how you fit the model when one card cannot hold it.

Measured on 2× A100-80GB, Qwen3.8-Flash-Next-UD-Q2_K_XL (73.4 GiB):

- greedy output is **byte-identical** between the 1-GPU and the 2-GPU run
  (same SHA-256).
- VRAM 24.2 GB + 26.2 GB — roughly half the model on each card instead of all
  of it on one.
- throughput unchanged: prefill ~1520–1550 t/s and decode ~56 t/s either way.
  For reference, llama.cpp on the same box: 1 GPU pp1536 1094 / tg128 61.2;
  2 GPUs `-sm layer` 1200 / 61.5 — so llama.cpp also gains ~10% prefill and
  ~0 decode from the second card.

Startup prints which mode ran and the per-GPU layer/byte split.
`TS_Q4E_LAYER_SPLIT=20,28` overrides the automatic balance with explicit layer
counts per GPU (llama.cpp's `--tensor-split` in spirit) and throws rather than
silently ignoring a value it cannot honour — useful because the automatic
balance prices weights and cannot see the vision tower, which loads later and
lands on GPU 0.

## Benchmark matrix

[`benchmark_config_glm53_qwen38.json`](../../benchmarks/engine_comparison/benchmark_config_glm53_qwen38.json)
registers this model as `qwen38-flash-next` at a pinned Hugging Face revision,
with its `mmproj-BF16.gguf` attached so the `image` scenario can run. Two
registry facts are worth repeating here.

The published Q8_0 shards carry **no** `nextn`/`mtp` tensors at all, so
`mtp_supported` is false and `--mtp on` cells are gated out with a reason
rather than quietly serving standard decode.

And **this model only runs on the column that passes `--tp N`**. The section
above is the reason: the split degree comes from `--tp`, so on a backend column
that passes none, TensorSharp builds a single-device context and all 175.3 GiB
land on one card. The config therefore gives it a `min_tp` (4, the weights-only
floor — 8 is the degree the 8×A40 box is meant to use) and the harness records
its cells on the no-`--tp` column as skips reading
`needs --tp 4 (does not fit 1 GPU(s))` instead of letting them OOM. Run it as:

```
python run_matrix.py --config benchmark_config_glm53_qwen38.json \
    --models qwen38-flash-next --backends ggml_cuda_tp
```

That column tells llama.cpp `--split-mode layer` over the same GPUs, so the
reference column is the same placement on both engines — which for `qwen4exp`
is the only one either engine has.
