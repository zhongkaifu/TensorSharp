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
it exactly; see [Retained-prefix reuse](#retained-prefix-reuse)).

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
  repeats its last frame to complete the final pair. The processor samples at
  2 fps, so the frames it merges are 0.5 s apart, and TensorSharp pairs two
  sampled frames only when they are that close (at most
  `QwenVideoFrames.MaxPairedFrameGapSeconds`, 0.575 s). A sparser frame — the
  default 1 fps, or a long clip spread over `max_frames` — is a different scene,
  so it fills its own temporal patch (repeated, like a still image) and keeps
  its own time label. Pairing such frames blended them: a three-frame 1 fps
  clip of the cards 17, 42, 86 read back as `["12", "47", "86"]` and reads
  `["17", "42", "86"]` with a patch per frame. The cost is one patch per
  sampled frame instead of per two. Each pair is encoded on
  its own (the reference tower attends within one temporal patch only) and
  yields the same merged-patch token count as one still frame. The clip is
  resized as a whole against one video pixel budget
  (`Qwen35ImageProcessor.VideoMinPixels` / `VideoMaxPixels`), so every pair of
  a clip shares one grid; a frame count that cannot fit that budget even at the
  minimum grid is rejected.
- **Prompt layout.** The template renders a video part as
  `<|vision_start|><|video_pad|><|vision_end|>`, and that whole outer span is
  replaced — as the Qwen3-VL processor (transformers v4.57.1
  `processing_qwen3_vl.py`) does, with no second pair of delimiters around the
  clip — by one `<t seconds><|vision_start|><|video_pad|>…<|vision_end|>` block
  per pair, `t` being the pair's mean source time with one decimal. Because a
  pair's label loses its two frames' individual times (and a `max_frames` cap
  can select non-adjacent frames), the blocks are preceded by one text line,
  `Sampled video frame times in chronological order: 0, 1, 2 seconds.`, listing
  every sampled source time; the per-pair vision-token layout is unchanged.
  Two `video_url` parts in one message render as two clips. Still images in the
  same message keep their `<|image_pad|>` spans, in attachment order.
- **Encoding cache.** A pair's embedding is cached under both frame paths and
  the clip size, and invalidated when either frame file's size or timestamp
  changes (not only the later one).
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

**Concurrent greedy output can differ from solo, and the reason is the prefill
shape.** The scheduler prefills a lone request in one large chunk (the smaller
of `TS_SCHED_SOLO_PREFILL_CHUNK` and `TS_SCHED_MAX_BATCHED_TOKENS`) and
concurrent requests in shares of the step budget, and this model's logits
depend on the chunk size. Measured with `benchmarks/ChunkParityProbe` on
UD-Q2_K_XL over a three-GPU layer split, a 19,121-token prompt: the same 4096
chunking reproduces itself bit for bit (max |Δlogit| 0), while 1024- and
512-token chunks move the logits by up to 1.3 and flip greedy decoding at
near-ties — first at output token 9, where the top-2 margin is 0.002 (the
heading's first word). Holding the shape equal removes the effect: four
concurrent requests x three waves of a 2,928-token prompt that every request
prefills in one chunk (`TS_SCHED_PREFILL_CHUNK=4096`,
`TS_SCHED_MAX_BATCHED_TOKENS=16384`) came back byte-identical to solo, 12/12
over 512 tokens, so no state leaks between the per-sequence holders and the
round-robin decode does not depend on concurrency. The strict chunked-versus-whole
logit gate is the open item recorded in
[Retained-prefix reuse](#retained-prefix-reuse).

## Retained-prefix reuse

`Qwen4ExpModel.RetainedCache.cs` gives `qwen4exp` the retained-holder reuse the
Qwen 3.5 and DeepSeek V4 paths have:

- A finished conversation's whole per-sequence holder is **retained** and
  re-keyed for the turn that extends it exactly. Nothing moves: the native
  state entries keyed on the holder, its captured graphs and the draft head's
  private K/V stay where they are.
- The state at the end of the prompt every chat shares is **checkpointed** as a
  host-authoritative deep copy (attention K/V, QSA raw keys and positions,
  GDN/PLE recurrent state, private MTP state) and **cloned** into each new chat.
  A clone refuses to copy missing authoritative native state rather than stale
  host seeds.
- Reuse is **exact-prefix only** (`IExactFusedCacheReuse`): a holder whose
  tokens the new prompt does not reproduce to the last one is not a
  continuation, and every partial match re-prefills.
- Both retained conversations and checkpoints count against one budget,
  `TS_Q4E_RETAINED_CACHE_MB` (default 4096, clamped by measured memory
  headroom; `0` or an unparsable value declines every retention), evicting the
  oldest retained conversation first. `TS_Q4E_RETAINED_CACHE=0` disables the
  feature.
- It needs the complete GGML token-span path (every piece of per-sequence state
  device-resident and keyed by the holder) and a GDN state layout the native
  entry can be copied through exactly. Retention works under a layer split;
  checkpoints are admitted under a layer split and refused under tensor
  parallelism.

Evidence (synthetic fixtures, not trained-model acceptance or performance):
[`eng/validation/qwen38_mtp_followup/retained-cache-20260916`](../../eng/validation/qwen38_mtp_followup/retained-cache-20260916/README.md)
— `Qwen4ExpRetainedCacheTests` / `Qwen4ExpRetainedCachePolicyTests` cover
retained A/B/A, checkpoint clones, speculative rebound, budget eviction,
missing-state refusal and QSA first/reset growth on CPU, and a physical
two-GPU layer-split checkpoint lifecycle on CUDA. Three strict bit-exactness
gates pass on CPU and remain failed on single-GPU CUDA, all with the same
greedy argmax on this untrained fixture: chunked 16+4 versus whole 20-token
prefill differs in full logits (`SharedPrefixChunking_…`); a four-token target
verify differs from four scalar forwards (`TeacherForcedTargetVerify_…`, max
abs 0.0070); and 32 teacher-forced tokens committed in blocks of 2-4 differ
from scalar decode (`RepeatedTargetBlocks_…`, max abs 0.0082). The CUDA target
graph's reductions depend on the batch width; the isolated precision
prototypes in the evidence README close some of these but are not integrated.

## Speculative decoding with the shared MTP head

`--draft-model mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` attaches the per-token
MTP block; it speculates for a solo request that prefilled from position 0
(steps shared with other sequences, and turns that continue a retained holder or
a shared-prefix clone, decode plainly — the head keeps its own K/V and cannot
draft across positions it never replayed). Measured on UD-Q2_K_XL over a
three-GPU layer split (A40), `--spec-draft 3`:

- **Parity.** A 192-token code-copy stream is identical to plain greedy at
  1.75-1.96x (141/141 drafts accepted, no rollbacks); the speculative prefill
  (`SpecForward`) is bit-identical to the plain one for the same chunking. The
  4-row verify kernel does flip near-ties against the 1-row decode kernel, which
  shows up as a fenced ```` ```json ```` answer where plain greedy wrote a bare
  object, or pretty-printed versus compact JSON.
- **Prose does not pay.** Over 18 prose requests acceptance was 68-70% (3.0
  tokens per verify), but a verify costs ~45 ms, a partial acceptance adds a
  ~43-46 ms rollback (restore the recurrent state, re-forward the kept rows), and
  a governor-parked step still runs the one-row speculative forward with hidden
  capture (~24-26 ms against a 19 ms plain decode): decode at c1 was 44-46 tok/s
  against 52 plain for 512-token prose, and 34-38 against 40 after an 8k
  prompt.

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
