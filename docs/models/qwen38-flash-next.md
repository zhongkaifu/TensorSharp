# Qwen 3.8 Flash Next (`qwen4exp`)

[← back to model index](README.md) | [中文](qwen38-flash-next_zh-cn.md)

> **Multi-node execution:** on `ggml_cuda`, add `--tp-node-id` and `--tp-peers`
> to `--layer-split N` or `--tp N`; N is the GPU count on each node. Use the
> same checkpoint, build and numerical settings everywhere. See the
> [multi-node guide](../multi-node-inference.md) for launch order and limits.
> Historical local measurements below do not qualify distributed performance.

Qwen3.8-Flash-Next is a hybrid MoE: GatedDeltaNet recurrent layers interleaved
with full-attention layers (some behind Qwen Sparse Attention's indexer), a
PLE n-gram embedding block, ×4 hyper-connection streams and a 512-expert MoE.
The GGUF architecture id is `qwen4exp`. Weights:
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
(multi-shard per quant directory; point `--model` at the `-00001-of-` shard;
image input needs its matching vision projector, such as `mmproj-BF16.gguf` or
`mmproj-F16.gguf`: the CLI loads it from beside the model when
`--image` is given, and the server needs `--mmproj`).

## Recommended presets

| Hardware | Preset | Files | Measured (2026-11-08) |
| --- | --- | --- | --- |
| Apple Silicon, 48 GB | `config/qwen3.8-flash-next.json` | ISTA-DASLab GSQ-RCO `Q2_0` (two shards, 66.4 GB) + DFlash2 Q4_K_M drafter, `ggml_metal` | M5 Pro: decode 43-48 tok/s, 4K / 16K prompt TTFT 5.8-6.0 / 22.7-24.3 s, four requests at once 43.4 tok/s, Strata's sanity checks 12/12 |
| NVIDIA, 32 GB VRAM + ~45 GB RAM | `config/qwen3.8-flash-next-cuda.json` | the same `Q2_0` + the shared MTP head, direct `cuda` engine | RTX 5090: decode 131-250 tok/s, 4K / 16K / 100K prompt TTFT 1.0 / 4.2 / 30.4 s |
| NVIDIA, several users | add `config/qwen3.8-flash-next-cuda-4lanes.json` | four sequence lanes | RTX 5090: four requests at once 243-258 tok/s, parallel_4 281-286 tok/s |
| Apple Silicon, UD-Q2_K_XL files | `config/qwen3.8-flash-next-ud-q2_k_xl.json` | UD-Q2_K_XL (three shards) + shared MTP head | M5 Pro: ~31 tok/s decode (the previous preset) |

The `Q2_0` files are [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(pinned at `ed59f920`) and the drafter is [incoai/Qwen3.8-Flash-Next-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2-GGUF)
(pinned at `4f5a39be`); the presets download and verify them (sha256).

## How TensorSharp runs it

Two implementations serve the family:

- `--backend cuda` loads the direct-CUDA engine (`Qwen4ExpCudaModel`, the Q4e kernels; see
  [Serving on the direct-CUDA engine](#serving-on-the-direct-cuda-engine)).
- Every other backend (`ggml_metal`, `ggml_cuda`, `ggml_vulkan`, `ggml_cpu`, `cpu`) runs it on the shared decoder
  runtime (`Qwen4ExpRuntimeModel`). It is the same runtime that serves Qwen 3.5. The family adds only its own parts:
  - the hyper-connection streams;
  - the sigmoid-gated recurrence;
  - the PLE layer, whose n-gram rows are gathered from the on-disk table;
  - QSA: dense up to the indexer's budget (2,051 cells on the shipped checkpoint), selecting whole blocks past it.
- `--backend mlx` is refused.

A step runs as one graph, with the Metal prompt kernels as device operations between graph segments:

- the GDN recurrence;
- the fused hyper-connection read;
- the hyper-connection scatter;
- the selected QSA attention over wide windows.

Every request has its own slot, so continuous batching, the prefix cache and tensor parallelism come from the shared
runtime. They are not family code. The same applies to:

- `--tp N`: per-rank graphs on `ggml_cuda` and `ggml_vulkan`.
- `--layer-split N`: whole-layer stages, and multi-node pipelines.
- Host experts with accelerator tiers: the routing profile is collected on discrete GPUs too, so the next load keeps
  each layer's most-routed experts in VRAM.
- Speculative decoding: the shared MTP head through `--draft-model mtp-*.gguf`, or a DFlash2 drafter.

`--tp` and `--layer-split` are separate placement modes for this family. Their hybrid is refused.

Vision rides the Qwen3.5-VL tower with (T,H,W) IMRoPE positions. Multi-image and multi-turn image sessions are
supported.

Radix reuse can continue past an identical image or video span. A slot's copies carry:

- the M-RoPE delta;
- the PLE conv and n-gram histories;
- the QSA indexer cache;
- the rotary positions of the recent cells. QSA turns a block's pooled key at its first cell's position, which is not
  its KV index after media.

A long conversation with images therefore selects past the budget too. The GDN recurrence cannot rewind, so a cached
prefix is reused only when the new prompt extends it exactly.

Thinking can be switched on or off. With it off, the assistant turn opens with the closed, empty `<think>\n\n</think>`
block the published template emits. Replayed history keeps that exact suffix, so cached prefixes still match.

The published Qwen4Exp vision patch merger uses GELU(erf), separately from the transformer blocks'
`gelu_pytorch_tanh` ([Transformers v5.16.1 reference](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1705)).
TensorSharp keeps the tanh merger, which is the shipped projector's validated default:

- The delivered default passes 12/12 image cases.
- An erf trial passed 10/12.
- A same-build OCR control read blue `9364` as `9334` with erf.

The erf opt-in (`TS_Q4E_VISION_MERGER_ERF`) went with the removed implementation. See the
[image validation guide](../../eng/validation/qwen38-parallel-vision.md).

**Against `Qwen4ExpModel`**, measured against `main` (`032fdf24`, which still served it), 3.6K-token prompt:

| Route | Prefill | Decode |
|---|---|---|
| `ggml_metal`, M5 Pro, Q2_0 file | 3.95-4.15 s vs 3.95-4.23 s | 30.4 vs 30.3 tok/s |
| `ggml_cuda`, one A40, 30 GB tier | 6.09 s vs 6.54 s | 41.3 vs 41.4 tok/s |
| `ggml_cuda`, 2x A40, `--layer-split 2` | 2.64-2.69 s vs 2.80 s | 51.6 vs 51.1 tok/s |

Further notes:

- On `ggml_metal` the outputs are identical, and the runtime peaks at 35-36 GB wired against 39 GB.
- Slot capacities are whole multiples of 256 cells. ggml-cuda groups a KV head's queries only over keys whose length
  is a multiple of 256, and an exact-size slot (prompt plus `max_tokens`) had kept the layer-split decode at 48.4
  tok/s: a QSA layer's attention took 133 us against 26.
- With a 55-token prompt and 128 tokens on the two A40s, the runtime's first token comes at 345-398 ms against
  517 ms and the whole answer about 60 ms sooner, but its decode rate is 54.4 against 57.1 tok/s (open).
- `Qwen4ExpModel`'s goldens on the synthetic fixtures (`InferenceWeb.Tests/Fixtures/Qwen4ExpLegacy`) hold the runtime
  to its numerics.

> **Historical sections.** The dated sections below that describe the GGML backends record `Qwen4ExpModel`, which was
> removed on 2026-11-13. That covers its token span, arenas, retained holders, one-token verify-row kernels and expert
> cache. Its GGML-only switches went with it: `TS_Q4E_VERIFY_ROW_KERNELS`, `TS_Q4E_VERIFY_ROW_COMMIT`,
> `TS_Q4E_DISABLE_ARENA_DECODE`, `TS_Q4E_QSA_POOL_CACHE`, `TS_Q4E_QSA_SPARSE_FA`, `TS_Q4E_LAYER_SPLIT`,
> `TS_Q4E_PLE_PREFETCH`, `TS_Q4E_METAL_FUSE`, `TS_Q4E_VISION_MERGER_ERF` and the debug dumps. Setting one now prints a
> removal notice. The direct-CUDA engine's `TS_Q4E_*` switches are unchanged.

## Vision encoder memory and validation

The shared Qwen-VL whole-encoder graph now writes CUDA attention tiles into one
output allocation, preserving the default attention arithmetic. On 2026-10-03,
the supplied photo produced 7,920 patches / 1,980 tokens with `mmproj-BF16.gguf`;
all 5,068,800 projected float32 values were bit-identical to the previous
encoder. Attention scratch fell from 394.8 MB to 343.6 MB (13%). One warmup
and three standalone samples measured a median of 2,947.4 ms versus 2,982.0 ms;
the small, noisy difference does not establish an end-to-end latency gain.

Experimental `TS_QWEN_VISION_F32=1` selects TensorSharp-owned streaming F32
CUDA attention for 72-wide heads: the same encoder measured 2,635.7 ms, an
11.6% improvement, with 36.5 MB of attention scratch. Its full-embedding
comparison failed the conservative minimum-row cosine gate (0.999856 versus
0.9999 required; aggregate relative L2 0.002069), so it remains opt-in without
end-to-end model-quality qualification. Validation used a single RTX 3080
Laptop GPU (16 GB, WDDM, CUDA 12.6) and unchanged upstream ggml
`353b63b439f27ab2cc19dac97ab1681ba6d2d084`: 168 native numerical cases and ten
broader CPU regression targets passed, with no failures or skips. The numeric
cases comprise 78 existing attention cases and six vision cases on each of CPU
and CUDA; other projectors, videos and devices were not benchmarked. See the
[shared encoder checks](qwen35.md#fused-vision-encoder-blocks) for reusable
tools. Generated evidence stays in ignored `artifacts/qwen-ttft/`.

## Tool calling and agent workflows

`qwen4exp` returns structured tool calls through the Qwen ChatML output parser.
Both JSON and `<function=...><parameter=...>` bodies inside `<tool_call>` are
accepted, including streamed fragments and thinking-enabled responses. Generic
client tools return OpenAI `tool_calls` with call IDs and
`finish_reason: "tool_calls"`; built-in skill and code tools run through the
server's agent loop when configured.

XML parameters use the declared tool schema: string values such as `123`,
`true`, and JSON source stay strings. The parser removes one framing newline
on each side, retaining code indentation and additional blank lines. Literal
`</tool_call>` inside a parameter or JSON string does not terminate the call.
Incomplete parameters/functions do not become executable calls; a complete
body at EOS retains the existing recovery for an omitted outer closing tag.

Enable skills with `--skills-dir`, skill scripts with `--skills-allow-exec`,
and workspace file/shell tools with `--code-exec`. Editing uses `apply_patch`.
See [Agent Skills](../agent_skills.md) for execution and sandbox configuration.

Because the family renders tool declarations and has this parser, it is also
eligible, on the server, for [sub-agent delegation](../multi_agent.md), which is
on by default on the chat paths and does not depend on skills or `--code-exec`.
`--no-multi-agent` (or `multi_agent: false` in a request) turns it off; the CLI
has no sub-agents. No delegation results are published for this family.

Reusable checks:

```bash
python3 eng/validation/validate-qwen38-tool-calls.py \
  --url http://127.0.0.1:5098 \
  --output docs/validation/qwen38-tool-calling/generic.json
python3 eng/validation/validate-release-agent-workflows.py \
  --url http://127.0.0.1:5098 --concurrency 1 \
  --output docs/validation/qwen38-tool-calling/workflows.json
```

The workflow check uses `eng/validation/fixtures/skills` as the server's skills
directory and verifies skill discovery/read/script execution, shell commands,
code generation, and read/patch/run workflows. Add `--thinking` to repeat with
reasoning enabled. `verify-agent-code-artifacts.py` independently executes the
final source artifacts against additional inputs. On a validation host started
with its sandbox disabled, pass `--sandbox-off` to both execution validators;
those reports establish functional execution only, not sandbox isolation.

Validation on 2026-09-19 used the supplied UD-IQ4_XS checkpoint on three NVIDIA
A40 GPUs (`ggml_cuda`, 15/16/17-layer split, MTP disabled), built against clean
upstream ggml `456172ec733a135778adcd32d00e576a58232e45`:

- 160 managed regression tests passed, with no failures or skips.
- All 12 ordinary generic-tool cases passed: weather, string-looking numbers /
  JSON source, and multiline Python with a trailing newline, each with streaming
  and thinking independently on/off. Each call also completed a tool-result
  round trip using its actual call ID.
- All required tools executed successfully across 12 skill/shell/code workflows.
  Four retained generated or edited programs passed independent execution with
  additional inputs. Strict final-answer checks passed 9/12 after separating
  pre-tool narration from the final turn; three responses added backticks or
  prose around the correct value and remain failures.
- All four literal-tool-marker stress prompts ended with EOS before completing
  the call. They remain failed end-to-end cases, although complete XML/JSON
  marker-containing calls pass parser regressions. The logs do not identify
  which terminal token was sampled; the tool-marker token IDs are not EOS IDs.

The VM denied user namespaces, so execution checks used an explicitly disabled
sandbox. This campaign tested one quantization and sequential requests; it did
not validate sandbox isolation, MTP, other devices, or performance. Full requests,
SSE events, artifacts, build provenance, and failures are retained in the ignored
`docs/validation/qwen38-tool-calling/` directory.

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

## Thinking budget

With thinking on, a reasoning block that reaches `TS_THINKING_BUDGET` (default
75% of `max_tokens` from 512 up) is closed and the answer follows inside
`max_tokens`, in the server and in the interactive CLI. `</think>` is one
trained token (248069), and ahead of it the host writes Qwen's published
hand-over sentence ("Considering the limited time by the user, I have to give
the solution based on the thinking directly now."). Before 2026-09-29 the family
had no closing token, so a turn whose reasoning reached the budget was stopped
with an EMPTY answer (`finish_reason` `thinking_budget`): four concurrent
three-turn conversations with `max_tokens` 2000 on 4x A40 (`--tp 4` and
`--layer-split 4`) passed 0/4, every failure an empty turn. With the hand-over,
the same `--tp 4` run passed 4/4: the sentence closed four turns, every turn
answered, and every conversation reused its previous turn (turn 3: 1564-3482 of
1591-3509 prompt tokens).

## Continuous batching

Concurrent requests are served through **per-sequence state holders**: each
in-flight request owns its attention KV + QSA indexer caches, its GDN conv +
delta-net state, its PLE conv history and n-gram window, its M-RoPE gap and
coordinate history, and its pinned kernel descriptors. Sampling and logits
belong to each request; the executor detaches borrowed solo logits before a
second request can overwrite the model's output buffer. Immutable PLE
convolution coefficients remain owned by the model throughout every holder's
lifetime.

On Metal a new request's holder starts at 2048 tokens of K/V and grows with
its request (a request reserves its prompt and reply before its first chunk),
while the primary cache keeps the whole window an explicit `MAX_CONTEXT` asks
for. Holders used to be the whole window too: at `MAX_CONTEXT=32768` each live
conversation took ~1.6 GB of the Mac's memory for a short prompt, and four of
them ran every step 20-50% slower under the memory pressure (48 GB M5 Pro,
parallel_4 29 -> 36-37 tok/s with small holders). A holder that grew answers
bit for bit as the whole-window primary does. `TS_KV_INITIAL_TOKENS` sizes the
primary and every holder alike (set it to `MAX_CONTEXT` for whole-window
holders).

**Concurrent decode uses the fused arena by default.** On a single-device GGML
CPU, Metal or CUDA backend with F16 KV caches and initialized span state, two
or more ready decoders run together in one reusable ggml graph. When routed
experts are offloaded to the host, execution crosses host-expert seams between
backend graph segments. This is a shared graph on Metal; CUDA graph capture is
a separate backend mechanism. Attention/QSA and GDN keep separate slot state.
Projections, routed/shared experts and the language head participate in the
shared graph; operations whose backend arithmetic depends on row geometry keep
the solo reductions instead of requiring a multi-row kernel for every node.
Each attention lane uses its solo padded KV window and
mask, rather than the longest request's window. Router softmax also keeps the
solo row reduction, because changing Metal's thread count with the total batch
size changes routing weights. A mixed scheduler step batches
its ready decode subset and runs new requests' prefill chunks through their own
holders. Arrivals, departures, cache growth and a return to solo execution
flush or retire the corresponding arena slot before another path reads its
state. Image requests join this path after media prefill; their compacted
rotary positions and QSA coordinate history stay with their holder.

Layer splitting, other KV dtypes and unavailable span or backend geometry
continue through the per-sequence fused path when supported. Under `--tp N` on
one node the arena runs on every rank: each rank builds the same graph with its
own FFN shard and its own replica of the requests' K/V, GDN, PLE and QSA state,
a request sits in the same slot on every rank, and the ranks meet at each
layer's two gathers as a solo step's ranks do. The batch shares one set of
gathers a step, where round-robin solo steps pay one set per request. Token rows
are gathered on the host, as solo tensor-parallel steps gather them, and rank 0
runs the head. A rank's memory is planned to its weights, so the arena is built
there only while its caches leave the usual headroom free (max(1 GiB, 1/32 of
the device)); otherwise the requests decode one at a time. A geometry no request
sits in is retired before another is sized, and a prompt span that cannot
allocate takes the arena back before its request fails. Tensor parallelism
across nodes keeps the per-sequence path.
GPU arena execution requires upstream flash attention support for its head
geometry; CPU uses the corresponding attention fallback. Every participating
holder must have authoritative initialized state, a matching position and room
for the next cache row. A holder needing growth takes the solo path before it
can rejoin the batch.
One ready decoder uses solo fused decode. `TS_BATCHED_FUSED_DECODE=0` provides
the round-robin control for comparisons. A native execution failure fails the
affected requests; partially advanced recurrent state is not retried as a solo
step. This implementation lives in TensorSharp-owned code and uses unchanged
upstream ggml. Source support alone does not establish throughput gains or
trained-model quality on every backend.

Reusable checks: [parallel text and request isolation](../../eng/validation/qwen4exp-concurrent-http.md)
and [parallel image content, attachment order and history](../../eng/validation/qwen38-parallel-vision.md).
The image guide includes pinned projector sources and hashes. Keep generated
reports in ignored `docs/validation/` or `artifacts/`; failed, skipped or
unavailable model/device cases are not passing validation.

### Local Metal validation, 2026-10-04

Both requested checkpoints were exercised on an Apple M5 Pro with 48 GiB unified
memory, using unchanged ggml revision
`353b63b439f27ab2cc19dac97ab1681ba6d2d084`. The
[native probe](../../eng/validation/qwen4exp-batched-decode-probe.md) covered
widths 2, 3 and 4, 64 teacher-forced decode steps and two repetitions. All 4,728
retained prefill/decode/continuation comparisons measured max |Δlogit| = 0 and
zero greedy differences. Raw logit bitwise identity was not separately checked.
The aggregate native decode measurements were:

| Checkpoint | Fused decode tokens/s | Gain versus round-robin | Delivered image cases |
| --- | ---: | ---: | ---: |
| Base UD-Q2_K_XL | 23.86–25.61 | 17.37–50.77% | 12/12 passed |
| Uncensored IQ2_XXS | 18.70–26.82 | 15.04–35.36% | 4/12 passed; suite failed |

Matched two-request HTTP topic answers and exact markers agree between enabled
and round-robin configurations, with runtime evidence of fused execution. Across
two repetitions, topic throughput improved 7.01% for base and 8.18% for
uncensored; uncensored markers improved 0.71%. Base marker results were mixed:
the cold-first, prefix-cache-off pair was 3.08% slower, while the delivered
serial-before-parallel, prefix-cache-on pair was 11.705% faster in one repetition.
All matched requests in the latter pair reported zero cached prompt tokens;
workload order does not establish a cache-hit or warming cause. These measurements
support no uniform speedup or no-regression guarantee.

The delivered tanh-default image checks reproduce all earlier default answers:
16 completed base turns and 14 completed uncensored turns. Exact C=1/C=2 parity
holds for all eight base and seven uncensored completed turn pairs. The
uncensored checkpoint reads blue `9364` as `9324` in single-image OCR and also
fails both attachment-order checks. Its two blue-image follow-up requests were
unreached after incorrect initial OCR; they are untested and not passing. Exact
parallel parity therefore does not establish image-quality acceptance. These
two-card OCR/color/order/history checks and the text relevance/marker checks
cover a limited quality scope.

Evidence is retained in ignored `docs/validation/qwen38-parallel/` and
`artifacts/validation/qwen38-locality-{base,uncensored}-native/`. Native, paired
HTTP and delivered vision runs have distinct managed build identities; the
final native library is unchanged across those phases. Native rates exclude
HTTP/prefill, while HTTP rates include admission and prefill. The model files
exceed physical RAM, so paging, process/cache state and short measurements limit
performance conclusions. This validation does not qualify other devices,
CUDA/TP execution, broad factual/visual quality or long-context performance.
The image harness compares assembled content and request hashes; its raw SSE
framing and completion IDs remain unverified.

**Prefill shape remains a separate numerical caveat.** The scheduler prefills
a lone request in one large chunk (the smaller
of `TS_SCHED_SOLO_PREFILL_CHUNK` and `TS_SCHED_MAX_BATCHED_TOKENS`) and
concurrent requests in shares of the step budget, and this model's logits
can depend on the chunk size. Historical CUDA checks with `benchmarks/ChunkParityProbe` on
UD-Q2_K_XL over a three-GPU layer split, a 19,121-token prompt: the same 4096
chunking reproduces itself bit for bit (max |Δlogit| 0), while 1024- and
512-token chunks move the logits by up to 1.3 and flip greedy decoding at
near-ties — first at output token 9, where the top-2 margin is 0.002 (the
heading's first word). Holding the shape equal removes the effect: four
concurrent requests x three waves of a 2,928-token prompt that every request
prefills in one chunk (`TS_SCHED_PREFILL_CHUNK=4096`,
`TS_SCHED_MAX_BATCHED_TOKENS=16384`) came back byte-identical to solo, 12/12
over 512 tokens. Those results qualified the earlier round-robin workload and
settings; they do not qualify the new arena or every concurrent workload.
Prefill shape is not promised
to be width-invariant on CUDA, so the fixture's chunked-versus-whole gate bounds
that difference instead of requiring bit equality; see
[Retained-prefix reuse](#retained-prefix-reuse).

## Retained-prefix reuse

`Qwen4ExpModel.RetainedCache.cs` gives `qwen4exp` the retained-holder reuse the
Qwen 3.5 and DeepSeek V4 paths have:

- An eligible finished primary cache stays **live** in the radix tree
  (`DeferPrimaryConversion`). An exact next turn claims that same cache without
  allocating a replacement. Its tree marker adds no retained-state bytes; the
  memory already belongs to the model's primary execution cache. Another request
  displaces it by converting it to a holder only when retention is available.
  A refused or failed conversion makes that request prefill normally.
- A finished conversation's whole per-sequence holder is **retained** and
  re-keyed for the turn that extends it exactly. Nothing moves: the native
  state entries keyed on the holder, its cached graphs and the draft head's
  private K/V stay where they are.
- The state at the end of the prompt every chat shares is **checkpointed** as a
  host-authoritative deep copy (attention K/V, QSA raw keys and positions,
  GDN/PLE recurrent state, private MTP state) and **cloned** into each new chat.
  A clone refuses to copy missing authoritative native state rather than stale
  host seeds.
- Reuse is **exact-prefix only** (`IExactFusedCacheReuse`): a holder whose
  tokens the new prompt does not reproduce to the last one is not a
  continuation, and every partial match re-prefills.
- Exact reuse may cross identical media. Holders and checkpoint clones preserve
  `MropeCacheGap`: after a staged chunk it is `KV length - 1 - last T`, and a
  following scalar token rotates at `KV index - gap`. The gap is the negative of
  Qwen 3.5's rotary delta, including signed video offsets. Changed media content,
  span boundaries or conversation scope cannot claim that conversation's state.
- Extra retained holders and checkpoints count against
  `TS_Q4E_RETAINED_CACHE_MB`, clamped by measured memory headroom; `0` or an
  unparsable value declines those payloads. This does not disable exact reuse of
  an already live primary. Unset, model admission uses half the current measured
  headroom, with 4096 MB only where no headroom can be measured. The radix tree
  also resolves its default device/state caps from half the spare memory at
  engine creation and checks current spare memory minus running-request reserves.
  Its live-primary marker is not charged again as an extra holder. The tree owns
  eviction; model admission refuses a holder that does not fit and the tree may
  release an older scoped payload before retrying.
- Before displacement allocates a replacement, the tree can measure the live
  primary's eventual holder footprint and decline conversion if it exceeds an
  absolute option or family cap. Qwen4Exp also checks its retention budget before
  allocating. Admission still rechecks after allocation because headroom can change.
- Primary adoption allocates its empty replacement before publishing the moved
  holder. An allocation failure leaves the live KV and recurrent state intact
  and publishes no partial holder.
- It needs the complete GGML token-span path (every piece of per-sequence state
  device-resident and keyed by the holder) and a GDN state layout the native
  entry can be copied through exactly. Retention works under a layer split;
  checkpoints are admitted under a layer split and refused under tensor
  parallelism.

Evidence (synthetic fixtures, not trained-model acceptance or performance):
`Qwen4ExpRetainedCacheTests` / `Qwen4ExpRetainedCachePolicyTests` cover
retained A/B/A, checkpoint clones, speculative rebound, budget refusal followed
by owner release, missing-state refusal and QSA first/reset growth. Engine tests
compare exact image follow-ups with cold greedy output and require zero reuse
for changed media or another scope. Allocation-failure tests verify unchanged
native continuation and no published holder. Measurement tests require the
live-primary estimate to equal the adopted holder's
footprint and reject checked-out holders or invalid lengths without tensor allocation.
`Qwen35MRopeReferencePositionTests`
also checks Qwen4Exp chunked positions, signed gaps and follow-up decode against
six independent SGLang fixtures. These fixtures exercise QSA/GDN/PLE/MTP state;
they do not evaluate a trained vision encoder. The checked-in small weights can
be materialized with [the fixture tool](../../eng/qwen4exp-mtp-fixture.py) using
`--sample-csharp InferenceWeb.Tests/Qwen4ExpMtpSample.cs --qsa`.
`DeferredPrimaryCacheTests` covers live-primary continuation with no replacement
allocation, actual displacement, a zero extra-retention budget and conversion
failure followed by cold-output parity. The physical two-GPU layer-split
checkpoint test remains gated and was skipped in the current single-GPU run.
On single-GPU CUDA a four-token target verify equals four one-token forwards
(`TeacherForcedTargetVerify_…`) and 32 teacher-forced tokens committed in blocks
of 2-4 equal scalar decode at every row (`RepeatedTargetBlocks_…`), bit for bit —
see [Verify rows run the one-token kernels](#verify-rows-run-the-one-token-kernels).
A 16-token prefill continued by 4 tokens is not bit-identical to one 20-token
prefill on CUDA, because prefill kernels are chosen by batch width:
`SharedPrefixChunking_…` bounds that difference at 1e-2 on CUDA (measured
1.7e-4 to 4.4e-4 in logits; the stale-seed defect it was written for moved them
by 0.3155) and allows a greedy change only at a near-tie within twice the
measured difference (measured on an A40, 2026-09-17).

## Single-GPU HTTP validation (2026-10-03)

The supplied UD-IQ1_M shards and BF16 projector were exercised through the Web
UI API with the original `20241021_022843061_iOS.jpg`, `请详细描述这幅图`, then
`请继续`. The machine was an i7-11800H with 32 GB RAM and one RTX 3080 Laptop
GPU with 16 GB VRAM. Both builds used GGML CUDA, default context and expert
placement, greedy sampling, repetition penalty 1, no skills or agent delegation,
and no speculative decoding. The owned streaming vision kernel remained off.
The original image produced 7,920 patches and 1,980 vision tokens.

| 128-token workflow | Original TTFT | Updated TTFT | Updated cache reuse |
|---|---:|---:|---:|
| Initial image question | 98.218 s | 45.981 s | 0/1,997 tokens |
| First `请继续` | 87.697 s | 2.381 s | 2,125/2,140 tokens |
| Second `请继续` | Not comparable | 1.243 s | 2,268/2,283 tokens |

The initial 128-token answer was identical. The first continuation had identical
input history but different output text; the second continuation therefore had
different histories between builds and has no qualified comparison. These are
single-pass observations, not an aggregate answer-parity or quality score.
The first two updated replies hit the 128-token cap; the third stopped after
124 tokens. GPU clocks, WDDM paging and the OS file cache were uncontrolled,
and model loading/startup warmup are excluded. The original image benchmark
followed two text requests, while the final image benchmark followed startup.
Do not attribute the entire first-request difference to vision attention alone.
The unchanged placement still runs routed experts for 40 of 48 layers on the
CPU, so a fresh 1,997-token multimodal prefill remains expensive.

A separate 1,024-token-limit run completed the initial image description at EOS
after 496 tokens; its beginning matched the original 128-token answer. Its
follow-up reused 2,494/2,508 tokens with 4.289 s TTFT, but hit the 1,024-token cap.
The saved descriptions contain unverified interpretive claims; there is no
uncapped original-build comparison or formal visual-accuracy grade.

Final CUDA cache coverage passed 13 distinct synthetic cases: 12 use supported
GDN/attention geometry with 64-wide attention heads; the small-buffer chunking
regression uses supported GDN geometry with 8-wide attention heads so its KV
buffers remain below 4,096 bytes. The physical two-GPU case was skipped. CPU
cache/ownership suites passed 410 cases with nine unavailable checks excluded;
26 focused adapter/failure tests passed after the final exception cleanup.
Upstream ggml remained unchanged at
`353b63b439f27ab2cc19dac97ab1681ba6d2d084`. Reproduction and comparison rules
are in [the HTTP benchmark guide](../../eng/validation/README-qwen-chat-cache-benchmark.md).
Generated reports, answers, provenance and coverage remain in ignored
`docs/validation/qwen-ttft/`; vision numerical evidence is under
`artifacts/qwen-ttft/`.

## Speculative decoding with the shared MTP head

Image requests can also use the learned head: the scheduler queues each image
embedding slice before speculative prefill and preserves its MRoPE positions.
Prepared image spans remain available for retries after prefill and no longer
block speculative decode. This still requires a solo request prefilling from
position 0; retained-prefix and concurrent-request restrictions below remain.

A separate 2026-09-27 UD-IQ1_S check on two RTX PRO 4000 Blackwell GPUs used
`--layer-split 2`, context 1024 and a resident shared Q8_0 MTP head plus BF16
projector. After one warmup, all three measured text/image passes matched plain
greedy token-for-token with active MTP and ngram drafting. Text was bounded at
64 tokens; the image answer completed at EOS after 204 visible tokens and
correctly described the number and color. Median paired worker decode-time
speedups were 1.215x for text and 1.292x for image with MTP. Image request timers
exclude synchronous image preparation and encoding; these short copy-heavy
checks are not broad quality or end-to-end media-latency benchmarks. Separate
plain/MTP HTTP checks passed 24/24 text requests and 3/3 image scenarios per mode,
including attachment order and image history. This configuration refused retained
cache admission for lack of headroom, so those follow-ups re-prefilled. Local
evidence: `docs/validation/model-matrix-20260927/qwen38/SUMMARY.md` (not committed).
Local tensor parallelism and distributed text execution are now available as
described below. These historical media/speculation checks do not qualify the
distributed modes, which require ordinary text prefill and decode.

`--draft-model mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` attaches the per-token
MTP block (GGML backends and `--backend cuda`; the head must be a single GGUF file, attached
when the model loads); it speculates for a solo request that prefilled from position 0
(steps shared with other sequences, and turns that continue a retained holder or
a shared-prefix clone, decode plainly — the head keeps its own K/V and cannot
draft across positions it never replayed). Measured on UD-Q2_K_XL over a
three-GPU layer split (A40), `--spec-draft 3`:

- **Parity.** A 192-token code-copy stream is identical to plain greedy at
  1.69x plain decode with the current verify-row kernels (1.75-1.96x before the
  2026-09-17 change below; 141/141 drafts accepted, no rollbacks); the speculative prefill
  (`SpecForward`) is bit-identical to the plain one for the same chunking. Before
  2026-09-17 a 4-row verify did not round like a 1-row decode step and could turn
  a plain-greedy bare JSON object into a fenced ```` ```json ```` answer; verify
  rows now run the one-token kernels (below).
- **Prose does not pay.** Over 18 prose requests acceptance was 68-70% (3.0
  tokens per verify), but a verify costs ~45 ms, a partial acceptance adds a
  ~43-46 ms rollback (restore the recurrent state, re-forward the kept rows), and
  a governor-parked step still runs the one-row speculative forward with hidden
  capture (~24-26 ms against a 19 ms plain decode): decode at c1 was 44-46 tok/s
  against 52 plain for 512-token prose, and 34-38 against 40 after an 8k
  prompt.

On `ggml_metal` with the expert tiers above (48 GiB M5 Pro, UD-Q2_K_XL), the MTP head
pays off on code and lists: a 4-row verify costs ~115 ms against ~50 ms for a plain step
(~197 ms with the old placement, where the host computed a verify's ~3x larger set of
experts), acceptance 81% (3.4 tokens per verify). On `ggml_metal` the rows are verified as
one batched forward by default, as NInfer, llama.cpp and vLLM do: 95 ms a verify and 1.3x the
speed of verifying each row with the one-token kernels on code. Greedy text then differs from
plain decode and depends on the verify widths the governor chose; `TS_Q4E_VERIFY_ROW_KERNELS=1`
restores the exact rows (speculative text bit-identical to plain greedy, as on CPU and, by
default, CUDA, which keep them; there `TS_Q4E_VERIFY_ROW_KERNELS=0` selects the batched
verify). A partial acceptance no longer restores the
recurrent state and re-forwards the kept rows (~60 ms each): the verify span keeps every
row's GDN conv/SSM state (upstream `ggml_gated_delta_net`'s per-row snapshots) and PLE conv
history, and the accepted row is copied into the live state (NInfer's ReplaySSM idea; by
`memcpy` where the state is host-visible, by one device copy graph on `ggml_cuda` and
`ggml_vulkan`, and by the old rollback on a layer split whose state spans devices;
`TS_Q4E_VERIFY_ROW_COMMIT=0` restores the old rollback everywhere). With the exact verify rows the
speculative text is still identical to plain greedy.

A batched verify also overlaps its window reads with the GPU, as NInfer's streaming window
does: up to two experts a row that miss the window are read in the background (a second I/O
team) while the GPU computes the routes the slots already hold, and run after a second cut
that only waits for those reads (skipped, with no extra submit, when a layer deferred none).
The host share of a 4-row verify fell from ~17-19 ms to 3-5 ms and the GPU's rose by ~9 ms
(the extra cut and the second group's products), so a verify went from ~97-99 to ~94 ms:
MTP decode 31.2 / 26.5 / 32.5 -> 32.5 / 27.2 / 33.6 tok/s (code / prose / list, greedy text
unchanged). One-token decode and the exact verify rows keep a single group.
`TS_MOE_TIER_STATS=1` counts the background reads.

### Verify rows run the one-token kernels

A verify and the replay of its accepted prefix push 2-8 tokens through one span
graph (as does any prefill that short), and on CUDA ggml picks several kernels by
batch width. At those widths a row does not round the way its one-token decode step does:
F32 projections leave `mul_mat_vec_f` after 3 columns for a tensor-core / cuBLAS
TF32 path (router logits move by 2.6e-3), the BF16 QSA indexer projections take a
half-precision path from 2 columns (5.9e-3), routed experts switch to the
multi-token MoE kernel (4.8e-7) and flash attention to a multi-query launch
(2.9e-5). Through 48 layers of MoE and QSA routing that is not last-bit noise: on
UD-Q2_K_XL over three A40s, teacher-forcing the first 48 greedy tokens after a
3,248-token prompt, every 2-, 3- and 4-row verify row differed from its decode step,
by up to 2.5 in logits, and 4-6 of the 48 rows changed the greedy token — not only
at near-ties.

So on CPU and CUDA a span graph of 2-8 tokens builds each row from the kernels its
one-token graph runs: float projections put the tokens on the broadcast axis (one
`mul_mat_vec_f` launch on CUDA). The measured Q4_K, Q5_K, Q6_K and Q8_0 projections
on NVIDIA A40 run in blocks of at most 4 rows. Other devices and quantized types
use one-column reductions on the broadcast axis: Turing and GB10 select different
MMVQ reductions at width 1, so A40's four-row grouping cannot be applied globally.
Routed experts and attention are expanded one
row at a time, each attention row over exactly the KV window and mask row its
decode step reads. Graphs of up to 8 tokens, decode included, also keep the inputs
of two ggml-cuda fusions whose use depends on memory reuse (MoE weighted reduction;
RMS norm + RoPE) allocated, so those fusions happen at every width. One-token
kernels are unchanged. CUDA prefills longer than 8 tokens materialize weighted
expert outputs before their sequential sum, preventing allocation-dependent
FMA fusion from changing rounding between layer and tensor layouts. CPU and
Metal retain their existing prefill graph construction.

CPU and CUDA cap speculative drafting at seven tokens because verification also
includes the pending anchor, giving at most eight rows. This hard limit also
applies to explicit `--spec-draft` values and custom drafters; the default
preferred window remains three drafts.

The completion tests cover every committed row at widths 1 through 8. CPU also
needs this construction on macOS ARM: without it, widths 2 and 4 differed from
scalar logits despite identical stored GDN, PLE and KV state. The strict fixture
suite passes with the CPU path enabled; its runtime is not a performance benchmark.
The timing measurements below describe the original A40 run, not a qualification
of other GPU architectures or the broadcast fallback. A test-hook build can set
`TS_Q4E_TEST_MMVQ_CHANNELS=1` at startup to exercise that fallback on A40.

Measured on the same setup, three interleaved repetitions: every verify row at
widths 2, 3 and 4 is now bit-identical to its decode step (0 of 48 rows differ, no
greedy flips). A 4-row verify costs 32.4-33.1 ms instead of 29.3-29.6 ms (+11%),
3 rows 28.3-29.9 against 26.8-27.2 (+8%), 2 rows 24.5-24.8 against 24.1-24.3 (+2%);
decode steps (20.6-20.7 ms against 20.6-21.1) and the 3,248-token prefill
(2,335-2,338 ms against 2,324-2,359) are unchanged. End to end on the 192-token
code-copy stream (six passes per kernel set, all streams identical to plain greedy),
MTP speculation ran at 83.2 tok/s instead of 86.5 (1.69x plain instead of 1.84x) and
n-gram speculation at 73.8 instead of 79.5, while plain decode (49.1 against 47.0) and
prefill (830 against 804 tok/s) did not regress: speculation pays for its exactness.

## Larger than memory

The UD-Q2_K_XL file is 78.9 GB in three shards: a 28.8 GB n-gram (PLE) table, 46.1 GB of
routed experts and about 4 GB of everything else. A token reads 16 rows of the table and
10 of each layer's 512 experts, so TensorSharp runs the file on machines that cannot hold
it and reads those parts from the SSD as tokens need them.

- **The n-gram table is never uploaded or copied.** On the GGML backends it stays in the
  GGUF memory mapping with random-access advice (`madvise(MADV_RANDOM)`), and a token's 16
  rows (90 bytes each) are gathered on demand, in parallel. This is the idea behind
  llama.cpp's `--lazy-mode`, on by default there for tensors over 4 GiB. Every such load
  logs it:
  `PLE n-gram table: 28.8 GB read on demand from the GGUF mapping, 16 rows a token (random-access advice).`
  The direct `cuda` engine reads the rows from the mapping too. Both it and the GGML
  backends warm the table into the page cache in the background after loading, but only
  where the host can keep it cached: Linux, where the room is `MemAvailable` capped by the
  process's memory cgroup (a container's limit, not the host's). The larger-than-memory
  Mac skips it, since reading the table there would only evict experts. On a network
  filesystem the warm matters most: an uncached row on a geesefs mount took 92-120 ms.
  On the GGML backends the same background warm also reads every routed expert the host
  serves (the ones outside the accelerator tier, most-routed first), within what the
  cgroup can keep cached: on a discrete GPU each route outside the tier is computed on the
  host from the mapping, and a cold expert there is several storage round trips inside a
  decode step. `TS_HOST_WARM_WAIT=1` makes the load wait for the warm (benchmarks).
- **Every layer's routed experts are scheduled at a host seam, and the GPU computes them
  from per-layer expert slots.** On `ggml_metal` with neither `--n-cpu-moe` nor `--cpu-moe`
  set, the engine plans the placement itself (the same scheme as NInfer's Flash Next expert
  cache, on unified memory):
  - *Fixed slots* hold each layer's most-routed experts for the whole load. They are chosen
    from a **routing profile**: every host seam counts the expert ids it routes, and the
    counts are merged into a per-model file (`~/.cache/tensorsharp/moe-routing/`,
    `TENSORSHARP_MOE_PROFILE_DIR` / `TENSORSHARP_MOE_PROFILE`) when the model is disposed.
    The budget is spent across layers where it covers the most routes per byte; by default
    it is the 14.4 GB the old plan wired as whole layers (`TS_MOE_HOT_TIER_GB`).
  - *Window slots* (7/8 of the explicit-I/O budget, `TS_HOST_MOE_IO_CACHE_GB`) are filled on
    demand: a routed expert without a slot is read with `pread` straight into the least
    recently used slot (Metal shared memory, the file opened with `F_NOCACHE`). On the M5 Pro
    SSD that runs at 13-14 GB/s with 8-16 reads in flight, against 1-3.5 GB/s for page faults.
  - The host only remaps ids and reads misses; the GPU computes all ten routes in the same
    graph with the all-GPU span's own kernels. A route left without a slot (a step wider
    than the window) runs on the host decode kernel as before.

  ```
  [moe-offload] qwen4exp (tiered): all 48 layers' routed experts run at a host seam; 48 layers keep their most-routed experts on the accelerator (7659 experts, 14.4 GB of a 14.4 GB budget) covering 67.6% of profiled routes, plus 7466 window slots (14.0 GB) filled on demand.
  ```

  With no profile covering the model yet, every slot of that load is a window slot (least
  recently used, no fixed set) while the seams record the profile: measured 18.4 / 19.4 /
  20.4 tok/s, within 3-5% of a profiled load; the fixed slots come with the next load.
  `TS_MOE_HOT_TIER=0` restores the old plan, which wires whole trailing layers. That plan
  wastes RAM: a wired layer holds all 512 experts, used or not, and takes page cache from
  the layers that read theirs from the SSD.
- **Decode** of a step whose experts all have slots never computes on the host. Overflow
  routes run on TensorSharp's host decode kernel (ggml's CPU dot products on a thread team
  that is woken once per layer and parked when the layer is done; a spinning team slowed
  the GPU's work between layers 1.5-2x on Apple silicon). `TS_HOST_MOE_DECODE=0` restores
  the ggml graph path for those.
- **Prefill** up to 2,048 tokens a chunk uses the slots like decode: the fixed slots hold
  nearly every route a prompt makes and the window reads the misses. A wider chunk streams
  each layer's used experts to the GPU, read with `pread` straight into the CPU-visible
  staging memory (`TS_HOST_MOE_STREAM_PREAD=0` faults the mapping in and copies instead).
  Streaming first stages the whole layer, about 0.7 GB a layer for ISTA `Q2_0` and about
  1.3 s a chunk whatever its width. When streaming started at 65 tokens, every short prompt
  paid that: cold TTFT for 88-473 token prompts went from 1.35-1.4 s to 0.33-0.76 s with
  the split, and 900 / 1,770 tokens from 1.76-2.0 / 2.7 s to 1.33-1.55 / 2.55 s. At 3,500
  tokens the two paths are level. `TS_HOST_MOE_TIER_MAX_ROWS` sets the width. A tier
  without window slots (a discrete GPU's) splits up to one row below where streaming starts (127): when it stopped at
  64, a 65-127 row span was neither split nor streamed and ran every route on the host (see "Concurrent requests on
  ggml_cuda" below). Routes that neither the slots nor the
  window take at a prompt width run on the GPU too, their experts read into the one
  staging buffer: on the host decode kernel, which works in blocks of 8 rows and reads
  every overflow expert again per block, a 1,024-token span of Strata's records prompt
  took 3.2-3.7 s (that prompt routes far outside the profile). With prompts forced into
  1,024-token spans, Strata's 4K / 16K prompts took 14.4-15.0 / 52.5-54.2 s with host
  overflow, 6.8-7.7 / 26.9-27.4 s with GPU overflow, and 7.0-7.7 / 28.1-29.7 s streaming
  whole layers.
- **`--backend mlx` is refused up front**: MLX has no kernels for the sparse-attention
  indexer, the hyper-connections, the n-gram table or the IQ2_XS/IQ3_XXS experts. Use
  `ggml_metal`.

Measured on that Mac (ggml `353b63b`, unmodified) against llama.cpp `a868c3e3` on the same
machine. llama.cpp's default full offload fails there (`Insufficient Memory
(kIOGPUCommandBufferCallbackErrorOutOfMemory)`); its best configuration was CPU only with
12 threads, with its lazy mode reading the n-gram table on demand.

| Real text: a 1,818-token prompt, 256 greedy tokens | prefill tok/s | decode tok/s |
| --- | --- | --- |
| TensorSharp `ggml_metal`, the planned split | 109.4-115.5 | 12.8-12.9 |
| llama.cpp `-ngl 0 -t 12` | 20.4-22.4 | 13.00-13.24 |
| llama.cpp `-ngl 0 -t 12 --no-op-offload` | 28.8-31.6 | 11.94-12.48 |

| llama-bench's method: random tokens, pp512 / tg128, one session | prefill tok/s | decode tok/s |
| --- | --- | --- |
| TensorSharp `ggml_metal`, the planned split (15 layers' experts on the GPU) | 147.6 | 21.1 |
| TensorSharp `ggml_metal`, `--n-cpu-moe 32` (16 on the GPU) | 153.4 | 21.2 |
| TensorSharp `ggml_metal`, `--n-cpu-moe 30` (18) | 183.9 | 20.6 |
| TensorSharp `ggml_metal`, `--n-cpu-moe 28` (20) | 171.9 | 20.3 |
| TensorSharp `ggml_metal`, `--n-cpu-moe 26` (22) | 80.7 | 19.0 |
| llama.cpp CPU only, `-t 12 -nopo 1` | 50.07 | 22.55 |

Decode is flat from 15 to 16 layers on the GPU (21.1-21.2) and slower with each layer past
16, and at 22 the page cache left for the host layers is too small and prefill collapses. Runs from earlier
the same day, not side by side with the rest: the planned split 137.3 / 19.2 and llama.cpp
48.43 / 22.26 (an hour and a half apart), `--cpu-moe` (no expert on the GPU) 142.3 / 17.5,
`ggml_cpu` with 18 threads 46.1 / 16.6, and llama.cpp's `-ngl 28 -t 12` 36.64 / 20.06.

TensorSharp prefills real text 3.5-5x faster and random tokens about 3x faster. Decode is
on par on real text and 6% behind on random tokens (21.1 against 22.55). On Metal, decode
is bound by ggml-metal's cost per dispatch (about 4,200 dispatches a token, 29.4 ms when
the whole token is a single graph) plus about 0.19 ms for each host seam. Real text also pays page faults for experts that are not in the page
cache, so it depends on what else the Mac is doing: the same run decoded at
10.3-11.3 tok/s while other work kept 5 GB compressed. TensorAgent offers this file on
48 GB Macs ([measured in the app](../../TensorAgent/README.md#the-macs-own-models)).

### Loading from network storage

A checkpoint larger than half the process's memory used to skip the parallel page-cache
warm entirely, and everything the load reads then arrived through serial page faults.
Measured on an RTX 5090 VM with the GGUF on a geesefs (S3 FUSE) mount and a 93 GB memory
cgroup, `ggml_cuda`, profiled tier: the load took **505 s** (F32 tensors 43 s, dense
weights uploaded to the GPU 94 s, the 23.0 GB tier filled through page faults 363 s at
~63 MB/s), and the first answer then stalled on cold host experts and PLE rows. Now:

- When the whole checkpoint is over the budget, the tensors the load reads at once
  (everything but the routed experts) are still warmed in parallel: 3.7 GiB in 13 s.
- The tier's experts are read with 32 parallel large reads before the slots are filled:
  31.2 GB (the tier plus neighbouring host experts the reads coalesce over) in 52 s.
- The PLE table and the host-served experts warm in the background (56 GiB in ~125 s).
- The kernel warm-up prefills 64 tokens instead of 2,048 when experts are host-served: a
  2,048-token dummy prompt routes to nearly every expert and streamed most of the 46 GB
  (minutes on network storage, before the first request could start).

The model is ready in about 70 s (202 s with `TS_HOST_WARM_WAIT=1`). Every read goes
through a pool of 32 handles per shard, opened once before any tensor data is read: geesefs
drops a file's entire page cache at every `open()` (a 64 MB range went from resident to 0
when a second descriptor was opened, the mapping still live), so a reader that opened its
own handle threw away what the others had read. For the same reason, do not open the model
files from another process (a second server, `dd`, a GGUF inspector) while one is serving
from such a mount: that open drops the serving process's warm cache.

### Expert tiers, 2026-11-06

Same Mac and file, ggml `57e2c2ca` (unmodified), real text: four requests in one CLI
process (`--input-jsonl`, a warm-up, then a ~50-token code prompt, a ~1,300-token prose
prompt and a ~30-token list prompt, 256 greedy tokens each). Medians of three interleaved
repetitions in one session; the profile came from a separate 12-prompt calibration run.

| | decode tok/s, code / prose / list | TTFT s, code / prose / list |
| --- | --- | --- |
| Before (whole layers wired) | 13.80 / 13.87 / 15.59 | 3.14 / 13.58 / 2.78 |
| Expert tiers + explicit reads | 19.16 / 19.88 / 20.70 | 0.78 / 6.39 / 0.53 |
| Before, MTP head | 14.67 / 11.98 / 16.66 | 5.31 / 13.61 / 3.27 |
| Expert tiers, MTP head, exact verify rows (`TS_Q4E_VERIFY_ROW_KERNELS=1`) | 23.47 / 19.01 / 24.60 | 0.77 / 7.39 / 0.57 |
| Expert tiers, MTP head (batched verify, the default) | 30.87 / 21.31 / 28.31 | 0.75 / 7.03 / 0.56 |
| `config/qwen3.8-flash-next-ud-q2_k_xl.json` (the line above, with its sampling settings) | 27.00 / 20.62 / 32.72 | 0.80 / 7.96 / 0.53 |

The tiered plan computes the experts with the GPU's F32 activations where the old plan's 33
host layers quantized them to Q8_K, so greedy text can differ from before (here the prose
answer, from character 655); with every route in a slot a tiered layer is bit-identical to
the all-GPU span (`TieredExperts_AnswerAsTheAccelerator_Metal`). Teacher-forced perplexity
over the same text: see the 2026-11-06 report.

Single-device `ggml_cuda` also plans expert placement against current free VRAM,
pending float weights and caches, driver headroom and a 3 GiB graph reserve.
Explicit `--n-cpu-moe` / `--cpu-moe` settings override this plan. Tensor parallel
and layer-split runs still require their explicit placement configuration.

With a routing profile that covers every layer, that plan uses the same fixed slots as
`ggml_metal` instead of whole layers: every layer keeps its most-routed experts in VRAM
(the bytes the layer plan would have given whole layers) and the rest of each step's routes
run on the host decode kernel; discrete VRAM has no host-writable window. A load without
such a profile keeps the layer plan, and its resident layers' graphs also record their
routing, so the profile it saves covers every layer and a later load is tiered. On an RTX
5090 (32 GB, UD-Q2_K_XL, 12,245 experts / 23.0 GB in slots covering 85% of the profiled
routes, everything in RAM) plain decode measured 51.3 / 50.4 / 45.8 tok/s against
44.5 / 44.8 / 44.9 for the layer plan (code / prose / list), teacher-forced perplexity
16.12 against 16.22 over 1,024 tokens. At first the MTP head did not pay off there: CUDA
verified each row with the one-token kernels (~75 ms a 4-row verify) and re-forwarded a
partial acceptance, and its 2.8 GB cut the slots to 20.1 GB, so code and prose decoded at
42.2 and 34.2 tok/s with it. `TS_MOE_HOT_TIER=0` keeps the layer plan.

#### CUDA tiers, 2026-11-07

Same VM and file, ggml unmodified, four requests in one CLI process (code, ~1,300-token
prose, list, and a 329-token code-edit answer; 256 greedy tokens otherwise). Every run starts
from its own copy of one calibration profile (`TENSORSHARP_MOE_PROFILE`): each process merges
its routing into the profile when it exits, and a chain of runs sharing one profile tunes the
slots to the benchmark prompts (plain decode went 42.8 -> 57.5 tok/s on code over a chain).
Repeated arms agreed within 1.5 tok/s.

| decode tok/s | code | prose | list | edit | greedy text against plain |
| --- | --- | --- | --- | --- | --- |
| Plain | 52.2 | 52.4 | 47.6 | 43.1 | - |
| Plain, `TS_HOST_MOE_OVERLAP=0` | 49.6 | 49.6 | 45.5 | 41.3 | identical |
| MTP, old rollback (`TS_Q4E_VERIFY_ROW_COMMIT=0`) | 49.6 | 39.5 | 47.8 | 52.5 | differs on all four |
| MTP, exact rows + row commit (default) | 60.9 | 49.6 | 58.1 | 53.1 | identical on code and edit |
| MTP, batched verify (`TS_Q4E_VERIFY_ROW_KERNELS=0`) | 64.3 | 55.0 | 57.9 | 54.8 | not exact by design |
| Plain, adaptive slots (`TS_MOE_TIER_ADAPT=1`) | 57.1 | 61.4 | 58.8 | 53.3 | depends on earlier output |
| MTP + adaptive slots | 62.6 | 56.9 | 60.7 | 57.5 | depends on earlier output |

- **Seam overlap** (default): a tiered seam launches the graph part for the routes its slots
  hold asynchronously, then computes the other routes on the host meanwhile. Bit-identical.
- **Row commit on the device**: the accepted row's GDN, conv and PLE state is copied by one
  ggml copy graph instead of restoring the state and re-forwarding the kept rows. The old
  rollback was not exact on tiered CUDA loads either.
- **Adaptive slots** (opt-in): Strata's adaptive expert cache. Each layer keeps a decayed
  route count per expert, and every 32 seams up to two clearly hotter host-served experts are
  copied into its coldest slots. Over teacher-forced perplexity (2,048 tokens, eight 512-token
  windows of these docs) the slots held 889k routes instead of 685k, and perplexity was
  22.90 against 23.07 with the slots fixed. Which routes run on the host then depends on what
  was generated before, so repeated requests need not give identical greedy text; that is
  why it is off by default.


An optional CUDA selected-expert cache follows Strata's compact quantized-slot
approach. It keeps only routed experts in persistent device buffers, preserves
every selected expert and the router's reduction order, and evicts slots by LRU.
It uses unchanged upstream ggml kernels and stores the original GGUF bytes.
Set `TS_HOST_MOE_EXPERT_CACHE_MB` before starting the process; `0` (the default)
keeps the existing host path. For example, in PowerShell:

```powershell
$env:TS_HOST_MOE_EXPERT_CACHE_MB = '4096'
# Start the CLI/server with --backend ggml_cuda and your normal model options.
```

When automatic placement cannot fit all experts, every layer is eligible, and
each layer's cache quota fits its selected experts, all expert layers use the host seam
so each can use compact slots. Otherwise the plan keeps fitting trailing layers
resident. The default
`TS_HOST_MOE_EXPERT_CACHE_LAYERS=48` divides the budget among Qwen3.8's layers;
smaller synthetic checkpoints override it. Eligible bias-free, separately
quantized gate/up/down SiLU experts use the cache for one through eight rows.
Short prefill and target-verification blocks replay the scalar graph row by row,
preserving decode arithmetic and recurrent rollback. Other shapes, backends,
insufficient budgets or unsupported layouts retain the existing execution path.
Mapped-weight invalidation and model disposal release the device slots.
Eligible segments copy activations, routing weights and outputs directly between
CUDA buffers. Host-MoE debug and GPU verification retain the staged host contract.
`TS_HOST_MOE_EXPERT_CACHE_OUTPUT_BRIDGE=0` restores output staging for A/B checks;
`TS_HOST_MOE_EXPERT_CACHE_BRIDGE=0` restores input and output staging.
`TS_HOST_MOE_EXPERT_CACHE_PREFETCH=1` optionally faults selected cache-miss byte
ranges in parallel before pageable uploads. It leaves weights evictable, copies
their original bytes, and defaults off pending cold/warm workload measurements.

The budget covers graph allocations plus a conservative workspace allowance;
CUDA's shared pool and driver allocations still require additional free VRAM;
retired CUDA capture objects can persist until upstream's idle sweep, so the
owned reservation is not total process VRAM or its immediate reduction at unload.
`TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS=1` reports slots, hits, misses and reservation.
An entry also requires physical free VRAM above the native safety reserve;
an oversized requested budget can leave some layers using CPU fallback.
Use `eng/validation/qwen4exp-expert-cache.py` for complete-logit A/B checks and
`eng/validation/Qwen4ExpExpertCacheProbe/README.md` for commands and timing scope.
The feature remains opt-in: synthetic performance depends on cache capacity,
and those fixtures cannot establish trained language quality or performance
parity with Strata.

On 2026-10-02, the trained UD-IQ1_M checkpoint was tested on an i7-11800H,
32 GiB RAM and an RTX 3080 Laptop GPU with 16 GiB VRAM, using CUDA 12.6.
All three shards passed publisher SHA-256 checks at Hugging Face revision
`38bb39ee97821de2c9009abb7e93950eec396e66`; inference used the NVMe SSD copy.
TensorSharp used unchanged ggml `353b63b439f27ab2cc19dac97ab1681ba6d2d084`.
Stock Strata `36fa455e579b23a9c909c2c6fe1bddd9e51cb8ca` used its pinned
llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`, native GGUF experts,
the stock expert profile and an 8 GiB CPU resident budget. Its pack converts
some dense projections to BF16; cross-engine logits are not claimed bit-exact.

Four independent semantic checks cover integer arithmetic, extraction, a Python
function and the ordered squares of 1 through 20. All 40 fresh-process arms
completed at EOS with identical generated IDs across TensorSharp and Strata.
Context was 512 with F16 KV, thinking off and scalar greedy generation without
MTP or suffix drafts. Strata's configured verifier capacity was two, with the
observed target windows all exactly one row and zero drafts. TensorSharp's
8192/9472 MiB cache budgets produced byte-identical complete final vocabulary
logits and cached every subsequent scalar layer. Automatic CUDA placement also
matched explicit host placement on the trained code case.

The 88-token squares answer was repeated with each of the four engines/budgets
first once. The table gives **median (range)** across those four fresh processes.
TensorSharp here uses `TS_HOST_MOE_EXPERT_CACHE_MB=9472` (9.25 GiB ceiling).

| Measurement | TensorSharp | Strata |
|---|---:|---:|
| Reported decode tokens/s | 11.09 (9.22–14.02) | 10.24 (9.37–10.46) |
| Whole-process seconds | 16.54 (14.95–19.31) | 62.15 (59.76–66.89) |
| Sampled device-wide GPU peak, MiB | 14832.5 (14831–14842) | 15729 (15719–15737) |
| OS peak working set, GiB | 19.74 (19.66–19.82) | 18.51 (18.48–18.53) |

TensorSharp counts 87 subsequent forwards; Strata counts all 88 target runs,
including the first generated token. Strata's reported TTFT includes resident
expert setup, while TensorSharp's excludes model construction. Whole-process
latency includes each engine's loading and output serialization. OS page-cache
history and clocks were not controlled: these are retained-page-cache runs,
not cold-storage or warmed-service parity. All prompts exceeded eight rows and
used TensorSharp's existing CPU expert prefill. Its prompt timing differences
are not proof of a prefill cache optimization. Short math/extraction decode
remained slower than Strata; the larger-cache code case was faster. GPU samples
include desktop memory and can miss transients. Working sets include mapped
pages; TensorSharp's higher host working set on squares prevents a claim that
it uses less memory in every tier.

A separate warmed TensorSharp code A/B (one warmup, three measured requests)
reached median 27.92 tokens/s with direct input/output copies, versus 23.29 with
input-only copies and 21.97 with full staging. Every variant matched complete
output IDs and final logits. Prefetch reached 21.43, so it remains off by default.
This A/B does not compare warmed TensorSharp with fresh Strata. CPU-offloaded
final logits differ from CUDA (code relative L2 0.09463) despite identical IDs;
strict CPU numerical parity and broad language quality are not established.

The final native binary (`66e50ad3…`) passed all 11 native tests without skips.
Qwen/MoE regressions passed 327 CUDA tests (17 skipped) and 319 CPU tests
(22 skipped); the changed validation tools passed 49 Python tests. Missing
target/head fixtures, Metal, a QSA opt-in and unavailable multi-GPU scenarios
remain outside coverage; trained MTP integration and TP classes were excluded.
The wider historical validation-script suite remains failed on Windows/path
assumptions, missing September evidence and archived hash mismatches; its
unmodified modules are recorded separately. No trained vision, long-context,
perplexity, MTP or multi-GPU quality/performance claim follows from these checks.
Local evidence is in ignored `docs/validation/qwen38-strata-trained-audit/`,
`qwen38-trained-transfer-ab-final/` and `strata-qwen38/`. Reusable runners and
commands remain in `eng/validation/`.

On CUDA, `--n-cpu-moe` serves the same purpose on a GPU too small for the file. On one
A40 (46 GB) with 12 layers' experts on the host, `ggml_cuda` measured 600 / 30.2 tok/s
(random tokens, pp512 / tg128) against llama-bench's 466.14 / 18.84 with `-ncmoe 12`.

The direct `cuda` engine runs UD-Q2_K_XL's IQ2_XS and IQ3_XXS experts: per-token kernels
ported from ggml's dot products for decode, which still runs as a captured CUDA graph, and
the same two layouts decoded into its tensor-core and register-staged grouped kernels for
prefill. Before the grouped kernels took them, its prefill of this file ran on the slowest
fallback at about 500 tok/s. It also runs UD-Q4_K_XL, which stores 43 of its 48 routed down
projections as Q5_1 (the other five Q8_0) under Q4_K/Q5_K gate/up: decode dots Q5_1 against
q8_1 activations with dp4a as ggml's `vec_dot_q5_1_q8_1` does (the offset term from the int8
activation sum), and prefill decodes it into the same grouped kernels, all held to ggml's
`dequantize_row_q5_1`. Before that the engine refused the file at load, and under
`--layer-split 2` the two devices' refusals aborted the CLI with an unhandled
`AggregateException`; an expert layout it does not decode is now refused before any upload,
with one `error: model load refused:` line naming the tensor. No UD-Q4_K_XL throughput has
been measured on the direct engine yet. A build without nvcc uses the committed PTX, which
includes the Q5_1 decoder; a kernel module built from source older than that decoder makes the
engine refuse Q5_1 experts rather than decode them as Q8_0. The engine has no host-expert seam,
so `--n-cpu-moe` there
prints a warning and keeps the experts on the GPU. The direct `cuda` engine uses
`--layer-split N` for multiple GPUs and refuses `--tp N`. On `ggml_cuda`,
UD-Q2_K_XL supports `--tp 2` with its original quantized weights; see
[Multi-GPU](#multi-gpu) for the supported formats and device/layout requirements.

Both A40s, warm, the same 1,818-token prompt and 256 greedy tokens. TensorSharp ran as
`TensorSharp.Server.Host` with `--no-multi-agent --no-skills`, three requests per process,
each with its own first line so that none reused another's prefix; llama-server answered two
requests with `cache_prompt` off.

| 2x A40, `--layer-split 2` (llama.cpp `-ngl 99`) | prefill tok/s | decode tok/s |
| --- | --- | --- |
| TensorSharp `cuda` | 1,612-1,613 | 56.85-56.93 |
| TensorSharp `ggml_cuda` | 1,174-1,217 | 52.9-53.1 |
| llama.cpp | 752-963 | 59.04-59.86 |

The direct engine's first request after a fresh kernel build ran its prefill at 335 tok/s
while the driver compiled the PTX; the driver caches the result. On random tokens
(pp512 / tg128) the direct engine measured 1,222.0 / 61.0 and `ggml_cuda` 410.9 / 43.2, where
llama-bench's two-GPU run gave 210.74 / 41.99, far below its own server's numbers on the same
machine, so the real-text table is the comparison to go by. There, TensorSharp prefills
1.2-2.1x faster and decodes at 88-90% (`ggml_cuda`) and 95-96% (`cuda`) of llama.cpp's speed.

### Memory limits, 2026-11-07

Two failures surfaced when this model ran beside Strata on ISTA-DASLab's `Q2_0` file, whose
small experts let the tier plan fill every byte it was given.

- **A prompt span that does not fit.** A span's graph grows with its length: 5.1 GB for a
  4K-token span, which the 3 GiB reserve beside a full expert tier cannot hold on a 32 GB
  card. The span used to mark itself unsupported, and with no other path for QSA every later
  request failed. Now a prompt span whose graph cannot be allocated, before anything ran, is
  prefilled again as two halves (recursively, down to 64 tokens), plain or for the MTP head,
  and the length that fit caps later prompt spans; the log says so once per halving. A
  verify window, an image prompt, tensor parallelism and layer splits keep their single span.
- **Metal running out of memory.** Metal cannot fail an allocation and go on: an out-of-memory
  command buffer ends the backend. The tier plan was sized against Metal's working set
  (40.2 GB of a 48 GB Mac), which is a ceiling, not what macOS can give with an editor and a
  browser open. The experts on the accelerator now also stay within what macOS can give
  without compressing anything, less max(7 GB, 15% of the RAM) - Strata's arena rule - and
  within 65% of the RAM wired in all; window slots fit what the tier leaves of the working
  set; retained KV is bounded the same way. On the 48 GB Mac with `Q2_0` that is a 23-26 GB
  tier (97.6-99% of the profiled routes), and the long-prompt scenarios that ran Metal out
  of memory before now pass, plain and with the MTP head.
- **What runs beside the tier.** The plan sets ~3 GB aside for scratch, but a 4K-token prompt
  span's arena alone is 3-5 GB, each concurrent request's state (K/V, indexer keys, recurrent
  state) ~1.1 GB at a 32K context, and idle graph slots keep their last arena. Strata's
  benchmark (16K-token prompts that think, then four concurrent requests) took a 48 GB Mac from
  14 GB free to 2.2 GB and Metal out of memory. Now a graph arena of 512 MB or more that would not
  leave max(1 GB, 1/32 of the working set) free is refused after the kept arenas are released,
  so the prompt runs in halves (`[ts] a 2.89 GB graph arena does not fit ...`); and a new
  concurrent request whose state would not fit waits for a running one to finish
  (`SequenceSlotUnavailableException`, "The model has no device slot for another concurrent
  request"). The same sequence then ran clean: the engine benchmark, Strata's bundle, 12/12
  sanity, three four-request rounds.

Side by side on the same file (`eng/qwen38-engine-bench.py`, greedy, 3 repeats), Strata
(github.com/Niko1221/Strata `146df3c4`) was faster everywhere but on cached follow-ups:

| | long answer decode | 4K prompt TTFT | 16K prompt TTFT |
| --- | --- | --- | --- |
| RTX 5090, Strata (MTP) | 306 tok/s | 0.92 s | 2.75 s |
| RTX 5090, TensorSharp (MTP) | 74 tok/s | 4.17 s | 27.4 s |
| RTX 5090, TensorSharp `--backend cuda`, plain (later the same day, see below) | 98 tok/s | 4.24 s | 17.8 s |
| M5 Pro 48 GB, Strata (DFlash2) | 48 tok/s | 6.32 s | 24.4 s |
| M5 Pro 48 GB, TensorSharp (MTP) | 32 tok/s | 16.8 s | 65.3 s |

A TensorSharp decode token on the 5090 took 11-13 ms: 8.6-10 ms in the 48 accelerator
segments of the tiered span (each a submit and a wait; CUDA graphs on or off changed it by
under 1 ms), 2.3-3.5 ms of host experts. Strata runs a token as one captured CUDA graph,
decides hits on the device and computes misses on a CPU pool in place, concurrently.

### The direct CUDA engine on one GPU that cannot hold the experts, 2026-11-07

`--backend cuda` runs a decode step as one captured CUDA graph. It used to need every routed expert in
VRAM; UD-Q2_K_XL (42.9 GiB of experts) and ISTA-DASLab's `Q2_0` (31.6 GiB) do not fit a 32 GB card.
It now plans **expert residency**, as Strata runs this model:

- The hottest experts by the routing profile, across all layers, get VRAM slots (98.4% of profiled
  routes for `Q2_0`, 86.4% for UD-Q2_K_XL on an RTX 5090). Each stacked expert weight carries a
  table of one address per expert, which the expert kernels read.
- **Host experts** (Strata's doorbell): the other experts are copied once into pinned host memory.
  In a decode step or verify window of up to 8 rows, each layer publishes its input and selection
  to host-mapped memory. A host thread computes the missed experts with ggml's CPU kernels while
  the GPU computes the resident ones, and a GPU kernel waits for the host's sum and adds it. The
  step never leaves its graph and no expert crosses the bus. A missed expert's table entry points
  at a zeroed expert in VRAM, so the GPU's share contributes nothing for it. Steps with misses
  differ from an all-resident run in the last bits (the host quantizes activations as ggml's CPU
  kernels do). `TS_Q4E_HOST_EXPERTS=0` stages every miss over the bus instead.
- Prompts stage the missed experts into VRAM by a copy kernel inside the step.
- The host's worker team takes the CPUs left after the two spinning threads (the stream wait and
  the doorbell server): on the VM's 13.6-CPU quota, 6 workers gave 65.5/45.9/58.8 tok/s, 10 gave
  69.0/48.0/63.5, and 16 (oversubscribed) 63.4/45.1/58.0. `--cpu-moe-threads` overrides.
- `TS_MOE_TIER_ADAPT=1` (opt-in; text then depends on the history): the host counts every step's
  selection, and every 8 steps up to two of a layer's hotter host experts trade places with its
  coldest VRAM experts. The slot's bytes take the host expert's packed place, so memory stays as
  planned.
- Splitting a layer's misses between the CPU and the bus (Strata's PCIe share) was measured and
  removed: with 10+ host workers it was slower (56.2/41.1/49.0 for one miss a layer on the CPU)
  than leaving every miss to the CPU, since the copies sit on the GPU's path and the CPU work
  overlaps it. Strata reports the share pays only with a small CPU pool.

The engine now also reads ISTA's `Q2_0` file: Q2_0, Q4_0 and Q5_0 decoders in the warp-per-row
kernels, a BF16 kernel with F32 activations for the BF16 hyper-connection and PLE value matrices,
and split-K for dense projections with few output rows (the 10240 -> 320 hyper-connection down
projection had run on 80 warps). Without split-K and with whole 64-value Q2_0 units, the engine ran
`Q2_0` at 55.9/43.1/56.9 tok/s, below ggml_cuda.

RTX 5090 (13.6-CPU quota), `requests.jsonl` (code / prose / list, 256 tokens each, greedy, no MTP),
frozen routing profile, one run each:

| file, backend | code | prose | list |
| --- | --- | --- | --- |
| UD-Q2_K_XL, ggml_cuda tiers | 52 | 52 | 48 |
| UD-Q2_K_XL, cuda, staging only | 52.0 | 38.7 | 45.7 |
| UD-Q2_K_XL, cuda, host experts | 70.6 | 47.9 | 64.5 |
| UD-Q2_K_XL, cuda, host experts + `TS_MOE_TIER_ADAPT=1` | 71.6 | 50.8 | 77.4 |
| `Q2_0`, ggml_cuda tiers | 82.6 | 46.4 | 88.4 |
| `Q2_0`, cuda, host experts | 94.6 | 65.6 | 97.3 |
| `Q2_0`, cuda, host experts + `TS_Q4E_HC_Q8=1` (Strata's Q8_0 hyper-connection read) | 106.0 | 70.4 | 109.1 |

A `Q2_0` step's GPU time (`TS_Q4E_PERF=3`, uncaptured, stage timers on) is about 11 ms (9.9 ms with
`TS_Q4E_HC_Q8=1`, whose hyper-connection stage takes 2.3 ms): hyper-connections 3.3 ms, routed experts 2.0, shared expert 1.2, GDN projections 1.1, attention
output 0.9, router 0.7, GDN core 0.7. The weights it reads take about 2.3 ms at the card's
bandwidth; the hyper-connection and shared-expert stages are mostly kernel boundaries, which is
where Strata's fused kernels are ahead.

Those `Q2_0` figures ran **uncaptured**. Layer 10's shared-expert down projection is IQ4_NL with a 640-wide
input. The decode-size dense kernel admitted IQ4_NL only at multiples of 256, so the first capture failed and the
engine issued every step from then on (`decode steps run uncaptured`). The rule now takes whole 32-value blocks for
the formats decoded a block at a time (Q4_0, Q5_0, Q8_0, IQ4_NL, MXFP4). Captured, plain long-generation decode is
103 tok/s instead of 98: the step is bound by the GPU, not by issuing kernels.

#### Speculative decoding on `--backend cuda`, 2026-11-07

- **MTP head.** `--draft-model mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` attaches the shared NextN block to the
  direct engine. It runs entirely on the trunk's kernels:
  - the token row is embedded into all four streams;
  - each stream joins `[rms(e) * enorm, rms(h_s) * hnorm_s]`, laid out by two strided copies;
  - `eh_proj` runs over tokens x streams rows;
  - then come the block's attention (its own KV per sequence) and MoE;
  - its output mixer comes last, then the trunk's LM head.

  The drafter reads the trunk's four streams before the output mixer, as on the GGML backends. On the synthetic
  fixture, three chained draft steps stay within 0.5-0.8% relative error of the native MTP executor's logits, with
  the same argmax.
- **Verify windows** of up to 8 rows run as one ubatch with every row's logits and streams. They are captured as
  CUDA graphs per sequence, QSA regime and row count. The accepted prefix is committed by replaying the recurrences
  over it, bit-exact against a plain decode of the same tokens.
- **One fused launch** for a half's hyper-connection scatter and the next mixer's first pass, about 95 fewer
  launches a step; bit-exact (`TS_Q4E_FUSE_HC=0` restores the two launches).

RTX 5090, `Q2_0` with `TS_Q4E_HC_Q8=1`, the same client as the comparison above:

| run | short chat TTFT / decode | long-gen decode | 4K TTFT | 16K TTFT | cached follow-up TTFT |
| --- | --- | --- | --- | --- | --- |
| plain, uncaptured (before the fix) | 0.11 s / 65 | 98.1 | 2.67 s | 11.4 s | 0.12 s |
| plain, captured | 0.10 s / 69 | 103 | 2.70 s | 11.4 s | 0.11 s |
| MTP, window 3, captured | 0.13 s / 62 | 110 (87% accepted) | 3.36 s | 14.2 s | 0.13 s |
| Strata, MTP | 0.24 s / 114 | 306 | 0.92 s | 2.75 s | 0.12 s |

MTP gains little so far because a 4-row verify window takes 27-32 ms, about three decode steps. A draft step takes
0.62 ms and the fused catch-up 0.88 ms. The head's 2.9 GiB (Q8_0 experts) comes out of the expert slots, leaving
95.0% of routes resident instead of 97.6%. The CPU therefore computes about twice as many missed experts. A verify
window's rows miss four times as many as one decode row, and the GPU waits for the CPU at every layer. Prompts stage
more experts for the same reason, which is why they read slower with the head loaded.

Where a 4-row verify window's GPU time goes (`TS_Q4E_PERF=3`, uncaptured, against a 1-row step): 18.5-23.6 ms
against 10.9 ms. The routed experts take 5.8-10.7 ms against 3.1. The dense projections and the LM head grow with
the rows: attention projections 1.8 against 0.33 ms, GDN projections 1.9 against 1.1, LM head 1.1 against 0.3. The
decode-size dense kernels read every weight row once per activation row. Staging a window's missed experts over
the bus instead of computing them on the CPU (`TS_Q4E_HOST_EXPERT_ROWS=1`) left the window at 27-30 ms. So the
next steps are a multi-row dense kernel that reads each weight row once, and fused expert kernels, not the CPU
side.

A larger prompt ubatch (`TS_Q4E_UBATCH=2048`, default 512) stages each missed expert once per 2048 rows instead of
per 512. 4K prompts take 1.75 s instead of 2.70, and 16K prompts 7.5 s instead of 11.4 (2,290 tok/s). Its scratch
takes 1.2 GiB from the expert slots (96.6% of routes resident instead of 97.6%), so plain decode drops from 103 to
98 tok/s. Strata reads the same prompts in 0.92 s and 2.75 s.

#### Porting Strata's verify window, 2026-11-08

Strata's own profile on the same RTX 5090 and `Q2_0` file (`STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1`):
its verify windows hold 2.4-2.75 rows on average (`--spec 4 --spec-min-p 0.5`). Each takes 7.2-7.7 ms of GPU time
and yields 1.9-2.3 tokens. 99.8% of the routed experts sit in its VRAM cache, which adapts every window, so the CPU
computes about 16 experts a window. Ported, each measured against the change before it:

- **Doorbell skip.** A layer whose selected experts all have slots no longer rings the host. The publish kernel
  records it in a device word and the wait returns at once, where every layer used to wait for the host's echo:
  20% of the GPU's time in an Nsight trace. Bit for bit the same (`TS_Q4E_DOORBELL_SKIP=0` rings every layer).
- **Dense weights read once a window.** Verify rows of the warp-per-row formats (Q2_0, Q4_K, Q5_K, IQ4_XS, ...)
  take one block for every row, a warp per output row: a 4-row window's LM head went from 1.11 to 0.48 ms. Bit for
  bit the one-row kernel's sums.
- **Shared expert on its own stream.** In decode steps and verify windows it runs beside the router and the routed
  experts and joins at their combine, inside the captured graph. Bit for bit.
- **Short expert rows.** The 640-wide expert down projection takes eight lanes a row instead of a warp.
- **BF16 router as stored.** The router is read as the file's BF16, half the bytes of its F32 widening, with the
  same sums.
- **Smaller draft experts.** `TS_Q4E_MTP_EXPERTS=q2_0` requantizes the MTP head's experts at load (Strata ships
  its draft layer so): 2,550 to 675 MiB, which the target's experts then use. This applies to the GGML backends too.
- **Per-token draft gate 0.5** for this model on every backend (Strata's `--spec-min-p`). A draft the head is
  unsure of costs a whole verify row; `--spec-pmin` still overrides it.
- **Adaptive residency for verify windows.** The doorbell kernel counts every layer's selections on the device.
  Every 4 forwards (decode steps, verify windows and batched steps alike) up to 8 trades run, largest gain first,
  between a layer's most-selected host expert and its least-selected VRAM expert. Before, adaptation ran only after
  one-token steps, so never under MTP. The live hit rate had been about 88% against the 96.8% the routing profile
  predicted; adaptation halves the misses. On by default since 2026-11-08 (text then depends on the history;
  `TS_MOE_TIER_ADAPT=0` turns it off); `TS_MOE_TIER_ADAPT_EVERY` and `TS_MOE_TIER_ADAPT_SWAPS` tune it.

RTX 5090, `Q2_0`, `TS_Q4E_HC_Q8=1`, MTP `--spec-draft 3`, the same client (long generation is three answers of
about 1,190 tokens in a row; the adaptive runs speed up as they learn):

| run | long-gen decode | 4K TTFT | 16K TTFT | 4-row verify | experts on the CPU a 4-row window |
| --- | --- | --- | --- | --- | --- |
| MTP before this work (captured) | 109-112 | 3.36 s | 14.2 s | 23.9 ms | |
| all of the above but adaptation, Q2_0 draft experts | 115-125 | 3.02 s | 12.7 s | 21.1 ms | 229 |
| ... with `TS_MOE_TIER_ADAPT=1` and the 0.5 gate | 132-167 | 2.99 s | 12.4 s | 18.3 ms | 123 |
| the same at 32K context (`MAX_CONTEXT=32768`) | 145-180 | 2.53 s | 10.6 s | 15.5 ms | 52 |
| Strata, MTP (131K context, int8 KV with 32K resident) | 306 | 0.92 s | 2.75 s | ~7.5 ms GPU at 2.5 rows | ~16 |

The 32K row shows what the 131K context's F16 attention cache costs: 2.9 GiB of expert slots. Strata keeps its KV
in int8 with only the last 32K positions in VRAM. Left in a 4-row window at 32K: experts 3-4 ms with the CPU's
share, GDN projections 1.9 ms, attention projections 1.8 ms, hyper-connections 3.1 ms, router 1.2 ms.

On the Mac (M5 Pro 48 GB, `ggml_metal`, `Q2_0`, MTP, 32K) the same draft-side changes apply:

| run | long-gen decode | 4K decode | 16K decode | accepted (code / prose / list) |
| --- | --- | --- | --- | --- |
| gate 0.15, Q8_0 draft experts (before) | 25.4-26.8 | 25.3-27.5 | 24.7-26.3 | 88 / 66 / 70% |
| gate 0.5, Q8_0 draft experts | 29.9-31.0 | 30.7-36.2 | 27.5-30.3 | 90 / 72 / 79% |
| gate 0.5, Q4_0 draft experts (`TS_Q4E_MTP_EXPERTS=q4_0`) | 29.3-30.4 | 32.0-37.5 | 29.0-34.4 | 90 / 72 / 81% |
| gate 0.5, Q2_0 draft experts | 29.1-30.6 | 29.6-35.5 | 29.5-32.6 | 89 / 68 / 80% |

The Metal expert tiers live in unified memory and follow their read-ahead window, so the adaptation above is
CUDA-only.

#### Q8_0 KV cache on the direct engine, 2026-11-08

`q8_0` is the KV cache of `--backend cuda` unless `--kv-cache-dtype` names another (`f16` keeps F16). The 12
attention layers keep K and V as int8 codes with
one F16 scale per 32 values (ggml's Q8_0 arithmetic; codes and scales stored apart so a lane reads its 8 codes in one
load), half the F16 cache's bytes. The planner gives the freed VRAM to expert slots. `q4_0` falls back to F16 with a
line on stderr. Against the F16 cache the synthetic fixture's logits move by 3.7e-3 to 4.8e-3 (relative), and a
verify window over the Q8_0 cache answers exactly as a plain decode on that cache.

Same session, RTX 5090, `Q2_0`, `TS_Q4E_HC_Q8=1`, MTP with Q2_0 draft experts, `TS_MOE_TIER_ADAPT=1`, 131K context:

| KV cache | experts in VRAM (profiled routes) | long-gen decode | 4K TTFT / decode | 16K TTFT / decode | 4-row verify | CPU experts a window |
| --- | --- | --- | --- | --- | --- | --- |
| F16 | 15,901 (96.9%) | 123-162 | 2.99 s / 87-96 | 12.5 s / 63-78 | 17.9 ms | 113 |
| `q8_0` | 17,085 (98.0%) | 135-165 | 2.67 s / 98-104 | 11.1 s / 84-85 | 16.9 ms | 82 |

#### Prompt groups, split attention and a lighter MTP catch-up, 2026-11-08

A `TS_Q4E_PERF=3` profile of the 4K prompt on the RTX 5090 showed where the direct engine's time went. Of each 512-row
prompt ubatch's 311 ms of GPU time, 197 ms were the routed experts, almost all of it staging the experts without a
VRAM slot over the bus, and the next ubatch staged most of them again. After the prompt, the first draft replayed it
through the whole MTP layer (attention over the prompt and every row's experts: 59 ms after a 4K prompt, 320 ms in
all for a 16K one, the last part of it in the answer's decode time). And the 12 attention layers of a 4-row verify
window took 2.4 ms at 4K context and 4.9 ms at 16K: one block per head and row, walking up to 2,051 cells one load
after another. Changed:

- **Prompts read layer by layer** (`TS_Q4E_PROMPT_GROUP`, default 4096 rows; one GPU under expert residency). Each
  layer runs over all the group's ubatches before the next one and keeps its staged experts between them
  (`ts_dsv4_moe_stage_copy_from` copies only the experts a ubatch adds), so a missed expert crosses the bus once per
  layer and group. Bit for bit the ubatch order's logits (`APromptReadLayerByLayer_AnswersBitForBitAsUbatchOrder`).
  The group's streams, 160 MiB, come out of the expert slots (16,954 instead of 17,085 resident, 97.9% of profiled
  routes instead of 98.0%).
- **Split attention** (`TS_Q4E_ATTN_SPLITS`, default 16). A decode-size row of more than 256 cells spreads them over
  16 blocks, and the last block to finish merges their partials (flash decoding), inside the captured graph. Shorter
  rows run whole, bit for bit as before; longer ones move within float reassociation (the test's bound is 1e-4
  relative), and a verify row still computes its own decode step.
- **K/V-only MTP catch-up** (`TS_Q4E_MTP_KV_ONLY`, on). A catch-up's rows only write the MTP layer's K/V; the rest of
  the layer runs for the last row alone when it drafts (Strata's `record_front` / `record_rest`): 59 to 23 ms after a
  4K prompt, 320 to 45 ms for a 16K one.
- **Greedy verify on the device** (`TS_Q4E_GREEDY_VERIFY`, on). A plain greedy sampler takes each verify row's first
  maximum from an argmax kernel in the captured window, instead of a megabyte of logits a row crossing the bus to the
  host's scan. Same tokens.
- **The draft chain on the device** (`TS_Q4E_MTP_CHAIN`, on; one GPU, or a layer split with `TS_Q4E_MTP_CHAIN_SPLIT`; no sampling penalties). A window's drafts run as
  one chain: a pick kernel takes each step's first-max token and its top-10 confidence and feeds the next step, so the
  host makes one call a window and reads no draft logits. Drafting went from 2.1-2.2 to 1.9-2.1 ms a window; the end
  to end numbers stayed within run-to-run noise (short chat 74-102, long generation 141-175, 4K 108-131, 16K 154-170
  tok/s). Most of a draft step is GPU work (the LM head over 248K tokens).
- **Q2_0 prompt rows through the tensor cores** (`TS_Q4E_Q2_0_PROMPT_GEMM`, on). ISTA keeps some attention
  projections, most shared experts and the PLE key in `Q2_0`, which the shared prompt route did not decode, so their
  prompt rows ran the warp-per-row kernel, reading the weight once per row (0.23 s of GPU time in a 4K prompt). They
  now take an F16 copy of the weight dequantized for the call and the cuBLAS GEMM: 4K TTFT 1.85 to 1.69 s, 16K 7.4-7.5
  to 6.7-6.9 s, decode unchanged.
- **Adaptive residency learns from prompts** (`TS_MOE_TIER_ADAPT_PROMPT_SWAPS`, default 64). A prompt's expert
  selections are counted on the device, weighed as 128 rows' worth, and up to 64 of its most-selected host experts
  trade places before its answer decodes. After 16K prompts it left 64-136 experts of a window to the CPU instead of
  156-200; after 4K prompts the misses stayed about where they were.

RTX 5090, ISTA `Q2_0`, `TS_Q4E_HC_Q8=1`, MTP (Q2_0 draft experts, `--spec-draft 3`), q8_0 KV, adaptive residency,
131K context, the same client and scenarios, 3 repetitions each (TTFT / decode tok/s; 15/15 PASS in every run):

| run | short chat | long generation | 4K prompt | 16K prompt | multi-turn first TTFT |
| --- | --- | --- | --- | --- | --- |
| before (2026-11-08 morning) | 0.10 s / 62-80 | 130-167 | 2.65 s / 79-115 | 11.1 s / 74-85 | 1.75-1.87 s |
| prompt groups, split attention, K/V-only catch-up | 0.10 s / 63-78 | 141-173 | 1.85 s / 89-129 | 7.5 s / 137-141 | 1.24-1.30 s |
| ... and greedy verify on the device | 0.10 s / 69-74 | 142-179 | 1.78-1.86 s / 96-132 | 7.5 s / 132-153 | 1.27-1.30 s |
| ... and prompts teach the residency (all defaults now) | 0.08-0.10 s / 75-96 | 137-175 | 1.77-1.86 s / 112-137 | 7.3-7.5 s / 152-164 | 1.24-1.28 s |
| Strata (MTP `--spec 4`, int8 KV with 32K resident, route tail skip on) | 0.24 s / 93-115 | 300-307 | 0.92 s / 376-408 | 2.75 s / 387-421 | 1.08-1.18 s |

What is left. After a long prompt 100-270 experts of a 4-row window still go to the CPU, and each costs 40-50 us of
the window (a 4-row window took 12.7 ms with 20 CPU experts and 20-22 ms with 170-270). Strata trades up to 96 experts
a round (`TS_MOE_TIER_ADAPT_SWAPS=96` here cut the misses to about 50 in the 16K requests that followed the 4K ones,
but each round stalls the GPU while it copies) and skips missed experts that every row of the window routes at rank 7 or lower
(`STRATA_ROUTE_TAIL_SKIP=7`, on by default, which changes its output). Prompts still read at 2,150-2,250 tok/s
against Strata's 4,360-5,950, and Strata's windows hold a fifth token (`--spec 4`).

#### Route-tail skip on the direct engine (default), 2026-11-08

Strata skips, by default, the missed experts a window routes only at its tail (`STRATA_ROUTE_TAIL_SKIP=7`). The direct
engine now does the same (`TS_MOE_ROUTE_TAIL_SKIP`, default 7; `TS_MOE_TAIL_SKIP_MAX_CACHED`, default 90; the switch
names are shared with the GGML backends' expert tiers). In a decode step or verify window, a host expert (no VRAM slot)
that every row of the window routes at 0-based rank 7 or lower, the 8th to 10th of Flash Next's 10, is neither
computed by the CPU nor waited for. Its routes add nothing and the weights are not renormalized, so answers become
approximate (Strata measures a KL of 0.003-0.006 and about 98% top-1 agreement on 2-3 bit packs). The doorbell kernel
(`ts_moe_doorbell_publish_tail`) applies the rule to whether a layer rings the host at all, so a layer whose only misses
are in the tail costs no round trip; the host server applies it to the slots of a layer that rings. It is off when
every expert is resident, when 90% or more of the experts have slots, and for prompts, which stage every route.
`TS_MOE_ROUTE_TAIL_SKIP=0` computes every route; the parity tests pin it there.

RTX 5090, ISTA `Q2_0`, MTP, same build and client, 3 repetitions (TTFT / decode tok/s; 15/15 PASS both):

| route-tail skip | short chat | long generation | 4K prompt | 16K prompt | CPU experts a window after 4K / 16K |
| --- | --- | --- | --- | --- | --- |
| `0` (every route) | 0.08-0.10 s / 87-113 | 140-175 | 1.62-1.70 s / 119-144 | 6.75-6.87 s / 150-170 | 185-242 / 95-147 |
| `7` (default) | 0.08-0.11 s / 83-99 | 153-188 | 1.62-1.70 s / 128-176 | 6.76-6.87 s / 163-183 | 128-140 / 42-96 |
| Strata (its default, `7`) | 0.24 s / 93-115 | 300-307 | 0.92 s / 376-408 | 2.75 s / 387-421 | |

Tried and dropped the same day: a prompt attention kernel with a KV head's query heads in one block (each cell's K and
V read once into shared memory, each head's online softmax in one warp) read the 4K prompt in 1.72 s against 1.69 s and
the 16K one in 7.0 s against 6.8 s; one warp walking every cell of a head was slower than eight warps sharing them.

#### Adaptive residency without stalls, 2026-11-08

Each expert the CPU computes costs a verify window 40-50 us, and after a long prompt the 8 blocking swaps a round left
100-170 of them in a window for many windows (a round of 96, `TS_MOE_TIER_ADAPT_SWAPS=96`, caught up faster but stalled
the GPU for its copies). A round now runs beside the decode (`TS_MOE_TIER_ADAPT_ASYNC`, on; Strata's overlapped
adaptive tier): its copies go on a stream of their own in three steps a forward boundary apart, so a window's tables
never name bytes in flight. First both experts of each trade are copied into the prompt staging (idle in decode),
then the tables switch to those copies while the copies go to their final places (the hot expert into the cold one's
slot, the cold one into the hot one's packed host place), then the tables switch to the final places. A prompt or a
wide step first finishes a round in flight. With the stalls gone a round carries 64 swaps.

RTX 5090, ISTA `Q2_0`, MTP, same client, 3 repetitions (TTFT / decode tok/s; 15/15 PASS):

| adaptation | short chat | long generation | 4K prompt | 16K prompt | CPU experts a window after 4K / 16K |
| --- | --- | --- | --- | --- | --- |
| 8 blocking swaps a round | 0.08-0.10 s / 88-102 | 154-182 | 1.62-1.70 s / 135-160 | 6.71-6.88 s / 167-184 | 115-171 / 69-98 |
| 64 overlapped swaps a round (default until 2026-11-10) | 0.07-0.09 s / 75-110 | 174-179 | 1.61-1.69 s / 137-188 | 6.76-6.81 s / 183-198 | 31-146 / 26-33 |
| 96 overlapped swaps a round | 0.07-0.09 s / 82-100 | 169-180 | 1.61-1.69 s / 151-193 | 6.71-6.79 s / 151-193 | 32-96 / 17-28 |

#### Larger adaptation rounds, 2026-11-10

A round that overlaps the decode can carry as many swaps as its staging slots hold (254 for Flash Next), and the
default budget is now 256 instead of 64 (`TS_MOE_TIER_ADAPT_SWAPS`, when set, is still every round's). The answer to a
long prompt routes elsewhere than the residency learned before it, and at 64 a round it kept missing those experts for
many windows. Counted with the host computing every miss (`TS_Q4E_HYBRID_MISSES=0`), the first 16K-token answer of a
sequence had 49 misses a window in its windows 1-8 and 21 in windows 17-64; at 256, 13-17 and 5-6. Moving the prompt's
experts in at the end of the prompt with the GPU waiting (a prototype, not merged) did less (32 / 5) and cost 50-280 ms
of TTFT per 4,096-token prefill chunk.

RTX 5090, ISTA `Q2_0`, MTP: four prompts (4K, 16K, 4K and 16K tokens of repo prose) each answered with a 768-token
summary; two sets of 3 rotating passes; median 4-row verify ms by window position after each prompt:

| swaps a round | windows 1-8 | 9-16 | 17-64 | 65-256 |
| --- | --- | --- | --- | --- |
| 64 (until 2026-11-10) | 13.41 / 12.73 | 11.54 / 11.53 | 11.01 / 10.63 | 10.64 / 10.25 |
| 256 (default) | 10.88 / 10.86 | 10.67 / 9.36 | 10.70 / 10.25 | 10.49 / 10.25 |

Elsewhere, 3 alternating passes each: the engine bench's 16K TTFT 3.28 -> 3.15 s (median) and four requests at once
152-175 -> 150-191 tok/s; a 16K prompt right after a long answer prefilled as before (synchronized stage times:
routed experts 772 -> 781 ms, all stages 3,951 -> 3,885 ms; TTFT 4.22-4.26 -> 4.12-4.24 s). No VRAM is added (a round
borrows the prompt staging), and the bus carried less during the requests (111-129 GB host to device against 133-192
at 64), as fewer missed experts were staged.

#### The next prompt layer staged ahead, 2026-11-08

Read layer by layer, a 4K prompt still spent 0.26 s of GPU time staging each layer's missed experts on its own
stream, the layer waiting for the copy kernel. The staging now holds two halves with their own plans
(`TS_Q4E_PROMPT_PREFETCH`, on): while a layer runs on the half that holds its experts, the copy engines bring every
host expert of the next layer into the other half. The host keeps a layer's experts packed in host-index order, which
becomes the staging slot order, so each matrix is one 2D copy and the plan (slots, list, the layer's tables pointing at
the copies) is written from the host; the next layer waits for those copies instead of staging per ubatch. A half is
reused two layers later, after its layer unstaged. Layers whose host experts do not fit a half (up to 228 of 256 on
the 5090 run) stage per ubatch over the whole staging. The bytes the kernels read are the same, so the prompt answers
bit for bit as before (the prompt-group test now runs it too). The first version used the copy kernel on the second
stream: its blocks wait on the bus while holding the SMs the layer needs, and the 4K prompt took 1.82 s.

RTX 5090, ISTA `Q2_0`, MTP, same client, 3 repetitions (TTFT / decode tok/s; 15/15 PASS):

| prompt staging | short chat | long generation | 4K prompt | 16K prompt | multi-turn first TTFT |
| --- | --- | --- | --- | --- | --- |
| per ubatch (after the overlapped swaps) | 0.07-0.09 s / 75-110 | 174-179 | 1.61-1.69 s / 137-188 | 6.76-6.81 s / 183-198 | 1.09-1.15 s |
| next layer ahead, copy kernel | 0.07-0.09 s / 84-105 | 168-189 | 1.82-1.84 s / 98-148 | 7.37-7.39 s / 194-199 | 1.27-1.33 s |
| next layer ahead, copy engines (default) | 0.07-0.09 s / 72-102 | 172-174 | 1.44-1.46 s / 135-146 | 5.84-5.87 s / 173-195 | 0.89-0.95 s |

#### Two latency-bound decode kernels, 2026-11-08

An nsys capture of a long generation (RTX 5090, ISTA `Q2_0`, MTP) showed two kernels waiting on their own loads. A
draft step's pick read the 248,320 logits in one block of 1,024 threads: 115 us a step, two or three steps a window. It
now runs a block per 1,024 logits, each keeping its slice's ten largest and the first index of its largest, then one
block merging those candidates by the same rules (`TS_Q4E_DRAFT_PICK_SPLIT`): 12.8 + 12.0 us, the same token and
confidence bit for bit. The BF16 router (512 x 2560, a warp per expert) issued each lane's loads one at a time; it now
keeps four in flight with the sums in their order: 14.2 -> 12.6 us a 4-row layer. About 0.3 ms of a ~13 ms window,
inside the run-to-run noise of the client below; the window is bound by the CPU's missed experts (a third of the
layer-windows wait 85-135 us for them) and the expert and dense kernels.

Tried and dropped in the same round: the multi-row dense kernel reading each weight unit once with a sum per row
(18.9 us a call against the current kernel's 17.8: its later passes over a row already hit L1; it was in the build of
the table's second row); the decode experts
grouped by expert, so a window's slots sharing an expert read it once (a verify window 12.8-13.1 ms against 10.3-11.6:
the owning blocks serialize their slots and the grid shrinks); and letting the host-expert team spin between layers
instead of sleeping (no change: a ringing layer's wait is the experts' arithmetic, not the wake-up).

Same client, 3 repetitions (TTFT / decode tok/s; 15/15 PASS):

| decode kernels | short chat | long generation | 4K prompt | 16K prompt | multi-turn first TTFT |
| --- | --- | --- | --- | --- | --- |
| before (the next prompt layer staged ahead) | 0.07-0.09 s / 72-102 | 172-174 | 1.44-1.46 s / 135-146 | 5.84-5.87 s / 173-195 | 0.89-0.95 s |
| two-stage pick, router loads in flight | 0.07-0.09 s / 82-102 | 169-182 | 1.38-1.44 s / 140-185 | 5.79-5.86 s / 189-193 | 0.89-0.95 s |

#### Prompt attention on tensor cores, 2026-11-08

A prompt's attention ran one block per query head and row, whose warps took the row's cells one warp-wide dot at a
time; the 12 query heads sharing a KV head each loaded and widened every cached key and value again. That was 0.18 s
of a 4K prompt, and a 16K prompt's rows each attend 2,051 cells. Prompt-size calls now run a block of four warps per
row and KV head (`TS_Q4E_ATTN_MMA`, on): the group's query heads are the rows of a 16-row tensor-core tile, each warp
widens a chunk of 16 cells' keys and then values to F16 in shared memory once, and S = QK^T and O += PV run on
`mma.sync` (F16 operands, F32 accumulation) under an online softmax; the warps merge at the end. Decode steps and
verify windows keep their kernels, so a verify row still answers as its own decode step. The operands round to F16:
on the test fixture a prompt's logits move 4e-3 to 5e-3 relative (the `q8_0` cache moves them 4e-3 to 5e-3 from F16),
and the kernel itself stays within 5.3e-4 a row of the F32 kernel (`Q4eAttentionMmaTests`). It needs sm_80 or newer and a module built
for the GPU; the committed compute_75 baseline has no tensor-core body, and the engine then says so once and keeps the
old kernel.

RTX 5090, ISTA `Q2_0`, MTP, same client, 3 repetitions (TTFT / decode tok/s; 15/15 PASS):

| prompt attention | short chat | long generation | 4K prompt | 16K prompt | multi-turn first TTFT |
| --- | --- | --- | --- | --- | --- |
| a block per head and row | 0.07-0.09 s / 82-102 | 169-182 | 1.38-1.44 s / 140-185 | 5.79-5.86 s / 189-193 | 0.89-0.95 s |
| tensor cores (default) | 0.07-0.09 s / 91-99 | 161-187 | 1.23-1.29 s / 99-161 | 4.99-5.08 s / 187-198 | 0.81-0.85 s |

#### A 1024-row prompt ubatch by default, 2026-11-08

With the next layer staged ahead and the attention on tensor cores, what a prompt ubatch pays per call (each expert's
weights decoded per 32-row pass of the grouped kernel, a layer's fixed launches and syncs) became the larger share, and
`TS_Q4E_UBATCH=1024` read a 4K prompt 17% faster. The engine now defaults to 1024 prompt rows a forward
(`TS_Q4E_UBATCH=512` restores the old value). The larger scratch takes 345 expert slots (16,609 instead of 16,954 in
VRAM, 97.6% instead of 97.9% of the profiled routes); long generation stayed within the run-to-run noise.

RTX 5090, ISTA `Q2_0`, MTP, the same build, same client, 3 repetitions (TTFT / decode tok/s; 15/15 PASS):

| prompt ubatch | short chat | long generation | 4K prompt | 16K prompt | multi-turn first TTFT |
| --- | --- | --- | --- | --- | --- |
| 512 | 0.07-0.09 s / 89-110 | 174-177 | 1.29 s / 121-155 | 5.05 s / 178-199 | 0.81-0.85 s |
| 1024 (default) | 0.07-0.09 s / 80-118 | 168-179 | 1.07-1.08 s / 114-185 | 4.16-4.19 s / 182-189 | 0.63-0.68 s |
| Strata (11-07) | 0.24 s / 93-115 | 300-307 | 0.92 s / 376-408 | 2.75 s / 387-421 | 1.08-1.18 s |

#### Decode-once dense kernels, 2026-11-08

A verify window's dense projections ran at 0.3-0.7 TB/s on the RTX 5090. A microbenchmark of the kernels at Flash
Next's shapes (weights rotated past L2) showed why: the multi-row kernel's time grew with the rows (the 2560 -> 10240
IQ4_XS projection 15 us for one row, 33 for four), because each activation row decoded every weight unit again and an
IQ4 nibble went through a byte load from the lookup table. The pack's Q3_K projections (25 of the 36 GDN gates, 18 qkv
projections) and the `TS_Q4E_HC_Q8` hyper-connections ran on the shared q8_1 kernels, up to 90 us a call.

The new kernels (`TS_DENSE_FAST`, on) decode each unit once - the IQ4 table held in registers and applied with
`__byte_perm`, as llama.cpp's `get_int_from_table_16` does - and dot it against every activation row, with 1, 2 or 4
warps an output row by shape. They cover Q8_0, Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS and IQ4_NL, decode steps (one row)
included, so a row's sums depend on the shape alone and a verify window's row is still its own decode step's bit for
bit (`Dsv4DenseRowInvarianceTests` holds rows 2-16 to the one-row call and to the dequantized weights). The router GEMV
takes a block of 256 threads per expert (`TS_DSV4_GEMV_BLOCK`), and the PLE and MTP group norms a block per group
instead of one per row (same sums).

4 rows, RTX 5090, cold weights (us): IQ4_XS 2560 -> 10240 33.4 -> 11.1; Q3_K 2560 -> 10240 32.8 -> 15.4; Q3_K
2560 -> 6144 22.9 -> 10.7; Q4_K 2560 -> 6144 18.9 -> 11.9; Q2_0 2560 -> 12288 18.7 -> 11.9; Q6_K 6144 -> 2560
21.4 -> 11.8; Q8_0 hyper-connection 10240 -> 320 11.5 (shared kernel) -> 5.1 and 320 -> 10240 6.5 -> 4.9; the
2560 -> 151643 Q5_K LM head 298 -> 208; the BF16 router 12.1 -> 4.4. Tried and slower: 2 or 4 warps an output for
the long-output shapes; a "one pass" kernel that kept the per-format decoder and summed the rows in one walk (the
compiler did not share the decode: no gain for IQ4_XS).

MTP, same client, 3 repetitions (TTFT / decode tok/s; 15/15 PASS):

| dense kernels | short chat | long generation | 4K prompt | 16K prompt | multi-turn first TTFT |
| --- | --- | --- | --- | --- | --- |
| before (d620953d) | 0.07-0.09 s / 84-108 | 166-178 | 1.07-1.08 s / 121-161 | 4.16-4.18 s / 179-189 | 0.63-0.68 s |
| decode once, block router, group norm | 0.07-0.10 s / 86-127 | 202-222 | 1.07-1.08 s / 149-167 | 4.16-4.18 s / 207-217 | 0.63-0.69 s |

#### A draft vocabulary for the MTP head, 2026-11-08

Each draft step ran the trunk's LM head over all 248,320 tokens: about 270 us of a ~1 ms draft chain on the RTX 5090.
The draft chain now scores token ids [0, 151643) only, the Qwen2 base vocabulary (`TS_Q4E_DRAFT_VOCAB`, `0` for the
whole vocabulary); the ids past it are the later multilingual additions and the special tokens, 0.1-0.2% of English
and mixed English/Chinese text and 0.01% of Chinese. The drafts only propose: the verify window scores every token, so
the answers do not change. With the answers made history-independent (`TS_MOE_ROUTE_TAIL_SKIP=0 TS_MOE_TIER_ADAPT=0`)
the English, Chinese and code answers were identical with the setting on and off, none of their tokens past the subset,
and the acceptance moved within noise: English 58.3 vs 58.4%, Chinese 55.3 vs 56.0%, code 85.9 vs 85.2%. The head of a
draft step reads 61% of its rows (~165 us). `eng/draft-acceptance.py` runs that comparison (`run` the prompts against a
server, `stats` from its log, `compare` two runs' texts).

Same build, same client, 3 repetitions (TTFT / decode tok/s; 15/15 PASS):

| draft head | short chat | long generation | 4K prompt | 16K prompt | multi-turn first TTFT |
| --- | --- | --- | --- | --- | --- |
| all 248,320 tokens | 0.07-0.10 s / 86-127 | 202-222 | 1.07-1.08 s / 149-167 | 4.16-4.18 s / 207-217 | 0.63-0.69 s |
| ids below 151,643 (default) | 0.07-0.09 s / 109-154 | 214-234 | 1.07-1.08 s / 150-183 | 4.16-4.17 s / 203-223 | 0.62-0.68 s |

#### Gate/up experts at sixteen lanes a row, 2026-11-08

The decode expert gate/up kernel gave each 2560-wide row a warp that walked it in 16-value Q2_0 quarters, five
dependent steps a lane and four rows in turn: a decode step's ten experts took 22.7 us on the RTX 5090 (412 GB/s over
their bytes) while their down projections took 3.9 us. It now takes sixteen lanes a row and two rows a warp, Q2_0 rows
in 32-value units (`TS_MOE_GATEUP_HALF`): 4.4 us, and a 4-row window's 40 slots 44 -> 23 us (14 us when 25 of them are
distinct). End to end it moved nothing measurable: in the layers that wait for the CPU's experts (a third of them) the
GPU's experts finish earlier only to wait longer, and the rest is inside the run-to-run noise (long generation 210-224 vs
211-225 tok/s). Plain decode steps (one row) gain the most.

#### The GPU's share of the missed experts, 2026-11-08

A third of a verify window's layers wait for the CPU's missed experts: the host computes each (expert, row) route at
DRAM speed, ~30-35 us on the pod's 7950X, while the GPU, its own experts done, waits (about 2 ms of an 11-13 ms window
in an nsys capture). The GPU can now take some of those misses itself (`TS_Q4E_HYBRID_MISSES`, distinct experts a
step): the doorbell kernel first plans the GPU's share - those experts' table entries point at slots of the expert
staging, their ids reach the host as -1 - and rings the host for the rest at once; the staging copy then reads the
planned experts over the bus, one copy per expert whatever rows use it, and the GPU's expert kernels compute them with
the resident ones. A layer whose misses the GPU takes whole rings no doorbell. All of it is device state, so a
captured step replays it.

`TS_Q4E_HYBRID_SPLIT` picks the share. `odd` takes odd-numbered experts only, so an expert goes to the same side
whatever else a call selects and a verify row computes what its own decode step does while the share holds them
(`ExpertResidency_GpuShareOfMisses_VerifiesAsItsOwnDecode`); `first` takes the first misses in slot order - more of
them, but which side a row's expert lands on depends on the window. The GPU's experts compute as all-resident ones do,
so the answers move within the host-compute tolerance, towards all-resident: a share that takes every miss answers bit
for bit as all-resident (`ExpertResidency_GpuShareOfMisses_AnswersAsAllResident`).

RTX 5090, ISTA `Q2_0`, MTP, long generation, same build and client (decode tok/s; a 4-row window's verify time and the
experts left to the CPU, from `TS_Q4E_PERF`):

| share | long generation | 4-row verify | CPU experts a window |
| --- | --- | --- | --- |
| 0 (CPU takes every miss) | 208-217 | 11.7-13.3 ms | 41-66 |
| 2, `odd` | 214-226 | 11.2-12.9 ms | 20-35 |
| 4, `odd` | 214-232 | 11.3-12.8 ms | 20-33 |
| 2, `first` | 210-232 | 10.9-12.2 ms | 6-13 |

On by default since 2026-11-08 (`TS_Q4E_HYBRID_MISSES=4`, `TS_Q4E_HYBRID_SPLIT=first`): with four misses a step the
GPU takes nearly all of them (one to three left to the CPU a window). Same client, the earlier build before and after
the others (ABBA), 3 repetitions each:

| build, share | long generation | 4-row verify | CPU experts a window |
| --- | --- | --- | --- |
| before (no share), first run | 215-226 | 11.6-13.1 ms | 41-64 |
| this build, no share | 213-229 | 11.4-13.0 ms | 40-67 |
| this build, 2 `odd` | 216-236 | 11.1-12.3 ms | 22-35 |
| this build, 2 `first` | 223-245 | 10.7-12.0 ms | 7-13 |
| this build, 4 `first` (default) | 231-239 | 10.7-12.1 ms | 1-3 |
| before (no share), second run | 203-227 | 11.7-13.1 ms | 41-67 |

4K and 16K TTFT and the multi-turn first TTFT did not move (prompts stage their misses as before). An nsys capture
(share 2, `first`) shows the trade: the doorbell waits 2.0 -> 0.28 ms a window, the staging copies 2.7 ms (about
47 GB/s over the bus, on the stream before the expert kernels). Overlapping those copies with the resident experts
(a second expert pass for the staged ones) is the next step.

#### Smaller decode kernels, 2026-11-08

Kernel by kernel, each the same bits as before (old and new compared byte for byte at Flash Next's shapes, RTX 5090,
CUDA-graph replays):

- The hyper-connection partials (and the fused scatter-and-partials) summed their five values - the sum of squares and
  four scatter logits - one block reduction after another, three barriers each; they now share one barrier, each value
  summed in the same order: 2.56 -> 1.61 us a 4-row call, about 90 a window.
- The mixer's collapse now also writes the q8_1 blocks of its output (`TS_Q4E_COLLAPSE_Q81`), which the attention, GDN,
  shared-expert and routed-expert projections read instead of quantizing it again: 2.30 -> 1.58 us, and one launch
  fewer on the routed experts' path.
- The QSA selection walked its 256-bin histograms on one thread, four passes a row; a block-wide scan finds the same
  bin: 15.2 -> 6.1 us at 4K context, 26.5 -> 18.1 us at 16K.
- The GDN front end runs two warps a block instead of eight (more SMs for its 48-192 warps): 5.26 -> 4.74 us.

Tried and dropped: four or two cells in flight per warp in the split attention (4-row QSA rows 32.8 -> 38.6 and 37.5
us: the registers cost more occupancy than the overlapped loads gained); keeping the router's picks in lanes and
storing them once (4.81 -> 4.74 us, inside the noise). A tensor-core verify attention (a window's rows sharing one
read of each KV head's cells) is not built: a decode step would have to run the same kernel for its row to stay equal
to a verify row, and the split kernel's 33 us for 4 QSA rows is about 0.4 ms of a window.

#### A resident KV window, 2026-11-08

At a 131K context one sequence's attention caches and QSA indexer keys take 2.3 GB of the RTX 5090's 32 GB (Q8_0
K/V 1.7 GB, raw keys 0.4, pooled keys 0.2) - about 1,700 expert slots. Strata keeps 32K cells of each attention
layer resident and the rest in host memory (`--kv-resident 32768`); `TS_Q4E_KV_RESIDENT` does the same for the direct
engine's QSA layers (Q8_0 KV only):

- The authoritative K/V of the whole context lives in pinned, device-mapped host memory in the usual layout; VRAM
  holds a window of 4-cell pages and a page table (page -> slot).
- A decode step or verify window (up to 16 rows) reads at most 16 x 2,051 cells a layer: before its attention, one
  block makes every page it reads resident - hits stamped, misses given slots no page of this call uses (round robin) -
  and a copy kernel reads the misses over the bus. Both run inside the captured step. The window holds at least one
  such call's pages (32,960 cells; a smaller request is raised to it).
- The K/V writer stores each new cell into the host copy and, when its page is resident, into the page's slot, so a
  slot never goes stale (a rejected draft's cells are simply written again).
- A prompt reads whole contexts: each layer runs on a staging pool of the full context (one per GPU, 143 MB at 131K,
  shared by the layers in turn), filled from the host copy once per layer and prompt group; the ubatch's new cells go
  back to the host copy by DMA and leave the window.
- The raw indexer keys become a ring of a ubatch plus two blocks (a block's raw keys are read only to pool it); the
  pooled keys stay whole, since every step scores every block.

The readers go through the page table with the resident kernels' arithmetic, cell for cell, so a streamed layer
answers bit for bit as a resident one (`AResidentKvWindow_AnswersBitForBitAsResidentCaches`: a 600-token prompt in
128-row ubatches, 150 decode steps whose selections move over the whole context and a verify window, through a
448-cell window, against resident caches). Unset, the engine streams with Strata's 32,768 cells (raised to the
32,960-cell floor) whenever the routed experts do not fit in VRAM whole, and keeps every layer resident when they do;
`0` keeps every layer resident.

On by default since 2026-11-08 when the experts do not fit whole. On the RTX 5090 (ISTA `Q2_0`, MTP, 131K context)
the window freed 1.43 GiB a sequence less the 136 MiB staging pool: 17,718 experts in VRAM instead of 16,608, 98.4% of
the profiled routes instead of 97.6%. Same build and client, 3 repetitions (TTFT / decode tok/s; every needle found):

| KV | short chat | long generation | 4K prompt | 16K prompt | 48K prompt | 100K prompt | multi-turn first |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all resident (`0`) | 0.07-0.09 s / 114-162 | 222-238 | 1.07-1.08 s / 167-172 | 4.16-4.18 s / 234-249 | 12.9-13.0 s / 184-210 | 29.5 s / 158-169 | 0.64-0.67 s |
| 32K window (default) | 0.07-0.09 s / 119-167 | 234-245 | 1.08 s / 181-185 | 4.38-4.40 s / 224-250 | 13.8-13.9 s / 201-221 | 31.7 s / 170-187 | 0.64-0.67 s |

Decode gains everywhere (more experts resident; a step's missed pages are few and small). Prompts of 16K and more pay
5-7% TTFT: each layer's cells are copied into the staging pool once per 4,096-row prompt group, on the prompt's
stream. Staging the next layer ahead on a stream of its own (as the experts are) would hide most of it.

#### Concurrent requests and Strata's standard-v1 workload, 2026-11-08

`eng/strata-standard-bench.py` replays Strata's frozen `bench/prompts/standard-v1` bundle (Strata checkout
146df3c4, `MANIFEST.sha256` verified) with exactly the payload Strata's runner sends: thinking on (`reasoning_effort`
xhigh, `preserve_thinking`, `reasoning_budget_tokens` 0), temperature 1.0, top_p 0.95, top_k 20, per-request seeds, a
512-token cap; short, 4K and 16K requests three times, 12 sanity checks scored with Strata's rules, and three rounds of
four requests at once. The server reads the fields as Strata's does: the GGUF's chat template is Strata's
`serve/chat_template.jinja` (the same text), `xhigh` reaches it as `high`, which the template maps back to xhigh;
`preserve_thinking` is reported as ignored, and the template's default without it is the same `true`; a budget of 0 is
no budget on both.

Concurrency was the gap. Under expert residency the plan left VRAM for one sequence's 131K-context caches (2.3 GB), so
a second request found no slot and waited ("The model has no device slot for another concurrent request"): four
requests at once ran one after another at the single-stream rate, TTFT 0.1 / 3 / 6 / 9 s. `TS_Q4E_LANES` (1-16,
default 1) has the plan leave VRAM for that many sequences; concurrent requests then get slots of their own and decode
together. A lane costs about 1.5 GB with the resident KV window (four lanes: 92.9% of the profiled routes resident
instead of 98.4%).

Concurrent requests also speculate together. Each scheduler step drafts every sequence's MTP chain on its own slot and
verifies all of their windows in one forward (`Q4eCudaEngine.ForwardBatchedVerify`): the recurrent layers scan each
sequence's rows from its own state, attention issues each sequence's window as its own verify does, and the commit
replays the kept rows of a sequence that rejected a draft. Greedy requests take each row's first-max token from the
device; sampled requests take every row's logits and draw them with their own samplers, sequences in parallel, a draft
kept while the draw equals it, as a solo step does. Two things had to change first:

- The MTP block keeps its own KV and refused to draft after positions it never replayed, so a request prefilled beside
  others (plainly) could never arm speculation, and concurrent requests decoded one token a step. The block now restarts
  its cache at the end of such a gap (`Q4eCudaEngine.MtpRestart`; positions count from the restart, RoPE scores depend
  on distance alone). This also lets a turn that reuses a prefix speculate. `TS_Q4E_MTP_RESUME=0` restores the old rule.
- Rows of different sequences share few experts, so a verify costs nearly in proportion to its rows: 4 / 8 / 12 rows in
  9.5 / 14.5 / 24.4 ms, each draft kept at about 0.62. Three sequences with 4-row windows decoded 211 tok/s in all
  against about 296 for plain batched steps. A step keeps to `TS_Q4E_BATCHED_VERIFY_ROWS` rows (default 8): two
  sequences verify 4 rows each, three or four 2 (365 tok/s at four sequences), more decode plainly.

Batched steps (batched decode and batched verify) run through captured CUDA graphs (`TS_Q4E_BATCH_GRAPHS`), and give
their memory back when a slot cannot be allocated or grown.

RTX 5090, ISTA Q2_0 with the MTP head, `--sampling-precedence request`, three repeats each, aggregate tok/s:

| | engine-bench parallel_2 | engine-bench parallel_4 | standard-v1 concurrency (4 at once) |
| --- | --- | --- | --- |
| Strata `--batch 4` | 214-216 | 293 | 293-329 |
| Strata, solo | | | 231-243 (serialized) |
| TensorSharp at the round's start (1 lane) | 149-169 | 161-165 | 148-161 (serialized) |
| TensorSharp, 1 lane (the default) | 191-203 | 198-216 | 200-201 (serialized, each request speculating) |
| TensorSharp, 4 lanes, plain batched decode | 202-212 | 282-296 | 263-272 |
| TensorSharp, 4 lanes, batched verify | 162-196 | 281-286 | 243-258 |

Per request the batched verify decodes 92-116 tok/s against 79-89 for plain batched decode. One request of four used to
wait 3-6 s for a slot in the concurrency rounds: the lane plan did not count the rewind checkpoint's copies, a slot
whose free was refused while it was the engine's selection was never freed, and the shared reclaim stopped at a
retained slot it could not free. With those fixed every request's TTFT is 0.62-0.67 s. Sanity stays 12/12 and no
request fails. Greedy answers
of four requests at once are byte-identical to the same requests one at a time with the history-dependent shortcuts off
(`TS_MOE_ROUTE_TAIL_SKIP=0 TS_MOE_TIER_ADAPT=0 TS_Q4E_HYBRID_MISSES=0`, prefix reuse off): 4 of 4, and so are two solo
runs. With the defaults even two solo runs differ (1 of 4), as the route-tail skip and adaptive residency intend.

ggml_cuda cannot serve this model on the 32 GB card: decode at 0.5-4.7 tok/s (the experts read from the network volume),
and four requests at once run it out of device memory inside ggml, which aborts the process; it has no admission check
like ggml_metal's.

#### The prompt DeltaNet recurrence in chunks, 2026-11-09

A prompt's gated DeltaNet layers walked their rows one at a time (KdaScan: a block per value head, each token a rank-1
update of the 128 x 128 state). That was 0.49 s of a 16K prompt; Strata spends 0.07 s. A single sequence's prompt of
128 rows or more now takes Strata's chunked form (`TS_Q4E_GDN_CHUNKED`, on), ported into `tensorsharp_q4e_kernels.cu`.
A prep kernel builds each 32-token chunk's WY factors (the in-chunk triangular solve and gate products) in FP32. A scan
kernel then applies a chunk as two small matrix products instead of 32 rank-1 updates; it runs 16 key heads x 8 column
tiles, 384 threads each, one wave on the GPU, and carries the state across chunks in shared memory. Super-blocks of
2,048 rows bound the scratch (about 25 MB, allocated at the first long prompt). Decode steps, verify windows, batched
steps, captured graphs and prompts under 128 rows keep KdaScan. Everything stays FP32, but the sums run in another
order: against the token-by-token scan the outputs move 3e-7 relative and the state at most 6.2e-7 a head
(`Q4eGdnChunkedTests`, 128, 200 and 4,171 rows). The scan needs 93,696 bytes of opt-in shared memory a block and a GPU
with at least 128 SMs, so that every block runs in one wave; with fewer it would run in waves and lose. Otherwise the
engine keeps KdaScan.

RTX 5090, ISTA `Q2_0`, MTP, model in `/dev/shm`, the same build (`TS_Q4E_GDN_CHUNKED=0` for the old scan). TTFT of
three runs, the first cold; the engine bench's long-prompt and long-generation needles pass 6/6 in both arms:

| prompt recurrence | 4K prompt (3,541 tokens) | 16K prompt (14,094 tokens) | engine bench 4K / 16K TTFT | long generation |
| --- | --- | --- | --- | --- |
| token by token | 1.06 / 0.79 / 0.79 s | 3.36 / 3.35 / 3.35 s | 1.00 s / 4.20-4.22 s | 239-242 tok/s |
| chunked (default) | 0.99 / 0.71 / 0.70 s | 3.04 / 3.02 / 3.02 s | 0.91 s / 3.79 s | 242-249 tok/s |

#### Q2_0 prompt experts on int8 tensor cores, 2026-11-09

A prompt's routed experts ran on `ts_dsv4_moe_mma_f32`. For every 128-token group, each CTA widened its expert's Q2_0
weights to F16 (scale times code, in float math) and the activations to F16 beside them, then ran F16 `mma.sync` with
F32 accumulation, which a GeForce card issues at half its F16 rate. SwiGLU and the down product's input were separate
passes over F32 buffers. This was the largest stage of a 16K prompt: 1.08 s of experts, where Strata spends 0.32 s.

Strata runs these experts on int8 tensor cores, and that is now ported (`TS_Q2_0_EXPERTS_INT8`, on) as
`ts_dsv4_moe_q2i8_*` in `tensorsharp_dsv4_kernels.cu`:

- The Q2_0 codes 0..3 go into `mma.m16n8k32` as int8 operands, unchanged.
- The activations are rounded to int8 per 32 values. The scale is amax / 127, as in the decode kernels' q8_1 rows, but
  kept in F32. Each block stores minus its own sum, so the codes' -1 offset costs nothing: one integer add turns the
  integer dot into a float exactly.
- The gate and up rows of eight features share an m16 tile, so SwiGLU, and the rounding of its output to the down
  product's int8 activations, happen in registers.
- Strata repacks the weights; these kernels instead read the GGUF blocks in place through the expert residency table.
  A pipeline stage copies two 18-byte blocks of each row (36 bytes on a 4-byte boundary) with `cp.async`.
- The scratch reuses the F32 gate/up outputs this route no longer writes.
- It needs sm_80 or newer and a module built for the GPU. On the committed compute_75 baseline the F16 kernel runs.

The rounding is the decode kernels'. On the model's expert shapes (`CudaMoeQ2_0Int8Tests`), relative to the expert block
computed in double, the int8 kernels move it 9.77e-3, the per-slot q8_1 kernels (the arithmetic of ggml-cuda's MMQ)
9.73e-3, and the F16 kernel 1.9e-4.

On the real model (`Qwen4ExpQ2_0Int8PromptTests`, six prompts of 22 to 12,347 tokens, the history-dependent shortcuts
off), the last prompt row's logits move 4.9e-2 to 1.1e-1 relative from the F16 kernel's. Running the same prompts'
experts on the per-slot q8_1 kernels, in ggml-cuda's arithmetic, moves them 4.8e-2 to 1.8e-1. With int8 the first token
is the same in 6 of 6 prompts, and the 64-token greedy continuations stay identical in 4 of 6. The other two part at a
near-tie ("1." against "-", "scan through the document" against "scan the document"); q8_1 parts at the first of them
too. The F16 control repeats bit for bit.

One layer of 512 experts, top 10, over 1,024 rows takes 0.85 ms instead of 1.64 ms.

RTX 5090, ISTA `Q2_0`, MTP, model in `/dev/shm`, the same build (`TS_Q2_0_EXPERTS_INT8=0` for the F16 kernel). TTFT of
three runs, the first cold. The engine bench's short_chat, long_prompt_4k/16k and long_generation checks pass 8/8 in
both arms:

| prompt experts | 4K prompt (3,540 tokens) | 16K prompt (14,092 tokens) | engine bench 4K / 16K TTFT | long generation |
| --- | --- | --- | --- | --- |
| F16 tensor cores | 0.99 / 0.71 / 0.70 s | 3.04 / 3.02 / 3.03 s | 0.86-0.90 s / 3.79-3.80 s | 248-255 tok/s |
| int8 tensor cores (default) | 0.84 / 0.57 / 0.55 s | 2.48 / 2.47 / 2.47 s | 0.74-0.75 s / 3.13-3.14 s | 240-249 tok/s |

#### QSA block scores of a prompt on tensor cores, 2026-11-09

A QSA layer scores every complete block a row can see (the sum over the four indexer heads of relu(block key . query
head)) before it keeps the best. A prompt's rows did this one warp per block on the CUDA cores, each row reading every
key again: 0.10 s of a 16K prompt, where Strata spends 0.02 s.

Strata scores a prompt's rows on tensor cores, and that is now ported (`TS_Q4E_QSA_SCORES_MMA`, on) as
`ts_q4e_qsa_scores_mma_f32`:

- A block of four warps takes 16 rows (their four heads, 64 query rows) against 64 blocks' keys, so a key is read
  once per 16 rows.
- Strata's keys are F16; TensorSharp keeps the pooled keys in F32. So both operands go in as F16 high and low halves:
  q . k = qh . kh + ql . kh + qh . kl, with F32 accumulation. The dropped ql . kl term is below 2^-22 of the product.
- The heads' relus are summed in head order, as before.
- Decode steps, verify windows and batched steps keep the warp kernel, so a row still scores as its own decode step.
- It needs sm_80 or newer and a module built for the GPU; the committed compute_75 baseline keeps the warp kernel.

On rows starting at the context's beginning and 3,000 and 9,000 positions in (`Q4eQsaScoresMmaTests`), the scores stay
within 4.6e-7 of the sum of |q . k| from the scores computed in double (the warp kernel: 6.1e-8). The blocks the
selection keeps are the same in all 1,045 rows, 712 of them past the selection width.

RTX 5090, ISTA `Q2_0`, MTP, model in `/dev/shm`, the same build (`TS_Q4E_QSA_SCORES_MMA=0` for the warp kernel). TTFT of
three runs, the first cold. The engine bench's short_chat, long_prompt_4k/16k and long_generation checks pass 8/8 in
both arms:

| prompt QSA scores | 4K prompt (3,541 tokens) | 16K prompt (14,093 tokens) | engine bench 4K / 16K TTFT |
| --- | --- | --- | --- |
| a warp per block | 0.85 / 0.58 / 0.56 s | 2.48 / 2.47 / 2.47 s | 0.74 s / 3.12-3.13 s |
| tensor cores (default) | 0.84 / 0.56 / 0.55 s | 2.39 / 2.38 / 2.38 s | 0.73-0.74 s / 3.00-3.01 s |

#### A layer's dense weights dequantized once per prompt group, 2026-11-09

A prompt's dense projections (Q4_K, IQ4_XS, Q5_K, Q3_K, Q4_0 and Q2_0 in the ISTA pack, and the Q8_0 hyper-connection
copies under `TS_Q4E_HC_Q8=1`) run as F16 tensor-core GEMMs on a copy of the weight dequantized for the call. A prompt
group (`TS_Q4E_PROMPT_GROUP`, 4,096 rows) already runs each layer over its four 1,024-row ubatches back to back, but
each ubatch dequantized the layer's weights again. On a 16K prompt that was 183 ms of the 242 ms that dequantizing and
converting took (nsys). Strata dequantizes per 8K-token chunk.

Now (`TS_Q4E_GROUP_F16`, on) the first ubatch of a layer dequantizes each weight into the group's F16 arena and the
others reuse that copy. The copy is the same kernel's output, so every product is the same bit for bit.
`APromptGroupsDenseWeights_DequantizedOncePerLayer_AnswerBitForBit` runs two 520-row ubatches a layer and checks that the
prompt's and the decode steps' logits are unchanged.

The arena holds the largest layer's quantized projections as F16: 194 MiB for the ISTA pack, a GDN layer with the PLE
key. Residency plans around it, as around the group's streams, so 147 experts fewer stay in VRAM (17,632 -> 17,485,
98.4% -> 98.3% of the profiled routes). That can move decode steps, which compute host experts on the CPU.

RTX 5090, ISTA `Q2_0`, MTP, model in `/dev/shm`, the same build (`TS_Q4E_GROUP_F16=0` dequantizes per ubatch). TTFT of three
runs, the first cold. The engine bench's short_chat, long_prompt_4k/16k and long_generation checks pass 8/8 in both
arms:

| dense weights in a prompt group | 4K prompt (3,541 tokens) | 16K prompt (14,093 tokens) | engine bench 4K / 16K TTFT | long generation |
| --- | --- | --- | --- | --- |
| dequantized per ubatch | 0.85 / 0.56 / 0.55 s | 2.39 / 2.38 / 2.38 s | 0.74 s / 3.00-3.01 s | 241-244 tok/s |
| once per layer and group (default) | 0.80 / 0.53 / 0.51 s | 2.28 / 2.27 / 2.27 s | 0.70 s / 2.85-2.86 s | 237-252 tok/s |

#### The routed experts' gate/up once per prompt group, 2026-11-09

On the int8 kernels, a 1,024-row ubatch gives an expert about 20 tokens, and every ubatch read every expert's weights
again. At 16K, that re-reading alone is about 476 GB, 264 ms at the RTX 5090's bandwidth before any arithmetic. One layer's
gate/up takes 0.52 ms over 1,024 rows and 1.27 ms over 4,096 (`CudaMoeQ2_0Int8Tests` timing). Strata reads its prompts in
8K-token chunks.

Within a prompt group the next layer starts only after all the group's ubatches have run this one. So a group's ubatches
now leave their routed experts to the end of the layer (`TS_Q4E_GROUP_EXPERTS`, on):

- Each ubatch stages its missed experts (kept for the layer) and runs its router and shared expert.
- It leaves its selections, int8 activations, shared-expert outputs and scatter logits in the group's buffers.
- After the last ubatch, one grouping plan and one gate/up run over the whole group, so each expert's gate and up
  weights are read once per layer and group.
- Down stays per ubatch, so its per-slot outputs keep their size. Each ubatch's down reads its slots' H through the
  group's plan (`ts_dsv4_moe_row_map_i32`, and the down kernel's act-row map). The combine and the scatter into the
  streams follow.

Every slot's products are ubatch order's, so a prompt answers the same bit for bit
(`APromptGroupsExpertGateUp_RunOncePerLayer_AnswersBitForBit`: 64-row ubatches in groups of 192 rows, Q2_0 experts).

The buffers take 85 MiB at 4,096 rows, mostly the shared-expert outputs and H, so 64 experts fewer stay in VRAM
(17,485 -> 17,421). A group whose last ubatch has
16 rows or fewer keeps ubatch order, because that ubatch runs the per-slot kernels. So does any layer the int8 route does
not take.

RTX 5090, ISTA `Q2_0`, MTP, model in `/dev/shm`, the same build (`TS_Q4E_GROUP_EXPERTS=0` runs gate/up per ubatch). TTFT
of three runs, the first cold. The engine bench's short_chat, long_prompt_4k/16k and long_generation
checks pass 8/8 in both arms:

| routed experts' gate/up in a prompt group | 4K prompt (3,541 tokens) | 16K prompt (14,092 tokens) | engine bench 4K / 16K TTFT | long generation |
| --- | --- | --- | --- | --- |
| per ubatch | 0.81 / 0.53 / 0.53 s | 2.28 / 2.27 / 2.27 s | 0.70 s / 2.86 s | 240-251 tok/s |
| once per layer and group (default) | 0.79 / 0.51 / 0.49 s | 2.23 / 2.21 / 2.21 s | 0.67 s / 2.74-2.76 s | 243-254 tok/s |
| Strata (11-07) | | | 0.92 s / 2.75 s | 300-307 tok/s |

#### A keyed router select, 2026-11-09

A decode step's router is a 512 x 2560 BF16 GEMV, then the top-10 select over the 512 softmax scores
(`ts_dsv4_moe_select_f32`). The select runs one warp a row. Each pick was a shuffle argmax that carried score, weight
and expert through three shuffles a step, and the winning lane then rescanned its 16 scores in order. In the captured
decode profile the select took 5.8 us a layer, more than the GEMV before it (5.1 us at 4 rows).

The keyed select (`ts_dsv4_moe_select_keys_f32`, `TS_DSV4_SELECT_KEYS`, on) turns each ranking score into one 64-bit
key: the score's order-preserving bits above the expert's complement. The larger key is the higher score, then the lower
expert, which is the argmax's order.

- A pick is a key max over two shuffles a step.
- The winning lane retakes its best as a four-step tree.
- Lane k keeps pick k.
- The weights are read back from shared memory and summed in pick order. Each lane writes its own pick. Before, lane 0
  stored each pick and read the weights back from global memory to scale them.

The picks and weights are the same bits as the float argmax's. `Dsv4RouterKernelTests` holds the two kernels to each
other on 2,048 rows built for the order's corner cases: exact ties at every rank, -inf, NaN, +inf, signed zeros, and
rows with no candidate. The rows cover 64-512 experts, top 6-32, all three gatings, with and without a bias. It also
holds both kernels to a host reference.

RTX 5090, a graph of 192 launches, 48 cold router matrices (us a call):

| rows | BF16 GEMV | select, float argmax | select, keyed |
| --- | --- | --- | --- |
| 1 | 2.78 | 4.83 | 3.11 |
| 4 | 4.22 | 4.85 | 3.11 |

Engine bench, ISTA `Q2_0`, MTP, one build (`TS_DSV4_SELECT_KEYS=0` for the argmax). Two passes per arm, alternating.
Every greedy output is identical across arms, and every check passes:

| select | long generation | 4-row verify window | 1-row step | 4K / 16K verify window |
| --- | --- | --- | --- | --- |
| float argmax | 231-244 tok/s | 10.09-10.10 ms | 7.01-7.03 ms | 13.37 / 10.63 ms |
| keyed (default) | 232-246 tok/s | 10.02-10.04 ms | 6.93-6.94 ms | 13.28-13.33 / 10.56-10.58 ms |

Tried and slower: a GEMV block computing 2 or 4 experts, with each thread loading its activations once for all of
them. At 4 rows it took 4.75 and 6.45 us against 4.22, because the smaller grid kept fewer weight reads in flight.

#### Row sums of the dense kernels as one reduce-scatter, 2026-11-09

Each decode-once dense kernel ends with a warp sum per activation row: five shuffles each, so 20 for a 4-row window,
against a main loop of one or two weight units per lane at Flash Next's 2560-wide rows. A window of 4 rows or more now
reduces its rows as one reduce-scatter (`ts_dsv4_rows_reduce_scatter`). At each step of the warp sum's xor butterfly
(16, 8, 4, 2, 1), a lane keeps half of its rows and adds its partner's values for them: 6 shuffles for 4 rows. Every add
is the butterfly's own (the lane's partial plus its partner's, the same partners in the same order), so each row's sum is
the same bits as before, at every width.

Below 4 rows the kernels keep a warp sum each: there the reduce-scatter took them from 64-80 registers to 128 and slowed
the 2- and 3-row calls. Done the same way, the router's block GEMV gained nothing (4.21 -> 4.20 us at 4 rows), so it
keeps its sums.

The dense fingerprints (`Dsv4DenseFastFingerprintTests`), the width tests and the row-invariance tests are unchanged.
RTX 5090, 4 rows, cold weights (us): qkv IQ4_XS 2560 -> 10240 11.15 -> 10.65; attention q Q2_0 2560 -> 12288
11.91 -> 11.30; ssm_out Q5_K 6144 -> 2560 12.61 -> 12.16; hyper-connection up Q8_0 320 -> 10240 5.10 -> 4.48; down
10240 -> 320 5.27 -> 5.36. 1-3 rows unchanged.

Engine bench, two builds, two alternating passes per arm. Every greedy output is identical across arms, and every check
passes:

| row sums | long generation | 4-row verify window | 4K / 16K 4-row window |
| --- | --- | --- | --- |
| a warp sum per row | 232-245 tok/s | 10.02-10.03 ms | 13.36 / 11.05-11.07 ms |
| one reduce-scatter (default) | 234-248 tok/s | 9.89-9.94 ms | 13.28-13.30 / 10.93-10.96 ms |

#### The MTP catch-up reads the prompt's streams on the device, 2026-11-09

After a long prompt, the first draft window runs the MTP block over the prompt's last 4,000 rows. Only their K/V are
written: the KV-only catch-up. On the RTX 5090 that took 22.6 ms, about 8% of a 4K answer's decode time, and only
4.6 ms of it was kernels. The rest was the rows' trunk streams, 4,000 x 10,240 floats (164 MB). They came back from the
runtime's host copy: each ubatch was copied into pinned memory, then host to device. The same values were still on the
device.

The last prompt group's output streams stay in the group's buffer until the next group runs. The engine now records
which positions of which sequence that buffer holds. The catch-up takes those rows' streams device to device, and only
the rest from the host (`TS_Q4E_MTP_D2D`, on).

The values are the ones the host's copy holds, so the drafts are the same bits. In
`TheMtpCatchUpAfterAGroupedPrompt_ReadsTheLastGroupsStreams_BitForBit` (a 300-token prompt in 192-row groups, catch-up
from position 150), the last group's 108 rows come from the device. The chain picks the same tokens with the same
confidences.

Engine bench, one build, two alternating passes per arm (decode tok/s of the three requests). Every greedy output is
identical across arms, and every check passes:

| catch-up rows | 4,000-row catch-up | long_prompt_4k | long_prompt_16k |
| --- | --- | --- | --- |
| from the host (`TS_Q4E_MTP_D2D=0`) | 21.6-22.7 ms | 177 / 186 / 206, 170 / 185 / 241 | 206 / 230 / 257, 209 / 234 / 256 |
| from the device (default) | 5.8-6.1 ms | 179 / 198 / 219, 176 / 201 / 255 | 208 / 257 / 277, 211 / 253 / 278 |

The prompt forward still copies the streams to the host (164 MB D2H) for the runtime, so TTFT is unchanged.

### DFlash2 drafter on ggml_metal, 2026-11-08

Strata drafts with DFlash2 on Macs: a block drafter that proposes up to 7 tokens per pass. TensorSharp now takes the
same drafter on the GGML backends: pass the official GGUF of
[incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2-GGUF) as
`--draft-model`. Strata's Mac pack is the same stock weights.

```bash
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --backend ggml_metal \
  --draft-model Qwen3.8-Flash-Next-DFlash2-Q4_K_M.gguf
```

- **Features.** The drafter was trained on the completed outputs of layers 5, 15, 25, 35 and 45, each averaged
  over the four hyper-connection streams (12,800 floats a position). The token span computes those means inside its
  own graph and writes them into the speculative forward's feature rows, so a verify, a plain step and a prompt
  chunk all feed the drafter's ring. The drafter borrows the target's token embedding and LM head.
- **Defaults.** Windows of up to 6 drafts, ended where the product of the drafter's confidences falls below 0.5
  (Strata uses 0.7; on this Mac 0.5 decoded the code and prose answers at 47.3 / 30.8 tok/s against 46.4 / 28.0).
  `--spec-draft` and `--spec-pmin` override them. Not under `--tp` or distributed stages.
- **The prompt now reaches the drafter.** A block drafter used to be armed after a PLAIN prefill (a shortcut meant
  for drafters that keep no context), so its ring held only the answer's own tokens. Prompts now prefill through the
  speculative context (`ISpeculator.ConditionsOnPrompt`): short answers 13 -> 18 tok/s, 4K-prompt answers
  30-39 -> 47-51. This applies to every DFlash target (Qwen 3.5/3.8, Muse-Glimmer).
- **Verify graphs per width.** A token that runs as one span keeps one persisted graph per width (1..8 rows); one
  shared graph was rebuilt whenever a window's width changed. Graphs that captured into a row buffer a wider window
  has since grown are rebuilt, not replayed.
- **Per-row states in place.** A verify window's GatedDeltaNet output, every row's state included, is allocated in
  the buffer the commit reads, which removes a copy of about 3 MB a row per layer (7 ms of GPU time in a 4-row
  window on the Mac).
- **Prompts stage tier experts from their slots.** A prompt-width batch streams every expert an offloaded layer
  routes into a staging tensor. The experts the layer's tier already holds are now copied from their slots in
  memory instead of read from the file again: 4K TTFT 12.7 -> 8.9-10.2 s, 16K 62-67 -> 47-48 s. Same bytes, same
  answers.

Mac M5 Pro 48 GB, ISTA `Q2_0`, `ggml_metal`, 32K context, `TS_Q4E_HC_Q8=1`, tiered experts, the same client as the
Strata comparison (3 repeats; long generation is a code answer, a prose answer and a passage copy):

| run | short chat TTFT / decode | long-gen decode (code / prose answers) | 4K TTFT / decode | 16K TTFT / decode | pass |
| --- | --- | --- | --- | --- | --- |
| TensorSharp MTP, before | 0.43 s / 18 | 32.2 | 16.8 s / 38 | 65.3 s / 33 | 15/15 |
| TensorSharp DFlash2 Q4_K_M | 0.32-0.35 s / 15-18 | 32.8-33.4 (50.1-52.7 / 32.8-33.5) | 8.9-10.2 s / 35-51 | 46.7-47.9 s / 37-45 | 15/15 |
| Strata DFlash2 (`--spec 2`, streamed experts) | 0.56 s / 19 | 47.8 (53.7-55.2 / 47.6-48.0) | 6.32 s / 61 | 24.4 s / 61 | 15/15 |

What did not help, measured and removed: every expert mapped on Metal (`--n-cpu-moe 0`; 36-63 tok/s but 34 GB of
experts plus a 4K prompt's buffers ran Metal out of memory on the 48 GB Mac); dense IQ4_XS rewritten as IQ4_NL for
ggml-metal's multi-row kernel; the speculation governor off. What remains is the verify window itself: 4 rows cost
about 70 ms of which 63 ms on the GPU, against 39 ms for one row; Strata spends about 55 ms a 3.6-row step.

Windows of more than 4 rows are verified as one batched forward on Metal (see "Verify rows run the one-token
kernels"), so DFlash text can differ from plain greedy decoding; `TS_Q4E_VERIFY_ROW_KERNELS=1` keeps it exact.

#### TensorSharp-owned Metal expert kernels, 2026-11-08

A node profile of the token span on the Mac (`TS_GGML_NODE_PROFILE=types`, every expert mapped, the per-node sync
subtracted) put the routed experts first: ggml-metal's MUL_MAT_ID reads each (row, route) pair's weights on its own,
6.9 ms of GPU time for a 1-row step (89 GB/s) and 16.2 ms for a 4-row verify window. The tiered layers now compute
the routes their slots hold with Strata's grouped expert kernels (MIT, ported unchanged into
`TensorSharp.GGML.Native/tsm/`): each distinct slot's rows are read once for every row of the window that routes to
it. The seam builds the grouping plan from the routes it already reads and encodes the kernels on ggml-metal's own
command queue, so they run in order between the two segments with no host round trip of their own. They agree with
MUL_MAT_ID to 1e-6 and with a double-precision reference to 2e-7 (`TsmMoeKernelTests`). `TS_METAL_MOE=0` keeps
MUL_MAT_ID.

M5 Pro, ISTA `Q2_0`, tiered: the experts of a 4-row window take 0.2 ms of GPU time a layer (0.08 ms at 1 row); a
4-row verify pass 69.9 -> 59.8 ms, text identical. The same client as above, DFlash2 Q4_K_M, 15/15:

| run | short chat decode | long-gen decode (code / prose answers) | 4K TTFT / decode | 16K TTFT / decode |
| --- | --- | --- | --- | --- |
| DFlash2, MUL_MAT_ID | 15-18 | 32.8-33.4 (50.1-52.7 / 32.8-33.5) | 8.9-10.2 s / 35-51 | 46.7-47.9 s / 37-45 |
| DFlash2, grouped kernels | 15.6-19.0 | 34.8-36.1 (53.8-57.0 / 34.9-36.2) | 9.1-10.3 s / 45-49 | 46-48 s / 29-45 |
| Strata DFlash2 | 19 | 47.8 (53.7-55.2 / 47.6-48.0) | 6.32 s / 61 | 24.4 s / 61 |

This is the first piece of a TensorSharp-owned Metal decode window (Strata's design: a whole window in one command
buffer, weights read in place, the host's turn at each layer through a shared event); the rest of the span still runs
on ggml-metal.

#### Seams on a shared event, and a GPU keep-alive, 2026-11-08

A tiered pass used to end each of its 48 segments with a submit and a wait: the GPU idled while the host woke, read
the routes, encoded the held routes and the next segment, and committed them. On Metal the seams now hand over on a
shared event, as Strata's host turns do (`run_graph_seg`), without patching ggml: the segments are enqueued two ahead
on ggml-metal's own queue (`ggml_backend_graph_compute_async`), and each seam is a TensorSharp command buffer between
two of them - the prepared input (hazard tracking orders it after the segment that wrote the routes), a signal, a wait
for the host's release, the grouped expert kernels with their grids read from the plan, and the combine whose output
the next segment reads. The host spins until the signal, reads the routes, computes the ones no slot holds, writes
the plan and releases the GPU. The kernels, their order and their buffers are those of a submit-and-wait seam, so the
logits are the same bits (`TsmSeams_OnASharedEvent_AnswerAsSubmitAndWait_Metal`, with every route held and with half
the experts left to the host). `TS_METAL_SEAM_EVENTS=0` keeps a submit and a wait per seam.

A handshake still costs a round trip: on an M5 Pro ~190 us of GPU time on a quiet GPU. While seamed passes run, short
spin kernels on a second queue keep the GPU awake (Strata's keep-alive): the round trip falls to ~83 us and its p90 from
880 to 475 us. `TS_METAL_KEEPALIVE=0` turns it off. `TS_METAL_SEAM_PROFILE=1` prints, on the GPU's clock, the segments'
time and the seams' (their kernels and the wait for the host); `TS_HOST_MOE_TIMING=3` now also prints the tier's routes
held, window reads and host routes per pass.

M5 Pro, ISTA `Q2_0`, tiered, per pass (`TS_HOST_MOE_TIMING=3`):

| pass | submit and wait | event seams | + keep-alive | Strata window (`STRATA_METAL_PROFILE=1`) |
| --- | --- | --- | --- | --- |
| 1 row | 37.7-38.9 ms | 37.0-38.3 ms | 32.8-33.9 ms | 28.1 ms |
| 4 rows | 59.0-64.5 ms | 56.7-62.1 ms | 52.8-59.0 ms | 45.7-46.2 ms |

At one row the GPU spends 21.9 ms in ggml's segments and 8.6-9.7 ms in the seams, of which 3.7 ms is the expert
kernels; the host's turns take 3.2-4.3 ms, mostly 9-13 window reads of ~1.3 MB each. At four rows the turns take 8-14 ms
for 26-47 window reads and up to 13 routes the window could not hold. Plain decoding of a 512-token answer took 17.3-17.6
s against 20.0 s. Splitting a window read into 64 or 128 KB requests was slower (more requests, more wake-ups) and was
not kept. Strata's default `STRATA_ROUTE_TAIL_SKIP=7` skips a missed expert that every row of a window routes at rank 7
or lower; TensorSharp computes every route.

The same client as above, DFlash2 Q4_K_M, 15/15 (`df2-seams`):

| run | short chat decode | long-gen decode (code / prose answers) | 4K TTFT / decode | 16K TTFT / decode |
| --- | --- | --- | --- | --- |
| DFlash2, grouped kernels | 15.6-19.0 | 34.8-36.1 (53.8-57.0 / 34.9-36.2) | 9.1-10.3 s / 45-49 | 46-48 s / 29-45 |
| DFlash2, event seams + keep-alive | 18.8-19.8 | 37.8-39.5 (56.6-59.6 / 37.9-39.6) | 8.9-10.0 s / 42-56 | 41.6-47.3 s / 38-48 |
| Strata DFlash2 | 19 | 47.8 (53.7-55.2 / 47.6-48.0) | 6.32 s / 61 | 24.4 s / 61 |

Short answers and code now match or pass Strata. Prose is behind at the same acceptance (298-305 of 440-478 drafts
against Strata's 314-318 of 469-501): a TensorSharp verify window of ~3.9 rows costs ~58 ms where Strata's 4-row
window costs 46 ms, and the drafter declines to draft for 38-59 of a prose answer's steps.

#### Long-context QSA and the expert window, 2026-11-08

After a 16K prompt a 1-row pass spent 30.0 ms of GPU time in ggml's segments against 21.9 ms at a short context. The
QSA layers explained all of it (timed by switching parts off): re-pooling every block's indexer key each step (3.8 ms),
the cell-level selection (1.1 ms), and attention over every cell of the window under the selection mask. Three changes,
all with upstream ggml ops:

- Each QSA layer keeps its pooled keys between steps (`Q4eSeqStateEntry::aux`), and a step pools only the blocks its
  cells complete. Block membership is fixed by position for text. A rollback leaves a block incomplete until it is
  re-pooled, and any other writer of the raw keys (an upload, the concurrent-decode arena) makes the next step pool
  everything. Same per-block arithmetic, so the logits are the same bits
  (`QsaPooledKeys_KeptBetweenSteps_AnswerAsPoolingEveryBlock_Cpu/_Metal`). `TS_Q4E_QSA_POOL_CACHE=0` pools every block.
- The selection mask comes from `ggml_top_k_mask` (one op for what was a fill, a scatter and an add over the whole
  window), and decode steps and verify windows attend only the selected cells (`ggml_flash_attn_ext_set_n_kv_max`).
  `TS_Q4E_QSA_SPARSE_FA=2` also does it for prompts; that was slower on an M5 Pro.
- On unified memory 12% of the tier budget goes to window slots instead of profile-ranked fixed slots
  (`TS_MOE_WINDOW_SHARE`). The window follows the text; the fixed slots follow other prompts.

| pass (M5 Pro, ISTA `Q2_0`, tiered) | before | after |
| --- | --- | --- |
| 1 row after a 16K prompt, ggml segments (GPU) | 30.0 ms | 24.7 ms |
| 4 rows, short context (window 0.3 -> 3.1 GB) | 52.8-59.0 ms (26-47 window reads) | 49.4-51.0 ms (14-19) |

The same client, DFlash2 Q4_K_M, 15/15 (`df2-qsa`):

| run | short chat decode | long-gen decode (code / prose answers) | 4K TTFT / decode | 16K TTFT / decode |
| --- | --- | --- | --- | --- |
| DFlash2, event seams + keep-alive | 18.8-19.8 | 37.8-39.5 (56.6-59.6 / 37.9-39.6) | 8.9-10.0 s / 42-56 | 41.6-47.3 s / 38-48 |
| + QSA pooled keys, sparse decode attention, window share | 22.6-24.8 | 42.1-43.4 (63.1-69.7 / 42.2-43.5) | 8.9-9.6 s / 50-71 | 42.9-45.4 s / 54-66 |
| Strata DFlash2 | 19 | 47.8 (53.7-55.2 / 47.6-48.0) | 6.32 s / 61 | 24.4 s / 61 |

Short answers, code and long-context decoding now match or pass Strata; prose is ~10% behind. Prompt reading is what
remains: a 4096-token chunk takes 8.2 s of compute at a short context (9.9 s at 16K), of which ~1 s is staging every
expert of each layer before the GPU may compute it (`TS_HOST_MOE_TIMING=1`).

#### Route tail skip (approximate, on by default), 2026-11-08

Strata skips, by default, a missed expert that every row of a decode or verify window routes only at rank 7 or lower
(`STRATA_ROUTE_TAIL_SKIP=7`). TensorSharp does the same now, by the user's decision, in the accelerator expert tiers
(ggml_metal and ggml_cuda): such an expert is neither read into a window slot nor computed on the host, and its routes
contribute nothing (`TS_MOE_ROUTE_TAIL_SKIP`, `0` computes every route). It stays off at prompt widths, when the tier
holds every expert, or 90% or more of them (`TS_MOE_TAIL_SKIP_MAX_CACHED`). Answers become approximate: Strata
measured KL 0.003-0.006 against the exact run and ~98% top-1 agreement on 2-3-bit packs. Parity tests that assert
exact answers set `TS_MOE_ROUTE_TAIL_SKIP=0`; `RouteTailSkip_SkipsOnlyMissedLowRankRoutesAtDecode_Metal` checks that
only decode steps skip, that the prompt keeps its exact bits and that a tier holding every expert skips nothing.

M5 Pro, ISTA `Q2_0`, DFlash2, two interleaved off/on pairs: window reads per decode pass 88-114 -> 33, short chat
22.4-24.6 -> 23.5-25.9 tok/s, long generation 39.8-44.2 (mean 42.1) -> 42.1-44.3 (mean 43.4) tok/s. A decode pass
spends ~26 of its ~31 ms in the accelerator's segments, so fewer reads move it only a little.

#### Prompt reading: staged experts ahead, PLE rows read explicitly, 2026-11-08

`TS_Q4E_PHASE=1` splits a prompt chunk's wall clock. Two host costs stood in front of the GPU:

- Every offloaded layer staged the experts it routes (tier copies and file reads) before its MoE graph could run, ~20 ms
  a layer. A prompt routes nearly every expert, so the next offloaded layer's stacks are now staged whole, on a thread,
  while the GPU runs the segment before it, and its MoE graph binds them (`TS_HOST_MOE_STAGE_PREFETCH`). Staging on the
  critical path fell from ~600 to 24 ms per 30 layers. One buffer, reused once the previous layer has computed: two ran a
  48 GB Mac's Metal out of memory on a 16K prompt.
- The PLE n-gram rows (16 a token from the 29 GB table) were faulted in through the mapping: 1.1-3.1 s for a 4096-token
  chunk's 65K rows. A prompt-sized gather now reads them with one `pread` each on the native I/O team, 0.52-0.54 s a
  chunk, the rate Strata reports (`TS_PLE_EXPLICIT_READ`).

Both stage the same bytes, so the logits are the same bits (`PromptExpertPrefetch_AnswersAsStagingEachLayer_Metal`,
`PleRows_ReadExplicitly_AnswerAsTheMapping_Metal`). A block-first QSA selection (the best ~515 blocks, then their cells)
was tried and dropped: no measurable gain, and it changes cutoff ties that the batched-arena parity tests pin.

| run (same client, DFlash2 Q4_K_M, 3 repeats) | 4K TTFT | 16K TTFT |
| --- | --- | --- |
| QSA pooled keys (`df2-qsa`) | 8.9-9.6 s | 42.9-45.4 s |
| + experts staged ahead, explicit PLE reads (`df2-pf1`) | 8.46-8.81 s | 38.0-38.3 s |
| Strata DFlash2 | 6.32 s | 24.4 s |

What remains is the chunk's compute, ~7.5 s per 4096 tokens: the routed experts (~1.5 s), ggml's gated delta net
(~0.9 s), the per-token element-wise and layout work of the hyper-connections and the GDN block (~3 s: a [T, 10240]
transpose, the conv input concat, q/k/v copies, the q/k head repeat), and the dense projections (~1.6 s). Strata runs
those as fused prompt kernels (`gdn_prompt.metal`: conv + norms + gates, a column-parallel state scan, the gated norm).

#### Prompt kernels: the GDN recurrence and the hyper-connection passes, 2026-11-08

A prompt-width span (64+ rows, `TS_METAL_PROMPT_OPS_MIN_ROWS`) on `ggml_metal` runs two kinds of work on TensorSharp's
own Metal kernels (`tsm/`), encoded on ggml-metal's queue between graph segments:

- The GDN recurrence (`tsm_gdn_prompt.metal`, ported from Strata's `gdn_prompt.metal`): conv + SiLU + q/k L2 norms +
  gates in one pass, a column-parallel state scan (a threadgroup keeps 32 columns of a head's state in registers), and
  the gated RMS norm. It replaces ggml's `[T, conv_dim]` transpose, the conv-input concat, `ssm_conv`, the q/k/v
  copies, the q/k head repeat, the gated delta net and the norm. The conv history and the state are updated in place in
  ggml's layout, so decode, which runs ggml's ops, continues from them (`TS_METAL_GDN_PROMPT`).
- The hyper-connection passes (`tsm_hc_prompt.metal`, after Strata's `gr_epi_up`): every half-layer's gated stream mean
  after the up projection and its scatter `res + b * 2 sigmoid(inject / hc)`, one pass each. ggml ran chains over
  `[hc * n, T]` for these: sigmoid, product, a copy and three adds, scale; then scale, sigmoid, a repeat, product, add
  (`TS_METAL_HC_PROMPT`).

The span cuts its graph at each kernel, as the device operations of the host-MoE segment executor. The kernels'
operands live in a per-slot IO buffer: the graph allocator frees a tensor after its last graph consumer, or never when
it has none, so leaving them in the graph would either hand their memory to later nodes or keep every layer's copy
alive. The residual stays in that buffer for the whole span, updated in place by each scatter. A build check fails the
span if a node reads a kernel's output before the kernel runs.

The span's graph arena is also kept across a prompt's chunk rebuilds. A new Metal buffer costs ~75 ms a GB to map and
make resident, and the window grows every chunk (3.1 -> 3.5 -> 4.2 GB at 16K), so each rebuild paid 100-400 ms.

Parity (`PromptKernels_OnTsm_AnswerAsGgml_Metal`: a 160-token prompt and six decode steps on the synthetic fixture):
GDN 6.4e-5, HC 5.3e-5, both 6.9e-5 relative error, every argmax equal; 3.0e-4 with every expert on the host, where
the Q8_K activations amplify the difference.

| 16K prompt, `TS_Q4E_PHASE=1` (M5 Pro) | compute per 4096-token chunk | prompt + 50-70 tokens |
| --- | --- | --- |
| ggml's ops | 7.6 / 8.3 / 9.0 / 9.5 s | 44.3 s |
| + GDN kernels | 6.3 / 6.8 / 7.4 / 8.0 s | 36.9 s |
| + hyper-connection kernels | 5.2 / 5.7 / 6.3 / 6.9 s | 33.1 s |

| run (same client, DFlash2 Q4_K_M, 3 repeats) | short chat | long-gen decode | 4K TTFT | 16K TTFT | multi-turn first turn |
| --- | --- | --- | --- | --- | --- |
| before (`df2-pf1`) | 0.30 s / 15-25 tok/s | 41-43 tok/s | 8.46-8.81 s | 38.0-38.3 s | 5.7-6.2 s |
| prompt kernels + kept arena (`df2-pk`) | 0.30-0.33 s / 22-25 tok/s | 42.7-43.3 tok/s | **6.04-6.44 s** | 27.2-28.1 s | 4.1-4.6 s |
| Strata DFlash2 | 0.56 s / 19 tok/s | 47.8 tok/s | 6.32 s | 24.4 s | - |

A 4K prompt now reads as fast as Strata's. At 16K each chunk still costs ~0.55 s more per 4K of window than the one
before: the QSA layers gather every cell's block score into `[cells, T]`, sort each row for its top 2051 cells, and
attend the whole window densely.

The selection now runs on TensorSharp's kernels as well (`tsm_qsa_prompt.metal`, `TS_METAL_QSA_SELECT`): the heads'
relu score sum and bias per block, then a per-token radix select of the best 2051 allowed cells, written straight into
the attention mask. Cells are compared exactly as ggml computes them, so the kept cells are the same except where a
block's cells tie at the cutoff (the window keeps 2048 + 3 cells, so most rows split a block): ggml's top-k fills ties
in atomic order, the kernels lowest index first, which also makes the choice reproducible
(`QsaSelection_KeepsTheBestCells_TiesLowestFirst` checks the contract cell by cell). The span's arena no longer grows
with the window (3.4 GB at every chunk of a 16K prompt). 16K, `TS_Q4E_PHASE=1`: chunk compute 5.2 / 5.7 / 6.3 / 6.9 s
-> 5.1 / 5.3 / 5.8 / 6.3 s, prompt + ~50 tokens 32.3 -> 30.4 s. What grows with the window now is the attention
itself, which reads every cell of it.

From a 12K window on, the attention reads only each token's chosen cells too (`TS_METAL_QSA_ATTN`): the selection
writes them as ascending lists, and one threadgroup per (token, kv head) gathers them 32 at a time, with the kv head's
12 query heads as the rows of the products (Strata's `qsa_attn_mm` arrangement, on F16 caches). Against ggml's flash
attention over the same cells it differs by rounding (4.7e-5). Its cost does not depend on the window, ggml's does:
+0.40 s a chunk at a 4K window, +0.20 s at 8K, -0.23 s at 12K, -0.62 s at 16K, hence the 12K threshold
(`TS_METAL_QSA_ATTN_MIN_CELLS`).

That first kernel computed with simdgroup matrices and ran ~3x below ggml's flash attention per unit of work: on an M5,
ggml's attention runs on the Metal 4 tensor API (the matrix units). The kernel was rewritten on that API too
(`tsm_qsa_attn_mm.metal`, `TS_METAL_QSA_ATTN_MM`): the structure of ggml's tensor flash attention (`matmul2d` for Q.K^T
and P.V over four simdgroups, lazy rescaling, O in a cooperative tensor) with the chosen cells' keys and values
gathered into threadgroup memory. It is compiled as its own library at Metal language 4.0, on M5-class devices only
(ggml's rule), so a device or SDK without the API keeps every other kernel and falls back to the simdgroup one. A
probe at Flash Next's shapes (`QsaAttention_FlashNextShape_Timing`: 4096 tokens, 24 query heads over 2 kv heads, a 16K
window) takes 42 ms a layer instead of 79; it answers as a double-precision reference to 1.9e-4. Against ggml's flash
attention per chunk: +0.18 s at a 4K window, -0.20 s at 8K, -0.57 s at 12K, -0.97 s at 16K, so the threshold is 8K
there. 16K smoke: chunks 5.06 / 5.24 / 5.29 / 5.34 s instead of 5.06 / 5.44 / 5.86 / 6.31 s with ggml's attention.

| run (same client, DFlash2 Q4_K_M, 3 repeats) | short chat | long-gen decode | 4K TTFT | 16K TTFT | multi-turn first turn |
| --- | --- | --- | --- | --- | --- |
| + QSA selection and attention kernels (`df2-sa`) | 0.30-0.32 s / 21-25 tok/s | 38.4-42.7 tok/s | **5.92-6.14 s** | **24.48-25.25 s** | 4.0-4.5 s |
| + attention on the tensor API, route tail skip (`df2-mm`) | 0.30-0.32 s / 20.5-25.8 tok/s | 43.0-45.0 tok/s | **5.97-6.22 s** | **23.91-24.23 s** | 4.0-4.5 s |
| Strata DFlash2 | 0.56 s / 19 tok/s | 47.8 tok/s | 6.32 s | 24.4 s | - |

### Concurrent requests on ggml_cuda, 2026-11-09

Requests at once on `--backend ggml_cuda` (RTX 5090, 32 GB; ISTA Q2_0 with the MTP head, MAX_CONTEXT 131072) failed:

- Every concurrent request's cache holder was the whole window: 3.6 GB of K/V and indexer keys.
- The 19.1 GB expert tier left about 4.6 GB, so the second request's state took the GPU's last memory.
- Its token span then could not allocate, and every concurrent request answered HTTP 500.
- In an earlier run, the fourth request's per-slot cuBLAS handle could not be created, and ggml aborted the server.

Now:

- A concurrent request's holder starts at 2048 rows and grows with its request, as on Metal.
- A request whose state does not fit beside a 1 GB headroom waits for a running request (the scheduler defers it).
  Its state is counted at 8192 rows plus its recurrent state, about 0.5 GB at F16. The prefix cache evicts retained
  conversations first.
- `TS_Q4E_LANES` has the expert tier keep that room for that many sequences. With 4 lanes the reserve is 1.5 GB: the
  tier goes from 19.1 to 17.7 GB, and from 94.2% to about 92% of the profiled routes.
- A per-request backend that cannot get its cuBLAS handle fails that request, not the process.

Same client as above. The engine bench runs 256-token decodes twice; standard-v1 concurrency is three rounds of four.

| run | long-gen decode | parallel_2 | parallel_4 | standard-v1 concurrency |
| --- | --- | --- | --- | --- |
| before | 85.6-94.2 tok/s | HTTP 500 | HTTP 500 | 12 of 12 failed |
| 1 lane | 80.7-98.1 tok/s | 59-65 tok/s, 2/2 correct | 92-95 tok/s, 4/4 | 12 of 12 completed, 31-37 tok/s each |
| 4 lanes | 79.0-88.6 tok/s | 66-75 tok/s, 2/2 | 91-94 tok/s, 4/4 | 12 of 12 completed, 30-36 tok/s each |

Those requests still waited ~5 s for their first tokens, and a lone 73-token prompt waited 1.1-1.4 s. A prompt span of
65-127 rows was wider than a discrete GPU's tier splits (64) and narrower than the seam streams (128). Every route of
every layer ran on the host: 1.1-1.4 s of compute against 0.27-0.29 s at 64 rows. The tier now splits up to the
streaming width (see Expert tiers above).

After that, at 1 lane following four concurrent requests, a 4K prompt failed. The draft head's catch-up over its
1,045-row chunk asked 1.29 GB of a GPU with ~1.4 GB free. A catch-up whose graph does not fit now goes in halves, as
the target's prompt spans do (that run halved it 1.32 GB -> 660 -> 330 -> 165 MB).

| run, both fixes | long-gen TTFT / decode | standard-v1 concurrency TTFT | parallel_2 | parallel_4 | 4K / 16K TTFT |
| --- | --- | --- | --- | --- | --- |
| 1 lane | 0.20-0.23 s / 90-97 tok/s | 0.84-1.47 s | 91-106 tok/s | 93-128 tok/s | 5.8 / 21-27 s (after the concurrency rounds) |
| 4 lanes | 0.20-0.25 s / 90-92 tok/s | 1.20-1.72 s | 90-104 tok/s | 122-126 tok/s | 5.8 / 22.1 s |

No request failed in either arm.

Greedy identity holds only in an exact arm that runs each sequence through its own one-row span:

- The exact arm turns off the prefix cache, the route tail skip, tier adaptation and hybrid misses
  (`TS_SCHED_PREFIX_CACHE=0 TS_MOE_ROUTE_TAIL_SKIP=0 TS_MOE_TIER_ADAPT=0 TS_Q4E_HYBRID_MISSES=0`).
- With `TS_Q4E_DISABLE_ARENA_DECODE=1` added, four requests at once give the same text as alone, 4 of 4, and so do
  repeats of each alone, 4 of 4. That arm does a parallel_4 wave in 18.5 s, against 14 s with the batched arena.
- With the batched arena, parallel matches solo 0 to 1 of 4, with or without speculation. ggml picks its matrix kernels
  by row count, so a batched step rounds differently from a lone request's, and greedy text can part at a near tie.
- Under the default settings solo repeats are not identical either (1 of 4), as on the direct engine.

The ggml path reads its host experts and the PLE n-gram table through the page cache. On this pod's network volume,
which drops a file's page cache whenever it is opened, ggml_cuda decoded at 1.8-4.7 tok/s. The runs above read the PLE
shard from RAM (`/dev/shm`).

## Multi-GPU

`--tp N` on `ggml_cuda` partitions every routed and shared FFN across N local GPUs. Every rank
retains all expert IDs. Gate/up projections produce local intermediate channels,
which are gathered before down projections compute disjoint output rows. A second
gather assembles those rows before the hyper-connection scatter. Keeping each
down dot product at its original full width avoids summation-order changes that
later activation quantization can amplify. Attention, GDN, QSA and PLE run with replicated weights
and independent state on each rank. The output head runs on rank 0. The image
and tool-call paths use the same target graph; speculative rollback restores
GDN and PLE state on every rank. Prefix checkpoints remain disabled under TP.
Two or more ready decoders share one batched step on every rank (the fused
arena, see "Concurrent decode uses the fused arena by default"), so their
layers' gathers are paid once a step and not once a request. Two H100 NVL, Q8_0,
`--tp 2`, `TS_KV_INITIAL_TOKENS=2048`, greedy, 384 tokens a request: 64 / 92 /
123 / 146 / 165 tok/s at 1 / 2 / 4 / 8 / 16 requests, against 64 / 62 / 32 / 28
/ 31 decoding them one at a time (`TS_Q4E_DISABLE_ARENA_DECODE=1`). As on one
device, the batched answers were the one-at-a-time text at 1 to 8 requests and
parted from it at near ties at 16, where a graph is wider than the 8 rows that
keep the one-token fusions (after 53 to 1382 characters under `--tp 2`, 6 to
1349 on one GPU). Under `--tp` a request's caches are sized for the whole context window
unless `TS_KV_INITIAL_TOKENS` says otherwise: 1.8 GB a rank at
`MAX_CONTEXT=65536` on this model, which left those GPUs room to batch 2 and 4
requests (83 and 88 tok/s against 60 and 52) but not 8, where the arena is
declined for memory.
The gathers preserve FP32 using TensorSharp's CUDA collective. Fallbacks use
zero-copy chunks below upstream CUDA's automatic BF16 threshold, or an FP32 host
sum when a device collective is unavailable. Two collectives per FFN communicate
more data than splitting the down dot product, preserving numerical fidelity
without changing ggml.

The intermediate and output widths must divide evenly. Each projection slice
contains complete output rows and retains the full input quantization blocks.
Quantized prefill preserves the original MMQ tile and reduction geometry;
gate/up slices retain overlapping 128-row edge tiles and crop their outputs.
For the checkpoint's 640 intermediate channels, TP2 stores 384 rows per rank
for each logical 320-row slice; TP4 stores 256 for each logical 160-row slice.
Unsupported degrees are refused from GGUF
metadata before bulk weight loading. The current model path requires CUDA MMQ
stream-K support and FFN weights in Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS,
IQ2_XS, IQ2_S, IQ3_XXS, IQ1_S, IQ3_S, IQ4_XS, IQ4_NL or Q8_0. This includes
UD-Q2_K_XL's IQ2_XS/IQ3_XXS routed gate/up weights, Q5_K/Q6_K shared weights,
and IQ4_NL/Q8_0 down weights. UD-Q4_K_XL's Q5_1 down weights use TensorSharp's
exact output-row kernels. Full output row counts and down output slices must
be multiples of 128. IQ1_M remains refused
because upstream ggml has no MMQ tile kernel for it. Other FFN types, including
F32/F16/BF16, and devices/layouts without exact strip kernels are refused.
The historical physical GPU validation below uses NVIDIA A40 GPUs and the
formats identified for each run. The expert slices currently require host
buffers totaling the routed-expert bytes plus these overlapping rows, in addition to the mapped checkpoint;
these buffers remain alive while the model runs. Loading skips a full prefault
of the sparse PLE table, whose rows are gathered on demand.

`--layer-split N` remains a separate option: each GPU holds a contiguous run of
whole layers. It is available on `ggml_cuda`, `ggml_vulkan` and the direct `cuda`
engine. Do not combine `--tp` and `--layer-split`. On `ggml_cuda`, either mode
can span nodes using `--tp-node-id` and `--tp-peers`.
The layer pipeline transfers the complete F32 hyper-connection residual;
GDN, convolution, PLE and QSA state remain with the layers that own them.
Distributed TP partitions expert and shared-FFN output rows across the global
rank count and gathers the complete activations before the next operation.
The weight-format and row-alignment checks above apply to the global degree.
Distributed execution currently serves serialized text forwards; media,
speculation and retained sequence snapshots are unavailable.
Older layer-split commands should use `--layer-split N` or
`TENSORSHARP_LAYER_SPLIT_DEGREE=N`.

The numerical fixture `eng/tests/qwen4exp-tensor-parallel.py` compares two-layer
spans with unsharded execution, including prefill, replay, QSA, multi-axis
RoPE, all-logit taps and recurrent rollback. Its quantized mode
(`--quantized-ffn --tokens 1,2,3,4,5,6,7,8 --rollback-width 8`) passes all 102
checks on both CUDA TP2 and TP4, including recurrent/QSA/PLE snapshot restoration
at verify width 8, with zero hidden or logit error. CPU TP4 loopback separately
passes 42 F32 checks and is correctness-only. The synthetic F32 CUDA TP4 replay
at width 17 exceeds its existing tolerance (maximum hidden error 3.49e-5);
F32 FFNs are refused by the public model capability check, and this case is
not counted as passing.
`eng/tests/qwen4exp-tp-quantized-ffn.py --hidden 2560` separately covers the real
640-channel FFN width with IQ3_S/IQ4_NL and IQ4_XS/Q8_0 gate/down weights,
Q8_0 shared weights, and widths 1 through 8, 17, 31 and 128; CUDA TP2 results are
bitwise identical with `--require-bitwise`. `eng/ForcedLogitProbe` compares full-model
logits while forcing identical token histories, so an early greedy mismatch
cannot hide the numerical error of subsequent decode steps.

### UD-Q2_K_XL TP2 validation, 2026-10-05

The supplied checkpoint runs with `--backend ggml_cuda --tp 2` on two NVIDIA
A40 GPUs, retaining its original quantized bytes and upstream MMQ arithmetic.
The deployed native SHA-256 is
`a7d30f8ac0648372b8fd63881b9a2c915032b75ee2e51299265d61349d793301`;
upstream ggml remains unchanged at
`ffa4e8b80930029a35991f94e7c8a93cd67730ab`.
This VM's behavioral CUDA peer-access probe fails, so NCCL uses shared-memory
transport for the F32 FFN gathers.

- All 39 managed TP tests passed, with no failures or skips.
- Native strip checks passed 410 CPU comparisons and 820 CUDA comparisons
  across all 14 supported formats, with 410 checks on each A40. CUDA gate/up
  inputs have width 2560; token widths include 1, 4, 8, 9, 17, 31 and 129.
  Of the CUDA checks, 524 use the owned MMQ strip path and 296 use the upstream
  CUDA path selected for smaller shapes.
- All 26 complete synthetic FFN comparisons were bit-identical at hidden
  width 2560 / FFN width 640, with 16 experts and 10 selected experts. They
  cover IQ2_XS/IQ4_NL routed gate/down with Q5_K/Q8_0 shared weights, and
  IQ3_XXS/IQ4_NL with Q6_K/Q8_0, at widths 1 through 9, 17, 31, 128 and 129.
- Five full-model forced-history cases, with 24 rows each, produced
  29,798,400 F32 logits byte-identical to the original layer-split-2 binary.
  Five greedy answer cases passed arithmetic, extraction, Python behavior,
  ordered-square and thinking checks; their outputs also match the candidate
  layer-split-2 outputs byte-for-byte.
- The exact interactive options `--interactive --think --max-tokens 20000`
  with this model and `--backend ggml_cuda --tp 2` completed two turns at EOS,
  answering `703` then `720`, generating 163 and 76 tokens at 46.1 and 45.4
  tokens/s. The second turn reused 232 of 256 prompt tokens.

CUDA graph-enabled numerical checks passed. Targeted Q5_K memcheck with
`GGML_CUDA_DISABLE_GRAPHS=1` reported zero errors. The graph-enabled sanitizer
attempt reported handled CUDA graph-update API status 910 and is not counted
as a clean sanitized capture run. The short interactive turns do not validate
a full 20,000-token generation. Long context, vision, perplexity, TP4 and other
devices were not evaluated in this campaign. Generated evidence stays in
ignored `docs/validation/qwen38-q2-tp/`; reusable answer validation is
`eng/validation/validate-qwen4exp-tp-answers.py`, with requests in
`InferenceWeb.Tests/Fixtures/Qwen4Exp/tp-quality-requests.jsonl`.

Four fresh CLI starts compared layer2, TP2, TP2, layer2 with F16 KV, context
4096, two CPU threads and fixed-input pp512/tg128. Each start performed normal
kernel warmup and four timed repetitions. Throughput cells show **first /
median of repetitions 2–4 / range of all four**, in tokens/s; load excludes
kernel warmup.

| Mode / start | Load (s) | Warmup (s) | pp512: first / median / range | tg128: first / median / range |
| --- | ---: | ---: | --- | --- |
| layer2 / 1 | 20.49 | 138.68 | 942.0 / 998.6 / 942.0–1010.6 | 53.6 / 53.9 / 53.6–54.1 |
| TP2 / 2 | 46.56 | 2.47 | 806.8 / 882.2 / 806.8–884.6 | 49.4 / 51.5 / 49.4–52.2 |
| TP2 / 3 | 48.42 | 2.43 | 797.2 / 824.8 / 797.2–841.8 | 48.5 / 50.3 / 48.5–51.4 |
| layer2 / 4 | 19.51 | 138.08 | 942.2 / 986.2 / 942.2–998.1 | 52.8 / 53.4 / 52.8–53.6 |

Pooling the six later repetitions per mode gives TP2 **848.0 pp / 51.4 tg**
versus layer2 **997.4 pp / 53.65 tg** tokens/s: TP2 is 15.0% slower for prefill
and 4.2% slower for decode in this workload. Use `--layer-split 2` for the
highest measured throughput here. TP2 performs additional overlapping edge-row
work, replicates attention and uses two F32 gathers per FFN. These measurements
do not isolate each cost. Weight uploads occur during different startup stages
in the two modes, so compare load together with warmup when assessing startup.

All four untimed greedy chains (prefill plus 128 decode tokens) were identical;
runtime binary hashes and clean upstream identity stayed unchanged. Timings
use a repeating 17-token input cycle that can favor hot PLE rows and expert
locality. OS caches were not cleared and GPU clocks were not locked. Results
exclude natural chat formatting, HTTP, scheduling and sampling, and do not
establish real-text, long-context or cold-storage throughput. Complete logs,
timings and identity snapshots are in the ignored
`docs/validation/qwen38-q2-tp/benchmark-20261006T010554Z-21317/` directory.

### Earlier UD-IQ4_XS validation

On the UD-IQ4_XS checkpoint and NVIDIA A40, TP2 and TP4 each match the stabilized
plain layer2 execution byte-for-byte over 120 full-vocabulary rows: three text
prompts, a one-token synthetic prompt and a 128-token synthetic prefill, with
24 forced steps per case. TP2 uses the same native build (`da25f156`) for
both paths; TP4 uses native `1ba6d7a4` against the saved `da25f156` plain reference.
The final HTTP and speculation checks below use native `473ee64d`.
These runs use unchanged ggml `353b63b439f27ab2cc19dac97ab1681ba6d2d084`.
Stabilizing CUDA prefill rounding can
change logits or low-margin greedy choices from older binaries; original
reference vectors are retained separately and are not claimed as bitwise
compatible. In the final HTTP comparison with the original binary, the first
logit vector has relative L2 error 0.0416; 14 of 16 text strings match exactly,
with punctuation-only differences in the other two. That historical numerical
comparison does not pass the strict parity gate.

The final CUDA TP2 HTTP run matches same-build layer2 execution on all 16 text
prompts, four tool round-trips and four image answer turns; its first real
prefill's 248,320 logits are byte-identical. Learned MTP and n-gram speculation
each preserve all 96 plain-greedy tokens in both text and image tests. The
stress configuration uses `TS_SPEC_DRAFT=7 TS_SPEC_PMIN=0`; learned MTP actually
reaches verify width 8 in both scenarios and exercises rejection rollback.
All runs exit cleanly, with unchanged runtime binaries verified before and
after execution.

A separate six-start benchmark used 2× NVIDIA A40, UD-IQ4_XS, context 4096,
F16 KV, 128-token kernel warmup and two CPU threads. The table preserves both
starts per mode in execution order. Each process times five fixed-input
pp512/tg128 repetitions; a separate untimed 128-token greedy chain checks
correctness. Throughput cells show **first / median of repetitions 2–5 / range
of all five**, in tokens/s. Loading excludes kernel warmup.

| Mode / start | Load (s) | Warmup (s) | pp512: first / median / range | tg128: first / median / range |
|---|---:|---:|---|---|
| Original layer2 / 1 | 64.85 | 20.67 | 259.2 / 406.60 / 259.2–439.7 | 24.9 / 29.45 / 24.9–36.6 |
| Current layer2 / 1 | 48.98 | 21.51 | 262.7 / 427.75 / 262.7–446.5 | 30.3 / 37.90 / 29.2–38.0 |
| Current TP2 / 1 | 85.68 | 10.78 | 229.0 / 345.75 / 229.0–358.9 | 22.0 / 24.10 / 18.2–25.9 |
| Original layer2 / 2 | 68.74 | 22.14 | 255.9 / 419.25 / 234.0–454.1 | 29.2 / 30.40 / 29.2–35.7 |
| Current TP2 / 2 | 80.22 | 12.56 | 230.3 / 349.15 / 213.9–356.7 | 18.6 / 27.30 / 14.2–34.3 |
| Current layer2 / 2 | 25.66 | 18.13 | 258.3 / 414.65 / 234.9–422.1 | 30.3 / 33.85 / 30.2–37.8 |

All six processes exit cleanly and pass runtime identity guards. Both current
layer2 and TP2 starts produce exactly the same full greedy chain. Their
aggregate steady rates are layer2 **421.20 / 35.875** versus TP2
**347.45 / 25.70** tokens/s: TP is **17.5% slower for prefill and 28.4% slower
for decode** on this machine. Attention and recurrent state are replicated;
two exact-F32 FFN collectives per layer add communication overhead on these
PCIe GPUs (`NCCL_P2P_DISABLE=1`). Layer splitting is the faster measured option.

The original binary's synthetic greedy chain diverges from all current runs
at zero-based decode index 50. Its strict compatibility comparison remains
failed; current/original throughput ratios are not qualified by token parity.
These are mixed/warm-cache loads without page-cache eviction, and GPU clocks
are observed rather than locked. The substantial timing and loading variation
precludes a cold-storage or universal speedup claim.

One additional TP2 diagnostic used the same binaries and settings with
`GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_BF16_THRESHOLD=0`. Its full greedy
chain and runtime guards passed, but the tradeoff was mixed: pp512
**216.3 / 309.15 / 202.7–321.4**, tg128 **16.5 / 30.30 / 16.5–40.1**
(first / steady median / all-five range), load 100.69 s and warmup 10.05 s.
This single start reduces prefill throughput and does not justify changing the
default NCCL transport.

The historical measurements below are for **layer splitting**, not tensor parallelism.

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

## Serving on the direct-CUDA engine

`--backend cuda` keeps each request in a sequence slot of its own (attention and
indexer caches, GDN and PLE state) and decodes the running slots together, up to 16
per step. These settings shape it as a server:

| Variable | Default | Effect |
| --- | --- | --- |
| `TS_Q4E_SLOT_TOKENS` | `32768` | A slot's initial capacity. It doubles as its sequence grows, up to the context; a slot that cannot grow frees the oldest retained conversation's slot first. `MAX_CONTEXT` keeps bounding each request. |
| `TS_Q4E_REWIND_CHECKPOINT` | on | Each slot keeps its recurrent states as they were at the end of its last multi-token forward. A next turn that matches the slot only partway into the previous reply (re-rendered reasoning or tool calls) rewinds there and replays the slot's own tokens up to the match; `0` turns it off. |
| `TS_SCHED_PIPELINE_DECODE`, `TS_Q4E_DECODE_LANES` | on | Under `--layer-split 2` and up, batched decode runs as two groups in flight, so the first GPU runs one group's layers while the last runs the other's (two H100 NVL, Q8_0: 582 against 377 tok/s at 16 streams, the same tokens); `0` for either decodes one batched step at a time. |
| `TS_Q4E_PROMPT_PIPELINE` | on | Under `--layer-split 2` and up, a prompt chunk that more of the prompt follows runs without the LM head and stays in flight on the last GPU while the first starts the next chunk (two H100 NVL, Q8_0: a 40K-token prompt at 6,229-6,262 tok/s against 4,535-4,597, the same answers); `0` waits for every chunk. `TS_Q4E_PLE_GATHER_THREADS` (min(16, half the CPUs)) is the threads that gather a prompt ubatch's PLE rows. |
| `TS_Q4E_BATCH_GRAPHS` | on | Batched steps (batched decode steps and batched speculative verifies) run as CUDA graphs, one per set of slots, their QSA regimes, rows and head form, captured on a set's third step and at most `TS_Q4E_BATCH_GRAPHS_MAX` (64) of them; `0` issues them kernel by kernel. |
| `TS_Q4E_Q80_PAIR` | on | Q8_0 decode projections (1 to 16 rows) as one arithmetic on three kernels (sm_80+): int8 tensor cores, a dp4a twin with four lanes on a block and a dp4a kernel with a lane on a block, the fastest for the shape and row count; `0` runs the decode-once dense kernels (`TS_DENSE_FAST`). The two sum in different orders, so answers differ in the last bits. Two kinds of weight keep the dense kernels at every row count: the `TS_Q4E_HC_Q8` copies of the hyper-connection down mixers and the MTP block's Q8_0 weights (on the RTX 5090 the pair ran them ~2x slower from three rows, measured before the lane kernel). |
| `TS_Q4E_EXPERT_VEC` | on | Q8_0 decode experts read each weight row four lanes per 32-value block; `0` runs the shared decode kernels (sixteen lanes a gate/up row, eight a short down row; a different order). Paged-in experts follow their residency table, as resident ones compute. |
| `TS_MOE_Q80_LANES` | on | Q8_0 decode experts read a weight row with eight lanes (four rows a warp, a lane on whole blocks as aligned 64-bit words): the down projection always, gate/up from 40 slots with a slot's gate and up rows in the same lanes; the same sums. `0` keeps a warp on every row. |
| `TS_MOE_GATEUP_Q80_PAIR` | on | A Q8_0 decode slot's gate and up rows computed together (the activation loaded once; the same sums): in eight lanes a row from 40 slots, or in one warp from 32 without `TS_MOE_Q80_LANES`; `0` keeps them in separate warps. |
| `TS_Q4E_PLACEMENT_SLOTS` | `16` | The layer split balances weights plus this many slots' caches, as many as fit; `1` balances for one. |
| `TS_Q4E_UBATCH` | `512` | Rows per prefill forward. |
| `TS_Q4E_PERF` | `0` | `1` logs each forward's time; `3` logs each stage's GPU time per step (uncaptured). |
| `TS_Q4E_PERF_SAMPLE` | `0` | Every Nth forward runs uncaptured with stage timers and records `tensorsharp.engine.stage.duration` per stage, step kind and rows, instead of a log line. A decode lane's step is sampled too (the other lane finishes first), and at most one forward every `TS_Q4E_PERF_SAMPLE_MIN_MS` (10,000) is sampled. With `TS_Q4E_PERF` at `2` or more it is off. |
| `TS_Q4E_PIPELINE_OVER_SPEC` | on | With the MTP head under `--layer-split 2` and up, a lone request speculates and two or more decode plainly as two groups in flight; `0` puts the batched verify first again (a draft head then turns the groups off). |
| `TS_Q4E_MTP_FOLLOW`, `TS_Q4E_MTP_CHAIN_SPLIT` | on | With the MTP head, the block takes a lone request's prompt on its own device as each ubatch finishes, instead of through the host; on a layer split the last GPU holds a second copy of the token embedding for it (0.67 GiB for Q8_0), which also keeps a window's draft chain on that GPU. `0` for the first keeps the host's catch-up; `0` for the second turns both off on a layer split. |
| `TS_KDA_SCAN_R4` | off | `1` runs the gated-delta scan four state rows a warp in prefill too; unset, a prefill ubatch's scan runs one row a warp and prefetches the next row's inputs (bit-identical). |

The scheduler sizes its reservation pool for one context per running sequence for this
engine (`IBatchedPagedModel.UsesNativeSequenceSlots`), as for DeepSeek V4.1, instead of
one context in all.

The engine reports its slots and graphs as metrics (`IServingTelemetrySource`, see
USAGE.md's telemetry section): slots running and retained, their memory and capacity,
grows and grow failures, retained slots reclaimed for a new slot or for growth, rewinds
by route (checkpoint, replay, reset, refused) and the tokens replayed, CUDA graph
captures and launches by kind, and the host time spent issuing forwards versus waiting
for them. A rising `tensorsharp.slots.reclaims` rate means conversations are evicting
each other; captures that keep pace with launches mean graphs are not reused.

Measured on 2026-11-07/08 on two H100 NVL 94 GB (`--layer-split 2`, Q8_0, 262,144
context, ubatch 2048), the server behind a LiteLLM gateway:

- Ten concurrent agent loops of four turns (22K-token system prompts, a tool call and
  a 1,500-token tool result per turn): 91.0% of later turns' prompt tokens came from
  cached slots, with first tokens at 3.5 s p50 and 6.3 s at most. With slots that
  held the whole context from the start (`TS_Q4E_SLOT_TOKENS=262144`) only 7 fit;
  the agents took each other's retained slots, 24.6% was reused, and later turns'
  first tokens took 73 s p50 and up to 417 s.
- One request decodes at 94 tok/s; 16 concurrent 512-token answers at 250 tok/s in
  all (173 with whole-context slots). Batched-decode graphs added 7-12% at 8-16
  streams.
- A 36K-token prompt prefilled at ~3,300 tok/s. The tensor-core MoE kernel now
  decodes each 128-row weight tile once for up to four 32-token passes; before, a
  4,096-token chunk's experts (about 80 member tokens each) decoded every tile three
  times.
- Per decode step (`TS_Q4E_PERF=3`), one row: hyper-connection mixers 3.8 ms,
  routed experts 2.4 ms, GDN projections 1.5 ms, shared expert 1.5 ms, router
  1.2 ms of 13.5 ms uncaptured; four rows 22.6 ms. A step is ~2,000 small kernels
  and latency-bound.

Decode projections and experts were the per-row cost of a batched step: from one row to 16 a
step's GPU time grew ~2.8 ms a row, the Q8_0 matvecs alone from 4.0 to 20.1 ms (their weight
words load once per call, but each row's dot product and block reduction run on their own) and
the experts from 2.4 to 18.9 ms (a lane per 34-byte block, ~1.5 TB/s for one row). With the pair
matvec and the experts' row layout (2026-11-09, the same two H100 NVL), greedy decode ran at
100 / 144 / 197 / 273 / 345 tok/s for 1 / 2 / 4 / 8 / 16 streams against 94 / 141 / 188 / 234 /
270 with both switched off, and a 16-row step's GPU time fell from 55.8 to 42.5 ms.
`Q80PairMatvecTests` holds every row of a 2-16-row pair call bit-equal to the one-row call at
this model's shapes and the experts bit-equal at every batch size, and the engine's
batched-decode tests still pass bit for bit. These two do change answers: against the shared
kernels their outputs move by ~1e-7 relative, enough to flip a near tie under greedy decoding
(of ten paraphrases of a two-city weather prompt with tools, 9 made both calls under either
arithmetic, a different one missing each time).

With each of the other switches on and off greedy answers are byte-identical. The CUDA tests
`BatchedDecode_ComputesEverySequencesOwnStep`, `AGrowingSlot_AnswersAsOneThatHeldTheWholeContext`
and `ARewindFromTheCheckpoint_ContinuesAsTheSequenceThatStoppedThere` cover them on the
synthetic fixture. The direct engine has no MTP drafter and refuses `--tp`; under a layer
split one GPU computes at a time.

With the shared MTP head on a layer split (2026-11-12, the same two H100 NVL, Q8_0, upstream
`567d171d`), one configuration now serves a lone request and a crowd. Before, loading the head
turned the two decode groups off and sent concurrent requests through the batched verify, one
step at a time, and a speculating request's prompt handed every row's streams to the host and
back:

| Greedy tok/s at streams | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| No draft head | 122 | 232 | 373 | 552 | 754 | 984 |
| MTP head, before | 178 | 209 | 273 | 363 | 470 | 449 |
| MTP head, now | 179 | 241 | 389 | 560 | 771 | 1,019 |

- **Two or more requests decode as groups** (`TS_Q4E_PIPELINE_OVER_SPEC`): the scheduler asks
  the model whether its pipelined groups outrun its batched verify
  (`IPipelinedBatchedDecode.PipelinedDecodeOutrunsSpeculation`) and, if so, speculates only for
  a lone decoder. A request that was speculating joins a group and takes up its drafts again
  when it is alone (the block restarts after the plain steps, `TS_Q4E_MTP_RESUME`).
- **The block follows a prompt on its device** (`TS_Q4E_MTP_FOLLOW`,
  `IDraftHead.DraftSelfCatchUp`): after each ubatch's last layer the block joins every row's
  token with the streams of the row before it (the previous ubatch's last row is kept on the
  device) and writes its K/V. The prompt therefore runs on the plain path, its chunks in
  flight, and the first draft continues from the block's rows: a 19K-token prompt of scripts
  and compose files read at 6,670 tok/s against 4,070 through the host, a 40K-token one at
  6,066 against 3,819, with 80-85% of the drafts kept either way and the same answers.
  `TheMtpBlock_FollowsAPromptOnItsDevice_AsTheHostCatchUp` holds the drafts after a followed
  prompt to those of the host's catch-up, on one GPU and on a layer split.
- **The token embedding beside the block** (`TS_Q4E_MTP_CHAIN_SPLIT`): on a layer split the
  embedding is on the first GPU and the block on the last, so following needs a second copy
  (0.67 GiB); with it a window's draft chain runs on the last GPU as it does on one GPU
  (`TheMtpDraftChain_RunsOnALayerSplit_AsOnOneGpu`). The chain by itself is worth about 1% there.

Every greedy answer above equals the plain build's at the same stream count, and six requests
overlapped with staggered starts and lengths answer as each does alone.

Tried and dropped: more than two decode lanes, each with its own stream and decode scratch, so
that a GPU runs several groups' steps at once. A step is about two thousand small kernels, but
two one-row halves side by side on a GPU take 6.0 ms against 5.35 ms for one two-row half:
4 / 8 / 16 / 32 streams ran at 335 / 505 / 720 / 949 tok/s on four lanes against 374 / 548 / 757 /
996 on two. Batching rows in a step beats running steps side by side.

A prefill ubatch's gated-delta scan with one state row a warp computes exactly what the
four-rows-a-warp scan computes (`Qwen4ExpPrefillKernelTests`); on the same two H100 NVL
(2026-11-09) the 2,048-token forward's scan fell from 137 to 113 ms.
Sampling a batched step's rows in parallel took 16 streams at temperature 1 (top-k 20) from
291 to 316 tok/s in all and 8 from 235 to 250; the draws are unchanged.

## Benchmark matrix

[`benchmark_config_glm53_qwen38.json`](../../benchmarks/engine_comparison/benchmark_config_glm53_qwen38.json)
registers this model as `qwen38-flash-next` at a pinned Hugging Face revision,
with its `mmproj-BF16.gguf` attached so the `image` scenario can run. Two
registry facts are worth repeating here.

The published Q8_0 shards carry **no** `nextn`/`mtp` tensors at all, so
`mtp_supported` is false and `--mtp on` cells are gated out with a reason
rather than quietly serving standard decode.

This historical matrix runs the model on the column that passes `--layer-split N`.
Its split degree comes from `--layer-split`, so on a backend column
that passes none, TensorSharp builds a single-device context and all 175.3 GiB
land on one card. The config therefore gives it a `min_tp` (4, the weights-only
floor — 8 is the degree the 8×A40 box is meant to use) and the harness records
its cells on the no-`--layer-split` column as skips reading
`needs --tp 4 (does not fit 1 GPU(s))` instead of letting them OOM. In this
harness message, `--tp` is the legacy GPU-count selector; the selected column
passes `--layer-split` to TensorSharp. Run it as:

```
python run_matrix.py --config benchmark_config_glm53_qwen38.json \
    --models qwen38-flash-next --backends ggml_cuda_split
```

That column tells llama.cpp `--split-mode layer` over the same GPUs, so the
reference column uses the same placement on both engines. It does not measure
the newer TensorSharp `--tp` implementation.
