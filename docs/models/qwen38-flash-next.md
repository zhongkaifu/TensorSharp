# Qwen 3.8 Flash Next (`qwen4exp`)

[← back to model index](README.md) | [中文](qwen38-flash-next_zh-cn.md)

Qwen3.8-Flash-Next is a hybrid MoE: GatedDeltaNet recurrent layers interleaved
with full-attention layers (some behind Qwen Sparse Attention's indexer), a
PLE n-gram embedding block, ×4 hyper-connection streams and a 512-expert MoE.
The GGUF architecture id is `qwen4exp`. Weights:
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
(multi-shard per quant directory; point `--model` at the `-00001-of-` shard;
image input needs its matching vision projector, such as `mmproj-BF16.gguf` or
`mmproj-F16.gguf`: the CLI loads it from beside the model when
`--image` is given, and the server needs `--mmproj`).

## How TensorSharp runs it

On the GGML backends the whole token runs as (almost) one graph — embedding,
PLE (in-graph), all 48 layers, the final mixer and the LM head — with a
shape-keyed cache of reusable ggml graphs (where the span declines, per-layer fused
kernels run instead, and those in turn fall back op-by-op). Vision rides the
Qwen3.5-VL tower with (T,H,W) IMRoPE positions; multi-image and multi-turn
image sessions are supported, with KV reuse across turns (the GDN recurrence
cannot rewind, so a cached prefix is reused only when the new prompt extends
it exactly; see [Retained-prefix reuse](#retained-prefix-reuse)). Radix reuse
can continue past an identical image or video span: the complete state includes
the M-RoPE cache gap and QSA position history, and the key checks media identity
and span boundaries as well as tokens. A finished primary cache stays in place
for the next exact turn; it becomes a retained holder only when another request
needs to displace it.

Thinking can be switched on or off. With it off, the assistant turn opens with
the closed, empty `<think>\n\n</think>` block the published template emits,
and replayed history keeps that exact suffix so cached prefixes still match.

The published Qwen4Exp vision patch merger uses GELU(erf), separately from the
transformer blocks' `gelu_pytorch_tanh`, matching the
[Transformers v5.16.1 reference](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1705).
The supplied GGUF path retains its existing tanh merger default. Set
`TS_Q4E_VISION_MERGER_ERF=1` before loading the projector to select the published
erf merger independently of the block activation. The delivered base-model default
passes 12/12 image cases, reproducing all 16 earlier tanh answers. The full erf
trial passed 10/12; a subsequent same-build C=1 OCR control passes 2/2 with tanh
and 1/2 with erf, which reads blue `9364` as `9334`. These scoped checks support
preserving the compatibility default; they do not explain the recognition
difference or establish broader activation quality. See the
[image validation guide](../../eng/validation/qwen38-parallel-vision.md).

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

Tensor parallelism, layer splitting, other KV dtypes and unavailable span or
backend geometry continue through the per-sequence fused path when supported.
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
Local tensor parallelism is now available as described below; multi-node execution remains unsupported.

`--draft-model mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` attaches the per-token
MTP block (GGML backends only, and the head must be a single GGUF file, attached
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
  The direct `cuda` engine reads the rows from the mapping too, and warms the table into
  the page cache after loading.
- **The first layers' routed experts run on the host, from the same mapping.**
  `--n-cpu-moe N` / `--cpu-moe` choose them on the GGML GPU backends (measured on
  `ggml_metal` and `ggml_cuda`). On `ggml_metal` with neither set, the engine plans the
  split itself: it keeps whole layers' experts on the GPU while they fit both the Metal
  working set and the RAM that the host layers need as page cache, and offloads the rest.
  On an M5 Pro with 48 GB (51.5 GB of RAM, a 40.2 GB Metal working set):

  ```
  [moe-offload] qwen4exp (planned): routed experts of 33 of 48 layers run on the host from the GGUF mapping (31.7 GB read on demand); the accelerator holds 15 layers' (14.4 GB). Metal working set 40.2 GB, RAM 51.5 GB; --n-cpu-moe N overrides.
  ```

  Past a point, wiring more layers does not make it faster: a wired layer holds all 512
  experts, used or not, and takes page cache from the layers that read theirs from the SSD.
- **Decode** runs each host layer's ten experts on TensorSharp's own kernel: ggml's CPU
  dot products on a thread team that is woken once per layer and parked when the layer is
  done. A spinning team slowed the GPU's work between layers 1.5-2x on Apple silicon, so
  the team sleeps. `TS_HOST_MOE_DECODE=0` restores the ggml graph path.
- **Prefill** of 128 tokens or more (`TS_HOST_MOE_DEVICE_MIN_BATCH`) streams each host
  layer's used experts to the GPU per chunk. Their pages are faulted in on 16 threads
  first: on the M5 Pro that took a 1,818-token prefill from 37.2-38.0 s to 15.7-16.6 s and
  left decode unchanged.
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

Single-device `ggml_cuda` also plans expert placement against current free VRAM,
pending float weights and caches, driver headroom and a 3 GiB graph reserve.
Explicit `--n-cpu-moe` / `--cpu-moe` settings override this plan. Tensor parallel
and layer-split runs still require their explicit placement configuration.

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
fallback at about 500 tok/s. The engine has no host-expert seam, so `--n-cpu-moe` there
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
stream-K support and FFN weights in Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS,
IQ2_XS, IQ2_S, IQ3_XXS, IQ1_S, IQ3_S, IQ4_XS, IQ4_NL or Q8_0. This includes
UD-Q2_K_XL's IQ2_XS/IQ3_XXS routed gate/up weights, Q5_K/Q6_K shared weights,
and IQ4_NL/Q8_0 down weights. Full output row counts must be multiples of 128,
and down output slices must also be multiples of 128. IQ1_M remains refused
because upstream ggml has no MMQ tile kernel for it. Other FFN types, including
F32/F16/BF16, and devices/layouts without exact strip kernels are refused.
The historical physical GPU validation below uses NVIDIA A40 GPUs and the
formats identified for each run. The expert slices currently require host
buffers totaling the routed-expert bytes plus these overlapping rows, in addition to the mapped checkpoint;
these buffers remain alive while the model runs. Loading skips a full prefault
of the sparse PLE table, whose rows are gathered on demand.

`--layer-split N` remains a separate option: each GPU holds a contiguous run of
whole layers. It is available on `ggml_cuda`, `ggml_vulkan` and the direct `cuda`
engine. Do not combine `--tp` and `--layer-split`; distributed
`--tp-node-id`/`--tp-peers` groups are
unsupported. Older layer-split commands should use `--layer-split N` or
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
