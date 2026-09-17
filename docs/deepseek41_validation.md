# DeepSeek V4.1 Flash validation protocol

This file contains the reproducible test plan and measured numerical evidence.
It does **not establish model-quality or performance parity**. The evidence
below distinguishes passing checks from remaining numerical differences.
The requested VM's [hardware and dependency versions](validation/deepseek41/environment.json)
and each artifact's binary hashes define the measured environment.

The final eight-GPU layer-placement profile passes all 138 end-to-end cases,
with median sustained single-request decode of 34.83 tokens/s. The final managed
code passes 3,651 correctness tests locally and on the VM; native CTests pass
13/13. See the [final placement results](validation/deepseek41/final-placements/README.md)
for CPU offload and routed-expert tensor parallelism. Acceptance remains
incomplete: those two profiles each introduce an additional dependent-tool-call
failure, the strict full-checkpoint numerical oracle differs, earlier Qwen3.5
JSON latency regressions are not explained by the follow-up controls, and a
compatible llama.cpp comparison is unavailable. The passing cases do not close
those requirements.

Two later reports extend this one and are not superseded by it. The
[quantization report](validation/deepseek41-quants/README.md) covers the
Q4_K_M release, where the Engram tables no longer fit in VRAM and the routed
experts need CPU offload. The
[direct-CUDA backend report](validation/deepseek41-cuda-backend/README.md)
covers `--backend cuda`, which runs V4.1 on its own kernels and is explicitly
not yet held to a numerical gate. The pure-C# `cpu` executor is checked against
`eng/dsv41-reference.py` at 2e-5 on the F32 fixture; quantized fixtures are
chaotic for every implementation, including the native one, and are not a
tight-tolerance target.

The checkpoint is `vcruz305/DeepSeek-V4.1-Flash-GGUF`, revision
`8e0c4de3cb6519bfc11ed69dc87184b457a57bb5`, Q2_K, seven shards beginning with
`DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf`. Use these exact files for both
engines. Record `sha256sum` of every shard, both source commits, build flags,
launch arguments/environment, GPU models/VRAM/driver, CPU affinity/quota, RAM,
and competing processes alongside results. A llama.cpp build rejecting
`deepseek41` leaves the comparison blocked; an unavailable reference is never a
passing baseline.

A [later primary-source recheck](validation/deepseek41/llama-reference-availability/README.md)
at 13:08–13:10 UTC on September 11 found no callable V4.1 runtime in official
upstream `5bda51bf…`, the current supplied patch, or its linked conversion PR.
Those newer revisions were inspected without building them; the actual
tested load failure remains pinned to `df03399b…`.

All seven downloaded Q2_K shards passed a complete SHA256 scan against the
pinned repository's LFS digests: 264,514,761,248 bytes verified. The bounded
three-worker scan took 283.8 seconds while no qualified benchmark was active.
The exact expected/observed digests and file sizes are preserved in
[checkpoint-sha256.json](validation/deepseek41/checkpoint-sha256.json). Reproduce
with `eng/dsv41-verify-download.py CHECKPOINT_DIRECTORY --report REPORT.json`;
this reads every model byte and should run outside throughput measurements.

The [matrix config](../benchmarks/engine_comparison/benchmark_config_deepseek41.json)
provides server lifecycle and placement plans. Its `--tp` number denotes **GPU
count** in the explicitly named `ggml_cuda_layer` and
`ggml_cuda_layer_cpu_moe4` profiles. Those are contiguous layer placement,
not tensor parallelism. `ggml_cuda_true_tp` explicitly sets `TS_DSV41_TP` to
the requested rank count: every routed expert's gate/up matrices use column
shards and its down matrix uses matching row shards. Ranks execute concurrently;
the current reduction stages F32 partial outputs through the host. Attention
and shared experts retain layer placement. This profile has independent
2/4/8-GPU fixture validation and a completed eight-GPU Q2_K run described
below. The initial host-staged TP profile was slower than layer placement. Its llama.cpp mapping remains unavailable until a compatible
runtime and equivalent execution profile can be tested.
The text checkpoint can be paired with `deepseek41.vision.gguf`, prepared from
the pinned official vision weights. The native image graph and text injection
have independent CPU/CUDA fixture coverage. The full Q2_K checkpoint passed
25 image/video requests at concurrency 1/4 and a separate image-after-long-text
request. Audio is unsupported by this model and is explicitly rejected.
Prepare the tokenizer-derived `deepseek41.engram.bin` sidecar using the
[model-card instructions](models/deepseek41.md#prepare-the-q2_k-checkpoint)
before running inference. The matrix defaults to `/workspace/models/DeepSeek-V4.1-Flash-Q2_K`;
`BENCH_DSV41_GGUF` overrides the first-shard path. Automatic GGUF downloads do
not prepare this required sidecar.

```bash
cd benchmarks/engine_comparison
# Use --download never once the seven files have been provisioned in /workspace.
python run_matrix.py --config benchmark_config_deepseek41.json \
  --engines tensorsharp,llamacpp --backends ggml_cuda_layer,ggml_cuda_layer_cpu_moe4,ggml_cuda_true_tp \
  --tp 8 --concurrency 1,4 --download never
```

The full-checkpoint VM profiles default to all eight A40s. Two/four-rank
numerical fixture checks do not establish that the full Q2_K weights fit on
those GPU counts; smaller full-model layouts need substantial explicit CPU
expert offload. Profiles set `TS_DSV41_ENGRAM_WARM=1` to touch the approximately
60 GiB of host-mapped Engram pages before reporting server readiness. Record
that preparation time and cold first-request latency separately. A warm
throughput comparison requires equivalent weight/page residency for both
engines; do not compare an unprepared network-file first request with a warm
run. Override warming explicitly when collecting the cold-load baseline.

The older matrix's text_short includes a ~2K preamble and its multi_turn uses
prewritten history. Use the strict endpoint suite below for an actually short
prompt, generated conversation history, validated tool arguments/results, and
multiple agent steps. Run engines sequentially on the same idle GPUs. Start
both with equal per-sequence context, batch size, microbatch, GPU visibility,
CPU expert placement/thread counts, KV precision, and speculative mode. Note
that llama.cpp's context setting is shared across its parallel slots; the
provided four-slot config requests 262144 total versus 65536 per TensorSharp
sequence. Confirm the effective allocation in both startup logs.

```bash
# Start the reference server, then read its model id from /v1/models.
python validate_inference.py --url http://127.0.0.1:5001 \
  --engine llamacpp --model SERVED_MODEL_ID \
  --weights-id 8e0c4de3cb6519bfc11ed69dc87184b457a57bb5-Q2_K \
  --profile layer8-context65536-ubatch256-cpumoe0-sparse1-warm \
  --repeats 3 --concurrency 1,4 --output results_deepseek41/llama-layer8.json

# Stop reference, start TensorSharp on the same hardware and placement, then:
python validate_inference.py --url http://127.0.0.1:5000 \
  --engine tensorsharp --model SERVED_MODEL_ID \
  --weights-id 8e0c4de3cb6519bfc11ed69dc87184b457a57bb5-Q2_K \
  --profile layer8-context65536-ubatch256-cpumoe0-sparse1-warm \
  --repeats 3 --concurrency 1,4 --output results_deepseek41/ts-layer8.json \
  --reference results_deepseek41/llama-layer8.json --tolerance 0.05
```

Every trial/client receives a deterministic distinct leading marker to prevent
repeated prefix reuse. Markers, sampling settings and canonical initial request
hashes match across engines. Warmup is recorded separately. All responses are
retained; every concurrent request must pass. HTTP failures, SSE error events,
missing usage, incorrect tool IDs/names/arguments, malformed JSON, incorrect
facts, and reasoning-only answers fail validation. Greedy settings include
seed 42, top_p 1, top_k 0 and neutral penalties. Verify that both engines honor
the settings and render the same chat template; equal prompt token counts
alone do not prove equal token IDs.

| Scenario | Check |
|---|---|
| short | Exact integer arithmetic answer, no artificial long preamble |
| decode | 512-token explanation; content sanity check and sustained timing |
| json / json_schema | Exact object, required JSON types, no extra prose |
| long_8k / long_32k | Exact retrieval of three facts near 10%, 50%, 90%; actual usage token counts recorded |
| multi_turn | Real first generation followed by a recall request using returned history |
| tool_round_trip | Validate weather call, supply deterministic tool result, check final JSON |
| agentic | Invoice lookup, calculation using its result, exact final JSON |
| concurrency 1 / 4 | Every individual workflow checked; whole-wave wall time and throughput retained |

Additional explicit scenarios are `decode_8k` (512-token sustained generation
after approximately 8k background tokens), `long_64k`, `short_zh`, and
`json_unicode`. The Unicode fixture requires the exact Chinese strings and
rocket emoji; translated words, replacement characters and extra prose fail.
Use `validate_deepseek41_media.py --fixtures FIXTURE_DIRECTORY --prepare` to
create the deterministic image/video assets, then run that script against the
vision-enabled endpoint at concurrency 1/4. Its optional
`--scenarios image_long_context` checks image injection after long text.
`validate_deepseek41_tools.py` separately exercises required/named/none choices,
serial and parallel calls, reasoning, generated tool history and HTTP 400
policy validation; its default plan contains 30 cases and 35 HTTP requests.

The tool fixtures do not execute generated code or shell commands. The agentic
case measures a bounded tool workflow, not a complete coding-agent evaluation.
Use `--blocking` to repeat protocol checks without SSE, and
`--thinking --max-tokens 2048` to exercise reasoning with an explicit generation
budget. The override is recorded in each request hash and the report. Long-prompt lengths are
character estimates; report actual server token counts. Cases exceeding the
configured context should fail and require a larger matched context profile.

`--structured-tool-results` requests JSON mode only for the final answer after
the tool round trips. It preserves the same strict output checks and records a
different request hash and comparison setting. Report these results separately
from unconstrained agent answers; Markdown-fenced JSON fails the unconstrained
fixture and is not silently stripped by the harness.

Comparison rejects missing/failed reference cases, mismatched weights/profile,
request hashes, prompt lengths or timing sources. It reports median latency
ratios and sustained decode ratios with a configurable 5% regression tolerance.
Raw per-request timings and parallel-wave throughput remain available for
variance and saturation analysis. A passing fixture suite establishes only
these cases; broad model-quality parity additionally requires token/logit or
perplexity comparisons and representative task evaluations. Run regression
models through the existing matrix and this endpoint suite where their tools
and structured output are supported, with the same before/after binaries and
weights. Do not aggregate unsupported scenarios as successes.

Focused harness tests (no models/GPUs needed):

```bash
python -m unittest discover -s benchmarks/engine_comparison -p 'test_*.py' -v
```

On the A40 VM, build the direct CUDA kernels for `compute_86` before running
the managed synthetic GPU lane. The committed PTX targets `sm_120` and cannot
run on an A40. A build without `nvcc` on `PATH` copies that baseline, so a
managed build succeeding alone does not establish GPU readiness. Keep the
already configured GGML native build intact with `TensorSharpSkipGgmlNative`:

```bash
export PATH=/usr/local/cuda/bin:/workspace/deepseek41-work/dotnet:$PATH
dotnet build InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release \
  --no-restore -p:TensorSharpSkipGgmlNative=true -p:CudaArch=compute_86
CUDA_VISIBLE_DEVICES=1 ctest --test-dir TensorSharp.GGML.Native/build \
  --output-on-failure
CUDA_VISIBLE_DEVICES=1 dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj \
  -c Release --no-build --filter 'Category!=Bench&Requires=Cuda&Requires!=Models'
# The actual TP tests require all eight devices visible. Each case explicitly
# skips if its required degree is unavailable; skipped cases are not evidence.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ctest --test-dir TensorSharp.GGML.Native/build \
  -R '^deepseek41-moe-tensor-parallel' --output-on-failure

TS_DSV41_TP=8 TS_DSV4_FA=0 TS_DSV4_GATHER=0 \
  python eng/tests/dsv41-inference.py CUDA_INDEX_FIXTURE_DIRECTORY \
  --library /absolute/path/to/libGgmlOps.so --backend CUDA --gpus 8 \
  --atol 2e-5 --rtol 2e-5 --report validation-cuda-tp8-strict.json
```

The generated fixture also supports a paired sparse-decode check:

```bash
python eng/tests/dsv41-gather.py FIXTURE_DIRECTORY \
  --library /absolute/path/to/libGgmlOps.so --backend CPU
# Same tiny fixture on the requested devices; not a throughput benchmark:
python eng/tests/dsv41-gather.py FIXTURE_DIRECTORY \
  --library /absolute/path/to/libGgmlOps.so --backend CUDA --gpus 8
```

This test loads separate models with `TS_DSV4_GATHER=0` and `1`, then compares
logits across the visibility threshold, compression boundaries, and candidate
pruning. The gather runs on the shared compressed-cache device, and attention
runs on the query device. Cross-device traffic carries the raw ring, selected
row IDs, and compact selected keys; it does not move the entire compressed
cache before gathering. Set `TS_DSV4_GATHER=0` when isolating the dense path.

The released checkpoint's `topk=512` also has a dedicated boundary regression.
At `p0=1024`, the ratio-two cache contains exactly 512 visible rows: its index
source can legitimately select every row without publishing a top-k tensor.
The former gather gate nevertheless requested that tensor and aborted. Gather
now starts only when both compression ratios have more than 512 visible rows.
The old `37c10366…` binary reproduces the abort after a 1,024-token prefill; the
fixed `4608dd0d…` binary passes 36/36 strict `atol=rtol=2e-5` comparisons around
prefixes 1,020/1,023/1,024/1,025, plus 1,033/1,033 comparisons for a one-token
prompt followed by 1,032 individual decode calls. Maximum absolute difference
is 1.699e-6. Production flash attention also completes all 36 boundary cases
with matching greedy tokens, but passes only 12/36 strict elementwise checks
(max absolute difference 0.003875; relative L2 0.001270). These existing flash
kernel differences remain visible in [gather-boundary.json](validation/deepseek41/gather-boundary.json).

```bash
python eng/dsv41-fixture.py BOUNDARY_FIXTURE_DIRECTORY \
  --f32 --cuda-index --index-topk 512 --token-count 1040
TS_DSV4_FA=0 python eng/tests/dsv41-gather.py BOUNDARY_FIXTURE_DIRECTORY \
  --library /absolute/path/to/libGgmlOps.so --backend CUDA --gpus 1 \
  --prefixes 1020 1023 1024 1025 --decode-tokens 8
```

| Evidence | Status / artifact |
|---|---|
| Harness protocol/unit tests | 33 Python tests passed locally and on the requested VM, including the optional serial-tool workflow policy, unchanged default request hashes, strict rejection of premature or wrongly parameterized dependent calls, UTF-8 SSE content, fragmented tool calls, final-tool JSON-mode settings, separation of reasoning from validated assistant content, and matching the explicit tensor-shard count to the GPU-count matrix axis; these do not test model inference |
| Native CTests | Shared-placement build `6b3b5ab3…`: 13/13 passed on the VM, including dedicated 2/4/8-rank TP cases. The earlier compact-gather build passed all 12 then-existing tests after its three insufficient-device skips were rerun with all GPUs visible. Coverage includes CPU/CUDA quantization/candidates, explicit F32-source matmul, scheduler capacity, Engram workers, TP, sparse attention and compact row gathering. [Current placement evidence](validation/deepseek41/shared-expert-placement/README.md), [recovery evidence](validation/deepseek41/native-recovery/README.md), [earlier compact-build provenance](validation/deepseek41/compact-raw-gather/README.md) |
| Activation quantization / candidate mask | CUDA custom-op outputs matched CPU references byte for byte on the tested fixtures |
| Eight-GPU scheduler capacity | VM and local CTests passed creation/allocation/computation with seventeen CPU backend handles; this exercises the 8 accelerator + 8 fused + 1 CPU scheduler capacity without requiring GPUs |
| Sparse decode gather | CPU: 64/64 paired checks passed, maximum absolute difference 1.1921e-6. Eight A40 GPUs, default flash attention: the 64-dimensional-index fixture passed 64/64 at `atol=rtol=2e-5`, max absolute difference 2.2441e-5; the CUDA-native 128-dimensional-index fixture passed 44/64 at that strict bound, max absolute difference 3.1883e-4. Both fixtures had 64/64 matching greedy tokens. See detailed bounds below |
| Opt-in raw-window compaction | CPU 1,200/1,200 comparisons passed. Two/eight-GPU FA0: 924/924 strict comparisons passed; FA1: 208/924 strict comparisons passed, with every argmax matching. Strict FA1 differences remain disclosed. Qualified two-GPU head-512 microbenchmark passed exact-copy and independent attention checks, with 1.19–1.25× speedup for a 512-row ring and 1.89–2.02× for a 1,280-row ring. At an approximately 8k prompt, matched full-checkpoint measurements increased sustained decode from 26.204 to 29.788 tokens/s at concurrency 1, and from 6.562 to 7.458 per request at concurrency 4 (13.7%); prefill was essentially unchanged. Default off. [Complete evidence](validation/deepseek41/compact-raw-gather/README.md) |
| Managed correctness regression | The latest managed stage passed 3651/3651 non-GPU correctness tests both locally and on the requested VM, including Responses audio rejection before media decoding. The final CPU4 benchmark uses the separately verified 3635-test host, and the initial TP baseline uses the 3566-test host. Coverage includes text, media, visual symbol retention, routed-TP configuration, unsupported audio rejection, and reasoning/JSON tool workflows. Exact commands, binaries and TRX hashes: [managed correctness evidence](validation/deepseek41/managed-correctness/README.md) |
| Managed synthetic CUDA regression | Requested VM, GPU 1 visible: 157/157 passed, zero skips/failures (18 s), after compiling PTX for A40 `compute_86`; `/workspace/deepseek41-work/managed-cuda-synthetic-sm86.log` and `test-results/managed-cuda-synthetic-sm86.trx`. Initial 98 kernel-unavailable failures came from the incompatible committed `sm_120` PTX and all cleared with the corrected build |
| Q2_K full checkpoint load and output quality | Final eight-GPU layer placement passes all 138 cases across short/long prompts, JSON/schema, generated history, tools, reasoning, images/video, Unicode and concurrency. Its 15 sustained-decode requests each complete 512 tokens. Final TP passes 129/130 and CPU4 passes 55/57 in their separately scoped plans; default-parallel workflow failures remain. [Final placements](validation/deepseek41/final-placements/README.md) |
| Full Q2_K numerical smoke | Final native produces expected tokens `[22,1]`; traced and untraced logits are bitwise equal. Strict F32-input oracle comparison fails: relative L2 0.146216, maximum absolute error 2.708920, matching top five IDs. Early differences follow Q2_K projections; native quantized activation arithmetic differs from the F32-input reference, without fully attributing the final discrepancy. [Complete result and stage analysis](validation/deepseek41/smoke18-reference/README.md) |
| Single-device / layer split / CPU MoE | Eight-GPU layer placement measured below. The [final CPU4 profile](validation/deepseek41/final-placements/README.md) completed 15/15 sustained decode, 2/2 long retrieval and 10/10 separate serial-tool workflows. Default-parallel quality passed 28/30 versus 29/30 in the historical profile; the additional premature dependent-call failure remains. Median single-request decode increased from 23.9825 to 29.7361 tokens/s. The full checkpoint does not fit on one A40 without substantial CPU offload |
| Routed-MoE tensor parallelism | Requested VM: standalone top-6 MoE suite passed 144 full-weight/sharded comparisons across CPU and 2/4/8 GPUs, covering F32/BF16/Q2_K, uneven quantization-block strips, rotated layer placement, strided routing IDs and changing batch sizes. Four CTests passed, zero skips/failures; `/workspace/deepseek41-work/native-tp-top6-ctest.log` and `native-tp-top6-ctest-detail.log`. Attention remains layer-distributed; reduction uses host staging |
| Mixed routed-MoE quantization preflight | Local CPU: expanded standard suite passed 96/96, including Q2_K gate/up with Q3_K down; exact checkpoint expert dimensions 5120×2304 passed 24/24 across 2/4/8 ranks and top 6 of 8 synthetic experts. Full-shape maximum relative L2 was 1.10e-7, absolute 1.16e-10. Source, commands and per-comparison metrics: [tp-mixed-quantization-local.json](validation/deepseek41/tp-mixed-quantization-local.json). Standard CUDA geometry passed 96/96. At checkpoint dimensions, batch-16 CUDA output exceeded the strict relative bound (up to 4.06e-5). Independently assembled single-device strips exactly matched TP output; the F32-down control passed. [Failures and controls](validation/deepseek41/tp-mixed-quantization.md) remain explicit. The first full-checkpoint TP placement run completed as recorded below; production TP code was unchanged for that run |
| TP complete numerical fixture | Requested VM: independent PyTorch oracle passed 41/41 checks at each of 2/4/8 GPUs (123 total), `atol=rtol=2e-5`, FA/gather disabled. Covers chunked prefill, decode, reset, interleaved slots and rewind; `/workspace/deepseek41-work/fixture-cuda-index/validation-cuda-tp{2,4,8}-strict.json`. The initial b26-native / 3566-test managed full Q2_K TP profile passed 30/30 standard quality cases, 15/15 sustained decode cases and 2/2 long-prompt retrieval cases; reasoning/tool failures and measured throughput are recorded below |
| llama.cpp same-weight reference | Pending compatible runtime and measured artifact |
| Initial HTTP quality | [24/30 passed](validation/deepseek41/http-initial-quality.json): short answers, JSON, JSON schema and generated multi-turn recall passed at concurrency 1/4. Three tool calls exposed plain/mixed parameter-tag parser defects; three agentic answers used Markdown fences. Later parser fixes and explicit JSON-mode workflow reruns are recorded below. These initial diagnostic timings overlap native build/tiny-GPU work and are not qualified benchmarks |
| Long-context HTTP | Qualified warm, single-request retrieval: 6/6 passed all three facts and exact JSON, three repetitions each at 7,706 tokens (41.18–41.40 s) and 30,585 tokens (162.08–162.86 s); [complete results](validation/deepseek41/full-checkpoint/layer8-context65536-ubatch256-cpumoe0-sparse1-warm-long.json). Initial cold/warming diagnostics were 324.19 s and 261.75 s and are not qualified throughput comparisons. Later concurrent long-context measurements and the verified scheduler-capacity fix are recorded below |
| Existing-model regression quality | Final managed 3651 / native 6b3: 75 matched cases across Qwen3, Qwen3.5 and Gemma4 introduce zero new failures against both preserved references. Final passes 39/75 versus HEAD 33/75 and earlier-after 34/75; all 36 remaining failures are retained. Separate Unicode coverage passes 15/15. Independent replay verifies all 90 cases, six warmups and 116 HTTP turn requests. Zero introduced failures describes that paired run only: the later repeated JSON comparison includes matching requests and exposes additional failures under its different warmup/workload order. A separate [12-case chunk control](validation/deepseek41/existing-model-regressions/qwen3-json-chunks/README.md) reproduces the same Qwen3 response change in both builds at matched prefill partitions; the original concurrent chunk traces were not recorded. [Final quality evidence](validation/deepseek41/existing-model-regressions/final3651-native6b3/README.md) |
| Earlier existing-model performance | The earlier-stage controlled 225-case comparison and two alternating Qwen3 reruns completed. Sustained decode ratios were 1.190/0.980, combined paired median 1.007; no newly failed cases. Residual first-pair total-wall ratio 0.916 and tool single-request TTFT ratio 0.910 remain disclosed. No blanket latency/parity claim; [sanitized evidence and telemetry](validation/deepseek41/existing-model-regressions/README.md) |
| Final existing-model JSON comparison | The final six-job comparison completed all 90 timed cases: baseline 20/45 and final 34/45, with one newly failed Qwen3 case. Only the complete Qwen3.5 groups have matched successful outputs and qualified timing ratios. Its c1 first-token latency changed from 65.04 to 72.03 ms and c4 from 258.40 to 276.21 ms; short-response decode improved in that run. The separate [60-case alternating control](validation/deepseek41/json-performance/qwen35-alternating/README.md) passed every answer but reproduced slower c1 first-token latency by 15.43/15.16 ms and request-wall time by 21.62/21.65 ms in its two run orders. These results remain preserved. [Complete r2 evidence](validation/deepseek41/json-performance/completed-r2/README.md) |
| Qwen3.5 latency follow-up controls | A [60-case managed/native swap diagnostic](validation/deepseek41/json-performance/qwen35-cross-phase/README.md) passes every answer and retains higher solo TTFT with final managed in that fixed-order diagnostic; native prefill medians remain 27–29 ms. A [30-case EventPipe diagnostic](validation/deepseek41/json-performance/qwen35-eventpipe/README.md) passes every answer and both trace integrity audits, but does not reproduce the median gap. Its slow solo observations contain longer inherited cache-reset residence in both builds, with no GC-reason suspension overlapping the six solo first-token intervals. The subsequent [72-case uninstrumented control](validation/deepseek41/json-performance/qwen35-solo72/README.md) holds final native fixed and passes all answers and both whole-18 timing comparisons in B/F/F/B order. Baseline/final median TTFT is 67.54/64.36 ms and 69.89/65.38 ms; request wall is 124.22/118.67 ms and 125.15/121.49 ms. All observations and the predeclared first-three/later-fifteen groups remain. This larger control does not reproduce the earlier slowdown. No production change was made, no fix is claimed, and the differing controls do not establish universal absence of regression. |
| Image/video | F32 native vision/mixed-input fixture: 158/158 passed on one GPU, eight-GPU layer placement, and eight-GPU routed TP with one CPU-MoE layer and microbatch three. Tiny BF16 dense/flash suites each passed 18/18 at separately stated bounds. Actual Q2_K media HTTP passed 25/25 requests across concurrency 1/4 (OCR, multiple-image order, image follow-up, video order, timestamp), plus 1/1 image-after-long-text request. The final real-image BF16 encoder comparison has relative L2 0.016584 and exceeds tiny-fixture numerical bounds; HTTP success does not establish strict encoder parity |
| Audio | Unsupported by the official model configuration; four actual HTTP rejection checks passed |

## Q4_K_M throughput work (eight A40s, 2026-09-12)

The Q4_K_M release is 414.2 GiB against 360 GiB of VRAM, so unlike Q2_K it
cannot hold the two 51.5 GiB Engram tables on the devices and it needs some
routed-expert CPU offload. Both facts drove the work below. Every row is a
cold-prompt measurement -- each repeat sends a different prompt body, so its
n-grams select Engram rows that run has not touched -- on the same host, same
checkpoint and same `MAX_CONTEXT` 65536 / `TS_DSV4_UBATCH` 1024 /
`TS_SCHED_MAX_RUNNING_SEQS` 4 profile.

| | before | after |
|---|---:|---:|
| Prefill, 4,924-token prompt (tok/s) | 199.7 / 252.5 / 250.9 | 451.8 / 463.9 / 492.1 |
| Decode, single stream (tok/s) | 23.5 / 25.8 / 25.9 | 31.9 / 31.0 / 32.5 |
| Decode aggregate, 2 concurrent (tok/s) | 24.8 | 39.3 |
| Decode aggregate, 4 concurrent (tok/s) | 24.3 | 48.9 |
| Decode aggregate, 8 concurrent (tok/s) | 26.5 | 48.5 |
| Four concurrent 10.8k-token documents | server aborted | 4/4 answered |
| Routed-expert layers on the host | 3 of 40 | 1 of 40 |

"before" is the same binary with `TS_DSV41_ENGRAM_WARM=0`,
`TS_DSV4_VRAM_RESERVE_MB=5240`, `TS_BATCHED_FUSED_DECODE=0` and
`TS_DSV4_GRAPH_CACHE_HEADROOM_MB=0`, which reproduces the previous behaviour on
one build.

Where the time went, from `TS_DSV4_PERF=2`, per 1024-token prefill chunk and per
decoded token:

| | before | after |
|---|---:|---:|
| Prefill input preparation | 1,435-2,565 ms | 56-81 ms |
| Prefill graph compute | 2,326-2,750 ms | 2,139-2,154 ms |
| Decode input preparation | 5-15 ms | 0.86-0.95 ms |
| Decode graph compute | 36-39 ms | 35.6-35.9 ms |
| Scheduler splits per prefill chunk | 14 | 10 |

Input preparation is the host Engram lookup, and it is what automatic warming
removes. Graph compute and the split count move with the routed-expert offload
count, which the smaller device-memory reserve cut from three layers to one.

### What the decode step is made of

`TS_DSV4_PERF=3` reports how a device subgraph is submitted. A single-token
V4.1 decode graph is 3,235-3,352 nodes, 2,194-2,269 of which compute something
(~55 a layer), submitted as 59 device subgraphs, 331-337 forwarded runs of
ordinary nodes and 400-406 fused kernel launches. The submitting thread spends
3.6-3.7 ms doing that and then waits ~31 ms, so the step is bound by the device,
not by submission.

Against that, one token's weights are 9.78 GiB at Q4_K_M -- 128.6 MiB of routed
experts, 21.4 MiB of shared expert and 76.5 MiB of attention per layer, plus a
543 MiB output head -- which is 14.0 ms at the A40's 696 GB/s. The remainder is
the floor under ~2,200 small kernels, which is why token-batched decode helps
and why raising single-stream decode further needs fewer kernels rather than
fewer bytes.

### Token-batched decode equivalence

Solo decode reproduced its own greedy output on 6/6 prompts at 200 tokens. A
four-wide batched step matched the solo text on 2-3/6, diverging mid-answer and
continuing coherently. The discriminating control is batch WIDTH: the same
prompt at width 2 and at width 4 uses the same code path with identical per-slot
wiring and only wider GEMMs, and those two disagree on 1/4 prompts -- the same
rate. Output therefore depends on which requests share a step.

Per-slot state separation is checked separately and is exact: four concurrent
10,836-token documents, each hiding a different secret, were answered 4/4 with
their own secret and 0/4 containing another slot's secret, both serially and
under batched decode.

### Routed-MoE tensor parallelism on this checkpoint

`TS_DSV41_TP=8` shards the routed experts evenly, which at Q4_K_M removes the
capacity cliff entirely: 38.3 GiB of shards a rank and **zero** CPU-offloaded
layers. It is still slower than layer split, because attention, the shared
expert and the caches keep their layer placement and the partial sums reduce
through host-staged F32 buffers on a box whose GPUs have no NVLink and straddle
two NUMA nodes.

| eight A40s, Q4_K_M | Prefill tok/s | Decode tok/s | CPU-MoE layers |
|---|---:|---:|---:|
| Layer split | 451.8-492.1 | 31.0-32.5 | 1 |
| `TS_DSV41_TP=8` | 391.9-410.4 | 21.4-22.0 | 0 |

Decode splits rise from 62 to 132 per step under TP. This reproduces the Q2_K
conclusion on a checkpoint where TP has a genuine placement advantage, so it is
the placement-independent result: on this topology the host-staged reduction
costs more than the CPU-MoE layer it removes.

## Full-checkpoint warm measurements

The isolated Q2_K run used eight A40 GPUs, layer placement, F16 KV caches,
65,536-token allocation, microbatch 256, no CPU expert offload, opt-in sparse
attention, and warmed Engram pages. All 30 short/decode checks passed across
three repetitions at concurrency 1/4. Every sustained-decode request emitted
512 tokens. Greedy sampling and complete binary hashes are recorded in the
[launch manifest](validation/deepseek41/full-checkpoint/layer8-context65536-ubatch256-cpumoe0-sparse1-warm-launch.json),
[request results](validation/deepseek41/full-checkpoint/layer8-context65536-ubatch256-cpumoe0-sparse1-warm-steady.json),
and [GPU telemetry](validation/deepseek41/full-checkpoint/layer8-context65536-ubatch256-cpumoe0-sparse1-warm-steady-telemetry.json).

| Measurement | Concurrency 1 | Concurrency 4 |
|---|---:|---:|
| Short-answer median time to first token | 248.9 ms | 1,116.3 ms |
| Sustained-decode median time to first token | 322.7 ms | 1,476.0 ms |
| Median decode rate per request | 34.734 tokens/s | 8.378 tokens/s |
| Four-request aggregate end-to-end rate | — | 32.78 tokens/s |

The aggregate rate divides 2,048 generated tokens by the median four-request
wave duration of 62.48 seconds; it includes first-token latency. Model loading
placed approximately 186.3 GiB on GPUs in 228.5 seconds, followed by 60.08 GiB
of Engram page warming in 142.82 seconds. Those startup durations are
diagnostic and are excluded from the warm throughput measurements.

The earlier [quality diagnostic](validation/deepseek41/full-checkpoint/layer8-context65536-ubatch256-cpumoe0-sparse1-warm-quality.json)
passed its 20 short, JSON, JSON-schema and multi-turn checks. Ten tool/agentic
checks failed because the server rejected final-turn JSON mode while tool
definitions remained present. That API defect was fixed; the later structured
workflow suite passed 30/30, as described below. In the
[unconstrained workflow diagnostic](validation/deepseek41/full-checkpoint/layer8-context65536-ubatch256-cpumoe0-sparse1-warm-unconstrained-tools.json),
weather tool round trips passed 5/5; agentic tasks passed 2/5, with three
Markdown-fenced final answers failing the requested raw-JSON contract.
These diagnostic runs overlapped preparation work and are not timing evidence.
No compatible same-weight llama.cpp run exists yet, so none of these results
establishes llama.cpp quality or performance parity.

## Later full-checkpoint measurements

The `layer8-context65536-ubatch1024-cpumoe0-sparse1-compact1-slots4-b26c`
profile uses the same eight A40 GPUs and Q2_K weights, microbatch 1024,
compact raw-window gather, sparse flash attention, four scheduler slots,
and a metadata pool sized for four complete contexts. Native SHA-256 is
`b26cac3e40ff67b6de077063cd7a3c68e683220f0bc60237edf728ec6d218f1f`.
These are qualified, sequential measurements without competing diagnostic
GPU work. Full requests and raw responses remain on the VM; the
[portable reports](validation/deepseek41/full-checkpoint/) retain request
hashes, responses, failures, launch hashes, telemetry and execution plans.

| Measurement | Earlier microbatch-1024 profile | Compact gather / four-context pool |
|---|---:|---:|
| 8k-prompt sustained decode, concurrency 1, median of 3 | 26.204 tokens/s | 29.788 tokens/s |
| 8k-prompt sustained decode, concurrency 4, median per request | 6.562 tokens/s | 7.458 tokens/s |
| Four 8k-prompt, 512-token completions, whole wave | about 188 s | 178.39 s |
| Four 30,585-token retrieval requests, whole wave | 549.40 s | 358.19 s |
| 61,095-token retrieval, single request | not measured | 174.89 s, exact facts and JSON |

Both sustained-decode profiles completed all seven requests with 512 output
tokens each. The 13.7% decode improvement isolates raw-window compaction:
prefill remained approximately 19.8 seconds at concurrency 1. The concurrent
32k improvement additionally fixes scheduler capacity. The earlier pool
preempted ten times and recomputed 60,717 prompt tokens. The corrected pool
forwarded exactly the submitted prompts plus warmup, with zero preemptions;
it does not preallocate four complete physical GPU caches.

This stage passed 30/30 short/JSON/schema/multi-turn/structured-tool workflows,
4/4 blocking protocol checks, and 3/4 reasoning checks. A weather workflow
previously ended after 1,536 reasoning tokens without a final answer. It now
commits the normal closing token and returns the correct final JSON. The
remaining reasoning-agent workflow hit repetitive reasoning; its subsequent
fix has separate managed tests and awaits full-checkpoint verification.
Unconstrained tool/agent answers passed 7/10: malformed tool markup and
Markdown-fenced final JSON remain recorded failures. The later tool grammar
is a separate build and must be verified before crediting it with a fix.

Concurrent retrieval passed 7/8 strict checks; all eight contained the right
facts, but one 8k response used Markdown fences. Multilingual arithmetic
passed 5/5. Exact Unicode JSON passed 0/5 because the generated emoji field
contained the Chinese word for rocket. Diagnostic controls returned the
correct emoji without JSON mode and with a constant JSON schema; an English
JSON-mode prompt produced U+FFFD. An actual-vocabulary probe against that
exact server Runtime reproduced rejection of all four literal/split emoji
paths while allowing escaped-surrogate paths. The updated partial-UTF-8 mask
passes all seven valid paths; full-checkpoint verification follows in a later
server stage. Separate negative probes also exposed acceptance of malformed
UTF-8, which has its own decoder regression tests. The final actual-vocabulary
check passed 16/16 and the focused grammar lane passed 114/114; exact before/after
token masks and runtime hashes are in the [Unicode validation evidence](validation/deepseek41/json-unicode/README.md).

Media passed 25/25 across concurrency 1/4 and one additional image after
approximately 8k text tokens. Single-request times ranged from 2.36 to
3.57 seconds across OCR, multiple images, follow-up, video order and video
timestamps. Four-request waves ranged from 8.63 to 16.77 seconds; the long
image request took 21.14 seconds. These are fixture tasks with frame-sampled
video, not broad visual benchmark scores or strict vision-encoder parity.

## Native request recovery and CPU thread control

Native build `3b885898…` fixes a TP error state that could reject healthy later
requests and a partially constructed graph cache entry that could crash on
retry. The VM passed 230/230 CUDA text/image slot and oracle checks, 18/18
standalone CUDA recovery checks, and 18/18 actual loader thread-count checks.
Isolated pre-fix source controls reproduce both recovery defects. The
[recovery evidence](validation/deepseek41/native-recovery/README.md) records
commands, hashes and limits.

The DeepSeek fallback/host-expert CPU pool now honors the existing native CLI
override. Positive inherited `TS_CPU_MOE_THREADS` takes precedence, followed
by the CLI override and then the existing automatic count. This avoids relying
on .NET managed environment changes reaching native `getenv`. Pool settings
are chosen at model load; the separate logical device used by CPU-only
inference retains its original loader thread argument.

## Full-checkpoint routed-MoE TP

The first qualified TP profile is
`tp8-context65536-ubatch1024-cpumoe0-sparse1-compact1-slots4-chunk1024-b26c-tools`.
It uses native `b26cac3e…` and the separately verified 3,566-test managed host.
Every rank holds 22.906 GiB of routed-expert shards. Attention and shared-expert weights
occupy devices 0–5; devices 6–7 participate in routed experts. The profile sets
both scheduler prefill chunk limits to 1,024, microbatch 1,024, four slots,
65,536 tokens per slot, sparse flash attention and compact raw gather.

| Check | Result |
|---|---|
| Standard short/JSON/schema/multi-turn/tool/agent workflows, concurrency 1/4 | 30/30 |
| Sustained 512-token decode, three repetitions at concurrency 1/4 | 15/15 |
| Single-request decode median | 15.417 tokens/s; 33.49 s whole request |
| Four-request decode median | 3.709 tokens/s per request; 139.28 s median whole wave |
| 7,706-token retrieval | exact facts and JSON, 36.66 s |
| 30,585-token retrieval | exact facts and JSON, 139.70 s |
| Multilingual arithmetic and exact Unicode JSON | 10/10, including all five emoji cases |
| Blocking protocol | 4/4 |
| Thinking protocol | 3/4 |
| Required/named/none/serial/parallel tool-policy suite | 29/30 |

The independent [completion and accounting audit](validation/deepseek41/placement-first-tp/README.md)
reconciles all 95 cases, both failures, warmup, prefix reuse and EOS forwards.
The placement runs had zero preemptions. This initial TP profile was slower
than the measured layer profile; sharding alone does not establish a speedup.
The CPU fallback thread setting is being tested separately. CLI thread flags
and the native process environment must not be conflated: the target .NET
runtime's managed environment updates do not change native `getenv`.

Early repetition closure worked in the arithmetic case: 643 reasoning tokens,
a normal closing token, and the correct answer. The weather final turn also
closed its 1,536-token reasoning budget and returned the correct JSON. The
agent workflow and named-thinking policy case instead emitted incorrect
parameter/invoke closing tags inside unfinished raw string arguments. Replay
confirmed that the grammar stayed active and the parser correctly withheld
incomplete calls. The [serialization replay](validation/deepseek41/tool-serialization-replay/README.md)
retains these failures. A subsequent tag-family restriction has managed and
actual-vocabulary coverage. The final 3,651-test managed / `6b3b5ab3…` native
TP run passes both cases, all 30 tool policies and all four thinking cases;
this verifies the final build without isolating the grammar change.

Startup is diagnostic: loading overlapped managed builds and a sequential
read of all 196,765,286,400 routed-expert bytes (23.425 s). The native loader
reported 186.3 GiB in 442.0 s, followed by Engram preparation. A short process
sample observed 5,323 major faults in five seconds with rank workers waiting
for file pages and GPUs at 0–1% utilization. No startup speedup is established.
The profile's launch, source warming, complete results, failures, native
forward accounting and summarized telemetry are retained in
[full-checkpoint evidence](validation/deepseek41/full-checkpoint/).

## TP worker synchronization optimization

Candidate native `8e66aaf4…` combines each rank's submission and queue wait
within one worker pass. Independent rank queues still execute concurrently;
each rank drains its queue on success or error, and the host reduces outputs
only after every worker finishes. This removes a second worker wakeup/join
per routed-expert call. The prior implementation remains available only as
a test-build control.

An exclusive same-binary comparison on the requested VM used the checkpoint's
5,120×2,304 expert dimensions, Q2_K gate/up and Q3_K down, 30 alternating pairs
per shape, and a one-thread enclosing CPU backend. All 180 old/new pairs were
bitwise identical. Median old/new speedups were:

| TP ranks | One token | 16 tokens |
|---|---:|---:|
| 2 | 1.061× | 1.025× |
| 4 | 1.183× | 1.051× |
| 8 | 1.199× | 1.045× |

The eight-rank one-token median changed from 0.282318 to 0.235547 ms per
synthetic MoE call, with the candidate faster in 29/30 pairs. Candidate CUDA
recovery passed 63/63 checks, text/image TP slot-oracle coverage passed 230/230,
and native CTests passed 13/13. These are isolated kernel and correctness
results; full-model timing must determine the end-to-end gain. The
[candidate evidence](validation/deepseek41/tp-single-fanout/README.md) records
commands, hashes, scope and per-shape results.

## CPU thread tuning experiment

The explicit one-thread TP experiment passed all 30 standard quality cases.
Its 50 returned turn texts, tool functions and token counts matched the first
TP baseline. Nevertheless, the same 10,330 native prefill tokens took 317.516 s
instead of 49.013 s; the same 1,318 decode forwards took 150.246 s instead of
100.034 s. The quality suite took 472.31 s versus 151.8 s. A transport-finalization
`FlushAsync` cancellation appeared after a completed response; the report does
not claim an error-free server log.

The remaining broad one-thread experiment was stopped. A separate, complete
three-repetition single-request control produced the same 512-token outputs
as the prior TP run: median decode fell from 15.417 to 10.083 tokens/s
(−34.6%), TTFT rose from 345.6 to 1,569.7 ms, and whole-request time rose
from 33.49 to 52.25 s. This control had no errors or preemptions. Sampled GPU
clocks matched and no competing compute process was observed.

Only the completed 30-case quality suite and three-case control are credited.
The original 15-case decode report stopped after one completed case and a
cancelled subsequent request; its other planned phases were not run. The
[thread experiment report](validation/deepseek41/thread-experiment/README.md)
retains that incomplete evidence and all comparison limits: Runtime changed
from the 3,566-test to 3,597-test build, and the earlier native pool width of
32 is inferred from source and the environment probe, not directly measured.
This experiment does not justify choosing one worker. It used the old
shared-expert placement code and does not measure thread tuning after the
placement fix described below.

## Shared-expert device placement correction

A scheduler audit found that the intended GPU placement of shared-expert
weights did not ensure GPU execution. In TP and CPU-MoE graphs, the routed
CPU boundary precedes unpinned shared gate/up matrix operations. Their
CUSTOM-only fused activation backend blocks ordinary GPU placement
propagation, allowing the scheduler to place those matrix operations on CPU.
This is a concrete explanation for the earlier CPU-thread sensitivity, not a
cost inherent to routed-expert tensor sharding.

Native `6b3b5ab3…` explicitly pins all three shared-expert projections to the
V4.1 layer device. This also covers CPU-MoE layers: their routed experts stay
on CPU while the shared expert stays on its layer device. Other architectures
and global buffer-placement rules are unchanged. CPU-only inference retains
its CPU layer device.

The [placement evidence](validation/deepseek41/shared-expert-placement/README.md)
uses a same-binary legacy control to observe the scheduler's actual assignments on
a five-layer CUDA fixture at three prefill/decode shapes. Legacy TP places
five shared gate and five shared up projections on CPU per shape; CPU-MoE1
places one gate and one up there. Shared down projections and the resident
layer control are already on their intended devices. Mixed TP plus CPU-MoE
reproduces the same five gate/five up misplacements. The fixed cases have
zero misplaced projections. All 597 placement and independent-logit checks
pass, with relative error below 9.1e-7; CUDA recovery passes 230/230 and native
CTests pass 13/13. These are directly observed fixture counts. Applying the
same graph mechanism to the old full-model runs is an inference; their exact
shared-operation backend counts were not captured before those hosts stopped.

The earlier full-checkpoint TP and CPU-MoE measurements therefore remain
historical baselines with this defect. The [final placement measurements](validation/deepseek41/final-placements/README.md)
complete 55/57 CPU4 and 129/130 TP inference cases. Single-request sustained
decode is 29.7361 tokens/s for CPU4 and 19.5159 tokens/s for TP8; four-request
per-request medians are 7.1519 and 4.9986 tokens/s. Both profiles introduce one
additional premature dependent-tool-call failure compared with their earlier
standard quality suites. Separate serial-policy workflows pass 10/10 on each
profile and do not erase these failures. Final TP tool-policy, thinking,
blocking, multilingual and media suites all pass. The final CPU0 layer run
passes all 138 cases, including all eight concurrent long prompts with exactly
153,187 prefill tokens and zero preemptions. Its sustained decode medians are
34.8274 tokens/s for one request and 8.4580 per request at concurrency four;
the four-request whole-wave throughput is 33.1292 tokens/s. Single-request
7,706/30,585-token TTFT is 19.985/80.240 seconds. Full reports, binary hashes,
historical comparison limits and accounting are linked above. Compatible
llama.cpp quality/performance parity remains unverified.

## Sparse flash attention experiment

`TS_DSV41_SPARSE_FA=1` uses the vendored CUDA sparse flash-attention kernel
through `ggml_flash_attn_ext_set_n_kv_max`. Each V4.1 query has at most 128
finite raw-window entries and 512 selected compressed entries, giving the
required bound of 640. The opt-in hint applies to a single query or at least
16,384 cached keys; the backend checks its supported geometry and device.
The default remains off; the qualified full-checkpoint profiles below explicitly enable it. This reduces
attention arithmetic without materializing a separate KV tensor per query;
prefill still transfers full shared caches between layer devices.

The standalone test uses the checkpoint's 512-component heads, 64 query heads,
one shared F16 K/V head, and sinks. It covers batches of 1, 17 and 256 queries,
4k/8k/16k/32k caches, causal visibility, candidate holes, finite mask biases,
single-key and fully masked rows. Every output is compared between CUDA paths
and with an independent CPU softmax using original F32 queries, exactly decoded
F16 keys, and double-precision accumulation. Reference loops visit only finite
keys. CPU reference work and warmup are excluded from timing.

```bash
CUDA_VISIBLE_DEVICES=0 ctest --test-dir TensorSharp.GGML.Native/build \
  -R '^deepseek41-sparse-flash-attention-cuda$' --output-on-failure
# Only run timing while other GPU and storage benchmarks are idle.
CUDA_VISIBLE_DEVICES=0 TensorSharp.GGML.Native/build/GgmlOpsDsv41SparseFaTest --bench
```

Initial complete-reference A40 measurements, nine alternating repetitions:

| Query batch | Cached keys | Dense median | Sparse median | Dense / sparse |
|---:|---:|---:|---:|---:|
| 1 | 4,096 | 73.415 us | 43.099 us | 1.70x |
| 1 | 8,192 | 96.442 us | 42.410 us | 2.27x |
| 1 | 32,768 | 286.860 us | 58.834 us | 4.88x |
| 17 | 4,096 | 252.282 us | 491.810 us | 0.51x |
| 17 | 8,192 | 430.017 us | 493.197 us | 0.87x |
| 17 | 32,768 | 1,510.256 us | 508.961 us | 2.97x |
| 256 | 4,096 | 2,553.275 us | 4,814.325 us | 0.53x |
| 256 | 8,192 | 4,956.252 us | 4,837.069 us | 1.02x |
| 256 | 32,768 | 19,379.012 us | 4,872.875 us | 3.98x |

These are attention-operator measurements, not model throughput. In those nine
shapes sparse attention had lower relative error against the CPU oracle in
every case. Its maximum absolute error was 1.866e-4 and maximum relative L2 was
7.797e-4. Existing dense attention reached 4.806e-4 and 1.940e-3 respectively.
The initial test therefore failed its dense-baseline envelope; it did not
identify a new sparse-path accuracy failure. The final regression check keeps
separate, explicit F16 envelopes: sparse `atol=3e-4, rtol=1.5e-3` with relative
L2 below 1.5e-3; dense uses twice those limits. Pairwise differences use the
dense envelope. Both modes retain F32 accumulation but internally round F16
MMA operands/probabilities. Full-checkpoint effects still need evaluation.
Raw evidence: `/workspace/deepseek41-work/native-sparse-fa-full-reference-benchmark.log`.

The added 16,384-key boundary also passed the full reference at all three batch
sizes. Diagnostic dense/sparse ratios were 3.31x (one query), 1.58x (17), and
2.01x (256); this run overlapped native compilation, so its timings are not
qualified. The largest sparse error in the expanded matrix was 1.933e-4.
Artifact: `/workspace/deepseek41-work/native-sparse-fa-16k-crossover.log`.
The expanded 12-shape test passed in the combined CTest suite. It compares
35,913,728 output values per CUDA path with the independent oracle.

Everything above describes ggml's flash-attention kernel with the
`ggml_flash_attn_ext_set_n_kv_max` hint, which is what V4.1 on CUDA ran when
this experiment and the full-checkpoint profiles (including the 19.985/80.240 s
long-prompt TTFT) were recorded with the initial V4.1 support (`3347b06b`). V4.1
on CUDA now runs TensorSharp's owned F32 attention instead, whose sparse prefill
is a different kernel with a different gate, described next.

### Owned F32 sparse prefill (default)

The owned attention (`tsg_attention_f32_on_backend`) gives a launch a sparse
capacity of `n_swa + indexer_top_k` = 640 keys when it has **more than 8 queries
and at least 8,192 keys** (`tsg_dsv41_owned_sparse_capacity` in
`ggml_ops_precision_policy.h`). A mask compaction lists each query's finite keys
in order and one F32 online-softmax partition per row attends to them; a row
with more finite keys than the capacity scans every key instead. One to eight
queries -- decode and every DSpark verify -- keep the split-key kernel, so a
verify still commits exactly the rows single-token decode would; shorter keys
keep the tiled SGEMM kernel. The gate is on by default; `TS_DSV41_SPARSE_FA=0`
restores tiled prefill. The ggml flash-attention hint above keeps its own gate
(one query or at least 16,384 keys) and stays opt-in with `=1`.

Before this change the owned kernel was sparse only with `TS_DSV41_SPARSE_FA=1`
and above 4 queries, and it partitioned a sparse row into up to 16 key splits
when fewer than 512 query rows shared the launch. A sparse row now always has
one partition, so its arithmetic no longer depends on the launch width; every
V4.1 launch that took the sparse kernel before (64 heads, so at least 576 rows)
already had one.

Measured on one A40 (GPU 2 of the 7x A40 VM, load average 16-26 from other
tenants), 64 heads, head width 512, 640 visible keys per query, each arm checked
against a decomposed F32 reference with `atol = rtol = 6e-6`:

```bash
GgmlOpsCudaAttentionPrecisionTest --benchmark-dsv41-prefill QUERIES KEYS 64 REPEATS
TS_DSV41_SPARSE_FA=0 GgmlOpsCudaAttentionPrecisionTest --benchmark-dsv41-prefill QUERIES KEYS 64 REPEATS
```

| Queries | Keys | Default (kernel) | `TS_DSV41_SPARSE_FA=0` (kernel) | Default max abs / rel L2 |
|---:|---:|---:|---:|---:|
| 512 | 8,960 | 34.4 ms (sparse) | 106.8 ms (tiled) | 1.4e-7 / 7.5e-7 |
| 512 | 33,536 | 34.1, 34.3 ms (sparse) | 1,548.5, 1,547.2 ms (tiled) | 1.1e-7 / 7.4e-7 |
| 512 | 66,304 | 34.8 ms (sparse) | 6,864.6 ms (tiled) | 1.4e-7 / 7.5e-7 |
| 1,024 | 33,536 | 68.2 ms (sparse) | 3,059.5 ms (tiled) | 1.4e-7 / 7.4e-7 |
| 6 | 33,536 | 9.23 ms (split-key) | 9.25 ms (split-key) | 5.6e-8 / 5.6e-7 |

Medians of 5 (default) and 3 (`=0`) samples after 3 warmups; the 33,536-key
row was run twice, alternating arms. The tiled kernel's maximum error was
8.9e-8 or less. The last row is a DSpark-verify-width launch: both settings
select the same kernel.

`attention_precision_test` (CTest `cpu-explicit-f32-attention`,
`cuda-explicit-f32-attention`) adds two checks. A gate check: capacity 0 for 1-8
queries at any key count, 640 for 9+ queries at 8,192 keys, 0 at 8,191 keys and
0 with the variable `0`. A sparse invariance check at 8,960 keys and 64 heads:
query 0 alone, inside 9 queries and inside 512 queries is bit-identical (head
width 512 on CUDA, 8 on the CPU reference path). The same query through the
decode kernel differs in 31,198 of 32,768 outputs on CUDA (max abs 5.2e-8),
which is why verify widths stay on the decode kernel. Without the one-partition
rule the CPU check fails (query 0 of 9 differs from query 0 alone in 221 of 512
outputs).

On the deterministic V4.1 text fixture on one CUDA A40
(`eng/tests/dsv41-inference.py --long-sparse-tokens 8500 --long-sparse-context
16384`, 256-token vocabulary, ubatch 32), an 8,500-token prompt reaches 16,640
attention keys. Default against `TS_DSV41_SPARSE_FA=0`, each a fresh load: the
prefill logits and 32 greedy decode steps differed by at most 1.19e-6 (atol
2e-5), the logits were not bit-identical (the sparse kernel ran), and the 32
greedy tokens were identical. Below 8,192 keys the fixtures are byte-identical
to the `06665adc` library: 1,341 logit rows each for CPU, CUDA and CUDA with
`--cpu-moe 2`, and every native array of the DSpark fixture (155 arrays over
prefixes 1/5/17 plus the state checks). Both libraries miss the same
`draft_confidence_prefix5` tolerance (max abs 1.03e-4), so that miss predates
this change.

These are operator and fixture measurements; the full-checkpoint time to first
token with this default has not been measured yet.

## Independent numerical reference

`eng/dsv41-reference.py` evaluates the GGUF weights with independent PyTorch
equations. It memory maps all shards and dequantizes selected rows and expert
matrices, including Engram, delayed hyper-connections, ratio-2 and ratio-1
compression, shared caches, sparse indexing, candidate pruning, and MoE. Its
default cache mode reproduces the model's BF16 boundaries and FP8/FP4 cache
quantization. Dense arithmetic is F32. It does not reproduce every backend's
internal input rounding or the original FP8 checkpoint's matrix kernels.

The deterministic five-layer fixture is deliberately much smaller than the
published checkpoint. It covers both compression ratios, shared KV/index
sources, Engram, sparse selection, candidate filtering, raw-window rollover,
and routed/shared experts. `--f32` stores the same dequantized fixture weights
as F32 to separate architecture from quantized-matmul input rounding.
`--cuda-index` uses index head dimension 128 and 32 index heads, exercising the
CUDA index kernel; the smaller default index geometry uses its CPU fallback.

```bash
# Dependencies: numpy, torch, gguf, tokenizers.
python eng/dsv41-fixture.py /tmp/dsv41-fixture --f32
python eng/dsv41-fixture.py /tmp/dsv41-cuda-index --f32 --cuda-index
python eng/tests/dsv41-inference.py /tmp/dsv41-fixture \
  --library /absolute/path/to/libGgmlOps.so --backend CPU
TS_DSV4_FA=0 TS_DSV4_GATHER=0 python eng/tests/dsv41-inference.py \
  /tmp/dsv41-fixture --library /absolute/path/to/libGgmlOps.so \
  --backend CUDA --gpus 1 --report /tmp/fixture-cuda-f32-attention.json
```

Each inference run now checks 79 outputs: full prefill, individual decode,
3-token and 5-token chunks, reset with a different prompt, and two interleaved
sequence slots, plus continuation after rejected rewind, plus 25 partial-KV-reuse
checks (below). V4.1 rejects the speculation-only rewind
without changing its position or compressor state; that regression check
passes with the others at `atol=rtol=2e-5`. JSON reports
retain individual errors, greedy-token agreement, tolerances, and relevant
environment variables. `--report` preserves separate precision-mode results.
Relative L2 is `norm(native-reference) / norm(reference)`; passing the
elementwise check requires `abs(error) <= atol + rtol*abs(reference)`.

### Partial KV reuse (conversational rewind)

`TSGgml_Dsv4Truncate` moves a slot's head back so a multi-turn chat keeps the
prefix its re-rendered prompt still matches. Three kinds of check cover it, and
they are separated on purpose because they answer different questions:

* **The checkpoint must be indistinguishable from the live rings.**
  `truncate_checkpoint_matches_live_*` performs the same rewind twice - once while
  the live rings still hold the window, once after decoding far enough past the
  prompt that only the prompt-boundary checkpoint can serve it - and compares the
  two continuations. Same retained prefix, same refill, so any discrepancy is the
  restore. Measured **bit-identical (0.0)** on CPU and on CUDA, for both fixtures.
* **The rewound continuation must be the oracle's answer.**
  `truncate_live_*` compares against `eng/dsv41-reference.py` at the run's
  tolerance, from several heads and depths, including a rewind to zero (which is a
  reset, so it also proves the Engram token history goes with it).
* **The rewind must not move the answer further than a chunk boundary already
  does.** `truncate_live_*_vs_cold` compares against this backend's own cold
  prefill of the same tokens, with a self-calibrated bar: the run first measures
  its own whole-prompt-versus-chunked spread (`cold_chunk_spread_*`) and requires
  the rewind to stay inside it. The retained prefix is computed inside a larger
  forward in one case and as its own call in the other, which on CUDA differs by
  up to 9e-3 for reasons that have nothing to do with the cache path.

Refusals are checked too, and each must leave the sequence usable: a misaligned
target, a target past the head, a negative target, and a rewind deeper than any
reachable state (which is the correct answer, not a failure - the dropped
positions' K rows are gone and recomputing one needs its own equally-gone
window). Every one of the 54 pre-existing checks is **bit-identical** before and
after the change, on CPU and CUDA and on both fixtures, so the truncation support
costs nothing numerically.

What the three kinds of number say together, on the CUDA-index fixture whose
512-wide shared head matches the released checkpoint's:

| quantity | FA on | FA off | what it is |
|---|---:|---:|---|
| `truncate_checkpoint_matches_live_*` | 0.0 | 0.0 | the rewind mechanism itself |
| `truncate_live_16_to_8_vs_cold` | 8.9e-3 | 0.0 | a rewound prefix vs a chunk-matched cold prefill |
| `cold_chunk_spread_16_at_8` | 3.6e-2 | 1.9e-6 | **no truncation at all**: one whole-prompt call vs two |

Read down the FA-off column: the rewind is arithmetically EXACT - restoring the
checkpoint and continuing reproduces a cold prefill of the same tokens bit for
bit. Read across: with flash attention on, the same rewind differs by 8.9e-3,
while merely splitting a prefill into two calls differs by 3.6e-2 with no
truncation involved at all. The sensitivity is the flash-attention kernel's
dependence on batch shape, it is four times larger for plain chunking than for a
rewind, and the rewind contributes none of it. Every one of these numbers is 0.0
on CPU.

The consequence worth stating plainly: greedy decoding can fork on differences of
that size, so a reused turn's text is not expected to be byte-identical to a
re-prefilled one. That is a property of prefix reuse in general - the
pure-extension reuse that `--think`-off and tool turns have always had included -
not of the rewind.

| Fixture / execution | Elementwise result | Maximum absolute error | Maximum relative L2 | Greedy agreement |
|---|---:|---:|---:|---:|
| Default fixture, local CPU | 41/41 at `atol=rtol=2e-5` | 5.1633e-6 | 1.3735e-6 | 41/41 |
| Default fixture, VM CPU | 40/40 at `atol=rtol=2e-5` | 5.0366e-6 | 1.4291e-6 | 40/40 |
| Default fixture, VM CUDA, F32 attention fallback | 41/41 at `atol=rtol=2e-5` | 5.4390e-6 | 1.5317e-6 | 41/41 |
| CUDA-index fixture, VM CUDA, F32 attention fallback | 41/41 at `atol=rtol=2e-5` | 7.3910e-6 | 2.3932e-6 | 41/41 |
| Default fixture, VM CPU, with partial-reuse checks | 79/79 at `atol=rtol=2e-5` | 5.1410e-6 | — | 79/79 |
| Default fixture, VM CUDA, F32 attention fallback, with partial-reuse checks | 79/79 at `atol=rtol=2e-5` | — | — | 79/79 |
| CUDA-index fixture, VM CUDA, F32 attention fallback, with partial-reuse checks | 74/79 at `atol=rtol=2e-5` | 6.5920e-5 | 2.5060e-5 | 79/79 |

The five CUDA-index rows outside tolerance are the pre-existing position-24
`chunk_*` / `interleaved_*` prefill checks at 6.6e-5, all with the reference's
greedy token; no partial-reuse check is among them. Those five are bit-identical
to a build from `HEAD`, as are the other 49 pre-existing checks, on both backends
and both fixtures.

The earlier CUDA failures in 16-token prefill were traced to cuBLAS dispatch:
`Sgemm` ignored the requested pedantic `GemmEx` computation mode. Engram injection
then differed by relative L2 4.17e-5, changing a following FP8 cache bin by 0.125
and amplifying the later logit error. A targeted 128-input, 1280-output GEMM
reproduced this source-precision violation independently: explicit F32-source
error was 0.004238 absolute and 2.59e-4 relative L2. Routing that request through
pedantic `GemmEx` reduced the error to 2.843e-6 and 1.10e-7, respectively, and
resolved both full-fixture prefill failures. These CUDA inference checks use
`TS_DSV4_FA=0 TS_DSV4_GATHER=0`; they do not validate default flash arithmetic.

Default flash attention has an additional, understood source of amplification:
it narrows attention inputs. With precise F32 projections, layer-0 query error
is about 4.05e-7 relative L2 and raw-cache values agree exactly, but the attention
output differs by 3.28e-4. At token 12 the fixture's second and third routed
expert scores are approximately 0.7372 and 0.7371. That small change can switch
the second expert, producing a large later logit difference. Consequently the
default-flash fixture is not a passing strict-F32 oracle comparison. Greedy
agreement alone does not establish model-quality parity.

An upstream CUDA precision omission was fixed and tested separately:
`ggml_prec_set_src(node, GGML_PREC_F32, 1)` now prevents TF32 or half conversion
of that input in ordinary/indexed matrix products. Accumulator precision alone
still permits lower input precision, as specified by ggml's API. V4.1 requests
full source precision for F32 weight projections and its non-flash attention
fallback; quantized weight projections retain their normal optimized paths.
The microtest covers F32 and F16 weights, ordinary and indexed products, and
batch sizes 1, 5, 16, and 31, plus wide Engram projections at 16 and 31 tokens,
against independent double arithmetic. All 18 full
source-precision cases pass (maximum absolute error 3.73e-6, maximum relative L2
1.59e-7); default-precision cases also execute and report
their error and timing. See `precision-wide-before.log` and
`precision-wide-after.log`. The tested native library SHA256 is
`f569e14fcea1bcaa706295bd1896d826e8d9d5f6c75eb2292e8b38eda8153932`.
Portable per-check results and the final prefill stage errors are recorded in
[`numerical-oracle.json`](validation/deepseek41/numerical-oracle.json). The fixed
Engram projection has relative L2 2.25e-7; its injection has 7.15e-7, and all five
raw-cache outputs exactly match the reference quantized values.

The separate eight-GPU gather test uses production flash attention and compares
two native model loads, with gather disabled and enabled. It checks 64 outputs
after several prefill sizes and across the sparse visibility threshold. For
the default fixture, all 64 pass `atol=rtol=2e-5`, maximum relative L2 6.72e-6.
For the CUDA-index fixture, 44/64 pass that same bound, maximum absolute error
3.19e-4 and relative L2 8.06e-5; all 64 greedy tokens agree. These small differences
are consistent with changing floating-point reduction order between dense and
gathered flash kernels; the strict elementwise failures remain recorded.

VM artifacts are under `/workspace/deepseek41-work`:

- `fixture-stable/validation-cpu-1.json` and `validation-cuda1-pedantic-noFA.json`.
- `fixture-cuda-index/validation-cuda1-pedantic-noFA-strict.json`.
- Each fixture's `validation-gather-cuda-8.json`.
- `fixture-stable/native-cuda-pedantic-trace/` and the corresponding trace log.
- `precision-wide-before.log` and `precision-wide-after.log`.

## Engram page-read concurrency

Long prefill exposed serialized page faults while gathering sparse Engram rows
from the network filesystem. The initial change introduced a bounded pool of
I/O workers per model: `TS_DSV41_ENGRAM_THREADS=1..32`, default 16 or the hardware
thread count if smaller. Prefill chunks of at least four tokens submit disjoint
row reads to that pool; that version retained serial lookup for smaller chunks.
The subsequent [CLI execution correction](validation/deepseek41/cli-gpu-execution/README.md)
also uses the pool for decode and other small batches. No worker changes
hash order, dequantization arithmetic, or per-sequence history.

`TS_DSV41_ENGRAM_WARM=1` optionally touches all Engram table pages during model
load in contiguous 8 MiB tasks. It neither copies nor pins the tables. Warming
is skipped if mapped host weights would leave less than 8 GiB of headroom in
the detected host/cgroup allowance. The default leaves pages demand-loaded.

`deepseek41-engram-parallel-io` verifies exact serial/parallel dequantized rows,
unchanged warming bytes/checksum, bounded concurrency, repeated jobs, concurrent
submitters, exception propagation and recovery. Local AddressSanitizer,
UndefinedBehaviorSanitizer, and ThreadSanitizer runs pass. The full CPU fixture
also passes all 41 strict checks with 16 workers and load-time warming enabled.
The requested VM passes that CTest and AddressSanitizer/UndefinedBehaviorSanitizer
run too. Its CUDA-index fixture passes all 41 strict checks with 16 workers and
warming enabled (maximum absolute error 7.40e-6, relative L2 2.40e-6), matching
the prior serial arithmetic. Tested library SHA256:
`585b3bd471e8c19c4b00650b42f0f7600e2972a354c323c2ad38678c6aa51232`.

The Linux `GgmlOpsDsv41EngramIoBench SCRATCH_DIRECTORY` target measures a private
64 MiB file on the chosen filesystem. It checks initial page residency, reads
1,024 selected rows with one or 16 workers, and compares cold/warmed outputs
exactly. It evicts only its own scratch-file pages and removes that file after
the run. These filesystem measurements are separate from end-to-end timings.

On the requested VM, every mode began with zero resident pages. Cold lookup
fell from 0.8264 s with one worker to 0.08282 s with 16 (9.98×); all output rows
agreed exactly. Whole-file warming took 0.2246 s serially or 0.1366 s with 16
workers, with identical checksums. Warm lookup itself was 0.265 ms serially and
0.533 ms in parallel, showing dispatch overhead for this small resident-table
case. These are filesystem microbenchmarks, not model token-throughput claims.
Raw results are in [`engram-io-filesystem.jsonl`](validation/deepseek41/engram-io-filesystem.jsonl).

## Independent vision and mixed-modality reference

The lossless companion preparer downloads only the pinned official vision
shard (970,533,624 bytes) plus HTTP ranges for the three learned image
delimiters and forty visual router biases. It verifies the complete isolated
shard against its published LFS SHA256 and records each selected tensor's
source range and SHA256. Partial text-shard ranges are explicitly recorded as
partial verification. The official tokenizer fingerprint must match the
parent Engram sidecar before the companion is written.

The synthetic oracle imports the pinned official `vision.py` and
`image_processor.py`. F32 complete spans passed all four grids (3×3, 4×5,
1×7, 6×4), including padded channel-major 3×3 patch merging, two-dimensional
rotary positions, and learned START/NEWLINE/END rows. The mixed text/image
suite passed 158 checks, including 43 numerical comparisons with maximum
absolute error 5.49e-6 and relative L2 error 1.54e-6, under `atol=rtol=2e-5`.
It covers image spans split across calls, 1/3/5-token chunks, independent
interleaved slots, reset, encoder handle release after attachment, and invalid
mask/count/ID/NaN rejection without changing subsequent inference. Negative
controls demonstrate that omitting visual router bias, Engram suppression, or
image history barriers changes the fixture's logits. The same numerical
checks also passed with all five synthetic routed-MoE layers on CPU and an
internal microbatch of three tokens. Text-only checks remained 41/41.

BF16 complete-span checks use separate bounds: observed maximum absolute
error was 0.00390625 and maximum relative L2 error 0.003116 on the tiny CPU
fixture. These reflect discrete BF16 rounding differences and do not meet the
strict F32 tolerance. The bounded BF16 fixture passed `atol=0.004, rtol=0.005`;
this is not a full-checkpoint vision quality result. Reproducible numerical
results and native/source hashes are in
[vision-fixtures.json](validation/deepseek41/vision-fixtures.json).

The real companion was prepared and its SHA256 is
`e7b0debed15706dd2f065879fa62a54f49e5c0472a54fa33d5ff40957f167e0c`;
[vision-companion.json](validation/deepseek41/vision-companion.json) preserves
all 306 tensors' provenance. The official real-weight tower was also run on
the 640×480 test card: preprocessing produced a 35×46 patch grid and 206 image
span rows. The native learned delimiters matched exactly. Projected features
differed from the official CUDA BF16 oracle by 4.02% relative L2 with native
flash attention and 1.89% with `TS_DSV41_VISION_FA=0`. Both failed the tiny
fixture's elementwise BF16 tolerance; these are unresolved numerical
differences, not a strict full-image oracle pass. The official source supplies
three-dimensional Q/K/V, so PyTorch's fused attention is unavailable and its
AUTO and MATH results were bitwise identical. A diagnostic that changed only
official linear accumulation to F32 with BF16 output rounding changed features
by a comparable 1.81% relative L2, demonstrating sensitivity to BF16 arithmetic
choices. This control does not establish which output is more accurate.
[vision-real-numerical.json](validation/deepseek41/vision-real-numerical.json)
records the separate comparisons and exact binary hash; end-to-end media
results are evaluated separately.

Matching the official rotary arithmetic order (`position * (1 / frequency)`)
reduced the dense projected-feature difference to 1.797%. An optional native
BF16 GEMM path, retaining F32 accumulation/output before adding bias and
rounding, further reduced it to 1.658%. These diagnostics also remain outside
the strict elementwise bound. Stage tracing found progressive differences:
the initial patch projection matched 99.9914% of elements exactly with relative
L2 error 1.93e-6, then small BF16 differences accumulated across the 32 blocks.
This supports arithmetic sensitivity rather than an initial patch-layout
failure; it does not establish numerical parity.

The final policy selects dense vision attention and native BF16 GEMM on
NVIDIA CUDA, retaining F32 accumulation and bias-before-output rounding.
`TS_DSV41_VISION_FA=1` explicitly opts into the flash approximation. The
vision-validation build (`37c10366…`) passed 158/158 mixed F32 checks on both CPU and
CUDA, plus 18/18 bounded BF16 vision checks on CUDA. The real-image diagnostics
above retain their original binary hashes rather than being relabeled as
measurements of a later build.

The shared text/image executor also has explicit recovery tests for backend
failure after state mutation. A test-build-only hook interrupts a real forward
after Engram updates or after graph execution, at the first or second internal
microbatch. All 134 CPU checks passed: both forward APIs reject a failed slot
until a full reset, the other slot remains usable, rewind/speculative calls
cannot bypass the failure, and reset restores exact fresh-oracle output.
A failed initial graph construction is discarded, and reset followed by an identical-shape retry matches the oracle. Invalid input is rejected before poisoning healthy state. Results and the
test binary hash are in [failure-state.json](validation/deepseek41/failure-state.json).
Run `eng/tests/dsv41-failure-state.py` with the text/vision fixtures and a native
test build to reproduce these checks; production builds exclude fault injection.

```bash
python eng/dsv41-vision-fixture.py VISION_FIXTURE_DIRECTORY \
  --reference-source-dir PINNED_OFFICIAL_SOURCE_DIRECTORY \
  --parent-engram TEXT_FIXTURE_DIRECTORY/deepseek41.engram.bin --preprocess
TS_DSV4_FA=0 TS_DSV4_GATHER=0 python eng/tests/dsv41-vision.py \
  VISION_FIXTURE_DIRECTORY --text-fixture TEXT_FIXTURE_DIRECTORY \
  --library /absolute/path/to/libGgmlOps.so --backend CPU

# Run the official tower on the prepared real companion, without text weights.
python eng/dsv41-vision-reference.py CHECKPOINT_DIRECTORY/deepseek41.vision.gguf \
  VISION_ORACLE_DIRECTORY --reference-source-dir PINNED_OFFICIAL_SOURCE_DIRECTORY \
  --image IMAGE_PATH --device cuda
```

The preprocessing fixtures include wide/tall padding, downsampling, ignored
alpha, non-square images, and a one-pixel image, with the exact Pillow version,
grid plan, token types, and complete normalized BF16 patch arrays recorded.
Full-checkpoint image interpretation and ordered video-frame quality require
separate end-to-end checks.

## Published Q2_K arithmetic smoke comparison

The final native `6b3b5ab3…` evaluated the same 18 input IDs as the retained
independent CPU reference. Its untraced primary run generated token 22 (`4`)
and then token 1 (EOS). A second replay in a fresh cache slot retained layer
traces and produced bitwise identical logits. All 81 required stage/terminal
trace files have exact sizes, and cleanup left no GPU compute process.

The strict `atol=rtol=2e-5` comparison **fails**: relative L2 0.1462157145,
maximum absolute difference 2.7089203596, cosine 0.9898177485. Top five IDs
`[22,20,21,24,19]` match. The final logits are also bitwise equal to the earlier
raw smoke artifact, whose producing binary was not bound in that artifact;
this does not demonstrate a before/after precision-fix improvement.

Of 247 available stage pairs, the embedding and first attention input meet
the strict tolerance. Query output after the first layer's Q2_K projections
has relative L2 0.010063. The reference uses dequantized matrices and F32
inputs; native quantized CUDA paths construct Q8_1 activations. This is a
concrete arithmetic difference consistent with early divergence, without
fully attributing the final error. It is not an exact quantized-kernel oracle,
an original-FP8 comparison, or a model-quality parity result.

The [complete evidence](validation/deepseek41/smoke18-reference/README.md)
retains source hashes, reference provenance, unrelaxed metrics, per-stage
analysis, paired logits and cleanup checks. Full traces remain under
`/workspace/deepseek41-work/final-smoke18-6b3/trace-after` and
`reference/q2-smoke18/`; the compact checked-in bundle records their hashes.


The V4.1 thinking budget now forces its trained `</think>` token through normal
sampling and forward execution, then continues within the original output
limit. Natural reasoning completion remains untouched; delayed JSON grammar
activates on the real closing token. Other architectures retain the existing
hard stop. The 1,536-token reasoning-only weather failure remains baseline
evidence. A subsequent full-checkpoint run with the 3,433-test budget host
completed that same weather answer in 1,550 generated tokens: 1,536 reasoning
tokens, the trained closing token, and 13 final-answer tokens. It returned
`{"city":"Paris","temperature_c":19}` with `finish_reason=stop`. The separate
362-token reasoning-loop case still failed on that host. The later loop-closure
policy was exercised by the first full-checkpoint TP run: short, schema and
weather reasoning passed, while the agent workflow exposed the separate malformed
tool-tag issue described above. The original layer-placement VM report
is `layer8-context65536-ubatch1024-cpumoe0-sparse1-compact1-slots4-b26c-thinking.json`.

The initial sixteen dedicated checks covered a real scheduler/executor with a deterministic
model, final JSON after the forwarded close, exact output limits, cancellation
during the closing-token forward, independent concurrent requests and grammar
states, natural closure/rollback, pending GPU argmax override, invalid-token
fallback, and the previous repetition stopping policy. That stage’s complete local managed lane
passed **3,416/3,416** in both Debug and Release, with zero skips or failures.
Release commands, counters, assembly hashes and TRX provenance are in
[thinking-budget-managed.json](validation/deepseek41/thinking-budget-managed.json).

The subsequent local grammar and reasoning-closure stage passed **124/124**
focused tests, including 32 budget/repetition cases and 28 exact-byte trigger
cases. V4.1 can now close a looping reasoning channel through the same normal
sampling path, while loops after `</think>` still stop. Ordered UTF-8 triggers
ignore tool markup quoted in reasoning, consume a trigger token’s answer suffix,
and mask an invalid suffix before sampling. Byte fallback, duplicate token
spellings, and independent request forks are covered. Subsequent full-checkpoint
TP evidence is recorded above; these focused checks are preserved in
[grammar-budget-focused.json](validation/deepseek41/grammar-budget-focused.json).

An independent tokenizer integration probe also passed **9/9** cases using the
Q2 checkpoint’s 129,280-token vocabulary, including DSML control token 128825
and reasoning-end token 128822. Every encoded token was checked against its
pre-sampling mask, then decoded incrementally through the strict output parser.
Unicode/raw XML arguments, nested JSON delimiter escaping, ordered reasoning
gates, and malformed-call rejection all passed. The reusable probe accepts the
first GGUF shard and reads only metadata; command and source hashes are recorded
in [actual-tokenizer-grammar.json](validation/deepseek41/actual-tokenizer-grammar.json).
Its local mask timings describe grammar overhead only, not model throughput.
