# Building against unchanged ggml

TensorSharp no longer applies patches to `ExternalProjects/ggml`. Both fetch
scripts consume the selected upstream revision unchanged. The changes below use
ggml's tensor, graph, allocation, and compute APIs in TensorSharp-owned code.

## What the patches did

1. **Metal flash-attention scratch allocation.** Upstream reserves space for an
   F16 copy of K/V even when a particular attention dispatch will not use it.
   The old patch omitted that reservation for those dispatches. TensorSharp's
   persistent graphs gave every layer a separate allocation, multiplying the
   reservation by the number of attention layers; at long contexts this could
   consume gigabytes beyond the quantized cache itself.
2. **Metal IQ4_XS small-batch matrix multiplication.** The old patch added
   IQ4_XS to upstream's optimized dispatch for batches of 4–8 rows. This reduced
   the cost of speculative verification, particularly on Gemma 4 E4B IQ4_XS.
   It did not accelerate single-token decode or expert `MUL_MAT_ID` operations.

## Replacements

`ggml_ops_attention_alloc.cpp` assigns attention workspaces according to their
graph lifetimes. It always requests the complete allocation size reported by the
backend, including every temporary buffer. A workspace can be shared only after
the preceding attention result's final consumer, and only when the next
attention already depends on that consumer. Independent branches, retained
outputs, input aliases, and other tensors keep separate storage. The same
planner works with persistent model buffers and the existing reusable compute
buffer. There are no assumptions about Metal's temporary-buffer layout,
dequantization thresholds, or private backend symbols.

Gemma's dense Metal models now prefer two speculative drafts when IQ4_XS
accounts for at least half of their matrix weight bytes. Shorter
windows avoid verifying long rejected suffixes on upstream's existing kernels.
Other quantization formats and MoE models retain their defaults; explicit
draft-window settings still take precedence. Verification also gathers and
projects per-layer embeddings (PLE) inside its existing fused graph, removing
separate dispatches and transfers. Unsupported PLE representations, including
projection weights with a separate scale multiplier, use the existing fallback.
Shared-KV layers also reuse their donor's prepared attention window instead of
repeating its concatenation and cache-dtype conversion. Reuse requires identical
bound K/V tensors and matching head dimensions, head counts, cache sizes, and
locality; other representations keep their existing construction.
Weights are neither requantized nor converted to a different storage format.

Qwen 3.5 keeps authoritative recurrent state on Metal across prefill,
verification, and accepted-prefix commits. The managed transition now follows
the native buffer-half tracking instead of downloading and uploading each
recurrent layer before every speculative step. Deferred native verification
retains its existing convolution snapshot and final GatedDeltaNet state view
until commit, which copies the selected state into durable storage in one GPU
copy graph. This avoids an additional copy into an unused output half. Metal
persistent verifiers use a private ggml lifetime allocator with stable addresses;
immutable weights, retained snapshots, and output aliases stay pinned across
replay. Other backends retain their existing allocation and state transitions.

Dense recurrent Qwen IQ4_XS models on Metal prefer twelve n-gram drafts, crossing
upstream's small-matvec boundary into a thirteen-row GEMM. Learned drafters keep
the existing three-draft preference: their per-token proposal cost differs from
n-gram matching. Explicit caps remain authoritative, as do non-default caps from
older programmatic callers. Wide Metal verify graphs retain the last three
recurrent states instead of a full state matrix for every prefix. Earlier
rejections use the existing snapshot restore and accepted-prefix re-forward.
This bounds snapshot memory while preserving exact continuation.

## Validation

The comparison uses Apple M5 Pro hardware with 48 GiB of memory, Release builds,
and upstream ggml commit `7840aaba1989c6deeefede1d77d5aaf8f52b947e`.
The reference library was built from TensorSharp `a776e7a` with both original
patches. The reference benchmark differs only by exporting its generated token
IDs and per-token delivery timestamps. Before/after runs use the same GGUF files
and prompts.

Confirmed checks:

- The managed CPU suite passed all **3,217 tests**, with zero failures or skips.
- The native allocation suite passed **46 cases across CPU and Metal**. A
  separate build with Metal disabled passed its **23 CPU cases**.
- Both fetch helpers passed **11 local repository checks each** (Bash and
  PowerShell 7), without network access. The comparator passed **19 tests**.
- The final Gemma suite passed **16 tests**: 14 draft-window policy cases and
  two real-model tests. Uploaded versus resident PLE had zero relative L2 and
  maximum absolute error across all 12 verification/context combinations.
  A separate comparison with PLE held constant required every hidden and logit
  bit to match with donor-window sharing disabled versus enabled; all 12 passed.
  Projection scales `0.5` and `-1` selected the uploaded fallback and also
  produced exactly matching hidden values and logits.
- Gemma 4 E2B Q8_0 sliding-window rollback passed **3 real-model tests**:
  context 300 with eight drafts, and context 900 with two or eight drafts.
  Subsequent greedy token streams matched the plain-decode reference after
  deliberately rejected drafts, including after the 512-position window wrapped.
- The initial real-model Qwen run passed **9 tests**. The final device-state-chain
  test compares the unpacked allocator with CPU copies against packed allocation
  with required GPU copies. Every hidden/logit bit matches across nine steps,
  including thirteen-row replay, full/partial acceptance, missing-snapshot
  restore/re-forward, single-row commits, and the next step after rollback.
  A separate checkpoint-free two-layer all-recurrent fixture passes on CPU and
  Metal, including initial allocation, two replays, three selected snapshots,
  state drains, and cleanup. It catches uploads to context-only inputs that
  ggml's graph allocator correctly leaves unallocated.

The native allocation suite checks numerical replay, strided externally bound
KV buffers, F32/F16/Q8_0 caches, retained views, independent and forked branches,
more than 64 attention nodes, reusable-buffer growth/shrink, and allocation
failure. In its 16-layer Metal Q8_0 case with 4,096 cache positions, allocation
falls from 34,419,712 bytes with unchanged upstream's separate allocations to
2,175,264 bytes with sharing. This comparison isolates the allocation strategy;
the end-to-end reference below includes the old patches. Sharing still reserves
one complete upstream workspace, so isolated decode allocations need not be
smaller than they were with the removed allocation patch.

Real Gemma tests compare every hidden value and logit with resident versus
uploaded PLE for verification sizes 1/2/3/8 at context lengths 509/520/1024.
The tests also assert which PLE path actually executed. Native A/B switches
update both the managed and libc environment tables so the comparisons exercise
different native graphs within the same loaded model. A proposed replacement of
separate PLE GELU/multiply operations with GEGLU was rejected: it produced
`3.419e-4` relative L2 error in hidden values at context 509 with two verify rows,
exceeding the existing `1e-4` bound. That optimization was removed; the bound was
not loosened. Qwen tests compare
F16/Q8_0/Q4_0 cache continuations and exact plain/speculative streams on retained
per-request caches.

`AgentTurnBench` exercises the inference engine, scheduler, prefill, decode,
speculation, and subsequent tool turns. Its JSON includes token IDs,
concurrent-request boundaries, and individual requests' token delivery times
from submission. `compare.py` fails on changed tokens, workload
shapes, benchmark errors, or performance regressions exceeding the configured
limit. Repeated runs can be compared using median timings; the default limit is
5%, rather than treating ordinary timing noise as a correctness failure.

Decode throughput divides tokens after the first by the interval from first
delivery to request completion. Moving first delivery earlier can lengthen this
interval even when every token and completion arrive earlier. The comparator
retains every raw metric delta and prints an explicit decode-rate exception only
when complete timelines in every repeat prove every corresponding median token
arrival and median completion is no later. Any later token, slower completion,
missing timeline, or concurrent aggregate retains the ordinary throughput gate;
prefill and TTFT checks still apply. Its tests cover these acceptance and failure
cases, malformed timelines, and unchanged strict behavior for older JSON.

### Gemma 4 E4B IQ4_XS: repeated end-to-end comparison

Three interleaved reference/candidate processes each ran one complete checked
warm-up pass, then the measured workload with normal .NET runtime settings.
Every run and warm-up produced the same **422 token IDs across eight rows**.
The `gemma-warm1/2/3.json` medians are:

| Workload | Decode tok/s, before → after | Change |
| --- | ---: | ---: |
| Short prompt | 61.09 → 60.05 | -1.70% |
| Long prompt | 61.92 → 61.12 | -1.30% |
| Plain greedy | 63.38 → 61.77 | -2.54% |
| N-gram speculation | 57.34 → 60.65 | +5.76% |
| Learned draft head | 73.92 → 73.45 | -0.64% |
| Tool turn 1 | 65.32 → 62.94 | -3.65% |
| Tool turn 2 | 60.68 → 65.19 | +7.43% |
| 16-token follow-up | 67.89 → 63.69 | -6.19% derived rate; see below |

Prefill and TTFT changes stay within the 5% gate. For the follow-up, every
corresponding median token arrives earlier, TTFT improves **130.85 → 112.55 ms**,
and completion improves **351.78 → 348.20 ms**. Its lower derived decode rate is
reported explicitly; complete token timelines establish improved delivery and
completion for that row.

Short first-process runs were variable: the three `gemma-timed` candidates
included a **15.71% median plain-decode rate regression**, so those results are
retained as failed comparisons. Disabling tiered compilation equally for both
binaries (`gemma-steady1`) passed all raw 5% checks; the full-workload warm-up
repeats above also passed, using the narrowly defined follow-up delivery
exception. Reference performance itself shifted from about 68 to 63 tok/s
during this session. The warmed measurements establish sustained performance
for these workloads; they do not establish identical startup latency under
all runtime and host-load conditions.

Gemma's median process peak footprint in the warmed comparison is
2,490,782,784 → 2,500,449,392 bytes (**+0.39%**).

### Qwen 3.5-9B IQ4_XS with Q8_0 KV cache

Two final interleaved reference/candidate pairs produced identical **463 token
IDs across seven rows**. `baseline/qwen-packed-final1/2.json` versus
`after/qwen-packed-final2/3.json` use the same short/long/speculative/tool
workloads and normal runtime settings:

| Workload | Decode tok/s, before → after | Change |
| --- | ---: | ---: |
| Long prompt | 47.27 → 46.81 | -0.98% |
| Plain greedy | 47.97 → 48.31 | +0.70% |
| N-gram speculation | 87.88 → 101.56 | +15.58% |
| Tool turn 1 | 43.03 → 43.64 | +1.42% |
| Tool turn 2 | 44.97 → 45.29 | +0.71% |

Every metric passed the raw 5% gate. Median process peak footprint falls from
**1,029,769,868 to 813,172,980 bytes (-21.03%)**. This is process footprint,
not total GPU allocation. Native allocation logs show a single-row persistent
verifier shrinking **73.2 → 7.7 MiB**, and the large verification entry changing
from **291.7 MiB for four rows to 182.5 MiB for thirteen rows**. The wider policy
uses more cached shapes; the minimum reported free device memory at persistent
allocations is **32,327.1 → 32,320.2 MiB**, about **6.9 MiB more retained GPU
memory**, rather than a reduction in the total. This distinction matters on
Apple's unified memory architecture.

The separate 32,768-token plain-decode run (`qwen-32k.json`) produced the same
32 output tokens. Prefill improved **733.39 → 811.05 tok/s**, decode improved
**41.03 → 43.28 tok/s**, and process peak changed **2.873 → 2.877 GB (+0.15%)**.
That run preceded the final persistent-verifier packing and does not exercise
speculative snapshot allocation.

### Gemma 4 E4B Q8_0: prefill, decode, and concurrency

The paired patched-reference and unchanged-ggml runs used 1,024-token prefill
chunks, a 4,096-token long-input target, and two/four concurrent requests. All
**280 output token IDs across five rows**, request boundaries, and workload
shapes matched exactly. The paired `baseline/gemma-q8.json` and
`after/gemma-q8.json` results are:

| Workload | Prefill tok/s, before → after | Decode tok/s, before → after (change) | TTFT ms, before → after |
| --- | ---: | ---: | ---: |
| Short prompt | 319.39 → 319.38 | 44.18 → 43.37 (-1.84%) | 97.06 → 97.06 |
| Long prompt | 1927.74 → 1947.93 | 45.26 → 44.85 (-0.90%) | 2182.35 → 2159.73 |
| Two concurrent requests | — | 44.62 → 42.84 (-3.99%) | 1192.03 → 1185.20 |
| Four concurrent requests | — | 49.00 → 48.13 (-1.78%) | 2727.74 → 2715.19 |
| Solo request after concurrency | 1927.40 → 1948.07 | 44.80 → 46.05 (+2.79%) | 415.07 → 410.66 |

Concurrent decode rates are aggregate throughput after the last request's first
token; TTFT is the maximum across requests. These aggregate rows do not report
prefill throughput. The comparator passed its 5% threshold for all five rows in
this pair. Process peak memory footprint measured by `/usr/bin/time -l` changed
from **4,094,385,400 to 4,078,132,496 bytes** (4.094 → 4.078 GB, -0.40%).

### Saved evidence

The local validation artifacts are in
`TestResults/ggml-without-patches-2026-09-10/` (local validation evidence, not committed).
They include successful final comparisons, every preceding benchmark probe
(including failed comparisons), build/test logs, and token/timing JSON. The
artifact README identifies final result files and the native binary hashes.
`TestResults` is intentionally ignored by Git; this report and the regression
harnesses are tracked source changes.

### Reproduce

```sh
TENSORSHARP_GGML_NO_UPDATE=1 dotnet build InferenceWeb.Tests -c Release
dotnet test InferenceWeb.Tests -c Release --no-build \
  --filter 'Category!=Bench&Requires!=Models&Requires!=Cuda&Requires!=Mlx'
cmake -S TensorSharp.GGML.Native -B TensorSharp.GGML.Native/build \
  -DCMAKE_BUILD_TYPE=Release -DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON
cmake --build TensorSharp.GGML.Native/build --parallel 8
ctest --test-dir TensorSharp.GGML.Native/build --output-on-failure
bash eng/tests/fetch-ggml.sh
bash eng/tests/fetch-ggml.sh --powershell # requires pwsh on PATH
python3 benchmarks/AgentTurnBench/test_compare.py

TS_TEST_GGML_BACKEND=metal TS_GMTP_TARGET=/models/gemma-4-E4B-it-IQ4_XS.gguf \
  dotnet test InferenceWeb.Tests -c Release --no-build \
  --filter FullyQualifiedName~Gemma4VerifyPleTests

TS_TEST_GGML_BACKEND=metal TS_TEST_MODEL_DIR=/models/gemma-4-E2B \
  dotnet test InferenceWeb.Tests -c Release --no-build \
  --filter FullyQualifiedName~Gemma4SwaRollbackExactnessTests

TS_TEST_GGML_BACKEND=metal TS_KVQ_BACKEND=ggml_metal \
TS_KVQ_MODEL=/models/Qwen3.5-9B-IQ4_XS.gguf \
TS_Q35_STATE_MODEL=/models/Qwen3.5-9B-IQ4_XS.gguf TS_TEST_MODEL_DIR=/models \
  dotnet test InferenceWeb.Tests -c Release --no-build \
  --filter 'FullyQualifiedName~QuantizedKvFusedGraphTests|FullyQualifiedName~Qwen35HolderSpeculationTests|FullyQualifiedName~Qwen35VerifyDeviceChainTests'

dotnet build benchmarks/AgentTurnBench -c Release
dotnet benchmarks/AgentTurnBench/bin/Release/net10.0/AgentTurnBench.dll \
  --model /models/gemma-4-E4B-it-IQ4_XS.gguf \
  --draft-model /models/gemma-4-E4B-it-assistant.Q8_0.gguf \
  --backend ggml_metal --chunk 1024 --scenarios short,long,spec \
  --long 4096 --tool 1024 --new 32 --spec-new 192 --warmup 1 --out candidate.json
python3 benchmarks/AgentTurnBench/compare.py baseline.json candidate.json
```

For Qwen, omit `--draft-model`, select its GGUF, and add `--kv q8_0`. The long
context memory run uses `--scenarios long --long 32768 --new 32` under
`/usr/bin/time -l`.

An older checkout may still contain patches applied by a previous build.
A successful normal fetch restores the selected upstream revision;
`TENSORSHARP_GGML_NO_UPDATE=1` deliberately preserves whatever is already on
disk. The validation checkout was cleaned of the two known patches before
building, and `git -C ExternalProjects/ggml diff --exit-code` verifies it remains
unchanged.

Explicit draft-window overrides can still expose the cost of upstream's missing
IQ4_XS small-batch optimization. Earlier forced-window probes measured Gemma
with seven drafts about 13–15% slower and Qwen with three drafts about 11–12%
slower than the patched reference. The replacement tunes normal defaults and
removes redundant graph/state work; it does not recreate that Metal kernel.
These results therefore do not establish universal performance parity for
explicitly forced speculative windows.

These measurements cover CPU and macOS Metal. They do not establish performance
on CUDA, Vulkan, or physical iOS devices, or promise identical timing for every
model and explicitly overridden speculative window.
