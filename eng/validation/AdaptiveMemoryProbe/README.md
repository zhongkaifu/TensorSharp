# Adaptive memory execution

The adaptive entry automatically reuses idle row-tiled CUDA sessions. Omit
`--workspace-cache-bytes` for automatic sizing, set zero to disable, or supply
a byte ceiling. Both weights and activations are replaced before each use;
this retains allocation capacity, not weight identity. Reuse requires the same
rank, weight format and input width, with sufficient row/token capacity.
ResidentCuda additionally requires the original logical matrix/token shape.
Complete-matrix multi-token Gemma sessions are excluded. Up to 16 completed
healthy sessions share the device budget. Admission leaves the execution peak
available and uses at most one quarter of discretionary device capacity after
other owners and headroom. Weight-cache admission holds that share available
without charging existing idle workspaces twice. Pressure releases idle
workspaces before retained weights. At request reset, sessions unused by the
last request are physically retired and the retained pool shrinks to current
slack. Healthy sessions used by the last request remain reusable. Failed cleanup preserves ownership
and blocks reuse until explicit reset physically retires the failed owner.

For an isolated allocation comparison, keep RAM/device weight-cache ceilings
fixed, and run four fresh processes with workspace caching off/on/on/off using
the timing/identity requirements below. Then run `compare-workspace-cache.py
--executions <four execution.json paths> --output <json>`. It requires complete
logit/history equality, actual reuse in every measured request, fewer session
creations, identical weight reads/uploads and operation counts, budget bounds,
successful shutdown and zero final owners. A cache that displaces weights fails
this isolated comparison; any such tradeoff needs a separate reported workload.
These checks do not themselves guarantee a speedup. The explicit low-level
`WeightStreamingOptions` constructor keeps its zero-cache defaults; automatic
selection belongs to `AdaptiveModelSession`, which supplies the hardware/request
forecast. The quarter-share rule is a bounded heuristic, not a global optimum.

For default policy validation, omit the workspace option in the two enabled
processes and pass `--workspace-cache-bytes 0` in the controls. Keep all other
settings identical, including automatic device/host weight-cache settings. Add
`--competition` to `compare-workspace-cache.py`; it requires actual workspace
reuse and weight-cache use, and checks conservation of consumed/projected weight
bytes while reporting changed uploads/reads. It preserves full-logit equality,
identical binaries, successful exits and zero-owner requirements. This is a
separate policy comparison; it does not relax the isolated allocation comparison.

`--alternate-prompt-tokens N` renders a second prompt. The excluded warmup uses
the primary prompt, then measured requests alternate second/primary/second. Both
prompt token arrays and each row's index are recorded. Competition comparisons
validate that schedule and report each prompt length separately. The first
measured second-shape request includes any shape-specific setup; it is not a
fully warmed constant-shape microbenchmark. Other comparison modes reject mixed
prompts. Explicit full-logit capture remains correctness-only.

The earlier opt-in `workspace-reuse-v1` comparison used Qwen0.8B, context512, 67 prompt
tokens, 16 predictions, 512 MiB host/device, 256 MiB host-cache and fixed 128 MiB
device-weight-cache ceilings. A 64 MiB workspace ceiling retained 31,555,584 B.
Six measured requests per arm gave prefill 82.297→89.414 tokens/s (+8.65%) and
decode 3.574→4.577 (+28.08%); each measured request replaced 2,224 fresh sessions
with reuse. All complete logits/history, weight uploads and logical reads were
unchanged. Shared owners returned to zero. Native `74c1b4f8…` was unchanged;
Models `e1162225…`. Quiet RTX 3080 Laptop, warm file cache, unlocked clocks.
This does not establish full-resident parity or a global cache-allocation policy.

`--device-cache-bytes` bounds optional complete file-weight CUDA arenas (zero
disables retention; omitted means use available forecast slack). The shared
budget charges retained weights, input/output capacity and arithmetic scratch.
Admission preserves the request's execution peak. Temporary workspaces reclaim
LRU arenas before reducing tile sizes. Normal KV reset keeps valid weights;
idle trim/disposal physically release them before refunding credit. Qwen's
FullPrecision path supports prefill/decode reuse; Gemma ResidentCuda currently
retains only N=1 and keeps its original larger-N arithmetic. FullPrecision
promotion removes duplicate same-source RAM ranges. This is synchronous CUDA
execution with bounded host staging, not asynchronous DMA or a global cache policy.

To qualify retention, run four fresh processes off/on/on/off with identical
native/managed files, model, prompt, ceilings, host-cache option and environment.
Use `TS_GGML_Q8_PARALLEL_VECTOR=0`, `TS_GGML_Q8_PARALLEL_SMALL_BATCH=0`, capture off,
one warmup and three measured requests per process, then run:

```sh
python eng/validation/AdaptiveMemoryProbe/compare-device-cache.py \
  --executions off-1/execution.json on-1/execution.json on-2/execution.json off-2/execution.json \
  --output artifacts/device-cache-comparison.json
```

The comparator requires completed nonoverlapping processes, full logits/history
equality, streamed placement, real device hits, unchanged other options, owner
cleanup, shared-budget bounds and consistent file/RAM/device consumption. It also
requires actual weight-upload reduction, rather than accepting a configured cache
that was never used. Logical reads do not measure physical SSD traffic.

Final local RTX 3080 Laptop validation (`device-weight-cache-v2`, native `74c1b4f8…`,
Models `c2404c9e…`) used Qwen0.8B Q8, context512, 67 prompt tokens, 16 predictions,
host/device ceilings 512 MiB, host cache ceiling 256 MiB, tile32 and default ordered
decode. A 256 MiB device-cache ceiling retained 184,048,640 bytes after headroom.
Six measured requests per arm gave prefill 75.096→86.556 tokens/s (+15.26%),
decode 3.035→3.851 (+26.90%), weight H2D 13,839,638,528→10,997,921,792 bytes/request
(−20.53%) and logical source reads 9,356,312,576→6,518,669,312 (−30.33%).
Every complete logit hash/history matched; all processes shut down and charged
owners returned to zero. No concurrent inference/build/download, warm file cache,
unlocked clocks. This constrained file mode remains far below the full-resident
baseline; no independent language-quality, multi-GPU or asynchronous-transfer
performance claim follows.

The adaptive file-weight path now keeps immutable source ranges in pageable RAM
when hardware and request forecasts leave room. `--host-cache-bytes` is an optional
ceiling (zero disables reuse); actual retention is bounded by source size, the
shared host pool, and prefill/decode workspace headroom. Loading-only temporary
memory is not held out after loading. The required read tile is already charged;
optional read-ahead staging competes for remaining slack. `RefreshCapacity()`
trims idle cache before shrinking a pool enough to starve the next request's
workspace, even when current payload owners would still fit.

Cache ranges are copied into existing staging before device use. No cached host
pointer enters a graph, no weight bytes change, and disposal refunds only after
freeing allocations. Full caches retain existing ranges during sequential scans
instead of replacing every tile before reuse; pressure trim is LRU. This is not
a measured cost model or GPU weight retention. Exact-range keys may duplicate
overlapping layouts; at most 4096 entries bound index overhead, which remains
outside aligned-payload accounting. Explicit `WeightStreamingOptions` still
defaults to no cache; this automatic policy applies only to the adaptive loader.

To compare reuse, run four fresh bounded probe processes in `0/C/C/0` cache-ceiling
order, using the same model, prompt, context, host/device ceilings and binaries.
Set `TS_GGML_Q8_PARALLEL_VECTOR=0`, `TS_GGML_Q8_PARALLEL_SMALL_BATCH=0`, keep the
prefill tile setting fixed, and disable full-logit capture. Use `--repeats 3`;
the recorded warmup is excluded. Force a supported file-weight placement through
the device ceiling and confirm it in the report, rather than assuming streaming.
`compare-host-cache.py --executions FIRST SECOND THIRD FOURTH --output SUMMARY`
checks completed non-overlapping processes, binary identities, unchanged full
logit hashes/history, actual cache hits, equal consumed bytes, fewer source reads,
budget snapshots and final owner cleanup. Logical file reads do not measure SSD
traffic. Warm OS cache, unlocked clocks, raw-greedy prompts and limited hardware
remain qualification limits; a pass is not a semantic or independent-engine gate.

The owned CUDA Q8/F32 prefill now selects 32, 64 or 128 columns automatically
when `TS_GGML_Q8_PREFILL_TILE` is unset (or `auto`). It uses each device's SM
count and actual kernel occupancy, the row/column geometry and a 25% maximum
padding allowance. Small row strips retain more parallelism. No global device
workspace or widened weight copy is added, and every output keeps the original
K-increasing FMA order. Explicit `32`, `64` and `128` remain available for
controlled comparisons. The separate parallel-K decode/small-batch experiments
remain opt-in; automatic prefill does not change their default arithmetic.

`compare-prefill-tiles.py --automatic --vector-mode serial --executions ...`
requires four fresh processes in `32/unset/unset/32` order, three measured
requests each, native selection logs, identical complete logit hashes and
successful exit/shutdown. Use `--vector-mode parallel` when that experiment is
enabled in all arms; do not mix it into the prefill comparison.

On RTX 3080 Laptop 16 GiB, Qwen3.5 0.8B Q8, 643 prompt tokens, context 2048,
64 prediction rows, native `9be5eabf71cb2554a2bf4a71866c4b2d12ce8a898854ba79fdba91b94f82ca89`
and unchanged ggml `ffa4e8b80930029a35991f94e7c8a93cd67730ab`, six samples/arm gave:

| Fixed decode arithmetic | Prefill fixed 32 → automatic (tokens/s) | Decode fixed 32 → automatic (tokens/s) |
| --- | ---: | ---: |
| Default K-ordered | 2106.63 → 2791.37 (+32.50%) | 19.519 → 19.442 |
| Opt-in parallel-K | 1994.47 → 2747.54 (+37.76%) | 177.943 → 181.950 |

Complete logits are byte-identical **within each row's prefill comparison**,
not between the two decode arithmetic policies. Eight isolated processes
finished successfully; compilation/downloads/other inference did not overlap.
Clocks were not locked and decode ranges overlap. This is one checkpoint and
geometry, not independent-engine parity or a semantic-quality pass. The default
decode remains slow and prefill is still below the earlier independent baseline.
Reports and full binary identities are ignored under `q8-prefill-auto-v1/`.

Mixed Gate/Up matrices now keep their original formats on single-rank GGML
CPU/CUDA families with a split FFN. Matching formats still fuse. The single-CUDA
Gemma/Qwen budget forecast no longer reserves a requantized replacement or its
conversion scratch. Tensor-parallel and other backend policies are unchanged.
Regression fixtures assert original objects/types/bytes and matching-format
concatenation; the old loader fails the mixed-format fixture and both affected
architecture forecasts. Current CPU checks pass 38/38, plus the CUDA fixture.

On local `Qwen3.8-27B-UD-IQ4_XS.gguf` (SHA256
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`),
28 mixed layers remain split and 37 matching pairs still fuse. One fresh process
per version observed load 85.14 → 7.36 seconds and post-load working set
16,397,885,440 → 14,483,877,888 bytes. This is a warmed file-cache observation,
not a cold-load or repeated performance result. Timed forwards with full-logit
capture were about 397/393 prefill and 7.57/7.55 decode tokens/s; capture perturbs
execution, so these are not quiet throughput qualifications.

An unchanged llama.cpp `4ebdf2c74acce30883d8e34b7c70b3eb8146f2fe` ran the same
checkpoint, 258 prompt IDs, context 512, F16 KV and 16 fixed-history predictions,
all 66/66 layers offloaded. Complete-vocabulary comparisons remove only a common
token offset from HTTP log probabilities. Old/new median relative errors were
0.007010 / 0.004672, maxima 0.026925 / 0.017443; both matched all 16 argmax IDs,
but **both fail the 0.001 numerical gate**. Some individual rows became worse.
Preserving the checkpoint removes load-time weight corruption; this observation
does not establish full-model numerical or language-quality correctness.

The same repaired model with experimental parallel-K vector reduction also
**fails** its matched-history comparison against serial reduction (maximum
relative L2 0.011107). This wider-model failure is why parallel decode remains
opt-in despite its earlier narrow-model performance result. Evidence and exact
binary identities are under ignored `qwen27-q8-precision-v1/`.

A separate local Gemma E4B Q8 regression (checkpoint `96c45581…`, 262 prompt
tokens, context 1024, 16 predictions, current Models `e85e3b25…`) produced
byte-identical complete outputs with fixed-32 versus unset configuration.
Its resident path uses the existing quantized arithmetic and did **not** invoke
the Q8/F32 automatic tiler. Thus `gemma-q8-auto-v1/` is an unaffected-path
regression, not wider automatic-kernel coverage or a performance improvement.

`llama-teacher.py` accepts one or more `--report <report.json>` captures, a
`--server` URL, `--identity <identity.json>` and a fresh `--output` directory.
The identity records `model_sha256`, `binary_sha256`, `source_revision`, the
actual `command` and `startup_evidence`. The caller owns/stops the server and
records its real configuration and libraries; HTTP alone cannot authenticate
that manifest. Reports must contain complete captures with identical checkpoint,
geometry and histories; different managed binaries are allowed. The tool verifies
all capture bytes, saves requests and full responses (gzip), rejects clipped or
incomplete probabilities, and reports the gate per input. A failing older input
keeps the combined exit nonzero even if a newer input passes. Contract tests:
`python -m unittest discover -s eng/validation/tests -p test_adaptive_llama_teacher.py`.

This probe exercises `AdaptiveModelSession` with the same `ForwardRefill` and
`Forward` entries as a regular resident model. It records load time separately,
one warmup followed by measured requests, complete raw-logit hashes, token IDs,
hardware free VRAM, process working set and the shared-budget ledger. Disposal
must release every charged allocation. Raw greedy histories are used to compare
arithmetic, **not** as language-quality validation; model-defined generation
suppression is tested separately by `GemmaRepetitionProbe`.

Build with the normal managed skip-native flags, then copy the independently
built `GgmlOps.dll` beside the probe. Never rebuild/copy into a running process.
Use an unchanged upstream checkout, record its revision, model SHA, loaded native
SHA and actual GPU. No missing or skipped scenario counts as a pass.

For a numerical candidate that intentionally changes accumulation order, add
`--capture-logits true --teacher path/to/teacher.json`. The teacher file must be
a JSON integer array with exactly `--steps` valid vocabulary IDs. The probe emits
`steps` prediction rows: prefill predicts row zero, then the first `steps - 1`
teacher IDs are consumed. The final teacher ID is retained for provenance but is
not forwarded. Every row still records the actual argmax; a changed argmax does
not silently change subsequent teacher conditioning. Warmup uses the same history
but only measured runs are written to `logits.f32` and `logits.json`. Use a fresh
output directory. Capture I/O is outside forward timers; these runs are for
correctness and cannot be treated as quiet throughput. Omit both options to keep
the original raw-greedy performance workload with capture disabled.

```powershell
python eng/validation/AdaptiveMemoryProbe/compare-captures.py `
  --left artifacts/control/report.json --right artifacts/candidate/report.json `
  --output artifacts/capture-comparison.json
```

The fixed per-row gate is relative L2 ≤ 0.001, cosine ≥ 0.999999 and equal argmax.
Maximum absolute error and byte equality are reported separately. Complete
checkpoint/managed identities, model geometry, history, requested run counts,
payload hashes, finite values and successful shutdown must agree; truncated or
failed captures never pass. Native hashes may differ. This numerical agreement
does not establish semantic quality; retain actual process exit evidence too.

The output directory must be absent or empty, even with capture disabled. Reports
record the inherited `TS_GGML_Q8_PARALLEL_VECTOR` value, probe SHA and every loaded
TensorSharp assembly SHA. The native experimental route also prints its selection
once to stderr; an environment setting alone is not proof it was exercised.

To isolate this flag, first build the probe once and copy the complete output
directory into two new directories, `serial` and `parallel`. Put the **same**
independently built native library into both. Hash the deployments before use;
never overwrite previous ABBA directories. Derive fixed conditioning from a
successful serial report without reusing its logits as the candidate reference:

```powershell
$prior = Get-Content artifacts/prior-serial/report.json -Raw | ConvertFrom-Json
$teacher = @($prior.Records | Where-Object { -not $_.Warmup })[0].Generated
if (-not $prior.Executed -or $teacher.Count -ne 64) { throw 'Incomplete teacher source' }
$teacher | ConvertTo-Json | Set-Content artifacts/teacher64.json
$env:TS_GGML_Q8_PARALLEL_VECTOR = '0' # use '1' for the separate parallel process
python eng/validation/run-bounded-probe.py --timeout 240 --output artifacts/serial-capture-process -- `
  dotnet artifacts/serial/AdaptiveMemoryProbe.dll --model path/to/model.gguf `
  --mode resident --prompt-tokens 640 --context 2048 --steps 64 --repeats 1 `
  --capture-logits true --teacher artifacts/teacher64.json --output artifacts/serial-capture
```

After the numerical gate passes, use fresh processes in
serial/parallel/parallel/serial order with `--capture-logits false`, one warmup
and `--repeats 3`. Keep the model, managed binaries, native binary, prompt and
other runtime knobs identical; set only the inherited experiment flag to 0/1.
Use a distinct `run-bounded-probe.py` output for each process. The strict timing
comparator consumes those wrapper reports and their adjacent logs:

```powershell
python eng/validation/AdaptiveMemoryProbe/compare-parallel-runs.py `
  --serial artifacts/perf-0-process/execution.json artifacts/perf-3-process/execution.json `
  --parallel artifacts/perf-1-process/execution.json artifacts/perf-2-process/execution.json `
  --output artifacts/parallel-performance.json
```

It requires matching argmax/consumed histories, successful shutdown and exit,
separate binary directories, actual selection logs, nonoverlapping ABBA order
and six measured requests per arm. Raw-logit hashes may differ but are retained;
this timing comparison never substitutes for the separate numerical gate.

```powershell
dotnet eng/validation/AdaptiveMemoryProbe/bin/Release/net10.0/AdaptiveMemoryProbe.dll `
  --model C:/Works/models/gemma-4-E4B-it-uncensored-Q8_0.gguf `
  --mode adaptive --prompt-tokens 640 --context 2048 --steps 64 --repeats 3 `
  --output artifacts/adaptive/e4b-adaptive-0
```

Run fresh processes in resident/adaptive/adaptive/resident order, with exclusive
GPU use and no simultaneous CPU inference, compilation or model hashing. Pass
all four `report.json` paths to `compare-runs.py --output artifacts/.../summary.json`.
Inspect both timing distributions; one fast pilot is not a performance claim.
`--device-bytes` and `--host-bytes` are explicit operator ceilings for constrained
tests. An unsupported quantization must be refused rather than silently running
an unbounded alternate path.

Current adaptive loading supports dense Gemma4 and Qwen35, one GGML CUDA device
and a sequential text lane. Model context and prefill chunk are per-model values;
the production API does not mutate process environment. The probe pins KV dtype
and clears distributed/speculation environment only to make its two test arms
comparable. Requests with another geometry need a new plan at a quiescent boundary.

Planning uses physical available RAM/VRAM, explicit headroom, model layout,
context, KV/state and loading/prefill/decode workspace forecasts. Residency keeps
the existing fused graph; insufficient capacity can select a supported bounded
file adapter. Host page cache is demand-paged by the OS, not an owned full copy.
Covered native caches/graph buffers and file-staging allocations reserve actual
payload before allocation. These hooks do **not** constrain all process RSS,
driver/library pools or every native model executor. `RefreshCapacity()` preserves
live owners and refuses new admission on pressure; it never evicts active KV.

## Local baseline comparison (2026-10-08)

RTX 3080 Laptop 16 GiB, unchanged ggml
`ffa4e8b80930029a35991f94e7c8a93cd67730ab`, E4B Q8_0/F16 checkpoint
SHA256 `96c455818ff64884f0e2ae3bc5517675896c4eae60676cc9135b9bb865eaf15c`.
Resident/adaptive/adaptive/resident fresh processes, one excluded warmup and
three measured requests per process, context 2048, 640 minimum prompt tokens,
64 raw-logit rows/request. The latest `v2` run uses native SHA256
`cf1e969d8f5618734ea43484f96f5485b6fc005ea59a2d0572dc59af4d9078d0`
and Models assembly SHA256
`4f264ce32f3e68d9cc48357108ac1dbf86e002bb757b14c0e9a56123e9afc08c`.

| Path | Prefill median (range), tokens/s | Decode median (range), tokens/s |
| --- | ---: | ---: |
| E4B resident | 2158.09 (1944.21–2258.53) | 55.96 (55.33–56.29) |
| E4B adaptive resident | 2264.80 (2207.56–2285.54) | 56.24 (55.81–56.32) |
| Qwen3.5 0.8B Q8 resident | 2171.81 (2096.44–2195.73) | 18.38 (18.32–18.43) |
| Qwen3.5 0.8B Q8 adaptive resident | 2168.53 (2128.54–2191.54) | 18.36 (18.33–18.37) |

Six measured requests per arm and checkpoint; each checkpoint's complete raw-logit
histories have the same SHA, all processes exit zero and all charged owners are
released. E4B adaptive/resident median ratios are 1.0494 prefill and 1.0051 decode;
Qwen's are 0.9985 and 0.9989. Ranges overlap. This comparison measures the overhead
of adaptive placement against the same build's resident execution; it does not
establish that either implementation is optimal or matches an independent engine.
In particular, Qwen's absolute decode throughput remains under investigation.
Qwen checkpoint SHA256 is
`0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c`.
Hashing runs outside timing but warms file cache, so these are not cold-storage
measurements. The earlier forced 256 MiB streaming result has a different capacity
constraint and is not used as the resident baseline. Generated reports: ignored
`artifacts/unified-memory-adaptive/{e4b,qwen08}-abba-v2-*`; prior `v1` evidence is
retained separately and has an older assembly/native identity.

## Single-token Q8 projection comparison

`compare-native-runs.py` compares two resident binary directories with identical
managed assemblies and checkpoint, allowing only the native-library identity to
differ. For current probes use `--executions FIRST SECOND THIRD FOURTH --output
SUMMARY`, passing the bounded runner's four execution records in
control/candidate/candidate/control order. Each process excludes one warmup and
measures three requests; capture and parallel-K options must be disabled.
The comparator checks non-overlapping process lifetimes, exit status, native
shutdown, complete binary/geometry/environment identities, frozen binary
directories, timing denominators and full-logit/history equality, including
warmup. The legacy `--control`/`--candidate` report-only interface remains for
older evidence, but does not qualify process completion or isolation.

The current default N=1 kernel gives each lane one output row and uses one warp
per CTA. Two-byte packed Q8 loads respect the format's 34-byte block alignment;
the same increasing-K FMA sequence consumes the original F32 activations. It
removes shared-memory transposition and per-block barriers without allocating
global scratch or expanding weights. On sm86, the compiled kernel uses 40
registers, no local stack and no shared memory. Parallel-K remains opt-in.

RTX 3080 Laptop 16 GiB, the same 643-token Qwen0.8B fixture/context2048/64 prediction
rows, four fresh ABBA processes and six measured requests per arm:

Prefill uses the automatic policy; parallel-K vector/small-batch are both off.

| Native | Prefill median (range), tokens/s | Decode median (range), tokens/s |
| --- | ---: | ---: |
| Previous `9be5eabf…` | 2783.714 (2731.641–2856.318) | 19.513 (19.461–19.528) |
| Packed ordered rows `3c6f0fb7…` | 2804.476 (2684.324–2839.527) | 84.985 (83.741–85.759) |

Complete logits and all histories, including warmup, are byte-identical. Decode
improved 4.355x; prefill intervals overlap. No concurrent inference/build/download
ran; clocks were not locked, and hashing warms the filesystem cache. This still
falls short of the previously measured independent llama.cpp 183.98 decode /
8792.20 prefill reference. It is a regression/performance check, not a semantic
quality claim or proof that the previous arithmetic is an exact oracle.

The same managed files and old/new native libraries also capture all 16
teacher-forced vocabulary rows of local Qwen3.8-27B-UD-IQ4_XS, prompt258/context512,
with byte-identical results. This avoids the new parallel-K divergence previously
observed on 27B; it does not resolve the existing gap against independent llama.
All 27B comparisons fix prefill tile32 and disable both parallel-K options.
Separate capture-disabled 27B ABBA runs with three measured requests per process
give decode 7.503 (7.422–7.541) → 14.749 (14.411–14.879) tokens/s, a 1.966x gain.
Prefill is 397.602 (374.978–400.633) → 391.744 (383.002–400.776), with overlapping
ranges; the measured decrease is retained. All warmup/measured logits and histories
match bitwise. These fixed-history results are not open-ended language quality.
Native CPU/CUDA precision and streaming checks pass 12/12 with no skips, and the
production default precision test passes CUDA memcheck with zero errors.
The existing strict FP64 thresholds are unchanged.

The standalone `GgmlOpsQ8OrderedVectorBench --check` checks ten synthetic research
routes on 60 shapes/stride layouts. Each route must match every output byte and
canary; source buffers are read back to detect writes. Its independent FP64
check uses the ordinary sequential-FMA forward-error bound, not the stricter
production precision gate. `--benchmark K M` is bounded to a 2 GiB weight fixture,
samples at most 62 FP64 rows, and balances forward/reverse route timing. It is
explicit research, not CTest or model throughput. Its control is the production
kernel linked at build time. The original-control executable was frozen before
this production change; rebuilding now uses the new control.

Evidence: ignored `artifacts/unified-memory-adaptive/q8-ordered-vector-v1/`.
Final native SHA `3c6f0fb7204b678a607fe38f74ba39fdcf436dd8503246adab16634d8beda7bc`,
Models `eb3781a969a37595de32db0e301420fb9dc3f461148526f5f7896f71af606397`,
AdaptiveMemoryProbe `402871974977554eaffffc303147d38e4820ad7dc446035cb3a21ad133ff9386`;
unchanged upstream ggml `ffa4e8b80930029a35991f94e7c8a93cd67730ab`.

The earlier N=1 specialization below is retained as historical evidence with
its separate native/managed identities.

On the same hardware and Qwen checkpoint above, context 2048, 643 prompt tokens
and 63 decode calls per request, the N=1 specialization keeps the previous
K-ordered FMA but skips seven unused activation columns. Native control SHA is
`cf1e969d8f5618734ea43484f96f5485b6fc005ea59a2d0572dc59af4d9078d0`;
candidate SHA is
`aad101dc524c20388a351e6b0b7bd05e4ee4901d41ab00b2655d55e4e6a8981d`.
Managed Models remains `4f264ce3...` as recorded above. Six measured requests per
arm, with no concurrent inference, build or download, gave:

| Native | Prefill median (range), tokens/s | Decode median (range), tokens/s |
| --- | ---: | ---: |
| Previous | 2174.71 (2130.12–2194.91) | 18.374 (18.354–18.390) |
| N=1 specialization | 2203.59 (2180.33–2234.07) | 20.122 (20.088–20.191) |

All complete raw-logit histories match bitwise and all four processes exit zero.
Decode improves 9.51% in this fixture; prefill ranges overlap and its kernel is
unchanged. This is still below the requested overall performance goal and is
not an independent-engine comparison. Resident mode has no shared-budget ledger,
so these runs do not establish allocation-owner release. Native tests separately
cover 96 shapes twice, independent FP64 checks, old/new N=1 byte equality, tail
canaries and both FullPrecision streaming entry paths. Evidence is ignored under
`artifacts/unified-memory-adaptive/q8-vector-native-abba-v1/` and
`q8-vector-*-v2*`. The opt-in `GgmlOpsQ8VectorBench --benchmark` uses synthetic
weights and reports projection timings separately; those are not model speeds.

## Independent llama.cpp throughput client

`llama-throughput.py` reuses the common greedy request builder and sends the
exact `prompt.json` token array to an independently started server. Its current
contract is Qwen3.5 0.8B Q8, 643 prompt tokens, context/batch/ubatch 2048, F16 K/V,
all layers on GPU, one slot, and no speculation. It does not start or own the
server. Keep the actual server command, startup/offload log, checkpoint SHA,
executable and loaded-library identities alongside the results.

The client requires one warmup and three measured TensorSharp records. Each
llama request uses `cache_prompt=false`, greedy sampling without penalties,
`n_predict=64`, and `ignore_eos=true`. The latter makes this a forced-length
throughput experiment, not complete-answer quality validation. At the supported
llama revision `4ebdf2c74acce30883d8e34b7c70b3eb8146f2fe`, `server-common.h`
defines timed generation steps as `n_gen - 1`. Thus 64 returned tokens match
the probe's 63 decode calls; the first returned token comes from prefill.

Every response must report `timings.cache_n=0`, `prompt_n=643`,
`predicted_n=64`, 64 valid token IDs, `truncated=false`, and length termination.
The client verifies actual sampling settings and the timing denominator too.
`tokens_cached` is final slot occupancy, so it is not used as the cache-hit
counter. All raw response bytes are retained before validation, including
failed responses. Warmup is excluded from median/range calculations. Raw token
histories are compared against each corresponding TensorSharp run; mismatch
returns nonzero but is not mislabeled as a mathematical or language-quality
failure. Token agreement alone establishes neither full-logit equivalence nor
independent mathematical correctness.

Pass a separately recorded server identity JSON with these fields (hash strings
below are placeholders and must be replaced with actual observations):

```json
{
  "source_revision": "4ebdf2c74acce30883d8e34b7c70b3eb8146f2fe",
  "binary_sha256": "actual 64-character executable SHA256",
  "model_sha256": "0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c",
  "command": ["actual llama-server executable", "actual startup arguments"],
  "startup_evidence": "absolute path to actual startup/offload log and identity capture",
  "configuration": {
    "context": 2048, "batch": 2048, "ubatch": 2048,
    "k_cache": "f16", "v_cache": "f16", "all_gpu": true,
    "parallel": 1, "speculative": "none"
  }
}
```

Example client command after starting that server on an otherwise idle GPU:

```powershell
python eng/validation/AdaptiveMemoryProbe/llama-throughput.py `
  --prompt artifacts/unified-memory-adaptive/q8-vector-native-abba-v1/0-control/prompt.json `
  --tensorsharp-report artifacts/unified-memory-adaptive/q8-vector-native-abba-v1/0-control/report.json `
  --server http://127.0.0.1:5099 --server-identity artifacts/llama-qwen/server-identity.json `
  --output artifacts/llama-qwen/comparison
```

`--prepare-only` validates and copies the prompt/reference, then writes the
request without network or device use; identity may be omitted in this mode.
Output directories must be new. The per-request timeout is a socket timeout;
wrap the client with `eng/validation/run-bounded-probe.py --timeout 800` when
a hard process deadline is required. A timeout/incomplete run is not a pass.
The startup manifest is caller evidence, not proof obtainable from HTTP alone.
Cross-engine timings include different overheads and arithmetic; report sample
ranges and execution conditions rather than inferring broad performance claims.

CPU contract checks:
`python -m unittest discover -s eng/validation/tests -p test_adaptive_llama_throughput.py -v`.

The local independent run used that exact Qwen checkpoint/prompt on the RTX 3080
Laptop, with an otherwise idle device and no concurrent builds or CPU inference.
One warmup and three measurements in a fresh llama server gave prefill
8792.20 (8669.27–8972.93) tokens/s and decode 183.98 (181.82–184.69) tokens/s.
The three measured TensorSharp control rows from `0-control/report.json` were
2171.41 and 18.3735 tokens/s respectively: approximately 4.05x and 10.01x slower.
The later TensorSharp N=1 candidate's separately measured 20.12 decode tokens/s
still leaves a substantial gap. All four llama responses had zero cached prompt
tokens, 643 evaluated prompt tokens, 64 returned IDs, 63 timed decode steps and
no truncation; their complete token histories matched the TensorSharp rows.
This does not compare full logits or establish arithmetic/quality equivalence.
The server used logging verbosity 4, all 25/25 layers on CUDA and F16 KV;
startup/device details and actual loaded-library hashes are retained under
ignored `qwen-llama-throughput-server-v2`, with all responses and measurements
under `qwen-llama-throughput-comparison-v2`. The earlier v1 client rejected the
server's duplicate `none,none` speculation serialization; that trial is retained
as failed evidence. The successful v2 run restarted the server and still rejects
every active speculation type.

The earlier Q8 prefill experiment used inherited `TS_GGML_Q8_PREFILL_TILE=64` or
`128`; at that revision unset meant `32`. Current automatic defaults are
documented at the top of this file. N below an explicit selected tile falls back
to a smaller tile. All outputs retain their original K-increasing F32 FMA
order. The experiment reuses decoded weights across more activation columns
within a CTA; it increases register/shared-memory use without a global weight
expansion or extra payload allocation. Explicit overrides remain available;
wider device, shape and mixed-request performance coverage is still incomplete.

Run six fresh processes in order 32/64/128/128/64/32, with
`TS_GGML_Q8_PARALLEL_VECTOR=1`, `TS_GGML_Q8_PARALLEL_SMALL_BATCH=0`, identical
native/managed files, capture off, one warmup and three measured requests each.
Pass their `run-bounded-probe.py` execution records to
`compare-prefill-tiles.py --executions <six execution.json paths> --output <json>`.
The comparator requires distinct process lifetimes, the actual native selection
logs, unchanged other settings, full raw-logit/token equality and valid timing
denominators. Its CPU regressions are in `test_prefill_tile_comparison.py`.

The first local Qwen0.8B run with the same 643-token fixture and 64 prediction
rows had prefill medians 1988.08 / 2346.67 / 2549.06 tokens/s for 32/64/128,
with six measurements each. All complete logits were byte-identical. Decode
medians were 178.61 / 175.78 / 171.85, with overlapping ranges; the reduction
is retained rather than attributed away. This is an approximately 28.2% prefill
gain for the largest tile, still well below the earlier independent 8792.20
baseline. No concurrent inference/build ran during this comparison, but clocks
were not locked and earlier image tests had heated the device. This is not a
statistical performance guarantee or an independent mathematical oracle.
Native SHA: `4ae03489d078b7f971dfdd3113477f9cfdfce5c75ee4a7ad1a237e60283b3f8c`;
upstream ggml is unchanged at `ffa4e8b80930029a35991f94e7c8a93cd67730ab`.
Evidence: ignored `artifacts/unified-memory-adaptive/q8-prefill-tiles-v1/`.
