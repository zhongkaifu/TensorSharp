# Qwen4Exp selected-expert cache validation

For optional Linux CUPTI kernel/API/copy attribution, see
[CUDA activity diagnostics](../cuda-activity-trace.md). `--cuda-trace-library`
enables the recorder; it is off by default and its timings are diagnostic only.

This probe exercises the opt-in `TS_HOST_MOE_EXPERT_CACHE_MB` CUDA cache with the
same eight-layer synthetic Qwen4Exp GGUF fixture used by engine regressions. It
includes PLE, GDN, QSA, mixed quantization, changing teacher-forced routes, reset
A/B/A and disposal/reload through a different model with identical shapes.
It also captures every target-verification row for widths 2, 4 and 8, checks
recurrent rollback and resumes an accepted prefix against a cold replay.
The fixture is seeded noise; it cannot validate language quality or establish
real-model speed or performance parity with Strata.

Build the native library, then the probe and managed tests:

```powershell
$env:TENSORSHARP_GGML_NO_UPDATE = '1'
.\TensorSharp.GGML.Native\build-windows.ps1 --cuda --no-vulkan --tests
dotnet build eng/validation/Qwen4ExpExpertCacheProbe -c Release -p:TensorSharpSkipGgmlNative=true
dotnet build InferenceWeb.Tests -c Release -p:TensorSharpSkipGgmlNative=true
python eng/validation/qwen4exp-expert-cache.py --output docs/validation/qwen4exp-expert-cache
```

The wrapper owns fresh CUDA processes with the cache disabled, a 12 MiB budget
that exercises eviction, and a 128 MiB budget. It also compares staged inputs
and outputs (`TS_HOST_MOE_EXPERT_CACHE_BRIDGE=0`) against input-only transfers
(`TS_HOST_MOE_EXPERT_CACHE_OUTPUT_BRIDGE=0`) and the default direct CUDA input/output
bridges. A fourth variant enables raw miss prefetch
(`TS_HOST_MOE_EXPERT_CACHE_PREFETCH=1`, forced rather than the adaptive default).
Explicit `=0` remains the no-prefetch control. Each variant runs in a fresh
process with explicit flags; the default wrapper now runs sixteen synthetic arms.
It retains complete logits and
checks relative L2 at `1e-6` with equal argmaxes against all-device execution and
between cache budgets. CPU-offloaded arithmetic is reported separately. Any
missing engagement, nonfinite logits, changed deterministic result or exceeded
budget fails. Cache statistics include the conservative graph workspace
allowance; these statistics are the cache reservation rather than total VRAM.
Native provenance is the actual mapped library path and SHA-256, and ggml must
remain unchanged.

Pass `--device-budget-bytes <positive-int64>` with `--backend ggml_cuda` to
attach a shared `MemoryBudget` before loading the model. The value is a separate
capacity for each layer-split rank (`gpu0`, `gpu1`, ...); it is not divided by the
number of devices. Single-device use retains the same `gpu0` meaning. This opts into
cache/preload and TensorSharp's explicitly routed graph-buffer charges. It is
not a cap on every CUDA driver/backend allocation, CPU memory, or OS mmap page.
The final report retains accounting snapshots, actual loaded binary hashes,
environment overrides, operation errors, and cleanup errors. It can pass only
after model disposal, host-cache clearing and reuse-buffer release return all
covered reservations and allocations to zero, the scope detaches, and native
shutdown completes. Failure to release physical ownership leaves the scope
rooted and the report failed rather than refunding live memory.

For pressure/reclamation checks, `--require-all-cache 1` requires every expert
row of a 1..8-token prefill and every subsequent decode call to use the cache.
This is stronger than `--require-cache 1`, which only proves that some rows used
and reused it. Unsupported long-prefill configurations are rejected by this
stricter mode. A native allocation/commit failure is never retried as ordinary
budget pressure; existing rollback tests remain required.

The cache now retries smaller measured slot tables when either shared admission
or observed free VRAM refuses the initial per-layer table. The lower bound holds
all selected experts, and each rejected candidate has no device payload. Warm
entries keep their slots and shared reservation; this avoids doing admission on
every token. Reductions use bounded geometric steps, so this is not an optimal
slot-allocation or fairness algorithm. Existing entries do not automatically grow
when pressure subsides, and insufficient space for the minimum still falls back.
The configured cache ceiling, per-layer partitioning, and native 512 MiB physical
headroom floor remain; callers must budget other workspace/KV/driver allocations.

`GgmlBasicOps.TrimHostMoeExpertCache(targetBytes)` allows a caller to reclaim
whole least-recently-used expert graphs between requests. It drains pending
copies, frees physical graph storage, then refunds shared credit. It preserves
model/KV state, recent entries within the target, and cumulative cache counters.
The target is process-wide across ranks; stop model work on **every** rank first.
It is not a new persistent ceiling, a driver-memory measurement, or an eviction
hook invoked from inside a budget callback. Subsequent requests can refill the
configured cache. Automatic service-wide pressure scheduling is not supplied by
this API.

Pass `--trim-target-bytes 0` to empty this cache before each measured iteration,
after warmup. The report records before/after cache accounting and shared-budget
snapshots. Pair it with full-logit capture and the comparator below. For example,
on the eight-layer fixture (actual allocator sizes are platform dependent):

```powershell
$env:TS_HOST_MOE_EXPERT_CACHE_MB = '32'
$env:TS_HOST_MOE_EXPERT_CACHE_LAYERS = '8'
$env:TS_HOST_MOE_PIN = '0'
dotnet eng/validation/Qwen4ExpExpertCacheProbe/bin/Release/net10.0/Qwen4ExpExpertCacheProbe.dll --output artifacts/expert-pressure/model.json --prefill-tokens 4 --decode-tokens 32 --warmup 1 --iterations 2 --device-budget-bytes 26000000 --require-cache 1 --require-all-cache 1 --trim-target-bytes 0 --logits-dir artifacts/expert-pressure/logits
```

Use a second fresh process with ample shared capacity and no trim as the control;
compare **each** measured iteration, with all other binaries and conditioning
unchanged. Captured runs test arithmetic and ownership, not quiet throughput.
Native `host-moe-expert-cache-*` tests additionally cover exact minimum/one-byte
refusal, injected physical-availability limits (not actual CUDA OOM), allocation
and commit rollback, three-entry LRU order, and queued device-output retirement.

For a controlled prefill comparison, retain the first run's prompt IDs using
`--prompt-tokens-output <file>`, then provide that exact file through
`--tokens-file <file>` in both new processes. Compare explicit
`TS_HOST_MOE_DEVICE_MIN_BATCH=9` and `=0` while keeping model, shared capacity,
private expert-cache cap, generation inputs and native binary identical. These
are opt-in diagnostics; the probe does not change the product's default policy.

Add `--logits-dir <fresh-directory>` to retain the full vocabulary prediction
after the prompt and after each consumed teacher/generated token. `rows.f32`
contains contiguous little-endian float32 rows; `rows.json` records byte offsets,
element counts, SHA-256, argmax, iteration and the exact input history for each
row. Empty/nonfinite rows fail immediately. Capture I/O is outside the forward
stopwatches, but the resulting run is diagnostic and is not a quiet performance
measurement. An interrupted run preserves its completed row index; that does not
turn the incomplete model report into a pass.

For a real GGML CUDA checkpoint, `--layer-split 2` runs contiguous layers on two
devices and records the degree in model geometry. Set `CUDA_VISIBLE_DEVICES`
explicitly and use matching placement in both capture processes. The synthetic
fixture rejects this option. With `--device-budget-bytes`, every selected rank
has its own pool in the same ledger; all must release their covered allocations
before the scope can detach. `--host-budget-bytes` remains one shared staging
pool, not one duplicate allowance per rank. Check per-pool snapshots and actual
engagement; this still does not establish a whole-model or process memory cap.

The probe records `TS_Q4E_FUSED_GLU`, `TS_GGML_UPLOAD_PREFETCH`,
`TS_GGML_PHASE_TIMING` and `TS_GGML_LOG_VRAM` so fresh-process captures can
qualify the default CUDA decode and loading paths. Set `TS_Q4E_FUSED_GLU=0`
for the separate SiLU/multiply control; the canonical GLU default is limited
to ordinary single-token CUDA FFNs without CPU experts or tensor parallelism.
On Linux CUDA, large admitted device-cache weights automatically overlap cold
page preparation with chunk uploads. `TS_GGML_UPLOAD_PREFETCH=0` disables this;
`=1` forces preparation for eligible weights, including resident sources.
Unset uses residency and CPU-parallelism checks. `TS_GGML_LOG_VRAM=1` records
the actually prepared byte ranges and bounded read windows. Logging/capture
runs are diagnostics, not quiet throughput measurements. Neither flag creates
a whole-process RAM cap or an asynchronous CUDA compute/transfer pipeline.

Compare two routes' matched captures, including the initial prefill row:

```powershell
python eng/validation/qwen38-compare-captures.py --left artifacts/flash-cpu/logits/rows.json --right artifacts/flash-device/logits/rows.json --left-report artifacts/flash-cpu/model.json --right-report artifacts/flash-device/model.json --output artifacts/flash-routes.json
```

Every captured row must meet relative L2 `1e-6` and equal argmax. Both bound model
reports must have completed and physically cleaned up successfully. The comparator
checks requested iteration/decode counts, every successive input history, model
geometry, checkpoint identity, complete file layout and the final prediction hash.
Available rows from incomplete executions may produce diagnostic differences,
but cannot produce a passing comparison, even if both files are equally truncated.

For a real checkpoint, pass `--model-identity-report <integrity.json>` to each
probe run. This accepts the completed publisher-hash report from the local
multimodal preflight tool, requires all GGUF shards, and checks their current
size and nanosecond mtime before loading. The probe records the manifest hash and
each verified shard hash without re-reading tens of GB during every arm. This
reuses earlier full verification; metadata equality is not a new content hash.
Reverify a checkpoint after modification. Synthetic checkpoints record their
complete file hash directly. Actual native and managed hashes remain in both
final reports; route agreement is not a claim of identical engine versions.
The input schema has a `files` array; a download report with only `shards` is
not interchangeable. Preserve the prior hash provenance and verify current
size, mtime and GGUF split metadata when preparing a compatible manifest.

For an independent matched-history diagnostic, start an unchanged llama.cpp
server on the same verified GGUF and retain its revision, binary hash, backend,
placement flags and logs. Then replay the captures without retokenizing:

```powershell
python eng/validation/qwen38-llama-teacher.py --logits-index artifacts/flash-cpu/logits/rows.json --model-report artifacts/flash-cpu/model.json --server http://127.0.0.1:5099 --output artifacts/flash-llama-teacher
python -m unittest discover -s eng/validation/tests -p test_qwen38_llama_teacher.py
```

The HTTP reference exposes float32 pre-sampling log probabilities rather than
raw logits. The tool compares values after subtracting the same reference
token's value in both engines, preserving pairwise logit differences while
removing the unobservable normalization constant. It requires a complete,
finite vocabulary and rejects softmax underflow/clipping; partial probabilities
cannot establish full-logit agreement. Raw responses and every exact request
history are retained. This is a localization tool, with no assumption that
either engine is ground truth and no automatic model-quality pass.
`--prepare-only` validates the TS captures and writes requests without inference.

Run the focused managed regressions in a fresh process:

```powershell
$env:TS_TEST_GGML_BACKEND = 'cuda'
$env:TS_HOST_MOE_EXPERT_CACHE_MB = '12'
$env:TS_HOST_MOE_EXPERT_CACHE_LAYERS = '8'
dotnet test InferenceWeb.Tests -c Release --no-build --filter 'FullyQualifiedName~Qwen4ExpExpertCacheTests'
```

Timing excludes prefill from decode and uses identical teacher-forced tokens.
Warmup and timed repetitions are retained separately. GPU telemetry includes
other processes under Windows WDDM; process peak working set is an OS measure.
Repeat with rotated arm order on quiet hardware before claiming a speedup.

For a complete real GGUF checkpoint, `--model <first-shard>` measures CPU-offloaded
decode with and without the cache and skips the all-device load. It does not
run the synthetic diagnostics. Transfer variants and cache budgets must agree on
the complete final vocabulary row at relative L2 `1e-6`, equal argmax and exact
prompt, forced and generated token IDs. The default wrapper checks the final
row; opt into `--logits-dir` and the separate capture comparison above to check
every decode row. Neither numerical comparison claims trained quality. Every real greedy warmup
and timed repetition must finish at EOS with consistent generated IDs. The CPU
baseline is reported separately from the strict cached-variant comparisons.
Use matching
tokenized prompts, quantization, context, speculation settings and separately
validated complete outputs when comparing TensorSharp with Strata.

For repeated trained-request transfer comparisons, pass an exported prompt ID
file to every arm and select greedy generation. For example:

```powershell
python eng/validation/qwen4exp-expert-cache.py --output docs/validation/qwen4exp-real-transfers --model C:/Works/models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf --generation greedy --tokens-file docs/validation/qwen38-strata-smoke/code/0/prompt.ids --cache-mb 8192 --decode-tokens 256 --warmup 1 --iterations 3 --timeout 1800
```

The default generation mode remains `teacher-forced`. `--skip-bridge-comparison`
omits the transfer variants; real validation then requires at least two cache
budgets. `language_quality_validated` stays false until complete answers are
checked independently.

For trained greedy generation, the probe renders the checkpoint's GGUF chat
template through the same public renderer as the CLI and can export the exact
prompt IDs for Strata. For example:

```powershell
dotnet eng/validation/Qwen4ExpExpertCacheProbe/bin/Release/net10.0/Qwen4ExpExpertCacheProbe.dll --model D:/Workspace/Models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf --output docs/validation/qwen4exp-real/math.json --generation greedy --prompt 'What is 17 plus 25? Answer with the number only.' --prompt-tokens-output docs/validation/qwen4exp-real/math.ids --decode-tokens 32 --iterations 1 --warmup 0 --placement host
```

`--tokens-file` accepts comma/whitespace separated token IDs verbatim.
`--prompt-raw-file` tokenizes already rendered ChatML, while `--prompt-file` reads
an ordinary user prompt and renders it. Reports retain all prompt/generated token
IDs, decoded output, EOS or length termination, the complete final logits and
their hash. `selected_text` excludes terminal EOS text; `raw_decoded_output` and
the complete generated token IDs preserve it. Greedy decode timing excludes the first token produced by prefill;
the timed decode count is the number of subsequent forward calls. Successful
execution alone does not assert that an answer is correct. The real model's
memory and speed must be measured separately from the synthetic fixture.

The matched trained comparison runner exports TensorSharp's rendered prompt IDs,
feeds them to fresh cache-disabled/cache-enabled processes and Strata, and checks
EOS-complete math, extraction, Python-function and ordered square answers
independently. The square oracle covers 1 through 20. Strata's extracted checkpoint
tokenizer decodes both engines' output IDs.
For example, after preparing Strata's unmodified build and native-expert pack:

```powershell
python eng/validation/qwen38-strata-compare.py --output docs/validation/qwen38-strata-smoke --model C:/Works/models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf --pack C:/Works/models/Qwen3.8-Flash-Next-GGUF/strata-pack --strata artifacts/strata-qwen38/strata-build/strata.exe --strata-root C:/Works/Strata --strata-dependency artifacts/strata-qwen38/llama-pinned --cache-mb 4096 8192 --context 512 --max-new 64 --cases math
```

Omit `--cases math` for all four semantic cases and supply `--max-new 256` (or
`--max-new 128` when deliberately testing the shorter completion budget). The
longer squares answer needs more generation space than a 64-token limit;
length-limited answers fail. `--repetitions` rotates engine
ordering after the first repetition; `--timeout` controls each process deadline.
`--initial-cache-mb` can run a tested cache budget first in supplemental runs,
so comparisons can reverse the order of the cache sizes. The first TensorSharp
arm still exports the exact GGUF-rendered prompt IDs.
For a single-case supplement, `--strata-first --prompt-tokens-file <previous
case/prompt.ids>` completes the remaining rotation without an untimed inference
warmup. The runner validates the selected prompt against decoded IDs and records
the token-file hash. Repeated runs retain OS file-cache history; fully rotating
the order does not make them cold-storage benchmarks.
`--verified-download-report` checks the downloader's successful publisher hashes
against every current shard path and size, without rereading all model weights.
Positive Strata resident CPU budgets require a static expert profile; the runner
uses the tracked `data/expert-profile.bin` by default or accepts
`--strata-expert-profile`, and records its identity.
Stock Strata native IQ experts require verify windows on some paths. The runner
defaults the verifier capacity (`--strata-spec-window`) to 2 and caps actual target
execution with `--mtp-max-t 1`. It supplies no MTP/oracle pack and explicitly
disables suffix drafting. A window histogram must prove that every target window
has one row and zero drafts before the report accepts scalar greedy execution.
Verifier setup still differs between engines and the rounded dense projections
can affect their arithmetic. `--strata-spec-window 0` requests a compatible
non-verifier scalar path; an unavailable native mode fails the run.
Incomplete answers, unavailable models/engines, missing arms and timeouts fail.
Reports keep full TensorSharp logits, mapped native/managed hashes, actual Strata
binary hash, pack conversion records, timing denominators, working set and GPU
telemetry. Strata counts its first generated token in decode; TensorSharp's probe
counts subsequent forward calls. Compare these labelled results and whole-process
latency without treating the denominators as identical. The optional
`--dump-strata-logits` adds diagnostic I/O and parses only complete known stride-1
positions; its timings cannot serve as quiet throughput. Rounded dense BF16 pack
conversions preclude a cross-engine bit-exact claim. This small semantic suite
does not establish broad language quality or performance parity.

All generated evidence belongs in ignored `docs/validation/` or `artifacts/`.

## Serial Flash decode comparison

Every request now hashes the complete vocabulary after prefill and every decode
call, outside forward timing, even without `--logits-dir`. Repetitions must have
identical complete histories. The report includes `full_logit_chain_sha256`,
per-step decode milliseconds and expert hit/miss counters at phase boundaries.
This proves unchanged output for the tested history; it is not an independent
mathematical or broad language-quality oracle.

Freeze two probe directories with identical managed binaries and separate native
libraries, then run the old/new/new/old sequence without concurrent GPU workloads:

```powershell
python eng/validation/flash-decode-bench.py --model C:/Works/models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf --model-identity-report artifacts/multimodal-local-preflight/integrity.json --control-dir artifacts/flash-control --candidate-dir artifacts/flash-candidate --output artifacts/flash-abba
```

The tool uses a 2 GiB expert cache, 8 CPU threads, context 512, one separately
reported first request and three measured repetitions in each process. Its default
order can be reversed with `--candidate-first`; preserve both sequences when
checking order sensitivity. This balances order, but does not reset OS pages.
The default 32-step teacher-forced history is a microbenchmark. For actual generation pass
`--generation greedy --prompt <text> --decode-tokens 256`; optionally supply
`--expected-output <answer>` to require that exact trimmed answer and EOS in every
request. Greedy comparisons always require EOS, including without an expected
answer. Generated outputs and timing denominators remain in the raw reports.

`compare-flash-decode-runs.py` validates checkpoint/managed/native identities,
settings, serial execution, all vocabulary histories including warmup, expert
reservation and physical ownership cleanup. Its `passed` means comparable,
unchanged output, not a speedup: regressions are retained in the result. First
requests are not controlled cold-storage tests; OS pages are not flushed. Process
peak working set and one-second whole-board VRAM samples are observed consumption,
not total RAM/VRAM caps; board samples include the desktop and can miss spikes.
Do not compare a three-call first request directly with warmed long-generation
throughput or claim independent engine parity from these checks.

For a separate capacity experiment use `--expert-cache-mb <MiB>` with a measured
workspace/KV/driver allowance appropriate to the device and request. Each ABBA
suite keeps that ceiling identical on both sides; changing it between suites
measures capacity effects, not a native-code speedup. A small fixed expert cache
can leave most VRAM unused while repeatedly reading and uploading the same
weights. These measurements do not supply an automatic whole-system RAM policy.

The native page reader now queues each completed gate/up/down projection for
upload on the submitting thread while bounded workers prepare other sources.
It waits for all workers even after read/upload exceptions. The native pool
tests force read/consume overlap, caller-thread consumption, exception recovery,
and mixed concurrent submissions; the expert-cache tests reverse completion
order and inject interrupted uploads to detect stale valid victim slots.

On Windows, `TS_HOST_MOE_FILE_READ=1` registers exact GGUF shard extents and
reads missing expert projections into one reusable host transfer arena, at most
32 MiB per process. CUDA host allocation permits reads and uploads to overlap;
upstream falls back to pageable RAM if pinning is unavailable. The arena grows
only to observed demand, drains before reuse after failures, and is released by
an empty-cache trim. This is separate from the device expert quota and is not a
whole-process RAM limit. The registered extents and handles retire on source
invalidation or teardown. A live model's `ReleaseGgmlDeviceResidency()` restores
its extents after invalidation so later execution can continue staging.

Without this variable, mapped CUDA experts exceeding available physical RAM at
load time select file staging when a valid, positive expert-cache budget exists.
`=0` preserves mmap reads and `=1` forces registration. This initial source policy
does not make the expert quota or total RSS automatically adapt to every request.
The file path currently covers the compact expert cache, not CPU long-prefill
reads, all model weights, KV, or other backends.

Pass `--candidate-file-read` to `flash-decode-bench.py` (and its standalone
comparator) to compare explicit mmap/file arms with the same adaptive mmap
prefetch setting. Both arms can use one frozen probe directory. The comparator
requires actual native file-read and bounded-workspace evidence, besides all
output/budget checks. The default prefetch comparison explicitly disables file
staging on both sides. Use `--release-residency true` directly on the probe for
a separate release/refill correctness run; its repetitions include reloading
device resources and must not be labelled warm throughput.

For I/O diagnosis, wrap a probe command with
`python eng/validation/flash-decode-io.py --output artifacts/flash-io -- dotnet ... --process-counters true`.
This requires `psutil`. The Windows phase snapshots distinguish process I/O,
page faults (including soft faults), private commit and working set. Physical
disk samples include every process; process I/O counters omit some mmap I/O.
Neither is a controlled cold-storage measurement or an attribution of every
SSD byte to model weights. Keep this diagnostic sampling separate from quiet
throughput measurements.

The independent Strata smoke runner also supports `--cases tool_json`. It
requires one complete JSON tool call with exact tool/argument values and no
duplicate keys; it does not execute a tool or validate a multi-turn agent loop.

`flash-decode-report.py --input <annotations.json> --output docs/validation/<run>`
renders an HTML/JSON report from completed comparisons. The input has `title`,
`notes`, `groups` (each with `name` and comparison `directories`), optional
`semantic_report`, and optional `sections` (each with `title` and `notes`). Paths
resolve relative to the input. Each group must use one binary/checkpoint/request
identity; the tool rechecks all comparisons, rejects overlapping processes or
different full vocabulary histories, separates first and repeated requests, and
preserves every independent semantic failure. It does not infer overall quality
or performance acceptance from a successful transport comparison.
An optional group `semantic_case` uses a case from `qwen38-strata-compare.py`;
the measured prompt must match exactly. Every request, including the first, must
satisfy that independent task check and EOS. Failures appear in the report without
changing the original output or the separate transport-comparison result.

Pass `--candidate-lfu` to compare legacy LRU against capacity-dependent decaying
frequency eviction. Both arms then force file staging and use identical device
quotas; the comparator also checks native policy logs. This option and
`--candidate-file-read` are mutually exclusive. Other experiment modes explicitly
disable LFU on both arms to avoid conflating the two changes. Production defaults
to LFU within an enabled expert cache; `TS_HOST_MOE_EXPERT_CACHE_LFU=0` restores LRU.

For diagnostic route captures, `flash-expert-route-capture.py` accepts
`--binary-dir`, `--model`, `--model-identity-report`, `--prompt` and `--output`.
It forces LRU and `TS_HOST_MOE_ROUTE_TRACE=1`, then invokes
`flash-expert-cache-replay.py` to compare miss counts at the recorded layer
capacities. LRU replay must exactly match native calls/hits/misses first. Tracing
prints each route and changes timing; neither capture timings nor replayed miss
counts establish throughput or language quality.

The probe's `--host-budget-bytes` requires `--device-budget-bytes` and installs
a host-pool mapping on the same budget. Host coverage is the expert file arena
payload only. `flash-host-budget-check.py` exercises a 32 MiB host allowance with
complete Chinese requests, residency release, cache trim and refill; a second
1-byte allowance must fail with the native budget cause and leave no credit.
It takes `--binary-dir`, `--model`, `--model-identity-report` and `--output`;
`--scenario admitted|refused` allows a targeted retest. These lifecycle timings
are not warm throughput. Default file-source and LFU selection stay unset so
the positive run checks actual defaults, including after reload.

Device scopes with graph coverage also charge Qwen4Exp graph arenas, recurrent
state and device state snapshots. They do not cover all model KV, driver/backend
pools or host allocations. `Qwen4ExpGraphBudgetTests` verifies snapshot charges,
rollback equivalence, refusal under exhausted credit, batched retry and release.

Single-token CUDA hyperconnection and PLE broadcasts use read-only zero-stride
views by default. `TS_Q4E_BROADCAST_VIEWS=0` restores materialized repeats for an
isolated comparison; it is independent of `TS_Q4E_DECODE_VIEWS`. Batched and
multi-token graphs keep their existing layouts. Record the switch and native
binary identity with full-vocabulary logits before comparing throughput.
