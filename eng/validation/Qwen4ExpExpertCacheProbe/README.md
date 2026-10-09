# Qwen4Exp selected-expert cache validation

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
(`TS_HOST_MOE_EXPERT_CACHE_PREFETCH=1`, off by default). Each variant runs in a fresh
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
attach a shared rank-0 `MemoryBudget` before loading the model. This opts into
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
