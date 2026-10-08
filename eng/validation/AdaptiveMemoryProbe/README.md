# Adaptive memory execution

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
differ. Use `--control` for the first/last reports and `--candidate` for the middle
two reports in a control/candidate/candidate/control sequence. It requires full
raw-logit equality, excluded warmups and at least two fresh processes per arm.

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

The later Q8 prefill experiment uses inherited `TS_GGML_Q8_PREFILL_TILE=64` or
`128`; unset/`32` preserves the default. N below the selected tile falls back
to a smaller tile. All outputs retain their original K-increasing F32 FMA
order. The experiment reuses decoded weights across more activation columns
within a CTA; it increases register/shared-memory use without a global weight
expansion or extra payload allocation. It remains opt-in pending wider device,
shape and mixed-request performance coverage.

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
