# Full-model CUDA validation probe

This fresh-process probe loads a real GGUF through `ModelBase`, measures fixed
teacher-forced prefill/decode, then performs a separate untimed replay capturing
every full logit row. It also checks complete greedy answers to arithmetic,
extraction and a square-list prompt through the checkpoint's no-thinking chat
template. These small semantic checks do not establish broad language quality.

Build after updating native code and PTX, using an unchanged upstream ggml tree:

```powershell
$env:TENSORSHARP_GGML_NO_UPDATE = '1'
dotnet build eng/validation/DirectCudaModelProbe -c Release -p:TensorSharpSkipGgmlNative=true
```

Only use `TensorSharpSkipGgmlNative` after independently rebuilding native code
when native sources changed. A direct CUDA build uses `nvcc` on PATH to compile
current kernels for the local GPU. Verify that the loaded module includes the
new entry point and supports the tested GPU; falling back to an older module
does not validate fusion. The report hashes copied PTX files and assemblies.

## Optimization switches and coverage

Both switches are read once per process; start a new process for each arm.

| Switch | Default | Reference / candidate |
| --- | --- | --- |
| `TENSORSHARP_CUDA_MOE_FUSION` | Enabled | `0` restores separate activation/quantization launches and separate shared-MoE scratch allocations; `1` enables fusion and scratch reuse. |
| `TS_Q4E_PREFILL_COMBINE` | Enabled | `0` uses the original ggml Qwen4Exp combine; otherwise the TensorSharp-owned CUDA prefill combine runs where eligible. |

The ggml reduction is enabled only for CUDA token spans with 64–65,535 tokens
and at least one resident expert layer. The 64-token deployment cutoff follows
full-engine measurements; it is not a claim of an optimal threshold on every
device. Smaller prompts and smaller final prefill chunks retain their original
graph even when the switch is `1`. The standalone helper's numerical tests also
exercise 9–63-token shapes, without enabling them in model execution.

Direct CUDA fusion covers the quantized expert SwiGLU paths in the shared
whole-model MoE code, Dsv4, and generic Qwen decode/batched execution. Scratch
reuse applies only to `CudaMoeScratch`, used by the GLM and Qwen4Exp engines:
gate/up and down activation buffers share capacity because their lifetimes do
not overlap, and the interleaved/split layouts are mutually exclusive. Dsv4's
separate scratch and generic Qwen3.5/3.6 scratch are unchanged. The direct CUDA
MMA prefill path retains its existing float activation operation.

The committed PTX includes the fused entry point, regenerated with CUDA 12.8.93
while preserving PTX 8.7 and the existing `sm_120` target. A build without `nvcc`
can use this bundled fusion on compatible hardware and drivers. SM120 was
cross-compiled only; it was not executed on the RTX 3080 Laptop GPU (SM86).
Compile with `nvcc` on PATH to produce PTX for the local architecture and deploy
the resulting `cuda_kernels` files; the deployed application itself does not
need a compiler. Older deployed modules that lack the entry point still safely
select the original two launches, with shared-MoE scratch reuse retained.
Focused fusion tests and the `--moe-fusion` benchmark explicitly reject modules
without the fused entry point.

Coverage includes direct split-layout quantization parity, per-slot experts,
grouped warp experts, and MMA prefill followed by decode. It does **not** include
a complete staged-expert dispatch: the current staged width/type requirements
are a subset of the earlier MMA branch, so that staged branch is unreachable.

An available 11.8 GB Qwen3.6 MoE checkpoint supports the resident direct CUDA path
on the 16 GB RTX 3080 Laptop GPU used for this work. Its expert down-projection
types are IQ3_XXS, IQ4_XS and Q3_K, which do not select the generic Qwen MoE DP4A
down path changed by this fusion. This checkpoint is a trained-model regression
control; timing differences on it do not establish fusion gains. A checkpoint's
filename quantization label does not identify each expert tensor's type. Inspect
the actual down-projection types, supported dimensions and selected dispatch
before claiming engagement. Generic Qwen fusion requires a supported DP4A down
projection (Q4_0, Q4_K, IQ2_XXS or IQ2_S), with its policy enabled and valid scratch.
Verify available memory and actual placement in each run. Run baseline and optimized
arms in separate processes, alternating arm order across independent repetitions:

```powershell
$env:TENSORSHARP_CUDA_MOE_FUSION = '0'
dotnet eng/validation/DirectCudaModelProbe/bin/Release/net10.0/DirectCudaModelProbe.dll --model C:/Works/models/mtp/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf --backend cuda --prefill-tokens 64 --decode-tokens 64 --warmup 2 --iterations 5 --max-new 128 --output artifacts/strata-perf/baseline-r1.json
$env:TENSORSHARP_CUDA_MOE_FUSION = '1'
dotnet eng/validation/DirectCudaModelProbe/bin/Release/net10.0/DirectCudaModelProbe.dll --model C:/Works/models/mtp/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf --backend cuda --prefill-tokens 64 --decode-tokens 64 --warmup 2 --iterations 5 --max-new 128 --output artifacts/strata-perf/fused-r1.json
```

The probe accepts `--backend ggml_cuda` and `--backend ggml_cpu`, and optional
`--moe-cpu-layers N` (or `all`) sets explicit expert offload before model loading.
For Qwen3.8 Flash Next on this host, `--moe-cpu-layers 47` leaves one of its 48
expert layers on the GPU. All-host expert placement does not exercise a resident
expert reduction kernel. Confirm actual placement and kernel engagement in the
retained process log and any implementation diagnostics.

`--ggml-host-staging-budget-bytes N` optionally attaches one shared
`MemoryBudget` before GGML model construction. It accepts nonnegative byte counts
only with `ggml_cuda` or `ggml_cpu`; **zero is a strict zero quota**. Omitting the
argument installs no budget. For example, `8388608` allows 8 MiB of instrumented
native host staging payload across all uploader workers. Set native loader
environment variables before launching the process, especially on Windows.

The device-cache pool is separate and has `long.MaxValue` capacity; graph buffers
are excluded. This option does not cap model weights, mmap/page-cache residency,
tokenizer/GC, FILE/thread/allocator metadata, backend pools or driver memory, and
is not a complete physical RAM cap. The native library must instrument the
specific staging path; a zero host-payload high-water mark does not prove that
the GLM uploader participated.

Explicit host-budget runs dispose the model, then detach the callback scope,
before writing a passing report. The report records pool high-water marks,
bounded first/last allocation refusals, callback errors and final owner counts.
A companion `.host-staging-budget.json` also preserves these observations when
construction or cleanup fails. Cleanup failure remains a probe failure; retry
does not erase its history. The model-load timer includes optional scope
attachment; forward timing excludes final cleanup and report serialization.

Use `--synthetic-q4e artifacts/strata-perf/fixture.gguf` instead of `--model` to
generate a deterministic eight-layer Qwen4Exp engine fixture. This automatically
disables quality generation. It supports `--synthetic-q2kxl true`,
`--synthetic-experts N`, `--synthetic-experts-used K` and
`--synthetic-indexer-top-k N` (defaults: false, 16, 4 and 16). Both arms must use
the same settings and the recorded fixture checksum must match. This mode
exercises full engine loading and execution but makes no trained-model claim.

`--quality false` skips semantic generation and explicitly reports it as
unchecked. `--max-context` defaults to 1024. The probe clears speculative decoding
switches to hold the generated work constant. Timing excludes logits serialization
and semantic checks; each `Forward` returns host logits, so the timed token work
has completed before its time is recorded. Timed repeats include the same prompt
and forced tokens, following two warmup repeats by default.
Forced decode traverses the token pool with step 37 and offset 5. Step 37 is
coprime to the observed 17-token trained-model pool and the 251-token synthetic
pool, providing varied inputs. Exact prompt and forced token IDs are recorded
and must match across arms. Older diagnostic reports used step 17 and remain
separate; their timings cannot be mixed with this workload.

For GPT-OSS expert-copy policy validation, a long ordinary prompt does not prove
that the per-expert fallback ran: whole-model prefill also accepts long chunks
and uses the original stacked weights in both copy-policy arms. Use the explicit
**probe-only** `--gptoss-per-expert-prefill true` option for a controlled fallback:

```powershell
# Run each arm in a fresh process with the SAME built probe/native libraries.
$env:TS_WEIGHT_FUSION_COPIES = '1' # reference; remove this variable for candidate
dotnet eng/validation/DirectCudaModelProbe/bin/Release/net10.0/DirectCudaModelProbe.dll --model C:/Works/models/gpt-oss-Q8_0.gguf --backend ggml_cuda --gptoss-per-expert-prefill true --prefill-tokens 1025 --decode-tokens 2 --max-context 2048 --warmup 0 --iterations 1 --quality false --output artifacts/gptoss-fallback/reference.json
```

Substitute the actual checkpoint path; keep every other argument identical in
the candidate, using a different output path. The option rejects non-GPT-OSS
models, non-GGML-CUDA backends, synthetic fixtures, `N<1025`, and semantic mode.
Before model loading it sets `TS_PREFILL_CHUNK`, `TS_GPTOSS_PREFILL_CHUNK`, and
`TS_GPTOSS_FUSED_ATTN_MAX_SEQ` to the requested exact prefill width. After loading,
reflection disables only the instance's whole-model prefill. Fused per-layer
attention remains eligible; MoE at `N>1024` must take the per-expert path.
Decode retains its normal dispatch. No production model API or feature flag is
introduced, and this invocation does not simulate an allocator failure.

After each timed prefill (outside the timer) and the separate logit replay, the
probe checks newly written per-expert routing scratch: its final assignment end
must equal `N * NumExpertsUsed`, with monotone nonnegative ends. A sentinel before
each call prevents stale scratch from a prior replay proving engagement. The
`gptoss_per_expert_prefill` report records this evidence and verified call count;
it does not claim hardware profiling or per-layer counters. Compare all captured
full-vocabulary rows and identical teacher IDs across arms. The probe's own
`passed` checks repeat stability, not cross-arm equality. Any difference needs
investigation against the actual per-expert arithmetic; do not loosen a gate or
infer byte equality merely because concatenated and split projections are
algebraically equivalent. This diagnostic may be much slower than ordinary
stacked execution and does not establish a default-path performance result.

If both forced per-expert arms exhaust device memory, use the same probe-only
`--ggml-device-copy-budget-bytes 1` in both fresh processes. This calls the
existing GGML cache API after model loading and before the first Forward, then
verifies the applied rank-0 value through native telemetry. `0` means unlimited;
negative values and non-GGML backends are rejected. One byte prevents new lazy
weight copies, so cache misses use the existing per-graph upload path with the
same quantization, input and projection shapes. It does not clear explicitly
preloaded weights (their host copies may already be released), evict any copies
created while loading, or cap graph/KV/driver allocations. Therefore a successful
run establishes fallback numerical coverage under this cache policy, not a
16-GiB total-memory guarantee or ordinary-path performance. Preserve the failed
unlimited-cache runs as separate evidence. The `ggml_device_copy_budget` report
records the requested cap and before/after counters; every `memory` snapshot
also records current native `ggml_cache` counters. Missing telemetry or a changed
cap fails the explicit diagnostic instead of silently ignoring its request.

Compare the ordinary or forced-fallback expert-copy arms with the same compiled
probe/model/native libraries, explicit `TS_WEIGHT_FUSION_COPIES=1` for the
reference, and that variable **unset** for the automatic candidate:

```powershell
python eng/validation/compare-strata-perf.py --policy gptoss-expert-copies --before artifacts/gptoss-fallback/reference.json --after artifacts/gptoss-fallback/candidate.json --output artifacts/gptoss-fallback/comparison.json
```

This validates capture checksums, complete logits, teacher tokens, repeat/warmup
hashes, captured environment, fallback engagement and the applied cache policy.
An explicit `copies=0` selects the split arithmetic and is a different experiment.
The default performance gate is 5% latency regression; do not relax the numerical
gate to pass a difference. A single process pair under background activity is
limited timing evidence even when all correctness checks pass.

`--quality-parser raw` preserves the historical default. Use
`--quality-parser production` to apply the model architecture's actual output
parser, initialized with the rendered generation-prompt suffix and thinking
disabled. This is required for GLM-5.3-Flash: its template opens a reasoning
block even for a thinking-disabled request. The report retains the entire raw
reply, parsed user-visible content, reasoning, tool calls and actual EOS status.
The same independent answer checks run on the selected content; an unfinished
generation or an unexpected tool call cannot pass. No arbitrary prose is removed.

`--quality-cases math,extraction,squares` remains the historical default suite. A
nonempty explicit subset, such as `--quality-cases math,extraction`, enables
staged runs on a slow file-backed checkpoint. The exact selection and
`--max-new` limit are recorded. A subset does not establish coverage of omitted
cases, and reaching the generation limit is a failure, including when reasoning
uses the available token budget. Do not increase that limit retrospectively and
discard the original result.

Explicit `--quality-cases code,tool_json` adds the same fixed questions used by
the Qwen comparison, without altering their wording or adding a grammar/schema
constraint. `code` requires the pure `square(x)` function returning `x*x`;
generated code is never executed. The managed checker recognizes a single
return statement, ordinary function headers, comments, redundant expression
parentheses and implicit continuation inside them. The independent
`fixed-task-semantic.py` checker
compares the entire Python AST to the expected function; an otherwise valid
format outside the managed grammar must be reviewed as a checker disagreement,
not automatically attributed to model quality. `tool_json` requires exactly
`{"name":"get_weather","arguments":{"city":"Hangzhou","unit":"celsius"}}`,
with JSON whitespace/key-order variation allowed. Markdown fences, repeated or
extra keys, extra text and wrong values fail. This is a raw JSON representation
task, not a real tool invocation or agent roundtrip.

Both new cases additionally require closed thinking when the rendered prompt
opens `<think>`, no parsed tool calls and actual checkpoint EOS within the
unchanged output bound. `thinking_closed`, original text and token IDs remain
in the report, so truncation, structure and semantic failures stay distinct.
Use `--quality-parser production --quality-reasoning-effort low --max-new 128`
for the GLM qualification. The old three case prompts/default selection and
answer rules are unchanged. Versioned policies identify the expanded tools:
`direct-answer-only-v3` and `fixed-answer-only-v2`. Historical frozen apps and
reports retain their original policy; the new tasks need a new app identity.

`--quality-reasoning-effort low|medium|high` explicitly selects the production
renderer effort. Omission preserves the historical family default; malformed or
empty explicit values fail before model loading. The report records the requested
effort and, for every semantic case, the exact rendered prompt, its UTF-8 SHA-256,
and token IDs. This option affects only the semantic prompts, not teacher-forced
timing. GLM-5.3-Flash maps `low` to `Low`, `high` to `High`, and `medium` or omission
to `Max`; it retains the open thinking block even with `enableThinking:false`.
Low effort is a model instruction, not a hard reasoning token cap. In particular,
it is not equivalent to llama.cpp's `reasoning_budget_tokens=0`, which forces the
reasoning end tokens. A like-for-like llama control must request the same effort
with an unrestricted reasoning budget and retain actual EOS/completion checks.

The parser and case-selection checks run without a model or GPU:

```powershell
dotnet run --project eng/validation/DirectCudaModelProbe.ParserTests -c Release
```

For a complete, independently SHA-verified GLM checkpoint on a memory-constrained
Windows host, first run a bounded 8-token prefill / 1-token decode pilot with
`--backend ggml_cuda --prefill-tokens 8 --decode-tokens 1 --warmup 0 --iterations 1
--max-context 1024 --quality false`. Leave expert placement at its default
capacity policy. Then run the semantic cases with `--quality true
--quality-parser production --quality-reasoning-effort low` and an explicit generation budget. Record loader
placement, loaded native SHA, process exit and the full shard manifest. The
metadata-only first shard does not establish checkpoint completeness. On USB
HDD with more weights than RAM, report file-backed storage and cache limitations:
a fresh process is neither proof of cold pages nor of a fully warm checkpoint.

JSON fields for an A/B runner:

- `runs`: warmup flag, prefill/decode token counts, milliseconds, token rates and
  final-logit hashes. Compare medians of non-warmup repetitions and preserve all
  individual samples. Match token counts before claiming a ratio.
- `logits`: absolute binary path, SHA-256, row count, vocabulary columns and
  `little-endian-float32` format. The first row is prefill; subsequent rows are
  one forced token each. Compare complete matrices, argmax and relative errors.
- `repeated_final_logits_exact`: checks every measured final state and the
  separate capture replay. A failure returns a nonzero exit status. Warmup hashes
  remain in `runs` and `warmup_final_logit_sha256`, with
  `warmup_final_logits_match_measured` reported separately. Compare corresponding
  warmup hashes across A/B arms as well. This does not claim all-lifecycle bitwise
  equality: the existing Qwen3.6 direct CUDA path can produce a different final
  hash during its first graph-capture execution, in both baseline and optimized
  arms. Its graph cache captures on the second use and persists across resets.
  Two warmups allow subsequent measured repeats and capture replay to use the
  stable execution state; the warmup discrepancy is retained rather than hidden.
- `quality`: rendered-input token IDs, generated IDs, full selected answer,
  EOS completion and independent answer checks. Quality failures return nonzero.
- `memory`: process working-set/private-byte snapshots and peak working set.
  Direct CUDA additionally reports driver free/total/used dedicated memory and
  existing allocator pool counters. Device memory is affected by the desktop,
  WDDM and other processes; these are not exact per-process peak live allocations.
- `moe_scratch`: whole-model CUDA engines expose an existing internal scratch
  object through diagnostic reflection. Its actual unique tensor-storage byte
  lengths and alias groups are recorded without allocating extra device memory.
  Generic Qwen3.5/3.6 CUDA models use a different path and have an empty list here.
- `managed_assemblies_sha256`, `ptx_sha256`, `native_sha256`: actual build identity.
  Native hashes cover loaded `GgmlOps` modules. The invoking runner must record
  repository/upstream revisions, clean upstream status, device and compiler.
- `model_files`: ordered path, length and timestamp for each already-open
  `GgufFile.FilePaths` entry. `model_file_count` must match the declared split
  count. `model_file_identity_incomplete` is true if actual loaded paths are
  unavailable or incomplete; then `model_total_file_bytes` is null, so a partial
  list cannot be mistaken for the complete checkpoint size. The existing
  `model_bytes` and `model_last_write_utc` refer only to the selected input file.
  In particular, a metadata-only first shard can be much smaller than its model.
  Paths, sizes and timestamps identify files without hashing weight payloads.
  Generated fixtures additionally have a full payload SHA-256; their rewritten
  paths and timestamps may differ between processes.

Keep reports, binaries, logs and captured logits in ignored `artifacts/` or
`docs/validation/`. This probe does not start a server or measure HTTP throughput.

Reports built with `quality_semantic_policy=direct-answer-only-v2` require the
extraction answer to be exactly the ordered comma-separated `cedar,maple` list
(spacing and case may vary). JSON arrays, Markdown fences, extra prose and
alternative separators fail. Older binaries only collected alphabetic words
and could accept those wrappers; do not silently relabel their saved verdicts.
Arithmetic and square-list rules remain unchanged. The Python
`fixed-task-semantic.py` checker independently binds all three known questions
to fixed expected answers; it never takes another model's answer as ground
truth. `rescore-maya-report.py --reports <report.json> ... --output
artifacts/<fresh-directory>` writes separate offline Maya rechecks with full
original-report and checker hashes. It does not run a model, re-decode tokens,
or overwrite the original report/runner policy.

Compare equal numbers of fresh reference and candidate processes with the
same binary and checkpoint. Set `TENSORSHARP_CUDA_MOE_FUSION=0` for the direct
CUDA reference and `1` for the candidate. For the ggml Qwen4Exp prefill combine,
use `TS_Q4E_PREFILL_COMBINE=0` and `1`. Alternate process order between pairs;
finish compilation and unrelated tests before recording timing comparisons.
The runner may set both feature switches to the arm's value; only the relevant
backend's switch changes execution. All other captured environment controls
must remain equal, as must measured and warmup iteration counts.

```powershell
python eng/validation/compare-strata-perf.py --before docs/validation/before1.json docs/validation/before2.json --after docs/validation/after1.json docs/validation/after2.json --output docs/validation/comparison.json
python -m unittest discover -s eng/validation/tests -p test_compare_strata_perf.py
```

The comparator verifies capture lengths and checksums, workload and loaded
binary identities (including source GGUF dimensions), explicit switch values
and effective CUDA policy, exact full logit matrices, repeated
final logits and complete semantic token streams. It compares the median of
each process's measured medians and fails latency regressions over 5% by default.
When shard identity fields are present, it rejects incomplete or mismatched
identities, including changes to later shards. Legacy reports lacking these
fields remain supported and explicitly report that separate shard provenance is
required for split checkpoints; do not mix old and new identity schemas in one
comparison. Existing run binaries should remain fixed within an A/B comparison.
A passing comparison alone does not establish a speedup: inspect its signed
changes and individual process medians, and report near-zero changes as noise.
