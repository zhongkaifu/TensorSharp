# Qwen public unified prefill admission probe

This tool uses the actual `UnifiedMemoryModelSession.Create(options)` and
`InferenceEngine` path with explicit RAM, device and expert ceilings. It never
injects a fake reservation or substitutes a native executor. No Server.Host
environment variable is used as a substitute for `MaximumHostBytes`.

Build normally, or compile only the tool against an explicitly selected qualified
managed closure (no project/native rebuild):

```powershell
dotnet build eng/validation/QwenUnifiedPrefillProbe/QwenUnifiedPrefillProbe.csproj -c Release -m:1 `
  -p:ValidationRuntimeDirectory=D:/qualified-managed-closure `
  -p:BaseIntermediateOutputPath=E:/fresh-probe/obj/ `
  -p:MSBuildProjectExtensionsPath=E:/fresh-probe/obj/ `
  -p:OutDir=D:/fresh-probe/bin/ -p:UseSharedCompilation=false
dotnet D:/fresh-probe/bin/QwenUnifiedPrefillProbe.dll --self-test
```

The self-test traverses **every public instance property** of the current
`RequestMemoryAdmission` and `SchedulerConfig`, so a new omitted property fails
instead of silently changing production admission. It also checks the actual
internal `eos` completion vocabulary, full terminal-EOS/publication/count
boundaries and both explicit answer policies. Its tiny pure policy fixtures do
not load any model, native library or GPU. The actual run always uses the real
session budget and original delegates. Native/PTX deployment is explicit; copy a
verified deployment closure and inject the exact qualified native binary before
running. Keep original managed/native/source hashes and deployment receipt.

Prepare and freeze input IDs once from the checkpoint's original metadata,
tokenizer and chat template. This mode opens metadata only; it does not load
weights, generate an answer, initialize an engine or run model work. Supply a
fixed task-neutral system text long enough that both prompts exceed the requested
prefill cap. Inspect the resulting counts and preserve its identity before any
GPU run:

```powershell
dotnet D:/fresh-probe/bin/QwenUnifiedPrefillProbe.dll `
  --prepare-inputs true --model C:/Works/models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf `
  --system-file D:/fixtures/fixed-context.txt --context 4096 `
  --output D:/fresh-input-preparation
```

`frozen-input.json` contains exact original messages and raw IDs for:

- Math: `What is 17 + 25? Reply with only the integer.` Independent oracle: exact
  production-parsed `Content == "42"`, no tool calls, natural EOS, no trimming.
- Occupier: `List the integers from 1 through 1000 in ascending order, separated
  by spaces. Output only the numbers.` Its cancellation is intentional; the tool
  does not claim to certify the complete counting answer.

Run separate fresh processes with the **same frozen input, checkpoint, native,
context and options**; only the stated Host budget/expectation changes:

```powershell
dotnet D:/fresh-probe/bin/QwenUnifiedPrefillProbe.dll `
  --model C:/Works/models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf `
  --mmproj C:/Works/models/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf `
  --model-identity D:/fixtures/qwen-original-full-sha-manifest.json `
  --native-sha256 QUALIFIED_NATIVE_SHA256 `
  --input D:/fresh-input-preparation/frozen-input.json --mode cancel-cohort `
  --context 4096 --prefill 512 --max-new 128 --occupier-max-new 256 `
  --host-bytes 12884901888 --device-bytes 15032385536 --expert-bytes 7340032000 `
  --cpu-threads 8 --answer-policy exact --expect-downgrade false --deadline-seconds 1200 `
  --output D:/fresh-normal-cohort
```

For the pressure arm use a separately documented **measured** Host ceiling and
`--expect-downgrade true`. There is deliberately no invented universally fitting
low number: the common live owners, available RAM/commit, context and checkpoint
geometry determine it. A configured single-request wide no-hit peak above the
ceiling can legitimately fail at `SubmitRequest`; this never counts as a
downgrade pass. The pressure arm must still admit the real first request, then
exercise shared ownership with a second request. `single` mode is also available
for a fresh natural-EOS reference, not for manufacturing sibling pressure.

`--answer-policy exact` is the default: unmodified production-parsed Content must
equal `42`. Explicit `--answer-policy json-integer` instead parses the complete
original Content as a JSON value and requires its sole number token to be the
integer literal `42`. Only JSON whitespace outside that token is accepted;
explanations, trailing garbage, quoted strings, decimal/exponent notation,
full-width digits and other integers are rejected. The tool never trims or
rewrites output. Every report keeps raw tokens/text/Content, `StrictExactContent42`,
the independent `JsonInteger42Content` classification and
`StrictFormatNonconformance`; `SemanticPassed` identifies the explicitly selected
policy. A JSON-integer pass with strict formatting failure must be reported as
such, and cannot retroactively change an older exact-policy failure.

For an actual pressure cohort, choose the occupier limit before running so it
can survive the primary's possible small prefill chunks. For the prepared
1,539-token occupier and context 4,096, `--occupier-max-new 2048` remains within
context; 256 can finish before the primary decodes after a large downgrade.
Only an observed successful joint decode and cancellation qualify the cohort.
Changing that limit or the answer policy requires a new command and fresh
evidence. Compare complete raw primary output arrays with a separate reference;
semantic classification alone is not greedy parity.

The occupier submits first; the math request submits after its first publication.
The real worker's first successful math decode scope queues the occupier's public
`engine.Abort`. Acceptance requires a successful joint execution, occupier abort,
subsequent actual math advancement and complete math EOS. No consumer delay or
paused compute gate substitutes for model progress.

The transparent observer records production `PrefillTokenLimit(sequence)` before
and after real scope disposal under the existing worker/compute lock. With
prefix off, no replay and plain AR, each positive serial prefill scope contains
one public `Forward` (the inspected BatchExecutor paths). All successful deltas
must respect the selected cap and sum exactly to each prompt length. It records
zero-delta holder preparation separately. Internal kernel tile widths are not
inferred from this observation.

Compare complete primary `OutputTokens` across normal/pressure/single reports for
exact greedy parity; semantic correctness is independently checked in every run.
Reports retain raw input/output, completion reason, timing attribution, selected
caps, actual scope deltas, budget high-water and all cleanup stages. Final shared
ledger, request handles and quarantined releases must be zero. Failed creation,
early admission refusal and cleanup errors remain failures in `report.json`.

`--async-prefill true|false` explicitly maps the public session option
`EnableAsyncExpertPrefill`; its default remains `false`. It does not enable an
environment override or change production defaults. Both settings capture public
`GetHostMoeFilePipelineUsage()` before session creation, after the engine disposal
attempt, and after successful session disposal. Unavailable or failed reads remain
explicit and fail telemetry qualification; they are never replaced with zero.
Cleanup still runs when telemetry fails, and failed engine disposal prevents a
purported quiescent final read and session disposal beneath a live worker.

An enabled run must actually increase **both** `Calls` and `SourceBytes`, alongside
all existing raw-output, EOS, cap, cancellation, identity and zero-ledger gates.
An unused enabled path is a failed coverage gate. A disabled run requires both
deltas zero. Final counters must equal the post-engine values and the retained
ring payload must be zero. These are process-cumulative successful execution and
logical-source-payload counters: they prove path use, not physical SSD reads,
measured transfer/compute overlap or a performance improvement. The current
session scope requires at least 64 rows, at least 32 MiB eligible file payload and
single-device CUDA; its two charged pinned ring slots total at most 16 MiB.
Smaller admitted prefill widths or other fallback paths may correctly do no
pipeline work and therefore cannot pass an explicitly enabled coverage run.

Use an external **owned-process** watchdog for a hard timeout: synchronous native
work cannot be safely interrupted by a managed token, and engine disposal can
wait. Do not run with another model, storage migration, compiler or whole-file
hash workload. Scope observation is a correctness configuration; timings are
observations with tool overhead, not a zero-instrumentation performance claim.
