# Native Qwen serving metadata trace

`TS_Q35_SERVING_TRACE=1` enables diagnostic host metadata in a fresh native Qwen
engine. Only the exact string `1` enables it, once at construction. Leave it
unset for ordinary serving and performance measurements. Enabling it creates
JSON snapshots and writes diagnostic records, which can change CPU timing and batching;
trace runs are separate diagnostic configurations, never performance samples.

The HTTP response continues to use its existing `X-TensorSharp-Request-Id`.
The existing HTTP logging scope's `RequestId` can be joined to the additional
`q35.serving.request` log field `EngineSequenceId`. That generated `chat-…`
identifier appears as `engine_sequence_id` in native trace records. No new HTTP
request, response or model API payload field is introduced.

For a fresh engine, prefer `TS_Q35_SERVING_TRACE_FILE=/absolute/fresh/path.jsonl`
with trace enabled. The parent must already exist and every path component must
be physical (no links); the path must be normalized and not already exist,
including dangling links. The launcher must additionally keep it inside its own
fresh diagnostic output and outside models/source/runtime. One engine owns one
UTF-8, AutoFlush writer opened with `CreateNew`. A second engine cannot reuse it.
An explicit file configuration that cannot be created refuses diagnostic engine
construction; it never silently falls back. With trace off the file environment
variable is not read and no file is touched.

Dedicated files contain plain JSONL schema `tensorsharp-q35-serving-trace/2`.
`trace_opened` is the first event, and idempotent close emits `trace_closed`
with the previous event index, last forward ID, last successful forward ID,
active forward (normally null), reason and observed dropped-event count. It is
written at the end of the existing engine disposal, including its failure path;
a later constructor failure closes only the new trace writer and does not add
CUDA cleanup or synchronization. `constructor_failed` is an incomplete run.
The terminal record is AutoFlushed before the owned writer is disposed. A close
reason describes the trace hook, not proof that CUDA cleanup succeeded. Missing
markers, event gaps, dropped writes, an active forward or abnormal close must
never be treated as a closed metadata stream. Even a clean file is not proof of
complete GPU/kernel/state capture. Runtime write errors remain best effort;
a failed terminal write or writer disposal can only make the evidence less
complete. Console is never disposed by tracing.

If the file option is absent, legacy stderr records start with `[q35-serving-trace] ` followed by one JSON object,
schema `tensorsharp-q35-serving-trace/1`. Each has an engine ID, event index,
timestamp, count of previously dropped diagnostic writes, and metadata:

- `sequence_slot_binding` assigns an opaque slot ID and binding generation.
  Rebinding the same slot to another sequence or the primary increments its
  generation; unowned primary slots are explicitly unmapped. IDs longer than
  128 characters are truncated with a flag. No pointer addresses are logged.
- `run_groups` records the actual grouped forward's padded and real row counts,
  sequence order, prefill/decode/verify kind, rows per sequence (the actual verify
  window), NPast, M-RoPE delta and block-table version before and after. Its
  projection branch is assigned only when the host enters the corresponding
  native layer-loop implementation, including Q81 override branches. A CUDA
  graph replay that skips this host branch says `NOT_OBSERVED_ON_HOST` and
  records the existing graph replay-counter delta. It does not guess a route
  from the row count. `host_submission_returned` is not device completion.
- `verify_commit_completed` records the existing window, kept counts, positions
  and per-slot verify-forward IDs after the original commit synchronization
  returns. A zero verify-forward ID means unavailable, not an inferred match.
- `head_host_return_completed` records requested copy byte counts only, after
  the original head synchronization and host copies return. It does not read,
  hash or serialize logits or sampled tokens.

The disabled path has no trace ledger, writer access, snapshots, diagnostic
counters, arrays, serialization or hashes. Enabled tracing adds no CUDA call,
event query, copy, poll or synchronization. Writer failures are counted and
suppressed so a failed diagnostic write does not fail the model operation;
missing records or dropped events cannot be treated as complete evidence.

This first stage observes ownership, grouped geometry, host branch selection,
commit and existing head-copy completion. It does **not** observe draft split
counts, all per-layer dispatches, cache payload clone/donate origins, host-tier
copy tickets, attention/GDN state contents, acceptance proposals, or GPU
hardware counters. `binding_generation` is a diagnostic ownership counter, not a
claim that cache values were checked. No prompt text, token arrays, image
contents, logits or raw cache payload keys are logged. Hash-bound manifests
must independently record source, runtime, model and configuration identities.

CPU tests in `Qwen35ServingTraceTests` cover disabled allocation behavior,
fresh file lifecycle/AutoFlush/terminal indices, refusal of existing or aliased
paths, file failure and constructor cleanup before CUDA, explicit ownership/rebinding, ragged mixed/verify metadata, absent graph-branch
observations, commit/copy linkage, writer failure, bounded IDs and the same
real `RunGroups` row-cap refusal before any CUDA call. Synthetic CPU records
are schema and control-flow tests, not proof of GPU execution or model quality.
Fresh Linux native and serving gates are required for every new build; prior
S4 numeric/quality evidence does not certify this instrumented binary. A/A
variation remains an observation, not attribution to a particular kernel,
atomic reduction or CPU sampler.
