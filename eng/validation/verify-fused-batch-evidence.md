# Native joint-batch engagement evidence

Start the qualification server with `TS_BATCHED_FUSED_TRACE=1`, then run
`unified-qwen-concurrency-bench.py` with two distinct Web UI sessions. After both
requests and server log writes finish, verify the report and actual JSONL log:

```powershell
python eng/validation/verify-fused-batch-evidence.py `
  --concurrency-report artifacts/my-run/http/report.json `
  --server-log artifacts/my-run/server.jsonl `
  --output artifacts/my-run/fused-evidence.json
```

The output must be a new file. Inputs and checker are SHA-256 identified.
This tool performs no model execution or process cleanup.

The concurrency runner can also invoke this checker directly using
`--server-log <actual-jsonl> --require-fused-graph`. It preserves a bounded log
copy and the checker hash. Without `--require-fused-graph`, engagement evidence
is diagnostic only. The source report's output-parity result is retained
separately: a real joint graph can execute while quality/parity still fails.

`generation.sequence.bound` maps the HTTP scope `RequestId` and response session
to the engine's independent internal sequence ID. Mapping does not establish
admission or execution. `batch.fused.accepted` is emitted only after native
success and publication of every participating managed sequence. Its native
execution snapshot must be read immediately after that successful call.

Startup prefix warm-up emits an internal binding without an HTTP owner. Such
unrelated mappings are counted and ignored; a missing or conflicting binding
for either current session/internal ID still fails. The production logger emits
the native `UInt64` execution ID as a decimal JSON string, so the checker accepts
canonical positive UInt64 strings or exact JSON integers and compares their
numeric values. Booleans, floats, signs, whitespace, leading zeros and overflow
are rejected. A checked-in six-line actual server excerpt exercises both logger
details; the immutable original failed report remains evidence of the old gate.

The gate requires two or more successful steps of the **same pair of current
HTTP generations**, distinct sequence IDs, native execution IDs that increase,
one-token position advancement without an intervening untraced/serial gap, one
logical graph, actual positive node count, and completed managed publication.
Missing trace, plan logs, one-time success warnings, SSE alternation, old warm-up
requests, duplicate records, and native success before publication do not pass.

`actual_joint_engagement_verified` separately records whether a valid repeated
joint interval exists. `full_interval_continuity_verified` requires no untraced
position advancement between the first and last accepted joint steps. A forward
gap splits the diagnostic `contiguous_joint_runs` and is retained once in
`continuity_gaps`; it cannot turn a real accepted graph into a plan-only event.
The strict final `passed` and `batched_fused_graph_verified` **still require full
interval continuity**. Backward/replayed positions or malformed identities
invalidate engagement itself. The checker does not infer the cause of a gap.

`NativeHostSegmentCount` records host MoE seams separately. A jointly built
logical graph can execute through multiple CUDA/CPU segments; this evidence does
not claim one physical CUDA launch or simultaneous GPU kernels. It proves the
qualified joint interval, not that every token from arrival to EOS was batched.

The original concurrent report remains immutable. Its natural-EOS/quality and
greedy serial-parity results remain separate: bounded FF7/Hawking introductions
can hit their output limit and must not be called complete answers. Trace is for
engagement qualification; disable it for separately identified quiet performance
runs. Process/native/model identity and teardown remain the outer runner's
responsibility.

When present, `generation.timings` is independently matched by HTTP `RequestId`
and response session. `generation_timings` retains per-generation attributed
prefill/decode durations, inclusive server-generation elapsed time, reused input
tokens, pipeline-recorded tokens and engine output tokens as distinct fields.
Shared batch compute time is apportioned between sequences. It is not request
wall time; preparation, queueing and tool execution are not separately measured.
Child-session rows are counted separately and never added to the root's time.
Missing completion metrics stay `null`; absence or invalid timing cannot
fabricate zero-cost work and does not change the independent native graph gate.
This checker does not derive throughput from timing rows or SSE frame counts.
Every observed generation under a root HTTP request must have measured engine
completion for `all_current_requests_measured`: a later complete tool round does
not replace an earlier `null`. The concurrency runner exposes the same timing,
actual-joint, continuity, step-count and gap fields in its summary, while its
required-fused final gate continues to use strict interval continuity.
