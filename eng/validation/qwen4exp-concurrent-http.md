# Qwen4Exp concurrent HTTP validation

Run `qwen4exp-concurrent-http.py` against an already running TensorSharp server.
It discovers the served model ID through `/v1/models` and sends streaming OpenAI
chat requests with explicit greedy sampling, seed, and neutral penalties. The
tool does not start a server or change its configuration. Every report must stay
in ignored `artifacts/` or `docs/validation/`.

```sh
python3 eng/validation/qwen4exp-concurrent-http.py \
  --url http://127.0.0.1:5001 \
  --groups topics,markers --modes serial,parallel,staggered \
  --max-tokens 384 --repeats 2 --cancellation \
  --server-log artifacts/validation/qwen38-iq2-server.log --require-fused \
  --provenance artifacts/validation/qwen38-iq2-identity.json \
  --output artifacts/validation/qwen38-iq2-concurrency.json
```

The topic pair uses the user's Chinese Final Fantasy VII and A Brief History of
Time prompts. Each answer must contain its own subject and relevant anchors,
enough substantive text, and none of the other topic's distinctive terms. The
marker pair combines a short arithmetic answer with a longer exact transcription
whose numbered lines avoid intentionally triggering the runtime's repetition
stopper. Checks reject wrong markers, arithmetic, missing lines, extra prose, or
another request's marker. Reports retain all answers for human inspection. These
checks screen relevance, instruction following, and obvious request mixups;
they do not certify every factual statement or unrestricted model quality.

By default, each selected group runs serially first, then simultaneously, then with a second
request admitted after the first model delta. The staggered check requires the
first request still to be active when the second arrives. The cancellation check
disconnects an active long stream, finishes an independent survivor, and sends a
fresh arithmetic task before waiting for the survivor. It records whether that
new admission overlapped the survivor and requires that overlap for a pass.
Cancelled/survivor spans must also overlap. A cancelled stream must disconnect
before `[DONE]` or any finish reason, and the exact marker task must remain
incomplete. HTTP cannot identify the actual native cache slot or prove exactly
when the server reclaimed a cancelled slot.

Client TTFT starts before HTTP upload and ends at the first nonempty answer or
reasoning delta; empty role headers do not count. Complete requests require HTTP
200, valid SSE, a final usage record, a normal stop or length finish, and `[DONE]`.
Group throughput is the sum of completed usage tokens divided by total group
wall time, including admission and prefill. It is an end-to-end throughput
measurement, not a native decode-only rate. The tool reports cached prompt
tokens because serial and concurrent runs on one host can differ in cache warmth.

Serial/concurrent speedups are qualified only when both runs pass their answer
checks, request hashes match, and completion token counts and finish reasons
match. Answer hashes are compared separately. `--require-exact-parity` fails any
serial/concurrent text difference. `--min-parallel-speedup 1.05` optionally gates
qualified speedups; missing comparable measurements fail that gate.

`--concurrent-only` omits local serial controls and runs only the selected
concurrent modes. It can screen request correctness without a comparison report;
such a run records the omitted serial control and produces no local speedup or
parity claim. `--min-parallel-speedup` requires a local serial control and is
rejected in this mode. Exact text parity requires `--compare-with` when no local
control runs.

To compare enabled batching against `TS_BATCHED_FUSED_DECODE=0`, restart the same
host with that environment override, run the same workload into a separate
report, then run the enabled host with:

```sh
python3 eng/validation/qwen4exp-concurrent-http.py \
  --url http://127.0.0.1:5001 --groups topics --modes serial,parallel \
  --max-tokens 384 --repeats 2 \
  --compare-with artifacts/validation/qwen38-iq2-round-robin.json \
  --min-baseline-speedup 0.95 \
  --output artifacts/validation/qwen38-iq2-batched.json
```

The baseline regression gate applies to all matched completed benchmark groups,
including the serial group. Intentional cancellation remains a correctness
scenario and has no performance ratio. The tool suppresses speedups for different
requests, model IDs, output token counts, or finish reasons. Review actual model
weight hashes, GGML revision and clean tree status, server build, device, backend,
process environment, and cache warmth in supplied provenance before attributing
timing differences to a change. Model IDs alone do not establish that identity.

Before comparing an external report, the runner recomputes its retained request
and answer hashes, checks its declared workload and unique group/request coverage,
recomputes completed-answer quality, reconstructs answers and completion/token
and model-delta timing metadata from retained SSE events, and checks finite wall
times and internally consistent group token/throughput/overlap metrics. Declared
follow-up requests undergo the same request/hash/SSE/quality/usage checks;
declared server health checks require matching coverage and the served model.
Corrupt or
incomplete stored evidence fails comparison. The exact compared report file path
and SHA-256 are retained even if integrity validation fails. This validates the
report's internal evidence; it does not authenticate its author or bind the model
server to its attached provenance. Historical report files are never rewritten.

To measure a round-robin host efficiently after completing the full enabled
workload, run just its parallel groups against the earlier enabled report:

```sh
python3 eng/validation/qwen4exp-concurrent-http.py \
  --url http://127.0.0.1:5001 --groups topics,markers --modes parallel \
  --concurrent-only --max-tokens 384 --repeats 2 \
  --compare-with artifacts/validation/qwen38-iq2-batched.json \
  --require-exact-parity \
  --output artifacts/validation/qwen38-iq2-round-robin.json
```

This compares the candidate's selected groups with matching groups in a baseline
superset. Extra baseline modes are explicitly listed as omitted comparison
coverage. Every selected group and every request inside it must match; missing or
duplicate coverage fails qualification. Here the earlier enabled report is the
comparison baseline, so `baseline_to_candidate_wall_speedup` is enabled wall time
divided by round-robin wall time. Its reciprocal is the batching speedup. The
runner does not infer which host has batching enabled from a label or model ID.

`--require-fused` needs a server log and accepts only a new runtime message saying
that a multi-sequence fused graph executed during this invocation. Startup plans
and old acceptance lines do not satisfy it. The runtime logs acceptance once per
executor, so run this tool before any other concurrent workload on a fresh host.
The report records the exact log byte range and hash, accepted batch widths, and
fallback warnings. This is process-level engagement evidence and cannot count
every accepted step. A require-fused run fails on a new decline/round-robin warning.

A quick screen uses `--groups topics --modes serial,parallel --max-tokens 192` and
makes four requests. Omitted marker, staggered, cancellation, or vision scenarios
are recorded as omitted coverage. Run a separate vision validation with the model's
compatible projector; this runner covers text only.

For the Qwen3.8 CUDA sparse-attention boundary regression, use long topic
requests and explicitly require actual prompt-plus-completion usage to exceed
the model's QSA top-k boundary. A token cap alone does not establish coverage:
an answer that stops early fails this gate. For a model with QSA top-k 2048:

```sh
python3 eng/validation/qwen4exp-concurrent-http.py \
  --url http://127.0.0.1:5288 --groups topics --modes parallel,staggered \
  --concurrent-only --max-tokens 4096 --min-topic-total-tokens 2050 \
  --repeats 2 --require-generation-overlap --check-server-health --follow-up \
  --progress-every-deltas 128 \
  --output docs/validation/qwen38-cuda-concurrent.json
```

The gate uses 2050 because the final sampled completion token may not itself
enter a forward graph; one extra token establishes that the forwarded sequence
crossed 2048.

`--require-generation-overlap` requires overlap between the first and last
nonempty model deltas of concurrent streams. It is stronger than overlapping
HTTP connection spans, but it does not prove simultaneous kernel execution;
use the native fused-decode log gate for execution-path evidence.
`--check-server-health` queries `/v1/models` after every group and requires the
served model to remain available. `--follow-up` ends with a fresh independent
arithmetic request whose exact answer is checked, so a successful stress run
also establishes subsequent inference usability. The retained report records
these optional checks separately from the benchmark groups.
`--progress-every-deltas` prints live counts of nonempty SSE events and answer
characters. Event counts provide progress; final token usage supplies the
token-boundary coverage check.

Raw topic answers can stop before both sequences enter sparse attention.
`--topic-system-prompt-file` adds retained system context only to the topic
requests, while preserving each exact Chinese user prompt. The reusable neutral
fixture is `InferenceWeb.Tests/Fixtures/Qwen4ExpServing/sparse-qsa-context.txt`.
Use it for supplemental simultaneous sparse coverage, record actual prompt
usage, and retain a separate raw-prompt control. The report labels the variant
and stores the prefix text, source path, and SHA-256; it never treats a contextual
run as the same request as an unmodified raw prompt. Marker and follow-up tasks
omit this prefix.

The supplemental fixture run can use a shorter completion cap:

```sh
python3 eng/validation/qwen4exp-concurrent-http.py \
  --url http://127.0.0.1:5288 --groups topics --modes parallel --concurrent-only \
  --max-tokens 1536 --min-topic-total-tokens 2050 \
  --topic-system-prompt-file InferenceWeb.Tests/Fixtures/Qwen4ExpServing/sparse-qsa-context.txt \
  --require-generation-overlap --check-server-health --follow-up \
  --progress-every-deltas 128 \
  --output docs/validation/qwen38-cuda-context-concurrent.json
```

Local runner tests use a threaded HTTP fixture, without loading a model:

```sh
python3 -X utf8 -m unittest eng/validation/tests/test_qwen4exp_concurrent_http.py
```
