# Offline serving metadata joins

This tool reads completed, preserved evidence. It launches no server, CUDA code
or network request and never changes the frozen replay, original 7 quality + 3
measurement checks, original verdicts, or typed answers.

```sh
python3 eng/validation/qwen38-serving-trace-analyze-v2.py \
  --campaign docs/validation/example/control/campaign.json \
  --profiles-json artifacts/serving-trace-profiles.json \
  --out docs/validation/example/serving-trace-analysis.json
```

The output must be fresh, nonaliased, and under the repository's ignored
`artifacts/` or `docs/validation/`. There is no implicit path search, missing-file
substitution, remote-path rewriting, log repair, timestamp-based matching, or
fallback to a different request or profile.

The operator supplies explicit local copies and original producer paths:

```json
{
  "profiles": [
    {
      "name": "A1-text",
      "campaign_attempt": "ScalarA1-text",
      "replay": "/absolute/local/original-replay.jsonl",
      "structured_logs": [
        {"path": "/absolute/local/structured.jsonl", "producer_path": "/original/control/ScalarA1-text/logs/tensorsharp-server-20261103.jsonl"}
      ],
      "stderr_logs": [
        {"path": "/absolute/local/server.log", "producer_path": "/original/control/ScalarA1-text/server/logs/actual-launch.log"}
      ]
    },
    {
      "name": "A2-text",
      "campaign_attempt": "ScalarA2-text",
      "replay": "/absolute/local/second-original-replay.jsonl",
      "structured_logs": [],
      "stderr_logs": []
    }
  ],
  "comparisons": [{"left": "A1-text", "right": "A2-text"}]
}
```

Empty log lists are allowed only to describe missing evidence: they produce
unavailable joins, never a successful no-difference conclusion. Supply all
actual structured/stderr files in producer order. The original replay SHA must
match the campaign attempt's `audit.sha256`, and replay label/launch SHA must
match the same attempt. Structured logs must originate in its captured
`TENSORSHARP_LOG_DIR`, and stderr must be its actual launch manifest's `log`.
All local input bytes/sizes/SHA are retained and checked stable while analyzing.
Runtime/model/source identities are recorded from the campaign/launch manifest;
this reader does not recertify a build, loaded module map or model bytes.

Join chain:

1. Original response `X-TensorSharp-Request-Id` header, exactly one value.
2. Structured template `q35.serving.request engine_sequence_id={EngineSequenceId}`.
   The original logger can emit duplicate scope `RequestId` keys (ASP.NET long
   ID and middleware short ID); all pairs are preserved. Exactly one candidate
   must equal a known original response-header ID. It never uses last-key wins.
3. Trace `sequence_slot_binding` associates engine/slot/generation with that
   exact chat sequence. Truncated, absent or ambiguous IDs remain unavailable.
4. Actual `run_groups` snapshots record total/padded/real rows, group kind,
   rows-per-sequence/window, positions/version and actual host branch.
5. Slot-specific `verify_commit_completed` records the original window/kept and
   positions, tied to its recorded verify-forward ID. This does not expose
   proposed tokens, recurrent state contents or complete acceptance logic.
6. `head_host_return_completed` is associated only to the recorded forward.
   Missing head/commit records are listed for every observed verify forward;
   intermediate prefill chunks legitimately need not have their own head.
   Multi-request mixed forwards do not identify the exact logit rows copied for
   an individual request. That ownership is explicitly unproven.

Strict trace JSON/schema/index checks reject interleaved lines, duplicate keys,
unknown events, missing indices, out-of-order or duplicated events, dropped-event
counts and corrupt fields. No joining/reassembly of broken lines is attempted.
Unmapped startup/internal records remain listed. Legacy stderr has no terminal
trace flush marker. Even a schema2 closed stream and perfect minimum joins never
prove complete GPU/kernel/state capture.
TraceON changes host timing/batching; its data is diagnostic, never performance.

Directed comparisons retain every original True-to-False quality/measurement
check and independently diagnose each expected typed field; positive changes
never cancel regressions. Input equality distinguishes independently recomputed
canonical full bodies from reported common/wire hashes. Original response wire
bytes are base64/size/SHA checked, but this tool does not independently reparse
SSE semantics or rebuild original request wire ordering. Geometry/position
comparisons are null when either request lacks required metadata. Same captured
binary/model/config/GPU identities are separately reported; missing inventories
do not certify A/A. No shape/window difference establishes a cause for an OCR
failure, and equal observed metadata does not establish equal GPU states/logits.

CPU fixtures test schema, mappings and negative controls only. They are not new
model, GPU, CUA-quality or performance evidence. The frozen schema1 reader
found interleaved stderr in actual S5 A/A and correctly refused full joins; this
v2 reader adds a dedicated-file path without repairing that old evidence.

## Schema2 dedicated files

This v2 tool preserves the frozen schema1 analyzer and original scoring. For a
run with `TS_Q35_SERVING_TRACE_FILE`, add exactly one explicit `trace_logs` entry
per profile, for example:

```json
"trace_logs": [{"path": "/local/archived/trace.jsonl", "producer_path": "/remote/owned/control/trace.jsonl"}]
```

The producer path must exactly equal both the actual campaign attempt environment
and launch manifest environment `TS_Q35_SERVING_TRACE_FILE`, with exact trace
opt-in `1` in both. Local archived paths can differ, but must be supplied and
hashed. An absent/mismatched explicit input is rejected; the analyzer never
substitutes stderr for a configured dedicated file. `stderr_logs` may still be
included as source-bound diagnostic inputs and are not parsed as trace events
when a dedicated file is configured.

Dedicated input is plain schema2 JSONL. The file must contain one engine, an
opening marker at index 1, continuous indices, and one terminal marker as the
last record. The opening marker binds the configured producer path. Terminal
previous-event index, forward sequence, last successful-forward value, no active
forward, normal reason and zero dropped writes must agree. Constructor failure,
missing/duplicate markers, interleaving, corrupt JSON/UTF-8, index gaps, dropped
records or inconsistent terminal counters make joins unavailable. No line
repair, prefix recovery, missing mappings or terminal events are synthesized.

`DEDICATED_HOST_METADATA_STREAM_CLOSED_ONLY` is limited to the observed host
metadata file. It is not complete GPU/kernel/state capture, exact head-row
ownership, checked cache contents, model quality, causality or performance.
Legacy schema1 stderr remains explicitly uncertified because it has no closing
marker. Each original quality/measurement result and every true-to-false change
is retained even when trace joins are unavailable. The per-engine private trace
file path is recorded but excluded from math/configuration flag equality; no
other policy flags are ignored by that addition.

A close reason `engine_dispose_finally` identifies where the trace close ran; it
does not assert successful CUDA cleanup or device completion. Stream markers
only bound the observed metadata transport interval.
