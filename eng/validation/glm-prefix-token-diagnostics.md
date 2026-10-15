# GLM prefix token diagnostics

`extract-glm-prefix-token-diagnostic.py` reads a completed owned
`prefix-warmup-host.py` / `prefix-warmup-http.py` run with
`validation_token_trace: true`. It does not start a server or run a model.

The capture must include these four completed text requests:
`ff7-first`, `ff7-repeat`, `ff7-repeat-client-followup`, and
`ff7-first-client-followup`. The requests must use explicit nonthinking greedy
sampling and finish at their declared token limit. Opening wire requests must
match, and follow-up wire requests must match, apart from session IDs.

Run from the repository root after the owned Host has exited and all cleanup
and launch bookends are complete. Relative and absolute CLI paths are supported:

```powershell
python -B eng/validation/extract-glm-prefix-token-diagnostic.py `
  --launch-audit artifacts/example/launch-audit.json `
  --http-report artifacts/example/http/report.json `
  --server-log artifacts/example/http/server-log.jsonl `
  --expected-sdk-version 26 `
  --expected-native-sha256 <actual-production-DLL-SHA256-from-qualified-receipt> `
  --output artifacts/example/raw-token-diagnostic.json
```

Use the HTTP helper's exact `http/server-log.jsonl` snapshot. Its SHA and byte
count must match the original source-log pin in the HTTP report. The live
`server-logs/` file can grow during shutdown and must not replace this snapshot.
Rotated captures use `tensorsharp.rotated-log-capture.v1`: at most 32 canonical
files, contiguous numeric rotation indices, and at most two adjacent UTC dates
in the same directory. The tool checks the ordered source byte lengths and
each source SHA against its exact range in the immutable combined snapshot,
as well as the combined pin. It neither inserts separators nor reopens live
source files. Unknown schemas, gaps, duplicate paths, foreign directories,
incomplete prior lines, and changed payloads are refused.
The tool reads bounded receipts, logs and producer source pins, and checks them
again before writing. The referenced producer sources must still match their
original pins. It uses the original launcher's immutable DLL and model
bookends; it does not freshly hash DLL/GGUF payloads. The output must be a new
file in an ignored validation directory.

Each request is linked by its actual HTTP scope ID to exactly one input,
timing and output event, with matching internal generation and session IDs.
The output contains complete prompt/output token arrays, SHA256 over signed
32-bit little-endian IDs, and the first differing token for both opening and
follow-up comparisons. It also records the exact observed autoregressive
commit convention, `P+O` or `P+O−1`, and compares the follow-up's cached prefix
against the opening's actually consumed prompt/output history.

A completed original visible-output failure can be diagnosed. Its failed
gate and errors remain in the output; extraction does not turn them into a
pass. Input-ID equality rules out a history-ID mismatch for those captures.
It does not prove hidden KV tensor equality, identify the cause of different
outputs, or qualify model quality, cross-engine consistency, performance or
shared-budget default readiness. Tracing itself is diagnostic and is not a
qualified performance measurement.

Run the model-free regression tests with:

```powershell
python -B -m unittest discover -s eng/validation/tests -p test_extract_glm_prefix_token_diagnostic.py -v
```
