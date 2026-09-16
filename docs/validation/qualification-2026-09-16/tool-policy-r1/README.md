# Tool policy input validation — r1

Malformed `tool_choice` values and non-boolean `parallel_tool_calls` now return HTTP 400 before queue admission or SSE output, across model families. Existing required/named client declaration checks remain intact. The new adapter cases cover strings, numbers, null, malformed named objects, and invalid parallel policy even with `tool_choice=none`.

**172 passed, 0 failed, 0 skipped** in the focused adapter and DeepSeek grammar regression suite; see [TRX](tool-policy-validation-r1.trx) and [source pins](source-pins.json). This is not full-model generation enforcement: Qwen required/named choices still need a constraint implementation, and successful requests in an HTTP campaign do not establish that guarantee.

The test invocation also rebuilt the local native library because its supplied skip property did not match the project property. It completed successfully; upstream ggml remained clean at `456172ec733a135778adcd32d00e576a58232e45`. No native numerical qualification is inferred from this managed test run.
