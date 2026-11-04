# NInfer fail-fast API regression smoke

Run this small smoke before another full evaluation. It submits at most five
historical arithmetic requests, in a fixed order, and stops at the first lost
selected truth or incomplete response. It is not a new benchmark, quality score,
performance measurement or engine ranking. Passing still requires the original
full quality/no-loss suites and end-to-end evaluation.

The committed fixture is
`InferenceWeb.Tests/TestData/Ninfer/ninfer-api-regression-smoke-r8.json`.
It was extracted from the actual R8 records using the pinned
`ninfer-api-no-loss-audit.audit_text`, after replaying the complete frozen45-case
matrix. It retains complete original messages/settings, historical response
content and hashes, selected old truths, and raw/manifest/evaluation/scorer pins.
CPU tests regenerate the requests with the unchanged original task generator.

The selected fields are result513, result1470, the sorted list for repeat2/task6,
result4649 and result2440. Four historical whole-task checks already failed on
other fields. The smoke preserves those facts and checks only the selected old
truths. It does not turn those historical tasks into complete passes. JSON is
parsed with the original content-parser behavior and type-sensitive field
comparison; no Markdown stripping, output repair or cap adjustment is applied.

PLAN sends no HTTP and does not load a model or read an unavailable manifest:

```sh
python3 eng/validation/ninfer-api-regression-smoke.py \
  --url http://127.0.0.1:5088 --model ACTUAL_MODEL_ID \
  --manifest /absolute/server-manifest.json \
  --out /fresh/artifacts/regression-smoke
```

Add `--execute` to submit requests. Execution requires the existing
`ninfer-target-server/1` manifest to declare `TensorSharp/native-cuda`, one
`--backend cuda`, an original `.ninfer` artifact, and the actual model/port.
The loopback endpoint is mandatory; redirects are not followed. The operator
still owns the exclusive device lease and proof of the actual live process,
runtime, sources and artifact. A manifest declaration alone is not live-server
attestation. Its bytes are checked before and after every request.

Every original request setting and prompt is retained; only the explicit
deployed model ID may differ. The five requests remain streamed, thinking OFF,
temperature0, top-p1, seed1234, repeat penalty1, output cap512, and no discovered
skills/delegation. There are no retries or warmups. The response must be HTTP200,
reach complete HTTP EOF and SSE DONE, finish with `stop`, contain one choice and
no tool calls or reasoning. Responses are bounded to2MiB and the default request
timeout is180seconds (`--timeout` can select up to1800).

The original session nonces are retained. Run timed full-suite comparisons in a
fresh server lifetime after a passing smoke so these identical requests do not
precondition its prompt/cache state. Keep the full suite's actual cache-usage
records; the smoke itself provides no timing result.

The fresh ignored output records exact request JSON, raw SSE, manifest snapshot,
transport checks and each selected field result. Unissued requests remain
`NOT_RUN_FAIL_FAST`. `result.json` is progress; write-once `final-result.json`
and its SHA sidecar preserve the terminal result, including rejection.
Exit0 is PLAN or all five smoke checks passing; exit1 is fail-fast rejection;
exit2 is admission/incomplete-execution failure. Both `qualityQualification` and
`performance` remain `NOT_RUN`, including after a successful smoke.

```sh
python3 -m unittest eng/validation/tests/test_ninfer_api_regression_smoke.py -v
```

Tests use original fixtures/generation and simulated transport responses. They
send no live HTTP, launch no VM/server and perform no GPU/model inference.
