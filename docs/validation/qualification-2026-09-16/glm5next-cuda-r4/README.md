# GLM5Next r4 replay — 2026-09-16 18:22 UTC

Original r4 fixture replay: **22 passed, 0 failed, 0 skipped**, including all three opt-in CUDA rows on physical NVIDIA A40 GPU5 (`GPU-c7df85ec-9f9d-da69-151a-32f871a5426b`). The process-global managed allocator remains CPU; the GLM native executor independently owns CUDA. This corrects the earlier campaign invocation that mixed the global allocator backend. Its original 7/17 pass record remains failed.

The filter `FullyQualifiedName~Glm5Next` omits `NativeGgmlIdentityTests`, explaining 22 versus the previous CPU20 + CUDA3 = 23 selected total. The original run's 19 non-CUDA rows include CPU exception containment and CPU A/B/A holder rollback. They are not GPU tests.

CUDA ngram replay proposed19 / accepted16 drafts, 8 verify windows and3 rollbacks. The deliberately wrong-tail replay proposed23 / accepted13 drafts,10 verify windows and10 rollbacks. Both matched all24 plain greedy output tokens; original full-vocabulary continuation and verify-row checks passed unchanged. The CUDA snapshot/verify/restore/rewind row passed too.

## Exact execution

Copied the existing r4 test output into `/workspace/ts-glm5spec-r4-replay-app`; frozen r4 files were not modified.

```sh
env CUDA_VISIBLE_DEVICES=5 TS_TEST_GGML_BACKEND=cpu TS_TEST_GLM_CUDA=1 TS_TEST_GLM_SNAPSHOT_BOUNDARY=1 /workspace/tensorsharp-no-patch-20260915/dotnet/dotnet vstest /workspace/ts-glm5spec-r4-replay-app/InferenceWeb.Tests.dll --TestCaseFilter:FullyQualifiedName~Glm5Next --Logger:trx --ResultsDirectory:/workspace/ts-glm5spec-r4-replay-results
```

SHA256 identities:

- native `7eaf98656edcd043884f3eb776bed68abbee29f5cb446e237bfa16afa5598af7`
- backend `17c60ff946c345e7f0ea8d3232936b60340b696b0011d6e5037cbced99a4af76`
- models `2ffdbde3afb31a6c63dfc905078fcc6d4cd1394ba24327b906806a4d1961e52f`
- tests `c9ac18deb139f0c5531e2c6dab9904604dac89688049a6d5ee44bce0f8d46128`

The parent r4 build pins source aggregate `f7431257148bd5168c0f80e3049b36fc5228c19a8d47448e09e60ff7eb43eb28` and unchanged upstream ggml `456172ec733a135778adcd32d00e576a58232e45`.

## Production export manifest regression

The r4 integrated portable run remains failed. Its Maui manifest test scans native `TSG_EXPORT` declarations, including a test-hook-only GLM declaration. The working-tree fix marks that declaration `TSG_TEST_EXPORT`, matching the existing QSA convention; no production export or P/Invoke changed. The independent working-tree Maui suite passes15/15 without skips (`maui-exports15.trx`). This is source-scanner evidence and does not claim a new native build.

## Limits

These deterministic tiny KDA checkpoints establish operator/state and speculative-loop behavior, not trained GLM checkpoint acceptance, mixed KDA/MLA state coverage, multi-GPU inference, or performance. GPU5 was shared with another workload, so elapsed fixture time is not a benchmark. No tolerance changed.

## Expanded CUDA fault and holder replay

`expanded-r1` and the final analyzer-compatible `expanded-r2` each pass **26/26 with no skips**. These add two CUDA fault rows (`std::bad_alloc` and a nonstandard C++ exception) and CUDA A/B/A holder continuation parity. Capture failure preserves live state; partial restore failure prevents forward/speculation/rewind until checked reset; a healthy peer holder continues normally; reset recovers identical full-vocabulary logits. A/B/A verifies the bound position, rollback and independent continuation for each holder. Native identity is the26th selected test.

Both runs use the same r4 native and production assemblies, with a separately compiled test-only DLL. Source and loaded-directory binary hashes, exact build/run commands and environment are in each manifest. r1 used overloaded private helper names; the repository xUnit analyzer rejected that source. r2 renames the helpers without changing assertions and repeats all26 rows. Main-tree analyzer validation is handled in the integrated managed lane. Neither r1 nor r2 changes the immutable r4 source/build.
