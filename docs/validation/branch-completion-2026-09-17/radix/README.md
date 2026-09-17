# Radix branch completion — 2026-09-17

The `claude/radix-prefix-cache` and `claude/radix-m2` branches are complete for
their existing milestones: Phase 0, M0a/M0c/M0e, M1 tree core, and M2 model
contracts. Every real model still advertises **Legacy readiness**. M3 coordinator
and Shadow mode, M4 authority, M5 family enablement, and M7 lanes/persistence are
future milestones, not enabled features of this completion.

## Changes made during completion

- Preserved the original branches' ancestry and included the allocation-benchmark
  GC configuration fix from `cb98c0fb`.
- Corrected mixed reclaim batches. Pressure, invalidation, and reset now force
  disposal even when an ordinary pooling release was queued first. Duplicate
  keys can promote the reason without double-counting bytes; the sink still
  receives one batch and performs one graph reset.
- Completed the previously untracked ABBA runner and summary tools. They reject
  failed children, missing passes/rounds, incomplete workload rows, and invalid
  metrics. Each native override is staged beside its own managed runner, where
  the native import resolver actually loads it, and recorded by SHA-256.
- Aligned model-test file overrides with the discovery gate. An explicit file
  accepted by `ModelSkip` is now loaded directly by both matching helpers.
  Directory filtering and companion exclusions remain covered by regression
  tests.

The original dirty/untracked work was left in its original worktrees. Completion
changes were made on `codex/complete-radix-prefix-cache`.

## Dependency and build identity

Both local and remote native builds used unchanged upstream ggml revision
`456172ec733a135778adcd32d00e576a58232e45`; `git status --short` was empty in
both dependency checkouts. No ggml patches were applied.

Local validation used an Apple M5 Pro, .NET SDK 10.0.302/runtime 10.0.10 and the
Metal native library SHA-256
`64e85b3b33c5fe9042fa1af66fb0fb6f7f1fb7d79784612ae0dfa2dde3640d3f`.
The unrelated MLX native fetch stalled and was stopped; rebuilding with
`-p:TensorSharpSkipMlxNative=true` succeeded. MLX was not tested.

Remote validation used physical GPU 4, an NVIDIA A40 with driver 570.195.03,
and the combined integration build supplied by the root task. Its native
library SHA-256 was
`0e75fe2580062b8335e0197f8a71021248aa8c665529a9e62f89aa205c92625b`.
The GPU lock was held for the test sequence and released afterward. This
validates the combined build, not a separate radix-only CUDA binary.

## Tests

| Lane | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Prefix-cache/retention CPU suite, 20,000 property seeds | 291 | 0 | 8 |
| M0 fingerprint/rebuild/truncation/block-transfer/injection tests | 51 | 0 | 0 |
| DeepSeek V4.1 managed fixture, CPU contract conformance | 1 | 0 | 0 |
| Real-model Metal checks listed below | 5 | 0 | 0 |
| Real-model CUDA checks listed below | 4 | 0 | 0 |
| Model path helper regressions | 3 | 0 | 0 |
| ABBA runner/summary Python regressions | 7 | 0 | 0 |
| Gemma Metal optimized-JIT lifetime regression and batched parity | 2 | 0 | 0 |

These are separate invocations, not a deduplicated suite total. The original
eight skips were opt-in model tests. Six were subsequently executed successfully
in the fixture and Metal lanes. Full-model Qwen3.8 Flash Next and DeepSeek V4.1
Flash radix conformance remain unexecuted and are not counted as passing.

The property harness completed 20,000 seeds in 425.5 seconds, with 5,577,492
plans and brute-force comparisons, 4,661,865 acquires, 4,531,099 finishes, and no
failing seeds. Coverage for PrefixTree, ResumabilityRules, EvictionLists, and
InflightTable was 100% of lines; branch coverage was 99.31% for PrefixTree and
100% for the other three.

Real-model checks used Gemma 4 E4B Q8_0 contract conformance; Qwen3.5 9B Q8_0
contract conformance, end-state byte accounting, and primary decode after a
holder; and GPT-OSS 20B Q8_0 contract conformance. All five passed on Metal.
The four Gemma/Qwen checks also passed on CUDA. GPT-OSS CUDA conformance was
not completed: the VM's available variant is MXFP4, while the original test's
directory selector requires Q8_0. The fixed explicit-file helper was also
exercised by a second passing Metal GPT-OSS run.

The DeepSeek fixture was the prepared managed metadata fixture
`deepseek41-fixture.gguf`, SHA-256
`b455020bd7500c5a835744fb189443451e0849331e15ac72557d58d5eafb13c4`.
It provides CPU contract coverage and is not evidence for full-model CUDA
conformance.

Exact test outcomes, coverage and model evidence are in
[local-validation.json](local-validation.json) and
[model-conformance.json](model-conformance.json). Setup attempts that failed
before model loading or skipped because of a nonrecursive directory selection
are recorded separately from the successful reruns.

## Benchmarks and limitations

The full-scale `RadixTreeBench --trace all` run passed all six gates. It recorded
zero allocated bytes across all 83,240 probes. The p99 measurements were 12.6 µs
for a 32k probe, 13.6 µs for a 32k insert, 126.6 µs for a 262k probe, 6.0 µs to
evict one victim, and 0.029 ms admission overhead at queue depth 128. See
[radix-tree-benchmark.json](radix-tree-benchmark.json).

An explicit `DOTNET_gcConcurrent=1` override was rejected with exit code 2, as
required for credible no-allocation measurements. These are synthetic CPU tree
measurements. An existing Metal benchmark was active during the run, so the
host was not isolated. They do not establish end-to-end model throughput.

The pre-existing Gemma ABBA series contains a `b-1` child exit 139 and only two
of its three expected measured passes. The previous wrapper incorrectly
returned success. The completed summary correctly rejects that series rather
than computing a partial median. A follow-up audit identified the crash in the
older control build, demonstrated premature collection of the batched hidden
tensor, and fixed its managed ownership. The forced-GC regression failed before
the fix and passed afterward with the optimized JIT; see
[gemma-batched-lifetime.md](gemma-batched-lifetime.md). No fresh large model ABBA series was run;
there is no new model performance nonregression claim. The seven automated
ABBA tests cover child failures, missing passes/rows, complete data, and actual
staging of the selected native override. Token parity still requires
`compare.py`, as documented in the benchmark README.

## Reproduction

From a built completion worktree:

```sh
TENSORSHARP_GGML_NO_UPDATE=1 PREFIX_CACHE_TREE_SEEDS=20000 \
  dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release \
  -p:TensorSharpSkipMlxNative=true \
  --filter 'FullyQualifiedName~PrefixCache|FullyQualifiedName~SlotRetention|FullyQualifiedName~RetainedCachePolicy|FullyQualifiedName~Qwen35PrimaryDecodeAfterHolder' \
  --collect:'XPlat Code Coverage'

dotnet run -c Release --project benchmarks/RadixTreeBench -- --trace all
python3 -m unittest discover -s benchmarks/AgentTurnBench -p test_abba.py
```

For the model lanes, run each family in a fresh `dotnet test --no-build`
process with `TS_TEST_GGML_BACKEND=metal` or `cuda` and set
`TS_TEST_MODEL_DIR` to that model's leaf directory or, after `0aaa01a2`, its
explicit file path. CUDA runs used `CUDA_VISIBLE_DEVICES=4`, `MAX_CONTEXT=8192`,
and `TENSORSHARP_TP_DEGREE=1`. Use the exact fully qualified test names recorded
in `model-conformance.json` as filters. The DeepSeek CPU fixture uses
`TS_TEST_DSV41_FIXTURE_DIR` instead.
