# Combined CUDA build r4 — 2026-09-16

Release remains blocked. This immutable snapshot passed compilation and native fixtures, but its integrated portable suite failed one test. Later fixes require separate records.

- Source manifest SHA256: `f7431257148bd5168c0f80e3049b36fc5228c19a8d47448e09e60ff7eb43eb28` ([source.json](source.json)).
- Fresh native SHA256: `7eaf98656edcd043884f3eb776bed68abbee29f5cb446e237bfa16afa5598af7`.
- Unchanged upstream ggml: `456172ec733a135778adcd32d00e576a58232e45`.
- Fresh object hashes, commands, managed identities, and final source/binary audit: [build.json](build.json).
- CUDA Release native, server, tests, and benchmark builds passed; C ABI boundary checks passed.
- Native CTest completed on physical GPUs 1 and 5. See [native-tests.xml](native-tests.xml) for individual passes and unavailable larger physical-rank cases. Missing physical ranks are not passes.
- MTP operator: four capacity/alignment combinations per CPU/CUDA backend, 592 checks per backend. Target state: 132 checks per backend. QSA: 20 numerical and 21 invalid-input cases per backend. These are untrained fixtures.
- Integrated portable suite: **4,288 passed, 1 failed, 4 skipped** ([raw log](managed-portable.log), [TRX](portable/portable.trx)). Failure: `Head_RetainsBonsaiNativeEntryPointsInRelease` scans native source and includes the test-only GLM fault-injection function because it uses the production `TSG_EXPORT` macro, even though it is behind a test-only preprocessor guard. The release export list correctly excludes that test-only symbol. The original failed result remains unchanged.

The VM was shared with other workloads. No latency, throughput, trained-model accuracy, or full-checkpoint tensor-parallel qualification follows from these build and fixture passes. Native verbose compiler output remains at `/workspace/ts-codex-20260916-r4/results/build1/native-build.log`.
