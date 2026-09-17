# Qwen retained cache and QSA growth review — 2026-09-16

This record contains synthetic CPU and physical CUDA verification, followed by isolated arithmetic diagnostics. The public two-GPU checkpoint lifecycle passed; strict CUDA chunk/output gates remain open. Trained acceptance/performance, true tensor parallelism, and historical trained-model output drift are not qualified by these fixtures.

Follow-up (2026-09-17): the three open CUDA gates are resolved in [`verify-row-kernels-20260917`](../verify-row-kernels-20260917/README.md). Verify-width span graphs now run the one-token kernels, so the teacher-forced and committed-block gates pass bit for bit on CUDA; the chunked-versus-whole prefill gate is a bounded contract on CUDA with its reasons and numbers recorded there. The isolated precision prototypes below were not integrated.

## Imported and corrected implementation

The reviewed remote candidate was `/workspace/ts-q4e-retained/repo`. It retained complete request holders and cloned attention KV, QSA raw keys/positions, recurrent GDN/PLE state, and private MTP state. Source was first staged under `/tmp/tensorsharp-qwen-retained-review` for review. Its old suite had failures and QSA skips; those results are not recast as passes.

The integrated implementation now refuses to clone missing authoritative native state instead of copying stale host seeds; bounds clone capacity by the source allocation; restores device-state authority alongside speculative metadata; and disposes partially allocated holder tensors on allocation failure. At this stage, prefix checkpoints declined layer splits and tensor parallelism pending physical verification; subsequent layer-split admission is recorded below.

Reset now clears the stale-host flag when it invalidates native state. Previously, a reset followed by a larger first prompt called QSA cache export for a state entry that had intentionally been invalidated. The regression verifies first-prompt growth, reset-before-growth, complete logits/raw keys, and the original growth and rewind expectations with `TS_KV_INITIAL_TOKENS=8`.

## Additional real regression found

With small fixture caches, chunked 16+4 and whole 20 prompts produced differing full-vocabulary logits (max absolute difference 0.31553268); a third checkpoint-cloned chat also produced different greedy output. This remained reproducible at equal capacity 32. Raw QSA keys, pooled keys, queries, scores, and selected positions were identical. Attention probes showed positions 0..15 became zero after the graph shape changed.

`Q4eBinder::add` only placed buffers of at least 4096 bytes in persistent storage. The fixture attention KV buffers were 1024 bytes at capacity 32, so they became graph-local leaves; rebuilding for the suffix uploaded stale zero host seeds. The fix persists mutable buffers at every size and refuses a graph-local fallback when persistent binding fails. This changes TensorSharp-owned code only. Temporary diagnostic tracing was removed. The retained regression checks three distinct suffixes, growth and equal-capacity paths, exact raw keys and complete logits, and confirms each exercised attention K buffer is below 4096 bytes.

## Exact verification

- Fresh CPU native SHA256 `337d7e8b22867568a84974822e7353096a7d95a56a103e6d296a48e38699d0e3`.
- Unmodified ggml revision `456172ec733a135778adcd32d00e576a58232e45`.
- Fixture target SHA256 `f6d57de601159214dbd7d18a4d779bdbdda17860059a34625834652e6dc05e07`; head SHA256 `b7c93fea8db74d2cdfa360ed54f3c6c862f450a2adcd21282af971701f0544e3`.
- `qwen-resident-kv-r9.trx`: both originally failing tests pass after native fix, 2/2, no skips.
- `qwen-retained-growth-r10.trx`: runner aborted before tests because sandbox disallowed its local socket; not a pass.
- `qwen-retained-growth-r11.trx`: rebuilt managed tests, 41 pass, 0 fail, 0 skip, pinned actual mapped native identity.
- Includes retained A/B/A, checkpoint clones, speculative rebound parity/engagement, budget eviction, missing-state refusal, QSA first/reset/actual growth, holder isolation/rollback, video-position forward, and metadata/history tests. All heads and targets here are untrained fixtures.
- `evidence-r11.json` records exact source hashes and dependency cleanliness.

The prior failed numerical cases and unsupported scenarios remain in historical run records. No tolerance was widened and no skipped/unavailable model or device scenario is counted as passing.

## CPU replay

Build TensorSharp's native library with CPU enabled, CUDA/Metal disabled and test hooks enabled, then pin the resulting mapped native SHA256. Point both `GgmlNativeBuildDir` and `GgmlNativeBuildDirForTests` at that fresh build. The recorded run used the following test filter and environment:

```text
FullyQualifiedName~Qwen4ExpMtpIntegrationTests|FullyQualifiedName~Qwen4ExpRetainedCache|FullyQualifiedName~Qwen4ExpVideoForwardTests|FullyQualifiedName~Qwen4ExpQsaHistoryTests|FullyQualifiedName~Qwen4ExpMtpStateTests
TS_TEST_GGML_BACKEND=cpu
TS_TEST_QWEN4EXP_MTP_BACKEND=GgmlCpu
TS_TEST_QWEN4EXP_QSA=1
TS_TEST_QWEN4EXP_MTP_FIXTURE=/tmp/tensorsharp-current-fixtures/qwen4exp-qsa
TS_TEST_QWEN4EXP_NATIVE_SHA256=337d7e8b22867568a84974822e7353096a7d95a56a103e6d296a48e38699d0e3
TS_KV_INITIAL_TOKENS=8
```

The source hashes in `evidence-r11.json` identify that run. A later test-only two-GPU candidate fixture is separately gated by `TS_TEST_QWEN4EXP_LAYER_SPLIT=2`, with `TS_TEST_QWEN4EXP_MTP_BACKEND=GgmlCuda` and a pair of explicitly reserved visible physical GPUs. It checks the underlying clone core while public checkpoint admission remains closed. `qwen-split-fixture-compile-r12.trx` records compilation plus 13 retained CPU passes and one explicit split skip. Its compilation or CPU skip does not qualify physical layer splitting.

## Subsequent physical CUDA checkpoint qualification

The exact frozen r4 native (`7eaf98656edcd043884f3eb776bed68abbee29f5cb446e237bfa16afa5598af7`) and production DLLs were copied into an independent test-only application. A candidate test exercised the production clone core while the public split admission guard remained closed. On physical A40 GPUs 1 and 5 it passed complete state placement and clone parity: GDN/PLE native entries existed only on rank 0; QSA only on rank 1; MTP executed on rank 1; two independent clones reproduced full-vocabulary continuation logits and draft-head hidden/logit outputs, with an intervening other request and media position history.

After that proof, the only production source change removed `LayerSplitDegree <= 1` from `SupportsPrefixCheckpoints`; tensor parallelism remains refused. The isolated rebuilt Models DLL SHA256 is `39fffcc9a982c3d1d21192bdd947a61b5b6ace958b406a1b0a6f606bded8f0b0`. The public `TryCheckpointActiveCache` / `TryCloneRetainedCache` fixture then passed on the same physical GPUs. This qualifies these synthetic layer-split lifecycle/state paths, not trained-model acceptance, performance, or weight tensor parallelism.

Both the frozen-r4 and admitted-public-split single-CUDA suites recorded **40 passes and one strict numerical failure**, zero skips. The failing chunk16+4/whole20 case now preserves its history correctly but retains a full-logit difference (same greedy argmax). This numerical gate is still failed; the two-GPU clone proof does not waive it. Raw logs, TRX, before/after DLL/native identities, fixture hashes and process exit records are under `cuda-r4/` and `cuda-r7/`. These were bounded correctness runs; shared GPU activity means no timing qualification.

The isolated harness's initial missing-dotnet launcher and two unsuccessful build derivations remain on the VM at `/workspace/ts-qwen-retained-r4/build.log`, `/workspace/ts-qwen-retained-public-r5/models-build.log`, and `/workspace/ts-qwen-retained-public-r6/tests-build.log`. No frozen source or binary was modified.

## Teacher-forced arithmetic diagnosis (isolated candidates)

The additional teacher-forced fixture uses identical four-token and sixteen-token prefixes, then compares four scalar target forwards with one four-token target verification. It checks all 260 vocabulary entries for each of eight output rows, without sampling. Baseline CPU passes exactly. Baseline CUDA fails all eight exact rows (maximum absolute difference 0.00701249), while preserving the same argmax on this synthetic fixture. This is a numerical failure, not a passing trained-output qualification.

Target graph experiments were compiled into new actual object files and linked with the immutable r4 objects. Their build records assert both original native and object hashes remain unchanged. No experiment changed upstream ggml or the original tolerances. The native prototypes below are not integrated into production source.

| Evidence | Target arithmetic change | Exact teacher result |
| --- | --- | --- |
| `teacher-r9/` | Frozen r4 | CPU pass; CUDA eight rows fail |
| `precision-r10/` | Route F32/F32 ordinary and indexed matmuls through existing owned precision helper | Stored GDN/PLE/K/V state exact; all eight output rows still fail |
| `precision-r11/` | Also use owned F16/BF16 indexed expert arithmetic | Maximum output difference 0.00249469; eight rows fail |
| `precision-r12/` | Also use owned ordinary F16/BF16 arithmetic, including fallback attention | Four rows exact; four fail at maximum 9.536743e-7 |
| `precision-r12/`, separate disabled-fusion run | Same native, `GGML_CUDA_DISABLE_FUSION=1` | Eight rows exact; diagnostic only |
| `precision-r13/` | Preserve weighted expert product as an output before reduction; global fusion enabled | Eight rows exact; physical two-GPU checkpoint fixture passes |

The last boundary prevents shape-dependent upstream fusion of the expert multiply/reduction through a public graph flag. Its complete CUDA subset still records **41 passes, one failure, zero skips**: chunk16+4/whole20 retains small PLE/GDN drift and F16 QSA/K/V cache differences. No full-subset pass is claimed. The r13 native SHA256 is `bf53d455a70008867850dfd02291a1e2942c52e6f47457568185fccb5548263d`; its source, object, link command, baseline identities, candidate patch, and raw results are in `precision-r13/`.

These experiments isolate floating arithmetic and reduction behavior. The tiny fixture uses head dimension 8 and fallback attention; it does not exercise the trained head dimension 256 flash-attention route or trained quantized expert GEMMs. Large-prefill throughput, memory use, trained full-output parity, and acceptance/speed remain unqualified. Correctness timings were collected with shared physical GPUs and are not performance evidence.

## Further wide-prefill and flash diagnostics

The separate r15 experiment changes only the copied precision helper's column threshold so all floating matmuls use its width-invariant warp reduction, retaining the r13 target object. The original chunk16+4/whole20 test then passes all three suffixes and both growth conditions exactly. Its native SHA256 is `3e07f9344ce5b0e06761fa86aacef96ade6edad6d57329a91b9e55e910a29edf`. This confirms the large-width reduction switch as a source of drift in this fixture. It is **not an integrated or performance-qualified default**; wide GEMM throughput remains a material concern. The first r14 diagnostic failed compilation due to a comment masking a brace, before any execution; its failed source/log/build record remains on the VM.

The generator now accepts explicit attention head dimensions 64/256. Replaying its unchanged default reproduces both original target/head SHA256 values above. The head256 fixture manifest is under `flash256-fixture/`; it retains synthetic F32/F16 weights and does not cover trained quantized experts. It reaches the production flash-attention builder with F16 KV. Baseline r4 fails the original eight-row teacher comparison; r13 passes it.

The stronger committed-block test then feeds 32 identical teacher tokens through widths 1, 2, 3, and 4 after two prefixes, comparing every full-vocabulary row and exporting the complete vectors. CPU passes. CUDA baseline fails widths 2–4 (maximum 0.00533342); r13 passes all 256 row comparisons within its own arithmetic. **Cross-native scalar comparison still fails for r13:** every row changes, with maximum differences 0.0203637 and 0.0157155 for the two prefixes. The unchanged greedy argmax in this fixture does not establish no regression. The trained r13 replay also retained failures, so r13 remains unintegrated.

The r19 graph-column experiment instead uses the original scalar matmul operations for short blocks. It preserves the baseline scalar vectors exactly after the unchanged sixteen-token prefill. It changes short four-token prefill arithmetic and retains transient output differences up to 9.536743e-7 even though all inspected GDN/PLE/K/V state is exact. Those strict failures remain. Raw cumulative results, vector bytes and binary identities are in `committed-r20*`; `ts-qwen-committed-comparison-r1.json` records cross-native and within-native comparisons. An initial r19 launcher ran before app preparation completed and failed before test discovery; no passing result is attributed to it.
