# Radix default integration validation — 2026-09-17

TensorSharp.Server, TensorSharp.Cli, and TensorAgent now select the shared
engine's radix prefix cache by default for models advertising Tree readiness.
The CLI's duplicate KV mirror, prefix planner, and normal-generation decode
loops were removed, along with unused ModelService compatibility shims.
The explicit `TS_PREFIX_CACHE_MODE=legacy` diagnostic path and standalone
paged-cache benchmarks remain. `TS_SCHED_PREFIX_CACHE=0` disables reuse;
Server and CLI also expose `--no-prefix-cache`.

## Automated coverage

| Run | Passed | Skipped | Failed |
| --- | ---: | ---: | ---: |
| InferenceWeb non-benchmark suite excluding external-model, CUDA, and MLX requirements | 5,143 | 4 | 0 |
| TensorAgent warmup, memory policy, and background generation | 65 | 0 | 0 |
| Gemma 4 E2B Q8_0, Metal checkpoint exactness | 1 | 0 | 0 |
| Qwen 3.5 9B IQ4_XS, Metal checkpoint exactness | 1 | 0 | 0 |

The main suite includes page and holder reuse, concurrent branches, cold-output
equivalence, private/public scope isolation, media identity boundaries, explicit
cache markers, opt-outs, persistence, pool pressure, hard resource caps, empty
scope reclamation, cancellation, shutdown, disabled-batching fallback, CLI
streaming lifecycle, host rebuilds, and tiny managed Hunyuan models. Model-family
capability tests cover metadata and operation gates; they do not establish
real-weight coverage for every family or backend.

The four skipped tests are not passes:

- `DeepSeekTeacherTokenExportFixture.ExportPinned113RequestsMetadataOnly`:
  explicit tokenizer-export opt-in was absent.
- `DeepSeekTeacherTokenExporterTests.All113PinnedBodiesTraverseProductionPreprocessingWithSyntheticTokenizer`:
  explicit pinned corpus path was absent.
- `Glm5NextNativeSnapshotBoundaryTests.CaptureAndPartialRestoreExceptionsAreContainedAndResetRecovers`:
  requires a native library with test hooks and `TS_TEST_GLM_SNAPSHOT_BOUNDARY=1`.
- `PrefixCheckpointOwnershipTests.GemmaThrowingPostCommitTrace_DoesNotReportFailureOrLosePublishedOwner`:
  requires `TS_CB_DEBUG=1` before testhost startup.

Main test commands:

```sh
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj --no-restore \
  --filter 'Category!=Bench&Requires!=Models&Requires!=Cuda&Requires!=Mlx' \
  --logger 'trx;LogFileName=radix-default-unit.trx' \
  --logger 'console;verbosity=minimal' -m:1 /nodeReuse:false -p:WarningLevel=0

dotnet test TensorAgent/tests/TensorAgent.Tests/TensorAgent.Tests.csproj --no-restore \
  --filter 'FullyQualifiedName~PrefixCacheWarmupTests|FullyQualifiedName~EngineMemoryPolicyTests|FullyQualifiedName~BackgroundGenerationTests' \
  --logger 'console;verbosity=minimal' -m:1 /nodeReuse:false -p:WarningLevel=0
```

## Real-model checks

Both Metal checks used `MAX_CONTEXT=4096`, default Tree mode, and 12 generated
tokens. Each compares a new conversation cloned from the public checkpoint
against a cold engine, then repeats the comparison after exporting and importing
the checkpoint into a fresh engine. Both output comparisons were exact.

| Model | Public prefix reused | Persisted checkpoint |
| --- | ---: | ---: |
| Gemma 4 E2B Q8_0 | 1,217 tokens | 13.1 MB |
| Qwen 3.5 9B IQ4_XS | 1,176 tokens | 87.0 MB |

These ran `PrefixCheckpointExactnessTests.Gemma4_NewChatFromACheckpoint` and
`PrefixCheckpointExactnessTests.Qwen35_NewChatFromACheckpoint` with
`TS_TEST_GGML_BACKEND=metal` and `TS_TEST_MODEL_DIR` pointing to the respective
local checkpoint. Qwen covers recurrent GDN state; Gemma covers its windowed
cache. Local detailed logs and TRX results are under `/tmp/radix-native-final`
and `/tmp/radix-{gemma4,qwen35}-metal-final.log`.

## Dependencies and limits

No native sources or upstream ggml files were edited or rebuilt. The upstream
checkout is clean at `456172ec733a135778adcd32d00e576a58232e45`.
The existing test `libGgmlOps.dylib` has SHA-256
`9a5f22ed7d0ad55360ea0782459450a6caed28f52dcb92f52a57df241e06dd74`.

This is correctness and lifecycle validation, not a throughput or peak-memory
benchmark. CUDA, Vulkan, physical iOS devices, and the full real-model/backend
matrix were not tested; `nvcc` was unavailable. The CLI now uses the shared
scheduler for normal generation, so its former standalone MLX greedy pipeline
is no longer used there. The direct model benchmark retains that optimization;
the effect on normal CLI MLX throughput has not been measured. CLI prefill
timing now reports time to first token, including scheduling and sampling.
