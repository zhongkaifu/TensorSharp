# Fresh CPU exception-boundary verification — 2026-09-16

The corrected DeepSeek fixture passes all **32/32 actual C ABI cases**: the original 30 execution/reset expectations and two additional head-present batched-decode refusals. The correction was already present in commit `8aae2f058abf58a93d01a71da4122b2daee17c56`; this run required no implementation or tolerance changes.

Ordinary batched-decode fault cases have no draft head, allowing the armed fault to reach execution. Separate head-present cases verify refusal occurs before execution and preserves slot positions, failure latches, Engram histories, slot and graph ownership, draft-head state, and output canaries before reset. The six originally failed batched cases retain their existing expectations.

A new macOS arm64 CPU-only Release build compiled `ggml_ops_deepseek4.cpp` with test hooks. [Compile evidence](compile-evidence.log) records the actual compiler invocation. [The manifest](manifest.json) records source, object, binary, complete build-log, and CMake cache hashes. The ggml checkout remained clean at `456172ec733a135778adcd32d00e576a58232e45` before and after the build.

| Check | Result | Scope |
|---|---:|---|
| [DeepSeek C ABI boundaries](execution-boundary.json) | 32/32 | Exception containment, reset eligibility, CPU tensor reset/restore, and head-present refusal invariants |
| Attention allocation exception boundary | 1/1 | C++ unwinding and changed-input compute after an intercepted allocation failure |
| DSpark ring guards | 1/1 | 281,727 production predicate/independent ring-simulation checks, 24,680 branches, 1,680 refusals |

The two CTest results and detailed scope are retained in [ctest-detail.log](ctest-detail.log). These fixtures do not qualify trained heads, physical multi-GPU feature transfer, CUDA/Metal execution, performance, or the final combined Release build. Earlier VM runs, including the stale-object run, retain their original status.

```sh
cmake -S TensorSharp.GGML.Native -B /tmp/tensorsharp-boundary-20260916/build \
  -DCMAKE_BUILD_TYPE=Release -DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON \
  -DTENSORSHARP_GGML_NATIVE_ENABLE_METAL=OFF \
  -DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA=OFF \
  -DTENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF
cmake --build /tmp/tensorsharp-boundary-20260916/build \
  --target GgmlOps GgmlOpsAttentionExceptionBoundaryTest GgmlOpsDsv41DsparkRingTest \
  --parallel 4 --verbose
python3 eng/tests/dsv4-execution-boundary.py \
  --library /tmp/tensorsharp-boundary-20260916/build/libGgmlOps.dylib \
  --report /tmp/tensorsharp-boundary-20260916/execution-boundary.json
ctest --test-dir /tmp/tensorsharp-boundary-20260916/build \
  -R '^(attention-exception-boundary|deepseek41-dspark-ring-guards)$' --output-on-failure
```
