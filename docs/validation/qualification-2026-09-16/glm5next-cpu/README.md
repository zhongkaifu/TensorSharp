# GLM5Next recurrent rollback — CPU qualification

GLM5Next now implements KDA recurrent-state capture and restore for n-gram speculative decoding on the managed path and native executor. Verify returns every requested hidden/logit row. A rejected window restores the captured convolution and SSM state, rewinds its position, then replays the accepted prefix. NextN/MTP draft-head execution remains unsupported.

The imported candidate needed three corrections before this run: native snapshot allocation/copy exceptions now terminate at the C ABI; a partially restored slot remains unusable until a successful checked reset; and binding/restoring a request slot synchronizes the managed cache position with the native slot. CPU and CUDA fixtures are separate, with explicit CUDA skips replacing the candidate's early-return passes.

The fresh macOS arm64 CPU-only Release library is SHA256 `9ae18f5d1fe201ce4ef4f30397969060c2ea60c5957d456dd1d3dd31712df442`. Upstream ggml remained unchanged at `456172ec733a135778adcd32d00e576a58232e45`. [The manifest](manifest.json) pins native source/object/library and managed assembly identities. [The build log](native-build.log.gz) records actual native compilation.

- [Managed fixtures](glm5next-cpu.trx): **20 passed, 3 CUDA cases explicitly skipped**, no failed tests. Coverage includes CPU/managed n-gram engagement, partial rejection, full-vocabulary continuation parity, hidden-row parity, A/B/A holder positions and rollback, native bad-allocation/unknown-exception injection after partial restoration, healthy-peer isolation, and reset recovery. Existing CPU tensor-parallel partition fixtures and the mapped-native-identity test also pass.
- [Legacy DeepSeek C ABI checks](legacy-boundary32.json): **32/32 passed** on this new binary, retaining all original expectations.

The initial VSTest launch was aborted when sandboxing blocked its local communication socket; [that result](aborted-run.trx) is retained separately. The approved direct rerun completed. The earlier boundary record remains historical: its original temporary build path was subsequently rebuilt and no longer contains its earlier recorded binary.

These are tiny KDA fixtures. They do not establish trained-checkpoint acceptance, performance, memory use, mixed KDA/MLA rollback, physical GPU tensor parallelism, or CUDA execution. The old VM CUDA run retains its 7-pass/10-fail status. Its process was pinned to GGML CUDA while GLM's native executor requires the managed allocator to remain GGML CPU; a new CUDA run must use `TS_TEST_GGML_BACKEND=cpu` with `TS_TEST_GLM_CUDA=1`.

```sh
cmake -S TensorSharp.GGML.Native -B /tmp/tensorsharp-glm5spec-build \
  -DCMAKE_BUILD_TYPE=Release -DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON \
  -DTENSORSHARP_GGML_NATIVE_ENABLE_METAL=OFF \
  -DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA=OFF \
  -DTENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF
cmake --build /tmp/tensorsharp-glm5spec-build --target GgmlOps --parallel 4
```

The managed build used `dotnet test -c Release --no-restore` with `TensorSharpSkipGgmlNative=true`, `TensorSharpSkipMlxNative=true`, and both `GgmlNativeBuildDir` and `GgmlNativeBuildDirForTests` set to `/tmp/tensorsharp-glm5spec-build`. The completed test command was:

```sh
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release --no-build --no-restore \
  --environment TS_TEST_GLM_SNAPSHOT_BOUNDARY=1 --environment TS_TEST_GGML_BACKEND=cpu \
  --filter 'FullyQualifiedName~Glm5NextSpeculationEligibilityTests|FullyQualifiedName~Glm5NextSpeculativeRollbackTests|FullyQualifiedName~Glm5NextNativeSnapshotBoundaryTests|FullyQualifiedName~Glm5NextNativeTensorParallelTests|FullyQualifiedName~NativeGgmlIdentityTests' \
  --logger 'console;verbosity=normal' --logger 'trx;LogFileName=glm5next-cpu-r2.trx' \
  --results-directory /tmp/tensorsharp-glm5spec-results -m:1 /nodeReuse:false
python3 eng/tests/dsv4-execution-boundary.py \
  --library /tmp/tensorsharp-glm5spec-build/libGgmlOps.dylib \
  --report /tmp/tensorsharp-glm5spec-boundary32.json
```
