# Teacher-forced full-logit comparison

`ForcedLogitProbe` records every vocabulary logit for a baseline, then feeds its
exact token histories to a candidate. This prevents divergent sampled tokens
from obscuring the numerical comparison. The default cases cover arithmetic,
science, code, a one-token synthetic prompt and a 128-token synthetic prompt.
Use at least 16 steps; the recorded validation below used 24 steps per case.

Each candidate row must be finite, have relative L2 error at most `0.001`, cosine
similarity at least `0.999999`, and the same top-1 token. All conditions must hold
for exit code 0. Exit code 1 means a failed gate, including finite outputs with
matching top-1 but excessive full-logit error. Baseline exit code 0 establishes
finite execution only; it does not qualify numerical equivalence.

Outputs include `cases.json`, `metrics.json`, and raw little-endian float32 rows
in `<case>.f32`. GGML metrics also record cache accounting separately for each
requested rank. These counters cover lazy device copies and explicit preloads,
not total VRAM or scratch/KV/driver allocations. Keep generated files in ignored
`artifacts/` or `docs/validation/`.

## CUDA TP1 versus TP2 reproduction

Run from the repository root on Linux with .NET 10, a CUDA toolkit and two GPUs.
Set the model path and architecture to match the deployment. Preserve the same
native library, managed binaries, model and environment between arms; only the
TP degree changes. Do not run other model workloads during this comparison.

This deployment requires host staging: direct CUDA peer transfers returned
corrupt data despite advertised peer support. `GGML_CUDA_NO_PEER_COPY=ON` selects
the unchanged upstream fallback. Disabling NCCL peer transfers alone does not
disable ggml's own peer-copy path.

```sh
MODEL=/absolute/path/Qwen3.5-0.8B-Q8_0.gguf
OUT="$PWD/artifacts/forced-logits"
NATIVE="$PWD/TensorSharp.GGML.Native/build-staged"
PROBE="$PWD/eng/ForcedLogitProbe/bin/Release/net10.0"
mkdir -p "$OUT"

cmake -S TensorSharp.GGML.Native -B "$NATIVE" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON \
  -DTENSORSHARP_GGML_NATIVE_ENABLE_METAL=OFF \
  -DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_NO_PEER_COPY=ON
cmake --build "$NATIVE" --parallel 12 \
  --target GgmlOps GgmlOpsCacheBudgetTest GgmlOpsCacheMemoryIntegrationTest
dotnet build eng/ForcedLogitProbe/ForcedLogitProbe.csproj -c Release -m:1 \
  -p:BuildInParallel=false -p:TensorSharpSkipGgmlNative=true \
  -p:TensorSharpSkipMlxNative=true
cp "$NATIVE/libGgmlOps.so" "$PROBE/libGgmlOps.so"

export CUDA_VISIBLE_DEVICES=0,1 GGML_CUDA_ALLREDUCE=none NCCL_P2P_DISABLE=1
export LD_LIBRARY_PATH="$NATIVE:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
unset GGML_CUDA_P2P TENSORSHARP_LAYER_SPLIT_DEGREE
unset TS_GGML_TP_PARALLEL TS_GGML_TP_CUDA_GRAPHS TS_GGML_TP_FUSED_MATMUL

ctest --test-dir "$NATIVE" --output-on-failure \
  -R '^(cache-allocation-budget|cuda-cache-memory-single|cuda-cache-memory-multi)$'

{
  git rev-parse HEAD
  git -C ExternalProjects/ggml rev-parse HEAD
  git -C ExternalProjects/ggml status --porcelain
  dotnet --info
  /usr/local/cuda-12.8/bin/nvcc --version
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
  grep -E 'CMAKE_CUDA_ARCHITECTURES:|GGML_CUDA_NO_PEER_COPY:|GGML_CUDA_NCCL:' "$NATIVE/CMakeCache.txt"
  sha256sum "$MODEL" "$PROBE/libGgmlOps.so" "$PROBE/ForcedLogitProbe.dll" \
    "$PROBE/TensorSharp.Models.dll" "$PROBE/TensorSharp.Backends.GGML.dll"
} > "$OUT/identity.txt"

TENSORSHARP_TP_DEGREE=1 dotnet "$PROBE/ForcedLogitProbe.dll" \
  "$MODEL" "$OUT/tp1" ggml_cuda --steps 24 > "$OUT/tp1.log" 2>&1
echo "TP1 exit: $?"
TENSORSHARP_TP_DEGREE=2 dotnet "$PROBE/ForcedLogitProbe.dll" \
  "$MODEL" "$OUT/tp2" ggml_cuda --reference "$OUT/tp1" > "$OUT/tp2.log" 2>&1
echo "TP2 exit: $?"
```

Require a clean upstream `ggml` checkout and preserve the identity file with the
results. If testing uncommitted TensorSharp changes, additionally record the diff
and source-file hashes; the commit alone does not identify that build. Confirm
the process loaded the copied library using `/proc/<pid>/maps`. During the run,
`nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv` associates
the same process with both physical GPUs. Model logs must show sharded weight
residency on both ranks, and both ranks must report available cache telemetry.

Controls can reuse the same baseline without changing thresholds:

```sh
# Same single-device implementation, on the other physical GPU.
CUDA_VISIBLE_DEVICES=1 TENSORSHARP_TP_DEGREE=1 dotnet "$PROBE/ForcedLogitProbe.dll" \
  "$MODEL" "$OUT/tp1-gpu1" ggml_cuda --reference "$OUT/tp1"
# Disable rank parallel dispatch and CUDA graph reuse together.
TENSORSHARP_TP_DEGREE=2 TS_GGML_TP_PARALLEL=0 TS_GGML_TP_CUDA_GRAPHS=0 \
  dotnet "$PROBE/ForcedLogitProbe.dll" "$MODEL" "$OUT/tp2-serial-no-graphs" \
  ggml_cuda --reference "$OUT/tp1"
# Exercise the existing opt-in fused-matmul switch where the model supports it.
TENSORSHARP_TP_DEGREE=2 TS_GGML_TP_FUSED_MATMUL=1 \
  dotnet "$PROBE/ForcedLogitProbe.dll" "$MODEL" "$OUT/tp2-fused-matmul" \
  ggml_cuda --reference "$OUT/tp1"
```

## Recorded result and limits

Validation on 2026-10-07 (2026-10-08 UTC) used Ubuntu 24.04, two NVIDIA A40 GPUs,
driver 570.211.01, CUDA 12.8.93, .NET SDK 10.0.401, and an unchanged ggml checkout
at `ffa4e8b80930029a35991f94e7c8a93cd67730ab`. The model SHA-256 was
`0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c`.
The final TensorSharp native library SHA-256 was
`c798d3e5b3d5675843aa59096af6887d6211c99e327d3a8c3c029e91f7bf4cb4`.

TP1 and TP2 each executed 120 rows of 248,320 logits (29,798,400 values). TP2
**failed** the unchanged full-logit gate: maximum relative L2 was
`0.041408444438891134`, minimum cosine was `0.9991471466963167`, and maximum
absolute error was `0.6113646030426025`. All values were finite and all 120 top-1
tokens matched. Differences began at step zero. TP2 loaded 97 quantized weights
on rank 0 and 96 on rank 1, with additional F32 shards on both ranks; process
sampling and per-rank telemetry confirmed actual two-GPU use.

The TP1 GPU-1 control was byte-identical to TP1 GPU-0. Disabling rank parallelism
and CUDA graphs did not alter any TP2 logit bytes. The fused-matmul switch likewise
did not alter TP2 output. This tied-embedding model uses the per-operation TP
decode fallback because its LM head is not column-sharded; the whole-model TP
decode path was unavailable and is not counted as tested. These controls narrow
the discrepancy but do not establish its cause or qualify TP2 accuracy.

The final native cache tests passed 3/3, including exact streamed/cached F32 and
Q8_0 results, quota fallback, per-rank accounting, and deliberately poisoned
abandoned cache initialization. Releasing retained TP graphs during cache cleanup
removed an observed CUDA shutdown abort. Final TP2 exits normally with code 1 for
the numerical gate. Before/after full logit files were byte-identical, so these
cache fixes did not resolve the TP numerical discrepancy.

This is one model, quantization, topology and batch width, with no multi-node,
NCCL, direct-peer, production concurrency, endurance or throughput qualification.
The 128-token case does not validate long-context behavior. Matching top-1 tokens
and successful residency tests must not be reported as passing TP logit parity.
