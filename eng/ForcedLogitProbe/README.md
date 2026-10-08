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
The TensorSharp native library SHA-256 after the precision fix was
`0146374922609a3f88038c61540467db26dc05dbf8098b128a87e6d2eacc5cfc`.
That library also underlies the full Qwen snapshot width matrix. A subsequent
diagnostic I/O safety change produced
`5bd4436676564091850c8862ed50e971ad8924147399742e4299fcf895d1c941`.
The complete TP comparison was repeated with that library and passed; all ten
logit files were byte-identical to the earlier run. The snapshot width matrix
was not repeated after this diagnostic-only change.

The default dense Qwen CUDA policy now registers model-owned Q8_0 weight keys for
the existing TensorSharp Q8×F32 projection kernel. It retains F32 activations and
quantized weight storage, with no full F32 weight cache. This applies consistently
to the fused TP1 and sharded TP2 paths, including prefill and decode. Other model
families, CPU execution and unregistered weights retain their previous policy.

With this policy, TP1 versus TP2 **passed** all unchanged gates for the complete
five-case, 24-step run: 120 rows of 248,320 logits (29,798,400 values), all finite,
with all 120 top-1 tokens equal. Maximum relative L2 was
`0.0005205465902312378`, minimum cosine was `0.9999998736272625`, and maximum
absolute error was `0.006324291229248047`. Both processes exited normally with
code 0. The candidate used both physical GPUs, with the sharding and per-rank
residency described below.

The precision choice has a cost. The same five prompts and identical generated
token histories took about 4.03 → 9.65 seconds for TP1 and 7.56 → 10.16 seconds
for TP2 in the recorded cold validation processes. These are launch-to-observed-
exit intervals from the sampling harness, whose polling interval was 0.5 seconds;
they include model loading, file output and comparison work. Native compilation
also ran during the final trial. They show an observed validation-time tradeoff,
not a controlled serving-throughput benchmark.

### Historical failure and localization

Before the precision fix, TP2 **failed** the same full-logit gate. The native
library was `c798d3e5b3d5675843aa59096af6887d6211c99e327d3a8c3c029e91f7bf4cb4`.
Maximum relative L2 was
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
the discrepancy but did not establish its cause or qualify that implementation.

Full intermediate tensors then localized the first substantial discrepancy for
the synthetic one-token prompt. Embeddings were byte-identical, and residuals
through layer 6 differed by only about `1e-7` relative L2. Layer 7's normalized
QKV input still differed by only `1.244878e-7`, but its quantized projection
differed by `0.000330994`. Reproducing the activation quantization from those
inputs showed a Q8_1 activation rounding from -27 to -26 because a block maximum
changed by two F32 ULPs. `q8-rounding-control.cpp` confirmed this with the actual
unchanged ggml CUDA quantizer on the A40: maxima `3.11877036` and `3.11877084`
produce different quantized values for the same activation `-0.650767148`.
The projected error then grew through later blocks.
Keeping activations in F32 removes that discontinuity; the independent Q8/F32
kernel tests compare against a double-precision decoded-weight oracle.

The final native cache tests passed 4/4, including shared-budget permit lifetime,
exact streamed/cached F32 and
Q8_0 results, quota fallback, per-rank accounting, and deliberately poisoned
abandoned cache initialization. Releasing retained TP graphs during cache cleanup
removed an observed CUDA shutdown abort. TP2 then exited normally with code 1 for
the numerical gate in the historical run. Before/after cache-fix logit files
were byte-identical. The later precision-policy fix resolves the strict TP gate.
The independent Q8/F32 CPU projection test and CUDA projection test passed;
the CUDA test ran on physical GPU 1 with 86 cases, each executed twice. Three
real-native weight-registration lifetime tests also passed, including rollback
after injected registration failure.

### Optional intermediate-tensor evidence

`TS_QWEN35_TENSOR_DUMP` writes raw F32 embeddings, per-layer residuals, and
post-attention/GDN residuals before the FFN. Use a fresh process and a one-token
first forward: the native fused capture covers the initial one-token graph,
while the managed TP capture covers its first forward. These diagnostics are off
by default. They add downloads and retain intermediates, so never enable them
for performance measurements. Filesystem failures are logged and leave the
successful forward intact. With the final library, an intentionally invalid
directory was tested on both TP1 and TP2: each logged the failure, exited normally,
and produced byte-identical logits to its same-arm baseline for all 24 steps
(5,959,680 values per arm), preserving recurrent state progression.
For example, after the setup above:

```sh
printf '[{"name":"synthetic_single","prompt_tokens":[1]}]\n' > "$OUT/trace-cases.json"
mkdir -p "$OUT/trace1/tensors" "$OUT/trace2/tensors"
TENSORSHARP_TP_DEGREE=1 TS_QWEN35_TENSOR_DUMP="$OUT/trace1/tensors" \
  dotnet "$PROBE/ForcedLogitProbe.dll" "$MODEL" "$OUT/trace1" ggml_cuda \
  --cases "$OUT/trace-cases.json" --steps 1
TENSORSHARP_TP_DEGREE=2 TS_QWEN35_TENSOR_DUMP="$OUT/trace2/tensors" \
  dotnet "$PROBE/ForcedLogitProbe.dll" "$MODEL" "$OUT/trace2" ggml_cuda \
  --reference "$OUT/trace1"
python3 eng/ForcedLogitProbe/compare-tensors.py \
  "$OUT/trace1/tensors" "$OUT/trace2/tensors" > "$OUT/tensor-comparison.json"
```

The one-step diagnostic does not replace the full qualification run.
The comparer exits nonzero for missing, empty, truncated, unreadable, or
nonfinite tensor data. Finite numerical differences are reported without a
failing exit code: this tool localizes differences and does not qualify them.

The small rounding control uses a dependency-internal launcher whose signature
is pinned to the recorded ggml revision. Build it separately from TensorSharp:

```sh
g++ -std=c++17 eng/ForcedLogitProbe/q8-rounding-control.cpp \
  -IExternalProjects/ggml/include -I/usr/local/cuda-12.8/include \
  -L/usr/local/cuda-12.8/lib64 -lcudart -L"$NATIVE" -lGgmlOps \
  -Wl,-rpath,"$NATIVE" -o "$OUT/q8-rounding-control"
CUDA_VISIBLE_DEVICES=0 "$OUT/q8-rounding-control"
```

This reports rounding sensitivity and is not another passing model test.

This is one model, quantization, topology and batch width, with no multi-node,
NCCL, direct-peer, production concurrency, endurance or throughput qualification.
The 128-token case does not validate long-context behavior. Matching top-1 tokens
and successful residency tests must not be reported as passing TP logit parity.
