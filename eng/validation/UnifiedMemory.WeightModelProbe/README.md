# Real Qwen35 file-backed weight execution

This .NET 10 CUDA probe compares an ordinary resident dense Qwen35 Q8_0 model
against the explicit `WeightStreamingOptions` model adapter. The candidate reads
original GGUF file regions through a bounded host buffer and evaluates each
projection in synchronous output-row tiles. It does not preload or concatenate
whole quantized weights, expose a weight pointer to cached graphs, or mmap the
checkpoint. Embeddings read only the requested rows.

Build against unchanged upstream ggml and the TensorSharp native streaming
operator, then copy the matching native library beside the probe:

```sh
dotnet build eng/validation/UnifiedMemory.WeightModelProbe -c Release --no-incremental \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
export CUDA_VISIBLE_DEVICES=0
export TS_VALIDATION_DEVICE=NVIDIA_A40_GPU0
export TS_VALIDATION_GGML_REVISION=ffa4e8b80930029a35991f94e7c8a93cd67730ab
dotnet eng/validation/UnifiedMemory.WeightModelProbe/bin/Release/net10.0/UnifiedMemory.WeightModelProbe.dll \
  --model /workspace/models/Qwen3.5-0.8B-Q8_0.gguf \
  --json artifacts/unified-memory/weight-model.json \
  --steps 4 --prompt-tokens 32 --cycles 2 --tile-bytes 1048576 --token-rows 32 \
  --host-bytes 2097152 --device-bytes 2097152
```

On Windows, copy `GgmlOps.dll` into the same output directory and set the
environment variables using PowerShell. The Linux run additionally examines
`/proc/self/maps` for a model mapping; Windows reports this check unavailable,
not passed. Always inspect the recorded native path/hash after replacing a DLL.

Two complete chat prompts generate independent resident references. The streamed
model receives the same prompts and reference token histories. Every vocabulary
row must be finite, preserve top-1, have relative L2 error at most `0.001`, and
cosine similarity at least `0.999999`, matching the existing ForcedLogitProbe
gate. Resident fused execution and streamed per-operator execution can differ in
floating-point operation order, so this is not a byte-exact comparison. Early
EOS fails rather than counting an empty answer as a pass.

By default the streamed model loads, runs both prompts and disposes twice,
reusing the same independent references. Each cycle must return its budget to
zero. With a test-enabled native build, optional counters also require the
GDN chunked graph cache to become nonempty during execution and both GDN graph
caches to return to zero after disposal. Missing test hooks are reported as
unavailable. The process explicitly shuts down its owned backend before exit.

The report verifies actual file reads, repeated operator tiles, embedding row
reads, host/device payload peaks below their configured budgets, no full-weight
preloads, and zero owned charges after disposal. An external reservation in the
same host pool must refuse construction before staging allocation; releasing it
must allow a subsequent model to run. Model, assembly and native hashes are
recorded. A second external reservation exhausts GPU credit before a forward;
after refusal, the model must reject another forward until `ResetKVCache`
succeeds. The subsequent parity cases verify recovery. These checks require a
real supported checkpoint and CUDA device.

The initial adapter supports dense `qwen35`, single-rank GGML CUDA, Q8_0
projection/embedding matrices and text-only inference. It rejects MoE, MTP,
external draft models, speculation (including N-gram), tensor parallelism, layer splitting, other backends and
other matrix encodings. Small named F32 normalization and GDN parameters remain
resident: at most 1 MiB each and 32 MiB total, with their actual bytes reported.

The shared budget covers the quantized-weight staging buffer, temporary host
output tile, and native streamed CUDA input/weight/output workspace. Resident
small parameters, model activations, attention/GDN state, other native caches,
backend allocator pools, driver/runtime memory and the OS file cache remain
outside that scope. This is a bounded weight-execution check, not a cap on total
RSS or VRAM, multi-GPU weight streaming, or a throughput benchmark. Overall timings
include model loading and validation. `ForwardTimings` separately measures only
successful prefill/decode calls, excluding loading, pressure checks, reset,
comparison and disposal. Compare token tile sizes on the same checkpoint and
hardware with alternating runs; `FileBytesRead` counts logical file reads,
including OS page-cache hits, not physical disk traffic. Keep generated reports in ignored
`artifacts/` or `docs/validation/`.

Additional cases validated on an RTX 3080 Laptop GPU with the same Qwen3.5 0.8B
Q8_0 checkpoint (SHA-256
`0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c`):

- Tight workspace: `--steps 4 --prompt-tokens 32 --cycles 2 --device-bytes 131072`.
- Longer sequence: `--steps 16 --prompt-tokens 256 --cycles 2`.
- Token tile comparison: the default short case with `--token-rows` set to
  `8, 32, 32, 8, 8, 32` in six separate sequential processes and separate reports.

The two correctness cases compare 16 and 64 complete vocabulary rows,
respectively, and require process exit code zero in addition to `Passed: true`.
The default maximum token tile is 32; it shrinks automatically when shared
capacity is too small. The same-host short comparison reduced logical reads by
33.2% and median successful Forward time by 29.5%, chiefly in prefill; this does
not establish a decode, cold-storage or production-throughput improvement.
Exact results, binary identities and unavailable scenarios are recorded in the
[design validation section](../../../docs/design/unified-memory.zh-CN.md#14-验证与本次证据).
