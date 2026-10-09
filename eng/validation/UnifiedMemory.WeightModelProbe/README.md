# Real file-backed model execution

This .NET 10 CUDA probe compares an ordinary resident model against its explicit
`WeightStreamingOptions` adapter. It accepts dense Qwen35 Q8_0 and dense Gemma4
Q8_0 with an optional F16 `per_layer_model_proj.weight`. Original GGUF regions
are read through a bounded host buffer; embeddings read only requested rows.
Gemma's fused QKV and gate/up matrices are logical file-source concatenations,
without a full host copy or mmap. No temporary weight pointer enters cached
model graphs.

`--host-cache-bytes BYTES` optionally retains original source ranges in pageable
RAM (default zero). The shared host budget includes aligned cache payload as well
as staging, with 4 MiB held available for output workspace. Cache admission can
therefore stop below the requested ceiling. `Usage` reports current/peak cache
payload, hits, hit bytes and evictions separately from staging. A hit copies to
existing staging; no cached pointer is retained by a native graph. Disposal must
refund both cache and staging. Repeated-consumption checks add source bytes and
cache-hit bytes; file bytes alone no longer describe every consumed range. The
managed index and OS page cache are outside this payload quota. Use the adaptive
probe's balanced comparison for isolated timing; this probe remains a numerical
and failure-recovery validation with an in-process resident reference.

`--device-cache-bytes BYTES` optionally retains complete immutable device weights
and their input/output/scratch arenas (default zero). All retained payload shares
the existing device budget, with a 4 MiB admission reserve. Qwen FullPrecision
reuses these arenas across prefill and decode; Gemma ResidentCuda currently
retains only N=1 decode projections. Larger Gemma batches keep their original
complete-matrix arithmetic and may evict retained arenas before allocating a
workspace. Host upload/download staging remains bounded. FullPrecision promotion
removes duplicate RAM ranges for the same source; GPU eviction causes a later
file/RAM reload. Normal KV reset keeps valid weights; idle trim and disposal
release them. Failed CUDA cleanup retains both ownership and budget until retry.
`PeakDeviceOwnedBytes` includes retained arenas plus temporary workspace;
`WeightUploadBytes` records original weight H2D bytes, and `DeviceCacheHitBytes`
records avoided uploads. Logical consumption adds file, RAM-hit and device-hit
bytes. These are payload counters, not physical SSD traffic or total VRAM/RSS.
If retention is being validated, require positive actual cache payload and hits;
merely setting the option does not establish coverage.

Qwen uses F32 activation arithmetic in synchronous output-row tiles. Gemma
preserves the pinned resident CUDA arithmetic: Q8 matrix batches above eight
tokens and F16 batches above sixteen temporarily stage one complete logical
projection on the GPU. Its full weights, input, output and arithmetic scratch
are charged before allocation and released after projection. Original M/N
determine MMQ stream-K and cuBLAS reduction order; shrinking them introduced
small differences that later quantization amplified. Smaller Gemma batches use
row tiles with resident MMVQ/MMVF/MMF arithmetic. These policies support NVIDIA
Ampere, Ada and Hopper; other hardware and arithmetic overrides may be refused.

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
Qwen GDN chunked graph cache to become nonempty during execution and both GDN
graph caches to return to zero after disposal. Gemma does not use these caches:
its zero counters do not establish populated-cache lifecycle coverage. Missing
test hooks are reported as unavailable. The process explicitly shuts down its
owned backend before exit.

The report verifies actual file reads, repeated operator tiles, embedding row
reads, host/device payload peaks below their configured budgets, no full-weight
preloads, and zero owned charges after disposal. An external reservation in the
same host pool must refuse construction before staging allocation; releasing it
must allow a subsequent model to run. Model, assembly and native hashes are
recorded. A second external reservation exhausts GPU credit before a forward;
after refusal, the model must reject another forward until `ResetKVCache`
succeeds. The subsequent parity cases verify recovery. These checks require a
real supported checkpoint and CUDA device.

Both adapters support single-rank GGML CUDA and sequential text inference.
Gemma additionally requires F16 or F32 KV storage. They reject MoE, MTP,
external draft models, speculation (including N-gram), tensor parallelism, layer splitting, other backends and
unsupported matrix encodings. Small named F32 normalization, PLE and GDN parameters remain
resident: at most 1 MiB each and 32 MiB total, with their actual bytes reported.

The shared budget covers the file-read buffer, temporary host output tile, and
native streamed CUDA input/weight/output workspace, including quantization,
padding, reduction fixup and explicit cuBLAS workspace. The current full-shape
F16 implementation includes two temporary device copies of its weight matrix;
both are charged. Resident
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

Gemma E4B example using an existing local checkpoint (8.03 GB, Q8_0/F16):

```powershell
dotnet eng/validation/UnifiedMemory.WeightModelProbe/bin/Release/net10.0/UnifiedMemory.WeightModelProbe.dll `
  --model C:/Works/models/gemma-4-E4B-it-uncensored-Q8_0.gguf `
  --json artifacts/unified-memory/gemma-e4b.json `
  --steps 16 --prompt-tokens 640 --cycles 2 `
  --tile-bytes 16777216 --token-rows 32 `
  --host-bytes 33554432 --device-bytes 268435456
```

The device minimum depends on the original matrix and full prompt shape. A
small `--token-rows` bounds host transfers; it cannot eliminate Gemma's full
device workspace. An insufficient shared budget refuses execution without
falling back to a different arithmetic policy. `CompleteMatrixProjections`
counts these projections; `LinearTiles` counts uploaded weight tiles (or
projected tiles in the row-tiled path).

Use `--prefill refill --prefill-chunk 256` to exercise public
`ModelBase.ForwardRefill` chunking on both arms. The default is `Forward`;
model refill chunks and weight-transfer token tiles are independent. With a
minimum prompt longer than 512, Gemma exercises sliding-window eviction and KV
sharing. Only completed strict-gate comparisons count as validation.

On the RTX 3080 Laptop, the E4B checkpoint above passed both Forward and
256-token ForwardRefill with two 645-token prompts, 16 vocabulary rows each,
and two load/dispose cycles: 128 rows / 33,554,432 logits were byte-exact.
These forced-streaming runs achieved about 88 / 48 prefill tokens/s respectively
and 0.48 decode tokens/s, substantially below resident execution. The budget
covers weight payload, not the whole process or all GPU memory.

To compare file tiles, keep `--host-bytes 134217728 --device-bytes 268435456`,
`--steps 4 --prompt-tokens 32 --cycles 1`, and run `--tile-bytes` in order
`16777216, 67108864, 67108864, 16777216`, with a distinct JSON path per process.
The measured 36-token prompts were byte-exact for all four runs. A 64 MiB tile
reduced upload tile counts but increased host payload; the mean Forward time
was 3.54% lower, with overlapping timing ranges and only two runs per setting.
This does not establish a stable speedup or justify changing the global default.
See the design document for separate timing, memory, binary and failure records.

For arithmetic localization, set `TS_GEMMA4_TENSOR_DUMP` to an ignored output
directory and optionally `TS_GEMMA4_TENSOR_DUMP_LAYERS` (default 6). Compare
matching F32 snapshots using `eng/validation/compare-gemma-streaming-tensors.py`.
The optional snapshots pin native fusion-boundary tensors and synchronize
streamed intermediates. They are diagnostic runs, not performance measurements.
The JSON records diagnostic/fusion overrides and each returned Forward call,
including a call whose subsequent strict comparison failed.

Use `--read-ahead false` as the sequential control for the optional second host
tile. With `true`, the executor reserves that tile only when the shared host
budget has enough space; a tight or shared RAM/device pool keeps one buffer.
`ReadAheadOperations` counts reads launched before consuming the previous tile.
Pending reads drain before disposal/refund. This overlaps file reads with the
current operation; CUDA transfers remain synchronous.

`--workspace-cache-bytes BYTES` (default zero) retains healthy idle row-tiled
CUDA sessions in the same device budget. It replaces both input and weight
bytes on reuse; compatible capacity is not a promise of cached weight identity.
The pool has at most 16 entries and yields to workspace pressure before the
weight cache. ResidentCuda reuses only identical original logical shapes;
its complete-matrix multi-token path is excluded. `DeviceWorkspaceReuses`
counts borrowed sessions, `DeviceSessionCreations` counts fresh allocations,
and `DeviceWorkspaceCacheBytes` counts idle payload. `PeakDeviceOwnedBytes`
includes retained weights, idle sessions and active workspace. These counters
exclude driver/stream metadata and other model allocations. Correctness, reset
and zero-owner checks still apply; successful execution alone is not a timing
or language-quality result. For an isolated ABBA comparison use the adaptive
probe's `compare-workspace-cache.py` with fixed weight-cache settings.

The 2026-10-08 E4B short check with 16 MiB tiles, 128 MiB host and 256 MiB device
quota executed 1,704 read-ahead operations and matched all eight full logit rows
byte-for-byte. Peak charged host/device payload was 34,343,936 / 117,170,176 bytes,
and logical file reads were 39,681,038,976 bytes. Concurrent native compilation
makes its timings unsuitable for a speedup claim. Evidence remains ignored at
`artifacts/unified-memory-adaptive/e4b-read-ahead-v5.json`.
