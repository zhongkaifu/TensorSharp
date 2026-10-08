# Real-model host snapshot validation

This .NET 10 probe compares the public inference engine with managed host KV
snapshots against the same per-sequence execution with bounded RAM/file-backed
snapshots. It requires a supported GGUF checkpoint and the matching native GGML
library. It is a correctness and lifecycle check, not an end-to-end weight
streaming benchmark or a cap on total process/device memory.

Build the native library against the pinned, unchanged ggml checkout. On the
two-A40 validation VM, direct peer transfers corrupt data, so multi-device model
runs must explicitly select ggml's existing staged transport:

```sh
export TENSORSHARP_GGML_GIT_REF=ffa4e8b80930029a35991f94e7c8a93cd67730ab
export PATH=/usr/local/cuda-12.8/bin:$PATH
bash TensorSharp.GGML.Native/build-linux.sh --cuda --no-vulkan --tests \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_NO_PEER_COPY=ON

dotnet build eng/validation/UnifiedMemory.ModelProbe -c Release --no-incremental \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
export LD_LIBRARY_PATH="$PWD/TensorSharp.GGML.Native/build:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0
export TS_VALIDATION_DEVICE=NVIDIA_A40_GPU0
export TS_VALIDATION_GGML_REVISION=$TENSORSHARP_GGML_GIT_REF
dotnet eng/validation/UnifiedMemory.ModelProbe/bin/Release/net10.0/UnifiedMemory.ModelProbe.dll \
  --model /workspace/models/gemma-4-E2B-it-Q4_K_M.gguf \
  --widths 1,2,4,8,16 --steps 8 --prompt-tokens 64 \
  --json artifacts/unified-memory/gemma-snapshots.json
```

Check the reported native library path and SHA-256, especially if another copy
exists beside the executable. Rebuild without incremental compilation after
copying source archives whose timestamps may predate existing assemblies.
`--backend ggml_cpu` can select the CPU backend; CUDA is the default.

Each case uses a complete rendered chat template and distinct counting prompts.
`--prompt-tokens` is a minimum, not a truncation length; the report contains the
actual tokens. Both arms disable batched execution and prefix reuse, warm the
same model, use identical prompts and greedy sampling, and alternate run order.
The candidate budget holds one resident snapshot page plus capture/transfer
scratch. Concurrency greater than one must actually spill and reload. The
single-request route is a control and need not capture snapshots.

Passing requires identical generated tokens, termination reasons and request
status, the requested output length without early EOS, and zero charged bytes
after physical cleanup. A second test captures two real model histories, forces
their pages through file storage, restores them, and compares every logit at
every teacher-forced step. The numerical gate is
`abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)` with finite logits and
identical argmax. Failure to clean up safely stops subsequent arms.

Use `--prompt-tokens 256 --steps 16 --widths 2` for a longer bounded-window case.
These cases remain within the model's declared restorable window; they do not
prove long-context attention beyond that window. A model which does not declare
safe cross-sequence snapshots must be tested as an explicit rejection:

```sh
dotnet eng/validation/UnifiedMemory.ModelProbe/bin/Release/net10.0/UnifiedMemory.ModelProbe.dll \
  --model /workspace/models/Qwen3.5-0.8B-Q8_0.gguf \
  --expect-unsupported true --json artifacts/unified-memory/qwen-snapshot-refusal.json
```

Successful rejection is a capability check, never positive snapshot coverage for
that model. JSON records model/assembly/native hashes, capability declarations,
prompts, tokens, spill counts, budgets, single-sample timings and native per-rank
cache payload counters when available. Native counters exclude graph/KV arenas,
backend pools and driver overhead. Timings include storage overhead and are not
repeat-sampled throughput or latency percentiles.

Validated checkpoint identities:

| Checkpoint | Repository revision | SHA-256 |
| --- | --- | --- |
| [Gemma 4 E2B IT Q4_K_M](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/blob/ecee195e9a5bf3846817d70a1a00cdc057926e18/gemma-4-E2B-it-Q4_K_M.gguf) | `ecee195e9a5bf3846817d70a1a00cdc057926e18` | `f3504b387ee0962b2b041cf3691b1520118822642d67c5294f85ea62c68614b3` |
| [Qwen 3.5 0.8B Q8_0](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/blob/cb02287e2172232d38d9eb470061351961ce1ba6/Qwen3.5-0.8B-Q8_0.gguf) | `cb02287e2172232d38d9eb470061351961ce1ba6` | `0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c` |

Keep generated logs, JSON and spill evidence in ignored `artifacts/` or
`docs/validation/`. The separate [CUDA residency probe](../UnifiedMemory.CudaProbe/README.md)
checks device allocation, events and transport. `eng/ForcedLogitProbe` separately
compares single-device and tensor-parallel model execution; host snapshot parity
alone does not prove tensor-parallel correctness.
