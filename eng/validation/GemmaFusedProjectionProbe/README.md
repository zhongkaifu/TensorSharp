# Gemma fused projection diagnostic

This standalone tool isolates the effect of concatenating actual GGUF QKV or
gate/up output rows. It compares three paths using the same exact F32 input:

1. Ordinary `AddmmQuant` with original Q8_0 rows concatenated in resident order.
2. Ordinary `AddmmQuant` for each original matrix separately.
3. A bounded `ResidentCuda` streaming session for each original matrix separately.

The report compares every output value per original matrix. The first comparison
isolates fused-row geometry; the third isolates streaming arithmetic. Original
logical N is preserved across streaming token chunks. No input or output rows
are padded, truncated or dropped by the tool. QKV duplicates K as V only when the
original GGUF lacks V, matching Gemma's resident concatenation. Shared-Q-only
layers, mixed types and sidecar-scaled weights are rejected.

```powershell
dotnet build eng/validation/GemmaFusedProjectionProbe/GemmaFusedProjectionProbe.csproj -c Release -m:1 -p:BuildInParallel=false -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
# Copy the validated GgmlOps.dll into this tool's bin/Release/net10.0 directory.
dotnet eng/validation/GemmaFusedProjectionProbe/bin/Release/net10.0/GemmaFusedProjectionProbe.dll --model C:/Works/models/gemma-4-E4B-it-uncensored-Q8_0.gguf --layer 0 --group both --tokens 1,8,9,36 --token-rows 32 --json artifacts/gemma-fused-projections/synthetic.json

# Replay the attention input RMSNorm output supplied to layer 0's QKV.
dotnet eng/validation/GemmaFusedProjectionProbe/bin/Release/net10.0/GemmaFusedProjectionProbe.dll --model C:/Works/models/gemma-4-E4B-it-uncensored-Q8_0.gguf --layer 0 --group qkv --tokens 36 --input-file artifacts/gemma-stage-dumps/attn-norm.f32 --json artifacts/gemma-fused-projections/captured-qkv.json

# Gate/up requires its own FFN input RMSNorm output, not the attention input.
dotnet eng/validation/GemmaFusedProjectionProbe/bin/Release/net10.0/GemmaFusedProjectionProbe.dll --model C:/Works/models/gemma-4-E4B-it-uncensored-Q8_0.gguf --layer 0 --group gate-up --tokens 36 --input-file artifacts/gemma-stage-dumps/ffn-norm.f32 --json artifacts/gemma-fused-projections/captured-gate-up.json
```

`--input-file` accepts exactly N×K packed little-endian Float32 values and requires
one group and one N. Record the capture's model, layer, stage and native-library
provenance; the report hashes the input and loaded native/probe binaries.
`--tile-bytes` defaults to 1 MiB. `--dump-dir` optionally saves only selected
projection outputs, never a GGUF copy. Model identity uses path, size and mtime;
the multi-gigabyte model is not rehashed by this small diagnostic.

Each comparison reports the existing per-token gates: relative L2 ≤ 0.001,
cosine ≥ 0.999999, identical top-1. Projection top-1 is not a model token. Exit 0
means the diagnostic completed with finite values and released reservations;
inspect `AllStrictGatesPassed` and each `StrictGate` for numerical outcomes.
Exit 2 denotes an execution or cleanup error. No complete-model or throughput
qualification is claimed. One concatenated group is held in ordinary host/GPU
reference storage; only the streaming CUDA workspace uses `MemoryBudget`.

Generated evidence belongs in ignored `artifacts/` or `docs/validation/`. Record
the unchanged dependency revision through `TS_VALIDATION_GGML_REVISION`.
