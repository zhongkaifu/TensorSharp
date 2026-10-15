# Gemma weight projection diagnostic

Compares the ordinary resident `GgmlBasicOps.AddmmQuant` API with the bounded
Q8_0/F16 streaming session using **the same original GGUF bytes and F32 input**.
It loads one selected tensor at a time, without constructing the model or loading
other tensors. Defaults cover the F16 PLE projection, first Q/K/V projections and
tied LM head at their full shapes, with token counts 1, 8, 9 and 32.

```powershell
dotnet build eng/validation/GemmaWeightProjectionProbe/GemmaWeightProjectionProbe.csproj -c Release -m:1 -p:BuildInParallel=false -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
# Copy the already validated native library to the output directory as appropriate.
dotnet eng/validation/GemmaWeightProjectionProbe/bin/Release/net10.0/GemmaWeightProjectionProbe.dll --model C:/Works/models/gemma-4-E4B-it-uncensored-Q8_0.gguf --json artifacts/gemma-projections/synthetic.json
dotnet eng/validation/GemmaWeightProjectionProbe/bin/Release/net10.0/GemmaWeightProjectionProbe.dll --model C:/Works/models/gemma-4-E4B-it-uncensored-Q8_0.gguf --weights per_layer_model_proj.weight --embedding-token 2 --json artifacts/gemma-projections/actual-ple-input.json

# Compare original resident arithmetic with a 36-token input split into 32 + 4.
dotnet eng/validation/GemmaWeightProjectionProbe/bin/Release/net10.0/GemmaWeightProjectionProbe.dll --model C:/Works/models/gemma-4-E4B-it-uncensored-Q8_0.gguf --arithmetic resident --tokens 1,8,9,16,32,36,65 --token-rows 32 --json artifacts/gemma-projections/resident-arithmetic.json
```

Default inputs are explicitly synthetic. The second invocation reads one actual
token embedding row, decodes and scales it by `sqrt(K)`, and repeats that true
PLE input across the requested token counts. `--input-file` accepts exactly
`max(tokens) * K` packed little-endian F32 values for a single `--weights` name;
record the capture's model/stage provenance separately. Each smaller batch uses
the same prefix, so the report can also compare a fixed first column across the
MMVQ/MMQ boundary. No model precision policy is registered or changed.

`--arithmetic full` (default) keeps F32 activations in the streaming arm.
`--arithmetic resident` uses the bounded resident-compatible CUDA dispatch,
passing the original token/output dimensions before tiling. The resident arm
always uses the ordinary unmodified API. `--token-rows` caps the streaming input
workspace; larger `--tokens` values exercise short tail chunks without changing
the original logical batch's arithmetic selection. Hardware and shape limits
are enforced by the selected session mode; an unsupported case is an execution
error, not a successful parity result.

The report compares every output value, and separately evaluates 129 evenly
spaced output rows per token against a double-precision dot product of decoded
original weights and the same F32 input. It reports the existing numerical gates
(per-token relative L2 <= 0.001, cosine >= 0.999999 and identical top-1), without
relaxing them. Projection top-1 is a diagnostic statistic, not a model token.
Optional `--dump-dir` writes full resident/streamed `.f32` output tensors.

Exit 0 means the diagnostic completed with finite results and clean disposal.
It **does not mean parity passed**: inspect every `StrictGate`. Exit 2 indicates
an execution/oracle/cleanup error. This is not a complete model or throughput
qualification. The streaming CUDA payload is charged before allocation and must
refund on release; full host weights, inputs/output, ordinary resident caches,
GGML scratch and context overhead are intentionally outside that small diagnostic
reservation. No process-memory cap or bounded host-weight residency is claimed.
Generated reports and tensor dumps belong in ignored `artifacts/` or
`docs/validation/`. Record the unchanged ggml revision with
`TS_VALIDATION_GGML_REVISION`; the tool hashes loaded native/probe binaries, not the
multi-gigabyte model.
