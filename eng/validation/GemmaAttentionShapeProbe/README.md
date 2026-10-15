# Gemma 26B attention shape diagnostic

This independent, weight-free CUDA app compares the same synthetic post-RoPE Q,
F16 K/V, and causal keys across query counts 1, 2, 4, and 8 at prefix127. It uses
the 26B-A4B geometry: 16 query heads, local 8 KV heads/dim256/cache1024 and global
2 KV heads/dim512/cache2048. Two deterministic profiles cover gentle attention
and unit RMS inputs; they are not captured model activations.

Each case has four phases: batched attention with the native key span, batched
attention with 256 physical keys, sequential Q1 with its native key span, and
sequential Q1 with 256 physical keys. Every phase is checked against a scalar
FP64 softmax oracle using the exact stored F32 Q and F16-rounded K/V. Output and
reference arrays are saved in full with SHA256, alongside per-row finite/error
metrics. Finite nonzero future/padding keys test causal exclusion. The dim512 API
always pads to256; the dim256 API leaves the native span unpadded.

Build only this app against already built product dependencies when preserving a
frozen validation build:

```powershell
dotnet build eng/validation/GemmaAttentionShapeProbe -c Release -p:BuildProjectReferences=false -p:TensorSharpSkipGgmlNative=true
dotnet eng/validation/GemmaAttentionShapeProbe/bin/Release/net10.0/GemmaAttentionShapeProbe.dll --self-test
python eng/validation/run-bounded-probe.py --output artifacts/gemma-attention-owned --timeout 180 --telemetry --managed-directory eng/validation/GemmaAttentionShapeProbe/bin/Release/net10.0 --expected-native-sha256 ACTUAL_SHA256 -- dotnet eng/validation/GemmaAttentionShapeProbe/bin/Release/net10.0/GemmaAttentionShapeProbe.dll --output artifacts/gemma-attention-data --expected-native-sha256 ACTUAL_SHA256 --repo-root .
```

Run from the repository root with the existing native loader configuration. The
app requires and checks the **actual loaded** GgmlOps hash, records managed/source
files before/after, and fails an identity change. The wrapper records the owned
process and memory telemetry. `--self-test` performs CPU-only oracle/layout/
finite-handling checks without initializing a native backend. Output directories
must be new. Exit0 means finite completed execution and stable identities;
numeric threshold diagnostics are reported separately, never as model quality.
The alert limits (max absolute 0.002 and relative L2 0.001) follow existing flash
tests, and are not an acceptance bound for whole-model greedy token parity.

The public `StreamingFlashAttention` API allocates contiguous device Q/K/V. It
does **not** reproduce the fused trunk's permuted Q views, cache head strides,
captured decode graph, cache writes, norm/RoPE, quantized matmuls, router, or
experts. Therefore this can falsify a proposed attention-only cause on these
inputs, or identify an isolated attention error, but cannot itself establish the
root cause of the Gemma MTP token mismatch. GPU accumulator precision is requested
as F32; the backend may still use narrower intermediates. The saved per-call
times include graph setup/upload/download and have no balanced warmup; do not
report them as model throughput. GGML upstream sources remain unchanged.
