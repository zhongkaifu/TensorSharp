# GPT-OSS single-expert numerical diagnostic

This tool reads one real expert's gate/up/down tensor ranges and their F32 bias
rows, plus the next expert for alternating temporary-owner validation, preserving
K and M. It does not load the full model. Actual GGUF types are
checked: Q8_0 (8) and MXFP4 (39) are supported. A filename containing `Q8_0` is
not evidence that experts are Q8; the local GPT checkpoint has MXFP4 experts.

```powershell
dotnet build eng/validation/GptOssExpertProjectionProbe -c Release -p:TensorSharpSkipGgmlNative=true
dotnet eng/validation/GptOssExpertProjectionProbe/bin/Release/net10.0/GptOssExpertProjectionProbe.dll --model C:/Works/models/gpt-oss-20b-Q8_0.gguf --output artifacts/gpt-expert/layer0-expert0 --backend cuda --layer 0 --expert 0 --tokens 1,8,9,64,96,100,112,128,129,256
```

Use the existing qualified native binary; do not rebuild or replace upstream
ggml. Run GPU work only in the coordinated validation window. `--backend cpu`
uses GGML CPU with the same shapes. N is the number of **tokens routed to this
expert**, not full-model prompt length. Inputs are deterministic synthetic F32
values, explicitly not captured model activations. No semantic or throughput
claim follows from this diagnostic.

Raw projection comparisons preserve fused M=2*FF versus separate M=FF. The tool
then compares production bias addition, the same production CPU SwiGLU, down
projections, and actual private `ExpertFFN`/`ExpertFFNSplit` calls through an
isolated reflection fixture. This checks whether a discrepancy first appears
before activation and whether the production contiguous-copy step changes data.
On sm86, M=5760 and M=2880 choose different MMQ row-tail configurations; the
96/100/112-token cases probe shape-dependent tile selection. No source or dispatch
policy is changed. The default one-byte lazy-copy cap matches the bounded
full-model diagnostic. `--device-cache-bytes 67108864` instead permits lazy copies
so the alternating-expert temporary arm also exercises cache retirement. There
are no explicit preloads. Each case compares the original fused projection with
production `ExpertFFNTemporary`, runs the next expert, and repeats the first
expert to detect stale pointer/cache reuse. The temporary owner must be absent
after every successful call.

The local independent decoder follows the pinned upstream MXFP4 E8M0 half-scale
and doubled E2M1 table (including exponent 0/1 subnormals), or Q8_0 half scales
and signed bytes. Every decoded element is checked **bitwise** against native
`ggml to_float`. FP64 scalar dot products check 32 output rows on the first/last
token. The complete expert oracle computes all intermediate gate/up rows for
those tokens, exact double exp and source biases before sampling 32 down rows.
Production's approximate exp, activation quantization, and reduction policies
can differ; errors are reported without an invented pass tolerance.

Exit 0 means execution, independent decode checks, finite results and cleanup
completed. It does **not** mean fused/split equality or numerical qualification:
inspect `all_fused_split_exact`, `all_fused_temporary_exact`, each stage's full-array bitwise comparison, and
both paths' FP64 errors. Binary full expert outputs and sampled FP64 outputs,
actual selected weight hashes/types, and loaded native hashes are retained in
the fresh ignored output directory. No entire checkpoint hash is computed.
