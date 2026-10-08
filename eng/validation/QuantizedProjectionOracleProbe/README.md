# Dense low-bit projection diagnosis

This probe keeps the original GGUF tensor's complete K, M and quantization in
`GgmlBasicOps.AddmmQuant`. It does not compact the output matrix or add a streaming
implementation for IQ2/IQ3. It reads one original tensor at a time, selects evenly
spaced output rows, checks independent managed dequantization bitwise against
native dequantization, then evaluates those rows with scalar FP64 dot products.
All native output elements must be finite; oracle comparisons cover sampled rows
only. Inputs are deterministic synthetic activations, not captured model states.

Build with the usual skip-native flags and place a separately validated native
library beside the executable. Never overwrite a running probe's directory.
Upstream ggml must remain unchanged; preserve its revision and build evidence.

```powershell
dotnet eng/validation/QuantizedProjectionOracleProbe/bin/Release/net10.0/QuantizedProjectionOracleProbe.dll --self-test true
python eng/validation/run-bounded-probe.py --timeout 180 --output artifacts/iq-oracle/cuda-process -- dotnet eng/validation/QuantizedProjectionOracleProbe/bin/Release/net10.0/QuantizedProjectionOracleProbe.dll --model artifacts/unified-memory-adaptive/models/gemma-4-12b-it-UD-IQ2_M.gguf --output artifacts/iq-oracle/cuda --backend cuda --weights blk.0.ffn_gate.weight,blk.1.ffn_gate.weight,blk.1.attn_q.weight --tokens 1,8,9,19,38 --rows 128
```

Run CPU and CUDA in separate processes and fresh output directories. The default
128 MiB limit is per original weight tensor; host arrays, native graph workspaces,
input/output and backend caches are not a shared model memory cap. A rejected
tensor or unavailable backend is not validated. Failed native cleanup retains
source pins until process exit, rather than releasing addresses still borrowed.

Reports include both the original F32-activation FP64 result and scalar models
of CPU Q8_K, CUDA MMVQ Q8_1 and MMQ D4 activation quantization. These models follow
unchanged ggml `ffa4e8b80930029a35991f94e7c8a93cd67730ab`; they do not capture
device buffers or prove actual dispatch. Do not select the closest hypothesis
after the fact and call it an independent kernel qualification. Subnormal or
overflow activation fixtures require dedicated device quantizer validation and
are rejected here. Self-checks cover zero blocks, signed rounding ties, half
versus float scales, token/row layout and zero-reference error reporting.

The first local Gemma IQ2 run covers three real tensors, IQ2_S/IQ3_XXS, original
K=3840 and M=15360/4096, N=1/8/9/19/38, 128 sampled rows each. All sampled weight
decodes agree bitwise. CPU results are close to the CPU quantized-activation
oracle (relative L2 around 1e-7); CUDA N>=9 results are close to the MMQ D4 model
(around 2e-7), while N<=8 residual error against the Q8_1 model is around 1e-5.
Original F32 inputs give larger differences because ordinary quantized execution
also quantizes activations. These observations narrow the investigation; they
do not establish full-model correctness or fix the reported FF7 repetition.
Exact per-case metrics, original byte hashes, native/managed identities and
process exits are in ignored `artifacts/unified-memory-adaptive/gemma-iq2-projections-v1/`.

Exit zero means the requested diagnosis, finite checks, sampled decoder agreement
and cleanup completed. There is deliberately no universal numeric or language
quality pass claimed by this tool.
