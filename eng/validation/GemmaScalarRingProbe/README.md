# Gemma MoE two-row scalar verification and ring diagnostic

This CUDA diagnostic loads a Gemma4 MoE checkpoint with F16 KV, obtains its actual
sliding window W, and tests prefixes W−1 and W+1. Each prefix has a rejected second
row and an accepted second row. Cache capacity is rounded up to a power of two
(2048 for W=1024), since an arbitrary cache stride can disable the native flash
kernel. The actual capacity is recorded. Both arms use the same `ForwardRefill` prompt
entry. Two successive `SpecForward(1)` calls provide the reference logits and
normalized hidden states; `SpecForward(2, allLogitsRows:true)` must match every
finite F32 bit. Acceptance uses `SpecOnVerifyAccepted`, with the usual cache
rewind for rejection, followed by four ordinary `Forward(1)` continuation rows.
Every continuation row must exactly match a scalar reference with the same
committed prefix. Complete arrays and hashes are retained, including failures.

```powershell
dotnet build eng/validation/GemmaScalarRingProbe -c Release -p:BuildProjectReferences=false -p:TensorSharpSkipGgmlNative=true
dotnet eng/validation/GemmaScalarRingProbe/bin/Release/net10.0/GemmaScalarRingProbe.dll --self-test
$env:TS_N_CPU_MOE = '8'
$env:TS_CPU_MOE_THREADS = '8'
$env:TS_GGML_CPU_THREADS = '8'
$env:OMP_NUM_THREADS = '8'
python eng/validation/run-bounded-probe.py --output artifacts/gemma-ring-owned --timeout 1200 --telemetry --managed-directory eng/validation/GemmaScalarRingProbe/bin/Release/net10.0 --expected-native-sha256 NATIVE_SHA256 -- dotnet eng/validation/GemmaScalarRingProbe/bin/Release/net10.0/GemmaScalarRingProbe.dll --model MODEL.gguf --output artifacts/gemma-ring-data --expected-native-sha256 NATIVE_SHA256 --expected-models-sha256 MODELS_SHA256
```

Use fresh output paths and real expected digests. For a frozen qualification,
build a separate app against the explicitly pinned managed references, rather
than rebuilding dependencies. The app verifies its loaded Models assembly and
actual loaded GgmlOps module. Top-level deployment files and checkpoint
size/mtime are checked before/after; these checks are not a new checkpoint hash
or an enumeration of every loaded managed module. The wrapper owns the process,
enforces the deadline, and samples memory; sampling overhead is unquantified.

Exit0 requires all four cases, all strict comparisons, cache positions, cleanup,
and identity continuity to pass. Exit3 includes numeric/protocol failures. The
CPU self-test only checks comparison logic; it does not qualify model or GPU
execution. This is a raw target/KV diagnostic, without the draft head, scheduler,
sampling, EOS/semantic scoring, or a speedup claim. Its matched `ForwardRefill`
prefill also differs from the scheduler's ordinary `Forward` entry; qualify the
actual MTP engine separately. No tolerance is substituted for byte parity.
