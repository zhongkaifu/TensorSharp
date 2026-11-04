# NInfer CUDA component probe

This executable checks synthetic numerical components. It does not load a model,
compare an engine's answer quality, certify all arithmetic inputs or measure
end-to-end performance. It requires actual CUDA device execution; compilation
alone is not a pass.

```
dotnet NinferCudaKernelProbe.dll HELPER_MODULE_PATH OUTPUT_JSON
```

The native NInfer attention module and original Q35 tensor-core module must also
be discoverable through their normal module lookup. The original Q35 module
must contain actual sm_120a/sm_121a FP4 tensor-core code. An unavailable module or
device fails the probe rather than silently skipping that coverage.

The fixed fixtures cover BF16 rounding, W8 embedding, residual/norm boundaries,
calibrated activation-pack bytes, original BF16 and weight-only NVFP4
projections, SwiGLU, GDN convolution and state movement, raw-A-log gating,
grouped heads, verification nonmutation and state commit, GDN output norm/gate,
and BF16 attention preparation and execution.

The integrated calibrated projection checks T=1,4,7,8,9,33,129 at N=128 and
K=512. It independently quantizes activation bytes on the CPU, checks those
bytes against the device pack, and decodes both quantized operands in FP64 to
form the projection reference. The original alpha is computed in F32 as
`1 / (inputDivisor * weightDivisor)`. Each unrounded output must satisfy the
predeclared bound

```
gamma_(3K+32) * abs(alpha) * sum(abs(decoded products))
    + 2u * abs(reference)
u = 2^-24; gamma_n = n*u / (1-n*u)
```

The operation budget conservatively covers scaled-product evaluation, F32
accumulation and output scaling on these finite fixtures. It is an arithmetic
component bound, not an allowance for model quantization error. Both configured
split limits (1 and 8) are exercised, but K=512 has only two stages and resolves
to **one actual K partition**; these cases do not cover a multi-partition K
reduction. Output guard regions must remain unchanged. The actual original
module image hash is recorded alongside each geometry.

The GDN output check uses a separate FP64 ideal RMSNorm/plain-weight/SiLU
formula, then admits one BF16 rounding envelope (half a BF16 ULP at the reference
magnitude) plus `gamma_1024 * abs(reference)` for F32 intermediate arithmetic.
The output must also have exactly BF16-representable bits. The gate and plain
norm vary across channels and heads; no unit offset is added to the GDN norm.
The bound is fixed in the source before device evaluation. It does not assert
that this adapter uses the same summation order as Ignis.

The attention fixtures and their separately declared bounds are described in
`AttentionChecks.cs`. All generated JSON and logs belong under ignored
`docs/validation/` or `artifacts/`.
