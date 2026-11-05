# Native-layout BF16 x FP4 AOT component gate

This independent synthetic probe uses only the CUDA driver at runtime. It tests
the original raw adjacent-K FP4 codes and compact M128x4 E4M3 scales, actual BF16
input, F32 MMA accumulation and an unrounded F32 output after global alpha.
It does not compare this arithmetic to the legacy scalar schedule or establish
unchanged whole-model quality.

The fixed first prototype is full N=34816, K=5120 at T=1 and T=8. Four families
cover signed zero, an exact binary lattice, a non-BF16 F32 output witness, and
heterogeneous finite payloads. Twelve nonzero rows cross nibble, warp, scale,
128-row tile, and final-tile boundaries; every output row, including zero rows,
is checked. Raw scale fixtures are serialized independently of the FP64 oracle's
offset formula. All 2,032 positive-finite E4M3/FP4 products are proved exactly
BF16-representable on the CPU before GPU execution.

Exact cases require identical F32 bits. The broad case uses the fixed normal
finite-input envelope `(1+u)*abs(alpha)*gamma_(K+64)*sum(abs(products)) +
u*abs(ideal)`, `u=2^-24`. BF16 input products are exact F32; the envelope allows
arbitrary F32 accumulation association and the final F32 alpha multiply. It
also includes the separate FP64 accumulation/alpha oracle envelope at `u64=2^-53`.
It includes no BF16-output rounding allowance and is not fitted to GPU data.

Eight cases must complete two module lifetimes and three poisoned-output passes,
with full input immutability, guards, repeated and cross-lifetime output bits.
The actual CUBIN image and five-pointer parameter ABI must match the AOT manifest.
Full inputs, decoded FP64 ideals/bounds and guarded outputs are retained in the
fresh ignored evidence directory. Only `completed-result.json` (written once
with its SHA sidecar) marks success; `result.json` is progress/failure evidence.

```sh
dotnet build eng/validation/NinferFlashInferA16Probe/NinferFlashInferA16Probe.csproj
dotnet eng/validation/NinferFlashInferA16Probe/bin/Debug/net10.0/NinferFlashInferA16Probe.dll --self-test
dotnet eng/validation/NinferFlashInferA16Probe/bin/Debug/net10.0/NinferFlashInferA16Probe.dll --execute --manifest /absolute/aot/module.json --out /fresh/artifacts/output
```

GPU execution, real-model weights, projection-to-SwiGLU composition, quality and
performance are separate pending gates. No serving hook or default is enabled.
