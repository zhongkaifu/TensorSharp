# Separate original-schedule NInfer A16 component candidate

This diagnostic explicitly loads the separate `CudaNinferA16Module`. It does not
select a serving option, load model weights or change the frozen NInfer body
module. No GPU work occurs with no arguments or with `--self-test`.

```sh
dotnet build eng/validation/NinferA16KernelProbe/NinferA16KernelProbe.csproj -c Release
dotnet NinferA16KernelProbe.dll --self-test
dotnet NinferA16KernelProbe.dll --execute \
  --module /absolute/artifacts/candidate/tensorsharp_q35_ninfer_a16.fatbin \
  --out /absolute/artifacts/a16-original-schedule-component
```

Use a fresh output directory under ignored `artifacts/` or `docs/validation/`
with canonical absolute paths and no symlinked output parents. The operator must
hold the exclusive GPU lease, pin the device/toolchain/source/runtime and
unchanged ggml revision, and build the candidate module for the actual GPU.
Unavailable devices/modules, load failures, incomplete records and skipped
scenarios are not passing validation.

The 90 unique fixtures exercise all nine K/T entries:

- K=5120, 6144, 17408; T=1, 2, 3; N=1, 17, 129 for row/CTA/128-row tails.
- Three patterns include every signed E2M1 code, positive-finite E4M3 scale
  encodings, non-power-of-two inverse divisors, BF16 activations over a wide
  exponent range, positive/negative zero and cancellation.
- Nine additional real-K analytical witnesses place `[2^24,1,-2^24,1]`
  repetitions in the first512 active columns and zero the remaining input.
  With unit weights/scales/inverse divisor, the original four-chain T1 schedule
  returns128 and the original single-chain T2/T3 schedule returns32. The exact
  mathematical dot is256. This intentionally distinguishes the original
  rounding schedules rather than treating their differences as equivalent.

Two separately loaded/disposed module lifetimes run the same90 fixtures. This
creates180 lifecycle records, not180 independent fixture types. Each record uses
three output initialization passes, A/changed-NaN-poison/A. Every active output
must exactly match the unrounded F32 original-schedule CPU oracle and the first
device output's bits. Output row padding and prefix/suffix guards must retain
their original bits; input activations, stride padding, weight codes/scales and
their guards are read back in full and must be unchanged.

The CPU oracle reads original adjacent-K codes and original m128x4 scales. The
probe independently materializes the five-axis original scale layout and the
candidate's TS low8/high8 planes from common logical values. It does not call the
production plane converter or kernel nibble helper to manufacture agreement.

There is also an independent FP64 mathematical dot and a bound fixed before GPU
evaluation:

`abs(actual - ideal) <= gamma_(K/32 + 16) * sum(abs(code * scale * inverseD * input))`

where `gamma_n = n*u/(1-n*u)` and `u=2^-24`. The operation budget covers scale/code
multiplication, the longest lane FMA chain, chain combination and five shuffle
reductions. The finite fixtures have no overflow/underflow. This bound is not
adjusted after observing device output; exact F32 oracle agreement remains a
separate mandatory gate. A numerically close output that misses the original
schedule's bits does not pass.

Every original/converted weight plane, input, full CPU result, FP64 ideal,
absolute-product sum, bound and guarded device output is retained with hashes.
Failures preserve partial records and raw outputs with `complete=false`. CPU
self-test builds all90 fixtures, checks finite active results and the fixed
FP64 bounds, and verifies the analytical schedules; it initializes no CUDA.

Passing this component does not demonstrate full-model logit equivalence, task
quality, speculative acceptance, long-context capacity, or end-to-end speed.
The candidate remains disconnected from serving until those actual model
evaluations qualify it. No performance result is produced by this tool.
