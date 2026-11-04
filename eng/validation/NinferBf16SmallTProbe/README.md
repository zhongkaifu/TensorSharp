# BF16 T8 exception projection component

Fixed shapes: GQA input14336×5120 (all six actual original BF16 matrices) and Output5120×6144 (all three). Three synthetic cases per shape cover exact dense dyadic products, cancellation/signed zeros and F32 low bits/BF16 ties with balanced exponent-range operands. Input/weight BF16 bytes remain unchanged, output stays raw F32. The isolated pristine Ignis small-T core is a source-guided candidate; the actual Ignis output dispatch can differ. There is no engine hook/default in this project.

The legacy arm calls the production `CudaNinferKernels.ProjectBf16` method: original F32→BF16 cast, real cuBLAS GemmEx BF16 operands/FP32 compute/F32 output. The candidate uses the original cast followed by the pristine SIMT schedule with an owned F32 store. A third pristine BF16-output reference must equal BF16_RNE(candidateF32) in every output bit; this comparison never substitutes for the independent raw-F32 FP64 gate. Candidate launches use ordinary `cuLaunchKernel`, with no new PDL or early consumer policy.

Build and CPU preparation (no CUDA context/model execution):

```sh
dotnet build eng/validation/NinferBf16SmallTProbe/NinferBf16SmallTProbe.csproj -c Release -p:TensorSharpSkipCudaPtx=true
dotnet PROBE.dll --self-test
dotnet PROBE.dll --prepare --artifact /ABS/ORIGINAL.ninfer \
  --expect-directory-sha DIRECTORY_SHA --out /ABS/artifacts/bf16-prepared-r1
python prepare-oracle.py --prepared /ABS/artifacts/bf16-prepared-r1/prepared-inputs.json \
  --expect-prepared-sha SHA --out /ABS/artifacts/bf16-oracle-r1
```

The CPU oracle uses NumPy float64 BLAS in256-row blocks. It streams original tensors and applies exactly the existing TS attention-row permutation, retaining original and mapped tensor hashes and full output/L1 arrays. It does not store original weight payloads. The prepared inputs retain their generator policy, source assembly, artifact directory/stat, complete BF16-representable inputs, consumer seeds and effective F32 unit-offset norm weights. Oracle preparation finishes before CUDA is initialized. No dependency installation is performed.

Root alone runs model/device work. Stage the actual compiled module, its emitted PTX and completed-build manifest together. Supply actual byte pins; no guessed mangled name or historic module substitution:

```sh
dotnet PROBE.dll --execute --timing --shape input \
  --oracle /ABS/artifacts/bf16-oracle-r1/oracle.json --expect-oracle-sha SHA \
  --helper /ABS/old-helper.fatbin --expect-helper-sha SHA \
  --candidate /ABS/tensorsharp_ninfer_bf16_small_t.fatbin --expect-candidate-sha SHA \
  --native-manifest /ABS/completed-build.json --expect-native-manifest-sha SHA \
  --out /ABS/artifacts/bf16-input-r1
# Repeat with --shape output and a fresh output directory.
```

Every shape runs two independent module lifetimes × three poisons × all real+three synthetic cases. It checks every raw F32 value against `gamma(K+64,u32)*sumAbs + 32*K*u64*sumAbs + K*2^-149`, where u32=2^-24 and u64=2^-53. Exact dyadic cases additionally require exact CPU F32 bits. Source BF16 output equivalence, full CPU cast bits, own-arm reset/lifetime equality, all tool-owned guards/input immutability and finite outputs are mandatory. There is no requirement for cuBLAS and candidate sum orders to produce identical F32 values.

The unchanged consumers are original Round for GQA and original AddRmsNorm for Output. Both arms validate the exact canonical residual and a fixed independent FP64 norm envelope (the prior norm component's conservative gamma/BF16 envelope). In particular dot1+2^-8=1.00390625 with residual−1 must retain a nonzero result; a premature BF16 projection store is rejected. The raw F32 and consumer captures are retained before declaring a gate pass. Opaque cuBLAS private workspace guards are outside the tool's control and are not claimed checked.

Timing starts only after the entire selected shape passes. It captures the **complete** original cast→GEMM→consumer chain in each arm; Output additionally includes the same D2D reset_x in both graphs on every replay. Actual library graph kernels/copies are retained, never assumed to be a single GEMM kernel or discarded. All known cast/candidate/consumer ABI, launch dimensions and reset-copy operands are checked. Each mode uses5ABBA,64 timed full-pass replays and one outside warmup per interval, symmetric stream/event boundaries and full own-arm output/guard/immutable checks after every interval. One-address and the real6/3 distinct-address sets are both measured; no extra fake Output copies are added. This is address-stream component timing, not proof of DRAM coldness or end-to-end speed.

The predeclared win gate is old/new median>1.05 and at least4/5 rounds>1.05 for **each** address mode. `timing.passed` means the measurement and numerical checks are valid; `predeclaredWin` is separate. The full model must still pass its paired no-old-True-loss quality evaluation and end-to-end measurement before production adoption. A legitimate different answer or an old wrong answer remaining wrong does not itself prove an engine bug.

Raw output/FP64 CAS has a512MiB hard cap. Weights are represented by original descriptors/tensor hashes, never retained as1GB raw copies. Complete evidence is written once after all device resources/context are disposed and all pinned inputs/PTX/native/CAS files are rehashed. No skipped, incomplete or unsupported case is a pass. Component correctness, synthetic coverage, model quality and performance are reported separately.

## Exact-old MMA admission screen

The original R48 SIMT source and evidence remain frozen in ignored validation snapshots.
The explicit `--execute --screen-only` mode uses a separately pinned pristine Ignis
MMA module (`tensorsharp-ninfer-bf16-mma-aot/1`) and the existing prepared/oracle
files. Do not pass `--shape` or `--timing`: this one direct pass covers **all nine
real matrices**, with no synthetic rerun or repeated-lifetime claim.

The actual emitted candidate ABI is four arguments (8/8/8/4 bytes at offsets
0/8/16/24); the original BF16-output reference is 8/8/16/4 bytes at offsets
0/8/16/32. Both receive tokens=8. The fixed original schedule uses N/32 blocks,
256 threads and 98,304 bytes of dynamic shared memory, opted in and checked before
launch. It uses ordinary launches. Every raw F32 output must equal the actual
public cuBLAS projection in all bits, including signed zero. All consumer outputs
and canonical residuals must also be bit-identical. Independent full FP64 bounds,
original BF16-reference rounding, CPU casts, finite outputs, guards and immutable
inputs remain mandatory.

```sh
dotnet PROBE.dll --execute --screen-only \
  --oracle /ABS/bf16-small-t-oracle-r48-r1/oracle.json --expect-oracle-sha SHA \
  --helper /ABS/old-helper.fatbin --expect-helper-sha SHA \
  --candidate /ABS/NEW_MMA.fatbin --expect-candidate-sha SHA \
  --native-manifest /ABS/NEW_MMA/completed-build.json --expect-native-manifest-sha SHA \
  --out /ABS/artifacts/bf16-mma-screen-r1
```

A mismatch retains complete raw arrays, counts and `failed-screen.json`, stops
immediately and never enters timing. Missing/unattempted cases remain unavailable.
The screen raw cap is 32 MiB. Passing this bounded screen is neither full repeated
component qualification nor a model quality/performance claim. A different legal
summation result rejected by this exact-old screen is not thereby an engine bug.

If and only if the real-only screen passes, Root may run the existing full modes
with this same MMA image: `--execute --shape input` and `--execute --shape output`
(optionally `--timing`, which follows the full selected-shape numeric gate).
The full modes retain all real+three synthetic matrices, two module lifetimes,
three poisons, AA/guards/fixed FP64 and stock rounding, and additionally require
all old/MMA raw F32, consumer and canonical-residual bits to match. Fresh timing
addresses must satisfy the same exact-old gate. The existing five ABBA ×64 replay
full-chain timing and one/real6/3 address modes are unchanged; the MMA graph's
fourth `tokens=8` argument, N/32 grid, 256-thread block and 98,304 shared bytes are
checked explicitly. R48 SIMT keeps its original non-parity contract.
