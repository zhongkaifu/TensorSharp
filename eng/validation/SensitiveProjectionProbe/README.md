# Sensitive projection probe

This standalone CMake driver compiles the existing
`eng/validation/SensitiveProjectionProbe/sensitive_projection_test.cpp` against an explicitly
specified, unchanged ggml checkout and its prebuilt static libraries. It does not
fetch, configure, compile or patch upstream ggml, and it does not build or load
`GgmlOps.dll`.

The CPU executable uses the ordinary upstream backend. The CUDA executable uses
the existing TensorSharp executor with its private pedantic cuBLAS handle, which
forwards ordinary operations to the unchanged CUDA backend. A plain CUDA
`GGML_PREC_F32` tag still permits TF32 and is insufficient for this contract.
Both executables use the same TensorSharp projection-role and graph-cast helpers.
They compare **every element** of two chained
BF16-weight projections with F32 activations, at widths 512 and 809, against an
independent FP64 oracle using the stored BF16 weights. The predeclared F32 gates
are `max_abs <= 5e-5`, `relative_l2 <= 5e-6`, and `max_abs <= 5e-5` for the 98,304
shared output elements. They also check that throughput/quantized roles retain
their original construction, inputs remain unchanged, and conflicting cuBLAS
precision overrides are recognized. The default-throughput results are
observations, with no retrospectively selected error threshold.

The separate `sensitive_q8_projection_test.cpp` fixture uses the same role-based
Q8 factory as the model graph. It writes checkpoint Q8_0 blocks directly, then
decodes the stored FP16 scale and signed bytes independently for an FP64 oracle.
Two chained projections at widths 1, 3, 8, 512 and 809 must meet the same
`max_abs <= 5e-5` and `relative_l2 <= 5e-6` gates for all 255,936 output values.
All 99,072 within-class overlapping output values must be bit-identical:
1 and 3 columns against 8, and 512 columns against 809. Cross-class differences
for the 2,304 shared values at widths 1/3/8 against 809 are observations only;
all widths independently retain the predeclared FP64 error gates. This follows
the existing short decode/verify versus wide prefill precision contract without
requiring inter-class bit identity. Each of the 1,333
columns also contains a small F32 activation residual that both projections must
preserve exactly; narrowing that activation to Q8 would erase it. Resident Q8
weight pointers, source activations and guard rows must remain unchanged. The
owned Q8 operator has no dense weight widening or private packing workspace.

The independent `sensitive_q5_projection_test.cpp` fixture exercises the shared
quantized factory with original checkpoint Q5_K weights and F32 activations.
Its two stages have shapes `[512,256]` and `[256,128]`, at widths
1, 3, 8, 512, 774 and 809. The fixture writes the 176-byte, 256-element blocks
directly: two FP16 super-scales, twelve bytes of unsigned six-bit group
scales/minima, thirty-two fifth-bit bytes and 128 low-nibble bytes. Independent
packing and decoding cover all eight groups, quant codes 0..31, six-bit fields
including their high bits, signed/zero super-scales, a subnormal scale, and
positive/negative/zero weights. Fixed format examples check packed bytes before
backend execution. The FP64 dot oracle retains golden weights computed directly
from the half bits, integer scales/minima and quant codes before packing. Every
stored weight is checked against that golden value with the independent byte
decoder and against the unchanged upstream `to_float` format decoder, after
rounding the golden value to F32. Upstream output never defines the dot oracle.

Every one of the 809,088 main-chain output values must satisfy the predeclared
`max_abs <= 5e-5` and `relative_l2 <= 5e-6` gates. The 495,360 within-class
overlap values must be bit-identical: widths 1 and 3 against 8, and 512 and 774
against 809. Cross-class comparisons of 4,608 values are observations only.
Each of the 2,107 main columns preserves a small F32 residual exactly through
both stages. Additional K=256/width-3/output-129, K=512/width-1/output-129 and
K=512/width-809/output-67 cases check 262,847 complete output values and 813 exact
residuals. These cover partial vector/64-row tiles, partial final column tiles,
contiguous Q5 rows and interleaved, padded F32 inputs. Main cases use padded Q5
rows and padded F32 columns.

An additional independent K=4096 narrow cancellation fixture covers widths
1, 3 and 8 with 67 and 129 output rows, including both row tails. Its original
Q5_K blocks use exact unit super-scales and group scales/minima, with alternating
quant codes 2/0 producing adjacent `+1/-1` weights; alternating rows reverse the
signs. The retained prepack FP64 weights receive the same independent stored-byte
and unchanged upstream format checks as the main fixture. F32 activations equal
one except for `x[0] = 1 + (column+1) * 2^-21`, giving the independently specified
signed ideal residual `(-1)^row * (column+1) * 2^-21`. All 2,352 output values
must pass the FP64 error gates and equal that ideal rounded to F32 bit-for-bit.
All 784 shared output values at widths 1 and 3 against 8 must also be
bit-identical. A separate host negative control uses explicitly rounded F32
products and lane sums at K strides of 32, followed by fixed offsets 16/8/4/2/1;
it must erase every residual to zero and never supplies an oracle value.
The intended narrow CUDA contract groups adjacent products in each lane before
its lane sum and fixed warp reduction. This focused fixture also checks padded
Q5 rows, interleaved/padded F32 inputs, complete source/guard byte preservation
and resident source pointers. It retains every existing main-chain gate and
establishes no bit identity between narrow and wide dispatch classes.

All resident source pointers, original checkpoint bytes, guard rows, row padding
and activation padding must remain unchanged. Factory assertions preserve the
Q5_K sources and contiguous F32 outputs, cover every sensitive role, reject
unsupported formats and throughput roles, and keep Q5 nodes out of the legacy
Q8-specific predicate. No dense weight cast or private weight packing is used.

## Windows: MSVC and Ninja

Run from an x64 Visual Studio developer shell with CMake and Ninja available.
The libraries must match the checkout revision, target architecture, compiler
runtime and OpenMP configuration. CUDA mode additionally needs a CUDA toolkit
compatible with the prebuilt `ggml-cuda.lib` and a working CUDA device to execute.
Use a fresh output directory; the example paths below match the current validation
cache but must be changed when qualifying a different frozen generation.

```powershell
$projectionBuild = 'E:/TensorSharp-validation-cache/master-merge-20261013/native-v28/sensitive-projection-v1'
$projectionObjectsRoot = 'E:/TensorSharp-validation-cache/master-merge-20261013/native-v28/build-v3/build-production/CMakeFiles/GgmlOps.dir'
$projectionObjects = @('ggml_ops_matmul_precision', 'ggml_ops_dsv4_fused', 'ggml_ops_attention_precision', 'ggml_ops_q8_precision') |
  ForEach-Object { "$projectionObjectsRoot/$_.cu.obj" }
cmake -S C:/Works/TensorSharp/eng/validation/SensitiveProjectionProbe -B $projectionBuild -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DTSG_GGML_ROOT=C:/Works/TensorSharp/ExternalProjects/ggml `
  -DTSG_GGML_LIBRARIES=D:/TensorSharp-validation-cache/master-merge-20261013/native-v16/components `
  -DTSG_PROJECTION_NATIVE_ROOT=C:/Works/TensorSharp/TensorSharp.GGML.Native `
  -DTSG_PROJECTION_ENABLE_CUDA=ON `
  "-DTSG_PROJECTION_OWNED_CUDA_OBJECTS=$($projectionObjects -join ';')" `
  '-DCUDAToolkit_ROOT=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'
cmake --build $projectionBuild --target SensitiveProjectionCpu SensitiveProjectionCuda --parallel 2
cmake --build $projectionBuild --target SensitiveQ8ProjectionCpu SensitiveQ8ProjectionCuda --parallel 2
cmake --build $projectionBuild --target SensitiveQ5ProjectionCpu SensitiveQ5ProjectionCuda --parallel 2
ctest --test-dir $projectionBuild --output-on-failure --verbose -R '^sensitive-projection-cpu$'
ctest --test-dir $projectionBuild --output-on-failure --verbose -R '^sensitive-projection-cuda$'
ctest --test-dir $projectionBuild --output-on-failure --verbose -R '^sensitive-q8-projection-cpu$'
ctest --test-dir $projectionBuild --output-on-failure --verbose -R '^sensitive-q8-projection-cuda$'
ctest --test-dir $projectionBuild --output-on-failure --verbose -R '^sensitive-q5-projection-cpu$'
ctest --test-dir $projectionBuild --output-on-failure --verbose -R '^sensitive-q5-projection-cuda$'
```

Both ggml paths are required and must be absolute. Windows CUDA mode imports
exactly `ggml.lib`, `ggml-base.lib`, `ggml-cpu.lib`, and `ggml-cuda.lib` from the
specified directory. It uses the same OpenMP, thread, CUDA runtime, cuBLAS and
driver link relationships as the frozen native validation driver. CUDA mode also
requires the four explicitly listed frozen TensorSharp CUDA objects; it links
them without compiling any CUDA source.
For an incremental build, use the actual object paths from its recorded
`FinalObjectPaths`; reused objects can reside in immutable ancestor directories.
Do not infer their location from the newest build directory or substitute
similarly named files from another generation.

The CPU part of
`ggml_ops_matmul_precision.cpp` is compiled from the selected native source
directory into each executable; the Q8 and Q5 executables additionally compile
`ggml_ops_q8_precision.cpp`. The CPU
executable links only `ggml-cpu` and `ggml-base`, so it does not require a CUDA
device. `TSG_PROJECTION_NATIVE_ROOT` can instead point to a frozen copy containing
the fixture and all its native headers; its default is this checkout's native
directory.

For CPU-only configuration use a new build directory and
`-DTSG_PROJECTION_ENABLE_CUDA=OFF`, then build `SensitiveProjectionCpu`. This
requires only the three non-CUDA ggml static library files and does not discover
a CUDA toolkit. On POSIX platforms the equivalent prebuilt filenames are
`libggml.a`, `libggml-base.a`, `libggml-cpu.a`, and optionally `libggml-cuda.a`.
POSIX, Apple devices and differing upstream revisions still need their own actual
qualification; providing this driver does not establish their results.

CPU-only Q8 qualification instead builds `SensitiveQ8ProjectionCpu` from that
same configuration. The Q8 fixture rejects
`TS_GGML_Q8_PARALLEL_VECTOR=1` and `TS_GGML_Q8_PARALLEL_SMALL_BATCH=1`: those
explicit experiments change the short-column reduction order and are outside
the qualified default column-identity contract. It reports the selected
`TS_GGML_Q8_PREFILL_TILE` without rewriting the environment; the tile geometry
changes reuse while preserving each output's increasing-K arithmetic.

CPU-only Q5 qualification builds `SensitiveQ5ProjectionCpu` from the same
configuration. `SensitiveQ5ProjectionCuda` links the exact same four frozen
owned CUDA objects as the other CUDA targets; it adds no CUDA source or object.
The Q5 fixture also rejects the two explicit Q8 parallel-reduction flags and
reports the selected prefill tile without modifying the environment. CUDA uses
the original backend wrapped by `tsg_dsv4_fused_backend_init`, and frees that
wrapper before the original backend. CUDA absence returns 77 and is unavailable,
not a passing Q5 qualification. These source fixtures establish no device,
dependency-revision or real-model result until their actual recorded execution.

## Evidence and limits

`SensitiveRouterProjectionCpu` / `SensitiveRouterProjectionCuda` independently
qualify the shared `routing_logits` role for F32 weights at widths
1, 3, 8, 512, 774 and 809. The fixture computes full logits, sigmoid probabilities,
selection-only bias, a hard top-8 cut and normalization of the selected **unbiased**
probabilities, compared with independent FP64 arithmetic on stored F32 inputs.
Near-tie weights compute `row * (x[0] - x[1])` with F32 residuals of
`(1..4) * 2^-16`; every expected selected ID/order is independently specified and
the cut gap must exceed eight times the predeclared probability error bound.
A dense deterministic case supplements this cancellation case. The declared
strict gates are logit `max_abs <= 5e-5`, `relative_l2 <= 5e-6`, probability
`max_abs <= 2e-7`, exact top-8 IDs/order, and normalized-weight error and sum error
`<= 1e-6`. The near-tie logits additionally equal FP64 rounded to F32 exactly.
The ordinary `GGML_PREC_F32` backend path is an observation, not a requirement
that every device select TF32 or fail the near-tie case. The fixture exposes
diagnostic intermediates, changing fusion/liveness; it is not a production model
graph or throughput comparison. Both inputs and guard bytes must remain unchanged,
and graph buffers, contexts and backends are explicitly released. **No shared
ledger is installed:** this lifecycle check does not prove zero shared-ledger
owners, vendor-workspace reclamation or any physical hard limit.

Build these targets using the same explicit inputs as the other projection
targets, then run `ctest --test-dir <build> --output-on-failure --verbose -R
'^sensitive-router-projection-(cpu|cuda)$'`. CPU-only configuration can build and
run `SensitiveRouterProjectionCpu` without CUDA; it does not qualify CUDA.

The driver validates file presence and links the selected files; it does not
certify that a library was built from the selected checkout. The caller must
record upstream revision and clean status, hashes of the imported libraries,
owned CUDA objects and
the fixture/header/driver files **before and after** the build and execution,
actual configure/build/run commands and exit codes, compiler/toolkit versions,
device identity, and complete stdout/stderr. Keep this generated evidence in
ignored `artifacts/`, `docs/validation/`, or an external validation cache. A
source or dependency change requires a fresh qualified build rather than reuse
of an earlier receipt.

`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16`, `fp16`, or `bf16` conflicts with the F32
contract and causes the CUDA fixture to fail. Leave it unset or use `auto` or
`f32`. Do not silently repair the test environment in an evidence wrapper.

Exit code **77** means no CUDA device is available. CTest marks that CUDA case
skipped; it must be reported as **unavailable, not passing**. CPU success cannot
qualify CUDA. These are operator numerical/lifecycle checks, not real-model
semantics, cross-chunk bit identity, end-to-end quality/performance, complete
physical memory ownership, or OS hard-limit acceptance. The fixture does not
measure or certify private vendor workspace reclamation.
