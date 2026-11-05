# Standalone NVFP4 GEMM pipeline experiment

This is a TensorSharp-owned validation tool. It changes no production kernel,
backend dispatch, ggml source, or earlier sealed experiment. CUDA compilation,
GPU quality, timing, and full-model validation are separate stages. The default
source generator/CPU tests never initialize CUDA or run NVCC.

The oracle imports the complete reviewed production prefix verbatim, including
`ts_q35_gemm_w4a4_128` and `_128_swiglu`. It uses BM128/BN128, WM4/WN2,
STAGES2, BK256, COLS_FAST=true. Only activation rows above64 are supported; this
matches the production dispatch to BN128. All launches use grid.z=1, so split-K
never changes the accumulation order. No activation/weight quantization or dtype
is changed.

| Arm | BN | Stages | BK | Dynamic shared bytes (sizeof, then runtime query) |
|---|---:|---:|---:|---:|
| Oracle0 |128|2|256|81920|
|1|128|1|256|40960|
|2|64|1|256|30720|
|3|64|3|128|50688|
|4|64|2|128|33792|

Each arm has plain and fused SwiGLU wrappers. Actual ptxas registers, stack,
spills, static shared and driver local/shared/active-CTA values are retained per
function. The only occupancy admission is actual driver launchability at256
threads and activeCTAs>=1. Smaller shared storage does not prove two CTAs fit:
registers, static/reserved shared storage and architecture limits also apply.
No forced register cap, minimum-block launch hint, fast math or hardware-counter
interpretation is used. Stack/spills remain visible and are not relabelled as a
no-spill result.

STAGES1 cannot directly instantiate the existing template: it would use
`wait_group<-1>` and read its first load before waiting. The private S1 copy only
changes scheduling: each iteration waits for all prior bank readers with a CTA
barrier, loads current stage0, commits, waits group0, barriers, then executes the
original MMA/epilogue bytes. A reverse-delta assertion recovers the entire exact
original template. Other arms call the unmodified template. BK128 uses the
existing8-byte cp.async scale path; k64 MMA steps remain0,64,128,... in the same
order as BK256. Source/layout reasoning is not a GPU precision pass.

The strict plan has131 shapes per selected arm:70 plain and61 fused. It includes
rows129/512/2048/4096, K multiples of256, weight-row tails, every FP4 code, finite
UE4M3 scales, cancellation, signed zero, tiny/extreme finite row scales and
non-unit alpha. Finite inputs can overflow or produce NaNs in the epilogue;
those outputs still require identical raw uint32 bits. Fused shapes require
outDim divisible by16 so each interleaved gate row has its original up row+8.
Fused output width is outDim/2;144/272-row tails are covered.

Output coverage uses two distinct interior poisons and exact repeated output
comparison. No float/NaN sentinel is assumed impossible as a legitimate output.
Every oracle and candidate has A/A checks; whole output bits, output guards and
all original input payload/guard bytes must match. Any difference fails with
raw reference/actual files and the first differing word. No tolerance, NaN
special case or argmax-only substitute is allowed.

Six timed shapes use rows512/4096 for each projection:

- GDN input: out16480, K5120, plain.
- GateUp: out34816, K5120, original fused SwiGLU.
- Down: out5120, K17408, plain.

Timed inputs use a separate moderate finite pattern6: uniform FP4 codes,
UE4M3 scale bytes0..63, positive rowScale in[2^-12,2^-10), alpha1. This avoids
selecting timings mainly from saturated/overflow SwiGLU inputs. These are still
synthetic inputs, not captured model activations or weights. Original broad
patterns0..5 remain in the strict plan. Both epilogues of the timed pattern6
shapes are also strictly checked, without changing the131/6 plan counts.

A bench run first completes all131 strict cases for that arm. Each timed shape
is rechecked before five ABBA rounds. Each half executes three complete GEMMs;
each half's entire final output and unchanged inputs are verified outside the
event interval. Events include all global/shared pipeline work, MMA and original
epilogue; no candidate-specific prepack exists or escapes the timer. Common
fixture allocation/upload and validation readback are excluded. These resident
synthetic-input kernel times do not measure activation quantization, target
prefill, token/s, end-to-end service latency or full-model quality.

Local CPU/source gates:

```sh
python3 -m unittest discover -s eng/validation/Qwen35Fp4GemmPipelineBench -p test_build.py -v
python3 eng/validation/Qwen35Fp4GemmPipelineBench/build.py --out artifacts/fp4-pipeline-generated-01 --generate-only
```

Later offline compilation (no GPU initialization or runner launch):

```sh
python3 eng/validation/Qwen35Fp4GemmPipelineBench/build.py --out artifacts/fp4-pipeline-built-01 --nvcc /usr/local/cuda/bin/nvcc --cuobjdump /usr/local/cuda/bin/cuobjdump
```

Only after the exclusive GPU owner schedules execution:

```sh
python3 eng/validation/Qwen35Fp4GemmPipelineBench/run.py --build artifacts/fp4-pipeline-built-01 --name v1-resources --variant 1 --resources-only
python3 eng/validation/Qwen35Fp4GemmPipelineBench/run.py --build artifacts/fp4-pipeline-built-01 --name v1-strict --variant 1
python3 eng/validation/Qwen35Fp4GemmPipelineBench/run.py --build artifacts/fp4-pipeline-built-01 --name v1-bench --variant 1 --bench
```

Repeat independently for variants2–4 with new result names. Build and result
directories must be fresh; failures cannot be overwritten or partial runs
stitched. A resource-only completion explicitly has quality_pass=false. Missing
or rejected build/source/artifact identity, driver resource failure, incomplete
strict/timing sets, wrong epilogue, any bit/coverage/guard/A-A failure, invalid
timing or process failure stays unqualified. A failed arm is not timed; another
arm needs its own complete gates. The standalone tool uses CUDA only, no ggml
headers/linking; the experiment coordinator must separately retain the unchanged
upstream ggml revision/status (current baseline facdc44a67da17df4433adcc468b424f48e9ded2).
Skipped or unavailable native/model/device scenarios are never passes.
