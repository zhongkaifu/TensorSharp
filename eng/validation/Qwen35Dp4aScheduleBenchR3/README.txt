Private N8/R4 register/cache ablation. Original production and frozen validation
tools are not modified. This is not the rejected R1/R2/R3 experiment.

Oracle: freshly recompiled original R4/128-thread template, NOT production BBD.
Candidates: launch_bounds(128,3) only; weights codes/scales __ldcs only;
both together. Activation xq/xd loads, UE4M3/E2M1 conversion, int dp4a order,
FP32 per-K accumulation/alpha and warp_sum remain the original source bytes.
The streaming helper must reverse exactly to the original template. The private
runner reverses byte-for-byte to the sealed runner after metadata-only changes.

CPU/source generation (never initializes CUDA):
  python3 eng/validation/Qwen35Dp4aScheduleBenchR3/build.py --out ABS_FRESH_IGNORED_DIR --generate-only
Actual offline build (no GPU launches):
  python3 eng/validation/Qwen35Dp4aScheduleBenchR3/build.py --out ABS_FRESH_BUILD --nvcc /usr/local/cuda-13.2/bin/nvcc
Default run plan performs no GPU work:
  python3 eng/validation/Qwen35Dp4aScheduleBenchR3/run.py --build ABS_FRESH_BUILD --name plan
Explicit GPU, inside ROOT's exclusive lease:
  python3 eng/validation/Qwen35Dp4aScheduleBenchR3/run.py --build ABS_FRESH_BUILD --name strict --execute
  python3 eng/validation/Qwen35Dp4aScheduleBenchR3/run.py --build ABS_FRESH_BUILD --name bench --execute --bench

Default all3 candidates: 96 sealed fixtures each (288 comparisons); 12 exact
timed shapes each x5 ABBA rounds (180 events). Original full bits, nonfinite
bits, external guards, complete-write checks, real N8 argmax, oracle/candidate
A/A and complete output validation for every timed half remain mandatory.
Bounds hint is experimental, not a forced max-register workaround. Actual
ptxas stack/spills and driver local bytes must be0; actual driver active CTA>=1
at128threads is required. >=3CTA or <=168registers is a hypothesis, not a pass
threshold or a speedup. Record all resources honestly; no hardware counters.

An invalid selected arm blocks that build: no automatic pruning or fallback.
--arms bounds|stream|combo explicitly builds a separate fresh one-arm evidence
set when needed; it never upgrades a prior all-arms resource failure.
Compilation, GPU exactness/occupancy/timing, real model and serving integration
remain NOT_RUN until actually executed. Synthetic timings are not an end-to-end
quality/performance claim. Production keeps original weights/KV/state/policy.

R3 optional all-data/L2 eviction control (R2 remains sealed in its original dir):
  python3 eng/validation/Qwen35Dp4aScheduleBenchR3/run.py --build ABS_FRESH_BUILD --name bench-sweep --execute --bench --cache-sweep
This option requires --bench and forces iterations=1 (explicit other values are
rejected). Warm mode remains the default with iterations=3 and no sweep. Before
each A1/B1/B2/A2 GPU-event start, a separate kernel reads a fresh guarded buffer
of at least 2 x the actual driver L2 capacity, using asm volatile ld.global.cg.u32
with a memory clobber. Per-block uint64 sums/counts and total full-read coverage
are verified after synchronization; the sweep and its checks are outside the
measured GEMV interval. The complete original 96-case gate still runs first.
Actual L2 bytes, sweep bytes, launch resources and every half identity are saved.

.cg is an L2 cache hint, not an eviction or physical-cold guarantee. This is an
all-data/L2 eviction control: it also displaces precomputed Q8 inputs, rather than
isolating weights. GEMV matrix-only measurements do not establish the real model's
streaming conditions. Only actual full-model E2E can establish a serving benefit.
New sweep entry is appended after the unchanged matrix source; reverse CPU gates
recover the exact R2 source/runner. No production/default/kernel ABI is modified.
PTX cache operators: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators
