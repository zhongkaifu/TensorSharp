Private N8/R4 register/cache ablation. Original production and frozen validation
tools are not modified. This is not the rejected R1/R2/R3 experiment.

Oracle: freshly recompiled original R4/128-thread template, NOT production BBD.
Candidates: launch_bounds(128,3) only; weights codes/scales __ldcs only;
both together. Activation xq/xd loads, UE4M3/E2M1 conversion, int dp4a order,
FP32 per-K accumulation/alpha and warp_sum remain the original source bytes.
The streaming helper must reverse exactly to the original template. The private
runner reverses byte-for-byte to the sealed runner after metadata-only changes.

CPU/source generation (never initializes CUDA):
  python3 eng/validation/Qwen35Dp4aScheduleBench/build.py --out ABS_FRESH_IGNORED_DIR --generate-only
Actual offline build (no GPU launches):
  python3 eng/validation/Qwen35Dp4aScheduleBench/build.py --out ABS_FRESH_BUILD --nvcc /usr/local/cuda-13.2/bin/nvcc
Default run plan performs no GPU work:
  python3 eng/validation/Qwen35Dp4aScheduleBench/run.py --build ABS_FRESH_BUILD --name plan
Explicit GPU, inside ROOT's exclusive lease:
  python3 eng/validation/Qwen35Dp4aScheduleBench/run.py --build ABS_FRESH_BUILD --name strict --execute
  python3 eng/validation/Qwen35Dp4aScheduleBench/run.py --build ABS_FRESH_BUILD --name bench --execute --bench

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
