Independent 64-query attention launchability campaign

This is a new experiment and new permission schema. It does not modify the
sealed Qwen35AttentionBench or Qwen35AttentionCompilerAblation, their markers,
their prior resource rejections, or production/ggml. The old <=128-register,
two-CTA, no-spill goal was an occupancy hypothesis. A rejection by that goal
was not an actual CUDA launch failure and was not GPU quality/performance data.

Five separately built arms are available: v2-original, v3-original,
v4-original, v3-ptxas-o1 and v4-ptxas-o1. Each has only baseline0 and one
candidate. Baseline0 always retains original O3/front-end/vectorization/ptxas
flags. The two O1 candidates change only ptxas optimization level. All generated
kernel sources are SHA-pinned to the sealed variants; no MMA, FP32 reduction,
KV precision, math flag, tile code or combine arithmetic is changed.

The private runner differs from the pinned sealed runner by exactly one
resource-predicate replacement, checked by reversing that delta byte for byte.
Actual driver occupancy >=1 CTA at the original 256-thread launch is required
for both baseline and candidate. Registers, shared memory, local storage,
stack and spills are recorded; no maxregcount or two-CTA/minblocks hint is used.
Offline compilation is never GPU quality PASS. Resource-only success is not
quality PASS. Each arm fails independently, so a bad arm cannot qualify or
conceal a different arm.

Offline generation, without invoking nvcc or CUDA:
  python3 eng/validation/Qwen35AttentionLaunchabilityBench/build.py \
    --arm v3-original --out artifacts/attention64-v3-source --generate-only

Offline native build, later on a machine with CUDA13:
  python3 eng/validation/Qwen35AttentionLaunchabilityBench/build.py \
    --arm v3-original --out artifacts/attention64-v3-built --nvcc /path/to/nvcc

Only an explicitly permitted later GPU window may run:
  python3 eng/validation/Qwen35AttentionLaunchabilityBench/run.py \
    --build artifacts/attention64-v3-built --name resources --resources-only
  python3 eng/validation/Qwen35AttentionLaunchabilityBench/run.py \
    --build artifacts/attention64-v3-built --name strict-long --long
  python3 eng/validation/Qwen35AttentionLaunchabilityBench/run.py \
    --build artifacts/attention64-v3-built --name bench-long --bench --long

Outputs must be new directories under ignored artifacts/ or docs/validation/.
Each arm needs its own build directory and every run a new result name.
Generate-only, compiler/version/runner failure or changed source/tool/artifact
hashes cannot issue this campaign's independent compiled permission. Native
execution rechecks all inputs before and after, and refuses sealed schemas.
The inherited qualified-variants.txt is an internal list of the two compiled
IDs inside this new directory, not a claim of passing the old occupancy goal.

The original strict fixture plan remains 79 shapes, or 81 with --long, three
complete candidate comparisons per shape. It checks the complete guarded F32
output and partial arrays, original canary-based write coverage, input-cache
guards, and baseline A/A before candidate comparisons. No tolerance, selected
token, NaN normalization or reduced-output comparison is introduced. Its
finite synthetic fixtures and canary policy are inherited, not comprehensive
quality evidence for arbitrary data.

Bench first runs that entire strict plan, then gates each exact timed shape.
There are four timed shapes or six with --long, each with five A-B-B-A rounds.
Each half resets output/partial buffers and issues three complete pipelines,
including both 64-query subtiles and the original combine when splits>1; the
full final output/partial arrays are compared after every half. The inherited
event key kernels_per_half=3 means three complete Issue calls, not three CUDA
kernels. All twenty/thirty events and positive finite intervals are required.
No preprocessing is omitted or new preprocessing introduced. Controller
evidence is controller.json; the unchanged native runner writes run.json.

This tool only prepares a bounded experiment. Smaller query tiles can increase
KV reads and launch overhead, and one launchable CTA does not imply speedup.
Strict failure forbids timing that arm. Even a future standalone full-bit PASS
and faster pipeline would require separate real-model/logit/state and end-to-end
quality/performance gates before any production integration.

CPU tests, no nvcc or GPU:
  python3 -m unittest discover \
    -s eng/validation/Qwen35AttentionLaunchabilityBench -p test_build.py -v
