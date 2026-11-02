# Standalone attention compiler ablation

This opt-in tool leaves the sealed `Qwen35AttentionBench` generator, variants0..4,
production kernels and ggml unchanged. It produces six separate arms: variants3
and4, each with exactly one of these experiments:

| Experiment | Candidate change | Baseline0 |
| --- | --- | --- |
| `no_vectorization` | Remove `--extra-device-vectorization` | Original source and flags |
| `ptxas_o1` | Add ptxas `-O1`; keep NVCC `-O3` | Original source and flags |
| `min_blocks2` | `__launch_bounds__(256, 2)` | Original source and flags |

The launch-bounds hint is explicitly a new resource experiment. It can cause
spills and does not establish useful occupancy. Every candidate still requires
actual ptxas registers ≤128, total shared memory ≤51,200 bytes, and zero stack,
spill loads and spill stores. The unchanged runner then requires actual driver
register/local/shared attributes and potential active CTA count ≥2 before any
candidate quality or timing run. No gate is relaxed for a hint.

Nonempty `NVCC_PREPEND_FLAGS` and `NVCC_APPEND_FLAGS` are rejected before creating
output or invoking a compiler, so hidden precision/resource switches cannot
silently alter baseline0. Relevant compiler environment values are recorded.
Native mode records `nvcc --version`; a failed version probe authorizes nothing.
Generation-only mode does not invoke or claim to validate the compiler.

Each arm receives its own unchanged baseline0 source/cubin, exact copied runner
and checker, generated candidate, commands, logs, hashes and manifest. Baseline0
keeps its original full-query/raw-bank implementation and compiler flags in all
arms. Its existing resource behavior is allowed as the oracle; candidate gates
are stricter. This is an isolated comparison, not production integration.

CPU-only checks and source generation:

```sh
python3 -m unittest discover -s eng/validation/Qwen35AttentionCompilerAblation -p 'test_*.py' -v
python3 eng/validation/Qwen35AttentionCompilerAblation/build.py \
  --out artifacts/qwen38-regression/attention-ablation-generated-01 --generate-only
```

The output directory must be **new** and inside ignored `artifacts/` or
`docs/validation/`. Existing output, including an earlier resource failure, is
never overwritten or resumed. A later native build must use another new path.
Generation does not publish `qualified-variants.txt`. CPU fake-compiler tests
only check control flow; their temporary outputs are deleted and are not native
evidence.

During a later authorized offline compiler window:

```sh
python3 eng/validation/Qwen35AttentionCompilerAblation/build.py \
  --out artifacts/qwen38-regression/attention-ablation-native-01 --arch sm_120a
```

Use `--variants 4 --experiments min_blocks2` to explicitly narrow coverage.
There are no hidden combinations of hints or compiler switches. Missing compiler,
failed compilation, incomplete resource logs or failed runner build leave the arm
unauthorized. Only actual successful compilation, complete resource evidence and
a candidate resource pass publish its marker, atomically after its final
manifest. A rejected arm receives no marker, including no baseline-only marker.
`complete=true` records a completed compile/resource experiment and can coexist
with `candidate_offline_qualified=false`; it is never a quality pass.

For each arm whose manifest has `candidate_offline_qualified=true`, ROOT can use
the exact sealed runner with its isolated build directory:

```sh
arm=artifacts/qwen38-regression/attention-ablation-native-01/variant4-min_blocks2
mkdir -p "$arm/resources" "$arm/strict" "$arm/bench"
"$arm/attention-bench" "$arm" "$arm/resources" --resources
"$arm/attention-bench" "$arm" "$arm/strict" --strict --long
"$arm/attention-bench" "$arm" "$arm/bench" --bench --long
```

Driver rejection still means candidate quality/timing **NOT_RUN**. The runner
checks baseline A/A, every output/partial float bit, write coverage and guards,
including exact timed shapes. Its ABBA events include both64-query subtiles and
combine. They exclude poisoning/readback but those transfers occur between
halves, so this is a controlled synthetic experiment, not hot-cache or end-to-end
prefill throughput. Five paired ABBA samples are not fifteen independent samples.
Hardware counters, real-model quality and production performance remain NOT_RUN
until separately measured. Preserve failures as evidence and never substitute
resource eligibility for correctness or speed.
