# Exact F16 intermediate for standalone prefill attention

This is an isolated experiment, with no production integration. It leaves ggml,
production sources, and the sealed `Qwen35AttentionBench` variants 0–4 unchanged.
Native compilation, device resources, CUDA correctness, timing, and real-model
quality are separate gates; local source-generation tests do not establish any
of those results. Hardware counters are not collected by this tool.

The private oracle is the original 128-query attention kernel, with its original
q8 cache loading, conversion, QK, softmax, PV, split layout, and combine. The whole
sealed baseline-0 generated source remains the candidate module's unchanged
prefix. A second private kernel obtains K/V from an exact F16 intermediate and
keeps the original query packing, per-element operations, reduction order, MMA
row positions, split ranges, masks, and output stores. The generated conversion
helper changes only its private name and destination stride from 264 to 256
halfs. Its int8 conversion and half2 multiply are original TensorSharp code.

The candidate first decodes the current layer's paged q8 K/V into
`[sequence][K/V][KV head][max keys][256 half]`. It then copies the F16 values into
two padded shared-memory K/V banks, totaling 67,584 bytes. The original two CTA
barriers, async wait/commit sequence, alternating banks, and inactive-warp
participation remain. Each copy uses the original split/query-tile `kHi` bound,
including zero fill beyond that bound. Merely checking the sequence's final key
would be incorrect for non-finite future V values. Inactive copies use a valid
dummy global address rather than forming an out-of-range pointer.

The fixture explicitly requires positive rows, nondecreasing causal key ends,
`0 <= last key end <= max keys <= page count * 256`, and resident pages through
the final key. Address and size products widen before multiplication. The
standalone predecode supports different final bounds per sequence; the strict
small-fixture probe checks unused allocated keys with independent bounds without
mutating the original attention fixture's ends. Pages wholly beyond a shortened
bound are set to null in that independent table, testing early return before
any inaccessible-page lookup.

An additional private variant 2 is compiled but is **not selected by default**.
It differs from variant 1 only by omitting the second loop CTA barrier, which
originally followed q8-to-F16 shared writes. Variant 1 retains both barriers.
In variant 2 the current F16 bank is already fully available after the first
`cp_wait<0>` and CTA barrier. No thread writes that bank in the rest of the
iteration; prefetch writes only the other bank. The next iteration's first
barrier prevents reusing the previous bank until every old consumer is done.
Inactive warps still take the first barrier and commit before continuing.
CPU source gates require this sole difference; device race/sync and strict-bit
validation remain necessary. Selecting variant 2 never replaces variant 1.

## Memory and timing scope

The F16 intermediate is for **one current layer and request group**, not all 16
attention layers held resident. For one sequence and four KV heads:

| Actual total keys | Candidate workspace | Strict-only padded reference |
| ---: | ---: | ---: |
| 131,072 | 512 MiB | 528 MiB |
| 250,000 | 976.5625 MiB | 1,007.080078125 MiB |
| 256,000 | 1,000 MiB | 1,031.25 MiB |

These exclude q8 source pages, Q/gates, output/partial buffers, and guards. The
strict checker temporarily holds both the plain workspace and a padded legacy
reference, plus host readback arrays. It releases the padded device reference
before attention and timing. The plain candidate workspace remains allocated
for both arms of ABBA, so this microbenchmark does not measure production peak
memory or the effect of making a second allocation only in the candidate arm.

**Every timed candidate call includes full predecode, attention, and any split
combine, in the same stream and event interval.** There is no cached predecode
or preprocessing outside the timer. Poisoning and full readback occur between
ABBA halves, outside events, and may affect cache and power state. Each half
contains three complete calls, then verifies full output and partial bits.
The five ABBA repetitions are five paired samples, not fifteen independent
samples. Expanded F16 K/V adds a global write and read and can be slower; no
speedup or hardware bottleneck is assumed.

## Local generation and CPU checks

Run from the repository root:

```sh
python3 -m unittest discover -s eng/validation/Qwen35PredecodedAttentionBench -p 'test_*.py' -v
python3 -m py_compile eng/validation/Qwen35PredecodedAttentionBench/build.py eng/validation/Qwen35PredecodedAttentionBench/run.py eng/validation/Qwen35PredecodedAttentionBench/test_build.py
python3 eng/validation/Qwen35PredecodedAttentionBench/build.py \
  --out artifacts/qwen38-regression/predecoded-attention-generated-01 \
  --generate-only
```

Use a new output directory on every invocation; existing evidence is never
overwritten or resumed. Output must be under ignored `artifacts/` or
`docs/validation/`. Generation-only output has `complete: false`, no permission
marker, and cannot authorize a GPU run. CPU tests exercise source equivalence,
every decode store and async-copy vector address, 64-bit workspace sizes,
failure/marker handling, and the actual host half checker. The host checker test
compiles a small C++ program and covers all 65,536 half bit patterns, common
missing writes, padding, unused keys, and guard-independent buffer sizes. It is
not a proof of device half rounding or CUDA synchronization.

## Future explicit native and GPU commands

Run only after the parent benchmark coordinator grants the device window. The
build itself compiles without initializing CUDA or running the benchmark:

```sh
python3 eng/validation/Qwen35PredecodedAttentionBench/build.py \
  --out artifacts/qwen38-regression/predecoded-attention-native-01 \
  --nvcc /usr/local/cuda/bin/nvcc --arch sm_120a
```

The build rejects nonempty `NVCC_PREPEND_FLAGS` and `NVCC_APPEND_FLAGS`, records
the compiler version and relevant environment, and hashes production source,
sealed fixture helpers, owned tools, generated source, PTX, cubin, and runner.
The math flags match the original oracle and do not enable fast math or cap
registers. A unique permission marker is published only after complete native
compilation and hash checks. Build failure, missing resource fields, or a
missing artifact cannot authorize execution. CPU tests use mocked native
processes only to test this control flow; they do not create retained native
evidence.

Use `run.py`, which checks all hashes and the independent permission marker
before invoking the compiled runner. Each name must be a new child directory:

```sh
python3 eng/validation/Qwen35PredecodedAttentionBench/run.py \
  --build artifacts/qwen38-regression/predecoded-attention-native-01 --name resources-01 --resources
python3 eng/validation/Qwen35PredecodedAttentionBench/run.py \
  --build artifacts/qwen38-regression/predecoded-attention-native-01 --name strict-long-01 --long
python3 eng/validation/Qwen35PredecodedAttentionBench/run.py \
  --build artifacts/qwen38-regression/predecoded-attention-native-01 --name bench-long-01 --bench --long
```

To explicitly include the independent barrier ablation, append
`--with-no-second-barrier` to each desired `run.py` invocation and choose a new
name, for example `--name bench-long-barrier-ablation-01 --bench --long
--with-no-second-barrier`. Default active IDs are `[0,1]`; this flag requires
`[0,1,2]`, doubles the candidate strict comparisons, and benchmarks both
candidates against the unchanged oracle. Requested candidate omission or a
resource rejection cannot pass. Both F16 candidates use the same fully checked
predecode helper, while each receives its own complete output/partial gates.

`--resources` only loads functions and obtains driver attributes. It requires
both arms to launch at the actual 256-thread block size with their actual
dynamic and static shared memory and at least one potential active CTA. It
records registers, local memory, and ptxas stack/spill bytes as facts. This is
**not** the separate 64-query occupancy experiment: there is no 128-register,
zero-spill, or two-CTA admission rule here. A rejected arm is not a resource or
quality pass. Potential residency is not measured utilization.

Default strict mode covers 79 sealed shapes; `--long` adds two, for 81. Fixtures
exercise page/tile/row tails, split counts 1/2/3/7/8, GQA ratios
1/2/3/6/8/16/64, two sequences, permuted physical pages, shared prefixes,
signed-zero/subnormal scales, finite random data, zero inputs, opposing query
maxima, and non-finite scales. Long strict cases retain the sealed **prefixes**
131,072 and 250,000 plus 129 query rows. Long timed cases have **total keys**
131,072 and 250,000. This synthetic plan does not claim coverage of the distinct
256,000-key end-to-end fixture; manifests and events record actual total keys.

For every strict and exact timed fixture, a padded reference uses the untouched
legacy decode helper twice with distinct poison patterns. Every valid half bit
must match A/A. Two independent plain decodes with different poisons must match
that reference, while padding, unused key tails, and global guards remain
untouched. No half sentinel is assumed impossible or excluded. Small fixtures
add 73 unused-key boundary probes, including unequal per-sequence bounds and
null inaccessible pages. The run wrapper requires this separate count as well
as the full fixture counts. Full attention outputs and partials
then require oracle A/A plus three candidate comparisons with zero tolerance,
including every NaN payload and signed-zero bit. Required output/partial words
must replace the sealed canary, and split cases must contain meaningful
partials. Every ABBA half repeats full output/partial/guard validation.

Mismatch stops qualification and saves full arrays and the first differing
location. Optional Compute Sanitizer racecheck/synccheck can be run separately
on the private runner after hash verification; record unavailable tools as
NOT_RUN. Before any production integration, require real-model full-logit,
state-dependent continuation, and full long-prefill evaluations with complete
pipeline cost and memory accounting. A synthetic bit or timing pass is not a
model-quality or end-to-end performance pass.

| Observation | Classification |
| --- | --- |
| Source generation / CPU tests only | Native, device, GPU quality and timing NOT_RUN |
| Native compiler unavailable | Native and GPU NOT_RUN |
| Compile/resource parsing/artifact failure | Rejected; no execution marker |
| Driver cannot launch either arm | Resource rejected; quality/timing NOT_RUN |
| Resources-only success | Device resources only; quality/timing NOT_RUN |
| Reference decode or attention A/A failure | Invalid comparison; do not claim candidate correctness |
| Candidate half/output/partial mismatch | Correctness failed; no timing qualification |
| Strict success | Synthetic exact-bit coverage for logged shapes only |
| ABBA success | Controlled complete-pipeline synthetic timing only |
