# Isolated Qwen3.5/3.8 prefill-attention experiments

This directory is a standalone prototype. It does not edit production kernels,
export their ABI into TensorSharp, change precision, patch ggml, or enable a
production optimization. Wait for the actual 128K/250K baseline before choosing
a candidate. A theoretical occupancy increase does not identify a bottleneck.
Hardware counters are **NOT_RUN** in the current VM permission environment.

`build.py` extracts the current TensorSharp-owned helpers, original prefill
kernel and split-combine kernel into ignored output storage. Variant 0's kernel
body is exactly the production body apart from its private name. It fails if
known arithmetic/layout constants or expected source anchors change. Every
generated kernel statically asserts its exact shared-memory size.

| Variant | Real queries/CTA | Raw banks | Dynamic shared bytes | Intended coverage |
| --- | ---: | ---: | ---: | --- |
| 0 | 128 | 2 | 68,608 | Original oracle |
| 1 | 128 | 1 | 51,200 | Single-bank scheduling only |
| 2 | 64 | 2 | 68,608 | Offline resource control; exceeds the 2-CTA shared limit |
| 3 | 64 | 1 | 51,200 | Combined candidate, only if its resource gate passes |
| 4 (optional) | 64 | 1 | 50,176 | Compact 256-half rows with vector-preserving bank swizzle |

Build and CPU checks require no GPU:

```sh
python3 -m unittest discover -s eng/validation/Qwen35AttentionBench -p 'test_*.py' -v
python3 eng/validation/Qwen35AttentionBench/build.py \
  --out artifacts/qwen38-regression/attention-standalone --arch sm_120a
```

Use `--generate-only` when nvcc is unavailable. That validates generation only;
it is not native compilation or device validation. To explicitly narrow scope
after investigating a 64-query compilation failure, use `--variants 0,1` and
retain that coverage limitation in the report.
The default remains variants 0/1/2/3. To additionally build the compact fifth
branch, pass `--variants 0,1,2,3,4` to the same offline entry point.

The build compiles four separate cubins with the same explicit math flags and
records source, prototype and binary hashes. It parses actual ptxas registers,
stack and spill bytes. A 64-query branch is allowed into GPU testing only when
registers/thread are at most 128, total shared memory is at most 51,200 bytes,
and stack/spill bytes are zero. Variant 2 therefore records
`resource_rejected_no_gpu_run`; a rejected or unverified branch is never a pass.
No register cap is applied to force a misleading result. Missing ptxas fields
fail verification. A failed or generation-only build clears any stale GPU
qualification marker.

The CUDA runner additionally checks the actual loaded function attributes and
potential active CTA count on the current device. This is a resource check,
not a measured utilization or SFU/tensor/L2 diagnosis.
The 51,200-byte offline threshold is necessary, not sufficient: driver-reserved
shared storage may further limit residency. A 64-query branch with fewer than
two potential active CTAs is recorded as rejected and receives no GPU gate or
timing run. The runner's output directory must be inside its ignored build
directory.

```sh
build=artifacts/qwen38-regression/attention-standalone
mkdir -p "$build/resources" "$build/strict" "$build/bench"
"$build/attention-bench" "$build" "$build/resources" --resources \
  > "$build/resources/events.jsonl"
"$build/attention-bench" "$build" "$build/strict" --strict --long \
  > "$build/strict/events.jsonl"
"$build/attention-bench" "$build" "$build/bench" --bench --long \
  > "$build/bench/events.jsonl"
```

GPU runs require an explicit mode. `--resources` does not execute quality
kernels. `--strict` checks original A/A determinism, then **every output and
partial float bit**, including guarded regions. There is no tolerance or NaN
exception. Split cases must exercise a meaningful partial maximum/denominator,
rather than merely comparing untouched scratch. Failure writes the full
reference/candidate arrays and the first differing bit pattern.

Fixtures cover row and page/tile tails, splits 1/2/3/7/8, GQA ratios
1/2/3/6/8/16/64, two sequences, permuted physical pages and shared prefix pages,
finite random data, signed-zero/subnormal scales, zeros, opposing buddy-query
score maxima, and non-finite scales. `--long` adds late 131,072/250,000-key
prefixes. A non-finite mismatch rejects the candidate; it does not relax the
comparison. These are synthetic kernels, not model-quality results.

`--bench` first runs strict gates and also gates every exact timed shape. It
then reports ABBA CUDA-event samples, including the entire doubled 64-query
grid and split-combine cost. It excludes host fixture/readback work. Run on an
otherwise idle device with fixed clocks/power policy and record the actual
environment. Event timing alone does not establish end-to-end prefill gains.

If the toolkit provides Compute Sanitizer, separately run racecheck and
synccheck against `--strict`, especially variant 3's buddy synchronization and
the single raw-bank overwrite. Record unavailable tools as NOT_RUN. Before
any production integration, complete real-model full-logit and subsequent
state-dependent decode comparisons as well as long-prefill performance tests.

## Scheduling and exactness constraints

Single-bank loading primes the first raw tile, then keeps both original CTA
barriers. After the dequantization barrier every raw reader has finished, so
the next DMA may overwrite that bank while QK/softmax/PV consume only the
separate F16 K/V tiles. The next iteration waits for DMA and for previous
F16 consumers before overwriting K/V. It loses the original overlap between
next DMA and current dequantization; speed is unknown.

The 64-query kernel splits each **original 128-row logical tile** into two
physical CTAs. Original `kAll`, 32-key split ranges, softmax reduction order
and partial layout remain unchanged. Original upper/lower MMA query rows stay
in their original positions. Only two real A registers per k-step and two
real F32 output accumulators per fragment remain live. Dummy MMA C is always
zero and its output is temporary, avoiding a dead accumulator kept live by
volatile assembly constraints.

Two buddy warps reconstruct the original 16-query warp's
`all(c0 == 1 && c1 == 1)` rescale decision. Each writes its vote into unused K-row
padding, and an additional CTA barrier makes both visible. Inactive buddies
also participate. This preserves the original multiplication-by-one branch
even for unusual floating-point values without adding shared storage beyond
51,200 bytes. The extra synchronization, duplicated QK work and duplicated KV
dequantization can outweigh any additional active warps. Exactness and speed
remain unverified until the real nvcc/device gates pass.

## Optional compact/swizzled variant 4

Variant 4 addresses a possible per-CTA driver shared-memory reserve. It removes
both eight-half row paddings, saving 1,024 bytes for a 50,176-byte allocation.
Every logical eight-half vector begins at physical column
`col ^ ((keyRow & 7) * 8)`. Only bits 3..5 move, so each 16-byte vector keeps its
internal bit order and alignment. The original half-scale conversion and four
uint4 stores per dequant thread are retained; each vector's destination moves.
Both QK `ldmatrix` and transposed PV `ldmatrix` use the matching per-row address.

CPU tests exhaust all 65,536 half bit patterns through the store permutation
and check every source-lane vector. They also check that every eight-row
address group covers 32 distinct word banks under NVIDIA's documented
[32-bank shared-memory model](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html).
This is an address proof, not measured bank-conflict counter evidence;
hardware counters remain NOT_RUN.

Without padding, buddy votes borrow the first 32 bytes of K. An additional
all-CTA barrier after QK finishes every K consumer before any vote overwrites
those values. The following vote barrier makes both buddies visible; PV reads
only V. Next iteration's normal dequantization restores K after its existing
barrier. Inactive buddies join both barriers. No extra shared object is added.

All register/spill and actual-device occupancy gates still apply. Swizzle
indexing may exceed the register limit, and extra synchronization may erase
any benefit. Variant 4 receives no GPU bit or timing run if resources reject
it; it has no production integration and remains native/device-unverified
until the separate build and strict gates run.

## Review hardening and result classification

The qualification marker is published atomically only after every requested
cubin and the runner compile successfully and the complete build manifest is
written. A failed compile cannot authorize a stale runner. Missing compiler
or resource fields leave the branch unvalidated. The runner rejects malformed
or duplicate marker entries.

Every required output/partial word must replace the fixture's reserved canary;
unused split-1 partial scratch must remain unchanged. This prevents a baseline
and candidate from both skipping the same rows and falsely matching. The
actual host checker is exercised by a local C++ test, including shared missing
writes, guard corruption and unexpected unused-scratch writes.

Strict comparison repeats each candidate three times and checks baseline A/A
for every strict and timed shape. Every ABBA half poisons its buffers, launches
three complete calls, and verifies its final full output/partials. GPU-event
time excludes poisoning and readback, but those transfers occur between
halves and can affect cache/power state; report this as a controlled synthetic
microbenchmark, not hot-cache or end-to-end prefill throughput. `total_keys`
records prefix plus query rows explicitly. The five ABBA repetitions are five
paired samples, not fifteen independent observations.

| Observed status | Honest classification |
| --- | --- |
| `source_generation_only` | CPU generation only; native and GPU NOT_RUN |
| compiler unavailable | Native compile NOT_RUN; GPU NOT_RUN |
| `compile_failed_not_validated` | Native compilation failed; no usable qualification marker |
| `resources_unverified_no_gpu_run` | Resource evidence incomplete; GPU NOT_RUN |
| `resource_rejected_no_gpu_run` | Resource branch rejected; quality/timing NOT_RUN, never a quality pass |
| `--resources` completes | Device resource checks only; quality/timing NOT_RUN |
| oracle A/A or write-coverage failure | Invalid baseline/comparison; stop before claiming candidate correctness |
| candidate bit mismatch after valid A/A | Candidate correctness failed; no timing qualification |
| strict completes with active variant IDs | Synthetic bit gate passed only for those IDs and logged shapes |
| ABBA completes | Synthetic timing evidence only; production/model quality remains untested |
