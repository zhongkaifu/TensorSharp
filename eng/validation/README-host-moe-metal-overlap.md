# HostMoE Metal shared-branch overlap

`TS_HOST_MOE_METAL_OVERLAP` is a startup opt-in for the common segmented host-expert executor:

| Value | Execution |
| --- | --- |
| Unset or `0` | Existing sequential execution |
| `1` | Submit the independent Metal prefix, run the existing CPU experts, then drain |
| `2` | Run the same prefix synchronously before CPU experts, as a serial-split diagnostic |

Eligibility follows capabilities rather than model names, layer indices or expected expert counts. Execution requires an actual Metal backend, one configured device, one decode row, and no GPU-reference verification. All three host inputs are downloaded before prefix submission.

The pending host result must be a fresh contiguous F32 non-view leaf, pinned as both input and output. Boundary tensors and their storage roots remain outputs. The cached original-order prefix must start with a complete matrix multiplication and end immediately before `ADD(pending, last_MUL)`. Only the existing matrix multiplication, SiLU/sigmoid, multiplication and metadata/contiguous-copy grammar is accepted, with at most 64 nodes. Allocation and alias checks run after binding refresh. Unsupported graphs retain ordinary execution; no graph reordering or wider arithmetic grammar is introduced.

The CPU expert implementation, quantization and reduction order stay unchanged. A failed or unsupported fast CPU call drains outstanding Metal work before generic fallback. The drain also runs during exception unwinding, and sticky backend errors fail the pass before result upload or dependent graph submission.

Callers must serialize use of a shared native backend until the complete forward call returns. The ordinary one-model server holds `GpuComputeLock` across the complete inference step, including the CPU work and drain. The single-device check counts configured devices; it does not exclude another caller. Separate model instances and raw native callers sharing a backend need one external lock. This option does not add concurrent-model support to the existing Metal bridge.

`[HOSTMOE-METAL-OVERLAP]` reports eligibility, rejected capabilities, cached cuts, submitted nodes, failures, fallbacks and aggregate wall timings. These timings include host encoding and waits; they are not GPU-device durations. A run with no submissions does not establish an overlap speedup.

The model-free `host-moe-metal-overlap-plan-drain` CTest covers varied hidden/shared dimensions and routing counts, pending-result representations, graph/fusion boundaries, allocation aliases and drain lifetime/error handling. Full-model qualification must use identical weights, quantization, placement, context, CPU thread settings, prompts and generation settings. Validate actual vocabulary widths and finite full logit rows before comparing repeated interleaved latency measurements. Generic naming does not establish support or a performance benefit for every model.

## Reproduce correctness

Build `eng/ForcedLogitProbe` and put the matching `libGgmlOps.dylib` beside its DLL. Then run a dry plan using existing weights:

```sh
python3 eng/validation/moe-metal-overlap-quality.py \
  --model '<existing-GGUF-or-first-shard>' \
  --host-layers 4 --steps 64 --expect-path active \
  --output artifacts/moe-metal-quality-new
```

Add `--execute` to launch the two sequential model runs. The tool never builds or downloads weights and refuses an existing output directory. It holds expert and generic CPU threads at 12, context at 8192, F16 KV, prefill chunk 512, speculation off and the host-to-device prefill threshold at 128. Host-layer count is explicit. `--expect-path fallback` checks that overlap remains inactive; use zero host layers for a dense model. Built-in prompts are five plaintext numeric-equivalence probes. `--cases` accepts the same five named cases with model-specific chat formatting when needed.

The candidate must match every full-vocabulary float32 file byte for byte, with independently finite rows, identical argmax histories and actual asynchronous engagement for an active lane. The probe's relative-L2 tolerance alone is insufficient. Source, binary, model-file statistics, unchanged ggml revision and requested versus observed path are recorded. Generated vectors use `2 * 5 * steps * vocab_size * 4` bytes, potentially over 1 GB. Quality-run duration includes vector capture and comparison and is not inference performance evidence.

Run model-free oracle checks with:

```sh
python3 eng/validation/tests/test_moe_metal_overlap_quality.py
```

The scheduling idea was informed by [KTransformers' CPU-expert/shared-GPU/join sequence](https://github.com/kvcache-ai/ktransformers/blob/c9873312643b615a717832946fba0346e69ab035/archive/ktransformers/operators/experts.py#L742-L749) and [MLX's explicit command-buffer completion and lifetime handling](https://github.com/ml-explore/mlx/blob/bd63d3803008edda03ce74080955d9879e2ef1eb/mlx/backend/metal/device.cpp#L456-L587). This implementation uses TensorSharp-owned code and the unchanged pinned ggml backend; it does not import either framework's kernels.

## Observed validation scope

Measured on 2026-11-01 using an 18-CPU Apple M5 Pro with 48 GiB unified RAM, macOS 27.2, .NET 10.0.10/SDK 10.0.302, runtime source `cd205c80f5d27f90cb54289aa0e65da8e214533b` based on main `2de8a63f`, and unchanged ggml `facdc44a67da17df4433adcc468b424f48e9ded2`. The native library SHA-256 was `0bcdf12e6ec0c2699e7ad1f5048bfff6fbbc1c1cdc108ac5fcd09eac59bf86d8`. Later commits add validation tooling, CI and documentation; runtime arithmetic is unchanged.

All comparisons below switch the same candidate binaries between `0` and `1`. Expert and generic CPU threads are both 12; this is a code comparison at a fixed setting, independent of the optional thread overlay in PR321. No precision, context, sampling or placement is changed between arms. OS/file/shader caches were not flushed. Fresh processes and first repetitions are recorded separately; they are not physical-cold or SSD benchmarks.

| Installed model / GGUF architecture | Fixed host layers | Full logit rows per arm | Actual candidate path | Result |
| --- | ---: | ---: | --- | --- |
| Qwen3.8 Flash Next UD-Q2_K_XL / `qwen4exp`, three shards, 78,869,128,864 bytes | 33 of 48 | 640 × 248,320 | 20,955 async calls; zero rejection/failure/fallback | Every file byte-identical |
| Qwen3.6-35B-A3B IQ4_XS / `qwen35moe`, 18,728,777,856 bytes | 4 of 40 | 320 × 248,320 | 1,260 async calls; zero rejection/failure/fallback | Every file byte-identical |
| Gemma 4 E2B Q8_0 / dense `gemma4`, 4,967,497,152 bytes | 0 | 80 × 262,144 | Inactive; no host-expert work | Every file byte-identical |
| Gemma 4 26B-A4B QAT UD-Q4_K_XL / `gemma4`, 14,249,045,120 bytes | 1 of 30 | 80 × 262,144 | 75 decode cuts rejected at the fusion boundary, 5 non-decode calls ineligible; zero async/failures | Every file byte-identical |

These are 1,120 complete vocabulary rows per arm across Chinese, English, arithmetic, code and JSON inputs. The Qwen4 suite uses recorded chat-formatted prompts; the other suites use the tool's plaintext prompts. They establish numerical equivalence for those histories, not an accuracy benchmark or proof for every possible input. Explicit host placement on the smaller Qwen3.6 model exercises the host path; it is not a recommendation to offload a model that fits entirely on the GPU.

Two separate Qwen4 HTTP series used orders `0,1,1,0,1,0,0,1` and `1,0,0,1,0,1,1,0`. Each launched eight fresh servers and made two rounds of arithmetic, Chinese and JSON requests: 48 requests per series, with four processes per arm. Greedy sampling, repeat penalty 1, `max_tokens=128`, thinking/speculation/skills/agents off, context 8192, F16 KV, chunk 512, prefill threshold 128 and 33 host / 15 Metal expert layers were fixed. All 96 answers, reasoning, usage, finish and cache checks match; cached tokens were zero. The 87-token Chinese answer, 11-token JSON and 4-token arithmetic answer keep output lengths identical. TTFT means first nonempty content delta; end-to-end includes stream completion. Post-first-delta throughput includes transport and is not native model throughput.

| Repeated-request medians, off → on | First series | Reverse-order confirmation |
| --- | --- | --- |
| Chinese TTFT | 2.351 → 2.340 s | 2.275 → 2.081 s |
| Chinese end-to-end | 10.253 → 9.877 s (3.66% lower) | 9.729 → 9.219 s (5.24% lower) |
| Chinese post-first-delta throughput | 11.00 → 11.31 token/s | 11.54 → 11.95 token/s |
| JSON TTFT | 3.017 → 3.355 s | 2.764 → 2.826 s |
| JSON end-to-end | 4.567 → 4.872 s (6.67% higher) | 4.261 → 4.231 s (0.71% lower) |
| Arithmetic end-to-end | 2.177 → 2.077 s | 2.116 → 2.059 s |

Chinese repeated-request end-to-end ranges overlap: first series 9.950–10.580 s off versus 9.708–10.266 s on. Adjacent-pair latency reductions were +5.657%, +5.567%, −0.392%, +1.781%; confirmation pairs were −4.374%, +6.470%, +3.983%, +1.704%. The confirmation is retained separately, not pooled into one speedup. First-repetition Chinese medians were 8.915 → 8.520 s and 8.301 → 8.063 s; first arithmetic requests, warmup and cache drift remain in the ignored reports. These observations support a small, workload-dependent opt-in, not a guaranteed request-latency improvement. Short requests, TTFT and individual pairs can regress.

CLI comparisons used a balanced `0,1,1,0` process order and exact fixed/random token streams, including untimed greedy-history and first-vector checks. First timed iterations are retained separately from the warmed figures below. All comparisons passed their output checks.

| Warm CLI result, off → on | Median and sample scope |
| --- | --- |
| Qwen4 fixed, pp128/tg64 decode | 21.4 [21.3–21.4] → 21.7 [21.5–21.7] token/s; four warm iterations per arm, two processes per arm |
| Qwen4 random, pp128/tg64 decode | 18.75 [18.5–19.0] → 19.35 [19.3–19.4] token/s; two warm iterations per arm, two processes per arm |
| Qwen4 random, pp512/tg64 decode | 19.8 [19.8–19.8] → 20.0 [19.9–20.1] token/s; two warm iterations per arm, two processes per arm |
| Qwen4 random, pp512 prefill | 167.75 [164.5–171.0] → 167.55 [165.7–169.4] token/s; no observed prefill benefit |
| Qwen3.6 random, pp128/tg64 decode | 69.9 [69.8–70.4] → 69.75 [69.7–69.8] token/s; four warm iterations per arm, two processes per arm; no observed benefit |

The first Qwen4 HTTP series' process peak RSS medians were 16.490 GiB off and 16.428 GiB on, ranges 16.265–16.570 and 16.276–16.764 GiB. Confirmation medians were 16.740 and 16.804 GiB, ranges 16.665–16.948 and 16.714–16.869 GiB. These `/usr/bin/time -l` values are process RSS, not total unified/GPU memory; they show no consistent memory reduction. Qwen3.6 CLI RSS ranged 2.805–2.806 GiB off and 2.805–3.021 GiB on. Gemma lanes were numerical/fallback checks, without a performance or peak-memory claim. Real Metal OOM/abort injection and other hardware are not covered.

An earlier overlap series may have spanned a user-reported memory release at 04:45 UTC and is excluded from performance conclusions. The requested prefault retest after that release retained its original safety limits and admitted all 27,984 jobs, but Chinese medians regressed by about 2–4%; that prototype is not included here. File-page/fault counts and touched-page spans do not establish SSD byte traffic.

Full generated manifests, logs, first/subsequent samples and SHA-linked analyses remain in ignored `artifacts/`; they are not repository fixtures. HTTP/CLI capture reused the unchanged [read-only performance helper at `1e1e2e95`](https://github.com/zhongkaifu/TensorSharp/blob/1e1e2e951d96a40e428882a298bdbb38642e12f1/eng/validation/qwen38-metal-performance.py) with an ignored shim setting its checkout root and extending its environment allowlist for the new switch. No PR321 runtime/configuration changes are required. Basic CLI replication needs only this branch's CLI, for example repeat the following with `TS_HOST_MOE_METAL_OVERLAP=0,1,1,0` in separate processes and retain each log:

```sh
TS_HOST_MOE_METAL_OVERLAP=1 TS_CPU_MOE_THREADS=12 TS_GGML_CPU_THREADS=12 \
KV_CACHE_DTYPE=f16 MAX_CONTEXT=8192 TS_PREFILL_CHUNK=512 TS_SPEC=0 TS_HOST_MOE_DEVICE_MIN_BATCH=128 \
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
  --model '<existing-first-GGUF-shard>' --backend ggml_metal --n-cpu-moe 33 \
  --benchmark --bench-prefill 128 --bench-decode 64 --bench-runs 3 \
  --bench-fixed-tokens --bench-chunked
```
