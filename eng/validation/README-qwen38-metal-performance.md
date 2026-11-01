# Qwen3.8 Flash Next: M5 Pro thread tuning

The optional [M5 Pro overlay](../../config/qwen3.8-flash-next-m5-pro.json)
sets only `cpu-moe-threads` to 12. It tunes the existing host expert execution;
it does not change kernels, model precision, context, expert placement or the
default thread policy. The setting is process-wide, including when supplied
by a model-specific launch configuration.

The measurement target is an Apple M5 Pro with 18 logical CPUs and
48 GiB RAM, running the installed three-shard Qwen3.8-Flash-Next UD-Q2_K_XL
checkpoint (78,869,128,864 bytes total) through `ggml_metal`. Text-only tests
fix 33 expert layers on the host and 15 on Metal, F16 KV, context 8192,
prefill chunk 512, generic GGML CPU threads 12, temperature 0 and speculation
off. This is hybrid
host/Metal execution. A different Mac, model quantization, memory pressure,
placement or request mix requires another comparison.

## Use the optional overlay

Use existing local weights; keep all three shards beside one another. These
commands contain no model download instructions. Replace the placeholder with
the installed first shard, and run from the repository root:

```sh
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release

TS_GGML_CPU_THREADS=12 dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
  --model /path/to/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf \
  --backend ggml_metal --mmproj none \
  --config config/qwen3.8-flash-next-m5-pro.json --chat

TS_GGML_CPU_THREADS=12 dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /path/to/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf \
  --backend ggml_metal --mmproj none \
  --config config/qwen3.8-flash-next-m5-pro.json \
  --host 127.0.0.1 --port 8080
```

Alternatively, merge it after a configuration that already names your local
model: `--config /path/to/local-model.json --config
config/qwen3.8-flash-next-m5-pro.json --backend ggml_metal --mmproj none`.
Later configuration files override earlier ones; explicit command-line options
override every configuration file. The standard
`config/qwen3.8-flash-next.json` contains download specifications, so inspect
those separately before using it as the first configuration.

Without an override, the native host expert policy chooses nine total
threads on this 18-CPU Mac: eight workers plus the calling thread. Twelve
means eleven workers plus the caller. `TS_GGML_CPU_THREADS` configures the
separate generic CPU backend; it does not set the host expert team size.
`TS_CPU_MOE_THREADS=0` means auto. Use fresh processes when comparing settings:
removing an environment variable does not reset an override installed earlier
in the same process. The existing sleeping worker policy remains in use.

## Measured configuration results

The 2026-11-01 revalidation used source `ca9940f2`, based on main
`2de8a63fa048d3a9bf25b04923aa124f745a7a10`, on macOS 27.2 and .NET 10.0.10.
Both configurations used the same binaries. The managed CLI and server were
rebuilt against this main base. The native library was reused from clean
`72e0fc81`: source comparison found no changes in `TensorSharp.GGML.Native`,
`TensorSharp.Models`, `TensorSharp.Runtime`, `TensorSharp.Core` or
`TensorSharp.Cli` between those revisions. It used unchanged upstream ggml
`facdc44a67da17df4433adcc468b424f48e9ded2`. The reused native SHA-256 was
`0a5eba984397531eee8176a52b61e17a00062feca60de8404aa247557ca5847a`.
This is configuration validation, not validation of a newly compiled native
engine. Later documentation and harness-validation edits do not change the
measured runtime or overlay; the source identifier above names the measured
revision.

The ignored evidence directory is `artifacts/qwen38-metal-performance/`:
`summary.json`, `native-reuse.json`, `cli-abba/report.json` and
`http-abba/report.json` retain the source/config/binary identities, individual
measurements and correctness checks.

The HTTP ABBA comparison ran auto/tuned/tuned/auto sequentially, with two
repetitions of three prompts per server: **24 requests, zero errors**. All
paired complete answers and token-usage records matched exactly; cached-token
usage was zero and every request ended with `stop`. Arithmetic, Chinese and
JSON responses used 4, 87 and 11 completion tokens respectively. The eight
JSON responses also pass a type-strict recheck: `answer` is integer `42` and
`ok` is boolean `true` throughout. All four
arithmetic first-token F32 vectors were validated, with all three comparisons
against the first vector bit-identical across all 248,320 elements. This
checks these deterministic requests; it does not establish broader language
or model-quality equivalence.

HTTP latency in milliseconds, **median [minimum, maximum]**, four samples per
configuration and workload:

| Workload and metric | Auto nine threads | Twelve threads |
| --- | ---: | ---: |
| Arithmetic TTFT | 2,886 [2,561, 4,165] | 2,356 [2,133, 2,669] |
| Arithmetic end to end | 3,367 [3,102, 4,768] | 2,828 [2,586, 3,105] |
| Chinese explanation TTFT | 4,840 [4,323, 5,783] | 3,711 [3,482, 4,115] |
| Chinese explanation end to end | 14,683 [13,760, 15,774] | 12,754 [12,268, 13,313] |
| Exact JSON TTFT | 7,112 [6,662, 7,862] | 5,576 [5,403, 5,782] |
| Exact JSON end to end | 9,265 [8,724, 10,130] | 7,593 [7,152, 7,995] |

Median end-to-end latency was **13.1% lower for the Chinese request** and
**18.0% lower for JSON** in this configuration comparison. Peak server process
RSS was 11.27–11.31 GiB for auto and 11.17–11.30 GiB for twelve threads. RSS
does not measure total unified memory, total GPU memory or other processes;
these overlapping ranges do not establish a memory reduction.

For the 87-token Chinese response, post-first-content SSE throughput was
8.78 [8.53, 9.11] versus 9.54 [9.31, 9.79] tok/s (auto versus twelve). This
transport estimate is a different workload from the fixed-token CLI benchmark
and does not measure native kernel throughput.

The CLI fixed-work pp128/tg64 ABBA comparison completed 16 timed iterations
without errors. Each process ran four iterations after standard startup
warmup; the warm summary excludes the first iteration of each process,
leaving six warm samples per configuration:

| Warm throughput, tok/s | Auto nine threads | Twelve threads |
| --- | ---: | ---: |
| Prefill | 121.12 [112.97, 125.61] | 120.53 [98.08, 123.31] |
| Decode | 20.61 [19.13, 20.77] | 20.82 [20.54, 20.99] |

The approximately 1% decode-median difference and overlapping ranges do not
establish a general decode improvement. Prefill was not faster in this
fixed-work comparison. Process-first iterations, retained separately in the
reports, measured prefill 33.8–48.9 versus 47.4–51.2 tok/s and decode 14.6–16.1
versus 15.3–15.7 tok/s (auto versus twelve, two samples each). These are fresh
processes after startup warmup, not physically cold caches. All compared
64-token untimed greedy correctness chains matched, and all compared CLI
first-token vectors were bit-identical.

Earlier experiments on `72e0fc81` also found matching output and lower Chinese
and JSON request latency at twelve threads. The tables above use the later
main-based revalidation; they do not pool samples from the earlier builds.

## Reproduce a comparison

[The harness](qwen38-metal-performance.py) defaults to a plan; add `--execute`
only when the shared Mac/GPU is free. It launches servers sequentially on
loopback and stops only processes it created. Build once, then compare both
configurations using the same binaries and existing model. Keep the generic
GGML CPU count fixed at 12 and host-MoE auto at zero for the baseline; only the
candidate receives the overlay.

```sh
python3 eng/validation/qwen38-metal-performance.py http \
  --variant baseline=. --variant candidate=. \
  --variant-config candidate=config/qwen3.8-flash-next-m5-pro.json \
  --order baseline,candidate,candidate,baseline \
  --model /path/to/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf \
  --env TS_CPU_MOE_THREADS=0 --env TS_GGML_CPU_THREADS=12 \
  --env KV_CACHE_DTYPE=f16 --env TS_SPEC=0 \
  --env TS_HOST_MOE_DEVICE_MIN_BATCH=128 \
  --context 8192 --chunk 512 --cpu-moe 33 --runs 2 --max-tokens 128 \
  --output artifacts/qwen38-metal-threads/http-abba

python3 eng/validation/qwen38-metal-performance.py cli \
  --variant baseline=. --variant candidate=. \
  --variant-config candidate=config/qwen3.8-flash-next-m5-pro.json \
  --order baseline,candidate,candidate,baseline \
  --model /path/to/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf \
  --env TS_CPU_MOE_THREADS=0 --env TS_GGML_CPU_THREADS=12 \
  --env KV_CACHE_DTYPE=f16 --env TS_SPEC=0 \
  --env TS_HOST_MOE_DEVICE_MIN_BATCH=128 \
  --context 8192 --chunk 512 --cpu-moe 33 --cases 128:64 --runs 4 \
  --token-stream fixed \
  --output artifacts/qwen38-metal-threads/cli-abba
```

Review the plan, then repeat the chosen command with `--execute` and a fresh
output directory. Keep reports, logs, vectors and snapshots in ignored
`artifacts/` or `docs/validation/`; do not add generated evidence to Git.

The HTTP report automatically compares complete first-token vectors. For an
optional independent inspection of the arithmetic vector from two processes,
run the comparison command and inspect the remaining ABBA pair as well:

```sh
python3 eng/validation/qwen38-metal-performance.py compare-logits \
  --first artifacts/qwen38-metal-threads/http-abba/00-baseline-http/logits.f32 \
  --second artifacts/qwen38-metal-threads/http-abba/01-candidate-http/logits.f32
```

HTTP TTFT ends at the first nonempty SSE content delta and includes queueing,
request preparation, prefill and transport. End-to-end latency ends when the
response stream completes. Inspect output text, finish reason, token counts
and cached-token usage alongside timings. Post-first-delta throughput is a
transport estimate, not a measured inference-kernel rate. CLI fixed-work
throughput is a different workload; keep it separate from generated text.
Process peak RSS excludes other processes and is not total unified-memory
use. Fresh processes do not imply cold OS/file/shader caches: this procedure
does not flush caches. Compare first and repeated requests separately, retain
ranges and failures, and monitor unrelated memory pressure and concurrent
work. No skipped or unavailable scenario counts as passing validation.

## Investigated alternatives

The investigation compared Metal HC graph fusion, strict-rounding owned HC
kernels, host-expert matrix ordering and a lower streamed-prefill threshold.
They are not part of this overlay. Some changed full-model logits/output;
others did not show a material benefit. Keep the original prefill streaming
threshold and numerical path when reproducing this thread-only comparison.

Primary references informed the investigation: unchanged ggml's
[`ggml_compute_forward_mul_mat_id`](https://github.com/ggml-org/ggml/blob/facdc44a67da17df4433adcc468b424f48e9ded2/src/ggml-cpu/ggml-cpu.c#L1647)
groups work by source expert matrix, and llama.cpp's
[`llama_context::sched_copy_experts`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp#L2540)
coalesces consecutive expert transfers. These are implementation references,
not comparative performance evidence. This configuration copies no upstream
implementation and requires no ggml patches.
