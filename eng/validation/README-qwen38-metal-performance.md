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

## Evidence status

**Final-head revalidation on base `2de8a63f` is pending.** The measurements
below are earlier configuration experiments on `72e0fc81`; they are not
validation of the current branch head. Record the actual source revision,
unchanged ggml revision and binary hashes from each new harness report before
publishing a current-head result.

The earlier HTTP ABBA comparison ran baseline/candidate/candidate/baseline
sequentially, with two repetitions of three prompts per server: 24 requests.
All paired answers and token-usage records matched exactly. The arithmetic
request's complete 248,320-element first-token F32 logit vector was also
bit-identical. This checks these requests, not broader language/model quality.

Earlier request medians, milliseconds:

| Workload | Auto nine: TTFT | Twelve: TTFT | Auto nine: end to end | Twelve: end to end |
| --- | ---: | ---: | ---: | ---: |
| Three-sentence Chinese explanation | 4,693 | 3,706 | 14,555 | 12,829 |
| Exact JSON object | 7,183 | 5,593 | 9,285 | 7,512 |

End-to-end latency was 11.9% lower for the Chinese request and 19.1% lower for
the JSON request in that experiment. Those are configuration results for the
specified cases. A separate warmed fixed-work pp128/tg64 comparison measured
auto-nine decode median 20.76 tok/s (range 19.60–20.87) and twelve-thread median
21.16 tok/s (21.00–21.45), about 1.9% higher. It does not establish a general
decode improvement or a kernel speedup.

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
