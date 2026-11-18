# LFM3 benchmark tools

TensorSharp against llama.cpp on one machine, with the arms interleaved so a shared machine's drift lands on both.
Outputs belong in the ignored `artifacts/lfm3/`.

Build first (Release, this checkout): `TENSORSHARP_GGML_NO_UPDATE=1 dotnet build eng/lfm3/Lfm3Probe -c Release` and
`dotnet build TensorSharp.Server.Host -c Release`. llama.cpp comes from `LLAMA_CPP_DIR` (default
`~/work/llama.cpp-lfm3`) and its build directory `LLAMA_BUILD` (default `build`, the Metal + BLAS build; for the CPU
lanes a CPU-only tree: `build-cpu` with Accelerate BLAS, llama.cpp's default CPU configuration on a Mac, or
`build-cpu-noblas`, ggml-cpu alone - configure with `-DGGML_METAL=OFF -DGGML_BLAS=ON|OFF`). MLX runs need
`TENSORSHARP_MLX_LIBRARY_DIR` (a directory holding `libmlxc`, e.g. `InferenceWeb.Tests/bin/Release/net10.0`).

- `run-matrix.sh [metal] [mlx] [cpu] [serve]`: the whole matrix (environment knobs in its header): every GPU quant's
  pp512/pp4096 and tg128 at depths 0/4K/16K/32K on ggml_metal, MLX next to ggml_metal and llama.cpp, the CPU lanes
  (ggml_cpu and the managed CPU against `llama-bench -ngl 0` at equal threads) and serving. Results under
  `artifacts/lfm3/bench/<stamp>/`, one markdown table per section plus the raw logs and `conditions.txt` (load
  before/after, commit).
- `ab.py [--rounds N] [--quiet PCT] [--json FILE] NAME=COMMAND ...`: runs the commands alternately (the first arm
  rotating each round) and tabulates each test's median of the rounds' medians with the min-max spread. Commands print
  `RESULT name median min max` lines. `--quiet PCT` waits before each arm (up to `--quiet-wait` seconds) until the
  whole machine is at most PCT % busy over a one-second `top` sample, and records that figure; `--json` keeps every
  round's results, busy figure and load average.
- `ab-merge.py [--arms A,B] [--ratio A/B] SOURCE...`: one table from several `ab.py --json` files or `--log`
  directories (a campaign split over several quiet windows), with the busy figures.
- `ts-bench.sh MODEL BACKEND [PP,..|-] [TG] [DEPTH,..] [--reps N] [--greedy] [--threads N]`: `Lfm3Probe bench`,
  llama-bench's tests: ppN is one prompt of N tokens (TensorSharp's `ForwardRefill`), tgN@dD decodes N pseudo-random
  tokens after a D-token prompt (no sampling in the timed loop). It also prints the K/V footprint (bytes a token and the
  sliding-window rings). `MAX_CONTEXT` must cover the deepest test. `--threads N` runs the CPU lanes on N threads as
  `llama-bench -t N` does: `TS_GGML_CPU_THREADS` (ggml_cpu's pool, default every physical core), `TS_CPU_THREADS`
  (the managed pool, default half the logical processors above 8) and `DOTNET_PROCESSOR_COUNT` all become N. `PROBE`
  points at another build's `Lfm3Probe.dll` (an A/B arm's copied bin directory).
- `llama-bench-result.sh MODEL [PP,..|-] [TG] [DEPTH,..] [REPS] [-- llama-bench args]`: llama-bench (flash attention
  `FA`, default 1; its defaults otherwise: n_batch 2048, n_ubatch 512, threads = the performance cores) with the same
  RESULT lines, and an INFO line per test (backends, threads, flash attention, batch sizes, K type, samples). For the
  CPU pass `-- -ngl 0 -t N` with a CPU-only `LLAMA_BUILD`.
- `serve-bench.py --engine {ts,llama} --model GGUF ...`: TensorSharp.Server.Host vs llama-server with one client over
  streamed OpenAI chat completions: concurrency 1/4/16, prompts of 128/1024/4096 tokens (cut with llama-tokenize, a
  unique first line per request), 256 output tokens (`ignore_eos`, greedy, no repetition penalty, thinking off). TTFT
  p50/p95, per-request decode tok/s, aggregate tok/s and the server's peak physical footprint (`footprint`, Metal
  buffers included; mapped weights are clean pages and not in it). llama-server runs `-np C` with continuous batching,
  TensorSharp its scheduler with `TS_SCHED_MAX_RUNNING_SEQS=C`. CPU runs: `--backend ggml_cpu|cpu` (llama-server gets
  `-ngl 0`), `--threads N` (llama-server `-t N -tb N`, TensorSharp as `ts-bench.sh --threads`), `--tag` to name each
  run's rows and files, `--waves` for fewer requests per level on a slow lane.
- `cpu-matrix.sh QUANT pp|tg [ROUNDS] [DEPTHS]`: TensorSharp ggml_cpu against llama.cpp CPU for one quant, as the CPU
  baseline lays it out (prompts at 18 threads, decode at 12, flash attention on, `ab.py --quiet`): llama.cpp noblas,
  noblas `--repack 0` (its weights placed as TensorSharp places them) and optionally the BLAS build; `TS_ARMS` adds
  TensorSharp A/B arms (`'ts-new=PROBE=/path/Lfm3Probe.dll'`). Results under `artifacts/lfm3/cpu/<stamp>-...`.
- `cpu-profile.sh [QUANT] [T]`: the CPU attribution in one run (per-op profiles of both engines, `sample` profiles of
  decode and a long prompt), reduced into `summary.txt` under `artifacts/lfm3/cpu/<stamp>-profile-...`.
- CPU attribution:
  - `llamacpp-op-profile.cpp` (build line in its header, against a CPU-only tree): llama.cpp's time per op, product
    shape and weight buffer (`@CPU_REPACK` or not) for one prompt and decode steps, every node computed and synchronized
    on its own through the scheduler's eval callback. TensorSharp's side is `TS_GGML_NODE_PROFILE=shapes` with
    `TS_GGML_NODE_PROFILE_TAG=prefill|decode` (ggml_ops_core.cpp), reduced by `node-profile-diff.py LOG [--first N]`
    (the graphs after the warm-up's report). The per-node synchronize inflates small decode nodes, much more on
    llama.cpp's side (its decode step profiles at ~4x its real time; TensorSharp's at ~1x), so compare products.
  - `sample-top.py SAMPLE.txt`: a macOS `sample PID SECONDS 1 -file SAMPLE.txt` profile of an unmodified run, grouped by
    leaf function (dot kernels, repacked GEMV/GEMM, flash attention, thread-pool spinning, idle threads, host code).
  - `speedscope-top.py FILE [--match REGEX]`: self / inclusive time per frame of `dotnet-trace collect --profile
    dotnet-sampled-thread-time --format Speedscope` for the managed lane.
- `metal-op-histogram.py LOG [--start REGEX]`: Metal dispatches per graph from a `GGML_METAL_GRAPH_DEBUG=1` log
  (TensorSharp also needs `TS_GGML_LOG_DEBUG=1`; llama-bench needs `-v`). Graph starts: TensorSharp
  `'^(GET_ROWS|RMS_NORM) node_0$'`, llama.cpp `'^RMS_NORM norm-0$'`.

Machine discipline: run with nothing else on the GPU (`uptime` load under ~2, no other llama-server or test run),
interleave (ab.py does), repeat at least three rounds, and quote the spreads. On the CPU lanes another process on any
core stalls a thread pool that spins at barriers (18 threads collapsed to 25-100 tok/s decode under a test run), so
use `ab.py --quiet` and compare the busy figures.
