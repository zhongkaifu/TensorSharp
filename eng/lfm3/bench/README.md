# LFM3 benchmark tools

TensorSharp against llama.cpp on one machine, with the arms interleaved so a shared machine's drift lands on both.
Outputs belong in the ignored `artifacts/lfm3/`.

Build first (Release, this checkout): `TENSORSHARP_GGML_NO_UPDATE=1 dotnet build eng/lfm3/Lfm3Probe -c Release` and
`dotnet build TensorSharp.Server.Host -c Release`. llama.cpp comes from `LLAMA_CPP_DIR` (default
`~/work/llama.cpp-lfm3`) and its build directory `LLAMA_BUILD` (default `build`, the Metal + BLAS build; for the CPU
lanes a CPU-only tree: `build-cpu` with Accelerate BLAS, llama.cpp's default CPU configuration on a Mac, or
`build-cpu-noblas`, ggml-cpu alone - configure with `-DGGML_METAL=OFF -DGGML_BLAS=ON|OFF`). MLX runs need
`TENSORSHARP_MLX_LIBRARY_DIR` (a directory holding `libmlxc`, e.g. `InferenceWeb.Tests/bin/Release/net10.0`).

- `run-matrix.sh [metal] [mlx] [cpu] [serve] [ttft] [cpu-serve] [cpu-omni]`: the whole matrix (environment knobs in
  its header): every GPU quant's pp512/pp4096/pp16384/pp32768 (`PP`) and tg128 at depths 0/4K/16K/32K on ggml_metal,
  MLX next to ggml_metal and llama.cpp, serving, and the servers' time to first token (`ttft`: `ttft.py`, 16 to
  32K-token prompts). `cpu` is the CPU lanes' matrix: every LFM3 file (Q4_K_M, Q8_0, QAD-Q4_0, NVFP4, BF16) on
  ggml_cpu and the managed CPU against `llama-bench -ngl 0` from llama.cpp's CPU-only builds (Accelerate BLAS and
  ggml-cpu alone; `llama-bench -ngl 0` on the Metal + BLAS build would run its prompt products on Accelerate, the
  `MTL,BLAS` backend column, not on ggml-cpu alone), prompts of 128-8192 tokens with every engine on all physical cores
  (both TensorSharp lanes' default and llama.cpp's best prompt count), tg128 at depths 0 and 4K (16K for Q4_K_M) with
  TensorSharp at its defaults and at `CPU_THREADS` (12) and llama.cpp at 12 and at `CPU_WIDE_THREADS` (16: ggml_cpu's
  decode width and llama.cpp's best decode count here; 12 is 4-11 % slower at depth 0 and 4K, 21 % on BF16 and at 16K),
  5 decode reps an arm (`CPU_TG_REPS`, odd so each median is one run), each pass `ab.py --quiet 12`; `BASE_PROBE` adds
  an older build as before arms. `cpu-serve` measures TTFT and 1 / 4 streams through both servers against
  llama-server, `cpu-omni` the LFM3-Omni vision and audio encoders and streaming speech on both lanes against
  llama.cpp's CPU build. Results under `artifacts/lfm3/bench/<stamp>/`, one markdown table per section plus the raw
  logs and `conditions.txt` (load before/after, the busiest processes, commit). The CPU sections take about 3 hours on
  an M5 Pro; run them on a machine nothing else uses.
- `ab.py [--rounds N] [--quiet PCT] [--json FILE] NAME=COMMAND ...`: runs the commands alternately (the first arm
  rotating each round) and tabulates each test's median of the rounds' medians with the min-max spread. Commands print
  `RESULT name median min max` lines. `--quiet PCT` waits before each arm (up to `--quiet-wait` seconds) until the
  whole machine is at most PCT % busy over a one-second `top` sample, and records that figure; `--json` keeps every
  round's results, busy figure and load average.
- `ab-merge.py [--arms A,B] [--ratio A/B] SOURCE...`: one table from several `ab.py --json` files or `--log`
  directories (a campaign split over several quiet windows), with the busy figures.
- `ts-bench.sh MODEL BACKEND [PP,..|-] [TG] [DEPTH,..] [--reps N] [--greedy] [--threads N]`: `Lfm3Probe bench`,
  llama-bench's tests: ppN is one prompt of N tokens (TensorSharp's `ForwardRefill`). It warms 40 prompt steps and 64
  decode steps first (fewer prompt steps when `MAX_CONTEXT` is under 5,184; an `INFO warm-up done` line on stderr marks
  the end), so the host code the timed steps run is past .NET's first (tier-0) compilation, as in a server that has run
  a while, and each prompt size by one untimed prompt of that size as llama-bench warms each test (a size's first step
  builds its graphs). The managed lane's kernels keep being tiered up by the JIT over its first few prompts of a size,
  so its short prompts need more than 3 reps for a steady-state figure (LFM3 Q8_0 pp128 on an M5 Pro: 660-918 tok/s
  over 3 reps, 826-957 over 10). tgN@dD decodes N
  pseudo-random tokens after a D-token prompt (no sampling in the timed loop), after one untimed run of N tokens at
  depth 0 (llama-bench warms each test with one token; the managed lane's first decode run was 25-35 % slower than the
  rest while the JIT tiered its kernels up). Each figure is the median of `--reps` runs, the mean of the middle two for
  an even count, as `llama-bench-result.sh` computes llama.cpp's. It also prints the K/V footprint (bytes
  a token and the sliding-window rings). `MAX_CONTEXT` must cover the deepest test. `--threads N` runs the CPU lanes on
  N threads as `llama-bench -t N` does: `TS_GGML_CPU_THREADS` (ggml_cpu's prompt and decode pools; defaults every
  physical core for prompts and every physical core but two for decode on a hybrid Mac), `TS_CPU_THREADS` (the
  managed pool, default every core on Apple silicon, and every core but two for decode on a hybrid Mac) and `DOTNET_PROCESSOR_COUNT` all become N. `PROBE` points at
  another build's `Lfm3Probe.dll` (an A/B arm's copied bin directory).
- `llama-bench-result.sh MODEL [PP,..|-] [TG] [DEPTH,..] [REPS] [-- llama-bench args]`: llama-bench (flash attention
  `FA`, default 1; its defaults otherwise: n_batch 2048, n_ubatch 512, threads = the performance cores) with the same
  RESULT lines, and an INFO line per test (backends, threads, flash attention, batch sizes, K type, samples). For the
  CPU pass `-- -ngl 0 -t N` with a CPU-only `LLAMA_BUILD`.
- `serve-bench.py --engine {ts,llama} --model GGUF ...`: TensorSharp.Server.Host vs llama-server with one client over
  streamed OpenAI chat completions: concurrency 1/4/16, prompts of 128/1024/4096 tokens (cut with llama-tokenize, a
  unique first line per request), 256 output tokens (`ignore_eos`, greedy, no repetition penalty, thinking off; the
  TensorSharp server runs with `--sampling-precedence request`, so a `--config`'s model-card sampling does not override
  the request's greedy settings). TTFT p50/p95, per-request decode tok/s, aggregate tok/s and the server's peak physical
  footprint (`footprint`, Metal buffers included; mapped weights are clean pages and not in it) and peak resident set
  size (`ps -o rss`, which counts those mapped pages too). llama-server runs
  `-np C` with continuous batching, TensorSharp its scheduler with `TS_SCHED_MAX_RUNNING_SEQS=C`. CPU runs:
  `--backend ggml_cpu|cpu` (llama-server gets `-ngl 0`), `--threads N` (llama-server `-t N -tb N`, TensorSharp as
  `ts-bench.sh --threads`), `--tag` to name each run's rows and files, `--waves` for fewer requests per level on a slow
  lane.
- `vllm-cpu-mac.sh [check] [correct] [sweep] [bench] [summary]` (helper `vllm-cpu-mac.py`; the header lists every
  knob): the CPU lanes on an Apple-silicon Mac against vLLM's CPU backend (built from source into a uv venv,
  `VLLM_VENV`, serving the HF checkpoint `HF`) and llama.cpp's CPU build, with one client for every engine (`vllm
  bench serve --backend openai-chat`: random prompts identical across engines, greedy, `ignore_eos`, thinking off,
  prefix/prompt caching off everywhere, one slot per concurrent request, 4096 context). `correct` checks the engines
  agree before anything is timed: greedy generations of five chat prompts (64 tokens) from vLLM offline
  (float16/bfloat16/float32), the vLLM, TensorSharp ggml_cpu / cpu and llama-server servers on the BF16 weights, and
  HF transformers float32 (`generate()`, and the reference: greedy by full forwards without a cache), reported as tokens
  identical before the first divergence, with the reference model's logit gap at each divergence (`margins.md`: near
  tie or bug). `sweep` picks vLLM's thread count (its CPU worker's OpenMP width, `VLLM_CPU_OMP_THREADS_BIND`); `bench`
  runs `ROUNDS` interleaved rounds of the arms (vLLM fp16 and fp32, TensorSharp ggml_cpu and cpu with BF16 and
  Q4_K_M, llama.cpp BF16 and Q4_K_M) over ISL/OSL 512/128 and 128/256 at concurrency 1 and 4; `summary` tabulates
  TTFT and TPOT p50/p99, output tok/s, requests/s and the server process tree's peak `footprint` as medians over
  rounds with min-max spreads. vLLM 0.34's Arm short-conv path is unusable on macOS (a grouped `F.conv1d` that
  PyTorch's macOS wheels, built without oneDNN, run one group at a time: 0.65 tok/s); the `vllm-*fix` arms load
  `vllm-mac-shortconv/sitecustomize.py`, which sends it to vLLM's own `causal_conv1d_update_cpu_vec` kernel
  (outputs bit-identical to stock). On a shared machine set `HEAVY` to a lock wrapper (one group of arms per hold).
  Results under `artifacts/lfm3/cpu/vllm/<stamp>/`.
- `cpu-matrix.sh QUANT pp|tg [ROUNDS] [DEPTHS]`: TensorSharp ggml_cpu against llama.cpp CPU for one quant, as the CPU
  baseline lays it out (prompts at 18 threads, decode at 12, flash attention on, `ab.py --quiet`): llama.cpp noblas,
  noblas `--repack 0` (its weights in their GGUF layout, as TensorSharp places them with `TS_GGML_CPU_REPACK=0`) and
  optionally the BLAS build; `TS_ARMS` adds TensorSharp A/B arms (`'ts-new=PROBE=/path/Lfm3Probe.dll'`). Results under `artifacts/lfm3/cpu/<stamp>-...`.
- Managed CPU lane kernels: `benchmarks/CpuQuantBench shapes TYPES ROWS SHAPES [REPS]` times the managed quantized
  products at LFM3's shapes (`lfm3`, or `KxN,...`) and prints RESULT lines for `ab.py` (weight GB/s at one row with the
  weights rotated past the caches, GOPS above); `TS_CPU_THREADS` sets the pool. `Lfm3Probe ppl` prints a digest of
  every scored logit's bits, which two builds whose kernels claim the same arithmetic must share.
- `cpu-profile.sh [QUANT] [T]`: the CPU attribution in one run (per-op profiles of both engines, `sample` profiles of
  decode and a long prompt, a TensorSharp run sampled after its start-up warm-up), reduced into `summary.txt` under
  `artifacts/lfm3/cpu/<stamp>-profile-...`.
- CPU attribution:
  - `llamacpp-op-profile.cpp` (build line in its header, against a CPU-only tree): llama.cpp's time per op, product
    shape and weight buffer (`@CPU_REPACK` or not) for one prompt and decode steps, every node computed and synchronized
    on its own through the scheduler's eval callback. TensorSharp's side is `TS_GGML_NODE_PROFILE=shapes` with
    `TS_GGML_NODE_PROFILE_TAG=prefill|decode` (ggml_ops_core.cpp), reduced by `node-profile-diff.py LOG [--first N]`
    (the graphs after Lfm3Probe's `INFO warm-up done` line, or after the first report). The per-node synchronize
    inflates small decode nodes, much more on llama.cpp's side (its decode step profiles at ~4x its real time;
    TensorSharp's at ~1x), so compare products.
  - `sample-top.py SAMPLE.txt`: a macOS `sample PID SECONDS 1 -file SAMPLE.txt` profile of an unmodified run, grouped by
    leaf function (dot kernels, repacked GEMV/GEMM, flash attention, thread-pool spinning, idle threads, host code).
  - `speedscope-top.py FILE [--match REGEX]`: self / inclusive time per frame of `dotnet-trace collect --profile
    dotnet-sampled-thread-time --format Speedscope` for the managed lane.
- `ttft.py --server NAME=URL ... [--lengths 16,128,..] [--reps N] [--tokenizer-url LLAMA_URL]`: time to first token
  from the client over streamed chat completions (max_tokens 1, greedy, thinking off), the servers' requests
  interleaved, every prompt opened by its own nonce so no prefix cache serves it (llama-server's `prompt_n` is recorded
  to show it). A thinking-off TensorSharp stream holds a reply's text until 4,096 chars or its end (see
  docs/models/lfm3.md), so with more output tokens its client-side TTFT is the reply's end: max_tokens 1 measures the
  engines.
- `gpu-bench.sh [QUANT ...]` (a Linux GPU machine): the single-sequence matrix of TensorSharp's ggml_cuda, direct cuda and
  ggml_vulkan lanes against llama-bench's CUDA and Vulkan builds with flash attention on and off (`LLAMA_CPP` holds
  `build-cuda/` and `build-vk/`, which `llama-bench-result.sh` takes as `LLAMA_BUILD`), pp512/4096/16384/32768 and tg128 at depths 0/4K/16K/32K by default, every arm
  interleaved; one table per quant under `artifacts/lfm3/bench/gpu-<stamp>/`. It refuses a busy GPU.
- `serve-vs.sh OUTDIR ENGINE ...` and `serve-vs-summary.py DIR ...` (a Linux GPU machine): TensorSharp.Server.Host
  (`ts-cuda`, `ts-ggml_cuda`, `ts-ggml_vulkan`) against vLLM, SGLang and llama-server (`llama-cuda`, `llama-vk`) measured
  by one client, `vllm bench serve --backend openai-chat` (random token prompts, ignore_eos, temperature 0, thinking off),
  over (ISL, OSL) shapes and concurrencies (environment knobs in the header: `SHAPES`, `CONC`, `PREFIX_CACHE`,
  `RANDOM_PREFIX`, `MAX_CONTEXT`, `MAX_SEQS`, `BATCH_TOKENS`, `KV_POOL`); every engine gets the same context,
  running-sequence limit and per-step token budget, prefix caching off unless asked. TensorSharp's KV block pool is sized
  for `MAX_SEQS` full contexts (`KV_POOL=sized`, as vLLM and SGLang size theirs from free memory); `KV_POOL=default`
  keeps the server's own 65,536-token pool, past which running requests wait or are preempted and re-prefilled. One JSON per cell plus the GPU memory in use after it; the
  summary tabulates rounds (median and min-max of each cell's metric over the directories given).
- `metal-op-histogram.py LOG [--start REGEX]`: Metal dispatches per graph from a `GGML_METAL_GRAPH_DEBUG=1` log
  (TensorSharp also needs `TS_GGML_LOG_DEBUG=1`; llama-bench needs `-v`). Graph starts: TensorSharp
  `'^(GET_ROWS|RMS_NORM) node_0$'`, llama.cpp `'^RMS_NORM norm-0$'`.

Machine discipline: run with nothing else on the GPU (`uptime` load under ~2, no other llama-server or test run),
interleave (ab.py does), repeat at least three rounds, and quote the spreads. On the CPU lanes another process on any
core stalls a thread pool that spins at barriers (18 threads collapsed to 25-100 tok/s decode under a test run), so
use `ab.py --quiet` and compare the busy figures.
