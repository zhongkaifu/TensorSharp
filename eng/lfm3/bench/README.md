# LFM3 benchmark tools

TensorSharp against llama.cpp on one machine, with the arms interleaved so a shared machine's drift lands on both.
Outputs belong in the ignored `artifacts/lfm3/`.

Build first (Release, this checkout): `TENSORSHARP_GGML_NO_UPDATE=1 dotnet build eng/lfm3/Lfm3Probe -c Release` and
`dotnet build TensorSharp.Server.Host -c Release`. llama.cpp comes from `LLAMA_CPP_DIR` (default
`~/work/llama.cpp-lfm3`, Metal + BLAS build). MLX runs need `TENSORSHARP_MLX_LIBRARY_DIR` (a directory holding
`libmlxc`, e.g. `InferenceWeb.Tests/bin/Release/net10.0`).

- `run-matrix.sh [metal] [mlx] [cpu] [serve] [ttft]`: the whole matrix (environment knobs in its header): every GPU
  quant's pp512/pp4096/pp16384/pp32768 (`PP`) and tg128 at depths 0/4K/16K/32K on ggml_metal, MLX next to ggml_metal
  and llama.cpp, the CPU lanes (ggml_cpu and the managed CPU against `llama-bench -ngl 0` at equal threads), serving,
  and the servers' time to first token (`ttft.py`, 16 to 32K-token prompts). Results under
  `artifacts/lfm3/bench/<stamp>/`, one markdown table per section plus the raw logs and `conditions.txt` (load
  before/after, commit). `llama-bench -ngl 0` on a Metal + BLAS build runs its prompt products on Accelerate (the
  `MTL,BLAS` backend column; `--repack 0` moves even more of them there), not on ggml-cpu alone.
- `ab.py [--rounds N] NAME=COMMAND ...`: runs the commands alternately (the first arm rotating each round) and
  tabulates each test's median of the rounds' medians with the min-max spread. Commands print
  `RESULT name median min max` lines.
- `ts-bench.sh MODEL BACKEND [PP,..|-] [TG] [DEPTH,..] [--reps N] [--greedy]`: `Lfm3Probe bench`, llama-bench's
  tests: ppN is one prompt of N tokens (TensorSharp's `ForwardRefill`), tgN@dD decodes N pseudo-random tokens after a
  D-token prompt (no sampling in the timed loop). It warms 40 prompt steps and 64 decode steps first, so the host code
  the timed steps run is past .NET's first (tier-0) compilation, as in a server that has run a while. It also prints the K/V footprint (bytes a token and the sliding-window
  rings). `MAX_CONTEXT` must cover the deepest test.
- `llama-bench-result.sh MODEL [PP,..|-] [TG] [DEPTH,..] [REPS] [-- llama-bench args]`: llama-bench (flash attention
  on, its defaults otherwise: n_batch 2048, n_ubatch 512, 6 threads) with the same RESULT lines.
- `serve-bench.py --engine {ts,llama} --model GGUF ...`: TensorSharp.Server.Host vs llama-server with one client over
  streamed OpenAI chat completions: concurrency 1/4/16, prompts of 128/1024/4096 tokens (cut with llama-tokenize, a
  unique first line per request), 256 output tokens (`ignore_eos`, greedy, no repetition penalty, thinking off). TTFT
  p50/p95, per-request decode tok/s, aggregate tok/s and the server's peak physical footprint (`footprint`, Metal
  buffers included; mapped weights are clean pages and not in it). llama-server runs `-np C` with continuous batching,
  TensorSharp its scheduler with `TS_SCHED_MAX_RUNNING_SEQS=C`.
- `ttft.py --server NAME=URL ... [--lengths 16,128,..] [--reps N] [--tokenizer-url LLAMA_URL]`: time to first token
  from the client over streamed chat completions (max_tokens 1, greedy, thinking off), the servers' requests
  interleaved, every prompt opened by its own nonce so no prefix cache serves it (llama-server's `prompt_n` is recorded
  to show it). A thinking-off TensorSharp stream holds a reply's text until 4,096 chars or its end (see
  docs/models/lfm3.md), so with more output tokens its client-side TTFT is the reply's end: max_tokens 1 measures the
  engines.
- `gpu-bench.sh [QUANT ...]` (a Linux GPU machine): the single-sequence matrix of TensorSharp's ggml_cuda, direct cuda and
  ggml_vulkan lanes against llama-bench's CUDA and Vulkan builds with flash attention on and off (`LLAMA_CPP` holds
  `build-cuda/` and `build-vk/`), pp512/4096/16384/32768 and tg128 at depths 0/4K/16K/32K by default, every arm
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
- `llama-bench-result.sh` takes `LLAMA_BENCH` (the binary, e.g. a `build-cuda` tree's) and `FA` (llama-bench's `-fa`).
- `metal-op-histogram.py LOG [--start REGEX]`: Metal dispatches per graph from a `GGML_METAL_GRAPH_DEBUG=1` log
  (TensorSharp also needs `TS_GGML_LOG_DEBUG=1`; llama-bench needs `-v`). Graph starts: TensorSharp
  `'^(GET_ROWS|RMS_NORM) node_0$'`, llama.cpp `'^RMS_NORM norm-0$'`.

Machine discipline: run with nothing else on the GPU (`uptime` load under ~2, no other llama-server or test run),
interleave (ab.py does), repeat at least three rounds, and quote the spreads.
