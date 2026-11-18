# LFM3 benchmark tools

TensorSharp against llama.cpp on one machine, with the arms interleaved so a shared machine's drift lands on both.
Outputs belong in the ignored `artifacts/lfm3/`.

Build first (Release, this checkout): `TENSORSHARP_GGML_NO_UPDATE=1 dotnet build eng/lfm3/Lfm3Probe -c Release` and
`dotnet build TensorSharp.Server.Host -c Release`. llama.cpp comes from `LLAMA_CPP_DIR` (default
`~/work/llama.cpp-lfm3`, Metal + BLAS build). MLX runs need `TENSORSHARP_MLX_LIBRARY_DIR` (a directory holding
`libmlxc`, e.g. `InferenceWeb.Tests/bin/Release/net10.0`).

- `run-matrix.sh [metal] [mlx] [cpu] [serve]`: the whole matrix (environment knobs in its header): every GPU quant's
  pp512/pp4096 and tg128 at depths 0/4K/16K/32K on ggml_metal, MLX next to ggml_metal and llama.cpp, the CPU lanes
  (ggml_cpu and the managed CPU against `llama-bench -ngl 0` at equal threads) and serving. Results under
  `artifacts/lfm3/bench/<stamp>/`, one markdown table per section plus the raw logs and `conditions.txt` (load
  before/after, commit).
- `ab.py [--rounds N] NAME=COMMAND ...`: runs the commands alternately (the first arm rotating each round) and
  tabulates each test's median of the rounds' medians with the min-max spread. Commands print
  `RESULT name median min max` lines.
- `ts-bench.sh MODEL BACKEND [PP,..|-] [TG] [DEPTH,..] [--reps N] [--greedy]`: `Lfm3Probe bench`, llama-bench's
  tests: ppN is one prompt of N tokens (TensorSharp's `ForwardRefill`), tgN@dD decodes N pseudo-random tokens after a
  D-token prompt (no sampling in the timed loop). It also prints the K/V footprint (bytes a token and the sliding-window
  rings). `MAX_CONTEXT` must cover the deepest test.
- `llama-bench-result.sh MODEL [PP,..|-] [TG] [DEPTH,..] [REPS] [-- llama-bench args]`: llama-bench (flash attention
  on, its defaults otherwise: n_batch 2048, n_ubatch 512, 6 threads) with the same RESULT lines.
- `serve-bench.py --engine {ts,llama} --model GGUF ...`: TensorSharp.Server.Host vs llama-server with one client over
  streamed OpenAI chat completions: concurrency 1/4/16, prompts of 128/1024/4096 tokens (cut with llama-tokenize, a
  unique first line per request), 256 output tokens (`ignore_eos`, greedy, no repetition penalty, thinking off). TTFT
  p50/p95, per-request decode tok/s, aggregate tok/s and the server's peak physical footprint (`footprint`, Metal
  buffers included; mapped weights are clean pages and not in it). llama-server runs `-np C` with continuous batching,
  TensorSharp its scheduler with `TS_SCHED_MAX_RUNNING_SEQS=C`.
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
- `metal-op-histogram.py LOG [--start REGEX]`: Metal dispatches per graph from a `GGML_METAL_GRAPH_DEBUG=1` log
  (TensorSharp also needs `TS_GGML_LOG_DEBUG=1`; llama-bench needs `-v`). Graph starts: TensorSharp
  `'^(GET_ROWS|RMS_NORM) node_0$'`, llama.cpp `'^RMS_NORM norm-0$'`.

Machine discipline: run with nothing else on the GPU (`uptime` load under ~2, no other llama-server or test run),
interleave (ab.py does), repeat at least three rounds, and quote the spreads.
