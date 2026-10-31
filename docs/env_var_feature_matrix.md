# Environment Variable x Feature Matrix

[English](env_var_feature_matrix.md) | [中文](env_var_feature_matrix_zh-cn.md)

This document is the curated runtime-flag reference used by
[`TensorSharp.TestMatrix`](../TensorSharp.TestMatrix/README.md). It focuses on
environment variables that materially change correctness, throughput, memory
use, or model routing for real inference workloads.

The code source of truth is
[`TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs`](../TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs).
The default sweep list is configured in
[`TensorSharp.TestMatrix/Defaults/matrix-config.json`](../TensorSharp.TestMatrix/Defaults/matrix-config.json).

## How TestMatrix Uses This

- Every applicable `(model, backend, feature)` cell first runs a **baseline**
  case with no forced sweep variable.
- For each selected env var, the runner creates one case per listed value and
  passes only that variable to the `TensorSharp.Cli` subprocess.
- Before each subprocess starts, inherited `TS_*`, `GDN_*`, `QWEN35_*`,
  `FUSED_*`, `KV_CACHE_DTYPE`, `MAX_CONTEXT`, `MAX_TOKENS`,
  `VIDEO_MAX_FRAMES`, and `VIDEO_SAMPLE_FPS` variables are scrubbed so the
  matrix value is authoritative.
- `--env-vars none` disables sweep cases. If a config file has an empty
  `default_env_vars` list and the CLI does not override it, the runner uses all
  registered `EnvVarMatrix.All` entries — each still only where it applies.

The "Runtime baseline" column below describes the behavior when the variable is
unset. The "Swept by default" column describes the current default config, not
the full set of registered variables.

DiffusionGemma is currently outside the registered TestMatrix feature catalog:
there is no diffusion prompt type, no diffusion-specific env sweep, and inherited
`DIFFUSION_*` variables are not scrubbed by the runner. Use explicit model
configs plus a dedicated feature/env registration before treating diffusion
results as part of the standard matrix.

## Continuous Batching / Batched Forward

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE` | Nemotron-H | Native batched Mamba2 step | OFF | `0`, `1` | no |
| `TS_PER_SEQ_FUSED` | fused-capable models (Gemma 4, GPT OSS and Qwen 3.8 Flash Next on GGML backends; Qwen 3.5/3.6/3.8 on `ggml_cuda` / `ggml_metal`; DeepSeek V4 / V4.1 and GLM 5.x on their native executors) | Per-request fused Forward for concurrent (N>=2) sequences, each on its own device K/V holder; `0` serves them on the op-by-op batched paged path, whose K/V lives in the shared host block pool (less device memory, a lower decode rate, no retained holders) | ON | not registered | no |
| `TS_BATCHED_FUSED_DECODE` | models with a token-batched fused decode (Gemma 4, Qwen 3.5/3.6/3.8, GPT OSS, GLM 5.x, DeepSeek V4 / V4.1) | True token-batched fused decode inside the per-seq fused path (one graph for all N). On GLM 5.x this is 1.81x aggregate decode at 4 concurrent requests; on DeepSeek V4.1 Flash at Q4_K_M it is 2.0x (24.3 → 48.9 tok/s), bounded by routing because each token picks its own 6 of 384 experts. Batching changes GEMM shapes and a 2-bit MoE can turn that into different expert picks; set `0` for a serial-path A/B. | ON | not registered | no |
| `TS_BATCHED_FUSED_MOE` | Gemma 4 MoE | `1` lets Gemma 4 MoE checkpoints take the token-batched fused decode. Off by default: its capture-safe graph, next to the KV holders, filled a 16 GB card and was not faster than round-robin there | OFF | not registered | no |
| `TS_RETAINED_FUSED_CACHE_MAX` | models with retainable request-owned fused holders (Gemma 4; Qwen 3.5/3.6/3.8; Qwen 3.8 Flash Next, whose byte budget is `TS_Q4E_RETAINED_CACHE_MB`; DeepSeek V4.1 unless a DSpark drafter is loaded; GLM 5.x on its native executor, a donated native slot per conversation) | How many finished conversations' end states (holders, native slots) stay retained for an exact-prefix continuation on their next turn; Qwen's holder includes attention K/V and matching GatedDeltaNet recurrent state. Unset, it follows `TS_SCHED_MAX_RUNNING_SEQS` (at least 4), so every request that can run in parallel keeps its own; memory, not this count, bounds the bytes. `0` turns retention off | the running-request limit (16), at least `4` | n/a | no |
| `TS_PREFIX_CHECKPOINTS_MAX` | Gemma 4 on GGML backends; Qwen 3.5/3.6/3.8 on `ggml_cuda` / `ggml_metal` / `mlx`, the backends where it runs per-request holders (not under TP); Qwen 3.8 Flash Next on its GGML token-span path, including under the `--layer-split N` layer split. Needs `TS_PER_SEQ_FUSED` on | Checkpoint the model's complete state where the prompt every conversation shares ends (system prompt, tools, skills) and start each NEW chat from a clone of it, so a new chat re-prefills only its own message. The value is how many distinct shared prefixes stay checkpointed (each holds one copy of the prefix's K/V and, for Qwen, recurrent state); `0` turns checkpoints off. It is the budget of public checkpoints the Radix prefix cache keeps; a prompt publishes one per boundary (system instructions, whole shared prefix), so 4 covers a host that warms both thinking modes | `4` | n/a | no |
| `TS_KV_INITIAL_TOKENS` | families that size their cache through `ModelBase.ResolveInitialCacheAllocationLength` (Qwen 3.5/3.6, Gemma 4, GPT-OSS and the other ModelBase families; not DeepSeek V4 / GLM 5.x, which size their own) | Tokens of K/V a cache is given when created (the primary cache at load, every per-request holder) before any request declares a budget; `0` keeps the engine policy (whole window when `MAX_CONTEXT` is explicit, else a backend default). The cache still grows on demand. Memory-constrained devices set this small because every kept holder is paid at this size, host and device mirror both | `0` | n/a | no |
| `TS_KV_GENERATION_RESERVE_MAX` | all | Cap on the generation share of the K/V a request reserves up front (prompt + max_new_tokens); a reply limit at or above the window otherwise reserves the whole window per request. Past the cap the cache grows on demand. `0` = no cap | `0` | n/a | no |
| `TS_KV_HOLDER_POOL_MAX` | models with per-request fused holders (Qwen 3.5/3.6/3.8, Gemma 4, GPT-OSS) | How many released holders may be parked for reuse instead of freed; each parked holder costs its whole K/V allocation | `64` | n/a | no |
| `TS_SCHED_DISABLE_BATCHED` | all | `1` forces the per-sequence KV-swap path for every model, speculation included (`--no-continuous-batching`) | OFF | `0`, `1` | yes |
| `TS_SCHED_PREFIX_CACHE` | all autoregressive models | `0` disables every form of admission-time prompt reuse (the radix prefix cache: pages, retained end states and public checkpoints in one index). `--no-prefix-cache` (CLI and server) sets it and also skips the startup warm-up of the shared prompt and, on the server, the checkpoint files | ON (`1`) | not registered | no |
| `TS_SCHED_MAX_BATCHED_TOKENS` / `TS_SCHED_MAX_RUNNING_SEQS` / `TS_SCHED_PREFILL_CHUNK` / `TS_SCHED_SOLO_PREFILL_CHUNK` / `TS_SCHED_NUM_BLOCKS` / `TS_SCHED_BLOCK_SIZE` / `TS_SCHED_DECODE_QUANTUM` | all | Scheduler budgets: per-step tokens, in-flight sequences, prefill chunk while a decode runs (server `--prefill-chunk-size`), solo prefill chunk, pool blocks, tokens per block (16 for a family whose only cross-request reuse is its pages, else 256), decode quantum. A model with its own lone-prompt steps — Muse Spark with pipelined prefill: 8192 — sizes them unless `TS_SCHED_SOLO_PREFILL_CHUNK` is set; a set value applies as documented and turns the model's steps off, while `TS_SCHED_MAX_BATCHED_TOKENS` alone does not. See the [configuration table](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md#configuration) | `4096` / `16` / `256` / `8192` / `256` / `16` or `256` / `256` | not registered | no |
| `TS_SCHED_STOP_REPETITION` | all | `0` lets a generation that has locked into a loop run to its token limit instead of ending with finish reason `repetition` | ON (`1`) | not registered | no |

The executor-level switches in this section (`TS_SCHED_DISABLE_BATCHED`,
`TS_PER_SEQ_FUSED`, `TS_BATCHED_FUSED_DECODE`, the retained-holder, checkpoint
and `TS_KV_*` budgets) are read through
`ExecutionOptions.FromEnvironment()` and consumed by `ExecutionPlanner`
(see `docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md`, "Execution Planning");
the `TS_SCHED_*` values are read by `SchedulerConfig.FromEnvironment()`; the MoE switches are
read by the model or its native kernel.

TensorAgent writes its own values before every model load
(`EngineMemoryPolicy`): `TS_KV_INITIAL_TOKENS=2048`,
`TS_KV_GENERATION_RESERVE_MAX=1024`, `TS_KV_HOLDER_POOL_MAX=0`,
`TS_RETAINED_FUSED_CACHE_MAX=1`, and `MAX_CONTEXT` / `KV_CACHE_DTYPE` from the
catalog entry or the user's setting. The iOS app (`TensorAgent.Maui`) also defaults
`TS_SCHED_SOLO_PREFILL_CHUNK=1024`, `TS_PREFIX_CHECKPOINTS_MAX=1` and
`GGML_METAL_NO_RESIDENCY=1` when they are not already set.

## KV Cache / Context

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `KV_CACHE_DTYPE` | all except the DeepSeek V4 / V4.1 executors, which keep F16 caches read by their own attention, gather and compressor kernels: `q8_0` / `q4_0` are **refused at load** with the reason (`NotSupportedException` before the checkpoint is opened), `f32` is announced and reported as `f16` | KV cache element type | auto (model-aligned: `f16` when the model's weights are below F32, else `f32`) | `f32`, `f16`, `q8_0` (runtime also accepts `q4_0`, not swept) | yes |
| `TS_N_CPU_MOE` | MoE models | Routed experts of the first N layers stay in system RAM: multiplied on the host at decode, streamed to the accelerator for one graph at prefill. Unset on `ggml_metal`, Qwen 3.8 Flash Next (`qwen4exp`) plans it from the Metal working set and RAM when its experts do not fit (15 of 48 layers stay on the GPU on a 48 GiB Mac) | off (`0`) | `0`, `16`, `all` | no (registered for the GGML GPU backends and MoE families, but not in the default config's list) |
| `TS_CPU_MOE` | MoE models | Offload every layer's routed experts (equivalent to `TS_N_CPU_MOE=all`) | off | not registered | no |
| `TS_CPU_MOE_THREADS` | MoE models | Worker threads for the host-side expert matmul. Above 8 usable CPUs (hardware threads clamped by the affinity mask and the cgroup CPU quota) the default is half of them, capped at 64: the decode-side matmul is one token wide, so past a few dozen workers each extra thread only adds a barrier participant (measured 7x slower at 192 threads than at 32 on a 2-socket Xeon). Smaller hosts keep almost every CPU. DeepSeek V4 / V4.1 default to every usable CPU instead (see below) | 1 at ≤2 usable CPUs, usable−1 at ≤8, else min(usable/2, 64) | - | no |
| `TS_HOST_MOE_DEVICE_MIN_BATCH` | MoE models with offload | Batch size at or above which an offloaded layer is computed on the accelerator with its experts streamed in, rather than on the host. `0` restores host-only offload | `128` | not registered | no |
| `TS_HOST_MOE_PIN` | MoE models with offload | Page-lock (`cudaHostRegister`) the offloaded expert ranges so the streamed prefill DMAs instead of staging through the driver (9.3 -> 55.6 GB/s on PCIe 5.0). `0` disables it for every architecture. DeepSeek V4 / V4.1 are the exception to the default: their loader computes offloaded experts on the CPU backend at every batch size, so nothing streams them, and it pins them only with `1` (pinning 48.2 GiB cost 20.4 s of load on the seven-A40 lane and made those pages unevictable) | ON; off for DeepSeek V4 / V4.1 | not registered | no |
| `TS_HOST_MOE_PIN_MAX_MB` | MoE models with offload | Budget for the pinned expert ranges | 60% of the cgroup/host memory limit | - | no |
| `TS_HOST_MOE_DECODE` | MoE models with offload | `0` runs one-token offloaded layers as a ggml CPU graph instead of TensorSharp's decode kernel (ggml's own CPU dot products on a team woken once per layer; 0.6 -> 0.32 ms a layer on Qwen3.8-Flash-Next on an M5 Pro). For A/B runs | ON | not registered | no |
| `TS_HOST_MOE_TIMING` | MoE models with offload | Diagnostics: `1` host per-call setup vs matmul and the streamed prefill's bytes, transfer rate and GPU time; `2` per-weight copy rate (synchronizes, changes what it measures); `3` per decode pass: accelerator segments, host experts, upload; `4` the one-token kernel's phases and worker wake-up | off | not registered | no |
| `TS_HOST_MOE_EXPERT_FILTER` | MoE models with offload | Stream only the experts the batch actually routes to, grouped into consecutive runs | ON | not registered | no |
| `MAX_CONTEXT` | long text / uploaded text | Hard context cap. Set, it is a requirement: honoured if the caches fit and refused with the numbers if not. Unset, the GGUF's advertised length is a ceiling the loader may cap to what the devices hold — GLM-5.2 advertises 1M tokens, which is ~93 GiB of KV | model default (a ceiling, not a promise) | `4096`, `8192`, `16384` | yes |

## Prefill / Decode Tuning

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_PREFILL_CHUNK` | swept on GPT OSS, Qwen 3.5 / 3.6 family long-context features; honored at runtime by Gemma 4, Nemotron-H, and Mistral 3 as well | Chunked prefill block size | architecture default | `256`, `512`, `1024` | yes |
| `TS_GGML_ASYNC_COMPUTE` | GGML backends | Async compute submission | ON on `ggml_metal` (`0` disables), OFF on other GGML backends | `0`, `1` | yes |

## Multimodal

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `VIDEO_SAMPLE_FPS` | video features | Time-based frame sampling rate | `1` | `1`, `2` | yes |
| `VIDEO_MAX_FRAMES` | video features | Upper bound on sampled video frames | no cap | `8`, `16` | yes |
| `TS_NEMOTRON_IMAGE_MAX_TILES` | Nemotron-H image features | Maximum image tiles | architecture default | `4`, `8`, `12` | yes |

## MLX-Specific

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_MLX_BATCHED_MOE_DECODE` | Qwen 3.5 / 3.6 MoE on MLX | One batched dispatch per gate/up/down over the stacked experts instead of per-expert dispatches; `0` runs the per-expert sequence and skips the stacked copy (memory-constrained machines) | ON | `0`, `1` | yes |
| `TS_MLX_HALF_MATMUL_MIN_ROWS` | MLX affine quantized matmuls | Rows from which a matmul with F16 scales runs with F16 activations; `0` keeps every matmul F32 | `1` (all) | `0`, `1`, `32` | no |
| `TS_MLX_Q6K_MATVEC_MAX_ROWS` | exact Q6_K matmuls on MLX | Row count up to which Q6_K runs as matrix-vector products (a port of ggml-metal's `kernel_mul_mv_q6_K_f32`); larger row counts dequantize Q6_K to F16 (in slices of at most 256 MB) and multiply with MLX's GEMM | `4` | integer >= 0 | no |
| `TS_MLX_IQ4XS_MATVEC_MAX_ROWS` | IQ4_XS matmuls on MLX | Row count up to which IQ4_XS runs as matrix-vector products (a port of ggml-metal's `kernel_mul_mv_iq4_xs_f32`); larger row counts dequantize IQ4_XS to F16 (in slices of at most 256 MB) and multiply with MLX's GEMM | `4` | integer >= 0 | no |

## Out-of-Matrix Pure-C# CPU Backend Knobs

These tune the persistent worker pool, the quantized-weight handling and the
managed SIMD kernels behind `--backend cpu`. They are real runtime knobs but are
not registered in `EnvVarMatrix.All` and are not swept by the default TestMatrix
config. The `0` switches restore the previous code path in the same binary, for
A/B runs; none of them is needed for correct output. Measurements quoted below
were taken on an i7-11800H (8 cores / 16 threads, AVX-512, 32 GB) unless a row
names another machine.

**Which instruction sets run.** Every hand-written kernel of this backend takes
its instruction set from one decision (`TensorSharp.Cpu.CpuIsa`), so a host never
runs AVX-512 in one kernel family and AVX2 in another: AVX-512 where the CPU has
AVX-512 F/BW/DQ and the runtime accelerates `Vector512` (the JIT leaves that off on
some parts where 512-bit vectors are slower, and under `DOTNET_EnableAVX512=0`),
AVX2+FMA otherwise, and the portable Vector128 / scalar code without AVX2 (ARM64
included, where the quantized matmuls keep the per-row path).

**Testing the AVX2 path.** `TS_CPU_DISABLE_AVX512=1` runs the hand-written
AVX-512 kernels in their AVX2 form on an AVX-512 host. It does not narrow the
.NET runtime itself (`TensorPrimitives`, plain copies,
`Vector512.IsHardwareAccelerated`). To emulate an AVX2-only host, start the
process with `DOTNET_EnableAVX512=0`; on .NET 10 the older
`DOTNET_EnableAVX512F=0` is ignored. `DOTNET_EnableAVX2=0` (or
`DOTNET_EnableHWIntrinsic=0`) leaves only the portable tier; the unit tests that
compare the GEMM kernels report themselves as skipped there instead of passing
with nothing to compare. The AVX2 kernels have been exercised this way on an
AVX-512 machine, not on AVX2-only hardware.

**What stays native.** With `--backend cpu` the model compute loads no native
library (no GgmlOps, no CUDA). Outside the model compute, image files are read
and written through Magick.NET on desktop (a native ImageMagick build), and the
server probes the GGML and CUDA backends at startup to list what the machine can
run, whatever `--backend` says; both predate these kernels.

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_CPU_THREADS` | `cpu` backend (100% pure C#) | Width of the persistent worker pool that runs the managed matmuls; the Core CPU kernels (F32 SGEMM, elementwise, norm, softmax, RoPE on `CpuStorage` tensors, through the `CpuParallel` hook `TensorSharp.Models` installs at module load); the DiffusionGemma and Qwen-Image-2.1 transformer kernels; and the Qwen-Image text encoder and vision tower. The Qwen-Image VAE has a wider pool of its own (`TS_CPU_GEMM_THREADS`). Default is HALF the usable CPUs above 8, deliberately not all of them. That width was tuned when the pool ran only the quantized matmuls and the rest of the CPU path used the ThreadPool: pool workers spin between jobs, so taking every core starved that other work. Measured on a 122-CPU quota, two interleaved runs per cell (prefill / decode tok/s): pool off 21.7,21.0 / 2.0,2.4; 32 threads 24.9,24.1 / 4.9,5.0; 48 threads 25.6,28.5 / 5.4,6.0; 61 threads 24.2,24.9 / 6.3,5.9; 122 threads 13.5 / 4.8. At 122 only prefill regresses - decode still beats the pool-off baseline. Re-measured with everything on the pool, 16 threads against the default 8 on the 8-core / 16-thread i7-11800H, two runs each: Qwen-Image-2.1 256x256 in 2 steps 15.7 / 16.5 s against 16.9 / 17.1 s (the transformer steps faster, 9.2-9.9 s against 11.0-11.1 s; the text and vision encode phase, which includes loading the text encoder, slower, 3.4-3.6 s against 3.1 s; the text-encoder forward alone, a 37-token prompt in `QwenImageStagesBench`, 1.65-2.26 s against 1.24-1.26 s on its first call and 0.72-1.42 s against 0.68-0.72 s after that, with identical outputs); a DiffusionGemma Jev read of a new prompt 1.80 s against 1.54-1.62 s on average; a read of a cached prompt 209-223 ms against 218-231 ms. No clear win, so the default stays | every core at <=8 CPUs, else max(8, usable/2) | not registered | no |
| `TS_CPU_POOL` | `cpu` backend | `0` reverts to ThreadPool `Parallel.For`, for hosts that cannot afford dedicated spinning threads and to A/B the two in one binary. It covers every managed kernel that forks through `CpuWorkers`, the quantized matmuls' `RunParallelBlocks` or the Core `CpuParallel` hook (whose binding is skipped): the quantized and float-panel GEMMs, the Core CPU kernels, the DiffusionGemma and Qwen-Image-2.1 transformer kernels, and the Qwen-Image packed GEMM behind the VAE, text encoder and vision tower (the VAE's wide pool becomes `Parallel.For` capped at `TS_CPU_GEMM_THREADS`). The kernels do not change results with the split, so outputs are the same either way. Still on a pool: the row loops of the Direct video networks (`DirectOps`, MiniMax-H3's own loops); DeepSeek V4's executor has threads of its own (`TS_DSV4_THREADS`) | ON | not registered | no |
| `TS_CPU_SPIN` | `cpu` backend | Spin iterations a pool worker takes before parking. Parking is the expensive part at this width (waking N workers costs more than the ~60 us of work being handed out), so the default spins long enough that the steady state never parks: at 256 the same model measured 0.1 tok/s against 7.0 at 4096 | `4096` | not registered | no |
| `TS_CPU_TASK_BYTES` / `TS_CPU_TASKS_PER_WORKER` | `cpu` backend | Chunking of a managed matmul: weight bytes per work item, and the cap on work items per worker. Sized from the WORK rather than the thread count - the old thread-count-scaled rule built 1024 tiny tasks per matmul at 122 threads and stopped scaling past 8 | `131072` / `4` | not registered | no |
| `TS_CPU_QGEMM_MIN_ROWS` | quantized matmuls on `cpu` (diagnostic) | Calls and batch jobs with fewer rows take the per-row path. Unset, every row count takes the GEMM, which makes a row's result independent of how many rows share the call (decode, speculative verify, continuous batching and MoE batches all give the same bits); a value above 1 gives that up | unset (1) | not registered | no |
| `TS_CPU_QGEMM_TASK_MACS` / `TS_CPU_QGEMM_L2_BYTES` | quantized matmuls on `cpu` (tuning) | Minimum multiply-accumulates per parallel task (about 50 us on one core), and the activation bytes one row block may hold; the block is re-read once per decoded column pair, so it has to stay in L2 | `1048576` / `524288` | not registered | no |
| `TS_CPU_QGEMM_VERIFY` | quantized matmuls on `cpu` (diagnostic) | `1` re-runs every GEMM through the per-row path and prints the largest relative difference seen so far to stderr, which checks the kernels on a real model's weights and activations. Slow | OFF | not registered | no |
| `TS_CPU_SGEMM_KERNEL` | F32 matmuls on `cpu` (`Ops.Addmm` / `AddmmBatch`, the Direct video networks' GEMMs) | Pins a micro-kernel: `avx512`, `avx2wide` (8x24 on the 32 EVEX registers, AVX-512 hardware only), `avx2` or `portable`. An unsupported choice falls back to the default | widest supported | not registered | no |
| `TS_CPU_SGEMM_KC` / `TS_CPU_SGEMM_MC` / `TS_CPU_SGEMM_NC` | F32 matmuls on `cpu` (tuning) | Cache-blocking overrides; MC and NC are rounded up to the register tile | 256 / 144 / 1024 (AVX-512, AVX2) | not registered | no |
| `TS_CPU_SGEMM_DOT_MAXN` | F32 matmuls on `cpu` | Widest N routed to the narrow dot path (small N, A rows and B columns contiguous along K); `0` turns the path off | 64 (AVX-512, AVX2), 40 (`avx2wide`) | not registered | no |
| `TS_CPU_DISABLE_AVX512` | every hand-written AVX-512 kernel of the pure-C# CPU path (quantized GEMM and its quantizer, per-row Q4_0 / Q8_0 dots, SGEMM, elementwise, DiffusionGemma attention, Qwen-Image DiT / VAE / text-encoder / vision kernels) | `1` runs their AVX2 form, so the AVX2 path can be tested on an AVX-512 host (see "Which instruction sets run" and "Testing the AVX2 path" above). All of them read it through the same decision, so none keeps AVX-512 while another drops it | OFF | not registered | no |

## Out-of-Matrix DiffusionGemma Knobs

These variables are real runtime knobs, but they are not registered in
`EnvVarMatrix.All` today and are not swept by the default TestMatrix config.

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `DIFFUSION_STEPS` | DiffusionGemma Web UI | Denoising steps per block in the server path | `48` | not registered | no |
| `DIFFUSION_MAX_BATCH` | DiffusionGemma Web UI | Max active requests in `DiffusionBatchScheduler` | `2` | not registered | no |
| `DIFFUSION_BATCHED_FORWARD` | DiffusionGemma | True batched canvas decode vs time-sliced fused single-canvas decode | OFF | not registered | no |
| `DIFFUSION_NO_PKV` | DiffusionGemma | Disable prompt-KV caching (the device-glue backends and `cpu`): every read and every denoising step then runs the unified `[prompt\|canvas]` forward | OFF | not registered | no |
| `DIFFUSION_CPU_ATTN_FAST` | DiffusionGemma on `cpu` | `1` selects FMA attention tiles (Vector512 when accelerated) and a vectorized softmax. The default kernel reproduces the previous arithmetic exactly, because a last-bit change can flip the top-8-of-128 expert routing; attention is a fraction of a percent of a forward at Jev and chat prompt lengths | OFF | not registered | no |
| `DIFFUSION_CPU_MOE_CHUNK` | DiffusionGemma on `cpu` | Tokens per batched-MoE pass; bounds the gathered per-route scratch (a 4k-token prefill would otherwise hold about 1 GB of routed rows) | `512` | not registered | no |
| `DIFFUSION_NO_SC` / `DIFFUSION_SC_TOPK` | DiffusionGemma | Self-conditioning enablement and experimental top-K cutoff | ON / `32` | not registered | no |
| `DIFFUSION_NO_FUSED_DECODE` / `DIFFUSION_NO_FUSED_LMHEAD_TAIL` | DiffusionGemma on GGML backends | Disable fused whole-model diffusion decode or fused lm-head tail | OFF | not registered | no |
| `DIFFUSION_LMHEAD_BATCH_CAP_MB` | DiffusionGemma | Transient lm-head logits memory cap before per-sequence fallback | `300` | not registered | no |
| `DIFFUSION_VRAM_HEADROOM_MB` | DiffusionGemma on ggml_cuda | VRAM kept free of preloaded weights (compute buffers, device copies) | `2048` | not registered | no |
| `DIFFUSION_DEVICE_COPY_BUDGET_MB` | DiffusionGemma on ggml_cuda | Device-copy cache cap when the model does not fit VRAM (prompt K/V, masks, activations) | `768` | not registered | no |
| `DIFFUSION_SEGMENTED_DECODE` | DiffusionGemma on ggml_cuda | Force per-layer fused decode on (`1`) / off (`0`); auto-selected when the model does not fit VRAM | auto | not registered | no |
| `DIFFUSION_PIN_STREAMED` | DiffusionGemma on ggml_cuda | Re-home streamed (non-resident) weights into page-locked copies for DMA-speed uploads (costs RAM) | OFF | not registered | no |
| `DIFFUSION_IMAGE_BIDIRECTIONAL` | DiffusionGemma image input | `0` makes attention inside image soft-token spans plain causal instead of bidirectional on the sliding-window layers | ON | not registered | no |
| `DIFFUSION_FUSED_PREFILL_ATTN` | DiffusionGemma on GGML backends | Fused prompt-prefill attention kernel. On by default on `ggml_cuda` (`0` restores the per-op reference); `1` opts in on other GGML backends. Image prompts always take the per-op mask path | ON on `ggml_cuda`, else OFF | not registered | no |
| `DIFFUSION_NO_DEVICE_SAMPLE` / `DIFFUSION_DEVICE_SAMPLE_FORCE` | DiffusionGemma on ggml_cuda | `1` disables on-device sampling (argmax, entropy, sample and self-conditioning top-K on the device logits); `FORCE=1` keeps it on even under segmented decode, where it is otherwise skipped as a measured loss | device sampling ON when the model fits VRAM | not registered | no |
| `DIFFUSION_ASYNC_COMPUTE` | DiffusionGemma on `ggml_metal` | `1` keeps asynchronous compute on. The model switches it off because Metal's lazy sync has no barrier for a host write followed by a device kernel, which corrupts its per-op prefill and loses the prompt; forcing it back is unsafe and exists only to A/B the cost | OFF | not registered | no |

## Out-of-Matrix Qwen-Image-2.1 Knobs

These tune Qwen-Image-2.1 generation and editing on the CLI and the server. The
matrix feature catalog has no image-generation feature, so none of them is
registered in `EnvVarMatrix.All`. On the pure-C# `cpu` backend a request that
names no size renders at the 1 MP automatic area (1024x1024, the first
reference's aspect ratio for an edit) rather than the native 2048x2048; `ggml_cpu`
and the GPU backends keep 2048x2048. The [Qwen-Image-2.1 card](models/qwenimage21.md)
has the measurements and the remaining diagnostic switches.

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_QWEN21_EDIT_NOISE` | Qwen-Image-2.1 edits | `references` (or unset): an edit's initial noise is drawn on a Philox stream keyed to its reference images (8-bit pixels, SHA-256), so it never restarts from the noise that drew its source at the same seed and size. `seed`: edits draw the seed's text-to-image noise, as stable-diffusion.cpp does, for matched-noise comparisons; each such edit prints a line. Any other value fails the request. Text-to-image noise is the same either way | `references` | not registered | no |
| `TS_QWEN21_PREFIX_CACHE` | Qwen-Image-2.1 DiT | `0` (or `false` / `off` / `no`) turns off the prefix KV cache, which stores the text and reference-image keys and values at the first denoising step and reuses them at every later step | ON | not registered | no |
| `TS_QWEN21_PREFIX_CACHE_TYPE` | same | Storage type of the cached prefix: `auto` (what attention reads, so output matches the uncached run), `f16`, `f32`, `q8_0` or `q8_0_v` (the 8-bit settings round the stored prefix). A misspelled value is an error even while the cache is off | `auto` | not registered | no |
| `TS_QWEN21_PREFIX_CACHE_MAX_MIB` | same | Cap on one cache in MiB, on top of the rule that a cache may use at most half of the device's free memory. A cache that does not fit is declined with a warning, and that request recomputes the prefix every step | unset | not registered | no |
| `TS_QWEN21_GRAPH_REUSE` | Qwen-Image-2.1 DiT on `ggml_cuda` / `ggml_metal` | Retain the step graphs across denoising steps, which lets ggml-cuda capture them as CUDA graphs; `0` rebuilds the graph for each prediction. CPU and Vulkan always build transient graphs | ON | not registered | no |
| `TS_QWEN21_FLASH` / `TS_QWEN21_PAD_MASK` | Qwen-Image-2.1 DiT | Diagnostic switches: `TS_QWEN21_FLASH=0` disables flash attention; on `ggml_cuda`, `TS_QWEN21_PAD_MASK=1` restores the padded-mask attention path (other backends ignore it) | ON / OFF | not registered | no |
| `TS_QWEN21_VAE_FUSED` | Qwen-Image-2.1 VAE | Whole-VAE graph, the default on CUDA and Metal. `0` selects the per-convolution path; `1` forces the graph on another GGML backend except Vulkan, where it is ignored with a warning | ON on CUDA / Metal | not registered | no |
| `TS_QWEN21_VISION_FUSED` | Qwen-Image-2.1 vision encoder on `ggml_cuda` | `0` restores the previous per-block vision path for A/B runs | ON | not registered | no |
| `TS_QWEN21_CPU_MATMUL` | Qwen-Image-2.1 DiT on `cpu` | Unset (or `q8`), the quantized projections quantize their activations to Q8_K / Q8_0 as ggml-cpu does and run the multi-row integer GEMM in `ManagedQuantizedOps`: the faster route with those kernels (a cached step 4.0 s against 7.2-9.0 s at 256x256, 17.6-21.7 s against 29-38 s at 512x512) and closer to ggml_cpu's image (512x512 Pruna-5 LoRA: PSNR 32.9 dB against 31.4). `f32`, the default before, multiplies F32 activations by dequantized weight tiles instead: slower, but numerically steadier - a 1e-6 relative change of the input latents moves the integer pipeline's velocity by about 2.5e-2 relative L2 (ggml-cpu's own by about 1e-2) and the F32 one by about 1e-5 | integer (Q8) activations | not registered | no |
| `TS_QWEN21_CPU_PROFILE` | same | `1` prints one line per forward with the time of each stage | OFF | not registered | no |
| `TS_QWEN21_CPU_GATHER_V` / `TS_QWEN21_CPU_MLP_ROWS` / `TS_QWEN21_CPU_DEPTH` | same (tuning) | Attention reads each head's values from a head-major copy (`0` reads the token-major V in place); rows per MLP chunk, which bounds its `[rows, 2 * ff]` activation; depth of one pass of the dot tiles (a multiple of 16) | ON / `1024` / `1024` | not registered | no |
| `TS_QWEN21_CPU_GELU_FP16` / `TS_QWEN21_CPU_ROUND_ACTIVATIONS` | same (parity) | Reproduce ggml-cpu's rounding for A/B comparisons: its F16 GELU table, and BF16-rounded inputs to the F16/BF16 `img_in` / `txt_in` weights. The defaults are the F32 tanh GELU (as CUDA and Metal compute it) and F32 inputs | OFF / OFF | not registered | no |
| `TS_QWEN_VAE_PROFILE` | Qwen-Image-2.1 VAE on `cpu` | `1` prints the time per op class (conv, norm, add, attention, resample, weight packing) of one encode or decode | OFF | not registered | no |
| `TS_QWEN_IMAGE_CPU_MEMORY_CHECK` | Qwen-Image-2.1 on `cpu` | Before any work, a size whose estimated peak (the VAE decode: 2304 bytes per output pixel plus about 1.8 GiB; the denoise: 128 KiB per token plus 2.5 GiB and the mapped transformer) exceeds the machine's memory is refused with the largest square size that fits; one that only exceeds the memory free right now gets a warning. `0` skips the refusal | ON | not registered | no |
| `TS_CPU_GEMM_THREADS` | Qwen-Image-2.1 VAE on `cpu` | Width of the dedicated pool the managed VAE runs on. One thread per logical CPU by default: the convolutions are FMA-bound, and two SMT threads per core keep the single 512-bit FMA port busier (512x512 decode 7.9 -> 7.0 s). The text encoder and vision tower stay on the shared pool, where the extra spinning workers cost more than they give. Under `TS_CPU_POOL=0` the VAE runs `Parallel.For` capped at this width instead of its own pool | logical CPUs, at most 64 (clamped 1-512) | not registered | no |
| `TS_CPU_GEMM_KC` / `TS_CPU_GEMM_NT` | Qwen-Image packed GEMM on `cpu` (VAE, text encoder, vision) | K chunk and N tile of the packed GEMM, for tuning; the result does not depend on them | `256` / `256` (clamped 16-4096 / 32-2048) | not registered | no |
| `TS_QWEN_TE_CPU_MATMUL` | Qwen-Image-2.1 text encoder (Qwen3-VL-8B) on `cpu` | Unset, the projections quantize their activations to 8 bits as ggml-cpu does and run the multi-row integer GEMM: 0.76-0.88 s for the 37-token default prompt against 1.8-1.9 s for the packed F32 GEMM (ggml_cpu about 1.4 s). `f32` selects the packed F32 GEMM on dequantized weight tiles (exact F32 activations: about 1e-6 relative to a double-precision reference per projection, against about 4e-3 for the 8-bit route). | integer (Q8) activations | not registered | no |
| `TS_QWEN_TE_PROFILE` | same | `1` prints a per-forward split of the per-op path (linear / attention / norms) | OFF | not registered | no |

On `cpu`, the prefix KV cache settings above apply to the managed cache, which
lives in host memory: "free memory" is the physical memory not in use. The
GGML-only switches (`TS_QWEN21_GRAPH_REUSE`, `TS_QWEN21_FLASH`,
`TS_QWEN21_PAD_MASK`, `TS_QWEN21_VAE_FUSED`, `TS_QWEN21_VISION_FUSED`) have no
effect there.

## Out-of-Matrix Speculative-Decoding Knobs

These gate the optional speculative decode path in `TensorSharp.Cli` and
`TensorSharp.Server` (the NextN block embedded in Qwen 3.6, Qwen 3.8 27B, GLM 5.2
and GLM-5.3; Gemma 4's separate `gemma4-assistant` draft GGUF; Qwen 3.8 Flash
Next's shared MTP head GGUF; DeepSeek V4 DSpark and DFlash / DFlash2 block
drafters for Muse-Glimmer and the Qwen 3.5 family; the experimental DeepSeek V4.1
DSpark path; initial text/image HTTP probes with trained weights passed using two-GPU layer split on `ggml_cuda`; broad quality and throughput remain unqualified; the weight-free n-gram
speculator). Speculation engages only for solo (non-concurrent) sequences and only
where the model declares it profitable, which is decided per model: Qwen
3.5/3.6/3.8 and GLM 5.2 / GLM-5.3 on every backend, GLM-5.3-Flash (n-gram only)
where its KDA rollback is available, Gemma 4 on the ggml backends and `cuda`, Qwen
3.8 Flash Next on its GGML token-graph path, DeepSeek V4 / V4.1 and Muse-Glimmer
only with their drafter loaded. Nemotron-H refuses every speculator, and GPT OSS,
Mistral 3 and Hunyuan Dense have no speculative trunk, so not even
n-gram runs there. They are not registered in `EnvVarMatrix.All` and are not
swept by the default TestMatrix config — the matrix feature catalog has no
speculative-decode feature today, so use explicit runs to exercise these.

Each knob has one `TS_SPEC_*` name, which the glm-dsa **native** loader reads too:
`TS_SPEC_DRAFT` from C++ while the model is loading (it sizes its graph cache from
it), and `TS_SPEC` on its managed side to decide whether to page a whole extra
256-expert decoder layer into VRAM. All are also settable via the `--spec*` flags
and `--draft-model` on both hosts. The old `--mtp-*` flags (and
`--spec-draft-model`, `--spec-draft-n-max`, `--spec-draft-conf-min`) and the old
`TS_MTP_*` variables were removed and now fail at startup naming their
replacement.

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_SPEC` | Qwen 3.5/3.6/3.8, GLM 5.2, GLM-5.3, GLM-5.3-Flash (n-gram only), Gemma 4, Qwen 3.8 Flash Next, DeepSeek V4 / V4.1, Muse-Glimmer, Muse Spark (n-gram only, not under its native `--tp`), MiniMax-M3.1 (DSpark or n-gram, layer split only) (CLI + server) | Enable speculative decode for solo sequences | OFF (`0`) | not registered | no |
| `TS_SPEC_TYPE` | all of the above | Speculation algorithm: `auto` \| `draft-head` \| `block` \| `ngram` | `auto` | not registered | no |
| `TS_SPEC_DRAFT` | all of the above | Max tokens drafted per speculative step (1-64) | `8` | not registered | no |
| `TS_SPEC_PMIN` | all of the above | Draft-confidence gate; meaning is per algorithm | per algorithm (`0.15` / `0.35` / `0`) | not registered | no |
| `TS_SPEC_DRAFT_MODEL` | models whose drafter ships as its own GGUF (CLI + server) | Path to a separate drafter, recognised from the file's architecture: Gemma 4's `gemma4-assistant`, Qwen 3.8 Flash Next's shared MTP head, a DeepSeek V4 DSpark (or experimental V4.1 `deepseek41-dspark`) drafter, a DFlash / DFlash2 drafter (Muse-Glimmer, Qwen 3.5 family), or MiniMax-M3.1's DSpark `dflash` drafter, which only its native executor loads (layer split). Set by `--draft-model`, which also enables speculation unless `--no-spec` is given | none | not registered | no |

The design behind these — the three-layer split of model architecture,
speculation algorithm and speculator weights — is documented in
[Speculative Decoding in TensorSharp](speculative_decoding.md).

## Out-of-Matrix Muse-Glimmer & DFlash Knobs

Muse-Glimmer's fused whole-model kernel and its DFlash block drafter each have an
A/B switch, plus long-context sizing knobs. None are registered in
`EnvVarMatrix.All`. The full list, including the layer-trace knobs, is in the
[Muse-Glimmer card](models/muse-glimmer.md#7-environment-variables).

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_MUSE_GLIMMER_PREFILL_CHUNK` | Muse-Glimmer | Tokens per prefill forward; `0` disables chunking | `2048` | not registered | no |
| `TS_MUSE_GLIMMER_SWA_RING` | Muse-Glimmer (fused) | Ring the 39 sliding-window layers at `pad(n_swa + chunk + 1, 256)` rows instead of sizing every layer for the full context | ON | not registered | no |
| `TS_MUSE_GLIMMER_SWA_ROWS` | Muse-Glimmer (fused) | Override the SWA ring size in rows (diagnostics) | auto | not registered | no |
| `TS_DFLASH_PREFILL_CHUNK` | any DFlash drafter | Tokens per speculative prefill forward (drives the TRUNK, not only the drafter) | `1024`, capped by the drafter ring and the trunk's own window | not registered | no |
| `TS_DFLASH_SELECTOR` | DFlash2 drafter | `0` drafts by per-position argmax instead of the candidate lattice (attribution only - the weights were trained with it) | ON | not registered | no |
| `TS_DFLASH_CONV` | DFlash2 drafter | `0` drops the grouped dynamic convolution (attribution only, as above) | ON | not registered | no |
| `TS_DFLASH_SELECTOR_DEBUG` | DFlash2 drafter (per-op path) | `1` prints the first blocks' lattice attribution: unary spread, transition spread, and whether the walk left the unary argmax | OFF | not registered | no |
| `TS_Q35_VERIFY_SNAPSHOTS` | Qwen 3.5 / 3.8 speculative verify | `0` reverts to restoring a pre-verify recurrent-state copy and re-forwarding the accepted prefix instead of keeping one snapshot per row | ON | not registered | no |
| `TS_QAB_BOUND` | Qwen 3.5 / 3.8 batched decode (CUDA) | `0` returns the N>=2 decode to arena-owned KV/state planes (join copy, flush, re-seed per window change). Default binds each slot to its holder's own resident K/V and recurrent-state copies, the buffers solo decode uses | ON | not registered | no |
| `TS_Q35_DEVICE_MASK` | Qwen 3.5 / 3.8 prefill verify (CUDA) | `0` fills the causal mask on the host and uploads it (window x chunk halves, ~170 MB for a 4K chunk at 21K context) instead of building it in-graph | ON | not registered | no |
| `TS_Q35_VERIFY_PACK` | Qwen 3.5 / 3.8 verify (CUDA, Metal) | `0` gives each persistent verify graph unpacked buffers instead of lifetime-packed gallocr buffers | ON | not registered | no |
| `TS_Q35_VERIFY_PERSIST_KV_STRIDE` | Qwen 3.5 / 3.8 speculative verify | Attention-window stride (rows) of persistent verify graphs; larger strides rebuild/recapture less often | `2048` | not registered | no |
| `TS_Q35_GPU_STATE_COMMIT` | Qwen 3.5 / 3.8 speculative verify (CUDA, Metal) | `0` commits the accepted recurrent-state snapshot through the host instead of an on-device copy | ON | not registered | no |
| `TS_Q35_VERIFY_LOG` | Qwen 3.5 / 3.8 | `1` prints every fused verify/prefill call (start, rows, logits, capture, state chaining, ms) | OFF | not registered | no |
| `TS_QWEN35_DEVICE_HOLDER_COPY` | Qwen 3.5 / 3.8 prefix cache (CUDA) | `0` makes checkpoints, clones and KV growth copy K/V through host memory instead of device-to-device | ON | not registered | no |
| `TS_QWEN35_DEVICE_RETAIN_RESERVE_MB` | Qwen 3.5 / 3.8 prefix cache (CUDA) | Free VRAM that idle retained holders' device K/V must leave; below it the least recently used are demoted to host | `max(4 GiB, VRAM/8)` | not registered | no |
| `TS_QWEN35_IDLE_KV_SYNC` | Qwen 3.5 / 3.8 prefix cache (CUDA) | `0` stops downloading idle retained holders' device K/V to host in the background ahead of memory pressure, so a later demotion syncs on the request's critical path again (~150 ms per GB instead of ~5 ms) | ON | not registered | no |
| `TS_PREFIX_RELEASE_DEBUG` | Qwen 3.5 / 3.8 prefix cache | `1` logs retain/plan/settle decisions, demotions (`[prefix-demote]`) and their host syncs (`[prefix-sync]`) | OFF | not registered | no |
| `TS_GGML_HOST_POOL_MB` | GGML CUDA host allocations | Byte budget for keeping freed host blocks of any size for reuse (fresh mappings page-fault on first touch); `0` keeps only the small count-bounded pool | `min(8 GiB, RAM/16)` | not registered | no |
| `TS_QWEN35_BATCHED_SPEC` | Qwen 3.5 / 3.8 batched speculative decoding (CUDA, DFlash drafter) | `0` makes concurrent decoding requests decode one token per step in the batched decode instead of verifying every request's draft window in one forward | ON | not registered | no |
| `TS_QWEN35_BATCHED_SPEC_WINDOW` | Qwen 3.5 / 3.8 batched speculative decoding | Widest verify window per request (anchor plus drafts), 2..8; a step verifies four rows when every request's confidence-gated draft fits them | `8` | not registered | no |
| `TS_QWEN35_SOLO_SPEC_DRAFT` | Qwen 3.5 / 3.8 speculative decoding (lone request) | Most drafts a lone request's verify takes per step | `3` | not registered | no |
| `TS_QWEN35_BATCHED_DRAFT` | Qwen 3.5 / 3.8 batched speculative decoding | `0` drafts each request's block in its own DFlash graph instead of all requests' blocks in one | ON | not registered | no |
| `TS_BATCHED_SPEC_LOG` | Batched speculative decoding | `1` prints each batched step's kept tokens and draft / verify / commit / drafter-commit milliseconds (`[batched-spec]`) | OFF | not registered | no |
| `TS_SCHED_STEP_LOG` | Continuous batching executor | `1` prints one line per step: the idle gap since the previous step, the serving path's milliseconds and each sequence's prefill/decode tokens (`[step]`) | OFF | not registered | no |
| `TS_GPU_LOCK_YIELD_MS` | Continuous batching executor | Longest a step waits for a request thread already waiting on the model's GPU lock (prompt media preparation) before taking the lock itself; `0` takes it first | `100` | not registered | no |
| `TS_QWEN35_BACKGROUND_CAPTURE` | Qwen 3.5 / 3.8 prefix cache (CUDA) | `0` makes a host-path prefix checkpoint download the request's K/V rows synchronously instead of on a background stream straight into the checkpoint | ON | not registered | no |
| `TS_QWEN35_DEVICE_STATE_HANDOFF` | Qwen 3.5 / 3.8 per-request caches (CUDA) | `0` moves the verifier's recurrent state through host memory whenever it changes hands (a 151 MB drain on every holder switch, an upload on every resume, host zeros at position 0) instead of device-to-device into each request's own device copies | ON | not registered | no |
| `TS_PREFIX_ASYNC_SAVE` | Prefix checkpoint persistence | `0` writes shared-prefix checkpoint files on the engine worker (inline export, write and fsync) instead of in the background; background saves need 32 GB of host memory | ON | not registered | no |
| `TS_SCHED_PREFILL_FCFS` | Continuous batching scheduler | `1` gives a step with no decode in flight to prompts oldest first instead of an even share per prompt | OFF | not registered | no |
| `TS_SCHED_MIXED_PREFILL_TOKENS` / `TS_SCHED_MIXED_PREFILL_FCFS` | Continuous batching scheduler | Prompt tokens a step that also decodes may carry across all its prefilling sequences (`0`: only the per-sequence chunk bound); FCFS keeps that budget on the oldest prompt. A model may set the bound itself when the variable is unset (Qwen 3.5 / 3.8 direct-CUDA engine: `1024`) | the model's bound, else none / OFF | not registered | no |
| `TS_SCHED_MIXED_STEP` / `TS_Q35_MIXED_STEP` | Qwen 3.5 / 3.8 direct-CUDA engine | `0` runs a step's prompt chunk and its batched speculative verify as two forwards instead of one mixed forward | ON | not registered | no |
| `TS_Q35_MIXED_FP4_VERIFY` | Qwen 3.5 / 3.8 direct-CUDA engine | `0` keeps verify windows and prompt tails of up to 8 rows on the q8_1 GEMV (8-bit activations) in a forward of their own instead of riding the mixed forward's FP4 GEMM (as wider batched verifies always do) | ON | not registered | no |
| `TS_Q35_NATIVE` | Qwen 3.5 / 3.8 dense NVFP4 on `--backend cuda` | `0` keeps the managed path instead of the direct-CUDA engine; the engine also steps aside without sm_12x or its nvcc-built kernels | ON | not registered | no |
| `TS_Q35_HOST_CACHE_MB` | Qwen 3.5 / 3.8 direct-CUDA engine | Pinned host memory for retained prefix slots moved out of device memory (`0` turns the host tier off) | `min(available / 2, 32 GiB)` | not registered | no |
| `TS_Q35_UBATCH` | Qwen 3.5 / 3.8 direct-CUDA engine | Rows per forward (prompt chunk micro-batch) | `2048` | not registered | no |
| `TS_Q35_ATTN_PREFILL` / `TS_Q35_GEMM_SPLITK` | Qwen 3.5 / 3.8 direct-CUDA engine | Diagnostics: `0` restores the older prefill attention kernel / keeps every FP4 GEMM to one K pass (a summation order independent of the row count) | ON / ON | not registered | no |
| `TS_QWEN_VISION_MERGED` | Qwen 3.5 / 3.8 vision encoder | `0` runs the post-LN spatial merger as separate ops after the fused encoder graph instead of inside it (bit-identical either way) | ON | not registered | no |
| `TS_QWEN_VISION_F16` | Qwen 3.5 / 3.8 vision encoder (CUDA) | `1` gives the merged encoder F16 linear weights (F32 accumulation): half the resident weight memory, relative L2 ~0.002-0.003 vs F32, no measured speedup | OFF | not registered | no |
| `TS_QWEN_VISION_DUMP` | Qwen 3.5 / 3.8 vision encoder | Directory to write each image's projected embeddings (int32 count + float32 values) for `eng/validation/compare-vision-embeddings.py` | unset | not registered | no |
| `TS_QWEN_VISION_ATTN` | Qwen 3.5 / 3.8 vision encoder (CUDA sm_80+) | Vision attention kernel: `mma` (tensor cores, F16 operands with F32 scores/softmax/output; the decomposed path's accuracy against exact F32 at about half the encoder time), `ggml` (decomposed TF32 matmul + softmax), `f32` (exact F32 kernel, slower) | `mma` | not registered | no |
| `TS_QWEN_VISION_F32` | Qwen 3.5 / 3.8 vision encoder (CUDA) | `1` selects the exact F32 attention kernel (same as `TS_QWEN_VISION_ATTN=f32`) | OFF | not registered | no |
| `TS_SPEC_ADAPTIVE` | Speculative decoding (all drafters) | `0` disables the cost governor, so drafting is never measured against a plain baseline and never parked. For A/B measurement: a governor round's baseline steps are plain decodes and they are not free | ON | not registered | no |
| `TS_GGML_LOG_DEBUG` | GGML backends | `1` passes ggml's DEBUG log channel through instead of dropping it. Carries the CUDA backend's "CUDA graph warmup complete"/"reset" lines, which are the only way to see whether a graph is actually being CUDA-graph-captured | OFF | not registered | no |

## Out-of-Matrix Muse Spark Knobs

Muse Spark's native whole-model executor is selected and sized by these switches.
None are registered in `EnvVarMatrix.All`. The full list, including the per-op
path's opt-in optimizations, is in the
[Muse Spark card](models/muse-spark.md#options).

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_MUSE_SPARK_NATIVE` | Muse Spark | `0` selects the per-op path; `1` also runs the native executor on `ggml_cpu`. `--tp N` on `ggml_cuda` runs the native executor tensor-parallel over the first N CUDA GPUs through ggml's meta backend (choose them with `CUDA_VISIBLE_DEVICES`; `TENSORSHARP_TP_DEVICES` is refused); `0` keeps the per-op TP path, which `--tp` on `ggml_vulkan` always uses | ON for `ggml_cuda` / `ggml_vulkan` | not registered | no |
| `TS_MUSE_SPARK_CPU_MOE` | Muse Spark (native) | `auto` places each expert stack separately, keeping as many on the GPUs as fit and as few bytes as possible on the host; overrides `--n-cpu-moe`. Under native `--tp` every expert stays in GPU memory | OFF | not registered | no |
| `TS_MUSE_SPARK_PLAN_SLOTS` | Muse Spark (native) | Sliding-window rings reserved in the placement plan; set to the concurrency plus one for the shared-prefix checkpoint | `1` | not registered | no |
| `TS_MUSE_SPARK_UBATCH` | Muse Spark (native) | Prefill micro-batch rows; a graph that does not fit runs that call in halved micro-batches | `2048` | not registered | no |
| `TS_MUSE_SPARK_DEV0_RESERVE_MB` | Muse Spark (native) | Extra memory kept free on GPU 0, e.g. for the vision encoder | `0` | not registered | no |
| `TS_MUSE_SPARK_HOST_NUMA` | Muse Spark (native) | `0` keeps host experts in the GGUF mapping instead of NUMA-local copies with pinned workers | ON with two or more NUMA nodes | not registered | no |
| `TS_MUSE_SPARK_WARM_POOLS` | Muse Spark (native, GPUs) | `0` skips the full-size prefill at load that grows ggml-cuda's kernel-scratch pools to their peak before slots take the memory; with pipelined prefill on, the load also runs a short pipelined prefill (which `0` skips too) that builds the pipeline's scheduler and checks that it engages | ON with GPUs | not registered | no |
| `TS_MUSE_SPARK_PIPELINE` | Muse Spark (native, layer split) | Pipelined prefill: a prompt call is cut into micro-batches, and GPU k works on one while GPU k+1 works on the one before. Unset = automatic (on with two or more GPUs, all CUDA-family, no host experts, `TS_MUSE_SPARK_PROFILE` unset, not `--tp`); `1` forces it (refused on a multi-GPU non-CUDA backend and under `--tp`); `0` turns it off; any other value refuses the load. The load log prints `pipelined prefill: on (...)` or `off (reason)` | automatic | not registered | no |
| `TS_MUSE_SPARK_PIPELINE_UBATCH` | Muse Spark (native, pipelined prefill) | Fixed micro-batch rows for pipelined prefill, an integer in [1, `TS_MUSE_SPARK_UBATCH`]; anything else refuses the load. Unset = adaptive: about a quarter of each call, between 512 rows and the prefill ubatch, smaller (down to 128 rows) where a long context's masks would exceed the staging budget | adaptive | not registered | no |

## Out-of-Matrix MiniMax-M3.1 Knobs

MiniMax-M3.1-Flash (`minimax-m3`) runs only on its native whole-model executor
(`ggml_cuda`, with `ggml_cpu` as a correctness path), and these variables are
every `TS_MINIMAX_M3_*` setting the executor and its managed glue read. None
are registered in `EnvVarMatrix.All`. The managed side hands `_LAYER_SPLIT`,
`_UBATCH`, `_THREADS` and `_CPU_MOE` to the executor as load arguments and
forwards the others, so values set inside the .NET process reach it too. The
placement flags themselves (`--layer-split N`, `--tp N`, `--n-cpu-moe N`,
`--draft-model`) are documented in the
[MiniMax-M3.1 card](models/minimax-m3.md#running-it).

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_MINIMAX_M3_MSA` | MiniMax-M3.1 | Attention path. `auto`: micro-batches whose last position is below 2048 run dense flash attention (exactly the sparse result there), others run sparse attention through TensorSharp's MSA kernels on CUDA GPUs and through upstream ggml ops elsewhere (always the ggml ops under `--tp`). `dense`: dense everywhere (debugging; changes results from position 2048 on). `ggml`: sparse attention through upstream ggml ops for every micro-batch. `kernel`: TensorSharp's kernels for every micro-batch (on a CPU-only load, their CPU reference); refused on a GPU that cannot run them and under `--tp`. Any other value refuses the load | `auto` | not registered | no |
| `TS_MINIMAX_M3_MSA_CHUNK_MB` | MiniMax-M3.1 (sparse prompt segments on the ggml path) | Byte budget of one token chunk of a sparse prompt segment's masked attention; also added to each GPU's placement reserve. Must be a positive number of MiB, or the load is refused | `256` | not registered | no |
| `TS_MINIMAX_M3_INDEXER_F16` | MiniMax-M3.1 | F16 copies of the F32 indexer projections for decode-shaped graphs (at most 8 rows), which halves the indexer bytes a decode step reads; prompts keep reading the F32 originals. `0` keeps F32 only | on with GPUs, off on a CPU-only load | not registered | no |
| `TS_MINIMAX_M3_FUSE_QKV` | MiniMax-M3.1 | Concatenate the same-typed Q, K and V projections, and the indexer's q and k projections, into one matrix each (one weight read and one launch). `0` keeps them separate | on | not registered | no |
| `TS_MINIMAX_M3_LAYER_SPLIT` | MiniMax-M3.1 (layer split) | Explicit layers per GPU, comma-separated non-negative counts summing to 60 and at most one per GPU; a longer list raises the GPU count. A malformed list, or one that does not fit, refuses the load. Refused with `--tp` | automatic placement | not registered | no |
| `TS_MINIMAX_M3_UBATCH` | MiniMax-M3.1 | Prefill micro-batch rows; also bounds one speculative verify (draft plus anchor) | `2048` | not registered | no |
| `TS_MINIMAX_M3_THREADS` | MiniMax-M3.1 | CPU backend threads of a GPU-only load. CPU-only loads use the available parallelism; host experts beside GPUs use half of it (their decode is memory-bandwidth bound, and the other half keeps the serving process's own threads off the expert workers' cores); `--cpu-moe-threads N` / `TS_CPU_MOE_THREADS` override both | available parallelism (half with host experts) | not registered | no |
| `TS_MINIMAX_M3_CPU_MOE` | MiniMax-M3.1 (layer split) | `auto` places only the routed-expert stacks (gate, up and down separately) that do not fit on the GPUs in system RAM, and overrides `--n-cpu-moe`. It does not cap the context (it moves experts to the host instead), so set `MAX_CONTEXT` with it. Under `--tp` every expert stays in GPU memory | off | not registered | no |
| `TS_MINIMAX_M3_PLAN_CTX_TOKENS` | MiniMax-M3.1 | K/V tokens the placement reserves across all slots; set to the context times the concurrent requests plus one (the shared-prefix checkpoint). A value that is not a positive token count is said and replaced by the default | one full context | not registered | no |
| `TS_MINIMAX_M3_VRAM_RESERVE_MB` | MiniMax-M3.1 | Per-GPU memory the placement keeps for compute buffers and ggml-cuda's scratch pool, on top of which come the sparse chunk budget and, under `--tp`, the reduction buffers | `2048` + ubatch / 4 | not registered | no |
| `TS_MINIMAX_M3_DEV0_RESERVE_MB` | MiniMax-M3.1 | Extra memory kept free on GPU 0, e.g. for the vision encoder | `0` | not registered | no |
| `TS_MINIMAX_M3_KV_PAD` | MiniMax-M3.1 | Granularity, in rows, of the attention windows a graph is built for: decode reuses one graph shape over a stretch of this many positions. Rounded up to a multiple of the 128-row MSA block, with a warning | `256` | not registered | no |
| `TS_MINIMAX_M3_BD_KV_PAD` | MiniMax-M3.1 (batched decode) | The window granularity of batched decode graphs, where N sequences cross window boundaries N times as often; kept a multiple of `TS_MINIMAX_M3_KV_PAD` | 2 × `TS_MINIMAX_M3_KV_PAD` | not registered | no |
| `TS_MINIMAX_M3_SLOT_INIT_ROWS` | MiniMax-M3.1 | Initial K/V rows of a sequence slot (rounded to the KV pad, at most the context); rows double on demand | `4096` | not registered | no |
| `TS_MINIMAX_M3_SLOT_HEADROOM_MB` | MiniMax-M3.1 | Free memory a new slot or KV growth must leave on each GPU | `256` | not registered | no |
| `TS_MINIMAX_M3_ADMIT_GRAPH_MB` | MiniMax-M3.1 | Most graph memory a new slot or KV growth keeps free on top of the headroom, so one long prefill's graph does not make every later slot evict retained prefix state | `512` | not registered | no |
| `TS_MINIMAX_M3_GRAPH_CACHE` | MiniMax-M3.1 | Built graphs kept in the shape-keyed LRU. Under `--tp` the cache keeps one graph, and a value above 1 refuses the load | `12` | not registered | no |
| `TS_MINIMAX_M3_NODES_PER_LAYER` | MiniMax-M3.1 (diagnostic) | Override the graph node budget per layer that the executor otherwise derives from the graph's segments | derived | not registered | no |
| `TS_MINIMAX_M3_PIPELINE` | MiniMax-M3.1 (layer split) | Pipelined prefill: a prompt call is cut into micro-batches, and GPU k works on one while GPU k+1 works on the one before. Unset = automatic (on with two or more GPUs, all CUDA-family, no host experts, `TS_MINIMAX_M3_PROFILE` unset, not `--tp`); `1` forces it (refused on a multi-GPU non-CUDA backend and under `--tp`); `0` turns it off; any other value refuses the load. The load log prints `pipelined prefill: on (...)` or `off (reason)` | automatic | not registered | no |
| `TS_MINIMAX_M3_PIPELINE_UBATCH` | MiniMax-M3.1 (pipelined prefill) | Fixed micro-batch rows for pipelined prefill, an integer in [1, `TS_MINIMAX_M3_UBATCH`]; anything else refuses the load. Unset = adaptive: the micro-batch count that minimizes the pipeline's estimated time (2 for 2,048 tokens, 3 for 4,096, 4 for 8,192 on eight A40s), between 256 rows and the prefill ubatch, smaller (down to 128 rows) where a long context's masks would exceed the 96 MiB staging budget | adaptive | not registered | no |
| `TS_MINIMAX_M3_WARM_POOLS` | MiniMax-M3.1 (GPUs) | `0` skips the full-size prefill at load that grows ggml-cuda's kernel-scratch pools to their peak before slots take the memory; with pipelined prefill on, the load also runs a short pipelined prefill (which `0` skips too) that builds the pipeline's scheduler and checks that it engages | on with GPUs | not registered | no |
| `TS_MINIMAX_M3_HOST_NUMA` | MiniMax-M3.1 (host experts, Linux) | With two or more NUMA nodes, copy every host expert stack into memory beside the pinned worker that reads it; `0` serves them from the GGUF mapping | on with two or more NUMA nodes | not registered | no |
| `TS_MINIMAX_M3_NUMA_BATCH_GB` | MiniMax-M3.1 (NUMA host experts) | GiB of host expert stacks the NUMA copy reads per batch before it drops that batch's file pages | `16` | not registered | no |
| `TS_MINIMAX_M3_HOST_MOE_TOKENS` | MiniMax-M3.1 (host experts) | Batches of 2 to N tokens run host experts through TensorSharp's static-row kernel (ggml's quantization and dot products); `0` keeps ggml's `mul_mat_id` | `16` | not registered | no |
| `TS_MINIMAX_M3_CPU_REPACK` | MiniMax-M3.1 (host experts) | `1` repacks host expert stacks into ggml's CPU_REPACK layout instead of serving them from the mapping; only without NUMA placement. Refused under `--tp` | off | not registered | no |
| `TS_MINIMAX_M3_OP_OFFLOAD` | MiniMax-M3.1 | Let ggml's scheduler copy host weights to a GPU for large batches | on only without host experts | not registered | no |
| `TS_MINIMAX_M3_WARM_HOST` | MiniMax-M3.1 (host experts) | Prefault the host-mapped expert weights at load when they plus 8 GiB fit the host allowance (the cgroup limit); otherwise the load logs that it skipped the prefault. `0` never prefaults | on | not registered | no |
| `TS_MINIMAX_M3_LOAD_CHUNK_MB` | MiniMax-M3.1 | Weight-upload chunk size | `64` | not registered | no |
| `TS_MINIMAX_M3_LOAD_THREADS` | MiniMax-M3.1 | Weight-upload reader threads (also the host-prefault threads) | `16` | not registered | no |
| `TS_MINIMAX_M3_DROP_UPLOAD_CACHE` | MiniMax-M3.1 | Release each uploaded chunk's page cache once it is on the device (`posix_fadvise` on Linux, `F_NOCACHE` on macOS; a warned no-op elsewhere). `0` keeps the pages cached | on | not registered | no |
| `TS_MINIMAX_M3_DSPARK_GRAPHS` | MiniMax-M3.1 DSpark drafter | Draft graphs kept (one per sequence slot, least recently used first) | `4` | not registered | no |
| `TS_MINIMAX_M3_EXEC_DEBUG` | MiniMax-M3.1 (diagnostic) | `1` logs one stderr line per state-changing executor call (what the scheduler asked for, the active slot's head and the wall time), plus slot growth, graph builds and host time per phase | off | not registered | no |
| `TS_MINIMAX_M3_PROFILE` | MiniMax-M3.1 (diagnostic) | `N` > 0 times every graph node synchronously and prints a per-backend, per-op summary every N graph computations. Defeats CUDA graphs, turns pipelined prefill off, and is refused under `--tp` | `0` | not registered | no |

## Out-of-Matrix DeepSeek V4 / V4.1 Knobs

These configure the DeepSeek whole-model executors. DeepSeek V4 has three of
them (direct CUDA, native ggml, pure C#); DeepSeek V4.1 serves on one native
`ggml_cuda` graph, with `ggml_cpu` as a scalar correctness path and its own
pure-C# and direct-CUDA executors as portability paths. None are
registered in `EnvVarMatrix.All`, so the default TestMatrix sweep does not touch
them. The full context is in the [V4 card](models/deepseek4.md) and the
[V4.1 card](models/deepseek41.md).

| Variable | Applies to | Effect | Baseline | In matrix |
|---|---|---|---|---|
| `TS_DSV4_UBATCH` | V4 and V4.1 | Prefill micro-batch width. Unset, V4.1 on a ggml GPU backend lets the native loader choose 1024, 512 or 256: the widest that needs no more routed-expert CPU layers than 256 would (or than an explicit `--n-cpu-moe`), logged as `[dsv4] prefill ubatch: N (auto; ...)`. A resident routed-expert layer costs about the same per chunk at each width, so wider is cheaper per prefill token. Any explicit positive value is used verbatim and turns the choice off; `256` restores the previous fixed V4.1 default | V4.1: auto on ggml GPU backends, `256` on the CPU executors and direct CUDA; V4: `1024` (`512` on the pure-C# executor) | no |
| `TS_DSV4_THREADS` | V4 and V4.1 | Native thread pool for GPU-only loads. CPU expert offload uses the detected available parallelism instead, and `--cpu-moe-threads N` / `TS_CPU_MOE_THREADS` sets that. On the pure-C# `--backend cpu` executor it sizes that executor's own worker pool and defaults to `ProcessorCount` rather than min(cores, 32) | min(cores, 32) | no |
| `TS_DSV4_PERF` | V4 and V4.1 | `1` prints per-stage timing | off | no |
| `TS_DSV4_ASYNC_SEND` | V4.1 `--backend cuda` distributed layer split, sending nodes | `1` sends a prefill ubatch's boundary payload during the next ubatch, after that ubatch's first-GPU layers are queued, so a node's GPUs overlap consecutive ubatches; `2` sends it at the next ubatch's own send; unset or `0` sends synchronously (S9). The last ubatch of a forward always sends before the forward returns. Timing only: logits are bit-identical in every mode. Opt-in: never run with each stage on GPUs of its own, so neither its gain nor its event wait across two CUDA contexts has been measured | `0` | no |
| `TS_DSV4_VRAM_RESERVE_MB` / `TS_DSV4_GRAPH_CACHE` / `TS_DSV4_LOAD_THREADS` / `TS_DSV4_LOAD_CHUNK_MB` / `TS_DSV4_MOE_MMAP` | V4 and V4.1 | Placement headroom, graph-cache depth, weight-load parallelism, and whether host-resident experts are multiplied in place out of the GGUF mapping | see the cards | no |
| `TS_DSV41_RETAINED_CACHE_MB` | V4.1, native executor | Budget for the native slots retained for finished conversations' next turns (retention is always on, except while a DSpark drafter is loaded); a value that is not a positive number keeps the default | `2048` | no |
| `TS_DSV41_TP_HOST_TOKENS` | V4.1 CUDA routed-MoE TP, host-driven node only (`TS_DSV4_FUSED=0`) | The default `--tp` path on the fused CUDA backend stages every rank exchange through pinned memory on the GPUs' own queues and ignores this. On the host-driven node: integer `0`–`4096`: batches at or below this token count use the exact pinned-host activation gather; larger batches use private F32 NCCL when available. `0` forces available device gathering. Automatic `16` applies only when NCCL has selected `NCCL_P2P_DISABLE=1`, based on six-A40 paired measurements; other configurations default to `0`. `TS_GGML_TP_F32_NCCL=0` still forces the host fallback for all batches. | automatic | no |
| `TS_DSV41_ENGRAM_DEVICE` | V4.1 (ggml backends and `--backend cuda`) | `1` requires GPU-resident Engram tables and fails if they do not fit; `0` forces host mappings, which is also what a bit-exact CPU-oracle comparison needs. Unset is automatic and conservative: GPU-resident unless that would force routed-expert CPU offload | auto (GPU-resident when it fits) | no |
| `TS_DSV41_ENGRAM_WARM` | V4.1, host mappings only | Reads the Engram table pages so a lookup is a RAM read instead of a storage round trip. Unset warms in the BACKGROUND after the model is serving (the default); `1` warms synchronously during load as before; `0` never warms. Worth 200-252 → 452-492 prefill tok/s and 23-26 → 31-33 decode tok/s on eight A40s at Q4_K_M, and irrelevant on the GPU-resident default. Costs the table bytes in host page cache (60 GiB at Q2_K, 103 GiB at Q4_K_M) and the time to read them: the synchronous form took 311.3 s for 103 GiB with the old page walk on the seven-A40 lane; the `pread` warm (`TS_DSV4_WARM_PREAD`) measured 2.24-2.54 GiB/s on that storage, i.e. 41-46 s of reads | background warm | no |
| `TS_DSV4_GRAPH_CACHE_HEADROOM_MB` | V4 and V4.1 | Device memory the graph cache must leave free for the next graph, on top of the largest entry it holds. Least-recently-used entries are freed until it fits. `0` restores the pure entry-count cap, which four concurrent 10.8k-token prefills could run out of memory | `1024` | no |
| `TS_DSV41_ENGRAM_THREADS` | V4.1, host mappings only | Persistent lookup workers, `1`-`32`. One token selects 24 independent rows per table, so serial reads mean serialized page faults. On `--backend cuda`: rows read at once over (token, column), the calling thread included, on the reader's own threads (never the .NET thread pool), which the prefetch shares, so a reader never has more rows in flight; `1` restores the S9 gather exactly (tokens in parallel, columns serial) and turns the advice off unless set | min(16, hardware threads) | no |
| `TS_DSV41_ENGRAM_RANDOM` | V4.1, host mappings on Linux | Random-access advice for the mapped Engram ranges. `0` disables, `1` forces | auto | no |
| `TS_DSV41_ENGRAM_PREFETCH` | V4.1 `--backend cuda`, host mappings | When a forward starts, the reader's threads read ahead every row the forward will gather from this node's tables, in layer order, while the host issues the layers; it hashes a copy of the history and only reads, and a gather always goes first. A reset, a failed forward and a page-cache eviction stop it. `1` on, unset or `0` off. Opt-in: its gain on cold network storage has not been measured on the full model | off | no |
| `TS_DSV41_ENGRAM_OWNER` | V4.1 `--backend cuda` distributed layer split, host mappings | Which node keeps (maps, warms, gathers) each Engram table. Unset or `local`: the node that runs the table's layer (S9). `balance`: tables in index order, each to the node with the fewest table bytes so far, to its reader on a tie (the 1017 at 2x2: node 0 keeps layer 14's slice, node 1 layer 1's, so each fits its cgroup warm). A table kept by another node crosses as row ids plus raw rows at the start of each forward; bit-identical either way. Part of the configuration digest, compared by meaning (unset equals `local`). `balance` is opt-in until a two-node run has measured it | `local` | no |
| `TS_DSV41_ENGRAM_WARM_THREADS` | V4.1 `--backend cuda`, host mappings | Threads of the Engram warm (`pread` on per-thread handles), `1`-`64`. The warm reads only this node's tables, only pages `mincore` reports missing, and only when the cgroup (limit minus anonymous and shmem memory minus 8 GiB) and `MemAvailable` hold them | `16` | no |
| `TS_DSV4_LOAD_CONTIGUOUS` | V4 and V4.1 | `0` reverts the weight loader to handing its chunk jobs out from a shared cursor, which makes every reader stride `threads x chunk` through the file instead of reading one contiguous run. Kept only so the contiguous default can be A/B'd: on a MooseFS mount it measured 2.5x slower (363-382 s against 144-155 s to load the Q4_K_M release on eight A40s) | on | no |
| `TS_DSV4_WARM_PREAD` | V4 and V4.1 host mappings; V4.1 TP routed-weight loading | Controls the host-expert prefault (`--n-cpu-moe`), synchronous/background Engram warming, and V4.1 TP's per-layer routed gate/up/down source preparation before GPU upload. Unset or `1` uses bounded parallel `pread` with `TS_DSV4_LOAD_THREADS`, skips ranges `mincore` reports resident, and reports read errors with shard/offset. TP warms only the current layer after opening its source mappings, rather than the whole checkpoint; it does not warm Engram tables through this path. `0` disables TP source preparation and restores the existing host-expert/Engram page-touch walks (256 MiB prefault spans and 8 MiB Engram chunks). The historical seven-A40 8 GiB range test measured 2.24-2.54 GiB/s for `pread` versus 0.62-0.74 GiB/s for page walks; this is not a TP model-load speedup measurement or a cold-storage guarantee | on | no |
| `TS_DSV4_LOAD_DROP_CACHE` | V4 and V4.1 ordinary layer uploads; the direct engine (`--backend cuda`) with the same rule, its upload being this node's weights and its host-mapped weights the Engram tables it keeps (Linux; shards on tmpfs are left alone) | Whether each uploaded weight chunk's page cache is released once the chunk is on the device. The private V4.1 TP routed-expert uploader does not apply this setting; its uploaded source pages remain in reclaimable file cache. Unset is automatic: drop when the upload bytes plus the host-mapped weights (experts, Engram tables) plus 8 GiB exceed the host allowance (the cgroup limit), keep otherwise and whenever the allowance is unknown, so a reload of a GPU-resident checkpoint stays warm; the load logs the decision with the three numbers. `1` always drops, `0` never drops (the previous default). The seven-A40 lane (263.0 GiB upload + 151.2 GiB mapped against 326.9 GiB) drops. Dropping costs 5.9-7.3 ms per resident 64 MiB chunk on the MooseFS mount and cannot speed the upload itself; its expected gain is on the prefault and Engram warm that follow | auto | no |
| `TS_DSV41_REWIND_CHECKPOINT` | V4.1, native executor | `0` drops the per-slot rewind checkpoint (a shadow copy of the raw sliding-window and compressor-state rings, taken at every prompt boundary). Without it a partial KV reuse can only rewind as far as the live ring reaches — 385 positions on the released checkpoint — so a multi-turn thinking chat re-prefills instead. Costs ~21 MiB of VRAM per sequence slot | on | no |
| `TS_DSV41_SPARSE_FA` | V4.1 | Sparse prefill attention. On the owned F32 CUDA path it is **on by default** for launches of more than 8 queries over at least 2,048 keys: each query attends to its sliding window plus the indexer's selection (at most 640 keys) instead of every key. CUDA runs that launch on an MQA kernel that loads each key tile once for four heads: 1,024 queries cost a constant 16.1-16.3 ms on an A40 against 22.5 ms tiled at 2,560 keys and 63.8 ms at 6,656 (512 x 8,960: 8.0 ms, tiled 108 ms), within 5.3e-7 relative L2 of an F32 reference; decode, DSpark verify and prompts below the threshold keep the dense kernels bit for bit. `0` restores tiled prefill. `1` additionally opts ggml's flash-attention path (non-CUDA GPUs, the CPU backend) into its mask-compacted kernel for one query or at least 16,384 keys, which has documented F16 differences | on for the owned CUDA path; off for ggml flash attention | no |
| `TS_DSV41_COMPACT_RAW_GATHER` | V4.1 | `1` selects compact gathering for the raw sliding window. Opt-in, with the same floating-point caveat | off | no |
| `TS_DSV41_MQA_DECODE` | V4.1 owned F32 CUDA attention | V4.1's 64 heads share one F16 K=V head of width 512, so CUDA runs that attention on an MQA kernel: a block loads each key tile once for four heads, splits its visible keys over four streams, and never loads a masked key. Decode-class launches (1-8 queries) and compacted prefill both use it, F32 throughout and still bit-identical across launch widths. One decode layer measured 171 -> 42 us on an A40 (sparse prefill 512 x 8,960 keys: 34.4 -> 8.0 ms). `0` restores the per-head kernel for A/B and diagnosis | on | no |
| `TS_DSV41_KV_MIRRORS` | V4.1 multi-GPU (layer split, and `--tp`, whose attention keeps layer placement) | Keeps a copy of each shared compressed / indexer cache on every GPU whose layers read it (the source's new rows are copied each step), so a layer reads its own GPU's copy instead of pulling selected rows or the whole visible cache across PCIe through the host. Logits are bit-identical either way; 5x A40 decode went 32.6 -> 34.5 tok/s at 2K context. A mirror that cannot be allocated is skipped with a warning. `0` reads across GPUs | on (multi-GPU) | no |
| `TS_DSV41_YOCO_KEEP` | V4.1 prefill (ggml executor: ggml_cuda, ggml_cpu) | YOCO bounded replay, after DeepInfer's YOCO prefill truncation. **On by default** with DeepInfer's keep of 140. From the first ratio-1 layer on (layer 20 of 40), every layer reads the compressed cache that layer 20 writes from its attention input, so a multi-token call runs those layers only over its last N positions, and each replayed position's sliding window is cut to the replay. That makes prefill **approximate**: `0` runs every layer over every position; other values must be at least the sliding window + 8 (136). Decode steps, DSpark verifies, calls with image tokens and one-token prefill micro-batches always run every layer (with `TS_DSV4_UBATCH=1` the default notes that replay is inactive and an explicit value is refused). A checkpoint whose sliding window is wider than 132 keeps at least the window + 8 by default. A rewind into a long call's skipped positions (more than 13 tokens before its end, and after its start) re-prefills instead of reusing the cache, with one log line. The server applies it per prefill call, i.e. per chunk of a long prompt (up to 4,096 tokens, fewer under concurrency). A checkpoint whose layout cannot replay (a layer after 20 that owns its own compressed cache, an Engram table on layer 20 or later, an indexer candidate source before layer 20, a DSpark drafter reading layer 20 or earlier) runs exact prefill with one note under the default and is refused with an explicit value. `--backend cuda` and `--backend cpu` do not implement it: they note that once and refuse an explicit non-zero value. With 140 on 5x A40 (UD-IQ2_XXS, random tokens): `--layer-split 5` prefill 463.5 -> 844.3 tok/s at 2,048 tokens and 444.3 -> 906.2 at 8,192; `--tp 5` 562.9 -> 1,046.7 and 540.8 -> 1,082.7. Wikitext-2 at 4,608-token windows (2,304 prefilled, the rest scored token by token, 10 chunks): PPL 2.4820 -> 2.4970 (+0.6%), mean KLD to llama.cpp 0.0160 -> 0.0172, concentrated in the first tokens after the prompt (0.026 over the first 128 against 0.015 exact) | `140` (on) | no |
| `TS_DSV41_ALLOW_NON_CUDA_GPU` | V4.1 | `1` permits `ggml_vulkan` / `ggml_metal`, where the ordinary graph runs on the GPU and only the architecture-specific ops fall back to the CPU backend, at a host round trip each. Opt-in because the refusal it lifts was closing a *silent* fallback | off | no |
| `TS_DSV41_VISION_FA` / `TS_DSV41_VISION_BF16_GEMM` | V4.1 vision companion | `TS_DSV41_VISION_FA=1` selects flash attention with F16 intermediates in the image encoder (larger real-image feature differences); `TS_DSV41_VISION_BF16_GEMM=0` selects the diagnostic F32-promoted matrix path | dense F32 attention, BF16 GEMM with F32 accumulation | no |
| `TS_DSV41_TRACE_DIR` / `TS_DSV41_VISION_TRACE_DIR` | V4.1 (diagnostic) | Directories for text-graph and vision-encoder tensor dumps. Both retain intermediates and add device transfers — leave unset for benchmarks | unset | no |
| `TS_DSV4_CPU_TRACE_DIR` / `TS_DSV4_CUDA_TRACE_DIR` | V4.1 (diagnostic) | The same per-tensor dumps from the pure-C# and direct-CUDA V4.1 executors, under one file naming (`p<pos>_blk<NN>_<stage>.f32`, `.i32` for the direct engine's selected rows and experts); `eng/validation/compare-dsv4-traces.py` diffs two of them, or one against `TS_DSV41_TRACE_DIR` (whose `v41.<NN>.<stage>` names it maps onto these), tensor by tensor to find the first divergence. Tracing only copies the stages out: the logits are the same to the bit | unset | no |

## Out-of-Matrix GLM 5.x (`glm-dsa`) Knobs

These configure the GLM 5.x (`glm-dsa`) executor — the native whole-model ggml
path used by `ggml_cuda` / `ggml_vulkan` / `ggml_cpu` / `ggml_metal`, and the
managed per-op path used by `cpu` and `cuda`. None are registered in
`EnvVarMatrix.All`, so the default TestMatrix sweep does not touch them; the
full list with context is in the [GLM card](models/glm.md#environment-knobs).
The tensor-parallel knobs (`TS_GLM_TP_SHARD`, `TS_GLM_TP_OVERSUBSCRIBE`) live in the
TP table below.

| Variable | Applies to | Effect | Baseline | Values swept | In matrix |
|---|---|---|---|---|---|
| `TS_GLM_NATIVE` | GLM 5.x | `0` runs the managed per-op path on a GGML backend instead of the native whole-model graph — the A/B that proves the two agree | `1` (native) | `0`, `1` | no |
| `TS_GLM_UBATCH` | GLM 5.x | Prefill micro-batch. `2048` is faster on long prompts when VRAM allows: pp2048 1145.8 vs 918.9 t/s on 3x RTX PRO 6000 | `1024` | `512`, `1024`, `2048` | no |
| `TS_GLM_THREADS` | GLM 5.x on `ggml_cpu` | CPU-backend thread count; every usable CPU instead with `--n-cpu-moe` / `--cpu-moe` or no GPU, and `--cpu-moe-threads` (then an inherited `TS_CPU_MOE_THREADS`) overrides either | min(cores, 32) | — | no |
| `TS_GLM_OP_OFFLOAD` | GLM 5.x on GGML | Scheduler op-offload; turned off automatically once any layer's experts are host-resident | auto | `0`, `1` | no |
| `TS_GLM_VRAM_RESERVE_MB` | GLM 5.x on GGML | Per-device headroom the layer split leaves for compute buffers before it starts placing layers | `3072` | — | no |
| `TS_GLM_GRAPH_CACHE` | GLM 5.x on GGML | How many built+allocated graphs are kept, so a repeated shape replays instead of rebuilding | `8` | — | no |
| `TS_GLM_PLAN_SLOTS` | GLM 5.x on GGML without `MAX_CONTEXT` | How many sequence slots the automatic context size leaves room for; each slot holds a whole context, so N concurrent requests at full context need N | `1` | `1`-`256` | no |
| `TS_GLM_SLOT_HEADROOM_MB` | GLM 5.x on GGML | Memory a sequence slot beyond the planned ones must leave on each device besides room for another graph as large as the largest cached one; a slot that does not fit is refused and its request waits | `1024` | — | no |
| `TS_GLM_NODES_PER_LAYER` | GLM 5.x on GGML | Graph node budget per layer per rank | `256` | — | no |
| `TS_GLM_MOE_MMAP` | GLM 5.x with `--n-cpu-moe` | `0` copies host-resident experts into a private buffer instead of multiplying them in place out of the GGUF mapping | `1` (mapped) | `0`, `1` | no |
| `TS_GLM_LOAD_THREADS` / `TS_GLM_LOAD_CHUNK_MB` | GLM 5.x | Weight-load parallelism and chunk size — 16 reader threads across the six shards move 218 GiB in ~37 s (5.9 GiB/s) from a warm page cache | `16` / `64` | — | no |
| `TS_GLM_TRACE` | GLM 5.x (diagnostic) | Layer list (or `all`) to dump per-layer activation sums in `llama-eval-callback`'s layout, for diffing against llama.cpp | unset | — | no |
| `TS_GLM_BD_DEBUG` | GLM 5.x (diagnostic) | `1` narrates each batched decode step: which slots took part, whether the graph was reused or rebuilt, and how far it got | `0` | `0`, `1` | no |
| `TS_GLM_DEBUG` / `TS_GLM_DEBUG_LAYERS` | GLM 5.x on the managed per-op path (`cpu` / `cuda` / `TS_GLM_NATIVE=0`, diagnostic) | Per-layer activation trace: shape, sum and leading values of every named intermediate, tagged to match `llama-eval-callback` so the two can be diffed tag by tag. `TS_GLM_DEBUG=1` traces layer 0 only; `TS_GLM_DEBUG_LAYERS` takes a layer list. For the native executor use `TS_GLM_TRACE` instead | unset | — | no |

## Out-of-Matrix Tensor Parallelism & Distributed Inference Knobs

These variables configure tensor parallelism (splitting a model across multiple
GPUs) and distributed multi-node TP over a peer-to-peer TCP mesh. They are
not registered in `EnvVarMatrix.All` and are not swept by the default TestMatrix
config — TP requires multiple GPUs, which the standard single-GPU test harness
does not exercise. TP runs on the direct `cuda` backend and on the GGML CUDA /
Vulkan backends (`ggml_cuda`, `ggml_vulkan`). `TENSORSHARP_TP_DEGREE`,
`TENSORSHARP_TP_NODE_ID`, and `TENSORSHARP_TP_PEERS` are also settable via the
`--tp`, `--tp-node-id`, and `--tp-peers` flags on both `TensorSharp.Cli` and
`TensorSharp.Server`.

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TENSORSHARP_TP_DEGREE` | all autoregressive models; `cuda`, `ggml_cuda`, `ggml_vulkan` backends | Number of local GPUs to split the model across (Megatron-LM column/row-parallel) | `1` (single GPU) | not registered | no |
| `TENSORSHARP_LAYER_SPLIT_DEGREE` | architectures supporting layer split | Local whole-layer placement GPU count, equivalent to `--layer-split N`; mutually exclusive with TP | `1` | not registered | no |
| `TENSORSHARP_LAYER_SPLIT_DEVICES` | Shared GGML layer executors, including Mistral 3 and Qwen 3.8 Flash Next | Comma-separated device ordinals, e.g. `0,2`; independent of `TENSORSHARP_TP_DEVICES`; native GLM/DeepSeek use `CUDA_VISIBLE_DEVICES` | `0..N-1` | not registered | no |
| `TENSORSHARP_TP_DEVICES` | local TP on the GGML backends | Comma-separated GPU ordinals the ranks map to (e.g. `0,2`). Muse Spark's native `--tp` uses the first N CUDA GPUs and refuses a load with this set; use `CUDA_VISIBLE_DEVICES` there | `0..tp-1` | not registered | no |
| `TENSORSHARP_TP_NODE_ID` | Architectures/backends supporting distributed TP or layer split | This node's 0-based ID; must be set with `TENSORSHARP_TP_PEERS` | unset (disabled) | not registered | no |
| `TENSORSHARP_TP_PEERS` | Architectures/backends supporting distributed TP or layer split | Comma-separated `host:port` list of all nodes; must be set with `TENSORSHARP_TP_NODE_ID`; IPv6 uses `[address]:port` | unset (disabled) | not registered | no |
| `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS` | distributed TP and layer split | Deadline for connection retries, inbound accepts and mesh handshakes | `120` seconds | not registered | no |
| `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS` | distributed TP and layer split | Read/write inactivity timeout on peer sockets; heartbeats keep idle control waits alive | `300` seconds | not registered | no |
| `TENSORSHARP_TP_OPERATION_TIMEOUT_SECONDS` | distributed TP and layer split | Complete data-frame deadline, including heartbeat-only waits and partial transfers; idle controls and load barriers exempt | receive timeout | not registered | no |
| `TENSORSHARP_TP_SOCKET_BUFFER_BYTES` | distributed TP and layer split | Optional socket send/receive buffer budget, at least 4096 bytes; tune only against measured routes | OS default | not registered | no |
| `TENSORSHARP_TP_DISABLE_P2P` | local TP, `cuda` backend | `1` forces every cross-GPU transfer through host staging instead of CUDA peer-to-peer DMA (matches no-peer hardware such as A16 vGPU profiles) | off (P2P used when the pair passes the DMA self-test) | not registered | no |
| `TENSORSHARP_TP_HOST_ALLREDUCE` | local TP, `cuda` backend | `1` runs the local AllReduce through host memory (device→host, sum, host→device) instead of the device-to-device path — diagnostic fallback | off (device-to-device) | not registered | no |
| `TS_GGML_TP_PARALLEL` | local TP, GGML backends | `0` drives the ranks sequentially instead of concurrently (diagnostic) | on (concurrent rank workers) | not registered | no |
| `TS_GGML_TP_FUSED_MATMUL` | local TP, GGML backends | `1` submits both ranks' linears from one thread; allocates a device buffer per rank per call and measured 2.3× slower on Qwen 3.5 35B | off (generic per-rank path) | not registered | no |
| `TS_GGML_TP_DEVICE_AR_THRESHOLD` | local TP, GGML backends | Element count above which AllReduce uses the device collective instead of the host reduction | `262144` | not registered | no |
| `TS_GGML_F32_RESIDENT` | GGML backends | `0` binds F32 linear weights per call instead of keeping them device-resident (diagnostic) | on (device-resident) | not registered | no |
| `TS_GEMMA4_TP_FUSED_MOE` | Gemma 4 MoE under TP on GGML | `0` falls back from the fused whole-model MoE trunk (Megatron split inside each expert) to the whole-expert per-op path | on (fused trunk) | not registered | no |
| `TS_GLM_TP_SHARD` | GLM 5.x under TP on GGML | Which halves of the split are applied: `1` heads, `2` routed experts, `3` both. The experts are split row-wise inside every expert rather than by expert id, because `ggml_mul_mat_id` needs a token's selected expert ids to stay distinct | `3` (both) | `1`, `2`, `3` | no |
| `TS_GLM_TP_OVERSUBSCRIBE` | GLM 5.x under TP on GGML | `1` packs several ranks onto one GPU so the split can be checked for correctness on a single-GPU machine | `0` (one rank per GPU) | `0`, `1` | no |
| `TS_Q4E_LAYER_SPLIT` | Qwen 3.8 Flash Next (`qwen4exp`) multi-GPU layer split under `--layer-split N` | Explicit layer counts per GPU, comma-separated (e.g. `20,28`), instead of the automatic VRAM balance; throws rather than silently ignoring a value it cannot honour. `--layer-split N` places whole layers; the independent `--tp N` mode shards routed/shared FFN channels and does not use this layer-placement override | automatic (layers bin-packed to each device's free VRAM) | not registered | no |
| `TS_Q4E_RETAINED_CACHE_MB` | Qwen 3.8 Flash Next (`qwen4exp`) retained reuse | Byte budget in MiB for retained conversations plus shared-prefix checkpoints, clamped by measured memory headroom. Unset, half the measured headroom is the budget (the rule Qwen 3.5 applies to idle holders), and 4096 applies only where no headroom can be measured; the fixed 4096 default held three of four concurrent 1.3 GB conversations on a 4x A40 tensor split. The Radix prefix cache owns eviction, so a holder that does not fit is refused (reported once). `0` or an unparsable value declines every retention | half the measured headroom (`4096` unmeasured) | not registered | no |
| `GGML_CUDA_ALLREDUCE` | local TP, `ggml_cuda` | `nccl` / `internal` / `none` — passed through to ggml's collective selection; setting it explicitly also skips the pre-flight probe | auto (NCCL when the build finds it and it passes the probe) | not registered | no |
| `TS_GGML_TP_CUDA_GRAPHS` | local TP, `ggml_cuda` | `0` turns CUDA graph capture off for multi-GPU runs. Capture is ON by default under TP because a tensor-parallel token is dozens of small per-rank submissions that replay far more cheaply than they re-issue (4×A40: Qwen3.5-9B tp4 88 → 128.5 tok/s, Qwen3.5-35B-A3B tp2 71.3 → 104.1). It was historically disabled over a capture-poisoning hazard that no longer applies — ggml captures with `cudaStreamCaptureModeRelaxed`. The opt-out is translated into a native `GGML_CUDA_DISABLE_GRAPHS` before the first backend call, because ggml latches that value on first use | capture enabled | not registered | no |
| `TS_GGML_TP_AR_PROBE` | local TP, `ggml_cuda` | `0` skips both pre-flight probes; `force` re-probes, ignoring the cached verdicts (`~/.cache/tensorsharp/tp-collective-probe`). Before model load the group checks that peer copies between advertised device pairs actually deliver bytes, and that one small NCCL AllReduce completes end to end — some cloud hosts advertise P2P that never arrives, and NCCL's first collective then spins every GPU forever. A failed peer check keeps NCCL but takes peer transport away from it (`NCCL_P2P_DISABLE=1`), which is what preserves a device collective past 2 GPUs | probes on, verdicts cached per driver/NCCL/GPU set | not registered | no |
| `TS_GGML_TP_AR_PROBE_MS` | local TP, `ggml_cuda` | Deadline for each probe (peer copy, then AllReduce) before that transport is declared broken; the collective then falls back to the pinned-host `internal` pipeline at 2 GPUs, or to the host reduction beyond it. `0` disables the probes | `10000` ms | not registered | no |
| `TS_GGML_TP_F32_NCCL` | Precision-sensitive local TP, `ggml_cuda` | `0` disables TensorSharp's F32 NCCL transport for diagnosis. Precision-sensitive generic plans retain F32 chunks or host reduction; DeepSeek V4.1 retains host row gathering. The default uses uncompressed NCCL operations, including a bit-preserving AllGather for DeepSeek's uneven activation shards, and honors an explicit `GGML_CUDA_ALLREDUCE=internal/none`. Ordinary TP plans are unchanged. | on when NCCL is available | not registered | no |
| `GGML_CUDA_AR_BF16_THRESHOLD` | local TP, `ggml_cuda` | Payload size above which ggml converts F32 collectives to BF16; TensorSharp raises ggml's default to 1 MB so decode-sized reductions stay exact | `1 MB` (set by `TSGgml_TensorParallelInit`) | not registered | no |
| `TS_QWEN35_LAYER_TRACE` | Qwen 3.5/3.6 | `1` prints a per-layer residual-stream summary for the first forward, from both the single-GPU and TP loops (diagnostic) | off | not registered | no |

## Out-of-Matrix Redis Shared-State Knobs

This variable configures optional Redis-backed state. It is not registered in
`EnvVarMatrix.All`, and `--redis-url` sets it. Redis holds no KV state.

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_RESPONSES_STORE_REDIS_URL` | `TensorSharp.Server` | Redis connection string for the OpenAI Responses API store; when set, `RedisResponsesStore` replaces the in-memory store | unset (disabled, in-memory) | not registered | no |

## Out-of-Matrix General Runtime Knobs

These variables are real runtime knobs, but they are not registered in
`EnvVarMatrix.All` today and are not swept by the default TestMatrix config.

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TENSORSHARP_UPLOAD_DIR` | `TensorSharp.Server` | Directory for uploaded media and extracted video frames; set an absolute path outside the application directory for immutable deployments | `uploads` beside the server binary | not registered | no |
| `TS_UPLOAD_MAX_MB` / `TS_UPLOAD_QUOTA_MB` / `TS_UPLOAD_TTL_HOURS` | `TensorSharp.Server` (also `--upload-max-mb`, `--upload-quota-mb`, `--upload-ttl-hours`) | Per-file cap on client uploads (HTTP 413 above it), total budget for the upload directory (HTTP 507 when exhausted), and the age after which files there are deleted. Jev's inline images are covered by the same limits | `500` / off / off | not registered | no |
| `TS_LORAS_DIR` | `TensorSharp.Server` (also `--loras-dir`, which wins) | Directory of text-model LoRA adapters (llama.cpp `.gguf` adapters, or `.safetensors` with `<tensor>.lora_a` / `.lora_b` tensors). Every top-level adapter loads at startup, after the model; `POST /v1/loras` loads more, only from inside this directory and named relative to it; at most 32 are held. A path that is not an existing directory is a startup error. An empty value means unset; an embedding server ignores it | unset (no adapters; `POST /v1/loras` answers 403) | not registered | no |
| `TENSORSHARP_PREFIX_CACHE_DIR` | `TensorSharp.Server` | Root for the shared-prefix checkpoint files that let a restart skip the shared prompt's prefill; each model gets its own `<model-file-stem>-<hash>` subdirectory with at most two files. `--no-prefix-cache` turns persistence off. TensorAgent keeps its own under the app's cache directory; the CLI keeps checkpoints in memory only | `prefix-cache` beside the server binary | not registered | no |
| `TS_NO_MULTI_AGENT` | `TensorSharp.Server` chat endpoints | Any value except `0` turns off sub-agent delegation, like `--no-multi-agent` | unset (delegation on) | not registered | no |
| `TS_THINKING_BUDGET` | chat turns with thinking on, in `TensorSharp.Server` and the interactive CLI | Tokens a reasoning block may run. Families with a trained closing token (DeepSeek V4.1, Nemotron 3.5 / Omni, Qwen 3.5-3.8, GLM 5.x) then close it and answer inside `max_tokens`; Qwen and GLM first write Qwen's hand-over sentence. Other families stop the turn with finish reason `thinking_budget`. `0` disables | 75% of `max_tokens` when it is at least 512, else none | not registered | no |
| `TS_JEV_MAX_BODY_MB` / `TS_JEV_MAX_CANVAS` / `TS_JEV_MAX_PENDING` | `POST /v1/systemone` (Jev, DiffusionGemma) | Request-body limit in MiB (1-64), answer-canvas width per question chunk in tokens (8-4096, also bounded by the checkpoint), and how many requests may be admitted at once (1-1024) before the endpoint answers HTTP 529. An out-of-range value is an error, not clamped | `8` / `64` / `32` | not registered | no |
| `TS_NEMOTRON_AUDIO_MMPROJ` | Nemotron-H with an `--mmproj` | Audio companion GGUF (NVIDIA `sound_encoder.*` / `sound_projection.*` tensors) to load the Parakeet audio tower from instead of the `--mmproj` file, so a vision mmproj and an audio companion can be used together. Audio stays refused (HTTP 400) unless the file's tensors validate; see `docs/models/nemotron.md` §4.7 | unset (audio tower read from `--mmproj` when it has the tensors) | not registered | no |
| `TS_PDF_MAX_PAGES` | PDF document input (CLI `--pdf`, server `/api/upload`) | Cap on the number of PDF pages read for text extraction and page-image rendering | `0` (all pages) | not registered | no |
| `TS_GGUF_PREFAULT` / `TS_GGUF_PREFAULT_THREADS` / `TS_GGUF_PREFAULT_RESIDENT` | Model load through `GgufReader` | `PREFAULT=0` skips parallel page-cache warming. Threads are capped at the processor count. The selected tensor ranges across all shards share one half-of-process-memory-budget limit; sparse Qwen PLE tables are excluded. Linux checks page residency without copying cached ranges; `RESIDENT=0` forces reads for A/B measurement. Other platforms read selected ranges normally; iOS/tvOS skip warming. The memory budget is not a measurement of currently free host RAM. | on, `min(16, cores)`, residency check on | not registered | no |
| `TS_DUMP_LOGITS` | all models, all backends, CLI and server | Path the logits that choose the run's FIRST GENERATED TOKEN are written to, once per process, as raw float32 (little-endian): the last prompt row after the WHOLE prompt has been processed, whichever path ran it - one forward or several chunks (prefix-cache checkpoint or recurrent block boundaries, chunk caps), per-sequence, fused or batched prefill, `--tp` / `--layer-split`, a model's own engine (q4e, GLM, DeepSeek V4, ...), or the CLI `--benchmark` (`Forward`, or `ForwardRefill` with `--bench-chunked`). It is taken where the engine samples a request's first token (and after the benchmark's / `--test`'s prompt forward), so it is never a warm-up forward, never a decode forward, and never a warm-up request: requests submitted inside a warm-up (`--warmup-runs`, the server's and TensorAgent's prefix-cache warm-up, the benchmarks' warm-up passes) are excluded for their whole life, even when a cancelled warm-up is still in the engine, while a real request submitted alongside one is not. Under concurrent requests the first request to sample is the one dumped, so compare single-request runs. A `<path>.json` sidecar records the vector's SHA-256, the prompt length and token-id fingerprint, the number of media spans and the LoRA adapter (media CONTENT is not fingerprinted), prefix-cache reuse, the scheduler plan path each prefill forward ran under (a label, not the kernel; the CLI's direct loops record one step for the whole prompt, so do not assert forward counts from it), and the first token. Both files are removed when a process starts (CLI start-up, each inference engine) so a run that never reaches its first token leaves none. `eng/validation/compare-parallel-logits.py` refuses to compare two vectors whose sidecars name different prompts, media counts or adapters, a sidecar that does not describe the vector beside it (a stale sidecar: hash, count or argmax mismatch) and an unreadable sidecar; `--require-sidecar` also refuses a vector with none. (Before this, the dump held the first non-warm-up `ModelBase.Forward`, which is a prompt CHUNK - or the first decode step when the prompt never ran through `Forward` - whenever the prompt took more than one forward, so two configurations could compare different computations.) Lets two backends be compared by logit vector instead of by generated text, where greedy decoding turns a near-tie into a visibly different sentence. Not written by DiffusionGemma (no autoregressive first token) | unset (no dump) | not registered | no |
| `TS_FUSED_QKNORM_ROPE` | Qwen 3.5 / 3.6 text-only prefill on the direct `cuda` backend | Fused QK-Norm + NeoX-RoPE CUDA kernel; `0` falls back to separate norm + RoPE ops (multimodal MRoPE and other backends always use the separate path) | ON | not registered | no |
| `TS_CUDA_QMM_F16GEMM_MIN_ROWS` | direct `cuda` backend | Activation-row threshold above which quantized matmuls dequantize the weight once to F16 and run a tensor-core cuBLAS GEMM (the ggml-style prefill route) instead of the block-tile quant kernels | `32` | not registered | no |
| `TS_CUDA_QMM_F16GEMM_MAX_MB` | direct `cuda` backend | F16 weight-scratch cap in MB; weights above it (e.g. the LM head) keep the quant kernels | `768` | not registered | no |
| `TS_CUDA_Q80_VEC_MIN_OUT` | direct `cuda` backend | Minimum output width for the Q8_0 dp4a matvec (diagnostic gate) | `0` | not registered | no |
| `TS_CUDA_Q80_MMQ_MAX_ROWS` | direct `cuda` backend | Q8_0 matmuls with 32 to this many activation rows run the direct int8 tensor-core GEMM over raw Q8_0 blocks (mma.m16n8k32, ggml MMQ-style); above it the F16 GEMM route wins (MMQ weight sweeps grow as ceil(rows/128)) | `512` | not registered | no |
| `TS_CUDA_PREFILL_GRAPH_MAX` | direct `cuda` backend | Cached prefill + decode CUDA graphs kept (LRU-evicted; each pins its captured working-set pool blocks). Qwen 3.5 / 3.6 text-only prefill and decode capture their per-op layer loop as a graph and replay it (bit-identical results; any capture failure falls back to the plain run) | `4` | not registered | no |
| `TS_CUDA_PREFILL_GRAPH_LOG` | direct `cuda` backend | Log graph capture/replay/abort events (`1`) | OFF | not registered | no |
| `TENSORSHARP_CUDA_POOL_LARGE_MB` | direct `cuda` backend | Budget for the global large-block (≥ 2 MB) device-memory cache; keeps prefill-sized activations pooled instead of re-issuing cuMemAlloc/cuMemFree per layer | `1024` | not registered | no |
| `TS_CUDA_PROFILE` | direct `cuda` backend | Print CPU-fallback op and host↔device sync counters at exit (`1`), with call-site attribution (`2`) | OFF | not registered | no |

## Out-of-Matrix Knowledge Configuration

These host settings are not registered or swept by TestMatrix. They configure
the [personal knowledge library](knowledge-system.md), rather than changing the
chat model's numerical backend. Set them before starting the host; reindex when
the extractor or embedding identity changes. The standalone server opts in with
`TS_KNOWLEDGE_DIRECTORY`; TensorAgent has its own persistent library directory.

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_KNOWLEDGE_DIRECTORY` | standalone server | Enables the local library and `/api/knowledge`; chat requests still select `knowledgeCollections` explicitly | unset (disabled) | not registered | no |
| `TS_KNOWLEDGE_EMBEDDING_MODEL` | knowledge runtime | Local sentence-embedding GGUF, owned separately from the chat model | unset (BM25 only) | not registered | no |
| `TS_KNOWLEDGE_EMBEDDING_BACKEND` | knowledge encoder | Encoder backend | `CPU` | not registered | no |
| `TS_KNOWLEDGE_QUERY_PREFIX` / `TS_KNOWLEDGE_DOCUMENT_PREFIX` | knowledge encoder | Model-required prefixes; changing them changes vector-space identity | empty / empty | not registered | no |
| `TS_KNOWLEDGE_RERANK` | knowledge-enabled chat | `0` disables the one bounded model relevance pass; invalid output or timeout retains retrieval order | enabled when a chat model is available and there are at least three candidates | not registered | no |
| `TS_KNOWLEDGE_PDF_OCR` | PDF knowledge extraction | `auto`, `apple`, `tesseract`, or `none`; `none` preserves selectable text | `auto` | not registered | no |
| `TS_KNOWLEDGE_PDFTOPPM_PATH` / `TS_KNOWLEDGE_TESSERACT_PATH` | portable PDF OCR | Explicit absolute tool paths; opts out of TensorAgent's managed installation | resolve installed tools or use the supplied managed provisioner | not registered | no |
| `TS_KNOWLEDGE_TESSDATA_DIRECTORY` | portable PDF OCR | Explicit language-data directory; opts out of managed installation | provider configuration | not registered | no |
| `TS_KNOWLEDGE_OCR_LANGUAGES` | portable PDF OCR | Installed Tesseract language names | managed `eng+chi_sim`; `eng+chi_tra` available | not registered | no |

Standalone Windows/Linux hosts need installed tools or an explicitly supplied
`PdfOcrProvisioner`; configuring `auto` alone does not provision TensorAgent's
private OCR runtime. Platform, media-model and validation limits are in the
[knowledge guide](knowledge-system.md#validation-and-current-boundaries).

## Feature Coverage

The matrix feature catalog lives in
[`TensorSharp.TestMatrix/Matrix/FeatureCatalog.cs`](../TensorSharp.TestMatrix/Matrix/FeatureCatalog.cs).
The current feature set is:

| Feature | Driver | Capability gate |
|---|---|---|
| `pp512` | `--benchmark --bench-prefill 512 --bench-decode 0` | all models |
| `pp2048` | `--benchmark --bench-prefill 2048 --bench-decode 0` | all models |
| `tg128` | `--benchmark --bench-prefill 32 --bench-decode 128` | all models |
| `short_text` | `--input prompts/short_text.txt --max-tokens 64` | all models |
| `long_text` | `--input prompts/long_text.txt --max-tokens 64` | all models |
| `uploaded_text` | `--input prompts/upload_text.txt --max-tokens 64` | all models |
| `multi_turn` | `--multi-turn-jsonl multi_turn/three_turn.jsonl` | all models |
| `tools` | `--tools tools/weather_tools.json` | models whose matrix capability says tool calling is supported |
| `thinking` | `--think` | models whose matrix capability says thinking is supported |
| `image` | `--image media/apple.png --mmproj ...` | image-capable models with an mmproj |
| `audio` | `--audio media/sample.mp3 --mmproj ...` | audio-capable models with an mmproj |
| `video` | `--video media/sample.mp4 --mmproj ...` | video-capable models with an mmproj |

Default semantic checks are intentionally weak and catch catastrophic failures:
`blue`, `paged`, `08:01:12`, `alex` + `teal`,
`get_current_weather` + `tokyo`, `10:38`, and `apple` for the relevant text,
multi-turn, tools, thinking, and image features. Audio and video have no default
expected substring because the sample media is runner-provided.

## Filters

The runner filters the combinatorial product before execution:

1. Backend availability: CUDA and Vulkan backends are skipped on macOS (Metal
   is the GPU backend there); MLX requires Apple Silicon; GGML Metal requires
   macOS.
2. Model capability: image/audio/video/tool/thinking features are skipped when
   the discovered or configured model does not advertise that capability.
3. Projector availability: multimodal features require an mmproj path.
4. Env-var applicability: each `EnvVarSpec.AppliesTo` predicate decides whether
   a variable is meaningful for the `(model, backend, feature)` cell.

## Updating The Matrix

To add a new high-impact env var:

1. Register an `EnvVarSpec` in
   [`TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs`](../TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs).
2. Add it to `default_env_vars` in
   [`Defaults/matrix-config.json`](../TensorSharp.TestMatrix/Defaults/matrix-config.json)
   if it should run in the default sweep.
3. Add or update the row in this document and its Chinese counterpart.
4. If the variable changes feature applicability, update
   [`FeatureCatalog.cs`](../TensorSharp.TestMatrix/Matrix/FeatureCatalog.cs)
   or model discovery capability heuristics as needed.
