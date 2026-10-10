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
| `TS_SCHED_MAX_BATCHED_TOKENS` / `TS_SCHED_MAX_RUNNING_SEQS` / `TS_SCHED_PREFILL_CHUNK` / `TS_SCHED_SOLO_PREFILL_CHUNK` / `TS_SCHED_NUM_BLOCKS` / `TS_SCHED_BLOCK_SIZE` / `TS_SCHED_DECODE_QUANTUM` | all | Scheduler budgets: per-step tokens, in-flight sequences, prefill chunk while a decode runs (server `--prefill-chunk-size`), solo prefill chunk, pool blocks, tokens per block (16 for a family whose only cross-request reuse is its pages, else 256), decode quantum. See the [configuration table](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md#configuration) | `4096` / `16` / `256` / `8192` / `256` / `16` or `256` / `256` | not registered | no |
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
| `TS_SPEC` | Qwen 3.5/3.6/3.8, GLM 5.2, GLM-5.3, GLM-5.3-Flash (n-gram only), Gemma 4, Qwen 3.8 Flash Next, DeepSeek V4 / V4.1, Muse-Glimmer (CLI + server) | Enable speculative decode for solo sequences | OFF (`0`) | not registered | no |
| `TS_SPEC_TYPE` | all of the above | Speculation algorithm: `auto` \| `draft-head` \| `block` \| `ngram` | `auto` | not registered | no |
| `TS_SPEC_DRAFT` | all of the above | Max tokens drafted per speculative step (1-64) | `8` | not registered | no |
| `TS_SPEC_PMIN` | all of the above | Draft-confidence gate; meaning is per algorithm | per algorithm (`0.15` / `0.35` / `0`) | not registered | no |
| `TS_SPEC_DRAFT_MODEL` | models whose drafter ships as its own GGUF (CLI + server) | Path to a separate drafter, recognised from the file's architecture: Gemma 4's `gemma4-assistant`, Qwen 3.8 Flash Next's shared MTP head, a DeepSeek V4 DSpark (or experimental V4.1 `deepseek41-dspark`) drafter, or a DFlash / DFlash2 drafter (Muse-Glimmer, Qwen 3.5 family). Set by `--draft-model`, which also enables speculation unless `--no-spec` is given | none | not registered | no |

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
| `TS_SPEC_ADAPTIVE` | Speculative decoding (all drafters) | `0` disables the cost governor, so drafting is never measured against a plain baseline and never parked. For A/B measurement: a governor round's baseline steps are plain decodes and they are not free | ON | not registered | no |
| `TS_GGML_LOG_DEBUG` | GGML backends | `1` passes ggml's DEBUG log channel through instead of dropping it. Carries the CUDA backend's "CUDA graph warmup complete"/"reset" lines, which are the only way to see whether a graph is actually being CUDA-graph-captured | OFF | not registered | no |

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
| `TS_DSV4_VRAM_RESERVE_MB` / `TS_DSV4_GRAPH_CACHE` / `TS_DSV4_LOAD_THREADS` / `TS_DSV4_LOAD_CHUNK_MB` / `TS_DSV4_MOE_MMAP` | V4 and V4.1 | Placement headroom, graph-cache depth, weight-load parallelism, and whether host-resident experts are multiplied in place out of the GGUF mapping | see the cards | no |
| `TS_DSV41_RETAINED_CACHE_MB` | V4.1, native executor | Budget for the native slots retained for finished conversations' next turns (retention is always on, except while a DSpark drafter is loaded); a value that is not a positive number keeps the default | `2048` | no |
| `TS_DSV41_TP_HOST_TOKENS` | V4.1 CUDA routed-MoE TP | Integer `0`–`4096`: batches at or below this token count use the exact pinned-host activation gather; larger batches use private F32 NCCL when available. `0` forces available device gathering. Automatic `16` applies only when NCCL has selected `NCCL_P2P_DISABLE=1`, based on six-A40 paired measurements; other configurations default to `0`. `TS_GGML_TP_F32_NCCL=0` still forces the host fallback for all batches. | automatic | no |
| `TS_DSV41_ENGRAM_DEVICE` | V4.1 | `1` requires GPU-resident Engram tables and fails if they do not fit; `0` forces host mappings, which is also what a bit-exact CPU-oracle comparison needs. Unset is automatic and conservative: GPU-resident unless that would force routed-expert CPU offload | auto (GPU-resident when it fits) | no |
| `TS_DSV41_ENGRAM_WARM` | V4.1, host mappings only | Reads the Engram table pages so a lookup is a RAM read instead of a storage round trip. Unset warms in the BACKGROUND after the model is serving (the default); `1` warms synchronously during load as before; `0` never warms. Worth 200-252 → 452-492 prefill tok/s and 23-26 → 31-33 decode tok/s on eight A40s at Q4_K_M, and irrelevant on the GPU-resident default. Costs the table bytes in host page cache (60 GiB at Q2_K, 103 GiB at Q4_K_M) and the time to read them: the synchronous form took 311.3 s for 103 GiB with the old page walk on the seven-A40 lane; the `pread` warm (`TS_DSV4_WARM_PREAD`) measured 2.24-2.54 GiB/s on that storage, i.e. 41-46 s of reads | background warm | no |
| `TS_DSV4_GRAPH_CACHE_HEADROOM_MB` | V4 and V4.1 | Device memory the graph cache must leave free for the next graph, on top of the largest entry it holds. Least-recently-used entries are freed until it fits. `0` restores the pure entry-count cap, which four concurrent 10.8k-token prefills could run out of memory | `1024` | no |
| `TS_DSV41_ENGRAM_THREADS` | V4.1, host mappings only | Persistent lookup workers, `1`-`32`. One token selects 24 independent rows per table, so serial reads mean serialized page faults | min(16, hardware threads) | no |
| `TS_DSV41_ENGRAM_RANDOM` | V4.1, host mappings on Linux | Random-access advice for the mapped Engram ranges. `0` disables, `1` forces | auto | no |
| `TS_DSV4_LOAD_CONTIGUOUS` | V4 and V4.1 | `0` reverts the weight loader to handing its chunk jobs out from a shared cursor, which makes every reader stride `threads x chunk` through the file instead of reading one contiguous run. Kept only so the contiguous default can be A/B'd: on a MooseFS mount it measured 2.5x slower (363-382 s against 144-155 s to load the Q4_K_M release on eight A40s) | on | no |
| `TS_DSV4_WARM_PREAD` | V4 and V4.1 host mappings; V4.1 TP routed-weight loading | Controls the host-expert prefault (`--n-cpu-moe`), synchronous/background Engram warming, and V4.1 TP's per-layer routed gate/up/down source preparation before GPU upload. Unset or `1` uses bounded parallel `pread` with `TS_DSV4_LOAD_THREADS`, skips ranges `mincore` reports resident, and reports read errors with shard/offset. TP warms only the current layer after opening its source mappings, rather than the whole checkpoint; it does not warm Engram tables through this path. `0` disables TP source preparation and restores the existing host-expert/Engram page-touch walks (256 MiB prefault spans and 8 MiB Engram chunks). The historical seven-A40 8 GiB range test measured 2.24-2.54 GiB/s for `pread` versus 0.62-0.74 GiB/s for page walks; this is not a TP model-load speedup measurement or a cold-storage guarantee | on | no |
| `TS_DSV4_LOAD_DROP_CACHE` | V4 and V4.1 ordinary layer uploads | Whether each uploaded weight chunk's page cache is released once the chunk is on the device. The private V4.1 TP routed-expert uploader does not apply this setting; its uploaded source pages remain in reclaimable file cache. Unset is automatic: drop when the upload bytes plus the host-mapped weights (experts, Engram tables) plus 8 GiB exceed the host allowance (the cgroup limit), keep otherwise and whenever the allowance is unknown, so a reload of a GPU-resident checkpoint stays warm; the load logs the decision with the three numbers. `1` always drops, `0` never drops (the previous default). The seven-A40 lane (263.0 GiB upload + 151.2 GiB mapped against 326.9 GiB) drops. Dropping costs 5.9-7.3 ms per resident 64 MiB chunk on the MooseFS mount and cannot speed the upload itself; its expected gain is on the prefault and Engram warm that follow | auto | no |
| `TS_DSV41_REWIND_CHECKPOINT` | V4.1, native executor | `0` drops the per-slot rewind checkpoint (a shadow copy of the raw sliding-window and compressor-state rings, taken at every prompt boundary). Without it a partial KV reuse can only rewind as far as the live ring reaches — 385 positions on the released checkpoint — so a multi-turn thinking chat re-prefills instead. Costs ~21 MiB of VRAM per sequence slot | on | no |
| `TS_DSV41_SPARSE_FA` | V4.1 | Sparse prefill attention. On the owned F32 CUDA path it is **on by default** for launches of more than 8 queries over at least 8,192 keys: each query attends to its sliding window plus the indexer's selection (at most 640 keys) instead of every key. 512 queries x 33,536 keys measured 34 ms against 1,548 ms tiled on an A40, both within 1.5e-7 of an F32 reference; decode, DSpark verify and shorter prompts keep the dense kernels bit for bit. `0` restores tiled prefill. `1` additionally opts ggml's flash-attention path (non-CUDA GPUs, the CPU backend) into its mask-compacted kernel for one query or at least 16,384 keys, which has documented F16 differences | on for the owned CUDA path; off for ggml flash attention | no |
| `TS_DSV41_COMPACT_RAW_GATHER` | V4.1 | `1` selects compact gathering for the raw sliding window. Opt-in, with the same floating-point caveat | off | no |
| `TS_DSV41_ALLOW_NON_CUDA_GPU` | V4.1 | `1` permits `ggml_vulkan` / `ggml_metal`, where the ordinary graph runs on the GPU and only the architecture-specific ops fall back to the CPU backend, at a host round trip each. Opt-in because the refusal it lifts was closing a *silent* fallback | off | no |
| `TS_DSV41_VISION_FA` / `TS_DSV41_VISION_BF16_GEMM` | V4.1 vision companion | `TS_DSV41_VISION_FA=1` selects flash attention with F16 intermediates in the image encoder (larger real-image feature differences); `TS_DSV41_VISION_BF16_GEMM=0` selects the diagnostic F32-promoted matrix path | dense F32 attention, BF16 GEMM with F32 accumulation | no |
| `TS_DSV41_TRACE_DIR` / `TS_DSV41_VISION_TRACE_DIR` | V4.1 (diagnostic) | Directories for text-graph and vision-encoder tensor dumps. Both retain intermediates and add device transfers — leave unset for benchmarks | unset | no |
| `TS_DSV4_CPU_TRACE_DIR` / `TS_DSV4_CUDA_TRACE_DIR` | V4.1 (diagnostic) | The same per-tensor dumps from the pure-C# and direct-CUDA V4.1 executors, in the same file naming as `TS_DSV41_TRACE_DIR`, so two backends can be diffed tensor by tensor to find the first divergence | unset | no |

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
| `TENSORSHARP_LAYER_SPLIT_DEVICES` | Qwen 3.8 Flash Next shared GGML layer executor | Comma-separated device ordinals, e.g. `0,2`; independent of `TENSORSHARP_TP_DEVICES`; native GLM/DeepSeek use `CUDA_VISIBLE_DEVICES` | `0..N-1` | not registered | no |
| `TENSORSHARP_TP_DEVICES` | local TP on the GGML backends | Comma-separated GPU ordinals the ranks map to (e.g. `0,2`) | `0..tp-1` | not registered | no |
| `TENSORSHARP_TP_NODE_ID` | all autoregressive models; `cuda`, `ggml_cuda`, `ggml_vulkan` backends | This node's 0-based ID for multi-node distributed TP; must be set with `TENSORSHARP_TP_PEERS` | unset (disabled) | not registered | no |
| `TENSORSHARP_TP_PEERS` | all autoregressive models; `cuda`, `ggml_cuda`, `ggml_vulkan` backends | Comma-separated `host:port` list of all nodes in the distributed TP cluster; must be set with `TENSORSHARP_TP_NODE_ID` | unset (disabled) | not registered | no |
| `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS` | distributed TP only | How long each node retries outbound connections to its peers before failing | `120` seconds | not registered | no |
| `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS` | distributed TP only | Per-receive timeout on a peer socket; a stalled peer fails the collective instead of hanging | `300` seconds | not registered | no |
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
| `TENSORSHARP_PREFIX_CACHE_DIR` | `TensorSharp.Server` | Root for the shared-prefix checkpoint files that let a restart skip the shared prompt's prefill; each model gets its own `<model-file-stem>-<hash>` subdirectory with at most two files. `--no-prefix-cache` turns persistence off. TensorAgent keeps its own under the app's cache directory; the CLI keeps checkpoints in memory only | `prefix-cache` beside the server binary | not registered | no |
| `TS_NO_MULTI_AGENT` | `TensorSharp.Server` chat endpoints | Any value except `0` turns off sub-agent delegation, like `--no-multi-agent` | unset (delegation on) | not registered | no |
| `TS_THINKING_BUDGET` | chat turns with thinking on, in `TensorSharp.Server` and the interactive CLI | Tokens a reasoning block may run. Families with a trained closing token (DeepSeek V4.1, Nemotron 3.5 / Omni, Qwen 3.5-3.8, GLM 5.x) then close it and answer inside `max_tokens`; Qwen and GLM first write Qwen's hand-over sentence. Other families stop the turn with finish reason `thinking_budget`. `0` disables | 75% of `max_tokens` when it is at least 512, else none | not registered | no |
| `TS_JEV_MAX_BODY_MB` / `TS_JEV_MAX_CANVAS` / `TS_JEV_MAX_PENDING` | `POST /v1/systemone` (Jev, DiffusionGemma) | Request-body limit in MiB (1-64), answer-canvas width per question chunk in tokens (8-4096, also bounded by the checkpoint), and how many requests may be admitted at once (1-1024) before the endpoint answers HTTP 529. An out-of-range value is an error, not clamped | `8` / `64` / `32` | not registered | no |
| `TS_NEMOTRON_AUDIO_MMPROJ` | Nemotron-H with an `--mmproj` | Audio companion GGUF (NVIDIA `sound_encoder.*` / `sound_projection.*` tensors) to load the Parakeet audio tower from instead of the `--mmproj` file, so a vision mmproj and an audio companion can be used together. Audio stays refused (HTTP 400) unless the file's tensors validate; see `docs/models/nemotron.md` §4.7 | unset (audio tower read from `--mmproj` when it has the tensors) | not registered | no |
| `TS_PDF_MAX_PAGES` | PDF document input (CLI `--pdf`, server `/api/upload`) | Cap on the number of PDF pages read for text extraction and page-image rendering | `0` (all pages) | not registered | no |
| `TS_GGUF_PREFAULT` / `TS_GGUF_PREFAULT_THREADS` / `TS_GGUF_PREFAULT_RESIDENT` | Model load through `GgufReader` | `PREFAULT=0` skips parallel page-cache warming. Threads are capped at the processor count. The selected tensor ranges across all shards share one half-of-process-memory-budget limit; sparse Qwen PLE tables are excluded. Linux checks page residency without copying cached ranges; `RESIDENT=0` forces reads for A/B measurement. Other platforms read selected ranges normally; iOS/tvOS skip warming. The memory budget is not a measurement of currently free host RAM. | on, `min(16, cores)`, residency check on | not registered | no |
| `TS_DUMP_LOGITS` | all models, all backends | Path the FIRST real forward's logits are written to, once, as raw float32. It deliberately SKIPS the warm-up forwards: `WarmUpKernels` drives its own throwaway decode and prefill before the real prompt, so dumping those would compare two executors on a meaningless token rather than on the model. Lets two backends be compared by logit vector instead of by generated text, where greedy decoding turns a near-tie into a visibly different sentence | unset (no dump) | not registered | no |
| `TS_FUSED_QKNORM_ROPE` | Qwen 3.5 / 3.6 text-only prefill on the direct `cuda` backend | Fused QK-Norm + NeoX-RoPE CUDA kernel; `0` falls back to separate norm + RoPE ops (multimodal MRoPE and other backends always use the separate path) | ON | not registered | no |
| `TS_CUDA_QMM_F16GEMM_MIN_ROWS` | direct `cuda` backend | Activation-row threshold above which quantized matmuls dequantize the weight once to F16 and run a tensor-core cuBLAS GEMM (the ggml-style prefill route) instead of the block-tile quant kernels | `32` | not registered | no |
| `TS_CUDA_QMM_F16GEMM_MAX_MB` | direct `cuda` backend | F16 weight-scratch cap in MB; weights above it (e.g. the LM head) keep the quant kernels | `768` | not registered | no |
| `TS_CUDA_Q80_VEC_MIN_OUT` | direct `cuda` backend | Minimum output width for the Q8_0 dp4a matvec (diagnostic gate) | `0` | not registered | no |
| `TS_CUDA_Q80_MMQ_MAX_ROWS` | direct `cuda` backend | Q8_0 matmuls with 32 to this many activation rows run the direct int8 tensor-core GEMM over raw Q8_0 blocks (mma.m16n8k32, ggml MMQ-style); above it the F16 GEMM route wins (MMQ weight sweeps grow as ceil(rows/128)) | `512` | not registered | no |
| `TS_CUDA_PREFILL_GRAPH_MAX` | direct `cuda` backend | Cached prefill + decode CUDA graphs kept (LRU-evicted; each pins its captured working-set pool blocks). Qwen 3.5 / 3.6 text-only prefill and decode capture their per-op layer loop as a graph and replay it (bit-identical results; any capture failure falls back to the plain run) | `4` | not registered | no |
| `TS_CUDA_PREFILL_GRAPH_LOG` | direct `cuda` backend | Log graph capture/replay/abort events (`1`) | OFF | not registered | no |
| `TENSORSHARP_CUDA_POOL_LARGE_MB` | direct `cuda` backend | Budget for the global large-block (≥ 2 MB) device-memory cache; keeps prefill-sized activations pooled instead of re-issuing cuMemAlloc/cuMemFree per layer | `1024` | not registered | no |
| `TS_CUDA_PROFILE` | direct `cuda` backend | Print CPU-fallback op and host↔device sync counters at exit (`1`), with call-site attribution (`2`) | OFF | not registered | no |

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
