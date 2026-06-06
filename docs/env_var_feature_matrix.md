# Environment Variable × Feature Matrix

A curated reference of the high-impact environment variables that change
TensorSharp inference behavior, and the features / models / backends they
affect. This matrix is the source of truth for
[`TensorSharp.TestMatrix`](../TensorSharp.TestMatrix/README.md), which exercises
each (env var × feature) cell in CI and tracks regressions over time.

The matrix is intentionally **curated**, not exhaustive — there are ~90 `TS_*`
debug knobs in the codebase, but most are diagnostic-only (`DUMP_LAYERS`,
`TEST_MATMUL`, `*_PROFILE`, `*_DIAG`, …). The flags below are the ones whose
on/off state materially changes correctness, throughput, or memory usage on a
real workload.

## Legend

- **Affects (model)** — model families the flag impacts (`*` = all)
- **Affects (backend)** — backends the flag impacts (`*` = all)
- **Affects (feature)** — prompt / inference paths the flag impacts
- **Values** — accepted values; the *default* is **bold**
- **Tested in matrix** — `yes` means the test runner sweeps this flag in CI

## 1. Continuous batching / batched forward

| Env var | Affects (model) | Affects (backend) | Affects (feature) | Values | Tested |
|---|---|---|---|---|---|
| `TS_GPTOSS_BATCHED` | GPT OSS | * | Continuous batching, parallel inference (default ON, set `0` to opt out) | `0`, **`1`** | yes |
| `TS_QWEN35_BATCHED` | Qwen 3.5 / 3.6 family | * | Continuous batching, parallel inference (default ON, set `0` to opt out) | `0`, **`1`** | yes |
| `TS_QWEN35_BATCHED_GDN_NATIVE` | Qwen 3.5 / 3.6 family | * | GatedDeltaNet batched kernel | **`0`**, `1` | yes |
| `TS_NEMOTRON_BATCHED` | Nemotron-H | * | Continuous batching, parallel inference (default ON, set `0` to opt out) | `0`, **`1`** | yes |
| `TS_GEMMA4_BATCHED` | Gemma 4 | * | Continuous batching (default ON, set `0` to opt out) | `0`, **`1`** | yes |
| `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE` | Nemotron-H | * | SSM Mamba2 batched native kernel | **`0`**, `1` | yes |
| `TS_NEMOTRON_MOE_PREFILL_BATCHED` | Nemotron-H MoE | * | Batched MoE prefill | **`0`**, `1` | yes |
| `TS_BATCHED_N1_FAST_PATH` | * | * | N=1 fast path through the batched scheduler | **`0`**, `1` | yes |
| `TS_SCHED_DISABLE_BATCHED` | * | * | Falls through to per-seq KV swap when set | **`0`**, `1` | yes |

## 2. KV cache: precision, paging, codec

| Env var | Affects (model) | Affects (backend) | Affects (feature) | Values | Tested |
|---|---|---|---|---|---|
| `KV_CACHE_DTYPE` | * | * | All prefill / decode / multimodal | **`f32`**, `f16`, `q8_0` | yes |
| `TS_KV_PAGED_QUANT_BITS` | * | * | Paged KV cache codec (TurboQuant) | **`0`** (off), `4`, `8` | yes |
| `TS_KV_CACHE_SSD_DIR` | * | * | SSD-backed paged KV tier for very large working sets | **`<unset>`**, path | no (env only) |
| `MAX_CONTEXT` | * | * | Hard cap on context length | **`<unset>`** (model default), integer | yes |

## 3. Prefill / decode tuning

| Env var | Affects (model) | Affects (backend) | Affects (feature) | Values | Tested |
|---|---|---|---|---|---|
| `TS_PREFILL_CHUNK` | GPT OSS, Qwen 3.5 / 3.6 | * | Chunked prefill block size | **`<unset>`**, integer | yes |
| `GDN_DISABLE_CHUNKED_PREFILL` | Qwen 3.5 / 3.6 (`qwen3next`) | * | GDN chunked prefill | **`0`**, `1` | yes |
| `GDN_CHUNK_PREFILL_MIN_SEQ_LEN` | Qwen 3.5 / 3.6 (`qwen3next`) | * | GDN chunked prefill threshold | **`<unset>`**, integer | no (rare) |
| `TS_PAGED_ATTN_KERNEL` | * | GGML | Paged-attention kernel selection | **`<unset>`** (auto), kernel id | no (advanced) |
| `TS_GGML_ASYNC_COMPUTE` | * | GGML | Async compute submission | **`0`**, `1` | yes |
| `TS_ENCODER_YIELD` | * | * | Yield during vision encoder | `0`, **`1`** | no |

## 4. Multimodal

| Env var | Affects (model) | Affects (backend) | Affects (feature) | Values | Tested |
|---|---|---|---|---|---|
| `VIDEO_SAMPLE_FPS` | Gemma 4 | * | Video prompts | **`1`**, positive number | yes |
| `VIDEO_MAX_FRAMES` | Gemma 4 | * | Video prompts | **`<unset>`** (no cap), integer | yes |
| `TS_NEMOTRON_IMAGE_MAX_TILES` | Nemotron-H Omni | * | Image prompts | **`<unset>`**, integer | yes |
| `TS_NEMOTRON_MULTIMODAL_WARMUP` | Nemotron-H Omni | * | First-call vision warmup | `0`, **`1`** | no |

## 5. MLX-specific tuning

| Env var | Affects (model) | Affects (backend) | Affects (feature) | Values | Tested |
|---|---|---|---|---|---|
| `TS_MLX_BATCHED_MOE_DECODE` | Qwen 3.5 / 3.6 MoE | MLX | MoE decode kernel (single batched dispatch per gate/up/down vs. K per-expert dispatches) | `0`, **`1`** | yes |
| `TS_MLX_MOE_FUSED_GATE_UP_SILU` | Qwen 3.5 / 3.6 MoE | MLX | Fuses MoE gate matmul + up matmul + SiLUMul into one Metal kernel | `0`, **`1`** | yes |
| `TS_MLX_DEVICE_ROUTER` | Qwen 3.5 / 3.6 MoE | MLX | Router top-K + softmax on-device (skips ~60 host syncs/token on Qwen3.6-35B-A3B) | **`0`**, `1` | yes |
| `TS_MLX_PIPELINED_DECODE` | * | MLX | Pipelined greedy decode (device-side argmax) | `0`, **`1`** | yes |
| `TS_MLX_KERNEL_WARMUP` | * | MLX | Force Metal kernel JIT warmup at load | **`0`**, `1` | no (load-time only) |
| `TS_MLX_MEMORY_LIMIT_MB` | * | MLX | MLX allocator hard cap (MB) | **`<unset>`** (host-derived), integer | no (HW-dependent) |
| `TS_MLX_CACHE_LIMIT_MB` | * | MLX | MLX unused-buffer cache cap (MB) | **`<unset>`** (host-derived), integer | no (HW-dependent) |
| `TS_MLX_WIRED_LIMIT_MB` | * | MLX | Wired-buffer residency cap (MB), set on the MLX side | **`<unset>`** (host-derived), integer | no (HW-dependent) |
| `TS_MLX_EXPERT_OFFLOAD_MB` | MoE families | MLX | LRU expert offload pool size (MB) | **`<unset>`**, integer | no (HW-dependent) |
| `TS_MLX_DEVICE_KV_COPY` | * | MLX | On-device KV scatter | `0`, **`1`** | yes |
| `TS_MLX_FUSED_KV_WRITE` | * | MLX | Single multi-dim `slice_update` write per per-token KV block | `0`, **`1`** | yes |
| `TS_MLX_MLOCK_GGUF` | * | MLX | `mlock(2)` the GGUF mmap so weights stay resident between forward passes | `0`, **`1`** | no (HW-dependent — needs sufficient `memlock` rlimit) |
| `TS_MLX_EVAL_EVERY_N_LAYERS` / `TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS` | Gemma 4, Qwen 3.5 / 3.6 | MLX | Periodic `mlx_async_eval` cadence to overlap GPU work with host queueing | **`4`** (default), integer (`0` = disabled) | no (perf-tuning) |
| `TS_MLX_QWEN35_GDN_PACKED_KERNELS` | Qwen 3.5 / 3.6 | MLX | Packed GDN kernels | **`0`**, `1` | yes |
| `TS_MLX_LOG_MEMORY_POLICY` | * | MLX | One-line memory-policy banners at load | `0`, **`1`** | no (informational) |

## 6. Feature × prompt-type coverage

This is the "what the matrix exercises" half. For every combination below the
test runner records `model_load_ms`, `prefill_tokens / ms / tps`,
`decode_tokens / ms / tps`, and `total_wall_ms` plus a short tail of the
output text.

| Feature / prompt type | Driver | Models with this capability |
|---|---|---|
| Synthetic prefill (`pp512`, `pp2048`) | `--benchmark --bench-prefill N --bench-decode 0` | all |
| Synthetic decode (`tg128`) | `--benchmark --bench-prefill 32 --bench-decode N` | all |
| Short text (single turn) | `--input <short_text.txt> --max-tokens 64` | all |
| Long text (single turn, ~1k tokens) | `--input <long_text.txt> --max-tokens 64` | all |
| Uploaded text (large file, truncated) | `--input <upload_text.txt> --max-tokens 64` | all |
| Multi-turn chat (KV reuse) | `--multi-turn-jsonl <multi_turn.jsonl>` | all |
| Function / tool calling | `--tools <tools.json> --input ...` | Gemma 4, Qwen 3, Qwen 3.5 / 3.6, Nemotron-H |
| Thinking / reasoning mode | `--think --input ...` | Gemma 4, Qwen 3, Qwen 3.5 / 3.6, GPT OSS, Nemotron-H |
| Image | `--image <apple.png> --mmproj ...` | Gemma 3, Gemma 4, Qwen 3.5 / 3.6, Mistral 3, Nemotron Omni |
| Audio | `--audio <speech.mp3> --mmproj ...` | Gemma 4 |
| Video | `--video <clip.mp4> --mmproj ...` | Gemma 4 |

## 7. Filters applied by the matrix runner

The combinatorial product backends × models × prompt-types × env-var sweeps is
prohibitively large. The runner applies these filters before scheduling:

1. **Backend availability** — `cuda` / `ggml_cuda` only on hosts with NVIDIA
   drivers; `mlx` / `ggml_metal` only on Apple Silicon. (See
   [`BackendCatalog.GetSupportedBackends`](../TensorSharp.Server/BackendCatalog.cs).)
2. **Model capability** — image/audio/video tasks are skipped on text-only
   models. Audio/video are skipped on every model except Gemma 4.
3. **Env-var applicability** — `TS_QWEN35_BATCHED` is only swept on Qwen 3.5 /
   3.6 GGUFs; `VIDEO_MAX_FRAMES` is only swept on video tasks; etc.
4. **Env-var × prompt-type relevance** — only env vars that actually change
   the code path for a given prompt type are swept on that prompt type.
   E.g. `TS_PREFILL_CHUNK` only varies on long-context / uploaded-text /
   multimodal prompts where the prefill path actually chunks.

## 8. Adding a new env var to the matrix

1. Add a row to the appropriate table above with the `Tested` column set to
   `yes`.
2. Register it in
   [`TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs`](../TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs)
   with the model / backend / feature predicates that gate when it applies.
3. The next CI run picks it up — no other plumbing needed.

## 9. Adding a new prompt type / feature

1. Add a row to §6 above.
2. Add a `FeatureSpec` to
   [`TensorSharp.TestMatrix/Matrix/FeatureCatalog.cs`](../TensorSharp.TestMatrix/Matrix/FeatureCatalog.cs)
   describing the CLI flags it emits and the models it applies to.
3. Drop any required sample prompts under
   `TensorSharp.TestMatrix/Inputs/prompts/` and any sample media under
   `TensorSharp.TestMatrix/Inputs/media/`.
