# Engine comparison benchmark

Automated, repeatable benchmark that compares **TensorSharp**, **llama.cpp** and
**vLLM** on the *same* GGUF files, the *same* host, through one uniform OpenAI
`/v1/chat/completions` surface — across **text, image, audio, video,
single-turn, multi-turn, function-call and structured-output** scenarios on
**GPU and CPU** backends.

Model families under test: **Gemma 4** (`gemma4-e4b` dense multimodal Q8_0 from
`models/`, plus `gemma4-12b` dense + `gemma4-26b-a4b` MoE, both QAT UD-Q4_K_XL
from `models/gemma_mtp/qat/`), **Qwen 3.6**, **DiffusionGemma**.

## Why an OpenAI-HTTP harness

All three engines expose an OpenAI-compatible chat endpoint with streaming and a
final `usage` block, so one client driver produces apples-to-apples numbers and
naturally covers every scenario (image via `image_url`, tools via
`tools`→`tool_calls`, multi-turn via message history). Metrics are derived from
the *streamed* response, independent of any engine's internal timer:

| Metric | Definition |
|---|---|
| `ttft_ms` | time to first streamed token (prefill latency proxy) |
| `prefill_tps` | `prompt_tokens / ttft` |
| `decode_tps` | `(completion_tokens - 1) / (t_last_token - t_first_token)` |

DiffusionGemma denoises whole blocks (no token stream); it is run non-streaming
and its `decode_tps` is wall-clock tokens/second.

`report.py` also derives a **performance ratio** of TensorSharp against each
reference engine on the *same* backend (so the comparison stays apples-to-apples):
a headline **per-model geomean** table near the top, and a per-scenario ratio
table (decode / prefill / TTFT) in each model's section. A value **> 1.0× means
TensorSharp is faster** (decode / prefill throughput) or lower-latency (TTFT). A
ratio is `—` when either side has no usable (`ok`) cell, and a reference column
is dropped entirely when that engine produced nothing comparable for the model
(e.g. an unreachable vLLM endpoint).

## Files

| File | Role |
|---|---|
| `benchmark_config.json` | **all settings** — host paths, model / scenario / engine / backend registries, run defaults. Edit this, not the code. |
| `config.py` | loads `benchmark_config.json`, resolves `${var}` paths + env overrides, exposes the registries + applicability gating |
| `engines.py` | OpenAI streaming client + server lifecycle managers (TensorSharp.Server, llama-server, vLLM connector) |
| `scenarios.py` | per-scenario, engine-aware request builders |
| `run_matrix.py` | orchestrator — launches one server per `(engine, backend, model)`, runs scenarios, writes per-cell JSON |
| `report.py` | aggregates `results/*.json` → `docs/engine_comparison_report.md` + `results/results.csv` |
| `assets/` | long-context prompt, `tools/weather.json` |

## Configuration

Every setting lives in **`benchmark_config.json`** — nothing is hardcoded in the
Python. It holds the host paths (`paths`), the model / scenario / engine
registries (`models`, `scenarios`, `engines`), the abstract→engine backend maps
(`maps`), the llama-server launch options (`llama`), per-size-class readiness
timeouts (`ready_timeout_s`), and the run defaults (`defaults`: which engines /
models / scenarios / backends to run, MTP modes, concurrency levels, max-tokens,
warmup count, server max-tokens headroom).

Values resolve with this precedence (highest first):

1. **Command-line flags** to `run_matrix.py` / `report.py` (e.g. `--models`, `--scenarios`, `--max-tokens`).
2. **Environment variables** — host paths only, for retargeting without editing the file (`BENCH_MODEL_ROOT`, `BENCH_TS_SERVER_DLL`, `BENCH_LLAMA_SERVER`, `BENCH_VLLM_URL`, `BENCH_IMAGE`, `BENCH_AUDIO`, `BENCH_VIDEO`, `BENCH_RESULTS`, `DIFFUSION_STEPS`, ...).
3. **`benchmark_config.json`** (or the file named by `--config PATH` / `BENCH_CONFIG`).
4. Built-in fallbacks in `config.py`.

Path strings in the config may use the placeholders `${repo_root}`, `${here}`,
`${model_root}`, `${gemma4_qat_dir}`. A path may also be written as
`{"path": "...", "env": "BENCH_X"}` so the named environment variable overrides
it. Point the harness at an alternate settings file with `--config other.json`
(or `BENCH_CONFIG=other.json`) — useful for keeping per-host configs side by side.

## Prerequisites

- **Python 3.10+** with `requests`, `opencv-python` (video frame sampling). Both already present on the dev box.
- **TensorSharp.Server** built: `TensorSharp.Server/bin/TensorSharp.Server.dll` (run with `dotnet`). Build with `dotnet build TensorSharp.Server -c Release` if missing/stale.
- **llama.cpp** server binary at `C:/Works/llama.cpp/build-cuda/bin/Release/llama-server.exe` (CUDA build).
- **vLLM** (optional): start an OpenAI server yourself and point the harness at it; otherwise vLLM cells record `skipped (engine unavailable)`.
- Models under `C:/Works/models` and media at `C:/Works/{test.jpg,obama_first_45_secs.mp3,concert.mp4}`.

All of these paths default to the values in `benchmark_config.json` (`paths`
section) and are overridable per-host either by editing that file or via the
environment variables listed under [Configuration](#configuration) above.

## Running

```bash
cd benchmarks/engine_comparison

# Smoke test: one cheap cell, end to end
python run_matrix.py --engines tensorsharp --backends gpu \
    --models gemma4-12b --scenarios text_short,multi_turn

# Full matrix (engines auto-skip when a binary / endpoint is missing,
# CPU auto-skips the large MoE models, diffusion auto-restricts to text)
python run_matrix.py --engines tensorsharp,llamacpp,vllm --backends gpu,cpu \
    --models gemma4-12b,gemma4-26b-a4b,qwen36-35b-a3b,diffusiongemma

# MTP / NextN speculative decoding, on vs off (TensorSharp), single stream
python run_matrix.py --engines tensorsharp --backends gpu \
    --models qwen36-35b-a3b,gemma4-12b --scenarios text_short --mtp off,on

# Parallel-request scaling — aggregate decode throughput under load
python run_matrix.py --engines tensorsharp,llamacpp --backends gpu \
    --models gemma4-12b --scenarios text_short --concurrency 1,4,8

# Use an alternate settings file (e.g. a second host)
python run_matrix.py --config configs/host-b.json --engines tensorsharp

# Generate the markdown + CSV report
python report.py
```

With no flags, `run_matrix.py` runs the full matrix defined by the `defaults`
section of `benchmark_config.json`. Any flag overrides the corresponding config
default for that run only (the file is never modified).

Useful flags: `--config <file>` (pick the settings file), `--engines`,
`--backends`, `--models`, `--scenarios`, `--mtp`, `--concurrency`,
`--max-tokens N`, `--warmup N` (0 disables), `--skip-existing` (reuse prior `ok`
cells), `--results <dir>`. `report.py` accepts `--config` and `--results`.

### MTP / NextN speculative decoding (`--mtp off | on | off,on`)

Benchmarks **with and without** TensorSharp's multi-token-prediction draft head.
Each mode relaunches the server (it is a load-time flag): `on` adds `--mtp-spec`,
and for Gemma 4 also `--mtp-draft-model <draft.gguf>` (Qwen 3.6 embeds its NextN
block in the trunk, so no extra file is needed). MTP is a TensorSharp feature —
`on` cells for llama.cpp / vLLM and for the diffusion model are recorded as
skipped. Gemma 4 drafts are target-paired (an E4B `gemma4-assistant` draft for
`gemma4-e4b`, a 12B draft for `gemma4-12b`, a 26B-A4B draft for
`gemma4-26b-a4b`; a mismatched draft fails fast at startup). The 12B / 26B-A4B
drafts default to `C:/Works/models/gemma_mtp/qat/gemma-4-{12B,26B-A4B}-it-Q4_0-MTP.gguf`
(override with `BENCH_GEMMA4_12B_MTP_DRAFT` / `BENCH_GEMMA4_26B_MTP_DRAFT`, or
relocate the whole set with `BENCH_GEMMA4_QAT_DIR`); the E4B draft defaults to
`C:/Works/models/gemma-4-E4B-it-assistant.Q8_0.gguf` (override with
`BENCH_GEMMA4_E4B_MTP_DRAFT`, trunk/mmproj via `BENCH_GEMMA4_E4B_GGUF` /
`BENCH_GEMMA4_E4B_MMPROJ`). `report.py` adds an
**MTP on-vs-off** table with the
per-cell speedup (a value `< 1.0×` means speculation cost more than it saved —
expected where the fused full-model decode path is already fastest).

### Parallel requests (`--concurrency 1,4,8`)

Fires N identical requests at the *same* server at once (the server's continuous
batching serves them concurrently) and records, per cell:

| Metric | Definition |
|---|---|
| `decode_tps` | mean **per-request** decode tok/s |
| `aggregate_decode_tps` | **system-wide** decode tok/s — total generated tokens / the wall window during which any sequence was decoding |

`report.py` adds a **parallel-request scaling** table (per-request vs aggregate
at each concurrency). The two axes compose: `--mtp on --concurrency 4` is valid,
though MTP only engages for solo sequences so it has little effect under load.

Result files keep their historical names for the baseline (`mtp` off,
`concurrency` 1); non-default cells add a `__mtp` and/or `__c<N>` suffix.

## Output

- `results/{engine}__{backend}__{model}__{scenario}[__mtp][__c<N>].json` — one
  record per cell (`status` ∈ `ok | fail | skipped`, plus token counts, `mtp`,
  `concurrency`, `aggregate_decode_tps`, `requests_ok`, and throughput). The
  baseline (MTP off, single request) keeps the suffix-free name.
- `results/logs/{engine}__{backend}__{model}.log` — captured server stdout/stderr
  (the first place to look when a group reports `fail`).
- `docs/engine_comparison_report.md` and `results/results.csv` from `report.py`.

## Scenario / engine coverage notes

- **vLLM** is connect-only and is not launched by the harness; it has
  historically not loaded these custom architectures, so its cells typically
  resolve to `skipped`. Point `BENCH_VLLM_URL` at a working server to include it.
- **DiffusionGemma** is a text-diffusion model — only TensorSharp runs it, and
  only on text / multi-turn scenarios.
- **Video** is sampled into frames and sent as an image sequence; only
  TensorSharp consumes it (llama.cpp has no video path).
- **CPU** runs are restricted to small/medium models; the 35B MoE is GPU-only.
- **MTP** (`--mtp on`) only applies to TensorSharp on models that ship a draft
  head (Qwen 3.6 embedded NextN; Gemma 4 with its paired `--mtp-draft-model`);
  every other engine/model `on` cell is recorded as skipped.
