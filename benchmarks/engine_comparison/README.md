# Engine comparison benchmark

Automated, repeatable benchmark that compares **TensorSharp**, **llama.cpp** and
**vLLM** on the *same* GGUF files, the *same* host, through one uniform OpenAI
`/v1/chat/completions` surface — across **text, image, audio, video,
single-turn, multi-turn, function-call, structured-output and long-prompt
prefill** scenarios, on any **compute backend** declared in the config's
`backends` registry (`ggml_cuda`, `ggml_vulkan`, `ggml_metal`, `ggml_cpu`,
`cpu`, …) — pick with `--backends`.

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
| `assets/` | long-context prompt (`long_text.txt`), prefill corpus (`prefill_corpus.txt`), `tools/weather.json` |
| `benchmark_config_prefill.json` | **prefill-only** variant — the same long-prompt sweep (2k/4k/8k/16k/32k/64k/128k tokens) but with the multimodal / diffusion scenarios and models stripped out, for a focused prefill run; select with `--config` |

## Configuration

Every setting lives in **`benchmark_config.json`** — nothing is hardcoded in the
Python. It holds the host paths (`paths`), the model / scenario / engine /
backend registries (`models`, `scenarios`, `engines`, `backends` — see
[Choosing compute backends](#choosing-compute-backends---backends) for the
backend entry format), the llama-server launch options (`llama`),
per-size-class readiness timeouts (`ready_timeout_s`), and the run defaults
(`defaults`: which engines / models / scenarios / backends to run, MTP modes,
concurrency levels, max-tokens, warmup count, server max-tokens headroom).

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
  Non-CUDA backend columns use per-backend builds declared in the `backends` registry
  (e.g. a Vulkan build at `build-vulkan/.../llama-server.exe`, overridable via
  `BENCH_LLAMA_SERVER_VULKAN`); a missing build just records that column's llama.cpp cells as skipped.
- **vLLM** (optional): start an OpenAI server yourself and point the harness at it; otherwise vLLM cells record `skipped (engine unavailable)`.
- Models under `C:/Works/models` and media at `C:/Works/{test.jpg,obama_first_45_secs.mp3,concert.mp4}`.

All of these paths default to the values in `benchmark_config.json` (`paths`
section) and are overridable per-host either by editing that file or via the
environment variables listed under [Configuration](#configuration) above.

## Running

```bash
cd benchmarks/engine_comparison

# Smoke test: one cheap cell, end to end
python run_matrix.py --engines tensorsharp --backends ggml_cuda \
    --models gemma4-12b --scenarios text_short,multi_turn

# Same model across several compute backends (one report column per backend)
python run_matrix.py --engines tensorsharp,llamacpp \
    --backends ggml_cuda,ggml_vulkan,ggml_cpu,cpu \
    --models gemma4-12b --scenarios text_short

# Full matrix (engines auto-skip when a binary / endpoint is missing,
# CPU auto-skips the large MoE models, diffusion auto-restricts to text)
python run_matrix.py --engines tensorsharp,llamacpp,vllm --backends ggml_cuda,cpu \
    --models gemma4-12b,gemma4-26b-a4b,qwen36-35b-a3b,diffusiongemma

# MTP / NextN speculative decoding, on vs off (TensorSharp), single stream
python run_matrix.py --engines tensorsharp --backends ggml_cuda \
    --models qwen36-35b-a3b,gemma4-12b --scenarios text_short --mtp off,on

# Parallel-request scaling — aggregate decode throughput under load
python run_matrix.py --engines tensorsharp,llamacpp --backends ggml_cuda \
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

### Choosing compute backends (`--backends`)

The backend axis is a **registry** in the config's `backends` section: one
entry per concrete compute backend, each becoming its own column in the report.
The default registry declares:

| id | kind | TensorSharp | llama.cpp | vLLM |
|---|---|---|---|---|
| `ggml_cuda` (alias `gpu`) | gpu | `--backend ggml_cuda` | CUDA build, `-ngl 999` | compared here |
| `ggml_vulkan` | gpu | `--backend ggml_vulkan` | Vulkan build (`BENCH_LLAMA_SERVER_VULKAN`), `-ngl 999` | — |
| `ggml_metal` | gpu | `--backend ggml_metal` | Metal build (`BENCH_LLAMA_SERVER_METAL`), `-ngl 999` | — |
| `cuda` | gpu | `--backend cuda` (direct cuBLAS) | — | — |
| `mlx` | gpu | `--backend mlx` (macOS) | — | — |
| `ggml_cpu` | cpu | `--backend ggml_cpu` | `-ngl 0` | — |
| `cpu` | cpu | `--backend cpu` (pure C#) | `-ngl 0` | — |

Select any subset with `--backends ggml_cuda,ggml_vulkan,...` (or
`defaults.backends` in the config); the legacy alias `gpu` still resolves to
`ggml_cuda`, and unknown ids fail fast with the list of available ones. An
engine with no launch mapping for a backend (e.g. llama.cpp on `mlx`) records
its cells as `skipped`, and `cpu`-kind backends auto-skip `large` models.

The `ggml_vulkan` column needs a Vulkan build of llama-server (the CUDA build
cannot run Vulkan). It can be built without installing the LunarG SDK by
reusing TensorSharp's portable Vulkan toolchain
(`ExternalProjects/vulkan-toolchain`, provisioned by
`eng/fetch-vulkan-toolchain.ps1`):

```powershell
$TC = "C:/Works/TensorSharp/ExternalProjects/vulkan-toolchain"
cmake -S C:/Works/llama.cpp -B C:/Works/llama.cpp/build-vulkan -G "Visual Studio 17 2022" -A x64 `
    -DGGML_VULKAN=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=ON `
    -DVulkan_INCLUDE_DIR="$TC/Vulkan-Headers/include" `
    -DVulkan_LIBRARY="$TC/loader/vulkan-1.lib" `
    -DVulkan_GLSLC_EXECUTABLE="$TC/shaderc/bin/glslc.exe" `
    -DSPIRV-Headers_DIR="$TC/spirv-headers-install/share/cmake/SPIRV-Headers" `
    -DCMAKE_CXX_FLAGS="-DWIN32 -D_WINDOWS -W3 -GR -EHsc -I$TC/spirv-headers-install/include"
cmake --build C:/Works/llama.cpp/build-vulkan --config Release --target llama-server -j 16
```

(The `CMAKE_CXX_FLAGS` include is needed because llama.cpp's
`find_package(SPIRV-Headers)` only checks that the package exists — it assumes
`spirv/unified1/spirv.hpp` is reachable through the Vulkan SDK include dir,
which the portable toolchain keeps in a separate install tree. The dash-style
MSVC flags are deliberate: they also work when the command is pasted into Git
Bash, where `/D...`-style flags get mangled into paths by MSYS conversion.)

Each registry entry says how every engine launches on that backend:
`tensorsharp: {backend, extra_args, env}` (e.g. `"extra_args": ["--gpu-device", "1"]`
or `"env": {"TS_GGML_VULKAN_DEVICE": "1"}` to pick the Vulkan GPU),
`llamacpp: {ngl, server_exe, extra_args, env}` — `server_exe` points at a
per-backend llama-server build (Vulkan/Metal builds live in separate llama.cpp
build trees; the `{"path", "env"}` form makes it host-overridable) and falls
back to `paths.llama_server_exe` — and `vllm: true` marks the single column the
external vLLM endpoint's numbers are comparable on. Add a new backend by adding
an entry; nothing in the Python needs to change.

Two caveats: **TensorSharp.Server silently falls back** to the first available
backend when the requested one isn't supported by the build/host (check
`results/logs/*.log` — the startup banner names the backend actually used — if
numbers look implausible), and llama.cpp's `cpu` and `ggml_cpu` cells are the
same engine configuration (`-ngl 0`), since llama.cpp has no pure-C# analogue.

Older configs using the legacy `"backends": ["gpu", "cpu"]` list + `maps`
form still load unchanged, and result files from old runs (backend ids `gpu` /
`cpu` in their names) still render in reports alongside new ids.

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

### Prefill (prompt-processing) benchmark (`prefill_2k` / `4k` / `8k` / `16k` / `32k` / `64k` / `128k`)

The plain text scenarios' longest prompt (`text_long`) is only ~1.2k tokens, where
time-to-first-token is dominated by **fixed per-request overhead** (HTTP,
scheduling, cold-graph launch, first-token sampling) rather than prefill compute —
so `prefill_tps` there understates and noisily estimates true prompt-processing
throughput. The `prefill_<N>` scenarios drive the prompt to controlled lengths long
enough for the per-token prefill cost to separate cleanly from that fixed overhead.

These scenarios are part of the **main `benchmark_config.json`** matrix (which runs
the 2k/4k/8k sweep by default). For a *focused* prefill run — the full
2k → 128k sweep, with the multimodal / diffusion scenarios and models stripped out
and results written to a separate `results_prefill/` — use the dedicated
**`benchmark_config_prefill.json`** (its `defaults.scenarios` runs every length
through `prefill_128k`).

The long-context lengths (`prefill_32k` / `64k` / `128k`) drive very large prompts:
`run_matrix.py` auto-raises llama.cpp's `-c` context to fit (≈170k tokens for the
128k case), and the engine needs enough KV VRAM/RAM to hold it — trim the selection
on smaller hosts.

```bash
# Just the prefill sweep, default matrix (selecting the scenarios from the main config)
python run_matrix.py --scenarios prefill_2k,prefill_4k,prefill_8k,prefill_16k,prefill_32k,prefill_64k,prefill_128k

# Focused prefill-only run (TensorSharp vs llama.cpp, GPU, separate results dir)
python run_matrix.py --config benchmark_config_prefill.json

# One length, one model
python run_matrix.py --config benchmark_config_prefill.json \
    --models gemma4-12b --scenarios prefill_8k

# Report it (writes into results_prefill/ per the config's results_dir)
python report.py --config benchmark_config_prefill.json
```

How it works:

- Scenarios `prefill_2k` / `prefill_4k` / `prefill_8k` / `prefill_16k` /
  `prefill_32k` / `prefill_64k` / `prefill_128k` slice
  `assets/prefill_corpus.txt` to a target **token** budget (the id names the
  target; `scenarios._prefill` converts it to a character budget at
  ~4.6 chars/token, tiling the corpus when a target exceeds it). The label is
  nominal — `prefill_tps = prompt_tokens / ttft` always uses each engine's own
  reported `prompt_tokens`, so tokenizer differences across engines are handled
  exactly.
- Each length gets a **unique position-0 header** (`[prefill-benchmark target=N …]`)
  so a longer prompt cannot hit the server's prompt/prefix cache off a shorter one
  run earlier on the same server (which would report a near-zero TTFT and a wildly
  inflated `prefill_tps`).
- `max_tokens` is tiny (8) — only the prefill phase / TTFT matters here. The main
  config sets `llama.context_size` to 24576 so the 16k prompt fits with headroom,
  and `run_matrix.py` additionally **auto-raises** llama.cpp's context at run time
  to fit whatever prefill lengths are selected (`max_prefill * 1.3 + 128`), so you
  never have to hand-tune it for the standard sweep.

Add lengths by naming them: `--scenarios prefill_1k,prefill_32k` works without a
config edit (`prefill_<N>` / `prefill_<N>k` is parsed generically); the driver
auto-raises llama.cpp's context to fit, so no `llama.context_size` edit is needed.

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
- **Stuck/leftover servers**: a crashed or interrupted run can leave a server
  process squatting its port — in the worst case unkillable (a thread stuck in
  a GPU-driver call survives `taskkill /F` until reboot) while the kernel still
  accepts TCP connects into its dead listen backlog, which looks like an
  endless "server not ready" wait. The harness defends itself: llama-server is
  auto-launched on the next free port when its configured port is taken, the
  TensorSharp group fails fast with the squatter's PID (its 0.0.0.0:5000 listen
  address is hard-coded), and `wait_ready` aborts with a diagnosis when the
  port's owner is not the process it launched.
- `docs/engine_comparison_report.md` and `results/results.csv` from `report.py`.

## Scenario / engine coverage notes

- **vLLM** is connect-only and is not launched by the harness; it has
  historically not loaded these custom architectures, so its cells typically
  resolve to `skipped`. Point `BENCH_VLLM_URL` at a working server to include it.
- **DiffusionGemma** is a text-diffusion model — only TensorSharp runs it, and
  only on text / multi-turn scenarios.
- **Video** is sampled into frames and sent as an image sequence; only
  TensorSharp consumes it (llama.cpp has no video path).
- **CPU-kind backends** (`ggml_cpu`, `cpu`) are restricted to small/medium
  models; the 35B MoE is GPU-only.
- **MTP** (`--mtp on`) only applies to TensorSharp on models that ship a draft
  head (Qwen 3.6 embedded NextN; Gemma 4 with its paired `--mtp-draft-model`);
  every other engine/model `on` cell is recorded as skipped.
