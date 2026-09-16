# Engine comparison benchmark

Automated, repeatable benchmark that compares **TensorSharp**, **llama.cpp** and
**vLLM** on the *same* GGUF files, the *same* host, through one uniform OpenAI
`/v1/chat/completions` surface — across **text, image, audio, video,
single-turn, multi-turn, function-call, structured-output and long-prompt
prefill** scenarios, on any **compute backend** declared in the config's
`backends` registry (`ggml_cuda`, `ggml_vulkan`, `ggml_metal`, `ggml_cpu`,
`cpu`, …) — pick with `--backends`.

It also benchmarks the **stable-diffusion image-editing engine**
(Qwen-Image-Edit) — TensorSharp's `/api/image-edit` pipeline vs the
**stable-diffusion.cpp** CLI on the same weights, image, prompt, resolution,
steps and seed; see [Image editing](#image-editing--stable-diffusion-image_edit).

Model families under test: **Gemma 4** (`gemma4-e4b` dense multimodal Q8_0 from
`models/`, plus `gemma4-12b` dense + `gemma4-26b-a4b` MoE, both QAT UD-Q4_K_XL
from `models/gemma_mtp/qat/`), **Qwen 3.6**, **DiffusionGemma**,
**Qwen-Image-Edit 2511** (Q2_K DiT + Lightning 4-step LoRA).

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

Beyond speed, `report.py` also compares **output quality**: both engines decode
the same GGUF greedily (`temperature=0`), so their outputs should agree closely.
The report's *Output quality* section scores each overlapping
TensorSharp-vs-llama.cpp cell with a whitespace-normalized text-similarity
ratio (1.00 = identical), checks the structural scenarios (valid JSON object in
`json_mode`, tool call emitted in `function_call`), and appends side-by-side
output excerpts, lowest agreement first. The full generated text is captured
per cell (`output_text`, capped at 8k chars) so the comparison works offline
from the result JSONs alone.

## Files

| File | Role |
|---|---|
| `benchmark_config.json` | **all settings** — host paths, model / scenario / engine / backend registries, run defaults. Edit this, not the code. |
| `config.py` | loads `benchmark_config.json`, resolves `${var}` paths + env overrides, exposes the registries + applicability gating |
| `engines.py` | OpenAI streaming client + server lifecycle managers (TensorSharp.Server, llama-server, vLLM connector) |
| `scenarios.py` | per-scenario, engine-aware request builders |
| `run_matrix.py` | orchestrator — launches one server per `(engine, backend, model, mtp, tp, cpu_moe)`, runs scenarios, writes per-cell JSON |
| `report.py` | aggregates `results/*.json` → `docs/engine_comparison_report.md` + `results/results.csv` |
| `downloads.py` | resumable fetcher — makes a config's model files exist locally (used automatically by `run_matrix.py`) |
| `assets/` | long-context prompt (`long_text.txt`), prefill corpus (`prefill_corpus.txt`), `tools/weather.json` |
| `benchmark_config_prefill.json` | **prefill-only** variant — the same long-prompt sweep (2k/4k/8k/16k/32k/64k/128k tokens) but with the multimodal / diffusion scenarios and models stripped out, for a focused prefill run; select with `--config` |
| `benchmark_config_multigpu.json` | **multi-GPU** variant — a 4-GPU Linux box (validated on 4×A40), tensor-parallel degrees 1/2/4, including a model that only fits across 4 GPUs (`min_tp`); select with `--config` |
| `benchmark_config_ci.json` | **CI** variant used by `.github/workflows/test-matrix.yml` — TensorSharp vs llama.cpp only, `ggml_cuda` only, text + prefill scenarios |
| `benchmark_config_glm53_qwen38.json` | **GLM-5.3 / GLM-5.3-Flash / Qwen3.8-Flash-Next** on an 8×A40 box — two columns, because the families disagree about multi-GPU placement: the GLM native executor takes every visible GPU with no `--tp` (default column), while `qwen4exp` is spread by the shared loader and needs `--tp N` (second column, `min_tp`-gated out of the default one). TensorSharp-only by default: that host's llama.cpp has no `glm5next`. Select with `--config` |
| `download_models.py` | pre-fetches the selected models from the `source` URLs in the config (optional — `run_matrix.py` already downloads what is missing) |

## Configuration

Every setting lives in **`benchmark_config.json`** — nothing is hardcoded in the
Python. It holds the host paths (`paths`), the model / scenario / engine /
backend registries (`models`, `scenarios`, `engines`, `backends` — see
[Choosing compute backends](#choosing-compute-backends---backends) for the
backend entry format), the llama-server launch options (`llama`),
per-size-class readiness timeouts (`ready_timeout_s`), and the run defaults
(`defaults`: which engines / models / scenarios / backends to run, MTP modes,
tensor-parallel degrees + the GPU pool they may use, concurrency levels,
max-tokens, warmup count, server max-tokens headroom).

Each model entry also declares **where its files come from** (`source`, plus
per-file `url` overrides) so a fresh host provisions itself — see
[Model provisioning](#model-provisioning-automatic-downloads) — and, for weights
that do not fit one GPU, the smallest tensor-parallel degree that can host them
(`min_tp`).

Values resolve with this precedence (highest first):

1. **Command-line flags** to `run_matrix.py` / `report.py` (e.g. `--models`, `--scenarios`, `--max-tokens`).
2. **Environment variables** — host paths only, for retargeting without editing the file (`BENCH_MODEL_ROOT`, `BENCH_TS_SERVER_DLL`, `BENCH_LLAMA_SERVER`, `BENCH_VLLM_URL`, `BENCH_SDCPP_EXE`, `BENCH_IMAGE`, `BENCH_AUDIO`, `BENCH_VIDEO`, `BENCH_RESULTS`, `BENCH_QWEN_IMAGE_DIT` / `BENCH_QWEN_IMAGE_VAE` / `BENCH_QWEN_IMAGE_VL` / `BENCH_QWEN_IMAGE_MMPROJ` / `BENCH_QWEN_IMAGE_LORA` (image-edit components), `DIFFUSION_STEPS`, ...).
3. **`benchmark_config.json`** (or the file named by `--config PATH` / `BENCH_CONFIG`).
4. Built-in fallbacks in `config.py`.

Path strings in the config may use the placeholders `${repo_root}`, `${here}`,
`${model_root}`, `${gemma4_qat_dir}`. A path may also be written as
`{"path": "...", "env": "BENCH_X", "url": "..."}` so the named environment
variable overrides it and a missing file can be downloaded from `url`.
Point the harness at an alternate settings file with `--config other.json`
(or `BENCH_CONFIG=other.json`) — useful for keeping per-host configs side by side.

## Prerequisites

- **Python 3.10+** with `requests`, `opencv-python` (video frame sampling). Both already present on the dev box.
- **TensorSharp.Server** built: `TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll` (run with `dotnet`). Build with `dotnet build TensorSharp.Server.Host -c Release` if missing/stale.
- **llama.cpp** server binary at `C:/Works/llama.cpp/build-cuda/bin/Release/llama-server.exe` (CUDA build).
  Non-CUDA backend columns use per-backend builds declared in the `backends` registry
  (e.g. a Vulkan build at `build-vulkan/.../llama-server.exe`, overridable via
  `BENCH_LLAMA_SERVER_VULKAN`); a missing build just records that column's llama.cpp cells as skipped.
- **vLLM** (optional): start an OpenAI server yourself and point the harness at it; otherwise vLLM cells record `skipped (engine unavailable)`.
- **stable-diffusion.cpp** (image-edit scenario): `sd-cli.exe` at
  `C:/Works/stable-diffusion.cpp/build/bin/Release/sd-cli.exe` (CUDA build;
  override via `paths.sdcpp_exe` / `BENCH_SDCPP_EXE`). Missing binary just
  records the `sdcpp` cells as skipped.
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

# Tensor parallelism — one model split across 1 / 2 / 4 GPUs (both engines)
python run_matrix.py --config benchmark_config_multigpu.json \
    --models qwen35-35b-a3b --scenarios text_short,text_long --tp 1,2,4

# Parallel-request scaling — aggregate decode throughput under load
python run_matrix.py --engines tensorsharp,llamacpp --backends ggml_cuda \
    --models gemma4-12b --scenarios text_short --concurrency 1,4,8

# MoE CPU offload — what "make it fit" costs, fully resident vs 8 layers vs all
python run_matrix.py --engines tensorsharp,llamacpp --backends ggml_cuda \
    --models qwen36-35b-a3b --scenarios text_short,prefill_8k \
    --n-cpu-moe off,8,all --cpu-moe-threads 48

# Multi-step agentic tool use, and code generation followed by a specific edit
python run_matrix.py --engines tensorsharp,llamacpp --backends ggml_cuda \
    --models qwen36-35b-a3b --scenarios agentic,code_edit

# Use an alternate settings file (e.g. a second host)
python run_matrix.py --config configs/host-b.json --engines tensorsharp

# Generate the markdown + CSV report
python report.py
```

With no flags, `run_matrix.py` runs the full matrix defined by the `defaults`
section of `benchmark_config.json`. Any flag overrides the corresponding config
default for that run only (the file is never modified).

Useful flags: `--config <file>` (pick the settings file), `--engines`,
`--backends`, `--models`, `--scenarios`, `--mtp`, `--tp`, `--n-cpu-moe`,
`--cpu-moe-threads`, `--concurrency`, `--max-tokens N`, `--warmup N`
(0 disables), `--download auto|never|force`, `--skip-existing` (reuse prior
`ok` cells), `--results <dir>`. `report.py` accepts `--config` and `--results`.

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
Each mode relaunches the server (it is a load-time flag): `on` adds `--spec`,
and for Gemma 4 also `--draft-model <draft.gguf>` (Qwen 3.6 embeds its NextN
block in the trunk, so no extra file is needed — but only GGUFs from the
`unsloth/Qwen3.6-35B-A3B-MTP-GGUF` repo retain that block; base-repo Qwen3.6
GGUFs with the same file names strip it and the server silently falls back to
standard decode, making the `on` and `off` cells measure the same thing). MTP is a TensorSharp feature —
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

### Reasoning mode (`defaults.thinking`)

Every request carries an explicit reasoning-mode flag, spelled per engine —
TensorSharp's top-level `think`, llama.cpp's `chat_template_kwargs.enable_thinking`
(which the GGUF's own chat template consumes). This is **not** cosmetic: the
engines do not default the same way, and with reasoning left on, llama.cpp
spends the whole per-scenario token budget thinking and never reaches the final
answer, so its output shares almost nothing with TensorSharp's direct answer and
its `json_mode` cell returns an unfinished (invalid) object. Pinning both sides
to the same mode is what makes the output-quality table mean anything — on
Qwen 3.5 9B it moved cross-engine agreement from 0.07 to 0.42 on `text_short`
and from 0.04 to **0.98** on `json_mode`. `defaults.thinking` (default `false`)
sets it for a run.

### Tensor parallelism (`--tp 1,2,4`)

Benchmarks one model **split across N GPUs in a single process**, on both server
engines. Each degree relaunches the server (it is a load-time decision):

| | how the model is split | how the GPUs are chosen |
|---|---|---|
| TensorSharp | `--tp N` | `CUDA_VISIBLE_DEVICES` (or `GGML_VK_VISIBLE_DEVICES` on Vulkan) = the first N ids of `defaults.tp_devices` |
| llama.cpp | `--split-mode tensor` (weights **and** KV split across the devices, NCCL-backed) | same env pinning |

Because the device set is pinned per cell, a `--tp 4` cell really occupies four
GPUs and the `tp=1` baseline really occupies one — on a multi-GPU box llama.cpp
would otherwise spread every model over all of them by default (layer split),
which is not the same measurement. Set `defaults.tp_devices` to the GPU ids the
harness may use (`BENCH_TP_DEVICES=0,1,2,3` overrides it per host); leave it
empty on a single-GPU box and nothing is pinned.

Configuration lives in the `backends` registry:
`tensorsharp: {"tp": true|false, "tp_arg": "--tp"}` and
`llamacpp: {"tp": true|false, "tp_extra_args": ["--split-mode", "tensor"]}`,
plus `"visible_devices_env"`. The defaults are inferred — TensorSharp can
tensor-parallelize on `cuda` / `ggml_cuda` / `ggml_vulkan`, llama.cpp on any
gpu-kind backend with `-ngl > 0` — so an existing config needs no edits to gain
the axis. Cells that cannot run a degree are recorded as skipped with the
reason (CPU backend, an engine without a TP path, fewer configured GPUs than
requested, or the image-edit pipeline, which is single-GPU).

A model whose weights do not fit one GPU declares **`"min_tp": N`**; its cells
below that degree are skipped ("needs `--tp N`") instead of being left to OOM
the box. **`"max_tp": N`** is the other end: some architectures cannot be
sharded past a small head count (Gemma 4 26B-A4B has 2 KV heads, so TensorSharp
rejects `--tp 4` at load time), and declaring it records those cells as skipped
rather than as a launch failure — llama.cpp splits the same model differently
and is not bound by it, so its column still fills in.

`report.py` renders each degree as its own column (`… · tp2`) and adds a
**Tensor parallelism** section with decode/prefill at every degree plus the
scaling factor over the smallest degree that ran.

Not every model reaches multiple GPUs the same way. A model with its own
whole-model executor (DeepSeek V4 Flash) passes `tpDegree = 1` down to
`ModelBase` and reinterprets `--tp N` as "spread my weights over N GPUs",
**layer-split, with no AllReduce** — that is capacity parallelism (it is what
makes a 150 GiB model hostable on 4×46 GB at all), not the per-layer tensor
split the other architectures do. Its row in the tensor-parallelism table is
therefore the one configuration that runs, not a scaling point; the config
entry says so in a `_tp_note`.

Two field notes from the 4×A40 validation host:

- Current llama.cpp CUDA builds **no longer implement `--split-mode row`**
  (`device CUDA0 does not support split buffers`); `--split-mode tensor` is the
  supported tensor-parallel mode, which is why it is the default here.
- Some cloud hosts **advertise GPU peer access that does not work**
  (`nvidia-smi topo -p2p r` says `OK` for every pair). llama.cpp then hangs
  forever in its first NCCL collective instead of loading; `NCCL_P2P_DISABLE=1`
  in the backend's `llamacpp.env` (as in `benchmark_config_multigpu.json`) fixes
  it. TensorSharp needs no such flag: it verifies peer delivery itself before
  the first communicator exists and, when the advertisement proves false, takes
  peer transport away from NCCL rather than giving up the device collective.

### MoE CPU offload (`--n-cpu-moe off,8,all`)

Keeps the routed experts of the first N layers in **system RAM** and multiplies
them on the host; attention, the norms, the router and the always-active shared
expert stay on the accelerator. Like `--tp`, it is a load-time decision, so each
point relaunches the server.

| | some layers | every layer | host threads |
|---|---|---|---|
| TensorSharp | `--n-cpu-moe N` | `--n-cpu-moe all` | `--cpu-moe-threads M` |
| llama.cpp | `--n-cpu-moe N` | `--cpu-moe` — it parses `-ncmoe`'s argument as an integer and spells "every layer" as its own switch, so `--n-cpu-moe all` would abort llama-server at startup | `--threads M` — its nearest equivalent; llama.cpp has no MoE-specific worker pool, so this moves *every* CPU-side op |

The harness sends whichever of those the point needs; both engines' spellings are
overridable per backend (`cpu_moe_arg`, `cpu_moe_all_arg`, `cpu_moe_threads_arg`),
and only the TensorSharp ones were read out of this repo's own
`ServerOptionsBuilder`. The llama.cpp mapping is taken from the equivalence
FEATURES.md states (`-ncmoe` / `-cmoe`) and has **not** been exercised against a
real llama-server here.

This axis is about **fitting, not speed**: it is what makes a checkpoint that
does not fit run at all (and it nearly doubles the context the loader can size),
and it pays for that in throughput — on 3×RTX PRO 6000, where GLM-5.2 already
fits, `--n-cpu-moe 30` takes pp2048 from 915.9 to 94.7 tok/s and tg64 from 43.9
to 16.4. A ratio below `1.0×` in the report's **MoE CPU offload** section is
therefore the expected shape, and the number you are buying VRAM with.

```bash
--n-cpu-moe off,8,all      # axis points: fully resident, 8 layers, every layer
--cpu-moe-threads 48       # applied to the offloaded points only
```

`--cpu-moe-threads` is a second axis (`32,64` sweeps both), but it only
multiplies the *offloaded* points — `off` collapses to one baseline cell however
many thread counts were asked for, because the baseline never runs a host
matmul. `0` (the default) sends no thread count at all and leaves each engine's
own default alone; TensorSharp's is half the CPU parallelism the process can
actually use, and sizing it near the CPU quota is a cliff rather than a slope.

`off` is the only word for the baseline point (`0` also works because the engine
flag itself takes it); anything else — `none`, a typo — is an error rather than a
quietly un-offloaded cell.

Configuration lives in the `backends` registry:
`tensorsharp: {"cpu_moe": true|false, "cpu_moe_arg": "--n-cpu-moe", "cpu_moe_threads_arg": "--cpu-moe-threads"}`
and the same keys under `llamacpp`, which additionally takes `cpu_moe_all_arg`.
Support is inferred — any gpu-kind backend the engine can launch — so an
existing config gains the axis without being edited. Cells that cannot offload
are recorded as skipped with the reason: a CPU-kind backend (its experts are
already host-resident), an engine this harness does not launch (vLLM, sd.cpp),
the image-edit pipeline, a model whose config entry explicitly declares
**`"is_moe": false`**, or a backend that already pins the offload itself.

That last one is the pre-axis way of measuring this — a cloned backend entry per
offload point, like `ggml_cuda_layer_cpu_moe4` in
`benchmark_config_deepseek41.json`, which hardcodes `--n-cpu-moe 4` in its
`extra_args`. Sweeping the axis on top of such a backend would hand the engine
the flag twice and every engine here keeps whichever it parsed last, so the cell
is refused by name instead of recorded under an offload point it did not run.
Note the converse, which the harness cannot fix: the baseline (`off`) point on
such a backend records `cpu_moe_layers: 0` while the backend is in fact
offloading 4 layers, because the offload is part of that backend's identity
rather than of this axis. Prefer `--backends ggml_cuda --n-cpu-moe 4`.

A model that says nothing about `is_moe` runs the cell and lets the engine be
the one to report a dense checkpoint — the harness does not guess an
architecture fact the config never stated. The cost of that choice is that a
dense model whose engine simply no-ops the flag produces a `1.00×` row that
looks like a measurement; declare `"is_moe": false` on dense entries to get a
skip with a reason instead.

Defaults live in `defaults.cpu_moe_layers` (default `["off"]`) and
`defaults.cpu_moe_threads` (default `[0]`), so a run that does not pass the
flags launches the exact command line it always did.

### Multi-turn workflows (`agentic`, `code_edit`)

Two scenarios drive **more than one round trip per cell**. Like `prefill_<N>`,
they are synthesized rather than declared, so `--scenarios agentic` works
against any config file — including one written before they existed — and
neither is in any config's `defaults.scenarios`, so existing runs are unchanged.

| Scenario | What it drives | What is checked |
|---|---|---|
| `agentic` | 3 round trips: `read_invoice("INV-472")` → `calculate_total(unit_price, quantity)` → a final answer with `tool_choice: "none"` | The tool-call *shape* (a structured `tool_calls` entry with a unique non-empty id, `type: "function"`, the declared name, arguments that are complete JSON) the way `validate_deepseek41_tools.py` checks it, **and** that the answer depends on the results |
| `code_edit` | 2 round trips: write `slugify(text)`, then rename it to `slugify_title` and add a `max_length=40` parameter that truncates | The edited source is **parsed** (`ast`): the new name, exactly `(text, max_length)`, the literal default `40`, and `max_length` actually read in the body — a declared-but-unused parameter is a rename, not the edit |

The dependence check in `agentic` is deliberate rather than incidental. `13.75`
and `5` appear nowhere in the prompt, so turn 2's arguments can only have come
from turn 1's result; and `calculate_total` returns `74.25`, **not**
`13.75 × 5 = 68.75`, because the fixture adds a handling fee the model was never
told about — so a model that answers from arithmetic instead of from the tool
result lands on 68.75 and is recorded as wrong. Every tool result is a fixed
fixture; nothing is executed.

Both loops are driven **from the client**. TensorSharp's own code-execution tool
surface (`--code-exec`: `shell`, `read_file`, `edit_file`, `write_file`,
`apply_patch`) is answered *inside the server* and never handed back to the API
client, it is off by default, and an OpenAI-request workspace is destroyed when
the response ends — so it can neither be observed round-trip-by-round-trip nor
carry a file from one request to the next, and llama.cpp and vLLM have no
equivalent at all. A client-driven loop is the only shape that is both
measurable and identical on every engine, which is what makes these cells
comparable.

A follow-up that cannot continue — the model emitted no structured tool call,
or called the wrong function — **stops the conversation there** instead of
fabricating the next turn out of a response that did not contain what that turn
needs. The cell still records `ok` (the server answered every request it was
given) with its timings, `turns` / `turns_expected` say how far the workflow got,
`tool_call_ok` is `false`, and `detail` names the turn and the reason.

Such a cell's timings describe a **shorter conversation** than its name, so
`report.py` prints it as `partial 1/3` in the throughput tables and leaves it out
of every ratio, rather than comparing one round trip against another engine's
three. The verdict itself is in the report's **Tool-call and workflow
correctness** table. Under `--concurrency N` the recorded `turns` is the worst
client's, so one copy that broke marks the whole cell partial.

For a multi-turn cell the reported `ttft_ms` / `prefill_tps` / `decode_tps` /
token counts are the **final** turn's — the turn whose prompt is the whole
conversation, which is what an agentic workload is actually paced by — while
`total_wall_ms` covers every turn end to end. `--concurrency N` runs N
independent copies of the workflow at once and aggregates them exactly like the
single-request scenarios.

### Model provisioning (automatic downloads)

Every model entry declares where its files come from, so a run on a host that
does not have them yet provisions itself and every later run reuses the files:

```jsonc
"qwen35-9b": {
  "source": "unsloth/Qwen3.5-9B-GGUF",            // HF repo id, {"hf_repo","revision"}, or a base URL
  "gguf":   "${model_root}/Qwen3.5-9B-Q8_0.gguf", // -> <repo>/resolve/main/<file name>
  "mmproj": {"path": "${model_root}/Qwen3.5-9B-Q8_0-mmproj.gguf",
             "url":  "mmproj-F16.gguf"}           // remote name differs from the local one
}
```

A file's `url` may also be an absolute URL from a different repo (the
image-edit model pulls its DiT, VAE, text encoder and LoRA from four different
places), and split GGUFs (`-00001-of-00005.gguf`) expand to every shard
automatically — including deriving each shard's URL from the first one.

`run_matrix.py` fetches whatever is missing before the matrix starts
(`--download auto`, the default), streaming to `<file>.part` and resuming with a
range request if interrupted, so a partial file never masquerades as a complete
model. `--download never` requires the files to already exist (missing ones are
recorded as skipped cells); `--download force` re-fetches. `HF_TOKEN` is honored
for gated repos. To provision ahead of time (a CI cache step, a fresh GPU box):

```bash
python download_models.py --config benchmark_config_multigpu.json
python download_models.py --config benchmark_config.json --models qwen35-9b,gemma4-12b
```

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

### Image editing / stable diffusion (`image_edit`)

The `image_edit` scenario benchmarks the **stable-diffusion image-editing
engine** — TensorSharp's Qwen-Image-Edit pipeline against the
**stable-diffusion.cpp** CLI (`sdcpp` engine) — on the *same* weights and the
*same* task:

- **Same everything**: the benchmark image (`paths.media.image`) is pre-resized
  once to the exact dims TensorSharp's `ResizeToArea` picks for the scenario's
  `edit.target_area` (aspect-preserving, multiple of 16) and saved as PNG; both
  engines then edit those identical pixels at that identical resolution
  (TensorSharp via `targetArea`, sd.cpp via `-W/-H`), with the same prompt,
  steps, cfg and seed from the scenario's `edit` block.
- **TensorSharp** runs as a server (launched with `--model <dit.gguf>
  --qwen-image-vae/-vl/-mmproj/-lora …` from the model's `components`) and is
  driven through multipart `POST /api/image-edit`. Each cell sends **two**
  requests: the *cold* first request (pays the per-request DiT rebuild + graph
  capture on a fresh server → `edit_first_total_ms`) and the *warm* steady-state
  request (the headline `edit_total_ms`).
- **stable-diffusion.cpp** runs one `sd-cli` process per cell
  (`--diffusion-model … --vae … --llm … --llm_vision … --model-args
  qwen_image_zero_cond_t=true --sampling-method euler --flow-shift 3`, LoRA via
  the `<lora:…:1>` prompt tag; per-backend `extra_args` such as
  `--diffusion-fa` come from the `backends.*.sdcpp` registry entry).
- **Metrics are each engine's own pipeline timers**, so weight-file loading and
  HTTP/process overhead are excluded on both sides: TensorSharp's
  `[pipe-timing]` phases + the server's `elapsedSeconds`; sd.cpp's
  `get_learned_condition` / `sampling` / `encode_first_stage` /
  `decode_first_stage` phase logs + its `generate_image` total. Recorded per
  cell: `edit_total_ms`, `edit_first_total_ms`, `edit_text_encode_ms`,
  `edit_vae_encode_ms`, `edit_sampling_ms`, `edit_per_step_ms`,
  `edit_vae_decode_ms`, output resolution, and the output image itself
  (`results/images/…png`, for visual verification).

Applicability gating keeps the matrix clean: `image_edit` only runs on
TensorSharp + sd.cpp with the image-edit model, the image-edit model runs no
other scenario, `sdcpp` runs no other scenario, MTP and `--concurrency > 1` are
recorded as skips, and `report.py` renders these cells in their own **Image
editing (stable-diffusion)** section (phase table + TensorSharp-vs-sd.cpp
speedups) instead of the token-throughput tables.

```bash
# Just the image-edit comparison
python run_matrix.py --engines tensorsharp,sdcpp --backends ggml_cuda \
    --models qwen-image-edit --scenarios image_edit
```

### Parallel requests (`--concurrency 1,4,8`)

Fires N identical requests at the *same* server at once (the server's continuous
batching serves them concurrently) and records, per cell:

| Metric | Definition |
|---|---|
| `decode_tps` | mean **per-request** decode tok/s |
| `aggregate_decode_tps` | **system-wide** decode tok/s — total generated tokens / the wall window during which any sequence was decoding |

plus `requests_ok` (how many of the N succeeded). The console prints
`ok=N/M` and `agg=` on every concurrent cell, and `report.py` adds a
**parallel-request scaling** table (per-request vs aggregate at each
concurrency). The axes compose: `--mtp on --concurrency 4` is valid, though MTP
only engages for solo sequences so it has little effect under load, and
`--concurrency N` on `agentic` / `code_edit` runs N independent copies of the
whole workflow.

Result files keep their historical names for the baseline (`mtp` off, `tp` 1,
no offload, `concurrency` 1); non-default cells add a `__mtp`, `__tp<N>`,
`__ncmoe<N>[t<M>]` and/or `__c<N>` suffix.

## Output

- `results/{engine}__{backend}__{model}__{scenario}[__mtp][__tp<N>][__ncmoe<N>[t<M>]][__c<N>].json`
  — one record per cell (`status` ∈ `ok | fail | skipped`, plus token counts,
  `mtp`, `tp`, `cpu_moe_layers`, `cpu_moe_threads`, `concurrency`,
  `aggregate_decode_tps`, `requests_ok`, `turns`, `turns_expected`, and
  throughput). The baseline (MTP off, one GPU, fully GPU-resident, single
  request) keeps the suffix-free name.
- `results/logs/{engine}__{backend}__{model}[__mtp][__tp<N>][__ncmoe<N>[t<M>]].log` — captured
  server stdout/stderr (the first place to look when a group reports `fail`).
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

The `results/` directory (per-cell JSONs, logs, images, CSV) is generated
locally by each run and is **not committed** to the repository — the committed
artifact is the generated `docs/engine_comparison_report.md`.

## CI (GitHub Actions)

`.github/workflows/test-matrix.yml` runs this harness on the self-hosted
`tensorsharp-cuda` runner in two profiles: a trimmed **smoke** profile on every
pull request (`gemma4-12b` only; `text_short`, `function_call`, `json_mode`,
`prefill_4k`; report posted as a PR comment), and the **full** CI set on
demand via `workflow_dispatch` (inputs select a custom subset) plus a weekly
schedule. Each run:

1. builds TensorSharp (native GGML CUDA library + `TensorSharp.Server`),
2. clones and builds **llama.cpp** (CUDA, `llama-server`; pick the ref with the
   `llama_ref` input),
3. downloads the benchmark models via `download_models.py` (from the `source`
   repo declared by each model in `benchmark_config_ci.json`),
4. runs `run_matrix.py --config benchmark_config_ci.json` (TensorSharp vs
   llama.cpp on `ggml_cuda`, text + prefill scenarios), and
5. generates the combined **performance + output-quality** report with
   `report.py`, uploads it (plus the per-cell JSONs/logs) as artifacts, and
   renders it into the job summary.

llama.cpp sources/build and the downloaded models live in a persistent
directory on the runner (`$HOME/tensorsharp-bench`, overridable with the
`BENCH_HOME` repository variable), so repeat runs only pay incremental costs.
The run fails if any benchmark cell reports `fail` or nothing ran `ok`.

### Verifying the macOS / MLX path

The CI workflow is CUDA-only, but `benchmark_config_ci.json` also registers
the `ggml_metal` and `mlx` backends, so the same config verifies the macOS
path on an Apple Silicon host (llama.cpp has no MLX backend — the
apples-to-apples reference on that host is llama.cpp on `ggml_metal`, and the
`mlx` column is TensorSharp-only):

```bash
# One-time host setup: native libs + server + a Metal llama-server build
bash TensorSharp.GGML.Native/build-macos.sh
bash TensorSharp.Backends.MLX/build-native-macos.sh
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release
git clone https://github.com/ggml-org/llama.cpp ~/tensorsharp-bench/llama.cpp
cmake -S ~/tensorsharp-bench/llama.cpp -B ~/tensorsharp-bench/llama.cpp/build \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=ON
cmake --build ~/tensorsharp-bench/llama.cpp/build --config Release --target llama-server -j

# Models, benchmark, report — same pipeline as CI
cd benchmarks/engine_comparison
export BENCH_MODEL_ROOT=~/tensorsharp-bench/models
export BENCH_LLAMA_SERVER=~/tensorsharp-bench/llama.cpp/build/bin/llama-server
python3 download_models.py --config benchmark_config_ci.json --models gemma4-12b
python3 run_matrix.py --config benchmark_config_ci.json \
    --backends ggml_metal,mlx --models gemma4-12b
python3 report.py --config benchmark_config_ci.json
```

To run it as a CI job instead, add a second job on the old
`[self-hosted, tensorsharp-mlx]` runner label with the build steps above and
`--backends ggml_metal,mlx` on the `run_matrix.py` line.

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
  head (Qwen 3.6 embedded NextN; Gemma 4 with its paired `--draft-model`);
  every other engine/model `on` cell is recorded as skipped.
- **`agentic`** needs an engine that returns *structured* `tool_calls`; a model
  that answers in prose instead stops the loop and the cell records
  `tool_call_ok=false` with the reason. DiffusionGemma cannot carry tool
  declarations at all, so its `agentic` cells are skipped — `code_edit`, being
  plain multi-turn text, still runs.
- **MoE CPU offload** (`--n-cpu-moe`) is skipped on CPU-kind backends (nothing
  to move), on the connect-only / CLI engines, on the image-edit pipeline, on
  models whose config entry declares `"is_moe": false`, and on backends that
  already pin the offload in their own `extra_args` (passing the flag twice
  would silently record the wrong point).

## DeepSeek V4.1 Flash strict validation

Use `benchmark_config_deepseek41.json` for the pinned seven-shard Q2_K model and
explicit layer-placement/CPU-MoE profiles. `validate_inference.py` validates
running endpoints with actual short prompts, long-context recall, strict JSON,
generated conversation history, tool-result round trips, a two-tool agent
workflow, repeated measurements and per-request concurrent correctness. It
retains full request/response artifacts and refuses to establish parity when
the reference is missing or invalid. See the
[validation protocol](../../docs/deepseek41_validation.md) for commands,
comparison requirements and uncovered capabilities.

For dependent tool workflows, the optional `--serial-tool-workflows` flag sends
`parallel_tool_calls: false` on their tool-bearing turns. It retains the exact
call order, argument and final-answer checks. The policy is recorded in report
metadata and affected request hashes; default requests remain unchanged. Record
this follow-up separately from the original quality run:

```bash
python validate_inference.py \
  --url http://127.0.0.1:5000 --engine tensorsharp --model deepseek-v4.1-flash \
  --weights-id 8e0c4de3cb6519bfc11ed69dc87184b457a57bb5-Q2_K \
  --profile layer8-ctx65536-ubatch1024-cpumoe0-compact1 \
  --scenarios tool_round_trip,agentic --concurrency 1,4 --repeats 1 \
  --structured-tool-results --serial-tool-workflows \
  --output results/deepseek41-serial-tools.json
```

These ten cases check a client serialization constraint. A successful follow-up
does not erase a default-policy failure or establish a numerical inference fix.

`--accept-fenced-json` is a second, opt-in verdict for the JSON scenarios: when
the only defect in an answer is a single ```` ```json ```` fence around
otherwise exact JSON, the case additionally records `lenient_status: "ok"` (and
`lenient_detail`). The strict `status` stays the case status, the exit code and
the `passed` count; the per-scenario summary adds `lenient_passed`, and the
progress line prints both. Wrong or truncated fenced answers stay lenient-fail.
The flag changes no request, so a run with it hashes identically to one without.

`validate_deepseek41_tools.py` separately exercises tool policy through actual
HTTP/SSE. Its default plan has 30 cases: `required`, a named weather function
among two declarations, `none` after a generated weather call and fixed result,
serial one-call output, and explicit parallel two-call output at concurrency
1/4; required/named thinking requests and three HTTP400 validation cases run at
concurrency 1. The history case makes two requests, for 35 requests in total.
Tools return fixed fixture data; the harness never executes external tools.

```bash
python validate_deepseek41_tools.py \
  --url http://127.0.0.1:5000 --model deepseek-v4.1-flash \
  --weights-id 8e0c4de3cb6519bfc11ed69dc87184b457a57bb5-Q2_K \
  --profile layer8-ctx65536-ubatch1024-cpumoe0-compact1 \
  --server-build server-v41-tools --output results/deepseek41-tool-policies.json
```

Set `--native-sha256` to the loaded library hash when recording a measured run.
`--scenarios`, `--concurrency`, and `--repetitions` can select a narrower rerun;
thinking calls use `--thinking-max-tokens` (default 2,048), while ordinary calls
use `--max-tokens` (default 512). The report records the exact execution plan,
model/profile, source and fixture hashes, every request and its hash, raw SSE
lines (including partial parser failures), metrics, and untruncated HTTP error
bodies. `run_complete` stays false until every planned case finishes. Exit
status is nonzero for any failed case, including an explicit two-call request
that returns only one call. Reasoning text cannot substitute for a final answer
or structured tool call; malformed/duplicate call IDs, malformed arguments,
wrong cities, missing usage, and unrelated HTTP400 errors fail validation.

Run the local harness checks with the same `requests` dependency as the engine
matrix:

```bash
python -m unittest test_validate_deepseek41_tools test_validate_inference \
    test_backend_launch test_scenarios test_model_registry
```

`test_model_registry` needs nothing but the config files: it loads every
`benchmark_config*.json` through `config.py` and checks that the registries
resolve (defaults name real models/engines/backends, every model file has a
download URL, nothing that is exported to a server process is a comment), plus
the per-matrix facts that decide whether a cell can run at all.
