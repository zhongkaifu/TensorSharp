# TensorSharp.TestMatrix

[English](README.md) | [中文](README_zh-cn.md)

A test and benchmark matrix runner for TensorSharp inference. It sweeps the
cartesian product of **(model × backend × feature × env-var)** by invoking
`TensorSharp.Cli` as a subprocess, parses per-run metrics from its structured
log output, persists one JSON file per cell, and emits an aggregate markdown
report.

The companion document [`docs/env_var_feature_matrix.md`](../docs/env_var_feature_matrix.md)
is the curated reference for which env vars get swept and what features they
affect — keep them in sync when you add a new flag.

## What gets exercised

| Axis | Values |
|---|---|
| **Models** | Auto-discovered GGUFs in `/Users/ZhongkaiFu/work/model` (configurable); explicit overrides also supported. DiffusionGemma is not auto-classified as a first-class matrix family yet; use an explicit config entry before experimenting with it. |
| **Backends** | `cpu`, `ggml_cpu`, `ggml_metal`, `ggml_cuda`, `cuda`, `mlx` (host-availability filtered) |
| **Features / prompt types** | Autoregressive CLI features: synthetic prefill (512, 2048), synthetic decode (128), short text, long text, uploaded text, multi-turn chat, function/tool calling, thinking mode, image, audio, video. There is no dedicated diffusion feature yet. |
| **Env-var sweeps** | Baseline cells plus the curated high-impact flags selected by `default_env_vars` in [`Defaults/matrix-config.json`](Defaults/matrix-config.json). The registered superset lives in `EnvVarMatrix.All`; DiffusionGemma `DIFFUSION_*` knobs are currently out of matrix and not scrubbed/swept by default. See the [matrix doc](../docs/env_var_feature_matrix.md). |

## Building

```bash
dotnet build TensorSharp.TestMatrix/TensorSharp.TestMatrix.csproj -c Release
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release   # required: subprocess driver target
```

The runner needs a built `TensorSharp.Cli` binary; it auto-discovers it at
`../TensorSharp.Cli/bin/TensorSharp.Cli` relative to its own bindir. Override
with `--cli-executable <path>` or `cli_executable` in the config.

## Running

Default run (everything the host supports, all curated env-var sweeps):

```bash
dotnet run --project TensorSharp.TestMatrix -c Release
```

Each applicable cell also gets a baseline run with no forced env var. Sweep
cases pass exactly one env var/value pair to the subprocess after inherited
`TS_*`/related variables are scrubbed.

Curated subset — useful for an interactive dev cycle:

```bash
dotnet run --project TensorSharp.TestMatrix -c Release -- \
  --backends ggml_metal,mlx \
  --features short_text,long_text,image \
  --env-vars KV_CACHE_DTYPE,TS_QWEN35_BATCHED \
  --model-dir /Users/ZhongkaiFu/work/model \
  --results-dir results/dev \
  --report report-dev.md
```

Dry-run (print every case id it would execute):

```bash
dotnet run --project TensorSharp.TestMatrix -c Release -- --dry-run
```

Disable env-var sweeps and run only baseline cells:

```bash
dotnet run --project TensorSharp.TestMatrix -c Release -- --env-vars none
```

Resume an interrupted run without re-executing cells already on disk:

```bash
dotnet run --project TensorSharp.TestMatrix -c Release -- --skip-existing
```

## Configuration

A JSON config file controls model discovery, defaults, and per-model overrides.
The default lives at [`Defaults/matrix-config.json`](Defaults/matrix-config.json)
and is copied next to the built assembly. Override fields on the command line —
those win over the file.

Per-model overrides look like this:

```json
{
    "models": [
        {
            "id": "gemma4-e4b-q8",
            "family": "gemma4",
            "display_name": "Gemma 4 E4B Q8_0",
            "gguf": "gemma-4-E4B-it-Q8_0.gguf",
            "mmproj": "gemma-4-mmproj-F16.gguf",
            "supports_image": true,
            "supports_audio": true,
            "supports_video": true,
            "supports_tools": true,
            "supports_thinking": true,
            "enabled": true
        }
    ]
}
```

Relative paths are resolved against `model_dir`. Auto-discovered models that
share an `id` with a config entry are replaced by the config entry.

DiffusionGemma GGUFs should be added through an explicit `models[]` entry until
`ModelDiscovery` grows architecture-aware detection for `diffusion-gemma` /
`diffusion_gemma`. The current feature catalog is still append/decode oriented,
so add a dedicated diffusion feature before using TestMatrix output as a formal
diffusion regression signal.

## Media files (image / audio / video)

The Inputs tree ships only the **prompts**, not the media. Drop sample media at:

- `TensorSharp.TestMatrix/Inputs/media/apple.png`
- `TensorSharp.TestMatrix/Inputs/media/sample.mp3`
- `TensorSharp.TestMatrix/Inputs/media/sample.mp4`

…or set `media_dir` in the config to point at a shared assets directory. Cases
whose media file is missing fail with a clear error; they do not silently skip.

## Output

- `results/<case-id>.json` — one file per cell with full metrics + stdout tail + correctness verdict
- `report.md` — aggregated markdown report (baseline tables, env-var sweep
  tables, sensitivity summary, skipped combinations, correctness failures, runtime failures)

The report is the artifact uploaded in CI; the JSON files are the source of
truth for delta analysis against previous runs.

## What "pass" means

A cell is marked `ok = true` only when **all** of these hold:

1. `TensorSharp.Cli` exited with code 0 (no crash, no timeout).
2. At least one of `prefill_tps`, `decode_tps`, or `model_load_ms` was parsed from the structured log output (proves the model loaded and ran).
3. **Correctness check** — if the feature configures `ExpectedContains`, every required substring must appear in the assistant's output (case-insensitive). Failures are reported in a dedicated "Correctness failures" section separate from runtime failures.

Per-feature expected content lives in [`Matrix/FeatureCatalog.cs`](Matrix/FeatureCatalog.cs):

| Feature | Required in output |
|---|---|
| `short_text` ("why is the sky blue") | `blue` |
| `long_text` (paged-KV report summary) | `paged` |
| `uploaded_text` (server-log analysis) | `08:01:12` (the ERROR timestamp) |
| `multi_turn` (3-turn name/colour) | `alex`, `teal` |
| `tools` (Tokyo weather) | `get_current_weather`, `tokyo` |
| `thinking` (two-train word problem) | `10:38` |
| `image` (default apple.png) | `apple` |
| `pp512`, `tg128`, `pp2048` | _(none — synthetic benchmark, no text output)_ |
| `audio`, `video` | _(none by default — depends on the sample media)_ |

The check is intentionally weak (a single keyword is a necessary, not sufficient, condition for a correct answer). It catches catastrophic regressions — model loaded but generated gibberish, multi-turn KV reuse silently broken, tool name not emitted — without trying to be a model-quality benchmark. For richer verification, override `ExpectedContains` on a feature or add a regression-mode comparison against recorded golden outputs.

When you change the default media file (image, audio, video), update the corresponding feature's `ExpectedContains` to match.

## Regression detection (baselines)

The matrix is only as useful as the ability to spot regressions. The runner
loads a per-host baseline file from
[`Baselines/baseline-<host>.json`](Baselines/README.md) and compares the
current run against it:

- **New runtime failure** — was passing, now crashes / times out / no metrics
- **New correctness failure** — was passing, now misses an `ExpectedContains` substring
- **Throughput regression** — decode TPS dropped more than `--regression-threshold-pct` (default 10%)
- **Improvement** — was failing, now passing (informational)
- **Untracked** — no baseline entry yet (new cell)

Runtime / correctness / throughput regressions are *blocking*: with
`--fail-on-regression` set, the runner exits non-zero and the PR fails.
Improvements and untracked cells are informational.

```bash
# Compare against the committed baseline, fail PR on blocking regressions
dotnet TensorSharp.TestMatrix/bin/TensorSharp.TestMatrix.dll --fail-on-regression

# Re-bless the baseline after a legitimate change (commit the diff yourself)
dotnet TensorSharp.TestMatrix/bin/TensorSharp.TestMatrix.dll --update-baseline

# Compare against a specific baseline file (A/B against an older snapshot)
dotnet TensorSharp.TestMatrix/bin/TensorSharp.TestMatrix.dll \
  --baseline ./oldbaseline.json --fail-on-regression
```

See [`Baselines/README.md`](Baselines/README.md) for the file format,
host-class layout, and update workflow.

## Extending the matrix

| To add | Edit |
|---|---|
| A new backend | [`Matrix/Backend.cs`](Matrix/Backend.cs) — add a `BackendInfo`, register in `BackendCatalog.All` |
| A new prompt type / feature | [`Matrix/FeatureCatalog.cs`](Matrix/FeatureCatalog.cs) + [`Runners/CliRunner.cs`](Runners/CliRunner.cs) `BuildArgs` switch; drop any new prompt under `Inputs/prompts/` |
| A new env-var sweep | [`Matrix/EnvVarMatrix.cs`](Matrix/EnvVarMatrix.cs) — add an `EnvVarSpec` with an `AppliesTo` predicate; add it to [`Defaults/matrix-config.json`](Defaults/matrix-config.json) if it should run by default; add a row to [`docs/env_var_feature_matrix.md`](../docs/env_var_feature_matrix.md) and its Chinese version |
| A model the auto-discovery misses | Add a `ModelConfig` entry under `models[]` in `matrix-config.json` |
| DiffusionGemma coverage | Add family detection or explicit configs, a diffusion feature in `FeatureCatalog.cs`, CLI argument mapping for diffusion generation, and `DIFFUSION_*` env-var specs if those knobs should be swept |

## CI

The repo's GitHub Actions workflow [`test-matrix.yml`](../.github/workflows/test-matrix.yml)
runs the full matrix on every PR using self-hosted runners labelled
`tensorsharp-cuda` and `tensorsharp-mlx`. The PR comment includes a link to
the uploaded `report.md` artifact. See the workflow file for runner setup
expectations (model directory, NVIDIA drivers, etc.).
