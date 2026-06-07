# Test matrix baselines

[English](README.md) | [中文](README_zh-cn.md)

This directory holds per-host baseline snapshots used by
[`TensorSharp.TestMatrix`](../README.md) to detect throughput and correctness
regressions in CI.

## File layout

One file per host class. The runner picks the file matching its host
automatically (see `BaselineStore.DefaultHostLabel`):

| File | Host class | Backends typically populated |
|---|---|---|
| `baseline-macos-mlx.json` | Apple Silicon macOS | `cpu`, `ggml_cpu`, `ggml_metal`, `mlx` |
| `baseline-linux-cuda.json` | Linux + NVIDIA GPU | `cpu`, `ggml_cpu`, `ggml_cuda`, `cuda` |
| `baseline-windows-cuda.json` | Windows + NVIDIA GPU | `cpu`, `ggml_cpu`, `ggml_cuda`, `cuda` |

Override with `--baseline <path>` to point at a specific file (useful for
A/B comparisons against a captured run from a different commit).

## What's stored

Each entry records only what's needed for regression detection — not the full
per-cell JSON. The minimal payload keeps the committed files small and
human-readable:

```json
{
    "host_label": "macos-mlx",
    "captured_at": "2026-05-26T00:00:00.000Z",
    "commit": "abc1234",
    "notes": "...",
    "entries": [
        {
            "case_id": "gemma-4-e4b-it-q8_0__ggml_metal__short_text__baseline",
            "ok": true,
            "correctness_ok": true,
            "prefill_tps": 2480.5,
            "decode_tps": 95.3,
            "model_load_ms": 1850.0
        }
    ]
}
```

Only **cells that passed** the runtime + correctness gate are recorded. A
flaky or broken cell at baseline-capture time is intentionally not enshrined
as the "expected" state.

## How regressions are detected

For each cell in the current run, the runner compares against its baseline
entry (matched by `case_id`):

| Condition | Classified as | Blocks `--fail-on-regression` |
|---|---|:-:|
| Baseline `ok=true`, current `ok=false` (runtime crash, timeout, parse failure) | New runtime failure | ✅ |
| Baseline `ok=true`, current `correctness_ok=false` (output check missed) | New correctness failure | ✅ |
| Both `ok=true`, current `decode_tps` (or `prefill_tps` for prefill-only) dropped by more than threshold % | Throughput regression | ✅ |
| Baseline `ok=false`, current `ok=true` | Improvement | ❌ |
| No baseline entry exists for this cell | Untracked | ❌ |

The default threshold is **10%** decode-TPS drop. Override with
`--regression-threshold-pct N` or the `regression_threshold_pct` field in
`matrix-config.json`.

## Updating a baseline

Baselines decay over time as the codebase legitimately gets faster, slower,
or grows new cells. Refresh them deliberately:

### Locally

```bash
dotnet TensorSharp.TestMatrix/bin/TensorSharp.TestMatrix.dll \
  --update-baseline \
  --model-dir /Users/ZhongkaiFu/work/model
```

The runner writes the per-host file in place. Inspect the diff and commit it
together with the change that justified the new numbers.

### From CI

Trigger the workflow with `update_baseline = true` from the
**Actions → Test Matrix → Run workflow** UI. Each host runner uploads its
refreshed `baseline-<host>.json` as an artifact; a maintainer downloads,
inspects, and opens a PR replacing the committed files.

This indirection is intentional: baseline drift should be a code-reviewed
change, not an automatic side-effect of CI.

## Stale baseline / untracked cells

Adding a new model, backend, feature, or env-var sweep will produce
"untracked" cells in the next run — they have no baseline to compare against
and are not gated. Re-run with `--update-baseline` after the new cells stabilize
to bring them under regression detection.

Removing a model from the matrix will leave stale entries in the baseline
file. They are silently ignored at compare time and are harmless; they get
cleaned up the next time you regenerate the baseline.
