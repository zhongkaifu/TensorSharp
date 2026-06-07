# 测试矩阵基线

[English](README.md) | [中文](README_zh-cn.md)

本目录保存每类主机的 baseline 快照，供
[`TensorSharp.TestMatrix`](../README_zh-cn.md) 在 CI 中检测吞吐与正确性回归。

## 文件布局

每类主机一个文件。运行器会根据当前主机自动选择对应文件
（见 `BaselineStore.DefaultHostLabel`）：

| 文件 | 主机类型 | 通常包含的后端 |
|---|---|---|
| `baseline-macos-mlx.json` | Apple Silicon macOS | `cpu`, `ggml_cpu`, `ggml_metal`, `mlx` |
| `baseline-linux-cuda.json` | Linux + NVIDIA GPU | `cpu`, `ggml_cpu`, `ggml_cuda`, `cuda` |
| `baseline-windows-cuda.json` | Windows + NVIDIA GPU | `cpu`, `ggml_cpu`, `ggml_cuda`, `cuda` |

可用 `--baseline <path>` 指向某个特定文件，这对和其他提交捕获的运行做 A/B 对比很有用。

## 存储内容

每条记录只保存回归检测所需的最小数据，而不是完整 per-cell JSON。这样提交的文件
较小，也便于人工阅读：

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

只有同时通过运行时与正确性门槛的 cell 才会写入 baseline。捕获 baseline 时本来
就 flaky 或失败的 cell 不会被固化为“期望状态”。

## 如何检测回归

当前运行中的每个 cell 会按 `case_id` 与 baseline 条目匹配：

| 条件 | 分类 | 是否阻断 `--fail-on-regression` |
|---|---|:-:|
| Baseline `ok=true`，当前 `ok=false`（崩溃、超时、解析失败） | 新的运行时失败 | 是 |
| Baseline `ok=true`，当前 `correctness_ok=false`（输出检查失败） | 新的正确性失败 | 是 |
| 两边 `ok=true`，当前 `decode_tps`（或 prefill-only 的 `prefill_tps`）下降超过阈值 | 吞吐回归 | 是 |
| Baseline `ok=false`，当前 `ok=true` | 改进 | 否 |
| 当前 cell 没有 baseline 条目 | 未跟踪 | 否 |

默认阈值是 **10%** decode TPS 下降。可用 `--regression-threshold-pct N`
或 `matrix-config.json` 中的 `regression_threshold_pct` 覆盖。

## 更新 baseline

Baseline 会随着代码合理变快、变慢或新增 cell 而过期。请有意识地刷新：

### 本地

```bash
dotnet TensorSharp.TestMatrix/bin/TensorSharp.TestMatrix.dll \
  --update-baseline \
  --model-dir /Users/ZhongkaiFu/work/model
```

运行器会就地写入当前主机文件。请检查 diff，并与导致这些数字变化的代码一起提交。

### 从 CI

在 **Actions -> Test Matrix -> Run workflow** UI 中触发 workflow，并设置
`update_baseline = true`。每台主机 runner 会把刷新的 `baseline-<host>.json`
作为 artifact 上传；维护者下载、检查，并开 PR 替换提交中的文件。

这个间接流程是刻意的：baseline 漂移应该通过代码评审，而不是 CI 自动副作用。

## 过期 baseline / 未跟踪 cell

新增模型、后端、feature 或环境变量 sweep 会在下一次运行中产生“untracked” cell：
它们没有 baseline 可比，不会被 gating。新 cell 稳定后，用 `--update-baseline`
重新捕获，让它们进入回归检测。

从矩阵中移除模型会让 baseline 文件里留下旧条目。比较时这些条目会被静默忽略，
没有危害；下次重新生成 baseline 时会被清理掉。
