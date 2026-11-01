# Local hardware-aware regression

`python eng/run-local-regression.py` is the local orchestration entry point. It
requires Python 3.10+ and the standard library, runs one model/process at a time,
and never uploads evidence. Planning is the default. Existing `TensorSharp.TestMatrix`,
the release validation tools, `ForcedLogitProbe`, and test projects remain useful;
their scope and recipes are inventoried in
[`config/regression/catalog.json`](../config/regression/catalog.json) and the
[coverage audit](local-regression-coverage.md).

This is a bounded regression suite, not a claim that every architecture, hardware,
quantization, context length, or feature combination has passed. An executable
adapter, a supported recipe, and an actual passing result are different things.
The catalog explicitly records all three. Image/video generation, speculative
variants, agent execution, distributed/multi-GPU paths, and missing model-family
presets retain explicit gaps where the unified runner has no validated adapter.

## 本地快速开始

先检查配置覆盖和本机计划，不下载权重、不初始化 GPU：

```powershell
python eng/run-local-regression.py --mode audit
python eng/run-local-regression.py --mode plan --model-root C:\work\model
```

已有模型放在不同目录时，明确提供查找根目录。仅按唯一文件名匹配，分片必须同目录；
真正执行前会校验文件大小、完整性及配置中的 SHA-256。不会覆盖或删除已有模型。

```powershell
python eng/run-local-regression.py --mode plan --model-root .\models --existing-root C:\work\model
```

默认离线计划无法知道缺失权重大小时会标记跳过。允许只查询已固定来源的 HTTP
响应头以完善计划：

```powershell
python eng/run-local-regression.py --mode plan --probe-network --allow-download --max-download-gib 20
```

上面的 `plan` 不下载载荷；`--allow-download` 仅用于展示下载允许时的计划。
完整执行需显式指定 `--mode run`。模型按顺序准备，下载总量受上限控制：

```powershell
python eng/run-local-regression.py --mode run --models embedding-minilm --backend ggml_cpu --server .\TensorSharp.Server.Host\bin\TensorSharp.Server.Host.dll --model-root C:\work\model --allow-download --max-download-gib 1
```

在已经部署对应 CLI/server/probe 的机器上，可遍历所有配置和已适配场景：

```powershell
python eng/run-local-regression.py --mode run --backend auto --allow-download --max-download-gib 40 --model-root C:\work\model --include-suites
```

`auto` 在有本机能力证据且资源预算满足的 backend 中选择一个，报告记录选择理由。
`--backend ggml_cuda` 等明确请求不会回落；`--backend all` 才会逐一测试配置声明的
所有 backend。`--models` 和 `--scenarios` 接受逗号分隔的 ID 通配符。

## Deploy binaries and test projects

Use a separate checkout and its own `bin`/`obj` outputs. Read
[DEVELOPMENT.md](../DEVELOPMENT.md) for the actual native prerequisites of the
chosen backend. The orchestrator does not install a toolkit, build hosts, or
modify another checkout implicitly. On a machine whose build resources are available:

```sh
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release
dotnet build eng/ForcedLogitProbe/ForcedLogitProbe.csproj -c Release
python eng/run-local-regression.py --mode plan
```

Pass `--cli`, `--server`, and `--logit-probe` when using release or isolated build
outputs. DLLs are launched through local `dotnet`. Host build/native dependencies
are not proven by the existence of a DLL. `--runtime-dir` can point at a native
build directory containing `GgmlOps` and `CMakeCache.txt` so CUDA/Metal capability
is distinguishable from a CPU-only wrapper.

`--include-suites` adds existing InferenceWeb, Knowledge, Studio and native/HTTP
client contract tests. By default these require prebuilt Release assemblies and
use `--no-build --no-restore`. `--allow-builds` explicitly permits their build and
restore steps; this can fetch NuGet/native dependencies and is a heavy operation.
The InferenceWeb lane excludes benchmark, real-model, CUDA and MLX requirements;
their separate coverage remains in the matrix, not inferred from this lane.
TRX counters must show executed passing tests, and skipped tests are disclosed.

## Budget and download policy

- The short text/embedding default estimates host RAM as `1.25 * weight bytes +
  2 GiB`, reserves at least 4 GiB or 20% of physical RAM, and charges GPU residency
  separately with another 2 GiB free-VRAM reserve. It does not sum several GPUs'
  VRAM or assume that disk-paged experts fit. These estimates are conservative
  planning gates, not measured peak usage or a guarantee against OOM.
- Unknown model sizes, RAM, VRAM, native runtime capability, missing fixtures and
  unavailable executables are recorded, not treated as successes. A plan only
  reads local file metadata; model hashes are computed during actual preparation.
- `--ram-budget-gib`, `--vram-budget-gib`, `--ram-required-gib`, and
  `--vram-required-gib` are explicit machine/measurement overrides. They do not
  enable partial offload. Use the separate documented architecture-specific
  recipe for paging/offload; do not lower estimates to make an OOM disappear.
  `--gpu-index` selects a single CUDA device, recorded with the run.
- `--assume-backend-available` permits launch when a deployment lacks build
  metadata. It never bypasses runtime backend assertions or source support.
- `--max-models`, `--max-download-gib` (default 20), `--disk-headroom-gib`
  (default 5), and `--timeout-seconds` bound work. The same download policy
  instance tracks transferred bytes across models; model selection is sequential.
- Automatic sources must be existing preset URLs using HTTPS Hugging Face,
  immutable 40-hex revisions and a SHA-256. Redirect handling restricts protocols
  and does not send credentials to unrelated hosts. Missing/gated credentials or
  license acceptance are reported; the runner never accepts an agreement.
  If already authorized, set `HF_TOKEN` in the process environment; tokens are
  never accepted on the command line or copied to reports.
- Downloads stay under `--model-root`, use identity-bound `.part` files, validate
  resumed ranges and final sizes/hashes, and atomically install complete files.
  Incomplete or incorrect existing files fail clearly and are not overwritten.
  Ctrl+C retains resumable partial data and terminates only owned child processes.

## Executed assertions and safe fixtures

CLI adapters assert task-specific text, positive prefill/decode metrics, multi-turn
history, tool name/arguments, or media-specific expected content. Server adapters
exercise chat, SSE completion, cancellation recovery, reset/cache, JSON schema,
tool contracts, and embeddings (finite dimensions, repeatability and semantic
similarity). Cache tests require an observed cache hit before claiming cache
coverage. `ForcedLogitProbe` replays fixed token histories, checks all finite rows,
completion metadata, identical top-1 and the existing numerical gates; it retains
first-divergence diagnostics. `forced_logits_sequence` uses 32/512-token prompts
and 32 teacher-forced decode steps for short-short, long-long, short-long-short,
and long-short orders, comparing every reset case against a separate fresh-process
baseline with matching prompt/forced-history hashes. This detects state that
depends on a previous input length even when equal-length repeats are stable.
Repeatability does not establish agreement with an
independent Transformers/reference implementation.

Media understanding needs the documented public/synthetic inputs from
`eng/make-test-media.sh` supplied through `--media-dir`. Missing media never
causes a pass. These inputs have known content and assertions; arbitrary private
files are not scanned or copied into reports. Model-generated tool calls are
validated as data and never executed. Agent configs are projected to an allowlist
of model options; host shell/network/install permissions are not inherited.

Host processes receive a scrubbed environment, fixed sampling and short context,
artifact-local caches, loopback-only HTTP binding and bounded log capture. A
request to another model/backend, nonfinite logits/embeddings, incorrect output,
truncated required evidence, timeout, OOM or crash fails the relevant case.

## Reports, exit codes and reproduction

Each invocation uses a new directory under `artifacts/local-regression/` (or an
empty `--output`). It writes `report.json`, `report.md`, `junit.xml`, sanitized
per-case logs/configs/inputs and `diagnostics.zip`. Report identity includes source
commit/dirty state, config/catalog hashes, binary hashes, model source/revision
and SHA-256, hardware/toolchain, requested/actual backend, budgets, sampling,
commands, stage, timings, assertions and reproduction arguments.

Only allowlisted text diagnostics enter the ZIP, with credential redaction and
size limits. No weights, process memory, or entire environment dumps are bundled.
Large fixed-history float traces remain locally beside the case and are not in
the ZIP; they may be inspected separately. Review diagnostics before sharing.
The tool does not publish them.

| Status | Meaning |
|---|---|
| `pass` | Actual execution satisfied its assertions |
| `fail` | Preparation or executed assertion failed |
| `skip` | Selected case could not run because a prerequisite or budget was absent |
| `unsupported` | Explicit source/config support limitation |
| `untested` | Planned, recipe-only, incomplete, or unobserved behavior |

| Exit | Meaning |
|---|---|
| 0 | Audit/plan completed, or executed tests passed within the stated scope |
| 1 | Preparation/assertion failure |
| 2 | Invalid invocation/configuration or harness error |
| 3 | No executable test passed, or `--strict` found selected skip/untested cases |
| 130 | Cancelled; owned processes stopped and evidence retained |

For CI/local acceptance use `--strict` and select a deliberately bounded scenario
matrix. A broad run containing known recipes cannot honestly satisfy strict full
coverage. Re-run a failed case using its `reproduce` argv in `report.json`; reruns
always create fresh evidence and do not import old passes.

## Harness verification without models

```sh
python -m unittest discover -s eng/regression/tests -p "test_*.py" -v
python eng/run-local-regression.py --mode audit
python eng/run-local-regression.py --mode plan --backend all
```

Harness tests use tiny synthetic files, subprocesses and loopback HTTP fixtures.
They cover partial/range/download/checksum failures, budgets, cancellation,
timeouts, backend fallback, incorrect outputs, nonfinite numbers, unsupported
configs and report semantics. They do not certify real inference on unrun
Windows/Linux/macOS or unavailable devices. Generated reports and logs stay in
ignored `artifacts/`, in accordance with `AGENTS.md`.
