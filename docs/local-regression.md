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
variants, broader agent workflows, distributed/multi-GPU paths, and missing model-family
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
use `--no-build --no-restore`. `--allow-builds` explicitly permits **managed-only**
build and restore steps, which can fetch NuGet packages. Every managed suite
`test`, `build` and `restore` command sets `TensorSharpSkipGgmlNative=true`,
`TensorSharpSkipMlxNative=true` and `TensorSharpSkipCudaPtx=true`; it never
configures or compiles native backends. The CUDA flag also suppresses nvcc
detection, PTX/fatbin selection and copying, and committed-baseline refreshes.
It requires prebuilt portable-kernel PTX with valid version/target headers under
the CUDA project's active output `cuda_kernels` directory. Missing assets fail
clearly even if a committed baseline or intermediate build exists elsewhere.
Architecture-specific optional kernels remain subject to runtime capability
checks; these header checks do not qualify CUDA execution. Referencing application
outputs must also be deployed beforehand: this mode does not stage native assets.
Existing PTX/fatbin/identity sidecars are retained across SDK incremental cleanup;
`Clean` and `Rebuild` are rejected while the CUDA skip flag is enabled.
The suite scopes runtime execution with `TS_TEST_GGML_BACKEND=cpu` instead of
changing CUDA/Vulkan build flags. Existing native libraries must be prepared
separately in an isolated build directory following `DEVELOPMENT.md`. Native
CTest likewise requires an explicit `--native-test-build-dir`; `--allow-builds`
does not build it. A managed build exit code alone is never a passing test:
required client assemblies and actual contract/TRX assertion evidence must exist.
Permitted suite builds run as separate steps, followed by `dotnet test --no-build
--no-restore`. Before assertions execute, each suite records its own source state,
project and resulting assembly hashes, plus native deployment candidates including
the conventional `cuda_kernels` directory. These are distinct from host identities
captured before the run, and do not claim independent observation of mapped modules.
The InferenceWeb lane excludes benchmark, real-model, CUDA and MLX requirements;
their separate coverage remains in the matrix, not inferred from this lane.
TRX counters must show executed passing tests, and skipped tests are disclosed.

## Budget and download policy

### Use another drive for models, builds and diagnostics

Model and report locations are independent. A nearly full system drive does not
require moving or deleting existing weights: keep them where they are and select
another drive for new files. The same options work with Linux mount paths.

| Location | Select with | Behavior |
|---|---|---|
| New models and resumable `.part` files | `--model-root D:\TensorSharp-regression-validation\models` | Download and free-space checks use this model/cache volume |
| Existing weights | One or more `--existing-root C:\work\model` options | Read-only discovery by unique filename; execution verifies the selected identity, without copying or deleting weights |
| Reports, logs, caches and logit traces | `--output E:\TensorSharp-regression-results\run-001` | Must be empty; logit artifact headroom is checked on this volume separately |
| Source and build outputs | A separate checkout, then explicit `--cli`, `--server`, `--logit-probe`, `--runtime-dir` paths | `--model-root` does not relocate source, `bin`, `obj` or native builds |
| Process temporary files and package caches | Process-local `TEMP`/`TMP` on Windows; `TMPDIR` on Unix; optionally `NUGET_PACKAGES` | Configure these before starting Python, .NET or build tools; the runner does not move existing caches |

For example, put an independent checkout in
`D:\TensorSharp-regression-validation\source`, then plan using D: for new model
files and artifacts while reusing existing C: weights:

```powershell
$validationRoot = 'D:\TensorSharp-regression-validation'
$checkout = Join-Path $validationRoot 'source'
$modelCache = Join-Path $validationRoot 'models'
$artifactRoot = Join-Path $validationRoot 'artifacts'
$temporaryRoot = Join-Path $validationRoot 'temp'
New-Item -ItemType Directory -Force -Path $modelCache,$artifactRoot,$temporaryRoot | Out-Null
$env:TEMP = $temporaryRoot
$env:TMP = $temporaryRoot
$planOutput = Join-Path $artifactRoot ('plan-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
python "$checkout\eng\run-local-regression.py" --repo $checkout --mode plan `
  --model-root $modelCache --existing-root 'C:\work\model' `
  --output $planOutput --disk-headroom-gib 10
```

To put only diagnostics on E:, change `$artifactRoot` to an existing writable
directory such as `E:\TensorSharp-regression-results` before creating the new
output path. Add another `--existing-root` only for an actual weights directory.
Ambiguous duplicate filenames are not selected automatically; verified model
identity still decides whether a local file is usable.

After the plan is reviewed and the chosen local backend has been built, use a new
output directory for execution. For example, with a server deployed under the
independent D: validation directory:

```powershell
$runOutput = Join-Path $artifactRoot ('embedding-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
python "$checkout\eng\run-local-regression.py" --repo $checkout --mode run `
  --models embedding-minilm --backend ggml_cpu `
  --server "$validationRoot\build\server\TensorSharp.Server.Host.dll" `
  --model-root $modelCache --existing-root 'C:\work\model' `
  --output $runOutput --allow-download --max-download-gib 1 --disk-headroom-gib 10
```

Disk failures identify the relevant location: model preparation reports
`disk_budget` with its model/cache path; logit scenarios record
`artifact_disk_checks` with the artifact path, available bytes, estimate and
headroom. Existing partial bytes already occupy their volume; only the validated
remaining bytes count toward further transfer and additional disk allocation.
The complete model size still counts toward default RAM/VRAM estimates; the
explicit CPU MoE profile below records caller-supplied resident requirements. A roomy model
drive cannot compensate for a full report drive. Small report writes and other
tools can still encounter ordinary filesystem errors, so leave space on every
selected write volume rather than lowering the headroom to conceal exhaustion.

The failure's `reproduce` command preserves the chosen model and existing roots.
When the original run supplied `--output`, it also specifies a fresh sibling
directory on that report volume; it does not reuse the old nonempty report
directory. To execute the same recorded command again after its first replay,
choose another empty output path. Without an explicit output in the original
invocation, a new report is created under the selected repository.

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
  enable partial offload by themselves. The explicit CPU MoE profile below
  connects declared resident requirements to the supported host-expert policy.
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

### Explicit CPU MoE profile

`--cpu-moe` opts into all-layer routed-expert execution on the host. It requires
both `--ram-required-gib` and `--vram-required-gib` as positive finite values
supplied by the operator, including for `ggml_cpu`. Use peak requirements for
the selected model, quantization, context, scenarios and runtime, with workload
headroom. The runner does not derive a smaller resident size from the number of
active experts or from a measurement of another model.

The initial source-verified matrix is deliberately limited:

| Catalog family/layout | Backends | Placement |
|---|---|---|
| `qwen35` / `moe` (including Qwen3.6) | `ggml_cuda`, `ggml_cpu` | All routed experts run on host |
| `qwen4exp` / `moe` (Qwen3.8 Flash Next) | `ggml_cuda`, `ggml_cpu` | Explicit host-expert placement; single device |

Dense variants, other families, direct `cuda`, Metal, Vulkan, MLX and distributed
placements are outside this profile. They receive `unsupported` results without
model preparation. On `ggml_cpu`, experts already execute on the host, so the
flag does not claim VRAM savings and its required VRAM value is recorded but
not charged to a GPU. `auto` chooses only a supported GGML backend for this profile;
the report records each rejected candidate and never relabels a CPU execution as
CUDA coverage.

For example, append these options to the selected model's plan/run invocation,
replacing the two placeholders with explicitly established requirements:

```text
--models qwen3.8-flash-next --backend ggml_cuda --cpu-moe --ram-required-gib <peak-RAM-GiB> --vram-required-gib <peak-VRAM-GiB>
```

The profile sets `TS_CPU_MOE=1` only in owned model processes. It leaves attention,
router and shared-expert execution on the requested GGML CUDA backend. Downloads,
hash verification and disk checks still use the complete model size, even when
host experts are memory mapped. Current RAM/VRAM budgets and reserves still apply;
the flag cannot make an over-budget plan eligible. Page residency and workspace
can change during execution, so declared requirements are not a memory guarantee.

Reports record the requested placement, explicit requirement source, sanitized
environment, source support evidence and separate runtime assertions. Hosts must
confirm the all-layer configuration; the probe records its effective configuration
before comparisons can pass. The requested backend remains independently checked.
Old executables that ignore the profile cannot establish a passing profile result.
Reproduction commands retain the flag and both requirements. Without `--cpu-moe`,
the conservative full-weight planning gate remains unchanged; native model-specific
placement behavior is not inferred from a successful preflight.

## Executed assertions and safe fixtures

CLI adapters assert task-specific text, positive prefill/decode metrics, multi-turn
history, tool name/arguments, or media-specific expected content. Server adapters
exercise chat, SSE completion, cancellation recovery, reset/cache, JSON schema,
tool contracts, and embeddings (finite dimensions, repeatability and semantic
similarity). The session fixture sends the full conversation history required by
`/api/chat`, with `newChat` only at the first request and explicit resets. It checks
A's marker after B starts, then B's marker after A resets. Both adjacent controls
and interleaved followups must show KV reuse beyond their request-correlated
public-prefix boundaries. Primary-only backends, eviction, absent telemetry, or
other missing retention evidence leave this combined scenario untested; correct
answers alone cannot establish retained state. A separate repeated-prefix request
must also report a positive cache hit. The arithmetic reset check covers the fresh
request boundary, not implicit recall of history omitted by the client.
`ForcedLogitProbe` replays fixed token histories, checks all finite rows,
completion metadata, identical top-1 and the existing numerical gates; it retains
first-divergence diagnostics. `forced_logits_sequence` uses 32/512-token prompts
and 32 teacher-forced decode steps for short-short, long-long, short-long-short,
and long-short orders, comparing every reset case against a separate fresh-process
baseline with matching prompt/forced-history hashes. This detects state that
depends on a previous input length even when equal-length repeats are stable.
Repeatability does not establish agreement with an
independent Transformers/reference implementation.

`jev_decision` runs the repository's 12 labeled text decision fixtures, 36 gold
decisions, fixed-seed replay and malformed requests. `jev_text_attachments` runs
14 pinned text/document fixtures as inline attachments, including office/PDF text
extraction, ordered sources and cross-file questions; it requires positive cache
replay evidence and rejects eight malformed inputs with exact typed errors. Its
five vision/audio-dependent manifest cases remain explicitly untested, as do
uploaded-reference and concurrent-isolation paths. `speculation_ngram` requires
a prebuilt `--benchmark` AgentTurnBench and a source-admitted family/backend;
it requires actual n-gram verification/acceptance and exact greedy token parity.
It never enables a learned drafter. `image_generation_smoke` runs Qwen base/Turbo
at 128px with its required VAE/text encoder, validates RGB/RGBA and noninterlaced
gray/palette PNG contents and error
responses, and records repeated-seed hashes without declaring semantic quality
or bitwise determinism. The corresponding broader quality, attachment and
learned-speculation recipes remain untested.

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

Prebuilt native CTest assertions can also run through this entry point:

```sh
python eng/run-local-regression.py --mode run --scenarios native_ctest --native-test-build-dir artifacts/native-regression-build --native-test-backend cpu
```

Configure and build that isolated directory separately using
`-DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON`. The runner checks that its CMake cache
points to this checkout and never configures or builds it implicitly. Tests run
sequentially, with exact selected names and per-test JUnit outcomes. Backend
arguments/names must identify the requested device; ambiguous tests stay
`untested`, other-device tests are `skip`. An optional `--native-test-filter`
regex narrows names further. Fixture dependencies outside the selection are
reported instead of allowing CTest to start another device automatically.
Device builds require the corresponding explicit `--native-test-backend` and
recorded compiler options. Binary/cache hashes identify the deployment; they do
not establish that an old binary was rebuilt from the current commit. Native
assertions do not count as real-model qualification.

Each invocation uses a new directory under `artifacts/local-regression/` (or an
empty `--output`). It writes `report.json`, `report.md`, `junit.xml`, sanitized
per-case logs/configs/inputs and `diagnostics.zip`. Report identity includes source
commit/dirty state, config/catalog hashes, binary hashes, model source/revision
and SHA-256, hardware/toolchain, requested/actual backend, budgets, sampling,
commands, stage, timings, assertions and reproduction arguments.
When using hosts built in another checkout, provide `--engine-repo` to record
that checkout's commit and dirty state separately. This is declared build
provenance; the report also hashes the supplied binaries and does not infer their
source commit from the harness checkout.

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
| 130 | Cancelled; process cleanup outcome and evidence retained |

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
