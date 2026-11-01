# Local hardware-aware regression

`python eng/run-local-regression.py` is the local orchestration entry point. It
requires Python 3.10+ and the standard library, prepares models and runs scenario
cells sequentially, and never uploads evidence. A scenario may own both server
and client processes. Planning is the default. Existing `TensorSharp.TestMatrix`,
the release validation tools, `ForcedLogitProbe`, and test projects remain useful;
their scope and recipes are inventoried in
[`config/regression/catalog.json`](../config/regression/catalog.json) and the
[coverage audit](local-regression-coverage.md).

This is a bounded regression suite, not a claim that every architecture, hardware,
quantization, context length, or feature combination has passed. An executable
adapter, a typed execution profile, and an actual passing result are different things.
The catalog records source support and execution requirements; execution reports record
actual outcomes. It inventories 67 config documents: 50 standalone presets,
16 LoRA plugins and one MTP overlay. Its 44 scenario definitions comprise
25 default adapters, 14 profile-driven executors and five global suite categories.
These are coverage declarations, not counts of qualified models or passing tests.
Profile executors require explicit model roles, local fixtures, independent
oracles and resource declarations. Unrun hardware, unsupported submodes and
broader quality or concurrency claims remain visible gaps. A
[Chinese deployment guide](local-regression.zh-CN.md) is also available.

## Start with an audit and a plan

Audit config coverage and inspect local eligibility without loading a model:

```powershell
python eng/run-local-regression.py --mode audit
python eng/run-local-regression.py --mode plan --model-root C:\work\model
```

Keep existing weights in place and select a separate cache for new files.
Discovery uses unique filenames, with shards in the same directory; actual
execution verifies complete file sizes and SHA-256 identities before loading them:

```powershell
python eng/run-local-regression.py --mode plan --model-root .\models --existing-root C:\work\model
```

Optional metadata probes resolve unknown file sizes from the configured sources.
Enable those probes explicitly when needed:

```powershell
python eng/run-local-regression.py --mode plan --probe-network --allow-download --max-download-gib 20
```

`plan` never downloads weight payloads. `--allow-download` authorizes downloads
only in `--mode run`, within the transfer budget. Start execution with a small
selected model and a deployed host:

```powershell
python eng/run-local-regression.py --mode run --models embedding-minilm --backend ggml_cpu --server .\TensorSharp.Server.Host\bin\TensorSharp.Server.Host.dll --model-root C:\work\model --allow-download --max-download-gib 1
```

After deploying the CLI, server, probe and selected test assemblies, a broader
invocation can use the same cache:

```powershell
python eng/run-local-regression.py --mode run --backend auto --allow-download --max-download-gib 40 --model-root C:\work\model --include-suites
```

For default adapters, `auto` considers source support, available runtimes and
selected scenario compatibility. An explicit `--backend ggml_cuda` retains that
requested identity; fallback cannot count as CUDA coverage. `--backend all`
plans the declared backends separately. `--models` and `--scenarios` accept
comma-separated catalog ID patterns. Profile-backed scenarios additionally need
the [typed profile file](#typed-scenario-profiles) described below; a broad run
does not invent its fixtures or turn missing profiles into passes.

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
is distinguishable from a CPU-only wrapper. This option inventories files only;
it does not copy native libraries or change the OS/.NET library search path.
The logit probe must have its matching prebuilt GGML library deployed beside its
DLL; its [build target and override options](../eng/ForcedLogitProbe/README.md)
copy from an explicitly selected native build without compiling GGML.

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
CUDA kernel outputs must also be deployed beforehand. Ordinary project targets
may copy existing GGML/MLX libraries, but do not compile them in this mode. Finish
the native library build before a managed build copies it; on Windows an active
linker can hold the source DLL open and make the copy fail.
Existing PTX/fatbin/identity sidecars are retained across SDK incremental cleanup;
`Clean` and `Rebuild` are rejected while the CUDA skip flag is enabled.
The suite scopes runtime execution with `TS_TEST_GGML_BACKEND=cpu` instead of
changing CUDA/Vulkan build flags. Its test filter also excludes `Cuda`, `Mlx`,
`GgmlCuda`, `GgmlMetal` and `GgmlVulkan` requirements while retaining `GgmlCpu`;
test discovery may still query GPU driver availability. Existing native libraries
must be prepared separately in an isolated build directory following `DEVELOPMENT.md`. Native
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
Explicit exclusions also block the historical Wan oracle class and UMT5 tokenizer
fixture in older prebuilt assemblies, whose hardcoded local paths bypassed model
traits. Reports list these excluded fixtures as untested. TRX counters must show
executed passing tests, and skipped tests are disclosed. The `--include-suites`
and `--native-test-build-dir` switches select global lanes independently of the
per-model `--models` and `--scenarios` selectors.

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
- `retrieval_vector_cpu` loads two independent MiniLM CPU encoders from the same
  verified file. Its default RAM estimate is `2 * (1.25 * weight bytes + 2 GiB)`;
  disk and downloads count the file once. An explicit `--ram-required-gib` is
  the caller's combined peak estimate for both encoders and their workspace.
  The report records this residency policy; it does not claim measured RAM.
- Unknown model sizes, RAM, VRAM, native runtime capability, missing fixtures and
  unavailable executables are recorded, not treated as successes. Planning reads
  model file metadata without reading or hashing model payloads; model hashes are
  computed during actual preparation. Deployment binaries and runtime metadata
  are inventoried separately and may be hashed during planning.
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

### Explicit context limit

`--context-tokens` accepts integers from **1 to 2048**, with **2048** as the
unchanged default. It sets `MAX_CONTEXT` explicitly in owned model processes;
an ambient `MAX_CONTEXT` is not inherited. The report records the value and
environment policy, and reproduction commands retain the option. There is no
automatic context reduction and no corresponding reduction in RAM/VRAM estimates.
CPU MoE still requires both explicit peak-memory requirements, established for
the selected context and workload.

Insufficient context produces a `skip` before model preparation or downloads.
Prompts, fixed token histories, decode steps and numerical gates remain unchanged:

| Scenario | Minimum context tokens |
|---|---:|
| `cli_text`, using the built-in bounded arithmetic prompt | 256 |
| `cli_prefill` (128 prefill, no decode) | 128 |
| `cli_decode` (32 prefill, 16 decode) | 48 |
| `forced_logits` (128-token prefill, 16 logit rows: initial prefill plus 15 decode forwards) | 144 |
| `forced_logits_sequence` (512-token long control, 32 logit rows: initial prefill plus 31 decode forwards, conservative margin) | 576 |
| Other model adapters or a custom `cli_text` prompt | 2048 |

For example, `--context-tokens 256 --scenarios cli_text,forced_logits` selects
bounded checks without shrinking their fixtures. A sequence-order test selected
with that context is reported as skipped, not replaced with a shorter sequence.

## Typed scenario profiles

`--scenario-profiles PATH` supplies explicit inputs for the 14 profile-driven
scenarios in the [execution matrix](local-regression-coverage.md#profile-driven-execution-matrix).
These scenarios run automatically once their typed profile, verified files,
executables and budgets are available. The runner never manufactures a passing
oracle from the candidate's own output. The common
[JSON Schema](../config/regression/scenario-profiles.schema.json) describes the
envelope; [the loader](../eng/regression/scenario_profiles.py) and each executor's
`validate_profile` function also enforce scenario-specific fields and limits.
Passing the common schema alone is insufficient.

Use a UTF-8 JSON document with `schema_version: 1` and a `profiles` array. The
loader also accepts the repository's JSONC comments and trailing commas. Unknown
common fields and unknown executor parameters are rejected. Documents are limited
to 1 MiB and 128 profiles. Each profile has the following contract:

| Field | Meaning |
|---|---|
| `id` | Unique safe identifier; the result ID is `profile::<id>` |
| `scenario` | Exact profile-backed scenario ID |
| `preset` | Exact catalog preset ID for the implicit primary model |
| `backend` | One exact backend ID; profiles do not substitute another backend |
| `models` | Executor-defined companion role to catalog preset ID, such as `embedding`; not model URLs or shell commands |
| `fixtures` | Named local files, each with `path` and a 64-hex `sha256`; relative paths resolve against the profile document directory |
| `parameters`, `expected` | Strict executor-specific inputs and independent semantic/numerical oracles |
| `resources` | Explicit positive, finite total `ram_required_gib`; `vram_required_gib` is additionally required for GPU backends; optional `per_gpu_required_gib` and `scratch_gib` |

Omit unused CPU VRAM and scratch fields instead of setting them to zero. Model
roles reference existing verified catalog sources. The runner retains required
weights and downloads optional companions only when the executor needs that
option, such as a media projector; optional draft/media weights are not fetched
for unrelated scenarios. Identical resolved files are prepared once even when
multiple roles use them. Every concurrently resident instance still counts in
the memory declaration. `native_unit` is model-free: its `preset` is only a
selection/classification key, and it neither resolves nor downloads model weights.

Fixture paths must be local, with no URL or UNC path. Files must be nonempty and
at most 512 MiB; executor-specific limits are smaller. For example, the Jev media
executor admits bounded PNG, MP4 and WebM fixtures up to 2 MiB, with image/video
dimensions at most 512 per side and video at most 10 seconds/32 frames. Valid
unhandled image encodings remain untested. Audio/ASR is outside that executor.
Fixture hashes are verified before preparation and rechecked after downloads.
Missing fixtures or tools remain explicit prerequisites; they never count as
passes. Use public or synthetic fixtures approved for this task. Hashing pins
identity but does not establish provenance, licensing or oracle correctness.

Memory declarations cover the combined peak of the primary model, all resident
companions/roles, staging buffers and workspace. They are caller estimates, not
measured guarantees or an implicit offload policy. Profile runs currently use
full residency and do not combine with `--cpu-moe`. Machine RAM/VRAM budgets and
headroom still apply. `per_gpu_required_gib` uses physical device indices and
checks each device's **free** VRAM independently; two GPUs' memory is never pooled
to admit one oversized allocation. Multi-GPU profiles must declare each selected
device. Remote distributed profiles additionally require explicit authorized
Linux nodes, preinstalled matching assets, pinned manifests, SSH authentication
and known hosts; they do not discover, deploy to or download models on machines.

Every distributed node must also provide a unique, lowercase
`expected_machine_id_sha256`. On each already authorized Linux node, the operator
can obtain this pin with the following read-only command; the runner does not
discover nodes or collect pins on the operator's behalf:

```sh
sha256sum /etc/machine-id
```

Copy the digest of the **exact raw file bytes, including any trailing newline**;
do not hash trimmed text. The remote worker checks this pin before creating its
run directory, inspecting resource budgets, hashing model files or launching a
probe/anchor process. A mismatch is a hard failure before model launch. Duplicate expected
pins are rejected during profile validation, and the returned identity is checked
again against the pin. A machine ID identifies an OS instance, not independent
physical hardware: containers or VMs can share host resources. The operator must
provide separate valid allocations and budgets for every selected node; unique
pins alone do not prove resource isolation.

`scratch_gib` reserves additional artifact/workspace capacity on `--output`'s
volume. Download remainder is charged to the model/cache volume. When both paths
share a volume, the planner checks their combined demand; otherwise each volume
must satisfy its own demand and disk headroom. Scenario-specific logit/media
guards still apply. RAM, free VRAM, per-device budgets, artifact space and fixture
identity are checked again after model preparation, before execution. A changed
resource situation can stop launch while retaining verified cached weights.

`--models` matches the profile's primary `preset`, `--scenarios` matches its
scenario, and an explicit `--backend` list filters its declared backend. `auto`
and `all` retain a profile's exact declared backend; they do not rewrite it.
There is no profile-ID selector: put only the desired matching profiles in the
supplied file when narrowing further. For scenarios bound to catalog presets,
each selected preset/backend combination without a matching selected profile
retains an `untested` row. A profile for one model or device cannot hide another
selected combination's gap; an unresolved `auto` request stays labeled `auto`.
Only profile classes without preset bindings use a single
`catalog::<scenario-id>`, `scope=catalog`, `status=untested` row when no profile
is supplied, without inventing a model/backend association.
Global `--include-suites` and native CTest selection remain independent.

### Example: a synthetic grounded retrieval profile

Run the following PowerShell in the checkout to create an original document and
its real SHA-256. The primary Qwen chat model and the MiniLM CPU embedding sidecar
are separate resident models. The declared 8 GiB is an illustrative combined
planning estimate; establish a suitable peak requirement on the target machine.
This example exercises grounded answers, source URLs and collection isolation;
it does not claim OCR, reranking or tenant authorization coverage.

```powershell
$profileDir = Join-Path (Get-Location) 'artifacts\profiles\retrieval-demo'
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $profileDir | Out-Null
$fixturePath = Join-Path $profileDir 'document.txt'
$profileFile = Join-Path $profileDir 'retrieval.json'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$document = 'The designated maintenance code for LANTERN-STATION is COBALT-7319.'
[System.IO.File]::WriteAllText($fixturePath, $document, $utf8)
$sha256 = [System.Security.Cryptography.SHA256]::Create()
try {
    $documentSha = ([BitConverter]::ToString($sha256.ComputeHash(
        [System.IO.File]::ReadAllBytes($fixturePath)))).Replace('-', '').ToLowerInvariant()
} finally {
    $sha256.Dispose()
}
$profile = @{
    schema_version = 1
    profiles = @(@{
        id = 'qwen08-document-retrieval'
        scenario = 'retrieval'
        preset = 'qwen3.5-0.8b'
        backend = 'cpu'
        models = @{ embedding = 'embedding-minilm' }
        fixtures = @{ document = @{ path = 'document.txt'; sha256 = $documentSha } }
        parameters = @{
            embedding_backend = 'cpu'
            question = 'What is the maintenance code for LANTERN-STATION? Cite the source URL.'
            search_query = 'LANTERN-STATION maintenance code'
            max_tokens = 384
        }
        expected = @{ answer_contains = 'COBALT-7319'; source_excerpt = $document }
        resources = @{ ram_required_gib = 8 }
    })
}
[System.IO.File]::WriteAllText($profileFile, ($profile | ConvertTo-Json -Depth 12), $utf8)
```

Plan using an already deployed host; `plan` does not start it:

```powershell
python eng/run-local-regression.py --mode plan --models qwen3.5-0.8b `
  --scenarios retrieval --scenario-profiles $profileFile `
  --model-root D:\TensorSharp-models --existing-root C:\work\model `
  --server .\TensorSharp.Server.Host\bin\TensorSharp.Server.Host.dll
```

For execution, change to `--mode run`, use a fresh `--output` directory and, if
the verified weights are absent, add `--allow-download --max-download-gib 2`.
Add `--strict` to require the selected case to execute and satisfy its assertions.
Reports retain the profile document hash, resolved role/file identities, fixture
hashes, expected/actual evidence and reproduction arguments. Generated media and
large oracle artifacts remain local; no upload or publication occurs.

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
and 32 compared logit rows (one initial prefill row plus 31 teacher-forced decode
forwards) for short-short, long-long, short-long-short,
and long-short orders, comparing every reset case against a separate fresh-process
baseline with matching prompt/forced-history hashes. This detects state that
depends on a previous input length even when equal-length repeats are stable.
Repeatability does not establish agreement with an
independent Transformers/reference implementation.

`client_http_live` uses the prebuilt `TensorSharp.LocalChat` example with an owned
server. It runs both OpenAI SDK and `IChatClient` paths in streaming and response
modes, each requiring exactly `4` then `12` from two conversation turns. Raw
HTTP/SSE controls check identity, normal completion and token accounting. SDK
sampling uses the server's fixed defaults; the example does not send seed or
temperature. This does not capture SDK wire history or qualify per-SDK metrics,
cancellation, deadlines or native-client lifecycle behavior. Provide a separately
built example with `--http-client`, or let the runner discover its Release output:

```powershell
python eng/run-local-regression.py --mode run --models qwen3.5-0.8b `
  --scenarios client_http_live --backend auto --existing-root C:\work\model `
  --http-client examples/dotnet-local-chat/bin/Release/net10.0/TensorSharp.LocalChat.dll
```

`retrieval_vector_cpu` is restricted to `embedding-minilm` and the managed `cpu`
backend. It starts an embedding-only host and CPU knowledge sidecar using the
same verified GGUF, with a fresh artifact-local library and three original
synthetic TXT files. It checks exact weight/prefix and source hashes, Ready
single-chunk metadata, a zero-lexical-overlap paraphrase's top result, finite RRF
scores, selected-collection scope, exact read/source content, rejected malformed
requests and atomic rejected uploads, and document/collection deletion. HTTP
bodies are bounded at 64 KiB. Failed fixture stores remain local diagnostics;
successful runs delete only their API-created fixtures. `--backend auto` prefers
the eligible backend covering the most selected adapters, so MiniLM retrieval
alone or alongside embeddings selects CPU; explicit GPU requests retain an
unsupported adapter cell, without switching backends. This restriction describes
the adapter's qualification scope, not the engine's complete backend support.

```powershell
python eng/run-local-regression.py --mode run --models embedding-minilm `
  --scenarios retrieval_vector_cpu --backend auto --model-root C:\work\model `
  --server .\TensorSharp.Server.Host\bin\TensorSharp.Server.Host.dll
```

The primary CPU backend is API-attested. The independent sidecar's CPU backend
is explicitly configured and source-admitted; its public API exposes the
weight/prefix identity but no backend telemetry. Grounded answers and citations,
reranking, OCR/media, persistence across restart, concurrency and tenant ACLs
remain outside this adapter. The explicit `retrieval` profile adds paired
chat/embedding execution with grounded-answer and source-URL assertions.
Collection IDs constrain retrieval scope and are not authorization credentials.

`jev_decision` runs the repository's 12 labeled text decision fixtures, 36 gold
decisions, fixed-seed replay and malformed requests. `jev_text_attachments` runs
14 pinned text/document fixtures as inline attachments, including office/PDF text
extraction, ordered sources and cross-file questions; it requires positive cache
replay evidence and rejects eight malformed inputs with exact typed errors. Its
five vision/audio-dependent manifest cases are not executed by this text adapter;
bounded image/video fixtures use `jev_media` profiles, while audio/ASR,
uploaded-reference and concurrent-isolation paths remain untested. `speculation_ngram` requires
a prebuilt `--benchmark` AgentTurnBench and a source-admitted family/backend;
it requires actual n-gram verification/acceptance and exact greedy token parity.
It never enables a learned drafter. `image_generation_smoke` runs Qwen base/Turbo
at 128px with its required VAE/text encoder, validates RGB/RGBA and noninterlaced
gray/palette PNG contents and error
responses, and records repeated-seed hashes without declaring semantic quality
or bitwise determinism. Explicit profiles add independent image/media oracles
and learned-head speculation assertions; their availability does not establish
real-model qualification or open-ended quality.

`server_agents` runs two bounded arithmetic reviewers through host coordination
tools, requiring completed child results, exact child/final JSON and a separate
request-level opt-out. Skills, code execution and worker tools remain disabled.
It checks distinct workspace identities, not filesystem confinement; the host
uses its normal per-user `TensorSharp/agent-workspaces` directory. Dependency
scheduling, mutable tools and cancellation propagation remain untested.

Media understanding needs the documented public/synthetic inputs from
`eng/make-test-media.sh` supplied through `--media-dir`. Missing media never
causes a pass. These inputs have known content and assertions; arbitrary private
files are not scanned or copied into reports. The `cli_tools` and `server_tools`
fixtures validate model-generated calls as data without executing them;
`server_agents` executes only its bounded coordination workflow. Agent configs
are projected to an allowlist of model options; host shell/network/install
permissions are not inherited.

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
Each execution records fresh artifact-local temporary and external-fixture
directories. `TMPDIR`, `TMP`, and `TEMP` point to the owned temporary directory;
`TS_MM3_FIXTURE_DIR` points to the separate empty directory. MiniMax executor
tests can generate their small synthetic GGUF without touching user files.
MiniMax DSpark tests that require separately generated `model.gguf` and
`dspark.gguf` remain skipped with code 77; the runner does not load an ambient
fixture or automatically generate one from a real checkpoint.
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
| `unsupported` | Explicit source/config or adapter qualification limitation, with its scope stated |
| `untested` | Planned, missing profile, incomplete, or unobserved behavior |

| Exit | Meaning |
|---|---|
| 0 | Audit/plan completed, or executed tests passed within the stated scope |
| 1 | Preparation/assertion failure |
| 2 | Invalid invocation/configuration or harness error |
| 3 | No executable test passed, or `--strict` found selected skip/untested cases |
| 130 | Cancelled; process cleanup outcome and evidence retained |

For CI/local acceptance use `--strict` and select a deliberately bounded scenario
matrix. A broad run with missing profiles, fixtures or hardware cannot honestly
satisfy strict full coverage. Explicit `unsupported` results remain visible but
do not by themselves trigger strict exit code 3. The 14 profile-backed scenario
IDs have automatic typed execution paths, but need selected profiles to run.
Missing profiles produce per-preset/backend or unbound catalog-scope untested
records as described above.
A run containing only these records exits 3; its plan exits 0 with untested
coverage. Consult the catalog and execution report together when assessing gaps.
Re-run a failed case using its `reproduce` argv in `report.json`; reruns
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
