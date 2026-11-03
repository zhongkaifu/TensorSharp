# Teacher-forced logit probe

This probe compares every vocabulary logit at prompt completion and every decode
step while consuming exactly the reference token history. A process loads one
model instance and calls `ResetKVCache()` before each case. It does not reload
the model between cases.

For GGML execution, deploy the prebuilt native library beside `ForcedLogitProbe.dll`.
The probe's `AfterBuild` deployment target looks under
`TensorSharp.GGML.Native/build-windows` on Windows or `TensorSharp.GGML.Native/build`
on Linux/macOS, choosing `Release/<library>` first and then `<library>` in that
directory. Names are `GgmlOps.dll`, `libGgmlOps.so`, and `libGgmlOps.dylib` respectively.
Override `GgmlNativeBuildDir` to use a separate native build with the intended
CPU/CUDA/Metal backend. The target only copies an existing file; it does not invoke
CMake, nvcc, or a native compiler, and a successful copy does not qualify backend
execution or prove dependent native libraries are installed.

After preparing native assets separately, a managed-only deployment build can use:

```powershell
dotnet build eng/ForcedLogitProbe/ForcedLogitProbe.csproj -c Release `
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -p:TensorSharpSkipCudaPtx=true `
  -p:GgmlNativeBuildDir="D:\TensorSharp-build\ggml" -p:ForcedLogitProbeRequireGgmlNative=true
```

Pass all three skip properties on the command line so they apply to referenced
projects as well. The CUDA skip policy requires its existing prebuilt PTX assets;
see [local regression build guidance](../../docs/local-regression.md). Keep the
complete output and matching dependencies together when relocating the probe.
Without `ForcedLogitProbeRequireGgmlNative=true`, a missing source produces an
explicit warning and leaves any existing output library untouched, allowing
managed-only compilation without claiming GGML availability. With that property,
the missing deployment fails the build.

The runner's `--runtime-dir` inventories native files and build metadata only.
It does **not** copy a DLL or alter the OS/.NET native-library search path. A GGML
library found in another host's output does not make the probe's output runnable.

```sh
dotnet run --project eng/ForcedLogitProbe -- model.gguf artifacts/fresh cpu --cases short.json --steps 32
dotnet run --project eng/ForcedLogitProbe -- model.gguf artifacts/sequence cpu --reference artifacts/fresh --cases repeated.json
```

`short.json` can contain `[{"name":"short","prompt":"A synthetic regression prompt.","target_tokens":32}]`.
`repeated.json` can contain:

```json
[
  {"name":"short_1","reference_case":"short"},
  {"name":"short_2","reference_case":"short"}
]
```

`--cases` selects the candidate schedule even when `--reference` is supplied.
Every output `name` must be unique, including case-insensitive filename aliases.
`reference_case` names a case in the external reference's `cases.json`; omitting
it retains the original same-name comparison. The candidate inherits the exact
recorded prompt and generated token IDs. Explicit conflicting tokens or lengths
are rejected before a forward pass.

Without `--reference`, `reference_case` may name an earlier case in that same
process. This checks repeated/reset execution against an earlier local capture;
it does **not** establish fresh-process equivalence. Forward/self references and
duplicate output names are refused.

The local regression adapter `forced_logits_sequence` creates independent
fresh-process short and long baselines, then evaluates short-short, long-long,
short-long-short and long-short schedules against those exact histories. Those
comparisons can reveal order-dependent differences but do not establish their
cause.

Each case records its canonical reference name and scope, reset operation,
session index, int32 little-endian SHA256 hashes of prompt/forced/input histories,
and the first row outside the numerical gate. Check `complete`, `cases_done`,
`all_finite`, `qualified`, `run_id`, and process exit status together. Reference
and output directories must differ; stale completion records are invalidated
before schedule parsing and model loading.

The probe applies `MoeCpuOffloadConfig.ConfigureFromEnvironment()` before model
creation, matching the CLI and server startup order. `TS_CPU_MOE=1` requests
all-layer host routed-expert execution on supported GGML MoE paths. Metrics
record `model_architecture` and the effective `offload_policy` (enabled, all layers,
layer count, explicit configuration, thread count and configuration source).
This is configuration evidence; backend identity and numerical comparisons must
also pass. The local runner's `--cpu-moe` profile requires explicit peak RAM/VRAM
requirements and limits this initial path to `qwen35`/MoE or `qwen4exp`/MoE on
`ggml_cpu` and `ggml_cuda`; other backend/family combinations are not covered.

Before creating each `.f32` file, the probe checks the exact byte count
`rows * vocabulary_size * sizeof(float)` against free space on the output volume,
plus `--disk-headroom-gib` (default **5 GiB**). The case's `artifact_disk_budget`
records the logit, headroom, available and total required bytes. Insufficient
space leaves incomplete metrics and exits with an error before that file is
created. This is a pre-write check, not an OS disk reservation; another process
can still consume space afterward. Zero headroom is an explicit override.

The schedule contract has a separate model-free executable, linking the same
`ProbeSchedule.cs` implementation without TensorSharp or native dependencies:

```sh
dotnet run --project eng/ForcedLogitProbe/ContractTests/ContractTests.csproj
```

It tests all four orders, identical teacher-forced histories, alias JSON round
trips, duplicate/unsafe names, missing/forward/self references, incomplete or
conflicting histories and portable token hashes. A successful run prints its
assertion count. Building or running this test is separate from model inference.

For CI, use the installed .NET 10 reference pack and the checked-in empty package
source list. This project has no package or model-project dependencies; these
commands cannot fetch native backends or model weights:

```sh
dotnet restore eng/ForcedLogitProbe/ContractTests/ContractTests.csproj --configfile eng/ForcedLogitProbe/ContractTests/NuGet.offline.config --disable-parallel -p:NuGetAudit=false -m:1
dotnet build eng/ForcedLogitProbe/ContractTests/ContractTests.csproj --no-restore -c Release -m:1 -p:UseSharedCompilation=false
dotnet run --project eng/ForcedLogitProbe/ContractTests/ContractTests.csproj --no-restore --no-build -c Release
```

Set `DOTNET_CLI_HOME`, `NUGET_PACKAGES`, `TMP` and `TEMP` to the job's writable
scratch directory when its default home/temp folders are restricted. The parent
probe excludes `ContractTests/**/*.cs`, so its top-level test program is never
compiled into the inference executable.
