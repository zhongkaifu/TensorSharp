# Teacher-forced logit probe

This probe compares every vocabulary logit at prompt completion and every decode
step while consuming exactly the reference token history. A process loads one
model instance and calls `ResetKVCache()` before each case. It does not reload
the model between cases.

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
