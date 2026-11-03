# Original W8 C# Driver probe

This independent probe exercises `CudaW8G32HeadModule`; it does not connect a model,
change Q35 dispatch, or measure serving quality. Its default plan and CPU self-test
do not initialize CUDA. Actual execution is Linux64 little-endian and ROOT-only.

The input contract is the frozen W8 ELF R2 `strict/run.json` and
`strict/full-array-inventory.json`, plus the original `data-strict` directory.
Only these eight real-shape directories are needed:
`r248320-k5120-t{1,7,8,14,16,28,32,33}-p3/`. Each needs `input.f32`,
`input.bf16`, `official.bf16`, `reference.f64`, and `sumAbs.f64`. All 40 files
are verified against the pinned inventory before use and rehashed after execution.
The original C++ bench SHA is recorded; code/scale generation reproduces its
pattern3 using integer hashing, signed int8 code bytes and original half scales.
This is synthetic data, not a newly read Ignis model artifact.

All T=1..33 are tested. Eight use their direct original ELF result. The remaining
25 use independent-token prefixes: 2..6→7, 9..13→14, 15→16, 17..27→28,
29..31→32. The original production source has the same per-token K order and
eight K warps for T<=32. Tile-token count changes independent accumulator slots;
Direct versus Shared scale access loads the same F16 bits. T33 has four K warps
and uses its separate direct reference. The probe checks all existing T<=32
input/output/FP64/sumAbs prefixes against T32 before testing new specializations.
These are 8 original executed cases plus 25 source-supported prefix references,
not 33 independently executed old ELF specializations. Any new bit difference
fails; this coverage rule never changes the numerical bound.

Each new T has two complete F32→BF16 cast + original W8 launches and two direct
BF16-input launches, alternating poison. All BF16 output bytes must equal the
frozen official output; A/A, both entry paths, the full original fixed FP64 bound,
input immutability, exact BF16 workspace bytes, allocation offset guards and
actual driver allocation ownership are checked. The module constructor requires
all 34 entries and checks actual `cuFuncGetParamInfo` offsets/sizes plus the
extra-parameter rejection. Actual registers/local memory/occupancy are recorded;
nonzero spills are not relabeled zero. A tail-corruption negative control runs
for every T. Wrong current context, disposed module and disposed buffer controls
must reject. Outstanding work is synchronized before buffer/module disposal.

Each successful output is saved as `T<n>.bf16`. The most recent pass is persisted
before its numeric comparison, so failed bytes survive; successful temporary
pass copies are removed. A full GPU weight readback SHA must match the generated
host payload. Reference metadata, all input files, backend/probe assemblies and
native artifacts must retain their initial SHA after execution. Incomplete
attempts retain `result.json`; they are never numerical passes.

## Build and offline checks

Use a fresh managed output. The native module has already been built separately;
this project performs no NVCC build. The command deliberately makes existing backend targets use their committed PTX
bootstrap without recompiling unrelated Q35 kernels; those bootstrap files are
not the separately supplied W8 module being tested.
`UseAppHost=false` allows `dotnet Probe.dll` on a noexec SHM mount.

```sh
dotnet build eng/validation/Qwen35W8HeadProbe/Qwen35W8HeadProbe.csproj \
  -c Release -o /dev/shm/w8-probe-managed \
  -p:CudaNvcc=/nonexistent/w8-probe-no-nvcc \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true

dotnet /dev/shm/w8-probe-managed/Qwen35W8HeadProbe.dll --self-test
dotnet /dev/shm/w8-probe-managed/Qwen35W8HeadProbe.dll

dotnet /dev/shm/w8-probe-managed/Qwen35W8HeadProbe.dll --check-references \
  --data /dev/shm/w8-reference/data-strict \
  --inventory /dev/shm/w8-reference/full-array-inventory.json \
  --original-run /dev/shm/w8-reference/run.json
```

## ROOT actual execution

The separately built module requires PTX, matching fatbin and its SHA sidecar
beside the actual native build manifest. The originally successful native manifest
is `9b487cf101ef03d8397152e2f84758c6963b1bad2bf5297d54891de13fa1cb15`;
all 33 head specializations and the cast entry were compiled. Use the actual
freshly built backend DLL SHA, not a historical assembly hash. Parent controllers
must provide the exclusive GPU lease, timeout/owned-process cleanup and unchanged
source/provider/runtime checks. This standalone executable does not acquire a VM
lease or kill other processes.

```sh
dotnet /dev/shm/w8-probe-managed/Qwen35W8HeadProbe.dll --execute \
  --data /dev/shm/w8-reference/data-strict \
  --inventory /dev/shm/w8-reference/full-array-inventory.json \
  --original-run /dev/shm/w8-reference/run.json \
  --native-manifest /dev/shm/ts-w8-native-component-nvcc-r1/source/artifacts/w8-native-built/manifest.json \
  --native-manifest-sha 9b487cf101ef03d8397152e2f84758c6963b1bad2bf5297d54891de13fa1cb15 \
  --backend-sha ACTUAL_FRESH_BACKEND_DLL_SHA256 \
  --out /dev/shm/w8-csharp-probe-fresh-output
```

Allow about 1.4 GiB GPU memory, 2.2 GiB host working memory, 625 MB reference
files and 279 MB successful outputs. No large weight file is written. Output is
required to be a fresh normalized nonalias SHM directory; failed cases may retain
up to four additional pass outputs. Event intervals are single-observation
launch diagnostics, never an ABBA performance comparison. Model quality and
performance qualification remain `NOT_RUN`, even when this driver gate passes.
