# Fresh Linux native qualification with pinned upstream libraries

`prepare.py` snapshots every qualified native input, including `.inc`, Objective-C,
Metal, CMake, and test files. It checks the complete current input list and raw
bytes against a successful native build, checks recorded upstream source bytes,
and verifies the four Linux static libraries against the actual successful Linux
build record. Preparation never starts WSL, configures CMake, or compiles code.
Linux symlinks that Windows cannot read are explicitly listed as pending; the
Linux build entry point must check every upstream byte before configuring.

Run preparation with explicit paths; the output must be new and outside both the
repository and the immutable dependency reference:

```powershell
python eng/validation/PinnedLinuxNative/prepare.py `
  --repo C:/Works/TensorSharp `
  --reference E:/TensorSharp-validation-cache/unified-native-pipeline-20261011/linux-v13-frozen `
  --native-summary D:/TensorSharp-validation-cache/master-merge-20261013/native-v15/build-summary.json `
  --output E:/TensorSharp-validation-cache/master-merge-20261013/native-linux-v15
```

Only after the model benchmark/CPU/GPU owner releases its window, execute the new
candidate's `build.py` under WSL. The driver imports the original static libraries;
it defines no upstream compilation target. All TensorSharp production, hook, and
component objects are rebuilt in fresh directories. Formal native sources and the
old source/build/binaries stay untouched. Generated CMake differs from formal CMake
only in upstream consumption and paths; this transformation is recorded separately.

```powershell
wsl -d Ubuntu -- python3 /mnt/e/TensorSharp-validation-cache/master-merge-20261013/native-linux-v15/build.py
```

The pinned configuration is GCC/G++ 12, CUDA 12.6, `86-real`, Release, Vulkan/NCCL
off. After CMake configures and before compiling, every owned compiler command is
compared with the actual reference commands. Output/source paths are normalized;
include root order is compared as a set, but `-I` versus `-isystem`, backend macros,
optimization, architecture, and precision arguments must match. Any source-list or
argument change fails closed. Upstream source/library/build-record identities and
current native source bookends are checked again at completion.

The WSL virtual disk may live on C even when candidates are on E. Temporary files,
XDG and compiler caches therefore go inside the D/E candidate. Compilation has two
jobs, an 8 GiB per-process address-space bound, and monitored memory/disk guards.
The build starts no inference or GPU execution and does not inherit earlier native
test qualification. New CPU, actual CUDA, and GLM output/budget matrices must run
separately against the new production/hooks binaries. A build report is not a model
quality, performance, Apple, or TP/MTP acceptance report.

`file_fixture_followup.py` qualifies the portable registered-file test bridge
without replacing the production library or the immutable native build. It allows
only the expert-cache fixture and its CMake test registration to differ from the
qualified Native16 sources. It verifies every other source and dependency byte,
reads the actual Ninja commands, rebuilds both owned fixture/cache objects with
the same flags in a new directory, and links only a new test executable. Both
CUDA cases must execute exact file extents, quota refusal, interrupted-read
recovery, release, and K10 N3 prefill microgroups; exit zero without those markers
does not pass. `--plan` performs read-only provenance checks without compiling.

```powershell
wsl -d Ubuntu -- python3 -B /mnt/c/Works/TensorSharp/eng/validation/PinnedLinuxNative/file_fixture_followup.py `
  --native-root /mnt/e/TensorSharp-validation-cache/master-merge-20261013/native-linux-v16-attempt2 `
  --repo /mnt/c/Works/TensorSharp `
  --output /mnt/e/TensorSharp-validation-cache/master-merge-20261013/native-linux-v16-file-followup-v1
```

The GLM runner in `../linux-native-qualification.py` requires N3 evidence for all
cases and primary-departure N2 evidence for the explicitly requested selected
expert branch. `--only-case` records subset coverage and cannot certify the full
matrix. `../linux-native-revalidate.py` checks completed GLM32 evidence after a
parser correction, preserves the original receipt, and explicitly records that
no new native execution or native fix is claimed.
