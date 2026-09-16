# Qwen 3.8 MTP operator fixture under compute-sanitizer with CUDA graphs

Status: **closed, no defect.** The "handled graph-update API errors" the earlier
capture-enabled sanitizer run retained are `cudaGraphExecUpdate` returning
`cudaErrorGraphExecUpdateFailure` (910) inside upstream ggml-cuda, each one
immediately cleared by ggml's own `cudaGetLastError()` and replaced by a fresh
`cudaGraphInstantiate`. The sanitizer counts both the failing call and the
clearing call as "errors"; the executable graph that runs afterwards is correct
(every forward in every run below matches the F64 oracle at atol=rtol=2e-5, and
replays are bitwise identical). Nothing in TensorSharp-owned code
(`ggml_ops_qwen4exp_mtp.cpp`, `ggml_ops_qwen4exp.cpp`, `ggml_ops_dsv4_fused.cu`)
issues or mishandles the call. memcheck, initcheck, synccheck and racecheck report
zero device-side errors with graphs on and off.

This file records exactly what was run so the finding can be closed on evidence.
`graph-update-probe.py` next to it is the driver that reproduces the retained
errors on demand; the operator fixture alone does not reach that code path (see
"Why the fixture run was clean").

## Environment

| item | value |
| --- | --- |
| GPU | NVIDIA A40 (sm_86), driver 570.195.03, `CUDA_VISIBLE_DEVICES=6` |
| CUDA | 12.8 (nvcc V12.8.93); compute-sanitizer 2025.1.0.0 (build 35583870) |
| native library | `/workspace/ts-main/repo/TensorSharp.GGML.Native/build/libGgmlOps.so`, sha256 `48abe2b1623358f1567861c304e355686b00d9cf7c7d6803a6316f570ab6b85d` (Release, CUDA on, unmodified) |
| ggml | `ExternalProjects/ggml` at `456172ec733a135778adcd32d00e576a58232e45` |
| fixture | `eng/tests/qwen4exp-mtp-operator.py` sha256 `c1ebfc19…5427d113`, `fixtures/qwen4exp-mtp/sample.json` sha256 `d6fe83db…d355216c` |
| logs | `/workspace/ts-sanitizer/` on the VM: `<name>.app.log`, `<name>.sanitizer.log`, `<name>.json` (fixture report); `summary*.log` index every run with exit codes |

Common prefix for every command below:

```sh
cd /workspace/ts-main/repo
export PYTHONPATH=/workspace/tensorsharp-no-patch-20260915/fixture-python-deps
export CUDA_VISIBLE_DEVICES=6
export TS_GGML_LOG_DEBUG=1     # surfaces ggml's "CUDA graph warmup complete/reset" lines
S=/usr/local/cuda/bin/compute-sanitizer
LIB=TensorSharp.GGML.Native/build/libGgmlOps.so
FIX="/usr/local/bin/python -B eng/tests/qwen4exp-mtp-operator.py --sample fixtures/qwen4exp-mtp/sample.json --library $LIB --backend CUDA"
```

## 1. The operator fixture itself

`$FIX --capacity C --kv-alignment A --report <fresh.json>`, plain and under each
sanitizer tool, with CUDA graphs enabled (default) and disabled
(`GGML_CUDA_DISABLE_GRAPHS=1`). "captures" is the count of ggml's
`CUDA graph warmup complete` debug lines in that process, i.e. graphs actually
captured and launched; it is only visible in the runs that had
`TS_GGML_LOG_DEBUG=1` (the first batch at 512/4 ran without it and is marked
"n/r", not recorded) and is 0 whenever graphs are disabled. Every debug-logged
run of this fixture with graphs on captured 23 to 29 graphs; the exact number
depends on which freed graph-context addresses the allocator reuses as keys.

| run | capacity/alignment | graphs (captures) | fixture | sanitizer result | exit |
| --- | --- | --- | --- | --- | --- |
| plain | 512/4, 32/4, 512/64, 32/64 | on (23, 23, 23, 23) | PASSED 148 checks each | – | 0 |
| memcheck | 512/4 | on (n/r) | PASSED 148 | `ERROR SUMMARY: 0 errors` | 0 |
| memcheck | 512/4 | off | PASSED 148 | `0 errors` | 0 |
| memcheck `--leak-check full --report-api-errors all` | 512/4 | on (23) | PASSED 148 | `0 errors`, `0 bytes leaked` | 0 |
| memcheck `--leak-check full --report-api-errors all` | 512/4 | off | PASSED 148 | `0 errors`, `0 bytes leaked` | 0 |
| memcheck `--report-api-errors all` | 32/4, 512/64, 32/64 | on (23, 29, 23) | PASSED 148 each | `0 errors` each | 0 |
| memcheck `--report-api-errors all` | 32/4, 512/64, 32/64 | off | PASSED 148 each | `0 errors` each | 0 |
| memcheck `--error-exitcode 3` | 512/4 | on (n/r) and off | PASSED 148 | `0 errors` | 0 |
| initcheck | 512/4, 32/64 | on (n/r, 25) and off | PASSED 148 | `0 errors` | 0 |
| initcheck `--track-unused-memory` | 512/4 | on (23) | PASSED 148 | 25 "Unused memory in allocation" notes: partially used allocations whose sizes (1024, 131072, 8388608 bytes and slices of one pool buffer) match the cuBLAS handle allocations and pool buffers seen in the snapshot leak backtraces; informational, not an access error | 0 |
| synccheck | 512/4, 32/64 | on (n/r, 29) and off | PASSED 148 | `0 errors` | 0 |
| racecheck | 512/4, 32/64 | on (n/r, 23) and off | PASSED 148 | `0 hazards (0 errors, 0 warnings)` | 0 |

`eng/tests/qwen4exp-target-snapshot.py --geometry gdn32` (same library, GPU 6):
plain PASSED 132 checks; initcheck, synccheck and racecheck clean; memcheck
`--leak-check full --report-api-errors all` reports `5 errors`, all five being
`LEAK SUMMARY: 14812160 bytes leaked in 5 allocations` at process exit, with
graphs on and off alike. Their backtraces are `cublasCreate_v2` (3),
`cudaMalloc` (1) and `ggml_cuda_pool_vmm::alloc` (1), all under
`ggml_cuda_mul_mat_cublas` / `ggml_cuda_mul_mat_id` in
`ggml_backend_cuda_graph_compute`: the lazily created cuBLAS handle, its
workspace and the VMM pool of the process-wide CUDA backend context, which
TensorSharp keeps alive for the life of the process and never frees before exit.
Not a per-call leak (the count does not grow with the 132 checks) and not
graph-related. The target snapshot captures no CUDA graph at all (0 warmup
lines: every span in that fixture has a different shape).

## 2. Why the fixture run was clean, and what the earlier run saw

ggml-cuda (`ExternalProjects/ggml/src/ggml-cuda/ggml-cuda.cu`,
`ggml_backend_cuda_graph_compute`) keys a CUDA graph on `cgraph->nodes[0]` and
captures only after two consecutive computes with identical node properties
("warmup"). The MTP executor (`TensorSharp.GGML.Native/ggml_ops_qwen4exp_mtp.cpp`,
`TSGgml_Qwen4ExpMtpForward`) rebuilds its ggml graph whenever the token count,
padded key count, head option or mrope mode changes, and the DSV4 fused backend
wrapper (`ggml_ops_dsv4_fused.cu`, `tsg_dsv4_backend_graph_compute`) hands each
maximal run of ordinary nodes to ggml-cuda as a `ggml_graph_view` (23 views per
MTP forward here). The three code paths that matter:

* `ggml_cuda_graph_update_required` (line ~2595): compares every node's op,
  shapes, strides and source pointers with the stored copy; a difference resets
  warmup ("CUDA graph warmup reset").
* `ggml_cuda_graph_evaluate_and_capture` (line ~4386): after capture, if the key
  has no executable yet it calls `cudaGraphInstantiate`; if the key already has
  one from an earlier topology it calls `ggml_cuda_graph_update_executable`.
* `ggml_cuda_graph_update_executable` (line ~2637): `cudaGraphExecUpdate(old
  instance, new graph)`; on `cudaErrorGraphExecUpdateFailure` it calls
  `(void)cudaGetLastError()` to clear the sticky error, `cudaGraphExecDestroy`,
  then `cudaGraphInstantiate` a fresh executable. Any other failure asserts.

The operator fixture's call sequence (`first_token`, `same_shape_changed_inputs`,
`same_shape_original_again`, then `prefix_five`, `append_one`, ... each with a
new shape) reaches capture exactly once per key (23 captures, then 23 resets in
the same process) and never again runs two identical forwards after a reset. So
the fixture instantiates fresh executables only and never calls
`cudaGraphExecUpdate` on an instance built for another topology. That is why
runs (b) and (c) of the fixture are both clean: the finding cannot be reproduced
or cleared by the fixture alone.

`graph-update-probe.py` forces the update path with the same `Fixture`, oracle
and tolerance as the operator fixture: it alternates token counts 1 and 5, calls
each shape three times back to back (record, capture+update, replay) for three
rounds, checks every forward against the F64 oracle and requires rounds 1 and 2
to be bitwise identical to round 0.

```sh
PROBE="/usr/local/bin/python -B docs/validation/qwen38-mtp-cuda-graphs/graph-update-probe.py \
  --operator eng/tests/qwen4exp-mtp-operator.py --sample fixtures/qwen4exp-mtp/sample.json --library $LIB"
$PROBE --report probe-plain-graphs-on.json
GGML_CUDA_DISABLE_GRAPHS=1 $PROBE --report probe-plain-graphs-off.json
$S --tool memcheck --print-limit 1000 --log-file probe-memcheck-graphs-on.sanitizer.log $PROBE --report probe-memcheck-graphs-on.json
GGML_CUDA_DISABLE_GRAPHS=1 $S --tool memcheck --print-limit 1000 --log-file probe-memcheck-graphs-off.sanitizer.log $PROBE --report probe-memcheck-graphs-off.json
$S --tool memcheck --leak-check full --report-api-errors all ... $PROBE ...      # graphs on and off
$S --tool memcheck --error-exitcode 3 ... $PROBE ...                             # graphs on and off
$S --tool initcheck ... / --tool synccheck ... / --tool racecheck ... $PROBE ...  # graphs on
```

| run | graphs | probe | sanitizer result | exit |
| --- | --- | --- | --- | --- |
| plain | on (138 captures, 92 resets) | PASSED 68 checks, 18 forwards | – | 0 |
| plain | off | PASSED 68 | – | 0 |
| memcheck | on (138 captures, 115 resets) | PASSED 68 | `ERROR SUMMARY: 60 errors` | 0 |
| memcheck | off | PASSED 68 | `0 errors` | 0 |
| memcheck `--leak-check full --report-api-errors all` | on | PASSED 68 | `60 errors`, `0 bytes leaked` | 0 |
| memcheck `--leak-check full --report-api-errors all` | off | PASSED 68 | `0 errors`, `0 bytes leaked` | 0 |
| memcheck `--error-exitcode 3` | on | PASSED 68 | `48 errors` | **3** |
| memcheck `--error-exitcode 3` | off | PASSED 68 | `0 errors` | 0 |
| initcheck | on (138 captures) | PASSED 68 | `0 errors` | 0 |
| synccheck | on (138 captures) | PASSED 68 | `0 errors` | 0 |
| racecheck | on (138 captures) | PASSED 68 | `0 hazards (0 errors, 0 warnings)` | 0 |

Every one of the 60 (respectively 48) reported errors is one of exactly two
lines, in equal numbers (30+30, 24+24):

```
Program hit cudaErrorGraphExecUpdateFailure (error 910) due to "the graph update was not
  performed because it included changes which violated constraints specific to
  instantiated graph update" on CUDA API call to cudaGraphExecUpdate.
Program hit cudaErrorGraphExecUpdateFailure (error 910) due to "..." on CUDA API call to cudaGetLastError.
```

with the identical host backtrace on all of them:

```
ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*)      libGgmlOps.so   <- ggml_cuda_graph_update_executable, inlined
tsg_dsv4_backend_graph_compute(ggml_backend*, ggml_cgraph*)       libGgmlOps.so   <- ggml_ops_dsv4_fused.cu, graph view hand-off
ggml_backend_graph_compute                                        libGgmlOps.so
tsg::graph_compute_profiled(ggml_backend*, ggml_cgraph*, char const*)
TSGgml_Qwen4ExpMtpForward                                         libGgmlOps.so   <- ggml_ops_qwen4exp_mtp.cpp
call / main                                                       graph-update-probe.py
```

So each pair is one `cudaGraphExecUpdate` that CUDA refused because the
re-captured view differs from the executable instantiated for the other token
count in a way an in-place update does not allow (the driver does not name the
constraint; a different kernel set or node count between the 1-row and 5-row
views is the expected cause), plus the `cudaGetLastError()` ggml issues to clear
it: the sanitizer reports the same sticky error twice. The counts match the mechanics: 5 shape switches with a live
executable x 23 views = 115 `warmup reset` lines and 115 updates, of which the
views whose topology changed (30 in one process, 24 in another: the set depends
on which freed graph-context addresses the allocator reuses as keys) took the
re-instantiate branch. No device-side error, no invalid access, no leak, no race
or barrier hazard accompanies them, with graphs on or off, and the outputs of
the re-instantiated graphs are the oracle's to 2e-5 and bitwise stable across
rounds.

Exit codes: compute-sanitizer returns the application's exit code unless
`--error-exitcode` is given, so the "nonzero exit" of the earlier run is that
flag (or an equivalent wrapper) counting the 910 pairs; it is 0 for the same run
without the flag and 0 under `GGML_CUDA_DISABLE_GRAPHS=1` with it.

## 3. Verdict

* Real defect in TensorSharp-owned code: **none.** The owned code never touches
  the CUDA graph API; it builds ggml graphs and hands views to ggml-cuda, whose
  documented behaviour is to try an in-place executable update and fall back to
  re-instantiation; llama.cpp ships this same ggml-cuda code path unchanged.
* The retained errors are benign, handled by ggml, and reproducible on demand
  with the probe; the operator fixture itself does not exercise the path.
* How to run API-error-sensitive sanitizer checks on this operator in future:
  either accept the `cudaGraphExecUpdate` + `cudaGetLastError` 910 pairs (only
  those two lines, only that backtrace), or run the API-error pass with
  `GGML_CUDA_DISABLE_GRAPHS=1` and the device-side passes (memcheck without
  `--error-exitcode`, initcheck, synccheck, racecheck) with graphs enabled, as
  done here. Both are documented above with their outputs.

## 4. Managed Qwen4ExpMtp tests on the CUDA backend

Command, from the shared tree with `--no-build`:

```sh
cd /workspace/ts-main/repo
CUDA_VISIBLE_DEVICES=6 TS_TEST_GGML_BACKEND=cuda TS_TEST_QWEN4EXP_MTP_BACKEND=GgmlCuda \
TS_TEST_QWEN4EXP_MTP_FIXTURE=/workspace/ts-main/repo/fixtures/qwen4exp-qsa \
TS_TEST_QWEN4EXP_NATIVE_SHA256=48abe2b1623358f1567861c304e355686b00d9cf7c7d6803a6316f570ab6b85d \
LD_LIBRARY_PATH=/workspace/ts-main/repo/TensorSharp.GGML.Native/build \
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release --no-build --filter 'FullyQualifiedName~Qwen4ExpMtp'
```

Result against the unmodified shared build: **103 passed, 6 failed, 2 skipped
(111 total)**. All six failures are the same assertion before any model code
runs:

```
Assert.Single() Failure: The collection was empty
   at Qwen4ExpMtpIntegrationTests.Fixture.Load() in InferenceWeb.Tests/Qwen4ExpMtpIntegrationTests.cs:line 429
```

`Fixture.Load()` located the mapped native module by the file name
`GgmlOps.dll`; on Linux the module is `libGgmlOps.so`, so the filter matched
nothing and every test that loads the model (`DisposingDraftHead_...`,
`TargetSnapshot_...`, `LearnedHead_...`, `DraftPrivateState_...`,
`DraftHead_F16CacheCatchUp...`, `HistoricalMediaPositions_...`) failed on that
line. The two skips are the QSA tests gated on `TS_TEST_QWEN4EXP_QSA=1`. This is
a Linux-portability defect of the test, not of the CUDA path; it was fixed in
`Qwen4ExpMtpIntegrationTests.cs` by resolving the platform file name the way
`Qwen35VerifyOwnerIsolationTests` and `GgmlNative` already do
(`GgmlOps.dll` / `libGgmlOps.dylib` / `libGgmlOps.so`). The fixture tests now
share one helper, `TestGates.MappedNativeGgmlOpsPath()`, which matches that
platform name and on macOS reads dyld's mapped-image table (`Process.Modules`
omits `dlopen`-loaded libraries there); the result below was recorded with the
per-test lookup.

Result with that fix (managed Release build of this branch in
`/workspace/ts-sanitizer/repo`, same shared native library, same environment):
**109 passed, 0 failed, 2 skipped (111 total)**; the six previously failing
tests each logged
`native /workspace/ts-main/repo/TensorSharp.GGML.Native/build/libGgmlOps.so sha256=48abe2b1…; backend=GgmlCuda`
and `ggml_cuda_init` reported the A40, so the CUDA path was the one exercised.
The two remaining skips are the `Qwen4ExpQsaTinyFact` tests, which additionally
need `TS_TEST_QWEN4EXP_QSA=1 TS_KV_INITIAL_TOKENS=8`; with those set the same
command on GPU 6 gives **111 passed, 0 failed, 0 skipped**.
