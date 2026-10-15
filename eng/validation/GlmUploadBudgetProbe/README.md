# GLM uploader ownership probe

This small native GLM fixture exercises the real model loader. It compares an
unbudgeted load with a one-byte host-staging refusal followed by a same-process
successful retry. All 512 logits from 37 prefill tokens and three single-token
steps must be bit-exact. The refusal must leave no staging owner, and successful
loads must record a positive host payload high-water mark and release it.

Set these variables **before process launch**; changing the managed environment
in `Main` is not a reliable way to update an initialized native CRT on Windows:

```text
TS_GLM_LOAD_THREADS=2
TS_GLM_LOAD_CHUNK_MB=1
TS_GLM_MOE_MMAP=0
TS_CPU_MOE_THREADS=2
```

Build the project, then run:

```text
dotnet GlmUploadBudgetProbe.dll CPU <new-output-directory>
dotnet GlmUploadBudgetProbe.dll CUDA <another-new-output-directory>
```

For the native **default-planning** regression, remove `TS_GLM_LOAD_THREADS`
and `TS_GLM_LOAD_CHUNK_MB` entirely, retain the other two variables, and append
`default-planning` to either command. This retries under **32 KiB** of shared
host payload credit, smaller than a single largest 64 KiB synthetic tensor job
and much smaller than the default multiworker workspace. The test requires
preflight negotiation refusals, then a successful bounded load, all 512 logits
bit-exact, and zero remaining owners/ledger. The native `upload plan` line
records actual workers/chunk/staging/TP slab/aggregate, whether accounting is
active, and negotiation refusal count. These refusals are pre-upload capacity
proposals, not GPU OOMs or failed model-load retries. A one-byte quota must still
refuse before upload; the planner does not shrink to one-byte I/O to evade it.

The native planner acquires one atomic reservation before any worker starts,
allocates/commits one exact payload arena, and lends disjoint non-growing
slices to workers. It reduces worker count before reducing chunk size, with a
portable 4 KiB useful-I/O floor (or the smaller job/request, respecting complete
TP rows). TP source-read slabs are part of that same arena; a quota too small
for staging plus a whole strip safely refuses. The bounded backoff is a safe
feasible plan, not a claim of optimal storage/worker throughput. Unaccounted
uploads retain their previous lazy worker allocation and file-read job shapes.
Explicit malformed/overflowing load options now fail before upload instead of
being partially parsed or wrapping an allocation size.

Run each through `run-bounded-probe.py` for a process deadline and durable exit
record. Keep output in ignored `artifacts/` or `docs/validation/`. The report
records the actual native module hash; retain assembly/source identities and
the unchanged ggml dependency revision with the invoking validation manifest.

The fixture is GLM-DSA, not trained GLM 5.3 Flash. The common uploader is under
test, including a refused real native allocation request. This does not cover
full model weight ownership, mmap residency, allocator headers, FILE buffers,
thread stacks, mapped-file prefault buffers/page cache, driver staging,
multi-GPU TP, semantic quality or throughput.
The portable C++ `glm_upload_buffer_test.cpp` separately injects allocation and
commit failure and exercises real strided file reads.
`glm_upload_plan_test.cpp` adds aggregate reservation/allocation/commit failures,
slice non-overlap, concurrent planners, minimum-quota rejection, TP geometry and
file-boundary checks. Those CPU tests do not establish CUDA or multi-GPU parity.
With a configured native build, the standalone target is
`GgmlOpsGlmUploadPlanTest` and its CTest name is
`glm-upload-budget-preplanning` (30-second timeout). It needs C++17 and threads,
but does not load ggml, a model, or a GPU.
