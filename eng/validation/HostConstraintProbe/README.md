# Host allocation constraint observation

This CPU-only probe samples `HostMemoryInfo.CaptureAllocationConstraints` before
and after an optional native allocation. Allocation is capped at 64 MiB, touches
each system page, and is freed in `finally`. It never loads a model or initializes
an accelerator. All arguments require values except the standalone `--self-test`.

```powershell
dotnet HostConstraintProbe.dll --self-test
dotnet HostConstraintProbe.dll --output artifacts/host-observation.json --allocation-bytes 67108864
python eng/validation/run-memory-job.py --detached --commit-bytes 536870912 --timeout 60 --output artifacts/host-job-observation -- dotnet HostConstraintProbe.dll --output artifacts/host-job-observation/api.json --allocation-bytes 67108864 --expect-commit-ceiling-bytes 536870912
```

Use a fresh evidence path. The existing Job launcher verifies membership and
allocation refusal before starting the probe. `--expect-commit-ceiling-bytes`
requires every sample to report both a valid commit limit and commit availability
no greater than that ceiling. It deliberately fails when the API does not expose
the expected constraint, even if the external Job enforces it. The probe does not
install or independently certify a limit. Preserve both reports and distinguish
API observability failure from unavailable or failed Job enforcement.
Each sample also exports the provider's `CommitConstraintScope`; a passing
immediate-Job case does not prove hidden-ancestor or shared-Job usage coverage.
The optional `--detached` launcher mode avoids a console-host member while still
creating the owned child suspended and assigning it before execution. Its report
records creation flags and actual Job members before resume. The default launcher
mode is unchanged: on some Windows environments it also assigns `conhost.exe` to
the Job, and the provider correctly refuses the unobserved shared commitment.
Keep that refusal as a separate negative case; do not classify it as API success.

Run without a commit ceiling on Linux. The current provider must explicitly
report commit as unknown while retaining its physical/cgroup observation. This
does not establish Windows Job, cgroup, Apple, physical RSS or GPU equivalence.
Freed payload need not cause an exact immediate counter rebound: allocator
retention and other processes can affect sampled counters. Pure `--self-test`
checks parameter bounds and classification only; it is not an OS constraint test.
