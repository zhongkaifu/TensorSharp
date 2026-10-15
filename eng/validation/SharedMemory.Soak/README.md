# Shared memory cancellation and fairness soak

This portable CPU tool drives four real `InferenceEngine` worker threads sharing
one `MemoryBudget`. The model is a deterministic stateful test double, not a
trained model or a simulated GPU. Each admitted request allocates and touches a
real 64 KiB unmanaged host buffer using a child of its request envelope. The
combined allocation ceiling is 128 KiB.

```powershell
dotnet build eng/validation/SharedMemory.Soak/SharedMemory.Soak.csproj -c Release -m:1
dotnet eng/validation/SharedMemory.Soak/bin/Release/net10.0/SharedMemory.Soak.dll `
  --seconds 7200 --output artifacts/default-readiness-20261010/cpu-soak
```

Add `--admission-hook true` to exercise `RequestMemoryAdmission.BeforeAdmission`
using a synthetic fixed 128 KiB host-pool capacity constraint. The hook asserts
that it runs under the model lock and only before envelope admission. Reports
record whether it is enabled and its total and per-engine call counts. This
mode exercises constrained reservation, cancellation and fairness; it does not
read hardware memory or refresh physical capacities. The default is `false`.

Use a separate output directory for every run. To keep evidence stable while
other projects are built, copy the entire executable output directory and run
that copy. Preserve the process exit code as well as `report.json`.
Check the copied Runtime, Memory, and tool hashes before launching; copying them
into a new directory prevents a later build from replacing a running fixture.
A duration claim requires the terminal `passed` report, elapsed seconds at least
the requested duration, and exit code zero. Normal progress followed by a missing
process without that terminal evidence is an interrupted run, even when the last
drained wave had no live owners. Preserve those logs and restart in a fresh
directory; do not add separate partial durations to claim a continuous soak.

Every wave submits eight requests to each engine while an external owner holds
all credit. Four requests per engine must complete with all six output tokens
equal to the independent stateful reference. The other four must be cancelled:
one before submission, two while queued, and one from its first actual forward.
Request IDs are reused on the next wave to exercise cancellation identity.
Every twentieth wave also closes one engine's compute gate while queued
cancellations drain. Each wave has a ten-second completion deadline. All four
engines must progress in every wave; this is a bounded progress check for this
fixed workload, not a proof of fairness for every arrival distribution.

After each wave, the tool checks both the physical host allocation count and
the shared ledger have returned to zero. Shutdown disposes every engine and
model, repeats the ownership checks, and fails on cleanup errors. `progress.jsonl`
records actual elapsed duration, completed/cancelled counts, per-engine progress
and maximum admission waits, physical allocation/release totals, ledger peaks,
process CPU time, and sampled private-memory/working-set peaks every 30 seconds.
Admission-wait maxima cover successfully completed requests; every cancellation
also has to finish within its wave deadline. `maximumWaveSeconds` records the
configured ten-second deadline, not a measured wave-duration maximum.
`report.json` includes final status, error details, and the loaded Runtime,
Memory and tool assembly hashes. A successful short smoke is not a successful
two-hour run.

The one-millisecond sleep in each forward is an explicit model-work stand-in;
it makes overlapping workers visible without consuming a CPU core. This tool
does not validate GPU kernels, trained-model quality, file swapping, process
RSS enforcement, or inference throughput. The ledger peak does not account for
all CLR/process memory. Even with the optional synthetic admission hook, it does
not validate hardware observation or physical-capacity refresh;
those admission boundaries have separate deterministic tests. Concurrent downloads,
builds, and GPU jobs must be
recorded separately; the process CPU counter can identify interference, but
does not make simultaneous throughput measurements controlled benchmarks.

Cross-engine policy regressions live in `SharedBudgetReclamationTests` and
`ContinuousBatchSchedulerTests`: overlapping-pool FIFO admission, independent
pool progress, cancelled or suspended queue owners, bounded donor attempts,
and demand-bound exclusive reclamation. External users of `MemoryBudget` keep
atomic accounting but do not participate in registered-engine FIFO ordering.
