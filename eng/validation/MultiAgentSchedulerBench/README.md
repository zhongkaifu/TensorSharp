# Dependency scheduler benchmark

Runs a fixed ten-task DAG (six independent inputs, three pairwise joins, one final
join) through the real `MultiAgentSession`. Compares a one-slot scheduler with
three slots and separately measures the same graph with zero simulated work.
Every run checks prerequisite ordering, prerequisite evidence handoff, completed
task count and peak concurrent generators. Warmups are excluded and run order
alternates. A failed correctness check returns a nonzero exit code; speedup is
reported without asserting a timing threshold.

```sh
dotnet run --project eng/validation/MultiAgentSchedulerBench -c Release -- --iterations 8 --warmup 1 --work-ms 40 --out artifacts/multi-agent/scheduler.json
```

The delays and generators are deterministic fixtures. Results measure queue and
dependency scheduling, not inference, autonomous task decomposition, model
quality, workspace merging or hardware throughput. The overhead arm includes
tool argument parsing, prompt construction and result collection. Timer
resolution and machine load affect timing. For model-driven extraction and
synthesis, use the neighboring `MultiAgentBench` endpoint mode and record the
actual model revision, backend and server settings. Generated reports must remain
in ignored `artifacts/` or `docs/validation/` directories.
