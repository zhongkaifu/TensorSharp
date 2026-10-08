# Linux memory-limit evidence

Run the read-only inspector before choosing a constrained inference experiment:

```sh
python3 eng/validation/inspect-linux-memory-cgroup.py \
  --model /path/to/model.gguf --proposed-limit-bytes 2147483648 \
  --output artifacts/unified-memory/linux-memory-cgroup.json
```

It resolves both cgroup v1 memory mounts and cgroup v2 mounts, including a container
whose mount root is its host cgroup path. `--pid` observes a running model process
instead of the inspector. File size is obtained with `stat`; tensor data is not read.
The script never creates groups, changes controller configuration or limits, moves
processes, or drops filesystem caches. A writable directory is only a candidate;
successful creation/delegation is deliberately recorded as unverified.

## Running a constrained experiment

Use an isolated child memory cgroup, or a container/job scope with a host-enforced
memory limit. Keep the supervisor and unrelated inference jobs outside it. Do not
lower the shared VM/container's existing limit. The launcher must perform these
steps, and treat any unavailable step as unavailable validation:

1. Create a private child in a delegated memory hierarchy and set its limit before
   starting the model. On v2, record `memory.max` and `memory.swap.max`. On v1,
   record `memory.limit_in_bytes` and, when supported, `memory.memsw.limit_in_bytes`;
   the latter is the combined memory-plus-swap limit, not the swap allowance alone.
2. Move the child process into the group before it starts the runtime or loads
   weights. Read `/proc/<pid>/cgroup` to verify membership. Do not move the launcher
   itself or other live model processes.
3. Run identical resident and streamed workloads in separate fresh groups, saving
   exit codes, numerical comparison results, payload budget peaks, process RSS and
   high-water RSS, cgroup peak usage, page-cache/RSS breakdown, swap usage, and OOM
   counters. Preserve the actual effective limit and model-file size in the report.
4. Wait for all descendants to exit before collecting final counters and removing
   the child group. Do not kill unrelated processes or drop global caches to make
   the result fit.

`ulimit -v`/`RLIMIT_AS` limits virtual address space and is not a substitute for a
cgroup physical-memory experiment. CUDA and .NET reserve large address ranges, so
an address-space failure alone cannot establish the claimed RAM requirement.

## What the results establish

- Weight-payload budgets bound only allocations owned by the streaming adapter.
  A successful budget test does not establish a whole-process RSS or VRAM limit.
- A host cgroup charges anonymous memory and attributable filesystem page cache;
  process RSS and adapter payload counters measure different things. Buffered
  reads can fill reclaimable cache even when staging is small. Cache already
  charged to another group can also bias an experiment; report that limitation.
- Leave capacity for the runtime, libraries, activations, KV, native pools and
  bookkeeping. A 2 GiB host limit around an approximately 811 MB GGUF can test a
  bounded process envelope, but cannot establish host capacity smaller than that
  model file. If the runtime's minimum working set cannot fit below the file size,
  report that boundary instead of claiming a physical out-of-memory demonstration.
- CUDA device memory is a separate resource. Record actual device memory and the
  adapter's charged device workspace; a host memory limit does not constrain VRAM.
- A denied cgroup setup, skipped workload, OOM-killed streamed run or missing
  numerical comparison is not a passing constrained-inference result.

The provided VM was inspected read-only on 2026-10-08. Its visible cgroup v1 memory
mount reported an existing 99,999,997,952-byte container limit. The memory-cgroup
directory failed the write-permission check despite the mount reporting `rw`;
child creation was not attempted. No additional host-memory limit was installed.
The generated report belongs under ignored `artifacts/`, not in source control.
