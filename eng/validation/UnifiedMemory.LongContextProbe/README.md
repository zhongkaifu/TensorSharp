# Shared-budget long-context acceptance

Runs one fresh process for each arm on a single CUDA rank. The reference uses
TensorSharp's resident model and managed host snapshots. The candidate uses
`AdaptiveModelSession`, instrumented host/native allocations, shared RAM/device/SSD
budgets, and `CreateRequestMemoryAdmission`. Both use the same context, F16 KV,
prefill chunk, greedy prompts and per-sequence execution path. No upstream ggml
changes are required.

Build the repository's native library against its unchanged ggml checkout, then:

```sh
dotnet build eng/validation/UnifiedMemory.LongContextProbe -c Release \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
export TS_VALIDATION_GGML_REVISION=$(git -C ExternalProjects/ggml rev-parse HEAD)
probe=eng/validation/UnifiedMemory.LongContextProbe/bin/Release/net10.0/UnifiedMemory.LongContextProbe.dll
dotnet "$probe" --model /workspace/models/Qwen3.5-0.8B-Q8_0.gguf \
  --json artifacts/long-context/reference.json --arm reference \
  --context 32768 --prompt 28000 --steps 16 --width 2 --chunk 256
dotnet "$probe" --model /workspace/models/Qwen3.5-0.8B-Q8_0.gguf \
  --json artifacts/long-context/candidate.json --arm candidate \
  --reference artifacts/long-context/reference.json \
  --context 32768 --prompt 28000 --steps 16 --width 2 --chunk 256 \
  --host-bytes 8589934592 --device-bytes 8589934592 --ssd-bytes 17179869184
```

Use isolated processes with no competing inference work. The candidate verifies
the model hash, context, chunk, rendered prompt hashes and every generated token
against the reference. A nonempty exact-length completion and released allocation
credit are also required. Failures return exit code 1 and include the error in
JSON. Unsupported architectures/restoration windows are failures, not passes.
This is storage/execution parity, not an independent semantic or llama.cpp oracle.

`--width` controls submitted concurrency; `--max-running` independently caps
admitted concurrency (defaults to width). For queued serial work, set
`--width 2 --max-running 1` on both arms. With prefix reuse disabled this lane
does not capture request snapshots, so `--ssd-bytes 0` is valid even when a
snapshot of the whole prompt would not fit RAM. Engine capture scratch and
transfer staging still count. A Gemma prompt may exceed its rolling-window
restore limit only in this no-swap lane; interleaved restoration keeps the limit.

Both arms set `PrefillChunkTokenLimit`; solo and contention scheduling cannot
silently grow the workspace beyond `--chunk`. Candidate admission also binds
the snapshot object/budget, block size, running limit, and maximum chunk used by
its estimate. Configuration/sequence mismatches are rejected before allocation.
`PrefillChunkLimitEnforced` distinguishes these runs from older probe results
that configured chunk hints without an unconditional per-forward ceiling.
`*.planning.json` preserves the selected plan and request peaks before execution,
including for native failures that prevent a final result. Planning evidence
alone is never a passing run.

Interpret the recorded fields as follows:

- `ContextTokens` is configured capacity. `Requests[].PromptTokens` is the actual
  rendered input length; `--prompt` is only its minimum. All runs generate `Steps`
  tokens, including the token produced by prefill.
- `PrefillTokensPerSecond` uses actual prompt tokens divided by prefill forward
  duration. `DecodeTokensPerSecond` uses `(Steps - 1)` divided by decode forward
  duration. Neither includes all scheduler and snapshot overhead. `WallSeconds`
  and `TtftSeconds` do include those costs.
- `Width` is submitted concurrency. Inspect `RunningSamples` for actual admission;
  smaller quotas can serialize requests. Two admitted requests still execute
  per sequence, rather than as a batched CUDA kernel.
- `RequestPeaks` reports execution workspace/live-state and total request
  reservations. `HighWatermarks` records exact ledger maxima: `Owned` includes
  unused request credit; `Committed` counts instrumented allocations. Retained
  buffers stay charged after a request closes. `AfterDispose` must be zero.
- Process working set/high-water mark and sampled CUDA total-minus-free are
  separate measurements. The latter covers the device, including other owners,
  and can miss brief peaks. Allocation quotas are not process RSS/VRAM hard limits.
- `Residency.Spills > 0` is required to establish actual spill. A lone request with
  prefix caching disabled takes no snapshots, regardless of context length.
- Cold graph construction is included. Model load is measured separately, prompt
  rendering is outside request timing, and OS file cache/clock frequency are not
  controlled. Repeat and reverse arm order before concluding a speed difference.

The process installs one allocation adapter before model construction. Prefix
caching, speculative/MTP execution and batched holders are explicitly disabled.
Media, multi-rank snapshot restoration and unqualified models must not be counted
as covered. See [the design evidence](../../../docs/design/unified-memory.zh-CN.md)
for executed hardware, model and numerical coverage.

## Physical RAM acceptance

`../run-memory-cgroup.py` can wrap either command in a **new delegated Linux
memory cgroup**. It never changes the parent controller or existing limits:

```sh
python eng/validation/run-memory-cgroup.py \
  --cgroup-parent /sys/fs/cgroup/my-delegation --ram-bytes 8589934592 \
  --output artifacts/long-context/physical-8g --timeout 1200 -- \
  dotnet "$probe" --model /workspace/models/Qwen3.5-0.8B-Q8_0.gguf \
  --json artifacts/long-context/physical-candidate.json --arm candidate \
  --reference artifacts/long-context/reference.json \
  --context 32768 --prompt 28000 --steps 16 --width 2 --chunk 256 \
  --host-bytes 4294967296 --device-bytes 4294967296
```

The wrapper joins before exec, verifies the RAM limit and disables swap. It logs
cgroup usage/peaks/events and process RSS/anonymous/file-backed counters. It only
cleans up its newly created group and owned process group. Missing delegation or
unsupported controllers return **77 / unavailable / passed=false**, without
launching an unrestricted fallback. An enforced run still requires a successful
model comparison; an OOM-killed child fails validation. No GPU hard limit is set.

Cgroup memory is not identical to process RSS: shared file pages may be charged
to a different first-touching group. See the kernel's
[v1 memory controller](https://docs.kernel.org/admin-guide/cgroup-v1/memory.html)
and [v2 memory controller](https://docs.kernel.org/admin-guide/cgroup-v2.html)
documentation. Record the actual enforcement and environment; never infer a
physical-cap pass from a `MemoryBudget` capacity alone.
