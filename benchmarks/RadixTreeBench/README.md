# RadixTreeBench

CPU-only benchmark of the radix prefix tree in `TensorSharp.Runtime/Scheduling/PrefixCache`
(radix prefix cache design, gate BG-15). It loads no model and needs no GPU: every trace drives
`PrefixTree` directly with synthetic token streams.

```bash
dotnet run -c Release --project benchmarks/RadixTreeBench -- --trace all --out rtb.json
# smaller and faster: --scale 0.1 ; one trace: --trace long262k
```

## Traces

| Trace | Shape | What it measures |
|---|---|---|
| `sessions` | 1,000 sessions × 50 turns growing to ~8k tokens, round-robin; 4 shared 512-token system prompts published as public boundaries; every 10th turn regenerates (a 50-token fork); a byte budget for 800 sessions | probe (Match + Evaluate), acquire, insert, eviction under a large tree (~40k nodes) |
| `long32k` | 64 sessions × 20 turns at 30k-32k tokens | probe and insert latency at 32k |
| `long262k` | 8 sessions × 20 turns at 258k-262k tokens | probe latency at 262k |
| `queue128` | 128 waiting requests, at most 32 re-matches per step, one admission + insert per step, a background conversation that changes the tree every step | admission overhead per step |
| `fanout` | one public 1,024-token node with 10,000 scoped children; probes from existing and new scopes | child lookup at a wide node |
| `churn` | 5,000 new 4k-token conversations under a budget for 64 | evict-one-victim latency |

Before measuring, the tool runs every trace at a small size for about two seconds and sleeps for
one second, so one-time JIT and type initialisation (latency outliers, and a small allocation on
the first probes of a process) and tiered compilation happen before the measured probes.

### The allocation counter needs concurrent GC off

`bytes allocated per probe` reads `GC.GetAllocatedBytesForCurrentThread()` around every probe. A
**background** GC makes that counter jump without any allocation: at the end of its mark phase it
suspends the process and voids every thread's allocation context (`repair_allocation_contexts(FALSE)`
in the runtime's `gc.cpp`) without taking the context's unused remainder out of the thread's
allocated bytes. When that suspension lands inside a probe, the probe reads anything from a few
hundred bytes to one allocation quantum (~8 KB), and the process counts no collection for it.

This was the intermittent BG-15 allocation failure (about one `fanout` run in 8-40, one probe of
512-7,792 bytes in the second half of the trace). With concurrent GC on, every measured `fanout`
run starts two gen2 collections, background ones, while it builds the 10,000-child tree, and a run
fails only when one of those suspensions happens to land inside a probe. It was not tier-up: it
happened with tiered compilation off too. The evidence:

- With another thread forcing background gen2 GCs, a window that only calls `Thread.SpinWait`
  read up to 8,112 bytes, and repeated `fanout` probe runs read 2,008-9,360 bytes per run, with
  single probes of 1,288 and 3,280 bytes that each took about 0.5 ms instead of 3 µs (the
  suspension) with no gen0 or gen2 count change across the probe.
- The same stress with `DOTNET_gcConcurrent=0` read 0 bytes in every window and every probe.

So `RadixTreeBench.csproj` sets `ConcurrentGarbageCollection=false`, and the tool exits with 2 if
the GC latency mode is not `Batch` (for example under `DOTNET_gcConcurrent=1`) instead of reporting
a gate it cannot measure. A blocking GC keeps the counter exact: it takes each context's unused
remainder out of the allocated bytes before it clears the context. The gate stays strict: any byte
in any measured probe fails it.

## Gates (BG-15)

| Gate | Threshold |
|---|---|
| probe p99 @ 32k | ≤ 200 µs |
| probe p99 @ 262k | ≤ 2 ms |
| insert p99 @ 32k | ≤ 500 µs |
| evict one victim p99 (model release excluded) | ≤ 50 µs |
| admission overhead per step p99 @ queue 128 | ≤ 5 ms |
| bytes allocated per probe (all traces) | 0 |

The process exits with 0 when every gate holds and 1 otherwise. `--out` writes the per-trace
percentiles (p50/p95/p99/max), reuse, node counts, eviction scan skips, rope compactions and the
gate table as JSON.
