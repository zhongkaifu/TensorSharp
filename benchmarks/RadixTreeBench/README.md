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
one second, so tiered compilation finishes before the measured probes (a tier-up inside a probe
shows up as a one-off allocation and a latency outlier).

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
