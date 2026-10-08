# Adaptive memory execution

This probe exercises `AdaptiveModelSession` with the same `ForwardRefill` and
`Forward` entries as a regular resident model. It records load time separately,
one warmup followed by measured requests, complete raw-logit hashes, token IDs,
hardware free VRAM, process working set and the shared-budget ledger. Disposal
must release every charged allocation. Raw greedy histories are used to compare
arithmetic, **not** as language-quality validation; model-defined generation
suppression is tested separately by `GemmaRepetitionProbe`.

Build with the normal managed skip-native flags, then copy the independently
built `GgmlOps.dll` beside the probe. Never rebuild/copy into a running process.
Use an unchanged upstream checkout, record its revision, model SHA, loaded native
SHA and actual GPU. No missing or skipped scenario counts as a pass.

```powershell
dotnet eng/validation/AdaptiveMemoryProbe/bin/Release/net10.0/AdaptiveMemoryProbe.dll `
  --model C:/Works/models/gemma-4-E4B-it-uncensored-Q8_0.gguf `
  --mode adaptive --prompt-tokens 640 --context 2048 --steps 64 --repeats 3 `
  --output artifacts/adaptive/e4b-adaptive-0
```

Run fresh processes in resident/adaptive/adaptive/resident order, with exclusive
GPU use and no simultaneous CPU inference, compilation or model hashing. Pass
all four `report.json` paths to `compare-runs.py --output artifacts/.../summary.json`.
Inspect both timing distributions; one fast pilot is not a performance claim.
`--device-bytes` and `--host-bytes` are explicit operator ceilings for constrained
tests. An unsupported quantization must be refused rather than silently running
an unbounded alternate path.

Current adaptive loading supports dense Gemma4 and Qwen35, one GGML CUDA device
and a sequential text lane. Model context and prefill chunk are per-model values;
the production API does not mutate process environment. The probe pins KV dtype
and clears distributed/speculation environment only to make its two test arms
comparable. Requests with another geometry need a new plan at a quiescent boundary.

Planning uses physical available RAM/VRAM, explicit headroom, model layout,
context, KV/state and loading/prefill/decode workspace forecasts. Residency keeps
the existing fused graph; insufficient capacity can select a supported bounded
file adapter. Host page cache is demand-paged by the OS, not an owned full copy.
Covered native caches/graph buffers and file-staging allocations reserve actual
payload before allocation. These hooks do **not** constrain all process RSS,
driver/library pools or every native model executor. `RefreshCapacity()` preserves
live owners and refuses new admission on pressure; it never evicts active KV.

## Local baseline comparison (2026-10-08)

RTX 3080 Laptop 16 GiB, unchanged ggml
`ffa4e8b80930029a35991f94e7c8a93cd67730ab`, E4B Q8_0/F16 checkpoint
SHA256 `96c455818ff64884f0e2ae3bc5517675896c4eae60676cc9135b9bb865eaf15c`.
Resident/adaptive/adaptive/resident fresh processes, one excluded warmup and
three measured requests per process, context 2048, 640 minimum prompt tokens,
64 raw-logit rows/request. The latest `v2` run uses native SHA256
`cf1e969d8f5618734ea43484f96f5485b6fc005ea59a2d0572dc59af4d9078d0`
and Models assembly SHA256
`4f264ce32f3e68d9cc48357108ac1dbf86e002bb757b14c0e9a56123e9afc08c`.

| Path | Prefill median (range), tokens/s | Decode median (range), tokens/s |
| --- | ---: | ---: |
| E4B resident | 2158.09 (1944.21–2258.53) | 55.96 (55.33–56.29) |
| E4B adaptive resident | 2264.80 (2207.56–2285.54) | 56.24 (55.81–56.32) |
| Qwen3.5 0.8B Q8 resident | 2171.81 (2096.44–2195.73) | 18.38 (18.32–18.43) |
| Qwen3.5 0.8B Q8 adaptive resident | 2168.53 (2128.54–2191.54) | 18.36 (18.33–18.37) |

Six measured requests per arm and checkpoint; each checkpoint's complete raw-logit
histories have the same SHA, all processes exit zero and all charged owners are
released. E4B adaptive/resident median ratios are 1.0494 prefill and 1.0051 decode;
Qwen's are 0.9985 and 0.9989. Ranges overlap. This comparison measures the overhead
of adaptive placement against the same build's resident execution; it does not
establish that either implementation is optimal or matches an independent engine.
In particular, Qwen's absolute decode throughput remains under investigation.
Qwen checkpoint SHA256 is
`0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c`.
Hashing runs outside timing but warms file cache, so these are not cold-storage
measurements. The earlier forced 256 MiB streaming result has a different capacity
constraint and is not used as the resident baseline. Generated reports: ignored
`artifacts/unified-memory-adaptive/{e4b,qwen08}-abba-v2-*`; prior `v1` evidence is
retained separately and has an older assembly/native identity.

## Single-token Q8 projection comparison

`compare-native-runs.py` compares two resident binary directories with identical
managed assemblies and checkpoint, allowing only the native-library identity to
differ. Use `--control` for the first/last reports and `--candidate` for the middle
two reports in a control/candidate/candidate/control sequence. It requires full
raw-logit equality, excluded warmups and at least two fresh processes per arm.

On the same hardware and Qwen checkpoint above, context 2048, 643 prompt tokens
and 63 decode calls per request, the N=1 specialization keeps the previous
K-ordered FMA but skips seven unused activation columns. Native control SHA is
`cf1e969d8f5618734ea43484f96f5485b6fc005ea59a2d0572dc59af4d9078d0`;
candidate SHA is
`aad101dc524c20388a351e6b0b7bd05e4ee4901d41ab00b2655d55e4e6a8981d`.
Managed Models remains `4f264ce3...` as recorded above. Six measured requests per
arm, with no concurrent inference, build or download, gave:

| Native | Prefill median (range), tokens/s | Decode median (range), tokens/s |
| --- | ---: | ---: |
| Previous | 2174.71 (2130.12–2194.91) | 18.374 (18.354–18.390) |
| N=1 specialization | 2203.59 (2180.33–2234.07) | 20.122 (20.088–20.191) |

All complete raw-logit histories match bitwise and all four processes exit zero.
Decode improves 9.51% in this fixture; prefill ranges overlap and its kernel is
unchanged. This is still below the requested overall performance goal and is
not an independent-engine comparison. Resident mode has no shared-budget ledger,
so these runs do not establish allocation-owner release. Native tests separately
cover 96 shapes twice, independent FP64 checks, old/new N=1 byte equality, tail
canaries and both FullPrecision streaming entry paths. Evidence is ignored under
`artifacts/unified-memory-adaptive/q8-vector-native-abba-v1/` and
`q8-vector-*-v2*`. The opt-in `GgmlOpsQ8VectorBench --benchmark` uses synthetic
weights and reports projection timings separately; those are not model speeds.
