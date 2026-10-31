# Direct-CUDA prefill: A/B harness and model-level tests

Tooling for the direct-CUDA (`--backend cuda`) prefill work planned in
`docs/validation/direct-cuda-prefill/design/PLAN.md` (an ignored evidence directory, kept
on the machines that ran the analysis). Each work package (WP1 flash attention, WP2 grouped MoE,
WP3/WP5 Mamba2 and GDN on the device, WP4 dense prefill matmuls, WP6 allocator, WP7 chunk
policy) is gated with the same measurements, against the same base.

Evidence goes to `docs/validation/direct-cuda-prefill/<stage>/` (git-ignored). Large `.nsys-rep`
files stay on the VM; record their path.

## Pieces

| piece | what it is |
|---|---|
| `eng/validation/direct-cuda-prefill-ab.py` | interleaved A/B driver and report (this file's subject) |
| `eng/validation/direct-cuda-prefill-floors.json` | Stage 0's ggml-vs-ggml parity floors per model, the G3 gates' input (`--parity-floors ... --floors-model NAME`) |
| `eng/validation/nsysan.py` | summarises the last prefill of an nsys-profiled `--benchmark` run: per-category kernel time, GPU busy/idle, memcpy and API counts, cuBLAS GEMM classes (one 512 and one 2048 window per gate model) |
| `eng/ForcedLogitProbe` | teacher-forced logits with `cuda` and `cpu` backends, `target_tokens` cases, per-row relative L2, cosine, KL(ref‖x) and top-5 overlap, a per-case and overall `summary`, a `complete` flag and `run_id` in `metrics.json`, and greedy text when run without `--reference` |
| `TensorSharp.Backends.Cuda/native/kernels/tensorsharp_{flash,moe,ssm}_prefill.cu` | the three new kernel modules (P0: only their `ts_<module>_caps` version probe), loaded by `CudaFlashPrefillKernels`, `CudaMoePrefillKernels`, `CudaSsmPrefillKernels` over `CudaPrefillModule` |
| `InferenceWeb.Tests/DirectCudaPrefillModelTests.cs` | model-level tests on `BackendType.Cuda` over synthetic GGUFs (below) |
| `InferenceWeb.Tests/DirectCudaRealModelTests.cs` | the same checks on real checkpoints, weights-gated (`TS_TEST_MODEL_DIR`) |
| `InferenceWeb.Tests/CudaPrefillModuleTests.cs` | the module version contract: source == C# == committed PTX (checked without a GPU too), stale images refused |
| `InferenceWeb.Tests/KnownDefectDataAttribute.cs` | `[KnownDefectData]` rows and the defect list `DirectCudaKnownDefects` |

## The harness

Python 3, standard library only, Linux (it signals process groups). One engine process runs at
a time. Each arm is a built TensorSharp tree (`TensorSharp.Cli/bin`, `TensorSharp.Server.Host/bin`,
`eng/ForcedLogitProbe/bin`).

```
H=eng/validation/direct-cuda-prefill-ab.py
# 1. A/A control first, every lane session (two copies of the base tree), one model per family.
#    Without its report every timing gate is NOT RUN; with a band above 3% they are NOT RUN (STOP).
python3 -I $H bench --aa --out OUT/aa --model M.gguf --arm A=/root/ts-base --arm C=/root/ts-base2 \
    --base A --candidate C --sequence ACCAACCA --lengths 512 --decode 128
# 2. G1/G2 (+G6): prefill at 512/2048 (and 8192 where the package needs it), decode at 512+128.
#    --role control for regression controls (G1c), --fallback-arm P for the committed-PTX-only B'.
python3 -I $H bench --out OUT/bench --model M.gguf --arm A=/root/ts-base --arm B=/root/ts-head \
    --sequence ABBAABBA --lengths 512,2048 --aa-report OUT/aa/report.json \
    --expected-gain 0.15,0.40 --vram-budget-mib 512
# 3. G2c: decode through the server at concurrency 4/32/64:
python3 -I $H conc --out OUT/conc --model M.gguf --arm A=... --arm B=... --sequence ABBA --levels 4,32,64 \
    --aa-report OUT/aa/report.json
# 4. G2b: speculative acceptance + tok/s (models with a speculative path; ngram needs no drafter):
python3 -I $H spec --out OUT/spec --model M.gguf --arm A=... --arm B=... --sequence ABBA \
    --spec-args "--spec --spec-type ngram --spec-draft 8" --aa-report OUT/aa/report.json
# 5. G3b (+G3x with --bit-exact): teacher-forced logits on PINNED real-text cases against
#    ggml_cuda, per case, with Stage 0's floors:
python3 -I $H parity --out OUT/parity --model M.gguf --arm A=... --arm B=... \
    --cases docs/validation/direct-cuda-prefill/stage0/VM1/raw/parity/qwen08/gR/cases.json \
    --parity-floors eng/validation/direct-cuda-prefill-floors.json --floors-model qwen08
#    or --prompts DIR (code.txt, prose.txt, chat.txt cut to --lengths), or --reference-dir to reuse
#    a reference already on the VM.
# 6. G3c: greedy smoke, 128 tokens of the same cases on the reference, A and B:
python3 -I $H smoke --out OUT/smoke --model M.gguf --arm A=... --arm B=... --cases ... --tokens 128
# Recompute every verdict from the JSON Lines; the selftest needs no GPU:
python3 -I $H report --out OUT/bench --aa-report OUT/aa/report.json
python3 -I $H selftest
```

- **Runs never mix.** Each invocation gets a run id; an older `bench.jsonl` / `conc.jsonl` /
  `spec.jsonl` / `parity.jsonl` / `smoke.jsonl` in the same `--out` is moved aside to
  `<name>.prev-<run id>` first, and the report reads only the newest run.
- **Statistic.** Every process's first benchmark run is dropped (`--warmup-runs`); prefill that
  still ramps inside a process keeps only the second half (flagged `ramped`). Each process
  contributes the median of its used runs; an arm is the median of its process medians; a paired
  cycle (ABBA) ratio is the mean of the candidate's process medians over the base's. Depth: at
  least 4 processes and 16 timed samples per arm at <= 4096 tokens, 2 processes above
  (`--min-procs`, `--min-samples` override; the report records them), else NOT RUN.
- **A/A noise band** (`bench --aa`): the relative difference of the two copies' pooled medians plus
  the largest per-cycle deviation, per prefill length and for decode. Above 3% the report says
  STOP and every timing gate that would use it is NOT RUN.
- **Gates** (`report.md`, `report.json`):

  | gate | rule |
  |---|---|
  | G1 (gate model) | B/A >= 1 + max(3%, 2 x noise) at every gated length, every cycle ratio > 1; `--expected-gain LO,HI` flags a gain below LO ("explain before merge") |
  | G1c (`--role control`) | B/A >= 1 - max(1%, noise) |
  | G1' (`--fallback-arm P`, B') | P/A >= 1 - max(1%, noise): the fallbacks are not slower than A beyond the noise |
  | G2 decode | B/A >= 1 - max(1%, decode noise); with decode noise above 1%, at least 4 ABBA cycles |
  | G2b speculative | acceptance(B) >= acceptance(A) - 0.02 (warm-up request excluded) and decode B/A >= 1 - max(1%, noise) |
  | G2c concurrent | B/A >= 1 - max(1%, noise) at every level; tokens counted from the final chunk's `usage` |
  | G3b parity | per case and metric (step-0 relative L2, step-0 KL, median relative L2, median KL): B <= max(1.10 x A, floor); top-1 mismatches B <= A + floor flips. ERROR unless both arms finished every case (exit 0/1, `complete`, this run's `run_id`, the reference's row counts, all rows finite) |
  | G3x (`--bit-exact`) | every logits row of B byte-identical to A's |
  | G3c greedy smoke | `--bit-exact`: tokens and logits identical to A; otherwise B's first divergence from the reference no earlier than A's on at least 2 of 3 prompts; the text is in the report for a human coherence check |
  | G6 VRAM | B's nvidia-smi peak <= A's + `--vram-budget-mib` |

  A failed process, a missing measurement, a server error, an absent A/A report or synthetic
  parity prompts make a gate NOT RUN or ERROR, never a pass.
- **Parity inputs.** Never random tokens. `--cases` takes a probe `cases.json` with pinned
  `prompt_tokens` (Stage 0's `gR/cases.json`); `--prompts` takes real text files;
  `--synthetic-prompts` (a few sentences repeated) is a smoke run whose G3b is NOT RUN.
- **Environment** set for every process: `TENSORSHARP_GGML_NO_UPDATE=1`, `TS_PREFILL_WARMUP=0`,
  `MAX_CONTEXT` 4096 (16384 above 4096 tokens), `CUDA_VISIBLE_DEVICES=0`; add more with `--env K=V`.
  The GGUF is read once before each block (`--no-warm-cache` skips it). VRAM is sampled every
  250 ms (`*.vram`, `vram_peak_mib` in the JSON Lines).
- **VM sharing.** The harness does not claim the VM. Before a GPU block, wait until `/root/vmlock`
  holds no other `main-*` claim and `nvidia-smi` shows no foreign compute process, then
  `/root/filler/vm-claim.sh <label>`; release with `/root/filler/vm-release.sh <label>` when the
  block ends (also on failure).

## The model-level tests

`DirectCudaPrefillModelTests` runs in the default test lane (`TS_TEST_GGML_BACKEND=cpu`) on any
CUDA machine, in the non-parallel `EngineEnvironmentCollection` (it sets `MAX_CONTEXT`,
`TS_SCHED_DISABLE_BATCHED`, the KV cache dtype and the MoE offload config). Each case writes a
synthetic GGUF of a few MB (`DirectCudaSyntheticModels.cs`, `NemotronSyntheticModelBuilder.cs`)
and loads it on `BackendType.Cuda` and, where it needs a reference, on `ggml_cpu`:

| family | layout it mirrors |
|---|---|
| `gptoss` | gpt-oss-20b: alternating SWA-32 / full attention with sinks (large enough that dropping them moves the rows 5-10x the backends' rounding), head dim 64 group 8, yarn, MXFP4 experts top-4 of 8, K = 288 |
| `qwen35moe` | Qwen3.5-35B-A3B: 3 GDN + 1 gated attention layer, head dim 256 group 8, IQ3_S/IQ4_XS experts top-4 of 16, shared expert |
| `qwen35` | Qwen3.5-0.8B (Stage 0's `qwen08`): dense, every projection BF16, tied head |
| `gemma4moe` / `gemma4moe_global` | gemma-4-26B-A4B with the 32-row sliding window / with only global layers (where default Gemma decode is pinned) |
| `nemotron_h` / `nemotron_h_moe` | Nemotron-H 8B (Mamba2 + attention + ReLU^2) / Lightning-Bolt (Mamba2 + MoE), width 64, Q8_0 |
| `nemotron_h_q4k` / `nemotron_h_moe_mx` | the same layouts at width 256 with the real storage types: Q4_K projections and head (8B Q4_K_M) / MXFP4 up and Q5_1 down experts (Lightning MXFP4_MOE), which width 64 cannot hold |

| test | what |
|---|---|
| `PrefillRows_TrackGgmlCpu` | last prompt row at 17 / 97 / 160 / 257 / 513 / 1100 tokens vs ggml_cpu |
| `DecodeRows_AfterPrefill_TrackGgmlCpu` | 96- and 97-token prefill + 10 decode rows through the default decode path |
| `ChunkedPrefill_ContinuesLikeSinglePass` | five chunkings vs one pass, prompt row + 4 decode rows; bit-identical where the base is |
| `ForwardRefill_TracksSinglePass` | the server's refill path (gpt-oss: 256-row chunks) vs one forward, 600 tokens + 4 decode rows; bit-identical except gpt-oss |
| `RepeatedPrefill_Reproduces` | the same trajectory 3 times on one model at 17 / 64 / 96 / 97 / 130 / 257 / 511 / 513 tokens (prefill-graph plan, capture, replay): runs 2 and 3 bit-identical, run 1's decode rows within 1e-4 (its first decode step runs per-op, later ones replay the decode graph; 7.2e-3 on the sliding-window Gemma file); on Qwen3.5 every GDN layer's conv-ring phase must equal the tokens fed mod 3 after each run (D5) |
| `RepeatedChunkedPrefill_Reproduces` | the same [31, 2, 64] chunking 3 times (a server's repeated chunk shapes), with the same ring-phase check |
| `DefaultDecode_Reproduces` | Gemma default (graph) decode three times, on the local/global and the all-global file (D3 cause 1) |
| `SlidingWindowRing_AfterAPrefillPastTheWindow_HoldsItsLastRows` | two 97- / 257-token prompts sharing their last 32 tokens leave bit-identical layer-0 sliding-window rings, cuda and cpu (D3 cause 2) |
| `SnapshotRestore_ReproducesTheStraightRun` | after a warm-up trajectory: extract the prefix block, reset, inject, continue: bit-identical (cpu rows are the control; D1) |
| `SnapshotBlock_RoundTripsThroughInject` | inject + extract returns the same bytes |
| `Qwen35PrefillGraph_AroundSnapshots_MatchesAFreshModel` | Qwen3.5's prefill graph replayed right after an inject, captured right after an inject or an extract, and a no-inject control: every row bit-identical to a fresh model's (D1's graph hazards) |
| `Qwen35DecodeGraph_CapturedRightAfterInject_MatchesAFreshModel` | the decode-side twin: the first one-token forward after an inject is the decode graph's capture; the capture row within 1e-6 of a fresh model's per-op step (measured 1.3e-7 dense, 0 MoE), every replayed row bit-identical (D1's decode-capture upload) |
| `HostReadsOfTheKvCache_DoNotQueueItForUpload` | Nemotron-H, Mistral 3, Hunyuan dense, Muse-Glimmer (host decode attention) and gpt-oss (host block extract): after every decode step and after a block extract no K/V cache storage is host-dirty, and the extract leaves the next rows bit-identical (the D1 append fix would otherwise re-upload the whole cache every step) |
| `VerifyWindow16_MatchesSequentialDecode` | `SpecForward` 16 rows vs 16 default decode steps |
| `GptOss_LastRowOf16TokenForward_MatchesSequentialDecode` | gpt-oss has no speculative path: the row a 16-token Forward returns vs decode, bit-identical |
| `ConcurrentEngineStreams_MatchSolo` | engine greedy streams, 3 prompts x 24 tokens, solo then together, on files whose argmax provably moves with the history (asserted; the dense Qwen and Gemma files use untied heads for that); a divergence must be a near-tie |
| `ForwardBatch_SequencesKeepTheirOwnState` | Qwen3.5 `ForwardBatch` below the scheduler: 3 sequences together vs each alone (logits rows, within 4.8e-3 dense / 1.1e-2 MoE: route rounding), and each alone again after the batch without a reset: bit-identical (D2) |
| `ForwardBatch_NeighbourContent_DoesNotReachTheMiddleSequence` | the middle sequence of a dense Qwen3.5 batch is bit-identical when only its neighbours' tokens change (D2) |

**Tolerances** are relative L2 of logits rows, derived per family and length the way Stage 0
derived the G3 floors: max(1.5 x the base's distance, the ggml-vs-ggml floor (ggml_cpu whole vs
L/4 chunks)), bit-identity wherever the base is bit-identical. The evidence (and the review's
mutations re-run against them) is in `docs/validation/direct-cuda-prefill/correctness/setup/`.
These tests catch faults; the parity gate is G3a/G3b on real weights.

**Known defects** (rows `[KnownDefectData("Dn", ...)]`: skipped visibly, never passes;
`TS_TEST_RUN_KNOWN_DEFECTS=Dn` or `all` runs them; the descriptions live in
`DirectCudaKnownDefects`). D1-D5 were found on the base revision and are fixed by the
direct-CUDA correctness work (evidence: `docs/validation/direct-cuda-prefill/correctness/`); their
rows are ordinary `[InlineData]` rows now and each fix was mutation-checked against them. D6 is
open:

- **Fixed: D1** (snapshot restore): the K/V append (`CudaKernelOps.TryCopyHeadFirstToCache`)
  treated the cache as write-only, so rows a snapshot inject had written to the host mirror never
  reached the device; it now uploads host-dirty rows first (and aborts a CUDA-graph capture rather
  than recording the upload), and Qwen3.5 makes its graph inputs device-current before a prefill
  replay and before every capture. Host code that only READS a cache (host decode attention,
  block extracts, cache-row copies) checks it out with `Storage.PtrAtElementForRead` /
  `GetFloatPtrForRead`, which does not mark it host-dirty: through `PtrAtElement` every such read
  made the next append upload the whole cache (64 MiB per decode step for an 8-KV-head layer at 8K).
- **Fixed: D2** (Qwen3.5 concurrent streams): the batched path swapped each sequence's host GDN
  conv ring but not the device ring the CUDA kernel reads; each slot now owns a device ring.
- **Fixed: D3/D4** (Gemma-4 sliding-window decode): a single all-ones norm vector was freed in the
  middle of a decode-graph capture (replays read scratch as norm weights), and a prefill chunk
  longer than the window raced its rows into the ring; one persistent vector per width, and only
  a chunk's last window of rows is written. D4's 257-token row was an ill-conditioned prompt, not
  a defect: that row now uses another prompt salt (`PrefillSalt`).
- **Fixed: D5** (Qwen3.5 prefill graph): a prefill-graph replay did not advance the host GDN
  conv-ring phase, and an aborted capture advanced it twice for the layers it reached.

The original descriptions follow.


- **D1** a K/V (+ Mamba2/GDN state) snapshot restored into an empty direct-CUDA cache does not
  reproduce the straight run (all six families; managed cpu and ggml_cpu are bit-exact). The block
  itself round-trips through inject + extract byte for byte, so the fault is in what the next
  forward reads after an inject.
- **D2** Qwen3.5 (MoE and dense) on direct CUDA: concurrent engine streams leave their solo twins
  in the first decode steps by far more than a near-tie (managed cpu matches). With `TS_SCHED_DISABLE_BATCHED=1` the per-sequence path still diverges; that path
  swaps sequences through `TryExtractKVBlock`/`TryInjectKVBlock` (`BatchExecutor`), i.e. D1's
  mechanism, so D2 probably shares D1's root cause.
- **D3** Gemma-4 with sliding-window layers on direct CUDA: decode is off (0.07-0.38 from cpu and
  ggml_cpu, which agree to ~5e-3) and not run-to-run deterministic, even inside the window. This
  is NOT confined to graph replay: with the real 32-row window the per-op decode (`ForceUnfused`)
  is nondeterministic too (1/13 rows repeat, 0.1-0.29 from cpu), which points at stale or
  uninitialised sliding-window ring reads. An all-global file is deterministic and tracks both
  references.
- **D4** Gemma-4 sliding-window rows: verify-shaped and decode-shaped rows differ by ~6%; at 257
  tokens the three implementations disagree on the prefill row (see the evidence).
- **D5** Qwen3.5 direct-CUDA prefill graph: repeating a prefill of the same length (or chunk shape)
  leaves the GDN conv-ring phase out of step whenever the length is not a multiple of 3 (up to the
  512-token graph limit): the MoE file goes wrong on the capture run, the dense BF16 file on the
  replay run (Stage 0's qwen08 `chat512`).
- **D6** Nemotron-H MoE on direct CUDA, single forwards of more than 512 rows: the last prompt row
  can land far from both references (Q8_0 experts: 0.33 at 1100 tokens; MXFP4/Q5_1 experts: 1.05 at
  513) while the decode rows after it are normal; in 512-row chunks the rows are fine.

Run them with
`dotnet test InferenceWeb.Tests -c Release --filter "FullyQualifiedName~DirectCudaPrefillModelTests|FullyQualifiedName~CudaPrefillModuleTests|FullyQualifiedName~CudaStorage_ReadOnlyHostCheckout"`,
and the real-model checks with `TS_TEST_MODEL_DIR` pointing at a directory holding
`Qwen3.5-0.8B*`, `Qwen3.5-9B*` or `gemma-4-E4B*` GGUFs.

**Building a fresh worktree** without rebuilding ggml (~40 min): the native sources
(`TensorSharp.GGML.Native`, `ExternalProjects/ggml`) must be identical to an already-built tree's.
Hash them on both sides, copy `TensorSharp.GGML.Native/build-windows` (or `build`),
`ExternalProjects/` and `TensorSharp.Backends.GGML/obj/<platform>/Release/ggml-native.{config,stamp}`,
then touch the stamp; MSBuild then logs `Skipping target "BuildGgmlNativeWindows"`. A tree that
only has a copied `GgmlOps.dll` (no `ExternalProjects/ggml`) must build with
`-p:TensorSharpSkipGgmlNative=true`, or the build starts a full native build.
