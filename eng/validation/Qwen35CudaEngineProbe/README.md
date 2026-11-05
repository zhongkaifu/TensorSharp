# Qwen3.5/3.8 direct CUDA numerical probes

Build the probe with a CUDA 13 toolkit and SM12 device for the NVFP4 Qwen3.8-27B
engine. Record the actual source/model revisions, model hashes, dependency state
and binary hashes with run evidence. Generated evidence belongs in ignored
`artifacts/` or `docs/validation/`; unavailable devices/models are not passes.

```sh
dotnet build eng/validation/Qwen35CudaEngineProbe -c Release -p:CudaArch=compute_120
export TS_TEST_QWEN35_MODEL=/models/Qwen3.8-27B-NVFP4-MTP-N4_0.gguf
export TS_TEST_QWEN35_DRAFT=/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

## DFlash CUDA graph equivalence

`TS_Q35_CUDA_GRAPHS=1` opts into lane-shaped DFlash CUDA graph submission. The
default remains ordinary launches until measured quality/performance qualify a
change. The flag is read when the engine is created. Metadata uploads, pointer
table refresh, synchronization and host readback remain outside the graph.

Run sequentially on an otherwise idle GPU:

```sh
probe=eng/validation/Qwen35CudaEngineProbe/bin/Release/net10.0/Qwen35CudaEngineProbe.dll
TS_Q35_CUDA_GRAPHS=0 dotnet "$probe" --graph-trace artifacts/q35-graphs/off
TS_Q35_CUDA_GRAPHS=1 dotnet "$probe" --graph-trace artifacts/q35-graphs/on
cmp artifacts/q35-graphs/off/trace.json artifacts/q35-graphs/on/trace.json
```

The trace exercises lanes 1, 2 and 4, reordered lane bindings, growing token
positions, a released/replaced slot, and a reset followed by prefill. It records
proposal tokens, confidence float bits, accepted prefixes and full-vocabulary
draft/verify logit SHA-256 values. Drafting is repeated before committing to
check that first capture and later replay do not mutate slot state. Requested
graph runs must report three successfully captured lane shapes; capture
fallbacks do not satisfy coverage. Both processes must complete successfully,
write `run.json` with `complete: true`, and produce identical `trace.json`.

Full binary logit snapshots are retained as `round-NNN.draft-a.f32`,
`round-NNN.draft-b.f32` and `round-NNN.verify.f32` (little-endian F32). These can
consume hundreds of MB per run. Non-finite logits fail the probe. A repeated
draft mismatch writes `repeat-failure.json` with the differing element count,
maximum absolute difference, relative L2, proposals and confidence bits, then
fails without writing a completed trace. Preserve these snapshots when
investigating a failure.

The original drafter GEMM can use atomic split-K accumulation. Its floating
point summation order is not guaranteed, so a graph-off A/A failure is possible
before any graph optimization is applied. Such a failure is evidence of
unqualified numerical determinism, not a graph equivalence pass; retain the
failure metrics and do not label near-equal hashes as bit-identical. Independent
speed benchmarks must avoid the diagnostic full-logit host copies.

Other probe modes are described at the top of `Program.cs` (teacher-forced
ggml comparison, slot cloning, batched prefill, verify/commit and vision).

## Greedy verify host transfer equivalence

`TS_Q35_GREEDY_VERIFY=1` opts into returning only row argmax ids for a solo
speculative verify. The default is off. The executor requests it only after
the first output token, for plain greedy sampling with no grammar, penalties,
unopened thinking-end masking or full-logit diagnostics. Each acceptance row
still applies forced thinking transitions against the updated committed
history. Plain steps and all existing full-logit APIs keep returning logits.
This changes host transfer and selection only; model kernels and precision
remain the same. GPU argmax follows `TokenSampler`'s leading-NaN/first-max
semantics, including signed-zero and infinity ties.

```sh
dotnet test InferenceWeb.Tests --filter 'FullyQualifiedName~Qwen35ArgmaxTests|FullyQualifiedName~SpeculativeExecutionTests'
TS_Q35_DRAFT_SPLITK=0 TS_Q35_GREEDY_VERIFY=0 dotnet "$probe" --greedy-verify-trace artifacts/q35-greedy/off
TS_Q35_DRAFT_SPLITK=0 TS_Q35_GREEDY_VERIFY=1 dotnet "$probe" --greedy-verify-trace artifacts/q35-greedy/on
cmp artifacts/q35-greedy/off/trace.json artifacts/q35-greedy/on/trace.json
```

Both traces must finish with `complete: true`; the enabled run must report
actual token-only verifies. The probe compares every device id with the
actual CPU sampler over full vocabulary logits, varying windows 2/3/4/8,
growing positions, partial/full acceptance, a forced thinking budget,
executor re-arming on retained device carry and a later prompt turn. The
full-logit oracle runs first and the token-only verify runs last, leaving
candidate state for commit and the following draft/decode. Trace hashes of
full logits and emitted tokens must match across the two processes.

This probe intentionally downloads oracle logits and repeats GPU work; its
timings are not benchmarks. Use the ordinary engine benchmark with the flag
off/on to measure the real transfer saving. Keep all other optimization
flags, model files, request/sampling settings and dependency revisions fixed.
CUDA/model skips do not qualify device correctness or performance.

## Target Verify CUDA graph equivalence

`TS_Q35_VERIFY_GRAPHS=1` separately opts into capturing the target's layer loop
for a single ordinary Verify group whose attention uses the split-key path.
Prefill, mixed steps, Decode, embeddings, metadata transfers, position updates,
commit/injection and the LM head retain their existing submission path. The
cache key contains lanes, window, attention chunk/split geometry and a feature
layout/drafter generation. At most 32 Verify graphs are retained.

```sh
TS_Q35_VERIFY_GRAPHS=0 dotnet "$probe" --verify-graph-trace artifacts/q35-verify/off
TS_Q35_VERIFY_GRAPHS=1 dotnet "$probe" --verify-graph-trace artifacts/q35-verify/on
cmp artifacts/q35-verify/off/trace.json artifacts/q35-verify/on/trace.json
```

This mode deliberately loads no drafter. It uses prescribed real-text tokens
and compares every vocabulary logit bit and argmax on repeated fixed-input
Verify calls. Default attention-key boundaries are 127, 128, 129, 255, 256,
257, 8192, 8193, 16384 and 16385. It also covers cloned prefix blocks and COW,
partial commit, reordered slots, all 64 lane/window combinations from 1..8,
LRU eviction and recapture, a freed slot restored from pinned host memory,
reset/refill versus a fresh slot, and feature-layout invalidation/growth.
Both runs must finish, write `complete: true`, and have identical `trace.json`.
Graph-on additionally requires actual captures, replays and an eviction;
capture fallback does not satisfy coverage.

An optional third argument replaces the boundary list, including long contexts
when the model/device memory budget permits them. For example:

```sh
TS_Q35_VERIFY_GRAPHS=1 dotnet "$probe" --verify-graph-trace \
  artifacts/q35-verify/long-on 131072,131073,250000 --snapshots
```

`--snapshots` writes the full first Verify logits of each check as little-endian
F32. Without it, every full logit array is hashed, and a repeated-logit failure
always saves both arrays and an incomplete failure record. Long traces may
produce several GB of snapshots per arm. These checks establish execution
equivalence for their inputs, not long-context retrieval quality.

With the real DFlash drafter attached, also run `--graph-trace` in independent
processes with `TS_Q35_VERIFY_GRAPHS=0/1` and the same draft settings. Preserve
production-default A/A results separately. `TS_Q35_DRAFT_SPLITK=0` disables
atomic draft split-K for a controlled fixed-summation diagnostic; it is a
different configuration and must be recorded as such.

```sh
TS_Q35_DRAFT_SPLITK=0 TS_Q35_CUDA_GRAPHS=1 TS_Q35_VERIFY_GRAPHS=0 \
  dotnet "$probe" --graph-trace artifacts/q35-verify/draft-control
TS_Q35_DRAFT_SPLITK=0 TS_Q35_CUDA_GRAPHS=1 TS_Q35_VERIFY_GRAPHS=1 \
  dotnet "$probe" --graph-trace artifacts/q35-verify/draft-target-graphs
cmp artifacts/q35-verify/draft-control/trace.json \
  artifacts/q35-verify/draft-target-graphs/trace.json
```

The Verify-on arm must exercise actual capture and reuse. This comparison
covers target DraftTap, commit/injection and the next draft as well as target
logits. The no-drafter `--verify-graph-trace` additionally reads back every
selected feature row, hashes its F32 bits in both arms, and checks repeated
feature values while changing the feature layout. Target logits alone cannot
qualify feature capture. `--snapshots` includes the selected feature arrays.

## Prepared GDN prefill

`TS_Q35_GDN_PREPARED_PREFILL=1` opts into preparing input-only Q/K packets and
GDN control matrices once for reuse by the four value tiles. It retains the
legacy BF16 high/residual decomposition, three MMA terms, F32 state, `expf`,
prefix/inverse arithmetic and state accumulation order. The original kernel
remains available as the strict numerical oracle. Decode and speculative
Verify use their existing kernels. Only non-Verify prefill groups with at least
32 rows per sequence use preparation. The default is off and reserves no
prepared scratch. At a 4096-token ubatch, packet capacity is about 96.1 MiB on
the Qwen3.8-27B shape and is reused across layers and sequence groups. When the
unused tail of the existing projection workspace fits those packets, the engine
uses an aligned alias there, without an additional device allocation. Otherwise
it allocates the packet workspace separately. Record the actual
`prepared_gdn_projection_tail_reused` field in the prefill trace: neither memory
savings nor alias-path validation may be inferred from the flag alone. These
kernels and allocations belong to TensorSharp; ggml sources are unchanged.

First run the isolated CUDA gates, on an idle device:

```sh
dotnet test InferenceWeb.Tests -c Release \
  --filter FullyQualifiedName~Qwen35PreparedGdnTests \
  --logger 'console;verbosity=detailed'
```

The correctness gate compares every output/state float bit against the legacy
kernel, including repeated/continued state and finite boundary inputs. Its
microbenchmark includes both preparation kernels and the consuming recurrence;
it reports GPU and wall time without a speed pass threshold. A CUDA skip or an
uncompiled module is not a pass. Any numerical mismatch disqualifies the
candidate; do not replace bit equality with a tolerance.

Then compare full-model prefill and subsequent teacher-forced decode logits:

```sh
TS_Q35_GDN_PREPARED_PREFILL=0 dotnet "$probe" --prefill-trace \
  artifacts/q35-gdn/off 31,32,33,127,128,129,511,512,513,1024,2049,4096 4096
TS_Q35_GDN_PREPARED_PREFILL=1 dotnet "$probe" --prefill-trace \
  artifacts/q35-gdn/on 31,32,33,127,128,129,511,512,513,1024,2049,4096 4096
cmp artifacts/q35-gdn/off/trace.json artifacts/q35-gdn/on/trace.json
```

The probe saves full-vocabulary F32 logits after each prefill and eight fixed
continuation tokens, repeats after reset, and includes variable-length padded
batched prefill. It loads no drafter. The opt-in arm must record actual prepared
launches. Both processes must complete and produce identical traces. These
checks establish numerical equivalence for their inputs, independently of the
separate answer-quality and long-context retrieval evaluations.

## Mixed prefill/Verify at full row capacity

`--mixed-capacity-trace <fresh-output>` loads one Qwen3.8-27B model with no
drafter and ubatch 4096. It uses the primary slot plus eight additional slots:
each first receives a fixed seven-token prefix, then one mixed call submits
4096 prefill rows and eight Verify lanes of eight rows. Thus all 4160 rows of
`MixedRowCapacity` and nine of sixteen slot-table entries are exercised.
The Verify lanes commit respectively 1..8 rows, then all nine slots receive
eight fixed teacher-forced batched decode steps. Reset repeats that exact
schedule for a full-bit A/A check. This does not compare separate and mixed
forwards, whose projection shapes and split-K numerics can differ.

Build managed code and check the plan without a model, CUDA calls or GPU:

```sh
dotnet build eng/validation/Qwen35CudaEngineProbe -c Release --no-restore \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true \
  -p:UseSharedCompilation=false -nodeReuse:false --disable-build-servers
probe=eng/validation/Qwen35CudaEngineProbe/bin/Release/net10.0/Qwen35CudaEngineProbe.dll
dotnet "$probe" --mixed-capacity-trace artifacts/q35-mixed/plan --plan-only
```

The plan must say `PLAN_ONLY_GPU_NOT_RUN`; its geometry and reflection checks
are CPU evidence only. Run the two actual arms sequentially, on an exclusively
leased GPU, with the same source, binary, model and other flags. Output paths
must not already exist. If running outside the checkout, set
`TS_PROBE_SOURCE_ROOT` to that checkout. These commands keep production split-K:

```sh
export TS_TEST_QWEN35_MODEL=/models/Qwen3.8-27B-NVFP4-MTP-N4_0.gguf
export TS_Q35_GEMM_SPLITK=1 TS_Q35_ATTN_PREFILL=1 TS_Q35_DRAFT_SPLITK=1
export TS_Q35_CUDA_GRAPHS=0 TS_Q35_VERIFY_GRAPHS=0
export TS_Q35_GREEDY_VERIFY=0 TS_Q35_TOPK_PARTITIONED=0
TS_Q35_GDN_PREPARED_PREFILL=0 dotnet "$probe" --mixed-capacity-trace artifacts/q35-mixed/off
TS_Q35_GDN_PREPARED_PREFILL=1 dotnet "$probe" --mixed-capacity-trace artifacts/q35-mixed/on
```

The last Head inside `MixedVerify` computes all 64 Verify logits but normally
returns only their IDs. This probe immediately synchronizes the engine stream
and reads that existing private `_logits` allocation before Commit or another
forward. It does not recompute Head or modify production APIs. Exact private
field/type and internal D2H-method signature checks fail closed if that layout
changes. Source and loaded assembly SHA values are recorded; external build
provenance must establish that the recorded source produced those assemblies.
This reflection diagnostic is deliberately limited to the reviewed implementation.

Each arm saves 38 complete F32 records (146 full vocabulary rows per repeat),
18 I32 drawn-token files, input arrays/hashes and exact positions. For a 248320
vocabulary this is about 277 MiB of F32 snapshots per arm. Non-finite logits,
missing dispatches, mismatching CPU first-max IDs or any reset A/A byte
difference fail without relaxing a tolerance. Raw mixed outputs are persisted
before those checks. Enabled runs must actually use the projection-tail alias
and launch one prepared pass per recurrent layer per mixed call. The public
prepared-launch counter is observed; other dispatch routes are explicitly
source-derived rather than presented as kernel-profiler measurements.

Check both completed manifests and compare every saved output byte:

```sh
python3 - artifacts/q35-mixed/off artifacts/q35-mixed/on <<'PY'
import hashlib, json, pathlib, sys
a, b = map(pathlib.Path, sys.argv[1:])
r = [json.loads((p / 'run.json').read_text()) for p in (a, b)]
assert all(x['complete'] is True and x['status'] == 'COMPLETE_STRICT_RESET_AA' for x in r)
assert [x['flags']['TS_Q35_GDN_PREPARED_PREFILL'] for x in r] == ['0', '1']
assert {k:v for k,v in r[0]['flags'].items() if k != 'TS_Q35_GDN_PREPARED_PREFILL'} == {k:v for k,v in r[1]['flags'].items() if k != 'TS_Q35_GDN_PREPARED_PREFILL'}
assert r[0]['definition'] == r[1]['definition'] and r[0]['model'] == r[1]['model']
for key in ('source_files', 'engine_assembly', 'probe_assembly'):
    assert r[0]['provenance'][key] == r[1]['provenance'][key]
inputs = [(p / 'inputs.json').read_bytes() for p in (a, b)]
assert inputs[0] == inputs[1] and all(hashlib.sha256(x).hexdigest() == y['inputs_sha256'] for x,y in zip(inputs,r))
assert (a / 'trace.json').read_bytes() == (b / 'trace.json').read_bytes()
trace = json.loads((a / 'trace.json').read_text())
keys = {f'prefix-{i}' for i in range(9)} | {'mixed-prefill', 'mixed-verify'} | {f'teacher-{i}' for i in range(8)}
assert len(trace) == 38 and {(x['repeat'], x['key']) for x in trace} == {(i,k) for i in range(2) for k in keys}
for x in trace:
    for name, digest, size in ((x['file'], x['sha256'], x['bytes']), (x['drawn_file'], x['drawn_sha256'], x['rows'] * 4)):
        if name is None: continue
        lhs, rhs = (a / name).read_bytes(), (b / name).read_bytes()
        assert len(lhs) == size and hashlib.sha256(lhs).hexdigest() == digest and lhs == rhs
print('PASS: completed mixed-capacity reset A/A and fresh-process off/on full-output bits')
PY
```

Do not label an incomplete arm, a failed A/A, or a CPU-only plan as a GPU pass.
This mode is a capacity/continuation correctness diagnostic and makes no
performance, arbitrary-partition equivalence or answer-quality claim.


### Shared-scratch allocation and full mixed-capacity equivalence

`--shared-scratch-trace <fresh-output> [--plan-only]` reuses the complete
`--mixed-capacity-trace` schedule: no drafter, 4160 rows, nine slots, 38 full F32
records / 292 vocabulary rows, 18 drawn-I32 records, partial commits, teacher
continuations and two exact reset repetitions. The existing CLI and its raw
record definition retain their meaning. Use four separate processes with
`TS_Q35_SHARED_SCRATCH=0/1` crossed with `TS_Q35_GDN_PREPARED_PREFILL=0/1`.
Keep the other feature opts off and production draft split-K unset.

The new mode additionally writes `shared-scratch-post-load.json` and
`shared-scratch-post-workflow.json`. They record the actual shared backing pointer,
requested bytes / avoided allocation bytes, all ten published and independently
read private pointers, logical / padded extents, phase ranges, ownership and
excluded persistent / quantized buffer pointers. The four `_v*` verification
buffers are allocated lazily by `EnsureVerifyBuffers`: zero before the first
verify and separate allocations afterwards is expected. The other nine excluded
pointers remain fixed. Ownership observations and driver queries in this probe
cover the ten scratch views; excluded-buffer allocation ownership is established
from the allocation source, not an additional recorded driver query.
The phase names describe reviewed
source liveness, not GPU-profiler timestamps. The probe rejects overlap within
one phase, misalignment, out-of-owner views, separately owned interior views and
unexpected pointer changes. If `cuMemGetAddressRange_v2` is available it also
checks actual driver allocation bounds; an unavailable symbol is explicitly
recorded and supplies no passing driver-bound coverage.

CPU `--plan-only` checks reflection and layout, including unsafe overlap,
alignment and owner-bound fixtures. It never loads a model or initializes CUDA.
Passing that plan is not passing the four GPU arms. The coordinator must retain
and compare every raw F32/I32/input/trace byte across all four arms, require
the shared allocation to be genuinely enabled, and observe 96 actual prepared
launches in each prepared arm. This is a fixed-schedule numerical and allocation
gate, not a performance or general model-quality result. Recurrent state is
exercised through commit, positions and teacher continuation; this workflow does
not separately dump all recurrent-state device memory.

Build the probe into a fresh executable directory, then overlay the exact latest
reviewed Host managed DLLs and validated native runtime. Record every copied SHA,
the actually loaded engine/probe assembly SHAs, full Host and runtime inventories,
model / source / unchanged upstream ggml identities before and after. Keep raw
snapshots on persistent ignored storage (about 1.16 GB for all four 27B arms);
do not confuse a current source checkout with proof of how a Host was built.

### Native model memory-pressure integration

Both memory switches default off and are selected before model construction:

- `TS_Q35_SHARED_SCRATCH=1` enables the shared phase allocation. For the tested
  27B geometry with 4160 scratch rows, requested allocation drops by 461660160
  bytes (440.273 MiB). This is requested storage saved, rather than a measurement
  of whole-device free memory.
- `TS_Q35_RECLAIM_VISION_POOL=1` lets `Qwen35CudaModel` drain returned, idle vision
  allocator blocks when the native engine needs memory. The callback attempts
  the media lock without waiting, synchronizes the allocator's stream, and
  preserves live allocations. Successful trims log cached bytes before and after;
  an observed positive reduction is required to count the path as exercised.

The switches preserve weights, KV/state formats, arithmetic and the device
reserve. Evaluate real image requests and DFlash long contexts separately from
this no-drafter layout probe. Concurrent multimodal comparisons have observed
per-request answer regressions; the answer-quality gate for these opt-ins remains
unaccepted in that workload. The separate greedy-verification opt-in also remains
off by default.
