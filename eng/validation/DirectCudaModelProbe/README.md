# Full-model CUDA validation probe

This fresh-process probe loads a real GGUF through `ModelBase`, measures fixed
teacher-forced prefill/decode, then performs a separate untimed replay capturing
every full logit row. It also checks complete greedy answers to arithmetic,
extraction and a square-list prompt through the checkpoint's no-thinking chat
template. These small semantic checks do not establish broad language quality.

Build after updating native code and PTX, using an unchanged upstream ggml tree:

```powershell
$env:TENSORSHARP_GGML_NO_UPDATE = '1'
dotnet build eng/validation/DirectCudaModelProbe -c Release -p:TensorSharpSkipGgmlNative=true
```

Only use `TensorSharpSkipGgmlNative` after independently rebuilding native code
when native sources changed. A direct CUDA build uses `nvcc` on PATH to compile
current kernels for the local GPU. Verify that the loaded module includes the
new entry point and supports the tested GPU; falling back to an older module
does not validate fusion. The report hashes copied PTX files and assemblies.

## Optimization switches and coverage

Both switches are read once per process; start a new process for each arm.

| Switch | Default | Reference / candidate |
| --- | --- | --- |
| `TENSORSHARP_CUDA_MOE_FUSION` | Enabled | `0` restores separate activation/quantization launches and separate shared-MoE scratch allocations; `1` enables fusion and scratch reuse. |
| `TS_Q4E_PREFILL_COMBINE` | Enabled | `0` uses the original ggml Qwen4Exp combine; otherwise the TensorSharp-owned CUDA prefill combine runs where eligible. |

The ggml reduction is enabled only for CUDA token spans with 64–65,535 tokens
and at least one resident expert layer. The 64-token deployment cutoff follows
full-engine measurements; it is not a claim of an optimal threshold on every
device. Smaller prompts and smaller final prefill chunks retain their original
graph even when the switch is `1`. The standalone helper's numerical tests also
exercise 9–63-token shapes, without enabling them in model execution.

Direct CUDA fusion covers the quantized expert SwiGLU paths in the shared
whole-model MoE code, Dsv4, and generic Qwen decode/batched execution. Scratch
reuse applies only to `CudaMoeScratch`, used by the GLM and Qwen4Exp engines:
gate/up and down activation buffers share capacity because their lifetimes do
not overlap, and the interleaved/split layouts are mutually exclusive. Dsv4's
separate scratch and generic Qwen3.5/3.6 scratch are unchanged. The direct CUDA
MMA prefill path retains its existing float activation operation.

The committed PTX includes the fused entry point, regenerated with CUDA 12.8.93
while preserving PTX 8.7 and the existing `sm_120` target. A build without `nvcc`
can use this bundled fusion on compatible hardware and drivers. SM120 was
cross-compiled only; it was not executed on the RTX 3080 Laptop GPU (SM86).
Compile with `nvcc` on PATH to produce PTX for the local architecture and deploy
the resulting `cuda_kernels` files; the deployed application itself does not
need a compiler. Older deployed modules that lack the entry point still safely
select the original two launches, with shared-MoE scratch reuse retained.
Focused fusion tests and the `--moe-fusion` benchmark explicitly reject modules
without the fused entry point.

Coverage includes direct split-layout quantization parity, per-slot experts,
grouped warp experts, and MMA prefill followed by decode. It does **not** include
a complete staged-expert dispatch: the current staged width/type requirements
are a subset of the earlier MMA branch, so that staged branch is unreachable.

An available 11.8 GB Qwen3.6 MoE checkpoint supports the resident direct CUDA path
on the 16 GB RTX 3080 Laptop GPU used for this work. Its expert down-projection
types are IQ3_XXS, IQ4_XS and Q3_K, which do not select the generic Qwen MoE DP4A
down path changed by this fusion. This checkpoint is a trained-model regression
control; timing differences on it do not establish fusion gains. A checkpoint's
filename quantization label does not identify each expert tensor's type. Inspect
the actual down-projection types, supported dimensions and selected dispatch
before claiming engagement. Generic Qwen fusion requires a supported DP4A down
projection (Q4_0, Q4_K, IQ2_XXS or IQ2_S), with its policy enabled and valid scratch.
Verify available memory and actual placement in each run. Run baseline and optimized
arms in separate processes, alternating arm order across independent repetitions:

```powershell
$env:TENSORSHARP_CUDA_MOE_FUSION = '0'
dotnet eng/validation/DirectCudaModelProbe/bin/Release/net10.0/DirectCudaModelProbe.dll --model C:/Works/models/mtp/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf --backend cuda --prefill-tokens 64 --decode-tokens 64 --warmup 2 --iterations 5 --max-new 128 --output artifacts/strata-perf/baseline-r1.json
$env:TENSORSHARP_CUDA_MOE_FUSION = '1'
dotnet eng/validation/DirectCudaModelProbe/bin/Release/net10.0/DirectCudaModelProbe.dll --model C:/Works/models/mtp/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf --backend cuda --prefill-tokens 64 --decode-tokens 64 --warmup 2 --iterations 5 --max-new 128 --output artifacts/strata-perf/fused-r1.json
```

The probe accepts `--backend ggml_cuda` and `--backend ggml_cpu`, and optional
`--moe-cpu-layers N` (or `all`) sets explicit expert offload before model loading.
For Qwen3.8 Flash Next on this host, `--moe-cpu-layers 47` leaves one of its 48
expert layers on the GPU. All-host expert placement does not exercise a resident
expert reduction kernel. Confirm actual placement and kernel engagement in the
retained process log and any implementation diagnostics.

Use `--synthetic-q4e artifacts/strata-perf/fixture.gguf` instead of `--model` to
generate a deterministic eight-layer Qwen4Exp engine fixture. This automatically
disables quality generation. It supports `--synthetic-q2kxl true`,
`--synthetic-experts N`, `--synthetic-experts-used K` and
`--synthetic-indexer-top-k N` (defaults: false, 16, 4 and 16). Both arms must use
the same settings and the recorded fixture checksum must match. This mode
exercises full engine loading and execution but makes no trained-model claim.

`--quality false` skips semantic generation and explicitly reports it as
unchecked. `--max-context` defaults to 1024. The probe clears speculative decoding
switches to hold the generated work constant. Timing excludes logits serialization
and semantic checks; each `Forward` returns host logits, so the timed token work
has completed before its time is recorded. Timed repeats include the same prompt
and forced tokens, following two warmup repeats by default.
Forced decode traverses the token pool with step 37 and offset 5. Step 37 is
coprime to the observed 17-token trained-model pool and the 251-token synthetic
pool, providing varied inputs. Exact prompt and forced token IDs are recorded
and must match across arms. Older diagnostic reports used step 17 and remain
separate; their timings cannot be mixed with this workload.

JSON fields for an A/B runner:

- `runs`: warmup flag, prefill/decode token counts, milliseconds, token rates and
  final-logit hashes. Compare medians of non-warmup repetitions and preserve all
  individual samples. Match token counts before claiming a ratio.
- `logits`: absolute binary path, SHA-256, row count, vocabulary columns and
  `little-endian-float32` format. The first row is prefill; subsequent rows are
  one forced token each. Compare complete matrices, argmax and relative errors.
- `repeated_final_logits_exact`: checks every measured final state and the
  separate capture replay. A failure returns a nonzero exit status. Warmup hashes
  remain in `runs` and `warmup_final_logit_sha256`, with
  `warmup_final_logits_match_measured` reported separately. Compare corresponding
  warmup hashes across A/B arms as well. This does not claim all-lifecycle bitwise
  equality: the existing Qwen3.6 direct CUDA path can produce a different final
  hash during its first graph-capture execution, in both baseline and optimized
  arms. Its graph cache captures on the second use and persists across resets.
  Two warmups allow subsequent measured repeats and capture replay to use the
  stable execution state; the warmup discrepancy is retained rather than hidden.
- `quality`: rendered-input token IDs, generated IDs, full selected answer,
  EOS completion and independent answer checks. Quality failures return nonzero.
- `memory`: process working-set/private-byte snapshots and peak working set.
  Direct CUDA additionally reports driver free/total/used dedicated memory and
  existing allocator pool counters. Device memory is affected by the desktop,
  WDDM and other processes; these are not exact per-process peak live allocations.
- `moe_scratch`: whole-model CUDA engines expose an existing internal scratch
  object through diagnostic reflection. Its actual unique tensor-storage byte
  lengths and alias groups are recorded without allocating extra device memory.
  Generic Qwen3.5/3.6 CUDA models use a different path and have an empty list here.
- `managed_assemblies_sha256`, `ptx_sha256`, `native_sha256`: actual build identity.
  Native hashes cover loaded `GgmlOps` modules. The invoking runner must record
  repository/upstream revisions, clean upstream status, device and compiler.
- `model_files`: ordered path, length and timestamp for each already-open
  `GgufFile.FilePaths` entry. `model_file_count` must match the declared split
  count. `model_file_identity_incomplete` is true if actual loaded paths are
  unavailable or incomplete; then `model_total_file_bytes` is null, so a partial
  list cannot be mistaken for the complete checkpoint size. The existing
  `model_bytes` and `model_last_write_utc` refer only to the selected input file.
  In particular, a metadata-only first shard can be much smaller than its model.
  Paths, sizes and timestamps identify files without hashing weight payloads.
  Generated fixtures additionally have a full payload SHA-256; their rewritten
  paths and timestamps may differ between processes.

Keep reports, binaries, logs and captured logits in ignored `artifacts/` or
`docs/validation/`. This probe does not start a server or measure HTTP throughput.

Compare equal numbers of fresh reference and candidate processes with the
same binary and checkpoint. Set `TENSORSHARP_CUDA_MOE_FUSION=0` for the direct
CUDA reference and `1` for the candidate. For the ggml Qwen4Exp prefill combine,
use `TS_Q4E_PREFILL_COMBINE=0` and `1`. Alternate process order between pairs;
finish compilation and unrelated tests before recording timing comparisons.
The runner may set both feature switches to the arm's value; only the relevant
backend's switch changes execution. All other captured environment controls
must remain equal, as must measured and warmup iteration counts.

```powershell
python eng/validation/compare-strata-perf.py --before docs/validation/before1.json docs/validation/before2.json --after docs/validation/after1.json docs/validation/after2.json --output docs/validation/comparison.json
python -m unittest discover -s eng/validation/tests -p test_compare_strata_perf.py
```

The comparator verifies capture lengths and checksums, workload and loaded
binary identities (including source GGUF dimensions), explicit switch values
and effective CUDA policy, exact full logit matrices, repeated
final logits and complete semantic token streams. It compares the median of
each process's measured medians and fails latency regressions over 5% by default.
When shard identity fields are present, it rejects incomplete or mismatched
identities, including changes to later shards. Legacy reports lacking these
fields remain supported and explicitly report that separate shard provenance is
required for split checkpoints; do not mix old and new identity schemas in one
comparison. Existing run binaries should remain fixed within an A/B comparison.
A passing comparison alone does not establish a speedup: inspect its signed
changes and individual process medians, and report near-zero changes as noise.
