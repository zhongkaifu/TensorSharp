# Paired original-NInfer decode graph probe

This separate tool uses only the public `Qwen35CudaExecutor` and `Q35CudaEngine`
model APIs. It creates one graph-OFF engine, disposes it, then creates one
graph-ON engine with the same artifact, inputs, native modules, A16 option, repack
option, context and ubatch. It changes only `TS_Q35_NINFER_DECODE_GRAPHS` and
restores its original process value on exit. No serving defaults are changed.

```sh
dotnet build eng/validation/NinferDecodeGraphProbe/NinferDecodeGraphProbe.csproj \
  -c Release -p:TensorSharpSkipCudaPtx=true
dotnet eng/validation/NinferDecodeGraphProbe/bin/Release/net10.0/NinferDecodeGraphProbe.dll --self-test
```

The managed-only build needs previously prepared backend assets; it does not
compile or replace native kernels. No arguments prints PLAN. `--self-test`
checks the fixture plan, manifest validation, output guard detection and refusal
of graph fallback, with **no model load or CUDA calls**.

Execute only on the operator's otherwise idle GPU, from an immutable runtime.
Use a fresh **local POSIX** ignored evidence directory, preserving all failed
attempts. The original artifact and input pins below come from the previous
body probe; the tool checks the actual directory and input SHA values.

```sh
dotnet /ABSOLUTE/IMMUTABLE_RUNTIME/NinferDecodeGraphProbe.dll \
  --execute \
  --artifact /workspace/hf/ignis/qwen3_8_27b_nvfp4full-v2.ninfer \
  --expect-directory-sha 222b7540e961a6455fff9f03cce451860e4d248785a4ef0a805f538c8eb82658 \
  --inputs /ABSOLUTE/ORIGINAL_BODY_PROBE/inputs.json \
  --expect-inputs-sha 4e07bc77f9fe75f9b23867b045fda6acd11ba4cf01f751ebfec7b7e96ce5c883 \
  --context 16384 --ubatch 512 \
  --out /root/ts-eval/artifacts/ninfer-decode-graphs-r1
```

Keep any chosen `TS_Q35_NINFER_A16_PTX` identical in both modes. For a strict
legacy-math graph qualification, leave it unset or select the independently
qualified exact-legacy A16 module. This tool does not select an A16 arithmetic
policy. An explicitly provided `TS_Q35_NINFER_REPACK_PTX` is also preserved in
both modes and its file/loaded-image identities and conversion counts are
recorded. Failed/unqualified GDN chunk and host embedding options are refused
to isolate this graph check. Context must be 8196..262144, ubatch 256..2048;
all fixed lane/position scenarios are mandatory.

The thirteen fixtures cover lanes **1, 2, 3, 4, 8, 16** near positions 127 and
255, with different lane histories. Three `DecodeBatch` steps cross the
128-key attention split and 256-token KV-block boundaries. A further one-lane
case starts at 8191 and crosses the attention chunk-doubling boundary. Complete
derived token arrays are retained: they are deterministic cycles of the pinned
original prompt/continuation token streams, never model-generated test input.

For each fixture, the tool resets and prefills the slots, runs three decode
steps, clones a slot and checks identical next-token logits after copy-on-write.
It then runs a **NUbatch-row prefill after graph capture**, frees that temporary
slot, resets the original slots and repeats the same prefill/decode/clone
sequence. This stresses graph reuse after changed table bindings, KV growth,
recurrent reset, slot reuse and a larger cuBLAS activation input requirement.
The second reset-repeat must reuse already captured graphs without recapturing.
Count-bearing `PrefillBatch` and the larger prefill must not use decode graphs.

Every eligible graph-ON `DecodeBatch` must increment the actual replay counter
by exactly one. There are 130 calls in each engine lifetime, including clone
continuations. ON must report real capture **and later replay**; a capture
refusal followed by ordinary kernel fallback is a failed qualification. OFF
counters must remain zero. The probe does not predict a cache-entry count or
relax equality for different attention plans.

All 248,320 F32 physical-head values per row must be finite, with a valid
248,077-token vocabulary argmax; the engine's returned physical-head argmax must
match the full vector. Caller-owned decode/forward arrays receive NaN output
poisons and tail guards. This public API has no output-offset parameter, so the
tool does **not** claim a prefix guard. `PrefillBatch` creates its own arrays;
every returned value is checked. Input tokens must remain unchanged, positions
must advance exactly once, clone logits and reset-repeat logits must be bitwise
equal, and every ON output must equal the corresponding OFF output over the
**complete head including padding**, with no tolerance.

`result.json` records all 338 capture observations, graph counters before/after
each call, module/assembly/environment identities, raw input hashes, exact
comparison coverage and explicit limits. Complete vector bytes are retained
under SHA-addressed names and deduplicated across OFF/ON/repeats/clone copies.
A passing fixed plan needs at most 302 full-head-row equivalents, 299,970,560
bytes (about 286 MiB); a hard 384 MiB cap preserves space for a failure vector.
No partial or skipped scenario becomes PASS. Failed output vectors and an
incomplete manifest remain available for diagnosis. All retained vectors and
input manifests are rehashed before success; `result.json.sha256` pins the final
successful or failed manifest.
After all success gates, `completed-result.json` and its SHA sidecar are written
once with `FileMode.CreateNew`. The digest comes from the exact serialized final
bytes, independently of a later progress-file read. Only a complete snapshot
whose downloaded bytes match that sidecar can be used as final evidence. This
does not promote an incomplete progress file or retry/re-score a failed run.

Artifact directory/frontend/stat and runtime/native files are checked before
and after each lifetime; the graph suffix is removed only when comparing the
otherwise identical numerical state namespace. The 19.4 GB whole-file artifact
SHA is **not recomputed**: an external original-file SHA/inode guard is still
required. Default native module files and explicit PTX/fatbin/sidecar paths are
pinned; the engine's namespace also records its actual numerical module images.

Engine wall timers are diagnostic and include the full-head copy. OFF then ON
is not an ABBA benchmark, HTTP latency comparison or proof of end-to-end speed.
Graph equality is not an independent model oracle or task-quality evaluation.
Only actual successful execution can qualify the recorded scenarios; local
managed builds/self-tests leave GPU, graph execution and model quality NOT_RUN.
