# Original NInfer target Verify/Commit state probe

This tool checks the explicitly enabled target-only `Verify`/`Commit` candidate
against ordinary model execution at the **same physical projection window**.
It does not run a DFlash proposer, sample tokens, compare engines or certify
broad model quality. No production code or defaults are changed by this tool.

```sh
dotnet build eng/validation/NinferTargetVerifyProbe/NinferTargetVerifyProbe.csproj \
  -c Release -p:TensorSharpSkipCudaPtx=true
dotnet eng/validation/NinferTargetVerifyProbe/bin/Release/net10.0/NinferTargetVerifyProbe.dll --self-test
```

No arguments prints PLAN. The CPU-only self-test checks the complete kept-count
matrix, lane/window coverage, duplicate prevention, guards and signed-zero byte
comparison. It does not load a model, initialize CUDA or qualify verification.
Managed-only builds require existing native assets and do not replace them.

Run from a new immutable runtime, on the operator's otherwise idle GPU, using
a fresh **local POSIX** ignored evidence directory:

```sh
TS_Q35_NINFER_VERIFY=1 TS_Q35_NINFER_DECODE_GRAPHS=1 \
dotnet /ABSOLUTE/IMMUTABLE_RUNTIME/NinferTargetVerifyProbe.dll \
  --execute \
  --artifact /workspace/hf/ignis/qwen3_8_27b_nvfp4full-v2.ninfer \
  --expect-directory-sha 222b7540e961a6455fff9f03cce451860e4d248785a4ef0a805f538c8eb82658 \
  --inputs /ABSOLUTE/ORIGINAL_BODY_PROBE/inputs.json \
  --expect-inputs-sha 4e07bc77f9fe75f9b23867b045fda6acd11ba4cf01f751ebfec7b7e96ce5c883 \
  --context 8192 --ubatch 512 \
  --out /root/ts-eval/artifacts/ninfer-target-verify-r1
```

The candidate flag must already equal `1`. Chosen A16/repack/module options are
recorded and remain fixed. Unqualified GDN chunk use is refused. Context must be
2048..262144 and ubatch 128..2048. A target-only NInfer model without a drafter
or feature taps is required. Every planned scenario is mandatory.

With `TS_Q35_NINFER_DECODE_GRAPHS=1`, every eligible public Verify and subsequent
one-token DecodeBatch call must record exactly one actual layer-graph replay;
a capture fallback is a failure. The ordinary counted controls must record no
graph capture or replay. Total actual captures and later replays must both be
positive. All tested physical Verify shapes are at most 32 rows, within the
candidate graph limit of 64 rows and window 1..8. Graph mode 0 or absent retains
the same exact head/state gates and requires graph counters to remain zero.
The chosen mode and every checked transition are recorded. This complements
the separate paired OFF/ON DecodeGraphProbe; it does not replace that proof.

The nine seed configurations use lanes **1, 2, 4** and rotated initial positions
**127, 255, 1023**, with distinct lane histories. Each tests windows
**1, 2, 3, 4, 7, 8**, every uniform keep count from zero through the full window,
and divergent mixed counts (including simultaneous zero/full/partial lanes).
The eight-token window covers seven DFlash drafts plus the anchor.
This is 456 distinct commit cases, repeated after resetting and rebuilding the
seed state: **912 actual successful commits** plus a rejected duplicate-commit
attempt after each. The complete derived token arrays are saved and come from
the pinned original prompt/continuation streams.

For each case:

1. Clone the unchanged seed into candidate and control slots before verification.
2. Before any pending verification, build an ordinary full-window all-row head
   reference on a fresh seed clone using the internal ordinary prefill seam's
   optional fifth argument. Run public `Verify` on the candidate. Every returned
   physical-head row must match that independent reference **bitwise at the
   same n=lanes×window head shape**, be finite, and have a valid argmax matching
   the complete vector. Guards and input tokens must survive; committed positions
   must stay unchanged. Complete verify-window head rows are archived once under
   their SHA names and reused across matching kept-count variants and repeats.
3. Immediately `Commit` the requested prefix. Positions must advance by exactly
   the supplied counts. A second commit must reject the consumed checkpoint.
4. Run `NinferValidationPrefillPadded` on the control, passing the same complete
   physical token window and per-lane counts. This internal friend-only seam
   uses the existing ordinary `RunRows(verify:false,counts)` path and no head;
   it does not call verification, commit, or replay helpers.
5. Compare a subsequent one-token `DecodeBatch` under the same lane/head shape
   between candidate and control, bitwise across all 248,320 F32 logits per row,
   including padding. All-zero keeps additionally compare against a no-operation
   seed clone. Full keeps additionally compare against ordinary public
   `Forward(window)` for one lane, or `PrefillBatch` for multiple lanes, followed
   by the same one-token decode. These separate controls must agree exactly.

Shortening ordinary input to `kept` would change A16/A4 and projection batch
arithmetic. Comparing a verify-window W8 head to a one-row head would also test
different arithmetic. Neither is used as an exact oracle here. Physical window,
slot order, starting positions and projection shape are held fixed; valid
prefixes are causal and ordinary count-bearing GDN scan independently checks
the captured-buffer replay and convolution-state updates.

Read-only diagnostics use bounded device-to-host copies to hash each seed's
complete original recurrent arena and allocated KV blocks before and after its
campaign. This checks parent clone/copy-on-write preservation, including bytes
beyond the current frontier. For all-zero keeps, recurrent/convolution bytes
are also hashed before Verify and after Commit and must remain identical. The
hashing calls synchronize and read memory; they do not mutate device state or
invoke verification helpers. The friend assembly is also used for these reads;
all actual model/state evolution otherwise uses public engine methods.

Caller arrays receive NaN active-output poison and tail guards. The public API
has no output-offset parameter, so no prefix guard is claimed. Every subsequent
head vector is finite and its reported argmax is checked against the full head.
Both independent-control comparisons and reset repeats require exact bytes,
preserving signed zero; no similarity tolerance, output repair or scorer change
is allowed. The identical full-window head comparison does not establish equal
sampling distributions between bulk verification and ordinary one-row decoding.
Future speculative KV writes remain in the candidate, while the
no-operation/ordinary controls contain different future data; subsequent decode
must mask all uncommitted rows.

Complete subsequent-head row bytes are saved under SHA-addressed names and
deduplicated across matching controls, repeats and kept-count variants. The
successful plan stores at most 1,716 full-head-row equivalents (1,704,468,480 bytes),
with a hard raw-vector budget of 2 GiB. The complete two-pass matrix requires
15,822 exact row pairs and 7,911 reset-repeat row pairs. Reaching that limit is a failed/incomplete run,
never a skipped success. Recurrent/KV snapshots retain hashes, not full state
copies. These full-state diagnostic transfers and repeated model calls are
deliberate correctness work; their timings are not inference benchmarks.

`result.json` remains progress/failure evidence. After all gates, model teardown,
source/input/native rechecks and vector rehashes finish, the tool writes
`completed-result.json` **once** with `FileMode.CreateNew`, followed by its
write-once SHA sidecar. The digest comes from the exact serialized final bytes.
Only downloaded complete bytes matching that sidecar qualify final evidence.
An incomplete/stale progress file is never upgraded and no failure is retried
or re-scored. Failed captures and manifests remain available for diagnosis.

The probe checks original artifact directory/frontend/stat, fixed input SHA,
selected native PTX/fatbin/sidecar files, assembly files and environment stability.
It does not rehash the complete 19.4 GB artifact; an external original-file
SHA/inode guard and physical-device provenance remain necessary. A passing
execution establishes the recorded target-state/rollback scenarios, not a full
DFlash integration, acceptance distribution, public quality rank, speculative
speedup, 128K/250K capacity or default-on deployment qualification.
