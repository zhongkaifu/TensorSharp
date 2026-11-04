# Original NInfer DFlash host integration probe

This separate tool exercises the actual original-artifact DFlash2 loader and
public native-CUDA engine integration. It does not implement a sampler, compare
an independent original engine, score general model quality or measure speed.
No production default is enabled by this tool.

```sh
dotnet build eng/validation/NinferDFlashHostProbe/NinferDFlashHostProbe.csproj \
  -c Release -p:TensorSharpSkipCudaPtx=true
dotnet eng/validation/NinferDFlashHostProbe/bin/Release/net10.0/NinferDFlashHostProbe.dll --self-test
```

No arguments prints PLAN. `--self-test` checks the fixed matrix, token/keep
shapes, ring-boundary coverage, raw capture bound and caller poison/guard
contracts, without loading a model or initializing CUDA. A managed-only build
preserves existing native assets and does not qualify native code.

Use a fresh immutable runtime and an otherwise idle operator-controlled GPU.
Write output to a new ignored **local POSIX** directory:

```sh
TS_Q35_NINFER_VERIFY=1 \
dotnet /ABSOLUTE/IMMUTABLE_RUNTIME/NinferDFlashHostProbe.dll \
  --execute \
  --artifact /workspace/hf/ignis/qwen3_8_27b_nvfp4full-v2.ninfer \
  --expect-directory-sha 222b7540e961a6455fff9f03cce451860e4d248785a4ef0a805f538c8eb82658 \
  --inputs /ABSOLUTE/ORIGINAL_BODY_PROBE/inputs.json \
  --expect-inputs-sha 4e07bc77f9fe75f9b23867b045fda6acd11ba4cf01f751ebfec7b7e96ce5c883 \
  --draft-module /ABSOLUTE/IMMUTABLE_RUNTIME/tensorsharp_q35_ninfer_draft.ptx \
  --context 8192 --ubatch 512 \
  --out /root/ts-eval/artifacts/ninfer-dflash-host-r1
```

Context must be 4096..262144 and ubatch 128..2048. All chosen body/A16/graph
options remain fixed and are recorded. Unqualified GDN-chunk and host-embedding
combinations are refused. The full target W8 head is mandatory; no shortlist,
requantization, external draft weights or alternate tokenizer is substituted.
The actual `NinferQwen38DraftSource` builds the original descriptors and public
`AttachNinferDrafter` uploads them. The borrowed sources and artifact reader are
closed before proposal execution, checking that the runtime owns its weights.
PTX and an available preferred fatbin/sidecar are pinned, and the actual loaded
draft image hash must match one of those precise image files.

The fixed matrix has lanes **1, 2, 3, 4**, initial bases **127 and 2047**, and
per-lane offsets 0..lanes-1 with distinct original token histories. This crosses
both a cache-block boundary and the 2048-position draft-ring wrap. Every seed
runs twice after reset with exactly the same prefill chunk/batch history.
All K7 proposals are deterministic supplied-token checks; anchors need not be
the target's greedy choice.

Each of seven phases starts from a fresh clone of that seed:

- Unchanged seed.
- Verify window8 followed by keep0 in every lane.
- Verify window8 with mixed keeps `[1,0,3,7]`, truncated to the lane count.
- Verify window8 with keep8 in every lane.
- Verify window1 and keep1, leaving the raw pending anchor unencoded.
- Pending window1 followed by ordinary Decode1, flushing before a new context row.
- Pending window1 followed by window8 mixed commit, flushing positive-kept lanes
  while a zero-kept lane retains its pending column until the next proposal.

Window8 zero/mixed/full tokens are the actual initial anchor plus seven proposal
IDs from the unchanged seed. Pending-to-bulk uses the pinned continuation window
so no earlier proposal can flush the pending column before that commit test.
All derived arrays, descriptor ranges and original input hashes are retained.

For every phase, a baseline and three separate routes must agree exactly:
ordinary clone, synchronous host copy, and asynchronous host copy whose events
are explicitly waited. Host donors are reset and freed **after** copy completion
and **before** restore, proving the capsule and ring survive source lifetime.
Restored complete recurrent state, all raw ring K/V and compatibility tags, and
all live attention blocks are byte-hashed against the source before any proposal.
Only blocks below `ceil(NPast/blockTokens)` participate in that cross-route
comparison: speculative future-only capacity may correctly be discarded. The
unchanged source itself is checked over every allocated block before/after.

Each route runs public `Draft` twice and direct `ProposeNinfer` once, comparing
all seven selector IDs and full-vocabulary logits bitwise. Proposal does not
advance target positions. A pending 51,200-byte BF16 feature column must flush
to frontier=NPast, then repeated proposals must preserve target/ring bytes.
The route next runs ordinary target Decode1 and another proposal; both target
and proposal heads must match the baseline. Confidence values must be exactly
1 as defined by this fixed-window admission API, and are never presented as
probabilities or acceptance rates.

Pending phases additionally compare against ordinary `DecodeBatch(anchor)` on
an independent seed clone, followed by the identical remaining operation
history. Target transition heads, later target heads and proposal heads must
match. Both routes encode the carried anchor using the original per-lane T1
context FC/QKV arithmetic. This control is deliberately restricted to that
same-shape T1 case. A shortened ordinary prefix is **not** used to validate a
wide partial commit's proposal: its original T1/four-chain versus wider-window
arithmetic can differ even when target state is equivalent.

Actual negative-admission tests cover self-copy, attached window9, short optional
Verify head, and Verify(A) followed by an ordinary Decode of other cloned slots.
Rejected requests must preserve complete state/positions and caller output;
the original valid Verify checkpoint must still commit and produce its pending
anchor capsule. These checks run independently for all eight seeds and reset
passes. No malformed request is retried as an inference success.

The full plan requires **112 phase cases**, **224 host handoffs**, **496 lifecycle
runs**, **2000 K7 proposal-batch calls**, **32,800 exact head-row pairs**, and
**2,980 reset-repeat head rows**. The successful raw-output bound is 2,980
248,320-F32-row equivalents, below the fixed **3 GiB** cap; identical outputs are
SHA-addressed and deduplicated. A budget refusal or unsupported shape is failure,
never a skipped PASS. All physical logits must be finite; padded vocabulary
rows cannot win. Caller outputs have NaN poison and tail guards, and every
compared row preserves signed zero. Selector choices are not required to equal
head argmax because the original predecessor/successor selector changes them.

Read-only diagnostics hash each original Conv/State allocation and each ring
allocation separately; attaching to the existing active slot may place its ring
outside its recurrent arena. Unwritten ring rows are not asserted finite or
deterministic after reset. Whole-ring equality applies only to a preserved
checkpoint, a copy/host restore, or keep0 before/after. Reset proof uses actual
complete outputs, selector choices and typed frontier metadata. Host accounting
must return to zero after all retained capsules are freed. Full-state transfers
and repeated forwards are deliberate correctness work; timings are not benchmarks.
Live seed/source slots can retain shared KV blocks, so this does not claim a
forced physical KV eviction/upload measurement.

`result.json` holds progress or failure. Only after all gates, model teardown,
input/native/assembly rechecks, artifact stat/directory checks and raw-vector
rehashes does the probe create `completed-result.json` once with `CreateNew`,
flush it to disk, and write its one-time SHA sidecar from those exact final bytes.
A stale/incomplete progress file is never upgraded. No failed output is repaired
or re-scored. The complete 19.4 GB artifact is not rehashed by this tool; device,
whole-file SHA and unchanged dependency provenance require the operator's pins.
A passing result qualifies only the supplied-token integration/lifecycle matrix.
It does not establish an independent Ignis draft oracle, speculative sampling
quality/distribution, task success, speedup, long-context capacity or default-on
production readiness.
