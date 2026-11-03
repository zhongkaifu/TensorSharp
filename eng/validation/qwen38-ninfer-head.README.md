# Experimental NInfer W8 output-head companion

The native-CUDA Qwen3.8 path has an explicit, default-OFF companion head. It keeps
the existing GGUF body, Q4 embedding, GGUF draft and tokenizer.

**S13 acceptance: the actual CUA no-loss gate FAILED. Default remains OFF; the
companion is not recommended as a quality-preserving optimization.** Earlier
finite checks do not override the new pointwise regressions.

The original NInfer `text/output_head` can supply only DFlash draft proposals when the explicit
`draft` scope is selected; the original target Q6 head then stays allocated and
continues to produce target logits. The older shared scope replaces both target
and draft consumers. Both are hybrid deployments with different draft-head weights
and BF16 activation/output storage from the original Q6 recipe. Checkpoint
equivalence and quality preservation must be measured separately.

This feature does not load the full NInfer body, apply its activation calibration,
select the imported frontend, render its chat template, or adopt Ignis's runtime.
The NInfer file is an existing quantized artifact; the companion does not quantize
the loaded GGUF head.

## Build the optional module

Use a managed Host build containing the companion support. The W8 native module
is a separate manual build; normal backend builds do not compile or distribute it.
The current builder targets `sm_120a` and preserves the original e600a7d9
Ignis header closure and its licenses/notices. The validated hardware is RTX 5090.
The managed loader requires CUDA 13+ and validates the actual function ABI and
launchability. Do not patch ggml or substitute a different kernel schedule.

From the repository root, choose a fresh output directory:

```sh
python3 -B eng/validation/build-qwen35-w8-head-module.py \
  --out "$PWD/artifacts/w8-head-native-FRESH" \
  --nvcc /usr/local/cuda-13.2/bin/nvcc \
  --compile
```

Without `--compile`, this tool only generates a source snapshot. Keep the emitted
PTX, matching sibling fatbin, generated `.fatbin.sha256` sidecar and manifest
together. The loader records the image actually loaded; a PTX path may select
its matching fatbin. Source/build identity and numerical validation are separate
from successful compilation. See the
[native module contract](../../TensorSharp.Backends.Cuda/native/w8/README.md)
and [standalone C# driver gate](Qwen35W8HeadProbe/README.md).

## Select the head explicitly

Both variables hold **absolute, normalized, physical file paths**:

```sh
export TS_Q35_NINFER_HEAD=/absolute/qwen3_8_27b_nvfp4full-v2.ninfer
export TS_Q35_W8_HEAD_PTX=/absolute/artifacts/w8-head-native-FRESH/tensorsharp_w8g32_head.ptx
export TS_Q35_NINFER_HEAD_SCOPE=draft
export TS_Q35_NATIVE=1
export TS_Q35_CUDA_GRAPHS=0
export TS_Q35_VERIFY_GRAPHS=0
```

Launch the same native-CUDA Host command and GGUF model/draft paths used for the
control, with `--backend cuda`. The companion options are snapshotted during
construction; restart the server to change them. Both paths must be supplied.
The current safety guard requires exactly `TS_Q35_NINFER_HEAD_SCOPE=draft` when
either path is supplied; all three settings unset selects OFF. Empty, partial or
other scope values refuse. Earlier S12/S13 binaries allowed unset scope with both
paths to select shared target-and-draft: that historical recipe failed its S12
no-loss gate. The later S14 guard prevents accidental selection of that recipe;
it changes option refusal only and is not a quality fix or a new model certificate.
The actual S13 CUA below used its immutable earlier runtime with explicit draft
scope, and its overall no-loss gate failed. The typed backend still contains the
shared TargetAndDraft component for isolated research; it is not a serving option
exposed by the current guarded Factory route.
Empty or half-specified settings, aliased files/parents, incompatible metadata
and nonnative Qwen35 Factory routes refuse rather than silently using Q6.

The known artifact is the author's
[Qwen3.8-27B nvfp4full v2](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer/tree/86d9a4bf76997639dcd64f78fde2d73740ad9a3e),
file `qwen3_8_27b_nvfp4full-v2.ninfer`, whole-file SHA256
`abb1e120d5f1f32d61689604d238227ff579ab76cbd9319628f3b3904fffd9af`.
The reader/target plan requires its known `qwen3.8-27b` / `nvfp4full` family and
the exact `W8G32_F16S`, `row-split-k128-v1`, `[248320,5120]` output-head descriptor.
Other NInfer versions, VI files or weight formats are not interchangeable here.
The GGUF body is checked for the known geometry and explicit other-family
metadata is rejected; matching dimensions do not establish the same checkpoint.

The actual graph switches are `TS_Q35_CUDA_GRAPHS` for draft capture and
`TS_Q35_VERIFY_GRAPHS` for verify capture. Either enabled with this head is
unsupported and rejected before the native engine's own `cuInit`/allocation.
The Factory capability probe may already have touched CUDA. Keep other model, sampling,
KV and optimization settings fixed when comparing head OFF and ON. A diagnostic
split-K-zero profile is a separate math policy from production-unset controls.

To return to the default Q6 head, unset **all three** companion settings and restart:

```sh
unset TS_Q35_NINFER_HEAD TS_Q35_W8_HEAD_PTX TS_Q35_NINFER_HEAD_SCOPE
```

An ON startup emits one `[q35-cuda] experimental output-head companion` record
with the uploaded payload SHA, actual module-image SHA, profile, memory sizes and
actual consumers/scope. Draft-only records `Consumers=draft-proposals-only`,
`Scope=1`, and provenance `targetQ6Preserved=true`. Shared scope records both consumers. These identities also enter the engine's state namespace. An OFF
startup has no companion record. Recording an option alone does not prove which
path ran.

## Memory and execution scope

In **shared scope only**, the original Q6 head is not allocated:

| Allocation | Bytes |
| --- | ---: |
| Replaced Q6 head | 1,042,944,000 |
| W8 codes and F16 scales | 1,350,860,800 |
| Dedicated BF16 input scratch, 33 rows | 337,920 |
| Dedicated BF16 logits scratch, 33 rows | 16,389,120 |
| Estimated net increase | 324,643,840, about **309.6 MiB** |

In **draft-only scope**, the target Q6 allocation is retained. The W8 payload and
the two scratch buffers add **1,367,587,840 bytes, about 1.274 GiB**, rather than the
shared scope's net 309.6 MiB. Neither estimate includes module/driver allocations,
context, allocator overhead,
other workspaces and caching. Upload uses existing bounded loader staging rather
than a whole-head managed array. Finite original F16 scales and the actual copied
payload SHA are checked during upload. The engine owns the copied weights and
scratch; the source reader closes after construction.

Shared scope routes both target and draft consumers to W8; draft-only keeps the
original target Q6 path and uses W8 only for draft proposals. W8 rows are processed in chunks of
at most 33; each BF16 result is converted into its full global F32 output range on
the same stream before scratch is reused. The selector's original Q81 quantization
still runs before either head branch because the following selector projection
consumes those buffers. No new head policy changes that dependency.

## Validation checkpoint, 2026-11-03

- R10 exercised the C# component for T=1..33 with synthetic weights: eight direct
  original C++ references and 25 source-supported token-prefix references.
  Complete saved outputs matched and passed the fixed FP64 bound. This did not
  use the real NInfer head or the later engine BF16-to-F32 bridge.
- S11 exercised the real read-only v2 reader, frontend resources and format probe.
  HF 0.23.3 matched 44 encode-ID and 22 decode comparisons; all 248077 valid raw token
  byte entries matched. These are bounded frontend/format results, not full-model
  inference support or an exhaustive text oracle.
- The S12 real-head smoke completed three scenarios with eight public
  `SpeculativeExecution.DecodeStep` rounds each: 24 rounds used speculation, plus
  three next-turn continuation calls. This was one native probe invocation,
  **not 27 HTTP benchmark requests**. Target full-vocabulary finite checks and the
  probe's oracle-verify gate ran; there was no separate full-F32 draft oracle.
  All 27 oracle-verify calls completed. The smoke used a diagnostic draft
  split-K-zero policy. Two smoke option names were not consumed by the runtime;
  the actual verify-graph and verify-Q81 switches remained unset/default-OFF.
  The smoke does not qualify signal-driven teardown, general quality or speed.
  GGUF/draft identity is linked through the short comparison's full-SHA checks,
  rather than independently established by the smoke.
- The full S12 OFF/ON short comparison completed 48 cases /156 HTTP requests per
  arm. Both retained 39 PASS /9 FAIL /0 ERROR, but there was **one original
  true-to-false check and zero recoveries**. At the finer arithmetic-field level,
  two fields changed from correct to incorrect. All 156 canonical request hashes
  matched; 122 complete normalized responses matched. Equal total scores did not
  satisfy the no-loss gate. The companion remains **default-OFF and not
  recommended as a quality-preserving optimization**. The single OFF-then-ON
  order with three repeats is descriptive, not a statistically established speed
  gain. Its code responses reached the 1024-token output limit and only passed the
  original finite pattern checks. This phase added no complete-program, CUA or
  128K/250K ON qualification.


- S13 independently built the draft-only recipe from the immutable S12 base:
  66 relevant CPU tests passed, with no skips. A no-drafter target probe compared
  all 81 saved full-vocabulary F32 outputs per arm (248320 values each): every bit,
  finite check and recomputed argmax matched. Reset/repeat A/A was observed by the
  producer in memory, not saved as a second set of arrays. This probe used
  `GEMM_SPLITK=0`; it is separate from the production-unset short test and is not
  a draft-logit oracle or a performance measurement.
- S13 draft-only OFF/ON short tests each completed 48 cases /156 requests and
  retained **39 PASS /9 FAIL /0 ERROR**, **266/327 original checks** and **66/96
  arithmetic fields**. Independent per-check and per-field comparison found
  **zero true-to-false and zero false-to-true transitions**. All 156 canonical
  inputs matched; 146 complete normalized responses matched, including all 138
  non-parallel responses. Retained failures mean overall `quality_passed=false`;
  this finite regression gate does not establish comprehensive model quality.
  Qualified code/prose/copy medians were 343.19→349.22, 161.28→163.52 and
  186.34→194.62 tok/s respectively. These are one OFF-then-ON order, n=3 each,
  descriptive only; no stable speed gain is established. All three code outputs
  hit 1024 tokens, and the original pattern check is not whole-program quality.
  The short-test packet alone contains no complete-program, CUA, ABBA or
  draft-only long-context results; the subsequent actual ABBA subset is recorded
  separately below.


- A separate S13 ABBA performance subset then ran four fresh services in the
  order OFF-A1 → ON-B1 → ON-B2 → OFF-A2. Each arm passed its 9 original cases
  (27 requests /36 checks): 144 original checks total, with zero pointwise
  regressions or recoveries in either directed pair. All canonical inputs matched;
  serial answers were 9/9 exact in both pairs, while concurrent outputs differed.
  Combined descriptive n=6 code/prose/copy medians were 344.84→347.48,
  165.63→166.62 and 186.59→191.78 tok/s (+0.77%, +0.59%, +2.78%). There were only two
  fresh processes per policy, not a statistical demonstration of stable or
  universal ≥5% gain. The performance subset's PASS does not upgrade the retained
  failures of the complete short set. Complete-program, CUA and W8 draft-only
  long-context results are not included in this checkpoint.


- S13's separate complete-program run used the unchanged 4093 tool, ten tasks ×3
  repetitions and max_tokens4096. Each arm retained **24 PASS /6 FAIL /0 ERROR**;
  all30 canonical request bodies and complete responses matched. All24 executed
  whole-module files also had identical SHA. Independent stdin/stdout/oracle checks
  passed for150 actually completed functional rows per arm, while **39/189 planned
  inputs were not executed** because six programs failed the original closed-AST
  contract (grid task: ImportFrom/isinstance; decimal task: zfill). No missing row
  is PASS and no policy was relaxed. Both per-program and functional-case paired
  transitions were zero, but overall quality/acceptance stayed false. This is a
  small restricted program-regression set, not comprehensive coding quality or a
  model-speed result. CUA and W8 draft-only long validation remain separate.


- S13 actual CUA then completed384 requests from four fresh services (180 text and
  12 media per policy), same65536/cap2/UB4096/shared-prefix/scalar0/ALT0 and
  production-unset split-K controls. Text quality was168/180→167/180, with
  **3 original checks true-to-false and2 recoveries**; typed fields exposed
  **4 true-to-false and2 recoveries**. Media stayed10/12 on both arms, not full
  quality PASS. All384 measurement checks were valid but **overall no-loss FAILED**.
  Independent audit reconstructed384 embedded SSE streams,384 request bodies/wire
  hashes, all3840 original7+3 booleans and972 typed-field pairs. Inputs were exact;
  text174/180 and media12/12 normalized answers matched. The failures remain:
  append/lane1/step5 code1001→1661, edit/lane1/step3 code9970→9976,
  truncate/lane0/step7 code2250→2256, plus an additional edit/lane1/step5 code
  regression hidden by an already failing revision. Improvements never net out
  regressions. This does not establish a causal kernel/head/cache bug, and earlier
  scalar A/A variation does not waive the gate. **Do not adopt/recommend the
  candidate; default remains OFF.** Any later long test characterizes an
  unqualified candidate's capacity and cannot restore quality acceptance.


- The separate S13 DFL/M1/JSON long run used CTX262144/cap1, shared scratch plus
  vision-pool reclaim, original3warmups and128K×3 then250K×3 per fresh server.
  OFF passed6/6 requests and60 original checks (all cached0); TTFT medians were
  26.7614s/80.6338s. Draft-only ON passed128K 3/3 and30 checks with exact paired
  answers (TTFT27.0207s), but all 250K 3 requests returned **HTTP500 /device OOM**.
  Three original HTTP log events each report `no device memory for 8 MB more`.
  No ON 250K usage, answer or TTFT exists. All6 canonical/user-prompt hashes match,
  but OFF usage 255939 is not an ON measurement. The 30 unavailable paired checks
  are explicit availability failures, not 30 wrong answers or a no-regression PASS;
  original ERROR/false-check records remain unchanged. This configuration's
  capacity failure is not an engine-wide/context-support claim. **The candidate
  remains rejected for CUA and capacity; default OFF /not recommended.** S14 merely
  refuses configured paths unless SCOPE is exactly `draft`; all 3 unset is OFF. It
  does not change native math or certify S14 from these immutable S13 results.


- Final S14 independently built only the safety-parser/test replacement from the
  immutable S13 base (4507 source /432 runtime). The updated 66 focused CPU tests all
  passed, with 0 skips. Two actual native probes, OFF and explicit `draft`, each
  completed 27 oracle calls:24 speculative rounds across3 scenarios plus3 next-turn
  continuations. This was one dotnet invocation per arm, not 27 HTTP benchmark calls.
  Valid scope/OFF operation remains functional. Their diagnostic policy was
  target+draft split-K0, greedy_verify1 and graphs0; persisted metadata/hashes were
  parsed locally, not new full-F32 arrays independently reread. No S14 complete
  short/CUA/program/long quality or performance upgrade is asserted. S13 raw and
  rejection remain immutable: **default OFF /not recommended, CUA+capacity failed**.
  Standalone eng numerical/benchmark tools retain their separate source and
  coverage certificates and are not implied by this final managed build.

Generated logs, manifests and snapshots belong in ignored `docs/validation/` or
`artifacts/`. Preserve failed attempts and exact source/runtime/model identities;
do not promote skipped or unavailable cases to passes. Older engine speed ranks
compare different weight/KV/head/drafter deployments and do not isolate the benefit
of this format or companion recipe. No model-level performance gain is established
by the component, frontend or smoke results above.
