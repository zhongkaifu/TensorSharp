# Qwen3.8 Flash Next learned-MTP follow-up

**Prepared, not executed on the full model. Release qualification remains false.**
This is a separate benchmark derived from the frozen pre-QSA v7 source package.
It never modifies ggml, the frozen production source/archive, or earlier reports.

## Fixed scope

| Lane | Actual path and checks |
| --- | --- |
| Dense scheduler | Short text, sustained copy, constrained JSON, shared-prefix A→B→A follow-up, four simultaneous requests, solo recovery, image→follow-up, ordered static frames→follow-up. Every rendered/injected prompt + output allowance + four verifier positions must fit within 2,051 positions. |
| Long scheduler | Same text, retained-prefix and concurrency paths with an approximately 8K copy prompt. Explicitly **QSA-unqualified** on this pre-QSA build. |
| HTTP replay | All original six Qwen HTTP suites and unchanged oracles, including the original oversized 64K failure; original 29 tool-policy cases remain separate. |
| Additional HTTP | 32 real API contracts: 16 invalid and 16 valid, split evenly between streaming/nonstreaming; a new exactly counted 62K case; actual two-frame `video_url` plus history follow-up in both response modes. |
| Actual agent | A separate 46-case WSL owner is still required. API tool calls here do not execute shell, skills, or sandboxed code. |

The dense benchmark has 16 rows/process, including one unmeasured warm-up; the
long tier has 12. The learned drafter must actually attach in **both** modes.
`plain` disables speculation; `mtp` explicitly selects the learned head using
`auto`, a three-token draft window, and the production default confidence gate
0.15. This is a within-build comparison. It does not use the old native baseline,
which lacks the new Qwen MTP implementation.

Run three independently warmed process pairs in the fixed order plain/MTP,
MTP/plain, plain/MTP. The comparator requires full output-token and finish-reason
equality, exact prompt-token/budget equality, per-request verify engagement,
accepted proposals, exercised rollback, and actual shared/retained reuse. A
missing head, fallback, zero engagement, unexercised rollback, missing telemetry,
failed process, or identity drift fails its gate. No truncated-prefix comparison
or numerical tolerance is used. The current comparator additionally checks each
mode's complete measured prompts, budgets, output tokens and finish reasons
across all three repeats. Two modes drifting together cannot pass this gate,
and concurrent timelines must carry the matching request ID. Previously frozen
packages retain their original comparators; this change needs a new package.
The original **5%** latency, throughput, and
memory gates remain unchanged; extra MTP memory is not waived. Both variants
attach the same head. Memory peaks are sampled, with the requested one-second
resolution recorded separately from allocator-level peak measurements.

## New long input and video limits

The separate `long_62k` request keeps the original corpus construction and exact
ALPHA/BETA/GAMMA JSON oracle. Metadata-only production tokenization counted
**65,065** prompt tokens, or **65,325** including 256 output tokens and four
verifier positions. The original **67,154**-token request remains untouched.
The count records exact frozen v5 managed DLLs, rendered UTF-8 bytes and token
IDs, with native libraries unmapped before/after. The future v7 HTTP response
must report the same prompt count; a new v7 metadata-only count can additionally
compare every rendered byte and token before execution. This is not an inference
pass or a claim that the v5 and v7 assemblies are identical.

The current Qwen4Exp protocol rejects `video_url` before generation. Its shared
Qwen-VL injector handles static images without video temporal coordinates. The
new actual video probe retains that failure and marks its subsequent history turn
unrun. For the pinned three-frame clip, `fps=1,max_frames=2` samples frame indices
0 and 2 (codes 17 and 86; times 0 and 2). Separately, the private scheduler uses
two static images (frames 0 and 1) with explicit text ordering. Passing that path
does not establish video support.

The frozen pre-QSA Qwen4Exp model does not implement retained cache/checkpoint
cloning and advertises no KV truncation support. The A→B→A fixture deliberately
records actual reuse and keeps a zero-reuse coverage failure; its shared system
length alone does not prove a clone. Existing active per-sequence holder tests
exercise a different lifetime and cannot substitute for this retained path.
The current working source adds exact retained-holder reuse and checkpoint
cloning, with independent synthetic CPU and physical two-GPU layer-split proofs.
Those edits require a new source/application binding and trained-model replay;
the frozen run's gap and true tensor-parallel limitation remain explicit.
The [retained-cache review](retained-cache-20260916/README.md) records the
separate exact CPU source/native identities, two additional fixes and 41 passing
synthetic checks, including the original failures retained for comparison.

## Preparation and binding

`prepare.py` extracts only the benchmark C# files from the exact frozen source
archive, checks every original digest, and makes explicit consumer changes:
partial `Bench`, a private scenario entry, mandatory real-head checks, and (for
older benchmark snapshots) passing the draft path to the model constructor.
That constructor argument is required before multi-device placement. The head
guard checks the actual Qwen model and `DraftHeadKind.PerToken`. Scheduler rows
use `max_tokens` for a token-budget finish; those rows must contain the complete
requested output budget. HTTP's `length` spelling is not used in scheduler rows.
`QwenMtpScenarios.cs` supplies the new paths. No production project reference or
native build is triggered by its direct-DLL project.

`prepare_package.py` snapshots the derived source, current reviewed lifecycle/
identity helper, exact original HTTP harness and configuration, new inputs,
token-count evidence, and model/head/fixture pins. Its HTTP profile is explicitly
an **unbound template**, with no invented native digest.

After root finishes the next build and creates the exact managed-application
manifest, bind locally without launching anything:

```powershell
python <package>/repo/eng/validation/qwen38_mtp_followup/bind_plan.py `
  --package <package>/package.json --package-sha256 <actual-package-sha> `
  --build <completed-native-build.json> --build-sha256 <actual-build-sha> `
  --managed-build <passed-managed-build.json> --managed-build-sha256 <actual-managed-sha> `
  --application-manifest <application-manifest.json> --application-manifest-sha256 <actual-app-sha> `
  --remote-package /workspace/qwen38-mtp-v7-followup `
  --remote-application-manifest <exact-VM-application-manifest-path> `
  --server-assembly <exact-VM-TensorSharp.Server.Host.dll> `
  --native <exact-VM-libGgmlOps.so> `
  --allow-unqualified-qwen-only `
  --remote-output /workspace/tensorsharp-no-patch-20260915/results/qwen38-mtp-v7-followup `
  --output <new-local-bound-directory>
```

The explicit unqualified flag preserves any original failed DS/TP gate and
`release_qualified=false`. It cannot bypass failed Qwen CPU/CUDA fixtures,
C-ABI checks, an unfinished source/native audit, or different managed files.
The output has complete argv/env records for the private build, 12 benchmark
processes, and two HTTP lifecycles, plus profile digests. Root must stage the
bound files into the named VM package and verify all digests before launching.

## Runtime owner requirements

Root owns the VM lane and first trained-head load. Run each benchmark under an
owner that records process start ticks, actual mapped native SHA, exact managed
app file set, checkpoint/head/mmproj identity, layer distribution, and before/
after audits. Copy only the bound native into the isolated benchmark output and
verify all production DLLs against the same server app. Do not start two listed
commands simultaneously. Record one resource sampler per owned process and its
exit; keep all raw logs on the VM. GPU0/1/2 use **layer placement**, not routed
expert tensor parallelism.

`compare.py --repeats <resolved-three-pairs.json> --output <report.json>` expects
three objects containing `plain`, `mtp` raw row arrays; `plain_inputs`,
`mtp_inputs` sidecar objects; `plain_memory`, `mtp_memory` with integer sampled
peaks `host_rss_peak_bytes`, `cgroup_current_peak_bytes`, `gpu_used_peak_bytes`;
and `plain_owner`, `mtp_owner` with process exit, before/after identity,
exclusive-timing and process-gone results plus native/managed/model/head SHA256.
These fields must come from retained owner/telemetry evidence, never assumptions.

The HTTP runner already binds the listening process, app, native, and request
policy when supplied its application-manifest arguments. Preserve every exit
code. Its 32-wire-case catalog is additional to the 32 model-free adapter unit
rows; it is not a claim that those architecture/internal-tool unit branches can
all be represented by a single loaded HTTP model.

## Local checks

The tiny tokenizer and derived benchmark compile passed. The benchmark retains
four nullable-annotation warnings from the pinned helper. Model-free tests cover
full-finish parity, prefix-only mismatch, no-verify fallback, absent rollback,
wrong per-request counters, changed history, context bounds, missing cache reuse,
three-repeat medians, unchanged 5% gates, missing memory/identity, failed process,
raw XML rejection, fragmented tool-call arguments and IDs, and limited build
eligibility. Full-model correctness and performance remain unrun.
