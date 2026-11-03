# Controlled CUA repeat diagnostics v2

This tool runs the same immutable S4 source/runtime through fresh servers and the
original public text/media replays. Default invocation prints a plan only. It
never builds, stages models, changes producer inputs, changes scores, or sends
messages. The tracked controller and its two tracked helper subsets are a complete
code closure; no ignored Python helper is required on a fresh checkout. The actual
frozen harness, prepared fixtures, models and S4 build evidence are operator inputs.

Start with `qwen38-cua-causality-production-aa.json`: ScalarA1 and ScalarA2 both
explicit CPU SIMD0, FFN0, all other CUDA options0, and both splitK variables UNSET.
Each profile has180 text requests and12 media requests, using separate fresh
servers per kind:384 requests total. CTX65536, configured server/client lanes2,
UBATCH4096, DFlash7/p_min0, trusted single tenant shared prefix cache, prescribed
history, original temperature0/top_p1/seed1234/max_tokens192 are fixed.

If a separately labelled math-policy diagnostic is needed, use
`qwen38-cua-causality-splitk0-aa.json` with a new work directory. Both
`TS_Q35_DRAFT_SPLITK=0` and `TS_Q35_GEMM_SPLITK=0` are then explicit. This is a
**new math policy**, not proof that every CUDA operation becomes deterministic.
Mixed-FP4 verify and dynamic verify-window selection remain original defaults;
the MIXED_FP4_VERIFY option is recorded as UNSET without changing it.
It adds another384 requests and is not pooled with production-UNSET performance.
The explicit policy map is shared by environment creation and actual manifest
validation. Missing a manifest key is rejected even when UNSET is expected.

| Policy | DRAFT_SPLITK | GEMM_SPLITK | Interpretation |
|---|---|---|---|
| production-unset | UNSET | UNSET | Original split settings; CPU still explicit0/1 |
| draft-zero | 0 | UNSET | Draft-only override, NEW MATH POLICY |
| target-zero | UNSET | 0 | Target-only override, NEW MATH POLICY |
| splitk-zero | 0 | 0 | Both overridden, NEW MATH POLICY |

For a repeated single-factor diagnostic, use
`qwen38-cua-causality-draft-only-aa.json` or
`qwen38-cua-causality-target-only-aa.json`. Each contains two fresh CPU0 profiles
with identical split settings and executes384 full requests. Use a distinct
work directory such as `draft-only-AA-r1` or `target-only-AA-r1`.

The optional `qwen38-cua-causality-factorial.json` measures four CPU0 conditions
and requests directed Prod→Draft0, Prod→Target0, Prod→Both0 comparisons. It has768
requests total, **one profile per condition and kind**; it cannot estimate
condition-specific variance or establish causality. Prod→Both0 changes two split
settings, so it cannot isolate a draft-only or target-only effect. No override
proves complete determinism: mixed precision, batch shape and scheduler may vary.

No reduced protocol is supported. `--selection text` or `media` explicitly limits
coverage to360 or24 requests per pair and cannot claim the missing kind passed.

```sh
python3 eng/validation/qwen38-cua-causality-r2.py \
  --source /dev/shm/ts-ignis-sampler-optin-r1/source \
  --runtime /dev/shm/ts-ignis-sampler-optin-r1/runtime \
  --build-campaign /dev/shm/ts-ignis-sampler-optin-r1/control/campaign.json \
  --jobs-json eng/validation/qwen38-cua-causality-production-aa.json \
  --harness /root/ts-harness-root-frozen \
  --lease /ABS/ACTIVE-ROOT-LEASE.json \
  --work /dev/shm/ts-qwen38-causal-n8-r1/production-AA-r1
```

Review this plan, then add `--execute`. For the splitK0 pair use its jobs file and
`--work /dev/shm/ts-qwen38-causal-n8-r1/splitk0-AA-r1`. An optional CPU A/B
profile sets `cpu_simd` to string `"1"` and adds an explicit directed comparison;
only CPU or splitK policy may change across one comparison. Names must be unique.
Profiles and every directed comparison remain separately labelled; correct→wrong
transitions are reported individually and never offset by improvements.

The controller accepts only the actual S4 build/catalog and complete415-runtime
inventory ending in Runtime SHA630b8835…aeea3. On execution it rehashes source,
runtime and resolved provider bytes, the unchanged ggml checkout facdc44a6…, model
inputs, original frozen tools4e/966/7e/fbd and CUA producerab25261e… plus all pinned
fixtures before/after. It requires a current ROOT-exclusive lease, actual GPU0
UUIDd0e9…, and the global GPU lock. It rejects occupied unknown ports and stops
only its exact owned record, preserving failed startup/cleanup evidence. Builtin
nginx8081 is reserved.

Only the original seven quality and three measurement checks qualify a replay.
Execution complete is distinct from quality passed. Raw full/canonical/wire input
hashes, consumed SSE bytes/base64/size/hash, parsed response hashes, usage and
cached token observations are retained and revalidated against the frozen
producer. There is no new independent SSE semantic parser: frozen parsed fields
are hash-bound and original scoring is rerun unchanged. Typed-field comparisons
are additional diagnostics and do not rewrite original verdicts.

A/A variation helps quantify repeatability; neither a difference nor an unchanged
net total proves a CPU-kernel cause. These public synthetic prescribed-history
replays are not closed-loop agent quality, a full BF16 model oracle or timing-gain
proof. No timing-gain or comprehensive quality claim is made by this tool.

Run CPU contracts without GPU/model access:

```sh
python3 -m unittest discover -s eng/validation -p test_qwen38_cua_causality_r2.py -v
```

The original R1 controller aed9195e…a88a7f9e, helpers and source packages remain
unchanged. R2 adds only policy glue/schema/labels and new configuration files;
all original input/quality/transport/source/runtime/cleanup functions are retained.
