# Trained Qwen target batching diagnostics

**The original trained speculative parity failures remain open.** These diagnostics isolate target computation with identical teacher-forced input tokens, independent of draft proposals, acceptance decisions, grammar masks, and scheduler rollback. Every row compares all 248,320 logits. No tolerance or original output comparison was relaxed.

The original r4 plain/MTP pair first differs at generated token 90 for `json` and token 31 for `retained-A`. The committed-history diagnostic resets and prefills the same prompt before each run, then commits the entire common output prefix either one token at a time or in blocks of width 1–4. Width 1 is bitwise exact for all 121 rows. Widths 2 and 3 reproduce both original next-token differences at precisely the same positions. This establishes accumulated differences in batched target execution as sufficient to reproduce the failures; it does not establish a corrected implementation.

| Case | Block width | Unequal full-vocabulary rows | Maximum absolute error | Changed argmax at zero-based input row |
| --- | ---: | ---: | ---: | --- |
| json | 1 | 0/90 | 0 | None |
| json | 2 | 90/90 | 2.5357 | 89: 3799 → 30043 |
| json | 3 | 90/90 | 2.5357 | 89: 3799 → 30043 |
| json | 4 | 90/90 | 2.704574 | 61: 1070 → 3568, 89: 3799 → 30043 |
| retained-A | 1 | 0/31 | 0 | None |
| retained-A | 2 | 31/31 | 2.942094 | 30: 2542 → 1698 |
| retained-A | 3 | 31/31 | 2.942094 | 30: 2542 → 1698 |
| retained-A | 4 | 31/31 | 3.147305 | 30: 2542 → 1698 |

## Earlier final-block comparison

`verify-probe-r2` first commits the common prefix scalarly and compares only the final verification block. Width 1 is exact; widths 2–4 differ in nearly all vocabulary entries, although this shorter comparison preserves argmax. It did not capture the accumulated error identified above.

`verify-probe-r13` tests the isolated floating-matmul rewrite and routed-expert reduction fence. The native binary is `bf53d455a70008867850dfd02291a1e2942c52e6f47457568185fccb5548263d`. It still fails trained full-vocabulary comparisons and changes one scalar reference argmax relative to r4. It is not integrated or qualified, despite passing small fixtures.

## Provenance and limits

- Baseline source manifest: `f7431257148bd5168c0f80e3049b36fc5228c19a8d47448e09e60ff7eb43eb28`.
- Baseline CUDA native: `7eaf98656edcd043884f3eb776bed68abbee29f5cb446e237bfa16afa5598af7`.
- Upstream ggml remains unchanged at `456172ec733a135778adcd32d00e576a58232e45`.
- Three physical A40 GPUs use layer placement; this is not weight tensor parallelism.
- Each run includes its exact plan, application manifest, mapped binary evidence, before/after asset verification, owner process log, telemetry, and raw comparison JSON. Process/identity completion does not pass the numerical or scenario gates.
- `full-vocabulary-vectors.json` in each run lists the remote path, byte size, and SHA256 for every complete F32 vector file. The approximately 600 MB history vectors remain on the VM. Local source copies permit reproduction.
- Shared-host timing is diagnostic only. These runs do not qualify trained MTP speed, cancellation, concurrency, prefix retention, or the full model matrix.
