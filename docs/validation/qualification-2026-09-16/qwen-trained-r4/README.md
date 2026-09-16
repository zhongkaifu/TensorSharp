# Trained Qwen3.8 MTP — r4 first process pair

Trained execution is established; full-output parity and release performance are **not** qualified. Both complete benchmark processes pass the corrected scheduler-finish checks. Native, mapped managed assemblies, model shards, shared Q8 head and mmproj identities were verified before/after; see the owner records.

The native is the [r4 CUDA build](../cuda-r4/README.md), SHA256 `7eaf98656edcd043884f3eb776bed68abbee29f5cb446e237bfa16afa5598af7`. Both modes attach the real shared head at construction. Physical GPUs0/1/5 hold contiguous layer groups; this is not tensor parallelism. The host also ran other workloads, so timings are descriptive, not a quiet repeated acceptance comparison.

[Complete matched comparison](single-pair-parity.json): **13/16 matched-input, full-output and finish comparisons pass** (including warmup); three do not:

- JSON first differs at output token90: plain110 tokens, MTP109.
- Retained-A first differs at output token31: plain35 tokens, MTP38.
- Retained-A-followup inherits the differing generated history. Its input tokens differ, so identical132-token output does not qualify as matched parity.

These same direct output differences occurred on r3. All four concurrent outputs match between modes on this dense workload; this does not clear the original8K comparison against different native variants.

The learned head actually drafted and accepted proposals, with rollback exercised on seven non-warmup text requests. For example, copy accepted140/142 proposals; JSON40/62 with12 rollbacks; short13/27 with6 rollbacks. Media requests and the multi-sequence step use explicit non-speculative paths. Ordered static-image rows are not video coverage.

The initial owner attempt failed before inference because `/proc/PID/cmdline` was transiently empty during exec. Its failed record remains. The corrected owner waits briefly for an exact command line and still refuses mismatch, process exit, or timeout; five identity tests pass. `plain-owner-r2` and `mtp-owner` are the completed process records.

The r4 source predates admission of layer-split prefix checkpoints. Retained-A/B rows have zero reused tokens; no full-model released-cache reuse is claimed here. Image follow-up live reuse (381 tokens) and ordered-image follow-up live reuse (724 tokens) are separate mechanisms. Cancellation, long QSA, trained retained clones, quiet three-pair timing and memory gates remain outstanding for this process pair.
