# Qwen 3.8 verify rows on CUDA — 2026-09-17

Resolves the three `Qwen4ExpRetainedCacheTests` gates that the September
qualification lanes carried as known CUDA failures
([retained-cache-20260916](../retained-cache-20260916/README.md)):
`TeacherForcedTargetVerify_MatchesSingleTokenFullVocabularyForDenseAndSparsePrefixes`,
`RepeatedTargetBlocks_MatchScalarTeacherForcingAtEveryCommittedRow` and
`SharedPrefixChunking_MatchesWholePromptForEachDistinctSuffix`. They passed on CPU
and failed on one A40 (evidence `/workspace/ts-int3/results/lanes2/cuda-qwen4exp.log`).

Environment: ggml 456172ec (unpatched), branch `claude/qwen4exp-cuda-exactness` from
01cda2b0, VM tree `/workspace/ts-q4x`, native built with
`TENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON` (test hooks), fixture
`fixtures/qwen4exp-qsa` (target f6d57de6…, head b7c93fea…), lane environment
`TS_TEST_QWEN4EXP_QSA=1 TS_KV_INITIAL_TOKENS=8 MAX_CONTEXT=1024`.

## 1. What diverged first, per assertion

`TS_Q4E_NODE_DUMP=<dir>` (test-hook builds) writes every node of every span graph;
[`nodediff.py`](nodediff.py) aligns two graphs by op sequence, matches rows by
absolute token position and names the first node whose in-graph sources are
bit-equal while its own row is not. Dumping flags every node as an output, which
disables ggml-cuda fusion, so fusion was bisected separately.

| Assertion (baseline CUDA) | First diverging node | Why | Dominant contribution |
| --- | --- | --- | --- |
| Teacher-forced verify: 4-row `SpecForward` vs 4 one-token forwards, max \|Δlogit\| 0.00181 (prefix 4) / 0.00701 (prefix 16), same argmax | node 0, the PLE value projection (F32 8→8), 6e-8; node 3 (F32 8→32) | one-token graph: `mul_mat_vec_f`; 4 columns: cuBLAS SGEMM (on Ampere `mul_mat_vec_f` takes at most 3 F32 columns, `mul_mat_f` needs width % 32) | attention. The fixture's head size 8 has no flash kernel, and V×probabilities (default precision) runs in cuBLAS `CUBLAS_COMPUTE_16F` from 2 rows, narrowing the probabilities to F16. Expanding only attention per row cut the maximum from 7.0e-3 to 3.1e-6 |
| Committed blocks: 32 tokens in blocks of 2/3/4 vs scalar decode, 32/32 rows differ, max 0.0082 | same nodes | same | widths 2-3 leave GDN/PLE/KV state bit-exact (F32 stays on `mul_mat_vec_f` up to 3 columns) and differ only through attention; width 4 also moves state. After attention and projections, 1-4 rows still differed by 9.5e-7: ggml-cuda's MoE weighted-reduction fusion (an FMA sum) was applied in some graph widths and not others, because it is skipped when the fused output overlaps an input's memory, which depends on allocator reuse |
| Chunked 16+4 vs whole 20-token prefill, max 2.6e-4, same argmax | node 49 in layer 0, an F32 32→3 hyper-connection projection, 2.4e-7 | cuBLAS SGEMM at 16 vs 20 columns | every batched kernel downstream (cuBLAS TF32, flash/MMQ); not a single op |

`GGML_CUDA_DISABLE_FUSION=1` alone left all three failing with the same maxima
(runs `nofusion-cuda`), so kernel selection, not fusion, dominates; fusion is the
last 1e-6.

## 2. The same kernels at the product's shapes

[`tests/qwen4exp_row_kernel_probe.cpp`](../../../../TensorSharp.GGML.Native/tests/qwen4exp_row_kernel_probe.cpp)
computes T rows batched, as T one-row graphs, and with the candidate construction,
at UD-Q2_K_XL shapes and types on one A40 ([raw](results/kernel-probe-a40.txt)).
Maximum \|batched − one-row\| at 4 rows (batched µs → candidate µs):

| Op (checkpoint tensor) | batched vs one-row | candidate, exact? |
| --- | --- | --- |
| Q5_K 2560→10240 (`attn_qkv`), Q6_K, Q8_0, Q4_K | exact up to 4 rows; 6e-7 to 1.1e-6 from 5 rows (MMVQ sizes its reduction by column group) | blocks of ≤4 rows: exact |
| F32 2560→512 (`ffn_gate_inp`), 10240→4 (`hc_*_inject`) | exact up to 3 rows; **2.6e-3 / 1.5e-3** from 4 | tokens on the broadcast axis: exact, 22.6→25.5 µs |
| BF16 2560→512 (`indexer.q_proj`) | **5.9e-3** from 2 rows | broadcast axis: exact, 15.5→18.7 µs |
| F16 2560→640 | 1.1e-3 from 2 rows | broadcast axis: exact |
| IQ2_XS / IQ4_NL routed experts (k=10) | 4.8e-7 / 2.4e-7 from 2 rows (multi-token MoE kernel) | one row at a time: exact, 51→78 / 74→98 µs |
| flash attention, head 256, 24/2 heads, 4096 keys | 2.9e-5 from 2 rows | one query at a time: exact, 45→123 µs |
| gated delta net, S=128, 48 heads | exact | — |

## 3. Change

Completion review tightened the scope of the measured MMVQ optimization: only
NVIDIA A40 with Q4_K, Q5_K, Q6_K and Q8_0 uses groups of four. Other devices/types
use the single-column broadcast construction already measured by the row probe.
Upstream Turing and GB10 reduction tables differ at width 1; the A40 group is not
portable. `TS_Q4E_TEST_MMVQ_CHANNELS=1` in a test-hook build forces the fallback
on A40. The committed-row fixture test now covers every width 1–8; the probe also
includes widths 6 and 7. The historical raw files below remain their original
measurements and do not claim fresh coverage of those added cases or other GPUs.

The combined completion branch also enables the construction on CPU. A fresh
macOS ARM run found width-dependent logits with exact stored recurrent/KV state;
the strict fixture suite passes with this path enabled. Metal is unchanged.

`ggml_ops_qwen4exp.cpp` (`Q4eRowKernels`): on CPU and CUDA, a span graph of 2 to
`TSG_PRECISION_DECODE_COLUMNS` (8) tokens builds each row from the kernels its
one-token graph runs — float projections with an even input width on the
broadcast axis (one `mul_mat_vec_f` launch; odd widths, which a one-token graph
sends to cuBLAS, one row at a time), quantized projections in blocks of at most 4
rows, routed experts and attention one row at a time with each attention row over
the KV window and mask row its own decode step reads (the first row's window joins
the graph replay key). Every CUDA span graph of up to 8 tokens, the one-token
decode graph included, keeps the inputs of the two allocation-dependent fusions it
contains (MoE weighted reduction; RMS norm × weight → RoPE) allocated as graph
outputs, which makes both fusions unconditional. One-token and prefill kernels are
unchanged. `TS_Q4E_TEST_BATCHED_VERIFY=1` or `TSGgml_Qwen4ExpTestBatchedVerify`
(test-hook builds only) restores the batched kernels for A/B measurement.

Bisection that fixed the design (runs `rs*`, `b*`, `e*`, `f*`):
attention + projections + experts without the fusion inputs kept: blocks gate 3-4
rows off by 9.5e-7 (`f1`); with them: exact (`f0`). Broadcast axis for the fixture's
odd-width 3→32 F32 projection: 25 rows off (`p1-cuda`), one row at a time: exact
(`e8`). Experts one row at a time is not needed by the F16 fixture (`f4` exact) but
is by IQ2_XS/IQ4_NL experts (probe).

## 4. Results

Fixture lanes, exact release-gate filter
`Category!=Bench&Requires=Models&FullyQualifiedName~Qwen4Exp`:

- CUDA (one A40): 19 passed, 1 skipped (`LayerSplitCheckpoint_…`, needs two reserved
  GPUs), 0 failed. Teacher-forced verify and committed blocks: every row at every
  width bit-identical (0 of 8 verify rows and 0 of 256 committed rows differ).
- The same library with `TS_Q4E_TEST_BATCHED_VERIFY=1`: both verify gates fail again
  with the original maxima (0.00701 / 0.0082) — the gates fail before and pass after.
- CPU: 19 passed, 1 skipped; all three gates bit-exact, as before.

Qwen3.8-Flash-Next UD-Q2_K_XL over a three-A40 layer split, `ChunkParityProbe
--chunks 4096 --verify-widths 2,3,4 --verify-tokens 48 --verify-reps 3`: a 3,248-token
prompt, then its first 48 greedy tokens teacher-forced as one-token decode steps and
as `SpecForward` blocks; each repetition alternates the two kernel sets in one process
([raw](results/qwen38-verify-parity.json)).

| width | kernels | rows differing / 48 | greedy flips | max \|Δlogit\| | median verify ms (reps 1, 2) |
| ---: | --- | ---: | ---: | ---: | --- |
| 2 | batched | 48 | 6 | 2.48 | 24.06, 24.31 |
| 2 | verify-row | 0 | 0 | 0 | 24.52, 24.84 (+2%) |
| 3 | batched | 48 | 4 | 2.34 | 26.77, 27.22 |
| 3 | verify-row | 0 | 0 | 0 | 28.27, 29.88 (+8%) |
| 4 | batched | 48 | 5 | 2.53 | 29.26, 29.56 |
| 4 | verify-row | 0 | 0 | 0 | 32.40, 33.07 (+11%) |

The batched flips were not confined to near-ties: only three of the reference's five
smallest top-2 margins (0.024-0.084) fall inside the 48 rows. Decode steps (median of
48) measured 20.69 and 20.64 ms with the change against 20.60 and 21.05 ms without, and
the 3,248-token prefill 2,335 and 2,338 ms against 2,324 and 2,359 ms: no decode or
prefill regression within run-to-run noise (0.8% and 0.2% in the change's favour).

End to end, `AgentTurnBench --scenarios spec --warmup 1 --measure-passes 3
--draft-model mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (1,388-token code-copy prompt,
192 tokens, `--spec-draft` default 3), four processes alternating the kernel sets
(verify-row, batched, verify-row, batched), six measured passes each
([raw](results/qwen38-spec-e2e.txt)). Every plain, n-gram and MTP stream was identical
to plain greedy under both kernel sets (this prompt has wide margins); MTP accepted
141/141 drafts in 47 verifies in every verify-row pass.

| request | verify-row: mean tok/s (range) | batched: mean tok/s (range) |
| --- | --- | --- |
| plain greedy decode | 49.1 (45.6-51.3) | 47.0 (45.8-49.3) |
| plain greedy prefill | 830 (802-865) | 804 (736-823) |
| n-gram speculation | 73.8 (68.7-78.9), -7% | 79.5 (73.8-84.5) |
| MTP head speculation | 83.2 (75.3-87.2), -4% | 86.5 (80.9-95.5)¹ |

¹ One batched pass drafted only 117 tokens in 39 verifies at 39.2 tok/s and is left
out of the mean. Plain decode and prefill show no regression. Speculation pays the
verify-row cost: MTP is 1.69x plain instead of 1.84x on this prompt, n-gram 1.50x
instead of 1.69x (n-gram also replays partially accepted rows through the same
kernels).

## 5. Prefill shape stays a bounded contract on CUDA

A 16-token prefill followed by 4 tokens and one 20-token prefill are different
batch widths, and the prefill kernels (cuBLAS TF32, tensor-core matmuls, flash and
MMQ) are chosen by width. Forcing a one-token kernel at prefill widths costs the
throughput those kernels exist for, and the product does not rely on prefill-shape
invariance for correctness: retained holders and checkpoint clones are exact copies
(their own bit-exact gates pass on CUDA), and a follow-up turn is only as close to a
cold prefill as two prefill shapes are. With the change the 4-token suffix runs the
verify-row kernels; measured on CUDA over the three suffixes: logits 1.7e-4, 4.4e-4,
3.8e-4, raw QSA keys 0, 0, 1.9e-6 (F16 cache), greedy identical.
`SharedPrefixChunking_…` therefore stays bit-exact on every backend but CUDA, and on
CUDA bounds logits and raw keys at 1e-2 — 23x the measured maximum and 32x below the
0.3155 of the stale-seed defect the test was written for — and allows a greedy change
only where the whole-prompt top-2 margin is at most twice the measured difference.

## Replay

```text
# fixture lanes: TS_TEST_QWEN4EXP_MTP_FIXTURE=<fixtures/qwen4exp-qsa> TS_TEST_QWEN4EXP_NATIVE_SHA256=<mapped libGgmlOps>
#   TS_TEST_QWEN4EXP_QSA=1 TS_KV_INITIAL_TOKENS=8 MAX_CONTEXT=1024, plus
#   CUDA: TS_TEST_GGML_BACKEND=cuda TS_TEST_QWEN4EXP_MTP_BACKEND=GgmlCuda   CPU: TS_TEST_GGML_BACKEND=cpu
dotnet test InferenceWeb.Tests -c Release --no-build --filter "Category!=Bench&Requires=Models&FullyQualifiedName~Qwen4Exp"
# before/after on one library: add TS_Q4E_TEST_BATCHED_VERIFY=1 to the CUDA run
# node dumps and first divergence (test-hook build)
TS_Q4E_NODE_DUMP=/path/dumps dotnet test ... --filter FullyQualifiedName~TeacherForcedTargetVerify
python3 nodediff.py /path/dumps 6:2 6:3 6:4 6:5 -v
# kernels at product shapes (CUDA build with TENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON)
GgmlOpsQwen4ExpRowKernelProbe --repeats 30
# real model: three GPUs, TENSORSHARP_TP_DEGREE=3 MAX_CONTEXT=16384, test-hook native library
ChunkParityProbe <UD-Q2_K_XL shard 1> --prompt-file prompt.txt --chunks 4096 --verify-widths 2,3,4 --verify-tokens 48 --verify-reps 3
AgentTurnBench --model <UD-Q2_K_XL shard 1> --backend ggml_cuda --draft-model <mtp head> --scenarios spec --warmup 1 --measure-passes 3
```
