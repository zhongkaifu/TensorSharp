# Branch completion and integration — 2026-09-17

Review target: `feature/fix_build_break_and_optimization`, initially `e10dd42b`.
The completion work is assembled with separate task merges on
`codex/complete-branch-integration`, then merged into the review target.

## Inventory and delivered scope

| Requested branch | Initial tip | Scope |
|---|---|---|
| `biglane-glm-fixes` | `b7f224ac` | GLM tokenizer, parsing, stop tokens, and per-device decode positions |
| `claude/dsv41-perf` | `d8222d94` | Parent of both DeepSeek performance branches below |
| `claude/dsv41-perf-load` | `17b86f19` | Pread warming, pageable host experts, uploaded-page eviction |
| `claude/dsv41-perf-prefill` | `94afb376` | Sparse F32 prefill and automatic microbatch selection |
| `claude/fix-fattn-abort` | `6ceb15a4` | Explicit attention fallback for unsupported backend shapes |
| `claude/gptoss-conc4-determinism` | `4cb73a64` | Pin uploaded MoE biases; reproducible concurrent benchmark arrivals |
| `claude/load-refusal-exit` | `c068cfce` | Controlled load refusals, exit code 2, concise diagnostics |
| `claude/qualification-2026-09-16` | `2ee41ab1` | Shared Phase 0 conversation/media cache fixes, Qwen image reuse, Gemma media prefill |
| `claude/qwen4exp-cuda-exactness` | `01cda2b0` + tracked/untracked work | Exact verify-row computations and bounded prefill-shape contract |
| `claude/radix-m2` | `9ae80b49` | Model cache contract and family adapters at Legacy readiness |
| `claude/radix-prefix-cache` | `cb98c0fb` | M0 correctness fixes, M1 tree core, property harness and benchmark |
| `claude/scheduler-followups` | `c94f2c46` | Admission accounting, prompt hash caching, atomic inject refusal and cache scratch |
| `review-wf_21939204-cd7-1-fixes` | `505f2b5b` | Capacity admission/video review fixes; included by qualification |
| `review/merge-082599d7-fixes` | `999b534b` | Already merged before this work; no outstanding unique commits |

The radix branches deliver M0–M2. They do **not** activate the radix tree in the
production scheduler: M3 wiring and later per-family Tree promotion remain the
design's future milestones. Existing uncommitted Qwen work was copied into an
isolated completion branch; its original worktree was preserved. Existing model
jobs, untracked build directories, original ABBA scripts, and `.vscode/` were
preserved as well.

## Additional fixes found during completion

- Contain exceptions from either kind of DeepSeek warming worker, join started
  threads and report failure instead of terminating the process.
- Preserve literal `</think>` inside GLM JSON strings, including escaped strings
  and every streaming split boundary.
- Restrict the Qwen four-row quantized reduction optimization to tested A40
  types. Other cases use one-column broadcast reductions; verify coverage now
  includes widths 1–8. Apply the construction to CPU too: this fixed the ARM
  fixture's previously non-exact committed rows without loosening assertions.
- Apply the flash-attention guard to the newly merged Qwen verify-row call.
- Use the single accounting-aware Qwen cache scratch allocator after merging
  the scheduler and radix fixes.
- Preserve disposal semantics when a radix reclaim batch mixes ordinary
  eviction with pressure, invalidation or reset.
- Make ABBA reject failed/partial processes and measured passes, and run each
  requested native arm from an isolated output copy with recorded identities.
- Retain Gemma batched-decode input storage through the native upload. An
  optimized-JIT forced-GC regression reproduced premature collection before
  the fix; native code is unchanged.
- Align explicit model-file overrides between test discovery and loading.
- Fail host-process refusal checks when required build outputs are missing,
  and report unavailable GLM device counts as skips.

## Dependency and environment

All native builds use unchanged upstream ggml
`456172ec733a135778adcd32d00e576a58232e45` (0.24.0). Git status/diff checks were
clean. No fetched source was patched or rewritten. TensorSharp-owned code
contains the new behavior.

Local validation used macOS arm64 and .NET SDK 10.0.302. Remote validation used
the supplied Linux VM, CUDA 12.8, and NVIDIA A40s, with isolated build outputs and
`/workspace/locks/gpuN.lock`. The main CUDA validation snapshot is under
`/workspace/ts-codex-branch-completion-20260917`; its initial integrated native
SHA-256 is `0e75fe2580062b8335e0197f8a71021248aa8c665529a9e62f89aa205c92625b`.
The final native rebuild used source `90701d93046770335ece4bf8bff981ee7e3683f1`;
all 4,421 tracked files matched the source manifest. Its SHA-256 is
`78e2ce677f68668f6e1c6c3d966a1cce6a96b6bb8042f9b2cf97df89f4a61a51`.
The managed-only Gemma lifetime follow-up is source
`668cc48167e6a9e8bd81460ef6e5f7ecf39ade9b`; it keeps that same native binary.
See [source verification](remote/final-source-verification.json) and
[native identities](remote/final-native-sha256.txt), and
[managed follow-up verification](remote/gemma-followup-source.json).

## Validation

| Check | Actual result |
|---|---|
| Final combined portable managed suite | 5,091 passed, 4 explicitly skipped, 0 failed |
| Benchmark comparison/ABBA tests | 29 passed |
| Final integrated Linux/CUDA native suite | 32 passed, 3 skipped (4-, 7-, 8-device TP), 0 failed |
| Final combined GLM 1/2/3-GPU, refusal, fixture-helper and fallback checks | 41 passed, 0 skipped |
| CUDA unsupported-attention/Hunyuan serving regressions | 9 passed, 0 skipped |
| Qwen 3.5-9B Q8 CUDA image reuse, inject refusal, primary scratch | 4 passed, 0 skipped |
| Gemma 4-12B CUDA image/audio after reused prefix | 5 passed, 0 skipped |
| Gemma E4B optimized-JIT lifetime and batched parity | Metal: 2 passed; CUDA: 2 passed; forced-GC regression fails before fix |
| Qwen 3.8 real model, three A40s | Default and forced broadcast: widths 2–8 × 96 rows each, bit-exact full logits |
| Qwen row operator probe | All 91 optimized cases exact across widths 2–8; 61 old batched controls differ |
| Qwen 3.8 tiny fixture on CUDA, default reductions | 20 passed, including two-device checkpoint and verify widths 1–8 |
| Same CUDA fixture, forced broadcast reductions | 20 passed |
| Qwen fixture after final GLM/DeepSeek native merge | 20 passed, 0 skipped |
| Same fixture on ARM CPU | 19 passed, 1 CUDA-only skip; before CPU fix: 18 passed, 1 failed, 1 skipped |
| Radix property/conformance suite | 291 passed, 8 model-gated skips; 20,000 seeds and 5,577,492 comparisons |
| Radix M0 focused regressions | 51 passed |
| Radix real-model Metal conformance/accounting | 5 passed, 0 skipped: Gemma E4B Q8, Qwen3.5 9B Q8, GPT-OSS 20B Q8 |
| Radix real-model CUDA conformance/accounting | 5 passed, 0 skipped: Gemma E4B Q8, Qwen3.5 9B Q8, GPT-OSS 20B MXFP4 |
| Radix core coverage | 100% lines; PrefixTree 99.31% branches, other requested core classes 100% |
| Radix BG-15 | All six gates passed, zero allocation per probe |
| GLM native layer split | 1/2/3 GPUs passed with exactly equal full logits; negative control fails |
| GLM real-tokenizer checks | All 422 cases against each of GLM-5.2, GLM-5.3, GLM-5.3-Flash |
| DeepSeek focused managed suite | 509 passed, 5 skipped |
| DeepSeek Linux/CUDA native suite | 27 passed, 4 multi-device skips |
| DeepSeek long sparse fixture | 100/100; 8,500 tokens and 32 identical greedy continuation tokens |
| DeepSeek pread/page-walk comparison | 66/66 each; 1,341 full logit rows byte-identical |
| DeepSeek automatic microbatch | CPU 256, CUDA 1,024, explicit 32; DSpark verify/cache-head checks passed |

The complete integrated portable command was:

```sh
dotnet test InferenceWeb.Tests -c Release --no-restore \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true \
  --filter 'Category!=Bench&Requires!=Models&Requires!=Cuda&Requires!=Mlx'
```

The native library was built separately first. CUDA native runs exposed three
A40s; the 4-, 7-, and 8-device native lanes were skipped and are not counted
as passed. Both native and managed builds completed with zero errors. The four portable skips are
fixture/export-hook prerequisites, not passing native/model coverage. Exact
skip reasons and counters are in [local-validation.json](local-validation.json).
Raw CUDA results and test logs are in [remote](remote), with executable
reproduction scripts alongside this report.

Further task evidence:
[radix](radix/README.md),
[Gemma lifetime regression](radix/gemma-batched-lifetime.md),
[GLM](../glm-branch-completion-2026-09-17/README.md),
[DeepSeek](../dsv41-perf-completion-2026-09-17/README.md),
[Qwen verify rows](../../../eng/validation/qwen38_mtp_followup/verify-row-kernels-20260917/README.md).

## Benchmark interpretation and remaining limits

The DeepSeek operator comparison measured 34.904 ms sparse versus 1,548.010 ms
tiled for 512 queries, 33,536 keys and 64 heads, both checked against the strict
F32 reference. This is an operator/fixture result; full-checkpoint cold-load and
TTFT gains were not measured.

Two fresh GPT-OSS MXFP4 processes with fixed arrival order, concurrency 1/4/8
and 32 output tokens produced identical tokens and workload shapes in all four
rows, including the solo request after concurrency. Their aggregate decode
rates were approximately 143–146, 345–349 and 567–577 tokens/s. The comparison
used `--require-concurrent-identity --max-regression-percent 100` to check
correctness, **not** a performance regression gate. Other work was active on
the host, and these are repeat measurements of the same build, not an old/new
speed comparison.

The real Qwen3.8 parity run used UD-Q2_K_XL, one 3,248-token prompt, 96
continuation rows and three A40s. Both the optimized A40 and forced broadcast
paths were bit-exact at all seven widths. The old batched control changed all
96 rows at every width and produced 5–10 greedy-token differences, confirming
that the comparison can detect the original problem. The row operator probe
checks lower-level reductions separately. These are correctness results;
concurrent work makes their timings unsuitable as performance claims. See
[qwen-validation.json](qwen-validation.json).

One pre-existing native check remains failing on ARM CPU:
`deepseek41-moe-tensor-parallel`, F16/F16, two ranks. Its result exactly matches
an independently partitioned reference, but upstream ARM F16 accumulation
differs from the unsplit graph (relative L2 `5.11463e-4`, max absolute
`2.32743e-7`). The same Linux check passes. No threshold was relaxed; see the
DeepSeek report for source identity and diagnosis. This is **not a passing
local native suite**.

Windows, iOS, Vulkan, non-A40 CUDA hardware, the unavailable eight-device lane,
and full-model GLM forward quality/throughput were not qualified by this work.
