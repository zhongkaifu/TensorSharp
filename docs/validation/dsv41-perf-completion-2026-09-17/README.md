# DeepSeek load and prefill branch completion — 2026-09-17

The reviewed branch `claude/dsv41-perf` at `d8222d94` already contained both
`claude/dsv41-perf-load` (`17b86f19`) and `claude/dsv41-perf-prefill`
(`94afb376`). Their delivered scope is the host-file pread warmer, pageable
offloaded experts, conditional uploaded-page eviction, default owned-F32 sparse
prefill, and automatic GPU microbatch selection. The child worktrees had no
tracked edits; their existing untracked build directories were preserved.

Review found one failure-path defect in the new warmer: an exception in a
spawned reader (including a bookkeeping allocation failure) escaped the thread
and terminated the process. Even an exception in the caller's reader could
unwind across still-joinable workers. Commit `05daf824` contains each worker's
exception without allocating in its catch, closes descriptors, joins every
started thread, then reports the failure. The existing model loader/background
warm error policies can consequently handle it. Fault injection covers both
reader types, standard and nonstandard exceptions, and a subsequent successful
warm. The unchanged header aborts under the same test with `SIGABRT`; see
[the red control](warm-exception-red-control.log).

## Dependency and build identity

Both fresh native builds use unchanged upstream ggml
`456172ec733a135778adcd32d00e576a58232e45` (0.24.0). Its checkout was clean
before and after validation. No fetched source was patched or rewritten.

Local: macOS 27.0 arm64, Apple Clang 21.0.0, .NET SDK 10.0.302; Release,
Metal enabled, native tests enabled. The final local library SHA-256 is
`c3599cb188635d2b5985a5d7c4ba4d9c0af120e19840c0b2d75c297ee4af395d`.

Remote: a fresh source copy under
`/workspace/ts-codex-20260917-dsv41-review/repo`, build output under
`/tmp/ts-codex-20260917-dsv41-native-make`; Release, CUDA `86-real`, NCCL on,
Vulkan off, GCC 13.3.0, CUDA 12.8.93. Existing `ts-dsv41perf-*` workflows and
outputs were not modified. Final library SHA-256:
`8e122c52a9a308caa5fad54b31bf07c78f97ba2775e7b4bd569c1c0168829e21`.

## Local validation

| Check | Result |
|---|---|
| Fresh full native build | Passed |
| Whole native CTest before the review fix | 19 passed, 1 failed; unresolved ARM F16 limitation below |
| Rebuilt load warmer, automatic ubatch and owned F32 attention checks | 3 passed, 0 skipped |
| Managed DeepSeek suite with pinned tiny CPU/DSpark fixtures | 509 passed, 5 skipped |
| Focused native-boundary/DSpark managed suite | 10 passed, 1 CUDA-only skip |

Logs: [full native](macos-native-ctest.log), [focused native](macos-focused-ctest.log),
[managed DeepSeek](macos-managed-deepseek.log).

The managed skips were CUDA auto-ubatch, two backend-gated KV dtype cases and
two teacher-export fixture cases. They are not passing coverage. The CPU
fixture covers auto-ubatch 256, native output-array boundaries, speculative
verify/rewind, greedy continuation and two-slot separation. The native warmer
test also passes on Linux, including cold reads, resident-range skipping,
truncation errors and cancellation.

### Pre-existing ARM F16 tensor-parallel limitation

`deepseek41-moe-tensor-parallel` fails the strict full-weight comparison on the
local ARM CPU at F16/F16, two ranks, one token: maximum absolute error
`2.32743e-7`, relative L2 `5.11463e-4` (required `< 1e-5`). The executor's
output exactly equals the independently sliced ggml reference. Every source
of this standalone executable is unchanged from `06665adc`, the parent of
the load/prefill work.

[The diagnostic run](macos-tp-diagnose.log) isolates this to the down
projection. Gate/up and SwiGLU values are bit-identical before splitting;
feeding the same hidden values through split down projections reproduces the
failure. Upstream's ARM NEON F16 dot uses F16 vector accumulators, whose
rounding changes when its dot is partitioned. Against the independent F64
oracle, the unsplit and TP graphs have relative L2 `0.00325811` and
`0.00323592`, respectively. This is a backend precision limitation, not an
incorrectly rounded test reference. The implementation and tolerance were
not changed to conceal it.

## Fresh Linux/CUDA validation

One A40 (physical GPU 3) was exposed to each GPU check, under
`/workspace/locks/gpu3.lock`. [The commands](run-remote-validation.sh) and
[complete step status](remote/summary.txt) accompany the raw results.

| Check | Result |
|---|---|
| Fresh full native CUDA build | Passed |
| Native CTest | 27 passed, 4 skipped, 0 failed, 133.06 seconds |
| Actual native C ABI boundaries | 32/32 passed |
| Tiny CUDA fixture plus 8,500-token sparse prefill | 100/100 passed |
| Host-mapped Engram + two CPU-MoE layers, pread warm | 66/66 passed |
| Same placement, original page-walk warm | 66/66 passed |
| Warm-mode numerical parity | All 1,341 × 256 captured logits byte-identical |
| DSpark loader auto/explicit ubatch and verify | CPU auto=256, CUDA auto=1024, CUDA explicit=32; all three passed |

The four CTest skips are the 2/4/7/8-device TP numerical cases; no multi-GPU
coverage is claimed. Unlike ARM, the Linux x86-64 CPU TP suite passes its
unchanged strict full-weight reference checks. Logs:
[CTest](remote/linux-cuda-ctest.log), [C ABI](remote/native-c-abi.json),
[long fixture](remote/cuda-fixture-long.json),
[pread](remote/cuda-cpumoe-host-pread.json),
[page walk](remote/cuda-cpumoe-host-pagewalk.json),
[warm-mode comparison](remote/warm-mode-parity.json), and
[auto-ubatch](remote/auto-ubatch.log).

The long prompt reaches 16,640 attention keys. Sparse default versus
`TS_DSV41_SPARSE_FA=0` differs by at most `1.19e-6` over 33 logit rows; all
32 greedy continuation tokens agree. The logits are not byte-identical, so
this exercises the changed attention path.

An initial standalone auto-ubatch harness assertion incorrectly expected
`ForwardSpec` success to be zero (unlike `Forward`, its success is one).
The initial failed log is retained. After correcting only the harness,
all three configurations loaded, returned the expected ubatch, and verified
three tokens with finite logits and cache head 8. The final reproduction
[script](auto-ubatch.py) contains the correct ABI convention.

### Operator benchmark

`GgmlOpsCudaAttentionPrecisionTest --benchmark-dsv41-prefill 512 33536 64 N`
checks each arm against a decomposed F32 reference before timing. At head
width 512 and 640 visible keys per query, after three warmups:

| Arm | Samples | Median | Maximum absolute error | Relative L2 |
|---|---:|---:|---:|---:|
| Default owned sparse F32 | 5 | 34.904 ms | 1.1175871e-7 | 7.445791e-7 |
| `TS_DSV41_SPARSE_FA=0`, owned tiled F32 | 3 | 1,548.010 ms | 8.9406967e-8 | 4.7586684e-7 |

Both pass the original numerical bound. Raw samples and the production gate
selection are in [sparse](remote/sparse-prefill-bench.log) and
[tiled](remote/tiled-prefill-bench.log). This is one operator on one A40 in a
shared VM, not full-model prefill throughput or time to first token.

## Scope and remaining measurement limits

Synthetic fixtures establish the tested execution paths, not trained model
quality or production throughput. The earlier operator/storage measurements
in the model card remain separately attributed to their original runs.
This completion does not establish full 414 GiB checkpoint cold-load speed,
full-model time to first token, multi-GPU numerical parity, or Windows/Vulkan
coverage. A full-checkpoint comparison of the automatic page-cache policy
and automatic ubatch remains unmeasured. Existing independent-reference
limitations of DSpark and tensor parallelism remain explicit in the model
validation documentation.
