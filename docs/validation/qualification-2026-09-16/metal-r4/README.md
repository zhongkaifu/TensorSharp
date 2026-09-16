# Apple Metal r4 verification — 2026-09-16

Actual Apple M5 Pro execution is now recorded, rather than treating Apple as unavailable. This is partial fixture coverage; release and trained-model qualification remain open.

Production source matches the [r4 source manifest](../cuda-r4/source.json). Unmodified ggml is `456172ec733a135778adcd32d00e576a58232e45`. Fresh Release native SHA256 is `b1ed8e429e1408cb5372807f09c21cf6b46c9bca41a5259ea4715d8b92b3c1f4`; [build.json](build.json) records compilation and object hashes. Test-only changes allow explicit Metal selection and retain the original numerical expectations. Python source copies and hashes are included.

| Check | Actual result |
| --- | --- |
| QSA on Metal | 20 numerical and 21 invalid-input cases passed |
| MTP draft operator on Metal | 592 checks passed across capacity32/512 and alignment4/64 |
| Target state on Metal | 132 checks passed, GDN32 geometry |
| Managed Qwen subset | 32 passed, 1 failed, 1 explicit two-CUDA-GPU skip |
| Initial native CTest | 16 passed, 2 failed |
| Attention allocation with actual GPU access | Passed in separate rerun |

The sandbox could not initially create a Metal command queue. `qsa-metal-r1` records unavailable execution (exit77, zero cases); `qsa-metal-r2` records actual M5 Pro execution. The separate attention rerun clears only that sandbox-related allocation failure. The original CTest record remains failed.

The CPU F16 tensor-parallel check retains its strict failure: relative L2 `0.000511463`, maximum absolute error `2.32743e-7`. Earlier unchanged-source baseline reproduced it; no tolerance was widened.

The managed `SharedPrefixChunking_MatchesWholePromptForEachDistinctSuffix` check fails comparing a whole20-token prefill against16+4 chunks. Correctly decoded F16 raw keys differ by up to `0.00390625`, and full logits by `0.0020632743835449219`; both argmax tokens are249. Same argmax does not clear full-logit/cache equality. A diagnostic with Metal tensor API and fusion disabled also fails (`0.001953125` raw-key, `0.0014123916625976562` logits). The original byte and float equality assertions remain intact.

The earlier r3 diagnostic mistakenly interpreted F16 cache bytes as F32; its raw-key magnitude is invalid. The r5 result corrects only that diagnostic conversion. r4 was a test compilation failure due to ambiguous `Half`; no tests ran. Production native code is identical throughout.

NumPy emitted floating-point status warnings during some reference matrix multiplications. The comparisons require finite actual and reference outputs and passed their original gates; warnings are retained in tool/VM logs. These results do not establish full-model Metal quality, memory, or performance.
