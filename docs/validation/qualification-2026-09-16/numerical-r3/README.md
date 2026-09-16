Release remains blocked. The original CUDA DSpark prefix-5 confidence gate still
fails; a wider CPU prefix sweep also found a full-vocabulary verification failure
at prefix 11. No tolerance or independent reference was changed. The corrected
`--keep-going` runner reports failed status and exits nonzero when any numerical
comparison fails; both recorded failures demonstrate that behavior.

| Actual execution | Result |
|---|---|
| Earlier reviewed DSpark candidate, x86 CPU, prefix 5 plus state/ring checks | 670/670 passed; binary `7d0c776a76c3a9225ee82186fc84aec1f4a65adc78e3a2eb42d0d4f2c767199a` |
| Combined r3, x86 CPU, prefixes 1–17, reference checks only | 70/71 passed; prefix-11 full-vocabulary verification failed |
| Combined r3, physical A40 GPUs 0 and 6, prefixes 1/5/17 plus state/ring checks | 688/689 passed; original prefix-5 confidence failed |
| macOS ARM, unchanged strict TP checkpoint geometry, Q2_K/Q3_K and Q4_K/Q6_K, CPU ranks 2/4/7/8 | 72/72 passed; CPU execution only |
| macOS ARM, FP8/FP4/BF16 quantizer and candidate-mask fixture | Passed |
| macOS ARM, explicit F32 matmul and attention after owned ARM fix | 2/2 native suites passed, including exact column/query invariance for widths 1–8 |

Combined r3 native SHA256:
`b6cf00377fca4154f8d318677e201044bf93adfa97ef26b0d6742ce1d4cbe751`.
The remote driver verifies that the binary and input files remain unchanged
during each run. The recorded clean ggml revision is
`456172ec733a135778adcd32d00e576a58232e45`.

The physical split fixture confirms target feature layers `[1,3,4]` reside on
logical devices `[0,0,1]`. Thus the existing TensorSharp graph pinning and transfer
path actually executed across two GPUs in this new binary. All ring, rewind,
slot-isolation and checkpoint state comparisons in that run passed. This is a
small deterministic fixture, not trained-head acceptance or throughput evidence.

The remaining numerical failures are localized:

- CUDA prefix 5: confidence max absolute error `0.00010269880294799805`, relative
  L2 `6.659053287868927e-05`, against the original per-element `atol=rtol=2e-5`.
  Exactly one committed draft KV value differs: stage 1, token 2, channel 35 is
  `0.4375` natively versus `0.40625` in the independent reference. The diagnostic
  oracle supplied with native ring values passes; it does not replace the
  independent failed gate.
- CPU prefix 11: the sixth verified token fails 44 vocabulary elements; maximum
  absolute logit error `6.306171417236328e-05`, relative L2
  `9.624329055627613e-06`. Layer traces locate the first quantized discrepancy at
  target layer 2, verify row 5, channel 19: native `-0.0107421875`, reference
  `-0.01171875`. Earlier layer discrepancies are about `1e-6`; this single cache
  bin difference grows through the remaining layers. All 17 CPU draft-confidence
  comparisons pass, including original prefix 5 (maximum error `1.7881393e-7`).
- Broader macOS ARM CPU TP coverage additionally fails the first F16/F16
  two-rank, one-token case: relative L2 `0.000511463`, maximum absolute error
  `2.32743e-7`. The same failure is reproduced with the original owned matmul
  source, before the ARM fix. This is separate from the passing mixed-quantized
  checkpoint cases; later standard-format cases after this failure were not run.
- The retained CUDA Q2_K/Q3_K and Q4_K/Q6_K full-dimension TP failures remain
  unresolved. No partitioned-reference substitution was imported from the
  independent VM investigation.

The integrated precision policy keeps widths 1–8 on per-column/per-query
arithmetic with an invariant key partition. This fixes batch-size-induced
rounding during small speculative verification. Broader widths still require
qualification. The additional ARM fix accumulates neighboring F32 products
before combining long SIMD partials. The original cancellation fixture lost
half of a `2^-16` weight residue (`0.0078125` versus expected `0.015625`); its
unchanged expectations pass after the fix. Upstream sources remain untouched.

The ARM benchmark runs 12 alternating old/new pairs, each containing 20,000
dot products against 64 resident vectors. At inner lengths
128/256/1024/2048/4096/5120/8192, the owned-to-upstream median time ratios are
0.9734/0.9919/1.0002/1.0076/0.9987/0.9975/0.9954. The largest measured regression
is 0.76%; these short resident-data measurements do not qualify memory-bound
serving, trained checkpoints, concurrent requests, CUDA precision performance,
or an end-to-end no-regression gate. Full pair timings and benchmark source are
retained alongside the logs and manifests.

Native CPU reproduction (AppleClang 21, Release):

```sh
cmake -S TensorSharp.GGML.Native -B /tmp/ts-numerical-build -DCMAKE_BUILD_TYPE=Release -DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON -DTENSORSHARP_GGML_NATIVE_ENABLE_METAL=OFF
cmake --build /tmp/ts-numerical-build --target GgmlOpsDsv41TpTest GgmlOpsDsv41QuantTest GgmlOpsMatmulPrecisionCpuTest GgmlOpsAttentionPrecisionCpuTest -j 6
/tmp/ts-numerical-build/GgmlOpsDsv41TpTest --checkpoint-shape
ctest --test-dir /tmp/ts-numerical-build -R cpu-explicit-f32 --output-on-failure
c++ -O3 -DNDEBUG -std=c++17 -I TensorSharp.GGML.Native -I ExternalProjects/ggml/include docs/validation/qualification-2026-09-16/numerical-r3/arm-dot-benchmark.cpp /tmp/ts-numerical-build/libggml-cpu.a /tmp/ts-numerical-build/libggml-base.a -framework Accelerate -lm -o /tmp/arm-dot-benchmark
/tmp/arm-dot-benchmark
```

Remote DSpark invocations used
`PYTHONPATH=/workspace/tensorsharp-no-patch-20260915/fixture-python-deps`,
`/usr/local/bin/python -B`, the deterministic fixture at
`/workspace/ts-fix-dspark/repo/fixtures/text-f32-small`, and the r3 library at
`/workspace/ts-codex-20260916-r3/repo/TensorSharp.GGML.Native/build/libGgmlOps.so`.
CPU used `CUDA_VISIBLE_DEVICES=` with `--backend CPU --reference-only
--reference-prefixes 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 --keep-going`.
The physical split used `CUDA_VISIBLE_DEVICES=0,6` with `--backend CUDA --gpus 2
--require-split-dspark --reference-prefixes 1 5 17 --native-ring-control
--keep-going`. Fresh output directories and complete reports are recorded in
the attached JSON artifacts. Prefix-11 tracing adds `--trace-native` and retains
independent reference layer outputs for diagnosis.
