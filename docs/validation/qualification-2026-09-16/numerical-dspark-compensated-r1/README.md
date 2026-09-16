# DSpark compensated floating-point reduction experiment

The original prefix 11 full-vocabulary target-logit gate remains failed. These isolated CUDA prototypes are not integrated and do not qualify a release or trained checkpoint.

The exact r3 baseline passes 64 of 71 original comparisons across prefixes 1–17. It fails confidence at prefixes 3–8 and target verification at prefix 11. The compensated F32 prototype passes 70 of 71 comparisons: all 17 confidence checks pass, while prefix 11 retains maximum absolute error 6.3061714e-5 against the unchanged F32 oracle. No tolerance or original reference is altered.

Independent variants using FP64 accumulation only for ordinary matmuls or only for indexed matmuls each pass all 12 prefix 5 checks. Compensation uses F32 product residuals and two-part summation, reducing sampled projection error against an independent FP64 dot by approximately 3–10 times. Its 40 paired CUDA-event timing samples per case remain mixed: approximately 0.79–1.21 times baseline, including regressions above 5%. All arrays, exact source/build records and binary hashes are retained. These device-kernel timings exclude full-model costs and were recorded while other VM GPU campaigns were active.

Dependency: unchanged upstream ggml `456172ec733a135778adcd32d00e576a58232e45`, using the r3 native library objects except the explicitly replaced precision CUDA object. The original failed reports and arrays remain in this archive.
