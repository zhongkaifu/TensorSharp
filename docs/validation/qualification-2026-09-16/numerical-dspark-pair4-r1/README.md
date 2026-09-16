# Rejected independent pair accumulators

This isolated CUDA experiment preserves adjacent opposite-sign products in four F32 accumulators and performs the final merge in FP64. It is not integrated. It passes 67 of 71 original prefix comparisons but retains the prefix 11 verification failure and introduces target, verification and confidence failures at prefix 17. Numerical acceptance fails; compiled benchmark and precision binaries are not claimed as executed passes. No tolerance or oracle was changed.

Source snapshots, build records and failing comparison arrays are retained. Upstream ggml is unchanged at `456172ec733a135778adcd32d00e576a58232e45`.
