# Rejected cheaper compensated reduction

This isolated Kahan/FMA variant is not integrated. It passes the original prefix 5 confidence check, all 17 confidence checks in the prefix sweep and the existing CUDA F32 precision suite, but introduces a new prefix 2 target-verification failure (maximum absolute error 0.0001776814). Prefix 11 also remains failed. Overall 69 of 71 comparisons pass, versus 70 of 71 for the earlier product-residual compensation.

The paired kernel microbenchmarks still retain approximately 6–17% regressions in several ordinary projection cases; indexed larger cases are near parity. No numerical or performance gate is waived. Source, binaries hashes, logs, all benchmark arrays and both failed output comparisons are retained. Upstream ggml remains unchanged at `456172ec733a135778adcd32d00e576a58232e45`.
