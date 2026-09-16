# DSpark four-accumulator compensation experiment

This isolated candidate is rejected for performance and is not integrated. The original prefix 11 full-vocabulary comparison remains failed at its unchanged tolerance.

The candidate passes 70 of 71 original checks across prefixes 1–17, including every confidence comparison, and passes the CUDA precision fixture. Four independent compensated sums preserve F32 product residuals and use error-free two-part additions before the warp reduction. Sampled projection errors against independent CPU FP64 dots are approximately 3–10 times smaller than the baseline. This does not replace the original full-model oracle.

Forty alternating CUDA-event sample pairs per case, each measuring 20 warmed kernel calls, yield median candidate/baseline latency ratios of 1.01–1.79. Ordinary 4096-wide matmuls regress approximately 79% at one column and 64% at eight columns. These GPU 0 kernel measurements were taken while other VM campaigns were active and do not qualify quiet full-model performance. The source, complete measurement arrays, original failed fixture arrays and binary hashes are retained.

The first runner invocation supplied a comma-separated prefix argument instead of the required separate integer arguments. It stopped before fixture execution; its argument-error log and original runner are retained. The corrected runner completed the numerical, precision and benchmark checks. No skipped or failed invocation is counted as a pass.

Dependency: unchanged upstream ggml `456172ec733a135778adcd32d00e576a58232e45`; the r3 native library objects are reused except for the explicitly replaced precision CUDA object.
