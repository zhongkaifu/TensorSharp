# Quantized tensor-parallel optimization experiments

Release and performance qualification remain blocked. These are isolated TensorSharp-owned prototypes, not a final combined build. They retain the unchanged full-weight ggml oracle and original numerical tolerances. Upstream ggml is unchanged at `456172ec733a135778adcd32d00e576a58232e45`.

All three prototypes pass 36 exact gate/up projection and strict full-MoE comparisons on physical GPU6. Paired gate/up shares activation quantization and sorting. The compact 32-row kernel is rejected because its microbenchmarks worsen narrow-strip performance. The captured paired version additionally passes 13 scratch-growth, failure/recovery, graph-reuse and teardown cases, and compute-sanitizer memcheck reports zero errors. Its counters record 228 captures, 452 replays, 228 invalidations, zero live graphs and scratch bytes after teardown, and 45,651,456 peak owned scratch bytes.

The captured version still regresses narrow-strip median timing by approximately 15–53%; Q4_K two-rank geometry at 16 tokens regresses approximately 10%. No performance gate is waived. Each case records 60 alternating warmed before/after measurements, including output copy and host dispatch, while other VM GPU campaigns were active. These are diagnostic per-strip MoE measurements, not quiet or full-checkpoint tensor-parallel benchmarks.

Each variant includes exact source snapshots, source/binary hashes, build commands, correctness logs and all benchmark arrays. The captured variant includes its completed memcheck log. These results do not clear the earlier capture-enabled MTP graph-update failure, do not establish physical four/seven/eight-GPU execution, and do not qualify nonaligned unsupported strip shapes.
