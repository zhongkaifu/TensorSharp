# Original-source GDN chunk candidate

This is a separate candidate. No default, environment variable or model route
selects it automatically. It does not modify the frozen sequential adapter or
the unchanged ggml dependency.

The `upstream/` subtree contains verbatim files from gpillon/ignis revision
`e600a7d9e0d5921d767a6cb464373583b3cdcaf4`, including its original LICENSE and
NOTICE files. `official-source-manifest.json` records every file's SHA-256 and
Git blob SHA. Keep these licenses, notices and provenance files with any
redistribution. The TensorSharp-owned `.cu` only instantiates the original
three chunk stages and adds representation bridges; it does not rewrite their
kernel bodies or change their arithmetic.

The original 64-token algorithm has three stages:

1. Prepare the triangular transform and produce BF16 W/U plus F32 cumulative
   decay, using the original BF16/TF32 tensor-core computation.
2. Carry F32 recurrent state across chunks, writing BF16 `v_new` and BF16
   per-chunk state snapshots for the output stage.
3. Compute outputs with BF16 Q/K/H/V operands and TF32 intermediates, with F32
   accumulation and final BF16 output.

This recipe differs from the first sequential adapter. In particular, original
NInfer materializes normalized Q/K in BF16 whenever the span contains a full
64-token chunk, whereas the sequential adapter keeps normalized Q/K in F32.
Consequently, component tolerance checks do not establish identical full-model
outputs or preserved task quality. The original Q/K normalization operation's
own reduction order remains a separate integration concern.

`CudaNinferGdnChunk.RunNormalized` accepts packed normalized F32 Q/K and BF16
represented V, applies the original BF16 materialization, and processes one
sequence containing only complete 64-token chunks. Caller-owned workspace
contains all bridges and stage scratch. Tail tokens, mixed/partial batches,
speculative state replay and model integration are intentionally not selected
by this standalone class. At T=4096 its total workspace is 368.75 MiB, including
the input/output representation bridges; the stage-only original scratch is
240.75 MiB. Workspace allocation must be included in future capacity checks.

Build the standalone fatbin with the reusable tool:

```
python3 eng/validation/build-ninfer-gdn-chunk-module.py --output artifacts/tensorsharp_ninfer_gdn_chunk.fatbin
```

The unchanged original prepare header requires C++20 template lambdas. The
builder verifies all source hashes before invoking CUDA 13+ NVCC for sm_120a
(or explicitly sm_121a), with fast math disabled. Build logs and module evidence
are generated next to the requested output. Compilation is not device validation.

The separate `eng/validation/NinferGdnChunkProbe` uses an independent FP64
recurrence and the original revision's published output/state acceptance
criteria. It checks AA repetition after different scratch poison, inputs,
state/output/workspace guards and the multi-job output schedule. Its optional
sequential comparison is component-only diagnostic timing, not an end-to-end
speedup or model-quality certification. All results must retain failed and
unavailable cases.
