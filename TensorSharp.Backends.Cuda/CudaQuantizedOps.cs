using System;
using System.Collections.Generic;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public static class CudaQuantizedOps
    {
        private sealed class DeviceWeight : IDisposable
        {
            public IntPtr DevicePtr;
            public long RawBytes;
            public int GgmlType;
            public long Ne0;
            public long Ne1;
            public int DeviceId;

            public void Dispose()
            {
                if (DevicePtr != IntPtr.Zero)
                {
                    CudaDriverApi.cuMemFree(DevicePtr);
                    DevicePtr = IntPtr.Zero;
                }
            }
        }

        private static readonly object Sync = new object();
        private static readonly Dictionary<CacheKey, DeviceWeight> Cache = new Dictionary<CacheKey, DeviceWeight>();

        private sealed class ScratchAllocation
        {
            public IntPtr Ptr;
            public long Bytes;
            public readonly List<IntPtr> Retired = new List<IntPtr>();
        }

        private sealed class ScratchPool
        {
            public readonly Dictionary<CudaAllocator, ScratchAllocation> Allocations =
                new Dictionary<CudaAllocator, ScratchAllocation>();
        }

        // Scratch is allocator-owned rather than device-owned: different
        // allocators have different streams and may execute concurrently.
        // Growth retains the previous generation because CUDA graphs bake raw
        // scratch pointers into kernel nodes. Geometric growth bounds all
        // retained generations to less than the current allocation for gradual
        // growth; CudaAllocator.Dispose releases them after synchronizing.
        //
        // q8_1-quantized activation scratch for the dp4a Q8_0 matmul path.
        private static readonly ScratchPool Q81Scratch = new ScratchPool();
        private static IntPtr EnsureQ81Scratch(CudaAllocator allocator, long bytes)
            => EnsureScratch(Q81Scratch, allocator, bytes);

        // Row-batched quantized matmul (weight-reuse across small row counts).
        // On by default; set TS_CUDA_QMM_BATCHED=0 to force the legacy per-row
        // kernels (A/B benchmarking).
        internal static readonly bool BatchedMatmulEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_QMM_BATCHED"), "0", StringComparison.Ordinal);

        // Single-row (decode) quantized matmul for generic quant types (K-quants,
        // IQ3_XXS/IQ3_S/IQ2_S/IQ4_XS -- anything without its own dedicated decode
        // kernel). One block per output column (full blockDim threads), not one
        // warp: see the TS_CUDA_QMM_VEC=0 A/B note at its call site for why the
        // row-batched kernel's warp-per-column design loses here. On by default.
        internal static readonly bool VecMatmulEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_QMM_VEC"), "0", StringComparison.Ordinal);

        // Large-row (prefill-sized) matmuls: dequantize the weight ONCE to f16 and
        // run a tensor-core cuBLAS GEMM (mirrors ggml_cuda's dequant+cuBLAS route).
        // The block-tile quant kernels re-read the whole weight every rows/tile-rows
        // output tile, so a 2048-token prefill costs hundreds of full weight sweeps;
        // this path costs ~3 sweeps (int8 read + f16 write + GEMM read) regardless
        // of row count. TS_CUDA_QMM_F16GEMM=0 disables; MIN_ROWS/MAX_MB tune the
        // activation-row threshold and the f16 scratch cap (the LM head exceeds it).
        internal static readonly bool F16GemmEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_QMM_F16GEMM"), "0", StringComparison.Ordinal);
        internal static readonly int F16GemmMinRows = EnvInt("TS_CUDA_QMM_F16GEMM_MIN_ROWS", 32);
        internal static readonly long F16GemmMaxWeightBytes = EnvInt("TS_CUDA_QMM_F16GEMM_MAX_MB", 768) * 1024L * 1024L;

        // ggml-style warp-cooperative Q8_0 -> F16 whole-weight conversion for
        // RunF16Gemm. Set TS_CUDA_Q80_F16_DEQUANT=0 to retain the generic
        // one-thread-per-scalar dequantizer for controlled A/B comparisons.
        public static bool Q80F16DequantEnabled { get; set; } =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_Q80_F16_DEQUANT"), "0", StringComparison.Ordinal);

        private static int EnvInt(string name, int fallback)
        {
            string s = Environment.GetEnvironmentVariable(name);
            return !string.IsNullOrEmpty(s) && int.TryParse(s, out int v) && v > 0 ? v : fallback;
        }

        // Per-allocator reusable f16 scratch for the dequant+GEMM path (weight
        // and activation panels).
        private static readonly ScratchPool WF16Scratch = new ScratchPool();
        private static readonly ScratchPool AF16Scratch = new ScratchPool();

        // Split q8_1 activation scratch for the cp.async MMQ path: dense qs byte
        // rows + separate float scales (see ts_quantize_q8_1_split_rows_f32).
        private static readonly ScratchPool Q81SplitQsScratch = new ScratchPool();
        private static readonly ScratchPool Q81SplitDScratch = new ScratchPool();

        private static IntPtr EnsureScratch(ScratchPool pool, CudaAllocator allocator, long bytes)
        {
            lock (Sync)
            {
                if (pool.Allocations.TryGetValue(allocator, out ScratchAllocation scratch) &&
                    scratch.Bytes >= bytes)
                {
                    return scratch.Ptr;
                }

                allocator.Context.MakeCurrent();
                long doubled = scratch == null || scratch.Bytes > long.MaxValue / 2
                    ? 0
                    : scratch.Bytes * 2;
                long alloc = Math.Max(bytes, Math.Max(64 * 1024L, doubled));
                CudaDriverApi.cuMemAlloc(out IntPtr ptr, new UIntPtr((ulong)alloc)).ThrowOnError();

                if (scratch == null)
                {
                    scratch = new ScratchAllocation();
                    pool.Allocations.Add(allocator, scratch);
                }
                else if (scratch.Ptr != IntPtr.Zero)
                {
                    scratch.Retired.Add(scratch.Ptr);
                }

                scratch.Ptr = ptr;
                scratch.Bytes = alloc;
                return ptr;
            }
        }

        internal static void ReleaseScratch(CudaAllocator allocator)
        {
            lock (Sync)
            {
                allocator.Context.MakeCurrent();
                ReleaseScratch(Q81Scratch, allocator);
                ReleaseScratch(WF16Scratch, allocator);
                ReleaseScratch(AF16Scratch, allocator);
                ReleaseScratch(Q81SplitQsScratch, allocator);
                ReleaseScratch(Q81SplitDScratch, allocator);
            }
        }

        private static void ReleaseScratch(ScratchPool pool, CudaAllocator allocator)
        {
            if (!pool.Allocations.TryGetValue(allocator, out ScratchAllocation scratch))
                return;

            pool.Allocations.Remove(allocator);
            if (scratch.Ptr != IntPtr.Zero)
                CudaDriverApi.cuMemFree(scratch.Ptr);
            foreach (IntPtr ptr in scratch.Retired)
                CudaDriverApi.cuMemFree(ptr);
        }

        private static void RunF16Gemm(
            CudaAllocator allocator, CudaKernels kernels,
            IntPtr weightPtr, int ggmlType, IntPtr inputPtr, IntPtr resultPtr,
            int inDim, int outDim, int rows)
        {
            long wElems = (long)inDim * outDim;
            long aElems = (long)rows * inDim;
            IntPtr aF16 = EnsureScratch(AF16Scratch, allocator, aElems * 2);
            allocator.Context.MakeCurrent();
            IntPtr stream = allocator.Stream.Handle;
            IntPtr wF16;
            if (ggmlType == 1)
            {
                // F16 weights are already the cuBLAS operand in the same
                // [outDim, inDim] row-major layout the dequant kernels emit --
                // use the resident weight directly, no scratch or dequant pass.
                wF16 = weightPtr;
            }
            else
            {
                wF16 = EnsureScratch(WF16Scratch, allocator, wElems * 2);
                if (ggmlType == 8 && Q80F16DequantEnabled)
                    kernels.LaunchDequantWeightQ80F16(weightPtr, wF16, wElems, stream);
                else
                    kernels.LaunchDequantWeightF16(weightPtr, wF16, ggmlType, inDim, wElems, stream);
            }
            kernels.LaunchConvertF32F16(inputPtr, aF16, aElems, stream);

            // C[rows, outDim] (row-major) == C_col[outDim, rows]:
            //   C_col = (W_col[inDim, outDim])^T x A_col[inDim, rows]
            // f16 inputs, f32 accumulate + f32 output (CUBLAS_COMPUTE_32F).
            allocator.Blas.SetStream(stream);
            float alpha = 1.0f, beta = 0.0f;
            CublasApi.cublasGemmEx(
                allocator.Blas.Handle,
                CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                outDim, rows, inDim,
                ref alpha,
                wF16, CublasApi.CUDA_R_16F, inDim,
                aF16, CublasApi.CUDA_R_16F, inDim,
                ref beta,
                resultPtr, CublasApi.CUDA_R_32F, outDim,
                CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
        }

        // Q4_0 (the dominant dense quant) uses the int8 dp4a matmul for decode AND
        // verify by default (~memory-bound, matches ggml's mul_mat_vec_q); set
        // TS_CUDA_Q40_DP4A=0 to revert to the FP32 dequant kernels. Settable so the
        // exact-reference tests can pin the FP32 path while a separate test checks
        // the dp4a path against the (looser) int8 tolerance.
        public static bool Q40Dp4aEnabled { get; set; } =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_Q40_DP4A"), "0", StringComparison.Ordinal);

        // Q8_0 single-row decode uses the warp-per-column dp4a matvec by default
        // (quantizes the activation row to q8_1, like ggml's mul_mat_vec_q; measured
        // 26 -> 33 tok/s on Qwen3.5-9B-Q8_0 decode vs the scalar per-byte kernel).
        // TS_CUDA_Q80_VEC=0 reverts to the exact FP32 dequant kernel. Settable for
        // the same exact-vs-int8-tolerance test split as Q40Dp4aEnabled.
        public static bool Q80VecDp4aEnabled { get; set; } =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_Q80_VEC"), "0", StringComparison.Ordinal);

        // Quantize each 32-value activation block cooperatively with one warp.
        // This mirrors ggml_cuda's q8_1 quantizer and replaces the legacy
        // one-thread-per-block implementation, whose 32 serial loads/stores are
        // poorly coalesced for decode-sized rows. Set TS_CUDA_Q81_WARP=0 to keep
        // the legacy kernel for A/B testing.
        public static bool Q81WarpQuantizeEnabled { get; set; } =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_Q81_WARP"), "0", StringComparison.Ordinal);

        // Diagnostic/tuning gate: only use the q8_1 vec matvec when the output is at
        // least this wide (0 = always). Lets A/B runs isolate which decode
        // projections tolerate the activation's q8_1 round-trip.
        internal static readonly int Q80VecMinOutDim = EnvInt("TS_CUDA_Q80_VEC_MIN_OUT", 0);

        // Direct int8 tensor-core GEMM over raw Q8_0 blocks for prefill-sized rows
        // (mma.m16n8k32, ggml MMQ-style). TS_CUDA_Q80_MMQ=0 falls back to the
        // dequant+cuBLAS F16 route; MAX_ROWS is the crossover where cuBLAS wins
        // (weight sweeps grow as ceil(rows/128) on the MMQ side).
        public static bool Q80MmqEnabled { get; set; } =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_Q80_MMQ"), "0", StringComparison.Ordinal);
        internal static readonly int Q80MmqMaxRows = EnvInt("TS_CUDA_Q80_MMQ_MAX_ROWS", 512);

        // cp.async staging variant of the MMQ kernel (split q8_1 scratch, raw
        // weight windows async-copied to shared). Requires inDim % 256 == 0
        // (true for every real model dim); bit-identical results to the base
        // MMQ kernel. TS_CUDA_Q80_MMQ2=0 pins the register-prefetch variant.
        public static bool Q80Mmq2Enabled { get; set; } =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_Q80_MMQ2"), "0", StringComparison.Ordinal);

        public static bool SupportsQuantizedType(int ggmlType)
        {
            return ggmlType == 2 ||  // Q4_0
                ggmlType == 3 ||     // Q4_1
                ggmlType == 6 ||     // Q5_0
                ggmlType == 7 ||     // Q5_1
                ggmlType == 8 ||     // Q8_0
                ggmlType == 9 ||     // Q8_1
                ggmlType == 10 ||    // Q2_K
                ggmlType == 11 ||    // Q3_K
                ggmlType == 12 ||    // Q4_K
                ggmlType == 13 ||    // Q5_K
                ggmlType == 14 ||    // Q6_K
                ggmlType == 16 ||    // IQ2_XXS
                ggmlType == 18 ||    // IQ3_XXS
                ggmlType == 21 ||    // IQ3_S
                ggmlType == 22 ||    // IQ2_S
                ggmlType == 23;      // IQ4_XS
        }

        public static void PreloadQuantizedWeight(
            CudaAllocator allocator,
            IntPtr cacheKey,
            IntPtr hostData,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes)
        {
            if (allocator == null)
                throw new ArgumentNullException(nameof(allocator));
            EnsureWeight(allocator, cacheKey, hostData, ggmlType, ne0, ne1, rawBytes);
        }

        public static void ReleaseQuantizedWeight(CudaAllocator allocator, IntPtr cacheKey)
        {
            if (allocator == null || cacheKey == IntPtr.Zero)
                return;

            var key = new CacheKey(allocator.DeviceId, cacheKey);
            lock (Sync)
            {
                if (Cache.TryGetValue(key, out DeviceWeight entry))
                {
                    allocator.Context.MakeCurrent();
                    entry.Dispose();
                    Cache.Remove(key);
                }
            }
        }

        public static void ClearDeviceCache(CudaAllocator allocator)
        {
            if (allocator == null)
                return;

            lock (Sync)
            {
                allocator.Context.MakeCurrent();
                var remove = new List<CacheKey>();
                foreach (var kv in Cache)
                {
                    if (kv.Key.DeviceId == allocator.DeviceId)
                    {
                        kv.Value.Dispose();
                        remove.Add(kv.Key);
                    }
                }

                foreach (CacheKey key in remove)
                    Cache.Remove(key);
            }
        }

        // q8Kernel override for the multi-row Q8_0 path (lets a test drive MMA and dp4a
        // in one process): 0 = auto (env flags), 1 = dp4a, 2 = tensor-core MMA, 3 = scalar.
        public static bool TryAddmmQuantizedToFloat32(
            Tensor result,
            Tensor input,
            IntPtr cacheKey,
            IntPtr hostData,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes,
            int q8Kernel = 0)
        {
            // ggmlType 1 (F16) is not a block-quantized type but is accepted
            // here: the resident weight bytes already ARE the f16 cuBLAS
            // operand (see RunF16Gemm). Without this, an F16 tensor inside a
            // quantized GGUF (e.g. Gemma E4B's 55 MB per_layer_model_proj)
            // fell to the managed CPU matmul on EVERY prefill chunk and decode
            // token, with DtoH/HtoD mirror churn on top.
            if (ggmlType != 1 && !SupportsQuantizedType(ggmlType))
                return false;

            if (!CudaKernelOps.TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int resultCount) ||
                !CudaKernelOps.TryGetContiguousFloat(input, out CudaStorage inputStorage, out IntPtr inputPtr, out int inputCount) ||
                input.DimensionCount != 2 ||
                result.DimensionCount != 2 ||
                input.Sizes[1] != ne0 ||
                result.Sizes[0] != input.Sizes[0] ||
                result.Sizes[1] != ne1 ||
                inputCount != input.Sizes[0] * ne0 ||
                resultCount != result.Sizes[0] * ne1)
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            CudaKernels kernels = allocator.Kernels;
            if (kernels == null)
                return false;

            DeviceWeight weight = EnsureWeight(allocator, cacheKey, hostData, ggmlType, ne0, ne1, rawBytes);
            inputStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            int inDim = checked((int)ne0);
            int outDim = checked((int)ne1);
            int rows = checked((int)input.Sizes[0]);

            // F16 weights: the resident bytes are already the GEMM operand, so
            // every row count (a rows==1 decode is just a GEMV) goes straight
            // to cuBLAS with only the activation's f32->f16 convert on top.
            if (ggmlType == 1)
            {
                RunF16Gemm(allocator, kernels, weight.DevicePtr, ggmlType, inputPtr, resultPtr, inDim, outDim, rows);
                resultStorage.MarkDeviceModified();
                return true;
            }

            // Q8_0 prefill-sized batches: direct int8 tensor-core GEMM over the raw
            // Q8_0 blocks (mma.m16n8k32, ggml MMQ-style). Weight DRAM traffic is
            // ceil(rows/128) sweeps with NO f16 dequant round trip, so it beats the
            // dequant+cuBLAS route up to ~TS_CUDA_Q80_MMQ_MAX_ROWS rows; above that
            // cuBLAS' larger tiles win and the F16 route below takes over.
            if (Q80MmqEnabled && ggmlType == 8 && q8Kernel == 0
                && rows >= F16GemmMinRows && rows <= Q80MmqMaxRows && (inDim & 31) == 0)
            {
                if (Q80Mmq2Enabled && (inDim & 255) == 0)
                {
                    // cp.async variant: split activation scratch (dense qs rows +
                    // float scales), raw weight windows async-staged in shared.
                    IntPtr qsScratch = EnsureScratch(Q81SplitQsScratch, allocator, (long)rows * inDim);
                    IntPtr dScratch = EnsureScratch(Q81SplitDScratch, allocator, (long)rows * (inDim / 32) * sizeof(float));
                    kernels.LaunchQuantizeQ81SplitRows(inputPtr, qsScratch, dScratch, inDim, rows, allocator.Stream.Handle);
                    kernels.LaunchQuantMatmulQ80Mmq2(
                        weight.DevicePtr, qsScratch, dScratch, resultPtr, inDim, outDim, rows, allocator.Stream.Handle);
                }
                else
                {
                    long mmqScratchBytes = (long)rows * (inDim / 32) * CudaKernels.Q81BlockBytes;
                    IntPtr mmqXq = EnsureQ81Scratch(allocator, mmqScratchBytes);
                    kernels.LaunchQuantizeQ81Rows(
                        inputPtr, mmqXq, inDim, rows, allocator.Stream.Handle, Q81WarpQuantizeEnabled);
                    kernels.LaunchQuantMatmulQ80Mmq(
                        weight.DevicePtr, mmqXq, resultPtr, inDim, outDim, rows, allocator.Stream.Handle);
                }
                resultStorage.MarkDeviceModified();
                return true;
            }

            // Prefill-sized batches: dequant-once + tensor-core cuBLAS GEMM (see
            // F16GemmEnabled). Applies to every supported quant type; q8Kernel != 0
            // means a test pinned a specific Q8_0 kernel, so honor that instead.
            if (F16GemmEnabled && q8Kernel == 0 && rows >= F16GemmMinRows
                && 2L * inDim * outDim <= F16GemmMaxWeightBytes)
            {
                RunF16Gemm(allocator, kernels, weight.DevicePtr, ggmlType, inputPtr, resultPtr, inDim, outDim, rows);
                resultStorage.MarkDeviceModified();
                return true;
            }
            // Small multi-row batches (speculative MTP verify windows, short
            // prefill chunks) run the per-row quant kernels once per output row --
            // the whole weight matrix is re-read (and re-dequantized) per row, so a
            // B-row matmul costs ~B x a single decode and speculative verification
            // can never amortize. The generic scalar batched kernel tiles
            // TS_QMM_ROW_TILE rows per block, streaming each weight ONCE and reusing
            // it across the tile (measured 2.7-2.9x at B>=4 for the k-quant
            // ffn_down / LM-head weights).
            //   * Q4_0 (the dominant dense quant -- the 12B QAT model is entirely
            //     Q4_0) has its OWN row-tiled kernel below that keeps the efficient
            //     per-row nibble unpacking AND amortizes the weight read + dequant
            //     across the tile; the generic scalar batched path is ~2x slower for
            //     it (warp-per-column under-fills vs the block-per-4-columns design,
            //     and qvalue_at re-branches per element).
            //   * IQ2_XXS / Q8_0 keep their own multi-row paths (the q8_1 dp4a/MMA
            //     GEMM and the IQ2_XXS q8_1 kernel) which already read the weight
            //     once per tile and beat the scalar batched kernel for those types.
            // The batched kernel tiles rows down grid.y (any row count), so it also
            // covers LARGE prefill batches: the generic quant types here (K-quants,
            // IQ2_S / IQ3_S / IQ3_XXS / IQ4_XS) have no specialized large-batch GEMM,
            // and the per-row scalar fallback re-dequantizes the whole weight for
            // EVERY output row -- a 2048-token prefill then costs ~2048x the (memory-
            // bound) weight traffic and dominates TTFT (and the startup warmup). Route
            // all batch sizes (not just <=QuantMatmulBatchMaxRows) through the tiled
            // kernel so weight traffic drops ~TS_QMM_ROW_TILE x; it is numerically
            // identical (verified by the K-quant / IQ4_XS matmul tests).
            //
            // rows == 1 (single-token decode) deliberately stays OUT of this path.
            // Mixed "unsloth dynamic" quants (e.g. UD-IQ2_XXS) keep a large share
            // of tensors at Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ3_XXS/IQ3_S/IQ2_S/IQ4_XS,
            // and at rows==1 this kernel's one-warp-per-column design (32 threads
            // striding in_dim, TS_QMM_ROW_TILE amortization providing zero benefit
            // with only one row) has LESS per-column parallelism than the scalar
            // path's cols_per_block=4/blockDim=256 split -- measured on
            // Qwen3.6-27B-UD-IQ2_XXS: routing rows==1 here made decode SLOWER
            // (5.18 -> 4.54 tok/s) despite the scalar kernel's 76% GPU-time share,
            // because the wide lm_head/large-in_dim tensors dominate that share
            // and regressed further under 32-thread columns. See
            // ts_quant_matmul_vec_f32 below for the actual decode fix: one BLOCK
            // (not one warp) per output column, matching the scalar path's
            // per-column thread budget without its 4-columns-per-block
            // serialization.
            if (BatchedMatmulEnabled && rows >= 2
                && ggmlType != 2 && ggmlType != 8 && ggmlType != 16)
            {
                kernels.LaunchQuantMatmulBatchedF32(
                    weight.DevicePtr, inputPtr, resultPtr,
                    ggmlType, inDim, outDim, rows, allocator.Stream.Handle);
                resultStorage.MarkDeviceModified();
                return true;
            }
            // Single-token decode for the same generic quant-type set: one BLOCK
            // per output column (full blockDim.x threads via block_reduce_sum),
            // not the row-batched kernel's one-WARP-per-column (32 threads). At
            // rows==1 there is no cross-row weight reuse to amortize, so the
            // batched kernel's only advantage disappears and its 8x-narrower
            // per-column thread count directly costs latency on wide tensors
            // (e.g. a K-quant lm_head/large-in_dim projection) -- TS_CUDA_QMM_VEC=0
            // falls back to the scalar per-4-columns kernel for A/B comparison.
            if (VecMatmulEnabled && rows == 1
                && ggmlType != 2 && ggmlType != 8 && ggmlType != 16)
            {
                kernels.LaunchQuantMatmulVecF32(
                    weight.DevicePtr, inputPtr, resultPtr,
                    ggmlType, inDim, outDim, allocator.Stream.Handle);
                resultStorage.MarkDeviceModified();
                return true;
            }
            if (ggmlType == 2)
            {
                // Q4_0 int8 dp4a: quantize the activation rows to q8_1 ONCE, then the
                // block-tile dp4a GEMM. This is the fast path for BOTH single-token
                // decode (rows==1 — the dominant cost, and where the scalar FP32
                // dequant kernel read the LM head's Q4_0 weights at ~26 GB/s) and the
                // verify window. dp4a does 4 int8 MACs/instruction so it is ~memory-
                // bound, matching ggml's mul_mat_vec_q. inDim is always a multiple of
                // 32 for these models; non-conforming shapes fall through to FP32.
                if (Q40Dp4aEnabled && (inDim & 31) == 0)
                {
                    long scratchBytes = (long)rows * (inDim / 32) * CudaKernels.Q81BlockBytes;
                    IntPtr xqScratch = EnsureQ81Scratch(allocator, scratchBytes);
                    kernels.LaunchQuantizeQ81Rows(
                        inputPtr, xqScratch, inDim, rows, allocator.Stream.Handle, Q81WarpQuantizeEnabled);
                    kernels.LaunchQuantMatmulQ40Dp4a(
                        weight.DevicePtr, xqScratch, resultPtr, inDim, outDim, rows, allocator.Stream.Handle);
                    resultStorage.MarkDeviceModified();
                    return true;
                }
                // FP32 fallback: row-tiled batched kernel for the verify window
                // (decode each weight nibble ONCE and reuse it across the tile's rows)
                // and the per-row kernel for single-row decode.
                if (BatchedMatmulEnabled && rows >= 2 && rows <= CudaKernels.QuantMatmulBatchMaxRows)
                {
                    kernels.LaunchQuantMatmulQ40BatchedF32(
                        weight.DevicePtr, inputPtr, resultPtr,
                        inDim, outDim, rows, allocator.Stream.Handle);
                    resultStorage.MarkDeviceModified();
                    return true;
                }
                kernels.LaunchQuantMatmulQ40F32(
                    weight.DevicePtr,
                    inputPtr,
                    resultPtr,
                    inDim,
                    outDim,
                    rows,
                    allocator.Stream.Handle);
            }
            else if (ggmlType == 8 && (inDim & 31) == 0
                     && q8Kernel != 3
                     && (rows >= 2 || (Q80VecDp4aEnabled && outDim >= Q80VecMinOutDim))
                     && (q8Kernel == 1 || q8Kernel == 2
                         || (q8Kernel == 0 && (CudaKernels.Q8MmaEnabled || CudaKernels.Q8Dp4aEnabled))))
            {
                // Q8_0 int8 fast path (rows >= 1): quantize the activation rows to q8_1
                // ONCE into a reused scratch (single-stream per allocator makes reuse
                // safe), then either the tensor-core MMA GEMM or the block-tile dp4a
                // GEMM. dp4a covers single-token decode too (4 int8 MACs/instruction,
                // ~memory-bound like ggml's mul_mat_vec_q) - the scalar per-byte
                // dequant kernel it replaces left ~25% of weight bandwidth unused,
                // and decode reads every weight once per token.
                bool useMma = q8Kernel == 2 || (q8Kernel == 0 && CudaKernels.Q8MmaEnabled && rows >= 2);
                long scratchBytes = (long)rows * (inDim / 32) * CudaKernels.Q81BlockBytes;
                IntPtr xqScratch = EnsureQ81Scratch(allocator, scratchBytes);
                kernels.LaunchQuantizeQ81Rows(
                    inputPtr, xqScratch, inDim, rows, allocator.Stream.Handle, Q81WarpQuantizeEnabled);
                if (useMma)
                    kernels.LaunchQuantMatmulQ80Mma(
                        weight.DevicePtr, xqScratch, resultPtr, inDim, outDim, rows, allocator.Stream.Handle);
                else if (rows == 1)
                    kernels.LaunchQuantMatmulQ80Vec(
                        weight.DevicePtr, xqScratch, resultPtr, inDim, outDim, allocator.Stream.Handle);
                else
                    kernels.LaunchQuantMatmulQ80Dp4a(
                        weight.DevicePtr, xqScratch, resultPtr, inDim, outDim, rows, allocator.Stream.Handle);
            }
            else if (ggmlType == 8)
            {
                kernels.LaunchQuantMatmulQ80F32(
                    weight.DevicePtr,
                    inputPtr,
                    resultPtr,
                    inDim,
                    outDim,
                    rows,
                    allocator.Stream.Handle);
            }
            else if (ggmlType == 16 && (inDim & 255) == 0 && ((inDim / 32) * 36) <= 48 * 1024)
            {
                kernels.LaunchQuantMatmulIq2XxsQ81F32(
                    weight.DevicePtr,
                    inputPtr,
                    resultPtr,
                    inDim,
                    outDim,
                    rows,
                    allocator.Stream.Handle);
            }
            else
            {
                kernels.LaunchQuantMatmulF32(
                    weight.DevicePtr,
                    inputPtr,
                    resultPtr,
                    ggmlType,
                    inDim,
                    outDim,
                    rows,
                    allocator.Stream.Handle);
            }

            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryGetRowsQuantizedToFloat32(
            Tensor result,
            IntPtr cacheKey,
            IntPtr hostData,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes,
            Tensor indices)
        {
            if (!SupportsQuantizedType(ggmlType))
                return false;

            if (!CudaKernelOps.TryGetContiguousRows(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int rows, out int cols) ||
                !CudaKernelOps.TryGetContiguous(indices, out CudaStorage indicesStorage, out IntPtr indicesPtr, out long indexCount) ||
                cols != ne0 ||
                indexCount != rows ||
                result.ElementType != DType.Float32 ||
                (indices.ElementType != DType.Int32 && indices.ElementType != DType.Float32))
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            CudaKernels kernels = allocator.Kernels;
            if (kernels == null)
                return false;

            DeviceWeight weight = EnsureWeight(allocator, cacheKey, hostData, ggmlType, ne0, ne1, rawBytes);
            indicesStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchQuantGetRowsF32(
                weight.DevicePtr,
                indicesPtr,
                resultPtr,
                ggmlType,
                checked((int)ne0),
                rows,
                indices.ElementType == DType.Int32 ? 1 : 0,
                allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        private static DeviceWeight EnsureWeight(
            CudaAllocator allocator,
            IntPtr cacheKey,
            IntPtr hostData,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes)
        {
            if (cacheKey == IntPtr.Zero)
                cacheKey = hostData;
            if (cacheKey == IntPtr.Zero)
                throw new ArgumentException("CUDA quantized weight cache key cannot be zero.", nameof(cacheKey));

            var key = new CacheKey(allocator.DeviceId, cacheKey);
            lock (Sync)
            {
                if (Cache.TryGetValue(key, out DeviceWeight existing))
                    return existing;

                if (hostData == IntPtr.Zero)
                    throw new InvalidOperationException("Quantized weight is not preloaded on this CUDA device and no host data was provided.");

                allocator.Context.MakeCurrent();
                // +16 slack bytes: the cp.async MMQ kernel stages weight windows as
                // 16-byte-aligned chunks whose final chunk may reach up to 8 bytes
                // past the last block of the last row (the bytes are never read).
                CudaDriverApi.cuMemAlloc(out IntPtr devicePtr, new UIntPtr((ulong)(rawBytes + 16))).ThrowOnError();
                try
                {
                    CudaDriverApi.cuMemcpyHtoD(devicePtr, hostData, new UIntPtr((ulong)rawBytes)).ThrowOnError();
                    var entry = new DeviceWeight
                    {
                        DevicePtr = devicePtr,
                        RawBytes = rawBytes,
                        GgmlType = ggmlType,
                        Ne0 = ne0,
                        Ne1 = ne1,
                        DeviceId = allocator.DeviceId,
                    };
                    Cache.Add(key, entry);
                    return entry;
                }
                catch
                {
                    CudaDriverApi.cuMemFree(devicePtr);
                    throw;
                }
            }
        }

        private readonly struct CacheKey : IEquatable<CacheKey>
        {
            public CacheKey(int deviceId, IntPtr key)
            {
                DeviceId = deviceId;
                Key = key;
            }

            public int DeviceId { get; }
            public IntPtr Key { get; }

            public bool Equals(CacheKey other)
            {
                return DeviceId == other.DeviceId && Key == other.Key;
            }

            public override bool Equals(object obj)
            {
                return obj is CacheKey other && Equals(other);
            }

            public override int GetHashCode()
            {
                return HashCode.Combine(DeviceId, Key);
            }
        }
    }
}
