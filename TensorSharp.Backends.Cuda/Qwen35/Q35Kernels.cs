// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Launchers for the Qwen3.5 / Qwen3.8 dense direct-CUDA kernels (native/kernels/
// tensorsharp_q35_kernels.cu, compiled to tensorsharp_q35_kernels.ptx and its sm_120a fatbin).
// One instance per device, loaded against the context current at construction.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal sealed unsafe partial class Q35Kernels : IDisposable
    {
        /// <summary>Tokens per attention cache block (TS_Q35_KV_BLOCK).</summary>
        public const int KvBlockTokens = 256;
        /// <summary>Attention head width the kernels are written for (TS_Q35_HD).</summary>
        public const int AttnHeadDim = 256;
        /// <summary>Gated DeltaNet head width (TS_Q35_GDN_HD).</summary>
        public const int GdnHeadDim = 128;
        /// <summary>Rotated dims per attention head (TS_Q35_ROPE_DIMS).</summary>
        public const int RopeDims = 64;
        /// <summary>Bytes of one 256-value q8_0 cache row.</summary>
        public const int Q8Row = 272;
        /// <summary>Activation rows the q8_1 GEMV takes; more run on the FP4 GEMM.</summary>
        public const int GemvMaxRows = 8;
        /// <summary>Weight rows per GEMM tile (TS_Q35_BM) and K values per stage (TS_Q35_BK).</summary>
        public const int GemmTileRows = 128, GemmStageK = 256;
        /// <summary>Query rows one attention block takes (4 warps x 16).</summary>
        public const int AttnBlockRows = 64;

        private const int Block = 256;
        private readonly CudaModule module;
        private readonly IntPtr repack, quantQ81, quantNvfp4, quantSwiGluNvfp4, normNvfp4, embed, addNorm, addNorm5120, attnPrep, gateMul, swiglu;
        private readonly IntPtr gdnConv, gdnConvState, gdnPrep, gdnScan, gdnScanVerify, gdnReplay, gdnChunk, gdnOut, copyRows, attn, attnSplit, attnCombine, argmax;
        private readonly IntPtr gdnPrepareQk, gdnPrepareControl, gdnChunkPrepared;
        private static readonly bool PreparedGdnPrefill = Environment.GetEnvironmentVariable("TS_Q35_GDN_PREPARED_PREFILL") == "1";
        internal static bool GdnPreparedPrefillEnabled => PreparedGdnPrefill;
        internal long PreparedGdnLaunches { get; private set; }
        private readonly IntPtr attnPrefill;
        private readonly int smemAttnPrefill;
        /// <summary>Prompt rows take ts_q35_attn_prefill_f32 (pipelined, short F16 accumulation runs; about 1.7x as fast)
        /// unless TS_Q35_ATTN_PREFILL=0 keeps them on ts_q35_attn_q8_f32.</summary>
        private static readonly bool PrefillAttention = Environment.GetEnvironmentVariable("TS_Q35_ATTN_PREFILL") != "0";
        // TS_Q35_GEMM_SPLITK=0 keeps every GEMM to one K pass (a summation order independent of the row count).
        private static readonly bool SplitK = Environment.GetEnvironmentVariable("TS_Q35_GEMM_SPLITK") != "0";
        // Diagnostic control for strict repeated-logit comparisons. Keep the production
        // reduction policy unless explicitly disabled; atomic draft split-K is not bit stable.
        private static readonly bool DraftSplitK = Environment.GetEnvironmentVariable("TS_Q35_DRAFT_SPLITK") != "0";
        private readonly IntPtr[] gemv = new IntPtr[GemvMaxRows + 1];
        private readonly IntPtr[] gemvQ6k = new IntPtr[2];
        private readonly IntPtr gemmQ6k16, gemmQ6k32, gemmQ6k64;
        private readonly IntPtr[] mmaQ4k = new IntPtr[4], mmaQ6k = new IntPtr[4];   // 8, 16, 32, 64 rows
        private readonly IntPtr rowsByIndex, dfPrep, dfAttn, dfCombine, dfConv, dfGather, topk, dfSelect;
        private readonly IntPtr topkPartial, topkMerge;
        internal const int TopKPartitions = 32;
        private static readonly bool PartitionedTopK = Environment.GetEnvironmentVariable("TS_Q35_TOPK_PARTITIONED") == "1";
        /// <summary>Activation rows one int8 tensor-core launch (Q4_K / Q6_K) takes.</summary>
        public const int QuantMmaRows = 64;
        /// <summary>Draft head width (TS_Q35_DF_HD).</summary>
        public const int DraftHeadDim = 128;
        /// <summary>Activation rows one Q6_K launch takes (the LM head over a verify window).</summary>
        public const int Q6kMaxRows = 64;
        private readonly int[] gemvRows = new int[GemvMaxRows + 1];
        private readonly IntPtr gemm32, gemm64, gemm128, gemm32SwiGlu, gemm64SwiGlu, gemm128SwiGlu;
        private readonly int smem32, smem64, smem128;

        private readonly int sms;

        private Q35Kernels(CudaModule module)
        {
            this.module = module;
            if (!RunsTensorCoreKernels(module))
                throw new NotSupportedException(NoTensorCoreCode);
            CudaDriverApi.cuDeviceGetAttribute(out sms, 16 /* CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT */, 0).ThrowOnError();
            repack = module.GetFunction("ts_q35_repack_nvfp4");
            quantQ81 = module.GetFunction("ts_q35_quantize_q81_f32");
            quantNvfp4 = module.GetFunction("ts_q35_quantize_nvfp4_f32");
            quantSwiGluNvfp4 = module.GetFunction("ts_q35_quantize_swiglu_nvfp4_f32");
            normNvfp4 = module.GetFunction("ts_q35_add_rmsnorm_nvfp4_f32");
            embed = module.GetFunction("ts_q35_embed_q4_0");
            addNorm = module.GetFunction("ts_q35_add_rmsnorm_f32");
            addNorm5120 = module.GetFunction("ts_q35_add_rmsnorm_5120_f32");
            attnPrep = module.GetFunction("ts_q35_attn_prep_f32");
            gateMul = module.GetFunction("ts_q35_gate_mul_f32");
            swiglu = module.GetFunction("ts_q35_swiglu_f32");
            gdnConv = module.GetFunction("ts_q35_gdn_conv_f32");
            gdnConvState = module.GetFunction("ts_q35_gdn_conv_state_f32");
            gdnPrep = module.GetFunction("ts_q35_gdn_prep_f32");
            gdnScan = module.GetFunction("ts_q35_gdn_scan_f32");
            gdnChunk = module.GetFunction("ts_q35_gdn_chunk_f32");
            if (PreparedGdnPrefill)
            {
                gdnPrepareQk = module.GetFunction("ts_q35_gdn_prepare_qk_f32");
                gdnPrepareControl = module.GetFunction("ts_q35_gdn_prepare_control_f32");
                gdnChunkPrepared = module.GetFunction("ts_q35_gdn_chunk_prepared_f32");
            }
            gdnScanVerify = module.GetFunction("ts_q35_gdn_scan_verify_f32");
            gdnReplay = module.GetFunction("ts_q35_gdn_replay_f32");
            copyRows = module.GetFunction("ts_q35_copy_rows_f32");
            gdnOut = module.GetFunction("ts_q35_gdn_out_f32");
            attn = module.GetFunction("ts_q35_attn_q8_f32");
            attnPrefill = module.GetFunction("ts_q35_attn_prefill_f32");
            attnSplit = module.GetFunction("ts_q35_attn_q8_split_f32");
            attnCombine = module.GetFunction("ts_q35_attn_combine_f32");
            argmax = module.GetFunction("ts_q35_argmax_f32");
            for (int n = 1; n <= GemvMaxRows; n++)
                gemv[n] = module.GetFunction($"ts_q35_gemv_nvfp4_{n}");
            gemvQ6k[1] = module.GetFunction("ts_q35_gemv_q6k_1");
            for (int i = 0; i < 4; i++)
            {
                mmaQ4k[i] = module.GetFunction($"ts_q35_gemm_q4k_{1 << i}");
                mmaQ6k[i] = module.GetFunction($"ts_q35_gemm_q6k_{1 << i}");
            }
            rowsByIndex = module.GetFunction("ts_q35_rows_by_index_f32");
            dfPrep = module.GetFunction("ts_q35_df_prep_f32");
            dfAttn = module.GetFunction("ts_q35_df_attn_f16");
            dfCombine = module.GetFunction("ts_q35_df_combine_f32");
            dfConv = module.GetFunction("ts_q35_df_conv_f32");
            dfGather = module.GetFunction("ts_q35_df_gather_f32");
            topk = module.GetFunction("ts_q35_topk_f32");
            topkPartial = module.GetFunction("ts_q35_topk_partial_f32");
            topkMerge = module.GetFunction("ts_q35_topk_merge_f32");
            dfSelect = module.GetFunction("ts_q35_df_select");
            gemmQ6k16 = module.GetFunction("ts_q35_gemm_q6k_2");
            gemmQ6k32 = module.GetFunction("ts_q35_gemm_q6k_4");
            gemmQ6k64 = module.GetFunction("ts_q35_gemm_q6k_8");
            gemm32 = module.GetFunction("ts_q35_gemm_w4a4_32");
            gemm64 = module.GetFunction("ts_q35_gemm_w4a4_64");
            gemm128 = module.GetFunction("ts_q35_gemm_w4a4_128");
            gemm32SwiGlu = module.GetFunction("ts_q35_gemm_w4a4_32_swiglu");
            gemm64SwiGlu = module.GetFunction("ts_q35_gemm_w4a4_64_swiglu");
            gemm128SwiGlu = module.GetFunction("ts_q35_gemm_w4a4_128_swiglu");

            // Rows per warp of each GEMV and the GEMMs' dynamic shared memory, from the module.
            CudaDriverApi.cuMemAlloc(out IntPtr probe, new UIntPtr(64 * 4)).ThrowOnError();
            try
            {
                IntPtr p = probe;
                void** args = stackalloc void*[] { &p };
                Launch(module.GetFunction("ts_q35_gemv_rows_per_warp"), 1, 1, 1, 1, 0, IntPtr.Zero, args);
                int* host = stackalloc int[16];
                CudaDriverApi.cuMemcpyDtoH((IntPtr)host, probe, new UIntPtr(9 * 4)).ThrowOnError();
                for (int n = 1; n <= GemvMaxRows; n++)
                    gemvRows[n] = host[n];
                Launch(module.GetFunction("ts_q35_gemm_smem_bytes"), 1, 1, 1, 1, 0, IntPtr.Zero, args);
                CudaDriverApi.cuMemcpyDtoH((IntPtr)host, probe, new UIntPtr(3 * 4)).ThrowOnError();
                smem32 = host[0];
                smem64 = host[1];
                smem128 = host[2];
                Launch(module.GetFunction("ts_q35_attn_smem_bytes"), 1, 1, 1, 1, 0, IntPtr.Zero, args);
                CudaDriverApi.cuMemcpyDtoH((IntPtr)host, probe, new UIntPtr(4)).ThrowOnError();
                smemAttnPrefill = host[0];
            }
            finally
            {
                CudaDriverApi.cuMemFree(probe);
            }
            const int CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8;
            // The quantizers stage a whole row (up to the FFN width) in shared memory.
            foreach (IntPtr f in new[] { quantNvfp4, quantSwiGluNvfp4, normNvfp4 })
                CudaDriverApi.cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, MaxQuantRowFloats * 4).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(gemm32, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem32).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(gemm64, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem64).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(gemm128, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem128).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(gemm32SwiGlu, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem32).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(gemm64SwiGlu, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem64).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(gemm128SwiGlu, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem128).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(attnPrefill, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smemAttnPrefill).ThrowOnError();
        }

        public static Q35Kernels Create()
            => Create(identifyImage: false);

        internal static Q35Kernels Create(bool identifyImage)
        {
            string path = CudaKernels.LocatePtxPath("tensorsharp_q35_kernels.ptx");
            if (path == null)
            {
                throw new InvalidOperationException(
                    "tensorsharp_q35_kernels.ptx could not be located next to the application or under " +
                    "a 'cuda_kernels' folder. Build TensorSharp.Backends.Cuda with a CUDA 13 nvcc on the PATH.");
            }
            return FromModule(CudaModule.LoadFromFile(path, identify: identifyImage));
        }

        /// <summary>Kernels over an already loaded module, which they own (and unload when they
        /// refuse it). Requires the module's context to be current.</summary>
        internal static Q35Kernels FromModule(CudaModule module)
        {
            try
            {
                return new Q35Kernels(module);
            }
            catch
            {
                module.Dispose();
                throw;
            }
        }

        private const string NoTensorCoreCode =
            "the loaded tensorsharp_q35_kernels image has no FP4 tensor-core code for this GPU: every GEMM needs " +
            "sm_120a / sm_121a SASS, and PTX-only or other-architecture code traps there. Build " +
            "TensorSharp.Backends.Cuda with nvcc on this GPU with the fatbin enabled (CudaEmitFatbin, the default)";

        /// <summary>
        /// Whether the code the driver loaded for <paramref name="module"/> carries the tensor-core
        /// bodies the GEMMs need (ts_q35_caps): sm_120a / sm_121a SASS does. The plain compute_12x
        /// PTX, or sm_8x / sm_9x code JIT-compiled for a CC 12 device, has <c>__trap()</c> there,
        /// so the first prompt would kill the context instead of being refused. A launch rather
        /// than a symbol lookup, since a fatbin holds both kinds of image. A module built before
        /// the marker existed counts as not. Requires the module's context to be current.
        /// </summary>
        internal static bool RunsTensorCoreKernels(CudaModule module)
        {
            IntPtr caps;
            try { caps = module.GetFunction("ts_q35_caps"); }
            catch (CudaException ex) when (ex.ErrorCode == 500) { return false; } // CUDA_ERROR_NOT_FOUND
            CudaDriverApi.cuMemAlloc(out IntPtr probe, new UIntPtr(4)).ThrowOnError();
            try
            {
                IntPtr p = probe;
                void** args = stackalloc void*[] { &p };
                Launch(caps, 1, 1, 1, 1, 0, IntPtr.Zero, args);
                int value = 0;
                CudaDriverApi.cuMemcpyDtoH((IntPtr)(&value), probe, new UIntPtr(4)).ThrowOnError();
                return value == 1;
            }
            finally
            {
                CudaDriverApi.cuMemFree(probe);
            }
        }

        /// <summary>
        /// <see cref="RunsTensorCoreKernels"/> for the module at <paramref name="path"/> on device 0,
        /// before any engine exists (the support check), leaving this thread's current context as
        /// it was. <paramref name="reason"/> says why not.
        /// </summary>
        internal static bool ProbeTensorCoreKernels(string path, out string reason)
        {
            CudaDriverApi.cuCtxGetCurrent(out IntPtr previous);
            try
            {
                using var context = CudaContext.Create(0);
                using var module = CudaModule.LoadFromFile(path);
                reason = RunsTensorCoreKernels(module) ? null : NoTensorCoreCode;
                return reason == null;
            }
            finally
            {
                CudaDriverApi.cuCtxSetCurrent(previous);
            }
        }

        private static void Launch(IntPtr fn, uint gx, uint gy, uint gz, int block, int smem, IntPtr stream, void** args)
            => CudaDriverApi.cuLaunchKernel(fn, gx, gy, gz, (uint)block, 1, 1, (uint)smem, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();

        private static uint CeilDiv(long n, long d) => (uint)Math.Max(1, (n + d - 1) / d);

        /// <summary>ggml block_nvfp4 rows -> code plane [rows][k/2] and scale plane [rows][k/16].</summary>
        /// <summary>ggml block_nvfp4 rows -> the code and scale planes at their start; with
        /// <paramref name="parts"/> &gt; 1 the planes' base instead, rows rowBase.. of part
        /// <paramref name="part"/> going to groups of 8 interleaved with the other parts'.</summary>
        public void RepackNvfp4(IntPtr src, long srcRowBytes, IntPtr codes, IntPtr scales, int rows, int k, IntPtr stream,
            int rowBase = 0, int parts = 1, int part = 0)
        {
            IntPtr a0 = src, a2 = codes, a3 = scales; long a1 = srcRowBytes; int a4 = rows, a5 = k, a6 = rowBase, a7 = parts, a8 = part;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
            Launch(repack, CeilDiv((long)rows * (k / 64), Block), 1, 1, Block, 0, stream, args);
        }

        /// <summary>F32 rows -> int8 rows plus one F32 scale per 32 values (the GEMV's activations).</summary>
        public void QuantizeQ81(IntPtr x, IntPtr xq, IntPtr xd, int k, int rows, IntPtr stream)
        {
            IntPtr a0 = x, a1 = xq, a2 = xd; int a3 = k, a4 = rows;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(quantQ81, CeilDiv((long)rows * (k / 32) * 32, Block), 1, 1, Block, 0, stream, args);
        }

        /// <summary>F32 rows -> NVFP4 code/scale planes plus a per-row scale (the GEMM's activations).</summary>
        /// <summary>Widest row the NVFP4 quantizers take (they stage it in shared memory).</summary>
        public const int MaxQuantRowFloats = 20 * 1024;

        private static int QuantSmem(int k) => k <= MaxQuantRowFloats
            ? k * 4 : throw new NotSupportedException($"[q35-cuda] NVFP4 quantization of {k}-wide rows (at most {MaxQuantRowFloats})");

        public void QuantizeNvfp4(IntPtr x, IntPtr codes, IntPtr scales, IntPtr rowScale, int k, int rows, IntPtr stream)
        {
            IntPtr a0 = x, a1 = codes, a2 = scales, a3 = rowScale; int a4 = k;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(quantNvfp4, (uint)rows, 1, 1, 512, QuantSmem(k), stream, args);
        }

        /// <summary>x += delta (zero: none), then NVFP4 planes of rmsnorm(x) * w: the pre-norm and the
        /// next projection's activation quantization in one pass.</summary>
        public void AddRmsNormNvfp4(IntPtr x, IntPtr delta, IntPtr w, float eps, IntPtr codes, IntPtr scales, IntPtr rowScale,
            int k, int rows, IntPtr stream)
        {
            IntPtr a0 = x, a1 = delta, a2 = w, a4 = codes, a5 = scales, a6 = rowScale; float a3 = eps; int a7 = k;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            Launch(normNvfp4, (uint)rows, 1, 1, 512, QuantSmem(k), stream, args);
        }

        /// <summary>NVFP4 planes of silu(gate) * up for fused gate|up rows [rows][2k] (the down
        /// projection's input, never stored as F32).</summary>
        public void QuantizeSwiGluNvfp4(IntPtr gu, IntPtr codes, IntPtr scales, IntPtr rowScale, int k, int rows, IntPtr stream)
        {
            IntPtr a0 = gu, a1 = codes, a2 = scales, a3 = rowScale; int a4 = k;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(quantSwiGluNvfp4, (uint)rows, 1, 1, 512, QuantSmem(k), stream, args);
        }

        /// <summary>y[n, outDim] = alpha * xq[n] W^T for n &lt;= 8 rows of q8_1 activations.</summary>
        public void GemvNvfp4(IntPtr codes, IntPtr scales, int outDim, int k, IntPtr xq, IntPtr xd, float alpha, IntPtr y, int n, IntPtr stream)
        {
            IntPtr a0 = codes, a1 = scales, a4 = xq, a5 = xd, a7 = y; int a2 = outDim, a3 = k; float a6 = alpha;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            int perBlock = 4 * gemvRows[n];
            Launch(gemv[n], CeilDiv(outDim, perBlock), 1, 1, 128, 0, stream, args);
        }

        /// <summary>y[n, outDim] = xq[n] W^T for a Q6_K weight: the dp4a GEMV for one row, the int8
        /// tensor-core kernel for up to <see cref="Q6kMaxRows"/> per launch above that (the GEMV
        /// re-reads every activation row per weight row and falls far behind at 8).</summary>
        public void GemvQ6k(IntPtr w, long rowBytes, int outDim, int k, IntPtr xq, IntPtr xd, IntPtr y, int n, IntPtr stream)
        {
            if (n == 1)
            {
                IntPtr a0 = w, a4 = xq, a5 = xd, a6 = y; long a1 = rowBytes; int a2 = outDim, a3 = k;
                void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
                Launch(gemvQ6k[1], CeilDiv(outDim, 4), 1, 1, 128, 0, stream, args);
                return;
            }
            for (int r0 = 0; r0 < n; r0 += Q6kMaxRows)
            {
                int rows = Math.Min(Q6kMaxRows, n - r0);
                IntPtr fn = rows <= 8 ? mmaQ6k[0] : rows <= 16 ? gemmQ6k16 : rows <= 32 ? gemmQ6k32 : gemmQ6k64;
                // One split (gridDim.y = 1): the LM head has row tiles to spare.
                IntPtr a0 = w, a4 = xq + (nint)((long)r0 * k), a5 = xd + (nint)((long)r0 * (k / 32) * 4), a7 = y + (nint)((long)r0 * outDim * 4);
                long a1 = rowBytes; int a2 = outDim, a3 = k, a6 = rows;
                void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
                Launch(fn, CeilDiv(outDim, 64), 1, 1, 128, 0, stream, args);
            }
        }

        /// <summary>y[m, outDim] = alpha * rowScale * (x W^T) for m NVFP4 activation rows on the FP4
        /// tensor cores. <paramref name="partial"/> (splits * m * outDim floats) and
        /// <paramref name="counters"/> (zeroed ints, one per output tile) serve split-K.</summary>
        /// <param name="swiglu">wCodes are the interleaved gate|up planes: y [m][outDim / 2] is the FFN
        /// activation silu(gate) * up (never split over K).</param>
        public void GemmW4A4(IntPtr wCodes, IntPtr wScales, int outDim, int k, IntPtr xCodes, IntPtr xScales, IntPtr rowScale,
            int m, float alpha, IntPtr y, IntPtr partial, IntPtr counters, int sms, int maxSplits, IntPtr stream,
            bool swiglu = false, bool ninferVerify8 = false)
        {
            _gemmWorkSubmitted = true;
            if (swiglu || !SplitK)
                maxSplits = 1;
            int bn = m <= 32 ? 32 : m <= 64 ? 64 : 128;
            int tilesX = (outDim + GemmTileRows - 1) / GemmTileRows, tilesY = (m + bn - 1) / bn;
            int stages = k / GemmStageK, splits = 1;
            while (tilesX * tilesY * splits < sms && splits * 2 <= stages / 2 && splits * 2 <= maxSplits)
                splits *= 2;
            IntPtr a0 = wCodes, a1 = wScales, a4 = xCodes, a5 = xScales, a6 = rowScale, a9 = y, a10 = partial, a11 = counters;
            int a2 = outDim, a3 = k, a7 = m; float a8 = alpha;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11 };
            var (fn, smem) = bn == 32 ? (swiglu ? gemm32SwiGlu : gemm32, smem32) : bn == 64 ? (swiglu ? gemm64SwiGlu : gemm64, smem64)
                : (swiglu ? gemm128SwiGlu : gemm128, smem128);
            bool useVerify8 = UseNinferVerify8Gemm(_ninferVerify8Gemm != IntPtr.Zero, ninferVerify8, swiglu, m, outDim, k);
            if (useVerify8) { fn = _ninferVerify8Gemm; smem = NinferVerify8GemmSharedBytes; }
            // The 128-row kernel walks activation tiles fastest (weight tiles shared in L2).
            if (bn == 128)
                Launch(fn, (uint)tilesY, (uint)tilesX, (uint)splits, 256, smem, stream, args);
            else
                Launch(fn, (uint)tilesX, (uint)tilesY, (uint)splits, 256, smem, stream, args);
            if (useVerify8) NinferVerify8GemmHostSubmissions++;
        }

        public void EmbedQ4_0(IntPtr table, long rowBytes, IntPtr tokens, IntPtr outRows, int rows, int width, IntPtr stream)
        {
            IntPtr a0 = table, a2 = tokens, a3 = outRows; long a1 = rowBytes; int a4 = width;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(embed, (uint)rows, 1, 1, Block, 0, stream, args);
        }

        /// <summary>x += delta (zero: none), y = rmsnorm(x) * w, per row.</summary>
        public void AddRmsNorm(IntPtr x, IntPtr delta, IntPtr w, IntPtr y, int rows, int width, float eps, IntPtr stream)
        {
            IntPtr a0 = x, a1 = delta, a2 = w, a3 = y; int a4 = width; float a5 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            // Width 5120 keeps the row in registers: the same arithmetic, bit-identical output.
            Launch(width == 5120 ? addNorm5120 : addNorm, (uint)rows, 1, 1, 512, 0, stream, args);
        }

        /// <param name="ropePos">The M-RoPE components [4][ropeStride] (component c of row i at c * ropeStride + i).</param>
        public void AttnPrep(IntPtr qkv, int rows, int rowsPerSeq, int nHead, int nKv, IntPtr qNorm, IntPtr kNorm, float eps,
            IntPtr ropePos, int ropeStride, int sectT, int sectH, int sectW, float ropeBase, IntPtr cacheRow, IntPtr blockTable, int maxBlocks,
            long kOff, long vOff, IntPtr q, IntPtr gate, IntPtr stream)
        {
            IntPtr a0 = qkv, a5 = qNorm, a6 = kNorm, a8 = ropePos, a13 = cacheRow, a14 = blockTable, a17 = q, a18 = gate;
            int a1 = rows, a2 = rowsPerSeq, a3 = nHead, a4 = nKv, a8s = ropeStride, a9 = sectT, a10 = sectH, a11 = sectW, a15 = maxBlocks;
            float a7 = eps, a12 = ropeBase; long a16 = kOff, b16 = vOff;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a8s, &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &b16, &a17, &a18 };
            long warps = (long)rows * (nHead + 2 * nKv);
            Launch(attnPrep, CeilDiv(warps * 32, Block), 1, 1, Block, 0, stream, args);
        }

        /// <summary>Causal attention of every row of nSeq sequences over their q8_0 caches
        /// ([tokens][nKv][272 B]), gate applied. Sequences of at most <see cref="AttnBlockRows"/> query rows per KV head take the
        /// split-key path (<paramref name="partial"/> sized by <see cref="AttnSplitFloats"/>). Longer prompt chunks split
        /// their keys too when that fills the device better, into <paramref name="scratch"/> (<paramref name="scratchBytes"/>).</summary>
        public void Attention(IntPtr q, IntPtr gate, IntPtr keyEnd, IntPtr blockTable, int maxBlocks, long kOff, long vOff,
            int rowsPerSeq, int nSeq, int nHead, int nKv, float scale, int maxKeys, IntPtr partial, IntPtr outRows, int sms, IntPtr stream,
            IntPtr scratch = default, long scratchBytes = 0)
        {
            int rowsPerHead = rowsPerSeq * (nHead / nKv);
            if (rowsPerHead <= AttnBlockRows && partial != IntPtr.Zero)
            {
                // Key ranges on fixed boundaries (a chunk that only doubles past 64 of them), so a
                // row's partial sums never depend on how far the other rows in the step reach:
                // a verify then draws exactly what plain steps of the same width draw.
                var plan = GetSplitAttentionPlan(maxKeys);
                int chunk = plan.Chunk, splits = plan.Splits;
                IntPtr a0 = q, a1 = keyEnd, a2 = blockTable, a10 = partial;
                int a3 = maxBlocks, a5 = rowsPerSeq, a6 = nHead, a7 = nKv, a9 = chunk; float a8 = scale; long a4 = kOff, a4v = vOff;
                void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a4v, &a5, &a6, &a7, &a8, &a9, &a10 };
                Launch(attnSplit, (uint)splits, (uint)nKv, (uint)nSeq, 128, 0, stream, args);
                IntPtr c0 = partial, c1 = gate, c3 = outRows; int c2 = splits;
                void** cargs = stackalloc void*[] { &c0, &c1, &c2, &c3 };
                Launch(attnCombine, (uint)(nSeq * rowsPerSeq * nHead), 1, 1, AttnHeadDim, 0, stream, cargs);
                return;
            }
            IntPtr b0 = q, b1 = gate, b2 = keyEnd, b3 = blockTable, b10 = outRows;
            int b4 = maxBlocks, b6 = rowsPerSeq, b7 = nHead, b8 = nKv; float b9 = scale; long b5 = kOff, c5 = vOff;
            // Both kernels take 128 rows (8 warps) per block.
            uint tiles = CeilDiv(rowsPerHead, 2 * AttnBlockRows);
            if (!PrefillAttention)
            {
                void** bargs = stackalloc void*[] { &b0, &b1, &b2, &b3, &b4, &b5, &c5, &b6, &b7, &b8, &b9, &b10 };
                Launch(attn, tiles, (uint)nKv, (uint)nSeq, 256, 0, stream, bargs);
                return;
            }
            int keySplits = PrefillSplits((long)tiles * nKv * nSeq, maxKeys, (long)nSeq * rowsPerSeq * nHead, scratchBytes, sms);
            IntPtr bp = keySplits > 1 ? scratch : IntPtr.Zero;
            void** pargs = stackalloc void*[] { &b0, &b1, &b2, &b3, &b4, &b5, &c5, &b6, &b7, &b8, &b9, &keySplits, &bp, &b10 };
            Launch(attnPrefill, tiles * (uint)keySplits, (uint)nKv, (uint)nSeq, 256, smemAttnPrefill, stream, pargs);
            if (keySplits > 1)
            {
                IntPtr d0 = scratch, d1 = gate, d3 = outRows;
                void** dargs = stackalloc void*[] { &d0, &d1, &keySplits, &d3 };
                Launch(attnCombine, (uint)(nSeq * rowsPerSeq * nHead), 1, 1, AttnHeadDim, 0, stream, dargs);
            }
        }

        /// <summary>Key splits for a prompt-chunk attention launch of <paramref name="blocks"/> blocks
        /// (one per SM at a time): the fewest whose last wave fills the device noticeably better,
        /// keeping at least 2048 keys per split and the partials within the scratch.</summary>
        private static int PrefillSplits(long blocks, int maxKeys, long rowHeads, long scratchBytes, int sms)
        {
            static double Fill(long b, int sms) => (double)b / ((b + sms - 1) / sms * sms);
            int best = 1;
            double fill = Fill(blocks, sms);
            for (int s = 2; s <= 8; s++)
            {
                if ((long)maxKeys < 2048L * s || rowHeads * s * (AttnHeadDim + 2) * 4 > scratchBytes)
                    break;
                double f = Fill(blocks * s, sms);
                if (f > fill + 0.05)
                {
                    best = s;
                    fill = f;
                }
            }
            return best;
        }

        /// <summary>Key splits of a q8_0 decode/verify attention launch: chunks of at least 128 keys,
        /// at most 160 splits (x 4 KV heads, about three 4-warp CTAs per SM). On the RTX 5090 the
        /// former cap of 64 left 61 splits (1.4 CTAs per SM) at 60K keys: long decode 135 tok/s there,
        /// 149 with 121 splits; 241 splits fell back to 134 (the partials cost more than they hide).</summary>
        private const int AttnSplitChunk = 128;
        internal const int AttnMaxSplits = 160;
        /// <summary>Split cap of the pipelined NInfer attention kernel (one CTA per SM) and of the
        /// partial buffer the NInfer route allocates.</summary>
        internal const int AttnMaxSplitCount = 64;

        internal readonly record struct SplitAttentionPlan(int Chunk, int Splits);

        // The launch and the Verify CUDA graph key must use the same plan. Token positions
        // remain device inputs; only this grid/reduction geometry is captured as constants.
        internal static SplitAttentionPlan GetSplitAttentionPlan(int maxKeys)
        {
            if (maxKeys < 1)
                throw new ArgumentOutOfRangeException(nameof(maxKeys));
            int chunk = AttnSplitChunk;
            while (((long)maxKeys + chunk - 1) / chunk > AttnMaxSplits)
                chunk *= 2;
            return new SplitAttentionPlan(chunk, (int)(((long)maxKeys + chunk - 1) / chunk));
        }

        /// <summary>Partial floats a split attention launch of at most <paramref name="maxSplits"/> splits writes.</summary>
        public static long AttnSplitFloats(int rows, int nHead, int maxSplits) => (long)rows * nHead * maxSplits * (AttnHeadDim + 2);

        public void GateMul(IntPtr attnRows, IntPtr gate, long n, IntPtr stream)
        {
            IntPtr a0 = attnRows, a1 = gate; long a2 = n;
            void** args = stackalloc void*[] { &a0, &a1, &a2 };
            Launch(gateMul, CeilDiv(n, Block), 1, 1, Block, 0, stream, args);
        }

        /// <summary>silu(gate) * up per row: gu rows [gate | up], or (interleaved) groups of 8 gate then
        /// 8 up values, as the target's fused gate|up planes produce them.</summary>
        public void SwiGlu(IntPtr gu, IntPtr outRows, int ff, int rows, IntPtr stream, bool interleaved = false)
        {
            IntPtr a0 = gu, a1 = outRows; int a2 = ff, a3 = interleaved ? 1 : 0;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3 };
            Launch(swiglu, CeilDiv(ff, Block), (uint)rows, 1, Block, 0, stream, args);
        }

        /// <summary>The causal convolution of nSeq windows of rowsPerSeq rows. <paramref name="rawOut"/>
        /// (optional) keeps the unconvolved inputs; <paramref name="advanceState"/> moves each
        /// sequence's convolution state past its window.</summary>
        public void GdnConv(IntPtr packed, int packedWidth, int rowsPerSeq, int nSeq, int convDim, int dConv, IntPtr convW,
            IntPtr convStates, IntPtr qkvOut, IntPtr rawOut, bool advanceState, IntPtr stream, IntPtr count = default)
        {
            IntPtr a0 = packed, a5 = convW, a6 = convStates, a7 = qkvOut, a8 = rawOut; int a1 = packedWidth, a2 = rowsPerSeq, a3 = convDim, a4 = dConv;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
            Launch(gdnConv, CeilDiv(convDim, Block), (uint)rowsPerSeq, (uint)nSeq, Block, 0, stream, args);
            // Every row has read the old state; now the window's tail replaces it.
            if (advanceState)
                GdnConvState(packed, packedWidth, 0, rowsPerSeq, count, nSeq, convDim, dConv, convStates, 0, 1, stream);
        }

        /// <summary>Moves convolution states past the first count[seq] (zero pointer: all
        /// rowsPerSeq) rows of a window held at <paramref name="src"/>, for
        /// <paramref name="layers"/> layers at once (layer z: src + z * layerStride floats, states
        /// convStates[z * ptrStride + seq]).</summary>
        public void GdnConvState(IntPtr src, int srcWidth, long layerStride, int rowsPerSeq, IntPtr count, int nSeq, int convDim,
            int dConv, IntPtr convStates, int ptrStride, int layers, IntPtr stream)
        {
            IntPtr a0 = src, a4 = count, a7 = convStates; int a1 = srcWidth, a3 = rowsPerSeq, a5 = convDim, a6 = dConv, a8 = ptrStride;
            long a2 = layerStride;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
            Launch(gdnConvState, CeilDiv(convDim, Block), (uint)nSeq, (uint)layers, Block, 0, stream, args);
        }

        /// <summary>L2 norms of q / k, betas and log decays; with <paramref name="count"/>, rows past
        /// count[s] of sequence s (rowsPerSeq each) are padding whose beta and decay leave the
        /// recurrent state untouched.</summary>
        public void GdnPrep(IntPtr qkv, IntPtr packed, int packedWidth, int rows, int hk, int hv, int convDim, int vDim,
            IntPtr dtBias, IntPtr ssmA, float eps, IntPtr g, IntPtr beta, IntPtr stream, IntPtr count = default, int rowsPerSeq = 1)
        {
            IntPtr a0 = qkv, a1 = packed, a8 = dtBias, a9 = ssmA, a11 = g, a12 = beta, a13 = count;
            int a2 = packedWidth, a3 = rows, a4 = hk, a5 = hv, a6 = convDim, a7 = vDim, a14 = rowsPerSeq; float a10 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14 };
            Launch(gdnPrep, CeilDiv((long)rows * (2 * hk + hv) * 32, Block), 1, 1, Block, 0, stream, args);
        }

        /// <summary>Rows per sequence from which the delta rule runs chunked (16 tokens per step on
        /// the tensor cores) instead of token by token.</summary>
        public const int GdnChunkMinRows = 32;
        internal const int GdnChunkRows = 16, GdnQkPacketBytes = 16384, GdnControlPacketBytes = 2176;

        internal static long GdnPreparedScratchBytes(int rowsPerSeq, int sequences, int hk, int hv)
        {
            if (rowsPerSeq < 1 || sequences < 1 || hk < 1 || hv < 1)
                throw new ArgumentOutOfRangeException(nameof(rowsPerSeq));
            long chunks = ((long)rowsPerSeq + GdnChunkRows - 1) / GdnChunkRows;
            return checked(chunks * sequences * ((long)hk * GdnQkPacketBytes + (long)hv * GdnControlPacketBytes));
        }

        // A step has at most totalRows rows, divided among at most sequences groups. Independent
        // rounding of each sequence's last chunk needs at most sequences-1 additional packets.
        internal static long GdnPreparedScratchCapacityBytes(int totalRows, int sequences, int hk, int hv)
        {
            if (totalRows < 1 || sequences < 1 || hk < 1 || hv < 1)
                throw new ArgumentOutOfRangeException(nameof(totalRows));
            long chunks = ((long)totalRows + GdnChunkRows - 1) / GdnChunkRows + sequences - 1;
            return checked(chunks * ((long)hk * GdnQkPacketBytes + (long)hv * GdnControlPacketBytes));
        }

        /// <summary>The gated delta rule over nSeq windows. <paramref name="verify"/> leaves the
        /// states as they were (outputs only).</summary>
        public void GdnScan(IntPtr qkv, int convDim, int hk, int hv, int rowsPerSeq, int nSeq, IntPtr g, IntPtr beta,
            IntPtr states, IntPtr outRows, bool verify, IntPtr stream, IntPtr preparedScratch = default, long preparedBytes = 0)
        {
            IntPtr a0 = qkv, a6 = g, a7 = beta, a8 = states, a9 = outRows; int a1 = convDim, a2 = hk, a3 = hv, a4 = rowsPerSeq;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a6, &a7, &a8, &a9 };
            if (verify)
                Launch(gdnScanVerify, (uint)hv, (uint)nSeq, GdnHeadDim / 16, 128, 0, stream, args);
            else if (rowsPerSeq >= GdnChunkMinRows)
            {
                if (PreparedGdnPrefill && preparedScratch != IntPtr.Zero &&
                    preparedBytes >= GdnPreparedScratchBytes(rowsPerSeq, nSeq, hk, hv))
                {
                    int chunks = (int)(((long)rowsPerSeq + GdnChunkRows - 1) / GdnChunkRows);
                    IntPtr packets = preparedScratch;
                    IntPtr control = packets + (nint)((long)nSeq * hk * chunks * GdnQkPacketBytes);
                    void** qkArgs = stackalloc void*[] { &a0, &a1, &a2, &a4, &packets };
                    Launch(gdnPrepareQk, (uint)chunks, (uint)hk, (uint)nSeq, 256, 0, stream, qkArgs);
                    void** controlArgs = stackalloc void*[] { &packets, &a6, &a7, &a2, &a3, &a4, &control };
                    Launch(gdnPrepareControl, (uint)chunks, (uint)hv, (uint)nSeq, 256, 0, stream, controlArgs);
                    void** preparedArgs = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a6, &a7, &a8, &a9, &packets, &control };
                    Launch(gdnChunkPrepared, (uint)hv * (GdnHeadDim / 32), (uint)nSeq, 1, 256, 0, stream, preparedArgs);
                    PreparedGdnLaunches++;
                }
                else
                    Launch(gdnChunk, (uint)hv * (GdnHeadDim / 32), (uint)nSeq, 1, 256, 0, stream, args);
            }
            else
                Launch(gdnScan, (uint)hv, (uint)nSeq, GdnHeadDim / 16, 128, 0, stream, args);
        }

        /// <summary>Advances the states of <paramref name="layers"/> recurrent layers over the first
        /// kept[seq] rows of a verified window (layer z: inputs at qkv + z * qkvStride and g / beta +
        /// z * gStride floats, state states[z * ptrStride + seq]).</summary>
        public void GdnReplay(IntPtr qkv, long qkvStride, int convDim, int hk, int hv, int rowsPerSeq, IntPtr kept, IntPtr g,
            IntPtr beta, long gStride, IntPtr states, int ptrStride, int nSeq, int layers, IntPtr stream)
        {
            IntPtr a0 = qkv, a6 = kept, a7 = g, a8 = beta, a10 = states; long a1 = qkvStride, a9 = gStride;
            int a2 = convDim, a3 = hk, a4 = hv, a5 = rowsPerSeq, a11 = ptrStride;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11 };
            Launch(gdnReplay, (uint)hv, (uint)nSeq, (uint)(layers * (GdnHeadDim / 16)), 128, 0, stream, args);
        }

        /// <summary>y[n, outDim] (+)= xq[n] W^T for a Q4_K (ggmlType 12) or Q6_K (14) weight on the
        /// int8 tensor cores, in launches of up to <see cref="QuantMmaRows"/> rows. Q6_K cannot
        /// accumulate.</summary>
        public void GemmQuant(int ggmlType, IntPtr w, long rowBytes, int outDim, int k, IntPtr xq, IntPtr xd, int n, IntPtr y,
            bool accumulate, IntPtr stream)
        {
            if (ggmlType != 12 && (ggmlType != 14 || accumulate))
                throw new NotSupportedException($"[q35-cuda] int8 GEMM of ggml type {ggmlType}{(accumulate ? " (accumulating)" : "")}");
            // Split K until the row tiles fill the GPU twice over (at least two superblocks per split);
            // split blocks add atomically, so a non-accumulating product starts from zero.
            uint tiles = CeilDiv(outDim, 64);
            int superblocks = k / 256, splits = 1;
            while (DraftSplitK && tiles * splits < 2 * sms && splits * 4 <= superblocks)
                splits *= 2;
            if (splits > 1 && !accumulate)
                CudaDriverApi.cuMemsetD32Async(y, 0, new UIntPtr((ulong)n * (ulong)outDim), stream).ThrowOnError();
            for (int r0 = 0; r0 < n; r0 += QuantMmaRows)
            {
                int rows = Math.Min(QuantMmaRows, n - r0);
                int i = rows <= 8 ? 0 : rows <= 16 ? 1 : rows <= 32 ? 2 : 3;
                IntPtr a0 = w, a4 = xq + (nint)((long)r0 * k), a5 = xd + (nint)((long)r0 * (k / 32) * 4), a7 = y + (nint)((long)r0 * outDim * 4);
                long a1 = rowBytes; int a2 = outDim, a3 = k, a6 = rows, a8 = accumulate ? 1 : 0;
                if (ggmlType == 12)
                {
                    void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
                    Launch(mmaQ4k[i], tiles, (uint)splits, 1, 128, 0, stream, args);
                }
                else
                {
                    void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
                    Launch(mmaQ6k[i], tiles, (uint)splits, 1, 128, 0, stream, args);
                }
            }
        }

        public void RowsByIndex(IntPtr src, int width, IntPtr idx, IntPtr dst, int rows, IntPtr stream)
        {
            IntPtr a0 = src, a2 = idx, a3 = dst; int a1 = width;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3 };
            Launch(rowsByIndex, (uint)rows, 1, 1, Block, 0, stream, args);
        }

        /// <summary>Draft q / k head norms and NeoX RoPE (see ts_q35_df_prep_f32).</summary>
        public void DfPrep(IntPtr q, IntPtr k, IntPtr v, int rows, int rowsPerLane, int nHead, int nKv, IntPtr qNorm, IntPtr kNorm,
            float eps, IntPtr pos, float ropeBase, IntPtr qOut, IntPtr kDst, IntPtr vDst, long headStride, int ringRows,
            IntPtr slotPos, IntPtr laneOf, IntPtr stream)
        {
            IntPtr a0 = q, a1 = k, a2 = v, a7 = qNorm, a8 = kNorm, a10 = pos, a12 = qOut, a13 = kDst, a14 = vDst, a17 = slotPos, a18 = laneOf;
            int a3 = rows, a4 = rowsPerLane, a5 = nHead, a6 = nKv, a16 = ringRows; float a9 = eps, a11 = ropeBase; long a15 = headStride;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18 };
            long warps = (long)rows * ((qOut != IntPtr.Zero ? nHead : 0) + 2 * nKv);
            Launch(dfPrep, CeilDiv(warps * 32, Block), 1, 1, Block, 0, stream, args);
        }

        /// <summary>Draft attention of every lane's block over its ring and itself, then the
        /// merge of the key ranges into outRows [lanes * b][nHead * 128].</summary>
        public void DfAttention(IntPtr q, IntPtr ringK, IntPtr ringV, IntPtr slotPos, int ringRows, IntPtr blockK, IntPtr blockV,
            int b, int lanes, int nHead, int nKv, IntPtr anchors, int swa, float scale, IntPtr partial, IntPtr outRows, IntPtr stream)
        {
            int chunk = DraftAttnChunk, splits = (ringRows + b + chunk - 1) / chunk;
            IntPtr a0 = q, a1 = ringK, a2 = ringV, a3 = slotPos, a5 = blockK, a6 = blockV, a10 = anchors, a15 = partial;
            int a4 = ringRows, a7 = b, a8 = nHead, a9 = nKv, a11 = swa, a13 = chunk, a14 = splits; float a12 = scale;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15 };
            Launch(dfAttn, (uint)splits, (uint)nKv, (uint)lanes, 128, 0, stream, args);
            IntPtr c0 = partial, c2 = outRows; int c1 = splits;
            void** cargs = stackalloc void*[] { &c0, &c1, &c2 };
            Launch(dfCombine, (uint)(lanes * b * nHead), 1, 1, DraftHeadDim, 0, stream, cargs);
        }

        private const int DraftAttnChunk = 256;

        /// <summary>Partial floats a draft attention launch writes.</summary>
        public static long DfAttnPartialFloats(int rows, int nHead, int ringRows, int b)
            => (long)rows * nHead * ((ringRows + b + DraftAttnChunk - 1) / DraftAttnChunk) * (DraftHeadDim + 2);

        public void DfConv(IntPtr x, IntPtr coef, int coefWidth, IntPtr convBase, int side, int taps, int groupSize, int hidden, int b,
            int rows, IntPtr outRows, IntPtr stream)
        {
            IntPtr a0 = x, a1 = coef, a3 = convBase, a9 = outRows; int a2 = coefWidth, a4 = side, a5 = taps, a6 = groupSize, a7 = hidden, a8 = b;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            Launch(dfConv, CeilDiv(hidden, Block), (uint)rows, 1, Block, 0, stream, args);
        }

        public void DfGather(IntPtr src, int width, int perLane, int first, int count, int lanes, IntPtr dst, IntPtr stream)
        {
            IntPtr a0 = src, a5 = dst; int a1 = width, a2 = perLane, a3 = first, a4 = count;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(dfGather, (uint)(lanes * count), 1, 1, Block, 0, stream, args);
        }

        public void TopK(IntPtr x, int rows, int n, int k, IntPtr idxOut, IntPtr valOut, IntPtr stream,
            IntPtr partialIdx = default, IntPtr partialVal = default)
        {
            if (k < 1 || k > 16)
                throw new NotSupportedException($"[q35-cuda] top-{k} (at most 16)");
            if (PartitionedTopK && n >= 16384 && partialIdx != IntPtr.Zero && partialVal != IntPtr.Zero)
            {
                IntPtr p0 = x, p4 = partialIdx, p5 = partialVal;
                int p1 = n, p2 = k, p3 = TopKPartitions;
                void** pargs = stackalloc void*[] { &p0, &p1, &p2, &p3, &p4, &p5 };
                Launch(topkPartial, (uint)TopKPartitions, (uint)rows, 1, 256, 0, stream, pargs);
                IntPtr m0 = partialIdx, m1 = partialVal, m4 = idxOut, m5 = valOut;
                int m2 = k, m3 = TopKPartitions;
                void** margs = stackalloc void*[] { &m0, &m1, &m2, &m3, &m4, &m5 };
                Launch(topkMerge, (uint)rows, 1, 1, 256, 0, stream, margs);
                return;
            }
            IntPtr a0 = x, a3 = idxOut, a4 = valOut; int a1 = n, a2 = k;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(topk, (uint)rows, 1, 1, 256, 0, stream, args);
        }

        public void DfSelect(IntPtr cand, IntPtr unary, IntPtr ph, IntPtr pred, IntPtr succ, long rowBytes, IntPtr anchors, int lanes,
            int gamma, int k, int rank, float logitScale, IntPtr draftsOut, IntPtr confOut, IntPtr stream)
        {
            IntPtr a0 = cand, a1 = unary, a2 = ph, a3 = pred, a4 = succ, a6 = anchors, a12 = draftsOut, a13 = confOut;
            long a5 = rowBytes; int a7 = gamma, a8 = k, a9 = rank; float a10 = logitScale;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a12, &a13 };
            Launch(dfSelect, (uint)lanes, 1, 1, 256, 0, stream, args);
        }

        public void CopyRows(IntPtr src, long srcStride, IntPtr dst, long dstStride, int width, int rows, IntPtr stream)
        {
            IntPtr a0 = src, a2 = dst; long a1 = srcStride, a3 = dstStride; int a4 = width;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(copyRows, (uint)rows, 1, 1, Block, 0, stream, args);
        }

        public void GdnOut(IntPtr o, IntPtr packed, int packedWidth, int convDim, int rows, int hv, IntPtr ssmNorm, float eps,
            IntPtr y, IntPtr stream)
        {
            IntPtr a0 = o, a1 = packed, a6 = ssmNorm, a8 = y; int a2 = packedWidth, a3 = convDim, a4 = rows, a5 = hv; float a7 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
            Launch(gdnOut, CeilDiv((long)rows * hv * 32, Block), 1, 1, Block, 0, stream, args);
        }

        public void Argmax(IntPtr x, int rows, int n, IntPtr outIdx, IntPtr stream)
        {
            IntPtr a0 = x, a2 = outIdx; int a1 = n;
            void** args = stackalloc void*[] { &a0, &a1, &a2 };
            Launch(argmax, (uint)rows, 1, 1, 1024, 0, stream, args);
        }

        public void Dispose()
        {
            if (_ninferVerify8Gemm != IntPtr.Zero)
            {
                Console.Error.WriteLine($"[q35-cuda] NInfer Verify8 GEMM host submissions={NinferVerify8GemmHostSubmissions} (capture included; not GPU graph replays), module={module.ImageSha256}");
                _ninferVerify8Gemm = IntPtr.Zero;
            }
            module.Dispose();
        }
    }
}
