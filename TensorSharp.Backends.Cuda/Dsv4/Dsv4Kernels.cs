// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Launchers for the DeepSeek V4 direct-CUDA kernels
// (native/kernels/tensorsharp_dsv4_kernels.cu, compiled to
// tensorsharp_dsv4_kernels.ptx). One instance per device; the module is
// loaded against whichever context is current at construction time.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal sealed unsafe class Dsv4Kernels : IDisposable
    {
        private const int BlockSize = 256;
        private const int HC = 4; // hyper-connection streams (Dsv4CudaEngine.HC)

        private readonly CudaModule module;

        private readonly IntPtr embed;
        private readonly IntPtr hcRms;
        private readonly IntPtr hcGatesComb;
        private readonly IntPtr hcCollapse;
        private readonly IntPtr hcPost;
        private readonly IntPtr hcHead;
        private readonly IntPtr attnPrep;
        private readonly IntPtr apeAdd;
        private readonly IntPtr compress;
        private readonly IntPtr persist;
        private readonly IntPtr idxPrep;
        private readonly IntPtr idxScores;
        private readonly IntPtr topk;
        private readonly IntPtr attention;
        private readonly IntPtr attnFinish;
        private readonly IntPtr regroup;
        private readonly IntPtr moeSelect;
        private readonly IntPtr moeCount;
        private readonly IntPtr moeScan;
        private readonly IntPtr moeScatter;
        private readonly IntPtr moeGateUp;
        private readonly IntPtr moeDown;
        private readonly IntPtr moeGateUpStaged;
        private readonly IntPtr moeDownStaged;
        private readonly IntPtr moeMma;
        private readonly IntPtr moeGateUpDecode;
        private readonly IntPtr moeDownDecode;
        private readonly IntPtr moeGateUpDecodeQ80V, moeDownDecodeQ80V, moeGateUpDecodeQ80VPair;
        private readonly IntPtr denseQ81Rows;
        private readonly IntPtr moeScatterAdd;
        private readonly IntPtr moeStagePlan, moeStageCopy, moeUnstage, doorbellPublish, doorbellWaitAdd;
        // Staging that keeps earlier calls' experts (optional: a module before it re-stages per call).
        private readonly IntPtr moeStageCopyFrom;
        private readonly IntPtr doorbellPublishSkip, doorbellWaitAddSkip, moeDownDecodeShort;
        // Gate/up decode with sixteen lanes a row (optional).
        private readonly IntPtr moeGateUpDecodeHalf;
        // TS_MOE_GATEUP_HALF=0: a warp a gate/up row again (the sums' order differs).
        private static readonly bool _moeGateUpHalf = Environment.GetEnvironmentVariable("TS_MOE_GATEUP_HALF") != "0";
        // The doorbell with the route-tail skip (optional).
        private readonly IntPtr doorbellPublishTail;
        // The doorbell that plans the GPU's share of a decode step's missed experts and leaves it out (optional).
        private readonly IntPtr doorbellPublishHybrid;
        private readonly IntPtr denseBf16Rows, denseQ81RowsSplitK, denseBf16RowsSplitK;
        // The same products for 2..16 rows in one block, the weights read once (optional: a module before it launches a
        // block per row).
        private readonly IntPtr denseQ81RowsMulti;
        // Decode once, dot every row (optional): [R index (1, 2, 4, 8 rows a pass), W index (1, 2, 4 warps an output)].
        private readonly IntPtr[,] denseFast = new IntPtr[4, 3];
        private readonly bool _hasDenseFast;

        /// <summary>Activation rows the one-block-for-every-row dense kernels take.</summary>
        public const int DenseMultiMaxRows = 16;

        // TS_DENSE_MULTI_ROWS=0: a block per row again (the products are the same; this is for measuring it).
        private static readonly bool _denseMulti = Environment.GetEnvironmentVariable("TS_DENSE_MULTI_ROWS") != "0";
        // TS_DENSE_FAST=0: the formats the decode-once kernels read go back to the per-row kernels above.
        private static readonly bool _denseFastEnv = Environment.GetEnvironmentVariable("TS_DENSE_FAST") != "0";

        /// <summary>Whether <see cref="DenseQ81Rows"/> runs a <paramref name="wtype"/> projection of <paramref name="inDim"/>
        /// inputs on the decode-once kernels (Q8_0, Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL; up to 16 rows).</summary>
        public bool DenseFastReads(int wtype, long inDim)
            => _hasDenseFast && _denseFastEnv && wtype switch
            {
                8 or 20 => inDim % 32 == 0,                       // Q8_0, IQ4_NL
                42 => inDim % 64 == 0,                            // Q2_0
                11 or 12 or 13 or 14 or 23 => inDim % 256 == 0,   // Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS
                _ => false,
            };

        /// <summary>Warps sharing an output row in the decode-once kernels: by shape alone, so a row's sums never depend on
        /// how many rows a call holds. Few outputs need the K split to fill the GPU (measured on the RTX 5090 at Qwen3.8
        /// Flash Next's shapes: 10240 -> 320 and 2560 -> 640 fastest at 4, 2560 -> 6144 at 2, the rest at 1).</summary>
        private static int DenseFastWarps(int outDim) => outDim <= 2048 ? 4 : outDim > 4096 && outDim <= 8192 ? 2 : 1;
        private readonly IntPtr hcMean, hcMixPartials, hcPreFinish;
        private readonly IntPtr dsparkPrep;
        private readonly IntPtr dsparkGather;
        private readonly IntPtr dsparkArgmax;
        private readonly IntPtr dsparkConf;

        // V4.1
        private readonly IntPtr v41AttnPrep;
        private readonly IntPtr v41Compress;
        private readonly IntPtr v41Commit;
        private readonly IntPtr v41Persist;
        private readonly IntPtr v41IdxPrep;
        private readonly IntPtr v41IdxScores;
        private readonly IntPtr v41Candidate;
        private readonly IntPtr v41EngramGate;
        private readonly IntPtr v41EngramGather;
        private readonly IntPtr[] gemv;
        private readonly IntPtr[] gemvWarp;
        // The warp GEMV over BF16 weights (null on a module without it).
        private readonly IntPtr[] gemvWarpBf16;
        // The block-per-output GEMV, F32 and BF16 weights (null on a module without it).
        private readonly IntPtr[] gemvBlock, gemvBlockBf16;
        // TS_DSV4_GEMV_BLOCK=0: wide outputs keep a warp each (the sums' order differs).
        private static readonly bool _gemvBlockEnv = Environment.GetEnvironmentVariable("TS_DSV4_GEMV_BLOCK") != "0";

        /// <summary>Threads of the block-per-output GEMV (the kernel's TS_DSV4_GEMV_BLOCK_T).</summary>
        private const int GemvBlockThreads = 256;

        /// <summary>Whether <see cref="GemvBf16"/> is available.</summary>
        public bool HasGemvBf16 => gemvWarpBf16 != null;
        private readonly IntPtr gemvCombine;

        /// <summary>
        /// Whether ts_dsv4_moe_mma_f32 carries its tensor-core body: it is guarded by
        /// __CUDA_ARCH__ &gt;= 800 and compiles to nothing below that, which is what the
        /// committed compute_75 baseline holds. See CudaKernels.ArchGuardedKernelsUsable.
        /// </summary>
        public bool HasMoeMma { get; }

        private Dsv4Kernels(CudaModule module)
        {
            this.module = module;
            CudaDriverApi.cuCtxGetDevice(out int device);
            CudaDriverApi.cuDeviceGetAttribute(out int ccMajor, 75 /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR */, device);
            HasMoeMma = CudaKernels.ArchGuardedKernelsUsable(module.CompiledArchOf("ts_dsv4_moe_mma_f32"), ccMajor);
            embed = module.GetFunction("ts_dsv4_embed_f32");
            v41AttnPrep = module.GetFunction("ts_dsv41_attn_prep_f32");
            v41Compress = module.GetFunction("ts_dsv41_compress_f32");
            v41Commit = module.GetFunction("ts_dsv41_commit_f32");
            v41Persist = module.GetFunction("ts_dsv41_persist_f32");
            v41IdxPrep = module.GetFunction("ts_dsv41_idx_prep_f32");
            v41IdxScores = module.GetFunction("ts_dsv41_idx_scores_f32");
            v41Candidate = module.GetFunction("ts_dsv41_candidate_f32");
            v41EngramGate = module.GetFunction("ts_dsv41_engram_gate_f32");
            v41EngramGather = module.GetFunction("ts_dsv41_engram_gather_f32");
            gemv = new[]
            {
                module.GetFunction("ts_dsv4_gemv_1_f32"), module.GetFunction("ts_dsv4_gemv_4_f32"),
                module.GetFunction("ts_dsv4_gemv_8_f32"), module.GetFunction("ts_dsv4_gemv_16_f32"),
            };
            gemvWarp = new[]
            {
                module.GetFunction("ts_dsv4_gemv_warp_1_f32"), module.GetFunction("ts_dsv4_gemv_warp_4_f32"),
                module.GetFunction("ts_dsv4_gemv_warp_8_f32"), module.GetFunction("ts_dsv4_gemv_warp_16_f32"),
            };
            try
            {
                gemvWarpBf16 = new[]
                {
                    module.GetFunction("ts_dsv4_gemv_warp_bf16_1_f32"), module.GetFunction("ts_dsv4_gemv_warp_bf16_4_f32"),
                    module.GetFunction("ts_dsv4_gemv_warp_bf16_8_f32"), module.GetFunction("ts_dsv4_gemv_warp_bf16_16_f32"),
                };
            }
            catch (CudaException ex) when (ex.ErrorCode == 500)
            {
                gemvWarpBf16 = null;
            }
            try
            {
                gemvBlock = new[]
                {
                    module.GetFunction("ts_dsv4_gemv_block_1_f32"), module.GetFunction("ts_dsv4_gemv_block_4_f32"),
                    module.GetFunction("ts_dsv4_gemv_block_8_f32"), module.GetFunction("ts_dsv4_gemv_block_16_f32"),
                };
                gemvBlockBf16 = new[]
                {
                    module.GetFunction("ts_dsv4_gemv_block_bf16_1_f32"), module.GetFunction("ts_dsv4_gemv_block_bf16_4_f32"),
                    module.GetFunction("ts_dsv4_gemv_block_bf16_8_f32"), module.GetFunction("ts_dsv4_gemv_block_bf16_16_f32"),
                };
            }
            catch (CudaException ex) when (ex.ErrorCode == 500)
            {
                gemvBlock = gemvBlockBf16 = null;
            }
            gemvCombine = module.GetFunction("ts_dsv4_gemv_combine_f32");
            hcRms = module.GetFunction("ts_dsv4_hc_rms_f32");
            hcGatesComb = module.GetFunction("ts_dsv4_hc_gates_comb_f32");
            hcCollapse = module.GetFunction("ts_dsv4_hc_collapse_f32");
            hcPost = module.GetFunction("ts_dsv4_hc_post_f32");
            hcHead = module.GetFunction("ts_dsv4_hc_head_f32");
            attnPrep = module.GetFunction("ts_dsv4_attn_prep_f32");
            apeAdd = module.GetFunction("ts_dsv4_ape_add_f32");
            compress = module.GetFunction("ts_dsv4_compress_f32");
            persist = module.GetFunction("ts_dsv4_persist_f32");
            idxPrep = module.GetFunction("ts_dsv4_idx_prep_f32");
            idxScores = module.GetFunction("ts_dsv4_idx_scores_f32");
            topk = module.GetFunction("ts_dsv4_topk_f32");
            attention = module.GetFunction("ts_dsv4_attention_f32");
            attnFinish = module.GetFunction("ts_dsv4_attn_finish_f32");
            regroup = module.GetFunction("ts_dsv4_regroup_f32");
            moeSelect = module.GetFunction("ts_dsv4_moe_select_f32");
            moeCount = module.GetFunction("ts_dsv4_moe_count_i32");
            moeScan = module.GetFunction("ts_dsv4_moe_scan_i32");
            moeScatter = module.GetFunction("ts_dsv4_moe_scatter_i32");
            moeGateUp = module.GetFunction("ts_dsv4_moe_gateup_f32");
            moeDown = module.GetFunction("ts_dsv4_moe_down_f32");
            moeGateUpStaged = module.GetFunction("ts_dsv4_moe_gateup_staged_f32");
            moeDownStaged = module.GetFunction("ts_dsv4_moe_down_staged_f32");
            moeMma = module.GetFunction("ts_dsv4_moe_mma_f32");
            moeGateUpDecode = module.GetFunction("ts_dsv4_moe_gateup_decode_f32");
            moeDownDecode = module.GetFunction("ts_dsv4_moe_down_decode_f32");
            try { moeDownDecodeShort = module.GetFunction("ts_dsv4_moe_down_decode_short_f32"); }
            catch (CudaException ex) when (ex.ErrorCode == 500) { moeDownDecodeShort = IntPtr.Zero; }
            try { moeGateUpDecodeHalf = module.GetFunction("ts_dsv4_moe_gateup_decode_half_f32"); }
            catch (CudaException ex) when (ex.ErrorCode == 500) { moeGateUpDecodeHalf = IntPtr.Zero; }
            try
            {
                moeGateUpDecodeQ80V = module.GetFunction("ts_dsv4_moe_gateup_decode_q80v_f32");
                moeDownDecodeQ80V = module.GetFunction("ts_dsv4_moe_down_decode_q80v_f32");
            }
            catch (CudaException ex) when (ex.ErrorCode == 500) { moeGateUpDecodeQ80V = moeDownDecodeQ80V = IntPtr.Zero; }
            try { moeGateUpDecodeQ80VPair = module.GetFunction("ts_dsv4_moe_gateup_decode_q80v_pair_f32"); }
            catch (CudaException ex) when (ex.ErrorCode == 500) { moeGateUpDecodeQ80VPair = IntPtr.Zero; }
            denseQ81Rows = module.GetFunction("ts_dsv4_dense_q81_rows_f32");
            moeScatterAdd = module.GetFunction("ts_dsv4_moe_scatter_add_f32");
            hcMean = module.GetFunction("ts_dsv4_hc_mean_f32");
            hcMixPartials = module.GetFunction("ts_dsv4_hc_mix_partials_f32");
            hcPreFinish = module.GetFunction("ts_dsv4_hc_pre_finish_f32");
            dsparkPrep = module.GetFunction("ts_dsv4_dspark_prep_f32");
            dsparkGather = module.GetFunction("ts_dsv4_dspark_gather_f32");
            dsparkArgmax = module.GetFunction("ts_dsv4_dspark_argmax_f32");
            dsparkConf = module.GetFunction("ts_dsv4_dspark_conf_f32");
            // A module built from source older than the Q5_1 expert decoder lacks the marker.
            try { module.GetFunction("ts_dsv4_caps_q5_1_experts"); DecodesQ5_1Experts = true; }
            catch (CudaException ex) when (ex.ErrorCode == 500) { } // CUDA_ERROR_NOT_FOUND
            // Likewise for the IQ1_S / IQ2_S decoders and IQ2_XXS's grouped one.
            try { module.GetFunction("ts_dsv4_caps_lowbit_iq_experts"); DecodesLowBitIqExperts = true; }
            catch (CudaException ex) when (ex.ErrorCode == 500) { }
            // Expert residency's per-expert tables and staging kernels, and the host experts' doorbell: a module
            // older than them (the committed baseline before its refresh) runs every model that fits as before,
            // and the engines refuse residency on it rather than run kernels that ignore the tables.
            try
            {
                moeStagePlan = module.GetFunction("ts_dsv4_moe_stage_plan");
                moeStageCopy = module.GetFunction("ts_dsv4_moe_stage_copy");
                moeUnstage = module.GetFunction("ts_dsv4_moe_unstage");
                HasExpertTables = true;
                try { moeStageCopyFrom = module.GetFunction("ts_dsv4_moe_stage_copy_from"); }
                catch (CudaException ex) when (ex.ErrorCode == 500) { moeStageCopyFrom = IntPtr.Zero; }
            }
            catch (CudaException ex) when (ex.ErrorCode == 500) { }
            try
            {
                doorbellPublish = module.GetFunction("ts_moe_doorbell_publish");
                doorbellWaitAdd = module.GetFunction("ts_moe_doorbell_wait_add");
                HasHostExperts = true;
                try
                {
                    doorbellPublishSkip = module.GetFunction("ts_moe_doorbell_publish_skip");
                    doorbellWaitAddSkip = module.GetFunction("ts_moe_doorbell_wait_add_skip");
                    try { doorbellPublishTail = module.GetFunction("ts_moe_doorbell_publish_tail"); }
                    catch (CudaException ex) when (ex.ErrorCode == 500) { doorbellPublishTail = IntPtr.Zero; }
                    try { doorbellPublishHybrid = module.GetFunction("ts_moe_doorbell_publish_hybrid"); }
                    catch (CudaException ex) when (ex.ErrorCode == 500) { doorbellPublishHybrid = IntPtr.Zero; }
                }
                catch (CudaException ex) when (ex.ErrorCode == 500)
                {
                    doorbellPublishSkip = doorbellWaitAddSkip = IntPtr.Zero;
                }
            }
            catch (CudaException ex) when (ex.ErrorCode == 500) { }
            try
            {
                module.GetFunction("ts_dsv4_caps_legacy_quants");
                DecodesLegacyQuants = true;
                denseBf16Rows = module.GetFunction("ts_dense_bf16_rows_f32");
                denseQ81RowsSplitK = module.GetFunction("ts_dsv4_dense_q81_rows_splitk_f32");
                try { denseQ81RowsMulti = module.GetFunction("ts_dsv4_dense_q81_rows_multi_f32"); }
                catch (CudaException ex) when (ex.ErrorCode == 500) { denseQ81RowsMulti = IntPtr.Zero; }
                try
                {
                    int[] rs = { 1, 2, 4, 8 }, ws = { 1, 2, 4 };
                    for (int ri = 0; ri < 4; ri++)
                        for (int wi = 0; wi < 3; wi++)
                            denseFast[ri, wi] = module.GetFunction($"ts_dsv4_dense_fast_r{rs[ri]}_w{ws[wi]}_f32");
                    _hasDenseFast = true;
                }
                catch (CudaException ex) when (ex.ErrorCode == 500) { _hasDenseFast = false; }
                denseBf16RowsSplitK = module.GetFunction("ts_dense_bf16_rows_splitk_f32");
            }
            catch (CudaException ex) when (ex.ErrorCode == 500) { }
        }

        /// <summary>Whether this module's expert kernels decode Q5_1 rows. False on a module built
        /// from older kernel source (the committed PTX baseline before its refresh), whose kernels
        /// would read a Q5_1 row as Q8_0; <see cref="CudaMoe.RequireExpertDecoders"/> refuses those
        /// experts there.</summary>
        public bool DecodesQ5_1Experts { get; }

        /// <summary>Whether this module's expert kernels decode IQ1_S and IQ2_S rows on every route,
        /// and IQ2_XXS rows on the grouped (register-staged and tensor-core) routes as well as the
        /// warp-per-row ones. False on a module built from older kernel source, which reads the first
        /// two as Q8_0 everywhere and IQ2_XXS as Q8_0 in a grouped kernel:
        /// <see cref="CudaMoe.RequireExpertDecoders"/> refuses IQ1_S and IQ2_S experts there, and
        /// <see cref="CudaMoe.GroupedSupportsType"/> keeps IQ2_XXS on the warp-per-row kernels.</summary>
        public bool DecodesLowBitIqExperts { get; }

        /// <summary>Whether the loaded module's warp-per-row decoders read Q2_0, Q4_0 and Q5_0 and it has the BF16
        /// dense kernel (ISTA-DASLab's Qwen3.8-Flash-Next Q2_0 pack); an older module answers NaN for them.</summary>
        public bool DecodesLegacyQuants { get; }

        /// <summary>Whether the module's expert kernels read per-expert address tables and it has the staging
        /// kernels (expert residency).</summary>
        public bool HasExpertTables { get; }

        /// <summary>Whether staging can keep what earlier calls staged (<see cref="MoeStageCopyNew"/>).</summary>
        public bool HasStageKeep => moeStageCopyFrom != IntPtr.Zero;

        /// <summary>Whether the module has the host experts' doorbell kernels.</summary>
        public bool HasHostExperts { get; }

        public static Dsv4Kernels Create()
        {
            string path = CudaKernels.LocatePtxPath("tensorsharp_dsv4_kernels.ptx");
            if (path == null)
            {
                throw new InvalidOperationException(
                    "tensorsharp_dsv4_kernels.ptx could not be located next to the application or under " +
                    "a 'cuda_kernels' folder. Build TensorSharp.Backends.Cuda with a CUDA 13 nvcc on the PATH.");
            }
            return new Dsv4Kernels(CudaModule.LoadFromFile(path, identify: true));
        }

        /// <summary>Kernels over an already loaded module (tests that load a specific image).
        /// Requires the module's context to be current.</summary>
        internal static Dsv4Kernels FromModule(CudaModule module) => new Dsv4Kernels(module);

        /// <summary>The SHA-256 of the module image the driver loaded (<see cref="CudaModule.ImageSha256"/>);
        /// null for a module loaded without identifying it.</summary>
        public string ModuleImageSha256 => module.ImageSha256;

        private static void Launch(IntPtr fn, uint gx, uint gy, uint gz, int block, uint sharedBytes, IntPtr stream, void** args)
        {
            CudaDriverApi.cuLaunchKernel(fn, gx, gy, gz, (uint)block, 1, 1, sharedBytes, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>Launch with a two-dimensional block, for the kernels that put
        /// one warp on the contracted dimension and independent rows on y.</summary>
        private static void Launch2D(IntPtr fn, uint gx, uint gy, int bx, int by, uint sharedBytes, IntPtr stream, void** args)
        {
            CudaDriverApi.cuLaunchKernel(fn, gx, gy, 1, (uint)bx, (uint)by, 1, sharedBytes, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        private static uint CeilDiv(long n, int d) => (uint)Math.Max(1, (n + d - 1) / d);

        /// <summary>The instantiation (row bound 1, 4, 8 or 16) a small-batch kernel runs <paramref name="rows"/> rows with.</summary>
        internal static int RowBoundIndex(int rows) => rows <= 1 ? 0 : rows <= 4 ? 1 : rows <= 8 ? 2 : 3;

        /// <summary>Device pointer behind an engine scratch tensor (null =>
        /// nullptr, which several kernels accept as "argument absent").</summary>
        private static IntPtr P(Tensor t) => Dsv4CudaEngine.Ptr(t);

        public void Embed(IntPtr w, Tensor tokens, Tensor xs, int wtype, long rowBytes, int nt, int e, IntPtr stream)
        {
            IntPtr a0 = w, a1 = P(tokens), a2 = P(xs);
            int a3 = wtype; long a4 = rowBytes; int a5 = nt, a6 = e;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
            Launch(embed, CeilDiv(e, BlockSize), (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void HcRms(Tensor xs, Tensor inv, int nt, int flatDim, float eps, IntPtr stream)
        {
            IntPtr a0 = P(xs), a1 = P(inv);
            int a2 = flatDim; float a3 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3 };
            Launch(hcRms, (uint)nt, 1, 1, BlockSize, 0, stream, args);
        }

        public void HcGatesComb(Tensor mixes, Tensor inv, IntPtr scale, IntPtr baseW, Tensor pre, Tensor post, Tensor comb,
            int nt, int iters, float eps, IntPtr stream)
        {
            IntPtr a0 = P(mixes), a1 = P(inv), a2 = scale, a3 = baseW, a4 = P(pre), a5 = P(post), a6 = P(comb);
            int a7 = nt, a8 = iters; float a9 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            Launch(hcGatesComb, CeilDiv(nt, 128), 1, 1, 128, 0, stream, args);
        }

        public void HcCollapse(Tensor xs, Tensor pre, Tensor cur, int nt, int e, IntPtr stream)
        {
            IntPtr a0 = P(xs), a1 = P(pre), a2 = P(cur);
            int a3 = nt, a4 = e;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(hcCollapse, CeilDiv(e, BlockSize), (uint)nt, 1, BlockSize, 0, stream, args);
        }

        /// <summary>Flattened-stream values per partial CTA of the hyper-connection pre-block (TS_HC_SLICE).</summary>
        public const int HcSlice = 512;

        /// <summary>Floats per (token, slice) partial: the 24 mixes, then the sum of squares.</summary>
        public const int HcPartialFloats = 25;

        /// <summary>The hyper-connection pre-block, first half: per slice of the flattened
        /// streams and token, the mixes' partial dot products with hc_fn (F32 or Q8_0) and the partial
        /// sum of squares.</summary>
        public void HcMixPartials(Tensor xs, IntPtr fn, int wtype, Tensor partials, int flat, int rows, IntPtr stream)
        {
            IntPtr a0 = P(xs), a1 = fn, a3 = P(partials);
            int a2 = wtype, a4 = flat;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(hcMixPartials, (uint)(flat / HcSlice), (uint)rows, 1, 256, 0, stream, args);
        }

        /// <summary>The hyper-connection pre-block, second half, per token: the gates into pre/post/comb,
        /// the streams collapsed into cur (with the token's own pre gates, or
        /// <paramref name="collapseWith"/>, or their plain mean) and cur RMS normed with
        /// <paramref name="normW"/>; <paramref name="publish"/>, when set, receives the pre gates.</summary>
        public void HcPreFinish(Tensor partials, int nSlices, Tensor xs, IntPtr scale, IntPtr baseW, IntPtr normW,
            Tensor pre, Tensor post, Tensor comb, Tensor collapseWith, bool meanCollapse, Tensor publish,
            Tensor cur, int e, int rows, int iters, float hcEps, float rmsEps, IntPtr stream)
        {
            IntPtr a0 = P(partials), a2 = P(xs), a3 = scale, a4 = baseW, a5 = normW, a6 = P(pre), a7 = P(post),
                a8 = P(comb), a9 = P(collapseWith), a11 = P(publish), a12 = P(cur);
            int a1 = nSlices, a10 = meanCollapse ? 1 : 0, a13 = e, a14 = iters;
            float a15 = hcEps, a16 = rmsEps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16 };
            Launch(hcPreFinish, (uint)rows, 1, 1, 256, 0, stream, args);
        }

        public void HcPost(Tensor xs, Tensor blockOut, Tensor post, Tensor comb, Tensor xsOut, int nt, int e, IntPtr stream)
        {
            IntPtr a0 = P(xs), a1 = P(blockOut), a2 = P(post), a3 = P(comb), a4 = P(xsOut);
            int a5 = nt, a6 = e;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
            Launch(hcPost, CeilDiv(4L * e, BlockSize), (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void HcHead(Tensor x, IntPtr fn, IntPtr scale, IntPtr baseW, Tensor cur, int e, int scaleCount, int baseCount, float eps, IntPtr stream, int rows = 1)
        {
            IntPtr a0 = P(x), a1 = fn, a2 = scale, a3 = baseW, a4 = P(cur);
            int a5 = e, a6 = scaleCount, a7 = baseCount; float a8 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
            Launch(hcHead, (uint)Math.Max(rows, 1), 1, 1, BlockSize, 0, stream, args);
        }

        public void AttnPrep(Tensor q, Tensor kvRaw, IntPtr kvNormW, IntPtr ropeTab, IntPtr ring,
            int p0, int ringRows, int nh, int hd, int nRot, float eps, int nt, IntPtr stream)
        {
            IntPtr a0 = P(q), a1 = P(kvRaw), a2 = kvNormW, a3 = ropeTab, a4 = ring;
            int a5 = p0, a6 = ringRows, a7 = nh, a8 = hd, a9 = nRot; float a10 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10 };
            Launch(attnPrep, (uint)(nh + 1), (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void ApeAdd(Tensor stScore, IntPtr ape, int p0, int ratio, int nt, int cw, IntPtr stream)
        {
            IntPtr a0 = P(stScore), a1 = ape;
            int a2 = p0, a3 = ratio, a4 = nt, a5 = cw;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(apeAdd, CeilDiv(cw, BlockSize), (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void Compress(Tensor stKv, Tensor stScore, IntPtr histKv, IntPtr histScore, IntPtr normW, IntPtr ropeTab, IntPtr cache,
            long firstBoundary, int nBlocks, int p0, int ratio, int coff, int head, int stateSize, int nRot, float eps, IntPtr stream)
        {
            IntPtr a0 = P(stKv), a1 = P(stScore), a2 = histKv, a3 = histScore, a4 = normW, a5 = ropeTab, a6 = cache;
            long a7 = firstBoundary;
            int a8 = nBlocks, a9 = p0, a10 = ratio, a11 = coff, a12 = head, a13 = stateSize, a14 = nRot;
            float a15 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15 };
            Launch(compress, (uint)nBlocks, 1, 1, BlockSize, 0, stream, args);
        }

        public void Persist(Tensor stKv, Tensor stScore, IntPtr histKv, IntPtr histScore,
            int p0, int nt, int stateSize, int cw, IntPtr stream)
        {
            IntPtr a0 = P(stKv), a1 = P(stScore), a2 = histKv, a3 = histScore;
            int a4 = p0, a5 = nt, a6 = stateSize, a7 = cw;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            long np = Math.Min(nt, stateSize);
            Launch(persist, CeilDiv(np * cw, BlockSize), 1, 1, BlockSize, 0, stream, args);
        }

        public void IdxPrep(Tensor iq, Tensor iw, IntPtr ropeTab, int p0, int ih, int id, int nRot, float iwScale, int nt, IntPtr stream)
        {
            IntPtr a0 = P(iq), a1 = P(iw), a2 = ropeTab;
            int a3 = p0, a4 = ih, a5 = id, a6 = nRot; float a7 = iwScale;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            Launch(idxPrep, (uint)ih, (uint)nt, 1, 128, 0, stream, args);
        }

        public void IdxScores(Tensor iq, Tensor iw, IntPtr lidK, Tensor scores,
            int p0, int ratio, int ih, int id, int nt, int scoreStride, int rowsMax, IntPtr stream)
        {
            IntPtr a0 = P(iq), a1 = P(iw), a2 = lidK, a3 = P(scores);
            int a4 = p0, a5 = ratio, a6 = ih, a7 = id, a8 = nt, a9 = scoreStride;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            uint shared = (uint)((ih * id + ih) * sizeof(float));
            Launch(idxScores, CeilDiv(rowsMax, 8), (uint)nt, 1, BlockSize, shared, stream, args);
        }

        /// <param name="positions">Device array of each row's position (a captured decode step), or
        /// zero for rows at p0, p0 + 1, ...</param>
        public void TopK(Tensor scores, Tensor topkIdx, Tensor topkCnt, int p0, int ratio, int k, int scoreStride, int nt, IntPtr stream,
            IntPtr positions = default)
        {
            IntPtr a0 = P(scores), a1 = P(topkIdx), a2 = P(topkCnt), a7 = positions;
            int a3 = p0, a4 = ratio, a5 = k, a6 = scoreStride;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            Launch(topk, (uint)nt, 1, 1, BlockSize, 0, stream, args);
        }

        /// <param name="positions">Device array of each row's position (a captured decode step), or
        /// zero for rows at p0, p0 + 1, ...</param>
        public void Attention(Tensor q, IntPtr ring, IntPtr comp, Tensor topkIdx, Tensor topkCnt, IntPtr sinks, Tensor attnO,
            int p0, int nSwa, int ringRows, int nh, int hd, int mode, int ratio, int k, float kqScale, int nt, IntPtr stream,
            IntPtr positions = default)
        {
            IntPtr a0 = P(q), a1 = ring, a2 = comp, a3 = P(topkIdx), a4 = P(topkCnt), a5 = sinks, a6 = P(attnO), a16 = positions;
            int a7 = p0, a8 = nSwa, a9 = ringRows, a10 = nh, a11 = hd, a12 = mode, a13 = ratio, a14 = k;
            float a15 = kqScale;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16 };
            Launch(attention, (uint)nh, (uint)nt, 1, BlockSize, 0, stream, args);
        }

        /// <param name="positions">Device array of each row's position (a batched
        /// decode step, one row per sequence), or zero for rows at p0, p0 + 1, ...</param>
        public void AttnFinish(Tensor attnO, IntPtr ropeTab, Tensor dst, int p0, int nh, int hd, int nRot, int headsPerGroup, int nt,
            IntPtr stream, IntPtr positions = default)
        {
            IntPtr a0 = P(attnO), a1 = ropeTab, a2 = P(dst), a9 = positions;
            int a3 = p0, a4 = nh, a5 = hd, a6 = nRot, a7 = headsPerGroup, a8 = nt;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            Launch(attnFinish, (uint)nh, (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void Regroup(Tensor oGtmp, Tensor oG, int g, int nt, int r, IntPtr stream)
        {
            IntPtr a0 = P(oGtmp), a1 = P(oG);
            int a2 = g, a3 = nt, a4 = r;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(regroup, CeilDiv((long)g * nt * r, BlockSize), 1, 1, BlockSize, 0, stream, args);
        }

        // ---- DSpark drafter ----

        /// <summary>Mean over the hyper-connection streams of every row, written
        /// into column block <paramref name="dstOff"/> of a [nt, dstStride] buffer
        /// (the drafter's concatenated target-layer features).</summary>
        public void HcMean(Tensor xs, Tensor dst, int nt, int e, int dstStride, int dstOff, IntPtr stream)
        {
            IntPtr a0 = P(xs), a1 = P(dst);
            int a2 = nt, a3 = e, a4 = HC, a5 = dstStride, a6 = dstOff;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
            Launch(hcMean, CeilDiv((long)nt * e, BlockSize), 1, 1, BlockSize, 0, stream, args);
        }

        /// <summary>Drafter RMS-norm + RoPE with the rope position
        /// (<paramref name="pos0"/>) and the destination slot
        /// (<paramref name="slot0"/> mod <paramref name="slotMod"/>) decoupled.
        /// <paramref name="kvOnly"/> skips the query heads (catch-up pass).</summary>
        public void DsparkPrep(Tensor q, Tensor kvRaw, IntPtr kvNormW, IntPtr ropeTab, IntPtr kvOut,
            int pos0, int slot0, int slotMod, int nh, int hd, int nRot, float eps, int nt, bool kvOnly, IntPtr stream)
        {
            IntPtr a0 = P(q), a1 = P(kvRaw), a2 = kvNormW, a3 = ropeTab, a4 = kvOut;
            int a5 = pos0, a6 = slot0, a7 = slotMod, a8 = nh, a9 = hd, a10 = nRot;
            float a11 = eps;
            int a12 = kvOnly ? 1 : 0;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12 };
            Launch(dsparkPrep, kvOnly ? 1u : (uint)(nh + 1), (uint)nt, 1, BlockSize, 0, stream, args);
        }

        /// <summary>Copy the Markov embedding row of the previous token
        /// (<paramref name="tokSlot"/> &lt; 0 selects <paramref name="anchorTok"/>).</summary>
        public void DsparkGather(IntPtr w1, Tensor toks, int tokSlot, int anchorTok, Tensor dst, int rank, IntPtr stream)
        {
            IntPtr a0 = w1, a1 = P(toks);
            int a2 = tokSlot, a3 = anchorTok;
            IntPtr a4 = P(dst);
            int a5 = rank;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(dsparkGather, 1, 1, 1, BlockSize, 0, stream, args);
        }

        /// <summary>toks[slot] = argmax(logits + bias) over the vocabulary.</summary>
        public void DsparkArgmax(Tensor logits, Tensor bias, Tensor toks, int slot, int vocab, IntPtr stream)
        {
            IntPtr a0 = P(logits), a1 = P(bias), a2 = P(toks);
            int a3 = slot, a4 = vocab;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(dsparkArgmax, 1, 1, 1, BlockSize, 0, stream, args);
        }

        /// <summary>Per-block-position acceptance probability from the
        /// confidence head.</summary>
        public void DsparkConf(Tensor x, Tensor w1Rows, IntPtr proj, Tensor conf, int e, int rank, int nt, IntPtr stream)
        {
            IntPtr a0 = P(x), a1 = P(w1Rows), a2 = proj, a3 = P(conf);
            int a4 = e, a5 = rank;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(dsparkConf, (uint)nt, 1, 1, BlockSize, 0, stream, args);
        }

        /// <summary>Router score functions of <see cref="MoeSelect"/>.</summary>
        public const int RouterSqrtSoftplus = 0, RouterSigmoid = 1, RouterSoftmax = 2;

        /// <summary>Top-<paramref name="nUsed"/> experts per token and their weights: the selection
        /// ranks score + bias, the weights are the unbiased scores (normalized when
        /// <paramref name="norm"/> is set) times <paramref name="wScale"/>. Hash layers take the
        /// selection from <paramref name="tid2eid"/> instead.</summary>
        public void MoeSelect(Tensor logits, IntPtr bias, IntPtr tid2eid, Tensor tokens, Tensor sel, Tensor wOut,
            int nExpert, int nUsed, int norm, float wScale, int nt, IntPtr stream, int gating = RouterSqrtSoftplus)
        {
            if (gating == RouterSoftmax && nExpert > 512)
                throw new NotSupportedException($"softmax routing over {nExpert} experts (the kernel's warp path takes up to 512)");
            IntPtr a0 = P(logits), a1 = bias, a2 = tid2eid, a3 = P(tokens), a4 = P(sel), a5 = P(wOut);
            int a6 = nExpert, a7 = nUsed, a8 = norm; float a9 = wScale; int a10 = gating;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10 };
            Launch(moeSelect, (uint)nt, 1, 1, BlockSize, (uint)(nExpert * sizeof(float)), stream, args);
        }

        public void MoeCount(Tensor sel, Tensor counts, int s, IntPtr stream)
        {
            IntPtr a0 = P(sel), a1 = P(counts);
            int a2 = s;
            void** args = stackalloc void*[] { &a0, &a1, &a2 };
            Launch(moeCount, CeilDiv(s, BlockSize), 1, 1, BlockSize, 0, stream, args);
        }

        public void MoeScan(Tensor counts, Tensor offsets, Tensor cursors, int nExpert, IntPtr stream)
        {
            IntPtr a0 = P(counts), a1 = P(offsets), a2 = P(cursors);
            int a3 = nExpert;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3 };
            Launch(moeScan, 1, 1, 1, 32, 0, stream, args);
        }

        public void MoeScatter(Tensor sel, Tensor cursors, Tensor rowOfSlot, Tensor slotToken, int s, int nUsed, IntPtr stream)
        {
            IntPtr a0 = P(sel), a1 = P(cursors), a2 = P(rowOfSlot), a3 = P(slotToken);
            int a4 = s, a5 = nUsed;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(moeScatter, CeilDiv(s, BlockSize), 1, 1, BlockSize, 0, stream, args);
        }

        public void MoeGateUp(IntPtr gateW, IntPtr upW, Tensor act, Tensor counts, Tensor offsets, Tensor slotToken,
            Tensor gateOut, Tensor upOut, int wtype, int ff, int inDim, long rowBytes, int nExpert, IntPtr stream,
            IntPtr gateTab = default, IntPtr upTab = default)
        {
            IntPtr a0 = gateW, a1 = upW, a2 = P(act), a3 = P(counts), a4 = P(offsets), a5 = P(slotToken), a6 = P(gateOut), a7 = P(upOut);
            int a8 = wtype, a9 = ff, a10 = inDim; long a11 = rowBytes;
            IntPtr a12 = gateTab, a13 = upTab;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13 };
            Launch(moeGateUp, (uint)nExpert, CeilDiv(ff, 32), 2, BlockSize, 0, stream, args);
        }

        public void MoeDown(IntPtr downW, Tensor act, Tensor counts, Tensor offsets, Tensor downOut,
            int wtype, int e, int ff, long rowBytes, int nExpert, IntPtr stream, IntPtr downTab = default)
        {
            IntPtr a0 = downW, a1 = P(act), a2 = P(counts), a3 = P(offsets), a4 = P(downOut);
            int a5 = wtype, a6 = e, a7 = ff; long a8 = rowBytes;
            IntPtr a9 = downTab;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            Launch(moeDown, (uint)nExpert, CeilDiv(e, 32), 1, BlockSize, 0, stream, args);
        }

        // Staged-weight variants (prefill): each weight row is decoded ONCE into
        // registers and reused across the expert's member tokens, reading the
        // activations from the dense split-q8_1 scratch.
        private const int StageWarps = 8;
        // must match TS_DSV4_STAGE_ROWS in tensorsharp_dsv4_kernels.cu
        private const int StageRows = 2;

        /// <summary>The widths the staged kernels take: whole 256-value blocks, up to 5120 (1..5 sub-blocks
        /// of 32 values per lane; a width that is not a multiple of 1024, like V4.1's 2304-wide experts,
        /// leaves the last sub-blocks to the first lanes).</summary>
        public static bool StagedSupports(int inDim) => inDim > 0 && inDim % 256 == 0 && inDim <= 5120;

        public void MoeGateUpStaged(IntPtr gateW, IntPtr upW, Tensor actQs, Tensor actD,
            Tensor counts, Tensor offsets, Tensor slotToken,
            Tensor gateOut, Tensor upOut, int wtype, int ff, int inDim, long rowBytes, int nExpert, IntPtr stream,
            IntPtr gateTab = default, IntPtr upTab = default)
        {
            IntPtr a0 = gateW, a1 = upW, a2 = P(actQs), a3 = P(actD), a4 = P(counts), a5 = P(offsets), a6 = P(slotToken),
                a7 = P(gateOut), a8 = P(upOut);
            int a9 = wtype, a10 = ff, a11 = inDim; long a12 = rowBytes;
            IntPtr a13 = gateTab, a14 = upTab;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14 };
            // grid.x = row tiles, grid.y = experts (blockIdx.x varies fastest,
            // so one expert's blocks stay co-resident and share its activations)
            uint rowTiles = CeilDiv(ff, StageWarps * StageRows);
            Launch(moeGateUpStaged, rowTiles, (uint)nExpert, 2, BlockSize, 0, stream, args);
        }

        public void MoeDownStaged(IntPtr downW, Tensor actQs, Tensor actD, Tensor counts, Tensor offsets, Tensor downOut,
            int wtype, int e, int ff, long rowBytes, int nExpert, IntPtr stream, IntPtr downTab = default)
        {
            IntPtr a0 = downW, a1 = P(actQs), a2 = P(actD), a3 = P(counts), a4 = P(offsets), a5 = P(downOut);
            int a6 = wtype, a7 = e, a8 = ff; long a9 = rowBytes;
            IntPtr a10 = downTab;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10 };
            uint rowTiles = CeilDiv(e, StageWarps * StageRows);
            Launch(moeDownStaged, rowTiles, (uint)nExpert, 1, BlockSize, 0, stream, args);
        }

        /// <summary>Rows of one expert a tensor-core CTA covers (TS_MOE_MMA_M), and the k-step
        /// the input width must be a multiple of (TS_MOE_MMA_K).</summary>
        public const int MoeMmaRows = 128, MoeMmaK = 128;

        /// <summary>
        /// Grouped expert projection on tensor cores: out[s][r] = W_e[r] . act[token(s)] for every
        /// member slot s of every expert, F16 operands and F32 accumulation. <paramref name="w1"/> /
        /// <paramref name="out1"/> are a second weight/output pair run in the same launch (gate and up),
        /// or zero/null. <paramref name="slotToken"/> null: <paramref name="act"/> is in slot order.
        /// </summary>
        public void MoeMma(IntPtr w0, IntPtr w1, Tensor act, Tensor counts, Tensor offsets, Tensor slotToken,
            Tensor out0, Tensor out1, int wtype, int rows, int inDim, long rowBytes, int nExpert, IntPtr stream,
            IntPtr tab0 = default, IntPtr tab1 = default)
        {
            if (inDim % MoeMmaK != 0)
                throw new ArgumentException($"expert input width {inDim} is not a multiple of {MoeMmaK}");
            IntPtr a0 = w0, a1 = w1, a2 = P(act), a3 = P(counts), a4 = P(offsets), a5 = P(slotToken),
                a6 = P(out0), a7 = P(out1);
            int a8 = wtype, a9 = rows, a10 = inDim; long a11 = rowBytes;
            IntPtr a12 = tab0, a13 = tab1;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13 };
            uint pairs = w1 != IntPtr.Zero || tab1 != IntPtr.Zero ? 2u : 1u;
            Launch(moeMma, CeilDiv(rows, MoeMmaRows), (uint)nExpert, pairs, BlockSize, 0, stream, args);
        }

        /// <summary>Gate/up of every (token, selected expert) slot of <paramref name="nt"/> tokens,
        /// each computed as a lone token's decode step computes it.</summary>
        public void MoeGateUpDecode(IntPtr gateW, IntPtr upW, Tensor act, Tensor sel,
            Tensor gateOut, Tensor upOut, int wtype, int ff, int inDim, long rowBytes, int nt, int nUsed, IntPtr stream,
            IntPtr gateTab = default, IntPtr upTab = default)
            => MoeGateUpDecode(gateW, upW, P(act), sel, gateOut, upOut, wtype, ff, inDim, rowBytes, nt, nUsed, stream, gateTab, upTab);

        /// <summary><see cref="MoeGateUpDecode(IntPtr, IntPtr, Tensor, Tensor, Tensor, Tensor, int, int, int, long, int, int, IntPtr, IntPtr, IntPtr)"/>
        /// of q8_1 activation rows at <paramref name="act"/>.</summary>
        public void MoeGateUpDecode(IntPtr gateW, IntPtr upW, IntPtr act, Tensor sel,
            Tensor gateOut, Tensor upOut, int wtype, int ff, int inDim, long rowBytes, int nt, int nUsed, IntPtr stream,
            IntPtr gateTab = default, IntPtr upTab = default)
        {
            IntPtr a0 = gateW, a1 = upW, a2 = act, a3 = P(sel), a4 = P(gateOut), a5 = P(upOut);
            int a6 = wtype, a7 = ff, a8 = inDim; long a9 = rowBytes; int a10 = nUsed;
            IntPtr a11 = gateTab, a12 = upTab;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12 };
            // Sixteen lanes a row, two rows a warp: a decode step's ten Q2_0 experts 22.7 -> 4.4 us on the RTX 5090.
            if (moeGateUpDecodeHalf != IntPtr.Zero && _moeGateUpHalf)
                Launch(moeGateUpDecodeHalf, (uint)(nt * nUsed), CeilDiv(ff, 16), 2, BlockSize, 0, stream, args);
            else
                Launch(moeGateUpDecode, (uint)(nt * nUsed), CeilDiv(ff, 32), 2, BlockSize, 0, stream, args);
        }

        /// <summary>Down projection of <paramref name="slots"/> (token, selected expert) slots, one
        /// activation row each.</summary>
        public void MoeDownDecode(IntPtr downW, Tensor act, Tensor sel, Tensor downOut,
            int wtype, int e, int ff, long rowBytes, int slots, IntPtr stream, IntPtr downTab = default)
        {
            IntPtr a0 = downW, a1 = P(act), a2 = P(sel), a3 = P(downOut);
            int a4 = wtype, a5 = e, a6 = ff; long a7 = rowBytes;
            IntPtr a8 = downTab;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
            // Short rows (Qwen3.8-Flash-Next's 640-wide down projection): eight lanes a row.
            bool shortRows = moeDownDecodeShort != IntPtr.Zero && ff <= MoeDownShortMaxFf && _moeDownShort;
            Launch(shortRows ? moeDownDecodeShort : moeDownDecode, (uint)slots, CeilDiv(e, 32), 1, BlockSize, 0, stream, args);
        }

        /// <summary>Expert widths up to which the down projection takes eight lanes a row.</summary>
        public const int MoeDownShortMaxFf = 1024;
        // TS_MOE_DOWN_SHORT=0: a warp a row at every width (for measuring it; the sums' order differs).
        private static readonly bool _moeDownShort = Environment.GetEnvironmentVariable("TS_MOE_DOWN_SHORT") != "0";

        /// <summary>Whether the module has the Q8_0 decode experts with the dense matvec's row layout
        /// (<see cref="MoeGateUpDecodeQ80Vec"/>, <see cref="MoeDownDecodeQ80Vec"/>).</summary>
        public bool HasQ80VecDecode => moeGateUpDecodeQ80V != IntPtr.Zero && moeDownDecodeQ80V != IntPtr.Zero;

        /// <summary><see cref="MoeGateUpDecode"/> for Q8_0 experts with a warp per weight row and four lanes per
        /// 32-value block (<c>ts_dsv4_moe_gateup_decode_q80v_f32</c>): its own arithmetic, the same for every
        /// slot whatever the row count, from q8_1 activation rows at <paramref name="act"/>; an expert's rows follow
        /// its residency table when one is given, as the shared decode kernels' do. Requires
        /// <paramref name="inDim"/> a multiple of 32 and <see cref="HasQ80VecDecode"/>.</summary>
        public void MoeGateUpDecodeQ80Vec(IntPtr gateW, IntPtr upW, IntPtr act, Tensor sel,
            Tensor gateOut, Tensor upOut, int ff, int inDim, long rowBytes, int nt, int nUsed, IntPtr stream,
            IntPtr gateTab = default, IntPtr upTab = default)
        {
            IntPtr a0 = gateW, a1 = upW, a2 = act, a3 = P(sel), a4 = P(gateOut), a5 = P(upOut);
            int a6 = ff, a7 = inDim; long a8 = rowBytes; int a9 = nUsed;
            IntPtr a10 = gateTab, a11 = upTab;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11 };
            // From 32 slots a warp takes a slot's gate row and up row together (the same sums; fewer activation loads).
            int slots = nt * nUsed;
            if (moeGateUpDecodeQ80VPair != IntPtr.Zero && slots >= MoeGateUpQ80PairMinSlots && _moeGateUpQ80Pair)
                Launch(moeGateUpDecodeQ80VPair, (uint)slots, CeilDiv(ff, 8), 1, BlockSize, 0, stream, args);
            else
                Launch(moeGateUpDecodeQ80V, (uint)slots, CeilDiv(ff, 8), 2, BlockSize, 0, stream, args);
        }

        /// <summary>Slots from which <see cref="MoeGateUpDecodeQ80Vec"/> computes a slot's gate and up rows in one warp.</summary>
        public const int MoeGateUpQ80PairMinSlots = 32;
        // TS_MOE_GATEUP_Q80_PAIR=0: gate and up rows in separate warps at every slot count (the same sums; for measuring it).
        private static readonly bool _moeGateUpQ80Pair = Environment.GetEnvironmentVariable("TS_MOE_GATEUP_Q80_PAIR") != "0";

        /// <summary><see cref="MoeDownDecode"/> for Q8_0 experts, as <see cref="MoeGateUpDecodeQ80Vec"/> reads a row.</summary>
        public void MoeDownDecodeQ80Vec(IntPtr downW, Tensor act, Tensor sel, Tensor downOut,
            int e, int ff, long rowBytes, int slots, IntPtr stream, IntPtr downTab = default)
        {
            IntPtr a0 = downW, a1 = P(act), a2 = P(sel), a3 = P(downOut);
            int a4 = e, a5 = ff; long a6 = rowBytes;
            IntPtr a7 = downTab;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            Launch(moeDownDecodeQ80V, (uint)slots, CeilDiv(e, 8), 1, BlockSize, 0, stream, args);
        }

        /// <summary>Whether <see cref="DenseQ81Rows"/> decodes this weight layout at this input width:
        /// the formats the warp-per-row dot handles, at widths of whole decode units (32-value blocks,
        /// 64-value Q2_0 blocks, 256-value super-blocks).</summary>
        public static bool DenseQ81RowsSupports(int wtype, int inDim)
            => inDim > 0 && (wtype switch
            {
                // 32-value blocks, decoded one block at a time: Q4_0, Q5_0, Q8_0, IQ4_NL, MXFP4. ISTA's Flash Next
                // keeps a 640-wide shared-expert down projection in IQ4_NL, which a 256 rule sent to the shared
                // matmul route: no decode step of that file was captured.
                2 or 6 or 8 or 20 or 39 => inDim % 32 == 0,
                42 => inDim % 64 == 0,       // Q2_0
                // 256-value super-blocks.
                10 or 11 or 12 or 13 or 14 or 16 or 17 or 18 or 21 or 23 => inDim % 256 == 0,
                _ => false,
            });

        /// <summary>Dense projections up to this many output rows (and inputs at least this long) run split-K.</summary>
        private const int SplitKMaxOutDim = 2048, SplitKMinInDim = 1024;

        /// <summary>A split-K block: a warp per 32 decode units, at most 8 warps.</summary>
        private static int SplitKThreads(int units) => Math.Clamp((units + 31) / 32 * 32, 32, 256);

        /// <summary>out[rows, outDim] = x[rows, inDim] x bf16(w)^T for up to 16 rows of F32 activations (inDim even);
        /// capturable. Needs <see cref="DecodesLegacyQuants"/>.</summary>
        public void DenseBf16Rows(IntPtr w, IntPtr x, IntPtr output, int outDim, int inDim, int rows, IntPtr stream)
        {
            IntPtr a0 = w, a1 = x, a2 = output; int a3 = outDim, a4 = inDim, a5 = rows;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            if (outDim <= SplitKMaxOutDim && inDim >= SplitKMinInDim)
            {
                Launch(denseBf16RowsSplitK, (uint)outDim, 1, 1, SplitKThreads(inDim / 8), 0, stream, args);
                return;
            }
            Launch(denseBf16Rows, CeilDiv((long)outDim * 32, BlockSize), 1, 1, BlockSize, 0, stream, args);
        }

        /// <summary>out[rows, outDim] = act[rows, inDim] x w^T for an input already quantized to q8_1,
        /// one warp per output row. Reads only the buffers passed, so it can be captured.</summary>
        public void DenseQ81Rows(IntPtr w, IntPtr act, IntPtr output, int wtype, int outDim, int inDim, long rowBytes,
            int rows, IntPtr stream)
        {
            IntPtr a0 = w, a1 = act, a2 = output;
            int a3 = wtype, a4 = outDim, a5 = inDim; long a6 = rowBytes; int a7 = rows;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            // The formats the decode-once kernels read: each unit decoded once and dotted against every row, W warps an
            // output row by shape. A 4-row window's 2560 -> 10240 IQ4_XS projection 33 -> 11.5 us on the RTX 5090.
            if (rows <= DenseMultiMaxRows && DenseFastReads(wtype, inDim))
            {
                int wi = DenseFastWarps(outDim);
                int ri = rows <= 1 ? 0 : rows <= 2 ? 1 : rows <= 4 ? 2 : 3;
                int perBlock = 8 / wi;
                Launch(denseFast[ri, wi == 1 ? 0 : wi == 2 ? 1 : 2], (uint)((outDim + perBlock - 1) / perBlock), 1, 1, BlockSize, 0, stream, args);
                return;
            }
            // Few output rows of a long input (a 10240 -> 320 hyper-connection projection): split each row's units
            // over a block's warps, or 32 rows a block would leave most of the GPU idle.
            if (denseQ81RowsSplitK != IntPtr.Zero && outDim <= SplitKMaxOutDim && inDim >= SplitKMinInDim)
            {
                Launch(denseQ81RowsSplitK, (uint)outDim, (uint)rows, 1, SplitKThreads(inDim / 32), 0, stream, args);
                return;
            }
            // A verify window or a batched step: every row in one block, a warp per output row, so each weight row is read
            // from memory once (a block per row read it once per row: a 4-row window's LM head took 4x one row's time).
            // Bit for bit the one-row kernel's sums. (A split-K grid of few outputs keeps a block per row: one block
            // looping over the rows left the GPU too few blocks, and the shared expert slowed down.)
            if (rows > 1 && rows <= DenseMultiMaxRows && denseQ81RowsMulti != IntPtr.Zero && _denseMulti)
                Launch(denseQ81RowsMulti, CeilDiv(outDim, 8), 1, 1, BlockSize, 0, stream, args);
            else
                Launch(denseQ81Rows, CeilDiv(outDim, 32), (uint)rows, 1, BlockSize, 0, stream, args);
        }

        // ---- expert staging (expert residency): see the kernels' comment in tensorsharp_dsv4_kernels.cu ----

        /// <summary>Staging slots for the distinct selected experts of <paramref name="slots"/> slots that live in
        /// host memory; their table entries move to the staging copies.</summary>
        public void MoeStagePlan(Tensor sel, int slots, in CudaExpertStageLayer layer, CudaExpertStage stage, IntPtr stream)
        {
            IntPtr a0 = P(sel); int a1 = slots;
            IntPtr a2 = layer.IsHost, a3 = stage.StageOf, a4 = stage.StageList, a5 = stage.StageCount;
            IntPtr a6 = layer.TabGate, a7 = layer.TabUp, a8 = layer.TabDown, a9 = stage.Staging;
            long a10 = stage.Stride, a11 = layer.OffUp, a12 = layer.OffDown; int a13 = stage.Capacity;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13 };
            Launch(moeStagePlan, CeilDiv(slots, BlockSize), 1, 1, BlockSize, 0, stream, args);
        }

        /// <summary>Copies at most <paramref name="maxStaged"/> planned experts into staging.</summary>
        public void MoeStageCopy(in CudaExpertStageLayer layer, CudaExpertStage stage, int maxStaged, IntPtr stream)
        {
            const int Chunks = 16;
            IntPtr a0 = stage.StageList, a1 = stage.StageCount, a2 = layer.SrcTabGate, a3 = layer.SrcTabUp, a4 = layer.SrcTabDown;
            long a5 = layer.BytesGate, a6 = layer.BytesUp, a7 = layer.BytesDown;
            IntPtr a8 = stage.Staging; long a9 = stage.Stride, a10 = layer.OffUp, a11 = layer.OffDown;
            int a12 = stage.Capacity, a13 = Chunks;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13 };
            Launch(moeStageCopy, (uint)(Math.Max(1, Math.Min(maxStaged, stage.Capacity)) * Chunks), 3, 1, BlockSize, 0, stream, args);
        }

        /// <summary>
        /// The plan's experts that a staging kept across calls does not hold yet: <see cref="CudaExpertStage.StageFrom"/>
        /// takes the planned count first (stream-ordered), then the plan adds this call's new experts and
        /// <see cref="MoeStageCopyNew"/> copies only those.
        /// </summary>
        public void MoeStageMarkFrom(CudaExpertStage stage, IntPtr stream)
            => CudaDriverApi.cuMemcpyDtoDAsync(stage.StageFrom, stage.StageCount, new UIntPtr(4), stream).ThrowOnError();

        /// <summary>Copies the experts planned since <see cref="MoeStageMarkFrom"/>.</summary>
        public void MoeStageCopyNew(in CudaExpertStageLayer layer, CudaExpertStage stage, int maxNew, IntPtr stream)
        {
            const int Chunks = 16;
            IntPtr a0 = stage.StageList, a1 = stage.StageCount, a2 = stage.StageFrom, a3 = layer.SrcTabGate, a4 = layer.SrcTabUp,
                a5 = layer.SrcTabDown;
            long a6 = layer.BytesGate, a7 = layer.BytesUp, a8 = layer.BytesDown;
            IntPtr a9 = stage.Staging; long a10 = stage.Stride, a11 = layer.OffUp, a12 = layer.OffDown;
            int a13 = stage.Capacity, a14 = Chunks;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14 };
            Launch(moeStageCopyFrom, (uint)(Math.Max(1, Math.Min(maxNew, stage.Capacity)) * Chunks), 3, 1, BlockSize, 0, stream, args);
        }

        /// <summary>Points the staged experts' table entries back at their resting addresses and clears the plan.</summary>
        public void MoeUnstage(in CudaExpertStageLayer layer, CudaExpertStage stage, IntPtr stream)
        {
            IntPtr a0 = stage.StageList, a1 = stage.StageCount, a2 = stage.StageOf;
            IntPtr a3 = layer.TabGate, a4 = layer.TabUp, a5 = layer.TabDown, a6 = layer.RestTabGate, a7 = layer.RestTabUp, a8 = layer.RestTabDown;
            int a9 = stage.Capacity;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            Launch(moeUnstage, 1, 1, 1, BlockSize, 0, stream, args);
        }

        // ---- host experts (Strata's doorbell): see the kernels' comment in tensorsharp_dsv4_kernels.cu ----

        /// <summary>Publishes a layer's FFN input (when a selected expert has no slot) and its selection into the
        /// host-mapped mailbox, then rings the doorbell.</summary>
        /// <summary>Whether the module rings the doorbell only for layers with a host expert (an older one rings it for every
        /// layer, and the GPU waits for the host's echo each time).</summary>
        public bool HasDoorbellSkip => doorbellPublishSkip != IntPtr.Zero;

        /// <param name="missDev">With <see cref="HasDoorbellSkip"/>: a device int that records whether the layer rang the
        /// host; <see cref="MoeDoorbellWaitAdd"/> then reads it. Zero rings for every layer.</param>
        /// <param name="counts">With <paramref name="missDev"/>: a device [layers, nExpert] table each selection is counted
        /// into (adaptive residency), or zero.</param>
        /// <summary>Whether <see cref="MoeDoorbellPublish"/> takes a route-tail rank (with a miss word).</summary>
        public bool HasDoorbellTail => doorbellPublishTail != IntPtr.Zero;

        /// <summary>Whether the module splits a decode step's missed experts between the GPU and the host.</summary>
        public bool HasHybridMisses => doorbellPublishHybrid != IntPtr.Zero;

        /// <summary><see cref="MoeDoorbellPublish"/> with the route-tail rule that first plans the GPU's share of the missed
        /// experts into <paramref name="stage"/> - the first <see cref="CudaExpertStage.Capacity"/> distinct ones in slot
        /// order, or with <paramref name="odd"/> the odd-numbered ones - and leaves them out of the host's: their table entries
        /// point at the staging copies, which <see cref="MoeStageCopy"/> fills and <see cref="MoeUnstage"/> restores.</summary>
        public void MoeDoorbellPublishHybrid(Tensor x, Tensor sel, Tensor selW, in CudaExpertStageLayer layerStage, int n, int slots,
            int layer, CudaHostExpertMailbox box, IntPtr stream, IntPtr missDev, IntPtr counts, int nExpert, int nUsed, int tailRank,
            CudaExpertStage stage, bool odd)
        {
            IntPtr a0 = P(x), a1 = P(sel), a2 = P(selW), a3 = layerStage.IsHost; int a4 = n, a5 = slots, a6 = layer;
            IntPtr a7 = box.XDev, a8 = box.IdsDev, a9 = box.WDev, a10 = box.HeaderDev, a11 = box.SeqDev, a12 = missDev, a13 = counts;
            int a14 = nExpert, a15 = nUsed, a16 = tailRank;
            IntPtr a17 = stage.StageOf, a18 = stage.StageList, a19 = stage.StageCount;
            IntPtr a20 = layerStage.TabGate, a21 = layerStage.TabUp, a22 = layerStage.TabDown, a23 = stage.Staging;
            long a24 = stage.Stride, a25 = layerStage.OffUp, a26 = layerStage.OffDown; int a27 = stage.Capacity, a28 = odd ? 1 : 0;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15,
                &a16, &a17, &a18, &a19, &a20, &a21, &a22, &a23, &a24, &a25, &a26, &a27, &a28 };
            Launch(doorbellPublishHybrid, 1, 1, 1, 1024, 0, stream, args);
        }

        /// <param name="tailRank">With <see cref="HasDoorbellTail"/> and a miss word: host experts every row routes at this
        /// rank or lower ring no doorbell (the route-tail skip); 0 rings for every host expert.</param>
        public void MoeDoorbellPublish(Tensor x, Tensor sel, Tensor selW, IntPtr isHost, int n, int slots, int layer,
            CudaHostExpertMailbox box, IntPtr stream, IntPtr missDev = default, IntPtr counts = default, int nExpert = 0,
            int nUsed = 0, int tailRank = 0)
        {
            IntPtr a0 = P(x), a1 = P(sel), a2 = P(selW), a3 = isHost; int a4 = n, a5 = slots, a6 = layer;
            IntPtr a7 = box.XDev, a8 = box.IdsDev, a9 = box.WDev, a10 = box.HeaderDev, a11 = box.SeqDev, a12 = missDev, a13 = counts;
            int a14 = nExpert, a15 = nUsed, a16 = tailRank;
            if (tailRank > 0 && nUsed > 0 && missDev != IntPtr.Zero && HasDoorbellTail)
            {
                void** tail = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16 };
                Launch(doorbellPublishTail, 1, 1, 1, 1024, 0, stream, tail);
                return;
            }
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14 };
            Launch(missDev != IntPtr.Zero ? doorbellPublishSkip : doorbellPublish, 1, 1, 1, 1024, 0, stream, args);
        }

        /// <summary>Waits for the host's answer to the last doorbell and adds its sum into <paramref name="dst"/>.</summary>
        public void MoeDoorbellWaitAdd(CudaHostExpertMailbox box, Tensor dst, int n, IntPtr stream, IntPtr missDev = default)
        {
            IntPtr a0 = box.FlagDev, a1 = box.SeqDev, a2 = box.CountDev, a3 = box.OutDev, a4 = P(dst); int a5 = n; IntPtr a6 = missDev;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
            Launch(missDev != IntPtr.Zero ? doorbellWaitAddSkip : doorbellWaitAdd, Math.Min(16u, CeilDiv(n, BlockSize)), 1, 1, BlockSize, 0,
                stream, args);
        }

        public void MoeScatterAdd(Tensor downOut, Tensor rowOfSlot, Tensor wSel, Tensor shexp, Tensor ffnOut,
            int nt, int nUsed, int e, IntPtr stream)
        {
            IntPtr a0 = P(downOut), a1 = P(rowOfSlot), a2 = P(wSel), a3 = P(shexp), a4 = P(ffnOut);
            int a5 = nt, a6 = nUsed, a7 = e;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            Launch(moeScatterAdd, CeilDiv(e, BlockSize), (uint)nt, 1, BlockSize, 0, stream, args);
        }

        // -------------------------------------------------------------------
        // V4.1
        // -------------------------------------------------------------------

        /// <summary>Query RoPE (no per-head norm) and the shared K row: norm,
        /// RoPE, trained FP8 quantization, F16 commit into the sliding ring.</summary>
        public void V41AttnPrep(Tensor q, Tensor kvRaw, IntPtr kvNormW, IntPtr ropeTab, IntPtr ring,
            int p0, int ringRows, int nh, int hd, int nRot, float eps, int nt, IntPtr stream)
        {
            IntPtr a0 = P(q), a1 = P(kvRaw), a2 = kvNormW, a3 = ropeTab, a4 = ring;
            int a5 = p0, a6 = ringRows, a7 = nh, a8 = hd, a9 = nRot; float a10 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10 };
            Launch(v41AttnPrep, (uint)(nh + 1), (uint)nt, 1, BlockSize, (uint)(hd * sizeof(float)), stream, args);
        }

        /// <summary>One normalized compressed latent per complete block. The state
        /// ring holds <paramref name="stateSize"/> rows (see <see cref="V41Persist"/>).</summary>
        public void V41Compress(Tensor stKv, Tensor stScore, IntPtr histKv, IntPtr histScore, IntPtr normW,
            Tensor latent, long firstBoundary, int nBlocks, int p0, int ratio, int stateSize, int hd, float eps, IntPtr stream)
        {
            IntPtr a0 = P(stKv), a1 = P(stScore), a2 = histKv, a3 = histScore, a4 = normW, a5 = P(latent);
            long a6 = firstBoundary;
            int a7 = p0, a8 = ratio, a9 = hd, a11 = stateSize; float a10 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11 };
            Launch(v41Compress, (uint)nBlocks, 1, 1, BlockSize, (uint)(hd * sizeof(float)), stream, args);
        }

        /// <summary>RoPE at the block-start position, quantize, commit F16.</summary>
        public void V41Commit(Tensor rows, IntPtr ropeTab, IntPtr cache,
            long firstBoundary, int nBlocks, int ratio, int dim, int nRot, int mode, IntPtr stream)
        {
            IntPtr a0 = P(rows), a1 = ropeTab, a2 = cache;
            long a3 = firstBoundary;
            int a4 = ratio, a5 = dim, a6 = nRot, a7 = mode;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            Launch(v41Commit, (uint)nBlocks, 1, 1, BlockSize, (uint)(dim * sizeof(float)), stream, args);
        }

        /// <summary>Keeps the last <paramref name="stateSize"/> rows at pos % stateSize: the
        /// compression ratio plus the drafter's block, so a rejected speculative tail never
        /// overwrites a row an accepted position still needs.</summary>
        public void V41Persist(Tensor stKv, Tensor stScore, IntPtr histKv, IntPtr histScore,
            int p0, int nt, int stateSize, int hd, IntPtr stream)
        {
            IntPtr a0 = P(stKv), a1 = P(stScore), a2 = histKv, a3 = histScore;
            int a4 = p0, a5 = nt, a6 = stateSize, a7 = hd;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            Launch(v41Persist, CeilDiv(hd, BlockSize), (uint)Math.Min(nt, stateSize), 1, BlockSize, 0, stream, args);
        }

        public void V41IdxPrep(Tensor iq, Tensor iw, IntPtr ropeTab, int p0, int ih, int id, int nRot,
            float iwScale, int nt, IntPtr stream)
        {
            IntPtr a0 = P(iq), a1 = P(iw), a2 = ropeTab;
            int a3 = p0, a4 = ih, a5 = id, a6 = nRot; float a7 = iwScale;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            Launch(v41IdxPrep, (uint)ih, (uint)nt, 1, BlockSize, (uint)(id * sizeof(float)), stream, args);
        }

        public void V41IdxScores(Tensor iq, Tensor iw, IntPtr lidK, IntPtr candidates, Tensor scores,
            int p0, int ratio, int ih, int id, int rows, int maxVis, int nt, IntPtr stream)
        {
            IntPtr a0 = P(iq), a1 = P(iw), a2 = lidK, a3 = candidates, a4 = P(scores);
            int a5 = p0, a6 = ratio, a7 = ih, a8 = id, a9 = rows, a10 = maxVis;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10 };
            const int RowsPerBlock = 8;   // one warp per row
            Launch2D(v41IdxScores, CeilDiv(maxVis, RowsPerBlock), (uint)nt, 32, RowsPerBlock, 0, stream, args);
        }

        public void V41Candidate(Tensor scores, IntPtr mask, int p0, int ratio, int rows,
            int blockLen, int topk, int nt, IntPtr stream)
        {
            IntPtr a0 = P(scores), a1 = mask;
            int a2 = p0, a3 = ratio, a4 = rows, a5 = blockLen, a6 = topk;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
            Launch(v41Candidate, (uint)nt, 1, 1, BlockSize, 0, stream, args);
        }

        public void V41EngramGate(Tensor xs, Tensor kv, IntPtr qw, IntPtr kw, int hc, int e, float eps,
            int nt, IntPtr stream)
        {
            IntPtr a0 = P(xs), a1 = P(kv), a2 = qw, a3 = kw;
            int a4 = e; float a5 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(v41EngramGate, (uint)hc, (uint)nt, 1, BlockSize, 0, stream, args);
        }

        /// <summary>Rows at or under which <see cref="Gemv"/> beats a cuBLAS GEMM for this engine's small
        /// F32 weights.</summary>
        public const int GemvMaxRows = 16;

        /// <summary>Floats of partial-sum scratch <see cref="Gemv"/> needs: outputs under 256 split
        /// the input so that outputs x splits stays under 512.</summary>
        public const int GemvScratchFloats = GemvMaxRows * 512;

        /// <summary>y[rows, outDim] = x[rows, inDim] x w^T for an F32 weight and up to
        /// <see cref="GemvMaxRows"/> rows (inDim % 4 == 0). A narrow output splits the input across
        /// CTAs and combines the partial sums through <paramref name="scratch"/>
        /// (<see cref="GemvScratchFloats"/> floats).</summary>
        /// <summary><see cref="Gemv"/> of a wide output (256 or more) over BF16 weights: the same sums, bit for bit, as
        /// the F32 weights they widen to, at half the bytes. Requires <see cref="HasGemvBf16"/>.</summary>
        public void GemvBf16(IntPtr x, IntPtr w, IntPtr y, int inDim, int outDim, int rows, IntPtr stream)
        {
            if ((inDim & 3) != 0 || rows > GemvMaxRows || outDim < 256)
                throw new ArgumentException($"the BF16 GEMV takes inDim % 4 == 0, at most {GemvMaxRows} rows and 256 or more outputs");
            IntPtr w0 = x, w1 = w, w2 = y;
            int w3 = inDim, w4 = outDim, w5 = rows;
            void** wargs = stackalloc void*[] { &w0, &w1, &w2, &w3, &w4, &w5 };
            if (gemvBlockBf16 != null && _gemvBlockEnv)
                Launch(gemvBlockBf16[RowBoundIndex(rows)], (uint)outDim, 1, 1, GemvBlockThreads, 0, stream, wargs);
            else
                Launch(gemvWarpBf16[RowBoundIndex(rows)], (uint)((outDim + 3) / 4), 1, 1, 128, 0, stream, wargs);
        }

        public void Gemv(IntPtr x, IntPtr w, IntPtr y, int inDim, int outDim, int rows, IntPtr stream, IntPtr scratch,
            int splitsOverride = 0)
        {
            if ((inDim & 3) != 0 || rows > GemvMaxRows)
                throw new ArgumentException($"GEMV takes inDim % 4 == 0 and at most {GemvMaxRows} rows (got {inDim}, {rows})");
            if (splitsOverride == 0 && outDim >= 256)
            {
                // Wide outputs (a router): a warp per output, four per CTA.
                IntPtr w0 = x, w1 = w, w2 = y;
                int w3 = inDim, w4 = outDim, w5 = rows;
                void** wargs = stackalloc void*[] { &w0, &w1, &w2, &w3, &w4, &w5 };
                // A block per output (the sums' order as the BF16 form's, so the two give the same bits).
                if (gemvBlock != null && _gemvBlockEnv)
                    Launch(gemvBlock[RowBoundIndex(rows)], (uint)outDim, 1, 1, GemvBlockThreads, 0, stream, wargs);
                else
                    Launch(gemvWarp[RowBoundIndex(rows)], (uint)((outDim + 3) / 4), 1, 1, 128, 0, stream, wargs);
                return;
            }
            int splits = splitsOverride > 0 ? splitsOverride : Math.Min(16, (256 + outDim - 1) / outDim);
            splits = Math.Max(1, Math.Min(splits, inDim / 4 / 128));
            IntPtr a0 = x, a1 = w, a2 = y, a3 = scratch;
            int a4 = inDim, a5 = outDim, a6 = rows, a7 = splits;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            Launch(gemv[RowBoundIndex(rows)], (uint)outDim, (uint)splits, 1, 128, 0, stream, args);
            if (splits > 1)
            {
                int count = rows * outDim;
                IntPtr c0 = scratch, c1 = y;
                int c2 = count, c3 = splits;
                void** cargs = stackalloc void*[] { &c0, &c1, &c2, &c3 };
                Launch(gemvCombine, CeilDiv(count, BlockSize), 1, 1, BlockSize, 0, stream, cargs);
            }
        }

        /// <summary>Dequantize the selected rows of a device-resident Engram table, [nt, columns, headDim].</summary>
        public void V41EngramGather(IntPtr table, int wtype, long rowBytes, Tensor rows, Tensor output,
            int columns, int headDim, int nt, IntPtr stream)
        {
            IntPtr a0 = table, a3 = P(rows), a4 = P(output);
            int a1 = wtype, a5 = headDim; long a2 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(v41EngramGather, (uint)((long)nt * columns), 1, 1, 128, 0, stream, args);
        }

        public void Dispose()
        {
            module.Dispose();
        }
    }
}
