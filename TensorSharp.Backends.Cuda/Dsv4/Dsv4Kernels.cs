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
        private readonly IntPtr moeGateUpDecodeQ80V, moeDownDecodeQ80V;
        private readonly IntPtr denseQ81Rows;
        private readonly IntPtr moeScatterAdd;
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
            moeGateUpDecodeQ80V = module.GetFunction("ts_dsv4_moe_gateup_decode_q80v_f32");
            moeDownDecodeQ80V = module.GetFunction("ts_dsv4_moe_down_decode_q80v_f32");
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
            Tensor gateOut, Tensor upOut, int wtype, int ff, int inDim, long rowBytes, int nExpert, IntPtr stream)
        {
            IntPtr a0 = gateW, a1 = upW, a2 = P(act), a3 = P(counts), a4 = P(offsets), a5 = P(slotToken), a6 = P(gateOut), a7 = P(upOut);
            int a8 = wtype, a9 = ff, a10 = inDim; long a11 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11 };
            Launch(moeGateUp, (uint)nExpert, CeilDiv(ff, 32), 2, BlockSize, 0, stream, args);
        }

        public void MoeDown(IntPtr downW, Tensor act, Tensor counts, Tensor offsets, Tensor downOut,
            int wtype, int e, int ff, long rowBytes, int nExpert, IntPtr stream)
        {
            IntPtr a0 = downW, a1 = P(act), a2 = P(counts), a3 = P(offsets), a4 = P(downOut);
            int a5 = wtype, a6 = e, a7 = ff; long a8 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
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
            Tensor gateOut, Tensor upOut, int wtype, int ff, int inDim, long rowBytes, int nExpert, IntPtr stream)
        {
            IntPtr a0 = gateW, a1 = upW, a2 = P(actQs), a3 = P(actD), a4 = P(counts), a5 = P(offsets), a6 = P(slotToken),
                a7 = P(gateOut), a8 = P(upOut);
            int a9 = wtype, a10 = ff, a11 = inDim; long a12 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12 };
            // grid.x = row tiles, grid.y = experts (blockIdx.x varies fastest,
            // so one expert's blocks stay co-resident and share its activations)
            uint rowTiles = CeilDiv(ff, StageWarps * StageRows);
            Launch(moeGateUpStaged, rowTiles, (uint)nExpert, 2, BlockSize, 0, stream, args);
        }

        public void MoeDownStaged(IntPtr downW, Tensor actQs, Tensor actD, Tensor counts, Tensor offsets, Tensor downOut,
            int wtype, int e, int ff, long rowBytes, int nExpert, IntPtr stream)
        {
            IntPtr a0 = downW, a1 = P(actQs), a2 = P(actD), a3 = P(counts), a4 = P(offsets), a5 = P(downOut);
            int a6 = wtype, a7 = e, a8 = ff; long a9 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
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
            Tensor out0, Tensor out1, int wtype, int rows, int inDim, long rowBytes, int nExpert, IntPtr stream)
        {
            if (inDim % MoeMmaK != 0)
                throw new ArgumentException($"expert input width {inDim} is not a multiple of {MoeMmaK}");
            IntPtr a0 = w0, a1 = w1, a2 = P(act), a3 = P(counts), a4 = P(offsets), a5 = P(slotToken),
                a6 = P(out0), a7 = P(out1);
            int a8 = wtype, a9 = rows, a10 = inDim; long a11 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11 };
            uint pairs = w1 != IntPtr.Zero ? 2u : 1u;
            Launch(moeMma, CeilDiv(rows, MoeMmaRows), (uint)nExpert, pairs, BlockSize, 0, stream, args);
        }

        /// <summary>Gate/up of every (token, selected expert) slot of <paramref name="nt"/> tokens,
        /// each computed as a lone token's decode step computes it.</summary>
        public void MoeGateUpDecode(IntPtr gateW, IntPtr upW, Tensor act, Tensor sel,
            Tensor gateOut, Tensor upOut, int wtype, int ff, int inDim, long rowBytes, int nt, int nUsed, IntPtr stream)
        {
            IntPtr a0 = gateW, a1 = upW, a2 = P(act), a3 = P(sel), a4 = P(gateOut), a5 = P(upOut);
            int a6 = wtype, a7 = ff, a8 = inDim; long a9 = rowBytes; int a10 = nUsed;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10 };
            Launch(moeGateUpDecode, (uint)(nt * nUsed), CeilDiv(ff, 32), 2, BlockSize, 0, stream, args);
        }

        /// <summary>Down projection of <paramref name="slots"/> (token, selected expert) slots, one
        /// activation row each.</summary>
        public void MoeDownDecode(IntPtr downW, Tensor act, Tensor sel, Tensor downOut,
            int wtype, int e, int ff, long rowBytes, int slots, IntPtr stream)
        {
            IntPtr a0 = downW, a1 = P(act), a2 = P(sel), a3 = P(downOut);
            int a4 = wtype, a5 = e, a6 = ff; long a7 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            Launch(moeDownDecode, (uint)slots, CeilDiv(e, 32), 1, BlockSize, 0, stream, args);
        }

        /// <summary><see cref="MoeGateUpDecode"/> for Q8_0 experts with a warp per weight row and four lanes per
        /// 32-value block (<c>ts_dsv4_moe_gateup_decode_q80v_f32</c>): its own arithmetic, the same for every
        /// slot whatever the row count. Requires <paramref name="inDim"/> a multiple of 32.</summary>
        public void MoeGateUpDecodeQ80Vec(IntPtr gateW, IntPtr upW, Tensor act, Tensor sel,
            Tensor gateOut, Tensor upOut, int ff, int inDim, long rowBytes, int nt, int nUsed, IntPtr stream)
        {
            IntPtr a0 = gateW, a1 = upW, a2 = P(act), a3 = P(sel), a4 = P(gateOut), a5 = P(upOut);
            int a6 = ff, a7 = inDim; long a8 = rowBytes; int a9 = nUsed;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            Launch(moeGateUpDecodeQ80V, (uint)(nt * nUsed), CeilDiv(ff, 8), 2, BlockSize, 0, stream, args);
        }

        /// <summary><see cref="MoeDownDecode"/> for Q8_0 experts, as <see cref="MoeGateUpDecodeQ80Vec"/> reads a row.</summary>
        public void MoeDownDecodeQ80Vec(IntPtr downW, Tensor act, Tensor sel, Tensor downOut,
            int e, int ff, long rowBytes, int slots, IntPtr stream)
        {
            IntPtr a0 = downW, a1 = P(act), a2 = P(sel), a3 = P(downOut);
            int a4 = e, a5 = ff; long a6 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
            Launch(moeDownDecodeQ80V, (uint)slots, CeilDiv(e, 8), 1, BlockSize, 0, stream, args);
        }

        /// <summary>Whether <see cref="DenseQ81Rows"/> decodes this weight layout at this input width:
        /// the formats the warp-per-row dot handles, at widths of whole 256-value super-blocks.</summary>
        public static bool DenseQ81RowsSupports(int wtype, int inDim)
            => inDim > 0 && inDim % 256 == 0 && wtype is 8 or 10 or 11 or 12 or 13 or 14 or 16 or 17 or 18 or 20 or 21 or 23 or 39;

        /// <summary>out[rows, outDim] = act[rows, inDim] x w^T for an input already quantized to q8_1,
        /// one warp per output row. Reads only the buffers passed, so it can be captured.</summary>
        public void DenseQ81Rows(IntPtr w, IntPtr act, IntPtr output, int wtype, int outDim, int inDim, long rowBytes,
            int rows, IntPtr stream)
        {
            IntPtr a0 = w, a1 = act, a2 = output;
            int a3 = wtype, a4 = outDim, a5 = inDim; long a6 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
            Launch(denseQ81Rows, CeilDiv(outDim, 32), (uint)rows, 1, BlockSize, 0, stream, args);
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
