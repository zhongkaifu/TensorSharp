// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Launchers for the Qwen3.8-Flash-Next direct-CUDA kernels
// (native/kernels/tensorsharp_q4e_kernels.cu, compiled to tensorsharp_q4e_kernels.ptx). The
// embedding, router and expert kernels are the DeepSeek V4 module's (Dsv4Kernels, CudaMoe) and
// the delta recurrence is the GLM module's (GlmKernels); these are the rest. One instance per
// device, loaded against the context current at construction.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal sealed unsafe class Q4eKernels : IDisposable
    {
        private const int BlockSize = 256;

        /// <summary>Head width of the attention kernels (Q4E_ATTN_HD).</summary>
        public const int AttnHeadDim = 256;

        /// <summary>Head width of the Gated DeltaNet recurrence (Q4E_GDN_HD, the GLM scan's).</summary>
        public const int GdnHeadDim = 128;

        /// <summary>Widest indexer head the QSA kernels take (their shared rows).</summary>
        public const int MaxIndexerDim = 256;

        /// <summary>Hyper-connection streams the kernels take at most (Q4E_HC_MAX).</summary>
        public const int MaxStreams = 4;

        /// <summary>CTAs the block scoring launches at most per row (its warps stride over the rest):
        /// about one full wave on the A40.</summary>
        private const uint MaxScoreBlocks = 512;

        private readonly CudaModule module;
        private readonly IntPtr hcNorm, siluScale, hcCollapse, hcPost, hcPartials, hcApply, siluQ81, hcPostPartials, bf16ToF32;
        // The collapse that also quantizes its output (optional: a module before it quantizes in a launch of its own).
        private readonly IntPtr hcCollapseQ81;

        /// <summary>Whether <see cref="HcCollapseQ81"/> runs here.</summary>
        public bool HasHcCollapseQ81 => hcCollapseQ81 != IntPtr.Zero;

        /// <summary>Whether the module widens BF16 to F32 (<see cref="Bf16ToF32"/>).</summary>
        public bool HasBf16ToF32 => bf16ToF32 != IntPtr.Zero;

        /// <summary>Whether the module carries the fused scatter-and-partials kernel (a committed baseline built before
        /// it does not; the engine then launches the two kernels).</summary>
        public bool HasHcPostPartials => hcPostPartials != IntPtr.Zero;

        /// <summary>Values per mixer slice (Q4E_HC_SLICE) and floats of partials per slice (Q4E_HC_PARTIAL).</summary>
        public const int HcSlice = 256, HcPartialFloats = 1 + MaxStreams;
        private readonly IntPtr gdnPrep, convUpdate;
        private readonly IntPtr attnPrep, attention, attnPrepQ8, attentionQ8;
        // Decode-size attention split over the cells (optional: a module before it runs a block per head and row).
        private readonly IntPtr attentionSplit, attentionSplitQ8;
        // Prompt attention on tensor cores, a block per (row, KV head) (optional; sm_80+ and a module built for it).
        private readonly IntPtr attentionMma, attentionMmaQ8;

        /// <summary>Whether <see cref="AttentionMma"/> runs here: the module carries it compiled for sm_80 or newer and the
        /// device is one (the committed compute_75 baseline compiles its body to nothing).</summary>
        public bool HasAttentionMma { get; }

        /// <summary>Query heads per KV head <see cref="AttentionMma"/> takes (the rows of its 16-row tile).</summary>
        public const int AttnMmaMaxGroup = 16;

        /// <summary>Whether the module stores and reads a Q8_0 KV cache (<c>--kv-cache-dtype q8_0</c>): int8 codes
        /// [nCtx, NKV, HD] beside F16 scales [nCtx, NKV, HD / 32].</summary>
        public bool HasKvQ8 => attnPrepQ8 != IntPtr.Zero && attentionQ8 != IntPtr.Zero;

        /// <summary>Values per Q8_0 KV scale (Q4E_KV_Q8_BLOCK).</summary>
        public const int KvQ8Block = 32;
        private readonly IntPtr qsaStore, qsaPool, qsaQuery, qsaScores, qsaSelect;
        // KV streaming (TS_Q4E_KV_RESIDENT; optional: a module before it keeps every layer's K/V resident).
        private readonly IntPtr attnPrepQ8Paged, attentionQ8Paged, attentionSplitQ8Paged, kvResolve, kvPageCopy, kvInvalidate,
            qsaStoreRing, qsaPoolRing;

        /// <summary>Whether the module streams a Q8_0 attention layer's K/V through a window of VRAM pages
        /// (<see cref="KvResolve"/> and the paged readers and writer), with the QSA raw keys in a ring.</summary>
        public bool HasKvStreaming => attnPrepQ8Paged != IntPtr.Zero && attentionQ8Paged != IntPtr.Zero
            && attentionSplitQ8Paged != IntPtr.Zero && kvResolve != IntPtr.Zero && kvPageCopy != IntPtr.Zero
            && kvInvalidate != IntPtr.Zero && qsaStoreRing != IntPtr.Zero && qsaPoolRing != IntPtr.Zero;

        /// <summary>Cells a streamed layer's residency page holds (Q4E_KV_PAGE).</summary>
        public const int KvPage = 4;

        /// <summary>Control words of a streamed layer's page map (Q4E_KV_CTL): [0] epoch, [1] hand, [2] the last call's
        /// misses, [3] overflowing calls, [4] misses and [5] calls so far.</summary>
        public const int KvCtlInts = 8;
        private readonly IntPtr pleGate, groupNorm, pleConv;
        private readonly IntPtr sharedGate, setStreamRows;
        // Each row's first-max token on the device (optional: a module before it returns logits to the host).
        private readonly IntPtr argmaxRows;
        // A prompt's expert selections into adaptive residency's counts (optional).
        private readonly IntPtr countSelected;
        // The MTP draft chain's per-step pick on the device (optional).
        private readonly IntPtr draftPick;
        // The pick in two stages (optional): a block per slice of the row, then one merging block.
        private readonly IntPtr draftPickPart, draftPickMerge;

        private Q4eKernels(CudaModule module)
        {
            this.module = module;
            hcNorm = module.GetFunction("ts_q4e_hc_norm_f32");
            siluScale = module.GetFunction("ts_q4e_silu_scale_f32");
            hcCollapse = module.GetFunction("ts_q4e_hc_collapse_f32");
            hcPost = module.GetFunction("ts_q4e_hc_post_f32");
            hcPartials = module.GetFunction("ts_q4e_hc_partials_f32");
            hcApply = module.GetFunction("ts_q4e_hc_apply_f32");
            siluQ81 = module.GetFunction("ts_q4e_silu_q81_f32");
            try { hcPostPartials = module.GetFunction("ts_q4e_hc_post_partials_f32"); }
            catch (CudaException ex) when (ex.ErrorCode == 500) { } // CUDA_ERROR_NOT_FOUND
            try { bf16ToF32 = module.GetFunction("ts_q4e_bf16_to_f32"); }
            catch (CudaException ex) when (ex.ErrorCode == 500) { }
            try { hcCollapseQ81 = module.GetFunction("ts_q4e_hc_collapse_q81_f32"); }
            catch (CudaException ex) when (ex.ErrorCode == 500) { }
            gdnPrep = module.GetFunction("ts_q4e_gdn_prep_f32");
            convUpdate = module.GetFunction("ts_q4e_conv_update_f32");
            attnPrep = module.GetFunction("ts_q4e_attn_prep_f32");
            attention = module.GetFunction("ts_q4e_attention_f32");
            try { attnPrepQ8 = module.GetFunction("ts_q4e_attn_prep_q8_f32"); attentionQ8 = module.GetFunction("ts_q4e_attention_q8_f32"); }
            catch (CudaException ex) when (ex.ErrorCode == 500) { attnPrepQ8 = attentionQ8 = IntPtr.Zero; }
            try { attentionSplit = module.GetFunction("ts_q4e_attention_split_f32"); attentionSplitQ8 = module.GetFunction("ts_q4e_attention_split_q8_f32"); }
            catch (CudaException ex) when (ex.ErrorCode == 500) { attentionSplit = attentionSplitQ8 = IntPtr.Zero; }
            try
            {
                attnPrepQ8Paged = module.GetFunction("ts_q4e_attn_prep_q8_paged_f32");
                attentionQ8Paged = module.GetFunction("ts_q4e_attention_q8_paged_f32");
                attentionSplitQ8Paged = module.GetFunction("ts_q4e_attention_split_q8_paged_f32");
                kvResolve = module.GetFunction("ts_q4e_kv_resolve");
                kvPageCopy = module.GetFunction("ts_q4e_kv_page_copy");
                kvInvalidate = module.GetFunction("ts_q4e_kv_invalidate");
                qsaStoreRing = module.GetFunction("ts_q4e_qsa_store_ring_f32");
                qsaPoolRing = module.GetFunction("ts_q4e_qsa_pool_ring_f32");
            }
            catch (CudaException ex) when (ex.ErrorCode == 500)
            {
                attnPrepQ8Paged = attentionQ8Paged = attentionSplitQ8Paged = kvResolve = kvPageCopy = kvInvalidate = IntPtr.Zero;
                qsaStoreRing = qsaPoolRing = IntPtr.Zero;
            }
            qsaStore = module.GetFunction("ts_q4e_qsa_store_f32");
            qsaPool = module.GetFunction("ts_q4e_qsa_pool_f32");
            qsaQuery = module.GetFunction("ts_q4e_qsa_query_f32");
            qsaScores = module.GetFunction("ts_q4e_qsa_scores_f32");
            qsaSelect = module.GetFunction("ts_q4e_qsa_select_i32");
            pleGate = module.GetFunction("ts_q4e_ple_gate_f32");
            groupNorm = module.GetFunction("ts_q4e_group_norm_f32");
            pleConv = module.GetFunction("ts_q4e_ple_conv_f32");
            sharedGate = module.GetFunction("ts_q4e_shared_gate_f32");
            setStreamRows = module.GetFunction("ts_q4e_set_stream_rows_f32");
            try { argmaxRows = module.GetFunction("ts_q4e_argmax_rows_f32"); }
            catch (CudaException ex) when (ex.ErrorCode == 500) { argmaxRows = IntPtr.Zero; }
            try { countSelected = module.GetFunction("ts_q4e_count_selected"); }
            catch (CudaException ex) when (ex.ErrorCode == 500) { countSelected = IntPtr.Zero; }
            try
            {
                attentionMma = module.GetFunction("ts_q4e_attention_mma_f32");
                attentionMmaQ8 = module.GetFunction("ts_q4e_attention_mma_q8_f32");
                CudaDriverApi.cuCtxGetDevice(out int device);
                CudaDriverApi.cuDeviceGetAttribute(out int ccMajor, 75 /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR */, device);
                HasAttentionMma = CudaKernels.ArchGuardedKernelsUsable(module.CompiledArchOf("ts_q4e_attention_mma_q8_f32"), ccMajor);
            }
            catch (CudaException ex) when (ex.ErrorCode == 500) { attentionMma = attentionMmaQ8 = IntPtr.Zero; }
            try { draftPick = module.GetFunction("ts_q4e_draft_pick_f32"); }
            catch (CudaException ex) when (ex.ErrorCode == 500) { draftPick = IntPtr.Zero; }
            try
            {
                draftPickPart = module.GetFunction("ts_q4e_draft_pick_part_f32");
                draftPickMerge = module.GetFunction("ts_q4e_draft_pick_merge_f32");
            }
            catch (CudaException ex) when (ex.ErrorCode == 500) { draftPickPart = draftPickMerge = IntPtr.Zero; }
        }

        public static Q4eKernels Create()
        {
            string path = CudaKernels.LocatePtxPath("tensorsharp_q4e_kernels.ptx");
            if (path == null)
            {
                throw new InvalidOperationException(
                    "tensorsharp_q4e_kernels.ptx could not be located next to the application or under " +
                    "a 'cuda_kernels' folder. Build TensorSharp.Backends.Cuda with a CUDA 13 nvcc on the PATH.");
            }
            return new Q4eKernels(CudaModule.LoadFromFile(path));
        }

        private static void Launch(IntPtr fn, uint gx, uint gy, uint gz, int block, IntPtr stream, void** args)
            => CudaDriverApi.cuLaunchKernel(fn, gx, gy, gz, (uint)block, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();

        private static uint CeilDiv(long n, int d) => (uint)Math.Max(1, (n + d - 1) / d);

        // ---- hyper-connections ----

        /// <summary>Each stream of every row RMS-normed times <paramref name="normW"/> into
        /// <paramref name="xn"/>, and the scatter logits injW . xn into <paramref name="inject"/>
        /// (<paramref name="injW"/> zero: the output mixer, which has none).</summary>
        public void HcNorm(IntPtr xs, IntPtr normW, IntPtr injW, IntPtr xn, IntPtr inject, int rows, int e, int hc,
            float eps, IntPtr stream)
        {
            IntPtr a0 = xs, a1 = normW, a2 = injW, a3 = xn, a4 = inject;
            int a5 = e, a6 = hc; float a7 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            Launch(hcNorm, (uint)rows, 1, 1, BlockSize, stream, args);
        }

        /// <summary>The decode-size mixer's first pass: per 256-value slice, its sum of squares and
        /// scatter-logit partials.</summary>
        public void HcPartials(IntPtr xs, IntPtr normW, IntPtr injW, IntPtr partials, int rows, int e, int hc, IntPtr stream)
        {
            IntPtr a0 = xs, a1 = normW, a2 = injW, a3 = partials;
            int a4 = e, a5 = hc;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(hcPartials, (uint)(hc * e / HcSlice), (uint)rows, 1, HcSlice, stream, args);
        }

        /// <summary>The second pass: the normed streams, their q8_1 blocks, and the scatter logits
        /// (<paramref name="inject"/> zero: the output mixer).</summary>
        public void HcApply(IntPtr xs, IntPtr normW, IntPtr partials, IntPtr xn, IntPtr xq, IntPtr inject, int rows, int e, int hc,
            float eps, IntPtr stream)
        {
            IntPtr a0 = xs, a1 = normW, a2 = partials, a3 = xn, a4 = xq, a5 = inject;
            int a6 = e, a7 = hc; float a8 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
            Launch(hcApply, (uint)(hc * e / HcSlice), (uint)rows, 1, HcSlice, stream, args);
        }

        /// <summary>x = silu(x * scale) over <paramref name="n"/> values (whole 32-value blocks), also
        /// quantized to q8_1 into <paramref name="xq"/>.</summary>
        public void SiluQ81(IntPtr x, IntPtr xq, long n, float scale, IntPtr stream)
        {
            IntPtr a0 = x, a1 = xq; long a2 = n / 32; float a3 = scale;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3 };
            Launch(siluQ81, CeilDiv(n, BlockSize), 1, 1, BlockSize, stream, args);
        }

        /// <summary>x = silu(x * scale) over <paramref name="n"/> values.</summary>
        public void SiluScale(IntPtr x, long n, float scale, IntPtr stream)
        {
            IntPtr a0 = x; long a1 = n; float a2 = scale;
            void** args = stackalloc void*[] { &a0, &a1, &a2 };
            Launch(siluScale, CeilDiv(n, BlockSize), 1, 1, BlockSize, stream, args);
        }

        /// <summary>cur = the stream mean of xn * sigmoid(g).</summary>
        public void HcCollapse(IntPtr xn, IntPtr g, IntPtr cur, int rows, int e, int hc, IntPtr stream)
        {
            IntPtr a0 = xn, a1 = g, a2 = cur;
            int a3 = e, a4 = hc;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(hcCollapse, CeilDiv(e, BlockSize), (uint)rows, 1, BlockSize, stream, args);
        }

        /// <summary><see cref="HcCollapse"/>, its output also quantized to q8_1 (row t at block t * e / 32, as
        /// LaunchQuantizeQ81Rows lays it out and rounds it) into <paramref name="xq"/>. <paramref name="e"/> % 32 == 0.</summary>
        public void HcCollapseQ81(IntPtr xn, IntPtr g, IntPtr cur, IntPtr xq, int rows, int e, int hc, IntPtr stream)
        {
            IntPtr a0 = xn, a1 = g, a2 = cur, a3 = xq;
            int a4 = e, a5 = hc;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(hcCollapseQ81, CeilDiv(e, BlockSize), (uint)rows, 1, BlockSize, stream, args);
        }

        /// <summary>xs[t, c] += out[t] * 2 sigmoid(inject[t, c] / hc).</summary>
        /// <summary><see cref="HcPost"/>, then the next mixer's <see cref="HcPartials"/> over the updated streams, in one
        /// launch (decode-size rows; requires <see cref="HasHcPostPartials"/>).</summary>
        public void HcPostPartials(IntPtr xs, IntPtr output, IntPtr inject, IntPtr normW, IntPtr injW, IntPtr partials,
            int rows, int e, int hc, IntPtr stream)
        {
            IntPtr a0 = xs, a1 = output, a2 = inject, a3 = normW, a4 = injW, a5 = partials;
            int a6 = e, a7 = hc;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            Launch(hcPostPartials, (uint)(hc * e / HcSlice), (uint)rows, 1, HcSlice, stream, args);
        }

        /// <summary>dst[i] = (float)src[i] for <paramref name="n"/> BF16 values.</summary>
        public void Bf16ToF32(IntPtr src, IntPtr dst, long n, IntPtr stream)
        {
            IntPtr a0 = src, a1 = dst; long a2 = n;
            void** args = stackalloc void*[] { &a0, &a1, &a2 };
            Launch(bf16ToF32, (uint)((n + BlockSize - 1) / BlockSize), 1, 1, BlockSize, stream, args);
        }

        public void HcPost(IntPtr xs, IntPtr output, IntPtr inject, int rows, int e, int hc, IntPtr stream)
        {
            IntPtr a0 = xs, a1 = output, a2 = inject;
            int a3 = e, a4 = hc;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(hcPost, CeilDiv(e, BlockSize), (uint)rows, 1, BlockSize, stream, args);
        }

        // ---- Gated DeltaNet ----

        /// <summary>The convolution, norms, decay and beta of <paramref name="nSeq"/> sequences of
        /// <paramref name="rowsPerSeq"/> rows into the GLM scan's scratch layout.</summary>
        public void GdnPrep(IntPtr qkv, IntPtr convStates, IntPtr convW, IntPtr alpha, IntPtr beta, IntPtr dtBias, IntPtr ssmA,
            IntPtr scr, int rowsPerSeq, int nSeq, int kHeads, int vHeads, int dConv, float eps, IntPtr stream)
        {
            IntPtr a0 = qkv, a1 = convStates, a2 = convW, a3 = alpha, a4 = beta, a5 = dtBias, a6 = ssmA, a7 = scr;
            int a8 = rowsPerSeq, a9 = nSeq, a10 = kHeads, a11 = vHeads, a12 = dConv;
            float a13 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13 };
            long warps = (long)rowsPerSeq * nSeq * vHeads;
            // Two warps a block: a decode step's 48 warps spread over 24 SMs instead of 6 (5.26 -> 4.74 us a 4-row window on
            // the RTX 5090; each warp's arithmetic is the same).
            const int GdnPrepBlock = 64;
            Launch(gdnPrep, CeilDiv(warps * 32, GdnPrepBlock), 1, 1, GdnPrepBlock, stream, args);
        }

        /// <summary>Keep each sequence's last <paramref name="hist"/> rows of [state; x] as its state.</summary>
        public void ConvUpdate(IntPtr x, IntPtr states, int rowsPerSeq, int nSeq, int channels, int hist, IntPtr stream)
        {
            IntPtr a0 = x, a1 = states;
            int a2 = rowsPerSeq, a3 = nSeq, a4 = channels, a5 = hist;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(convUpdate, CeilDiv((long)nSeq * channels, BlockSize), 1, 1, BlockSize, stream, args);
        }

        // ---- attention ----

        /// <summary>Norms and rotary of the query and key heads, the gates split out, K and V
        /// stored into the caches: F16, or Q8_0 codes with their scales when <paramref name="kScale"/> is set.</summary>
        public void AttnPrep(IntPtr qg, IntPtr k, IntPtr v, IntPtr qNorm, IntPtr kNorm, IntPtr q, IntPtr gate,
            IntPtr kCache, IntPtr vCache, IntPtr positions, int p0, int nHead, int nKvHead, int nRot,
            float ropeBase, float freqScale, float eps, int rows, IntPtr stream, IntPtr kScale = default, IntPtr vScale = default)
        {
            IntPtr a0 = qg, a1 = k, a2 = v, a3 = qNorm, a4 = kNorm, a5 = q, a6 = gate, a7 = kCache, a8 = vCache, a9 = positions;
            int a10 = p0, a11 = nHead, a12 = nKvHead, a13 = nRot;
            float a14 = ropeBase, a15 = freqScale, a16 = eps;
            uint grid = (uint)(nHead + nKvHead);
            if (kScale != IntPtr.Zero)
            {
                RequireKvQ8();
                IntPtr s0 = kScale, s1 = vScale;
                void** q8 = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &s0, &s1, &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16 };
                Launch(attnPrepQ8, grid, (uint)rows, 1, AttnHeadDim, stream, q8);
                return;
            }
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16 };
            Launch(attnPrep, grid, (uint)rows, 1, AttnHeadDim, stream, args);
        }

        /// <summary>Gated attention of every query head over 0..p, or over the listed cells where
        /// <paramref name="cellCnt"/>[row] is not negative. A Q8_0 cache passes its scales.</summary>
        public void Attention(IntPtr q, IntPtr kCache, IntPtr vCache, IntPtr gate, IntPtr cells, IntPtr cellCnt, IntPtr output,
            IntPtr positions, int p0, int nHead, int nKvHead, float scale, int cellStride, int rows, IntPtr stream,
            IntPtr kScale = default, IntPtr vScale = default)
        {
            IntPtr a0 = q, a1 = kCache, a2 = vCache, a3 = gate, a4 = cells, a5 = cellCnt, a6 = output, a7 = positions;
            int a8 = p0, a9 = nHead, a10 = nKvHead; float a11 = scale; int a12 = cellStride;
            if (kScale != IntPtr.Zero)
            {
                RequireKvQ8();
                IntPtr s0 = kScale, s1 = vScale;
                void** q8 = stackalloc void*[] { &a0, &a1, &a2, &s0, &s1, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12 };
                Launch(attentionQ8, (uint)nHead, (uint)rows, 1, BlockSize, stream, q8);
                return;
            }
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12 };
            Launch(attention, (uint)nHead, (uint)rows, 1, BlockSize, stream, args);
        }

        /// <summary><see cref="Attention"/> for prompt-size calls on tensor cores (needs <see cref="HasAttentionMma"/>, a head
        /// dim of 256 and at most <see cref="AttnMmaMaxGroup"/> query heads per KV head): a block of four warps per (row, KV
        /// head) computes the group's heads together, each cached key and value widened once to F16.</summary>
        public void AttentionMma(IntPtr q, IntPtr kCache, IntPtr vCache, IntPtr gate, IntPtr cells, IntPtr cellCnt, IntPtr output,
            IntPtr positions, int p0, int nHead, int nKvHead, float scale, int cellStride, int rows, IntPtr stream,
            IntPtr kScale = default, IntPtr vScale = default)
        {
            if (!HasAttentionMma || nHead % nKvHead != 0 || nHead / nKvHead > AttnMmaMaxGroup)
                throw new InvalidOperationException("[q4e-cuda] the tensor-core prompt attention is not available for this layout");
            IntPtr a0 = q, a1 = kCache, a2 = vCache, a3 = gate, a4 = cells, a5 = cellCnt, a6 = output, a7 = positions;
            int a8 = p0, a9 = nHead, a10 = nKvHead; float a11 = scale; int a12 = cellStride;
            if (kScale != IntPtr.Zero)
            {
                IntPtr s0 = kScale, s1 = vScale;
                void** q8 = stackalloc void*[] { &a0, &a1, &a2, &s0, &s1, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12 };
                Launch(attentionMmaQ8, (uint)nKvHead, (uint)rows, 1, 128, stream, q8);
                return;
            }
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12 };
            Launch(attentionMma, (uint)nKvHead, (uint)rows, 1, 128, stream, args);
        }

        /// <summary>Whether <see cref="AttentionSplit"/> is available.</summary>
        public bool HasAttentionSplit => attentionSplit != IntPtr.Zero && attentionSplitQ8 != IntPtr.Zero;

        /// <summary>Floats of one (row, head, split) partial of <see cref="AttentionSplit"/>: max, sum and the output.</summary>
        public const int AttnSplitPartFloats = 2 + AttnHeadDim;

        /// <summary>Cells up to which a row runs whole in one block (the kernel's Q4E_ATTN_SPLIT_MIN).</summary>
        public const int AttnSplitMinCells = 256;

        /// <summary>
        /// <see cref="Attention"/> for decode-size calls, each row of more than <see cref="AttnSplitMinCells"/> cells split
        /// over <paramref name="splits"/> blocks (flash decoding), merged by the last block to finish.
        /// <paramref name="part"/> holds rows x heads x splits x <see cref="AttnSplitPartFloats"/> floats and
        /// <paramref name="arrive"/> rows x heads ints, zero between calls (the kernel leaves them so).
        /// </summary>
        public void AttentionSplit(IntPtr q, IntPtr kCache, IntPtr vCache, IntPtr gate, IntPtr cells, IntPtr cellCnt, IntPtr output,
            IntPtr positions, int p0, int nHead, int nKvHead, float scale, int cellStride, int rows, int splits, IntPtr part,
            IntPtr arrive, IntPtr stream, IntPtr kScale = default, IntPtr vScale = default)
        {
            IntPtr a0 = q, a1 = kCache, a2 = vCache, a3 = gate, a4 = cells, a5 = cellCnt, a6 = output, a7 = positions;
            int a8 = p0, a9 = nHead, a10 = nKvHead; float a11 = scale; int a12 = cellStride;
            IntPtr a13 = part, a14 = arrive;
            if (kScale != IntPtr.Zero)
            {
                IntPtr s0 = kScale, s1 = vScale;
                void** q8 = stackalloc void*[] { &a0, &a1, &a2, &s0, &s1, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14 };
                Launch(attentionSplitQ8, (uint)nHead, (uint)rows, (uint)splits, BlockSize, stream, q8);
                return;
            }
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14 };
            Launch(attentionSplit, (uint)nHead, (uint)rows, (uint)splits, BlockSize, stream, args);
        }

        // ---- KV streaming (TS_Q4E_KV_RESIDENT) ----

        /// <summary><see cref="AttnPrep"/> of a Q8_0 layer whose K/V streams: the host copy (device-mapped pointers) always,
        /// the VRAM slot of a resident page through <paramref name="table"/>.</summary>
        public void AttnPrepPaged(IntPtr qg, IntPtr k, IntPtr v, IntPtr qNorm, IntPtr kNorm, IntPtr q, IntPtr gate,
            IntPtr kCache, IntPtr vCache, IntPtr kScale, IntPtr vScale, IntPtr positions, int p0, int nHead, int nKvHead, int nRot,
            float ropeBase, float freqScale, float eps, int rows, IntPtr table, IntPtr hostK, IntPtr hostV, IntPtr hostKScale,
            IntPtr hostVScale, IntPtr stream)
        {
            IntPtr a0 = qg, a1 = k, a2 = v, a3 = qNorm, a4 = kNorm, a5 = q, a6 = gate, a7 = kCache, a8 = vCache, a9 = kScale, a10 = vScale;
            IntPtr a11 = positions; int a12 = p0, a13 = nHead, a14 = nKvHead, a15 = nRot;
            float a16 = ropeBase, a17 = freqScale, a18 = eps;
            IntPtr a19 = table, a20 = hostK, a21 = hostV, a22 = hostKScale, a23 = hostVScale;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15,
                &a16, &a17, &a18, &a19, &a20, &a21, &a22, &a23 };
            Launch(attnPrepQ8Paged, (uint)(nHead + nKvHead), (uint)rows, 1, AttnHeadDim, stream, args);
        }

        /// <summary><see cref="Attention"/> (Q8_0) of a streamed layer: cells read through <paramref name="table"/>.</summary>
        public void AttentionPaged(IntPtr q, IntPtr kCache, IntPtr vCache, IntPtr kScale, IntPtr vScale, IntPtr gate, IntPtr cells,
            IntPtr cellCnt, IntPtr output, IntPtr positions, int p0, int nHead, int nKvHead, float scale, int cellStride, int rows,
            IntPtr table, IntPtr stream)
        {
            IntPtr a0 = q, a1 = kCache, a2 = vCache, a3 = kScale, a4 = vScale, a5 = gate, a6 = cells, a7 = cellCnt, a8 = output;
            IntPtr a9 = positions; int a10 = p0, a11 = nHead, a12 = nKvHead; float a13 = scale; int a14 = cellStride;
            IntPtr a15 = table;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15 };
            Launch(attentionQ8Paged, (uint)nHead, (uint)rows, 1, BlockSize, stream, args);
        }

        /// <summary><see cref="AttentionSplit"/> (Q8_0) of a streamed layer: cells read through <paramref name="table"/>.</summary>
        public void AttentionSplitPaged(IntPtr q, IntPtr kCache, IntPtr vCache, IntPtr kScale, IntPtr vScale, IntPtr gate, IntPtr cells,
            IntPtr cellCnt, IntPtr output, IntPtr positions, int p0, int nHead, int nKvHead, float scale, int cellStride, int rows,
            int splits, IntPtr part, IntPtr arrive, IntPtr table, IntPtr stream)
        {
            IntPtr a0 = q, a1 = kCache, a2 = vCache, a3 = kScale, a4 = vScale, a5 = gate, a6 = cells, a7 = cellCnt, a8 = output;
            IntPtr a9 = positions; int a10 = p0, a11 = nHead, a12 = nKvHead; float a13 = scale; int a14 = cellStride;
            IntPtr a15 = part, a16 = arrive, a17 = table;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15,
                &a16, &a17 };
            Launch(attentionSplitQ8Paged, (uint)nHead, (uint)rows, (uint)splits, BlockSize, stream, args);
        }

        /// <summary>Makes every page the call's rows read resident (their listed cells, or 0..p where
        /// <paramref name="cellCnt"/> is zero or a row's count negative), claiming slots for the misses, which
        /// <see cref="KvPageCopy"/> then fills. One block; capturable.</summary>
        public void KvResolve(IntPtr cells, IntPtr cellCnt, IntPtr positions, int p0, int rows, int cellStride, IntPtr table,
            IntPtr slotPage, IntPtr slotStamp, IntPtr ctl, IntPtr missPage, IntPtr missSlot, int nSlots, IntPtr stream)
        {
            IntPtr a0 = cellCnt == IntPtr.Zero ? IntPtr.Zero : cells, a1 = cellCnt, a2 = positions;
            int a3 = p0, a4 = rows, a5 = cellStride;
            IntPtr a6 = table, a7 = slotPage, a8 = slotStamp, a9 = ctl, a10 = missPage, a11 = missSlot;
            int a12 = nSlots, a13 = nSlots;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13 };
            Launch(kvResolve, 1, 1, 1, 1024, stream, args);
        }

        /// <summary>The pages the last <see cref="KvResolve"/> placed, copied from the host copy into their slots.</summary>
        public void KvPageCopy(IntPtr ctl, IntPtr missPage, IntPtr missSlot, IntPtr hostK, IntPtr hostV, IntPtr hostKScale,
            IntPtr hostVScale, IntPtr kCache, IntPtr vCache, IntPtr kScale, IntPtr vScale, int nKvHead, IntPtr stream)
        {
            IntPtr a0 = ctl, a1 = missPage, a2 = missSlot, a3 = hostK, a4 = hostV, a5 = hostKScale, a6 = hostVScale;
            IntPtr a7 = kCache, a8 = vCache, a9 = kScale, a10 = vScale; int a11 = nKvHead;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11 };
            Launch(kvPageCopy, 128, 1, 1, BlockSize, stream, args);
        }

        /// <summary>Pages [first, first + count) leave the window.</summary>
        public void KvInvalidate(IntPtr table, IntPtr slotPage, int first, int count, IntPtr stream)
        {
            if (count <= 0)
                return;
            IntPtr a0 = table, a1 = slotPage; int a2 = first, a3 = count;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3 };
            Launch(kvInvalidate, CeilDiv(count, BlockSize), 1, 1, BlockSize, stream, args);
        }

        /// <summary><see cref="QsaStore"/> into a ring of <paramref name="ring"/> cells.</summary>
        public void QsaStoreRing(IntPtr raw, IntPtr cache, IntPtr positions, int p0, int d, int rows, int ring, IntPtr stream)
        {
            IntPtr a0 = raw, a1 = cache, a2 = positions;
            int a3 = p0, a4 = d, a5 = ring;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(qsaStoreRing, (uint)rows, 1, 1, Math.Min(d, BlockSize), stream, args);
        }

        /// <summary><see cref="QsaPool"/> from a ring of <paramref name="ring"/> raw keys.</summary>
        public void QsaPoolRing(IntPtr cache, IntPtr kNorm, IntPtr pooled, IntPtr positions, int first, int count, int ratio, int d,
            int nRot, float ropeBase, float freqScale, float eps, int ring, IntPtr stream)
        {
            if (positions == IntPtr.Zero && count <= 0)
                return;
            IntPtr a0 = cache, a1 = kNorm, a2 = pooled, a3 = positions;
            int a4 = first, a5 = ratio, a6 = d, a7 = nRot;
            float a8 = ropeBase, a9 = freqScale, a10 = eps; int a11 = ring;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11 };
            Launch(qsaPoolRing, positions == IntPtr.Zero ? (uint)count : 1u, 1, 1, d, stream, args);
        }

        private void RequireKvQ8()
        {
            if (!HasKvQ8)
                throw new InvalidOperationException("The loaded q4e kernel module has no Q8_0 KV kernels; rebuild the CUDA kernels.");
        }

        // ---- QSA ----

        public void QsaStore(IntPtr raw, IntPtr cache, IntPtr positions, int p0, int d, int rows, IntPtr stream)
        {
            IntPtr a0 = raw, a1 = cache, a2 = positions;
            int a3 = p0, a4 = d;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(qsaStore, (uint)rows, 1, 1, Math.Min(d, BlockSize), stream, args);
        }

        /// <summary>The keys of blocks [first, first + count) from their cached raw keys; with
        /// <paramref name="positions"/>, the block the row at positions[0] completes (count ignored).</summary>
        public void QsaPool(IntPtr cache, IntPtr kNorm, IntPtr pooled, IntPtr positions, int first, int count, int ratio, int d,
            int nRot, float ropeBase, float freqScale, float eps, IntPtr stream)
        {
            if (positions == IntPtr.Zero && count <= 0)
                return;
            IntPtr a0 = cache, a1 = kNorm, a2 = pooled, a3 = positions;
            int a4 = first, a5 = ratio, a6 = d, a7 = nRot;
            float a8 = ropeBase, a9 = freqScale, a10 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10 };
            Launch(qsaPool, positions == IntPtr.Zero ? (uint)count : 1u, 1, 1, d, stream, args);
        }

        public void QsaQuery(IntPtr q, IntPtr qNorm, IntPtr positions, int p0, int heads, int d, int nRot,
            float ropeBase, float freqScale, float eps, int rows, IntPtr stream)
        {
            IntPtr a0 = q, a1 = qNorm, a2 = positions;
            int a3 = p0, a4 = heads, a5 = d, a6 = nRot;
            float a7 = ropeBase, a8 = freqScale, a9 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            Launch(qsaQuery, (uint)heads, (uint)rows, 1, d, stream, args);
        }

        public void QsaScores(IntPtr q, IntPtr pooled, IntPtr scores, IntPtr positions, int p0, int heads, int d, int ratio,
            int stride, int maxBlocks, int rows, IntPtr stream)
        {
            if (maxBlocks <= 0)
                return;
            IntPtr a0 = q, a1 = pooled, a2 = scores, a3 = positions;
            int a4 = p0, a5 = heads, a6 = d, a7 = ratio, a8 = stride;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
            Launch(qsaScores, Math.Min(CeilDiv(maxBlocks, 8), MaxScoreBlocks), (uint)rows, 1, BlockSize, stream, args);
        }

        public void QsaSelect(IntPtr scores, IntPtr cells, IntPtr cellCnt, IntPtr positions, int p0, int ratio, int topK,
            int stride, int cellStride, int rows, IntPtr stream)
        {
            IntPtr a0 = scores, a1 = cells, a2 = cellCnt, a3 = positions;
            int a4 = p0, a5 = ratio, a6 = topK, a7 = stride, a8 = cellStride;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
            Launch(qsaSelect, (uint)rows, 1, 1, BlockSize, stream, args);
        }

        // ---- PLE ----

        public void PleGate(IntPtr key, IntPtr res, IntPtr normKey, IntPtr normQuery, IntPtr value, IntPtr gated,
            int rows, int e, int hc, float eps, IntPtr stream)
        {
            IntPtr a0 = key, a1 = res, a2 = normKey, a3 = normQuery, a4 = value, a5 = gated;
            int a6 = e, a7 = hc; float a8 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
            Launch(pleGate, (uint)rows, 1, 1, BlockSize, stream, args);
        }

        /// <summary>y = each stream of x RMS-normed times w.</summary>
        public void GroupNorm(IntPtr x, IntPtr w, IntPtr y, int rows, int e, int hc, float eps, IntPtr stream)
        {
            IntPtr a0 = x, a1 = w, a2 = y;
            int a3 = e, a4 = hc; float a5 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(groupNorm, (uint)rows, (uint)hc, 1, BlockSize, stream, args);
        }

        public void PleConv(IntPtr gated, IntPtr normc, IntPtr hists, IntPtr convWT, IntPtr res,
            int rowsPerSeq, int nSeq, int channels, int kern, int dil, IntPtr stream)
        {
            IntPtr a0 = gated, a1 = normc, a2 = hists, a3 = convWT, a4 = res;
            int a5 = rowsPerSeq, a6 = nSeq, a7 = channels, a8 = kern, a9 = dil;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            Launch(pleConv, CeilDiv((long)rowsPerSeq * nSeq * channels, BlockSize), 1, 1, BlockSize, stream, args);
        }

        // ---- MoE ----

        /// <summary>sh[t] *= sigmoid(cur[t] . w).</summary>
        public void SharedGate(IntPtr cur, IntPtr w, IntPtr sh, int rows, int e, IntPtr stream)
        {
            IntPtr a0 = cur, a1 = w, a2 = sh;
            int a3 = e;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3 };
            Launch(sharedGate, (uint)rows, 1, 1, BlockSize, stream, args);
        }

        /// <summary>xs[firstRow + r, every stream] = rows[r] for <paramref name="nRows"/> rows.</summary>
        public void SetStreamRows(IntPtr rows, IntPtr xs, int firstRow, int nRows, int e, int hc, IntPtr stream)
        {
            IntPtr a0 = rows, a1 = xs;
            int a2 = firstRow, a3 = e, a4 = hc;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(setStreamRows, CeilDiv(e, BlockSize), (uint)nRows, 1, BlockSize, stream, args);
        }

        /// <summary>Whether <see cref="ArgmaxRows"/> is available.</summary>
        public bool HasArgmaxRows => argmaxRows != IntPtr.Zero;

        /// <summary>ids[r] = the first index of row r's largest logit (0 when it has no finite one).</summary>
        public void ArgmaxRows(IntPtr logits, IntPtr ids, int rows, int vocab, IntPtr stream)
        {
            IntPtr a0 = logits, a1 = ids;
            int a2 = vocab;
            void** args = stackalloc void*[] { &a0, &a1, &a2 };
            Launch(argmaxRows, (uint)rows, 1, 1, 1024, stream, args);
        }

        /// <summary>Whether <see cref="CountSelected"/> is available.</summary>
        public bool HasCountSelected => countSelected != IntPtr.Zero;

        /// <summary>counts[sel[i]] += 1 for the <paramref name="n"/> selections (unsigned counts of one layer).</summary>
        public void CountSelected(IntPtr sel, int n, IntPtr counts, IntPtr stream)
        {
            IntPtr a0 = sel, a2 = counts;
            int a1 = n;
            void** args = stackalloc void*[] { &a0, &a1, &a2 };
            Launch(countSelected, CeilDiv(n, BlockSize), 1, 1, BlockSize, stream, args);
        }

        /// <summary>Whether <see cref="DraftPick"/> is available.</summary>
        public bool HasDraftPick => draftPick != IntPtr.Zero;

        /// <summary>Logits each block of the two-stage pick's first stage reads.</summary>
        public const int DraftPickChunk = 1024;

        /// <summary>Candidate entries (a float and an int each) the two-stage pick needs for <paramref name="vocab"/> logits.</summary>
        public static int DraftPickCandidates(int vocab) => CeilDivInt(vocab, DraftPickChunk) * 10;

        private static int CeilDivInt(int a, int b) => (a + b - 1) / b;

        /// <summary>A draft step's first-max token into ids[step] and nextToken[0], and its top-10 softmax confidence
        /// into probs[step]. With <paramref name="candidates"/> (<see cref="DraftPickCandidates"/> floats then as many
        /// ints) a block per 1,024 logits keeps its slice's ten largest and one block merges them: the same pick, bit for
        /// bit (TS_Q4E_DRAFT_PICK_SPLIT=0 runs the one-block kernel).</summary>
        public void DraftPick(IntPtr logits, int vocab, IntPtr ids, IntPtr probs, int step, IntPtr nextToken, IntPtr stream,
            IntPtr candidates = default)
        {
            IntPtr a0 = logits, a2 = ids, a3 = probs, a5 = nextToken;
            int a1 = vocab, a4 = step;
            if (candidates != IntPtr.Zero && draftPickPart != IntPtr.Zero && _draftPickSplit)
            {
                int blocks = CeilDivInt(vocab, DraftPickChunk), n = blocks * 10, chunk = DraftPickChunk;
                IntPtr cv = candidates, ck = (IntPtr)((long)candidates + 4L * n);
                void** pa = stackalloc void*[] { &a0, &a1, &chunk, &cv, &ck };
                Launch(draftPickPart, (uint)blocks, 1, 1, 256, stream, pa);
                void** ma = stackalloc void*[] { &cv, &ck, &n, &a2, &a3, &a4, &a5 };
                Launch(draftPickMerge, 1, 1, 1, 1024, stream, ma);
                return;
            }
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(draftPick, 1, 1, 1, 1024, stream, args);
        }

        private static readonly bool _draftPickSplit = Environment.GetEnvironmentVariable("TS_Q4E_DRAFT_PICK_SPLIT") != "0";

        public void Dispose() => module.Dispose();
    }
}
