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
        private readonly IntPtr hcNorm, siluScale, hcCollapse, hcPost, hcPartials, hcApply, siluQ81;

        /// <summary>Values per mixer slice (Q4E_HC_SLICE) and floats of partials per slice (Q4E_HC_PARTIAL).</summary>
        public const int HcSlice = 256, HcPartialFloats = 1 + MaxStreams;
        private readonly IntPtr gdnPrep, convUpdate;
        private readonly IntPtr attnPrep, attention, attentionG4;
        private readonly IntPtr qsaStore, qsaPool, qsaQuery, qsaScores, qsaSelect;
        private readonly IntPtr pleGate, groupNorm, pleConv;
        private readonly IntPtr sharedGate, setStreamRows;

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
            gdnPrep = module.GetFunction("ts_q4e_gdn_prep_f32");
            convUpdate = module.GetFunction("ts_q4e_conv_update_f32");
            attnPrep = module.GetFunction("ts_q4e_attn_prep_f32");
            attention = module.GetFunction("ts_q4e_attention_f32");
            attentionG4 = module.GetFunction("ts_q4e_attention_g4_f32");
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

        /// <summary>xs[t, c] += out[t] * 2 sigmoid(inject[t, c] / hc).</summary>
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
            Launch(gdnPrep, CeilDiv(warps * 32, BlockSize), 1, 1, BlockSize, stream, args);
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
        /// stored into the F16 caches.</summary>
        public void AttnPrep(IntPtr qg, IntPtr k, IntPtr v, IntPtr qNorm, IntPtr kNorm, IntPtr q, IntPtr gate,
            IntPtr kCache, IntPtr vCache, IntPtr positions, int p0, int nHead, int nKvHead, int nRot,
            float ropeBase, float freqScale, float eps, int rows, IntPtr stream)
        {
            IntPtr a0 = qg, a1 = k, a2 = v, a3 = qNorm, a4 = kNorm, a5 = q, a6 = gate, a7 = kCache, a8 = vCache, a9 = positions;
            int a10 = p0, a11 = nHead, a12 = nKvHead, a13 = nRot;
            float a14 = ropeBase, a15 = freqScale, a16 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16 };
            Launch(attnPrep, (uint)(nHead + nKvHead), (uint)rows, 1, AttnHeadDim, stream, args);
        }

        /// <summary>Gated attention of every query head over 0..p, or over the listed cells where
        /// <paramref name="cellCnt"/>[row] is not negative.</summary>
        public void Attention(IntPtr q, IntPtr kCache, IntPtr vCache, IntPtr gate, IntPtr cells, IntPtr cellCnt, IntPtr output,
            IntPtr positions, int p0, int nHead, int nKvHead, float scale, int cellStride, int rows, IntPtr stream)
        {
            IntPtr a0 = q, a1 = kCache, a2 = vCache, a3 = gate, a4 = cells, a5 = cellCnt, a6 = output, a7 = positions;
            int a8 = p0, a9 = nHead, a10 = nKvHead; float a11 = scale; int a12 = cellStride;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12 };
            // A prefill ubatch's rows: four query heads of a KV head to a block, so each cell's K and V load
            // once for the four (bit-identical: each head computes as below). Decode-size steps keep a block a
            // head, which leaves more blocks to fill the GPU with few rows.
            if (rows > PrefillAttentionMinRows && GroupedPrefillAttention && nHead % 4 == 0 && (nHead / nKvHead) % 4 == 0)
                Launch(attentionG4, (uint)(nHead / 4), (uint)rows, 1, BlockSize, stream, args);
            else
                Launch(attention, (uint)nHead, (uint)rows, 1, BlockSize, stream, args);
        }

        /// <summary>Rows above which <see cref="Attention"/> groups query heads.</summary>
        private const int PrefillAttentionMinRows = 16;

        /// <summary>TS_Q4E_GROUPED_ATTENTION=0: every attention runs a block per query head (A/B; settable for
        /// tests).</summary>
        internal static bool GroupedPrefillAttention { get; set; } =
            Environment.GetEnvironmentVariable("TS_Q4E_GROUPED_ATTENTION") != "0";

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
            Launch(groupNorm, (uint)rows, 1, 1, BlockSize, stream, args);
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

        public void Dispose() => module.Dispose();
    }
}
