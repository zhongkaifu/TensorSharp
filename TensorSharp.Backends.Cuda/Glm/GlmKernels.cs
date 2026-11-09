// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Launchers for the GLM-5.3-Flash direct-CUDA kernels
// (native/kernels/tensorsharp_glm_kernels.cu, compiled to
// tensorsharp_glm_kernels.ptx). The model's hyper-connection, MoE and attention
// core kernels are the DeepSeek V4 module's (Dsv4Kernels); these are the rest.
// One instance per device, loaded against the context current at construction.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal sealed unsafe class GlmKernels : IDisposable
    {
        private const int BlockSize = 256;

        /// <summary>KDA head width the kernels are written for (TS_GLM_KDA_HD).</summary>
        public const int KdaHeadDim = 128;
        /// <summary>Floats of scan scratch per (sequence row, head) (TS_GLM_KDA_SCR).</summary>
        public const int KdaScratch = 4 * KdaHeadDim + 1;

        private readonly CudaModule module;
        private readonly IntPtr kdaPrep;
        private readonly IntPtr kdaScan;
        private readonly IntPtr kdaScanR1;

        /// <summary>Four state rows a warp for every scan, not only decode steps (<c>TS_KDA_SCAN_R4=1</c>;
        /// settable for tests). Both kernels compute every state element alike.</summary>
        internal static bool KdaScanFourRows { get; set; } = Environment.GetEnvironmentVariable("TS_KDA_SCAN_R4") == "1";
        private readonly IntPtr kdaOut;
        private readonly IntPtr kdaConvUpdate;
        private readonly IntPtr mlaKvStore;
        private readonly IntPtr indexerStore;
        private readonly IntPtr poolKeys;
        private readonly IntPtr poolScores;
        private readonly IntPtr expandCells;
        private readonly IntPtr[] headGemv;
        private readonly IntPtr setStreamRows;

        /// <summary>Rows the per-head projection GEMV takes (TS_GLM head GEMV instantiations).</summary>
        public const int HeadGemvMaxRows = 16;

        /// <summary>Cells per indexer pool (TS_GLM_KPOOL).</summary>
        public const int KPool = 4;

        private GlmKernels(CudaModule module)
        {
            this.module = module;
            kdaPrep = module.GetFunction("ts_glm_kda_prep_f32");
            kdaScan = module.GetFunction("ts_glm_kda_scan_f32");
            kdaScanR1 = module.GetFunction("ts_glm_kda_scan_r1_f32");
            kdaOut = module.GetFunction("ts_glm_kda_out_f32");
            kdaConvUpdate = module.GetFunction("ts_glm_kda_conv_update_f32");
            mlaKvStore = module.GetFunction("ts_glm_mla_kv_store_f32");
            indexerStore = module.GetFunction("ts_glm_indexer_store_f32");
            poolKeys = module.GetFunction("ts_glm_pool_keys_f32");
            poolScores = module.GetFunction("ts_glm_pool_scores_f32");
            expandCells = module.GetFunction("ts_glm_expand_cells_i32");
            setStreamRows = module.GetFunction("ts_glm_set_stream_rows_f32");
            headGemv = new[]
            {
                module.GetFunction("ts_glm_head_gemv_1_f32"), module.GetFunction("ts_glm_head_gemv_4_f32"),
                module.GetFunction("ts_glm_head_gemv_8_f32"), module.GetFunction("ts_glm_head_gemv_16_f32"),
            };
        }

        public static GlmKernels Create()
        {
            string path = CudaKernels.LocatePtxPath("tensorsharp_glm_kernels.ptx");
            if (path == null)
            {
                throw new InvalidOperationException(
                    "tensorsharp_glm_kernels.ptx could not be located next to the application or under " +
                    "a 'cuda_kernels' folder. Build TensorSharp.Backends.Cuda with a CUDA 13 nvcc on the PATH.");
            }
            return new GlmKernels(CudaModule.LoadFromFile(path));
        }

        private static void Launch(IntPtr fn, uint gx, uint gy, uint gz, int block, IntPtr stream, void** args)
        {
            CudaDriverApi.cuLaunchKernel(fn, gx, gy, gz, (uint)block, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        private static uint CeilDiv(long n, int d) => (uint)Math.Max(1, (n + d - 1) / d);

        /// <summary>
        /// Short convolution, L2 norms, per-channel log decay and beta for <paramref name="nSeq"/>
        /// sequences of <paramref name="rowsPerSeq"/> rows. <paramref name="convStates"/> is a device
        /// array of <paramref name="nSeq"/> pointers to [dc-1, 3 * H * 128] states.
        /// </summary>
        public void KdaPrep(IntPtr q, IntPtr k, IntPtr v, IntPtr convStates, IntPtr convQ, IntPtr convK, IntPtr convV,
            IntPtr f, IntPtr dtBias, IntPtr aVec, IntPtr betaRaw, IntPtr scr,
            int rowsPerSeq, int nSeq, int heads, int dConv, float gateLowerBound, IntPtr stream)
        {
            IntPtr a0 = q, a1 = k, a2 = v, a3 = convStates, a4 = convQ, a5 = convK, a6 = convV, a7 = f,
                a8 = dtBias, a9 = aVec, a10 = betaRaw, a11 = scr;
            int a12 = rowsPerSeq, a13 = nSeq, a14 = heads, a15 = dConv;
            float a16 = gateLowerBound;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11,
                &a12, &a13, &a14, &a15, &a16 };
            long warps = (long)rowsPerSeq * nSeq * heads;
            Launch(kdaPrep, CeilDiv(warps * 32, BlockSize), 1, 1, BlockSize, stream, args);
        }

        /// <summary>The gated delta recurrence over each sequence's rows, in its own state
        /// (<paramref name="ssmStates"/>: device array of <paramref name="nSeq"/> pointers to
        /// [H, 128, 128]); writes the per-head outputs into <paramref name="core"/>.</summary>
        public void KdaScan(IntPtr scr, IntPtr ssmStates, IntPtr core, int rowsPerSeq, int nSeq, int heads, IntPtr stream)
        {
            IntPtr a0 = scr, a1 = ssmStates, a2 = core;
            int a3 = rowsPerSeq, a4 = heads;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            // A prefill ubatch walks its rows in order: a state row per warp (four per block), as
            // ts_glm_kda_scan_f32 computes each, so the SMs hold four times the warps. A decode step (one row a
            // sequence) keeps four rows a warp, which needs a quarter of the blocks. TS_KDA_SCAN_R4=1 runs four
            // rows a warp everywhere (A/B); both are bit-identical.
            const int warpsPerBlock = 4;
            if (KdaScanFourRows || rowsPerSeq <= 1)
                Launch(kdaScan, (uint)heads, (uint)(KdaHeadDim / (4 * warpsPerBlock)), (uint)nSeq, warpsPerBlock * 32, stream, args);
            else
                Launch(kdaScanR1, (uint)heads, (uint)(KdaHeadDim / warpsPerBlock), (uint)nSeq, warpsPerBlock * 32, stream, args);
        }

        public void KdaOut(IntPtr core, IntPtr gate, IntPtr normW, IntPtr output, int rows, int heads, float eps, IntPtr stream)
        {
            IntPtr a0 = core, a1 = gate, a2 = normW, a3 = output;
            int a4 = rows, a5 = heads; float a6 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
            Launch(kdaOut, CeilDiv((long)rows * heads * 32, BlockSize), 1, 1, BlockSize, stream, args);
        }

        public void KdaConvUpdate(IntPtr q, IntPtr k, IntPtr v, IntPtr convStates, int rowsPerSeq, int nSeq, int dInner,
            int dConv, IntPtr stream)
        {
            IntPtr a0 = q, a1 = k, a2 = v, a3 = convStates;
            int a4 = rowsPerSeq, a5 = nSeq, a6 = dInner, a7 = dConv;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            Launch(kdaConvUpdate, CeilDiv((long)nSeq * 3 * dInner, BlockSize), 1, 1, BlockSize, stream, args);
        }

        /// <summary>RMS-normed latent rows into an F16 cache at positions p0, p0 + 1, ...</summary>
        public void MlaKvStore(IntPtr kv, IntPtr normW, IntPtr cache, int p0, int nt, int width, float eps, IntPtr stream,
            IntPtr positions = default)
        {
            IntPtr a0 = kv, a1 = normW, a2 = cache, a6 = positions;
            int a3 = p0, a4 = width; float a5 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
            Launch(mlaKvStore, (uint)nt, 1, 1, BlockSize, stream, args);
        }

        /// <summary>LayerNormed indexer keys beside their pooling gates, into an F16 cache.</summary>
        public void IndexerStore(IntPtr key, IntPtr gate, IntPtr lnW, IntPtr lnB, IntPtr cache, int p0, int nt, int d,
            float eps, IntPtr stream, IntPtr positions = default)
        {
            IntPtr a0 = key, a1 = gate, a2 = lnW, a3 = lnB, a4 = cache, a8 = positions;
            int a5 = p0, a6 = d; float a7 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
            Launch(indexerStore, (uint)nt, 1, 1, 128, stream, args);
        }

        /// <summary>Keys of the pools [firstPool, firstPool + count) from their members' cached keys
        /// and pooling gates; with <paramref name="positions"/> (a captured decode step), the pool the
        /// row at positions[0] completes, if any.</summary>
        public void PoolKeys(IntPtr idxCache, IntPtr ape, IntPtr poolKeysOut, int firstPool, int count, int d, IntPtr stream,
            IntPtr positions = default)
        {
            if (positions == IntPtr.Zero && count <= 0)
                return;
            IntPtr a0 = idxCache, a1 = ape, a2 = poolKeysOut, a5 = positions;
            int a3 = firstPool, a4 = d;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(poolKeys, positions == IntPtr.Zero ? (uint)count : 1u, 1, 1, Math.Min(d, 256), stream, args);
        }

        /// <summary>Each query's score of every pool it sees.</summary>
        public void PoolScores(IntPtr q, IntPtr w, IntPtr poolKeysIn, IntPtr scores, int p0, int heads, int d,
            int stride, int maxVis, int nt, IntPtr stream, IntPtr positions = default)
        {
            IntPtr a0 = q, a1 = w, a2 = poolKeysIn, a3 = scores, a9 = positions;
            int a4 = p0, a5 = heads, a6 = d, a7 = stride, a8 = maxVis;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            const int poolsPerBlock = 8;   // one warp per pool
            CudaDriverApi.cuLaunchKernel(poolScores, Math.Min(CeilDiv(maxVis, poolsPerBlock), MaxScoreBlocks), (uint)nt, 1,
                32, poolsPerBlock, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>CTAs the pool scoring launches at most per row (its warps stride over the rest):
        /// about one full wave on the A40.</summary>
        private const uint MaxScoreBlocks = 512;

        /// <summary>The cells each query attends over: its selected pools' cells, then its own
        /// pool's cells up to itself.</summary>
        public void ExpandCells(IntPtr sel, IntPtr selCnt, IntPtr cells, IntPtr cellCnt, int p0, int k, int cellStride,
            int nt, IntPtr stream, IntPtr positions = default)
        {
            IntPtr a0 = sel, a1 = selCnt, a2 = cells, a3 = cellCnt, a7 = positions;
            int a4 = p0, a5 = k, a6 = cellStride;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            Launch(expandCells, (uint)nt, 1, 1, BlockSize, stream, args);
        }

        /// <summary>y[t, h, :] = W_h x[t, h, :] for up to <see cref="HeadGemvMaxRows"/> rows: W an F16 stack
        /// of heads [outDim, inDim] matrices, x and y F32.</summary>
        public void HeadGemv(IntPtr w, IntPtr x, IntPtr y, int heads, int outDim, int inDim, int rows, IntPtr stream)
        {
            IntPtr a0 = w, a1 = x, a2 = y;
            int a3 = heads, a4 = outDim, a5 = inDim, a6 = rows;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
            int idx = rows <= 1 ? 0 : rows <= 4 ? 1 : rows <= 8 ? 2 : 3;
            Launch(headGemv[idx], CeilDiv(outDim, BlockSize / 32), (uint)heads, 1, BlockSize, stream, args);
        }

        /// <summary>xs[firstRow + r, every stream, :] = rows[r, :] for <paramref name="nRows"/> rows.</summary>
        public void SetStreamRows(IntPtr rows, IntPtr xs, int firstRow, int nRows, int e, IntPtr stream)
        {
            IntPtr a0 = rows, a1 = xs;
            int a2 = firstRow, a3 = e;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3 };
            Launch(setStreamRows, CeilDiv(e, BlockSize), (uint)nRows, 1, BlockSize, stream, args);
        }

        public void Dispose() => module.Dispose();
    }
}
