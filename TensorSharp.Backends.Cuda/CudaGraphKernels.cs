// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Launchers for the graph executor's kernels (native/kernels/tensorsharp_graph_kernels.cu, compiled to
// tensorsharp_graph_kernels.ptx): the program ops with no generic tensor-op equivalent, with ggml's
// semantics.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal sealed unsafe class CudaGraphKernels : CudaPrefillModule
    {
        public const string PtxFileName = "tensorsharp_graph_kernels.ptx";
        public const string CapsKernelName = "ts_graph_caps";

        /// <summary>TS_GRAPH_KERNELS_VERSION in tensorsharp_graph_kernels.cu: bump both together whenever a
        /// kernel is added or changes its signature.</summary>
        public const int CurrentVersion = 26;

        private readonly IntPtr _gatedDeltaNet, _gatedDeltaNetRows4, _softplus, _sqrt, _topk, _topkWide, _argmaxPart, _argmaxMerge, _topkFewPart, _topkFewMerge, _getRows, _getRowsF16, _setRows, _lightningIndexer, _topkMask, _maskCells, _sumRows, _smallMatvec, _countRoutes, _binaryBcast, _hcNorm, _hcPartials, _hcPostPartials, _hcApply, _scaleSilu, _hcGateMix, _hcScatter, _f16ToF32, _f32ToF16, _routeSplit, _waitAdd, _stageCopy, _ssmConv, _deltaNetPrep, _deltaNetPrepTiled, _gatedRmsNorm, _addRmsNorm, _sigmoidGate, _repackQ6kTiled, _ropeStrided, _sharedGate, _qsaScores, _qsaSelect, _qsaAttention;
        private readonly IntPtr[] _gemmQ6kTiled = new IntPtr[4];
        private readonly int[] _smemQ6kTiled = new int[4];
        private readonly IntPtr[] _gemmQ4k = new IntPtr[4];
        private readonly int _multiprocessors;

        private CudaGraphKernels(CudaModule module) : base(module, CapsKernelName, CurrentVersion)
        {
            if (!IsCurrent) return;
            _gatedDeltaNet = module.GetFunction("ts_graph_gated_delta_net_f32");
            _gatedDeltaNetRows4 = module.GetFunction("ts_graph_gated_delta_net_rows4_f32");
            _softplus = module.GetFunction("ts_graph_softplus_f32");
            _sqrt = module.GetFunction("ts_graph_sqrt_f32");
            _topk = module.GetFunction("ts_graph_topk_f32");
            _topkWide = module.GetFunction("ts_graph_topk_wide_f32");
            _argmaxPart = module.GetFunction("ts_graph_argmax_part_f32");
            _argmaxMerge = module.GetFunction("ts_graph_argmax_merge");
            _topkFewPart = module.GetFunction("ts_graph_topk_few_part_f32");
            _topkFewMerge = module.GetFunction("ts_graph_topk_few_merge");
            _getRows = module.GetFunction("ts_graph_get_rows_f32");
            _getRowsF16 = module.GetFunction("ts_graph_get_rows_f16");
            _setRows = module.GetFunction("ts_graph_set_rows_f32");
            _lightningIndexer = module.GetFunction("ts_graph_lightning_indexer_f32");
            _topkMask = module.GetFunction("ts_graph_topk_mask_f32");
            _maskCells = module.GetFunction("ts_graph_mask_cells");
            _sumRows = module.GetFunction("ts_graph_sum_rows_f32");
            _smallMatvec = module.GetFunction("ts_graph_small_matvec_f32");
            _countRoutes = module.GetFunction("ts_graph_count_routes");
            _binaryBcast = module.GetFunction("ts_graph_binary_bcast_f32");
            _hcNorm = module.GetFunction("ts_graph_hc_norm_f32");
            _hcPartials = module.GetFunction("ts_graph_hc_partials_f32");
            _hcPostPartials = module.GetFunction("ts_graph_hc_post_partials_f32");
            _ropeStrided = module.GetFunction("ts_graph_rope_ex_strided_f32");
            _sharedGate = module.GetFunction("ts_graph_shared_gate_add_f32");
            _hcApply = module.GetFunction("ts_graph_hc_apply_f32");
            _scaleSilu = module.GetFunction("ts_graph_scale_silu_f32");
            _hcGateMix = module.GetFunction("ts_graph_hc_gate_mix_f32");
            _hcScatter = module.GetFunction("ts_graph_hc_scatter_f32");
            _f16ToF32 = module.GetFunction("ts_graph_f16_to_f32");
            _f32ToF16 = module.GetFunction("ts_graph_f32_to_f16");
            _routeSplit = module.GetFunction("ts_graph_moe_route_split");
            _waitAdd = module.GetFunction("ts_graph_moe_wait_add");
            _stageCopy = module.GetFunction("ts_graph_moe_stage_copy");
            CudaDriverApi.cuFuncSetAttribute(_topk, 8 /* MAX_DYNAMIC_SHARED_SIZE_BYTES */, TopKMaxColumns * 4).ThrowOnError();
            _ssmConv = module.GetFunction("ts_graph_ssm_conv_f32");
            _deltaNetPrep = module.GetFunction("ts_graph_deltanet_prep_f32");
            _deltaNetPrepTiled = module.GetFunction("ts_graph_deltanet_prep_tiled_f32");
            _sigmoidGate = module.GetFunction("ts_graph_sigmoid_gate_f32");
            _gatedRmsNorm = module.GetFunction("ts_graph_gated_rms_norm_f32");
            _addRmsNorm = module.GetFunction("ts_graph_add_rms_norm_f32");
            _repackQ6kTiled = module.GetFunction("ts_graph_repack_q6k_tiled");
            // The tiled head's products stream with cp.async into the int8 tensor cores: only an image with their sm_80 bodies.
            if (ArchGuardedKernelsUsable)
            {
                for (int i = 0; i < 4; i++)
                    _gemmQ6kTiled[i] = module.GetFunction($"ts_graph_gemm_q6kt_{1 << i}");
                CudaDriverApi.cuMemAlloc(out IntPtr probe, new UIntPtr(16)).ThrowOnError();
                try
                {
                    IntPtr p = probe;
                    void** args = stackalloc void*[] { &p };
                    CudaDriverApi.cuLaunchKernel(module.GetFunction("ts_graph_gemm_q6kt_smem_bytes"), 1, 1, 1, 1, 1, 1, 0, IntPtr.Zero, (IntPtr)args,
                        IntPtr.Zero).ThrowOnError();
                    int* host = stackalloc int[4];
                    CudaDriverApi.cuMemcpyDtoH((IntPtr)host, probe, new UIntPtr(16)).ThrowOnError();
                    for (int i = 0; i < 4; i++)
                    {
                        _smemQ6kTiled[i] = host[i];
                        CudaDriverApi.cuFuncSetAttribute(_gemmQ6kTiled[i], 8 /* MAX_DYNAMIC_SHARED_SIZE_BYTES */, host[i]).ThrowOnError();
                    }
                }
                finally
                {
                    CudaDriverApi.cuMemFree(probe);
                }
                TakesTiledQ6kHeads = true;
                for (int i = 0; i < 4; i++)
                    _gemmQ4k[i] = module.GetFunction($"ts_graph_gemm_q4k_{1 << i}");
                CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError();
                CudaDriverApi.cuDeviceGetAttribute(out _multiprocessors, 16 /* CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT */, device).ThrowOnError();
                TakesQ4kMma = true;
                // A QSA region's kernels: its prompt attention runs on the tensor cores, and its decode rows on the flash
                // kernels (sm_80 as well).
                _qsaScores = module.GetFunction("ts_graph_qsa_scores_f32");
                _qsaSelect = module.GetFunction("ts_graph_qsa_select_i32");
                _qsaAttention = module.GetFunction("ts_graph_qsa_attn_mma_f16_d256");
                TakesQsaAttention = true;
            }
        }

        /// <summary>Whether this image runs a QSA region's kernels (<see cref="LaunchQsaScores"/>, <see cref="LaunchQsaSelect"/>
        /// and <see cref="LaunchQsaAttention"/>, whose sm_80 body is in it).</summary>
        public bool TakesQsaAttention { get; }

        /// <summary>The head width <see cref="LaunchQsaAttention"/> takes, and the query heads of a KV head (the rows of its
        /// 16-row mma tile).</summary>
        public const int QsaAttentionHeadDim = 256, QsaAttentionMaxGroup = 16;

        /// <summary>Most cells a row of <see cref="LaunchQsaSelect"/> lists: its picks' whole blocks and its tail block's cells.</summary>
        public static int QsaCellStride(int picks, int ratio) => picks * ratio + ratio - 1;

        /// <summary>A QSA region's block scores (see the kernel's contract): q [rows][heads][d] F32 (d &lt;= 256), the indexer
        /// cache's rows <paramref name="cachePitch"/> halves apart with each block's pooled key at offset d of its last member's
        /// row (F16), the rows' KV indices; out [rows][pools], -inf where a row cannot see the block.</summary>
        public void LaunchQsaScores(IntPtr q, IntPtr cache, long cachePitch, IntPtr positions, IntPtr output, float weight, int d, int heads,
            int rows, int pools, int ratio, IntPtr stream)
        {
            void** args = stackalloc void*[] { &q, &cache, &cachePitch, &positions, &output, &weight, &d, &heads, &pools, &ratio };
            CudaDriverApi.cuLaunchKernel(_qsaScores, (uint)((pools + 7) / 8), (uint)rows, 1, 256, 1, 1, (uint)(heads * d * 4), stream,
                (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>A QSA region's selection from its scores [rows][pools]: each row's attended cells, ascending, into
        /// cells [rows][<paramref name="cellStride"/>] (at least <see cref="QsaCellStride"/>) and their number into count [rows].</summary>
        public void LaunchQsaSelect(IntPtr scores, IntPtr positions, IntPtr cells, IntPtr count, int rows, int pools, int ratio, int picks,
            int cellStride, IntPtr stream)
        {
            void** args = stackalloc void*[] { &scores, &positions, &cells, &count, &pools, &ratio, &picks, &cellStride };
            CudaDriverApi.cuLaunchKernel(_qsaSelect, (uint)rows, 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>A QSA region's prompt rows over their selected cells on the tensor cores: q [rows][heads][256] F32 (16-byte
        /// aligned), the head-first F16 caches k, v [kvHeads][cap][256], the cells and counts <see cref="LaunchQsaSelect"/>
        /// wrote; out [rows][heads][256] F32. heads / kvHeads &lt;= <see cref="QsaAttentionMaxGroup"/>.</summary>
        public void LaunchQsaAttention(IntPtr q, IntPtr k, IntPtr v, IntPtr cells, IntPtr count, IntPtr output, int cap, int rows, int heads,
            int kvHeads, float scale, int cellStride, IntPtr stream)
        {
            void** args = stackalloc void*[] { &q, &k, &v, &cells, &count, &output, &cap, &heads, &kvHeads, &scale, &cellStride };
            CudaDriverApi.cuLaunchKernel(_qsaAttention, (uint)kvHeads, (uint)rows, 1, 128, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>Weight rows of a tile of the tiled Q6_K head layout, and rows of activations one product takes at most.</summary>
        public const int Q6kTileRows = 64, Q6kTiledMaxRows = 64;

        /// <summary>Whether this image runs <see cref="LaunchGemmQ6kTiled"/> (its sm_80 bodies are in it).</summary>
        public bool TakesTiledQ6kHeads { get; }

        /// <summary>Whether this image runs <see cref="LaunchGemmQ4k"/> (its sm_80 bodies are in it).</summary>
        public bool TakesQ4kMma { get; }

        /// <summary>Activation rows <see cref="LaunchGemmQ4k"/> takes in one launch.</summary>
        public const int Q4kMmaMaxRows = 64;

        /// <summary>y [n][outDim] = xq [n] W^T for a Q4_K weight (rows <paramref name="rowBytes"/> apart, k a multiple of 256)
        /// on the int8 tensor cores, the direct engines' drafter product: <paramref name="xq"/> int8 [n][k] and
        /// <paramref name="xd"/> [n][k / 32] (the split q8_1 rows). K splits until the row tiles fill the device twice over (at
        /// least four superblocks a split), their shares added atomically: the last bits vary from run to run.</summary>
        public void LaunchGemmQ4k(IntPtr w, long rowBytes, int outDim, int k, IntPtr xq, IntPtr xd, int n, IntPtr y, IntPtr stream)
        {
            uint tiles = (uint)((outDim + 63) / 64);
            int superblocks = k / 256, splits = 1;
            while (tiles * splits < 2 * _multiprocessors && splits * 4 <= superblocks)
                splits *= 2;
            if (splits > 1)
                CudaDriverApi.cuMemsetD32Async(y, 0, new UIntPtr((ulong)n * (ulong)outDim), stream).ThrowOnError();
            IntPtr a4 = IntPtr.Zero, a5 = IntPtr.Zero, a7 = IntPtr.Zero;
            int rows = 0;
            void** args = stackalloc void*[] { &w, &rowBytes, &outDim, &k, &a4, &a5, &rows, &a7 };
            for (int r0 = 0; r0 < n; r0 += Q4kMmaMaxRows)
            {
                rows = Math.Min(Q4kMmaMaxRows, n - r0);
                int i = rows <= 8 ? 0 : rows <= 16 ? 1 : rows <= 32 ? 2 : 3;
                a4 = xq + (nint)((long)r0 * k);
                a5 = xd + (nint)((long)r0 * (k / 32) * 4);
                a7 = y + (nint)((long)r0 * outDim * 4);
                CudaDriverApi.cuLaunchKernel(_gemmQ4k[i], tiles, (uint)splits, 1, 128, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
            }
        }

        /// <summary>A Q6_K matrix's stored rows [rows][srcRowBytes] (rows rowBase .. of the head) into the tiled layout at
        /// <paramref name="dst"/> (the head's base); rows of whole 64-row tiles land on the same bytes they came from.</summary>
        public void LaunchRepackQ6kTiled(IntPtr src, long srcRowBytes, int rows, int k, int rowBase, IntPtr dst, IntPtr stream)
        {
            void** args = stackalloc void*[] { &src, &srcRowBytes, &rows, &k, &rowBase, &dst };
            long threads = (long)rows * (k / 256);
            CudaDriverApi.cuLaunchKernel(_repackQ6kTiled, (uint)((threads + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>y [n][outDim] = xq [n] W^T for a tiled Q6_K head: <paramref name="xq"/> int8 [n][k] and <paramref name="xd"/>
        /// [n][k / 32] (the split q8_1 rows). Up to 64 rows a launch on the int8 tensor cores, the narrowest of 8 / 16 / 32 / 64
        /// that holds them; a row's logits are the same in any of them.</summary>
        public void LaunchGemmQ6kTiled(IntPtr w, int outDim, int k, IntPtr xq, IntPtr xd, int n, IntPtr y, IntPtr stream)
        {
            IntPtr a3 = IntPtr.Zero, a4 = IntPtr.Zero, a6 = IntPtr.Zero;
            int rows = 0;
            void** args = stackalloc void*[] { &w, &outDim, &k, &a3, &a4, &rows, &a6 };
            for (int r0 = 0; r0 < n; r0 += Q6kTiledMaxRows)
            {
                rows = Math.Min(Q6kTiledMaxRows, n - r0);
                int i = rows <= 8 ? 0 : rows <= 16 ? 1 : rows <= 32 ? 2 : 3;
                a3 = xq + (nint)((long)r0 * k);
                a4 = xd + (nint)((long)r0 * (k / 32) * 4);
                a6 = y + (nint)((long)r0 * outDim * 4);
                CudaDriverApi.cuLaunchKernel(_gemmQ6kTiled[i], (uint)((outDim + Q6kTileRows - 1) / Q6kTileRows), 1, 1, 128, 1, 1, (uint)_smemQ6kTiled[i],
                    stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
            }
        }

        /// <summary>The current module, or false with the reason it cannot be used (requires a current context).</summary>
        public static bool TryCreate(out CudaGraphKernels kernels, out string reason)
            => TryCreate(PtxFileName, m => new CudaGraphKernels(m), out kernels, out reason);

        /// <summary>ggml_gated_delta_net over contiguous F32 device buffers (see the kernel's contract);
        /// <paramref name="s"/> % 32 == 0 and <= 256.</summary>
        public void LaunchGatedDeltaNet(IntPtr q, IntPtr k, IntPtr v, IntPtr g, IntPtr beta, IntPtr state, IntPtr output,
            int s, int heads, int keyHeads, int tokens, int sequences, int slots, bool perKeyDecay, IntPtr stream, IntPtr saved = default,
            IntPtr inputs = default, int inputRows = 0)
        {
            int kda = perKeyDecay ? 1 : 0;
            void** args = stackalloc void*[] { &q, &k, &v, &g, &beta, &state, &output, &s, &heads, &keyHeads, &tokens, &sequences, &slots, &kda,
                &saved, &inputs, &inputRows };
            CudaDriverApi.cuLaunchKernel(_gatedDeltaNet, (uint)(sequences * heads), (uint)((s + 3) / 4), 1, 128, 1, 1, 0, stream,
                (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary><see cref="LaunchGatedDeltaNet"/> with four state rows a warp (a prompt chunk reads each token's q and k
        /// once for four rows); the same results. With <paramref name="saved"/> ([heads, s, s], one sequence) the state before
        /// the rows is written there as it is read; with <paramref name="inputs"/> (<see cref="GatedDeltaNetInputFloats"/>) the
        /// rows' q, k, v, gates and betas in planar regions of <paramref name="inputRows"/> tokens, for a later re-step. The
        /// state may be the output's state part (one slot, one sequence).</summary>
        public void LaunchGatedDeltaNetRows4(IntPtr q, IntPtr k, IntPtr v, IntPtr g, IntPtr beta, IntPtr state, IntPtr output,
            int s, int heads, int keyHeads, int tokens, int sequences, int slots, bool perKeyDecay, IntPtr stream, IntPtr saved = default,
            IntPtr inputs = default, int inputRows = 0)
        {
            int kda = perKeyDecay ? 1 : 0;
            void** args = stackalloc void*[] { &q, &k, &v, &g, &beta, &state, &output, &s, &heads, &keyHeads, &tokens, &sequences, &slots, &kda,
                &saved, &inputs, &inputRows };
            CudaDriverApi.cuLaunchKernel(_gatedDeltaNetRows4, (uint)(sequences * heads), (uint)((s + 15) / 16), 1, 128, 1, 1, 0, stream,
                (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>Floats of a recurrence's saved inputs for <paramref name="rows"/> tokens: q and k [rows, keyHeads, s], v
        /// [rows, heads, s], the gates [rows, heads, s or 1] and betas [rows, heads], each a region of its own.</summary>
        public static long GatedDeltaNetInputFloats(int rows, int s, int heads, int keyHeads, bool perKeyDecay)
            => (long)rows * (2L * keyHeads * s + (long)heads * s + (long)heads * (perKeyDecay ? s : 1) + heads);

        /// <summary>ggml_softplus over <paramref name="n"/> contiguous floats.</summary>
        public void LaunchSoftplus(IntPtr x, IntPtr y, long n, IntPtr stream)
        {
            void** args = stackalloc void*[] { &x, &y, &n };
            CudaDriverApi.cuLaunchKernel(_softplus, (uint)((n + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>Widest row <see cref="LaunchTopK"/> takes.</summary>
        public const int TopKMaxColumns = 4096;

        /// <summary>ggml_top_k over <paramref name="rows"/> contiguous rows of <paramref name="cols"/> floats: ids [rows, k],
        /// largest first, a tie going to the lower index.</summary>
        public void LaunchTopK(IntPtr x, IntPtr ids, int rows, int cols, int k, IntPtr stream)
        {
            void** args = stackalloc void*[] { &x, &ids, &cols, &k };
            CudaDriverApi.cuLaunchKernel(_topk, (uint)rows, 1, 1, 256, 1, 1, (uint)(cols * 4), stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>Most picks <see cref="LaunchTopKWide"/> takes.</summary>
        public const int TopKWideMaxPicks = 2048;

        /// <summary>ggml_top_k for many picks or wide rows (radix-selected, then sorted): the order and ties of
        /// <see cref="LaunchTopK"/>; <paramref name="k"/> &lt;= <see cref="TopKWideMaxPicks"/>.</summary>
        public void LaunchTopKWide(IntPtr x, IntPtr ids, int rows, int cols, int k, IntPtr stream)
        {
            int n2 = 1;
            while (n2 < k) n2 <<= 1;
            void** args = stackalloc void*[] { &x, &ids, &cols, &k, &n2 };
            CudaDriverApi.cuLaunchKernel(_topkWide, (uint)rows, 1, 1, 1024, 1, 1, (uint)(n2 * 8), stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>Columns each block of <see cref="LaunchArgmax"/>'s first stage takes.</summary>
        public const int ArgmaxSlice = 4096;

        /// <summary>Bytes of the slices' picks <see cref="LaunchArgmax"/> needs for <paramref name="rows"/> rows.</summary>
        public static long ArgmaxPartBytes(int rows, int cols) => (long)rows * ((cols + ArgmaxSlice - 1) / ArgmaxSlice) * 8;

        /// <summary><see cref="LaunchTopKWide"/> at k = 1 (the same pick: the largest value, a tie to the lower index), over
        /// every SM: the slices' picks into <paramref name="part"/> (<see cref="ArgmaxPartBytes"/>), then a merge a row.</summary>
        public void LaunchArgmax(IntPtr x, IntPtr ids, IntPtr part, int rows, int cols, IntPtr stream)
        {
            int slice = ArgmaxSlice, slices = (cols + slice - 1) / slice;
            void** args = stackalloc void*[] { &x, &part, &cols, &slice };
            CudaDriverApi.cuLaunchKernel(_argmaxPart, (uint)slices, (uint)rows, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
            void** margs = stackalloc void*[] { &part, &ids, &slices };
            CudaDriverApi.cuLaunchKernel(_argmaxMerge, (uint)rows, 1, 1, 32, 1, 1, 0, stream, (IntPtr)margs, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>Most picks <see cref="LaunchTopKFew"/> takes.</summary>
        public const int TopKFewMaxPicks = 16;

        /// <summary>Bytes of the slices' picks <see cref="LaunchTopKFew"/> needs.</summary>
        public static long TopKFewPartBytes(int rows, int cols, int k) => ArgmaxPartBytes(rows, cols) * k;

        /// <summary><see cref="LaunchTopKWide"/> for a few picks (<paramref name="k"/> &lt;= <see cref="TopKFewMaxPicks"/>), over every
        /// SM: each slice's k largest into <paramref name="part"/> (<see cref="TopKFewPartBytes"/>), then a merge a row. The same
        /// picks in the same order.</summary>
        public void LaunchTopKFew(IntPtr x, IntPtr ids, IntPtr part, int rows, int cols, int k, IntPtr stream)
        {
            int slice = ArgmaxSlice, slices = (cols + slice - 1) / slice, picks = slices * k;
            void** args = stackalloc void*[] { &x, &part, &cols, &slice, &k };
            CudaDriverApi.cuLaunchKernel(_topkFewPart, (uint)slices, (uint)rows, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
            void** margs = stackalloc void*[] { &part, &ids, &picks, &k };
            CudaDriverApi.cuLaunchKernel(_topkFewMerge, (uint)rows, 1, 1, 256, 1, 1, 0, stream, (IntPtr)margs, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>ggml_get_rows into F32 rows by device int32 ids: src [batches, rowsPerBatch, rowLen] (F32, or F16 when
        /// <paramref name="half"/>) with rows <paramref name="rowPitch"/> elements apart, ids [batches, n].</summary>
        public void LaunchGetRows(IntPtr src, IntPtr ids, IntPtr output, int rowLen, int n, int rowsPerBatch, int batches, long rowPitch,
            bool half, IntPtr stream)
        {
            void** args = stackalloc void*[] { &src, &ids, &output, &rowLen, &n, &rowsPerBatch, &rowPitch };
            CudaDriverApi.cuLaunchKernel(half ? _getRowsF16 : _getRows, (uint)(n * batches), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args,
                IntPtr.Zero).ThrowOnError();
        }

        /// <summary>ggml_set_rows by device int32 ids: dst row ids[i] (F32, or F16 when <paramref name="half"/>; rows
        /// <paramref name="dstPitch"/> elements apart) takes contiguous F32 src row i.</summary>
        public void LaunchSetRows(IntPtr src, IntPtr ids, IntPtr dst, int count, int rowLen, long dstPitch, int dstRows, bool half, IntPtr stream)
        {
            int h = half ? 1 : 0;
            void** args = stackalloc void*[] { &src, &ids, &dst, &rowLen, &dstPitch, &dstRows, &h };
            CudaDriverApi.cuLaunchKernel(_setRows, (uint)count, 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>Most q floats (heads * d) a row of <see cref="LaunchLightningIndexer"/> holds; d &lt;= 256.</summary>
        public const int LightningIndexerMaxQ = 12 * 1024;

        /// <summary>ggml_lightning_indexer over F32: q [rows, heads, d], keys [n, d], w [rows, heads], mask [maskRows, n];
        /// out [rows, n].</summary>
        public void LaunchLightningIndexer(IntPtr q, IntPtr keys, IntPtr w, IntPtr mask, IntPtr output, int d, int heads, int n, int rows,
            int maskRows, IntPtr stream)
        {
            void** args = stackalloc void*[] { &q, &keys, &w, &mask, &output, &d, &heads, &n, &maskRows };
            CudaDriverApi.cuLaunchKernel(_lightningIndexer, (uint)((n + 7) / 8), (uint)rows, 1, 256, 1, 1, (uint)(heads * d * 4), stream,
                (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>The cells each of <paramref name="rows"/> F32 mask rows of <paramref name="n"/> keeps (above -inf), in
        /// order: cells [rows, n], count [rows].</summary>
        public void LaunchMaskCells(IntPtr mask, IntPtr cells, IntPtr count, int rows, int n, IntPtr stream)
        {
            void** args = stackalloc void*[] { &mask, &cells, &count, &n };
            CudaDriverApi.cuLaunchKernel(_maskCells, (uint)rows, 1, 1, 1024, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>ggml_top_k_mask over F32 (see the kernel's contract); <paramref name="score"/>, <paramref name="blocks"/>
        /// and <paramref name="extra"/> may be zero.</summary>
        public void LaunchTopKMask(IntPtr mask, IntPtr picks, IntPtr score, IntPtr blocks, IntPtr extra, IntPtr output, int rows, int nkv,
            int maskRows, int top, int nPick, int cells, int nBlocks, int nExtra, IntPtr stream)
        {
            void** args = stackalloc void*[] { &mask, &picks, &score, &blocks, &extra, &output, &nkv, &maskRows, &top, &nPick, &cells, &nBlocks, &nExtra };
            CudaDriverApi.cuLaunchKernel(_topkMask, (uint)rows, 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>Convert <paramref name="n"/> contiguous elements between F16 and F32 (<paramref name="toHalf"/>: F32 to F16).</summary>
        public void LaunchConvert(IntPtr x, IntPtr y, long n, bool toHalf, IntPtr stream)
        {
            void** args = stackalloc void*[] { &x, &y, &n };
            CudaDriverApi.cuLaunchKernel(toHalf ? _f32ToF16 : _f16ToF32, (uint)((n + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args,
                IntPtr.Zero).ThrowOnError();
        }

        /// <summary>A layer's routes split between the device (its tier: <paramref name="onDevice"/>[e] == 1, and up to
        /// <paramref name="gpuMisses"/> distinct experts the host holds where the device can read them: onDevice[e] == 2) and the host
        /// (less those at <paramref name="tailRank"/> or lower that no row routes higher: the route-tail skip, 0 off), for
        /// <paramref name="rows"/> rows of <paramref name="perRow"/> routes, each counted in <paramref name="counts"/> (per expert):
        /// the device's into <paramref name="devSel"/> (-1 for a host route); when any route is the host's, the input and
        /// routes published to <paramref name="box"/> and its counter advanced (<paramref name="missDev"/> says which).</summary>
        public void LaunchMoeRouteSplit(IntPtr x, IntPtr sel, IntPtr selW, IntPtr onDevice, int n, int rows, int perRow, int layer,
            int tailRank, IntPtr devSel, IntPtr counts, CudaHostExpertMailbox box, IntPtr missDev, IntPtr stream, in MoeStaging staging = default)
        {
            IntPtr xOut = box.XDev, idsOut = box.IdsDev, wOut = box.WDev, hdr = box.HeaderDev, seq = box.SeqDev;
            int gpuMisses = staging.Capacity;
            IntPtr tg = staging.Gate, tu = staging.Up, td = staging.Down, list = staging.List, count = staging.Count;
            ulong slots = (ulong)staging.Slots.ToInt64();
            long stride = staging.Stride, offUp = staging.OffUp, offDown = staging.OffDown;
            void** args = stackalloc void*[] { &x, &sel, &selW, &onDevice, &n, &rows, &perRow, &layer, &tailRank, &devSel, &counts, &xOut, &idsOut, &wOut,
                &hdr, &seq, &missDev, &gpuMisses, &tg, &tu, &td, &slots, &stride, &offUp, &offDown, &list, &count };
            CudaDriverApi.cuLaunchKernel(_routeSplit, 1, 1, 1, 1024, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>A layer's staging of host experts for its device kernels (<see cref="LaunchMoeRouteSplit"/>): the tables
        /// the split points at the slots and their resting entries (Src*), the slots, and the plan (List, Count). Capacity 0:
        /// the host computes every miss.</summary>
        public readonly struct MoeStaging
        {
            public int Capacity { get; init; }
            public IntPtr Gate { get; init; }
            public IntPtr Up { get; init; }
            public IntPtr Down { get; init; }
            public IntPtr SrcGate { get; init; }
            public IntPtr SrcUp { get; init; }
            public IntPtr SrcDown { get; init; }
            /// <summary>The tables' resting entries the wait restores (where they differ from the copy's sources: the host
            /// experts rest at a zeroed expert while their bytes stay mapped), else zero (the sources).</summary>
            public IntPtr HomeGate { get; init; }
            public IntPtr HomeUp { get; init; }
            public IntPtr HomeDown { get; init; }
            public IntPtr Slots { get; init; }
            public long Stride { get; init; }
            public long OffUp { get; init; }
            public long OffDown { get; init; }
            public long BytesGate { get; init; }
            public long BytesUp { get; init; }
            public long BytesDown { get; init; }
            public IntPtr List { get; init; }
            public IntPtr Count { get; init; }
        }

        /// <summary>The staged experts' bytes into their slots over the link (<see cref="MoeStaging"/>).</summary>
        public void LaunchMoeStageCopy(in MoeStaging staging, IntPtr stream)
        {
            const int chunks = 16;
            IntPtr list = staging.List, count = staging.Count, sg = staging.SrcGate, su = staging.SrcUp, sd = staging.SrcDown, slots = staging.Slots;
            long bg = staging.BytesGate, bu = staging.BytesUp, bd = staging.BytesDown, stride = staging.Stride, offUp = staging.OffUp, offDown = staging.OffDown;
            int c = chunks;
            void** args = stackalloc void*[] { &list, &count, &sg, &su, &sd, &bg, &bu, &bd, &slots, &stride, &offUp, &offDown, &c };
            CudaDriverApi.cuLaunchKernel(_stageCopy, (uint)(staging.Capacity * chunks), 3, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>Wait for the host's answer to the layer's doorbell (when one rang) and add its sum into
        /// <paramref name="dst"/> (<paramref name="n"/> floats); with a staging, its experts' table entries restored first.</summary>
        public void LaunchMoeWaitAdd(CudaHostExpertMailbox box, IntPtr dst, int n, IntPtr missDev, IntPtr stream, in MoeStaging staging = default)
        {
            IntPtr flag = box.FlagDev, seq = box.SeqDev, count = box.CountDev, outDev = box.OutDev;
            IntPtr list = staging.Capacity > 0 ? staging.List : IntPtr.Zero, staged = staging.Count;
            IntPtr tg = staging.Gate, tu = staging.Up, td = staging.Down;
            IntPtr sg = staging.HomeGate != IntPtr.Zero ? staging.HomeGate : staging.SrcGate;
            IntPtr su = staging.HomeUp != IntPtr.Zero ? staging.HomeUp : staging.SrcUp;
            IntPtr sd = staging.HomeDown != IntPtr.Zero ? staging.HomeDown : staging.SrcDown;
            void** args = stackalloc void*[] { &flag, &seq, &count, &outDev, &dst, &n, &missDev, &list, &staged, &tg, &tu, &td, &sg, &su, &sd };
            CudaDriverApi.cuLaunchKernel(_waitAdd, (uint)Math.Min(16, Math.Max(1, (n + 255) / 256)), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args,
                IntPtr.Zero).ThrowOnError();
        }

        /// <summary>The hyper-connection read's grouped norm: each of <paramref name="rows"/> x <paramref name="hc"/> streams
        /// of <paramref name="n"/> RMS-normed, times the [hc * n] gammas. With <paramref name="injW"/> (the injection weights
        /// [hc, hc * n] as F32, hc at most 8) each stream's slice of the normed row is also dotted with every row's slice into
        /// <paramref name="injPart"/> [rows * hc, hc], which <see cref="LaunchHcGateMix"/> sums into the injection logits.</summary>
        public void LaunchHcNorm(IntPtr x, IntPtr gamma, IntPtr y, int rows, int n, int hc, float eps, IntPtr stream, IntPtr yq = default,
            IntPtr injW = default, IntPtr injPart = default)
        {
            void** args = stackalloc void*[] { &x, &gamma, &y, &n, &hc, &eps, &yq, &injW, &injPart };
            // Few streams (a decode or verify step): wide blocks; a prompt chunk's many streams fill the device at 256.
            uint threads = rows * hc < 64 ? 1024u : 256u;
            CudaDriverApi.cuLaunchKernel(_hcNorm, (uint)(rows * hc), 1, 1, threads, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>Slices of 256 values <see cref="LaunchHcPartials"/> and <see cref="LaunchHcApply"/> split a stream into, and the
        /// floats each slice's partials take.</summary>
        public const int HcSlice = 256, HcPartialFloats = 9;

        /// <summary>The decode-size grouped norm's first pass (<paramref name="n"/> % <see cref="HcSlice"/> == 0, hc at most 8):
        /// per row and slice of 256 values, the sum of squares and, with <paramref name="injW"/>, the injection weights' partial
        /// dots of x * gamma, into <paramref name="partials"/> [rows, hc * n / 256, <see cref="HcPartialFloats"/>].</summary>
        public void LaunchHcPartials(IntPtr x, IntPtr gamma, IntPtr injW, IntPtr partials, int rows, int n, int hc, IntPtr stream)
        {
            void** args = stackalloc void*[] { &x, &gamma, &injW, &partials, &n, &hc };
            CudaDriverApi.cuLaunchKernel(_hcPartials, (uint)(hc * n / HcSlice), (uint)rows, 1, HcSlice, 1, 1, 0, stream, (IntPtr)args,
                IntPtr.Zero).ThrowOnError();
        }

        /// <summary>The second pass: <paramref name="y"/> the normed streams (and <paramref name="yq"/> their q8_1 copy unless
        /// zero) from <see cref="LaunchHcPartials"/>' partials; with <paramref name="inject"/> also the injection logits [rows, hc].</summary>
        public void LaunchHcApply(IntPtr x, IntPtr gamma, IntPtr partials, IntPtr y, IntPtr yq, IntPtr inject, int rows, int n, int hc,
            float eps, IntPtr stream)
        {
            void** args = stackalloc void*[] { &x, &gamma, &partials, &y, &yq, &inject, &n, &hc, &eps };
            CudaDriverApi.cuLaunchKernel(_hcApply, (uint)(hc * n / HcSlice), (uint)rows, 1, HcSlice, 1, 1, 0, stream, (IntPtr)args,
                IntPtr.Zero).ThrowOnError();
        }

        /// <summary>y = silu(x * scale) over <paramref name="n"/> contiguous floats; with <paramref name="yq"/> (n a multiple of
        /// 32) y also as q8_1, as <see cref="CudaKernels"/>' row quantizer writes it.</summary>
        public void LaunchScaleSilu(IntPtr x, IntPtr y, long n, float scale, IntPtr stream, IntPtr yq = default)
        {
            void** args = stackalloc void*[] { &x, &y, &n, &scale, &yq };
            CudaDriverApi.cuLaunchKernel(_scaleSilu, (uint)((n + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>The read's mix: each row's streams' mean of xn * sigmoid(gates); with <paramref name="outq"/> (n a multiple
        /// of 32) also as q8_1. With <paramref name="injPart"/> (<see cref="LaunchHcNorm"/>'s) also the injection logits
        /// <paramref name="inject"/> [rows, hc], each the sum of its row's streams' partial dots in stream order.</summary>
        public void LaunchHcGateMix(IntPtr xn, IntPtr gates, IntPtr output, int rows, int n, int hc, IntPtr stream, IntPtr outq = default,
            IntPtr injPart = default, IntPtr inject = default)
        {
            void** args = stackalloc void*[] { &xn, &gates, &output, &rows, &n, &hc, &outq, &injPart, &inject };
            long total = (long)rows * n;
            CudaDriverApi.cuLaunchKernel(_hcGateMix, (uint)((total + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>The scatter: each stream plus the block output weighted by 2 * sigmoid(inject / hc).</summary>
        /// <summary>A block's scatter into <paramref name="output"/> (<see cref="LaunchHcScatter"/>'s arithmetic) and the next
        /// read's first pass over the scattered streams (<see cref="LaunchHcPartials"/>' arithmetic) in one launch.</summary>
        public void LaunchHcPostPartials(IntPtr streamIn, IntPtr block, IntPtr inject, IntPtr output, IntPtr gamma, IntPtr injW,
            IntPtr partials, int rows, int n, int hc, IntPtr stream)
        {
            void** args = stackalloc void*[] { &streamIn, &block, &inject, &output, &gamma, &injW, &partials, &n, &hc };
            CudaDriverApi.cuLaunchKernel(_hcPostPartials, (uint)(hc * n / HcSlice), (uint)rows, 1, HcSlice, 1, 1, 0, stream, (IntPtr)args,
                IntPtr.Zero).ThrowOnError();
        }

        /// <summary>ts_rope_ex_f32's rotary over a strided source (row r head r % heads of token r / heads, at token and head
        /// strides in floats) into contiguous rows: the rotated values that kernel writes.</summary>
        public void LaunchRopeExStrided(IntPtr input, long tokenStride, long headStride, int heads, IntPtr positions, IntPtr output, int rows,
            int cols, int ropeDim, int mode, bool positionsAreInt32, int ctx, float freqBase, float freqScale, float extFactor, float attnFactor,
            float betaFast, float betaSlow, IntPtr stream)
        {
            int ints = positionsAreInt32 ? 1 : 0, add = 0;
            void** args = stackalloc void*[] { &input, &tokenStride, &headStride, &heads, &positions, &output, &rows, &cols, &ropeDim, &mode, &ints,
                &ctx, &freqBase, &freqScale, &extFactor, &attnFactor, &betaFast, &betaSlow, &add };
            long work = Math.Max((long)rows * (Math.Min(ropeDim, cols) / 2), (long)rows * cols);
            CudaDriverApi.cuLaunchKernel(_ropeStrided, (uint)((work + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        public void LaunchHcScatter(IntPtr streamIn, IntPtr block, IntPtr inject, IntPtr output, int rows, int n, int hc, IntPtr stream)
        {
            void** args = stackalloc void*[] { &streamIn, &block, &inject, &output, &rows, &n, &hc };
            long total = (long)rows * hc * n;
            CudaDriverApi.cuLaunchKernel(_hcScatter, (uint)((total + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>ggml_sum_rows over <paramref name="rows"/> contiguous rows of <paramref name="cols"/> floats.</summary>
        public void LaunchSumRows(IntPtr x, IntPtr y, int rows, int cols, IntPtr stream)
        {
            void** args = stackalloc void*[] { &x, &y, &cols };
            CudaDriverApi.cuLaunchKernel(_sumRows, (uint)rows, 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>y [rows, outs] = x [rows, k] (rows <paramref name="xPitch"/> floats apart) times the F32 weight
        /// [outs, k] transposed, one block an output and eight rows; for products with few outputs and rows.</summary>
        /// <summary>moe + shared * sigmoid(x . w) per row (GraphFusion.SharedExpertGate), the dot summed as
        /// <see cref="LaunchSmallMatvec"/> sums it.</summary>
        public void LaunchSharedGateAdd(IntPtr moe, IntPtr shared, IntPtr x, IntPtr w, IntPtr output, int rows, int k, long xPitch, int n, IntPtr stream)
        {
            void** args = stackalloc void*[] { &moe, &shared, &x, &w, &output, &rows, &k, &xPitch, &n };
            uint threads = (uint)Math.Clamp(((k + 3) / 4 + 31) / 32 * 32, 32, 1024);
            CudaDriverApi.cuLaunchKernel(_sharedGate, 1, (uint)((rows + 7) / 8), 1, threads, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        public void LaunchSmallMatvec(IntPtr x, IntPtr w, IntPtr y, int rows, int k, int outs, long xPitch, IntPtr stream)
        {
            void** args = stackalloc void*[] { &x, &w, &y, &rows, &k, &outs, &xPitch };
            // Threads across k: a float4 each, up to 1024.
            uint threads = (uint)Math.Clamp(((k + 3) / 4 + 31) / 32 * 32, 32, 1024);
            CudaDriverApi.cuLaunchKernel(_smallMatvec, (uint)outs, (uint)((rows + 7) / 8), 1, threads, 1, 1, 0, stream, (IntPtr)args,
                IntPtr.Zero).ThrowOnError();
        }

        /// <summary>counts[ids[i]] += 1 for the <paramref name="n"/> route ids in [0, <paramref name="experts"/>).</summary>
        public void LaunchCountRoutes(IntPtr ids, int n, IntPtr counts, int experts, IntPtr stream)
        {
            void** args = stackalloc void*[] { &ids, &n, &counts, &experts };
            CudaDriverApi.cuLaunchKernel(_countRoutes, (uint)((n + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>out = a op b over a's <paramref name="na"/> extents (ne0 first), b repeating where its
        /// <paramref name="nb"/> extent is 1; op 0 add, 1 sub, 2 mul, 3 div; contiguous F32.</summary>
        public void LaunchBinaryBroadcast(IntPtr a, IntPtr b, IntPtr output, long[] na, long[] nb, int op, IntPtr stream)
        {
            long n = na[0] * na[1] * na[2] * na[3];
            long a0 = na[0], a1 = na[1], a2 = na[2], b0 = nb[0], b1 = nb[1], b2 = nb[2], b3 = nb[3];
            void** args = stackalloc void*[] { &a, &b, &output, &n, &a0, &a1, &a2, &b0, &b1, &b2, &b3, &op };
            CudaDriverApi.cuLaunchKernel(_binaryBcast, (uint)((n + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>ggml_sqrt over <paramref name="n"/> contiguous floats.</summary>
        public void LaunchSqrt(IntPtr x, IntPtr y, long n, IntPtr stream)
        {
            void** args = stackalloc void*[] { &x, &y, &n };
            CudaDriverApi.cuLaunchKernel(_sqrt, (uint)((n + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>ggml_ssm_conv: x [n_s, channels, length] (time innermost), w [channels, width] (contiguous):
        /// out [n_s, length - width + 1, channels].</summary>
        public void LaunchSsmConv(IntPtr x, IntPtr w, IntPtr output, int channels, int length, int width, int sequences, IntPtr stream)
        {
            int nt = length - width + 1;
            long total = (long)sequences * nt * channels;
            void** args = stackalloc void*[] { &x, &w, &output, &channels, &length, &width, &nt, &sequences };
            CudaDriverApi.cuLaunchKernel(_ssmConv, (uint)((total + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>Tokens a block of the tiled DeltaNet prep takes (TS_GRAPH_PREP_TILE).</summary>
        internal const int DeltaNetPrepTile = 32;

        /// <summary>
        /// GraphFusion.DeltaNetPrep over contiguous F32 buffers (see the kernel's contract). A prompt chunk (a tile of rows or
        /// more, heads of up to 256, a conv of up to 4 taps) runs on the tiled kernel, the same values bit for bit: its conv
        /// inputs read once and conv_in written along time (the per-row kernel wrote it a float per channel row).
        /// </summary>
        public void LaunchDeltaNetPrep(IntPtr window, IntPtr qkv, IntPtr convW, IntPtr alpha, IntPtr beta, IntPtr dt, IntPtr a,
            IntPtr convIn, IntPtr q, IntPtr k, IntPtr v, IntPtr g, IntPtr betaOut,
            int channels, int convRows, int width, int d, int kHeads, int vHeads, int rows, float eps, long qkvPitch, long alphaBetaPitch,
            IntPtr stream)
            => LaunchDeltaNetPrep(window, qkv, convW, alpha, beta, dt, a, convIn, q, k, v, g, betaOut, channels, convRows, width, d, kHeads,
                vHeads, rows, eps, qkvPitch, alphaBetaPitch, stream, TakesTiledDeltaNetPrep(convRows, width, d, rows));

        /// <summary>Whether <see cref="LaunchDeltaNetPrep(IntPtr, IntPtr, IntPtr, IntPtr, IntPtr, IntPtr, IntPtr, IntPtr, IntPtr, IntPtr,
        /// IntPtr, IntPtr, IntPtr, int, int, int, int, int, int, int, float, long, long, IntPtr)"/> takes the tiled kernel.</summary>
        internal static bool TakesTiledDeltaNetPrep(int convRows, int width, int d, int rows)
            => rows >= DeltaNetPrepTile && convRows == width - 1 && width <= 4 && d % 32 == 0 && d <= 256;

        /// <summary>The per-row kernel, or the tiled one (<paramref name="tiled"/>; tests compare the two).</summary>
        internal void LaunchDeltaNetPrep(IntPtr window, IntPtr qkv, IntPtr convW, IntPtr alpha, IntPtr beta, IntPtr dt, IntPtr a,
            IntPtr convIn, IntPtr q, IntPtr k, IntPtr v, IntPtr g, IntPtr betaOut,
            int channels, int convRows, int width, int d, int kHeads, int vHeads, int rows, float eps, long qkvPitch, long alphaBetaPitch,
            IntPtr stream, bool tiled)
        {
            void** args = stackalloc void*[]
            {
                &window, &qkv, &convW, &alpha, &beta, &dt, &a, &convIn, &q, &k, &v, &g, &betaOut,
                &channels, &convRows, &width, &d, &kHeads, &vHeads, &rows, &eps, &qkvPitch, &alphaBetaPitch,
            };
            if (tiled)
            {
                uint smem = (uint)(((DeltaNetPrepTile + width - 1) * (d + 1) + DeltaNetPrepTile * d / 32) * sizeof(float));
                CudaDriverApi.cuLaunchKernel(_deltaNetPrepTiled, (uint)(2 * kHeads + vHeads), (uint)((rows + DeltaNetPrepTile - 1) / DeltaNetPrepTile),
                    1, (uint)d, 1, 1, smem, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
                return;
            }
            CudaDriverApi.cuLaunchKernel(_deltaNetPrep, (uint)(2 * kHeads + vHeads), (uint)rows, 1, (uint)d, 1, 1, 0, stream,
                (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>GraphFusion.SigmoidGate: output = attn * sigmoid(gate) over <paramref name="rows"/> x <paramref name="heads"/>
        /// x <paramref name="d"/> (attn and output contiguous), gate's heads <paramref name="gateHead"/> floats and its rows
        /// <paramref name="gateRow"/> floats apart.</summary>
        public void LaunchSigmoidGate(IntPtr attn, IntPtr gate, IntPtr output, long rows, int heads, int d, long gateRow, long gateHead,
            IntPtr stream, IntPtr outputQ81 = default)
        {
            long n = rows * heads * d;
            if (outputQ81 != IntPtr.Zero && n % 32 != 0) throw new ArgumentException("a q8_1 copy needs whole 32-value blocks");
            void** args = stackalloc void*[] { &attn, &gate, &output, &heads, &d, &gateRow, &gateHead, &n, &outputQ81 };
            CudaDriverApi.cuLaunchKernel(_sigmoidGate, (uint)((n + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>sum = a + b and normed = rms_norm(sum) * w over <paramref name="rows"/> contiguous rows of
        /// <paramref name="cols"/>, one pass.</summary>
        public void LaunchAddRmsNorm(IntPtr a, IntPtr b, IntPtr w, IntPtr sum, IntPtr normed, int rows, int cols, float eps, IntPtr stream)
        {
            void** args = stackalloc void*[] { &a, &b, &w, &sum, &normed, &cols, &eps };
            CudaDriverApi.cuLaunchKernel(_addRmsNorm, (uint)rows, 1, 1, 512, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>GraphFusion.GatedRmsNorm over <paramref name="rows"/> rows of <paramref name="d"/> (head r % heads of token
        /// r / heads); z's tokens are <paramref name="zPitch"/> floats apart.</summary>
        public void LaunchGatedRmsNorm(IntPtr y, IntPtr z, IntPtr w, IntPtr output, int rows, int d, float eps, int heads, long zPitch,
            IntPtr stream, bool sigmoidGate = false, IntPtr outputQ81 = default)
        {
            int gate = sigmoidGate ? 1 : 0;
            void** args = stackalloc void*[] { &y, &z, &w, &output, &d, &eps, &heads, &zPitch, &gate, &outputQ81 };
            CudaDriverApi.cuLaunchKernel(_gatedRmsNorm, (uint)rows, 1, 1, (uint)d, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }
    }
}
