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
        public const int CurrentVersion = 14;

        private readonly IntPtr _gatedDeltaNet, _gatedDeltaNetRows4, _softplus, _sqrt, _topk, _topkWide, _argmaxPart, _argmaxMerge, _getRows, _getRowsF16, _setRows, _lightningIndexer, _topkMask, _maskCells, _sumRows, _smallMatvec, _countRoutes, _binaryBcast, _hcNorm, _scaleSilu, _hcGateMix, _hcScatter, _f16ToF32, _f32ToF16, _routeSplit, _waitAdd, _ssmConv, _deltaNetPrep, _gatedRmsNorm, _addRmsNorm;

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
            _scaleSilu = module.GetFunction("ts_graph_scale_silu_f32");
            _hcGateMix = module.GetFunction("ts_graph_hc_gate_mix_f32");
            _hcScatter = module.GetFunction("ts_graph_hc_scatter_f32");
            _f16ToF32 = module.GetFunction("ts_graph_f16_to_f32");
            _f32ToF16 = module.GetFunction("ts_graph_f32_to_f16");
            _routeSplit = module.GetFunction("ts_graph_moe_route_split");
            _waitAdd = module.GetFunction("ts_graph_moe_wait_add");
            CudaDriverApi.cuFuncSetAttribute(_topk, 8 /* MAX_DYNAMIC_SHARED_SIZE_BYTES */, TopKMaxColumns * 4).ThrowOnError();
            _ssmConv = module.GetFunction("ts_graph_ssm_conv_f32");
            _deltaNetPrep = module.GetFunction("ts_graph_deltanet_prep_f32");
            _gatedRmsNorm = module.GetFunction("ts_graph_gated_rms_norm_f32");
            _addRmsNorm = module.GetFunction("ts_graph_add_rms_norm_f32");
        }

        /// <summary>The current module, or false with the reason it cannot be used (requires a current context).</summary>
        public static bool TryCreate(out CudaGraphKernels kernels, out string reason)
            => TryCreate(PtxFileName, m => new CudaGraphKernels(m), out kernels, out reason);

        /// <summary>ggml_gated_delta_net over contiguous F32 device buffers (see the kernel's contract);
        /// <paramref name="s"/> % 32 == 0 and <= 256.</summary>
        public void LaunchGatedDeltaNet(IntPtr q, IntPtr k, IntPtr v, IntPtr g, IntPtr beta, IntPtr state, IntPtr output,
            int s, int heads, int keyHeads, int tokens, int sequences, int slots, bool perKeyDecay, IntPtr stream)
        {
            int kda = perKeyDecay ? 1 : 0;
            void** args = stackalloc void*[] { &q, &k, &v, &g, &beta, &state, &output, &s, &heads, &keyHeads, &tokens, &sequences, &slots, &kda };
            CudaDriverApi.cuLaunchKernel(_gatedDeltaNet, (uint)(sequences * heads), (uint)((s + 3) / 4), 1, 128, 1, 1, 0, stream,
                (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary><see cref="LaunchGatedDeltaNet"/> with four state rows a warp (a prompt chunk reads each token's q and k
        /// once for four rows); the same results.</summary>
        public void LaunchGatedDeltaNetRows4(IntPtr q, IntPtr k, IntPtr v, IntPtr g, IntPtr beta, IntPtr state, IntPtr output,
            int s, int heads, int keyHeads, int tokens, int sequences, int slots, bool perKeyDecay, IntPtr stream)
        {
            int kda = perKeyDecay ? 1 : 0;
            void** args = stackalloc void*[] { &q, &k, &v, &g, &beta, &state, &output, &s, &heads, &keyHeads, &tokens, &sequences, &slots, &kda };
            CudaDriverApi.cuLaunchKernel(_gatedDeltaNetRows4, (uint)(sequences * heads), (uint)((s + 15) / 16), 1, 128, 1, 1, 0, stream,
                (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

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

        /// <summary>A layer's routes split between the device (its tier: <paramref name="onDevice"/>[e] != 0) and the host
        /// (less those at <paramref name="tailRank"/> or lower that no row routes higher: the route-tail skip, 0 off), for
        /// <paramref name="rows"/> rows of <paramref name="perRow"/> routes, each counted in <paramref name="counts"/> (per expert):
        /// the device's into <paramref name="devSel"/> (-1 for a host route); when any route is the host's, the input and
        /// routes published to <paramref name="box"/> and its counter advanced (<paramref name="missDev"/> says which).</summary>
        public void LaunchMoeRouteSplit(IntPtr x, IntPtr sel, IntPtr selW, IntPtr onDevice, int n, int rows, int perRow, int layer,
            int tailRank, IntPtr devSel, IntPtr counts, CudaHostExpertMailbox box, IntPtr missDev, IntPtr stream)
        {
            IntPtr xOut = box.XDev, idsOut = box.IdsDev, wOut = box.WDev, hdr = box.HeaderDev, seq = box.SeqDev;
            void** args = stackalloc void*[] { &x, &sel, &selW, &onDevice, &n, &rows, &perRow, &layer, &tailRank, &devSel, &counts, &xOut, &idsOut, &wOut, &hdr, &seq, &missDev };
            CudaDriverApi.cuLaunchKernel(_routeSplit, 1, 1, 1, 1024, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>Wait for the host's answer to the layer's doorbell (when one rang) and add its sum into
        /// <paramref name="dst"/> (<paramref name="n"/> floats).</summary>
        public void LaunchMoeWaitAdd(CudaHostExpertMailbox box, IntPtr dst, int n, IntPtr missDev, IntPtr stream)
        {
            IntPtr flag = box.FlagDev, seq = box.SeqDev, count = box.CountDev, outDev = box.OutDev;
            void** args = stackalloc void*[] { &flag, &seq, &count, &outDev, &dst, &n, &missDev };
            CudaDriverApi.cuLaunchKernel(_waitAdd, (uint)Math.Min(16, Math.Max(1, (n + 255) / 256)), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args,
                IntPtr.Zero).ThrowOnError();
        }

        /// <summary>The hyper-connection read's grouped norm: each of <paramref name="rows"/> x <paramref name="hc"/> streams
        /// of <paramref name="n"/> RMS-normed, times the [hc * n] gammas.</summary>
        public void LaunchHcNorm(IntPtr x, IntPtr gamma, IntPtr y, int rows, int n, int hc, float eps, IntPtr stream, IntPtr yq = default)
        {
            void** args = stackalloc void*[] { &x, &gamma, &y, &n, &hc, &eps, &yq };
            // Few streams (a decode or verify step): wide blocks; a prompt chunk's many streams fill the device at 256.
            uint threads = rows * hc < 64 ? 1024u : 256u;
            CudaDriverApi.cuLaunchKernel(_hcNorm, (uint)(rows * hc), 1, 1, threads, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>y = silu(x * scale) over <paramref name="n"/> contiguous floats; with <paramref name="yq"/> (n a multiple of
        /// 32) y also as q8_1, as <see cref="CudaKernels"/>' row quantizer writes it.</summary>
        public void LaunchScaleSilu(IntPtr x, IntPtr y, long n, float scale, IntPtr stream, IntPtr yq = default)
        {
            void** args = stackalloc void*[] { &x, &y, &n, &scale, &yq };
            CudaDriverApi.cuLaunchKernel(_scaleSilu, (uint)((n + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>The read's mix: each row's streams' mean of xn * sigmoid(gates); with <paramref name="outq"/> (n a multiple
        /// of 32) also as q8_1.</summary>
        public void LaunchHcGateMix(IntPtr xn, IntPtr gates, IntPtr output, int rows, int n, int hc, IntPtr stream, IntPtr outq = default)
        {
            void** args = stackalloc void*[] { &xn, &gates, &output, &rows, &n, &hc, &outq };
            long total = (long)rows * n;
            CudaDriverApi.cuLaunchKernel(_hcGateMix, (uint)((total + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>The scatter: each stream plus the block output weighted by 2 * sigmoid(inject / hc).</summary>
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

        /// <summary>GraphFusion.DeltaNetPrep over contiguous F32 buffers (see the kernel's contract).</summary>
        public void LaunchDeltaNetPrep(IntPtr window, IntPtr qkv, IntPtr convW, IntPtr alpha, IntPtr beta, IntPtr dt, IntPtr a,
            IntPtr convIn, IntPtr q, IntPtr k, IntPtr v, IntPtr g, IntPtr betaOut,
            int channels, int convRows, int width, int d, int kHeads, int vHeads, int rows, float eps, long qkvPitch, long alphaBetaPitch,
            IntPtr stream)
        {
            void** args = stackalloc void*[]
            {
                &window, &qkv, &convW, &alpha, &beta, &dt, &a, &convIn, &q, &k, &v, &g, &betaOut,
                &channels, &convRows, &width, &d, &kHeads, &vHeads, &rows, &eps, &qkvPitch, &alphaBetaPitch,
            };
            CudaDriverApi.cuLaunchKernel(_deltaNetPrep, (uint)(2 * kHeads + vHeads), (uint)rows, 1, (uint)d, 1, 1, 0, stream,
                (IntPtr)args, IntPtr.Zero).ThrowOnError();
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
            IntPtr stream)
        {
            void** args = stackalloc void*[] { &y, &z, &w, &output, &d, &eps, &heads, &zPitch };
            CudaDriverApi.cuLaunchKernel(_gatedRmsNorm, (uint)rows, 1, 1, (uint)d, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }
    }
}
