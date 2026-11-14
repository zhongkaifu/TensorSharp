// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

using TensorSharp.Cuda.Interop;
using TensorSharp.Graph;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// The direct CUDA backend's executor plugin: programs run node by node on CUDA tensors, with the
    /// backend's quantized products (weights resident under their device cache keys).
    /// </summary>
    public sealed partial class CudaGraphExecutor : TensorGraphExecutor, IDisposable, IDecodeCaptureExecutor
    {
        [ModuleInitializer]
        internal static void Register()
            => GraphExecutors.Register(typeof(CudaAllocator), allocator => new CudaGraphExecutor(allocator));

        public CudaGraphExecutor(IAllocator allocator) : base(allocator) { }

        /// <summary>The base executor's capabilities, and a decode row's recurrence updates its state in place (see
        /// <see cref="GatedDeltaNet"/>); decode steps replay their program a position on over windows padded to 256 rows,
        /// whose unwritten keys the attention stops before (<see cref="FlashAttention"/>), and record as one CUDA graph
        /// (<see cref="OpenDecodeCapture"/>).</summary>
        public override GraphCapabilities Capabilities { get; } = new()
        {
            StridedViews = true, DecodeWindowRows = 256, Replays = true, IndexedCacheWrites = true, InPlaceStateAlias = true,
            HostExperts = true, RecordsHostExperts = true, RegionsAtAnyWidth = true,
        };

        /// <summary>A recorded decode step reads its rotary positions from the device.</summary>
        protected override bool BroadcastsPositionsOnDevice => true;

        public IDecodeCapture OpenDecodeCapture()
            => Allocator is CudaAllocator cuda ? new RuntimeDecodeCapture(cuda) : null;

        /// <summary>Device memory a new recording leaves free (with the pool's cached blocks): a speculating request records
        /// a step for each window and draft width, each keeping the blocks its kernels used, and Qwen3.8 Flash Next's ISTA
        /// pack with its MTP head on an A40 ran out of memory recording or launching one after a 4K prompt.</summary>
        private const long RecordingHeadroomBytes = 1L << 30;

        public bool HasRoomToRecord()
        {
            if (Allocator is not CudaAllocator cuda) return false;
            try { return cuda.GetMemoryInfo().freeBytes + cuda.GetStats().CachedBytes >= RecordingHeadroomBytes; }
            catch (Exception) { return true; }
        }

        /// <summary>CUDA's binary kernels take same-shaped contiguous operands: a broadcast operand is gathered.</summary>
        protected override bool BroadcastsInKernels => false;

        /// <summary>
        /// Many short strided rows (a conv window's update: each channel's last width-1 columns, 6144 rows of 3 floats on
        /// Qwen3.5-0.8B) gather through the head-first kernel, a thread an element. The generic pitched copy runs a block a
        /// row: 124 us a layer, a third of that model's decode GPU time.
        /// </summary>
        protected override void CopyInto(Tensor dst, Tensor src)
        {
            if (dst.ElementType != src.ElementType && TryConvert(dst, src)) return;
            if (!TryGatherShortRows(dst, src)) base.CopyInto(dst, src);
        }

        /// <summary>An F16 / F32 copy that converts (the CUDA copy op copies one type only): the source made contiguous in
        /// its own type first where it is a strided view.</summary>
        private bool TryConvert(Tensor dst, Tensor src)
        {
            bool toHalf = dst.ElementType == DType.Float16 && src.ElementType == DType.Float32;
            if (!(toHalf || (dst.ElementType == DType.Float32 && src.ElementType == DType.Float16)) || Kernels == null
                || dst.Storage is not CudaStorage || src.Storage is not CudaStorage || !dst.IsContiguous()
                || dst.ElementCount() != src.ElementCount())
                return false;
            Tensor from = src.IsContiguous() ? src : Ops.NewContiguous(src);
            try
            {
                Launch(new[] { from }, dst, (k, stream) => k.LaunchConvert(Ptr(from), Ptr(dst), dst.ElementCount(), toHalf, stream));
            }
            finally
            {
                if (!ReferenceEquals(from, src)) from.Dispose();
            }
            return true;
        }

        private static bool TryGatherShortRows(Tensor dst, Tensor src)
        {
            if (src.DimensionCount != 2 || src.ElementType != DType.Float32 || dst.ElementType != DType.Float32
                || !dst.IsContiguous() || dst.ElementCount() != src.ElementCount() || src.Strides[1] != 1
                || src.Storage is not CudaStorage from || dst.Storage is not CudaStorage to || from.DeviceId != to.DeviceId)
                return false;
            long rows = src.Sizes[0], width = src.Sizes[1], pitch = src.Strides[0];
            if (rows < 64 || width > 16 || pitch <= width || pitch > int.MaxValue)
                return false;
            long column = src.StorageOffset % pitch, start = src.StorageOffset - column;
            if (column + width > pitch || start + rows * pitch > src.Storage.ElementCount)
                return false;
            using var cache = new Tensor(new[] { rows, pitch, 1L }, new[] { pitch, 1L, 1L }, src.Storage, start);
            using Tensor result = dst.View(rows, width, 1);
            return CudaFusedOps.TryGatherCircularHeadFirst(result, cache, (int)column, (int)width, (int)pitch);
        }

        /// <summary>The recurrence wrote its result into the leaf already (<see cref="GatedDeltaNet"/>): nothing to copy.</summary>
        protected override bool TryAlias(Tensor result, Tensor leaf, long offsetBytes)
            => ReferenceEquals(result.Storage, leaf.Storage)
               && result.StorageOffset * result.ElementType.Size() == leaf.StorageOffset * leaf.ElementType.Size() + offsetBytes;

        /// <summary>A prompt chunk's or a decode row's K/V rows into the head-first cache (F32 or F16) with the
        /// per-op path's kernel.</summary>
        protected override bool TryWriteRows(Tensor dst, Tensor src, int first, int count)
        {
            if (dst.DimensionCount is < 2 or > 3 || !dst.IsContiguous()) return false;
            using Tensor cache = HeadFirst(dst);
            using Tensor rows = src.View(src.Sizes[^3], src.Sizes[^2], src.Sizes[^1]);
            Tensor contiguous = rows.IsContiguous() ? rows.CopyRef() : Ops.NewContiguous(rows);
            using (contiguous)
                return CudaFusedOps.TryCopyHeadFirstToCache(cache, contiguous, first, count, (int)cache.Sizes[1], false);
        }

        /// <summary>A head-first cache or window [heads, cells, dim] as the kernels take it: a single head's binds as
        /// [cells, dim] (a leaf's trailing unit extents are dropped), so it is viewed with its head again.</summary>
        private static Tensor HeadFirst(Tensor t) => t.DimensionCount == 2 ? t.View(1, t.Sizes[0], t.Sizes[1]) : t.CopyRef();

        /// <summary>A prompt chunk's row blocks of a packed product stay views: the gated delta net's fused input region reads
        /// its q/k/v, alpha and beta blocks strided, where each block had been copied out first (6.5 GB a 2048-row step of
        /// Qwen3.8-27B; its 4K prompt 4-11% faster on an RTX PRO 6000 slice, the same tokens). Decode-size steps keep the
        /// copies (their own measured speed).</summary>
        protected override bool ProductRowBlocksAsViews(long rows) => rows > DenseDecodeRows;

        /// <summary>
        /// Attention over cache windows with the per-op path's kernels: a decode row runs the GQA decode kernel
        /// on the head-first cache in place (F32 or F16); a prompt chunk expands the window's KV heads, runs the
        /// half-operand GEMMs of the per-op prefill with the causal mask, and the softmax. Anything else (an
        /// explicit mask, an operand that is not a cache window) takes the generic path.
        /// </summary>
        protected override Tensor FlashAttention(Tensor q, Tensor k, Tensor v, AttentionMask mask, float scale,
            CacheWindow kWindow, CacheWindow vWindow)
        {
            // A padded decode window (its unwritten tail masked) runs its pattern over the written keys only.
            bool trimmed = mask.Pattern == MaskPattern.Explicit && mask.Trimmed != MaskPattern.Explicit && mask.Keys > 0;
            if (q.DimensionCount == 3 && q.Sizes[1] > 1 && q.Sizes[1] <= CudaDecodeDynParams.WindowRowBlocks
                && TryWindowRowsAsDecode(q, k, v, mask, trimmed, scale, kWindow, vWindow, out Tensor byRow))
                return byRow;
            if (mask.Pattern == MaskPattern.Explicit && !trimmed && TrySparseDecodeAttention(q, k, v, mask, scale, kWindow, vWindow, out Tensor sparse))
                return sparse;
            // An explicit mask (a sparse attention's selection over prompt or verify rows) runs on the tiled path below, its rows
            // added to each tile's scores: the generic attention would hold every head's scores of the whole chunk at once.
            bool explicitRows = mask.Pattern == MaskPattern.Explicit && !trimmed && mask.HasTensor && q.DimensionCount == 3 && q.Sizes[1] > 1;
            if (kWindow == null || vWindow == null || (mask.Pattern == MaskPattern.Explicit && !trimmed && !explicitRows) || q.ElementType != DType.Float32
                || q.DimensionCount != 3 || k.DimensionCount is < 2 or > 3 || v.DimensionCount is < 2 or > 3
                || kWindow.Cache.DimensionCount is < 2 or > 3 || vWindow.Cache.DimensionCount is < 2 or > 3
                || k.Sizes[^1] != v.Sizes[^1])
                return base.FlashAttention(q, k, v, mask, scale, kWindow, vWindow);
            int heads = (int)q.Sizes[0], nq = (int)q.Sizes[1], d = (int)q.Sizes[2];
            // A single KV head's window and cache bind without their head extent.
            int kvHeads = k.DimensionCount == 3 ? (int)k.Sizes[0] : 1;
            int len = trimmed ? mask.Keys : (int)k.Sizes[^2], floor = trimmed ? mask.TrimmedFloor : mask.Floor;
            using Tensor kc = HeadFirst(kWindow.Cache), vc = HeadFirst(vWindow.Cache);
            int cap = (int)kc.Sizes[1];
            using Tensor qc = q.IsContiguous() ? q.CopyRef() : Ops.NewContiguous(q);
            // A decode row over an F16 cache: the flash kernel split over the keys (the length read on the device when the
            // step is recorded), where the GQA decode kernels took one block a head over the whole window (200 us a layer
            // at 4K keys on Qwen3.8-27B).
            if (nq == 1 && kWindow.Start == 0 && vWindow.Start == 0 && heads % kvHeads == 0
                && kc.ElementType == DType.Float16 && vc.ElementType == DType.Float16 && kc.Sizes[^1] == d && vc.Sizes[^1] == d
                && (trimmed ? mask.Trimmed : mask.Pattern) is MaskPattern.Causal or MaskPattern.None
                && ((CudaAllocator)Allocator).FlashPrefillKernels is { } decodeFlash && decodeFlash.RunsAttention(d))
            {
                var cudaAllocator = (CudaAllocator)Allocator;
                IntPtr dyn = CudaDecodeDynParams.GetActiveDevicePtr(cudaAllocator);
                // Splits from the window, recorded or not: a recorded step replays at every length its window covers, and a
                // row's arithmetic must not depend on whether its step was recorded (a verify window's row and the decode
                // row at its position, a speculating request and a plain one).
                int splits = decodeFlash.Splits(heads / kvHeads, kvHeads, (int)k.Sizes[^2]);
                var attended = new Tensor(Allocator, DType.Float32, 1, heads, d);
                using Tensor partial = splits > 1
                    ? new Tensor(Allocator, DType.Float32, CudaFlashPrefillKernels.PartialFloats(1, heads, d, splits)) : null;
                Tensor[] written = partial == null ? new[] { attended } : new[] { attended, partial };
                Launch(new[] { qc, kc, vc }, written, (_, stream) => decodeFlash.LaunchAttention(Ptr(qc), Ptr(kc), Ptr(vc), cap, 1, len, floor,
                    heads, kvHeads, d, scale, splits, partial == null ? IntPtr.Zero : Ptr(partial), Ptr(attended), stream, dyn));
                return attended;
            }
            if (nq == 1)
            {
                using Tensor query = qc.View(1, heads * d);
                var result = new Tensor(Allocator, DType.Float32, 1, heads * d);
                if (CudaFusedOps.TryGqaDecodeAttention(result, query, kc, vc, heads, kvHeads, d,
                        (int)kWindow.Start + floor, len - floor, cap, false, scale))
                {
                    using (result) return result.View(1, heads, d);
                }
                result.Dispose();
                return base.FlashAttention(q, k, v, mask, scale, kWindow, vWindow);
            }
            if (kWindow.Start != 0 || vWindow.Start != 0 || heads % kvHeads != 0)
                return base.FlashAttention(q, k, v, mask, scale, kWindow, vWindow);
            // Causal rows over an F16 cache: the tensor-core flash attention (K and V tiles streamed once per block of
            // 128 query rows, split over the keys when the rows are few), not the scores of every query and key.
            if (kc.ElementType == DType.Float16 && vc.ElementType == DType.Float16 && kc.Sizes[^1] == d && vc.Sizes[^1] == d
                && (trimmed ? mask.Trimmed : mask.Pattern) == MaskPattern.Causal
                && ((CudaAllocator)Allocator).FlashPrefillKernels is { } flash && flash.RunsAttention(d))
            {
                var attended = new Tensor(Allocator, DType.Float32, nq, heads, d);
                // A recorded verify window reads its length on the device (the window's rows the last nq of it), as a
                // recorded decode row does: the splits then follow the window bucket, which every replay fits.
                IntPtr dyn = CudaDecodeDynParams.GetActiveDevicePtr((CudaAllocator)Allocator);
                int splits = flash.Splits(nq * (heads / kvHeads), kvHeads, dyn != IntPtr.Zero ? (int)k.Sizes[^2] : len);
                using Tensor partial = splits > 1
                    ? new Tensor(Allocator, DType.Float32, CudaFlashPrefillKernels.PartialFloats(nq, heads, d, splits)) : null;
                Tensor[] written = partial == null ? new[] { attended } : new[] { attended, partial };
                Launch(new[] { qc, kc, vc }, written, (_, stream) => flash.LaunchAttention(Ptr(qc), Ptr(kc), Ptr(vc), cap, nq, len, floor,
                    heads, kvHeads, d, scale, splits, partial == null ? IntPtr.Zero : Ptr(partial), Ptr(attended), stream, dyn));
                return attended;
            }
            // A recorded verify window replays at later positions, where the written length this classification read (the
            // capture's) is short: such a window runs over the whole cache window with the mask itself, which every launch
            // rewrites (the flash kernel above reads the length on the device instead). Only when recording: the window's
            // length is the step's, so a sequence batched with others would round with its neighbours' lengths.
            if (trimmed && nq > 1 && mask.HasTensor && CudaDecodeDynParams.GetActiveDevicePtr((CudaAllocator)Allocator) != IntPtr.Zero)
            {
                Tensor whole = mask.Tensor;
                return FlashAttention(q, k, v, new AttentionMask(MaskPattern.Explicit, 0, () => whole), scale, kWindow, vWindow);
            }
            int group = heads / kvHeads;
            // K and V once as F32 [kvHeads, len, d] (the cache may hold F16); the query heads of a group share
            // their KV head, so q [heads, rows, d] is [kvHeads, group * rows, d] and each product is one batch
            // per KV head (no per-head copies of the cache).
            using var kx = new Tensor(Allocator, DType.Float32, kvHeads, len, d);
            using var vx = new Tensor(Allocator, DType.Float32, kvHeads, len, d);
            if (!CudaFusedOps.TryExpandKvHeads(kx, kc, 1, len) || !CudaFusedOps.TryExpandKvHeads(vx, vc, 1, len))
                return base.FlashAttention(q, k, v, mask, scale, kWindow, vWindow);
            using Tensor kt = kx.Transpose(1, 2);
            using Tensor maskTensor = explicitRows ? Contiguous2(mask.Tensor, nq, len) : null;
            if (explicitRows && maskTensor == null)
                return base.FlashAttention(q, k, v, mask, scale, kWindow, vWindow);
            var output = new Tensor(Allocator, DType.Float32, nq, heads, d);
            // Query tiles keep the scores [heads, tile, len] within ScoreBudgetBytes at any context length.
            int tile = (int)Math.Clamp(ScoreBudgetBytes / ((long)heads * len * sizeof(float)), 1, nq);
            for (int first = 0; first < nq; first += tile)
            {
                int rows = Math.Min(tile, nq - first);
                using Tensor qRows = qc.Narrow(1, first, rows);
                using Tensor qTile = rows == nq ? qc.CopyRef() : Ops.NewContiguous(qRows);
                using Tensor q3 = qTile.View(kvHeads, (long)group * rows, d);
                using var scores = new Tensor(Allocator, DType.Float32, kvHeads, (long)group * rows, len);
                if (!CudaHalfOperandGemm.TryAddmmBatch(scores, 0, scores, scale, q3, kt)) Ops.AddmmBatch(scores, 0, scores, scale, q3, kt);
                using Tensor perHead = scores.View(heads, rows, len);
                if (floor > 0)
                {
                    using Tensor below = perHead.Narrow(2, 0, Math.Min(floor, len));
                    Ops.Fill(below, float.NegativeInfinity);
                }
                if ((trimmed ? mask.Trimmed : mask.Pattern) == MaskPattern.Causal) Ops.AddCausalMask(perHead, rows, len - nq + first, float.NegativeInfinity);
                if (explicitRows)
                {
                    using Tensor maskRows = maskTensor.Narrow(0, first, rows);
                    for (int h = 0; h < heads; h++)
                    {
                        using Tensor head = perHead.Select(0, h);
                        Ops.Add(head, head, maskRows);
                    }
                }
                Ops.Softmax(perHead, perHead);
                using var attn = new Tensor(Allocator, DType.Float32, kvHeads, (long)group * rows, d);
                if (!CudaHalfOperandGemm.TryAddmmBatch(attn, 0, attn, 1.0f, scores, vx)) Ops.AddmmBatch(attn, 0, attn, 1.0f, scores, vx);
                using Tensor attnHeads = attn.View(heads, rows, d);
                using Tensor permuted = attnHeads.Permute(1, 0, 2);
                using Tensor target = output.Narrow(0, first, rows);
                Ops.Copy(target, permuted);
            }
            return output;
        }

        /// <summary>
        /// A verify window's rows (up to 8 a sequence, causal over the cache window or a sparse attention's selection) each
        /// run as the decode row at their position: row r attends the window's keys less the nq - 1 - r after it, read on
        /// the device from that row's block of the step's parameters when the step is recorded. The multi-row kernels split
        /// the keys by the window's length (the flash kernel's split share), or ran another product altogether (an F32
        /// cache, a selection), so a window row rounded apart from the decode row at its position, and speculative
        /// decoding could answer other than plain greedy at a near-tie.
        /// </summary>
        private bool TryWindowRowsAsDecode(Tensor q, Tensor k, Tensor v, AttentionMask mask, bool trimmed, float scale,
            CacheWindow kWindow, CacheWindow vWindow, out Tensor attended)
        {
            attended = null;
            if (kWindow == null || vWindow == null || kWindow.Start != 0 || vWindow.Start != 0 || q.ElementType != DType.Float32
                || k.DimensionCount is < 2 or > 3 || v.DimensionCount is < 2 or > 3 || k.Sizes[^1] != v.Sizes[^1])
                return false;
            int heads = (int)q.Sizes[0], nq = (int)q.Sizes[1], d = (int)q.Sizes[2];
            int kvHeads = k.DimensionCount == 3 ? (int)k.Sizes[0] : 1;
            bool selection = mask.Pattern == MaskPattern.Explicit && !trimmed;
            int keys = trimmed ? mask.Keys : (int)k.Sizes[^2], floor = trimmed ? mask.TrimmedFloor : mask.Floor;
            if (heads % kvHeads != 0 || (selection ? !mask.HasTensor : (trimmed ? mask.Trimmed : mask.Pattern) != MaskPattern.Causal)
                || keys - (nq - 1) <= floor)
                return false;
            Tensor whole = mask.HasTensor ? mask.Tensor : null;
            if (selection && (whole == null || whole.DimensionCount != 2 || whole.Sizes[0] != nq)) return false;
            var rowMasks = new List<Tensor>();
            using Tensor permuted = q.Permute(1, 0, 2);
            using Tensor byRow = Ops.NewContiguous(permuted);
            var output = new Tensor(Allocator, DType.Float32, nq, heads, d);
            bool done = false;
            try
            {
                for (int r = 0; r < nq; r++)
                {
                    int fewer = nq - 1 - r, row = r;
                    Func<Tensor> rowOf = whole == null ? null : () =>
                    {
                        Tensor t = whole.Narrow(0, row, 1);
                        rowMasks.Add(t);
                        return t;
                    };
                    // The decode row's own mask: a selection's row, or the written keys up to its position.
                    AttentionMask rowMask = selection
                        ? new AttentionMask(MaskPattern.Explicit, 0, rowOf) { KvMax = mask.KvMax }
                        : new AttentionMask(MaskPattern.Explicit, 0, rowOf, keys - fewer, MaskPattern.None, floor);
                    using Tensor qRow = byRow.Narrow(0, r, 1);
                    using Tensor q1 = qRow.View(heads, 1, d);
                    CudaDecodeDynParams.WindowRow(fewer);
                    try
                    {
                        using Tensor one = FlashAttention(q1, k, v, rowMask, scale, kWindow, vWindow);
                        using Tensor target = output.Narrow(0, r, 1);
                        using Tensor source = one.View(1, heads, d);
                        Ops.Copy(target, source);
                    }
                    finally
                    {
                        CudaDecodeDynParams.WindowRow(null);
                    }
                }
                done = true;
            }
            finally
            {
                foreach (Tensor t in rowMasks) t.Dispose();
                if (!done) output.Dispose();
            }
            attended = output;
            return true;
        }

        /// <summary>A mask as contiguous F32 [rows, keys] on this device, or null.</summary>
        private static Tensor Contiguous2(Tensor m, int rows, int keys)
        {
            if (m == null || m.Storage is not CudaStorage || m.ElementType != DType.Float32 || m.ElementCount() != (long)rows * keys)
                return null;
            using Tensor flat = m.IsContiguous() ? m.CopyRef() : Ops.NewContiguous(m);
            return flat.View(rows, keys);
        }

        /// <summary>
        /// A decode row whose mask was computed on the device (a sparse attention's selection: QSA past its budget) over an
        /// F16 cache window: the cells the mask keeps are listed on the device and the flash kernel reads those rows alone
        /// (about the budget's 2K cells, not the whole window), the mask's values added. The list stays on the device, so a
        /// recorded step replays it with each position's selection.
        /// </summary>
        private bool TrySparseDecodeAttention(Tensor q, Tensor k, Tensor v, AttentionMask mask, float scale, CacheWindow kWindow,
            CacheWindow vWindow, out Tensor attended)
        {
            attended = null;
            var cuda = (CudaAllocator)Allocator;
            if (!mask.HasTensor || kWindow == null || vWindow == null || kWindow.Start != 0 || vWindow.Start != 0
                || q.ElementType != DType.Float32 || q.DimensionCount != 3 || q.Sizes[1] != 1 || Kernels == null
                || k.DimensionCount is < 2 or > 3 || v.DimensionCount is < 2 or > 3 || k.Sizes[^1] != v.Sizes[^1])
                return false;
            int heads = (int)q.Sizes[0], d = (int)q.Sizes[2];
            int kvHeads = k.DimensionCount == 3 ? (int)k.Sizes[0] : 1;
            int nkv = (int)k.Sizes[^2];
            if (heads % kvHeads != 0 || cuda.FlashPrefillKernels is not { } flash || !flash.RunsAttention(d)) return false;
            using Tensor kc = HeadFirst(kWindow.Cache), vc = HeadFirst(vWindow.Cache);
            if (kc.ElementType != DType.Float16 || vc.ElementType != DType.Float16 || kc.Sizes[^1] != d || vc.Sizes[^1] != d) return false;
            Tensor m = mask.Tensor;
            if (m == null || m.Storage is not CudaStorage || m.ElementType != DType.Float32 || m.Sizes[^1] != nkv) return false;
            using Tensor row = m.IsContiguous() ? m.CopyRef() : Ops.NewContiguous(m);
            int cap = (int)kc.Sizes[1];
            using Tensor qc = q.IsContiguous() ? q.CopyRef() : Ops.NewContiguous(q);
            using var cells = new Tensor(Allocator, DType.Int32, nkv);
            using var count = new Tensor(Allocator, DType.Int32, 1);
            Launch(new[] { row }, new[] { cells, count }, (kn, stream) => kn.LaunchMaskCells(Ptr(row), Ptr(cells), Ptr(count), 1, nkv, stream));
            int splits = flash.Splits(heads / kvHeads, kvHeads, mask.KvMax > 0 ? Math.Min(mask.KvMax, nkv) : nkv);
            var result = new Tensor(Allocator, DType.Float32, 1, heads, d);
            using Tensor partial = splits > 1 ? new Tensor(Allocator, DType.Float32, CudaFlashPrefillKernels.PartialFloats(1, heads, d, splits)) : null;
            Tensor[] written = partial == null ? new[] { result } : new[] { result, partial };
            Launch(new[] { qc, kc, vc, cells, count, row }, written, (_, stream) => flash.LaunchAttention(Ptr(qc), Ptr(kc), Ptr(vc), cap, 1, nkv, 0,
                heads, kvHeads, d, scale, splits, partial == null ? IntPtr.Zero : Ptr(partial), Ptr(result), stream, IntPtr.Zero,
                Ptr(cells), Ptr(count), Ptr(row)));
            attended = result;
            return true;
        }

        /// <summary>Most bytes of attention scores a prompt chunk holds at once (its queries run in tiles).</summary>
        private const long ScoreBudgetBytes = 256L << 20;

        private CudaGraphKernels _kernels;
        private string _kernelsMissing;

        /// <summary>The executor's kernel module in this allocator's context, or null (with the reason
        /// reported once) where its PTX is missing or stale: those ops then run on the host.</summary>
        private CudaGraphKernels Kernels
        {
            get
            {
                if (_kernels != null || _kernelsMissing != null) return _kernels;
                var cuda = (CudaAllocator)Allocator;
                cuda.Context.MakeCurrent();
                if (!CudaGraphKernels.TryCreate(out _kernels, out _kernelsMissing))
                    Console.Error.WriteLine($"WARNING: CUDA graph kernels unavailable ({_kernelsMissing}); the gated delta rule runs on the host.");
                return _kernels;
            }
        }

        /// <summary>ggml_top_k on the device (the router's selection: no read-back per layer, so a decode step can be
        /// recorded; a sparse attention's blocks: many picks over a wide row, radix-selected); the result carries no host
        /// ints.</summary>
        protected override Value TopK(Tensor x, int k, RunState run)
        {
            CudaGraphKernels kernels = Kernels;
            long cols = x.Sizes[^1], rows = x.ElementCount() / Math.Max(1, cols);
            bool wide = k > 32 || cols > CudaGraphKernels.TopKMaxColumns;
            if (kernels == null || x.Storage is not CudaStorage || !x.IsContiguous() || x.ElementType != DType.Float32
                || k < 1 || k > cols || rows > int.MaxValue || cols > int.MaxValue || (wide && k > CudaGraphKernels.TopKWideMaxPicks))
                return base.TopK(x, k, run);
            var ids = new Tensor(Allocator, DType.Int32, Sizes(stackalloc long[] { k, rows }));
            Launch(new[] { x }, ids, (kn, stream) =>
            {
                if (wide) kn.LaunchTopKWide(Ptr(x), Ptr(ids), (int)rows, (int)cols, k, stream);
                else kn.LaunchTopK(Ptr(x), Ptr(ids), (int)rows, (int)cols, k, stream);
            });
            return new Value { Tensor = run.Keep(ids) };
        }

        /// <summary>ggml_get_rows of F32 or F16 rows (a strided view's read in place: a cache's half) by ids on the device (a
        /// top-k's selection, a recorded step's varying rows): gathered there, not read back.</summary>
        protected override Tensor GetRows(Tensor src, Value ids, RunState run)
        {
            CudaGraphKernels kernels = Kernels;
            Tensor idx = ids.Tensor;
            if (kernels == null || ids.Ints != null || idx?.Storage is not CudaStorage || idx.ElementType != DType.Int32 || !idx.IsContiguous()
                || src.Storage is not CudaStorage || src.ElementType is not (DType.Float32 or DType.Float16))
                return base.GetRows(src, ids, run);
            long[] na = Ne(src), ni = Ne(idx);
            long n = ni[0], batches = ni[1] * ni[2];
            if (batches != na[2] * na[3] || na[0] > int.MaxValue || na[1] > int.MaxValue || n * batches > int.MaxValue)
                return base.GetRows(src, ids, run);
            // One batch reads its rows where they are (unit-stride rows any pitch apart); several from packed rows.
            Tensor s = src;
            long pitch = na[0];
            if (batches == 1 && src.Strides[^1] == 1 && src.DimensionCount >= 2) pitch = src.Strides[^2];
            else if (!src.IsContiguous()) s = run.Keep(Ops.NewContiguous(src));
            bool half = s.ElementType == DType.Float16;
            var result = new Tensor(Allocator, DType.Float32, Sizes(stackalloc long[] { na[0], n, ni[1], ni[2] }));
            Launch(new[] { s, idx }, result, (k, stream) => k.LaunchGetRows(Ptr(s), Ptr(idx), Ptr(result), (int)na[0], (int)n, (int)na[1],
                (int)batches, pitch, half, stream));
            return result;
        }

        /// <summary>ggml_set_rows with the rows on the device (a recorded step's varying input): one block of F32 or F16
        /// rows written there.</summary>
        protected override bool TrySetRowsOnDevice(Tensor dst, Tensor src, Tensor ids)
        {
            CudaGraphKernels kernels = Kernels;
            if (kernels == null || ids.Storage is not CudaStorage || ids.ElementType != DType.Int32 || !ids.IsContiguous()
                || dst.Storage is not CudaStorage || dst.ElementType is not (DType.Float32 or DType.Float16) || dst.Strides[^1] != 1
                || src.Storage is not CudaStorage || src.ElementType != DType.Float32 || !src.IsContiguous())
                return false;
            long[] nd = Ne(dst), ns = Ne(src);
            if (nd[2] * nd[3] != 1 || ns[2] * ns[3] != 1 || nd[0] != ns[0] || ids.ElementCount() < ns[1] || nd[1] > int.MaxValue)
                return false;
            long pitch = dst.DimensionCount >= 2 ? dst.Strides[^2] : nd[0];
            int count = (int)ns[1];
            Launch(new[] { src, ids }, dst, (k, stream) => k.LaunchSetRows(Ptr(src), Ptr(ids), Ptr(dst), count, (int)ns[0], pitch, (int)nd[1],
                dst.ElementType == DType.Float16, stream));
            return true;
        }

        /// <summary>ggml_lightning_indexer on the device (a warp a key): a sparse attention's block scores, not read back.</summary>
        protected override Tensor LightningIndexer(Tensor q, Tensor keys, Tensor weights, Tensor mask, RunState run)
        {
            CudaGraphKernels kernels = Kernels;
            long[] nq = Ne(q), nk = Ne(keys), nm = Ne(mask);
            long d = nq[0], heads = nq[1], rows = nq[2], n = nk[2];
            if (kernels == null || d > 256 || heads * d > CudaGraphKernels.LightningIndexerMaxQ || nk[0] != d || nm[0] != n || n > int.MaxValue
                || rows > int.MaxValue || weights.ElementCount() < heads * rows
                || !AllF32OnDevice(q, keys, weights, mask))
                return base.LightningIndexer(q, keys, weights, mask, run);
            Tensor qc = q.IsContiguous() ? q : run.Keep(Ops.NewContiguous(q)), kc = keys.IsContiguous() ? keys : run.Keep(Ops.NewContiguous(keys));
            Tensor wc = weights.IsContiguous() ? weights : run.Keep(Ops.NewContiguous(weights)), mc = mask.IsContiguous() ? mask : run.Keep(Ops.NewContiguous(mask));
            int maskRows = (int)Math.Max(1, mask.ElementCount() / n);
            var result = new Tensor(Allocator, DType.Float32, rows, n);
            Launch(new[] { qc, kc, wc, mc }, result, (k, stream) => k.LaunchLightningIndexer(Ptr(qc), Ptr(kc), Ptr(wc), Ptr(mc), Ptr(result),
                (int)d, (int)heads, (int)n, (int)rows, maskRows, stream));
            return result;
        }

        /// <summary>ggml_top_k_mask on the device: the picks a device top-k made stay there.</summary>
        protected override Tensor TopKMask(Tensor mask, Value picks, Tensor score, Value blocks, Value extra, RunState run)
        {
            CudaGraphKernels kernels = Kernels;
            Tensor p = picks.Tensor, b = blocks.Tensor, e = extra.Tensor;
            bool hasBlocks = b != null || blocks.Ints != null, hasExtra = e != null || extra.Ints != null;
            if (kernels == null || !DeviceInts(p) || (hasBlocks && !DeviceInts(b)) || (hasExtra && !DeviceInts(e))
                || !AllF32OnDevice(mask) || (score != null && !AllF32OnDevice(score)))
                return base.TopKMask(mask, picks, score, blocks, extra, run);
            long nkv = Ne(mask)[0], top = Ne(p)[0], rows = p.ElementCount() / Math.Max(1, top);
            long nPick = score != null ? Ne(score)[0] : 0, cells = hasBlocks ? Ne(b)[0] : 1, nBlocks = hasBlocks ? Ne(b)[1] : 0;
            long nExtra = hasExtra ? Ne(e)[0] : 0;
            if (nkv > int.MaxValue || rows > int.MaxValue || top * cells > int.MaxValue) return base.TopKMask(mask, picks, score, blocks, extra, run);
            Tensor mc = mask.IsContiguous() ? mask : run.Keep(Ops.NewContiguous(mask));
            Tensor sc = score == null || score.IsContiguous() ? score : run.Keep(Ops.NewContiguous(score));
            int maskRows = (int)Math.Max(1, mask.ElementCount() / nkv);
            var result = new Tensor(Allocator, DType.Float32, rows, nkv);
            var inputs = new List<Tensor> { mc, p };
            foreach (Tensor t in new[] { sc, b, e }) if (t != null) inputs.Add(t);
            Launch(inputs.ToArray(), result, (k, stream) => k.LaunchTopKMask(Ptr(mc), Ptr(p), sc == null ? IntPtr.Zero : Ptr(sc),
                hasBlocks ? Ptr(b) : IntPtr.Zero, hasExtra ? Ptr(e) : IntPtr.Zero, Ptr(result), (int)rows, (int)nkv, maskRows, (int)top,
                (int)nPick, (int)cells, (int)nBlocks, (int)nExtra, stream));
            return result;
        }

        /// <summary>
        /// A hyper-connection read (<see cref="GraphFusion.HcRead"/>) as five launches: the grouped norm, the low-rank down
        /// product, its scaled silu, the up product and the gated mix in one pass, then the injection weights; the region's
        /// ops were about sixteen, a decode step's dominant node count on Qwen3.8 Flash Next (two reads a layer).
        /// </summary>
        private bool TryHcRead(Tensor[] inputs, IGraphWeight[] weights, long[] ip, float[] fp, Tensor[] results)
        {
            CudaGraphKernels kernels = Kernels;
            if (kernels == null || inputs.Length < 5 || ip == null || ip.Length < 3 || results.Length < 2) return false;
            Tensor x = inputs[0], gamma = inputs[1];
            int n = (int)ip[0], hc = (int)ip[1], lr = (int)ip[2];
            long width = (long)hc * n;
            // The matrices are quantized weights, or F32 tensors [out, in] (a checkpoint keeping them unquantized).
            bool Fits(int i, long inDim, long outDim) => weights[i] != null
                ? weights[i].Ne0 == inDim && weights[i].Ne1 == outDim
                : inputs[i] is { } w && w.Storage is CudaStorage && w.ElementType == DType.Float32 && w.DimensionCount == 2
                  && w.Sizes[0] == outDim && w.Sizes[1] == inDim;
            if (!AllF32OnDevice(x, gamma) || !x.IsContiguous() || !gamma.IsContiguous() || gamma.ElementCount() != width
                || x.ElementCount() % width != 0 || !Fits(2, width, lr) || !Fits(3, width, hc) || !Fits(4, lr, width))
                return false;
            int rows = (int)(x.ElementCount() / width);
            float eps = fp != null && fp.Length > 0 ? fp[0] : 1e-6f;
            Tensor Product(int i, Tensor input)
            {
                if (weights[i] != null) return QuantizedMatMul(weights[i], input);
                Tensor w = inputs[i];
                var y = new Tensor(Allocator, DType.Float32, input.Sizes[0], w.Sizes[0]);
                DenseProduct(y, input, w);
                return y;
            }
            using var xn = new Tensor(Allocator, DType.Float32, rows, width);
            Launch(new[] { x, gamma }, xn, (k, stream) => k.LaunchHcNorm(Ptr(x), Ptr(gamma), Ptr(xn), rows, n, hc, eps, stream));
            using Tensor lo = Product(2, xn);
            Launch(new[] { lo }, lo, (k, stream) => k.LaunchScaleSilu(Ptr(lo), Ptr(lo), lo.ElementCount(), 1f / hc, stream));
            using Tensor gates = Product(4, lo);
            var mix = new Tensor(Allocator, DType.Float32, rows, n);
            Launch(new[] { xn, gates }, mix, (k, stream) => k.LaunchHcGateMix(Ptr(xn), Ptr(gates), Ptr(mix), rows, n, hc, stream));
            results[0] = mix;
            results[1] = Product(3, xn);
            return true;
        }

        /// <summary>A hyper-connection scatter (<see cref="GraphFusion.HcScatter"/>) as one pass over the streams.</summary>
        private bool TryHcScatter(Tensor[] inputs, long[] ip, Tensor[] results)
        {
            CudaGraphKernels kernels = Kernels;
            if (kernels == null || inputs.Length < 3 || ip == null || ip.Length < 2 || results.Length < 1) return false;
            Tensor streams = inputs[0], block = inputs[1], inject = inputs[2];
            int n = (int)ip[0], hc = (int)ip[1];
            long width = (long)hc * n;
            if (!AllF32OnDevice(streams, block, inject) || !streams.IsContiguous() || !block.IsContiguous() || !inject.IsContiguous()
                || streams.ElementCount() % width != 0)
                return false;
            int rows = (int)(streams.ElementCount() / width);
            if (block.ElementCount() != (long)rows * n || inject.ElementCount() != (long)rows * hc) return false;
            var output = new Tensor(Allocator, DType.Float32, rows, width);
            Launch(new[] { streams, block, inject }, output, (k, stream) => k.LaunchHcScatter(Ptr(streams), Ptr(block), Ptr(inject), Ptr(output),
                rows, n, hc, stream));
            results[0] = output;
            return true;
        }

        private static bool DeviceInts(Tensor t) => t != null && t.Storage is CudaStorage && t.ElementType == DType.Int32 && t.IsContiguous();

        private static bool AllF32OnDevice(params Tensor[] tensors)
        {
            foreach (Tensor t in tensors)
                if (t == null || t.Storage is not CudaStorage || t.ElementType != DType.Float32) return false;
            return true;
        }


        /// <summary>Unquantized products with at most this many rows and outputs take the small-matvec kernel (one block an
        /// output, threads across the inner dimension): cuBLAS's GEMV spreads only the outputs over the device.</summary>
        internal const int SmallProductMaxRows = 16, SmallProductMaxOutputs = 64;

        protected override void DenseProduct(Tensor result, Tensor rows, Tensor weight)
        {
            long n = rows.Sizes[0], k = rows.Sizes[1], m = weight.Sizes[0];
            if (Kernels != null && n <= SmallProductMaxRows && m <= SmallProductMaxOutputs && k <= int.MaxValue
                && AllF32OnDevice(result, rows, weight) && result.IsContiguous() && rows.IsContiguous() && weight.IsContiguous())
            {
                Launch(new[] { rows, weight }, result, (kernels, stream) => kernels.LaunchSmallMatvec(Ptr(rows), Ptr(weight), Ptr(result),
                    (int)n, (int)k, (int)m, k, stream));
                return;
            }
            base.DenseProduct(result, rows, weight);
        }

        protected override bool TryBinaryBroadcast(GraphOp kind, Tensor result, Tensor a, Tensor b, long[] na, long[] nb)
        {
            int op = kind switch { GraphOp.Add => 0, GraphOp.Sub => 1, GraphOp.Mul => 2, GraphOp.Div => 3, _ => -1 };
            if (op < 0 || Kernels == null || !AllF32OnDevice(result, a, b) || !result.IsContiguous() || !a.IsContiguous() || !b.IsContiguous())
                return false;
            Launch(new[] { a, b }, result, (kn, stream) => kn.LaunchBinaryBroadcast(Ptr(a), Ptr(b), Ptr(result), na, nb, op, stream));
            return true;
        }

        /// <summary>ggml_sum_rows as one kernel (the CUDA reductions run on the host).</summary>
        protected override Tensor SumRows(Tensor x)
        {
            CudaGraphKernels kernels = Kernels;
            long cols = x.Sizes[^1], rows = x.ElementCount() / Math.Max(1, cols);
            if (kernels == null || x.Storage is not CudaStorage || !x.IsContiguous() || x.ElementType != DType.Float32
                || cols > int.MaxValue || rows > int.MaxValue || cols == 0)
                return base.SumRows(x);
            long[] sizes = x.Sizes.ToArray();
            sizes[^1] = 1;
            var result = new Tensor(Allocator, DType.Float32, sizes);
            Launch(new[] { x }, result, (k, stream) => k.LaunchSumRows(Ptr(x), Ptr(result), (int)rows, (int)cols, stream));
            return result;
        }

        /// <summary>ggml_sqrt as one kernel (the CUDA ops have no pow).</summary>
        protected override Tensor Sqrt(Tensor x)
        {
            CudaGraphKernels kernels = Kernels;
            if (kernels == null || x.Storage is not CudaStorage || !x.IsContiguous() || x.ElementType != DType.Float32)
                return base.Sqrt(x);
            var result = new Tensor(Allocator, DType.Float32, x.Sizes);
            Launch(new[] { x }, result, (k, stream) => k.LaunchSqrt(Ptr(x), Ptr(result), x.ElementCount(), stream));
            return result;
        }

        /// <summary>ggml's softplus as one kernel (the generic decomposition's abs/exp/log have no CUDA kernels).</summary>
        protected override Tensor Softplus(Tensor x)
        {
            CudaGraphKernels kernels = Kernels;
            if (kernels == null || x.Storage is not CudaStorage || !x.IsContiguous() || x.ElementType != DType.Float32)
                return base.Softplus(x);
            var result = new Tensor(Allocator, DType.Float32, x.Sizes);
            Launch(new[] { x }, result, (k, stream) => k.LaunchSoftplus(Ptr(x), Ptr(result), x.ElementCount(), stream));
            return result;
        }

        /// <summary>ggml_ssm_conv as one kernel.</summary>
        protected override Tensor SsmConv(Tensor x, Tensor kernel)
        {
            CudaGraphKernels kernels = Kernels;
            if (kernels == null || x.Storage is not CudaStorage || kernel.Storage is not CudaStorage
                || !x.IsContiguous() || !kernel.IsContiguous() || x.ElementType != DType.Float32 || kernel.ElementType != DType.Float32)
                return base.SsmConv(x, kernel);
            int ns = (int)x.Sizes[0], channels = (int)x.Sizes[1], length = (int)x.Sizes[2], width = (int)kernel.Sizes[1];
            var result = new Tensor(Allocator, DType.Float32, ns, length - width + 1, channels);
            Launch(new[] { x, kernel }, result, (k, stream) => k.LaunchSsmConv(Ptr(x), Ptr(kernel), Ptr(result), channels, length, width, ns, stream));
            return result;
        }

        protected override bool Fuses(GraphFusion kind) => kind switch
        {
            GraphFusion.DeltaNetPrep or GraphFusion.GatedRmsNorm or GraphFusion.HcRead or GraphFusion.HcScatter => Kernels != null,
            GraphFusion.RoutedExperts or GraphFusion.SwigluFfn => ((CudaAllocator)Allocator).Kernels != null,
            _ => false,
        };

        /// <summary>The delta rule's prep and the gated norm after it as one kernel each; routed experts with the per-op
        /// path's MoE kernels.</summary>
        protected override bool TryFused(GraphFusion kind, Tensor[] inputs, IGraphWeight[] weights, long[] ip, float[] fp, Tensor[] results)
        {
            if (kind == GraphFusion.RoutedExperts) return TryRoutedExperts(inputs, weights, ip, results);
            if (kind == GraphFusion.SwigluFfn) return TrySwigluFfn(inputs, weights, ip, results);
            if (kind == GraphFusion.HcRead) return TryHcRead(inputs, weights, ip, fp, results);
            if (kind == GraphFusion.HcScatter) return TryHcScatter(inputs, ip, results);
            CudaGraphKernels kernels = Kernels;
            if (kernels == null || Array.Exists(inputs, t => t == null || t.Storage is not CudaStorage || t.ElementType != DType.Float32))
                return false;
            float eps = fp != null && fp.Length > 0 ? fp[0] : 1e-6f;
            switch (kind)
            {
                case GraphFusion.DeltaNetPrep:
                {
                    int dk = (int)ip[0], kHeads = (int)ip[1], dv = (int)ip[2], vHeads = (int)ip[3];
                    Tensor window = inputs[0], qkv = inputs[1], convW = inputs[2];
                    int channels = (int)window.Sizes[0], convRows = (int)window.Sizes[^1], width = (int)convW.Sizes[^1];
                    int rows = (int)(qkv.ElementCount() / channels);
                    // qkv, alpha and beta are read where they are (views of the packed in-projection): rows a pitch apart.
                    if (dk != dv || dk % 32 != 0 || dk > 1024 || channels != (2 * kHeads + vHeads) * dk
                        || window.ElementCount() != (long)channels * convRows || convW.ElementCount() != (long)channels * width
                        || inputs[3].ElementCount() != (long)rows * vHeads || inputs[4].ElementCount() != (long)rows * vHeads
                        || !TryRows(qkv, channels, out long qkvPitch) || !TryRows(inputs[3], vHeads, out long alphaPitch)
                        || !TryRows(inputs[4], vHeads, out long betaPitch) || alphaPitch != betaPitch)
                        return false;
                    // The conv window, its weights, dt and a are read contiguous (copied when they are not).
                    var copies = new System.Collections.Generic.List<Tensor>();
                    foreach (int i in new[] { 0, 2, 5, 6 })
                        if (!inputs[i].IsContiguous()) copies.Add(inputs[i] = Ops.NewContiguous(inputs[i]));
                    results[0] = new Tensor(Allocator, DType.Float32, channels, convRows + rows);
                    results[1] = new Tensor(Allocator, DType.Float32, rows, kHeads, dk);
                    results[2] = new Tensor(Allocator, DType.Float32, rows, kHeads, dk);
                    results[3] = new Tensor(Allocator, DType.Float32, rows, vHeads, dv);
                    results[4] = new Tensor(Allocator, DType.Float32, rows, vHeads, 1);
                    results[5] = new Tensor(Allocator, DType.Float32, rows, vHeads, 1);
                    Launch(inputs, results, (k, stream) => k.LaunchDeltaNetPrep(Ptr(inputs[0]), Ptr(inputs[1]), Ptr(inputs[2]), Ptr(inputs[3]),
                        Ptr(inputs[4]), Ptr(inputs[5]), Ptr(inputs[6]), Ptr(results[0]), Ptr(results[1]), Ptr(results[2]), Ptr(results[3]),
                        Ptr(results[4]), Ptr(results[5]), channels, convRows, width, dk, kHeads, vHeads, rows, eps, qkvPitch, alphaPitch, stream));
                    foreach (Tensor c in copies) c.Dispose();
                    return true;
                }
                case GraphFusion.GatedRmsNorm:
                {
                    if (ip != null && ip.Length > 0 && ip[0] != 0) return false;   // the kernel gates with silu only
                    Tensor y = inputs[0], z = inputs[1];
                    int d = (int)inputs[2].ElementCount();
                    // z is read where it is (a view of the packed in-projection): a token's heads * d values a pitch apart.
                    int heads = y.DimensionCount >= 2 ? (int)y.Sizes[^2] : 1;
                    if (d % 32 != 0 || d > 1024 || y.ElementCount() % d != 0 || z.ElementCount() != y.ElementCount()
                        || !y.IsContiguous() || !inputs[2].IsContiguous() || y.Sizes[^1] != d)
                        return false;
                    if (!TryRows(z, (long)heads * d, out long zPitch))
                        return false;
                    results[0] = new Tensor(Allocator, DType.Float32, y.Sizes);
                    int rows = (int)(y.ElementCount() / d);
                    Launch(inputs, results, (k, stream) => k.LaunchGatedRmsNorm(Ptr(inputs[0]), Ptr(inputs[1]), Ptr(inputs[2]), Ptr(results[0]),
                        rows, d, eps, heads, zPitch, stream));
                    return true;
                }
                default:
                    return false;
            }
        }

        /// <summary>A block's residual add and the next block's input norm in one kernel (the stream and the normed row
        /// written in one pass; three otherwise).</summary>
        protected override bool TryAddWeightedNorm(Tensor a, Tensor b, Tensor weight, float eps, out Tensor sum, out Tensor normed)
        {
            sum = normed = null;
            CudaGraphKernels kernels = Kernels;
            Tensor[] inputs = { a, b, weight };
            if (kernels == null || Array.Exists(inputs, t => t.Storage is not CudaStorage || !t.IsContiguous() || t.ElementType != DType.Float32))
                return false;
            long cols = a.Sizes[^1];
            if (weight.ElementCount() != cols || a.ElementCount() != b.ElementCount() || a.ElementCount() % cols != 0 || cols > int.MaxValue)
                return false;
            int rows = checked((int)(a.ElementCount() / cols));
            Tensor total = new Tensor(Allocator, DType.Float32, a.Sizes), norm = new Tensor(Allocator, DType.Float32, a.Sizes);
            Launch(inputs, new[] { total, norm }, (k, stream) => k.LaunchAddRmsNorm(Ptr(a), Ptr(b), Ptr(weight), Ptr(total), Ptr(norm), rows,
                (int)cols, eps, stream));
            (sum, normed) = (total, norm);
            return true;
        }

        /// <summary>The gated delta rule's regions read views of a packed in-projection where they are (<see cref="TryRows"/>);
        /// the others take contiguous inputs.</summary>
        protected override bool RegionReadsStridedInputs(GraphFusion kind) => kind is GraphFusion.DeltaNetPrep or GraphFusion.GatedRmsNorm;

        /// <summary>
        /// Whether <paramref name="t"/> is rows of <paramref name="rowFloats"/> contiguous floats, each <paramref name="pitch"/>
        /// floats after the previous (pitch &gt;= rowFloats; a contiguous tensor has pitch rowFloats): the layout of a column
        /// block of a packed product's rows.
        /// </summary>
        private static bool TryRows(Tensor t, long rowFloats, out long pitch)
        {
            pitch = rowFloats;
            if (rowFloats <= 0 || t.ElementCount() % rowFloats != 0) return false;
            if (t.IsContiguous()) return true;
            int dim = t.DimensionCount - 1;
            long inner = 1;
            for (; dim >= 0 && inner < rowFloats; dim--)
            {
                if (t.Sizes[dim] != 1 && t.Strides[dim] != inner) return false;
                inner *= t.Sizes[dim];
            }
            if (inner != rowFloats) return false;
            long expected = -1;
            for (; dim >= 0; dim--)
            {
                if (t.Sizes[dim] == 1) continue;
                if (expected < 0) pitch = t.Strides[dim];
                else if (t.Strides[dim] != expected) return false;
                expected = t.Strides[dim] * t.Sizes[dim];
            }
            return pitch >= rowFloats;
        }

        private static IntPtr Ptr(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

        private static void RequireDevice(Tensor t, int device, string role, int index)
        {
            if (t.Storage is CudaStorage s && s.DeviceId != device)
                throw new InvalidOperationException(
                    $"CUDA graph kernel {role} {index} [{string.Join(", ", t.Sizes.ToArray())}] lives on device {s.DeviceId}; the executor runs on device {device}.");
        }

        /// <summary>Make the inputs and the output device-current, launch on the allocator's stream, mark the output written.</summary>
        private void Launch(Tensor[] inputs, Tensor output, Action<CudaGraphKernels, IntPtr> launch)
            => Launch(inputs, new[] { output }, launch);

        private void Launch(Tensor[] inputs, Tensor[] outputs, Action<CudaGraphKernels, IntPtr> launch)
        {
            var cuda = (CudaAllocator)Allocator;
            // A kernel reads its operands by device pointer in this executor's context: an operand on another GPU (a
            // layer-split stage reading another stage's memory) would be an illegal access, reported only later.
            for (int i = 0; i < inputs.Length; i++) RequireDevice(inputs[i], cuda.DeviceId, "input", i);
            for (int i = 0; i < outputs.Length; i++) RequireDevice(outputs[i], cuda.DeviceId, "output", i);
            foreach (Tensor t in inputs) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            foreach (Tensor t in outputs) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            cuda.Context.MakeCurrent();
            launch(Kernels, cuda.Stream.Handle);
            foreach (Tensor t in outputs) ((CudaStorage)t.Storage).MarkDeviceModified();
        }

        /// <summary>ggml's gated delta rule on the device (one warp per state row).</summary>
        protected override Tensor GatedDeltaNet(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor state, int snapshots)
        {
            long[] nq = Ne(q), nv = Ne(v), ng = Ne(g);
            int s = (int)nv[0], heads = (int)nv[1], tokens = (int)nv[2], sequences = (int)nv[3], keyHeads = (int)nq[1];
            int slots = Math.Max(1, snapshots);
            bool kda = ng[0] == s && s > 1;
            CudaGraphKernels kernels = s % 32 == 0 && s <= 256 && nq[0] == s && heads % keyHeads == 0 ? Kernels : null;
            Tensor[] inputs = { q, k, v, g, beta, state };
            if (kernels == null || Array.Exists(inputs, t => t.Storage is not CudaStorage || !t.IsContiguous() || t.ElementType != DType.Float32))
                return base.GatedDeltaNet(q, k, v, g, beta, state, snapshots);

            long rowFloats = (long)sequences * tokens * heads * s;
            long total = rowFloats + (long)slots * sequences * heads * s * s;
            var cuda = (CudaAllocator)Allocator;
            // A decode row whose state is the tail of a [row outputs | state] buffer (a slot's backing): the result goes
            // there, the new state over the old (each warp reads its state row before writing it back), and the alias
            // that follows finds it in place - a 1 MB state copy per layer and token otherwise (Qwen3.5-0.8B: 32% of
            // the decode GPU time).
            bool inPlace = slots == 1 && tokens == 1 && sequences == 1 && state.Storage is CudaStorage held
                && state.StorageOffset == rowFloats && held.ElementCount == total && held.DeviceId == cuda.DeviceId;
            Tensor result = inPlace
                ? new Tensor(new[] { total }, new[] { 1L }, state.Storage, 0)
                : new Tensor(Allocator, DType.Float32, total);
            foreach (Tensor t in inputs) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            var output = (CudaStorage)result.Storage;
            output.EnsureDeviceCurrent();
            IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);
            cuda.Context.MakeCurrent();
            // A prompt chunk of 128-wide heads with one state slot runs in chunks of 16 rows on the tensor cores (the WY
            // form; 2.9 ms a layer for 3.6K rows on the rows4 kernel, Qwen3.8-27B on an RTX PRO 4500).
            if (tokens > 1 && slots == 1 && !kda && s == CudaSsmPrefillKernels.GdnHeadDim
                && cuda.SsmPrefillKernels is { RunsGatedDeltaNetChunks: true } ssm)
                ssm.LaunchGatedDeltaNetChunks(P(q), P(k), P(v), P(g), P(beta), P(state), output.DevicePtrAtElement(rowFloats),
                    output.DevicePtrAtElement(0), keyHeads, heads, tokens, sequences, cuda.Stream.Handle);
            // A chunk of rows reads each token's q and k once per four state rows; a decode row keeps the one-row kernel.
            else if (tokens > 1)
                kernels.LaunchGatedDeltaNetRows4(P(q), P(k), P(v), P(g), P(beta), P(state), output.DevicePtrAtElement(0),
                    s, heads, keyHeads, tokens, sequences, slots, kda, cuda.Stream.Handle);
            else
                kernels.LaunchGatedDeltaNet(P(q), P(k), P(v), P(g), P(beta), P(state), output.DevicePtrAtElement(0),
                    s, heads, keyHeads, tokens, sequences, slots, kda, cuda.Stream.Handle);
            output.MarkDeviceModified();
            return result;
        }

        /// <summary>
        /// A SwiGLU feed-forward with a fused gate|up weight (<see cref="GraphFusion.SwigluFfn"/>): the gate|up product, then
        /// silu(gate) * up read from its rows in place (the region's ops copied both halves whole first: 0.5 GB a layer
        /// for a 3.6K-token prompt of Qwen3.8-27B), and the down product. An NVFP4 down projection on the FP4 tensor cores
        /// takes the activation straight into its quantized planes. False (the ops run) for separate gate and up weights.
        /// </summary>
        private bool TrySwigluFfn(Tensor[] inputs, IGraphWeight[] weights, long[] ip, Tensor[] results)
        {
            if (ip != null && ip.Length > 0 && ip[0] == 1) return false;
            if (weights.Length < 3 || weights[1] is not { } gateUp || weights[2] is not { } down) return false;
            Tensor x = inputs[0];
            long ff = gateUp.Ne1 / 2;
            if (x?.Storage is not CudaStorage || x.ElementType != DType.Float32 || !x.IsContiguous() || gateUp.Ne2 > 1 || down.Ne2 > 1
                || gateUp.Ne1 % 2 != 0 || down.Ne0 != ff || x.ElementCount() % gateUp.Ne0 != 0 || ff > int.MaxValue)
                return false;
            var cuda = (CudaAllocator)Allocator;
            long rows = x.ElementCount() / gateUp.Ne0;
            using Tensor x2 = x.View(rows, gateUp.Ne0);
            using Tensor gu = QuantizedMatMul(gateUp, x2);   // [rows, 2 ff], gate first
            var y = new Tensor(Allocator, DType.Float32, rows, down.Ne1);
            if (CudaQuantizedOps.TryAddmmSwiGluNvfp4(y, gu, down.CacheKey, down.Data, down.Format, down.Ne0, down.Ne1, down.Bytes))
            {
                results[0] = y;
                return true;
            }
            y.Dispose();
            using var act = new Tensor(Allocator, DType.Float32, rows, ff);
            Launch(new[] { gu }, act, (_, stream) => cuda.Kernels.LaunchSiLUMulSplitF32(Ptr(gu), Ptr(act), checked((int)rows), (int)ff, stream));
            results[0] = QuantizedMatMul(down, act);
            return true;
        }

        // ---------------------------------------------------------------- host experts

        private CudaExpertDoorbell _doorbell;

        /// <summary>A stack of offloaded experts as the device reads its tier: one pointer table per projection (an expert
        /// outside the tier keeps a null entry no kernel reads) and the tier as a byte per expert.</summary>
        private sealed class HostedTables
        {
            public Tensor Gate, Up, Down, OnDevice;
            /// <summary>The layer's routes counted on the device since the last harvest ([experts] uint).</summary>
            public Tensor Counts;
            public IGraphExpertHost Host;
            public int Layer, Experts;
            /// <summary>The layer's stacks, and the tables' host copies (pointer words per matrix, the on-device flags):
            /// an adaptive trade rewrites them and uploads them again.</summary>
            public IGraphExpertWeights[] Sets;
            public int[][] Words;
            public byte[] Mask;
            /// <summary>A long prompt chunk's staged-ahead path (CudaGraphExecutor.PromptPrefetch): the host experts in
            /// slot order, their matrices' bytes, the tables per arena half and the generation they were built for, and
            /// the routes counted on the device since the last harvest.</summary>
            public int[] HostList;
            public long PerExpert;
            public long[] MemberBytes;
            public Tensor[][] Half;
            public long[] HalfGeneration;
            public Tensor PromptCounts;
        }

        private long _countsHarvestedAt = System.Diagnostics.Stopwatch.GetTimestamp();

        /// <summary>Seconds between harvests of the device's route counts (and adaptive trades after each).</summary>
        private const double HarvestSeconds = 0.25;

        /// <summary>Trades an adaptive round makes at most (each a few MB copied before the next step).</summary>
        private const int AdaptSwapsPerRound = 64;

        /// <summary>
        /// An adaptive round (<see cref="IGraphExpertHost.ProposeSwaps"/>): each trade's incoming expert copied into the
        /// outgoing one's device slot from the stack's (page-locked) mapping, the resident copies re-keyed and the layer's
        /// tables and flags rewritten, all on the stream before the next step: a recorded step reads the tables when it runs.
        /// </summary>
        private void AdaptTiers(HostedTables[] layers)
        {
            if (layers.Length == 0) return;
            var cuda = (CudaAllocator)Allocator;
            IGraphExpertHost host = layers[0].Host;
            IReadOnlyList<(int Layer, int Out, int In)> swaps = host.ProposeSwaps(AdaptSwapsPerRound);
            if (swaps.Count == 0) return;
            var byLayer = new Dictionary<int, HostedTables>();
            foreach (HostedTables t in layers) byLayer[t.Layer] = t;
            var dirty = new HashSet<HostedTables>();
            var fromKeys = new IntPtr[3];
            var toKeys = new IntPtr[3];
            var types = new int[3];
            var bytes = new long[3];
            var ptrs = new IntPtr[3];
            cuda.Context.MakeCurrent();
            foreach ((int layer, int outOf, int into) in swaps)
            {
                if (!byLayer.TryGetValue(layer, out HostedTables t) || t.Sets == null || t.Mask[outOf] == 0 || t.Mask[into] != 0) continue;
                bool ok = true;
                for (int m = 0; m < 3 && ok; m++)
                {
                    IGraphExpertWeights set = t.Sets[m];
                    IGraphWeight a = set.Expert(outOf), b = set.Expert(into);
                    fromKeys[m] = a.CacheKey;
                    toKeys[m] = b.CacheKey;
                    types[m] = set.Format;
                    bytes[m] = b.Bytes;
                    ok = a.Bytes == b.Bytes && set.Data != IntPtr.Zero;
                }
                if (!ok || !CudaQuantizedOps.TryMoveResident(cuda, fromKeys, toKeys, types, bytes, ptrs)) continue;
                for (int m = 0; m < 3; m++)
                {
                    IGraphExpertWeights set = t.Sets[m];
                    IntPtr source = set.Data + (nint)(set.Bytes / set.Ne2 * into);
                    CudaHostPins.CopyToDevice(ptrs[m], source, bytes[m], cuda.Stream.Handle).ThrowOnError();
                    long p = ptrs[m].ToInt64();
                    t.Words[m][2 * into] = unchecked((int)p);
                    t.Words[m][2 * into + 1] = unchecked((int)(p >> 32));
                    t.Words[m][2 * outOf] = t.Words[m][2 * outOf + 1] = 0;
                }
                t.Mask[into] = 1;
                t.Mask[outOf] = 0;
                TierMoved(t);
                host.CommitSwap(layer, outOf, into);
                dirty.Add(t);
            }
            foreach (HostedTables t in dirty)
            {
                Tensor[] tables = { t.Gate, t.Up, t.Down };
                for (int m = 0; m < 3; m++)
                {
                    tables[m].SetElementsAsInt(t.Words[m]);
                    ((CudaStorage)tables[m].Storage).EnsureDeviceCurrent();
                }
                unsafe
                {
                    fixed (byte* mask = t.Mask)
                        CudaDriverApi.cuMemcpyHtoDAsync(Ptr(t.OnDevice), (IntPtr)mask, new UIntPtr((ulong)t.Mask.Length), cuda.Stream.Handle).ThrowOnError();
                }
                ((CudaStorage)t.OnDevice.Storage).MarkDeviceModified();
            }
        }

        /// <summary>
        /// The routes the doorbell layers counted on the device, handed to their expert hosts (the routing profile) and
        /// zeroed, at most every two seconds and never inside a recording: every route counts, the device's as the host's,
        /// though a step whose routes all stay on the device never rings.
        /// </summary>
        private void HarvestRouteCounts(bool force = false)
        {
            var cuda = (CudaAllocator)Allocator;
            long now = System.Diagnostics.Stopwatch.GetTimestamp();
            if (!force && now - _countsHarvestedAt < HarvestSeconds * System.Diagnostics.Stopwatch.Frequency) return;
            if (CudaGraphCapture.IsCapturing(cuda)) return;
            _countsHarvestedAt = now;
            HarvestPromptCounts();
            HostedTables[] layers;
            lock (_hostedTables) layers = new List<HostedTables>(_hostedTables.Values).ToArray();
            var ids = new List<int>();
            foreach (HostedTables t in layers)
            {
                if (t.Counts == null) continue;
                ((CudaStorage)t.Counts.Storage).MarkDeviceModified();
                int[] counts = t.Counts.GetElementsAsInt(t.Experts);
                ids.Clear();
                for (int e = 0; e < counts.Length; e++)
                    for (int c = 0; c < counts[e]; c++) ids.Add(e);
                if (ids.Count == 0) continue;
                cuda.Context.MakeCurrent();
                CudaDriverApi.cuMemsetD32Async(Ptr(t.Counts), 0, new UIntPtr((ulong)t.Experts), cuda.Stream.Handle).ThrowOnError();
                ((CudaStorage)t.Counts.Storage).MarkDeviceModified();
                t.Host.Record(t.Layer, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(ids));
            }
            if (!force) AdaptTiers(layers);
        }

        private readonly Dictionary<IGraphHostedExperts, HostedTables> _hostedTables = new(ReferenceEqualityComparer.Instance);

        /// <summary>
        /// A decode row's host experts on the device's stream (<see cref="CudaExpertDoorbell"/>): the routes the layer's tier
        /// holds run on the MoE kernels through device tables; the others are rung to the host, which computes them while
        /// the device works and the wait kernel adds them. Nothing returns to the host, so the step can be recorded. Other
        /// steps (prompt chunks, verify windows, routes read on the host) take the executor's own path.
        /// </summary>
        protected override Tensor HostExperts(Tensor x, Value selected, Value weightValue, Value gateValue, Value upValue, Value downValue,
            int layer, RunState run)
        {
            if (gateValue.Weight is IGraphHostedExperts gate && upValue.Weight is IGraphExpertWeights up
                && downValue.Weight is IGraphExpertWeights down)
            {
                // Every MoE layer the executor meets is remembered, so a long prompt chunk stages each next layer ahead.
                if (!_moeLayers.ContainsKey(layer) && gate.Format == up.Format)
                    _moeLayers[layer] = Tables(gate, up, down, layer);
                SizeStagingArenas(gate, up, down);
                if (TryDoorbellExperts(x, selected, weightValue, gate, up, down, layer, out Tensor result)
                    || TryStagedExperts(x, selected, weightValue, gate, up, down, layer, out result))
                    return result;
            }
            return base.HostExperts(x, selected, weightValue, gateValue, upValue, downValue, layer, run);
        }

        private IntPtr _stagingArena;
        private long _stagingBytes;

        /// <summary>
        /// A prompt chunk's routed experts (rows past what the host computes): routed on the host, each selected expert run
        /// once over its rows, from its tier copy where the device holds one and otherwise from a staging arena the expert's
        /// bytes are copied into from the model file's mapping (page-locked at load where the host allows: DMA speed). The arena holds as many experts as
        /// device memory allows; a layer whose staged experts do not fit runs in batches. Every route is computed (the
        /// route-tail skip is for decode rows).
        /// </summary>
        private bool TryStagedExperts(Tensor x, Value selected, Value weightValue, IGraphHostedExperts gate, IGraphExpertWeights up,
            IGraphExpertWeights down, int layer, out Tensor result)
        {
            result = null;
            var cuda = (CudaAllocator)Allocator;
            int hidden = (int)gate.Ne0, ff = (int)gate.Ne1, experts = (int)gate.Ne2;
            if (x.Storage is not CudaStorage || x.ElementType != DType.Float32 || selected.Tensor == null && selected.Ints == null
                || cuda.Kernels == null || !CudaQuantizedOps.SupportsMatmulType(gate.Format) || !CudaQuantizedOps.SupportsMatmulType(up.Format)
                || !CudaQuantizedOps.SupportsMatmulType(down.Format) || up.Ne0 != hidden || up.Ne1 != ff || down.Ne0 != ff || down.Ne1 != hidden)
                return false;
            long[] nid = Ne(selected.Tensor);
            int k = (int)nid[0], rows = (int)nid[1];
            if ((long)rows * hidden != x.ElementCount() || !gate.Host.StreamsStep(layer, rows))
                return false;
            if (TryPrefetchedExperts(x, selected, weightValue, gate, up, down, layer, rows, k, out result))
                return true;
            int[] ids = Ints(selected);
            float[] weights = weightValue.Floats ?? weightValue.Tensor.GetElementsAsFloat(rows * k);
            gate.Host.Record(layer, ids, GraphExpertAdapt.Weight(rows));

            // The routes grouped by expert: rowOf / weightOf from start[e], count[e] of them.
            var count = new int[experts];
            foreach (int e in ids) if (e >= 0 && e < experts) count[e]++;
            var start = new int[experts + 1];
            for (int e = 0; e < experts; e++) start[e + 1] = start[e] + count[e];
            var rowOf = new int[start[experts]];
            var weightOf = new float[rowOf.Length];
            var fill = (int[])start.Clone();
            for (int i = 0; i < ids.Length; i++)
            {
                int e = ids[i];
                if (e < 0 || e >= experts) continue;
                rowOf[fill[e]] = i / k;
                weightOf[fill[e]++] = weights[i];
            }

            // Where each selected expert's three matrices are read: the device tier, or staged.
            IGraphExpertWeights[] sets = { gate, up, down };
            var resident = new IntPtr[3 * experts];
            var staged = new List<int>();
            long[] memberBytes = new long[3];
            for (int e = 0; e < experts; e++)
            {
                if (count[e] == 0) continue;
                bool all = true;
                for (int m = 0; m < 3; m++)
                {
                    IGraphWeight member = sets[m].Expert(e);
                    memberBytes[m] = member.Bytes;
                    if (!CudaQuantizedOps.TryGetResidentDevicePtr(cuda, member.CacheKey, out IntPtr p, out int type) || type != sets[m].Format)
                        all = false;
                    else resident[3 * e + m] = p;
                }
                if (!all) staged.Add(e);
            }
            long perExpert = memberBytes[0] + memberBytes[1] + memberBytes[2];
            // A staged expert's bytes: its place in the stack's host mapping (a member's own host copy may be released).
            IntPtr Source(int m, int e)
            {
                IGraphExpertWeights set = sets[m];
                return set.Data != IntPtr.Zero ? set.Data + (nint)(set.Bytes / set.Ne2 * e) : sets[m].Expert(e).Data;
            }
            foreach (int e in staged)
                for (int m = 0; m < 3; m++)
                    if (Source(m, e) == IntPtr.Zero) return false;
            if (staged.Count > 0 && !EnsureStagingArena(perExpert, staged.Count))
                return false;

            using Tensor x2 = x.View(rows, hidden);
            using var rowIdx = new Tensor(Allocator, DType.Int32, Math.Max(1, rowOf.Length));
            using var routeWeights = new Tensor(Allocator, DType.Float32, Math.Max(1, rowOf.Length));
            if (rowOf.Length > 0)
            {
                rowIdx.SetElementsAsInt(rowOf);
                routeWeights.SetElementsAsFloat(weightOf);
            }
            var output = new Tensor(Allocator, DType.Float32, rows, hidden);
            Ops.Fill(output, 0f);
            void Run(int e, IntPtr g, IntPtr u, IntPtr d)
            {
                int n = count[e];
                using Tensor at = rowIdx.Narrow(0, start[e], n), w = routeWeights.Narrow(0, start[e], n);
                using Tensor xs = Ops.IndexSelect(null, x2, at);
                using var gateOut = new Tensor(Allocator, DType.Float32, n, ff);
                using var upOut = new Tensor(Allocator, DType.Float32, n, ff);
                CudaQuantizedOps.AddmmResidentToFloat32(gateOut, xs, g, gate.Format, hidden, ff);
                CudaQuantizedOps.AddmmResidentToFloat32(upOut, xs, u, up.Format, hidden, ff);
                Ops.SiLUMul(gateOut, gateOut, upOut);
                using var downOut = new Tensor(Allocator, DType.Float32, n, hidden);
                CudaQuantizedOps.AddmmResidentToFloat32(downOut, gateOut, d, down.Format, ff, hidden);
                if (!CudaFusedOps.TryMoEScatterAddWeightedRows(output, downOut, at, w))
                    throw new InvalidOperationException($"layer {layer}: the weighted scatter of expert {e}'s rows failed");
            }
            // Staged experts that fit the arena at once: every route through the grouped MoE kernels (the engines' routes:
            // tensor-core grouped GEMMs for a long chunk, per-slot kernels for a short one) over tables of the step's expert
            // copies, else the batched per-row kernels; a handful of launches an expert only where neither takes them.
            if ((long)staged.Count * perExpert <= _stagingBytes
                && TryStagedTables(selected, weightValue, sets, resident, staged, memberBytes, perExpert, Source, experts, layer))
            {
                if (TryGroupedMoe(x2, output, selected.Tensor, weightValue.Tensor, sets, _stepTables, rows, k, hidden, ff, experts)
                    || (rows <= StagedPerRowMaxRows && TryStagedPerRow(x2, output, selected, weightValue, sets, rows, k, hidden, ff, experts)))
                {
                    result = output;
                    return true;
                }
            }
            try
            {
                for (int e = 0; e < experts; e++)
                    if (count[e] > 0 && resident[3 * e] != IntPtr.Zero && staged.IndexOf(e) < 0)
                        Run(e, resident[3 * e], resident[3 * e + 1], resident[3 * e + 2]);
                // The staged experts, as many at a time as the arena holds: copied in on the stream (after the previous
                // batch's kernels read the arena), then run.
                int perBatch = (int)Math.Max(1, _stagingBytes / perExpert);
                cuda.Context.MakeCurrent();
                for (int b = 0; b < staged.Count; b += perBatch)
                {
                    int batch = Math.Min(perBatch, staged.Count - b);
                    for (int j = 0; j < batch; j++)
                    {
                        int e = staged[b + j];
                        long offset = (long)j * perExpert;
                        for (int m = 0; m < 3; m++)
                        {
                            int rc = CudaHostPins.CopyToDevice(_stagingArena + (nint)offset, Source(m, e), memberBytes[m], cuda.Stream.Handle);
                            if (rc != 0)
                                throw new InvalidOperationException($"layer {layer}: staging expert {e} matrix {m} failed (CUDA error {rc}): " +
                                    $"{memberBytes[m]} bytes from 0x{Source(m, e).ToInt64():x} (stack 0x{sets[m].Data.ToInt64():x}, {sets[m].Bytes} " +
                                    $"bytes) to arena offset {offset} of {_stagingBytes}; {CudaHostPins.PinnedBytes} bytes page-locked");
                            resident[3 * e + m] = _stagingArena + (nint)offset;
                            offset += memberBytes[m];
                        }
                    }
                    for (int j = 0; j < batch; j++)
                    {
                        int e = staged[b + j];
                        Run(e, resident[3 * e], resident[3 * e + 1], resident[3 * e + 2]);
                    }
                }
            }
            catch
            {
                output.Dispose();
                throw;
            }
            result = output;
            return true;
        }

        /// <summary>Rows a staged step runs through the batched per-row MoE kernels; longer chunks run each expert once over
        /// its rows (the weights read once for all of them).</summary>
        private const int StagedPerRowMaxRows = 64;

        private Tensor[] _stepTables;

        /// <summary>
        /// The staged experts copied in at once and this step's pointer tables built over them and the tier's copies
        /// (<see cref="_stepTables"/>). False, with nothing launched, where the routes are not on the device.
        /// </summary>
        private bool TryStagedTables(Value selected, Value weightValue, IGraphExpertWeights[] sets, IntPtr[] resident, List<int> staged,
            long[] memberBytes, long perExpert, Func<int, int, IntPtr> source, int experts, int layer)
        {
            var cuda = (CudaAllocator)Allocator;
            Tensor sel = selected.Tensor, w = weightValue.Tensor;
            if (sel?.Storage is not CudaStorage || w?.Storage is not CudaStorage || sel.ElementType != DType.Int32 || w.ElementType != DType.Float32
                || !sel.IsContiguous() || !w.IsContiguous())
                return false;
            cuda.Context.MakeCurrent();
            for (int j = 0; j < staged.Count; j++)
            {
                int e = staged[j];
                long offset = (long)j * perExpert;
                for (int m = 0; m < 3; m++)
                {
                    int rc = CudaHostPins.CopyToDevice(_stagingArena + (nint)offset, source(m, e), memberBytes[m], cuda.Stream.Handle);
                    if (rc != 0)
                        throw new InvalidOperationException($"layer {layer}: staging expert {e} matrix {m} failed (CUDA error {rc})");
                    resident[3 * e + m] = _stagingArena + (nint)offset;
                    offset += memberBytes[m];
                }
            }
            _stepTables ??= new[] { new Tensor(Allocator, DType.Int32, 2L * experts), new Tensor(Allocator, DType.Int32, 2L * experts),
                new Tensor(Allocator, DType.Int32, 2L * experts) };
            for (int m = 0; m < 3; m++)
            {
                if (_stepTables[m].ElementCount() < 2L * experts)
                {
                    _stepTables[m].Dispose();
                    _stepTables[m] = new Tensor(Allocator, DType.Int32, 2L * experts);
                }
                var words = new int[_stepTables[m].ElementCount()];
                for (int e = 0; e < experts; e++)
                {
                    long p = resident[3 * e + m].ToInt64();
                    words[2 * e] = unchecked((int)p);
                    words[2 * e + 1] = unchecked((int)(p >> 32));
                }
                _stepTables[m].SetElementsAsInt(words);
                ((CudaStorage)_stepTables[m].Storage).EnsureDeviceCurrent();
            }
            return true;
        }

        /// <summary>The chunk's routes through the batched per-row MoE kernels over this step's tables (one launch a
        /// projection).</summary>
        private bool TryStagedPerRow(Tensor x2, Tensor output, Value selected, Value weightValue, IGraphExpertWeights[] sets, int rows,
            int k, int hidden, int ff, int experts)
        {
            int gateType = sets[0].Format, downType = sets[2].Format;
            if (sets[1].Format != gateType || !CudaQuantizedOps.SupportsQuantizedType(gateType) || !CudaQuantizedOps.SupportsQuantizedType(downType))
                return false;
            long routes = (long)rows * k;
            bool gateUpDp4a = CudaQuantizedOps.IsMoeDp4aEnabledForType(gateType) && CudaFusedOps.IsMoeDp4aDimensionSupported(gateType, hidden);
            bool downDp4a = CudaQuantizedOps.IsMoeDp4aEnabledForType(downType) && CudaFusedOps.IsMoeDp4aDimensionSupported(downType, ff);
            using Tensor ids = selected.Tensor.View(routes), routeWeights = weightValue.Tensor.View(routes);
            using var devSel = Ops.NewContiguous(ids);   // the kernels' scratch for the routes they run (the router's ids kept)
            using var gateOut = new Tensor(Allocator, DType.Float32, routes, ff);
            using var upOut = new Tensor(Allocator, DType.Float32, routes, ff);
            using Tensor inputQ8 = gateUpDp4a ? new Tensor(Allocator, DType.UInt8, (long)rows * (hidden / 32) * CudaFusedOps.Q81BlockBytes) : null;
            using Tensor gateQ8 = downDp4a ? new Tensor(Allocator, DType.UInt8, routes * (ff / 32) * CudaFusedOps.Q81BlockBytes) : null;
            return CudaFusedOps.TryMoEExpertFFNPrefillSwiGLU(null, x2, output, devSel, routeWeights, gateOut, upOut, IntPtr.Zero,
                Ptr(_stepTables[0]), Ptr(_stepTables[1]), Ptr(_stepTables[2]), gateType, downType, experts, k, hidden, ff, rows, null,
                IntPtr.Zero, inputQ8, gateQ8, gateUpDp4a, downDp4a);
        }

        private Dsv4Kernels _moeKernels;
        private bool _moeKernelsMissing;

        /// <summary>The grouped MoE kernels' module (the direct engines' expert kernels) in this allocator's context, or
        /// null (reported once) where it cannot load.</summary>
        private Dsv4Kernels MoeKernels
        {
            get
            {
                if (_moeKernels != null || _moeKernelsMissing) return _moeKernels;
                try
                {
                    ((CudaAllocator)Allocator).Context.MakeCurrent();
                    _moeKernels = Dsv4Kernels.Create();
                }
                catch (Exception ex) when (ex is InvalidOperationException or CudaException or System.IO.IOException)
                {
                    _moeKernelsMissing = true;
                    Console.Error.WriteLine($"[cuda] the grouped MoE kernels are unavailable ({ex.Message}); prompt experts run per expert.");
                }
                return _moeKernels;
            }
        }

        /// <summary>
        /// A chunk's routed experts on the grouped MoE kernels (CudaMoe: the routes grouped by expert on the device, then
        /// tensor-core grouped GEMMs, F16 operands and F32 accumulation, for a long chunk, the per-slot kernels for a short
        /// one, and the weighted scatter) over this step's pointer tables, the router's ids and weights as they are.
        /// False, with nothing launched, for a layout or width the kernels do not take.
        /// </summary>
        private bool TryGroupedMoe(Tensor x2, Tensor output, Tensor sel, Tensor w, IGraphExpertWeights[] sets, Tensor[] tables, int rows,
            int k, int hidden, int ff, int experts)
        {
            var cuda = (CudaAllocator)Allocator;
            Dsv4Kernels dk = MoeKernels;
            int gateType = sets[0].Format, upType = sets[1].Format, downType = sets[2].Format;
            if (!GroupedTakes(sets, rows, hidden, ff))
                return false;
            var owned = new List<Tensor>();
            try
            {
                var scratch = new CudaMoeScratch((type, sizes) =>
                {
                    var t = new Tensor(Allocator, type, sizes);
                    owned.Add(t);
                    return t;
                }, rows, experts, k, hidden, ff);
                long routes = (long)rows * k;
                using (Tensor ids = sel.View(routes), weights = w.View(routes))
                using (Tensor selDst = scratch.Sel.View(routes), wDst = scratch.SelW.View(routes))
                {
                    Ops.Copy(selDst, ids);
                    Ops.Copy(wDst, weights);
                }
                long RowBytes(IGraphExpertWeights set) => set.Expert(0).Bytes / set.Ne1;
                var gate = new DeviceWeight { Type = gateType, Ne0 = hidden, Ne1 = ff, RowBytes = RowBytes(sets[0]), Table = Ptr(tables[0]) };
                var up = new DeviceWeight { Type = upType, Ne0 = hidden, Ne1 = ff, RowBytes = RowBytes(sets[1]), Table = Ptr(tables[1]) };
                var down = new DeviceWeight { Type = downType, Ne0 = ff, Ne1 = hidden, RowBytes = RowBytes(sets[2]), Table = Ptr(tables[2]) };
                foreach (Tensor t in new[] { x2, output }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
                cuda.Context.MakeCurrent();
                CudaMoe.Experts(dk, cuda.Kernels, scratch, gate, up, down, x2, null, output, rows, k, experts, hidden, ff, 0f,
                    GroupedPerSlotMaxRows, cuda.Stream.Handle);
                ((CudaStorage)output.Storage).MarkDeviceModified();
                return true;
            }
            finally
            {
                // The kernels are queued on the allocator's stream: the scratch returns to its pool for later work there.
                foreach (Tensor t in owned) t.Dispose();
            }
        }

        /// <summary>Whether the grouped MoE kernels take a chunk of <paramref name="rows"/> rows of these experts.</summary>
        private bool GroupedTakes(IGraphExpertWeights[] sets, int rows, int hidden, int ff)
        {
            Dsv4Kernels dk = MoeKernels;
            int gateType = sets[0].Format, downType = sets[2].Format;
            return dk != null && ((CudaAllocator)Allocator).Kernels != null && gateType == sets[1].Format
                && CudaMoe.GroupedSupportsType(gateType, dk.DecodesLowBitIqExperts, dk.DecodesLegacyQuants)
                && CudaMoe.GroupedSupportsType(downType, dk.DecodesLowBitIqExperts, dk.DecodesLegacyQuants)
                && CudaMoe.SelectRoute(rows, GroupedPerSlotMaxRows, dk.HasMoeMma, dk.DecodesLowBitIqExperts, gateType, downType, hidden, ff,
                    dk.DecodesLegacyQuants) != CudaMoe.ExpertRoute.Warp;
        }

        /// <summary>Rows the grouped MoE path runs on its per-slot kernels (the engines' decode-window bound).</summary>
        private const int GroupedPerSlotMaxRows = 16;

        /// <summary>Most bytes the staging arena takes (the placement's reserve for it: one layer's experts at most).</summary>
        private const long StagingArenaCapBytes = 2L << 30;

        /// <summary>Device memory the arena leaves free for the step's own buffers.</summary>
        private const long StagingFreeFloorBytes = 1L << 30;

        /// <summary>A staging arena for at least one and up to <paramref name="wanted"/> experts of
        /// <paramref name="perExpert"/> bytes, within <see cref="StagingArenaCapBytes"/> and the device's free memory less
        /// <see cref="StagingFreeFloorBytes"/>; false when not even one fits.</summary>
        private bool EnsureStagingArena(long perExpert, int wanted)
        {
            long need = perExpert * wanted;
            // Big enough, or as big as the device allowed (a later layer then runs in batches rather than re-asking).
            if (_stagingBytes >= need || (_stagingCapped && _stagingBytes >= perExpert)) return true;
            var cuda = (CudaAllocator)Allocator;
            cuda.Context.MakeCurrent();
            cuda.Stream.Synchronize();
            if (_stagingArena != IntPtr.Zero)
            {
                CudaDriverApi.cuMemFree(_stagingArena);
                _stagingArena = IntPtr.Zero;
                _stagingBytes = 0;
            }
            CudaDriverApi.cuMemGetInfo(out UIntPtr free, out _).ThrowOnError();
            if ((long)free.ToUInt64() - StagingFreeFloorBytes < Math.Min(need, StagingArenaCapBytes))
            {
                // The memory pool keeps the blocks earlier steps returned: release them before settling for a smaller
                // arena (a capped arena runs a long chunk's staged experts one at a time).
                cuda.TrimPool();
                cuda.Context.MakeCurrent();
                CudaDriverApi.cuMemGetInfo(out free, out _).ThrowOnError();
            }
            long room = Math.Min(StagingArenaCapBytes, (long)free.ToUInt64() - StagingFreeFloorBytes);
            long bytes = Math.Min(need, room) / perExpert * perExpert;
            if (bytes < perExpert) return false;
            if (CudaDriverApi.cuMemAlloc(out IntPtr arena, new UIntPtr((ulong)bytes)) != 0) return false;
            _stagingArena = arena;
            _stagingBytes = bytes;
            _stagingCapped = bytes < need;
            return true;
        }

        private bool _stagingCapped;
        private Action _onLaunching;

        /// <summary>
        /// The model's resources this executor holds on the device and the host, released with the model (before its
        /// mappings and tiers go: a table outliving them would serve the next model mapped at the same addresses): the
        /// doorbell stopped and unsubscribed, the last route counts handed to the routing profile, the staging arena and the
        /// expert tables freed.
        /// </summary>
        public void Dispose()
        {
            var cuda = (CudaAllocator)Allocator;
            if (_onLaunching != null) cuda.RecordedStepLaunching -= _onLaunching;
            _onLaunching = null;
            try { HarvestRouteCounts(force: true); }
            catch (CudaException) { }
            _doorbell?.Dispose();
            _doorbell = null;
            DisposePrefetch();
            cuda.Context.MakeCurrent();
            if (_stagingArena != IntPtr.Zero)
            {
                cuda.Stream.Synchronize();
                CudaDriverApi.cuMemFree(_stagingArena);
            }
            _stagingArena = IntPtr.Zero;
            _stagingBytes = 0;
            if (_stepTables != null)
                foreach (Tensor t in _stepTables) t.Dispose();
            _stepTables = null;
            foreach (Tensor t in _decodeScratchTensors) t.Dispose();
            _decodeScratchTensors.Clear();
            _decodeScratch.Clear();
            _moeKernels?.Dispose();
            _moeKernels = null;
            lock (_hostedTables)
            {
                foreach (HostedTables t in _hostedTables.Values)
                    foreach (Tensor tensor in new[] { t.Gate, t.Up, t.Down, t.OnDevice, t.Counts }) tensor?.Dispose();
                _hostedTables.Clear();
            }
        }

        private bool TryDoorbellExperts(Tensor x, Value selected, Value weightValue, IGraphHostedExperts gate, IGraphExpertWeights up,
            IGraphExpertWeights down, int layer, out Tensor result)
        {
            result = null;
            var cuda = (CudaAllocator)Allocator;
            Tensor sel = selected.Tensor, w = weightValue.Tensor;
            if (Kernels == null || cuda.Kernels == null || sel?.Storage is not CudaStorage || w?.Storage is not CudaStorage
                || x.Storage is not CudaStorage || sel.ElementType != DType.Int32 || w.ElementType != DType.Float32
                || x.ElementType != DType.Float32 || !sel.IsContiguous() || !w.IsContiguous() || !x.IsContiguous())
                return false;
            long[] nid = Ne(sel);
            int k = (int)nid[0], rows = (int)nid[1];
            int hidden = (int)gate.Ne0, ff = (int)gate.Ne1, experts = (int)gate.Ne2;
            if (rows < 1 || rows > CudaExpertDoorbell.MaxRows || x.ElementCount() != (long)rows * hidden || w.ElementCount() != (long)rows * k
                || up.Ne1 != ff
                || down.Ne0 != ff || down.Ne1 != hidden || gate.Format != up.Format
                || !CudaQuantizedOps.SupportsQuantizedType(gate.Format) || !CudaQuantizedOps.SupportsQuantizedType(down.Format))
                return false;
            HostedTables tables = Tables(gate, up, down, layer);
            if (_doorbell != null && (_doorbell.Hidden != hidden || _doorbell.RoutesPerRow != k))
                return false;   // another shape already has the executor's mailbox
            // The mailbox holds one request: the previous layer's host sum is added before this layer rings.
            FinishJoins();
            if (_doorbell == null)
            {
                _doorbell = new CudaExpertDoorbell(cuda, hidden, k);
                _onLaunching = () => { _doorbell?.Arm(); HarvestRouteCounts(); };
                cuda.RecordedStepLaunching += _onLaunching;
            }
            HarvestRouteCounts();
            _doorbell.Register(layer, gate.Host);
            _doorbell.Arm();

            bool gateUpDp4a = CudaQuantizedOps.IsMoeDp4aEnabledForType(gate.Format) && CudaFusedOps.IsMoeDp4aDimensionSupported(gate.Format, hidden);
            bool downDp4a = CudaQuantizedOps.IsMoeDp4aEnabledForType(down.Format) && CudaFusedOps.IsMoeDp4aDimensionSupported(down.Format, ff);
            long routes = (long)rows * k;
            using var devSel = new Tensor(Allocator, DType.Int32, routes);
            using var gateOut = new Tensor(Allocator, DType.Float32, routes, ff);
            using var upOut = new Tensor(Allocator, DType.Float32, routes, ff);
            using Tensor inputQ8 = gateUpDp4a ? new Tensor(Allocator, DType.UInt8, (long)rows * (hidden / 32) * CudaFusedOps.Q81BlockBytes) : null;
            using Tensor gateQ8 = downDp4a ? new Tensor(Allocator, DType.UInt8, routes * (ff / 32) * CudaFusedOps.Q81BlockBytes) : null;
            using Tensor x2 = x.View(rows, hidden);
            using Tensor w1 = w.View(routes);   // the fused kernels take the routes' weights as one vector
            var output = new Tensor(Allocator, DType.Float32, rows, hidden);
            CudaExpertDoorbell bell = _doorbell;
            int tailRank = gate.Host.TailSkipRank(layer);
            Launch(new[] { x, sel, w, tables.OnDevice }, new[] { devSel, tables.Counts }, (kn, stream) => kn.LaunchMoeRouteSplit(Ptr(x),
                Ptr(sel), Ptr(w), Ptr(tables.OnDevice), hidden, rows, k, layer, tailRank, Ptr(devSel), Ptr(tables.Counts), bell.Box,
                bell.MissDev, stream));
            // The device's routes (the host's read -1), the router's weights: a decode row, or a verify window's rows.
            bool ran = rows == 1
                ? CudaFusedOps.TryMoEExpertFFNDecodeSwiGLU(null, x2, output, devSel, w1, gateOut, upOut, IntPtr.Zero, Ptr(tables.Gate),
                    Ptr(tables.Up), Ptr(tables.Down), gate.Format, down.Format, experts, k, hidden, ff, null, IntPtr.Zero, inputQ8, gateQ8,
                    gateUpDp4a, downDp4a)
                : CudaFusedOps.TryMoEExpertFFNPrefillSwiGLU(null, x2, output, devSel, w1, gateOut, upOut, IntPtr.Zero, Ptr(tables.Gate),
                    Ptr(tables.Up), Ptr(tables.Down), gate.Format, down.Format, experts, k, hidden, ff, rows, null, IntPtr.Zero, inputQ8,
                    gateQ8, gateUpDp4a, downDp4a);
            int sum = rows * hidden;
            if (!ran)
            {
                // The split rang already: answer the host's computation so the stream does not wait on a flag later.
                Launch(Array.Empty<Tensor>(), output, (kn, stream) => kn.LaunchMoeWaitAdd(bell.Box, Ptr(output), sum, bell.MissDev, stream));
                output.Dispose();
                throw new NotSupportedException($"layer {layer}: the CUDA MoE kernels do not take these experts ({experts} experts, top-{k}, "
                    + $"types {gate.Format}/{down.Format}, {hidden}x{ff}, {rows} row(s))");
            }
            // The host's sum joins when something first reads the output: the work the program emits before that (a shared
            // expert reading the same input) runs on the device while the host computes, ahead of the wait.
            _joins.Add(((CudaStorage)output.Storage, () =>
            {
                Launch(Array.Empty<Tensor>(), output, (kn, stream) => kn.LaunchMoeWaitAdd(bell.Box, Ptr(output), sum, bell.MissDev, stream));
                if (bell.Error is { } error)
                    throw new InvalidOperationException($"layer {layer}: the host experts failed", error);
            }));
            PendingWrites = true;
            result = output;
            return true;
        }

        /// <summary>Doorbell sums not yet added (the output's storage, the wait-add that finishes it).</summary>
        private readonly List<(CudaStorage Storage, Action Finish)> _joins = new();

        protected override void BeforeRead(Tensor t)
        {
            if (t.Storage is not CudaStorage storage) return;
            for (int i = 0; i < _joins.Count; i++)
            {
                if (!ReferenceEquals(_joins[i].Storage, storage)) continue;
                Action finish = _joins[i].Finish;
                _joins.RemoveAt(i);
                PendingWrites = _joins.Count > 0;
                finish();
                return;
            }
        }

        protected override void FinishWrites() => FinishJoins();

        private void FinishJoins()
        {
            while (_joins.Count > 0)
            {
                Action finish = _joins[0].Finish;
                _joins.RemoveAt(0);
                PendingWrites = _joins.Count > 0;
                finish();
            }
            PendingWrites = false;
        }

        /// <summary>The device tables of a stack of offloaded experts: tier members (resident on this device) by their
        /// resident copies, the rest left null and marked off the device.</summary>
        private HostedTables Tables(IGraphHostedExperts gate, IGraphExpertWeights up, IGraphExpertWeights down, int layer)
        {
            lock (_hostedTables)
            {
                if (_hostedTables.TryGetValue(gate, out HostedTables t)) return t;
                var cuda = (CudaAllocator)Allocator;
                int experts = (int)gate.Ne2;
                var words = new int[3][];
                for (int m = 0; m < 3; m++) words[m] = new int[2 * experts];
                var onDevice = new byte[experts];
                IGraphExpertWeights[] sets = { gate, up, down };
                for (int e = 0; e < experts; e++)
                {
                    if (!gate.Host.IsResident(layer, e)) continue;
                    var ptrs = new long[3];
                    bool resident = true;
                    for (int m = 0; m < 3 && resident; m++)
                    {
                        IGraphWeight member;
                        try { member = sets[m].Expert(e); }
                        catch (NotSupportedException) { member = null; }
                        resident = member != null && CudaQuantizedOps.TryGetResidentDevicePtr(cuda, member.CacheKey, out IntPtr p, out int type)
                            && type == sets[m].Format && (ptrs[m] = p.ToInt64()) != 0;
                    }
                    if (!resident) continue;   // the host computes it
                    onDevice[e] = 1;
                    for (int m = 0; m < 3; m++)
                    {
                        words[m][2 * e] = unchecked((int)ptrs[m]);
                        words[m][2 * e + 1] = unchecked((int)(ptrs[m] >> 32));
                    }
                }
                Tensor Upload(int[] values)
                {
                    var table = new Tensor(Allocator, DType.Int32, values.Length);
                    table.SetElementsAsInt(values);
                    ((CudaStorage)table.Storage).EnsureDeviceCurrent();
                    return table;
                }
                var mask = new Tensor(Allocator, DType.UInt8, experts);
                var maskInts = new int[(experts + 3) / 4];
                Buffer.BlockCopy(onDevice, 0, maskInts, 0, experts);
                t = new HostedTables
                {
                    Gate = Upload(words[0]), Up = Upload(words[1]), Down = Upload(words[2]), OnDevice = mask, Counts = Upload(new int[experts]),
                    Host = gate.Host, Layer = layer, Experts = experts, Sets = sets, Words = words, Mask = onDevice,
                };
                using (Tensor bytes = Upload(maskInts))
                {
                    // The mask as bytes: the int words carry them in order.
                    cuda.Context.MakeCurrent();
                    CudaDriverApi.cuMemcpyDtoD(Ptr(mask), Ptr(bytes), new UIntPtr((ulong)experts)).ThrowOnError();
                    ((CudaStorage)mask.Storage).MarkDeviceModified();
                }
                return _hostedTables[gate] = t;
            }
        }

        // ---------------------------------------------------------------- routed experts

        /// <summary>Most rows the per-row MoE kernels take; a longer chunk groups its rows by expert (each expert's
        /// weights read once for all its rows), as the per-op prefill does.</summary>
        private const int PerRowExpertRows = 32;

        /// <summary>
        /// A routed-experts region (<see cref="GraphFusion.RoutedExperts"/>) with the per-op path's kernels: the router
        /// (top-K of the logits, softmax over the selected: the renormalized routing) and each row's experts through
        /// device tables of the members' resident copies, all on the device for decode and verify rows; a prompt chunk
        /// routes on the host and runs each selected expert once over its rows. False (the region's ops run) for the
        /// unnormalized routing, members that are not resident, or a geometry the kernels do not take.
        /// </summary>
        private bool TryRoutedExperts(Tensor[] inputs, IGraphWeight[] weights, long[] ip, Tensor[] results)
        {
            if (ip == null || ip.Length < 2 || ip[1] != 1) return false;
            if (weights[2] is not IGraphExpertWeights gate || weights[3] is not IGraphExpertWeights up
                || weights[4] is not IGraphExpertWeights down) return false;
            Tensor x = inputs[0], logits = inputs[1];
            if (x?.Storage is not CudaStorage || logits?.Storage is not CudaStorage
                || x.ElementType != DType.Float32 || logits.ElementType != DType.Float32) return false;
            int experts = (int)gate.Ne2, k = (int)ip[0], hidden = (int)gate.Ne0, ff = (int)gate.Ne1;
            if (up.Ne0 != hidden || up.Ne1 != ff || up.Ne2 != experts || down.Ne0 != ff || down.Ne1 != hidden
                || down.Ne2 != experts || gate.Format != up.Format
                || !CudaFusedOps.IsMoERouterConfigurationSupported(experts, k)) return false;
            long count = x.ElementCount() / hidden;
            if (count < 1 || count > 65535 || x.ElementCount() != count * hidden || logits.ElementCount() != count * experts)
                return false;
            int rows = (int)count;
            var cuda = (CudaAllocator)Allocator;
            RequireDevice(x, cuda.DeviceId, "input", 0);
            RequireDevice(logits, cuda.DeviceId, "input", 1);
            using Tensor x2 = x.View(rows, hidden);
            using Tensor logits2 = logits.View(rows, experts);
            var output = new Tensor(Allocator, DType.Float32, rows, hidden);
            bool ok = TryDecodeExperts(x2, logits2, output, gate, up, down, experts, k, hidden, ff, rows)
                || (rows <= PerRowExpertRows
                    ? PerRowExperts(x2, logits2, output, gate, up, down, experts, k, hidden, ff, rows)
                    : TryChunkExperts(x2, logits2, output, gate, up, down, experts, k, hidden, ff, rows)
                      || GroupedExperts(x2, logits2, output, gate, up, down, experts, k));
            if (!ok)
            {
                output.Dispose();
                return false;
            }
            results[0] = output;
            return true;
        }

        /// <summary>Whether the loaded module's grouped expert kernels read experts of <paramref name="type"/> (an older module
        /// reads some layouts as Q8_0: CudaMoe.RequireExpertDecoders).</summary>
        private static bool ChunkKernelsRead(Dsv4Kernels dk, int type) => CudaMoe.SupportsExpertType(type) && type switch
        {
            7 => dk.DecodesQ5_1Experts,
            19 or 22 => dk.DecodesLowBitIqExperts,
            42 => dk.DecodesLegacyQuants,
            _ => true,
        };

        /// <summary>
        /// A prompt chunk's device-resident routed experts on the grouped kernels the tiered layers already take
        /// (CudaMoe.Experts: the routes grouped by expert on the device, then the tensor-core or int8 grouped GEMMs). The
        /// per-expert loop below them spent 3.0 of a 2048-row step's 3.6 s on 46K launches once every expert of Qwen3.8 Flash
        /// Next's ISTA pack fitted an A40 (its prompt then took 7.4 s against 1.8). The routing is the engines' selection (the K
        /// largest logits, softmax over them; every score NaN picks expert 0, where the executor's router wrote -1,
        /// which the grouped kernels index with).
        /// </summary>
        private bool TryChunkExperts(Tensor x, Tensor logits, Tensor output, IGraphExpertWeights gate, IGraphExpertWeights up,
            IGraphExpertWeights down, int experts, int k, int hidden, int ff, int rows)
        {
            var cuda = (CudaAllocator)Allocator;
            if (hidden % 32 != 0 || ff % 32 != 0 || gate.Format != up.Format || k > 32 || experts > 512 || cuda.Kernels == null
                || CudaGraphCapture.IsCapturing(cuda))
                return false;
            Dsv4Kernels dk = MoeKernels;
            if (dk == null || !ChunkKernelsRead(dk, gate.Format) || !ChunkKernelsRead(dk, down.Format)
                || !GroupedTakes(new[] { gate, up, down }, rows, hidden, ff))
                return false;
            Tensor gateTable = ExpertTable(gate), upTable = ExpertTable(up), downTable = ExpertTable(down);
            if (gateTable == null || upTable == null || downTable == null) return false;
            if (!TryExpertRowBytes(gate, out long gateRow) || !TryExpertRowBytes(up, out long upRow) || !TryExpertRowBytes(down, out long downRow))
                return false;
            var owned = new List<Tensor>();
            try
            {
                var s = new CudaMoeScratch((type, sizes) =>
                {
                    var t = new Tensor(Allocator, type, sizes);
                    owned.Add(t);
                    return t;
                }, rows, experts, k, hidden, ff);
                foreach (Tensor t in new[] { x, logits, output }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
                cuda.Context.MakeCurrent();
                IntPtr stream = cuda.Stream.Handle;
                dk.MoeSelect(logits, IntPtr.Zero, IntPtr.Zero, null, s.Sel, s.SelW, experts, k, 1, 1f, rows, stream, Dsv4Kernels.RouterSoftmax, keys: true);
                var gw = new DeviceWeight { Type = gate.Format, Ne0 = hidden, Ne1 = ff, RowBytes = gateRow, Table = Ptr(gateTable) };
                var uw = new DeviceWeight { Type = up.Format, Ne0 = hidden, Ne1 = ff, RowBytes = upRow, Table = Ptr(upTable) };
                var dw = new DeviceWeight { Type = down.Format, Ne0 = ff, Ne1 = hidden, RowBytes = downRow, Table = Ptr(downTable) };
                CudaMoe.Experts(dk, cuda.Kernels, s, gw, uw, dw, x, null, output, rows, k, experts, hidden, ff, 0f, GroupedPerSlotMaxRows, stream);
                ((CudaStorage)output.Storage).MarkDeviceModified();
                return true;
            }
            finally
            {
                // The kernels are queued on the allocator's stream: the scratch returns to its pool for later work there.
                foreach (Tensor t in owned) t.Dispose();
            }
        }

        /// <summary>Bytes of one row of a stacked expert weight (every member alike).</summary>
        private static bool TryExpertRowBytes(IGraphExpertWeights set, out long rowBytes)
        {
            rowBytes = 0;
            IGraphWeight first;
            try { first = set.Expert(0); }
            catch (NotSupportedException) { return false; }
            if (first == null || first.Ne1 <= 0 || first.Bytes % first.Ne1 != 0) return false;
            rowBytes = first.Bytes / first.Ne1;
            return true;
        }

        /// <summary>Rows up to which the routed experts run on the direct engines' per-(token, expert) decode kernels, in
        /// pieces of <see cref="GroupedPerSlotMaxRows"/>.</summary>
        private const int DecodeExpertRows = 32;

        /// <summary>
        /// Expert formats the per-slot decode kernels read at least as close to the exact product as the executor's per-row
        /// kernels (the median relative L2 of a two-layer fixture's logits against the managed CPU, A40): Q8_0 8.0e-3 against
        /// 2.5e-2, MXFP4 8.9e-3 against 2.1e-2, IQ3_S 4.0e-2 against 4.9e-2, Q4_0 / IQ2_XXS / IQ2_S / IQ4_NL the same bits,
        /// Q5_1 / Q4_K / Q6_K / IQ2_XS / IQ3_XXS within a tenth. IQ4_XS (2.1e-2 against 8.8e-3) and IQ1_S (1.9e-2 against
        /// 8.6e-3) keep the per-row kernels.
        /// </summary>
        private static bool DecodeKernelsAsExact(int type) => type is not (19 or 23);

        /// <summary>The per-slot kernels' scratch for each expert geometry (never freed while recorded steps may bind it).</summary>
        private readonly Dictionary<(int Experts, int K, int Hidden, int Ff), CudaMoeScratch> _decodeScratch = new();
        private readonly List<Tensor> _decodeScratchTensors = new();

        /// <summary>
        /// The routed experts of decode steps and verify windows (up to <see cref="DecodeExpertRows"/> rows, several
        /// sequences' alike) on the direct engines' per-(token, expert) decode kernels (CudaMoe.Experts' per-slot route, the
        /// engines' own): each slot is computed as a lone token's decode step computes it, so a row's experts never depend
        /// on how many rows the step holds (a verify window's rows are a decode step's bit for bit, and a speculating
        /// request's are a plain one's however many share the step). The executor's batched kernels took 160 us a layer for
        /// a 4-row window of Qwen3.8 Flash Next's Q2_0 experts against these kernels' 83 (A40, nsys). The routing is the
        /// engines' selection (the K largest logits, softmax over them; never an expert -1, which these kernels would index
        /// with); Q8_0 experts are read in the dense matvec's row layout, as the engine reads them. A fixed choice by format
        /// and rows.
        /// </summary>
        private bool TryDecodeExperts(Tensor x, Tensor logits, Tensor output, IGraphExpertWeights gate, IGraphExpertWeights up,
            IGraphExpertWeights down, int experts, int k, int hidden, int ff, int rows)
        {
            var cuda = (CudaAllocator)Allocator;
            if (rows > DecodeExpertRows || hidden % 32 != 0 || ff % 32 != 0 || gate.Format != up.Format || k > 32 || experts > 512
                || cuda.Kernels == null)
                return false;
            // The module and the scratch are made on an uncaptured step (never inside a recording, which would then bind
            // another route than the unrecorded steps take: the first steps of a shape always run unrecorded).
            if (_moeKernels == null && (_moeKernelsMissing || CudaGraphCapture.IsCapturing(cuda))) return false;
            Dsv4Kernels dk = MoeKernels;
            if (dk == null || !ChunkKernelsRead(dk, gate.Format) || !ChunkKernelsRead(dk, down.Format)
                || !DecodeKernelsAsExact(gate.Format) || !DecodeKernelsAsExact(down.Format)) return false;
            Tensor gateTable = ExpertTable(gate), upTable = ExpertTable(up), downTable = ExpertTable(down);
            if (gateTable == null || upTable == null || downTable == null) return false;
            if (!TryExpertRowBytes(gate, out long gateRow) || !TryExpertRowBytes(up, out long upRow) || !TryExpertRowBytes(down, out long downRow))
                return false;
            var shape = (experts, k, hidden, ff);
            if (!_decodeScratch.TryGetValue(shape, out CudaMoeScratch s))
            {
                if (CudaGraphCapture.IsCapturing(cuda)) return false;
                s = new CudaMoeScratch((type, sizes) =>
                {
                    var t = new Tensor(Allocator, type, sizes);
                    ((CudaStorage)t.Storage).EnsureDeviceCurrent();
                    _decodeScratchTensors.Add(t);
                    return t;
                }, GroupedPerSlotMaxRows, experts, k, hidden, ff);
                _decodeScratch[shape] = s;
            }
            foreach (Tensor t in new[] { x, logits, output }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            cuda.Context.MakeCurrent();
            IntPtr stream = cuda.Stream.Handle;
            var gw = new DeviceWeight { Type = gate.Format, Ne0 = hidden, Ne1 = ff, RowBytes = gateRow, Table = Ptr(gateTable) };
            var uw = new DeviceWeight { Type = up.Format, Ne0 = hidden, Ne1 = ff, RowBytes = upRow, Table = Ptr(upTable) };
            var dw = new DeviceWeight { Type = down.Format, Ne0 = ff, Ne1 = hidden, RowBytes = downRow, Table = Ptr(downTable) };
            for (int first = 0; first < rows; first += GroupedPerSlotMaxRows)
            {
                int n = Math.Min(GroupedPerSlotMaxRows, rows - first);
                using Tensor xs = x.Narrow(0, first, n), ls = logits.Narrow(0, first, n), outs = output.Narrow(0, first, n);
                dk.MoeSelect(ls, IntPtr.Zero, IntPtr.Zero, null, s.Sel, s.SelW, experts, k, 1, 1f, n, stream, Dsv4Kernels.RouterSoftmax, keys: true);
                CudaMoe.Experts(dk, cuda.Kernels, s, gw, uw, dw, xs, null, outs, n, k, experts, hidden, ff, 0f, GroupedPerSlotMaxRows, stream,
                    q80VecDecode: true);
            }
            ((CudaStorage)output.Storage).MarkDeviceModified();
            return true;
        }

        /// <summary>
        /// The routed experts on the executor's fused kernels. Up to <see cref="DecodeExpertRows"/> rows each run as a lone
        /// decode row (the one-row kernel a row): the multi-row kernel rounds apart from it, so a verify window's rows had
        /// differed from the decode rows at their positions where the per-slot kernels do not take the format.
        /// </summary>
        private bool PerRowExperts(Tensor x, Tensor logits, Tensor output, IGraphExpertWeights gate, IGraphExpertWeights up,
            IGraphExpertWeights down, int experts, int k, int hidden, int ff, int rows)
        {
            if (rows > 1 && rows <= DecodeExpertRows)
            {
                for (int r = 0; r < rows; r++)
                {
                    using Tensor xr = x.Narrow(0, r, 1), lr = logits.Narrow(0, r, 1), outr = output.Narrow(0, r, 1);
                    if (!PerRowExperts(xr, lr, outr, gate, up, down, experts, k, hidden, ff, 1)) return false;
                }
                return true;
            }
            Tensor gateTable = ExpertTable(gate), upTable = ExpertTable(up), downTable = ExpertTable(down);
            if (gateTable == null || upTable == null || downTable == null) return false;
            bool gateUpDp4a = CudaQuantizedOps.IsMoeDp4aEnabledForType(gate.Format) && CudaFusedOps.IsMoeDp4aDimensionSupported(gate.Format, hidden);
            bool downDp4a = CudaQuantizedOps.IsMoeDp4aEnabledForType(down.Format) && CudaFusedOps.IsMoeDp4aDimensionSupported(down.Format, ff);
            long routed = (long)rows * k;
            using var selected = new Tensor(Allocator, DType.Int32, routed);
            using var routing = new Tensor(Allocator, DType.Float32, routed);
            using var gateOut = new Tensor(Allocator, DType.Float32, routed, ff);
            using var upOut = new Tensor(Allocator, DType.Float32, routed, ff);
            using Tensor inputQ8 = gateUpDp4a ? new Tensor(Allocator, DType.UInt8, (long)rows * (hidden / 32) * CudaFusedOps.Q81BlockBytes) : null;
            using Tensor gateQ8 = downDp4a ? new Tensor(Allocator, DType.UInt8, routed * (ff / 32) * CudaFusedOps.Q81BlockBytes) : null;
            return rows == 1
                ? CudaFusedOps.TryMoEExpertFFNDecodeSwiGLU(logits, x, output, selected, routing, gateOut, upOut, IntPtr.Zero,
                    Ptr(gateTable), Ptr(upTable), Ptr(downTable), gate.Format, down.Format, experts, k, hidden, ff,
                    null, IntPtr.Zero, inputQ8, gateQ8, gateUpDp4a, downDp4a)
                : CudaFusedOps.TryMoEExpertFFNPrefillSwiGLU(logits, x, output, selected, routing, gateOut, upOut, IntPtr.Zero,
                    Ptr(gateTable), Ptr(upTable), Ptr(downTable), gate.Format, down.Format, experts, k, hidden, ff, rows,
                    null, IntPtr.Zero, inputQ8, gateQ8, gateUpDp4a, downDp4a);
        }

        /// <summary>A prompt chunk's experts: the routing on the host (the device router's selection and weights), then
        /// each selected expert once over its rows (gathered), scattered back weighted.</summary>
        private bool GroupedExperts(Tensor x, Tensor logits, Tensor output, IGraphExpertWeights gate, IGraphExpertWeights up,
            IGraphExpertWeights down, int experts, int k)
        {
            int rows = (int)x.Sizes[0];
            float[] scores = logits.GetElementsAsFloat(rows * experts);
            var byExpert = new List<(int Row, float Weight)>[experts];
            var top = new float[k];
            var index = new int[k];
            for (int r = 0; r < rows; r++)
            {
                RouteRow(scores, r * experts, experts, k, top, index);
                for (int j = 0; j < k; j++)
                    if (index[j] >= 0) (byExpert[index[j]] ??= new()).Add((r, top[j]));
            }
            Ops.Fill(output, 0f);
            for (int e = 0; e < experts; e++)
            {
                var batch = byExpert[e];
                if (batch == null) continue;
                var rowIds = new int[batch.Count];
                var rowWeights = new float[batch.Count];
                for (int i = 0; i < rowIds.Length; i++) (rowIds[i], rowWeights[i]) = batch[i];
                using var ids = new Tensor(Allocator, DType.Int32, rowIds.Length);
                ids.SetElementsAsInt(rowIds);
                using var w = new Tensor(Allocator, DType.Float32, rowWeights.Length);
                w.SetElementsAsFloat(rowWeights);
                using Tensor xs = Ops.IndexSelect(null, x, ids);
                using Tensor g = ExpertMatMul(gate, e, xs);
                using Tensor u = ExpertMatMul(up, e, xs);
                Ops.SiLUMul(g, g, u);
                using Tensor d = ExpertMatMul(down, e, g);
                if (!CudaFusedOps.TryMoEScatterAddWeightedRows(output, d, ids, w)) return false;
            }
            return true;
        }

        /// <summary>ts_moe_router_f32 on the host: the K largest logits (a later equal logit does not displace an
        /// earlier one), softmax over them.</summary>
        private static void RouteRow(float[] logits, int offset, int experts, int k, float[] top, int[] index)
        {
            Array.Fill(top, float.MinValue);
            Array.Fill(index, -1);
            for (int e = 0; e < experts; e++)
            {
                float v = logits[offset + e];
                int min = 0;
                for (int j = 1; j < k; j++) if (top[j] < top[min]) min = j;
                if (v > top[min]) { top[min] = v; index[min] = e; }
            }
            float max = float.MinValue;
            for (int j = 0; j < k; j++) max = MathF.Max(max, top[j]);
            float sum = 0f;
            for (int j = 0; j < k; j++) { top[j] = MathF.Exp(top[j] - max); sum += top[j]; }
            float inv = sum > 0f ? 1f / sum : 0f;
            for (int j = 0; j < k; j++) top[j] *= inv;
        }

        private readonly Dictionary<IGraphExpertWeights, Tensor> _expertTables = new(ReferenceEqualityComparer.Instance);

        /// <summary>A stacked expert weight's table of its members' resident device copies ([experts] pointers, as
        /// int pairs), or null when a member is not resident on this device (then reported once).</summary>
        private Tensor ExpertTable(IGraphExpertWeights set)
        {
            lock (_expertTables)
            {
                if (_expertTables.TryGetValue(set, out Tensor table)) return table;
                var cuda = (CudaAllocator)Allocator;
                var words = new int[2 * set.Ne2];
                for (int e = 0; e < set.Ne2; e++)
                {
                    IGraphWeight member;
                    try { member = set.Expert(e); }
                    catch (NotSupportedException) { member = null; }
                    if (member == null || !CudaQuantizedOps.TryGetResidentDevicePtr(cuda, member.CacheKey, out IntPtr ptr, out int type)
                        || type != set.Format)
                    {
                        Console.Error.WriteLine($"WARNING: CUDA routed experts: expert {e} of a stacked weight is not resident on " +
                            $"device {cuda.DeviceId}; its layer's experts run as separate products.");
                        return _expertTables[set] = null;
                    }
                    long v = ptr.ToInt64();
                    words[2 * e] = unchecked((int)v);
                    words[2 * e + 1] = unchecked((int)(v >> 32));
                }
                table = new Tensor(Allocator, DType.Int32, words.Length);
                table.SetElementsAsInt(words);
                ((CudaStorage)table.Storage).EnsureDeviceCurrent();
                return _expertTables[set] = table;
            }
        }

        /// <summary>A streamed expert's device copy goes once the step used it: only the layer's tier stays resident.</summary>
        protected override void ReleaseStreamedExpert(IGraphWeight expert)
        {
            if (Allocator is CudaAllocator cuda && expert.CacheKey != IntPtr.Zero)
                CudaQuantizedOps.ReleaseQuantizedWeight(cuda, expert.CacheKey);
        }

        /// <summary>A step's activations quantized to q8_1 for the decode-size products (rows up to 16), by the storage and
        /// offset they were read from: the products of one activation (q, k and v; a block's gate and up) quantize it once.
        /// Cleared at every run (a storage object is never written again while it is a node's value).</summary>
        private readonly Dictionary<(CudaStorage Storage, long Offset, long InDim, long Rows), Tensor> _quantized = new();

        protected override void BeginRun()
        {
            foreach (Tensor t in _quantized.Values) t.Dispose();
            _quantized.Clear();
            _runOrdinal++;
            HarvestPromptCounts();
        }

        /// <summary>The decode-once kernels' module where they take a decode-size product of this weight (up to 8 rows of a
        /// 2-D weight in one of their formats), else null.</summary>
        private Dsv4Kernels DecodeOnce(int type, long inDim, long rows, IGraphWeight weight)
        {
            if (rows > 8 || weight.Ne2 > 1 || weight.Bytes % weight.Ne1 != 0) return null;
            // The module loads on an uncaptured step (never inside a recording).
            if (_moeKernels == null && (_moeKernelsMissing || CudaGraphCapture.IsCapturing((CudaAllocator)Allocator))) return null;
            Dsv4Kernels dk = MoeKernels;
            return dk != null && dk.DenseFastReads(type, inDim) ? dk : null;
        }

        /// <summary>BF16 products of up to this many rows take the row-invariant kernels (decode steps and verify windows,
        /// several sequences' alike).</summary>
        private const int Bf16DecodeRows = 32;
        /// <summary>Rows the decode-once kernels take in one call, and quantized products of up to <see cref="DenseDecodeRows"/>
        /// rows run on them in pieces of that many.</summary>
        private const int DecodeOnceRows = 8, DenseDecodeRows = 32;

        protected override Tensor QuantizedMatMul(IGraphWeight weight, Tensor x)
        {
            var cuda = (CudaAllocator)Allocator;
            long rows = x.Sizes[0], inDim = weight.Ne0, outDim = weight.Ne1;
            // A decode-size product chooses its kernel by the weight's resident copy: made resident first, so a weight's first
            // product takes the kernel every later one takes (it had fallen to the uploading product once, which rounds
            // otherwise: a prompt's first logits then differed from the same prompt's in a later batch).
            if (rows <= 32 && !CudaGraphCapture.IsCapturing(cuda) && weight.CacheKey != IntPtr.Zero)
                CudaQuantizedOps.TryEnsureResident(cuda, weight.CacheKey, weight.Data, weight.Format, inDim, outDim, weight.Bytes, out _, out _);
            // BF16 weights (MoE routers, recurrent gates, sparse-attention indexer projections) at decode size, F32 activations,
            // with sums whose order never depends on the row count: a verify window's rows are then a plain decode step's bit
            // for bit, however many sequences share the step. A wide output (a router's 256+ experts) runs on the direct
            // engines' block GEMV (one block an output, every row of the call through it: 4.4 us a 4-row window's router on
            // an RTX 5090), a narrower one on the one-row matvec a row. The multi-row product had gone to cuBLAS with the
            // activations rounded to BF16, so a window's router could pick other experts than the same row alone, and took
            // 113 us a product (Qwen3.8 Flash Next, 4-row windows on an A40); a row each then cost 55K launches for 192
            // tokens. A fixed choice by format, width and rows.
            if (rows >= 1 && rows <= Bf16DecodeRows && inDim % 8 == 0 && inDim <= int.MaxValue && outDim <= int.MaxValue
                && x.Storage is CudaStorage bs && x.ElementType == DType.Float32 && x.IsContiguous() && x.ElementCount() == rows * inDim
                && cuda.Kernels != null && CudaQuantizedOps.TryGetResidentDevicePtr(cuda, weight.CacheKey, out IntPtr bPtr, out int bType)
                && bType == 30 && weight.Bytes == inDim * outDim * 2)
            {
                var product = new Tensor(Allocator, DType.Float32, rows, outDim);
                ((CudaStorage)product.Storage).EnsureDeviceCurrent();
                bs.EnsureDeviceCurrent();
                cuda.Context.MakeCurrent();
                IntPtr input = Ptr(x), output = Ptr(product);
                Dsv4Kernels wide = outDim >= 256 && !(_moeKernels == null && (_moeKernelsMissing || CudaGraphCapture.IsCapturing(cuda)))
                    ? MoeKernels : null;
                if (wide != null && wide.HasGemvBf16)
                    for (long first = 0; first < rows; first += Dsv4Kernels.GemvMaxRows)
                        wide.GemvBf16(input + (nint)(first * inDim * sizeof(float)), bPtr, output + (nint)(first * outDim * sizeof(float)),
                            (int)inDim, (int)outDim, (int)Math.Min(Dsv4Kernels.GemvMaxRows, rows - first), cuda.Stream.Handle);
                else
                    for (long r = 0; r < rows; r++)
                        cuda.Kernels.LaunchMatvecBf16(bPtr, input + (nint)(r * inDim * sizeof(float)), output + (nint)(r * outDim * sizeof(float)),
                            (int)inDim, (int)outDim, cuda.Stream.Handle);
                ((CudaStorage)product.Storage).MarkDeviceModified();
                return product;
            }
            // A decode-size product over q8_1 activations the step already quantized (or quantizes now, once).
            if (rows >= 1 && rows <= DenseDecodeRows && x.Storage is CudaStorage xs && x.ElementType == DType.Float32 && x.IsContiguous()
                && x.ElementCount() == rows * inDim && inDim % 32 == 0 && inDim <= int.MaxValue && outDim <= int.MaxValue
                && cuda.Kernels != null && CudaQuantizedOps.TryGetResidentDevicePtr(cuda, weight.CacheKey, out IntPtr wPtr, out int type)
                && (DecodeOnce(type, inDim, Math.Min(rows, DecodeOnceRows), weight) is not null
                    || (rows <= 16 && CudaQuantizedOps.ResidentMatmulTakesQ81(type, (int)inDim, (int)outDim, (int)rows))))
            {
                var key = (xs, x.StorageOffset, inDim, rows);
                if (!_quantized.TryGetValue(key, out Tensor xq))
                {
                    xq = new Tensor(Allocator, DType.UInt8, rows * (inDim / 32) * CudaKernels.Q81BlockBytes);
                    xs.EnsureDeviceCurrent();
                    cuda.Context.MakeCurrent();
                    CudaQuantizedOps.QuantizeRowsQ81(cuda, Ptr(x), Ptr(xq), (int)inDim, (int)rows);
                    ((CudaStorage)xq.Storage).MarkDeviceModified();
                    _quantized[key] = xq;
                }
                var product = new Tensor(Allocator, DType.Float32, rows, outDim);
                ((CudaStorage)product.Storage).EnsureDeviceCurrent();
                cuda.Context.MakeCurrent();
                // Up to 8 rows of a format the direct engines' decode-once kernels read run on them: a row's sums never depend
                // on how many rows the call holds, so a verify window's rows are a plain decode step's bit for bit (a
                // speculative step answers as the plain greedy stream), and a K-quant window no longer goes through a
                // dequantizing product (3-5x slower on an A40, an RTX 2000 Ada and an RTX PRO 6000 slice). A fixed choice
                // by format and rows, never by timing (the per-type q8_1 kernels round apart in the last bits).
                // Past 8 rows (several sequences' windows in one step) in pieces of 8: still a decode row's sums, and faster than
                // the one multi-row product (4 sequences' 4-row windows of Qwen3.8 Flash Next: 50-58 tok/s served against 40).
                if (DecodeOnce(type, inDim, Math.Min(rows, DecodeOnceRows), weight) is { } dk)
                {
                    long qRow = inDim / 32 * CudaKernels.Q81BlockBytes;
                    for (long first = 0; first < rows; first += DecodeOnceRows)
                        dk.DenseQ81Rows(wPtr, Ptr(xq) + (nint)(first * qRow), Ptr(product) + (nint)(first * outDim * sizeof(float)), type,
                            (int)outDim, (int)inDim, weight.Bytes / outDim, (int)Math.Min(DecodeOnceRows, rows - first), cuda.Stream.Handle);
                    ((CudaStorage)product.Storage).MarkDeviceModified();
                    return product;
                }
                if (rows <= 16 && CudaQuantizedOps.ResidentMatmulTakesQ81(type, (int)inDim, (int)outDim, (int)rows)
                    && CudaQuantizedOps.TryResidentMatmulQ81(cuda, wPtr, type, Ptr(xq), Ptr(product), (int)inDim, (int)outDim, (int)rows))
                {
                    ((CudaStorage)product.Storage).MarkDeviceModified();
                    return product;
                }
                product.Dispose();
            }
            var result = new Tensor(Allocator, DType.Float32, x.Sizes[0], weight.Ne1);
            if (CudaQuantizedOps.TryAddmmQuantizedToFloat32(result, x, weight.CacheKey, weight.Data, weight.Format,
                    weight.Ne0, weight.Ne1, weight.Bytes))
                return result;
            result.Dispose();
            return HostProduct(weight, x);
        }
    }
}
