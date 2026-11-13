// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models.Architecture;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Models
{

    // Writing K/V into a linear cache and reading it back: the F32, F16 and
    // block-quantized layouts, the MLX head-first fast paths, and GQA head expansion.
    public abstract partial class ModelBase
    {
        protected void CopyToCache(Tensor cache, Tensor src, int startPos, int seqLen)
        {
            if (TryCopyHeadFirstToCacheMlx(cache, src, startPos, seqLen))
                return;

            if (CudaFusedOps.TryCopyHeadFirstToCache(cache, src, startPos, seqLen, (int)cache.Sizes[1], false))
                return;

            if (cache.ElementType == DType.Float16)
            {
                CopyToCacheF16(cache, src, startPos, seqLen);
                return;
            }
            if (IsBlockQuantCacheDType(cache.ElementType))
            {
                CopyToCacheBlockQuant(cache, src, startPos, seqLen);
                return;
            }

            using var cacheSlice = cache.Narrow(1, startPos, seqLen);
            Ops.Copy(cacheSlice, src);
            InvalidateTensorDeviceCache(cache);
        }

        /// <summary>
        /// Append <paramref name="seqLen"/> rows of an F32 (numKVHeads, seqLen, headDim)
        /// tensor to a block-quantized (Q4_0 / Q8_0) linear cache starting at
        /// <paramref name="startPos"/>, quantizing each appended position into the cache's
        /// block layout. The bytes written are dequantized identically by ggml's native
        /// kernels on the subsequent fused-decode read. Block-quant analogue of
        /// CopyToCacheF16.
        /// </summary>
        private unsafe void CopyToCacheBlockQuant(Tensor cache, Tensor src, int startPos, int seqLen)
        {
            int numKVHeads = (int)cache.Sizes[0];
            int maxSeqLen = (int)cache.Sizes[1];
            int headDim = (int)cache.Sizes[2];
            int ggmlType = GgmlTypeForCacheDType(cache.ElementType);
            long rowBytes = ManagedQuantizedOps.RowSize(ggmlType, headDim);

            cache.Storage.EnsureHostReadable();
            byte* dstBase = (byte*)TensorComputePrimitives.GetStoragePointer(cache);
            float* srcBase = GetFloatPtr(src);

            // Linear (global) cache: positions [startPos, startPos+seqLen) are appended
            // contiguously, so each head's seqLen rows form one contiguous block-aligned
            // run that quantizes in a single pass.
            for (int h = 0; h < numKVHeads; h++)
            {
                byte* dstHead = dstBase + (long)h * maxSeqLen * rowBytes + (long)startPos * rowBytes;
                float* srcHead = srcBase + (long)h * seqLen * headDim;
                ManagedQuantizedOps.QuantizeRowFromFloat32(ggmlType, srcHead, (IntPtr)dstHead,
                    (long)seqLen * headDim);
            }

            InvalidateTensorDeviceCache(cache);
        }

        /// <summary>
        /// Append <paramref name="seqLen"/> rows of an F32 (numKVHeads, seqLen, headDim) tensor
        /// to a Float16 cache of layout (numKVHeads, maxSeqLen, headDim) starting at
        /// <paramref name="startPos"/>. Performs a per-element F32-&gt;F16 conversion.
        /// </summary>
        private unsafe void CopyToCacheF16(Tensor cache, Tensor src, int startPos, int seqLen)
        {
            int numKVHeads = (int)cache.Sizes[0];
            int maxSeqLen = (int)cache.Sizes[1];
            int headDim = (int)cache.Sizes[2];

            ushort* dstBase = TensorComputePrimitives.GetHalfPointer(cache);
            float* srcBase = GetFloatPtr(src);

            // Source layout (head-first contiguous after ReshapeToHeads): (numKVHeads, seqLen, headDim).
            for (int h = 0; h < numKVHeads; h++)
            {
                ushort* dstHead = dstBase + (long)h * maxSeqLen * headDim + (long)startPos * headDim;
                float* srcHead = srcBase + (long)h * seqLen * headDim;
                TensorComputePrimitives.F32ToF16(dstHead, srcHead, seqLen * headDim);
            }

            InvalidateTensorDeviceCache(cache);
        }

        /// <summary>
        /// Return a contiguous F32 view of the active (0..totalSeqLen) region of the K
        /// or V cache, broadcasting along the head axis when GQA group_size &gt; 1.
        /// For Float16 caches the active region is dequantized into a freshly-allocated
        /// F32 tensor before broadcasting; for Float32 caches the existing fast path is used.
        /// </summary>
        protected unsafe Tensor ExpandKVHeads(Tensor cache, int groupSize, int totalSeqLen)
        {
            if (cache.ElementType == DType.Float16 || cache.ElementType == DType.Float32)
            {
                int numKVHeads = (int)cache.Sizes[0];
                int headDim = (int)cache.Sizes[2];
                // Beside the cache: under tensor parallelism the cache lives on its
                // rank's device, and the expansion must not cross devices.
                var expanded = new Tensor(
                    cache.Storage.Allocator, DType.Float32,
                    numKVHeads * groupSize, totalSeqLen, headDim);
                if (CudaFusedOps.TryExpandKvHeads(expanded, cache, groupSize, totalSeqLen))
                    return expanded;
                expanded.Dispose();
            }

            if (cache.ElementType == DType.Float16)
                return ExpandKVHeadsF16(cache, groupSize, totalSeqLen);
            if (IsBlockQuantCacheDType(cache.ElementType))
                return ExpandKVHeadsBlockQuant(cache, groupSize, totalSeqLen);

            using var active = cache.Narrow(1, 0, totalSeqLen);
            if (groupSize == 1)
                return Ops.NewContiguous(active);
            return Ops.RepeatInterleave(null, active, groupSize, 0);
        }

        // Block-quantized (Q4_0 / Q8_0) caches cannot be walked as a flat float
        // buffer, so the per-op prefill attention reads them by dequantizing the
        // active [0, totalSeqLen) window into a fresh F32 tensor (GQA-broadcast
        // along the head axis when group_size > 1) — the block-quant analogue of
        // ExpandKVHeadsF16. mul_mat then accumulates in F32, identical to having
        // dequantized in the native kernel.
        protected static bool IsBlockQuantCacheDType(DType dt) =>
            dt == DType.Q4_0 || dt == DType.Q8_0;

        // ggml type id for a block-quantized KV-cache dtype (must match ggml.h /
        // KvCacheDtypeExtensions.GgmlType): Q4_0 -> 2, Q8_0 -> 8.
        protected static int GgmlTypeForCacheDType(DType dt) => dt switch
        {
            DType.Q4_0 => 2,
            DType.Q8_0 => 8,
            _ => throw new NotSupportedException($"Not a block-quantized KV-cache dtype: {dt}"),
        };

        protected unsafe Tensor ExpandKVHeadsBlockQuant(Tensor cache, int groupSize, int totalSeqLen)
        {
            int numKVHeads = (int)cache.Sizes[0];
            int maxSeqLen = (int)cache.Sizes[1];
            int headDim = (int)cache.Sizes[2];
            int outHeads = numKVHeads * groupSize;
            int ggmlType = GgmlTypeForCacheDType(cache.ElementType);
            long rowBytes = ManagedQuantizedOps.RowSize(ggmlType, headDim);

            cache.Storage.EnsureHostReadable();
            var f32 = new Tensor(cache.Storage.Allocator, DType.Float32, outHeads, totalSeqLen, headDim);
            float* dstBase = GetFloatPtr(f32);
            byte* srcBase = (byte*)TensorComputePrimitives.GetStoragePointer(cache);

            for (int h = 0; h < numKVHeads; h++)
            {
                // Active window is the first totalSeqLen slots of this head, which are
                // contiguous (each slot = headDim elements = a whole number of blocks).
                byte* srcHead = srcBase + (long)h * maxSeqLen * rowBytes;
                for (int g = 0; g < groupSize; g++)
                {
                    float* dstHead = dstBase + (long)(h * groupSize + g) * totalSeqLen * headDim;
                    ManagedQuantizedOps.DequantizeRowToFloat32(ggmlType, (IntPtr)srcHead,
                        dstHead, (long)totalSeqLen * headDim);
                }
            }

            InvalidateTensorDeviceCache(f32);
            return f32;
        }

        protected unsafe Tensor ExpandKVHeadsF16(Tensor cache, int groupSize, int totalSeqLen)
        {
            int numKVHeads = (int)cache.Sizes[0];
            int maxSeqLen = (int)cache.Sizes[1];
            int headDim = (int)cache.Sizes[2];
            int outHeads = numKVHeads * groupSize;

            var f32 = new Tensor(cache.Storage.Allocator, DType.Float32, outHeads, totalSeqLen, headDim);
            float* dstBase = GetFloatPtr(f32);
            ushort* srcBase = TensorComputePrimitives.GetHalfPointer(cache);

            for (int h = 0; h < numKVHeads; h++)
            {
                ushort* srcHead = srcBase + (long)h * maxSeqLen * headDim;
                for (int g = 0; g < groupSize; g++)
                {
                    float* dstHead = dstBase + (long)(h * groupSize + g) * totalSeqLen * headDim;
                    TensorComputePrimitives.F16ToF32(dstHead, srcHead, totalSeqLen * headDim);
                }
            }

            InvalidateTensorDeviceCache(f32);
            return f32;
        }

        protected unsafe void CopyToCacheDecode(Tensor kCache, Tensor kTensor,
            Tensor vCache, Tensor vTensor, int numKVHeads, int headDim, int startPos)
        {
            using (var kHeads = kTensor.View(numKVHeads, 1, headDim))
            using (var vHeads = vTensor.View(numKVHeads, 1, headDim))
            {
                if (TryCopyHeadFirstToCacheMlx(kCache, kHeads, startPos, 1) &&
                    TryCopyHeadFirstToCacheMlx(vCache, vHeads, startPos, 1))
                {
                    return;
                }

                int cacheSize = (int)kCache.Sizes[1];
                if (CudaFusedOps.TryCopyHeadFirstToCache(kCache, kHeads, startPos, 1, cacheSize, false) &&
                    CudaFusedOps.TryCopyHeadFirstToCache(vCache, vHeads, startPos, 1, cacheSize, false))
                {
                    return;
                }
            }

            if (kCache.ElementType == DType.Float16 && vCache.ElementType == DType.Float16)
            {
                CopyToCacheDecodeF16(kCache, kTensor, vCache, vTensor, numKVHeads, headDim, startPos);
                return;
            }

            if (IsBlockQuantCacheDType(kCache.ElementType) && vCache.ElementType == kCache.ElementType)
            {
                CopyToCacheDecodeBlockQuant(kCache, kTensor, vCache, vTensor, numKVHeads, headDim, startPos);
                return;
            }

            float* kSrc = GetFloatPtr(kTensor);
            float* vSrc = GetFloatPtr(vTensor);
            float* kCachePtr = GetFloatPtr(kCache);
            float* vCachePtr = GetFloatPtr(vCache);
            int maxSeqLen = (int)kCache.Sizes[1];
            int headBytes = headDim * sizeof(float);

            for (int h = 0; h < numKVHeads; h++)
            {
                int cacheOffset = h * maxSeqLen * headDim + startPos * headDim;
                int srcOffset = h * headDim;
                Buffer.MemoryCopy(kSrc + srcOffset, kCachePtr + cacheOffset, headBytes, headBytes);
                Buffer.MemoryCopy(vSrc + srcOffset, vCachePtr + cacheOffset, headBytes, headBytes);
            }

            InvalidateTensorDeviceCache(kCache);
            InvalidateTensorDeviceCache(vCache);
        }

        protected bool TryCopyHeadFirstToCacheMlx(Tensor cache, Tensor src, int startPos, int seqLen, bool circular = false)
        {
            if (circular)
                return TryCopyHeadFirstToCacheCircularMlx(cache, src, startPos, seqLen);

            if (_backend != BackendType.Mlx
                || cache == null
                || src == null
                || cache.Storage is not MlxStorage
                || src.Storage is not MlxStorage
                || cache.DimensionCount != 3
                || src.DimensionCount != 3
                || cache.Sizes[0] != src.Sizes[0]
                || src.Sizes[1] != seqLen
                || cache.Sizes[2] != src.Sizes[2]
                || startPos < 0
                || startPos + seqLen > cache.Sizes[1])
            {
                return false;
            }

            // Single multi-dim slice_update beats the per-head loop below by
            // ~8× MLX dispatches per cache write — for decode (kvHeads=2, K+V
            // per layer × 42 layers) that's ~600 MLX op dispatches/token
            // collapsed into ~80. Falls back to the per-head loop if the
            // fused path declines (e.g. dtype mismatch, sub-view storage).
            if (MlxFusedOps.TryWriteKvCacheBlock(cache, src, startPos, seqLen))
                return true;

            int heads = (int)cache.Sizes[0];
            for (int h = 0; h < heads; h++)
            {
                using Tensor cacheHead = cache.Select(0, h);
                using Tensor cacheSlice = cacheHead.Narrow(0, startPos, seqLen);
                using Tensor srcHead = src.Select(0, h);
                Ops.Copy(cacheSlice, srcHead);
            }

            return true;
        }

        private bool TryCopyHeadFirstToCacheCircularMlx(Tensor cache, Tensor src, int startPos, int seqLen)
        {
            if (_backend != BackendType.Mlx
                || cache == null
                || src == null
                || cache.Storage is not MlxStorage
                || src.Storage is not MlxStorage
                || cache.DimensionCount != 3
                || src.DimensionCount != 3
                || cache.Sizes[0] != src.Sizes[0]
                || src.Sizes[1] != seqLen
                || cache.Sizes[2] != src.Sizes[2]
                || startPos < 0
                || seqLen <= 0
                || cache.Sizes[1] <= 0)
            {
                return false;
            }

            int cacheSize = checked((int)cache.Sizes[1]);
            int srcOffset = 0;
            int remaining = seqLen;
            int logicalStart = startPos;
            if (remaining > cacheSize)
            {
                srcOffset = remaining - cacheSize;
                logicalStart += srcOffset;
                remaining = cacheSize;
            }

            while (remaining > 0)
            {
                int dstOffset = logicalStart % cacheSize;
                int chunk = Math.Min(remaining, cacheSize - dstOffset);
                if (!TryCopyHeadFirstRangeToCacheMlx(cache, src, srcOffset, dstOffset, chunk))
                    return false;

                srcOffset += chunk;
                logicalStart += chunk;
                remaining -= chunk;
            }

            return true;
        }

        private bool TryCopyHeadFirstRangeToCacheMlx(Tensor cache, Tensor src, int srcOffset, int dstOffset, int length)
        {
            // Fast path: single multi-dim slice_update for the full
            // [heads, length, headDim] block when src happens to start at
            // offset 0. For the wrap-around case we fall back to the per-head
            // loop with a manually narrowed src.
            if (srcOffset == 0 && length == src.Sizes[1])
            {
                if (MlxFusedOps.TryWriteKvCacheBlock(cache, src, dstOffset, length))
                    return true;
            }

            int heads = checked((int)cache.Sizes[0]);
            for (int h = 0; h < heads; h++)
            {
                using Tensor cacheHead = cache.Select(0, h);
                using Tensor cacheSlice = cacheHead.Narrow(0, dstOffset, length);
                using Tensor srcHead = src.Select(0, h);
                using Tensor srcSlice = srcHead.Narrow(0, srcOffset, length);
                Ops.Copy(cacheSlice, srcSlice);
            }

            return true;
        }

        /// <summary>
        /// Single-position decode append into a block-quantized (Q4_0 / Q8_0) linear
        /// cache: each head's new K/V row (headDim elements, a whole number of
        /// 32-element blocks) is quantized in place at its row offset. Decode analogue
        /// of <see cref="CopyToCacheBlockQuant"/>; bytes match ggml's block layout so
        /// the fused native kernels dequantize them identically on later reads.
        /// </summary>
        private unsafe void CopyToCacheDecodeBlockQuant(Tensor kCache, Tensor kTensor,
            Tensor vCache, Tensor vTensor, int numKVHeads, int headDim, int startPos)
        {
            int ggmlType = GgmlTypeForCacheDType(kCache.ElementType);
            long rowBytes = ManagedQuantizedOps.RowSize(ggmlType, headDim);
            int maxSeqLen = (int)kCache.Sizes[1];

            kCache.Storage.EnsureHostReadable();
            vCache.Storage.EnsureHostReadable();
            float* kSrc = GetFloatPtr(kTensor);
            float* vSrc = GetFloatPtr(vTensor);
            byte* kBase = (byte*)TensorComputePrimitives.GetStoragePointer(kCache);
            byte* vBase = (byte*)TensorComputePrimitives.GetStoragePointer(vCache);

            for (int h = 0; h < numKVHeads; h++)
            {
                long cacheOffset = (long)h * maxSeqLen * rowBytes + (long)startPos * rowBytes;
                int srcOffset = h * headDim;
                ManagedQuantizedOps.QuantizeRowFromFloat32(ggmlType, kSrc + srcOffset,
                    (IntPtr)(kBase + cacheOffset), headDim);
                ManagedQuantizedOps.QuantizeRowFromFloat32(ggmlType, vSrc + srcOffset,
                    (IntPtr)(vBase + cacheOffset), headDim);
            }

            InvalidateTensorDeviceCache(kCache);
            InvalidateTensorDeviceCache(vCache);
        }

        private unsafe void CopyToCacheDecodeF16(Tensor kCache, Tensor kTensor,
            Tensor vCache, Tensor vTensor, int numKVHeads, int headDim, int startPos)
        {
            float* kSrc = GetFloatPtr(kTensor);
            float* vSrc = GetFloatPtr(vTensor);
            ushort* kDst = TensorComputePrimitives.GetHalfPointer(kCache);
            ushort* vDst = TensorComputePrimitives.GetHalfPointer(vCache);
            int maxSeqLen = (int)kCache.Sizes[1];

            for (int h = 0; h < numKVHeads; h++)
            {
                long cacheOffset = (long)h * maxSeqLen * headDim + (long)startPos * headDim;
                int srcOffset = h * headDim;
                TensorComputePrimitives.F32ToF16(kDst + cacheOffset, kSrc + srcOffset, headDim);
                TensorComputePrimitives.F32ToF16(vDst + cacheOffset, vSrc + srcOffset, headDim);
            }

            InvalidateTensorDeviceCache(kCache);
            InvalidateTensorDeviceCache(vCache);
        }

    }
}
