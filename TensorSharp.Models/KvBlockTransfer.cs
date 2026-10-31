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
using System.Collections.Generic;
using TensorSharp;

namespace TensorSharp.Models
{
    /// <summary>
    /// Backend-portable byte serialization helper for per-layer K/V cache tensors of
    /// shape <c>(numKVHeads, rows, headDim)</c>. Used by the paged KV cache
    /// machinery to capture and restore block-aligned token ranges across sessions.
    ///
    /// We deliberately do NOT route through <see cref="Ops.Copy"/> because the GGML
    /// op registry only services F32 tensors, while the KV cache routinely lives in
    /// F16 or Q8_0 storage. Instead we drain pending async compute via
    /// <see cref="Storage.EnsureHostReadable"/>, take the storage's base host
    /// pointer (every backend in the repo - GGML, CUDA, MLX, CPU - exposes a host
    /// mirror through <see cref="Storage.PtrAtElement"/> at index 0) and then walk
    /// the (head, position) rows with a per-row <c>memcpy</c>. For Q8_0 we use the
    /// block-bytes-per-row figure derived from <see cref="Storage.ByteLength"/>
    /// instead of <c>ElementType.Size()</c>, which is the only safe way to address
    /// Q8_0 at offsets other than zero.
    ///
    /// A cache tensor is addressed one of two ways:
    ///  - LINEAR (the default): row index == absolute token position, so the tensor
    ///    must have at least <c>startToken + tokenCount</c> rows.
    ///  - RING: row index == <c>position % rows</c>, and only the most recent
    ///    <c>rows</c> positions are physically present. Muse-Glimmer sizes its
    ///    sliding-window layers this way (see <c>MuseGlimmerModel._kvSwaRows</c>) so
    ///    a 64K context does not pay for 64K rows on the 39 layers that never look
    ///    back further than the window. Callers opt in per model by passing
    ///    <c>ringRows</c>; a tensor is treated as a ring exactly when its row count
    ///    equals that value.
    ///
    /// Every range is validated against each tensor's OWN row count before a single
    /// byte moves. That check is what stops a short ring layer from being addressed
    /// as if it had the full context's rows - which read past the end of the
    /// storage and faulted the process (AccessViolationException in
    /// <c>Buffer.MemoryCopy</c>) once a Muse-Glimmer sequence passed the ring size.
    /// </summary>
    internal static class KvBlockTransfer
    {
        /// <summary>
        /// Total bytes required for one block of <paramref name="tokenCount"/> tokens
        /// across every <c>(K, V)</c> pair in <paramref name="kCache"/> /
        /// <paramref name="vCache"/>. Storage objects shared across layers (e.g.
        /// Gemma 4's donor/alias map) are counted exactly once - that's what makes
        /// it safe to extract/inject without double-writing the same bytes.
        /// </summary>
        public static long ComputeBlockByteSize(Tensor[] kCache, Tensor[] vCache, int tokenCount)
        {
            if (kCache == null || vCache == null || kCache.Length != vCache.Length || tokenCount <= 0)
                return 0;

            long total = 0;
            foreach (var t in EnumerateUniqueStorageTensors(kCache, vCache))
                total += PerLayerBlockBytes(t, tokenCount);
            return total;
        }

        /// <param name="ringRows">Row count that identifies a ring-addressed layer,
        /// or 0 when every layer is linear. See the type remarks.</param>
        public static bool Extract(
            IAllocator allocator,
            Tensor[] kCache,
            Tensor[] vCache,
            int currentSeqLen,
            int startToken,
            int tokenCount,
            Span<byte> destination,
            int ringRows = 0)
        {
            if (!ValidateArgs(kCache, vCache, tokenCount))
                return false;
            if (startToken < 0 || startToken + tokenCount > currentSeqLen)
                return false;
            long expected = ComputeBlockByteSize(kCache, vCache, tokenCount);
            if (destination.Length != expected)
                return false;
            // Validate every layer BEFORE moving any bytes: a mid-way bail-out would
            // leave the caller holding a half-written block that looks like a failure
            // but has already clobbered its scratch buffer.
            if (!ValidateRows(kCache, vCache, currentSeqLen, startToken, tokenCount, ringRows, reading: true))
                return false;

            int offset = 0;
            foreach (var t in EnumerateUniqueStorageTensors(kCache, vCache))
            {
                if (!CopyOneOut(t, RowStart(t, startToken, ringRows), tokenCount, destination[offset..], out int written))
                    return false;
                offset += written;
            }
            return true;
        }

        /// <param name="ringRows">Row count that identifies a ring-addressed layer,
        /// or 0 when every layer is linear. See the type remarks.</param>
        public static bool Inject(
            IAllocator allocator,
            Tensor[] kCache,
            Tensor[] vCache,
            int currentSeqLen,
            int destToken,
            int tokenCount,
            ReadOnlySpan<byte> source,
            int ringRows = 0)
        {
            if (!ValidateArgs(kCache, vCache, tokenCount))
                return false;
            // We only permit appending in order. Higher-level code resets the cache
            // before restoration and accumulates blocks from position 0 onward. That
            // ordering is also what makes ring layers restorable: replaying the
            // positions in the order the original forward wrote them leaves the ring
            // holding exactly the rows a real prefill would have left.
            if (destToken != currentSeqLen)
                return false;
            long expected = ComputeBlockByteSize(kCache, vCache, tokenCount);
            if (source.Length != expected)
                return false;
            if (!ValidateRows(kCache, vCache, currentSeqLen, destToken, tokenCount, ringRows, reading: false))
                return false;

            int offset = 0;
            foreach (var t in EnumerateUniqueStorageTensors(kCache, vCache))
            {
                if (!CopyOneIn(t, RowStart(t, destToken, ringRows), tokenCount, source[offset..], out int read))
                    return false;
                offset += read;
            }
            return true;
        }

        /// <summary>
        /// Walk K[0], V[0], K[1], V[1], ... but yield each unique <see cref="Storage"/>
        /// at most once. The deterministic order matters because the captured byte
        /// blob is laid out in the same order the enumerator visits tensors; flipping
        /// the order would silently corrupt restoration.
        /// </summary>
        private static IEnumerable<Tensor> EnumerateUniqueStorageTensors(Tensor[] kCache, Tensor[] vCache)
        {
            var seen = new HashSet<Storage>(ReferenceEqualityComparer.Instance);
            for (int l = 0; l < kCache.Length; l++)
            {
                Tensor k = kCache[l];
                if (k != null && seen.Add(k.Storage))
                    yield return k;
                Tensor v = vCache[l];
                if (v != null && seen.Add(v.Storage))
                    yield return v;
            }
        }

        private static bool ValidateArgs(Tensor[] kCache, Tensor[] vCache, int tokenCount)
        {
            if (kCache == null || vCache == null || kCache.Length == 0)
                return false;
            if (kCache.Length != vCache.Length)
                return false;
            if (tokenCount <= 0)
                return false;
            DType firstDtype = DType.Float32;
            bool first = true;
            for (int l = 0; l < kCache.Length; l++)
            {
                if (kCache[l] == null || vCache[l] == null)
                    return false;
                if (kCache[l].Sizes.Length != 3 || vCache[l].Sizes.Length != 3)
                    return false;
                if (kCache[l].StorageOffset != 0 || vCache[l].StorageOffset != 0)
                    return false;
                if (kCache[l].Storage.ElementType != vCache[l].Storage.ElementType)
                    return false;
                if (first)
                {
                    firstDtype = kCache[l].Storage.ElementType;
                    first = false;
                }
                else if (kCache[l].Storage.ElementType != firstDtype)
                {
                    return false; // mixed dtypes across layers would invalidate the fingerprint
                }
            }
            return true;
        }

        /// <summary>
        /// Per-tensor range check. Returns false - rather than reading or writing out
        /// of bounds - whenever the requested positions are not addressable in that
        /// tensor's own rows.
        /// </summary>
        private static bool ValidateRows(
            Tensor[] kCache,
            Tensor[] vCache,
            int currentSeqLen,
            int position,
            int tokenCount,
            int ringRows,
            bool reading)
        {
            foreach (var t in EnumerateUniqueStorageTensors(kCache, vCache))
            {
                long rows = t.Sizes[1];
                if (rows <= 0)
                    return false;
                if (IsRing(t, ringRows))
                {
                    // A block wider than the ring would alias onto itself.
                    if (tokenCount > rows)
                        return false;
                    // The ring physically holds only [currentSeqLen - rows, currentSeqLen).
                    // Anything older has been overwritten and cannot be recovered.
                    if (reading && (long)currentSeqLen - position > rows)
                        return false;
                }
                else if ((long)position + tokenCount > rows)
                {
                    return false;
                }
            }
            return true;
        }

        private static bool IsRing(Tensor tensor, int ringRows)
            => ringRows > 0 && tensor.Sizes[1] == ringRows;

        private static long RowStart(Tensor tensor, int position, int ringRows)
            => IsRing(tensor, ringRows) ? position % tensor.Sizes[1] : position;

        private static long PerLayerBlockBytes(Tensor tensor, int tokenCount)
        {
            // Use the actual on-storage row stride - this gracefully handles Q8_0
            // (where headDim/32 * 34 bytes per row != headDim * 1 byte/elem).
            return RowBytes(tensor) * tensor.Sizes[0] * tokenCount;
        }

        private static long RowBytes(Tensor tensor)
        {
            // Bytes per (head, position) row, derived from the storage byte length
            // and the tensor extents. Works for F32 / F16 / Q8_0 alike provided the
            // tensor is contiguous at offset 0 (validated upstream).
            long numKVHeads = tensor.Sizes[0];
            long rows = tensor.Sizes[1];
            return tensor.Storage.ByteLength / (numKVHeads * rows);
        }

        private static bool CopyOneOut(Tensor cacheTensor, long rowStart, int tokenCount, Span<byte> destination, out int bytesWritten)
        {
            bytesWritten = 0;
            cacheTensor.Storage.EnsureHostReadable();

            long numKVHeads = cacheTensor.Sizes[0];
            long rows = cacheTensor.Sizes[1];
            long rowBytes = RowBytes(cacheTensor);
            long blockBytes = numKVHeads * tokenCount * rowBytes;
            if (destination.Length < blockBytes)
                return false;
            if (rowStart < 0 || rowStart >= rows || tokenCount > rows)
                return false;

            // Read only: PtrAtElement would mark a direct-CUDA cache host-dirty, and the next K/V
            // append would then upload the whole unchanged cache before writing its rows.
            IntPtr storageBase = cacheTensor.Storage.PtrAtElementForRead(0);
            if (storageBase == IntPtr.Zero)
                return false;

            // On a ring layer the run [rowStart, rowStart+tokenCount) can spill past
            // the last row and continue at row 0; split it so both halves stay inside
            // the head's row span. Linear layers always take the first branch only.
            long headRows = Math.Min(tokenCount, rows - rowStart);
            long wrapRows = tokenCount - headRows;

            unsafe
            {
                byte* src = (byte*)storageBase;
                fixed (byte* dstBase = destination)
                {
                    long perHead = tokenCount * rowBytes;
                    for (long h = 0; h < numKVHeads; h++)
                    {
                        long headBase = h * rows * rowBytes;
                        long dstOffset = h * perHead;
                        Buffer.MemoryCopy(
                            src + headBase + rowStart * rowBytes,
                            dstBase + dstOffset,
                            destination.Length - dstOffset,
                            headRows * rowBytes);
                        if (wrapRows > 0)
                        {
                            long wrapDst = dstOffset + headRows * rowBytes;
                            Buffer.MemoryCopy(
                                src + headBase,
                                dstBase + wrapDst,
                                destination.Length - wrapDst,
                                wrapRows * rowBytes);
                        }
                    }
                }
            }

            bytesWritten = (int)blockBytes;
            return true;
        }

        private static bool CopyOneIn(Tensor cacheTensor, long rowStart, int tokenCount, ReadOnlySpan<byte> source, out int bytesRead)
        {
            bytesRead = 0;
            cacheTensor.Storage.EnsureHostReadable();

            long numKVHeads = cacheTensor.Sizes[0];
            long rows = cacheTensor.Sizes[1];
            long rowBytes = RowBytes(cacheTensor);
            long blockBytes = numKVHeads * tokenCount * rowBytes;
            if (source.Length < blockBytes)
                return false;
            if (rowStart < 0 || rowStart >= rows || tokenCount > rows)
                return false;

            IntPtr storageBase = cacheTensor.Storage.PtrAtElement(0);
            if (storageBase == IntPtr.Zero)
                return false;

            long headRows = Math.Min(tokenCount, rows - rowStart);
            long wrapRows = tokenCount - headRows;
            long storageBytes = cacheTensor.Storage.ByteLength;

            unsafe
            {
                byte* dst = (byte*)storageBase;
                fixed (byte* srcBase = source)
                {
                    long perHead = tokenCount * rowBytes;
                    for (long h = 0; h < numKVHeads; h++)
                    {
                        long headBase = h * rows * rowBytes;
                        long dstOffset = headBase + rowStart * rowBytes;
                        long srcOffset = h * perHead;
                        Buffer.MemoryCopy(
                            srcBase + srcOffset,
                            dst + dstOffset,
                            storageBytes - dstOffset,
                            headRows * rowBytes);
                        if (wrapRows > 0)
                        {
                            Buffer.MemoryCopy(
                                srcBase + srcOffset + headRows * rowBytes,
                                dst + headBase,
                                storageBytes - headBase,
                                wrapRows * rowBytes);
                        }
                    }
                }
            }

            bytesRead = (int)blockBytes;
            return true;
        }
    }
}
