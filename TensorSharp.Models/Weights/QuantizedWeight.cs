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
using TensorSharp.Runtime;
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
    public partial class QuantizedWeight : IDisposable
    {
        private IntPtr _data;
        private GCHandle _cacheKeyHandle;

        public IntPtr Data => _data;
        public IntPtr CacheKey { get; private set; }
        public int GgmlType { get; }
        public long Ne0 { get; }
        public long Ne1 { get; }
        public long RawBytes { get; }

        /// <summary>Per-tensor matmul-output multiplier from an optional
        /// sidecar "&lt;base&gt;.scale" tensor (NVFP4 scale2, HF weight_scale_2).
        /// The true weight is (quantized blocks) x Scale; consumers multiply
        /// the projection output by it. 1.0 = no sidecar.</summary>
        public float Scale { get; internal set; } = 1.0f;
        private bool _ownsBuffer;
        private bool _ownsCacheKeyHandle;
        private object _ownerToken;
        public bool HasHostData => _data != IntPtr.Zero;
        public bool HasExternalHostView => _data != IntPtr.Zero && !_ownsBuffer && _ownerToken != null;
        internal bool IsExternalHostViewOwnedBy(object owner)
            => HasExternalHostView && ReferenceEquals(_ownerToken, owner);

        /// <summary>
        /// True when the active device could not hold this weight in a single
        /// backend buffer (e.g. ggml-vulkan rejects any buffer above the driver's
        /// maxBufferSize; WSL's dzn layer caps it under 3 GB), so the device
        /// preload was skipped and the host copy retained. Consumers must serve
        /// this weight through their host-gather/dequant fallback instead of
        /// device-side lookups keyed by <see cref="CacheKey"/>.
        /// </summary>
        public bool DevicePreloadTooLarge { get; private set; }

        public QuantizedWeight(byte[] raw, int ggmlType, long ne0, long ne1)
        {
            GgmlType = ggmlType;
            Ne0 = ne0;
            Ne1 = ne1;
            RawBytes = raw.Length;
            _data = AllocateBuffer(raw.Length);
            CacheKey = _data;
            _ownsBuffer = true;
            Marshal.Copy(raw, 0, _data, raw.Length);
        }

        public QuantizedWeight(IntPtr data, long rawBytes, int ggmlType, long ne0, long ne1)
            : this(data, rawBytes, ggmlType, ne0, ne1, true, null)
        {
        }

        private QuantizedWeight(IntPtr data, long rawBytes, int ggmlType, long ne0, long ne1, bool ownsBuffer, object ownerToken)
        {
            _data = data;
            CacheKey = data;
            RawBytes = rawBytes;
            GgmlType = ggmlType;
            Ne0 = ne0;
            Ne1 = ne1;
            _ownsBuffer = ownsBuffer;
            _ownerToken = ownerToken;
        }

        /// <summary>
        /// A non-owning view over ONE expert of a stacked expert tensor
        /// (<c>ffn_gate_exps</c> and friends), so a model whose GGUF stacks its
        /// experts can still drive the per-expert linear paths that unstacked files
        /// get for free. The view borrows the stack's buffer - it must not free it -
        /// and holds the stack as its owner token so the memory outlives it.
        /// </summary>
        public static QuantizedWeight CreateExpertView(StackedExpertWeights stacked, int expert)
        {
            if (stacked == null)
                throw new ArgumentNullException(nameof(stacked));
            if (expert < 0 || expert >= stacked.NumExperts)
                throw new ArgumentOutOfRangeException(nameof(expert));

            return new QuantizedWeight(
                stacked.Data + (nint)(expert * stacked.PerExpertRawBytes),
                stacked.PerExpertRawBytes,
                stacked.GgmlType,
                stacked.PerExpertNe0,
                stacked.PerExpertNe1,
                ownsBuffer: false,
                ownerToken: stacked);
        }

        public void Dispose()
        {
            try
            {
                UnregisterBonsaiWeights();
            }
            finally
            {
                // A failed native cleanup must never strand an owned buffer or
                // the GCHandle that roots this weight during construction.
                try { ReleaseHostData(); }
                finally
                {
                    if (_ownsCacheKeyHandle)
                    {
                        try { UnregisterBonsaiCacheKey(CacheKey); }
                        finally
                        {
                            _cacheKeyHandle.Free();
                            _ownsCacheKeyHandle = false;
                            CacheKey = IntPtr.Zero;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// True when this view's bytes live in a FILE mapping rather than in
        /// heap memory.
        ///
        /// <para>The distinction matters because releasing a view calls
        /// <c>madvise(MADV_DONTNEED)</c> on it. On a file mapping that drops
        /// clean page-cache pages, which is the whole point — the bytes come
        /// back from the file if anything touches them again. On PRIVATE ANONYMOUS
        /// memory the same call is destructive: the kernel discards the pages and
        /// zero-fills them on next touch. A view is an interior pointer into a
        /// larger buffer and the advised range is rounded outward to page
        /// boundaries, so on a small allocation that zero-fill lands on the
        /// allocator's metadata for NEIGHBOURING chunks and the next unrelated
        /// free() aborts the process. That is not hypothetical: it is what a
        /// quantized model whose expert tensors could not be mmapped did, every
        /// time, on glibc.</para>
        ///
        /// <para>Owners are recursive — a tensor-parallel shard is a view of a
        /// parent view — so the question is answered by walking to the root.</para>
        /// </summary>
        private bool ViewIsFileBacked => _ownsBuffer
            ? false
            : _ownerToken switch
            {
                GgufFile => true,
                StackedExpertWeights stacked => stacked.IsExternalView,
                QuantizedWeight parent => parent.ViewIsFileBacked,
                _ => false,
            };

        public static QuantizedWeight CreateExternalView(IntPtr data, long rawBytes, int ggmlType, long ne0, long ne1, object ownerToken)
        {
            if (data == IntPtr.Zero)
                throw new ArgumentException("External quantized weight view requires a non-zero data pointer.", nameof(data));
            if (ownerToken == null)
                throw new ArgumentNullException(nameof(ownerToken));

            return new QuantizedWeight(data, rawBytes, ggmlType, ne0, ne1, false, ownerToken);
        }

        public static bool TryCreateConcatenatedView(out QuantizedWeight fused, params QuantizedWeight[] weights)
        {
            fused = null;
            if (weights == null || weights.Length < 2 || weights[0] == null)
                return false;

            QuantizedWeight first = weights[0];
            if (!first.HasHostData || first._ownsBuffer || first._ownerToken == null)
                return false;

            long totalBytes = 0;
            long totalNe1 = 0;
            long expectedAddress = first.Data.ToInt64();

            for (int i = 0; i < weights.Length; i++)
            {
                QuantizedWeight weight = weights[i];
                if (weight == null ||
                    weight._ownsBuffer ||
                    !ReferenceEquals(weight._ownerToken, first._ownerToken) ||
                    weight.GgmlType != first.GgmlType ||
                    weight.Ne0 != first.Ne0 ||
                    weight.Data.ToInt64() != expectedAddress)
                {
                    return false;
                }

                totalBytes += weight.RawBytes;
                totalNe1 += weight.Ne1;
                expectedAddress += weight.RawBytes;
            }

            fused = new QuantizedWeight(first.Data, totalBytes, first.GgmlType, first.Ne0, totalNe1, false, first._ownerToken);
            return true;
        }

        public static unsafe QuantizedWeight ConcatOrCreateCopy(params QuantizedWeight[] weights)
        {
            if (weights == null || weights.Length == 0 || weights[0] == null)
                throw new ArgumentException("At least one quantized weight is required.", nameof(weights));

            if (TryCreateConcatenatedView(out QuantizedWeight fused, weights))
                return fused;

            QuantizedWeight first = weights[0];
            long totalBytes = 0;
            long totalNe1 = 0;
            for (int i = 0; i < weights.Length; i++)
            {
                QuantizedWeight weight = weights[i] ?? throw new ArgumentException("Quantized weight list cannot contain null entries.", nameof(weights));
                if (!weight.HasHostData)
                    throw new InvalidOperationException("Cannot concatenate quantized weights after their host storage has been released.");
                totalBytes += weight.RawBytes;
                totalNe1 += weight.Ne1;
            }

            IntPtr fusedPtr = AllocateBuffer(totalBytes);
            byte* fusedDst = (byte*)fusedPtr.ToPointer();
            long offset = 0;
            for (int i = 0; i < weights.Length; i++)
            {
                QuantizedWeight weight = weights[i];
                Buffer.MemoryCopy(weight.Data.ToPointer(), fusedDst + offset, totalBytes - offset, weight.RawBytes);
                offset += weight.RawBytes;
            }

            return new QuantizedWeight(fusedPtr, totalBytes, first.GgmlType, first.Ne0, totalNe1);
        }

        public IntPtr EnsureDeviceCacheKey()
        {
            if (_ownsCacheKeyHandle)
                return CacheKey;

            // Once flagged too-large the cache key must stay the host data
            // pointer: no device-resident entry exists for this weight, and a
            // native cache miss on an opaque GCHandle key would dereference it
            // as if it were weight bytes.
            if (DevicePreloadTooLarge)
                return CacheKey;

            _cacheKeyHandle = GCHandle.Alloc(this, GCHandleType.Normal);
            CacheKey = GCHandle.ToIntPtr(_cacheKeyHandle);
            _ownsCacheKeyHandle = true;
            RegisterBonsaiCacheKey(CacheKey);
            return CacheKey;
        }

        /// <summary>
        /// Record that the device preload was skipped because this weight exceeds
        /// the device's single-buffer size limit. Frees any GCHandle-based device
        /// cache key and restores <see cref="CacheKey"/> to the host data pointer,
        /// so a native call that still receives the key resolves through the
        /// host-pointer path instead of dereferencing an opaque GCHandle.
        /// </summary>
        public void MarkDevicePreloadTooLarge()
        {
            DevicePreloadTooLarge = true;
            if (_ownsCacheKeyHandle)
            {
                UnregisterBonsaiCacheKey(CacheKey);
                _cacheKeyHandle.Free();
                _ownsCacheKeyHandle = false;
            }
            CacheKey = _data;
        }

        public void ReleaseHostData()
        {
            if (_data == IntPtr.Zero)
                return;

            IntPtr currentData = _data;
            try { UnregisterBonsaiCacheKey(currentData); }
            finally
            {
                if (_ownsBuffer)
                    FreeBuffer(currentData);
                else if (ViewIsFileBacked)
                    AdviseExternalViewCanBePagedOut(currentData, RawBytes);

                if (CacheKey == currentData)
                    CacheKey = IntPtr.Zero;

                _data = IntPtr.Zero;
                _ownsBuffer = false;
                _ownerToken = null;
            }
        }

        // Large buffers are mapped rather than malloc'd on Apple platforms, so the ones
        // a load frees after uploading them leave the footprint (see HostBuffers).
        public static IntPtr AllocateBuffer(long size) => HostBuffers.Allocate(size);

        public static void FreeBuffer(IntPtr ptr) => HostBuffers.Free(ptr);

        /// <summary>
        /// Hint that a released FILE-MAPPED view's pages can go. Callers must
        /// have established that (see <see cref="ViewIsFileBacked"/>) — on
        /// anonymous memory this call destroys data rather than freeing it.
        /// </summary>
        private static unsafe void AdviseExternalViewCanBePagedOut(IntPtr data, long byteCount)
        {
            if (data == IntPtr.Zero || byteCount <= 0)
                return;
            // Darwin (macOS/iOS/iPadOS/Mac Catalyst) and Linux share the madvise
            // entry point; MADV_DONTNEED = 4 is correct on both.
            if (!OperatingSystem.IsMacOS() && !OperatingSystem.IsLinux()
                && !OperatingSystem.IsIOS() && !OperatingSystem.IsMacCatalyst())
                return;

            long pageSize = Environment.SystemPageSize;
            long address = data.ToInt64();
            long pageMask = ~(pageSize - 1);
            long alignedAddress = address & pageMask;
            long prefixBytes = address - alignedAddress;
            ulong length = checked((ulong)(byteCount + prefixBytes));
            ulong roundedLength = (length + (ulong)pageSize - 1) & ~((ulong)pageSize - 1);

            try
            {
                _ = madvise((void*)alignedAddress, (nuint)roundedLength, MadvDontNeed);
            }
            catch (DllNotFoundException)
            {
            }
            catch (EntryPointNotFoundException)
            {
            }
        }

        private const int MadvDontNeed = 4;

        [LibraryImport("libc", EntryPoint = "madvise", SetLastError = true)]
        private static unsafe partial int madvise(void* addr, nuint len, int advice);
    }

    /// <summary>
    /// A view of a per-layer 3D MoE expert weight tensor as stored on disk
    /// (<c>[ne0, ne1, num_experts]</c> contiguous). Built when the per-expert
    /// quantized weights are split out of the original 3D GGUF tensor in
    /// <see cref="ModelBase.LoadWeights"/>, so it costs nothing on top of the
    /// per-expert weights for mmap'd models — the base pointer is the start
    /// of the original 3D block and the bytes are the same bytes the per-expert
    /// views point into.
    ///
    /// The <see cref="MoEFFNPrefillSwiGLU"/> kernel consumes this directly to
    /// run an entire MoE layer's gate/up/down via three <c>ggml_mul_mat_id</c>
    /// dispatches (mirroring llama.cpp's <c>build_moe_ffn</c>) instead of the
    /// previous per-active-expert loop that issued thousands of dispatches per
    /// pp2048 forward.
    /// </summary>
    public sealed class StackedExpertWeights
    {
        public IntPtr Data { get; }
        public int GgmlType { get; }
        public long PerExpertNe0 { get; }
        public long PerExpertNe1 { get; }
        public int NumExperts { get; }
        public long TotalRawBytes { get; }
        public long PerExpertRawBytes => TotalRawBytes / NumExperts;
        public bool IsExternalView { get; }

        // Strong reference held to keep the underlying memory alive when this
        // is an external view (e.g. into a GgufFile mmap or a sibling owning
        // QuantizedWeight buffer). For owned buffers this is null.
        private readonly object _ownerToken;

        // For the non-mmap fallback path we own a pinned native buffer and
        // free it on disposal of the parent ModelBase. Tracked so the buffer
        // doesn't leak when ModelBase exits.
        public IntPtr OwnedBuffer { get; }

        public StackedExpertWeights(
            IntPtr data,
            int ggmlType,
            long perExpertNe0,
            long perExpertNe1,
            int numExperts,
            long totalRawBytes,
            bool isExternalView,
            object ownerToken,
            IntPtr ownedBuffer)
        {
            Data = data;
            GgmlType = ggmlType;
            PerExpertNe0 = perExpertNe0;
            PerExpertNe1 = perExpertNe1;
            NumExperts = numExperts;
            TotalRawBytes = totalRawBytes;
            IsExternalView = isExternalView;
            _ownerToken = ownerToken;
            OwnedBuffer = ownedBuffer;
        }
    }
}
