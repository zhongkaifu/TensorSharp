// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Core;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    public partial class Qwen4ExpModel
    {
        private Qwen4ExpQsaArgs[] _qsaArgs;
        // Complete cell history, not the draft's pruned catch-up ranges. This
        // travels with its holder and is truncated on speculative rollback.
        private int[] _qsaPositions;
        private int _qsaPositionCount;
        private bool? _qsaApiAvailable;

        private bool HasQsa
        {
            get
            {
                if (_compressRatios == null || _indexerHeads <= 0 || _indexerHeadDim <= 0) return false;
                for (int l = 0; l < Config.NumLayers; ++l) if (UsesQsa(l)) return true;
                return false;
            }
        }

        internal static int CompareQsaPositions(int[] positions, int a, int b)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                int cmp = positions[checked(3 * a + axis)].CompareTo(positions[checked(3 * b + axis)]);
                if (cmp != 0) return cmp;
            }
            return 0;
        }

        internal static void WriteQsaPositions(int[] destination, int start, int count,
            int ropeStart, int[] multiAxis)
        {
            if (start < 0 || count <= 0 || (long)(start + (long)count) * 3 > destination.LongLength
                || ropeStart < 0 || (multiAxis != null && multiAxis.LongLength < (long)count * 3))
                throw new ArgumentException("qwen4exp QSA: invalid position history span.");
            // Validate before modifying history, including scalar overflow.
            for (int i = 0; i < count; ++i)
                for (int axis = 0; axis < 3; ++axis)
                    if ((multiAxis == null ? (long)ropeStart + i : multiAxis[3 * i + axis]) is < 0 or > int.MaxValue)
                        throw new ArgumentException("qwen4exp QSA: invalid rotary coordinate.");
            for (int i = 0; i < count; ++i)
                for (int axis = 0; axis < 3; ++axis)
                    destination[3 * (start + i) + axis] = multiAxis == null ? ropeStart + i : multiAxis[3 * i + axis];
        }

        private void PrepareQsaHistory(int start, int count)
        {
            if (!HasQsa) return;
            if (!(_qsaApiAvailable ??= GgmlBasicOps.Qwen4ExpQsaApiAvailable()))
                throw new NotSupportedException("qwen4exp QSA requires native Qwen API version 2; update the matching TensorSharp native library.");
            if (!IsGgmlBackend || !_tokenGraphEnabled || !_spanAttnEnabled || _tokenGraphUnsupported
                || _fusedGateUpExperts || !_fusedFfnEnabled || !_fusedGdnEnabled || !_fusedAttnEnabled
                || _gdnMaxLayers >= 0 || _gdnVerify)
                throw new NotSupportedException("qwen4exp QSA requires the complete GGML token-span path; the configured per-layer fallback cannot preserve its indexer state.");
            if (start != _qsaPositionCount)
                throw new InvalidOperationException("qwen4exp QSA: position history does not match the active cache head.");
            int length = checked(3 * _kvCacheCapacity);
            if (_qsaPositions == null || _qsaPositions.Length < length)
            {
                var grown = GC.AllocateArray<int>(length, pinned: true);
                if (_qsaPositions != null) Array.Copy(_qsaPositions, grown, checked(3 * start));
                _qsaPositions = grown;
            }
            int ropeStart = checked((int)Math.Max(0L, (long)start - _mropeCacheGap));
            WriteQsaPositions(_qsaPositions, start, count, ropeStart, _pendingMRoPEPositions);
            _qsaPositionCount = checked(start + count);
        }

        private unsafe bool EnsureQsaArgs()
        {
            if (!HasQsa || _qsaArgs != null) return true;
            if (!EnsureMropeSections()) throw new NotSupportedException("qwen4exp QSA requires valid rotary sections.");
            var args = GC.AllocateArray<Qwen4ExpQsaArgs>(Config.NumLayers, pinned: true);
            for (int l = 0; l < Config.NumLayers; ++l)
            {
                if (!UsesQsa(l)) continue;
                ref var a = ref args[l];
                string prefix = $"blk.{l}.indexer.";
                if (!TryResolveQuant(prefix + "k_proj.weight", out a.KProj, out a.KType, out a.KBytes)
                    || !TryResolveQuant(prefix + "q_proj.weight", out a.QProj, out a.QType, out a.QBytes))
                    throw new NotSupportedException("qwen4exp QSA indexer projection storage is unavailable.");
                a.KNorm = (IntPtr)GetFloatPtr(_weights[prefix + "k_norm.weight"]);
                a.QNorm = (IntPtr)GetFloatPtr(_weights[prefix + "q_norm.weight"]);
                a.Cache = (IntPtr)TensorComputePrimitives.GetStoragePointer(_idxKCache[l]);
                a.CacheBytes = _idxKCache[l].Storage.ByteLength;
                a.CacheType = _idxKCache[l].ElementType == DType.Float16 ? 1 : 0;
                a.HeadDim = _indexerHeadDim; a.Heads = _indexerHeads;
                a.Ratio = _compressRatios[l]; a.TopK = _indexerTopK;
                a.Section0 = _mropeSectionsPinned[0]; a.Section1 = _mropeSectionsPinned[1];
                a.Section2 = _mropeSectionsPinned[2]; a.Section3 = _mropeSectionsPinned[3];
            }
            _qsaArgs = args;
            return true;
        }

        private unsafe void InitializeQsaCache(Tensor tensor)
        {
            tensor.Storage.EnsureHostReadable();
            InvalidateTensorDeviceCache(tensor);
            NativeMemory.Clear(TensorComputePrimitives.GetStoragePointer(tensor).ToPointer(), checked((nuint)tensor.Storage.ByteLength));
        }

        private unsafe void SyncQsaCache(int layer)
        {
            var cache = _idxKCache[layer];
            cache.Storage.EnsureHostReadable();
            var pointer = (IntPtr)TensorComputePrimitives.GetStoragePointer(cache);
            if (!GgmlBasicOps.Qwen4ExpCopyQsaCache(pointer, pointer, cache.Storage.ByteLength, DeviceForLayer(layer)))
                throw new InvalidOperationException("qwen4exp QSA could not export the live indexer cache before growth.");
            InvalidateTensorDeviceCache(cache);
        }

        private unsafe void GrowQsaCache(int layer, int capacity, DType type)
        {
            Tensor old = _idxKCache[layer];
            Tensor grown = new Tensor(_allocator, type, 1, capacity, _indexerHeadDim);
            try
            {
                InitializeQsaCache(grown);
                // The one-head raw-key cache has a contiguous live prefix.
                long liveBytes = checked((long)_cacheSeqLen * _indexerHeadDim * (type == DType.Float16 ? 2 : 4));
                Buffer.MemoryCopy(TensorComputePrimitives.GetStoragePointer(old).ToPointer(),
                    TensorComputePrimitives.GetStoragePointer(grown).ToPointer(), grown.Storage.ByteLength, liveBytes);
                GgmlBasicOps.Qwen4ExpReleaseSeqState([(IntPtr)TensorComputePrimitives.GetStoragePointer(old)]);
                InvalidateTensorDeviceCache(old);
                old.Dispose();
                _idxKCache[layer] = grown;
                grown = null;
            }
            finally { grown?.Dispose(); }
        }
    }
}
