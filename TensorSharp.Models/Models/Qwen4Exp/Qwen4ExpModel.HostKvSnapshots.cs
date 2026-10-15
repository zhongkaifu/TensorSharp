// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Runtime.InteropServices;
using System.Threading;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public partial class Qwen4ExpModel
{
    /// <summary>Pressure-only demotion of an inactive complete holder to its
    /// existing, charged host mirrors. This is not the independent block-KV
    /// snapshot route: GDN, PLE and QSA end state travels with the entire holder.
    /// MTP and multi-device layouts require their own accepted state transfer.</summary>
    internal bool SupportsUnifiedRetainedHostKvDemotion => SupportsRetainedFusedCache
        && !IsTensorParallel && LayerSplitDegree <= 1 && !HasDraftHead;

    internal int UnifiedRetainedDeviceHolderCount
    {
        get
        {
            if (!SupportsUnifiedRetainedHostKvDemotion || _retainedFusedHolders == null) return 0;
            int count = 0;
            foreach (var holder in _retainedFusedHolders.Values)
                if (CanDemoteRetainedHostKv(holder)) count++;
            return count;
        }
    }

    internal long UnifiedRetainedHostKvDemotions { get; private set; }

    /// <summary>Peak additional host payload for one sequential export. The
    /// caller must reserve this under a new maintenance owner before entering
    /// the host allocation scope; retained tensors may belong to closed request
    /// envelopes and cannot fund new scratch allocations themselves.</summary>
    internal long GetUnifiedRetainedHostKvDemotionScratchBytes()
    {
        if (!SupportsUnifiedRetainedHostKvDemotion || _retainedFusedHolders == null) return 0;
        long peak = 0;
        foreach (var holder in _retainedFusedHolders.Values)
            if (CanDemoteRetainedHostKv(holder)) peak = Math.Max(peak, ValidateHostKvDemotion(holder));
        // GGML host allocations charge their physically page-aligned extent,
        // rather than the smaller logical tensor. This adapter currently runs
        // on one CUDA device; its host allocator uses the OS page size.
        long alignment = Math.Max(32, Environment.SystemPageSize);
        return checked((peak + alignment - 1) / alignment * alignment);
    }

    /// <summary>Called only at a fenced compute boundary, under a separately
    /// admitted host maintenance scope. All authoritative state is exported
    /// before any device binding is freed. Cancellation before eviction keeps
    /// the device owner usable; once physical eviction begins it completes
    /// without cancellation. Failed physical release quarantines the reachable
    /// holder rather than advertising a partially released device state.</summary>
    internal int DemoteUnifiedRetainedHostKv(int maximumHolders = 1,
        CancellationToken cancellationToken = default, HostKvTransferHooks hooks = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(maximumHolders);
        cancellationToken.ThrowIfCancellationRequested();
        if (maximumHolders == 0 || !SupportsUnifiedRetainedHostKvDemotion || _retainedFusedHolders == null)
            return 0;
        int demoted = 0;
        // Retained insertion order supplies a stable oldest-first pressure
        // order. This adapter neither removes payloads nor invalidates their
        // radix keys; a subsequent clone/donation restores from host state.
        foreach (var pair in _retainedFusedHolders)
        {
            var holder = pair.Value;
            if (!CanDemoteRetainedHostKv(holder)) continue;
            ValidateHostKvDemotion(holder);
            var previousFootprint = MeasureHolderFootprint(holder);
            ExportRetainedHostKv(holder, cancellationToken, hooks);
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                var keys = HolderStateKeys(holder);
                if (hooks == null) GgmlBasicOps.Qwen4ExpReleaseSeqState(keys);
                else hooks.ReleaseState(keys);
                CountDecodeGraphReset();
                foreach (var set in new[] { holder.K, holder.V, holder.IdxK, holder.GdnConvStateT, holder.GdnStateT })
                    if (set != null)
                        foreach (var tensor in set)
                            if (tensor != null)
                            {
                                if (hooks == null) InvalidateTensorDeviceCache(tensor);
                                else hooks.InvalidateTensor(tensor);
                            }
            }
            catch
            {
                // Exports succeeded, but physical ownership may now be split.
                // Do not let rebind/clone or a later graph assume seeded device
                // entries still exist. Normal holder disposal retains physical
                // allocation ownership until each release succeeds on retry.
                holder.SpecStateFailed = true;
                holder.Retired = true;
                throw;
            }
            holder.KvHostStale = false;
            holder.DeviceStateSeeded = false;
            holder.DeviceMirrored = false;
            UnifiedRetainedHostKvDemotions++;
            demoted++;
            _prefixCacheSink?.OnPayloadDeviceDemoted(pair.Key, previousFootprint, MeasureHolderFootprint(holder));
            if (demoted == maximumHolders) break;
        }
        return demoted;
    }

    private bool CanDemoteRetainedHostKv(Qwen4ExpKvCacheHolder holder)
    {
        if (holder == null || holder.Retired || holder.Disposed || holder.SpecStateFailed
            || !holder.DeviceMirrored || holder.CacheSeqLen <= 0 || holder.K == null
            || ReferenceEquals(holder.K, _kCache) || MtpStateBytes((object)holder.GdnConvStateT ?? holder.K) > 0)
            return false;
        // The retention registry normally excludes live holders. Refuse an
        // alias even if an earlier failed publication left it in both sets.
        if (_fusedHolders != null)
            foreach (var live in _fusedHolders.Values)
                if (ReferenceEquals(live, holder) || ReferenceEquals(live.K, holder.K)) return false;
        return true;
    }

    private long ValidateHostKvDemotion(Qwen4ExpKvCacheHolder holder)
    {
        int layers = Config.NumLayers;
        if (holder.KvCapacity <= 0 || holder.CacheSeqLen > holder.KvCapacity
            || holder.K.Length != layers || holder.V?.Length != layers || holder.IdxK?.Length != layers
            || _isRecurrent?.Length != layers || holder.GdnConvStateT?.Length != layers
            || holder.GdnStateT?.Length != layers || holder.GdnConvState?.Length != layers
            || holder.GdnConvWriteIdx?.Length != layers || holder.QsaPositionCount < 0
            || (holder.QsaPositionCount > 0 && (holder.QsaPositions == null
                || checked(3L * holder.QsaPositionCount) > holder.QsaPositions.LongLength)))
            throw new InvalidOperationException("qwen4exp host KV: incomplete holder shape or position history.");
        long peak = 0;
        for (int layer = 0; layer < layers; ++layer)
        {
            if (!_isRecurrent[layer])
            {
                if (holder.K[layer] == null || holder.V[layer] == null
                    || holder.K[layer].Sizes.Length != 3 || holder.V[layer].Sizes.Length != 3
                    || holder.K[layer].Sizes[1] != holder.KvCapacity || holder.V[layer].Sizes[1] != holder.KvCapacity
                    || (UsesQsa(layer) && (holder.IdxK[layer] == null
                        || holder.QsaPositionCount != holder.CacheSeqLen)))
                    throw new InvalidOperationException("qwen4exp host KV: attention state is incomplete.");
                continue;
            }
            var conv = holder.GdnConvStateT[layer];
            var state = holder.GdnStateT[layer];
            long convBytes = checked((long)(_convKernel - 1) * _convDim * sizeof(float));
            long stateBytes = checked((long)_numVHeads * _headVDim * _headVDim * sizeof(float));
            if (conv == null || state == null || conv.ElementType != DType.Float32 || state.ElementType != DType.Float32
                || conv.Storage.ByteLength != convBytes || state.Storage.ByteLength != stateBytes
                || holder.GdnConvState[layer]?.LongLength != convBytes / sizeof(float)
                || holder.GdnConvWriteIdx[layer] < 0 || holder.GdnConvWriteIdx[layer] >= Math.Max(1, _convKernel - 1))
                throw new InvalidOperationException("qwen4exp host KV: the complete GDN state layout cannot be exported.");
            peak = Math.Max(peak, checked(((convBytes + 255) & ~255L) + stateBytes));
        }
        if (_pleHeads > 0 && (holder.PleConvState == null
            || holder.PleConvState.LongLength != checked((long)(_pleConvKernel - 1) * _pleNgram * _hcDim)
            || holder.PleHistory == null || holder.PleNextPos < holder.CacheSeqLen))
            throw new InvalidOperationException("qwen4exp host KV: PLE conv/history state is incomplete.");
        return peak;
    }

    private unsafe void ExportRetainedHostKv(Qwen4ExpKvCacheHolder holder,
        CancellationToken cancellationToken, HostKvTransferHooks hooks)
    {
        cancellationToken.ThrowIfCancellationRequested();
        // Arena rows and recurrence may be newer than solo device entries.
        // Flush is status checked and fences a failed conversation; never copy
        // the older host seeds as a fallback for a seeded device entry.
        if (hooks == null) FlushArenaCache(holder.K);
        else hooks.FlushArena(holder.K);
        for (int layer = 0; layer < Config.NumLayers; ++layer)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (_isRecurrent[layer])
            {
                if (!holder.DeviceStateSeeded) continue; // Existing host seeds are authoritative.
                var conv = holder.GdnConvStateT[layer];
                var state = holder.GdnStateT[layer];
                long convBytes = conv.Storage.ByteLength, stateBytes = state.Storage.ByteLength;
                long stateOffset = (convBytes + 255) & ~255L;
                long bytes = checked(stateOffset + stateBytes);
                // This physical payload is charged by the active maintenance
                // allocation scope. One layer's scratch is freed before the
                // next layer, and the full holder is never duplicated.
                using var scratch = new Tensor(_allocator, DType.Float32, bytes / sizeof(float));
                Export((IntPtr)GetFloatPtr(conv), (IntPtr)GetFloatPtr(scratch), bytes, DeviceForLayer(layer), "GDN");
                conv.Storage.EnsureHostReadable(); state.Storage.EnsureHostReadable();
                Buffer.MemoryCopy(GetFloatPtr(scratch), GetFloatPtr(conv), convBytes, convBytes);
                Buffer.MemoryCopy((byte*)GetFloatPtr(scratch) + stateOffset, GetFloatPtr(state), stateBytes, stateBytes);
                continue;
            }
            if (holder.KvHostStale)
            {
                Sync(holder.K[layer]); Sync(holder.V[layer]);
            }
            if (holder.IdxK[layer] is { } indexer && holder.DeviceStateSeeded)
            {
                indexer.Storage.EnsureHostReadable();
                IntPtr pointer = TensorComputePrimitives.GetStoragePointer(indexer);
                Export(pointer, pointer, indexer.Storage.ByteLength, DeviceForLayer(layer), "QSA raw keys");
            }
        }
        if (holder.DeviceStateSeeded && holder.PleConvState is { Length: > 0 } ple)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var pointer = Marshal.UnsafeAddrOfPinnedArrayElement(ple, 0);
            Export(pointer, pointer, checked((long)ple.Length * sizeof(float)),
                DeviceForLayer(Math.Max(0, _pleLayerIndex)), "PLE conv history");
        }
        // Ring indices, PLE token/next-position state, three-axis QSA history,
        // mRoPE gap, capacity and logits already belong to this same holder.
        // They remain byte-for-byte unchanged throughout the transfer.
        void Sync(Tensor tensor)
        {
            if (hooks == null) SyncTensorHostCache(tensor);
            else hooks.SyncTensor(tensor);
        }
        void Export(IntPtr key, IntPtr target, long bytes, int device, string what)
        {
            if (hooks == null)
            {
                TryExportSeqState(key, target, bytes, device, seeded: true, what);
                return;
            }
            if (!hooks.ExportState(key, target, bytes, device))
                throw new InvalidOperationException($"qwen4exp host KV: authoritative {what} export failed.");
        }
    }

    /// <summary>Fault-injection seam for the actual complete-holder transfer.
    /// CPU ownership tests provide copied device bytes and fail individual
    /// transfers/releases; production always uses the native fenced methods.</summary>
    internal sealed class HostKvTransferHooks
    {
        internal required Action<Tensor[]> FlushArena { get; init; }
        internal required Action<Tensor> SyncTensor { get; init; }
        internal required Func<IntPtr, IntPtr, long, int, bool> ExportState { get; init; }
        internal required Action<IntPtr[]> ReleaseState { get; init; }
        internal required Action<Tensor> InvalidateTensor { get; init; }
    }
}
