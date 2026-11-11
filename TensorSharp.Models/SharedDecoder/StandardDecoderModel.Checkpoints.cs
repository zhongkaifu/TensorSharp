// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

using TensorSharp.GGML;
using TensorSharp.Graph;

namespace TensorSharp.Models
{
    /// <summary>
    /// Shared-prefix checkpoints for every family on the runtime: a copy of a request's whole state
    /// (attention rows and recurrent states) at the boundary every conversation shares, cloned into
    /// each request that starts from it, and written to / read from files so a later process
    /// running the same weights starts there too. Copies go through the host (rows via the block
    /// transfer, which is right for every KV dtype; states via <see cref="RecurrentState.CopyFrom"/>).
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        private const uint CheckpointMagic = 0x4C535354; // "TSSL"
        private const int CheckpointVersion = 2;   // 2: the rotary delta

        public bool SupportsPrefixCheckpoints => SupportsPerSequenceFusedForward && RecurrentStatesCopy;

        public bool SupportsRetainedCacheSerialization => SupportsPrefixCheckpoints;

        /// <summary>The namespace a checkpoint belongs to: the cache identity and the backend kind
        /// (GGML backends keep recurrent state in device tensors, the others in managed arrays).</summary>
        private string CheckpointNamespace => KVStateFingerprint + "|" + (IsGgmlBackend ? "ggml" : _backend.ToString().ToLowerInvariant());

        private bool RecurrentStatesCopy
        {
            get
            {
                if (!_anyRecurrent) return true;
                if (_recurrentPrimary == null) return false;
                foreach (RecurrentState[] layer in _recurrentPrimary)
                    if (layer != null)
                        foreach (RecurrentState state in layer)
                            if (state != null && !state.CanCopy) return false;
                return true;
            }
        }

        public bool TryCheckpointActiveCache(string key)
        {
            if (!SupportsPrefixCheckpoints || string.IsNullOrEmpty(key)) return false;
            _retainedSlots ??= new Dictionary<string, Slot>(StringComparer.Ordinal);
            if (_retainedSlots.ContainsKey(key)) return false;
            Slot active = SnapshotActive();
            if (active.K == null || active.Length <= 0) return false;
            _retainedSlots.Add(key, CopySlot(active, active.Length));
            return true;
        }

        public bool TryCloneRetainedCache(string retainedKey, string newRequestId)
            => TryCloneRetainedCache(retainedKey, newRequestId, 0);

        public bool TryCloneRetainedCache(string retainedKey, string newRequestId, int requiredContextTokens)
        {
            if (_retainedSlots == null || string.IsNullOrEmpty(retainedKey) || string.IsNullOrEmpty(newRequestId)) return false;
            if (!_retainedSlots.TryGetValue(retainedKey, out Slot source) || source.K == null) return false;
            _slots ??= new Dictionary<string, Slot>(StringComparer.Ordinal);
            if (_slots.ContainsKey(newRequestId) || string.Equals(_activeKey, newRequestId, StringComparison.Ordinal)) return false;
            _slots.Add(newRequestId, CopySlot(source, Math.Max(source.Length, requiredContextTokens)));
            return true;
        }

        /// <summary>The retained slot can be copied (<see cref="CopySlot"/> reads its bytes wherever they are current).</summary>
        public bool SettleForCopy(string payloadKey)
            => _retainedSlots != null && payloadKey != null && _retainedSlots.ContainsKey(payloadKey);

        /// <summary>Slot copies made on the device (<see cref="TryCopySlotOnDevice"/>); tests check the path was taken.</summary>
        internal int DeviceSlotCopies { get; private set; }

        /// <summary>Copy on the device where the host copy is the cheaper one too (tests run the copy program on any
        /// GGML backend).</summary>
        internal bool ForceDeviceSlotCopies { get; set; }

        /// <summary>The executor keeps separate device copies, so a slot copy a graph left there stays there.</summary>
        internal bool SlotCopiesStayOnDevice => ForceDeviceSlotCopies
            || GraphExecutor is { Capabilities: { BindsHostPointers: true, SeparateDeviceCopies: true } };

        /// <summary>A deep copy of <paramref name="source"/> sized for <paramref name="capacityTokens"/>.</summary>
        private Slot CopySlot(Slot source, int capacityTokens)
        {
            int capacity = Math.Min(_maxContextLength, Math.Max(FreshSlotTokens, capacityTokens));
            // A pooled slot's caches and states are overwritten whole and keep their device copies (cached graphs
            // bind them), so the host route writes them in place.
            Slot slot = TakePooledSlot(capacity, mustFit: true);
            bool inPlace = slot != null;
            if (slot == null)
            {
                slot = new Slot { Capacity = capacity, Recurrent = _anyRecurrent ? NewRecurrentStates() : null };
                (slot.K, slot.V) = AllocateCaches(capacity);
            }
            slot.Length = source.Length;
            slot.DraftStart = source.DraftStart;
            slot.RopeDelta = source.RopeDelta;
            if (TryCopySlotOnDevice(source, slot))
                return slot;
            SyncToHost(source);
            CopyRowsOnHost(source.K, slot.K, source.Length, inPlace);
            CopyRowsOnHost(source.V, slot.V, source.Length, inPlace);
            bool kept = true;
            if (_anyRecurrent)
                for (int l = 0; l < source.Recurrent.Length; l++)
                    if (source.Recurrent[l] != null)
                        for (int r = 0; r < source.Recurrent[l].Length; r++)
                        {
                            if (inPlace) kept &= slot.Recurrent[l][r].CopyFromInPlace(source.Recurrent[l][r]);
                            else slot.Recurrent[l][r].CopyFrom(source.Recurrent[l][r]);
                        }
            if (!kept) ClearGraphCache();
            // Its device copies hold what the host does now.
            slot.DeviceDirty = false;
            return slot;
        }

        /// <summary>
        /// Copy a slot whose newest bytes a graph left in device memory without bringing them to the host: one small
        /// program per stage copies each attention layer's rows [0, length) and each state's graph-bound tensors
        /// between the two slots' device copies, and the target is left device-current. Syncing the source to the
        /// host instead read back every cache at its whole capacity (134 ms for a 55-token prompt's checkpoint on
        /// CUDA). False (nothing copied) where the copy goes through the host: a host-current source, an executor
        /// that binds no host memory, a device that reads the host bytes themselves (Metal's shared buffers, the CPU:
        /// a memcpy there took 4.4 ms for Qwen3.6-35B-A3B's 54-token checkpoint on an M5 Pro, a copy program binding
        /// the new slot's 40 buffers 8.7 ms), tensor-parallel caches, quantized rows or a state graphs do not bind.
        /// </summary>
        private bool TryCopySlotOnDevice(Slot source, Slot target)
        {
            if (!source.DeviceDirty || source.Length <= 0 || _graphTpRanks != null || _mesh.TensorParallel != 1
                || GraphExecutor is not { Capabilities.BindsHostPointers: true } executor
                || !(executor.Capabilities.SeparateDeviceCopies || ForceDeviceSlotCopies))
                return false;
            int kvType = KvGgmlType(_kvCacheDtype.ToDType());
            if (kvType != GraphFormat.F32 && kvType != GraphFormat.F16) return false;
            if (_anyRecurrent)
                foreach (RecurrentState[] layer in source.Recurrent)
                    if (layer != null && (layer[0]?.GraphTensors == null))
                        return false;

            int rows = source.Length, d = _keyDim, dv = _valueDim, kvHeads = _localKvHeads;
            long kRow = RowBytes(kvType, d), vRow = RowBytes(kvType, dv);
            for (int stage = 0; stage < _mesh.Stages; stage++)
            {
                var g = new GraphProgramBuilder(record: true);
                int copies = 0;
                for (int l = 0; l < TotalLayers; l++)
                {
                    if (!OwnsLayer(l) || StageOfLayer(l) != stage) continue;
                    if (source.K?[l] != null)
                    {
                        CopyRows(source.K[l][0], target.K[l][0], d, kRow);
                        CopyRows(source.V[l][0], target.V[l][0], dv, vRow);
                    }
                    if (_anyRecurrent && source.Recurrent[l] != null)
                    {
                        Tensor[] from = source.Recurrent[l][0].GraphTensors, to = target.Recurrent[l][0].GraphTensors;
                        for (int i = 0; i < from.Length; i++)
                        {
                            long n = from[i].ElementCount();
                            g.Effect(g.Op(GraphOp.Cpy, Leaf(from[i], GraphFormat.F32, n), Leaf(to[i], GraphFormat.F32, n)));
                            copies++;
                        }
                    }
                }
                if (copies == 0) continue;
                EnterStage(stage);
                try
                {
                    using IGraphPlan plan = executor.Compile(g.ToProgram(replayable: false, reorder: false));
                    if (!plan.Run(ReadOnlySpan<int>.Empty, ReadOnlySpan<IntPtr>.Empty, ReadOnlySpan<long>.Empty))
                        throw new InvalidOperationException($"{Config.Architecture} slot copy failed: {plan.LastError}");
                }
                finally
                {
                    EnterStage(0);
                }

                void CopyRows(Tensor from, Tensor to, int dim, long rowBytes)
                {
                    int src = g.Op(GraphOp.View, stackalloc[] { Leaf(from, kvType, dim, source.Capacity, kvHeads) },
                        stackalloc long[] { 3, dim, rows, kvHeads, 1, rowBytes, rowBytes * source.Capacity, 0, 0 });
                    int dst = g.Op(GraphOp.View, stackalloc[] { Leaf(to, kvType, dim, target.Capacity, kvHeads) },
                        stackalloc long[] { 3, dim, rows, kvHeads, 1, rowBytes, rowBytes * target.Capacity, 0, 0 });
                    g.Effect(g.Op(GraphOp.Cpy, src, dst));
                    copies++;
                }

                int Leaf(Tensor t, int type, params long[] ne)
                    => g.Leaf(GraphLeafRole.State, type, ne, TensorComputePrimitives.GetStoragePointer(t), t.Storage.ByteLength, handle: t);
            }
            if (_anyRecurrent)
                for (int l = 0; l < source.Recurrent.Length; l++)
                    if (source.Recurrent[l] != null)
                        target.Recurrent[l][0].AfterDeviceCopy(source.Recurrent[l][0]);
            target.DeviceDirty = true;
            DeviceSlotCopies++;
            return true;
        }

        /// <param name="inPlace">Write the rows into the targets' device copies rather than dropping them.</param>
        private void CopyRowsOnHost(Tensor[][] from, Tensor[][] to, int rows, bool inPlace)
        {
            if (from == null || rows <= 0) return;
            for (int l = 0; l < from.Length; l++)
            {
                if (from[l] == null) continue;
                for (int r = 0; r < from[l].Length; r++)
                {
                    CopyCacheRows(from[l][r], to[l][r], rows);
                    if (inPlace) UploadCacheRows(to[l][r], rows);
                    else InvalidateTensorDeviceCache(to[l][r]);
                }
            }
        }

        /// <summary>Rows [0, <paramref name="rows"/>) of every head of a cache written on the host, into its device copy.</summary>
        private void UploadCacheRows(Tensor cache, int rows)
        {
            if (!IsGgmlBackend || rows <= 0) return;
            int heads = (int)cache.Sizes[0];
            long capacity = cache.Sizes[1];
            long rowBytes = RowBytes(KvGgmlType(_kvCacheDtype.ToDType()), cache.Sizes[2]);
            var offsets = new long[heads];
            var lengths = new long[heads];
            for (int h = 0; h < heads; h++)
            {
                offsets[h] = h * capacity * rowBytes;
                lengths[h] = rows * rowBytes;
            }
            GgmlBasicOps.UploadHostBufferRanges(TensorComputePrimitives.GetStoragePointer(cache), offsets, lengths);
        }

        /// <summary>Bytes a clone of a retained slot takes: KV rows for <paramref name="targetTokens"/>
        /// plus its recurrent states.</summary>
        private long CloneBytes(Slot slot, int targetTokens)
        {
            long bytes = 0;
            foreach (Tensor t in AllCacheTensors(slot.K).Concat(AllCacheTensors(slot.V)))
                bytes += t.Storage.ByteLength / Math.Max(1, t.Sizes[1]) * Math.Max(targetTokens, slot.Length);
            foreach (Tensor t in CheckpointStateTensors(slot)) bytes += t.ElementCount() * t.ElementType.Size();
            return bytes;
        }

        // ---------------------------------------------------------------- files

        public bool TryExportRetainedCache(string key, Stream destination)
        {
            if (!SupportsRetainedCacheSerialization || _retainedSlots == null || key == null || destination == null) return false;
            if (!_retainedSlots.TryGetValue(key, out Slot slot) || slot.K == null || slot.Length <= 0) return false;
            SyncToHost(slot);
            using var writer = new BinaryWriter(destination, Encoding.UTF8, leaveOpen: true);
            writer.Write(CheckpointMagic);
            writer.Write(CheckpointVersion);
            writer.Write(CheckpointNamespace);
            writer.Write(slot.Length);
            writer.Write(slot.RopeDelta);
            var block = new byte[ComputeBlockBytes(slot, slot.Length)];
            if (!KvBlockTransfer.Extract(_allocator, FlatCaches(slot.K), FlatCaches(slot.V), slot.Length, 0, slot.Length, block))
                return false;
            writer.Write(block.Length);
            writer.Write(block);
            foreach (Tensor t in CheckpointStateTensors(slot))
            {
                byte[] bytes = StateBytes(t);
                writer.Write(bytes.Length);
                writer.Write(bytes);
            }
            return true;
        }

        public bool TryImportRetainedCache(string key, Stream source)
        {
            if (!SupportsRetainedCacheSerialization || string.IsNullOrEmpty(key) || source == null) return false;
            _retainedSlots ??= new Dictionary<string, Slot>(StringComparer.Ordinal);
            if (_retainedSlots.ContainsKey(key)) return false;
            Slot slot = null;
            try
            {
                using var reader = new BinaryReader(source, Encoding.UTF8, leaveOpen: true);
                if (reader.ReadUInt32() != CheckpointMagic || reader.ReadInt32() != CheckpointVersion) return false;
                if (reader.ReadString() != CheckpointNamespace) return false;
                int length = reader.ReadInt32();
                int ropeDelta = reader.ReadInt32();
                if (length <= 0 || length > _maxContextLength) return false;
                int capacity = Math.Min(_maxContextLength, Math.Max(FreshSlotTokens, length));
                slot = new Slot { Capacity = capacity, Length = 0, Recurrent = _anyRecurrent ? NewRecurrentStates() : null };
                (slot.K, slot.V) = AllocateCaches(capacity);
                int blockBytes = reader.ReadInt32();
                if (blockBytes != ComputeBlockBytes(slot, length)) { DisposeSlot(slot); return false; }
                byte[] block = reader.ReadBytes(blockBytes);
                if (block.Length != blockBytes
                    || !KvBlockTransfer.Inject(_allocator, FlatCaches(slot.K), FlatCaches(slot.V), 0, 0, length, block))
                {
                    DisposeSlot(slot);
                    return false;
                }
                foreach (Tensor t in AllCacheTensors(slot.K).Concat(AllCacheTensors(slot.V))) InvalidateTensorDeviceCache(t);
                foreach (Tensor t in CheckpointStateTensors(slot))
                {
                    int bytes = reader.ReadInt32();
                    if (bytes != t.ElementCount() * t.ElementType.Size()) { DisposeSlot(slot); return false; }
                    WriteStateBytes(t, reader.ReadBytes(bytes));
                }
                if (slot.Recurrent != null)
                    foreach (RecurrentState[] layer in slot.Recurrent)
                        if (layer != null)
                            foreach (RecurrentState state in layer) state?.OnHostWritten();
                slot.Length = length;
                slot.RopeDelta = ropeDelta;
                slot.DraftStart = length;   // the file holds no draft-layer rows
                _retainedSlots.Add(key, slot);
                return true;
            }
            catch (EndOfStreamException)
            {
                if (slot != null) DisposeSlot(slot);
                return false;
            }
        }

        private long ComputeBlockBytes(Slot slot, int tokens)
            => KvBlockTransfer.ComputeBlockByteSize(FlatCaches(slot.K), FlatCaches(slot.V), tokens);

        private static IEnumerable<Tensor> CheckpointStateTensors(Slot slot)
        {
            if (slot.Recurrent == null) yield break;
            foreach (RecurrentState[] layer in slot.Recurrent)
                if (layer != null)
                    foreach (RecurrentState state in layer)
                        if (state != null)
                            foreach (Tensor t in state.CheckpointTensors) yield return t;
        }

        private static unsafe byte[] StateBytes(Tensor t)
        {
            var bytes = new byte[t.ElementCount() * t.ElementType.Size()];
            fixed (byte* dst = bytes)
                Buffer.MemoryCopy((void*)TensorComputePrimitives.GetStoragePointer(t), dst, bytes.Length, bytes.Length);
            return bytes;
        }

        private unsafe void WriteStateBytes(Tensor t, byte[] bytes)
        {
            IntPtr p = TensorComputePrimitives.GetStoragePointer(t);
            fixed (byte* src = bytes)
                Buffer.MemoryCopy(src, (void*)p, bytes.Length, bytes.Length);
            InvalidateTensorDeviceCache(t);
            HostWritten(t);
        }
    }
}
