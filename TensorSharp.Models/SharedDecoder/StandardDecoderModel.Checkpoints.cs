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
        private const int CheckpointVersion = 1;

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

        /// <summary>The retained slot's bytes are current on the host (copies read them there).</summary>
        public bool SettleForCopy(string payloadKey)
        {
            if (_retainedSlots == null || payloadKey == null || !_retainedSlots.TryGetValue(payloadKey, out Slot slot)) return false;
            SyncToHost(slot);
            return true;
        }

        /// <summary>A deep copy of <paramref name="source"/> sized for <paramref name="capacityTokens"/>.</summary>
        private Slot CopySlot(Slot source, int capacityTokens)
        {
            SyncToHost(source);
            int capacity = Math.Min(_maxContextLength, Math.Max(FreshSlotTokens, capacityTokens));
            var slot = new Slot { Capacity = capacity, Length = source.Length, Recurrent = _anyRecurrent ? NewRecurrentStates() : null };
            (slot.K, slot.V) = AllocateCaches(capacity);
            CopyRowsOnHost(source.K, slot.K, source.Length);
            CopyRowsOnHost(source.V, slot.V, source.Length);
            if (_anyRecurrent)
                for (int l = 0; l < source.Recurrent.Length; l++)
                    if (source.Recurrent[l] != null)
                        for (int r = 0; r < source.Recurrent[l].Length; r++)
                            slot.Recurrent[l][r].CopyFrom(source.Recurrent[l][r]);
            return slot;
        }

        private void CopyRowsOnHost(Tensor[][] from, Tensor[][] to, int rows)
        {
            if (from == null || rows <= 0) return;
            for (int l = 0; l < from.Length; l++)
            {
                if (from[l] == null) continue;
                for (int r = 0; r < from[l].Length; r++)
                {
                    CopyCacheRows(from[l][r], to[l][r], rows);
                    InvalidateTensorDeviceCache(to[l][r]);
                }
            }
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
