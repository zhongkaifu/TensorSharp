// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;

using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    /// <summary>
    /// The radix prefix cache contract, once for every family on the runtime: host-slab pages where
    /// snapshots restore across sequences, the resident primary cache, and each finished request's
    /// slot donated to the turn that continues it (zero copy). A stack with recurrent layers is
    /// class R: its pages do not carry the running state and it cannot rewind, so only a whole
    /// donated slot (which carries the state) continues a conversation; a shared prefix is
    /// otherwise re-prefilled rather than restored wrong. Reuse across an image span stays off
    /// pending image-turn exactness coverage.
    /// </summary>
    public abstract partial class StandardDecoderModel : IHolderPrefixCacheModel, IPrefixCacheModelDiagnostics
    {
        private IPrefixPayloadSink _prefixCacheSink;

        public PrefixCacheCapabilities GetPrefixCacheCapabilities()
        {
            // A slot, a checkpoint and a checkpoint file keep the sequence's M-RoPE delta (Slot.RopeDelta), so a cache
            // that went through an image continues past it at the positions a re-prefill of the history uses; the
            // other families rotate at absolute positions. Either way the reuse is exact (Qwen35ImageFollowUpExactnessTests).
            PrefixCacheCapabilities pages = _anyRecurrent
                ? PageFamilyCapabilities(FamilyClass.R, TruncationKind.None, pagesNeedStateAtEnd: true, reuseAcrossMediaSpan: true)
                : PageFamilyCapabilities(FamilyClass.P, TruncationKind.Any, reuseAcrossMediaSpan: true);
            if (!SupportsRetainedFusedCache) return pages;
            bool copies = SupportsPrefixCheckpoints;
            return pages with
            {
                NamespaceFingerprint = CheckpointNamespace,
                EndState = copies ? EndStateSupport.CopyAndDonate : EndStateSupport.DonateOnly,
                CanCaptureCopy = copies,
                AdoptPrimaryOnDisplacement = true,
                MinRetainTokens = 32,
                RewindCapTokens = _anyRecurrent ? pages.RewindCapTokens : int.MaxValue,
                Persistable = copies && SupportsRetainedCacheSerialization,
            };
        }

        public void AttachPrefixCache(IPrefixPayloadSink sink)
            => _prefixCacheSink = sink ?? throw new ArgumentNullException(nameof(sink));

        public void DetachPrefixCache() => _prefixCacheSink = null;

        public long QuerySpareBytes(ResourceClass cls) => QueryPrefixCacheSpareBytes(cls);

        // ---------------------------------------------------------------- end states

        public bool TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
        {
            footprint = default;
            return SupportsRetainedFusedCache
                && HolderPrefixCacheAdapter.TryConvertPrimary(this, payloadKey, length, out footprint);
        }

        public bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens)
        {
            if (!TryGetRetained(payloadKey, out Slot slot)) return false;
            if (slot.Length != payloadTokens || targetTokens < 0 || targetTokens > payloadTokens) return false;
            return targetTokens == payloadTokens || CanTruncateKVCache(payloadTokens, targetTokens);
        }


        public PayloadFootprint MeasureEndState(string payloadKey)
        {
            if (!TryGetRetained(payloadKey, out Slot slot)) return default;
            long bytes = RecurrentStateBytes(slot);
            foreach (Tensor t in AllCacheTensors(slot.K)) bytes = checked(bytes + t.Storage.ByteLength);
            foreach (Tensor t in AllCacheTensors(slot.V)) bytes = checked(bytes + t.Storage.ByteLength);
            bool host = _backend is BackendType.Cpu or BackendType.GgmlCpu;
            var vector = new ResourceVector { HostKv = host ? bytes : 0, DeviceKv = host ? 0 : bytes };
            return new PayloadFootprint(slot.Length, slot.Capacity, vector, PositionDelta: 0);
        }

        public ResourceVector EstimateCloneBytes(string payloadKey, int targetTokens)
        {
            if (!TryGetRetained(payloadKey, out Slot slot)) return default;
            long bytes = CloneBytes(slot, targetTokens);
            bool host = _backend is BackendType.Cpu or BackendType.GgmlCpu;
            return new ResourceVector { HostKv = host ? bytes : 0, DeviceKv = host ? 0 : bytes };
        }

        /// <summary>A slot's recurrent states, which the prefix cache's budget charges with its caches.</summary>
        private static long RecurrentStateBytes(Slot slot)
        {
            long bytes = 0;
            if (slot.Recurrent != null)
                foreach (RecurrentState[] layer in slot.Recurrent)
                    if (layer != null)
                        foreach (RecurrentState state in layer)
                            if (state != null) bytes = checked(bytes + state.StateBytes);
            return bytes;
        }

        /// <summary>A slot entering the retained set no longer speculates: its states give up their verify buffers.</summary>
        private static Slot Retained(Slot slot)
        {
            if (slot?.Recurrent != null)
                foreach (RecurrentState[] layer in slot.Recurrent)
                    if (layer != null)
                        foreach (RecurrentState state in layer) state?.ReleaseSpecBuffers();
            return slot;
        }

        private bool TryGetRetained(string payloadKey, out Slot slot)
        {
            slot = null;
            return payloadKey != null && _retainedSlots != null
                && _retainedSlots.TryGetValue(payloadKey, out slot) && slot.K != null;
        }

        // ---------------------------------------------------------------- diagnostics

        public IReadOnlyCollection<string> RetainedPayloadKeys =>
            _retainedSlots == null ? Array.Empty<string>() : _retainedSlots.Keys.ToArray();

        public int PrivateHolderCount => _slots?.Count ?? 0;

        public int PrimaryCacheLength => _activeKey == null ? _cacheSeqLen : (_primary?.Length ?? 0);
    }
}
