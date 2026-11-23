// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

namespace TensorSharp.Models
{
    /// <summary>
    /// A family's own block drafter (Inkling 2's DSpark): it drafts a block from the anchor token, its position and the
    /// features of the last committed row, and keeps its context from the features the trunk captures at
    /// <see cref="CaptureLayers"/> (each the stream entering that layer as the family's EmitCaptureRow gives it; a layer of
    /// Config.NumLayers is the head's input). The scheduler's block speculation (BlockDraftSpeculator, the verify windows,
    /// the batched verify) runs it through the model's IDraftHead members, as it runs a DFlash drafter.
    /// </summary>
    internal abstract class FamilyBlockDrafter : IDisposable
    {
        /// <summary>Tokens a block drafts.</summary>
        public abstract int DraftTokens { get; }

        /// <summary>The trunk layers whose entry rows make a row's features, in feature order.</summary>
        public abstract int[] CaptureLayers { get; }

        /// <summary>The prompt chunk a speculating request's prefill takes at most (its features come back to the host).</summary>
        public abstract int PrefillChunkCap { get; }

        /// <summary>Draft from <paramref name="lastToken"/> at <paramref name="position"/>; <paramref name="hPrev"/> is the
        /// features of the row before it. Returns the drafts written.</summary>
        public abstract int DraftBlock(int lastToken, float[] hPrev, int position, int[] draftOut, float[] confOut);

        /// <summary>The committed rows' features (row-major) from <paramref name="startPos"/> on go into the context.</summary>
        public abstract void CatchUp(int[] tokens, float[] hRows, int startPos);

        /// <summary>Drafts a block of <see cref="DraftBatch"/> fills per request (0: the drafter drafts one request at a time).</summary>
        public virtual int BatchDraftWidth => 0;

        /// <summary><see cref="DraftBatch"/> drafts only as many as its blocks hold, when they are narrower than
        /// <see cref="BatchDraftWidth"/>.</summary>
        public virtual bool DraftsNarrowBatches => false;

        /// <summary>One block for each of <paramref name="requestIds"/> in one program (IBatchedSpeculativeTarget.TryDraftBatch):
        /// lane i's anchor at <paramref name="positions"/>[i], <paramref name="carries"/>[i] the features of the row before it.
        /// False, with nothing drafted, when it does not run; the caller then drafts each request on its own.</summary>
        public virtual bool DraftBatch(IReadOnlyList<string> requestIds, int[] anchors, float[][] carries, int[] positions,
            int[][] tokensOut, float[][] confOut) => false;

        /// <summary>The request whose context the next calls use.</summary>
        public abstract void Bind(string requestId);

        /// <summary>The request ended: its context goes.</summary>
        public abstract void Release(string requestId);

        public virtual void Dispose() { }
    }

    public abstract partial class StandardDecoderModel
    {
        /// <summary>The family's own block drafter, or null (<see cref="FamilyBlockDrafter"/>).</summary>
        internal FamilyBlockDrafter FamilyDrafter { get; private protected set; }

        /// <summary>The trunk's steps capture block-drafter features: a DFlash drafter's or a family drafter's.</summary>
        private bool HasBlockFeatures => HasDFlash || FamilyDrafter != null;

        private int[] BlockFeatureLayers => HasDFlash ? DFlashSettings.TargetLayerIds : FamilyDrafter?.CaptureLayers ?? Array.Empty<int>();

        private int BlockFeatureSize => BlockFeatureLayers.Length * Config.HiddenSize;

        private Dictionary<Tensor[], Slot> _familyDraftSlots;

        /// <summary>
        /// A family drafter's program over its own caches (<paramref name="cacheK"/> / <paramref name="cacheV"/> one tensor a
        /// draft layer, <paramref name="capacity"/> rows): bound as its step's sequence caches, traced and replayed by
        /// signature as the DFlash drafter's programs are. <paramref name="rows"/> rows at positions from
        /// <paramref name="start"/>.
        /// </summary>
        private protected CachedGraph RunFamilyDraftProgram(Tensor[] cacheK, Tensor[] cacheV, int capacity, int rows, int start,
            Func<GraphStep, StepOutputs> emit, out StepOutputs outputs, out bool owned, string shape)
        {
            _familyDraftSlots ??= new Dictionary<Tensor[], Slot>(ReferenceEqualityComparer.Instance);
            if (!_familyDraftSlots.TryGetValue(cacheK, out Slot slot) || slot.Capacity != capacity || !SameTensors(slot, cacheK, cacheV))
            {
                if (slot != null) ForgetGraphsOf(slot);
                var k = new Tensor[cacheK.Length][];
                var v = new Tensor[cacheV.Length][];
                for (int l = 0; l < k.Length; l++)
                {
                    k[l] = new[] { cacheK[l] };
                    v[l] = new[] { cacheV[l] };
                }
                slot = new Slot { K = k, V = v, Capacity = capacity, DraftRing = true };
                _familyDraftSlots[cacheK] = slot;
            }
            return RunTraced(rows, new[] { slot }, new[] { 0 }, new[] { rows }, new[] { start }, replayable: rows <= SmallChunkRows,
                snapshots: false, attentionFloor: 0, emit, out outputs, out owned, shape: DraftShapePrefix + shape);
        }

        /// <summary>
        /// A family drafter's program over several requests' caches at once (one lane a request: <paramref name="cacheK"/>[j] /
        /// <paramref name="cacheV"/>[j] its caches, <paramref name="capacities"/>[j] their rows), each lane <paramref name="rows"/>
        /// rows at positions from <paramref name="starts"/>[j], bound as the step's sequences and replayed by signature.
        /// </summary>
        private protected CachedGraph RunFamilyDraftLanes(Tensor[][] cacheK, Tensor[][] cacheV, int[] capacities, int rows, int[] starts,
            Func<GraphStep, StepOutputs> emit, out StepOutputs outputs, out bool owned, string shape)
        {
            _familyDraftSlots ??= new Dictionary<Tensor[], Slot>(ReferenceEqualityComparer.Instance);
            int m = cacheK.Length;
            var slots = new Slot[m];
            var first = new int[m];
            var counts = new int[m];
            for (int j = 0; j < m; j++)
            {
                if (!_familyDraftSlots.TryGetValue(cacheK[j], out Slot slot) || slot.Capacity != capacities[j] || !SameTensors(slot, cacheK[j], cacheV[j]))
                {
                    if (slot != null) ForgetGraphsOf(slot);
                    var k = new Tensor[cacheK[j].Length][];
                    var v = new Tensor[cacheV[j].Length][];
                    for (int l = 0; l < k.Length; l++)
                    {
                        k[l] = new[] { cacheK[j][l] };
                        v[l] = new[] { cacheV[j][l] };
                    }
                    slot = new Slot { K = k, V = v, Capacity = capacities[j], DraftRing = true };
                    _familyDraftSlots[cacheK[j]] = slot;
                }
                slots[j] = slot;
                first[j] = j * rows;
                counts[j] = rows;
            }
            return RunTraced(m * rows, slots, first, counts, (int[])starts.Clone(), replayable: rows <= SmallChunkRows,
                snapshots: false, attentionFloor: 0, emit, out outputs, out owned, shape: DraftShapePrefix + shape);
        }

        private static bool SameTensors(Slot slot, Tensor[] cacheK, Tensor[] cacheV)
        {
            for (int l = 0; l < cacheK.Length; l++)
                if (!ReferenceEquals(slot.K[l][0], cacheK[l]) || !ReferenceEquals(slot.V[l][0], cacheV[l])) return false;
            return true;
        }

        /// <summary>A family drafter's caches go: the graphs over them with them.</summary>
        internal void ForgetFamilyDraftCaches(Tensor[] cacheK)
        {
            if (_familyDraftSlots != null && _familyDraftSlots.Remove(cacheK, out Slot slot)) ForgetGraphsOf(slot);
        }
    }
}
