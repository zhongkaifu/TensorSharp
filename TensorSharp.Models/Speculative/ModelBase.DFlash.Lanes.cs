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

namespace TensorSharp.Models
{
    /// <summary>
    /// The DFlash rings of several requests drafting in one step (a batched speculative step: one verify forward over every
    /// request's window). Each request keeps its own ring (ModelBase.DFlash.Rings.cs); what this adds is the rings' work done
    /// for every request at once instead of one program a request:
    ///
    ///   the block   each request's block, carried row first, drafted in one program over the requests' rings
    ///               (<see cref="PrepareDFlashLaneRings"/> hands a model that runs one the rings and their bookkeeping);
    ///   the commits each request's catch-up (its kept tokens' feature rows) queued on its ring while the step runs
    ///               (<see cref="BeginDFlashInjectDeferral"/>) and written for every ring at once when it ends
    ///               (<see cref="TryInjectDFlashLanes"/>).
    ///
    /// A queued row is bookkept as written when it is queued (frontier, valid run), so the queue changes no draft's mask;
    /// anything that reads or writes a ring's rows on the device writes its queue first, in order, and a ring that changes
    /// hands drops it.
    /// </summary>
    public abstract partial class ModelBase
    {
        /// <summary>One request's ring in a program over several rings: its tensors per draft layer and, for a block, which
        /// position each slot holds (the mask's bookkeeping, taken after the request's carried row was noted).</summary>
        private protected readonly record struct DFlashLaneRing(Tensor[] K, Tensor[] V, int[] SlotPositions);

        /// <summary>One ring's queued rows in an inject over several rings: its tensors, its feature rows
        /// [count, feature size] and the position of the first.</summary>
        private protected readonly record struct DFlashLaneRows(Tensor[] K, Tensor[] V, float[] Rows, int Start, int Count);

        private int _dflashDeferDepth;

        /// <summary>Rows an inject may wait with: a commit's catch-up (its accepted drafts and their anchor) or a carried row.
        /// A prompt chunk's rows are written at once.</summary>
        private int DFlashDeferredRowsLimit => _dflash?.BlockSize ?? 0;

        /// <summary>From here until the matching <see cref="EndDFlashInjectDeferral"/>, a small inject into a ring (a commit's
        /// catch-up) waits on the ring; the end writes every ring's in one program where the model runs one.</summary>
        private protected void BeginDFlashInjectDeferral() => _dflashDeferDepth++;

        private protected void EndDFlashInjectDeferral()
        {
            if (_dflashDeferDepth <= 0 || --_dflashDeferDepth > 0)
                return;
            FlushAllPendingDFlashRows();
        }

        /// <summary>Rings whose queued rows a model's inject program over several rings wrote (tests check it ran).</summary>
        internal int DFlashLaneInjects { get; private set; }

        /// <summary>
        /// Queue <paramref name="n"/> feature rows at <paramref name="startPos"/> on the bound ring instead of writing them,
        /// while a batched step runs (<see cref="BeginDFlashInjectDeferral"/>). False (nothing queued) for rows the step does
        /// not hold back. The ring's queue stays one run of consecutive positions: rows that do not continue it are written
        /// after the queue.
        /// </summary>
        private bool TryDeferDFlashInject(float[] hRows, int rowOffset, int n, int startPos)
        {
            DFlashRing ring = _dflashActiveRing;
            int capacity = DFlashDeferredRowsLimit + 1;
            if (_dflashDeferDepth == 0 || ring == null || n <= 0 || n > DFlashDeferredRowsLimit)
                return false;
            int feat = _dflash.FeatureSize;
            if (hRows == null || hRows.LongLength < (long)(rowOffset + n) * feat)
                throw new ArgumentException($"DFlash encoder needs {(long)(rowOffset + n) * feat} feature floats but got {hRows?.LongLength ?? 0}.", nameof(hRows));
            if (ring.PendingCount > 0 && (ring.PendingStart + ring.PendingCount != startPos || ring.PendingCount + n > capacity))
                FlushPendingDFlashRows(ring);
            // One buffer a ring, kept: a 27B DFlash2 catch-up's rows are 0.9 MB.
            if (ring.PendingRows == null || ring.PendingRows.LongLength < (long)capacity * feat)
                ring.PendingRows = new float[(long)capacity * feat];
            if (ring.PendingCount == 0)
                ring.PendingStart = startPos;
            Array.Copy(hRows, (long)rowOffset * feat, ring.PendingRows, (long)ring.PendingCount * feat, (long)n * feat);
            ring.PendingCount += n;
            // The rows are bookkept as written (DFlashInjectKv's frontier and valid run): a draft's mask reads them as context,
            // and the queue is written before anything reads the ring.
            NoteDFlashRingWrite(startPos, n);
            _dflashRingHostStale = true;
            return true;
        }

        /// <summary>
        /// Write <paramref name="ring"/>'s queued rows into it through the single-ring inject (the graph program, the fused
        /// graph or the per-op pair). Their bookkeeping was done as they were queued: the write changes no frontier.
        /// </summary>
        private void FlushPendingDFlashRows(DFlashRing ring)
        {
            if (ring == null || ring.PendingCount == 0)
                return;
            int count = ring.PendingCount, start = ring.PendingStart;
            ring.PendingCount = 0;
            DFlashRing previous = _dflashActiveRing;
            SelectDFlashRing(ring);
            var kept = (_dflashRingFilled, _dflashRingValidFrom, _dflashRingHigh, _dflashRingCarryEnd);
            int depth = _dflashDeferDepth;
            _dflashDeferDepth = 0;
            try
            {
                WriteDFlashRows(ring.PendingRows, 0, count, start);
            }
            finally
            {
                _dflashDeferDepth = depth;
                (_dflashRingFilled, _dflashRingValidFrom, _dflashRingHigh, _dflashRingCarryEnd) = kept;
                _dflashRingHostStale = true;
                SelectDFlashRing(previous);
            }
        }

        /// <summary>The bound ring's queued rows, written (<see cref="FlushPendingDFlashRows"/>).</summary>
        private void FlushBoundDFlashRows()
        {
            if (_dflashActiveRing is { PendingCount: > 0 })
                FlushPendingDFlashRows(_dflashActiveRing);
        }

        /// <summary>Every ring's queued rows: in one program over the rings where the model runs one and two or more hold
        /// rows (<see cref="TryInjectDFlashLanes"/>), else ring by ring.</summary>
        private void FlushAllPendingDFlashRows()
        {
            List<DFlashRing> queued = null;
            foreach (DFlashRing ring in AllDFlashRings())
                if (ring.PendingCount > 0)
                    (queued ??= new List<DFlashRing>()).Add(ring);
            if (queued == null)
                return;
            if (queued.Count >= 2)
            {
                var lanes = new DFlashLaneRows[queued.Count];
                for (int j = 0; j < queued.Count; j++)
                {
                    DFlashRing ring = queued[j];
                    // The bound ring's tensors are its object's too (only its bookkeeping lives in the fields).
                    lanes[j] = new DFlashLaneRows(ring.K, ring.V, ring.PendingRows, ring.PendingStart, ring.PendingCount);
                }
                if (TryInjectDFlashLanes(lanes))
                {
                    foreach (DFlashRing ring in queued)
                        ring.PendingCount = 0;
                    DFlashLaneInjects += queued.Count;
                    return;
                }
            }
            foreach (DFlashRing ring in queued)
                FlushPendingDFlashRows(ring);
        }

        /// <summary>
        /// Write several rings' rows in one program (each lane's rows at its positions into its own ring; their bookkeeping
        /// was done when they were queued). False when the model does not run one (nothing written): the rings are then
        /// written one at a time.
        /// </summary>
        private protected virtual bool TryInjectDFlashLanes(DFlashLaneRows[] lanes) => false;

        /// <summary>
        /// The rings of <paramref name="requestIds"/> for a block drafted for every one of them in one program: each request's
        /// ring (assigned as <see cref="BindDFlashRing"/> assigns it), its queued rows written, its carried row (the features
        /// of position <paramref name="positions"/>[i] - 1, which the program writes with the block) bookkept as written, and
        /// which position each of its slots holds then. Null, with no row bookkept, when a request has no ring of its own (two
        /// share one), drafts no more (a failed inject) or has no position before its anchor. The bound ring stays bound.
        /// </summary>
        private protected DFlashLaneRing[] PrepareDFlashLaneRings(IReadOnlyList<string> requestIds, int[] positions)
        {
            int m = requestIds.Count;
            if (!_hasDFlash || _dflashRingK == null || m < 1 || positions.Length < m)
                return null;
            EnsureDFlashRingObjects();
            DFlashRing previous = _dflashActiveRing;
            var rings = new DFlashRing[m];
            try
            {
                for (int j = 0; j < m; j++)
                {
                    if (requestIds[j] == null || positions[j] <= 0 || !TryBindDFlashRing(requestIds[j], mayTake: true) || DFlashDraftingDeclined)
                        return null;
                    rings[j] = _dflashActiveRing;
                    for (int o = 0; o < j; o++)
                        if (ReferenceEquals(rings[o], rings[j]))
                            return null;
                }
                var lanes = new DFlashLaneRing[m];
                for (int j = 0; j < m; j++)
                {
                    SelectDFlashRing(rings[j]);
                    FlushBoundDFlashRows();
                    NoteDFlashRingWrite(positions[j] - 1, 1);
                    _dflashRingHostStale = true;
                    lanes[j] = new DFlashLaneRing(_dflashRingK, _dflashRingV, (int[])BuildRingSlotPositions().Clone());
                }
                return lanes;
            }
            finally
            {
                SelectDFlashRing(previous);
            }
        }
    }
}
