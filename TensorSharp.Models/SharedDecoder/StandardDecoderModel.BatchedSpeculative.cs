// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models
{
    /// <summary>
    /// Speculation inside the unified step (<see cref="IBatchedSpeculativeTarget"/>): every speculating request drafts
    /// over its own drafter state (its slot's draft-layer rows, its own DFlash ring), and one forward verifies every
    /// request's window on its own slot, each row's logits and drafter features coming back. A hybrid stack keeps the
    /// recurrent state after each row of a window, and the commit sets each slot's state to the one after the rows it
    /// kept, as a lone request's verify does.
    /// </summary>
    public abstract partial class StandardDecoderModel : IBatchedSpeculativeTarget
    {
        /// <summary>Batched verifies served (tests check the step used them).</summary>
        internal int BatchedSpecVerifySteps { get; private set; }

        /// <summary>The widest window a verify takes per sequence: the rows a step writes by index and replays
        /// (<see cref="SmallChunkRows"/>); the executor narrows it to the drafter's block.</summary>
        public int BatchedSpecWindow(int sequences)
            => sequences < 1 || !SupportsUnifiedBatching || SpeculationRefusal != null || GraphUnsupportedReason() != null
                ? 0 : SmallChunkRows;

        /// <summary>Verify graphs replay per width, so a window is never padded past the longest draft.</summary>
        public bool BatchedSpecExactWindow => true;

        public int FusedCacheSeqLen(string requestId)
        {
            if (string.IsNullOrEmpty(requestId)) return -1;
            if (string.Equals(_activeKey, requestId, StringComparison.Ordinal)) return _cacheSeqLen;
            return _slots != null && _slots.TryGetValue(requestId, out Slot slot) && slot.K != null ? slot.Length : -1;
        }

        /// <summary>Rows past the committed ones up to the context limit: a verify grows the slot when it needs to.</summary>
        public int FusedCacheFreeRows(string requestId)
        {
            int length = FusedCacheSeqLen(requestId);
            return length < 0 ? 0 : Math.Max(0, _maxContextLength - length);
        }

        public IDisposable BeginBatchedSpeculation() => new BatchedSpeculationScope(this);

        private sealed class BatchedSpeculationScope : IDisposable
        {
            private StandardDecoderModel _model;
            public BatchedSpeculationScope(StandardDecoderModel model) => _model = model;
            public void Dispose()
            {
                _model?.RestorePrimaryCache();
                _model = null;
            }
        }

        /// <summary>The request's drafter state: its slot is bound (a draft layer's rows live there) and its DFlash
        /// ring selected.</summary>
        public void BindDraftState(string requestId)
        {
            BindDFlashRing(requestId);
            if (requestId != null && _slots != null && _slots.ContainsKey(requestId))
                BindSequenceCache(requestId);
        }

        public void ReleaseDraftState(string requestId) => ReleaseDFlashRing(requestId);

        public bool SupportsBatchedSpecLogits => true;

        public bool TryBatchedSpecVerify(IReadOnlyList<string> requestIds, int[] positions, int window,
            int[] tokens, int[] drawn, float[] features)
        {
            int vocab = Config.VocabSize;
            long needed = (long)requestIds.Count * window * vocab;
            if (_verifyLogits == null || _verifyLogits.LongLength < needed) _verifyLogits = new float[needed];
            if (!TryVerifySlots(requestIds, positions, window, tokens, _verifyLogits, features)) return false;
            for (int row = 0; row < requestIds.Count * window; row++)
            {
                var span = new ReadOnlySpan<float>(_verifyLogits, row * vocab, vocab);
                int best = 0;
                for (int v = 1; v < vocab; v++)
                    if (span[v] > span[best]) best = v;
                drawn[row] = best;
            }
            return true;
        }

        private float[] _verifyLogits;

        public bool TryBatchedSpecVerifyLogits(IReadOnlyList<string> requestIds, int[] positions, int window,
            int[] tokens, float[] logits, float[] features)
            => TryVerifySlots(requestIds, positions, window, tokens, logits, features);

        /// <summary>One forward of every sequence's window on its own slot: all rows' logits, the drafter's feature
        /// rows, and on a hybrid stack the recurrent state after each row (settled by the commit).</summary>
        private bool TryVerifySlots(IReadOnlyList<string> requestIds, int[] positions, int window, int[] tokens,
            float[] logitsOut, float[] featuresOut)
        {
            int n = requestIds.Count;
            if (n == 0 || window < 1 || window > SmallChunkRows || tokens.Length < n * window || positions.Length != n
                || _slots == null || HasPendingMedia || _pendingPositionTable != null || GraphUnsupportedReason() != null)
                return false;
            RestorePrimaryCache();
            var slots = new Slot[n];
            var first = new int[n];
            var counts = new int[n];
            for (int s = 0; s < n; s++)
            {
                if (!_slots.TryGetValue(requestIds[s], out slots[s]) || slots[s].K == null
                    || slots[s].Length != positions[s] || slots[s].VerifyRows != 0)
                    return false;
                for (int o = 0; o < s; o++)
                    if (ReferenceEquals(slots[o], slots[s])) return false;
                first[s] = s * window;
                counts[s] = window;
            }
            for (int s = 0; s < n; s++)
                EnsureSlotCapacity(slots[s], checked(positions[s] + window));

            bool dflash = HasDFlash;
            var request = new GraphRequest
            {
                LastRowLogits = false,
                AllRowLogits = true,
                CaptureLayers = dflash && featuresOut != null ? DFlashSettings.TargetLayerIds : Array.Empty<int>(),
                CaptureFinal = !dflash && featuresOut != null,
                RecurrentSnapshots = _anyRecurrent && window > 1,
            };
            int rows = n * window;
            var flat = new int[rows];
            Array.Copy(tokens, flat, rows);
            Tensor hidden = EmbedStep(flat);
            GraphResult result;
            _forwardSw.Start();
            try
            {
                if (!TryRunGraphStep(hidden, slots, first, counts, (int[])positions.Clone(), request, out result))
                    return false;
            }
            finally
            {
                hidden.Dispose();
                _forwardSw.Stop();
            }
            for (int s = 0; s < n; s++)
            {
                slots[s].Length = positions[s] + window;
                slots[s].VerifyRows = request.RecurrentSnapshots ? window : 0;
            }
            Array.Copy(result.Logits, 0, logitsOut, 0, Math.Min((long)rows * Config.VocabSize, logitsOut.LongLength));
            WriteSpecFeatures(featuresOut, rows, result.Captures, result.Final);
            BatchedSpecVerifySteps++;
            return true;
        }

        /// <summary>A unified step whose prompt chunks feed a prompt-conditioned drafter: each sequence's last-row
        /// logits and, for each sequence that asked (<paramref name="features"/>[s] not null), its rows' drafter
        /// features, row-major [rows][feature width].</summary>
        private bool TryRunCapturingGraph(Tensor hidden, Slot[] slots, int[] first, int[] counts, int[] starts,
            float[][] outLogits, float[][] features)
        {
            bool dflash = HasDFlash;
            int feat0 = SpecFeatureSize;
            // One sequence: the graph returns only the rows its drafter needs (features[0] holds that many).
            int tail = slots.Length == 1 && dflash && features[0] != null ? (int)Math.Min(counts[0], features[0].LongLength / feat0) : 0;
            var request = new GraphRequest
            {
                LastRowLogits = true,
                CaptureLayers = dflash ? DFlashSettings.TargetLayerIds : Array.Empty<int>(),
                CaptureFinal = !dflash,
                CaptureTail = tail,
            };
            if (!TryRunGraphStep(hidden, slots, first, counts, (int[])starts.Clone(), request, out GraphResult result))
                return false;
            int n = slots.Length, vocab = Config.VocabSize, hiddenSize = Config.HiddenSize, feat = SpecFeatureSize;
            for (int s = 0; s < n; s++)
            {
                if (n == 1)
                {
                    outLogits[0] = result.Logits;
                }
                else
                {
                    outLogits[s] = new float[vocab];
                    Array.Copy(result.Logits, (long)s * vocab, outLogits[s], 0, vocab);
                }
                float[] rows = features[s];
                if (rows == null) continue;
                // The sequence's last `wanted` rows (the rows the drafter's context needs).
                int wanted = (int)Math.Min(counts[s], rows.LongLength / feat);
                if (wanted < 1)
                    throw new ArgumentException($"A prompt chunk's features need at least {feat} floats.", nameof(features));
                for (int r = 0; r < wanted; r++)
                {
                    long row = tail > 0 ? r : first[s] + counts[s] - wanted + r;
                    if (dflash)
                        for (int c = 0; c < result.Captures.Length; c++)
                            Array.Copy(result.Captures[c], row * hiddenSize, rows, (long)r * feat + (long)c * hiddenSize, hiddenSize);
                    else
                        Array.Copy(result.Final, row * feat, rows, (long)r * feat, feat);
                }
            }
            return true;
        }

        /// <summary>Each sequence keeps <c>kept[i]</c> rows of its window: its slot's length, and on a hybrid stack the
        /// recurrent state after those rows.</summary>
        public void CommitBatchedSpecVerify(IReadOnlyList<string> requestIds, int[] positions, int[] kept)
        {
            RestorePrimaryCache();
            try
            {
                for (int i = 0; i < requestIds.Count; i++)
                {
                    if (!_slots.TryGetValue(requestIds[i], out Slot slot))
                        throw new InvalidOperationException($"{requestIds[i]} has no slot to commit.");
                    int rows = slot.VerifyRows;
                    if (kept[i] < 1 || (rows > 0 && kept[i] > rows) || slot.Length < positions[i] + kept[i])
                        throw new ArgumentOutOfRangeException(nameof(kept), $"{requestIds[i]} kept {kept[i]} rows of its window.");
                    slot.Length = positions[i] + kept[i];
                    if (rows <= 1) continue;
                    slot.VerifyRows = 0;
                    BindSequenceCache(requestIds[i]);   // the commit sets the active request's states
                    CommitRecurrentPrefix(kept[i], rows);
                }
            }
            finally
            {
                RestorePrimaryCache();
            }
        }
    }
}
