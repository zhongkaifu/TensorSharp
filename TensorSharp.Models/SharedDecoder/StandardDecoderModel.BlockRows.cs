// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.Graph;

namespace TensorSharp.Models
{
    /// <summary>
    /// Block rows (DiffusionGemma's canvas): rows that stand after a sequence's committed rows and attend each other in both
    /// directions on every layer, the cached rows as a prompt chunk at that position does (a sliding-window layer reads the
    /// window of the block's first row), and commit nothing. Their keys and values are written past the committed rows, in
    /// cells the cache's next rows overwrite, so the sequence's committed state is what it was; a sliding-window layer's ring
    /// keeps the window and a block (<see cref="RingStepRows"/> &gt;= the block) where they reach past its first lap.
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        /// <summary>Set while a block-rows program traces (<see cref="GraphStep.BlockRows"/>).</summary>
        private bool _blockRowsStep;

        /// <summary>A sequence's cache is bound as the active one (<see cref="BindSequenceCache"/>), not the model's own.</summary>
        private protected bool SequenceBound => _activeKey != null;

        /// <summary>
        /// A block-rows program over the active cache (<paramref name="requestIds"/> null) or over the bound sequences'
        /// caches (sequence j's <paramref name="counts"/>[j] rows after its committed rows): traced and replayed by signature, so
        /// the steps of one block (same positions, same rows) run one recorded graph. Returns the graph whose outputs hold what
        /// <paramref name="emit"/> asked for (the caller disposes it when <paramref name="owned"/>).
        /// </summary>
        private protected CachedGraph RunBlockRows(IReadOnlyList<string> requestIds, int[] counts, Func<GraphStep, StepOutputs> emit,
            out StepOutputs outputs, out bool owned, string shape)
        {
            if (_rankGraphs || IsMultiNodePipeline || !_mesh.IsSingleDevice)
                throw new NotSupportedException($"{Config.Architecture}: block rows run on one device (no tensor parallelism or layer split).");
            if (GraphExecutor == null)
                throw new NotSupportedException($"{Config.Architecture}: block rows run as graphs, and this backend has no graph executor.");
            SettlePendingVerify();
            int n = requestIds?.Count ?? 1;
            if (counts.Length != n) throw new ArgumentException($"{counts.Length} row counts for {n} sequences.", nameof(counts));
            var slots = new Slot[n];
            var starts = new int[n];
            var first = new int[n];
            int rows = 0;
            if (requestIds == null)
            {
                EnsureCacheCapacity(checked(_cacheSeqLen + counts[0]));
                slots[0] = SnapshotActive();
                starts[0] = _cacheSeqLen;
            }
            else
            {
                RestorePrimaryCache();
                for (int j = 0; j < n; j++)
                {
                    if (!TryGetSlot(requestIds[j], out Slot slot))
                        throw new InvalidOperationException($"{Config.Architecture}: sequence '{requestIds[j]}' has no cache.");
                    EnsureSlotCapacity(slot, checked(slot.Length + counts[j]));
                    slots[j] = slot;
                    starts[j] = slot.Length;
                }
            }
            for (int j = 0; j < n; j++)
            {
                if (counts[j] < 1 || counts[j] > MaxSequenceStepRows)
                    throw new ArgumentOutOfRangeException(nameof(counts), counts[j], $"a block takes 1 to {MaxSequenceStepRows} rows a sequence");
                first[j] = rows;
                rows += counts[j];
            }
            EnterStage(0);
            // The media rows a forward opens its masks for are its own: a prompt forward's never reach block rows.
            _mediaRows = null;
            _blockRowsStep = true;
            try
            {
                return RunTraced(rows, slots, first, counts, starts, replayable: true, snapshots: false, attentionFloor: 0, emit,
                    out outputs, out owned, shape: shape);
            }
            finally
            {
                _blockRowsStep = false;
                if (requestIds == null) _cacheDeviceDirty = true;
            }
        }

        /// <summary>Every decoder layer over the stream <paramref name="h"/> [hidden, rows] of a block-rows step (each row at its
        /// sequence's position), as a step's layers run: the stream after the last layer.</summary>
        private protected int EmitDecoderLayers(GraphStep step, int h)
        {
            var positions = new int[step.Rows];
            for (int s = 0; s < step.Sequences; s++)
                for (int i = 0; i < step.Counts[s]; i++) positions[step.First[s] + i] = step.Starts[s] + i + step.Slots[s].RopeDelta;
            step.SetPositions(positions);
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                h = EmitLayerPrologue(step, layer, h);
                if (HasMixer(layer))
                {
                    int normed = EmitBlockInput(step, layer, h, BlockSite.Mixer, out int mixContext);
                    int mix = _recurrentLayer[layer] ? EmitRecurrentMixer(step, layer, normed) : EmitAttention(step, layer, normed);
                    h = EmitBlockOutput(step, layer, h, BlockSite.Mixer, mixContext, mix);
                }
                if (HasFeedForward(layer))
                {
                    int ffnIn = EmitBlockInput(step, layer, h, BlockSite.Ffn, out int ffnContext);
                    int ffn = EmitFeedForward(step, layer, ffnIn, ffnContext);
                    h = EmitBlockOutput(step, layer, h, BlockSite.Ffn, ffnContext, ffn);
                }
                step.Boundary(h);
            }
            return h;
        }

        /// <summary>The head's logits [vocab, rows] over final-normed rows <paramref name="normed"/>, with the family's logit
        /// transform (<see cref="FinalLogitSoftcap"/>): the tied <c>token_embd</c> unless the model has an <c>output</c>.</summary>
        private protected int EmitHeadLogits(GraphStep step, int normed, int rows)
        {
            string headName = _quantWeights.ContainsKey("output.weight") || _weights.ContainsKey("output.weight") ? "output.weight" : "token_embd.weight";
            return EmitLogitTransform(step, EmitLogits(step, step.MatMul(headName, normed), rows));
        }
    }
}
