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
    /// Hybrid stacks: layers whose token mixer is a recurrent block (linear attention, an SSM)
    /// instead of attention. The family supplies the block and the shape of its state; the
    /// runtime owns where that state lives - one per (layer, rank) for the single sequence of
    /// <c>Forward</c>, one per (sequence, layer, rank) for continuous batching - and therefore
    /// every placement (tensor parallelism, layer split, hybrid) and batching work for recurrent
    /// layers exactly as for attention ones.
    ///
    /// A recurrent state summarizes every token so far and cannot be rewound, so a stack with
    /// recurrent layers truncates only to its current length or to zero, and offers no K/V
    /// snapshots (the engine re-prefills a shared prefix instead of restoring it).
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        /// <summary>One recurrent layer's state on one rank for one sequence (family defined).</summary>
        protected internal abstract class RecurrentState : IDisposable
        {
            /// <summary>Back to the state before any token.</summary>
            public abstract void Reset();

            public virtual void Dispose() { }
        }

        private bool[] _recurrentLayer;
        private bool _anyRecurrent;
        private RecurrentState[][] _recurrentPrimary;   // [layer][rank]
        private readonly Dictionary<string, RecurrentState[][]> _recurrentBySequence = new(StringComparer.Ordinal);
        private readonly object _recurrentLock = new();

        // ---------------------------------------------------------------- family hooks

        /// <summary>True for a layer whose token mixer is a recurrent block. Default: none.</summary>
        protected virtual bool IsRecurrentLayer(int layer) => false;

        /// <summary>A fresh (reset) state for <paramref name="layer"/> on <paramref name="scope"/>'s rank.</summary>
        protected virtual RecurrentState CreateRecurrentState(RankScope scope, int layer)
            => throw new NotSupportedException($"{Config.Architecture} declares recurrent layers but no recurrent state.");

        /// <summary>
        /// The recurrent mixer for one rank and one sequence: <paramref name="normed"/> is the
        /// replicated, normalized input [tokens, hidden]; the result is this rank's partial
        /// output [tokens, hidden] (the output projection is row-parallel, like attention's).
        /// Advances <paramref name="state"/> by the tokens.
        /// </summary>
        protected virtual Tensor RecurrentMixer(RankScope scope, Tensor normed, int layer, RecurrentState state)
            => throw new NotSupportedException($"{Config.Architecture} declares recurrent layers but no recurrent mixer.");

        /// <summary>True when the recurrent mixer applies the layer's input norm itself (a fused
        /// native block): it then receives the raw residual instead of the normalized input.</summary>
        protected virtual bool RecurrentMixerTakesRawInput => false;

        /// <summary>Shard a recurrent layer's weights for the stage group bound by the runtime
        /// (ModelBase sharding helpers: ShardOutputRows, ShardInputColumnGroups, ...).</summary>
        protected virtual void ShardRecurrentLayer(int layer)
            => throw new NotSupportedException($"{Config.Architecture} declares recurrent layers but cannot shard them.");

        /// <summary>Whether the stack has recurrent layers (known after InitializeDecoder).</summary>
        protected bool HasRecurrentLayers => _anyRecurrent;

        // ---------------------------------------------------------------- state ownership

        private void DetectRecurrentLayers()
        {
            _recurrentLayer = new bool[Config.NumLayers];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                _recurrentLayer[l] = IsRecurrentLayer(l);
                _anyRecurrent |= _recurrentLayer[l];
            }
        }

        private void InitRecurrentStates()
        {
            if (!_anyRecurrent) return;
            _recurrentPrimary = NewRecurrentStates();
        }

        private RecurrentState[][] NewRecurrentStates()
        {
            var states = new RecurrentState[Config.NumLayers][];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_recurrentLayer[l] || !OwnsLayer(l)) continue;
                int stage = _mesh.StageOf(l);
                states[l] = new RecurrentState[_mesh.TensorParallel];
                for (int r = 0; r < _mesh.TensorParallel; r++)
                    states[l][r] = CreateRecurrentState(Scope(stage, r), l);
            }
            return states;
        }

        private static void ResetStates(RecurrentState[][] states)
        {
            if (states == null) return;
            foreach (RecurrentState[] layer in states)
                if (layer != null)
                    foreach (RecurrentState s in layer) s?.Reset();
        }

        private static void DisposeStates(RecurrentState[][] states)
        {
            if (states == null) return;
            foreach (RecurrentState[] layer in states)
                if (layer != null)
                    foreach (RecurrentState s in layer) s?.Dispose();
        }

        /// <summary>A batched sequence's states, created reset on first sight and reset again
        /// when the engine (re)starts it from token zero. Resolved once per batch, before any
        /// layer runs.</summary>
        private RecurrentState[][] SequenceStates(string sequenceId, bool fresh)
        {
            lock (_recurrentLock)
            {
                if (!_recurrentBySequence.TryGetValue(sequenceId, out var states))
                {
                    states = NewRecurrentStates();
                    _recurrentBySequence[sequenceId] = states;
                    return states;
                }
                if (fresh) ResetStates(states);
                return states;
            }
        }

        /// <summary>The engine finished with a sequence: its recurrent states go.</summary>
        public void OnSequenceReleased(string requestId)
        {
            if (requestId == null) return;
            lock (_recurrentLock)
            {
                if (_recurrentBySequence.Remove(requestId, out var states))
                    DisposeStates(states);
            }
        }

        private void ResetRecurrentPrimary() => ResetStates(_recurrentPrimary);

        private void DisposeRecurrentStates()
        {
            DisposeStates(_recurrentPrimary);
            _recurrentPrimary = null;
            lock (_recurrentLock)
            {
                foreach (var states in _recurrentBySequence.Values) DisposeStates(states);
                _recurrentBySequence.Clear();
            }
        }

        // ---------------------------------------------------------------- the mixer branch

        /// <summary>
        /// The token mixer of one layer on one rank: attention, or the family's recurrent block
        /// - run once per sequence of a paged batch, each with its own state.
        /// </summary>
        private Tensor Mixer(RankScope scope, Tensor normed, int layer, TokenPositions positions, PagedBatch batch)
        {
            if (!_recurrentLayer[layer])
                return Attention(scope, normed, layer, positions, batch);
            if (batch == null)
                return RecurrentMixer(scope, normed, layer, _recurrentPrimary[layer][scope.Rank]);

            int tokens = (int)normed.Sizes[0];
            var output = new Tensor(scope.Allocator, DType.Float32, tokens, Config.HiddenSize);
            for (int s = 0; s < batch.NumSequences; s++)
            {
                int start = batch.QueryStart[s];
                int count = (s + 1 < batch.NumSequences ? batch.QueryStart[s + 1] : tokens) - start;
                if (count <= 0) continue;
                RecurrentState state = batch.RecurrentStates[s][layer][scope.Rank];
                using Tensor rowsView = normed.Narrow(0, start, count);
                using Tensor rows = Ops.NewContiguous(rowsView);
                using Tensor part = RecurrentMixer(scope, rows, layer, state);
                using Tensor dst = output.Narrow(0, start, count);
                Ops.Copy(dst, part);
            }
            return output;
        }

        // ---------------------------------------------------------------- truncation

        public override bool CanTruncateKVCache(int cachedTokenCount, int targetTokenCount)
            => _anyRecurrent
                ? targetTokenCount == cachedTokenCount || targetTokenCount == 0
                : base.CanTruncateKVCache(cachedTokenCount, targetTokenCount);

        protected override bool TryTruncateKVCacheCore(int tokenCount)
        {
            if (_anyRecurrent && tokenCount != _cacheSeqLen && tokenCount != 0)
                return false;
            TruncateKVCacheCore(tokenCount);
            return true;
        }
    }
}
