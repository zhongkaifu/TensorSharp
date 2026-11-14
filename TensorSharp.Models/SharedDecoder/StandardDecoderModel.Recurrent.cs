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

            /// <summary><see cref="Reset"/> that rewrites every device copy in place instead of dropping it, so the
            /// graphs cached over this state stay valid (a pooled slot's next request replays them). False: the
            /// state reset through <see cref="Reset"/>, and its device copies are gone.</summary>
            public virtual bool ResetInPlace()
            {
                Reset();
                return false;
            }

            /// <summary>Host tensors holding the state, which a whole-forward graph may update on
            /// the device; synced back before any host code reads them.</summary>
            protected internal virtual System.Collections.Generic.IEnumerable<Tensor> HostTensors => Array.Empty<Tensor>();

            /// <summary>The state can be copied (shared-prefix checkpoints, clones).</summary>
            protected internal virtual bool CanCopy => false;

            /// <summary>Copy <paramref name="source"/>'s host-current state into this one.</summary>
            protected internal virtual void CopyFrom(RecurrentState source)
                => throw new NotSupportedException($"{GetType().Name} cannot be copied.");
            /// <summary><see cref="CopyFrom"/> that writes each existing device copy in place (see
            /// <see cref="ResetInPlace"/>); false when it dropped them instead.</summary>
            protected internal virtual bool CopyFromInPlace(RecurrentState source)
            {
                CopyFrom(source);
                return false;
            }

            /// <summary>The tensors a checkpoint file holds, without aliases.</summary>
            protected internal virtual System.Collections.Generic.IEnumerable<Tensor> CheckpointTensors => HostTensors;
            /// <summary>The F32 tensors whole-forward graphs bind for this state, each through its own key, whose device
            /// copies hold the state between steps: a copy on the device copies these. Null: the state copies on the
            /// host only.</summary>
            protected internal virtual Tensor[] GraphTensors => null;
            /// <summary>A reset state's <see cref="GraphTensors"/> hold zeros only, so a fresh state's device copies may be
            /// created zero-filled on the device instead of uploaded (<see cref="StandardDecoderModel.PrimeDeviceCopies"/>).</summary>
            protected internal virtual bool GraphTensorsResetToZero => false;
            /// <summary>After <see cref="GraphTensors"/> were copied from <paramref name="source"/> on the device: copy
            /// what the state keeps on the host.</summary>
            protected internal virtual void AfterDeviceCopy(RecurrentState source) { }

            /// <summary>The state's host bytes were rewritten: drop every device copy of them.</summary>
            protected internal virtual void OnHostWritten() { }

            /// <summary>The slot holding this state no longer speculates (it was retained for prefix reuse): free what
            /// only a verify window needs; a later verify allocates it again.</summary>
            protected internal virtual void ReleaseSpecBuffers() { }

            /// <summary>Counts every change of a state's graph-bound buffers (a verify buffer grown, or freed when its slot
            /// was retained or pooled). A decode step kept to advance binds the buffers it was traced with, so it is never
            /// advanced or replayed across a change: a recorded verify window had written a freed verify buffer of the
            /// pooled slot the next request took (CUDA error 700).</summary>
            internal static long BindingChanges;

            protected static void NoteBindingsChanged() => System.Threading.Interlocked.Increment(ref BindingChanges);

            /// <summary>Bytes of the state's tensors (a retained slot's footprint counts them with its caches).</summary>
            protected internal virtual long StateBytes => 0;

            /// <summary>The per-op path can verify a draft window on this state: it keeps the state
            /// before the window and after each of its rows (<see cref="BeginVerify"/>).</summary>
            protected internal virtual bool SupportsVerify => false;

            /// <summary>The next multi-row step is a verify window: keep the state before it and
            /// after each of its rows until <see cref="CommitVerify"/>.</summary>
            protected internal virtual void BeginVerify()
                => throw new NotSupportedException($"{GetType().Name} cannot verify a draft window.");

            /// <summary>Keep the state after <paramref name="kept"/> rows of the window (0: before it).</summary>
            protected internal virtual void CommitVerify(int kept)
                => throw new NotSupportedException($"{GetType().Name} cannot verify a draft window.");

            public virtual void Dispose() { }
        }

        private bool[] _recurrentLayer;
        private bool _anyRecurrent;
        private RecurrentState[][] _recurrentPrimary;   // [layer][rank]

        /// <summary>The active cache's states give up their verify buffers, as a retained slot's do (tests).</summary>
        internal void ReleaseActiveSpecBuffersForTests()
        {
            if (_recurrentPrimary == null) return;
            foreach (RecurrentState[] layer in _recurrentPrimary)
                if (layer != null)
                    foreach (RecurrentState state in layer) state?.ReleaseSpecBuffers();
        }

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
            _recurrentLayer = new bool[TotalLayers];   // draft layers are attention layers
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

        /// <summary><see cref="RecurrentState.ResetInPlace"/> for every state; false when any dropped its device copies.</summary>
        private static bool ResetStatesInPlace(RecurrentState[][] states)
        {
            bool kept = true;
            if (states == null) return kept;
            foreach (RecurrentState[] layer in states)
                if (layer != null)
                    foreach (RecurrentState s in layer)
                        if (s != null) kept &= s.ResetInPlace();
            return kept;
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
        private void ResetRecurrentPrimary() => ResetStates(_recurrentPrimary);

        private void DisposeRecurrentStates()
        {
            DisposeStates(_recurrentPrimary);
            _recurrentPrimary = null;
        }

        // ---------------------------------------------------------------- the mixer branch

        /// <summary>
        /// The token mixer of one layer on one rank: attention, or the family's recurrent block
        /// - run once per sequence of a paged batch, each with its own state.
        /// </summary>
        private Tensor Mixer(RankScope scope, Tensor normed, int layer, TokenPositions positions, SlotBatch batch)
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
                int count = batch.Counts[s];
                RecurrentState state = batch.Slots[s].Recurrent[layer][scope.Rank];
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
