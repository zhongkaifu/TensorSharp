// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.Extensions.Logging;

namespace TensorSharp.Runtime.Scheduling
{
    public sealed partial class BatchExecutor
    {
        /// <summary>
        /// How the unified step binds, rewinds and reconciles each request's state (D01). A model that implements only the
        /// unified execution contract is served through its <see cref="ISequenceStateStore"/> alone
        /// (<see cref="StoreStateBinding"/>). A model that still implements the legacy per-request holder protocol
        /// (<see cref="IBatchedPagedModel"/>: the shared decoder keeps it for its paged-block prefix injection) is served
        /// through <see cref="LegacyHolderBinding"/>, the adapter that keeps that protocol's bind, inject and restore order.
        /// Chosen once per executor, so the unified branch itself never casts to the legacy contract.
        /// </summary>
        private abstract class UnifiedStateBinding
        {
            protected UnifiedStateBinding(BatchExecutor executor) => Executor = executor;

            protected BatchExecutor Executor { get; }

            /// <summary>Prepares <paramref name="seq"/>'s state before its slot is ensured: a reused state adopted past the
            /// prompt is rewound, a revoked live-cache claim dropped. False when the step is over for the sequence (a result
            /// was added).</summary>
            public abstract bool Prepare(IUnifiedBatchModel model, SequenceState seq, List<SequenceStepResult> results);

            /// <summary>Reconciles the tokens the store holds (<paramref name="held"/>) with the sequence's computed tokens
            /// once its slot is ensured. Throws when they cannot be reconciled.</summary>
            public abstract void Reconcile(IUnifiedBatchModel model, SequenceState seq, int held);
        }

        private UnifiedStateBinding? _unifiedState;

        /// <summary>The unified step's state binding for this executor's model.</summary>
        private UnifiedStateBinding UnifiedState
            => _unifiedState ??= _model is IBatchedPagedModel legacy
                ? new LegacyHolderBinding(this, legacy)
                : new StoreStateBinding(this);

        /// <summary>Whether this executor serves its unified model through the state store alone (no legacy holder
        /// protocol): tests check that a model implementing only the new contract runs on it.</summary>
        internal bool ServesUnifiedThroughStateStore => UnifiedState is StoreStateBinding;

        /// <summary>
        /// The adapter that serves a unified model through the legacy holder protocol (<see cref="IBatchedPagedModel"/>): a
        /// request without a holder but with computed tokens, a live-cache claim or a planned rewind binds its holder the way
        /// the per-sequence fused path does (<see cref="BindFusedHolder"/>: rewind, revoke, inject the reused prefix from the
        /// paged blocks or recompute it), then the primary cache is reinstated.
        /// </summary>
        private sealed class LegacyHolderBinding : UnifiedStateBinding
        {
            private readonly IBatchedPagedModel _holders;

            public LegacyHolderBinding(BatchExecutor executor, IBatchedPagedModel holders) : base(executor) => _holders = holders;

            public override bool Prepare(IUnifiedBatchModel model, SequenceState seq, List<SequenceStepResult> results)
            {
                if (!Executor._pendingRetainedFusedTruncations.ContainsKey(seq.RequestId)
                    && (_holders.HasFusedSequenceCache(seq.RequestId) || (seq.NumComputedTokens == 0 && !seq.UsesLiveCacheContinuation)))
                    return true;
                try { return Executor.BindFusedHolder(_holders, seq, results); }
                finally { _holders.RestorePrimaryCache(); }
            }

            public override void Reconcile(IUnifiedBatchModel model, SequenceState seq, int held)
            {
                if (held != seq.NumComputedTokens)
                    throw new InvalidOperationException($"{seq.RequestId}: its slot holds {held} tokens, the sequence {seq.NumComputedTokens}.");
            }
        }

        /// <summary>
        /// The unified execution contract's own binding: everything goes through the model's <see cref="ISequenceStateStore"/>.
        /// A planned rewind is <see cref="ISequenceStateStore.TryRewindSequenceState"/> (a state that cannot rewind is released
        /// and the request runs from its start); a live-cache claim without state is dropped; a sequence whose store holds
        /// fewer tokens than it computed (a reused state lost before its first step) has the lost tail forwarded again on its
        /// own slot; one that holds more is rewound to it.
        /// </summary>
        private sealed class StoreStateBinding : UnifiedStateBinding
        {
            public StoreStateBinding(BatchExecutor executor) : base(executor) { }

            public override bool Prepare(IUnifiedBatchModel model, SequenceState seq, List<SequenceStepResult> results)
            {
                if (Executor._pendingRetainedFusedTruncations.Remove(seq.RequestId, out int target))
                {
                    if (!model.HasSequenceState(seq.RequestId))
                        throw new InvalidOperationException($"The reused state of {seq.RequestId} was adopted but the store holds none.");
                    if (!model.TryRewindSequenceState(seq.RequestId, target))
                    {
                        Executor._logger.LogInformation(
                            "The reused state of {EngineRequestId} could not rewind to {Tokens}; re-running its prompt from the start.",
                            seq.RequestId, target);
                        model.ReleaseSequenceState(seq.RequestId);
                        seq.ClearLiveCacheContinuation();
                        seq.PrefixCheckpointTaken = false;
                        seq.PrefixCacheReusedTokens = 0;
                        seq.RevokeComputedTokensTo(0);
                        // The scheduled chunk describes the old suffix: the next step plans from zero.
                        results.Add(new SequenceStepResult { Sequence = seq, TokensForwarded = 0, IsPrefill = true });
                        return false;
                    }
                }
                if (seq.UsesLiveCacheContinuation && !model.HasSequenceState(seq.RequestId))
                {
                    // A unified model has no live primary cache to continue: the claimed state is gone.
                    Executor._logger.LogInformation(
                        "Planned live-cache continuation for {EngineRequestId} has no state in the store; re-running the prompt.",
                        seq.RequestId);
                    seq.ClearLiveCacheContinuation();
                }
                return true;
            }

            public override void Reconcile(IUnifiedBatchModel model, SequenceState seq, int held)
            {
                int computed = seq.NumComputedTokens;
                if (held == computed) return;
                if (held > computed)
                {
                    if (!model.TryRewindSequenceState(seq.RequestId, computed))
                        throw new InvalidOperationException(
                            $"{seq.RequestId}: its state holds {held} tokens past the sequence's {computed} and cannot rewind.");
                    return;
                }
                Executor._logger.LogWarning(
                    "The state of {EngineRequestId} holds {Held} of its {Computed} computed tokens; the remaining {Lost} are forwarded again.",
                    seq.RequestId, held, computed, computed - held);
                seq.PrefixCacheReusedTokens = Math.Min(seq.PrefixCacheReusedTokens, held);
                seq.RevokeComputedTokensTo(held);
                Executor.RecomputeLostTailUnified(model, seq, computed);
            }
        }

        /// <summary><see cref="RecomputeLostTail"/> on the request's own slot (the unified forward, not the primary cache):
        /// forwards <paramref name="seq"/>'s tokens from its computed count up to <paramref name="target"/> in scheduler-sized
        /// chunks that never mix prompt and generated tokens, without sampling. The logits the sequence samples from next are
        /// kept.</summary>
        private void RecomputeLostTailUnified(IUnifiedBatchModel model, SequenceState seq, int target)
        {
            int start = seq.NumComputedTokens;
            if (start >= target) return;
            float[]? pendingLogits = seq.LastLogits != null ? (float[])seq.LastLogits.Clone() : null;
            float[]? lastLogits = null;
            int chunk = Math.Max(1, _scheduler.Config.MaxNumBatchedTokens);
            int promptTokens = seq.PromptTokens.Count;
            var injector = _model.MultimodalInjector;
            string[] ids = { seq.RequestId };
            while (start < target)
            {
                int n = Math.Min(chunk, target - start);
                if (start < promptTokens) n = Math.Min(n, promptTokens - start);
                var tokens = new int[n];
                for (int i = 0; i < n; i++) tokens[i] = seq.TokenAt(start + i);
                long started = Stopwatch.GetTimestamp();
                if (injector != null && start < promptTokens && injector.HasPendingEmbeddings(seq.RequestId))
                {
                    injector.QueuePromptEmbeddingsForSlice(start, n, seq.RequestId);
                    lastLogits = model.ForwardUnifiedMedia(seq.RequestId, tokens, start);
                }
                else
                {
                    var logits = new float[1][];
                    if (!model.TryForwardUnified(ids, new[] { tokens }, new[] { start }, logits))
                        throw new InvalidOperationException($"{seq.RequestId}: its slot declined the re-forward of tokens {start}..{start + n}.");
                    lastLogits = logits[0];
                }
                long ticks = Stopwatch.GetTimestamp() - started;
                if (start < promptTokens) seq.ReplayPrefillElapsedTicks += ticks;
                else seq.ReplayDecodeElapsedTicks += ticks;
                seq.AdvanceComputedTokens(n);
                start += n;
            }
            seq.LastLogits = pendingLogits ?? (lastLogits != null ? (float[])lastLogits.Clone() : null);
        }
    }
}
