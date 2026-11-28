// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// Where a model keeps each request's state, which decides what the scheduler's block pool stands for: the physical KV
    /// storage the blocks describe, or only the reservations it budgets admission with.
    /// </summary>
    public enum SequenceStateStorage
    {
        /// <summary>Per-request device slots the model sizes from the scheduler's reservations: the block pool holds one
        /// aggregate context's budget.</summary>
        Slots = 0,

        /// <summary>Each request's whole state lives in a native slot of its own (DeepSeek V4.1's executor): the block pool's
        /// blocks are zero-byte reservations only, so it is sized for every running request's context rather than one
        /// aggregate context (InferenceEngine.ResolveEffectiveNumBlocks), and a request longer than one slot is refused at
        /// submission.</summary>
        NativeSlots = 1,
    }

    /// <summary>
    /// The sequence state store of the unified execution contract: it creates, reserves, grows and releases each request's
    /// state. Together with the execution capability (<see cref="IUnifiedBatchModel"/>), the state reuse capability
    /// (<see cref="PrefixCache.IPrefixCacheModel"/>, optional) and the speculative execution capability
    /// (<see cref="TensorSharp.Runtime.Speculative.IBatchedSpeculativeTarget"/>, optional) it is everything a host needs to
    /// serve a model: a model that implements these does not implement <see cref="IBatchedPagedModel"/> or the KV snapshot
    /// contract to be admitted (<see cref="EngineServability"/>), and the executor binds, rewinds and releases its requests'
    /// state through this interface alone.
    /// </summary>
    public interface ISequenceStateStore
    {
        /// <summary>The request's state, created empty when it has none (a restored prefix already made one), with room for
        /// <paramref name="reserveTokens"/> (its prompt and generation budget). A store that sizes state to the device's memory
        /// may hold only <paramref name="requiredTokens"/> now and grow when a later step needs it. Returns the tokens the state
        /// holds; throws <see cref="SequenceSlotUnavailableException"/> when no state can be made beside the running ones.</summary>
        int EnsureSequenceSlot(string requestId, int reserveTokens, int requiredTokens);

        /// <summary>Whether the request has state in this store (a slot it ran on, or one a restored prefix made).</summary>
        bool HasSequenceState(string requestId);

        /// <summary>Rewinds the request's state to its first <paramref name="tokens"/> tokens (a reused state adopted past the
        /// prompt that continues it). False when the state cannot rewind there: the caller then discards it and re-runs the
        /// request from its start.</summary>
        bool TryRewindSequenceState(string requestId, int tokens);

        /// <summary>Frees the request's state (the request finished, was aborted or failed). A state the reuse capability
        /// retained for later requests (<see cref="PrefixCache.IPrefixCacheModel.TryCaptureDonate"/>) is no longer the
        /// request's, and is left alone.</summary>
        void ReleaseSequenceState(string requestId);

        /// <summary>Where the state lives (what the scheduler's block pool stands for).</summary>
        SequenceStateStorage StateStorage => SequenceStateStorage.Slots;
    }

    /// <summary>
    /// What a host needs to know to serve a model with an <see cref="InferenceEngine"/>: which execution contract the model
    /// is served on, or why it cannot be served. One decision, shared by every host (the chat host, the CLI, the server).
    /// </summary>
    public static class EngineServability
    {
        /// <summary>The contract <paramref name="model"/> is served on, or <see cref="EngineContract.None"/>.</summary>
        public static EngineContract ContractOf(IModelArchitecture model)
        {
            ArgumentNullException.ThrowIfNull(model);
            if (model is IUnifiedBatchModel { SupportsUnifiedBatching: true } and ISequenceStateStore)
                return EngineContract.Unified;
            if (model is IBatchedPagedModel)
                return EngineContract.BatchedPaged;
            if (model.SupportsKVStateSnapshot)
                return EngineContract.KvStateSnapshot;
            return EngineContract.None;
        }

        /// <summary>Whether an engine can serve <paramref name="model"/>.</summary>
        public static bool CanServe(IModelArchitecture model) => ContractOf(model) != EngineContract.None;

        /// <summary>Why <paramref name="model"/> cannot be served (null when it can).</summary>
        public static string? RefusalReason(IModelArchitecture model)
            => CanServe(model)
                ? null
                : $"The loaded model ({model.Config?.Architecture ?? model.GetType().Name}) implements no serving contract: neither the "
                  + "unified execution contract (IUnifiedBatchModel with an ISequenceStateStore), nor batched paged attention "
                  + "(IBatchedPagedModel), nor KV state snapshots (IModelArchitecture.SupportsKVStateSnapshot).";
    }

    /// <summary>The execution contract an engine serves a model on (<see cref="EngineServability"/>).</summary>
    public enum EngineContract
    {
        /// <summary>No contract: the model cannot be served.</summary>
        None = 0,
        /// <summary>The unified execution contract: per-request state in an <see cref="ISequenceStateStore"/>, every step one
        /// batched forward (<see cref="IUnifiedBatchModel"/>).</summary>
        Unified = 1,
        /// <summary>The legacy batched paged-attention contract (<see cref="IBatchedPagedModel"/>).</summary>
        BatchedPaged = 2,
        /// <summary>The legacy single-stream contract: one primary cache swapped between requests through KV snapshots.</summary>
        KvStateSnapshot = 3,
    }
}
