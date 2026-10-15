// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using TensorSharp.Runtime.Paged;

namespace TensorSharp.Runtime.Scheduling.PrefixCache;

/// <summary>
/// End states of the batched route (<see cref="PrefixCacheCapabilities.PagedEndStates"/>). A family whose
/// concurrent requests run on the batched paged forward keeps each sequence's K/V in the engine's pool blocks
/// and its own per-sequence state beside them (Nemotron-H: a Mamba2 slot). Neither outlived a finished turn,
/// and pages cannot resume such a family anywhere but where a forward ended, so every conversation that ran
/// beside another re-prefilled its whole history each turn: 0 of 205-533 tokens for all eight parallel
/// Nemotron 3.5 Lightning conversations. A finished sequence now leaves its blocks and its model state behind
/// as its conversation's end state, which the next turn takes over at exact length.
/// </summary>
internal sealed partial class PrefixCacheCoordinator
{
    // payload key -> the pool blocks its K/V is in, the partial last one included; the tree holds one
    // reference on each until the end state is donated (the references become the sequence's) or released.
    private readonly Dictionary<string, KvBlock[]> _pagedEndStates = new(StringComparer.Ordinal);

    /// <summary>Keep a cleanly finished batched sequence for its conversation's next turn. Called by the
    /// scheduler before it frees the sequence's blocks, and so before the model's release frees its state.</summary>
    internal bool RetainPagedFinished(SequenceState sequence)
    {
        if (!_tree.Caps.PagedEndStates || !RetentionEnabled || !sequence.KvStateInPagedStorage
            || sequence.CacheBreakpoints is not null)
            return false;
        Drain();
        int length = Math.Min(sequence.NumComputedTokens, sequence.NumTotalTokens);
        if (length < _tree.Caps.MinRetainTokens) return false;
        int count = (int)(((long)length + _pool.BlockSize - 1) / _pool.BlockSize);
        if (sequence.BlockTable.NumBlocks < count) return false;
        KeyRope key = GetKey(sequence);
        RadixNode node = _tree.Insert(key, length, GetScope(sequence), sequence.SharedPrefixTokens,
            NodeFlags.None, GetSpans(sequence));
        if (node.EndState is not null) return false;
        string payloadKey = _tree.MintKey();
        if (!_cacheModel.TryCaptureDonate(sequence.RequestId, payloadKey, length, out PayloadFootprint footprint))
        {
            _tree.CollectIfEmpty(node);
            return false;
        }
        var blocks = new KvBlock[count];
        for (int i = 0; i < count; i++)
        {
            blocks[i] = sequence.BlockTable.Blocks[i];
            _pool.Touch(blocks[i]);
        }
        _pagedEndStates.Add(payloadKey, blocks);
        ResourceVector bytes = footprint.Bytes;
        bytes.PoolPages += count;
        bool published = Publish(node, payloadKey, footprint with { Bytes = bytes }, length, PayloadOrigin.Donation);
        Trim();
        return published;
    }

    private bool IsPagedEndState(MatchPlan plan)
        => plan.PayloadNode?.EndState is { } payload && _pagedEndStates.ContainsKey(payload.Key);

    /// <summary>Hand a paged end state to <paramref name="sequence"/>: the model moves its state to the request,
    /// and the tree's reference on every block becomes the sequence's. Exact length only: the blocks are the
    /// K/V, and the family's own state cannot rewind.</summary>
    private bool TryAdoptPagedEndState(SequenceState sequence, RadixNode node, EndStatePayload payload, int length)
    {
        if (_plan.Mode != MaterializeMode.DonateEndState || length != payload.Footprint.Tokens) return false;
        KvBlock[] blocks = _pagedEndStates[payload.Key];
        sequence.BlockTable.EnsureBlockCapacity(blocks.Length);
        var materialize = new MaterializeRequest(MaterializeOp.Donate, payload.Key, sequence.RequestId,
            payload.Footprint.Tokens, length);
        if (!_cacheModel.TryMaterialize(materialize)) return false;
        _pagedEndStates.Remove(payload.Key);
        _tree.DetachEndState(node, ReleaseReason.Rollback, enqueue: false);
        foreach (KvBlock block in blocks) sequence.BlockTable.AppendBlock(block);
        sequence.SetComputedTokensForPrefixAdoption(length);
        sequence.PrefixCacheReusedTokens = length;
        sequence.KvStateInPagedStorage = true;
        LastSource = "radix batched end state";
        return true;
    }

    /// <summary>Release model state before recycling its pool pages. A partial
    /// failure keeps every remaining page reference available for retry.</summary>
    private void ReleasePayloads(ReadOnlySpan<string> payloadKeys, ReleaseReason reason)
    {
        _cacheModel.ReleasePayloads(payloadKeys, reason);
        foreach (string key in payloadKeys)
        {
            if (key != null && _pagedEndStates.TryGetValue(key, out KvBlock[]? blocks))
            {
                for (int i = blocks.Length - 1; i >= 0; i--)
                {
                    _pool.Free(blocks[i]);
                    // These arrays are no longer adoptable after the tree queues
                    // their payload for release. Null marks only a confirmed free.
                    blocks[i] = null!;
                }
                _pagedEndStates.Remove(key);
            }
        }
    }
}
