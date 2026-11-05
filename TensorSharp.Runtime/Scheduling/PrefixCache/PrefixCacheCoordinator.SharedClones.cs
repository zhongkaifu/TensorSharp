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

// Requests that clone the same cached prefix share its blocks in the scheduler's pool, as they
// share its K/V in the model (PrefixCacheCapabilities.ClonesShareKv). Without this every clone was
// charged its whole prefix: on a pool sized for one context, a second chat on a cached 61K-token
// document waited for the first to finish although the model holds that document once.
internal sealed partial class PrefixCacheCoordinator
{
    // Payload key -> running requests that cloned it, with the full blocks of their cloned prefix.
    private readonly Dictionary<string, List<(SequenceState Sequence, KvBlock[] Blocks)>> _cloneSharers =
        new(StringComparer.Ordinal);

    /// <summary>The leading full blocks (at most <paramref name="maxBlocks"/>) a running request
    /// holds for a clone of <paramref name="payloadKey"/>, or none. Entries of requests that
    /// finished, were preempted or no longer hold those blocks are dropped.</summary>
    private KvBlock[] FindCloneSharer(string payloadKey, SequenceState sequence, int maxBlocks)
    {
        if (maxBlocks <= 0 || !_cloneSharers.TryGetValue(payloadKey, out var list))
            return Array.Empty<KvBlock>();
        KvBlock[] best = Array.Empty<KvBlock>();
        for (int i = list.Count - 1; i >= 0; i--)
        {
            var (other, blocks) = list[i];
            if (other.Status.IsFinished() || !StillHolds(other, blocks))
            {
                list.RemoveAt(i);
                continue;
            }
            if (!ReferenceEquals(other, sequence) && blocks.Length > best.Length)
                best = blocks;
        }
        if (list.Count == 0)
            _cloneSharers.Remove(payloadKey);
        return best.Length <= maxBlocks ? best : best[..maxBlocks];
    }

    private static bool StillHolds(SequenceState sequence, KvBlock[] blocks)
    {
        IReadOnlyList<KvBlock> table = sequence.BlockTable.Blocks;
        if (table.Count < blocks.Length)
            return false;
        for (int i = 0; i < blocks.Length; i++)
            if (!ReferenceEquals(table[i], blocks[i]) || blocks[i].RefCount <= 0)
                return false;
        return true;
    }

    private void RecordCloneSharer(string payloadKey, SequenceState sequence, int fullBlocks)
    {
        if (fullBlocks <= 0)
            return;
        var blocks = new KvBlock[fullBlocks];
        for (int i = 0; i < fullBlocks; i++)
            blocks[i] = sequence.BlockTable.Blocks[i];
        if (!_cloneSharers.TryGetValue(payloadKey, out var list))
            _cloneSharers[payloadKey] = list = new List<(SequenceState, KvBlock[])>();
        list.Add((sequence, blocks));
    }

    /// <summary>Blocks of <paramref name="sequence"/>'s prompt that admission would share with a
    /// running request instead of allocating: the full blocks of a cached prefix both clone.
    /// 0 when the model's clones copy their K/V, or no running request cloned that prefix.</summary>
    internal int ShareablePrefixBlocks(SequenceState sequence)
    {
        if (!_tree.Caps.ClonesShareKv || _cloneSharers.Count == 0 || sequence.BlockTable.NumBlocks != 0)
            return 0;
        int length = ComputeReusablePrefix(sequence);
        if (length <= 0 || _plan.Kind == CandidateKind.Pages || _plan.Mode != MaterializeMode.CloneEndState
            || _plan.PayloadNode?.EndState is not { } payload)
            return 0;
        return FindCloneSharer(payload.Key, sequence, length / _pool.BlockSize).Length;
    }
}
