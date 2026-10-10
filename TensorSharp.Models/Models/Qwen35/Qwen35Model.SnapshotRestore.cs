// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using TensorSharp.Runtime.Paged;

namespace TensorSharp.Models;

public partial class Qwen35Model
{
    /// <summary>Attention rows accumulate; recurrent state is replaced at every
    /// endpoint. Copy only the last accepted endpoint, while validating every
    /// block before its attention writes. At most one snapshot lease is live.</summary>
    public int RestoreKvSnapshots(int blockTokens, int tokens, Func<int, KvSnapshotLease> acquire)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockTokens);
        ArgumentOutOfRangeException.ThrowIfNegative(tokens);
        ArgumentNullException.ThrowIfNull(acquire);
        if (_cacheSeqLen != 0) throw new InvalidOperationException("Bulk KV restoration requires an empty model cache.");
        int restored = 0, lastCount = 0;
        bool deferred = false;
        for (int b = 0; restored < tokens; b++)
        {
            int count = Math.Min(blockTokens, tokens - restored);
            long expected = ComputeKVBlockByteSize(count);
            if (expected <= 0 || expected > int.MaxValue) break;
            using var lease = acquire(b);
            if (lease.ReadOnlySpan.Length < expected) break;
            bool defer = restored + count < tokens;
            if (!TryInjectKvBlockCore(restored, count, lease.ReadOnlySpan[..(int)expected], defer, endStateOnly: false)) break;
            restored += count;
            lastCount = count;
            deferred = defer;
        }
        // An invalid next block must still leave the preceding endpoint exact.
        // Reacquire after releasing its successor; this also works with one RAM
        // page and SSD spill, without retaining a pointer or uncharged state copy.
        if (deferred)
        {
            int start = restored - lastCount;
            using var lease = acquire(start / blockTokens);
            int expected = checked((int)ComputeKVBlockByteSize(lastCount));
            if (lease.ReadOnlySpan.Length < expected
                || !TryInjectKvBlockCore(start, lastCount, lease.ReadOnlySpan[..expected], deferEndState: false, endStateOnly: true))
                throw new InvalidOperationException("The accepted KV endpoint could not be restored.");
        }
        return restored;
    }
}
