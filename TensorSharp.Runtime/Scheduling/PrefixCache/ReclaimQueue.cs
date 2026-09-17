// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

namespace TensorSharp.Runtime.Scheduling.PrefixCache;

/// <summary>Why the tree hands a payload back to the model (§4.8).</summary>
public enum ReleaseReason : byte { Evicted, Duplicate, ScopeRetired, Invalidated, Rollback, Reset, Pressure }

/// <summary>The model side of a batched release: <c>IPrefixCacheModel.ReleasePayloads</c> (M2).</summary>
internal delegate void PayloadReleaseSink(ReadOnlySpan<string> payloadKeys, ReleaseReason reason);

/// <summary>
/// Batched deferred payload release (DEC-24). Evicted and duplicate payload keys queue here and
/// reach the model once per step through one <see cref="PayloadReleaseSink"/> call, so one
/// eviction no longer resets every running request's decode graphs.
/// </summary>
internal sealed class ReclaimQueue
{
    private readonly struct Entry
    {
        internal readonly string Key;
        internal readonly ReleaseReason Reason;
        internal readonly ResourceVector Bytes;

        internal Entry(string key, ReleaseReason reason, ResourceVector bytes)
        {
            Key = key; Reason = reason; Bytes = bytes;
        }
    }

    private readonly List<Entry> _entries = new();
    private readonly HashSet<string> _keys = new(StringComparer.Ordinal);
    private string[] _drainBuffer = new string[16];
    private ResourceVector _pending;
    private ReleaseReason _batchReason;

    /// <summary>Bytes of queued payloads not yet released to the model.</summary>
    internal ResourceVector PendingReclaim => _pending;

    internal int Count => _entries.Count;

    internal long DrainCalls { get; private set; }

    internal long ReleasedKeys { get; private set; }

    internal bool Contains(string key) => _keys.Contains(key);

    internal IEnumerable<string> Keys => _keys;

    /// <summary>Queues <paramref name="key"/> once. A repeated key can still upgrade the batch to a
    /// disposal reason, without counting its bytes twice.</summary>
    internal bool Enqueue(string key, ReleaseReason reason, ResourceVector bytes)
    {
        if (string.IsNullOrEmpty(key)) throw new ArgumentException("A payload key is required.", nameof(key));
        // Models pool ordinary evictions, but must free invalidated payloads and allocations released
        // for pressure/reset. Keep one model call (and one graph reset) while ensuring a prior ordinary
        // eviction cannot turn a later mandatory disposal into pooled memory.
        if (_entries.Count == 0 || (!RequiresDisposal(_batchReason) && RequiresDisposal(reason)))
            _batchReason = reason;
        if (!_keys.Add(key)) return false;
        _entries.Add(new Entry(key, reason, bytes));
        _pending += bytes;
        return true;
    }

    private static bool RequiresDisposal(ReleaseReason reason) =>
        reason is ReleaseReason.Invalidated or ReleaseReason.Pressure or ReleaseReason.Reset;

    /// <summary>
    /// Releases every queued key through one <paramref name="sink"/> call. The first disposal reason
    /// takes precedence over reasons that allow pooling; otherwise the first entry's reason is used.
    /// Returns the number of keys released (0 means the sink was not called).
    /// </summary>
    internal int Drain(PayloadReleaseSink? sink)
    {
        int n = _entries.Count;
        if (n == 0) return 0;
        if (_drainBuffer.Length < n)
            _drainBuffer = new string[Math.Max(n, _drainBuffer.Length * 2)];
        for (int i = 0; i < n; i++)
            _drainBuffer[i] = _entries[i].Key;
        ReleaseReason reason = _batchReason;
        try
        {
            sink?.Invoke(new ReadOnlySpan<string>(_drainBuffer, 0, n), reason);
        }
        finally
        {
            Array.Clear(_drainBuffer, 0, n);
            _entries.Clear();
            _keys.Clear();
            _pending = default;
            DrainCalls++;
            ReleasedKeys += n;
        }
        return n;
    }

    /// <summary>Test hook for invariant mutation tests: queues a key without accounting.</summary>
    internal void UnsafeEnqueueKeyOnly(string key)
    {
        if (_keys.Add(key)) _entries.Add(new Entry(key, ReleaseReason.Evicted, default));
    }
}
