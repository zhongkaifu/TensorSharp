// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;

namespace TensorSharp.Runtime.Scheduling.PrefixCache;

/// <summary>
/// A replayable lock (DESIGN §4.5, P4-P6). A path lock increments <c>LockRef</c> from
/// <see cref="PathAnchor"/> to the root (root excluded); a state lock increments
/// <c>StateLockRef</c> on <see cref="StateAnchor"/> only. <c>default</c> is an empty receipt,
/// and releasing it is a no-op: a lost receipt leaks instead of over-releasing.
/// </summary>
public readonly struct LockReceipt
{
    internal readonly long TreeSerial;       // a receipt from another tree or a Reset epoch → ignored + counted (stale)
    internal readonly long Serial;           // ledger key (Debug/tests)
    internal readonly RadixNode? PathAnchor; internal readonly int PathGeneration;    // null → nothing path-locked
    internal readonly RadixNode? StateAnchor; internal readonly int StateGeneration;  // null → no state lock

    internal LockReceipt(long treeSerial, long serial, RadixNode? pathAnchor, RadixNode? stateAnchor)
    {
        TreeSerial = treeSerial;
        Serial = serial;
        PathAnchor = pathAnchor;
        PathGeneration = pathAnchor?.Generation ?? 0;
        StateAnchor = stateAnchor;
        StateGeneration = stateAnchor?.Generation ?? 0;
    }

    public bool IsEmpty => PathAnchor is null && StateAnchor is null;

    /// <summary>Depth of the path anchor (0 when nothing is path-locked).</summary>
    public int PathDepth => PathAnchor?.Depth ?? 0;

    public override string ToString() =>
        IsEmpty ? "receipt(empty)" : $"receipt(#{Serial}, path={PathAnchor?.Id.ToString() ?? "-"}, state={StateAnchor?.Id.ToString() ?? "-"})";
}

/// <summary>Open receipts, recorded when <see cref="PrefixTreeOptions.TrackReceipts"/> is set (Debug and tests; I6, I24).</summary>
internal sealed class ReceiptLedger
{
    private readonly Dictionary<long, LockReceipt> _open = new();

    internal int Count => _open.Count;

    internal IEnumerable<LockReceipt> Open => _open.Values;

    internal void Add(in LockReceipt r)
    {
        if (!r.IsEmpty) _open[r.Serial] = r;
    }

    internal bool Remove(long serial) => _open.Remove(serial);

    internal bool Contains(long serial) => _open.ContainsKey(serial);

    internal void Replace(in LockReceipt r)
    {
        if (r.IsEmpty) _open.Remove(r.Serial);
        else _open[r.Serial] = r;
    }

    internal void Clear() => _open.Clear();

    /// <summary>Test hook for mutation tests: records a receipt the tree never issued.</summary>
    internal void UnsafeAdd(in LockReceipt r) => _open[r.Serial] = r;
}
