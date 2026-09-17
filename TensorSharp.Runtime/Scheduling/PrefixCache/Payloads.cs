// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.CompilerServices;
using TensorSharp.Runtime.Paged;

namespace TensorSharp.Runtime.Scheduling.PrefixCache;

/// <summary>Accounting classes (DEC-30). Disk is tracked by the checkpoint store's own LRU.</summary>
public enum ResourceClass : byte { PoolPages, HostKv, DeviceKv, StateSnapshot, NativeSlot }

/// <summary>Per-class amounts. A value type with no arrays; checked arithmetic in Debug builds.</summary>
public struct ResourceVector : IEquatable<ResourceVector>
{
    public const int ClassCount = 5;

    public long PoolPages, HostKv, DeviceKv, StateSnapshot, NativeSlot;

    public long this[ResourceClass c]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        readonly get => c switch
        {
            ResourceClass.PoolPages => PoolPages,
            ResourceClass.HostKv => HostKv,
            ResourceClass.DeviceKv => DeviceKv,
            ResourceClass.StateSnapshot => StateSnapshot,
            ResourceClass.NativeSlot => NativeSlot,
            _ => throw new ArgumentOutOfRangeException(nameof(c)),
        };
        set
        {
            switch (c)
            {
                case ResourceClass.PoolPages: PoolPages = value; break;
                case ResourceClass.HostKv: HostKv = value; break;
                case ResourceClass.DeviceKv: DeviceKv = value; break;
                case ResourceClass.StateSnapshot: StateSnapshot = value; break;
                case ResourceClass.NativeSlot: NativeSlot = value; break;
                default: throw new ArgumentOutOfRangeException(nameof(c));
            }
        }
    }

    public static ResourceVector operator +(ResourceVector a, ResourceVector b)
    {
#if DEBUG
        checked
        {
#endif
            return new ResourceVector
            {
                PoolPages = a.PoolPages + b.PoolPages,
                HostKv = a.HostKv + b.HostKv,
                DeviceKv = a.DeviceKv + b.DeviceKv,
                StateSnapshot = a.StateSnapshot + b.StateSnapshot,
                NativeSlot = a.NativeSlot + b.NativeSlot,
            };
#if DEBUG
        }
#endif
    }

    public static ResourceVector operator -(ResourceVector a, ResourceVector b)
    {
#if DEBUG
        checked
        {
#endif
            return new ResourceVector
            {
                PoolPages = a.PoolPages - b.PoolPages,
                HostKv = a.HostKv - b.HostKv,
                DeviceKv = a.DeviceKv - b.DeviceKv,
                StateSnapshot = a.StateSnapshot - b.StateSnapshot,
                NativeSlot = a.NativeSlot - b.NativeSlot,
            };
#if DEBUG
        }
#endif
    }

    public static ResourceVector Negate(ResourceVector a) => default(ResourceVector) - a;

    public readonly bool AnyNegative => PoolPages < 0 || HostKv < 0 || DeviceKv < 0 || StateSnapshot < 0 || NativeSlot < 0;

    public readonly bool IsZero => PoolPages == 0 && HostKv == 0 && DeviceKv == 0 && StateSnapshot == 0 && NativeSlot == 0;

    /// <summary>Σ of the byte classes (every class except <see cref="ResourceClass.PoolPages"/>).</summary>
    public readonly long TotalBytes => HostKv + DeviceKv + StateSnapshot + NativeSlot;

    public readonly bool Equals(ResourceVector o) =>
        PoolPages == o.PoolPages && HostKv == o.HostKv && DeviceKv == o.DeviceKv && StateSnapshot == o.StateSnapshot && NativeSlot == o.NativeSlot;

    public override readonly bool Equals(object? obj) => obj is ResourceVector o && Equals(o);

    public override readonly int GetHashCode() => HashCode.Combine(PoolPages, HostKv, DeviceKv, StateSnapshot, NativeSlot);

    public static bool operator ==(ResourceVector a, ResourceVector b) => a.Equals(b);

    public static bool operator !=(ResourceVector a, ResourceVector b) => !a.Equals(b);

    public override readonly string ToString() =>
        $"pages={PoolPages} host={HostKv} device={DeviceKv} state={StateSnapshot} native={NativeSlot}";
}

/// <summary>What the model reports for an end state it owns.</summary>
public readonly record struct PayloadFootprint(
    int Tokens,              // rows of state (== node depth)
    int CapacityTokens,      // allocated rows (holders grow by doubling; charge capacity)
    ResourceVector Bytes,    // host K/V, device mirrors, recurrent/PLE/QSA state, native slot
    int PositionDelta);      // M-RoPE delta at Tokens (validation only; 0 for absolute-position models)

internal enum EndStateKind : byte { Holder, NativeSlot, PrimaryResident }

internal enum PayloadOrigin : byte { CaptureCopy, Donation, PrimaryConversion, DiskImport }

/// <summary>An end state owned by the tree: the model state after token <c>Depth − 1</c> of its node.</summary>
internal sealed class EndStatePayload
{
    internal string Key = string.Empty;   // tree-minted "pc:{engineSerial}:{payloadSerial}"; never a request id (I10)
    internal EndStateKind Kind;
    internal PayloadOrigin Origin;
    internal PayloadFootprint Footprint;  // PrimaryResident: all zero (the primary is not cache-owned)
    internal bool DeviceDirty;            // donated holder with device-authoritative state; settle before clone (§6)
    internal bool Persisted;

    internal ResourceVector Bytes => Kind == EndStateKind.PrimaryResident ? default : Footprint.Bytes;
}

/// <summary>The store of record of a page's bytes (P15, G-03).</summary>
public enum PageStore : byte { A1HostSlab = 1, A2ModelPaged = 2, Both = 3 }

/// <summary>One page reference owned by the tree (one <see cref="KvBlock.RefCount"/> unit).</summary>
internal readonly record struct PageRef(KvBlock Block, int PageIndex, PageStore Store, bool StateAtEnd)
{
    internal bool HasA1 => (Store & PageStore.A1HostSlab) != 0;
    internal bool HasA2 => (Store & PageStore.A2ModelPaged) != 0;
}

/// <summary>
/// What the tree needs from the block pool for the pages it owns. The engine implementation
/// (M4) wraps <see cref="BlockPool"/> and the Hunyuan branch's <c>KvBlock.HoldsModelPagedKv</c> /
/// <c>HoldsSnapshotBytes</c> flags, which are not on the M1 base commit.
/// </summary>
internal interface IPrefixTreePageHost
{
    /// <summary>Takes one reference for the tree (<c>BlockPool.Touch</c>).</summary>
    void RetainPage(KvBlock block);
    /// <summary>Drops the tree's reference (<c>BlockPool.Free</c>).</summary>
    void FreePage(KvBlock block);
    bool HoldsModelPagedKv(KvBlock block);
    bool HoldsSnapshotBytes(KvBlock block);
    int UsedTokens(KvBlock block);
    int RefCount(KvBlock block);
}

/// <summary>A page host that owns nothing: every flag is true and refcounts are tracked locally.</summary>
internal sealed class NullPageHost : IPrefixTreePageHost
{
    internal static readonly NullPageHost Instance = new();
    public void RetainPage(KvBlock block) { }
    public void FreePage(KvBlock block) { }
    public bool HoldsModelPagedKv(KvBlock block) => true;
    public bool HoldsSnapshotBytes(KvBlock block) => true;
    public int UsedTokens(KvBlock block) => int.MaxValue;
    public int RefCount(KvBlock block) => int.MaxValue;
}
