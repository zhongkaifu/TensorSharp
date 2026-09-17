// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Security.Cryptography;

namespace TensorSharp.Runtime.Scheduling.PrefixCache;

/// <summary>Isolation modes for stateless requests (DEC-08).</summary>
public enum IsolationMode : byte { Lineage, Strict, Shared }

/// <summary>How a request's scope was derived (DESIGN §4.2).</summary>
public enum ScopeKind : byte { Public, ClientKey, Session, Lineage, Fresh, Shared, Warmup, Unscoped }

/// <summary>
/// A 128-bit cache scope id (DEC-06). The zero value is the public scope. The engine only ever
/// sees Phase 0's SHA-256-derived hex, never a raw session id, client key or tenant.
/// </summary>
public readonly record struct ScopeId(UInt128 Value)
{
    public static ScopeId Public => default;

    public bool IsPublic => Value == UInt128.Zero;

    /// <summary>
    /// Phase 0 scope string: 32 hex chars (16 bytes, M0f), or 16 hex chars (Phase 0 before M0f,
    /// zero-extended). Throws <see cref="ArgumentException"/> for anything else, and for an
    /// all-zero id (which would alias the public scope).
    /// </summary>
    public static ScopeId FromHex(string hex)
    {
        if (hex is null) throw new ArgumentNullException(nameof(hex));
        UInt128 value;
        if (hex.Length == 32)
        {
            if (!ulong.TryParse(hex.AsSpan(0, 16), NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture, out ulong hi)
                || !ulong.TryParse(hex.AsSpan(16, 16), NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture, out ulong lo))
                throw new ArgumentException($"Scope id '{Truncate(hex)}' is not hexadecimal.", nameof(hex));
            value = new UInt128(hi, lo);
        }
        else if (hex.Length == 16)
        {
            if (!ulong.TryParse(hex, NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture, out ulong lo))
                throw new ArgumentException($"Scope id '{Truncate(hex)}' is not hexadecimal.", nameof(hex));
            value = new UInt128(0, lo);
        }
        else
        {
            throw new ArgumentException($"Scope id must be 32 or 16 hex characters (got {hex.Length}).", nameof(hex));
        }
        if (value == UInt128.Zero)
            throw new ArgumentException("An all-zero scope id would alias the public scope.", nameof(hex));
        return new ScopeId(value);
    }

    /// <summary>128 random bits from <see cref="RandomNumberGenerator"/>; never zero.</summary>
    public static ScopeId NewFresh()
    {
        Span<byte> bytes = stackalloc byte[16];
        while (true)
        {
            RandomNumberGenerator.Fill(bytes);
            var value = new UInt128(
                System.Buffers.Binary.BinaryPrimitives.ReadUInt64BigEndian(bytes.Slice(0, 8)),
                System.Buffers.Binary.BinaryPrimitives.ReadUInt64BigEndian(bytes.Slice(8, 8)));
            if (value != UInt128.Zero)
                return new ScopeId(value);
        }
    }

    /// <summary>First 8 hex chars only (O6): enough to correlate log lines, never the full id.</summary>
    public string ToLogToken() => ((ulong)(Value >> 96)).ToString("x8", CultureInfo.InvariantCulture).Substring(0, 8);

    public override string ToString() => IsPublic ? "public" : ToLogToken();

    private static string Truncate(string s) => s.Length <= 8 ? s : s.Substring(0, 8) + "…";
}

/// <summary>Per-scope bookkeeping, interned to a small index for hot-path child keys.</summary>
internal sealed class ScopeRecord
{
    public ScopeId Id; public ScopeKind Kind; public int Index;   // Index > 0; 0 = Public
    public RadixNode? FirstNode;                                  // intrusive list of this scope's nodes (I15)
    public ResourceVector Bytes;
    public RadixNode? NewestLeaf;                                 // tier ScopeNewest candidate
    public long LastUseTick; public int RunningRequests; public int WaitingRequests;
    public bool Retired;
    public bool Live;                                             // false once recycled into the free list
    public int NodeCount;
    public long LastUseMs;                                        // clock of the last request activity (idle rule)
    public bool Active;                                           // cached: running + waiting > 0, or idle < ScopeIdleMs

    internal void ResetForReuse()
    {
        Id = default; Kind = ScopeKind.Public; FirstNode = null; Bytes = default; NewestLeaf = null;
        LastUseTick = 0; RunningRequests = 0; WaitingRequests = 0; Retired = false; NodeCount = 0;
        LastUseMs = 0; Active = false;
    }
}

/// <summary>Interns <see cref="ScopeId"/> → int index; index 0 is the public scope.</summary>
internal sealed class ScopeTable
{
    private readonly List<ScopeRecord> _records = new();
    private readonly Dictionary<ScopeId, int> _byId = new();
    private readonly Stack<int> _free = new();

    internal ScopeTable()
    {
        _records.Add(new ScopeRecord { Id = ScopeId.Public, Kind = ScopeKind.Public, Index = 0, Live = true, Active = true });
    }

    /// <summary>Number of index slots (live or free), including the public slot.</summary>
    internal int Capacity => _records.Count;

    internal int LiveCount => _records.Count - _free.Count;

    /// <summary>
    /// Returns the index of <paramref name="id"/>, creating a record when the id is new or its
    /// previous record was retired. Public → 0. Interning an <see cref="ScopeKind.Unscoped"/> id that
    /// is already live throws: unscoped ids are fresh per request (I29).
    /// </summary>
    internal int Intern(ScopeId id, ScopeKind kind)
    {
        if (id.IsPublic) return 0;
        if (_byId.TryGetValue(id, out int existing))
        {
            ScopeRecord rec = _records[existing];
            if (!rec.Retired)
            {
                if (kind == ScopeKind.Unscoped || rec.Kind == ScopeKind.Unscoped)
                    throw new InvalidOperationException("An unscoped request's fresh scope id was interned twice (I29).");
                return existing;
            }
        }
        int index;
        ScopeRecord record;
        if (_free.Count > 0)
        {
            index = _free.Pop();
            record = _records[index];
            record.ResetForReuse();
        }
        else
        {
            index = _records.Count;
            record = new ScopeRecord();
            _records.Add(record);
        }
        record.Id = id; record.Kind = kind; record.Index = index; record.Live = true; record.Active = true;
        _byId[id] = index;
        return index;
    }

    internal bool TryGetIndex(ScopeId id, out int index)
    {
        if (id.IsPublic) { index = 0; return true; }
        if (_byId.TryGetValue(id, out index) && !_records[index].Retired)
            return true;
        index = -1;
        return false;
    }

    internal ScopeRecord this[int index] => _records[index];

    internal bool IsLive(int index) => index >= 0 && index < _records.Count && _records[index].Live;

    /// <summary>Recycles a retired scope that holds no node and no request.</summary>
    internal void Recycle(int index)
    {
        if (index <= 0) throw new ArgumentOutOfRangeException(nameof(index), "The public scope is never recycled.");
        ScopeRecord rec = _records[index];
        if (!rec.Live) return;
        if (!rec.Retired || rec.NodeCount != 0 || rec.RunningRequests != 0 || rec.WaitingRequests != 0)
            throw new InvalidOperationException("Only a retired scope with no nodes and no requests can be recycled.");
        if (_byId.TryGetValue(rec.Id, out int mapped) && mapped == index)
            _byId.Remove(rec.Id);
        rec.ResetForReuse();
        rec.Live = false;
        _free.Push(index);
    }

    internal IEnumerable<ScopeRecord> LiveRecords()
    {
        for (int i = 0; i < _records.Count; i++)
            if (_records[i].Live) yield return _records[i];
    }
}
