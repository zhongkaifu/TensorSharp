// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace TensorSharp.Runtime.Scheduling.PrefixCache;

internal enum InflightKind : byte { Prefill, DiskRestore }

/// <summary>Why a dedup leader stopped leading (§5.13).</summary>
internal enum LeaderGoneReason : byte { Finished, Aborted, Preempted, Errored, Stalled, RestoreFailed }

/// <summary>
/// What the dedup table reads from a request. M1 deviation: the design names <c>SequenceState</c>;
/// the tree library has no product references, so the coordinator (M3) adapts its request state.
/// </summary>
internal interface IInflightRequest
{
    bool IsRunning { get; }
    int NumComputedTokens { get; }
    /// <summary>The prompt key (at least <c>P</c> elements), compared on join.</summary>
    KeyRope Key { get; }
    /// <summary>The entry this request follows (null = not following). Set and cleared by the table.</summary>
    InflightEntry? FollowingLeader { get; set; }
}

/// <summary>One in-flight public prefix: its leader (or disk restore) and the requests waiting on it.</summary>
internal sealed class InflightEntry
{
    public int P; public ulong Hash;                      // XxHash64 over key[0..P), computed once per request when P ≥ 512
    public InflightKind Kind;                             // Prefill | DiskRestore
    public IInflightRequest? Leader;                      // null for DiskRestore
    public KeyRope? LeaderKey;                            // the key followers are verified against
    public readonly List<IInflightRequest> Followers = new();
    public int LeaderComputedAtProgress; public int ScheduledStepsWithoutProgress;
    public bool RestoreInFlight;
    public bool Removed;

    internal const int StallSteps = 8;

    /// <summary>Leader running and progressing (fewer than 8 scheduled steps without progress), or a restore in flight.</summary>
    public bool IsActive(long step)
    {
        if (Removed) return false;
        if (Kind == InflightKind.DiskRestore) return RestoreInFlight;
        return Leader is not null && Leader.IsRunning && ScheduledStepsWithoutProgress < StallSteps;
    }
}

/// <summary>
/// In-flight dedup of identical public prefixes (P11, DEC-25, §5.13): a follower is skipped, not
/// blocking, while its leader is alive and progressing; a leader that stops leading promotes the
/// oldest follower in the same step. There is no wall-clock timer.
/// </summary>
internal sealed class InflightTable
{
    internal const int MinDedupTokens = 512;
    private readonly Dictionary<(int P, ulong Hash), InflightEntry> _entries = new();
    private readonly Dictionary<IInflightRequest, InflightEntry> _byLeader = new(ReferenceEqualityComparer.Instance);

    internal int Count => _entries.Count;
    internal long HashCollisions { get; private set; }
    internal long Joins { get; private set; }
    internal long Promotions { get; private set; }

    /// <summary>XxHash64 of <c>key[0..P)</c> (element bytes, little-endian).</summary>
    internal static ulong ComputeHash(KeyRope key, int p) => XxHash64Longs.Hash(key, p, seed: 0);

    internal bool TryGet(int p, ulong hash, out InflightEntry entry) => _entries.TryGetValue((p, hash), out entry!);

    /// <summary>
    /// Registers <paramref name="leader"/> as the prefill leader of <c>(P, hash)</c> (at admission commit).
    /// Returns the existing entry when one is already registered.
    /// </summary>
    internal InflightEntry RegisterLeader(IInflightRequest leader, int p, ulong hash)
    {
        if (leader is null) throw new ArgumentNullException(nameof(leader));
        if (_entries.TryGetValue((p, hash), out InflightEntry? existing)) return existing;
        var entry = new InflightEntry
        {
            P = p, Hash = hash, Kind = InflightKind.Prefill, Leader = leader, LeaderKey = leader.Key,
            LeaderComputedAtProgress = leader.NumComputedTokens,
        };
        _entries[(p, hash)] = entry;
        _byLeader[leader] = entry;
        return entry;
    }

    /// <summary>A restore-lane job registers as the leader of its prefix (M7c).</summary>
    internal InflightEntry RegisterDiskRestore(KeyRope key, int p, ulong hash)
    {
        if (_entries.TryGetValue((p, hash), out InflightEntry? existing)) return existing;
        var entry = new InflightEntry { P = p, Hash = hash, Kind = InflightKind.DiskRestore, LeaderKey = key, RestoreInFlight = true };
        _entries[(p, hash)] = entry;
        return entry;
    }

    /// <summary>
    /// Following (§5.13): when an active entry exists for <c>(P, hash)</c> and the request's key equals
    /// the leader's over <c>[0, P)</c> (one vectorized compare, so a hash collision cannot join), the
    /// request follows it and the caller defers its admission. Returns true when following.
    /// </summary>
    internal bool TryFollow(IInflightRequest seq, int p, ulong hash, long step)
    {
        if (seq is null) throw new ArgumentNullException(nameof(seq));
        if (!_entries.TryGetValue((p, hash), out InflightEntry? entry)) return false;
        if (ReferenceEquals(entry.Leader, seq)) return false;
        if (!entry.IsActive(step)) return false;
        if (!ReferenceEquals(seq.FollowingLeader, entry))
        {
            KeyRope? leaderKey = entry.LeaderKey;
            if (leaderKey is null || seq.Key is null || seq.Key.Length < p || leaderKey.Length < p
                || KeyCompare.CommonPrefixLength(leaderKey, seq.Key, p) != p)
            {
                HashCollisions++;
                return false;
            }
            seq.FollowingLeader = entry;
            entry.Followers.Add(seq);
            Joins++;
        }
        return true;
    }

    /// <summary>
    /// Progress (§5.13): called once per executed step in which the leader was scheduled. The counter
    /// resets when <c>NumComputedTokens</c> advanced and increments otherwise; a stall promotes.
    /// </summary>
    internal void OnLeaderScheduled(IInflightRequest leader, long step)
    {
        if (!_byLeader.TryGetValue(leader, out InflightEntry? entry)) return;
        if (leader.NumComputedTokens > entry.LeaderComputedAtProgress)
        {
            entry.LeaderComputedAtProgress = leader.NumComputedTokens;
            entry.ScheduledStepsWithoutProgress = 0;
        }
        else if (++entry.ScheduledStepsWithoutProgress >= InflightEntry.StallSteps)
        {
            Remove(entry, promote: true);
        }
    }

    /// <summary>Completion: the public prefix is published. Followers re-match next step (their plans are stale).</summary>
    internal InflightEntry? OnPublished(int p, ulong hash)
    {
        if (!_entries.TryGetValue((p, hash), out InflightEntry? entry)) return null;
        Remove(entry, promote: false);
        foreach (IInflightRequest f in entry.Followers)
            if (ReferenceEquals(f.FollowingLeader, entry)) f.FollowingLeader = null;
        return entry;
    }

    /// <summary>
    /// Promotion (§5.13): the leader finished before P, aborted, was preempted, errored or stalled.
    /// The entry is removed and the oldest follower is released in the same step so it is admitted
    /// normally (and becomes the next leader at its commit). Returns the promoted follower.
    /// </summary>
    internal IInflightRequest? OnLeaderGone(IInflightRequest leader, LeaderGoneReason reason)
    {
        if (leader is null || !_byLeader.TryGetValue(leader, out InflightEntry? entry)) return null;
        return Remove(entry, promote: true);
    }

    /// <summary>A restore lane job failed: remove its entry and promote the oldest follower.</summary>
    internal IInflightRequest? OnRestoreFailed(int p, ulong hash)
    {
        if (!_entries.TryGetValue((p, hash), out InflightEntry? entry) || entry.Kind != InflightKind.DiskRestore) return null;
        entry.RestoreInFlight = false;
        return Remove(entry, promote: true);
    }

    /// <summary>A follower left the queue (abort, error): forget it.</summary>
    internal void OnFollowerGone(IInflightRequest follower)
    {
        InflightEntry? entry = follower.FollowingLeader;
        if (entry is null) return;
        entry.Followers.Remove(follower);
        follower.FollowingLeader = null;
    }

    private IInflightRequest? Remove(InflightEntry entry, bool promote)
    {
        entry.Removed = true;
        _entries.Remove((entry.P, entry.Hash));
        if (entry.Leader is not null) _byLeader.Remove(entry.Leader);
        if (!promote) return null;
        IInflightRequest? oldest = null;
        foreach (IInflightRequest f in entry.Followers)
        {
            if (!ReferenceEquals(f.FollowingLeader, entry)) continue;
            oldest = f;
            break;
        }
        if (oldest is not null)
        {
            oldest.FollowingLeader = null;
            Promotions++;
        }
        return oldest;
    }
}

/// <summary>XXH64 over the little-endian bytes of a key prefix, streamed across rope chunks.</summary>
internal static class XxHash64Longs
{
    private const ulong Prime1 = 11400714785074694791UL;
    private const ulong Prime2 = 14029467366897019727UL;
    private const ulong Prime3 = 1609587929392839161UL;
    private const ulong Prime4 = 9650029242287828579UL;
    private const ulong Prime5 = 2870177450012600261UL;

    internal static ulong Hash(KeyRope key, int count, ulong seed)
    {
        count = Math.Min(count, key.Length);
        ulong h;
        int i = 0;
        if (count >= 4)
        {
            ulong v1 = seed + Prime1 + Prime2, v2 = seed + Prime2, v3 = seed, v4 = seed - Prime1;
            for (; i + 4 <= count; i += 4)
            {
                v1 = Round(v1, (ulong)key[i]);
                v2 = Round(v2, (ulong)key[i + 1]);
                v3 = Round(v3, (ulong)key[i + 2]);
                v4 = Round(v4, (ulong)key[i + 3]);
            }
            h = BitOperations.RotateLeft(v1, 1) + BitOperations.RotateLeft(v2, 7) + BitOperations.RotateLeft(v3, 12) + BitOperations.RotateLeft(v4, 18);
            h = MergeRound(h, v1); h = MergeRound(h, v2); h = MergeRound(h, v3); h = MergeRound(h, v4);
        }
        else
        {
            h = seed + Prime5;
        }
        h += (ulong)count * 8UL;
        for (; i < count; i++)
        {
            ulong k1 = Round(0, (ulong)key[i]);
            h ^= k1;
            h = BitOperations.RotateLeft(h, 27) * Prime1 + Prime4;
        }
        h ^= h >> 33; h *= Prime2; h ^= h >> 29; h *= Prime3; h ^= h >> 32;
        return h;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Round(ulong acc, ulong input)
    {
        acc += input * Prime2;
        acc = BitOperations.RotateLeft(acc, 31);
        return acc * Prime1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MergeRound(ulong acc, ulong val)
    {
        val = Round(0, val);
        acc ^= val;
        return acc * Prime1 + Prime4;
    }
}
