// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

public class InflightTableTests
{
    private sealed class Req : IInflightRequest
    {
        public bool IsRunning { get; set; }
        public int NumComputedTokens { get; set; }
        public KeyRope Key { get; init; } = null!;
        public InflightEntry? FollowingLeader { get; set; }
    }

    private const int P = 1024;
    private static readonly KeyChunkPool Pool = new();

    private static Req NewReq(int[] tokens) => new() { Key = RadixKeyBuilder.BuildPrompt(tokens, Array.Empty<MediaSpanRecord>(), Pool) };

    private static int[] Prompt(int suffixSeed) => Tk.Cat(Tk.Seq(1, P), Tk.Seq(10_000 + suffixSeed * 100, 50));

    [Fact]
    public void Follow_SkipNotBlock_Completion()
    {
        var table = new InflightTable();
        Req leader = NewReq(Prompt(0));
        ulong hash = InflightTable.ComputeHash(leader.Key, P);
        Assert.Equal(hash, InflightTable.ComputeHash(NewReq(Prompt(1)).Key, P));
        Assert.NotEqual(hash, InflightTable.ComputeHash(leader.Key, P + 1));
        Assert.False(table.TryFollow(NewReq(Prompt(1)), P, hash, step: 1));   // no entry yet
        leader.IsRunning = true;
        InflightEntry entry = table.RegisterLeader(leader, P, hash);
        Assert.Same(entry, table.RegisterLeader(NewReq(Prompt(9)), P, hash));
        Assert.True(table.TryGet(P, hash, out _));

        Req f1 = NewReq(Prompt(1)), f2 = NewReq(Prompt(2));
        Assert.True(table.TryFollow(f1, P, hash, 1));
        Assert.True(table.TryFollow(f2, P, hash, 1));
        Assert.True(table.TryFollow(f1, P, hash, 2));           // already following: no second join
        Assert.Equal(2, table.Joins);
        Assert.Same(entry, f1.FollowingLeader);
        Assert.False(table.TryFollow(leader, P, hash, 1));      // the leader never follows itself
        // Skip, not block: a request with a different prefix is not affected.
        Req unrelated = NewReq(Tk.Seq(50_000, P + 10));
        Assert.False(table.TryFollow(unrelated, P, InflightTable.ComputeHash(unrelated.Key, P), 1));
        Assert.Null(unrelated.FollowingLeader);

        // Completion: the entry goes away and every follower re-plans.
        Assert.Same(entry, table.OnPublished(P, hash));
        Assert.Null(f1.FollowingLeader);
        Assert.Null(f2.FollowingLeader);
        Assert.Equal(0, table.Count);
        Assert.Null(table.OnPublished(P, hash));
        Assert.False(entry.IsActive(3));
    }

    [Theory]
    [InlineData((int)LeaderGoneReason.Aborted)]
    [InlineData((int)LeaderGoneReason.Preempted)]
    [InlineData((int)LeaderGoneReason.Errored)]
    [InlineData((int)LeaderGoneReason.Finished)]
    public void Promotion_OnLeaderGone_ReleasesTheOldestFollower(int reason)
    {
        var table = new InflightTable();
        Req leader = NewReq(Prompt(0));
        leader.IsRunning = true;
        ulong hash = InflightTable.ComputeHash(leader.Key, P);
        InflightEntry entry = table.RegisterLeader(leader, P, hash);
        Req f1 = NewReq(Prompt(1)), f2 = NewReq(Prompt(2));
        table.TryFollow(f1, P, hash, 1);
        table.TryFollow(f2, P, hash, 1);
        IInflightRequest? promoted = table.OnLeaderGone(leader, (LeaderGoneReason)reason);
        Assert.Same(f1, promoted);
        Assert.Null(f1.FollowingLeader);
        Assert.False(entry.IsActive(2));                        // the others stop waiting too
        Assert.Equal(1, table.Promotions);
        // The promoted request becomes the next leader at its commit; the other follower joins it.
        f1.IsRunning = true;
        InflightEntry next = table.RegisterLeader(f1, P, hash);
        Assert.NotSame(entry, next);
        Assert.True(table.TryFollow(f2, P, hash, 2));
        Assert.Same(next, f2.FollowingLeader);
        Assert.Null(table.OnLeaderGone(NewReq(Prompt(5)), LeaderGoneReason.Aborted));
        Assert.Null(table.OnLeaderGone(null!, LeaderGoneReason.Aborted));
    }

    [Fact]
    public void Promotion_OnStall_AfterEightScheduledStepsWithoutProgress()
    {
        var table = new InflightTable();
        Req leader = NewReq(Prompt(0));
        leader.IsRunning = true;
        ulong hash = InflightTable.ComputeHash(leader.Key, P);
        InflightEntry entry = table.RegisterLeader(leader, P, hash);
        Req f = NewReq(Prompt(1));
        table.TryFollow(f, P, hash, 1);
        // Progress resets the counter.
        for (int step = 0; step < 7; step++) table.OnLeaderScheduled(leader, step);
        leader.NumComputedTokens = 512;
        table.OnLeaderScheduled(leader, 7);
        Assert.Equal(0, entry.ScheduledStepsWithoutProgress);
        Assert.True(entry.IsActive(7));
        // Steps in which the leader is not scheduled do not count (the caller does not report them).
        for (int step = 8; step < 15; step++) table.OnLeaderScheduled(leader, step);
        Assert.True(entry.IsActive(15));
        Assert.NotNull(f.FollowingLeader);
        table.OnLeaderScheduled(leader, 15);                    // the 8th scheduled step without progress
        Assert.False(entry.IsActive(16));
        Assert.Null(f.FollowingLeader);
        Assert.Equal(0, table.Count);
        table.OnLeaderScheduled(leader, 16);                    // unknown leader: ignored
        // A leader that is not running is not active either.
        Req stopped = NewReq(Prompt(3));
        InflightEntry e2 = table.RegisterLeader(stopped, P, hash);
        Assert.False(e2.IsActive(1));
        Assert.False(table.TryFollow(NewReq(Prompt(4)), P, hash, 1));
    }

    [Fact]
    public void DiskRestore_IsALeader_AndItsFailurePromotes()
    {
        var table = new InflightTable();
        Req any = NewReq(Prompt(0));
        ulong hash = InflightTable.ComputeHash(any.Key, P);
        InflightEntry restore = table.RegisterDiskRestore(any.Key, P, hash);
        Assert.Equal(InflightKind.DiskRestore, restore.Kind);
        Assert.Same(restore, table.RegisterDiskRestore(any.Key, P, hash));
        Req f1 = NewReq(Prompt(1)), f2 = NewReq(Prompt(2));
        Assert.True(table.TryFollow(f1, P, hash, 1));
        Assert.True(table.TryFollow(f2, P, hash, 1));
        Assert.Same(f1, table.OnRestoreFailed(P, hash));
        Assert.False(restore.IsActive(2));
        Assert.Null(table.OnRestoreFailed(P, hash));
        // A prefill entry is not a restore.
        Req leader = NewReq(Prompt(3));
        leader.IsRunning = true;
        table.RegisterLeader(leader, P, hash);
        Assert.Null(table.OnRestoreFailed(P, hash));
        Assert.NotSame(restore, table.RegisterDiskRestore(any.Key, P + 1, hash));
    }

    [Fact]
    public void HashCollision_IsRejectedByTheFullCompare()
    {
        var table = new InflightTable();
        Req leader = NewReq(Prompt(0));
        leader.IsRunning = true;
        const ulong forged = 12345;
        table.RegisterLeader(leader, P, forged);
        Req impostor = NewReq(Tk.Cat(Tk.Seq(1, P - 1), new[] { 777 }, Tk.Seq(1, 20)));   // differs at P-1
        Assert.False(table.TryFollow(impostor, P, forged, 1));
        Assert.Equal(1, table.HashCollisions);
        Assert.Null(impostor.FollowingLeader);
        Req shortKey = NewReq(Tk.Seq(1, 10));
        Assert.False(table.TryFollow(shortKey, P, forged, 1));
        Assert.Equal(2, table.HashCollisions);
        Assert.Throws<ArgumentNullException>(() => table.TryFollow(null!, P, forged, 1));
        Assert.Throws<ArgumentNullException>(() => table.RegisterLeader(null!, P, forged));
        // A follower that leaves is forgotten.
        Req f = NewReq(Prompt(2));
        Assert.True(table.TryFollow(f, P, forged, 1));
        table.OnFollowerGone(f);
        Assert.Null(f.FollowingLeader);
        table.OnFollowerGone(f);
    }

    [Fact]
    public void XxHash64_MatchesTheReferenceVectors()
    {
        // XXH64 of the little-endian bytes of the longs, seed 0, computed with the reference algorithm.
        var pool = new KeyChunkPool();
        KeyRope empty = KeyRope.FromKeys(ReadOnlySpan<long>.Empty, pool);
        Assert.Equal(0xEF46DB3751D8E999UL, XxHash64Longs.Hash(empty, 0, 0));      // XXH64("") seed 0
        KeyRope zeros = KeyRope.FromKeys(new long[4], pool);
        Assert.Equal(0xF6E9BE5D70632CF5UL, XxHash64Longs.Hash(zeros, 4, 0));      // XXH64(32 zero bytes)
        Assert.Equal(0xBB3302E8A9608868UL, XxHash64Longs.Hash(zeros, 3, 0));      // XXH64(24 zero bytes)
        Assert.Equal(XxHash64Longs.Hash(zeros, 4, 0), XxHash64Longs.Hash(zeros, 99, 0));   // clamped to the key
        KeyRope mixed = KeyRope.FromKeys(new long[] { 1, 2, 3, 4, -5 }, pool);
        Assert.Equal(0xDF725DB067551579UL, XxHash64Longs.Hash(mixed, 5, 0));
    }
}
