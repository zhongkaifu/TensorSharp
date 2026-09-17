// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

public class RadixKeyTests
{
    [Fact]
    public void TextAndMediaElements_AreSeparatedByTheSignBit()
    {
        Assert.Equal(42L, KeyElem.Text(42));
        Assert.False(KeyElem.IsMedia(KeyElem.Text(int.MaxValue)));
        Assert.Throws<ArgumentOutOfRangeException>(() => KeyElem.Text(-1));
        var id = MediaId256.FromContentId(new string('f', 64));
        long m = KeyElem.Media(id);
        Assert.True(KeyElem.IsMedia(m));
        Assert.Equal(unchecked((long)0xFFFF_FFFF_FFFF_FFFFUL), m);
        Assert.Equal(KeyElem.MediaBit | 0x0123456789abcdefL, KeyElem.Media(MediaId256.FromContentId(new string('0', 48) + "0123456789abcdef")));
    }

    [Fact]
    public void MediaId_ParsesLowercaseHex_AndFallsBackToAPathHash()
    {
        string hex = "00112233445566778899aabbccddeeff0123456789abcdeffedcba9876543210";
        var id = MediaId256.FromContentId(hex);
        Assert.Equal(0x0011223344556677UL, id.A);
        Assert.Equal(0x8899aabbccddeeffUL, id.B);
        Assert.Equal(0x0123456789abcdefUL, id.C);
        Assert.Equal(0xfedcba9876543210UL, id.D);
        Assert.Equal(hex, id.ToString());

        // Anything else (uppercase, wrong length, a path) → SHA-256("path:" + id).
        var upper = MediaId256.FromContentId(hex.ToUpperInvariant());
        Assert.Equal(MediaId256.FromDigest(SHA256.HashData(Encoding.UTF8.GetBytes("path:" + hex.ToUpperInvariant()))), upper);
        Assert.NotEqual(id, upper);
        var path = MediaId256.FromContentId("img:photo.png");
        Assert.Equal(MediaId256.FromDigest(SHA256.HashData(Encoding.UTF8.GetBytes("path:img:photo.png"))), path);
        var empty = MediaId256.FromContentId("");
        var nul = MediaId256.FromContentId(null);
        Assert.Equal(empty, nul);
        Assert.Equal(MediaId256.FromDigest(SHA256.HashData(Encoding.UTF8.GetBytes("path:"))), empty);
        Assert.True(empty == nul);
        Assert.False(empty != nul);
        Assert.True(id.Equals((object)MediaId256.FromContentId(hex)));
        Assert.Equal(id.GetHashCode(), MediaId256.FromContentId(hex).GetHashCode());
        Assert.Throws<ArgumentException>(() => MediaId256.FromDigest(new byte[31]));
    }

    [Fact]
    public void BuildPrompt_OverwritesSpanPositionsWithMediaKeys()
    {
        var pool = new KeyChunkPool();
        int[] tokens = { 1, 2, 999, 999, 999, 3, 4 };
        var spans = new (int, int, string)[] { (2, 5, "img:a") };
        KeyRope rope = RadixKeyBuilder.BuildPrompt(tokens, spans, pool, out MediaSpanRecord[] records);
        Assert.Single(records);
        long media = KeyElem.Media(MediaId256.FromContentId("img:a"));
        Assert.Equal(new long[] { 1, 2, media, media, media, 3, 4 }, Enumerable.Range(0, rope.Length).Select(i => rope[i]).ToArray());
        // Model input ids are untouched.
        Assert.Equal(999, tokens[2]);
        KeyRope noSpans = RadixKeyBuilder.BuildPrompt(tokens, (System.Collections.Generic.IReadOnlyList<(int, int, string)>?)null, pool, out MediaSpanRecord[] none);
        Assert.Empty(none);
        Assert.Equal(999L, noSpans[2]);
    }

    [Theory]
    [InlineData(-1, 2)]      // outside
    [InlineData(5, 9)]       // past the prompt
    [InlineData(3, 3)]       // empty
    public void BuildPrompt_RejectsInvalidSpans(int start, int end)
    {
        var pool = new KeyChunkPool();
        Assert.Throws<ArgumentException>(() =>
            RadixKeyBuilder.BuildPrompt(Tk.Seq(0, 8), new (int, int, string)[] { (start, end, "x") }, pool, out _));
    }

    [Fact]
    public void BuildPrompt_RejectsUnsortedOrOverlappingSpans_AndNegativeTokens()
    {
        var pool = new KeyChunkPool();
        Assert.Throws<ArgumentException>(() =>
            RadixKeyBuilder.BuildPrompt(Tk.Seq(0, 10), new (int, int, string)[] { (5, 7, "a"), (1, 3, "b") }, pool, out _));
        Assert.Throws<ArgumentException>(() =>
            RadixKeyBuilder.BuildPrompt(Tk.Seq(0, 10), new (int, int, string)[] { (1, 5, "a"), (4, 6, "b") }, pool, out _));
        // Token ids must lie in [0, 2^31): a negative id is a submit failure, never a silent miss.
        var ex = Assert.Throws<ArgumentException>(() =>
            RadixKeyBuilder.BuildPrompt(new[] { 1, -5, 2 }, Array.Empty<MediaSpanRecord>(), pool));
        Assert.Contains("negative", ex.Message);
        // A negative id inside a span is replaced by the media key, so it is accepted.
        KeyRope ok = RadixKeyBuilder.BuildPrompt(new[] { 1, -5, 2 }, new[] { Tk.Span(1, 2, "img") }, pool);
        Assert.True(KeyElem.IsMedia(ok[1]));
        // Adjacent spans are valid.
        RadixKeyBuilder.BuildPrompt(Tk.Seq(0, 10), new (int, int, string)[] { (1, 3, "a"), (3, 6, "b") }, pool, out MediaSpanRecord[] adj);
        Assert.Equal(2, adj.Length);
        Assert.Throws<ArgumentNullException>(() => RadixKeyBuilder.BuildPrompt(null!, Array.Empty<MediaSpanRecord>(), pool));
        Assert.Throws<ArgumentNullException>(() => RadixKeyBuilder.BuildPrompt(null!, (System.Collections.Generic.IReadOnlyList<(int, int, string)>?)null, pool, out _));
    }

    [Fact]
    public void AppendOutput_WritesOnlyPastTheLength()
    {
        var pool = new KeyChunkPool();
        KeyRope rope = RadixKeyBuilder.BuildPrompt(new[] { 1, 2, 3 }, Array.Empty<MediaSpanRecord>(), pool);
        RadixKeyBuilder.AppendOutput(rope, 3, new[] { 7, 8 }, pool);
        Assert.Equal(5, rope.Length);
        // Re-appending an overlapping range writes nothing already present.
        RadixKeyBuilder.AppendOutput(rope, 4, new[] { 99, 9 }, pool);
        Assert.Equal(6, rope.Length);
        Assert.Equal(new long[] { 1, 2, 3, 7, 8, 9 }, Enumerable.Range(0, 6).Select(i => rope[i]).ToArray());
        Assert.Throws<ArgumentOutOfRangeException>(() => RadixKeyBuilder.AppendOutput(rope, 7, new[] { 1 }, pool));
        Assert.Throws<ArgumentException>(() => RadixKeyBuilder.AppendOutput(rope, 6, new[] { -1 }, pool));
        Assert.Throws<ArgumentNullException>(() => RadixKeyBuilder.AppendOutput(null!, 0, new[] { 1 }, pool));
    }

    [Fact]
    public void KeyRope_CommonPrefixAcrossChunkBoundaries()
    {
        var pool = new KeyChunkPool();
        int n = KeyChunk.Size * 2 + 100;
        int[] a = Tk.Seq(0, n);
        int[] b = (int[])a.Clone();
        b[KeyChunk.Size + 5] = 123456;   // differ just past the first chunk boundary
        KeyRope ra = RadixKeyBuilder.BuildPrompt(a, Array.Empty<MediaSpanRecord>(), pool);
        KeyRope rb = RadixKeyBuilder.BuildPrompt(b, Array.Empty<MediaSpanRecord>(), pool);
        Assert.Equal(3, ra.ChunkCount);
        Assert.Equal(KeyChunk.Size + 5, KeyCompare.CommonPrefixLength(ra, rb, n));
        Assert.Equal(n, KeyCompare.CommonPrefixLength(ra, ra, n));
        Assert.Equal(10, KeyCompare.CommonPrefixLength(ra, ra, 10));
        // Offsets on both sides, straddling a boundary.
        var edge = new KeySlice(ra, KeyChunk.Size - 3, 50);
        Assert.Equal(50, KeyCompare.CommonPrefixLength(edge, 0, ra, KeyChunk.Size - 3, 1000));
        Assert.Equal(40, KeyCompare.CommonPrefixLength(edge, 10, ra, KeyChunk.Size + 7, 1000));
        Assert.Equal(0, KeyCompare.CommonPrefixLength(edge, 50, ra, 0, 10));
        Assert.Equal(ra[KeyChunk.Size - 3], edge[0]);
        Assert.False(edge.IsEmpty);
        // Segment never crosses a chunk.
        Assert.Equal(3, ra.Segment(KeyChunk.Size - 3, 100).Length);
        Assert.Equal(0, ra.Segment(n, 10).Length);
        // Chunks come back to the pool.
        ra.ReturnChunks(pool);
        Assert.True(ra.Disposed);
        Assert.Equal(3, pool.FreeCount);
        ra.ReturnChunks(pool);   // idempotent
        Assert.Throws<ObjectDisposedException>(() => ra.Append(1, pool));
        Assert.Throws<ObjectDisposedException>(() => ra.Append(new long[] { 1 }, pool));
        KeyRope reused = KeyRope.FromKeys(new long[] { 5, 6 }, pool);
        Assert.Equal(2, pool.FreeCount);
        Assert.Equal(6L, reused[1]);
        Assert.True(pool.Allocated >= 6);
    }

    [Fact]
    public void KeyChunk_StaysBelowTheLargeObjectHeap()
    {
        Assert.True(KeyChunk.Size * sizeof(long) < 85_000);
        var pool = new KeyChunkPool(maxFree: 1);
        KeyChunk c1 = pool.Rent(), c2 = pool.Rent();
        pool.Return(c1);
        pool.Return(c2);            // beyond the bound: dropped
        pool.Return(null!);
        Assert.Equal(1, pool.FreeCount);
        Assert.True(GC.GetGeneration(new KeyChunk().Data) < 2, "a key chunk must not be allocated on the LOH");
    }

    [Fact]
    public void RopeCompaction_CopiesSurvivingSlicesAndReturnsChunks()
    {
        var host = new FakePageHost(Tk.B);
        PrefixTree t = Tk.Tree(host: host, scopedMax: 0);
        int s = Tk.Scope(t);
        int big = KeyChunk.Size + 200;
        int[] tokens = Tk.Seq(1, big);
        KeyRope key = Tk.Key(t, tokens);
        // A short surviving node and a long leaf, both sliced from one rope.
        RadixNode shortNode = Tk.Put(t, key, 10, s);
        RadixNode longNode = Tk.Put(t, key, big, s);
        Assert.Same(key, longNode.Edge.Rope);
        t.ReleaseRopeOwner(key);
        Assert.False(key.Disposed);
        // Evicting the long leaf leaves 10 of ~8.4k tokens referenced: the rope is compacted.
        t.DeleteLeafCascade(longNode, ReleaseReason.Evicted);
        Assert.Equal(1, t.Counters.RopeCompactions);
        Assert.True(key.Disposed);
        Assert.NotSame(key, shortNode.Edge.Rope);
        Assert.Equal(10, shortNode.Edge.Rope.Length);
        // Right-sized: 10 surviving elements do not pin a whole 64 KB chunk, and the sealed rope cannot grow.
        Assert.Equal(1, shortNode.Edge.Rope.ChunkCount);
        Assert.Equal(10, shortNode.Edge.Rope.Chunks[0].Data.Length);
        Assert.Throws<InvalidOperationException>(() => shortNode.Edge.Rope.Append(7, t.KeyPool));
        Assert.Equal(Enumerable.Range(1, 10).Select(i => (long)i), Enumerable.Range(0, 10).Select(i => shortNode.Edge[i]));
        Tk.Valid(t);
        // The compacted rope still matches.
        KeyRope probe = Tk.Key(t, tokens);
        MatchPlan plan = Tk.Plan(t, Tk.Req(probe, s));
        Assert.Equal(10, plan.Length);
        // Deleting the last slice returns the fresh rope's chunks.
        KeyRope fresh = shortNode.Edge.Rope;
        t.DeleteLeafCascade(shortNode, ReleaseReason.Evicted);
        Assert.True(fresh.Disposed);
        Tk.Valid(t);
        // A sealed rope spanning chunks: whole chunks come from the pool, only the tail is right-sized,
        // and a right-sized chunk is never pooled.
        var pool = new KeyChunkPool();
        KeyRope sealedRope = KeyRope.Sealed(KeyChunk.Size + 3, pool);
        sealedRope.Append(Enumerable.Range(0, KeyChunk.Size + 3).Select(i => (long)i).ToArray(), pool);
        Assert.Equal(2, sealedRope.ChunkCount);
        Assert.Equal(KeyChunk.Size, sealedRope.Chunks[0].Data.Length);
        Assert.Equal(3, sealedRope.Chunks[1].Data.Length);
        Assert.Equal(KeyChunk.Size + 2, sealedRope[KeyChunk.Size + 2]);
        sealedRope.ReturnChunks(pool);
        Assert.Equal(1, pool.FreeCount);
    }

    [Fact]
    public void ReleasingTheOwnerOfAnUnreferencedRope_ReturnsItsChunks()
    {
        PrefixTree t = Tk.Tree();
        KeyRope key = Tk.Key(t, Tk.Seq(1, 20));
        int before = t.KeyPool.FreeCount;
        t.ReleaseRopeOwner(key);
        Assert.True(key.Disposed);
        Assert.Equal(before + 1, t.KeyPool.FreeCount);
        t.ReleaseRopeOwner(key);   // idempotent
        t.ReleaseRopeOwner(null!);
    }
}

public class CacheScopeTests
{
    [Fact]
    public void ScopeId_FromHex_AcceptsPhase0Widths()
    {
        ScopeId wide = ScopeId.FromHex("0123456789abcdef0011223344556677");
        Assert.Equal(new UInt128(0x0123456789abcdefUL, 0x0011223344556677UL), wide.Value);
        Assert.Equal("01234567", wide.ToLogToken());
        Assert.Equal("01234567", wide.ToString());
        ScopeId narrow = ScopeId.FromHex("00000000000000ff");
        Assert.Equal(new UInt128(0, 0xff), narrow.Value);
        Assert.False(narrow.IsPublic);
        Assert.True(ScopeId.Public.IsPublic);
        Assert.Equal("public", ScopeId.Public.ToString());
        Assert.Throws<ArgumentException>(() => ScopeId.FromHex("abc"));
        Assert.Throws<ArgumentException>(() => ScopeId.FromHex(new string('0', 32)));
        Assert.Throws<ArgumentException>(() => ScopeId.FromHex(new string('z', 32)));
        Assert.Throws<ArgumentException>(() => ScopeId.FromHex(new string('z', 16)));
        Assert.Throws<ArgumentNullException>(() => ScopeId.FromHex(null!));
    }

    [Fact]
    public void ScopeId_NewFresh_IsRandomAndNeverPublic()
    {
        var ids = Enumerable.Range(0, 1000).Select(_ => ScopeId.NewFresh()).ToHashSet();
        Assert.Equal(1000, ids.Count);
        Assert.DoesNotContain(ScopeId.Public, ids);
        Assert.All(ids, id => Assert.Equal(8, id.ToLogToken().Length));
    }

    [Fact]
    public void ScopeTable_InternsRecyclesAndIsolatesUnscoped()
    {
        var table = new ScopeTable();
        Assert.Equal(0, table.Intern(ScopeId.Public, ScopeKind.Public));
        ScopeId a = ScopeId.NewFresh();
        int ia = table.Intern(a, ScopeKind.Session);
        Assert.Equal(ia, table.Intern(a, ScopeKind.Session));
        Assert.True(table.TryGetIndex(a, out int found) && found == ia);
        Assert.True(table.TryGetIndex(ScopeId.Public, out int pub) && pub == 0);
        Assert.False(table.TryGetIndex(ScopeId.NewFresh(), out _));
        ScopeId u = ScopeId.NewFresh();
        table.Intern(u, ScopeKind.Unscoped);
        Assert.Throws<InvalidOperationException>(() => table.Intern(u, ScopeKind.Unscoped));   // I29
        Assert.Throws<InvalidOperationException>(() => table.Recycle(ia));                      // not retired
        Assert.Throws<ArgumentOutOfRangeException>(() => table.Recycle(0));
        table[ia].Retired = true;
        Assert.False(table.TryGetIndex(a, out _));
        int again = table.Intern(a, ScopeKind.Session);                                          // a retired id gets a new record
        Assert.NotEqual(ia, again);
        table.Recycle(ia);
        Assert.False(table.IsLive(ia));
        table.Recycle(ia);                                                                       // idempotent
        int reused = table.Intern(ScopeId.NewFresh(), ScopeKind.Lineage);
        Assert.Equal(ia, reused);
        Assert.Equal(table.Capacity, table.LiveCount);
        Assert.Equal(table.LiveCount, table.LiveRecords().Count());
        Assert.False(table.IsLive(-1));
        Assert.False(table.IsLive(1000));
    }
}
