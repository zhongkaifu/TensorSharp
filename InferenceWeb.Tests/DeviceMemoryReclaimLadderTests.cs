// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// What a CUDA allocator does when the device is full (issue #376: one prompt step of Qwen3.8-27B at a 24K context spent
// 1,091 ms in 1,614 cuMemFree calls of the old full GC and drain). The ladder runs here over the real pool and a fake
// device of fixed capacity, so the tests see which blocks go back to the "driver", in which order, and which rung made
// the room, without a GPU.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Cuda;
using Xunit;

namespace InferenceWeb.Tests;

public class DeviceMemoryReclaimLadderTests
{
    private const long MiB = 1L << 20;
    private const long GiB = 1L << 30;

    /// <summary>A device of fixed capacity: an allocation past it fails as cuMemAlloc does (zero), a free gives its bytes
    /// back; the pool over it is the allocator's, and the ladder is wired to both as CudaAllocator wires it.</summary>
    private sealed class Rig
    {
        private long _next = 0x10000;
        private readonly Dictionary<IntPtr, long> _live = new();
        public readonly long Capacity;
        public long Used;
        public readonly List<string> Events = new();
        public readonly List<long> Frees = new();
        public readonly List<string> Warnings = new();
        public readonly List<IDeviceMemoryReclaimer> Reclaimers = new();
        public readonly List<string> Trace = new();
        public bool Capturing;
        /// <summary>The synchronize throws, as the real one does while a graph is captured on the allocator.</summary>
        public bool SyncThrows;
        public int Collections;
        /// <summary>What a full GC's finalizers release (a view chain's storage), when set.</summary>
        public Action OnCollect;

        public readonly CudaDeviceMemoryPool Pool;
        public readonly DeviceMemoryReclaimLadder Ladder;

        public Rig(long capacity, long freeReserve = 0)
        {
            Capacity = capacity;
            Pool = new CudaDeviceMemoryPool(maxCachedBytes: 256 * MiB, maxCachedBlockBytes: 256 * MiB, enabled: true,
                backingAllocate: Allocate, backingFree: Free, shardCount: 1, largeCachedBytesCap: 64 * GiB);
            Ladder = new DeviceMemoryReclaimLadder(
                TryAllocate,
                () =>
                {
                    if (SyncThrows) throw new CudaGraphCaptureAbortedException("capturing");
                    Events.Add("sync");
                },
                bytes => Log("trim", Pool.TrimLarge(bytes)),
                () => Log("drain-small", Pool.DrainSmall()),
                () => Reclaimers,
                () => Capturing,
                () =>
                {
                    Events.Add("gc+drain");
                    Collections++;
                    OnCollect?.Invoke();
                    Pool.DrainAndFree();
                },
                new ReportOnce(Warnings.Add),
                deviceId: 3,
                Trace.Add,
                freeReserve);
        }

        private long Log(string what, long bytes)
        {
            Events.Add($"{what} {bytes / MiB}");
            return bytes;
        }

        public IntPtr TryAllocate(long bytes)
        {
            if (Used + bytes > Capacity)
            {
                Events.Add($"alloc {bytes / MiB} failed");
                return IntPtr.Zero;
            }
            Used += bytes;
            var ptr = new IntPtr(_next += 0x10000);
            _live[ptr] = bytes;
            Events.Add($"alloc {bytes / MiB}");
            return ptr;
        }

        private IntPtr Allocate(long bytes)
        {
            IntPtr ptr = Ladder.Allocate(bytes);
            if (ptr == IntPtr.Zero) throw new OutOfMemoryException($"fake device: {bytes} B");
            return ptr;
        }

        private void Free(IntPtr ptr)
        {
            Used -= _live[ptr];
            Frees.Add(_live[ptr]);
            _live.Remove(ptr);
        }

        public (IntPtr Ptr, long Bytes) Rent(long bytes) => (Pool.Rent(bytes, out long b), b);

        public void Return((IntPtr Ptr, long Bytes) block) => Pool.Return(block.Ptr, block.Bytes);

        /// <summary>Rent each size, then return them all to the cache.</summary>
        public void Cache(params long[] sizes)
        {
            var blocks = sizes.Select(Rent).ToList();
            blocks.ForEach(Return);
        }

        public void Clear()
        {
            Events.Clear();
            Frees.Clear();
        }
    }

    /// <summary>A holder of live blocks (a model's retained conversations) that gives them back when asked.</summary>
    private sealed class FakeReclaimer : IDeviceMemoryReclaimer
    {
        private readonly Rig _rig;
        private readonly List<(IntPtr, long)> _held = new();
        public readonly List<long> Asked = new();
        public Func<long, long> Instead;

        public FakeReclaimer(Rig rig, params long[] sizes)
        {
            _rig = rig;
            foreach (long size in sizes) _held.Add(rig.Rent(size));
        }

        public long Reclaim(long bytes)
        {
            Asked.Add(bytes);
            _rig.Events.Add("reclaim");
            if (Instead != null) return Instead(bytes);
            long released = 0;
            foreach (var block in _held)
            {
                _rig.Return(block);   // back to the pool's cache, as a disposed storage's block goes
                released += block.Item2;
            }
            _held.Clear();
            return released;
        }
    }

    [Fact]
    public void AnAllocationThatFits_ClimbsNothing()
    {
        var rig = new Rig(8 * GiB);
        rig.Cache(GiB, 512 * MiB);
        rig.Clear();

        rig.Rent(2 * GiB);

        Assert.Equal(new[] { "alloc 2048" }, rig.Events);
        Assert.Empty(rig.Frees);
        Assert.Equal((0L, 0L, 0L), (rig.Ladder.OomTrims, rig.Ladder.OomReclaims, rig.Ladder.FullDrains));
    }

    /// <summary>Rung 1: the stream is synchronized, then the least recently used cached sizes go back until the allocation
    /// plus the slack is free, and nothing else: the size in use (here the largest), the other large blocks and every small
    /// block stay cached, and no GC runs.</summary>
    [Fact]
    public void Rung1_FreesTheLeastRecentlyUsedCachedBlocksItNeeds_AndNothingElse()
    {
        var rig = new Rig(8 * GiB);
        rig.Cache(256 * MiB, 256 * MiB, 256 * MiB, 256 * MiB, 512 * MiB, 512 * MiB, GiB, GiB);
        rig.Cache(2 * GiB);                                  // the size the work uses now
        rig.Cache(Enumerable.Repeat(MiB, 64).ToArray());   // small blocks: the shard's
        var live = rig.Rent(1536 * MiB);                     // 7.56 GiB in use or cached: 0.44 GiB free
        rig.Clear();

        var block = rig.Rent(1536 * MiB);

        Assert.NotEqual(IntPtr.Zero, block.Ptr);
        Assert.Equal(new[] { "alloc 1536 failed", "sync", "trim 2048", "alloc 1536" }, rig.Events);
        Assert.Equal(new[] { 256 * MiB, 256 * MiB, 256 * MiB, 256 * MiB, 512 * MiB, 512 * MiB }, rig.Frees);
        Assert.Equal(4 * GiB, rig.Pool.LargeCachedBytes);
        Assert.Equal(64 * MiB, rig.Pool.GetStats().CachedBytes - rig.Pool.LargeCachedBytes);
        rig.Clear();
        rig.Rent(2 * GiB);
        Assert.Empty(rig.Events);                            // still cached
        Assert.Equal(0, rig.Collections);
        Assert.Equal((1L, 0L, 0L), (rig.Ladder.OomTrims, rig.Ladder.OomReclaims, rig.Ladder.FullDrains));
        Assert.Empty(rig.Warnings);
        Assert.StartsWith("[vram] GPU 3: 1536 MiB found the device full; rung 1 made room (trimmed 2048 MiB of large blocks, "
            + "0 MiB of small blocks, holders released 0 MiB) in ", Assert.Single(rig.Trace));
        GC.KeepAlive(live);
    }

    /// <summary>A synchronize that throws (the real one aborts a capture on the allocator) frees nothing and asks nobody;
    /// the ladder stays usable: the next climb on this thread is not taken for a nested one.</summary>
    [Fact]
    public void AnAbortingSynchronize_FreesNothing_AndLeavesTheLadderUsable()
    {
        var rig = new Rig(1100 * MiB);
        var holder = new FakeReclaimer(rig, 256 * MiB);
        rig.Reclaimers.Add(holder);
        rig.Cache(256 * MiB);
        var live = rig.Rent(300 * MiB);   // 812 MiB used or cached: only the holder's release makes 768 MiB fit
        rig.SyncThrows = true;
        rig.Clear();

        Assert.Throws<CudaGraphCaptureAbortedException>(() => rig.Rent(768 * MiB));
        Assert.Empty(rig.Frees);
        Assert.Empty(holder.Asked);
        Assert.Contains("rung 1 threw", Assert.Single(rig.Trace));

        rig.SyncThrows = false;
        rig.Rent(768 * MiB);
        Assert.Single(holder.Asked);
        Assert.Equal(1, rig.Ladder.OomReclaims);
        GC.KeepAlive(live);
    }

    /// <summary>The pool's free-memory reserve: a trim leaves it free beside the allocation and the slack. One that left the
    /// device below it made the pool hand every returned block to the driver (a synchronizing cuMemFree each) and allocate it
    /// again at the next rent.</summary>
    [Fact]
    public void Rung1_LeavesThePoolsFreeReserveBesideTheAllocation()
    {
        var rig = new Rig(GiB, freeReserve: 256 * MiB);
        rig.Cache(Enumerable.Repeat(32 * MiB, 16).ToArray());  // 512 MiB cached in 32 MiB blocks
        var live = rig.Rent(512 * MiB);                          // the device is full
        rig.Clear();

        rig.Rent(40 * MiB);

        // 40 + 64 + 256 MiB asked: twelve 32 MiB blocks (384 MiB).
        Assert.Equal(12, rig.Frees.Count);
        Assert.Equal(1, rig.Ladder.OomTrims);
        GC.KeepAlive(live);
    }

    /// <summary>The sizes every layer uses stay cached while the device is full: one climb frees stale sizes and the layers
    /// that follow allocate nothing. Largest-first trimming freed one of them at every climb (they are the largest blocks),
    /// the next layer allocated it again and climbed again: 750 climbs in a 24K-token prompt of Qwen3.8-27B on an RTX PRO
    /// 4500, a 55% slower prefill.</summary>
    [Fact]
    public void TheSizesTheWorkKeepsUsing_StayCachedAcrossClimbs()
    {
        var rig = new Rig(2 * GiB);
        long[] layer = { 136 * MiB, 129 * MiB, 112 * MiB, 81 * MiB };
        rig.Cache(Enumerable.Repeat(64 * MiB, 8).ToArray());   // stale: an earlier request's sizes
        rig.Cache(layer);
        var slot = rig.Rent(1060 * MiB);                         // 2,030 of 2,048 MiB in use or cached
        var grown = rig.Rent(36 * MiB);                          // a new size: the one climb
        Assert.Equal(1, rig.Ladder.OomTrims);
        Assert.All(rig.Frees, size => Assert.Equal(64 * MiB, size));
        rig.Clear();

        for (int i = 0; i < 20; i++)
            rig.Cache(layer);

        Assert.Empty(rig.Events);
        Assert.Equal(1, rig.Ladder.OomTrims);
        GC.KeepAlive(slot);
        GC.KeepAlive(grown);
    }

    [Theory]
    [InlineData(100, 0, 164)]          // the allocation and the slack
    [InlineData(100, 256, 420)]        // and the pool's free-memory reserve
    public void TrimTarget_IsTheAllocationTheSlackAndTheReserve(long mib, long reserveMib, long expectedMib)
        => Assert.Equal(expectedMib * MiB, DeviceMemoryReclaimLadder.TrimTarget(mib * MiB, reserveMib * MiB));

    /// <summary>The slack: a trim frees the allocation plus 64 MiB, so the next small allocation does not climb again.</summary>
    [Fact]
    public void Rung1_TrimsTheAllocationPlusTheSlack()
    {
        var rig = new Rig(GiB);
        rig.Cache(Enumerable.Repeat(32 * MiB, 8).ToArray());   // 256 MiB cached in 32 MiB blocks
        var live = rig.Rent(768 * MiB);                         // the device is full
        rig.Clear();

        rig.Rent(40 * MiB);

        // 40 + 64 MiB asked: four 32 MiB blocks (128 MiB), not two.
        Assert.Equal(4, rig.Frees.Count);
        Assert.Equal(1, rig.Ladder.OomTrims);
        Assert.Equal(DeviceMemoryReclaimLadder.TrimSlackBytes, 64 * MiB);
        GC.KeepAlive(live);
    }

    /// <summary>Rung 2: only the shards' small blocks hold the room, so they go back once the trim freed nothing.</summary>
    [Fact]
    public void Rung2_DrainsTheSmallBlocks_WhenNoLargeBlockIsCached()
    {
        var rig = new Rig(128 * MiB);
        rig.Cache(Enumerable.Repeat(MiB, 40).ToArray());
        var live = rig.Rent(80 * MiB);   // 120 MiB used or cached
        rig.Clear();

        rig.Rent(16 * MiB);

        Assert.Equal(new[] { "alloc 16 failed", "sync", "trim 0", "drain-small 40", "alloc 16" }, rig.Events);
        Assert.Equal(40, rig.Frees.Count);
        Assert.Equal(0, rig.Collections);
        Assert.Equal((1L, 0L, 0L), (rig.Ladder.OomTrims, rig.Ladder.OomReclaims, rig.Ladder.FullDrains));
        GC.KeepAlive(live);
    }

    /// <summary>Rung 3: the cached blocks were not enough, so the registered holder is asked for the allocation's bytes; what
    /// it released lands in the cache and is trimmed back to the driver; no GC, no drain.</summary>
    [Fact]
    public void Rung3_AsksTheReclaimers_AndTrimsWhatTheyReleased()
    {
        var rig = new Rig(4 * GiB);
        var holder = new FakeReclaimer(rig, GiB, GiB);   // two retained conversations
        rig.Reclaimers.Add(holder);
        rig.Cache(256 * MiB);
        rig.Cache(Enumerable.Repeat(MiB, 16).ToArray());
        var live = rig.Rent(1500 * MiB);
        rig.Clear();

        var block = rig.Rent(GiB);

        Assert.NotEqual(IntPtr.Zero, block.Ptr);
        Assert.Equal(new[] { GiB }, holder.Asked);
        Assert.Equal(new[]
        {
            "alloc 1024 failed", "sync", "trim 256", "alloc 1024 failed", "drain-small 16", "alloc 1024 failed",
            "reclaim", "sync", "trim 2048", "alloc 1024",
        }, rig.Events);
        Assert.Equal(0, rig.Collections);
        Assert.Equal((0L, 1L, 0L), (rig.Ladder.OomTrims, rig.Ladder.OomReclaims, rig.Ladder.FullDrains));
        Assert.Empty(rig.Warnings);
        GC.KeepAlive(live);
    }

    /// <summary>Reclaimers are asked in order, and the climb stops at the first one whose release made the room.</summary>
    [Fact]
    public void Rung3_StopsAtTheFirstReclaimerThatMadeRoom()
    {
        var rig = new Rig(2 * GiB);
        var first = new FakeReclaimer(rig, 512 * MiB);
        var second = new FakeReclaimer(rig, GiB);
        var third = new FakeReclaimer(rig, 256 * MiB);
        rig.Reclaimers.AddRange(new[] { first, second, third });
        var live = rig.Rent(256 * MiB);
        rig.Clear();

        rig.Rent(GiB);

        Assert.Single(first.Asked);
        Assert.Single(second.Asked);
        Assert.Empty(third.Asked);
        Assert.Equal(1, rig.Ladder.OomReclaims);
        GC.KeepAlive(live);
    }

    /// <summary>While a CUDA graph is captured on the allocator no reclaimer runs (the real synchronize aborts the capture
    /// before; here the fake lets the climb go on): the allocation reaches the last resort.</summary>
    [Fact]
    public void Rung3_IsSkippedWhileCapturing()
    {
        var rig = new Rig(2 * GiB);
        var holder = new FakeReclaimer(rig, GiB);
        rig.Reclaimers.Add(holder);
        var live = rig.Rent(512 * MiB);
        rig.Capturing = true;
        rig.Clear();

        Assert.Throws<OutOfMemoryException>(() => rig.Rent(GiB));

        Assert.Empty(holder.Asked);
        Assert.Equal(1, rig.Ladder.FullDrains);
        GC.KeepAlive(live);
    }

    /// <summary>A reclaimer that allocates while it is asked, and fails, does not ask the reclaimers again (no recursion):
    /// the nested climb skips rung 3.</summary>
    [Fact]
    public void Rung3_IsNotReenteredByAReclaimersOwnAllocation()
    {
        var rig = new Rig(GiB);
        var live = rig.Rent(768 * MiB);
        var holder = new FakeReclaimer(rig);
        int depth = 0, deepest = 0;
        holder.Instead = _ =>
        {
            deepest = Math.Max(deepest, ++depth);
            try { Assert.Throws<OutOfMemoryException>(() => rig.Rent(512 * MiB)); }
            finally { depth--; }
            return 0;
        };
        rig.Reclaimers.Add(holder);
        rig.Clear();

        Assert.Throws<OutOfMemoryException>(() => rig.Rent(512 * MiB));

        Assert.Single(holder.Asked);
        Assert.Equal(1, deepest);
        Assert.Equal(2, rig.Ladder.FullDrains);   // the nested climb's and the outer one's
        GC.KeepAlive(live);
    }

    /// <summary>Rung 4 only when rungs 1-3 freed too little: a full GC (here a finalizer releases a view chain's block) and
    /// a drain of every cached block; warned about once in the process, naming the asked bytes, what the holders released,
    /// and the consequence.</summary>
    [Fact]
    public void Rung4_IsTheLastResort_WarnedAboutOnce()
    {
        var rig = new Rig(2 * GiB);
        var holder = new FakeReclaimer(rig, 128 * MiB);
        rig.Reclaimers.Add(holder);
        rig.Cache(256 * MiB);
        var orphan = rig.Rent(GiB);   // freed only by "finalization"
        var live = rig.Rent(512 * MiB);
        rig.OnCollect = () => { rig.Return(orphan); rig.OnCollect = null; };
        rig.Clear();

        var block = rig.Rent(GiB);

        Assert.NotEqual(IntPtr.Zero, block.Ptr);
        Assert.Equal(1, rig.Collections);
        Assert.Equal((0L, 0L, 1L), (rig.Ladder.OomTrims, rig.Ladder.OomReclaims, rig.Ladder.FullDrains));
        string warning = Assert.Single(rig.Warnings);
        Assert.StartsWith("WARNING: [cuda] device memory ran out on GPU 3 (1024 MiB asked)", warning);
        Assert.Contains("128 MiB its holders released", warning);
        Assert.Contains("ran a full GC and drained the pool", warning);
        Assert.Contains("later steps allocate their buffers again. Reported once.", warning);
        Assert.DoesNotContain("still failed", warning);

        // A second last resort, which fails: counted, not warned about again.
        Assert.Throws<OutOfMemoryException>(() => rig.Rent(2 * GiB));
        Assert.Equal(2, rig.Ladder.FullDrains);
        Assert.Single(rig.Warnings);
        GC.KeepAlive(live);
    }

    [Fact]
    public void FullDrainWarning_SaysWhenTheAllocationStillFailed()
    {
        string line = DeviceMemoryReclaimLadder.FullDrainWarning(0, 3 * MiB / 2, 0, 12.4, succeeded: false);

        Assert.Equal("WARNING: [cuda] device memory ran out on GPU 0 (2 MiB asked) after the pool's cached blocks went back "
            + "to the driver: ran a full GC and drained the pool (12 ms), and the allocation still failed; later steps "
            + "allocate their buffers again. Reported once.", line);
    }

    /// <summary>Another failure than out-of-memory is not the ladder's: it propagates before any rung.</summary>
    [Fact]
    public void AnotherDriverError_PropagatesWithoutClimbing()
    {
        var events = new List<string>();
        var ladder = new DeviceMemoryReclaimLadder(
            _ => throw new InvalidOperationException("CUDA error 700"),
            () => events.Add("sync"),
            _ => { events.Add("trim"); return 0; },
            () => { events.Add("drain"); return 0; },
            null, null,
            () => events.Add("gc"),
            new ReportOnce(_ => events.Add("warn")),
            deviceId: 0);

        Assert.Throws<InvalidOperationException>(() => ladder.Allocate(MiB));
        Assert.Empty(events);
    }
}
