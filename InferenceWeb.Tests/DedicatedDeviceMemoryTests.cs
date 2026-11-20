// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Issue #383: on Windows (WDDM) cuMemGetInfo counts shared system memory in its total (an RTX Spark N1X: 50,803 MiB against
// NVML's 38,912 MiB dedicated) and only the calling process's allocations in its free, and a cuMemAlloc past dedicated memory
// succeeds by spilling to system memory. Every room check passed while dedicated memory was full, and the out-of-memory
// reclaim ladder never ran. DeviceMemoryView bounds the reading by NVML's dedicated figures (and a deployment budget), and the
// ladder gives memory back before an allocation would pass them. These tests run the view and the ladder over fake drivers,
// a fake NVML and a fake device that spills, without a GPU.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Cuda;
using Xunit;

namespace InferenceWeb.Tests;

public class DedicatedDeviceMemoryTests
{
    private const long MiB = 1L << 20;
    private const long GiB = 1L << 30;

    /// <summary>NVML's reading: device-wide use and the dedicated total, as the test sets them.</summary>
    private sealed class FakeNvml : IDedicatedMemoryProbe
    {
        public long Used, Total;
        public int Reads;
        public bool Fails;

        public bool TryRead(out long usedBytes, out long totalBytes)
        {
            Reads++;
            usedBytes = Used;
            totalBytes = Total;
            return !Fails && Total > 0;
        }
    }

    /// <summary>The reporter's machine: 38,912 MiB dedicated, cuMemGetInfo's total 1.3x that (shared memory counted), its free
    /// this process's view only (another process's 19.7 GiB not subtracted), NVML's use device-wide.</summary>
    private sealed class Wddm
    {
        public readonly long Dedicated = 38912 * MiB;
        public long DriverTotal => Dedicated * 13 / 10;
        public long OwnUse;
        public long OtherProcess;
        public readonly FakeNvml Nvml = new();

        public (long Free, long Total)? Driver() => (DriverTotal - OwnUse, DriverTotal);

        public void Sync()
        {
            Nvml.Total = Dedicated;
            Nvml.Used = Math.Min(Dedicated, OwnUse + OtherProcess);
        }

        public DeviceMemoryView View(long budget = 0) => new(Driver, Nvml, budget, driverMaySpill: true);
    }

    [Fact]
    public void OnWddm_TheReadingIsDedicatedMemoryLessDeviceWideUse_NotCuMemGetInfos()
    {
        var w = new Wddm { OwnUse = 18 * GiB, OtherProcess = 12 * GiB };
        w.Sync();
        DeviceMemoryView view = w.View();

        Assert.True(view.TryRead(out long free, out long total));
        Assert.Equal(w.Dedicated, total);                                  // not cuMemGetInfo's 50,585 MiB
        Assert.Equal(w.Dedicated - 30 * GiB, free);                        // the other process's 12 GiB counted
        Assert.True(w.Driver()!.Value.Free > 31 * GiB);                    // what cuMemGetInfo alone had said
        Assert.True(view.AllocationsMaySpill);
    }

    [Fact]
    public void WithoutNvml_TheReadingIsCuMemGetInfos_AndNothingSpills()
    {
        var view = new DeviceMemoryView(() => (5 * GiB, 32 * GiB), dedicated: null, budgetBytes: 0, driverMaySpill: false);
        Assert.True(view.TryRead(out long free, out long total));
        Assert.Equal((5 * GiB, 32 * GiB), (free, total));
        Assert.False(view.AllocationsMaySpill);
        Assert.Equal(0, view.ShortfallBytes(64 * GiB, 0));                 // a full device fails the allocation itself

        // NVML that fails to read leaves cuMemGetInfo's figures.
        var broken = new DeviceMemoryView(() => (5 * GiB, 32 * GiB), new FakeNvml { Fails = true, Total = 1 }, 0, driverMaySpill: true);
        Assert.True(broken.TryRead(out free, out total));
        Assert.Equal((5 * GiB, 32 * GiB), (free, total));
    }

    [Fact]
    public void TheSmallerFreeAndTheSmallerTotalWin()
    {
        // Linux: the figures nearly agree; NVML's reserved memory makes its free the smaller.
        var nvml = new FakeNvml { Total = 32623 * MiB, Used = 20000 * MiB };
        var view = new DeviceMemoryView(() => (12500 * MiB, 32126 * MiB), nvml, 0, driverMaySpill: false);
        Assert.True(view.TryRead(out long free, out long total));
        Assert.Equal(32126 * MiB, total);
        Assert.Equal(12500 * MiB, free);
        nvml.Used = 21000 * MiB;
        Assert.True(view.TryRead(out free, out _));
        Assert.Equal(11623 * MiB, free);
        Assert.False(view.AllocationsMaySpill);                             // Linux fails a full device's cuMemAlloc
    }

    [Fact]
    public void ABudget_BoundsTheTotal_AndTheFreeByDeviceWideUse()
    {
        var w = new Wddm { OwnUse = 20 * GiB };
        w.Sync();
        DeviceMemoryView view = w.View(budget: 36 * GiB);
        Assert.True(view.TryRead(out long free, out long total));
        Assert.Equal(36 * GiB, total);
        Assert.Equal(16 * GiB, free);

        // Without NVML (a Linux device, the budget an emulation of a smaller one): the driver's use counts.
        var linux = new DeviceMemoryView(() => (12 * GiB, 32 * GiB), null, 24 * GiB, driverMaySpill: false);
        Assert.True(linux.TryRead(out free, out total));
        Assert.Equal((4 * GiB, 24 * GiB), (free, total));
        Assert.True(linux.AllocationsMaySpill);                             // past the budget the real device still has room
        Assert.Equal(0, linux.ShortfallBytes(3 * GiB, 512 * MiB));
        Assert.Equal(GiB / 2, linux.ShortfallBytes(4 * GiB, 512 * MiB));
    }

    [Fact]
    public void TheShortfall_CountsThisProcesssAllocationsSinceTheLastNvmlReading()
    {
        var w = new Wddm { OwnUse = 36 * GiB };
        w.Sync();
        DeviceMemoryView view = w.View();
        long reserve = 256 * MiB;
        Assert.Equal(0, view.ShortfallBytes(GiB, reserve));
        int reads = w.Nvml.Reads;

        // Allocated since the reading (NVML not read again within 100 ms): 2 GiB more of our own.
        view.NoteAllocated(2 * GiB);
        long expected = 38 * GiB + GiB + reserve - w.Dedicated;
        Assert.Equal(expected, view.ShortfallBytes(GiB, reserve));
        Assert.Equal(reads, w.Nvml.Reads);

        // Freed again: the guard sees it at once.
        view.NoteAllocated(-2 * GiB);
        Assert.Equal(0, view.ShortfallBytes(GiB, reserve));

        // Another process took 1 GiB: the next reading (Invalidate: memory changed) shows it.
        w.OtherProcess = GiB;
        w.Sync();
        view.Invalidate();
        Assert.Equal(36 * GiB + GiB + GiB + reserve - w.Dedicated, view.ShortfallBytes(GiB, reserve));
    }

    // ---------------------------------------------------------------- the ladder's budget guard

    /// <summary>A device that spills: every allocation succeeds (WDDM places one past dedicated memory in system memory); the
    /// pool and the ladder over it are wired as CudaAllocator wires them, with the budget guard reading its use.</summary>
    private sealed class SpillingRig
    {
        private long _next = 0x10000;
        private readonly Dictionary<IntPtr, long> _live = new();
        public readonly long Dedicated;
        public readonly long Reserve;
        public long Used, PeakUsed;
        public readonly List<string> Events = new();
        public readonly List<IDeviceMemoryReclaimer> Reclaimers = new();
        public readonly List<string> Trace = new();
        public bool Capturing;
        public readonly CudaDeviceMemoryPool Pool;
        public readonly DeviceMemoryReclaimLadder Ladder;

        public SpillingRig(long dedicated, long reserve = 256 * MiB)
        {
            Dedicated = dedicated;
            Reserve = reserve;
            Pool = new CudaDeviceMemoryPool(maxCachedBytes: 256 * MiB, maxCachedBlockBytes: 256 * MiB, enabled: true,
                backingAllocate: bytes => Ladder.Allocate(bytes), backingFree: Free, shardCount: 1, largeCachedBytesCap: 64 * GiB);
            Ladder = new DeviceMemoryReclaimLadder(
                TryAllocate,
                () => Events.Add("sync"),
                bytes => Log("trim", Pool.TrimLarge(bytes)),
                () => Log("drain-small", Pool.DrainSmall()),
                () => Reclaimers,
                () => Capturing,
                () => { Events.Add("gc+drain"); Pool.DrainAndFree(); },
                new ReportOnce(_ => { }),
                deviceId: 0,
                Trace.Add,
                reserve,
                bytes => Math.Max(0, Used + bytes + Reserve - Dedicated));
        }

        private long Log(string what, long bytes)
        {
            Events.Add($"{what} {bytes / MiB}");
            return bytes;
        }

        private IntPtr TryAllocate(long bytes)
        {
            Used += bytes;
            PeakUsed = Math.Max(PeakUsed, Used);
            var ptr = new IntPtr(_next += 0x10000);
            _live[ptr] = bytes;
            Events.Add($"alloc {bytes / MiB}{(Used > Dedicated ? " past" : "")}");
            return ptr;
        }

        private void Free(IntPtr ptr)
        {
            Used -= _live[ptr];
            _live.Remove(ptr);
        }

        public (IntPtr Ptr, long Bytes) Rent(long bytes) => (Pool.Rent(bytes, out long b), b);

        public void Return((IntPtr Ptr, long Bytes) block) => Pool.Return(block.Ptr, block.Bytes);

        public void Cache(params long[] sizes) => sizes.Select(Rent).ToList().ForEach(Return);
    }

    private sealed class HolderReclaimer : IDeviceMemoryReclaimer
    {
        private readonly SpillingRig _rig;
        private readonly List<(IntPtr, long)> _held = new();
        public readonly List<long> Asked = new();

        public HolderReclaimer(SpillingRig rig, params long[] sizes)
        {
            _rig = rig;
            foreach (long size in sizes) _held.Add(rig.Rent(size));
        }

        public long Reclaim(long bytes)
        {
            Asked.Add(bytes);
            _rig.Events.Add("reclaim");
            long released = 0;
            foreach (var block in _held)
            {
                _rig.Return(block);
                released += block.Item2;
            }
            _held.Clear();
            return released;
        }
    }

    /// <summary>An allocation that would pass dedicated memory gives back the cached blocks the work stopped using first, only
    /// as many as make it fit, and is then placed within dedicated memory: no out-of-memory error came to start the climb.</summary>
    [Fact]
    public void AnAllocationThatWouldSpill_TrimsTheCachedBlocksItNeeds_AndStaysWithinDedicatedMemory()
    {
        var rig = new SpillingRig(16 * GiB);
        var live = rig.Rent(12 * GiB);
        rig.Cache(512 * MiB, 512 * MiB, GiB, GiB);   // 15 GiB used or cached
        rig.Events.Clear();

        var block = rig.Rent(1536 * MiB);

        Assert.NotEqual(IntPtr.Zero, block.Ptr);
        Assert.Equal(new[] { "sync", "trim 1024", "alloc 1536" }, rig.Events);
        Assert.True(rig.PeakUsed <= rig.Dedicated - rig.Reserve, $"peak {rig.PeakUsed / MiB} MiB");
        Assert.Equal((1L, 0L, 0L), (rig.Ladder.BudgetTrims, rig.Ladder.BudgetReclaims, rig.Ladder.BudgetPasses));
        Assert.Equal((0L, 0L, 0L), (rig.Ladder.OomTrims, rig.Ladder.OomReclaims, rig.Ladder.FullDrains));
        Assert.Contains("would pass dedicated memory; rung 1 kept it within", Assert.Single(rig.Trace));
        GC.KeepAlive(live);
    }

    /// <summary>With nothing cached, a holder (a model's finished conversations) is asked before the allocation spills.</summary>
    [Fact]
    public void WithNothingCached_AHolderGivesMemoryBackBeforeTheAllocationSpills()
    {
        var rig = new SpillingRig(16 * GiB);
        var live = rig.Rent(10 * GiB);
        var holder = new HolderReclaimer(rig, 2 * GiB, 2 * GiB);
        rig.Reclaimers.Add(holder);
        rig.Events.Clear();

        rig.Rent(3 * GiB);

        Assert.Single(holder.Asked);
        Assert.Equal(new[] { "sync", "trim 0", "drain-small 0", "reclaim", "sync", "trim 2048", "alloc 3072" }, rig.Events);
        Assert.True(rig.PeakUsed <= rig.Dedicated - rig.Reserve);
        Assert.Equal((0L, 1L, 0L), (rig.Ladder.BudgetTrims, rig.Ladder.BudgetReclaims, rig.Ladder.BudgetPasses));
        GC.KeepAlive(live);
    }

    /// <summary>Where nothing can be given back the allocation still succeeds (past dedicated memory, as the driver allows):
    /// the guard never fails one the device would serve, never runs the last resort, and does not synchronize again for the
    /// next allocations once a climb freed nothing.</summary>
    [Fact]
    public void WithNothingToGiveBack_TheAllocationGoesAhead_AndTheNextOnesDoNotClimbAgain()
    {
        var rig = new SpillingRig(16 * GiB);
        var live = rig.Rent(15 * GiB + 512 * MiB);
        rig.Events.Clear();

        var past = rig.Rent(GiB);
        Assert.NotEqual(IntPtr.Zero, past.Ptr);
        Assert.Equal(new[] { "sync", "trim 0", "drain-small 0", "alloc 1024 past" }, rig.Events);
        Assert.Equal(1, rig.Ladder.BudgetPasses);
        Assert.Equal(0, rig.Ladder.FullDrains);

        rig.Events.Clear();
        rig.Rent(GiB);
        Assert.Equal(new[] { "alloc 1024 past" }, rig.Events);   // no synchronize for nothing
        Assert.Equal(2, rig.Ladder.BudgetPasses);
        GC.KeepAlive(live);
    }

    /// <summary>While a graph is captured the guard frees nothing (its synchronize would abort the capture).</summary>
    [Fact]
    public void WhileAGraphIsCaptured_TheGuardFreesNothing()
    {
        var rig = new SpillingRig(16 * GiB);
        var live = rig.Rent(14 * GiB + 512 * MiB);
        rig.Cache(GiB);
        rig.Capturing = true;
        rig.Events.Clear();

        rig.Rent(768 * MiB);
        Assert.Equal(new[] { "alloc 768 past" }, rig.Events);
        Assert.Equal(1, rig.Ladder.BudgetPasses);
        GC.KeepAlive(live);
    }

    // ---------------------------------------------------------------- a model's room checks over the reading

    /// <summary>A model's room checks over the reporter's machine: with cuMemGetInfo's figures a second request's slot is made
    /// with dedicated memory nearly full (12 GiB "free" that is Windows' RAM); over the dedicated reading it waits beside the
    /// running request, the prefix cache sees the deficit, and alone it is refused before any of it runs.</summary>
    [Fact]
    public void OverTheDedicatedReading_ASlotThatOnlyCuMemGetInfoHadRoomForWaitsOrIsRefused()
    {
        string dir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "ts-dedicated-" + Guid.NewGuid().ToString("N"));
        System.IO.Directory.CreateDirectory(dir);
        try
        {
            using var dtype = TensorSharp.Models.KvCacheDtypeConfig.ForCurrentFlowInTests(TensorSharp.Models.KvCacheDtype.F32);
            string path = new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(System.IO.Path.Combine(dir, "q35.gguf"));
            using var model = new TensorSharp.Models.Qwen35RuntimeModel(path, TensorSharp.Runtime.BackendType.Cpu, 1, null, 1);
            var unified = (TensorSharp.Runtime.Scheduling.IUnifiedBatchModel)model;
            int[] prompt = Enumerable.Range(0, 20).Select(i => 40 + i % 7).ToArray();
            Assert.Equal(0, unified.EnsureSequenceSlot("run", 64));
            Assert.True(unified.TryForwardUnified(new[] { "run" }, new[] { prompt }, new[] { 0 }, new float[1][]));

            // Our 30 GiB and another process's 7 GiB of the 38,912 MiB dedicated: about 1 GiB left.
            var w = new Wddm { OwnUse = 30 * GiB, OtherProcess = 7 * GiB };
            w.Sync();
            DeviceMemoryView view = w.View();

            model.DeviceRoomForTests = () => w.Driver()!.Value;   // what the room checks read before
            Assert.Equal(0, unified.EnsureSequenceSlot("granted", 2048));
            Assert.Equal(0, model.DeviceSlotDeficitBytes);
            model.OnSequenceReleased("granted");   // pooled (2,048 cells): the requests below ask for more than it holds

            model.DeviceRoomForTests = () => view.TryRead(out long free, out long total) ? (free, total) : (0, 0);
            Assert.Throws<TensorSharp.Runtime.Scheduling.SequenceSlotUnavailableException>(() => unified.EnsureSequenceSlot("waits", 4096));
            Assert.True(model.DeviceSlotDeficitBytes > 0);       // the prefix cache gives conversations up for the headroom
            Assert.True(((TensorSharp.Runtime.Scheduling.IUnifiedBatchModel)model).TryForwardUnified(new[] { "run" }, new[] { new[] { 7 } },
                new[] { prompt.Length }, new float[1][]));

            model.OnSequenceReleased("run");
            var refused = Assert.Throws<InvalidOperationException>(() => unified.EnsureSequenceSlot("alone", 4096));
            Assert.Contains("refused before any of it ran", refused.Message);
            Assert.False(model.HasFusedSequenceCache("alone"));
        }
        finally
        {
            try { System.IO.Directory.Delete(dir, recursive: true); } catch (System.IO.IOException) { }
        }
    }

    /// <summary>Without a budget guard (a Linux device, where a full device fails the allocation) nothing changes: the guard
    /// is asked and answers 0.</summary>
    [Fact]
    public void AViewThatDoesNotSpill_LeavesTheLadderAsItWas()
    {
        var view = new DeviceMemoryView(() => (GiB, 32 * GiB), null, 0, driverMaySpill: false);
        for (long bytes = MiB; bytes < 64 * GiB; bytes *= 4)
            Assert.Equal(0, view.ShortfallBytes(bytes, 256 * MiB));
    }
}
