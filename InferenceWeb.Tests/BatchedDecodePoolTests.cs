using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>
/// The batched fused decode's device K/V pool (BatchedDecodePool): its block remap and the order of its host/device
/// transfers, against a host-array stand-in for the device copies. The model drives the same object with the
/// ggml_cuda range copies (Qwen35Model.GgmlBfdPoolTransfers); these tests need no device.
/// </summary>
public class BatchedDecodePoolTests
{
    /// <summary>Host and device pools as arrays of <see cref="Width"/>-float rows, with a log of the transfers.</summary>
    private sealed class FakeTransfers : IBatchedDecodePoolTransfers
    {
        public const int Width = 3;
        public readonly float[] Host;
        public float[] Device = Array.Empty<float>();
        public int DeviceSlots { get; private set; }
        public bool Resident { get; set; }
        public string LastError { get; private set; } = "";
        public int FailResizes, FailUploads;
        public readonly List<string> Log = new();

        public FakeTransfers(int hostSlots) => Host = new float[hostSlots * Width];

        public bool Resize(int deviceSlots)
        {
            Log.Add($"resize {deviceSlots}");
            if (FailResizes > 0) { FailResizes--; LastError = "test: no memory"; return false; }
            Device = new float[deviceSlots * Width];
            DeviceSlots = deviceSlots;
            Resident = true;
            return true;
        }

        public bool Upload(IReadOnlyList<PoolRowRange> ranges, bool zeroFirst)
        {
            Log.Add(zeroFirst ? $"seed {ranges.Count}" : $"join {ranges.Count}");
            if (FailUploads > 0) { FailUploads--; LastError = "test: upload failed"; return false; }
            if (zeroFirst) Array.Clear(Device);
            foreach (var r in ranges)
                Array.Copy(Host, r.HostSlot * Width, Device, r.DeviceSlot * Width, r.Rows * Width);
            return true;
        }

        public bool Download(IReadOnlyList<PoolRowRange> ranges)
        {
            Log.Add($"download {ranges.Count}");
            foreach (var r in ranges)
                Array.Copy(Device, r.DeviceSlot * Width, Host, r.HostSlot * Width, r.Rows * Width);
            return true;
        }
    }

    private const int Bs = 4;   // rows per block in the pool-level tests; the pool grows 2 blocks at a time

    private static BatchedDecodePool NewPool(FakeTransfers io) => new(Bs, io, growthSlots: 2 * Bs);

    private static float[] Row(float value) => new[] { value, value + 0.25f, value + 0.5f };

    /// <summary>The fused step's graph: writes each token's row at its device slot.</summary>
    private static void GraphWrites(BatchedDecodePool pool, FakeTransfers io, IReadOnlyDictionary<int, float> rows)
    {
        foreach (var (hostSlot, value) in rows)
            Array.Copy(Row(value), 0, io.Device, pool.DeviceSlotOf(hostSlot) * FakeTransfers.Width, FakeTransfers.Width);
    }

    /// <summary>The op-by-op step: writes the host pool, after invalidating the seed as ForwardBatch does.</summary>
    private static void HostWrites(BatchedDecodePool pool, FakeTransfers io, IReadOnlyDictionary<int, float> rows)
    {
        pool.InvalidateSeed();
        foreach (var (hostSlot, value) in rows)
            Array.Copy(Row(value), 0, io.Host, hostSlot * FakeTransfers.Width, FakeTransfers.Width);
    }

    /// <summary>One fused decode step: prepare, owe the written rows, compute, download.</summary>
    private static void FusedStep(BatchedDecodePool pool, FakeTransfers io, int[] blocks, Func<int, bool> live,
        IReadOnlyDictionary<int, float> writes)
    {
        Assert.True(pool.TryPrepare(blocks, live, out string reason), reason);
        pool.BeginStep(writes.Keys.ToArray());
        GraphWrites(pool, io, writes);
        pool.CompleteStep();
    }

    /// <summary>What the step's attention reads for these host slots: the device rows at their device slots.</summary>
    private static float[] Gather(BatchedDecodePool pool, FakeTransfers io, IEnumerable<int> hostSlots)
        => hostSlots.SelectMany(h => io.Device.Skip((int)pool.DeviceSlotOf(h) * FakeTransfers.Width).Take(FakeTransfers.Width)).ToArray();

    private static float[] HostRows(FakeTransfers io, IEnumerable<int> hostSlots)
        => hostSlots.SelectMany(h => io.Host.Skip(h * FakeTransfers.Width).Take(FakeTransfers.Width)).ToArray();

    private static IEnumerable<int> Slots(params int[] blocks) => blocks.SelectMany(b => Enumerable.Range(b * Bs, Bs));

    // ---- The block map --------------------------------------------------------------------------------------------

    [Fact]
    public void HighBlockIds_MapIntoASmallPool()
    {
        // The block pool hands ids out first-in-first-out, so a short request reaches ids above 200 of 256; the
        // device pool used to be indexed by id and sized for all 256 (8 GiB of F32 K/V on Qwen3.8-27B).
        var map = new BatchedDecodeBlockMap(blockSize: 256);

        var joined = map.Map(new[] { 201, 230, 255 }, _ => true, out bool grew);

        Assert.True(grew);
        Assert.Equal(BatchedDecodeBlockMap.GrowthSlots, map.DeviceSlots);
        Assert.Equal(new[] { 0, 1, 2 }, joined);
        Assert.Equal(2L * 256 + 7, map.DeviceSlotOf(255L * 256 + 7));
        Assert.Equal(-1, map.DeviceBlockOf(200));
    }

    [Fact]
    public void ThePoolGrowsInGrowthSlotSteps_OnlyWhenNoDeviceBlockIsFree()
    {
        var map = new BatchedDecodeBlockMap(blockSize: 256);
        map.Map(Enumerable.Range(100, 8).ToArray(), _ => true, out bool grew);
        Assert.True(grew);
        Assert.Equal(8, map.DeviceBlocks);

        map.Map(Enumerable.Range(100, 8).ToArray(), _ => true, out grew);   // the same blocks: nothing to do
        Assert.False(grew);

        map.Map(new[] { 150 }, _ => true, out grew);   // a ninth live block
        Assert.True(grew);
        Assert.Equal(16, map.DeviceBlocks);
        Assert.Equal(2 * BatchedDecodeBlockMap.GrowthSlots, map.DeviceSlots);
    }

    [Fact]
    public void LiveBlocksKeepTheirDeviceBlock_FreedOnesAreRecycled()
    {
        var map = new BatchedDecodeBlockMap(blockSize: 256);
        var live = new HashSet<int>(Enumerable.Range(1, 8));
        map.Map(live.ToArray(), live.Contains, out _);
        var before = Enumerable.Range(1, 8).ToDictionary(b => b, map.DeviceBlockOf);

        live.Remove(3);                       // its request ended: the block pool freed it
        live.Add(9);
        var joined = map.Map(new[] { 1, 2, 4, 9 }, live.Contains, out bool grew);

        Assert.False(grew);
        Assert.Equal(new[] { before[3] }, joined);     // 9 took 3's device block
        Assert.Equal(before[3], map.DeviceBlockOf(9));
        Assert.Equal(-1, map.DeviceBlockOf(3));
        foreach (int b in new[] { 1, 2, 4, 5, 6, 7, 8 })
            Assert.Equal(before[b], map.DeviceBlockOf(b));
    }

    [Fact]
    public void AFreedBlockThisStepStillNeeds_IsNotRecycled()
    {
        var map = new BatchedDecodeBlockMap(blockSize: 256);
        map.Map(Enumerable.Range(1, 8).ToArray(), _ => true, out _);

        // Liveness reports block 2 free, but this step lists it (a stale table must not lose rows mid-step).
        map.Map(new[] { 2, 50 }, b => b != 2 && b != 5, out bool grew);

        Assert.False(grew);
        Assert.NotEqual(-1, map.DeviceBlockOf(2));
        Assert.Equal(-1, map.DeviceBlockOf(5));
    }

    // ---- Coherence with the host pool -----------------------------------------------------------------------------

    [Fact]
    public void ABlockFreedAndReallocatedMidRun_ReadsWhatAFreshPoolReads()
    {
        var io = new FakeTransfers(hostSlots: 64 * Bs);
        var pool = NewPool(io);
        var live = new HashSet<int> { 10, 11, 12 };
        // A on blocks 10-11, B on 12: prompts by the op-by-op prefill, then fused decode steps.
        HostWrites(pool, io, Enumerable.Range(0, 6).ToDictionary(p => 10 * Bs + p, p => 100f + p));
        HostWrites(pool, io, Enumerable.Range(0, 2).ToDictionary(p => 12 * Bs + p, p => 200f + p));
        for (int step = 0; step < 2; step++)
            FusedStep(pool, io, new[] { 10, 11, 12 }, live.Contains,
                new Dictionary<int, float> { [10 * Bs + 6 + step] = 150f + step, [12 * Bs + 2 + step] = 250f + step });

        // A ends (blocks 10 and 11 freed); C is handed block 11 again (the same id, other rows) and block 20, and
        // is prefilled op-by-op; B moves on to block 13.
        live.Remove(10);
        live.Add(20);
        live.Add(13);
        HostWrites(pool, io, Enumerable.Range(0, 7).ToDictionary(p => p < 4 ? 11 * Bs + p : 20 * Bs + p - 4, p => 300f + p));
        FusedStep(pool, io, new[] { 12, 13, 11, 20 }, live.Contains,
            new Dictionary<int, float> { [13 * Bs] = 260f, [20 * Bs + 3] = 350f });

        // 11 kept its device block, 20 took freed 10's, and nothing grew.
        Assert.Equal(4 * Bs, pool.DeviceSlots);
        Assert.Equal(-1, pool.Map.DeviceBlockOf(10));
        // A fresh pool built from the same host rows reads exactly what the recycled one reads.
        var freshIo = new FakeTransfers(hostSlots: 64 * Bs);
        Array.Copy(io.Host, freshIo.Host, io.Host.Length);
        var fresh = NewPool(freshIo);
        Assert.True(fresh.TryPrepare(new[] { 12, 13, 11, 20 }, live.Contains, out _));
        var slots = Slots(12, 13, 11, 20).ToArray();
        Assert.Equal(Gather(fresh, freshIo, slots), Gather(pool, io, slots));
        Assert.Equal(HostRows(io, slots), Gather(pool, io, slots));
        Assert.Equal(Row(300f), Gather(pool, io, new[] { 11 * Bs }));   // C's rows, not A's
    }

    [Fact]
    public void GrowthRightAfterAStepThatWroteRows_KeepsThoseRows()
    {
        var io = new FakeTransfers(hostSlots: 16 * Bs);
        var pool = NewPool(io);
        var live = new HashSet<int> { 0, 1, 2 };
        HostWrites(pool, io, new Dictionary<int, float> { [0] = 1f, [1 * Bs] = 2f });
        // A step that fills the 2-block pool and writes a row in each block.
        FusedStep(pool, io, new[] { 0, 1 }, live.Contains, new Dictionary<int, float> { [1] = 11f, [1 * Bs + 1] = 21f });
        io.Log.Clear();

        // The next step needs a third block: the pool regrows and is reseeded from the host pool.
        Assert.True(pool.TryPrepare(new[] { 0, 2, 1 }, live.Contains, out _));

        Assert.Equal(new[] { $"resize {4 * Bs}", "seed 3" }, io.Log);
        Assert.Equal(Row(11f), Gather(pool, io, new[] { 1 }));
        Assert.Equal(Row(21f), Gather(pool, io, new[] { 1 * Bs + 1 }));
    }

    [Fact]
    public void AStepThatThrewAfterItsCompute_IsDownloadedBeforeTheHostPoolIsTouched()
    {
        var io = new FakeTransfers(hostSlots: 16 * Bs);
        var pool = NewPool(io);
        Assert.True(pool.TryPrepare(new[] { 0 }, _ => true, out _));
        pool.BeginStep(new[] { 2 });
        GraphWrites(pool, io, new Dictionary<int, float> { [2] = 7f });
        io.Log.Clear();
        // (the GDN scatter threw: CompleteStep never ran)

        pool.InvalidateSeed();   // the op-by-op step that follows

        Assert.Equal(new[] { "download 1" }, io.Log);
        Assert.Equal(Row(7f), HostRows(io, new[] { 2 }));
    }

    [Fact]
    public void ADeclinedStep_DownloadsNothing_AndTheNextStepReseeds()
    {
        var io = new FakeTransfers(hostSlots: 16 * Bs);
        var pool = NewPool(io);
        Assert.True(pool.TryPrepare(new[] { 0 }, _ => true, out _));
        pool.BeginStep(new[] { 2 });
        io.Log.Clear();

        pool.AbandonStep();      // the native graph declined
        pool.InvalidateSeed();   // the op-by-op step
        Assert.True(pool.TryPrepare(new[] { 0 }, _ => true, out _));

        Assert.Equal(new[] { "seed 1" }, io.Log);
    }

    [Fact]
    public void AJoiningBlock_IsUploadedAlone()
    {
        var io = new FakeTransfers(hostSlots: 16 * Bs);
        var pool = NewPool(io);
        HostWrites(pool, io, new Dictionary<int, float> { [1 * Bs] = 5f });
        Assert.True(pool.TryPrepare(new[] { 0 }, _ => true, out _));
        io.Log.Clear();

        Assert.True(pool.TryPrepare(new[] { 0, 1 }, _ => true, out _));

        Assert.Equal(new[] { "join 1" }, io.Log);
        Assert.Equal(Row(5f), Gather(pool, io, new[] { 1 * Bs }));
    }

    [Fact]
    public void AFailedAllocation_DefersTheStep_AndTheNextOneReseeds()
    {
        var io = new FakeTransfers(hostSlots: 16 * Bs) { FailResizes = 1 };
        var pool = NewPool(io);

        Assert.False(pool.TryPrepare(new[] { 3 }, _ => true, out string reason));
        Assert.Contains("no memory", reason);

        io.Log.Clear();
        Assert.True(pool.TryPrepare(new[] { 3 }, _ => true, out _));
        Assert.Equal(new[] { $"resize {2 * Bs}", "seed 1" }, io.Log);
    }

    [Fact]
    public void AFailedUpload_DefersTheStep_AndTheNextOneReseedsEverything()
    {
        var io = new FakeTransfers(hostSlots: 16 * Bs);
        var pool = NewPool(io);
        Assert.True(pool.TryPrepare(new[] { 0 }, _ => true, out _));
        io.FailUploads = 1;

        Assert.False(pool.TryPrepare(new[] { 0, 1 }, _ => true, out string reason));
        Assert.Contains("upload failed", reason);

        io.Log.Clear();
        Assert.True(pool.TryPrepare(new[] { 0, 1 }, _ => true, out _));
        Assert.Equal(new[] { "seed 2" }, io.Log);
    }

    [Fact]
    public void ALostDeviceCopy_IsReseeded()
    {
        var io = new FakeTransfers(hostSlots: 16 * Bs);
        var pool = NewPool(io);
        Assert.True(pool.TryPrepare(new[] { 0 }, _ => true, out _));
        io.Resident = false;   // something dropped the pool's device copies
        io.Log.Clear();

        Assert.True(pool.TryPrepare(new[] { 0 }, _ => true, out _));

        Assert.Equal(new[] { "seed 1" }, io.Log);
    }

    // ---- Declines ---------------------------------------------------------------------------------------------------

    [Theory]
    [InlineData("Qwen3.5 batched decode: failed to allocate backend buffer.", true)]
    [InlineData("Qwen3.5 batched decode: failed to acquire ggml context.", true)]
    [InlineData("upload_to_device_copy_ranges: device-copy budget exhausted.", true)]
    [InlineData("upload_to_device_copy_ranges: device allocation failed.", true)]
    [InlineData("Qwen3.5 batched decode: descriptor size mismatch.", false)]
    [InlineData("Qwen3.5 batched decode: V1 requires one token per sequence (n_tokens==n_seqs).", false)]
    [InlineData("", false)]
    [InlineData(null, false)]
    public void ShortagesDefer_EverythingElseStopsTheKernel(string? nativeError, bool deferrable)
        => Assert.Equal(deferrable, Qwen35Model.BfdDeclineIsDeferrable(nativeError!));
}
