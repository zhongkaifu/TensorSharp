// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The partitioned group-4 decode attention routes (the d=256 ring route the dense Qwen3.5 line and Gemma's local layers
/// take, and the recorded d=512 route of Gemma's global layers) split a row's keys by its live length, never by its cache's
/// capacity. A slot's capacity depends on its request's history (pooled and donated slots, growth headroom, capacity
/// classes), and these routes are what the CUDA executor's decode fallback (a module compiled below sm_80, e.g. the
/// committed compute_75 PTX) and the per-op decode path (a LoRA adapter, a layer split) run, so the same rows
/// <c>[0, len)</c> must give the same bits in a cache of any capacity, recorded or not.
/// </summary>
public sealed class CudaGqaDecodeCapacityIndependenceTests
{
    private readonly ITestOutputHelper _output;

    public CudaGqaDecodeCapacityIndependenceTests(ITestOutputHelper output) => _output = output;

    private static IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    public void LivePartitions_CoverEveryLength_InTheGridOfAnyLargerCache(int basePartition)
    {
        // Up to 64 x the base, the base size; past it the partitions grow with the length, 64 at most.
        Assert.Equal(basePartition, CudaKernelOps.GqaLivePartitionSize(0, basePartition));
        Assert.Equal(basePartition, CudaKernelOps.GqaLivePartitionSize(64 * basePartition, basePartition));
        Assert.Equal(basePartition + 32, CudaKernelOps.GqaLivePartitionSize(64 * basePartition + 1, basePartition));
        Assert.Equal(384, CudaKernelOps.GqaLivePartitionSize(24_000, basePartition));
        Assert.Equal(2048, CudaKernelOps.GqaLivePartitionSize(131_072, basePartition));

        int[] caps = { 1, 63, 64, 65, 129, 1000, 4096, 4097, 5120, 8192, 8193, 24_576, 28_672, 57_344, 131_072, 262_144 };
        foreach (int cap in caps)
        {
            int grid = CudaKernelOps.GqaLivePartitionGrid(cap, basePartition);
            int room = CudaKernelOps.GqaLivePartitionSize(cap, basePartition);
            Assert.InRange(grid, 1, 64);
            for (int len = 1; len <= cap; len++)
            {
                int part = CudaKernelOps.GqaLivePartitionSize(len, basePartition);
                Assert.Equal(0, part % 32);
                // A recorded launch over this cache holds the shared-memory room and the partitions of every length it may read.
                Assert.True(part <= room, $"cap {cap} len {len}: partition {part} past the launch's room {room}");
                Assert.True((len + part - 1) / part <= grid, $"cap {cap} len {len}: {(len + part - 1) / part} partitions past the grid {grid}");
            }
        }
    }

    /// <summary>The same rows in caches of two capacities, unrecorded and recorded (the length read on the device): one set of
    /// bits. The group-6 row is the generic route's control (fixed 512-key partitions).</summary>
    [CudaTheory]
    [InlineData(16, 4, 256, 1000, 1024, 4096)]       // group 4, d 256 (Qwen3.5-9B): both capacities at most 4,096
    [InlineData(16, 4, 256, 4000, 4096, 5120)]       // across 4,096 cells
    [InlineData(16, 4, 256, 24000, 24576, 28672)]    // a 1,024-rounded slot against its capacity class
    [InlineData(16, 4, 256, 47100, 48128, 57344)]
    [InlineData(8, 2, 256, 24000, 24576, 131072)]    // Qwen3.5-0.8B, a slot of the whole window
    [InlineData(8, 2, 256, 6000, 6144, 7168)]        // a length below 64 x 64 in caches above it
    [InlineData(8, 2, 512, 6000, 8192, 16384)]       // group 4, d 512 (Gemma 4's global layers): recorded past 8,192 cells
    [InlineData(8, 2, 512, 20000, 20480, 24576)]
    [InlineData(24, 4, 256, 24000, 24576, 28672)]    // group 6 (Qwen3.8-27B): the generic route
    public void DecodeRows_AreBitIdentical_AcrossCacheCapacities(int heads, int kvHeads, int d, int len, int capA, int capB)
    {
        using var allocator = new CudaAllocator();
        Assert.True(allocator.Kernels.GqaLivePartitions, "the loaded tensorsharp_kernels module has no live-length partitions");
        float[] q = Query(heads, d);
        float[] plainA = Attend(allocator, heads, kvHeads, d, len, capA, q, recorded: false);
        float[] plainB = Attend(allocator, heads, kvHeads, d, len, capB, q, recorded: false);
        float[] recA = Attend(allocator, heads, kvHeads, d, len, capA, q, recorded: true);
        float[] recB = Attend(allocator, heads, kvHeads, d, len, capB, q, recorded: true);
        AssertBits(plainA, plainB, $"unrecorded, {capA} against {capB} cells");
        AssertBits(recA, recB, $"recorded, {capA} against {capB} cells");
        // The d=512 route splits an unrecorded row into fixed 128-key partitions and a recorded one by its length (they
        // agree up to 8,192 keys); the d=256 ring route takes the live length's partitions either way.
        if (d == 256 || len <= 8192)
            AssertBits(plainA, recA, "recorded against unrecorded");
    }

    /// <summary>The committed compute_75 PTX, which every build without nvcc ships and which runs these routes for every decode
    /// row (its flash kernels are arch-guarded off): its partitions follow the length too.</summary>
    [CudaFact]
    public void CommittedBaseline_DecodeRows_AreBitIdentical_AcrossCacheCapacities()
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        string ptx = RepoFile("TensorSharp.Backends.Cuda", "native", "ptx", "tensorsharp_kernels.ptx");
        using var baseline = CudaKernels.FromModule(CudaModule.LoadFromBytes(File.ReadAllBytes(ptx)));
        Assert.Equal(75, baseline.CompiledArch);
        Assert.True(baseline.GqaLivePartitions, "the committed tensorsharp_kernels.ptx predates live-length partitions: refresh native/ptx");
        foreach (var (heads, kvHeads, d, len, capA, capB) in new[]
                 {
                     (16, 4, 256, 24000, 24576, 28672),
                     (8, 2, 256, 4000, 4096, 5120),
                     (8, 2, 512, 20000, 20480, 24576),
                 })
        {
            float[] q = Query(heads, d);
            float[] recA = Attend(allocator, heads, kvHeads, d, len, capA, q, recorded: true, baseline);
            float[] recB = Attend(allocator, heads, kvHeads, d, len, capB, q, recorded: true, baseline);
            AssertBits(recA, recB, $"committed PTX, recorded, {heads}/{kvHeads}/{d} len {len}: {capA} against {capB} cells");
            float[] plainA = Attend(allocator, heads, kvHeads, d, len, capA, q, recorded: false, baseline);
            float[] plainB = Attend(allocator, heads, kvHeads, d, len, capB, q, recorded: false, baseline);
            AssertBits(plainA, plainB, $"committed PTX, unrecorded, {heads}/{kvHeads}/{d} len {len}: {capA} against {capB} cells");
            if (d == 256)
                AssertBits(plainA, recA, $"committed PTX, {heads}/{kvHeads}/{d} len {len}: recorded against unrecorded");
        }
    }

    /// <summary>
    /// Timing only (no assert): the group-4 routes with the partitions sized from the capacity (the parent's launches: the
    /// same kernels with no live base and the capacity's partition size and grid) against the live length's, graph-captured,
    /// at 4K-64K keys in a cache as long as the row (rounded to 1,024) and in one twice as long.
    /// </summary>
    [CudaFact]
    [Trait("Category", "Bench")]
    public void Bench_LivePartitionsAgainstCapacityPartitions()
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        CudaKernels kernels = allocator.Kernels;
        IntPtr stream = allocator.Stream.Handle;
        using var dyn = new CudaDecodeDynParams(allocator);
        Assert.True(dyn.IsValid);
        var rows = new List<string>
        {
            "| heads/kv/d | len | cap | mode | capacity partitions (us) | live partitions (us) | live / capacity |",
            "|---|---:|---:|---|---:|---:|---:|",
        };
        foreach (var (heads, kvHeads, d) in new[] { (16, 4, 256), (8, 2, 256), (8, 2, 512) })
            foreach (int len in new[] { 4096, 8192, 16384, 32768, 65536 })
                foreach (int cap in new[] { (len + 1024) / 1024 * 1024, 2 * len })
                {
                    using var kc = new Tensor(allocator, DType.Float16, kvHeads, cap, d);
                    using var vc = new Tensor(allocator, DType.Float16, kvHeads, cap, d);
                    Assert.True(CudaKernelOps.TryFill(kc, 0.01f));
                    Assert.True(CudaKernelOps.TryFill(vc, 0.02f));
                    using var query = new Tensor(allocator, DType.Float32, 1, heads * d);
                    Assert.True(CudaKernelOps.TryFill(query, 0.03f));
                    using var result = new Tensor(allocator, DType.Float32, 1, heads * d);
                    using var partial = new Tensor(allocator, DType.Float32, heads, 64, d + 2);
                    foreach (bool recorded in d == 512 ? new[] { true } : new[] { false, true })
                    {
                        dyn.Write(len, 0, 0, 0);
                        dyn.EnqueueUpload();
                        CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
                        IntPtr dynPtr = recorded ? dyn.DevicePtr : IntPtr.Zero;
                        float scale = 1f / MathF.Sqrt(d);
                        Action Launcher(bool live)
                        {
                            int basePart = d == 256 ? 64 : 128;
                            int part, grid, liveBase;
                            if (live)
                            {
                                liveBase = recorded || d == 256 ? basePart : 0;
                                part = recorded ? CudaKernelOps.GqaLivePartitionSize(cap, basePart)
                                    : d == 256 ? CudaKernelOps.GqaLivePartitionSize(len, basePart) : basePart;
                                grid = recorded ? CudaKernelOps.GqaLivePartitionGrid(cap, basePart) : (len + part - 1) / part;
                            }
                            else
                            {
                                liveBase = 0;
                                part = d == 256 || recorded ? CudaKernelOps.GqaLivePartitionSize(cap, basePart) : basePart;
                                grid = ((recorded ? cap : len) + part - 1) / part;
                            }
                            return () =>
                            {
                                if (d == 256)
                                    kernels.LaunchGqaDecodeAttentionPartitionGroup4D256F16(P(query), P(kc), P(vc), IntPtr.Zero, P(partial),
                                        kvHeads, len, cap, scale, 0, grid, part, stream, dynPtr, liveBase);
                                else
                                    kernels.LaunchGqaDecodeAttentionPartitionGroup4D512F16(P(query), P(kc), P(vc), P(partial),
                                        kvHeads, 0, len, cap, scale, grid, part, stream, dynPtr, liveBase);
                                kernels.LaunchGqaDecodeAttentionPartitionReduceF32(P(partial), P(result), heads, d, grid, stream);
                            };
                        }
                        Action old = Launcher(live: false), now = Launcher(live: true);
                        double bestOld = double.MaxValue, bestNew = double.MaxValue;
                        for (int round = 0; round < 3; round++)
                        {
                            bestOld = Math.Min(bestOld, GraphMicros(stream, old, n: 16, reps: 20));
                            bestNew = Math.Min(bestNew, GraphMicros(stream, now, n: 16, reps: 20));
                        }
                        rows.Add($"| {heads}/{kvHeads}/{d} | {len} | {cap} | {(recorded ? "recorded" : "unrecorded")} | {bestOld:F1} | {bestNew:F1} | {bestNew / bestOld:F3} |");
                    }
                }
        foreach (string row in rows)
            _output.WriteLine(row);
    }

    private static double GraphMicros(IntPtr stream, Action launch, int n, int reps)
    {
        CudaDriverApi.cuEventCreate(out IntPtr start, 0).ThrowOnError();
        CudaDriverApi.cuEventCreate(out IntPtr end, 0).ThrowOnError();
        try
        {
            CudaDriverApi.cuStreamBeginCapture(stream, CudaDriverApi.CU_STREAM_CAPTURE_MODE_RELAXED).ThrowOnError();
            for (int i = 0; i < n; i++)
                launch();
            CudaDriverApi.cuStreamEndCapture(stream, out IntPtr graph).ThrowOnError();
            CudaDriverApi.cuGraphInstantiateWithFlags(out IntPtr exec, graph, 0).ThrowOnError();
            try
            {
                for (int r = 0; r < 3; r++)
                    CudaDriverApi.cuGraphLaunch(exec, stream).ThrowOnError();
                CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
                CudaDriverApi.cuEventRecord(start, stream).ThrowOnError();
                for (int r = 0; r < reps; r++)
                    CudaDriverApi.cuGraphLaunch(exec, stream).ThrowOnError();
                CudaDriverApi.cuEventRecord(end, stream).ThrowOnError();
                CudaDriverApi.cuEventSynchronize(end).ThrowOnError();
                CudaDriverApi.cuEventElapsedTime(out float ms, start, end).ThrowOnError();
                return ms * 1000.0 / ((double)n * reps);
            }
            finally
            {
                CudaDriverApi.cuGraphExecDestroy(exec);
                CudaDriverApi.cuGraphDestroy(graph);
            }
        }
        finally
        {
            CudaDriverApi.cuEventDestroy(start);
            CudaDriverApi.cuEventDestroy(end);
        }
    }

    private void AssertBits(float[] expected, float[] actual, string what)
    {
        int differ = 0;
        for (int i = 0; i < expected.Length; i++)
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                differ++;
        _output.WriteLine($"{what}: {differ} of {expected.Length} outputs differ in bits");
        Assert.True(differ == 0, $"{what}: {differ} of {expected.Length} outputs differ in bits");
    }

    private static float[] Query(int heads, int d)
    {
        var q = new float[heads * d];
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < d; i++)
                q[h * d + i] = MathF.Sin((h + 1) * 0.13f + (i + 3) * 0.011f);
        return q;
    }

    /// <summary>Attends the rows [0, len) - the same F16-exact values at every capacity, zeros past them - through
    /// <see cref="CudaFusedOps.TryGqaDecodeAttention"/> (the allocator's module), or through the partitioned route of
    /// <paramref name="kernels"/> when given; <paramref name="recorded"/> reads the length from a dyn block on the device.</summary>
    private static float[] Attend(CudaAllocator allocator, int heads, int kvHeads, int d, int len, int cap, float[] q, bool recorded,
        CudaKernels kernels = null)
    {
        var k = new float[(long)kvHeads * cap * d];
        var v = new float[(long)kvHeads * cap * d];
        for (int h = 0; h < kvHeads; h++)
            for (int t = 0; t < len; t++)
                for (int i = 0; i < d; i++)
                {
                    long at = ((long)h * cap + t) * d + i;
                    k[at] = (float)(System.Half)MathF.Cos((h + 2) * 0.19f + (t + 1) * 0.023f + (i + 1) * 0.007f);
                    v[at] = (float)(System.Half)(MathF.Sin((h + 3) * 0.11f + (t + 2) * 0.029f + (i + 1) * 0.005f) * 0.5f);
                }
        using var k32 = Tensor.FromArray(allocator, k);
        using var v32 = Tensor.FromArray(allocator, v);
        using var kFlat = new Tensor(allocator, DType.Float16, (long)kvHeads * cap * d);
        using var vFlat = new Tensor(allocator, DType.Float16, (long)kvHeads * cap * d);
        foreach (Tensor t in new[] { k32, v32, kFlat, vFlat })
            ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        allocator.Context.MakeCurrent();
        allocator.Kernels.LaunchConvertF32F16(P(k32), P(kFlat), k.Length, allocator.Stream.Handle);
        allocator.Kernels.LaunchConvertF32F16(P(v32), P(vFlat), v.Length, allocator.Stream.Handle);
        ((CudaStorage)kFlat.Storage).MarkDeviceModified();
        ((CudaStorage)vFlat.Storage).MarkDeviceModified();
        using var kc = kFlat.View(kvHeads, cap, d);
        using var vc = vFlat.View(kvHeads, cap, d);
        using var query = new Tensor(allocator, DType.Float32, 1, heads * d);
        query.SetElementsAsFloat(q);
        using var result = new Tensor(allocator, DType.Float32, 1, heads * d);
        float scale = 1f / MathF.Sqrt(d);
        using var dyn = recorded ? new CudaDecodeDynParams(allocator) : null;
        if (recorded)
        {
            Assert.True(dyn.IsValid);
            dyn.Write(len, 0, 0, 0);
            dyn.EnqueueUpload();
            dyn.Activate();
        }
        try
        {
            if (kernels == null)
            {
                Assert.True(CudaFusedOps.TryGqaDecodeAttention(result, query, kc, vc, heads, kvHeads, d, 0, len, cap, false, scale));
            }
            else
            {
                ((CudaStorage)query.Storage).EnsureDeviceCurrent();
                ((CudaStorage)result.Storage).EnsureDeviceCurrent();
                allocator.Context.MakeCurrent();
                Assert.True(CudaKernelOps.TryLaunchPartitionedGqaDecodeAttention(kernels, allocator, P(query), P(kc), P(vc), IntPtr.Zero,
                    P(result), heads, kvHeads, d, 0, len, cap, false, scale, 0, keyIsHalf: true,
                    recorded ? CudaDecodeDynParams.GetActiveDevicePtr(allocator) : IntPtr.Zero));
                ((CudaStorage)result.Storage).MarkDeviceModified();
            }
        }
        finally
        {
            if (recorded)
                CudaDecodeDynParams.Deactivate();
        }
        return result.GetElementsAsFloat(heads * d);
    }

    private static string RepoFile(params string[] parts)
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, Path.Combine(parts));
            if (File.Exists(candidate))
                return candidate;
        }
        throw new FileNotFoundException(string.Join("/", parts));
    }
}
