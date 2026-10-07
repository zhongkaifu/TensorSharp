using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;

// Focused Strata-inspired fusion/activation-lifetime A/B. Synthetic finite Q8_0
// weights, CUDA event timing, complete routed expert blocks, identical inputs.
internal static class MoeFusionBench
{
    public static void Run(CudaAllocator allocator, int warmup, int iterations)
    {
        if (iterations <= 0 || warmup < 0) throw new ArgumentOutOfRangeException(nameof(iterations));
        if (allocator.Kernels?.SupportsFusedMoeQuantize != true)
            throw new InvalidOperationException("Compile the fused CUDA kernels before benchmarking them.");
        using var dk = Dsv4Kernels.Create();
        foreach ((int nt, int width, int ff) in new[] { (1, 2048, 768), (8, 2048, 768), (19, 96, 96), (32, 2048, 768) })
            Case(allocator, dk, nt, width, ff, warmup, iterations);
    }

    private static void Case(CudaAllocator allocator, Dsv4Kernels dk, int nt, int e, int ff, int warmup, int iterations)
    {
        const int experts = 16, used = 8;
        using Tensor gate = Upload(allocator, Weights(experts * ff, e, 1));
        using Tensor up = Upload(allocator, Weights(experts * ff, e, 2));
        using Tensor down = Upload(allocator, Weights(experts * e, ff, 3));
        using Tensor cur = Upload(allocator, Enumerable.Range(0, nt * e).Select(i => MathF.Sin(i * .237f)).ToArray());
        using Tensor shared = Upload(allocator, new float[nt * e]);
        using var baselineOutput = new Tensor(allocator, DType.Float32, nt, e);
        using var fusedOutput = new Tensor(allocator, DType.Float32, nt, e);
        var g = new DeviceWeight { Ptr = Ptr(gate), Type = 8, Ne0 = e, Ne1 = ff, RowBytes = e / 32 * 34 };
        var u = new DeviceWeight { Ptr = Ptr(up), Type = 8, Ne0 = e, Ne1 = ff, RowBytes = e / 32 * 34 };
        var d = new DeviceWeight { Ptr = Ptr(down), Type = 8, Ne0 = ff, Ne1 = e, RowBytes = ff / 32 * 34 };
        var owned = new List<Tensor>();
        Tensor Alloc(DType type, long[] sizes) { var t = new Tensor(allocator, type, sizes); owned.Add(t); return t; }
        IntPtr stream = allocator.Stream.Handle;
        CudaDriverApi.cuEventCreate(out IntPtr start, 0).ThrowOnError();
        CudaDriverApi.cuEventCreate(out IntPtr end, 0).ThrowOnError();
        try
        {
            var baseline = new CudaMoeScratch(Alloc, nt, experts, used, e, ff, false);
            var fused = new CudaMoeScratch(Alloc, nt, experts, used, e, ff, true);
            foreach (CudaMoeScratch scratch in new[] { baseline, fused })
            {
                scratch.Sel.SetElementsAsInt(Enumerable.Range(0, nt * used).Select(i => i % experts).ToArray());
                scratch.SelW.SetElementsAsFloat(Enumerable.Repeat(1f / used, nt * used).ToArray());
                scratch.Sel.Storage.EnsureDeviceCurrent();
                scratch.SelW.Storage.EnsureDeviceCurrent();
            }
            void Run(bool optimized) => CudaMoe.Experts(dk, allocator.Kernels, optimized ? fused : baseline,
                g, u, d, cur, shared, optimized ? fusedOutput : baselineOutput, nt, used, experts, e, ff, 7f, 16, stream,
                fused: optimized);
            for (int i = 0; i < warmup; i++) { Run(false); Run(true); }
            allocator.Synchronize();
            Run(false); Run(true);
            float[] Read(Tensor t)
            {
                ((CudaStorage)t.Storage).MarkDeviceModified();
                return t.GetElementsAsFloat(nt * e);
            }
            float[] a = Read(baselineOutput), b = Read(fusedOutput);
            if (a.Any(v => !float.IsFinite(v)) || !a.Select(BitConverter.SingleToInt32Bits).SequenceEqual(b.Select(BitConverter.SingleToInt32Bits)))
                throw new InvalidOperationException("MoE output is not finite and bitwise identical.");
            double Time(bool optimized)
            {
                CudaDriverApi.cuEventRecord(start, stream).ThrowOnError();
                for (int i = 0; i < iterations; i++) Run(optimized);
                CudaDriverApi.cuEventRecord(end, stream).ThrowOnError();
                CudaDriverApi.cuEventSynchronize(end).ThrowOnError();
                CudaDriverApi.cuEventElapsedTime(out float ms, start, end).ThrowOnError();
                return ms * 1000 / iterations;
            }
            var baselineTimes = new List<double>(); var fusedTimes = new List<double>();
            for (int round = 0; round < 8; round++)
            {
                if (round % 2 == 0) { baselineTimes.Add(Time(false)); fusedTimes.Add(Time(true)); }
                else { fusedTimes.Add(Time(true)); baselineTimes.Add(Time(false)); }
            }
            double Median(List<double> times) { times.Sort(); return (times[3] + times[4]) / 2; }
            double oldUs = Median(baselineTimes), newUs = Median(fusedTimes);
            Console.WriteLine($"moe_q8_0 nt={nt} e={e} ff={ff} baseline_us={oldUs:F3} fused_us={newUs:F3} speedup={oldUs/newUs:F4} " +
                $"baseline_scratch_bytes={CudaMoeScratch.Bytes(nt, used, e, ff, experts, false)} " +
                $"fused_scratch_bytes={CudaMoeScratch.Bytes(nt, used, e, ff, experts, true)} bitwise=true rounds=8");
        }
        finally
        {
            allocator.Synchronize();
            foreach (Tensor t in owned) t.Dispose();
            CudaDriverApi.cuEventDestroy(start); CudaDriverApi.cuEventDestroy(end);
        }
    }
    private static byte[] Weights(int rows, int width, int seed)
    {
        var rng = new Random(seed);
        var result = new byte[rows * width / 32 * 34 + 16];
        for (int i = 0; i < result.Length - 16; i += 34)
        {
            BitConverter.TryWriteBytes(result.AsSpan(i, 2), BitConverter.HalfToUInt16Bits((System.Half).002f));
            for (int j = 2; j < 34; j++) result[i + j] = unchecked((byte)rng.Next(-127, 128));
        }
        return result;
    }
    private static Tensor Upload(CudaAllocator allocator, Array data)
    {
        Tensor t = Tensor.FromArray(allocator, data);
        t.Storage.EnsureDeviceCurrent();
        return t;
    }
    private static IntPtr Ptr(Tensor t) => Dsv4CudaEngine.Ptr(t);
}
