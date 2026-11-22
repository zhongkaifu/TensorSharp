using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;

// Device time of the direct CUDA executor's few-row products of i-quant weights (Dsv4Kernels: the q8_1 rows, DenseQ81Rows,
// and the F32 rows, DenseF32Rows, for IQ3_XXS and IQ3_S), at Muse-Glimmer-30B's decode shapes: the FFN down projection
// (IQ3_XXS, 19968 -> 6656), the attention output (IQ3_S, 8192 -> 6656) and an FFN gate/up half (IQ2_S, 6656 -> 19968), 1 to
// 16 rows a call. Synthetic rows (random bytes under a finite scale), CUDA-event timing, the kernels alternated over 8
// rounds and the median printed; also checks that each row of an F32-rows call is the one-row call's bits. Prints only.
//   dotnet run -c Release --project benchmarks/CudaQuantizedMatmulBench -- --dense-rows --iterations 200 --warmup 20
internal static class DenseRowsBench
{
    public static void Run(CudaAllocator allocator, int warmup, int iterations)
    {
        if (iterations <= 0 || warmup < 0) throw new ArgumentOutOfRangeException(nameof(iterations));
        using var dk = Dsv4Kernels.Create();
        if (!dk.HasDenseF32Rows) throw new InvalidOperationException("This dsv4 kernel module has no F32 dense-row kernels.");
        // (ggml type, name, bytes a 256-value super-block, input, output)
        foreach ((int type, string name, int blockBytes, int inDim, int outDim) in new[]
                 { (18, "IQ3_XXS", 98, 19968, 6656), (21, "IQ3_S", 110, 8192, 6656), (22, "IQ2_S", 82, 6656, 19968) })
            Case(allocator, dk, type, name, (long)inDim / 256 * blockBytes, blockBytes, inDim, outDim, warmup, iterations);
    }

    private static void Case(CudaAllocator allocator, Dsv4Kernels dk, int type, string name, long rowBytes, int blockBytes,
        int inDim, int outDim, int warmup, int iterations)
    {
        const int MaxRows = 16;
        bool f32 = type is 18 or 21;
        IntPtr stream = allocator.Stream.Handle;
        var act = new float[MaxRows * inDim];
        for (int t = 0; t < MaxRows; t++)
            for (int k = 0; k < inDim; k++)
                act[t * inDim + k] = MathF.Sin(0.0173f * (k + 1) * (t + 2));
        using Tensor w = Upload(allocator, Weights(outDim, rowBytes, blockBytes, seed: 5));
        using Tensor x = Upload(allocator, act);
        using var q81 = new Tensor(allocator, DType.UInt8, (long)MaxRows * (inDim / 32) * CudaKernels.Q81BlockBytes);
        using var y = new Tensor(allocator, DType.Float32, MaxRows, outDim);
        using var one = new Tensor(allocator, DType.Float32, 1, outDim);
        allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(x), Ptr(q81), inDim, MaxRows, stream, true);
        CudaDriverApi.cuEventCreate(out IntPtr start, 0).ThrowOnError();
        CudaDriverApi.cuEventCreate(out IntPtr end, 0).ThrowOnError();
        try
        {
            double mb = (double)rowBytes * outDim / 1e6;
            foreach (int rows in new[] { 1, 2, 4, 8, 16 })
            {
                void Q81() => dk.DenseQ81Rows(Ptr(w), Ptr(q81), Ptr(y), type, outDim, inDim, rowBytes, rows, stream);
                void F32() => dk.DenseF32Rows(Ptr(w), Ptr(x), Ptr(y), type, outDim, inDim, rowBytes, rows, stream);
                double Time(Action launch)
                {
                    for (int i = 0; i < warmup; i++) launch();
                    CudaDriverApi.cuEventRecord(start, stream).ThrowOnError();
                    for (int i = 0; i < iterations; i++) launch();
                    CudaDriverApi.cuEventRecord(end, stream).ThrowOnError();
                    CudaDriverApi.cuEventSynchronize(end).ThrowOnError();
                    CudaDriverApi.cuEventElapsedTime(out float ms, start, end).ThrowOnError();
                    return ms * 1000 / iterations;
                }
                var q = new List<double>(); var f = new List<double>();
                for (int round = 0; round < 8; round++)
                {
                    if (round % 2 == 0) { q.Add(Time(Q81)); if (f32) f.Add(Time(F32)); }
                    else { if (f32) f.Add(Time(F32)); q.Add(Time(Q81)); }
                }
                string line = $"{name} {inDim}->{outDim} ({mb:F1} MB) rows {rows}: q8_1 {Median(q):F1} us ({mb * 1e3 / Median(q):F0} GB/s)";
                if (f32)
                {
                    // Each row of the call against the one-row call on that row alone.
                    F32();
                    float[] all = Read(y, rows * outDim);
                    bool bits = true;
                    for (int r = 0; r < rows && bits; r++)
                    {
                        dk.DenseF32Rows(Ptr(w), Ptr(x) + r * inDim * sizeof(float), Ptr(one), type, outDim, inDim, rowBytes, 1, stream);
                        float[] solo = Read(one, outDim);
                        for (int o = 0; o < outDim && bits; o++)
                            bits = BitConverter.SingleToInt32Bits(all[r * outDim + o]) == BitConverter.SingleToInt32Bits(solo[o]) && float.IsFinite(solo[o]);
                    }
                    line += $", f32 {Median(f):F1} us ({mb * 1e3 / Median(f):F0} GB/s), rows are the one-row call's bits: {bits}";
                }
                Console.WriteLine(line);
            }
        }
        finally
        {
            allocator.Synchronize();
            CudaDriverApi.cuEventDestroy(start); CudaDriverApi.cuEventDestroy(end);
        }
    }

    private static double Median(List<double> times)
    {
        var s = times.OrderBy(v => v).ToList();
        return (s[(s.Count - 1) / 2] + s[s.Count / 2]) / 2;
    }

    /// <summary>Random rows whose super-blocks each start with a small finite F16 scale (every i-quant's d comes first).</summary>
    private static byte[] Weights(int rows, long rowBytes, int blockBytes, int seed)
    {
        var bytes = new byte[rows * rowBytes];
        new Random(seed).NextBytes(bytes);
        for (long b = 0; b < bytes.Length; b += blockBytes)
            BitConverter.TryWriteBytes(bytes.AsSpan((int)b, 2), BitConverter.HalfToUInt16Bits((System.Half)0.01f));
        return bytes;
    }

    private static float[] Read(Tensor t, int count)
    {
        ((CudaStorage)t.Storage).MarkDeviceModified();
        return t.GetElementsAsFloat(count);
    }

    private static Tensor Upload(CudaAllocator allocator, Array data)
    {
        Tensor t = Tensor.FromArray(allocator, data);
        t.Storage.EnsureDeviceCurrent();
        return t;
    }

    private static IntPtr Ptr(Tensor t) => Dsv4CudaEngine.Ptr(t);
}
