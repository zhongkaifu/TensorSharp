using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class CudaMoeFusionTests
{
    [CudaTheory]
    [InlineData(1, 32, 0f)]
    [InlineData(3, 96, 0f)] // trailing warp in the final CUDA block
    [InlineData(3, 96, 1.5f)]
    [InlineData(7, 256, 7f)]
    [InlineData(17, 2048, 0f)]
    [InlineData(17, 2048, 7f)]
    [InlineData(1, 32, -1f)]
    [InlineData(5, 96, float.NaN)]
    [InlineData(0, 32, 7f)]
    public void FusedQuantization_IsBitwiseEqualToTwoLaunches(int rows, int width, float limit)
    {
        using var allocator = new CudaAllocator();
        Assert.True(allocator.Kernels?.SupportsFusedMoeQuantize, "This test requires freshly compiled fused CUDA kernels.");
        int n = rows * width;
        const int guard = 64;
        float[] g = Enumerable.Range(0, n + guard).Select(i =>
            i < 32 && n > 32 ? 0f : i % 11 == 0 ? -100f : 9f * MathF.Sin(i * 0.1237f)).ToArray();
        float[] u = Enumerable.Range(0, n + guard).Select(i => 3f * MathF.Cos(i * 0.239f)).ToArray();
        if (float.IsNaN(limit))
        {
            g[33] = float.NaN;
            g[34] = float.PositiveInfinity;
            g[35] = float.NegativeInfinity;
            g[36] = float.Epsilon;
            u[37] = float.NegativeInfinity;
            u[38] = -0f;
        }
        foreach (bool split in new[] { false, true })
        {
            using Tensor reference = Upload(allocator, g), fused = Upload(allocator, g), up = Upload(allocator, u);
            int bytes = split ? n : n / 32 * CudaKernels.Q81BlockBytes;
            byte[] sentinel = Enumerable.Repeat((byte)0xA5, bytes + guard).ToArray();
            using Tensor qr = Upload(allocator, sentinel), qf = Upload(allocator, sentinel);
            float[] ds = Enumerable.Repeat(-123f, n / 32 + guard).ToArray();
            using Tensor dr = Upload(allocator, ds), df = Upload(allocator, ds);
            if (limit == 0f && rows > 0)
            {
                // The generic Qwen path historically used the unclamped operator.
                allocator.Kernels.LaunchSiluMulF32(Ptr(reference), Ptr(reference), Ptr(up), n, allocator.Stream.Handle);
                if (split)
                    allocator.Kernels.LaunchQuantizeQ81SplitRows(Ptr(reference), Ptr(qr), Ptr(dr), width, rows, allocator.Stream.Handle);
                else
                    allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(reference), Ptr(qr), width, rows, allocator.Stream.Handle, true);
            }
            else
                allocator.Kernels.LaunchSiluMulClampQuantizeQ81(Ptr(reference), Ptr(up), Ptr(qr), split ? Ptr(dr) : IntPtr.Zero,
                    width, rows, limit, allocator.Stream.Handle, fused: false);
            allocator.Kernels.LaunchSiluMulClampQuantizeQ81(Ptr(fused), Ptr(up), Ptr(qf), split ? Ptr(df) : IntPtr.Zero,
                width, rows, limit, allocator.Stream.Handle, fused: true);
            Assert.Equal(Read<byte>(qr), Read<byte>(qf));
            Assert.Equal(Bits(Read<float>(reference)), Bits(Read<float>(fused)));
            Assert.Equal(Bits(Read<float>(dr)), Bits(Read<float>(df)));
            Assert.Equal(sentinel.Skip(bytes), Read<byte>(qf).Skip(bytes));
            Assert.Equal(Bits(g.Skip(n).ToArray()), Bits(Read<float>(fused).Skip(n).ToArray()));
            Assert.Equal(Bits(ds.Skip(n / 32).ToArray()), Bits(Read<float>(df).Skip(n / 32).ToArray()));
        }
        Assert.Throws<ArgumentOutOfRangeException>(() => allocator.Kernels.LaunchSiluMulClampQuantizeQ81(
            IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, 33, 1, 0, allocator.Stream.Handle));
    }

    [Theory]
    [InlineData(1, 4, 2, 64, 96)]
    [InlineData(32, 8, 2, 256, 128)]
    [InlineData(16, 8, 1, 512, 64)] // input capacity exceeds expert hidden capacity
    public void SharedScratch_AccountsForEveryAllocation_AndReducesBytes(int nt, int experts, int used, int e, int ff)
    {
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        long[] totals = new long[2];
        foreach (bool shared in new[] { false, true })
        {
            var owned = new List<Tensor>();
            try
            {
                var scratch = new CudaMoeScratch((type, sizes) =>
                {
                    var tensor = new Tensor(allocator, type, sizes);
                    owned.Add(tensor);
                    return tensor;
                }, nt, experts, used, e, ff, shareActivations: shared);
                long bytes = owned.Sum(t => t.ElementCount() * t.ElementType.Size());
                totals[shared ? 1 : 0] = bytes;
                Assert.Equal(bytes, CudaMoeScratch.Bytes(nt, used, e, ff, experts, shared));
                if (shared)
                {
                    Assert.Same(scratch.ActQ8A, scratch.ActQ8B);
                    Assert.Same(scratch.ActQ8A, scratch.SplitQsA);
                    Assert.Same(scratch.ActQ8B, scratch.SplitQsB);
                    Assert.Same(scratch.SplitDA, scratch.SplitDB);
                    Assert.True(scratch.ActQ8A.ElementCount() >= (long)nt * e / 32 * 36);
                    Assert.True(scratch.ActQ8B.ElementCount() >= (long)nt * used * ff / 32 * 36);
                }
            }
            finally { foreach (Tensor t in owned) t.Dispose(); }
        }
        Assert.True(totals[1] < totals[0]);
    }

    [CudaTheory]
    [InlineData(64, 96)]   // grouped warp fallback and decode, partial launch blocks
    [InlineData(256, 256)] // tensor-core prefill followed by per-slot decode
    public void SharedScratch_AndFusion_PreserveCompleteExpertOutputAcrossUbatchSizes(int e, int ff)
    {
        const int cap = 19, experts = 4, used = 2;
        using var allocator = new CudaAllocator();
        Assert.True(allocator.Kernels?.SupportsFusedMoeQuantize);
        using var dk = Dsv4Kernels.Create();
        using Tensor gate = Upload(allocator, Q80Weights(experts * ff, e, 1));
        using Tensor up = Upload(allocator, Q80Weights(experts * ff, e, 2));
        using Tensor down = Upload(allocator, Q80Weights(experts * e, ff, 3));
        var g = new DeviceWeight { Ptr = Ptr(gate), Type = 8, Ne0 = e, Ne1 = ff, RowBytes = e / 32 * 34 };
        var u = new DeviceWeight { Ptr = Ptr(up), Type = 8, Ne0 = e, Ne1 = ff, RowBytes = e / 32 * 34 };
        var d = new DeviceWeight { Ptr = Ptr(down), Type = 8, Ne0 = ff, Ne1 = e, RowBytes = ff / 32 * 34 };
        using Tensor cur = Upload(allocator, Enumerable.Range(0, cap * e).Select(i => MathF.Sin(i * .237f)).ToArray());
        using Tensor sharedDown = Upload(allocator, Enumerable.Repeat(.125f, cap * e).ToArray());
        using var outputR = new Tensor(allocator, DType.Float32, cap, e);
        using var outputF = new Tensor(allocator, DType.Float32, cap, e);
        var owned = new List<Tensor>();
        Tensor Alloc(DType type, long[] sizes) { var t = new Tensor(allocator, type, sizes); owned.Add(t); return t; }
        try
        {
            var reference = new CudaMoeScratch(Alloc, cap, experts, used, e, ff, false);
            var fused = new CudaMoeScratch(Alloc, cap, experts, used, e, ff, true);
            foreach (var scratch in new[] { reference, fused })
            {
                scratch.Sel.SetElementsAsInt(Enumerable.Range(0, cap * used).Select(i => i % experts).ToArray());
                scratch.SelW.SetElementsAsFloat(Enumerable.Repeat(.5f, cap * used).ToArray());
                scratch.Sel.Storage.EnsureDeviceCurrent();
                scratch.SelW.Storage.EnsureDeviceCurrent();
            }
            foreach (int nt in new[] { cap, 1, 3, cap, 1 })
            {
                CudaMoe.Experts(dk, allocator.Kernels, reference, g, u, d, cur, sharedDown, outputR,
                    nt, used, experts, e, ff, 7f, 3, allocator.Stream.Handle, fused: false);
                CudaMoe.Experts(dk, allocator.Kernels, fused, g, u, d, cur, sharedDown, outputF,
                    nt, used, experts, e, ff, 7f, 3, allocator.Stream.Handle, fused: true);
                float[] actual = Read<float>(outputF).Take(nt * e).ToArray();
                Assert.All(actual, v => Assert.True(float.IsFinite(v)));
                Assert.Equal(Bits(Read<float>(outputR).Take(nt * e).ToArray()), Bits(actual));
            }
        }
        finally { allocator.Synchronize(); foreach (Tensor t in owned) t.Dispose(); }
    }

    private static byte[] Q80Weights(int rows, int width, int seed)
    {
        var rng = new Random(seed);
        var result = new byte[rows * width / 32 * 34 + 16]; // vector tail slack
        for (int i = 0; i < result.Length - 16; i += 34)
        {
            BitConverter.TryWriteBytes(result.AsSpan(i, 2), BitConverter.HalfToUInt16Bits((System.Half).002f));
            for (int j = 2; j < 34; j++) result[i + j] = unchecked((byte)rng.Next(-127, 128));
        }
        return result;
    }

    private static Tensor Upload(CudaAllocator allocator, Array data)
    {
        var result = Tensor.FromArray(allocator, data);
        result.Storage.EnsureDeviceCurrent();
        return result;
    }
    private static IntPtr Ptr(Tensor t) => Dsv4CudaEngine.Ptr(t);
    private static T[] Read<T>(Tensor t)
    {
        ((CudaStorage)t.Storage).MarkDeviceModified();
        var result = new T[checked((int)t.ElementCount())];
        t.CopyToArray(result);
        return result;
    }
    private static int[] Bits(float[] a) => a.Select(BitConverter.SingleToInt32Bits).ToArray();
}
