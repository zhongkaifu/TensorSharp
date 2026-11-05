using System.Collections;
using System.Reflection;
using TensorSharp;
using TensorSharp.MLX;
using TensorSharp.Models.Direct;
using TensorSharp.Runtime;
using Half = System.Half;

namespace InferenceWeb.Tests;

/// <summary>
/// Unsloth's Dynamic 2.0 GGUFs mix Q2_K, Q3_K and BF16 into the direct models' projections
/// (LTX 2.5 UD-Q4_K_XL: 4.4G such weights). MLX has no repacked form for them, and expanded to
/// dense F32 at load they took a 14 GB file to a 40 GB process. They now stay as their mapped
/// bytes and are dequantized to F32 per call, which must change nothing but the memory: every
/// weight the projection multiplies is the CPU dequantizer's, bit for bit.
/// </summary>
public sealed class DirectLinearMlxRawWeightTests
{
    // 4100 rows of 4096 F32 are more than the 64 MiB a slice may hold, so the wide layer runs
    // as two slices and a remainder row count.
    [MlxTheory]
    [InlineData(GgmlTensorType.Q2_K, 512, 24)]
    [InlineData(GgmlTensorType.Q3_K, 512, 24)]
    [InlineData(GgmlTensorType.BF16, 160, 24)]
    [InlineData(GgmlTensorType.Q2_K, 4096, 4100)]
    [InlineData(GgmlTensorType.Q3_K, 4096, 4100)]
    [InlineData(GgmlTensorType.BF16, 4096, 4100)]
    public void RawWeightsProjectExactlyWhatTheCpuDequantizes(GgmlTensorType type, int inDim, int outDim)
    {
        string path = Path.Combine(Path.GetTempPath(), "ts-direct-raw-" + Guid.NewGuid().ToString("N") + ".gguf");
        byte[] data = RandomWeight(type, inDim, outDim, seed: (int)type * 7919 + inDim);
        WriteGguf(path, "weight", type, inDim, outDim, data);
        try
        {
            using var gguf = new GgufFile(path);
            float[] reference = DirectOps.DequantTensor(gguf, "weight");
            Assert.All(reference, value => Assert.True(float.IsFinite(value)));
            using var allocator = new MlxAllocator();
            using var context = new DirectContext(allocator);
            var linear = DirectLinear.FromGguf(context, gguf, "weight", null, false);

            // Kept as the file's bytes: no dense F32 copy, one cached raw MLX array.
            Assert.Null(typeof(DirectLinear).GetField("_wCpu", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(linear));
            Assert.True(MlxQuantizedOps.CanPreloadDirectFloatType((int)type));
            // Q2_K and Q3_K also have quantized-matmul kernels now (MlxLowBitQuantKernels.cs); the
            // direct F32 projection still dequantizes them exactly (TryAddmmDirectFloat32), BF16 only here.
            Assert.Equal(type != GgmlTensorType.BF16, MlxQuantizedOps.CanPreloadQuantizedType((int)type));
            Assert.NotEqual(IntPtr.Zero, CachedMlxWeight(allocator.DeviceId, linear.CacheKey));

            // Every weight the projection multiplies is the CPU dequantizer's, bit for bit. Read
            // back before any GEMM: MLX picks TF32 or F32 once per process, LTX always F32.
            float[] dequantized = MlxQuantizedOps.ReadDirectRawRows(allocator, linear.CacheKey, 0, outDim);
            for (long i = 0; i < reference.LongLength; ++i)
                if (BitConverter.SingleToInt32Bits(reference[i]) != BitConverter.SingleToInt32Bits(dequantized[i]))
                    Assert.Fail($"{type} weight {i / inDim},{i % inDim}: GPU {dequantized[i]:R} vs CPU {reference[i]:R}");
            float[] tail = MlxQuantizedOps.ReadDirectRawRows(allocator, linear.CacheKey, outDim - 3, 3);
            Assert.Equal(reference.AsSpan((outDim - 3) * inDim).ToArray(), tail);

            // And a real product against a double-precision reference, as tight as the process's
            // GEMM mode allows (TF32 keeps 10 mantissa bits of each operand).
            bool tf32 = Environment.GetEnvironmentVariable("MLX_ENABLE_TF32") != "0";
            double tolerance = tf32 ? 4e-3 : 1e-5;
            var random = new Random(17);
            var x = new float[3, inDim];
            for (int r = 0; r < 3; ++r) for (int c = 0; c < inDim; ++c) x[r, c] = (float)(random.NextDouble() * 4 - 2);
            using (var input = Tensor.FromArray(allocator, x))
            using (var output = linear.Forward(input))
            {
                float[] actual = output.GetElementsAsFloat(3 * outDim);
                for (int r = 0; r < 3; ++r)
                    for (int row = 0; row < outDim; ++row)
                    {
                        double expected = 0, magnitude = 0;
                        for (int c = 0; c < inDim; ++c)
                        {
                            double term = (double)x[r, c] * reference[(long)row * inDim + c];
                            expected += term; magnitude += Math.Abs(term);
                        }
                        Assert.True(Math.Abs(actual[(long)r * outDim + row] - expected) <= tolerance * magnitude + 1e-6,
                            $"{type} [{r},{row}]: {actual[(long)r * outDim + row]} vs {expected} (|terms| {magnitude}, tf32 {tf32})");
                    }
            }

            // Released and re-wrapped from the same mapped bytes on the next call.
            context.ReleaseDeviceResidency();
            Assert.Equal(IntPtr.Zero, CachedMlxWeight(allocator.DeviceId, linear.CacheKey));
            using (var input = Tensor.FromArray(allocator, x))
            using (var output = linear.Forward(input))
                Assert.Equal(3L * outDim, output.ElementCount());
            Assert.NotEqual(IntPtr.Zero, CachedMlxWeight(allocator.DeviceId, linear.CacheKey));
            Assert.Equal(reference.AsSpan(0, inDim).ToArray(), MlxQuantizedOps.ReadDirectRawRows(allocator, linear.CacheKey, 0, 1));
        }
        finally { File.Delete(path); }
    }

    /// <summary>Valid blocks with random quants and scales: Q2_K is scales[16] qs[64] d dmin,
    /// Q3_K hmask[32] qs[64] scales[12] d (super-block scales are small finite halves).</summary>
    private static byte[] RandomWeight(GgmlTensorType type, int inDim, int outDim, int seed)
    {
        var random = new Random(seed);
        if (type == GgmlTensorType.BF16)
        {
            var bytes = new byte[checked((long)inDim * outDim * 2)];
            for (long i = 0; i < (long)inDim * outDim; ++i)
            {
                float value = (float)(random.NextDouble() * 2 - 1) * MathF.Pow(2, random.Next(-12, 4));
                ushort bf16 = (ushort)(BitConverter.SingleToUInt32Bits(value) >> 16);
                bytes[2 * i] = (byte)bf16; bytes[2 * i + 1] = (byte)(bf16 >> 8);
            }
            return bytes;
        }
        int blockBytes = type == GgmlTensorType.Q2_K ? 84 : 110;
        int blocks = checked(outDim * (inDim / 256));
        var data = new byte[(long)blocks * blockBytes];
        random.NextBytes(data);
        for (long b = 0; b < blocks; ++b)
        {
            long at = b * blockBytes;
            void PutHalf(long offset, float value)
            {
                ushort bits = BitConverter.HalfToUInt16Bits((Half)value);
                data[offset] = (byte)bits; data[offset + 1] = (byte)(bits >> 8);
            }
            float scale = (float)(random.NextDouble() * 0.02 + 1e-4) * MathF.Pow(2, random.Next(-6, 3));
            if (type == GgmlTensorType.Q2_K)
            {
                PutHalf(at + 80, scale);
                PutHalf(at + 82, (float)(random.NextDouble() * 0.02) * MathF.Pow(2, random.Next(-10, 3)));
            }
            else
                PutHalf(at + 108, scale);
        }
        return data;
    }

    private static void WriteGguf(string path, string name, GgmlTensorType type, int inDim, int outDim, byte[] data)
    {
        using var writer = new BinaryWriter(File.Create(path));
        writer.Write(0x46554747U); writer.Write(3U); writer.Write(1UL); writer.Write(0UL);
        writer.Write((ulong)name.Length); writer.Write(System.Text.Encoding.UTF8.GetBytes(name));
        writer.Write(2U); writer.Write((ulong)inDim); writer.Write((ulong)outDim);
        writer.Write((uint)type); writer.Write(0UL);
        while (writer.BaseStream.Position % 32 != 0) writer.Write((byte)0);
        writer.Write(data);
    }

    private static IntPtr CachedMlxWeight(int device, IntPtr host)
    {
        const BindingFlags flags = BindingFlags.Static | BindingFlags.NonPublic;
        Type type = typeof(MlxQuantizedOps);
        object sync = type.GetField("Sync", flags)!.GetValue(null)!;
        var cache = (IDictionary)type.GetField("Cache", flags)!.GetValue(null)!;
        Type keyType = type.GetNestedType("CacheKey", BindingFlags.NonPublic)!;
        object key = Activator.CreateInstance(keyType, device, host)!;
        lock (sync)
        {
            if (!cache.Contains(key)) return IntPtr.Zero;
            object node = cache[key]!;
            object entry = node.GetType().GetProperty("Value")!.GetValue(node)!;
            return ((MlxNative.MlxArray)entry.GetType().GetField("Weight")!.GetValue(entry)!).Ctx;
        }
    }
}
