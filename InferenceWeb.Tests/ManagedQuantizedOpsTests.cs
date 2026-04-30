using System.Buffers.Binary;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace InferenceWeb.Tests;

public class ManagedQuantizedOpsTests
{
    [Fact]
    public void ShouldStoreWeightQuantized_UsesManagedCpuSupportMatrix()
    {
        var supported = new GgufTensorInfo
        {
            Name = "blk.0.attn_q.weight",
            Shape = new ulong[] { 256, 128 },
            Type = GgmlTensorType.Q4_K,
        };

        var unsupported = new GgufTensorInfo
        {
            Name = "blk.0.attn_q.weight",
            Shape = new ulong[] { 256, 128 },
            Type = GgmlTensorType.IQ2_XXS,
        };

        Assert.True(ModelBase.ShouldStoreWeightQuantized(BackendType.Cpu, supported));
        Assert.False(ModelBase.ShouldStoreWeightQuantized(BackendType.Cpu, unsupported));
    }

    [Fact]
    public void NativeDequant_DequantizesQ80InManagedCode()
    {
        byte[] raw = new byte[2 + 32];
        WriteHalf(raw, 0, 0.5f);
        raw[2] = unchecked((byte)(sbyte)2);
        raw[3] = unchecked((byte)(sbyte)-4);
        raw[4] = unchecked((byte)(sbyte)7);

        float[] dst = new float[32];
        NativeDequant.DequantizeToFloat32((int)GgmlTensorType.Q8_0, raw, 0, dst, 0, 32);

        Assert.Equal(1.0f, dst[0], 5);
        Assert.Equal(-2.0f, dst[1], 5);
        Assert.Equal(3.5f, dst[2], 5);
        Assert.Equal(0.0f, dst[3], 5);
    }

    [Fact]
    public void NativeDequant_DequantizesQ4KInManagedCode()
    {
        byte[] raw = new byte[144];
        WriteHalf(raw, 0, 0.5f);
        WriteHalf(raw, 2, 0.25f);

        raw[4] = 2; // scale for sub-block 0
        raw[8] = 1; // min for sub-block 0
        raw[5] = 3; // scale for sub-block 1
        raw[9] = 4; // min for sub-block 1

        raw[16] = 0x21; // first low nibble = 1, first high nibble = 2

        float[] dst = new float[256];
        NativeDequant.DequantizeToFloat32((int)GgmlTensorType.Q4_K, raw, 0, dst, 0, 256);

        Assert.Equal(0.75f, dst[0], 5);
        Assert.Equal(-0.25f, dst[1], 5);
        Assert.Equal(2.0f, dst[32], 5);
        Assert.Equal(-1.0f, dst[33], 5);
    }

    [Fact]
    public void NativeDequant_DequantizesQ4KToUnmanagedBuffer()
    {
        byte[] raw = new byte[144];
        WriteHalf(raw, 0, 0.5f);
        WriteHalf(raw, 2, 0.25f);

        raw[4] = 2;
        raw[8] = 1;
        raw[5] = 3;
        raw[9] = 4;
        raw[16] = 0x21;

        IntPtr src = Marshal.AllocHGlobal(raw.Length);
        IntPtr dst = Marshal.AllocHGlobal(256 * sizeof(float));
        try
        {
            Marshal.Copy(raw, 0, src, raw.Length);
            NativeDequant.DequantizeToFloat32Native((int)GgmlTensorType.Q4_K, src, dst, 256);

            float[] managed = new float[256];
            Marshal.Copy(dst, managed, 0, managed.Length);

            Assert.Equal(0.75f, managed[0], 5);
            Assert.Equal(-0.25f, managed[1], 5);
            Assert.Equal(2.0f, managed[32], 5);
            Assert.Equal(-1.0f, managed[33], 5);
        }
        finally
        {
            Marshal.FreeHGlobal(src);
            Marshal.FreeHGlobal(dst);
        }
    }

    [Fact]
    public void NativeDequant_RowSizeSupportsIq2Xxs()
    {
        Assert.Equal(
            GgufFile.GetTypeSize(GgmlTensorType.IQ2_XXS),
            NativeDequant.RowSize((int)GgmlTensorType.IQ2_XXS, 256));
    }

    [Fact]
    public void DotRowBatchToFloat32_MatchesDequantizedDotForQ80()
    {
        byte[] raw = new byte[2 + 32];
        WriteHalf(raw, 0, 0.5f);
        raw[2] = unchecked((byte)(sbyte)2);
        raw[3] = unchecked((byte)(sbyte)-4);
        raw[4] = unchecked((byte)(sbyte)7);

        float[] inputs = new float[64];
        for (int i = 0; i < 32; i++)
        {
            inputs[i] = i * 0.125f;
            inputs[32 + i] = 1.0f - i * 0.03125f;
        }

        float[] actual = new float[2];
        ManagedQuantizedOps.DotRowBatchToFloat32(
            (int)GgmlTensorType.Q8_0,
            raw,
            0,
            inputs,
            0,
            32,
            2,
            32,
            actual,
            0);

        float[] dequantized = new float[32];
        NativeDequant.DequantizeToFloat32((int)GgmlTensorType.Q8_0, raw, 0, dequantized, 0, 32);

        Assert.Equal(Dot(dequantized, inputs, 0, 32), actual[0], 5);
        Assert.Equal(Dot(dequantized, inputs, 32, 32), actual[1], 5);
    }

    [Fact]
    public void DotRowBatchToFloat32_MatchesDequantizedDotForQ4K()
    {
        byte[] raw = new byte[144];
        WriteHalf(raw, 0, 0.5f);
        WriteHalf(raw, 2, 0.25f);

        raw[4] = 2;
        raw[8] = 1;
        raw[5] = 3;
        raw[9] = 4;
        raw[16] = 0x21;
        raw[48] = 0x34;
        raw[80] = 0x87;
        raw[112] = 0x65;

        float[] inputs = new float[256 * 3];
        for (int row = 0; row < 3; row++)
        {
            int baseOffset = row * 256;
            for (int i = 0; i < 256; i++)
            {
                inputs[baseOffset + i] = (row + 1) * 0.01f * ((i % 17) - 8);
            }
        }

        float[] actual = new float[3];
        ManagedQuantizedOps.DotRowBatchToFloat32(
            (int)GgmlTensorType.Q4_K,
            raw,
            0,
            inputs,
            0,
            256,
            3,
            256,
            actual,
            0);

        float[] dequantized = new float[256];
        NativeDequant.DequantizeToFloat32((int)GgmlTensorType.Q4_K, raw, 0, dequantized, 0, 256);

        Assert.Equal(Dot(dequantized, inputs, 0, 256), actual[0], 5);
        Assert.Equal(Dot(dequantized, inputs, 256, 256), actual[1], 5);
        Assert.Equal(Dot(dequantized, inputs, 512, 256), actual[2], 5);
    }

    [Fact]
    public void TryAddmmQuantizedToFloat32_UsesDirectQ80Path()
    {
        const int inDim = 64;
        const int outDim = 5;
        const int rows = 3;

        float[] weightsF32 = Enumerable.Range(0, outDim * inDim)
            .Select(i => MathF.Sin(i * 0.07f) * 0.35f)
            .ToArray();
        byte[] weightsQ80 = QuantizeRowsQ80(weightsF32, outDim, inDim);

        float[] input = Enumerable.Range(0, rows * inDim)
            .Select(i => MathF.Cos(i * 0.11f) * 0.2f)
            .ToArray();
        float[] actual = new float[rows * outDim];

        Assert.True(ManagedQuantizedOps.TryAddmmQuantizedToFloat32(
            (int)GgmlTensorType.Q8_0,
            weightsQ80,
            0,
            inDim,
            outDim,
            input,
            0,
            inDim,
            rows,
            actual,
            0,
            outDim));

        float[] expected = DequantizedMatmul(weightsQ80, GgmlTensorType.Q8_0, outDim, inDim, input, rows);
        AssertClose(expected, actual, 0.03f);
    }

    [Fact]
    public void TryAddmmQuantizedToFloat32_UsesDirectQ4KPath()
    {
        const int inDim = 256;
        const int outDim = 2;
        const int rows = 2;

        byte[] row = new byte[144];
        WriteHalf(row, 0, 0.5f);
        WriteHalf(row, 2, 0.25f);
        row[4] = 2;
        row[8] = 1;
        row[5] = 3;
        row[9] = 4;
        row[16] = 0x21;
        row[48] = 0x34;
        row[80] = 0x87;
        row[112] = 0x65;

        byte[] weights = new byte[row.Length * outDim];
        Buffer.BlockCopy(row, 0, weights, 0, row.Length);
        Buffer.BlockCopy(row, 0, weights, row.Length, row.Length);

        float[] input = Enumerable.Range(0, rows * inDim)
            .Select(i => 0.03f * ((i % 23) - 11))
            .ToArray();
        float[] actual = new float[rows * outDim];

        Assert.True(ManagedQuantizedOps.TryAddmmQuantizedToFloat32(
            (int)GgmlTensorType.Q4_K,
            weights,
            0,
            inDim,
            outDim,
            input,
            0,
            inDim,
            rows,
            actual,
            0,
            outDim));

        float[] expected = DequantizedMatmul(weights, GgmlTensorType.Q4_K, outDim, inDim, input, rows);
        AssertClose(expected, actual, 0.2f);
    }

    [Fact]
    public void Benchmark_Q80DirectMatmul_VsDequantizedBlockDot()
    {
        const int inDim = 1024;
        const int outDim = 256;
        const int rows = 4;
        const int warmup = 2;
        const int iterations = 8;

        float[] weightsF32 = Enumerable.Range(0, outDim * inDim)
            .Select(i => MathF.Sin(i * 0.013f) * 0.08f)
            .ToArray();
        byte[] weightsQ80 = QuantizeRowsQ80(weightsF32, outDim, inDim);
        float[] input = Enumerable.Range(0, rows * inDim)
            .Select(i => MathF.Cos(i * 0.017f) * 0.08f)
            .ToArray();

        float[] oldPath = new float[rows * outDim];
        float[] direct = new float[rows * outDim];
        float[] sums = new float[rows];

        void RunOld()
        {
            long rowBytes = NativeDequant.RowSize((int)GgmlTensorType.Q8_0, inDim);
            for (int col = 0; col < outDim; col++)
            {
                ManagedQuantizedOps.DotRowBatchToFloat32(
                    (int)GgmlTensorType.Q8_0,
                    weightsQ80,
                    (int)(col * rowBytes),
                    input,
                    0,
                    inDim,
                    rows,
                    inDim,
                    sums,
                    0);

                for (int row = 0; row < rows; row++)
                    oldPath[row * outDim + col] = sums[row];
            }
        }

        void RunDirect()
        {
            Assert.True(ManagedQuantizedOps.TryAddmmQuantizedToFloat32(
                (int)GgmlTensorType.Q8_0,
                weightsQ80,
                0,
                inDim,
                outDim,
                input,
                0,
                inDim,
                rows,
                direct,
                0,
                outDim));
        }

        for (int i = 0; i < warmup; i++)
        {
            RunOld();
            RunDirect();
        }

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            RunOld();
        double oldMs = sw.Elapsed.TotalMilliseconds;

        sw.Restart();
        for (int i = 0; i < iterations; i++)
            RunDirect();
        double directMs = sw.Elapsed.TotalMilliseconds;

        float maxDiff = MaxAbsDiff(oldPath, direct);
        Console.WriteLine(
            $"[ManagedQuantizedOps Q8_0] dequant-block: {oldMs / iterations:F3} ms/iter, " +
            $"direct-int8: {directMs / iterations:F3} ms/iter, " +
            $"speedup: {oldMs / directMs:F2}x, max diff: {maxDiff:E3}, " +
            $"AVX512F={Avx512F.IsSupported}, AVX512BW={Avx512BW.IsSupported}");

        Assert.True(maxDiff < 0.08f, $"Direct quantized path drifted too far from dequantized reference: {maxDiff}");
    }

    private static void WriteHalf(byte[] buffer, int offset, float value)
    {
        BinaryPrimitives.WriteUInt16LittleEndian(
            buffer.AsSpan(offset, 2),
            BitConverter.HalfToUInt16Bits((Half)value));
    }

    private static float Dot(float[] lhs, float[] rhs, int rhsOffset, int length)
    {
        float sum = 0.0f;
        for (int i = 0; i < length; i++)
        {
            sum += lhs[i] * rhs[rhsOffset + i];
        }

        return sum;
    }

    private static byte[] QuantizeRowsQ80(float[] values, int rows, int cols)
    {
        const int blockSize = 32;
        const int blockBytes = 34;
        Assert.Equal(0, cols % blockSize);

        int blocksPerRow = cols / blockSize;
        byte[] raw = new byte[rows * blocksPerRow * blockBytes];
        for (int row = 0; row < rows; row++)
        {
            for (int block = 0; block < blocksPerRow; block++)
            {
                int srcOffset = row * cols + block * blockSize;
                int dstOffset = row * blocksPerRow * blockBytes + block * blockBytes;
                float maxAbs = 0.0f;
                for (int i = 0; i < blockSize; i++)
                    maxAbs = MathF.Max(maxAbs, MathF.Abs(values[srcOffset + i]));

                float scale = maxAbs / 127.0f;
                WriteHalf(raw, dstOffset, scale);
                if (scale == 0.0f)
                    continue;

                float invScale = 1.0f / scale;
                for (int i = 0; i < blockSize; i++)
                {
                    int q = (int)MathF.Round(values[srcOffset + i] * invScale);
                    q = Math.Clamp(q, -127, 127);
                    raw[dstOffset + 2 + i] = unchecked((byte)(sbyte)q);
                }
            }
        }

        return raw;
    }

    private static float[] DequantizedMatmul(byte[] weights, GgmlTensorType type, int outDim, int inDim, float[] input, int rows)
    {
        long rowBytes = NativeDequant.RowSize((int)type, inDim);
        float[] expected = new float[rows * outDim];
        float[] weightRow = new float[inDim];
        for (int col = 0; col < outDim; col++)
        {
            NativeDequant.DequantizeToFloat32((int)type, weights, (int)(col * rowBytes), weightRow, 0, inDim);
            for (int row = 0; row < rows; row++)
                expected[row * outDim + col] = Dot(weightRow, input, row * inDim, inDim);
        }

        return expected;
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(
                MathF.Abs(expected[i] - actual[i]) <= tolerance,
                $"index {i}: expected {expected[i]}, observed {actual[i]}, tolerance {tolerance}");
        }
    }

    private static float MaxAbsDiff(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        float max = 0.0f;
        for (int i = 0; i < expected.Length; i++)
            max = MathF.Max(max, MathF.Abs(expected[i] - actual[i]));
        return max;
    }
}
