using System.Buffers.Binary;
using System.Runtime.InteropServices;

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
}
