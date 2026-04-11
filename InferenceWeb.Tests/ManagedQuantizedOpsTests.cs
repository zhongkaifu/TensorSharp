using System.Buffers.Binary;
using InferenceEngine;

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

    private static void WriteHalf(byte[] buffer, int offset, float value)
    {
        BinaryPrimitives.WriteUInt16LittleEndian(
            buffer.AsSpan(offset, 2),
            BitConverter.HalfToUInt16Bits((Half)value));
    }
}
