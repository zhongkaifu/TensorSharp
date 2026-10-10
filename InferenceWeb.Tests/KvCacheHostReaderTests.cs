// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The linear-to-paged K/V migration of Nemotron-H and gpt-oss reads the linear cache back
// as float32. It used to read only F32/F16, so a q8_0 / q4_0 cache (TensorAgent's default
// K/V setting) declined the migration, the planner refused the N=1 fused path and every
// request ran the batched route, which reuses nothing. KvCacheHostReader dequantizes the
// block-quantized caches; these tests pin it, byte layout included, against an
// independent scalar decoder of ggml's reference blocks. It reads only the rows a sequence
// holds, and a quantized cache migrates only where Nemotron-H's N=1 decode reads it in place.
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Cpu;

namespace InferenceWeb.Tests;

public class KvCacheHostReaderTests
{
    private const int Heads = 2, Capacity = 48, HeadDim = 128;
    private static readonly IAllocator Cpu = new CpuAllocator(BlasEnum.DotNet);

    [Theory]
    [InlineData(KvCacheDtype.Q8_0)]
    [InlineData(KvCacheDtype.Q4_0)]
    public void BlockQuantizedCache_ReadsAsItsExactDequantization(KvCacheDtype dtype)
    {
        int count = Heads * Capacity * HeadDim;
        float[] values = Values(count, seed: (int)dtype);
        using var cache = new Tensor(Cpu, dtype.ToDType(), Heads, Capacity, HeadDim);
        byte[] blocks = Quantize(dtype, values);
        Assert.Equal(cache.Storage.ByteLength, blocks.LongLength);
        WriteBytes(cache, blocks);

        Assert.True(KvCacheHostReader.TryReadHeadRowsAsFloat32(cache, Capacity, out float[] flat));

        float[] reference = ReferenceDequantize(dtype, blocks, count);
        Assert.Equal(count, flat.Length);
        for (int i = 0; i < count; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(reference[i]), BitConverter.SingleToInt32Bits(flat[i]));

        // And it is the cache's own row layout [heads, capacity, headDim]: each element
        // decodes to the value written there, within one quantization step of its block.
        float steps = dtype == KvCacheDtype.Q8_0 ? 127f : 8f;
        for (int h = 0; h < Heads; h++)
            for (int p = 0; p < Capacity; p += 7)
                for (int d = 0; d < HeadDim; d += 5)
                {
                    int at = (h * Capacity + p) * HeadDim + d;
                    float blockMax = 0;
                    for (int k = at / 32 * 32; k < at / 32 * 32 + 32; k++)
                        blockMax = Math.Max(blockMax, Math.Abs(values[k]));
                    float tolerance = blockMax / steps * 1.01f;
                    Assert.InRange(flat[at] - values[at], -tolerance, tolerance);
                }
    }

    /// <summary>A migration moves the rows a sequence holds, not the whole context the
    /// cache is sized for: each head's leading rows, packed [heads, rows, headDim]. On a
    /// 32k-row cache reading everything dequantized two 134 MB arrays per attention layer
    /// of Nemotron-H 8B to move a few hundred rows.</summary>
    [Theory]
    [InlineData(KvCacheDtype.Q8_0)]
    [InlineData(KvCacheDtype.Q4_0)]
    [InlineData(KvCacheDtype.F16)]
    [InlineData(KvCacheDtype.F32)]
    public void ReadsOnlyTheHeldRowsOfEachHead(KvCacheDtype dtype)
    {
        int count = Heads * Capacity * HeadDim;
        float[] values = Values(count, seed: 3 + (int)dtype);
        using var cache = new Tensor(Cpu, dtype.ToDType(), Heads, Capacity, HeadDim);
        float[] stored = Store(cache, dtype, values);

        const int rows = 5;
        Assert.True(KvCacheHostReader.TryReadHeadRowsAsFloat32(cache, rows, out float[] flat));
        Assert.Equal(Heads * rows * HeadDim, flat.Length);
        for (int h = 0; h < Heads; h++)
            for (int i = 0; i < rows * HeadDim; i++)
                Assert.Equal(BitConverter.SingleToInt32Bits(stored[h * Capacity * HeadDim + i]),
                    BitConverter.SingleToInt32Bits(flat[h * rows * HeadDim + i]));

        Assert.True(KvCacheHostReader.TryReadHeadRowsAsFloat32(cache, 0, out float[] none));
        Assert.Empty(none);
        // More rows than the cache holds: refused.
        Assert.False(KvCacheHostReader.TryReadHeadRowsAsFloat32(cache, Capacity + 1, out _));
    }

    [Fact]
    public void QuantizedRowsThatAreNotWholeBlocks_AreRefused()
    {
        // A 48-wide row is one and a half 32-element blocks: no head starts on a block.
        using var cache = new Tensor(Cpu, DType.Q8_0, Heads, Capacity, 48);
        Assert.False(KvCacheHostReader.TryReadHeadRowsAsFloat32(cache, 4, out _));
    }

    /// <summary>Unloaded models report no migration (PrefixCacheFamilyCoverageTests relies on
    /// it): the gate needs the caches.</summary>
    [Fact]
    public void UnloadedModels_StillReportNoLinearMigration()
    {
        var nemotron = (NemotronModel)RuntimeHelpers.GetUninitializedObject(typeof(NemotronModel));
        Assert.False(nemotron.SupportsLinearKVMigration);
        var gptOss = (GptOssModel)RuntimeHelpers.GetUninitializedObject(typeof(GptOssModel));
        Assert.False(gptOss.SupportsLinearKVMigration);
    }

    /// <summary>A loaded Nemotron-H migrates a block-quantized cache (dequantized on the way)
    /// where its N=1 decode is validated to read and append to that cache in place -
    /// ggml-metal - and keeps it on the batched route elsewhere: ggml-cpu aborts on the
    /// strided quantized append, ggml-cuda and ggml-vulkan have not run the path with this
    /// model, and the host walk the N=1 decode would fall back to dequantizes the whole
    /// window every token. A float cache
    /// migrates everywhere, as before. (gpt-oss's gate is dtype-blind: a real gpt-oss load
    /// turns a q8_0 / q4_0 request into f16.)</summary>
    [Theory]
    [InlineData(BackendType.GgmlMetal, KvCacheDtype.Q8_0, true)]
    [InlineData(BackendType.GgmlMetal, KvCacheDtype.Q4_0, true)]
    [InlineData(BackendType.GgmlCuda, KvCacheDtype.Q4_0, false)]
    [InlineData(BackendType.GgmlCuda, KvCacheDtype.F16, true)]
    [InlineData(BackendType.GgmlCpu, KvCacheDtype.Q8_0, false)]
    [InlineData(BackendType.GgmlVulkan, KvCacheDtype.Q8_0, false)]
    [InlineData(BackendType.GgmlVulkan, KvCacheDtype.Q4_0, false)]
    [InlineData(BackendType.GgmlCpu, KvCacheDtype.F16, true)]
    [InlineData(BackendType.GgmlVulkan, KvCacheDtype.F16, true)]
    [InlineData(BackendType.GgmlMetal, KvCacheDtype.F16, true)]
    public void LoadedNemotron_MigratesAQuantizedCacheWhereItsDecodeReadsItInPlace(
        BackendType backend, KvCacheDtype dtype, bool migrates)
    {
        var nemotron = (NemotronModel)RuntimeHelpers.GetUninitializedObject(typeof(NemotronModel));
        SetField(typeof(NemotronModel), nemotron, "_kvCacheK", new Tensor[1]);
        SetField(typeof(NemotronModel), nemotron, "_kvCacheV", new Tensor[1]);
        SetField(typeof(NemotronModel), nemotron, "_convState", new float[1][]);
        SetField(typeof(NemotronModel), nemotron, "_ssmState", new float[1][]);
        SetField(typeof(ModelBase), nemotron, "_kvCacheDtype", dtype);
        SetField(typeof(ModelBase), nemotron, "_backend", backend);
        Assert.Equal(migrates, nemotron.SupportsLinearKVMigration);

        var gptOss = (GptOssModel)RuntimeHelpers.GetUninitializedObject(typeof(GptOssModel));
        SetField(typeof(GptOssModel), gptOss, "_kvCacheK", new Tensor[1]);
        SetField(typeof(GptOssModel), gptOss, "_kvCacheV", new Tensor[1]);
        SetField(typeof(ModelBase), gptOss, "_kvCacheDtype", dtype);
        SetField(typeof(ModelBase), gptOss, "_backend", backend);
        Assert.True(gptOss.SupportsLinearKVMigration);
    }

    private static void SetField(Type owner, object target, string name, object value)
        => owner.GetField(name, System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
            .SetValue(target, value);

    // ------------------------------------------------------------------ helpers

    private static float[] Values(int count, int seed)
    {
        var rng = new Random(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = (float)(rng.NextDouble() * 2 - 1) * (1 + (i / 32) % 5);
        return values;
    }

    /// <summary>Write <paramref name="values"/> into the cache in its own dtype; returns the
    /// float value each element now holds (for a quantized cache, its exact dequantization).</summary>
    private static float[] Store(Tensor cache, KvCacheDtype dtype, float[] values)
    {
        switch (dtype)
        {
            case KvCacheDtype.F32:
                cache.SetElementsAsFloat(values);
                return values;
            case KvCacheDtype.F16:
            {
                var halves = new ushort[values.Length];
                var held = new float[values.Length];
                for (int i = 0; i < values.Length; i++)
                {
                    halves[i] = BitConverter.HalfToUInt16Bits((System.Half)values[i]);
                    held[i] = (float)(System.Half)values[i];
                }
                WriteBytes(cache, System.Runtime.InteropServices.MemoryMarshal.AsBytes(halves.AsSpan()).ToArray());
                return held;
            }
            default:
            {
                byte[] blocks = Quantize(dtype, values);
                WriteBytes(cache, blocks);
                return ReferenceDequantize(dtype, blocks, values.Length);
            }
        }
    }

    private static byte[] Quantize(KvCacheDtype dtype, float[] values)
    {
        var bytes = new byte[dtype.ByteLengthFor(values.Length)];
        ManagedQuantizedOps.QuantizeRowFromFloat32(dtype.GgmlType(), values, 0, bytes, 0, values.Length);
        return bytes;
    }

    private static unsafe void WriteBytes(Tensor tensor, byte[] bytes)
    {
        fixed (byte* src = bytes)
            tensor.Storage.CopyToStorage(0, (IntPtr)src, bytes.Length);
    }

    /// <summary>ggml's reference blocks, decoded independently: a little-endian f16 scale,
    /// then 32 int8 quants (Q8_0) or 16 bytes whose low nibbles are elements 0-15 and high
    /// nibbles 16-31, offset by 8 (Q4_0).</summary>
    private static float[] ReferenceDequantize(KvCacheDtype dtype, byte[] blocks, int count)
    {
        var result = new float[count];
        int blockBytes = dtype == KvCacheDtype.Q8_0 ? 34 : 18;
        for (int b = 0; b < count / 32; b++)
        {
            int at = b * blockBytes;
            float d = (float)BitConverter.UInt16BitsToHalf((ushort)(blocks[at] | (blocks[at + 1] << 8)));
            for (int j = 0; j < 32; j++)
            {
                int q = dtype == KvCacheDtype.Q8_0
                    ? (sbyte)blocks[at + 2 + j]
                    : (j < 16 ? blocks[at + 2 + j] & 0x0F : blocks[at + 2 + j - 16] >> 4) - 8;
                result[b * 32 + j] = q * d;
            }
        }
        return result;
    }
}
