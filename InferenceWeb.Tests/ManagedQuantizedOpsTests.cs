using System.Buffers.Binary;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace InferenceWeb.Tests;

public class ManagedQuantizedOpsTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(7)]
    [InlineData(32)]
    public unsafe void Q80IntegerDot_MatchesScalarSignedBlocksAndUnalignedBuffers(int blocks)
    {
        const int offset = 3;
        byte[] weights = new byte[offset + blocks * 34];
        byte[] activations = new byte[offset + blocks * 34];
        float expected = 0;
        float[] scales = { 0.03125f, 0.5f, 2f, 0.125f };
        for (int b = 0; b < blocks; ++b)
        {
            int start = offset + b * 34;
            float dw = scales[b % scales.Length];
            float dx = scales[(b + 1) % scales.Length];
            BinaryPrimitives.WriteUInt16LittleEndian(weights.AsSpan(start), BitConverter.HalfToUInt16Bits((Half)dw));
            BinaryPrimitives.WriteUInt16LittleEndian(activations.AsSpan(start), BitConverter.HalfToUInt16Bits((Half)dx));
            int dot = 0;
            for (int i = 0; i < 32; ++i)
            {
                sbyte w = unchecked((sbyte)((b * 43 + i * 71) % 256 - 128));
                sbyte x = (sbyte)((b * 31 + i * 47) % 255 - 127);
                weights[start + 2 + i] = unchecked((byte)w);
                activations[start + 2 + i] = unchecked((byte)x);
                dot += w * x;
            }
            expected += dw * dx * dot;
        }
        fixed (byte* w = weights)
        fixed (byte* x = activations)
        {
            float actual = ManagedQuantizedOps.DotQuantizedRow(GgmlTensorType.Q8_0, w + offset, x + offset, blocks * 32);
            Assert.InRange(MathF.Abs(actual - expected), 0, 1e-5f * MathF.Max(1, MathF.Abs(expected)));
        }
    }

    [Fact]
    public void ShouldStoreWeightQuantized_UsesManagedCpuSupportMatrix()
    {
        GgufTensorInfo Weight(GgmlTensorType type) => new GgufTensorInfo
        {
            Name = "blk.0.attn_q.weight",
            Shape = new ulong[] { 256, 128 },
            Type = type,
        };

        // Managed CPU quantized storage now covers the K-quants AND the low-bit i-quants
        // (Q2_K/Q3_K, IQ2_XXS/IQ2_S/IQ3_S) — these must be stored quantized (and matmul'd
        // via the managed path), not dequantized to F32 at load.
        foreach (var t in new[]
                 {
                     GgmlTensorType.Q4_K, GgmlTensorType.Q2_K, GgmlTensorType.Q3_K,
                     GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ2_S, GgmlTensorType.IQ3_S,
                 })
            Assert.True(ModelBase.ShouldStoreWeightQuantized(BackendType.Cpu, Weight(t)), $"{t} should be CPU-quantized");

        // IQ1_S / IQ1_M joined the matrix so Qwen3.8-Flash-Next's UD-IQ1_S and
        // GLM-5.3-Flash's UD-IQ1_M can run on the pure-C# backend at all: without
        // managed storage their expert tensors were dequantized to F32 at load,
        // which left _stackedExpertWeights empty and made Qwen4Exp abort with
        // "GGUF is missing 34 expected tensor(s): blk.0.ffn_gate_exps.weight".
        foreach (var t in new[] { GgmlTensorType.IQ1_S, GgmlTensorType.IQ1_M })
            Assert.True(ModelBase.ShouldStoreWeightQuantized(BackendType.Cpu, Weight(t)), $"{t} should be CPU-quantized");

        // IQ2_XS / IQ4_XS joined for GLM-5.3-Flash's UD-Q2_K_XL, which is 82 IQ2_XS
        // tensors plus 3 IQ4_XS ones. Without managed storage the loader expanded
        // them to F32 -- 765 GB for that model -- and --backend cpu never finished
        // loading rather than failing with a message.
        foreach (var t in new[] { GgmlTensorType.IQ2_XS, GgmlTensorType.IQ4_XS })
            Assert.True(ModelBase.ShouldStoreWeightQuantized(BackendType.Cpu, Weight(t)), $"{t} should be CPU-quantized");

        // A type the managed backend still does not implement stays out of the matrix.
        Assert.False(ModelBase.ShouldStoreWeightQuantized(BackendType.Cpu, Weight(GgmlTensorType.TQ1_0)));
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
    public void ManagedIq3Xxs_MatchesNativeGgmlDequantizer()
    {
        // IQ3_XXS is what Unsloth's "UD-IQ2_XXS" mixed quants put on the
        // sensitive tensors. MLX has no native IQ3_XXS kernel, so those tensors
        // fall through to the managed dequantizer; before it existed,
        // Muse-Glimmer-30B / Qwen3.6-27B / Qwen3.6-35B-A3B all aborted on their
        // first FFN with "Pure C# backend does not support GGUF tensor type
        // IQ3_XXS". Ground truth is ggml's own dequantize_row_iq3_xxs.
        const int blocks = 7;
        const int elems = blocks * 256;
        int blockBytes = (int)GgufFile.GetTypeSize(GgmlTensorType.IQ3_XXS);
        Assert.Equal(98, blockBytes);
        Assert.Equal(256, GgufFile.GetBlockSize(GgmlTensorType.IQ3_XXS));
        Assert.Equal(blocks * blockBytes, NativeDequant.RowSize((int)GgmlTensorType.IQ3_XXS, elems));

        // Deterministic pseudo-random block payloads: every grid index, every
        // 7-bit sign selector and every 4-bit scale nibble gets exercised.
        byte[] raw = new byte[blocks * blockBytes];
        uint state = 0x9E3779B9u;
        for (int i = 0; i < raw.Length; i++)
        {
            state = state * 1664525u + 1013904223u;
            raw[i] = (byte)(state >> 24);
        }
        // Keep the fp16 scale of every block finite and sane.
        for (int b = 0; b < blocks; b++)
            WriteHalf(raw, b * blockBytes, 0.25f + 0.125f * b);

        var managed = new float[elems];
        ManagedQuantizedOps.DequantizeToFloat32((int)GgmlTensorType.IQ3_XXS, raw, 0, managed, 0, elems);

        var native = new float[elems];
        TensorSharp.GGML.GgmlGgufTensorDequant.DequantizeToFloat32(
            (int)GgmlTensorType.IQ3_XXS, raw, 0, native, 0, elems);

        Assert.Equal(native.Length, managed.Length);
        bool anyNonZero = false;
        for (int i = 0; i < elems; i++)
        {
            if (native[i] != 0f) anyNonZero = true;
            Assert.True(Math.Abs(native[i] - managed[i]) <= 1e-5f * Math.Max(1f, Math.Abs(native[i])),
                $"element {i}: native {native[i]}, managed {managed[i]}");
        }
        Assert.True(anyNonZero, "dequantized block was entirely zero - the fixture is not exercising the codebook");
    }

    [Fact]
    public void ManagedNvfp4_MatchesNativeGgmlDequantizer()
    {
        // NVFP4 (ggml type 40): 36-byte block = 4 UE4M3 sub-block scales + 32
        // packed E2M1 nibble bytes for 64 elements. Ground truth is ggml's own
        // dequantize_row_nvfp4 (doubled codebook, halved UE4M3 decode).
        const int blocks = 9;
        const int elems = blocks * 64;
        int blockBytes = (int)GgufFile.GetTypeSize(GgmlTensorType.NVFP4);
        Assert.Equal(36, blockBytes);
        Assert.Equal(64, GgufFile.GetBlockSize(GgmlTensorType.NVFP4));
        Assert.Equal(blocks * blockBytes, NativeDequant.RowSize((int)GgmlTensorType.NVFP4, elems));

        byte[] raw = new byte[blocks * blockBytes];
        uint state = 0x9E3779B9u;
        for (int i = 0; i < raw.Length; i++)
        {
            state = state * 1664525u + 1013904223u;
            raw[i] = (byte)(state >> 24);
        }
        // Exercise the special scale encodings (0x00 and the 0x7F NaN pattern
        // both decode to 0) on the first block's sub-scales.
        raw[0] = 0x00;
        raw[1] = 0x7F;

        var managed = new float[elems];
        ManagedQuantizedOps.DequantizeToFloat32((int)GgmlTensorType.NVFP4, raw, 0, managed, 0, elems);

        var native = new float[elems];
        TensorSharp.GGML.GgmlGgufTensorDequant.DequantizeToFloat32(
            (int)GgmlTensorType.NVFP4, raw, 0, native, 0, elems);

        bool anyNonZero = false;
        for (int i = 0; i < elems; i++)
        {
            if (native[i] != 0f) anyNonZero = true;
            Assert.True(Math.Abs(native[i] - managed[i]) <= 1e-6f * Math.Max(1f, Math.Abs(native[i])),
                $"element {i}: native {native[i]}, managed {managed[i]}");
        }
        Assert.True(anyNonZero, "dequantized block was entirely zero - the fixture is not exercising the codebook");
    }

    [Theory]
    [InlineData((int)GgmlTensorType.IQ1_S, 50)]
    [InlineData((int)GgmlTensorType.IQ1_M, 56)]
    public void ManagedIq1_MatchesNativeGgmlDequantizer(int typeId, int expectedBlockBytes)
    {
        // The 1-bit i-quants are what Unsloth ships for the biggest models
        // (Qwen3.8-Flash-Next UD-IQ1_S, GLM-5.3-Flash UD-IQ1_M). Ground truth is
        // ggml's own dequantize_row_iq1_s / _iq1_m, and the subtle part is the
        // +/-IQ1S_DELTA offset applied to every grid lane - drop it and the
        // weights are quietly biased rather than obviously wrong.
        var type = (GgmlTensorType)typeId;
        const int blocks = 7;
        const int elems = blocks * 256;
        int blockBytes = (int)GgufFile.GetTypeSize(type);
        Assert.Equal(expectedBlockBytes, blockBytes);
        Assert.Equal(256, GgufFile.GetBlockSize(type));
        Assert.Equal(blocks * blockBytes, NativeDequant.RowSize(typeId, elems));

        byte[] raw = new byte[blocks * blockBytes];
        uint state = 0x9E3779B9u;
        for (int i = 0; i < raw.Length; i++)
        {
            state = state * 1664525u + 1013904223u;
            raw[i] = (byte)(state >> 24);
        }
        // IQ1_S carries a per-block fp16 scale; IQ1_M reassembles its scale from
        // nibbles spread across the four scale uint16s, so leave those random.
        if (type == GgmlTensorType.IQ1_S)
        {
            for (int b = 0; b < blocks; b++)
                WriteHalf(raw, b * blockBytes, 0.25f + 0.125f * b);
        }

        var managed = new float[elems];
        ManagedQuantizedOps.DequantizeToFloat32(typeId, raw, 0, managed, 0, elems);

        var native = new float[elems];
        TensorSharp.GGML.GgmlGgufTensorDequant.DequantizeToFloat32(typeId, raw, 0, native, 0, elems);

        bool anyNonZero = false;
        for (int i = 0; i < elems; i++)
        {
            if (native[i] != 0f) anyNonZero = true;
            Assert.True(Math.Abs(native[i] - managed[i]) <= 1e-5f * Math.Max(1f, Math.Abs(native[i])),
                $"{type} element {i}: native {native[i]}, managed {managed[i]}");
        }
        Assert.True(anyNonZero, $"{type}: dequantized block was entirely zero - the fixture is not exercising the codebook");
    }

    [Theory]
    [InlineData((int)GgmlTensorType.IQ2_XS, 74)]
    [InlineData((int)GgmlTensorType.IQ4_XS, 136)]
    public void ManagedIqXs_MatchesNativeGgmlDequantizer(int typeId, int expectedBlockBytes)
    {
        // GLM-5.3-Flash UD-Q2_K_XL is 82 IQ2_XS tensors plus 3 IQ4_XS ones, and
        // neither had a managed path: on --backend cpu the loader fell through to
        // expanding them to F32, which for that model is 765 GB of weights and a
        // load that never finishes. Ground truth is ggml's own
        // dequantize_row_iq2_xs / _iq4_xs.
        //
        // The two traps: IQ2_XS indexes a 512-entry grid with 9 bits and takes its
        // sign byte from the TOP 7 (IQ2_XXS uses a byte index and a different
        // grid), and IQ4_XS splits each 6-bit scale across scales_l and scales_h
        // and biases it by 32.
        var type = (GgmlTensorType)typeId;
        const int blocks = 7;
        const int elems = blocks * 256;
        int blockBytes = (int)GgufFile.GetTypeSize(type);
        Assert.Equal(expectedBlockBytes, blockBytes);
        Assert.Equal(256, GgufFile.GetBlockSize(type));
        Assert.Equal(blocks * blockBytes, NativeDequant.RowSize(typeId, elems));

        byte[] raw = new byte[blocks * blockBytes];
        uint state = 0x9E3779B9u;
        for (int i = 0; i < raw.Length; i++)
        {
            state = state * 1664525u + 1013904223u;
            raw[i] = (byte)(state >> 24);
        }
        // Both carry the fp16 block scale at offset 0; keep it finite so the
        // comparison is about the codebook rather than about inf/NaN.
        for (int b = 0; b < blocks; b++)
            WriteHalf(raw, b * blockBytes, 0.25f + 0.125f * b);

        var managed = new float[elems];
        ManagedQuantizedOps.DequantizeToFloat32(typeId, raw, 0, managed, 0, elems);

        var native = new float[elems];
        TensorSharp.GGML.GgmlGgufTensorDequant.DequantizeToFloat32(typeId, raw, 0, native, 0, elems);

        bool anyNonZero = false;
        for (int i = 0; i < elems; i++)
        {
            if (native[i] != 0f) anyNonZero = true;
            Assert.True(Math.Abs(native[i] - managed[i]) <= 1e-5f * Math.Max(1f, Math.Abs(native[i])),
                $"{type} element {i}: native {native[i]}, managed {managed[i]}");
        }
        Assert.True(anyNonZero, $"{type}: dequantized block was entirely zero - the fixture is not exercising the codebook");
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

    [Theory]
    [InlineData((int)GgmlTensorType.IQ2_XS, 74)]
    [InlineData((int)GgmlTensorType.IQ3_XXS, 98)]
    public void DotRowBatch_DirectIQuantKernelMatchesDequantizedDot(int typeId, int blockBytes)
    {
        // These two types ARE the MoE of GLM-5.3-Flash UD-Q2_K_XL. Before they had
        // direct kernels every expert matmul dequantized the weight row to F32
        // first -- a ~12x expansion on a 2-bit weight, and 91% of decode time.
        // A wrong grid index, sign selector or scale nibble does not produce a
        // small error here, it produces a different number entirely, so comparing
        // against the dequantized dot is a sharp check even at a loose tolerance
        // (the direct path quantizes the ACTIVATION to Q8_K, so it is not exact).
        var type = (GgmlTensorType)typeId;
        const int n = 256 * 2;
        const int rows = 3;
        Assert.Equal(blockBytes, (int)GgufFile.GetTypeSize(type));

        byte[] raw = new byte[(n / 256) * blockBytes];
        uint state = 0x12345678u;
        for (int i = 0; i < raw.Length; i++)
        {
            state = state * 1664525u + 1013904223u;
            raw[i] = (byte)(state >> 24);
        }
        for (int b = 0; b < n / 256; b++)
            WriteHalf(raw, b * blockBytes, 0.5f + 0.25f * b);

        float[] inputs = new float[n * rows];
        for (int r = 0; r < rows; r++)
            for (int i = 0; i < n; i++)
                inputs[r * n + i] = (r + 1) * 0.05f * ((i % 23) - 11);

        float[] actual = new float[rows];
        ManagedQuantizedOps.DotRowBatchToFloat32(typeId, raw, 0, inputs, 0, n, rows, n, actual, 0);

        float[] dequantized = new float[n];
        NativeDequant.DequantizeToFloat32(typeId, raw, 0, dequantized, 0, n);

        // RELATIVE tolerance against a dot that must itself be large enough to be
        // meaningful. An earlier version of this test allowed 0.02 ABSOLUTE on
        // small dots, which let a missing 0.125 folded scale - an 8x error that
        // produced pure garbage in a real model - pass as green.
        for (int r = 0; r < rows; r++)
        {
            double expected = Dot(dequantized, inputs, r * n, n);
            Assert.True(Math.Abs(expected) > 1.0,
                $"{type} row {r}: reference dot {expected} is too small to discriminate; fix the fixture");
            double relative = Math.Abs(expected - actual[r]) / Math.Abs(expected);
            Assert.True(relative <= 0.02,
                $"{type} row {r}: dequantized dot {expected}, direct kernel {actual[r]} (relative {relative:P2})");
        }
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

    /// <summary>
    /// The batched entry point exists purely to collapse a Mixture-of-Experts
    /// layer's many tiny matmuls into one parallel dispatch (DeepSeek V4's
    /// --n-cpu-moe layers issue one per selected expert per projection). It has
    /// to agree with the one-at-a-time path element for element — including the
    /// case two jobs share an input block, where the batch quantizes those
    /// activations once and both jobs read the same quantized rows.
    /// </summary>
    [Fact]
    public void TryAddmmQuantizedBatch_MatchesPerJobResults()
    {
        const int inDim = 64;
        const int outDim = 5;
        const int rows = 3;
        const int jobCount = 4;

        var weights = new byte[jobCount][];
        for (int j = 0; j < jobCount; j++)
        {
            float[] w = Enumerable.Range(0, outDim * inDim)
                .Select(i => MathF.Sin((i + j * 31) * 0.07f) * 0.35f)
                .ToArray();
            weights[j] = QuantizeRowsQ80(w, outDim, inDim);
        }

        // Two distinct input blocks; jobs 0/1 share the first (the gate/up
        // pattern), jobs 2/3 the second.
        float[] inputA = Enumerable.Range(0, rows * inDim).Select(i => MathF.Cos(i * 0.11f) * 0.2f).ToArray();
        float[] inputB = Enumerable.Range(0, rows * inDim).Select(i => MathF.Sin(i * 0.13f) * 0.3f).ToArray();

        var perJob = new float[jobCount][];
        var batched = new float[jobCount][];
        for (int j = 0; j < jobCount; j++)
        {
            perJob[j] = new float[rows * outDim];
            batched[j] = new float[rows * outDim];
        }

        unsafe
        {
            fixed (float* pa = inputA)
            fixed (float* pb = inputB)
            fixed (byte* w0 = weights[0])
            fixed (byte* w1 = weights[1])
            fixed (byte* w2 = weights[2])
            fixed (byte* w3 = weights[3])
            fixed (float* r0 = perJob[0])
            fixed (float* r1 = perJob[1])
            fixed (float* r2 = perJob[2])
            fixed (float* r3 = perJob[3])
            fixed (float* b0 = batched[0])
            fixed (float* b1 = batched[1])
            fixed (float* b2 = batched[2])
            fixed (float* b3 = batched[3])
            {
                byte*[] w = { w0, w1, w2, w3 };
                float*[] inp = { pa, pa, pb, pb };
                float*[] outPer = { r0, r1, r2, r3 };
                float*[] outBatch = { b0, b1, b2, b3 };

                for (int j = 0; j < jobCount; j++)
                {
                    Assert.True(ManagedQuantizedOps.TryAddmmQuantizedToFloat32(
                        (int)GgmlTensorType.Q8_0, (IntPtr)w[j], inDim, outDim,
                        inp[j], inDim, rows, outPer[j], outDim));
                }

                var jobs = new ManagedQuantizedOps.QuantMatMulJob[jobCount];
                for (int j = 0; j < jobCount; j++)
                    jobs[j] = new ManagedQuantizedOps.QuantMatMulJob(
                        (IntPtr)w[j], (IntPtr)inp[j], (IntPtr)outBatch[j], outDim, rows, outDim);

                Assert.True(ManagedQuantizedOps.TryAddmmQuantizedBatch(
                    (int)GgmlTensorType.Q8_0, inDim, inDim, jobs));
            }
        }

        for (int j = 0; j < jobCount; j++)
            AssertClose(perJob[j], batched[j], 1e-6f);
    }

    [Fact]
    public void TryAddmmQuantizedBatch_RejectsTypeWithoutDirectKernel()
    {
        // No direct host kernel => false, so the caller reports an unsupported
        // offload instead of shipping wrong numbers.
        Assert.False(ManagedQuantizedOps.TryAddmmQuantizedBatch(
            (int)GgmlTensorType.IQ1_S, 256, 256,
            new[] { new ManagedQuantizedOps.QuantMatMulJob(IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, 1, 1, 1) }));
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

    [Theory]
    [InlineData(256, 5, 1)]
    [InlineData(256, 5, 3)]
    [InlineData(1024, 33, 4)]
    public void TryAddmmQuantizedToFloat32_Q40Path_MatchesDequantReference(int inDim, int outDim, int rows)
    {
        // Validates the SIMD (AVX2/AVX-512) Q4_0 x Q8_0 dot against the
        // dequantize-then-dot reference over the SAME weight bytes.
        var rng = new Random(1234);
        byte[] weights = BuildRandomQ40(rng, outDim, inDim);
        float[] input = Enumerable.Range(0, rows * inDim)
            .Select(i => 0.05f * MathF.Sin(i * 0.021f))
            .ToArray();
        float[] actual = new float[rows * outDim];

        Assert.True(ManagedQuantizedOps.TryAddmmQuantizedToFloat32(
            (int)GgmlTensorType.Q4_0, weights, 0, inDim, outDim,
            input, 0, inDim, rows, actual, 0, outDim));

        float[] expected = DequantizedMatmul(weights, GgmlTensorType.Q4_0, outDim, inDim, input, rows);

        float maxRef = 0f;
        for (int i = 0; i < expected.Length; i++) maxRef = MathF.Max(maxRef, MathF.Abs(expected[i]));
        float tol = 0.02f * maxRef + 1e-3f;   // Q8_0 activation quant noise
        AssertClose(expected, actual, tol);
    }

    [Theory]
    [InlineData(GgmlTensorType.Q4_K, 256, 1, 1)]
    [InlineData(GgmlTensorType.Q4_K, 256, 7, 1)]
    [InlineData(GgmlTensorType.Q4_K, 512, 5, 3)]
    [InlineData(GgmlTensorType.Q4_K, 1024, 33, 4)]
    [InlineData(GgmlTensorType.Q5_K, 256, 1, 1)]
    [InlineData(GgmlTensorType.Q5_K, 512, 5, 3)]
    [InlineData(GgmlTensorType.Q5_K, 1024, 33, 4)]
    [InlineData(GgmlTensorType.Q6_K, 256, 1, 1)]
    [InlineData(GgmlTensorType.Q6_K, 512, 5, 3)]
    [InlineData(GgmlTensorType.Q6_K, 1024, 33, 4)]
    public void TryAddmmQuantizedToFloat32_KQuantSimd_MatchesDequantReference(
        GgmlTensorType type, int inDim, int outDim, int rows)
    {
        // Validates the SIMD K-quant dots (VecDotQ{4,5,6}_KQ8_K) against the
        // dequantize-then-dot reference over the SAME random weight bytes.
        var rng = new Random(20260629 + (int)type * 7 + inDim);
        byte[] weights = BuildRandomKQuant(rng, type, outDim, inDim);
        float[] input = Enumerable.Range(0, rows * inDim)
            .Select(i => 0.07f * MathF.Sin(i * 0.013f + (int)type))
            .ToArray();
        float[] actual = new float[rows * outDim];

        Assert.True(ManagedQuantizedOps.TryAddmmQuantizedToFloat32(
            (int)type, weights, 0, inDim, outDim,
            input, 0, inDim, rows, actual, 0, outDim));

        float[] expected = DequantizedMatmul(weights, type, outDim, inDim, input, rows);

        float maxRef = 0f;
        for (int i = 0; i < expected.Length; i++) maxRef = MathF.Max(maxRef, MathF.Abs(expected[i]));
        float tol = 0.02f * maxRef + 1e-3f;   // Q8_K activation quant noise
        AssertClose(expected, actual, tol);
    }

    [Theory]
    [InlineData(GgmlTensorType.IQ3_S, 256, 5, 1)]
    [InlineData(GgmlTensorType.IQ3_S, 512, 5, 3)]
    [InlineData(GgmlTensorType.IQ3_S, 1024, 33, 4)]
    [InlineData(GgmlTensorType.MXFP4, 256, 5, 1)]
    [InlineData(GgmlTensorType.MXFP4, 512, 5, 3)]
    [InlineData(GgmlTensorType.MXFP4, 1024, 33, 4)]
    [InlineData(GgmlTensorType.NVFP4, 256, 5, 1)]
    [InlineData(GgmlTensorType.NVFP4, 512, 5, 3)]
    [InlineData(GgmlTensorType.NVFP4, 1024, 33, 4)]
    public void TryAddmmQuantizedToFloat32_MoeQuantDots_MatchDequantReference(
        GgmlTensorType type, int inDim, int outDim, int rows)
    {
        // Validates the IQ3_S x Q8_K and MXFP4 x Q8_0 integer dots (used by the
        // DeepSeek4 CPU executor's MoE experts) against the dequantize-then-dot
        // reference over the SAME random weight bytes.
        var rng = new Random(20260729 + (int)type * 11 + inDim);
        byte[] weights = BuildRandomMoeQuant(rng, type, outDim, inDim);
        float[] input = Enumerable.Range(0, rows * inDim)
            .Select(i => 0.06f * MathF.Sin(i * 0.017f + (int)type))
            .ToArray();
        float[] actual = new float[rows * outDim];

        Assert.True(ManagedQuantizedOps.TryAddmmQuantizedToFloat32(
            (int)type, weights, 0, inDim, outDim,
            input, 0, inDim, rows, actual, 0, outDim));

        float[] expected = DequantizedMatmul(weights, type, outDim, inDim, input, rows);

        float maxRef = 0f;
        for (int i = 0; i < expected.Length; i++) maxRef = MathF.Max(maxRef, MathF.Abs(expected[i]));
        float tol = 0.02f * maxRef + 1e-3f;   // activation quant noise
        AssertClose(expected, actual, tol);
    }

    private static byte[] BuildRandomMoeQuant(Random rng, GgmlTensorType type, int outDim, int inDim)
    {
        int blockBytes = (int)GgufFile.GetTypeSize(type);
        int blockSize = (int)GgufFile.GetBlockSize(type);
        Assert.Equal(0, inDim % blockSize);
        int blocksPerRow = inDim / blockSize;
        byte[] raw = new byte[(long)outDim * blocksPerRow * blockBytes];
        int o = 0;
        for (int r = 0; r < outDim; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                switch (type)
                {
                    case GgmlTensorType.IQ3_S:
                        // half d + qs/qh/signs/scales, any bit pattern decodes
                        WriteHalf(raw, o, 0.01f + 0.02f * (float)rng.NextDouble());
                        for (int i = 2; i < blockBytes; i++) raw[o + i] = (byte)rng.Next(0, 256);
                        break;
                    case GgmlTensorType.MXFP4:
                        // E8M0 exponent constrained so scales stay finite/sane
                        raw[o] = (byte)(118 + rng.Next(0, 10));
                        for (int i = 1; i < blockBytes; i++) raw[o + i] = (byte)rng.Next(0, 256);
                        break;
                    case GgmlTensorType.NVFP4:
                        // 4 UE4M3 sub-block scales constrained sane (around 0.5..8),
                        // then 32 random packed E2M1 nibble bytes.
                        for (int i = 0; i < 4; i++) raw[o + i] = (byte)(0x30 + rng.Next(0, 22));
                        for (int i = 4; i < blockBytes; i++) raw[o + i] = (byte)rng.Next(0, 256);
                        break;
                    default:
                        throw new NotSupportedException(type.ToString());
                }
                o += blockBytes;
            }
        }
        return raw;
    }

    private static byte[] BuildRandomKQuant(Random rng, GgmlTensorType type, int outDim, int inDim)
    {
        Assert.Equal(0, inDim % 256);
        int blockBytes = (int)GgufFile.GetTypeSize(type);
        int sbPerRow = inDim / 256;
        byte[] raw = new byte[(long)outDim * sbPerRow * blockBytes];
        int o = 0;
        for (int r = 0; r < outDim; r++)
        {
            for (int b = 0; b < sbPerRow; b++)
            {
                int sb = o;
                switch (type)
                {
                    case GgmlTensorType.Q4_K:
                    case GgmlTensorType.Q5_K:
                        WriteHalf(raw, sb, 0.02f + 0.03f * (float)rng.NextDouble());      // d
                        WriteHalf(raw, sb + 2, 0.01f + 0.02f * (float)rng.NextDouble());  // dmin
                        for (int i = 0; i < blockBytes - 4; i++) raw[sb + 4 + i] = (byte)rng.Next(0, 256);
                        break;
                    case GgmlTensorType.Q6_K:
                        for (int i = 0; i < blockBytes - 2; i++) raw[sb + i] = (byte)rng.Next(0, 256);
                        WriteHalf(raw, sb + blockBytes - 2, 0.01f + 0.02f * (float)rng.NextDouble()); // d
                        break;
                }
                o += blockBytes;
            }
        }
        return raw;
    }

    private static byte[] BuildRandomQ40(Random rng, int outDim, int inDim)
    {
        const int blockSize = 32;
        const int blockBytes = 18; // 2 (f16 scale) + 16 packed nibbles
        Assert.Equal(0, inDim % blockSize);
        int blocksPerRow = inDim / blockSize;
        byte[] raw = new byte[(long)outDim * blocksPerRow * blockBytes];
        int o = 0;
        for (int r = 0; r < outDim; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                WriteHalf(raw, o, 0.03f + 0.02f * (float)rng.NextDouble());
                o += 2;
                for (int i = 0; i < 16; i++) raw[o++] = (byte)rng.Next(0, 256);
            }
        }
        return raw;
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

    [Theory]
    [InlineData(GgmlTensorType.Q4_0, 32)]
    [InlineData(GgmlTensorType.Q4_0, 256)]
    [InlineData(GgmlTensorType.Q4_0, 512)]
    [InlineData(GgmlTensorType.Q8_0, 32)]
    [InlineData(GgmlTensorType.Q8_0, 256)]
    public void QuantizeRowFromFloat32_ProducesGgmlCompatibleBytes(GgmlTensorType type, int n)
    {
        // The managed KV-cache write path (CopyToCache[Circular]/ExpandKVHeads for
        // block-quant) quantizes fresh K/V into the cache with QuantizeRowFromFloat32
        // and the subsequent fused decode reads it back with ggml's native dequant.
        // Validate that round trip: quantize here, dequantize with the native ggml
        // kernel, and assert the values come back within the tier's quant step.
        var rng = new Random(4242);
        float[] src = Enumerable.Range(0, n)
            .Select(i => (float)(rng.NextDouble() * 2.0 - 1.0) * (0.2f + 0.3f * (i % 7)))
            .ToArray();

        int rowBytes = (int)NativeDequant.RowSize((int)type, n);
        byte[] quant = new byte[rowBytes];
        ManagedQuantizedOps.QuantizeRowFromFloat32((int)type, src, 0, quant, 0, n);

        float[] back = new float[n];
        NativeDequant.DequantizeToFloat32((int)type, quant, 0, back, 0, n);

        float maxAbs = 0f;
        foreach (var v in src) maxAbs = MathF.Max(maxAbs, MathF.Abs(v));
        // Q4_0: 4-bit over [-8d,7d], step = maxAbs/8. Q8_0: 8-bit, step = maxAbs/127.
        float tol = (type == GgmlTensorType.Q4_0 ? maxAbs / 8f : maxAbs / 100f) + 1e-4f;
        for (int i = 0; i < n; i++)
            Assert.True(MathF.Abs(src[i] - back[i]) <= tol,
                $"index {i}: src {src[i]}, dequant {back[i]}, tol {tol} ({type})");
    }

    [Fact]
    public void QuantizeRowFromFloat32_AllZeros_DequantizesToZero()
    {
        float[] src = new float[256];
        byte[] quant = new byte[(int)NativeDequant.RowSize((int)GgmlTensorType.Q4_0, 256)];
        ManagedQuantizedOps.QuantizeRowFromFloat32((int)GgmlTensorType.Q4_0, src, 0, quant, 0, 256);

        float[] back = new float[256];
        NativeDequant.DequantizeToFloat32((int)GgmlTensorType.Q4_0, quant, 0, back, 0, 256);
        Assert.All(back, v => Assert.Equal(0f, v));
    }

    private static void WriteHalf(byte[] buffer, int offset, float value)
    {
        BinaryPrimitives.WriteUInt16LittleEndian(
            buffer.AsSpan(offset, 2),
            BitConverter.HalfToUInt16Bits((Half)value));
    }

    /// <summary>
    /// Q2_K now has an integer dot kernel, so a Q2_K matmul quantizes its
    /// activations to Q8_K and multiplies in integers instead of dequantizing the
    /// weight row to float first. The result must still agree with the float
    /// answer to within the activation quantization, which is one part in 127 of
    /// the row's largest magnitude.
    /// </summary>
    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    public void Q2KMatmul_IntegerPathMatchesDequantizedFloat(int elementCount)
    {
        int superBlocks = elementCount / 256;
        byte[] raw = new byte[superBlocks * 84];
        var rng = new Random(20260911 + elementCount);
        for (int b = 0; b < superBlocks; b++)
        {
            int o = b * 84;
            for (int i = 0; i < 16; i++)                    // scales: low nibble scale, high nibble min
                raw[o + i] = (byte)rng.Next(256);
            for (int i = 0; i < 64; i++)                    // qs: four 2-bit values per byte
                raw[o + 16 + i] = (byte)rng.Next(256);
            BinaryPrimitives.WriteUInt16LittleEndian(raw.AsSpan(o + 80), HalfBits(0.0125f));
            BinaryPrimitives.WriteUInt16LittleEndian(raw.AsSpan(o + 82), HalfBits(0.0037f));
        }

        float[] input = new float[elementCount];
        for (int i = 0; i < elementCount; i++)
            input[i] = (float)((i % 23) - 11) * 0.031f;

        float[] actual = new float[1];
        ManagedQuantizedOps.DotRowBatchToFloat32(
            (int)GgmlTensorType.Q2_K, raw, 0, input, 0, elementCount, 1, elementCount, actual, 0);

        float[] dequantized = new float[elementCount];
        NativeDequant.DequantizeToFloat32((int)GgmlTensorType.Q2_K, raw, 0, dequantized, 0, elementCount);
        float expected = Dot(dequantized, input, 0, elementCount);

        // Q8_K keeps 7 bits of the activation row's peak, so bound the error by
        // that step times the summed weight magnitude rather than by an absolute.
        float peak = 0.0f, weightMass = 0.0f;
        for (int i = 0; i < elementCount; i++)
        {
            peak = Math.Max(peak, Math.Abs(input[i]));
            weightMass += Math.Abs(dequantized[i]);
        }

        float tolerance = peak / 127.0f * weightMass + 1e-4f;
        Assert.True(Math.Abs(expected - actual[0]) <= tolerance,
            $"Q2_K integer dot {actual[0]} against dequantized {expected}, tolerance {tolerance}");
    }

    /// <summary>
    /// The tolerance test above would still pass if a vector lane were misplaced.
    /// This one removes the tolerance: every activation is an exact multiple of
    /// peak/127, so quantizing the row to Q8_K is lossless and the integer path
    /// must reproduce the float answer to within float rounding alone. That makes
    /// it sensitive to any lane, shift or scale-pairing mistake in the AVX2
    /// kernel, which the scalar reference cannot be.
    /// </summary>
    [Theory]
    [InlineData(256, 7)]
    [InlineData(512, 11)]
    [InlineData(1024, 13)]
    [InlineData(2048, 17)]
    public void Q2KMatmul_IsExactWhenActivationsQuantizeLosslessly(int elementCount, int seed)
    {
        int superBlocks = elementCount / 256;
        byte[] raw = new byte[superBlocks * 84];
        var rng = new Random(seed);
        for (int b = 0; b < superBlocks; b++)
        {
            int o = b * 84;
            for (int i = 0; i < 80; i++)
                raw[o + i] = (byte)rng.Next(256);
            BinaryPrimitives.WriteUInt16LittleEndian(raw.AsSpan(o + 80), HalfBits(0.02f));
            BinaryPrimitives.WriteUInt16LittleEndian(raw.AsSpan(o + 82), HalfBits(0.005f));
        }

        // Q8_K scales by max|x|/127, so integer multiples of that step round-trip
        // exactly. Pin the peak to element 0 so the scale is known.
        const float step = 1.0f / 127.0f;
        float[] input = new float[elementCount];
        input[0] = 1.0f;
        for (int i = 1; i < elementCount; i++)
            input[i] = (rng.Next(-127, 128)) * step;

        float[] actual = new float[1];
        ManagedQuantizedOps.DotRowBatchToFloat32(
            (int)GgmlTensorType.Q2_K, raw, 0, input, 0, elementCount, 1, elementCount, actual, 0);

        float[] dequantized = new float[elementCount];
        NativeDequant.DequantizeToFloat32((int)GgmlTensorType.Q2_K, raw, 0, dequantized, 0, elementCount);
        double expected = 0.0;
        for (int i = 0; i < elementCount; i++)
            expected += (double)dequantized[i] * input[i];

        double scale = Math.Max(1.0, Math.Abs(expected));
        Assert.True(Math.Abs(expected - actual[0]) <= 1e-4 * scale,
            $"Q2_K integer dot {actual[0]} against exact {expected} over {elementCount} elements");
    }

    /// <summary>
    /// The integer kernels are reached through TryAddmmQuantizedToFloat32, NOT
    /// through DotRowBatchToFloat32 (which always dequantizes). A test written
    /// against the latter compares dequantize-then-dot with itself and passes no
    /// matter what the kernel does, so every integer-dot type is checked here
    /// against a dequantized reference computed the same way ggml's own
    /// dequantizer does.
    /// </summary>
    [Theory]
    [InlineData(GgmlTensorType.Q2_K, 256, 7)]
    [InlineData(GgmlTensorType.Q2_K, 512, 11)]
    [InlineData(GgmlTensorType.Q2_K, 1024, 13)]
    [InlineData(GgmlTensorType.Q4_K, 512, 17)]
    [InlineData(GgmlTensorType.Q5_K, 512, 19)]
    [InlineData(GgmlTensorType.Q6_K, 512, 23)]
    public void IntegerDotAddmmMatchesDequantizedFloat(GgmlTensorType type, int cols, int seed)
    {
        const int rows = 3, batch = 5;
        int rowBytes = (int)ManagedQuantizedOps.RowSize((int)type, cols);
        byte[] weights = new byte[rows * rowBytes];
        var rng = new Random(seed);
        rng.NextBytes(weights);
        // Keep the super-block scales small so the dequantized magnitudes stay
        // in a range where Q8_K activation rounding is the only error term.
        int superBlocks = cols / 256;
        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < superBlocks; b++)
            {
                int o = r * rowBytes + b * (rowBytes / superBlocks);
                int scaleOffset = type == GgmlTensorType.Q2_K ? o + 80 : o;
                BinaryPrimitives.WriteUInt16LittleEndian(weights.AsSpan(scaleOffset), HalfBits(0.02f));
                BinaryPrimitives.WriteUInt16LittleEndian(weights.AsSpan(scaleOffset + 2), HalfBits(0.005f));
            }
        }

        float[] input = new float[batch * cols];
        for (int i = 0; i < input.Length; i++)
            input[i] = ((i * 37 % 61) - 30) * 0.021f;

        float[] actual = new float[batch * rows];
        Assert.True(ManagedQuantizedOps.TryAddmmQuantizedToFloat32(
            (int)type, weights, 0, cols, rows, input, 0, cols, batch, actual, 0, rows));

        float[] dequantized = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            NativeDequant.DequantizeToFloat32((int)type, weights, r * rowBytes, dequantized, r * cols, cols);

        float peak = 0f;
        foreach (float v in input) peak = Math.Max(peak, Math.Abs(v));
        for (int b = 0; b < batch; b++)
        {
            for (int r = 0; r < rows; r++)
            {
                double expected = 0, mass = 0;
                for (int i = 0; i < cols; i++)
                {
                    expected += (double)dequantized[r * cols + i] * input[b * cols + i];
                    mass += Math.Abs(dequantized[r * cols + i]);
                }
                double tolerance = peak / 127.0 * mass + 1e-3;
                Assert.True(Math.Abs(expected - actual[b * rows + r]) <= tolerance,
                    $"{type} integer addmm row {r} batch {b}: {actual[b * rows + r]} against dequantized {expected} (tolerance {tolerance})");
            }
        }
    }

    [Theory]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(9)]
    public void BatchedQ2KRowMatchesScalarReferenceWithOffsetsAndStrides(int rows)
    {
        const int cols = 512, stride = cols + 16, sourceOffset = 8, inputOffset = 13, outputOffset = 7;
        byte[] weights = new byte[sourceOffset + 2 * 84];
        new Random(1729).NextBytes(weights);
        for (int block = 0; block < 2; block++)
        {
            WriteHalf(weights, sourceOffset + block * 84 + 80, 0.02f);
            WriteHalf(weights, sourceOffset + block * 84 + 82, 0.005f);
        }
        float[] inputs = Enumerable.Range(0, inputOffset + rows * stride)
            .Select(i => (i % 31 - 15) * 0.017f).ToArray();
        float[] output = Enumerable.Repeat(-999f, outputOffset + rows + 1).ToArray();
        ManagedQuantizedOps.DotRowBatchToFloat32((int)GgmlTensorType.Q2_K, weights, sourceOffset,
            inputs, inputOffset, stride, rows, cols, output, outputOffset);
        float[] dequantized = new float[cols];
        NativeDequant.DequantizeToFloat32((int)GgmlTensorType.Q2_K, weights, sourceOffset, dequantized, 0, cols);
        for (int row = 0; row < rows; row++)
        {
            double expected = 0;
            for (int i = 0; i < cols; i++)
                expected += (double)dequantized[i] * inputs[inputOffset + row * stride + i];
            Assert.InRange(Math.Abs(output[outputOffset + row] - expected), 0, 1e-4 * Math.Max(1, Math.Abs(expected)));
        }
        Assert.All(output.Take(outputOffset), value => Assert.Equal(-999f, value));
        Assert.Equal(-999f, output[^1]);
    }

    /// <summary>
    /// The batched row helper must not repeatedly sweep the activation matrix
    /// for each quantization block. Compare with dequantizing the full row once.
    /// <para>
    /// A benchmark, not a correctness test: it compares two wall-clock times, which
    /// a shared CI runner does not hold steady (6.71 ms against 4.46 ms failed PR
    /// #387's ubuntu-latest job, where the code under test had not changed). So it
    /// runs with the other <c>Category=Bench</c> tests, outside the CI filter. The
    /// batched row's answers are checked against the scalar reference by
    /// <see cref="BatchedQ2KRowMatchesScalarReferenceWithOffsetsAndStrides"/>.
    /// </para>
    /// </summary>
    [Fact]
    [Trait("Category", "Bench")]
    public void Q2KMatmul_BatchedRowIsNotSlowerThanDequantizing()
    {
        // DotRowBatchToFloat32 dots ONE weight row against `batch` activation
        // vectors, which is the shape the kernel is measured on.
        const int cols = 4096, batch = 512, reps = 6;
        int superBlocks = cols / 256;
        byte[] weights = new byte[superBlocks * 84];
        var rng = new Random(4242);
        rng.NextBytes(weights);
        // d and dmin must stay small or the products overflow into noise.
        BinaryPrimitives.WriteUInt16LittleEndian(weights.AsSpan(80), HalfBits(0.02f));
        BinaryPrimitives.WriteUInt16LittleEndian(weights.AsSpan(82), HalfBits(0.005f));

        float[] input = new float[(long)batch * cols is var n && n < int.MaxValue ? batch * cols : 0];
        for (int i = 0; i < input.Length; i++) input[i] = ((i % 31) - 15) * 0.017f;
        float[] outBuf = new float[batch];

        ManagedQuantizedOps.DotRowBatchToFloat32((int)GgmlTensorType.Q2_K, weights, 0, input, 0, cols, batch, cols, outBuf, 0);
        var sw = Stopwatch.StartNew();
        for (int r = 0; r < reps; r++)
            ManagedQuantizedOps.DotRowBatchToFloat32((int)GgmlTensorType.Q2_K, weights, 0, input, 0, cols, batch, cols, outBuf, 0);
        sw.Stop();
        double batchedMs = sw.Elapsed.TotalMilliseconds / reps;

        // What Q2_K did before it had a kernel: dequantize the row, then float dot.
        float[] row = new float[cols];
        sw.Restart();
        for (int r = 0; r < reps; r++)
        {
            NativeDequant.DequantizeToFloat32((int)GgmlTensorType.Q2_K, weights, 0, row, 0, cols);
            for (int i = 0; i < batch; i++)
                outBuf[i] = Dot(row, input, i * cols, cols);
        }
        sw.Stop();
        double dequantMs = sw.Elapsed.TotalMilliseconds / reps;

        double gflops = 2.0 * batch * cols / 1e9;
        Console.WriteLine($"Q2_K 1x{cols} against {batch} activations: batched {batchedMs:F2} ms "
                        + $"({gflops / (batchedMs / 1000):F1} GFLOP/s), dequantize+float {dequantMs:F2} ms "
                        + $"({gflops / (dequantMs / 1000):F1} GFLOP/s), ratio {dequantMs / batchedMs:F2}x");
        Assert.True(batchedMs <= dequantMs * 1.5,
            $"batched path {batchedMs:F2} ms should not be materially slower than dequantizing {dequantMs:F2} ms");
    }

    private static ushort HalfBits(float value) => BitConverter.HalfToUInt16Bits((Half)value);

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
