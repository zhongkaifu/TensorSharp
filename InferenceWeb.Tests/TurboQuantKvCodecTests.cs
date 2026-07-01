// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Buffers.Binary;
using System.Linq;

namespace InferenceWeb.Tests;

/// <summary>
/// Tests for <see cref="TurboQuantKvCodec"/> — the int4 / int8 group-quantization
/// codec used by the paged KV tier. We exercise the codec directly (not through
/// the paged manager) so we can assert the precision envelope of each bit width
/// against ground-truth float buffers without involving a model stub. Realistic
/// values are sampled from a small symmetric range that mirrors the post-RoPE
/// K/V activations the codec sees in production.
/// </summary>
public class TurboQuantKvCodecTests
{
    private const int GroupSize = 32;

    [Fact]
    public void Int4_Float32_RoundTrip_BoundsRelativeError()
    {
        var codec = new TurboQuantKvCodec(KvCodecElementType.Float32, bits: 4);
        var values = SampleFloats(seed: 1, count: 4 * GroupSize, range: 4f);
        byte[] raw = FloatsToBytes(values);

        byte[] encoded = codec.Encode(raw);
        var decoded = new byte[raw.Length];
        Assert.True(codec.TryDecode(encoded, decoded));

        float[] roundTripped = BytesToFloats(decoded);
        // int4 with per-32 fp16 scale gives roughly 1/(2*qMax) = 1/14 relative
        // error in the worst case. Add slack for fp16 scale quantization.
        AssertRelativeError(values, roundTripped, maxAvgError: 0.10f, maxMaxError: 0.25f);
    }

    [Fact]
    public void Int2_Float32_RoundTrip_BoundsRelativeError()
    {
        var codec = new TurboQuantKvCodec(KvCodecElementType.Float32, bits: 2);
        var values = SampleFloats(seed: 21, count: 4 * GroupSize, range: 4f);
        byte[] raw = FloatsToBytes(values);

        byte[] encoded = codec.Encode(raw);
        var decoded = new byte[raw.Length];
        Assert.True(codec.TryDecode(encoded, decoded));

        float[] roundTripped = BytesToFloats(decoded);
        // Affine 2-bit spreads 4 codes across [min,max]; worst-case quant error
        // is half a step = (max-min)/6 ≈ 1/3 of the group range. fp16 scale/min
        // rounding adds a hair on top.
        AssertRelativeError(values, roundTripped, maxAvgError: 0.20f, maxMaxError: 0.50f);
    }

    [Fact]
    public void Int2_IsLessPreciseThanInt4_ButCompressesMore()
    {
        var values = SampleFloats(seed: 22, count: 8 * GroupSize, range: 4f);
        byte[] raw = FloatsToBytes(values);

        var int2 = new TurboQuantKvCodec(KvCodecElementType.Float32, bits: 2);
        var int4 = new TurboQuantKvCodec(KvCodecElementType.Float32, bits: 4);

        byte[] enc2 = int2.Encode(raw);
        byte[] enc4 = int4.Encode(raw);

        var dec2 = RoundTrip(int2, raw, raw.Length);
        var dec4 = RoundTrip(int4, raw, raw.Length);
        float err2 = MeanAbsError(values, BytesToFloats(dec2));
        float err4 = MeanAbsError(values, BytesToFloats(dec4));

        // 2-bit trades precision for footprint: coarser than int4 but a smaller
        // encoded block.
        Assert.True(err2 > err4,
            $"int2 ({err2:G4}) should be coarser than int4 ({err4:G4})");
        Assert.True(enc2.Length < enc4.Length,
            $"int2 ({enc2.Length} B) should be smaller than int4 ({enc4.Length} B)");
    }

    [Fact]
    public void Int2_CompressesFloat32_ByMoreThan8x()
    {
        // 32 fp32 = 128 raw bytes. Encoded per group: 2 (scale) + 2 (min) +
        // 8 (codes) = 12 bytes + 16-byte header amortized. For many groups this
        // approaches 128/12 ≈ 10.7x.
        var codec = new TurboQuantKvCodec(KvCodecElementType.Float32, bits: 2);
        var values = SampleFloats(seed: 23, count: 64 * GroupSize, range: 1f);
        byte[] raw = FloatsToBytes(values);

        byte[] encoded = codec.Encode(raw);
        Assert.True(encoded.Length * 8 < raw.Length,
            $"int2 should achieve >8x compression on fp32; got {raw.Length}->{encoded.Length}");
    }

    [Fact]
    public void Int2_Float16_RoundTrip_BoundsRelativeError()
    {
        var codec = new TurboQuantKvCodec(KvCodecElementType.Float16, bits: 2);
        var values = SampleFloats(seed: 24, count: 4 * GroupSize, range: 4f);
        byte[] raw = HalvesToBytes(values);

        byte[] encoded = codec.Encode(raw);
        var decoded = new byte[raw.Length];
        Assert.True(codec.TryDecode(encoded, decoded));

        float[] roundTripped = BytesToHalves(decoded);
        AssertRelativeError(values, roundTripped, maxAvgError: 0.20f, maxMaxError: 0.50f);
    }

    [Fact]
    public void Int8_Float32_RoundTrip_IsMorePreciseThanInt4()
    {
        var values = SampleFloats(seed: 2, count: 8 * GroupSize, range: 4f);
        byte[] raw = FloatsToBytes(values);

        var int4 = new TurboQuantKvCodec(KvCodecElementType.Float32, bits: 4);
        var int8 = new TurboQuantKvCodec(KvCodecElementType.Float32, bits: 8);

        var dec4 = RoundTrip(int4, raw, raw.Length);
        var dec8 = RoundTrip(int8, raw, raw.Length);
        float[] r4 = BytesToFloats(dec4);
        float[] r8 = BytesToFloats(dec8);

        float err4 = MeanAbsError(values, r4);
        float err8 = MeanAbsError(values, r8);
        Assert.True(err8 < err4,
            $"int8 ({err8:G4}) should be strictly more precise than int4 ({err4:G4})");
    }

    [Fact]
    public void Int4_Float16_RoundTrip_BoundsRelativeError()
    {
        var codec = new TurboQuantKvCodec(KvCodecElementType.Float16, bits: 4);
        var values = SampleFloats(seed: 3, count: 4 * GroupSize, range: 4f);

        // Snap inputs to fp16 first so the round-trip target is what fp16 can
        // represent (the test must not measure fp32→fp16 quantization too).
        var fp16Inputs = values
            .Select(v => (float)System.BitConverter.UInt16BitsToHalf(
                System.BitConverter.HalfToUInt16Bits((System.Half)v)))
            .ToArray();
        byte[] raw = HalvesToBytes(values);

        byte[] encoded = codec.Encode(raw);
        var decoded = new byte[raw.Length];
        Assert.True(codec.TryDecode(encoded, decoded));

        float[] roundTripped = BytesToHalves(decoded);
        AssertRelativeError(fp16Inputs, roundTripped, maxAvgError: 0.10f, maxMaxError: 0.25f);
    }

    [Fact]
    public void Int4_CompressesPayload_ByExpectedRatio()
    {
        // 32 fp32 elements = 128 raw bytes. Encoded: 16-byte header + 2-byte
        // scale + 16-byte packed nibbles per group = 16 + 18*groups. For 4
        // groups: 16 + 72 = 88 bytes vs raw 512 -> ~5.8x.
        var codec = new TurboQuantKvCodec(KvCodecElementType.Float32, bits: 4);
        var values = SampleFloats(seed: 5, count: 4 * GroupSize, range: 1f);
        byte[] raw = FloatsToBytes(values);

        byte[] encoded = codec.Encode(raw);
        Assert.True(encoded.Length < raw.Length / 4,
            $"int4 should achieve at least 4x compression on fp32 input; got {raw.Length}->{encoded.Length}");
    }

    [Fact]
    public void Int8_CompressesPayload_ByExpectedRatio()
    {
        // 32 fp16 elements = 64 raw bytes. Encoded: 16 + 34*groups.
        // 8 groups: 16 + 272 = 288 vs raw 512 = ~1.8x.
        var codec = new TurboQuantKvCodec(KvCodecElementType.Float16, bits: 8);
        var values = SampleFloats(seed: 6, count: 8 * GroupSize, range: 1f);
        byte[] raw = HalvesToBytes(values);

        byte[] encoded = codec.Encode(raw);
        Assert.True(encoded.Length < raw.Length,
            $"int8 on fp16 should still be a net compression; got {raw.Length}->{encoded.Length}");
    }

    [Fact]
    public void Q8_0_Passthrough_RoundTripsExactly()
    {
        // For Q8_0 the codec is required to leave the payload alone (header
        // notwithstanding) so the decode reproduces every byte verbatim.
        var codec = new TurboQuantKvCodec(KvCodecElementType.Q8_0, bits: 4);
        var rng = new Random(7);
        byte[] raw = new byte[34 * 4]; // four nominal q8_0 blocks
        rng.NextBytes(raw);

        byte[] encoded = codec.Encode(raw);
        var decoded = new byte[raw.Length];
        Assert.True(codec.TryDecode(encoded, decoded));
        Assert.Equal(raw, decoded);
    }

    [Fact]
    public void Decode_RejectsCrossCodecPayloads()
    {
        var fp32Int4 = new TurboQuantKvCodec(KvCodecElementType.Float32, bits: 4);
        var fp32Int8 = new TurboQuantKvCodec(KvCodecElementType.Float32, bits: 8);

        var values = SampleFloats(seed: 8, count: 2 * GroupSize, range: 4f);
        byte[] raw = FloatsToBytes(values);
        byte[] encoded4 = fp32Int4.Encode(raw);
        var decoded = new byte[raw.Length];

        // Wrong bit-width codec must reject without producing partial output.
        Assert.False(fp32Int8.TryDecode(encoded4, decoded));
    }

    [Fact]
    public void Decode_RejectsCrossDtypePayloads()
    {
        var fp32 = new TurboQuantKvCodec(KvCodecElementType.Float32, bits: 4);
        var fp16 = new TurboQuantKvCodec(KvCodecElementType.Float16, bits: 4);

        var values = SampleFloats(seed: 9, count: 2 * GroupSize, range: 4f);
        byte[] raw = FloatsToBytes(values);
        byte[] encoded = fp32.Encode(raw);

        // fp16 codec sees the fp32 header dtype byte and bails out.
        var decoded = new byte[raw.Length / 2];
        Assert.False(fp16.TryDecode(encoded, decoded));
    }

    [Fact]
    public void Decode_RejectsTruncatedPayload()
    {
        var codec = new TurboQuantKvCodec(KvCodecElementType.Float32, bits: 4);
        var values = SampleFloats(seed: 10, count: 2 * GroupSize, range: 4f);
        byte[] encoded = codec.Encode(FloatsToBytes(values));

        // Lose the last group's bytes - decoder must refuse rather than
        // returning silently corrupted floats.
        byte[] truncated = encoded[..(encoded.Length - 4)];
        var decoded = new byte[FloatsToBytes(values).Length];
        Assert.False(codec.TryDecode(truncated, decoded));
    }

    [Fact]
    public void FromEnvironment_ReturnsNullWhenVarUnset()
    {
        var previous = Environment.GetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS");
        try
        {
            Environment.SetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS", null);
            Assert.Null(TurboQuantKvCodec.FromEnvironment(KvCodecElementType.Float32));
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS", previous);
        }
    }

    [Fact]
    public void FromEnvironment_HonorsInt2()
    {
        var previous = Environment.GetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS");
        try
        {
            Environment.SetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS", "2");
            var codec = TurboQuantKvCodec.FromEnvironment(KvCodecElementType.Float16);
            Assert.NotNull(codec);
            Assert.Equal(2, codec.BitsPerElement);
            Assert.Equal("turboquant-int2", codec.Name);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS", previous);
        }
    }

    [Fact]
    public void Decode_RejectsInt2PayloadUnderInt4Codec()
    {
        var int2 = new TurboQuantKvCodec(KvCodecElementType.Float32, bits: 2);
        var int4 = new TurboQuantKvCodec(KvCodecElementType.Float32, bits: 4);

        var values = SampleFloats(seed: 25, count: 2 * GroupSize, range: 4f);
        byte[] raw = FloatsToBytes(values);
        byte[] encoded2 = int2.Encode(raw);

        // A codec configured for a different bit width must refuse the block
        // rather than misinterpreting the affine layout as symmetric nibbles.
        var decoded = new byte[raw.Length];
        Assert.False(int4.TryDecode(encoded2, decoded));
    }

    [Fact]
    public void FromEnvironment_HonorsInt4()
    {
        var previous = Environment.GetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS");
        try
        {
            Environment.SetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS", "4");
            var codec = TurboQuantKvCodec.FromEnvironment(KvCodecElementType.Float16);
            Assert.NotNull(codec);
            Assert.Equal(4, codec.BitsPerElement);
            Assert.Equal("turboquant-int4", codec.Name);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS", previous);
        }
    }

    [Fact]
    public void FromEnvironment_SkipsQ8_0_EvenWhenSet()
    {
        var previous = Environment.GetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS");
        try
        {
            // Q8_0 caches are already 8-bit quantized; the codec must short-
            // circuit instead of stacking a second quantization on top.
            Environment.SetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS", "4");
            Assert.Null(TurboQuantKvCodec.FromEnvironment(KvCodecElementType.Q8_0));
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS", previous);
        }
    }

    [Fact]
    public void FromEnvironment_RejectsUnsupportedBitWidths()
    {
        var previous = Environment.GetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS");
        try
        {
            Environment.SetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS", "6");
            Assert.Null(TurboQuantKvCodec.FromEnvironment(KvCodecElementType.Float32));
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS", previous);
        }
    }

    [Fact]
    public void FromEnvironment_ModelOverload_BlocksRecurrentStateModels()
    {
        // Empirical regression test: Qwen3.5 / 3.6 GatedDeltaNet snapshots
        // record running SSM state per block. Quantizing those bytes
        // through int4 collapses next-token logits to all zeros on greedy
        // sampling. The model-aware overload must return null when
        // RequiresPerBlockCapture=true, even with the env var set.
        var previous = Environment.GetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS");
        try
        {
            Environment.SetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS", "4");
            var recurrent = new GatedFakeModel(KvCodecElementType.Float16, requiresPerBlockCapture: true);
            var pureAttn = new GatedFakeModel(KvCodecElementType.Float16, requiresPerBlockCapture: false);

            Assert.Null(TurboQuantKvCodec.FromEnvironment(recurrent));
            var codec = TurboQuantKvCodec.FromEnvironment(pureAttn);
            Assert.NotNull(codec);
            Assert.Equal(4, codec.BitsPerElement);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS", previous);
        }
    }

    [Fact]
    public void FromEnvironment_ModelOverload_ReturnsNullForNullModel()
    {
        var previous = Environment.GetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS");
        try
        {
            Environment.SetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS", "4");
            Assert.Null(TurboQuantKvCodec.FromEnvironment((IModelArchitecture)null));
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS", previous);
        }
    }

    private sealed class GatedFakeModel : IModelArchitecture
    {
        private readonly bool _requiresPerBlockCapture;

        public GatedFakeModel(KvCodecElementType dtype, bool requiresPerBlockCapture)
        {
            KVStateElementType = dtype;
            _requiresPerBlockCapture = requiresPerBlockCapture;
        }

        public ModelConfig Config { get; } = new ModelConfig();
        public ITokenizer Tokenizer => null;
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;
        public float[] Forward(int[] tokens) => Array.Empty<float>();
        public void ResetKVCache() { }
        public void TruncateKVCache(int tokenCount) { }
        public void Dispose() { }
        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint => "gated-fake";
        public KvCodecElementType KVStateElementType { get; }
        public bool RequiresPerBlockCapture => _requiresPerBlockCapture;
    }

    // ---- Helpers ----

    private static float[] SampleFloats(int seed, int count, float range)
    {
        var rng = new Random(seed);
        var buf = new float[count];
        for (int i = 0; i < count; i++)
            buf[i] = ((float)rng.NextDouble() * 2f - 1f) * range;
        return buf;
    }

    private static byte[] FloatsToBytes(float[] values)
    {
        byte[] bytes = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4, 4), values[i]);
        return bytes;
    }

    private static float[] BytesToFloats(byte[] bytes)
    {
        float[] values = new float[bytes.Length / 4];
        for (int i = 0; i < values.Length; i++)
            values[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4, 4));
        return values;
    }

    private static byte[] HalvesToBytes(float[] values)
    {
        byte[] bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
        {
            ushort bits = System.BitConverter.HalfToUInt16Bits((System.Half)values[i]);
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i * 2, 2), bits);
        }
        return bytes;
    }

    private static float[] BytesToHalves(byte[] bytes)
    {
        float[] values = new float[bytes.Length / 2];
        for (int i = 0; i < values.Length; i++)
        {
            ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(i * 2, 2));
            values[i] = (float)System.BitConverter.UInt16BitsToHalf(bits);
        }
        return values;
    }

    private static byte[] RoundTrip(IKvBlockCodec codec, byte[] raw, int decodedLength)
    {
        byte[] encoded = codec.Encode(raw);
        var decoded = new byte[decodedLength];
        Assert.True(codec.TryDecode(encoded, decoded));
        return decoded;
    }

    private static float MeanAbsError(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        double sum = 0;
        for (int i = 0; i < expected.Length; i++)
            sum += Math.Abs(expected[i] - actual[i]);
        return (float)(sum / expected.Length);
    }

    private static void AssertRelativeError(float[] expected, float[] actual, float maxAvgError, float maxMaxError)
    {
        Assert.Equal(expected.Length, actual.Length);

        // Per-group scale, so we measure error relative to each group's
        // dynamic range rather than the global maximum.
        int groups = expected.Length / GroupSize;
        double globalAvg = 0;
        double globalMax = 0;

        for (int g = 0; g < groups; g++)
        {
            float groupMax = 0;
            for (int i = 0; i < GroupSize; i++)
            {
                float a = Math.Abs(expected[g * GroupSize + i]);
                if (a > groupMax) groupMax = a;
            }

            // Avoid divide-by-zero on degenerate all-zero groups.
            float refMag = groupMax > 1e-6f ? groupMax : 1f;
            double sumAbs = 0;
            double localMax = 0;
            for (int i = 0; i < GroupSize; i++)
            {
                double err = Math.Abs(expected[g * GroupSize + i] - actual[g * GroupSize + i]) / refMag;
                sumAbs += err;
                if (err > localMax) localMax = err;
            }
            globalAvg += sumAbs / GroupSize;
            if (localMax > globalMax) globalMax = localMax;
        }

        double avg = globalAvg / groups;
        Assert.True(avg < maxAvgError,
            $"Average relative error {avg:G4} exceeded bound {maxAvgError:G4}");
        Assert.True(globalMax < maxMaxError,
            $"Max relative error {globalMax:G4} exceeded bound {maxMaxError:G4}");
    }
}
