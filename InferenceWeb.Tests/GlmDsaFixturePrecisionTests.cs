// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Buffers.Binary;
using System.Text;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>GGUF serialization and managed dtype conversion only. No native
/// library, device initialization, model execution or performance claim.</summary>
public sealed class GlmDsaFixturePrecisionTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-glm-fixture-precision-" + Guid.NewGuid().ToString("N"));

    public GlmDsaFixturePrecisionTests() => Directory.CreateDirectory(_directory);
    public void Dispose() => Directory.Delete(_directory, recursive: true);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EveryTensorOffsetMatchesItsTypeSizedPayloadAndFileExtent(bool bf16)
    {
        string path = GlmDsaSyntheticModelBuilder.WriteGlm5NextCudaFixture(Path.Combine(_directory, "offsets.gguf"),
            bf16SensitiveProjections: bf16);
        using var file = GgufFile.OpenForMemoryPlanning(path);
        ulong offset = 0;
        foreach (var tensor in file.Tensors.Values)
        {
            Assert.Equal(offset, tensor.Offset);
            long bytes = tensor.Type switch
            {
                GgmlTensorType.F32 => checked(tensor.NumElements * 4),
                GgmlTensorType.BF16 => checked(tensor.NumElements * 2),
                GgmlTensorType.Q8_0 => checked(tensor.NumElements / 32 * 34),
                _ => throw new InvalidDataException("Unexpected synthetic fixture type: " + tensor.Type),
            };
            Assert.Equal(bytes, file.GetTensorByteCount(tensor));
            Assert.Equal(bytes, file.ReadTensorData(tensor).LongLength);
            offset += ((ulong)bytes + 31) / 32 * 32;
        }
        Assert.Equal(file.DataOffset + checked((long)offset), new FileInfo(path).Length);
        file.ThrowIfTruncated();
        Assert.Equal(bf16 ? 18 : 0, file.Tensors.Values.Count(tensor => tensor.Type == GgmlTensorType.BF16));
    }

    [Fact]
    public unsafe void All18Bf16SensitivePayloadsMatchOriginalF32RoundingAndManagedRoundtrip()
    {
        string path = GlmDsaSyntheticModelBuilder.WriteGlm5NextCudaFixture(Path.Combine(_directory, "payloads.gguf"),
            bf16SensitiveProjections: true);
        using var file = GgufFile.OpenForMemoryPlanning(path);
        string[] names = Enumerable.Range(0, 4).SelectMany(layer => new[] { "hc_attn_fn.weight", "hc_ffn_fn.weight" }
            .Concat(layer % 2 == 0 ? new[] { "ssm_f_a.weight", "ssm_f_b.weight", "ssm_g_a.weight", "ssm_g_b.weight", "ssm_beta.weight" } : [])
            .Select(suffix => $"blk.{layer}.{suffix}")).ToArray();
        Assert.Equal(18, names.Length);
        Assert.Equal(names.Order(), file.Tensors.Values.Where(t => t.Type == GgmlTensorType.BF16).Select(t => t.Name).Order());
        long checkedElements = 0;
        foreach (string name in names)
        {
            var tensor = file.Tensors[name];
            byte[] raw = file.ReadTensorData(tensor);
            Assert.Equal(checked(tensor.NumElements * 2), raw.LongLength);
            int count = checked((int)tensor.NumElements);
            var encoded = new ushort[count];
            var decoded = new float[count];
            for (int i = 0; i < count; ++i)
                encoded[i] = BinaryPrimitives.ReadUInt16LittleEndian(raw.AsSpan(i * 2, 2));
            fixed (ushort* source = encoded)
            fixed (float* destination = decoded)
                SafetensorsKernels.Bf16ToF32(source, destination, count);

            // Independently reconstruct the documented deterministic fixture
            // source. The BF16 oracle compares discarded bits, rather than
            // repeating the writer's integer-bias rounding implementation.
            uint state = 0x811c9dc5;
            foreach (byte value in Encoding.UTF8.GetBytes(name)) state = unchecked((state ^ value) * 0x01000193);
            if (state == 0) state = 0x9e3779b9;
            float scale = name.Contains("hc_", StringComparison.Ordinal) ? 0.05f : 0.08f;
            for (int i = 0; i < count; ++i)
            {
                double sum = 0;
                for (int draw = 0; draw < 4; ++draw)
                {
                    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                    sum += state / 4294967296.0;
                }
                uint bits = BitConverter.SingleToUInt32Bits((float)((sum - 2.0) * scale));
                uint high = bits >> 16, low = bits & 0xffff;
                if (low > 0x8000 || (low == 0x8000 && (high & 1) != 0)) ++high;
                Assert.Equal((ushort)high, encoded[i]);
                Assert.Equal(high << 16, BitConverter.SingleToUInt32Bits(decoded[i]));
                ++checkedElements;
            }
        }
        Assert.True(checkedElements > 100000, "All original sensitive projection values must be checked.");
    }

    [Fact]
    public void Bf16EncodingHandlesTiesSignedZeroSubnormalsInfinityAndNaNPayloads()
    {
        uint[] bits = [0u, 0x80000000u, 0x3f808000u, 0x3f818000u, 0xbf808000u, 0xbf818000u,
            0x00008000u, 0x00018000u, 0x7f800000u, 0xff800000u, 0x7f800001u, 0xffa12345u, 0x7f7fffffu];
        ushort[] expected = [0, 0x8000, 0x3f80, 0x3f82, 0xbf80, 0xbf82, 0, 2, 0x7f80, 0xff80, 0x7fc0, 0xffe1, 0x7f80];
        byte[] raw = GlmDsaSyntheticModelBuilder.EncodeBf16Fixture(bits.Select(BitConverter.UInt32BitsToSingle).ToArray());
        Assert.Equal(expected.Length * 2, raw.Length);
        for (int i = 0; i < expected.Length; ++i)
            Assert.Equal(expected[i], BinaryPrimitives.ReadUInt16LittleEndian(raw.AsSpan(i * 2, 2)));
    }

    [Fact]
    public void HistoricalDefaultQ8FixtureIsIdenticalToExplicitlyDisabledBf16Variant()
    {
        string baseline = GlmDsaSyntheticModelBuilder.WriteGlm5NextCudaFixture(Path.Combine(_directory, "default.gguf"));
        string disabled = GlmDsaSyntheticModelBuilder.WriteGlm5NextCudaFixture(Path.Combine(_directory, "disabled.gguf"),
            bf16SensitiveProjections: false);
        Assert.Equal(File.ReadAllBytes(baseline), File.ReadAllBytes(disabled));
    }
}
