using System.Text;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class NemotronAudioEncoderTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(false, false, 8)]
    [InlineData(false, true, 8)]
    [InlineData(true, false, 8)]
    [InlineData(true, true, 8)]
    [InlineData(false, false, 128)]
    [InlineData(false, true, 128)]
    [InlineData(true, false, 128)]
    [InlineData(true, true, 128)]
    public void OfficialParakeetAndProjection_AllRowsMatch(bool bf16, bool native, int melBins)
    {
        using var json = ReadReference(melBins);
        JsonElement fixture = json.RootElement.GetProperty("fixtures")[bf16 ? 1 : 0];
        string path = Path.Combine(Path.GetTempPath(), "ts-nemotron-audio-" + Guid.NewGuid().ToString("N") + ".gguf");
        WriteCompanion(path, fixture, bf16, melBins);
        try
        {
            IAllocator allocator = native ? new GgmlAllocator(new GgmlContext(new[] { 0 }, GgmlBackendType.Cpu), 0)
                : new CpuAllocator(BlasEnum.DotNet);
            var trace = new Dictionary<string, float[]>();
            using var encoder = new NemotronAudioEncoder(path, allocator, (name, values) => trace[name] = values);
            double worst = 0;
            foreach (var example in fixture.GetProperty("cases").EnumerateArray())
            {
                int frames = example.GetProperty("frames").GetInt32(), valid = example.GetProperty("valid_frames").GetInt32();
                float[] input = Floats(example.GetProperty("input")), before = (float[])input.Clone();
                float[] expected = Floats(example.GetProperty("expected"));
                using var result = encoder.Encode(input, frames, valid);
                Assert.Equal(new long[] { NemotronAudioEncoder.OutputLength(frames), 6 }, result.Sizes);
                float[] actual = result.GetElementsAsFloat(expected.Length);
                if (bf16 && example.TryGetProperty("details", out var details))
                {
                    foreach (var (name, values) in trace)
                    {
                        if (!details.TryGetProperty(name, out var reference)) continue;
                        float[] refValues = Floats(reference);
                        if (values.Length != refValues.Length) continue;
                        double delta = values.Zip(refValues, (a, b) => Math.Abs(a - b)).Max();
                        if (delta > 0) output.WriteLine($"frames={frames} stage={name} max_abs={delta:R}");
                        if (delta > 0 && name.EndsWith(".context", StringComparison.Ordinal))
                        {
                            output.WriteLine("expected=" + string.Join(",", refValues));
                            output.WriteLine("actual=" + string.Join(",", values));
                        }
                    }
                }
                for (int i = 0; i < actual.Length; i++)
                {
                    double delta = Math.Abs(actual[i] - expected[i]); worst = Math.Max(worst, delta);
                    // Set before observing this implementation: F32 arithmetic
                    // tolerance, or a small absolute/relative BF16 rounding bound.
                    double tolerance = bf16 ? 0.001 + 0.01 * Math.Abs(expected[i]) : 2e-5 + 2e-4 * Math.Abs(expected[i]);
                    Assert.True(double.IsFinite(delta) && delta <= tolerance,
                        $"bf16={bf16} native={native} frames={frames} valid={valid} row={i / 6} channel={i % 6}: expected={expected[i]:R} actual={actual[i]:R} delta={delta:R} tolerance={tolerance:R}");
                }
                Assert.Equal(before, input);
                using var repeat = encoder.Encode(input, frames, valid);
                Assert.Equal(actual, repeat.GetElementsAsFloat(actual.Length));
            }
            output.WriteLine($"All six official-reference clips passed; maximum absolute difference={worst:R}.");
            Assert.Throws<ArgumentException>(() => encoder.Encode(new float[melBins], 1, 2));
            Assert.Throws<ArgumentException>(() => encoder.Encode(new float[melBins - 1], 1, 1));
            float[] invalid = new float[melBins]; invalid[0] = float.NaN;
            Assert.Throws<ArgumentException>(() => encoder.Encode(invalid, 1, 1));
        }
        finally { File.Delete(path); }
    }

    internal static JsonDocument ReadReference(int melBins)
    {
        string file = melBins == 128 ? "reference128.json" : "reference.json";
        using var stream = typeof(NemotronAudioEncoderTests).Assembly.GetManifestResourceStream("InferenceWeb.Tests.Fixtures.NemotronAudio." + file)!;
        return JsonDocument.Parse(stream);
    }
    private static float[] Floats(JsonElement array) => array.EnumerateArray().Select(x => x.GetSingle()).ToArray();

    internal static void WriteCompanion(string path, JsonElement fixture, bool bf16, int melBins)
    {
        var metadata = new Dictionary<string, object>
        {
            ["general.architecture"] = "nemotron_audio", ["nemotron.audio.hidden_size"] = 8u,
            ["nemotron.audio.num_attention_heads"] = 2u, ["nemotron.audio.num_hidden_layers"] = 2u,
            ["nemotron.audio.subsampling_conv_channels"] = 4u, ["nemotron.audio.num_mel_bins"] = (uint)melBins,
            ["nemotron.audio.intermediate_size"] = 16u, ["nemotron.audio.conv_kernel_size"] = 3u,
            ["nemotron.audio.projection_hidden_size"] = 12u, ["nemotron.audio.projection_dim"] = 6u,
            ["nemotron.audio.compute_bf16"] = bf16,
        };
        var weights = fixture.GetProperty("weights").EnumerateArray().ToArray();
        using var writer = new BinaryWriter(File.Create(path), Encoding.UTF8);
        void String(string value) { byte[] bytes = Encoding.UTF8.GetBytes(value); writer.Write((ulong)bytes.Length); writer.Write(bytes); }
        writer.Write(0x46554747u); writer.Write(3u); writer.Write((ulong)weights.Length); writer.Write((ulong)metadata.Count);
        foreach (var (name, value) in metadata)
        {
            String(name);
            if (value is uint number) { writer.Write(4u); writer.Write(number); }
            else if (value is bool flag) { writer.Write(7u); writer.Write(flag); }
            else { writer.Write(8u); String((string)value); }
        }
        ulong offset = 0;
        foreach (var weight in weights)
        {
            String(weight.GetProperty("name").GetString()!);
            var dims = weight.GetProperty("shape").EnumerateArray().Select(v => v.GetUInt64()).Reverse().ToArray();
            writer.Write((uint)dims.Length); foreach (ulong dim in dims) writer.Write(dim);
            writer.Write(0u); writer.Write(offset);
            offset += ((ulong)weight.GetProperty("values").GetArrayLength() * 4 + 31) / 32 * 32;
        }
        void Pad() { while (writer.BaseStream.Position % 32 != 0) writer.Write((byte)0); }
        Pad();
        foreach (var weight in weights) { foreach (float value in Floats(weight.GetProperty("values"))) writer.Write(value); Pad(); }
    }
}
