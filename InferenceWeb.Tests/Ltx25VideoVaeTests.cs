using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Ltx25;

namespace InferenceWeb.Tests;

public sealed class Ltx25VideoVaeTests
{
    private static Stream Resource(string name) => typeof(Ltx25VideoVaeTests).Assembly.GetManifestResourceStream(
        "InferenceWeb.Tests.Fixtures.Ltx25." + name)!;
    private static JsonDocument Reference(string name) => JsonDocument.Parse(Resource(name));
    private static float[] Values(JsonElement node) => node.EnumerateArray().Select(value => value.GetSingle()).ToArray();
    private static int[] Shape(JsonElement node) => node.EnumerateArray().Select(value => value.GetInt32()).ToArray();
    private static IAllocator Allocator() => Environment.GetEnvironmentVariable("TS_TEST_LTX25_VAE_BACKEND")?.ToLowerInvariant() switch
    {
        "cuda" => new CudaAllocator(), "mlx" => new MlxAllocator(),
        null or "" or "cpu" => new CpuAllocator(BlasEnum.DotNet),
        var name => throw new ArgumentException("Unknown video VAE test backend " + name),
    };
    private static void Close(float[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < actual.Length; ++i)
            Assert.True(float.IsFinite(actual[i]) && Math.Abs(actual[i] - expected[i]) <= 2e-5f + 2e-5f * Math.Abs(expected[i]),
                $"{label}[{i}] expected {expected[i]:R}, actual {actual[i]:R}");
    }

    [Theory]
    [InlineData("video-vae-two-frame.json")]
    [InlineData("video-vae-single-frame.json")]
    public void AllDecoderStagesMatchUnmodifiedOriginalWithBf16StoredWeights(string fixture)
    {
        using var reference = Reference(fixture);
        var expected = reference.RootElement;
        Assert.Equal("be6d6d0a69d11a0a399c462ed92606692e5f8697", expected.GetProperty("source_revision").GetString());
        using var checkpoint = new Checkpoint();
        Assert.Equal(expected.GetProperty("checkpoint_sha256").GetString(), checkpoint.Hash);
        IAllocator allocator = Allocator();
        try
        {
            using var decoder = new Ltx25VideoVaeDecoder(checkpoint.Path, allocator);
            Assert.Equal(2, decoder.LatentChannels);
            Assert.Equal(8, decoder.TemporalScale); Assert.Equal(16, decoder.SpatialScale);
            var stages = new Dictionary<string, (float[] Values, int[] Shape)>();
            int[] shape = Shape(expected.GetProperty("input_shape"));
            float[] input = Values(expected.GetProperty("input")), preserved = (float[])input.Clone();
            var actual = decoder.DecodeRaw(input, shape[1], shape[2], shape[3], trace: (name, values, dimensions) => stages.Add(name, (values, dimensions)));
            Assert.Equal(preserved, input);
            var oracleStages = expected.GetProperty("stages");
            Assert.Equal(oracleStages.EnumerateObject().Count(), stages.Count);
            foreach (var stage in oracleStages.EnumerateObject())
            {
                Assert.True(stages.TryGetValue(stage.Name, out var observed), stage.Name);
                Assert.Equal(Shape(stage.Value.GetProperty("shape")), observed.Shape);
                Close(Values(stage.Value.GetProperty("values")), observed.Values, stage.Name);
            }
            Close(Values(oracleStages.GetProperty("decoded").GetProperty("values")), actual, "output");
        }
        finally { (allocator as IDisposable)?.Dispose(); }
    }

    [Fact]
    public void RgbConversionUsesFrameMajorChannelsAndIndependentTraceSnapshots()
    {
        using var reference = Reference("video-vae-single-frame.json");
        using var checkpoint = new Checkpoint();
        IAllocator allocator = Allocator();
        try
        {
            using var decoder = new Ltx25VideoVaeDecoder(checkpoint.Path, allocator);
            var root = reference.RootElement;
            int[] inputShape = Shape(root.GetProperty("input_shape"));
            float[] raw = Values(root.GetProperty("stages").GetProperty("decoded").GetProperty("values"));
            var rgb = decoder.Decode(Values(root.GetProperty("input")), inputShape[1], inputShape[2], inputShape[3],
                trace: (_, values, _) => Array.Fill(values, 1000f));
            var expected = new float[raw.Length];
            int pixels = raw.Length / 3;
            for (int pixel = 0; pixel < pixels; ++pixel)
                for (int c = 0; c < 3; ++c) expected[pixel * 3 + c] = Math.Clamp((raw[c * pixels + pixel] + 1) * .5f, 0, 1);
            Close(expected, rgb, "rgb");
            Assert.All(rgb, value => Assert.InRange(value, 0f, 1f));
        }
        finally { (allocator as IDisposable)?.Dispose(); }
    }

    [Theory]
    [InlineData("video-vae-two-frame.json")]
    [InlineData("video-vae-single-frame.json")]
    public void UntracedDecoderMatchesOriginalWithoutTraceInducedMaterialization(string fixture)
    {
        // Stage traces read back and materialize every activation. Exercise the
        // normal generation path too, with only its deliberate residency bounds.
        using var reference = Reference(fixture);
        using var checkpoint = new Checkpoint();
        IAllocator allocator = Allocator();
        try
        {
            using var decoder = new Ltx25VideoVaeDecoder(checkpoint.Path, allocator);
            var root = reference.RootElement;
            int[] shape = Shape(root.GetProperty("input_shape"));
            var actual = decoder.DecodeRaw(Values(root.GetProperty("input")), shape[1], shape[2], shape[3]);
            Close(Values(root.GetProperty("stages").GetProperty("decoded").GetProperty("values")), actual, "untraced output");
        }
        finally { (allocator as IDisposable)?.Dispose(); }
    }

    [Theory]
    [InlineData(0, 1, 1)]
    [InlineData(1, 0, 1)]
    [InlineData(1, 1, -1)]
    public void NonPositiveGeometryFailsBeforeExecutingDecoder(int frames, int height, int width)
    {
        using var checkpoint = new Checkpoint();
        using var decoder = new Ltx25VideoVaeDecoder(checkpoint.Path, new CpuAllocator(BlasEnum.DotNet));
        Assert.Throws<ArgumentOutOfRangeException>(() => decoder.DecodeRaw([], frames, height, width));
    }

    [Fact]
    public void WrongLengthAndNonFiniteLatentsAreRejected()
    {
        using var checkpoint = new Checkpoint();
        using var decoder = new Ltx25VideoVaeDecoder(checkpoint.Path, new CpuAllocator(BlasEnum.DotNet));
        Assert.Throws<ArgumentException>(() => decoder.DecodeRaw([1], 1, 1, 1));
        Assert.Throws<ArgumentException>(() => decoder.DecodeRaw([float.NaN, 1], 1, 1, 1));
        Assert.Throws<ArgumentException>(() => decoder.DecodeRaw([1, float.PositiveInfinity], 1, 1, 1));
    }

    [Fact]
    public void CancellationAndDisposalFailBeforeStageCallbacks()
    {
        using var checkpoint = new Checkpoint();
        using var decoder = new Ltx25VideoVaeDecoder(checkpoint.Path, new CpuAllocator(BlasEnum.DotNet));
        int called = 0;
        using var cancellation = new CancellationTokenSource(); cancellation.Cancel();
        Assert.Throws<OperationCanceledException>(() => decoder.DecodeRaw([1, 2], 1, 1, 1,
            cancellation.Token, (_, _, _) => called++));
        decoder.Dispose();
        Assert.Throws<ObjectDisposedException>(() => decoder.DecodeRaw([1, 2], 1, 1, 1));
        Assert.Equal(0, called);
    }

    [Fact]
    public void RepeatedDecodeDoesNotRetainTemporalState()
    {
        using var reference = Reference("video-vae-two-frame.json");
        using var checkpoint = new Checkpoint();
        IAllocator allocator = Allocator();
        try
        {
            using var decoder = new Ltx25VideoVaeDecoder(checkpoint.Path, allocator);
            var input = Values(reference.RootElement.GetProperty("input"));
            var first = decoder.DecodeRaw(input, 2, 1, 2);
            var second = decoder.DecodeRaw(input, 2, 1, 2);
            Assert.Equal(first, second);
        }
        finally { (allocator as IDisposable)?.Dispose(); }
    }

    [Theory]
    [InlineData("_class_name", "CausalDiffusionVAE")]
    [InlineData("norm_layer", "group_norm")]
    [InlineData("spatial_padding_mode", "reflect")]
    public void DifferentDecoderArchitecturesCannotSilentlyUseConvolutionalPath(string field, string value)
    {
        using var checkpoint = new Checkpoint(header =>
        {
            var metadata = header["__metadata__"]!.AsObject();
            var config = JsonNode.Parse(metadata["config"]!.GetValue<string>())!;
            config["vae"]![field] = value;
            metadata["config"] = config.ToJsonString();
        });
        Assert.Throws<NotSupportedException>(() => new Ltx25VideoVaeDecoder(checkpoint.Path, new CpuAllocator(BlasEnum.DotNet)));
    }

    [Theory]
    [InlineData("causal_decoder")]
    [InlineData("timestep_conditioning")]
    [InlineData("use_quant_conv")]
    public void UnsupportedConditioningCannotSilentlyChangeDecode(string field)
    {
        using var checkpoint = new Checkpoint(header =>
        {
            var metadata = header["__metadata__"]!.AsObject();
            var config = JsonNode.Parse(metadata["config"]!.GetValue<string>())!;
            config["vae"]![field] = true; metadata["config"] = config.ToJsonString();
        });
        Assert.Throws<NotSupportedException>(() => new Ltx25VideoVaeDecoder(checkpoint.Path, new CpuAllocator(BlasEnum.DotNet)));
    }

    [Theory]
    [InlineData("inject_noise")]
    [InlineData("residual")]
    public void UnsupportedBlockVariantsFailClosed(string flag)
    {
        using var checkpoint = new Checkpoint(header =>
        {
            var metadata = header["__metadata__"]!.AsObject();
            var config = JsonNode.Parse(metadata["config"]!.GetValue<string>())!;
            config["vae"]!["decoder_blocks"]![0]![1]![flag] = true;
            metadata["config"] = config.ToJsonString();
        });
        Assert.Throws<NotSupportedException>(() => new Ltx25VideoVaeDecoder(checkpoint.Path, new CpuAllocator(BlasEnum.DotNet)));
    }

    [Fact]
    public void WeightShapesAreValidatedBeforeInference()
    {
        // The byte count stays valid, while exchanging input/output axes breaks the expected architecture.
        using var checkpoint = new Checkpoint(header =>
            header["decoder.conv_in.conv.weight"]!["shape"] = new JsonArray(2, 16, 3, 3, 3));
        Assert.Throws<InvalidDataException>(() => new Ltx25VideoVaeDecoder(checkpoint.Path, new CpuAllocator(BlasEnum.DotNet)));
    }

    private sealed class Checkpoint : IDisposable
    {
        public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "ltx25-video-vae-" + Guid.NewGuid().ToString("N") + ".safetensors");
        public string Hash { get; }
        public Checkpoint(Action<JsonObject>? change = null)
        {
            using var source = Resource("video-vae-tiny.safetensors");
            using var memory = new MemoryStream(); source.CopyTo(memory);
            byte[] bytes = memory.ToArray();
            if (change == null) File.WriteAllBytes(Path, bytes);
            else
            {
                int headerSize = checked((int)BitConverter.ToUInt64(bytes, 0));
                var header = JsonNode.Parse(Encoding.UTF8.GetString(bytes, 8, headerSize))!.AsObject(); change(header);
                byte[] json = Encoding.UTF8.GetBytes(header.ToJsonString());
                using var target = File.Create(Path); target.Write(BitConverter.GetBytes((ulong)json.Length));
                target.Write(json); target.Write(bytes, 8 + headerSize, bytes.Length - 8 - headerSize);
            }
            using var file = File.OpenRead(Path); Hash = Convert.ToHexStringLower(SHA256.HashData(file));
        }
        public void Dispose() => File.Delete(Path);
    }
}
