using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Ltx25;

namespace InferenceWeb.Tests;

public sealed class Ltx25PromptEncoderTests
{
    private static JsonElement Oracle()
    {
        using var stream = Resource("prompt-oracle.json"); using var document = JsonDocument.Parse(stream);
        Assert.Equal("be6d6d0a69d11a0a399c462ed92606692e5f8697", document.RootElement.GetProperty("source_revision").GetString());
        Assert.Equal("5.10.1", document.RootElement.GetProperty("transformers_version").GetString());
        return document.RootElement.Clone();
    }
    private static Stream Resource(string name) => typeof(Ltx25PromptEncoderTests).Assembly.GetManifestResourceStream("InferenceWeb.Tests.Fixtures.Ltx25." + name)!;
    private static float[] Values(JsonElement value) => value.EnumerateArray().Select(v => v.GetSingle()).ToArray();
    private static void Equal(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i) Assert.True(float.IsFinite(actual[i]) && Math.Abs(expected[i] - actual[i]) <= tolerance,
            $"[{i}] expected {expected[i]:R}, actual {actual[i]:R}, tolerance {tolerance}");
    }
    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)]
    public void AuthenticBf16GemmaHiddenStatesAndLearnedAggregatesMatchHuggingFace(int index)
    {
        using var fixture = new Fixture(); using var backend = new Backend();
        using var encoder = new Ltx25PromptEncoder(fixture.Encoder, fixture.Connectors, backend.Allocator);
        var trace = new Dictionary<string, float[]>();
        if (Environment.GetEnvironmentVariable("TS_TEST_LTX25_PROMPT_CAPTURE") != null) encoder.Trace = (name, data) => trace[name] = data;
        var test = Oracle().GetProperty("cases")[index];
        var ids = encoder.Tokenizer.Encode(test.GetProperty("prompt").GetString()!);
        Assert.Equal(test.GetProperty("ids").EnumerateArray().Select(v => v.GetInt32()), ids);
        float[][] expected = test.GetProperty("hidden_states").EnumerateArray().Select(Values).ToArray();
        float[][] actual = encoder.EncodeHiddenStates(ids);
        if (Environment.GetEnvironmentVariable("TS_TEST_LTX25_PROMPT_CAPTURE") is string directory)
        {
            Directory.CreateDirectory(directory);
            File.WriteAllText(Path.Combine(directory, "case-" + index + ".json"), JsonSerializer.Serialize(new { expected, actual, trace }));
        }
        Assert.Equal(expected.Length, actual.Length);
        for (int layer = 0; layer < actual.Length; ++layer) Equal(expected[layer], actual[layer], 1e-5f);
        using var features = new Tensor(backend.Allocator, DType.Float32, ids.Length, 48);
        var normalized = encoder.Features(actual, ids.Length); features.SetElementsAsFloat(normalized);
        var video = encoder.Aggregate(features, 16, "video"); var audio = encoder.Aggregate(features, 8, "audio");
        if (Environment.GetEnvironmentVariable("TS_TEST_LTX25_PROMPT_CAPTURE") is string capture)
            File.WriteAllText(Path.Combine(capture, "aggregate-" + index + ".json"), JsonSerializer.Serialize(new { normalized, video, audio }));
        Equal(Values(test.GetProperty("normalized")), normalized, 1e-5f);
        Equal(Values(test.GetProperty("video_aggregate")), video, 1e-5f);
        Equal(Values(test.GetProperty("audio_aggregate")), audio, 1e-5f);
        if (backend.Allocator is CudaAllocator) Assert.Equal(15, encoder.CudaMappedProjections);
        if (backend.Allocator is CudaAllocator)
            Assert.Equal(CudaDevice.GetDevice(((CudaAllocator)backend.Allocator).DeviceId).ComputeCapabilityMajor >= 8 ? 15 : 0,
                encoder.CudaBf16Projections);
    }
    [Theory]
    [InlineData(0, "video")] [InlineData(1, "video")] [InlineData(2, "video")] [InlineData(3, "video")]
    [InlineData(0, "audio")] [InlineData(1, "audio")] [InlineData(2, "audio")] [InlineData(3, "audio")]
    public void LearnedRegistersGatesAndNoncausalConnectorsMatchPinnedOriginal(int index, string stream)
    {
        using var fixture = new Fixture(); using var backend = new Backend();
        using var connector = new Ltx25PromptConnector(fixture.Connectors, backend.Allocator, stream + "_embeddings_connector");
        var test = Oracle().GetProperty("cases")[index]; int count = test.GetProperty("ids").GetArrayLength();
        var output = connector.Forward(Values(test.GetProperty(stream + "_aggregate")), count,
            test.GetProperty(stream + "_connector_tokens").GetInt32(), default);
        Equal(Values(test.GetProperty(stream + "_connector")), output, 2e-5f);
        connector.ReleaseDeviceResidency();
        Equal(output, connector.Forward(Values(test.GetProperty(stream + "_aggregate")), count,
            test.GetProperty(stream + "_connector_tokens").GetInt32(), default), 1e-6f);
    }
    [Fact]
    public void EncoderReturnsAllRegisterContextsAndHonorsCancellationAndDisposal()
    {
        using var fixture = new Fixture(); using var backend = new Backend();
        var encoder = new Ltx25PromptEncoder(fixture.Encoder, fixture.Connectors, backend.Allocator);
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        Assert.Throws<OperationCanceledException>(() => encoder.Encode("cat", cancelled.Token));
        var result = encoder.Encode("cat"); Assert.Equal(1024, result.Tokens);
        Assert.Equal(1024 * 16, result.VideoContext.Length); Assert.Equal(1024 * 8, result.AudioContext.Length);
        Assert.All(result.AttentionMask, value => Assert.Equal(1, value));
        Assert.All(result.VideoContext, value => Assert.True(float.IsFinite(value)));
        Assert.All(result.AudioContext, value => Assert.True(float.IsFinite(value)));
        encoder.ReleaseDeviceResidency();
        var changed = encoder.Encode(" A cat.\nHello! ");
        using (var fresh = new Ltx25PromptEncoder(fixture.Encoder, fixture.Connectors, backend.Allocator))
        {
            var expectedChanged = fresh.Encode(" A cat.\nHello! ");
            Equal(expectedChanged.VideoContext, changed.VideoContext, 1e-6f);
            Equal(expectedChanged.AudioContext, changed.AudioContext, 1e-6f);
        }
        encoder.ReleaseDeviceResidency();
        var repeated = encoder.Encode("cat");
        Equal(result.VideoContext, repeated.VideoContext, 1e-6f);
        Equal(result.AudioContext, repeated.AudioContext, 1e-6f);
        encoder.Dispose();
        Assert.Throws<ObjectDisposedException>(() => encoder.Encode("cat"));
    }
    [Fact]
    public void Bf16RoundsTiesToEvenAndPreservesInfAndSign()
    {
        Assert.Equal(1f, Ltx25PromptMath.Bf16(BitConverter.UInt32BitsToSingle(0x3f808000)));
        Assert.Equal(BitConverter.UInt32BitsToSingle(0x3f820000), Ltx25PromptMath.Bf16(BitConverter.UInt32BitsToSingle(0x3f818000)));
        Assert.Equal(float.PositiveInfinity, Ltx25PromptMath.Bf16(float.PositiveInfinity));
        Assert.Equal(0x80000000u, BitConverter.SingleToUInt32Bits(Ltx25PromptMath.Bf16(-0f)));
    }
    private sealed class Fixture : IDisposable
    {
        public string Encoder { get; } = Path.Combine(Path.GetTempPath(), "ltx25-prompt-" + Guid.NewGuid() + ".bin");
        public string Connectors { get; } = Path.Combine(Path.GetTempPath(), "ltx25-connectors-" + Guid.NewGuid() + ".bin");
        public Fixture() { Copy("prompt-encoder.bin", Encoder); Copy("prompt-connectors.bin", Connectors); }
        private static void Copy(string name, string path) { using var source = Resource(name); using var target = File.Create(path); source.CopyTo(target); }
        public void Dispose() { File.Delete(Encoder); File.Delete(Connectors); }
    }
    private sealed class Backend : IDisposable
    {
        public IAllocator Allocator { get; } = Environment.GetEnvironmentVariable("TS_TEST_LTX25_BACKEND")?.ToLowerInvariant() switch
        { "cuda" => new CudaAllocator(), "mlx" => new MlxAllocator(), null or "" or "cpu" => new CpuAllocator(BlasEnum.DotNet), var value => throw new ArgumentException(value) };
        public void Dispose() => (Allocator as IDisposable)?.Dispose();
    }
}
