using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Ltx25;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>Complete tiny AV graph with independently generated official LTX
/// outputs. The default is real CPU execution, with explicit CUDA/MLX opt-ins;
/// these are component tests, not pretrained end-to-end video quality tests.</summary>
public sealed partial class Ltx25DenoiserTests
{
    private static readonly JsonElement Oracle = Load();
    public static IEnumerable<object[]> Cases => Enumerable.Range(0, Oracle.GetProperty("cases").GetArrayLength())
        .Select(index => new object[] { index });

    [Theory]
    [MemberData(nameof(Cases))]
    public void CompleteAudioVideoGraphMatchesUnchangedOfficialLtx(int index)
    {
        using var fixture = new Checkpoint();
        using var backend = new Backend();
        using var model = new Ltx25Denoiser(fixture.Path, backend.Allocator);
        JsonElement reference = Oracle.GetProperty("cases")[index];
        var output = model.Forward(Input(reference));
        Assert.Equal(2, output.ExecutedBlocks);
        Equal(Floats(reference, "videoVelocity"), output.VideoVelocity, 2e-5, "video velocity");
        Equal(Floats(reference, "audioVelocity"), output.AudioVelocity, 2e-5, "audio velocity");
    }

    [Fact]
    public void DiagnosticObserverReceivesExactForwardInputAndFiniteOracleOutput()
    {
        using var fixture = new Checkpoint();
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        using var model = new Ltx25Denoiser(fixture.Path, allocator);
        var reference = Oracle.GetProperty("cases")[0];
        var input = Input(reference);
        Ltx25DenoiserOutput? observed = null;
        int calls = 0;
        model.ForwardObserver = (actualInput, output) =>
        {
            ++calls;
            Assert.Same(input, actualInput);
            observed = output;
            Equal(Floats(reference, "videoVelocity"), output.VideoVelocity, 2e-5, "observed video");
            Equal(Floats(reference, "audioVelocity"), output.AudioVelocity, 2e-5, "observed audio");
        };
        var result = model.Forward(input);
        Assert.Equal(1, calls);
        Assert.Same(result, observed);
    }

    [Fact]
    public void DiagnosticObserverCanStopBeforeSamplerUpdateAndReleaseForwardTemporaries()
    {
        using var fixture = new Checkpoint();
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        using var model = new Ltx25Denoiser(fixture.Path, allocator);
        var reference = Oracle.GetProperty("cases")[0];
        var input = Input(reference);
        var stopped = new InvalidOperationException("diagnostic capture complete");
        model.ForwardObserver = (_, _) => throw stopped;
        Assert.Same(stopped, Assert.Throws<InvalidOperationException>(() => model.Forward(input)));
        model.ForwardObserver = null!;
        var result = model.Forward(input);
        Equal(Floats(reference, "videoVelocity"), result.VideoVelocity, 2e-5, "video after observer stop");
        Equal(Floats(reference, "audioVelocity"), result.AudioVelocity, 2e-5, "audio after observer stop");
    }

    [Fact]
    public void CachedPositionTablesTrackCallerMutationsAndResidencyEviction()
    {
        using var fixture = new Checkpoint();
        using var backend = new Backend();
        using var model = new Ltx25Denoiser(fixture.Path, backend.Allocator);
        var original = Oracle.GetProperty("cases")[0];
        var changed = Oracle.GetProperty("cases")[3];
        var input = Input(original);
        var first = model.Forward(input);
        var repeated = model.Forward(Input(original)); // equal content, distinct arrays
        Equal(first.VideoVelocity, repeated.VideoVelocity, 1e-6, "cached video");
        Equal(first.AudioVelocity, repeated.AudioVelocity, 1e-6, "cached audio");
        Floats(changed, "videoPositions").CopyTo(input.VideoPositions.Boundaries, 0);
        var shifted = model.Forward(input);
        Equal(Floats(changed, "videoVelocity"), shifted.VideoVelocity, 2e-5, "changed video coordinates");
        Equal(Floats(changed, "audioVelocity"), shifted.AudioVelocity, 2e-5, "changed cross-stream coordinates");
        Assert.True(first.VideoVelocity.Zip(shifted.VideoVelocity).Any(pair => Math.Abs(pair.First - pair.Second) > 1e-5));
        model.ReleaseDeviceResidency();
        var rebuilt = model.Forward(Input(original));
        Equal(first.VideoVelocity, rebuilt.VideoVelocity, 1e-6, "rebuilt video");
        Equal(first.AudioVelocity, rebuilt.AudioVelocity, 1e-6, "rebuilt audio");
    }

    [Fact]
    public void PartialGraphReportsExecutedDepthAndDisposedModelRejectsForward()
    {
        using var fixture = new Checkpoint();
        using var backend = new Backend();
        var model = new Ltx25Denoiser(fixture.Path, backend.Allocator);
        var input = Input(Oracle.GetProperty("cases")[0]);
        Assert.Equal(1, model.Forward(input, 1).ExecutedBlocks);
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Forward(input, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Forward(input, 3));
        model.Dispose();
        Assert.Throws<ObjectDisposedException>(() => model.Forward(input));
    }

    [MlxFact]
    public void MlxDeviceResidencyCanBeReleasedBeforeDecodeAndRebuiltForTheNextRequest()
    {
        using var fixture = new Checkpoint();
        using var allocator = new MlxAllocator();
        using var model = new Ltx25Denoiser(fixture.Path, allocator);
        var reference = Oracle.GetProperty("cases")[0];
        var before = model.Forward(Input(reference));
        model.ReleaseDeviceResidency();
        MlxBackend.ClearCache();
        var after = model.Forward(Input(reference));
        Equal(before.VideoVelocity, after.VideoVelocity, 1e-6, "MLX video after residency reload");
        Equal(before.AudioVelocity, after.AudioVelocity, 1e-6, "MLX audio after residency reload");
        Assert.Equal(before.ExecutedBlocks, after.ExecutedBlocks);
    }

    [Fact]
    public void ConfigDerivesStreamGeometryAndRejectsIncompleteLayer()
    {
        using var fixture = new Checkpoint();
        using var file = new GgufFile(fixture.Path);
        var config = Ltx25Config.FromGguf(file);
        Assert.Equal(16, config.VideoDim); Assert.Equal(8, config.AudioDim);
        Assert.Equal(2, config.VideoHeads); Assert.Equal(2, config.AudioHeads);
        Assert.Equal(3, config.VideoChannels); Assert.Equal(4, config.AudioChannels);
        Assert.False(config.VideoFfnBias); Assert.True(config.AudioFfnBias);
        Assert.True(config.KeyframeEmbedding);
        // The fixture declares the published multipliers in its GGUF config, as the
        // pinned checkpoint does; the a2v/v2a gate therefore sees sigma * 1000.
        Assert.Equal(1000f, config.TimestepScale);
        Assert.Equal(1000f, config.CrossGateTimestepScale);
        var shapes = file.Tensors.ToDictionary(pair => pair.Key, pair => pair.Value.Shape);
        shapes.Remove("transformer_blocks.1.audio_attn2.to_v.weight");
        Assert.Throws<InvalidDataException>(() => Ltx25Config.FromShapes(shapes));
    }

    [Theory]
    // The original LTXModelConfigurator.from_metadata defaults: 1000 and av_ca 1.
    [InlineData("{}", 1000f, 1f)]
    [InlineData("{\"transformer\":{}}", 1000f, 1f)]
    [InlineData("{\"transformer\":{\"timestep_scale_multiplier\":1000,\"av_ca_timestep_scale_multiplier\":1000.0}}", 1000f, 1000f)]
    [InlineData("{\"transformer\":{\"timestep_scale_multiplier\":500,\"av_ca_timestep_scale_multiplier\":250}}", 500f, 250f)]
    // No config at all: the values every published LTX 2.5 checkpoint declares.
    [InlineData(null, 1000f, 1000f)]
    public void TimestepMultipliersFollowCheckpointConfig(string json, float timestep, float crossGate)
    {
        var (actualTimestep, actualCrossGate) = Ltx25Config.Multipliers(json);
        Assert.Equal(timestep, actualTimestep);
        Assert.Equal(crossGate, actualCrossGate);
    }

    [Theory]
    [InlineData("not json")]
    [InlineData("{\"transformer\":{\"av_ca_timestep_scale_multiplier\":\"x\"}}")]
    [InlineData("{\"transformer\":{\"timestep_scale_multiplier\":0}}")]
    public void InvalidCheckpointConfigIsRejected(string json)
        => Assert.Throws<InvalidDataException>(() => Ltx25Config.Multipliers(json));

    [Fact]
    public void TimestepFeaturesMatchOfficialCosineFirstEncoding()
        => Equal(Floats(Oracle, "timestepFeatures"), Ltx25Denoiser.TimestepFeatures(Floats(Oracle, "timestepInputs")), 7e-5, "timestep feature");

    [Theory]
    [MemberData(nameof(Cases))]
    public void RotaryTablesMatchOfficialPaddedHeadSplitCoordinates(int index)
    {
        var reference = Oracle.GetProperty("cases")[index];
        var input = Input(reference);
        var video = Ltx25Geometry.BuildRope(input.VideoPositions, 16, 2, input.FrequencyPrecision);
        var audio = Ltx25Geometry.BuildRope(input.AudioPositions, 8, 2, input.FrequencyPrecision);
        Equal(Floats(reference, "videoRopeCos"), video.Cosine, 1e-6, "video cosine");
        Equal(Floats(reference, "videoRopeSin"), video.Sine, 1e-6, "video sine");
        Equal(Floats(reference, "audioRopeCos"), audio.Cosine, 1e-6, "audio cosine");
        Equal(Floats(reference, "audioRopeSin"), audio.Sine, 1e-6, "audio sine");
    }

    [Fact]
    public void RotaryTablesRejectNonfinitePhaseBeforeDeviceExecution()
        => Assert.Throws<ArgumentException>(() => Ltx25Geometry.BuildRope(new(1, 1, [float.MaxValue, float.MaxValue]), 8, 2));

    [Fact]
    public void PretrainedStreamWidthRotarySamplesMatchOfficialFrequencyPrecision()
    {
        var input = Input(Oracle.GetProperty("cases")[0]);
        foreach (var sample in Oracle.GetProperty("rotarySamples").EnumerateArray())
        {
            string stream = sample.GetProperty("stream").GetString()!;
            var rope = Ltx25Geometry.BuildRope(stream == "video" ? input.VideoPositions : input.AudioPositions,
                sample.GetProperty("dim").GetInt32(), sample.GetProperty("heads").GetInt32(),
                Enum.Parse<Ltx25FrequencyPrecision>(sample.GetProperty("precision").GetString()!));
            int[] indices = sample.GetProperty("indices").EnumerateArray().Select(value => value.GetInt32()).ToArray();
            // CPU powf implementations may differ by one Float32 ULP. At
            // theta*pi/2 ~= 15708 the frequency and phase each round at ~1e-3,
            // so the Float32 grid has a bounded 2e-3 trig tolerance. The default
            // NumPy/Float64-derived frequency grid retains the strict 1e-6 gate.
            double tolerance = sample.GetProperty("precision").GetString() == "Float32" ? 2e-3 : 1e-6;
            Equal(Floats(sample, "cosine"), indices.Select(index => rope.Cosine[index]).ToArray(), tolerance, stream + " pretrained cosine");
            Equal(Floats(sample, "sine"), indices.Select(index => rope.Sine[index]).ToArray(), tolerance, stream + " pretrained sine");
        }
    }

    private static Ltx25DenoiserInput Input(JsonElement value, bool useReferenceLora = false, CancellationToken cancellationToken = default) => new()
    {
        Video = Floats(value, "video"), Audio = Floats(value, "audio"),
        VideoTokens = value.GetProperty("videoTokens").GetInt32(), AudioTokens = value.GetProperty("audioTokens").GetInt32(),
        VideoContext = Floats(value, "videoContext"), AudioContext = Floats(value, "audioContext"),
        VideoContextTokens = value.GetProperty("videoContextTokens").GetInt32(), AudioContextTokens = value.GetProperty("audioContextTokens").GetInt32(),
        VideoPositions = new(value.GetProperty("videoTokens").GetInt32(), 3, Floats(value, "videoPositions")),
        AudioPositions = new(value.GetProperty("audioTokens").GetInt32(), 1, Floats(value, "audioPositions")),
        VideoSigma = value.GetProperty("videoSigma").GetSingle(), AudioSigma = value.GetProperty("audioSigma").GetSingle(),
        VideoTimesteps = Floats(value, "videoTimesteps"), AudioTimesteps = Floats(value, "audioTimesteps"),
        KeyframeMask = value.GetProperty("keyframeMask").ValueKind == JsonValueKind.Null ? null : Floats(value, "keyframeMask"),
        FrequencyPrecision = Enum.Parse<Ltx25FrequencyPrecision>(value.GetProperty("precision").GetString()!),
        UseReferenceLora = useReferenceLora, CancellationToken = cancellationToken,
    };
    private static float[] Floats(JsonElement value, string name) => value.GetProperty(name).EnumerateArray().Select(item => item.GetSingle()).ToArray();
    private static void Equal(float[] expected, float[] actual, double tolerance, string stage)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < actual.Length; ++i)
            Assert.True(float.IsFinite(actual[i]) && Math.Abs((double)expected[i] - actual[i]) <= tolerance,
                $"{stage}[{i}] expected {expected[i]:G9}, actual {actual[i]:G9}, tolerance {tolerance:G9}");
    }
    private static Stream Resource(string name) => Assembly.GetExecutingAssembly().GetManifestResourceStream("InferenceWeb.Tests.Fixtures.Ltx25." + name)
        ?? throw new InvalidOperationException("Missing required LTX oracle fixture: " + name);
    private static JsonElement Load()
    {
        using var resource = Resource("oracle.json");
        using var document = JsonDocument.Parse(resource);
        Assert.Equal("be6d6d0a69d11a0a399c462ed92606692e5f8697", document.RootElement.GetProperty("reference").GetProperty("revision").GetString());
        return document.RootElement.Clone();
    }
    private sealed class Checkpoint : IDisposable
    {
        public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "tensorsharp-ltx25-" + Guid.NewGuid().ToString("N") + ".gguf");
        public Checkpoint()
        {
            using var resource = Resource("denoiser.bin");
            using var target = File.Create(Path);
            resource.CopyTo(target);
            target.Flush();
            target.Position = 0;
            Assert.Equal(Oracle.GetProperty("reference").GetProperty("checkpointSha256").GetString(),
                Convert.ToHexStringLower(SHA256.HashData(target)));
        }
        public void Dispose() => File.Delete(Path);
    }
    private sealed class Backend : IDisposable
    {
        public IAllocator Allocator { get; } = Environment.GetEnvironmentVariable("TS_TEST_LTX25_BACKEND")?.ToLowerInvariant() switch
        {
            "mlx" => new MlxAllocator(), "cuda" => new CudaAllocator(),
            null or "" or "cpu" => new CpuAllocator(BlasEnum.DotNet),
            var name => throw new ArgumentException("Unsupported LTX oracle backend: " + name),
        };
        public void Dispose() => (Allocator as IDisposable)?.Dispose();
    }
}
