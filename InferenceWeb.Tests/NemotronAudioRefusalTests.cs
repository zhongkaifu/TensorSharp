// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// Nemotron-H (Nemotron 3 Nano Omni included) serves audio only when a companion
/// GGUF carrying the Parakeet/FastConformer tower and sound projector is loaded
/// (<see cref="NemotronAudioEncoder"/>); the public mmproj carries only the RADIO
/// vision encoder. Audio used to be decoded, warned about once on stderr, and
/// the request then generated as if no audio had been sent. Without a loaded
/// tower every entry point now refuses it with one message, before any upload is
/// written; with one, the same gates let it through. The encoder's numerics are
/// covered by NemotronAudioEncoderTests / NemotronAudioInjectorTests.
/// </summary>
[Collection("NemotronAudioCompanionEnvironment")]
public sealed class NemotronAudioRefusalTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "nemotron-audio-" + Guid.NewGuid().ToString("N"));

    public NemotronAudioRefusalTests() => Directory.CreateDirectory(_directory);

    public void Dispose() => Directory.Delete(_directory, true);

    [Theory]
    [InlineData("nemotron_h")]
    [InlineData("nemotron_h_moe")]
    [InlineData("nemotron_h_omni")]
    [InlineData("NEMOTRON_H_MOE")]
    public void EveryNemotronAliasGetsTheSameRefusal(string architecture)
    {
        Assert.Equal(NemotronModel.AudioInputUnsupportedMessage, ChatGenerationPipeline.AudioInputErrorFor(architecture));
        Assert.Contains("does not support audio input", NemotronModel.AudioInputUnsupportedMessage);
        Assert.Contains("RADIO vision", NemotronModel.AudioInputUnsupportedMessage);
        Assert.Contains("no audio tower is loaded", NemotronModel.AudioInputUnsupportedMessage);

        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Describe this", ImagePaths = new() { "photo.png" } },
            new() { Role = "assistant", Content = "A photo." },
            new() { Role = "user", Content = "And this?", AudioPaths = new() { "clip.wav" } },
        };
        Assert.Equal(NemotronModel.AudioInputUnsupportedMessage,
            ChatGenerationPipeline.UnsupportedAudioInputError(architecture, history));

        // A loaded audio tower lifts the refusal for the same history.
        Assert.Null(ChatGenerationPipeline.AudioInputErrorFor(architecture, audioEncoderLoaded: true));
        Assert.Null(ChatGenerationPipeline.UnsupportedAudioInputError(architecture, history, audioEncoderLoaded: true));

        // Image-only and text-only histories are untouched by the gate.
        history.RemoveAt(2);
        Assert.Null(ChatGenerationPipeline.UnsupportedAudioInputError(architecture, history));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("gemma4")]
    [InlineData("qwen35")]
    [InlineData("not-a-registered-architecture")]
    public void FamiliesWithAnAudioTowerOrUnknownToTheRegistryAreNotGated(string? architecture)
    {
        Assert.Null(ChatGenerationPipeline.AudioInputErrorFor(architecture));
    }

    [Fact]
    public void DeepSeek41KeepsItsOwnRefusal()
    {
        Assert.Equal(ChatGenerationPipeline.DeepSeek41AudioInputError, ChatGenerationPipeline.AudioInputErrorFor("deepseek41"));
        // V4.1 has no optional tower: nothing a caller reports lifts its refusal.
        Assert.Equal(ChatGenerationPipeline.DeepSeek41AudioInputError,
            ChatGenerationPipeline.AudioInputErrorFor("deepseek41", audioEncoderLoaded: true));
    }

    [Theory]
    [InlineData("{\"type\":\"input_audio\",\"input_audio\":{\"format\":\"wav\",\"data\":\"AQID\"}}")]
    [InlineData("{\"type\":\"input_audio\"}")]
    [InlineData("{\"type\":\"audio_url\",\"audio_url\":{\"url\":\"data:audio/wav;base64,AQID\"}}")]
    [InlineData("{\"type\":\"audio_url\",\"audio_url\":null}")]
    public void OpenAIChatRequestIsRejectedBeforeAnyUploadIsWritten(string audioPart)
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var messages = JsonDocument.Parse("""
            [{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}}]},
             {"role":"assistant","content":"An earlier image."},
             {"role":"user","content":[{"type":"text","text":"What is said here?"},
            """ + audioPart + "]}]");

        JsonException error = Assert.Throws<JsonException>(() =>
            ChatMessageParser.ParseOpenAI(messages.RootElement, uploads, architecture: "nemotron_h_moe"));

        Assert.Equal(NemotronModel.AudioInputUnsupportedMessage, error.Message);
        Assert.Empty(Directory.EnumerateFiles(_directory));
        Assert.Equal(0, uploads.UsedBytes);
    }

    [Theory]
    [InlineData("{\"type\":\"input_audio\",\"input_audio\":{\"format\":\"wav\",\"data\":\"AQID\"}}")]
    [InlineData("{\"type\":\"input_audio\"}")]
    [InlineData("{\"type\":\"audio_url\",\"audio_url\":{\"url\":\"https://example.invalid/audio.wav\"}}")]
    public void ResponsesRequestIsRejectedBeforeAnyUploadIsWritten(string audioPart)
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var input = JsonDocument.Parse("""
            [{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,AQID"}]},
             {"type":"message","role":"user","content":[
            """ + audioPart + "]}]");

        JsonException error = Assert.Throws<JsonException>(() => ChatMessageParser.ParseResponsesInput(
            input.RootElement, null, uploads, architecture: "nemotron_h_omni"));

        Assert.Equal(NemotronModel.AudioInputUnsupportedMessage, error.Message);
        Assert.Empty(Directory.EnumerateFiles(_directory));
        Assert.Equal(0, uploads.UsedBytes);
    }

    [Fact]
    public void OpenAIChatAudioIsAcceptedWhenTheAudioTowerIsLoaded()
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var messages = JsonDocument.Parse("""
            [{"role":"user","content":[{"type":"text","text":"What is said here?"},
              {"type":"input_audio","input_audio":{"format":"wav","data":"AQID"}}]}]
            """);

        var parsed = ChatMessageParser.ParseOpenAI(messages.RootElement, uploads, architecture: "nemotron_h_omni",
            audioEncoderLoaded: true);

        Assert.Equal(new byte[] { 1, 2, 3 }, File.ReadAllBytes(Assert.Single(parsed[0].AudioPaths)));
    }

    /// <summary>
    /// The gate reads the loaded companion, not a flag: a vision-only projector
    /// (the published Omni mmproj's shape) loads no tower and keeps the refusal,
    /// a companion with the full sound_encoder/sound_projection tensor set lifts
    /// it, and a companion that names the projector but lacks encoder tensors is
    /// rejected at load rather than served.
    /// </summary>
    [Fact]
    public void RefusalFollowsTheTensorsOfTheLoadedCompanion()
    {
        string previous = Environment.GetEnvironmentVariable("TS_NEMOTRON_AUDIO_MMPROJ");
        Environment.SetEnvironmentVariable("TS_NEMOTRON_AUDIO_MMPROJ", null);
        using var allocatorOwner = new AllocatorModel();
        try
        {
            NemotronModel model = allocatorOwner.Model;
            Assert.False(model.IsAudioEncoderLoaded);
            Assert.Equal(NemotronModel.AudioInputUnsupportedMessage, AudioInputSupport.UnsupportedReasonFor(model));

            string visionOnly = Path.Combine(_directory, "vision-only.gguf");
            WriteVisionOnlyProjector(visionOnly);
            model.LoadAudioEncoder(visionOnly);
            Assert.False(model.IsAudioEncoderLoaded);
            Assert.Equal(NemotronModel.AudioInputUnsupportedMessage, AudioInputSupport.UnsupportedReasonFor(model));

            using var reference = NemotronAudioEncoderTests.ReadReference(128);
            JsonElement fixture = reference.RootElement.GetProperty("fixtures")[0];
            string incomplete = Path.Combine(_directory, "incomplete.gguf");
            NemotronAudioEncoderTests.WriteCompanion(incomplete, WithoutTensor(fixture, "sound_encoder.encoder.layers.1.conv.norm.running_var"), false, 128);
            Assert.Throws<InvalidDataException>(() => model.LoadAudioEncoder(incomplete));
            Assert.False(model.IsAudioEncoderLoaded);
            Assert.Equal(NemotronModel.AudioInputUnsupportedMessage, AudioInputSupport.UnsupportedReasonFor(model));

            string companion = Path.Combine(_directory, "audio.gguf");
            NemotronAudioEncoderTests.WriteCompanion(companion, fixture, false, 128);
            model.LoadAudioEncoder(companion);
            Assert.True(model.IsAudioEncoderLoaded);
            Assert.True(AudioInputSupport.IsAudioEncoderLoaded(model));
            Assert.Null(AudioInputSupport.UnsupportedReasonFor(model));
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_NEMOTRON_AUDIO_MMPROJ", previous);
        }
    }

    [Fact]
    public void InjectorRefusesAudioWhenNoTowerIsLoaded()
    {
        using var owner = new AllocatorModel();
        using var injector = new ModelMultimodalInjector(owner.Model);
        var history = new List<ChatMessage> { new() { Role = "user", AudioPaths = new() { "clip.wav" } } };

        NotSupportedException error = Assert.Throws<NotSupportedException>(() =>
            injector.ProcessNemotronHistory(owner.Model, history, new List<int> { 1, 27, 2 }));

        Assert.Equal(NemotronModel.AudioInputUnsupportedMessage, error.Message);
    }

    /// <summary>An uninitialised NemotronModel with only what LoadAudioEncoder reads.</summary>
    private sealed class AllocatorModel : IDisposable
    {
        public NemotronModel Model { get; } = (NemotronModel)RuntimeHelpers.GetUninitializedObject(typeof(NemotronModel));

        public AllocatorModel()
        {
            // The small reference fixture projects to 6 channels.
            typeof(ModelBase).GetProperty(nameof(ModelBase.Config))!
                .SetValue(Model, new ModelConfig { Architecture = "nemotron_h_omni", HiddenSize = 6 });
            typeof(ModelBase).GetField("_allocator", BindingFlags.Instance | BindingFlags.NonPublic)!
                .SetValue(Model, new CpuAllocator(BlasEnum.DotNet));
            typeof(ModelBase).GetField("_backend", BindingFlags.Instance | BindingFlags.NonPublic)!
                .SetValue(Model, BackendType.Cpu);
        }

        public void Dispose()
        {
            (typeof(NemotronModel).GetField("_audioEncoder", BindingFlags.Instance | BindingFlags.NonPublic)!
                .GetValue(Model) as IDisposable)?.Dispose();
            GC.SuppressFinalize(Model);
        }
    }

    private static JsonElement WithoutTensor(JsonElement fixture, string name)
    {
        var node = System.Text.Json.Nodes.JsonNode.Parse(fixture.GetRawText())!.AsObject();
        var weights = node["weights"]!.AsArray();
        var match = weights.Single(w => w!["name"]!.GetValue<string>() == name);
        weights.Remove(match);
        return JsonDocument.Parse(node.ToJsonString()).RootElement.Clone();
    }

    /// <summary>A GGUF with the published Omni mmproj's layout: v.* and mm.* tensors only.</summary>
    private static void WriteVisionOnlyProjector(string path)
    {
        using var writer = new BinaryWriter(File.Create(path), Encoding.UTF8);
        void String(string value) { byte[] bytes = Encoding.UTF8.GetBytes(value); writer.Write((ulong)bytes.Length); writer.Write(bytes); }
        string[] tensors = { "v.blk.0.attn_qkv.weight", "mm.model.mlp.3.weight" };
        writer.Write(0x46554747u); writer.Write(3u); writer.Write((ulong)tensors.Length); writer.Write(2ul);
        String("general.architecture"); writer.Write(8u); String("clip");
        String("clip.has_vision_encoder"); writer.Write(7u); writer.Write(true);
        ulong offset = 0;
        foreach (string tensor in tensors)
        {
            String(tensor); writer.Write(1u); writer.Write(4ul); writer.Write(0u); writer.Write(offset);
            offset += 32;
        }
        while (writer.BaseStream.Position % 32 != 0) writer.Write((byte)0);
        for (int i = 0; i < tensors.Length * 32; i++) writer.Write((byte)0);
    }

    [Fact]
    public void ImageOnlyNemotronRequestsStillParse()
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var messages = JsonDocument.Parse("""
            [{"role":"user","content":[{"type":"text","text":"Describe"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}}]}]
            """);

        var parsed = ChatMessageParser.ParseOpenAI(messages.RootElement, uploads, architecture: "nemotron_h_moe");

        Assert.Equal(new byte[] { 1, 2, 3 }, File.ReadAllBytes(Assert.Single(parsed[0].ImagePaths)));
        Assert.Null(parsed[0].AudioPaths);
    }
}

/// <summary>
/// The fact the default refusal rests on, checked against the real files: the
/// published Omni mmproj is a vision-only projector, so loading it alone leaves
/// no audio tower and audio is refused. If a distribution ever ships the tower
/// in that file, this is the test that says the documentation must change.
/// </summary>
[Trait("Requires", "Models")]
public sealed class NemotronOmniMmprojContractTests
{
    private sealed class MmprojFactAttribute : FactAttribute
    {
        public MmprojFactAttribute()
        {
            if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_TEST_NEMOTRON_MMPROJ")))
                Skip = "Requires TS_TEST_NEMOTRON_MMPROJ: the Nemotron 3 Nano Omni mmproj GGUF.";
        }
    }

    [MmprojFact]
    public void TheOmniMmprojCarriesOnlyTheVisionTower()
    {
        using var gguf = GgufFile.OpenWithoutSiblingShards(Environment.GetEnvironmentVariable("TS_TEST_NEMOTRON_MMPROJ")!);

        Assert.Equal("clip", gguf.GetString("general.architecture"));
        Assert.Equal("nemotron_v2_vl", gguf.GetString("clip.projector_type"));
        Assert.True(gguf.GetBool("clip.has_vision_encoder"));
        Assert.False(gguf.Metadata.ContainsKey("clip.has_audio_encoder"));
        Assert.Empty(gguf.Metadata.Keys.Where(k => k.StartsWith("clip.audio", StringComparison.OrdinalIgnoreCase)));

        string[] names = gguf.Tensors.Keys.ToArray();
        Assert.NotEmpty(names);
        // Vision blocks and the nemotron_v2_vl MLP projector, nothing else.
        Assert.All(names, n => Assert.True(n.StartsWith("v.", StringComparison.Ordinal) ||
                                            n.StartsWith("mm.", StringComparison.Ordinal), n));
        Assert.Empty(names.Where(n => n.StartsWith("a.", StringComparison.Ordinal) ||
                                      n.Contains("audio", StringComparison.OrdinalIgnoreCase)));
        Assert.Contains("v.blk.0.attn_qkv.weight", names);
        Assert.Contains("mm.model.mlp.3.weight", names);
        // The tensor NemotronModel.LoadAudioEncoder looks for is absent.
        Assert.DoesNotContain("sound_projection.linear2.weight", names);
    }
}
