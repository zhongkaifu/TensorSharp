// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// Nemotron-H (Nemotron 3 Nano Omni included) has an audio front-end but no
/// audio tower: the public mmproj carries only the RADIO vision encoder, so a
/// clip cannot be turned into embeddings. Audio used to be decoded, warned
/// about once on stderr, and the request then generated as if no audio had
/// been sent. Every entry point now refuses it with one message, before any
/// upload is written.
/// </summary>
public sealed class NemotronAudioInputTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "nemotron-audio-" + Guid.NewGuid().ToString("N"));

    public NemotronAudioInputTests() => Directory.CreateDirectory(_directory);

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
        Assert.Contains("RADIO vision tower", NemotronModel.AudioInputUnsupportedMessage);

        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Describe this", ImagePaths = new() { "photo.png" } },
            new() { Role = "assistant", Content = "A photo." },
            new() { Role = "user", Content = "And this?", AudioPaths = new() { "clip.wav" } },
        };
        Assert.Equal(NemotronModel.AudioInputUnsupportedMessage,
            ChatGenerationPipeline.UnsupportedAudioInputError(architecture, history));

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
/// The fact the refusal rests on, checked against the real files: the Omni
/// mmproj is a vision-only projector. If a distribution ever ships the audio
/// tower, this is the test that says the refusal must go.
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
    }
}
