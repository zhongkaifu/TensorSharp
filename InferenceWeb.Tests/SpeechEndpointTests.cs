// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Models.Audio;
using TensorSharp.Models.Lfm3;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.Responses;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The OpenAI speech surfaces on a language model that speaks, without a model: a fake service stands for LFM3-Omni (its
/// speaker reports voices; its engine stream yields three 80 ms chunks and a terminal update), so the request rules, the
/// response shapes and the streaming formats are checked exactly. <c>/v1/audio/speech</c>: WAV, PCM16, chunked audio with an
/// unknown-length WAV header, SSE; the 400s (OpenAI's default voice, speed, formats, a model without speech). Chat
/// completions and Responses: the <c>modalities</c> / <c>audio</c> rules (they stop before the hosted-model guard), and
/// an assistant turn sent back by its audio id.
/// </summary>
public sealed class SpeechEndpointTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-speech-endpoints-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    // ------------------------------------------------------------------------------------------------ fakes

    private sealed class FakeSpeaker : ISpeechOutputModel
    {
        public bool Loaded { get; init; } = true;
        public bool HasSpeechOutput => Loaded;
        public string SpeechOutputUnavailableReason => Loaded ? null : "lfm3: no speech decoder is loaded; load LFM3-Omni's mmproj";
        public IReadOnlyList<string> SpeechVoices => new[] { "female", "male" };
        public int SpeechSampleRate => 24000;
        public bool IsSpeechVoice(string voice) => Lfm3SpeechPrompt.IsVoice(voice);
        public List<int> SpeechSynthesisPrompt(string text, string language, string voice) => new() { 1, 2, 3 };
        public string SpeechReplyInstruction(SpeechReplyOptions options) => "";
        public int SpeechOpeningToken(SpeechReplyMode mode) => -1;
        public SpeechReplyDriver CreateSpeechDriver(IReadOnlyList<int> promptTokens, SpeechReplyOptions options) => throw new NotSupportedException();
    }

    /// <summary>Three chunks of 80 ms (a ramp), with text pieces between them for a spoken chat reply.</summary>
    internal static float[] Chunk(int index) => Enumerable.Range(0, 1920).Select(i => (float)Math.Sin(i * 0.01 + index) * 0.5f).ToArray();

    private sealed class SpeakingService : ModelService
    {
        public FakeSpeaker Speaker { get; init; } = new();
        public SpeechSynthesisOptions LastSynthesis;
        public SpeechReplyOptions LastReply;
        public List<ChatMessage> LastMessages;
        public override string Architecture => "lfm3";
        public override ISpeechOutputModel SpeechOutputModel => Speaker;

        internal override async IAsyncEnumerable<ChatStreamUpdate> SpeechSynthesisStreamAsync(string text, SpeechSynthesisOptions options,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            LastSynthesis = options;
            for (int c = 0; c < 3; c++)
            {
                await Task.Yield();
                yield return ChatStreamUpdate.AudioChunk(Chunk(c), 24000);
            }
            yield return new ChatStreamUpdate("", true, 3, 3, 0, 0, 0, 0, "eos") { SpeechFrames = new[] { new int[12] }, AudioSampleRate = 24000 };
        }

        internal override async IAsyncEnumerable<ChatStreamUpdate> ChatStreamWithSpeechAsync(List<ChatMessage> history, int maxTokens,
            [EnumeratorCancellation] CancellationToken cancellationToken, TensorSharp.Runtime.SamplingConfig samplingConfig, bool enableThinking,
            SpeechReplyOptions speech, TensorSharp.Models.Lora.TextLoraAdapter lora = null)
        {
            LastReply = speech;
            LastMessages = history;
            string[] words = { "Hello", ", Anna", "!" };
            for (int c = 0; c < 3; c++)
            {
                await Task.Yield();
                yield return ChatStreamUpdate.Text(words[c]);
                yield return ChatStreamUpdate.AudioChunk(Chunk(c), 24000);
            }
            yield return new ChatStreamUpdate("", true, 5, 9, 0, 0, 0, 0, "eos") { SpeechFrames = new[] { new int[12], new int[12] }, AudioSampleRate = 24000 };
        }
    }

    private sealed class TextOnlyService : ModelService { public override string Architecture => "qwen35"; }

    private static async Task<(int Status, byte[] Body, string ContentType)> Speech(ModelService service, string body)
    {
        var context = new DefaultHttpContext();
        context.RequestServices = new ServiceCollection().AddLogging().BuildServiceProvider();
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(body));
        context.Request.ContentType = "application/json";
        context.Response.Body = new MemoryStream();
        IResult result = await new AudioAdapter(service, NullLoggerFactory.Instance).SpeechAsync(context.Request);
        await result.ExecuteAsync(context);
        return (context.Response.StatusCode, ((MemoryStream)context.Response.Body).ToArray(), context.Response.ContentType);
    }

    private static string Message(byte[] body)
    {
        using var doc = JsonDocument.Parse(body);
        return doc.RootElement.GetProperty("error").GetProperty("message").GetString();
    }

    private static byte[] Pcm(params int[] chunks) => chunks.SelectMany(c => Lfm3SpeechAudio.Pcm16(Chunk(c))).ToArray();

    // ------------------------------------------------------------------------------------------------ /v1/audio/speech

    [Fact]
    public async Task Speech_Wav_IsTheWholeReplyAsOneWavFile()
    {
        var service = new SpeakingService();
        var (status, body, type) = await Speech(service, "{\"input\":\"Hello\",\"voice\":\"Male\",\"seed\":1234,\"temperature\":0.5,\"max_frames\":40}");
        Assert.Equal(200, status);
        Assert.Equal("audio/wav", type);
        Assert.Equal(Lfm3SpeechAudio.Wav(Enumerable.Range(0, 3).SelectMany(Chunk).ToArray(), 24000), body);
        Assert.Equal(1234u, service.LastSynthesis.Seed);
        Assert.Equal(0.5f, service.LastSynthesis.Temperature);
        Assert.Equal(40, service.LastSynthesis.MaxFrames);
        Assert.Equal("Male", service.LastSynthesis.Voice);
    }

    [Theory]
    [InlineData("pcm")]
    [InlineData("pcm16")]
    public async Task Speech_Pcm_IsRawPcm16(string format)
    {
        var (status, body, type) = await Speech(new SpeakingService(), $"{{\"input\":\"Hello\",\"response_format\":\"{format}\"}}");
        Assert.Equal(200, status);
        Assert.Equal("audio/pcm", type);
        Assert.Equal(Pcm(0, 1, 2), body);
    }

    [Theory]
    [InlineData("\"stream\":true")]
    [InlineData("\"stream_format\":\"audio\"")]
    public async Task Speech_StreamedWav_IsAnUnknownLengthHeaderThenThePcm(string streaming)
    {
        var (status, body, type) = await Speech(new SpeakingService(), $"{{\"input\":\"Hello\",{streaming}}}");
        Assert.Equal(200, status);
        Assert.Equal("audio/wav", type);
        Assert.Equal(Lfm3SpeechAudio.StreamingWavHeader(24000), body[..44]);
        Assert.Equal(0xFFFFFFFFu, BinaryPrimitives.ReadUInt32LittleEndian(body.AsSpan(4)));
        Assert.Equal(0xFFFFFFFFu, BinaryPrimitives.ReadUInt32LittleEndian(body.AsSpan(40)));
        Assert.Equal(Pcm(0, 1, 2), body[44..]);
    }

    [Fact]
    public async Task Speech_StreamedPcm_IsThePcmAlone()
    {
        var (status, body, _) = await Speech(new SpeakingService(), "{\"input\":\"Hello\",\"stream\":true,\"response_format\":\"pcm\"}");
        Assert.Equal(200, status);
        Assert.Equal(Pcm(0, 1, 2), body);
    }

    [Fact]
    public async Task Speech_Sse_SendsDeltasThenDoneWithUsage()
    {
        var (status, body, type) = await Speech(new SpeakingService(), "{\"input\":\"Hello\",\"stream_format\":\"sse\",\"response_format\":\"pcm\"}");
        Assert.Equal(200, status);
        Assert.Equal("text/event-stream", type);
        var events = Encoding.UTF8.GetString(body).Split("\n\n", StringSplitOptions.RemoveEmptyEntries)
            .Select(e => JsonDocument.Parse(e["data: ".Length..]).RootElement).ToList();
        Assert.Equal(4, events.Count);
        for (int c = 0; c < 3; c++)
        {
            Assert.Equal("speech.audio.delta", events[c].GetProperty("type").GetString());
            Assert.Equal(Pcm(c), Convert.FromBase64String(events[c].GetProperty("audio").GetString()));
        }
        Assert.Equal("speech.audio.done", events[3].GetProperty("type").GetString());
        Assert.Equal(3, events[3].GetProperty("usage").GetProperty("output_tokens").GetInt32());
    }

    [Theory]
    [InlineData("{\"input\":\"Hello\",\"voice\":\"alloy\"}", "alloy")]
    [InlineData("{\"input\":\"Hello\",\"voice\":\"alloy\"}", "female, male")]
    [InlineData("{\"input\":\"Hello\",\"speed\":1.5}", "speaking-rate")]
    [InlineData("{\"input\":\"Hello\",\"response_format\":\"mp3\"}", "wav, pcm or pcm16")]
    [InlineData("{\"input\":\"Hello\",\"stream_format\":\"sse\"}", "PCM16")]
    [InlineData("{\"input\":\"Hello\",\"stream_format\":\"opus\"}", "stream_format")]
    [InlineData("{\"input\":\"\"}", "input is required")]
    [InlineData("{\"input\":\"Hello\",\"seed\":-1}", "seed")]
    [InlineData("{\"input\":\"Hello\",\"temperature\":-0.1}", "temperature")]
    [InlineData("{\"input\":\"Hello\",\"max_frames\":0}", "max_frames")]
    [InlineData("{\"input\":\"Hello\",\"voice\":3}", "voice must be a string")]
    public async Task Speech_RefusesWhatTheModelCannotDo_With400(string body, string expected)
    {
        var (status, response, _) = await Speech(new SpeakingService(), body);
        Assert.Equal(400, status);
        Assert.Contains(expected, Message(response), StringComparison.Ordinal);
    }

    [Fact]
    public async Task Speech_ASpeakerWithoutItsDecoder_Is400NamingWhatToLoad()
    {
        var (status, response, _) = await Speech(new SpeakingService { Speaker = new FakeSpeaker { Loaded = false } }, "{\"input\":\"Hello\"}");
        Assert.Equal(400, status);
        Assert.Contains("mmproj", Message(response), StringComparison.Ordinal);
    }

    [Fact]
    public async Task Speech_ATextModel_Is400()
    {
        var (status, response, _) = await Speech(new TextOnlyService(), "{\"input\":\"Hello\"}");
        Assert.Equal(400, status);
        Assert.Contains("not a speech synthesis model", Message(response), StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------------------------------------ chat and Responses rules

    private async Task<DefaultHttpContext> Chat(ModelService service, string request)
    {
        var context = new DefaultHttpContext();
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(request));
        context.Request.ContentType = "application/json";
        context.Response.Body = new MemoryStream();
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        var options = ServerOptionsBuilder.Build(new[] { "--model", Path.Combine(_root, "hosted.gguf"), "--no-skills" }, _root);
        await new OpenAIChatAdapter(service, options, new UploadStoragePolicy(Path.Combine(_root, "uploads")), registry,
            null, new SessionWorkspaceManager(Path.Combine(_root, "workspaces")), NullLoggerFactory.Instance).ChatCompletionsAsync(context);
        return context;
    }

    private async Task<DefaultHttpContext> Responses(ModelService service, string request)
    {
        var context = new DefaultHttpContext();
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(request));
        context.Request.ContentType = "application/json";
        context.Response.Body = new MemoryStream();
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        var options = ServerOptionsBuilder.Build(new[] { "--model", Path.Combine(_root, "hosted.gguf"), "--no-skills" }, _root);
        using var store = new InMemoryResponsesStore();
        await new OpenAIResponsesAdapter(service, options, new UploadStoragePolicy(Path.Combine(_root, "uploads")), registry, null,
            new SessionWorkspaceManager(Path.Combine(_root, "workspaces")), NullLoggerFactory.Instance, store).CreateResponseAsync(context);
        return context;
    }

    private static string Body(DefaultHttpContext context) => Encoding.UTF8.GetString(((MemoryStream)context.Response.Body).ToArray());

    private const string Ask = "\"model\":\"hosted.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]";

    [Theory]
    [InlineData(",\"modalities\":[\"text\",\"audio\"],\"stream\":true,\"audio\":{\"voice\":\"female\",\"format\":\"wav\"}", "pcm16")]
    [InlineData(",\"modalities\":[\"text\",\"audio\"],\"audio\":{\"voice\":\"alloy\",\"format\":\"wav\"}", "alloy")]
    [InlineData(",\"modalities\":[\"text\",\"audio\"],\"audio\":{\"voice\":\"female\",\"format\":\"mp3\"}", "audio.format")]
    [InlineData(",\"modalities\":[\"text\",\"video\"]", "modalities")]
    [InlineData(",\"modalities\":\"audio\"", "modalities must be an array")]
    [InlineData(",\"audio\":{\"voice\":\"female\"}", "modalities is not")]
    [InlineData(",\"modalities\":[\"text\"],\"audio\":{\"voice\":\"female\"}", "does not ask for")]
    [InlineData(",\"modalities\":[\"audio\"],\"audio\":{\"seed\":-3}", "audio.seed")]
    [InlineData(",\"modalities\":[\"audio\"],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"p\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]", "tools")]
    [InlineData(",\"modalities\":[\"audio\"],\"response_format\":{\"type\":\"json_object\"}", "response_format")]
    [InlineData(",\"modalities\":[\"audio\"],\"think\":true", "thinking")]
    public async Task ChatAudio_RefusesWhatItCannotServe_With400(string extra, string expected)
    {
        DefaultHttpContext context = await Chat(new SpeakingService(), "{" + Ask + extra + "}");
        Assert.Equal(400, context.Response.StatusCode);
        Assert.Contains(expected, Body(context), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ChatAudio_OnAModelThatDoesNotSpeak_Is400(bool speakerWithoutDecoder)
    {
        ModelService service = speakerWithoutDecoder ? new SpeakingService { Speaker = new FakeSpeaker { Loaded = false } } : new TextOnlyService();
        DefaultHttpContext context = await Chat(service, "{" + Ask + ",\"modalities\":[\"text\",\"audio\"],\"audio\":{\"voice\":\"female\",\"format\":\"wav\"}}");
        Assert.Equal(400, context.Response.StatusCode);
        Assert.Contains(speakerWithoutDecoder ? "mmproj" : "no speech output", Body(context), StringComparison.Ordinal);
    }

    [Fact]
    public async Task ChatAudio_AcceptedRequestsReachTheHostedModelGuard()
    {
        // A valid spoken request passes every rule and stops where a model is needed.
        DefaultHttpContext context = await Chat(new SpeakingService(), "{\"model\":\"not-hosted.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]," +
            "\"modalities\":[\"text\",\"audio\"],\"audio\":{\"voice\":\"MALE\",\"format\":\"pcm16\"}}");
        Assert.Equal(404, context.Response.StatusCode);
    }

    [Fact]
    public async Task ChatAudio_AnUnknownAudioIdInTheHistory_Is400()
    {
        DefaultHttpContext context = await Chat(new SpeakingService(), "{\"model\":\"not-hosted.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}," +
            "{\"role\":\"assistant\",\"content\":null,\"audio\":{\"id\":\"audio_unknown\"}},{\"role\":\"user\",\"content\":\"again\"}]}");
        Assert.Equal(400, context.Response.StatusCode);
        Assert.Contains("audio_unknown", Body(context), StringComparison.Ordinal);
    }

    [Fact]
    public void AStoredReply_RendersByItsTranscript()
    {
        var entry = OpenAIChatAdapter.SpeechReplies.Add("Hello, Anna!", new[] { new int[12] });
        using var doc = JsonDocument.Parse("[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":null,\"audio\":{\"id\":\"" + entry.Id + "\"}}]");
        List<ChatMessage> messages = TensorSharp.Server.RequestParsers.ChatMessageParser.ParseOpenAI(doc.RootElement,
            new UploadStoragePolicy(Path.Combine(_root, "uploads")), null, "lfm3", false, OpenAIChatAdapter.SpeechReplies.TranscriptOf);
        Assert.Equal("Hello, Anna!", messages[1].Content);
        Assert.True(entry.ExpiresAt > DateTimeOffset.UtcNow.AddMinutes(59));
    }

    [Theory]
    [InlineData(",\"modalities\":[\"text\",\"audio\"],\"audio\":{\"voice\":\"alloy\"}", "alloy")]
    [InlineData(",\"modalities\":[\"audio\"],\"stream\":true,\"audio\":{\"format\":\"wav\"}", "pcm16")]
    [InlineData(",\"modalities\":[\"audio\"],\"reasoning\":{}", "reasoning")]
    public async Task ResponsesAudio_RefusesWhatItCannotServe_With400(string extra, string expected)
    {
        DefaultHttpContext context = await Responses(new SpeakingService(), "{\"model\":\"hosted.gguf\",\"input\":\"hi\",\"store\":false" + extra + "}");
        Assert.Equal(400, context.Response.StatusCode);
        Assert.Contains(expected, Body(context), StringComparison.Ordinal);
    }
}
