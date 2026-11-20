// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Models;
using TensorSharp.Models.Audio;
using TensorSharp.Models.Lfm3;
using TensorSharp.Models.Lora;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// A spoken chat completion with the client's tools (or reasoning) is the text completion with its content spoken: the
/// tool calls go out as a text completion's, the content as the audio transcript, and the speech of the content beside
/// them. Without a model: a loaded probe whose completion is scripted (the text before a call, then the call) and whose
/// TTS records what it says.
/// </summary>
public sealed class OpenAISpokenCompletionTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-openai-spoken-" + Guid.NewGuid().ToString("N"));

    public OpenAISpokenCompletionTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best effort */ }
    }

    private sealed class OmniSpeaker : ISpeechOutputModel
    {
        public bool HasSpeechOutput => true;
        public string SpeechOutputUnavailableReason => null;
        public IReadOnlyList<string> SpeechVoices => new[] { "female", "male" };
        public int SpeechSampleRate => 24000;
        public bool IsSpeechVoice(string voice) => Lfm3SpeechPrompt.IsVoice(voice);
        public List<int> SpeechSynthesisPrompt(string text, string language, string voice) => new() { 1 };
        public string SpeechReplyInstruction(SpeechReplyOptions options) => "";
        public int SpeechOpeningToken(SpeechReplyMode mode) => -1;
        public SpeechReplyPath SpeechReplyPathFor(string language)
            => Lfm3SpeechPrompt.SpeaksRepliesIn(language) ? SpeechReplyPath.Direct : SpeechReplyPath.Synthesized;
        public SpeechReplyDriver CreateSpeechDriver(IReadOnlyList<int> promptTokens, SpeechReplyOptions options) => throw new NotSupportedException();
    }

    private sealed class ProbeModel(string path) : ModelBase(path, BackendType.Cpu)
    {
        protected override float[] ForwardCore(int[] tokens) => [];
        protected override void ResetKVCacheCore() { }
    }

    private sealed class ToolCallingService : ModelService
    {
        public ToolCallingService() : base(NullLogger<ModelService>.Instance, (path, _, _, _) => new ProbeModel(path)) { }

        public readonly List<string> Said = new();
        public List<ToolFunction> Tools;
        public int DirectCalls;
        public override string Architecture => "lfm3";
        public override ISpeechOutputModel SpeechOutputModel { get; } = new OmniSpeaker();

        internal override async IAsyncEnumerable<ChatStreamUpdate> ChatStreamWithSkillsAsync(ChatSession session,
            List<ChatMessage> history, int maxTokens, [EnumeratorCancellation] CancellationToken cancellationToken,
            SamplingConfig samplingConfig, List<ToolFunction> tools, bool enableThinking, SkillRequestPlan skills,
            ILogger logger = null, int? thinkingTokenLimit = null, TextLoraAdapter lora = null)
        {
            Tools = tools;
            await Task.Yield();
            yield return ChatStreamUpdate.Parsed(string.Empty, "The user wants the weather.", null);
            yield return ChatStreamUpdate.Parsed("Let me check the weather", null, null);
            yield return ChatStreamUpdate.Parsed(string.Empty, null, new[]
            {
                new ToolCall { Name = "get_weather", Arguments = new Dictionary<string, object> { ["city"] = "Paris" } },
            });
            yield return new ChatStreamUpdate("", true, 50, 12, 40, 0, 0, 0, "tool_calls");
        }

        internal override async IAsyncEnumerable<ChatStreamUpdate> SpeechSynthesisStreamAsync(string text,
            SpeechSynthesisOptions options, [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            lock (Said) Said.Add(text);
            await Task.Yield();
            yield return ChatStreamUpdate.AudioChunk(SpeechEndpointTests.Chunk(0), 24000);
            yield return new ChatStreamUpdate("", true, 3, 1, 0, 0, 0, 0, "eos") { SpeechFrames = new[] { new int[12] }, AudioSampleRate = 24000 };
        }

        internal override async IAsyncEnumerable<ChatStreamUpdate> ChatStreamWithSpeechAsync(List<ChatMessage> history, int maxTokens,
            [EnumeratorCancellation] CancellationToken cancellationToken, SamplingConfig samplingConfig, bool enableThinking,
            SpeechReplyOptions speech, TextLoraAdapter lora = null)
        {
            DirectCalls++;
            await Task.Yield();
            yield return new ChatStreamUpdate("", true, 1, 1, 0, 0, 0, 0, "eos");
        }
    }

    private async Task<(ToolCallingService Service, DefaultHttpContext Context)> Chat(string extra)
    {
        string model = Path.Combine(_root, "hosted.gguf");
        if (!File.Exists(model)) BackendFailureWarmupTests.WriteProbeGguf(model);
        var service = new ToolCallingService();
        service.LoadModel(model, null, "cpu");
        var context = new DefaultHttpContext();
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes("{\"model\":\"hosted.gguf\",\"messages\":[{\"role\":\"user\"," +
            "\"content\":\"What is the weather in Paris?\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\"," +
            "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}}],\"modalities\":[\"text\",\"audio\"]" + extra + "}"));
        context.Request.ContentType = "application/json";
        context.Response.Body = new MemoryStream();
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        var options = ServerOptionsBuilder.Build(new[] { "--model", model, "--no-skills" }, _root);
        await new OpenAIChatAdapter(service, options, new UploadStoragePolicy(Path.Combine(_root, "uploads")), registry,
            null, new SessionWorkspaceManager(Path.Combine(_root, "workspaces")), NullLoggerFactory.Instance).ChatCompletionsAsync(context);
        return (service, context);
    }

    private static string Body(DefaultHttpContext context) => Encoding.UTF8.GetString(((MemoryStream)context.Response.Body).ToArray());

    [Fact]
    public async Task AStreamedSpokenCompletionWithToolsSendsTheCallsAndSpeaksTheContent()
    {
        var (service, context) = await Chat(",\"stream\":true,\"audio\":{\"voice\":\"female\",\"format\":\"pcm16\"}");
        Assert.Equal(200, context.Response.StatusCode);
        List<JsonElement> chunks = Body(context).Split('\n').Where(l => l.StartsWith("data: {", StringComparison.Ordinal))
            .Select(l => JsonSerializer.Deserialize<JsonElement>(l[6..])).ToList();
        JsonElement Delta(JsonElement c) => c.GetProperty("choices")[0].GetProperty("delta");
        bool Audio(JsonElement c, string key) => Delta(c).TryGetProperty("audio", out JsonElement a) && a.TryGetProperty(key, out _);

        // The text completion ran with the client's tool, and no driven reply.
        Assert.Equal(0, service.DirectCalls);
        Assert.Contains(service.Tools, t => t.Name == "get_weather");
        // id first; the content as the transcript; the reasoning and the call as a text completion's; the speech; expiry.
        Assert.True(Audio(chunks[0], "id"));
        Assert.Equal("Let me check the weather", string.Concat(chunks.Where(c => Audio(c, "transcript"))
            .Select(c => Delta(c).GetProperty("audio").GetProperty("transcript").GetString())));
        Assert.DoesNotContain(chunks, c => Delta(c).TryGetProperty("content", out JsonElement t) && t.ValueKind == JsonValueKind.String && t.GetString().Length > 0);
        Assert.Contains(chunks, c => Delta(c).TryGetProperty("reasoning_content", out _));
        int call = chunks.FindIndex(c => Delta(c).TryGetProperty("tool_calls", out _));
        Assert.True(call > 0);
        Assert.Equal("get_weather", Delta(chunks[call]).GetProperty("tool_calls")[0].GetProperty("function").GetProperty("name").GetString());
        JsonElement data = Assert.Single(chunks, c => Audio(c, "data"));
        Assert.Equal(Lfm3SpeechAudio.Pcm16(SpeechEndpointTests.Chunk(0)),
            Convert.FromBase64String(Delta(data).GetProperty("audio").GetProperty("data").GetString()));
        int expires = chunks.FindIndex(c => Audio(c, "expires_at"));
        Assert.True(expires > chunks.IndexOf(data));
        Assert.Equal("tool_calls", chunks[^1].GetProperty("choices")[0].GetProperty("finish_reason").GetString());
        // Spoken: the content, never the reasoning or the call.
        Assert.Equal(new[] { "Let me check the weather" }, service.Said);
    }

    [Fact]
    public async Task AWholeSpokenCompletionWithToolsCarriesTheCallsAndTheContentsAudio()
    {
        var (service, context) = await Chat(",\"audio\":{\"voice\":\"female\",\"format\":\"wav\"}");
        Assert.Equal(200, context.Response.StatusCode);
        JsonElement message = JsonSerializer.Deserialize<JsonElement>(Body(context)).GetProperty("choices")[0].GetProperty("message");
        Assert.Equal("get_weather", message.GetProperty("tool_calls")[0].GetProperty("function").GetProperty("name").GetString());
        JsonElement audio = message.GetProperty("audio");
        Assert.Equal("Let me check the weather", audio.GetProperty("transcript").GetString());
        Assert.Equal(Lfm3SpeechAudio.Wav(SpeechEndpointTests.Chunk(0), 24000), Convert.FromBase64String(audio.GetProperty("data").GetString()));
        Assert.StartsWith("audio_", audio.GetProperty("id").GetString(), StringComparison.Ordinal);
        Assert.True(OpenAIChatAdapter.SpeechReplies.TryGet(audio.GetProperty("id").GetString(), out var entry));
        Assert.Null(entry.Turn);
        Assert.Equal(new[] { "Let me check the weather" }, service.Said);
    }
}
