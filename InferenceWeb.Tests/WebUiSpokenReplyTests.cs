// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Chat;
using TensorSharp.Models;
using TensorSharp.Models.Audio;
using TensorSharp.Models.Lfm3;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// Spoken web chat replies (<c>/api/chat</c> with <c>speech</c>) without a model: a service whose loaded model is a
/// probe, whose speaker stands for LFM3-Omni, whose text turn yields three pieces of one sentence and whose TTS yields
/// three 80 ms chunks a piece. A web chat turn is its text turn with its answer spoken. Checked: the frames in order,
/// the done frame's audio id and WAV (written, valid, the whole reply), the generation call (the text turn's, with no
/// speech line), the voice and language the TTS is given, the refusals, and the next turn rendering the spoken turn as
/// its text. Spoken turns that run reasoning, tools or skills, and the prompt prefix across spoken and text turns:
/// <see cref="WebUiSpokenAgentTurnTests"/>.
/// </summary>
public sealed class WebUiSpokenReplyTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-webui-speech-" + Guid.NewGuid().ToString("N"));

    public WebUiSpokenReplyTests()
    {
        Directory.CreateDirectory(_root);
        Directory.CreateDirectory(Uploads);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best effort */ }
    }

    private string Uploads => Path.Combine(_root, "uploads");

    // ------------------------------------------------------------------------------------------------ fakes

    private sealed class Speaker : ISpeechOutputModel
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

    private sealed class ProbeModel(string path) : ModelBase(path, BackendType.Cpu)
    {
        protected override float[] ForwardCore(int[] tokens) => [];
        protected override void ResetKVCacheCore() { }
    }

    internal static readonly int[][] FrameCodes = { Enumerable.Repeat(5, 12).ToArray(), Enumerable.Repeat(2016, 12).ToArray() };
    internal static readonly SpokenTurnRecord Turn = new(new[] { 241, 60, 243, 61, 243 }, FrameCodes, "", "<think></think>");
    private static readonly string[] Words = { "Hello", ", Anna", "!" };

    /// <summary>
    /// The text turn (three pieces of "Hello, Anna!", then its terminal update) and the TTS (three 80 ms chunks a
    /// piece). <see cref="ChatStreamWithSpeechAsync"/> (a driven spoken reply) is never what a web chat turn runs.
    /// </summary>
    private sealed class SpeakingService : ModelService
    {
        public SpeakingService() : base(NullLogger<ModelService>.Instance, (path, _, _, _) => new ProbeModel(path)) { }

        public ISpeechOutputModel Speaker { get; init; } = new Speaker();
        public string Arch { get; init; } = "lfm3";
        /// <summary>Throw a cancellation after this many text pieces (-1: never).</summary>
        public int CancelAfter { get; init; } = -1;
        public List<(ChatSession Session, List<ChatMessage> History, bool Think)> Calls { get; } = new();
        public List<(string Text, SpeechSynthesisOptions Options)> Spoken { get; } = new();
        public int DrivenReplies;
        public override string Architecture => Arch;
        public override ISpeechOutputModel SpeechOutputModel => Speaker;

        internal override async IAsyncEnumerable<ChatStreamUpdate> ChatStreamWithSkillsAsync(ChatSession session,
            List<ChatMessage> history, int maxTokens, [EnumeratorCancellation] CancellationToken cancellationToken,
            SamplingConfig samplingConfig, List<ToolFunction> tools, bool enableThinking, TensorSharp.Server.Skills.SkillRequestPlan skills,
            Microsoft.Extensions.Logging.ILogger logger = null, int? thinkingTokenLimit = null, TensorSharp.Models.Lora.TextLoraAdapter lora = null)
        {
            Calls.Add((session, history, enableThinking));
            for (int c = 0; c < 3; c++)
            {
                await Task.Yield();
                if (c == CancelAfter) throw new OperationCanceledException();
                yield return ChatStreamUpdate.Text(Words[c]);
            }
            yield return new ChatStreamUpdate("", true, 40, 9, 31, 0, 0, 1_000_000, "eos");
        }

        internal override async IAsyncEnumerable<ChatStreamUpdate> SpeechSynthesisStreamAsync(string text, SpeechSynthesisOptions options,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            lock (Spoken) Spoken.Add((text, options));
            for (int c = 0; c < 3; c++)
            {
                await Task.Yield();
                yield return ChatStreamUpdate.AudioChunk(SpeechEndpointTests.Chunk(c), 24000);
            }
            yield return new ChatStreamUpdate("", true, 3, 3, 0, 0, 0, 0, "eos") { SpeechFrames = FrameCodes, AudioSampleRate = 24000 };
        }

        internal override IAsyncEnumerable<ChatStreamUpdate> ChatStreamWithSpeechAsync(ChatSession session,
            List<ChatMessage> history, int maxTokens, CancellationToken cancellationToken,
            SamplingConfig samplingConfig, bool enableThinking, SpeechReplyOptions speech, TensorSharp.Models.Lora.TextLoraAdapter lora = null)
        {
            DrivenReplies++;
            throw new InvalidOperationException("a web chat turn is its text turn, spoken");
        }
    }

    private sealed class TextService : ModelService
    {
        public TextService() : base(NullLogger<ModelService>.Instance, (path, _, _, _) => new ProbeModel(path)) { }
        public override string Architecture => "qwen35";
    }

    private (WebUiChatService Chat, ChatSession Session) Build(ModelService model, bool skills = false)
    {
        string modelPath = Path.Combine(_root, "probe.gguf");
        if (!File.Exists(modelPath)) BackendFailureWarmupTests.WriteProbeGguf(modelPath);
        model.LoadModel(modelPath, null, "cpu");
        var sessions = new SessionManager();
        ChatSession session = sessions.CreateSession();
        var options = new ServerHostingOptions(
            startupModelPath: modelPath, startupMmProjPath: null, defaultBackend: "cpu",
            supportedBackends: new[] { new BackendOption("cpu", "CPU") }, defaultMaxTokens: 100, maxTokensPinned: false,
            defaultVideoFrames: 0, defaultVideoFps: 0, defaultVideoWidth: 0, defaultVideoHeight: 0, defaultVideoSteps: 0,
            defaultVideoMode: null, uploadDirectory: Uploads, logDirectory: Path.Combine(_root, "logs"),
            fileLoggingEnabled: false, samplingDefaults: null, skillsEnabled: skills, skillsAllowScripts: false,
            skillsSandbox: SkillSandboxMode.Required, skillsAllowNetwork: false,
            // A host that offers no delegation: a plain spoken turn here needs nothing agentic, so it is one driven reply.
            multiAgent: new TensorSharp.AgentHost.Agents.MultiAgentOptions { Enabled = false });
        var chat = new WebUiChatService(model, sessions, options, new UploadStoragePolicy(Uploads),
            new SkillRegistry(new SkillRegistryOptions { Roots = [] }), codeRunner: null, workspaces: null,
            codeArtifacts: null, NullLoggerFactory.Instance);
        return (chat, session);
    }

    private static JsonElement Body(object body) => JsonSerializer.SerializeToElement(body);

    private static async Task<List<JsonElement>> Frames(WebUiChatService chat, JsonElement body)
    {
        var frames = new List<JsonElement>();
        await foreach (object frame in chat.ChatStreamAsync(body, CancellationToken.None))
            frames.Add(JsonSerializer.SerializeToElement(frame));
        return frames;
    }

    private static string Text(JsonElement frame, string name)
        => frame.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;

    private static object[] FirstTurn() => new object[]
    {
        new { role = "user", content = "Say hello to Anna." },
    };

    // ------------------------------------------------------------------------------------------------ frames

    [Fact]
    public async Task ASpokenTurnStreamsItsTextThenItsSpeechThenItsWav()
    {
        var model = new SpeakingService();
        var (chat, session) = Build(model);

        List<JsonElement> frames = await Frames(chat, Body(new
        {
            sessionId = session.Id, messages = FirstTurn(), speech = true,
        }));

        // token x3 (the text turn's), audio x3 (the one sentence, spoken once complete), done.
        Assert.Equal(7, frames.Count);
        for (int c = 0; c < 3; c++)
        {
            Assert.Equal(Words[c], Text(frames[c], "token"));
            JsonElement audio = frames[3 + c];
            Assert.Equal(Lfm3SpeechAudio.Pcm16(SpeechEndpointTests.Chunk(c)), Convert.FromBase64String(Text(audio, "audio")));
            Assert.Equal(24000, audio.GetProperty("sampleRate").GetInt32());
        }

        JsonElement done = frames[^1];
        Assert.True(done.GetProperty("done").GetBoolean());
        Assert.Equal(JsonValueKind.Null, done.GetProperty("error").ValueKind);
        Assert.False(done.GetProperty("aborted").GetBoolean());
        Assert.Equal(40, done.GetProperty("promptTokens").GetInt32());
        Assert.Equal(31, done.GetProperty("kvReusedTokens").GetInt32());
        Assert.StartsWith("audio_", Text(done, "audioId"), StringComparison.Ordinal);
        Assert.Equal(3 * 1920 / 24000.0, done.GetProperty("audioSeconds").GetDouble(), 6);

        // The whole reply as one WAV under the uploads, served as /uploads/<name>.
        string url = Text(done, "audioUrl");
        Assert.StartsWith("/uploads/speech-", url, StringComparison.Ordinal);
        Assert.EndsWith(".wav", url, StringComparison.Ordinal);
        string path = Path.Combine(Uploads, Uri.UnescapeDataString(url["/uploads/".Length..]));
        byte[] wav = File.ReadAllBytes(path);
        float[] all = Enumerable.Range(0, 3).SelectMany(SpeechEndpointTests.Chunk).ToArray();
        Assert.Equal(Lfm3SpeechAudio.Wav(all, 24000), wav);
        Assert.Equal("RIFF", System.Text.Encoding.ASCII.GetString(wav, 0, 4));
        Assert.Equal("WAVE", System.Text.Encoding.ASCII.GetString(wav, 8, 4));
        Assert.Equal(1, BinaryPrimitives.ReadInt16LittleEndian(wav.AsSpan(22)));       // mono
        Assert.Equal(24000, BinaryPrimitives.ReadInt32LittleEndian(wav.AsSpan(24)));   // rate
        Assert.Equal(16, BinaryPrimitives.ReadInt16LittleEndian(wav.AsSpan(34)));      // PCM16
        Assert.Equal(all.Length * 2, BinaryPrimitives.ReadInt32LittleEndian(wav.AsSpan(40)));

        // The text turn of the request, as a text turn's: its session, thinking off, the history as sent (no speech line
        // anywhere), and no driven reply; the TTS spoke its sentence in the model's first voice.
        var call = Assert.Single(model.Calls);
        Assert.Same(session, call.Session);
        Assert.False(call.Think);
        Assert.Equal("Say hello to Anna.", Assert.Single(call.History).Content);
        Assert.Equal(0, model.DrivenReplies);
        var said = Assert.Single(model.Spoken);
        Assert.Equal("Hello, Anna!", said.Text);
        Assert.Equal("female", said.Options.Voice);
    }

    [Fact]
    public async Task TheVoiceAndLanguageAskedForReachTheSpeech()
    {
        var model = new SpeakingService();
        var (chat, session) = Build(model);
        await Frames(chat, Body(new
        {
            sessionId = session.Id, messages = FirstTurn(), speech = new { voice = "Male", language = "fr" },
        }));
        var said = Assert.Single(model.Spoken);
        Assert.Equal("Male", said.Options.Voice);
        Assert.Equal("fr", said.Options.Language);
        Assert.Single(model.Calls);
    }

    [Fact]
    public async Task APlainSpokenTurnHasNoNotice()
    {
        var model = new SpeakingService();
        var (chat, session) = Build(model);
        List<JsonElement> frames = await Frames(chat, Body(new { sessionId = session.Id, messages = FirstTurn(), speech = true }));
        Assert.DoesNotContain(frames, f => f.TryGetProperty("notice", out _));
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("{\"speech\":null}")]
    [InlineData("{\"speech\":false}")]
    public void WithoutSpeechTheReplyIsText(string body)
    {
        Assert.Null(WebUiSpeechRequest.Parse(JsonSerializer.Deserialize<JsonElement>(body), out string error));
        Assert.Null(error);
    }

    [Fact]
    public async Task AStoppedSpokenTurnWritesNoWavAndHasNoAudioId()
    {
        var model = new SpeakingService { CancelAfter = 2 };
        var (chat, session) = Build(model);
        List<JsonElement> frames = await Frames(chat, Body(new { sessionId = session.Id, messages = FirstTurn(), speech = true }));

        JsonElement done = frames[^1];
        Assert.True(done.GetProperty("aborted").GetBoolean());
        Assert.False(done.TryGetProperty("audioId", out _));
        Assert.False(done.TryGetProperty("audioUrl", out _));
        Assert.Empty(Directory.EnumerateFiles(Uploads, "speech-*.wav"));
        // The text shown before the stop was not a whole sentence, so nothing was spoken.
        Assert.Empty(model.Spoken);
    }

    // ------------------------------------------------------------------------------------------------ refusals

    [Fact]
    public async Task AModelWithoutSpeechOutputRefusesSpeechWithItsReason()
    {
        var (textChat, textSession) = Build(new TextService());
        var refused = await Assert.ThrowsAsync<WebUiRequestRejectedException>(async () =>
            await Frames(textChat, Body(new { sessionId = textSession.Id, messages = FirstTurn(), speech = true })));
        Assert.Equal(400, refused.StatusCode);
        Assert.Contains("(qwen35) has no speech output", refused.Message, StringComparison.Ordinal);
        Assert.Contains("speech_unavailable", JsonSerializer.Serialize(refused.Payload), StringComparison.Ordinal);

        var silent = new SpeakingService { Speaker = new Speaker { Loaded = false } };
        var (silentChat, silentSession) = Build(silent);
        refused = await Assert.ThrowsAsync<WebUiRequestRejectedException>(async () =>
            await Frames(silentChat, Body(new { sessionId = silentSession.Id, messages = FirstTurn(), speech = true })));
        Assert.Equal(400, refused.StatusCode);
        Assert.Contains("load LFM3-Omni's mmproj", refused.Message, StringComparison.Ordinal);
        Assert.Empty(silent.Calls);
    }

    [Theory]
    [InlineData("\"yes\"", "speech must be true or an object")]
    [InlineData("{\"voice\":\"alloy\"}", "female, male")]
    [InlineData("{\"voice\":\"\"}", "nonempty string")]
    [InlineData("{\"seed\":1}", "only voice and language")]
    public async Task ASpeechFieldItCannotServeIsRefused(string speech, string expected)
    {
        var model = new SpeakingService();
        var (chat, session) = Build(model);
        JsonElement body = JsonSerializer.Deserialize<JsonElement>(
            "{\"sessionId\":\"" + session.Id + "\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"speech\":" + speech + "}");
        var refused = await Assert.ThrowsAsync<WebUiRequestRejectedException>(async () => await Frames(chat, body));
        Assert.Equal(400, refused.StatusCode);
        Assert.Contains(expected, refused.Message, StringComparison.Ordinal);
        Assert.Empty(model.Calls);
    }

    // ------------------------------------------------------------------------------------------------ next turn

    [Fact]
    public async Task TheNextTurnRendersTheSpokenTurnAsItsText()
    {
        var model = new SpeakingService();
        var (chat, session) = Build(model);
        JsonElement done = (await Frames(chat, Body(new { sessionId = session.Id, messages = FirstTurn(), speech = true })))[^1];
        string audioId = Text(done, "audioId");

        // The page sends the reply back as it holds it: the transcript, its audio id and its WAV.
        await Frames(chat, Body(new
        {
            sessionId = session.Id, speech = true,
            messages = new object[]
            {
                new { role = "user", content = "Say hello to Anna." },
                new { role = "assistant", content = "Hello, Anna!", audioId, audioUrl = Text(done, "audioUrl"), audioSeconds = 0.24 },
                new { role = "user", content = "And to Ben?" },
            },
        }));

        // A text turn's history: the reply as its text, which the session's transcript store matches to its tokens.
        List<ChatMessage> second = model.Calls[1].History;
        Assert.Equal(3, second.Count);
        Assert.Equal("Hello, Anna!", second[1].Content);
        Assert.Null(second[1].RawOutputTokens);
        Assert.Null(second[1].SpeechFrames);
    }

    [Fact]
    public async Task AnEditedOrUnknownSpokenTurnRendersAsItsText()
    {
        var model = new SpeakingService();
        var (chat, session) = Build(model);
        JsonElement done = (await Frames(chat, Body(new { sessionId = session.Id, messages = FirstTurn(), speech = true })))[^1];

        await Frames(chat, Body(new
        {
            sessionId = session.Id, speech = true,
            messages = new object[]
            {
                new { role = "user", content = "Say hello to Anna." },
                new { role = "assistant", content = "Hello, Bob!", audioId = Text(done, "audioId") },
                new { role = "user", content = "Again." },
                new { role = "assistant", content = "Hi.", audioId = "audio_0123456789abcdef0123456789abcdef" },
                new { role = "user", content = "Once more." },
            },
        }));

        List<ChatMessage> history = model.Calls[1].History;
        Assert.Null(history[1].RawOutputTokens);
        Assert.Null(history[1].SpeechFrames);
        Assert.Equal("Hello, Bob!", history[1].Content);
        Assert.Null(history[3].RawOutputTokens);
        Assert.Equal("Hi.", history[3].Content);
    }

    // ------------------------------------------------------------------------------------------------ /api/models

    [Fact]
    public void ModelsReportWhetherRepliesCanBeSpoken()
    {
        JsonElement speaking = JsonSerializer.SerializeToElement(Build(new SpeakingService()).Chat.GetModels()).GetProperty("speechOutput");
        Assert.True(speaking.GetProperty("available").GetBoolean());
        Assert.Equal(new[] { "female", "male" }, speaking.GetProperty("voices").EnumerateArray().Select(v => v.GetString()));
        Assert.Equal(JsonValueKind.Null, speaking.GetProperty("reason").ValueKind);

        JsonElement silent = JsonSerializer.SerializeToElement(
            Build(new SpeakingService { Speaker = new Speaker { Loaded = false } }).Chat.GetModels()).GetProperty("speechOutput");
        Assert.False(silent.GetProperty("available").GetBoolean());
        Assert.Empty(silent.GetProperty("voices").EnumerateArray());
        Assert.Contains("load LFM3-Omni's mmproj", silent.GetProperty("reason").GetString(), StringComparison.Ordinal);

        JsonElement text = JsonSerializer.SerializeToElement(Build(new TextService()).Chat.GetModels()).GetProperty("speechOutput");
        Assert.False(text.GetProperty("available").GetBoolean());
        Assert.Contains("(qwen35) has no speech output", text.GetProperty("reason").GetString(), StringComparison.Ordinal);
    }
}
