// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Models;
using TensorSharp.Models.Audio;
using TensorSharp.Models.Lfm3;
using TensorSharp.Models.Lora;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// A spoken web chat turn is the request's text turn with its answer spoken: whatever the text turn runs (reasoning, the
/// client's tools, the host's code tools, skills, every round) it runs, through the same generation call and the same
/// frames, and the model's TTS speaks the answer text of each round as it is written. Without a model: a service whose
/// turn is a scripted tool round (the skill loop's parsed updates: reasoning, the text before a call, the call's
/// progress, the answer) and whose TTS records what it is asked to say. Checked: which turns stay one driven generation,
/// what is spoken per round (never reasoning, the call or its result; not code, tables or a host note), that the speech
/// streams during the tool round, the one WAV and audio id over every round, the next turn rendering it as text (so the
/// transcript store reuses it as a text turn's), a speech failure, and a stop in the middle of a tool round.
/// </summary>
public sealed class WebUiSpokenAgentTurnTests : IDisposable
{
    private static readonly TimeSpan Patience = TimeSpan.FromSeconds(30);
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-webui-speech-agent-" + Guid.NewGuid().ToString("N"));

    public WebUiSpokenAgentTurnTests()
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

    /// <summary>LFM3-Omni's speech: its interleaved reply speaks English only, every other language from the text.</summary>
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
        public string SpeechLanguageOf(string text, string preferred) => Lfm3SpeechPrompt.SpeechLanguageOf(text, preferred);
        public SpeechReplyPath SpeechReplyPathFor(string language)
            => Lfm3SpeechPrompt.SpeaksRepliesIn(language) ? SpeechReplyPath.Direct : SpeechReplyPath.Synthesized;
        public SpeechReplyDriver CreateSpeechDriver(IReadOnlyList<int> promptTokens, SpeechReplyOptions options) => throw new NotSupportedException();
    }

    private sealed class ProbeModel(string path) : ModelBase(path, BackendType.Cpu)
    {
        protected override float[] ForwardCore(int[] tokens) => [];
        protected override void ResetKVCacheCore() { }
    }

    internal sealed record TurnCall(List<ChatMessage> History, List<ToolFunction> Tools, bool Think, SkillRequestPlan Plan);

    internal sealed record TtsCall(string Text, string Language, string Voice);

    /// <summary>
    /// A text turn of two rounds as the skill loop streams it: reasoning, "Let me run that" and a shell call written
    /// (progress, never content), the call run, then the answer with code, a table, a path and a URL. While the call
    /// runs, the turn waits for the speech's first audio, so the audio of the text before the call is sent during the
    /// tool round, or the test fails. <see cref="StopInTool"/> waits instead for the turn's cancellation.
    /// </summary>
    private sealed class AgentService : ModelService
    {
        public AgentService() : base(NullLogger<ModelService>.Instance, (path, _, _, _) => new ProbeModel(path)) { }

        public ISpeechOutputModel Speaker { get; init; } = new OmniSpeaker();
        public bool StopInTool { get; init; }
        public Exception TtsFails { get; init; }
        public bool HoldTts { get; init; }
        public string[] Answer { get; init; } =
        {
            "The answer is **42**.\n\n",
            "```python\nprint(6 * 7)\n```\n",
            "| x | y |\n|---|---|\n| 6 | 7 |\n",
            "Saved to /Users/me/work/out/result.csv, see https://docs.python.org/3/library/math.html for more.",
        };

        /// <summary>A plain reply a call (by its index) instead of the scripted tool round.</summary>
        public Func<int, string> PlainReply { get; init; }

        public List<TurnCall> TurnCalls { get; } = new();
        public List<(List<ChatMessage> History, SpeechReplyOptions Speech)> DirectCalls { get; } = new();
        public List<TtsCall> TtsCalls { get; } = new();
        public TaskCompletionSource FirstAudioTaken { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        /// <summary>Set once a <see cref="StopInTool"/> turn's tool round is waiting for the turn's cancellation: until then its
        /// speech (unless <see cref="HoldTts"/>) holds its first audio, so that audio reaches the consumer while the pull of
        /// the tool round is in flight.</summary>
        public TaskCompletionSource ToolRoundWaiting { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public bool TtsSawCancellation;
        public bool ToolSawCancellation;

        public override string Architecture => "lfm3";
        public override ISpeechOutputModel SpeechOutputModel => Speaker;

        internal override async IAsyncEnumerable<ChatStreamUpdate> ChatStreamWithSkillsAsync(ChatSession session,
            List<ChatMessage> history, int maxTokens, [EnumeratorCancellation] CancellationToken cancellationToken,
            SamplingConfig samplingConfig, List<ToolFunction> tools, bool enableThinking, SkillRequestPlan skills,
            ILogger logger = null, int? thinkingTokenLimit = null, TextLoraAdapter lora = null)
        {
            TurnCalls.Add(new TurnCall(history, tools, enableThinking, skills));
            await Task.Yield();
            if (PlainReply != null)
            {
                yield return ChatStreamUpdate.Text(PlainReply(TurnCalls.Count - 1));
                yield return new ChatStreamUpdate("", true, 10, 5, 0, 0, 0, 0, "stop");
                yield break;
            }
            yield return ChatStreamUpdate.Parsed(string.Empty, "I should compute 6 times 7.", null);
            yield return ChatStreamUpdate.Parsed("Let me run that", null, null);
            yield return ChatStreamUpdate.ToolProgress("writing", "shell", "python3 -c 'print(6 * 7)'");
            yield return ChatStreamUpdate.ToolProgress("running", "shell", detail: "python3 -c 'print(6 * 7)'");
            if (StopInTool)
            {
                ToolRoundWaiting.TrySetResult();
                try { await Task.Delay(Timeout.Infinite, cancellationToken); }
                catch (OperationCanceledException) { ToolSawCancellation = true; throw; }
            }
            else if (TtsFails == null)
                await FirstAudioTaken.Task.WaitAsync(Patience, cancellationToken);
            yield return ChatStreamUpdate.ToolProgress("finished", "shell", seconds: 0.2);
            foreach (string piece in Answer)
            {
                await Task.Yield();
                yield return ChatStreamUpdate.Parsed(piece, null, null);
            }
            yield return new ChatStreamUpdate("", true, 120, 30, 100, 0, 1_000_000, 2_000_000, "stop");
        }

        internal override async IAsyncEnumerable<ChatStreamUpdate> SpeechSynthesisStreamAsync(string text,
            SpeechSynthesisOptions options, [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            int index;
            lock (TtsCalls)
            {
                index = TtsCalls.Count;
                TtsCalls.Add(new TtsCall(text, options.Language, options.Voice));
            }
            if (TtsFails != null) throw TtsFails;
            // A turn that stops in its tool round speaks the text before the call once the round waits (the web chat yields
            // audio it already holds between two pulls too, when no pull is in flight: a consumer leaving there would stop
            // the turn before its tool round, which these tests do not mean). Held speech yields no audio to race with.
            if (StopInTool && !HoldTts) await ToolRoundWaiting.Task.WaitAsync(Patience, cancellationToken);
            for (int f = 0; f < 2; f++)
            {
                if (HoldTts)
                {
                    try { await Task.Delay(Timeout.Infinite, cancellationToken); }
                    catch (OperationCanceledException) { TtsSawCancellation = true; throw; }
                }
                await Task.Yield();
                yield return ChatStreamUpdate.AudioChunk(new float[] { index, f, 0.25f }, 24000);
                if (index == 0 && f == 0) FirstAudioTaken.TrySetResult();
            }
            yield return new ChatStreamUpdate("", true, 3, 2, 0, 0, 0, 0, "eos")
            {
                SpeechFrames = new[] { new[] { index, 0 }, new[] { index, 1 } }, AudioSampleRate = 24000,
            };
        }

        internal override async IAsyncEnumerable<ChatStreamUpdate> ChatStreamWithSpeechAsync(ChatSession session,
            List<ChatMessage> history, int maxTokens, [EnumeratorCancellation] CancellationToken cancellationToken,
            SamplingConfig samplingConfig, bool enableThinking, SpeechReplyOptions speech, TextLoraAdapter lora = null)
        {
            DirectCalls.Add((history, speech));
            await Task.Yield();
            yield return ChatStreamUpdate.Text("Hello.");
            yield return ChatStreamUpdate.AudioChunk(new float[] { 9, 9 }, 24000);
            yield return new ChatStreamUpdate("", true, 10, 2, 0, 0, 0, 1_000, "eos")
            {
                SpeechFrames = new[] { new[] { 1, 2 } }, AudioSampleRate = 24000,
                SpokenTurn = WebUiSpokenReplyTests.Turn,
            };
        }
    }

    /// <summary>The host's code tools (nothing is ever run: the scripted turn reports the call itself).</summary>
    private sealed class CodeTools : ICodeRunner
    {
        public bool CanRun => true;
        public string UnavailableReason => null;
        public ToolFunction Declare() => new() { Name = SkillToolNames.Shell };
        public IReadOnlyList<ToolFunction> DeclareTools(bool persists) =>
            new[] { SkillToolNames.ReadFile, SkillToolNames.ApplyPatch, SkillToolNames.WriteFile, SkillToolNames.Shell }
                .Select(name => new ToolFunction { Name = name }).ToArray();
        public SkillToolResult Execute(ToolCall call, IReadOnlyList<CodeInputFile> inputFiles = null,
            Action<string> onOutput = null, SessionWorkspace workspace = null, IReadOnlyList<string> skillDirectories = null) =>
            throw new InvalidOperationException("the scripted turn runs no tool");
    }

    private (WebUiChatService Chat, ChatSession Session) Build(AgentService model, bool codeTools, bool delegation = false)
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
            fileLoggingEnabled: false, samplingDefaults: null, skillsEnabled: false, skillsAllowScripts: false,
            skillsSandbox: SkillSandboxMode.Required, skillsAllowNetwork: false,
            multiAgent: new TensorSharp.AgentHost.Agents.MultiAgentOptions { Enabled = delegation });
        var chat = new WebUiChatService(model, sessions, options, new UploadStoragePolicy(Uploads),
            new SkillRegistry(new SkillRegistryOptions { Roots = [] }),
            codeTools ? new CodeTools() : null,
            codeTools ? new SessionWorkspaceManager(Path.Combine(_root, "workspaces")) : null,
            codeArtifacts: null, NullLoggerFactory.Instance);
        return (chat, session);
    }

    private static JsonElement Body(object body) => JsonSerializer.SerializeToElement(body);

    private static async Task<List<JsonElement>> Frames(WebUiChatService chat, JsonElement body, CancellationToken ct = default)
    {
        var frames = new List<JsonElement>();
        await foreach (object frame in chat.ChatStreamAsync(body, ct))
            frames.Add(JsonSerializer.SerializeToElement(frame));
        return frames;
    }

    private static string Text(JsonElement frame, string name)
        => frame.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;

    private static bool Has(JsonElement frame, string name) => frame.TryGetProperty(name, out _);

    private static object[] Ask(string question) => new object[] { new { role = "user", content = question } };

    private static readonly object[] WeatherTool =
    {
        new { type = "function", function = new { name = "get_weather", description = "weather", parameters = new { type = "object" } } },
    };

    // ------------------------------------------------------------------------------------------------ routing

    [Fact]
    public async Task APlainEnglishSpokenTurnIsTheTextTurnSpokenToo()
    {
        var model = new AgentService { PlainReply = _ => "Hello." };
        var (chat, session) = Build(model, codeTools: false);
        List<JsonElement> frames = await Frames(chat, Body(new { sessionId = session.Id, messages = Ask("Say hello."), speech = true }));

        Assert.Empty(model.DirectCalls);
        Assert.Single(model.TurnCalls);
        Assert.Equal("Hello.", Assert.Single(model.TtsCalls).Text);
        Assert.DoesNotContain(frames, f => Has(f, "notice"));
        Assert.StartsWith("audio_", Text(frames[^1], "audioId"), StringComparison.Ordinal);
    }

    public static IEnumerable<object[]> AgenticRequests() => new[]
    {
        new object[] { "think", false, false },
        new object[] { "tools", false, false },
        new object[] { "code", true, false },
        new object[] { "delegation", false, true },
        new object[] { "chinese", false, false },
    };

    [Theory]
    [MemberData(nameof(AgenticRequests))]
    public async Task ASpokenTurnThatNeedsAnythingAgenticRunsTheTextTurnAndSpeaksIt(string what, bool codeTools, bool delegation)
    {
        var model = new AgentService();
        var (chat, session) = Build(model, codeTools, delegation);
        object body = what switch
        {
            "think" => new { sessionId = session.Id, messages = Ask("What is 6 times 7?"), speech = true, think = true },
            "tools" => new { sessionId = session.Id, messages = Ask("What is the weather?"), speech = true, tools = WeatherTool },
            "code" => new { sessionId = session.Id, messages = Ask("Compute 6 times 7 with Python."), speech = true },
            // The host offers sub-agents (its default): the text turn declares their tools, so the spoken turn does too.
            "delegation" => new { sessionId = session.Id, messages = Ask("Say hello."), speech = true },
            _ => (object)new { sessionId = session.Id, messages = Ask("用 Python 算一下六乘七。"), speech = true },
        };
        List<JsonElement> frames = await Frames(chat, Body(body));

        // The text turn's generation call, as the text turn of the same request makes it, and no driven reply.
        Assert.Empty(model.DirectCalls);
        TurnCall call = Assert.Single(model.TurnCalls);
        Assert.Equal(what == "think", call.Think);
        if (what == "tools") Assert.Contains(call.Tools, t => t.Name == "get_weather");
        if (what == "code")
        {
            Assert.True(call.Plan?.ToolsOffered);
            Assert.Contains(call.Tools, t => t.Name == SkillToolNames.Shell);
        }
        if (what == "delegation") Assert.True(call.Plan?.ToolsOffered);
        Assert.NotEmpty(model.TtsCalls);
        Assert.DoesNotContain(frames, f => Has(f, "notice"));
        if (what == "chinese") Assert.All(model.TtsCalls, c => Assert.Equal("en", c.Language));   // the replies are English text
    }

    // ------------------------------------------------------------------------------------------------ the prompt

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task SwitchingSpeechOnAndOffKeepsEachPromptAPrefixOfTheNext(bool codeTools)
    {
        // Speak replies switched on and off between turns, as a user does: s t s t s, on a host without tools and on one
        // with its code tools.
        var model = new AgentService { PlainReply = i => i % 2 == 0 ? "Paris is the capital of France." : "Berlin is the capital of Germany." };
        var (chat, session) = Build(model, codeTools);
        string[] questions = { "Capital of France?", "And of Germany?", "Which did you name first?", "Thanks.", "Bye." };
        var history = new List<object>();
        var replies = new List<string>();
        for (int turn = 0; turn < questions.Length; turn++)
        {
            bool spoken = turn % 2 == 0;
            history.Add(new { role = "user", content = questions[turn] });
            List<JsonElement> frames = await Frames(chat, Body(spoken
                ? new { sessionId = session.Id, messages = history.ToArray(), speech = (object)true }
                : (object)new { sessionId = session.Id, messages = history.ToArray() }));
            string reply = string.Concat(frames.Where(f => Has(f, "token")).Select(f => Text(f, "token")));
            JsonElement done = frames[^1];
            history.Add(spoken
                ? new { role = "assistant", content = reply, audioId = Text(done, "audioId"), audioUrl = Text(done, "audioUrl") }
                : (object)new { role = "assistant", content = reply });
            replies.Add(reply);
        }

        // Every turn is the text turn's generation call (no driven reply), and each rendered prompt extends the previous
        // one by its reply: no speech line, the same system prompt and tools, earlier spoken replies as their text.
        Assert.Empty(model.DirectCalls);
        Assert.Equal(questions.Length, model.TurnCalls.Count);
        string previous = null;
        for (int turn = 0; turn < questions.Length; turn++)
        {
            TurnCall call = model.TurnCalls[turn];
            string prompt = ChatTemplate.RenderLfm3(call.History, addGenerationPrompt: true, tools: call.Tools, enableThinking: call.Think);
            Assert.DoesNotContain("interleaved", prompt, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("Speak using", prompt, StringComparison.Ordinal);
            if (codeTools) Assert.Contains("<tools>", prompt, StringComparison.Ordinal);
            if (previous != null)
                Assert.True(prompt.StartsWith(previous + replies[turn - 1], StringComparison.Ordinal),
                    $"turn {turn + 1}'s prompt does not extend turn {turn}'s:\n{previous}\n---\n{prompt}");
            previous = prompt;
        }
        Assert.Equal(3, model.TtsCalls.Count);   // the three spoken turns' replies, one sentence each
    }

    // ------------------------------------------------------------------------------------------------ what is spoken

    [Fact]
    public async Task EachRoundsAnswerTextIsSpokenAndNothingElse()
    {
        var model = new AgentService();
        var (chat, session) = Build(model, codeTools: true);
        List<JsonElement> frames = await Frames(chat, Body(new
        {
            sessionId = session.Id, messages = Ask("Compute 6 times 7 with Python and tell me."), speech = new { voice = "male" },
        }));

        // The text before the call (spoken when the call began, without waiting for a sentence end), then the answer's
        // sentences: code, the table and the URL's path are not read, the path is its file name, the URL its host.
        Assert.Equal(new[]
        {
            "Let me run that",
            "The answer is 42.",
            "Saved to result.csv, see docs.python.org for more.",
        }, model.TtsCalls.Select(c => c.Text).ToArray());
        Assert.All(model.TtsCalls, c => Assert.Equal("male", c.Voice));
        // Never the reasoning, the call's body or its result.
        Assert.DoesNotContain(model.TtsCalls, c => c.Text.Contains("compute", StringComparison.OrdinalIgnoreCase));
        Assert.DoesNotContain(model.TtsCalls, c => c.Text.Contains("print", StringComparison.Ordinal));

        // The text turn's frames, unchanged: reasoning, the answer tokens, the tool's progress.
        Assert.Contains(frames, f => Text(f, "thinking") == "I should compute 6 times 7.");
        Assert.Equal("Let me run that" + string.Concat(model.Answer),
            string.Concat(frames.Where(f => Has(f, "token")).Select(f => Text(f, "token"))));
        Assert.Contains(frames, f => Text(f, "tool_progress") == "finished");

        // The speech of the text before the call reached the page while the call ran.
        int firstAudio = frames.FindIndex(f => Has(f, "audio"));
        int toolFinished = frames.FindIndex(f => Text(f, "tool_progress") == "finished");
        Assert.True(firstAudio >= 0 && firstAudio < toolFinished, $"first audio frame {firstAudio}, tool finished {toolFinished}");
        Assert.Equal(model.TtsCalls.Count * 2, frames.Count(f => Has(f, "audio")));
    }

    [Fact]
    public async Task AHostNoteAndRetractedTextAreNotSpoken()
    {
        var heard = new List<string>();
        var turn = new WebUiSpokenTurn(new TensorSharp.Server.ReplySpeaker(
            (text, index, ct) => Speak(text, index, ct), new SpeechTextSegmenter(), CancellationToken.None), "en");
        async IAsyncEnumerable<ChatStreamUpdate> Speak(string text, int index, [EnumeratorCancellation] CancellationToken ct)
        {
            lock (heard) heard.Add(text);
            await Task.Yield();
            yield return ChatStreamUpdate.AudioChunk(new float[] { index }, 24000);
        }

        turn.Say("Hmm, so the");
        turn.Retract("Hmm, so the");                 // it was reasoning after all
        turn.Say("Seven.");
        turn.EndOfRound();
        turn.Say("\n\n_(The model's output started repeating itself and was stopped. Ask it to try again.)_");
        turn.Say(" Done.");
        turn.Speaker.Complete();
        await turn.Speaker.Completion.WaitAsync(Patience);

        Assert.Equal(new[] { "Seven.", "Done." }, heard);
        Assert.Equal(2, turn.TakeAudioFrames().Count());
        Assert.Equal(new[] { 0f, 1f }, turn.Pcm);
    }

    // ------------------------------------------------------------------------------------------------ the turn's audio

    [Fact]
    public async Task OneWavAndAudioIdCoverEveryRoundAndTheNextTurnRendersItAsText()
    {
        var model = new AgentService();
        var (chat, session) = Build(model, codeTools: true);
        List<JsonElement> frames = await Frames(chat, Body(new
        {
            sessionId = session.Id, messages = Ask("Compute 6 times 7 with Python and tell me."), speech = true,
        }));

        JsonElement done = frames[^1];
        Assert.True(done.GetProperty("done").GetBoolean());
        Assert.False(done.GetProperty("aborted").GetBoolean());
        Assert.Equal(JsonValueKind.Null, done.GetProperty("error").ValueKind);
        Assert.Equal(120, done.GetProperty("promptTokens").GetInt32());
        Assert.Equal(100, done.GetProperty("kvReusedTokens").GetInt32());
        string audioId = Text(done, "audioId");
        Assert.StartsWith("audio_", audioId, StringComparison.Ordinal);

        // Every piece's audio, in order, in the one WAV.
        float[] all = Enumerable.Range(0, model.TtsCalls.Count)
            .SelectMany(i => new[] { new float[] { i, 0, 0.25f }, new float[] { i, 1, 0.25f } }).SelectMany(c => c).ToArray();
        string url = Text(done, "audioUrl");
        byte[] wav = File.ReadAllBytes(Path.Combine(Uploads, Uri.UnescapeDataString(url["/uploads/".Length..])));
        Assert.Equal(Lfm3SpeechAudio.Wav(all, 24000), wav);
        Assert.Equal(Math.Round(all.Length / 24000.0, 3), done.GetProperty("audioSeconds").GetDouble(), 6);
        Assert.True(chat.SpeechReplies.TryGet(audioId, out var entry));
        Assert.Null(entry.Turn);                      // a text turn: no generation record to splice

        // The page sends the turn back as it holds it; it renders as its text (the transcript store matches it to the
        // turn's tokens, tool rounds included, as for a text turn), through the text turn again.
        string answer = string.Concat(frames.Where(f => Has(f, "token")).Select(f => Text(f, "token")));
        await Frames(chat, Body(new
        {
            sessionId = session.Id, speech = true,
            messages = new object[]
            {
                new { role = "user", content = "Compute 6 times 7 with Python and tell me." },
                new { role = "assistant", content = answer, audioId, audioUrl = url },
                new { role = "user", content = "And 7 times 8?" },
            },
        }));
        Assert.Equal(2, model.TurnCalls.Count);
        ChatMessage replayed = model.TurnCalls[1].History.Single(m => m.Role == "assistant");
        Assert.Equal(answer, replayed.Content);
        Assert.Null(replayed.RawOutputTokens);
        Assert.Null(replayed.SpeechFrames);
    }

    [Fact]
    public async Task ASpeechFailureKeepsTheTextTurnAndSaysSo()
    {
        var model = new AgentService { TtsFails = new InvalidOperationException("speech decoder failed") };
        var (chat, session) = Build(model, codeTools: true);
        List<JsonElement> frames = await Frames(chat, Body(new
        {
            sessionId = session.Id, messages = Ask("Compute 6 times 7 with Python and tell me."), speech = true,
        }));

        JsonElement notice = Assert.Single(frames, f => Has(f, "notice"));
        Assert.Equal("speech_failed", Text(notice, "code"));
        Assert.Contains("speech decoder failed", Text(notice, "notice"), StringComparison.Ordinal);
        Assert.Equal("Let me run that" + string.Concat(model.Answer),
            string.Concat(frames.Where(f => Has(f, "token")).Select(f => Text(f, "token"))));
        JsonElement done = frames[^1];
        Assert.Equal(JsonValueKind.Null, done.GetProperty("error").ValueKind);
        Assert.StartsWith("audio_", Text(done, "audioId"), StringComparison.Ordinal);
        Assert.Equal(JsonValueKind.Null, done.GetProperty("audioUrl").ValueKind);   // nothing was spoken
    }

    [Fact]
    public async Task AConsumerThatLeavesDuringAToolRoundsAudioStopsTheTurnCleanly()
    {
        // The speech of the text before the call is sent while the call runs (a pull is in flight); the page goes away
        // at that frame, without cancelling anything.
        var model = new AgentService { StopInTool = true };
        var (chat, session) = Build(model, codeTools: true);
        async Task<int> ReadToFirstAudio()
        {
            int frames = 0;
            await foreach (object frame in chat.ChatStreamAsync(Body(new
            {
                sessionId = session.Id, messages = Ask("Compute 6 times 7 with Python and tell me."), speech = true,
            }), CancellationToken.None))
            {
                frames++;
                if (JsonSerializer.SerializeToElement(frame).TryGetProperty("audio", out _)) break;
            }
            return frames;
        }

        int read = await ReadToFirstAudio().WaitAsync(Patience);
        Assert.True(read > 0);
        // Leaving stopped the generation (its tool round saw the cancellation) before its enumerator was disposed.
        Assert.True(model.ToolSawCancellation);
        Assert.Equal("Let me run that", Assert.Single(model.TtsCalls).Text);
    }

    [Fact]
    public async Task AStopDuringAToolRoundStopsTheSpeechAndKeepsNoAudio()
    {
        var model = new AgentService { StopInTool = true, HoldTts = true };
        var (chat, session) = Build(model, codeTools: true);
        using var stop = new CancellationTokenSource();
        var frames = new List<JsonElement>();
        await foreach (object frame in chat.ChatStreamAsync(Body(new
        {
            sessionId = session.Id, messages = Ask("Compute 6 times 7 with Python and tell me."), speech = true,
        }), stop.Token))
        {
            JsonElement f = JsonSerializer.SerializeToElement(frame);
            frames.Add(f);
            // The call is running and the text before it is being spoken: the user presses Stop.
            if (Text(f, "tool_progress") == "running")
            {
                for (int i = 0; i < 3000 && model.TtsCalls.Count == 0; i++) await Task.Delay(10);
                stop.Cancel();
            }
        }

        JsonElement done = frames[^1];
        Assert.True(done.GetProperty("aborted").GetBoolean());
        Assert.False(Has(done, "audioId"));
        Assert.Empty(Directory.EnumerateFiles(Uploads, "speech-*.wav"));
        Assert.Equal("Let me run that", Assert.Single(model.TtsCalls).Text);
        Assert.True(model.TtsSawCancellation);
        Assert.DoesNotContain(frames, f => Text(f, "tool_progress") == "finished");
    }
}
