// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
// Licensed under the BSD-3-Clause license in the root directory.

using System.Text;
using System.Text.Json;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// Exercises the actual OpenAI parser, history preparation, transcript lookup and Qwen
/// prompt renderer. An injective UTF-8 tokenizer keeps these CPU tests independent of a
/// downloaded vocabulary; they compare the entire resulting prompt, not just a helper.
/// </summary>
public class TranscriptSemanticPreservationTests
{
    [Theory]
    [InlineData("a b", "ab")]
    [InlineData("value = \"a b\"", "value = \"ab\"")]
    [InlineData("if True:\n    print(\"a\")\nprint(\"b\")", "if True:\n    print(\"a\")\n    print(\"b\")")]
    [InlineData("first\tsecond", "firstsecond")]
    [InlineData("first\nsecond", "firstsecond")]
    public void EditedAssistantWhitespace_RendersClientHistoryWithoutRawReplay(string original, string edited)
        => new Replay(original, edited).AssertRejected();

    [Theory]
    [InlineData("9007199254740992", "9007199254740993")]
    [InlineData("9007199254740993", "9007199254740992")]
    [InlineData("-9007199254740992", "-9007199254740993")]
    [InlineData("1.000000000000000000000001", "1.000000000000000000000002")]
    [InlineData("1e1000", "2e1000")]
    [InlineData("-0", "0")]
    // Equivalent number spellings may conservatively miss; they must never round
    // distinct values into the same ownership proof.
    [InlineData("1", "1.0")]
    [InlineData("1e2", "100")]
    public void ChangedExactNumber_RendersClientArgumentWithoutRounding(string original, string edited)
        => new Replay("", "", "{\"id\":" + original + "}", "{\"id\":" + edited + "}").AssertRejected();

    [Theory]
    [InlineData("a b", "ab")]
    [InlineData("if x:\n    y()", "if x:\ny()")]
    [InlineData(null, "new reasoning")]
    public void EditedReasoning_IsNotReplacedWithOldRawTokens(string? original, string edited)
        => new Replay("answer", "answer", originalThinking: original, clientThinking: edited).AssertRejected();

    [Fact]
    public void ExactUnchangedAnswer_StillReplaysRawTokens()
        => new Replay("a b", "a b").AssertAccepted();

    [Fact]
    public void OmittedReasoning_StillReplaysTheRecordedTurn()
        => new Replay("answer", "answer", originalThinking: "a b", clientThinking: null).AssertAccepted();

    [Fact]
    public void ToolArgumentObjectOrderAndCallId_StillMatchExactValues()
        => new Replay("", "",
            "{\"id\":9007199254740993,\"nested\":{\"a\":1,\"b\":[\"a b\",2]}}",
            "{ \"nested\": {\"b\":[\"a b\",2], \"a\":1}, \"id\":9007199254740993 }").AssertAccepted();

    [Theory]
    [InlineData("Once upon a time there was a small robot who lov", true)]
    [InlineData("Onceupon a time there was a small robot who lov", false)]
    public void CancelledTurn_PreservesOnlyItsExistingExactPrefixShape(string client, bool accepts)
    {
        var replay = new Replay("Once upon a time there was a small robot who loved", client, cancelled: true);
        if (accepts) replay.AssertAccepted(); else replay.AssertRejected();
    }

    [Theory]
    [InlineData("First.\n\nHere is a b.\n\n[Download](/files/result)", true)]
    [InlineData("First.\n\nHere is ab.\n\n[Download](/files/result)", false)]
    [InlineData("First.\n\nHere is a b. [Download](/files/result)", false)]
    public void CollapsedHostNote_RequiresTheExactEmittedPrefixAndBlankLine(string client, bool accepts)
    {
        var replay = new Replay("Here is a b.", client, collapsedToolLoop: true);
        if (accepts) replay.AssertAccepted(); else replay.AssertRejected();
    }

    [Theory]
    [InlineData("first plansecond planall done", true)]
    [InlineData("all done", true)]
    [InlineData(null, true)]
    [InlineData("done", false)]
    [InlineData("ll done", false)]
    [InlineData("second planall done", false)]
    [InlineData("planall done", false)]
    public void CollapsedReasoning_MatchesOnlyAllRoundsOrTheCompleteFinalRound(string? clientThinking, bool accepts)
    {
        var replay = new Replay("answer", "First.\n\nSecond.\n\nanswer", originalThinking: "all done",
            clientThinking: clientThinking, collapsedToolLoop: true, priorReasoning: new[] { "first plan", "second plan" });
        if (accepts) replay.AssertAccepted(); else replay.AssertRejected();
    }

    private sealed class Replay
    {
        private readonly Utf8Tokenizer _tokenizer = new();
        private readonly KVCachePromptRenderer _renderer = new(new GgufPromptRenderer());
        private readonly ConversationTranscriptStore _store = new(64, 100_000);
        private readonly List<ChatMessage> _incoming;
        private readonly List<int> _raw;

        internal Replay(string original, string edited, string? originalArguments = null, string? editedArguments = null,
            string? originalThinking = null, string? clientThinking = null, bool cancelled = false, bool collapsedToolLoop = false,
            string[]? priorReasoning = null)
        {
            var old = Parse(original, originalArguments, originalThinking, "call-old");
            _incoming = Parse(edited, editedArguments, clientThinking, "call-new");
            string rawText = old[1].ToolCalls is { Count: > 0 } calls ? RawTool(calls[0]) : original;
            if (originalThinking != null)
                rawText = originalThinking + "</think>\n\n" + rawText;
            _raw = _tokenizer.Encode(rawText, addSpecial: false);
            var generated = new ChatMessage
            {
                Role = "assistant", Content = rawText, RawOutputTokens = _raw,
                RawGenerationSuffix = originalThinking == null ? "<think>\n\n</think>\n\n" : "<think>\n",
                RawPromptTrailingWhitespace = originalThinking == null ? "\n\n" : "\n",
            };
            var preceding = new List<ChatMessage> { old[0] };
            if (collapsedToolLoop)
            {
                for (int i = 0; i < (priorReasoning?.Length ?? 1); i++)
                {
                    string content = i == 0 ? "First.\n\n" : "Second.\n\n";
                    string? thinking = priorReasoning?[i];
                    preceding.Add(new()
                    {
                        Role = "assistant", Content = content, Thinking = thinking,
                        RawOutputTokens = _tokenizer.Encode(thinking == null ? content : thinking + "</think>\n\n" + content, false),
                        RawGenerationSuffix = thinking == null ? "<think>\n\n</think>\n\n" : "<think>\n",
                        RawPromptTrailingWhitespace = thinking == null ? "\n\n" : "\n",
                    });
                    preceding.Add(new() { Role = "tool", Content = "result" });
                }
            }
            _store.Record(preceding, generated,
                new EmittedAssistantTurn(original, old[1].ToolCalls, originalThinking, rawText, cancelled), "previous-scope", epoch: 7);
        }

        internal void AssertRejected()
        {
            var augmented = _store.Augment(_incoming, epoch: 7);
            // Render before asserting the splice decision, so a failing regression
            // also traverses the actual raw-token replacement path.
            Assert.Equal(Render(_incoming), Render(augmented.History));
            Assert.Equal(0, augmented.SplicedTurns);
            Assert.Null(augmented.InheritedScope);
            Assert.All(augmented.History, message => Assert.Null(message.RawOutputTokens));
        }

        internal void AssertAccepted()
        {
            var augmented = _store.Augment(_incoming, epoch: 7);
            Assert.Equal(1, augmented.SplicedTurns);
            Assert.Equal("previous-scope", augmented.InheritedScope);
            Assert.Contains(augmented.History, message => ReferenceEquals(message.RawOutputTokens, _raw));
            // Exercise final production rendering, including raw-token placeholder replay.
            List<int> rendered = Render(augmented.History);
            Assert.NotEmpty(rendered);
            Assert.Equal(rendered, Render(_store.Augment(_incoming, epoch: 7).History));
        }

        private List<int> Render(List<ChatMessage> history)
            => _renderer.RenderToTokens(_tokenizer, null, history, "qwen35", addGenerationPrompt: true, enableThinking: false);

        private static List<ChatMessage> Parse(string content, string? arguments, string? thinking, string callId)
        {
            var messages = new object[]
            {
                new { role = "user", content = "Remember this assistant turn exactly." },
                new { role = "assistant", content, reasoning_content = thinking,
                    tool_calls = arguments == null ? null : new[] { new { id = callId, type = "function", function = new { name = "set_record", arguments } } } },
                new { role = "user", content = "What exactly did the prior assistant say or call?" },
            };
            using var document = JsonDocument.Parse(JsonSerializer.Serialize(messages));
            // This text-only path does not access upload storage or the filesystem.
            return ChatHistoryPreparer.PrepareHistoryForInference(
                ChatMessageParser.ParseOpenAI(document.RootElement, null, architecture: "qwen35"), "qwen35");
        }

        private static string RawTool(ToolCall call)
            => "<tool_call>\n<function=" + call.Name + ">\n"
                + string.Concat(call.Arguments.Select(pair => "<parameter=" + pair.Key + ">\n" + pair.Value + "\n</parameter>\n"))
                + "</function>\n</tool_call>";
    }

    private sealed class Utf8Tokenizer : ITokenizer
    {
        public string[] Vocab { get; } = Enumerable.Range(0, 256).Select(i => ((char)i).ToString()).Concat(new[] { "<bos>", "<eos>" }).ToArray();
        public int BosTokenId => 256;
        public int[] EosTokenIds => new[] { 257 };
        public int VocabSize => Vocab.Length;
        public List<int> Encode(string text, bool addSpecial = true)
        {
            var tokens = new List<int>();
            if (addSpecial) tokens.Add(BosTokenId);
            tokens.AddRange(Encoding.UTF8.GetBytes(text).Select(value => (int)value));
            return tokens;
        }
        public string Decode(List<int> ids) => Encoding.UTF8.GetString(ids.Where(id => id < 256).Select(id => (byte)id).ToArray());
        public void AppendTokenBytes(int tokenId, List<byte> buffer) { if (tokenId < 256) buffer.Add((byte)tokenId); }
        public bool IsEos(int tokenId) => tokenId == 257;
        public int LookupToken(string tokenStr) => tokenStr.Length == 1 && tokenStr[0] < 256 ? tokenStr[0] : -1;
    }
}
