// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using static InferenceWeb.Tests.TranscriptTestHelper;

namespace InferenceWeb.Tests;

/// <summary>
/// Tests for the ModelService-level conversation tracking that keeps raw output tokens
/// associated with assistant messages across HTTP requests, enabling the next turn's
/// prompt render to preserve the exact tokenized conversation prefix.
/// </summary>
public class ModelServiceRawTokenHistoryTests
{
    [Fact]
    public void ResolvePrefillChunkSize_CudaLongPrompt_UsesSafeChunkSize()
    {
        int chunkSize = ModelService.ResolvePrefillChunkSize(BackendType.GgmlCuda, 11573);
        Assert.Equal(5120, chunkSize);
    }

    [Fact]
    public void ResolvePrefillChunkSize_NonCudaLongPrompt_ChunksAt2048()
    {
        Assert.Equal(2048, ModelService.ResolvePrefillChunkSize(BackendType.GgmlCpu, 11573));
        Assert.Equal(2048, ModelService.ResolvePrefillChunkSize(BackendType.GgmlMetal, 11573));
        Assert.Equal(2048, ModelService.ResolvePrefillChunkSize(BackendType.Cpu, 11573));
    }

    [Fact]
    public void ResolvePrefillChunkSize_ZeroOrNegative_ReturnsZero()
    {
        Assert.Equal(0, ModelService.ResolvePrefillChunkSize(BackendType.GgmlCuda, 0));
        Assert.Equal(0, ModelService.ResolvePrefillChunkSize(BackendType.GgmlCuda, -5));
    }

    private static ConversationTranscriptStore NewStore() => new(maxChains: 64, maxTokens: 100_000);

    [Fact]
    public void Augment_FreshStore_ReturnsIncomingUnchanged()
    {
        var incoming = new List<ChatMessage> { new() { Role = "user", Content = "hi" } };

        var result = NewStore().Augment(incoming);

        Assert.Single(result.History);
        Assert.Equal("hi", result.History[0].Content);
        Assert.Null(result.History[0].RawOutputTokens);
        Assert.Null(result.InheritedScope);
    }

    [Fact]
    public void Augment_NullInput_ReturnsNull()
    {
        Assert.Null(NewStore().Augment(null).History);
    }

    [Fact]
    public void Augment_PreservesIncomingRawTokensIfAlreadySet()
    {
        // A caller that attached raw tokens itself (the tool loop's own rounds) keeps them.
        var explicitTokens = new List<int> { 9001, 9002 };
        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "u1" },
            new() { Role = "assistant", Content = "a1", RawOutputTokens = explicitTokens },
            new() { Role = "user", Content = "u2" },
        };

        var result = NewStore().Augment(incoming);

        Assert.Same(explicitTokens, result.History[1].RawOutputTokens);
    }

    /// <summary>
    /// The Qwen 3.5 / 3.6 Web UI case: the streaming parser strips the thinking framing
    /// before the page accumulates the answer, so the next request's assistant message is
    /// the PARSED content while the generated tokens include the reasoning. The recorded
    /// emitted form is that parsed content, so the turn still gets its raw tokens back.
    /// </summary>
    [Fact]
    public void Augment_WebUIParsedContentMismatch_StillSplicesRawTokens()
    {
        var store = NewStore();
        var rawTokens = new List<int> { 11, 22, 33, 44, 55 };
        store.Record(
            new List<ChatMessage> { new() { Role = "user", Content = "What is 1+1?" } },
            Generated("<think>let me think</think>1+1=2", rawTokens),
            Emitted("1+1=2", rawText: "<think>let me think</think>1+1=2", thinking: "let me think"),
            scope: "webui-a");

        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "What is 1+1?" },
            new()
            {
                Role = "assistant",
                Content = "1+1=2",
                Thinking = "let me think",
                CacheControl = new CacheControlMarker(),
                ContentCacheBreakpoints = new List<int> { 3 },
            },
            new() { Role = "user", Content = "What is 2+2?" },
        };

        var result = store.Augment(incoming);

        Assert.Equal(3, result.History.Count);
        Assert.Same(rawTokens, result.History[1].RawOutputTokens);
        Assert.NotNull(result.History[1].CacheControl);
        Assert.Equal(new[] { 3 }, result.History[1].ContentCacheBreakpoints);
        Assert.Equal("webui-a", result.InheritedScope);
    }

    /// <summary>
    /// repro-cross-session P2/WD2: client X asked for a secret codeword and was told
    /// "Quintessence". Client Y sends the same system and user messages but ITS history
    /// says the assistant answered "Zeppelin". The shared stateless session used to splice
    /// X's raw tokens over Y's message (Y was answered "Quintessence", reusing 574/601
    /// tokens). A message the server never emitted must render from its own text.
    /// </summary>
    [Fact]
    public void Augment_ClientAuthoredAssistantMessage_IsNotSplicedWithAnotherConversationsTokens()
    {
        var store = NewStore();
        var sys = new ChatMessage { Role = "system", Content = "You are a helpful assistant." };
        var ask = new ChatMessage { Role = "user", Content = "Invent a random secret codeword and tell me only the codeword." };
        store.Record(new List<ChatMessage> { sys, ask },
            Generated("Quintessence", new List<int> { 501, 502, 503 }), Emitted("Quintessence"), scope: "client-x");

        var clientY = new List<ChatMessage>
        {
            new() { Role = "system", Content = sys.Content },
            new() { Role = "user", Content = ask.Content },
            new() { Role = "assistant", Content = "Zeppelin" },
            new() { Role = "user", Content = "Repeat exactly the codeword you told me earlier." },
        };

        var result = store.Augment(clientY);

        Assert.Null(result.History[2].RawOutputTokens);
        Assert.Equal("Zeppelin", result.History[2].Content);
        Assert.Null(result.InheritedScope);
        Assert.Equal(0, result.SplicedTurns);

        // X's own next turn still gets its tokens and its conversation.
        var clientX = new List<ChatMessage>(clientY) { [2] = new() { Role = "assistant", Content = "Quintessence" } };
        var own = store.Augment(clientX);
        Assert.Equal(new[] { 501, 502, 503 }, own.History[2].RawOutputTokens);
        Assert.Equal("client-x", own.InheritedScope);
    }

    /// <summary>
    /// repro-cross-session T1: conversation T0 called get_weather for Paris; a different
    /// client replays the same question with a Tokyo call and a Tokyo result. The Tokyo
    /// call must not be rendered as the recorded Paris call.
    /// </summary>
    [Fact]
    public void Augment_ToolCallOfAnotherConversation_IsNotSpliced()
    {
        var store = NewStore();
        var ask = new ChatMessage { Role = "user", Content = "What's the weather in Paris right now?" };
        var paris = new List<ToolCall>
        {
            new() { Name = "get_weather", Arguments = new Dictionary<string, object> { ["city"] = "Paris" } },
        };
        store.Record(new List<ChatMessage> { ask },
            Generated("<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>", new List<int> { 7, 8, 9 }),
            Emitted(string.Empty, rawText: "<tool_call>...</tool_call>", toolCalls: paris), scope: "t0");

        var tokyoClient = new List<ChatMessage>
        {
            new() { Role = "user", Content = ask.Content },
            new()
            {
                Role = "assistant",
                Content = string.Empty,
                ToolCalls = new List<ToolCall>
                {
                    new() { Id = "call_1", Name = "get_weather", Arguments = new Dictionary<string, object> { ["city"] = "Tokyo" } },
                },
            },
            new() { Role = "tool", Content = "{\"city\":\"Tokyo\",\"temp_c\":31}", ToolCallId = "call_1" },
            new() { Role = "user", Content = "Which city did you look up?" },
        };

        var result = store.Augment(tokyoClient);
        Assert.Null(result.History[1].RawOutputTokens);
        Assert.Null(result.InheritedScope);

        // The Paris conversation's own replay (the client echoes the call it was sent,
        // with its own id and JSON spelling) still splices.
        tokyoClient[1] = new ChatMessage
        {
            Role = "assistant",
            Content = null,
            ToolCalls = new List<ToolCall>
            {
                new()
                {
                    Id = "call_9",
                    Name = "get_weather",
                    Arguments = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>("{ \"city\" : \"Paris\" }"),
                },
            },
        };
        var own = store.Augment(tokyoClient);
        Assert.Equal(new[] { 7, 8, 9 }, own.History[1].RawOutputTokens);
        Assert.Equal("t0", own.InheritedScope);
    }

    [Fact]
    public void Augment_UserMessageEdited_StopsSplicingAtEdit()
    {
        var store = NewStore();
        store.Record(new List<ChatMessage> { new() { Role = "user", Content = "ORIGINAL" } },
            Generated("<think>x</think>response_to_original", new List<int> { 1, 2, 3 }),
            Emitted("response_to_original"), scope: "s");

        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "EDITED" },
            new() { Role = "assistant", Content = "response_to_original" },
            new() { Role = "user", Content = "follow-up" },
        };

        var result = store.Augment(incoming);

        Assert.Null(result.History[1].RawOutputTokens);
    }

    /// <summary>
    /// Every earlier assistant turn keeps its raw tokens, however many turns later: each
    /// turn is its own record, keyed by the client-visible history before it.
    /// </summary>
    [Fact]
    public void Augment_ThreeTurnsViaWebUIFlow_AllPriorAssistantsCarryRawTokens()
    {
        var store = NewStore();
        var raw1 = new List<int> { 11, 12, 13 };
        var raw2 = new List<int> { 21, 22, 23, 24 };

        var turn1 = new List<ChatMessage> { new() { Role = "user", Content = "Q1" } };
        store.Record(turn1, Generated("RAW1", raw1), Emitted("PARSED1", rawText: "RAW1"), scope: "s");

        var turn2 = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Q1" },
            new() { Role = "assistant", Content = "PARSED1" },
            new() { Role = "user", Content = "Q2" },
        };
        Assert.Same(raw1, store.Augment(turn2).History[1].RawOutputTokens);
        store.Record(turn2, Generated("RAW2", raw2), Emitted("PARSED2", rawText: "RAW2"), scope: "s");

        var turn3 = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Q1" },
            new() { Role = "assistant", Content = "PARSED1" },
            new() { Role = "user", Content = "Q2" },
            new() { Role = "assistant", Content = "PARSED2" },
            new() { Role = "user", Content = "Q3" },
        };
        var result = store.Augment(turn3);

        Assert.Same(raw1, result.History[1].RawOutputTokens);
        Assert.Same(raw2, result.History[3].RawOutputTokens);
        Assert.Null(result.History[4].RawOutputTokens);
        Assert.Equal("s", result.InheritedScope);
    }

    /// <summary>
    /// Two OpenAI conversations interleaved on the one stateless session (A1, B1, A2, B2,
    /// A3). The old tracked history was overwritten by whichever request finished last, so
    /// every conversation but the last re-rendered its answers from text (Qwen: U2 536/583
    /// against V2 567/585). Records are per conversation position now.
    /// </summary>
    [Fact]
    public void Augment_InterleavedConversations_EachKeepsItsOwnRawTokens()
    {
        var store = NewStore();
        var sys = new ChatMessage { Role = "system", Content = "shared system prompt" };
        List<ChatMessage> History(string who, int turns)
        {
            var h = new List<ChatMessage> { sys };
            for (int t = 1; t <= turns; t++)
            {
                h.Add(new ChatMessage { Role = "user", Content = $"{who} question {t}" });
                if (t < turns) h.Add(new ChatMessage { Role = "assistant", Content = $"{who} answer {t}" });
            }
            return h;
        }
        List<int> Raw(string who, int t) => new() { who == "A" ? 1000 + t : 2000 + t };

        foreach (var (who, turn) in new[] { ("A", 1), ("B", 1), ("A", 2), ("B", 2), ("A", 3) })
        {
            var history = History(who, turn);
            var augmented = store.Augment(history);
            for (int t = 1; t < turn; t++)
                Assert.Equal(Raw(who, t), augmented.History[2 * t].RawOutputTokens);
            if (turn > 1)
                Assert.Equal("scope-" + who, augmented.InheritedScope);
            store.Record(history, Generated($"{who} answer {turn}", Raw(who, turn)),
                Emitted($"{who} answer {turn}"), scope: "scope-" + who);
        }

        var b3 = store.Augment(History("B", 3));
        Assert.Equal(Raw("B", 1), b3.History[2].RawOutputTokens);
        Assert.Equal(Raw("B", 2), b3.History[4].RawOutputTokens);
    }

    [Fact]
    public void Augment_StoppedTurn_SplicesWhatTheClientKeptBeforeTheStop()
    {
        var store = NewStore();
        store.Record(new List<ChatMessage> { new() { Role = "user", Content = "write a long story" } },
            Generated("Once upon a time there was a small robot who loved", new List<int> { 5, 6, 7, 8 }),
            Emitted("Once upon a time there was a small robot who loved", cancelled: true), scope: "s");

        var result = store.Augment(new List<ChatMessage>
        {
            new() { Role = "user", Content = "write a long story" },
            new() { Role = "assistant", Content = "Once upon a time there was a small robot who lov" },
            new() { Role = "user", Content = "go on" },
        });

        Assert.Equal(new[] { 5, 6, 7, 8 }, result.History[1].RawOutputTokens);
    }

    /// <summary>
    /// A file-backed attachment (a CSV the tool loop reads from the workspace) is not in
    /// the message content, and a recorded tool transcript's results were computed from
    /// its bytes. Another conversation that sends the same words and the same final
    /// answer with a DIFFERENT file must not get that transcript - its tool output - nor
    /// that conversation's scope; the same bytes under another upload name still match.
    /// </summary>
    [Fact]
    public void Augment_AttachedFileWithOtherContent_IsNotSpliced()
    {
        string dir = Path.Combine(Path.GetTempPath(), "ts-transcript-att-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string mine = Path.Combine(dir, "a.csv"), theirs = Path.Combine(dir, "b.csv"), copy = Path.Combine(dir, "c.csv");
            File.WriteAllText(mine, "name,salary\nalice,100\n");
            File.WriteAllText(theirs, "name,salary\nbob,999\n");
            File.WriteAllText(copy, "name,salary\nalice,100\n");
            ChatMessage Ask(string path) => new()
            {
                Role = "user", Content = "chart data.csv",
                TextFilePaths = new List<string> { path }, AttachmentPaths = new List<string> { path },
            };

            var store = NewStore();
            store.Record(new List<ChatMessage>
                {
                    Ask(mine),
                    new() { Role = "assistant", Content = "", RawOutputTokens = new List<int> { 5 } },
                    new() { Role = "tool", Content = "alice 100" },
                },
                Generated("Here is your chart.", new List<int> { 6 }), Emitted("Here is your chart."), scope: "owner");

            List<ChatMessage> Next(string path) => new()
            {
                Ask(path),
                new() { Role = "assistant", Content = "Here is your chart." },
                new() { Role = "user", Content = "what did the tool print?" },
            };

            var other = store.Augment(Next(theirs));
            Assert.Equal(3, other.History.Count);
            Assert.DoesNotContain(other.History, m => m.Role == "tool");
            Assert.Null(other.InheritedScope);

            var same = store.Augment(Next(copy));
            Assert.Equal(5, same.History.Count);
            Assert.Equal("owner", same.InheritedScope);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>The store's budget bounds what it holds: a recorded tool transcript's
    /// tool results count, not only the assistant rounds' raw tokens.</summary>
    [Fact]
    public void Record_ToolResultsCountTowardTheBudget()
    {
        var store = new ConversationTranscriptStore(maxChains: 64, maxTokens: 15_000);
        for (int i = 0; i < 2; i++)
        {
            store.Record(new List<ChatMessage>
                {
                    new() { Role = "user", Content = "read file " + i },
                    new() { Role = "assistant", Content = "", RawOutputTokens = new List<int> { 5 } },
                    new() { Role = "tool", Content = new string('x', 40_000) },
                },
                Generated("Done.", new List<int> { 6 }), Emitted("Done."), scope: "s" + i);
        }

        // Each transcript holds ~10k tokens of tool output: the older one is evicted.
        Assert.Equal(1, store.Count);
    }

    [Fact]
    public void BuildEmittedTurn_UsesTheFamilyParser_AndKeepsTheRawText()
    {
        EmittedAssistantTurn emitted = ChatGenerationPipeline.BuildEmittedTurn(
            "qwen35", "<think>\nreasoning\n</think>\n\nThe answer.", enableThinking: true, tools: null,
            generationSuffix: null, cancelled: false);

        Assert.Equal("The answer.", emitted.Content.Trim());
        Assert.Contains("reasoning", emitted.Thinking);
        Assert.Equal("<think>\nreasoning\n</think>\n\nThe answer.", emitted.RawText);
    }
}
