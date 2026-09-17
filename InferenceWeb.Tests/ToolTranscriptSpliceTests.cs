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
/// Pins the transcript EXPANSION in <c>ConversationTranscriptStore.Augment</c>.
///
/// <para>
/// The incident this encodes: a turn that ran the skills/code tool loop leaves
/// <c>assistant, tool, assistant, ...</c> in the model's cache (each assistant round
/// carrying its raw output tokens), but the client sends that turn back as ONE clean
/// assistant message. A positional walk broke at the first tool result, the next render
/// diverged thousands of tokens before the live cache's end, and the engine re-prefilled
/// the whole conversation: 15.7k tokens, 10.7s to first token, 0% KV reuse. Expansion
/// substitutes the recorded transcript for the clean message so the rendered prefix stays
/// byte-identical to the cache - now only when the clean message is what the loop sent.
/// </para>
/// </summary>
public class ToolTranscriptSpliceTests
{
    private static readonly List<int> Raw1 = new() { 11, 12, 13 };
    private static readonly List<int> Raw2 = new() { 21, 22 };
    private static readonly List<int> Raw3 = new() { 31, 32, 33, 34 };

    /// <summary>What the Web UI sends back: the rounds' parsed contents, concatenated.</summary>
    private const string CleanAssistantText = "Using the pdf skill.\n\nRetrying with packages.\n\nHere is your PDF.";

    /// <summary>A store holding a skills turn as the loop's last generation records it:
    /// the request history ends with the loop's own rounds, then the final round.</summary>
    private static ConversationTranscriptStore StoreWithTranscript()
    {
        var store = new ConversationTranscriptStore(maxChains: 64, maxTokens: 100_000);
        var lastRoundHistory = new List<ChatMessage>
        {
            new() { Role = "user", Content = "convert this file" },
            new() { Role = "assistant", Content = "Using the pdf skill.\n\n", RawOutputTokens = Raw1 },
            new() { Role = "tool", Content = "Ran python (exit code 1)\nModuleNotFoundError" },
            new() { Role = "assistant", Content = "Retrying with packages.\n\n", RawOutputTokens = Raw2 },
            new() { Role = "tool", Content = "Ran python (exit code 0)\nFiles produced" },
        };
        store.Record(lastRoundHistory,
            Generated("<thought>done</thought>Here is your PDF.", Raw3),
            Emitted("Here is your PDF.", rawText: "<thought>done</thought>Here is your PDF."),
            scope: "chat");
        return store;
    }

    [Fact]
    public void ACleanAssistantTurn_ExpandsIntoTheRecordedToolTranscript()
    {
        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "convert this file" },
            new() { Role = "assistant", Content = CleanAssistantText },
            new() { Role = "user", Content = "thanks, and now translate it" },
        };

        var result = StoreWithTranscript().Augment(incoming).History;

        // 1 user + 5 transcript messages + 1 new user.
        Assert.Equal(7, result.Count);
        Assert.Equal("user", result[0].Role);
        Assert.Same(Raw1, result[1].RawOutputTokens);
        Assert.Equal("tool", result[2].Role);
        Assert.Same(Raw2, result[3].RawOutputTokens);
        Assert.Equal("tool", result[4].Role);
        Assert.Same(Raw3, result[5].RawOutputTokens);
        Assert.Equal("thanks, and now translate it", result[6].Content);
    }

    /// <summary>
    /// An OpenAI client that sends <c>reasoning_content</c> back gets the reasoning of
    /// EVERY round of a tool-loop turn, since the loop streams each round's thinking. The
    /// transcript must still expand: the record's reasoning is the rounds' reasoning, not
    /// only the final round's (which read as a conflict and re-prefilled the whole turn).
    /// </summary>
    [Theory]
    [InlineData("plan the conversion\n\nretry with packages\n\nall done")]
    [InlineData("all done")]
    public void AClientEchoingTheLoopsReasoning_StillExpands(string clientThinking)
    {
        var store = new ConversationTranscriptStore(maxChains: 64, maxTokens: 100_000);
        store.Record(new List<ChatMessage>
            {
                new() { Role = "user", Content = "convert this file" },
                new() { Role = "assistant", Content = "Using the pdf skill.\n\n", Thinking = "plan the conversion", RawOutputTokens = Raw1 },
                new() { Role = "tool", Content = "Ran python (exit code 1)" },
                new() { Role = "assistant", Content = "Retrying with packages.\n\n", Thinking = "retry with packages", RawOutputTokens = Raw2 },
                new() { Role = "tool", Content = "Ran python (exit code 0)" },
            },
            Generated("Here is your PDF.", Raw3),
            Emitted("Here is your PDF.", thinking: "all done"),
            scope: "chat");

        var result = store.Augment(new List<ChatMessage>
        {
            new() { Role = "user", Content = "convert this file" },
            new() { Role = "assistant", Content = CleanAssistantText, Thinking = clientThinking },
            new() { Role = "user", Content = "next" },
        });

        Assert.Equal(7, result.History.Count);
        Assert.Same(Raw3, result.History[5].RawOutputTokens);
        Assert.Equal("chat", result.InheritedScope);

        // Reasoning that is not what the loop produced still refuses the splice.
        var edited = store.Augment(new List<ChatMessage>
        {
            new() { Role = "user", Content = "convert this file" },
            new() { Role = "assistant", Content = CleanAssistantText, Thinking = "something else entirely" },
            new() { Role = "user", Content = "next" },
        });
        Assert.Equal(3, edited.History.Count);
    }

    [Fact]
    public void AHostNoteAfterTheLoopsAnswer_StillExpands()
    {
        // The loop appends its own paragraph after a verified artifact; that is host
        // text, not model output, and the generated part is all there.
        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "convert this file" },
            new() { Role = "assistant", Content = CleanAssistantText + "\n\n[Download report.pdf](/files/report.pdf)" },
            new() { Role = "user", Content = "next" },
        };

        var result = StoreWithTranscript().Augment(incoming).History;

        Assert.Equal(7, result.Count);
        Assert.Same(Raw3, result[5].RawOutputTokens);
    }

    [Fact]
    public void ExpandedTurn_PreservesClientCacheMarkersAtSafeTranscriptBoundaries()
    {
        int firstRoundLength = "Using the pdf skill.\n\n".Length;
        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "convert this file" },
            new()
            {
                Role = "assistant",
                Content = CleanAssistantText,
                CacheControl = new CacheControlMarker(),
                ContentCacheBreakpoints = new List<int> { 5, firstRoundLength + 5 },
            },
            new() { Role = "user", Content = "next" },
        };

        var store = StoreWithTranscript();
        var result = store.Augment(incoming).History;

        Assert.Equal(new[] { 5 }, result[1].ContentCacheBreakpoints);
        Assert.Equal(new[] { 5 }, result[3].ContentCacheBreakpoints);
        Assert.NotNull(result[5].CacheControl);

        // The record itself is untouched by the mapping.
        var again = store.Augment(new List<ChatMessage>
        {
            new() { Role = "user", Content = "convert this file" },
            new() { Role = "assistant", Content = CleanAssistantText },
            new() { Role = "user", Content = "next" },
        }).History;
        Assert.Null(again[1].ContentCacheBreakpoints);
        Assert.Null(again[5].CacheControl);
    }

    [Fact]
    public void TheTurnAfterAnExpandedTurn_SplicesBothTurns()
    {
        var raw4 = new List<int> { 41, 42 };
        var store = StoreWithTranscript();
        var turn2 = new List<ChatMessage>
        {
            new() { Role = "user", Content = "convert this file" },
            new() { Role = "assistant", Content = CleanAssistantText },
            new() { Role = "user", Content = "thanks, and now translate it" },
        };
        store.Record(turn2, Generated("RAW turn-2 text", raw4),
            Emitted("parsed turn-2 text", rawText: "RAW turn-2 text"), scope: "chat");

        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "convert this file" },
            new() { Role = "assistant", Content = CleanAssistantText },
            new() { Role = "user", Content = "thanks, and now translate it" },
            new() { Role = "assistant", Content = "parsed turn-2 text" },
            new() { Role = "user", Content = "one more thing" },
        };

        var result = store.Augment(incoming).History;

        Assert.Equal(9, result.Count);
        Assert.Same(Raw3, result[5].RawOutputTokens);       // the transcript's final round
        Assert.Equal("user", result[6].Role);
        Assert.Same(raw4, result[7].RawOutputTokens);       // turn 2
        Assert.Equal("one more thing", result[8].Content);
    }

    [Fact]
    public void AnEditedAssistantTurn_IsNeitherExpandedNorSpliced()
    {
        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "convert this file" },
            // The user (or another client) rewrote the assistant text. The old walk fell
            // back to splicing the first round's raw tokens under it anyway; a message
            // the loop did not send renders from its own text.
            new() { Role = "assistant", Content = "Something entirely different." },
            new() { Role = "user", Content = "next" },
        };

        var result = StoreWithTranscript().Augment(incoming).History;

        Assert.Equal(3, result.Count);
        Assert.Null(result[1].RawOutputTokens);
        Assert.Equal("Something entirely different.", result[1].Content);
    }

    [Fact]
    public void AnEditedUserMessage_StopsAllSplicingAtTheEdit()
    {
        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "EDITED PROMPT" },
            new() { Role = "assistant", Content = CleanAssistantText },
            new() { Role = "user", Content = "next" },
        };

        var result = StoreWithTranscript().Augment(incoming).History;

        Assert.Equal(3, result.Count);
        Assert.Null(result[1].RawOutputTokens);
    }

    [Fact]
    public void AClientThatSendsItsOwnToolMessages_IsNotExpanded()
    {
        // A caller that carries a tool transcript itself sends a different visible
        // history than the loop's client, so the loop's record does not apply to it and
        // nothing is inserted.
        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "convert this file" },
            new() { Role = "assistant", Content = "Using the pdf skill.\n\n" },
            new() { Role = "tool", Content = "Ran python (exit code 1)\nModuleNotFoundError" },
            new() { Role = "assistant", Content = "Retrying with packages.\n\n" },
            new() { Role = "tool", Content = "Ran python (exit code 0)\nFiles produced" },
            new() { Role = "assistant", Content = "Here is your PDF." },
            new() { Role = "user", Content = "next" },
        };

        var result = StoreWithTranscript().Augment(incoming).History;

        Assert.Equal(7, result.Count);
        Assert.All(result, m => Assert.Null(m.RawOutputTokens));
    }

    [Fact]
    public void ToolResultsFedBackAsUserTurns_ExpandLikeAnyTranscript()
    {
        // Mistral 3 renders tool results as user messages. The loop's rounds are
        // identified by their raw tokens, not by role, so its transcript expands too.
        var store = new ConversationTranscriptStore(maxChains: 64, maxTokens: 100_000);
        store.Record(new List<ChatMessage>
            {
                new() { Role = "user", Content = "q" },
                new() { Role = "assistant", Content = "calling", RawOutputTokens = Raw1 },
                new() { Role = "user", Content = "Result of your shell call: ok" },
            },
            Generated("answer", Raw2), Emitted("answer"), scope: "chat");

        var result = store.Augment(new List<ChatMessage>
        {
            new() { Role = "user", Content = "q" },
            new() { Role = "assistant", Content = "callinganswer" },
            new() { Role = "user", Content = "next" },
        }).History;

        Assert.Equal(5, result.Count);
        Assert.Same(Raw1, result[1].RawOutputTokens);
        Assert.Equal("user", result[2].Role);
        Assert.Same(Raw2, result[3].RawOutputTokens);
        Assert.Equal("next", result[4].Content);
    }

    [Fact]
    public void ARoundWithSeveralToolResults_ExpandsAsOneRun()
    {
        var store = new ConversationTranscriptStore(maxChains: 64, maxTokens: 100_000);
        store.Record(new List<ChatMessage>
            {
                new() { Role = "user", Content = "q" },
                new() { Role = "assistant", Content = "reading\n", RawOutputTokens = Raw1 },
                new() { Role = "tool", Content = "file one" },
                new() { Role = "tool", Content = "file two" },
            },
            Generated("RAW answer", Raw2), Emitted("the answer", rawText: "RAW answer"), scope: "chat");

        var result = store.Augment(new List<ChatMessage>
        {
            new() { Role = "user", Content = "q" },
            new() { Role = "assistant", Content = "reading\nthe answer" },
            new() { Role = "user", Content = "next" },
        }).History;

        Assert.Equal(6, result.Count);
        Assert.Same(Raw1, result[1].RawOutputTokens);
        Assert.Equal("tool", result[2].Role);
        Assert.Equal("tool", result[3].Role);
        Assert.Same(Raw2, result[4].RawOutputTokens);
    }
}
