// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

namespace InferenceWeb.Tests;

/// <summary>Builds recorded assistant turns for tests of the transcript store.</summary>
internal static class TranscriptTestHelper
{
    /// <summary>Record one generated turn in <paramref name="session"/>: a user message
    /// <paramref name="userContent"/> answered with raw tokens.</summary>
    public static void RecordTurn(ChatSession session, string userContent)
    {
        lock (session.HistoryLock)
        {
            session.Transcripts.Record(
                new List<ChatMessage> { new() { Role = "user", Content = userContent } },
                Generated("answer to " + userContent, new List<int> { 1, 2, 3 }),
                Emitted("answer to " + userContent),
                scope: "test");
        }
    }

    public static ChatMessage Generated(string rawText, List<int> tokens, string trailingWhitespace = null) => new()
    {
        Role = "assistant",
        Content = rawText,
        RawOutputTokens = tokens,
        RawPromptTrailingWhitespace = trailingWhitespace,
    };

    public static EmittedAssistantTurn Emitted(
        string content, string rawText = null, List<ToolCall> toolCalls = null, string thinking = null,
        bool cancelled = false)
        => new(content, toolCalls, thinking, rawText ?? content, cancelled);
}
