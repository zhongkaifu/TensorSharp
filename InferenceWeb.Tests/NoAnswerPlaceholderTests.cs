// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.IO;
using System.Linq;

namespace InferenceWeb.Tests;

/// <summary>
/// "The model ended this turn without writing an answer" is an accusation, and this
/// codebase's rule is that every message accusing the model has to be right.
///
/// <para>
/// It used to be decided by one flag, <c>sawContent</c>, and three different places stream
/// answer text: the ordinary parse loop, the parser's final flush, and the retry that runs
/// a truncated turn again with thinking off. Only the first set the flag. Recorded
/// 2026-09-10: a turn that produced a ten-slide deck, described it in ten numbered lines
/// and gave the user a download link ended with a sentence telling them nothing had been
/// written — the placeholder accusing the model of exactly what the retry had just fixed.
/// </para>
/// <para>
/// It is now decided by the answer the page is left showing, <c>visibleAnswer</c>, which
/// only <c>AnswerFrames</c> writes: text that was shown and then taken back as reasoning
/// (Nemotron-H Reasoning-128K closing a block its thinking-off prompt had closed) no longer
/// counts as an answer. So every path that streams answer text has to go through it.
/// </para>
/// <para>
/// A source check rather than a behavioural one because reaching those branches needs a
/// model that truncates inside its own reasoning, which no unit test can stage. It is the
/// same shape as the process-creation guards elsewhere in this suite: it fails when
/// someone adds a fourth way to stream an answer and forgets the flag.
/// </para>
/// </summary>
public class NoAnswerPlaceholderTests
{
    private static string Source()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        Assert.NotNull(dir);
        string path = Path.Combine(dir!.FullName, "TensorSharp.Chat", "WebUiChatService.cs");
        Assert.True(File.Exists(path), path);
        return File.ReadAllText(path);
    }

    /// <summary>The region between two anchors, so the assertion is about one branch.</summary>
    private static string Between(string source, string start, string end)
    {
        int from = source.IndexOf(start, StringComparison.Ordinal);
        Assert.True(from >= 0, $"anchor not found: {start}");
        int to = source.IndexOf(end, from, StringComparison.Ordinal);
        Assert.True(to > from, $"anchor not found after the first: {end}");
        return source[from..to];
    }

    [Fact]
    public void TheRetryWithoutThinking_CountsWhatItStreamsAsAnAnswer()
    {
        string retry = Between(
            Source(),
            "if (reasonedPastItsBudget && tokenCount == 0",
            "if (retryCompleted)");

        Assert.Contains("AnswerFrames(", retry, StringComparison.Ordinal);
        Assert.Contains("visibleAnswer, update.Piece", retry, StringComparison.Ordinal);
    }

    /// <summary>No answer text reaches the page except through <c>AnswerFrames</c>, which
    /// keeps the record the placeholder is decided by.</summary>
    [Fact]
    public void EveryStreamedAnswer_GoesThroughTheVisibleAnswer()
    {
        string turn = Between(
            Source(),
            "IOutputParser uiParser = null;",
            "foreach (object frame in FinalFrames(");

        Assert.DoesNotContain("WebUiSseEvents.Token(", turn, StringComparison.Ordinal);
        Assert.DoesNotContain("WebUiSseEvents.Replace(", turn, StringComparison.Ordinal);
        Assert.True(turn.Split("AnswerFrames(").Length - 1 >= 4, "a streaming path no longer goes through AnswerFrames");
    }

    /// <summary>
    /// The retry is only worth its cost for the failure it was written for.
    ///
    /// <para>
    /// Flipping thinking off re-renders the conversation from the first system
    /// block (the shipped Qwen template puts its reasoning-effort paragraph there),
    /// so the prompt diverges at token 3 and the whole conversation re-prefills -
    /// 160 s on a 36k-token chat, measured 2026-09-10. Gating on `turnTruncated`
    /// spent that on any truncated turn, including the startup prefix warm-up,
    /// which asks for ONE token with thinking on and therefore always stops on
    /// `max_tokens` with no content.
    /// </para>
    /// </summary>
    [Fact]
    public void TheRetryWithoutThinking_FiresOnlyForAThinkingBudgetStop()
    {
        string source = Source();

        // The gate reads the finish reason, not the generic truncation flag.
        Assert.Contains(
            "bool reasonedPastItsBudget = string.Equals(",
            source,
            StringComparison.Ordinal);
        Assert.Contains(
            "turnFinishReason, FinishReasonMapper.PipelineThinkingBudget, StringComparison.Ordinal);",
            source,
            StringComparison.Ordinal);
        // And the old, wider gate is gone: a plain max_tokens stop must not retry.
        Assert.DoesNotContain(
            "if (turnTruncated && tokenCount == 0",
            source,
            StringComparison.Ordinal);
    }

    [Fact]
    public void TheParsersFinalFlush_CountsWhatItStreamsAsAnAnswer()
    {
        string flush = Between(
            Source(),
            "var finalParsed = uiParser.Add(\"\", true);",
            "ended this turn without writing an answer");

        Assert.Contains("AnswerFrames(", flush, StringComparison.Ordinal);
        Assert.Contains("visibleAnswer, finalParsed.Content", flush, StringComparison.Ordinal);
    }

    /// <summary>
    /// An anti-vacuity check: the placeholder these guard still exists and is still gated
    /// on the flag, so a rename cannot leave the tests above passing over nothing.
    /// </summary>
    [Fact]
    public void ThePlaceholderIsStillGatedOnTheFlag()
    {
        string source = Source();
        Assert.Contains("ended this turn without writing an answer", source, StringComparison.Ordinal);

        string guard = Between(source, "&& IsNullOrWhiteSpace(visibleAnswer)", "ended this turn without writing an answer");
        Assert.True(guard.Length < 400, "the placeholder moved away from its guard");
    }
}
