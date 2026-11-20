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
using System.Text.RegularExpressions;
using TensorSharp.Chat;

namespace InferenceWeb.Tests;

/// <summary>
/// The Web UI's stats line under a finished answer. For a tool-using turn it used to
/// count only the answer's pieces and divide by the whole turn's time: "464 tokens ·
/// 226.7s · 2.0 tok/s" for a Qwen3.8 27B turn whose cache grew by about 13,800 tokens.
/// </summary>
public class WebUiTurnStatsTests
{
    [Fact]
    public void TheEnginesCountsWin_OverTheAnswerPiecesTheTurnStreamed()
    {
        // A tool-using turn: 9,000 tokens generated over its rounds, 225 s of them
        // decoding, 464 answer pieces streamed, 226.7 s on the wall clock.
        var (tokens, perSecond) = WebUiTurnStats.Summarize(
            evalTokens: 9_000, evalNs: 225_000_000_000, streamedPieces: 464, elapsedSeconds: 226.7);

        Assert.Equal(9_000, tokens);
        Assert.Equal(40.0, perSecond, precision: 6);
    }

    [Fact]
    public void TheSpeedIsTheDecodeSpeed_NotTheTurnsWallClock()
    {
        // A long tool run inside the turn adds wall-clock seconds, not decode time.
        var (_, perSecond) = WebUiTurnStats.Summarize(
            evalTokens: 1_200, evalNs: 20_000_000_000, streamedPieces: 300, elapsedSeconds: 120);

        Assert.Equal(60.0, perSecond, precision: 6);
    }

    [Fact]
    public void WithoutADecodeTime_TheCountIsSpreadOverTheWallClock()
    {
        var (tokens, perSecond) = WebUiTurnStats.Summarize(
            evalTokens: 500, evalNs: 0, streamedPieces: 480, elapsedSeconds: 10);

        Assert.Equal(500, tokens);
        Assert.Equal(50.0, perSecond, precision: 6);
    }

    [Fact]
    public void AnAbortedTurn_FallsBackToWhatItStreamed()
    {
        // No terminal update arrived, so the engine's counts are zero.
        var (tokens, perSecond) = WebUiTurnStats.Summarize(
            evalTokens: 0, evalNs: 0, streamedPieces: 120, elapsedSeconds: 4);

        Assert.Equal(120, tokens);
        Assert.Equal(30.0, perSecond, precision: 6);
    }

    [Fact]
    public void AnEmptyTurn_ReportsZeroRatherThanDividingByZero()
    {
        Assert.Equal((0, 0.0), WebUiTurnStats.Summarize(0, 0, 0, 0));
        Assert.Equal((0, 0.0), WebUiTurnStats.Summarize(0, 0, 0, 3.5));
    }

    /// <summary>
    /// Both places that read a terminal update -- the turn itself and the retry
    /// without thinking -- must add that generation to the totals the frame reports,
    /// and the totals must reach it. A source check, like NoAnswerPlaceholderTests,
    /// because reaching the retry needs a model that reasons past its budget.
    /// </summary>
    [Fact]
    public void EveryTerminalUpdate_AddsItsGenerationToTheTurnTotals()
    {
        string source = ChatServiceSource();

        var handlers = Regex.Matches(source, @"turnPromptTokens = update\.PromptTokens;(?<body>[\s\S]{0,1500}?)continue;");
        Assert.True(handlers.Count >= 2, $"expected the turn's and the retry's terminal handlers, found {handlers.Count}");
        foreach (Match handler in handlers)
        {
            Assert.Contains("turnEvalTokens += update.EvalTokens;", handler.Groups["body"].Value, StringComparison.Ordinal);
            Assert.Contains("turnEvalNs += update.EvalNs;", handler.Groups["body"].Value, StringComparison.Ordinal);
        }

        // The totals reach the final frames (a spoken turn's call also passes its audio and materializes the frames).
        Assert.Matches(@"turnRepetitionExplained,\s*turnEvalTokens,\s*turnEvalNs(?:,\s*knowledge)?(?:,\s*spoken:\s*\w+)?\)(?:\)|\.ToList\(\))", source);
        Assert.Contains("WebUiTurnStats.Summarize(evalTokens, evalNs, tokenCount, sw.Elapsed.TotalSeconds)", source, StringComparison.Ordinal);
    }

    private static string ChatServiceSource()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        Assert.NotNull(dir);
        return File.ReadAllText(Path.Combine(dir!.FullName, "TensorSharp.Chat", "WebUiChatService.cs"));
    }
}
