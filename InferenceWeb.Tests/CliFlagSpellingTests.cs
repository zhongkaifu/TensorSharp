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
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp.Cli;
using TensorSharp.Server.Host.Hosting;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// One spelling per option, across both hosts. A config file's keys ARE flags and one
/// file is expected to drive either host, so a name only one host knows is a key the
/// other silently drops (the CLI has no unknown-flag trap) or refuses to start on (the
/// server does). The penalty window was exactly that: <c>--penalty-last-n</c> on the
/// CLI, <c>--repeat-last-n</c> on the server, <c>repeat_last_n</c> in a request, and
/// <c>config/qwen3.5-9b-uncensored-q8.json</c>'s window reached only the server.
/// </summary>
public sealed class CliFlagSpellingTests : IDisposable
{
    private readonly string _dir;

    public CliFlagSpellingTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "ts-cli-flag-spelling-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    // ---- the sampling family ------------------------------------------------------------

    /// <summary>Every sampling flag, with a value it accepts.</summary>
    public static IEnumerable<object[]> SamplingFlags() => new[]
    {
        new object[] { "--temperature", "0.7" },
        new object[] { "--top-k", "40" },
        new object[] { "--top-p", "0.9" },
        new object[] { "--min-p", "0.05" },
        new object[] { "--repeat-penalty", "1.1" },
        new object[] { "--repeat-last-n", "128" },
        new object[] { "--presence-penalty", "0.2" },
        new object[] { "--frequency-penalty", "0.3" },
        new object[] { "--seed", "42" },
        new object[] { "--stop", "</s>" },
    };

    [Theory]
    [MemberData(nameof(SamplingFlags))]
    public void EverySamplingFlag_IsOneSpellingBothHostsAcceptAndDocument(string flag, string value)
    {
        var cfg = SamplingConfig.Greedy;
        var pinned = SamplingFields.None;
        string[] args = { flag, value, "--model", "m.gguf" };
        int i = 0;

        Assert.True(TensorSharp.Cli.Program.TryParseSamplingFlag(args, ref i, cfg, ref pinned),
            $"the CLI does not accept {flag}");
        Assert.Equal(1, i);   // consumed its value and nothing after it
        Assert.Contains(flag, CliUsage.DocumentedFlags());
        Assert.Contains(flag, ServerUsage.DocumentedFlags());
    }

    [Fact]
    public void RepeatLastN_SetsTheWindowAndPinsItAgainstChatDefaults()
    {
        var cfg = SamplingConfig.Greedy;
        var pinned = SamplingFields.None;
        int i = 0;

        Assert.True(TensorSharp.Cli.Program.TryParseSamplingFlag(
            new[] { "--repeat-last-n", "128" }, ref i, cfg, ref pinned));

        Assert.Equal(128, cfg.PenaltyLastN);
        Assert.Equal(SamplingFields.PenaltyLastN, pinned);

        // Pinned means the interactive chat defaults (a 64-token window) leave it alone.
        TensorSharp.Cli.Program.ResolveChatSamplingDefaults(cfg, ref pinned, model: null);
        Assert.Equal(128, cfg.PenaltyLastN);
    }

    [Theory]
    [InlineData("--penalty-last-n")]
    [InlineData("--model")]
    [InlineData("--repeat-last")]
    public void TryParseSamplingFlag_AnythingElse_IsLeftUntouched(string flag)
    {
        var cfg = SamplingConfig.Greedy;
        int before = cfg.PenaltyLastN;
        var pinned = SamplingFields.None;
        int i = 0;

        Assert.False(TensorSharp.Cli.Program.TryParseSamplingFlag(
            new[] { flag, "128" }, ref i, cfg, ref pinned));

        Assert.Equal(0, i);
        Assert.Equal(before, cfg.PenaltyLastN);
        Assert.Equal(SamplingFields.None, pinned);
    }

    // ---- the retired spelling -----------------------------------------------------------

    [Theory]
    [InlineData("--penalty-last-n")]
    [InlineData("--penalty-last-n=128")]
    [InlineData("--PENALTY-LAST-N")]
    [InlineData("penalty-last-n")]   // a bare config-file key
    public void PenaltyLastN_IsRefusedByNamingTheSurvivor(string arg)
    {
        string? message = RemovedCliFlags.Describe(arg);

        Assert.NotNull(message);
        Assert.StartsWith("--penalty-last-n was removed:", message, StringComparison.Ordinal);
        Assert.Contains("--repeat-last-n", message, StringComparison.Ordinal);
        // The survivor is a live option on both hosts, never itself reported as removed.
        Assert.Null(RemovedCliFlags.Describe("--repeat-last-n"));
    }

    [Fact]
    public void PenaltyLastN_OnTheCommandLine_IsAHardErrorRatherThanADroppedFlag()
    {
        // The CLI's switch drops a flag it has no case for, so a script still passing the
        // old name would lose its window without a word. MainCore calls RejectRemoved first.
        var ex = Assert.Throws<ArgumentException>(() =>
            RemovedCliFlags.RejectRemoved(new[] { "--model", "m.gguf", "--penalty-last-n", "128" }));

        Assert.Equal(RemovedCliFlags.Describe("--penalty-last-n"), ex.Message);
    }

    [Fact]
    public void PenaltyLastN_AsAConfigKey_IsRefusedWithTheSameMessage()
    {
        string cfg = Path.Combine(_dir, "stale-window.json");
        File.WriteAllText(cfg, "{ \"backend\": \"ggml_cpu\", \"penalty-last-n\": 128 }");

        var ex = Assert.Throws<ArgumentException>(() =>
            ConfigFileArgs.Expand(new[] { "--config", cfg }, TextWriter.Null, interactiveProgress: false));

        Assert.Contains(RemovedCliFlags.Describe("--penalty-last-n")!, ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RepeatLastN_AsAConfigKey_ReachesTheCommandLineUnchanged()
    {
        // The shipped qwen3.5-9b-uncensored-q8.json spells the window this way.
        string cfg = Path.Combine(_dir, "window.json");
        File.WriteAllText(cfg, "{ \"repeat-last-n\": 128 }");

        string[] expanded = ConfigFileArgs.Expand(new[] { "--config", cfg }, TextWriter.Null, interactiveProgress: false);

        Assert.Equal(new[] { "--repeat-last-n", "128" }, expanded);
    }

    [Fact]
    public void CliUsage_DocumentsOnlyTheSurvivingWindowSpelling()
    {
        var documented = CliUsage.DocumentedFlags().ToList();

        Assert.Contains("--repeat-last-n", documented);
        Assert.DoesNotContain("--penalty-last-n", documented);
    }

    // ---- JSONL request lines --------------------------------------------------------------

    [Fact]
    public void JsonlLine_KeepsTheCommandLineWindow_WhenItSetsOtherSamplingFields()
    {
        // The per-line config used to be rebuilt without the window, so any line that set
        // a temperature quietly fell back to the 64-token default.
        var fallback = SamplingConfig.Greedy;
        fallback.PenaltyLastN = 128;
        using var doc = JsonDocument.Parse("{ \"temperature\": 0.5 }");

        var cfg = TensorSharp.Cli.Program.ParseSamplingFromJson(doc.RootElement, fallback);

        Assert.Equal(0.5f, cfg.Temperature);
        Assert.Equal(128, cfg.PenaltyLastN);
    }

    [Fact]
    public void JsonlLine_RepeatLastN_SetsTheWindowForThatLine()
    {
        var fallback = SamplingConfig.Greedy;
        fallback.PenaltyLastN = 128;
        using var doc = JsonDocument.Parse("{ \"repeat_last_n\": 32 }");

        var cfg = TensorSharp.Cli.Program.ParseSamplingFromJson(doc.RootElement, fallback);

        Assert.Equal(32, cfg.PenaltyLastN);
        Assert.Equal(128, fallback.PenaltyLastN);   // the fallback itself is never edited
    }

    // ---- --mmproj none --------------------------------------------------------------------

    [Theory]
    [InlineData("none", true)]
    [InlineData("NONE", true)]
    [InlineData("None", true)]
    [InlineData("mmproj-gemma-4-E4B-it-Q8_0.gguf", false)]
    [InlineData("none.gguf", false)]
    [InlineData("", false)]
    [InlineData(null, false)]
    public void MmProjNone_DisablesTheProjectorLikeTheServer(string? value, bool disabled)
    {
        Assert.Equal(disabled, TensorSharp.Cli.Program.IsProjectorDisabled(value));
    }

    [Fact]
    public void CliUsage_DocumentsMmProjNone()
    {
        var sw = new StringWriter();
        CliUsage.PrintUsage(sw);

        Assert.Contains("--mmproj <path|none>", sw.ToString(), StringComparison.Ordinal);
        Assert.Contains("--mmproj", CliUsage.DocumentedFlags());
    }

    // ---- the companion-projector lookup ---------------------------------------------------

    [Theory]
    // A video's frames go to the vision encoder: a vision-only family must look too.
    [InlineData(false, null, null, "clip.mp4", true, false, true)]
    [InlineData(false, "p.png", null, null, true, false, true)]
    [InlineData(false, null, "a.wav", null, false, true, true)]
    [InlineData(false, null, null, "clip.mp4", false, true, true)]
    // Nothing the model can consume, or nothing attached: no lookup.
    [InlineData(false, null, "a.wav", null, true, false, false)]
    [InlineData(false, null, null, null, true, true, false)]
    // --mmproj none turns the lookup off whatever is attached.
    [InlineData(true, "p.png", "a.wav", "clip.mp4", true, true, false)]
    public void CompanionProjectorLookup_FollowsWhatTheInputsNeed(
        bool disabled, string? image, string? audio, string? video, bool vision, bool audioCapable, bool expected)
    {
        Assert.Equal(expected, TensorSharp.Cli.Program.WantsCompanionProjector(
            disabled, image, audio, video, vision, audioCapable));
    }

    // ---- one spelling reaches MainCore ----------------------------------------------------

    [Theory]
    [MemberData(nameof(SamplingFlags))]
    public void EverySamplingFlag_IsAlsoReadInTheJoinedAndUpperCaseSpellings(string flag, string value)
    {
        // The server reads --flag=value in any case; the CLI's switch reads only the spaced
        // lower-case form, so without normalisation these were silently dropped.
        foreach (string[] spelling in new[]
                 {
                     new[] { flag + "=" + value },
                     new[] { flag.ToUpperInvariant(), value },
                     new[] { flag.ToUpperInvariant() + "=" + value },
                 })
        {
            string[] normalized = CliUsage.NormalizeOptionSpellings(spelling);
            Assert.Equal(new[] { flag, value }, normalized);

            var cfg = SamplingConfig.Greedy;
            var pinned = SamplingFields.None;
            int i = 0;
            Assert.True(TensorSharp.Cli.Program.TryParseSamplingFlag(normalized, ref i, cfg, ref pinned));
        }
    }

    [Fact]
    public void Normalization_RewritesOnlyDocumentedOptions_AndNeverAValue()
    {
        string[] args =
        {
            "--Model=m.gguf", "--system", "--model is not a flag here", "--port=5000",
            "--wan-vae=vae.safetensors", "-i", "--THINK", "prompt",
        };

        Assert.Equal(
            new[]
            {
                "--model", "m.gguf", "--system", "--model is not a flag here", "--port=5000",
                "--video-vae", "vae.safetensors", "-i", "--think", "prompt",
            },
            CliUsage.NormalizeOptionSpellings(args));
    }

    [Theory]
    [InlineData("--think=on")]      // a switch never takes a value
    [InlineData("--model")]         // a value option with nothing after it
    [InlineData("--temperature")]
    public void Normalization_RefusesAMalformedOption_InsteadOfDroppingIt(string arg)
    {
        Assert.Throws<ArgumentException>(() => CliUsage.NormalizeOptionSpellings(new[] { "--input", "q.txt", arg }));
    }

    [Theory]
    [InlineData("--temperature", "warm")]
    [InlineData("--top-k", "4.5")]
    [InlineData("--seed", "--top-k")]
    public void AnUnreadableSamplingValue_IsAConfigurationError(string flag, string value)
    {
        var cfg = SamplingConfig.Greedy;
        var pinned = SamplingFields.None;
        int i = 0;

        var ex = Assert.Throws<ArgumentException>(() =>
            TensorSharp.Cli.Program.TryParseSamplingFlag(new[] { flag, value }, ref i, cfg, ref pinned));
        Assert.Contains(flag, ex.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("--backend=ggml_metal")]
    [InlineData("--BACKEND", "ggml_metal")]
    public void ACommandLineOverrideTheConfigDrops_IsOneTheCliReads(params string[] overrideArgs)
    {
        // The --config override counts --flag=value in any case as setting the option and
        // drops the file's entry; the CLI must then read that same token, or neither value
        // applies (the run silently fell back to the default backend).
        string cfg = Path.Combine(_dir, "backend.json");
        File.WriteAllText(cfg, "{ \"backend\": \"ggml_cpu\" }");

        string[] expanded = ConfigFileArgs.Expand(
            new[] { "--config", cfg }.Concat(overrideArgs).ToArray(), TextWriter.Null, interactiveProgress: false);

        Assert.Equal(new[] { "--backend", "ggml_metal" }, CliUsage.NormalizeOptionSpellings(expanded));
    }

    [Fact]
    public void MainCore_ReadsTheSamplingFlagsAndHonoursMmProjNone()
    {
        // MainCore is one long private method with no seam; these calls are what connect it
        // to the helpers tested above, so their removal must fail a test.
        string source = File.ReadAllText(Path.Combine(FindRepoRoot(), "TensorSharp.Cli", "Program.cs"));

        Assert.Contains("TryParseSamplingFlag(args, ref i, samplingConfig, ref pinnedSampling)", source, StringComparison.Ordinal);
        Assert.Contains("bool mmProjDisabled = IsProjectorDisabled(mmProjPath);", source, StringComparison.Ordinal);
        Assert.Contains("WantsCompanionProjector(mmProjDisabled,", source, StringComparison.Ordinal);
        Assert.Contains("args = CliUsage.NormalizeOptionSpellings(args);", source, StringComparison.Ordinal);
    }

    // ---- /backend ------------------------------------------------------------------------

    [Fact]
    public void EveryBackendTheSessionLists_IsOneItAccepts()
    {
        Assert.Contains("ggml_vulkan", InteractiveSession.BackendNames);
        foreach (string name in InteractiveSession.BackendNames)
            Assert.True(InteractiveSession.TryParseBackend(name, out _), $"/backend lists '{name}' but refuses it");
    }

    private static string FindRepoRoot()
    {
        var here = new DirectoryInfo(AppContext.BaseDirectory);
        while (here != null && !Directory.Exists(Path.Combine(here.FullName, "TensorSharp.Runtime")))
            here = here.Parent;
        return here?.FullName ?? throw new DirectoryNotFoundException("no repository root above " + AppContext.BaseDirectory);
    }
}
