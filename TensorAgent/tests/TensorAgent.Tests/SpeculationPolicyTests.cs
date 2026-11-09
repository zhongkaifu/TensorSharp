// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorAgent.Core;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;
using TensorSharp.Runtime.Speculative;

namespace TensorAgent.Tests;

/// <summary>
/// The speculative-decoding setting and the downloaded draft head have to REACH
/// the engine, through the same environment the loader and the scheduler read.
/// </summary>
[Collection(ProcessEnvironmentCollection.Name)]
public sealed class SpeculationPolicyTests : IDisposable
{
    private readonly string? _savedEnabled = Environment.GetEnvironmentVariable(SpeculationPolicy.EnabledVariable);
    private readonly string? _savedType = Environment.GetEnvironmentVariable(SpeculationPolicy.TypeVariable);
    private readonly string? _savedDraft = Environment.GetEnvironmentVariable(SpeculationPolicy.DraftModelVariable);
    private readonly string? _savedDraftTokens = Environment.GetEnvironmentVariable(SpeculationPolicy.DraftTokensVariable);
    private readonly string? _savedMinDraftProb = Environment.GetEnvironmentVariable(SpeculationPolicy.MinDraftProbVariable);

    public SpeculationPolicyTests()
    {
        // The launch environment is captured once per process; another test may have
        // set these variables before this type was first touched.
        SpeculationPolicy.UseLaunchEnvironment(null, null);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(SpeculationPolicy.EnabledVariable, _savedEnabled);
        Environment.SetEnvironmentVariable(SpeculationPolicy.TypeVariable, _savedType);
        Environment.SetEnvironmentVariable(SpeculationPolicy.DraftModelVariable, _savedDraft);
        Environment.SetEnvironmentVariable(SpeculationPolicy.DraftTokensVariable, _savedDraftTokens);
        Environment.SetEnvironmentVariable(SpeculationPolicy.MinDraftProbVariable, _savedMinDraftProb);
    }

    [Fact]
    public void OnByDefault_AndTheDraftHeadIsNamedForTheLoader()
    {
        string note = SpeculationPolicy.PrepareLoad(new AppSettings(), "/models/e4b/mtp-gemma-4-E4B-it-Q8_0.gguf", launchEnabled: null);
        Assert.Equal("1", Environment.GetEnvironmentVariable(SpeculationPolicy.EnabledVariable));
        Assert.Equal("/models/e4b/mtp-gemma-4-E4B-it-Q8_0.gguf", Environment.GetEnvironmentVariable(SpeculationPolicy.DraftModelVariable));
        Assert.Contains("mtp-gemma-4-E4B-it-Q8_0.gguf", note);
    }

    [Fact]
    public void TheSettingTurnsItOff_AndAStaleDraftHeadIsCleared()
    {
        Environment.SetEnvironmentVariable(SpeculationPolicy.DraftModelVariable, "/models/old/draft.gguf");
        SpeculationPolicy.PrepareLoad(new AppSettings { SpeculativeDecoding = false }, null, launchEnabled: null);
        Assert.Equal("0", Environment.GetEnvironmentVariable(SpeculationPolicy.EnabledVariable));
        Assert.Null(Environment.GetEnvironmentVariable(SpeculationPolicy.DraftModelVariable));
    }

    [Fact]
    public void TheLaunchEnvironmentWinsOverTheSetting()
    {
        Environment.SetEnvironmentVariable(SpeculationPolicy.EnabledVariable, "0");
        SpeculationPolicy.UseLaunchEnvironment("0", null);
        string note = SpeculationPolicy.PrepareLoad(new AppSettings { SpeculativeDecoding = true }, null, launchEnabled: "0");
        Assert.Equal("0", Environment.GetEnvironmentVariable(SpeculationPolicy.EnabledVariable));
        Assert.Contains("launch environment", note);

        Assert.Equal("ngram", SpeculationPolicy.ChooseAlgorithm(draftHeadAttached: true, launchType: "ngram"));
    }

    [Fact]
    public void TheHostAppliesTheSwitchToTheEnvironmentAtOnce()
    {
        // No model is loaded in this test host, so the account says the policy waits
        // for the next load; the environment still changes immediately, which is what
        // an engine built later reads.
        string root = Path.Combine(Path.GetTempPath(), "tensoragent-spec-" + Guid.NewGuid().ToString("N"));
        var paths = new AgentPaths(Path.Combine(root, "data"), Path.Combine(root, "cache"));
        using var host = new AgentAppHost(paths);
        try
        {
            AppSettings settings = host.Settings.Load();
            settings.SpeculativeDecoding = false;
            string offAccount = host.ApplySpeculationSetting(settings);
            Assert.Contains("off", offAccount, StringComparison.Ordinal);
            Assert.Equal("0", Environment.GetEnvironmentVariable(SpeculationPolicy.EnabledVariable));

            settings.SpeculativeDecoding = true;
            string onAccount = host.ApplySpeculationSetting(settings);
            Assert.Contains("on", onAccount, StringComparison.Ordinal);
            Assert.Equal("1", Environment.GetEnvironmentVariable(SpeculationPolicy.EnabledVariable));
            Assert.Equal(SpeculatorRegistry.NGram, Environment.GetEnvironmentVariable(SpeculationPolicy.TypeVariable));
        }
        finally
        {
            try { Directory.Delete(root, recursive: true); } catch { /* best effort */ }
        }
    }

    [Theory]
    [InlineData(true, SpeculatorRegistry.Auto)]
    [InlineData(false, SpeculatorRegistry.NGram)]
    public void TheAlgorithmFollowsWhetherADraftHeadActuallyAttached(bool attached, string expected)
    {
        // "auto" with no draft head declines outright instead of falling back, so
        // a model whose optional draft file is not on the phone must ask for n-gram.
        Assert.Equal(expected, SpeculationPolicy.ChooseAlgorithm(attached, launchType: null));
        Assert.Equal(expected, Environment.GetEnvironmentVariable(SpeculationPolicy.TypeVariable));
    }

    /// <summary>
    /// Only a draft head the catalog ships makes the app speculate with a head. Qwen3.8
    /// 27B's weights file carries its own NextN/MTP layer, the engine attaches it, and
    /// with it the Mac app decoded slower in every turn shape (0.83-0.93x) where n-gram
    /// gave 1.76-1.90x on the quoting turns agents are made of.
    /// </summary>
    [Theory]
    [InlineData("/models/gemma-4-e4b/mtp-gemma-4-E4B-it-Q8_0.gguf", true, true)]
    [InlineData("/models/gemma-4-e4b/mtp-gemma-4-E4B-it-Q8_0.gguf", false, false)]
    [InlineData(null, true, false)]
    [InlineData(null, false, false)]
    [InlineData("  ", true, false)]
    public void OnlyTheCatalogsOwnDraftHeadSelectsAuto(string? catalogHead, bool attached, bool speculatesWithHead)
    {
        Assert.Equal(speculatesWithHead, SpeculationPolicy.SpeculatesWithDraftHead(catalogHead, attached));
        Assert.Equal(speculatesWithHead ? SpeculatorRegistry.Auto : SpeculatorRegistry.NGram,
            SpeculationPolicy.ChooseAlgorithm(SpeculationPolicy.SpeculatesWithDraftHead(catalogHead, attached), launchType: null));
    }

    /// <summary>
    /// Qwen3.8 27B's DFlash2 drafter measured fastest at 7 drafts with every one verified
    /// (config/qwen3.8-27b-metal-dflash2.json). The app hands those to the engine the way
    /// the CLI's --spec-draft / --spec-pmin do, only while that drafter speculates, and
    /// clears them for the next model so they never reach n-gram or another drafter.
    /// </summary>
    [Fact]
    public void TheEntrysDraftTuningReachesTheEngineOnlyWhileItsDraftSpeculates()
    {
        var tuning = new CatalogDraftTuning(MaxDraftTokens: 7, MinDraftProb: 0f);

        string note = SpeculationPolicy.ApplyDraftTuning(tuning, catalogDraftSpeculates: true,
            launchDraftTokens: null, launchMinDraftProb: null);
        Assert.Equal("7", Environment.GetEnvironmentVariable(SpeculationPolicy.DraftTokensVariable));
        Assert.Equal("0", Environment.GetEnvironmentVariable(SpeculationPolicy.MinDraftProbVariable));
        Assert.Equal("draft window 7, confidence gate 0", note);
        SpeculationOptions options = SpeculationOptions.FromEnvironment();
        Assert.Equal(7, options.MaxDraftTokens);
        Assert.True(options.MaxDraftTokensExplicit);
        Assert.Equal(0f, options.MinDraftProb);

        // The drafter did not attach (n-gram instead): the defaults, not the tuning.
        SpeculationPolicy.ApplyDraftTuning(tuning, catalogDraftSpeculates: false,
            launchDraftTokens: null, launchMinDraftProb: null);
        Assert.Null(Environment.GetEnvironmentVariable(SpeculationPolicy.DraftTokensVariable));
        Assert.Null(Environment.GetEnvironmentVariable(SpeculationPolicy.MinDraftProbVariable));

        // The next model declares none: what the previous one set is cleared.
        SpeculationPolicy.ApplyDraftTuning(tuning, catalogDraftSpeculates: true,
            launchDraftTokens: null, launchMinDraftProb: null);
        string defaults = SpeculationPolicy.ApplyDraftTuning(null, catalogDraftSpeculates: true,
            launchDraftTokens: null, launchMinDraftProb: null);
        Assert.Null(Environment.GetEnvironmentVariable(SpeculationPolicy.DraftTokensVariable));
        Assert.Null(Environment.GetEnvironmentVariable(SpeculationPolicy.MinDraftProbVariable));
        Assert.Equal("draft window default, confidence gate default", defaults);
    }

    [Fact]
    public void TheLaunchEnvironmentWinsOverTheEntrysDraftTuning()
    {
        Environment.SetEnvironmentVariable(SpeculationPolicy.DraftTokensVariable, "4");
        Environment.SetEnvironmentVariable(SpeculationPolicy.MinDraftProbVariable, "0.5");
        SpeculationPolicy.ApplyDraftTuning(new CatalogDraftTuning(7, 0f), catalogDraftSpeculates: true,
            launchDraftTokens: "4", launchMinDraftProb: "0.5");
        Assert.Equal("4", Environment.GetEnvironmentVariable(SpeculationPolicy.DraftTokensVariable));
        Assert.Equal("0.5", Environment.GetEnvironmentVariable(SpeculationPolicy.MinDraftProbVariable));

        // And a developer's value is not cleared when the next model declares nothing.
        SpeculationPolicy.ApplyDraftTuning(null, catalogDraftSpeculates: false,
            launchDraftTokens: "4", launchMinDraftProb: "0.5");
        Assert.Equal("4", Environment.GetEnvironmentVariable(SpeculationPolicy.DraftTokensVariable));
        Assert.Equal("0.5", Environment.GetEnvironmentVariable(SpeculationPolicy.MinDraftProbVariable));
    }
}
