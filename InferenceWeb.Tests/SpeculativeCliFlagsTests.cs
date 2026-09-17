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
using TensorSharp.Cli;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The speculative-decoding flags translated to their environment variables.
/// One spelling per option (<c>--spec*</c> / <c>--draft-model</c>); the removed
/// duplicate spellings (<c>--mtp-*</c>, <c>--spec-draft-model</c>,
/// <c>--spec-draft-n-max</c>, <c>--spec-draft-conf-min</c>) hard-error with a
/// pointer to their survivor. Shared by TensorSharp.Cli and TensorSharp.Server.
/// This is the only seam either host's speculative command line can be tested
/// through: the CLI's own parser is a switch inside a private <c>MainCore</c>
/// with no return value.
///
/// The env var is the contract rather than a parsed value object because the
/// request has to reach the model LOADER — glm-dsa decides from the enabled
/// variable whether to page a whole extra 256-expert decoder layer into VRAM,
/// and sizes its graph cache from <c>TS_MTP_DRAFT</c> (still read from C++),
/// both while the model is loading.
/// </summary>
public sealed class SpeculativeCliFlagsTests : IDisposable
{
    private readonly EnvScope _env = new();

    public SpeculativeCliFlagsTests()
    {
        // Every test starts from "operator configured nothing".
        _env.ClearSpeculationVars();
    }

    public void Dispose() => _env.Dispose();

    [Fact]
    public void Apply_Spec_TurnsSpeculationOnForTheScheduler()
    {
        bool applied = SpeculativeCliFlags.Apply(new[] { "--spec" });

        Assert.True(applied);
        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_MTP_SPEC"));
        Assert.True(SchedulerConfig.FromEnvironment().Speculation.Enabled);
    }

    [Fact]
    public void Apply_NoSpec_TurnsSpeculationOffOverAnExportedEnvVar()
    {
        _env.Set("TS_MTP_SPEC", "1");

        Assert.True(SpeculativeCliFlags.Apply(new[] { "--no-spec" }));

        Assert.Equal("0", Environment.GetEnvironmentVariable("TS_MTP_SPEC"));
        Assert.False(SchedulerConfig.FromEnvironment().Speculation.Enabled);
    }

    [Fact]
    public void Apply_NoFlags_LeavesTheEnvironmentAlone()
    {
        _env.Set("TS_MTP_SPEC", "1");

        Assert.False(SpeculativeCliFlags.Apply(new[] { "--model", "x.gguf", "--backend", "ggml_cuda" }));

        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_MTP_SPEC"));
    }

    [Fact]
    public void DraftModelTestScope_RestoresArchitectureLoaderFallbacks()
    {
        string[] names = { "TS_DSV4_DSPARK", "TS_QWEN35_DFLASH", "TS_MUSE_GLIMMER_DFLASH", "TS_NEMOTRON_DFLASH" };
        foreach (string name in names)
            _env.Set(name, "/original/drafter.gguf");

        string gguf = Path.Combine(Path.GetTempPath(), $"drafter-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(gguf, new byte[] { 1, 2, 3 });
        try
        {
            using (var inner = new EnvScope())
            {
                inner.ClearSpeculationVars();
                Assert.True(TensorSharp.Server.Hosting.ServerOptionsBuilder.ApplySpeculativeCliFlags(
                    new[] { "--draft-model", gguf }));
                foreach (string name in names)
                    Assert.Equal(gguf, Environment.GetEnvironmentVariable(name));
            }
            foreach (string name in names)
                Assert.Equal("/original/drafter.gguf", Environment.GetEnvironmentVariable(name));
        }
        finally
        {
            File.Delete(gguf);
        }
    }

    [Theory]
    [InlineData("--spec-draft", "4")]
    [InlineData("--spec-draft=4", null)]
    public void Apply_SpecDraft_AcceptsBothValueSpellings(string first, string second)
    {
        string[] args = second == null ? new[] { first } : new[] { first, second };

        Assert.True(SpeculativeCliFlags.Apply(args));

        Assert.Equal("4", Environment.GetEnvironmentVariable("TS_MTP_DRAFT"));
        Assert.Equal(4, SchedulerConfig.FromEnvironment().Speculation.MaxDraftTokens);
    }

    [Fact]
    public void Apply_SpecPmin_ReachesTheSchedulerAsANullableProbability()
    {
        Assert.True(SpeculativeCliFlags.Apply(new[] { "--spec-pmin", "0.55" }));

        Assert.Equal(0.55f, SchedulerConfig.FromEnvironment().Speculation.MinDraftProb);
    }

    [Fact]
    public void Apply_SpecPminZero_MeansNeverGateAndSurvivesTheEnvRoundTrip()
    {
        // 0 is a real value ("never gate a draft on confidence") that the removed
        // --spec-draft-conf-min spelling could express; its survivor must keep
        // expressing it, all the way through the env reader that used to treat
        // 0 as unset.
        Assert.True(SpeculativeCliFlags.Apply(new[] { "--spec-pmin", "0" }));

        Assert.Equal(0f, SchedulerConfig.FromEnvironment().Speculation.MinDraftProb);
    }

    [Fact]
    public void Apply_WithoutPmin_LeavesTheGateUnsetForTheDrafterToChoose()
    {
        // The per-token gate (top-1 probability over the head's top-10 logits,
        // default 0.15) and the block gate (cumulative prefix-acceptance product,
        // default 0.35) threshold different quantities, so "unset" has to survive
        // all the way to SpeculativeExecution rather than collapsing to one
        // shared number that badly mis-gates the other kind.
        Assert.True(SpeculativeCliFlags.Apply(new[] { "--spec" }));

        Assert.Null(SchedulerConfig.FromEnvironment().Speculation.MinDraftProb);
    }

    [Theory]
    [InlineData("0")]
    [InlineData("-1")]
    [InlineData("abc")]
    // Above the bound the glm-dsa native loader silently ignores when it sizes its
    // graph cache from the same variable, so a larger window would decode through a
    // cache too small for the graph shapes it produces.
    [InlineData("65")]
    [InlineData("1000")]
    public void Apply_SpecDraftWithAnUnusableValue_FailsFastNamingTheFlag(string value)
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            SpeculativeCliFlags.Apply(new[] { "--spec-draft", value }));

        Assert.Contains("--spec-draft", ex.Message);
        Assert.Null(Environment.GetEnvironmentVariable("TS_MTP_DRAFT"));
    }

    [Theory]
    [InlineData("-0.1")]
    [InlineData("1.5")]
    [InlineData("nope")]
    [InlineData("NaN")]
    [InlineData("Infinity")]
    [InlineData("-Infinity")]
    public void Apply_SpecPminOutsideTheUnitInterval_FailsFastNamingTheFlag(string value)
    {
        // The [0, 1] bound exists ONLY here: SchedulerConfig reads the variable
        // back with a plain float parse, so a value of 5 would be accepted there
        // and would reject every draft while speculation still logged as armed.
        var ex = Assert.Throws<ArgumentException>(() =>
            SpeculativeCliFlags.Apply(new[] { "--spec-pmin", value }));

        Assert.Contains("--spec-pmin", ex.Message);
        Assert.Null(Environment.GetEnvironmentVariable("TS_MTP_PMIN"));
    }

    [Fact]
    public void Apply_SpecPminZero_IsAcceptedAsNeverDecline()
    {
        // 0 is a meaningful setting, not an out-of-range one: never decline to
        // draft. It is llama.cpp's own default for the same knob (p_min = 0.0).
        SpeculativeCliFlags.Apply(new[] { "--spec-pmin", "0" });
        Assert.Equal("0", Environment.GetEnvironmentVariable(SpeculativeCliFlags.PMinEnvVar));
    }

    [Fact]
    public void Apply_ValueFlagWithNoValue_SaysSoInsteadOfIndexingPastTheEnd()
    {
        // Every other CLI flag does args[++i] and surfaces as an unhandled
        // IndexOutOfRangeException; these say which option is missing a value.
        var ex = Assert.Throws<ArgumentException>(() =>
            SpeculativeCliFlags.Apply(new[] { "--model", "x.gguf", "--spec-draft" }));

        Assert.Contains("--spec-draft", ex.Message);
    }

    [Fact]
    public void Apply_DraftModelThatDoesNotExist_FailsFast()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            SpeculativeCliFlags.Apply(new[] { "--draft-model", "/no/such/draft.gguf" }));

        Assert.Contains("--draft-model", ex.Message);
    }

    [Fact]
    public void Apply_SpecDraftAtTheBound_IsAccepted()
    {
        Assert.True(SpeculativeCliFlags.Apply(
            new[] { "--spec-draft", SpeculativeCliFlags.MaxDraftTokens.ToString() }));

        Assert.Equal(SpeculativeCliFlags.MaxDraftTokens,
            SchedulerConfig.FromEnvironment().Speculation.MaxDraftTokens);
    }

    [Fact]
    public void Apply_LastOccurrenceWins()
    {
        // Config-file expansion (ConfigFileArgs.Expand) splices file-derived
        // tokens ahead of the real command line, so the operator's own flag has
        // to be the one that survives.
        Assert.True(SpeculativeCliFlags.Apply(new[] { "--spec", "--no-spec" }));
        Assert.False(SchedulerConfig.FromEnvironment().Speculation.Enabled);

        Assert.True(SpeculativeCliFlags.Apply(new[] { "--no-spec", "--spec" }));
        Assert.True(SchedulerConfig.FromEnvironment().Speculation.Enabled);
    }

    // ----- naming a draft file IS the request -----

    [Fact]
    public void Apply_DraftModelAlone_EnablesSpeculation()
    {
        // Nobody passes --draft-model hoping the file stays idle. Requiring a
        // separate --spec beside it was a trap both hosts fell into differently:
        // the CLI engaged, the server silently did not.
        string gguf = Path.Combine(Path.GetTempPath(), $"drafter-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(gguf, new byte[] { 1, 2, 3 });
        try
        {
            Assert.True(SpeculativeCliFlags.Apply(new[] { "--draft-model", gguf }));

            Assert.True(SchedulerConfig.FromEnvironment().Speculation.Enabled);
            Assert.Equal(gguf, Environment.GetEnvironmentVariable(SpeculationEnvVars.DraftModel));
            Assert.Equal(gguf, Environment.GetEnvironmentVariable(SpeculationEnvVars.LegacyDraftModel));
        }
        finally
        {
            File.Delete(gguf);
        }
    }

    [Fact]
    public void Apply_ExplicitNoSpec_VetoesTheDraftModelAutoEnable_InEitherOrder()
    {
        // The operator said "off" in words; a convenience must not override them.
        string gguf = Path.Combine(Path.GetTempPath(), $"drafter-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(gguf, new byte[] { 1, 2, 3 });
        try
        {
            Assert.True(SpeculativeCliFlags.Apply(new[] { "--no-spec", "--draft-model", gguf }));
            Assert.False(SchedulerConfig.FromEnvironment().Speculation.Enabled);

            _env.ClearSpeculationVars();
            Assert.True(SpeculativeCliFlags.Apply(new[] { "--draft-model", gguf, "--no-spec" }));
            Assert.False(SchedulerConfig.FromEnvironment().Speculation.Enabled);
        }
        finally
        {
            File.Delete(gguf);
        }
    }

    [Fact]
    public void Resolve_ExplicitNoSpec_PreservesEngineVeto()
    {
        SpeculativeCliFlags.Apply(new[] { "--no-spec" });

        var settings = SpeculativeDecodingOptions.Resolve(0, -1f);

        Assert.True(settings.ExplicitlyDisabled);
        Assert.False(settings.Enabled);
    }

    [Fact]
    public void Resolve_DefaultOff_PreservesEnginePolicy()
    {
        var settings = SpeculativeDecodingOptions.Resolve(0, -1f);

        Assert.False(settings.ExplicitlyDisabled);
        Assert.False(settings.Enabled);
        Assert.Equal(SpeculatorRegistry.Auto, settings.SpeculatorName);
    }

    [Fact]
    public void CliHost_DraftModelEqualsForm_ReachesTheModelFactoryPath()
    {
        string gguf = Path.Combine(Path.GetTempPath(), $"drafter-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(gguf, new byte[] { 1, 2, 3 });
        try
        {
            SpeculativeCliFlags.Apply(new[] { "--draft-model=" + gguf });

            Assert.Equal(gguf, TensorSharp.Cli.Program.ResolveConfiguredDraftModelPath());
        }
        finally
        {
            File.Delete(gguf);
        }
    }

    [Theory]
    [InlineData(SpeculationEnvVars.DraftModel)]
    [InlineData(SpeculationEnvVars.LegacyDraftModel)]
    public void CliHost_EnvironmentDraftModel_ReachesTheModelFactoryPath(string variable)
    {
        const string gguf = "/configured/drafter.gguf";
        _env.Set(variable, gguf);

        Assert.Equal(gguf, TensorSharp.Cli.Program.ResolveConfiguredDraftModelPath());
    }

    // ----- the dual TS_SPEC_* / TS_MTP_* env spelling -----

    [Fact]
    public void Apply_PublishesBothEnvSpellings_SoTheNativeLoaderStillSeesTheRequest()
    {
        // The glm-dsa NATIVE loader reads TS_MTP_DRAFT from C++ while the model is
        // loading (it sizes its graph cache from it), and the managed half of the
        // same loader reads the enabled variable to decide whether to page a whole
        // extra 256-expert decoder layer into VRAM. Publishing only the current
        // spelling would leave that loader blind, and speculation would go quiet
        // with nothing in the log to explain it. Only the ENV spelling is dual;
        // the FLAG spelling is one name per option.
        Assert.True(SpeculativeCliFlags.Apply(new[] { "--spec", "--spec-draft", "5", "--spec-pmin", "0.6" }));

        Assert.Equal("1", Environment.GetEnvironmentVariable(SpeculationEnvVars.Enabled));
        Assert.Equal("1", Environment.GetEnvironmentVariable(SpeculationEnvVars.LegacyEnabled));
        Assert.Equal("5", Environment.GetEnvironmentVariable(SpeculationEnvVars.Draft));
        Assert.Equal("5", Environment.GetEnvironmentVariable(SpeculationEnvVars.LegacyDraft));
        Assert.Equal("0.6", Environment.GetEnvironmentVariable(SpeculationEnvVars.PMin));
        Assert.Equal("0.6", Environment.GetEnvironmentVariable(SpeculationEnvVars.LegacyPMin));
    }

    [Fact]
    public void FromEnvironment_ReadsLegacyWhenOnlyLegacyIsSet()
    {
        // A deployment that exports TS_MTP_* directly (no CLI flag) must keep
        // working unchanged.
        _env.Set(SpeculationEnvVars.LegacyEnabled, "1");
        _env.Set(SpeculationEnvVars.LegacyDraft, "12");

        var cfg = SpeculationOptions.FromEnvironment();
        Assert.True(cfg.Enabled);
        Assert.Equal(12, cfg.MaxDraftTokens);
    }

    [Theory]
    [InlineData(SpeculationEnvVars.Draft, "65")]
    [InlineData(SpeculationEnvVars.Draft, "1000")]
    [InlineData(SpeculationEnvVars.LegacyDraft, "65")]
    [InlineData(SpeculationEnvVars.LegacyDraft, "1000")]
    public void FromEnvironment_DraftWindowAboveTheBound_IsIgnored(string variable, string value)
    {
        _env.Set(variable, value);

        SpeculationOptions options = SpeculationOptions.FromEnvironment();

        Assert.Equal(SpeculationOptions.DefaultMaxDraftTokens, options.MaxDraftTokens);
        Assert.False(options.MaxDraftTokensExplicit);
    }

    [Theory]
    [InlineData("NaN")]
    [InlineData("Infinity")]
    [InlineData("-Infinity")]
    public void FromEnvironment_NonFinitePmin_IsIgnored(string value)
    {
        _env.Set(SpeculationEnvVars.PMin, value);

        Assert.Null(SpeculationOptions.FromEnvironment().MinDraftProb);
    }

    // ----- removed duplicate spellings -----

    [Fact]
    public void Apply_EveryRemovedSpelling_ErrorsNamingItsSurvivor()
    {
        // Driven off the shared table so a spelling removed later cannot dodge the
        // guard. A hard error, never a silent ignore: the CLI's argument switch
        // drops unknown flags, and "speculation quietly off" is exactly the
        // failure the table exists to prevent.
        Assert.NotEmpty(SpeculativeCliFlags.RemovedFlags);
        foreach ((string flag, string survivor) in SpeculativeCliFlags.RemovedFlags)
        {
            var ex = Assert.Throws<ArgumentException>(() =>
                SpeculativeCliFlags.Apply(new[] { flag, "1" }));
            Assert.Contains(flag, ex.Message);
            Assert.Contains(survivor, ex.Message);

            var eq = Assert.Throws<ArgumentException>(() =>
                SpeculativeCliFlags.Apply(new[] { flag + "=1" }));
            Assert.Contains(survivor, eq.Message);
        }

        // And nothing leaked into the environment before the throw.
        Assert.Null(Environment.GetEnvironmentVariable(SpeculationEnvVars.Enabled));
        Assert.Null(Environment.GetEnvironmentVariable(SpeculationEnvVars.LegacyEnabled));
    }

    [Fact]
    public void Apply_RemovedSpellingAnywhereOnTheLine_StillErrors()
    {
        // RejectRemoved must run before any flag is applied, so a valid --spec
        // earlier on the line does not half-configure the environment first.
        var ex = Assert.Throws<ArgumentException>(() =>
            SpeculativeCliFlags.Apply(new[] { "--spec", "--mtp-draft", "4" }));

        Assert.Contains("--spec-draft", ex.Message);
        Assert.Null(Environment.GetEnvironmentVariable(SpeculationEnvVars.Enabled));
    }

    [Fact]
    public void SpecType_UnknownAlgorithm_FailsFastListingTheKnownOnes()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            SpeculativeCliFlags.Apply(new[] { "--spec-type", "eagle-9000" }));

        Assert.Contains("--spec-type", ex.Message);
        Assert.Contains("eagle-9000", ex.Message);
        Assert.Contains(SpeculatorRegistry.NGram, ex.Message);
    }

    [Fact]
    public void SpecType_KnownAlgorithm_ReachesTheSchedulerConfig()
    {
        Assert.True(SpeculativeCliFlags.Apply(new[] { "--spec", "--spec-type", SpeculatorRegistry.NGram }));

        Assert.Equal(SpeculatorRegistry.NGram, SchedulerConfig.FromEnvironment().Speculation.SpeculatorName);
    }

    [Fact]
    public void DraftModel_DoesNotCollideWithSpecDraft()
    {
        // --spec-draft once had longer siblings (--spec-draft-model and friends);
        // the value flags must keep routing exactly, never by prefix, so
        // --spec-draft 6 and --draft-model PATH land in their own variables.
        string gguf = Path.Combine(Path.GetTempPath(), $"spec-draft-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(gguf, new byte[] { 1, 2, 3 });
        try
        {
            Assert.True(SpeculativeCliFlags.Apply(
                new[] { "--spec-draft", "6", "--draft-model", gguf }));

            Assert.Equal("6", Environment.GetEnvironmentVariable(SpeculationEnvVars.Draft));
            Assert.Equal(gguf, Environment.GetEnvironmentVariable(SpeculationEnvVars.DraftModel));
            Assert.Equal(gguf, Environment.GetEnvironmentVariable(SpeculationEnvVars.LegacyDraftModel));
        }
        finally
        {
            File.Delete(gguf);
        }
    }
}
