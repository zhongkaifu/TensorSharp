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
using System.Globalization;
using System.IO;
using System.Linq;
using TensorSharp.Cli;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime.Speculative;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The CLI usage page's coverage guards, mirroring the server's. They matter
/// MORE here: the CLI's argument switch has no unknown-flag trap, so a flag
/// missing from --help is not merely undiscoverable — a user typing it gets no
/// error either. The page is the only contract a user can see.
/// </summary>
public class CliUsageTests
{
    private static string Usage()
    {
        var sw = new StringWriter();
        CliUsage.PrintUsage(sw);
        return sw.ToString();
    }

    [Fact]
    public void PrintUsage_DocumentsTheSharedFlagFamilies()
    {
        // The inverse guard — accepted flags must be documented — for the
        // families whose names live in shared constant tables. This is the
        // direction that drifted on the server: all six --code-exec* flags were
        // parsed and working while --help never named them.
        string usage = Usage();

        var accepted = new List<string>
        {
            SkillHostOptions.RootsFlag, SkillHostOptions.SelectFlag, SkillHostOptions.ListFlag,
            SkillHostOptions.DisableFlag, SkillHostOptions.NoDiscoveryFlag,
            SkillHostOptions.AllowScriptsFlag, SkillHostOptions.MaxRoundsFlag,
            SkillHostOptions.SandboxFlag, SkillHostOptions.AllowNetworkFlag,
        };
        accepted.AddRange(SpeculativeCliFlags.SwitchFlags);
        accepted.AddRange(SpeculativeCliFlags.ValueFlags);
        // Driven off CodeExecOptions' own tables rather than a copied list, so a flag
        // added there is required to be documented from the moment it is accepted.
        accepted.AddRange(CodeExecOptions.SwitchFlags);
        accepted.AddRange(CodeExecOptions.ValueFlags);

        var missing = accepted.Where(f => !usage.Contains(f, StringComparison.Ordinal)).ToList();
        Assert.True(missing.Count == 0,
            "The CLI accepts these flags but its --help never mentions them:\n  "
            + string.Join("\n  ", missing));
    }

    [Fact]
    public void PrintUsage_DoesNotDocumentRemovedSpeculativeSpellings()
    {
        // A removed spelling on the help page would advertise a flag that only
        // errors; the migration pointer lives in the error message, not here.
        var documented = new HashSet<string>(CliUsage.DocumentedFlags(), StringComparer.Ordinal);
        foreach ((string flag, _) in SpeculativeCliFlags.RemovedFlags)
            Assert.DoesNotContain(flag, documented);
    }

    [Fact]
    public void PrintUsage_ListsRemovedQwenImageFlagsOnlyAsRemoved()
    {
        // The CLI has no unknown-flag trap, so the removed-options note is the one place
        // a user reading --help learns these went; as live entries they would advertise
        // a flag that only errors.
        var documented = new HashSet<string>(CliUsage.DocumentedFlags(), StringComparer.OrdinalIgnoreCase);
        string flattened = System.Text.RegularExpressions.Regex.Replace(Usage(), @"\s+", " ");

        Assert.Contains("Removed options", flattened, StringComparison.Ordinal);
        Assert.NotEmpty(RemovedCliFlags.RemovedFlags);
        foreach ((string flag, string advice) in RemovedCliFlags.RemovedFlags)
        {
            Assert.DoesNotContain(flag, documented);
            Assert.Contains(flag + " Removed: " + advice, flattened, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void PrintUsage_QwenImageExamplesNameOnlyQwenImage21Files()
    {
        string usage = Usage();

        Assert.Contains("config/qwen-image-2.1.json", usage, StringComparison.Ordinal);
        // The removed-options note may name the retired pipeline; nothing else may.
        string options = usage.Substring(0, usage.IndexOf("Removed options", StringComparison.Ordinal));
        string examples = usage.Substring(usage.IndexOf("Examples:", StringComparison.Ordinal));
        foreach (string retired in new[] { "2511", "Qwen2.5-VL", "Qwen-Image-Edit" })
        {
            Assert.DoesNotContain(retired, options, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain(retired, examples, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void DocumentedFlags_YieldsARealList()
    {
        // Guard against a vacuous inverse test: an accessor that yielded nothing
        // would make PrintUsage_DocumentsTheSharedFlagFamilies prove nothing.
        var flags = CliUsage.DocumentedFlags().ToList();
        Assert.True(flags.Count > 40, $"DocumentedFlags() yielded only {flags.Count} flags.");
        Assert.Contains("--model", flags);
        Assert.Contains("--code-exec", flags);
        Assert.Contains("--code-exec-unconfined", flags);
        Assert.Contains("--draft-model", flags);
    }

    [Fact]
    public void PrintUsage_EveryEntryHasAnExample()
    {
        string usage = Usage();
        Assert.Contains("Default:", usage);
        Assert.Contains("Example:", usage);
    }

    [Fact]
    public void PrintUsage_DocumentsTheCurrentPerTokenSpeculationGate()
    {
        string usage = Usage();
        string flattened = System.Text.RegularExpressions.Regex.Replace(usage, @"\s+", " ");

        Assert.Contains("0.15 for a per-token head", flattened);
        Assert.DoesNotContain("0.75 for a per-token head", flattened);
    }

    // ---- the LoRA and sampling entries against the MiniMax-H3 Turbo plug-ins ---------------

    /// <summary>One option's description, flattened to one line.</summary>
    private static string Entry(string option) => UsageEntry(Usage(), option);

    internal static string UsageEntry(string usage, string option)
    {
        usage = usage.Replace("\r\n", "\n", StringComparison.Ordinal);
        string header = "\n  " + option + "\n";
        int start = usage.IndexOf(header, StringComparison.Ordinal);
        Assert.True(start >= 0, $"No usage entry for '{option}'.");
        start += header.Length;
        int end = usage.IndexOf("      Example:", start, StringComparison.Ordinal);
        return System.Text.RegularExpressions.Regex.Replace(usage.Substring(start, end - start), @"\s+", " ").Trim();
    }

    /// <summary>The shipped MiniMax-H3 plug-in configs, by their repository-relative path.</summary>
    internal static Dictionary<string, MiniMaxH3LoraConfig> ShippedH3Plugins()
    {
        string root = LoraHostFixtures.RepoRoot();
        var plugins = Directory.GetFiles(Path.Combine(root, "config", "lora"), "minimax-h3-*.json")
            .ToDictionary(p => "config/lora/" + Path.GetFileName(p), MiniMaxH3LoraConfig.Load);
        Assert.NotEmpty(plugins);
        return plugins;
    }

    [Fact]
    public void Lora_NamesBothFamilies_AndEveryShippedTurboConfigWithItsStepCount()
    {
        string lora = Entry("--lora <path>");
        var plugins = ShippedH3Plugins();

        Assert.Contains("Qwen-Image-2.1 or MiniMax-H3 (FL2VA)", lora, StringComparison.Ordinal);
        Assert.Contains("refused on the Ref2VA checkpoint", lora, StringComparison.Ordinal);
        // Refused before the plug-in downloads, and never claimed to precede a --config's model
        // download: the model's family is read from its header, so it is checked after that download.
        Assert.Contains("refused at startup, before the plug-in downloads", lora, StringComparison.Ordinal);
        Assert.Contains("checked right after that download", lora, StringComparison.Ordinal);
        Assert.DoesNotContain("before anything downloads", lora, StringComparison.Ordinal);
        // Every plug-in it names is shipped, and the step count beside it is the plug-in's own.
        var named = System.Text.RegularExpressions.Regex.Matches(lora, @"config/lora/minimax-h3-[\w.\-]+\.json")
            .Select(m => m.Value).ToHashSet(StringComparer.Ordinal);
        Assert.Equal(plugins.Keys.OrderBy(k => k, StringComparer.Ordinal), named.OrderBy(k => k, StringComparer.Ordinal));
        foreach (var (path, plugin) in plugins)
            Assert.Contains($"{path} ({plugin.Recipe.Steps} steps)", lora, StringComparison.Ordinal);
        foreach (string locked in new[] { "--diffusion-steps", "--flow-shift", "--sampler", "--cfg" })
            Assert.Contains(locked, lora, StringComparison.Ordinal);
    }

    [Fact]
    public void LoraScaleAndConfig_SayWhatMiniMaxH3Refuses()
    {
        Assert.Contains("MiniMax-H3 refuses a negative strength", Entry("--lora-scale <f>"), StringComparison.Ordinal);
        Assert.Contains("must name the preceding --lora's weights file", Entry("--lora-config <path>"), StringComparison.Ordinal);
    }

    [Fact]
    public void DiffusionStepsFlowShiftAndSampler_StateTheMiniMaxH3DefaultsAndTheTurboRecipes()
    {
        var plugins = ShippedH3Plugins().Values.ToList();
        string steps = Entry("--diffusion-steps <N>");
        string shift = Entry("--flow-shift <F>");
        string sampler = Entry("--sampler <name>");

        Assert.Contains($"{MiniMaxH3Scheduler.DefaultSteps} for MiniMax-H3", steps, StringComparison.Ordinal);
        Assert.Equal(new[] { 4, 8 }, plugins.Select(p => p.Recipe.Steps).OrderBy(s => s));
        Assert.Contains("exactly the 8 or 4 steps a MiniMax-H3 Turbo plug-in", steps, StringComparison.Ordinal);

        Assert.Contains(MiniMaxH3Scheduler.VideoShift.ToString("0.0", CultureInfo.InvariantCulture) + " for MiniMax-H3", shift, StringComparison.Ordinal);
        float turboShift = Assert.Single(plugins.Select(p => p.Recipe.VideoShift).Distinct());
        Assert.Contains($"or {turboShift.ToString("0.0", CultureInfo.InvariantCulture)} under a MiniMax-H3 Turbo plug-in, which refuses any other value", shift, StringComparison.Ordinal);

        Assert.All(plugins, p => Assert.Equal("euler", p.Recipe.Sampler));
        Assert.Contains("under a Turbo plug-in any sampler but euler is refused", sampler, StringComparison.Ordinal);
        Assert.Contains("euler for MiniMax-H3", sampler, StringComparison.Ordinal);
    }

    [Fact]
    public void Examples_IncludeAMiniMaxH3TurboRun_WithShippedFiles()
    {
        string usage = Usage();
        const string Example = "TensorSharp.Cli --config config/minimax-h3-fl2va.json --lora config/lora/minimax-h3-fl2v-turbo-8step.json ";

        Assert.Contains(Example, usage.Substring(usage.IndexOf("Examples:", StringComparison.Ordinal)), StringComparison.Ordinal);
        string root = LoraHostFixtures.RepoRoot();
        Assert.True(File.Exists(Path.Combine(root, "config", "minimax-h3-fl2va.json")));
        Assert.Equal(8, ShippedH3Plugins()["config/lora/minimax-h3-fl2v-turbo-8step.json"].Recipe.Steps);
    }
}
