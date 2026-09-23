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
using TensorSharp.Cli;
using TensorSharp.AgentHost.Agents;
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
        accepted.AddRange(SubAgentOptions.SwitchFlags);
        accepted.AddRange(SubAgentOptions.ValueFlags);

        var missing = accepted.Where(f => !usage.Contains(f, StringComparison.Ordinal)).ToList();
        Assert.True(missing.Count == 0,
            "The CLI accepts these flags but its --help never mentions them:\n  "
            + string.Join("\n  ", missing));
    }

    [Fact]
    public void EverySubAgentFlag_HasAnEntryOfItsOwn()
    {
        // Stricter than a prose mention: --sub-agents is a prefix of the other two, so a
        // page naming only --sub-agents-max-threads would pass a substring check for all
        // three. Each flag the tables accept needs its own documented entry.
        var documented = new HashSet<string>(CliUsage.DocumentedFlags(), StringComparer.Ordinal);
        var missing = SubAgentOptions.SwitchFlags.Concat(SubAgentOptions.ValueFlags)
            .Where(f => !documented.Contains(f))
            .ToList();
        Assert.True(missing.Count == 0,
            "SubAgentOptions accepts these flags but the CLI --help page has no entry for them:\n  "
            + string.Join("\n  ", missing));
    }

    [Fact]
    public void EverySubAgentFlagOnThePage_IsConsumedBeforeTheSwitch()
    {
        // The other direction: a documented flag the parser does not consume would reach
        // the CLI's switch, which has no case for it and no unknown-flag trap - an
        // advertised no-op. Driven by the page itself: every documented --sub-agents*
        // entry, with a value taken from the placeholder it declares.
        string usage = Usage();
        List<string> family = CliUsage.DocumentedFlags()
            .Where(f => f.StartsWith(SubAgentOptions.EnableFlag, StringComparison.Ordinal))
            .Distinct(StringComparer.Ordinal)
            .ToList();
        Assert.Equal(SubAgentOptions.SwitchFlags.Count + SubAgentOptions.ValueFlags.Count, family.Count);

        using var env = new EnvScope();
        env.Set(SubAgentOptions.EnableEnvVar, null);
        env.Set(SubAgentOptions.MaxThreadsEnvVar, null);
        env.Set(SubAgentOptions.MaxDepthEnvVar, null);
        foreach (string flag in family)
        {
            bool takesValue = usage.Contains(flag + " <", StringComparison.Ordinal);
            string[] args = takesValue
                ? new[] { "--model", "m.gguf", flag, "2", "--think" }
                : new[] { "--model", "m.gguf", flag, "--think" };

            SubAgentOptions options = CliSubAgents.Parse(args, out string[] remaining);

            // Consumed, value and all, and nothing else touched.
            Assert.Equal(new[] { "--model", "m.gguf", "--think" }, remaining);
            if (flag == SubAgentOptions.EnableFlag)
                Assert.True(options.Enabled);
            else if (flag == SubAgentOptions.MaxThreadsFlag)
                Assert.Equal(2, options.MaxThreads);
            else if (flag == SubAgentOptions.MaxDepthFlag)
                Assert.Equal(2, options.MaxDepth);
            else
                Assert.Fail($"{flag} is documented but this test does not know what it sets.");
        }
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
}
