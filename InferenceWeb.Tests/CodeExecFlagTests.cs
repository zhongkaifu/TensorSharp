// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp.AgentHost.CodeExec;
using TensorSharp.Cli;
using TensorSharp.Server.Host.Hosting;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins the operator-facing surface of code execution: what <see cref="CodeExecOptions"/>
/// accepts, what it refuses, and what the two usage pages promise about it.
///
/// <para>
/// The rewrite from five program-shaped tools to <c>shell</c> plus <c>apply_patch</c>
/// retired two flags, and a retired flag is the dangerous kind of change here. Neither
/// host's argument reader has an unknown-flag trap that would catch a
/// <c>--code-exec-packages</c> left in a script or a config file — the CLI parses with a
/// <c>switch</c> that simply ignores what it has no case for, and the code-execution
/// family is consumed and removed before the server's own trap ever runs — so the flag
/// would not error, it would quietly stop applying. An operator who believed a
/// package-name allow-list was still confining their model would be running with no
/// allow-list at all. <c>RejectRemoved</c> exists so that ends at startup with a message
/// naming the survivor, and these tests exist so it cannot regress into silence.
/// </para>
/// <para>
/// The same reasoning drives the parse-failure tests. A value flag whose value would not
/// parse used to be swallowed: the value was consumed, the number stayed at its default,
/// and a bare flag was handed on to the host's own parser to trip over somewhere else.
/// The operator's explicit choice was discarded without a word. Every value flag now
/// throws an <see cref="ArgumentException"/> that names itself.
/// </para>
/// <para>
/// The last group closes the loop in both directions between the parsers and the two
/// usage pages, mirroring the guards the speculative-decoding family already carries: a
/// flag that is accepted but undocumented is undiscoverable (and on the CLI, typing it
/// yields no error either), while a flag that is documented but not accepted is an
/// advertised no-op. <c>--code-exec-unconfined</c> used to be a deliberate asymmetry, the
/// server refusing it at startup; it is now accepted by both hosts, because refusing it
/// made <c>--code-exec</c> permanently inert on Windows, which has no confining sandbox
/// for a shell at all. Both pages must therefore offer it.
/// </para>
/// </summary>
public class CodeExecFlagTests
{
    // ---- helpers -----------------------------------------------------------

    /// <summary>Every flag the parser owns, switches and value flags alike.</summary>
    private static List<string> AcceptedFlags() =>
        CodeExecOptions.SwitchFlags.Concat(CodeExecOptions.ValueFlags).ToList();

    private static string CliUsageText()
    {
        var writer = new StringWriter();
        CliUsage.PrintUsage(writer);
        return writer.ToString();
    }

    private static string ServerUsageText()
    {
        var writer = new StringWriter();
        ServerUsage.PrintUsage(writer);
        return writer.ToString();
    }

    /// <summary>
    /// Clear every code-execution environment variable inside a restoring scope. The
    /// ambient environment of whoever runs the suite is not ours to assume, and a
    /// developer who exports TS_CODE_EXEC would otherwise see failures nobody else can
    /// reproduce.
    /// </summary>
    private static EnvScope CleanEnvironment()
    {
        var scope = new EnvScope();
        scope.Set(CodeExecOptions.EnabledEnvVar, null);
        scope.Set(CodeExecOptions.AllowInstallEnvVar, null);
        scope.Set(CodeExecOptions.AllowNetworkEnvVar, null);
        scope.Set(CodeExecOptions.InstallDomainsEnvVar, null);
        return scope;
    }

    /// <summary>
    /// A plausible command line for a flag, keyed on the placeholder its usage entry
    /// declares — so the "is every documented flag really accepted?" guards read the
    /// page rather than being handed the answer from the same tables they check.
    /// </summary>
    private static string[] SampleArgsFor(string flag, string usage)
    {
        int idx = usage.IndexOf(flag + " <", StringComparison.Ordinal);
        if (idx < 0)
            return new[] { flag };                       // bare switch

        int lt = idx + flag.Length + 1;
        int gt = usage.IndexOf('>', lt);
        string placeholder = gt > lt ? usage.Substring(lt + 1, gt - lt - 1) : string.Empty;

        string value = placeholder switch
        {
            "list" => "pypi.org",
            "seconds" => "60",
            "bytes" => "4096",
            "path|name" => "bash",
            _ => "1",
        };
        return new[] { flag, value };
    }

    // ---- parsing: the surviving flags --------------------------------------

    [Fact]
    public void Parse_AnEmptyCommandLine_LeavesTheFeatureAbsent()
    {
        CodeExecOptions options = CodeExecOptions.Parse(Array.Empty<string>(), out List<string> remaining);

        Assert.False(options.Enabled);
        Assert.False(options.AllowInstall);
        Assert.False(options.AllowNetwork);
        Assert.False(options.Unconfined);
        Assert.False(options.IsConfigured);
        Assert.Empty(remaining);

        // The defaults are part of the contract the usage page prints.
        Assert.Equal(TimeSpan.FromSeconds(120), options.Timeout);
        Assert.Equal(32 * 1024, options.MaxOutputBytes);
        Assert.Equal(
            new[] { "pypi.org", "files.pythonhosted.org", "registry.npmjs.org" },
            options.InstallDomains);
    }

    [Fact]
    public void Parse_TheFourSwitches_AreRead()
    {
        CodeExecOptions options = CodeExecOptions.Parse(
            new[]
            {
                CodeExecOptions.EnabledFlag,
                CodeExecOptions.AllowInstallFlag,
                CodeExecOptions.AllowNetworkFlag,
                CodeExecOptions.UnconfinedFlag,
            },
            out List<string> remaining);

        Assert.True(options.Enabled);
        Assert.True(options.AllowInstall);
        Assert.True(options.AllowNetwork);
        Assert.True(options.Unconfined);
        Assert.True(options.IsConfigured);
        Assert.Empty(remaining);
    }

    [Fact]
    public void Parse_TheSwitches_AreCaseInsensitive()
    {
        // A config file's keys are round-tripped into flags and an operator may have
        // typed one in any case; the two hosts must not disagree about which spellings
        // count.
        CodeExecOptions options = CodeExecOptions.Parse(
            new[]
            {
                "--CODE-EXEC",
                "--Code-Exec-Allow-Install",
                "--code-EXEC-Allow-Network",
                "--CODE-exec-UNCONFINED",
            },
            out List<string> remaining);

        Assert.True(options.Enabled);
        Assert.True(options.AllowInstall);
        Assert.True(options.AllowNetwork);
        Assert.True(options.Unconfined);
        Assert.Empty(remaining);
    }

    [Fact]
    public void Parse_EveryValueFlag_AcceptsTheSpacedSpelling()
    {
        CodeExecOptions options = CodeExecOptions.Parse(
            new[]
            {
                CodeExecOptions.TimeoutFlag, "45",
                CodeExecOptions.MaxOutputFlag, "65536",
                CodeExecOptions.ShellFlag, "/opt/homebrew/bin/bash",
                CodeExecOptions.InstallDomainsFlag, "pypi.org,files.pythonhosted.org",
            },
            out List<string> remaining);

        Assert.Equal(TimeSpan.FromSeconds(45), options.Timeout);
        Assert.Equal(65536, options.MaxOutputBytes);
        Assert.Equal("/opt/homebrew/bin/bash", options.Shell);
        Assert.Equal(new[] { "pypi.org", "files.pythonhosted.org" }, options.InstallDomains);
        Assert.Empty(remaining);
    }

    [Fact]
    public void Parse_EveryValueFlag_AcceptsTheEqualsSpelling()
    {
        CodeExecOptions options = CodeExecOptions.Parse(
            new[]
            {
                CodeExecOptions.TimeoutFlag + "=45",
                CodeExecOptions.MaxOutputFlag + "=65536",
                CodeExecOptions.ShellFlag + "=/opt/homebrew/bin/bash",
                CodeExecOptions.InstallDomainsFlag + "=pypi.org,files.pythonhosted.org",
            },
            out List<string> remaining);

        Assert.Equal(TimeSpan.FromSeconds(45), options.Timeout);
        Assert.Equal(65536, options.MaxOutputBytes);
        Assert.Equal("/opt/homebrew/bin/bash", options.Shell);
        Assert.Equal(new[] { "pypi.org", "files.pythonhosted.org" }, options.InstallDomains);
        Assert.Empty(remaining);
    }

    [Fact]
    public void Parse_TheValueFlags_AreCaseInsensitiveInBothSpellings()
    {
        CodeExecOptions spaced = CodeExecOptions.Parse(
            new[] { "--Code-Exec-Timeout", "45", "--CODE-EXEC-SHELL", "pwsh" }, out List<string> spacedRest);
        Assert.Equal(TimeSpan.FromSeconds(45), spaced.Timeout);
        Assert.Equal("pwsh", spaced.Shell);
        Assert.Empty(spacedRest);

        CodeExecOptions joined = CodeExecOptions.Parse(
            new[] { "--CODE-EXEC-MAX-OUTPUT=4096", "--Code-Exec-Install-Domains=pypi.org" }, out List<string> joinedRest);
        Assert.Equal(4096, joined.MaxOutputBytes);
        Assert.Equal(new[] { "pypi.org" }, joined.InstallDomains);
        Assert.Empty(joinedRest);
    }

    [Theory]
    [InlineData("pypi.org,registry.npmjs.org")]
    [InlineData("pypi.org;registry.npmjs.org")]
    [InlineData("pypi.org registry.npmjs.org")]
    [InlineData(" pypi.org , , registry.npmjs.org ")]
    public void Parse_TheInstallDomainList_TakesAnySeparatorAnOperatorMightType(string value)
    {
        CodeExecOptions options = CodeExecOptions.Parse(
            new[] { CodeExecOptions.InstallDomainsFlag, value }, out _);

        Assert.Equal(new[] { "pypi.org", "registry.npmjs.org" }, options.InstallDomains);
    }

    [Fact]
    public void Parse_AnEmptyInstallDomainList_DisablesThePinningRatherThanRestoringTheDefault()
    {
        // Deliberate, and documented on both pages: naming no host at all means the
        // installer is not pinned to the egress proxy. Silently falling back to the
        // default three would make an operator think they had opened the network when
        // they had not — or the reverse.
        CodeExecOptions options = CodeExecOptions.Parse(
            new[] { CodeExecOptions.InstallDomainsFlag + "=" }, out List<string> remaining);

        Assert.Empty(options.InstallDomains);
        Assert.Empty(remaining);
    }

    [Fact]
    public void Parse_AnEmptyShellValue_LeavesTheChoiceToTheHost()
    {
        CodeExecOptions options = CodeExecOptions.Parse(
            new[] { CodeExecOptions.ShellFlag, "   " }, out _);

        Assert.Null(options.Shell);
    }

    [Fact]
    public void Parse_WhatItDoesNotOwn_IsHandedBackInOrder()
    {
        // Both hosts parse the rest of their own command line from `remaining`, so
        // anything consumed here that was not ours would vanish, and anything left
        // behind that WAS ours would reach a parser with no case for it.
        CodeExecOptions options = CodeExecOptions.Parse(
            new[]
            {
                "--model", "m.gguf",
                CodeExecOptions.EnabledFlag,
                "--backend", "ggml_cuda",
                CodeExecOptions.TimeoutFlag, "300",
                "--chat",
            },
            out List<string> remaining);

        Assert.True(options.Enabled);
        Assert.Equal(TimeSpan.FromSeconds(300), options.Timeout);
        Assert.Equal(new[] { "--model", "m.gguf", "--backend", "ggml_cuda", "--chat" }, remaining);
    }

    // ---- parsing: a value that will not parse is refused, never swallowed ----

    [Theory]
    [InlineData("abc")]
    [InlineData("0")]
    [InlineData("-5")]
    [InlineData("12.5")]
    public void Parse_ATimeoutThatIsNotAPositiveWholeNumber_IsRefusedByName(string value)
    {
        // Regression: this used to consume the value, leave the default in place and
        // hand a bare --code-exec-timeout on to the host's own parser. The operator's
        // explicit choice was discarded and the only symptom was a confusing error
        // about a different flag, somewhere else.
        var spaced = Assert.Throws<ArgumentException>(
            () => CodeExecOptions.Parse(new[] { CodeExecOptions.TimeoutFlag, value }, out _));
        Assert.Contains(CodeExecOptions.TimeoutFlag, spaced.Message, StringComparison.Ordinal);
        Assert.Contains(value, spaced.Message, StringComparison.Ordinal);

        var joined = Assert.Throws<ArgumentException>(
            () => CodeExecOptions.Parse(new[] { CodeExecOptions.TimeoutFlag + "=" + value }, out _));
        Assert.Contains(CodeExecOptions.TimeoutFlag, joined.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("lots")]
    [InlineData("0")]
    [InlineData("-1")]
    [InlineData("64k")]
    public void Parse_AMaxOutputThatIsNotAPositiveWholeNumber_IsRefusedByName(string value)
    {
        var spaced = Assert.Throws<ArgumentException>(
            () => CodeExecOptions.Parse(new[] { CodeExecOptions.MaxOutputFlag, value }, out _));
        Assert.Contains(CodeExecOptions.MaxOutputFlag, spaced.Message, StringComparison.Ordinal);
        Assert.Contains(value, spaced.Message, StringComparison.Ordinal);

        var joined = Assert.Throws<ArgumentException>(
            () => CodeExecOptions.Parse(new[] { CodeExecOptions.MaxOutputFlag + "=" + value }, out _));
        Assert.Contains(CodeExecOptions.MaxOutputFlag, joined.Message, StringComparison.Ordinal);
    }

    // ---- retired spellings --------------------------------------------------

    [Fact]
    public void TheFlagTables_AreNotEmpty()
    {
        // Anti-vacuity for everything below and for the documentation guards: a table
        // that yielded nothing would make half this file pass while proving nothing.
        Assert.NotEmpty(CodeExecOptions.RemovedFlags);
        Assert.NotEmpty(CodeExecOptions.SwitchFlags);
        Assert.NotEmpty(CodeExecOptions.ValueFlags);
    }

    [Fact]
    public void RejectRemoved_RefusesEveryRetiredSpelling_NamingItsSurvivor()
    {
        foreach ((string flag, string survivor, string _) in CodeExecOptions.RemovedFlags)
        {
            string? message = CodeExecOptions.RejectRemoved(new[] { "--model", "m.gguf", flag, "numpy" });

            Assert.NotNull(message);
            Assert.Contains(flag, message!, StringComparison.Ordinal);
            // The whole point of the message: an operator must not have to guess what
            // to write instead. The pointer lives here, not on the help page.
            Assert.Contains(survivor, message!, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void RejectRemoved_RefusesTheValueSpellingAndAnyCasing()
    {
        foreach ((string flag, string survivor, string _) in CodeExecOptions.RemovedFlags)
        {
            string? joined = CodeExecOptions.RejectRemoved(new[] { flag + "=numpy,pandas" });
            Assert.NotNull(joined);
            Assert.Contains(survivor, joined!, StringComparison.Ordinal);

            string? shouted = CodeExecOptions.RejectRemoved(new[] { flag.ToUpperInvariant() });
            Assert.NotNull(shouted);
            // The message quotes the canonical spelling even when the operator shouted.
            Assert.Contains(flag, shouted!, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void RejectRemoved_LetsACleanLineThrough()
    {
        Assert.Null(CodeExecOptions.RejectRemoved(null));
        Assert.Null(CodeExecOptions.RejectRemoved(Array.Empty<string>()));
        Assert.Null(CodeExecOptions.RejectRemoved(AcceptedFlags()));
        Assert.Null(CodeExecOptions.RejectRemoved(new[] { "--model", "m.gguf", CodeExecOptions.EnabledFlag }));
    }

    // ---- environment overrides ----------------------------------------------

    [Theory]
    [InlineData("1", true)]
    [InlineData("true", true)]
    // The documented rule, and the one that surprises people: ANY value but 0 is on,
    // so TS_CODE_EXEC=false enables it. Both usage pages say so out loud.
    [InlineData("false", true)]
    [InlineData("0", false)]
    public void ApplyEnvironment_AnyValueButZero_TurnsTheToolOn(string value, bool expected)
    {
        using EnvScope env = CleanEnvironment();
        env.Set(CodeExecOptions.EnabledEnvVar, value);

        var options = new CodeExecOptions();
        options.ApplyEnvironment();

        Assert.Equal(expected, options.Enabled);
    }

    [Fact]
    public void ApplyEnvironment_TheInstallSwitch_HasItsOwnVariable()
    {
        using EnvScope env = CleanEnvironment();
        env.Set(CodeExecOptions.AllowInstallEnvVar, "1");

        var options = new CodeExecOptions();
        options.ApplyEnvironment();

        // Reaching the network is the separate, more dangerous decision: turning the
        // tool on must not turn installs on, nor the other way round.
        Assert.True(options.AllowInstall);
        Assert.False(options.AllowNetwork);
        Assert.False(options.Enabled);
    }

    [Fact]
    public void ApplyEnvironment_TheNetworkSwitch_HasItsOwnVariable()
    {
        using EnvScope env = CleanEnvironment();
        env.Set(CodeExecOptions.AllowNetworkEnvVar, "1");

        var options = new CodeExecOptions();
        options.ApplyEnvironment();

        // Network access is broader than a host-performed package install and must be a
        // separate decision. It must not silently turn either neighbouring feature on.
        Assert.True(options.AllowNetwork);
        Assert.False(options.AllowInstall);
        Assert.False(options.Enabled);
    }

    [Fact]
    public void ApplyEnvironment_NeverTurnsOffWhatTheCommandLineTurnedOn()
    {
        using EnvScope env = CleanEnvironment();
        env.Set(CodeExecOptions.EnabledEnvVar, "0");

        CodeExecOptions options = CodeExecOptions.Parse(new[] { CodeExecOptions.EnabledFlag }, out _);
        options.ApplyEnvironment();

        // The variable is an override for what was NOT set, not a veto over the flag
        // the operator typed on this very command line.
        Assert.True(options.Enabled);
    }

    [Fact]
    public void ApplyEnvironment_TheInstallDomainsVariable_ReplacesTheList()
    {
        using EnvScope env = CleanEnvironment();
        env.Set(CodeExecOptions.InstallDomainsEnvVar, "mirror.internal;proxy.internal");

        var options = new CodeExecOptions();
        options.ApplyEnvironment();

        // Replaces, not adds: an operator who names their own mirror is not also
        // asking to keep reaching pypi.org.
        Assert.Equal(new[] { "mirror.internal", "proxy.internal" }, options.InstallDomains);
    }

    [Fact]
    public void ApplyEnvironment_AnInstallDomainsVariableNamingNoHost_YieldsAnEmptyList()
    {
        // An empty value deliberately disables the proxy pinning rather than restoring
        // the default three. It cannot be staged with a literally empty string here —
        // Environment.SetEnvironmentVariable deletes a variable it is handed "" for —
        // so this drives the same override branch with a value that carries only
        // separators, which is what an operator's stray comma produces.
        using EnvScope env = CleanEnvironment();
        env.Set(CodeExecOptions.InstallDomainsEnvVar, ",");

        var options = new CodeExecOptions();
        options.ApplyEnvironment();

        Assert.Empty(options.InstallDomains);
    }

    // ---- the two usage pages, in both directions ----------------------------

    [Fact]
    public void EveryAcceptedFlag_IsDocumentedOnTheCliUsagePage()
    {
        // The direction that actually drifted: all the --code-exec* flags were parsed
        // and working while --help never named them. On the CLI that is worse than
        // undiscoverable — its argument switch has no unknown-flag trap, so a user who
        // guesses wrong gets no error either.
        var documented = new HashSet<string>(CliUsage.DocumentedFlags(), StringComparer.Ordinal);
        List<string> flags = AcceptedFlags();
        Assert.NotEmpty(flags);

        List<string> missing = flags.Where(f => !documented.Contains(f)).ToList();
        Assert.True(missing.Count == 0,
            "CodeExecOptions accepts these flags but the CLI --help page never names them:\n  "
            + string.Join("\n  ", missing));
    }

    [Fact]
    public void EveryAcceptedFlag_IsDocumentedOnTheServerUsagePage()
    {
        var documented = new HashSet<string>(ServerUsage.DocumentedFlags(), StringComparer.Ordinal);
        // In full, including --code-exec-unconfined: the server accepts it now, and a
        // flag that is accepted but undocumented is undiscoverable.
        List<string> flags = AcceptedFlags();
        Assert.NotEmpty(flags);

        List<string> missing = flags.Where(f => !documented.Contains(f)).ToList();
        Assert.True(missing.Count == 0,
            "CodeExecOptions accepts these flags but the server --help page never names them:\n  "
            + string.Join("\n  ", missing));
    }

    [Fact]
    public void NeitherUsagePage_NamesARetiredSpelling()
    {
        // A retired spelling on a help page advertises a flag that can only error. The
        // migration pointer belongs in RejectRemoved's message, where the operator who
        // still uses the old name will actually meet it.
        var cliDocumented = new HashSet<string>(CliUsage.DocumentedFlags(), StringComparer.Ordinal);
        var serverDocumented = new HashSet<string>(ServerUsage.DocumentedFlags(), StringComparer.Ordinal);
        string cliText = CliUsageText();
        string serverText = ServerUsageText();

        foreach ((string flag, _, _) in CodeExecOptions.RemovedFlags)
        {
            Assert.DoesNotContain(flag, cliDocumented);
            Assert.DoesNotContain(flag, serverDocumented);
            // Not merely absent from the option table — absent from the prose too, so a
            // description cannot keep recommending a flag that no longer exists.
            Assert.DoesNotContain(flag, cliText, StringComparison.Ordinal);
            Assert.DoesNotContain(flag, serverText, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void EveryCodeExecFlagOnTheCliPage_IsOneTheParserAccepts()
    {
        AssertNoAdvertisedNoOps(CliUsage.DocumentedFlags(), CliUsageText(), "the CLI --help page");
    }

    [Fact]
    public void EveryCodeExecFlagOnTheServerPage_IsOneTheParserAccepts()
    {
        AssertNoAdvertisedNoOps(ServerUsage.DocumentedFlags(), ServerUsageText(), "the server --help page");
    }

    /// <summary>
    /// The inverse guard: a documented flag the parser does not consume is an
    /// advertised no-op. Only the server page gives the family its own heading — on the
    /// CLI these entries sit among the skills flags — so the family is identified by its
    /// <c>--code-exec</c> prefix, which is the namespace any new flag would be added
    /// under anyway. The sample value comes from the placeholder the PAGE declares, so
    /// this reads the documentation rather than being handed the answer by the same
    /// tables it is checking.
    /// </summary>
    private static void AssertNoAdvertisedNoOps(IEnumerable<string> documentedFlags, string usage, string pageName)
    {
        List<string> family = documentedFlags
            .Where(f => f.StartsWith(CodeExecOptions.EnabledFlag, StringComparison.Ordinal))
            .Distinct(StringComparer.Ordinal)
            .ToList();

        // Anti-vacuity: a DocumentedFlags() that yielded nothing would make this green
        // while checking nothing at all.
        Assert.True(family.Count >= CodeExecOptions.ValueFlags.Count + 1,
            $"{pageName} documents only {family.Count} code-execution flags.");

        var ignored = new List<string>();
        foreach (string flag in family)
        {
            CodeExecOptions.Parse(SampleArgsFor(flag, usage), out List<string> remaining);
            if (remaining.Count > 0)
                ignored.Add(flag);
        }

        Assert.True(ignored.Count == 0,
            $"These flags are documented on {pageName} but CodeExecOptions.Parse does not consume them:\n  "
            + string.Join("\n  ", ignored));
    }

    [Fact]
    public void TheEscapeHatch_IsOfferedByBothHosts()
    {
        // It used to be CLI-only, on the reasoning that "run model-written commands with
        // the filesystem open" is a trade the owner of a machine can make and the
        // operator of a shared port cannot. The reasoning holds; applying it by refusing
        // the flag did not. On Windows there is no confining sandbox for a shell to fall
        // back to - a job object restricts no file and no socket, PowerShell cannot
        // initialise its filesystem provider inside an AppContainer, and an msys bash
        // fails to load in one at all - so the refusal did not make the server safer, it
        // made --code-exec a flag that could never do anything there. Both hosts now
        // offer the same explicit opt-in, and both say plainly what it gives up.
        Assert.Contains(CodeExecOptions.UnconfinedFlag, CliUsage.DocumentedFlags());
        Assert.Contains(CodeExecOptions.UnconfinedFlag, ServerUsage.DocumentedFlags());
    }
}
