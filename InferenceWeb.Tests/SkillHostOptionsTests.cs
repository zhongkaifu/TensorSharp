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

namespace InferenceWeb.Tests;

/// <summary>
/// Pins the flags both hosts share, and the model-capability table skills route on.
///
/// <para>
/// The CLI parses its arguments with a <c>switch</c> and the server with its own options
/// builder, but a config JSON key <i>is</i> a CLI flag — <c>"skills-dir": [...]</c>
/// becomes <c>--skills-dir ... --skills-dir ...</c> — so the same config file is expected
/// to drive either host. Both spellings of a valued flag (<c>--skills-dir x</c> and
/// <c>--skills-dir=x</c>) therefore have to work, unknown arguments have to be ignored
/// (each host parses the rest of its own command line), and a missing value or an
/// out-of-range round cap has to fail at startup naming the flag — a mistyped path that
/// produced a silently empty registry would surface as "my skills do not show up",
/// which is the hardest kind of bug for a user to diagnose.
/// </para>
/// <para>
/// <see cref="SkillCapabilities"/> is the more consequential half. It decides whether a
/// family gets progressive disclosure through <c>skills_read</c> or has to have skill
/// bodies written into the prompt up front, and whether a tool result may be fed back as
/// a <c>role: "tool"</c> message at all. Get it wrong in the permissive direction and
/// skills appear to work on the affected family while doing nothing whatsoever: the tool
/// is declared into a renderer that discards it, the model never calls what it was never
/// told about, and every request succeeds.
/// </para>
/// </summary>
public class SkillHostOptionsTests : IDisposable
{
    private readonly string _baseDir;

    public SkillHostOptionsTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-skill-options-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_baseDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    // ---- helpers -----------------------------------------------------------

    /// <summary>
    /// Clear every skills environment variable inside a restoring scope. The ambient
    /// environment of whoever runs the suite is not ours to assume, and a developer who
    /// happens to export TS_SKILLS_DIR would otherwise see failures nobody else can
    /// reproduce.
    /// </summary>
    private static EnvScope CleanEnvironment()
    {
        var scope = new EnvScope();
        scope.Set(SkillHostOptions.RootsEnvVar, null);
        scope.Set(SkillHostOptions.DisableEnvVar, null);
        scope.Set(SkillHostOptions.AllowScriptsEnvVar, null);
        scope.Set(SkillHostOptions.MaxRoundsEnvVar, null);
        return scope;
    }

    // ---- parsing -----------------------------------------------------------

    [Fact]
    public void Parse_TheSelectFlagIsRepeatable()
    {
        SkillHostOptions options = SkillHostOptions.Parse(new[] { "--skill", "pdf", "--skill", "xlsx" });

        Assert.Equal(new[] { "pdf", "xlsx" }, options.Selected);
        Assert.True(options.IsConfigured);
    }

    [Fact]
    public void Parse_BothSpellingsOfAValuedFlagAreAccepted()
    {
        // The server's own option reader accepts both, and a config file may produce
        // either; a host that took only one would reject a config the other host loads.
        Assert.Equal(
            new[] { "/opt/skills" },
            SkillHostOptions.Parse(new[] { "--skills-dir", "/opt/skills" }).Roots);

        Assert.Equal(
            new[] { "/opt/skills" },
            SkillHostOptions.Parse(new[] { "--skills-dir=/opt/skills" }).Roots);
    }

    [Fact]
    public void Parse_TheRootsFlagIsRepeatableAndKeepsPrecedenceOrder()
    {
        SkillHostOptions options = SkillHostOptions.Parse(
            new[] { "--skills-dir", "/first", "--skills-dir=/second" });

        // Order is precedence: the first root's copy of a duplicated skill wins, so the
        // order the user typed has to survive parsing intact.
        Assert.Equal(new[] { "/first", "/second" }, options.Roots);
    }

    [Fact]
    public void Parse_TheFourBareSwitches_AreRead()
    {
        SkillHostOptions options = SkillHostOptions.Parse(new[]
        {
            "--no-skills", "--list-skills", "--skills-no-discovery", "--skills-allow-exec",
        });

        Assert.False(options.Enabled);
        Assert.True(options.ListOnly);
        Assert.False(options.Discovery);
        Assert.True(options.AllowScripts);
    }

    [Fact]
    public void Parse_Defaults_LeaveSkillsOnAndScriptsOff()
    {
        SkillHostOptions options = SkillHostOptions.Parse(Array.Empty<string>());

        Assert.True(options.Enabled);
        Assert.True(options.Discovery);
        Assert.False(options.AllowScripts);
        Assert.False(options.IsConfigured);   // a host with no flags stays silent

        // Running a skill's script is arbitrary code execution on the host under the
        // host's account, decided by a model reading untrusted Markdown. Nothing but an
        // explicit flag may turn it on.
        Assert.False(SkillHostOptions.Parse(null).AllowScripts);
    }

    [Fact]
    public void Parse_MaxRounds_IsRead()
    {
        Assert.Equal(3, SkillHostOptions.Parse(new[] { "--skills-max-rounds", "3" }).MaxRounds);
        Assert.Equal(12, SkillHostOptions.Parse(new[] { "--skills-max-rounds=12" }).MaxRounds);
    }

    [Theory]
    [InlineData("--skills-dir")]
    [InlineData("--skill")]
    [InlineData("--skills-max-rounds")]
    public void Parse_AFlagWithNoValue_ThrowsNamingTheFlag(string flag)
    {
        // At startup, before a model is loaded — not on the first request, where the
        // operator has already walked away.
        var ex = Assert.Throws<ArgumentException>(() => SkillHostOptions.Parse(new[] { flag }));

        Assert.Contains(flag, ex.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("0")]
    [InlineData("65")]
    [InlineData("-1")]
    [InlineData("eight")]
    public void Parse_AnUnusableRoundCap_Throws(string value)
    {
        var ex = Assert.Throws<ArgumentException>(
            () => SkillHostOptions.Parse(new[] { "--skills-max-rounds", value }));

        Assert.Contains("--skills-max-rounds", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Parse_UnknownArgumentsAreIgnored()
    {
        // Both hosts hand their whole command line to this parser and then parse the
        // rest themselves; claiming anything it does not own would break every other flag.
        SkillHostOptions options = SkillHostOptions.Parse(new[]
        {
            "--model", "/models/gemma.gguf", "--port", "8080", "--skill", "pdf", "--ctx", "8192",
        });

        Assert.Equal(new[] { "pdf" }, options.Selected);
        Assert.Empty(options.Roots);
    }

    // ---- environment and defaults -------------------------------------------

    [Fact]
    public void ApplyEnvironmentAndDefaults_ReadsTheRootsVariable()
    {
        using EnvScope env = CleanEnvironment();
        string roots = string.Join(Path.PathSeparator, "/opt/skills", "/srv/skills");
        env.Set(SkillHostOptions.RootsEnvVar, roots);

        SkillHostOptions options = SkillHostOptions.Parse(Array.Empty<string>())
            .ApplyEnvironmentAndDefaults(_baseDir);

        Assert.Equal(new[] { "/opt/skills", "/srv/skills" }, options.Roots);
    }

    [Fact]
    public void ApplyEnvironmentAndDefaults_TheCommandLineWinsOverTheRootsVariable()
    {
        // Environment is the deployment's default; the command line is what the person
        // in front of the machine just typed, and it has to win.
        using EnvScope env = CleanEnvironment();
        env.Set(SkillHostOptions.RootsEnvVar, "/opt/skills");

        SkillHostOptions options = SkillHostOptions.Parse(new[] { "--skills-dir", "/typed" })
            .ApplyEnvironmentAndDefaults(_baseDir);

        Assert.Equal(new[] { "/typed" }, options.Roots);
    }

    [Fact]
    public void ApplyEnvironmentAndDefaults_TheDisableVariableTurnsSkillsOff()
    {
        using EnvScope env = CleanEnvironment();
        env.Set(SkillHostOptions.DisableEnvVar, "1");

        Assert.False(SkillHostOptions.Parse(Array.Empty<string>()).ApplyEnvironmentAndDefaults(_baseDir).Enabled);

        // Explicitly "0" is the off switch for the off switch: a deployment that exports
        // the variable unconditionally needs a value that means "no".
        env.Set(SkillHostOptions.DisableEnvVar, "0");
        Assert.True(SkillHostOptions.Parse(Array.Empty<string>()).ApplyEnvironmentAndDefaults(_baseDir).Enabled);
    }

    [Fact]
    public void ApplyEnvironmentAndDefaults_TheAllowExecVariableTurnsScriptsOn()
    {
        using EnvScope env = CleanEnvironment();
        env.Set(SkillHostOptions.AllowScriptsEnvVar, "1");

        Assert.True(SkillHostOptions.Parse(Array.Empty<string>()).ApplyEnvironmentAndDefaults(_baseDir).AllowScripts);

        env.Set(SkillHostOptions.AllowScriptsEnvVar, "0");
        Assert.False(SkillHostOptions.Parse(Array.Empty<string>()).ApplyEnvironmentAndDefaults(_baseDir).AllowScripts);
    }

    [Fact]
    public void ApplyEnvironmentAndDefaults_TheRoundCapVariableIsRead()
    {
        using EnvScope env = CleanEnvironment();
        env.Set(SkillHostOptions.MaxRoundsEnvVar, "5");

        Assert.Equal(5, SkillHostOptions.Parse(Array.Empty<string>()).ApplyEnvironmentAndDefaults(_baseDir).MaxRounds);

        // Out of range is ignored rather than fatal: an environment variable is often
        // set by a deployment the operator does not control, and killing the server over
        // it is worse than falling back to the default.
        env.Set(SkillHostOptions.MaxRoundsEnvVar, "999");
        Assert.Equal(
            SkillAgentLoopOptions.Default.MaxRounds,
            SkillHostOptions.Parse(Array.Empty<string>()).ApplyEnvironmentAndDefaults(_baseDir).MaxRounds);
    }

    [Fact]
    public void ApplyEnvironmentAndDefaults_WithNothingConfigured_FallsBackToTheDirectoryNextToTheBinary()
    {
        // So an operator can drop a skill directory next to the executable and restart,
        // with no flag and no environment variable at all.
        using EnvScope env = CleanEnvironment();

        SkillHostOptions options = SkillHostOptions.Parse(Array.Empty<string>())
            .ApplyEnvironmentAndDefaults(_baseDir);

        Assert.Equal(new[] { Path.Combine(_baseDir, SkillHostOptions.DefaultDirectoryName) }, options.Roots);
    }

    // ---- root validation ----------------------------------------------------

    [Fact]
    public void ValidateRoots_AMissingRootTheOperatorTyped_Throws()
    {
        SkillHostOptions options = SkillHostOptions.Parse(
            new[] { "--skills-dir", Path.Combine(_baseDir, "nope") });

        var ex = Assert.Throws<ArgumentException>(() => options.ValidateRoots());

        Assert.Contains("--skills-dir", ex.Message, StringComparison.Ordinal);
        Assert.Contains("not an existing directory", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ValidateRoots_TheConventionalDefault_IsCreatedRatherThanRejected()
    {
        // The default directory is expected not to exist on a fresh machine, so
        // rejecting it would make every unconfigured host fail to start.
        using EnvScope env = CleanEnvironment();
        SkillHostOptions options = SkillHostOptions.Parse(Array.Empty<string>())
            .ApplyEnvironmentAndDefaults(_baseDir);

        options.ValidateRoots(createDefault: true);

        Assert.True(Directory.Exists(Path.Combine(_baseDir, SkillHostOptions.DefaultDirectoryName)));
    }

    [Fact]
    public void ToRegistryOptions_CarriesTheRootsAndTheInstallDirectory()
    {
        SkillHostOptions options = SkillHostOptions.Parse(new[] { "--skills-dir", _baseDir });

        SkillRegistryOptions registryOptions = options.ToRegistryOptions(Path.Combine(_baseDir, "installed"));

        Assert.Equal(new[] { _baseDir }, registryOptions.Roots);
        Assert.Equal(Path.Combine(_baseDir, "installed"), registryOptions.InstallDirectory);

        // Null keeps the registry read-only, which is what the CLI wants: it registers
        // directories the user already has rather than copying them anywhere.
        Assert.Null(options.ToRegistryOptions().InstallDirectory);
    }

    // ---- model capabilities -------------------------------------------------

    [Theory]
    [InlineData("gemma4")]
    [InlineData("qwen35")]
    [InlineData("qwen4exp")]
    public void Capabilities_AFamilyThatCarriesToolDeclarations_GetsProgressiveDisclosure(string architecture)
    {
        // Both render declarations AND have a parser that reads the call back out, so the
        // full round trip is available and skills use it.
        SkillModelCapabilities capabilities = SkillCapabilities.For(architecture);

        Assert.True(capabilities.ToolsRendered);
        Assert.True(capabilities.ToolResultsRendered);
    }

    [Theory]
    [InlineData("harmony-not-a-real-architecture")]
    public void Capabilities_AFamilyNothingCanParse_IsNotOfferedTools(string architecture)
    {
        // An unrecognised architecture lands on the generic path and ends up on
        // PassthroughOutputParser, which returns every byte as content and can never
        // yield a tool call. (`qwen4exp` used to be the registered example of this
        // until it gained its Qwen XML-style parser; it now takes the round trip
        // above.) Being permissive here was the original bug: skills_read was
        // declared to a model whose replies nothing would parse, so the model called it,
        // nobody answered, and the raw markup reached the user as the answer.
        SkillModelCapabilities capabilities = SkillCapabilities.For(architecture);

        Assert.False(capabilities.ToolsRendered);
        // Tool RESULTS are the renderer's business and unaffected by any of this.
        Assert.True(capabilities.ToolResultsRendered);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    public void Capabilities_NoArchitectureAtAll_WithholdsToolsButKeepsResults(string? architecture)
    {
        // Same reasoning as the unrecognised-architecture case above: nothing can parse a
        // tool call out of a model we know nothing about, so nothing is declared to it.
        SkillModelCapabilities capabilities = SkillCapabilities.For(architecture);

        Assert.False(capabilities.ToolsRendered);
        Assert.True(capabilities.ToolResultsRendered);
    }

    [Theory]
    [InlineData("mistral3")]
    public void Capabilities_AFamilyWhoseRendererDiscardsTools_ReportsSo(string architecture)
    {
        // This family's ChatProtocol entry passes only (Messages, AddGenerationPrompt)
        // to its renderer, so the tool list is dropped before the renderer sees it.
        // Declaring a tool for it is not an error anyone can observe — the request
        // succeeds and the model simply never calls what it was never told about — so
        // Agent Skills has to learn it from this table instead. Getting it wrong makes
        // skills appear to work and silently do nothing.
        Assert.False(SkillCapabilities.For(architecture).ToolsRendered);
    }

    [Fact]
    public void Capabilities_Mistral3_AlsoDropsToolResultMessages()
    {
        // RenderMistral3's message loop handles only "user" and "assistant", so a
        // role:"tool" message is written nowhere at all. An agentic loop that fed a
        // result back that way would ask the model to continue from an answer absent
        // from its prompt, and it would call the same tool again until the round budget
        // ran out. The loop feeds results back as a user turn here instead.
        Assert.False(SkillCapabilities.For("mistral3").ToolResultsRendered);
    }

    // ---- script argument splitting -----------------------------------------

    [Fact]
    public void SplitArguments_NothingToSplit_IsAnEmptyVector()
    {
        Assert.Empty(SkillScriptRunner.SplitArguments(null));
        Assert.Empty(SkillScriptRunner.SplitArguments(""));
        Assert.Empty(SkillScriptRunner.SplitArguments("   "));
    }

    [Fact]
    public void SplitArguments_QuotesHoldAnArgumentTogether()
    {
        // The model writes arguments as one string because a tool parameter cannot be an
        // array here, and a file name with a space in it is the common case.
        Assert.Equal(
            new[] { "--out", "my file.pdf" },
            SkillScriptRunner.SplitArguments("--out \"my file.pdf\""));

        Assert.Equal(
            new[] { "--out", "my file.pdf" },
            SkillScriptRunner.SplitArguments("--out 'my file.pdf'"));
    }

    [Fact]
    public void SplitArguments_BackslashEscapesAreHonoured()
    {
        Assert.Equal(new[] { "a b" }, SkillScriptRunner.SplitArguments("a\\ b"));
        Assert.Equal(new[] { "say \"hi\"" }, SkillScriptRunner.SplitArguments("\"say \\\"hi\\\"\""));
    }

    [Fact]
    public void SplitArguments_AnExplicitlyEmptyArgumentSurvives()
    {
        // "" is a real argument — a script's --prefix "" is not the same as omitting it.
        Assert.Equal(new[] { "--prefix", "" }, SkillScriptRunner.SplitArguments("--prefix \"\""));
    }

    [Fact]
    public void SplitArguments_ShellMetacharactersStayLiteral()
    {
        // THE reason this splitter exists instead of a shell. No shell is involved, so
        // the vector below is five pieces of data — one of which is the harmless text
        // "rm" — rather than two commands. Handing the string to /bin/sh instead would
        // turn every model-authored argument into executable syntax.
        Assert.Equal(
            new[] { "--out", "report.pdf;", "rm", "-rf", "~" },
            SkillScriptRunner.SplitArguments("--out report.pdf; rm -rf ~"));

        Assert.Equal(
            new[] { "$HOME", "|", "cat", ">", "/etc/passwd", "`id`" },
            SkillScriptRunner.SplitArguments("$HOME | cat > /etc/passwd `id`"));
    }
}
