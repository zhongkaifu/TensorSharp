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
/// Covers the containment around <c>skills_run</c>.
///
/// <para>
/// A skill is content somebody uploaded, and the decision to run one of its scripts is
/// made by a model reading that same person's Markdown — so what the script can reach
/// when it runs is the whole of the security story. These tests pin the parts that can
/// be asserted without depending on which sandbox a given CI machine happens to have:
/// the refuse-rather-than-degrade policy, the interpreter allow-list, the path guard,
/// the environment scrub, and the argument splitting that keeps shell metacharacters
/// inert. The OS confinement itself (network denied, home directory unreadable, writes
/// confined to the scratch directory) is verified against a live hostile script; see
/// docs/agent_skills.md for what that probe showed.
/// </para>
/// </summary>
public class SkillSandboxTests : IDisposable
{
    private readonly string _baseDir;

    public SkillSandboxTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-skill-sandbox-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_baseDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    private Skill WriteSkill(string name = "runner", string scriptBody = "print('hello')\n")
    {
        string dir = Path.Combine(_baseDir, name);
        Directory.CreateDirectory(Path.Combine(dir, "scripts"));
        File.WriteAllText(
            Path.Combine(dir, "SKILL.md"),
            $"---\nname: {name}\ndescription: A skill used to exercise script execution and its confinement.\n---\nbody\n");
        File.WriteAllText(Path.Combine(dir, "scripts", "run.py"), scriptBody);
        File.WriteAllText(Path.Combine(dir, "notes.md"), "# Notes\n");

        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _baseDir } });
        return registry.Skills.Single(s => s.Id == name);
    }

    // ---- the refuse-rather-than-degrade policy ------------------------------

    [Fact]
    public void Required_OnAHostWithNoSandbox_RefusesInsteadOfRunningUnconfined()
    {
        // This is the property the whole mode exists for: "isolation was unavailable"
        // must never quietly become "isolation was skipped". The assertion is written
        // so it holds either way — where a sandbox exists the runner runs, where it
        // does not it must refuse AND explain.
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions { Sandbox = SkillSandboxMode.Required });

        // "No sandbox" is not the only way isolation can be unavailable, and the
        // distinction is the reason this assertion is written against CONFINEMENT
        // rather than against the existence of an ISkillSandbox. Windows always has
        // one - a job object - and it restricts no file and no socket, which its own
        // Capabilities say plainly. Testing for existence made this pass on Windows
        // while `required`, the DEFAULT, behaved exactly like `preferred` there.
        ISkillSandbox? detected = SkillSandboxFactory.Detect();
        bool confines = detected is not null
            && detected.Capabilities.ConfinesWrites
            && detected.Capabilities.ConfinesNetwork;

        if (!confines)
        {
            Assert.False(runner.CanRun);
            Assert.NotNull(runner.UnavailableReason);
            Assert.Contains("confined", runner.UnavailableReason!, StringComparison.OrdinalIgnoreCase);

            // And the refusal must reach the model as a tool result, not an exception.
            Skill skill = WriteSkill();
            SkillToolResult result = runner.Run(skill, "scripts/run.py", Array.Empty<string>());
            Assert.False(result.Ok);
        }
        else
        {
            Assert.True(runner.CanRun);
            Assert.Equal(detected!.Name, runner.Sandbox!.Name);
        }
    }

    [Fact]
    public void Off_RunsWithoutASandboxAndSaysSoInTheResult()
    {
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions { Sandbox = SkillSandboxMode.Off });

        Assert.True(runner.CanRun);
        Assert.Null(runner.Sandbox);
    }

    [Fact]
    public void Preferred_NeverRefuses()
    {
        // The developer-machine mode: sandbox where possible, run regardless.
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions { Sandbox = SkillSandboxMode.Preferred });
        Assert.True(runner.CanRun);
    }

    // ---- what may be run at all --------------------------------------------

    [Fact]
    public void APathThatEscapesTheSkill_IsRefusedBeforeAnyProcessStarts()
    {
        Skill skill = WriteSkill();
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions { Sandbox = SkillSandboxMode.Off });

        SkillToolResult result = runner.Run(skill, "../../../../bin/sh", new[] { "-c", "echo pwned" });

        Assert.False(result.Ok);
        Assert.Contains("escapes the skill directory", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AFileTypeWithNoInterpreter_IsRefusedAndListsWhatIsAllowed()
    {
        // An allow-list rather than "chmod +x and exec": otherwise a shipped binary
        // would run, and a shebang inside an uploaded script would pick the interpreter
        // instead of this table.
        Skill skill = WriteSkill();
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions { Sandbox = SkillSandboxMode.Off });

        SkillToolResult result = runner.Run(skill, "notes.md", Array.Empty<string>());

        Assert.False(result.Ok);
        Assert.Contains(".md", result.Content, StringComparison.Ordinal);
        Assert.Contains(".py", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ShorteningTheInterpreterMap_ForbidsThatLanguage()
    {
        Skill skill = WriteSkill();
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Off,
            Interpreters = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase) { [".sh"] = "/bin/sh" },
        });

        SkillToolResult result = runner.Run(skill, "scripts/run.py", Array.Empty<string>());

        Assert.False(result.Ok);
        Assert.Contains("'.py' files cannot be run here", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void NoRunnerConfigured_TellsTheModelToReadTheScriptInstead()
    {
        // The default everywhere. The message has to be actionable: the model can still
        // read the script and carry out its steps, and saying so is better than a bare
        // refusal it will retry.
        Skill skill = WriteSkill();
        var context = new SkillToolContext(new[] { skill });

        SkillToolResult result = SkillTools.Execute(
            new ToolCall { Name = "skills_run", Arguments = new() { ["skill"] = "runner", ["path"] = "scripts/run.py" } },
            context);

        Assert.False(result.Ok);
        Assert.Contains("disabled", result.Content, StringComparison.OrdinalIgnoreCase);
        Assert.Contains(SkillTools.ReadToolName, result.Content, StringComparison.Ordinal);
    }

    // ---- arguments are data, never syntax ----------------------------------

    [Theory]
    [InlineData("a b c", new[] { "a", "b", "c" })]
    [InlineData("\"one arg\" two", new[] { "one arg", "two" })]
    [InlineData("'single quoted' x", new[] { "single quoted", "x" })]
    [InlineData("back\\ slash", new[] { "back slash" })]
    [InlineData("", new string[0])]
    [InlineData("   ", new string[0])]
    public void SplitArguments_MatchesShellQuotingWithoutAShell(string commandLine, string[] expected)
        => Assert.Equal(expected, SkillScriptRunner.SplitArguments(commandLine));

    [Fact]
    public void SplitArguments_LeavesShellMetacharactersInert()
    {
        // Handing this string to a shell would delete a home directory. Split without
        // one it is five literal arguments: the ';' is just a character on the end of a
        // filename, and "rm", "-rf" and "~" are text the script receives as data. The
        // interpreter is launched with this vector directly, so nothing in it can be
        // interpreted as an operator by anyone.
        List<string> args = SkillScriptRunner.SplitArguments("--out report.pdf; rm -rf ~");

        Assert.Equal(new[] { "--out", "report.pdf;", "rm", "-rf", "~" }, args);
    }

    // ---- the environment the child sees ------------------------------------

    [Fact]
    public void PassThroughEnvironment_IsAShortAllowListNotTheHostsWholeEnvironment()
    {
        // The host process's environment is where credentials live. Inheriting it
        // wholesale would hand every one of them to an uploaded script, and no sandbox
        // can help — the values are already in the process image.
        var options = new SkillScriptRunnerOptions();

        Assert.Contains("PATH", options.PassThroughEnvironmentVariables);
        Assert.DoesNotContain(options.PassThroughEnvironmentVariables,
            name => name.Contains("KEY", StringComparison.OrdinalIgnoreCase)
                 || name.Contains("TOKEN", StringComparison.OrdinalIgnoreCase)
                 || name.Contains("SECRET", StringComparison.OrdinalIgnoreCase));
        Assert.True(options.PassThroughEnvironmentVariables.Count <= 8,
            "the pass-through list is meant to stay short; every addition is a new way for a host secret to reach an uploaded script");
    }

    // ---- defaults ----------------------------------------------------------

    [Fact]
    public void Defaults_AreTheSafeOnes()
    {
        var options = new SkillScriptRunnerOptions();

        Assert.Equal(SkillSandboxMode.Required, options.Sandbox);
        Assert.False(options.AllowNetwork);
        Assert.True(options.DeleteScratchDirectory);
        Assert.True(options.Timeout > TimeSpan.Zero);
        Assert.True(options.MaxOutputBytes > 0);
    }

    [Fact]
    public void HostOptions_DefaultToRequiredSandboxAndNoNetwork()
    {
        var options = new SkillHostOptions();

        Assert.Equal(SkillSandboxMode.Required, options.Sandbox);
        Assert.False(options.AllowNetwork);
        Assert.False(options.AllowScripts);   // and scripts are off entirely by default
    }

    [Theory]
    [InlineData("off", SkillSandboxMode.Off)]
    [InlineData("none", SkillSandboxMode.Off)]
    [InlineData("preferred", SkillSandboxMode.Preferred)]
    [InlineData("required", SkillSandboxMode.Required)]
    [InlineData("REQUIRED", SkillSandboxMode.Required)]
    [InlineData(" strict ", SkillSandboxMode.Required)]
    public void ParseSandboxMode_AcceptsTheDocumentedSpellings(string value, SkillSandboxMode expected)
        => Assert.Equal(expected, SkillHostOptions.ParseSandboxMode(value));

    [Fact]
    public void ParseSandboxMode_RejectsATypoRatherThanFallingBackToTheWeakerOption()
    {
        // A misspelt value for the setting that decides whether uploaded code runs
        // confined must be fatal. Silently resolving "requred" to Off would be the
        // worst possible outcome.
        Assert.Throws<ArgumentException>(() => SkillHostOptions.ParseSandboxMode("requred"));
    }

    [Fact]
    public void Parse_ReadsTheSandboxFlags()
    {
        SkillHostOptions options = SkillHostOptions.Parse(new[]
        {
            "--skills-allow-exec", "--skills-sandbox", "preferred", "--skills-allow-network",
        });

        Assert.True(options.AllowScripts);
        Assert.Equal(SkillSandboxMode.Preferred, options.Sandbox);
        Assert.True(options.AllowNetwork);
    }

    [Fact]
    public void ToScriptRunnerOptions_CarriesTheHostsPolicy()
    {
        var host = new SkillHostOptions { Sandbox = SkillSandboxMode.Preferred, AllowNetwork = true };
        SkillScriptRunnerOptions runner = host.ToScriptRunnerOptions();

        Assert.Equal(SkillSandboxMode.Preferred, runner.Sandbox);
        Assert.True(runner.AllowNetwork);
    }

    // ---- detection ---------------------------------------------------------

    [Fact]
    public void DescribeHost_NamesTheSandboxOrSaysThereIsNone()
    {
        // The startup banner and --list-skills print this, so it must never be a bare
        // boolean: an operator needs to know WHICH mechanism is in force.
        string description = SkillSandboxFactory.DescribeHost();

        Assert.False(string.IsNullOrWhiteSpace(description));
        ISkillSandbox? sandbox = SkillSandboxFactory.Detect();
        if (sandbox == null)
            Assert.Contains("no OS sandbox", description, StringComparison.OrdinalIgnoreCase);
        else
            Assert.Contains(sandbox.Name, description, StringComparison.Ordinal);
    }

    [Fact]
    public void TheNoSandboxSummary_CarriesTheGreppablePhrase_OnEveryPlatform()
    {
        // DescribeHost's no-sandbox branch runs only on a host that HAS no sandbox, so
        // its wording goes unexercised everywhere else — which is how the Linux spelling
        // drifted to "no SAFE OS sandbox available" without any test noticing, and why
        // the test above passed on macOS and on Linux-with-bubblewrap while failing on
        // the one host shape it describes. Both spellings are pinned by value here,
        // where every platform runs them.
        Assert.Equal(
            "no OS sandbox available on this platform",
            SkillSandboxFactory.NoSandboxSummary(string.Empty));
        Assert.Equal(
            "no OS sandbox available: bwrap (bubblewrap) is not installed on this host",
            SkillSandboxFactory.NoSandboxSummary("bwrap (bubblewrap) is not installed on this host"));
        // A reason must never replace the phrase, only follow it.
        Assert.StartsWith(
            "no OS sandbox", SkillSandboxFactory.NoSandboxSummary("anything"), StringComparison.Ordinal);
    }

    [Fact]
    public void ADetectedSandbox_DescribesWhatItActuallyEnforces()
    {
        ISkillSandbox? sandbox = SkillSandboxFactory.Detect();
        if (sandbox == null)
            return;   // nothing to describe on this host

        Assert.True(sandbox.IsAvailable);
        Assert.False(string.IsNullOrWhiteSpace(sandbox.Describe()));
        // The three properties the docs promise; the wording may evolve but the claims
        // must stay present, because this string is what an operator reads.
        Assert.Contains("network", sandbox.Describe(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("write", sandbox.Describe(), StringComparison.OrdinalIgnoreCase);
    }
}
