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
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins the parts of Agent Skills that differ between macOS, Linux and Windows.
///
/// <para>
/// The feature has to work on all three, and only one of them can be exercised on any
/// given CI machine — so the platform-specific pieces are pinned two ways. What can be
/// asserted anywhere is asserted anywhere: path handling accepts both separators and
/// always reports skill-relative paths with forward slashes, so a skill authored on
/// Linux reads identically on Windows. What can only run on one platform is pinned by
/// its <b>construction</b> instead: the sandbox command line each platform builds is
/// asserted here without executing it, which catches a wrong flag or a missing bind on
/// a machine that could never run it.
/// </para>
/// <para>
/// The honest limit: these tests prove the Linux and Windows code paths are shaped
/// correctly, not that they were observed confining a process. Only the host's own
/// sandbox is executed, by <see cref="SkillSandboxTests"/>.
/// </para>
/// </summary>
public class SkillCrossPlatformTests : IDisposable
{
    private readonly string _baseDir;

    public SkillCrossPlatformTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-skill-xplat-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_baseDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    // ---- paths: identical meaning on every platform -------------------------

    [Theory]
    [InlineData("scripts/run.py")]
    [InlineData("scripts\\run.py")]
    [InlineData("./scripts/run.py")]
    [InlineData(".\\scripts\\run.py")]
    public void BothSeparators_NameTheSameFile(string spelling)
    {
        // A SKILL.md written on Windows says scripts\run.py and one written on Linux
        // says scripts/run.py. The same skill has to work either way, on either host,
        // or a skill becomes non-portable the moment its author's editor picks a
        // separator.
        string skill = Path.Combine(_baseDir, "sep");
        Directory.CreateDirectory(Path.Combine(skill, "scripts"));
        File.WriteAllText(Path.Combine(skill, "scripts", "run.py"), "print(1)\n");

        Assert.True(SkillPathGuard.TryResolveExistingFile(skill, spelling, out string full, out string error), error);
        Assert.Equal("run.py", Path.GetFileName(full));
    }

    [Fact]
    public void SkillRelativePaths_AreAlwaysReportedWithForwardSlashes()
    {
        // This string goes into the prompt and into skills_read arguments. If it were
        // rendered with backslashes on Windows, the model would echo them back, and a
        // skill's own SKILL.md (which uses forward slashes) would disagree with what
        // the model was told the files are called.
        string skill = Path.Combine(_baseDir, "slashes");
        Directory.CreateDirectory(Path.Combine(skill, "references"));
        File.WriteAllText(Path.Combine(skill, "SKILL.md"),
            "---\nname: slashes\ndescription: Checks that bundled paths are reported portably.\n---\nbody\n");
        File.WriteAllText(Path.Combine(skill, "references", "api.md"), "# API\n");

        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _baseDir } });
        Skill loaded = registry.Skills.Single();

        Assert.Contains(loaded.Files, f => f.Path == "references/api.md");
        Assert.DoesNotContain(loaded.Files, f => f.Path.Contains('\\'));
    }

    [Theory]
    [InlineData("C:\\Windows\\System32\\config\\SAM")]
    [InlineData("\\\\server\\share\\secret")]
    [InlineData("/etc/passwd")]
    [InlineData("~/.ssh/id_rsa")]
    [InlineData("..\\..\\outside.txt")]
    [InlineData("../../outside.txt")]
    public void EveryPlatformsAbsoluteAndEscapingSpellings_AreRejectedOnEveryPlatform(string hostile)
    {
        // The guard has to reject a Windows-shaped escape when running on Linux and a
        // POSIX-shaped one when running on Windows: the string comes from a model or
        // from a ZIP entry, not from the local filesystem, so it can be shaped like any
        // platform regardless of where the server runs.
        string skill = Path.Combine(_baseDir, "guard");
        Directory.CreateDirectory(skill);

        Assert.False(SkillPathGuard.TryResolve(skill, hostile, out _, out string error));
        Assert.False(string.IsNullOrWhiteSpace(error));
    }

    [Fact]
    public void CrLfFrontmatter_ParsesTheSameAsLf()
    {
        // A skill checked out on Windows has CRLF line endings. Its frontmatter must
        // parse to the same values, or half the published skills fail to load there.
        const string lf = "---\nname: endings\ndescription: Checks CRLF handling.\nlicense: MIT\n---\n\n# Body\nText.\n";
        string crlf = lf.Replace("\n", "\r\n");

        Assert.True(SkillManifestParser.TryParse(lf, "endings", out var a, out _));
        Assert.True(SkillManifestParser.TryParse(crlf, "endings", out var b, out _));

        Assert.Equal(a!.Name, b!.Name);
        Assert.Equal(a.Description, b.Description);
        Assert.Equal(a.License, b.License);
        Assert.Equal(a.Body.Replace("\r\n", "\n"), b.Body.Replace("\r\n", "\n"));
    }

    // ---- the sandbox each platform builds -----------------------------------

    [Fact]
    public void EveryPlatformHasASandboxImplementation()
    {
        // Not that one is AVAILABLE here — that depends on the machine — but that the
        // code knows what to reach for. A platform with no implementation at all would
        // mean skills_run could never be enabled there.
        Assert.True(OperatingSystem.IsMacOS() || OperatingSystem.IsLinux() || OperatingSystem.IsWindows(),
            "the supported platform set has changed; add a sandbox for the new one");

        string description = SkillSandboxFactory.DescribeHost();
        Assert.False(string.IsNullOrWhiteSpace(description));
    }

    [Fact]
    public void ADetectedSandboxStatesItsGapsRatherThanImplyingThereAreNone()
    {
        // The whole point of the capability record. A sandbox that confines less must
        // say what it does not confine, because the operator's risk decision depends on
        // it and "sandboxed" alone would be misleading.
        ISkillSandbox sandbox = SkillSandboxFactory.Detect();
        if (sandbox == null)
            return;

        SkillSandboxCapabilities capabilities = sandbox.Capabilities;
        IReadOnlyList<string> gaps = capabilities.Gaps();

        // Whatever the platform, the reported gaps must agree with the capability flags.
        Assert.Equal(!capabilities.ConfinesWrites, gaps.Any(g => g.Contains("write", StringComparison.OrdinalIgnoreCase)));
        Assert.Equal(!capabilities.ConfinesNetwork, gaps.Any(g => g.Contains("network", StringComparison.OrdinalIgnoreCase)));

        // And DescribeHost must surface them, not bury them.
        string described = SkillSandboxFactory.DescribeHost();
        if (gaps.Count > 0)
            Assert.Contains("NOT confined", described, StringComparison.Ordinal);
    }

    [Fact]
    public void TheHostsSandbox_ProducesALaunchableCommand()
    {
        ISkillSandbox sandbox = SkillSandboxFactory.Detect();
        if (sandbox == null)
            return;

        string skill = Path.Combine(_baseDir, "wrap");
        string work = Path.Combine(_baseDir, "wrap-work");
        Directory.CreateDirectory(skill);
        Directory.CreateDirectory(work);

        var request = new SkillSandboxRequest(
            "python3",
            new[] { Path.Combine(skill, "run.py"), "--flag", "value" },
            skill,
            work,
            AllowNetwork: false,
            ReadablePaths: Array.Empty<string>());

        Assert.True(sandbox.TryWrap(request, out string fileName, out IReadOnlyList<string> argv,
            out IDisposable cleanup, out string error), error);
        using (cleanup)
        {
            Assert.False(string.IsNullOrWhiteSpace(fileName));

            // Whatever the wrapper, the interpreter and its arguments must survive into
            // the final command in order — a sandbox that drops the script path or an
            // argument would fail in a way that looks like a broken skill.
            Assert.Contains("python3", argv.Append(fileName));
            Assert.Contains(argv, a => a.EndsWith("run.py", StringComparison.Ordinal));
            Assert.Contains("--flag", argv);
            Assert.Contains("value", argv);
        }
    }

    [Fact]
    public void TheHostsSandbox_KeepsArgumentsSeparateSoNoShellCanReinterpretThem()
    {
        // The wrapper must not flatten the vector into one string anywhere: that is how
        // a sandbox accidentally reintroduces shell interpretation of an argument the
        // model supplied.
        ISkillSandbox sandbox = SkillSandboxFactory.Detect();
        if (sandbox == null)
            return;

        string skill = Path.Combine(_baseDir, "argv");
        string work = Path.Combine(_baseDir, "argv-work");
        Directory.CreateDirectory(skill);
        Directory.CreateDirectory(work);

        var request = new SkillSandboxRequest(
            "python3",
            new[] { Path.Combine(skill, "run.py"), "a b; rm -rf /", "$(whoami)" },
            skill, work, AllowNetwork: false, ReadablePaths: Array.Empty<string>());

        Assert.True(sandbox.TryWrap(request, out _, out IReadOnlyList<string> argv, out IDisposable cleanup, out _));
        using (cleanup)
        {
            Assert.Contains("a b; rm -rf /", argv);
            Assert.Contains("$(whoami)", argv);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SeatbeltNetworkPolicy_IsPinnedOnEveryPlatform(bool allowNetwork)
    {
        SkillSandboxRequest request = SandboxRequest("seatbelt-policy", allowNetwork);

        string profile = SeatbeltSandbox.BuildProfile(request);

        Assert.Equal(allowNetwork, profile.Contains("(allow network*)", StringComparison.Ordinal));
        Assert.Equal(!allowNetwork, profile.Contains("(deny network*)", StringComparison.Ordinal));
        Assert.Equal(allowNetwork,
            profile.Contains("(literal \"/private/var/run/mDNSResponder\")", StringComparison.Ordinal));
        // Network is an orthogonal axis: opening it must not remove either filesystem
        // boundary from the generated profile.
        Assert.Contains("(deny default)", profile, StringComparison.Ordinal);
        Assert.DoesNotContain("(allow file-write*)", profile, StringComparison.Ordinal);
        Assert.Contains("(allow file-read* file-write* (subpath", profile, StringComparison.Ordinal);
        if (!string.IsNullOrEmpty(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)))
            Assert.Contains("(deny file-read* (subpath", profile, StringComparison.Ordinal);
    }

    [Fact]
    public void SeatbeltDoesNotClaimItCanContainASetsidDescendant()
    {
        var sandbox = new SeatbeltSandbox();

        Assert.False(sandbox.Capabilities.BoundsProcessTree);
        Assert.Contains(sandbox.Capabilities.Gaps(),
            gap => gap.Contains("outlive", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BubblewrapNetworkPolicy_IsPinnedOnEveryPlatform(bool allowNetwork)
    {
        SkillSandboxRequest request = SandboxRequest("bwrap-policy", allowNetwork);

        IReadOnlyList<string> arguments = BubblewrapSandbox.BuildArguments(request);

        Assert.Equal(!allowNetwork, arguments.Contains("--unshare-net"));
        Assert.True(arguments.Zip(arguments.Skip(1))
            .Any(pair => pair.First == "--tmpfs" && pair.Second == "/run"));
        // The enabled fast path removes only the network namespace split. Read-only root,
        // writable workspace and process namespaces remain present in both policies.
        Assert.Contains("--ro-bind", arguments);
        Assert.Contains("--bind", arguments);
        Assert.Contains("--new-session", arguments);
        Assert.Contains("--unshare-pid", arguments);
    }

    /// <summary>
    /// Quote a path for SBPL. A path is host-controlled rather than model-controlled
    /// (it is the skill root the operator configured plus a GUID), but a stray quote
    /// would silently truncate the profile and widen the sandbox, so it is escaped
    /// rather than trusted.
    /// </summary>
    /// <remarks>This is the same function the private one in <see cref="SeatbeltSandbox"/></remarks>
    private static string Quote(string path) =>
        "\"" + path.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";

    [Fact]
    public void ExactReadableFilesAndAdditionalWritableRoots_ArePinnedOnEveryPlatform()
    {
        string skill = Path.Combine(_baseDir, "extra-policy-skill");
        string work = Path.Combine(_baseDir, "extra-policy-work");
        string state = Path.Combine(_baseDir, "extra-policy-state");
        string ca = Path.Combine(_baseDir, "enterprise-ca.pem");
        Directory.CreateDirectory(skill);
        Directory.CreateDirectory(work);
        Directory.CreateDirectory(state);
        File.WriteAllText(ca, "TEST-CA\n");
        var request = new SkillSandboxRequest(
            "python3", new[] { Path.Combine(skill, "run.py") }, skill, work,
            AllowNetwork: true, ReadablePaths: new[] { ca })
        {
            WritablePaths = new[] { state },
            StartDirectory = skill,
        };

        string profile = SeatbeltSandbox.BuildProfile(request);
        Assert.Contains("(allow file-read* (literal " + Quote(ca), profile, StringComparison.Ordinal);
        Assert.Contains("(deny file-write* (literal " + Quote(ca), profile, StringComparison.Ordinal);
        Assert.Contains("(allow file-read* file-write* (subpath " + Quote(state),
            profile, StringComparison.Ordinal);
        Assert.Contains("(allow network-bind (local unix-socket (subpath " + Quote(state),
            profile, StringComparison.Ordinal);

        IReadOnlyList<string> arguments = BubblewrapSandbox.BuildArguments(request);
        Assert.True(HasTriplet(arguments, "--ro-bind", ca, ca));
        Assert.True(HasTriplet(arguments, "--bind", state, state));
        int chdir = arguments.ToList().IndexOf("--chdir");
        Assert.InRange(chdir, 0, arguments.Count - 2);
        Assert.Equal(skill, arguments[chdir + 1]);
    }

    [Fact]
    public void WindowsJobObject_ExplicitlyReportsThatItCannotConfineNetworkOrFiles()
    {
#pragma warning disable CA1416 // Construction reads managed capability metadata only; no Windows API is invoked.
        var sandbox = new WindowsJobObjectSandbox();

        Assert.False(sandbox.Capabilities.ConfinesNetwork);
        Assert.False(sandbox.Capabilities.ConfinesWrites);
        Assert.True(sandbox.Capabilities.BoundsProcessTree);
#pragma warning restore CA1416
    }

    [Theory]
    [InlineData("bubblewrap 0.11.0", false)]
    [InlineData("bubblewrap 0.12.0", true)]
    [InlineData("bwrap 1.2.3", true)]
    [InlineData("not-a-version", false)]
    public void BubblewrapVersionFloor_RejectsKnownVulnerableReleases(
        string output, bool expectedSafe)
    {
        bool parsed = BubblewrapSandbox.TryParseVersion(output, out Version? version);

        Assert.Equal(output != "not-a-version", parsed);
        Assert.Equal(expectedSafe,
            parsed && version != null && version >= BubblewrapSandbox.MinimumSafeVersion);
    }

    [Fact]
    public void BubblewrapVersionProbe_BoundsAHungExecutable()
    {
        if (OperatingSystem.IsWindows()) return;

        string probe = WriteExecutable("bwrap-hangs", "#!/bin/sh\nwhile :; do :; done\n");
        var stopwatch = Stopwatch.StartNew();

        bool parsed = BubblewrapSandbox.TryReadVersion(
            probe, out Version? version, timeoutMilliseconds: 200);

        stopwatch.Stop();
        Assert.False(parsed);
        Assert.Null(version);
        Assert.True(stopwatch.Elapsed < TimeSpan.FromSeconds(3),
            $"a version probe configured for 200 ms took {stopwatch.Elapsed}");
    }

    [Fact]
    public void BubblewrapVersionProbe_DrainsStderrWithoutDeadlocking()
    {
        if (OperatingSystem.IsWindows()) return;

        string probe = WriteExecutable(
            "bwrap-fills-stderr",
            "#!/bin/sh\n"
            + "i=0\n"
            + "while [ \"$i\" -lt 4096 ]; do\n"
            + "  printf '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\\n'\n"
            + "  i=$((i + 1))\n"
            + "done >&2\n"
            + "printf 'bubblewrap 0.12.0\\n'\n");

        bool parsed = BubblewrapSandbox.TryReadVersion(
            probe, out Version? version, timeoutMilliseconds: 5000);

        Assert.True(parsed);
        Assert.Equal(new Version(0, 12, 0), version);
    }

    private string WriteExecutable(string name, string body)
    {
        string path = Path.Combine(_baseDir, name);
        File.WriteAllText(path, body);
        if (!OperatingSystem.IsWindows())
        {
            File.SetUnixFileMode(
                path,
                UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        }
        return path;
    }

    private SkillSandboxRequest SandboxRequest(string name, bool allowNetwork)
    {
        string skill = Path.Combine(_baseDir, name + "-skill");
        string work = Path.Combine(_baseDir, name + "-work");
        Directory.CreateDirectory(skill);
        Directory.CreateDirectory(work);
        return new SkillSandboxRequest(
            "python3", new[] { Path.Combine(skill, "run.py") }, skill, work,
            AllowNetwork: allowNetwork, ReadablePaths: Array.Empty<string>());
    }

    private static bool HasTriplet(
        IReadOnlyList<string> values, string first, string second, string third)
    {
        for (int i = 0; i + 2 < values.Count; i++)
        {
            if (values[i] == first && values[i + 1] == second && values[i + 2] == third)
                return true;
        }
        return false;
    }

    // ---- everything except script execution works with no sandbox at all ----

    [Fact]
    public void EverySkillFeatureExceptScriptExecution_WorksWithoutAnySandbox()
    {
        // The important portability statement. Script execution is off unless an
        // operator turns it on; everything else — discovery, parsing, the prompt block,
        // skills_list and skills_read — is pure managed code with no platform
        // dependency, so it behaves identically on all three.
        string skill = Path.Combine(_baseDir, "portable");
        Directory.CreateDirectory(Path.Combine(skill, "references"));
        File.WriteAllText(Path.Combine(skill, "SKILL.md"),
            "---\nname: portable\ndescription: Works the same on every platform.\n---\n\n# Portable\nDo the thing.\n");
        File.WriteAllText(Path.Combine(skill, "references", "detail.md"), "# Detail\nMore.\n");

        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _baseDir } });
        Skill loaded = registry.Skills.Single();

        SkillPlan plan = SkillPrompt.Plan(new[] { loaded }, Array.Empty<Skill>(), SkillPromptOptions.Default);
        Assert.False(plan.IsEmpty);
        // The prompt announces the skill; the body is one skills_read away, on every
        // platform alike — the disclosure tiers are pure managed code.
        Assert.Contains("- portable: Works the same on every platform.", plan.Instructions, StringComparison.Ordinal);
        Assert.DoesNotContain("Do the thing.", plan.Instructions, StringComparison.Ordinal);

        // No ScriptRunner: skills_list and skills_read still work, skills_run refuses.
        var context = new SkillToolContext(new[] { loaded });

        SkillToolResult list = SkillTools.Execute(new ToolCall { Name = SkillTools.ListToolName }, context);
        Assert.True(list.Ok);

        SkillToolResult read = SkillTools.Execute(
            new ToolCall { Name = SkillTools.ReadToolName, Arguments = new() { ["skill"] = "portable", ["path"] = "references/detail.md" } },
            context);
        Assert.True(read.Ok);
        Assert.Contains("More.", read.Content, StringComparison.Ordinal);

        SkillToolResult run = SkillTools.Execute(
            new ToolCall { Name = SkillTools.RunToolName, Arguments = new() { ["skill"] = "portable", ["path"] = "x.py" } },
            context);
        Assert.False(run.Ok);
    }
}
