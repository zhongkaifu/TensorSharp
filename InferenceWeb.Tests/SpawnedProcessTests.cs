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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.AgentHost.CodeExec;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// Starting a child process without forking.
///
/// <para>
/// The first test is the one the rest exist to protect: on Unix the mechanism must be
/// <c>posix_spawn</c>, because a <c>fork()</c> from a host with this many threads can wedge
/// permanently inside libmalloc's atfork handler and take the calling thread with it. The
/// others assert that avoiding the fork did not quietly cost anything the forked path gave
/// — separated streams, exit codes, a working directory, an environment built from nothing
/// — and that two things it got WRONG are now right: killing the whole tree, and refusing
/// to wait forever on a pipe a grandchild still holds.
/// </para>
/// </summary>
public class SpawnedProcessTests
{
    private static bool OnUnix => !OperatingSystem.IsWindows();

    private static SpawnRequest Request(
        string file, string[] args, string? cwd = null,
        Action<string>? onOut = null, Action<string>? onErr = null,
        Dictionary<string, string>? env = null) =>
        new()
        {
            FileName = file,
            Arguments = args,
            WorkingDirectory = cwd,
            Environment = env ?? new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["PATH"] = "/usr/bin:/bin",
                ["LANG"] = "C.UTF-8",
            },
            OnStdoutLine = onOut,
            OnStderrLine = onErr,
        };

    // ---- the point of the whole thing ------------------------------------

    [Fact]
    public void OnUnixTheChildIsSpawned_NotForked()
    {
        if (!OnUnix)
            return;

        var lines = new List<string>();
        Assert.True(
            SpawnedProcess.TryStart(Request("/bin/echo", new[] { "x" }, onOut: lines.Add),
                out SpawnedProcess? p, out string error),
            error);

        using (p)
        {
            // If this ever reads "fork", the hazard is back: .NET's Process starts children
            // with fork()+execve(), and the window between the two is where libmalloc's
            // atfork handler can spin forever.
            Assert.Equal("posix_spawn", p!.Mechanism);
            Assert.True(p.WaitForExit(30_000));
        }
    }

    // ---- everything the forked path used to give --------------------------

    [Fact]
    public void StdoutAndStderrStayApart_AndTheExitCodeSurvives()
    {
        if (!OnUnix)
            return;

        var outLines = new List<string>();
        var errLines = new List<string>();
        Assert.True(
            SpawnedProcess.TryStart(
                Request("/bin/sh", new[] { "-c", "echo to-out; echo to-err >&2; exit 7" },
                    onOut: outLines.Add, onErr: errLines.Add),
                out SpawnedProcess? p, out string error),
            error);

        using (p)
        {
            Assert.True(p!.WaitForExit(30_000));
            Assert.True(p.WaitForDrain(10_000));
            Assert.Equal(7, p.ExitCode);
        }

        Assert.Equal(new[] { "to-out" }, outLines);
        Assert.Equal(new[] { "to-err" }, errLines);
    }

    [Fact]
    public void TheChildStartsInTheDirectoryItWasGiven()
    {
        if (!OnUnix)
            return;

        string dir = Directory.CreateTempSubdirectory("spawn-cwd-").FullName;
        try
        {
            var lines = new List<string>();
            Assert.True(
                SpawnedProcess.TryStart(Request("/bin/sh", new[] { "-c", "pwd" }, cwd: dir, onOut: lines.Add),
                    out SpawnedProcess? p, out string error),
                error);
            using (p) { p!.WaitForExit(30_000); p.WaitForDrain(10_000); }

            // Resolved, because a temp directory is reached through a symlink on macOS.
            Assert.Single(lines);
            Assert.Equal(
                Path.GetFullPath(dir).TrimEnd('/').Split('/')[^1],
                lines[0].TrimEnd('/').Split('/')[^1]);
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch (IOException) { }
        }
    }

    [Fact]
    public void TheChildGetsTheGivenEnvironmentAndNothingElse()
    {
        if (!OnUnix)
            return;

        // The security property the whole launch path is built on: the host's environment
        // is where credentials live, and no sandbox can take back a value that is already
        // in the child's image.
        const string secret = "TENSORSHARP_SPAWN_TEST_SECRET";
        Environment.SetEnvironmentVariable(secret, "must-not-leak");
        try
        {
            var lines = new List<string>();
            Assert.True(
                SpawnedProcess.TryStart(
                    Request("/bin/sh", new[] { "-c", $"echo \"[${secret}]\"; echo \"[$MARKER]\"" },
                        onOut: lines.Add,
                        env: new Dictionary<string, string>(StringComparer.Ordinal)
                        {
                            ["PATH"] = "/usr/bin:/bin",
                            ["MARKER"] = "handed-over",
                        }),
                    out SpawnedProcess? p, out string error),
                error);
            using (p) { p!.WaitForExit(30_000); p.WaitForDrain(10_000); }

            Assert.Equal(new[] { "[]", "[handed-over]" }, lines);
        }
        finally
        {
            Environment.SetEnvironmentVariable(secret, null);
        }
    }

    [Fact]
    public void AMissingExecutableIsReported_NotThrown()
    {
        string missing = Path.Combine(Path.GetTempPath(), "no-such-" + Guid.NewGuid().ToString("N"));
        Assert.False(
            SpawnedProcess.TryStart(Request(missing, Array.Empty<string>()),
                out SpawnedProcess? p, out string error));
        Assert.Null(p);
        Assert.Contains("not found", error, StringComparison.OrdinalIgnoreCase);
    }

    // ---- two things the forked path got wrong -----------------------------

    [Fact]
    public void KillTakesTheWholeTree_NotJustTheChildWeStarted()
    {
        if (!OnUnix)
            return;

        // The child is its own process group leader, so one signal to -pgid reaches the
        // grandchild too. The shell exits immediately and the grandchild sleeps; if the
        // group kill did not work, the drain below would sit here for 30 seconds.
        Assert.True(
            SpawnedProcess.TryStart(Request("/bin/sh", new[] { "-c", "sleep 30 & sleep 30" }),
                out SpawnedProcess? p, out string error),
            error);

        var sw = Stopwatch.StartNew();
        using (p)
        {
            Assert.False(p!.WaitForExit(300), "the child was expected to still be running");
            p.Kill();
            Assert.True(p.WaitForExit(10_000));
            Assert.True(p.WaitForDrain(10_000), "the grandchild kept the pipe open, so it was not killed");
        }
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(20), $"took {sw.Elapsed}");
    }

    [Fact]
    public void ReaperStopsTheProcessGroupBeforeReapingItsExitedLeader()
    {
        if (!OnUnix)
            return;

        // The old kill path first asked getpgid(leader). The reaper has already waited
        // this shell by the time WaitForExit returns, so that query failed and the group
        // signal was skipped even though the background child still belonged to it.
        // Closing the child's stdio makes the drain complete too, reproducing the exact
        // case where there was no remaining pipe to reveal the leaked process.
        var lines = new List<string>();
        Assert.True(
            SpawnedProcess.TryStart(
                Request("/bin/sh", new[]
                {
                    "-c", "sleep 30 </dev/null >/dev/null 2>&1 & echo $!",
                }, onOut: lines.Add),
                out SpawnedProcess? p, out string error),
            error);

        int descendant = -1;
        try
        {
            Assert.True(p!.WaitForExit(20_000), "the group leader should exit immediately");
            Assert.True(p.WaitForDrain(10_000));
            Assert.True(int.TryParse(Assert.Single(lines), out descendant));

            Assert.True(
                SpinWait.SpinUntil(() => PosixSpawn.kill(descendant, 0) != 0, 10_000),
                $"background descendant {descendant} survived the reaped group leader");
        }
        finally
        {
            if (descendant > 0)
                PosixSpawn.kill(descendant, PosixSpawn.Sigkill);
            p?.Dispose();
        }
    }

    [Fact]
    public void DrainingIsBoundedWhenSomethingElseHoldsThePipe()
    {
        if (!OnUnix)
            return;

        // The bounded drain still matters for a process that deliberately leaves the
        // launch group. Linux has setsid(1); macOS /bin/sh can put a monitored job in its
        // own group. Either child keeps inherited stdout open after the reaper safely
        // stops the shell's original group.
        string command;
        if (OperatingSystem.IsLinux())
        {
            string? setsid = File.Exists("/usr/bin/setsid") ? "/usr/bin/setsid"
                : File.Exists("/bin/setsid") ? "/bin/setsid"
                : null;
            if (setsid == null)
                return;

            // The escape is ASYNCHRONOUS, and the shell must outlive it. `sh -c` runs
            // without job control, so the background job is forked INTO the launch group
            // and only leaves it once it has exec'd setsid(1) and that has called
            // setsid(2) — while StartReaper SIGKILLs the whole launch group the instant
            // this shell exits. Written as `setsid sleep 30 & echo done`, the two race,
            // and on a host where the reaper wins the grandchild dies, both pipes reach
            // EOF, and the drain "succeeds" for a reason that has nothing to do with what
            // is being tested. That is not hypothetical: it is why this test failed on
            // Linux CI and passed locally.
            //
            // `sleep 1` is a FOREGROUND child, so the shell blocks on it for about a
            // second — three orders of magnitude longer than execve + ld.so + setsid(2).
            // `$$` inside the setsid'd sh is its post-setsid pid and `exec` keeps it, so
            // the pid reported is the process that actually holds the pipe.
            command = setsid + " /bin/sh -c 'echo child:$$; exec sleep 30' & sleep 1; echo done";
        }
        else if (OperatingSystem.IsMacOS())
        {
            // Job control puts the job in its own group in the PARENT before it runs, so
            // there is no window for the reaper here.
            command = "set -m; sleep 30 & echo child:$!; echo done";
        }
        else
        {
            return;
        }

        var lines = new ConcurrentQueue<string>();
        var errors = new ConcurrentQueue<string>();
        Assert.True(
            SpawnedProcess.TryStart(
                Request("/bin/sh", new[] { "-c", command },
                    onOut: lines.Enqueue, onErr: errors.Enqueue),
                out SpawnedProcess? p, out string error),
            error);

        int escaped = -1;
        try
        {
            Assert.True(p!.WaitForExit(20_000), "the shell should exit once its foreground child returns");
            Assert.True(SpinWait.SpinUntil(() => lines.Count >= 2, 5000), string.Join("; ", lines));
            string childLine = lines.Single(line => line.StartsWith("child:", StringComparison.Ordinal));
            Assert.True(int.TryParse(childLine.AsSpan("child:".Length), out escaped));

            // Fail HERE, naming the real cause, if the escape never happened. A child
            // that lost the race or failed to exec otherwise surfaces as "the pipe cannot
            // be at EOF", which points at the drain instead of at the child.
            Assert.True(
                PosixSpawn.kill(escaped, 0) == 0,
                $"the escaped child {escaped} was already dead before the drain check; "
                + $"stderr: {string.Join("; ", errors)}");

            var sw = Stopwatch.StartNew();
            bool drained = p.WaitForDrain(500);
            sw.Stop();

            Assert.False(drained, "the pipe cannot be at EOF while the grandchild holds it");
            Assert.True(sw.Elapsed < TimeSpan.FromSeconds(5), $"the bound did not hold: {sw.Elapsed}");
        }
        finally
        {
            if (escaped > 0)
                PosixSpawn.kill(escaped, PosixSpawn.Sigkill);
            p?.Dispose();
        }
    }

    [Fact]
    public void SignalsAreBackToDefaultInTheChild()
    {
        if (!OnUnix)
            return;

        // This host ignores SIGPIPE, and a child that inherited that runs `yes | head -1`
        // forever instead of dying on the closed pipe the way every shell expects. The
        // forked path had to reset dispositions by hand; the spawn attributes do it.
        var lines = new List<string>();
        Assert.True(
            SpawnedProcess.TryStart(Request("/bin/sh", new[] { "-c", "yes | head -1" }, onOut: lines.Add),
                out SpawnedProcess? p, out string error),
            error);

        using (p)
        {
            Assert.True(p!.WaitForExit(15_000), "SIGPIPE was ignored in the child, so `yes` never stopped");
            Assert.True(p.WaitForDrain(10_000));
            Assert.Equal(0, p.ExitCode);
        }
        Assert.Equal(new[] { "y" }, lines);
    }

    // ---- the conditions that produced the original wedge ------------------

    [Fact]
    public void ManyConcurrentStartsUnderAllocationPressureAllComplete()
    {
        if (!OnUnix)
            return;

        // This is the shape that wedged: concurrent starts from a process whose other
        // threads are inside malloc. On the forked path each start was a race against
        // libmalloc's atfork handler; here there is no fork to lose it.
        using var stop = new CancellationTokenSource();
        var churn = new List<Thread>();
        for (int i = 0; i < 4; i++)
        {
            var thread = new Thread(() =>
            {
                var sink = new List<byte[]>();
                while (!stop.IsCancellationRequested)
                {
                    sink.Add(new byte[4096]);
                    if (sink.Count > 512) sink.Clear();
                }
            })
            { IsBackground = true };
            churn.Add(thread);
            thread.Start();
        }

        try
        {
            var failures = new ConcurrentBag<string>();
            var sw = Stopwatch.StartNew();

            Parallel.For(0, 120, new ParallelOptions { MaxDegreeOfParallelism = 12 }, i =>
            {
                var lines = new List<string>();
                if (!SpawnedProcess.TryStart(
                        Request("/bin/echo", new[] { "n" + i.ToString(System.Globalization.CultureInfo.InvariantCulture) },
                            onOut: lines.Add),
                        out SpawnedProcess? p, out string error))
                {
                    failures.Add($"start {i}: {error}");
                    return;
                }

                using (p)
                {
                    if (!p!.WaitForExit(30_000)) { failures.Add($"exit {i}: timed out"); return; }
                    if (!p.WaitForDrain(10_000)) { failures.Add($"drain {i}: timed out"); return; }
                    if (p.ExitCode != 0) failures.Add($"exit {i}: code {p.ExitCode}");
                }

                lock (failures)
                {
                    if (lines.Count != 1 || lines[0] != "n" + i.ToString(System.Globalization.CultureInfo.InvariantCulture))
                        failures.Add($"output {i}: [{string.Join(",", lines)}]");
                }
            });

            sw.Stop();
            Assert.True(failures.IsEmpty, string.Join("\n", failures.Take(10)));

            // Not a benchmark, a liveness bound: a single wedge would have parked a thread
            // permanently and this would never have finished.
            Assert.True(sw.Elapsed < TimeSpan.FromMinutes(2), $"took {sw.Elapsed}");
        }
        finally
        {
            stop.Cancel();
            foreach (Thread thread in churn)
                thread.Join(2000);
        }
    }

    // ---- the security boundary, through the new spawn ---------------------

    [Fact]
    public void ARealSandboxedRunStillRunsAndIsStillConfined()
    {
        if (!OperatingSystem.IsMacOS())
            return;

        TensorSharp.AgentHost.Skills.ISkillSandbox sandbox =
            TensorSharp.AgentHost.Skills.SkillSandboxFactory.Detect();
        if (sandbox == null || !sandbox.IsAvailable)
            return;
        if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out string? python, out _)
            || python == null)
        {
            return;
        }

        // Confinement is the whole point of this launch path, and it now starts through a
        // different mechanism. The wrapper is asserted elsewhere; what is asserted here is
        // that a REAL seatbelt-wrapped child still runs and is still refused, because a
        // sandbox that silently stopped applying would look exactly like one that works.
        string work = Directory.CreateTempSubdirectory("confined-spawn-").FullName;
        string escape = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".tensorsharp-spawn-escape-" + Guid.NewGuid().ToString("N"));
        try
        {
            // Witnessed against $HOME rather than the temp directory: the profile
            // deliberately allows /private/tmp, so an escape attempt there would succeed
            // and prove nothing.
            string script =
                "open('inside.txt','w').write('ok')\n"
                + "try:\n"
                + "    open(r'" + escape + "','w').write('x')\n"
                + "    print('ESCAPED')\n"
                + "except Exception:\n"
                + "    print('DENIED')\n";

            var launch = new ConfinedLaunch
            {
                Interpreter = python,
                Arguments = new[] { "-c", script },
                WriteDirectory = work,
                WorkingDirectory = work,
                ReadOnlyDirectory = work,
                AllowNetwork = false,
                Timeout = TimeSpan.FromSeconds(60),
                MaxOutputBytes = 16 * 1024,
                EnvironmentVariables = new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["HOME"] = work,
                    ["PYTHONDONTWRITEBYTECODE"] = "1",
                },
            };

            ConfinedResult result = ConfinedProcess.Run(
                launch, sandbox, TensorSharp.AgentHost.Skills.SkillSandboxMode.Required);

            Assert.True(result.Started, result.Error ?? "not started");
            Assert.False(result.TimedOut, "the sandboxed run did not finish");
            Assert.Equal("DENIED", result.Stdout.Trim());
            Assert.True(File.Exists(Path.Combine(work, "inside.txt")),
                "the child could not write inside its own workspace");
            Assert.False(File.Exists(escape), "the sandbox let a write reach $HOME");
        }
        finally
        {
            try { File.Delete(escape); } catch (IOException) { }
            try { Directory.Delete(work, recursive: true); } catch (IOException) { }
        }
    }

    // ---- drift ------------------------------------------------------------

    [Fact]
    public void NothingInTheAgentHostStartsAProcessExceptSpawnedProcess()
    {
        string? repoRoot = FindRepoRoot();
        if (repoRoot == null)
            return;

        string host = Path.Combine(repoRoot, "TensorSharp.AgentHost");
        if (!Directory.Exists(host))
            return;

        // System.Diagnostics.Process forks on Unix, so reaching for it directly puts the
        // wedge back. Exactly two files may name it: SpawnedProcess, which is the one place
        // that decides between spawning and forking, and ForkWatchdog, which is what makes
        // the forked fallback survivable.
        var allowed = new[] { "SpawnedProcess.cs", "ForkWatchdog.cs" };
        var pattern = new Regex(@"\bnew\s+(System\.Diagnostics\.)?Process\s*[({]|\bProcess\.Start\s*\(|\bnew\s+(System\.Diagnostics\.)?ProcessStartInfo\b",
            RegexOptions.Compiled);

        var offenders = new List<string>();
        foreach (string file in Directory.EnumerateFiles(host, "*.cs", SearchOption.AllDirectories))
        {
            if (file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                || file.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                || allowed.Contains(Path.GetFileName(file), StringComparer.Ordinal))
            {
                continue;
            }

            string[] lines = File.ReadAllLines(file);
            for (int i = 0; i < lines.Length; i++)
            {
                if (pattern.IsMatch(lines[i]))
                    offenders.Add($"{Path.GetFileName(file)}:{i + 1}: {lines[i].Trim()}");
            }
        }

        Assert.True(
            offenders.Count == 0,
            "these start a process outside SpawnedProcess, which on Unix means forking from a "
            + "60-thread host and risking a permanent wedge in libmalloc's atfork handler:\n  "
            + string.Join("\n  ", offenders));
    }

    private static string? FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        return dir?.FullName;
    }
}
