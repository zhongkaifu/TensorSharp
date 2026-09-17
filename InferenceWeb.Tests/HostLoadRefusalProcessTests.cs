// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Diagnostics;
using System.Text;

namespace InferenceWeb.Tests;

/// <summary>
/// The real host processes, launched the way an operator launches them, with a model
/// the load refuses. Every refusal used to leave through an unhandled exception: the
/// stack trace was printed after the refusal and the runtime called abort(), exit code
/// 134 (the campaign logs of DeepSeek V4 on too few GPUs, V4.1 with KV_CACHE_DTYPE=q8_0,
/// and GLM with --tp). The contract is one line on stderr and exit code 2.
/// </summary>
/// <remarks>
/// Deliberately written against the process boundary only (exit code, stdout, stderr)
/// and not against any type the fix introduced, so the same file runs against a build
/// from before the fix and fails there.
/// </remarks>
public class HostLoadRefusalProcessTests : IDisposable
{
    /// <summary>USAGE.md "Exit codes": the model load was refused.</summary>
    private const int ModelLoadRefusedExitCode = 2;

    private const string ErrorPrefix = "error: model load refused: ";

    private readonly string _dir;

    public HostLoadRefusalProcessTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "ts-load-refusal-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    [Fact]
    public void Server_MissingModel_ExitsWithTheRefusalCodeAndOneStderrLine()
    {
        string missing = Path.Combine(_dir, "missing.gguf");
        HostRun? run = RunServer(missing);
        if (run == null) return;

        AssertRefused(run, "missing.gguf");
    }

    [Fact]
    public void Server_FileThatIsNotAGguf_ExitsWithTheRefusalCodeAndOneStderrLine()
    {
        HostRun? run = RunServer(WriteJunkModel());
        if (run == null) return;

        AssertRefused(run, "Not a GGUF file");
        // Refused, not crashed: no stack trace anywhere at the default log level.
        Assert.DoesNotContain("Unhandled exception", run.Stdout + run.Stderr, StringComparison.Ordinal);
        Assert.DoesNotContain("   at ", run.Stdout, StringComparison.Ordinal);
    }

    [Fact]
    public void Server_DebugLogLevel_StillOneStderrLine_ButTheStackTraceIsLogged()
    {
        HostRun? run = RunServer(WriteJunkModel(), logLevel: "Debug");
        if (run == null) return;

        AssertRefused(run, "Not a GGUF file");
        Assert.Contains("   at ", run.Stdout, StringComparison.Ordinal);
    }

    [Fact]
    public void Cli_MissingModel_ExitsWithTheRefusalCodeAndOneStderrLine()
    {
        HostRun? run = RunCli(Path.Combine(_dir, "missing.gguf"));
        if (run == null) return;

        // This one used to exit 0 after printing the usage lines, so a script could not
        // tell a missing model from a completed run.
        AssertRefused(run, "missing.gguf");
    }

    [Fact]
    public void Cli_FileThatIsNotAGguf_ExitsWithTheRefusalCodeAndOneStderrLine()
    {
        HostRun? run = RunCli(WriteJunkModel());
        if (run == null) return;

        AssertRefused(run, "Not a GGUF file");
        Assert.DoesNotContain("Unhandled exception", run.Stdout + run.Stderr, StringComparison.Ordinal);
    }

    private static void AssertRefused(HostRun run, string reasonFragment)
    {
        string[] lines = run.Stderr
            .Split('\n')
            .Select(l => l.TrimEnd('\r'))
            .Where(l => l.Length > 0)
            .ToArray();
        string context = $"exit={run.ExitCode}\n--- stderr ---\n{run.Stderr}\n--- stdout (tail) ---\n{Tail(run.Stdout)}";

        Assert.True(run.ExitCode == ModelLoadRefusedExitCode, context);
        Assert.True(lines.Length == 1, context);
        Assert.StartsWith(ErrorPrefix, lines[0], StringComparison.Ordinal);
        Assert.Contains(reasonFragment, lines[0], StringComparison.Ordinal);
    }

    private string WriteJunkModel()
    {
        string path = Path.Combine(_dir, "junk.gguf");
        File.WriteAllText(path, "not a gguf file at all");
        return path;
    }

    private HostRun? RunServer(string modelPath, string logLevel = "Information")
    {
        string? dll = HostAssembly("TensorSharp.Server.Host", "TensorSharp.Server.Host.dll");
        if (dll == null) return null;
        return Run(dll, logLevel,
            "--model", modelPath, "--backend", "cpu", "--no-webui", "--no-skills", "--no-prefix-cache");
    }

    private HostRun? RunCli(string modelPath)
    {
        string? dll = HostAssembly("TensorSharp.Cli", "TensorSharp.Cli.dll");
        if (dll == null) return null;
        string input = Path.Combine(_dir, "prompt.txt");
        File.WriteAllText(input, "hello");
        return Run(dll, "Information",
            "--model", modelPath, "--backend", "cpu", "--input", input, "--log-dir", Path.Combine(_dir, "cli-logs"));
    }

    private HostRun Run(string dll, string logLevel, params string[] args)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = DotnetHost(),
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = _dir,
        };
        startInfo.ArgumentList.Add(dll);
        foreach (string arg in args)
            startInfo.ArgumentList.Add(arg);
        startInfo.Environment["TENSORSHARP_LOG_LEVEL"] = logLevel;
        startInfo.Environment["TENSORSHARP_LOG_DIR"] = Path.Combine(_dir, "logs");
        // Nothing inherited from the test run may change what the host loads.
        foreach (string name in new[] { "KV_CACHE_DTYPE", "MAX_CONTEXT", "TENSORSHARP_TP_DEGREE", "TS_SPEC_DRAFT_MODEL", "TS_MTP_DRAFT_MODEL" })
            startInfo.Environment.Remove(name);

        using var process = new Process { StartInfo = startInfo };
        var stdout = new StringBuilder();
        var stderr = new StringBuilder();
        process.OutputDataReceived += (_, e) => { if (e.Data != null) lock (stdout) stdout.AppendLine(e.Data); };
        process.ErrorDataReceived += (_, e) => { if (e.Data != null) lock (stderr) stderr.AppendLine(e.Data); };
        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        if (!process.WaitForExit(120_000))
        {
            try { process.Kill(entireProcessTree: true); } catch { /* already gone */ }
            Assert.Fail($"{Path.GetFileName(dll)} did not exit within 120 s.\n{stderr}\n{Tail(stdout.ToString())}");
        }
        process.WaitForExit();
        return new HostRun(process.ExitCode, stdout.ToString(), stderr.ToString());
    }

    private static string DotnetHost()
    {
        string? fromSdk = Environment.GetEnvironmentVariable("DOTNET_HOST_PATH");
        if (!string.IsNullOrWhiteSpace(fromSdk) && File.Exists(fromSdk))
            return fromSdk;
        string? root = Environment.GetEnvironmentVariable("DOTNET_ROOT");
        if (!string.IsNullOrWhiteSpace(root))
        {
            string candidate = Path.Combine(root, OperatingSystem.IsWindows() ? "dotnet.exe" : "dotnet");
            if (File.Exists(candidate))
                return candidate;
        }
        return "dotnet";
    }

    /// <summary>
    /// The host's own build output (both projects build into <c>&lt;project&gt;/bin/</c>),
    /// not the copy beside the test assembly, whose deps.json is the test project's.
    /// Null, which skips the test, when the host has not been built.
    /// </summary>
    private static string? HostAssembly(string project, string fileName)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        if (dir == null)
            return null;
        string path = Path.Combine(dir.FullName, project, "bin", fileName);
        return File.Exists(path) ? path : null;
    }

    private static string Tail(string text)
    {
        string[] lines = text.Split('\n');
        return string.Join('\n', lines.Skip(Math.Max(0, lines.Length - 25)));
    }

    private sealed record HostRun(int ExitCode, string Stdout, string Stderr);
}
