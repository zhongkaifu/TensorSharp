// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license. See LICENSE in the repo root.

using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace TensorSharp.TestMatrix.Reporting;

/// <summary>
/// Loads and writes per-host baseline files. The store auto-derives a host
/// label from <see cref="RuntimeInformation"/> when one is not provided, so
/// runners on different hardware never collide on the same snapshot.
/// </summary>
public sealed class BaselineStore
{
    private static readonly JsonSerializerOptions WriteOptions = new()
    {
        WriteIndented = true,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    private static readonly JsonSerializerOptions ReadOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    public static string DefaultHostLabel()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            return RuntimeInformation.OSArchitecture == Architecture.Arm64
                ? "macos-mlx"
                : "macos-x64";
        }
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            return "linux-cuda";
        }
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return "windows-cuda";
        }
        return "unknown";
    }

    public static string FileNameForHost(string hostLabel) => $"baseline-{hostLabel}.json";

    /// <summary>
    /// Resolve the baseline path to use: an explicit override path wins;
    /// otherwise the per-host file under the baselines directory.
    /// </summary>
    public static string ResolvePath(string? explicitPath, string baselinesDir, string hostLabel)
    {
        if (!string.IsNullOrWhiteSpace(explicitPath))
        {
            return Path.GetFullPath(explicitPath);
        }
        return Path.GetFullPath(Path.Combine(baselinesDir, FileNameForHost(hostLabel)));
    }

    public static BaselineFile? TryLoad(string path)
    {
        if (!File.Exists(path))
        {
            return null;
        }
        try
        {
            BaselineFile? bf = JsonSerializer.Deserialize<BaselineFile>(File.ReadAllText(path), ReadOptions);
            return bf;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[testmatrix] could not parse baseline '{path}': {ex.Message}");
            return null;
        }
    }

    public static void Write(string path, BaselineFile file)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
        string json = JsonSerializer.Serialize(file, WriteOptions);
        File.WriteAllText(path, json);
    }

    /// <summary>
    /// Build a baseline snapshot from the current run's results. Skipped and
    /// errored cells are excluded — the baseline only captures cells that
    /// passed the runtime+correctness gate, so future regressions are flagged
    /// against a known-good state.
    /// </summary>
    public static BaselineFile BuildFromResults(
        IReadOnlyList<TestResult> results,
        string hostLabel,
        string? commit,
        string? notes)
    {
        var bf = new BaselineFile
        {
            HostLabel = hostLabel,
            CapturedAt = DateTimeOffset.UtcNow,
            Commit = commit,
            Notes = notes,
        };
        foreach (TestResult r in results)
        {
            if (r.Skipped) continue;
            if (!r.Ok) continue;
            bf.Entries.Add(new BaselineEntry
            {
                CaseId = r.CaseId,
                Ok = r.Ok,
                CorrectnessOk = r.CorrectnessOk,
                PrefillTps = r.PrefillTps,
                DecodeTps = r.DecodeTps,
                ModelLoadMs = r.ModelLoadMs,
            });
        }
        bf.Entries.Sort((a, b) => string.CompareOrdinal(a.CaseId, b.CaseId));
        return bf;
    }

    public static string? TryGetGitHeadShort()
    {
        try
        {
            var psi = new ProcessStartInfo("git", "rev-parse --short HEAD")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            using var p = Process.Start(psi);
            if (p is null) return null;
            string s = p.StandardOutput.ReadToEnd().Trim();
            p.WaitForExit(1000);
            if (p.ExitCode != 0) return null;
            return string.IsNullOrEmpty(s) ? null : s;
        }
        catch
        {
            return null;
        }
    }
}
