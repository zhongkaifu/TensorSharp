// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.RegularExpressions;

namespace InferenceWeb.Tests;

/// <summary>
/// Drift guards for the ggml_cpu backend's knobs (TensorSharp.GGML.Native/ggml_ops_cpu_graph.cpp), in both directions:
/// every variable the native code reads has exactly one row in docs/env_var_feature_matrix(.md, _zh-cn.md), and the
/// rows' defaults are what the code does.
/// <list type="bullet">
/// <item><c>TS_GGML_CPU_THREADS</c>: the prompt pool's width (every physical core by default).</item>
/// <item><c>TS_GGML_CPU_DECODE_THREADS</c>: the decode pool's width, for steps of at most <c>kCpuDecodeRows</c> activation
/// rows (on a hybrid Apple CPU every physical core but two by default).</item>
/// <item><c>TS_GGML_CPU_REPACK</c>: weights only products read in ggml-cpu's repacked layouts (on by default on ARM64).</item>
/// </list>
/// </summary>
public class GgmlCpuKnobsDocsTests
{
    private static readonly string RepoRoot = FindRepoRoot();

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        return dir?.FullName;
    }

    private static string Read(params string[] parts) => File.ReadAllText(Path.Combine([RepoRoot, .. parts]));

    private static string Row(string markdown, string variable, string file)
    {
        string[] rows = markdown.Split('\n').Where(line => line.StartsWith($"| `{variable}` |", StringComparison.Ordinal)).ToArray();
        Assert.True(rows.Length == 1, $"{file} must document {variable} in exactly one row; found {rows.Length}.");
        return rows[0];
    }

    [Fact]
    public void CpuKnobs_CodeAndDocsAgree()
    {
        if (RepoRoot is null) return;   // running outside a source checkout
        string code = Read("TensorSharp.GGML.Native", "ggml_ops_cpu_graph.cpp");
        string header = Read("TensorSharp.GGML.Native", "ggml_ops_cpu_graph.h");
        string en = Read("docs", "env_var_feature_matrix.md");
        string zh = Read("docs", "env_var_feature_matrix_zh-cn.md");

        // accepted -> documented: the variables the native code reads.
        foreach (string name in Regex.Matches(code, "\"(TS_GGML_CPU_[A-Z_]+)\"").Select(m => m.Groups[1].Value).Distinct())
        {
            Row(en, name, "env_var_feature_matrix.md");
            Row(zh, name, "env_var_feature_matrix_zh-cn.md");
        }

        // documented -> accepted, with the defaults the rows promise.
        Assert.Contains("env_threads(\"TS_GGML_CPU_THREADS\")", code);
        Assert.Contains("sysctl_int(\"hw.physicalcpu\")", code);
        Assert.Contains("env_threads(\"TS_GGML_CPU_DECODE_THREADS\")", code);
        Assert.Contains("std::max(perf, prompt - 2)", code);
        Assert.Contains("std::getenv(\"TS_GGML_CPU_REPACK\")", code);
        Match rows = Regex.Match(header, @"kCpuDecodeRows\s*=\s*(\d+)\s*;");
        Assert.True(rows.Success, "kCpuDecodeRows not found");
        foreach ((string markdown, string file) in new[] { (en, "env_var_feature_matrix.md"), (zh, "env_var_feature_matrix_zh-cn.md") })
        {
            Assert.Contains("hw.physicalcpu", Row(markdown, "TS_GGML_CPU_THREADS", file));
            string decode = Row(markdown, "TS_GGML_CPU_DECODE_THREADS", file);
            Assert.Contains($"{rows.Groups[1].Value}", decode);
            Assert.Contains("hw.perflevel0", decode);
            Assert.Contains("hw.perflevel1", decode);
            string repack = Row(markdown, "TS_GGML_CPU_REPACK", file);
            Assert.Contains("CPU_REPACK", repack);
            Assert.Contains("ARM64", repack);
        }
    }
}
