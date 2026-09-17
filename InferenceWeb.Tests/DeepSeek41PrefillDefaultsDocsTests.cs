// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Globalization;
using System.Text.RegularExpressions;

namespace InferenceWeb.Tests;

/// <summary>
/// Drift guards for the DeepSeek V4.1 prefill defaults, in both directions:
/// what the native loader implements must be what
/// docs/env_var_feature_matrix(.md, _zh-cn.md) and the V4.1 model cards say,
/// and every value those rows promise must still exist in the code.
///
/// <list type="bullet">
/// <item><c>TS_DSV41_SPARSE_FA</c>: sparse prefill on the owned F32 CUDA path for
/// launches of more than TSG_PRECISION_DECODE_COLUMNS queries over at least
/// TSG_DSV41_SPARSE_MIN_KEYS keys, <c>0</c> restoring tiled prefill.</item>
/// </list>
/// </summary>
public class DeepSeek41PrefillDefaultsDocsTests
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

    private static long NativeConstant(string source, string name)
    {
        Match match = Regex.Match(source, $@"\b{name}\s*=\s*(\d+)\s*;");
        Assert.True(match.Success, $"native constant {name} not found");
        return long.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture);
    }

    [Fact]
    public void SparsePrefillGate_CodeAndDocsAgree()
    {
        if (RepoRoot is null) return;   // running outside a source checkout
        string policy = Read("TensorSharp.GGML.Native", "ggml_ops_precision_policy.h");
        string loader = Read("TensorSharp.GGML.Native", "ggml_ops_deepseek4.cpp");
        long queries = NativeConstant(policy, "TSG_PRECISION_DECODE_COLUMNS");
        long keys = NativeConstant(policy, "TSG_DSV41_SPARSE_MIN_KEYS");
        Assert.Equal(8, queries);
        Assert.Equal(8192, keys);
        // accepted -> documented: the attention dispatch and the banner both go through the gate.
        Assert.Contains("inline int tsg_dsv41_owned_sparse_capacity(", policy);
        Assert.Contains("std::atoi(sparse_env) == 0", policy);
        Assert.True(Regex.Matches(loader, @"tsg_dsv41_owned_sparse_capacity\(").Count >= 2,
            "attn_mha and the V4.1 precision banner must share the gate");
        Assert.Contains("getenv(\"TS_DSV41_SPARSE_FA\")", loader);
        Assert.Contains("sparse prefill for queries>" + queries.ToString(CultureInfo.InvariantCulture) + ", keys>=" +
                        keys.ToString(CultureInfo.InvariantCulture), loader);

        // documented -> accepted
        string keysText = keys.ToString("N0", CultureInfo.InvariantCulture);
        string en = Row(Read("docs", "env_var_feature_matrix.md"), "TS_DSV41_SPARSE_FA", "env_var_feature_matrix.md");
        Assert.Contains($"more than {queries} queries", en);
        Assert.Contains($"at least {keysText} keys", en);
        Assert.Contains("on by default", en);
        Assert.Contains("`0` restores tiled prefill", en);
        Assert.Contains("16,384", en);
        string zh = Row(Read("docs", "env_var_feature_matrix_zh-cn.md"), "TS_DSV41_SPARSE_FA", "env_var_feature_matrix_zh-cn.md");
        Assert.Contains($"超过 {queries} 个 query", zh);
        Assert.Contains($"至少 {keysText} 个 key", zh);
        Assert.Contains("默认开启", zh);
        Assert.Contains("`0` 恢复分块 prefill", zh);
        Assert.Contains("16,384", zh);
        foreach (string card in new[] { "deepseek41.md", "deepseek41_zh-cn.md" })
        {
            string text = Read("docs", "models", card);
            Assert.Contains("TS_DSV41_SPARSE_FA=0", text);
            Assert.Contains(keysText, text);
            Assert.Contains("--benchmark-dsv41-prefill", text);
        }
        Assert.Contains("--benchmark-dsv41-prefill", Read("TensorSharp.GGML.Native", "tests", "attention_precision_test.cpp"));
    }
}
