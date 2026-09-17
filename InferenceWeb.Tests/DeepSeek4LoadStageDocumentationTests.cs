// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.RegularExpressions;

namespace InferenceWeb.Tests;

/// <summary>
/// Drift guard for the environment variables of the DeepSeek V4 / V4.1 native load
/// stages (weight upload, host-expert prefault and page-lock, Engram warm). Both
/// directions: every such variable the native loader reads has a row in the English and
/// Chinese environment-variable matrices, and every such row names a variable the native
/// loader still reads. The rows' stated defaults are checked against the policies in
/// <c>TensorSharp.GGML.Native/dsv4_file_warm.h</c>.
/// </summary>
public class DeepSeek4LoadStageDocumentationTests
{
    private static readonly Regex LoadStageVariable = new(
        @"^TS_(DSV4_LOAD_[A-Z0-9_]+|DSV4_WARM_[A-Z0-9_]+|DSV41_ENGRAM_WARM|HOST_MOE_PIN[A-Z0-9_]*)$",
        RegexOptions.CultureInvariant);

    private static readonly string[] NativeSources =
    {
        "TensorSharp.GGML.Native/ggml_ops_deepseek4.cpp",
        "TensorSharp.GGML.Native/dsv4_file_warm.h",
        "TensorSharp.GGML.Native/ggml_ops_host_pin.cu",
    };

    private static readonly string[] Matrices =
    {
        "docs/env_var_feature_matrix.md",
        "docs/env_var_feature_matrix_zh-cn.md",
    };

    [Fact]
    public void EveryLoadStageVariableTheNativeLoaderReads_HasAMatrixRowInBothLanguages()
    {
        string root = RepoRoot();
        SortedSet<string> read = NativeReads(root);
        Assert.Contains("TS_DSV4_WARM_PREAD", read);
        Assert.Contains("TS_DSV4_LOAD_DROP_CACHE", read);
        Assert.Contains("TS_HOST_MOE_PIN", read);

        foreach (string matrix in Matrices)
        {
            var documented = new SortedSet<string>(MatrixRows(root, matrix).SelectMany(row => row.Names));
            string[] missing = read.Where(name => !documented.Contains(name)).ToArray();
            Assert.True(missing.Length == 0, $"{matrix} has no row for {string.Join(", ", missing)}");
        }
    }

    [Fact]
    public void EveryLoadStageMatrixRow_NamesAVariableTheNativeLoaderReads()
    {
        string root = RepoRoot();
        SortedSet<string> read = NativeReads(root);
        foreach (string matrix in Matrices)
        {
            string[] stale = MatrixRows(root, matrix).SelectMany(row => row.Names)
                .Where(name => LoadStageVariable.IsMatch(name) && !read.Contains(name))
                .Distinct().ToArray();
            Assert.True(stale.Length == 0, $"{matrix} documents {string.Join(", ", stale)}, which no native load stage reads");
        }
    }

    [Fact]
    public void WarmPreadRow_FollowsTheContiguousLoaderRow_AndDefaultsOn()
    {
        string root = RepoRoot();
        foreach (string matrix in Matrices)
        {
            List<MatrixRow> rows = MatrixRows(root, matrix);
            int contiguous = rows.FindIndex(row => row.Names.Contains("TS_DSV4_LOAD_CONTIGUOUS"));
            int pread = rows.FindIndex(row => row.Names.Contains("TS_DSV4_WARM_PREAD"));
            Assert.True(contiguous >= 0 && pread == contiguous + 1,
                $"{matrix}: the TS_DSV4_WARM_PREAD row must directly follow TS_DSV4_LOAD_CONTIGUOUS");
            MatrixRow row = rows[pread];
            Assert.Contains(row.Default, new[] { "on", "开" });
            Assert.Contains("`0`", row.Text);
        }
    }

    [Fact]
    public void HostMoePinRow_StatesTheDeepSeekDefault_WhichTheLoaderImplements()
    {
        string root = RepoRoot();
        string loader = File.ReadAllText(Path.Combine(root, "TensorSharp.GGML.Native/ggml_ops_deepseek4.cpp"));
        Assert.Contains("dsv4_host_expert_pin_requested(getenv(\"TS_HOST_MOE_PIN\"))", loader);

        var expected = new Dictionary<string, string>
        {
            ["docs/env_var_feature_matrix.md"] = "off for DeepSeek V4 / V4.1",
            ["docs/env_var_feature_matrix_zh-cn.md"] = "DeepSeek V4 / V4.1 为关闭",
        };
        foreach ((string matrix, string defaultText) in expected)
        {
            MatrixRow row = MatrixRows(root, matrix).Single(r => r.Names.SequenceEqual(new[] { "TS_HOST_MOE_PIN" }));
            Assert.Contains(defaultText, row.Text);
            Assert.Contains("`0`", row.Text);
        }
        foreach (string card in new[] { "docs/models/deepseek41.md", "docs/models/deepseek41_zh-cn.md", "USAGE.md", "USAGE_zh-cn.md" })
            Assert.Contains("TS_HOST_MOE_PIN=1", File.ReadAllText(Path.Combine(root, card)));
    }

    [Fact]
    public void ModelCards_DescribeThePreadWarmInBothLanguages()
    {
        string root = RepoRoot();
        foreach (string card in new[] { "docs/models/deepseek41.md", "docs/models/deepseek41_zh-cn.md" })
        {
            string text = File.ReadAllText(Path.Combine(root, card));
            Assert.Contains("TS_DSV4_WARM_PREAD=0", text);
            Assert.Contains("readahead(2)", text);
        }
    }

    internal sealed record MatrixRow(string[] Names, string Default, string Text);

    internal static List<MatrixRow> MatrixRows(string root, string relative)
    {
        var rows = new List<MatrixRow>();
        foreach (string line in File.ReadLines(Path.Combine(root, relative)))
        {
            if (!line.StartsWith("| `", StringComparison.Ordinal)) continue;
            string[] cells = line.Split(" | ");
            string[] names = Regex.Matches(cells[0], "`(TS_[A-Z0-9_]+)`").Select(m => m.Groups[1].Value).ToArray();
            if (names.Length == 0) continue;
            // Out-of-matrix tables end with "| Baseline | In matrix |", so the default is
            // the second-to-last cell.
            string defaultCell = cells.Length >= 2 ? cells[^2].Trim() : string.Empty;
            rows.Add(new MatrixRow(names, defaultCell, line));
        }
        return rows;
    }

    internal static SortedSet<string> NativeReads(string root)
    {
        var read = new SortedSet<string>(StringComparer.Ordinal);
        foreach (string source in NativeSources)
        {
            string text = File.ReadAllText(Path.Combine(root, source));
            foreach (Match match in Regex.Matches(text, @"getenv\(""(TS_[A-Z0-9_]+)""\)"))
            {
                string name = match.Groups[1].Value;
                if (LoadStageVariable.IsMatch(name)) read.Add(name);
            }
        }
        return read;
    }

    internal static string RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        Assert.True(dir != null, "repository root (TensorSharp.slnx) not found above the test binaries");
        return dir!.FullName;
    }
}
