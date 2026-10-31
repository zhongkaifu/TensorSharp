// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Xunit.Sdk;

namespace InferenceWeb.Tests;

/// <summary>
/// A theory row that fails on the base revision because of a known, characterised defect. It
/// skips VISIBLY with the defect's id and summary (a skipped row is never a pass), unless the
/// environment variable <see cref="EnvironmentVariable"/> names the defect (comma-separated ids,
/// or <c>all</c>): then it runs like any other row, so whoever fixes the defect can watch the row
/// fail before the fix and pass after it, and then replace this attribute with <c>[InlineData]</c>.
/// </summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = true)]
public sealed class KnownDefectDataAttribute : DataAttribute
{
    public const string EnvironmentVariable = "TS_TEST_RUN_KNOWN_DEFECTS";

    private readonly object[] _data;

    public KnownDefectDataAttribute(string defect, params object[] data)
    {
        Defect = defect;
        _data = data;
        Skip = Runs(defect) ? null : $"KnownDefect {defect}: {DirectCudaKnownDefects.Describe(defect)} (set {EnvironmentVariable}={defect} to run it)";
    }

    public string Defect { get; }

    public override IEnumerable<object[]> GetData(MethodInfo testMethod) => new[] { _data };

    public static bool Runs(string defect)
    {
        string value = Environment.GetEnvironmentVariable(EnvironmentVariable);
        if (string.IsNullOrWhiteSpace(value))
            return false;
        return value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Any(v => v.Equals("all", StringComparison.OrdinalIgnoreCase) || v.Equals(defect, StringComparison.OrdinalIgnoreCase));
    }
}

/// <summary>
/// The direct-CUDA defects found on the base revision by the direct-CUDA prefill and correctness
/// stages (evidence: docs/validation/direct-cuda-prefill/p0/defects and
/// docs/validation/direct-cuda-prefill/correctness/, git-ignored). Each is reproduced on the
/// laptop RTX 3080 (sm_86) on the synthetic files of DirectCudaSyntheticModels; managed cpu and
/// ggml_cpu pass the same check on the same file unless the entry says otherwise.
/// </summary>
internal static class DirectCudaKnownDefects
{
    public static string Describe(string defect) => defect switch
    {
        "D6" => "Nemotron-H MoE direct-CUDA prefill of more than 512 rows in one forward: the last prompt row can land far " +
                "from managed cpu and ggml_cpu (Q8_0 experts: 0.33 at 1100 tokens, the references agreeing to 6.6e-3; " +
                "MXFP4/Q5_1 experts at width 256: 1.05 at 513 tokens) while the decode rows after it are normal; up to " +
                "512 rows the Q8_0 file's row matches managed cpu to 6e-8, and the same 1100 tokens in 512-row chunks " +
                "match it to 7e-8",
        _ => "see docs/validation/direct-cuda-prefill/",
    };
}
