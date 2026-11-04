// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class NinferFixedGovernorPolicyTests
{
    [Theory]
    [InlineData("qualified")] [InlineData("recipe")] [InlineData("sms")]
    [InlineData("verification")] [InlineData("native")] [InlineData("full-head")]
    [InlineData("window")] [InlineData("verify8")] [InlineData("short-context")]
    [InlineData("large-context")] [InlineData("graphs")] [InlineData("vector")]
    [InlineData("flashinfer")] [InlineData("exact-verify")] [InlineData("shortlist")]
    public void FixedGovernorRequiresTheCompleteMeasuredBackendProfile(string change)
        => Assert.Equal(change == "qualified", Q35CudaEngine.FixedDraftGovernorGeometry(
            change != "recipe", change == "sms" ? 169 : 170, change != "verification",
            change != "native", change != "full-head", change == "window" ? 6 : 7,
            change != "verify8", change == "short-context" ? 40959 : change == "large-context" ? 40961 : 40960,
            change != "graphs", change != "vector", change == "flashinfer", change == "exact-verify", change == "shortlist"));

    [Theory]
    [InlineData("auto-sm120", "/cuda_kernels/tensorsharp_q35_ninfer_a16_legacy_exact_vector.ptx", true)]
    [InlineData("explicit-legacy-exact", "/custom/tensorsharp_q35_ninfer_a16_legacy_exact_vector.fatbin", true)]
    [InlineData("explicit-original", "/custom/tensorsharp_q35_ninfer_a16_legacy_exact_vector.fatbin", false)]
    [InlineData("explicit-legacy-exact", "/custom/tensorsharp_q35_ninfer_a16_legacy_exact.fatbin", false)]
    [InlineData("asset-missing", null, false)]
    [InlineData("explicit-legacy-exact", "/custom/renamed-vector.fatbin", false)]
    public void FixedGovernorDeclinesOriginalScalarOrUnidentifiedA16Selections(string selection, string? path, bool expected)
        => Assert.Equal(expected, Q35CudaEngine.HasQualifiedGovernorVector(selection, path!));
}
