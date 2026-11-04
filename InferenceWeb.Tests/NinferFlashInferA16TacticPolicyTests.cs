// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda;
using Xunit;

public sealed class NinferFlashInferA16TacticPolicyTests
{
    [Theory]
    [InlineData(CudaNinferFlashInferA16Module.NumericalPolicy, 4, 2, 31232)]
    [InlineData(CudaNinferFlashInferA16Module.EightWarpNumericalPolicy, 8, 2, 31232)]
    [InlineData(CudaNinferFlashInferA16Module.ThreeStageNumericalPolicy, 4, 3, 46848)]
    public void OnlyExplicitConsistentTacticHasItsOwnIdentity(string policy, int warps, int stages, int bytes)
    {
        var tactic = CudaNinferFlashInferA16Module.ValidateTactic(policy, warps, stages);
        Assert.Equal(policy, tactic.Policy); Assert.Equal(warps, tactic.Warps);
        Assert.Equal(stages, tactic.Stages); Assert.Equal(bytes, tactic.SharedBytes);
    }
    [Theory]
    [InlineData(CudaNinferFlashInferA16Module.NumericalPolicy, 8, 2)]
    [InlineData(CudaNinferFlashInferA16Module.NumericalPolicy, 4, 3)]
    [InlineData(CudaNinferFlashInferA16Module.EightWarpNumericalPolicy, 4, 2)]
    [InlineData(CudaNinferFlashInferA16Module.ThreeStageNumericalPolicy, 8, 3)]
    [InlineData("unknown", 4, 2)]
    public void UnknownOrMislabeledTacticIsRejected(string policy, int warps, int stages)
        => Assert.Throws<ArgumentException>(() => CudaNinferFlashInferA16Module.ValidateTactic(policy, warps, stages));
}
