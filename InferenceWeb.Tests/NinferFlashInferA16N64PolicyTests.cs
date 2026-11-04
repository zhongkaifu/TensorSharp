// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda;
using Xunit;

public sealed class NinferFlashInferA16N64PolicyTests
{
    [Theory]
    [InlineData(1, 13376)]
    [InlineData(8, 17184)]
    public void LiveRowsAndOneZeroRowDetermineSeparateSharedReservations(int rows, int shared)
    {
        var plan = CudaNinferFlashInferA16N64Module.Plan(rows);
        int independentlySummed = 2 * (Math.Min(rows + 1, 16) * 68 * 4 + 64 * 20 * 4 + 1024);
        Assert.Equal(shared, independentlySummed); Assert.Equal(shared, plan.SharedBytes);
        Assert.Equal(rows * 5120L * 2, plan.InputBytes); Assert.Equal(rows * 34816L * 4, plan.OutputBytes);
    }
    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(9)]
    [InlineData(16)]
    public void StaticPrototypeDoesNotSilentlyPadOrChunk(int rows)
        => Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferFlashInferA16N64Module.Plan(rows));
    [Fact]
    public void OriginalN128PolicyCannotBeMislabeledAsN64()
        => Assert.Throws<ArgumentException>(() => CudaNinferFlashInferA16N64Module.ValidateTactic(CudaNinferFlashInferA16Module.NumericalPolicy, 4, 2));
    [Fact]
    public void OnlyTheExplicitN64TacticIsAdmitted()
    {
        var tactic = CudaNinferFlashInferA16N64Module.ValidateTactic(CudaNinferFlashInferA16N64Module.NumericalPolicy, 4, 2);
        Assert.Equal(4, tactic.Warps); Assert.Equal(2, tactic.Stages);
        Assert.Throws<ArgumentException>(() => CudaNinferFlashInferA16N64Module.ValidateTactic(tactic.Policy, 8, 2));
    }
}
