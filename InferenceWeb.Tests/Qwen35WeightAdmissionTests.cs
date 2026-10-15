// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Models;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

public sealed class Qwen35WeightAdmissionTests
{
    [Fact]
    public void AdmittedTrunkPolicyCannotLoadDraftWeightsAfterEnvironmentPolicyChanges()
    {
        var policy = new ModelMemoryPolicy(1024, 64) { OmitEmbeddedDraftWeights = true };
        Assert.False(Qwen35Model.ShouldLoadEmbeddedMtpWeights(SpeculationOptions.Disabled, policy));
        Assert.False(Qwen35Model.ShouldLoadEmbeddedMtpWeights(new() { Enabled = true }, policy));
        Assert.True(Qwen35Model.ShouldLoadEmbeddedMtpWeights(new() { Enabled = true }, new(1024, 64)));
    }

    [Fact]
    public void UnconfiguredDirectApiRetainsItsLearnedDraftCapability()
    {
        Assert.True(Qwen35Model.ShouldLoadEmbeddedMtpWeights(SpeculationOptions.Disabled));
        Assert.True(Qwen35Model.ShouldLoadEmbeddedMtpWeights(new() { Enabled = true }));
    }

    [Theory]
    [InlineData("auto")]
    [InlineData("draft-head")]
    [InlineData("ngram")]
    public void ExplicitDisableAlwaysOmitsTheUnusedLayer(string algorithm)
        => Assert.False(Qwen35Model.ShouldLoadEmbeddedMtpWeights(new() {
            Enabled = false, ExplicitlyDisabled = true, SpeculatorName = algorithm }));

    [Fact]
    public void NGramKeepsTrunkVerificationWithoutAdmittingLearnedWeights()
    {
        Assert.False(Qwen35Model.ShouldLoadEmbeddedMtpWeights(new() { Enabled = true, SpeculatorName = "NGRAM" }));
        Assert.True(Qwen35Model.ShouldLoadEmbeddedMtpWeights(new() { Enabled = false, SpeculatorName = "ngram" }));
    }

    [Theory]
    [InlineData("blk.39.ffn_gate_exps.weight", false)]
    [InlineData("blk.40.ffn_gate_exps.weight", true)]
    [InlineData("blk.40.attn_norm.weight", true)]
    [InlineData("blk.40.nextn.eh_proj.weight", true)]
    [InlineData("blk.41.attn_q.weight", true)]
    [InlineData("blk.42.attn_q.weight", false)]
    [InlineData("blk.400.attn_q.weight", false)]
    [InlineData("blk.-1.attn_q.weight", false)]
    [InlineData("dflash.blk.40.attn_q.weight", false)]
    [InlineData("v.blk.40.attn_q.weight", false)]
    [InlineData("output.weight", false)]
    [InlineData("token_embd.weight", false)]
    public void AdmissionOnlyOmitsTheDeclaredTrailingDecoderBlocks(string name, bool expected)
        => Assert.Equal(expected, Qwen35Model.IsEmbeddedMtpWeight(name, 40, 2));

    [Fact]
    public void NoNextnMetadataDoesNotRemoveAnOtherwiseNamedLayer()
        => Assert.False(Qwen35Model.IsEmbeddedMtpWeight("blk.40.nextn.eh_proj.weight", 40, 0));
}
