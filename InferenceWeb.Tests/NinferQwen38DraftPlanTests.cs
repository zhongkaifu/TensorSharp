using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class NinferQwen38DraftPlanTests
{
    [Fact]
    public void BindsTheQ8G32DFlash2CompanionAndItsProposalHead()
    {
        using var f = NinferV3Qwen38.Build().Write();
        using var reader = new NinferArtifactReader(f.Path);
        var plan = NinferQwen38DraftPlan.Create(reader);
        Assert.True(plan.HasShortlist);
        var qkv = plan.GetTensor("dflash2/layers/0/attention/query_key_value");
        Assert.Equal(NinferQwen38DraftRole.QueryKeyValue, qkv.Role);
        Assert.Equal(NinferTensorFormat.W8G32Float16Scale, qkv.Descriptor.Format);
        Assert.Equal(new ulong[] { 6144, 5120 }, qkv.Descriptor.Shape);
        Assert.Equal(new ulong[] { 34816, 5120 }, plan.GetTensor("dflash2/layers/4/mlp/gate_up").Descriptor.Shape);
        Assert.Equal(NinferTensorFormat.BFloat16, plan.GetTensor("dflash2/candidate_selector/successor_codebook").Descriptor.Format);
        Assert.Equal(NinferTensorFormat.Q4G64Float16Scale, plan.GetTensor("proposal/head").Descriptor.Format);
    }

    [Fact]
    public void ProposalHeadIsOptional()
    {
        using var f = NinferV3Qwen38.Build(shortlist: false).Write();
        using var reader = new NinferArtifactReader(f.Path);
        Assert.False(NinferQwen38DraftPlan.Create(reader).HasShortlist);
    }

    [Fact]
    public void ArtifactsWithoutDFlash2AreRefused()
    {
        using var f = NinferV3Qwen38.Build(draft: false).Write();
        using var reader = new NinferArtifactReader(f.Path);
        Assert.Throws<NotSupportedException>(() => NinferQwen38DraftPlan.Create(reader));
    }
}
