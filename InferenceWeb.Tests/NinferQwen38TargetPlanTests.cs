using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

// The full NInfer v3 Qwen3.8-27B binding layout in a sparse generated file.
public sealed class NinferQwen38TargetPlanTests
{
    [Fact]
    public void BindsTheV3TextModelWithItsFormatsAndCalibration()
    {
        using var f = NinferV3Qwen38.Build(draft: false).Write();
        using var reader = new NinferArtifactReader(f.Path);
        var plan = NinferQwen38TargetPlan.Create(reader);
        Assert.Equal(64, plan.Layers.Count);
        Assert.Equal(NinferTensorFormat.Fp8E4M3FnRowBFloat16Scale, plan.TokenEmbedding.Descriptor.Format);
        Assert.Equal(NinferTensorFormat.Fp8E4M3FnRowBFloat16Scale, plan.OutputHead.Descriptor.Format);
        var gdn = plan.Layers[0];
        Assert.Equal(NinferQwen38LayerKind.GatedDeltaNet, gdn.Kind);
        Assert.Equal(new ulong[] { 16384, 5120 }, gdn.InputProjection.Descriptor.Shape);
        Assert.Equal(NinferQwen38RowOrder.GdnQueryKeyValueZ, gdn.InputProjection.RowOrder);
        Assert.Equal(NinferActivationPolicy.AllowA8, gdn.InputProjection.Policy);
        Assert.Equal(NinferTensorFormat.BFloat16, gdn.AlphaBeta!.Descriptor.Format);
        Assert.Equal(NinferTensorFormat.NvFp4, gdn.GateUp.Descriptor.Format);
        Assert.Equal(836f, reader.ReadScalar(gdn.GateUp.InputScaleDivisor!));
        Assert.Equal(161f, reader.ReadScalar(gdn.Down.InputScaleDivisor!));
        var attention = plan.Layers[3];
        Assert.Equal(NinferQwen38LayerKind.GroupedQueryAttention, attention.Kind);
        Assert.Equal(NinferQwen38RowOrder.AttentionQueryKeyGateValue, attention.InputProjection.RowOrder);
        Assert.Equal(new ulong[] { 14336, 5120 }, attention.InputProjection.Descriptor.Shape);
        var late = plan.Layers[56];
        Assert.Equal(NinferTensorFormat.Fp8E4M3FnRowBFloat16Scale, late.GateUp.Descriptor.Format);
        Assert.Null(late.GateUp.InputScaleDivisor);
        Assert.Equal(NinferQwen38NormConvention.StoredUnitOffset, plan.FinalNorm.NormConvention);
        Assert.Equal(NinferQwen38NormConvention.StoredPlain, gdn.GdnNorm!.NormConvention);
    }

    [Fact]
    public void GateAndUpMustShareTheirActivationDivisor()
    {
        using var f = NinferV3Qwen38.Build(draft: false, upDivisor: 660f).Write();
        using var reader = new NinferArtifactReader(f.Path);
        var error = Assert.Throws<InvalidDataException>(() => NinferQwen38TargetPlan.Create(reader));
        Assert.Contains("activation divisors", error.Message);
    }

    [Fact]
    public void FusedPartsMustTileTheirObjectInOrder()
    {
        var b = NinferV3Qwen38.Build(draft: false, layerOverride: (b, i) =>
        {
            if (i != 0) { NinferV3Qwen38.Layer(b, i); return; }
            string p = "text/layers/0/";
            b.Bind(p + "input_norm", b.Tensor("bf16", new ulong[] { 5120 }));
            b.Bind(p + "post_attention_norm", b.Tensor("bf16", new ulong[] { 5120 }));
            // key before query: the stored order is Q|K|V|Z.
            b.Fused("fp8_e4m3fn_row_bf16", 5120, new[] { (p + "gdn/key", 2048UL), (p + "gdn/query", 2048UL), (p + "gdn/value", 6144UL), (p + "gdn/z", 6144UL) }, "AllowA8");
        });
        using var f = b.Write();
        using var reader = new NinferArtifactReader(f.Path);
        Assert.Throws<InvalidDataException>(() => NinferQwen38TargetPlan.Create(reader));
    }

    [Fact]
    public void OtherModelsAreRefused()
    {
        var b = NinferV3Qwen38.Build(draft: false);
        b.Model = "qwen3.6-27b";
        using var f = b.Write();
        using var reader = new NinferArtifactReader(f.Path);
        Assert.Throws<NotSupportedException>(() => NinferQwen38TargetPlan.Create(reader));
    }

    [Fact]
    public void GdnHeadGroupingIsThreeValueHeadsPerKeyHead()
    {
        Assert.Equal(0, NinferQwen38TargetPlan.GdnQueryKeyHead(2));
        Assert.Equal(15, NinferQwen38TargetPlan.GdnQueryKeyHead(47));
        Assert.Throws<ArgumentOutOfRangeException>(() => NinferQwen38TargetPlan.GdnQueryKeyHead(48));
    }
}
