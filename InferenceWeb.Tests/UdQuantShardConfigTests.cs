using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// Regression tests for the community/UD-quant weight layouts the shipped
/// configs never exercised:
///  - unsloth's gpt-oss requants quantize Q, K and V with DIFFERENT types
///    (Q2_K: IQ4_NL/IQ4_NL/Q5_0; Q4_K_M: per-layer Q5_0 vs Q8_0 attn_v), so
///    load-time QKV fusion must be all-or-nothing — a per-layer decision left
///    half the layers fused while the model-wide name table said "separate",
///    aborting the first forward (exit 134) and, worse, the unconditional
///    bias fusion orphaned attn_q/k/v.bias so every backend silently dropped
///    the attention projection biases (garbage/empty output on CUDA at tp=1).
///  - Qwen3.5 UD-IQ2_XXS ships ssm_out.weight as Q4_K, whose 256-element
///    super-block does not divide the 128-column V-head slice: the TP
///    block-cyclic gather silently produced zero-byte shards that crashed the
///    tp=2 preload ("PreloadQuantizedWeight requires valid cache key, host
///    data, and size").
/// </summary>
public class UdQuantShardConfigTests
{
    private static (int ggmlType, long ne0)? Info(
        Dictionary<(int layer, int proj), (int type, long ne0)> map, int idx)
    {
        int layer = idx / 3, proj = idx % 3;
        return map.TryGetValue((layer, proj), out var v) ? (v.type, v.ne0) : null;
    }

    [Fact]
    public void QkvFusion_UniformTypes_FusesAllLayers()
    {
        // The ggml-org reference quants (all projections Q8_0) must keep fusing.
        var map = new Dictionary<(int, int), (int, long)>();
        for (int l = 0; l < 24; l++)
            for (int p = 0; p < 3; p++)
                map[(l, p)] = ((int)GgmlTensorType.Q8_0, 2880);

        Assert.True(GptOssModel.CanFuseAllQkvLayers(24, idx => Info(map, idx), _ => false));
    }

    [Fact]
    public void QkvFusion_MixedTypesWithinLayer_DeclinesForWholeModel()
    {
        // unsloth gpt-oss-20b-Q2_K: q/k = IQ4_NL, v = Q5_0 on every layer.
        var map = new Dictionary<(int, int), (int, long)>();
        for (int l = 0; l < 24; l++)
        {
            map[(l, 0)] = ((int)GgmlTensorType.IQ4_NL, 2880);
            map[(l, 1)] = ((int)GgmlTensorType.IQ4_NL, 2880);
            map[(l, 2)] = ((int)GgmlTensorType.Q5_0, 2880);
        }

        Assert.False(GptOssModel.CanFuseAllQkvLayers(24, idx => Info(map, idx), _ => false));
    }

    [Fact]
    public void QkvFusion_MixedTypesAcrossLayers_DeclinesForWholeModel()
    {
        // unsloth gpt-oss-20b-Q4_K_M: half the layers are uniformly Q5_0
        // (fusable), the other half keep attn_v at Q8_0 (not fusable). The old
        // per-layer decision fused exactly the 12 uniform layers and crashed
        // the forward; the policy must refuse to fuse ANY layer here so the
        // model-wide separate-QKV name table stays truthful.
        var map = new Dictionary<(int, int), (int, long)>();
        for (int l = 0; l < 24; l++)
        {
            int vType = (l % 2 == 0) ? (int)GgmlTensorType.Q8_0 : (int)GgmlTensorType.Q5_0;
            map[(l, 0)] = ((int)GgmlTensorType.Q5_0, 2880);
            map[(l, 1)] = ((int)GgmlTensorType.Q5_0, 2880);
            map[(l, 2)] = (vType, 2880);
        }

        Assert.False(GptOssModel.CanFuseAllQkvLayers(24, idx => Info(map, idx), _ => false));
    }

    [Fact]
    public void QkvFusion_AlreadyFusedGguf_DeclinesQuietly()
    {
        // A GGUF that ships attn_qkv pre-fused has no separate q/k/v at all:
        // fusion has nothing to do and must simply decline.
        Assert.False(GptOssModel.CanFuseAllQkvLayers(24, _ => null, _ => false));
    }

    [Fact]
    public void QkvFusion_F32Triplets_StillFuse()
    {
        // Float (unquantized) q/k/v layers fuse through the F32 branch.
        Assert.False(GptOssModel.CanFuseAllQkvLayers(2, _ => null, l => l == 0));
        Assert.True(GptOssModel.CanFuseAllQkvLayers(2, _ => null, _ => true));
    }

}
