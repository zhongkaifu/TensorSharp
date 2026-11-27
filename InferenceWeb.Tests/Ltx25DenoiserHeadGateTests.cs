using TensorSharp.Cuda;
using TensorSharp.Models.Ltx25;

namespace InferenceWeb.Tests;

public sealed partial class Ltx25DenoiserTests
{
    /// <summary>On CUDA the step program's regions take the kernels the direct engines ran: every attention on the F32
    /// attention with its 2 * sigmoid head gate fused, every roped q and k on the NeoX per-head table kernel with the q/k norm
    /// before it, every projection a direct product; the complete graph still matches the unchanged official LTX.</summary>
    [CudaTheory]
    [MemberData(nameof(Cases))]
    public void CudaServesAttentionGatesAndRotariesWithTheDirectKernels(int index)
    {
        using var checkpoint = new Checkpoint();
        using var allocator = new CudaAllocator();
        var reference = Oracle.GetProperty("cases")[index];
        using var model = new Ltx25Denoiser(checkpoint.Path, allocator);
        var executor = Assert.IsType<CudaGraphExecutor>(model.Executor);
        Assert.True(executor.DirectProducts);
        var actual = model.Forward(Input(reference));
        // Six attentions a block (video self and text, audio self and text, audio-to-video, video-to-audio), each gated.
        Assert.Equal(6L * actual.ExecutedBlocks, executor.FullAttentionsServedForTests);
        Assert.Equal(6L * actual.ExecutedBlocks, executor.HeadGatesServedForTests);
        // q and k of the four roped attentions a block.
        Assert.Equal(8L * actual.ExecutedBlocks, executor.HeadRotariesServedForTests);
        Assert.True(executor.DirectProductsForTests > 0);
        Equal(Floats(reference, "videoVelocity"), actual.VideoVelocity, 2e-5, "CUDA official video");
        Equal(Floats(reference, "audioVelocity"), actual.AudioVelocity, 2e-5, "CUDA official audio");
    }
}
