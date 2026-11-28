using System;
using System.Runtime.InteropServices;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

internal static class Ltx25Architecture
{
    public static ModelArchitectureDescriptor Descriptor { get; } = new()
    {
        Id = Ltx25Model.ArchitectureId,
        DisplayName = "LTX 2.5 Distilled",
        Aliases = new[] { Ltx25Model.ArchitectureId, "ltx", "ltxv", "ltx2", "ltx-2", "ltx_2_5" },
        Factory = context => new Ltx25Model(context.GgufPath, context.Backend, context.LayerSplitDegree, context.TpDegree, context.TpGroup),
        ApplyNativeTunables = ConfigurePrecision,
        DetectFromTensors = file => file.Tensors.ContainsKey("patchify_proj.weight") &&
            file.Tensors.ContainsKey("audio_patchify_proj.weight") &&
            file.Tensors.ContainsKey("transformer_blocks.0.prompt_scale_shift_table") &&
            file.Tensors.ContainsKey("transformer_blocks.0.attn1.to_gate_logits.weight"),
        // A clip runs on the shared diffusion runtime's video pipeline (DiffusionVideoPipeline with Ltx25Family). On the CUDA
        // tensor executor (--backend cuda) the denoiser's 48 blocks split over GPUs as whole-block stages (DenoiserStages,
        // --layer-split N) or shard over them by heads and feed-forward width (DenoiserRanks, --tp N); the ggml_cuda name
        // selects that executor on one GPU, and tensor-parallel stages (both at once) are not implemented.
        Serving = ServingContract.SharedDiffusionRuntime(
            new[] { BackendType.Cpu, BackendType.Cuda, BackendType.Mlx, BackendType.GgmlMetal, BackendType.GgmlCuda },
            tensorParallelBackends: new[] { BackendType.Cuda },
            tensorParallelGap: "ltx-2.5 shards its denoiser's blocks over GPUs on --backend cuda (its CUDA tensor executor); " +
                "ggml_cuda selects that executor on one GPU: use --backend cuda --tp N.",
            layerSplitGap: "ltx-2.5 splits its blocks over GPUs on --backend cuda (its CUDA tensor executor); ggml_cuda selects " +
                "that executor on one GPU: use --backend cuda --layer-split N.",
            layerSplitBackends: new[] { BackendType.Cuda }),
        // The denoiser's ranks are this process's GPUs (DenoiserRanks refuses a group over nodes): --tp-node-id/--tp-peers is
        // refused before any GPU is touched.
        SupportsDistributedTensorParallel = false,
    };

    private static void ConfigurePrecision(ModelCreateContext context)
    {
        if (context.Backend is not (BackendType.Mlx or BackendType.GgmlMetal) ||
            !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("MLX_ENABLE_TF32"))) return;
        // MLX caches this choice on its first GEMM. Configure the original F32
        // quality path before this family's allocator/graphs are initialized.
        // A process that previously ran MLX should set it at launch instead.
        if (OperatingSystem.IsMacOS() || OperatingSystem.IsMacCatalyst())
        {
            if (SetEnvironment("MLX_ENABLE_TF32", "0", 1) != 0)
                throw new InvalidOperationException("Unable to configure LTX Float32 MLX precision.");
            Environment.SetEnvironmentVariable("MLX_ENABLE_TF32", "0");
        }
    }

    [DllImport("/usr/lib/libSystem.B.dylib", EntryPoint = "setenv", CallingConvention = CallingConvention.Cdecl)]
    private static extern int SetEnvironment(string name, string value, int overwrite);
}
