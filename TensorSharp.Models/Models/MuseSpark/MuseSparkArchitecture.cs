// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models;

internal static class MuseSparkArchitecture
{
    public static ModelArchitectureDescriptor Descriptor { get; } = new()
    {
        Id = "muse-spark",
        DisplayName = "Muse Spark",
        Aliases = new[] { "muse-spark", "muse_spark" },
        Factory = c => new MuseSparkModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.LayerSplitDegree),
        Serving = new ServingContract
        {
            Implementation = ServingImplementation.Bespoke,
            Backends = new[] { BackendType.Cpu, BackendType.GgmlCpu, BackendType.GgmlMetal, BackendType.GgmlCuda, BackendType.GgmlVulkan },
            HasRoutedExperts = true,
            // Native slots on ggml_cuda / ggml_vulkan; the per-op path batches packed paged text elsewhere.
            ContinuousBatching = MechanismSupport.On(BackendType.Cpu, BackendType.GgmlCpu, BackendType.GgmlMetal,
                BackendType.GgmlCuda, BackendType.GgmlVulkan),
            // --tp runs the native executor on ggml_cuda and the per-op path on ggml_vulkan.
            TensorParallel = MechanismSupport.On(BackendType.GgmlCuda, BackendType.GgmlVulkan).KeepsBatchingOn(BackendType.GgmlCuda),
            LayerSplit = MechanismSupport.On(BackendType.GgmlCuda, BackendType.GgmlVulkan).KeepsBatching(),
            HybridParallel = MechanismSupport.None("Muse Spark tensor parallelism and layer split are separate placement modes."),
            ExpertOffload = MechanismSupport.On(BackendType.Cpu, BackendType.GgmlCpu, BackendType.GgmlMetal,
                BackendType.GgmlCuda, BackendType.GgmlVulkan),
        },
        SupportsDistributedTensorParallel = false,
        ProjectorFileHints = new[] { "*mmproj*Muse*Spark*.gguf", "*mmproj*.gguf" },
    };
}
