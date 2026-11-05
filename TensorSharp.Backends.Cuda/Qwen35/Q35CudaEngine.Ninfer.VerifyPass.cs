// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
namespace TensorSharp.Cuda;

public sealed unsafe partial class Q35CudaEngine
{
    /// <summary>The single-lane T8 verification pass the fixed-shape fused verify kernels serve.</summary>
    internal static bool UseNinferT8VerifyPass(bool enabled, int groups, int window, int lanes,
        bool verify, bool counts, bool embeddings, bool featureTaps, bool ggufDrafter, int tapFirst)
        => enabled && groups == 1 && window == 8 && lanes == 1 && verify && !counts
            && !embeddings && !featureTaps && !ggufDrafter && tapFirst == 0;
}
