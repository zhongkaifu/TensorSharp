// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>GLM-5.x (DeepSeek Sparse Attention) and GLM-5.3-Flash architecture plug-in.</summary>
    internal static class GlmDsaArchitecture
    {
        // Both published GGUF spellings use the same schema. Keep the spelling
        // from the file when reading metadata: it is also the key prefix.
        internal static bool IsGlm5Next(string architecture)
            => architecture is "glm5next" or "glm5-next";

        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            // glm-dsa: MLA + lightning indexer + sigmoid MoE.
            // glm5next (GLM-5.3-Flash): hybrid KDA linear attention + nope-only MLA with a
            // pooled DSA indexer, Sinkhorn hyper-connections, 288-expert MoE. Same class.
            Id = "glm-dsa",
            DisplayName = "GLM-5.x (DeepSeek Sparse Attention) and GLM-5.3-Flash",
            Aliases = new[] { "glm-dsa", "glm_dsa", "glm5next", "glm5-next" },
            Factory = Create,
            SupportsLayerSplit = true,
            // GLM-5.3-Flash also places whole layers on --backend cuda (the direct-CUDA engine).
            LayerSplitBackends = new[] { BackendType.GgmlCuda, BackendType.GgmlVulkan, BackendType.Cuda },
            SupportsDistributedTensorParallel = false,
            // Explicit CPU loopback fixture mode exercises native sharding; it is
            // never evidence of multi-GPU execution or performance.
            SupportsNativeTensorParallel = (_, backend) => backend == BackendType.GgmlCpu &&
                Environment.GetEnvironmentVariable("TS_GLM_NATIVE") != "0" &&
                Environment.GetEnvironmentVariable("TS_GLM_TP_OVERSUBSCRIBE") == "1",
            ProjectorFileHints = new[] { "*mmproj*.gguf" },
        };

        private static ModelBase Create(ModelCreateContext context)
        {
            // The native whole-model executor owns its local GPU ranks and does
            // not consume the managed group's cross-node collectives. Accepting
            // one would make every node run an independent local model while the
            // CLI advertised distributed TP.
            if (context.TpGroup != null)
            {
                throw new NotSupportedException(
                    "GLM 5.x tensor parallelism is local/single-process only and cannot use " +
                    "--tp-node-id/--tp-peers. Use --tp N without the node options.");
            }

            return new GlmDsaModel(context.GgufPath, context.Backend, context.TpDegree, null,
                context.LayerSplitDegree, GlmDsaModel.DirectCudaApplies(context.Backend, context.Probe));
        }
    }
}
