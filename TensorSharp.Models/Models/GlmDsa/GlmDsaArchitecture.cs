// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>GLM-5.x (DeepSeek Sparse Attention) and GLM-5.3-Flash architecture plug-in.</summary>
    internal static class GlmDsaArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            // glm-dsa: MLA + lightning indexer + sigmoid MoE.
            // glm5next (GLM-5.3-Flash): hybrid KDA linear attention + nope-only MLA with a
            // pooled DSA indexer, Sinkhorn hyper-connections, 288-expert MoE. Same class.
            Id = "glm-dsa",
            DisplayName = "GLM-5.x (DeepSeek Sparse Attention) and GLM-5.3-Flash",
            Aliases = new[] { "glm-dsa", "glm_dsa" },
            Factory = Create,
            Serving = GlmServing(flash: false),
            SupportsDistributedTensorParallel = false,
            ProjectorFileHints = new[] { "*mmproj*.gguf" },
        };

        // The hybrid Flash graph keeps each layer's recurrent and attention
        // state local and has rank-local TP segments. glm-dsa's reused indexer
        // state needs its own adapter before it can advertise these capabilities.
        public static ModelArchitectureDescriptor FlashDescriptor { get; } = new()
        {
            Id = "glm5next",
            DisplayName = "GLM-5.3-Flash",
            Aliases = new[] { "glm5next", "glm5-next" },
            Factory = CreateFlash,
            Serving = GlmServing(flash: true),
            SupportsDistributedLayerSplit = true,
            DistributedLayerSplitBackends = new[] { BackendType.GgmlCuda },
            SupportsDistributedTensorParallel = true,
            DistributedTensorParallelBackends = new[] { BackendType.GgmlCuda },
            ProjectorFileHints = new[] { "*mmproj*.gguf" },
        };

        private static bool CpuOversubscription() =>
            Environment.GetEnvironmentVariable("TS_GLM_NATIVE") != "0" &&
            Environment.GetEnvironmentVariable("TS_GLM_TP_OVERSUBSCRIBE") == "1";

        // A property, not a field: the descriptors above are initialized first.
        private static BackendType[] GlmBackends =>
            new[] { BackendType.Cpu, BackendType.GgmlCpu, BackendType.GgmlMetal, BackendType.GgmlCuda, BackendType.GgmlVulkan };

        private static ServingContract GlmServing(bool flash)
        {
            BackendType[] backends = flash ? GlmBackends.Append(BackendType.Cuda).ToArray() : GlmBackends;
            BackendType[] layerSplit = flash
                ? new[] { BackendType.GgmlCuda, BackendType.GgmlVulkan, BackendType.Cuda }
                : new[] { BackendType.GgmlCuda, BackendType.GgmlVulkan };
            return new ServingContract
            {
                Implementation = ServingImplementation.Bespoke,
                Backends = backends,
                HasRoutedExperts = true,
                // Native per-sequence slots; the managed per-op reference (--backend cpu) and the
                // direct-CUDA engine take turns.
                ContinuousBatching = MechanismSupport.On(ServingBackends.Ggml)
                    .ElsewhereGap("GLM's managed per-op reference and its direct-CUDA engine serve concurrent requests one at a time; use a GGML backend."),
                TensorParallel = MechanismSupport.On(ServingBackends.GgmlMultiDevice).KeepsBatching()
                    // Explicit CPU loopback fixture mode exercises native sharding; it is
                    // never evidence of multi-GPU execution or performance.
                    .WithLoopback(BackendType.GgmlCpu, CpuOversubscription)
                    .ElsewhereGap(flash
                        ? "GLM-5.3-Flash on --backend cuda places whole layers per GPU; use --layer-split N instead of --tp."
                        : "GLM tensor parallelism runs in the native ggml executor (ggml_cuda, ggml_vulkan)."),
                LayerSplit = MechanismSupport.On(layerSplit).KeepsBatchingOn(ServingBackends.GgmlMultiDevice),
                HybridParallel = MechanismSupport.None("GLM runs --tp and --layer-split as separate placement modes."),
                ExpertOffload = MechanismSupport.On(GlmBackends).ElsewhereGap(
                    "the direct-CUDA GLM-5.3-Flash engine keeps every routed expert in VRAM; use --backend ggml_cuda for --n-cpu-moe."),
            };
        }

        private static ModelBase CreateFlash(ModelCreateContext context)
            => new GlmDsaModel(context.GgufPath, context.Backend, context.TpDegree, context.TpGroup,
                context.LayerSplitDegree, GlmDsaModel.DirectCudaApplies(context.Backend, context.Probe));

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
