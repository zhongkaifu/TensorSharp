// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

using TensorSharp.Runtime;

namespace TensorSharp.Models.Architecture
{
    /// <summary>
    /// Everything an architecture plug-in needs to build its model, passed as one
    /// object so a new construction input (a companion GGUF, a device split, ...)
    /// can be added without touching every architecture's factory.
    ///
    /// <see cref="Probe"/> is the already-open GGUF the loader used to identify the
    /// architecture. It is owned by the loader and closed when
    /// <see cref="ModelBase.Create"/> returns, so a factory may read metadata from
    /// it but must not retain it.
    /// </summary>
    public sealed class ModelCreateContext
    {
        public ModelCreateContext(string ggufPath, BackendType backend, GgufFile probe,
            int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            string draftModelPath = null, int layerSplitDegree = 1, WeightStreamingOptions weightStreaming = null)
            : this(ggufPath, backend, probe, tpDegree, tpGroup, draftModelPath, layerSplitDegree, weightStreaming, null)
        {
        }

        public ModelCreateContext(string ggufPath, BackendType backend, GgufFile probe,
            int tpDegree, ITensorParallelGroup tpGroup,
            string draftModelPath, int layerSplitDegree, WeightStreamingOptions weightStreaming, ModelMemoryPolicy memoryPolicy)
        {
            GgufPath = ggufPath ?? throw new ArgumentNullException(nameof(ggufPath));
            Backend = backend;
            Probe = probe;
            TpDegree = tpDegree;
            TpGroup = tpGroup;
            DraftModelPath = draftModelPath;
            LayerSplitDegree = layerSplitDegree;
            WeightStreaming = weightStreaming;
            MemoryPolicy = memoryPolicy;
        }

        /// <summary>Path of the main model GGUF.</summary>
        public string GgufPath { get; }

        /// <summary>Compute backend the model must build itself on.</summary>
        public BackendType Backend { get; }

        /// <summary>Open GGUF used for architecture detection. Do not retain or dispose.</summary>
        public GgufFile Probe { get; }

        /// <summary>Tensor-parallel degree AFTER the multi-GPU gate has run: 1 unless
        /// the architecture actually shards weights.</summary>
        public int TpDegree { get; }

        /// <summary>Live tensor-parallel group, or null for single-device, layer-split,
        /// and architecture-owned native executors.</summary>
        public ITensorParallelGroup TpGroup { get; }

        /// <summary>Optional speculative-decoding draft model GGUF; null when the run
        /// asked for none. Architectures without a drafter ignore it.</summary>
        public string DraftModelPath { get; }

        /// <summary>Explicit number of GPUs to spread whole layers across, for
        /// architectures that implement layer placement; 1 when not requested.</summary>
        public int LayerSplitDegree { get; }

        /// <summary>Explicit bounded weight execution; unsupported adapters must reject it.</summary>
        public WeightStreamingOptions WeightStreaming { get; }

        public ModelMemoryPolicy MemoryPolicy { get; }

        /// <summary>Rebuild with a different tensor-parallel / layer-split resolution.</summary>
        public ModelCreateContext With(int tpDegree, ITensorParallelGroup tpGroup, int layerSplitDegree)
            => new(GgufPath, Backend, Probe, tpDegree, tpGroup, DraftModelPath, layerSplitDegree, WeightStreaming, MemoryPolicy);
    }
}
