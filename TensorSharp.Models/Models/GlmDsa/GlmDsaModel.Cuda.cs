// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// GLM-5.3-Flash on --backend cuda: the direct-CUDA whole-model engine
// (TensorSharp.Backends.Cuda/Glm), layer-split across the visible GPUs. The model's
// managed state stays on a host allocator; the engine owns every weight and cache.
using System;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    public partial class GlmDsaModel
    {
        /// <summary>Whether a checkpoint loads on the direct-CUDA engine: GLM-5.3-Flash on
        /// <c>--backend cuda</c>.</summary>
        internal static bool DirectCudaApplies(BackendType backend, GgufFile probe)
            => backend == BackendType.Cuda && GlmDsaArchitecture.IsGlm5Next(probe?.GetString("general.architecture"));

        /// <summary>The direct-CUDA engine places whole layers itself; the base class only needs a
        /// host allocator.</summary>
        private static BackendType ValidateDirectCuda(int tpDegree, ITensorParallelGroup tpGroup, int layerSplitDegree)
        {
            if (tpDegree < 1) throw new ArgumentOutOfRangeException(nameof(tpDegree));
            if (layerSplitDegree < 1) throw new ArgumentOutOfRangeException(nameof(layerSplitDegree));
            if (tpGroup != null || tpDegree > 1)
                throw new NotSupportedException(
                    "GLM-5.3-Flash on --backend cuda places whole layers per GPU; use --layer-split N instead of --tp.");
            ResolveNativeGpuCount(1, layerSplitDegree);
            return BackendType.Cpu;
        }

        private void InitCudaExecutor(string ggufPath, int layerSplitDegree, int maxContext)
        {
            // GLM-5.3-Flash advertises a 1M-token context; the MLA and indexer rows scale with it,
            // so keep a practical default unless MAX_CONTEXT names one.
            if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("MAX_CONTEXT")))
                maxContext = Math.Min(maxContext, 65536);
            int nGpu = ResolveNativeGpuCount(1, layerSplitDegree);
            int nUbatch = ParseEnvInt("TS_GLM_UBATCH", 1024);
            var exec = new GlmDsaCudaExecutor(ggufPath, maxContext, nUbatch, nGpu);
            _exec = exec;
            _nativeUbatch = exec.UBatch;
            _maxContextLength = exec.ContextSize;
            if (exec.VocabSize > 0)
                Config.VocabSize = exec.VocabSize;
            HasDraftHead = false;
            _logitsBuffer = new float[Config.VocabSize];
        }
    }
}
