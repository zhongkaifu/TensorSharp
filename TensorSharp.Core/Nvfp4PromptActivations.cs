// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

namespace TensorSharp
{
    /// <summary>
    /// How an NVFP4 weight's products quantize their activation rows when there are more rows than the 8-bit matvec takes
    /// (prompt rows; --nvfp4-prompt-activations). Decode rows use 8-bit activations either way. Only GPUs with
    /// block-scaled FP4 tensor cores (CC 12.x) have a 4-bit route, so elsewhere both values compute the same.
    /// </summary>
    public enum Nvfp4PromptActivations
    {
        /// <summary>NVFP4 activations on the FP4 tensor cores (W4A4), as ggml-cuda and llama.cpp do by default. The default.</summary>
        Fp4 = 0,

        /// <summary>8-bit activations (W4A8): ggml-cuda's q8 product route, and on direct CUDA the int8 tensor-core GEMM.
        /// Lower perplexity, slower prompts.</summary>
        Q8 = 1,
    }
}
