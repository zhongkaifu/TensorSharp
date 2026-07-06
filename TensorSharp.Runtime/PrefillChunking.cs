// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Shared chunked-prefill policy used by both the long-running server pipeline
    /// and the CLI interactive session. Keeping the chunk-size selection in one
    /// place ensures the two front-ends never drift apart on what counts as "too
    /// large to feed through the model in a single forward call".
    ///
    /// The chunk size caps the height of the [numHeads, seqLen, seqLen] attention
    /// score tensor at <c>numHeads * chunkSize * (chunkSize + prevContext)</c> so
    /// 32K-token prompts no longer try to allocate gigabytes of scores in one go.
    /// </summary>
    public static class PrefillChunking
    {
        /// <summary>
        /// Maximum tokens fed through the model in a single ForwardRefill call.
        /// GgmlCuda has wider kernel buffers and tolerates a larger chunk; other
        /// backends use a more conservative cap that still keeps the score tensor
        /// bounded under typical (16 head, 128 head-dim) configurations.
        /// </summary>
        public static int ResolveChunkSize(BackendType backend, int tokenCount)
        {
            if (tokenCount <= 0)
                return 0;

            return backend == BackendType.GgmlCuda || backend == BackendType.GgmlVulkan
                ? Math.Min(tokenCount, 5120)
                : Math.Min(tokenCount, 2048);
        }

        /// <summary>
        /// Convenience wrapper around <see cref="ResolveChunkSize"/> that returns true
        /// when the prompt would actually be split (so callers can avoid the chunked
        /// code path's per-chunk array copies for short prompts).
        /// </summary>
        public static bool ShouldChunk(BackendType backend, int tokenCount, out int chunkSize)
        {
            chunkSize = ResolveChunkSize(backend, tokenCount);
            return chunkSize > 0 && chunkSize < tokenCount;
        }
    }
}
