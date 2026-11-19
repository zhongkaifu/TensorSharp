// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3-Image's conditioner on the diffusion runtime: the MiniMax-H3 text encoder (Qwen3-VL-32B truncated to 50
// layers, no final norm; the native trunk MiniMax-H3 video uses too) behind IConditioner. The raw prompt is tokenized with
// no chat template and the layer-50 hidden state of every token is the conditioning, exactly as MiniMaxH3ImageModel
// encodes it. Pictures (editing) are stage S5's.
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using TensorSharp.GGML;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3Image
{
    /// <summary>
    /// The MiniMax-H3 text encoder as an <see cref="IConditioner"/>. Making one maps the encoder GGUF, reads its tokenizer
    /// files and its small tensors on the host; its first encode makes the trunk resident. <see cref="ReleaseDevice"/>
    /// hands the trunk's device copies back and keeps the mapping, so the next encode uploads from memory.
    /// </summary>
    public sealed class MiniMaxH3ImageConditioner : IConditioner
    {
        private readonly MiniMaxH3TextEncoder _encoder;
        private readonly object _gate = new();
        private bool _disposed;

        /// <param name="path">The encoder GGUF (unsloth/MiniMax-H3-GGUF qwen3vl_32b_minimax_h3-*.gguf) with vocab.json,
        /// merges.txt and tokenizer_config.json beside it.</param>
        public MiniMaxH3ImageConditioner(string path, BackendType backend)
        {
            ArgumentNullException.ThrowIfNull(path);
            if (backend is not (BackendType.GgmlCuda or BackendType.GgmlMetal or BackendType.GgmlCpu))
                throw new NotSupportedException($"MiniMax-H3-Image's text encoder runs on ggml_cuda, ggml_metal or ggml_cpu, not {backend}.");
            Backend = backend;
            _encoder = new MiniMaxH3TextEncoder(path, null, backend, null);
        }

        public BackendType Backend { get; }

        /// <summary>The hidden width of a conditioning row.</summary>
        public int Width => _encoder.Config.Hidden;

        /// <summary>Tokens the last encode read (the timing line's "text tokens").</summary>
        public int LastTokens { get; private set; }

        public long WeightBytes => _encoder.DeviceWeightBytes;

        /// <summary>The trunk's compute memory for <paramref name="workload"/>'s prompt (<see cref="WorkingBytesFor"/>).</summary>
        public long WorkingBytes(DiffusionWorkload workload) => WorkingBytesFor(_encoder.Config, workload.Prompt);

        /// <summary>
        /// The trunk's compute memory for <paramref name="prompt"/>: a fixed part (the graph's buffers and the backend's
        /// handles beside the weights: 615 MiB measured on ggml_cuda for a 59-token prompt, much less on ggml_metal) and per
        /// token the hidden state's copies, the q/k/v rows and the MLP's gate and up rows in F32 (about three characters a
        /// token; prompts are a few hundred tokens at most).
        /// </summary>
        internal static long WorkingBytesFor(MiniMaxH3TextEncoderConfig c, string prompt)
        {
            ArgumentNullException.ThrowIfNull(c);
            long tokens = Math.Max(16, (prompt?.Length ?? 0) / 3 + 16);
            long perToken = 4L * (6L * c.Hidden + 2L * c.Heads * c.HeadDim + 2L * c.KvHeads * c.HeadDim + 3L * c.Intermediate);
            return 640L * 1024 * 1024 + tokens * perToken;
        }

        public ConditioningSet Encode(string prompt, IReadOnlyList<PresentedImage> pictures, CancellationToken cancellationToken)
        {
            if (pictures is { Count: > 0 })
                throw new NotSupportedException("MiniMax-H3-Image's conditioner does not read pictures yet (editing is stage S5).");
            cancellationToken.ThrowIfCancellationRequested();
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                // Raw prompt tokens (no chat template); an empty prompt is the encoder's BOS alone.
                List<int> ids = _encoder.Tokenize(prompt ?? string.Empty);
                if (ids.Count == 0) ids.Add(_encoder.Tokenizer.BosTokenId);
                float[] hidden = _encoder.Encode(ids);
                LastTokens = ids.Count;
                return new ConditioningSet(hidden, ids.Count, _encoder.Config.Hidden);
            }
        }

        /// <summary>The native trunk's compute buffers (MiniMax-H3's shared reuse allocator, which only this component uses
        /// on the runtime).</summary>
        public void ReleaseWorkspace()
        {
            lock (_gate)
                if (!_disposed) GgmlBasicOps.ReleaseReuseComputeBuffers();
        }

        public void ReleaseDevice()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _encoder.ReleaseDeviceResidency();
                GgmlBasicOps.ReleaseReuseComputeBuffers();
            }
        }

        public void Dispose()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _disposed = true;
                _encoder.Dispose();
            }
        }
    }
}
