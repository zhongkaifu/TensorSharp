// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

namespace TensorSharp.Models
{
    /// <summary>
    /// Media embeddings (an image's encoder rows), once for every family: the multimodal injector
    /// queues each span's rows at its offset in the next forward, and the forward copies them over the
    /// placeholder tokens' embeddings before the first layer.
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        private readonly List<(Tensor Rows, int Offset)> _pendingMedia = new();

        /// <summary>Queue <paramref name="embeddings"/> [rows, hidden] for the next forward, written at
        /// its row <paramref name="insertPosition"/>; the forward takes ownership. A family with a media
        /// tower exposes this as its <c>IVisionCapableModel.SetVisionEmbeddings</c>.</summary>
        protected void QueueMedia(Tensor embeddings, int insertPosition)
        {
            ArgumentNullException.ThrowIfNull(embeddings);
            RefuseMediaOffFirstStage();
            if (embeddings.DimensionCount != 2 || embeddings.Sizes[1] != Config.HiddenSize)
                throw new ArgumentException(
                    $"{Config.Architecture}: media embeddings are [{string.Join(", ", embeddings.Sizes.ToArray())}], the decoder takes rows of {Config.HiddenSize}.",
                    nameof(embeddings));
            if (insertPosition < 0) throw new ArgumentOutOfRangeException(nameof(insertPosition));
            _pendingMedia.Add((embeddings, insertPosition));
        }

        /// <summary>True while media embeddings wait for the forward that holds their rows (prompts are
        /// then not chunked, so the offsets stay the injector's).</summary>
        protected bool HasPendingMedia => _pendingMedia.Count > 0;

        /// <summary>Media rows are embedded where this node embeds tokens. A multi-node pipeline whose
        /// first stage is another node (node 0 runs the last stage by default) refuses them rather than
        /// dropping them.</summary>
        private void RefuseMediaOffFirstStage()
        {
            if (LayerSplitGroup?.NodeCount > 1 && LayerPlacement != null && !LayerPlacement.IsFirst)
                throw new NotSupportedException(
                    "Multi-node layer split runs node 0 as the last stage, so it cannot inject media embeddings. " +
                    $"Set {LayerSplitPlan.DriverStageVariable}=first on every node to serve images.");
        }

        private unsafe void InjectMedia(Tensor hidden, int absoluteStart)
        {
            if (_pendingMedia.Count == 0) return;
            int dim = Config.HiddenSize, rows = (int)hidden.Sizes[0];
            float* destination = GetFloatPtr(hidden);
            try
            {
                foreach (var (media, offset) in _pendingMedia)
                {
                    int count = (int)media.Sizes[0];
                    if (offset + count > rows)
                        throw new InvalidOperationException(
                            $"{Config.Architecture}: {count} media rows at offset {offset} overrun a {rows}-row forward (absolute position {absoluteStart}).");
                    long bytes = (long)count * dim * sizeof(float);
                    Buffer.MemoryCopy(GetFloatPtr(media), destination + (long)offset * dim, bytes, bytes);
                }
            }
            finally
            {
                DropPendingMedia();
            }
            HostWritten(hidden);
        }

        private void DropPendingMedia()
        {
            foreach (var (media, _) in _pendingMedia) media.Dispose();
            _pendingMedia.Clear();
        }
    }
}
