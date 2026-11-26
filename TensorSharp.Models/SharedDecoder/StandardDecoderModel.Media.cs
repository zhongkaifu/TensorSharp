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

        /// <summary><see cref="QueueMedia"/> for tests of the runtime's own media handling (families expose it through their
        /// multimodal interfaces).</summary>
        internal void QueueMediaForTests(Tensor embeddings, int insertPosition) => QueueMedia(embeddings, insertPosition);

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

        /// <summary>A forward takes the queued media rows; its first row is KV index <paramref name="absoluteStart"/>. A
        /// family that keeps data beside the rows for later layers (Qwen3-VL's DeepStack taps) anchors it here, once: a
        /// forward split into pieces injects each piece's rows, the first piece starting at the forward's first row.
        /// Default: nothing.</summary>
        protected virtual void OnMediaInjected(int absoluteStart) { }

        /// <summary>
        /// Whether media rows attend each other in both directions within the forward that carries them (Gemma 4's soft tokens:
        /// query q reads key k iff k is causally visible inside q's window, or q and k are both media rows of this forward;
        /// llama.cpp's mtmd non-causal image spans). The graph masks of such a forward open those cells; earlier forwards' media
        /// rows are reached causally only. Default: no (causal, as text).
        /// </summary>
        protected virtual bool MediaAttendsBidirectionally => false;

        /// <summary>The media rows of the forward being run (set when its media are injected, cleared when the next forward
        /// embeds), for <see cref="MediaAttendsBidirectionally"/>: row i of the forward is a media row; null without media.</summary>
        private bool[] _mediaRows;

        /// <summary>The media rows a step of <paramref name="rows"/> rows of <paramref name="sequences"/> sequences opens its masks
        /// for (<see cref="MediaAttendsBidirectionally"/>), or null.</summary>
        private bool[] MediaRowsOfStep(int rows, int sequences)
            => _mediaRows != null && sequences == 1 && _mediaRows.Length == rows ? _mediaRows : null;

        private unsafe void InjectMedia(Tensor hidden, int absoluteStart)
        {
            if (_pendingMedia.Count == 0) return;
            OnMediaInjected(absoluteStart);
            int dim = Config.HiddenSize, rows = (int)hidden.Sizes[0];
            if (MediaAttendsBidirectionally)
            {
                _mediaRows = new bool[rows];
                foreach (var (media, offset) in _pendingMedia)
                    for (int i = 0; i < media.Sizes[0] && offset + i < rows; i++) _mediaRows[offset + i] = true;
            }
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
