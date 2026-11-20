// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using TensorSharp.Models.Architecture;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    public partial class MuseSparkModel : IVisionCapableModel, IMultimodalPromptExpander
    {
        private MuseGlimmerVisionTower _visionEncoder;
        private readonly List<(Tensor embeddings, int position)> _pendingVisionEmbeddingsList = new();

        /// <summary>The vision tower of the mmproj (projector type muse-glimmer, Muse-Glimmer's tower at Spark's width) as graph
        /// programs on rank 0's executor, or null.</summary>
        public MuseGlimmerVisionTower VisionEncoder => _visionEncoder;
        bool IVisionCapableModel.IsVisionEncoderLoaded => _visionEncoder != null;
        internal bool HasPendingVisionEmbeddings => _pendingVisionEmbeddingsList.Count != 0;

        public void LoadVisionEncoder(string mmProjPath)
        {
            // A Glimmer projector has the same tower but a different text width;
            // reject it before allocating a multi-billion-parameter encoder.
            using (var gguf = new GgufFile(mmProjPath))
                ValidateVisionProjector(gguf, Config.HiddenSize);
            bool pinRank = IsGgmlBackend && (LayerSplitDegree > 1 || IsTensorParallel);
            int priorRank = pinRank ? GgmlBasicOps.GetActiveRank() : 0;
            bool priorPin = pinRank && GgmlBasicOps.PinRank(0);
            try
            {
                var encoder = new MuseGlimmerVisionTower(mmProjPath, _allocator, _backend);
                _visionEncoder?.Dispose();
                _visionEncoder = encoder;
            }
            finally
            {
                if (pinRank) GgmlBasicOps.RestoreRankPin(priorPin, priorRank);
            }
        }

        internal static void ValidateVisionProjector(GgufFile gguf, int hiddenSize)
        {
            // Mixed vision/audio projectors use the modality-specific key.
            string projector = gguf.GetString("clip.vision.projector_type")
                ?? gguf.GetString("clip.projector_type");
            if (projector != "muse-glimmer")
                throw new ArgumentException("Muse Spark requires its Muse-Glimmer-type vision projector.");
            if (gguf.GetUint32("clip.vision.projection_dim") != hiddenSize)
                throw new ArgumentException("The vision projector output width does not match the Muse Spark model.");
        }

        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition)
        {
            ArgumentNullException.ThrowIfNull(embeddings);
            if (insertPosition < 0)
                throw new ArgumentOutOfRangeException(nameof(insertPosition));
            if (embeddings.DimensionCount != 2 || embeddings.Sizes[0] < 1 || embeddings.Sizes[1] != Config.HiddenSize)
                throw new ArgumentException("Vision embeddings must have shape [tokens, hidden size].", nameof(embeddings));
            // The public contract uses an offset into the next submitted batch.
            // Absolute positions let a large image span survive internal chunking.
            int position = checked(_cacheSeqLen + insertPosition);
            _ = checked(position + (int)embeddings.Sizes[0]);
            _pendingVisionEmbeddingsList.Add((embeddings, position));
        }

        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(
            ModelMultimodalInjector injector, List<ChatMessage> history, List<int> inputTokens)
        {
            bool pinRank = IsGgmlBackend && (LayerSplitDegree > 1 || IsTensorParallel);
            int priorRank = pinRank ? GgmlBasicOps.GetActiveRank() : 0;
            bool priorPin = pinRank && GgmlBasicOps.PinRank(0);
            try { return injector.ProcessMuseGlimmerHistory(_visionEncoder, history, inputTokens); }
            finally
            {
                if (pinRank) GgmlBasicOps.RestoreRankPin(priorPin, priorRank);
            }
        }

        private void DrainPendingVisionEmbeddings(Tensor hidden, int startPos)
        {
            int endPos = checked(startPos + (int)hidden.Sizes[0]);
            for (int i = 0; i < _pendingVisionEmbeddingsList.Count;)
            {
                var (embeddings, position) = _pendingVisionEmbeddingsList[i];
                int spanEnd = checked(position + (int)embeddings.Sizes[0]);
                int from = Math.Max(position, startPos);
                int to = Math.Min(spanEnd, endPos);
                if (from < to)
                {
                    using var source = embeddings.Narrow(0, from - position, to - from);
                    using var target = hidden.Narrow(0, from - startPos, to - from);
                    if (ReferenceEquals(source.Allocator, target.Allocator))
                        Ops.Copy(target, source);
                    else
                    {
                        using var staged = new Tensor(hidden.Allocator, embeddings.ElementType, source.Sizes);
                        staged.SetElementsAsFloat(source.GetElementsAsFloat((int)source.ElementCount()));
                        Ops.Copy(target, staged);
                    }
                }
                if (spanEnd <= endPos)
                {
                    embeddings.Dispose();
                    _pendingVisionEmbeddingsList.RemoveAt(i);
                }
                else
                    ++i;
            }
        }

        private void ClearPendingVisionEmbeddings()
        {
            foreach (var (embeddings, _) in _pendingVisionEmbeddingsList)
                embeddings.Dispose();
            _pendingVisionEmbeddingsList.Clear();
        }

        private void DisposeVisionState()
        {
            ClearPendingVisionEmbeddings();
            _visionEncoder?.Dispose();
            _visionEncoder = null;
        }
    }
}
