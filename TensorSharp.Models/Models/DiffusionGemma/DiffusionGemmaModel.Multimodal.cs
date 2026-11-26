// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.Cpu;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>One request's (or the model's own) image spans: projected rows [soft tokens, hidden] at their first soft token's
    /// prompt position, ascending; <see cref="Version"/> moves with every change, so a cache prefilled with other images is
    /// never reused for these.</summary>
    internal sealed class VisionSet : IDisposable
    {
        private static int s_versions;

        public List<(Tensor Rows, int Position)> Spans { get; } = new();

        public int Version { get; private set; } = System.Threading.Interlocked.Increment(ref s_versions);

        /// <summary>Install <paramref name="rows"/> at <paramref name="position"/> (taking ownership): a span already at that
        /// position is replaced, an overlapping one refused.</summary>
        public void Set(Tensor rows, int position)
        {
            int count = (int)rows.Sizes[0];
            for (int i = 0; i < Spans.Count; i++)
            {
                (Tensor existing, int at) = Spans[i];
                if (at == position)
                {
                    if (!ReferenceEquals(existing, rows)) existing.Dispose();
                    Spans[i] = (rows, position);
                    Bump();
                    return;
                }
                if (position < at + existing.Sizes[0] && at < position + count)
                {
                    rows.Dispose();
                    throw new ArgumentException($"Image span [{position},{position + count}) overlaps the installed span " +
                        $"[{at},{at + existing.Sizes[0]}). Clear the spans between requests.", nameof(position));
                }
            }
            Spans.Add((rows, position));
            Spans.Sort(static (a, b) => a.Position.CompareTo(b.Position));
            Bump();
        }

        private void Bump() => Version = System.Threading.Interlocked.Increment(ref s_versions);

        public void Dispose()
        {
            foreach ((Tensor rows, _) in Spans) rows.Dispose();
            Spans.Clear();
            Bump();
        }
    }

    /// <summary>
    /// Image input for <see cref="DiffusionGemmaModel"/>: the Gemma 4 SigLIP tower (<see cref="Gemma4VisionEncoder"/>, projector
    /// <c>gemma4v</c>, loaded from an mmproj GGUF or the HuggingFace <c>.safetensors</c> shard), Gemma 4's prompt expansion
    /// (<c>&lt;|image&gt;</c> into BOI + soft rows + EOI; no audio tower), and the projected rows written over their placeholder
    /// rows when the prompt prefills (<see cref="StandardDecoderModel.QueueMedia"/>, after the sqrt(n_embd) scale, which skips
    /// them). The spans are retained, not consumed: a block's prompt reuses the rows the earlier blocks cached, and a prompt the
    /// cache does not hold prefills its images again. The model's own set serves the single-request paths; the batched
    /// scheduler's requests each carry theirs (<see cref="SetSequenceVisionEmbeddings"/>).
    /// </summary>
    public sealed partial class DiffusionGemmaModel : IVisionCapableModel, IMultimodalPromptExpander
    {
        private Gemma4VisionEncoder _visionEncoder;
        private readonly VisionSet _ownVision = new();
        private VisionSet _visionOverride;

        /// <summary>The spans the model's own prompts read: a sequence's during <see cref="UseSequenceVision"/>, else the model's.</summary>
        private VisionSet ActiveVision => _visionOverride ?? _ownVision;

        /// <summary>The loaded gemma4v tower, or null.</summary>
        public Gemma4VisionEncoder VisionEncoder => _visionEncoder;

        bool IVisionCapableModel.IsVisionEncoderLoaded => _visionEncoder != null;

        /// <summary>True while the model's own image spans are installed.</summary>
        public bool HasPendingVisionEmbeddings => _ownVision.Spans.Count > 0;

        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(ModelMultimodalInjector injector, List<ChatMessage> history, List<int> inputTokens)
            => injector.ProcessGemma4VisionHistory(_visionEncoder, history, inputTokens);

        /// <param name="projectorPath">An mmproj GGUF, or a HuggingFace <c>.safetensors</c> shard with the
        /// <c>model.encoder.vision_tower.*</c> tower (the published DiffusionGemma GGUFs carry no vision tensors).</param>
        public void LoadVisionEncoder(string projectorPath)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(projectorPath);
            // As Gemma 4's: the direct CUDA backend's gemma4v stack diverged numerically, so its tower runs on the CPU.
            IAllocator visionAllocator = _backend == BackendType.Cuda ? new CpuAllocator(BlasEnum.DotNet) : _allocator;
            var encoder = new Gemma4VisionEncoder(projectorPath, visionAllocator);
            if (encoder.ProjectionDim != Config.HiddenSize)
            {
                encoder.Dispose();
                throw new InvalidOperationException($"Vision tower '{projectorPath}' projects to {encoder.ProjectionDim} dimensions but this " +
                    $"DiffusionGemma has embedding_length {Config.HiddenSize}. The tower does not belong to this model.");
            }
            _visionEncoder?.Dispose();
            _visionEncoder = encoder;
            _visionEncoder.SetHostModel(this);
        }

        /// <summary>Install one image's projected rows at <paramref name="insertPosition"/> (its first soft token in the prompt)
        /// for the model's own prompts; ownership moves to the model, and a span at the same position is replaced.</summary>
        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition)
        {
            ValidateVisionRows(embeddings, insertPosition);
            _ownVision.Set(embeddings, insertPosition);
            Console.WriteLine($"DiffusionGemma: installed {embeddings.Sizes[0]} image soft tokens at prompt position {insertPosition}.");
        }

        /// <summary>Drop the model's own image spans (the host calls this when a turn ends).</summary>
        public void ClearVisionEmbeddings()
        {
            if (_ownVision.Spans.Count > 0) _ownVision.Dispose();
        }

        /// <summary>Attach already-encoded image rows to one sequence (the concurrency-safe path: they live and go with it).</summary>
        public void SetSequenceVisionEmbeddings(DiffusionSeqState seq, Tensor embeddings, int insertPosition)
        {
            ArgumentNullException.ThrowIfNull(seq);
            ValidateVisionRows(embeddings, insertPosition);
            (seq.Vision ??= new VisionSet()).Set(embeddings, insertPosition);
        }

        /// <summary>Move the image spans prepared under <paramref name="mediaRequestId"/> onto <paramref name="seq"/>. False when
        /// there were none: the prompt then carries image placeholders nothing fills, which the caller reports.</summary>
        public bool QueueSequenceVisionEmbeddings(DiffusionSeqState seq, string mediaRequestId)
        {
            ArgumentNullException.ThrowIfNull(seq);
            if (MultimodalInjector is not ModelMultimodalInjector injector) return false;
            return injector.QueuePromptEmbeddings(0, mediaRequestId, (embeddings, position) => SetSequenceVisionEmbeddings(seq, embeddings, position));
        }

        /// <summary>The model's own prompts read <paramref name="seq"/>'s spans until the scope ends (the sampler's unified path
        /// for a sequence).</summary>
        internal IDisposable UseSequenceVision(DiffusionSeqState seq) => new VisionScope(this, seq?.Vision);

        private sealed class VisionScope : IDisposable
        {
            private readonly DiffusionGemmaModel _model;
            private readonly VisionSet _previous;

            public VisionScope(DiffusionGemmaModel model, VisionSet vision)
            {
                _model = model;
                _previous = model._visionOverride;
                model._visionOverride = vision ?? s_noVision;
            }

            public void Dispose() => _model._visionOverride = _previous;
        }

        private static readonly VisionSet s_noVision = new();

        private void ValidateVisionRows(Tensor embeddings, int insertPosition)
        {
            ArgumentNullException.ThrowIfNull(embeddings);
            if (insertPosition < 0)
            {
                embeddings.Dispose();
                throw new ArgumentOutOfRangeException(nameof(insertPosition));
            }
            if (embeddings.Sizes.Length != 2 || embeddings.Sizes[1] != Config.HiddenSize || embeddings.Sizes[0] <= 0)
            {
                string shape = string.Join("x", embeddings.Sizes.ToArray());
                embeddings.Dispose();
                throw new ArgumentException($"Vision embeddings must be [tokens > 0, {Config.HiddenSize}]; got [{shape}].", nameof(embeddings));
            }
        }

        private void DisposeVisionState()
        {
            _visionEncoder?.Dispose();
            _visionEncoder = null;
            _ownVision.Dispose();
        }
    }
}
