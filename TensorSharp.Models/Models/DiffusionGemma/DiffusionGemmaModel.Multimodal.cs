// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// DiffusionGemma's multimodal contract: the Gemma-4 SigLIP vision tower (gemma4v), loaded from
// either an mmproj GGUF or a raw HuggingFace .safetensors shard.
using System;
using System.Collections.Generic;

using TensorSharp.Cpu;
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>
    /// Image support for <see cref="DiffusionGemmaModel"/>.
    ///
    /// The tower is the SAME one Gemma 4 uses (<see cref="Gemma4VisionEncoder"/>, projector type
    /// <c>gemma4v</c>, projection dim 2816 = this model's <c>embedding_length</c>), so nothing about
    /// the encoder is re-implemented here. What IS different is how the embeddings reach the graph:
    ///
    ///  - Gemma 4 is autoregressive. Its injector queues a span per prefill chunk, the forward splices
    ///    it once, disposes it, and the KV cache carries the result forward. Consume-on-use is right.
    ///  - DiffusionGemma is a block diffuser. One user turn runs <c>PrefillPrompt</c> once per BLOCK
    ///    (<see cref="DiffusionGemmaSampler"/>.DenoiseBlock) and, on backends without prompt-KV caching,
    ///    re-runs the whole <c>[prompt|canvas]</c> forward once per DENOISING STEP - tens of times per
    ///    block. A consume-on-use queue would splice the image into step 1 and feed raw
    ///    <c>&lt;|image|&gt;</c> placeholder embeddings to every step after it.
    ///
    /// So the spans here are RETAINED, not consumed: <see cref="SetVisionEmbeddings"/> installs them and
    /// every prompt embedding re-applies them until <see cref="ClearVisionEmbeddings"/> or dispose.
    /// Re-queueing the same insert position replaces (and disposes) the previous tensor, so a host that
    /// calls <see cref="SetVisionEmbeddings"/> once per forward - the Gemma 4 rhythm - is also correct;
    /// it just does redundant work.
    ///
    /// Because the retained set is model-global state, it belongs to ONE request at a time. The host
    /// must call <see cref="ClearVisionEmbeddings"/> when a turn ends, and must not prefill a second,
    /// image-free sequence (the batched <c>PrefillSeq</c> path) while a set is installed - an out of
    /// range span throws rather than silently landing on the wrong tokens.
    /// </summary>
    public sealed partial class DiffusionGemmaModel : IVisionCapableModel, IMultimodalPromptExpander
    {
        /// <summary>
        /// Expands the rendered <c>&lt;|image&gt;</c> marker into [BOI] + N soft rows + [EOI] and
        /// queues the encoded spans, reusing Gemma 4's expander unchanged — the token layout, the
        /// image preprocessing and the per-image soft-token count are identical, and the only
        /// difference (no audio tower) is handled by passing no audio encoder.
        ///
        /// Without this, <see cref="ModelMultimodalInjector.ProcessPromptTokens"/> throws rather
        /// than leaving unexpanded placeholders in the prompt, because that failure mode is a
        /// fluent answer about an image the model never received.
        /// </summary>
        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(
            ModelMultimodalInjector injector, List<ChatMessage> history, List<int> inputTokens)
            => injector.ProcessGemma4VisionHistory(_visionEncoder, history, inputTokens);

        private Gemma4VisionEncoder _visionEncoder;

        /// <summary>The model-level retained set, used only by the SINGLE-REQUEST paths
        /// (<c>PrefillPrompt</c> / <c>ForwardCanvas</c> / the structured reader), which have no
        /// <see cref="DiffusionSeqState"/> to hang spans off. The batched scheduler does NOT use this:
        /// its spans live on each sequence, so two concurrent image requests cannot collide.</summary>
        private readonly List<(Tensor Embeddings, int Position)> _ownedVisionEmbeddingsList = new();


        /// <summary>The span set the prompt embedding and the attention mask actually read. Points at
        /// <see cref="_ownedVisionEmbeddingsList"/> by default and is swapped, for the duration of one
        /// prefill, to the spans owned by the sequence being prefilled (see <c>PrefillSeq</c>).
        ///
        /// <para>Scoping rather than threading a parameter is safe because every prefill runs under
        /// <c>GpuComputeLock</c>, one at a time; it keeps the swap off the attention inner loops.</para></summary>
        private List<(Tensor Embeddings, int Position)> _pendingVisionEmbeddingsList;

        /// <summary>Points the active set at the model-level list. Called from the constructor path
        /// via <see cref="HasPendingVisionEmbeddings"/>'s first use is not enough, so it is explicit.</summary>
        private List<(Tensor Embeddings, int Position)> ActiveVisionList =>
            _pendingVisionEmbeddingsList ??= _ownedVisionEmbeddingsList;

        /// <summary>Scopes the active span set to one sequence's spans for the duration of a single
        /// prefill, restoring the previous set on dispose. Concurrency safety rests on every prefill
        /// running under <c>GpuComputeLock</c>.</summary>
        internal readonly struct VisionScope : IDisposable
        {
            private readonly DiffusionGemmaModel _model;
            private readonly List<(Tensor Embeddings, int Position)> _prevList;
            private readonly (int Start, int Length)[] _prevSpans;

            internal VisionScope(DiffusionGemmaModel model, DiffusionSeqState seq)
            {
                _model = model;
                _prevList = model.ActiveVisionList;
                _prevSpans = model._visionSpans;

                model._pendingVisionEmbeddingsList = seq.VisionEmbeddings ?? EmptyVisionList;
                model._visionSpans = seq.VisionSpans ?? Array.Empty<(int Start, int Length)>();
                // The mask cache is keyed partly on the span version, so entering a different
                // sequence's spans must invalidate it exactly as installing new spans would.
                model._visionSpanVersion++;
            }

            public void Dispose()
            {
                if (_model == null) return;
                _model._pendingVisionEmbeddingsList = _prevList;
                _model._visionSpans = _prevSpans;
                _model._visionSpanVersion++;
            }
        }

        private static readonly List<(Tensor Embeddings, int Position)> EmptyVisionList = new();

        /// <summary>Run one prefill against <paramref name="seq"/>'s own image spans. A no-op scope when
        /// neither the sequence nor the model has any: swapping one empty set for another changes nothing
        /// the forward reads, but the version bump would discard the span-keyed caches (the attention mask
        /// the unified forward otherwise rebuilds every denoising step, and the Jev prompt K/V).</summary>
        internal VisionScope UseSequenceVision(DiffusionSeqState seq)
        {
            bool seqHasSpans = seq.HasVision || (seq.VisionSpans != null && seq.VisionSpans.Length != 0);
            if (!seqHasSpans && ActiveVisionList.Count == 0 && _visionSpans.Length == 0)
                return default;
            return new(this, seq);
        }

        /// <summary>
        /// Attach already-encoded image spans to ONE sequence. This is the concurrency-safe entry
        /// point: the tensors live on the sequence, are applied only while that sequence is being
        /// prefilled, and are freed with it.
        ///
        /// <para>Mirrors how vLLM and SGLang carry multimodal features (vLLM hangs
        /// <c>MultiModalFeatureSpec</c> off the Request; SGLang hangs <c>MultimodalInputs</c> off
        /// Req) -- neither ever parks pending embeddings on the model.</para>
        /// </summary>
        public void SetSequenceVisionEmbeddings(DiffusionSeqState seq, Tensor embeddings, int insertPosition)
        {
            ArgumentNullException.ThrowIfNull(seq);
            ArgumentNullException.ThrowIfNull(embeddings);
            ArgumentOutOfRangeException.ThrowIfNegative(insertPosition);
            ValidateVisionEmbeddingShape(embeddings);

            seq.VisionEmbeddings ??= new List<(Tensor Embeddings, int Position)>();
            int rows = (int)embeddings.Sizes[0];
            for (int i = 0; i < seq.VisionEmbeddings.Count; i++)
            {
                var (existing, position) = seq.VisionEmbeddings[i];
                if (position == insertPosition)
                {
                    if (!ReferenceEquals(existing, embeddings))
                    {
                        existing?.Dispose();
                        seq.VisionEmbeddings[i] = (embeddings, insertPosition);
                    }
                    seq.VisionSpans = BuildVisionSpans(seq.VisionEmbeddings);
                    seq.VisionSpanVersion++;
                    return;
                }
                int existingRows = existing == null ? 0 : (int)existing.Sizes[0];
                if (insertPosition < position + existingRows && position < insertPosition + rows)
                {
                    embeddings.Dispose();
                    throw new ArgumentException(
                        $"Image span [{insertPosition},{insertPosition + rows}) overlaps this sequence's " +
                        $"installed span [{position},{position + existingRows}).", nameof(insertPosition));
                }
            }

            seq.VisionEmbeddings.Add((embeddings, insertPosition));
            seq.VisionSpans = BuildVisionSpans(seq.VisionEmbeddings);
            seq.VisionSpanVersion++;
        }

        /// <summary>
        /// Move the image spans prepared under <paramref name="mediaRequestId"/> onto
        /// <paramref name="seq"/>. Returns false when the bucket was empty, which means the prompt
        /// carries expanded image placeholders that nothing will fill -- the caller should say so
        /// rather than let the model answer from the filler rows.
        ///
        /// <para>This is the concurrency-safe path the batched scheduler uses. It exists on the
        /// model rather than on <c>IMultimodalInjector</c> because that interface lives in
        /// TensorSharp.Runtime, which deliberately does not know the Tensor type.</para>
        /// </summary>
        public bool QueueSequenceVisionEmbeddings(DiffusionSeqState seq, string mediaRequestId)
        {
            ArgumentNullException.ThrowIfNull(seq);
            if (MultimodalInjector is not ModelMultimodalInjector injector)
                return false;
            return injector.QueuePromptEmbeddings(0, mediaRequestId,
                (embeddings, position) => SetSequenceVisionEmbeddings(seq, embeddings, position));
        }

        private void ValidateVisionEmbeddingShape(Tensor embeddings)
        {
            if (embeddings.Sizes.Length != 2 || embeddings.Sizes[1] != Config.HiddenSize)
            {
                string shape = string.Join("x", embeddings.Sizes.ToArray());
                embeddings.Dispose();
                throw new ArgumentException(
                    $"Vision embeddings must be [tokens, {Config.HiddenSize}]; got [{shape}].", nameof(embeddings));
            }
            if (embeddings.Sizes[0] <= 0)
            {
                embeddings.Dispose();
                throw new ArgumentException("Vision embeddings must contain at least one token.", nameof(embeddings));
            }
        }

        /// <summary>Soft-token spans, ascending, from a span list. Empty when bidirectional image
        /// attention is disabled, which leaves the mask exactly as a text-only prompt's.</summary>
        private static (int Start, int Length)[] BuildVisionSpans(
            List<(Tensor Embeddings, int Position)> list)
        {
            if (list == null || list.Count == 0 || !VisionBidirectionalEnabled)
                return Array.Empty<(int Start, int Length)>();

            var spans = new (int Start, int Length)[list.Count];
            for (int i = 0; i < list.Count; i++)
                spans[i] = (list[i].Position, (int)list[i].Embeddings.Sizes[0]);
            Array.Sort(spans, static (a, b) => a.Start.CompareTo(b.Start));
            return spans;
        }

        /// <summary>Soft-token spans in prompt coordinates, ascending by <c>Start</c>, used by the
        /// attention mask. Empty when bidirectional image attention is off (then the model runs plain
        /// causal over image tokens and nothing about masking changes).</summary>
        private (int Start, int Length)[] _visionSpans = Array.Empty<(int Start, int Length)>();

        /// <summary>Bumped on every span change. <c>GetAttentionMask</c>'s cache is keyed on
        /// (N, P) alone, which would happily hand a second image the FIRST image's mask whenever the two
        /// prompts happen to have the same length; this is the third key component.</summary>
        private int _visionSpanVersion;
        private int _maskSpanVersion = -1;

        private bool _loggedVisionMaskPolicy;

        /// <summary>
        /// Bidirectional attention inside image soft-token spans, on SLIDING (local) layers only.
        /// Set <c>DIFFUSION_IMAGE_BIDIRECTIONAL=0</c> for plain causal over image tokens, which is what
        /// HuggingFace's own DiffusionGemma effectively runs (it builds the vision mask and then
        /// discards it) - hence the escape hatch rather than a hardcoded choice.
        /// </summary>
        private static readonly bool VisionBidirectionalEnabled =
            Environment.GetEnvironmentVariable("DIFFUSION_IMAGE_BIDIRECTIONAL") != "0";

        /// <summary>The loaded gemma4v tower, or null.</summary>
        public Gemma4VisionEncoder VisionEncoder => _visionEncoder;

        bool IVisionCapableModel.IsVisionEncoderLoaded => _visionEncoder != null;

        /// <summary>True while a retained image span set is installed.</summary>
        public bool HasPendingVisionEmbeddings => ActiveVisionList.Count > 0;

        /// <param name="projectorPath">
        /// An mmproj GGUF, or a HuggingFace <c>.safetensors</c> shard holding the
        /// <c>model.encoder.vision_tower.*</c> tower. <see cref="Gemma4VisionEncoder"/> dispatches on the
        /// extension; the published DiffusionGemma GGUF conversions carry no vision tensors at all, so
        /// the safetensors form is the one that works out of the box.
        /// </param>
        public void LoadVisionEncoder(string projectorPath)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(projectorPath);

            // Mirrors Gemma4RuntimeModel.LoadVisionEncoder: the direct (non-GGML) CUDA backend diverges
            // numerically inside the gemma4v stack, so keep that one tower on the CPU allocator and let
            // the splice copy the finished embeddings across.
            IAllocator visionAllocator = _backend == BackendType.Cuda
                ? new CpuAllocator(BlasEnum.DotNet)
                : _allocator;

            var encoder = new Gemma4VisionEncoder(projectorPath, visionAllocator);
            if (encoder.ProjectionDim != Config.HiddenSize)
            {
                encoder.Dispose();
                throw new InvalidOperationException(
                    $"Vision tower '{projectorPath}' projects to {encoder.ProjectionDim} dimensions but this " +
                    $"DiffusionGemma has embedding_length {Config.HiddenSize}. The tower does not belong to this model.");
            }

            _visionEncoder?.Dispose();
            _visionEncoder = encoder;
            // Lets the per-block encode loop yield ModelBase.GpuComputeLock so the engine stays
            // responsive during a long image encode.
            _visionEncoder.SetHostModel(this);
        }

        /// <summary>
        /// Install one image's projected embeddings at <paramref name="insertPosition"/>, the index of
        /// its FIRST SOFT token inside the prompt token array. The BOI / EOI tokens around the run stay
        /// ordinary text tokens and are NOT part of the span.
        ///
        /// The rows are used as given: they are NOT multiplied by <c>sqrt(hidden)</c>. They REPLACE
        /// prompt rows that were already scaled, and the vision side has already applied its own
        /// <c>sqrt(1152)</c> pooling scale inside <c>Gemma4VisionEncoder.PoolAndProject</c>.
        ///
        /// Ownership transfers to the model. Re-installing at an existing position disposes the tensor
        /// that was there.
        /// </summary>
        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition)
        {
            ArgumentNullException.ThrowIfNull(embeddings);
            ArgumentOutOfRangeException.ThrowIfNegative(insertPosition);
            if (embeddings.Sizes.Length != 2 || embeddings.Sizes[1] != Config.HiddenSize)
            {
                string shape = string.Join("x", embeddings.Sizes.ToArray());
                embeddings.Dispose();
                throw new ArgumentException(
                    $"Vision embeddings must be [tokens, {Config.HiddenSize}]; got [{shape}].", nameof(embeddings));
            }

            int rows = (int)embeddings.Sizes[0];
            if (rows <= 0)
            {
                embeddings.Dispose();
                throw new ArgumentException("Vision embeddings must contain at least one token.", nameof(embeddings));
            }

            for (int i = 0; i < _ownedVisionEmbeddingsList.Count; i++)
            {
                var (existing, position) = _ownedVisionEmbeddingsList[i];
                if (position == insertPosition)
                {
                    // Same span re-queued (a host that queues once per forward). Replace in place.
                    if (!ReferenceEquals(existing, embeddings))
                    {
                        existing?.Dispose();
                        _ownedVisionEmbeddingsList[i] = (embeddings, insertPosition);
                    }
                    RebuildVisionSpans();
                    return;
                }

                int existingRows = existing == null ? 0 : (int)existing.Sizes[0];
                if (insertPosition < position + existingRows && position < insertPosition + rows)
                {
                    embeddings.Dispose();
                    throw new ArgumentException(
                        $"Image span [{insertPosition},{insertPosition + rows}) overlaps the installed span " +
                        $"[{position},{position + existingRows}). Call ClearVisionEmbeddings between requests.",
                        nameof(insertPosition));
                }
            }

            _ownedVisionEmbeddingsList.Add((embeddings, insertPosition));
            RebuildVisionSpans();
            Console.WriteLine($"DiffusionGemma: installed {rows} image soft tokens at prompt position {insertPosition} " +
                $"(span [{insertPosition},{insertPosition + rows})).");
        }

        /// <summary>Drop every retained image span. The host calls this when a turn ends; a stale span
        /// would otherwise be spliced into the next request's prompt.</summary>
        public void ClearVisionEmbeddings()
        {
            if (_ownedVisionEmbeddingsList.Count == 0) return;
            foreach (var (embeddings, _) in _ownedVisionEmbeddingsList)
                embeddings?.Dispose();
            _ownedVisionEmbeddingsList.Clear();
            RebuildVisionSpans();
        }

        /// <summary>
        /// Splice every retained image span into the prompt region of <paramref name="hidden"/>, which
        /// must already hold <c>embed * sqrt(n_embd)</c>. Called from both prompt paths:
        /// <c>PrefillPromptInto</c> (prompt-KV caching) and <c>ForwardCanvasHidden</c> (the unified
        /// <c>[prompt|canvas]</c> forward), always before any layer runs.
        /// </summary>
        /// <param name="hidden">[rows, hidden] embeddings; rows >= <paramref name="promptLen"/>.</param>
        /// <param name="promptLen">Prompt length P. Image rows live in [0,P); the canvas occupies
        /// [P, P+C) and is written afterwards by <c>EmbedCanvasRegion</c>, so the two never overlap.</param>
        private void ApplyPendingVisionEmbeddings(Tensor hidden, int promptLen)
        {
            var active = ActiveVisionList;
            if (active.Count == 0) return;

            foreach (var (embeddings, position) in active)
            {
                int rows = (int)embeddings.Sizes[0];
                if (position + rows > promptLen)
                {
                    throw new InvalidOperationException(
                        $"Image span [{position},{position + rows}) does not fit a {promptLen}-token prompt. " +
                        "The retained span belongs to a different request; call ClearVisionEmbeddings when a turn ends.");
                }

                using Tensor target = hidden.Narrow(0, position, rows);
                if (ReferenceEquals(target.Allocator, embeddings.Allocator))
                {
                    Ops.Copy(target, embeddings);
                    continue;
                }

                // Tower on a different allocator (the direct-CUDA carve-out in LoadVisionEncoder):
                // round-trip through the host once.
                float[] host = embeddings.GetElementsAsFloat((int)embeddings.ElementCount());
                using var staged = new Tensor(hidden.Allocator, embeddings.ElementType, embeddings.Sizes);
                staged.SetElementsAsFloat(host);
                Ops.Copy(target, staged);
            }
        }

        /// <summary>Recompute <see cref="_visionSpans"/> from the retained set and invalidate every
        /// span-dependent cache.</summary>
        private void RebuildVisionSpans()
        {
            _visionSpanVersion++;

            _visionSpans = BuildVisionSpans(_ownedVisionEmbeddingsList);

            if (_loggedVisionMaskPolicy) return;
            _loggedVisionMaskPolicy = true;
            Console.WriteLine("DiffusionGemma image attention: bidirectional inside each soft-token span on " +
                "SLIDING (local) layers; GLOBAL layers (blk 5/11/17/23/29) stay plain causal, per the vLLM " +
                "gemma4_mm reference. Spans longer than the sliding window fall back to causal. " +
                "DIFFUSION_IMAGE_BIDIRECTIONAL=0 disables this entirely.");
        }

        /// <summary>
        /// Widen a PROMPT query's causal key window so it also covers the rest of the image soft-token
        /// span the query sits in - i.e. bidirectional attention inside the image block.
        ///
        /// The result stays ONE contiguous interval, which is the whole reason this is a khi extension
        /// and not a second interval threaded through every masking path. A span of length L is only
        /// honoured when <c>L &lt;= swa</c>, and a query inside it satisfies
        /// <c>qi - (L-1) &lt;= Start</c>, so <c>Start &gt;= max(0, qi - swa + 1) == klo</c>: the span can
        /// never begin before the sliding window does, and the union of [klo, qi+1) and
        /// [Start, Start+L) is exactly [klo, max(qi+1, Start+L)). The same bound makes every key in the
        /// span reachable within the window (max forward distance L-1 &lt;= swa-1), which is why spans
        /// longer than the window are dropped rather than clipped - a clipped image block would attend
        /// to an arbitrary sub-rectangle of itself.
        /// </summary>
        private static int ExtendKhiForVisionSpan(int qi, int khi, int swa, (int Start, int Length)[] spans)
        {
            for (int i = 0; i < spans.Length; i++)
            {
                (int start, int length) = spans[i];
                if (qi < start) break;                   // sorted ascending: no later span can contain qi
                if (qi >= start + length) continue;
                if (length > swa) return khi;            // longer than the window -> causal, see remarks
                int end = start + length;
                return end > khi ? end : khi;
            }
            return khi;
        }

        private void DisposeVisionState()
        {
            _visionEncoder?.Dispose();
            _visionEncoder = null;
            foreach (var (embeddings, _) in _ownedVisionEmbeddingsList)
                embeddings?.Dispose();
            _ownedVisionEmbeddingsList.Clear();
            _visionSpans = Array.Empty<(int Start, int Length)>();
        }
    }
}
