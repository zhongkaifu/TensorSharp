// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;

using TensorSharp.Runtime;

namespace TensorSharp.Models.Architecture
{
    /// <summary>
    /// A model that can consume image embeddings produced by an mmproj tower.
    ///
    /// This is the seam that keeps <see cref="ModelMultimodalInjector"/> free of model
    /// types. The injector owns everything that is the same for every architecture -
    /// per-request embedding buckets, span bookkeeping, prefix clamping, trimming and
    /// slicing - and calls through this interface for the two things that are not:
    /// loading the tower, and handing a span of embeddings to the next forward pass.
    /// A new vision model implements this and edits none of that machinery.
    ///
    /// Public because it is also how a host (CLI, server) asks "can this model see?"
    /// without matching on architecture names.
    /// </summary>
    public interface IVisionCapableModel
    {
        /// <summary>Load the vision tower from an mmproj GGUF.</summary>
        void LoadVisionEncoder(string mmProjPath);

        /// <summary>
        /// True once <see cref="LoadVisionEncoder"/> has produced a usable tower.
        ///
        /// <c>ModelBase.HasVisionEncoder()</c> answers this by scanning the MAIN model's
        /// weight tables for <c>v.</c>-prefixed tensors, which only finds a tower that
        /// was baked into the model GGUF. Every family that loads its tower from a
        /// separate mmproj answered "no" even with the encoder loaded and running - the
        /// interactive REPL printed "Vision enc: (none)" for Gemma 4 throughout.
        /// </summary>
        bool IsVisionEncoderLoaded { get; }

        /// <summary>
        /// Queue one span of image embeddings for the next forward pass, spliced in
        /// <paramref name="insertPosition"/> tokens into the batch about to be
        /// submitted. Called once per span per forward; the model consumes and clears
        /// the queue itself.
        /// </summary>
        void SetVisionEmbeddings(Tensor embeddings, int insertPosition);
    }

    /// <summary>A model that can consume audio embeddings.</summary>
    public interface IAudioCapableModel
    {
        /// <summary>Queue one span of audio embeddings for the next forward pass.</summary>
        void SetAudioEmbeddings(Tensor embeddings, int insertPosition);
    }

    /// <summary>
    /// A model with an optional audio tower loaded from a projector file or an
    /// architecture-specific companion. Separate from <see cref="IAudioCapableModel"/>
    /// so audio sinks can also receive embeddings from an externally managed encoder.
    /// </summary>
    public interface IAudioEncoderLoader
    {
        void LoadAudioEncoder(string mmProjPath);
    }

    /// <summary>
    /// A model whose rotary embedding takes interleaved per-axis (t, h, w) positions
    /// for image regions. The injector pushes the slice matching each forward batch;
    /// text-only requests never call this and keep scalar positions.
    /// </summary>
    public interface IMRoPEPositionSink
    {
        /// <summary>Flat (T,H,W) position table for the tokens about to be forwarded;
        /// length is 3 x tokenCount.</summary>
        void SetMRoPEPositions(int[] flatThw);
    }

    /// <summary>
    /// How an architecture expands its own media placeholders in a rendered prompt.
    ///
    /// Prompt formats differ far more between families than queueing does (Qwen-VL's
    /// vision-pad runs plus M-RoPE, Mistral's row/break tokens, GLM's
    /// <c>&lt;|image|&gt;</c>), so each family names its own
    /// expansion here instead of being named in a dispatch chain inside the injector.
    /// The expansion itself runs on the injector's shared machinery - media caching,
    /// span registration, token splicing - which is why the injector is the argument.
    ///
    /// Internal: this is an extension point inside TensorSharp.Models, not part of the
    /// public model surface. Hosts ask about capability through
    /// <see cref="IVisionCapableModel"/>.
    /// </summary>
    internal interface IMultimodalPromptExpander
    {
        List<int> ExpandMultimodalPrompt(ModelMultimodalInjector injector,
            List<ChatMessage> history, List<int> inputTokens);
    }
}
