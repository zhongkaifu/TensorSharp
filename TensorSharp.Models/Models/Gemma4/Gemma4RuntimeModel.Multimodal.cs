// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;

using TensorSharp.Cpu;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// Gemma 4's image, video and audio input on the shared runtime. The towers are the family's (<see cref="Gemma4VisionEncoder"/>
    /// SigLIP / unified, <see cref="Gemma4AudioEncoder"/>), loaded from the mmproj (or, for the vision tower, a HuggingFace
    /// shard); the injector expands each <c>&lt;|image&gt;</c> / <c>&lt;|audio&gt;</c> into its soft-token span
    /// (<see cref="ModelMultimodalInjector.ProcessGemma4History(Gemma4VisionEncoder, Gemma4AudioEncoder, List{ChatMessage}, List{int})"/>),
    /// and the rows enter the forward through <see cref="StandardDecoderModel.QueueMedia"/>, written over their placeholder rows
    /// before the stream input's sqrt(n_embd) scale skips them; the rows of one forward attend each other in both directions
    /// (<see cref="MediaAttendsBidirectionally"/>).
    /// </summary>
    internal sealed partial class Gemma4RuntimeModel : IVisionCapableModel, IAudioCapableModel, IAudioEncoderLoader, IMultimodalPromptExpander,
        IGemma4MediaTowers
    {
        public Gemma4VisionEncoder VisionEncoder { get; private set; }

        public Gemma4AudioEncoder AudioEncoder { get; private set; }

        bool IVisionCapableModel.IsVisionEncoderLoaded => VisionEncoder != null;

        public void LoadVisionEncoder(string mmProjPath)
        {
            // The direct CUDA backend's vision stack diverged numerically (legacy's note): its projector runs on the CPU and the
            // embeddings are copied into the language model's forward.
            IAllocator visionAllocator = _backend == BackendType.Cuda ? new CpuAllocator(BlasEnum.DotNet) : _allocator;
            var encoder = new Gemma4VisionEncoder(mmProjPath, visionAllocator);
            encoder.SetHostModel(this);
            VisionEncoder?.Dispose();
            VisionEncoder = encoder;
        }

        public void LoadAudioEncoder(string mmProjPath)
        {
            var encoder = new Gemma4AudioEncoder(mmProjPath, _allocator);
            encoder.SetHostModel(this);
            AudioEncoder?.Dispose();
            AudioEncoder = encoder;
        }

        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition) => QueueMedia(embeddings, insertPosition);

        public void SetAudioEmbeddings(Tensor embeddings, int insertPosition) => QueueMedia(embeddings, insertPosition);

        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(
            ModelMultimodalInjector injector, List<ChatMessage> history, List<int> inputTokens)
            => injector.ProcessGemma4History(VisionEncoder, AudioEncoder, history, inputTokens);

        public override void Dispose()
        {
            VisionEncoder?.Dispose();
            VisionEncoder = null;
            AudioEncoder?.Dispose();
            AudioEncoder = null;
            FamilyTokenDrafter?.Dispose();
            FamilyTokenDrafter = null;
            base.Dispose();
        }
    }

    /// <summary>A Gemma 4 language model's media towers (the injector and the knowledge extractor read them).</summary>
    public interface IGemma4MediaTowers
    {
        Gemma4VisionEncoder VisionEncoder { get; }

        Gemma4AudioEncoder AudioEncoder { get; }
    }
}
