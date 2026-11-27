// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using TensorSharp.Cpu;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// Nemotron-H's image and audio input on the shared runtime. The towers are the family's: the RADIO / v2_vl vision tower
    /// (<see cref="NemotronVisionEncoder"/>, from the Omni / Super VL / Bolt mmproj) and the Parakeet audio tower
    /// (<see cref="NemotronAudioEncoder"/>, from a companion GGUF no public repository ships). The injector expands each
    /// <c>&lt;image&gt;</c> into <c>&lt;img&gt;</c> + rows + <c>&lt;/img&gt;</c> and each <c>&lt;so_embedding&gt;</c> into
    /// <c>&lt;so_start&gt;</c> + rows + <c>&lt;so_end&gt;</c> (<see cref="ModelMultimodalInjector.ProcessNemotronHistory"/>), and
    /// the rows enter the forward through <see cref="StandardDecoderModel.QueueMedia"/>, written over their placeholder rows;
    /// they attend causally, as text does.
    /// </summary>
    public sealed partial class NemotronRuntimeModel : IVisionCapableModel, IAudioCapableModel, IAudioEncoderLoader, IMultimodalPromptExpander
    {
        /// <summary>
        /// The one refusal every entry point gives an audio attachment on this family when no audio tower is loaded (OpenAI
        /// chat/Responses parsers, the Web UI, the CLI and the multimodal injector). The public GGUF distribution of Nemotron 3
        /// Nano Omni ships no audio encoder: its <c>mmproj</c> carries the RADIO vision tower only, and the language GGUF has
        /// only the <c>&lt;so_embedding&gt;</c> placeholder token. <see cref="NemotronAudioEncoder"/> runs the tower when a
        /// companion GGUF carrying NVIDIA's <c>sound_encoder.*</c> / <c>sound_projection.*</c> tensors is loaded
        /// (<see cref="LoadAudioEncoder"/>); then <see cref="IsAudioEncoderLoaded"/> is true and audio is served.
        /// </summary>
        public const string AudioInputUnsupportedMessage =
            "This Nemotron-H model does not support audio input: no audio tower is loaded. The public GGUF " +
            "distribution of Nemotron 3 Nano Omni ships no audio encoder - its mmproj carries only the RADIO vision " +
            "tower, not the Parakeet/FastConformer audio tower or its sound projector - so an audio clip cannot be " +
            "encoded. Load an audio companion GGUF that contains the sound_encoder/sound_projection tensors " +
            "(--mmproj, or TS_NEMOTRON_AUDIO_MMPROJ beside a vision mmproj), remove the audio attachments, or use " +
            "a model with an audio encoder (Gemma 4).";

        public NemotronVisionEncoder VisionEncoder { get; private set; }

        public NemotronAudioEncoder AudioEncoder { get; private set; }

        /// <summary>True when a supported Parakeet audio tower and sound projector were loaded from a companion GGUF.</summary>
        public bool IsAudioEncoderLoaded => AudioEncoder != null;

        bool IVisionCapableModel.IsVisionEncoderLoaded => VisionEncoder != null;

        public NemotronImageProcessor ImageProcessor =>
            VisionEncoder?.ImageProcessor
            ?? throw new InvalidOperationException("Vision encoder must be loaded before accessing the image processor.");

        /// <summary>The towers run on the language model's allocator, except on the direct cuda backend, whose vision stack
        /// runs on the CPU: the embeddings are copied into the forward.</summary>
        private IAllocator TowerAllocator => _backend == BackendType.Cuda ? new CpuAllocator(BlasEnum.DotNet) : _allocator;

        public void LoadVisionEncoder(string mmProjPath)
        {
            using (var metadata = new GgufFile(mmProjPath))
                if (!metadata.Tensors.Keys.Any(name => name.StartsWith("v.", StringComparison.Ordinal)))
                    return;
            // The encoder validates its own tensors at load (InvalidDataException naming them); the projector's output width
            // must also be this language model's hidden size, or another Nemotron's mmproj (Nano Omni projects to 2688, Super VL
            // and Bolt to 4096) would load and fail at the first image.
            // The tower runs as graph programs on this model's executor (direct CUDA included); its embeddings come back on the
            // tower allocator, as the managed path's did.
            var encoder = NemotronVisionEncoder.Load(mmProjPath, _allocator, _backend, TowerAllocator);
            if (encoder.ProjectionDim != Config.HiddenSize)
            {
                encoder.Dispose();
                throw new InvalidDataException(
                    $"Nemotron mmproj '{Path.GetFileName(mmProjPath)}' projects images to {encoder.ProjectionDim}-wide " +
                    $"embeddings (mm.model.mlp.3.weight), but this language model's hidden size is {Config.HiddenSize}; " +
                    "the projector belongs to another Nemotron model.");
            }
            encoder.SetHostModel(this);
            VisionEncoder?.Dispose();
            VisionEncoder = encoder;
        }

        public void LoadAudioEncoder(string mmProjPath)
        {
            string path = Environment.GetEnvironmentVariable("TS_NEMOTRON_AUDIO_MMPROJ");
            if (string.IsNullOrWhiteSpace(path)) path = mmProjPath;
            using (var metadata = new GgufFile(path))
                if (!metadata.Tensors.ContainsKey("sound_projection.linear2.weight"))
                    return;
            // An F32-compute companion encodes as graph programs on this model's executor; a BF16-emulating one on the managed
            // path over the tower allocator.
            var encoder = NemotronAudioEncoder.Load(path, _allocator, _backend, TowerAllocator);
            if (encoder.ProjectionDim != Config.HiddenSize || encoder.MelBins != NemotronAudioPreprocessor.MelBins)
            {
                encoder.Dispose();
                throw new InvalidDataException("Nemotron audio companion dimensions do not match the language model.");
            }
            encoder.SetHostModel(this);
            AudioEncoder?.Dispose();
            AudioEncoder = encoder;
        }

        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition) => QueueMedia(embeddings, insertPosition);

        public void SetAudioEmbeddings(Tensor embeddings, int insertPosition) => QueueMedia(embeddings, insertPosition);

        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(
            ModelMultimodalInjector injector, List<ChatMessage> history, List<int> inputTokens)
            => injector.ProcessNemotronHistory(this, history, inputTokens);

        private void DisposeMultimodal()
        {
            VisionEncoder?.Dispose();
            VisionEncoder = null;
            AudioEncoder?.Dispose();
            AudioEncoder = null;
        }
    }
}
