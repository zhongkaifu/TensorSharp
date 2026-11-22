// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;

using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// Muse-Glimmer's image input on the shared runtime. The vision tower is <see cref="MuseGlimmerVisionTower"/> (graph
    /// programs on this model's executor), loaded from the mmproj. The injector lays the prompt out: each <c>&lt;|patch|&gt;</c>
    /// becomes <c>&lt;|image_start|&gt;</c>, the image's rows and <c>&lt;|image_end|&gt;</c>
    /// (<see cref="ModelMultimodalInjector.ProcessMuseGlimmerHistory(IMuseGlimmerVisionTower, List{ChatMessage}, List{int})"/>,
    /// llama.cpp's mtmd layout, shared with Muse Spark). The rows enter the forward through
    /// <see cref="StandardDecoderModel.QueueMedia"/>: written over their filler rows in the embedded stream before the
    /// weightless input norm (<see cref="EmitStreamInput"/>, <see cref="ApplyStreamInput"/>), which so runs over the merged
    /// stream, as legacy and llama.cpp's <c>build_inp_embd</c> run it. Decoding and resizing an image run off the GPU lock, and
    /// a large image's tower parts yield the lock between them, so concurrent requests keep stepping.
    /// </summary>
    internal sealed partial class MuseGlimmerRuntimeModel : IVisionCapableModel, IMultimodalPromptExpander
    {
        /// <summary>The vision tower, or null (no mmproj loaded).</summary>
        public MuseGlimmerVisionTower VisionTower { get; private set; }

        bool IVisionCapableModel.IsVisionEncoderLoaded => VisionTower != null;

        /// <summary>Load the vision tower of a Muse-Glimmer mmproj (projector type <c>muse-glimmer</c> at this model's width:
        /// Muse Spark's projector has the same tower at another width and is refused).</summary>
        public void LoadVisionEncoder(string mmProjPath)
        {
            ArgumentException.ThrowIfNullOrEmpty(mmProjPath);
            MuseGlimmerVisionTower tower;
            using (var gguf = new GgufFile(mmProjPath))
            {
                int width = (int)gguf.GetUint32("clip.vision.projection_dim", 0);
                if (width != Config.HiddenSize)
                    throw new NotSupportedException(
                        $"'{Path.GetFileName(mmProjPath)}' projects images to rows of {width}; {Config.Architecture} embeds rows of {Config.HiddenSize}. " +
                        "Load the mmproj published with this model.");
                tower = new MuseGlimmerVisionTower(gguf, _allocator, _backend);
            }
            VisionTower?.Dispose();
            VisionTower = tower;
            Console.WriteLine($"{Config.Architecture}: vision tower from '{Path.GetFileName(mmProjPath)}': {tower.BlockCount} layers of " +
                $"{tower.EmbeddingLength} on {tower.Graph.Executor.GetType().Name}, up to {MuseGlimmerVisionTower.MaxTokens} rows an image");
        }

        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition) => QueueMedia(embeddings, insertPosition);

        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(
            ModelMultimodalInjector injector, List<ChatMessage> history, List<int> inputTokens)
            => injector.ProcessMuseGlimmerHistory(VisionTower, history, inputTokens);

        private void DisposeVisionTower()
        {
            VisionTower?.Dispose();
            VisionTower = null;
        }
    }
}
