// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// Qwen3-VL's multimodal contract on the runtime: the Qwen-VL tower (with its DeepStack mergers),
    /// the injector's image and video pad expansion, and the (T, H, W) position tables the runtime
    /// rotates with the family's interleaved M-RoPE sections.
    /// </summary>
    public sealed partial class Qwen3VLModel : IVisionCapableModel, IMRoPEPositionSink, IMultimodalPromptExpander
    {
        public Qwen35VisionEncoder VisionEncoder { get; private set; }

        bool IVisionCapableModel.IsVisionEncoderLoaded => VisionEncoder != null;

        public void LoadVisionEncoder(string mmProjPath)
        {
            VisionEncoder?.Dispose();
            VisionEncoder = null;
            // Qwen3-VL's tower: tanh GELU in its blocks, exact (erf) GELU in the final and DeepStack mergers.
            var encoder = new Qwen35VisionEncoder(mmProjPath, _allocator, mergerGeluErf: true);
            if (encoder.ProjectionDim != Config.HiddenSize || encoder.DeepStackCount != _deepStackLayers)
            {
                int projection = encoder.ProjectionDim, taps = encoder.DeepStackCount;
                encoder.Dispose();
                throw new NotSupportedException(
                    $"{Config.Architecture}: the projector {mmProjPath} projects to {projection} with {taps} DeepStack taps; " +
                    $"this model takes {Config.HiddenSize} with {_deepStackLayers}. Use the projector published with the model.");
            }
            // The taps ride each media row: [main | tap 0 | ... ], split again in SetVisionEmbeddings.
            // TS_QWEN3VL_DEEPSTACK=0 leaves them out (a diagnostic A/B: the model then answers without them).
            encoder.ConcatenateDeepStack = _deepStackLayers > 0 && Environment.GetEnvironmentVariable("TS_QWEN3VL_DEEPSTACK") != "0";
            encoder.SetHostModel(this);
            VisionEncoder = encoder;
            encoder.WarmUp();
        }

        /// <summary>
        /// A media span's rows: [rows, hidden] from a tower without DeepStack, or [rows, hidden * (1 + taps)] with them. The
        /// main part becomes the runtime's media rows, the taps wait for the layers they join (Qwen3VLModel.Graph.cs).
        /// </summary>
        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition)
        {
            ArgumentNullException.ThrowIfNull(embeddings);
            int hidden = Config.HiddenSize;
            if (embeddings.DimensionCount != 2 || embeddings.Sizes[1] == hidden || _deepStackLayers == 0)
            {
                QueueMedia(embeddings, insertPosition);
                return;
            }
            long width = embeddings.Sizes[1];
            if (width != (long)hidden * (1 + _deepStackLayers))
            {
                embeddings.Dispose();
                throw new ArgumentException(
                    $"{Config.Architecture}: media rows of {width} values, expected {hidden} or {hidden * (1 + _deepStackLayers)} " +
                    $"({_deepStackLayers} DeepStack taps).", nameof(embeddings));
            }
            int rows = (int)embeddings.Sizes[0];
            Tensor main;
            var taps = new float[_deepStackLayers][];
            try
            {
                using (Tensor view = embeddings.Narrow(1, 0, hidden))
                    main = Ops.NewContiguous(view);
                for (int i = 0; i < _deepStackLayers; i++)
                {
                    using Tensor view = embeddings.Narrow(1, (long)hidden * (1 + i), hidden);
                    using Tensor tap = Ops.NewContiguous(view);
                    taps[i] = tap.GetElementsAsFloat(rows * hidden);
                }
            }
            finally
            {
                embeddings.Dispose();
            }
            QueueMedia(main, insertPosition);
            QueueDeepStack(insertPosition, rows, taps);
        }

        void IMRoPEPositionSink.SetMRoPEPositions(int[] flatThw) => StagePositionTable(flatThw);

        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(
            ModelMultimodalInjector injector, List<ChatMessage> history, List<int> inputTokens)
            => injector.ProcessQwenVLHistory(VisionEncoder, history, inputTokens);

        public override void Dispose()
        {
            VisionEncoder?.Dispose();
            VisionEncoder = null;
            base.Dispose();
        }
    }
}
