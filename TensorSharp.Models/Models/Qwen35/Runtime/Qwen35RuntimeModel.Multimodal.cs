// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;

using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// Qwen 3.5's multimodal contract on the runtime: the Qwen-VL tower, the injector's image-pad
    /// expansion, and its (T, H, W) position tables, which the runtime rotates with the family's
    /// interleaved M-RoPE sections (images and their media rows are the runtime's generic path).
    /// </summary>
    public partial class Qwen35RuntimeModel : IVisionCapableModel, IMRoPEPositionSink, IMultimodalPromptExpander
    {
        private int[] _ropeSections;

        protected override int[] RotarySections => _ropeSections;

        public Qwen35VisionEncoder VisionEncoder { get; private set; }

        bool IVisionCapableModel.IsVisionEncoderLoaded => VisionEncoder != null;

        public void LoadVisionEncoder(string mmProjPath)
        {
            VisionEncoder?.Dispose();
            VisionEncoder = new Qwen35VisionEncoder(mmProjPath, _allocator);
            VisionEncoder.SetHostModel(this);
        }

        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition) => QueueMedia(embeddings, insertPosition);

        void IMRoPEPositionSink.SetMRoPEPositions(int[] flatThw) => StagePositionTable(flatThw);

        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(
            ModelMultimodalInjector injector, List<ChatMessage> history, List<int> inputTokens)
            => injector.ProcessQwenVLHistory(VisionEncoder, history, inputTokens);

        public override void Dispose()
        {
            VisionEncoder?.Dispose();
            VisionEncoder = null;
            base.Dispose();
            // Its mapped weights were the draft layer's: released with the model's weights above.
            _separateDraftHead?.Dispose();
            _separateDraftHead = null;
        }
    }
}
