// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

namespace TensorSharp.Models
{
    /// <summary>
    /// The autoregressive Gemma 4 checkpoints (E2B/E4B, 12B, 31B, 26B-A4B) on the shared runtime: the
    /// <see cref="Gemma4DecoderModel"/> recipe, its image / video / audio towers (Gemma4RuntimeModel.Multimodal.cs) and its
    /// draft heads (the gemma4-assistant MTP head, Gemma4RuntimeModel.Assistant.cs, or a DFlash drafter).
    /// </summary>
    internal sealed partial class Gemma4RuntimeModel : Gemma4DecoderModel
    {
        internal Gemma4RuntimeModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            int layerSplitDegree = 1, string draftModelPath = null)
            : base(ggufPath, backend, tpDegree, tpGroup, layerSplitDegree)
        {
            // A DFlash drafter attaches with the trunk; the gemma4-assistant head attaches after the load
            // (SpeculativeDraftHeadLoader), as it did for the legacy model.
            if (!string.IsNullOrWhiteSpace(draftModelPath) && SpeculativeDraftHeadLoader.IsDFlashDrafter(draftModelPath))
                AttachDraftModel(draftModelPath);
        }
    }
}
