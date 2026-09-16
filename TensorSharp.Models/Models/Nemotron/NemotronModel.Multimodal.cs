// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Nemotron-H's multimodal contract. Audio weights are an optional companion.
using System.Collections.Generic;

using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    public partial class NemotronModel : IVisionCapableModel, IAudioCapableModel, IAudioEncoderLoader, IMultimodalPromptExpander
    {
        bool IVisionCapableModel.IsVisionEncoderLoaded => VisionEncoder != null;

        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(
            ModelMultimodalInjector injector, List<ChatMessage> history, List<int> inputTokens)
            => injector.ProcessNemotronHistory(this, history, inputTokens);
    }
}
