// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System.Collections.Generic;

namespace TensorSharp.Runtime
{
    public sealed class GgufPromptRenderer : IPromptRenderer
    {
        public string Render(
            string template,
            List<ChatMessage> messages,
            bool addGenerationPrompt = true,
            string architecture = null,
            List<ToolFunction> tools = null,
            bool enableThinking = false)
        {
            return ChatTemplate.RenderFromGgufTemplate(
                template,
                messages,
                addGenerationPrompt,
                architecture,
                tools,
                enableThinking);
        }
    }
}

