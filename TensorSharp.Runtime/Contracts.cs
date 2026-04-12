// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;

namespace TensorSharp.Runtime
{
    public interface IModelArchitecture : IDisposable
    {
        ModelConfig Config { get; }
        ITokenizer Tokenizer { get; }
        IKVCachePolicy KVCachePolicy { get; }
        IMultimodalInjector MultimodalInjector { get; }
        IBackendExecutionPlan ExecutionPlan { get; }
        float[] Forward(int[] tokens);
        void ResetKVCache();
        bool SupportsKVCacheTruncation { get; }
        void TruncateKVCache(int tokenCount);
    }

    public interface IPromptRenderer
    {
        string Render(
            string template,
            List<ChatMessage> messages,
            bool addGenerationPrompt = true,
            string architecture = null,
            List<ToolFunction> tools = null,
            bool enableThinking = false);
    }

    public interface IOutputProtocolParser
    {
        void Init(bool enableThinking, List<ToolFunction> tools);
        ParsedOutput Add(string text, bool done);
        bool HasThinkingSupport { get; }
        bool HasToolSupport { get; }
        bool AlwaysRequired { get; }
    }

    public interface IMultimodalInjector
    {
        void LoadProjectors(string mmProjPath);
        List<int> ProcessPromptTokens(List<ChatMessage> history, List<int> inputTokens);
    }

    public interface IKVCachePolicy
    {
        int ComputeReusablePrefix(IModelArchitecture model, List<int> cachedTokens, List<int> inputTokens, bool hasMultimodal);
    }

    public interface IBackendExecutionPlan
    {
        BackendType BackendType { get; }
        bool UsesGgmlBackend { get; }
        bool ShouldStoreWeightQuantized(GgufTensorInfo info);
    }
}

