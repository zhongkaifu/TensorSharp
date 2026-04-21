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
        IMultimodalInjector MultimodalInjector { get; }
        IBackendExecutionPlan ExecutionPlan { get; }
        float[] Forward(int[] tokens);
        void ResetKVCache();

        /// <summary>
        /// Whether this architecture can rewind its KV state to an earlier prefix length.
        /// Models with recurrent / SSM state (e.g. Qwen3.5 GatedDeltaNet, Nemotron Mamba2)
        /// cannot truncate because their running state cannot be reversed; for those the
        /// only valid reuse pattern is "cached prefix is a prefix of the new input".
        /// </summary>
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

        /// <summary>
        /// Queue any media embeddings whose insertion span lies AFTER <paramref name="reusablePrefixTokenCount"/>.
        /// Returns true if any embedding span overlaps the suffix that will be re-forwarded.
        /// </summary>
        bool QueuePromptEmbeddings(int reusablePrefixTokenCount);

        /// <summary>
        /// Find the largest prefix length &lt;= <paramref name="reusablePrefixTokenCount"/> that does
        /// not split a multimodal embedding span. The model's KV cache for any such span
        /// is only valid when the entire span has been forwarded.
        /// </summary>
        int ClampReusablePrefix(int reusablePrefixTokenCount);

        /// <summary>
        /// Find the smallest trim-start position &gt;= <paramref name="trimStartTokenCount"/> that does
        /// not split a multimodal embedding span (used when truncating prompts that are too long).
        /// </summary>
        int ClampTrimStart(int trimStartTokenCount);

        /// <summary>
        /// Drop / shift queued embedding spans after the prompt has been trimmed at the front.
        /// </summary>
        void TrimPreparedPrompt(int trimStartTokenCount);
    }

    public interface IBackendExecutionPlan
    {
        BackendType BackendType { get; }
        bool UsesGgmlBackend { get; }
        bool ShouldStoreWeightQuantized(GgufTensorInfo info);
    }
}
