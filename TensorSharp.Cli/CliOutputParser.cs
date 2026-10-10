// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
// Licensed under the BSD-3-Clause license in the repository root.

using System;
using System.Collections.Generic;

namespace TensorSharp.Cli
{
    /// <summary>Initializes parsers for CLI paths that generate without the chat pipeline.</summary>
    internal static class CliOutputParser
    {
        internal static IOutputParser Create(string architecture, bool enableThinking,
            List<ToolFunction> tools, string promptTail)
        {
            var parser = OutputParserFactory.Create(architecture);
            parser.Init(enableThinking, tools);
            parser.SetGenerationPromptSuffix(promptTail);
            return parser;
        }

        /// <summary>
        /// The prompt's tail, for a family whose parsers start where the prompt left off
        /// rather than where the request's flag says; null for every other family. A Gemma 4
        /// tool-result prompt may already open thought even with --think off, and a template
        /// whose thinking-off replies may reason anyway
        /// (<see cref="ChatProtocol.ThinkingOffReplyMayReason"/>) is told whether its block
        /// is open. Only the actual tail is inspected: a family default cannot identify that
        /// state, and decoding a long prompt again would add needless work. The same tail
        /// primes the turn's <see cref="ToolCallTurnEnd"/>, which must parse as the printed
        /// output does.
        /// </summary>
        internal static string PromptTail(string architecture, string chatTemplate,
            ITokenizer tokenizer, List<int> promptTokens)
        {
            bool readsTail = OutputParserFactory.Create(architecture) is Gemma4OutputParser
                || OutputParserFactory.ThinkingOffReplyMayReason(architecture, chatTemplate);
            if (!readsTail || tokenizer == null || promptTokens == null || promptTokens.Count == 0)
                return null;
            int count = Math.Min(promptTokens.Count, 64);
            return tokenizer.Decode(promptTokens.GetRange(promptTokens.Count - count, count));
        }
    }
}
