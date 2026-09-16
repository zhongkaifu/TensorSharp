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
            List<ToolFunction> tools, ITokenizer tokenizer, List<int> promptTokens)
        {
            var parser = OutputParserFactory.Create(architecture);
            parser.Init(enableThinking, tools);

            if (parser is Gemma4OutputParser)
            {
                // A tool-result prompt may already open thought even with --think off.
                // Inspect only the actual tail; a family default cannot identify that
                // state, and decoding a long prompt again would add needless work.
                int count = Math.Min(promptTokens.Count, 64);
                string tail = tokenizer.Decode(promptTokens.GetRange(promptTokens.Count - count, count));
                parser.SetGenerationPromptSuffix(tail);
            }

            return parser;
        }
    }
}
