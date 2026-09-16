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
using System.Text;

namespace TensorSharp.Server.ProtocolAdapters
{
    /// <summary>
    /// Accumulates a chat stream for the non-streaming endpoints, which need the whole
    /// answer before they can write anything.
    ///
    /// <para>
    /// A stream arrives in one of two shapes. Ordinarily it is raw model text and the
    /// caller parses it once at the end. When the request selected skills it arrives
    /// pre-separated instead: the disclosure loop had to parse it anyway to find the
    /// <c>skills_read</c> calls it answers itself, so it forwards content, reasoning and
    /// tool calls on their own fields with <see cref="ChatStreamUpdate.IsParsed"/> set
    /// (see <c>SkillChatLoop</c>). Parsing that a second time would be parsing parsed
    /// text — for most protocols a no-op, but for any whose markers can occur in prose
    /// a way to lose the answer.
    /// </para>
    /// <para>
    /// This collects whichever shape it is given and hands back one
    /// <see cref="ParsedOutput"/>, so the endpoints do not each have to know the
    /// difference.
    /// </para>
    /// </summary>
    internal sealed class ChatStreamCollector
    {
        private readonly StringBuilder _raw = new StringBuilder();
        private readonly StringBuilder _content = new StringBuilder();
        private readonly StringBuilder _thinking = new StringBuilder();
        private readonly List<ToolCall> _toolCalls = new List<ToolCall>();
        private string? _generationSuffix;

        /// <summary>Whether any update arrived pre-separated.</summary>
        public bool IsParsed { get; private set; }

        /// <summary>
        /// The raw text as the model produced it. Empty for a pre-separated stream,
        /// which never carries the markup; use <see cref="Resolve"/> instead.
        /// </summary>
        public string RawText => _raw.ToString();

        /// <summary>Add one non-terminal update.</summary>
        public void Add(ChatStreamUpdate update)
        {
            if (!update.Done && update.RawGenerationSuffix != null)
                _generationSuffix = update.RawGenerationSuffix;
            if (update.IsParsed)
            {
                IsParsed = true;
                if (!string.IsNullOrEmpty(update.Piece))
                    _content.Append(update.Piece);
                if (!string.IsNullOrEmpty(update.ThinkingPiece))
                    _thinking.Append(update.ThinkingPiece);
                if (update.ParsedToolCalls != null && update.ParsedToolCalls.Count > 0)
                    _toolCalls.AddRange(update.ParsedToolCalls);
                return;
            }

            if (!string.IsNullOrEmpty(update.Piece))
                _raw.Append(update.Piece);
        }

        /// <summary>
        /// Produce the finished answer, parsing the raw text with
        /// <paramref name="architecture"/>'s parser unless the stream was already parsed
        /// for us.
        /// </summary>
        public ParsedOutput Resolve(string architecture, bool enableThinking, List<ToolFunction> tools)
        {
            if (IsParsed)
            {
                return new ParsedOutput
                {
                    Content = _content.ToString(),
                    Thinking = _thinking.ToString(),
                    ToolCalls = _toolCalls.Count > 0 ? new List<ToolCall>(_toolCalls) : null,
                };
            }

            var parser = OutputParserFactory.Create(architecture);
            parser.Init(enableThinking, tools);
            parser.SetGenerationPromptSuffix(_generationSuffix);
            return parser.Add(_raw.ToString(), true);
        }

        /// <summary>
        /// The text a caller that does no parsing at all should emit: the raw stream, or
        /// just the content when the stream was pre-separated.
        /// </summary>
        public string PlainText() => IsParsed ? _content.ToString() : _raw.ToString();
    }
}
