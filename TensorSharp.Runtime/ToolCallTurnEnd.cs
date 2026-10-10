// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Ends a turn at its first complete tool call, for a family that would otherwise go
    /// on to write the call's result itself (<see cref="ChatProtocol.ToolCallEndsTurn"/>).
    ///
    /// <para>
    /// The family's own output parser decides, fed the same pieces the client gets. A
    /// substring match on the close tag stopped turns whose reasoning only NAMED the tag
    /// (the tool instructions spell it out) and cut calls whose string argument contained
    /// it, leaving an empty answer or no call. The parser already knows both cases: an open
    /// tag needs a call body, and a close inside a string is data.
    /// </para>
    /// <para>
    /// One instance per generated turn; the server pipeline and the CLI loops share it.
    /// </para>
    /// </summary>
    public sealed class ToolCallTurnEnd
    {
        private readonly IOutputParser _parser;

        private ToolCallTurnEnd(IOutputParser parser) => _parser = parser;

        /// <summary>A watcher for this turn, or null when the turn does not end at a call:
        /// no tools declared, or a family/template that keeps writing after its calls.</summary>
        public static ToolCallTurnEnd? For(string? architecture, string? ggufTemplate, bool enableThinking,
            List<ToolFunction>? tools, string? generationSuffix = null)
        {
            if (tools is not { Count: > 0 }) return null;
            ChatProtocol? protocol = ChatProtocolRegistry.For(architecture);
            if (protocol?.ToolCallEndsTurn?.Invoke(ggufTemplate) != true || protocol.CreateOutputParser == null)
                return null;
            IOutputParser parser = protocol.CreateOutputParser();
            parser.Init(enableThinking, tools);
            parser.SetGenerationPromptSuffix(generationSuffix);
            return new ToolCallTurnEnd(parser);
        }

        /// <summary>Feed the next decoded piece; true once the output holds a complete call.</summary>
        public bool Observe(string piece)
        {
            if (string.IsNullOrEmpty(piece)) return false;
            return _parser.Add(piece, done: false).ToolCalls is { Count: > 0 };
        }

        private readonly List<byte> _bytes = new();
        private int _decodedBytes;

        /// <summary>Feed the next generated token, for a loop that has tokens rather than
        /// text: its bytes are decoded as soon as they complete a UTF-8 character.</summary>
        public bool ObserveToken(ITokenizer tokenizer, int token)
        {
            tokenizer.AppendTokenBytes(token, _bytes);
            int valid = ValidUtf8Length(_bytes);
            if (valid <= _decodedBytes) return false;
            string piece = Encoding.UTF8.GetString(CollectionsMarshal.AsSpan(_bytes).Slice(_decodedBytes, valid - _decodedBytes));
            _decodedBytes = valid;
            return Observe(piece);
        }

        // The length of the longest prefix that does not end inside a multi-byte character.
        private static int ValidUtf8Length(List<byte> bytes)
        {
            int n = bytes.Count;
            for (int back = 1; back <= Math.Min(3, n); back++)
            {
                byte b = bytes[n - back];
                if ((b & 0xC0) == 0x80) continue;          // a continuation byte: keep looking
                int need = (b & 0xE0) == 0xC0 ? 2 : (b & 0xF0) == 0xE0 ? 3 : (b & 0xF8) == 0xF0 ? 4 : 1;
                return back < need ? n - back : n;
            }
            return n;
        }
    }
}
