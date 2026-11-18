// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Collections.Generic;
using System.Text;

namespace TensorSharp.Server;

/// <summary>
/// How many of a reply's generated tokens were reasoning (OpenAI's <c>reasoning_tokens</c>): the reply's tokens decoded one
/// at a time and read by the family's output parser as a client reads them, each token counted where the parser put its
/// text. A token whose text the parser holds back (a partial tag, a close marker, a thinking-off reply not decided yet) is
/// counted with the text that follows it, or with the reasoning before it when the answer follows: the close marker of a
/// reasoning block counts as reasoning, and a reply that reasoned past a closed block counts that stretch once a close
/// shows it was reasoning. Tool calls and answer text are not reasoning. A reply cut short (<c>replyWasCut</c>)
/// is read as a whole API reply of it is (ChatStreamCollector.ParseWhole): for LFM3 a thinking-off reply's stray reasoning
/// is then content and counts as answer.
/// </summary>
internal static class ReasoningTokenCount
{
    private enum Kind { None, Reasoning, Answer }

    internal static int Count(string architecture, ITokenizer tokenizer, IReadOnlyList<int> tokens, bool enableThinking,
        List<ToolFunction> tools, string generationSuffix, bool replyWasCut = false)
    {
        if (tokenizer == null || tokens == null || tokens.Count == 0)
            return 0;
        IOutputParser parser = OutputParserFactory.Create(architecture);
        parser.Init(enableThinking, tools);
        if (replyWasCut)
            parser.StreamWithoutHolding();
        if (generationSuffix != null)
            parser.SetGenerationPromptSuffix(generationSuffix);

        Decoder decoder = new UTF8Encoding(false).GetDecoder();
        var bytes = new List<byte>(16);
        var chars = new char[64];
        int reasoning = 0, pending = 0;
        Kind last = Kind.None;

        void Settle(ParsedOutput parsed, bool final)
        {
            bool thought = !string.IsNullOrEmpty(parsed.Thinking);
            bool answered = !string.IsNullOrEmpty(parsed.Content) || parsed.ToolCalls is { Count: > 0 }
                            || !string.IsNullOrEmpty(parsed.ToolCallText);
            if (thought)
            {
                // Held tokens and this one end in reasoning (a stray block decided at its close included).
                reasoning += pending;
                pending = 0;
                last = answered ? Kind.Answer : Kind.Reasoning;
            }
            else if (answered)
            {
                // Tokens held before the answer resumed close the reasoning that preceded them; what the final flush
                // returns as answer was answer text held back.
                if (last == Kind.Reasoning && !final)
                    reasoning += pending - 1;
                pending = 0;
                last = Kind.Answer;
            }
            else if (final && last == Kind.Reasoning)
            {
                reasoning += pending;
                pending = 0;
            }
        }

        foreach (int token in tokens)
        {
            bytes.Clear();
            tokenizer.AppendTokenBytes(token, bytes);
            byte[] piece = bytes.ToArray();
            int needed = decoder.GetCharCount(piece, 0, piece.Length, flush: false);
            if (needed > chars.Length) chars = new char[needed * 2];
            int n = decoder.GetChars(piece, 0, piece.Length, chars, 0, flush: false);
            pending++;
            Settle(parser.Add(new string(chars, 0, n), false), final: false);
        }
        Settle(parser.Add(string.Empty, true), final: true);
        return reasoning;
    }
}
