// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp.Runtime;

/// <summary>Immutable policy for protocols whose reasoning and answer are separate
/// messages. Transition tokens are sampled normally, retaining one inference request
/// and its KV/history. Each sampler owns its parser and committed-history tracker.</summary>
public sealed class MessageThinkingTokenBudget
{
    public const int MaxTransitions = 3;
    public MessageThinkingTokenBudget(int tokenLimit, ITokenizer tokenizer, string architecture,
        string? generationSuffix, IReadOnlyList<int> transitionTokenIds)
    {
        if (tokenLimit <= 0) throw new ArgumentOutOfRangeException(nameof(tokenLimit));
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(transitionTokenIds);
        if (transitionTokenIds.Count is < 1 or > 128
            || transitionTokenIds.Any(id => id < 0 || id >= tokenizer.VocabSize || tokenizer.IsEos(id)))
            throw new ArgumentException("A reasoning transition must contain bounded, non-EOS vocabulary tokens.", nameof(transitionTokenIds));
        TokenLimit = tokenLimit;
        Tokenizer = tokenizer;
        Architecture = architecture;
        GenerationSuffix = generationSuffix;
        TransitionTokenIds = Array.AsReadOnly(transitionTokenIds.ToArray());
    }

    public int TokenLimit { get; }
    public IReadOnlyList<int> TransitionTokenIds { get; }
    internal ITokenizer Tokenizer { get; }
    internal string Architecture { get; }
    internal string? GenerationSuffix { get; }
    internal Tracker CreateTracker() => new(this);

    internal sealed class Tracker
    {
        private readonly MessageThinkingTokenBudget _policy;
        private IOutputParser _parser = null!;
        private Decoder _decoder = null!;
        private readonly List<byte> _bytes = new();
        private char[] _characters = new char[128];
        private int _scanned, _lastToken = -1, _thinkingTokens, _transitionFrom = -1, _transitions;
        private volatile int _exhaustedAt = -1;
        internal bool Exhausted => _exhaustedAt >= 0;
        internal bool ExhaustedAt(int observedOutputCount)
        {
            int exhaustedAt = _exhaustedAt;
            return exhaustedAt >= 0 && observedOutputCount >= exhaustedAt;
        }

        internal Tracker(MessageThinkingTokenBudget policy) { _policy = policy; Reset(); }

        private void Reset()
        {
            _parser = OutputParserFactory.Create(_policy.Architecture);
            _parser.Init(true, null);
            _parser.SetGenerationPromptSuffix(_policy.GenerationSuffix);
            _decoder = Encoding.UTF8.GetDecoder();
            _scanned = _thinkingTokens = 0;
            _transitions = 0;
            _exhaustedAt = -1;
            _lastToken = _transitionFrom = -1;
        }

        internal bool TryGetForcedToken(IList<int>? history, bool grammarActive, out int token)
        {
            token = -1;
            Scan(history);
            // A delayed grammar may have activated on the last committed token.
            // Never override it, including halfway through a transition.
            if (grammarActive || Exhausted || _transitionFrom < 0) return false;
            int offset = (history?.Count ?? 0) - _transitionFrom;
            if (offset < 0 || offset >= _policy.TransitionTokenIds.Count) return false;
            token = _policy.TransitionTokenIds[offset];
            return true;
        }

        internal bool IsThinkingOpen(IList<int> history)
        {
            Scan(history);
            return _parser.IsThinkingChannelOpen;
        }

        internal bool RequestClosure(IList<int> history, bool grammarActive)
        {
            Scan(history);
            if (grammarActive || Exhausted || !_parser.IsThinkingChannelOpen) return false;
            if (_transitionFrom >= 0) return true;   // one under way keeps its place
            if (_transitions >= MaxTransitions) { _exhaustedAt = history.Count; return false; }
            _transitionFrom = history.Count;
            return true;
        }

        private void Scan(IList<int>? history)
        {
            int count = history?.Count ?? 0;
            // Peeks do not consume tokens. Rewound committed history rebuilds the
            // parser and UTF-8 decoder, just as the ordinary tag-budget scanner does.
            if (_scanned > count || (_scanned > 0 && history![_scanned - 1] != _lastToken)) Reset();
            for (int index = _scanned; index < count; index++)
            {
                int token = history![index];
                if (_transitionFrom >= 0 && index >= _transitionFrom)
                {
                    int offset = index - _transitionFrom;
                    if (offset >= _policy.TransitionTokenIds.Count || token != _policy.TransitionTokenIds[offset])
                        _transitionFrom = -1;
                }
                _bytes.Clear();
                _policy.Tokenizer.AppendTokenBytes(token, _bytes);
                if (_characters.Length < _bytes.Count + 2) _characters = new char[_bytes.Count + 2];
                int written = _decoder.GetChars(_bytes.ToArray(), 0, _bytes.Count, _characters, 0, flush: false);
                bool wasThinking = _parser.IsThinkingChannelOpen;
                var delta = _parser.Add(written == 0 ? string.Empty : new string(_characters, 0, written), false);
                if (!string.IsNullOrEmpty(delta.Thinking) || (wasThinking && _parser.IsThinkingChannelOpen))
                    _thinkingTokens++;
                if (_transitionFrom >= 0 && index - _transitionFrom + 1 == _policy.TransitionTokenIds.Count)
                {
                    _transitions++;
                    _transitionFrom = -1;
                }
                if (_transitionFrom < 0 && _parser.IsThinkingChannelOpen && _thinkingTokens >= _policy.TokenLimit)
                {
                    if (_transitions >= MaxTransitions && !Exhausted) _exhaustedAt = index + 1;
                    else _transitionFrom = index + 1;
                }
            }
            _scanned = count;
            _lastToken = count > 0 ? history![count - 1] : -1;
        }
    }
}
