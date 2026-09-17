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

namespace TensorSharp.Runtime.Grammar
{
    /// <summary>Immutable UTF-8 marker matcher; requests own only the matched-prefix index.</summary>
    internal sealed class GrammarByteTrigger
    {
        internal readonly string Text;
        internal readonly byte[] Bytes;
        private readonly int[] _failure;

        internal GrammarByteTrigger(string text)
        {
            Text = text;
            Bytes = System.Text.Encoding.UTF8.GetBytes(text);
            _failure = new int[Bytes.Length];
            for (int i = 1, matched = 0; i < Bytes.Length; i++)
            {
                while (matched > 0 && Bytes[i] != Bytes[matched]) matched = _failure[matched - 1];
                if (Bytes[i] == Bytes[matched]) matched++;
                _failure[i] = matched;
            }
        }

        internal int Advance(int matched, byte value)
        {
            while (matched > 0 && value != Bytes[matched]) matched = _failure[matched - 1];
            return value == Bytes[matched] ? matched + 1 : matched;
        }
    }

    /// <summary>Immutable ordered gates; text between gates is unconstrained.</summary>
    internal sealed class GrammarByteTriggers
    {
        internal readonly GrammarByteTrigger[] Items;
        internal readonly string CacheKey;

        internal GrammarByteTriggers(string[] text)
        {
            Items = new GrammarByteTrigger[text.Length];
            var key = new System.Text.StringBuilder();
            for (int i = 0; i < text.Length; i++)
            {
                if (string.IsNullOrEmpty(text[i]))
                    throw new ArgumentException("Grammar activation markers must be nonempty.", nameof(text));
                Items[i] = new GrammarByteTrigger(text[i]);
                key.Append(text[i].Length).Append(':').Append(text[i]);
            }
            CacheKey = key.ToString();
        }

        internal bool Advance(ref int stage, ref int matched, byte value)
        {
            GrammarByteTrigger marker = Items[stage];
            matched = marker.Advance(matched, value);
            if (matched != marker.Bytes.Length) return false;
            matched = 0;
            return ++stage == Items.Length;
        }
    }

    /// <summary>
    /// Per-(grammar, vocabulary) cache of everything that does not depend on a
    /// particular request: interned parser states, character transitions between
    /// them, and the token bitmask each state admits. Thread-safe and shared by
    /// every request using the same grammar.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Three layers of reuse, in increasing order of value:
    /// </para>
    /// <list type="number">
    /// <item><b>State interning</b> — equal states become the same object, so the
    /// two caches below can key on them cheaply and the mask cache actually
    /// hits.</item>
    /// <item><b>Transition memo</b> — <c>(state, code point) → state</c>. Without
    /// it the trie walk would re-derive the same character transition once per
    /// trie node; inside a JSON string that is hundreds of thousands of
    /// redundant grammar advances for a few hundred distinct characters.</item>
    /// <item><b>Mask cache</b> — <c>state → bitmask</c>. JSON generation cycles
    /// through a handful of states ("expecting a key", "inside a string", "after
    /// a comma"), so in steady state producing a mask is a dictionary lookup.
    /// This is the adaptive-cache idea from xgrammar.</item>
    /// </list>
    /// </remarks>
    public sealed class GrammarMaskCache
    {
        private readonly GrammarMatcher _matcher;
        private readonly GrammarTokenVocabulary _vocab;

        private readonly Dictionary<GrammarState, GrammarState> _interned = new();
        private readonly Dictionary<TransitionKey, GrammarState> _transitions = new();
        private readonly Dictionary<GrammarState, ulong[]> _masks = new();
        private readonly Dictionary<(GrammarState, PartialUtf8), ulong[]> _partialMasks = new();
        private readonly Dictionary<(GrammarState, PartialUtf8, string, int, int, bool), ulong[]> _delayedMasks = new();
        private readonly Dictionary<GrammarState, ulong[]> _leadingMasks = new();
        private readonly object _lock = new();

        /// <summary>
        /// Direct-mapped transitions for ASCII, which is nearly every edge in the
        /// trie: an array index instead of hashing a (state, code point) tuple.
        /// Null = not yet computed. Measured worth only ~2% of a cold mask on a
        /// 592k-node trie — the walk itself dominates, not the lookup — but it is
        /// a few KB per state and removes the tuple hashing from the inner loop.
        /// </summary>
        private readonly Dictionary<GrammarState, GrammarState?[]> _asciiTransitions = new();
        private const int AsciiLimit = 128;

        /// <summary>Cap on retained masks; each costs vocabSize/8 bytes.</summary>
        private const int MaxCachedMasks = 4096;

        public GrammarMaskCache(Grammar grammar, GrammarTokenVocabulary vocab)
        {
            if (grammar == null) throw new ArgumentNullException(nameof(grammar));
            _vocab = vocab ?? throw new ArgumentNullException(nameof(vocab));
            _matcher = new GrammarMatcher(grammar);
            InitialState = Intern(_matcher.InitialState);
        }

        public GrammarState InitialState { get; }
        public GrammarTokenVocabulary Vocabulary => _vocab;
        public GrammarMatcher Matcher => _matcher;

        private readonly struct TransitionKey : IEquatable<TransitionKey>
        {
            public readonly GrammarState State;
            public readonly uint CodePoint;
            public TransitionKey(GrammarState s, uint c) { State = s; CodePoint = c; }
            public bool Equals(TransitionKey o) =>
                ReferenceEquals(State, o.State) && CodePoint == o.CodePoint;
            public override bool Equals(object? o) => o is TransitionKey k && Equals(k);
            public override int GetHashCode() =>
                System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(State) * 397 ^ (int)CodePoint;
        }

        private GrammarState Intern(GrammarState state)
        {
            if (_interned.TryGetValue(state, out GrammarState? existing)) return existing;
            _interned[state] = state;
            return state;
        }

        /// <summary>Advance one code point, memoized.</summary>
        public GrammarState Advance(GrammarState state, uint codePoint)
        {
            lock (_lock)
            {
                return AdvanceLocked(state, codePoint);
            }
        }

        private GrammarState AdvanceLocked(GrammarState state, uint codePoint)
        {
            if (codePoint < AsciiLimit)
            {
                if (!_asciiTransitions.TryGetValue(state, out GrammarState?[]? row))
                {
                    row = new GrammarState?[AsciiLimit];
                    _asciiTransitions[state] = row;
                }
                GrammarState? hit = row[codePoint];
                if (hit != null) return hit;
                GrammarState computed = Intern(_matcher.AcceptCodePoint(state, codePoint));
                row[codePoint] = computed;
                return computed;
            }

            var key = new TransitionKey(state, codePoint);
            if (_transitions.TryGetValue(key, out GrammarState? next)) return next;
            next = Intern(_matcher.AcceptCodePoint(state, codePoint));
            _transitions[key] = next;
            return next;
        }

        /// <summary>
        /// Bitmask of the tokens admitted in <paramref name="state"/>. Bit
        /// <c>i</c> of word <c>i&gt;&gt;6</c> set means token <c>i</c> is legal.
        /// The returned array is shared and must not be modified.
        /// </summary>
        public ulong[] GetMask(GrammarState state)
        {
            lock (_lock)
            {
                if (_masks.TryGetValue(state, out ulong[]? cached)) return cached;

                var mask = new ulong[_vocab.MaskWords];
                // Walk the shared-prefix trie once, carrying the grammar state
                // down each branch and abandoning a subtree the moment its prefix
                // stops being derivable.
                Descend(0, state, PartialUtf8.Empty, mask);

                if (_masks.Count >= MaxCachedMasks) _masks.Clear();
                _masks[state] = mask;
                return mask;
            }
        }

        internal ulong[] GetMask(GrammarState state, PartialUtf8 partial)
        {
            if (partial.Remaining == 0) return GetMask(state);
            lock (_lock)
            {
                var key = (state, partial);
                if (_partialMasks.TryGetValue(key, out ulong[]? cached)) return cached;
                var mask = new ulong[_vocab.MaskWords];
                Descend(0, state, partial, mask);
                if (_partialMasks.Count >= MaxCachedMasks) _partialMasks.Clear();
                return _partialMasks[key] = mask;
            }
        }

        // Before activation only tokens completing the marker with an invalid
        // grammar suffix are forbidden. Traverse shared byte prefixes once per
        // marker-prefix state, then reuse the mask across tokens and requests.
        internal ulong[] GetDelayedMask(GrammarState state, PartialUtf8 partial,
            GrammarByteTriggers trigger, int stage, int matched, ITokenizer tokenizer,
            bool skipLeadingWhitespace = false)
        {
            lock (_lock)
            {
                var key = (state, partial, trigger.CacheKey, stage, matched, skipLeadingWhitespace);
                if (_delayedMasks.TryGetValue(key, out ulong[]? cached)) return cached;
                var mask = new ulong[_vocab.MaskWords];
                Array.Fill(mask, ulong.MaxValue);
                DescendDelayed(0, state, partial, trigger, stage, matched, Phase.Dormant, skipLeadingWhitespace, mask);
                // Control tokens are absent from the grammar trie. They remain
                // free before activation, but their bytes can also complete a
                // marker and contain a suffix that must be checked.
                var bytes = new List<byte>(64);
                foreach (int id in _vocab.SpecialTokenIds)
                {
                    if ((uint)id >= (uint)_vocab.VocabSize || tokenizer.IsEos(id)) continue;
                    bytes.Clear();
                    try { tokenizer.AppendTokenBytes(id, bytes); }
                    catch { continue; }
                    int prefix = matched, nextStage = stage;
                    Phase phase = Phase.Dormant;
                    bool allowed = true;
                    GrammarState next = state;
                    PartialUtf8 utf8 = partial;
                    foreach (byte value in bytes)
                    {
                        if (phase == Phase.Dormant)
                        {
                            if (trigger.Advance(ref nextStage, ref prefix, value))
                                phase = skipLeadingWhitespace ? Phase.Leading : Phase.Active;
                        }
                        else if (phase == Phase.Leading && IsLeadingWhitespace(value)) { }
                        else
                        {
                            phase = Phase.Active;
                            if (!AdvanceByte(ref next, ref utf8, value)) { allowed = false; break; }
                        }
                    }
                    if (!allowed) mask[id >> 6] &= ~(1UL << (id & 63));
                }
                if (_delayedMasks.Count >= MaxCachedMasks) _delayedMasks.Clear();
                return _delayedMasks[key] = mask;
            }
        }

        private bool AdvanceByte(ref GrammarState state, ref PartialUtf8 partial, byte value)
        {
            if (!GrammarMatcher.TryFeedByte(ref partial, value, out uint codePoint, out bool complete)) return false;
            if (complete)
            {
                state = AdvanceLocked(state, codePoint);
                return !state.IsDead;
            }
            return _matcher.AcceptsPartial(state, partial);
        }

        private void ClearSubtree(int node, ulong[] mask)
        {
            foreach (int token in _vocab.TokensAt(node))
                mask[token >> 6] &= ~(1UL << (token & 63));
            for (int edge = _vocab.ChildStart(node); edge < _vocab.ChildEnd(node); edge++)
                ClearSubtree(_vocab.EdgeNode(edge), mask);
        }

        /// <summary>Where a delayed constraint is along one token's bytes.</summary>
        private enum Phase { Dormant, Leading, Active }

        /// <summary>Whitespace a model may write between a trigger and the value.</summary>
        internal static bool IsLeadingWhitespace(byte value) =>
            value == (byte)' ' || value == (byte)'\t' || value == (byte)'\n' || value == (byte)'\r';

        private void DescendDelayed(int node, GrammarState state, PartialUtf8 partial,
            GrammarByteTriggers trigger, int stage, int matched, Phase phase, bool skipLeadingWhitespace, ulong[] mask)
        {
            for (int edge = _vocab.ChildStart(node); edge < _vocab.ChildEnd(node); edge++)
            {
                int child = _vocab.EdgeNode(edge), prefix = matched, nextStage = stage;
                byte value = _vocab.EdgeByte(edge);
                GrammarState next = state;
                PartialUtf8 utf8 = partial;
                Phase nextPhase = phase;
                if (phase == Phase.Dormant)
                {
                    if (trigger.Advance(ref nextStage, ref prefix, value))
                        nextPhase = skipLeadingWhitespace ? Phase.Leading : Phase.Active;
                }
                else if (phase == Phase.Leading && IsLeadingWhitespace(value)) { }
                else
                {
                    nextPhase = Phase.Active;
                    if (!AdvanceByte(ref next, ref utf8, value))
                    {
                        ClearSubtree(child, mask);
                        continue;
                    }
                }
                DescendDelayed(child, next, utf8, trigger, nextStage, prefix, nextPhase, skipLeadingWhitespace, mask);
            }
        }

        /// <summary>
        /// Mask right after a trigger fired when leading whitespace is skipped: a token
        /// may be whitespace, or whitespace followed by text the grammar admits from
        /// <paramref name="state"/>. Control tokens stay forbidden, as in
        /// <see cref="GetMask(GrammarState)"/>.
        /// </summary>
        internal ulong[] GetLeadingWhitespaceMask(GrammarState state)
        {
            lock (_lock)
            {
                if (_leadingMasks.TryGetValue(state, out ulong[]? cached)) return cached;
                var mask = new ulong[_vocab.MaskWords];
                DescendLeading(0, state, mask);
                if (_leadingMasks.Count >= MaxCachedMasks) _leadingMasks.Clear();
                return _leadingMasks[state] = mask;
            }
        }

        private void DescendLeading(int node, GrammarState state, ulong[] mask)
        {
            int end = _vocab.ChildEnd(node);
            for (int edge = _vocab.ChildStart(node); edge < end; edge++)
            {
                if (!IsLeadingWhitespace(_vocab.EdgeByte(edge)))
                {
                    DescendEdge(edge, state, PartialUtf8.Empty, mask);
                    continue;
                }
                int child = _vocab.EdgeNode(edge);
                foreach (int tokenId in _vocab.TokensAt(child))
                    mask[tokenId >> 6] |= 1UL << (tokenId & 63);
                DescendLeading(child, state, mask);
            }
        }

        private void Descend(int node, GrammarState state, PartialUtf8 partial, ulong[] mask)
        {
            int end = _vocab.ChildEnd(node);
            for (int edge = _vocab.ChildStart(node); edge < end; edge++)
                DescendEdge(edge, state, partial, mask);
        }

        private void DescendEdge(int edge, GrammarState state, PartialUtf8 partial, ulong[] mask)
        {
            byte b = _vocab.EdgeByte(edge);

            PartialUtf8 nextPartial = partial;
            if (!GrammarMatcher.TryFeedByte(ref nextPartial, b, out uint codePoint, out bool complete))
                return;                         // malformed UTF-8: no token below this can be text

            GrammarState nextState;
            if (complete)
            {
                nextState = AdvanceLocked(state, codePoint);
                if (nextState.IsDead) return;   // prunes the entire subtree
            }
            else
            {
                // Mid-character: keep the branch only if some completion of
                // this partial code point could still satisfy the grammar.
                if (!_matcher.AcceptsPartial(state, nextPartial)) return;
                nextState = state;
            }

            int child = _vocab.EdgeNode(edge);
            // Byte-fallback tokens can end mid-character. Accept preserves
            // that UTF-8 prefix, and the next mask checks its continuation.
            foreach (int tokenId in _vocab.TokensAt(child))
                mask[tokenId >> 6] |= 1UL << (tokenId & 63);

            Descend(child, nextState, nextPartial, mask);
        }
    }

    /// <summary>
    /// Per-request grammar constraint: the live parser position plus the shared
    /// caches. One instance per sequence; not thread-safe on its own, which is
    /// fine because a sequence decodes serially.
    /// </summary>
    public sealed class GrammarConstraint
    {
        private readonly GrammarMaskCache _cache;
        private GrammarState _state;
        private PartialUtf8 _partial;
        private readonly List<byte> _tokenBytes = new(64);
        private readonly ITokenizer _tokenizer;

        // --- lazy activation (see ActivateAfter) ---
        private GrammarByteTriggers? _trigger;
        private int _triggerStage;
        private int _triggerMatched;
        private bool _active = true;
        // Whitespace right after the trigger is prelude (see ActivateAfter).
        private bool _skipLeadingWhitespace;
        private bool _inLeadingWhitespace;

        public GrammarConstraint(Grammar grammar, ITokenizer tokenizer)
            : this(new GrammarMaskCache(grammar, GrammarTokenVocabulary.ForTokenizer(tokenizer)), tokenizer)
        {
        }

        public GrammarConstraint(GrammarMaskCache cache, ITokenizer tokenizer)
        {
            _cache = cache ?? throw new ArgumentNullException(nameof(cache));
            _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            _state = cache.InitialState;
            _partial = PartialUtf8.Empty;
        }

        /// <summary>
        /// Copy the current parser and delayed-trigger position for an independent
        /// request. Interned grammar states and synchronized mask caches are shared;
        /// mutable UTF-8, trigger-tail and token scratch state are request-owned.
        /// The source must not be mutated concurrently while taking the copy.
        /// </summary>
        public GrammarConstraint Fork() => new GrammarConstraint(_cache, _tokenizer)
        {
            _state = _state,
            _partial = _partial,
            _trigger = _trigger,
            _triggerStage = _triggerStage,
            _triggerMatched = _triggerMatched,
            _active = _active,
            _skipLeadingWhitespace = _skipLeadingWhitespace,
            _inLeadingWhitespace = _inLeadingWhitespace,
        };

        /// <summary>
        /// Hold the constraint dormant until <paramref name="triggerText"/> has
        /// been generated, then enforce the grammar over everything after it.
        ///
        /// This is what makes structured output work on a model that speaks
        /// before it answers. GPT-OSS's harmony format opens every reply with a
        /// channel header (<c>&lt;|channel|&gt;analysis&lt;|message|&gt;…</c>),
        /// so a grammar armed from token 0 forbids the model's own first token
        /// and it is pushed straight into a JSON object it has done no reasoning
        /// for. It answers the shape and not the question — schema-valid
        /// <c>{"city":"...","population":1}</c>. Arming on the final channel's
        /// header instead lets the analysis channel run free and constrains only
        /// the answer. Mirrors llama.cpp's lazy grammar triggers.
        ///
        /// The prelude is not part of the structured output. While dormant,
        /// only a token that completes the marker with an invalid grammar
        /// suffix is masked; ordinary prelude tokens remain unrestricted.
        /// </summary>
        /// <param name="triggerText">Marker after which the grammar enforces.</param>
        /// <param name="skipLeadingWhitespace">
        /// Treat whitespace between the marker and the first grammar byte as part of
        /// the prelude, as llama.cpp's <c>&lt;/think&gt;\s*</c> trigger patterns do.
        /// A vocabulary may merge the marker's last byte with the line break that
        /// follows it (Nemotron-H Reasoning's <c>&gt;\n\n</c>): with a JSON root that
        /// cannot start with whitespace that token is masked, the model's next choice
        /// (<c>}</c>) no longer spells the marker, and the grammar never arms. Only for
        /// grammars whose root does not itself begin with whitespace, such as the JSON
        /// response_format grammars.
        /// </param>
        public void ActivateAfter(string triggerText, bool skipLeadingWhitespace)
        {
            if (string.IsNullOrEmpty(triggerText)) return;
            ActivateAfterTriggers(triggerText);
            _skipLeadingWhitespace = skipLeadingWhitespace;
        }

        public void ActivateAfter(string triggerText) => ActivateAfter(triggerText, skipLeadingWhitespace: false);

        /// <summary>
        /// Activate after all markers appear in order. Each marker is matched
        /// against exact token bytes; text before and between markers is free.
        /// For example, a tool marker quoted in reasoning must not activate a
        /// tool grammar that first requires the trained reasoning-end marker.
        /// </summary>
        public void ActivateAfterTriggers(params string[] triggerTexts)
        {
            if (triggerTexts == null) throw new ArgumentNullException(nameof(triggerTexts));
            if (triggerTexts.Length == 0) throw new ArgumentException("At least one activation marker is required.", nameof(triggerTexts));
            _trigger = new GrammarByteTriggers(triggerTexts);
            _triggerStage = _triggerMatched = 0;
            _active = false;
            _skipLeadingWhitespace = _inLeadingWhitespace = false;
        }

        /// <summary>Whether the grammar is currently enforcing (see <see cref="ActivateAfter"/>).</summary>
        public bool IsActive => _active;

        /// <summary>The grammar can legally end here, so EOS is permitted.</summary>
        public bool IsComplete => !_active || (_partial.Remaining == 0 && _state.CanTerminate);

        /// <summary>
        /// No continuation exists. Should not happen while the sampler honours
        /// the mask; it is the signal that something bypassed the constraint.
        /// </summary>
        public bool IsDead => _state.IsDead;

        /// <summary>Current mask; bit <c>i</c> set means token <c>i</c> is legal.</summary>
        public ulong[] CurrentMask() => !_active
            ? _cache.GetDelayedMask(_state, _partial, _trigger!, _triggerStage, _triggerMatched, _tokenizer, _skipLeadingWhitespace)
            : _inLeadingWhitespace ? _cache.GetLeadingWhitespaceMask(_state)
            : _cache.GetMask(_state, _partial);

        /// <summary>Opaque snapshot for rollback (speculative decoding).</summary>
        public (GrammarState State, PartialUtf8 Partial) Snapshot() => (_state, _partial);

        /// <summary>Restore a snapshot taken earlier.</summary>
        public void Restore((GrammarState State, PartialUtf8 Partial) snapshot)
        {
            _state = snapshot.State;
            _partial = snapshot.Partial;
        }

        /// <summary>
        /// Advance the parser over raw UTF-8 bytes. Use when text has already
        /// been produced outside the sampler — a forced prefix, a prefilled
        /// partial response being resumed, or a tool-call envelope the server
        /// emitted itself — so the constraint stays aligned with what the client
        /// will actually see.
        /// </summary>
        public void AcceptBytes(ReadOnlySpan<byte> bytes)
        {
            foreach (byte b in bytes)
            {
                if (!AcceptByte(b)) return;
            }
        }

        private bool AcceptByte(byte value)
        {
            if (!_active)
            {
                if (_trigger!.Advance(ref _triggerStage, ref _triggerMatched, value))
                {
                    _active = true;
                    _trigger = null;
                    _triggerMatched = 0;
                    _inLeadingWhitespace = _skipLeadingWhitespace;
                }
                return true;
            }
            if (_inLeadingWhitespace)
            {
                if (GrammarMaskCache.IsLeadingWhitespace(value)) return true;
                _inLeadingWhitespace = false;
            }
            if (!GrammarMatcher.TryFeedByte(ref _partial, value, out uint cp, out bool complete))
            {
                _state = new GrammarState(Array.Empty<int[]>());
                return false;
            }
            if (complete) _state = _cache.Advance(_state, cp);
            return !_state.IsDead;
        }

        // Latched once a token's bytes could not be decoded in Accept(): the
        // token is emitted without advancing the parser, so warning per token
        // would spam every remaining step of the response.
        private static bool _acceptDecodeFailReported;

        /// <summary>
        /// Commit a token the sampler chose, advancing the parser. EOS and other
        /// special ids carry no text and leave the state untouched.
        /// </summary>
        public void Accept(int tokenId)
        {
            if (_tokenizer.IsEos(tokenId)) return;

            _tokenBytes.Clear();
            try
            {
                _tokenizer.AppendTokenBytes(tokenId, _tokenBytes);
            }
            catch (Exception ex)
            {
                if (!_acceptDecodeFailReported)
                {
                    _acceptDecodeFailReported = true;
                    Console.Error.WriteLine(
                        $"[GrammarConstraint] Tokenizer failed to decode the bytes of accepted token {tokenId} ({ex.Message}); " +
                        "the token is emitted without advancing the grammar parser, so constrained output may " +
                        "no longer match the grammar/schema from this point on. Reported once.");
                }
                return;
            }
            if (_tokenBytes.Count == 0) return;

            for (int i = 0; i < _tokenBytes.Count; i++)
            {
                if (!AcceptByte(_tokenBytes[i])) return;
            }
        }

        /// <summary>
        /// Set every token the grammar forbids to negative infinity, so any
        /// downstream sampler — greedy, top-k, top-p — can only pick a legal one.
        /// </summary>
        /// <param name="logits">Logit buffer, modified in place.</param>
        /// <param name="allowEos">
        /// Whether EOS ids may survive; callers pass <see cref="IsComplete"/>.
        /// </param>
        public void ApplyMask(float[] logits, bool allowEos)
        {
            if (logits == null) throw new ArgumentNullException(nameof(logits));
            ulong[] mask = CurrentMask();
            int vocab = Math.Min(logits.Length, _cache.Vocabulary.VocabSize);

            // EOS ids are deliberately absent from the trie, so the sweep below
            // would bury them. Save the model's own scores first and put them
            // back afterwards: whether generation should stop is a judgement the
            // grammar can only *permit*, not make. Overwriting the logit with a
            // constant would either force an early stop or suppress a wanted one.
            int[] eosIds = _tokenizer.EosTokenIds;
            Span<float> savedEos = eosIds.Length <= 8 ? stackalloc float[eosIds.Length] : new float[eosIds.Length];
            if (allowEos)
            {
                for (int i = 0; i < eosIds.Length; i++)
                    savedEos[i] = (uint)eosIds[i] < (uint)logits.Length
                        ? logits[eosIds[i]]
                        : float.NegativeInfinity;
            }

            for (int word = 0; word < mask.Length; word++)
            {
                ulong bits = mask[word];
                int baseId = word << 6;
                int limit = Math.Min(64, vocab - baseId);
                if (limit <= 0) break;
                if (bits == ulong.MaxValue && limit == 64) continue;   // every id legal
                for (int b = 0; b < limit; b++)
                {
                    if ((bits & (1UL << b)) == 0)
                        logits[baseId + b] = float.NegativeInfinity;
                }
            }

            if (allowEos)
            {
                for (int i = 0; i < eosIds.Length; i++)
                    if ((uint)eosIds[i] < (uint)logits.Length)
                        logits[eosIds[i]] = savedEos[i];
            }
        }
    }
}
