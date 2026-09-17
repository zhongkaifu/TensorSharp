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
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Optional capability for tokenizers that know which of their ids are
    /// control / special tokens rather than literal text.
    /// </summary>
    /// <remarks>
    /// Grammar-constrained decoding needs this: a control token such as
    /// <c>&lt;|im_start|&gt;</c> has a printable spelling, so a grammar that
    /// admits ordinary characters (any JSON string, say) would happily accept it
    /// as if the model had typed those angle brackets. Excluding them from the
    /// vocabulary trie keeps constraint decisions about *text*.
    /// Implemented by the real tokenizers; test doubles need not.
    /// </remarks>
    public interface ISpecialTokenVocabulary
    {
        /// <summary>Ids that are control/special markers, not literal text.</summary>
        IReadOnlyCollection<int> SpecialTokenIds { get; }
    }

    public interface ITokenizer
    {
        string[] Vocab { get; }
        int BosTokenId { get; }
        int[] EosTokenIds { get; }
        int VocabSize { get; }
        List<int> Encode(string text, bool addSpecial = true);
        string Decode(List<int> ids);
        void AppendTokenBytes(int tokenId, List<byte> buffer);
        bool IsEos(int tokenId);
        int LookupToken(string tokenStr);
    }

    public class BpeTokenizer : ITokenizer, ISpecialTokenVocabulary
    {
        private int[]? _specialTokenIds;

        /// <inheritdoc/>
        public IReadOnlyCollection<int> SpecialTokenIds =>
            _specialTokenIds ??= BuildSpecialTokenIds();

        private int[] BuildSpecialTokenIds()
        {
            var ids = new List<int>(_specialTokens.Count);
            foreach (string s in _specialTokens)
                if (_vocabLookup.TryGetValue(s, out int id)) ids.Add(id);
            foreach (int id in _eosTokenIds) ids.Add(id);
            return ids.Distinct().ToArray();
        }

        private readonly string[] _vocab;
        private readonly Dictionary<string, int> _vocabLookup;
        private readonly Dictionary<string, int> _mergeLookup;
        private readonly List<string> _specialTokens;
        private readonly Regex[] _pretokenizerPasses;
        private readonly int _bosTokenId;
        private readonly int[] _eosTokenIds;
        private readonly bool _addBos;
        private readonly bool _addEos;
        private readonly bool _spmStyleBpe;

        public string[] Vocab => _vocab;
        public int BosTokenId => _bosTokenId;
        public int[] EosTokenIds => _eosTokenIds;
        public int VocabSize => _vocab.Length;

        public BpeTokenizer(string[] vocab, int[] tokenTypes, string[] merges,
            int bosTokenId, int[] eosTokenIds, bool addBos, bool addEos,
            string? preTokenizerType = null)
        {
            _vocab = vocab;
            _bosTokenId = bosTokenId;
            _eosTokenIds = eosTokenIds;
            _addBos = addBos;
            _addEos = addEos;
            _spmStyleBpe = string.Equals(
                preTokenizerType, "gemma4", StringComparison.OrdinalIgnoreCase);

            _vocabLookup = new Dictionary<string, int>(vocab.Length);
            for (int i = 0; i < vocab.Length; i++)
                _vocabLookup[vocab[i]] = i;

            _mergeLookup = new Dictionary<string, int>(merges.Length);
            for (int i = 0; i < merges.Length; i++)
                _mergeLookup.TryAdd(merges[i], i);

            const int TOKEN_TYPE_CONTROL = 3;
            const int TOKEN_TYPE_USER_DEFINED = 4;
            var eogIds = new HashSet<int>(eosTokenIds);
            _specialTokens = new List<string>();
            for (int i = 0; i < vocab.Length; i++)
            {
                if ((tokenTypes != null && i < tokenTypes.Length &&
                     (tokenTypes[i] == TOKEN_TYPE_CONTROL ||
                      tokenTypes[i] == TOKEN_TYPE_USER_DEFINED)) ||
                    eogIds.Contains(i))
                {
                    _specialTokens.Add(vocab[i]);
                }
            }

            string pattern = ResolvePreTokenizerPattern(preTokenizerType);

            _pretokenizerPasses = preTokenizerType is "joyai-llm" or "deepseek-v3" or "hunyuan-dense"
                ? new[]
                {
                    new Regex(@"\p{N}{1,3}", RegexOptions.Compiled),
                    new Regex(@"[一-龥぀-ゟ゠-ヿ]+", RegexOptions.Compiled),
                    new Regex(pattern, RegexOptions.Compiled),
                }
                : new[] { new Regex(pattern, RegexOptions.Compiled) };
        }

        /// <summary>
        /// The pre-tokenizer split regex for a given <c>tokenizer.ggml.pre</c> value,
        /// mirroring llama.cpp's <c>LLAMA_VOCAB_PRE_TYPE_*</c> table. Extracted from the
        /// constructor so the mapping can be unit-tested without a vocabulary.
        /// </summary>
        internal static string ResolvePreTokenizerPattern(string preTokenizerType)
        {
            return preTokenizerType switch
            {
                // llama.cpp maps tokenizer.ggml.pre in {gpt-4o, llama4, kanana2, talkie}
                // to the same LLAMA_VOCAB_PRE_TYPE_GPT4O regex (llama-vocab.cpp:2293-2299).
                // Without "llama4" here these vocabs fell through to the default
                // pattern, whose \p{N} splits every digit individually - so "17"
                // tokenized as '1','7' instead of the single "17" token and every
                // numeric prompt diverged from llama.cpp.
                "gpt-4o" or "llama4" or "kanana2" or "talkie" =>
                    @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|" +
                    @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|" +
                    @"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                // GLM-4 / GLM-5.x (llama.cpp LLAMA_VOCAB_PRE_TYPE_CHATGLM4, selected by
                // tokenizer.ggml.pre "glm4" or "chatglm-bpe"). The default below splits
                // every digit, so "INV-472" reached GLM-5.3-Flash as digits it was never
                // trained on and it quoted it back as "INV-4472".
                "glm4" or "chatglm-bpe" =>
                    @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                "tekken" =>
                    @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|" +
                    @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|" +
                    @"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                // Qwen3.5 keeps Unicode combining marks attached to letters and
                // excludes them from the punctuation run. This matches llama.cpp's
                // tokenizer.ggml.pre == "qwen35" pre-tokenizer exactly.
                "qwen35" =>
                    @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                // DeepSeek V3/V4 family pre-tokenizer (llama.cpp DEEPSEEK3_LLM /
                // JOYAI_LLM). Numbers and CJK are isolated in preceding passes.
                // Combining these into one alternation changes boundaries at mixed
                // CJK/Latin text and can incorrectly attach whitespace to CJK.
                "joyai-llm" or "deepseek-v3" or "hunyuan-dense" =>
                    @"[!""#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|" +
                    @"[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+|" +
                    @" ?[\p{P}\p{S}]+[\r\n]*|" +
                    @"\s*[\r\n]+|\s+(?!\S)|\s+",
                // Gemma 4 uses BPE merge ranks, but with SentencePiece-style
                // whitespace escaping and raw UTF-8 code points rather than
                // GPT-2's byte-to-Unicode alphabet.  Merges run across the
                // complete text, splitting only at newline runs.
                "gemma4" =>
                    @"[^\n]+|[\n]+",
                _ =>
                    @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            };
        }

        public List<int> Encode(string text, bool addSpecial = true)
        {
            var fragments = new List<(string text, List<int>? ids)>();
            fragments.Add((text, null));

            foreach (var special in _specialTokens)
            {
                int id = _vocabLookup.TryGetValue(special, out var sid) ? sid : -1;
                if (id < 0) continue;

                var newFragments = new List<(string text, List<int>? ids)>();
                foreach (var frag in fragments)
                {
                    if (frag.ids != null)
                    {
                        newFragments.Add(frag);
                        continue;
                    }

                    int startIdx = 0;
                    while (true)
                    {
                        int idx = frag.text.IndexOf(special, startIdx, StringComparison.Ordinal);
                        if (idx < 0)
                        {
                            if (startIdx < frag.text.Length)
                                newFragments.Add((frag.text.Substring(startIdx), null));
                            break;
                        }

                        if (idx > startIdx)
                            newFragments.Add((frag.text.Substring(startIdx, idx - startIdx), null));
                        newFragments.Add((special, new List<int> { id }));
                        startIdx = idx + special.Length;
                    }
                }
                fragments = newFragments;
            }

            var ids = new List<int>();
            foreach (var frag in fragments)
            {
                if (frag.ids != null)
                {
                    ids.AddRange(frag.ids);
                    continue;
                }

                foreach (string split in SplitForBpe(frag.text))
                {
                    if (_spmStyleBpe)
                    {
                        string spmNormalized = split.Replace(" ", "\u2581", StringComparison.Ordinal);

                        // llama.cpp preserves a complete newline run when the
                        // vocabulary contains it.  Otherwise normal BPE merging
                        // starts from Unicode code points.
                        if (spmNormalized.All(c => c == '\n') &&
                            _vocabLookup.TryGetValue(spmNormalized, out int newlineId))
                        {
                            ids.Add(newlineId);
                            continue;
                        }

                        ids.AddRange(BpeMerge(
                            spmNormalized,
                            rawUnicodeCodePoints: true,
                            byteFallback: true));
                        continue;
                    }

                    string normalized = NormalizeSplit(split);

                    if (_vocabLookup.TryGetValue(normalized, out int directId))
                    {
                        ids.Add(directId);
                        continue;
                    }

                    ids.AddRange(BpeMerge(normalized));
                }
            }

            if (addSpecial)
            {
                if (_addBos)
                    ids.Insert(0, _bosTokenId);
                if (_addEos && _eosTokenIds.Length > 0)
                    ids.Add(_eosTokenIds[0]);
            }

            return ids;
        }

        /// <summary>
        /// Apply isolated split passes without dropping unmatched text. .NET regex
        /// Unicode categories classify UTF-16 code units, unlike the code-point
        /// categories in the reference tokenizers. Match a category-equivalent BMP
        /// projection for supplementary characters, then recover original spans.
        /// The common BMP-only path uses the original string without projection.
        /// </summary>
        internal IReadOnlyList<string> SplitForBpe(string text)
        {
            string matchText = text;
            List<int>? offsets = null;
            if (text.AsSpan().IndexOfAnyInRange('\uD800', '\uDFFF') >= 0)
            {
                var projected = new StringBuilder(text.Length);
                offsets = new List<int>(text.Length + 1);
                int sourceOffset = 0;
                foreach (Rune rune in text.EnumerateRunes())
                {
                    offsets.Add(sourceOffset);
                    projected.Append(rune.IsBmp ? (char)rune.Value : CategoryRepresentative(Rune.GetUnicodeCategory(rune)));
                    sourceOffset += rune.Utf16SequenceLength;
                }
                offsets.Add(text.Length);
                matchText = projected.ToString();
            }

            var ranges = new List<(int Start, int Length)> { (0, matchText.Length) };
            foreach (Regex pass in _pretokenizerPasses)
            {
                var next = new List<(int Start, int Length)>();
                foreach (var range in ranges)
                {
                    int position = 0;
                    foreach (var match in pass.EnumerateMatches(matchText.AsSpan(range.Start, range.Length)))
                    {
                        if (match.Index > position) next.Add((range.Start + position, match.Index - position));
                        if (match.Length > 0) next.Add((range.Start + match.Index, match.Length));
                        position = match.Index + match.Length;
                    }
                    if (position < range.Length) next.Add((range.Start + position, range.Length - position));
                }
                ranges = next;
            }

            var splits = new List<string>(ranges.Count);
            foreach (var range in ranges)
            {
                int start = offsets == null ? range.Start : offsets[range.Start];
                int end = offsets == null ? range.Start + range.Length : offsets[range.Start + range.Length];
                splits.Add(text.Substring(start, end - start));
            }
            return splits;
        }

        // Representatives avoid ASCII and the explicit CJK/kana ranges used by
        // pre-tokenizers; only the Unicode category of a supplementary rune should
        // affect their generic letter, number, punctuation and symbol branches.
        private static char CategoryRepresentative(UnicodeCategory category) => category switch
        {
            UnicodeCategory.UppercaseLetter => 'Ω',
            UnicodeCategory.LowercaseLetter => 'α',
            UnicodeCategory.TitlecaseLetter => 'ǅ',
            UnicodeCategory.ModifierLetter => 'ʰ',
            UnicodeCategory.OtherLetter => 'ꀀ',
            UnicodeCategory.NonSpacingMark => '\u0301',
            UnicodeCategory.SpacingCombiningMark => '\u0903',
            UnicodeCategory.EnclosingMark => '\u0488',
            UnicodeCategory.DecimalDigitNumber => '٠',
            UnicodeCategory.LetterNumber => 'Ⅻ',
            UnicodeCategory.OtherNumber => '²',
            UnicodeCategory.ConnectorPunctuation => '‿',
            UnicodeCategory.DashPunctuation => '‐',
            UnicodeCategory.OpenPunctuation => '❨',
            UnicodeCategory.ClosePunctuation => '❩',
            UnicodeCategory.InitialQuotePunctuation => '‘',
            UnicodeCategory.FinalQuotePunctuation => '’',
            UnicodeCategory.OtherPunctuation => '※',
            UnicodeCategory.MathSymbol => '∑',
            UnicodeCategory.CurrencySymbol => '¢',
            UnicodeCategory.ModifierSymbol => '˂',
            UnicodeCategory.OtherSymbol => '♥',
            UnicodeCategory.Format => '\u200B',
            UnicodeCategory.PrivateUse => '\uE000',
            _ => '\u0378',
        };

        private string NormalizeSplit(string split)
        {
            var sb = new StringBuilder();
            foreach (byte b in Encoding.UTF8.GetBytes(split))
            {
                char r = (char)b;
                if (r == 0x00ad)
                    r = (char)0x0143;
                else if (r <= 0x0020)
                    r = (char)(r + 0x0100);
                else if (r >= 0x007f && r <= 0x00a0)
                    r = (char)(r + 0x00a2);
                sb.Append(r);
            }
            return sb.ToString();
        }

        private List<int> BpeMerge(
            string normalized,
            bool rawUnicodeCodePoints = false,
            bool byteFallback = false)
        {
            var runes = rawUnicodeCodePoints
                ? normalized.EnumerateRunes().Select(r => r.ToString()).ToList()
                : normalized.ToCharArray().Select(c => c.ToString()).ToList();
            if (runes.Count == 0) return new List<int>();
            if (runes.Count == 1)
            {
                if (_vocabLookup.TryGetValue(runes[0], out int id))
                    return new List<int> { id };
                return byteFallback ? ByteFallback(runes[0]) : new List<int>();
            }

            var mergeNodes = new List<MergeNode>();
            for (int i = 0; i < runes.Count; i++)
            {
                mergeNodes.Add(new MergeNode
                {
                    Runes = runes[i],
                    Prev = i - 1,
                    Next = i + 1,
                    Active = true
                });
            }

            var pq = new SortedSet<(int rank, int a, int b)>();
            for (int i = 0; i < runes.Count - 1; i++)
            {
                int rank = GetMergeRank(mergeNodes[i].Runes, mergeNodes[i + 1].Runes);
                if (rank >= 0)
                    pq.Add((rank, i, i + 1));
            }

            while (pq.Count > 0)
            {
                var best = pq.Min;
                pq.Remove(best);

                int a = best.a, b = best.b;
                if (!mergeNodes[a].Active || !mergeNodes[b].Active)
                    continue;

                // A neighboring merge can change the text stored in an active
                // node while an older candidate for this node pair remains in
                // the queue. Only apply a candidate if the nodes are still
                // adjacent and their current pair still has the queued rank.
                // Otherwise the stale rank can incorrectly jump ahead of a
                // newly-created, higher-priority merge.
                if (mergeNodes[a].Next != b || mergeNodes[b].Prev != a ||
                    GetMergeRank(mergeNodes[a].Runes, mergeNodes[b].Runes) != best.rank)
                    continue;

                string merged = mergeNodes[a].Runes + mergeNodes[b].Runes;
                if (!_vocabLookup.ContainsKey(merged))
                    continue;

                mergeNodes[a] = new MergeNode
                {
                    Runes = merged,
                    Prev = mergeNodes[a].Prev,
                    Next = mergeNodes[b].Next,
                    Active = true
                };

                mergeNodes[b] = new MergeNode { Active = false };

                if (mergeNodes[a].Next < mergeNodes.Count)
                {
                    var nextNode = mergeNodes[mergeNodes[a].Next];
                    nextNode.Prev = a;
                    mergeNodes[mergeNodes[a].Next] = nextNode;
                }

                int prevIdx = mergeNodes[a].Prev;
                if (prevIdx >= 0 && mergeNodes[prevIdx].Active)
                {
                    int rank = GetMergeRank(mergeNodes[prevIdx].Runes, mergeNodes[a].Runes);
                    if (rank >= 0)
                        pq.Add((rank, prevIdx, a));
                }

                int nextIdx = mergeNodes[a].Next;
                if (nextIdx < mergeNodes.Count && mergeNodes[nextIdx].Active)
                {
                    int rank = GetMergeRank(mergeNodes[a].Runes, mergeNodes[nextIdx].Runes);
                    if (rank >= 0)
                        pq.Add((rank, a, nextIdx));
                }
            }

            var result = new List<int>();
            foreach (var node in mergeNodes)
            {
                if (!node.Active)
                    continue;

                if (_vocabLookup.TryGetValue(node.Runes, out int id))
                    result.Add(id);
                else if (byteFallback)
                    result.AddRange(ByteFallback(node.Runes));
            }
            return result;
        }

        private List<int> ByteFallback(string text)
        {
            var result = new List<int>();
            foreach (byte b in Encoding.UTF8.GetBytes(text))
            {
                string byteToken = $"<0x{b:X2}>";
                if (_vocabLookup.TryGetValue(byteToken, out int id))
                    result.Add(id);
            }
            return result;
        }

        private int GetMergeRank(string left, string right)
        {
            string key = left + " " + right;
            return _mergeLookup.TryGetValue(key, out int rank) ? rank : -1;
        }

        public void AppendTokenBytes(int tokenId, List<byte> buffer)
        {
            string token = _vocab[tokenId];

            if (_spmStyleBpe)
            {
                if (token.Length == 6 &&
                    token.StartsWith("<0x", StringComparison.Ordinal) &&
                    token.EndsWith(">", StringComparison.Ordinal) &&
                    byte.TryParse(
                        token.Substring(3, 2),
                        System.Globalization.NumberStyles.HexNumber,
                        null,
                        out byte byteValue))
                {
                    buffer.Add(byteValue);
                    return;
                }

                token = token.Replace("\u2581", " ", StringComparison.Ordinal);
                buffer.AddRange(Encoding.UTF8.GetBytes(token));
                return;
            }

            foreach (char r in token)
            {
                if (r == 0x0100) continue;
                if (r > 0x0100 && r <= 0x0120) { buffer.Add((byte)(r - 0x0100)); continue; }
                if (r >= 0x0121 && r <= 0x0142) { buffer.Add((byte)(r - 0x00A2)); continue; }
                if (r == 0x0143) { buffer.Add(0xAD); continue; }
                if (r == 0x2581) { buffer.Add(0x20); continue; }
                if (r < 0x100) { buffer.Add((byte)r); continue; }
                foreach (byte b in Encoding.UTF8.GetBytes(new[] { r }))
                    buffer.Add(b);
            }
        }

        public string Decode(List<int> ids)
        {
            var bytes = new List<byte>();
            foreach (int id in ids)
                AppendTokenBytes(id, bytes);
            return Encoding.UTF8.GetString(bytes.ToArray());
        }

        public bool IsEos(int tokenId) => _eosTokenIds.Contains(tokenId);

        /// <summary>
        /// Look up a token string (e.g. "&lt;|image_pad|&gt;") and return its token ID, or -1 if not found.
        /// </summary>
        public int LookupToken(string tokenStr)
        {
            return _vocabLookup.TryGetValue(tokenStr, out int id) ? id : -1;
        }

        private struct MergeNode
        {
            public string Runes;
            public int Prev, Next;
            public bool Active;
        }
    }
}
