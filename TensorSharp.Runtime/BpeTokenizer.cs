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
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace TensorSharp.Runtime
{
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

    public class BpeTokenizer : ITokenizer
    {
        private readonly string[] _vocab;
        private readonly int[] _tokenTypes;
        private readonly Dictionary<string, int> _vocabLookup;
        private readonly Dictionary<string, int> _mergeLookup;
        private readonly Regex _pretokenizerRegex;
        private readonly int _bosTokenId;
        private readonly int[] _eosTokenIds;
        private readonly bool _addBos;
        private readonly bool _addEos;

        public string[] Vocab => _vocab;
        public int BosTokenId => _bosTokenId;
        public int[] EosTokenIds => _eosTokenIds;
        public int VocabSize => _vocab.Length;

        public BpeTokenizer(string[] vocab, int[] tokenTypes, string[] merges,
            int bosTokenId, int[] eosTokenIds, bool addBos, bool addEos,
            string preTokenizerType = null)
        {
            _vocab = vocab;
            _tokenTypes = tokenTypes;
            _bosTokenId = bosTokenId;
            _eosTokenIds = eosTokenIds;
            _addBos = addBos;
            _addEos = addEos;

            _vocabLookup = new Dictionary<string, int>(vocab.Length);
            for (int i = 0; i < vocab.Length; i++)
                _vocabLookup[vocab[i]] = i;

            _mergeLookup = new Dictionary<string, int>(merges.Length);
            for (int i = 0; i < merges.Length; i++)
                _mergeLookup[merges[i]] = i;

            string pattern = preTokenizerType switch
            {
                "gpt-4o" =>
                    @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|" +
                    @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|" +
                    @"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                _ =>
                    @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            };

            _pretokenizerRegex = new Regex(pattern, RegexOptions.Compiled);
        }

        private List<string> GetSpecialTokens()
        {
            const int TOKEN_TYPE_CONTROL = 3;
            const int TOKEN_TYPE_USER_DEFINED = 4;
            var specials = new List<string>();
            for (int i = 0; i < _vocab.Length; i++)
            {
                if (_tokenTypes != null && i < _tokenTypes.Length &&
                    (_tokenTypes[i] == TOKEN_TYPE_CONTROL || _tokenTypes[i] == TOKEN_TYPE_USER_DEFINED))
                {
                    specials.Add(_vocab[i]);
                }
            }
            return specials;
        }

        public List<int> Encode(string text, bool addSpecial = true)
        {
            var specials = GetSpecialTokens();
            var fragments = new List<(string text, List<int> ids)>();
            fragments.Add((text, null));

            foreach (var special in specials)
            {
                int id = _vocabLookup.TryGetValue(special, out var sid) ? sid : -1;
                if (id < 0) continue;

                var newFragments = new List<(string text, List<int> ids)>();
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

                var matches = _pretokenizerRegex.Matches(frag.text);
                foreach (Match match in matches)
                {
                    string split = match.Value;
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

        private List<int> BpeMerge(string normalized)
        {
            var runes = normalized.ToCharArray().Select(c => c.ToString()).ToList();
            if (runes.Count == 0) return new List<int>();
            if (runes.Count == 1)
            {
                if (_vocabLookup.TryGetValue(runes[0], out int id))
                    return new List<int> { id };
                return new List<int>();
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

                string merged = mergeNodes[a].Runes + mergeNodes[b].Runes;
                if (merged != mergeNodes[a].Runes + mergeNodes[b].Runes)
                    continue;

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
                if (node.Active && _vocabLookup.TryGetValue(node.Runes, out int id))
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

