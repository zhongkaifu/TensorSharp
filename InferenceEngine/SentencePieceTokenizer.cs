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

namespace InferenceEngine
{
    /// <summary>
    /// SentencePiece unigram tokenizer matching ollama's implementation.
    /// Uses score-based merging: merges the pair with the highest vocabulary score.
    /// Spaces are replaced with ▁ (U+2581) before tokenization.
    /// Falls back to byte tokens (&lt;0xNN&gt;) for unknown characters.
    /// </summary>
    public class SentencePieceTokenizer : ITokenizer
    {
        private const string SpaceReplacement = "▁";

        private readonly string[] _vocab;
        private readonly int[] _tokenTypes;
        private readonly float[] _scores;
        private readonly Dictionary<string, int> _vocabLookup;
        private readonly int _bosTokenId;
        private readonly int[] _eosTokenIds;
        private readonly bool _addBos;
        private readonly bool _addEos;
        private readonly List<string> _specialTokens;

        public string[] Vocab => _vocab;
        public int BosTokenId => _bosTokenId;
        public int[] EosTokenIds => _eosTokenIds;
        public int VocabSize => _vocab.Length;

        public SentencePieceTokenizer(string[] vocab, int[] tokenTypes, float[] scores,
            int bosTokenId, int[] eosTokenIds, bool addBos, bool addEos)
        {
            _vocab = vocab;
            _tokenTypes = tokenTypes;
            _scores = scores;
            _bosTokenId = bosTokenId;
            _eosTokenIds = eosTokenIds;
            _addBos = addBos;
            _addEos = addEos;

            _vocabLookup = new Dictionary<string, int>(vocab.Length);
            for (int i = 0; i < vocab.Length; i++)
                _vocabLookup[vocab[i]] = i;

            const int TOKEN_TYPE_CONTROL = 3;
            const int TOKEN_TYPE_USER_DEFINED = 4;
            _specialTokens = new List<string>();
            for (int i = 0; i < vocab.Length; i++)
            {
                if (tokenTypes != null && i < tokenTypes.Length &&
                    (tokenTypes[i] == TOKEN_TYPE_CONTROL || tokenTypes[i] == TOKEN_TYPE_USER_DEFINED))
                {
                    _specialTokens.Add(vocab[i]);
                }
            }
        }

        public List<int> Encode(string text, bool addSpecial = true)
        {
            var fragments = new List<(string text, List<int> ids)>();
            fragments.Add((text, null));

            foreach (var special in _specialTokens)
            {
                if (!_vocabLookup.TryGetValue(special, out int specialId))
                    continue;

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
                        newFragments.Add((special, new List<int> { specialId }));
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

                string spText = frag.text.Replace(" ", SpaceReplacement);

                if (_vocabLookup.TryGetValue(spText, out int directId))
                {
                    ids.Add(directId);
                    continue;
                }

                ids.AddRange(TokenizePiece(spText));
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
        /// Tokenize a text fragment using score-based merging (SentencePiece unigram-style).
        /// Starts with individual runes, greedily merges pairs with highest vocabulary score.
        /// </summary>
        private List<int> TokenizePiece(string text)
        {
            var runes = new List<string>();
            foreach (var r in text.EnumerateRunes())
                runes.Add(r.ToString());

            if (runes.Count == 0) return new List<int>();
            if (runes.Count == 1)
            {
                if (_vocabLookup.TryGetValue(runes[0], out int id))
                    return new List<int> { id };
                return ByteFallback(runes[0]);
            }

            var nodes = new MergeNode[runes.Count];
            for (int i = 0; i < runes.Count; i++)
            {
                nodes[i] = new MergeNode
                {
                    Text = runes[i],
                    Prev = i - 1,
                    Next = i + 1,
                    Active = true
                };
            }

            var pq = new SortedSet<(float score, int a, int b)>(new MergeCandidateComparer());

            for (int i = 0; i < runes.Count - 1; i++)
            {
                TryAddMerge(pq, nodes, i, i + 1);
            }

            while (pq.Count > 0)
            {
                var best = pq.Max;
                pq.Remove(best);

                int a = best.a, b = best.b;
                if (!nodes[a].Active || !nodes[b].Active)
                    continue;

                string merged = nodes[a].Text + nodes[b].Text;
                if (merged.Length != nodes[a].Text.Length + nodes[b].Text.Length)
                    continue;

                nodes[a].Text = merged;
                nodes[a].Next = nodes[b].Next;
                nodes[b].Active = false;

                if (nodes[a].Next < nodes.Length)
                    nodes[nodes[a].Next].Prev = a;

                if (nodes[a].Prev >= 0 && nodes[nodes[a].Prev].Active)
                    TryAddMerge(pq, nodes, nodes[a].Prev, a);

                if (nodes[a].Next < nodes.Length && nodes[nodes[a].Next].Active)
                    TryAddMerge(pq, nodes, a, nodes[a].Next);
            }

            var result = new List<int>();
            foreach (var node in nodes)
            {
                if (!node.Active || string.IsNullOrEmpty(node.Text))
                    continue;

                if (_vocabLookup.TryGetValue(node.Text, out int id))
                {
                    result.Add(id);
                }
                else
                {
                    result.AddRange(ByteFallback(node.Text));
                }
            }
            return result;
        }

        private void TryAddMerge(SortedSet<(float score, int a, int b)> pq,
            MergeNode[] nodes, int a, int b)
        {
            string combined = nodes[a].Text + nodes[b].Text;
            if (_vocabLookup.TryGetValue(combined, out int id))
            {
                float score = (id < _scores.Length) ? _scores[id] : 0f;
                pq.Add((score, a, b));
            }
        }

        private List<int> ByteFallback(string token)
        {
            var result = new List<int>();
            foreach (byte b in Encoding.UTF8.GetBytes(token))
            {
                string byteToken = $"<0x{b:X2}>";
                if (_vocabLookup.TryGetValue(byteToken, out int id))
                    result.Add(id);
            }
            return result;
        }

        public void AppendTokenBytes(int tokenId, List<byte> buffer)
        {
            string token = _vocab[tokenId];
            if (token.Length == 6 && token.StartsWith("<0x") && token.EndsWith(">"))
            {
                if (byte.TryParse(token.Substring(3, 2),
                    System.Globalization.NumberStyles.HexNumber, null, out byte byteVal))
                {
                    buffer.Add(byteVal);
                    return;
                }
            }
            token = token.Replace(SpaceReplacement, " ");
            foreach (byte b in Encoding.UTF8.GetBytes(token))
                buffer.Add(b);
        }

        public string Decode(List<int> ids)
        {
            var sb = new StringBuilder();
            var pendingBytes = new List<byte>();
            foreach (int id in ids)
            {
                string token = _vocab[id];

                if (token.Length == 6 && token.StartsWith("<0x") && token.EndsWith(">"))
                {
                    if (byte.TryParse(token.Substring(3, 2),
                        System.Globalization.NumberStyles.HexNumber, null, out byte byteVal))
                    {
                        pendingBytes.Add(byteVal);
                        continue;
                    }
                }

                if (pendingBytes.Count > 0)
                {
                    sb.Append(Encoding.UTF8.GetString(pendingBytes.ToArray()));
                    pendingBytes.Clear();
                }

                token = token.Replace(SpaceReplacement, " ");
                sb.Append(token);
            }
            if (pendingBytes.Count > 0)
                sb.Append(Encoding.UTF8.GetString(pendingBytes.ToArray()));
            return sb.ToString();
        }

        public bool IsEos(int tokenId) => _eosTokenIds.Contains(tokenId);

        public int LookupToken(string tokenStr)
        {
            return _vocabLookup.TryGetValue(tokenStr, out int id) ? id : -1;
        }

        private class MergeNode
        {
            public string Text;
            public int Prev, Next;
            public bool Active;
        }

        private class MergeCandidateComparer : IComparer<(float score, int a, int b)>
        {
            public int Compare((float score, int a, int b) x, (float score, int a, int b) y)
            {
                int c = x.score.CompareTo(y.score);
                if (c != 0) return c;
                c = y.a.CompareTo(x.a);
                if (c != 0) return c;
                return y.b.CompareTo(x.b);
            }
        }
    }
}
