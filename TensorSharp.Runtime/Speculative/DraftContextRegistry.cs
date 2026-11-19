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
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Runtime.Speculative
{
    /// <summary>A drafter state a model kept past its request (<see cref="IPerRequestDraftState.KeepDraftState"/>): the request
    /// it was kept for, the tokens of the positions it covers, the request's media and cache scope, and when it was kept.</summary>
    internal sealed record KeptDraftContext(string Key, int[] Tokens, IReadOnlyList<PromptMediaSpan> Media, string? Scope, long Serial);

    /// <summary>
    /// The drafter contexts a model keeps past their requests, by the tokens each covers: a new request continues the one
    /// that shares the longest prefix with its tokens (<see cref="IPerRequestDraftState.ContinueDraftState"/>). The runtime
    /// matches tokens; the model keeps the state, and gives a kept one up when it needs the storage (a continuation of it
    /// then fails and the entry goes). Holds at most <see cref="Capacity"/> entries, the oldest dropped first.
    /// </summary>
    internal sealed class DraftContextRegistry
    {
        internal const int Capacity = 32;

        private readonly List<KeptDraftContext> _kept = new();
        private long _serial;

        public int Count => _kept.Count;

        /// <summary>Records the context kept for <paramref name="key"/>: <paramref name="tokens"/> are its sequence's first
        /// positions (a copy is taken); a previous entry under the same key is replaced. Returns the key of an entry dropped
        /// to stay within <see cref="Capacity"/>, or null.</summary>
        public string? Note(string key, IReadOnlyList<int> tokens, int count, IReadOnlyList<PromptMediaSpan> media, string? scope)
        {
            ArgumentNullException.ThrowIfNull(key);
            Forget(key);
            count = Math.Clamp(count, 0, tokens.Count);
            var copy = new int[count];
            for (int i = 0; i < count; i++) copy[i] = tokens[i];
            _kept.Add(new KeptDraftContext(key, copy, media ?? Array.Empty<PromptMediaSpan>(), scope, ++_serial));
            if (_kept.Count <= Capacity) return null;
            string dropped = _kept[0].Key;
            _kept.RemoveAt(0);
            return dropped;
        }

        public bool Forget(string key)
        {
            for (int i = 0; i < _kept.Count; i++)
                if (string.Equals(_kept[i].Key, key, StringComparison.Ordinal))
                {
                    _kept.RemoveAt(i);
                    return true;
                }
            return false;
        }

        /// <summary>The kept contexts a request may continue, the longest usable prefix first (the most recently kept on a
        /// tie), each with that prefix: <paramref name="usable"/> gives it (0: not usable).</summary>
        public List<(string Key, int Tokens)> Candidates(Func<KeptDraftContext, int> usable)
        {
            var found = new List<(string Key, int Tokens, long Serial)>();
            foreach (KeptDraftContext kept in _kept)
            {
                int tokens = usable(kept);
                if (tokens > 0) found.Add((kept.Key, tokens, kept.Serial));
            }
            found.Sort((a, b) => a.Tokens != b.Tokens ? b.Tokens.CompareTo(a.Tokens) : b.Serial.CompareTo(a.Serial));
            return found.ConvertAll(c => (c.Key, c.Tokens));
        }

        /// <summary>How many of <paramref name="kept"/>'s tokens <paramref name="tokens"/> (the first <paramref name="known"/>
        /// of a sequence) begin with.</summary>
        public static int CommonPrefix(KeptDraftContext kept, Func<int, int> tokens, int known)
        {
            int limit = Math.Min(kept.Tokens.Length, known), n = 0;
            while (n < limit && kept.Tokens[n] == tokens(n)) n++;
            return n;
        }
    }
}
