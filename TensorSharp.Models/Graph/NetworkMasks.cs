// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Threading.Tasks;

namespace TensorSharp.Models.Graph
{
    /// <summary>
    /// Attention masks for <see cref="NetworkBuilder.Attention"/>: F16 bits [keys, rows] (row-major by query row), 0 where a
    /// query sees a key and -inf (0xFC00) where it does not. Keys and queries carry absolute positions: query row q sits at
    /// <c>queryStart + q</c>, key j at <c>keyStart + j</c> (a stream's carried keys come first, at negative or earlier
    /// positions); a key at a negative position is never seen (an empty slot of a cold stream's state).
    /// </summary>
    public static class NetworkMasks
    {
        public const ushort Open = 0, Closed = 0xFC00;

        /// <summary>Every row sees every key at a non-negative position.</summary>
        public static ushort[] Full(int rows, int keys, int keyStart = 0)
            => Build(rows, keys, (q, kp) => kp >= 0, 0, keyStart);

        /// <summary>Causal: query at position p sees keys at positions 0 .. p.</summary>
        public static ushort[] Causal(int rows, int keys, int queryStart, int keyStart = 0)
            => Build(rows, keys, (qp, kp) => kp >= 0 && kp <= qp, queryStart, keyStart);

        /// <summary>Causal sliding window of <paramref name="window"/> keys including the query's own: query at position p sees
        /// keys at positions max(0, p - window + 1) .. p (LFM3's audio encoder: 32; its codec: 30).</summary>
        public static ushort[] SlidingWindow(int rows, int keys, int window, int queryStart, int keyStart = 0)
        {
            if (window < 1) throw new ArgumentOutOfRangeException(nameof(window));
            return Build(rows, keys, (qp, kp) => kp >= 0 && kp <= qp && kp > qp - window, queryStart, keyStart);
        }

        /// <summary>Bidirectional within segments: rows (and keys: the same sequence) split into consecutive segments of
        /// <paramref name="lengths"/>; a row sees every key of its own segment (frames of a batch, each its own image).</summary>
        public static ushort[] Segments(params int[] lengths)
        {
            int n = 0;
            foreach (int l in lengths) n += l;
            var segment = new int[n];
            for (int s = 0, at = 0; s < lengths.Length; s++)
                for (int i = 0; i < lengths[s]; i++) segment[at++] = s;
            return Build(n, n, (q, k) => segment[q] == segment[k], 0, 0);
        }

        /// <summary>A mask of <paramref name="rows"/> x <paramref name="keys"/> where row q sees key j iff
        /// <paramref name="sees"/>(queryStart + q, keyStart + j).</summary>
        public static ushort[] Build(int rows, int keys, Func<int, int, bool> sees, int queryStart, int keyStart)
        {
            ArgumentNullException.ThrowIfNull(sees);
            var bits = new ushort[(long)rows * keys];
            void Row(int q)
            {
                long at = (long)q * keys;
                for (int j = 0; j < keys; j++) bits[at + j] = sees(queryStart + q, keyStart + j) ? Open : Closed;
            }
            if ((long)rows * keys >= 1 << 20) Parallel.For(0, rows, Row);
            else for (int q = 0; q < rows; q++) Row(q);
            return bits;
        }
    }
}
