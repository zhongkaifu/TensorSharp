// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;

using TensorSharp.Graph;

namespace TensorSharp.Models
{
    public abstract partial class StandardDecoderModel
    {
        /// <summary>
        /// A batched step's per-sequence results [width, count_s] (sequence order) as one [width, rows]: concatenations along the
        /// rows in a balanced tree, so each part is copied about log2 n times. The left fold of one concatenation a sequence had
        /// copied the first sequence's rows n - 1 times: 2,220 concatenations moving O(n^2) bytes, 4.6 ms of a 64-sequence
        /// LFM3-1.7B decode step on ggml-cuda (B0 census, A40). The same rows in the same order, so the same bits.
        /// </summary>
        protected int JoinSequenceRows(GraphStep step, List<int> parts)
        {
            if (parts.Count == 1) return parts[0];
            if (LegacyBatchedJoinForTests)
            {
                int folded = parts[0];
                for (int i = 1; i < parts.Count; i++)
                    folded = step.Op(GraphOp.Concat, stackalloc[] { folded, parts[i] }, stackalloc long[] { 1 });
                return folded;
            }
            var level = parts;
            while (level.Count > 1)
            {
                var next = new List<int>((level.Count + 1) / 2);
                for (int i = 0; i + 1 < level.Count; i += 2)
                    next.Add(step.Op(GraphOp.Concat, stackalloc[] { level[i], level[i + 1] }, stackalloc long[] { 1 }));
                if ((level.Count & 1) == 1) next.Add(level[^1]);
                level = next;
            }
            return level[0];
        }

        /// <summary>Tests: batched steps emit what they emitted before the hoisted short-convolution products and the
        /// balanced joins (one product pair a sequence, a left fold of concatenations), the reference those are bitwise
        /// against.</summary>
        internal bool LegacyBatchedJoinForTests { get; set; }
    }
}
