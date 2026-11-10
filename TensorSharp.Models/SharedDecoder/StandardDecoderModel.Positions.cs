// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Models
{
    /// <summary>
    /// Rotary positions, once for every family: the KV index of a row, or, for a family with a
    /// multi-axis rotary (Qwen-VL's interleaved M-RoPE), the (T, H, W) rows of a position table the
    /// prompt expander stages for the next forward. A media span occupies more KV rows than rotary
    /// positions, so past a table every row of the sequence rotates at KV index + a per-sequence
    /// delta, which the slot carries (and checkpoints keep).
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        /// <summary>Rotated pairs per axis (T, H, W, extra) of a multi-axis rotary (ggml's interleaved
        /// M-RoPE), or null for scalar positions.</summary>
        protected virtual int[] RotarySections => null;

        // The active sequence's rotary delta (see Slot.RopeDelta) and the table staged for the next forward.
        private int _ropeDelta;
        private int[] _pendingPositionTable;

        /// <summary>The active sequence's rotary delta (tests, diagnostics).</summary>
        internal int ActiveRopeDelta => _ropeDelta;

        /// <summary>Stage the flat (T, H, W) rotary positions of the next forward's rows.</summary>
        protected void StagePositionTable(int[] flatThw)
        {
            if (RotarySections == null)
                throw new NotSupportedException($"{Config.Architecture} has no multi-axis rotary.");
            _pendingPositionTable = flatThw;
        }

        /// <summary>
        /// The rotary positions of a forward of <paramref name="rows"/> rows at KV index
        /// <paramref name="start"/>, settling the sequence's delta first: with a staged table the delta
        /// becomes the one that holds after these rows (the table drives them, unless every row is
        /// text at KV index + delta: the scalar form is the same rotation); without one, a forward
        /// from index 0 starts a new history (delta 0) and any other continues the delta.
        /// </summary>
        private TokenPositions BeginRotaryPositions(int start, int rows)
        {
            int[] table = _pendingPositionTable;
            _pendingPositionTable = null;
            if (table != null && rows > 0 && table.Length >= 3 * rows)
            {
                _ropeDelta = Qwen35RopePositions.DeltaAfterRows(table, rows, start + rows);
                if (table.Length > 3 * rows)
                    _pendingPositionTable = table[(3 * rows)..];   // the rest of a chunked prompt
                if (!Qwen35RopePositions.IsScalarContinuation(table, rows, start, _ropeDelta))
                    return TokenPositions.Contiguous(start, rows).WithRotary(0, table.Length == 3 * rows ? table : table[..(3 * rows)]);
            }
            else if (start == 0)
            {
                _ropeDelta = 0;
            }
            return TokenPositions.Contiguous(start, rows).WithRotary(_ropeDelta, null);
        }

        /// <summary>ggml_rope_multi's positions for a table: [4 * rows], one block per axis (the extra axis 0).</summary>
        private static int[] AxisMajorPositions(int[] flatThw, int rows)
        {
            var result = new int[4 * rows];
            for (int i = 0; i < rows; i++)
            {
                result[i] = flatThw[3 * i];
                result[rows + i] = flatThw[3 * i + 1];
                result[2 * rows + i] = flatThw[3 * i + 2];
            }
            return result;
        }
    }
}
