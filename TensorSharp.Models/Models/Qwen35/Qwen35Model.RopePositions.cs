// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen-VL M-RoPE position delta for Qwen 3.5 / 3.6.
//
// A prompt image whose merged grid is H x W occupies H*W KV rows but only
// max(H, W) rotary positions: the text after it resumes at base + max(H, W)
// (ModelMultimodalInjector.LayoutQwenVLPrompt, the HF / SGLang get_rope_index
// layout). Every token past the prompt's position table therefore sits at
//
//     rope position = KV index + delta,   delta = max(last table row) + 1 - prompt length
//
// which is SGLang's mrope_position_delta (decode: seq_len - 1 + delta). The delta
// is per sequence: it lives with the active cache fields, is swapped with each
// per-request holder, is copied with a checkpoint and written into a checkpoint
// file, and is reset with the cache. Before this, decode and the fused verify
// used the KV index itself as the position, so the reply to an image was produced
// at positions a re-prefill of the same history would not use, and a follow-up
// turn could not reuse that cache exactly.
using System;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>The position arithmetic of Qwen-VL M-RoPE, free of model state so the
    /// reference-position tests can drive it directly.</summary>
    internal static class Qwen35RopePositions
    {
        /// <summary>The delta that holds after a forward of <paramref name="rows"/> table
        /// rows ending at KV index <paramref name="endIndex"/> (exclusive): the next
        /// scalar position is one past the largest axis of the last row. The last row of
        /// a prompt is text (T = H = W) or the end of a span (its largest axis is
        /// base + max(H, W) - 1), and positions never decrease across spans, so this is
        /// HF's <c>llm_positions.max() + 1 - len(input_ids)</c>. A chunk that ends inside
        /// a span yields an interim value the next chunk's table replaces.</summary>
        public static int DeltaAfterRows(ReadOnlySpan<int> flatThw, int rows, int endIndex)
        {
            if (rows <= 0 || flatThw.Length < 3 * rows)
                throw new ArgumentException("the position table holds fewer rows than the forward", nameof(flatThw));
            int last = 3 * (rows - 1);
            int max = Math.Max(flatThw[last], Math.Max(flatThw[last + 1], flatThw[last + 2]));
            return checked(max + 1 - endIndex);
        }

        /// <summary>Whether every row is text at the scalar position KV index +
        /// <paramref name="delta"/> (T = H = W = startIndex + i + delta), in which case
        /// the table adds nothing a scalar RoPE with the delta does not already say.</summary>
        public static bool IsScalarContinuation(ReadOnlySpan<int> flatThw, int rows, int startIndex, int delta)
        {
            if (rows <= 0 || flatThw.Length < 3 * rows)
                return false;
            for (int i = 0; i < rows; i++)
            {
                int p = startIndex + i + delta;
                int o = 3 * i;
                if (flatThw[o] != p || flatThw[o + 1] != p || flatThw[o + 2] != p)
                    return false;
            }
            return true;
        }

        /// <summary>The rotary position of the token at KV index <paramref name="index"/>
        /// past the position table.</summary>
        public static int ScalarPosition(int index, int delta) => checked(index + delta);
    }

    public partial class Qwen35Model
    {
        // M-RoPE delta of the ACTIVE sequence (the primary cache or the checked-out
        // holder): rope position = KV index + _ropeDelta for every token the pending
        // position table does not cover. Zero for text-only history.
        private int _ropeDelta;

        /// <summary>The M-RoPE delta of the active sequence (test and diagnostics hook).</summary>
        internal int ActiveRopePositionDelta => _ropeDelta;

        private int RopePosition(int kvIndex) => Qwen35RopePositions.ScalarPosition(kvIndex, _ropeDelta);

        /// <summary>
        /// Settle the rope delta for a forward of <paramref name="seqLen"/> tokens at KV
        /// index <paramref name="startPos"/>, before any kernel reads a position.
        /// <list type="bullet">
        /// <item>With a staged position table, the delta becomes the one that holds after
        /// these rows. The table itself still drives this forward's rows, except when every
        /// row is plain text at KV index + delta: the table is then dropped, so the text
        /// after an image runs the ordinary scalar-RoPE graphs (the same rotation).</item>
        /// <item>Without one, a forward from index 0 starts a new history (delta 0);
        /// any other forward continues the active sequence's delta.</item>
        /// </list>
        /// </summary>
        private void BeginRopePositions(int startPos, int seqLen)
        {
            int[] table = _pendingMRoPEPositions;
            if (table != null && seqLen > 0 && table.Length >= 3 * seqLen)
            {
                int delta = Qwen35RopePositions.DeltaAfterRows(table, seqLen, startPos + seqLen);
                _ropeDelta = delta;
                if (Qwen35RopePositions.IsScalarContinuation(table, seqLen, startPos, delta))
                    _pendingMRoPEPositions = null;
                return;
            }
            if (startPos == 0)
                _ropeDelta = 0;
        }

        // The fused Qwen3.5 graphs take the RoPE position separately from the KV index
        // (TSGgml_Qwen35RopePositionAbi >= 1). A library built before that has entry
        // points with fewer arguments, and calling them would shift every argument, so
        // such a library is refused once, loudly, and the per-op paths run instead.
        private static int _nativeRopeAbi = -1;
        private static bool _nativeRopeAbiWarned;

        private bool NativeRopePositionAbiSupported()
        {
            if (!IsGgmlBackend)
                return false;
            if (_nativeRopeAbi < 0)
                _nativeRopeAbi = GgmlBasicOps.Qwen35RopePositionAbi();
            if (_nativeRopeAbi >= 1)
                return true;
            if (!_nativeRopeAbiWarned)
            {
                _nativeRopeAbiWarned = true;
                Console.Error.WriteLine(
                    "[qwen35] the loaded GgmlOps library predates the M-RoPE position contract " +
                    "(TSGgml_Qwen35RopePositionAbi missing); the fused whole-model graphs are disabled " +
                    "and every forward runs the per-op path. Rebuild TensorSharp.GGML.Native.");
            }
            return false;
        }
    }
}
