// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.Graph;

namespace TensorSharp.Models
{
    /// <summary>
    /// Qwen3-VL in whole-forward graphs: per-head QK-norm before the NeoX rotary (the step's rope is
    /// the interleaved M-RoPE over a staged position table), and DeepStack.
    ///
    /// <para>DeepStack: the vision tower projects blocks 8, 16 and 24 as well as its last one, and
    /// the language model adds tap i to the residual stream at the media rows after decoder layer i
    /// (Hugging Face <c>Qwen3VLMoeTextModel._deepstack_process</c>, llama.cpp's qwen3vl-moe graph).
    /// The tower's rows arrive concatenated, <c>[main | tap 0 | tap 1 | tap 2]</c>; the main part is
    /// the runtime's media row, the taps wait here, anchored to the KV index of the forward that
    /// injects their media (<see cref="OnMediaInjected"/>). A step adds the taps of the rows it holds
    /// as one per-run input per tap layer (zeros on text rows). Only graphs add them, so a step that
    /// cannot run as a graph while taps are pending fails instead of answering without them.</para>
    /// </summary>
    public sealed partial class Qwen3VLModel
    {
        /// <summary>One media span's DeepStack taps: rows [Offset, Offset + Rows) of the forward that injects it (then
        /// rows [Start, Start + Rows) of the sequence once anchored), and its taps, each [Rows, hidden] row-major.</summary>
        private sealed class DeepStackSpan
        {
            public int Offset;
            public int Start = -1;
            public int Rows;
            public float[][] Taps;
            public object Slot;
        }

        private readonly List<DeepStackSpan> _deepStack = new();

        protected override RotarySpec GraphRotary => _rotary;

        /// <summary>DeepStack taps exist only in graphs: while any wait, the per-op layers must not run instead.</summary>
        protected override bool RequiresGraphs => _deepStack.Count > 0;

        protected override void EmitQueryKey(GraphStep step, int layer, ref int q, ref int k)
        {
            q = step.Rope(step.RmsNorm(q, LayerName(layer, "attn_q_norm.weight")), _rotary);
            k = step.Rope(step.RmsNorm(k, LayerName(layer, "attn_k_norm.weight")), _rotary);
        }

        /// <summary>The residual add, then (after the feed-forward of a DeepStack layer) the tap of the step's media rows.</summary>
        protected override int EmitBlockOutput(GraphStep step, int layer, int stream, BlockSite site, int context, int output)
        {
            int h = base.EmitBlockOutput(step, layer, stream, site, context, output);
            if (site != BlockSite.Ffn || layer >= _deepStackLayers || _deepStack.Count == 0)
                return h;
            float[] tap = DeepStackRows(step, layer);
            return tap == null ? h : step.Op(GraphOp.Add, h, step.InputF32(tap, Config.HiddenSize, step.Rows));
        }

        /// <summary>The forward that injects the queued media rows starts at <paramref name="absoluteStart"/>: the taps queued
        /// with them anchor there, and the taps of an earlier media forward are dropped. The later pieces of a split forward
        /// inject again with nothing new to anchor, and keep the first piece's anchors.</summary>
        protected override void OnMediaInjected(int absoluteStart)
        {
            if (!_deepStack.Exists(s => s.Start < 0)) return;
            _deepStack.RemoveAll(s => s.Start >= 0);
            foreach (DeepStackSpan span in _deepStack)
                span.Start = absoluteStart + span.Offset;
        }

        /// <summary>
        /// Tap <paramref name="layer"/> of the step's rows [hidden, rows] (zeros on rows without media), or null when no anchored span
        /// overlaps them. A span binds to the first step's sequence; a decode step, or one past a span, retires it.
        /// </summary>
        private float[] DeepStackRows(GraphStep step, int layer)
        {
            if (step.IsDecode)
            {
                _deepStack.Clear();
                return null;
            }
            float[] rows = null;
            int hidden = Config.HiddenSize;
            for (int s = 0; s < step.Sequences; s++)
            {
                int start = step.Starts[s], count = step.Counts[s], first = step.First[s];
                object slot = step.Slots[s];
                foreach (DeepStackSpan span in _deepStack)
                {
                    if (span.Start < 0) continue;
                    if (span.Slot != null && !ReferenceEquals(span.Slot, slot)) continue;
                    int from = Math.Max(span.Start, start), to = Math.Min(span.Start + span.Rows, start + count);
                    if (from >= to) continue;
                    span.Slot ??= slot;
                    rows ??= new float[(long)hidden * step.Rows];
                    Array.Copy(span.Taps[layer], (long)(from - span.Start) * hidden,
                        rows, (long)(first + from - start) * hidden, (long)(to - from) * hidden);
                }
            }
            // A span the step has moved past (its sequence's rows start at or after its end) is done.
            if (step.Sequences == 1)
            {
                int start = step.Starts[0];
                object slot = step.Slots[0];
                if (layer == _deepStackLayers - 1)
                    _deepStack.RemoveAll(span => span.Start >= 0 && ReferenceEquals(span.Slot, slot) && span.Start + span.Rows <= start);
            }
            return rows;
        }

        /// <summary>Queue a span's taps beside its media rows (see <see cref="SetVisionEmbeddings"/>).</summary>
        private void QueueDeepStack(int insertPosition, int rows, float[][] taps)
            => _deepStack.Add(new DeepStackSpan { Offset = insertPosition, Rows = rows, Taps = taps });

        /// <summary>A reset ends every sequence the anchored taps belonged to. Taps not yet anchored wait with their media
        /// rows, which a reset does not drop either.</summary>
        protected override void ResetKVCacheCore()
        {
            _deepStack.RemoveAll(s => s.Start >= 0);
            base.ResetKVCacheCore();
        }
    }
}
