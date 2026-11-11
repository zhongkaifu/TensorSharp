// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3-Image packed-sequence layout: [text | reference images | target image].
//
// The H3 DiT without its audio stream, denoising one latent frame. RoPE positions
// follow the video layout (MiniMaxH3Layout): text token i at (i, 0, 0), each
// reference image one unit of the timeline after the text with its own aspect-
// normalized spatial grid, and the target after the last reference. Text and the
// target run at the current timestep t = 1 - sigma (no flow shift); reference rows
// sit near clean, at max(t, 0.999).
using System;
using System.Collections.Generic;

namespace TensorSharp.Models.MiniMaxH3
{
    internal static class MiniMaxH3ImageLayout
    {
        /// <param name="textLength">Text tokens (the Qwen3-VL hidden states).</param>
        /// <param name="latentHeight">Target latent rows (pixels / 16), a multiple of 2.</param>
        /// <param name="latentWidth">Target latent columns, a multiple of 2.</param>
        /// <param name="sigma">The step's sigma.</param>
        /// <param name="textTokenModalities">Per text token AdaLN modality: 1 for text, 0 for
        /// the vision blocks of an edit's pictures (with their start/end markers); null is all text.</param>
        /// <param name="references">Each reference image's latent (height, width).</param>
        public static MiniMaxH3PackedLayout Build(
            int textLength, int latentHeight, int latentWidth, float sigma,
            IReadOnlyList<int> textTokenModalities = null,
            IReadOnlyList<(int LatentHeight, int LatentWidth)> references = null)
        {
            if (textLength < 0) throw new ArgumentOutOfRangeException(nameof(textLength));
            if (latentHeight <= 0 || latentWidth <= 0 ||
                latentHeight % MiniMaxH3Geometry.PatchH != 0 || latentWidth % MiniMaxH3Geometry.PatchW != 0)
                throw new ArgumentException("the target latent must be a positive multiple of the 2x2 patch.");

            float t = 1f - sigma;
            var timesteps = new List<float> { t };
            const int targetRow = 0;
            int referenceRow = -1;
            if (references is { Count: > 0 })
            {
                float rt = Math.Max(t, MiniMaxH3Scheduler.VisualConditionTimestep);
                referenceRow = rt.Equals(t) ? targetRow : 1;
                if (referenceRow == 1) timesteps.Add(rt);
            }

            var positions = new List<float>();
            var spans = new List<MiniMaxH3ModulationSpan>();
            var segments = new List<MiniMaxH3SequenceSegment>();
            void Append(float pt, float ph, float pw) { positions.Add(pt); positions.Add(ph); positions.Add(pw); }

            // --- text, one span per run of equal modality ---------------------
            for (int i = 0; i < textLength; i++) Append(i, 0f, 0f);
            if (textLength > 0)
            {
                segments.Add(new MiniMaxH3SequenceSegment(0, textLength, MiniMaxH3SegmentKind.Text));
                int Tag(int i) => textTokenModalities is { Count: > 0 } ? textTokenModalities[i] : MiniMaxH3Modality.Text;
                int runStart = 0, current = Tag(0);
                for (int i = 1; i <= textLength; i++)
                {
                    if (i < textLength && Tag(i) == current) continue;
                    spans.Add(new MiniMaxH3ModulationSpan(runStart, i, targetRow * MiniMaxH3Modality.Count + current));
                    if (i < textLength) { runStart = i; current = Tag(i); }
                }
            }

            int row = textLength;
            float cursor = textLength;

            // --- reference images: one timeline unit each ---------------------
            if (references is { Count: > 0 })
            {
                for (int index = 0; index < references.Count; index++)
                {
                    var (rh, rw) = references[index];
                    if (rh <= 0 || rw <= 0 || rh % MiniMaxH3Geometry.PatchH != 0 || rw % MiniMaxH3Geometry.PatchW != 0)
                        throw new ArgumentException($"reference {index + 1}'s latent must be a positive multiple of the 2x2 patch.");
                    var (refH, refW) = MiniMaxH3Layout.SpatialAxes(rh, rw);
                    int count = refH.Length * refW.Length;
                    foreach (float h in refH)
                        foreach (float w in refW)
                            Append(cursor, h, w);
                    segments.Add(new MiniMaxH3SequenceSegment(row, row + count, MiniMaxH3SegmentKind.ConditionVideo, index));
                    spans.Add(new MiniMaxH3ModulationSpan(row, row + count,
                        referenceRow * MiniMaxH3Modality.Count + MiniMaxH3Modality.Visual));
                    row += count;
                    cursor += 1f;
                }
            }

            // --- target image -------------------------------------------------
            var (hAxis, wAxis) = MiniMaxH3Layout.SpatialAxes(latentHeight, latentWidth);
            int start = row;
            foreach (float h in hAxis)
                foreach (float w in wAxis)
                    Append(cursor, h, w);
            int tokens = hAxis.Length * wAxis.Length;
            segments.Add(new MiniMaxH3SequenceSegment(start, start + tokens, MiniMaxH3SegmentKind.TargetVideo));
            spans.Add(new MiniMaxH3ModulationSpan(start, start + tokens,
                targetRow * MiniMaxH3Modality.Count + MiniMaxH3Modality.Visual));
            row += tokens;

            return new MiniMaxH3PackedLayout
            {
                Positions = positions.ToArray(),
                ModulationSpans = spans,
                Segments = segments,
                Timesteps = timesteps.ToArray(),
                TokenCount = row,
                AudioSpan = new MiniMaxH3ModulationSpan(start, start, targetRow),
                VideoSpan = new MiniMaxH3ModulationSpan(start, start + tokens, targetRow),
            };
        }
    }
}
