// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorSharp.Models.QwenImage21
{
    /// <summary>A run of the packed sequence: rows [Start, End) taken from the text (<see cref="IsImage"/> false) or the image
    /// tokens (references, then the target) from <see cref="SourceStart"/>.</summary>
    internal readonly record struct QwenImage21Segment(int Start, int End, int SourceStart, bool IsImage)
    {
        public int Length => End - Start;
    }

    /// <summary>
    /// Qwen-Image-2.1's single-stream sequence for one conditioning and set of references: text runs (causal) and each
    /// reference's latent tokens where the text holds its vision slots (one slot stands for four latent tokens; bidirectional
    /// over everything before its end), then the target's tokens (bidirectional over the whole sequence). Every row's 3-axis
    /// rotary angles (frame, row, column; 16/56/56 of a 128-wide head) are tabled as cos and sin [head/2, rows]. Text and
    /// references form the <see cref="Prefix"/>, which is modulated at t = 0 and never attends the target.
    /// </summary>
    internal sealed class QwenImage21Layout
    {
        private QwenImage21Layout(QwenImage21Segment[] segments, float[] cos, float[] sin, int prefix, int headDim)
        {
            Segments = segments;
            Cos = cos;
            Sin = sin;
            Prefix = prefix;
            HeadDim = headDim;
        }

        public IReadOnlyList<QwenImage21Segment> Segments { get; }
        public float[] Cos { get; }
        public float[] Sin { get; }

        /// <summary>Rows before the target (text and references).</summary>
        public int Prefix { get; }
        public int HeadDim { get; }
        public int Rows => Cos.Length / (HeadDim / 2);
        public QwenImage21Segment Target => Segments[^1];

        /// <summary>The rotary axes' widths of a <paramref name="headDim"/>-wide head: an eighth for the frame, the rest halved
        /// between the row and the column (16, 56, 56 of 128).</summary>
        internal static (int T, int H, int W) Axes(int headDim)
        {
            int t = headDim / 8, h = (headDim - t) / 2;
            if (headDim % 16 != 0 || t % 2 != 0 || h % 2 != 0)
                throw new ArgumentException($"A {headDim}-wide head has no Qwen-Image rotary split.");
            return (t, h, headDim - t - h);
        }

        /// <param name="textLength">The conditioning rows.</param>
        /// <param name="imageSlots">One tag a text row: 0 text, n the n-th reference's vision slots (null: all text).</param>
        /// <param name="shapes">Each reference's latent (height, width), then the target's.</param>
        internal static QwenImage21Layout Build(int textLength, int[] imageSlots, IReadOnlyList<(int Height, int Width)> shapes, int headDim)
        {
            if (textLength <= 0 || shapes == null || shapes.Count == 0 ||
                shapes.Any(s => s.Height <= 0 || s.Width <= 0) || (imageSlots != null && imageSlots.Length != textLength))
                throw new ArgumentException("Invalid Qwen-Image-2.1 token layout.");
            var segments = new List<QwenImage21Segment>();
            var positions = new List<(int T, int H, int W)>();
            int position = 0, nextImage = 0, imageOffset = 0;
            void AppendImage(int index)
            {
                var (height, width) = shapes[index];
                int count = checked(height * width), start = positions.Count;
                segments.Add(new QwenImage21Segment(start, checked(start + count), imageOffset, true));
                for (int h = 0; h < height; ++h)
                    for (int w = 0; w < width; ++w)
                        positions.Add((position, h - (height - height / 2), w - (width - width / 2)));
                position += Math.Max(height, width);
                imageOffset += count;
            }
            for (int i = 0; i < textLength;)
            {
                int begin = i, tag = imageSlots?[i] ?? 0;
                while (i < textLength && (imageSlots?[i] ?? 0) == tag) ++i;
                if (tag != 0)
                {
                    if (tag != nextImage + 1 || nextImage + 1 >= shapes.Count ||
                        (long)(i - begin) * 4 != (long)shapes[nextImage].Height * shapes[nextImage].Width)
                        throw new ArgumentException("Vision slots and reference latents must have matching sizes and ordered image tags.");
                    AppendImage(nextImage++);
                }
                else
                {
                    int start = positions.Count;
                    segments.Add(new QwenImage21Segment(start, start + i - begin, begin, false));
                    for (int j = begin; j < i; ++j, ++position) positions.Add((position, position, position));
                }
            }
            if (nextImage + 1 != shapes.Count) throw new ArgumentException("Missing reference image slots.");
            int prefix = positions.Count;
            AppendImage(nextImage);
            int half = headDim / 2;
            var cos = new float[checked(positions.Count * half)];
            var sin = new float[cos.Length];
            var (dt, dh, dw) = Axes(headDim);
            int[] dims = { dt, dh, dw };
            for (int token = 0; token < positions.Count; ++token)
            {
                var p = positions[token];
                int[] coords = { p.T, p.H, p.W };
                int channel = 0;
                for (int axis = 0; axis < 3; ++axis)
                    for (int j = 0; j < dims[axis] / 2; ++j, ++channel)
                    {
                        double angle = coords[axis] / Math.Pow(10000.0, 2.0 * j / dims[axis]);
                        cos[token * half + channel] = (float)Math.Cos(angle);
                        sin[token * half + channel] = (float)Math.Sin(angle);
                    }
            }
            return new QwenImage21Layout(segments.ToArray(), cos, sin, prefix, headDim);
        }

        // The cached step's tables: the target rows' angles, made once.
        private float[] _targetCos, _targetSin;

        /// <summary>The rotary tables of the target rows alone (a cached step's).</summary>
        internal (float[] Cos, float[] Sin) TargetTables()
        {
            if (_targetCos == null)
            {
                int half = HeadDim / 2;
                _targetCos = Cos.AsSpan(Prefix * half).ToArray();
                _targetSin = Sin.AsSpan(Prefix * half).ToArray();
            }
            return (_targetCos, _targetSin);
        }
    }
}
