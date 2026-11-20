// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

namespace TensorSharp.Models
{
    public enum Lfm3VisionPieceKind
    {
        /// <summary>One special token (<see cref="Lfm3VisionPiece.Token"/>).</summary>
        Token,

        /// <summary>Plain text the tokenizer splits (a video group's <c>"&lt;t.t seconds&gt;"</c> label).</summary>
        Text,

        /// <summary><see cref="Lfm3VisionPiece.RowCount"/> embedding rows of block <see cref="Lfm3VisionPiece.Block"/> (an
        /// image's rows, or one video group's).</summary>
        Rows,
    }

    /// <summary>One piece of a media span in prompt order.</summary>
    public readonly record struct Lfm3VisionPiece(Lfm3VisionPieceKind Kind, int Token, string Text, int Block, int RowCount);

    /// <summary>A row slot run of a laid-out span: <see cref="RowCount"/> rows of block <see cref="Block"/> start at token
    /// <see cref="Offset"/> of <see cref="Lfm3VisionLayout.ToTokens"/>'s list.</summary>
    public readonly record struct Lfm3VisionRowSpan(int Offset, int Block, int RowCount);

    /// <summary>
    /// The prompt span of an image or a video, llama.cpp's mtmd layout for <c>lfm3vl</c> (= HF's
    /// <c>Lfm3OmniProcessor</c> after its placeholders are expanded):
    /// <list type="bullet">
    /// <item>image: <c>&lt;|vision_start|&gt;</c> + N rows + <c>&lt;|vision_end|&gt;</c>;</item>
    /// <item>video: <c>&lt;|vision_start|&gt;</c>, then per frame pair <c>"&lt;t.t seconds&gt;"</c> +
    /// <c>&lt;|vision_start|&gt;</c> + the pair's rows + <c>&lt;|vision_end|&gt;</c> (on LFM3-Omni with audio-in-video, the
    /// pair's soundtrack rows follow here unmarked: the audio port's), then <c>&lt;|vision_end|&gt;</c>.</item>
    /// </list>
    /// No separators; rows take plain 1-D positions and are not scaled. The chat template already renders the outer
    /// <c>&lt;|vision_start|&gt;&lt;|image|&gt;&lt;|vision_end|&gt;</c> (or <c>&lt;|video|&gt;</c>): an expander replaces
    /// the inner token with <see cref="Inner"/>.
    /// </summary>
    public sealed class Lfm3VisionLayout
    {
        private Lfm3VisionLayout(List<Lfm3VisionPiece> pieces)
        {
            Pieces = pieces;
            foreach (Lfm3VisionPiece p in pieces)
                if (p.Kind == Lfm3VisionPieceKind.Rows)
                {
                    RowCount += p.RowCount;
                    Blocks++;
                }
        }

        /// <summary>The whole span, outer markers included.</summary>
        public IReadOnlyList<Lfm3VisionPiece> Pieces { get; }

        /// <summary>Embedding rows in the span.</summary>
        public int RowCount { get; }

        /// <summary>Row blocks (1 for an image, the group count for a video).</summary>
        public int Blocks { get; }

        /// <summary>The span without its outer markers: what replaces the template's <c>&lt;|image|&gt;</c> or
        /// <c>&lt;|video|&gt;</c> token.</summary>
        public IEnumerable<Lfm3VisionPiece> Inner
        {
            get
            {
                for (int i = 1; i < Pieces.Count - 1; i++) yield return Pieces[i];
            }
        }

        /// <summary>An image's span: markers around <paramref name="rows"/> rows of block 0.</summary>
        public static Lfm3VisionLayout Image(int rows, int visionStart = Lfm3VisionEncoder.VisionStartToken, int visionEnd = Lfm3VisionEncoder.VisionEndToken)
        {
            if (rows < 1) throw new ArgumentOutOfRangeException(nameof(rows));
            return new Lfm3VisionLayout(new List<Lfm3VisionPiece>
            {
                Token(visionStart),
                new(Lfm3VisionPieceKind.Rows, 0, null, 0, rows),
                Token(visionEnd),
            });
        }

        /// <summary>A video's span: per group (block g) its label, markers and <paramref name="rowsPerGroup"/> rows, inside
        /// outer markers.</summary>
        public static Lfm3VisionLayout Video(IReadOnlyList<Lfm3VideoGroup> groups, int rowsPerGroup,
            int visionStart = Lfm3VisionEncoder.VisionStartToken, int visionEnd = Lfm3VisionEncoder.VisionEndToken)
        {
            ArgumentNullException.ThrowIfNull(groups);
            if (groups.Count < 1 || rowsPerGroup < 1) throw new ArgumentException($"a video of {groups.Count} groups of {rowsPerGroup} rows");
            var pieces = new List<Lfm3VisionPiece>(groups.Count * 4 + 2) { Token(visionStart) };
            for (int g = 0; g < groups.Count; g++)
            {
                pieces.Add(new Lfm3VisionPiece(Lfm3VisionPieceKind.Text, 0, groups[g].Label, -1, 0));
                pieces.Add(Token(visionStart));
                pieces.Add(new Lfm3VisionPiece(Lfm3VisionPieceKind.Rows, 0, null, g, rowsPerGroup));
                pieces.Add(Token(visionEnd));
            }
            pieces.Add(Token(visionEnd));
            return new Lfm3VisionLayout(pieces);
        }

        private static Lfm3VisionPiece Token(int id) => new(Lfm3VisionPieceKind.Token, id, null, -1, 0);

        /// <summary>
        /// The span as token ids: special tokens as they are, text through <paramref name="tokenize"/> (no special-token
        /// parsing needed: labels are plain text), and <paramref name="rowToken"/> in every row's slot (the slot the media
        /// rows overwrite: <see cref="Lfm3VisionEncoder.ImageToken"/> as HF fills it). <paramref name="rows"/> (optional)
        /// receives where each block's rows start. <paramref name="inner"/> lays out <see cref="Inner"/> only.
        /// </summary>
        public List<int> ToTokens(Func<string, IReadOnlyList<int>> tokenize, int rowToken = Lfm3VisionEncoder.ImageToken,
            List<Lfm3VisionRowSpan> rows = null, bool inner = false)
        {
            ArgumentNullException.ThrowIfNull(tokenize);
            var tokens = new List<int>(RowCount + Pieces.Count * 4);
            int first = inner ? 1 : 0, last = inner ? Pieces.Count - 1 : Pieces.Count;
            for (int i = first; i < last; i++)
            {
                Lfm3VisionPiece p = Pieces[i];
                switch (p.Kind)
                {
                    case Lfm3VisionPieceKind.Token:
                        tokens.Add(p.Token);
                        break;
                    case Lfm3VisionPieceKind.Text:
                        tokens.AddRange(tokenize(p.Text));
                        break;
                    default:
                        rows?.Add(new Lfm3VisionRowSpan(tokens.Count, p.Block, p.RowCount));
                        for (int r = 0; r < p.RowCount; r++) tokens.Add(rowToken);
                        break;
                }
            }
            return tokens;
        }
    }
}
