// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Graph
{
    /// <summary>What a mask is over its keys: none (every key from the floor on visible), causal aligned to the last valid
    /// key, or anything else (applied as given).</summary>
    public enum CausalMaskPattern { None, Causal, Explicit }

    /// <summary>How an attention reads a mask (an executor's classification of its values): the pattern over every key and
    /// the floor before which every row masks; <see cref="Keys"/>, the keys before the trailing ones every row masks; and the
    /// pattern over keys [<see cref="TrimmedFloor"/>, <see cref="Keys"/>) alone, for kernels that stop at Keys.</summary>
    public readonly record struct MaskClassification(CausalMaskPattern Pattern, int Floor, int Keys, CausalMaskPattern Trimmed,
        int TrimmedFloor);

    /// <summary>
    /// An F16 causal mask [<see cref="Width"/> keys, <see cref="Count"/> rows] given as its pattern: rows continuing at
    /// <see cref="Start"/>, row q -inf before the attention floor <see cref="Floor"/> and past Start + q, zero between. An
    /// executor that declares <see cref="GraphCapabilities.StructuralCausalMasks"/> takes a prompt chunk's mask as this
    /// handle (no host bytes): its causal kernels read the pattern, and only an op that reads the values makes them, the
    /// values the host array held. The mask of a 2048-row chunk at 67K keys had been 137M values built, widened to F32 and
    /// scanned on the host at every step, only for the kernel to learn that it was causal (about 2 ns a value).
    /// </summary>
    public sealed record CausalMaskInput
    {
        /// <summary>Rows at least this many values are filled row-parallel (a 3.6K-row prompt chunk's mask is 26 MB: element
        /// loops took ~10 ms of its trace).</summary>
        private const long ParallelValues = 1 << 20;

        private const ushort NegativeInfinityF16 = 0xFC00;

        public CausalMaskInput(int width, int start, int count, int floor)
        {
            if (width < 1) throw new ArgumentOutOfRangeException(nameof(width), width, "a mask has at least one key");
            if (count < 1) throw new ArgumentOutOfRangeException(nameof(count), count, "a mask has at least one row");
            if (start < 0) throw new ArgumentOutOfRangeException(nameof(start), start, "rows start at a position");
            if (floor < 0) throw new ArgumentOutOfRangeException(nameof(floor), floor, "the attention floor is a key");
            Width = width;
            Start = start;
            Count = count;
            Floor = floor;
        }

        /// <summary>Keys a row holds (the mask's ne0).</summary>
        public int Width { get; }

        /// <summary>The first row's position: row q sees keys up to Start + q.</summary>
        public int Start { get; }

        /// <summary>Rows (the mask's ne1).</summary>
        public int Count { get; }

        /// <summary>Keys before this are masked in every row (a draft layer that resumed after a gap).</summary>
        public int Floor { get; }

        /// <summary>The mask's values: <see cref="Width"/> x <see cref="Count"/>.</summary>
        public long Values => (long)Width * Count;

        /// <summary>Rows [<paramref name="firstRow"/>, + <paramref name="rowCount"/>) into <paramref name="rows"/>, F16 bits.</summary>
        public void FillF16(Span<ushort> rows, int firstRow, int rowCount)
        {
            CheckRows(rows.Length, firstRow, rowCount);
            for (int r = 0; r < rowCount; r++) Row(rows.Slice(r * Width, Width), firstRow + r, NegativeInfinityF16, (ushort)0);
        }

        /// <summary>Rows [<paramref name="firstRow"/>, + <paramref name="rowCount"/>) into <paramref name="rows"/> as F32: the
        /// F16 values widened (-inf and +0 are exact in both).</summary>
        public void FillF32(Span<float> rows, int firstRow, int rowCount)
        {
            CheckRows(rows.Length, firstRow, rowCount);
            for (int r = 0; r < rowCount; r++) Row(rows.Slice(r * Width, Width), firstRow + r, float.NegativeInfinity, 0f);
        }

        /// <summary>The whole mask into <paramref name="mask"/> (F16 bits), large masks row-parallel.</summary>
        public void FillF16(ushort[] mask) => FillWhole(mask, NegativeInfinityF16, (ushort)0);

        /// <summary>The whole mask into <paramref name="mask"/> as F32, large masks row-parallel.</summary>
        public void FillF32(float[] mask) => FillWhole(mask, float.NegativeInfinity, 0f);

        private void FillWhole<T>(T[] mask, T negativeInfinity, T zero) where T : unmanaged
        {
            ArgumentNullException.ThrowIfNull(mask);
            CheckRows(mask.LongLength, 0, Count);
            if (Values >= ParallelValues)
                System.Threading.Tasks.Parallel.For(0, Count, q => Row(mask.AsSpan((int)((long)q * Width), Width), q, negativeInfinity, zero));
            else
                for (int q = 0; q < Count; q++) Row(mask.AsSpan(q * Width, Width), q, negativeInfinity, zero);
        }

        private void CheckRows(long length, int firstRow, int rowCount)
        {
            if (firstRow < 0 || rowCount < 0 || (long)firstRow + rowCount > Count)
                throw new ArgumentOutOfRangeException(nameof(firstRow), $"rows [{firstRow}, +{rowCount}) of a {Count}-row mask");
            if (length < (long)rowCount * Width)
                throw new ArgumentException($"{length} values hold fewer than {rowCount} rows of {Width} keys");
        }

        /// <summary>Row q: -inf before the floor and past Start + q, zero between.</summary>
        private void Row<T>(Span<T> row, int q, T negativeInfinity, T zero) where T : unmanaged
        {
            int floor = Math.Min(Floor, Width);
            int open = (int)Math.Min(Width, (long)Start + q + 1);
            row[..floor].Fill(negativeInfinity);
            if (open > floor) row[floor..open].Fill(zero);
            row[Math.Max(open, floor)..].Fill(negativeInfinity);
        }

        /// <summary>
        /// What a scan of the mask's first <paramref name="rows"/> rows over its <paramref name="keys"/> keys finds
        /// (TensorGraphExecutor's classification of host values), from the pattern alone. With E = min(W, S + rows) the last
        /// row's end and F = min(Floor, W): every row masks keys before F and from E on, so the floor is F and E keys are
        /// valid; F at or past E masks every key (None from the floor W). Row r opens max(min(W, S + r + 1), F), which is the
        /// causal row's max(E - rows + r + 1, F) exactly when S + rows reaches no further than W; a single row, or rows that
        /// all open every key (S + 1 at or past W), are None. A pattern over fewer than the mask's keys holds only for
        /// kernels that stop at Keys: the mask is Explicit to every other reader.
        /// </summary>
        public MaskClassification Classify(long rows, long keys)
        {
            if (keys != Width) throw new ArgumentException($"the mask has {Width} keys, not {keys}", nameof(keys));
            if (rows < 1 || rows > Count) throw new ArgumentOutOfRangeException(nameof(rows), rows, $"rows of a {Count}-row mask");
            long end = Math.Min(Width, (long)Start + rows);
            int floor = Math.Min(Floor, Width);
            if (floor >= end) return new MaskClassification(CausalMaskPattern.None, Width, Width, CausalMaskPattern.None, Width);
            int valid = (int)end;
            CausalMaskPattern pattern = rows == 1 || (long)Start + 1 >= Width ? CausalMaskPattern.None
                : (long)Start + rows <= Width ? CausalMaskPattern.Causal
                : CausalMaskPattern.Explicit;
            int trimmedFloor = pattern == CausalMaskPattern.Explicit ? 0 : floor;
            if (pattern != CausalMaskPattern.Explicit && valid < keys)
                return new MaskClassification(CausalMaskPattern.Explicit, 0, valid, pattern, trimmedFloor);
            return new MaskClassification(pattern, trimmedFloor, valid, pattern, trimmedFloor);
        }
    }
}
