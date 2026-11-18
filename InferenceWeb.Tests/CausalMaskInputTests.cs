// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// A causal mask given as its pattern (CausalMaskInput) holds the values the shared runtime's host masks held, and its
// analytic classification is what the interpreter's scan of those values finds, field by field.
using TensorSharp.Graph;

namespace InferenceWeb.Tests;

public sealed class CausalMaskInputTests
{
    /// <summary>The runtime's host mask before the pattern moved to Core (StandardDecoderModel.GraphStep.FillCausalRows): row q
    /// -inf before the floor and past start + q, zero between.</summary>
    private static ushort[] ReferenceRows(int width, int start, int count, int attentionFloor)
    {
        const ushort negInf = 0xFC00;
        var mask = new ushort[(long)width * count];
        int floor = Math.Min(attentionFloor, width);
        for (int q = 0; q < count; q++)
        {
            Span<ushort> row = mask.AsSpan(q * width, width);
            int open = Math.Min(width, start + q + 1);
            row[..floor].Fill(negInf);
            if (open > floor) row[floor..open].Clear();
            row[Math.Max(open, floor)..].Fill(negInf);
        }
        return mask;
    }

    private static IEnumerable<(int Width, int Start, int Count, int Floor)> Grid()
    {
        foreach (int width in new[] { 1, 2, 3, 5, 16, 17, 64, 256, 300 })
        foreach (int count in new[] { 1, 2, 3, 8, 17, 24 })
        {
            var starts = new SortedSet<int> { 0, 1, 2, width / 2, width - count - 1, width - count, width - count + 1, width - 2, width - 1, width, width + 1 };
            starts.RemoveWhere(s => s < 0);
            foreach (int start in starts)
            {
                var floors = new SortedSet<int>
                {
                    0, 1, 2, start, start + 1, start + count - 1, start + count, start + count + 1, width - 1, width, width + 5,
                };
                floors.RemoveWhere(f => f < 0);
                foreach (int floor in floors) yield return (width, start, count, floor);
            }
        }
    }

    /// <summary>F16 and F32 values equal the runtime's host mask (widened exactly), whole and row by row.</summary>
    [Fact]
    public void ValuesAreTheHostMasks()
    {
        int cases = 0;
        foreach ((int width, int start, int count, int floor) in Grid())
        {
            var pattern = new CausalMaskInput(width, start, count, floor);
            ushort[] expected = ReferenceRows(width, start, count, floor);
            var half = new ushort[pattern.Values];
            pattern.FillF16(half);
            Assert.True(expected.AsSpan().SequenceEqual(half), $"F16 values of {pattern}");
            var single = new float[pattern.Values];
            pattern.FillF32(single);
            for (long i = 0; i < expected.LongLength; i++)
                Assert.True(BitConverter.SingleToInt32Bits((float)BitConverter.UInt16BitsToHalf(expected[i])) == BitConverter.SingleToInt32Bits(single[i]),
                    $"F32 value {i} of {pattern}: {single[i]} where the host mask widens to {(float)BitConverter.UInt16BitsToHalf(expected[i])}");
            // A row range lands where the whole fill puts it.
            if (count > 2)
            {
                var rows = new float[2L * width];
                pattern.FillF32(rows, 1, 2);
                Assert.True(single.AsSpan(width, 2 * width).SequenceEqual(rows), $"rows 1-2 of {pattern}");
                var halfRows = new ushort[2L * width];
                pattern.FillF16(halfRows, 1, 2);
                Assert.True(half.AsSpan(width, 2 * width).SequenceEqual(halfRows), $"F16 rows 1-2 of {pattern}");
            }
            cases++;
        }
        Assert.True(cases > 1000, $"only {cases} masks checked");
    }

    /// <summary>A mask past the row-parallel threshold fills as the serial rows do.</summary>
    [Fact]
    public void LargeMasksFillAsSerialRows()
    {
        var pattern = new CausalMaskInput(4096, 1500, 300, 7);
        Assert.True(pattern.Values >= 1 << 20);
        var half = new ushort[pattern.Values];
        pattern.FillF16(half);
        Assert.True(ReferenceRows(4096, 1500, 300, 7).AsSpan().SequenceEqual(half));
        var single = new float[pattern.Values];
        pattern.FillF32(single);
        var rows = new float[pattern.Values];
        pattern.FillF32(rows, 0, 300);
        Assert.True(single.AsSpan().SequenceEqual(rows));
    }

    /// <summary>
    /// The analytic classification equals the interpreter's scan of the values (TensorGraphExecutor.Classify) in every field
    /// - pattern, floor, keys, trimmed pattern and trimmed floor - for every row count a reader may classify (all the
    /// mask's rows or fewer), over a grid with the edges: one row; a floor of zero, at or past the rows' end, past the
    /// width; rows ending at the width or before it (a padded window) or past it; a start of zero.
    /// </summary>
    [Fact]
    public void ClassificationIsTheScansOfTheValues()
    {
        int cases = 0;
        var seen = new HashSet<(CausalMaskPattern, CausalMaskPattern)>();
        foreach ((int width, int start, int count, int floor) in Grid())
        {
            var pattern = new CausalMaskInput(width, start, count, floor);
            var values = new float[pattern.Values];
            pattern.FillF32(values);
            foreach (int rows in new[] { count, count - 1, 1 }.Distinct().Where(r => r >= 1))
            {
                MaskClassification scanned = TensorSharp.Graph.TensorGraphExecutor.ClassifyForTests(values, rows, width);
                MaskClassification analytic = pattern.Classify(rows, width);
                Assert.True(scanned == analytic, $"{pattern}, {rows} rows: the scan finds {scanned}, the pattern gives {analytic}");
                seen.Add((analytic.Pattern, analytic.Trimmed));
                cases++;
            }
        }
        Assert.True(cases > 2000, $"only {cases} classifications checked");
        // The grid reaches every outcome: a whole causal or open mask, a padded window's trimmed ones, and rows past the width.
        Assert.Contains((CausalMaskPattern.Causal, CausalMaskPattern.Causal), seen);
        Assert.Contains((CausalMaskPattern.None, CausalMaskPattern.None), seen);
        Assert.Contains((CausalMaskPattern.Explicit, CausalMaskPattern.Causal), seen);
        Assert.Contains((CausalMaskPattern.Explicit, CausalMaskPattern.None), seen);
        Assert.Contains((CausalMaskPattern.Explicit, CausalMaskPattern.Explicit), seen);
    }

    [Fact]
    public void ClassifyingAnotherShapeIsRefused()
    {
        var pattern = new CausalMaskInput(256, 10, 20, 0);
        Assert.Throws<ArgumentException>(() => pattern.Classify(20, 255));
        Assert.Throws<ArgumentOutOfRangeException>(() => pattern.Classify(21, 256));
        Assert.Throws<ArgumentOutOfRangeException>(() => pattern.Classify(0, 256));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CausalMaskInput(256, 10, 20, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CausalMaskInput(0, 0, 1, 0));
    }
}
