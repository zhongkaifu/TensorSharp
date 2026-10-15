// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

public class Gemma4ScalarVerifyTests
{
    [Fact]
    public void PolicyOnlySelectsTwoRowSingleDeviceCudaMoeVerification()
    {
        foreach (BackendType backend in Enum.GetValues<BackendType>())
        foreach (bool tp in new[] { false, true })
        foreach (int experts in new[] { 0, 128 })
        foreach (int rows in new[] { 1, 2, 3, 4, 8, 33 })
        foreach (bool allRows in new[] { false, true })
        {
            bool expected = backend == BackendType.GgmlCuda && !tp && experts == 128 && rows == 2 && allRows;
            Assert.Equal(expected, Gemma4Model.UseScalarTwoRowVerify(backend, tp, experts, rows, allRows));
        }
    }

    [Fact]
    public void PerRowHiddenAndLogitsComeFromAdvancingSequentialState()
    {
        var workspace = new Gemma4Model.ScalarVerifyWorkspace(2, 3);
        float[] hidden = Enumerable.Repeat(-1f, 5).ToArray(), logits = Enumerable.Repeat(-1f, 7).ToArray();
        int position = 127;
        var seen = new List<int>();
        int[] firstTokenBuffer = null;
        float[] firstHiddenBuffer = null, firstLogitBuffer = null;
        void Forward(int[] tokens, float[] h, float[] l)
        {
            Assert.Single(tokens);
            if (firstTokenBuffer == null) { firstTokenBuffer = tokens; firstHiddenBuffer = h; firstLogitBuffer = l; }
            else { Assert.Same(firstTokenBuffer, tokens); Assert.Same(firstHiddenBuffer, h); Assert.Same(firstLogitBuffer, l); }
            seen.Add(tokens[0]);
            h[0] = position; h[1] = tokens[0];
            l[0] = position; l[1] = tokens[0]; l[2] = position + tokens[0];
            position++;
        }
        workspace.Run(new[] { 10, 20 }, hidden, logits, Forward);
        Assert.Equal(new[] { 10, 20 }, seen);
        Assert.Equal(129, position);
        Assert.Equal(new float[] { 127, 10, 128, 20, -1 }, hidden);
        Assert.Equal(new float[] { 127, 10, 137, 128, 20, 148, -1 }, logits);
        workspace.Run(new[] { 30, 40 }, hidden, logits, Forward);
        Assert.Equal(new float[] { 129, 30, 159, 130, 40, 170, -1 }, logits);
    }

    [Fact]
    public void HiddenCaptureCanBeOmittedWithoutChangingLogitRows()
    {
        var workspace = new Gemma4Model.ScalarVerifyWorkspace(2, 3);
        float[] logits = new float[6];
        workspace.Run(new[] { 3, 4 }, null, logits, (t, h, l) =>
        {
            Assert.Null(h);
            Array.Fill(l, t[0]);
        });
        Assert.Equal(new float[] { 3, 3, 3, 4, 4, 4 }, logits);
    }

    [Fact]
    public void InvalidOutputCapacityFailsBeforeAnyTargetStateMutation()
    {
        var workspace = new Gemma4Model.ScalarVerifyWorkspace(2, 3);
        int calls = 0;
        void Forward(int[] t, float[] h, float[] l) => calls++;
        Assert.Throws<ArgumentException>(() => workspace.Run(new[] { 1, 2 }, new float[3], new float[6], Forward));
        Assert.Throws<ArgumentException>(() => workspace.Run(new[] { 1, 2 }, new float[4], new float[5], Forward));
        Assert.Throws<ArgumentException>(() => workspace.Run(new[] { 1 }, new float[4], new float[6], Forward));
        Assert.Equal(0, calls);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void TargetFailurePropagatesWithoutRetryingOrCopyingTheFailedRow(int failRow)
    {
        var workspace = new Gemma4Model.ScalarVerifyWorkspace(2, 3);
        float[] hidden = Enumerable.Repeat(-1f, 4).ToArray(), logits = Enumerable.Repeat(-1f, 6).ToArray();
        var failure = new InvalidOperationException("target failed");
        int calls = 0;
        var actual = Assert.Throws<InvalidOperationException>(() => workspace.Run(new[] { 10, 20 }, hidden, logits, (t, h, l) =>
        {
            int row = calls++;
            Array.Fill(h, 10f); Array.Fill(l, 20f);
            if (row == failRow) throw failure;
        }));
        Assert.Same(failure, actual);
        Assert.Equal(failRow + 1, calls);
        Assert.All(hidden.Skip(failRow * 2), x => Assert.Equal(-1f, x));
        Assert.All(logits.Skip(failRow * 3), x => Assert.Equal(-1f, x));
    }

    [Theory]
    [InlineData(7, 0)]
    [InlineData(7, 1)]
    [InlineData(8, 0)]
    [InlineData(8, 1)]
    [InlineData(9, 0)]
    [InlineData(9, 1)]
    public void RejectedSecondScalarRowRestoresEvictedSlotAndKeepsFirstRow(int start, int accepted)
    {
        const int window = 8;
        int[] cache = Enumerable.Range(start - window, window).ToArray();
        // Initialize slot order by absolute position, independent of ring wrap.
        for (int p = Math.Max(0, start - window); p < start; p++) cache[p % window] = p;
        int[] expected = (int[])cache.Clone();
        for (int r = 0; r <= accepted; r++) expected[(start + r) % window] = start + r;
        // The production outer verify saves row1 once; scalar core leaves it intact.
        int backup = cache[(start + 1) % window];
        var workspace = new Gemma4Model.ScalarVerifyWorkspace(1, 1);
        int position = start;
        workspace.Run(new[] { 10, 20 }, null, new float[2], (_, _, logits) =>
        {
            cache[position % window] = position;
            logits[0] = position++;
        });
        var (first, count) = Gemma4Model.SwaSlotsToRestore(1, 2, accepted, 1, verifyKvKept: true);
        if (count != 0) { Assert.Equal(1, first); Assert.Equal(1, count); cache[(start + first) % window] = backup; }
        position = start + accepted + 1;
        Assert.Equal(expected, cache);
        Assert.Equal(start + accepted + 1, position);
    }
}
