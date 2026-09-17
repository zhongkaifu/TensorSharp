// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The two coordinate conversions a Gemma 4 media chunk after a reused prefix depends
// on: absolute soft-token positions to the fused kernels' per-chunk mask, and to the
// frame of the per-op path's key buffer. Model-free; the numeric check is
// Gemma4MediaAfterReusedPrefixExactnessTests (and the native
// gemma4-multimodal-mask-after-reused-prefix test for the kernel rows).
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public class Gemma4SoftTokenMaskTests
{
    [Fact]
    public void ChunkMask_IsIndexedByChunkPosition_AtAnyStartPosition()
    {
        // An image at absolute positions [1420, 1682) in a chunk that starts at 1416.
        var soft = new HashSet<int>(Enumerable.Range(1420, 262));
        byte[] mask = Gemma4Model.ChunkSoftTokenMask(soft, startPos: 1416, n: 278);

        Assert.Equal(278, mask.Length);
        Assert.Equal(Enumerable.Range(0, 278).Select(i => (byte)(i >= 4 && i < 266 ? 1 : 0)), mask);
    }

    [Fact]
    public void ChunkMask_AtStartZero_IsTheOldAbsoluteMask()
    {
        var soft = new HashSet<int> { 3, 4, 5, 9 };
        byte[] mask = Gemma4Model.ChunkSoftTokenMask(soft, startPos: 0, n: 8);
        // Positions past the chunk are ignored, as before.
        Assert.Equal(new byte[] { 0, 0, 0, 1, 1, 1, 0, 0 }, mask);
    }

    [Fact]
    public void ChunkMask_IgnoresPositionsOutsideTheChunk_AndIsNullForText()
    {
        var soft = new HashSet<int> { 10, 11, 30, 31 };
        Assert.Equal(new byte[] { 1, 1, 0, 0 }, Gemma4Model.ChunkSoftTokenMask(soft, startPos: 10, n: 4));
        Assert.Null(Gemma4Model.ChunkSoftTokenMask(null, startPos: 10, n: 4));
        Assert.Null(Gemma4Model.ChunkSoftTokenMask(new HashSet<int>(), startPos: 10, n: 4));
    }

    [Fact]
    public void ShiftPositions_MovesAbsolutePositionsIntoTheKeyBufferFrame()
    {
        // Past the 512-token window the per-op path attends [prev window (511) ++ chunk]:
        // the buffer starts at totalSeqLen - kvLen = 1694 - (511 + 278) = 905.
        var soft = new HashSet<int>(Enumerable.Range(1420, 262));
        HashSet<int> shifted = Gemma4Model.ShiftPositions(soft, 1694 - (511 + 278));

        Assert.Equal(Enumerable.Range(515, 262), shifted.OrderBy(p => p));
        // Nothing to move within the window or on a global layer: the same set comes back.
        Assert.Same(soft, Gemma4Model.ShiftPositions(soft, 0));
        Assert.Null(Gemma4Model.ShiftPositions(null, 905));
    }
}
