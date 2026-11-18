// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Issue #376: where slots are sized to the device's room (the CUDA executor on one device), a slot is made or grown to a
// capacity class: whole 1,024s up to 4,096 cells, then four capacities an octave (1.25, 1.5, 1.75 and 2 x a power of two),
// up to the context limit.
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class SlotCapacityClassTests
{
    [Theory]
    [InlineData(1, 1024)]
    [InlineData(1024, 1024)]
    [InlineData(1025, 2048)]
    [InlineData(4096, 4096)]
    [InlineData(4097, 5120)]
    [InlineData(6145, 7168)]
    [InlineData(8192, 8192)]
    [InlineData(8193, 10240)]
    [InlineData(24153 + 3019, 28672)]    // a 24K-token prompt's reservation and its eighth
    [InlineData(28672, 28672)]
    [InlineData(28673, 32768)]
    [InlineData(52908, 57344)]
    [InlineData(78023, 81920)]
    [InlineData(131072, 131072)]
    [InlineData(146616, 163840)]
    public void ClassesAreWhole1024sThenFourAnOctave(long cells, int expected)
        => Assert.Equal(expected, StandardDecoderModel.SlotCapacityClass(cells, 1 << 20));

    [Theory]
    [InlineData(146616, 163840, 163840)]
    [InlineData(200000, 163840, 163840)]
    [InlineData(5000, 4096, 4096)]
    [InlineData(3000, 3500, 3072)]
    public void ClassesStopAtTheContextLimit(long cells, int maxContext, int expected)
        => Assert.Equal(expected, StandardDecoderModel.SlotCapacityClass(cells, maxContext));

    [Fact]
    public void EveryClassHoldsItsCellsAndIsAWhole1024()
    {
        for (long cells = 1; cells <= 300_000; cells += 97)
        {
            int capacity = StandardDecoderModel.SlotCapacityClass(cells, 1 << 20);
            Assert.True(capacity >= cells, $"{cells} cells: class {capacity}");
            Assert.Equal(0, capacity % 1024);
            // Never more than a quarter past what it holds above 4,096 cells (one class step), a 1,024 below.
            Assert.True(cells <= 4096 ? capacity - cells < 1024 : capacity <= cells * 1.25 + 1024, $"{cells} cells: class {capacity}");
        }
    }
}
