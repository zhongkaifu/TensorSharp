// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

public sealed class GgmlQ8StreamingCompatibilityTests
{
    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(1)]
    [InlineData(9)]
    public void LegacySessionMatchesFullPrecisionAndReleasesOnlyItsOwnBudget(int tokens)
    {
        const int width = 64, rows = 5;
        string[] pools = ["shared", "gpu0"];
        GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
        long bytes = GgmlQ8StreamingSession.GetPayloadBytes(width, rows, tokens);
        Assert.Equal(bytes, GgmlWeightStreamingSession.GetPayloadBytes(8, width, rows, tokens,
            GgmlWeightStreamingArithmetic.FullPrecision));
        var budget = new MemoryBudget(pools.Select(pool => new MemoryCharge(pool, 2 * bytes + 64)));
        using var foreign = budget.Reserve(pools.Select(pool => new MemoryCharge(pool, 64)));
        foreign.Commit();

        float[] input = Enumerable.Range(0, tokens * width).Select(i => (i % 17 - 8) / 8f).ToArray();
        byte[] weights = new byte[rows * 68];
        float[] expected = new float[tokens * rows];
        for (int row = 0; row < rows; row++)
        for (int block = 0; block < 2; block++)
        {
            int offset = row * 68 + block * 34;
            // Exact FP16 scales 1/4 and 1/2; signed integer Q8 values.
            weights[offset + 1] = block == 0 ? (byte)0x34 : (byte)0x38;
            for (int k = 0; k < 32; k++)
            {
                int quantized = (row * 13 + block * 7 + k) % 31 - 15;
                weights[offset + 2 + k] = unchecked((byte)(sbyte)quantized);
                for (int token = 0; token < tokens; token++)
                    expected[token * rows + row] += input[token * width + block * 32 + k]
                        * quantized * (block == 0 ? 0.25f : 0.5f);
            }
        }
        float[] legacyOutput = Enumerable.Repeat(float.NaN, expected.Length).ToArray();
        float[] genericOutput = Enumerable.Repeat(float.NaN, expected.Length).ToArray();
        GCHandle inputPin = default, weightPin = default, legacyPin = default, genericPin = default;
        try
        {
            inputPin = GCHandle.Alloc(input, GCHandleType.Pinned);
            weightPin = GCHandle.Alloc(weights, GCHandleType.Pinned);
            legacyPin = GCHandle.Alloc(legacyOutput, GCHandleType.Pinned);
            genericPin = GCHandle.Alloc(genericOutput, GCHandleType.Pinned);
            using var legacy = new GgmlQ8StreamingSession(budget, pools, 0, width, rows, tokens,
                inputPin.AddrOfPinnedObject());
            AssertCharges(bytes + 64);
            using var generic = new GgmlWeightStreamingSession(budget, pools, 0, 8, width, rows, tokens,
                inputPin.AddrOfPinnedObject(), GgmlWeightStreamingArithmetic.FullPrecision);
            AssertCharges(2 * bytes + 64);
            Assert.Null(budget.TryReserve([new MemoryCharge("shared", 1)]));
            legacy.Execute(weightPin.AddrOfPinnedObject(), rows, legacyPin.AddrOfPinnedObject());
            generic.Execute(weightPin.AddrOfPinnedObject(), rows, genericPin.AddrOfPinnedObject());
            // All products and sums are small binary fractions, hence exact in
            // F32: this independent answer catches two equally broken wrappers.
            Assert.Equal(expected, legacyOutput);
            Assert.Equal(legacyOutput, genericOutput);
            legacy.Dispose();
            AssertCharges(bytes + 64);
            Array.Fill(genericOutput, float.NaN);
            generic.Execute(weightPin.AddrOfPinnedObject(), rows, genericPin.AddrOfPinnedObject());
            Assert.Equal(expected, genericOutput);
            generic.Dispose();
            AssertCharges(64);
            legacy.Dispose();
            generic.Dispose();
            AssertCharges(64);
            foreign.Dispose();
            AssertCharges(0);
        }
        finally
        {
            if (genericPin.IsAllocated) genericPin.Free();
            if (legacyPin.IsAllocated) legacyPin.Free();
            if (weightPin.IsAllocated) weightPin.Free();
            if (inputPin.IsAllocated) inputPin.Free();
        }

        void AssertCharges(long expectedBytes)
            => Assert.All(budget.Snapshot(), pool =>
            {
                Assert.Equal(0, pool.Reserved);
                Assert.Equal(expectedBytes, pool.Committed);
            });
    }
}
