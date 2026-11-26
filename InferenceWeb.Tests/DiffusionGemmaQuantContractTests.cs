// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// What DiffusionGemma's pure-C# (cpu backend) forward needs from the managed quantized matmul, held
// without the 16 GB model. The prompt-KV canvas decode and the unified [prompt|canvas] forward push
// the SAME canvas row through the same weights in different company: another row count per call
// (an MoE expert sees only the canvas rows routed to it, or those plus prompt rows; the pruned last
// decode layer projects one or two rows), and other column slices (RunExpertBatch cuts each expert
// by the batch's total work). Their outputs are only bitwise equal - and with Q8 activations and
// top-8 routing a last-bit difference is a visibly different answer - if a row's result depends on
// nothing but that row and the weights. These tests pin that for every weight type the model ships
// (Q4_K, Q5_0, Q8_0, Q6_K) and their neighbours, through both entry points the forward uses.
using System.Buffers.Binary;
using TensorSharp;
using TensorSharp.Cpu;

namespace InferenceWeb.Tests;

public sealed unsafe class DiffusionGemmaQuantContractTests
{
    private const int InDim = 512, OutDim = 45, Rows = 17;   // odd column count: pair/tile tails

    public static IEnumerable<object[]> Types() =>
        new[] { GgmlTensorType.Q4_K, GgmlTensorType.Q5_0, GgmlTensorType.Q8_0, GgmlTensorType.Q6_K,
                GgmlTensorType.Q4_0, GgmlTensorType.Q5_K }
            .Select(t => new object[] { t });

    // Random blocks with sane fp16 scales (any bit pattern decodes for the quant payload).
    private static byte[] RandomWeights(GgmlTensorType type, int outDim, int inDim, int seed)
    {
        var rng = new Random(seed);
        int blockBytes = (int)GgufFile.GetTypeSize(type), blockSize = (int)GgufFile.GetBlockSize(type);
        Assert.Equal(0, inDim % blockSize);
        byte[] raw = new byte[(long)outDim * (inDim / blockSize) * blockBytes];
        rng.NextBytes(raw);
        for (int o = 0; o < raw.Length; o += blockBytes)
        {
            switch (type)
            {
                case GgmlTensorType.Q6_K:
                    WriteHalf(raw, o + blockBytes - 2, 0.01f + 0.02f * (float)rng.NextDouble());
                    break;
                case GgmlTensorType.Q4_K:
                case GgmlTensorType.Q5_K:
                    WriteHalf(raw, o, 0.02f + 0.03f * (float)rng.NextDouble());
                    WriteHalf(raw, o + 2, 0.01f + 0.02f * (float)rng.NextDouble());
                    break;
                default:   // Q4_0 / Q5_0 / Q8_0: fp16 d first
                    WriteHalf(raw, o, 0.01f + 0.03f * (float)rng.NextDouble());
                    break;
            }
        }
        return raw;
    }

    private static void WriteHalf(byte[] raw, int offset, float value)
        => BinaryPrimitives.WriteUInt16LittleEndian(raw.AsSpan(offset), BitConverter.HalfToUInt16Bits((System.Half)value));

    private static float[] RandomRows(int rows, int dim, int seed)
    {
        var rng = new Random(seed);
        var a = new float[rows * dim];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * (1 + (i % 7));
        return a;
    }

    private static void AssertRowsEqual(float[] expected, float[] actual, int firstRow, int rowCount, string what)
    {
        for (int r = 0; r < rowCount; r++)
            for (int c = 0; c < OutDim; c++)
            {
                int i = (firstRow + r) * OutDim + c;
                Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                    $"{what}: row {firstRow + r} col {c}: {expected[i]:R} vs {actual[i]:R}");
            }
    }

    /// <summary>Every row on its own: the reference a row must reproduce in any company.</summary>
    private static float[] SingleRowReference(GgmlTensorType type, byte[] w, float[] x)
    {
        var y = new float[Rows * OutDim];
        fixed (byte* wp = w) fixed (float* xp = x) fixed (float* yp = y)
            for (int r = 0; r < Rows; r++)
                ManagedQuantizedOps.AddmmQuantizedToFloat32((int)type, (IntPtr)wp, InDim, OutDim,
                    xp + r * InDim, InDim, 1, yp + r * OutDim, OutDim);
        return y;
    }

    [Theory]
    [MemberData(nameof(Types))]
    public void SingleCall_RowResult_DoesNotDependOnRowCount(GgmlTensorType type)
    {
        byte[] w = RandomWeights(type, OutDim, InDim, 11 + (int)type);
        float[] x = RandomRows(Rows, InDim, 5);
        float[] reference = SingleRowReference(type, w, x);
        foreach (int count in new[] { 2, 3, 5, 8, 17 })
            foreach (int first in new[] { 0, Rows - count })
            {
                var y = new float[Rows * OutDim];
                fixed (byte* wp = w) fixed (float* xp = x) fixed (float* yp = y)
                    ManagedQuantizedOps.AddmmQuantizedToFloat32((int)type, (IntPtr)wp, InDim, OutDim,
                        xp + first * InDim, InDim, count, yp + first * OutDim, OutDim);
                AssertRowsEqual(reference, y, first, count, $"{type} {count} rows from {first}");
            }
    }

    /// <summary>One batch over the <see cref="Rows"/> rows: a job per (row group, column slice), the way
    /// RunExpertBatch cuts experts by the batch's total work. Null when the type has no batch plan.</summary>
    private static float[] RunBatch(GgmlTensorType type, byte[] w, float[] x, int[] groups, int[] cuts)
    {
        int rowBytes = (int)ManagedQuantizedOps.RowSize((int)type, InDim);
        var y = new float[Rows * OutDim];
        var jobs = new List<ManagedQuantizedOps.QuantMatMulJob>();
        fixed (byte* wp = w) fixed (float* xp = x) fixed (float* yp = y)
        {
            int row = 0;
            foreach (int g in groups)
            {
                for (int s = 0; s + 1 < cuts.Length; s++)
                    jobs.Add(new ManagedQuantizedOps.QuantMatMulJob((IntPtr)(wp + (long)cuts[s] * rowBytes),
                        (IntPtr)(xp + row * InDim), (IntPtr)(yp + row * OutDim + cuts[s]), cuts[s + 1] - cuts[s], g, OutDim));
                row += g;
            }
            if (!ManagedQuantizedOps.TryAddmmQuantizedBatch((int)type, InDim, InDim, jobs.ToArray()))
            {
                // Only types without a direct quantized-dot plan may decline; the model's must not.
                Assert.DoesNotContain(type, new[] { GgmlTensorType.Q4_K, GgmlTensorType.Q5_0, GgmlTensorType.Q8_0, GgmlTensorType.Q6_K });
                return null;
            }
        }
        return y;
    }

    private static readonly int[] EveryRowAlone = Enumerable.Repeat(1, Rows).ToArray();

    // RunExpertBatch / CpuLinearMulti: jobs of mixed row counts and column slices in one batch.
    [Theory]
    [MemberData(nameof(Types))]
    public void Batch_RowResult_DoesNotDependOnRowCountOrColumnSlice(GgmlTensorType type)
    {
        byte[] w = RandomWeights(type, OutDim, InDim, 23 + (int)type);
        float[] x = RandomRows(Rows, InDim, 9);
        float[] reference = RunBatch(type, w, x, EveryRowAlone, [0, OutDim]);
        if (reference == null) return;
        var layouts = new (int[] Groups, int[] Cuts)[]
        {
            ([Rows], [0, OutDim]),                                      // all rows in one job
            ([1, 2, 3, 5, 6], [0, OutDim]),                             // expert-like row counts
            ([1, 2, 3, 5, 6], [0, 7, 23, OutDim]),                      // ...cut at odd columns
            ([4, 13], [0, 1, 2, 44, OutDim]),                           // single-column slices
            (EveryRowAlone, [0, 20, OutDim]),
        };
        foreach (var (groups, cuts) in layouts)
            AssertRowsEqual(reference, RunBatch(type, w, x, groups, cuts), 0, Rows,
                $"{type} groups [{string.Join(",", groups)}] cuts [{string.Join(",", cuts)}]");
    }

    // The two entry points must agree too: the pruned last decode layer projects a row or two through
    // the single-call linear while the unified forward sends the same row through it with many.
    [Theory]
    [MemberData(nameof(Types))]
    public void Batch_MatchesTheSingleCall(GgmlTensorType type)
    {
        byte[] w = RandomWeights(type, OutDim, InDim, 29 + (int)type);
        float[] x = RandomRows(Rows, InDim, 13);
        float[] reference = SingleRowReference(type, w, x);
        foreach (int[] groups in new[] { EveryRowAlone, [Rows], [1, 2, 3, 5, 6] })
        {
            float[] y = RunBatch(type, w, x, groups, [0, OutDim]);
            if (y == null) return;
            AssertRowsEqual(reference, y, 0, Rows, $"{type} batch groups [{string.Join(",", groups)}] vs single-row calls");
        }
    }
}
