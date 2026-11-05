// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.Validation;

/// <summary>Batch contract expressed as independent one-token invocations of the
/// existing frozen legacy CPU oracle. Original encoded planes stay unchanged.</summary>
public static class NinferA16BatchExactReference
{
    public const int PaddingBits = NinferA16LegacyExactReference.PaddingBits;
    public static float[] Compute(ReadOnlySpan<byte> originalCodes, ReadOnlySpan<byte> originalScales,
        ReadOnlySpan<float> input, int outputRows, int columns, int tokens, float inverseDivisor,
        int inputStride = 0, int outputStride = 0)
    {
        inputStride = inputStride == 0 ? columns : inputStride;
        outputStride = outputStride == 0 ? outputRows : outputStride;
        if (tokens < 1 || tokens > 8 || inputStride < columns || outputStride < outputRows
            || input.Length < checked((tokens - 1) * inputStride + columns))
            throw new ArgumentOutOfRangeException(nameof(tokens));
        var result = new float[checked(tokens * outputStride)];
        for (int token = 0; token < tokens; token++)
        {
            float[] one = NinferA16LegacyExactReference.Compute(originalCodes, originalScales,
                input.Slice(token * inputStride, columns), outputRows, columns, 1, inverseDivisor,
                columns, outputStride);
            one.CopyTo(result, token * outputStride);
        }
        return result;
    }
    public static float DecodeCode(int code) => NinferA16LegacyExactReference.DecodeCode(code);
    public static float DecodeScale(byte code) => NinferA16LegacyExactReference.DecodeScale(code);
    public static int OriginalScaleOffset(int row, int group, int columns) =>
        NinferA16LegacyExactReference.OriginalScaleOffset(row, group, columns);

    public static void ValidateStagePermutation()
    {
        static int Index(int column) => column ^ ((column >> 4) & 31);
        var seen = new bool[512];
        for (int i = 0; i < 512; i++)
        {
            int p = Index(i);
            if ((uint)p >= 512 || seen[p]) throw new InvalidOperationException("Stage layout is not bijective.");
            seen[p] = true;
        }
        for (int first = 0; first < 512; first += 32)
        {
            uint banks = 0;
            for (int lane = 0; lane < 32; lane++) banks |= 1u << (Index(first + lane) & 31);
            if (banks != uint.MaxValue) throw new InvalidOperationException("Contiguous stage stores conflict.");
        }
        for (int j = 0; j < 16; j++)
        {
            uint banks = 0;
            for (int lane = 0; lane < 32; lane++) banks |= 1u << (Index(lane * 16 + j) & 31);
            if (banks != uint.MaxValue) throw new InvalidOperationException("Legacy lane-group stage reads conflict.");
        }
    }
}
