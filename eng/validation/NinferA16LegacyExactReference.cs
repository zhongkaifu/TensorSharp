// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// CPU oracle expressed independently over original NInfer encoded bytes.
// Arithmetic: frozen TensorSharp ts_ni_project_a16; independent original-byte storage decoder.
using System;

namespace TensorSharp.Validation;

/// <summary>Exact F32 frozen legacy-schedule oracle; no CUDA or model loading. It reads original
/// adjacent-K codes and original m128x4 scales, independently of the candidate's TS layout.</summary>
public static class NinferA16LegacyExactReference
{
    public const int PaddingBits = unchecked((int)0x7fc0a16f);

    public static float[] Compute(ReadOnlySpan<byte> originalAdjacentCodes, ReadOnlySpan<byte> originalSwizzledScales,
        ReadOnlySpan<float> input, int outputRows, int columns, int tokens, float inverseWeightDivisor,
        int inputStride = 0, int outputStride = 0)
    {
        if (outputRows <= 0 || columns <= 0 || columns % 512 != 0 || tokens < 1 || tokens > 3)
            throw new ArgumentOutOfRangeException(nameof(columns));
        if (!float.IsFinite(inverseWeightDivisor) || inverseWeightDivisor <= 0)
            throw new ArgumentOutOfRangeException(nameof(inverseWeightDivisor));
        inputStride = inputStride == 0 ? columns : inputStride;
        outputStride = outputStride == 0 ? outputRows : outputStride;
        if (inputStride < columns || outputStride < outputRows) throw new ArgumentOutOfRangeException(nameof(inputStride));
        int storageRows = checked((outputRows + 127) / 128 * 128);
        if (originalAdjacentCodes.Length < checked(storageRows * columns / 2)
            || originalSwizzledScales.Length < checked(storageRows * columns / 16)
            || input.Length < checked((tokens - 1) * inputStride + columns))
            throw new ArgumentException("Original encoded storage or activation span is truncated.");
        for (int token = 0; token < tokens; token++)
            for (int c = 0; c < columns; c++)
            {
                float value = input[token * inputStride + c];
                if (!float.IsFinite(value) || (BitConverter.SingleToInt32Bits(value) & 65535) != 0)
                    throw new ArgumentException("Original A16 activations must already be finite BF16-representable values.");
            }

        var result = new float[checked(tokens * outputStride)];
        Array.Fill(result, BitConverter.Int32BitsToSingle(PaddingBits));
        const int chains = 1;
        var accumulators = new float[32 * chains];
        var laneTotals = new float[32];
        var previous = new float[32];
        for (int token = 0; token < tokens; token++)
            for (int row = 0; row < outputRows; row++)
            {
                Array.Clear(accumulators);
                for (int lane = 0; lane < 32; lane++)
                    for (int group = lane; group < columns / 16; group += 32)
                    {
                        int scaleOffset = OriginalScaleOffset(row, group, columns);
                        float coefficient = DecodeScale(originalSwizzledScales[scaleOffset]) * inverseWeightDivisor;
                        int codeStart = checked(row * (columns / 2) + group * 8);
                        for (int step = 0; step < 16; step++)
                        {
                            int logicalK = step / 2 + (step % 2) * 8;
                            byte packed = originalAdjacentCodes[codeStart + logicalK / 2];
                            int code = (packed >> ((logicalK & 1) * 4)) & 15;
                            float weight = DecodeCode(code) * coefficient;
                            int chain = logicalK & (chains - 1);
                            int index = lane * chains + chain;
                            accumulators[index] = MathF.FusedMultiplyAdd(weight,
                                input[token * inputStride + group * 16 + logicalK], accumulators[index]);
                        }
                    }
                for (int lane = 0; lane < 32; lane++)
                {
                    laneTotals[lane] = accumulators[lane];
                }
                // Simultaneous descending SHFL_XOR butterfly, exactly as the legacy helper.
                // Do not add a separate zero when transferring the accumulator.
                for (int distance = 16; distance > 0; distance /= 2)
                {
                    laneTotals.CopyTo(previous, 0);
                    for (int lane = 0; lane < 32; lane++)
                        laneTotals[lane] = previous[lane] + previous[lane ^ distance];
                }
                result[token * outputStride + row] = laneTotals[0];
            }
        return result;
    }

    public static int OriginalScaleOffset(int row, int group, int columns)
    {
        if (row < 0 || group < 0 || columns <= 0 || columns % 64 != 0 || group >= columns / 16)
            throw new ArgumentOutOfRangeException(nameof(group));
        // The original directory geometry: [M/128][K/64][row%32][row/32][K16%4].
        return checked((row / 128 * (columns / 64) + group / 4) * 512 + (row % 32) * 16 + (row % 128 / 32) * 4 + group % 4);
    }

    public static float DecodeScale(byte code)
    {
        if (code >= 127) throw new ArgumentException("Original positive-finite E4M3 scale required.", nameof(code));
        int exponent = code / 8, mantissa = code % 8;
        return exponent == 0 ? MathF.ScaleB(mantissa, -9) : MathF.ScaleB(8 + mantissa, exponent - 10);
    }

    public static float DecodeCode(int code)
    {
        if ((uint)code >= 16) throw new ArgumentOutOfRangeException(nameof(code));
        float magnitude = (code & 7) switch { 0 => 0f, 1 => 0.5f, 2 => 1f, 3 => 1.5f, 4 => 2f, 5 => 3f, 6 => 4f, _ => 6f };
        return (code & 8) == 0 ? magnitude : -magnitude;
    }
}
