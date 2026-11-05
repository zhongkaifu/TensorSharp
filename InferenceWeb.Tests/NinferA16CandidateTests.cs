// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;
using TensorSharp.Validation;

namespace InferenceWeb.Tests;

/// <summary>CPU contract/oracle gates only. No GPU result or speedup is claimed.</summary>
public sealed class NinferA16CandidateTests
{
    [Fact]
    public void CandidateIsAbsentUnlessAnExplicitModuleIsProvided()
    {
        Assert.Null(CudaNinferA16Module.CreateExplicit(null));
        Assert.Throws<ArgumentException>(() => CudaNinferA16Module.CreateExplicit(" "));
        Assert.Throws<FileNotFoundException>(() => CudaNinferA16Module.CreateExplicit(
            Path.Combine(Path.GetTempPath(), "missing-a16-" + Guid.NewGuid().ToString("N") + ".fatbin")));
    }

    [Theory]
    [InlineData(5120, 0)] [InlineData(6144, 1)] [InlineData(17408, 2)]
    public void CandidateAccountsForFullLogicalRowsAndExplicitStrides(int columns, int widthIndex)
    {
        var layout = CudaNinferA16Module.ValidateLayout(17, columns, 3, 1.25f, columns + 7, 29);
        Assert.Equal(widthIndex, layout.WidthIndex);
        Assert.Equal(columns + 7, layout.InputStride);
        Assert.Equal(29, layout.OutputStride);
        Assert.Equal(17L * columns / 2, layout.CodeBytes);
        Assert.Equal(17L * columns / 16, layout.ScaleBytes);
        Assert.Equal((2L * (columns + 7) + columns) * 4, layout.InputBytes);
        Assert.Equal((2L * 29 + 17) * 4, layout.OutputBytes);
    }

    [Theory]
    [InlineData(0, 5120, 1, 0, 0)] [InlineData(17, 512, 1, 0, 0)]
    [InlineData(17, 5120, 0, 0, 0)] [InlineData(17, 5120, 4, 0, 0)]
    [InlineData(17, 5120, 1, 5119, 0)] [InlineData(17, 5120, 1, 0, 16)]
    public void UnsupportedGeometryCannotLaunch(int n, int k, int tokens, int inputStride, int outputStride)
        => Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferA16Module.ValidateLayout(n, k, tokens, 1, inputStride, outputStride));

    [Theory]
    [InlineData(0u)] [InlineData(0x80000000u)] [InlineData(0xbf800000u)] [InlineData(0x7f800000u)] [InlineData(0x7fc00000u)]
    public void InvalidDivisorCannotLaunch(uint bits)
        => Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferA16Module.ValidateLayout(17, 5120, 1,
            BitConverter.Int32BitsToSingle(unchecked((int)bits))));

    [Theory]
    [InlineData(1, 128f)] [InlineData(2, 32f)] [InlineData(3, 32f)]
    public void AnalyticalCancellationFixtureDistinguishesTheOriginalFourAndOneChainSchedules(int tokens, float expected)
    {
        const int k = 512, n = 3, inputStride = k + 5, outputStride = n + 7;
        byte[] codes = Enumerable.Repeat((byte)0x22, 128 * k / 2).ToArray(); // Every original E2M1 value is exactly1.
        byte[] scales = Enumerable.Repeat((byte)0x38, 128 * k / 16).ToArray(); // Every E4M3 scale is exactly1.
        float[] input = new float[tokens * inputStride];
        Array.Fill(input, float.NaN); // Padding must never enter the dot.
        for (int t = 0; t < tokens; t++)
            for (int i = 0; i < k; i++) input[t * inputStride + i] = (i % 4) switch { 0 => 16777216f, 2 => -16777216f, _ => 1f };
        float[] result = NinferA16ScheduleReference.Compute(codes, scales, input, n, k, tokens, 1, inputStride, outputStride);
        for (int t = 0; t < tokens; t++)
        {
            Assert.All(result.AsSpan(t * outputStride, n).ToArray(), value => Assert.Equal(expected, value));
            Assert.All(result.AsSpan(t * outputStride + n, outputStride - n).ToArray(),
                value => Assert.Equal(NinferA16ScheduleReference.PaddingBits, BitConverter.SingleToInt32Bits(value)));
        }
    }

    [Theory]
    [InlineData(1)] [InlineData(2)] [InlineData(3)]
    public void ScheduleOracleStaysWithinAFixedFp64ErrorBoundWithOriginalScaleSwizzling(int tokens)
    {
        const int n = 5, k = 512, storageRows = 128, groups = k / 16;
        byte[] codes = Enumerable.Range(0, storageRows * k / 2).Select(i => (byte)(i * 37 + 17)).ToArray();
        byte[] logicalScales = Enumerable.Range(0, storageRows * groups).Select(i => (byte)(0x28 + i % 7)).ToArray();
        byte[] swizzled = new byte[logicalScales.Length];
        int at = 0;
        // Original producer's five axes; does not call the oracle's offset helper.
        for (int mb = 0; mb < storageRows / 128; mb++)
            for (int kb = 0; kb < k / 64; kb++)
                for (int rowMod = 0; rowMod < 32; rowMod++)
                    for (int quarter = 0; quarter < 4; quarter++)
                        for (int lane = 0; lane < 4; lane++)
                            swizzled[at++] = logicalScales[(mb * 128 + quarter * 32 + rowMod) * groups + kb * 4 + lane];
        float[] input = Enumerable.Range(0, tokens * k).Select(i => ((i * 31 % 513) - 256) / 32f).ToArray();
        float inverse = 0.73f;
        float[] actual = NinferA16ScheduleReference.Compute(codes, swizzled, input, n, k, tokens, inverse);
        const double unitRoundoff = 1.0 / 16777216;
        double operations = k / 32 + 4 + 5 + 2;
        double gamma = operations * unitRoundoff / (1 - operations * unitRoundoff);
        for (int t = 0; t < tokens; t++)
            for (int row = 0; row < n; row++)
            {
                double exact = 0, absoluteSum = 0;
                for (int c = 0; c < k; c++)
                {
                    byte packed = codes[row * (k / 2) + c / 2];
                    int code = (packed >> ((c & 1) * 4)) & 15;
                    double term = NinferA16ScheduleReference.DecodeCode(code)
                        * (double)NinferA16ScheduleReference.DecodeScale(logicalScales[row * groups + c / 16])
                        * inverse * input[t * k + c];
                    exact += term; absoluteSum += Math.Abs(term);
                }
                Assert.InRange(Math.Abs(actual[t * n + row] - exact), 0, gamma * absoluteSum);
            }
    }

    [Fact]
    public void ReferenceRejectsActivationValuesThatHaveNotCrossedTheOriginalBfloatBoundary()
    {
        byte[] codes = new byte[128 * 512 / 2], scales = new byte[128 * 512 / 16];
        var input = new float[512]; input[0] = 1f + 1f / 65536;
        Assert.Throws<ArgumentException>(() => NinferA16ScheduleReference.Compute(codes, scales, input, 1, 512, 1, 1));
        Assert.Equal(unchecked((int)0x80000000), BitConverter.SingleToInt32Bits(NinferA16ScheduleReference.DecodeCode(8)));
    }
}
