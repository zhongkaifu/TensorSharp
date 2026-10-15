// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
internal static class OracleMath
{
    // These are independently evaluated scalar models of the pinned ggml input
    // quantizers, not captured GPU buffers or proof of actual kernel dispatch.
    public static double[] Activation(float[] x, int width, string kind)
    {
        if (width <= 0 || width % 256 != 0 || x.Length % width != 0 || x.Any(v => !float.IsFinite(v)))
            throw new ArgumentException("Finite block-aligned activation rows required.");
        int block = kind == "cpu-q8-k" ? 256 : kind is "cuda-mmvq-q8-1" or "cuda-mmq-d4" ? 32 :
            throw new ArgumentException("Unknown activation quantizer.");
        var result = new double[x.Length];
        for (int start = 0; start < x.Length; start += block)
        {
            float amax = 0, signedMax = 0;
            for (int j = 0; j < block; j++)
                if (Math.Abs(x[start + j]) > amax) { amax = Math.Abs(x[start + j]); signedMax = x[start + j]; }
            if (amax == 0) continue;
            float inverse = kind == "cpu-q8-k" ? -127f / signedMax : 127f / amax;
            float scale = kind == "cuda-mmvq-q8-1" ? amax / 127f : 1f / inverse;
            double storedScale = kind == "cuda-mmvq-q8-1" ? (double)(System.Half)scale : scale;
            if (!double.IsFinite(storedScale) || !float.IsFinite(inverse) || scale == 0)
                throw new NotSupportedException("Subnormal/overflow quantizer inputs require a dedicated device oracle.");
            for (int j = 0; j < block; j++)
            {
                float value = kind == "cuda-mmvq-q8-1" ? x[start + j] / scale : x[start + j] * inverse;
                float rounded = MathF.Round(value, kind == "cpu-q8-k" ? MidpointRounding.ToEven : MidpointRounding.AwayFromZero);
                if (rounded < -127 || rounded > 127) throw new InvalidDataException("Activation integer out of range.");
                result[start + j] = storedScale * rounded;
            }
        }
        return result;
    }

    public static double[] Project(float[][] weightRows, double[] input, int width, int tokens)
    {
        var output = new double[checked(weightRows.Length * tokens)];
        for (int token = 0; token < tokens; token++)
        for (int row = 0; row < weightRows.Length; row++)
        {
            double sum = 0;
            for (int k = 0; k < width; k++) sum += (double)weightRows[row][k] * input[token * width + k];
            if (!double.IsFinite(sum)) throw new InvalidDataException("Nonfinite scalar FP64 projection.");
            output[token * weightRows.Length + row] = sum;
        }
        return output;
    }

    public static Metrics Compare(double[] expected, double[] actual)
    {
        if (expected.Length == 0 || expected.Length != actual.Length) throw new ArgumentException("Mismatched metrics.");
        double squareError = 0, squareExpected = 0, squareActual = 0, dot = 0, max = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            if (!double.IsFinite(expected[i]) || !double.IsFinite(actual[i])) throw new InvalidDataException("Nonfinite comparison.");
            double delta = expected[i] - actual[i];
            squareError += delta * delta; squareExpected += expected[i] * expected[i];
            squareActual += actual[i] * actual[i]; dot += expected[i] * actual[i]; max = Math.Max(max, Math.Abs(delta));
        }
        return new(expected.Length, squareExpected == 0 ? null : Math.Sqrt(squareError / squareExpected), max,
            Math.Sqrt(squareError / expected.Length), Math.Sqrt(squareExpected), Math.Sqrt(squareActual),
            squareExpected == 0 || squareActual == 0 ? squareError == 0 ? 1 : null : dot / Math.Sqrt(squareExpected * squareActual));
    }

    public static void SelfTest()
    {
        static void Check(bool result, string message) { if (!result) throw new InvalidOperationException(message); }
        foreach (string kind in new[] { "cpu-q8-k", "cuda-mmvq-q8-1", "cuda-mmq-d4" })
        {
            Check(Activation(new float[256], 256, kind).All(v => v == 0), kind + " zero block");
            float[] x = new float[256]; x[0] = 127; x[1] = .5f; x[2] = 1.5f; x[3] = -.5f; x[4] = -1.5f;
            double[] q = Activation(x, 256, kind);
            Check(q[0] == 127 && q[2] == 2 && q[4] == -2, kind + " signed values");
            Check(q[1] == (kind == "cpu-q8-k" ? 0 : 1) && q[3] == (kind == "cpu-q8-k" ? 0 : -1), kind + " tie rounding");
        }
        float[] scaleInput = Enumerable.Repeat(1f, 256).ToArray();
        double half = Activation(scaleInput, 256, "cuda-mmvq-q8-1")[0];
        double full = Activation(scaleInput, 256, "cuda-mmq-d4")[0];
        Check(half != full && Math.Abs(full - 1) < 1e-7, "MMVQ half scale must differ from MMQ float scale");
        double[] result = Project([[1, 2], [-1, .5f]], [3, 4, 5, 6], 2, 2);
        Check(result.SequenceEqual(new double[] { 11, -1, 17, -2 }), "Token-major original K projection");
        Check(Compare([0], [1]).RelativeL2 == null && Compare([0], [1]).MaxAbsolute == 1, "Zero norm reporting");
        Console.WriteLine("PASS: zero blocks, signed ties, half/float scale, projection layout, zero-reference metrics.");
    }

    internal sealed record Metrics(int Elements, double? RelativeL2, double MaxAbsolute, double AbsoluteRms,
        double ReferenceL2, double ActualL2, double? Cosine);
}
