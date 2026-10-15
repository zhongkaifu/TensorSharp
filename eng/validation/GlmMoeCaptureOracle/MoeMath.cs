// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
internal static class MoeMath
{
    internal sealed record Stages(double[] Gate, double[] Up, double[] Activation, double[] Down, double[] Output);
    internal sealed record Metrics(int Elements, double? RelativeL2, double MaxAbsolute, double AbsoluteRms);

    // Captured F32 activations and independently decoded F32 weights are exact
    // inputs to this FP64 reference. It does not emulate a CUDA reduction tree,
    // tensor-core precision or the kernel's quantized activation representation.
    internal static Stages Evaluate(double[] input, int[] ids, double[] routes,
        float[][][] gateWeights, float[][][] upWeights, float[][][] downWeights,
        int hidden, int ffn, int used, int rows, double clamp, bool roundStages = false)
    {
        if (input.Length != checked(hidden * rows) || ids.Length != checked(used * rows)
            || routes.Length != ids.Length || hidden <= 0 || ffn <= 0 || used <= 0 || rows <= 0
            || !double.IsFinite(clamp) || clamp < 0)
            throw new ArgumentException("Invalid MoE reference geometry.");
        double Round(double value) => roundStages ? (double)(float)value : value;
        var gate = new double[checked(ffn * used * rows)];
        var up = new double[gate.Length];
        var activation = new double[gate.Length];
        var down = new double[checked(hidden * used * rows)];
        var output = new double[checked(hidden * rows)];
        for (int row = 0; row < rows; ++row)
        for (int route = 0; route < used; ++route)
        {
            int expert = ids[row * used + route];
            if (expert < 0 || expert >= gateWeights.Length || expert >= upWeights.Length || expert >= downWeights.Length)
                throw new InvalidDataException("Captured expert ID outside original weight tensor.");
            for (int ff = 0; ff < ffn; ++ff)
            {
                double g = 0, u = 0;
                for (int h = 0; h < hidden; ++h)
                {
                    g += gateWeights[expert][ff][h] * input[row * hidden + h];
                    u += upWeights[expert][ff][h] * input[row * hidden + h];
                }
                int p = (row * used + route) * ffn + ff;
                gate[p] = g = Round(g); up[p] = u = Round(u);
                if (clamp > 0) { g = Math.Min(g, clamp); u = Math.Clamp(u, -clamp, clamp); }
                activation[p] = Round(Round(Silu(g)) * u);
            }
            for (int h = 0; h < hidden; ++h)
            {
                double value = 0;
                for (int ff = 0; ff < ffn; ++ff)
                    value += downWeights[expert][h][ff] * activation[(row * used + route) * ffn + ff];
                down[(row * used + route) * hidden + h] = Round(value);
            }
        }
        // Preserve original route-slot order. The FP32-stage variant rounds
        // each multiplication and ADD, without assuming fused FMA equivalence.
        for (int row = 0; row < rows; ++row)
        for (int h = 0; h < hidden; ++h)
        {
            double sum = 0;
            for (int route = 0; route < used; ++route)
            {
                double weighted = Round(down[(row * used + route) * hidden + h] * routes[row * used + route]);
                sum = route == 0 ? weighted : Round(sum + weighted);
            }
            output[row * hidden + h] = sum;
        }
        foreach (var values in new[] { gate, up, activation, down, output })
            if (values.Any(x => !double.IsFinite(x))) throw new ArithmeticException("Nonfinite independent MoE reference.");
        return new(gate, up, activation, down, output);
    }

    internal static double Silu(double value)
    {
        // Stable negative branch avoids exp(-x) overflow in the mathematical
        // reference; this is not a change to TensorSharp model execution.
        if (value >= 0) return value / (1 + Math.Exp(-value));
        double exp = Math.Exp(value);
        return value * exp / (1 + exp);
    }

    internal static object LocalStageMetrics(Stages actual, int[] ids, double[] routes, float[][][] downWeights,
        int hidden, int ffn, int used, int rows, double clamp)
    {
        var activation = new double[actual.Activation.Length];
        for (int i = 0; i < activation.Length; ++i)
        {
            double g = actual.Gate[i], u = actual.Up[i];
            if (clamp > 0) { g = Math.Min(g, clamp); u = Math.Clamp(u, -clamp, clamp); }
            activation[i] = Silu(g) * u;
        }
        var down = new double[actual.Down.Length];
        for (int row = 0; row < rows; ++row) for (int route = 0; route < used; ++route) for (int h = 0; h < hidden; ++h)
        {
            double sum = 0;
            for (int ff = 0; ff < ffn; ++ff)
                sum += downWeights[ids[row * used + route]][h][ff] * actual.Activation[(row * used + route) * ffn + ff];
            down[(row * used + route) * hidden + h] = sum;
        }
        var output = new double[actual.Output.Length];
        for (int row = 0; row < rows; ++row) for (int h = 0; h < hidden; ++h)
        {
            double sum = 0;
            for (int route = 0; route < used; ++route)
            {
                double weighted = actual.Down[(row * used + route) * hidden + h] * routes[row * used + route];
                sum = route == 0 ? weighted : sum + weighted;
            }
            output[row * hidden + h] = sum;
        }
        return new { activation_from_actual_projections = Compare(activation, actual.Activation),
            down_from_actual_activation = Compare(down, actual.Down), fold_from_actual_down = Compare(output, actual.Output) };
    }

    internal static Metrics Compare(double[] expected, double[] actual)
    {
        if (expected.Length == 0 || expected.Length != actual.Length) throw new ArgumentException("Metric shape mismatch.");
        double squareError = 0, squareExpected = 0, max = 0;
        for (int i = 0; i < expected.Length; ++i)
        {
            if (!double.IsFinite(expected[i]) || !double.IsFinite(actual[i])) throw new ArithmeticException("Nonfinite capture/reference.");
            double error = expected[i] - actual[i];
            squareError += error * error; squareExpected += expected[i] * expected[i];
            max = Math.Max(max, Math.Abs(error));
        }
        return new(expected.Length, squareExpected == 0 ? null : Math.Sqrt(squareError / squareExpected), max,
            Math.Sqrt(squareError / expected.Length));
    }

    internal static void SelfTest()
    {
        static void Check(bool value, string name) { if (!value) throw new InvalidOperationException(name); }
        float[][][] gate = [[[1, 0], [0, 1]], [[-1, 0], [0, -1]]];
        float[][][] up = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]];
        float[][][] down = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]];
        var a = Evaluate([2, -3], [0, 1], [.25, .75], gate, up, down, 2, 2, 2, 1, 1);
        Check(a.Gate.SequenceEqual(new double[] { 2, -3, -2, 3 }), "Expert selection and gate layout");
        Check(a.Up.SequenceEqual(new double[] { 2, -3, 2, -3 }), "Up projection layout");
        Check(Math.Abs(a.Activation[0] - Silu(1)) < 1e-15 && Math.Abs(a.Activation[1] + Silu(-3)) < 1e-15,
            "Gate upper-only and up symmetric clamp");
        Check(Math.Abs(a.Output[0] - (.25 * Silu(1) + .75 * Silu(-2))) < 1e-15, "Weighted original route order");
        var b = Evaluate([2, -3, 4, 5], [0, 1, 1, 0], [.25, .75, .6, .4], gate, up, down, 2, 2, 2, 2, 0);
        var first = Evaluate([2, -3], [0, 1], [.25, .75], gate, up, down, 2, 2, 2, 1, 0);
        var second = Evaluate([4, 5], [1, 0], [.6, .4], gate, up, down, 2, 2, 2, 1, 0);
        Check(b.Output.SequenceEqual(first.Output.Concat(second.Output)), "Batch rows preserve independent inputs/routing");
        Check(Silu(-1000) == 0 && Silu(1000) == 1000 && Silu(0) == 0, "Stable extreme SiLU");
        var rounded = Evaluate([2, -3], [0, 1], [.25, .75], gate, up, down, 2, 2, 2, 1, 1, true);
        Check(rounded.Output.All(x => x == (double)(float)x), "Explicit FP32 stage boundaries");
        Check(Compare([0], [1]).RelativeL2 == null && Compare([0], [1]).MaxAbsolute == 1, "Zero reference norm");
        try { Evaluate([1], [2], [1], gate, up, down, 1, 1, 1, 1, 0); throw new Exception("Expected ID refusal"); }
        catch (InvalidDataException) { }
        Console.WriteLine("PASS: 9 independent MoE math, clamp, layout, rounding, finite and bounds checks.");
    }
}
