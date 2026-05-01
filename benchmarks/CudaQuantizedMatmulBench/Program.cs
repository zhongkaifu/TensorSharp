using System.Diagnostics;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;

const int GgmlTypeQ8_0 = 8;
const int GgmlTypeQ4_0 = 2;

if (!CudaBackend.IsAvailable())
{
    Console.WriteLine("CUDA is not available.");
    return 1;
}

int iterations = GetArg("--iterations", 30);
int warmup = GetArg("--warmup", 5);
var cases = new[]
{
    new BenchCase("decode-q8_0-4096x4096-r1", 1, 4096, 4096),
    new BenchCase("smallbatch-q8_0-4096x4096-r8", 8, 4096, 4096),
};

using var allocator = new CudaAllocator();
Console.WriteLine($"CUDA quantized matmul benchmark, iterations={iterations}, warmup={warmup}");

foreach (BenchCase bench in cases)
{
    RunCase(allocator, bench, warmup, iterations);
}

RunQ4Case(allocator, new BenchCase("decode-q4_0-4096x4096-r1", 1, 4096, 4096), warmup, iterations);
RunQ4Case(allocator, new BenchCase("smallbatch-q4_0-4096x4096-r8", 8, 4096, 4096), warmup, iterations);
RunBatchedAddmmCase(allocator, warmup, iterations);
RunScaledDotProductAttentionCase(allocator, warmup, iterations);
RunElementwiseCase(allocator, warmup, iterations);
RunFusedElementwiseCase(allocator, warmup, iterations);

return 0;

static void RunCase(CudaAllocator allocator, BenchCase bench, int warmup, int iterations)
{
    byte[] weights = CreateQ8_0Rows(bench.OutDim, bench.InDim);
    float[,] inputValues = CreateInput(bench.Rows, bench.InDim);
    IntPtr hostWeights = Marshal.AllocHGlobal(weights.Length);

    try
    {
        Marshal.Copy(weights, 0, hostWeights, weights.Length);
        using var input = Tensor.FromArray(allocator, inputValues);
        using var output = new Tensor(allocator, DType.Float32, bench.Rows, bench.OutDim);

        bool ok = CudaQuantizedOps.TryAddmmQuantizedToFloat32(
            output,
            input,
            hostWeights,
            hostWeights,
            GgmlTypeQ8_0,
            bench.InDim,
            bench.OutDim,
            weights.Length);
        if (!ok)
            throw new InvalidOperationException("CUDA quantized matmul dispatch failed.");

        allocator.Synchronize();
        float[] expected = DequantizedMatmulQ80(weights, bench.OutDim, bench.InDim, inputValues, bench.Rows);
        float[] actual = output.GetElementsAsFloat(bench.Rows * bench.OutDim);
        float maxAbsDiff = MaxAbsDiff(expected, actual);

        for (int i = 0; i < warmup; i++)
        {
            CudaQuantizedOps.TryAddmmQuantizedToFloat32(
                output,
                input,
                hostWeights,
                hostWeights,
                GgmlTypeQ8_0,
                bench.InDim,
                bench.OutDim,
                weights.Length);
        }
        allocator.Synchronize();

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            CudaQuantizedOps.TryAddmmQuantizedToFloat32(
                output,
                input,
                hostWeights,
                hostWeights,
                GgmlTypeQ8_0,
                bench.InDim,
                bench.OutDim,
                weights.Length);
        }
        allocator.Synchronize();
        sw.Stop();

        actual = output.GetElementsAsFloat(bench.Rows * bench.OutDim);
        float postBenchmarkDiff = MaxAbsDiff(expected, actual);
        double ms = sw.Elapsed.TotalMilliseconds / iterations;
        double gflops = (2.0 * bench.Rows * bench.InDim * bench.OutDim) / (ms * 1.0e6);

        Console.WriteLine($"{bench.Name}: {ms:F3} ms/op, {gflops:F1} GFLOP/s, max_abs_diff={maxAbsDiff:G6}, post_diff={postBenchmarkDiff:G6}");
    }
    finally
    {
        CudaQuantizedOps.ReleaseQuantizedWeight(allocator, hostWeights);
        Marshal.FreeHGlobal(hostWeights);
    }
}

static void RunQ4Case(CudaAllocator allocator, BenchCase bench, int warmup, int iterations)
{
    byte[] weights = CreateQ4_0Rows(bench.OutDim, bench.InDim);
    float[,] inputValues = CreateInput(bench.Rows, bench.InDim);
    IntPtr hostWeights = Marshal.AllocHGlobal(weights.Length);

    try
    {
        Marshal.Copy(weights, 0, hostWeights, weights.Length);
        using var input = Tensor.FromArray(allocator, inputValues);
        using var output = new Tensor(allocator, DType.Float32, bench.Rows, bench.OutDim);

        bool ok = CudaQuantizedOps.TryAddmmQuantizedToFloat32(
            output,
            input,
            hostWeights,
            hostWeights,
            GgmlTypeQ4_0,
            bench.InDim,
            bench.OutDim,
            weights.Length);
        if (!ok)
            throw new InvalidOperationException("CUDA Q4_0 quantized matmul dispatch failed.");

        allocator.Synchronize();
        float[] expected = DequantizedMatmulQ40(weights, bench.OutDim, bench.InDim, inputValues, bench.Rows);
        float[] actual = output.GetElementsAsFloat(bench.Rows * bench.OutDim);
        float maxAbsDiff = MaxAbsDiff(expected, actual);

        for (int i = 0; i < warmup; i++)
        {
            CudaQuantizedOps.TryAddmmQuantizedToFloat32(
                output,
                input,
                hostWeights,
                hostWeights,
                GgmlTypeQ4_0,
                bench.InDim,
                bench.OutDim,
                weights.Length);
        }
        allocator.Synchronize();

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            CudaQuantizedOps.TryAddmmQuantizedToFloat32(
                output,
                input,
                hostWeights,
                hostWeights,
                GgmlTypeQ4_0,
                bench.InDim,
                bench.OutDim,
                weights.Length);
        }
        allocator.Synchronize();
        sw.Stop();

        actual = output.GetElementsAsFloat(bench.Rows * bench.OutDim);
        float postBenchmarkDiff = MaxAbsDiff(expected, actual);
        double ms = sw.Elapsed.TotalMilliseconds / iterations;
        double gflops = (2.0 * bench.Rows * bench.InDim * bench.OutDim) / (ms * 1.0e6);

        Console.WriteLine($"{bench.Name}: {ms:F3} ms/op, {gflops:F1} GFLOP/s, max_abs_diff={maxAbsDiff:G6}, post_diff={postBenchmarkDiff:G6}");
    }
    finally
    {
        CudaQuantizedOps.ReleaseQuantizedWeight(allocator, hostWeights);
        Marshal.FreeHGlobal(hostWeights);
    }
}

static int GetArg(string name, int defaultValue)
{
    string[] args = Environment.GetCommandLineArgs();
    for (int i = 0; i + 1 < args.Length; i++)
    {
        if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase) &&
            int.TryParse(args[i + 1], out int value))
        {
            return value;
        }
    }

    return defaultValue;
}

static byte[] CreateQ8_0Rows(int rows, int cols)
{
    int blocks = cols / 32;
    byte[] data = new byte[rows * blocks * 34];
    for (int row = 0; row < rows; row++)
    {
        for (int block = 0; block < blocks; block++)
        {
            int offset = (row * blocks + block) * 34;
            float scale = 0.015625f + ((row + block) % 13) * 0.0009765625f;
            ushort scaleBits = BitConverter.HalfToUInt16Bits((System.Half)scale);
            data[offset] = (byte)(scaleBits & 0xFF);
            data[offset + 1] = (byte)(scaleBits >> 8);

            for (int i = 0; i < 32; i++)
            {
                int v = ((row * 31 + block * 17 + i * 7) % 255) - 127;
                data[offset + 2 + i] = unchecked((byte)(sbyte)v);
            }
        }
    }

    return data;
}

static byte[] CreateQ4_0Rows(int rows, int cols)
{
    int blocks = cols / 32;
    byte[] data = new byte[rows * blocks * 18];
    for (int row = 0; row < rows; row++)
    {
        for (int block = 0; block < blocks; block++)
        {
            int offset = (row * blocks + block) * 18;
            float scale = 0.0625f + ((row + block) % 11) * 0.00390625f;
            ushort scaleBits = BitConverter.HalfToUInt16Bits((System.Half)scale);
            data[offset] = (byte)(scaleBits & 0xFF);
            data[offset + 1] = (byte)(scaleBits >> 8);

            for (int i = 0; i < 16; i++)
            {
                int low = ((row * 13 + block * 7 + i * 5) & 15);
                int high = ((row * 17 + block * 11 + i * 3 + 16) & 15);
                data[offset + 2 + i] = (byte)(low | (high << 4));
            }
        }
    }

    return data;
}

static float[,] CreateInput(int rows, int cols)
{
    var input = new float[rows, cols];
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            input[r, c] = MathF.Sin((r + 1) * (c + 3) * 0.0071f) * 0.5f +
                MathF.Cos((r + 5) * (c + 1) * 0.0037f) * 0.25f;
        }
    }

    return input;
}

static float[] DequantizedMatmulQ80(byte[] weights, int outDim, int inDim, float[,] input, int rows)
{
    int blocks = inDim / 32;
    float[] expected = new float[rows * outDim];
    for (int r = 0; r < rows; r++)
    {
        for (int o = 0; o < outDim; o++)
        {
            float sum = 0;
            for (int block = 0; block < blocks; block++)
            {
                int offset = (o * blocks + block) * 34;
                float scale = (float)BitConverter.UInt16BitsToHalf((ushort)(weights[offset] | (weights[offset + 1] << 8)));
                for (int i = 0; i < 32; i++)
                {
                    sbyte q = unchecked((sbyte)weights[offset + 2 + i]);
                    sum += scale * q * input[r, block * 32 + i];
                }
            }

            expected[r * outDim + o] = sum;
        }
    }

    return expected;
}

static float[] DequantizedMatmulQ40(byte[] weights, int outDim, int inDim, float[,] input, int rows)
{
    int blocks = inDim / 32;
    float[] expected = new float[rows * outDim];
    for (int r = 0; r < rows; r++)
    {
        for (int o = 0; o < outDim; o++)
        {
            float sum = 0;
            for (int block = 0; block < blocks; block++)
            {
                int offset = (o * blocks + block) * 18;
                float scale = (float)BitConverter.UInt16BitsToHalf((ushort)(weights[offset] | (weights[offset + 1] << 8)));
                for (int i = 0; i < 16; i++)
                {
                    byte packed = weights[offset + 2 + i];
                    int low = (packed & 0x0F) - 8;
                    int high = (packed >> 4) - 8;
                    sum += scale * low * input[r, block * 32 + i];
                    sum += scale * high * input[r, block * 32 + i + 16];
                }
            }

            expected[r * outDim + o] = sum;
        }
    }

    return expected;
}

static float MaxAbsDiff(float[] expected, float[] actual)
{
    float max = 0;
    for (int i = 0; i < expected.Length; i++)
        max = MathF.Max(max, MathF.Abs(expected[i] - actual[i]));
    return max;
}

static void RunBatchedAddmmCase(CudaAllocator allocator, int warmup, int iterations)
{
    const int batch = 32;
    const int rows = 64;
    const int shared = 128;
    const int cols = 128;

    using var lhs = new Tensor(allocator, DType.Float32, batch, rows, shared);
    using var rhs = new Tensor(allocator, DType.Float32, batch, cols, shared);
    FillTensor(lhs, 0.0061f, 0.25f);
    FillTensor(rhs, -0.0047f, -0.125f);
    using var rhsT = rhs.Transpose(1, 2);
    using var batchedOutput = new Tensor(allocator, DType.Float32, batch, rows, cols);
    using var loopOutput = new Tensor(allocator, DType.Float32, batch, rows, cols);

    Tensor[] lhsSlices = new Tensor[batch];
    Tensor[] rhsSlices = new Tensor[batch];
    Tensor[] loopOutputSlices = new Tensor[batch];
    try
    {
        for (int b = 0; b < batch; b++)
        {
            lhsSlices[b] = lhs.Select(0, b);
            rhsSlices[b] = rhsT.Select(0, b);
            loopOutputSlices[b] = loopOutput.Select(0, b);
        }

        Ops.AddmmBatch(batchedOutput, 0, batchedOutput, 1, lhs, rhsT);
        RunLoopedAddmm(lhsSlices, rhsSlices, loopOutputSlices, allocator);
        float[] batched = batchedOutput.GetElementsAsFloat((int)batchedOutput.ElementCount());
        float[] looped = loopOutput.GetElementsAsFloat((int)loopOutput.ElementCount());
        float maxAbsDiff = MaxAbsDiff(looped, batched);

        for (int i = 0; i < warmup; i++)
        {
            Ops.AddmmBatch(batchedOutput, 0, batchedOutput, 1, lhs, rhsT);
            allocator.Synchronize();
        }

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            Ops.AddmmBatch(batchedOutput, 0, batchedOutput, 1, lhs, rhsT);
            allocator.Synchronize();
        }
        sw.Stop();
        double batchedMs = sw.Elapsed.TotalMilliseconds / iterations;

        for (int i = 0; i < warmup; i++)
            RunLoopedAddmm(lhsSlices, rhsSlices, loopOutputSlices, allocator);

        sw.Restart();
        for (int i = 0; i < iterations; i++)
            RunLoopedAddmm(lhsSlices, rhsSlices, loopOutputSlices, allocator);
        sw.Stop();
        double loopedMs = sw.Elapsed.TotalMilliseconds / iterations;

        double gflops = (2.0 * batch * rows * shared * cols) / (batchedMs * 1.0e6);
        Console.WriteLine($"addmmbatch-f32-b{batch}-{rows}x{shared}x{cols}: {batchedMs:F3} ms/op, {gflops:F1} GFLOP/s, looped_sgemm={loopedMs:F3} ms/op, speedup={loopedMs / batchedMs:F2}x, max_abs_diff={maxAbsDiff:G6}");
    }
    finally
    {
        for (int i = 0; i < batch; i++)
        {
            lhsSlices[i]?.Dispose();
            rhsSlices[i]?.Dispose();
            loopOutputSlices[i]?.Dispose();
        }
    }
}

static void RunScaledDotProductAttentionCase(CudaAllocator allocator, int warmup, int iterations)
{
    const int batch = 1;
    const int seqQ = 256;
    const int seqK = 256;
    const int heads = 8;
    const int keyDim = 64;
    const int valueDim = 64;
    float scale = 1.0f / MathF.Sqrt(keyDim);

    float[,,,] q = new float[batch, seqQ, heads, keyDim];
    float[,,,] k = new float[batch, seqK, heads, keyDim];
    float[,,,] v = new float[batch, seqK, heads, valueDim];
    for (int b = 0; b < batch; b++)
    {
        for (int t = 0; t < seqQ; t++)
            for (int h = 0; h < heads; h++)
                for (int d = 0; d < keyDim; d++)
                    q[b, t, h, d] = MathF.Sin((t + 1) * (h + 2) * (d + 3) * 0.0017f);

        for (int t = 0; t < seqK; t++)
        {
            for (int h = 0; h < heads; h++)
            {
                for (int d = 0; d < keyDim; d++)
                    k[b, t, h, d] = MathF.Cos((t + 3) * (h + 1) * (d + 5) * 0.0013f);
                for (int d = 0; d < valueDim; d++)
                    v[b, t, h, d] = MathF.Sin((t + 5) * (h + 3) * (d + 1) * 0.0011f) * 0.25f;
            }
        }
    }

    using var qTensor = Tensor.FromArray(allocator, q);
    using var kTensor = Tensor.FromArray(allocator, k);
    using var vTensor = Tensor.FromArray(allocator, v);
    using var output = Ops.ScaledDotProductAttention(null, qTensor, kTensor, vTensor, null, scale);
    allocator.Synchronize();

    float[] actualPrefix = output.GetElementsAsFloat(Math.Min(256, (int)output.ElementCount()));
    float[] expectedPrefix = ScaledDotProductAttentionPrefix(q, k, v, scale, actualPrefix.Length);
    float maxAbsDiff = MaxAbsDiff(expectedPrefix, actualPrefix);

    for (int i = 0; i < warmup; i++)
    {
        Ops.ScaledDotProductAttention(output, qTensor, kTensor, vTensor, null, scale);
    }
    allocator.Synchronize();

    var sw = Stopwatch.StartNew();
    for (int i = 0; i < iterations; i++)
    {
        Ops.ScaledDotProductAttention(output, qTensor, kTensor, vTensor, null, scale);
    }
    allocator.Synchronize();
    sw.Stop();

    double ms = sw.Elapsed.TotalMilliseconds / iterations;
    double gflops = (2.0 * batch * heads * seqQ * seqK * (keyDim + valueDim)) / (ms * 1.0e6);
    Console.WriteLine($"sdpa-f32-b{batch}-s{seqQ}x{seqK}-h{heads}-d{keyDim}: {ms:F3} ms/op, approx {gflops:F1} GFLOP/s, prefix_max_abs_diff={maxAbsDiff:G6}");
}

static void RunElementwiseCase(CudaAllocator allocator, int warmup, int iterations)
{
    const int count = 8 * 1024 * 1024;
    using var lhs = new Tensor(allocator, DType.Float32, count);
    using var rhs = new Tensor(allocator, DType.Float32, count);
    FillTensor(lhs, 0.000013f, 0.1f);
    FillTensor(rhs, -0.000017f, -0.2f);
    using var output = new Tensor(allocator, DType.Float32, count);

    Ops.Add(output, lhs, rhs);
    allocator.Synchronize();
    float[] prefix = output.GetElementsAsFloat(8);
    float[] lhsPrefix = lhs.GetElementsAsFloat(8);
    float[] rhsPrefix = rhs.GetElementsAsFloat(8);
    float[] expected = new float[8];
    for (int i = 0; i < expected.Length; i++)
        expected[i] = lhsPrefix[i] + rhsPrefix[i];
    float maxAbsDiff = MaxAbsDiff(expected, prefix);

    for (int i = 0; i < warmup; i++)
        Ops.Add(output, lhs, rhs);
    allocator.Synchronize();

    var sw = Stopwatch.StartNew();
    for (int i = 0; i < iterations; i++)
        Ops.Add(output, lhs, rhs);
    allocator.Synchronize();
    sw.Stop();

    double ms = sw.Elapsed.TotalMilliseconds / iterations;
    double gibPerSecond = (3.0 * count * sizeof(float)) / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    Console.WriteLine($"elementwise-add-f32-{count}: {ms:F3} ms/op, {gibPerSecond:F1} GiB/s, prefix_max_abs_diff={maxAbsDiff:G6}");
}

static void RunFusedElementwiseCase(CudaAllocator allocator, int warmup, int iterations)
{
    const int count = 8 * 1024 * 1024;
    using var x = new Tensor(allocator, DType.Float32, count);
    using var y = new Tensor(allocator, DType.Float32, count);
    using var z = new Tensor(allocator, DType.Float32, count);
    using var w = new Tensor(allocator, DType.Float32, count);
    FillTensor(x, 0.000013f, 0.1f);
    FillTensor(y, -0.000017f, -0.2f);
    FillTensor(z, 0.000019f, 0.3f);
    FillTensor(w, -0.000023f, 0.05f);
    using var output = new Tensor(allocator, DType.Float32, count);

    Ops.MulMulAdd(output, x, y, z, w);
    allocator.Synchronize();
    float[] prefix = output.GetElementsAsFloat(8);
    float[] xPrefix = x.GetElementsAsFloat(8);
    float[] yPrefix = y.GetElementsAsFloat(8);
    float[] zPrefix = z.GetElementsAsFloat(8);
    float[] wPrefix = w.GetElementsAsFloat(8);
    float[] expected = new float[8];
    for (int i = 0; i < expected.Length; i++)
        expected[i] = xPrefix[i] * yPrefix[i] + zPrefix[i] * wPrefix[i];
    float maxAbsDiff = MaxAbsDiff(expected, prefix);

    for (int i = 0; i < warmup; i++)
        Ops.MulMulAdd(output, x, y, z, w);
    allocator.Synchronize();

    var sw = Stopwatch.StartNew();
    for (int i = 0; i < iterations; i++)
        Ops.MulMulAdd(output, x, y, z, w);
    allocator.Synchronize();
    sw.Stop();

    double ms = sw.Elapsed.TotalMilliseconds / iterations;
    double gibPerSecond = (5.0 * count * sizeof(float)) / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    Console.WriteLine($"fused-mulmuladd-f32-{count}: {ms:F3} ms/op, {gibPerSecond:F1} GiB/s, prefix_max_abs_diff={maxAbsDiff:G6}");
}

static void RunLoopedAddmm(Tensor[] lhsSlices, Tensor[] rhsSlices, Tensor[] outputSlices, CudaAllocator allocator)
{
    for (int b = 0; b < lhsSlices.Length; b++)
        Ops.Addmm(outputSlices[b], 0, outputSlices[b], 1, lhsSlices[b], rhsSlices[b]);
    allocator.Synchronize();
}

static void FillTensor(Tensor tensor, float scale, float offset)
{
    float[] values = new float[tensor.ElementCount()];
    for (int i = 0; i < values.Length; i++)
        values[i] = MathF.Sin((i + 1) * scale) * 0.5f + MathF.Cos((i + 3) * scale * 0.37f) * 0.25f + offset;
    tensor.SetElementsAsFloat(values);
}

static float[] ScaledDotProductAttentionPrefix(float[,,,] q, float[,,,] k, float[,,,] v, float scale, int prefixLength)
{
    int batch = q.GetLength(0);
    int seqQ = q.GetLength(1);
    int heads = q.GetLength(2);
    int keyDim = q.GetLength(3);
    int seqK = k.GetLength(1);
    int valueDim = v.GetLength(3);
    float[] result = new float[prefixLength];

    for (int b = 0; b < batch; b++)
    {
        for (int tq = 0; tq < seqQ; tq++)
        {
            for (int h = 0; h < heads; h++)
            {
                float[] scores = new float[seqK];
                float max = float.NegativeInfinity;
                for (int tk = 0; tk < seqK; tk++)
                {
                    float dot = 0;
                    for (int d = 0; d < keyDim; d++)
                        dot += q[b, tq, h, d] * k[b, tk, h, d];
                    scores[tk] = dot * scale;
                    max = MathF.Max(max, scores[tk]);
                }

                float sum = 0;
                for (int tk = 0; tk < seqK; tk++)
                {
                    scores[tk] = MathF.Exp(scores[tk] - max);
                    sum += scores[tk];
                }

                for (int d = 0; d < valueDim; d++)
                {
                    int flat = ((b * seqQ + tq) * heads + h) * valueDim + d;
                    if (flat >= prefixLength)
                        return result;

                    float acc = 0;
                    for (int tk = 0; tk < seqK; tk++)
                        acc += scores[tk] / sum * v[b, tk, h, d];
                    result[flat] = acc;
                }
            }
        }
    }

    return result;
}

internal readonly record struct BenchCase(string Name, int Rows, int InDim, int OutDim);
