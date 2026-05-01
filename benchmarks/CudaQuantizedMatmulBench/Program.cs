using System.Diagnostics;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;

const int GgmlTypeQ8_0 = 8;

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

RunBatchedAddmmCase(allocator, warmup, iterations);

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

internal readonly record struct BenchCase(string Name, int Rows, int InDim, int OutDim);
