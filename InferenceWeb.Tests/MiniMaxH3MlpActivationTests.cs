using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Direct;
using TensorSharp.Models.MiniMaxH3;

namespace InferenceWeb.Tests;

public sealed class MiniMaxH3MlpActivationTests
{
    [Fact]
    public void CpuActivationPreservesPreviousExpressionExactly()
    {
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        using var context = new DirectContext(allocator);
        Compare(context);
    }

    [CudaFact]
    public void CudaFusedSplitPreservesPreviousExpressionExactly()
    {
        using var allocator = new CudaAllocator();
        using var context = new DirectContext(allocator);
        Compare(context);
    }

    [MlxFact]
    public void MlxFusedSplitPreservesPreviousExpressionExactly() => MlxWorker.Shared.Invoke(() =>
    {
        using var allocator = new MlxAllocator();
        using var context = new DirectContext(allocator);
        Compare(context);
    });

    private static void Compare(DirectContext context)
    {
        var random = new Random(6831);
        float[] extremes = [0f, -0f, float.Epsilon, -float.Epsilon, 1e-30f, -1e-30f,
            .125f, -.125f, 1f, -1f, 80f, -80f, 1000f, -1000f, 80000f, -80000f,
            1e30f, -1e30f, float.MaxValue, float.MinValue];
        foreach (var (rows, width) in new[] { (1, 3), (3, 33), (17, 128), (129, 14336), (129, 8192) })
        {
            var values = new float[checked(rows * width * 2)];
            for (int row = 0; row < rows; row++)
            for (int column = 0; column < width; column++)
            {
                int offset = row * width * 2 + column;
                values[offset] = column < extremes.Length ? extremes[column]
                    : (float)(random.NextDouble() * 200 - 100);
                values[offset + width] = column < extremes.Length ? extremes[(column + row) % extremes.Length]
                    : (float)(random.NextDouble() * 200000 - 100000);
            }
            using var projected = context.FromFloats(values, rows, width * 2);
            using var gateView = projected.Narrow(1, 0, width);
            using var valueView = projected.Narrow(1, width, width);
            using var gate = Ops.NewContiguous(gateView);
            using var value = Ops.NewContiguous(valueView);
            using var previous = DirectOps.Silu(context, gate);
            Ops.Mul(previous, previous, value);
            float[] expected = DirectOps.ToArray(previous);

            using var fused = MiniMaxH3DirectVideoVae.ActivateMlp(context, projected, width);
            float[] actual = DirectOps.ToArray(fused);
            Assert.True(MemoryMarshal.AsBytes(expected.AsSpan()).SequenceEqual(
                MemoryMarshal.AsBytes(actual.AsSpan())),
                $"H3 SwiGLU changed float bits for [{rows}, {width}].");
            Assert.True(MemoryMarshal.AsBytes(values.AsSpan()).SequenceEqual(
                MemoryMarshal.AsBytes(DirectOps.ToArray(projected).AsSpan())),
                "H3 SwiGLU modified its packed projection input.");
        }
    }
}
