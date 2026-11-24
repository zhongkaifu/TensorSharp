// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>
/// The Qwen3-VL tower on GGML CUDA as it serves: one merged graph with the DeepStack mergers, ggml's vision rope, the
/// tensor-core vision attention and the F16 linears (tsg_linear_f16: cuBLASLt GEMMs with the bias, tanh GELU and
/// residual in the epilogue), and video pairs stacked in one graph. Against the managed CPU tower on the same pixels,
/// every column block ([main | tap 0 | tap 1 | tap 2]) stays inside the 3e-3 relative L2 the vision-embedding
/// comparison gates on. TENSORSHARP_QWEN3VL_MMPROJ names the projector (or a directory holding it).
/// </summary>
public sealed class Qwen3VLTowerCudaTests
{
    private const int Height = 256, Width = 448;   // 448 patches a frame: quick on the CPU reference

    [ModelFact("TENSORSHARP_QWEN3VL_MMPROJ", GgmlBackend = BackendType.GgmlCuda)]
    public void CudaTowerMatchesTheManagedTower()
    {
        string path = Environment.GetEnvironmentVariable("TENSORSHARP_QWEN3VL_MMPROJ")!;
        if (Directory.Exists(path)) path = TestGates.FindGguf(path, "mmproj")!;
        float[] a = Pixels(1), b = Pixels(2), c = Pixels(3), d = Pixels(4);

        using var cuda = new Qwen35VisionEncoder(path, new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Cuda), 0),
            mergerGeluErf: true) { ConcatenateDeepStack = true };
        using var cpu = new Qwen35VisionEncoder(path, new CpuAllocator(BlasEnum.DotNet), mergerGeluErf: true)
            { ConcatenateDeepStack = true };
        Assert.True(cuda.DeepStackCount > 0, "a Qwen3-VL projector carries DeepStack mergers");

        Tensor[] pairs = cuda.EncodeVideoPairs(new List<float[]> { a, c }, new List<float[]> { b, d }, Height, Width);
        try
        {
            using Tensor still = cuda.Encode(a, Height, Width);
            using Tensor stillReference = cpu.Encode(a, Height, Width);
            AssertClose(Values(still), Values(stillReference), (int)still.Sizes[1], cuda.ProjectionDim, "still image");
            using Tensor pairReference = cpu.Encode(c, d, Height, Width);
            AssertClose(Values(pairs[1]), Values(pairReference), (int)pairs[1].Sizes[1], cuda.ProjectionDim, "second stacked pair");
        }
        finally
        {
            foreach (Tensor t in pairs) t.Dispose();
        }
    }

    private static void AssertClose(float[] actual, float[] expected, int width, int projection, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        int rows = expected.Length / width;
        for (int block = 0; block < width / projection; block++)
        {
            double diff = 0, norm = 0;
            for (int r = 0; r < rows; r++)
                for (int k = 0; k < projection; k++)
                {
                    int i = r * width + block * projection + k;
                    diff += (actual[i] - expected[i]) * (double)(actual[i] - expected[i]);
                    norm += expected[i] * (double)expected[i];
                }
            double rel = Math.Sqrt(diff / norm);
            Assert.True(rel < 3e-3, $"{what}, column block {block}: relative L2 {rel:E2} against the managed tower");
        }
    }

    private static float[] Pixels(int seed)
    {
        var data = new float[3 * Height * Width];
        for (int ch = 0; ch < 3; ch++)
            for (int y = 0; y < Height; y++)
                for (int x = 0; x < Width; x++)
                    data[(ch * Height + y) * Width + x] =
                        (float)(0.6 * Math.Sin(0.05 * seed * x + 0.03 * y + ch) + 0.3 * Math.Cos(0.11 * y * seed - 0.07 * x));
        return data;
    }

    private static float[] Values(Tensor t)
    {
        using var contiguous = t.IsContiguous() ? null : Ops.NewContiguous(t);
        return (contiguous ?? t).GetElementsAsFloat((int)t.ElementCount());
    }
}
