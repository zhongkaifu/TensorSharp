// Direct test of the global KV-cache growth copy used by Gemma4Model.EnsureCacheCapacity:
//
//   using var srcK = _kvCacheK[l].Narrow(1, 0, _cacheSeqLen);
//   using var dstK = newK.Narrow(1, 0, _cacheSeqLen);
//   Ops.Copy(dstK, srcK);
//
// Both operands are strided views of [heads, capacity, headDim] tensors with
// DIFFERENT dim-0 strides (old capacity vs new capacity). For CUDA this lands on
// the strided device-copy path. If that copy is wrong, every global-attention
// read after the first resize (at seq pos 2048 on CUDA) sees scrambled history —
// which would garble generation right when the sequence crosses the cap.
using System;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class CudaKvCacheResizeTests
{
    private readonly ITestOutputHelper _output;
    public CudaKvCacheResizeTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(8, 2048, 4096, 32, 2000)]   // the real first-grow shape (cross 2048)
    [InlineData(4, 64, 128, 16, 50)]        // tiny
    public void ResizeCopy_PreservesStridedHistory(int heads, int oldCap, int newCap, int headDim, int seqLen)
    {
        if (!CudaBackend.IsAvailable())
        {
            _output.WriteLine("[resize] CUDA unavailable; skipping.");
            return;
        }

        using var allocator = new CudaAllocator();

        static float KVal(int pos, int h, int d) => MathF.Sin(0.0021f * pos + 0.31f * h + 0.07f * d);

        using var oldK = new Tensor(allocator, DType.Float32, heads, oldCap, headDim);
        using var newK = new Tensor(allocator, DType.Float32, heads, newCap, headDim);

        var oldFlat = new float[heads * oldCap * headDim];
        for (int h = 0; h < heads; h++)
            for (int p = 0; p < oldCap; p++)
                for (int d = 0; d < headDim; d++)
                    oldFlat[(h * oldCap + p) * headDim + d] = p < seqLen ? KVal(p, h, d) : -123456f;
        oldK.SetElementsAsFloat(oldFlat);

        // Sentinel-fill the destination so we can tell copied vs untouched.
        var newFlat = new float[heads * newCap * headDim];
        Array.Fill(newFlat, -987654f);
        newK.SetElementsAsFloat(newFlat);

        using (var srcK = oldK.Narrow(1, 0, seqLen))
        using (var dstK = newK.Narrow(1, 0, seqLen))
            Ops.Copy(dstK, srcK);

        float[] got = newK.GetElementsAsFloat(heads * newCap * headDim);

        double worst = 0; int badH = -1, badP = -1, badD = -1;
        for (int h = 0; h < heads; h++)
            for (int p = 0; p < seqLen; p++)
                for (int d = 0; d < headDim; d++)
                {
                    float expected = KVal(p, h, d);
                    float actual = got[(h * newCap + p) * headDim + d];
                    double err = Math.Abs(expected - actual);
                    if (err > worst) { worst = err; badH = h; badP = p; badD = d; }
                }

        _output.WriteLine($"[resize] heads={heads} {oldCap}->{newCap} hd={headDim} seqLen={seqLen} worstErr={worst:E3} at h={badH},p={badP},d={badD}");
        Assert.True(worst < 1e-5, $"Resize copy corrupted history: worstErr={worst:E3} at h={badH} pos={badP} d={badD}");
    }
}
