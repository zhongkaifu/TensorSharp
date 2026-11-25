// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.GGML;
using TensorSharp.Models.QwenImage;

namespace InferenceWeb.Tests;

/// <summary>
/// The Qwen-Image-2.1 per-conv VAE path (Metal and CPU; the fused VAE graph is CUDA-only)
/// splits a device convolution into horizontal output bands once its im2col estimate
/// exceeds the im2col budget (1 GiB, which 2048² convolutions do).
/// Each band re-runs the same convolution on a manually zero-padded input slice, so the
/// result must match the un-tiled device convolution and the scalar reference.
/// Runs on the GGML backend this test process pins (TS_TEST_GGML_BACKEND, default cpu).
/// </summary>
public sealed class QwenVaeConvTilingTests
{
    [Theory]
    [InlineData(8, 6, 3, 3, 1, 1, 1, 1, 1, 1, 37, 41)]   // standard pad-1 stride-1
    [InlineData(8, 6, 3, 3, 2, 2, 0, 1, 0, 1, 38, 40)]   // encoder downsample: asymmetric pad, stride 2
    [InlineData(8, 6, 1, 1, 1, 1, 0, 0, 0, 0, 37, 41)]   // 1x1 projection
    [InlineData(4, 5, 3, 3, 1, 1, 1, 1, 1, 1, 9, 64)]    // wide, few rows
    public void BandTiledDeviceConvolutionMatchesTheUntiledConvolution(
        int ic, int oc, int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr, int h, int w)
    {
        GgmlBasicOps.EnsureBackendAvailable(TestGates.PinnedGgmlBackendType);
        var rng = new Random(1234);
        var x = new Feature(ic, h, w);
        for (int i = 0; i < x.D.Length; i++) x.D[i] = (float)(rng.NextDouble() * 2 - 1);
        var weight = new float[oc * ic * kh * kw];
        for (int i = 0; i < weight.Length; i++) weight[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        var bias = new float[oc];
        for (int i = 0; i < bias.Length; i++) bias[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        int ho = (h + pt + pb - kh) / sh + 1, wo = (w + pl + pr - kw) / sw + 1;

        try
        {
            // The direct scalar loop, not Conv2d: off the device Conv2d now runs the packed-GEMM
            // convolution (VaeCpuOps), which is a third implementation rather than the oracle.
            Feature reference = VaeReferenceMath.Conv2dScalar(x, weight, oc, ic, kh, kw, bias, sh, sw, pt, pb, pl, pr);

            // TryGpuConv2dMaybeTiled never falls back to the managed loop: false means a
            // device call failed, so a true result proves both paths ran on the device.
            VaeReferenceMath.Im2colBudgetForTest = 999999999999;
            Assert.True(VaeReferenceMath.TryGpuConv2dMaybeTiled(x, weight, oc, ic, kh, kw, bias,
                sh, sw, pt, pb, pl, pr, ho, wo, out Feature full), GgmlBasicOps.LastNativeError("untiled device convolution failed"));

            // 256 bytes is below one output row's im2col, so every band is a single row.
            Assert.True((long)ic * kh * kw * wo * sizeof(float) > 256);
            VaeReferenceMath.Im2colBudgetForTest = 256;
            Assert.True(VaeReferenceMath.TryGpuConv2dMaybeTiled(x, weight, oc, ic, kh, kw, bias,
                sh, sw, pt, pb, pl, pr, ho, wo, out Feature tiled), GgmlBasicOps.LastNativeError("band-tiled device convolution failed"));

            Assert.Equal((oc, ho, wo), (full.C, full.H, full.W));
            Assert.Equal((oc, ho, wo), (tiled.C, tiled.H, tiled.W));
            Assert.Equal((oc, ho, wo), (reference.C, reference.H, reference.W));
            Assert.All(tiled.D, v => Assert.True(float.IsFinite(v)));
            // CPU (F32 GEMM) and Metal (direct F32 conv) differ only in reduction order.
            // ggml-cuda lowers these shapes (K not a multiple of 32, OC > 3) to cublasSgemm on
            // a TF32 tensor-op handle, whose 2^-11 input rounding alone gives relL2 of a few
            // 1e-4, and cuBLAS may pick different kernels for a one-row band and the whole
            // output; Vulkan's F32 matmul may likewise use reduced-precision operands.
            double tolerance = TestGates.PinnedGgmlBackendType is GgmlBackendType.Cuda or GgmlBackendType.Vulkan
                ? 2e-3 : 1e-4;
            Assert.True(RelL2(full.D, tiled.D) < tolerance, $"tiled vs untiled relL2={RelL2(full.D, tiled.D):E3}");
            Assert.True(RelL2(reference.D, full.D) < tolerance, $"device vs scalar relL2={RelL2(reference.D, full.D):E3}");
        }
        finally
        {
            VaeReferenceMath.Im2colBudgetForTest = 0;
        }
    }

    private static double RelL2(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        double num = 0, den = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = expected[i] - actual[i];
            num += d * d;
            den += (double)expected[i] * expected[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-12));
    }
}
