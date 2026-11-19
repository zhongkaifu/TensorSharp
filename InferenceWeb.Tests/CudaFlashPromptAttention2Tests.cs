// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The prompt kernels with F32 accumulation (ts_flash_prompt2_*, CudaFlashPrefillKernels.LaunchPromptAttention), the CUDA executor's
/// 16-bit prompt attention from flash prefill v13 on where the tensor cores accumulate F32 at full rate: against a double-precision
/// host reference over the same F16 cache they are closer than the 16-bit-run prompt kernels they replace (ts_flash_prompt_*,
/// LaunchAttention's prompt) on Gaussian, peaked and outlier-channel data, with a floor, fresh causal chunks, few rows over a long
/// window, and the last wave's tiles split over their keys; the executor takes them for a prompt chunk by default there and the
/// 16-bit-run ones under its seam.
/// </summary>
public sealed class CudaFlashPromptAttention2Tests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Directory.CreateTempSubdirectory("prompt-attn2").FullName;

    public CudaFlashPromptAttention2Tests(ITestOutputHelper output) => _output = output;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

    private static double Gaussian(Random rng) => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());

    /// <summary>
    /// "gauss": q, K, V standard normal; "peaked": q four times wider (a sharp softmax); "outlier": a K channel in 64 twenty times
    /// wider, a V channel in 64 offset by 30 (the accumulators' magnitude far past each tile's product), q's matching channel a
    /// quarter. K and V are values an F16 cache holds exactly. <paramref name="sms"/> &gt; 0 plans the grid as on a device of that
    /// many multiprocessors (a tail split wherever this one's count would leave none).
    /// </summary>
    [CudaTheory]
    [InlineData("gauss", 256, 24, 4, 300, 700, 33, 0)]       // Qwen3.8-27B's heads, keys below a floor masked
    [InlineData("peaked", 256, 24, 4, 300, 700, 33, 0)]
    [InlineData("outlier", 256, 24, 4, 300, 700, 33, 0)]
    [InlineData("gauss", 256, 24, 4, 600, 4000, 0, 23)]      // the last wave's tiles split over their keys
    [InlineData("outlier", 256, 24, 4, 600, 4000, 0, 23)]
    [InlineData("outlier", 256, 8, 1, 40, 12000, 0, 0)]      // a long window: each tile's product flushed into the accumulators
    [InlineData("peaked", 256, 24, 4, 1100, 1100, 0, 0)]     // a fresh chunk: the causal triangle
    [InlineData("gauss", 128, 16, 2, 300, 700, 33, 0)]       // 128-wide heads
    [InlineData("outlier", 128, 32, 8, 1024, 2048, 0, 0)]
    public void CloserToAnExactReferenceThanThe16BitRunKernel(string dist, int d, int heads, int kvHeads, int nq, int len, int floor, int sms)
    {
        using var allocator = new CudaAllocator();
        CudaFlashPrefillKernels flash = allocator.FlashPrefillKernels;
        if (flash == null || !flash.RunsAttention(d))
        {
            _output.WriteLine("no flash prefill attention for this GPU");
            return;
        }
        int cap = len + 29, group = heads / kvHeads;
        var rng = new Random(d * 131 + nq * 7 + len + dist.Length);
        float qScale = dist == "peaked" ? 4f : 1f;
        var qv = new float[heads * nq * d];
        for (int i = 0; i < qv.Length; i++)
            qv[i] = (float)Gaussian(rng) * qScale * (dist == "outlier" && i % d % 64 == 3 ? 0.25f : 1f);
        var kh = new float[kvHeads * cap * d];
        var vh = new float[kvHeads * cap * d];
        for (int i = 0; i < kh.Length; i++)
        {
            int x = i % d;
            double k = Gaussian(rng), v = Gaussian(rng);
            if (dist == "outlier")
            {
                if (x % 64 == 3) k *= 20;
                if (x % 64 == 7) v = v * 4 + 30;
            }
            kh[i] = (float)(System.Half)k;
            vh[i] = (float)(System.Half)v;
        }
        float scale = 1f / MathF.Sqrt(d);

        using var q = Tensor.FromArray(allocator, qv);
        using var k32 = Tensor.FromArray(allocator, kh);
        using var v32 = Tensor.FromArray(allocator, vh);
        using var k16 = new Tensor(allocator, DType.Float16, kh.Length);
        using var v16 = new Tensor(allocator, DType.Float16, vh.Length);
        using var legacyOut = new Tensor(allocator, DType.Float32, (long)nq * heads * d);
        using var newOut = new Tensor(allocator, DType.Float32, (long)nq * heads * d);
        int splits = flash.Splits(nq * group, kvHeads, len);
        using Tensor partial = splits > 1 ? new Tensor(allocator, DType.Float32, CudaFlashPrefillKernels.PartialFloats(nq, heads, d, splits)) : null;
        int keys = Math.Max(1, len - nq + 1 - floor);
        var plan = sms > 0 ? CudaFlashPrefillKernels.PromptPlan(nq * group, kvHeads, keys, sms) : flash.PromptPlan(nq * group, kvHeads, keys);
        if (sms > 0) Assert.True(plan.TailSplits > 1, $"the plan {plan} splits no tail tile");
        long tailFloats = CudaFlashPrefillKernels.PromptPartialFloats(nq * group, kvHeads, d, plan);
        using Tensor tail = tailFloats > 0 ? new Tensor(allocator, DType.Float32, tailFloats) : null;
        foreach (Tensor t in new[] { q, k32, v32, k16, v16, legacyOut, newOut, partial, tail })
            if (t != null) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        IntPtr stream = allocator.Stream.Handle;
        allocator.Context.MakeCurrent();
        allocator.Kernels.LaunchConvertF32F16(P(k32), P(k16), kh.Length, stream);
        allocator.Kernels.LaunchConvertF32F16(P(v32), P(v16), vh.Length, stream);
        flash.LaunchAttention(P(q), P(k16), P(v16), cap, nq, len, floor, heads, kvHeads, d, scale, splits,
            partial == null ? IntPtr.Zero : P(partial), P(legacyOut), stream, prompt: true);
        flash.LaunchPromptAttention(P(q), P(k16), P(v16), cap, nq, len, floor, heads, kvHeads, d, scale, plan,
            tail == null ? IntPtr.Zero : P(tail), P(newOut), stream);
        allocator.Stream.Synchronize();
        foreach (Tensor t in new[] { legacyOut, newOut }) ((CudaStorage)t.Storage).MarkDeviceModified();
        float[] legacy = legacyOut.GetElementsAsFloat(nq * heads * d), got = newOut.GetElementsAsFloat(nq * heads * d);

        // The reference at sampled rows: the first four queries (the lightest row tiles, a split tail's), the last four and 56
        // others, each with a random head.
        var rows = Enumerable.Range(0, Math.Min(4, nq)).Concat(Enumerable.Range(Math.Max(0, nq - 4), Math.Min(4, nq)))
            .Concat(Enumerable.Range(0, 56).Select(_ => rng.Next(nq))).Select(t => (T: t, H: rng.Next(heads))).ToArray();
        double numLegacy = 0, numNew = 0, den = 0, maxNew = 0;
        var w = new double[len];
        foreach (var (t, h) in rows)
        {
            int kvh = h / group, end = len - nq + t;
            double mx = double.NegativeInfinity;
            for (int j = floor; j <= end; j++)
            {
                double dot = 0;
                for (int x = 0; x < d; x++) dot += qv[(h * nq + t) * d + x] * (double)kh[(kvh * cap + j) * d + x];
                w[j] = dot * scale;
                mx = Math.Max(mx, w[j]);
            }
            double sum = 0;
            for (int j = floor; j <= end; j++) { w[j] = Math.Exp(w[j] - mx); sum += w[j]; }
            for (int x = 0; x < d; x++)
            {
                double want = 0;
                for (int j = floor; j <= end; j++) want += w[j] * vh[(kvh * cap + j) * d + x];
                want /= sum;
                int at = (t * heads + h) * d + x;
                numLegacy += (legacy[at] - want) * (legacy[at] - want);
                numNew += (got[at] - want) * (got[at] - want);
                den += want * want;
                maxNew = Math.Max(maxNew, Math.Abs(got[at] - want));
            }
        }
        double relLegacy = Math.Sqrt(numLegacy / den), relNew = Math.Sqrt(numNew / den);
        _output.WriteLine($"{dist}: d {d}, {heads}/{kvHeads} heads, {nq} rows over {len} keys (floor {floor}), plan {plan}: relative RMS " +
            $"{relNew:E3} (max |d| {maxNew:E3}) against the 16-bit-run kernel's {relLegacy:E3}; F32 accumulation at " +
            $"{flash.F32AccumulationRate:F2} of the F16 rate here");
        Assert.True(relNew <= relLegacy, $"relative RMS {relNew:E3} above the 16-bit-run kernel's {relLegacy:E3}");
        Assert.True(relNew <= 1e-3, $"relative RMS {relNew:E3}");
    }

    /// <summary>The executor's prompt chunks: by default the kernels with F32 accumulation where the tensor cores accumulate F32 at
    /// full rate (a 1,300-token prompt's last-row logits near the 16-bit-run kernels', not their bits; elsewhere their bits), the
    /// 16-bit-run kernels under CudaGraphExecutor.PromptAttentionLegacyForTests (their bits, run after run).</summary>
    [CudaFact]
    public void TheExecutorsPromptChunksTakeThemUnlessTheSeamAsksForThe16BitRunKernel()
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        string trunk = new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "trunk.gguf"));
        int[] tokens = Enumerable.Range(0, 1300).Select(i => 3 + (i * 37 + i / 7) % 250).ToArray();
        using var model = new Qwen35RuntimeModel(trunk, BackendType.Cuda);
        var executor = Assert.IsType<CudaGraphExecutor>(model.GraphExecutorForTests);
        float[] Run(bool legacy)
        {
            executor.PromptAttentionLegacyForTests = legacy;
            model.ResetKVCache();
            return (float[])model.ForwardPrompt(tokens).Clone();
        }
        CudaFlashPrefillKernels flash = ((CudaAllocator)model.DirectAllocatorForTelemetry).FlashPrefillKernels;
        float[] fresh = Run(false), legacyA = Run(true), legacyB = Run(true), again = Run(false);
        _output.WriteLine($"F32 accumulation at {flash.F32AccumulationRate:F2} of the F16-accumulation rate: prompt kernels with F32 " +
            $"accumulation {(flash.PromptAccumulatesInF32 ? "on" : "off")}");
        Assert.True(fresh.AsSpan().SequenceEqual(again), "two default runs differ");
        Assert.True(legacyA.AsSpan().SequenceEqual(legacyB), "two runs on the 16-bit-run kernels differ");
        if (!flash.PromptAccumulatesInF32)
        {
            Assert.True(fresh.AsSpan().SequenceEqual(legacyA), "a part without full-rate F32 accumulation left the 16-bit-run kernels");
            return;
        }
        Assert.False(fresh.AsSpan().SequenceEqual(legacyA), "the default run gave the 16-bit-run kernels' bits: the seam changes nothing");
        double diff = 0, norm = 0;
        for (int i = 0; i < fresh.Length; i++)
        {
            diff += ((double)fresh[i] - legacyA[i]) * ((double)fresh[i] - legacyA[i]);
            norm += (double)legacyA[i] * legacyA[i];
        }
        double rel = Math.Sqrt(diff / Math.Max(norm, 1e-30));
        _output.WriteLine($"1,300-token prompt: last-row logits rel L2 {rel:E3} between the default and the 16-bit-run prompt kernels");
        Assert.True(rel <= 1e-2, $"last-row logits rel L2 {rel:E3} between the default and the 16-bit-run prompt kernels");
    }
}
