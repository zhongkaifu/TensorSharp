// Focused reproduction for the "garbled output after a few hundred tokens" bug
// on the CUDA backend with Gemma-style sliding-window (circular) KV caches.
//
// Gemma 4 local-attention layers keep only the last `slidingWindow` positions in
// a circular cache: logical position p lives at slot (p % cacheSize). During
// decode the model calls CudaFusedOps.TryGqaDecodeAttention with circular: true,
// attendStart = max(0, pos+1 - window), attendLen = pos+1 - attendStart.
//
// While pos < cacheSize there is no wrap and everything matches. Once pos crosses
// cacheSize the read window straddles the wrap point. This test sweeps pos across
// several wrap boundaries and compares the CUDA kernel against an independent CPU
// reference, isolating the kernel from the full model.
using System;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class CudaDecodeAttentionWrapTests
{
    private readonly ITestOutputHelper _output;

    public CudaDecodeAttentionWrapTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(4, 2, 8, 16)]    // small, exercises wrap quickly
    [InlineData(8, 1, 16, 64)]   // MQA-ish
    [InlineData(8, 4, 32, 128)]  // closer to a real local layer shape
    public void CircularDecodeAttention_MatchesCpuAcrossWrapBoundary(
        int numQHeads, int numKVHeads, int headDim, int cacheSize)
    {
        if (!CudaBackend.IsAvailable())
        {
            _output.WriteLine("[wrap] CUDA unavailable; skipping.");
            return;
        }

        using var allocator = new CudaAllocator();

        int window = cacheSize;                  // local layer attends the whole window
        float scale = 1.0f / MathF.Sqrt(headDim);

        // Deterministic per-(logical position, kv head, dim) cache content.
        static float KVal(int pos, int h, int d) => MathF.Sin(0.013f * pos + 0.31f * h + 0.07f * d);
        static float VVal(int pos, int h, int d) => MathF.Cos(0.017f * pos + 0.23f * h + 0.05f * d);
        static float QVal(int pos, int qh, int d) => MathF.Sin(0.019f * pos + 0.11f * qh - 0.03f * d);

        int group = numQHeads / numKVHeads;

        // One persistent circular cache, written exactly like the model does.
        using var keyCache = new Tensor(allocator, DType.Float32, numKVHeads, cacheSize, headDim);
        using var valueCache = new Tensor(allocator, DType.Float32, numKVHeads, cacheSize, headDim);
        var kFlat = new float[numKVHeads * cacheSize * headDim];
        var vFlat = new float[numKVHeads * cacheSize * headDim];

        int maxPos = 3 * cacheSize + 5;          // cross several wrap boundaries
        double worstErr = 0;
        int worstPos = -1;

        for (int pos = 0; pos <= maxPos; pos++)
        {
            // Write position `pos` into its circular slot (mirrors CopyToCacheDecode
            // with cachePos = startPos % cacheSize).
            int slot = pos % cacheSize;
            for (int h = 0; h < numKVHeads; h++)
            {
                int baseIdx = (h * cacheSize + slot) * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    kFlat[baseIdx + d] = KVal(pos, h, d);
                    vFlat[baseIdx + d] = VVal(pos, h, d);
                }
            }
            keyCache.SetElementsAsFloat(kFlat);
            valueCache.SetElementsAsFloat(vFlat);

            // Query for this step.
            var qFlat = new float[numQHeads * headDim];
            for (int qh = 0; qh < numQHeads; qh++)
                for (int d = 0; d < headDim; d++)
                    qFlat[qh * headDim + d] = QVal(pos, qh, d);
            using var query = new Tensor(allocator, DType.Float32, 1, numQHeads * headDim);
            query.SetElementsAsFloat(qFlat);

            int attendLen = Math.Min(pos + 1, window);
            int attendStart = Math.Max(0, pos + 1 - attendLen);

            using var result = new Tensor(allocator, DType.Float32, 1, numQHeads * headDim);
            bool ok = CudaFusedOps.TryGqaDecodeAttention(
                result, query, keyCache, valueCache,
                numQHeads, numKVHeads, headDim,
                attendStart, pos + 1 - attendStart, cacheSize, circular: true, scale);
            Assert.True(ok, $"TryGqaDecodeAttention declined at pos={pos}");

            float[] got = result.GetElementsAsFloat(numQHeads * headDim);

            // CPU reference over the attended logical window.
            for (int qh = 0; qh < numQHeads; qh++)
            {
                int kvh = qh / group;
                var scores = new float[attendLen];
                float maxS = float.NegativeInfinity;
                for (int t = 0; t < attendLen; t++)
                {
                    int logical = attendStart + t;
                    // Slot (logical % cacheSize) currently holds logical position
                    // `logical` (its most recent occupant), so the cached K there is KVal(logical,...).
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                        dot += QVal(pos, qh, d) * KVal(logical, kvh, d);
                    scores[t] = dot * scale;
                    if (scores[t] > maxS) maxS = scores[t];
                }
                float sum = 0f;
                for (int t = 0; t < attendLen; t++) { scores[t] = MathF.Exp(scores[t] - maxS); sum += scores[t]; }
                float inv = sum > 0 ? 1f / sum : 0f;
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0f;
                    for (int t = 0; t < attendLen; t++)
                    {
                        int logical = attendStart + t;
                        acc += scores[t] * inv * VVal(logical, kvh, d);
                    }
                    double err = Math.Abs(acc - got[qh * headDim + d]);
                    if (err > worstErr) { worstErr = err; worstPos = pos; }
                }
            }
        }

        _output.WriteLine($"[wrap] heads={numQHeads}/{numKVHeads} hd={headDim} cache={cacheSize} maxErr={worstErr:E3} @pos={worstPos}");
        Assert.True(worstErr < 1e-3, $"CUDA circular decode attention diverged from CPU: maxErr={worstErr:E3} at pos={worstPos}");
    }

    // Same wrap sweep, but the KV cache is Float16 (the default for q4_0 models)
    // and is populated through the real CUDA write kernel (TryCopyHeadFirstToCache),
    // exactly as Gemma4Model.CopyToCacheDecode does. This exercises the F16 write
    // and F16 read kernels together across the wrap boundary.
    [Theory]
    [InlineData(4, 2, 8, 16)]
    [InlineData(8, 4, 32, 128)]
    public void CircularDecodeAttentionF16_MatchesCpuAcrossWrapBoundary(
        int numQHeads, int numKVHeads, int headDim, int cacheSize)
    {
        if (!CudaBackend.IsAvailable())
        {
            _output.WriteLine("[wrap-f16] CUDA unavailable; skipping.");
            return;
        }

        using var allocator = new CudaAllocator();
        int window = cacheSize;
        float scale = 1.0f / MathF.Sqrt(headDim);

        static float KVal(int pos, int h, int d) => MathF.Sin(0.013f * pos + 0.31f * h + 0.07f * d);
        static float VVal(int pos, int h, int d) => MathF.Cos(0.017f * pos + 0.23f * h + 0.05f * d);
        static float QVal(int pos, int qh, int d) => MathF.Sin(0.019f * pos + 0.11f * qh - 0.03f * d);
        // The cache stores Float16, so the reference must read back the rounded value.
        static float H(float v) => (float)(System.Half)v;

        int group = numQHeads / numKVHeads;

        using var keyCache = new Tensor(allocator, DType.Float16, numKVHeads, cacheSize, headDim);
        using var valueCache = new Tensor(allocator, DType.Float16, numKVHeads, cacheSize, headDim);

        int maxPos = 3 * cacheSize + 5;
        double worstErr = 0;
        int worstPos = -1;

        for (int pos = 0; pos <= maxPos; pos++)
        {
            // Write position `pos` head-first into its circular slot via the CUDA kernel.
            var kHeadFlat = new float[numKVHeads * headDim];
            var vHeadFlat = new float[numKVHeads * headDim];
            for (int h = 0; h < numKVHeads; h++)
                for (int d = 0; d < headDim; d++)
                {
                    kHeadFlat[h * headDim + d] = KVal(pos, h, d);
                    vHeadFlat[h * headDim + d] = VVal(pos, h, d);
                }
            using var kHead = new Tensor(allocator, DType.Float32, numKVHeads, 1, headDim);
            using var vHead = new Tensor(allocator, DType.Float32, numKVHeads, 1, headDim);
            kHead.SetElementsAsFloat(kHeadFlat);
            vHead.SetElementsAsFloat(vHeadFlat);
            int slot = pos % cacheSize;
            Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(keyCache, kHead, slot, 1, cacheSize, false),
                $"TryCopyHeadFirstToCache (K) declined at pos={pos}");
            Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(valueCache, vHead, slot, 1, cacheSize, false),
                $"TryCopyHeadFirstToCache (V) declined at pos={pos}");

            var qFlat = new float[numQHeads * headDim];
            for (int qh = 0; qh < numQHeads; qh++)
                for (int d = 0; d < headDim; d++)
                    qFlat[qh * headDim + d] = QVal(pos, qh, d);
            using var query = new Tensor(allocator, DType.Float32, 1, numQHeads * headDim);
            query.SetElementsAsFloat(qFlat);

            int attendLen = Math.Min(pos + 1, window);
            int attendStart = Math.Max(0, pos + 1 - attendLen);

            using var result = new Tensor(allocator, DType.Float32, 1, numQHeads * headDim);
            Assert.True(CudaFusedOps.TryGqaDecodeAttention(
                result, query, keyCache, valueCache,
                numQHeads, numKVHeads, headDim,
                attendStart, pos + 1 - attendStart, cacheSize, circular: true, scale),
                $"TryGqaDecodeAttention declined at pos={pos}");
            float[] got = result.GetElementsAsFloat(numQHeads * headDim);

            for (int qh = 0; qh < numQHeads; qh++)
            {
                int kvh = qh / group;
                var scores = new float[attendLen];
                float maxS = float.NegativeInfinity;
                for (int t = 0; t < attendLen; t++)
                {
                    int logical = attendStart + t;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                        dot += QVal(pos, qh, d) * H(KVal(logical, kvh, d));
                    scores[t] = dot * scale;
                    if (scores[t] > maxS) maxS = scores[t];
                }
                float sum = 0f;
                for (int t = 0; t < attendLen; t++) { scores[t] = MathF.Exp(scores[t] - maxS); sum += scores[t]; }
                float inv = sum > 0 ? 1f / sum : 0f;
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0f;
                    for (int t = 0; t < attendLen; t++)
                        acc += scores[t] * inv * H(VVal(attendStart + t, kvh, d));
                    double err = Math.Abs(acc - got[qh * headDim + d]);
                    if (err > worstErr) { worstErr = err; worstPos = pos; }
                }
            }
        }

        _output.WriteLine($"[wrap-f16] heads={numQHeads}/{numKVHeads} hd={headDim} cache={cacheSize} maxErr={worstErr:E3} @pos={worstPos}");
        // F16 rounding inflates the tolerance vs the F32 case, but a wrap/indexing
        // bug attends entirely wrong positions and blows well past this.
        Assert.True(worstErr < 5e-3, $"CUDA F16 circular decode attention diverged from CPU: maxErr={worstErr:E3} at pos={worstPos}");
    }

    // Global (non-sliding) layer path: circular=false, attendStart=0, and
    // attendLen == total sequence length, which grows every decode step. This is
    // the case that crosses blockDim (256 -> multiple score elements per thread)
    // and the partitioned-attention threshold (2048). Sweeps attendLen across all
    // those boundaries and compares CUDA against a CPU reference.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]   // true => Float16 cache (the q4_0 model default)
    public void GlobalDecodeAttention_MatchesCpuAsSequenceGrows(bool halfCache)
    {
        if (!CudaBackend.IsAvailable())
        {
            _output.WriteLine("[global] CUDA unavailable; skipping.");
            return;
        }

        using var allocator = new CudaAllocator();
        int numQHeads = 8, numKVHeads = 2, headDim = 32;
        int cacheSize = 4096;                 // big enough to hold every position
        float scale = 1.0f / MathF.Sqrt(headDim);
        int group = numQHeads / numKVHeads;

        static float KVal(int pos, int h, int d) => MathF.Sin(0.0021f * pos + 0.31f * h + 0.07f * d);
        static float VVal(int pos, int h, int d) => MathF.Cos(0.0017f * pos + 0.23f * h + 0.05f * d);
        static float QVal(int qh, int d) => MathF.Sin(0.11f * qh - 0.03f * d);
        float Q(float v) => halfCache ? (float)(System.Half)v : v;

        DType cacheDtype = halfCache ? DType.Float16 : DType.Float32;
        using var keyCache = new Tensor(allocator, cacheDtype, numKVHeads, cacheSize, headDim);
        using var valueCache = new Tensor(allocator, cacheDtype, numKVHeads, cacheSize, headDim);

        // Populate every position 0..cacheSize-1 at its absolute slot via the real
        // CUDA write kernel (or directly for the F32 case).
        if (halfCache)
        {
            // Write in head-first blocks of 1 position at a time through the kernel.
            for (int pos = 0; pos < cacheSize; pos++)
            {
                var kf = new float[numKVHeads * headDim];
                var vf = new float[numKVHeads * headDim];
                for (int h = 0; h < numKVHeads; h++)
                    for (int d = 0; d < headDim; d++)
                    {
                        kf[h * headDim + d] = KVal(pos, h, d);
                        vf[h * headDim + d] = VVal(pos, h, d);
                    }
                using var kHead = new Tensor(allocator, DType.Float32, numKVHeads, 1, headDim);
                using var vHead = new Tensor(allocator, DType.Float32, numKVHeads, 1, headDim);
                kHead.SetElementsAsFloat(kf);
                vHead.SetElementsAsFloat(vf);
                Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(keyCache, kHead, pos, 1, cacheSize, false));
                Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(valueCache, vHead, pos, 1, cacheSize, false));
            }
        }
        else
        {
            var kFlat = new float[numKVHeads * cacheSize * headDim];
            var vFlat = new float[numKVHeads * cacheSize * headDim];
            for (int h = 0; h < numKVHeads; h++)
                for (int pos = 0; pos < cacheSize; pos++)
                    for (int d = 0; d < headDim; d++)
                    {
                        int idx = (h * cacheSize + pos) * headDim + d;
                        kFlat[idx] = KVal(pos, h, d);
                        vFlat[idx] = VVal(pos, h, d);
                    }
            keyCache.SetElementsAsFloat(kFlat);
            valueCache.SetElementsAsFloat(vFlat);
        }

        var qFlat = new float[numQHeads * headDim];
        for (int qh = 0; qh < numQHeads; qh++)
            for (int d = 0; d < headDim; d++)
                qFlat[qh * headDim + d] = QVal(qh, d);
        using var query = new Tensor(allocator, DType.Float32, 1, numQHeads * headDim);
        query.SetElementsAsFloat(qFlat);

        int[] lengths = { 1, 64, 200, 255, 256, 257, 300, 511, 512, 1000, 2047, 2048, 2049, 3000, 4096 };
        double worstErr = 0; int worstLen = -1;

        foreach (int len in lengths)
        {
            using var result = new Tensor(allocator, DType.Float32, 1, numQHeads * headDim);
            Assert.True(CudaFusedOps.TryGqaDecodeAttention(
                result, query, keyCache, valueCache,
                numQHeads, numKVHeads, headDim,
                attendStart: 0, attendLen: len, cacheSize, circular: false, scale),
                $"TryGqaDecodeAttention declined at len={len}");
            float[] got = result.GetElementsAsFloat(numQHeads * headDim);

            double lenErr = 0;
            for (int qh = 0; qh < numQHeads; qh++)
            {
                int kvh = qh / group;
                var scores = new float[len];
                float maxS = float.NegativeInfinity;
                for (int t = 0; t < len; t++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                        dot += QVal(qh, d) * Q(KVal(t, kvh, d));
                    scores[t] = dot * scale;
                    if (scores[t] > maxS) maxS = scores[t];
                }
                float sum = 0f;
                for (int t = 0; t < len; t++) { scores[t] = MathF.Exp(scores[t] - maxS); sum += scores[t]; }
                float inv = sum > 0 ? 1f / sum : 0f;
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0f;
                    for (int t = 0; t < len; t++)
                        acc += scores[t] * inv * Q(VVal(t, kvh, d));
                    double err = Math.Abs(acc - got[qh * headDim + d]);
                    if (err > lenErr) lenErr = err;
                }
            }
            _output.WriteLine($"[global half={halfCache}] len={len,5} err={lenErr:E3}");
            if (lenErr > worstErr) { worstErr = lenErr; worstLen = len; }
        }

        double tol = halfCache ? 5e-3 : 1e-3;
        Assert.True(worstErr < tol, $"CUDA global decode attention diverged from CPU: maxErr={worstErr:E3} at len={worstLen}");
    }
}
