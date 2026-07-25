// Standalone microbench for the direct-CUDA flash prefill attention kernel
// (ts_gqa_prefill_flash_group4_*). Times CudaFusedOps.TryGqaPrefillAttention at
// the real Qwen3.5 / Gemma4 prefill shapes and verifies against a CPU reference,
// so the kernel can be tuned without a full model load. nsys-profilable (plain exe).
//
// Env knobs:
//   TS_FLASH_LENS   comma list of seqLen==kvLen prefill sizes (default 2048,3200)
//   TS_FLASH_ITERS  timed iterations per shape (default 30)
using System.Diagnostics;
using TensorSharp;
using TensorSharp.Cuda;

static int[] EnvIntList(string name, int[] fallback)
{
    string s = Environment.GetEnvironmentVariable(name);
    if (string.IsNullOrEmpty(s)) return fallback;
    var list = new List<int>();
    foreach (var p in s.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        if (int.TryParse(p, out int v) && v > 0) list.Add(v);
    return list.Count > 0 ? list.ToArray() : fallback;
}

if (!CudaBackend.IsAvailable())
{
    Console.WriteLine("CUDA not available.");
    return;
}

int[] lens = EnvIntList("TS_FLASH_LENS", new[] { 2048, 3200 });
int iters = EnvIntList("TS_FLASH_ITERS", new[] { 30 })[0];

// (name, numQHeads, numKVHeads, headDim, window)  — group ratio 4, real geometries.
var shapes = new (string name, int nq, int nkv, int hd, int win)[]
{
    ("qwen-full  d256 w0   ", 16, 4, 256, 0),   // Qwen3.5 full-attn layers
    ("gemma-local d256 w512", 8, 2, 256, 512),  // Gemma4 E4B SWA local layers
    ("gemma-glob  d512 w0   ", 8, 2, 512, 0),   // Gemma4 E4B global layers
};

Console.WriteLine($"[flash-bench] iters={iters}  lens={string.Join(",", lens)}");
Console.WriteLine($"{"shape",-24} {"len",6} {"ms",10} {"maxErr",12} {"route",8}");

foreach (var sh in shapes)
{
    foreach (int len in lens)
    {
        RunOne(sh.name, sh.nq, sh.nkv, sh.hd, sh.win, len, iters);
    }
}

static void RunOne(string name, int numQHeads, int numKVHeads, int headDim, int windowSize, int len, int iters)
{
    int seqLen = len;
    int kvLen = len;             // fresh full-prompt prefill: kv == seq, maskStart 0
    int cacheSize = len + 64;    // live-cache stride > logical prefix
    int maskStart = kvLen - seqLen; // == 0
    float scale = 1.0f / MathF.Sqrt(headDim);

    using var allocator = new CudaAllocator();
    var q = new float[numQHeads, seqLen, headDim];
    var k = new float[numKVHeads, kvLen, headDim];
    var v = new float[numKVHeads, kvLen, headDim];
    for (int h = 0; h < numQHeads; h++)
        for (int t = 0; t < seqLen; t++)
            for (int d = 0; d < headDim; d++)
                q[h, t, d] = MathF.Sin((h + 1) * 0.13f + (t + 2) * 0.017f + (d + 3) * 0.011f);
    for (int h = 0; h < numKVHeads; h++)
        for (int t = 0; t < kvLen; t++)
            for (int d = 0; d < headDim; d++)
            {
                k[h, t, d] = MathF.Cos((h + 2) * 0.19f + (t + 1) * 0.023f + (d + 1) * 0.007f);
                v[h, t, d] = MathF.Sin((h + 3) * 0.11f + (t + 2) * 0.029f + (d + 1) * 0.005f) * 0.5f;
            }

    using var qTensor = Tensor.FromArray(allocator, q);
    using var kActive = Tensor.FromArray(allocator, k);
    using var vActive = Tensor.FromArray(allocator, v);
    // f16 cache for Qwen/Gemma-global mirrors the model; f32 for Gemma-local.
    DType cacheType = windowSize == 0 ? DType.Float16 : DType.Float32;
    using var kCache = new Tensor(allocator, cacheType, numKVHeads, cacheSize, headDim);
    using var vCache = new Tensor(allocator, cacheType, numKVHeads, cacheSize, headDim);
    CudaFusedOps.TryCopyHeadFirstToCache(kCache, kActive, 0, kvLen, cacheSize, circular: false);
    CudaFusedOps.TryCopyHeadFirstToCache(vCache, vActive, 0, kvLen, cacheSize, circular: false);

    using var actual = new Tensor(allocator, DType.Float32, seqLen, numQHeads * headDim);

    // WithSinks(sinks=null) is the entry the models actually use (Qwen full-attn
    // and Gemma both reach flash through it) and routes d256/d512 at any window.
    bool ok = CudaFusedOps.TryGqaPrefillAttentionWithSinks(
        actual, qTensor, kCache, vCache, sinks: null,
        numQHeads, numKVHeads, headDim, seqLen, kvLen, cacheSize, maskStart, windowSize, scale);
    if (!ok)
    {
        Console.WriteLine($"{name,-24} {len,6} {"n/a",10} {"declined",12}");
        return;
    }

    // Correctness: CPU reference on a few sampled query rows (full ref over all
    // rows is O(seq*kv*hd) which is fine but slow; sample to keep it snappy).
    float maxErr = VerifySampled(actual, q, k, v, numQHeads, numKVHeads, seqLen, kvLen, headDim, maskStart, windowSize, scale);

    // Warmup + timed.
    for (int w = 0; w < 5; w++)
        CudaFusedOps.TryGqaPrefillAttentionWithSinks(actual, qTensor, kCache, vCache, sinks: null,
            numQHeads, numKVHeads, headDim, seqLen, kvLen, cacheSize, maskStart, windowSize, scale);
    allocator.Synchronize();

    var times = new List<double>();
    for (int it = 0; it < iters; it++)
    {
        var sw = Stopwatch.StartNew();
        CudaFusedOps.TryGqaPrefillAttentionWithSinks(actual, qTensor, kCache, vCache, sinks: null,
            numQHeads, numKVHeads, headDim, seqLen, kvLen, cacheSize, maskStart, windowSize, scale);
        allocator.Synchronize();
        sw.Stop();
        times.Add(sw.Elapsed.TotalMilliseconds);
    }
    times.Sort();
    double med = times[times.Count / 2];
    Console.WriteLine($"{name,-24} {len,6} {med,10:F3} {maxErr,12:E2} {"flash",8}");
}

static float VerifySampled(
    Tensor actual, float[,,] q, float[,,] k, float[,,] v,
    int numQHeads, int numKVHeads, int seqLen, int kvLen, int headDim, int maskStart, int windowSize, float scale)
{
    float[] outv = actual.GetElementsAsFloat(seqLen * numQHeads * headDim);
    int group = numQHeads / numKVHeads;
    float maxErr = 0f;
    int[] sampleRows = { 0, seqLen / 3, seqLen / 2, seqLen - 1 };
    foreach (int t in sampleRows)
    {
        for (int h = 0; h < numQHeads; h++)
        {
            int kvh = h / group;
            int visible = maskStart + t;
            int lo = windowSize > 0 ? Math.Max(0, visible - windowSize + 1) : 0;
            int hi = Math.Min(visible, kvLen - 1);
            // online softmax reference
            float m = float.NegativeInfinity, l = 0f;
            var acc = new float[headDim];
            for (int kp = lo; kp <= hi; kp++)
            {
                float dot = 0f;
                for (int d = 0; d < headDim; d++) dot += q[h, t, d] * k[kvh, kp, d];
                dot *= scale;
                float mnew = MathF.Max(m, dot);
                float alpha = MathF.Exp(m - mnew);
                float p = MathF.Exp(dot - mnew);
                l = l * alpha + p;
                for (int d = 0; d < headDim; d++) acc[d] = acc[d] * alpha + p * v[kvh, kp, d];
                m = mnew;
            }
            float inv = l > 0 ? 1f / l : 0f;
            for (int d = 0; d < headDim; d++)
            {
                float refv = acc[d] * inv;
                float got = outv[(t * numQHeads + h) * headDim + d];
                maxErr = MathF.Max(maxErr, MathF.Abs(refv - got));
            }
        }
    }
    return maxErr;
}
