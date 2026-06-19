// Standalone vision-encoder benchmark + correctness checksum harness.
//
// Loads a single model's vision encoder (no language model), preprocesses a test
// image, runs the encoder Encode() N times and reports per-encode latency plus a
// reproducible output checksum so an optimization can be proven equivalent.
//
// The encoder is built on a GGML CPU allocator, which is what TensorSharp uses
// for the ggml_cpu / ggml_cuda backends (allocator is GgmlAllocator =>
// _useNativeAttention = true). The scalar CPU loops we optimize (patch-embed
// conv, RoPE, pooling) run on the CPU regardless of backend, so this isolates
// exactly the work being optimized.
//
// Env knobs:
//   TS_VBENCH_TYPE    gemma4 | mistral3 | qwen35   (default gemma4)
//   TS_VBENCH_MMPROJ  path to the mmproj .gguf
//   TS_VBENCH_IMAGE   path to a test image (default imgs/banner_1.png)
//   TS_VBENCH_ITERS   timed iterations (default 5)
//   TS_VBENCH_WARMUP  warmup iterations (default 1)
using System.Diagnostics;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;

static string Env(string k, string fallback)
{
    string s = Environment.GetEnvironmentVariable(k);
    return string.IsNullOrEmpty(s) ? fallback : s;
}
static int EnvInt(string k, int fallback)
{
    string s = Environment.GetEnvironmentVariable(k);
    return !string.IsNullOrEmpty(s) && int.TryParse(s, out int v) ? v : fallback;
}

string type = Env("TS_VBENCH_TYPE", "gemma4").ToLowerInvariant();
string image = Env("TS_VBENCH_IMAGE", @"C:\Works\TensorSharp\imgs\banner_1.png");
int iters = EnvInt("TS_VBENCH_ITERS", 5);
int warmup = EnvInt("TS_VBENCH_WARMUP", 1);

string mmproj = Env("TS_VBENCH_MMPROJ", type switch
{
    "mistral3" => @"C:\Works\models\Ministral-3-8B-Instruct-2512-BF16-mmproj.gguf",
    "qwen35" => @"C:\Works\models\Qwen3.6-35B-A3B-mmproj-F16.gguf",
    _ => @"C:\Works\models\mmproj-Gemma4-F16.gguf",
});

Console.WriteLine($"[vbench] type={type} mmproj={Path.GetFileName(mmproj)} image={Path.GetFileName(image)} iters={iters} warmup={warmup}");
if (!File.Exists(mmproj)) { Console.WriteLine($"[vbench] missing mmproj: {mmproj}"); return; }
if (!File.Exists(image)) { Console.WriteLine($"[vbench] missing image: {image}"); return; }

// TS_VBENCH_BACKEND=cpu (default) | cuda — exercise the encoder on the GGML CPU
// or CUDA backend. On CUDA the heavy ops run on the GPU, but GGML tensors are
// host-resident so each op uploads inputs / downloads outputs (no cross-op
// device residency); this lets us measure that round-trip cost directly.
var ggmlBackend = Env("TS_VBENCH_BACKEND", "cpu").ToLowerInvariant() == "cuda"
    ? GgmlBackendType.Cuda : GgmlBackendType.Cpu;
Console.WriteLine($"[vbench] ggml backend = {ggmlBackend}");
var context = new GgmlContext(new[] { 0 }, ggmlBackend);
var allocator = new GgmlAllocator(context, 0);

// Each model's encoder exposes a uniform Encode(pixels, dimA, dimB) -> Tensor.
// We wrap construction + preprocessing per type and return a closure that runs
// one encode, plus a label describing the geometry.
Func<Tensor> encodeOnce;
string geom;

switch (type)
{
    case "mistral3":
    {
        var enc = new Mistral3VisionEncoder(mmproj, allocator);
        var proc = new Mistral3ImageProcessor(enc.ImageSize, enc.PatchSize);
        var (pixels, w, h) = proc.ProcessImage(image);
        geom = $"{w}x{h}, patch={enc.PatchSize}, merge={enc.SpatialMergeSize}";
        encodeOnce = () => enc.Encode(pixels, w, h);
        break;
    }
    case "qwen35":
    {
        var enc = new Qwen35VisionEncoder(mmproj, allocator);
        var proc = new Qwen35ImageProcessor(enc.PatchSize, enc.SpatialMergeSize);
        var (pixels, resH, resW) = proc.ProcessImage(image);
        geom = $"{resW}x{resH}, patch={enc.PatchSize}, merge={enc.SpatialMergeSize}";
        encodeOnce = () => enc.Encode(pixels, resH, resW);
        break;
    }
    default: // gemma4
    {
        var enc = new Gemma4VisionEncoder(mmproj, allocator);
        var proc = enc.IsUnified
            ? new Gemma4ImageProcessor(imageMean: enc.ImageMean, imageStd: enc.ImageStd)
            : new Gemma4ImageProcessor();
        var (pixels, w, h) = proc.ProcessImage(image);
        geom = $"{w}x{h}, unified={enc.IsUnified}";
        encodeOnce = () => enc.Encode(pixels, w, h);
        break;
    }
}

Console.WriteLine($"[vbench] geometry: {geom}");

// Warmup (also builds caches: rope tables, transposed weights, position embd).
Tensor last = null;
for (int i = 0; i < warmup; i++)
{
    last?.Dispose();
    last = encodeOnce();
}

// Checksum the last warmup output for correctness comparison across versions.
using (var contig = Ops.NewContiguous(last))
{
    int n = (int)contig.ElementCount();
    float[] data = contig.GetElementsAsFloat(n);
    double sum = 0, sumsq = 0; float min = float.MaxValue, max = float.MinValue;
    for (int i = 0; i < n; i++)
    {
        float v = data[i];
        sum += v; sumsq += (double)v * v;
        if (v < min) min = v;
        if (v > max) max = v;
    }
    string shape = string.Join("x", contig.Sizes.ToArray());
    Console.WriteLine($"[vbench] output shape: {shape}  ({n} elems)");
    Console.WriteLine($"[vbench] CHECKSUM sum={sum:R} sumsq={sumsq:R} min={min:R} max={max:R}");
    Console.Write("[vbench] first8: ");
    for (int i = 0; i < Math.Min(8, n); i++) Console.Write($"{data[i]:R} ");
    Console.WriteLine();
}
last?.Dispose();

// Timed iterations.
var times = new List<double>();
for (int i = 0; i < iters; i++)
{
    var sw = Stopwatch.StartNew();
    var outp = encodeOnce();
    sw.Stop();
    outp.Dispose();
    times.Add(sw.Elapsed.TotalMilliseconds);
    Console.WriteLine($"[vbench] iter {i + 1}: {sw.Elapsed.TotalMilliseconds:F1} ms");
}
times.Sort();
double median = times[times.Count / 2];
double mean = times.Sum() / times.Count;
Console.WriteLine($"[vbench] RESULT median={median:F1} ms  mean={mean:F1} ms  min={times[0]:F1} ms  max={times[^1]:F1} ms");
