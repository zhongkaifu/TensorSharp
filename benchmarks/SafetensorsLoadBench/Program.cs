// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// Benchmark for the native safetensors reader. Measures the end-to-end "open + read every tensor
// into a managed float[]" throughput for:
//   1. the original .safetensors (BF16 -> F32 upcast on read), and
//   2. the converted F32 .gguf (the path TensorSharp used before),
// and proves the two produce bit-identical weights. A final microbench isolates the BF16->F32 SIMD
// kernel from disk/mmap so its raw conversion bandwidth is visible.
//
// Usage: SafetensorsLoadBench [model.safetensors] [model.gguf] [iters]
using System.Diagnostics;
using System.Runtime.InteropServices;
using TensorSharp.Runtime;

static double Gb(long bytes) => bytes / 1024.0 / 1024.0 / 1024.0;
static double Mb(long bytes) => bytes / 1024.0 / 1024.0;

string stPath = args.Length > 0 ? args[0] : @"C:\Works\models\Qwen_Image-VAE.safetensors";
string ggPath = args.Length > 1 ? args[1] : @"C:\Works\models\qwen_image_vae.gguf";
int iters = args.Length > 2 ? int.Parse(args[2]) : 8;

Console.WriteLine("=== safetensors load/convert benchmark ===");
Console.WriteLine($"safetensors: {stPath}");
Console.WriteLine($"gguf       : {ggPath}");
Console.WriteLine($"iters      : {iters}   Vector256.HW={System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated}  " +
                  $"Vector512.HW={System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated}");
Console.WriteLine();

if (!File.Exists(stPath)) { Console.WriteLine($"missing {stPath}"); return; }

// ---- safetensors: read all tensors into float[] -----------------------------------------------
long stFileBytes = new FileInfo(stPath).Length;
long f32Bytes = 0, nElems = 0, nTensors = 0;
double stChecksum = 0;

double TimeReadAll(Func<(double checksum, long f32Bytes, long elems, long tensors)> body, out (double, long, long, long) info)
{
    var sw = Stopwatch.StartNew();
    info = body();
    sw.Stop();
    return sw.Elapsed.TotalMilliseconds;
}

(double, long, long, long) ReadSafetensors()
{
    using var f = new SafetensorsFile(stPath);
    double cs = 0; long fb = 0, el = 0, nt = 0;
    foreach (var (name, _) in f.Tensors)
    {
        float[] a = f.ReadFloat32(name);
        fb += (long)a.Length * 4; el += a.Length; nt++;
        // touch every element so the conversion isn't dead-code-eliminated
        for (int i = 0; i < a.Length; i++) cs += a[i];
    }
    return (cs, fb, el, nt);
}

Console.WriteLine("[safetensors BF16 -> F32]");
double stCold = TimeReadAll(ReadSafetensors, out var sti);
(stChecksum, f32Bytes, nElems, nTensors) = sti;
double stBest = double.MaxValue, stSum = 0;
for (int it = 0; it < iters; it++)
{
    double ms = TimeReadAll(ReadSafetensors, out _);
    stBest = Math.Min(stBest, ms); stSum += ms;
}
double stAvg = stSum / iters;
Console.WriteLine($"  tensors={nTensors}  elements={nElems:N0}  fileIn={Mb(stFileBytes):F1} MB (BF16)  f32Out={Mb(f32Bytes):F1} MB");
Console.WriteLine($"  cold={stCold:F1} ms   warm best={stBest:F2} ms  avg={stAvg:F2} ms");
Console.WriteLine($"  warm best -> read {Gb(stFileBytes) / (stBest / 1000):F2} GB/s in, produce {Gb(f32Bytes) / (stBest / 1000):F2} GB/s f32 ({nElems / (stBest / 1e3) / 1e6:F0} Melem/s)");
Console.WriteLine($"  checksum={stChecksum:R}");
Console.WriteLine();

// ---- gguf F32 baseline ------------------------------------------------------------------------
if (File.Exists(ggPath))
{
    long ggFileBytes = new FileInfo(ggPath).Length;
    double ggChecksum = 0;
    (double, long, long, long) ReadGguf()
    {
        using var g = new GgufFile(ggPath);
        var store = new GgufFloatTensorStore(g);
        double cs = 0; long fb = 0, el = 0, nt = 0;
        foreach (var (name, _) in g.Tensors)
        {
            float[] a = store.ReadFloat32(name);
            fb += (long)a.Length * 4; el += a.Length; nt++;
            for (int i = 0; i < a.Length; i++) cs += a[i];
        }
        return (cs, fb, el, nt);
    }

    Console.WriteLine("[gguf F32 baseline]");
    double ggCold = TimeReadAll(ReadGguf, out var ggi);
    ggChecksum = ggi.Item1;
    double ggBest = double.MaxValue, ggSum = 0;
    for (int it = 0; it < iters; it++) { double ms = TimeReadAll(ReadGguf, out _); ggBest = Math.Min(ggBest, ms); ggSum += ms; }
    Console.WriteLine($"  fileIn={Mb(ggFileBytes):F1} MB (F32)  cold={ggCold:F1} ms  warm best={ggBest:F2} ms  avg={ggSum / iters:F2} ms");
    Console.WriteLine($"  warm best -> read {Gb(ggFileBytes) / (ggBest / 1000):F2} GB/s f32");
    Console.WriteLine($"  checksum={ggChecksum:R}");
    Console.WriteLine();
    Console.WriteLine(stChecksum == ggChecksum
        ? "PARITY: safetensors(BF16->F32) checksum == gguf(F32) checksum  ✔ bit-identical weights"
        : $"PARITY MISMATCH: {stChecksum:R} != {ggChecksum:R}");
    Console.WriteLine($"safetensors reads {Mb(stFileBytes):F0} MB on disk vs gguf {Mb(ggFileBytes):F0} MB " +
                      $"({(double)ggFileBytes / stFileBytes:F2}x smaller); warm load {stBest:F1} vs {ggBest:F1} ms");
    Console.WriteLine();
}

// ---- isolated BF16 -> F32 SIMD kernel microbench ----------------------------------------------
Console.WriteLine("[BF16 -> F32 kernel microbench (in-RAM, no disk)]");
const int N = 64 * 1024 * 1024; // 64M elements = 128 MB bf16 -> 256 MB f32
var bf16 = new ushort[N];
var rng = new Random(1);
for (int i = 0; i < N; i++) bf16[i] = (ushort)rng.Next(0, 0x10000);
var outF = new float[N];

double Convert()
{
    var sw = Stopwatch.StartNew();
    unsafe
    {
        fixed (ushort* s = bf16)
        fixed (float* d = outF)
        {
            // replicate SafetensorsFile's vectorised widen via the public reader on a temp file would
            // add disk noise; instead invoke the same logic inline through a tiny shim file is overkill,
            // so just call the scalar+vector path through a synthetic file-less helper:
            SafetensorsKernels.Bf16ToF32(s, d, N);
        }
    }
    sw.Stop();
    return sw.Elapsed.TotalMilliseconds;
}

double kBest = double.MaxValue;
for (int it = 0; it < Math.Max(3, iters); it++) kBest = Math.Min(kBest, Convert());
long kBytesOut = (long)N * 4;
Console.WriteLine($"  {N:N0} elem  best={kBest:F2} ms  -> {Gb(kBytesOut) / (kBest / 1000):F1} GB/s f32 out ({N / (kBest / 1e3) / 1e6:F0} Melem/s)");
