// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The CUDA executor's prompt attention, measured: the 16-bit prompt kernel (ts_flash_prompt_f16_d256) against the 8-bit pair
// (ts_flash_prompt_band_f16_d256 + ts_flash_prompt_i8e4_d256, with their combine) on synthetic operands, one layer of one
// sequence's prompt chunk, each kernel's own split rule. See README.md. Prints and writes timings; asserts nothing.
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i++)
{
    if (!args[i].StartsWith("--", StringComparison.Ordinal)) { options["mode"] = args[i]; continue; }
    if (i + 1 >= args.Length) throw new ArgumentException($"{args[i]} needs a value.");
    options[args[i][2..]] = args[++i];
}
string mode = options.GetValueOrDefault("mode", "kernel");
if (mode != "kernel")
    throw new ArgumentException($"Unknown mode '{mode}': this probe has the 'kernel' mode (the quality modes come with the default change).");
string output = options.GetValueOrDefault("output");
int repeats = int.Parse(options.GetValueOrDefault("repeats", "5"), CultureInfo.InvariantCulture);
bool sweep = options.GetValueOrDefault("sweep", "1") != "0";

using var allocator = new CudaAllocator();
allocator.Context.MakeCurrent();
CudaFlashPrefillKernels flash = allocator.FlashPrefillKernels
    ?? throw new InvalidOperationException("tensorsharp_flash_prefill did not load (stale or missing module).");
if (!flash.Runs8BitPrompt(256))
    throw new InvalidOperationException($"This image has no 8-bit prompt attention (image arch {flash.ImageArch}, attn8 probe {(flash.Attn8Code ? 1 : 0)}): " +
        "it runs on an sm_120a / sm_121a build only.");
var name = new byte[256];
CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError();
CudaDriverApi.cuDeviceGetName(name, name.Length, device).ThrowOnError();
string deviceName = Encoding.UTF8.GetString(name).TrimEnd('\0');
Console.WriteLine($"device {deviceName}, {flash.MultiprocessorCount} SMs, flash prefill image arch {flash.ImageArch}, repeats {repeats}");

const int D = 256, KvHeads = 4;
int maxKeys = 67_584, maxRows = 2048, maxHeads = 24;
int cap = maxKeys + 32;
var rng = new Random(376);
float Next() => (float)(rng.NextDouble() * 2 - 1);
var qHost = new float[maxHeads * maxRows * D];
for (int i = 0; i < qHost.Length; i++) qHost[i] = Next();
using Tensor q = Tensor.FromArray(allocator, qHost);
using Tensor k = HalfCache(KvHeads * cap * D), v = HalfCache(KvHeads * cap * D);
((CudaStorage)q.Storage).EnsureDeviceCurrent();

var results = new List<Dictionary<string, object>>();
// The three shapes of the issue's measurements (Qwen3.8-27B: 24 query heads over 4 KV heads).
var shapes = new List<(int Heads, int Nq, int Len, string Label)>
{
    (24, 968, 47_981, "B turn"), (24, 2048, 67_584, "A chunk"), (24, 968, 25_104, "24K turn"),
};
if (sweep)
    foreach (int heads in new[] { 24, 16 })
        foreach (int keys in new[] { 8192, 32_768, 65_536 })
            foreach (int nq in new[] { 17, 32, 64, 128, 192, 256, 384, 512, 1024 })
                shapes.Add((heads, nq, keys, "sweep"));

Console.WriteLine("| shape | heads/kv | rows | keys | 16-bit ms (splits) | band ms | 8-bit attn ms (splits) | 8-bit total ms | speedup |");
Console.WriteLine("|---|---|---:|---:|---:|---:|---:|---:|---:|");
foreach (var (heads, nq, len, label) in shapes)
{
    int rows = nq * (heads / KvHeads);
    int splits = flash.Splits(rows, KvHeads, len);
    int bandKeys = CudaFlashPrefillKernels.Attn8BandKeys(len);
    using var band = new Tensor(allocator, DType.Float32, CudaFlashPrefillKernels.Attn8BandBytes(bandKeys, KvHeads) / 4);
    using var out16 = new Tensor(allocator, DType.Float32, (long)nq * heads * D);
    using var out8 = new Tensor(allocator, DType.Float32, (long)nq * heads * D);
    using Tensor partial = splits > 1 ? new Tensor(allocator, DType.Float32, CudaFlashPrefillKernels.PartialFloats(nq, heads, D, splits)) : null;
    foreach (Tensor t in new[] { band, out16, out8, partial }) if (t != null) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
    IntPtr stream = allocator.Stream.Handle;
    float scale = 1f / MathF.Sqrt(D);
    IntPtr partialPtr = partial == null ? IntPtr.Zero : P(partial);
    double f16 = Time(() => flash.LaunchAttention(P(q), P(k), P(v), cap, nq, len, 0, heads, KvHeads, D, scale, splits, partialPtr, P(out16), stream,
        prompt: true));
    double bandMs = Time(() => flash.LaunchPromptBand(P(k), P(v), cap, len, 0, KvHeads, P(band), bandKeys, stream));
    double total = Time(() => flash.LaunchPromptAttention8(P(q), P(k), P(v), cap, nq, len, 0, heads, KvHeads, scale, splits, P(band), bandKeys,
        partialPtr, P(out8), stream));
    Console.WriteLine($"| {label} | {heads}/{KvHeads} | {nq} | {len} | {f16:F3} ({splits}) | {bandMs:F3} | {total - bandMs:F3} ({splits}) | {total:F3} | {f16 / total:F2}x |");
    results.Add(new Dictionary<string, object>
    {
        ["label"] = label, ["heads"] = heads, ["kv_heads"] = KvHeads, ["rows"] = nq, ["keys"] = len, ["splits"] = splits,
        ["f16_ms"] = f16, ["band_ms"] = bandMs, ["attn8_total_ms"] = total, ["speedup"] = f16 / total,
    });
}

// Attn8MinRows by the design's rule: the smallest swept row count from which the 8-bit pair beats the 16-bit kernel at every
// swept key count and both GQA groups, rounded up to a power of two (reported; the constant is set by hand per architecture).
var swept = results.Where(r => (string)r["label"] == "sweep").ToList();
int? minRows = swept.Select(r => (int)r["rows"]).Distinct().OrderBy(n => n)
    .Where(n => swept.Where(r => (int)r["rows"] >= n).All(r => (double)r["speedup"] > 1.0))
    .Select(n => (int?)n).FirstOrDefault();
string rule = minRows is int m ? $"{m} rows (power of two: {1 << (int)Math.Ceiling(Math.Log2(m))})" : "none of the swept row counts";
if (sweep) Console.WriteLine($"8-bit + band beats 16-bit at every swept key count and group from {rule}");
if (output != null)
{
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
    File.WriteAllText(output, JsonSerializer.Serialize(new Dictionary<string, object>
    {
        ["device"] = deviceName, ["sms"] = flash.MultiprocessorCount, ["image_arch"] = flash.ImageArch, ["repeats"] = repeats,
        ["min_rows_rule"] = sweep ? rule : null, ["results"] = results,
    }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
}
return 0;

static IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

Tensor HalfCache(long count)
{
    var host = new float[count];
    for (long i = 0; i < count; i++) host[i] = Next();
    using Tensor f32 = Tensor.FromArray(allocator, host);
    var f16 = new Tensor(allocator, DType.Float16, count);
    ((CudaStorage)f32.Storage).EnsureDeviceCurrent();
    ((CudaStorage)f16.Storage).EnsureDeviceCurrent();
    allocator.Kernels.LaunchConvertF32F16(P(f32), P(f16), count, allocator.Stream.Handle);
    allocator.Stream.Synchronize();
    ((CudaStorage)f16.Storage).MarkDeviceModified();
    return f16;
}

// The median of `repeats` timed launches (CUDA events on the stream) after one warm-up launch.
double Time(Action launch)
{
    allocator.Context.MakeCurrent();
    launch();
    allocator.Stream.Synchronize();
    CudaDriverApi.cuEventCreate(out IntPtr start, 0).ThrowOnError();
    CudaDriverApi.cuEventCreate(out IntPtr end, 0).ThrowOnError();
    try
    {
        var samples = new List<double>();
        for (int r = 0; r < repeats; r++)
        {
            CudaDriverApi.cuEventRecord(start, allocator.Stream.Handle).ThrowOnError();
            launch();
            CudaDriverApi.cuEventRecord(end, allocator.Stream.Handle).ThrowOnError();
            CudaDriverApi.cuEventSynchronize(end).ThrowOnError();
            CudaDriverApi.cuEventElapsedTime(out float ms, start, end).ThrowOnError();
            samples.Add(ms);
        }
        samples.Sort();
        return samples[samples.Count / 2];
    }
    finally
    {
        CudaDriverApi.cuEventDestroy(start);
        CudaDriverApi.cuEventDestroy(end);
    }
}
