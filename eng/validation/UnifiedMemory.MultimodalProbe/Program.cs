// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.Video;

// Explicit validation harness: covered native allocations share one quota.
// This quota is NOT a hard process RSS/VRAM cap. See the report's Scope field.
var options = new Dictionary<string, string>();
for (int i = 0; i < args.Length; i += 2)
{
    if (i + 1 == args.Length || !args[i].StartsWith("--")) throw new ArgumentException("Expected --name value pairs.");
    options.Add(args[i][2..], args[i + 1]);
}
string Get(string name, string fallback) => options.GetValueOrDefault(name, fallback);
int Number(string name, int fallback) => int.Parse(Get(name, fallback.ToString(CultureInfo.InvariantCulture)), CultureInfo.InvariantCulture);
string kind = Get("kind", "image"), modelPath = Path.GetFullPath(options["model"]);
string output = Path.GetFullPath(options["output"]);
Directory.CreateDirectory(output);
double quotaGiB = double.Parse(Get("budget-gib", "32"), CultureInfo.InvariantCulture);
if (!double.IsFinite(quotaGiB) || quotaGiB < -1 || quotaGiB > 1024) throw new ArgumentOutOfRangeException("budget-gib");
int width = Number("width", 512), height = Number("height", 512), steps = Number("steps", 20);
int frames = Number("frames", 22), repeats = Number("repeat", 1);
long seed = long.Parse(Get("seed", "42"), CultureInfo.InvariantCulture);
float cfg = float.Parse(Get("cfg", "1"), CultureInfo.InvariantCulture);
if (kind is not ("image" or "video") || repeats < 1) throw new ArgumentException("Invalid kind/repeat.");
string prompt = Get("prompt", "A red ceramic teapot on a wooden table, soft daylight, product photograph.");
GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
GgmlBasicOps.ClearHostBufferCache();
GgmlBasicOps.ReleaseReuseComputeBuffers();
var budget = new MemoryBudget([new("covered-native", (long)(Math.Max(quotaGiB, 0) * (1L << 30)))]);
GgmlCacheBudgetScope? scope = quotaGiB < 0 ? null : new(budget, [["covered-native"]], true, ["covered-native"]);
long sampledPeakCredit = 0, sampledPeakRss = 0;
var rows = new List<object>();
using var stop = new CancellationTokenSource();
var sampler = Task.Run(() => {
    using var process = Process.GetCurrentProcess();
    while (!stop.IsCancellationRequested) {
        var usage = budget.Snapshot().Single();
        sampledPeakCredit = Math.Max(sampledPeakCredit, usage.Committed + usage.Reserved);
        process.Refresh(); sampledPeakRss = Math.Max(sampledPeakRss, process.WorkingSet64);
        Thread.Sleep(10);
    }
});
Exception? failure = null;
long afterModelDispose = -1;
var wall = Stopwatch.StartNew();
try
{
    if (kind == "image")
    {
        using var model = new QwenImageModel(modelPath, BackendType.GgmlCuda);
        var parameters = new QwenImageParams { Width = width, Height = height, Steps = steps, Seed = seed, CfgScale = cfg };
        var inputs = Get("images", "").Split('|', StringSplitOptions.RemoveEmptyEntries).Select(ImageIO.Load).ToArray();
        for (int run = 0; run < repeats; ++run) {
            var timer = Stopwatch.StartNew();
            var result = inputs.Length == 0 ? model.GenerateImage(prompt, parameters) : model.EditImage(prompt, inputs, parameters);
            timer.Stop();
            rows.Add(new { Run = run, Seconds = timer.Elapsed.TotalSeconds, Pixels = SaveImage(result, $"image-{run}"), Budget = budget.Snapshot() });
        }
    }
    else
    {
        using var model = new MiniMaxH3Model(modelPath, BackendType.GgmlCuda);
        for (int run = 0; run < repeats; ++run) {
            var parameters = new VideoGenerationParams { Width = width, Height = height, Frames = frames, Steps = steps, Seed = seed, CfgScale = cfg };
            if (options.TryGetValue("image", out var firstImage)) parameters.ImagePath = firstImage;
            if (options.TryGetValue("end-image", out var lastImage)) parameters.EndImagePath = lastImage;
            var timer = Stopwatch.StartNew();
            var result = model.GenerateVideo(prompt, parameters);
            timer.Stop();
            var images = result.Frames.Select((im, i) => SaveImage(im, $"video-{run}-{i:D3}")).ToArray();
            var audio = result.Audio?.Channels.Select((ch, i) => SaveFloats(ch, $"audio-{run}-{i}")).ToArray();
            rows.Add(new { Run = run, Seconds = timer.Elapsed.TotalSeconds, Frames = images, Audio = audio, result.Fps,
                SampleRate = result.Audio?.SampleRate, Budget = budget.Snapshot() });
        }
    }
    afterModelDispose = budget.Snapshot().Single().Committed;
}
catch (Exception error) { failure = error; Console.Error.WriteLine(error); }
finally
{
    Native.TSGgml_QwenImage21ResetForwardCache();
    Native.TSGgml_QwenImage21ReleasePrefixCaches();
    GgmlBasicOps.ReleaseReuseComputeBuffers();
    GgmlBasicOps.ClearHostBufferCache();
    stop.Cancel(); await sampler;
    int active = scope?.ActiveAllocations ?? 0;
    var callbackError = scope?.CallbackError;
    try { scope?.Dispose(); } catch (Exception error) { failure ??= error; }
    wall.Stop();
    File.WriteAllText(Path.Combine(output, "report.json"), JsonSerializer.Serialize(new {
        Status = failure == null && active == 0 && callbackError == null ? "passed" : "failed",
        Error = failure?.ToString(), CallbackError = callbackError?.ToString(), Options = options,
        ModelPath = modelPath, ModelBytes = new FileInfo(modelPath).Length, QuotaGiB = quotaGiB,
        NativeSha256 = HashFile(Path.Combine(AppContext.BaseDirectory, OperatingSystem.IsWindows() ? "GgmlOps.dll" : "libGgmlOps.so")),
        ProbeSha256 = HashFile(typeof(Native).Assembly.Location), ModelsSha256 = HashFile(typeof(QwenImageModel).Assembly.Location),
        SampledPeakCreditBytes = sampledPeakCredit, SampledPeakRssBytes = sampledPeakRss, SamplingIntervalMs = 10,
        AfterModelDisposeBytes = afterModelDispose, FinalBudget = budget.Snapshot(), ActiveAllocations = active,
        WallSecondsIncludingExports = wall.Elapsed.TotalSeconds, Rows = rows,
        Scope = "Explicit opt-in quota for routed native weights, graph/prefix storage and expert staging; excludes unmanaged model mappings, managed arrays, other native allocators, backend pools and driver overhead. Sampled peaks are lower bounds. Finite outputs and stable hashes are numerical checks, not semantic quality acceptance. Model file hashes belong in the companion download manifest."
    }, new JsonSerializerOptions { WriteIndented = true }));
    if (active != 0 || callbackError != null || failure != null) Environment.ExitCode = 1;
}

object SaveImage(RgbImage result, string name) {
    var pixels = SaveFloats(result.Pixels, name);
    ImageIO.SavePng(Path.Combine(output, name + ".png"), result);
    return new { result.Width, result.Height, Pixels = pixels };
}
static string HashFile(string path) {
    using var file = File.OpenRead(path);
    return Convert.ToHexStringLower(SHA256.HashData(file));
}
object SaveFloats(float[] values, string name) {
    if (values.Length == 0 || values.Any(x => !float.IsFinite(x))) throw new InvalidDataException("Empty/nonfinite output: " + name);
    var bytes = MemoryMarshal.AsBytes(values.AsSpan());
    using (var file = File.Create(Path.Combine(output, name + ".f32"))) file.Write(bytes);
    return new { Count = values.Length, Min = values.Min(), Max = values.Max(), Mean = values.Average(x => (double)x),
        Sha256 = Convert.ToHexStringLower(SHA256.HashData(bytes)) };
}
static class Native {
    [DllImport("GgmlOps", CallingConvention = CallingConvention.Cdecl)] public static extern void TSGgml_QwenImage21ResetForwardCache();
    [DllImport("GgmlOps", CallingConvention = CallingConvention.Cdecl)] public static extern void TSGgml_QwenImage21ReleasePrefixCaches();
}
