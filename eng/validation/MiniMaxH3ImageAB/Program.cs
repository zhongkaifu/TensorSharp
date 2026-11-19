// MiniMax-H3-Image end to end, several images in one process, through either engine: the A/B harness of stage S4
// (docs/design/diffusion-runtime.md section 7.2/7.3), the production path against the legacy oracle.
//
//   dotnet MiniMaxH3ImageAB.dll --engine runtime|legacy --backend ggml_cuda|ggml_metal|ggml_cpu \
//       --model <denoiser.gguf|.safetensors> --te <qwen3vl...gguf> --vae <minimax_h3_image_vae_fp16.safetensors> \
//       --prompts <prompts.json> --jobs id[,id...] [--size N] --out <dir> [--label name]
//       [--legacy-release-te] [--legacy-vae-batch N]
//
// runtime: the production load (ModelBase.Create, the registry's MiniMax-H3-Image family on the diffusion runtime).
// legacy:  the test seam MiniMaxH3ImageModel (one native graph a step, the native batched VAE decode); --legacy-release-te
//          disposes its text encoder after each encode (the bf16 verification's local patch, needed where the 40 GB bf16
//          DiT does not fit beside it), --legacy-vae-batch caps its tiles a decode call (TS_H3_IMAGE_VAE_BATCH; its graph
//          overflows past 4 on ggml_metal and ggml_cpu).
// The jobs run in order in this one process (the first is the cold first job, the rest warm); prompts.json is a list of
// {id, prompt, seed, width, height}; --size overrides the size. Each job writes <out>/<label>-<n>-<id>.png; <out>/<label>.json
// holds per job the phases' seconds (text, steps, denoise, decode, total), the decode's tiles, the PNG's SHA-256, the
// device memory peak over the job and over each phase (sampled every 50 ms: CUDA's used bytes, Metal's allocated size),
// the residency plan (runtime), and the load and process figures. Device memory is the device's whole use, so run alone.
using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.MiniMaxH3Image;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling.Diffusion;

string? Value(string name) => args.SkipWhile(a => a != name).Skip(1).FirstOrDefault();
bool Flag(string name) => args.Contains(name);
string Required(string name) => Value(name) ?? throw new ArgumentException($"{name} is required");

string engine = Required("--engine");
if (engine is not ("runtime" or "legacy")) throw new ArgumentException("--engine is runtime or legacy");
BackendType backend = Required("--backend") switch
{
    "ggml_metal" => BackendType.GgmlMetal,
    "ggml_cuda" => BackendType.GgmlCuda,
    "ggml_cpu" => BackendType.GgmlCpu,
    _ => throw new ArgumentException("--backend is ggml_metal, ggml_cuda or ggml_cpu"),
};
string model = Path.GetFullPath(Required("--model"));
string te = Path.GetFullPath(Required("--te"));
string vae = Path.GetFullPath(Required("--vae"));
JsonArray prompts = JsonNode.Parse(File.ReadAllText(Required("--prompts")))!.AsArray();
string[] jobs = Required("--jobs").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
int size = int.Parse(Value("--size") ?? "0", CultureInfo.InvariantCulture);
string outDir = Path.GetFullPath(Required("--out"));
string label = Value("--label") ?? engine;
bool releaseTe = Flag("--legacy-release-te");
int vaeBatch = int.Parse(Value("--legacy-vae-batch") ?? "0", CultureInfo.InvariantCulture);
if (engine == "runtime" && (releaseTe || vaeBatch > 0)) throw new ArgumentException("--legacy-* options are the legacy engine's");
Directory.CreateDirectory(outDir);
string resultPath = Path.Combine(outDir, label + ".json");
if (File.Exists(resultPath)) throw new InvalidOperationException($"{resultPath} already exists");

// The companions (as --video-text-encoder / --video-vae set them), and the process's one GGML backend, pinned first.
Environment.SetEnvironmentVariable("TS_VIDEO_TEXT_ENCODER", te);
Environment.SetEnvironmentVariable("TS_VIDEO_VAE", vae);
GgmlBasicOps.EnsureBackendAvailable(backend switch
{
    BackendType.GgmlMetal => GgmlBackendType.Metal,
    BackendType.GgmlCuda => GgmlBackendType.Cuda,
    _ => GgmlBackendType.Cpu,
});

// ---- device memory sampler (the device's used bytes: CUDA total - free, Metal's currentAllocatedSize)
var samples = new List<(double T, long Used)>();
var clock = Stopwatch.StartNew();
using var stop = new CancellationTokenSource();
var sampler = new Thread(() =>
{
    while (!stop.IsCancellationRequested)
    {
        if (GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out long total))
            lock (samples) samples.Add((clock.Elapsed.TotalSeconds, total - free));
        Thread.Sleep(50);
    }
}) { IsBackground = true, Name = "h3img-ab-memory" };
sampler.Start();
long PeakBetween(double from, double to)
{
    long peak = 0;
    lock (samples)
        foreach (var (t, used) in samples)
            if (t >= from && t <= to && used > peak) peak = used;
    return peak;
}
string Load() => File.Exists("/proc/loadavg") ? File.ReadAllText("/proc/loadavg").Trim()
    : RunText("sysctl", "-n vm.loadavg");
static string RunText(string file, string arguments)
{
    try
    {
        using var p = Process.Start(new ProcessStartInfo(file, arguments) { RedirectStandardOutput = true, UseShellExecute = false })!;
        string text = p.StandardOutput.ReadToEnd().Trim();
        p.WaitForExit();
        return text;
    }
    catch (Exception e) { return "unavailable: " + e.Message; }
}

var result = new JsonObject
{
    ["engine"] = engine,
    ["label"] = label,
    ["backend"] = backend.ToString(),
    ["model"] = model,
    ["te"] = te,
    ["vae"] = vae,
    ["legacy_release_te"] = releaseTe,
    ["legacy_vae_batch"] = vaeBatch,
    ["env_int8"] = Environment.GetEnvironmentVariable("TS_H3_INT8_MATMUL") ?? "",
    ["load_start"] = Load(),
    ["started_utc"] = DateTime.UtcNow.ToString("O"),
};
var jobResults = new JsonArray();
result["jobs"] = jobResults;
void Save()
{
    result["peak_device_bytes"] = PeakBetween(0, double.MaxValue);
    result["process_peak_working_set"] = Process.GetCurrentProcess().PeakWorkingSet64;
    File.WriteAllText(resultPath, result.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
}

// ---- --vae-latent: the decoders alone, on one latent, both engines in this process (A B B A ... by round)
if (Value("--vae-latent") is { } latentPath)
{
    int lh = int.Parse(Required("--latent-h"), CultureInfo.InvariantCulture), lw = int.Parse(Required("--latent-w"), CultureInfo.InvariantCulture);
    int rounds = int.Parse(Value("--rounds") ?? "2", CultureInfo.InvariantCulture);
    byte[] latentBytes = File.ReadAllBytes(latentPath);
    var latent = new float[latentBytes.Length / sizeof(float)];
    Buffer.BlockCopy(latentBytes, 0, latent, 0, latentBytes.Length);
    if (latent.LongLength != (long)lh * lw * 24) throw new ArgumentException($"{latentPath} holds {latent.Length} floats, not [{lh}][{lw}][24]");
    var legacyVae = new MiniMaxH3ImageVae(vae) { BatchOverride = vaeBatch };
    var codec = new MiniMaxH3ImageVaeCodec(vae, backend);
    var decodes = new JsonArray();
    var shas = new Dictionary<string, string>();
    for (int r = 0; r < rounds; r++)
        foreach (string arm in r % 2 == 0 ? new[] { "legacy", "runtime" } : new[] { "runtime", "legacy" })
        {
            double t0 = clock.Elapsed.TotalSeconds;
            RgbImage decoded = arm == "legacy" ? legacyVae.Decode(latent, lh, lw) : codec.Decode(latent, lh, lw);
            double t1 = clock.Elapsed.TotalSeconds;
            byte[] png = ImageIO.EncodePng(decoded);
            string sha = Convert.ToHexStringLower(SHA256.HashData(png));
            if (!shas.ContainsKey(arm))
            {
                shas[arm] = sha;
                File.WriteAllBytes(Path.Combine(outDir, $"{label}-vae-{arm}.png"), png);
            }
            decodes.Add(new JsonObject
            {
                ["round"] = r + 1, ["arm"] = arm, ["seconds"] = t1 - t0,
                ["native_seconds"] = arm == "legacy" ? legacyVae.LastNativeSeconds : codec.LastNativeSeconds,
                ["tiles"] = arm == "legacy" ? legacyVae.LastTileCount : codec.LastTileCount,
                ["png_sha256"] = sha, ["peak_device_bytes"] = PeakBetween(t0, t1),
            });
            Console.WriteLine($"[h3img-ab] vae round {r + 1} {arm}: {t1 - t0:F3}s ({(arm == "legacy" ? legacyVae.LastTileCount : codec.LastTileCount)} tiles) sha {sha}");
        }
    result["vae_decodes"] = decodes;
    result["vae_bit_exact"] = shas.Count == 2 && shas["legacy"] == shas["runtime"];
    result["vae_compiles"] = codec.Compiles;
    stop.Cancel();
    sampler.Join();
    Save();
    codec.Dispose();
    legacyVae.Dispose();
    // The backends go before the process's static destructors (as the CLI's exit does), or ggml-metal aborts in them.
    GgmlBasicOps.Shutdown();
    return 0;
}

// ---- load
double loadStart = clock.Elapsed.TotalSeconds;
IImageGenerationModel image;
ModelBase loaded;
MiniMaxH3ImageModel? legacy = null;
MiniMaxH3ImageDiffusionModel? runtime = null;
var phases = new List<(DiffusionPhase Phase, double T)>();
if (engine == "runtime")
{
    loaded = ModelBase.Create(model, backend);
    runtime = loaded as MiniMaxH3ImageDiffusionModel
        ?? throw new InvalidOperationException($"the registry built {loaded.GetType().Name}, not the runtime's MiniMax-H3-Image model");
    runtime.ProgressObserver = p => { lock (phases) phases.Add((p.Phase, clock.Elapsed.TotalSeconds)); };
    image = runtime;
}
else
{
    legacy = new MiniMaxH3ImageModel(model, backend) { ReleaseTextEncoderAfterEncode = releaseTe, VaeBatchOverride = vaeBatch };
    loaded = legacy;
    image = legacy;
}
result["load_seconds"] = clock.Elapsed.TotalSeconds - loadStart;
result["model_type"] = loaded.GetType().FullName;
Save();

// ---- jobs
for (int n = 0; n < jobs.Length; n++)
{
    string id = jobs[n];
    JsonNode p = prompts.FirstOrDefault(x => (string?)x!["id"] == id) ?? throw new ArgumentException($"no prompt '{id}'");
    int w = size > 0 ? size : (int)p["width"]!, h = size > 0 ? size : (int)p["height"]!;
    long seed = (long)p["seed"]!;
    var stepTimes = new List<double>();
    double last = 0;
    var parameters = new QwenImageParams
    {
        Width = w,
        Height = h,
        Seed = seed,
        OnStep = (step, total, _) =>
        {
            double now = clock.Elapsed.TotalSeconds;
            stepTimes.Add(now - last);
            last = now;
        },
    };
    lock (phases) phases.Clear();
    string load0 = Load();
    double start = clock.Elapsed.TotalSeconds;
    last = start;
    var job = new JsonObject { ["n"] = n + 1, ["id"] = id, ["seed"] = seed, ["width"] = w, ["height"] = h, ["load_before"] = load0 };
    try
    {
        RgbImage picture = image.GenerateImage((string)p["prompt"]!, parameters);
        double end = clock.Elapsed.TotalSeconds;
        string png = Path.Combine(outDir, $"{label}-{n + 1}-{id}.png");
        byte[] bytes = ImageIO.EncodePng(picture);
        File.WriteAllBytes(png, bytes);
        job["png"] = png;
        job["png_sha256"] = Convert.ToHexStringLower(SHA256.HashData(bytes));
        job["wall_seconds"] = end - start;
        // The step callbacks time from the job's start, so the first one carries the text phase: the model's own steps.
        double text, denoise, decode, total;
        IReadOnlyList<double> steps;
        int tiles;
        double native;
        if (runtime != null)
        {
            var t = runtime.LastResult!.Timings;
            (text, denoise, decode, total, steps) = (t.Text, t.Denoise, t.Decode, t.Total, t.Steps);
            tiles = runtime.LastTileCount;
            native = runtime.LastDecodeNativeSeconds;
            job["residency"] = runtime.LastResult.Residency?.Describe();
            job["text_tokens"] = runtime.LastTextTokens;
            // Per phase: from the phase's entry to the next one's (the run's end for the last).
            var marks = new List<(DiffusionPhase, double)>();
            lock (phases) marks.AddRange(phases.Where(x => x.Phase is DiffusionPhase.Condition or DiffusionPhase.Denoise or DiffusionPhase.Decode or DiffusionPhase.Composite)
                .GroupBy(x => x.Phase).Select(g => (g.Key, g.First().T)));
            var peaks = new JsonObject();
            for (int i = 0; i < marks.Count; i++)
                peaks[marks[i].Item1.ToString()] = PeakBetween(marks[i].Item2, i + 1 < marks.Count ? marks[i + 1].Item2 : end);
            job["phase_peak_device_bytes"] = peaks;
        }
        else
        {
            var lt = legacy!.LastTimings;
            (text, denoise, decode) = (lt.Text, lt.Denoise, lt.Decode);
            total = end - start;
            steps = stepTimes.ToArray();
            // The legacy loop times its steps inside the denoise; the callbacks' first interval holds the text phase.
            var adjusted = stepTimes.ToArray();
            if (adjusted.Length > 0) adjusted[0] = Math.Max(0, adjusted[0] - text - legacy.LastTextEncoderReleaseSeconds);
            steps = adjusted;
            tiles = legacy.LastVaeTileCount;
            native = legacy.LastVaeNativeSeconds;
            job["text_encoder_release_seconds"] = legacy.LastTextEncoderReleaseSeconds;
            double denoiseStart = start + text + legacy.LastTextEncoderReleaseSeconds, decodeStart = denoiseStart + denoise;
            job["phase_peak_device_bytes"] = new JsonObject
            {
                ["Condition"] = PeakBetween(start, denoiseStart),
                ["Denoise"] = PeakBetween(denoiseStart, decodeStart),
                ["Decode"] = PeakBetween(decodeStart, end),
            };
        }
        job["text_seconds"] = text;
        job["denoise_seconds"] = denoise;
        job["decode_seconds"] = decode;
        job["total_seconds"] = total;
        job["step_seconds"] = new JsonArray(steps.Select(s => (JsonNode)s).ToArray());
        job["tiles"] = tiles;
        job["decode_native_seconds"] = native;
        job["peak_device_bytes"] = PeakBetween(start, end);
        job["rc"] = 0;
        Console.WriteLine($"[h3img-ab] {label} job {n + 1} {id} {w}x{h}: text {text:F2}s denoise {denoise:F2}s decode {decode:F2}s " +
                          $"total {total:F2}s peak {PeakBetween(start, end) / 1e9:F2} GB sha {job["png_sha256"]}");
    }
    catch (Exception e)
    {
        job["rc"] = 1;
        job["error"] = e.GetType().Name + ": " + e.Message;
        job["wall_seconds"] = clock.Elapsed.TotalSeconds - start;
        Console.Error.WriteLine($"[h3img-ab] {label} job {n + 1} {id} failed: {e}");
    }
    job["load_after"] = Load();
    jobResults.Add(job);
    Save();
    if ((int)job["rc"]! != 0) break;
}

result["load_end"] = Load();
stop.Cancel();
sampler.Join();
Save();
loaded.Dispose();
GgmlBasicOps.Shutdown();
return jobResults.All(j => (int)j!["rc"]! == 0) ? 0 : 1;
