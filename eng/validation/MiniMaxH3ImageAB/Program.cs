// MiniMax-H3-Image end to end, several images in one process, on the diffusion runtime: the A/B harness of stage S4
// (docs/design/diffusion-runtime.md section 7.2/7.3). Its legacy engine (the MiniMaxH3ImageModel seam) was deleted with the
// legacy code in stage S10; an A/B is now two builds of this tool.
//
//   dotnet MiniMaxH3ImageAB.dll [--engine runtime] --backend ggml_cuda|ggml_metal|ggml_cpu \
//       --model <denoiser.gguf|.safetensors> --te <qwen3vl...gguf> --vae <minimax_h3_image_vae_fp16.safetensors> \
//       --prompts <prompts.json> --jobs id[,id...] [--size N] --out <dir> [--label name]
//       [--lora <plug-in .json | weights> [--lora-scale S] [--lora-config C] ...] [--dump-cond <dir>]
//       [--diffusion-products bf16|int8|int8-all]
//
// The production load (ModelBase.Create, the registry's MiniMax-H3-Image family on the diffusion runtime).
// The jobs run in order in this one process (the first is the cold first job, the rest warm); prompts.json is a list of
// {id, prompt, seed, width, height[, images: [paths], keepSourceSize, mask: path, maskMode: grayscale|alpha, maskInvert,
// maskFeather, maskCrop, maskCropPadding]}; a job with images is an edit (EditImage; the paths are relative to
// prompts.json, in <Picture n> order; width and height 0 are the automatic size; a mask belongs to <Picture 1>, as the
// image routes' fields); --size overrides the size. --parity-dump <dir> runs the first job only, through the production pipeline with its components recorded, up to
// its first step (ParityDump.cs: the copies, posteriors, conditioning, rows, layout and the step's velocity, for
// eng/validation/MiniMaxH3ImageEdit/diffusers_parity.py). Each job writes <out>/<label>-<n>-<id>.png; <out>/<label>.json
// holds per job the phases' seconds (text, steps, denoise, decode, total), the decode's tiles, the PNG's SHA-256, the
// device memory peak over the job and over each phase (sampled every 50 ms: CUDA's used bytes, Metal's allocated size),
// the residency plan (runtime), and the load and process figures. Device memory is the device's whole use, so run alone.
// --lora (runtime only, repeatable, the hosts' spelling): the plug-ins applied through ILoraPluginHost.SetLoras after the load
// (stage S7); each job then records its recipe (steps, sigmas, heads). --dump-cond <dir>: before the jobs, each job's text
// conditioning (a separate encoder, released after) as <dir>/<id>.hidden.f32 ([rows][width] float32) and <id>.json (rows,
// width, modality tags, seed, size), the inputs a diffusers reference denoises from (eng/validation/MiniMaxH3ImageLoraRows).
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

string engine = Value("--engine") ?? "runtime";
if (engine != "runtime")
    throw new ArgumentException("--engine is runtime: the legacy engine (MiniMaxH3ImageModel) was deleted in stage S10; compare two builds instead");
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
if (Flag("--legacy-release-te") || Value("--legacy-vae-batch") != null)
    throw new ArgumentException("--legacy-* options were the legacy engine's, deleted in stage S10");
List<LoraSpec> loraSpecs = LoraCliFlags.Parse(args);
string? dumpCond = Value("--dump-cond");
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
    ["env_int8"] = Environment.GetEnvironmentVariable("TS_H3_INT8_MATMUL") ?? "",
    ["loras"] = LoraCliFlags.Describe(loraSpecs),
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

// ---- --vae-latent: the decoder alone, on one latent, --rounds times in this process
if (Value("--vae-latent") is { } latentPath)
{
    int lh = int.Parse(Required("--latent-h"), CultureInfo.InvariantCulture), lw = int.Parse(Required("--latent-w"), CultureInfo.InvariantCulture);
    int rounds = int.Parse(Value("--rounds") ?? "2", CultureInfo.InvariantCulture);
    byte[] latentBytes = File.ReadAllBytes(latentPath);
    var latent = new float[latentBytes.Length / sizeof(float)];
    Buffer.BlockCopy(latentBytes, 0, latent, 0, latentBytes.Length);
    if (latent.LongLength != (long)lh * lw * 24) throw new ArgumentException($"{latentPath} holds {latent.Length} floats, not [{lh}][{lw}][24]");
    var codec = new MiniMaxH3ImageVaeCodec(vae, backend);
    var decodes = new JsonArray();
    string? first = null;
    for (int r = 0; r < rounds; r++)
    {
        double t0 = clock.Elapsed.TotalSeconds;
        RgbImage decoded = codec.Decode(latent, lh, lw);
        double t1 = clock.Elapsed.TotalSeconds;
        byte[] png = ImageIO.EncodePng(decoded);
        string sha = Convert.ToHexStringLower(SHA256.HashData(png));
        if (first == null)
        {
            first = sha;
            File.WriteAllBytes(Path.Combine(outDir, $"{label}-vae.png"), png);
        }
        decodes.Add(new JsonObject
        {
            ["round"] = r + 1, ["seconds"] = t1 - t0, ["native_seconds"] = codec.LastNativeSeconds, ["tiles"] = codec.LastTileCount,
            ["png_sha256"] = sha, ["peak_device_bytes"] = PeakBetween(t0, t1),
        });
        Console.WriteLine($"[h3img-ab] vae round {r + 1}: {t1 - t0:F3}s ({codec.LastTileCount} tiles) sha {sha}");
    }
    result["vae_decodes"] = decodes;
    result["vae_compiles"] = codec.Compiles;
    stop.Cancel();
    sampler.Join();
    Save();
    codec.Dispose();
    // The backends go before the process's static destructors (as the CLI's exit does), or ggml-metal aborts in them.
    GgmlBasicOps.Shutdown();
    return 0;
}

// ---- --lora-rows <out.json>: the plug-ins' updates as the backend applies them, for an independent merge
if (Value("--lora-rows") is { } rowsOut)
{
    int[] blocks = (Value("--rows-blocks") ?? "0,25,49").Split(',').Select(v => int.Parse(v, CultureInfo.InvariantCulture)).ToArray();
    int tokens = int.Parse(Value("--rows-tokens") ?? "1024", CultureInfo.InvariantCulture);
    const int Kept = 8;
    using var file = new MiniMaxH3WeightFile(model);
    MiniMaxH3Config cfg = MiniMaxH3Config.Detect(file.Tensors);
    var sw = Stopwatch.StartNew();
    using var set = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(loraSpecs), cfg, backend);
    double loadSeconds = sw.Elapsed.TotalSeconds;
    var executor = GgmlGraphExecutor.For(backend switch
    {
        BackendType.GgmlMetal => GgmlBackendType.Metal,
        BackendType.GgmlCuda => GgmlBackendType.Cuda,
        _ => GgmlBackendType.Cpu,
    });
    // x[t][i]: a 32-bit LCG (exact in any language) mapped to [-1, 1): eng/validation/MiniMaxH3ImageLoraRows/lora_rows.py draws the same.
    static float[] Activations(long input, int count, uint seed)
    {
        var x = new float[input * count];
        uint state = seed;
        for (long k = 0; k < x.LongLength; k++)
        {
            state = unchecked(state * 1664525u + 1013904223u);
            x[k] = (float)((state >> 8) / 16777216.0 * 2.0 - 1.0);
        }
        return x;
    }
    var outputs = new JsonArray();
    var projections = new[]
    {
        (Name: "attn.qkv_proj", P: MiniMaxH3ImageLoraSet.Projection.Qkv),
        (Name: "attn.out_proj", P: MiniMaxH3ImageLoraSet.Projection.Out),
        (Name: "mlp.fc1", P: MiniMaxH3ImageLoraSet.Projection.Fc1),
        (Name: "mlp.fc2", P: MiniMaxH3ImageLoraSet.Projection.Fc2),
    };
    foreach (var (refiner, block) in blocks.Select(b => (false, b)).Concat(new[] { (true, 0) }))
        foreach (var (pname, projection) in projections)
        {
            var factors = set.Get(refiner, block, projection);
            if (factors == null) continue;
            string tensor = cfg.Prefix + (refiner ? "token_refiner.blocks." : "blocks.") + block + "." + pname + ".weight";
            GgufTensorInfo info = file.Tensors[tensor];
            if (!file.TryGetTensorDataPointer(info, out IntPtr wPtr)) throw new InvalidOperationException("cannot map " + tensor);
            long input = (long)info.Shape[0], output = (long)info.Shape[1];
            long wBytes = input / GgufFile.GetBlockSize(info.Type) * GgufFile.GetTypeSize(info.Type) * output;
            uint seed = (uint)(1000 * block + 10 * (int)projection + (refiner ? 7 : 0));
            float[] x = Activations(input, tokens, seed);
            // The update is y - y_twin, where the twin runs the same LoraLinear with zero up factors: the same kernels
            // and the same base arithmetic, so a backend whose adapted path computes W x differently from a plain
            // MulMat (ggml_cuda's K-concat GEMM on Q6_K weights, for one) does not put that difference in the update.
            var zeroBlocks = factors.Parts.Select(p =>
            {
                IntPtr raw = System.Runtime.InteropServices.Marshal.AllocHGlobal((nint)(p.UpBytes + 4096));
                for (long k = 0; k < p.UpBytes + 4096; k += 1 << 20)
                    System.Runtime.InteropServices.Marshal.Copy(new byte[Math.Min(1 << 20, p.UpBytes + 4096 - k)], 0, raw + (nint)k, (int)Math.Min(1 << 20, p.UpBytes + 4096 - k));
                return raw;
            }).ToArray();
            IntPtr[] zeroUps = zeroBlocks.Select(raw => (IntPtr)(((long)raw + 4095) & ~4095L)).ToArray();
            float[] Run(bool adapted, bool zeroUp = false)
            {
                using var e = new TensorSharp.Models.Graph.GraphEmitter(executor, record: true);
                var b = e.Builder;
                e.MarkWideTokenProducts();
                int xi = e.InputF32(x, input, tokens);
                int w = e.HostWeight((int)info.Type, stackalloc long[] { input, output }, wPtr, wBytes);
                int y;
                if (adapted)
                {
                    var parts = factors.Parts.Select((p, k) => new TensorSharp.Graph.GraphDitRegions.LoraPart(p.Slot,
                        e.HostWeight(p.Format, new[] { p.In, (long)p.Rank }, p.Down, p.DownBytes),
                        e.HostWeight(p.Format, new[] { (long)p.Rank, p.Out }, zeroUp ? zeroUps[k] : p.Up, p.UpBytes))).ToArray();
                    y = TensorSharp.Graph.GraphDitRegions.LoraLinear(b, xi, w, parts, output, factors.SlotRows, tokens);
                }
                else y = b.Op(TensorSharp.Graph.GraphOp.MulMat, w, xi);
                e.Output(y, output * tokens);
                using var plan = e.Compile(replayable: false);
                if (!e.Run(plan)) throw new InvalidOperationException("the LoraLinear program failed: " + plan.LastError);
                var kept = new float[output * Kept];
                Array.Copy(plan.Outputs[0], kept, kept.Length);
                return kept;
            }
            float[] y1 = Run(true), y0 = Run(false), yTwin = Run(true, zeroUp: true);
            foreach (IntPtr raw in zeroBlocks)
                System.Runtime.InteropServices.Marshal.FreeHGlobal(raw);
            string B64(float[] v) { var raw = new byte[v.Length * 4]; Buffer.BlockCopy(v, 0, raw, 0, raw.Length); return Convert.ToBase64String(raw); }
            outputs.Add(new JsonObject
            {
                ["refiner"] = refiner, ["block"] = block, ["projection"] = pname, ["tensor"] = tensor, ["weight_type"] = info.Type.ToString(),
                ["in"] = input, ["out"] = output, ["tokens"] = tokens, ["kept_tokens"] = Kept, ["x_seed"] = seed,
                ["slot_rows"] = factors.SlotRows,
                ["parts"] = new JsonArray(factors.Parts.Select(p => (JsonNode)new JsonObject { ["slot"] = p.Slot, ["rank"] = p.Rank, ["format"] = p.Format }).ToArray()),
                ["y"] = B64(y1), ["y_base"] = B64(y0), ["y_twin"] = B64(yTwin),
            });
            Console.WriteLine($"[h3img-ab] lora-rows {(refiner ? "refiner " : "")}block {block} {pname}: {factors.Parts.Length} part(s), rank {string.Join("/", factors.Parts.Select(p => p.Rank))}");
        }
    result["lora_rows"] = outputs;
    result["lora_load_seconds"] = loadSeconds;
    result["lora_summary"] = set.Summary;
    result["lora_factor_bytes"] = set.FactorBytes;
    stop.Cancel();
    sampler.Join();
    File.WriteAllText(rowsOut, result.ToJsonString(new JsonSerializerOptions { WriteIndented = false }));
    GgmlBasicOps.Shutdown();
    return 0;
}

// ---- --parity-dump: the first job up to its first step, recorded
if (Value("--parity-dump") is { } dumpDir)
{
    if (engine != "runtime") throw new ArgumentException("--parity-dump records the runtime");
    JsonNode first = prompts.FirstOrDefault(x => (string?)x!["id"] == jobs[0]) ?? throw new ArgumentException($"no prompt '{jobs[0]}'");
    string baseDir = Path.GetDirectoryName(Path.GetFullPath(Required("--prompts")))!;
    var dumpResult = ParityDump.Run(backend, model, te, vae, first, baseDir, size, Path.GetFullPath(dumpDir));
    result["parity_dump"] = dumpResult;
    stop.Cancel();
    sampler.Join();
    Save();
    GgmlBasicOps.Shutdown();
    return 0;
}

// ---- --dump-cond: the jobs' text conditioning from their own encoder, released before the model loads
if (dumpCond != null)
{
    Directory.CreateDirectory(dumpCond);
    using var encoder = new MiniMaxH3ImageConditioner(te, backend);
    foreach (string id in jobs)
    {
        JsonNode p = prompts.FirstOrDefault(x => (string?)x!["id"] == id) ?? throw new ArgumentException($"no prompt '{id}'");
        var cond = encoder.Encode((string)p["prompt"]!, Array.Empty<TensorSharp.Models.SharedDiffusion.PresentedImage>(), CancellationToken.None);
        byte[] raw = new byte[cond.Hidden.Length * sizeof(float)];
        Buffer.BlockCopy(cond.Hidden, 0, raw, 0, raw.Length);
        File.WriteAllBytes(Path.Combine(dumpCond, id + ".hidden.f32"), raw);
        int w0 = size > 0 ? size : (int)p["width"]!, h0 = size > 0 ? size : (int)p["height"]!;
        File.WriteAllText(Path.Combine(dumpCond, id + ".json"), new JsonObject
        {
            ["id"] = id, ["rows"] = cond.Rows, ["width"] = cond.Width, ["seed"] = (long)p["seed"]!, ["image_width"] = w0, ["image_height"] = h0,
            ["prompt"] = (string)p["prompt"]!,
            ["modalities"] = cond.Modalities == null ? null : new JsonArray(cond.Modalities.Select(m => (JsonNode)m).ToArray()),
            ["hidden_sha256"] = Convert.ToHexStringLower(SHA256.HashData(raw)),
        }.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"[h3img-ab] conditioning {id}: {cond.Rows} rows x {cond.Width}");
    }
}

// ---- --vae-dump <dir>: the VAE encoder alone on the first job's references
if (Value("--vae-dump") is { } vaeDumpDir)
{
    JsonNode first = prompts.FirstOrDefault(x => (string?)x!["id"] == jobs[0]) ?? throw new ArgumentException($"no prompt '{jobs[0]}'");
    result["vae_dump"] = ParityDump.VaeDump(backend, vae, first, Path.GetDirectoryName(Path.GetFullPath(Required("--prompts")))!, Path.GetFullPath(vaeDumpDir));
    stop.Cancel();
    sampler.Join();
    Save();
    GgmlBasicOps.Shutdown();
    return 0;
}

// ---- --te-attention-check [--te-chunk N[,N...]]: the text encoder's query-block attention (each size, or the
// conditioner's choice) and its flash attention against the whole materialized scores, on the first job
if (Flag("--te-attention-check"))
{
    JsonNode first = prompts.FirstOrDefault(x => (string?)x!["id"] == jobs[0]) ?? throw new ArgumentException($"no prompt '{jobs[0]}'");
    int[] chunks = Value("--te-chunk") is { } c
        ? c.Split(',').Select(x => int.Parse(x, System.Globalization.CultureInfo.InvariantCulture)).Where(x => x > 0).ToArray()
        : Array.Empty<int>();
    result["te_attention_check"] = ParityDump.AttentionCheck(backend, te, first, Path.GetDirectoryName(Path.GetFullPath(Required("--prompts")))!, chunks);
    stop.Cancel();
    sampler.Join();
    Save();
    GgmlBasicOps.Shutdown();
    return 0;
}

// ---- load
double loadStart = clock.Elapsed.TotalSeconds;
IImageGenerationModel image;
ModelBase loaded;
MiniMaxH3ImageDiffusionModel runtime;
var phases = new List<(DiffusionPhase Phase, double T)>();
{
    loaded = ModelBase.Create(model, backend);
    runtime = loaded as MiniMaxH3ImageDiffusionModel
        ?? throw new InvalidOperationException($"the registry built {loaded.GetType().Name}, not the runtime's MiniMax-H3-Image model");
    runtime.ProgressObserver = p => { lock (phases) phases.Add((p.Phase, clock.Elapsed.TotalSeconds)); };
    image = runtime;
    if (loraSpecs.Count > 0)
    {
        double loraStart = clock.Elapsed.TotalSeconds;
        runtime.SetLoras(loraSpecs);
        result["lora_load_seconds"] = clock.Elapsed.TotalSeconds - loraStart;
        result["lora_factor_bytes"] = runtime.Loras?.FactorBytes ?? 0;
        result["lora_summary"] = runtime.Loras?.Summary;
    }
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
    string promptsDir = Path.GetDirectoryName(Path.GetFullPath(Required("--prompts")))!;
    List<RgbImage> inputs = p["images"] is JsonArray imgs
        ? imgs.Select(x => ImageIO.Load(Path.Combine(promptsDir, (string)x!), preserveAlpha: true)).ToList()
        : new List<RgbImage>();
    var stepTimes = new List<double>();
    double last = 0;
    var parameters = new QwenImageParams
    {
        Width = w,
        Height = h,
        Seed = seed,
        KeepSourceSize = (bool?)p["keepSourceSize"] ?? false,
        Mask = p["mask"] is { } maskPath ? ImageIO.Load(Path.Combine(promptsDir, (string)maskPath!), preserveAlpha: true) : null,
        MaskMode = (string?)p["maskMode"] == "alpha" ? QwenImageMaskMode.Alpha : QwenImageMaskMode.Grayscale,
        MaskInvert = (bool?)p["maskInvert"] ?? false,
        MaskFeather = (int?)p["maskFeather"] ?? 0,
        MaskCrop = (bool?)p["maskCrop"] ?? false,
        MaskCropPadding = (int?)p["maskCropPadding"] ?? 64,
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
    var job = new JsonObject
    {
        ["n"] = n + 1, ["id"] = id, ["seed"] = seed, ["width"] = w, ["height"] = h, ["load_before"] = load0,
        ["references"] = new JsonArray(inputs.Select(i => (JsonNode)$"{i.Width}x{i.Height}").ToArray()),
    };
    if (parameters.Mask != null)
        job["mask"] = $"{(string)p["mask"]!} ({parameters.MaskMode}{(parameters.MaskInvert ? ", inverted" : "")}, feather {parameters.MaskFeather}" +
                      $"{(parameters.MaskCrop ? $", crop + {parameters.MaskCropPadding}" : "")})";
    try
    {
        RgbImage picture = inputs.Count > 0
            ? image.EditImage((string)p["prompt"]!, inputs, parameters)
            : image.GenerateImage((string)p["prompt"]!, parameters);
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
        {
            var t = runtime.LastResult!.Timings;
            (text, denoise, decode, total, steps) = (t.Text, t.Denoise, t.Decode, t.Total, t.Steps);
            tiles = runtime.LastTileCount;
            native = runtime.LastDecodeNativeSeconds;
            job["residency"] = runtime.LastResult.Residency?.Describe();
            var recipe = runtime.LastResult.Recipe;
            job["recipe"] = new JsonObject
            {
                ["steps"] = recipe.Steps,
                ["sigmas"] = new JsonArray(recipe.Sigmas.Select(x => (JsonNode)x).ToArray()),
                ["heads"] = recipe.Recipe.Heads.ToString(),
                ["describe"] = recipe.Recipe.Describe(recipe.Steps),
                ["adapters"] = new JsonArray(recipe.Adapters.Select(a => (JsonNode)a).ToArray()),
            };
            job["text_tokens"] = runtime.LastTextTokens;
            // Per phase: from the phase's entry to the next one's (the run's end for the last).
            var marks = new List<(DiffusionPhase, double)>();
            lock (phases) marks.AddRange(phases.Where(x => x.Phase is DiffusionPhase.Condition or DiffusionPhase.EncodeReferences
                    or DiffusionPhase.Denoise or DiffusionPhase.Decode or DiffusionPhase.Composite)
                .GroupBy(x => x.Phase).Select(g => (g.Key, g.First().T)));
            job["vision_seconds"] = t.Vision;
            job["vae_encode_seconds"] = t.VaeEncode;
            job["vae_encode_tiles"] = runtime.LastEncodeTiles;
            job["output_size"] = $"{picture.Width}x{picture.Height}";
            job["sampling_size"] = $"{runtime.LastResult.Recipe.Canvas.Sampling.Width}x{runtime.LastResult.Recipe.Canvas.Sampling.Height}";
            var peaks = new JsonObject();
            for (int i = 0; i < marks.Count; i++)
                peaks[marks[i].Item1.ToString()] = PeakBetween(marks[i].Item2, i + 1 < marks.Count ? marks[i + 1].Item2 : end);
            job["phase_peak_device_bytes"] = peaks;
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
        Console.WriteLine($"[h3img-ab] {label} job {n + 1} {id} {w}x{h} refs {inputs.Count}: text {text:F2}s denoise {denoise:F2}s decode {decode:F2}s " +
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
