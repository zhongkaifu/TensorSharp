using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models;
using TensorSharp.Models.Ltx25;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Runtime;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i += 2)
    if (i + 1 >= args.Length || !args[i].StartsWith("--") || !options.TryAdd(args[i][2..], args[i + 1]))
        throw new ArgumentException("Expected unique --option value pairs.");
string Required(string name) => options.GetValueOrDefault(name) ?? throw new ArgumentException("Missing --" + name);
int Number(string name, int fallback) => int.Parse(options.GetValueOrDefault(name, fallback.ToString()), CultureInfo.InvariantCulture);
string modelPath = Path.GetFullPath(Required("model")), output = Path.GetFullPath(Required("output"));
BackendType backend = Enum.Parse<BackendType>(Required("backend"), true);
int repeats = Number("repeats", 1);
if (repeats < 1) throw new ArgumentException("Repeats must be positive.");
if (File.Exists(output)) throw new IOException("Refusing to overwrite an existing generation report.");
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
string? mode = options.GetValueOrDefault("mode");
bool identityMode = mode == "ref";
var parameters = new VideoGenerationParams
{
    Mode = mode!,
    Width = Number("width", identityMode ? 768 : 512), Height = Number("height", identityMode ? 448 : 512), Frames = Number("frames", identityMode ? 121 : 17),
    Fps = Number("fps", 24), Steps = 8, CfgScale = 1, Sampler = options.GetValueOrDefault("sampler", Ltx25GenerationSettings.AncestralSampler), Seed = long.Parse(options.GetValueOrDefault("seed", "42"), CultureInfo.InvariantCulture),
    GenerateAudio = options.GetValueOrDefault("audio", "1") == "1",
};
string prompt = options.GetValueOrDefault("prompt", "A red toy car drives slowly along a quiet road in warm afternoon light.");
string? extraPrompt = options.GetValueOrDefault("extra-prompt");
parameters.ImagePath = options.GetValueOrDefault("image")!;
parameters.EndImagePath = options.GetValueOrDefault("end-image")!;
if (options.TryGetValue("reference-images", out string? referenceImages))
    parameters.ReferenceImagePaths = JsonSerializer.Deserialize<string[]>(File.ReadAllText(referenceImages))
        ?? throw new ArgumentException("Reference image file must contain a JSON array of paths.");
if (options.TryGetValue("image-conditions", out string? imageConditions))
{
    using var input = JsonDocument.Parse(File.ReadAllText(imageConditions));
    parameters.ImageConditions = input.RootElement.EnumerateArray().Select(item => new VideoImageCondition
    {
        ImagePath = item.GetProperty("image_path").GetString() ?? throw new ArgumentException("Each guide requires image_path."),
        FrameIndex = item.GetProperty("frame_index").GetInt32(),
        Strength = item.TryGetProperty("strength", out var strength) ? strength.GetSingle() : 1,
    }).ToArray();
}
var referenceInputs = new List<object>();
void RecordReference(string? path, int? frame, float strength, string role)
{
    if (string.IsNullOrWhiteSpace(path)) return;
    string absolute = Path.GetFullPath(path);
    referenceInputs.Add(new { path = absolute, frame_index = frame, strength, role,
        bytes = new FileInfo(absolute).Length, sha256 = HashFile(absolute) });
}
RecordReference(parameters.ImagePath, 0, 1, "first-frame");
RecordReference(parameters.EndImagePath, parameters.Frames - 1, 1, "last-frame");
if (parameters.ReferenceImagePaths is { Count: > 0 } paths)
    for (int i = 0; i < paths.Count; ++i)
        RecordReference(paths[i], identityMode ? null : paths.Count == 1 ? 0 : (int)Math.Round((double)i * (parameters.Frames - 1) / (paths.Count - 1)),
            1, identityMode ? "identity-reference-sheet-panel" : "ordered-keyframe");
if (parameters.ImageConditions != null)
    foreach (var condition in parameters.ImageConditions)
        RecordReference(condition.ImagePath, condition.FrameIndex == -1 ? parameters.Frames - 1 : condition.FrameIndex, condition.Strength, "explicit-keyframe");
var report = new Dictionary<string, object?>
{
    ["completed"] = false, ["created_utc"] = DateTimeOffset.UtcNow,
    ["scope"] = "Real registered TensorSharp model: tokenization, Gemma conditioning, aggregation/connectors, full48-block eight-step AV Euler sampling (ancestral unless --sampler euler), video VAE, audio VAE/vocoder/BWE and MP4 encoding. No TensorAgent or HTTP path in this probe.",
    ["model"] = modelPath, ["backend_requested"] = backend.ToString(), ["prompt"] = prompt,
    ["mode"] = mode, ["sampler"] = parameters.Sampler, ["reference_inputs"] = referenceInputs, ["settings"] = Ltx25GenerationSettings.Resolve(parameters, supportsIdentityReferences: identityMode), ["sigmas"] = Ltx25Scheduler.DistilledSigmas(),
    ["environment"] = new[] { "MLX_ENABLE_TF32", "TS_MLX_CACHE_LIMIT_MB", "TS_MLX_WIRED_LIMIT_MB",
        "TS_CUDA_QMM_F16GEMM_MIN_ROWS", "TS_MLX_Q5K_REPACK" }
        .ToDictionary(name => name, Environment.GetEnvironmentVariable),
    ["timing_scope"] = "Model factory separately; each generation includes sequential phase weight loading/release, excludes codec time. Cached file pages and native/JIT caches may be warm. Neither cold filesystem nor always-resident weight latency is established.",
};
var executions = new List<object>();
report["executions"] = executions;
void Save() => File.WriteAllText(output, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + "\n");
string HashFile(string path) { using var file = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(file)); }
void WriteFloats(string path, float[] values)
{
    using var file = new FileStream(path, FileMode.CreateNew);
    file.Write(MemoryMarshal.AsBytes(values.AsSpan()));
}
Save();
try
{
    report["managed_binaries"] = new[] { Assembly.GetExecutingAssembly(), typeof(Ltx25Model).Assembly, typeof(CudaAllocator).Assembly,
        typeof(MlxAllocator).Assembly, typeof(GgufFile).Assembly }.Select(assembly => new { name = assembly.GetName().Name, path = assembly.Location, sha256 = HashFile(assembly.Location) }).ToArray();
    if (backend is BackendType.Cuda or BackendType.GgmlCuda)
        report["cuda_modules"] = new[] { "tensorsharp_kernels.ptx", "tensorsharp_ltx25.ptx" }.Select(name =>
        {
            string path = Path.Combine(AppContext.BaseDirectory, "cuda_kernels", name);
            return new { name, path, sha256 = HashFile(path) };
        }).ToArray();
    var clock = Stopwatch.StartNew();
    using ModelBase model = ModelBase.Create(modelPath, backend);
    report["model_factory_ms"] = clock.Elapsed.TotalMilliseconds;
    if (model is not Ltx25Model videoModel) throw new InvalidOperationException("Architecture registry did not create the LTX generation model.");
    report["model_family"] = videoModel.VideoModelFamily;
    report["supports_identity_references"] = videoModel.SupportsReferenceConditioning;
    if (identityMode)
    {
        string adapter = Environment.GetEnvironmentVariable("TS_LTX25_INGREDIENTS_LORA")
            ?? Path.Combine(Path.GetDirectoryName(modelPath)!, "loras", "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors");
        if (!videoModel.SupportsReferenceConditioning || !File.Exists(adapter))
            throw new InvalidOperationException("The Ingredients identity adapter was not activated.");
        report["identity_adapter"] = new { path = Path.GetFullPath(adapter), bytes = new FileInfo(adapter).Length, sha256 = HashFile(adapter) };
    }
    for (int repeat = 0; repeat < repeats + (extraPrompt != null ? 1 : 0); ++repeat)
    {
        string currentPrompt = repeat < repeats ? prompt : extraPrompt!;
        var events = new List<object>();
        parameters.OnProgress = progress =>
            events.Add(new { progress.Phase, progress.Step, progress.TotalSteps, progress.ElapsedSeconds, progress.Detail });
        long cacheHitsBefore = videoModel.PromptCacheHits;
        clock.Restart();
        GeneratedVideo result = videoModel.GenerateVideo(currentPrompt, parameters);
        double generationMs = clock.Elapsed.TotalMilliseconds;
        if (result.Frames.Length != parameters.Frames || result.Fps != parameters.Fps ||
            result.Frames.Any(frame => frame.Width != parameters.Width || frame.Height != parameters.Height || frame.Pixels.Any(value => !float.IsFinite(value))))
            throw new InvalidOperationException("Decoded video geometry or samples are invalid.");
        if (parameters.GenerateAudio && (result.Audio is not { SampleRate: 48000, ChannelCount: 2, SampleCount: > 0 } ||
            result.Audio.Channels.Any(channel => channel.Any(value => !float.IsFinite(value)))))
            throw new InvalidOperationException("Decoded48k stereo audio is invalid.");
        string stem = output + "." + repeat;
        using (var raw = new FileStream(stem + ".rgb.f32", FileMode.CreateNew))
            foreach (var frame in result.Frames) raw.Write(MemoryMarshal.AsBytes(frame.Pixels.AsSpan()));
        if (result.Audio != null)
            for (int channel = 0; channel < result.Audio.ChannelCount; ++channel)
                WriteFloats(stem + ".audio-" + channel + ".f32", result.Audio.Channels[channel]);
        clock.Restart();
        bool muxed = false;
        string codec = result.Audio != null
            ? VideoIO.SaveMp4(stem + ".mp4", result.Frames, result.Fps, result.Audio, out muxed)
            : VideoIO.SaveMp4(stem + ".mp4", result.Frames, result.Fps);
        double codecMs = clock.Elapsed.TotalMilliseconds;
        if (parameters.GenerateAudio && !muxed) throw new InvalidOperationException("The output encoder did not mux the generated audio.");
        float[] first = result.Frames[0].Pixels;
        double mean = first.Average(value => (double)value), variance = first.Average(value => (value - mean) * (value - mean));
        if (variance < 1e-8) throw new InvalidOperationException("The generated first frame is degenerate.");
        executions.Add(new { repeat, prompt = currentPrompt, kind = repeat < repeats ? "benchmark" : "uncached-prompt-lifecycle",
            generation_ms = generationMs, codec_ms = codecMs, codec, audio_muxed = muxed,
            conditioning_cache_hit = videoModel.PromptCacheHits > cacheHitsBefore,
            frames = result.Frames.Length, fps = result.Fps, audio_samples = result.Audio?.SampleCount,
            audio_seconds = result.Audio?.DurationSeconds, first_frame_variance = variance,
            video_sha256 = HashFile(stem + ".rgb.f32"), media_sha256 = HashFile(stem + ".mp4"),
            media_path = stem + ".mp4", phase_events = events });
        report["process_peak_working_set_bytes"] = Process.GetCurrentProcess().PeakWorkingSet64;
        if (backend is BackendType.Mlx or BackendType.GgmlMetal)
            report["mlx_memory"] = MlxBackend.GetMemorySnapshot();
        Save();
    }
    // Exercise cancellation on the backend worker without entering a denoise
    // pass. The successful requests above separately cover cache reuse.
    using (var cancelled = new CancellationTokenSource())
    {
        cancelled.Cancel();
        var cancelledRequest = new VideoGenerationParams { CancellationToken = cancelled.Token };
        try
        {
            videoModel.GenerateVideo(prompt, cancelledRequest);
            throw new InvalidOperationException("An already cancelled video request executed.");
        }
        catch (OperationCanceledException error) when (error.CancellationToken == cancelled.Token)
        { report["cancelled_before_conditioning"] = true; }
    }
    videoModel.TrimIdleMemory();
    if (backend is BackendType.Mlx or BackendType.GgmlMetal)
        report["mlx_memory_after_idle_trim"] = MlxBackend.GetMemorySnapshot();
    report["completed"] = true;
}
catch (Exception error)
{
    report["error"] = error.ToString();
    throw;
}
finally { Save(); }
