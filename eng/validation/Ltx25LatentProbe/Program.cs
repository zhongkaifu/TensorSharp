using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Models.Ltx25;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Runtime;

if (args is ["--help"])
{
    Console.WriteLine("""
        Capture final LTX latents once and compare decoders without resampling.
        Generate: --operation generate --model MODEL --prompt-file FILE --image IMAGE
                  --seed INT64 --width 1024 --height 1024 --frames 145 --fps 24
                  --backend ggml_metal --output NEW_DIRECTORY --decoders both
                  --convolutional-vae FILE --diffusion-vae FILE
        First forward: --operation forward with the same generation options; stops
                       after first complete denoiser Forward, before sampling/decode.
        Re-decode: --operation decode --capture capture.json --backend mlx
                   --output NEW_DIRECTORY --decoders convolutional|diffusion|both
                   --convolutional-vae FILE --diffusion-vae FILE
        Validate only: --operation verify --capture capture.json --output NEW_DIRECTORY
        Optional: --sampler euler_ancestral|euler, --mode i2v|t2v, --prompt TEXT,
                  --image-strength 0..1 (explicit both-stage strength; omitted uses
                  production image policy), --end-image IMAGE,
                  --first-stage-image-strength 0..1 (overrides first pass only),
                  --save-rgb 1 (default 0), --decoders none.
        Both decoder paths consume identical normalized channel-major video latents.
        MP4s are silent; joint audio latents are captured but audio is not decoded.
        Use MLX_ENABLE_TF32=0 at process launch for an explicit F32 comparison.
        """);
    return;
}

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i += 2)
    if (i + 1 >= args.Length || !args[i].StartsWith("--") || !options.TryAdd(args[i][2..], args[i + 1]))
        throw new ArgumentException("Expected unique --option value pairs; use --help for usage.");
string[] allowed = ["operation", "model", "prompt-file", "prompt", "image", "end-image", "seed", "width", "height", "frames", "fps",
    "backend", "output", "decoders", "convolutional-vae", "diffusion-vae", "capture", "sampler", "mode", "save-rgb", "image-strength", "first-stage-image-strength"];
foreach (string key in options.Keys)
    if (!allowed.Contains(key)) throw new ArgumentException("Unknown option --" + key);
string Required(string name) => options.GetValueOrDefault(name) ?? throw new ArgumentException("Missing --" + name);
int Number(string name, int fallback) => int.Parse(options.GetValueOrDefault(name, fallback.ToString(CultureInfo.InvariantCulture)), CultureInfo.InvariantCulture);
string DecoderCheckpoint(string model, string decoder) => Path.GetFullPath(options.GetValueOrDefault(decoder + "-vae") ??
    Path.Combine(Path.GetDirectoryName(model)!, "vae", decoder == "convolutional" ? "ltx-2.5-video-vae-conv-bf16.safetensors" : "ltx-2.5-video-vae-bf16.safetensors"));
string operation = Required("operation");
if (operation is not ("generate" or "forward" or "decode" or "verify")) throw new ArgumentException("Unknown operation.");
bool generatesInputs = operation is "generate" or "forward";
bool imageStrengthExplicit = options.ContainsKey("image-strength");
float imageStrength = float.Parse(options.GetValueOrDefault("image-strength", "1"), CultureInfo.InvariantCulture);
if (!float.IsFinite(imageStrength) || imageStrength < 0 || imageStrength > 1)
    throw new ArgumentException("--image-strength must be finite and in [0,1].");
if (options.ContainsKey("image-strength") && (!generatesInputs || !options.ContainsKey("image")))
    throw new ArgumentException("--image-strength requires --operation generate or forward and --image.");
float? firstStageImageStrength = options.TryGetValue("first-stage-image-strength", out string? firstStrength)
    ? float.Parse(firstStrength, CultureInfo.InvariantCulture) : null;
if (firstStageImageStrength is float firstValue && (!float.IsFinite(firstValue) || firstValue < 0 || firstValue > 1))
    throw new ArgumentException("--first-stage-image-strength must be finite and in [0,1].");
if (firstStageImageStrength.HasValue && (!generatesInputs || !options.ContainsKey("image")))
    throw new ArgumentException("--first-stage-image-strength requires --operation generate or forward and --image.");
string output = Path.GetFullPath(Required("output"));
if (Directory.Exists(output) && Directory.EnumerateFileSystemEntries(output).Any())
    throw new IOException("Refusing to replace a nonempty output directory.");
if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("The capture format uses little-endian Float32.");
string[] decoders = options.GetValueOrDefault("decoders", operation is "verify" or "forward" ? "none" : "both") switch
{
    "none" => [], "both" => ["convolutional", "diffusion"], "convolutional" => ["convolutional"], "diffusion" => ["diffusion"],
    _ => throw new ArgumentException("Decoders must be none, convolutional, diffusion or both."),
};
if (operation == "verify" && decoders.Length != 0) throw new ArgumentException("Verify cannot run a decoder.");
if (operation == "forward" && decoders.Length != 0) throw new ArgumentException("Forward capture stops before decoding; use --decoders none.");
BackendType backend = operation == "verify" ? BackendType.Cpu : Required("backend").ToLowerInvariant() switch
{
    "cpu" or "ggml_cpu" => BackendType.Cpu,
    "mlx" or "ggml_metal" => BackendType.Mlx,
    "cuda" or "ggml_cuda" => BackendType.Cuda,
    _ => throw new ArgumentException("Supported backends: cpu, mlx/ggml_metal, cuda/ggml_cuda."),
};
if (backend == BackendType.Cuda && decoders.Contains("diffusion"))
    throw new NotSupportedException("The diffusion VAE supports CPU and MLX only.");
bool saveRgb = options.GetValueOrDefault("save-rgb", "0") switch
{
    "0" => false, "1" => true, _ => throw new ArgumentException("--save-rgb must be 0 or 1."),
};
Directory.CreateDirectory(output);
var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
var report = new Dictionary<string, object?>
{
    ["completed"] = false, ["created_utc"] = DateTimeOffset.UtcNow, ["operation"] = operation,
    ["backend"] = backend.ToString(), ["save_rgb"] = saveRgb,
    ["scope"] = operation == "forward"
        ? "Native image/text preprocessing and first full denoiser Forward only. Captures actual sampler Predict input and output velocity before any sampler update. No complete generation, VAE decode, HTTP/Studio or perceptual-quality assertion."
        : "Native LTX pipeline through final AV sampling, or re-decode of a validated prior capture. Decoder A/B uses the same final video latents and seed. Silent MP4s; no HTTP/Studio, audio decode or perceptual-quality assertion.",
    ["timing_scope"] = "Diagnostic wall time; includes synchronous capture/validation and warm filesystem/native caches. This is not a controlled benchmark.",
    ["managed_binaries"] = new[] { Assembly.GetExecutingAssembly(), typeof(Ltx25Model).Assembly, typeof(MlxAllocator).Assembly,
        typeof(CudaAllocator).Assembly }.Distinct().Select(assembly => new { path = assembly.Location, sha256 = HashFile(assembly.Location) }).ToArray(),
};
var executions = new List<object>();
report["decodes"] = executions;
void SaveReport() => File.WriteAllText(Path.Combine(output, "report.json"), JsonSerializer.Serialize(report, jsonOptions) + "\n");
void SaveCapture(Capture capture) => File.WriteAllText(Path.Combine(output, "capture.json"), JsonSerializer.Serialize(capture, jsonOptions) + "\n");
SaveReport();
try
{
    string capturePath;
    if (generatesInputs)
    {
        if (options.ContainsKey("capture")) throw new ArgumentException("Generate cannot accept a previous capture.");
        if (options.ContainsKey("prompt") && options.ContainsKey("prompt-file")) throw new ArgumentException("Choose --prompt or --prompt-file.");
        string prompt = options.TryGetValue("prompt-file", out string? promptFile) ? File.ReadAllText(promptFile).Trim() : Required("prompt");
        if (string.IsNullOrWhiteSpace(prompt)) throw new ArgumentException("The prompt is empty.");
        string modelPath = Path.GetFullPath(Required("model"));
        foreach (string decoder in decoders)
            if (!File.Exists(DecoderCheckpoint(modelPath, decoder)))
                throw new FileNotFoundException("Requested decoder checkpoint is missing.", DecoderCheckpoint(modelPath, decoder));
        long seed = long.Parse(Required("seed"), CultureInfo.InvariantCulture);
        if (seed < 0) throw new ArgumentException("Provide an exact nonnegative Int64 seed for reproducible capture.");
        string? firstImagePath = options.TryGetValue("image", out string? image) ? Path.GetFullPath(image) : null;
        var parameters = new VideoGenerationParams
        {
            Width = Number("width", 1024), Height = Number("height", 1024), Frames = Number("frames", 145), Fps = Number("fps", 24),
            Steps = 8, CfgScale = 1, Seed = seed, GenerateAudio = false,
            ImagePath = firstImagePath!,
            EndImagePath = options.TryGetValue("end-image", out string? endImage) ? Path.GetFullPath(endImage) : null!,
            Mode = options.GetValueOrDefault("mode", options.ContainsKey("image") ? "i2v" : "t2v"),
            Sampler = options.GetValueOrDefault("sampler", Ltx25GenerationSettings.AncestralSampler),
        };
        if (imageStrengthExplicit)
        {
            // An explicitly supplied 1 is still a user choice. Explicit guides
            // retain their strength in both stages; only omitted flags use the
            // production shorthand image policy.
            parameters.ImagePath = null!;
            parameters.ImageConditions = [new VideoImageCondition { ImagePath = firstImagePath!, FrameIndex = 0, Strength = imageStrength }];
        }
        Ltx25GenerationSettings requestedSettings = Ltx25GenerationSettings.Resolve(parameters);
        if (firstStageImageStrength.HasValue && !requestedSettings.TwoStage)
            throw new ArgumentException("--first-stage-image-strength requires a two-stage generation mode.");
        IReadOnlyList<VideoImageCondition> resolvedGuides = Ltx25ImageConditioning.Resolve(parameters, requestedSettings.Frames);
        var pipelineType = typeof(Ltx25Model).Assembly.GetType("TensorSharp.Models.Ltx25.Ltx25Pipeline", throwOnError: true)!;
        var strengthResolver = pipelineType.GetMethod("ResolveFirstStageImageStrength", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new MissingMethodException("Build the shared first-stage image policy before running this probe.");
        float? resolvedFirstStageOverride = (float?)strengthResolver.Invoke(null,
            [parameters, requestedSettings.TwoStage, resolvedGuides, firstStageImageStrength]);
        float? refinementStrength = resolvedGuides.FirstOrDefault(guide => guide.FrameIndex == 0)?.Strength;
        float? effectiveFirstStrength = refinementStrength.HasValue ? resolvedFirstStageOverride ?? refinementStrength : null;
        var events = new List<VideoGenerationProgress>();
        parameters.OnProgress = progress => events.Add(progress);
        report["phase_events"] = events;
        report["model"] = new { path = modelPath, bytes = new FileInfo(modelPath).Length, last_write_utc = File.GetLastWriteTimeUtc(modelPath) };
        string? modelHash = operation == "forward" ? HashFile(modelPath) : null;
        report["reference_inputs"] = new[]
        {
            (Path: firstImagePath, FrameIndex: 0, Strength: imageStrength),
            (Path: (string?)parameters.EndImagePath, FrameIndex: parameters.Frames - 1, Strength: 1f),
        }.Where(input => input.Path != null)
            .Select(input => new { path = input.Path, sha256 = HashFile(input.Path!), frame_index = input.FrameIndex, strength = input.Strength }).ToArray();
        report["first_image_stage_strengths"] = firstImagePath == null ? null : new
        {
            first_pass = effectiveFirstStrength,
            refinement = refinementStrength,
            image_strength_explicit = imageStrengthExplicit,
            policy_first_stage_override = resolvedFirstStageOverride,
            diagnostic_first_stage_override = firstStageImageStrength,
        };
        capturePath = Path.Combine(output, "capture.json");
        var timer = Stopwatch.StartNew();
        void GenerateCapture()
        {
            using ModelBase loaded = ModelBase.Create(modelPath, backend);
            if (loaded is not Ltx25Model model) throw new InvalidOperationException("The registered model is not LTX 2.5.");
            report["model_factory_ms"] = timer.Elapsed.TotalMilliseconds;
            report["environment"] = EnvironmentSnapshot();
            if (operation == "forward")
            {
                var allocator = (IAllocator)typeof(Ltx25Model).GetProperty("DirectAllocator", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
                var denoiserField = typeof(Ltx25Model).GetField("_denoiser", BindingFlags.Instance | BindingFlags.NonPublic)!;
                if (denoiserField.GetValue(model) != null) throw new InvalidOperationException("Expected a fresh model before installing the diagnostic denoiser.");
                var denoiser = new Ltx25Denoiser(modelPath, allocator);
                try
                {
                    int forwards = 0;
                    Action<Ltx25DenoiserInput, Ltx25DenoiserOutput> observeForward = (input, velocity) =>
                    {
                        if (++forwards != 1 || velocity.ExecutedBlocks != 48 || velocity.ExecutedBlocks != denoiser.Config.Layers)
                            throw new InvalidOperationException("Expected exactly one complete 48-block Forward.");
                        if (input.UseReferenceLora) throw new NotSupportedException("Captured official replay does not support Ingredients LoRA.");
                        var config = denoiser.Config;
                        object? Buffer(string name, float[]? values, params int[] shape)
                        {
                            if (values == null) return null;
                            if (shape.Aggregate(1L, (size, dim) => checked(size * dim)) != values.Length)
                                throw new InvalidDataException("Forward buffer shape mismatch: " + name);
                            ValidateFloats(values);
                            string path = Path.Combine(output, name + ".f32");
                            WriteFloats(path, values);
                            return new { path, sha256 = HashFile(path), shape, dtype = "float32-le" };
                        }
                        var inputs = new Dictionary<string, object?>
                        {
                            ["video"] = Buffer("input-video", input.Video, input.VideoTokens, config.VideoChannels),
                            ["audio"] = Buffer("input-audio", input.Audio, input.AudioTokens, config.AudioChannels),
                            ["video_context"] = Buffer("input-video-context", input.VideoContext, input.VideoContextTokens, config.VideoContextDim),
                            ["audio_context"] = Buffer("input-audio-context", input.AudioContext, input.AudioContextTokens, config.AudioContextDim),
                            ["video_positions"] = Buffer("input-video-positions", input.VideoPositions.Boundaries, input.VideoTokens, 3, 2),
                            ["audio_positions"] = Buffer("input-audio-positions", input.AudioPositions.Boundaries, input.AudioTokens, 1, 2),
                            ["video_timesteps"] = Buffer("input-video-timesteps", input.VideoTimesteps, input.VideoTimesteps?.Length ?? 0),
                            ["audio_timesteps"] = Buffer("input-audio-timesteps", input.AudioTimesteps, input.AudioTimesteps?.Length ?? 0),
                            ["keyframe_mask"] = Buffer("input-keyframe-mask", input.KeyframeMask, input.VideoTokens),
                            ["video_denoise_mask"] = Buffer("input-video-denoise-mask", input.VideoDenoiseMask, input.VideoTokens),
                            ["clean_video"] = Buffer("input-clean-video", input.CleanVideo, input.VideoTokens, config.VideoChannels),
                        };
                        _ = Buffer("output-video", velocity.VideoVelocity, input.VideoTokens, config.VideoChannels);
                        _ = Buffer("output-audio", velocity.AudioVelocity, input.AudioTokens, config.AudioChannels);
                        string videoPath = Path.Combine(output, "output-video.f32"), audioPath = Path.Combine(output, "output-audio.f32");
                        var forward = new
                        {
                            completed = true, created_utc = DateTimeOffset.UtcNow, backend = backend.ToString().ToUpperInvariant(),
                            input_mode = "captured", schedule = "forward", output_kind = "velocity",
                            model = modelPath, model_sha256 = modelHash, config, executed_blocks = velocity.ExecutedBlocks, full_depth = true,
                            frequency_precision = input.FrequencyPrecision.ToString(), video_sigma = input.VideoSigma, audio_sigma = input.AudioSigma,
                            use_reference_lora = input.UseReferenceLora,
                            tokens = new { video = input.VideoTokens, audio = input.AudioTokens, video_context = input.VideoContextTokens, audio_context = input.AudioContextTokens },
                            inputs, outputs = new { video = videoPath, audio = audioPath },
                            output_sha256 = new { video = HashFile(videoPath), audio = HashFile(audioPath) },
                            output_shapes = new { video = new[] { input.VideoTokens, config.VideoChannels }, audio = new[] { input.AudioTokens, config.AudioChannels } },
                            output_dtype = "float32-le", requested_geometry = new[] { requestedSettings.Width, requestedSettings.Height, requestedSettings.Frames },
                            stage_geometry = requestedSettings.TwoStage ? new[] { requestedSettings.Width / 2, requestedSettings.Height / 2, requestedSettings.Frames }
                                : new[] { requestedSettings.Width, requestedSettings.Height, requestedSettings.Frames },
                            seed = requestedSettings.Seed.ToString(CultureInfo.InvariantCulture), prompt,
                            first_image_stage_strengths = report["first_image_stage_strengths"], reference_inputs = report["reference_inputs"],
                            environment = report["environment"], managed_binaries = report["managed_binaries"],
                            scope = report["scope"], sampler_steps_completed = 0, forward_count = forwards,
                        };
                        string manifest = Path.Combine(output, "forward.json");
                        File.WriteAllText(manifest, JsonSerializer.Serialize(forward, jsonOptions) + "\n");
                        report["forward_capture"] = manifest;
                        report["capture_completed"] = true;
                        SaveReport();
                        throw new CaptureCompleteException();
                    };
                    typeof(Ltx25Denoiser).GetProperty("ForwardObserver", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(denoiser, observeForward);
                    denoiserField.SetValue(model, denoiser); // Model owns disposal, including the sampler's failure cleanup.
                }
                catch { denoiser.Dispose(); throw; }
            }
            var pipeline = typeof(Ltx25Model).Assembly.GetType("TensorSharp.Models.Ltx25.Ltx25Pipeline", throwOnError: true)!;
            var method = pipeline.GetMethod("Generate", BindingFlags.NonPublic | BindingFlags.Static, null,
                [typeof(Ltx25Model), typeof(string), typeof(VideoGenerationParams), typeof(Action<Ltx25GenerationSettings, float[], float[]>), typeof(float?)], null)
                ?? throw new MissingMethodException("Build the diagnostic final-latent overload before running this probe.");
            Action<Ltx25GenerationSettings, float[], float[]> observer = (settings, video, audio) =>
            {
                ValidateFloats(video); ValidateFloats(audio);
                string videoPath = Path.Combine(output, "video-latents.f32"), audioPath = Path.Combine(output, "audio-latents.f32");
                WriteFloats(videoPath, video); WriteFloats(audioPath, audio);
                SaveCapture(new Capture
                {
                    ModelPath = modelPath, Prompt = prompt, Sampler = parameters.Sampler, Seed = settings.Seed, Fps = settings.Fps,
                    ImageStrength = firstImagePath != null ? imageStrength : null,
                    ImageStrengthExplicit = imageStrengthExplicit,
                    FirstStageImageStrength = effectiveFirstStrength,
                    RefinementImageStrength = refinementStrength,
                    Video = new BufferFile("video-latents.f32", HashFile(videoPath), [128, settings.LatentFrames, settings.LatentHeight, settings.LatentWidth]),
                    Audio = new BufferFile("audio-latents.f32", HashFile(audioPath), [settings.AudioTokens, 128]),
                    Width = settings.Width, Height = settings.Height, Frames = settings.Frames,
                    Settings = settings,
                });
                report["capture_completed"] = true;
                SaveReport();
                // Stop only this diagnostic invocation before decode. The normal
                // pipeline finally and model Dispose still release every resource.
                throw new CaptureCompleteException();
            };
            try { method.Invoke(null, [model, prompt, parameters, observer, firstStageImageStrength]); }
            catch (TargetInvocationException error) when (error.InnerException is CaptureCompleteException) { return; }
            catch (TargetInvocationException error) when (error.InnerException != null)
            { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); }
            throw new InvalidOperationException("The pipeline did not emit a final-latent capture.");
        }
        if (backend == BackendType.Mlx) MlxWorker.Shared.Invoke(GenerateCapture); else GenerateCapture();
        report["capture_wall_ms"] = timer.Elapsed.TotalMilliseconds;
        if (operation == "forward")
        {
            if (!report.ContainsKey("forward_capture")) throw new InvalidOperationException("No first-forward capture was produced.");
            report["completed"] = true;
            Console.WriteLine(JsonSerializer.Serialize(new { completed = true, operation, output, forward = report["forward_capture"] }));
            return;
        }
    }
    else capturePath = Path.GetFullPath(Required("capture"));

    Capture capture = JsonSerializer.Deserialize<Capture>(File.ReadAllText(capturePath))
        ?? throw new InvalidDataException("Empty capture manifest.");
    ValidateCapture(capture);
    string captureDirectory = Path.GetDirectoryName(capturePath)!;
    float[] videoLatents = ReadBuffer(captureDirectory, capture.Video);
    // Validate audio too, even though this visual A/B does not decode it.
    _ = ReadBuffer(captureDirectory, capture.Audio);
    report["capture"] = new { path = capturePath, manifest_sha256 = HashFile(capturePath), capture.Video, capture.Audio, capture.Seed, capture.Width, capture.Height, capture.Frames, capture.Fps };
    SaveReport();
    if (decoders.Length != 0)
    {
        // Apply exactly the model's precision policy before initializing a direct
        // allocator in decode-only runs; no denoiser checkpoint is needed here.
        if (!ModelArchitectureRegistry.TryGet(Ltx25Model.ArchitectureId, out var architecture)) throw new InvalidOperationException("LTX architecture is unavailable.");
        architecture.ApplyNativeTunables?.Invoke(new ModelCreateContext(capture.ModelPath, backend, null!));
        report["environment"] = EnvironmentSnapshot();
    }
    foreach (string decoder in decoders)
    {
        string checkpoint = DecoderCheckpoint(capture.ModelPath, decoder);
        if (!File.Exists(checkpoint)) throw new FileNotFoundException("Requested decoder checkpoint is missing.", checkpoint);
        string checkpointHash = HashFile(checkpoint);
        void Decode()
        {
            IAllocator allocator = backend switch
            {
                BackendType.Cpu => new CpuAllocator(BlasEnum.DotNet), BackendType.Mlx => new MlxAllocator(),
                BackendType.Cuda => new CudaAllocator(), _ => throw new NotSupportedException(),
            };
            try
            {
                var timer = Stopwatch.StartNew();
                float[] pixels;
                if (decoder == "convolutional")
                {
                    using var vae = new Ltx25VideoVaeDecoder(checkpoint, allocator);
                    pixels = vae.Decode(videoLatents, capture.Video.Shape[1], capture.Video.Shape[2], capture.Video.Shape[3]);
                }
                else
                {
                    using var vae = new Ltx25DiffusionVideoDecoder(checkpoint, allocator);
                    pixels = vae.Decode(videoLatents, capture.Video.Shape[1], capture.Video.Shape[2], capture.Video.Shape[3], capture.Seed);
                }
                double decodeMs = timer.Elapsed.TotalMilliseconds;
                int frameSize = checked(capture.Width * capture.Height * 3);
                if (pixels.Length != checked(frameSize * capture.Frames)) throw new InvalidDataException("Unexpected decoded geometry.");
                ValidateFloats(pixels, rgb: true);
                string rgbHash = HashFloats(pixels);
                if (saveRgb) WriteFloats(Path.Combine(output, decoder + ".rgb.f32"), pixels);
                var frames = new RgbImage[capture.Frames];
                for (int i = 0; i < frames.Length; ++i)
                    frames[i] = new RgbImage(capture.Width, capture.Height, pixels.AsSpan(i * frameSize, frameSize).ToArray());
                double mean = frames[0].Pixels.Average(value => (double)value);
                double variance = frames[0].Pixels.Average(value => (value - mean) * (value - mean));
                string media = Path.Combine(output, decoder + ".mp4");
                timer.Restart();
                string codec = VideoIO.SaveMp4(media, frames, capture.Fps);
                double codecMs = timer.Elapsed.TotalMilliseconds;
                if (HashFloats(videoLatents) != capture.Video.Sha256) throw new InvalidOperationException("A decoder mutated the shared input latents.");
                executions.Add(new { decoder, checkpoint, checkpoint_sha256 = checkpointHash, decode_ms = decodeMs, codec_ms = codecMs, codec,
                    video_latents_sha256 = capture.Video.Sha256, rgb_sha256 = rgbHash, media_path = media, media_sha256 = HashFile(media),
                    first_frame_variance = variance, audio_muxed = false,
                    mlx_memory = backend == BackendType.Mlx ? (object)MlxBackend.GetMemorySnapshot() : null });
            }
            finally { (allocator as IDisposable)?.Dispose(); }
            if (backend == BackendType.Mlx) MlxBackend.ClearCache();
        }
        if (backend == BackendType.Mlx) MlxWorker.Shared.Invoke(Decode); else Decode();
        SaveReport();
        // Do not retain a full RGB clip while the second decoder runs.
        GC.Collect(); GC.WaitForPendingFinalizers();
    }
    report["completed"] = true;
    report["process_peak_working_set_bytes"] = Process.GetCurrentProcess().PeakWorkingSet64;
    Console.WriteLine(JsonSerializer.Serialize(new { completed = true, operation, output, capture = capturePath, decoders }));
}
catch (Exception error) { report["error"] = error.ToString(); throw; }
finally { SaveReport(); }

static object EnvironmentSnapshot() => new[] { "MLX_ENABLE_TF32", "TS_LTX25_MLX_BF16_GEMM", "TS_LTX25_DEVICE_PROMPT_MATH",
    "TS_VIDEO_TEXT_ENCODER", "TS_VIDEO_VAE", "TS_VIDEO_DIFFUSION_VAE", "TS_VIDEO_UPSCALER", "TS_MLX_CACHE_LIMIT_MB", "TS_MLX_WIRED_LIMIT_MB" }
    .ToDictionary(name => name, Environment.GetEnvironmentVariable);
static string HashFile(string path) { using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream)); }
static string HashFloats(float[] values)
{
    using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
    const int chunk = 262144;
    for (int offset = 0; offset < values.Length; offset += chunk)
        hash.AppendData(MemoryMarshal.AsBytes(values.AsSpan(offset, Math.Min(chunk, values.Length - offset))));
    return Convert.ToHexStringLower(hash.GetHashAndReset());
}
static void WriteFloats(string path, float[] values)
{
    using var stream = new FileStream(path, FileMode.CreateNew);
    const int chunk = 262144;
    for (int offset = 0; offset < values.Length; offset += chunk)
        stream.Write(MemoryMarshal.AsBytes(values.AsSpan(offset, Math.Min(chunk, values.Length - offset))));
}
static void ValidateFloats(float[] values, bool rgb = false)
{
    foreach (float value in values)
        if (!float.IsFinite(value) || (rgb && (value < 0 || value > 1))) throw new InvalidDataException("Non-finite or out-of-range samples.");
}
static float[] ReadBuffer(string directory, BufferFile buffer)
{
    if (Path.GetFileName(buffer.File) != buffer.File) throw new InvalidDataException("Capture buffers must be beside the manifest.");
    string path = Path.Combine(directory, buffer.File);
    long count = buffer.Shape.Aggregate(1L, (size, dimension) => checked(size * dimension));
    if (count <= 0 || count > int.MaxValue || new FileInfo(path).Length != checked(count * sizeof(float)))
        throw new InvalidDataException("Capture buffer byte count disagrees with its shape.");
    if (HashFile(path) != buffer.Sha256) throw new InvalidDataException("Capture buffer hash mismatch.");
    float[] values = new float[(int)count];
    using var stream = File.OpenRead(path);
    stream.ReadExactly(MemoryMarshal.AsBytes(values.AsSpan()));
    ValidateFloats(values);
    return values;
}
static void ValidateCapture(Capture capture)
{
    if (capture.Version != 1 || capture.Dtype != "float32-le" || capture.VideoLayout != "C,T,H,W" || capture.AudioLayout != "tokens,channels")
        throw new InvalidDataException("Unsupported capture format.");
    if (capture.Video?.Shape is not [128, > 0, > 0, > 0] || capture.Audio?.Shape is not [> 0, 128] || capture.Seed < 0 || capture.Fps is <= 0 or > 240)
        throw new InvalidDataException("Invalid latent shape, seed or frame rate.");
    if (capture.Width != checked(capture.Video.Shape[3] * 32) || capture.Height != checked(capture.Video.Shape[2] * 32) ||
        capture.Frames != checked((capture.Video.Shape[1] - 1) * 8 + 1)) throw new InvalidDataException("Capture output geometry disagrees with latent shape.");
}
sealed class CaptureCompleteException : Exception;
sealed record BufferFile(string File, string Sha256, int[] Shape);
sealed record Capture
{
    public int Version { get; init; } = 1;
    public string Dtype { get; init; } = "float32-le";
    public string VideoLayout { get; init; } = "C,T,H,W";
    public string AudioLayout { get; init; } = "tokens,channels";
    public required string ModelPath { get; init; }
    public required string Prompt { get; init; }
    public required string Sampler { get; init; }
    public float? ImageStrength { get; init; }
    public bool ImageStrengthExplicit { get; init; }
    public float? FirstStageImageStrength { get; init; }
    public float? RefinementImageStrength { get; init; }
    [JsonNumberHandling(JsonNumberHandling.AllowReadingFromString | JsonNumberHandling.WriteAsString)]
    public long Seed { get; init; }
    public int Fps { get; init; }
    public int Width { get; init; }
    public int Height { get; init; }
    public int Frames { get; init; }
    public required BufferFile Video { get; init; }
    public required BufferFile Audio { get; init; }
    public Ltx25GenerationSettings? Settings { get; init; }
}
