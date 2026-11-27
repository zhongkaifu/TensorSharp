using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Ltx25;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i += 2)
    if (i + 1 >= args.Length || !args[i].StartsWith("--") || !options.TryAdd(args[i][2..], args[i + 1]))
        throw new ArgumentException("Expected unique --option value pairs.");
string Required(string name) => options.GetValueOrDefault(name) ?? throw new ArgumentException("Missing --" + name);
int Number(string name, int fallback) => int.Parse(options.GetValueOrDefault(name, fallback.ToString()), CultureInfo.InvariantCulture);
string model = Path.GetFullPath(Required("model")), output = Path.GetFullPath(Required("output"));
string backend = options.GetValueOrDefault("backend", "CPU").ToUpperInvariant();
int frames = Number("frames", 9), height = Number("height", 64), width = Number("width", 64);
int audioTokens = Number("audio-tokens", 16), contextTokens = Number("context-tokens", 8);
int warmup = Number("warmup", 1), repeats = Number("repeats", 3);
string schedule = options.GetValueOrDefault("schedule", "forward");
if (schedule is not ("forward" or "distilled")) throw new ArgumentException("--schedule must be forward or distilled.");
float sigma = float.Parse(options.GetValueOrDefault("sigma", "0.725"), CultureInfo.InvariantCulture);
if (warmup < 0 || repeats < 1 || audioTokens < 1 || contextTokens < 1)
    throw new ArgumentException("Invalid probe counts.");
if (File.Exists(output) || File.Exists(output + ".video.f32") || File.Exists(output + ".audio.f32"))
    throw new IOException("Refusing to replace an existing report or output array.");
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
var report = new Dictionary<string, object?>
{
    ["completed"] = false, ["backend"] = backend, ["created_utc"] = DateTimeOffset.UtcNow,
    ["model"] = model, ["model_bytes"] = new FileInfo(model).Length,
    ["geometry"] = new[] { width, height, frames }, ["sigma"] = schedule == "forward" ? (float?)sigma : null,
    ["template_sigma"] = sigma,
    ["warmup"] = warmup, ["repeats"] = repeats,
    ["schedule"] = schedule,
    ["timing_kind"] = "synchronized_output_readbacks",
    ["environment"] = new[] { "MLX_ENABLE_TF32", "TS_WAN_DIRECT_FUSED_ATTN", "TS_CUDA_QMM_F16GEMM_MIN_ROWS", "TS_MLX_Q5K_REPACK" }
        .ToDictionary(name => name, Environment.GetEnvironmentVariable),
    ["scope"] = schedule == "forward"
        ? "AV denoiser Forward with deterministic synthetic projected contexts/latents, synchronized by both output readbacks. Excludes tokenizer, Gemma4, connectors, sampler, VAEs, codecs, HTTP and TensorAgent. This is not end-to-end generation or a quality proof."
        : "Eight-step unmasked AV Float32 Euler sampling with deterministic synthetic projected contexts/initial latents. Includes all denoiser forwards and output readbacks. Excludes noise RNG, tokenizer, Gemma4, connectors, conditioning, guidance, VAEs, codecs, HTTP and TensorAgent. This is not end-to-end generation or a quality proof.",
};
void Save() => File.WriteAllText(output, JsonSerializer.Serialize(report, jsonOptions) + "\n");
byte[] Bytes(float[] values)
{
    var bytes = new byte[checked(values.Length * sizeof(float))];
    Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
    return bytes;
}
string Hash(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
string HashFile(string path)
{
    using var stream = File.OpenRead(path);
    return Convert.ToHexStringLower(SHA256.HashData(stream));
}
float[] Values(int length, uint seed)
{
    var values = new float[length];
    for (int i = 0; i < length; ++i)
    {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        values[i] = ((seed >> 8) * (1f / 16777216f) - .5f) * .2f;
    }
    return values;
}
try
{
    report["model_sha256"] = HashFile(model);
    report["managed_binaries"] = new[] { Assembly.GetExecutingAssembly(), typeof(Ltx25Denoiser).Assembly,
        typeof(CudaAllocator).Assembly, typeof(MlxAllocator).Assembly, typeof(CpuAllocator).Assembly }
        .Distinct().Select(assembly => new { name = assembly.GetName().Name, path = assembly.Location,
            sha256 = HashFile(assembly.Location) }).ToArray();
    Ltx25Config config;
    using (var file = new GgufFile(model)) config = Ltx25Config.FromGguf(file);
    var latent = Ltx25Geometry.VideoLatentShape(frames, height, width);
    int videoTokens = checked(latent.Frames * latent.Height * latent.Width);
    int blocks = Number("blocks", config.Layers);
    var input = new Ltx25DenoiserInput
    {
        Video = Values(checked(videoTokens * config.VideoChannels), 1), VideoTokens = videoTokens,
        Audio = Values(checked(audioTokens * config.AudioChannels), 2), AudioTokens = audioTokens,
        VideoContext = Values(checked(contextTokens * config.VideoContextDim), 3), VideoContextTokens = contextTokens,
        AudioContext = Values(checked(contextTokens * config.AudioContextDim), 4), AudioContextTokens = contextTokens,
        VideoPositions = Ltx25Geometry.VideoPositions(latent.Frames, latent.Height, latent.Width, 24),
        AudioPositions = Ltx25Geometry.AudioPositions(audioTokens), VideoSigma = sigma, AudioSigma = sigma,
        FrequencyPrecision = Enum.Parse<Ltx25FrequencyPrecision>(options.GetValueOrDefault("frequency-precision", "Float64")),
    };
    report["config"] = config; report["executed_blocks"] = blocks;
    report["frequency_precision"] = input.FrequencyPrecision.ToString();
    report["sigmas"] = schedule == "distilled" ? Ltx25Scheduler.DistilledSigmas() : null;
    report["full_depth"] = blocks == config.Layers;
    report["tokens"] = new { video = videoTokens, audio = audioTokens, context = contextTokens };
    report["input_sha256"] = new { video = Hash(Bytes(input.Video)), audio = Hash(Bytes(input.Audio)),
        video_context = Hash(Bytes(input.VideoContext)), audio_context = Hash(Bytes(input.AudioContext)) };
    Save();
    IAllocator allocator = backend switch
    {
        "CPU" => new CpuAllocator(BlasEnum.DotNet), "CUDA" => new CudaAllocator(0), "MLX" => new MlxAllocator(),
        _ => throw new ArgumentException("Supported backends: CPU, CUDA, MLX."),
    };
    try
    {
        using var denoiser = new Ltx25Denoiser(model, allocator);
        var samples = new List<double>();
        Ltx25DenoiserOutput? result = null;
        double? firstMilliseconds = null;
        byte[]? firstVideo = null, firstAudio = null;
        double maximumRepeatDifference = 0;
        for (int i = 0; i < warmup + repeats; ++i)
        {
            Console.Error.WriteLine($"Forward {i + 1}/{warmup + repeats}, {blocks}/{config.Layers} blocks");
            var timer = Stopwatch.StartNew();
            if (schedule == "forward") result = denoiser.Forward(input, blocks);
            else
            {
                // The distilled schedule on the runtime's x0 Euler sampler, as a clip's Euler pass steps it.
                float[] sigmas = Ltx25Scheduler.DistilledSigmas();
                float[] video = (float[])input.Video.Clone(), audio = (float[])input.Audio.Clone();
                var sampler = new EulerDenoisedSampler();
                int executed = 0;
                for (int step = 0; step + 1 < sigmas.Length; ++step)
                {
                    var prediction = denoiser.Forward(new Ltx25DenoiserInput
                    {
                        Video = video, VideoTokens = input.VideoTokens, Audio = audio, AudioTokens = input.AudioTokens,
                        VideoContext = input.VideoContext, VideoContextTokens = input.VideoContextTokens,
                        AudioContext = input.AudioContext, AudioContextTokens = input.AudioContextTokens,
                        VideoPositions = input.VideoPositions, AudioPositions = input.AudioPositions,
                        FrequencyPrecision = input.FrequencyPrecision, VideoSigma = sigmas[step], AudioSigma = sigmas[step],
                    }, blocks);
                    sampler.Step(video, prediction.VideoVelocity, sigmas[step], sigmas[step + 1]);
                    sampler.Step(audio, prediction.AudioVelocity, sigmas[step], sigmas[step + 1]);
                    executed = prediction.ExecutedBlocks;
                }
                result = new(video, audio, executed);
            }
            timer.Stop();
            firstMilliseconds ??= timer.Elapsed.TotalMilliseconds;
            if (i >= warmup) samples.Add(timer.Elapsed.TotalMilliseconds);
            Console.Error.WriteLine($"Forward {i + 1}: {timer.Elapsed.TotalMilliseconds:F3} ms");
            if (result.ExecutedBlocks != blocks || result.VideoVelocity.Any(x => !float.IsFinite(x)) ||
                result.AudioVelocity.Any(x => !float.IsFinite(x))) throw new InvalidDataException("Invalid denoiser output.");
            if (firstVideo == null)
            {
                firstVideo = Bytes(result.VideoVelocity); firstAudio = Bytes(result.AudioVelocity);
            }
            else
            {
                var originalVideo = new float[result.VideoVelocity.Length];
                var originalAudio = new float[result.AudioVelocity.Length];
                Buffer.BlockCopy(firstVideo, 0, originalVideo, 0, firstVideo.Length);
                Buffer.BlockCopy(firstAudio!, 0, originalAudio, 0, firstAudio!.Length);
                for (int j = 0; j < originalVideo.Length; ++j)
                    maximumRepeatDifference = Math.Max(maximumRepeatDifference, Math.Abs((double)originalVideo[j] - result.VideoVelocity[j]));
                for (int j = 0; j < originalAudio.Length; ++j)
                    maximumRepeatDifference = Math.Max(maximumRepeatDifference, Math.Abs((double)originalAudio[j] - result.AudioVelocity[j]));
            }
        }
        string videoPath = output + ".video.f32", audioPath = output + ".audio.f32";
        File.WriteAllBytes(videoPath, Bytes(result!.VideoVelocity)); File.WriteAllBytes(audioPath, Bytes(result.AudioVelocity));
        report["output_sha256"] = new { video = Hash(Bytes(result.VideoVelocity)), audio = Hash(Bytes(result.AudioVelocity)) };
        report["outputs"] = new { video = videoPath, audio = audioPath };
        report["maximum_repeat_difference"] = maximumRepeatDifference;
        report["first_forward_ms_including_weight_upload"] = firstMilliseconds;
        var sorted = samples.Order().ToArray();
        report["forward_samples_ms"] = samples;
        report["median_ms"] = (sorted[(sorted.Length - 1) / 2] + sorted[sorted.Length / 2]) / 2;
        report["peak_working_set_bytes"] = Process.GetCurrentProcess().PeakWorkingSet64;
        if (allocator is MlxAllocator mlx) report["mlx_memory"] = mlx.GetMemorySnapshot();
        if (allocator is CudaAllocator cuda)
        {
            var memory = cuda.GetMemoryInfo();
            report["cuda_memory_after_sampling"] = new { free_bytes = memory.freeBytes,
                total_bytes = memory.totalBytes, used_bytes = memory.totalBytes - memory.freeBytes,
                scope = "Device-wide snapshot after final output readbacks; not a sampled peak or process-exclusive allocation count." };
        }
        report["completed"] = true;
        report["steps_per_sample"] = schedule == "forward" ? 1 : Ltx25Scheduler.DistilledSigmas().Length - 1;
        report["output_kind"] = schedule == "forward" ? "velocity" : "sampled latent";
    }
    finally { (allocator as IDisposable)?.Dispose(); }
    Save();
    Console.WriteLine(JsonSerializer.Serialize(report, jsonOptions));
    return 0;
}
catch (Exception error)
{
    report["error_type"] = error.GetType().FullName; report["error"] = error.Message;
    Save();
    Console.Error.WriteLine(error);
    return 1;
}
