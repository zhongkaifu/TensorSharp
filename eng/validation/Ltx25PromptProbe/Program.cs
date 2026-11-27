using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Ltx25;
using TensorSharp.Runtime;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i += 2)
    if (i + 1 >= args.Length || !args[i].StartsWith("--") || !options.TryAdd(args[i][2..], args[i + 1]))
        throw new ArgumentException("Expected unique --option value pairs.");
string Required(string name) => options.GetValueOrDefault(name) ?? throw new ArgumentException("Missing --" + name);
string checkpoint = Path.GetFullPath(Required("encoder")), output = Path.GetFullPath(Required("output"));
string stage = options.GetValueOrDefault("stage", "full"), prompt = options.GetValueOrDefault("prompt", "A red ball rolls across a wooden table.");
if (stage is not ("tokenizer" or "hidden" or "features" or "full")) throw new ArgumentException("Invalid stage.");
int rounds = int.Parse(options.GetValueOrDefault("rounds", "1"), System.Globalization.CultureInfo.InvariantCulture);
if (rounds < 1 || rounds > 1 && stage != "full") throw new ArgumentException("--rounds takes a positive count, above 1 only at the full stage.");
bool prefault = options.GetValueOrDefault("prefault", "0") == "1";
if (File.Exists(output)) throw new IOException("Refusing to replace report.");
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
var report = new Dictionary<string, object?> { ["completed"] = false, ["stage"] = stage,
    ["encoder"] = checkpoint, ["encoder_bytes"] = new FileInfo(checkpoint).Length, ["prompt"] = prompt,
    ["created_utc"] = DateTimeOffset.UtcNow, ["backend"] = options.GetValueOrDefault("backend", "CPU"),
    ["MLX_ENABLE_TF32"] = Environment.GetEnvironmentVariable("MLX_ENABLE_TF32"),
    ["TS_MLX_Q5K_REPACK"] = Environment.GetEnvironmentVariable("TS_MLX_Q5K_REPACK"),
    ["TS_VIDEO_READAHEAD"] = Environment.GetEnvironmentVariable("TS_VIDEO_READAHEAD"),
    ["TS_VIDEO_READAHEAD_WORKERS"] = Environment.GetEnvironmentVariable("TS_VIDEO_READAHEAD_WORKERS"),
    ["scope"] = "Native prompt conditioning only; excludes denoising, VAEs and media output." };
void Save() => File.WriteAllText(output, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + "\n");
StoredTensor Store(string suffix, float[] values)
{
    if (values.Any(value => !float.IsFinite(value))) throw new ArithmeticException("Nonfinite prompt output.");
    byte[] bytes = new byte[checked(values.Length * 4)]; Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
    string path = output + "." + suffix + ".f32"; if (File.Exists(path)) throw new IOException("Refusing to replace array.");
    File.WriteAllBytes(path, bytes); return new(path, values.Length, Convert.ToHexStringLower(SHA256.HashData(bytes)));
}
try
{
    report["managed"] = new[] { Assembly.GetExecutingAssembly(), typeof(Ltx25PromptEncoder).Assembly,
        typeof(CudaAllocator).Assembly, typeof(MlxAllocator).Assembly }.Select(a => new { name = a.GetName().Name,
            sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(a.Location))) }).ToArray();
    var resolver = typeof(CudaAllocator).Assembly.GetType("TensorSharp.Cuda.CudaKernels")!
        .GetMethod("LocatePtxPath", BindingFlags.Static | BindingFlags.NonPublic, null, [typeof(string)], null)!;
    report["cuda_modules"] = new[] { "tensorsharp_kernels.ptx", "tensorsharp_ltx25.ptx" }.Select(name =>
    {
        string? path = (string?)resolver.Invoke(null, [name]);
        return new { name, path, sha256 = path == null ? null : Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path))) };
    }).ToArray();
    using var file = new SafetensorsFile(checkpoint);
    var tokenizer = new Ltx25PromptTokenizer(file.ReadRawBytes("tokenizer_json"));
    int[] ids = tokenizer.Encode(prompt); report["token_ids"] = ids;
    if (stage != "tokenizer")
    {
        string model = Path.GetFullPath(Required("model")); report["denoiser"] = model;
        IAllocator allocator = options.GetValueOrDefault("backend", "CPU").ToUpperInvariant() switch
        { "CPU" => new CpuAllocator(BlasEnum.DotNet), "MLX" => new MlxAllocator(), "CUDA" => new CudaAllocator(), _ => throw new ArgumentException("Unknown backend.") };
        try
        {
            using var encoder = new Ltx25PromptEncoder(checkpoint, model, allocator);
            var timer = Stopwatch.StartNew();
            if (prefault)
            {
                // Encoder construction has already opened the mmap. Warming
                // before that open can be invalidated by a FUSE filesystem.
                // Reserve16GiB for the process, activations and other mappings.
                long available = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
                long budget = Math.Max(0, available - (16L << 30));
                double seconds = -1;
                if (budget > 0)
                    seconds = (double)typeof(Ltx25PromptEncoder).Assembly.GetType("TensorSharp.Models.MiniMaxH3.MiniMaxH3Model")!
                        .GetMethod("PrefaultWeightFile", BindingFlags.Static | BindingFlags.NonPublic)!.Invoke(null, [checkpoint, budget])!;
                report["prefault"] = new { seconds, available_bytes = available, budget_bytes = budget,
                    included_in_elapsed = true, strategy = "owned bounded sequential8MiB reader after mmap open" };
            }
            if (stage == "full")
            {
                if (rounds == 1)
                {
                    var result = encoder.Encode(prompt); timer.Stop(); report["tokens"] = result.Tokens;
                    report["video_context"] = Store("video_context", result.VideoContext);
                    report["audio_context"] = Store("audio_context", result.AudioContext);
                    report["phases_ms"] = new { gemma = encoder.LastGemmaMilliseconds,
                        aggregation = encoder.LastAggregationMilliseconds, connectors = encoder.LastConnectorMilliseconds };
                    report["readahead"] = encoder.LastReadAhead;
                }
                else
                {
                    var completed = new List<object>();
                    (string Video, string Audio)? firstHashes = null;
                    double prefaultMs = timer.Elapsed.TotalMilliseconds;
                    for (int round = 0; round < rounds; ++round)
                    {
                        long count = encoder.CudaBf16Projections;
                        long read = encoder.CudaPrefaultReadBytes, resident = encoder.CudaPrefaultResidentBytes,
                            ranges = encoder.CudaPrefaultRanges;
                        double mappedPrefaultElapsed = encoder.CudaPrefaultMilliseconds;
                        var roundTimer = Stopwatch.StartNew();
                        var result = encoder.Encode(prompt); roundTimer.Stop();
                        StoredTensor video = Store("round_" + round + "_video_context", result.VideoContext);
                        StoredTensor audio = Store("round_" + round + "_audio_context", result.AudioContext);
                        firstHashes ??= (video.sha256, audio.sha256);
                        bool exact = firstHashes.Value == (video.sha256, audio.sha256);
                        completed.Add(new { round, tokens = result.Tokens,
                            elapsed_ms = roundTimer.Elapsed.TotalMilliseconds + (round == 0 ? prefaultMs : 0),
                            prefault_ms = round == 0 ? prefaultMs : 0,
                            phases_ms = new { gemma = encoder.LastGemmaMilliseconds,
                                aggregation = encoder.LastAggregationMilliseconds, connectors = encoder.LastConnectorMilliseconds },
                            readahead = encoder.LastReadAhead,
                            cuda_bf16_projections = encoder.CudaBf16Projections - count,
                            mapped_prefault = new { read_bytes = encoder.CudaPrefaultReadBytes - read,
                                resident_bytes = encoder.CudaPrefaultResidentBytes - resident,
                                ranges = encoder.CudaPrefaultRanges - ranges,
                                elapsed_ms = encoder.CudaPrefaultMilliseconds - mappedPrefaultElapsed,
                                failure = encoder.CudaPrefaultFailure },
                            video_context = video, audio_context = audio,
                            bitwise_equal_to_first_round = exact });
                        report["rounds"] = completed;
                        report["elapsed_ms"] = timer.Elapsed.TotalMilliseconds;
                        Save();
                        Console.WriteLine($"Completed round {round}: {roundTimer.Elapsed.TotalMilliseconds:F3}ms, bitwise_equal_to_first={exact}");
                        if (!exact) throw new InvalidOperationException("A repeated encode changed the exact video/audio conditioning bytes.");
                    }
                    timer.Stop();
                    report["rounds_scope"] = "Repeated Encode on one encoder; per-round timings exclude saved arrays. First includes optional prefault; total elapsed also includes array exports. Every round must be bitwise identical.";
                }
            }
            else
            {
                // Reflection reaches only diagnostic internal checkpoints; the
                // production path above uses the complete public native encoder.
                var hidden = (float[][])typeof(Ltx25PromptEncoder).GetMethod("EncodeHiddenStates", BindingFlags.Instance | BindingFlags.NonPublic)!
                    .Invoke(encoder, [ids, CancellationToken.None])!;
                report["hidden_states"] = hidden.Select((value, index) => Store("hidden_" + index, value)).ToArray();
                if (stage == "features")
                {
                    float[] data = (float[])typeof(Ltx25PromptEncoder).GetMethod("Features", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(encoder, [hidden, ids.Length])!;
                    using var features = new Tensor(allocator, DType.Float32, ids.Length, data.Length / ids.Length); features.SetElementsAsFloat(data);
                    using var gguf = new GgufFile(model);
                    foreach (string stream in new[] { "video", "audio" })
                    {
                        int dimension = checked((int)gguf.Tensors[stream + "_embeddings_connector.learnable_registers"].Shape[0]);
                        var value = (float[])typeof(Ltx25PromptEncoder).GetMethod("Aggregate", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(encoder, [features, dimension, stream, CancellationToken.None])!;
                        report[stream + "_aggregate"] = Store(stream + "_aggregate", value);
                    }
                }
                timer.Stop();
            }
            report["elapsed_ms"] = timer.Elapsed.TotalMilliseconds;
            report["executed_projections"] = new { cuda_dense = encoder.CudaMappedProjections, cuda_connector_f16 = encoder.CudaConnectorProjections,
                cuda_bf16 = encoder.CudaBf16Projections,
                cuda_connector_attention = encoder.CudaConnectorAttentions };
            report["mapped_prefault"] = new { read_bytes = encoder.CudaPrefaultReadBytes,
                resident_bytes = encoder.CudaPrefaultResidentBytes, ranges = encoder.CudaPrefaultRanges,
                elapsed_ms = encoder.CudaPrefaultMilliseconds, failure = encoder.CudaPrefaultFailure,
                workers = typeof(Ltx25PromptEncoder).GetField("_cudaPrefault", BindingFlags.Instance | BindingFlags.NonPublic)!
                    .GetValue(encoder)?.GetType().GetProperty("Workers", BindingFlags.Instance | BindingFlags.NonPublic)!
                    .GetValue(typeof(Ltx25PromptEncoder).GetField("_cudaPrefault", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(encoder)) };
            if (allocator is MlxAllocator mlx) report["mlx_memory"] = mlx.GetMemorySnapshot();
            encoder.ReleaseDeviceResidency();
        }
        finally { (allocator as IDisposable)?.Dispose(); }
    }
    report["completed"] = true; Save(); Console.WriteLine(JsonSerializer.Serialize(report));
}
catch (Exception error) { report["error"] = error.ToString(); Save(); throw; }

sealed record StoredTensor(string path, int elements, string sha256);
