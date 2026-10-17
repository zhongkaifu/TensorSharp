using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Ltx25;

var options = new Dictionary<string, string>();
for (int i = 0; i < args.Length; i += 2)
    if (i + 1 == args.Length || !args[i].StartsWith("--") || !options.TryAdd(args[i][2..], args[i + 1]))
        throw new ArgumentException("Expected unique --option value pairs.");
string Required(string name) => options.GetValueOrDefault(name) ?? throw new ArgumentException("Missing --" + name);
int Number(string name, int fallback) => int.Parse(options.GetValueOrDefault(name, fallback.ToString()), System.Globalization.CultureInfo.InvariantCulture);
string modelPath = Path.GetFullPath(Required("model")), outputPath = Path.GetFullPath(Required("output"));
string backend = options.GetValueOrDefault("backend", "cpu").ToLowerInvariant();
int frames = Number("latent-frames", 2), height = Number("latent-height", 2), width = Number("latent-width", 2);
int warmup = Number("warmup", 0), repeats = Number("repeats", 1);
bool traceEnabled = Number("trace", 1) == 1;
if (frames <= 0 || height <= 0 || width <= 0 || warmup < 0 || repeats <= 0)
    throw new ArgumentException("Invalid geometry or repetition count.");
if (Directory.Exists(outputPath) && Directory.EnumerateFileSystemEntries(outputPath).Any())
    throw new IOException("Refusing to replace an existing probe output directory.");
Directory.CreateDirectory(outputPath);
string Hash(string path)
{
    using var stream = File.OpenRead(path);
    return Convert.ToHexStringLower(SHA256.HashData(stream));
}
void SaveFloats(string path, float[] data)
{
    // A 1024x1024x145 output is 1.8 GB. Write its existing storage in bounded
    // spans instead of allocating a second full-size byte array for evidence.
    using var file = File.Create(path);
    const int chunkFloats = 262144;
    for (int start = 0; start < data.Length; start += chunkFloats)
        file.Write(MemoryMarshal.AsBytes(data.AsSpan(start, Math.Min(chunkFloats, data.Length - start))));
}
var report = new Dictionary<string, object?>
{
    ["completed"] = false, ["created_utc"] = DateTimeOffset.UtcNow,
    ["scope"] = "Native TensorSharp convolutional video decoder only. BF16-stored checkpoint weights widened to Float32; no text encoder, sampling, audio, codecs, perceptual-quality or end-to-end performance coverage. Traced runs add device readbacks and are diagnostic only.",
    ["backend"] = backend, ["model"] = modelPath, ["model_sha256"] = Hash(modelPath),
    ["input_latent_geometry"] = new[] {frames, height, width}, ["warmup"] = warmup, ["repeats"] = repeats,
    ["environment"] = new[] { "MLX_ENABLE_TF32", "TENSORSHARP_MLX_LIBRARY", "TENSORSHARP_MLX_LIBRARY_DIR" }
        .ToDictionary(name => name, Environment.GetEnvironmentVariable),
    ["managed_binaries"] = new[] { Assembly.GetExecutingAssembly(), typeof(Ltx25VideoVaeDecoder).Assembly,
        typeof(CudaAllocator).Assembly, typeof(MlxAllocator).Assembly, typeof(CpuAllocator).Assembly }.Distinct()
        .Select(assembly => new { path = assembly.Location, sha256 = Hash(assembly.Location) }).ToArray(),
};
void Save() => File.WriteAllText(Path.Combine(outputPath,"probe.json"), JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Save();
try
{
    if (backend == "cuda")
    {
        string primary = Path.Combine(AppContext.BaseDirectory, "cuda_kernels", "tensorsharp_kernels.ptx");
        if (!File.Exists(primary))
            throw new FileNotFoundException("Deploy the verified CUDA primary PTX under the probe's cuda_kernels directory before numerical validation.", primary);
        report["cuda_primary_ptx"] = new { path = primary, sha256 = Hash(primary) };
    }
    if (backend == "mlx")
    {
        string directory = Environment.GetEnvironmentVariable("TENSORSHARP_MLX_LIBRARY_DIR") ?? AppContext.BaseDirectory;
        report["configured_mlx_native_files"] = Directory.GetFiles(directory, "*", SearchOption.AllDirectories)
            .Where(path => Path.GetExtension(path) is ".dylib" or ".metallib")
            .Select(path => new { path, sha256 = Hash(path) }).ToArray();
    }
    IAllocator allocator = backend switch
    {
        "cpu" => new CpuAllocator(BlasEnum.DotNet), "cuda" => new CudaAllocator(), "mlx" => new MlxAllocator(),
        _ => throw new ArgumentException("Unknown backend."),
    };
    try
    {
        if (allocator is MlxAllocator mlxBefore) report["mlx_memory_before_decoder"] = mlxBefore.GetMemorySnapshot();
        using var decoder = new Ltx25VideoVaeDecoder(modelPath, allocator);
        int count = checked(decoder.LatentChannels * frames * height * width);
        float[] input = new float[count];
        if (options.TryGetValue("input", out var inputPath))
        {
            var bytes = File.ReadAllBytes(inputPath);
            if (bytes.Length != checked(count * 4)) throw new ArgumentException("Wrong input latent byte count.");
            Buffer.BlockCopy(bytes,0,input,0,bytes.Length);
        }
        else
        {
            uint seed=1;
            for (int i=0;i<input.Length;i++)
            {
                seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
                input[i] = ((seed >> 8) * (1f / 16777216f) - .5f) * 1.2f;
            }
        }
        var inputFile = Path.Combine(outputPath,"input.f32"); SaveFloats(inputFile,input);
        report["input_sha256"] = Hash(inputFile);
        report["input_shape"] = new[] {decoder.LatentChannels, frames,height,width};
        report["output_shape"] = new[] {3,decoder.TemporalScale*(frames-1)+1,decoder.SpatialScale*height,decoder.SpatialScale*width};
        for (int run=0;run<warmup;run++) decoder.DecodeRaw(input,frames,height,width);
        var stages = new Dictionary<string,object>();
        Action<string,float[],int[]> trace = (name, values, shape) =>
        {
            string file = name.Replace('.','_') + ".f32"; var path=Path.Combine(outputPath,file);
            SaveFloats(path,values); stages[name] = new {shape,file,sha256=Hash(path)};
        };
        var timings = new List<double>();
        var memorySnapshots = new List<MlxMemorySnapshot>();
        float[]? previous = null;
        double repeatDifference = 0;
        for(int run=0;run<repeats;run++)
        {
            var timer = Stopwatch.StartNew();
            var result = decoder.DecodeRaw(input,frames,height,width,trace:traceEnabled && run==0 ? trace : null);
            timer.Stop(); timings.Add(timer.Elapsed.TotalMilliseconds);
            if (allocator is MlxAllocator mlx) memorySnapshots.Add(mlx.GetMemorySnapshot());
            if(result.Any(value => !float.IsFinite(value))) throw new InvalidOperationException("Non-finite decoder output.");
            if(previous != null) for(int i=0;i<result.Length;i++) repeatDifference=Math.Max(repeatDifference,Math.Abs(result[i]-previous[i]));
            previous=result;
        }
        var outputFile=Path.Combine(outputPath,"output.f32"); SaveFloats(outputFile,previous!);
        report["stages"] = stages; report["output_sha256"] = Hash(outputFile);
        report["times_ms"] = timings; report["timing_scope"] = traceEnabled ? "diagnostic including first-run stage readbacks" : "decode plus synchronized final raw output readback";
        if (memorySnapshots.Count != 0)
        {
            report["mlx_memory_after_runs"] = memorySnapshots;
            report["mlx_memory_observation_available"] = memorySnapshots.All(snapshot => snapshot.PeakBytes != 0);
        }
        report["repeat_max_absolute_difference"] = repeatDifference;
        report["completed"] = true; Save();
        Console.WriteLine(JsonSerializer.Serialize(new {completed=true,backend,outputPath,timings,repeatDifference}));
    }
    finally { (allocator as IDisposable)?.Dispose(); }
}
catch(Exception error)
{
    report["error"] = error.ToString(); Save(); throw;
}
