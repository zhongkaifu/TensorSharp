// MiniMax-H3-Image's denoise loop, step by step, with every step's velocity written out: the bit-exactness
// check for changes under the DiT (ggml_ops_minimax_h3.cpp, ggml_ops_dit_offload.cpp, the diffusion runtime).
//
//   dotnet MiniMaxH3ImageStepProbe.dll --dit <denoiser.gguf|.safetensors> --te <qwen3vl...gguf> --backend ggml_metal \
//       --prompt "..." --width 2048 --height 2048 --seed 7 [--steps 6] --out <dir> [--text <dir of an earlier run>]
//
// The loop is MiniMaxH3ImageModel.Generate's (the same layout, schedule, PDD heads, noise and Euler update), driven
// through the legacy denoiser (MiniMaxH3DiT.ForwardImage). The text encoder runs once and its layer-50 hidden state is
// written to <out>/text-hidden.f32; --text reuses another run's, so two builds' denoisers see identical inputs. Writes
// <out>/velocity-<i>.f32 per step, <out>/latent.f32 (after the last step) and <out>/result.json with their SHA-256s.
//
// The members the loop needs are TensorSharp.Models internals: they are reached by reflection, so one build of this
// probe runs against any TensorSharp.Models.dll placed beside it (a base build's and a changed build's).
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Runtime;
using TensorSharp.Models.MiniMaxH3;

string? Value(string name) => args.SkipWhile(a => a != name).Skip(1).FirstOrDefault();
string Required(string name) => Value(name) ?? throw new ArgumentException($"{name} is required");
string ditPath = Path.GetFullPath(Required("--dit"));
string tePath = Path.GetFullPath(Required("--te"));
string backendName = Required("--backend");
string prompt = Required("--prompt");
int width = int.Parse(Value("--width") ?? "2048", CultureInfo.InvariantCulture);
int height = int.Parse(Value("--height") ?? "2048", CultureInfo.InvariantCulture);
long seed = long.Parse(Value("--seed") ?? "7", CultureInfo.InvariantCulture);
int steps = int.Parse(Value("--steps") ?? "6", CultureInfo.InvariantCulture);
string outDir = Path.GetFullPath(Required("--out"));
string? textDir = Value("--text");
BackendType backend = backendName switch
{
    "ggml_metal" => BackendType.GgmlMetal,
    "ggml_cuda" => BackendType.GgmlCuda,
    "ggml_cpu" => BackendType.GgmlCpu,
    _ => throw new ArgumentException("--backend is ggml_metal, ggml_cuda or ggml_cpu"),
};
if (width % 32 != 0 || height % 32 != 0) throw new ArgumentException("sides are multiples of 32");
// The native graphs run on the process's one GGML backend: pin the one asked for first (an unpinned process takes the
// platform's default, ggml-metal on a Mac).
TensorSharp.GGML.GgmlBasicOps.EnsureBackendAvailable(backend switch
{
    BackendType.GgmlMetal => TensorSharp.GGML.GgmlBackendType.Metal,
    BackendType.GgmlCuda => TensorSharp.GGML.GgmlBackendType.Cuda,
    _ => TensorSharp.GGML.GgmlBackendType.Cpu,
});
if (File.Exists(Path.Combine(outDir, "result.json"))) throw new InvalidOperationException($"{outDir} already holds a run");
Directory.CreateDirectory(outDir);

Assembly models = typeof(MiniMaxH3DiT).Assembly;
const BindingFlags Any = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance;
Type Internal(string name) => models.GetType("TensorSharp.Models.MiniMaxH3." + name, throwOnError: true)!;
object? Invoke(Type type, object? target, string name, params object?[] values)
{
    MethodInfo method = type.GetMethods(Any).Single(m => m.Name == name && m.IsStatic == (target == null) && m.GetParameters().Length == values.Length);
    return method.Invoke(target, values);
}

string Sha(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
byte[] Bytes(float[] values)
{
    var bytes = new byte[values.Length * sizeof(float)];
    Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
    return bytes;
}
float[] Floats(byte[] bytes)
{
    var values = new float[bytes.Length / sizeof(float)];
    Buffer.BlockCopy(bytes, 0, values, 0, values.Length * sizeof(float));
    return values;
}

// ---- text ----
float[] textHidden;
int textLength;
double textSeconds = 0;
if (textDir != null)
{
    var meta = JsonSerializer.Deserialize<JsonElement>(File.ReadAllText(Path.Combine(textDir, "text.json")));
    if (meta.GetProperty("prompt").GetString() != prompt) throw new InvalidOperationException("--text holds another prompt's hidden state");
    textLength = meta.GetProperty("tokens").GetInt32();
    textHidden = Floats(File.ReadAllBytes(Path.Combine(textDir, "text-hidden.f32")));
}
else
{
    var clock = Stopwatch.StartNew();
    using var encoder = new MiniMaxH3TextEncoder(tePath, null, backend, null);
    List<int> ids = encoder.Tokenize(prompt);
    if (ids.Count == 0) ids.Add(encoder.Tokenizer.BosTokenId);
    textHidden = encoder.Encode(ids);
    textLength = ids.Count;
    textSeconds = clock.Elapsed.TotalSeconds;
}
File.WriteAllBytes(Path.Combine(outDir, "text-hidden.f32"), Bytes(textHidden));
File.WriteAllText(Path.Combine(outDir, "text.json"), JsonSerializer.Serialize(new { prompt, tokens = textLength, sha256 = Sha(Bytes(textHidden)) }));

// ---- noise, as MiniMaxH3ImageModel.Generate draws it ----
const int Channels = 24, Downsample = 16;
int lh = height / Downsample, lw = width / Downsample;
// The CPU generator's draw (MiniMaxH3ImageSampling's, or the diffusion runtime's once it moved there).
Type randn = models.GetType("TensorSharp.Models.MiniMaxH3.TorchCpuRandn")
    ?? models.GetType("TensorSharp.Models.SharedDiffusion.TorchCpuRandn", throwOnError: true)!;
var chw = (float[])Invoke(randn, null, "Normal", (long)Channels * lh * lw, unchecked((ulong)seed))!;
var hwc = new float[chw.Length];
int plane = lh * lw;
for (int c = 0; c < Channels; c++)
    for (int i = 0; i < plane; i++)
        hwc[(long)i * Channels + c] = chw[(long)c * plane + i];
float[] x = MiniMaxH3DiT.PatchifyVideo(hwc, 1, lh, lw, Channels);
int imageTokens = (lh / 2) * (lw / 2);

// ---- denoise ----
Type sampling = Internal("MiniMaxH3ImageSampling");
Type layoutType = Internal("MiniMaxH3ImageLayout");
var stepRecords = new List<object>();
var total = Stopwatch.StartNew();
using (var dit = new MiniMaxH3DiT(ditPath, backend, null))
{
    float[] grid;
    {
        Type weightFile = Internal("MiniMaxH3WeightFile");
        using var probe = (IDisposable)Activator.CreateInstance(weightFile, Any, null, new object[] { ditPath }, null)!;
        var metadata = (IReadOnlyDictionary<string, string>?)weightFile.GetProperty("SafetensorsMetadata", Any)!.GetValue(probe);
        grid = (float[])Invoke(sampling, null, "GridFrom", metadata, dit.Config.PddSteps)!;
    }
    var sigmas = (float[])Invoke(sampling, null, "Schedule", grid, steps)!;
    for (int i = 0; i < steps; i++)
    {
        float sigma = sigmas[i], next = sigmas[i + 1];
        object layout = Invoke(layoutType, null, "Build", textLength, lh, lw, sigma, null, null)!;
        var interval = Invoke(sampling, null, "Interval", grid, sigma, next)!;
        int start = (int)interval.GetType().GetField("Item1")!.GetValue(interval)!;
        int stop = (int)interval.GetType().GetField("Item2")!.GetValue(interval)!;
        var clock = Stopwatch.StartNew();
        var v = (float[])Invoke(typeof(MiniMaxH3DiT), dit, "ForwardImage", x, imageTokens, textHidden, textLength, layout, grid, start,
            stop, null, 0)!;
        double seconds = clock.Elapsed.TotalSeconds;
        if (v.Any(f => !float.IsFinite(f))) throw new InvalidDataException($"step {i + 1}: a non-finite velocity");
        byte[] vb = Bytes(v);
        File.WriteAllBytes(Path.Combine(outDir, $"velocity-{i + 1}.f32"), vb);
        float dt = next - sigma;
        for (long k = 0; k < x.Length; k++) x[k] += v[k] * dt;
        stepRecords.Add(new { step = i + 1, sigma, next, heads = new[] { start, stop }, seconds, sha256 = Sha(vb) });
        Console.WriteLine($"step {i + 1}/{steps} sigma {sigma:0.#####} -> {next:0.#####} heads {start}..{stop} {seconds:F2}s {Sha(vb)}");
    }
}
byte[] latent = Bytes(x);
File.WriteAllBytes(Path.Combine(outDir, "latent.f32"), latent);
string native = Path.Combine(AppContext.BaseDirectory, OperatingSystem.IsMacOS() ? "libGgmlOps.dylib" : OperatingSystem.IsWindows() ? "GgmlOps.dll" : "libGgmlOps.so");
var result = new
{
    backend = backendName, dit = ditPath, te = tePath, prompt, width, height, seed, steps, textTokens = textLength, imageTokens,
    textSeconds, textReused = textDir != null, denoiseSeconds = total.Elapsed.TotalSeconds,
    nativeLibrary = native, nativeSha256 = File.Exists(native) ? Sha(File.ReadAllBytes(native)) : null,
    models = models.Location, modelsSha256 = Sha(File.ReadAllBytes(models.Location)),
    stepsRecorded = stepRecords, latentSha256 = Sha(latent),
};
File.WriteAllText(Path.Combine(outDir, "result.json"), JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine($"latent {Sha(latent)}");
// Release the ggml backends before exit, as the CLI does (ggml-metal asserts on residency sets still held at teardown).
TensorSharp.GGML.GgmlBasicOps.Shutdown();
return 0;
