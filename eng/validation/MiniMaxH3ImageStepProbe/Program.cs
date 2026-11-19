// MiniMax-H3-Image's denoise loop, step by step, with every step's velocity written out: the bit-exactness
// check for changes under the DiT (ggml_ops_minimax_h3.cpp, ggml_ops_dit_offload.cpp, the diffusion runtime).
//
//   dotnet MiniMaxH3ImageStepProbe.dll --dit <denoiser.gguf|.safetensors> --te <qwen3vl...gguf> --backend ggml_metal \
//       --prompt "..." --width 2048 --height 2048 --seed 7 [--steps 6] --out <dir> [--text <dir of an earlier run>]
//       [--engine legacy|runtime|both] [--rounds N]
//
// The loop is MiniMaxH3ImageModel.Generate's (the same layout, schedule, PDD heads, noise and Euler update), driven
// through the legacy denoiser (MiniMaxH3DiT.ForwardImage). The text encoder runs once and its layer-50 hidden state is
// written to <out>/text-hidden.f32; --text reuses another run's, so two builds' denoisers see identical inputs. Writes
// <out>/velocity-<i>.f32 per step, <out>/latent.f32 (after the last step) and <out>/result.json with their SHA-256s.
//
// --engine runtime denoises through the diffusion runtime's denoiser instead (MiniMaxH3ImageDenoiser, one GraphProgram a
// step; stage S3 of docs/design/diffusion-runtime.md). --engine both runs the two on the same inputs at every step (the
// runtime's over the legacy DiT's file mapping, so the weights are resident once), compares the velocities bit for bit,
// times each, and advances with the legacy one's; --rounds repeats the loop from the same noise, the two arms' order
// alternating by round and step (A B, then B A). Writes <out>/runtime-velocity-<i>.f32 beside the legacy ones (round 1)
// and the per-step comparison into result.json.
//
// The members the loop needs are TensorSharp.Models internals: they are reached by reflection, so one build of this
// probe runs against any TensorSharp.Models.dll placed beside it (a base build's and a changed build's; --engine legacy).
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
string engine = Value("--engine") ?? "legacy";
int rounds = int.Parse(Value("--rounds") ?? "1", CultureInfo.InvariantCulture);
if (engine is not ("legacy" or "runtime" or "both")) throw new ArgumentException("--engine is legacy, runtime or both");
if (rounds < 1 || (rounds > 1 && engine != "both")) throw new ArgumentException("--rounds takes a positive count, with --engine both");
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
var comparisons = new List<object>();
var total = Stopwatch.StartNew();
float[] x0 = (float[])x.Clone();
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
    // The runtime's denoiser over the legacy DiT's mapping (one device copy of each weight), made at its first use.
    Type? runtimeType = engine == "legacy" ? null : models.GetType("TensorSharp.Models.MiniMaxH3Image.MiniMaxH3ImageDenoiser", throwOnError: true);
    IDisposable? runtime = null;
    double preloadSeconds = 0;
    try
    {
        if (runtimeType != null)
        {
            runtime = (IDisposable)Invoke(runtimeType, null, "SharingLegacyWeights", dit, backend)!;
            if (engine == "runtime")
            {
                var clock = Stopwatch.StartNew();
                Invoke(runtimeType, runtime, "Preload");
                preloadSeconds = clock.Elapsed.TotalSeconds;
            }
        }
        float[] Legacy(object layout, int start, int stop) =>
            (float[])Invoke(typeof(MiniMaxH3DiT), dit, "ForwardImage", x, imageTokens, textHidden, textLength, layout, grid, start, stop, null, 0)!;
        float[] Runtime(object layout, int start, int stop) =>
            (float[])((float[])Invoke(runtimeType!, runtime, "Forward", x, imageTokens, textHidden, textLength, layout, grid, start, stop, null, 0)!).Clone();
        for (int round = 0; round < rounds; round++)
        {
            Array.Copy(x0, x, x.Length);
            for (int i = 0; i < steps; i++)
            {
                float sigma = sigmas[i], next = sigmas[i + 1];
                object layout = Invoke(layoutType, null, "Build", textLength, lh, lw, sigma, null, null)!;
                var interval = Invoke(sampling, null, "Interval", grid, sigma, next)!;
                int start = (int)interval.GetType().GetField("Item1")!.GetValue(interval)!;
                int stop = (int)interval.GetType().GetField("Item2")!.GetValue(interval)!;
                float[]? v = null, r = null;
                double legacySeconds = 0, runtimeSeconds = 0;
                bool legacyFirst = (round + i) % 2 == 0;
                for (int arm = 0; arm < 2; arm++)
                {
                    bool legacyArm = (arm == 0) == legacyFirst;
                    if (legacyArm ? engine == "runtime" : engine == "legacy") continue;
                    var clock = Stopwatch.StartNew();
                    if (legacyArm) { v = Legacy(layout, start, stop); legacySeconds = clock.Elapsed.TotalSeconds; }
                    else { r = Runtime(layout, start, stop); runtimeSeconds = clock.Elapsed.TotalSeconds; }
                }
                float[] advance = v ?? r!;
                if (advance.Any(f => !float.IsFinite(f))) throw new InvalidDataException($"step {i + 1}: a non-finite velocity");
                if (round == 0)
                {
                    byte[] vb = Bytes(advance);
                    File.WriteAllBytes(Path.Combine(outDir, $"velocity-{i + 1}.f32"), vb);
                    if (v != null && r != null) File.WriteAllBytes(Path.Combine(outDir, $"runtime-velocity-{i + 1}.f32"), Bytes(r));
                    stepRecords.Add(new { step = i + 1, sigma, next, heads = new[] { start, stop },
                        seconds = v != null ? legacySeconds : runtimeSeconds, sha256 = Sha(vb) });
                }
                string line = $"round {round + 1} step {i + 1}/{steps} sigma {sigma:0.#####} -> {next:0.#####} heads {start}..{stop}";
                if (v != null && r != null)
                {
                    int differ = 0;
                    double maxAbs = 0, maxV = 0;
                    for (int k = 0; k < v.Length; k++)
                    {
                        if (BitConverter.SingleToInt32Bits(v[k]) != BitConverter.SingleToInt32Bits(r[k])) differ++;
                        maxAbs = Math.Max(maxAbs, Math.Abs((double)r[k] - v[k]));
                        maxV = Math.Max(maxV, Math.Abs((double)v[k]));
                    }
                    comparisons.Add(new { round = round + 1, step = i + 1, legacyFirst, legacySeconds, runtimeSeconds, valuesDiffering = differ,
                        maxAbsDiff = maxAbs, maxRelDiff = maxAbs / Math.Max(maxV, 1e-30), legacySha256 = Sha(Bytes(v)), runtimeSha256 = Sha(Bytes(r)) });
                    line += $" legacy {legacySeconds:F2}s runtime {runtimeSeconds:F2}s ({(legacyFirst ? "A B" : "B A")}) " +
                            (differ == 0 ? "bit-exact" : $"{differ} values differ, max |d| {maxAbs:E3} ({maxAbs / Math.Max(maxV, 1e-30):E3} rel)");
                }
                else line += $" {(v != null ? legacySeconds : runtimeSeconds):F2}s {Sha(Bytes(advance))}";
                Console.WriteLine(line);
                float dt = next - sigma;
                for (long k = 0; k < x.Length; k++) x[k] += advance[k] * dt;
            }
        }
        if (runtime != null && preloadSeconds > 0) Console.WriteLine($"runtime preload {preloadSeconds:F2}s");
    }
    finally
    {
        runtime?.Dispose();
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
    stepsRecorded = stepRecords, latentSha256 = Sha(latent), engine, rounds, comparisons,
};
File.WriteAllText(Path.Combine(outDir, "result.json"), JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine($"latent {Sha(latent)}");
// Release the ggml backends before exit, as the CLI does (ggml-metal asserts on residency sets still held at teardown).
TensorSharp.GGML.GgmlBasicOps.Shutdown();
return 0;
