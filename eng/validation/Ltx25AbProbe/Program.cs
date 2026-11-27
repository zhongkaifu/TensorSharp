// A/B probe for LTX 2.5 generation through the public model API only (ModelBase.Create and
// IVideoGenerationModel.GenerateVideo), so the same source builds against any tree: the retired direct
// pipeline and the diffusion runtime alike. Each case writes its decoded RGB (F32, frame-major) and its
// PCM channels (F32) beside a JSON report of the request, the phase events and the timings; compare two
// runs with eng/validation/ltx25-ab-compare.py.
//
//   dotnet run -c Release --project eng/validation/Ltx25AbProbe -- --model <denoiser.gguf> --backend cuda
//       --cases cases.json --output <dir> [--label legacy] [--repeats 1] [--only name,name]
//
// cases.json: [{ "name": "t2v", "prompt": "...", "width": 1024, "height": 1024, "frames": 17, "seed": 42,
//               "mode": null, "image": null, "endImage": null, "referenceImages": [], "imageConditions":
//               [{ "path": "...", "frame": 0, "strength": 1 }], "audio": true, "sampler": null }]
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.Models;
using TensorSharp.Models.Video;
using TensorSharp.Runtime;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i += 2)
    if (i + 1 >= args.Length || !args[i].StartsWith("--") || !options.TryAdd(args[i][2..], args[i + 1]))
        throw new ArgumentException("Expected unique --option value pairs.");
string Required(string name) => options.GetValueOrDefault(name) ?? throw new ArgumentException("Missing --" + name);
string model = Path.GetFullPath(Required("model"));
BackendType backend = Enum.Parse<BackendType>(Required("backend"), true);
string outputDir = Path.GetFullPath(Required("output"));
string label = options.GetValueOrDefault("label", "run");
int repeats = int.Parse(options.GetValueOrDefault("repeats", "1"), CultureInfo.InvariantCulture);
var only = options.TryGetValue("only", out string? names) ? names.Split(',').ToHashSet(StringComparer.Ordinal) : null;
Directory.CreateDirectory(outputDir);
var cases = JsonNode.Parse(File.ReadAllText(Required("cases")))!.AsArray();

string Hash(string path) { using var f = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(f)); }
static string? Str(JsonNode? n) => n?.GetValue<string>();
static int Int(JsonNode? n, int fallback) => n == null ? fallback : n.GetValue<int>();

var clock = Stopwatch.StartNew();
using ModelBase loaded = ModelBase.Create(model, backend);
double factoryMs = clock.Elapsed.TotalMilliseconds;
if (loaded is not IVideoGenerationModel video) throw new InvalidOperationException("Not a video model: " + loaded.GetType().Name);
Console.WriteLine($"[ab] {label}: {loaded.GetType().FullName} on {backend}, factory {factoryMs:F0} ms");

var report = new JsonObject
{
    ["label"] = label, ["model"] = model, ["backend"] = backend.ToString(), ["model_type"] = loaded.GetType().FullName,
    ["factory_ms"] = factoryMs, ["started_utc"] = DateTimeOffset.UtcNow.ToString("O"), ["runs"] = new JsonArray(),
};
void Save() => File.WriteAllText(Path.Combine(outputDir, label + ".json"), report.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
Save();
for (int repeat = 0; repeat < repeats; repeat++)
foreach (JsonNode? node in cases)
{
    var c = node!.AsObject();
    string name = Str(c["name"]) ?? throw new ArgumentException("a case needs a name");
    if (only != null && !only.Contains(name)) continue;
    var p = new VideoGenerationParams
    {
        Mode = Str(c["mode"])!,
        Width = Int(c["width"], 0), Height = Int(c["height"], 0), Frames = Int(c["frames"], 0), Fps = Int(c["fps"], 0),
        Seed = c["seed"]?.GetValue<long>() ?? 42, Sampler = Str(c["sampler"])!,
        GenerateAudio = c["audio"]?.GetValue<bool>() ?? true,
        ImagePath = Str(c["image"])!, EndImagePath = Str(c["endImage"])!,
    };
    if (c["referenceImages"] is JsonArray refs && refs.Count > 0) p.ReferenceImagePaths = refs.Select(r => r!.GetValue<string>()).ToArray();
    if (c["referenceVideos"] is JsonArray vids && vids.Count > 0) p.ReferenceVideoPaths = vids.Select(r => r!.GetValue<string>()).ToArray();
    if (c["imageConditions"] is JsonArray conds && conds.Count > 0)
        p.ImageConditions = conds.Select(x => new VideoImageCondition
        {
            ImagePath = x!["path"]!.GetValue<string>(), FrameIndex = x["frame"]!.GetValue<int>(),
            Strength = x["strength"]?.GetValue<float>() ?? 1f,
        }).ToArray();
    var events = new JsonArray();
    var phaseStart = new Dictionary<string, double>(StringComparer.Ordinal);
    int stepsSeen = 0;
    p.OnStep = (step, total) => stepsSeen = Math.Max(stepsSeen, step);
    p.OnProgress = e => events.Add(new JsonObject
    {
        ["phase"] = e.Phase, ["stage"] = e.Stage, ["step"] = e.Step, ["total"] = e.TotalSteps,
        ["elapsed"] = Math.Round(e.ElapsedSeconds, 4), ["detail"] = e.Detail,
    });
    string prompt = Str(c["prompt"]) ?? "A red toy car drives slowly along a quiet road in warm afternoon light.";
    Console.WriteLine($"[ab] {label} {name} (round {repeat}): {p.Width}x{p.Height}x{p.Frames} mode={p.Mode ?? "auto"} seed={p.Seed}");
    clock.Restart();
    var result = video.GenerateVideo(prompt, p);
    double ms = clock.Elapsed.TotalMilliseconds;
    string stem = Path.Combine(outputDir, $"{label}.{name}.{repeat}");
    using (var raw = new FileStream(stem + ".rgb.f32", FileMode.Create))
        foreach (var frame in result.Frames) raw.Write(MemoryMarshal.AsBytes(frame.Pixels.AsSpan()));
    var audioHashes = new JsonArray();
    if (result.Audio != null)
        for (int ch = 0; ch < result.Audio.ChannelCount; ch++)
        {
            string path = stem + $".audio-{ch}.f32";
            File.WriteAllBytes(path, MemoryMarshal.AsBytes(result.Audio.Channels[ch].AsSpan()).ToArray());
            audioHashes.Add(Hash(path));
        }
    var run = new JsonObject
    {
        ["case"] = name, ["round"] = repeat, ["generation_ms"] = ms, ["steps"] = stepsSeen,
        ["frames"] = result.Frames.Length, ["width"] = result.Frames[0].Width, ["height"] = result.Frames[0].Height,
        ["fps"] = result.Fps, ["seed"] = result.Seed, ["audio_rate"] = result.Audio?.SampleRate, ["audio_samples"] = result.Audio?.SampleCount,
        ["rgb_sha256"] = Hash(stem + ".rgb.f32"), ["audio_sha256"] = audioHashes, ["events"] = events,
        ["process_peak_working_set"] = Process.GetCurrentProcess().PeakWorkingSet64,
    };
    report["runs"]!.AsArray().Add(run);
    Save();
    Console.WriteLine($"[ab] {label} {name} (round {repeat}): {ms / 1000:F2} s, rgb {run["rgb_sha256"]}");
}
report["finished_utc"] = DateTimeOffset.UtcNow.ToString("O");
Save();
