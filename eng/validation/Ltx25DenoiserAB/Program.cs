// LTX 2.5 denoiser steps, frozen and compared across implementations. It uses only the denoiser's component surface
// (new Ltx25Denoiser(checkpoint, allocator, ingredientsAdapter) and Forward(Ltx25DenoiserInput)), which the retired direct
// denoiser and the diffusion runtime's graph-program denoiser both have, so the same source builds against either tree:
//
//   fixture  the synthetic checkpoint's oracle cases (InferenceWeb.Tests/Fixtures/Ltx25/denoiser.bin, oracle.json), each also
//            with a synthetic Ingredients adapter over all ten adapted projections of both blocks; every output's bits are
//            written (the lane file InferenceWeb.Tests/Fixtures/Ltx25/LegacyGoldens/<lane>.json holds them).
//   real     steps of the real checkpoint at generation geometries (a first pass, an image guide's per-token timesteps, the
//            refinement's full size, appended guides, an Ingredients reference pass with an adapter): each output's SHA-256 and
//            statistics in the report, the raw F32 beside it.
//   compare  two reports: per case bitwise equality, max |a - b| and relative L2.
//
//   dotnet run -c Release --project eng/validation/Ltx25DenoiserAB -- fixture --backend cpu --fixtures InferenceWeb.Tests/Fixtures/Ltx25 --out lane.json
//   dotnet run -c Release --project eng/validation/Ltx25DenoiserAB -- real --backend cuda --checkpoint <gguf> [--lora <ingredients>] [--cold N] --out real.json
//   dotnet run -c Release --project eng/validation/Ltx25DenoiserAB -- write-lora --checkpoint <gguf> --out synthetic-ingredients.safetensors
//   dotnet run -c Release --project eng/validation/Ltx25DenoiserAB -- compare <a.json> <b.json>
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Ltx25;
using TensorSharp.Runtime;

string mode = args.Length > 0 ? args[0] : throw new ArgumentException("fixture | real | write-lora | compare");
var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 1; i + 1 < args.Length; i += 2) if (args[i].StartsWith("--")) options[args[i][2..]] = args[i + 1];
string Opt(string name, string fallback = null) => options.TryGetValue(name, out string v) ? v : fallback ?? throw new ArgumentException("--" + name);

if (mode == "compare") { Compare(args[1], args[2]); return; }
if (mode == "write-lora") { WriteRealLora(Opt("checkpoint"), Opt("out")); return; }

IAllocator allocator = Opt("backend").ToLowerInvariant() switch
{
    "cpu" => new CpuAllocator(BlasEnum.DotNet),
    "cuda" => new CudaAllocator(),
    "mlx" => new MlxAllocator(),
    var b => throw new ArgumentException("backend " + b),
};
string lane = $"{Opt("backend").ToLowerInvariant()}|{RuntimeInformation.RuntimeIdentifier}|{Device()}";
var report = new JsonObject { ["lane"] = lane, ["implementation"] = typeof(Ltx25Denoiser).Assembly.GetName().Version?.ToString(), ["cases"] = new JsonObject() };
string outPath = Path.GetFullPath(Opt("out"));
Directory.CreateDirectory(Path.GetDirectoryName(outPath));
if (mode == "fixture") Fixture(Opt("fixtures"));
else if (mode == "real") Real(Opt("checkpoint"), options.GetValueOrDefault("lora"), options.TryGetValue("blocks", out string bl) ? int.Parse(bl) : null,
    options.GetValueOrDefault("only"), options.TryGetValue("cold", out string cl) ? int.Parse(cl) : 0);
else throw new ArgumentException(mode);
File.WriteAllText(outPath, report.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"[ltx25-ab] {lane}: {((JsonObject)report["cases"]).Count} cases -> {outPath}");
(allocator as IDisposable)?.Dispose();

string Device()
{
    if (allocator is CudaAllocator)
    {
        try { var p = Process.Start(new ProcessStartInfo("nvidia-smi", "--query-gpu=name --format=csv,noheader") { RedirectStandardOutput = true }); string n = p.StandardOutput.ReadLine(); p.WaitForExit(); return n?.Trim() ?? "cuda"; }
        catch { return "cuda"; }
    }
    try { foreach (string l in File.ReadLines("/proc/cpuinfo")) if (l.StartsWith("model name")) return l[(l.IndexOf(':') + 1)..].Trim(); } catch { }
    return RuntimeInformation.ProcessArchitecture.ToString();
}

static string Hex(float[] v) { var sb = new StringBuilder(v.Length * 8); foreach (float f in v) sb.Append(BitConverter.SingleToInt32Bits(f).ToString("x8")); return sb.ToString(); }
static string Sha(float[] v) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(v.AsSpan())));
static float[] Floats(JsonElement e, string name) => e.GetProperty(name).EnumerateArray().Select(x => x.GetSingle()).ToArray();

void Record(string name, Ltx25DenoiserOutput o, bool full, double ms)
{
    JsonObject Out(float[] v)
    {
        double max = 0, l2 = 0; foreach (float f in v) { max = Math.Max(max, Math.Abs(f)); l2 += (double)f * f; }
        var j = new JsonObject { ["sha256"] = Sha(v), ["count"] = v.Length, ["maxAbs"] = max, ["l2"] = Math.Sqrt(l2) };
        if (full) j["bits"] = Hex(v);
        return j;
    }
    ((JsonObject)report["cases"])[name] = new JsonObject { ["video"] = Out(o.VideoVelocity), ["audio"] = Out(o.AudioVelocity), ["blocks"] = o.ExecutedBlocks, ["ms"] = ms };
    if (!full)
    {
        string stem = Path.Combine(Path.GetDirectoryName(outPath), Path.GetFileNameWithoutExtension(outPath) + "." + name);
        File.WriteAllBytes(stem + ".video.f32", MemoryMarshal.AsBytes(o.VideoVelocity.AsSpan()).ToArray());
        File.WriteAllBytes(stem + ".audio.f32", MemoryMarshal.AsBytes(o.AudioVelocity.AsSpan()).ToArray());
    }
    Console.WriteLine($"[ltx25-ab] {name}: video {Sha(o.VideoVelocity)[..16]} audio {Sha(o.AudioVelocity)[..16]} {ms:F1} ms");
}

void Fixture(string dir)
{
    using var oracle = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "oracle.json")));
    string checkpoint = Path.Combine(Path.GetTempPath(), "ltx25-ab-" + Guid.NewGuid().ToString("N") + ".gguf");
    string lora = Path.ChangeExtension(checkpoint, ".safetensors");
    File.Copy(Path.Combine(dir, "denoiser.bin"), checkpoint);
    try
    {
        using (var file = new GgufFile(checkpoint)) WriteFixtureLora(file, Ltx25Config.FromGguf(file), lora);
        report["checkpointSha256"] = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(checkpoint)));
        report["adapterSha256"] = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(lora)));
        using var model = new Ltx25Denoiser(checkpoint, allocator, lora);
        var cases = oracle.RootElement.GetProperty("cases");
        for (int i = 0; i < cases.GetArrayLength(); i++)
            foreach (bool adapted in new[] { false, true })
            {
                var clock = Stopwatch.StartNew();
                var o = model.Forward(FixtureInput(cases[i], adapted));
                Record($"case{i}{(adapted ? "-ingredients" : "")}", o, full: true, clock.Elapsed.TotalMilliseconds);
            }
    }
    finally { File.Delete(checkpoint); File.Delete(lora); }
}

static Ltx25DenoiserInput FixtureInput(JsonElement value, bool lora) => new()
{
    Video = Floats(value, "video"), Audio = Floats(value, "audio"),
    VideoTokens = value.GetProperty("videoTokens").GetInt32(), AudioTokens = value.GetProperty("audioTokens").GetInt32(),
    VideoContext = Floats(value, "videoContext"), AudioContext = Floats(value, "audioContext"),
    VideoContextTokens = value.GetProperty("videoContextTokens").GetInt32(), AudioContextTokens = value.GetProperty("audioContextTokens").GetInt32(),
    VideoPositions = new(value.GetProperty("videoTokens").GetInt32(), 3, Floats(value, "videoPositions")),
    AudioPositions = new(value.GetProperty("audioTokens").GetInt32(), 1, Floats(value, "audioPositions")),
    VideoSigma = value.GetProperty("videoSigma").GetSingle(), AudioSigma = value.GetProperty("audioSigma").GetSingle(),
    VideoTimesteps = Floats(value, "videoTimesteps"), AudioTimesteps = Floats(value, "audioTimesteps"),
    KeyframeMask = value.GetProperty("keyframeMask").ValueKind == JsonValueKind.Null ? null : Floats(value, "keyframeMask"),
    FrequencyPrecision = Enum.Parse<Ltx25FrequencyPrecision>(value.GetProperty("precision").GetString()),
    UseReferenceLora = lora,
};

static void WriteFixtureLora(GgufFile file, Ltx25Config config, string path)
{
    var factors = new List<(string Name, int[] Shape, float[] Values, string Dtype)>();
    int seed = 1;
    for (int b = 0; b < config.Layers; b++)
        foreach (string t in Ab.Targets)
        {
            var shape = file.Tensors[config.Prefix + $"transformer_blocks.{b}.{t}.weight"].Shape;
            int inner = (int)shape[0], outer = (int)shape[1], rank = 2;
            float[] a = Enumerable.Range(0, rank * inner).Select(i => MathF.Sin(i * .37f + seed) * .25f).ToArray();
            float[] bv = Enumerable.Range(0, outer * rank).Select(i => MathF.Cos(i * .23f + seed) * .125f).ToArray();
            factors.Add(($"diffusion_model.transformer_blocks.{b}.{t}.lora_A.weight", new[] { rank, inner }, a, "F32"));
            factors.Add(($"diffusion_model.transformer_blocks.{b}.{t}.lora_B.weight", new[] { outer, rank }, bv, "F32"));
            seed++;
        }
    WriteSafetensors(path, factors);
}

// A synthetic adapter with the Ingredients layout over the real checkpoint (rank 128, BF16, every adapted projection of the
// 48 blocks): deterministic factors whose updates are a few percent of their projections', for parity and timing where the
// gated publisher file is not available. It says nothing about the trained adapter's output quality.
static void WriteRealLora(string checkpoint, string path)
{
    using var file = new GgufFile(checkpoint);
    var config = Ltx25Config.FromGguf(file);
    var factors = new List<(string Name, int[] Shape, float[] Values, string Dtype)>();
    ulong state = 0x9E3779B97F4A7C15UL;
    float Normal()
    {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        double u1 = ((state >> 11) + 1.0) / (1UL << 53);
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        double u2 = ((state >> 11) + 0.5) / (1UL << 53);
        return (float)(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
    }
    for (int b = 0; b < config.Layers; b++)
        foreach (string t in Ab.Targets)
        {
            var shape = file.Tensors[config.Prefix + $"transformer_blocks.{b}.{t}.weight"].Shape;
            int inner = (int)shape[0], outer = (int)shape[1], rank = 128;
            float sa = 1f / MathF.Sqrt(inner), sb = .02f / MathF.Sqrt(rank);
            var a = new float[rank * inner]; for (int i = 0; i < a.Length; i++) a[i] = Normal() * sa;
            var bv = new float[outer * rank]; for (int i = 0; i < bv.Length; i++) bv[i] = Normal() * sb;
            factors.Add(($"diffusion_model.transformer_blocks.{b}.{t}.lora_A.weight", new[] { rank, inner }, a, "BF16"));
            factors.Add(($"diffusion_model.transformer_blocks.{b}.{t}.lora_B.weight", new[] { outer, rank }, bv, "BF16"));
        }
    WriteSafetensors(path, factors);
    Console.WriteLine($"[ltx25-ab] wrote {factors.Count} synthetic Ingredients factors to {path}");
}

static void WriteSafetensors(string path, List<(string Name, int[] Shape, float[] Values, string Dtype)> tensors)
{
    var header = new Dictionary<string, object> { ["__metadata__"] = new Dictionary<string, string> { ["model_version"] = "2.5", ["reference_downscale_factor"] = "1" } };
    long offset = 0;
    foreach (var t in tensors)
    {
        long bytes = t.Values.LongLength * (t.Dtype == "F32" ? 4 : 2);
        header[t.Name] = new { dtype = t.Dtype, shape = t.Shape, data_offsets = new[] { offset, offset + bytes } };
        offset += bytes;
    }
    byte[] json = JsonSerializer.SerializeToUtf8Bytes(header);
    int padded = (json.Length + 7) / 8 * 8;
    using var output = new BinaryWriter(File.Create(path));
    output.Write((ulong)padded); output.Write(json);
    for (int i = json.Length; i < padded; i++) output.Write((byte)' ');
    foreach (var t in tensors)
        foreach (float v in t.Values)
            if (t.Dtype == "F32") output.Write(v); else output.Write((ushort)((BitConverter.SingleToUInt32Bits(v) + 0x7FFF + ((BitConverter.SingleToUInt32Bits(v) >> 16) & 1)) >> 16));
}

void Real(string checkpoint, string lora, int? blocks, string only, int cold)
{
    // --cold N: each case's step N times as a clip's first step runs it (the device copies released and the pool trimmed
    // before each, as after a clip's decode), recording every time; the median is the case's ms.
    var release = typeof(Ltx25Denoiser).GetMethod("ReleaseDevice", Type.EmptyTypes)
        ?? typeof(Ltx25Denoiser).GetMethod("ReleaseDeviceResidency", Type.EmptyTypes);
    using var model = new Ltx25Denoiser(checkpoint, allocator, lora);
    ulong state = 0x2545F4914F6CDD1DUL;
    float[] Normal(long n, float scale = 1f)
    {
        var v = new float[n];
        for (long i = 0; i < n; i++)
        {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            double u1 = ((state >> 11) + 1.0) / (1UL << 53);
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            double u2 = ((state >> 11) + 0.5) / (1UL << 53);
            v[i] = (float)(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2)) * scale;
        }
        return v;
    }
    int contextTokens = 1024;
    float[] videoContext = Normal(contextTokens * 4096L), audioContext = Normal(contextTokens * 2048L);
    Ltx25GenerationSettings Settings(int frames, int width, int height)
    {
        var latent = Ltx25Geometry.VideoLatentShape(frames, height, width);
        int audioTokens = Math.Max(1, (int)Math.Round(frames / 24.0 * 25, MidpointRounding.ToEven));
        return new Ltx25GenerationSettings(width, height, frames, 24, latent.Width, latent.Height, latent.Frames, audioTokens, 1);
    }
    Ltx25DenoiserInput Plain(int frames, int width, int height, float sigma, int extraFrames = 0)
    {
        var settings = Settings(frames, width, height);
        int spatial = settings.LatentWidth * settings.LatentHeight, tokens = settings.VideoTokens + extraFrames * spatial;
        return new Ltx25DenoiserInput
        {
            Video = Normal(tokens * 128L), VideoTokens = tokens,
            Audio = Normal((long)settings.AudioTokens * 128), AudioTokens = settings.AudioTokens,
            VideoContext = videoContext, VideoContextTokens = contextTokens, AudioContext = audioContext, AudioContextTokens = contextTokens,
            VideoPositions = extraFrames == 0
                ? Ltx25Geometry.VideoPositions(settings.LatentFrames, settings.LatentHeight, settings.LatentWidth, settings.Fps)
                : new(tokens, 3, Ltx25Geometry.VideoPositions(settings.LatentFrames, settings.LatentHeight, settings.LatentWidth, settings.Fps).Boundaries
                    .Concat(Ltx25Geometry.VideoPositions(extraFrames, settings.LatentHeight, settings.LatentWidth, settings.Fps).Boundaries).ToArray()),
            AudioPositions = Ltx25Geometry.AudioPositions(settings.AudioTokens),
            VideoSigma = sigma, AudioSigma = sigma, FrequencyPrecision = Ltx25FrequencyPrecision.Float64,
            KeyframeMask = Enumerable.Range(0, tokens).Select(t => t < spatial ? 1f : 0f).ToArray(),
        };
    }
    Ltx25DenoiserInput Guided(int frames, int width, int height, float sigma, (int Frame, float Strength)[] guides)
    {
        var settings = Settings(frames, width, height);
        int spatial = settings.LatentWidth * settings.LatentHeight, appended = guides.Count(g => g.Frame != 0);
        var plain = Plain(frames, width, height, sigma);
        plain = new Ltx25DenoiserInput
        {
            Video = Normal((long)(settings.VideoTokens + appended * spatial) * 128), VideoTokens = settings.VideoTokens + appended * spatial,
            Audio = plain.Audio, AudioTokens = plain.AudioTokens, VideoContext = videoContext, VideoContextTokens = contextTokens,
            AudioContext = audioContext, AudioContextTokens = contextTokens, VideoPositions = plain.VideoPositions,
            AudioPositions = plain.AudioPositions, VideoSigma = sigma, AudioSigma = sigma, FrequencyPrecision = plain.FrequencyPrecision,
        };
        var images = guides.Select(g => new Ltx25EncodedImage(Normal(spatial * 128L), g.Frame, g.Strength)).ToList();
        var conditioned = Ltx25ImageConditioning.Apply(plain, settings, images);
        return new Ltx25DenoiserInput
        {
            Video = conditioned.Video, VideoTokens = conditioned.VideoTokens, Audio = conditioned.Audio, AudioTokens = conditioned.AudioTokens,
            VideoContext = conditioned.VideoContext, VideoContextTokens = conditioned.VideoContextTokens,
            AudioContext = conditioned.AudioContext, AudioContextTokens = conditioned.AudioContextTokens,
            VideoPositions = conditioned.VideoPositions, AudioPositions = conditioned.AudioPositions,
            VideoSigma = sigma, AudioSigma = sigma, FrequencyPrecision = conditioned.FrequencyPrecision,
            VideoTimesteps = conditioned.VideoDenoiseMask.Select(m => m * sigma).ToArray(), KeyframeMask = conditioned.KeyframeMask,
        };
    }
    void Run(string name, Ltx25DenoiserInput input, int repeat = 2)
    {
        if (only != null && !only.Split(',').Contains(name)) return;
        Ltx25DenoiserOutput o = null;
        double best = double.MaxValue;
        var times = new List<double>();
        var peaks = new List<long>();
        var backings = new List<long>();
        for (int r = 0; r < (cold > 0 ? cold : repeat); r++)
        {
            if (cold > 0)
            {
                release.Invoke(model, null);
                if (allocator is CudaAllocator c) { c.Synchronize(); c.TrimPool(); }
            }
            // CUDA: the step's peak live device bytes (the allocator's backing less its cached free blocks) and peak backing
            // (live plus the blocks the pool caches), polled beside it.
            long peak = 0, peakBacking = 0;
            using var sampling = new CancellationTokenSource();
            Task sampler = allocator is CudaAllocator live ? Task.Run(() =>
            {
                while (!sampling.IsCancellationRequested)
                {
                    long l = live.LiveBytes;
                    peak = Math.Max(peak, l);
                    peakBacking = Math.Max(peakBacking, l + live.GetStats().CachedBytes);
                    Thread.Sleep(1);
                }
            }) : Task.CompletedTask;
            var clock = Stopwatch.StartNew();
            o = model.Forward(input, blocks);
            times.Add(clock.Elapsed.TotalMilliseconds);
            best = Math.Min(best, times[^1]);
            sampling.Cancel();
            sampler.Wait();
            peaks.Add(peak);
            backings.Add(peakBacking);
        }
        if (cold > 0) best = times.Order().ElementAt(times.Count / 2);
        Record(name, o, full: false, best);
        ((JsonObject)((JsonObject)report["cases"])[name])["times"] = new JsonArray(times.Select(t => (JsonNode)Math.Round(t, 1)).ToArray());
        if (allocator is CudaAllocator)
        {
            ((JsonObject)((JsonObject)report["cases"])[name])["peakLiveBytes"] = new JsonArray(peaks.Select(p => (JsonNode)p).ToArray());
            ((JsonObject)((JsonObject)report["cases"])[name])["peakBackingBytes"] = new JsonArray(backings.Select(p => (JsonNode)p).ToArray());
        }
    }
    Run("first-pass-512x512x17", Plain(17, 512, 512, 1f));
    Run("i2v-first-pass-512x512x17", Guided(17, 512, 512, .9875f, new[] { (0, .7f) }));
    Run("refine-1024x1024x17", Guided(17, 1024, 1024, .909375f, new[] { (0, 1f) }));
    Run("keyframes-384x256x49", Guided(49, 384, 256, .725f, new[] { (0, 1f), (24, .8f), (48, 1f) }));
    if (lora != null)
    {
        // An Ingredients pass: the 768x448x121 canvas and a reference clip's 16 latent frames appended clean (timestep 0).
        var refInput = Plain(121, 768, 448, .975f, extraFrames: 16);
        int canvas = 16 * 14 * 24;
        Run("ingredients-768x448x121", new Ltx25DenoiserInput
        {
            Video = refInput.Video, VideoTokens = refInput.VideoTokens, Audio = refInput.Audio, AudioTokens = refInput.AudioTokens,
            VideoContext = refInput.VideoContext, VideoContextTokens = refInput.VideoContextTokens,
            AudioContext = refInput.AudioContext, AudioContextTokens = refInput.AudioContextTokens,
            VideoPositions = refInput.VideoPositions, AudioPositions = refInput.AudioPositions, VideoSigma = .975f, AudioSigma = .975f,
            FrequencyPrecision = refInput.FrequencyPrecision,
            VideoTimesteps = Enumerable.Range(0, refInput.VideoTokens).Select(t => t < canvas ? .975f : 0f).ToArray(),
            KeyframeMask = refInput.KeyframeMask, UseReferenceLora = true,
        }, repeat: int.Parse(options.GetValueOrDefault("ingredients-repeat", "1")));
    }
}

static void Compare(string aPath, string bPath)
{
    using var a = JsonDocument.Parse(File.ReadAllText(aPath));
    using var b = JsonDocument.Parse(File.ReadAllText(bPath));
    var bc = b.RootElement.GetProperty("cases");
    Console.WriteLine($"{a.RootElement.GetProperty("lane").GetString()}  vs  {b.RootElement.GetProperty("lane").GetString()}");
    foreach (var c in a.RootElement.GetProperty("cases").EnumerateObject())
    {
        if (!bc.TryGetProperty(c.Name, out var other)) { Console.WriteLine($"{c.Name,-34} missing in b"); continue; }
        foreach (string stream in new[] { "video", "audio" })
        {
            var x = c.Value.GetProperty(stream); var y = other.GetProperty(stream);
            bool same = x.GetProperty("sha256").GetString() == y.GetProperty("sha256").GetString();
            string detail = "";
            float[] fa = Values(aPath, c.Name, stream, x), fb = Values(bPath, c.Name, stream, y);
            if (!same && fa != null && fb != null && fa.Length == fb.Length)
            {
                double max = 0, d2 = 0, n2 = 0;
                for (int i = 0; i < fa.Length; i++) { double d = (double)fa[i] - fb[i]; max = Math.Max(max, Math.Abs(d)); d2 += d * d; n2 += (double)fa[i] * fa[i]; }
                detail = $"max |a-b| {max:G4}, rel L2 {Math.Sqrt(d2 / Math.Max(n2, 1e-30)):G4}";
            }
            Console.WriteLine($"{c.Name,-34} {stream,-5} {(same ? "bitwise" : "DIFFERS")} {detail}  ({c.Value.GetProperty("ms").GetDouble():F0} vs {other.GetProperty("ms").GetDouble():F0} ms)");
        }
    }
}

static float[] Values(string report, string name, string stream, JsonElement e)
{
    if (e.TryGetProperty("bits", out var bits))
    {
        string h = bits.GetString(); var v = new float[h.Length / 8];
        for (int i = 0; i < v.Length; i++) v[i] = BitConverter.Int32BitsToSingle(Convert.ToInt32(h.Substring(i * 8, 8), 16));
        return v;
    }
    string path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(report)), Path.GetFileNameWithoutExtension(report) + "." + name + "." + stream + ".f32");
    return File.Exists(path) ? MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(path)).ToArray() : null;
}

static class Ab
{
    /// <summary>The ten projections the Ingredients adapter updates in every block (unscaled B @ A).</summary>
    internal static readonly string[] Targets = { "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0", "attn2.to_q", "attn2.to_k",
        "attn2.to_v", "attn2.to_out.0", "ff.net.0.proj", "ff.net.2" };
}
