// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen-Image-2.1 denoiser steps on a real transformer: the runtime's QwenImage21Denoiser (graph programs), and before the
// legacy model's deletion also its QwenImage21DiT, on the same inputs (random conditioning and latents from a seed), with each
// step's time and the two velocities' distance. Both see the same trajectory (the first engine's Euler updates).
//
//   dotnet run --project eng/validation/QwenImage21DenoiserAB -c Release -- --dit <gguf> --backend ggml_cpu --size 512 \
//       [--steps 4] [--text-rows 23] [--refs 0] [--engines runtime,legacy] [--out <dir>]
//
// --out writes each engine's velocities (little-endian F32, one file a step) so a later run (after the legacy engine is
// gone) compares against them with --against <dir>.
using System.Diagnostics;
using System.Globalization;
using TensorSharp.Models.QwenImage21;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;
using TensorSharp.Models;

static string Arg(string[] a, string name, string fallback)
{
    int i = Array.IndexOf(a, name);
    return i >= 0 && i + 1 < a.Length ? a[i + 1] : fallback;
}

string dit = Arg(args, "--dit", null) ?? throw new ArgumentException("--dit <gguf> is required");
if (args.Contains("--e2e")) { EndToEnd.Run(args, dit); return; }
BackendType backend = Arg(args, "--backend", "ggml_cpu") switch
{
    "cpu" => BackendType.Cpu,
    "ggml_cpu" => BackendType.GgmlCpu,
    "ggml_metal" => BackendType.GgmlMetal,
    "ggml_cuda" => BackendType.GgmlCuda,
    "ggml_vulkan" => BackendType.GgmlVulkan,
    string other => throw new ArgumentException($"unknown backend {other}"),
};
int size = int.Parse(Arg(args, "--size", "512"), CultureInfo.InvariantCulture);
int steps = int.Parse(Arg(args, "--steps", "4"), CultureInfo.InvariantCulture);
int textRows = int.Parse(Arg(args, "--text-rows", "23"), CultureInfo.InvariantCulture);
int refs = int.Parse(Arg(args, "--refs", "0"), CultureInfo.InvariantCulture);
string[] engines = Arg(args, "--engines", "runtime,legacy").Split(',', StringSplitOptions.RemoveEmptyEntries);
string outDir = Arg(args, "--out", null);
string against = Arg(args, "--against", null);
const int Channels = 64, TextDim = 4096, HeadDim = 128;

float[] Rand(long n, int seed, float scale)
{
    var rng = new Random(seed);
    var v = new float[n];
    for (long i = 0; i < n; i++) v[i] = scale * (float)(rng.NextDouble() * 2 - 1);
    return v;
}

// The request: [text, (ref slots, text) x refs, text], refs of 1 MP-ish reference latents at a quarter of the target, the target.
int h = size / 16, w = size / 16;
int rh = Math.Max(2, h / 2 / 2 * 2), rw = Math.Max(2, w / 2 / 2 * 2);
var slots = new List<int>();
for (int i = 0; i < textRows / 2; i++) slots.Add(0);
for (int r = 0; r < refs; r++)
{
    for (int i = 0; i < rh * rw / 4; i++) slots.Add(r + 1);
    slots.Add(0);
}
while (slots.Count < textRows + refs * (rh * rw / 4 + 1)) slots.Add(0);
int rows = slots.Count;
var conditioning = new ConditioningSet(Rand((long)rows * TextDim, 1, 2f), rows, TextDim, slots.ToArray());
var references = new LatentRef[refs];
var shapes = new List<(int, int)>();
for (int r = 0; r < refs; r++)
{
    references[r] = new LatentRef(r, new LatentShape(Channels, 1, rh, rw), Rand((long)rh * rw * Channels, 10 + r, 1f));
    shapes.Add((rh, rw));
}
shapes.Add((h, w));
QwenImage21Layout layout = QwenImage21Layout.Build(rows, refs > 0 ? slots.ToArray() : null, shapes, HeadDim);
float[] start = Rand((long)h * w * Channels, 99, 1.5f);
float[] sigmas = { 1.0f, 0.978453f, 0.95418f, 0.926626f, 0.89508f, 0.845148f, 0.704534f, 0.414568f, 0f };
Console.WriteLine($"Qwen-Image-2.1 denoiser A/B: {Path.GetFileName(dit)} on {backend}, {size}x{size} ({h * w} target rows), " +
                  $"{rows} conditioning rows, {refs} reference(s) {rh}x{rw}, prefix {layout.Prefix}, {steps} steps");

var results = new Dictionary<string, List<float[]>>();
// --against: the frozen run's velocities drive the trajectory, so this run's engines predict from that run's latents.
var frozenSteps = new List<float[]>();
if (against != null)
    for (int step = 0; step < steps; step++)
    {
        string file = Directory.GetFiles(against, $"*-step{step}.f32").OrderBy(f => f).FirstOrDefault();
        if (file == null) break;
        byte[] bytes = File.ReadAllBytes(file);
        var frozen = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, frozen, 0, bytes.Length);
        frozenSteps.Add(frozen);
    }
foreach (string engineName in engines)
{
    // "legacy2": a second legacy request in this process (warm JIT); likewise "runtime2".
    string engine = engineName.TrimEnd('2');
    var outputs = new List<float[]>();
    float[] x = (float[])start.Clone();
    var load = Stopwatch.StartNew();
    Func<int, float[]> predict;
    IDisposable owner;
    Action finish;
    switch (engine)
    {
        case "runtime":
        {
            var d = new QwenImage21Denoiser(dit, backend);
            owner = d;
            predict = step => d.Predict(x, conditioning, references, layout, sigmas[step], step, false, null);
            finish = d.FinishRequest;
            break;
        }
        case "legacy":
        {
            // Reflection: the legacy engine is deleted once its outputs are frozen; this probe then runs the runtime alone.
            var asm = typeof(QwenImage21Denoiser).Assembly;
            Type ditType = asm.GetType("TensorSharp.Models.QwenImage.QwenImage21DiT", throwOnError: true)!;
            object model = Activator.CreateInstance(ditType, System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public |
                System.Reflection.BindingFlags.NonPublic, null, new object[] { dit, backend, null, null }, null)!;
            owner = (IDisposable)model;
            var create = ditType.GetMethod("CreatePrefixCache", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic,
                new[] { typeof(float[]), typeof(int[]), typeof(float[][]) })!;
            float[][] refTokens = references.Select(r => r.Rows).ToArray();
            int[] refH = references.Select(r => r.Shape.Height).ToArray(), refW = references.Select(r => r.Shape.Width).ToArray();
            int[] slotArray = refs > 0 ? slots.ToArray() : null;
            object cache = create.Invoke(null, new object[] { conditioning.Hidden, slotArray, refTokens });
            var predictMethod = ditType.GetMethod("Predict", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
            predict = step => (float[])predictMethod.Invoke(model, new object[]
            {
                x, h, w, conditioning.Hidden, rows, sigmas[step], slotArray, refTokens, refH, refW, cache, step, false,
            })!;
            finish = () => ((IDisposable)cache)?.Dispose();
            break;
        }
        default: throw new ArgumentException($"unknown engine {engine}");
    }
    Console.WriteLine($"[{engineName}] load {load.Elapsed.TotalSeconds:F2}s");
    for (int step = 0; step < steps; step++)
    {
        var clock = Stopwatch.StartNew();
        float[] v = (float[])predict(step).Clone();
        double seconds = clock.Elapsed.TotalSeconds;
        outputs.Add(v);
        Console.WriteLine($"[{engineName}] step {step + 1}/{steps}: {seconds:F3}s");
        // The trajectory follows the first engine, so every engine predicts from the same latents.
        float[] driver = step < frozenSteps.Count ? frozenSteps[step] : results.Count == 0 ? v : results.First().Value[step];
        float dt = sigmas[step + 1] - sigmas[step];
        for (int i = 0; i < x.Length; i++) x[i] += dt * driver[i];
    }
    finish();
    owner.Dispose();
    results[engineName] = outputs;
    if (outDir != null)
    {
        Directory.CreateDirectory(outDir);
        for (int step = 0; step < outputs.Count; step++)
        {
            var bytes = new byte[outputs[step].Length * 4];
            Buffer.BlockCopy(outputs[step], 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(Path.Combine(outDir, $"{engine}-step{step}.f32"), bytes);
        }
    }
}

static (double Rel, double MaxAbs) Distance(float[] a, float[] b)
{
    double err = 0, norm = 0, max = 0;
    for (int i = 0; i < a.Length; i++)
    {
        double d = a[i] - (double)b[i];
        err += d * d;
        norm += a[i] * (double)a[i];
        max = Math.Max(max, Math.Abs(d));
    }
    return (Math.Sqrt(err / Math.Max(norm, 1e-30)), max);
}

if (results.Count == 2)
{
    var (first, second) = (results.First(), results.Last());
    for (int step = 0; step < steps; step++)
    {
        var (rel, max) = Distance(first.Value[step], second.Value[step]);
        bool same = first.Value[step].AsSpan().SequenceEqual(second.Value[step]);
        Console.WriteLine($"step {step + 1}: {first.Key} vs {second.Key}: relative L2 {rel:E3}, max |d| {max:E3}{(same ? " (bit-identical)" : "")}");
    }
}
foreach (var (engine, outputs) in results)
    for (int step = 0; step < Math.Min(outputs.Count, frozenSteps.Count); step++)
    {
        var (rel, max) = Distance(frozenSteps[step], outputs[step]);
        Console.WriteLine($"step {step + 1}: frozen vs {engine}: relative L2 {rel:E3}, max |d| {max:E3}");
    }

// End to end through a model's public API (TS_QWEN_IMAGE_* companions and TS_QWEN_IMAGE_VARIANT from the environment):
//   --e2e --model legacy|runtime --backend B --prompt P --out x.png [--image a.png ...] [--mask m.png] [--width W --height H]
//   [--seed S] [--steps N] [--cfg C] [--negative N] [--keep-source-size] [--previews K] [--lora path[:scale]]
public static class EndToEnd
{
    static string Arg(string[] a, string name, string fallback)
    {
        int i = Array.IndexOf(a, name);
        return i >= 0 && i + 1 < a.Length ? a[i + 1] : fallback;
    }

    public static void Run(string[] args, string dit)
    {
        BackendType backend = Arg(args, "--backend", "ggml_cpu") switch
        {
            "cpu" => BackendType.Cpu, "ggml_cpu" => BackendType.GgmlCpu, "ggml_metal" => BackendType.GgmlMetal,
            "ggml_cuda" => BackendType.GgmlCuda, "ggml_vulkan" => BackendType.GgmlVulkan, string o => throw new ArgumentException(o),
        };
        string kind = Arg(args, "--model", "runtime");
        var images = new List<TensorSharp.Models.QwenImage.RgbImage>();
        for (int i = 0; i < args.Length - 1; i++)
            if (args[i] == "--image") images.Add(TensorSharp.Models.QwenImage.ImageIO.Load(args[i + 1], preserveAlpha: true));
        var p = new TensorSharp.Models.QwenImage.QwenImageParams
        {
            Width = int.Parse(Arg(args, "--width", "0")), Height = int.Parse(Arg(args, "--height", "0")),
            Seed = long.Parse(Arg(args, "--seed", "0")), Steps = int.Parse(Arg(args, "--steps", "0")),
            CfgScale = float.Parse(Arg(args, "--cfg", "0"), CultureInfo.InvariantCulture),
            KeepSourceSize = args.Contains("--keep-source-size"),
            PreviewCount = int.Parse(Arg(args, "--previews", "0")),
        };
        if (Arg(args, "--negative", null) is string negative) p.NegativePrompt = negative;
        if (Arg(args, "--mask", null) is string mask) p.Mask = TensorSharp.Models.QwenImage.ImageIO.Load(mask, preserveAlpha: true);
        var previews = new List<(int, int)>();
        p.OnStep = (step, total, preview) => { if (preview != null) previews.Add((step, preview.Width)); };
        var loras = new List<LoraSpec>();
        for (int i = 0; i < args.Length - 1; i++)
            if (args[i] == "--lora")
            {
                string[] parts = args[i + 1].Split(':');
                loras.Add(new LoraSpec(parts[0], parts.Length > 1 ? float.Parse(parts[1], CultureInfo.InvariantCulture) : null));
            }
        var clock = Stopwatch.StartNew();
        TensorSharp.Models.QwenImage.RgbImage result;
        if (kind == "legacy")
        {
            var asm = typeof(QwenImage21Denoiser).Assembly;
            Type type = asm.GetType("TensorSharp.Models.QwenImage.QwenImageModel", throwOnError: true)!;
            using var model = (TensorSharp.Models.ModelBase)Activator.CreateInstance(type, dit, backend, 1, null)!;
            var image = (TensorSharp.Models.QwenImage.IImageGenerationModel)model;
            if (loras.Count > 0) ((ILoraPluginHost)model).SetLoras(loras);
            Console.WriteLine($"[e2e] legacy loaded {clock.Elapsed.TotalSeconds:F1}s");
            clock.Restart();
            result = images.Count == 0 ? image.GenerateImage(Arg(args, "--prompt", ""), p) : image.EditImage(Arg(args, "--prompt", ""), images, p);
        }
        else
        {
            using var model = new QwenImage21DiffusionModel(dit, backend);
            if (loras.Count > 0) model.SetLoras(loras);
            Console.WriteLine($"[e2e] runtime loaded {clock.Elapsed.TotalSeconds:F1}s");
            clock.Restart();
            result = images.Count == 0 ? model.GenerateImage(Arg(args, "--prompt", ""), p) : model.EditImage(Arg(args, "--prompt", ""), images, p);
        }
        Console.WriteLine($"[e2e] {kind} request {clock.Elapsed.TotalSeconds:F2}s -> {result.Width}x{result.Height}, previews " +
                          string.Join(",", previews.Select(x => $"{x.Item1}@{x.Item2}px")));
        TensorSharp.Models.QwenImage.ImageIO.SavePng(Arg(args, "--out", "out.png"), result);
    }
}
