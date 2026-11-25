// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Qwen-Image-2.1 companion stages (text encoder, vision encoder, VAE encode/decode) on the
// pure-C# cpu backend vs ggml_cpu. Every stage dumps its F32 output so `compare` can measure
// parity between two runs; timings exclude model loading. The ggml_cpu runs need GgmlOps
// (native) next to the executable; the cpu runs never load it, so running them from a
// directory WITHOUT the native library proves the managed path makes no native call.
using System.Diagnostics;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.QwenImage21;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

static void Usage()
{
    Console.Error.WriteLine("conv [avx512|avx2|portable|all] [reps=3] [--scalar]       VAE conv layers: packed GEMM vs scalar");
    Console.Error.WriteLine("vae <vae.safetensors> <cpu|scalar|ggml_cpu> <decode|encode|roundtrip> <W> <H> <outPrefix> [--image png] [--reps N]");
    Console.Error.WriteLine("    [--latent file.f32] [--no-pool]   decode input (default: seeded noise); --no-pool: VAE feature pool off (A/B)");
    Console.Error.WriteLine("text <Qwen3VL.gguf> <cpu|ggml_cpu> <out.f32> [--prompt text] [--reps N] [--f64-linear]");
    Console.Error.WriteLine("     --f64-linear (cpu): every projection summed in double over exact weights - the parity reference");
    Console.Error.WriteLine("vision <mmproj.gguf> <cpu|ggml_cpu> <outPrefix> [--image png] [--size WxH] [--reps N]");
    Console.Error.WriteLine("compare <reference.f32> <actual.f32> [--image]              cosine, relL2, max error (PSNR for [0,1] images)");
    Console.Error.WriteLine("vattn [patches=4096] [reps=3] [avx512|avx2|portable|all]     vision attention kernel (16 heads x 72) alone");
}

static string Opt(string[] a, string name, string fallback)
{
    int i = Array.IndexOf(a, name);
    return i >= 0 && i + 1 < a.Length ? a[i + 1] : fallback;
}

static void WriteFloats(string path, float[] values)
{
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)));
    var bytes = new byte[checked(values.Length * sizeof(float))];
    Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
    File.WriteAllBytes(path, bytes);
}

static float[] ReadFloats(string path)
{
    byte[] bytes = File.ReadAllBytes(path);
    var values = new float[bytes.Length / sizeof(float)];
    Buffer.BlockCopy(bytes, 0, values, 0, values.Length * sizeof(float));
    return values;
}

static object Stats(float[] expected, float[] actual, bool image)
{
    if (expected.Length != actual.Length) throw new InvalidDataException($"length {expected.Length} != {actual.Length}");
    double dot = 0, ee = 0, aa = 0, sq = 0, maxErr = 0, maxRef = 0;
    int nonFinite = 0;
    for (int i = 0; i < expected.Length; i++)
    {
        double e = expected[i], v = actual[i];
        if (!double.IsFinite(e) || !double.IsFinite(v)) { nonFinite++; continue; }
        dot += e * v; ee += e * e; aa += v * v;
        double d = v - e;
        sq += d * d;
        maxErr = Math.Max(maxErr, Math.Abs(d));
        maxRef = Math.Max(maxRef, Math.Abs(e));
    }
    double cosine = dot / Math.Sqrt(Math.Max(ee * aa, 1e-300));
    double relL2 = Math.Sqrt(sq / Math.Max(ee, 1e-300));
    double mse = sq / expected.Length;
    // Identical outputs have no finite PSNR (and JSON has no infinity): report identical instead.
    double? psnr = image && mse > 0 ? 10 * Math.Log10(1.0 / mse) : null;
    return new { values = expected.Length, nonFinite, identical = sq == 0, cosine, relL2, maxAbsError = maxErr,
        normalizedMax = maxErr / Math.Max(maxRef, 1e-30), psnrDb = psnr };
}

static long PeakMb() { using var p = Process.GetCurrentProcess(); p.Refresh(); return p.PeakWorkingSet64 >> 20; }
// Peak commit charge (private bytes; Windows PeakPagefileUsage): what the process actually had to
// allocate, without the file-mapped weight pages that PeakWorkingSet64 also counts.
static long PeakPrivateMb() { using var p = Process.GetCurrentProcess(); p.Refresh(); return p.PeakPagedMemorySize64 >> 20; }

static RgbImage Gradient(int width, int height)
{
    var pixels = new float[checked(width * height * 3)];
    var alpha = new float[checked(width * height)];
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            int p = y * width + x;
            pixels[3 * p] = x / (float)(width - 1);
            pixels[3 * p + 1] = y / (float)(height - 1);
            pixels[3 * p + 2] = ((x / 16 + y / 16) % 2 == 0) ? .2f : .8f;
            alpha[p] = (x + y) / (float)(width + height - 2);
        }
    return new RgbImage(width, height, pixels, alpha);
}

static float[] Rgba(RgbImage image)
{
    int n = image.Width * image.Height;
    var rgba = new float[checked(n * 4)];
    for (int i = 0; i < n; ++i)
    {
        Array.Copy(image.Pixels, i * 3, rgba, i * 4, 3);
        rgba[i * 4 + 3] = image.Alpha?[i] ?? 1f;
    }
    return rgba;
}

if (args.Length == 0) { Usage(); return 2; }
string json;
switch (args[0])
{
    case "conv": return ConvBench.Run(args);
    case "vattn": return VisionAttentionBench.Run(args);
    case "te-linear": return QuantLinearCheck.Run(args);
    case "compare":
    {
        bool image = args.Contains("--image");
        json = JsonSerializer.Serialize(Stats(ReadFloats(args[1]), ReadFloats(args[2]), image));
        Console.WriteLine(json);
        return 0;
    }
    case "vae":
    {
        string backend = args[2], mode = args[3];
        int width = int.Parse(args[4]), height = int.Parse(args[5]);
        string prefix = Path.GetFullPath(args[6]);
        int reps = int.Parse(Opt(args, "--reps", "1"));
        string imagePath = Opt(args, "--image", null);
        bool ggml = backend == "ggml_cpu";
        VaeReferenceMath.UseScalarCpu = backend == "scalar";
        VaeReferenceMath.UseGpuConv = ggml;
        VaeReferenceMath.UseFusedGraph21 = false;   // QwenImage21Vae's default on GgmlCpu: per-conv device path
        if (args.Contains("--no-pool")) VaeFeaturePool.Enabled = false;
        if (ggml) GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cpu);
        using var file = new SafetensorsFile(args[1]);
        var weights = VaeWeights.Load(new QwenImage21VaeTensorStore(file));
        RgbImage source = imagePath != null ? ImageIO.Resize(ImageIO.Load(imagePath, preserveAlpha: false), width, height) : Gradient(width, height);
        var timings = new List<object>();
        Directory.CreateDirectory(Path.GetDirectoryName(prefix));
        for (int rep = 0; rep < reps; rep++)
        {
            double encodeSeconds = 0, decodeSeconds = 0;
            VaeLatent latent;
            var watch = Stopwatch.StartNew();
            if (mode == "decode")
            {
                string latentPath = Opt(args, "--latent", null);
                latent = latentPath != null
                    ? new VaeLatent(64, height / 16, width / 16, ReadFloats(latentPath))
                    : new VaeLatent(64, height / 16, width / 16, PhiloxNoise.Draw(checked(width / 16 * (height / 16) * 64), 42));
            }
            else
            {
                latent = VaeReferenceMath.Encode21(weights, source);
                encodeSeconds = watch.Elapsed.TotalSeconds;
                if (rep == 0) WriteFloats(prefix + ".latent.f32", latent.Data);
            }
            if (mode != "encode")
            {
                watch.Restart();
                RgbImage decoded = VaeReferenceMath.Decode21(weights, latent);
                decodeSeconds = watch.Elapsed.TotalSeconds;
                if (rep == 0)
                {
                    WriteFloats(prefix + ".rgba.f32", Rgba(decoded));
                    ImageIO.SavePng(prefix + ".png", decoded);
                    if (mode == "roundtrip")
                        Console.WriteLine(JsonSerializer.Serialize(new { roundtripVsInput = Stats(Rgba(source), Rgba(decoded), true) }));
                }
            }
            long workingSetMb, privateMb;
            using (var self = Process.GetCurrentProcess()) { self.Refresh(); workingSetMb = self.WorkingSet64 >> 20; privateMb = self.PrivateMemorySize64 >> 20; }
            timings.Add(new { rep, encodeSeconds, decodeSeconds, workingSetMb, privateMb });
            Console.WriteLine(JsonSerializer.Serialize(timings[^1]));
        }
        json = JsonSerializer.Serialize(new { scenario = "vae", backend, mode, width, height, isa = CpuPackedGemm.Isa.ToString(),
            timings, peakWorkingSetMb = PeakMb(), peakPrivateMb = PeakPrivateMb(), pool = VaeFeaturePool.Enabled, output = prefix });
        Console.WriteLine(json);
        if (ggml) { GgmlBasicOps.ReleaseReuseComputeBuffers(); GgmlBasicOps.ClearHostBufferCache(); GgmlBasicOps.Shutdown(); }
        return 0;
    }
    case "text":
    {
        var backend = args[2] == "ggml_cpu" ? BackendType.GgmlCpu : BackendType.Cpu;
        string output = Path.GetFullPath(args[3]);
        string prompt = Opt(args, "--prompt", "A small orange cat beside a blue ceramic vase, soft daylight, detailed photograph");
        int reps = int.Parse(Opt(args, "--reps", "1"));
        QwenImageTextEncoder.ReferenceF64Linear = args.Contains("--f64-linear");
        var load = Stopwatch.StartNew();
        using var encoder = new QwenImageTextEncoder(args[1], backend);
        double loadSeconds = load.Elapsed.TotalSeconds;
        int[] tokens = encoder.Tokenizer.Encode(QwenImage21Conditioner.SystemPrompt +
            "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n", addSpecial: false).ToArray();
        var seconds = new List<double>();
        float[] hidden = null;
        for (int rep = 0; rep < reps; rep++)
        {
            var watch = Stopwatch.StartNew();
            hidden = encoder.EncodeHidden(tokens);
            seconds.Add(watch.Elapsed.TotalSeconds);
            Console.WriteLine(JsonSerializer.Serialize(new { rep, seconds = seconds[^1] }));
        }
        WriteFloats(output, hidden);
        // The DiT conditioning, as QwenImage21Conditioner hands it over: the system prompt dropped.
        int drop = encoder.Tokenizer.Encode(QwenImage21Conditioner.SystemPrompt, addSpecial: false).Count;
        WriteFloats(Path.ChangeExtension(output, ".cond.f32"), hidden[(drop * encoder.HiddenSize)..]);
        json = JsonSerializer.Serialize(new { scenario = "text", backend = backend.ToString(), f64Linear = QwenImageTextEncoder.ReferenceF64Linear,
            tokens = tokens.Length, loadSeconds, seconds,
            nonFinite = hidden.Count(v => !float.IsFinite(v)), peakWorkingSetMb = PeakMb(), output });
        Console.WriteLine(json);
        if (backend == BackendType.GgmlCpu) { GgmlBasicOps.ReleaseReuseComputeBuffers(); GgmlBasicOps.ClearHostBufferCache(); GgmlBasicOps.Shutdown(); }
        return 0;
    }
    case "dit-cond":
    {
        // Downstream effect of conditioning differences: one DiT velocity prediction (ggml_cpu,
        // seeded noise latent) per conditioning file, each compared with the first.
        int size = int.Parse(args[2]);
        float t = float.Parse(Opt(args, "--t", "0.9"));
        var conds = args.Skip(3).Where(a => a.EndsWith(".f32")).ToArray();
        var family = new QwenImage21Family(BackendType.GgmlCpu, QwenImageVariant.Base);
        float[] latents = family.ToRows(new LatentTensor(new LatentShape(64, 1, size / 16, size / 16),
            PhiloxNoise.Draw(size / 16 * (size / 16) * 64, 42)));
        using var dit = new QwenImage21Denoiser(args[1], BackendType.GgmlCpu);
        float[] first = null;
        foreach (var path in conds)
        {
            float[] cond = ReadFloats(path);
            var watch = Stopwatch.StartNew();
            var text = new ConditioningSet(cond, cond.Length / 4096, 4096);
            var layout = QwenImage21Layout.Build(text.Rows, null, new[] { (size / 16, size / 16) }, 128);
            float[] velocity = (float[])dit.Predict(latents, text, Array.Empty<LatentRef>(), layout, t, 0, false, null).Clone();
            dit.FinishRequest();
            Console.WriteLine(JsonSerializer.Serialize(new { cond = Path.GetFileName(path), seconds = watch.Elapsed.TotalSeconds,
                vsFirst = first == null ? null : Stats(first, velocity, false) }));
            first ??= velocity;
        }
        GgmlBasicOps.ReleaseReuseComputeBuffers(); GgmlBasicOps.ClearHostBufferCache(); GgmlBasicOps.Shutdown();
        return 0;
    }
    case "vision":
    {
        bool ggml = args[2] == "ggml_cpu";
        string prefix = Path.GetFullPath(args[3]);
        string imagePath = Opt(args, "--image", null);
        string[] size = Opt(args, "--size", "256x256").Split('x');
        int width = int.Parse(size[0]), height = int.Parse(size[1]);
        int reps = int.Parse(Opt(args, "--reps", "1"));
        RgbImage image = imagePath != null ? ImageIO.Resize(ImageIO.Load(imagePath, preserveAlpha: true), width, height) : Gradient(width, height);
        // As QwenImage21Conditioner prepares a reference image.
        float[] pixels = image.ToPlanarChw();
        int hw = width * height;
        for (int j = 0; j < pixels.Length; j++)
        {
            float alpha = image.Alpha == null ? 1f : image.Alpha[j % hw];
            pixels[j] = 2f * (pixels[j] * alpha + 1f - alpha) - 1f;
        }
        GgmlContext context = ggml ? new GgmlContext(new[] { 0 }, GgmlBackendType.Cpu) : null;
        IAllocator allocator = ggml ? new GgmlAllocator(context, 0) : new CpuAllocator(BlasEnum.DotNet);
        var load = Stopwatch.StartNew();
        using var encoder = new Qwen35VisionEncoder(args[1], allocator, qwenImage21: true);
        double loadSeconds = load.Elapsed.TotalSeconds;
        var seconds = new List<double>();
        float[][] outputs = null;
        for (int rep = 0; rep < reps; rep++)
        {
            var watch = Stopwatch.StartNew();
            var tensors = encoder.EncodeWithDeepStack(pixels, height, width);
            outputs = tensors.Select(t => t.GetElementsAsFloat((int)t.ElementCount())).ToArray();
            seconds.Add(watch.Elapsed.TotalSeconds);
            foreach (var t in tensors) t.Dispose();
            Console.WriteLine(JsonSerializer.Serialize(new { rep, seconds = seconds[^1] }));
        }
        for (int i = 0; i < outputs.Length; i++)
            WriteFloats($"{prefix}.{(i == 0 ? "main" : $"deepstack{i - 1}")}.f32", outputs[i]);
        json = JsonSerializer.Serialize(new { scenario = "vision", backend = args[2], width, height, loadSeconds, seconds,
            outputs = outputs.Length, peakWorkingSetMb = PeakMb(), peakPrivateMb = PeakPrivateMb(), output = prefix });
        Console.WriteLine(json);
        if (ggml) { GgmlBasicOps.ReleaseReuseComputeBuffers(); GgmlBasicOps.ClearHostBufferCache(); context.ReleasePooledMemory(); GgmlBasicOps.Shutdown(); }
        return 0;
    }
    default:
        Usage();
        return 2;
}

/// <summary>
/// Real quantized text-encoder weights through the packed GEMM (QuantRowsPanelSource) and
/// through ManagedQuantizedOps (8-bit activations), each against a double-precision
/// dequantize-then-dot reference, for every projection of the given layers.
/// </summary>
static unsafe class QuantLinearCheck
{
    public static int Run(string[] args)
    {
        using var gguf = new GgufFile(args[1]);
        int rows = args.Length > 2 ? int.Parse(args[2]) : 37;
        int[] layers = args.Length > 3 ? args[3].Split(',').Select(int.Parse).ToArray() : new[] { 0, 17, 35 };
        string[] names = { "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down" };
        var rng = new Random(3);
        foreach (int layer in layers)
            foreach (string name in names)
            {
                var info = gguf.Tensors[$"blk.{layer}.{name}.weight"];
                gguf.TryGetTensorDataPointer(info, out IntPtr data);
                int inDim = (int)info.Shape[0], outDim = (int)info.Shape[1];
                var x = new float[(long)rows * inDim];
                for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
                // Reference: exact dequant (ManagedQuantizedOps scalar), double dots.
                var expected = new float[(long)rows * outDim];
                var w = new float[inDim];
                long rowBytes = ManagedQuantizedOps.RowSize((int)info.Type, inDim);
                for (int o = 0; o < outDim; o++)
                {
                    fixed (float* wp = w) ManagedQuantizedOps.DequantizeRowToFloat32((int)info.Type, data + (nint)(o * rowBytes), wp, inDim);
                    for (int r = 0; r < rows; r++)
                    {
                        double acc = 0;
                        for (int t = 0; t < inDim; t++) acc += (double)x[(long)r * inDim + t] * w[t];
                        expected[(long)r * outDim + o] = (float)acc;
                    }
                }
                var gemm = new float[expected.Length];
                var generic = new float[expected.Length];
                double gemmMs, genericMs;
                fixed (float* xp = x, gp = gemm, mp = generic)
                {
                    var watch = Stopwatch.StartNew();
                    var a = CpuPackedGemm.PackA(xp, rows, inDim, inDim, 1, CpuPackedGemm.Isa);
                    CpuPackedGemm.Gemm(a, new QuantRowsPanelSource(data, (int)info.Type, inDim, outDim), outDim, gp, outDim);
                    gemmMs = watch.Elapsed.TotalMilliseconds;
                    watch.Restart();
                    ManagedQuantizedOps.AddmmQuantizedToFloat32((int)info.Type, data, inDim, outDim, xp, inDim, rows, mp, outDim);
                    genericMs = watch.Elapsed.TotalMilliseconds;
                }
                Console.WriteLine($"blk.{layer}.{name,-12} {info.Type,-5} [{outDim}x{inDim}] gemm {gemmMs,7:F1} ms relL2={RelL2(expected, gemm):E2} | " +
                    $"managed-q8 {genericMs,7:F1} ms relL2={RelL2(expected, generic):E2}");
            }
        return 0;
    }

    private static double RelL2(float[] e, float[] a)
    {
        double s = 0, r = 0;
        for (int i = 0; i < e.Length; i++) { double d = a[i] - e[i]; s += d * d; r += (double)e[i] * e[i]; }
        return Math.Sqrt(s / r);
    }
}

/// <summary>The pure-C# vision attention (CpuFullAttention) alone at the Qwen3-VL tower's shape
/// (16 heads x 72, one layer), per ISA, on the shared pool the encoder uses.</summary>
static unsafe class VisionAttentionBench
{
    public static int Run(string[] args)
    {
        int n = args.Length > 1 ? int.Parse(args[1]) : 4096;
        int reps = args.Length > 2 ? int.Parse(args[2]) : 3;
        string which = args.Length > 3 ? args[3] : "native";
        var isas = which switch
        {
            "all" => new[] { CpuGemmIsa.Avx512, CpuGemmIsa.Avx2, CpuGemmIsa.Portable },
            "avx512" => new[] { CpuGemmIsa.Avx512 },
            "avx2" => new[] { CpuGemmIsa.Avx2 },
            "portable" => new[] { CpuGemmIsa.Portable },
            _ => new[] { CpuPackedGemm.Isa },
        };
        const int heads = 16, dim = 72;
        var rng = new Random(5);
        float[] q = new float[(long)n * heads * dim], k = new float[q.Length], v = new float[q.Length], o = new float[q.Length];
        for (int i = 0; i < q.Length; i++) { q[i] = (float)(rng.NextDouble() * 4 - 2); k[i] = (float)(rng.NextDouble() * 4 - 2); v[i] = (float)(rng.NextDouble() * 2 - 1); }
        double flops = 4.0 * heads * n * (double)n * dim;
        var workspace = new CpuAttentionWorkspace();
        Console.WriteLine($"patches={n} heads={heads} dim={dim} threads={CpuWorkerPool.Shared.ThreadCount} ({flops / 1e9:F1} GFLOP per call)");
        foreach (var isa in isas)
        {
            if (!CpuPackedGemm.IsaSupported(isa)) continue;
            CpuPackedGemm.Isa = isa;
            double best = double.MaxValue;
            fixed (float* qp = q, kp = k, vp = v, op = o)
                for (int r = 0; r <= reps; r++)   // rep 0 warms the JIT and the workspace
                {
                    var watch = Stopwatch.StartNew();
                    CpuFullAttention.Run(qp, kp, vp, op, n, heads, dim, 1f / MathF.Sqrt(dim), workspace);
                    if (r > 0) best = Math.Min(best, watch.Elapsed.TotalMilliseconds);
                }
            Console.WriteLine($"{isa,-8} {best,8:F1} ms {flops / best / 1e6,7:F1} GF/s");
        }
        CpuPackedGemm.Isa = CpuPackedGemm.DetectIsa();
        return 0;
    }
}

/// <summary>Representative VAE convolutions (256x256 decode geometry plus encoder/attention
/// shapes): packed GEMM per ISA vs the original scalar direct convolution.</summary>
static class ConvBench
{
    private sealed record Layer(string Name, int IC, int OC, int H, int W, int K, int Stride, int Pad, bool Up = false);

    private static readonly Layer[] Layers =
    {
        new("mid 1152->1152 3x3 @16", 1152, 1152, 16, 16, 3, 1, 1),
        new("up0 1152->1152 3x3 @32 (from 16 via up2x)", 1152, 1152, 16, 16, 3, 1, 1, true),
        new("up1 1152->1152 3x3 @32", 1152, 1152, 32, 32, 3, 1, 1),
        new("up2 576->576 3x3 @64", 576, 576, 64, 64, 3, 1, 1),
        new("up3 288->288 3x3 @128", 288, 288, 128, 128, 3, 1, 1),
        new("up4 144->144 3x3 @256", 144, 144, 256, 256, 3, 1, 1),
        new("up4 288->144 1x1 @256 (shortcut)", 288, 144, 256, 256, 1, 1, 0),
        new("head 144->4 3x3 @256", 144, 4, 256, 256, 3, 1, 1),
        new("dec.conv1 64->1152 3x3 @16", 64, 1152, 16, 16, 3, 1, 1),
        new("enc.down 96->96 3x3 s2 @256", 96, 96, 256, 256, 3, 2, 0),
    };

    public static int Run(string[] args)
    {
        string which = args.Length > 1 && !args[1].StartsWith("--") ? args[1] : "native";
        int reps = args.Length > 2 && int.TryParse(args[2], out int r) ? r : 3;
        bool scalar = args.Contains("--scalar");
        var isas = which switch
        {
            "all" => new[] { CpuGemmIsa.Avx512, CpuGemmIsa.Avx2, CpuGemmIsa.Portable },
            "avx512" => new[] { CpuGemmIsa.Avx512 },
            "avx2" => new[] { CpuGemmIsa.Avx2 },
            "portable" => new[] { CpuGemmIsa.Portable },
            _ => new[] { CpuPackedGemm.Isa },
        };
        VaeReferenceMath.UseGpuConv = false;   // the managed decode's configuration (wide pool)
        Console.WriteLine($"threads={VaeReferenceMath.CpuPool.ThreadCount} detected={CpuPackedGemm.DetectIsa()}");
        foreach (var layer in Layers)
        {
            var rng = new Random(7);
            var x = new Feature(layer.IC, layer.H, layer.W);
            for (int i = 0; i < x.D.Length; i++) x.D[i] = (float)(rng.NextDouble() * 2 - 1);
            var w = new float[(long)layer.OC * layer.IC * layer.K * layer.K];
            float wScale = 1f / MathF.Sqrt(layer.IC * layer.K * layer.K);
            for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1) * wScale;
            var bias = new float[layer.OC];
            for (int i = 0; i < bias.Length; i++) bias[i] = (float)(rng.NextDouble() - .5);
            int padB = layer.Stride == 2 ? 1 : layer.Pad, padR = padB;
            int h = layer.Up ? 2 * layer.H : layer.H, wd = layer.Up ? 2 * layer.W : layer.W;
            int ho = (h + layer.Pad + padB - layer.K) / layer.Stride + 1, wo = (wd + layer.Pad + padR - layer.K) / layer.Stride + 1;
            double flops = 2.0 * layer.OC * layer.IC * layer.K * layer.K * ho * wo;
            Feature reference = null;
            double scalarMs = double.NaN;
            if (scalar)
            {
                var src = layer.Up ? Upsample(x) : x;
                var sw = Stopwatch.StartNew();
                reference = VaeReferenceMath.Conv2dScalar(src, w, layer.OC, layer.IC, layer.K, layer.K, bias,
                    layer.Stride, layer.Stride, layer.Pad, padB, layer.Pad, padR);
                scalarMs = sw.Elapsed.TotalMilliseconds;
            }
            foreach (var isa in isas)
            {
                if (!CpuPackedGemm.IsaSupported(isa)) continue;
                CpuPackedGemm.Isa = isa;
                var packWatch = Stopwatch.StartNew();
                var packed = VaeReferenceMath.PackConvWeight(w, layer.OC, layer.IC * layer.K * layer.K);
                double packMs = packWatch.Elapsed.TotalMilliseconds;
                // One untimed run so every helper is past tier-0 JIT before timing.
                Feature y = VaeReferenceMath.Conv2dCpu(x, packed, bias, layer.OC, layer.K, layer.K, layer.Stride, layer.Stride,
                    layer.Pad, padB, layer.Pad, padR, layer.Up);
                double best = double.MaxValue;
                for (int i = 0; i < reps; i++)
                {
                    var sw = Stopwatch.StartNew();
                    y = VaeReferenceMath.Conv2dCpu(x, packed, bias, layer.OC, layer.K, layer.K, layer.Stride, layer.Stride,
                        layer.Pad, padB, layer.Pad, padR, layer.Up);
                    best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
                }
                string parity = "";
                if (reference != null)
                {
                    double sq = 0, ref2 = 0, maxErr = 0, maxRef = 0;
                    for (int i = 0; i < y.D.Length; i++)
                    {
                        double d = y.D[i] - reference.D[i];
                        sq += d * d; ref2 += (double)reference.D[i] * reference.D[i];
                        maxErr = Math.Max(maxErr, Math.Abs(d)); maxRef = Math.Max(maxRef, Math.Abs(reference.D[i]));
                    }
                    parity = $" relL2={Math.Sqrt(sq / ref2):E2} maxErr/maxRef={maxErr / maxRef:E2} scalar={scalarMs:F0}ms ({flops / scalarMs / 1e6:F1} GF/s)";
                }
                Console.WriteLine($"{layer.Name,-44} {isa,-8} {best,8:F1} ms {flops / best / 1e6,7:F1} GF/s pack={packMs:F0}ms{parity}");
            }
            CpuPackedGemm.Isa = CpuPackedGemm.DetectIsa();
        }
        return 0;
    }

    private static Feature Upsample(Feature x)
    {
        var y = new Feature(x.C, 2 * x.H, 2 * x.W);
        for (int c = 0; c < x.C; c++)
            for (int oy = 0; oy < y.H; oy++)
                for (int ox = 0; ox < y.W; ox++)
                    y.D[(c * y.H + oy) * y.W + ox] = x.D[(c * x.H + oy / 2) * x.W + ox / 2];
        return y;
    }
}
