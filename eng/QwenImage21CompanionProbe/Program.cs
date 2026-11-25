// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.QwenImage21;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

static void WriteFloats(string path, float[] values)
{
    if (values.Any(x => !float.IsFinite(x))) throw new InvalidDataException("Non-finite output.");
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)));
    byte[] bytes = new byte[checked(values.Length * sizeof(float))];
    Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
    File.WriteAllBytes(path, bytes);
}

// Isolate the actual vision tower, including all three DeepStack projectors.
// Its inputs match QwenImage21References/Conditioner, including alpha compositing.
if (args.Length >= 4 && args[0] == "vision")
{
    if (args.Length > 8) throw new ArgumentException("Too many vision arguments.");
    var device = Enum.Parse<GgmlBackendType>(args[3], true);
    int requestedWidth = args.Length > 5 ? int.Parse(args[5]) : 512;
    int requestedHeight = args.Length > 6 ? int.Parse(args[6]) : requestedWidth;
    int iterations = args.Length > 7 ? int.Parse(args[7]) : 1;
    if (requestedWidth <= 0 || requestedHeight <= 0 || requestedWidth % 32 != 0 || requestedHeight % 32 != 0 || iterations < 1)
        throw new ArgumentException("Positive dimensions divisible by 32 and at least one iteration required.");
    string imagePath = args.Length > 4 && args[4] != "-" ? Path.GetFullPath(args[4]) : null;
    string prefix = Path.GetFullPath(args[2]);
    Directory.CreateDirectory(Path.GetDirectoryName(prefix));
    var watch = Stopwatch.StartNew();
    RgbImage image;
    if (imagePath != null) image = ImageIO.Load(imagePath, preserveAlpha: true);
    else
    {
        var rgb = new float[checked(requestedWidth * requestedHeight * 3)];
        var alpha = new float[checked(requestedWidth * requestedHeight)];
        for (int y = 0; y < requestedHeight; ++y) for (int x = 0; x < requestedWidth; ++x)
        {
            int p = y * requestedWidth + x;
            rgb[3 * p] = x / (float)(requestedWidth - 1);
            rgb[3 * p + 1] = y / (float)(requestedHeight - 1);
            rgb[3 * p + 2] = ((x / 16 + y / 16) % 2 == 0) ? .2f : .8f;
            alpha[p] = (x + y) / (float)(requestedWidth + requestedHeight - 2);
        }
        image = new RgbImage(requestedWidth, requestedHeight, rgb, alpha);
    }
    int sourceWidth = image.Width, sourceHeight = image.Height;
    var (width, height) = QwenImage21Canvas.ReferenceDimensions(image, requestedWidth, requestedHeight);
    image = ImageIO.Resize(image, width, height);
    float[] pixels = image.ToPlanarChw();
    int hw = checked(width * height);
    for (int i = 0; i < pixels.Length; ++i)
    {
        float alpha = image.Alpha?[i % hw] ?? 1f;
        pixels[i] = 2f * (pixels[i] * alpha + 1f - alpha) - 1f;
    }
    if (pixels.Any(x => !float.IsFinite(x))) throw new InvalidDataException("Non-finite vision input.");
    double preprocessingSeconds = watch.Elapsed.TotalSeconds;
    byte[] inputBytes = new byte[checked(pixels.Length * sizeof(float))];
    Buffer.BlockCopy(pixels, 0, inputBytes, 0, inputBytes.Length);
    string inputSha256 = Convert.ToHexString(SHA256.HashData(inputBytes)).ToLowerInvariant();
    GgmlContext context = null;
    try
    {
        context = new GgmlContext(new[] { 0 }, device);
        var allocator = new GgmlAllocator(context, 0);
        watch.Restart();
        using var encoder = new Qwen35VisionEncoder(args[1], allocator, qwenImage21: true);
        double loadSeconds = watch.Elapsed.TotalSeconds;
        int tokenCount = checked(width / 32 * (height / 32));
        if (encoder.ProjectionDim != 4096 || encoder.PatchSize != 16 || encoder.SpatialMergeSize != 2)
            throw new InvalidDataException("Unexpected Qwen3-VL-8B vision projector geometry.");
        var timings = new List<double>();
        float[][] outputs = null;
        long[][] shapes = null;
        for (int iteration = 0; iteration < iterations; ++iteration)
        {
            watch.Restart();
            var tensors = encoder.EncodeWithDeepStack(pixels, height, width);
            try
            {
                if (tensors.Length != 4) throw new InvalidDataException("Expected main vision embedding and exactly three DeepStack outputs.");
                outputs = new float[tensors.Length][];
                shapes = new long[tensors.Length][];
                for (int i = 0; i < tensors.Length; ++i)
                {
                    if (tensors[i].Sizes.Length != 2 || tensors[i].Sizes[0] != tokenCount || tensors[i].Sizes[1] != 4096 ||
                        tensors[i].ElementCount() != checked(tokenCount * 4096))
                        throw new InvalidDataException($"Vision output {i} has unexpected shape.");
                    shapes[i] = tensors[i].Sizes.ToArray();
                    outputs[i] = tensors[i].GetElementsAsFloat(checked(tokenCount * 4096));
                    if (outputs[i].Any(x => !float.IsFinite(x))) throw new InvalidDataException($"Non-finite vision output {i}.");
                }
                timings.Add(watch.Elapsed.TotalSeconds); // includes synchronous host readback
            }
            finally { foreach (var tensor in tensors) tensor.Dispose(); }
            Console.WriteLine(JsonSerializer.Serialize(new { scenario = "vision-iteration", iteration, seconds = timings[^1] }));
        }
        var files = new List<object>();
        for (int i = 0; i < outputs.Length; ++i)
        {
            string name = i == 0 ? "main" : $"deepstack{i - 1}";
            string path = prefix + "." + name + ".f32";
            WriteFloats(path, outputs[i]);
            files.Add(new { name, path, shape = shapes[i], values = outputs[i].Length, finite = true });
        }
        object Stamp(string path)
        {
            if (!File.Exists(path)) return new { path, exists = false };
            using var stream = File.OpenRead(path);
            return new { path, exists = true, bytes = stream.Length,
                sha256 = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant() };
        }
        string nativeName = OperatingSystem.IsWindows() ? "GgmlOps.dll" : OperatingSystem.IsMacOS() ? "libGgmlOps.dylib" : "libGgmlOps.so";
        string json = JsonSerializer.Serialize(new {
            scenario = "real-weight-vision-deepstack", model = Path.GetFullPath(args[1]),
            image = imagePath ?? "deterministic RGBA gradient/checkerboard", sourceWidth, sourceHeight,
            requestedWidth, requestedHeight, width, height, tokenCount, channels = 4096, deepStackCount = 3,
            inputSha256, preprocessingSeconds, loadSeconds, iterations, seconds = timings,
            timingScope = "EncodeWithDeepStack plus readback/finite checks; excludes model loading and output-file writes",
            backend = device.ToString(), fusedRequested = Environment.GetEnvironmentVariable("TS_QWEN21_VISION_FUSED") != "0",
            outputs = files, binaries = new {
                native = Stamp(Path.Combine(AppContext.BaseDirectory, nativeName)),
                models = Stamp(typeof(Qwen35VisionEncoder).Assembly.Location),
                probe = Stamp(System.Reflection.Assembly.GetExecutingAssembly().Location) }
        });
        File.WriteAllText(prefix + ".json", json);
        Console.WriteLine(json);
    }
    finally
    {
        GgmlBasicOps.ReleaseReuseComputeBuffers();
        GgmlBasicOps.ClearHostBufferCache();
        context?.ReleasePooledMemory();
        GgmlBasicOps.Shutdown();
    }
    return 0;
}

// Isolate real VAE decode from transformer/text timing. Random normalized
// latents are a performance fixture, not a meaningful image-quality example.
if (args.Length >= 4 && (args[0] == "vae" || args[0] == "vae-encode"))
{
    var backendName = Enum.Parse<GgmlBackendType>(args[3], true);
    int width = args.Length > 4 ? int.Parse(args[4]) : 256;
    int height = args.Length > 5 ? int.Parse(args[5]) : width;
    if (width <= 0 || height <= 0 || width % 32 != 0 || height % 32 != 0)
        throw new ArgumentException("Positive dimensions divisible by 32 required.");
    GgmlBasicOps.EnsureBackendAvailable(backendName);
    // Mirror QwenImage21Vae: fused by default on CUDA and Metal, never on Vulkan.
    string fused = Environment.GetEnvironmentVariable("TS_QWEN21_VAE_FUSED");
    VaeReferenceMath.UseFusedGraph21 = backendName != GgmlBackendType.Vulkan &&
        (fused == "1" || backendName is GgmlBackendType.Cuda or GgmlBackendType.Metal);
    using var file = new SafetensorsFile(args[1]);
    // As the model opens it: the published file uses diffusers names (decoder.conv_in, ...).
    var source = new QwenImage21VaeTensorStore(file);
    QwenImage21CompanionValidation.ValidateVae(source);
    var weights = VaeWeights.Load(source);
    var latent = new VaeLatent(64, height / 16, width / 16,
        PhiloxNoise.Draw(checked(width / 16 * (height / 16) * 64), 42));
    if (latent.Data.Any(x => !float.IsFinite(x))) throw new InvalidDataException("Non-finite input noise.");
    var pixels = new float[checked(width * height * 3)];
    var alpha = new float[checked(width * height)];
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
    {
        int p = y * width + x;
        pixels[3 * p] = x / (float)(width - 1);
        pixels[3 * p + 1] = y / (float)(height - 1);
        pixels[3 * p + 2] = ((x / 16 + y / 16) % 2 == 0) ? .2f : .8f;
        alpha[p] = (x + y) / (float)(width + height - 2);
    }
    var testImage = new RgbImage(width, height, pixels, alpha);
    try
    {
        string destination = Path.GetFullPath(args[2]);
        Directory.CreateDirectory(Path.GetDirectoryName(destination));
        var watch = Stopwatch.StartNew();
        RgbImage decoded = null;
        VaeLatent encoded = null;
        if (args[0] == "vae-encode") encoded = VaeReferenceMath.Encode21(weights, testImage);
        else decoded = VaeReferenceMath.Decode21(weights, latent);
        double seconds = watch.Elapsed.TotalSeconds;
        if (encoded != null) WriteFloats(destination, encoded.Data);
        else
        {
            // Keep F32 pixels/alpha as well as the rounded PNG for numerical checks.
            var rgba = new float[checked(width * height * 4)];
            for (int i = 0; i < width * height; ++i)
            {
                Array.Copy(decoded.Pixels, i * 3, rgba, i * 4, 3);
                rgba[i * 4 + 3] = decoded.Alpha?[i] ?? 1f;
            }
            WriteFloats(destination + ".rgba.f32", rgba);
            ImageIO.SavePng(destination, decoded);
        }
        Console.WriteLine(JsonSerializer.Serialize(new { scenario = encoded != null ? "real-weight-vae-gradient-encode" : "real-weight-vae-random-latents", width, height,
            seconds, backend = backendName.ToString(), output = destination }));
    }
    finally
    {
        weights.FusedGraph?.Dispose();
        GgmlBasicOps.ReleaseReuseComputeBuffers();
        GgmlBasicOps.ClearHostBufferCache();
        GgmlBasicOps.Shutdown();
    }
    return 0;
}

// Real-weight DiT probe: conditioning must come from the text command below.
// Repeated predictions isolate transformer timing from encoder/VAE startup.
if (args.Length >= 5 && args[0] == "dit")
{
    var device = Enum.Parse<BackendType>(args[3], true);
    int width = args.Length > 5 ? int.Parse(args[5]) : 256;
    int height = args.Length > 6 ? int.Parse(args[6]) : width;
    int iterations = args.Length > 7 ? int.Parse(args[7]) : 6;
    if (width <= 0 || height <= 0 || width % 32 != 0 || height % 32 != 0 || iterations < 1)
        throw new ArgumentException("Positive dimensions divisible by 32 and at least one iteration required.");
    byte[] conditioningBytes = File.ReadAllBytes(args[4]);
    if (conditioningBytes.Length == 0 || conditioningBytes.Length % (4096 * sizeof(float)) != 0)
        throw new InvalidDataException("Expected token-major 4096-channel F32 conditioning.");
    float[] conditioning = new float[conditioningBytes.Length / sizeof(float)];
    Buffer.BlockCopy(conditioningBytes, 0, conditioning, 0, conditioningBytes.Length);
    if (conditioning.Any(x => !float.IsFinite(x))) throw new InvalidDataException("Non-finite conditioning.");
    int textTokens = conditioning.Length / 4096;
    var family = new QwenImage21Family(device, QwenImageVariant.Base);
    float[] latents = family.ToRows(new LatentTensor(new LatentShape(64, 1, height / 16, width / 16),
        PhiloxNoise.Draw(width / 16 * (height / 16) * 64, 42)));
    var timings = new List<double>();
    float[] velocity = null;
    try
    {
        using var dit = new QwenImage21Denoiser(args[1], device);
        var text = new ConditioningSet(conditioning, textTokens, 4096);
        var layout = QwenImage21Layout.Build(textTokens, null, new[] { (height / 16, width / 16) }, 128);
        for (int i = 0; i < iterations; ++i)
        {
            var watch = Stopwatch.StartNew();
            // Same latent at a changing timestep: the first step extracts the text prefix, the later ones read it; this is a
            // prediction benchmark, not an image sampler.
            velocity = (float[])dit.Predict(latents, text, Array.Empty<LatentRef>(), layout, 1f - i / (float)iterations, i, false, null).Clone();
            timings.Add(watch.Elapsed.TotalSeconds);
            Console.WriteLine(JsonSerializer.Serialize(new { iteration = i, seconds = timings[^1], mode = dit.LastMode.ToString() }));
        }
        dit.FinishRequest();
        string destination = Path.GetFullPath(args[2]);
        Directory.CreateDirectory(Path.GetDirectoryName(destination));
        WriteFloats(destination, velocity);
        Console.WriteLine(JsonSerializer.Serialize(new { scenario = "real-weight-dit", width, height, textTokens,
            iterations, seconds = timings, backend = device.ToString(), output = destination }));
    }
    finally
    {
        GgmlBasicOps.ReleaseReuseComputeBuffers();
        GgmlBasicOps.ClearHostBufferCache();
        GgmlBasicOps.Shutdown();
    }
    return 0;
}

// Real-weight equivalence of the prefix KV cache and tensor parallelism. One request
// layout (text, one reference image, target) at two denoising steps is predicted by
// the whole-sequence graph, then by the cache (step 1 stores the prefix, step 2 reads
// it), then by the transformer sharded over a loopback group of ranks on this one
// device. Conditioning and latents are seeded noise: this checks the kernels on the
// real quantized weights, not image quality.
if (args.Length >= 4 && args[0] == "dit-parity")
{
    var device = Enum.Parse<BackendType>(args[3], true);
    int width = args.Length > 4 ? int.Parse(args[4]) : 512;
    int height = args.Length > 5 ? int.Parse(args[5]) : width;
    int reference = args.Length > 6 ? int.Parse(args[6]) : 256;
    int ranks = args.Length > 7 ? int.Parse(args[7]) : 2;
    if (width % 32 != 0 || height % 32 != 0 || reference % 32 != 0 || reference <= 0 || ranks < 0)
        throw new ArgumentException("Dimensions must be positive multiples of 32.");
    int h = height / 16, w = width / 16, rh = reference / 16, rw = reference / 16;
    var rng = new Random(1234);
    float[] Gaussian(int count, float scale)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = scale * MathF.Sqrt(-2f * MathF.Log(1f - rng.NextSingle())) * MathF.Cos(2f * MathF.PI * rng.NextSingle());
        return values;
    }
    // Text, the reference's vision slots (one per 2x2 latent tokens), more text.
    int before = 24, slots = rh * rw / 4, after = 40, textTokens = before + slots + after;
    var imageSlots = new int[textTokens];
    for (int i = before; i < before + slots; i++) imageSlots[i] = 1;
    float[] text = Gaussian(textTokens * 4096, 1f);
    float[][] references = { Gaussian(rh * rw * 64, 1f) };
    float[] stepA = Gaussian(h * w * 64, 1f), stepB = Gaussian(h * w * 64, 1f);
    var report = new List<object>();
    var failures = new List<string>();
    var conditioning = new ConditioningSet(text, textTokens, 4096, imageSlots);
    LatentRef[] latentRefs = { new LatentRef(0, new LatentShape(64, 1, rh, rw), references[0]) };
    var layout = QwenImage21Layout.Build(textTokens, imageSlots, new[] { (rh, rw), (h, w) }, 128);
    float[] Predict(QwenImage21Denoiser dit, float[] latents, float t, int step, string label)
    {
        var watch = Stopwatch.StartNew();
        float[] result = (float[])dit.Predict(latents, conditioning, latentRefs, layout, t, step, false, null).Clone();
        string dumps = Path.GetFullPath(args[2]) + ".d";
        Directory.CreateDirectory(dumps);   // parity-dumps: each prediction's velocity, for comparing two builds byte for byte
        WriteFloats(Path.Combine(dumps, label.Replace(' ', '-') + ".f32"), result);
        Console.WriteLine(JsonSerializer.Serialize(new { label, seconds = watch.Elapsed.TotalSeconds, path = dit.LastMode.ToString() }));
        return result;
    }
    void Compare(string label, float[] expected, float[] actual, double tolerance)
    {
        double maxError = 0, scale = 0, square = 0, referenceSquare = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double delta = actual[i] - expected[i];
            maxError = Math.Max(maxError, Math.Abs(delta));
            scale = Math.Max(scale, Math.Abs(expected[i]));
            square += delta * delta;
            referenceSquare += (double)expected[i] * expected[i];
        }
        bool identical = expected.AsSpan().SequenceEqual(actual);
        double relativeL2 = Math.Sqrt(square / Math.Max(referenceSquare, 1e-30));
        bool passed = relativeL2 <= tolerance;
        report.Add(new { label, identical, maxError, normalizedMax = maxError / Math.Max(scale, 1e-12), relativeL2, tolerance, passed });
        Console.WriteLine(JsonSerializer.Serialize(report[^1]));
        if (!passed) failures.Add($"{label}: relative L2 {relativeL2} exceeds {tolerance}.");
    }
    try
    {
        // The whole-sequence graph: a cache that cannot fit (no free memory) is declined, so every step recomputes the prefix.
        float[] fullA, fullB;
        using (var full = new QwenImage21Denoiser(new QwenImage21Weights(args[1]), device, null, null) { FreeMemory = () => 0 })
        {
            fullA = Predict(full, stepA, 0.95f, 0, "full step 1");
            fullB = Predict(full, stepB, 0.60f, 1, "full step 2");
        }
        // On ggml-cpu a cached step is the whole sequence's bytes. A GPU backend picks its kernels, and their work splits, by
        // the rows a product computes, and a cached step computes fewer: the retired engine's two modes were 2.3e-3 to 4.6e-3
        // apart on ggml-cuda (RTX 4000 Ada, 512 x 512), so the cached step is held to 1% there.
        double cachedTolerance = device == BackendType.GgmlCpu ? 0 : 0.01;
        using (var dit = new QwenImage21Denoiser(new QwenImage21Weights(args[1]), device, null, null) { FreeMemory = () => -1 })
        {
            Compare("cache extract (step 1)", fullA, Predict(dit, stepA, 0.95f, 0, "cache step 1"), 0);
            Compare("cache cached (step 2)", fullB, Predict(dit, stepB, 0.60f, 1, "cache step 2"), cachedTolerance);
            dit.FinishRequest();
        }
        GgmlBasicOps.ReleaseReuseComputeBuffers();
        GgmlBasicOps.ClearHostBufferCache();
        if (ranks > 1)
        {
            GgmlBasicOps.TensorParallelInitLoopback(device == BackendType.GgmlCpu ? GgmlBackendType.Cpu : GgmlBackendType.Metal, ranks);
            using var group = new LoopbackGroup(ranks);
            // Reordered partial sums round differently on GPU matmuls; a missing or doubled reduction is off by O(1).
            using (var shardedFull = new QwenImage21Denoiser(new QwenImage21Weights(args[1]), device, null, null, group) { FreeMemory = () => 0 })
                Compare($"tp{ranks} full (step 1)", fullA, Predict(shardedFull, stepA, 0.95f, 0, $"tp{ranks} full step 1"), 0.01);
            using var sharded = new QwenImage21Denoiser(new QwenImage21Weights(args[1]), device, null, null, group) { FreeMemory = () => -1 };
            Compare($"tp{ranks} cache extract (step 1)", fullA, Predict(sharded, stepA, 0.95f, 0, $"tp{ranks} cache step 1"), 0.01);
            Compare($"tp{ranks} cache cached (step 2)", fullB, Predict(sharded, stepB, 0.60f, 1, $"tp{ranks} cache step 2"), 0.01);
            sharded.FinishRequest();
        }
        string destination = Path.GetFullPath(args[2]);
        Directory.CreateDirectory(Path.GetDirectoryName(destination));
        File.WriteAllText(destination, JsonSerializer.Serialize(new { scenario = "real-weight-dit-parity", width, height, reference,
            textTokens, prefixTokens = textTokens - slots + rh * rw, ranks, backend = device.ToString(), results = report },
            new JsonSerializerOptions { WriteIndented = true }));
        if (failures.Count > 0) throw new InvalidDataException(string.Join(" ", failures));
    }
    finally
    {
        GgmlBasicOps.ReleaseReuseComputeBuffers();
        GgmlBasicOps.ClearHostBufferCache();
        GgmlBasicOps.Shutdown();
    }
    return 0;
}

// Run twice, setting TS_QWEN_TE_FUSED=0/1 before process startup, then compare.
// No model/device scenario is considered passing unless this tool executes it.
if (args.Length == 3 && args[0] == "compare")
{
    float[] Read(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length % sizeof(float) != 0) throw new InvalidDataException("Expected raw F32.");
        var values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }
    var expected = Read(args[1]);
    var actual = Read(args[2]);
    if (expected.Length != actual.Length) throw new InvalidDataException("Output lengths differ.");
    double square = 0, referenceSquare = 0, maxError = 0;
    for (int i = 0; i < expected.Length; i++)
    {
        if (!float.IsFinite(expected[i]) || !float.IsFinite(actual[i]))
            throw new InvalidDataException("Non-finite hidden state.");
        double delta = actual[i] - expected[i];
        square += delta * delta;
        referenceSquare += (double)expected[i] * expected[i];
        maxError = Math.Max(maxError, Math.Abs(delta));
    }
    double relativeL2 = Math.Sqrt(square / Math.Max(referenceSquare, 1e-30));
    // Per-op vs fused reduction order differs; quantized execution is also
    // backend dependent. Threshold measures the entire conditioning tensor.
    bool passed = relativeL2 < 0.01;
    Console.WriteLine(JsonSerializer.Serialize(new { passed, values = expected.Length, relativeL2, maxError }));
    return passed ? 0 : 1;
}

if (args.Length < 3 || args[0] != "text")
{
    Console.Error.WriteLine("text <Qwen3VL-8B.gguf> <output.f32> [GgmlCpu|GgmlMetal|GgmlCuda] [prompt]");
    Console.Error.WriteLine("dit <QwenImage2.1.gguf> <output.f32> <GgmlCpu|GgmlMetal|GgmlCuda> <conditioning.f32> [width=256] [height=width] [iterations=6]");
    Console.Error.WriteLine("vae <QwenImage2.1Vae.safetensors> <output.png> <Cpu|Metal|Cuda> [width=256] [height=width]");
    Console.Error.WriteLine("vae-encode <QwenImage2.1Vae.safetensors> <output.f32> <Cpu|Metal|Cuda> [width=256] [height=width]");
    Console.Error.WriteLine("vision <Qwen3VL-mmproj.gguf> <output-prefix> <Cpu|Metal|Cuda> [input.png|-] [width=512] [height=width] [iterations=1]");
    Console.Error.WriteLine("dit-parity <QwenImage2.1.gguf> <report.json> <GgmlCpu|GgmlMetal> [width=512] [height=width] [reference=256] [ranks=2]");
    Console.Error.WriteLine("compare <reference.f32> <actual.f32>");
    return 2;
}
BackendType backend = args.Length > 3 ? Enum.Parse<BackendType>(args[3], true) : BackendType.GgmlCpu;
string prompt = args.Length > 4 ? args[4] : "A red cube on a white table.";
string output = Path.GetFullPath(args[2]);
Directory.CreateDirectory(Path.GetDirectoryName(output));
var timer = Stopwatch.StartNew();
using (var encoder = new QwenImageTextEncoder(args[1], backend))
{
    int[] tokens = encoder.Tokenizer.Encode(QwenImage21Conditioner.SystemPrompt +
        "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n", addSpecial: false).ToArray();
    timer.Restart();
    float[] hidden = encoder.EncodeHidden(tokens);
    if (hidden.Any(x => !float.IsFinite(x))) throw new InvalidDataException("Non-finite hidden state.");
    double seconds = timer.Elapsed.TotalSeconds;
    byte[] bytes = new byte[hidden.Length * sizeof(float)];
    Buffer.BlockCopy(hidden, 0, bytes, 0, bytes.Length);
    File.WriteAllBytes(output, bytes);
    Console.WriteLine(JsonSerializer.Serialize(new { tokens = tokens.Length, hidden = encoder.HiddenSize,
        seconds, backend = backend.ToString(), fusedRequested = Environment.GetEnvironmentVariable("TS_QWEN_TE_FUSED") != "0", output }));
}
GgmlBasicOps.ReleaseReuseComputeBuffers();
GgmlBasicOps.ClearHostBufferCache();
GgmlBasicOps.Shutdown();
return 0;

// A tensor-parallel group whose ranks the native loopback init placed on one device.
sealed class LoopbackGroup(int degree) : TensorSharp.ITensorParallelGroup
{
    public int Degree => degree;
    public bool IsActive => degree > 1;
    public int GlobalDegree => degree;
    public int GlobalRankOffset => 0;
    public int NodeCount => 1;
    public TensorSharp.IAllocator GetAllocator(int rank) => throw new NotSupportedException();
    public void AllReduce(TensorSharp.Tensor[] tensors) => throw new NotSupportedException();
    public void Synchronize() { }
    public void Barrier() { }
    public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
    public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
    public void Dispose() { }
}
