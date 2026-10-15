// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.IO.MemoryMappedFiles;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;

return Probe.Run(args);

static class Probe
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNameCaseInsensitive = true };
    private static List<Weight>? _unreleasedNativeSources;
    // Use the existing independent C# decoder without widening a product API for
    // this diagnostic. The delegate is resolved once, before any measurements.
    private static readonly Action<int, byte[], int, float[], int, long> ManagedDecode =
        typeof(ModelBase).Assembly.GetType("TensorSharp.Models.ManagedQuantizedOps", true)!
        .GetMethod("DequantizeToFloat32", BindingFlags.Public | BindingFlags.Static, null,
            [typeof(int), typeof(byte[]), typeof(int), typeof(float[]), typeof(int), typeof(long)], null)!
        .CreateDelegate<Action<int, byte[], int, float[], int, long>>();

    public static int Run(string[] args)
    {
        var options = new Dictionary<string, string>();
        for (int i = 0; i < args.Length; i += 2)
        {
            if (i + 1 == args.Length || args[i] is not ("--model" or "--output" or "--route" or "--layer" or
                "--experts" or "--tokens" or "--used" or "--output-rows" or "--input-file" or "--routing-file" or "--max-weight-bytes" or "--preserve-expert-axis"))
                throw new ArgumentException("Use --model GGUF --output DIRECTORY --route cpu|gpu-stream|gpu-resident " +
                    "[--layer 0 --experts 7,311 --used 2 --tokens 1,8,9,38 --output-rows 64] " +
                    "[--input-file input.f32 --routing-file routing.json --max-weight-bytes 134217728 --preserve-expert-axis false].");
            options.Add(args[i], args[i + 1]);
        }
        string model = Path.GetFullPath(options["--model"]), output = Path.GetFullPath(options["--output"]);
        string route = options.GetValueOrDefault("--route", "cpu");
        if (route is not ("cpu" or "gpu-stream" or "gpu-resident")) throw new ArgumentException("Unknown route.");
        int layer = int.Parse(options.GetValueOrDefault("--layer", "0"));
        int[] experts = options.GetValueOrDefault("--experts", "7,311").Split(',').Select(int.Parse).ToArray();
        int used = int.Parse(options.GetValueOrDefault("--used", Math.Min(2, experts.Length).ToString()));
        int[] counts = options.GetValueOrDefault("--tokens", "1,8,9,38").Split(',').Select(int.Parse).ToArray();
        int sampleCount = int.Parse(options.GetValueOrDefault("--output-rows", "64"));
        bool preserveExperts = bool.Parse(options.GetValueOrDefault("--preserve-expert-axis", "false"));
        long maxWeightBytes = long.Parse(options.GetValueOrDefault("--max-weight-bytes", preserveExperts ? "1073741824" : "134217728"));
        string? inputFile = options.GetValueOrDefault("--input-file"), routingFile = options.GetValueOrDefault("--routing-file");
        if (layer < 0 || experts.Length is < 1 or > 32 || experts.Distinct().Count() != experts.Length || experts.Any(e => e < 0) ||
            used < 1 || used > experts.Length || counts.Length == 0 || counts.Any(n => n is < 1 or > 256) || counts.Distinct().Count() != counts.Length || sampleCount < 0 ||
            maxWeightBytes is < 1 or > 4294967296 || (inputFile == null) != (routingFile == null))
            throw new ArgumentException("Invalid bounds; recorded inputs and routing must be supplied together.");
        if (Directory.Exists(output)) throw new IOException("Use a fresh output directory to preserve earlier evidence.");
        Directory.CreateDirectory(output);
        // In-process .NET updates on Windows do not necessarily update native
        // CRT getenv. Require inherited configuration before loading the native
        // library instead of claiming a requested route actually ran.
        foreach (var setting in new[] {
            ("TS_HOST_MOE_DEVICE_MIN_BATCH", route == "gpu-stream" ? "1" : "0"),
            ("TS_HOST_MOE_EXPERT_CACHE_MB", "0"), ("TS_HOST_MOE_PIN", "0") })
            if (Environment.GetEnvironmentVariable(setting.Item1) != setting.Item2)
                throw new ArgumentException($"Set {setting.Item1}={setting.Item2} in the parent process before launching this probe.");
        if (route == "gpu-stream" && Environment.GetEnvironmentVariable("TS_HOST_MOE_TIMING") != "2")
            throw new ArgumentException("Set TS_HOST_MOE_TIMING=2 externally; retain native HOSTMOE-COPY lines as actual streaming-route evidence.");
        var cases = new List<object>();
        var weights = new List<Weight>();
        GgmlContext? context = null;
        string? error = null;
        object? geometry = null;
        object? native = null;
        try
        {
            using var file = new GgufFile(model);
            string prefix = $"blk.{layer}.ffn_";
            long remaining = maxWeightBytes;
            var gate = Read(file, prefix + "gate_exps.weight", experts, preserveExperts, ref remaining); weights.Add(gate);
            var up = Read(file, prefix + "up_exps.weight", experts, preserveExperts, ref remaining); weights.Add(up);
            var down = Read(file, prefix + "down_exps.weight", experts, preserveExperts, ref remaining); weights.Add(down);
            int h = gate.K, f = gate.M;
            if (up.K != h || up.M != f || down.K != f || down.M != h || weights.Any(w => w.OriginalExperts != gate.OriginalExperts))
                throw new InvalidDataException("Expected separate, equally stacked gate/up [H,FF,E] and down [FF,H,E] tensors.");
            // A sidecar or bias changes the formula; this first diagnostic refuses it.
            if (file.Tensors.Keys.Any(k => k.StartsWith(prefix, StringComparison.Ordinal) &&
                (k.Contains("_exps.bias", StringComparison.Ordinal) || k.Contains("_exps.scale", StringComparison.Ordinal))))
                throw new NotSupportedException("Expert bias/scale sidecars require an explicit oracle extension.");
            int[] columns = Enumerable.Range(0, sampleCount == 0 ? h : Math.Min(h, sampleCount))
                .Select(i => sampleCount == 0 || sampleCount >= h ? i : (int)((long)i * h / sampleCount)).ToArray();
            int nativeExperts = preserveExperts ? gate.OriginalExperts : experts.Length;
            geometry = new { Hidden = h, FeedForward = f, OriginalExperts = gate.OriginalExperts, LoadedExperts = experts,
                NativeExpertCount = nativeExperts, OracleOutputColumns = columns,
                OriginalExpertAxisPreserved = preserveExperts, NativeWeightBytes = maxWeightBytes - remaining,
                SelectedQuantizedWeightBytesRead = weights.Sum(w => w.Raw.LongLength) };
            context = new GgmlContext([0], route == "cpu" ? GgmlBackendType.Cpu : GgmlBackendType.Cuda);
            var allocator = new GgmlAllocator(context, 0);
            GgmlBasicOps.SetDeviceCopyBudget(0);
            native = LoadedNative();
            foreach (var weight in weights) weight.DecodeAndCrosscheck();
            int maxTokens = counts.Max();
            float[] inputs = inputFile == null ? Inputs(maxTokens, h) : ReadFloats(inputFile, checked(maxTokens * h));
            Routing routing = routingFile == null ? Routes(maxTokens, experts, used) :
                JsonSerializer.Deserialize<Routing>(File.ReadAllText(routingFile), Json) ?? throw new InvalidDataException("Empty routing JSON.");
            if (routing.Used is < 1 || routing.Used > experts.Length || routing.SelectedExperts.Length != maxTokens * routing.Used ||
                routing.Weights.Length != routing.SelectedExperts.Length || routing.Weights.Any(w => !float.IsFinite(w)))
                throw new InvalidDataException("Routing must contain Used, token-major original SelectedExperts and finite Weights for max(tokens).");
            if (routingFile != null && options.ContainsKey("--used") && routing.Used != used)
                throw new ArgumentException("--used disagrees with the routing file.");
            int[] ids = routing.SelectedExperts.Select(id => Array.IndexOf(experts, id)).ToArray();
            if (ids.Any(id => id < 0)) throw new InvalidDataException("Every routed expert must be explicitly loaded with --experts.");
            int[] nativeIds = preserveExperts ? routing.SelectedExperts : ids;
            foreach (int n in counts)
            {
                float[] actual;
                double executionMs;
                using (var input = new Tensor(allocator, DType.Float32, n, h))
                using (var result = new Tensor(allocator, DType.Float32, n, h))
                {
                    Marshal.Copy(inputs, 0, input.Storage.PtrAtElement(0), n * h);
                    // Nonfinite canaries detect a silently unexecuted operator.
                    Marshal.Copy(Enumerable.Repeat(float.NaN, n * h).ToArray(), 0, result.Storage.PtrAtElement(0), n * h);
                    var timer = Stopwatch.StartNew();
                    GgmlBasicOps.MoEFFNPrefill(input, result, n, h, f, nativeExperts, routing.Used, nativeIds, routing.Weights,
                        gate.Pointer, gate.Type, gate.K, gate.M, gate.NativeBytes,
                        up.Pointer, up.Type, up.K, up.M, up.NativeBytes,
                        down.Pointer, down.Type, down.K, down.M, down.NativeBytes,
                        null!, null!, null!, GgmlBasicOps.MoEActivation.SwiGLUSplit, runOnCpu: route != "gpu-resident");
                    actual = result.GetElementsAsFloat(n * h);
                    executionMs = timer.Elapsed.TotalMilliseconds;
                }
                if (actual.Any(x => !float.IsFinite(x))) throw new InvalidOperationException("Operator returned nonfinite/unwritten output.");
                var watch = Stopwatch.StartNew();
                double[] expected = Oracle(gate, up, down, inputs, ids, routing.Weights, routing.Used, n, columns);
                float[] sampled = Enumerable.Range(0, n).SelectMany(t => columns.Select(c => actual[t * h + c])).ToArray();
                var metrics = Compare(expected, sampled);
                string stem = $"n{n}";
                File.WriteAllBytes(Path.Combine(output, stem + ".actual.f32"), MemoryMarshal.AsBytes(actual.AsSpan()).ToArray());
                File.WriteAllBytes(Path.Combine(output, stem + ".oracle.f64"), MemoryMarshal.AsBytes(expected.AsSpan()).ToArray());
                File.WriteAllBytes(Path.Combine(output, stem + ".input.f32"), MemoryMarshal.AsBytes(inputs.AsSpan(0, n * h)).ToArray());
                var observation = new { Tokens = n, UsedExpertsPerToken = routing.Used, Metrics = metrics, ExecutionMilliseconds = executionMs,
                    OracleMilliseconds = watch.Elapsed.TotalMilliseconds, FullOutputElementsCheckedFinite = actual.Length,
                    OracleElementsCompared = expected.Length, FullOutputSha256 = Hash(actual), InputSha256 = Hash(inputs.AsSpan(0, n * h)),
                    OriginalExpertIds = routing.SelectedExperts.Take(n * routing.Used).ToArray(),
                    NativeExpertIds = nativeIds.Take(n * routing.Used).ToArray(),
                    OracleCompactExpertIds = ids.Take(n * routing.Used).ToArray(), RoutingWeights = routing.Weights.Take(n * routing.Used).ToArray() };
                cases.Add(observation);
                File.WriteAllText(Path.Combine(output, stem + ".json"), JsonSerializer.Serialize(observation, Json));
                Console.WriteLine($"N={n} route={route} oracle sampled={expected.Length} relL2={metrics.RelativeL2:G9} maxabs={metrics.MaxAbsolute:G9} cosine={metrics.Cosine:G12}");
            }
        }
        catch (Exception ex) { error = ex.ToString(); Console.Error.WriteLine(error); }
        finally
        {
            // Native cache keys borrow pinned bytes; release all native owners before unpinning.
            bool released = false;
            try { GgmlBasicOps.ReleaseReuseComputeBuffers(); GgmlBasicOps.ClearHostBufferCache(); context?.ReleasePooledMemory(); GgmlBasicOps.Shutdown(); released = true; }
            catch (Exception ex) { error = (error == null ? "" : error + "\nCleanup: ") + ex; }
            // On cleanup failure keep GC roots pinned until process exit: a
            // remaining native owner may still borrow those addresses.
            if (released) foreach (var weight in weights) weight.Dispose();
            else _unreleasedNativeSources = weights;
        }
        File.WriteAllText(Path.Combine(output, "report.json"), JsonSerializer.Serialize(new
        {
            Status = error == null ? "diagnostic-completed" : "execution-failed", Error = error,
            QualityOrNumericalPassClaimed = false, Route = route, Model = model, Layer = layer, Native = native,
            InputProvenance = inputFile == null ? "synthetic inputs and fixed routes; real GGUF weight slices" :
                "external input/routing files; model-capture provenance is not independently verified",
            InputFile = inputFile, RoutingFile = routingFile,
            Environment = new Dictionary<string, string?>
            {
                ["TS_HOST_MOE_DEVICE_MIN_BATCH"] = System.Environment.GetEnvironmentVariable("TS_HOST_MOE_DEVICE_MIN_BATCH"),
                ["TS_HOST_MOE_EXPERT_CACHE_MB"] = System.Environment.GetEnvironmentVariable("TS_HOST_MOE_EXPERT_CACHE_MB"),
                ["TS_HOST_MOE_EXPERT_FILTER"] = System.Environment.GetEnvironmentVariable("TS_HOST_MOE_EXPERT_FILTER"),
                ["TS_HOST_MOE_TIMING"] = System.Environment.GetEnvironmentVariable("TS_HOST_MOE_TIMING"),
                ["GGML_CUDA_DISABLE_FUSION"] = System.Environment.GetEnvironmentVariable("GGML_CUDA_DISABLE_FUSION"),
                ["GGML_CUDA_FORCE_MMQ"] = System.Environment.GetEnvironmentVariable("GGML_CUDA_FORCE_MMQ"),
                ["GGML_CUDA_FORCE_CUBLAS"] = System.Environment.GetEnvironmentVariable("GGML_CUDA_FORCE_CUBLAS"),
            },
            Geometry = geometry, Weights = weights.Select(w => w.Report()).ToArray(), Cases = cases,
            Limitations = "Independent managed dequantization plus scalar FP64 dot/SiLU/aggregation on sampled output columns. " +
                "CPU/CUDA activation quantization and reduction policies differ; no arbitrary error gate or model-quality conclusion. " +
                (preserveExperts ? "Native expert-axis shape and original selected IDs are preserved through read-only tensor-range mappings. " :
                "Expert axis is compacted, preserving each selected expert's H/FF and bytes, not original E dispatch geometry. ") +
                "Timings are diagnostic only and include neither a controlled benchmark nor model inference."
        }, Json));
        return error == null ? 0 : 1;
    }

    private static Weight Read(GgufFile file, string name, int[] experts, bool preserveExperts, ref long remaining)
    {
        var info = file.Tensors[name];
        if (info.Shape.Length != 3) throw new InvalidDataException($"{name} is not [K,M,E].");
        int k = checked((int)info.Shape[0]), m = checked((int)info.Shape[1]), e = checked((int)info.Shape[2]);
        if (experts.Any(id => id >= e)) throw new ArgumentException($"Expert id outside {name}.");
        long block = GgufFile.GetBlockSize(info.Type);
        if (k % block != 0) throw new InvalidDataException("Unaligned weight row.");
        long perExpert = checked(k / block * GgufFile.GetTypeSize(info.Type) * m), bytes = checked(perExpert * experts.Length);
        var region = file.GetTensorFileRegion(name);
        if (region.ByteLength != checked(perExpert * e)) throw new InvalidDataException("Unexpected GGUF tensor size.");
        long nativeBytes = preserveExperts ? region.ByteLength : bytes;
        if (nativeBytes > remaining || bytes > int.MaxValue) throw new ArgumentException("Native weight ranges exceed --max-weight-bytes.");
        byte[] raw = new byte[checked((int)bytes)];
        using var handle = File.OpenHandle(region.Path, FileMode.Open, FileAccess.Read, FileShare.Read);
        for (int i = 0; i < experts.Length; i++)
        {
            var slice = raw.AsSpan(checked((int)(i * perExpert)), checked((int)perExpert));
            long offset = checked(region.Offset + experts[i] * perExpert);
            int read = 0;
            while (read < slice.Length)
            {
                int count = RandomAccess.Read(handle, slice[read..], offset + read);
                if (count == 0) throw new EndOfStreamException(name);
                read += count;
            }
        }
        remaining -= nativeBytes;
        var weight = new Weight(name, (int)info.Type, k, m, e, region, experts, perExpert, raw);
        try { if (preserveExperts) weight.MapNativeTensor(); return weight; }
        catch { weight.Dispose(); throw; }
    }

    private static double[] Oracle(Weight gate, Weight up, Weight down, float[] x, int[] ids, float[] routes, int used, int n, int[] columns)
    {
        int h = gate.K, f = gate.M;
        double[] result = new double[n * columns.Length];
        // Deliberately scalar, double accumulation, no GGML/managed quantized dot.
        for (int t = 0; t < n; t++)
        for (int slot = 0; slot < used; slot++)
        {
            int expert = ids[t * used + slot];
            double[] activated = new double[f];
            for (int row = 0; row < f; row++)
            {
                double g = 0, u = 0;
                int source = (expert * f + row) * h;
                for (int col = 0; col < h; col++)
                {
                    double input = x[t * h + col];
                    g += (double)gate.Decoded[source + col] * input;
                    u += (double)up.Decoded[source + col] * input;
                }
                activated[row] = (g >= 0 ? g / (1 + Math.Exp(-g)) : g * Math.Exp(g) / (1 + Math.Exp(g))) * u;
            }
            for (int index = 0; index < columns.Length; index++)
            {
                double d = 0;
                int source = (expert * h + columns[index]) * f;
                for (int col = 0; col < f; col++) d += (double)down.Decoded[source + col] * activated[col];
                result[t * columns.Length + index] += d * routes[t * used + slot];
            }
        }
        if (result.Any(v => !double.IsFinite(v))) throw new InvalidDataException("FP64 oracle became nonfinite.");
        return result;
    }

    private static float[] Inputs(int n, int h) => Enumerable.Range(0, n * h)
        .Select(i => (float)(0.25 * Math.Sin((i + 1) * 0.073) + 0.125 * Math.Cos((i + 7) * 0.019))).ToArray();

    private static Routing Routes(int n, int[] experts, int used)
    {
        int[] ids = new int[n * used];
        float[] weights = new float[ids.Length];
        for (int t = 0; t < n; t++)
        for (int slot = 0; slot < used; slot++)
        {
            // The two-slot control also exercises duplicate-ID aggregation. The
            // full used-count geometry has distinct routes, as a top-k router does.
            ids[t * used + slot] = experts[(t + (used == 2 && t % 3 == 0 ? 0 : slot)) % experts.Length];
            weights[t * used + slot] = used == 1 ? 1 : used == 2 ? (slot == 0 ? 0.375f : 0.625f) :
                (slot + 1) / (used * (used + 1) / 2f);
        }
        return new(used, ids, weights);
    }

    private static float[] ReadFloats(string path, int count)
    {
        byte[] raw = File.ReadAllBytes(path);
        if (raw.Length != checked(count * sizeof(float))) throw new InvalidDataException("Input file must exactly match max(tokens) * H F32 elements.");
        float[] result = MemoryMarshal.Cast<byte, float>(raw).ToArray();
        if (result.Any(x => !float.IsFinite(x))) throw new InvalidDataException("Input is nonfinite.");
        return result;
    }

    private static Metrics Compare(double[] expected, float[] actual)
    {
        if (expected.Length != actual.Length || expected.Length == 0) throw new ArgumentException("Mismatched metric lengths.");
        double error = 0, expectedNorm = 0, actualNorm = 0, dot = 0, max = 0, expectedMax = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double a = expected[i], b = actual[i], delta = a - b;
            error += delta * delta; expectedNorm += a * a; actualNorm += b * b; dot += a * b;
            max = Math.Max(max, Math.Abs(delta));
            expectedMax = Math.Max(expectedMax, Math.Abs(a));
        }
        // Relative error is undefined for a zero reference. Do not invent a tiny
        // denominator: preserve absolute/RMS errors and make that case explicit.
        return new(expected.Length, expectedNorm == 0 ? null : Math.Sqrt(error / expectedNorm), max,
            expectedNorm == 0 || actualNorm == 0 ? (error == 0 ? 1 : null) : dot / Math.Sqrt(expectedNorm * actualNorm),
            Math.Sqrt(expectedNorm), Math.Sqrt(actualNorm), Math.Sqrt(error), Math.Sqrt(error / expected.Length), expectedMax);
    }

    private static string Hash(float[] values) => Hash(values.AsSpan());
    private static string Hash(ReadOnlySpan<float> values) => Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(values))).ToLowerInvariant();
    private static object LoadedNative() => Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
        .Where(m => Path.GetFileName(m.FileName).Contains("GgmlOps", StringComparison.OrdinalIgnoreCase))
        .Select(m => new { m.FileName, Sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(m.FileName))).ToLowerInvariant() }).ToArray();

    private sealed record Routing(int Used, int[] SelectedExperts, float[] Weights);
    private sealed record Metrics(int Elements, double? RelativeL2, double MaxAbsolute, double? Cosine,
        double ReferenceL2, double ActualL2, double ErrorL2, double AbsoluteRmsError, double MaxReferenceAbsolute);

    private sealed unsafe class Weight(string name, int type, int k, int m, int originalExperts, GgufFileRegion region,
        int[] selectedExperts, long bytesPerExpert, byte[] raw) : IDisposable
    {
        private GCHandle _pin = GCHandle.Alloc(raw, GCHandleType.Pinned);
        public readonly int Type = type, K = k, M = m, OriginalExperts = originalExperts;
        public readonly byte[] Raw = raw;
        public float[] Decoded = [];
        private MemoryMappedFile? _mapping;
        private MemoryMappedViewAccessor? _view;
        private byte* _mappedBase;
        public IntPtr Pointer => _mappedBase == null ? _pin.AddrOfPinnedObject() : (IntPtr)(_mappedBase + _view!.PointerOffset);
        public long NativeBytes => _mappedBase == null ? Raw.LongLength : region.ByteLength;
        private Metrics? _decodeComparison;
        public void MapNativeTensor()
        {
            _mapping = MemoryMappedFile.CreateFromFile(region.Path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            _view = _mapping.CreateViewAccessor(region.Offset, region.ByteLength, MemoryMappedFileAccess.Read);
            _view.SafeMemoryMappedViewHandle.AcquirePointer(ref _mappedBase);
        }
        public void DecodeAndCrosscheck()
        {
            Decoded = new float[checked(K * M * selectedExperts.Length)];
            ManagedDecode(Type, Raw, 0, Decoded, 0, Decoded.LongLength);
            float[] native = new float[Decoded.Length];
            GgmlGgufTensorDequant.DequantizeToFloat32(Type, Raw, 0, native, 0, native.LongLength);
            if (Decoded.Any(v => !float.IsFinite(v)) || native.Any(v => !float.IsFinite(v))) throw new InvalidDataException("Nonfinite weight decode.");
            _decodeComparison = Compare(Decoded.Select(x => (double)x).ToArray(), native);
            for (int i = 0; i < Decoded.Length; i++)
                if (Math.Abs(Decoded[i] - native[i]) > 1e-5 * Math.Max(1, Math.Abs((double)native[i])))
                    throw new InvalidDataException($"Independent managed/native dequantization disagree: {name} element {i}.");
        }
        public object Report() => new { Name = name, Type, K, M, OriginalExperts, SelectedExperts = selectedExperts,
            Region = region, BytesPerExpert = bytesPerExpert, SelectedBytes = Raw.LongLength,
            NativeSource = _mapping == null ? "compacted pinned selected bytes" : "original read-only tensor-range mapping",
            NativeWeightBytes = _mapping == null ? Raw.LongLength : region.ByteLength,
            SelectedRawSha256 = Convert.ToHexString(SHA256.HashData(Raw)).ToLowerInvariant(),
            Dequantization = "C# ManagedQuantizedOps; separate native ggml to_float crosscheck", DequantizationComparison = _decodeComparison };
        public void Dispose()
        {
            if (_mappedBase != null) { _view!.SafeMemoryMappedViewHandle.ReleasePointer(); _mappedBase = null; }
            _view?.Dispose(); _mapping?.Dispose();
            if (_pin.IsAllocated) _pin.Free();
        }
    }
}
