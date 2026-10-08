// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;

return Probe.Run(args);

static unsafe class Probe
{
    public static int Run(string[] args)
    {
        var options = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i += 2)
        {
            if (i + 1 == args.Length || args[i] is not ("--model" or "--json" or "--layer" or "--group" or "--tokens"
                or "--tile-bytes" or "--token-rows" or "--input-file" or "--dump-dir"))
                throw new ArgumentException("Use --model PATH --json PATH [--layer 0] [--group qkv|gate-up|both] [--tokens 36] [--tile-bytes 1048576] [--token-rows 32] [--input-file packed-f32-path] [--dump-dir PATH].");
            options.Add(args[i], args[i + 1]);
        }
        string model = Path.GetFullPath(options["--model"]);
        string report = Path.GetFullPath(options.GetValueOrDefault("--json", "artifacts/gemma-fused-projections/report.json"));
        int layer = int.Parse(options.GetValueOrDefault("--layer", "0"));
        string selection = options.GetValueOrDefault("--group", "both");
        string[] groups = selection switch { "qkv" => ["qkv"], "gate-up" => ["gate-up"], "both" => ["qkv", "gate-up"], _ => throw new ArgumentException("Unknown projection group.") };
        int[] counts = options.GetValueOrDefault("--tokens", "36").Split(',').Select(int.Parse).ToArray();
        int tileBytes = int.Parse(options.GetValueOrDefault("--tile-bytes", "1048576"));
        int tokenRows = int.Parse(options.GetValueOrDefault("--token-rows", "32"));
        string? inputPath = options.TryGetValue("--input-file", out string? inputValue) ? Path.GetFullPath(inputValue) : null;
        string? dump = options.GetValueOrDefault("--dump-dir");
        if (layer < 0 || counts.Length == 0 || counts.Any(n => n is < 1 or > 4096) || tileBytes <= 0 || tokenRows is < 1 or > 4096)
            throw new ArgumentException("Layer must be nonnegative, tile bytes positive, and token counts/capacity in 1..4096.");
        if (inputPath != null && (groups.Length != 1 || counts.Length != 1))
            throw new ArgumentException("A captured input requires exactly one group and one exact token count; it is never padded or truncated.");

        var observations = new List<object>();
        bool allStrict = true;
        string? capturedInputSha256 = null;
        string? error = null;
        GgmlContext? context = null;
        try
        {
            using var file = new GgufFile(model);
            if (file.GetString("general.architecture") != "gemma4") throw new ArgumentException("Expected gemma4 GGUF.");
            context = new GgmlContext([0], GgmlBackendType.Cuda);
            var allocator = new GgmlAllocator(context, 0);
            GgmlBasicOps.SetDeviceCopyBudget(0);
            foreach (string group in groups)
            {
                string prefix = $"blk.{layer}";
                string[] names = group == "qkv"
                    ? [$"{prefix}.attn_q.weight", $"{prefix}.attn_k.weight", file.Tensors.ContainsKey($"{prefix}.attn_v.weight") ? $"{prefix}.attn_v.weight" : $"{prefix}.attn_k.weight"]
                    : [$"{prefix}.ffn_gate.weight", $"{prefix}.ffn_up.weight"];
                var parts = new List<Part>();
                long totalBytes = 0;
                int totalRows = 0, inner = 0;
                foreach (string name in names)
                {
                    if (!file.Tensors.TryGetValue(name, out var info) || info.Shape.Length != 2 || info.Type != GgmlTensorType.Q8_0)
                        throw new ArgumentException($"{name} must exist as original two-dimensional Q8_0 weights; shared-Q-only layers are not a QKV group.");
                    if (file.Tensors.ContainsKey(name[..^7] + ".scale"))
                        throw new NotSupportedException("Sidecar-scaled weights require a separate diagnostic; raw concatenation must preserve reference semantics.");
                    int width = checked((int)info.Shape[0]), rows = checked((int)info.Shape[1]);
                    if (width <= 0 || width % 32 != 0 || rows <= 0 || (inner != 0 && width != inner))
                        throw new InvalidDataException("Concatenated weights require positive compatible Q8_0 shapes.");
                    inner = width;
                    long bytes = checked(width / 32L * 34 * rows);
                    if (file.GetTensorByteCount(info) != bytes) throw new InvalidDataException("Weight payload does not match its declared shape.");
                    parts.Add(new Part(name, rows, totalRows, totalBytes, bytes, checked(file.DataOffset + (long)info.Offset)));
                    totalRows = checked(totalRows + rows); totalBytes = checked(totalBytes + bytes);
                }
                using var joined = new NativeBuffer(totalBytes);
                foreach (Part part in parts)
                    file.ReadTensorDataToNative(file.Tensors[part.Name], joined.Pointer + checked((nint)part.ByteOffset), part.Bytes);
                float[] inputs = ReadInput(inner, counts.Max(), inputPath);
                if (inputPath != null)
                    capturedInputSha256 = Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(inputs.AsSpan())));
                try
                {
                    foreach (int n in counts)
                    {
                        using var input = new Tensor(allocator, DType.Float32, n, inner);
                        Marshal.Copy(inputs, 0, input.Storage.PtrAtElement(0), checked(n * inner));
                        using var fusedResult = new Tensor(allocator, DType.Float32, n, totalRows);
                        GgmlBasicOps.AddmmQuant(fusedResult, input, joined.Pointer, 8, inner, totalRows, totalBytes);
                        float[] fused = fusedResult.GetElementsAsFloat(checked(n * totalRows));
                        // Each independent reference is completed before changing
                        // cache key/shape. No captured graph retains the joined view.
                        GgmlBasicOps.ClearHostBufferCache();
                        foreach (Part part in parts)
                        {
                            IntPtr pointer = joined.Pointer + checked((nint)part.ByteOffset);
                            using var separateResult = new Tensor(allocator, DType.Float32, n, part.Rows);
                            GgmlBasicOps.AddmmQuant(separateResult, input, pointer, 8, inner, part.Rows, part.Bytes);
                            float[] separate = separateResult.GetElementsAsFloat(checked(n * part.Rows));
                            GgmlBasicOps.ClearHostBufferCache();
                            float[] fusedSlice = new float[separate.Length];
                            for (int token = 0; token < n; token++)
                                Array.Copy(fused, token * totalRows + part.RowOffset, fusedSlice, token * part.Rows, part.Rows);
                            int tileRows = checked((int)Math.Min(part.Rows, tileBytes / (inner / 32L * 34)));
                            if (tileRows < 1) throw new ArgumentException("Tile capacity cannot hold one Q8 row.");
                            var streamed = Stream(pointer, input, inner, part.Rows, n, tileRows, Math.Min(n, tokenRows));
                            Metrics shape = Compare(fusedSlice, separate, part.Rows);
                            Metrics overall = Compare(fusedSlice, streamed.Values, part.Rows);
                            Metrics kernel = Compare(separate, streamed.Values, part.Rows);
                            allStrict &= shape.StrictGate && overall.StrictGate && kernel.StrictGate;
                            observations.Add(new { Group = group, Layer = layer, Weight = part.Name, part.RowOffset, part.FileOffset,
                                InputWidth = inner, Tokens = n, FusedRows = totalRows, SeparateRows = part.Rows, ConcatenatedWeightBytes = totalBytes,
                                TileRows = tileRows, TokenCapacity = Math.Min(n, tokenRows), streamed.PayloadBytes,
                                InputSha256 = Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(inputs.AsSpan(0, checked(n * inner))))),
                                FusedVsSeparateResident = shape, FusedVsSeparateStreamed = overall, SeparateResidentVsStreamed = kernel,
                                FinalBudget = streamed.FinalBudget });
                            Console.WriteLine($"{group} {part.Name} K={inner} M={totalRows}/{part.Rows} N={n} fused/separate relL2={shape.RelativeL2:G9}; fused/stream={overall.RelativeL2:G9}; separate/stream={kernel.RelativeL2:G9}; strict={shape.StrictGate && overall.StrictGate && kernel.StrictGate}");
                            if (dump != null)
                            {
                                Directory.CreateDirectory(dump);
                                string stem = $"{group}.{part.Name}.offset{part.RowOffset}.n{n}";
                                WriteFloats(Path.Combine(dump, stem + ".fused.f32"), fusedSlice);
                                WriteFloats(Path.Combine(dump, stem + ".separate.f32"), separate);
                                WriteFloats(Path.Combine(dump, stem + ".streamed.f32"), streamed.Values);
                            }
                        }
                    }
                }
                finally { GgmlBasicOps.ClearHostBufferCache(); }
            }
        }
        catch (Exception ex) { error = ex.ToString(); Console.Error.WriteLine(error); }
        finally
        {
            try { GgmlBasicOps.ClearHostBufferCache(); context?.ReleasePooledMemory(); GgmlBasicOps.Shutdown(); }
            catch (Exception ex) { error = (error == null ? "" : error + "\nCleanup: ") + ex; }
        }
        var modules = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
            .Where(m => Path.GetFileName(m.FileName).Contains("GgmlOps", StringComparison.OrdinalIgnoreCase))
            .Select(m => new { m.FileName, Sha256 = Hash(m.FileName) }).ToArray();
        Directory.CreateDirectory(Path.GetDirectoryName(report)!);
        File.WriteAllText(report, JsonSerializer.Serialize(new { Completed = error == null, AllStrictGatesPassed = error == null && allStrict,
            Error = error, Model = model, ModelFileBytes = File.Exists(model) ? (long?)new FileInfo(model).Length : null,
            ModelLastWriteUtc = File.Exists(model) ? (DateTime?)File.GetLastWriteTimeUtc(model) : null,
            Layer = layer, Group = selection, InputFile = inputPath, InputFileSha256 = capturedInputSha256,
            Input = inputPath == null ? "Deterministic synthetic F32; same prefix for each N" : "Exact caller-supplied packed little-endian [N,K] F32; no padding/truncation",
            Native = modules, ProbeSha256 = Hash(typeof(Probe).Assembly.Location),
            GgmlRevision = Environment.GetEnvironmentVariable("TS_VALIDATION_GGML_REVISION"),
            CudaDisableFusion = Environment.GetEnvironmentVariable("GGML_CUDA_DISABLE_FUSION"), Observations = observations,
            Scope = "Numerical localization only; loads one concatenated QKV or gate/up group, never the complete model. Ordinary full-N fused and separate AddmmQuant references are unchanged. Streamed sessions use each original matrix's logical output rows and the original N. Every output is compared. Exit0 means completed finite diagnostics and cleanup, not strict parity. CUDA streaming payload alone is budgeted; host joined weights/input/output and ordinary reference caches/scratch are outside that reservation. No throughput or whole-model qualification is claimed." }, new JsonSerializerOptions { WriteIndented = true }));
        return error == null ? 0 : 2;
    }

    static (float[] Values, long PayloadBytes, object FinalBudget) Stream(IntPtr weights, Tensor input, int k, int rows, int n, int tileRows, int capacity)
    {
        long payload = GgmlWeightStreamingSession.GetPayloadBytes(8, k, tileRows, capacity, GgmlWeightStreamingArithmetic.ResidentCuda, n, rows);
        var budget = new MemoryBudget([new MemoryCharge("stream/gpu0", payload)]);
        using var output = new NativeBuffer(checked((long)tileRows * capacity * sizeof(float)));
        float[] values = new float[checked(rows * n)];
        GgmlWeightStreamingSession? session = null;
        try
        {
            session = new GgmlWeightStreamingSession(budget, ["stream/gpu0"], 0, 8, k, tileRows, capacity,
                input.Storage.PtrAtElement(0), GgmlWeightStreamingArithmetic.ResidentCuda, n, rows);
            for (int token = 0; token < n; token += capacity)
            {
                int active = Math.Min(capacity, n - token);
                if (token != 0) session.UploadInput(input.Storage.PtrAtElement(checked((long)token * k)), active);
                for (int row = 0; row < rows; row += tileRows)
                {
                    int count = Math.Min(tileRows, rows - row);
                    session.Execute(weights + checked((nint)(row * (k / 32L * 34))), count, output.Pointer);
                    for (int column = 0; column < active; column++)
                        Marshal.Copy(output.Pointer + checked(column * count * sizeof(float)), values, (token + column) * rows + row, count);
                }
            }
        }
        catch (GgmlWeightStreamingAllocationException ex) { session = ex.UnreleasedSession; throw; }
        finally { session?.Dispose(); }
        var final = budget.Snapshot();
        if (final.Any(p => p.Reserved != 0 || p.Committed != 0)) throw new InvalidOperationException("Streaming payload reservation leaked.");
        return (values, payload, final);
    }

    static float[] ReadInput(int k, int n, string? path)
    {
        float[] values = new float[checked(k * n)];
        if (path != null)
        {
            byte[] bytes = File.ReadAllBytes(path);
            if (!BitConverter.IsLittleEndian || bytes.Length != checked(values.Length * sizeof(float)))
                throw new InvalidDataException("Input must contain exactly N*K little-endian F32 values on a little-endian host.");
            MemoryMarshal.Cast<byte, float>(bytes).CopyTo(values);
        }
        else
            for (int token = 0; token < n; token++) for (int j = 0; j < k; j++)
                values[token * k + j] = MathF.Sin((j + 1) * .173f + token * .37f) * (.2f + j % 13 * .21f)
                    + MathF.Cos((j + 1) * .071f - token * .19f) * .41f;
        if (values.Any(v => !float.IsFinite(v))) throw new InvalidDataException("Input contains nonfinite values.");
        return values;
    }

    static Metrics Compare(float[] expected, float[] actual, int width)
    {
        double ref2 = 0, act2 = 0, err2 = 0, dot = 0, max = 0, worstL2 = 0, minCos = 1;
        int topDifferences = 0, changed = 0;
        if (expected.Length != actual.Length || width <= 0 || expected.Length % width != 0) throw new InvalidDataException("Output shapes differ.");
        for (int start = 0; start < expected.Length; start += width)
        {
            double r2 = 0, a2 = 0, e2 = 0, d = 0;
            int refTop = start, actTop = start;
            for (int i = start; i < start + width; i++)
            {
                double a = expected[i], b = actual[i];
                if (!double.IsFinite(a) || !double.IsFinite(b)) throw new InvalidDataException("Nonfinite projection output.");
                r2 += a * a; a2 += b * b; e2 += (a - b) * (a - b); d += a * b;
                max = Math.Max(max, Math.Abs(a - b));
                if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i])) changed++;
                if (expected[i] > expected[refTop]) refTop = i;
                if (actual[i] > actual[actTop]) actTop = i;
            }
            if (refTop != actTop) topDifferences++;
            worstL2 = Math.Max(worstL2, Math.Sqrt(e2 / Math.Max(r2, 1e-300)));
            minCos = Math.Min(minCos, r2 == 0 && a2 == 0 ? 1 : d / Math.Sqrt(Math.Max(r2 * a2, 1e-300)));
            ref2 += r2; act2 += a2; err2 += e2; dot += d;
        }
        return new(Math.Sqrt(err2 / Math.Max(ref2, 1e-300)), ref2 == 0 && act2 == 0 ? 1 : dot / Math.Sqrt(Math.Max(ref2 * act2, 1e-300)),
            max, worstL2, minCos, topDifferences, changed, worstL2 <= .001 && minCos >= .999999 && topDifferences == 0);
    }

    static string Hash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)); }
    static void WriteFloats(string path, float[] values) => File.WriteAllBytes(path, MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
    sealed record Part(string Name, int Rows, int RowOffset, long ByteOffset, long Bytes, long FileOffset);
    sealed record Metrics(double RelativeL2, double Cosine, double MaxAbsolute, double WorstTokenRelativeL2, double MinimumTokenCosine,
        int Top1Differences, int ChangedFloatBits, bool StrictGate);
    sealed class NativeBuffer : IDisposable
    {
        public IntPtr Pointer { get; private set; }
        public NativeBuffer(long bytes)
        {
            Pointer = (IntPtr)NativeMemory.AlignedAlloc(checked((nuint)((bytes + 63) & ~63L)), 64);
            if (Pointer == IntPtr.Zero) throw new OutOfMemoryException();
        }
        public void Dispose() { NativeMemory.AlignedFree((void*)Pointer); Pointer = IntPtr.Zero; }
    }
}
