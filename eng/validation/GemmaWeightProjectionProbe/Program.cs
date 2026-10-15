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
            if (i + 1 == args.Length || args[i] is not ("--model" or "--json" or "--weights" or "--tokens" or "--tile-bytes" or "--input-file" or "--embedding-token" or "--dump-dir" or "--arithmetic" or "--token-rows"))
                throw new ArgumentException("Use --model PATH --json PATH [--weights comma-separated-names] [--tokens 1,8,9,32,36] [--tile-bytes 1048576] [--arithmetic full|resident] [--token-rows 32] [--input-file packed-f32-path | --embedding-token ID] [--dump-dir PATH].");
            options.Add(args[i], args[i + 1]);
        }
        string model = Path.GetFullPath(options["--model"]);
        string report = Path.GetFullPath(options.GetValueOrDefault("--json", "artifacts/gemma-weight-projections/report.json"));
        string[] names = options.GetValueOrDefault("--weights", "per_layer_model_proj.weight,blk.0.attn_q.weight,blk.0.attn_k.weight,blk.0.attn_v.weight,token_embd.weight").Split(',');
        int[] counts = options.GetValueOrDefault("--tokens", "1,8,9,32").Split(',').Select(int.Parse).ToArray();
        int tileBytes = int.Parse(options.GetValueOrDefault("--tile-bytes", "1048576"));
        int tokenRows = int.Parse(options.GetValueOrDefault("--token-rows", "32"));
        var arithmetic = options.GetValueOrDefault("--arithmetic", "full") switch
        {
            "full" => GgmlWeightStreamingArithmetic.FullPrecision,
            "resident" => GgmlWeightStreamingArithmetic.ResidentCuda,
            _ => throw new ArgumentException("--arithmetic must be full or resident.")
        };
        if (names.Length == 0 || counts.Length == 0 || counts.Any(n => n is < 1 or > 1024) || tileBytes <= 0 || tokenRows is < 1 or > 1024)
            throw new ArgumentException("Supply tensor names, token counts and token rows in 1..1024, and a positive tile capacity.");
        string? inputFile = options.GetValueOrDefault("--input-file"), dump = options.GetValueOrDefault("--dump-dir");
        int? embeddingToken = options.TryGetValue("--embedding-token", out var token) ? int.Parse(token) : null;
        if (inputFile != null && (embeddingToken != null || names.Length != 1))
            throw new ArgumentException("--input-file requires exactly one weight and cannot be combined with --embedding-token.");
        if (embeddingToken != null && (names.Length != 1 || names[0] != "per_layer_model_proj.weight"))
            throw new ArgumentException("Actual scaled embedding input is valid only for per_layer_model_proj.weight.");
        var observations = new List<object>();
        string? error = null;
        GgmlContext? context = null;
        try
        {
            using var file = new GgufFile(model);
            if (file.GetString("general.architecture") != "gemma4") throw new ArgumentException("Expected a Gemma4 GGUF.");
            context = new GgmlContext([0], GgmlBackendType.Cuda);
            var allocator = new GgmlAllocator(context, 0);
            // No precision-key registration: this arm uses the ordinary resident API.
            GgmlBasicOps.SetDeviceCopyBudget(0);
            foreach (string name in names)
            {
                var info = file.Tensors[name];
                if (info.Shape.Length != 2 || info.Type is not (GgmlTensorType.Q8_0 or GgmlTensorType.F16))
                    throw new ArgumentException($"{name} must be a two-dimensional Q8_0/F16 tensor.");
                int k = checked((int)info.Shape[0]), rows = checked((int)info.Shape[1]);
                int type = (int)info.Type;
                long rowBytes = type == 8 ? checked(k / 32L * 34) : checked(k * 2L);
                if (type == 8 && k % 32 != 0) throw new ArgumentException("Q8_0 K is not block aligned.");
                long bytes = checked(rowBytes * rows);
                if (file.GetTensorByteCount(info) != bytes) throw new InvalidDataException("Tensor payload size mismatch.");
                int tileRows = checked((int)Math.Min(rows, tileBytes / rowBytes));
                if (tileRows < 1) throw new ArgumentException($"Tile does not fit one row of {name}.");
                using var weight = new NativeBuffer(bytes);
                file.ReadTensorDataToNative(info, weight.Pointer, bytes);
                float[] inputs = Inputs(file, model, k, counts.Max(), inputFile, embeddingToken);
                float[]? firstResident = null, firstStreamed = null;
                int firstTokens = 0;
                try
                {
                    foreach (int n in counts)
                    {
                        using var input = new Tensor(allocator, DType.Float32, n, k);
                        Marshal.Copy(inputs, 0, input.Storage.PtrAtElement(0), checked(n * k));
                        using var result = new Tensor(allocator, DType.Float32, n, rows);
                        var timer = Stopwatch.StartNew();
                        GgmlBasicOps.AddmmQuant(result, input, weight.Pointer, type, k, rows, bytes);
                        float[] resident = result.GetElementsAsFloat(checked(n * rows));
                        double residentMs = timer.Elapsed.TotalMilliseconds;
                        int tokenCapacity = Math.Min(n, tokenRows);
                        long payload = GgmlWeightStreamingSession.GetPayloadBytes(type, k, tileRows, tokenCapacity, arithmetic, n, rows);
                        var budget = new MemoryBudget([new MemoryCharge("stream/gpu0", payload)]);
                        using var output = new NativeBuffer(checked(tokenCapacity * (long)tileRows * sizeof(float)));
                        float[] streamed = new float[checked(n * rows)];
                        timer.Restart();
                        GgmlWeightStreamingSession? session = null;
                        try
                        {
                            session = new GgmlWeightStreamingSession(budget, ["stream/gpu0"], 0, type, k, tileRows, tokenCapacity,
                                input.Storage.PtrAtElement(0), arithmetic, n, rows);
                            for (int firstToken = 0; firstToken < n; firstToken += tokenCapacity)
                            {
                                int activeTokens = Math.Min(tokenCapacity, n - firstToken);
                                if (firstToken != 0) session.UploadInput(input.Storage.PtrAtElement(checked((long)firstToken * k)), activeTokens);
                                for (int row = 0; row < rows;)
                                {
                                    int count = Math.Min(tileRows, rows - row);
                                    session.Execute(weight.Pointer + checked((nint)(row * rowBytes)), count, output.Pointer);
                                    for (int column = 0; column < activeTokens; column++)
                                        Marshal.Copy(output.Pointer + checked(column * count * sizeof(float)), streamed, (firstToken + column) * rows + row, count);
                                    row += count;
                                }
                            }
                        }
                        catch (GgmlWeightStreamingAllocationException ex) { session = ex.UnreleasedSession; throw; }
                        finally { session?.Dispose(); }
                        double streamMs = timer.Elapsed.TotalMilliseconds;
                        if (budget.Snapshot().Any(p => p.Reserved != 0 || p.Committed != 0)) throw new InvalidOperationException("Streaming reservation leaked.");
                        Metrics parity = Compare(resident, streamed, rows);
                        var oracle = Oracle(weight.Pointer, type, k, rows, n, inputs, resident, streamed);
                        var crossResident = firstResident == null ? null : Compare(firstResident, resident.AsSpan(0, rows).ToArray(), rows);
                        var crossStream = firstStreamed == null ? null : Compare(firstStreamed, streamed.AsSpan(0, rows).ToArray(), rows);
                        Console.WriteLine($"{name} type={type} K={k} rows={rows} N={n} resident/stream relL2={parity.RelativeL2:G9} cosine={parity.Cosine:G12} maxabs={parity.MaxAbsolute:G9} strict={parity.StrictGate} oracle resident={oracle.Resident.RelativeL2:G9} stream={oracle.Streamed.RelativeL2:G9}");
                        observations.Add(new { Weight = name, Type = type, InputWidth = k, OutputRows = rows, Tokens = n,
                            WeightBytes = bytes, FileOffset = checked(file.DataOffset + (long)info.Offset), TileRows = tileRows, TokenCapacity = tokenCapacity,
                            CudaPayloadBytes = payload, ResidentMilliseconds = residentMs, StreamedMilliseconds = streamMs,
                            Parity = parity, Oracle = oracle, FirstTokenCount = firstTokens,
                            FirstColumnVsFirstBatchResident = crossResident, FirstColumnVsFirstBatchStreamed = crossStream,
                            FinalBudget = budget.Snapshot() });
                        if (firstResident == null)
                        {
                            firstResident = resident.AsSpan(0, rows).ToArray();
                            firstStreamed = streamed.AsSpan(0, rows).ToArray();
                            firstTokens = n;
                        }
                        if (dump != null)
                        {
                            Directory.CreateDirectory(dump);
                            File.WriteAllBytes(Path.Combine(dump, $"{name}.n{n}.resident.f32"), MemoryMarshal.AsBytes(resident.AsSpan()).ToArray());
                            File.WriteAllBytes(Path.Combine(dump, $"{name}.n{n}.streamed.f32"), MemoryMarshal.AsBytes(streamed.AsSpan()).ToArray());
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
        File.WriteAllText(report, JsonSerializer.Serialize(new { Completed = error == null, Error = error, Model = model,
            ModelFileBytes = new FileInfo(model).Length, ModelLastWriteUtc = File.GetLastWriteTimeUtc(model),
            Input = inputFile == null ? embeddingToken == null ? "deterministic synthetic F32, identical first token across all N" : "actual decoded token embedding scaled by sqrt(K), repeated across N; true PLE input" : "caller supplied packed F32; provenance supplied externally",
            InputFile = inputFile, EmbeddingToken = embeddingToken, Arithmetic = arithmetic.ToString(), TokenRows = tokenRows,
            Observations = observations, Native = modules,
            ProbeSha256 = Hash(typeof(Probe).Assembly.Location), GgmlRevision = Environment.GetEnvironmentVariable("TS_VALIDATION_GGML_REVISION"),
            Scope = "Projection diagnostic, not model qualification or a process-memory budget test. Loads one original tensor at a time; keeps the full tensor in ordinary host memory for both arms. Streaming CUDA workspace alone is reserved; host inputs, resident weights/output and native GGML scratch/cache are outside this diagnostic reservation. Sampled double oracle decodes original F16/Q8_0 weights and keeps the same F32 inputs. No activation-precision registration, thresholds or resident-reference changes. Exit 0 means diagnostic completed with finite values and cleanup, not numerical equivalence; inspect each StrictGate." }, new JsonSerializerOptions { WriteIndented = true }));
        return error == null ? 0 : 2;
    }

    static float[] Inputs(GgufFile file, string model, int k, int n, string? path, int? token)
    {
        float[] values = new float[checked(k * n)];
        if (path != null)
        {
            byte[] bytes = File.ReadAllBytes(path);
            if (bytes.Length != checked(values.Length * sizeof(float))) throw new ArgumentException("Input file must contain exactly max(N)*K little-endian F32 values.");
            MemoryMarshal.Cast<byte, float>(bytes).CopyTo(values);
        }
        else if (token != null)
        {
            var embedding = file.Tensors["token_embd.weight"];
            if (embedding.Type != GgmlTensorType.Q8_0 || embedding.Shape[0] != (ulong)k || token < 0 || (ulong)token.Value >= embedding.Shape[1])
                throw new ArgumentException("Embedding token, width or type is invalid.");
            int rowBytes = checked(k / 32 * 34);
            byte[] row = new byte[rowBytes];
            using var source = File.OpenRead(model);
            source.Position = checked(file.DataOffset + (long)embedding.Offset + (long)token.Value * rowBytes);
            source.ReadExactly(row);
            fixed (byte* raw = row)
                for (int c = 0; c < n; c++)
                    for (int j = 0; j < k; j++) values[c * k + j] = (float)Decode(raw, 8, j) * MathF.Sqrt(k);
        }
        else
            for (int c = 0; c < n; c++)
                for (int j = 0; j < k; j++)
                    values[c * k + j] = MathF.Sin((j + 1) * 0.173f + c * 0.37f) * (0.2f + j % 13 * 0.21f)
                        + MathF.Cos((j + 1) * 0.071f - c * 0.19f) * 0.41f;
        if (values.Any(v => !float.IsFinite(v))) throw new InvalidDataException("Nonfinite input.");
        return values;
    }

    static OracleResult Oracle(IntPtr pointer, int type, int k, int rows, int n, float[] input, float[] resident, float[] streamed)
    {
        int[] sampleRows = Enumerable.Range(0, Math.Min(129, rows))
            .Select(i => checked((int)((long)i * (rows - 1) / Math.Max(1, Math.Min(129, rows) - 1)))).Distinct().ToArray();
        var reference = new double[sampleRows.Length * n];
        var residentSamples = new float[reference.Length];
        var streamedSamples = new float[reference.Length];
        long rowBytes = type == 8 ? k / 32L * 34 : k * 2L;
        for (int c = 0; c < n; c++)
        for (int ri = 0; ri < sampleRows.Length; ri++)
        {
            int row = sampleRows[ri], index = c * sampleRows.Length + ri;
            byte* raw = (byte*)pointer + row * rowBytes;
            double sum = 0;
            for (int j = 0; j < k; j++) sum += Decode(raw, type, j) * input[c * k + j];
            reference[index] = sum;
            residentSamples[index] = resident[c * rows + row];
            streamedSamples[index] = streamed[c * rows + row];
        }
        return new(sampleRows, reference.Length, OracleCompare(reference, residentSamples), OracleCompare(reference, streamedSamples));
    }

    static double Decode(byte* raw, int type, int j)
        => type == 1 ? (double)BitConverter.UInt16BitsToHalf(((ushort*)raw)[j])
            : (double)BitConverter.UInt16BitsToHalf(*(ushort*)(raw + (j / 32) * 34)) * *(sbyte*)(raw + (j / 32) * 34 + 2 + j % 32);

    static ErrorMetrics OracleCompare(double[] expected, float[] actual)
    {
        double ref2 = 0, err2 = 0, max = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            if (!double.IsFinite(expected[i]) || !float.IsFinite(actual[i])) throw new InvalidDataException("Nonfinite projection/oracle.");
            double error = actual[i] - expected[i];
            ref2 += expected[i] * expected[i]; err2 += error * error; max = Math.Max(max, Math.Abs(error));
        }
        return new(Math.Sqrt(err2 / Math.Max(ref2, 1e-300)), max);
    }

    static Metrics Compare(float[] expected, float[] actual, int width)
    {
        double ref2 = 0, act2 = 0, err2 = 0, dot = 0, max = 0, worstL2 = 0, minCos = 1;
        int topDifferences = 0, changed = 0;
        for (int start = 0; start < expected.Length; start += width)
        {
            double r2 = 0, a2 = 0, e2 = 0, d = 0;
            int refTop = start, actTop = start;
            for (int i = start; i < start + width; i++)
            {
                double a = expected[i], b = actual[i];
                if (!double.IsFinite(a) || !double.IsFinite(b)) throw new InvalidDataException("Nonfinite projection.");
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
        return new(Math.Sqrt(err2 / Math.Max(ref2, 1e-300)), ref2 == 0 && act2 == 0 ? 1 : dot / Math.Sqrt(Math.Max(ref2 * act2, 1e-300)), max,
            worstL2, minCos, topDifferences, changed, worstL2 <= 0.001 && minCos >= 0.999999 && topDifferences == 0);
    }

    static string Hash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)); }
    sealed record Metrics(double RelativeL2, double Cosine, double MaxAbsolute, double WorstTokenRelativeL2, double MinimumTokenCosine, int Top1Differences, int ChangedFloatBits, bool StrictGate);
    sealed record ErrorMetrics(double RelativeL2, double MaxAbsolute);
    sealed record OracleResult(int[] OutputRows, int Elements, ErrorMetrics Resident, ErrorMetrics Streamed);
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
