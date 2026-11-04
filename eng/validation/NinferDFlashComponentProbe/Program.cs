// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using TensorSharp.Validation;
using R = TensorSharp.Validation.NinferDFlashReference;

internal static partial class Program
{
    private const int Guard = 256;
    private static readonly byte[] Poisons = { 0xc1, 0xd2, 0xc1 };
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
    private sealed record Expected(byte[]? Exact, double[]? Ideal = null, double[]? Bounds = null, int ScratchBytes = 0)
    {
        internal int Bytes => ScratchBytes > 0 ? ScratchBytes : Exact?.Length ?? checked(Ideal!.Length * 2);
    }
    private sealed record Case(string Name, Dictionary<string, byte[]> Inputs, Dictionary<string, Expected> Outputs,
        Action<CudaNinferDraftModule, Dictionary<string, IntPtr>> Run, object Contract);
    private static byte[] Bytes<T>(T[] array) where T : unmanaged => MemoryMarshal.AsBytes(array.AsSpan()).ToArray();
    private static string Sha(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    private static void Need(bool ok, string message) { if (!ok) throw new InvalidDataException(message); }
    private static ushort[] Values(int n, int salt = 0) => Enumerable.Range(0, n).Select(i => R.BFloat((((i * 19 + salt * 23 + i / 7) % 127) - 63) / 32f)).ToArray();
    private static byte[] Guarded(byte[] body) { var b = Enumerable.Repeat((byte)0xa5, checked(body.Length + 2 * Guard)).ToArray(); body.CopyTo(b, Guard); return b; }
    private static void Save(string path, object result) { using var stream = File.Create(path); JsonSerializer.Serialize(stream, result, Json); stream.WriteByte(10); }
    private static void SaveCompleted(string folder, object result)
    {
        const string name = "completed-result.json";
        byte[] json = JsonSerializer.SerializeToUtf8Bytes(result, Json);
        byte[] complete = new byte[json.Length + 1]; json.CopyTo(complete, 0); complete[^1] = 10;
        static void WriteNew(string path, byte[] bytes)
        {
            using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
            stream.Write(bytes); stream.Flush(flushToDisk: true);
        }
        WriteNew(Path.Combine(folder, name), complete);
        WriteNew(Path.Combine(folder, name + ".sha256"), System.Text.Encoding.ASCII.GetBytes(Sha(complete) + "  " + name + "\n"));
    }
    private static void Fresh(string path)
    {
        Need(Path.IsPathFullyQualified(path) && Path.GetFullPath(path) == path, "Canonical absolute output required.");
        string[] parts = path.Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries);
        Need(parts.Contains("artifacts", StringComparer.Ordinal) || Enumerable.Range(0, parts.Length - 1).Any(i => parts[i] == "docs" && parts[i + 1] == "validation"), "Ignored evidence directory required.");
        bool Link(string p) => new FileInfo(p).LinkTarget != null || new DirectoryInfo(p).LinkTarget != null;
        Need(!Path.Exists(path) && !Link(path), "Fresh evidence output required.");
        for (string? p = Path.GetDirectoryName(path); p != null; p = Path.GetDirectoryName(p)) Need(!Link(p) && !File.Exists(p), "Aliased output parent refused.");
    }
    private sealed unsafe class Memory : IDisposable
    {
        private readonly List<IntPtr> _owned = new();
        internal IntPtr Put(byte[] bytes) { CudaDriverApi.cuMemAlloc(out var p, (UIntPtr)bytes.Length).ThrowOnError(); _owned.Add(p); Write(p, bytes); return p; }
        internal static void Write(IntPtr p, byte[] b) { fixed (byte* v = b) CudaDriverApi.cuMemcpyHtoD(p, (IntPtr)v, (UIntPtr)b.Length).ThrowOnError(); }
        internal static byte[] Read(IntPtr p, int n) { byte[] b = new byte[n]; fixed (byte* v = b) CudaDriverApi.cuMemcpyDtoH((IntPtr)v, p, (UIntPtr)n).ThrowOnError(); return b; }
        public void Dispose() { foreach (var p in _owned) CudaDriverApi.cuMemFree(p).ThrowOnError(); }
    }
    private static void RunCase(CudaNinferDraftModule module, Case c, int lifetime, string folder, List<object> records, Action checkpoint)
    {
        string id = $"lifetime{lifetime}-{c.Name}";
        var passes = new List<object>();
        var record = new Dictionary<string, object?> { ["id"] = id, ["case"] = c.Name, ["lifetime"] = lifetime, ["passed"] = false, ["contract"] = c.Contract,
            ["inputs"] = c.Inputs.ToDictionary(p => p.Key, p => new { bytes = p.Value.Length, sha256 = Sha(p.Value) }), ["passes"] = passes };
        records.Add(record); checkpoint();
        using var memory = new Memory();
        var pointers = new Dictionary<string, IntPtr>();
        foreach (var p in c.Inputs) pointers[p.Key] = memory.Put(Guarded(p.Value));
        foreach (var p in c.Outputs)
        {
            if (!pointers.ContainsKey(p.Key)) pointers[p.Key] = memory.Put(Guarded(new byte[p.Value.Bytes]));
            if (p.Value.Exact != null) File.WriteAllBytes(Path.Combine(folder, id + "-" + p.Key + "-expected.bin"), p.Value.Exact);
            if (p.Value.Ideal != null) { File.WriteAllBytes(Path.Combine(folder, id + "-" + p.Key + "-ideal.f64"), Bytes(p.Value.Ideal)); File.WriteAllBytes(Path.Combine(folder, id + "-" + p.Key + "-bound.f64"), Bytes(p.Value.Bounds!)); }
        }
        var bodyPointers = pointers.ToDictionary(p => p.Key, p => p.Value + Guard);
        var first = new Dictionary<string, byte[]>();
        foreach (byte poison in Poisons)
        {
            int pass = passes.Count;
            foreach (var p in pointers)
            {
                byte[] body = c.Inputs.TryGetValue(p.Key, out var initial) ? initial : Enumerable.Repeat(poison, c.Outputs[p.Key].Bytes).ToArray();
                Memory.Write(p.Value, Guarded(body));
            }
            c.Run(module, bodyPointers);
            var outputs = new List<object>();
            bool all = true;
            foreach (var p in c.Outputs)
            {
                byte[] actual = Memory.Read(pointers[p.Key], p.Value.Bytes + 2 * Guard);
                string file = id + $"-pass{pass}-" + p.Key + ".guarded.bin"; File.WriteAllBytes(Path.Combine(folder, file), actual);
                var body = actual.AsSpan(Guard, p.Value.Bytes);
                int different = p.Value.Exact == null ? 0 : body.SequenceEqual(p.Value.Exact) ? 0 : body.ToArray().Zip(p.Value.Exact).Count(x => x.First != x.Second);
                int boundFailures = 0, nonfinite = 0; double maxError = 0, maxBoundFraction = 0;
                if (p.Value.Ideal != null) for (int i = 0; i < p.Value.Ideal.Length; i++)
                {
                    float value = R.Float(BinaryPrimitives.ReadUInt16LittleEndian(body.Slice(i * 2)));
                    if (!float.IsFinite(value)) { nonfinite++; continue; }
                    double error = Math.Abs(value - p.Value.Ideal[i]); maxError = Math.Max(maxError, error);
                    if (error > p.Value.Bounds![i]) boundFailures++;
                    if (p.Value.Bounds![i] > 0) maxBoundFraction = Math.Max(maxBoundFraction, error / p.Value.Bounds![i]);
                }
                bool guards = actual.AsSpan(0, Guard).ToArray().All(x => x == 0xa5) && actual.AsSpan(actual.Length - Guard).ToArray().All(x => x == 0xa5);
                bool repeat = p.Value.ScratchBytes > 0 || !first.TryGetValue(p.Key, out var earlier) || body.SequenceEqual(earlier); if (p.Value.ScratchBytes == 0) first.TryAdd(p.Key, body.ToArray());
                all &= different == 0 && boundFailures == 0 && nonfinite == 0 && guards && repeat;
                outputs.Add(new { name = p.Key, file, sha256 = Sha(actual), differentBytesFromExact = p.Value.Exact == null ? (int?)null : different, boundFailures, nonfinite, guardsIntact = guards, repeatBits = p.Value.ScratchBytes > 0 ? (bool?)null : repeat, scratchGuardsOnly = p.Value.ScratchBytes > 0, maxAbsoluteError = maxError, maxBoundFraction });
            }
            bool immutable = true;
            foreach (var p in c.Inputs.Where(p => !c.Outputs.ContainsKey(p.Key))) immutable &= Memory.Read(pointers[p.Key], p.Value.Length + 2 * Guard).AsSpan().SequenceEqual(Guarded(p.Value));
            passes.Add(new { pass, poison, outputs, immutableInputAllocations = immutable, passed = all && immutable }); checkpoint();
            Need(all && immutable, id + ": exact/bound/guard/immutability/repeat gate failed; full outputs retained.");
        }
        record["passed"] = true; checkpoint();
    }
    private static IEnumerable<Case> CoreCases()
    {
        foreach (int width in new[] { 32, 5120 }) foreach (int block in new[] { 1, 2, 8 }) foreach (int lanes in new[] { 1, 2, 8 }) foreach (int side in new[] { 0, 1 })
        {
            var x = Values(width * block * lanes, 1); var dynamic = Values(width / 16 * 4 * block * lanes, 2); var basis = Values(width * 4, 3);
            yield return new($"conv-h{width}-b{block}-l{lanes}-s{side}", new() { ["x"] = Bytes(x), ["dynamic"] = Bytes(dynamic), ["base"] = Bytes(basis) }, new() { ["y"] = new(Bytes(R.DynamicConv(x, dynamic, basis, width, block, lanes, side))) },
                (m, p) => m.DynamicConv(p["x"], p["dynamic"], p["base"], p["y"], width, block, lanes, side, IntPtr.Zero), new { oracle = "original raw side/tap/channel and group16; two ordered FMA; final BF16", exactBits = true });
        }
        foreach (int count in new[] { 1, 128, 5120, 40960 })
        {
            var input = Values(count, 7); var residual = Values(count, 9);
            yield return new($"residual-n{count}", new() { ["input"] = Bytes(input), ["residual"] = Bytes(residual) }, new() { ["residual"] = new(Bytes(R.ResidualAdd(input, residual))) },
                (m, p) => m.ResidualAdd(p["input"], p["residual"], count, IntPtr.Zero), new { oracle = "BF16(F32(input)+F32(residual))", exactBits = true });
        }
        foreach (int width in new[] { 128, 5120 }) foreach (int rows in new[] { 1, 7, 32 })
        {
            var input = Values(width * rows, 11); var weight = Values(width, 13); float eps = 1e-6f;
            double[] ideal = R.PlainNormIdeal(input, weight, width, rows, eps), bounds = ideal.Select(x => R.NormBound(x, width)).ToArray();
            yield return new($"plain-rms-d{width}-r{rows}", new() { ["input"] = Bytes(input), ["weight"] = Bytes(weight) }, new() { ["y"] = new(null, ideal, bounds) },
                (m, p) => m.PlainRms(p["input"], p["weight"], p["y"], width, rows, eps, IntPtr.Zero), new { oracle = "FP64 RMSNorm with plain BF16 weight; no +1", bound = "abs(ideal)*(2^-8+(width+32)*2^-24)", exactBits = false });
        }
        foreach (int tokens in new[] { 1, 7, 8, 32 }) foreach (int mode in new[] { 0, 1, 2 })
        {
            int[] positions = Enumerable.Range(0, tokens).Select(i => i == 0 ? 0 : new[] { 31, 32767, 131071, 249999, 1048576, 4000001 }[(i - 1) % 6]).ToArray();
            double[] frequencies = Enumerable.Range(0, 64).Select(i => Math.Pow(10000000.0, -2.0 * i / 128)).ToArray();
            var inputs = new Dictionary<string, byte[]> { ["positions"] = Bytes(positions), ["frequencies"] = Bytes(frequencies) }; var outputs = new Dictionary<string, Expected>();
            foreach (var (name, heads, enabled) in new[] { ("q", 32, mode != 1), ("k", 8, mode != 0) }) if (enabled)
            {
                var values = Values(tokens * heads * 128, heads); var ideal = R.RopeIdeal(values, heads, positions, frequencies); var bounds = new double[ideal.Length];
                for (int i = 0; i < ideal.Length; i++) { int partner = i % 128 < 64 ? i + 64 : i - 64; bounds[i] = R.RopeBound(ideal[i], Math.Abs(R.Float(values[i])) + Math.Abs(R.Float(values[partner]))); }
                inputs[name] = Bytes(values); outputs[name] = new(tokens == 1 ? Bytes(values) : null, ideal, bounds);
            }
            yield return new($"rope-t{tokens}-mode{mode}", inputs, outputs, (m, p) => m.QkRope(p.GetValueOrDefault("q"), p.GetValueOrDefault("k"), p["positions"], p["frequencies"], tokens, IntPtr.Zero),
                new { oracle = "FP64 theta1e7 frequencies and range reduction, ideal trigonometry at reduced F32 angle", bound = "2^-8*abs(ideal)+32*2^-24*(abs(first)+abs(second))", positionZeroExact = tokens == 1 });
        }
    }
    private static IEnumerable<Case> AllCases() => CoreCases().Concat(ProjectionCases()).Concat(SelectorAndRingCases()).Concat(SwaCases()).Concat(SwiGluCases());
    private static void SelfTestAll()
    {
        var names = new HashSet<string>(); long exactBytes = 0, boundedValues = 0;
        foreach (var c in AllCases())
        {
            Need(names.Add(c.Name), "Duplicate component fixture name.");
            foreach (var e in c.Outputs.Values)
            {
                Need(e.Bytes > 0, "Empty output contract.");
                if (e.Exact != null) exactBytes += e.Exact.Length;
                if (e.Ideal != null)
                {
                    Need(e.Bounds != null && e.Bounds.Length == e.Ideal.Length, "Bound geometry differs.");
                    for (int i = 0; i < e.Ideal.Length; i++)
                    {
                        Need(double.IsFinite(e.Ideal[i]) && double.IsFinite(e.Bounds![i]) && e.Bounds![i] >= 0, "Nonfinite CPU oracle/bound.");
                        if (e.Exact != null) Need(Math.Abs(R.Float(BinaryPrimitives.ReadUInt16LittleEndian(e.Exact.AsSpan(i * 2))) - e.Ideal[i]) <= e.Bounds![i], "Exact CPU output exceeds declared ideal bound.");
                    }
                    boundedValues += e.Ideal.Length;
                }
            }
        }
        Need(names.Count == 199, "Full component fixture coverage changed.");
        Console.WriteLine(JsonSerializer.Serialize(new { CPU = "PASS", constructedCases = names.Count, exactOracleBytes = exactBytes, boundedOracleValues = boundedValues, GPU = "NOT_RUN", model = "NOT_RUN", performance = "NOT_RUN" }));
    }
    private static void SelfTest()
    {
        const int n = 128, k = 5120;
        var codes = Enumerable.Repeat((byte)0x22, n * k / 2).ToArray(); var scales = Enumerable.Repeat((byte)0x38, n * k / 16).ToArray();
        var input = new ushort[k]; for (int c = 0; c < 512; c++) input[c] = R.BFloat((c % 4) switch { 0 => 16777216f, 2 => -16777216f, _ => 1f });
        foreach (int t in new[] { 1, 2, 4, 8, 16, 17, 20, 21, 32 }) Need(R.ProjectRow(codes, scales, input, 127, k, t) == (t == 1 ? 128 : 32), "Original schedule analytical cancellation differs.");
        Need(R.BFloat(-0f) == 0x8000 && BitConverter.SingleToInt32Bits(R.Code(8)) == unchecked((int)0x80000000), "Signed zero changed.");
        var cases = CoreCases().ToArray();
        Need(cases.Length == 58 && cases.All(c => c.Outputs.Values.All(e => e.Exact != null || e.Ideal!.Zip(e.Bounds!).All(p => double.IsFinite(p.First) && double.IsFinite(p.Second) && p.Second >= 0))), "CPU oracle/case construction failed.");
        Console.WriteLine(JsonSerializer.Serialize(new { CPU = "PASS", coreCases = cases.Length, analyticalA16Schedules = 9, GPU = "NOT_RUN", model = "NOT_RUN", performance = "NOT_RUN" }));
    }
    public static int Main(string[] args)
    {
        string? output = null; bool owned = false; var records = new List<object>();
        var state = new Dictionary<string, object?> { ["schema"] = "ninfer-dflash-component-probe/1", ["complete"] = false, ["GPU"] = "NOT_RUN", ["modelQuality"] = "NOT_RUN", ["servingIntegration"] = "NOT_RUN", ["performance"] = "NOT_RUN", ["records"] = records };
        try
        {
            Need(BitConverter.IsLittleEndian, "Little-endian binary capture format required.");
            if (args.SequenceEqual(new[] { "--self-test-all" })) { SelfTestAll(); return 0; }
            if (args.SequenceEqual(new[] { "--self-test" })) { SelfTest(); return 0; }
            if (args.Length == 0) { Console.WriteLine("PLAN: --execute --module ABSOLUTE_MODULE --out FRESH_IGNORED_DIRECTORY; --self-test CPU only."); return 0; }
            Need(args.Length == 5 && args[0] == "--execute" && args[1] == "--module" && args[3] == "--out", "Use --execute --module FILE --out FRESH_IGNORED_DIRECTORY.");
            string path = Path.GetFullPath(args[2]); output = args[4]; Fresh(output); Need(File.Exists(path), "Native module unavailable.");
            Directory.CreateDirectory(output); owned = true; state["modulePath"] = path; state["moduleSha256"] = Sha(File.ReadAllBytes(path));
            state["probeAssemblySha256"] = Sha(File.ReadAllBytes(typeof(Program).Assembly.Location)); state["backendAssemblySha256"] = Sha(File.ReadAllBytes(typeof(CudaBackend).Assembly.Location));
            state["uniqueCases"] = 199; state["moduleLifetimes"] = 2; state["poisonPassesPerCase"] = 3;
            state["scope"] = "Synthetic component contracts; raw BF16 and original NVFP4 schedules. No drafter proposals/model/cache integration qualification.";
            void Checkpoint() => Save(Path.Combine(output!, "result.json"), state); Checkpoint();
            CudaBackend.Register(); CudaDriverApi.cuInit(0).ThrowOnError(); using var context = CudaContext.Create(0); state["GPU"] = "EXECUTED";
            for (int lifetime = 0; lifetime < 2; lifetime++)
            {
                using var module = new CudaNinferDraftModule(path); Need(ValidationSha256.Matches(module.ImageSha256, state["moduleSha256"]!.ToString()!), "Loaded module image differs.");
                foreach (var c in AllCases()) RunCase(module, c, lifetime, output, records, Checkpoint);
            }
            Need(Sha(File.ReadAllBytes(path)) == state["moduleSha256"]?.ToString(), "Module file changed during execution.");
            state["complete"] = true; state["passed"] = true; Checkpoint(); SaveCompleted(output, state); Console.WriteLine(JsonSerializer.Serialize(new { complete = true, records = records.Count, modelQuality = "NOT_RUN", servingIntegration = "NOT_RUN", performance = "NOT_RUN", output, authoritativeResult = "completed-result.json" })); return 0;
        }
        catch (Exception e) { state["error"] = e.ToString(); state["passed"] = false; if (owned && output != null) Save(Path.Combine(output, "result.json"), state); Console.Error.WriteLine(e); return 1; }
    }
}
