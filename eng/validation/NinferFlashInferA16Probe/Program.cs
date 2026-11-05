// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using R = TensorSharp.Validation.NinferFlashInferA16Reference;

internal static class Program
{
    private const int Guard = 256, N = CudaNinferFlashInferA16Module.OutputRows, K = CudaNinferFlashInferA16Module.Columns;
    private static readonly int[] Active = { 0, 1, 7, 8, 31, 32, 127, 128, 255, 256, N - 2, N - 1 };
    private static readonly string[] Families = { "signed-zero", "exact-lattice", "f32-output", "broad-finite" };
    private static readonly byte[] Poisons = { 0xc1, 0xd2, 0xc1 };
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
    private sealed record Fixture(string Id, string Family, int Rows, byte[] Codes, byte[] Scales, ushort[] Input, float Alpha, R.Oracle Oracle);
    private static byte[] Bytes<T>(T[] array) where T : unmanaged => MemoryMarshal.AsBytes(array.AsSpan()).ToArray();
    private static string Sha(ReadOnlySpan<byte> bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    private static void Need(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
    private static byte[] Guarded(byte[] body) { var result = Enumerable.Repeat((byte)0xa5, body.Length + 2 * Guard).ToArray(); body.CopyTo(result, Guard); return result; }
    private static bool Guards(byte[] body) => body.AsSpan(0, Guard).ToArray().All(x => x == 0xa5) && body.AsSpan(body.Length - Guard).ToArray().All(x => x == 0xa5);
    private static void NewFile(string path, byte[] bytes) { using var f = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None); f.Write(bytes); f.Flush(flushToDisk: true); }
    private static void Save(string path, object value) { using var f = File.Create(path); JsonSerializer.Serialize(f, value, Json); f.WriteByte(10); }
    private static void Complete(string directory, object value)
    {
        byte[] bytes = JsonSerializer.SerializeToUtf8Bytes(value, Json).Concat(new byte[] { 10 }).ToArray();
        NewFile(Path.Combine(directory, "completed-result.json"), bytes);
        NewFile(Path.Combine(directory, "completed-result.json.sha256"), System.Text.Encoding.ASCII.GetBytes(Sha(bytes) + "  completed-result.json\n"));
    }
    private static void Fresh(string path)
    {
        Need(Path.IsPathFullyQualified(path) && Path.GetFullPath(path) == path, "Canonical absolute output required.");
        string[] p = path.Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries);
        Need(p.Contains("artifacts") || Enumerable.Range(0, p.Length - 1).Any(i => p[i] == "docs" && p[i + 1] == "validation"), "Ignored evidence output required.");
        bool Link(string x) => new FileInfo(x).LinkTarget != null || new DirectoryInfo(x).LinkTarget != null;
        Need(!Path.Exists(path) && !Link(path), "Fresh output required.");
        for (string? parent = Path.GetDirectoryName(path); parent != null; parent = Path.GetDirectoryName(parent)) Need(!Link(parent) && !File.Exists(parent), "Aliased output parent refused.");
    }

    private static Fixture Make(string family, int rows)
    {
        _ = CudaNinferFlashInferA16Module.Plan(rows);
        byte[] codes = new byte[checked((int)CudaNinferFlashInferA16Module.CodeBytes)], scales = new byte[checked((int)CudaNinferFlashInferA16Module.ScaleBytes)];
        // Fixture writer uses independent five-axis scale serialization rather
        // than the oracle's logical-row offset formula.
        int at = 0;
        for (int tile = 0; tile < N / 128; tile++) for (int kg = 0; kg < K / 64; kg++)
            for (int low = 0; low < 32; low++) for (int quarter = 0; quarter < 4; quarter++) for (int g = 0; g < 4; g++)
            {
                int row = tile * 128 + quarter * 32 + low, group = kg * 4 + g;
                scales[at++] = family == "broad-finite" ? (byte)((row * 17 + group * 29 + group / 7) % 127) : (byte)0x38;
            }
        Need(at == scales.Length, "Scale serialization length differs.");
        foreach (int row in Active) for (int c = 0; c < K; c += 2)
        {
            int Code(int column) => family switch
            {
                "signed-zero" => ((row + column) & 1) * 8,
                "f32-output" => column == 0 ? 2 : 0,
                _ => (row * 7 + column * 11 + column / 16) & 15
            };
            codes[row * (K / 2) + c / 2] = (byte)(Code(c) | (Code(c + 1) << 4));
        }
        ushort[] input = new ushort[rows * K];
        for (int t = 0; t < rows; t++) for (int c = 0; c < K; c++)
        {
            float value = family == "f32-output" ? (c == 0 ? 1 : 0)
                : family == "broad-finite" ? (((c * 19 + t * 31 + c / 7) % 509) - 254) / 128f
                : (((c * 17 + t * 31) % 127) - 63) / 64f;
            input[t * K + c] = c % 251 == 0 && family != "f32-output" ? (ushort)0x8000 : R.Bf(value);
        }
        float alpha = family is "broad-finite" or "f32-output" ? .7f : .75f;
        var oracle = R.Compute(codes, scales, input, N, K, rows, alpha, Active, family);
        if (family == "f32-output") foreach (int row in Active) for (int t = 0; t < rows; t++)
            Need(oracle.Exact![t * N + row] == alpha && R.Float(R.Bf(alpha)) != alpha, "The unrounded F32 epilogue witness collapsed.");
        return new(family + "-t" + rows, family, rows, codes, scales, input, alpha, oracle);
    }

    private sealed unsafe class Memory : IDisposable
    {
        private readonly List<IntPtr> _owned = new();
        internal IntPtr Put(byte[] data) { CudaDriverApi.cuMemAlloc(out var pointer, (UIntPtr)data.Length).ThrowOnError(); _owned.Add(pointer); Write(pointer, data); return pointer; }
        internal static void Write(IntPtr pointer, byte[] data) { fixed (byte* source = data) CudaDriverApi.cuMemcpyHtoD(pointer, (IntPtr)source, (UIntPtr)data.Length).ThrowOnError(); }
        internal static byte[] Read(IntPtr pointer, int bytes) { byte[] result = new byte[bytes]; fixed (byte* target = result) CudaDriverApi.cuMemcpyDtoH((IntPtr)target, pointer, (UIntPtr)bytes).ThrowOnError(); return result; }
        public void Dispose() { foreach (var pointer in _owned) CudaDriverApi.cuMemFree(pointer).ThrowOnError(); }
    }

    private static void Run(CudaNinferFlashInferA16Module module, Fixture f, int lifetime, string folder,
        List<Dictionary<string, object?>> records, Dictionary<string, string> firstBody, Action checkpoint)
    {
        string id = "lifetime" + lifetime + "-" + f.Id;
        Need(records.All(x => x["id"]?.ToString() != id), "Duplicate lifecycle record.");
        var passes = new List<object>();
        var record = new Dictionary<string, object?> { ["id"] = id, ["case"] = f.Id, ["lifetime"] = lifetime, ["passed"] = false,
            ["family"] = f.Family, ["rows"] = f.Rows, ["imageSha256"] = module.ImageSha256ForRows(f.Rows), ["passes"] = passes,
            ["exactBitsRequired"] = f.Oracle.Exact != null, ["maximumBinaryLatticeUnits"] = f.Oracle.MaximumLatticeUnits,
            ["boundRule"] = "(1+u)*abs(alpha)*gamma_(K+64)*sumabs + u*abs(ideal) plus (1+u64)*abs(alpha)*gamma64_K*sumabs + u64*abs(ideal) for FP64 oracle accumulation/epilogue; u=2^-24,u64=2^-53; scoped normal finite fixtures; no BF16 output allowance" };
        records.Add(record); checkpoint();
        string weights = Path.Combine(folder, "weights-" + f.Family);
        if (!Directory.Exists(weights))
        {
            Directory.CreateDirectory(weights); NewFile(Path.Combine(weights, "original-codes.bin"), f.Codes); NewFile(Path.Combine(weights, "original-scales.bin"), f.Scales);
            NewFile(Path.Combine(weights, "sha256.json"), JsonSerializer.SerializeToUtf8Bytes(new { codes = Sha(f.Codes), scales = Sha(f.Scales) }, Json));
        }
        string fixture = Path.Combine(folder, f.Id);
        if (!Directory.Exists(fixture))
        {
            Directory.CreateDirectory(fixture); NewFile(Path.Combine(fixture, "input.bf16"), Bytes(f.Input)); NewFile(Path.Combine(fixture, "alpha.f32"), Bytes(new[] { f.Alpha }));
            NewFile(Path.Combine(fixture, "ideal.f64"), Bytes(f.Oracle.Ideal)); NewFile(Path.Combine(fixture, "bounds.f64"), Bytes(f.Oracle.Bounds));
            if (f.Oracle.Exact != null) NewFile(Path.Combine(fixture, "expected.f32"), Bytes(f.Oracle.Exact));
        }
        var inputs = new Dictionary<string, byte[]> { ["codes"] = f.Codes, ["scales"] = f.Scales, ["input"] = Bytes(f.Input), ["alpha"] = Bytes(new[] { f.Alpha }) };
        using var memory = new Memory();
        var pointers = inputs.ToDictionary(x => x.Key, x => memory.Put(Guarded(x.Value)));
        int outputBytes = f.Rows * N * 4;
        var output = memory.Put(Guarded(new byte[outputBytes]));
        string? first = null;
        for (int pass = 0; pass < Poisons.Length; pass++)
        {
            Memory.Write(output, Guarded(Enumerable.Repeat(Poisons[pass], outputBytes).ToArray()));
            module.ProjectRaw(pointers["codes"] + Guard, pointers["scales"] + Guard, pointers["input"] + Guard,
                output + Guard, pointers["alpha"] + Guard, f.Rows, IntPtr.Zero);
            byte[] actual = Memory.Read(output, outputBytes + 2 * Guard);
            string file = id + "-pass" + pass + ".guarded.bin"; NewFile(Path.Combine(folder, file), actual);
            var values = MemoryMarshal.Cast<byte, float>(actual.AsSpan(Guard, outputBytes));
            int nonfinite = 0, boundFailures = 0, different = 0; double maximum = 0, maxFraction = 0;
            for (int i = 0; i < values.Length; i++)
            {
                float value = values[i];
                if (!float.IsFinite(value)) { nonfinite++; continue; }
                double error = Math.Abs(value - f.Oracle.Ideal[i]); maximum = Math.Max(maximum, error);
                if (error > f.Oracle.Bounds[i]) boundFailures++;
                if (f.Oracle.Bounds[i] > 0) maxFraction = Math.Max(maxFraction, error / f.Oracle.Bounds[i]);
                if (f.Oracle.Exact != null && BitConverter.SingleToInt32Bits(value) != BitConverter.SingleToInt32Bits(f.Oracle.Exact[i])) different++;
            }
            bool immutable = inputs.All(x => Memory.Read(pointers[x.Key], x.Value.Length + 2 * Guard).AsSpan().SequenceEqual(Guarded(x.Value)));
            string bodySha = Sha(actual.AsSpan(Guard, outputBytes)); first ??= bodySha;
            bool repeat = bodySha == first, crossLifetime = lifetime == 0 || firstBody.TryGetValue(f.Id, out var previous) && previous == bodySha;
            if (lifetime == 0) firstBody.TryAdd(f.Id, bodySha);
            bool passed = nonfinite == 0 && boundFailures == 0 && different == 0 && Guards(actual) && immutable && repeat && crossLifetime;
            passes.Add(new { pass, poison = Poisons[pass], file, sha256 = Sha(actual), bodySha256 = bodySha, nonfinite, boundFailures,
                differentF32BitsFromExact = f.Oracle.Exact == null ? (int?)null : different, guardsIntact = Guards(actual), inputsImmutable = immutable,
                repeatedBits = repeat, crossLifetimeBits = lifetime == 0 ? (bool?)null : crossLifetime, maximumAbsoluteError = maximum, maximumBoundFraction = maxFraction, passed });
            checkpoint(); Need(passed, "Exact/bound/guard/immutability/repeat gate failed; complete raw output retained: " + id);
        }
        record["passed"] = true; checkpoint();
    }

    private static void SelfTest()
    {
        Need(R.ProveLocalWeightProducts() == 2032, "Local weight product proof count differs.");
        long checkedValues = 0;
        foreach (string family in Families) foreach (int rows in new[] { 1, 8 })
        {
            var f = Make(family, rows); checkedValues += f.Oracle.Ideal.Length;
            Need(f.Oracle.Ideal.Length == rows * N && f.Oracle.Bounds.All(x => double.IsFinite(x) && x >= 0), "CPU oracle incomplete.");
        }
        foreach (int unsupported in new[] { 0, 2, 3, 7, 9, 16 })
        {
            bool rejected = false; try { CudaNinferFlashInferA16Module.Plan(unsupported); } catch (ArgumentOutOfRangeException) { rejected = true; }
            Need(rejected, "The static AOT module must not silently pad or change physical shape.");
        }
        Console.WriteLine(JsonSerializer.Serialize(new { cpu = "PASS", localProducts = 2032, cases = 8, fullOutputValues = checkedValues,
            nativeCompilation = "NOT_RUN", GPU = "NOT_RUN", wholeModel = "NOT_RUN", quality = "NOT_RUN" }, Json));
    }

    internal static int Main(string[] args)
    {
        string? output = null; bool owned = false;
        var records = new List<Dictionary<string, object?>>(); var firstBodies = new Dictionary<string, string>();
        var state = new Dictionary<string, object?> { ["schema"] = "ninfer-flashinfer-native-w4a16-component/1", ["complete"] = false, ["GPU"] = "NOT_RUN",
            ["wholeModel"] = "NOT_RUN", ["modelQuality"] = "NOT_RUN", ["performance"] = "NOT_RUN", ["records"] = records };
        try
        {
            Need(BitConverter.IsLittleEndian, "Little-endian captures required.");
            if (args.SequenceEqual(new[] { "--self-test" })) { SelfTest(); return 0; }
            if (args.Length == 2 && args[0] == "--inspect-manifest")
            {
                var specs = CudaNinferFlashInferA16Module.ValidateManifest(File.ReadAllBytes(args[1]));
                Console.WriteLine(JsonSerializer.Serialize(new { manifestPolicy = "PASS", rows = specs.Select(s => s.Rows).ToArray(), GPU = "NOT_RUN" }, Json)); return 0;
            }
            if (args.Length == 0) { Console.WriteLine("PLAN: --execute --manifest ABSOLUTE_MODULE_JSON --out FRESH_IGNORED_DIRECTORY; --self-test is CPU-only."); return 0; }
            Need(args.Length == 5 && args[0] == "--execute" && args[1] == "--manifest" && args[3] == "--out", "Use --execute --manifest FILE --out FRESH_IGNORED_DIRECTORY.");
            string manifestPath = Path.GetFullPath(args[2]); Need(File.Exists(manifestPath), "AOT manifest is unavailable.");
            output = args[4]; Fresh(output); Directory.CreateDirectory(output); owned = true;
            byte[] manifest = File.ReadAllBytes(manifestPath); _ = CudaNinferFlashInferA16Module.ValidateManifest(manifest);
            state["manifestPath"] = manifestPath; state["manifestSha256"] = Sha(manifest); NewFile(Path.Combine(output, "module-input.json"), manifest);
            state["probeAssemblySha256"] = Sha(File.ReadAllBytes(typeof(Program).Assembly.Location)); state["backendAssemblySha256"] = Sha(File.ReadAllBytes(typeof(CudaBackend).Assembly.Location));
            state["localWeightProductsExactlyBf16"] = R.ProveLocalWeightProducts();
            state["scope"] = "Synthetic raw adjacent-K/M128x4 full N34816/K5120 T1/T8; F32 output and alpha-after-MMA; no target equivalence, model-quality or performance claim.";
            void Checkpoint() => Save(Path.Combine(output!, "result.json"), state); Checkpoint();
            CudaBackend.Register(); CudaDriverApi.cuInit(0).ThrowOnError();
            using (var context = CudaContext.Create(0))
            {
                state["GPU"] = "EXECUTED";
                CudaDriverApi.cuDriverGetVersion(out int driver).ThrowOnError(); CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError();
                byte[] name = new byte[256]; CudaDriverApi.cuDeviceGetName(name, name.Length, device).ThrowOnError();
                state["device"] = new { ordinal = device, name = System.Text.Encoding.UTF8.GetString(name).TrimEnd('\0'), driverVersion = driver };
                for (int lifetime = 0; lifetime < 2; lifetime++)
                {
                    using var module = new CudaNinferFlashInferA16Module(manifestPath);
                    Need(module.ManifestSha256 == state["manifestSha256"]!.ToString(), "The loaded AOT manifest changed.");
                    foreach (string family in Families) foreach (int rows in new[] { 1, 8 }) Run(module, Make(family, rows), lifetime, output, records, firstBodies, Checkpoint);
                }
            }
            Need(records.Count == 16 && records.Select(x => x["id"]!.ToString()).Distinct().Count() == 16
                && records.GroupBy(x => x["case"]!.ToString()).Count() == 8
                && records.All(x => (bool)x["passed"]! && ((List<object>)x["passes"]!).Count == 3), "All16 lifecycle records/8 cases/3 poison passes are mandatory.");
            Need(Sha(File.ReadAllBytes(manifestPath)) == state["manifestSha256"]!.ToString(), "Manifest changed during execution.");
            state["complete"] = true; state["passed"] = true; state["actualLifecycleRecords"] = records.Count; Checkpoint(); Complete(output, state);
            Console.WriteLine(JsonSerializer.Serialize(new { complete = true, passed = true, lifecycleRecords = records.Count, authoritativeResult = "completed-result.json", wholeModel = "NOT_RUN", modelQuality = "NOT_RUN", performance = "NOT_RUN" })); return 0;
        }
        catch (Exception error)
        {
            state["error"] = error.ToString(); state["passed"] = false;
            if (owned && output != null) Save(Path.Combine(output, "result.json"), state);
            Console.Error.WriteLine(error); return 1;
        }
    }
}
