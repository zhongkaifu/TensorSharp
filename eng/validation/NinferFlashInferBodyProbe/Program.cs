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
    private const int Guard = 256;
    private sealed record Shape(string Name, int N, int K);
    private static readonly Shape[] Shapes = { new("gate-up", 34816, 5120), new("down", 5120, 17408), new("output", 5120, 6144), new("gqa-input", 14336, 5120) };
    private static readonly string[] Families = { "signed-zero", "exact-lattice", "f32-output", "broad-finite" };
    private static readonly byte[] Poisons = { 0xc1, 0xd2, 0xc1 };
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
    private sealed record Payload(byte[] Codes, byte[] Scales, int[] ActiveRows);
    private sealed record Fixture(string Id, Shape Shape, string Family, int Rows, Payload Payload, ushort[] Input, float Alpha, R.Oracle Oracle);
    private static byte[] Bytes<T>(T[] array) where T : unmanaged => MemoryMarshal.AsBytes(array.AsSpan()).ToArray();
    private static string Sha(ReadOnlySpan<byte> bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    private static void Need(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
    private static byte[] Guarded(byte[] body) { var result = Enumerable.Repeat((byte)0xa5, body.Length + 2 * Guard).ToArray(); body.CopyTo(result, Guard); return result; }
    private static bool Guards(byte[] all) => all.AsSpan(0, Guard).ToArray().All(x => x == 0xa5) && all.AsSpan(all.Length - Guard).ToArray().All(x => x == 0xa5);
    private static void NewFile(string path, byte[] bytes) { using var f = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None); f.Write(bytes); f.Flush(flushToDisk: true); }
    private static void Save(string path, object value) { using var f = File.Create(path); JsonSerializer.Serialize(f, value, Json); f.WriteByte(10); }
    private static void Complete(string folder, object value)
    {
        byte[] bytes = JsonSerializer.SerializeToUtf8Bytes(value, Json).Concat(new byte[] { 10 }).ToArray();
        NewFile(Path.Combine(folder, "completed-result.json"), bytes);
        NewFile(Path.Combine(folder, "completed-result.json.sha256"), System.Text.Encoding.ASCII.GetBytes(Sha(bytes) + "  completed-result.json\n"));
    }
    private static void Fresh(string path)
    {
        Need(Path.IsPathFullyQualified(path) && Path.GetFullPath(path) == path, "Canonical absolute output required.");
        string[] parts = path.Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries);
        Need(parts.Contains("artifacts") || Enumerable.Range(0, parts.Length - 1).Any(i => parts[i] == "docs" && parts[i + 1] == "validation"), "Ignored evidence output required.");
        bool Link(string p) => new FileInfo(p).LinkTarget != null || new DirectoryInfo(p).LinkTarget != null;
        Need(!Path.Exists(path) && !Link(path), "Fresh evidence output required.");
        for (string? parent = Path.GetDirectoryName(path); parent != null; parent = Path.GetDirectoryName(parent)) Need(!Link(parent) && !File.Exists(parent), "Aliased evidence parent refused.");
    }

    private static Payload MakePayload(Shape shape, string family)
    {
        int n = shape.N, k = shape.K;
        var active = new List<int> { 0, 1, 7, 8, 31, 32, 63, 64, 127, 128, 255, 256, n / 2 - 1, n / 2, n - 2, n - 1 };
        if (n == 14336) active.AddRange(new[] { 6143, 6144, 7167, 7168, 13311, 13312 });
        int[] selected = active.Distinct().Order().ToArray();
        byte[] codes = new byte[checked(n * k / 2)], scales = new byte[checked(n * k / 16)];
        // Independent physical five-axis serialization; the oracle separately
        // computes a logical original-scale offset from row and group.
        int at = 0;
        for (int tile = 0; tile < n / 128; tile++) for (int kg = 0; kg < k / 64; kg++)
            for (int low = 0; low < 32; low++) for (int quarter = 0; quarter < 4; quarter++) for (int g = 0; g < 4; g++)
            {
                int row = tile * 128 + quarter * 32 + low, group = kg * 4 + g;
                scales[at++] = family == "broad-finite" ? (byte)((row * 17 + group * 29 + group / 7) % 127) : (byte)0x38;
            }
        Need(at == scales.Length, "Original scale byte count differs.");
        foreach (int row in selected) for (int c = 0; c < k; c += 2)
        {
            int Code(int column) => family switch { "signed-zero" => ((row + column) & 1) * 8, "f32-output" => column == 0 ? 2 : 0,
                _ => (row * 7 + column * 11 + column / 16) & 15 };
            codes[row * (k / 2) + c / 2] = (byte)(Code(c) | (Code(c + 1) << 4));
        }
        return new(codes, scales, selected);
    }

    private static Fixture Make(Shape shape, string family, int rows, Payload payload)
    {
        var layout = CudaNinferFlashInferBodyModule.Plan(shape.N, shape.K, rows);
        Need(payload.Codes.LongLength == layout.CodeBytes && payload.Scales.LongLength == layout.ScaleBytes, "Incomplete original body payload.");
        ushort[] input = new ushort[rows * shape.K];
        for (int t = 0; t < rows; t++) for (int c = 0; c < shape.K; c++)
        {
            float value = family == "f32-output" ? (c == 0 ? 1 : 0)
                : family == "broad-finite" ? (((c * 19 + t * 31 + c / 7) % 509) - 254) / 128f
                : (((c * 17 + t * 31) % 127) - 63) / 64f;
            input[t * shape.K + c] = c % 251 == 0 && family != "f32-output" ? (ushort)0x8000 : R.Bf(value);
        }
        float alpha = family is "broad-finite" or "f32-output" ? .7f : .75f;
        var oracle = R.Compute(payload.Codes, payload.Scales, input, shape.N, shape.K, rows, alpha, payload.ActiveRows, family);
        if (family == "f32-output") foreach (int row in payload.ActiveRows) for (int t = 0; t < rows; t++)
            Need(oracle.Exact![t * shape.N + row] == alpha && R.Float(R.Bf(alpha)) != alpha, "The unrounded output witness collapsed.");
        return new(shape.Name + "-" + family + "-t" + rows, shape, family, rows, payload, input, alpha, oracle);
    }

    private sealed unsafe class Memory : IDisposable
    {
        private readonly List<IntPtr> _owned = new();
        internal IntPtr Put(byte[] bytes) { CudaDriverApi.cuMemAlloc(out var pointer, (UIntPtr)bytes.Length).ThrowOnError(); _owned.Add(pointer); Write(pointer, bytes); return pointer; }
        internal static void Write(IntPtr pointer, byte[] bytes) { fixed (byte* source = bytes) CudaDriverApi.cuMemcpyHtoD(pointer, (IntPtr)source, (UIntPtr)bytes.Length).ThrowOnError(); }
        internal static byte[] Read(IntPtr pointer, int bytes) { byte[] result = new byte[bytes]; fixed (byte* target = result) CudaDriverApi.cuMemcpyDtoH((IntPtr)target, pointer, (UIntPtr)bytes).ThrowOnError(); return result; }
        public void Dispose() { foreach (var pointer in _owned) CudaDriverApi.cuMemFree(pointer).ThrowOnError(); }
    }

    private static void Run(CudaNinferFlashInferBodyModule module, Fixture fixture, int lifetime, string folder,
        List<Dictionary<string, object?>> records, Dictionary<string, string> firstBodies, Action checkpoint)
    {
        string id = "lifetime" + lifetime + "-" + fixture.Id;
        Need(records.All(x => x["id"]?.ToString() != id), "Duplicate lifecycle identity.");
        var passes = new List<object>();
        var record = new Dictionary<string, object?> { ["id"] = id, ["case"] = fixture.Id, ["lifetime"] = lifetime, ["shape"] = fixture.Shape.Name,
            ["family"] = fixture.Family, ["n"] = fixture.Shape.N, ["k"] = fixture.Shape.K, ["rows"] = fixture.Rows, ["passed"] = false,
            ["imageSha256"] = module.ImageSha256For(fixture.Shape.N, fixture.Shape.K, fixture.Rows), ["exactBitsRequired"] = fixture.Oracle.Exact != null,
            ["maximumBinaryLatticeUnits"] = fixture.Oracle.MaximumLatticeUnits, ["activeRows"] = fixture.Payload.ActiveRows, ["passes"] = passes };
        records.Add(record); checkpoint();
        string weights = Path.Combine(folder, "weights-" + fixture.Shape.Name + "-" + fixture.Family);
        if (!Directory.Exists(weights))
        {
            Directory.CreateDirectory(weights); NewFile(Path.Combine(weights, "original-codes.bin"), fixture.Payload.Codes); NewFile(Path.Combine(weights, "original-scales.bin"), fixture.Payload.Scales);
            NewFile(Path.Combine(weights, "sha256.json"), JsonSerializer.SerializeToUtf8Bytes(new { codes = Sha(fixture.Payload.Codes), scales = Sha(fixture.Payload.Scales) }, Json));
        }
        string fixtureFolder = Path.Combine(folder, fixture.Id);
        if (!Directory.Exists(fixtureFolder))
        {
            Directory.CreateDirectory(fixtureFolder); NewFile(Path.Combine(fixtureFolder, "input.bf16"), Bytes(fixture.Input)); NewFile(Path.Combine(fixtureFolder, "alpha.f32"), Bytes(new[] { fixture.Alpha }));
            NewFile(Path.Combine(fixtureFolder, "ideal.f64"), Bytes(fixture.Oracle.Ideal)); NewFile(Path.Combine(fixtureFolder, "bounds.f64"), Bytes(fixture.Oracle.Bounds));
            if (fixture.Oracle.Exact != null) NewFile(Path.Combine(fixtureFolder, "expected.f32"), Bytes(fixture.Oracle.Exact));
        }
        var inputs = new Dictionary<string, byte[]> { ["codes"] = fixture.Payload.Codes, ["scales"] = fixture.Payload.Scales,
            ["input"] = Bytes(fixture.Input), ["alpha"] = Bytes(new[] { fixture.Alpha }) };
        using var memory = new Memory();
        var pointers = inputs.ToDictionary(x => x.Key, x => memory.Put(Guarded(x.Value)));
        int outputBytes = checked(fixture.Rows * fixture.Shape.N * 4);
        var output = memory.Put(Guarded(new byte[outputBytes])); string? first = null;
        for (int pass = 0; pass < Poisons.Length; pass++)
        {
            Memory.Write(output, Guarded(Enumerable.Repeat(Poisons[pass], outputBytes).ToArray()));
            module.ProjectRaw(pointers["codes"] + Guard, pointers["scales"] + Guard, pointers["input"] + Guard, output + Guard,
                pointers["alpha"] + Guard, fixture.Shape.N, fixture.Shape.K, fixture.Rows, IntPtr.Zero);
            byte[] actual = Memory.Read(output, outputBytes + 2 * Guard);
            string file = id + "-pass" + pass + ".guarded.bin"; NewFile(Path.Combine(folder, file), actual);
            var values = MemoryMarshal.Cast<byte, float>(actual.AsSpan(Guard, outputBytes));
            int nonfinite = 0, boundFailures = 0, different = 0; double maxError = 0, maxFraction = 0;
            for (int i = 0; i < values.Length; i++)
            {
                float value = values[i];
                if (!float.IsFinite(value)) { nonfinite++; continue; }
                double error = Math.Abs(value - fixture.Oracle.Ideal[i]); maxError = Math.Max(maxError, error);
                if (error > fixture.Oracle.Bounds[i]) boundFailures++;
                if (fixture.Oracle.Bounds[i] > 0) maxFraction = Math.Max(maxFraction, error / fixture.Oracle.Bounds[i]);
                if (fixture.Oracle.Exact != null && BitConverter.SingleToInt32Bits(value) != BitConverter.SingleToInt32Bits(fixture.Oracle.Exact[i])) different++;
            }
            bool immutable = inputs.All(x => Memory.Read(pointers[x.Key], x.Value.Length + 2 * Guard).AsSpan().SequenceEqual(Guarded(x.Value)));
            string bodySha = Sha(actual.AsSpan(Guard, outputBytes)); first ??= bodySha;
            bool repeat = bodySha == first, cross = lifetime == 0 || firstBodies.TryGetValue(fixture.Id, out var previous) && previous == bodySha;
            if (lifetime == 0) firstBodies.TryAdd(fixture.Id, bodySha);
            bool passed = nonfinite == 0 && boundFailures == 0 && different == 0 && Guards(actual) && immutable && repeat && cross;
            passes.Add(new { pass, poison = Poisons[pass], file, sha256 = Sha(actual), bodySha256 = bodySha, nonfinite, boundFailures,
                differentF32BitsFromExact = fixture.Oracle.Exact == null ? (int?)null : different, guardsIntact = Guards(actual), inputsImmutable = immutable,
                repeatedBits = repeat, crossLifetimeBits = lifetime == 0 ? (bool?)null : cross, maximumAbsoluteError = maxError, maximumBoundFraction = maxFraction, passed });
            checkpoint(); Need(passed, "Original exact/bound/guard/immutability/repeat gate failed; complete output retained: " + id);
        }
        record["passed"] = true; checkpoint();
    }

    private static void SelfTest()
    {
        Need(R.ProveLocalWeightProducts() == 2032, "Original local product proof count differs.");
        long values = 0; int cases = 0; var indices = new HashSet<int>();
        foreach (var shape in Shapes) foreach (string family in Families)
        {
            var payload = MakePayload(shape, family);
            for (int rows = 1; rows <= 8; rows++)
            {
                var fixture = Make(shape, family, rows, payload); values += fixture.Oracle.Ideal.Length; cases++;
                Need(fixture.Oracle.Bounds.All(x => double.IsFinite(x) && x >= 0), "Nonfinite CPU body bound.");
                indices.Add(CudaNinferFlashInferBodyModule.Plan(shape.N, shape.K, rows).Index);
            }
        }
        Need(cases == 128 && indices.Count == 32 && values == 8552448, "The full fixed body CPU coverage differs.");
        Console.WriteLine(JsonSerializer.Serialize(new { CPU = "PASS", cases, uniqueSpecializations = indices.Count, fullOutputValues = values, localProducts = 2032,
            nativeCompilation = "NOT_RUN", GPU = "NOT_RUN", wholeModel = "NOT_RUN", quality = "NOT_RUN" }, Json));
    }

    internal static int Main(string[] args)
    {
        string? output = null; bool owned = false;
        var records = new List<Dictionary<string, object?>>(); var firstBodies = new Dictionary<string, string>();
        var state = new Dictionary<string, object?> { ["schema"] = "ninfer-flashinfer-body-component/1", ["complete"] = false, ["GPU"] = "NOT_RUN", ["wholeModel"] = "NOT_RUN",
            ["modelQuality"] = "NOT_RUN", ["performance"] = "NOT_RUN", ["records"] = records,
            ["boundRule"] = "Frozen independent BF16/rawNV4 FP64 oracle: (1+u)*abs(alpha)*gamma_(K+64)*sumabs+u*abs(ideal), plus separate FP64 oracle accumulation/epilogue envelope; u=2^-24,u64=2^-53. Exact families require F32 bits; no BF16 output allowance." };
        try
        {
            Need(BitConverter.IsLittleEndian, "Little-endian captures required.");
            if (args.SequenceEqual(new[] { "--self-test" })) { SelfTest(); return 0; }
            if (args.Length == 2 && args[0] == "--inspect-manifest")
            {
                var inspected = CudaNinferFlashInferBodyModule.ValidateManifest(File.ReadAllBytes(args[1]));
                Console.WriteLine(JsonSerializer.Serialize(new { manifestPolicy = "PASS", family = inspected.Family, specializations = inspected.Images.Length, GPU = "NOT_RUN" }, Json)); return 0;
            }
            if (args.Length == 0) { Console.WriteLine("PLAN: --execute --manifest ABSOLUTE_32_IMAGE_JSON --out FRESH_IGNORED_DIRECTORY; --self-test is CPU-only."); return 0; }
            Need(args.Length == 5 && args[0] == "--execute" && args[1] == "--manifest" && args[3] == "--out", "Use --execute --manifest FILE --out FRESH_IGNORED_DIRECTORY.");
            string path = Path.GetFullPath(args[2]); byte[] manifest = File.ReadAllBytes(path); var contract = CudaNinferFlashInferBodyModule.ValidateManifest(manifest);
            output = args[4]; Fresh(output); Directory.CreateDirectory(output); owned = true;
            state["manifestPath"] = path; state["manifestSha256"] = Sha(manifest); state["kernelFamily"] = contract.Family; state["arithmeticPolicy"] = contract.Policy;
            state["probeAssemblySha256"] = Sha(File.ReadAllBytes(typeof(Program).Assembly.Location)); state["backendAssemblySha256"] = Sha(File.ReadAllBytes(typeof(CudaBackend).Assembly.Location));
            state["localWeightProductsExactlyBf16"] = R.ProveLocalWeightProducts(); state["uniqueCases"] = 128; state["moduleLifetimes"] = 2; state["poisonPassesPerCase"] = 3;
            state["scope"] = "Four full original body matrix shapes x physicalT1..8, synthetic raw adjacent-K/M128x4 payloads; no row scatter, cast, full target equivalence, model quality or speed qualification.";
            NewFile(Path.Combine(output, "module-input.json"), manifest);
            void Checkpoint() => Save(Path.Combine(output!, "result.json"), state); Checkpoint();
            CudaBackend.Register(); CudaDriverApi.cuInit(0).ThrowOnError();
            using (var context = CudaContext.Create(0))
            {
                CudaDriverApi.cuDriverGetVersion(out int driver).ThrowOnError(); CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError(); byte[] name = new byte[256]; CudaDriverApi.cuDeviceGetName(name, name.Length, device).ThrowOnError();
                state["device"] = new { ordinal = device, name = System.Text.Encoding.UTF8.GetString(name).TrimEnd('\0'), driverVersion = driver }; state["GPU"] = "EXECUTED";
                for (int lifetime = 0; lifetime < 2; lifetime++)
                {
                    using var module = new CudaNinferFlashInferBodyModule(path);
                    Need(module.ManifestSha256 == state["manifestSha256"]!.ToString(), "The loaded body manifest changed.");
                    if (lifetime == 0) state["imageSetSha256"] = module.ImageSetSha256;
                    else Need(state["imageSetSha256"]!.ToString() == module.ImageSetSha256, "Loaded image set changed across lifetimes.");
                    foreach (var shape in Shapes) foreach (string family in Families)
                    {
                        var payload = MakePayload(shape, family);
                        for (int rows = 1; rows <= 8; rows++) Run(module, Make(shape, family, rows, payload), lifetime, output, records, firstBodies, Checkpoint);
                    }
                }
            }
            Need(records.Count == 256 && records.Select(x => x["id"]!.ToString()).Distinct().Count() == 256, "All256 distinct lifecycle records are mandatory.");
            var cases = records.GroupBy(x => x["case"]!.ToString()).ToArray();
            Need(cases.Length == 128 && cases.All(g => g.Count() == 2 && g.Select(x => (int)x["lifetime"]!).Order().SequenceEqual(new[] { 0, 1 })
                && g.All(x => (bool)x["passed"]! && ((List<object>)x["passes"]!).Count == 3)), "Every fixed body case requires both lifetimes and all poison passes.");
            Need(Sha(File.ReadAllBytes(path)) == state["manifestSha256"]!.ToString(), "Body manifest changed during execution.");
            state["complete"] = true; state["passed"] = true; state["actualLifecycleRecords"] = 256; state["actualKernelLaunches"] = 768; Checkpoint(); Complete(output, state);
            Console.WriteLine(JsonSerializer.Serialize(new { complete = true, passed = true, lifecycleRecords = 256, kernelLaunches = 768, authoritativeResult = "completed-result.json", wholeModel = "NOT_RUN", modelQuality = "NOT_RUN", performance = "NOT_RUN" })); return 0;
        }
        catch (Exception error) { state["error"] = error.ToString(); state["passed"] = false; if (owned && output != null) Save(Path.Combine(output, "result.json"), state); Console.Error.WriteLine(error); return 1; }
    }
}
