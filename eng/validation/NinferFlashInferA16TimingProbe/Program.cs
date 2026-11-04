// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using R = TensorSharp.Validation.NinferFlashInferA16Reference;

internal static class Program
{
    private const int Guard = 256, N = 34816, K = 5120, Blocks = 5;
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
    private static void Need(bool value, string message) { if (!value) throw new InvalidDataException(message); }
    private static string Sha(ReadOnlySpan<byte> data) => Convert.ToHexString(SHA256.HashData(data)).ToLowerInvariant();
    private static string FileSha(string path) { using var f = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(f)).ToLowerInvariant(); }
    private static byte[] Bytes<T>(T[] values) where T : unmanaged => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
    private static void Save(string path, object value) { using var f = File.Create(path); JsonSerializer.Serialize(f, value, Json); f.WriteByte(10); }
    private static void NewFile(string path, byte[] data) { using var f = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None); f.Write(data); f.Flush(true); }
    private static void Completed(string folder, object value)
    {
        byte[] data = JsonSerializer.SerializeToUtf8Bytes(value, Json).Concat(new byte[] { 10 }).ToArray();
        NewFile(Path.Combine(folder, "completed-result.json"), data);
        NewFile(Path.Combine(folder, "completed-result.json.sha256"), System.Text.Encoding.ASCII.GetBytes(Sha(data) + "  completed-result.json\n"));
    }
    private static void Fresh(string path)
    {
        Need(Path.IsPathFullyQualified(path) && Path.GetFullPath(path) == path, "Canonical absolute output required.");
        string[] p = path.Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries);
        Need(p.Contains("artifacts") || Enumerable.Range(0, p.Length - 1).Any(i => p[i] == "docs" && p[i + 1] == "validation"), "Ignored output required.");
        bool Link(string x) => new FileInfo(x).LinkTarget != null || new DirectoryInfo(x).LinkTarget != null;
        Need(!Path.Exists(path) && !Link(path), "Fresh output required.");
        for (string? parent = Path.GetDirectoryName(path); parent != null; parent = Path.GetDirectoryName(parent)) Need(!Link(parent) && !File.Exists(parent), "Aliased output parent refused.");
    }
    private static void CaptureSha(string parent, string file, string sha)
    {
        Need(Path.GetFileName(file) == file, "Unsafe capture path.");
        Need(FileSha(Path.Combine(parent, file)).Equals(sha, StringComparison.OrdinalIgnoreCase), "Original component capture SHA differs.");
    }
    private static void Sidecar(string path)
    {
        Need(Path.GetFileName(path) == "completed-result.json", "Authoritative write-once component certificate required.");
        Need(File.ReadAllText(path + ".sha256") == FileSha(path) + "  completed-result.json\n", "Completed certificate sidecar differs.");
    }
    private static void LegacyCertificate(string path, string image, bool batch)
    {
        if (batch) Sidecar(path);
        using var json = JsonDocument.Parse(File.ReadAllBytes(path)); var root = json.RootElement;
        int fixtures = batch ? 240 : 90;
        Need(root.GetProperty("schema").GetString() == (batch ? "ninfer-a16-batch-exact-component/1" : "ninfer-a16-legacy-exact-component/1")
            && root.GetProperty("complete").GetBoolean() && root.GetProperty("passed").GetBoolean()
            && root.GetProperty("GPU").GetString() == "EXECUTED" && root.GetProperty("moduleFileSha256").GetString()!.Equals(image, StringComparison.OrdinalIgnoreCase)
            && root.GetProperty("uniqueFixtures").GetInt32() == fixtures && root.GetProperty("moduleLifetimes").GetInt32() == 2,
            "Matching actual reference component qualification required.");
        var records = root.GetProperty("records").EnumerateArray().ToArray();
        Need(records.Length == fixtures * 2 && records.Select(x => x.GetProperty("id").GetString()).Distinct().Count() == records.Length, "Reference lifecycle coverage differs.");
        foreach (var record in records)
        {
            Need(record.GetProperty("passed").GetBoolean() && record.GetProperty("sourceAndInputAllocationsUnchanged").GetBoolean(), "Reference component input/numeric failure.");
            var outputs = record.GetProperty("outputs").EnumerateArray().ToArray(); Need(outputs.Length == 3, "Reference poison coverage differs.");
            foreach (var value in outputs)
            {
                foreach (string key in new[] { "legacyCpuBitMismatches", "legacyGpuBitMismatches", "bitMismatches", "fp64Failures", "paddingMismatches" }) Need(value.GetProperty(key).GetInt32() == 0, "Reference numerical component failure.");
                foreach (string key in new[] { "legacyGuardsIntact", "guardsIntact", "sameWrittenBitsAsFirst" }) Need(value.GetProperty(key).GetBoolean(), "Reference guard/repeat component failure.");
                CaptureSha(Path.GetDirectoryName(path)!, value.GetProperty("file").GetString()!, value.GetProperty("sha256").GetString()!);
                CaptureSha(Path.GetDirectoryName(path)!, value.GetProperty("legacyFile").GetString()!, value.GetProperty("legacySha256").GetString()!);
                if (batch)
                {
                    Need(value.GetProperty("oneTokenLegacyBitMismatches").GetInt32() == 0 && value.GetProperty("oneTokenGuardsIntact").GetBoolean(), "One-token reference gate failed.");
                    CaptureSha(Path.GetDirectoryName(path)!, value.GetProperty("oneFile").GetString()!, value.GetProperty("oneSha256").GetString()!);
                }
            }
        }
    }
    private static void CandidateCertificate(string path, string manifest)
    {
        Sidecar(path); using var json = JsonDocument.Parse(File.ReadAllBytes(path)); var root = json.RootElement;
        Need(root.GetProperty("schema").GetString() == "ninfer-flashinfer-native-w4a16-component/1" && root.GetProperty("complete").GetBoolean()
            && root.GetProperty("passed").GetBoolean() && root.GetProperty("GPU").GetString() == "EXECUTED"
            && root.GetProperty("manifestSha256").GetString() == manifest && root.GetProperty("localWeightProductsExactlyBf16").GetInt32() == 2032,
            "Matching actual FlashInfer component qualification required.");
        var records = root.GetProperty("records").EnumerateArray().ToArray();
        Need(records.Length == 16 && records.Select(x => x.GetProperty("id").GetString()).Distinct().Count() == 16
            && records.Select(x => x.GetProperty("case").GetString()).Distinct().Count() == 8, "FlashInfer lifecycle coverage differs.");
        foreach (var record in records)
        {
            Need(record.GetProperty("passed").GetBoolean(), "FlashInfer case did not pass.");
            var passes = record.GetProperty("passes").EnumerateArray().ToArray(); Need(passes.Length == 3, "FlashInfer poison coverage differs.");
            foreach (var pass in passes)
            {
                foreach (string key in new[] { "passed", "guardsIntact", "inputsImmutable", "repeatedBits" }) Need(pass.GetProperty(key).GetBoolean(), "FlashInfer component guard/repeat failure.");
                Need(pass.GetProperty("nonfinite").GetInt32() == 0 && pass.GetProperty("boundFailures").GetInt32() == 0, "FlashInfer numerical component failure.");
                if (record.GetProperty("exactBitsRequired").GetBoolean()) Need(pass.GetProperty("differentF32BitsFromExact").GetInt32() == 0, "FlashInfer exact component failure.");
                if (record.GetProperty("lifetime").GetInt32() == 1) Need(pass.GetProperty("crossLifetimeBits").GetBoolean(), "FlashInfer cross-lifetime repeat failure.");
                CaptureSha(Path.GetDirectoryName(path)!, pass.GetProperty("file").GetString()!, pass.GetProperty("sha256").GetString()!);
            }
        }
    }

    private sealed record Shape(byte[] Codes, byte[] Scales, byte[] RawCodes, byte[] RawScales, float[] Input, ushort[] CompactInput,
        float[] ExactT1, float[] ExactT8, double[] Ideal, double[] LegacyBounds, double[] MmaBounds, int InputStride, int OutputStride, float Alpha, string GeneratorAssemblySha256);
    private static Shape FrozenShape(int n)
    {
        // Execute the actual frozen generator, not a rewritten equivalent. No
        // file in its project is edited by this candidate or timing tool.
        var assembly = Assembly.Load("NinferA16BatchExactTimingProbe");
        var type = assembly.GetType("Program", throwOnError: true)!;
        var make = type.GetMethod("Make", BindingFlags.Static | BindingFlags.NonPublic)!;
        object value = make.Invoke(null, new object[] { "gateup", n, K })!;
        T Read<T>(string property) => (T)value.GetType().GetProperty(property)!.GetValue(value)!;
        var input = Read<float[]>("Input"); int inputStride = Read<int>("InputStride"), outputStride = Read<int>("OutputStride");
        ushort[] compact = new ushort[8 * K];
        for (int t = 0; t < 8; t++) for (int c = 0; c < K; c++)
        {
            float x = input[t * inputStride + c]; compact[t * K + c] = R.Bf(x);
            Need(BitConverter.SingleToInt32Bits(R.Float(compact[t * K + c])) == BitConverter.SingleToInt32Bits(x), "Input compaction changed a frozen BF16 value.");
        }
        var ideal = Read<double[]>("Ideal"); var legacy = Read<double[]>("Bound"); double[] mma = new double[ideal.Length];
        const double u = R.UnitRoundoff, u64 = 1.0 / 9007199254740992.0;
        double gammaOld = (K / 32 + 16) * u / (1 - (K / 32 + 16) * u), gamma = (K + 64) * u / (1 - (K + 64) * u), gamma64 = K * u64 / (1 - K * u64);
        for (int i = 0; i < mma.Length; i++)
        {
            // The frozen generator stores gammaOld*sumabs (already including
            // alpha). Four upward ULPs enclose the multiply/divide roundoff;
            // 1/(1-gamma64) encloses its FP64 positive accumulation error.
            double absolute = legacy[i] / gammaOld;
            for (int up = 0; up < 4; up++) absolute = Math.BitIncrement(absolute);
            absolute /= 1 - gamma64;
            mma[i] = (1 + u) * gamma * absolute + u * Math.Abs(ideal[i])
                + (1 + u64) * gamma64 * absolute + u64 * Math.Abs(ideal[i]);
        }
        return new(Read<byte[]>("Codes"), Read<byte[]>("Scales"), Read<byte[]>("OriginalCodes"), Read<byte[]>("OriginalScales"), input, compact,
            Read<float[]>("CandidateT1"), Read<float[]>("CandidateT8"), ideal, legacy, mma, inputStride, outputStride, Read<float>("Inverse"), FileSha(assembly.Location));
    }
    private sealed unsafe class Memory : IDisposable
    {
        private readonly List<IntPtr> _owned = new();
        internal IntPtr Put(byte[] data) { CudaDriverApi.cuMemAlloc(out var p, (UIntPtr)data.Length).ThrowOnError(); _owned.Add(p); Write(p, data); return p; }
        internal static void Write(IntPtr p, byte[] data) { fixed (byte* source = data) CudaDriverApi.cuMemcpyHtoD(p, (IntPtr)source, (UIntPtr)data.Length).ThrowOnError(); }
        internal static byte[] Read(IntPtr p, int bytes) { byte[] result = new byte[bytes]; fixed (byte* target = result) CudaDriverApi.cuMemcpyDtoH((IntPtr)target, p, (UIntPtr)bytes).ThrowOnError(); return result; }
        public void Dispose() { foreach (var p in _owned) CudaDriverApi.cuMemFree(p).ThrowOnError(); }
    }
    private static byte[] Guarded(byte[] body) { byte[] result = Enumerable.Repeat((byte)0xa5, body.Length + 2 * Guard).ToArray(); body.CopyTo(result, Guard); return result; }
    private static byte[] Check(byte[] all, Shape shape, int rows, bool candidate, byte[]? previous, string file)
    {
        NewFile(file, all);
        Need(all.AsSpan(0, Guard).ToArray().All(x => x == 0xa5) && all.AsSpan(all.Length - Guard).ToArray().All(x => x == 0xa5), "Output guard changed; raw evidence retained.");
        int stride = candidate ? N : shape.OutputStride; float[] expected = rows == 1 ? shape.ExactT1 : shape.ExactT8;
        byte[] written = new byte[rows * N * 4];
        for (int t = 0; t < rows; t++) for (int row = 0; row < stride; row++)
        {
            int bits = BinaryPrimitives.ReadInt32LittleEndian(all.AsSpan(Guard + (t * stride + row) * 4));
            if (row >= N) { Need(bits == unchecked((int)0xc1c1c1c1), "Reference stride padding changed."); continue; }
            float actual = BitConverter.Int32BitsToSingle(bits); int index = t * N + row;
            Need(float.IsFinite(actual) && Math.Abs(actual - shape.Ideal[index]) <= (candidate ? shape.MmaBounds[index] : shape.LegacyBounds[index]), "Projection misses its fixed FP64 numerical gate.");
            if (!candidate) Need(bits == BitConverter.SingleToInt32Bits(expected[t * shape.OutputStride + row]), "Qualified reference misses its exact frozen CPU schedule.");
            BinaryPrimitives.WriteInt32LittleEndian(written.AsSpan(index * 4), bits);
        }
        if (previous != null) Need(previous.AsSpan().SequenceEqual(written), "Same-arm output bits changed across resident addresses or timing intervals.");
        return written;
    }
    private static double Median(IEnumerable<double> source) { var x = source.Order().ToArray(); return x.Length % 2 != 0 ? x[x.Length / 2] : (x[x.Length / 2 - 1] + x[x.Length / 2]) / 2; }

    private static void Profile(Shape shape, int rows, CudaNinferFlashInferA16Module candidate, CudaNinferA16LegacyExactModule vector,
        CudaNinferA16BatchExactModule batch, int l2, string output, List<Dictionary<string, object?>> profiles, Action checkpoint)
    {
        using var memory = new Memory();
        long matrixBytes = CudaNinferFlashInferA16Module.CodeBytes + CudaNinferFlashInferA16Module.ScaleBytes;
        int copies = checked((int)(2L * l2 / matrixBytes + 1)); Need(copies is >= 1 and <= 64 && copies * matrixBytes > 2L * l2, "Weight rotation must exceed twice L2 per arm.");
        var source = new Dictionary<string, byte[]> { ["codes"] = shape.Codes, ["scales"] = shape.Scales, ["raw-codes"] = shape.RawCodes, ["raw-scales"] = shape.RawScales,
            ["input-f32"] = Bytes(shape.Input), ["input-bf16"] = Bytes(shape.CompactInput), ["alpha"] = Bytes(new[] { shape.Alpha }) };
        var weights = Enumerable.Range(0, copies).Select(_ => new[] { memory.Put(Guarded(shape.Codes)), memory.Put(Guarded(shape.Scales)), memory.Put(Guarded(shape.RawCodes)), memory.Put(Guarded(shape.RawScales)) }).ToArray();
        var input = memory.Put(Guarded(source["input-f32"])); var compact = memory.Put(Guarded(source["input-bf16"])); var alpha = memory.Put(Guarded(source["alpha"]));
        int oldBytes = rows * shape.OutputStride * 4, newBytes = rows * N * 4;
        var oldOut = memory.Put(Guarded(new byte[oldBytes])); var newOut = memory.Put(Guarded(new byte[newBytes]));
        var intervals = new List<object>(); var validations = new List<object>();
        var record = new Dictionary<string, object?> { ["rows"] = rows, ["reference"] = rows == 1 ? "qualified-vector-exact-T1" : "qualified-batch-exact-T8", ["complete"] = false,
            ["residentCopiesPerArm"] = copies, ["matrixBytesPerCopy"] = matrixBytes, ["addressBytesPerArm"] = copies * matrixBytes,
            ["intervals"] = intervals, ["validations"] = validations }; profiles.Add(record); checkpoint();
        void Launch(bool fi, int copy)
        {
            var w = weights[copy];
            if (fi) candidate.ProjectRaw(w[2] + Guard, w[3] + Guard, compact + Guard, newOut + Guard, alpha + Guard, rows, IntPtr.Zero);
            else if (rows == 1) vector.Project(w[0] + Guard, w[1] + Guard, input + Guard, oldOut + Guard, N, K, 1, shape.Alpha, IntPtr.Zero, shape.InputStride, shape.OutputStride);
            else batch.Project(w[0] + Guard, w[1] + Guard, input + Guard, oldOut + Guard, N, K, 8, shape.Alpha, IntPtr.Zero, shape.InputStride, shape.OutputStride);
        }
        byte[]?[] first = new byte[]?[2];
        string dir = Path.Combine(output, "t" + rows); Directory.CreateDirectory(dir);
        foreach (bool fi in new[] { false, true }) for (int copy = 0; copy < copies; copy++)
        {
            int index = fi ? 1 : 0; IntPtr dest = fi ? newOut : oldOut; int bytes = fi ? newBytes : oldBytes;
            Memory.Write(dest, Guarded(Enumerable.Repeat((byte)0xc1, bytes).ToArray())); Launch(fi, copy);
            byte[] actual = Memory.Read(dest, bytes + 2 * Guard); string name = "validate-" + (fi ? "fi" : "reference") + "-copy" + copy + ".bin";
            var written = Check(actual, shape, rows, fi, first[index], Path.Combine(dir, name)); first[index] ??= written;
            validations.Add(new { arm = fi ? "flashinfer" : "reference", copy, file = "t" + rows + "/" + name, sha256 = Sha(actual), numericalGate = "PASS" });
        }
        var times = new Dictionary<(string, bool), List<double>>();
        foreach (string mode in new[] { "warm-repeat-address", "rotation-over-2L2-addresses" })
        {
            int active = mode == "warm-repeat-address" ? 1 : copies, launches = mode == "warm-repeat-address" ? 16 : 2 * copies;
            times[(mode, false)] = new(); times[(mode, true)] = new();
            for (int block = 0; block < Blocks; block++) for (int arm = 0; arm < 4; arm++)
            {
                bool fi = arm is 1 or 2; int index = fi ? 1 : 0, warmups = Math.Max(8, 2 * active);
                for (int i = 0; i < warmups; i++) Launch(fi, i % active);
                CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();
                IntPtr dest = fi ? newOut : oldOut; int bytes = fi ? newBytes : oldBytes; Memory.Write(dest, Guarded(Enumerable.Repeat((byte)0xc1, bytes).ToArray()));
                CudaDriverApi.cuEventCreate(out var begin, 0).ThrowOnError(); CudaDriverApi.cuEventCreate(out var end, 0).ThrowOnError(); float ms;
                try
                {
                    CudaDriverApi.cuEventRecord(begin, IntPtr.Zero).ThrowOnError();
                    for (int i = 0; i < launches; i++) Launch(fi, i % active);
                    CudaDriverApi.cuEventRecord(end, IntPtr.Zero).ThrowOnError(); CudaDriverApi.cuEventSynchronize(end).ThrowOnError();
                    CudaDriverApi.cuEventElapsedTime(out ms, begin, end).ThrowOnError();
                }
                finally { CudaDriverApi.cuEventDestroy(begin).ThrowOnError(); CudaDriverApi.cuEventDestroy(end).ThrowOnError(); }
                Need(float.IsFinite(ms) && ms > 0, "Invalid CUDA-event time.");
                double perCall = ms / (double)launches; times[(mode, fi)].Add(perCall);
                var actual = Memory.Read(dest, bytes + 2 * Guard); string name = mode + "-b" + block + "-a" + arm + ".bin";
                var written = Check(actual, shape, rows, fi, first[index], Path.Combine(dir, name));
                intervals.Add(new { mode, block, arm, engine = fi ? "flashinfer" : "reference", launches, warmupsOutsideInterval = warmups,
                    activeWeightCopies = active, activeWeightBytes = active * matrixBytes, millisecondsPerProjection = perCall, cudaEventMilliseconds = ms,
                    fullOutputValidatedOutsideInterval = true, file = "t" + rows + "/" + name, sha256 = Sha(actual), writtenSha256 = Sha(written), physicalColdCacheVerified = false }); checkpoint();
            }
        }
        foreach (var w in weights) for (int i = 0; i < 4; i++) Need(Memory.Read(w[i], source[new[] { "codes", "scales", "raw-codes", "raw-scales" }[i]].Length + 2 * Guard).AsSpan().SequenceEqual(Guarded(source[new[] { "codes", "scales", "raw-codes", "raw-scales" }[i]])), "Weight allocation/guard changed.");
        foreach (var pair in new[] { (input, "input-f32"), (compact, "input-bf16"), (alpha, "alpha") }) Need(Memory.Read(pair.Item1, source[pair.Item2].Length + 2 * Guard).AsSpan().SequenceEqual(Guarded(source[pair.Item2])), "Activation/alpha allocation changed.");
        record["summary"] = new[] { "warm-repeat-address", "rotation-over-2L2-addresses" }.Select(mode => new { mode,
            referenceMedianMilliseconds = Median(times[(mode, false)]), flashInferMedianMilliseconds = Median(times[(mode, true)]),
            referenceOverFlashInferDescriptiveRatio = Median(times[(mode, false)]) / Median(times[(mode, true)]),
            intervalsPerEngine = times[(mode, false)].Count, significanceEstablished = false }).ToArray();
        record["sourceAllocationsImmutable"] = true; record["complete"] = true; checkpoint();
    }

    internal static int Main(string[] args)
    {
        string? output = null; bool owned = false; var profiles = new List<Dictionary<string, object?>>();
        var state = new Dictionary<string, object?> { ["schema"] = "ninfer-flashinfer-vs-qualified-exact-event-timing/1", ["complete"] = false,
            ["GPU"] = "NOT_RUN", ["modelQuality"] = "NOT_RUN", ["endToEndPerformance"] = "NOT_RUN", ["physicalColdCacheVerified"] = false, ["profiles"] = profiles };
        try
        {
            if (args.SequenceEqual(new[] { "--self-test" }))
            {
                var s = FrozenShape(17); Need(s.CompactInput.Length == 8 * K && s.MmaBounds.Length == 8 * 17 && s.MmaBounds.All(x => double.IsFinite(x) && x > 0), "Frozen data extraction failed.");
                Console.WriteLine(JsonSerializer.Serialize(new { CPU = "PASS", frozenGeneratorAssemblySha256 = s.GeneratorAssemblySha256, losslessBf16Inputs = s.CompactInput.Length, GPU = "NOT_RUN", quality = "NOT_RUN" })); return 0;
            }
            if (args.Length == 0) { Console.WriteLine("PLAN: --execute --manifest JSON --component COMPLETED --vector-module FILE --vector-component JSON --batch-module FILE --batch-component COMPLETED --out FRESH_IGNORED_DIR"); return 0; }
            Need(args.Length == 15 && args[0] == "--execute", "Explicit execution and all seven option/value pairs required.");
            var options = new Dictionary<string, string>(); for (int i = 1; i < args.Length; i += 2) Need(options.TryAdd(args[i], args[i + 1]), "Duplicate option.");
            Need(options.Keys.Order().SequenceEqual(new[] { "--batch-component", "--batch-module", "--component", "--manifest", "--out", "--vector-component", "--vector-module" }), "Unknown/missing option.");
            output = options["--out"]; Fresh(output);
            string manifestPath = Path.GetFullPath(options["--manifest"]), vectorPath = Path.GetFullPath(options["--vector-module"]), batchPath = Path.GetFullPath(options["--batch-module"]);
            string manifestSha = FileSha(manifestPath), vectorSha = FileSha(vectorPath), batchSha = FileSha(batchPath);
            CandidateCertificate(options["--component"], manifestSha); LegacyCertificate(options["--vector-component"], vectorSha, false); LegacyCertificate(options["--batch-component"], batchSha, true);
            Directory.CreateDirectory(output); owned = true;
            state["manifestSha256"] = manifestSha; state["referenceImages"] = new { vector = vectorSha, batch = batchSha };
            state["certificates"] = new[] { "--component", "--vector-component", "--batch-component" }.ToDictionary(k => k, k => new { path = options[k], sha256 = FileSha(options[k]) });
            state["toolAssemblySha256"] = FileSha(typeof(Program).Assembly.Location); state["backendAssemblySha256"] = FileSha(typeof(CudaBackend).Assembly.Location);
            state["scope"] = "One full synthetic gate/up shape, independent arm math gates, same frozen original weights/BF16 input, five ABBA blocks; no end-to-end/quality/fastest claim.";
            void Checkpoint() => Save(Path.Combine(output!, "result.json"), state); Checkpoint();
            var shape = FrozenShape(N); state["generatorAssemblySha256"] = shape.GeneratorAssemblySha256;
            state["sourceHashes"] = new { originalCodes = Sha(shape.RawCodes), originalScales = Sha(shape.RawScales), tsCodes = Sha(shape.Codes), tsScales = Sha(shape.Scales), f32Input = Sha(Bytes(shape.Input)), bf16Input = Sha(Bytes(shape.CompactInput)) };
            NewFile(Path.Combine(output, "input.f32"), Bytes(shape.Input)); NewFile(Path.Combine(output, "input.bf16"), Bytes(shape.CompactInput));
            NewFile(Path.Combine(output, "ideal.f64"), Bytes(shape.Ideal)); NewFile(Path.Combine(output, "mma-bounds.f64"), Bytes(shape.MmaBounds)); NewFile(Path.Combine(output, "legacy-bounds.f64"), Bytes(shape.LegacyBounds));
            CudaBackend.Register(); CudaDriverApi.cuInit(0).ThrowOnError();
            using (var context = CudaContext.Create(0))
            {
                CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError(); CudaDriverApi.cuDeviceGetAttribute(out int l2, 38, device).ThrowOnError(); Need(l2 > 0, "L2 size unavailable.");
                CudaDriverApi.cuDriverGetVersion(out int driver).ThrowOnError(); byte[] name = new byte[256]; CudaDriverApi.cuDeviceGetName(name, name.Length, device).ThrowOnError();
                state["device"] = new { name = System.Text.Encoding.UTF8.GetString(name).TrimEnd('\0'), driverVersion = driver, l2Bytes = l2 }; state["GPU"] = "EXECUTED";
                using var candidate = new CudaNinferFlashInferA16Module(manifestPath); using var vector = new CudaNinferA16LegacyExactModule(vectorPath); using var batch = new CudaNinferA16BatchExactModule(batchPath);
                Need(candidate.ManifestSha256 == manifestSha && vector.ImageSha256.Equals(vectorSha, StringComparison.OrdinalIgnoreCase) && batch.ImageSha256.Equals(batchSha, StringComparison.OrdinalIgnoreCase), "Loaded image identity differs.");
                foreach (int rows in new[] { 1, 8 }) Profile(shape, rows, candidate, vector, batch, l2, output, profiles, Checkpoint);
            }
            Need(profiles.Count == 2 && profiles.All(x => (bool)x["complete"]!), "Both static shape comparisons are required.");
            Need(FileSha(manifestPath) == manifestSha && FileSha(vectorPath) == vectorSha && FileSha(batchPath) == batchSha, "Module sources changed during execution.");
            state["complete"] = true; state["passed"] = true; Checkpoint(); Completed(output, state);
            Console.WriteLine(JsonSerializer.Serialize(new { complete = true, profiles = profiles.Count, significanceEstablished = false, modelQuality = "NOT_RUN", endToEndPerformance = "NOT_RUN" })); return 0;
        }
        catch (Exception error) { state["error"] = error.ToString(); state["passed"] = false; if (owned && output != null) Save(Path.Combine(output, "result.json"), state); Console.Error.WriteLine(error); return 1; }
    }
}
