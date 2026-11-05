// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using TensorSharp.Validation;

internal static class Program
{
    private const int Guard = 256, GuardBits = 0x4e56789a, Blocks = 5;
    private const double U = 1.0 / 16777216;
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
    private sealed record Shape(string Name, int N, int K, int InputStride, int OutputStride, float Inverse,
        byte[] Codes, byte[] Scales, byte[] OriginalCodes, byte[] OriginalScales, float[] Input,
        float[] CandidateT1, float[] CandidateT3, double[] Ideal, double[] Bound);
    private static void Need(bool ok, string message) { if (!ok) throw new InvalidDataException(message); }
    private static string Hash(ReadOnlySpan<byte> bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    private static string FileHash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant(); }
    private static byte[] Bytes<T>(T[] values) where T : unmanaged => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
    private static void Save(string path, object value) { using var stream = File.Create(path); JsonSerializer.Serialize(stream, value, Json); stream.WriteByte(10); }
    private static float Bf(float v) { uint x = (uint)BitConverter.SingleToInt32Bits(v); return BitConverter.Int32BitsToSingle((int)((x + 0x7fffU + ((x >> 16) & 1U)) & 0xffff0000U)); }
    private static ulong Next(ref ulong state) { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state; }
    private static Shape Make(string name, int n, int k)
    {
        int rows = (n + 127) / 128 * 128, groups = k / 16, inputStride = k + 8, outputStride = n + 7;
        const float inverse = 0.13f;
        byte[] originalCodes = new byte[checked(rows * k / 2)], originalScales = new byte[checked(rows * groups)];
        byte[] codes = new byte[checked(n * k / 2)], scales = new byte[checked(n * groups)];
        ulong rng = 0x946a5d4312baf0e1UL ^ ((ulong)n << 32) ^ (uint)k;
        for (int i = 0; i < originalCodes.Length; i++) originalCodes[i] = (byte)Next(ref rng);
        byte[] rowScales = new byte[rows * groups];
        for (int i = 0; i < rowScales.Length; i++) rowScales[i] = (byte)(Next(ref rng) % 127);
        int at = 0;
        for (int tile = 0; tile < rows / 128; tile++) for (int kg = 0; kg < k / 64; kg++)
            for (int r32 = 0; r32 < 32; r32++) for (int quarter = 0; quarter < 4; quarter++) for (int g = 0; g < 4; g++)
                originalScales[at++] = rowScales[(tile * 128 + quarter * 32 + r32) * groups + kg * 4 + g];
        for (int row = 0; row < n; row++) for (int g = 0; g < groups; g++)
        {
            scales[row * groups + g] = rowScales[row * groups + g];
            for (int j = 0; j < 8; j++)
            {
                int lo = g * 16 + j, hi = lo + 8;
                byte a = originalCodes[row * (k / 2) + lo / 2], b = originalCodes[row * (k / 2) + hi / 2];
                codes[row * (k / 2) + g * 8 + j] = (byte)(((a >> ((lo & 1) * 4)) & 15) | (((b >> ((hi & 1) * 4)) & 15) << 4));
            }
        }
        float[] input = Enumerable.Repeat(BitConverter.Int32BitsToSingle(unchecked((int)0x7fc3a16f)), 3 * inputStride).ToArray();
        for (int t = 0; t < 3; t++) for (int c = 0; c < k; c++) input[t * inputStride + c] = Bf(((int)(Next(ref rng) % 65521) - 32760) / 32768f);
        Console.Error.WriteLine($"CPU reference preparation: {name} {n}x{k}; not a measured interval.");
        float[] t1 = NinferA16ScheduleReference.Compute(originalCodes, originalScales, input, n, k, 1, inverse, inputStride, outputStride);
        float[] t3 = NinferA16ScheduleReference.Compute(originalCodes, originalScales, input, n, k, 3, inverse, inputStride, outputStride);
        double[] ideal = new double[3 * n], bound = new double[3 * n];
        double operations = k / 32 + 16, gamma = operations * U / (1 - operations * U);
        float[] decoded = Enumerable.Range(0, 16).Select(NinferA16ScheduleReference.DecodeCode).ToArray();
        float[] decodedScales = Enumerable.Range(0, 127).Select(i => NinferA16ScheduleReference.DecodeScale((byte)i)).ToArray();
        for (int row = 0; row < n; row++)
        {
            double s0 = 0, s1 = 0, s2 = 0, a0 = 0, a1 = 0, a2 = 0;
            for (int g = 0; g < groups; g++)
            {
                double coefficient = (double)decodedScales[rowScales[row * groups + g]] * inverse;
                for (int j = 0; j < 8; j++)
                {
                    byte packed = originalCodes[row * (k / 2) + g * 8 + j];
                    for (int half = 0; half < 2; half++)
                    {
                        int c = g * 16 + j * 2 + half;
                        double w = decoded[(packed >> (half * 4)) & 15] * coefficient;
                        double v0 = w * input[c], v1 = w * input[inputStride + c], v2 = w * input[2 * inputStride + c];
                        s0 += v0; s1 += v1; s2 += v2; a0 += Math.Abs(v0); a1 += Math.Abs(v1); a2 += Math.Abs(v2);
                    }
                }
            }
            ideal[row] = s0; ideal[n + row] = s1; ideal[2 * n + row] = s2;
            bound[row] = gamma * a0; bound[n + row] = gamma * a1; bound[2 * n + row] = gamma * a2;
        }
        for (int t = 0; t < 3; t++) for (int row = 0; row < n; row++)
        {
            int index = t * n + row;
            Need(float.IsFinite(t3[t * outputStride + row]) && Math.Abs(t3[t * outputStride + row] - ideal[index]) <= bound[index], "CPU candidate T3 reference exceeds fixed FP64 bound.");
            if (t == 0) Need(Math.Abs(t1[row] - ideal[row]) <= bound[row], "CPU candidate T1 reference exceeds fixed FP64 bound.");
        }
        return new(name, n, k, inputStride, outputStride, inverse, codes, scales, originalCodes, originalScales, input, t1, t3, ideal, bound);
    }
    private static void Fresh(string path)
    {
        Need(Path.IsPathFullyQualified(path) && Path.GetFullPath(path) == path, "Canonical absolute evidence path required.");
        string[] p = path.Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries);
        Need(p.Contains("artifacts", StringComparer.Ordinal) || Enumerable.Range(0, p.Length - 1).Any(i => p[i] == "docs" && p[i + 1] == "validation"), "Ignored artifacts/ or docs/validation/ required.");
        bool Link(string v) => new FileInfo(v).LinkTarget != null || new DirectoryInfo(v).LinkTarget != null;
        Need(!Path.Exists(path) && !Link(path), "Fresh evidence directory required.");
        for (string? v = Path.GetDirectoryName(path); v != null; v = Path.GetDirectoryName(v)) Need(!Link(v) && !File.Exists(v), "Output parent alias refused.");
    }
    private static void Certificate(string path, string moduleSha, bool candidate)
    {
        using var json = JsonDocument.Parse(File.ReadAllBytes(path)); var root = json.RootElement;
        if (!candidate)
        {
            Need(root.GetProperty("schema").GetString() == "ninfer-cuda-kernel-probe/1" && root.GetProperty("passed").GetBoolean()
                && ValidationSha256.Matches(root.GetProperty("moduleSha256").GetString()!, moduleSha), "Legacy component certificate/module differs.");
            Need(root.GetProperty("checks").EnumerateArray().All(x => x.GetProperty("passed").GetBoolean()), "Legacy component has a failed check.");
            return;
        }
        Need(root.GetProperty("schema").GetString() == "ninfer-a16-original-schedule-component/1"
            && root.GetProperty("complete").GetBoolean() && root.GetProperty("passed").GetBoolean()
            && root.GetProperty("GPU").GetString() == "EXECUTED" && ValidationSha256.Matches(root.GetProperty("moduleFileSha256").GetString()!, moduleSha)
            && root.GetProperty("uniqueFixtures").GetInt32() == 90 && root.GetProperty("moduleLifetimes").GetInt32() == 2,
            "Matching completed candidate component certificate required before timing.");
        var records = root.GetProperty("records").EnumerateArray().ToArray();
        Need(records.Length == 180 && records.Select(x => x.GetProperty("id").GetString()).Distinct().Count() == 180, "Candidate component lifecycle coverage differs.");
        foreach (var record in records)
        {
            Need(record.GetProperty("passed").GetBoolean() && record.GetProperty("sourceAndInputAllocationsUnchanged").GetBoolean(), "Candidate source/numeric gate failed.");
            var outputs = record.GetProperty("outputs").EnumerateArray().ToArray(); Need(outputs.Length == 3, "Candidate poison coverage differs.");
            foreach (var value in outputs)
            {
                Need(value.GetProperty("bitMismatches").GetInt32() == 0 && value.GetProperty("fp64Failures").GetInt32() == 0
                    && value.GetProperty("paddingMismatches").GetInt32() == 0 && value.GetProperty("guardsIntact").GetBoolean()
                    && value.GetProperty("sameWrittenBitsAsFirst").GetBoolean(), "Candidate component numerical/guard gate failed.");
                string file = value.GetProperty("file").GetString()!; Need(Path.GetFileName(file) == file, "Unsafe component capture path.");
                Need(FileHash(Path.Combine(Path.GetDirectoryName(path)!, file)) == value.GetProperty("sha256").GetString(), "Candidate original capture SHA differs.");
            }
        }
    }
    private sealed unsafe class Memory : IDisposable
    {
        private readonly List<IntPtr> _pointers = new();
        internal IntPtr Put(byte[] bytes) { CudaDriverApi.cuMemAlloc(out var p, (UIntPtr)bytes.Length).ThrowOnError(); _pointers.Add(p); Write(p, bytes); return p; }
        internal static void Write(IntPtr p, byte[] b) { fixed (byte* src = b) CudaDriverApi.cuMemcpyHtoD(p, (IntPtr)src, (UIntPtr)b.Length).ThrowOnError(); }
        internal static byte[] Read(IntPtr p, int n) { byte[] b = new byte[n]; fixed (byte* dst = b) CudaDriverApi.cuMemcpyDtoH((IntPtr)dst, p, (UIntPtr)n).ThrowOnError(); return b; }
        public void Dispose() { foreach (var p in _pointers) CudaDriverApi.cuMemFree(p).ThrowOnError(); }
    }
    private static byte[] Guarded(byte[] body)
    {
        var b = new byte[body.Length + 2 * Guard]; body.CopyTo(b, Guard);
        for (int i = 0; i < Guard / 4; i++) { BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(i * 4), GuardBits); BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(Guard + body.Length + i * 4), GuardBits); }
        return b;
    }
    private static byte[] Check(byte[] all, Shape shape, int tokens, bool candidate, int poison, byte[]? repeat, string path)
    {
        File.WriteAllBytes(path, all); var written = new byte[tokens * shape.N * 4];
        for (int i = 0; i < Guard / 4; i++)
            Need(BinaryPrimitives.ReadInt32LittleEndian(all.AsSpan(i * 4)) == GuardBits
                && BinaryPrimitives.ReadInt32LittleEndian(all.AsSpan(all.Length - Guard + i * 4)) == GuardBits, "Output guard changed; raw output retained.");
        float[] expected = tokens == 1 ? shape.CandidateT1 : shape.CandidateT3;
        for (int t = 0; t < tokens; t++) for (int row = 0; row < shape.OutputStride; row++)
        {
            int bits = BinaryPrimitives.ReadInt32LittleEndian(all.AsSpan(Guard + (t * shape.OutputStride + row) * 4));
            if (row >= shape.N) { Need(bits == poison, "Output stride padding changed."); continue; }
            float actual = BitConverter.Int32BitsToSingle(bits); int index = t * shape.N + row;
            Need(float.IsFinite(actual) && Math.Abs(actual - shape.Ideal[index]) <= shape.Bound[index], "Output exceeds fixed FP64 bound or is nonfinite.");
            if (candidate) Need(bits == BitConverter.SingleToInt32Bits(expected[t * shape.OutputStride + row]), "Candidate misses original-schedule exact F32 oracle.");
            BinaryPrimitives.WriteInt32LittleEndian(written.AsSpan(index * 4), bits);
        }
        if (repeat != null) Need(repeat.AsSpan().SequenceEqual(written), "Same-path full-output replay differs.");
        return written;
    }
    private static double Median(IEnumerable<double> values) { var a = values.Order().ToArray(); return a.Length % 2 != 0 ? a[a.Length / 2] : (a[a.Length / 2 - 1] + a[a.Length / 2]) / 2; }
    private static void Run(Shape shape, CudaNinferKernels legacy, CudaNinferA16Module candidate, int l2, string output, List<object> profiles, Action checkpoint)
    {
        string dir = Path.Combine(output, shape.Name); Directory.CreateDirectory(dir);
        var sourceFiles = new Dictionary<string, object>();
        foreach (var (name, bytes) in new[] { ("original-codes.bin", shape.OriginalCodes), ("original-scales.bin", shape.OriginalScales),
            ("candidate-codes.bin", shape.Codes), ("candidate-scales.bin", shape.Scales), ("input.f32", Bytes(shape.Input)),
            ("candidate-t1.f32", Bytes(shape.CandidateT1)), ("candidate-t3.f32", Bytes(shape.CandidateT3)), ("ideal.f64", Bytes(shape.Ideal)), ("bounds.f64", Bytes(shape.Bound)) })
        {
            bool save = !name.Contains("codes", StringComparison.Ordinal) && !name.Contains("scales", StringComparison.Ordinal);
            if (save) File.WriteAllBytes(Path.Combine(dir, name), bytes);
            sourceFiles[name] = new { sha256 = Hash(bytes), bytes = bytes.Length, saved = save };
        }
        using var memory = new Memory(); long matrixBytes = (long)shape.Codes.Length + shape.Scales.Length;
        int copies = checked((int)(2L * l2 / matrixBytes + 1)); Need(copies is >= 1 and <= 64 && copies * matrixBytes > 2L * l2, "Controlled rotation geometry invalid.");
        byte[] codeGuard = Guarded(shape.Codes), scaleGuard = Guarded(shape.Scales), inputGuard = Guarded(Bytes(shape.Input));
        var weights = Enumerable.Range(0, copies).Select(_ => (Code: memory.Put(codeGuard), Scale: memory.Put(scaleGuard))).ToArray();
        var input = memory.Put(inputGuard); var destination = memory.Put(Guarded(new byte[3 * shape.OutputStride * 4]));
        for (int tokens = 1; tokens <= 3; tokens++)
        {
            var intervals = new List<object>(); var validation = new List<object>();
            var profile = new Dictionary<string, object?> { ["shape"] = shape.Name, ["n"] = shape.N, ["k"] = shape.K, ["tokens"] = tokens,
                ["sourceFiles"] = sourceFiles, ["matrixBytes"] = matrixBytes, ["residentWeightCopies"] = copies,
                ["residentWeightBytes"] = copies * matrixBytes, ["inputStride"] = shape.InputStride, ["outputStride"] = shape.OutputStride,
                ["inverseWeightDivisor"] = shape.Inverse, ["complete"] = false, ["validation"] = validation, ["intervals"] = intervals };
            profiles.Add(profile); checkpoint();
            void Launch(bool useCandidate, int copy)
            {
                var w = weights[copy];
                if (useCandidate) candidate.Project(w.Code + Guard, w.Scale + Guard, input + Guard, destination + Guard,
                    shape.N, shape.K, tokens, shape.Inverse, IntPtr.Zero, shape.InputStride, shape.OutputStride);
                else legacy.ProjectA16(w.Code + Guard, w.Scale + Guard, input + Guard, destination + Guard,
                    shape.N, shape.K, tokens, shape.Inverse, IntPtr.Zero, shape.InputStride, shape.OutputStride);
            }
            byte[] Poison(int bits) => Guarded(Bytes(Enumerable.Repeat(BitConverter.Int32BitsToSingle(bits), tokens * shape.OutputStride).ToArray()));
            byte[][] reference = new byte[2][];
            // Both paths, every resident address, and every full output are validated before any timing.
            for (int engine = 0; engine < 2; engine++) for (int copy = 0; copy < copies; copy++)
            {
                int poison = unchecked((int)0x7fc1a16f); byte[] initial = Poison(poison); Memory.Write(destination, initial);
                Launch(engine == 1, copy); byte[] all = Memory.Read(destination, initial.Length);
                string name = $"t{tokens}-validate-e{engine}-copy{copy}.f32";
                byte[] values = Check(all, shape, tokens, engine == 1, poison, reference[engine], Path.Combine(dir, name));
                reference[engine] ??= values;
                validation.Add(new { engine = engine == 0 ? "legacy" : "candidate", copy, file = name, sha256 = Hash(all), writtenSha256 = Hash(values), finite = true, fp64BoundPassed = true, exactOriginalSchedule = engine == 1 });
            }
            int differing = 0; double maximumDifference = 0;
            for (int i = 0; i < reference[0].Length; i += 4)
            {
                int a = BinaryPrimitives.ReadInt32LittleEndian(reference[0].AsSpan(i)), b = BinaryPrimitives.ReadInt32LittleEndian(reference[1].AsSpan(i));
                if (a != b) differing++; maximumDifference = Math.Max(maximumDifference, Math.Abs((double)BitConverter.Int32BitsToSingle(a) - BitConverter.Int32BitsToSingle(b)));
            }
            profile["oldVsCandidateDifferentF32Elements"] = differing; profile["oldVsCandidateMaximumAbsoluteDifference"] = maximumDifference;
            var times = new Dictionary<(string Mode, bool Candidate), List<double>>();
            foreach (string mode in new[] { "warm-repeat-address", "rotation-over-2L2-addresses" })
            {
                int activeCopies = mode == "warm-repeat-address" ? 1 : copies;
                int launches = mode == "warm-repeat-address" ? 16 : 2 * copies;
                foreach (bool engine in new[] { false, true }) times[(mode, engine)] = new();
                for (int block = 0; block < Blocks; block++) for (int arm = 0; arm < 4; arm++)
                {
                    bool useCandidate = arm is 1 or 2;
                    int warmups = Math.Max(8, 2 * activeCopies);
                    for (int i = 0; i < warmups; i++) Launch(useCandidate, i % activeCopies);
                    CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();
                    int poison = unchecked((int)0x7fc10000) + block * 4 + arm;
                    byte[] initial = Poison(poison); Memory.Write(destination, initial);
                    CudaDriverApi.cuEventCreate(out var begin, 0).ThrowOnError();
                    CudaDriverApi.cuEventCreate(out var end, 0).ThrowOnError();
                    float milliseconds;
                    try
                    {
                        CudaDriverApi.cuEventRecord(begin, IntPtr.Zero).ThrowOnError();
                        for (int i = 0; i < launches; i++) Launch(useCandidate, i % activeCopies);
                        CudaDriverApi.cuEventRecord(end, IntPtr.Zero).ThrowOnError();
                        CudaDriverApi.cuEventSynchronize(end).ThrowOnError();
                        CudaDriverApi.cuEventElapsedTime(out milliseconds, begin, end).ThrowOnError();
                    }
                    finally { CudaDriverApi.cuEventDestroy(begin).ThrowOnError(); CudaDriverApi.cuEventDestroy(end).ThrowOnError(); }
                    Need(float.IsFinite(milliseconds) && milliseconds > 0, "Invalid CUDA-event interval.");
                    double perCall = (double)milliseconds / launches; times[(mode, useCandidate)].Add(perCall);
                    byte[] actual = Memory.Read(destination, initial.Length);
                    string name = $"t{tokens}-{mode}-b{block}-arm{arm}.f32";
                    byte[] values = Check(actual, shape, tokens, useCandidate, poison, reference[useCandidate ? 1 : 0], Path.Combine(dir, name));
                    intervals.Add(new { mode, block, arm, engine = useCandidate ? "candidate" : "legacy", launches, warmupsOutsideInterval = warmups,
                        activeWeightCopies = activeCopies, activeWeightAddressBytes = activeCopies * matrixBytes,
                        cudaEventMilliseconds = milliseconds, millisecondsPerProjection = perCall, file = name, sha256 = Hash(actual), writtenSha256 = Hash(values),
                        fullOutputValidatedOutsideInterval = true, physicalColdCacheVerified = false });
                    checkpoint();
                }
            }
            profile["summary"] = new[] { "warm-repeat-address", "rotation-over-2L2-addresses" }.Select(mode => new
            { mode, legacyMedianMilliseconds = Median(times[(mode, false)]), candidateMedianMilliseconds = Median(times[(mode, true)]),
                legacyOverCandidateDescriptiveRatio = Median(times[(mode, false)]) / Median(times[(mode, true)]),
                legacyMin = times[(mode, false)].Min(), legacyMax = times[(mode, false)].Max(), candidateMin = times[(mode, true)].Min(), candidateMax = times[(mode, true)].Max(),
                intervalsPerEngine = times[(mode, false)].Count, significanceEstablished = false }).ToArray();
            profile["complete"] = true; checkpoint();
        }
        foreach (var w in weights)
        { Need(Memory.Read(w.Code, codeGuard.Length).AsSpan().SequenceEqual(codeGuard), "Resident weight codes/guards changed."); Need(Memory.Read(w.Scale, scaleGuard.Length).AsSpan().SequenceEqual(scaleGuard), "Resident scales/guards changed."); }
        Need(Memory.Read(input, inputGuard.Length).AsSpan().SequenceEqual(inputGuard), "Activation/stride guards changed.");
    }
    public static int Main(string[] args)
    {
        string? output = null; bool owned = false; var profiles = new List<object>();
        var state = new Dictionary<string, object?> { ["schema"] = "ninfer-a16-shape-event-timing/1", ["complete"] = false, ["GPU"] = "NOT_RUN",
            ["modelQuality"] = "NOT_RUN", ["endToEndPerformance"] = "NOT_RUN", ["physicalColdCacheVerified"] = false, ["profiles"] = profiles };
        try
        {
            Need(BitConverter.IsLittleEndian, "Little-endian capture host required.");
            if (args.Length == 1 && args[0] == "--self-test") { var small = Make("CPU-only", 17, 5120); Console.WriteLine(JsonSerializer.Serialize(new { CPU = "PASS", candidateF32 = 4 * small.N, fp64Checks = 3 * small.N, GPU = "NOT_RUN", model = "NOT_RUN" })); return 0; }
            if (args.Length == 0) { Console.WriteLine("PLAN: --execute --legacy-module FILE --candidate-module FILE --legacy-component JSON --candidate-component JSON --out FRESH_IGNORED_DIR. --self-test is CPU only."); return 0; }
            Need(args.Length == 11 && args[0] == "--execute", "Use explicit execution and five option/value pairs.");
            var options = new Dictionary<string, string>(StringComparer.Ordinal);
            for (int i = 1; i < args.Length; i += 2) Need(options.TryAdd(args[i], args[i + 1]), "Duplicate option.");
            Need(options.Keys.Order().SequenceEqual(new[] { "--candidate-component", "--candidate-module", "--legacy-component", "--legacy-module", "--out" }), "Unknown/missing options.");
            output = options["--out"]; Fresh(output);
            string oldPath = Path.GetFullPath(options["--legacy-module"]), newPath = Path.GetFullPath(options["--candidate-module"]);
            string oldSha = FileHash(oldPath), newSha = FileHash(newPath);
            Certificate(options["--legacy-component"], oldSha, false); Certificate(options["--candidate-component"], newSha, true);
            Directory.CreateDirectory(output); owned = true;
            state["moduleFiles"] = new { legacy = new { path = oldPath, sha256 = oldSha }, candidate = new { path = newPath, sha256 = newSha } };
            state["componentCertificates"] = new { legacy = new { path = options["--legacy-component"], sha256 = FileHash(options["--legacy-component"]) }, candidate = new { path = options["--candidate-component"], sha256 = FileHash(options["--candidate-component"]) } };
            state["toolAssemblySha256"] = FileHash(typeof(Program).Assembly.Location); state["backendAssemblySha256"] = FileHash(typeof(CudaBackend).Assembly.Location);
            state["scope"] = "synthetic resident weights; actual projection dimensions; original candidate F32 math differs from legacy";
            state["syntheticWeightGenerator"] = "xorshift64(13,7,17); seed=0x946a5d4312baf0e1 XOR (uint64(N)<<32) XOR uint32(K); original adjacent bytes then positive scale bytes modulo127; source-pinned Make function";
            state["fullSyntheticWeightFilesSaved"] = false;
            state["fixedBound"] = "gamma_(K/32+16)*sum(abs(original_fp4*fp8*inverseD*BF16input)); u=2^-24";
            state["timingProtocol"] = "5 ABBA blocks per shape/T/cache label; warmups, allocations, CPU oracle, readbacks and validation outside events; no cache flush cost hidden";
            void Checkpoint() => Save(Path.Combine(output, "result.json"), state);
            Checkpoint();
            var shapes = new[] { Make("gate-up", 34816, 5120), Make("down", 5120, 17408), Make("output", 5120, 6144), Make("attention-input", 14336, 5120) };
            CudaBackend.Register(); CudaDriverApi.cuInit(0).ThrowOnError(); using var context = CudaContext.Create(0);
            CudaDriverApi.cuDeviceGet(out int device, 0).ThrowOnError(); CudaDriverApi.cuDriverGetVersion(out int version).ThrowOnError();
            CudaDriverApi.cuDeviceGetAttribute(out int l2, 38, device).ThrowOnError(); CudaDriverApi.cuDeviceGetAttribute(out int sms, 16, device).ThrowOnError();
            byte[] name = new byte[256]; CudaDriverApi.cuDeviceGetName(name, name.Length, device).ThrowOnError(); Need(l2 > 0, "Actual L2 size unavailable.");
            state["device"] = new { ordinal = device, name = System.Text.Encoding.UTF8.GetString(name).TrimEnd('\0'), driverVersion = version, l2Bytes = l2, sms,
                cudaVisibleDevices = Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES") }; state["GPU"] = "EXECUTED";
            using var legacy = CudaNinferKernels.Create(oldPath); using var candidate = new CudaNinferA16Module(newPath);
            Need(ValidationSha256.Matches(legacy.ImageSha256, oldSha) && ValidationSha256.Matches(candidate.ImageSha256, newSha), "Loaded module differs from component-qualified bytes.");
            foreach (var shape in shapes) Run(shape, legacy, candidate, l2, output, profiles, Checkpoint);
            Need(FileHash(oldPath) == oldSha && FileHash(newPath) == newSha, "Module files changed during timing.");
            state["complete"] = true; Checkpoint(); Console.WriteLine(JsonSerializer.Serialize(new { complete = true, profiles = profiles.Count, modelQuality = "NOT_RUN", endToEnd = "NOT_RUN", output })); return 0;
        }
        catch (Exception error) { state["error"] = error.ToString(); if (owned && output != null) Save(Path.Combine(output, "result.json"), state); Console.Error.WriteLine(error); return 1; }
    }
}
