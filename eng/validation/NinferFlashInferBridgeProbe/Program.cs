// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using R = TensorSharp.Validation.NinferFlashInferBridgeReference;

internal static class Program
{
    private const int Guard = 256, InputPrefix = 260;
    private static readonly byte[] Poisons = { 0xc1, 0xd2, 0xc1 };
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
    private sealed record Fixture(string Id, bool Cast, int N, int Rows, CudaNinferOriginalRowMapping Mapping, byte[] Input, byte[] Expected);
    private static void Need(bool ok, string message) { if (!ok) throw new InvalidDataException(message); }
    private static byte[] Bytes<T>(T[] input) where T : unmanaged => MemoryMarshal.AsBytes(input.AsSpan()).ToArray();
    private static string Sha(ReadOnlySpan<byte> bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    private static string FileSha(string path) { using var f = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(f)).ToLowerInvariant(); }
    private static byte[] Guarded(byte[] body, int prefix) { var b = Enumerable.Repeat((byte)0xa5, checked(prefix + body.Length + Guard)).ToArray(); body.CopyTo(b, prefix); return b; }
    private static bool Guards(byte[] bytes, int prefix) => bytes.AsSpan(0, prefix).ToArray().All(x => x == 0xa5) && bytes.AsSpan(bytes.Length - Guard).ToArray().All(x => x == 0xa5);
    private static void NewFile(string path, byte[] data) { using var f = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None); f.Write(data); f.Flush(true); }
    private static void Save(string path, object value) { using var f = File.Create(path); JsonSerializer.Serialize(f, value, Json); f.WriteByte(10); }
    private static void Complete(string folder, object value)
    {
        byte[] data = JsonSerializer.SerializeToUtf8Bytes(value, Json).Concat(new byte[] { 10 }).ToArray();
        NewFile(Path.Combine(folder, "completed-result.json"), data);
        NewFile(Path.Combine(folder, "completed-result.json.sha256"), System.Text.Encoding.ASCII.GetBytes(Sha(data) + "  completed-result.json\n"));
    }
    private static void Fresh(string path)
    {
        Need(Path.IsPathFullyQualified(path) && Path.GetFullPath(path) == path, "Canonical absolute output required.");
        var parts = path.Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries);
        Need(parts.Contains("artifacts") || Enumerable.Range(0, Math.Max(0, parts.Length - 1)).Any(i => parts[i] == "docs" && parts[i + 1] == "validation"), "Ignored output required.");
        bool Link(string x) => new FileInfo(x).LinkTarget != null || new DirectoryInfo(x).LinkTarget != null;
        Need(!Path.Exists(path) && !Link(path), "Fresh output required.");
        for (string? p = Path.GetDirectoryName(path); p != null; p = Path.GetDirectoryName(p)) Need(!Link(p) && !File.Exists(p), "Aliased output parent refused.");
    }
    private static IEnumerable<Fixture> Fixtures()
    {
        Fixture Cast(string id, uint[] bits) => new(id, true, bits.Length, 1, 0, Bytes(bits), Bytes(R.Cast(bits)));
        yield return Cast("cast-all-finite-bf16", R.FiniteBf16Inputs());
        yield return Cast("cast-all-midpoint-neighbors-positive", R.Midpoints(false));
        yield return Cast("cast-all-midpoint-neighbors-negative", R.Midpoints(true));
        uint[] edges = { 0, 0x80000000, 1, 0x80000001, 0x7fff, 0x8000, 0x8001, 0x007fffff, 0x00800000, 0x00010000, 0x80010000, 0x3f807fff, 0x3f808000, 0x3f808001, 0x3f818000, 0x7f7f7fff, 0x7f7f8000, 0x7f7f8001, 0x7f7fffff, 0xff7fffff };
        foreach (int count in new[] { 1, 7, 255, 256, 257, 5120, 40959, 40960, 131071, 131072, 139263, 139264 })
            yield return Cast("cast-edge-count-" + count, Enumerable.Range(0, count).Select(i => edges[i % edges.Length]).ToArray());
        foreach (var mapping in new[] { CudaNinferOriginalRowMapping.Identity, CudaNinferOriginalRowMapping.GateUpRows8, CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue })
            foreach (int n in mapping == CudaNinferOriginalRowMapping.Identity ? new[] { 128, 5120, 6144, 17408, 34816 } : mapping == CudaNinferOriginalRowMapping.GateUpRows8 ? new[] { 34816 } : new[] { 14336 })
                foreach (int rows in new[] { 1, 7, 8 }) foreach (bool arbitrary in new[] { false, true })
                {
                    var input = new uint[n * rows];
                    uint[] special = { 0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345, 0x7f812345, 0xff812345 };
                    for (int i = 0; i < input.Length; i++) input[i] = !arbitrary ? 0x3f000000U + (uint)i : i % 127 < special.Length ? special[i % 127] : unchecked(((uint)i * 2654435761U) ^ 0x9e3779b9U);
                    yield return new($"scatter-{mapping}-n{n}-t{rows}-" + (arbitrary ? "arbitrary-bits" : "unique-row-tags"), false, n, rows, mapping, Bytes(input), Bytes(R.Scatter(input, n, rows, mapping)));
                }
    }
    private static void Refused(Action action)
    {
        bool refused = false; try { action(); } catch (ArgumentException) { refused = true; }
        Need(refused, "Invalid bridge admission accepted.");
    }
    private static void SelfTest()
    {
        var finite = R.FiniteBf16Inputs(); Need(finite.Length == 65280 && finite.All(x => R.NearestBf16(x) == x >> 16), "Exact finite BF16 proof differs.");
        foreach (bool negative in new[] { false, true })
        {
            uint[] values = R.Midpoints(negative);
            for (int h = 0; h < 32640; h++) for (int j = 0; j < 3; j++)
            {
                ushort magnitude = (ushort)(j == 0 || j == 1 && (h & 1) == 0 ? h : h + 1);
                Need(R.NearestBf16(values[h * 3 + j]) == (ushort)(magnitude | (negative ? 0x8000 : 0)), "Midpoint/parity proof differs.");
            }
        }
        Need(R.NearestBf16(1) == 0 && R.NearestBf16(0x80000001) == 0x8000 && R.NearestBf16(0x00010000) == 1 && R.NearestBf16(0x7f7fffff) == 0x7f80, "Subnormal/overflow proof differs.");
        var gate = R.Scatter(Enumerable.Range(0, 34816).Select(i => (uint)i).ToArray(), 34816, 1, CudaNinferOriginalRowMapping.GateUpRows8);
        Need(gate[7] == 7 && gate[8] == 17408 && gate[15] == 17415 && gate[16] == 8 && gate.Distinct().Count() == gate.Length, "Gate semantic permutation differs.");
        var gqa = R.Scatter(Enumerable.Range(0, 14336).Select(i => (uint)i).ToArray(), 14336, 1, CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue);
        Need(gqa[255] == 255 && gqa[256] == 7168 && gqa[512] == 256 && gqa[12287] == 13311 && gqa[12288] == 6144 && gqa[13311] == 7167 && gqa[13312] == 13312 && gqa.Distinct().Count() == gqa.Length, "GQA semantic permutation differs.");
        foreach (long count in new long[] { 0, -1, 139265, long.MaxValue }) Refused(() => CudaNinferFlashInferBridge.ValidateCast(count));
        Refused(() => CudaNinferFlashInferBridge.ValidateScatter(34816, 9, 0)); Refused(() => CudaNinferFlashInferBridge.ValidateScatter(128, 1, CudaNinferOriginalRowMapping.GateUpRows8));
        Refused(() => CudaNinferFlashInferBridge.ValidateScatter(14336, 1, (CudaNinferOriginalRowMapping)99));
        Refused(() => CudaNinferFlashInferBridge.ValidateSpans((IntPtr)4096, 64, 4, (IntPtr)4100, 32, 2));
        Refused(() => CudaNinferFlashInferBridge.ValidateSpans((IntPtr)4097, 64, 4, (IntPtr)8192, 32, 2));
        long expectedBytes = 0; int cases = 0; var ids = new HashSet<string>(); foreach (var f in Fixtures()) { Need(f.Input.Length > 0 && f.Expected.Length > 0 && ids.Add(f.Id), "Empty/duplicate fixture."); expectedBytes += f.Expected.Length; cases++; }
        Need(cases == 57, "Fixture coverage differs.");
        Console.WriteLine(JsonSerializer.Serialize(new { CPU = "PASS", finiteBf16Values = finite.Length, midpointNeighborValues = 195840, cases, lifecycleRecords = cases * 2, GPUlaunchesPlanned = cases * 6, expectedOutputBytesPerPass = expectedBytes, GPU = "NOT_RUN", modelQuality = "NOT_RUN", performance = "NOT_RUN" }));
    }
    private sealed unsafe class Memory : IDisposable
    {
        private readonly List<IntPtr> _owned = new();
        internal IntPtr Put(byte[] bytes) { CudaDriverApi.cuMemAlloc(out var p, (UIntPtr)bytes.Length).ThrowOnError(); _owned.Add(p); Write(p, bytes); return p; }
        internal static void Write(IntPtr p, byte[] bytes) { fixed (byte* src = bytes) CudaDriverApi.cuMemcpyHtoD(p, (IntPtr)src, (UIntPtr)bytes.Length).ThrowOnError(); }
        internal static byte[] Read(IntPtr p, int count) { byte[] bytes = new byte[count]; fixed (byte* dst = bytes) CudaDriverApi.cuMemcpyDtoH((IntPtr)dst, p, (UIntPtr)count).ThrowOnError(); return bytes; }
        public void Dispose() { foreach (var p in _owned) CudaDriverApi.cuMemFree(p).ThrowOnError(); }
    }
    private static void Run(CudaNinferFlashInferBridge module, Fixture f, int lifetime, IntPtr stream, string output, List<object> records, Dictionary<string, string> first, Dictionary<string, string> files, Action checkpoint)
    {
        int prefix = f.Cast ? 258 : 268; byte[] input = Guarded(f.Input, InputPrefix); using var memory = new Memory();
        var src = memory.Put(input); var dst = memory.Put(Guarded(new byte[f.Expected.Length], prefix));
        string folder = Path.Combine(output, f.Id); if (!Directory.Exists(folder)) { Directory.CreateDirectory(folder); NewFile(Path.Combine(folder, "input.bin"), f.Input); NewFile(Path.Combine(folder, "expected.bin"), f.Expected); files.Add(f.Id + "/input.bin", Sha(f.Input)); files.Add(f.Id + "/expected.bin", Sha(f.Expected)); }
        var passes = new List<object>(); var record = new Dictionary<string, object?> { ["id"] = $"life{lifetime}/{f.Id}", ["fixture"] = f.Id, ["lifetime"] = lifetime, ["cast"] = f.Cast, ["N"] = f.N, ["rows"] = f.Rows, ["mapping"] = f.Mapping.ToString(), ["inputSha256"] = Sha(f.Input), ["expectedSha256"] = Sha(f.Expected), ["inputPrefixBytes"] = InputPrefix, ["outputPrefixBytes"] = prefix, ["passed"] = false, ["passes"] = passes }; records.Add(record); checkpoint();
        for (int pass = 0; pass < 3; pass++)
        {
            Memory.Write(dst, Guarded(Enumerable.Repeat(Poisons[pass], f.Expected.Length).ToArray(), prefix));
            if (f.Cast) module.Cast(src + InputPrefix, dst + prefix, f.N, stream); else module.Scatter(src + InputPrefix, dst + prefix, f.N, f.Rows, f.Mapping, stream);
            CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError(); byte[] raw = Memory.Read(dst, prefix + f.Expected.Length + Guard); string file = $"{f.Id}/life{lifetime}-pass{pass}.guarded.bin"; NewFile(Path.Combine(output, file), raw);
            files.Add(file, Sha(raw));
            byte[] body = raw.AsSpan(prefix, f.Expected.Length).ToArray(); string hash = Sha(body); bool repeated = !first.TryGetValue(f.Id, out var previous) || previous == hash; first.TryAdd(f.Id, hash);
            bool immutable = Memory.Read(src, input.Length).AsSpan().SequenceEqual(input); bool exact = body.AsSpan().SequenceEqual(f.Expected), guards = Guards(raw, prefix);
            passes.Add(new { pass, poisonByte = Poisons[pass], file, sha256 = Sha(raw), bodySha256 = hash, exactExpectedBits = exact, guardsIntact = guards, sourceAllocationUnchanged = immutable, aaAndCrossLifetimeBits = repeated, passed = exact && guards && immutable && repeated }); checkpoint(); Need(exact && guards && immutable && repeated, "Exact bridge/guard/immutability/repeat gate failed; full raw output retained.");
        }
        byte[] before = Memory.Read(dst, prefix + f.Expected.Length + Guard);
        Refused(() => module.Cast(src + InputPrefix, dst + prefix, 0, stream));
        Refused(() => module.Scatter(src + InputPrefix, dst + prefix, 128, 9, 0, stream));
        Refused(() => module.Cast(src + InputPrefix, src + InputPrefix, 1, stream));
        Need(Memory.Read(dst, before.Length).AsSpan().SequenceEqual(before) && Memory.Read(src, input.Length).AsSpan().SequenceEqual(input), "Rejected operation mutated allocations.");
        record["rejectedCallsUnchanged"] = true; record["passed"] = true; checkpoint();
    }
    internal static int Main(string[] args)
    {
        string? output = null; bool owned = false; var records = new List<object>();
        var state = new Dictionary<string, object?> { ["schema"] = "ninfer-flashinfer-owned-bridge-component/1", ["complete"] = false, ["GPU"] = "NOT_RUN", ["modelQuality"] = "NOT_RUN", ["performance"] = "NOT_RUN", ["records"] = records };
        try
        {
            Need(BitConverter.IsLittleEndian, "Little-endian fixture bytes required.");
            if (args.SequenceEqual(new[] { "--self-test" })) { SelfTest(); return 0; }
            if (args.Length == 0) { Console.WriteLine("PLAN: --execute --module FILE --expect-sha256 SHA --out FRESH_IGNORED_DIR; --self-test is CPU-only."); return 0; }
            Need(args.Length == 7 && args[0] == "--execute" && args[1] == "--module" && args[3] == "--expect-sha256" && args[5] == "--out", "Exact explicit arguments required.");
            string modulePath = Path.GetFullPath(args[2]), moduleSha = args[4].ToLowerInvariant(); Need(moduleSha.Length == 64 && moduleSha.All(Uri.IsHexDigit) && FileSha(modulePath) == moduleSha, "Pinned module identity differs.");
            output = args[6]; Fresh(output); Directory.CreateDirectory(output); owned = true;
            state["modulePath"] = modulePath; state["moduleSha256"] = moduleSha; state["toolAssemblySha256"] = FileSha(typeof(Program).Assembly.Location); state["backendAssemblySha256"] = FileSha(typeof(CudaBackend).Assembly.Location);
            state["scope"] = "Placement-only finiteF32→RNBF16 and rawu32 original→TS semantic row scatter; compact T1/7/8, no projection/SwiGLU/model/quality/performance qualification.";
            void Checkpoint() => Save(Path.Combine(output!, "result.json"), state); Checkpoint();
            var first = new Dictionary<string, string>(); var files = new Dictionary<string, string>(); long written = 0; int expectedCases = Fixtures().Count(); Need(expectedCases == 57, "Expected fixture count differs.");
            CudaBackend.Register(); CudaDriverApi.cuInit(0).ThrowOnError();
            using (var context = CudaContext.Create(0))
            {
                CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError(); CudaDriverApi.cuDriverGetVersion(out int driver).ThrowOnError(); byte[] name = new byte[256]; CudaDriverApi.cuDeviceGetName(name, name.Length, device).ThrowOnError(); state["device"] = new { ordinal = device, name = System.Text.Encoding.UTF8.GetString(name).TrimEnd('\0'), driverVersion = driver }; state["GPU"] = "EXECUTED";
                CudaDriverApi.cuStreamCreate(out var stream, 0).ThrowOnError();
                try
                {
                    for (int lifetime = 0; lifetime < 2; lifetime++)
                    {
                        using var module = new CudaNinferFlashInferBridge(modulePath); Need(Convert.FromHexString(module.ImageSha256).AsSpan().SequenceEqual(Convert.FromHexString(moduleSha)), "Loaded module image differs.");
                        foreach (var f in Fixtures()) { written += 3L * (f.Expected.Length + 524); Need(written <= 1L << 30, "Fixed raw-output budget exceeded."); Run(module, f, lifetime, stream, output, records, first, files, Checkpoint); }
                    }
                    CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
                }
                finally { CudaDriverApi.cuStreamDestroy(stream).ThrowOnError(); }
            }
            Need(records.Count == 114 && first.Count == 57 && records.Cast<Dictionary<string, object?>>().Select(r => (string)r["id"]!).Distinct().Count() == 114 && records.Cast<Dictionary<string, object?>>().All(r => (bool)r["passed"]! && ((List<object>)r["passes"]!).Count == 3), "All fixtures/lifetimes/poison passes are mandatory.");
            Need(files.Count == 456 && files.All(f => FileSha(Path.Combine(output, f.Key)) == f.Value), "Retained full input/expected/output evidence differs before completion.");
            Need(FileSha(modulePath) == moduleSha, "Module source changed."); state["complete"] = true; state["passed"] = true; state["fixtures"] = 57; state["moduleLifetimes"] = 2; state["actualKernelLaunches"] = 342; state["rawOutputBudgetBytes"] = 1L << 30; Checkpoint(); Complete(output, state);
            Console.WriteLine(JsonSerializer.Serialize(new { complete = true, passed = true, fixtures = 57, launches = 342, modelQuality = "NOT_RUN", performance = "NOT_RUN" })); return 0;
        }
        catch (Exception e) { state["passed"] = false; state["error"] = e.ToString(); if (owned && output != null) Save(Path.Combine(output, "result.json"), state); Console.Error.WriteLine(e); return 1; }
    }
}
