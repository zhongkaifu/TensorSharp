// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using TensorSharp.Validation;
using R = TensorSharp.Validation.NinferDraftShortlistReference;

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
        Action<CudaNinferDraftShortlistModule, CudaNinferDraftModule, Dictionary<string, IntPtr>> Run, object Contract);
    private static byte[] Bytes<T>(T[] array) where T : unmanaged => MemoryMarshal.AsBytes(array.AsSpan()).ToArray();
    private static string Sha(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    private static void Need(bool ok, string message) { if (!ok) throw new InvalidDataException(message); }
    private static ushort[] Values(int n, int salt = 0) => Enumerable.Range(0, n).Select(i => R.Bf((((i * 19 + salt * 23 + i / 7) % 127) - 63) / 32f)).ToArray();
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
    private static void RunCase(CudaNinferDraftShortlistModule module, CudaNinferDraftModule draft, Case c, int lifetime, string folder, List<object> records, Dictionary<string, string> firstLifetimeBodies, Action checkpoint)
    {
        string id = $"lifetime{lifetime}-{c.Name}";
        Need(!records.Cast<Dictionary<string, object?>>().Any(r => (string)r["id"]! == id), "Duplicate lifecycle identity would overwrite original evidence.");
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
            c.Run(module, draft, bodyPointers);
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
                bool crossLifetime = true; string? writtenSha = null;
                if (p.Value.ScratchBytes == 0)
                {
                    writtenSha = Sha(body.ToArray()); string key = c.Name + "/" + p.Key;
                    if (lifetime == 0) firstLifetimeBodies.TryAdd(key, writtenSha);
                    else crossLifetime = firstLifetimeBodies.TryGetValue(key, out var captured) && captured == writtenSha;
                }
                all &= different == 0 && boundFailures == 0 && nonfinite == 0 && guards && repeat && crossLifetime;
                outputs.Add(new { name = p.Key, file, sha256 = Sha(actual), differentBytesFromExact = p.Value.Exact == null ? (int?)null : different, boundFailures, nonfinite, guardsIntact = guards, repeatBits = p.Value.ScratchBytes > 0 ? (bool?)null : repeat, scratchGuardsOnly = p.Value.ScratchBytes > 0, writtenSha256 = writtenSha, crossLifetimeCompared = lifetime > 0 && p.Value.ScratchBytes == 0, sameBitsAsFirstLifetime = lifetime > 0 && p.Value.ScratchBytes == 0 ? (bool?)crossLifetime : null, maxAbsoluteError = maxError, maxBoundFraction });
            }
            bool immutable = true;
            foreach (var p in c.Inputs.Where(p => !c.Outputs.ContainsKey(p.Key))) immutable &= Memory.Read(pointers[p.Key], p.Value.Length + 2 * Guard).AsSpan().SequenceEqual(Guarded(p.Value));
            passes.Add(new { pass, poison, outputs, immutableInputAllocations = immutable, passed = all && immutable }); checkpoint();
            Need(all && immutable, id + ": exact/bound/guard/immutability/repeat gate failed; full outputs retained.");
        }
        record["passed"] = true; checkpoint();
    }
    public static int Main(string[] args)
    {
        string? output = null; bool owned = false; var records = new List<object>(); var firstLifetimeBodies = new Dictionary<string, string>();
        var state = new Dictionary<string, object?> { ["schema"] = "ninfer-original-shortlist-component-probe/1", ["complete"] = false, ["GPU"] = "NOT_RUN", ["modelQuality"] = "NOT_RUN", ["servingIntegration"] = "NOT_RUN", ["performance"] = "NOT_RUN", ["records"] = records };
        try
        {
            Need(BitConverter.IsLittleEndian, "Little-endian binary capture format required.");
            if (args.SequenceEqual(new[] { "--self-test-all" })) { SelfTest(true); return 0; }
            if (args.SequenceEqual(new[] { "--self-test" })) { SelfTest(false); return 0; }
            if (args.Length == 0) { Console.WriteLine("PLAN: --execute --module ABSOLUTE_SHORTLIST_MODULE --draft-module ABSOLUTE_DRAFT_MODULE --out FRESH_IGNORED_DIRECTORY; --self-test CPU only."); return 0; }
            Need(args.Length == 7 && args[0] == "--execute" && args[1] == "--module" && args[3] == "--draft-module" && args[5] == "--out", "Use --execute --module FILE --draft-module FILE --out FRESH_IGNORED_DIRECTORY.");
            string path = Path.GetFullPath(args[2]), draftPath = Path.GetFullPath(args[4]); output = args[6]; Fresh(output); Need(File.Exists(path) && File.Exists(draftPath), "Native shortlist/draft module unavailable.");
            Directory.CreateDirectory(output); owned = true; state["modulePath"] = path; state["moduleSha256"] = Sha(File.ReadAllBytes(path));
            state["draftModulePath"] = draftPath; state["draftModuleSha256"] = Sha(File.ReadAllBytes(draftPath));
            state["probeAssemblySha256"] = Sha(File.ReadAllBytes(typeof(Program).Assembly.Location)); state["backendAssemblySha256"] = Sha(File.ReadAllBytes(typeof(CudaBackend).Assembly.Location));
            state["projectionCases"] = 43; state["mapPrimitiveCases"] = 5; state["uniqueCases"] = 48; state["moduleLifetimes"] = 2; state["poisonPassesPerCase"] = 3;
            state["scope"] = "Synthetic full131072x5120 original Q4G64/F16S head; original dispatch/rounding; projection→Top16→globalI32map composition. No model/speculative/cache quality or speed qualification.";
            void Checkpoint() => Save(Path.Combine(output!, "result.json"), state); Checkpoint();
            CudaBackend.Register(); CudaDriverApi.cuInit(0).ThrowOnError(); using var context = CudaContext.Create(0); state["GPU"] = "EXECUTED";
            for (int lifetime = 0; lifetime < 2; lifetime++)
            {
                using var module = new CudaNinferDraftShortlistModule(path); using var draft = new CudaNinferDraftModule(draftPath);
                Need(ValidationSha256.Matches(draft.ImageSha256, state["draftModuleSha256"]!.ToString()!), "Loaded Top16 module image differs."); Need(ValidationSha256.Matches(module.ImageSha256, state["moduleSha256"]!.ToString()!), "Loaded module image differs.");
                foreach (var c in AllCases()) RunCase(module, draft, c, lifetime, output, records, firstLifetimeBodies, Checkpoint);
            }
            var lifecycle = records.Cast<Dictionary<string, object?>>().ToArray();
            Need(lifecycle.Length == 96 && lifecycle.Select(r => (string)r["id"]!).Distinct().Count() == 96, "All96 unique lifecycle records are required.");
            var identities = lifecycle.GroupBy(r => (string)r["case"]!).ToArray();
            Need(identities.Length == 48 && identities.All(g => g.Count() == 2 && g.Select(r => (int)r["lifetime"]!).Order().SequenceEqual(new[] { 0, 1 })
                && g.All(r => (bool)r["passed"]! && ((List<object>)r["passes"]!).Count == 3)), "Every original case must complete both lifetimes and all poison passes.");
            state["actualLifecycleRecords"] = 96; state["actualDistinctCases"] = 48; state["crossLifetimeNonScratchOutputBits"] = "PASS";
            Need(Sha(File.ReadAllBytes(path)) == state["moduleSha256"]?.ToString() && Sha(File.ReadAllBytes(draftPath)) == state["draftModuleSha256"]?.ToString(), "Module files changed during execution.");
            state["complete"] = true; state["passed"] = true; Checkpoint(); SaveCompleted(output, state); Console.WriteLine(JsonSerializer.Serialize(new { complete = true, records = records.Count, modelQuality = "NOT_RUN", servingIntegration = "NOT_RUN", performance = "NOT_RUN", output, authoritativeResult = "completed-result.json" })); return 0;
        }
        catch (Exception e) { state["error"] = e.ToString(); state["passed"] = false; if (owned && output != null) Save(Path.Combine(output, "result.json"), state); Console.Error.WriteLine(e); return 1; }
    }
}
