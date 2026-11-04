// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Validation;

// Public engine APIs only. This isolates graph submission, not A16 arithmetic,
// model task quality, sampling, HTTP serving, or cross-engine performance.
internal static class Program
{
    private const string GraphEnvironment = "TS_Q35_NINFER_DECODE_GRAPHS";
    private const string GraphSuffix = "|ninfer-decode-graphs=v1";
    private const int GuardFloats = 64, GuardBits = 0x4e67a912, GuardInt = 0x6173942e;
    private const long CaptureBudget = 384L << 20;
    private const int Steps = 3, Passes = 2;
    private static readonly int[] Lanes = { 1, 2, 3, 4, 8, 16 };
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
    private sealed record InputPool(string Id, int[] Prompt, int[] Continuation);
    private sealed record Fixture(string Id, int[][] Prompts, int[][] DecodeTokens, int CloneToken, int[] StressTokens);
    private sealed record Pin(string Path, long Bytes, string Sha256);
    private sealed record Counters(int Resident, long Captures, long Replays, long Evictions);
    private sealed record Reference(string Sha256, string File, int Elements);
    private static void Need(bool value, string message) { if (!value) throw new InvalidDataException(message); }
    private static string Hash(ReadOnlySpan<byte> bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    private static Pin FilePin(string path)
    {
        using var file = File.OpenRead(path);
        return new(Path.GetFullPath(path), file.Length, Convert.ToHexString(SHA256.HashData(file)).ToLowerInvariant());
    }
    private static void Save(string path, object value)
    {
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        JsonSerializer.Serialize(stream, value, Json); stream.WriteByte(10);
    }
    private static void SaveCompleted(string output, object value)
    {
        byte[] json = JsonSerializer.SerializeToUtf8Bytes(value, Json), bytes = new byte[json.Length + 1];
        json.CopyTo(bytes, 0); bytes[^1] = 10;
        using (var stream = new FileStream(Path.Combine(output, "completed-result.json"), FileMode.CreateNew, FileAccess.Write, FileShare.None))
        { stream.Write(bytes); stream.Flush(flushToDisk: true); }
        byte[] sidecar = System.Text.Encoding.ASCII.GetBytes(Hash(bytes) + "  completed-result.json\n");
        using var digest = new FileStream(Path.Combine(output, "completed-result.json.sha256"), FileMode.CreateNew, FileAccess.Write, FileShare.None);
        digest.Write(sidecar); digest.Flush(flushToDisk: true);
    }
    private static bool IsLink(string path) => new FileInfo(path).LinkTarget != null || new DirectoryInfo(path).LinkTarget != null;
    private static void Fresh(string path)
    {
        Need(Path.IsPathFullyQualified(path) && Path.GetFullPath(path) == path, "Canonical absolute output required.");
        string[] parts = path.Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries);
        Need(parts.Contains("artifacts", StringComparer.Ordinal) || Enumerable.Range(0, parts.Length - 1).Any(i => parts[i] == "docs" && parts[i + 1] == "validation"), "Ignored evidence output required.");
        Need(!Path.Exists(path) && !IsLink(path), "Fresh output required.");
        for (string? p = Path.GetDirectoryName(path); p != null; p = Path.GetDirectoryName(p))
            Need(!IsLink(p) && !File.Exists(p), "Aliased output parent refused.");
    }
    private static Dictionary<string, string> Options(string[] args)
    {
        var options = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i++)
        {
            string key = args[i];
            Need(key is "--execute" or "--self-test" or "--artifact" or "--expect-directory-sha" or "--inputs" or "--expect-inputs-sha" or "--out" or "--context" or "--ubatch", "Unknown option: " + key);
            Need(!options.ContainsKey(key), "Duplicate option: " + key);
            if (key is "--execute" or "--self-test") options.Add(key, "1");
            else { Need(++i < args.Length && !string.IsNullOrWhiteSpace(args[i]), "Missing option value."); options.Add(key, args[i]); }
        }
        Need(!(options.ContainsKey("--execute") && options.ContainsKey("--self-test")), "Exclusive execution modes required.");
        return options;
    }
    private static string Required(Dictionary<string, string> args, string key) => args.TryGetValue(key, out var value) ? value : throw new ArgumentException("Missing " + key);
    private static int Positive(Dictionary<string, string> args, string key, int fallback)
    {
        if (!args.TryGetValue(key, out string? value)) return fallback;
        Need(int.TryParse(value, out int result) && result > 0, "Positive integer required: " + key); return result;
    }
    private static object ArtifactStat(string path)
    {
        var file = new FileInfo(path); Need(file.Exists && !IsLink(path), "Original regular artifact required.");
        return new { file.FullName, file.Length, file.LastWriteTimeUtc, file.CreationTimeUtc };
    }
    private static void Unique(JsonElement element)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (var property in element.EnumerateObject()) { Need(names.Add(property.Name), "Duplicate input JSON field."); Unique(property.Value); }
        }
        else if (element.ValueKind == JsonValueKind.Array) foreach (var item in element.EnumerateArray()) Unique(item);
    }
    private static InputPool ReadInputs(byte[] bytes, int valid)
    {
        using var doc = JsonDocument.Parse(bytes); var root = doc.RootElement; Unique(root);
        Need(root.EnumerateObject().Select(p => p.Name).Order().SequenceEqual(new[] { "cases", "schema" })
            && root.GetProperty("schema").GetString() == "ninfer-qwen38-body-inputs/1", "Pinned body input schema required.");
        var pools = new List<InputPool>(); var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (var row in root.GetProperty("cases").EnumerateArray())
        {
            Need(row.EnumerateObject().Select(p => p.Name).Order().SequenceEqual(new[] { "chunkSize", "continuationTokens", "id", "promptTokens" }), "Exact input case fields required.");
            string id = row.GetProperty("id").GetString() ?? "";
            Need(id.Length is > 0 and <= 80 && id.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_') && ids.Add(id), "Portable unique input case IDs required.");
            int[] prompt = row.GetProperty("promptTokens").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            int[] continuation = row.GetProperty("continuationTokens").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            Need(prompt.Length is > 0 and <= 16384 && continuation.Length is >= 4 and <= 128 && row.GetProperty("chunkSize").GetInt32() > 0, "Bounded fixed token input required.");
            Need(prompt.Concat(continuation).All(x => x >= 0 && x < valid), "Invalid/padded token in fixed input.");
            pools.Add(new(id, prompt, continuation));
        }
        Need(pools.Count is > 0 and <= 128, "Bounded input case count required.");
        var pool = pools.OrderByDescending(x => x.Prompt.Length).ThenBy(x => x.Id, StringComparer.Ordinal).First();
        Need(pool.Prompt.Length >= 4096, "A pinned original prompt of at least4096 tokens is required."); return pool;
    }
    private static int[] Cycle(int[] source, int length, int offset) => Enumerable.Range(0, length).Select(i => source[(i + offset) % source.Length]).ToArray();
    private static Fixture[] Fixtures(InputPool source, int ubatch)
    {
        var fixtures = new List<Fixture>();
        foreach (int lanes in Lanes) foreach (int prefix in new[] { 127, 255 })
        {
            int ordinal = fixtures.Count;
            int[][] prompts = Enumerable.Range(0, lanes).Select(i => Cycle(source.Prompt, prefix - i % 3, ordinal * 11 + i * 19)).ToArray();
            int[][] decode = Enumerable.Range(0, Steps).Select(step => Enumerable.Range(0, lanes)
                .Select(i => source.Continuation[(ordinal + step + i * 3) % source.Continuation.Length]).ToArray()).ToArray();
            fixtures.Add(new($"lanes{lanes}-past{prefix}", prompts, decode,
                source.Continuation[(ordinal + Steps) % source.Continuation.Length], Cycle(source.Prompt, ubatch, ordinal * 23 + 113)));
        }
        int last = fixtures.Count;
        fixtures.Add(new("lanes1-past8191", new[] { Cycle(source.Prompt, 8191, last * 11) },
            Enumerable.Range(0, Steps).Select(i => new[] { source.Continuation[(last + i) % source.Continuation.Length] }).ToArray(),
            source.Continuation[(last + Steps) % source.Continuation.Length], Cycle(source.Prompt, ubatch, last * 23 + 113)));
        return fixtures.ToArray();
    }
    private static Counters GraphCounters(Q35CudaEngine engine) => new(engine.CapturedNinferDecodeGraphCount,
        engine.NinferDecodeGraphCaptureCount, engine.NinferDecodeGraphReplayCount, engine.NinferDecodeGraphEvictionCount);
    private static Dictionary<string, string?> EnvironmentSnapshot() => Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>()
        .Where(p => ((string)p.Key).StartsWith("TS_Q35_", StringComparison.Ordinal) || (string)p.Key is "MAX_CONTEXT" or "CUDA_VISIBLE_DEVICES" or "CUDA_CACHE_DISABLE" or "LD_LIBRARY_PATH")
        .OrderBy(p => (string)p.Key, StringComparer.Ordinal).ToDictionary(p => (string)p.Key, p => (string?)p.Value, StringComparer.Ordinal);
    private static string StableEnvironment(Dictionary<string, string?> environment)
        => JsonSerializer.Serialize(environment.Where(p => p.Key != GraphEnvironment).ToDictionary(p => p.Key, p => p.Value), Json);
    private static Pin[] NativePins(Dictionary<string, string?> environment)
    {
        var files = new HashSet<string>(StringComparer.Ordinal);
        foreach (string key in new[] { "TS_Q35_NINFER_PTX", "TS_Q35_W8_HEAD_PTX", "TS_Q35_NINFER_A16_PTX", "TS_Q35_NINFER_REPACK_PTX" })
        {
            if (!environment.TryGetValue(key, out var configured)) continue;
            Need(!string.IsNullOrWhiteSpace(configured) && Path.IsPathFullyQualified(configured), "Explicit absolute native module paths required when configured.");
            string path = Path.GetFullPath(configured!); Need(File.Exists(path), "Configured native module unavailable: " + key); files.Add(path);
            if (path.EndsWith(".ptx", StringComparison.OrdinalIgnoreCase))
                foreach (string sibling in new[] { Path.ChangeExtension(path, ".fatbin"), Path.ChangeExtension(path, ".fatbin") + ".sha256" })
                    if (File.Exists(sibling)) files.Add(sibling);
        }
        // These are found by normal runtime lookup, independently of optional paths.
        foreach (string name in new[] { "tensorsharp_q35_kernels", "tensorsharp_q35_ninfer_attention", "tensorsharp_q35_ninfer", "tensorsharp_w8g32_head" })
            foreach (string directory in new[] { AppContext.BaseDirectory, Path.Combine(AppContext.BaseDirectory, "cuda_kernels") })
                foreach (string extension in new[] { ".ptx", ".fatbin", ".fatbin.sha256" })
                { string file = Path.Combine(directory, name + extension); if (File.Exists(file)) files.Add(file); }
        return files.Order(StringComparer.Ordinal).Select(FilePin).ToArray();
    }
    private static Pin[] AssemblyPins() => new[] { typeof(Program).Assembly, typeof(Qwen35CudaExecutor).Assembly,
        typeof(Q35CudaEngine).Assembly, typeof(NinferArtifactReader).Assembly }.Select(a => FilePin(a.Location)).ToArray();
    private static void VerifyPins(IEnumerable<Pin> pins)
    {
        foreach (var pin in pins) Need(FilePin(pin.Path) == pin, "Source/runtime/native file changed: " + pin.Path);
    }

    private sealed class Captures
    {
        private readonly string _output;
        private readonly Dictionary<string, Reference> _baseline = new(StringComparer.Ordinal), _first = new(StringComparer.Ordinal);
        private readonly Dictionary<string, int> _stored = new(StringComparer.Ordinal);
        private long _bytes;
        internal readonly List<object> Records = new();
        internal int OnComparisons, RepeatComparisons, CloneComparisons;
        internal long ScalarValuesChecked;
        internal long StoredBytes => _bytes;
        internal Captures(string output) { _output = output; Directory.CreateDirectory(Path.Combine(output, "vectors")); }
        internal void VerifyStoredVectors()
        {
            foreach (var pair in _stored)
            {
                var pin = FilePin(Path.Combine(_output, "vectors", pair.Key + ".f32"));
                Need(pin.Bytes == pair.Value && ValidationSha256.Matches(pin.Sha256, pair.Key), "A retained raw vector changed before finalization.");
            }
        }
        internal string Capture(string mode, Fixture fixture, int pass, string phase, float[] values, int rows, int vocab, int valid,
            bool guards, double elapsed, Counters before, Counters after, int[]? positions = null, int[]? suppliedArgmax = null, string? requireSha = null)
        {
            int elements = checked(rows * vocab); Need(values.Length >= elements, "Output shape failed.");
            var raw = MemoryMarshal.AsBytes(values.AsSpan(0, elements)); string digest = Hash(raw);
            string relative = Path.Combine("vectors", digest + ".f32"), file = Path.Combine(_output, relative);
            if (!File.Exists(file))
            {
                Need(_bytes + raw.Length <= CaptureBudget, "Full-logit capture budget exceeded; no passing result.");
                using var stream = new FileStream(file, FileMode.CreateNew, FileAccess.Write, FileShare.None); stream.Write(raw); _bytes += raw.Length;
            }
            else Need(FilePin(file).Sha256 == digest && new FileInfo(file).Length == raw.Length, "Stored vector identity differs.");
            _stored[digest] = raw.Length;
            var rowStats = new List<object>();
            var record = new Dictionary<string, object?>
            {
                ["mode"] = mode, ["fixtureId"] = fixture.Id, ["pass"] = pass, ["phase"] = phase, ["rows"] = rows,
                ["elements"] = elements, ["file"] = relative, ["sha256"] = digest, ["passed"] = false,
                ["guardsIntact"] = guards, ["rowStats"] = rowStats, ["graphCountersBefore"] = before,
                ["graphCountersAfter"] = after, ["graphReplayDelta"] = after.Replays - before.Replays,
                ["wallMillisecondsDiagnosticOnly"] = elapsed
            };
            Records.Add(record);
            Need(guards, "Caller output tail guard changed; complete vector retained.");
            for (int r = 0; r < rows; r++)
            {
                int at = r * vocab, argmax = 0, headArgmax = 0;
                for (int c = 0; c < vocab; c++)
                {
                    float value = values[at + c]; Need(float.IsFinite(value), "Nonfinite physical-head logit; complete capture retained.");
                    if (value > values[at + headArgmax]) headArgmax = c;
                    if (c < valid && value > values[at + argmax]) argmax = c;
                }
                Need(headArgmax < valid, "A padded physical-head row wins; no sampling/quality success.");
                if (suppliedArgmax != null) Need(suppliedArgmax[r] == headArgmax, "Engine argmax differs from complete physical-head vector.");
                rowStats.Add(new { row = r, argmax, headArgmax, paddedRowWins = false, position = positions?[r] });
            }
            string key = fixture.Id + "/" + phase; bool paired = mode == "on", repeat = pass == 1;
            if (mode == "off" && pass == 0) Need(_baseline.TryAdd(key, new(digest, relative, elements)), "Duplicate baseline capture identity.");
            else
            {
                Need(_baseline.TryGetValue(key, out var expected), "Missing OFF reference capture.");
                Need(expected!.Elements == elements && ValidationSha256.Matches(expected.Sha256, digest)
                    && File.ReadAllBytes(Path.Combine(_output, expected.File)).AsSpan().SequenceEqual(raw), "OFF/ON or OFF reset-repeat full bytes differ; actual vector retained.");
                if (paired) OnComparisons++;
            }
            string firstKey = mode + "/" + key;
            if (pass == 0) _first[firstKey] = new(digest, relative, elements);
            else
            {
                Need(_first.TryGetValue(firstKey, out var first) && ValidationSha256.Matches(first.Sha256, digest), "Same-mode reset-repeat differs.");
                RepeatComparisons++;
            }
            if (requireSha != null) { Need(ValidationSha256.Matches(requireSha, digest), "Original/clone full logits differ."); CloneComparisons++; }
            record["offOnExact"] = paired ? true : null; record["resetRepeatExact"] = repeat ? true : null;
            record["cloneExact"] = requireSha != null ? true : null;
            if (phase.StartsWith("decode-step", StringComparison.Ordinal) || phase.StartsWith("clone-", StringComparison.Ordinal))
                RequireGraphCall(mode == "on", before, after);
            record["passed"] = true;
            ScalarValuesChecked += elements; return digest;
        }
    }
    private static float[] Poisoned(int elements, int pass)
    {
        var values = Enumerable.Repeat(BitConverter.Int32BitsToSingle(pass == 0 ? unchecked((int)0x7fc13a9d) : unchecked((int)0x7fc23a9d)), checked(elements + GuardFloats)).ToArray();
        for (int i = elements; i < values.Length; i++) values[i] = BitConverter.Int32BitsToSingle(GuardBits); return values;
    }
    private static bool TailGuard(float[] values, int elements) => values.AsSpan(elements).ToArray().All(x => BitConverter.SingleToInt32Bits(x) == GuardBits);
    private static void RequireGraphCall(bool graphs, Counters before, Counters after)
    {
        if (graphs) Need(after.Replays == before.Replays + 1, "Eligible ON DecodeBatch did not execute exactly one graph; fallback cannot pass.");
        else Need(after.Captures == 0 && after.Replays == 0 && after.Resident == 0, "OFF control unexpectedly used a graph.");
    }
    private static string Decode(Captures captures, bool graphs, Q35CudaEngine engine, Fixture fixture, int pass, string phase,
        Q35CudaEngine.Slot[] slots, int[] tokens, int valid, string? requireSha = null)
    {
        int vocab = engine.VocabSize, elements = checked(slots.Length * vocab);
        float[] logits = Poisoned(elements, pass); var argmax = Enumerable.Repeat(GuardInt, slots.Length + 16).ToArray();
        int[] unchanged = tokens.ToArray(), positions = slots.Select(s => s.NPast).ToArray(); var before = GraphCounters(engine);
        var timer = Stopwatch.StartNew(); engine.DecodeBatch(slots, tokens, logits, argmax); double elapsed = timer.Elapsed.TotalMilliseconds;
        var after = GraphCounters(engine);
        Need(tokens.AsSpan().SequenceEqual(unchanged), "Caller token input changed.");
        Need(slots.Select((s, i) => s.NPast == positions[i] + 1).All(x => x), "Decode position did not advance exactly once.");
        return captures.Capture(graphs ? "on" : "off", fixture, pass, phase, logits, slots.Length, vocab, valid,
            TailGuard(logits, elements) && argmax.AsSpan(slots.Length).ToArray().All(x => x == GuardInt), elapsed,
            before, after, slots.Select(s => s.NPast).ToArray(), argmax, requireSha);
    }
    private static void Prefill(Captures captures, bool graphs, Q35CudaEngine engine, Fixture fixture, int pass,
        Q35CudaEngine.Slot[] slots, int valid)
    {
        foreach (var slot in slots) { engine.ResetSlot(slot); Need(slot.NPast == 0, "Reset failed."); }
        var final = new float[slots.Length][]; var before = GraphCounters(engine); var timer = Stopwatch.StartNew();
        for (int start = 0; start < fixture.Prompts.Max(x => x.Length); start += engine.UBatch)
        {
            int offset = start;
            int[] indices = Enumerable.Range(0, slots.Length).Where(i => fixture.Prompts[i].Length > offset).ToArray();
            int[][] chunks = indices.Select(i => fixture.Prompts[i].AsSpan(offset, Math.Min(engine.UBatch, fixture.Prompts[i].Length - offset)).ToArray()).ToArray();
            int[][] unchanged = chunks.Select(x => x.ToArray()).ToArray(); var heads = new float[indices.Length][];
            engine.PrefillBatch(indices.Select(i => slots[i]).ToArray(), chunks, heads);
            for (int j = 0; j < indices.Length; j++)
            {
                Need(chunks[j].AsSpan().SequenceEqual(unchanged[j]), "Prefill input changed.");
                Need(heads[j].Length == engine.VocabSize, "Prefill full-head output is absent."); final[indices[j]] = heads[j];
            }
        }
        double elapsed = timer.Elapsed.TotalMilliseconds; var after = GraphCounters(engine);
        Need(before.Replays == after.Replays && before.Captures == after.Captures, "Count-bearing PrefillBatch unexpectedly entered decode graph.");
        Need(slots.Select((s, i) => s.NPast == fixture.Prompts[i].Length).All(x => x), "Prefill position differs from fixed token input.");
        float[] flat = final.SelectMany(x => x).ToArray();
        captures.Capture(graphs ? "on" : "off", fixture, pass, "prefill", flat, slots.Length, engine.VocabSize, valid, true,
            elapsed, before, after, slots.Select(s => s.NPast).ToArray());
    }
    private static void Clone(Captures captures, bool graphs, Q35CudaEngine engine, Fixture fixture, int pass,
        Q35CudaEngine.Slot source, int valid)
    {
        var clone = engine.CreateSlot(0);
        try
        {
            engine.CopySlot(source, clone); Need(source.NPast == clone.NPast, "Clone position differs.");
            string original = Decode(captures, graphs, engine, fixture, pass, "clone-original", new[] { source }, new[] { fixture.CloneToken }, valid);
            Decode(captures, graphs, engine, fixture, pass, "clone-copy", new[] { clone }, new[] { fixture.CloneToken }, valid, original);
        }
        finally { engine.Active = source; engine.FreeSlot(clone); }
    }
    private static void LargerPrefill(Captures captures, bool graphs, Q35CudaEngine engine, Fixture fixture, int valid,
        Q35CudaEngine.Slot restore)
    {
        var scratchSlot = engine.CreateSlot(0); var before = GraphCounters(engine);
        try
        {
            Need(fixture.StressTokens.Length == engine.UBatch && engine.UBatch > Lanes.Max(), "Stress must exceed every decode row count.");
            engine.Active = scratchSlot; float[] logits = Poisoned(engine.VocabSize, 0); int[] unchanged = fixture.StressTokens.ToArray();
            var timer = Stopwatch.StartNew(); engine.Forward(fixture.StressTokens, logits); double elapsed = timer.Elapsed.TotalMilliseconds;
            var after = GraphCounters(engine);
            Need(before.Replays == after.Replays && before.Captures == after.Captures, "Larger prefill unexpectedly used decode graph.");
            Need(fixture.StressTokens.AsSpan().SequenceEqual(unchanged) && scratchSlot.NPast == fixture.StressTokens.Length, "Larger prefill input/position differs.");
            captures.Capture(graphs ? "on" : "off", fixture, 0, "larger-prefill-between-repeats", logits, 1, engine.VocabSize, valid,
                TailGuard(logits, engine.VocabSize), elapsed, before, after, new[] { scratchSlot.NPast });
        }
        finally { engine.Active = restore; engine.FreeSlot(scratchSlot); }
    }
    private static object RunMode(bool graphs, string artifact, int context, int ubatch, int valid, Fixture[] fixtures,
        Captures captures, string stableEnvironment, Pin[] nativePins, Pin[] assemblies, ref string? arithmeticNamespace, Action persist, Action modelLoaded)
    {
        Environment.SetEnvironmentVariable(GraphEnvironment, graphs ? "1" : "0");
        Need(StableEnvironment(EnvironmentSnapshot()) == stableEnvironment, "A non-graph environment option changed.");
        VerifyPins(nativePins); VerifyPins(assemblies);
        var load = Stopwatch.StartNew();
        using var executor = new Qwen35CudaExecutor(artifact, context, ubatch); double loadMs = load.Elapsed.TotalMilliseconds;
        modelLoaded();
        var engine = executor.Engine; var owned = new List<Q35CudaEngine.Slot>(); var first = engine.Active;
        Need(executor.Draft == null && engine.FeatureWidth == 0 && !engine.NinferGdnChunkEnabled, "Graph-only text route required.");
        Need(engine.VocabSize == 248320 && engine.ContextSize == context && engine.UBatch == ubatch, "Executor geometry differs.");
        var initial = GraphCounters(engine); Need(initial == new Counters(0, 0, 0, 0), "Fresh graph counters required.");
        string stateNamespace = engine.KVStateNamespaceSuffix;
        Need(stateNamespace.Contains(GraphSuffix, StringComparison.Ordinal) == graphs, "Graph state namespace does not match requested mode.");
        string mathematical = stateNamespace.Replace(GraphSuffix, "", StringComparison.Ordinal);
        if (arithmeticNamespace == null) arithmeticNamespace = mathematical;
        else Need(arithmeticNamespace == mathematical, "A non-graph arithmetic/module state policy changed between modes.");
        try
        {
            var slots = new Q35CudaEngine.Slot[Lanes.Max()]; slots[0] = first;
            for (int i = 1; i < slots.Length; i++) { slots[i] = engine.CreateSlot(0); owned.Add(slots[i]); }
            foreach (var fixture in fixtures)
            {
                var group = slots.Take(fixture.Prompts.Length).ToArray();
                for (int pass = 0; pass < Passes; pass++)
                {
                    var passStart = GraphCounters(engine);
                    Prefill(captures, graphs, engine, fixture, pass, group, valid); persist();
                    for (int step = 0; step < Steps; step++)
                    {
                        Decode(captures, graphs, engine, fixture, pass, "decode-step" + step, group, fixture.DecodeTokens[step], valid); persist();
                    }
                    Clone(captures, graphs, engine, fixture, pass, group[0], valid); persist();
                    if (graphs && pass == 1)
                        Need(GraphCounters(engine).Captures == passStart.Captures, "Reset-repeat recaptured graphs after larger prefill; stable replay was not demonstrated.");
                    if (pass == 0) { LargerPrefill(captures, graphs, engine, fixture, valid, first); persist(); }
                }
            }
            var final = GraphCounters(engine); int decodeCalls = fixtures.Length * Passes * (Steps + 2);
            if (graphs) Need(final.Captures > 0 && final.Resident > 0 && final.Replays == decodeCalls && final.Replays > final.Captures,
                "Actual graph capture and later replay coverage are mandatory.");
            else Need(final == initial, "OFF graph counters changed.");
            Need(StableEnvironment(EnvironmentSnapshot()) == stableEnvironment, "Non-graph options changed during mode.");
            VerifyPins(nativePins); VerifyPins(assemblies);
            return new { mode = graphs ? "on" : "off", complete = true, stateNamespace, arithmeticNamespace = mathematical,
                graphCountersBefore = initial, graphCountersAfter = final, eligibleDecodeBatchCalls = decodeCalls,
                nUbatchRowPrefillsBetweenResetRepeats = fixtures.Length, clonedContinuationPairs = fixtures.Length * Passes,
                loadWallMillisecondsDiagnosticOnly = loadMs, environment = EnvironmentSnapshot(),
                engine.NinferLoadRepackImageSha256, engine.NinferLoadRepackedMatrices, engine.NinferLoadRepackedOriginalBytes };
        }
        finally
        {
            engine.Active = first;
            foreach (var slot in owned) engine.FreeSlot(slot);
        }
    }
    private static object SelfTest()
    {
        var pool = new InputPool("cpu", Enumerable.Range(0, 4096).Select(i => i % 257).ToArray(), new[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        var fixtures = Fixtures(pool, 512); int contracts = 0;
        Need(fixtures.Length == 13 && fixtures.Take(12).Select(x => x.Prompts.Length).Distinct().SequenceEqual(Lanes), "Lane fixture coverage."); contracts++;
        Need(fixtures.All(x => x.DecodeTokens.Length == Steps && x.DecodeTokens.All(t => t.Length == x.Prompts.Length)), "Decode step coverage."); contracts++;
        Need(fixtures.All(x => x.Prompts.All(t => t.Length + Steps + 1 < 16384) && x.StressTokens.Length == 512), "Context/stress plan."); contracts++;
        Need(fixtures.Last().Prompts[0].Length == 8191 && fixtures.Any(x => x.Prompts[0].Length == 127) && fixtures.Any(x => x.Prompts[0].Length == 255), "Position boundary plan."); contracts++;
        long uniqueRows = fixtures.Sum(x => 4L * x.Prompts.Length + 2);
        Need(uniqueRows == 302 && uniqueRows * 248320 * 4 < CaptureBudget, "Bounded successful raw capture plan."); contracts++;
        var poisoned = Poisoned(17, 0); Need(TailGuard(poisoned, 17), "Tail guard construction."); contracts++;
        poisoned[^1] = 0; Need(!TailGuard(poisoned, 17), "Tail corruption detection."); contracts++;
        RequireGraphCall(false, new(0, 0, 0, 0), new(0, 0, 0, 0)); contracts++;
        RequireGraphCall(true, new(1, 1, 1, 0), new(1, 1, 2, 0)); contracts++;
        bool refusal = false; try { RequireGraphCall(true, new(0, 0, 0, 0), new(0, 0, 0, 0)); } catch (InvalidDataException) { refusal = true; }
        Need(refusal, "Fallback cannot count as graph success."); contracts++;
        string encoded = JsonSerializer.Serialize(new { schema = "ninfer-qwen38-body-inputs/1", cases = new[] { new { id = pool.Id, promptTokens = pool.Prompt, continuationTokens = pool.Continuation, chunkSize = 512 } } });
        byte[] bytes = System.Text.Encoding.UTF8.GetBytes(encoded); Need(ReadInputs(bytes, 257).Prompt.SequenceEqual(pool.Prompt), "Valid fixed source."); contracts++;
        foreach (string invalid in new[] { encoded.Replace("\"chunkSize\":512", "\"chunkSize\":0"), encoded.Replace("\"id\":\"cpu\"", "\"id\":\"../cpu\""), encoded.Replace("\"schema\":", "\"schema\":\"duplicate\",\"schema\":") })
        {
            bool rejected = false; try { ReadInputs(System.Text.Encoding.UTF8.GetBytes(invalid), 257); } catch (InvalidDataException) { rejected = true; }
            Need(rejected, "Malformed source token manifest accepted."); contracts++;
        }
        return new { CPU = "PASS", contracts, fixtures = fixtures.Length, lanes = Lanes, steps = Steps, resetRepeats = Passes,
            potentialUniqueFullHeadRows = uniqueRows, potentialSuccessfulRawBytes = uniqueRows * 248320 * 4, hardCaptureBudgetBytes = CaptureBudget,
            GPU = "NOT_RUN", graphCapture = "NOT_RUN", modelQuality = "NOT_RUN", performance = "NOT_RUN" };
    }
    public static int Main(string[] args)
    {
        string? output = null; bool owned = false; string? originalGraph = Environment.GetEnvironmentVariable(GraphEnvironment);
        Captures? captures = null; var modes = new List<object>();
        var state = new Dictionary<string, object?> { ["schema"] = "ninfer-public-decode-graph-paired-probe/1", ["complete"] = false,
            ["GPU"] = "NOT_RUN", ["modelQuality"] = "NOT_RUN", ["independentModelOracle"] = "NOT_RUN",
            ["endToEndPerformance"] = "NOT_QUALIFIED", ["modes"] = modes };
        void Persist() { if (owned && output != null) Save(Path.Combine(output, "result.json"), state); }
        void FinalizeManifest()
        {
            Persist();
            if (owned && output != null)
                File.WriteAllText(Path.Combine(output, "result.json.sha256"), FilePin(Path.Combine(output, "result.json")).Sha256 + "  result.json\n");
        }
        try
        {
            Need(BitConverter.IsLittleEndian, "Little-endian complete F32 vector format required."); var options = Options(args);
            if (options.ContainsKey("--self-test")) { Console.WriteLine(JsonSerializer.Serialize(SelfTest(), Json)); return 0; }
            if (!options.ContainsKey("--execute"))
            {
                Console.WriteLine("PLAN: --execute --artifact FILE --expect-directory-sha SHA --inputs ORIGINAL_BODY_INPUTS --expect-inputs-sha SHA --out FRESH_LOCAL_IGNORED_DIRECTORY [--context16384] [--ubatch512]. Paired OFF/ON public executor lifetimes; lanes1/2/3/4/8/16, positions127/255 plus8191, three batched steps, reset AA, clone/COW, later larger prefill, full-head byte equality and mandatory actual graph replay. --self-test performs no CUDA/model work."); return 0;
            }
            int context = Positive(options, "--context", 16384), ubatch = Positive(options, "--ubatch", 512);
            Need(context is >= 8196 and <= 262144 && ubatch is >= 256 and <= 2048 && ubatch <= context, "Require context8196..262144 and ubatch256..2048.");
            string artifact = Path.GetFullPath(Required(options, "--artifact")), inputsPath = Path.GetFullPath(Required(options, "--inputs"));
            string directorySha = ValidationSha256.Normalize(Required(options, "--expect-directory-sha")), inputSha = ValidationSha256.Normalize(Required(options, "--expect-inputs-sha"));
            output = Required(options, "--out"); Fresh(output); object before = ArtifactStat(artifact);
            Need(File.Exists(inputsPath) && !IsLink(inputsPath), "Original regular token-input manifest required.");
            byte[] inputBytes = File.ReadAllBytes(inputsPath);
            Need(ValidationSha256.Matches(Hash(inputBytes), inputSha), "Pinned original token-input SHA differs.");
            var initialEnvironment = EnvironmentSnapshot(); string stable = StableEnvironment(initialEnvironment);
            Need(!initialEnvironment.ContainsKey("TS_Q35_NINFER_GDN_CHUNK_PTX"), "Failed/unqualified GDN chunk option must remain outside the graph-only check.");
            Need(!initialEnvironment.TryGetValue("TS_Q35_NINFER_HOST_EMBEDDING", out var host) || host is null or "0", "Graph-only probe requires ordinary GPU embedding placement.");
            var nativePins = NativePins(initialEnvironment); var assemblyPins = AssemblyPins();
            using (var reader = new NinferArtifactReader(artifact))
            {
                Need(ValidationSha256.Matches(reader.DirectorySha256, directorySha), "Original artifact directory differs.");
                var frontend = NinferQwen38Frontend.FromReader(reader); int valid = frontend.Tokenizer.VocabSize;
                Need(frontend.Tokenizer.HeadVocabSize == 248320 && valid == 248077 && context <= frontend.MaximumContextTokens, "Original frontend geometry differs.");
                InputPool pool = ReadInputs(inputBytes, valid); var fixtures = Fixtures(pool, ubatch);
                Directory.CreateDirectory(output); owned = true; captures = new(output); state["captures"] = captures.Records;
                File.WriteAllBytes(Path.Combine(output, "source-inputs.json"), inputBytes);
                Save(Path.Combine(output, "derived-inputs.json"), new { schema = "ninfer-decode-graph-fixed-inputs/1", originalInputSha256 = inputSha,
                    sourceCase = pool.Id, construction = "Cyclic original prompt token stream at declared case/lane offsets; original continuation token stream; all complete derived arrays retained", fixtures });
                state["derivedInputs"] = FilePin(Path.Combine(output, "derived-inputs.json")); state["sourceInputs"] = FilePin(Path.Combine(output, "source-inputs.json"));
                state["artifactBefore"] = before; state["artifactDirectorySha256"] = directorySha; state["fullArtifactSha256Recomputed"] = false;
                state["frontendResourceHashes"] = frontend.Resources.ToDictionary(p => p.Key, p => p.Value.Sha256);
                state["nativeFiles"] = nativePins; state["assemblies"] = assemblyPins; state["context"] = context; state["ubatch"] = ubatch;
                state["chosenNonGraphEnvironment"] = initialEnvironment.Where(p => p.Key != GraphEnvironment).ToDictionary(p => p.Key, p => p.Value);
                state["maximumRawCaptureBytes"] = CaptureBudget; state["scope"] = "Fixed raw text token inputs, no template/sampler/drafter/vision/features/count-bearing graph; same chosen A16/repack modules in both lifetimes. Full physical-head F32 exact comparison, not logits similarity.";
                state["guardScope"] = "Public DecodeBatch/Forward caller arrays have tail guards and NaN active-output poison; the API accepts no output offset, so no prefix guard is claimed. PrefillBatch allocates its own outputs; every physical-head element is checked.";
                state["limitations"] = new[] { "No task-quality or independent original-engine oracle is provided by graph OFF/ON equality.",
                    "Engine wall timers include full head copy; OFF then ON are separate lifetimes, not ABBA/E2E performance evidence.",
                    "Artifact directory/frontend/stat are checked; external original whole-file SHA/inode guard is required.",
                    "Only recorded lane/position/stress/clone/reset cases are qualified; unsupported shapes and capacity failures are never PASS.",
                    "Use a new immutable runtime and local POSIX ignored evidence path; retain failed captures. No source, default, or module arithmetic changes are made by the tool." };
                Persist(); string? arithmetic = null;
                foreach (bool graphs in new[] { false, true })
                {
                    if (!Equals(state["GPU"], "EXECUTED")) state["GPU"] = "ATTEMPTED";
                    state["activeMode"] = graphs ? "on" : "off"; Persist();
                    modes.Add(RunMode(graphs, artifact, context, ubatch, valid, fixtures, captures, stable, nativePins, assemblyPins, ref arithmetic, Persist,
                        () => { state["GPU"] = "EXECUTED"; Persist(); }));
                    Need(JsonSerializer.Serialize(before) == JsonSerializer.Serialize(ArtifactStat(artifact)), "Artifact stat changed between modes.");
                    using var afterReader = new NinferArtifactReader(artifact); Need(afterReader.DirectorySha256 == directorySha, "Artifact directory changed."); Persist();
                }
                Need(captures.Records.Count == 338 && captures.OnComparisons == 169 && captures.RepeatComparisons == 156 && captures.CloneComparisons == 52,
                    "Incomplete fixed capture/paired/reset/clone coverage.");
                Need(ValidationSha256.Matches(FilePin(inputsPath).Sha256, inputSha), "Original input manifest changed.");
                VerifyPins(new[] { (Pin)state["derivedInputs"]!, (Pin)state["sourceInputs"]! }); captures.VerifyStoredVectors();
                state["storedRawCaptureBytes"] = captures.StoredBytes; state["physicalHeadScalarValuesChecked"] = captures.ScalarValuesChecked;
                state["offOnExactComparisons"] = captures.OnComparisons; state["sameModeResetRepeatComparisons"] = captures.RepeatComparisons;
                state["cloneExactComparisons"] = captures.CloneComparisons; state["artifactAfter"] = ArtifactStat(artifact);
                state["complete"] = true; state["passed"] = true; state.Remove("activeMode"); FinalizeManifest();
                SaveCompleted(output, state);
                Console.WriteLine(JsonSerializer.Serialize(new { complete = true, captures = captures.Records.Count, captures.OnComparisons,
                    captures.RepeatComparisons, captures.CloneComparisons, storedRawBytes = captures.StoredBytes,
                    modelQuality = "NOT_RUN", independentModelOracle = "NOT_RUN", endToEndPerformance = "NOT_QUALIFIED" })); return 0;
            }
        }
        catch (Exception error)
        {
            state["passed"] = false; state["error"] = error.ToString(); if (captures != null) state["storedRawCaptureBytes"] = captures.StoredBytes;
            FinalizeManifest(); Console.Error.WriteLine(error); return 1;
        }
        finally { Environment.SetEnvironmentVariable(GraphEnvironment, originalGraph); }
    }
}
