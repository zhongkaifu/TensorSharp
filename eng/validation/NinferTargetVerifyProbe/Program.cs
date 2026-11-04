// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Validation;

// Ordinary count-bearing prefill is the independent state control. Never compare
// a T-window W8 head with a one-row head or shorten the projection shape to kept.
internal static class Program
{
    private static readonly int[] Windows = { 1, 2, 3, 4, 7, 8 }, LaneCounts = { 1, 2, 4 }, Bases = { 127, 255, 1023 };
    private const int Guard = 64, GuardBits = 0x4e57a193, IntGuard = 0x6327194e;
    private const long Budget = 2048L << 20;
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
    private sealed record Seed(string Id, int[][] Prompts, int[] NextTokens, int Ordinal);
    private sealed record Pin(string Path, long Bytes, string Sha256);
    private static void Need(bool value, string message) { if (!value) throw new InvalidDataException(message); }
    private static string Hash(ReadOnlySpan<byte> bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    private static Pin FilePin(string path) { using var stream = File.OpenRead(path); return new(Path.GetFullPath(path), stream.Length, Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant()); }
    private static void Save(string path, object value) { using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None); JsonSerializer.Serialize(stream, value, Json); stream.WriteByte(10); }
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
    private static bool Link(string path) => new FileInfo(path).LinkTarget != null || new DirectoryInfo(path).LinkTarget != null;
    private static void Fresh(string path)
    {
        Need(Path.IsPathFullyQualified(path) && path == Path.GetFullPath(path), "Canonical absolute output required.");
        string[] parts = path.Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries);
        Need(parts.Contains("artifacts", StringComparer.Ordinal) || Enumerable.Range(0, parts.Length - 1).Any(i => parts[i] == "docs" && parts[i + 1] == "validation"), "Ignored evidence path required.");
        Need(!Path.Exists(path) && !Link(path), "Fresh output required.");
        for (string? p = Path.GetDirectoryName(path); p != null; p = Path.GetDirectoryName(p)) Need(!Link(p) && !File.Exists(p), "Aliased output parent refused.");
    }
    private static Dictionary<string, string> Options(string[] args)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i++)
        {
            string key = args[i]; Need(key is "--execute" or "--self-test" or "--artifact" or "--expect-directory-sha" or "--inputs" or "--expect-inputs-sha" or "--out" or "--context" or "--ubatch", "Unknown option.");
            Need(!result.ContainsKey(key), "Duplicate option.");
            if (key is "--execute" or "--self-test") result.Add(key, "1");
            else { Need(++i < args.Length, "Missing option value."); result.Add(key, args[i]); }
        }
        Need(!(result.ContainsKey("--execute") && result.ContainsKey("--self-test")), "Exclusive modes required."); return result;
    }
    private static string Required(Dictionary<string, string> options, string key) => options.TryGetValue(key, out var value) ? value : throw new ArgumentException("Missing " + key);
    private static int Positive(Dictionary<string, string> options, string key, int fallback) { if (!options.TryGetValue(key, out var raw)) return fallback; Need(int.TryParse(raw, out int n) && n > 0, "Positive integer required."); return n; }
    private static object Stat(string path) { var f = new FileInfo(path); Need(f.Exists && !Link(path), "Original regular artifact required."); return new { f.FullName, f.Length, f.LastWriteTimeUtc, f.CreationTimeUtc }; }
    private static void Unique(JsonElement element)
    {
        if (element.ValueKind == JsonValueKind.Object) { var seen = new HashSet<string>(StringComparer.Ordinal); foreach (var p in element.EnumerateObject()) { Need(seen.Add(p.Name), "Duplicate input field."); Unique(p.Value); } }
        else if (element.ValueKind == JsonValueKind.Array) foreach (var row in element.EnumerateArray()) Unique(row);
    }
    private static (int[] Prompt, int[] Continuation) Inputs(byte[] bytes, int valid)
    {
        using var doc = JsonDocument.Parse(bytes); var root = doc.RootElement; Unique(root);
        Need(root.GetProperty("schema").GetString() == "ninfer-qwen38-body-inputs/1", "Pinned original input schema required.");
        var pools = new List<(int[] Prompt, int[] Continuation)>();
        foreach (var row in root.GetProperty("cases").EnumerateArray())
        {
            int[] p = row.GetProperty("promptTokens").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            int[] c = row.GetProperty("continuationTokens").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            Need(p.Length is > 0 and <= 16384 && c.Length is >= 8 and <= 128 && p.Concat(c).All(t => t >= 0 && t < valid), "Bounded valid original token arrays required."); pools.Add((p, c));
        }
        Need(pools.Count is > 0 and <= 128, "Bounded source cases required."); var longest = pools.OrderByDescending(x => x.Prompt.Length).First(); Need(longest.Prompt.Length >= 4096, "Original4096-token pool required."); return longest;
    }
    private static Seed[] Seeds(int[] pool)
    {
        var seeds = new List<Seed>();
        foreach (int lanes in LaneCounts) for (int rotation = 0; rotation < Bases.Length; rotation++)
        {
            int ordinal = seeds.Count;
            int[][] prompts = Enumerable.Range(0, lanes).Select(i => Enumerable.Range(0, Bases[(rotation + i) % Bases.Length] + (i / 3))
                .Select(j => pool[(j + ordinal * 13 + i * 29) % pool.Length]).ToArray()).ToArray();
            int[] next = Enumerable.Range(0, lanes).Select(i => pool[(321 + ordinal * 17 + i * 53) % pool.Length]).ToArray();
            seeds.Add(new($"lanes{lanes}-positions{rotation}", prompts, next, ordinal));
        }
        return seeds.ToArray();
    }
    private static int[] Tokens(Seed seed, int window, int[] continuation) => Enumerable.Range(0, seed.Prompts.Length)
        .SelectMany(i => Enumerable.Range(0, window).Select(j => continuation[(seed.Ordinal + i * 3 + j) % continuation.Length])).ToArray();
    private static int[][] Counts(int lanes, int window)
    {
        var result = new List<int[]>(); var seen = new HashSet<string>(StringComparer.Ordinal);
        void Add(int[] values) { if (seen.Add(string.Join(',', values))) result.Add(values); }
        for (int k = 0; k <= window; k++) Add(Enumerable.Repeat(k, lanes).ToArray());
        if (lanes > 1)
        {
            Add(Enumerable.Range(0, lanes).Select(i => i % 2 == 0 ? 0 : window).ToArray());
            Add(Enumerable.Range(0, lanes).Select(i => i % 2 == 0 ? window : 0).ToArray());
            for (int k = 1; k < window; k++) Add(Enumerable.Range(0, lanes).Select(i => (i % 4) switch { 0 => k, 1 => window - k, 2 => 0, _ => window }).ToArray());
        }
        return result.ToArray();
    }
    private static float[] Poison(int count, int pass) { var values = Enumerable.Repeat(BitConverter.Int32BitsToSingle(pass == 0 ? unchecked((int)0x7fc1f17e) : unchecked((int)0x7fc2f17e)), count + Guard).ToArray(); for (int i = count; i < values.Length; i++) values[i] = BitConverter.Int32BitsToSingle(GuardBits); return values; }
    private static bool Guarded(float[] values, int count) => values.AsSpan(count).ToArray().All(x => BitConverter.SingleToInt32Bits(x) == GuardBits);
    private static void Head(float[] logits, int rows, int vocab, int valid, int[]? argmax = null)
    {
        Need(logits.Length >= rows * vocab && Guarded(logits, rows * vocab), "Caller head buffer guard changed.");
        for (int r = 0; r < rows; r++)
        {
            int best = 0; for (int c = 0; c < vocab; c++) { Need(float.IsFinite(logits[r * vocab + c]), "Nonfinite physical-head logit."); if (logits[r * vocab + c] > logits[r * vocab + best]) best = c; }
            Need(best < valid, "Padded vocabulary row wins."); if (argmax != null) Need(argmax[r] == best, "Returned argmax differs from full logits.");
        }
        if (argmax != null) Need(argmax.AsSpan(rows).ToArray().All(x => x == IntGuard), "Caller argmax tail guard changed.");
    }
    private sealed class Store
    {
        private readonly string _output; private readonly Dictionary<string, string> _first = new(StringComparer.Ordinal); private readonly HashSet<string> _files = new(StringComparer.Ordinal);
        internal long StoredBytes, ScalarsChecked; internal int Pairs, Repeats;
        internal readonly List<object> Rows = new();
        internal Store(string output) { _output = output; Directory.CreateDirectory(Path.Combine(output, "vectors")); }
        private string Put(ReadOnlySpan<byte> bytes)
        {
            string digest = Hash(bytes), relative = Path.Combine("vectors", digest + ".f32"), path = Path.Combine(_output, relative);
            if (_files.Add(digest))
            {
                Need(StoredBytes + bytes.Length <= Budget, "Full-output evidence budget exceeded; no passing result.");
                using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None); stream.Write(bytes); StoredBytes += bytes.Length;
            }
            else Need(FilePin(path).Sha256 == digest && new FileInfo(path).Length == bytes.Length, "Retained vector changed.");
            return relative;
        }
        internal void Pair(string id, int pass, string kind, float[] candidate, float[] reference, int rows, int vocab, int valid)
        {
            for (int r = 0; r < rows; r++)
            {
                var a = MemoryMarshal.AsBytes(candidate.AsSpan(r * vocab, vocab)); var b = MemoryMarshal.AsBytes(reference.AsSpan(r * vocab, vocab));
                string af = Put(a), bf = Put(b), ah = Hash(a), bh = Hash(b); bool same = a.SequenceEqual(b), repeated = true;
                string key = id + '/' + kind + '/' + r;
                if (pass == 0) Need(_first.TryAdd(key, ah), "Duplicate first-pass pair identity.");
                else { Need(_first.TryGetValue(key, out string? first), "Missing reset-repeat baseline."); repeated = ValidationSha256.Matches(first!, ah); Repeats++; }
                Rows.Add(new { id, pass, kind, row = r, candidateFile = af, referenceFile = bf, candidateSha256 = ah, referenceSha256 = bh,
                    fullPhysicalHeadBytesEqual = same, resetRepeatEqual = pass == 1 ? repeated : (bool?)null, elements = vocab });
                Need(same && repeated, "Complete subsequent Decode1 logits differ; both outputs retained."); Pairs++; ScalarsChecked += vocab * 2L;
            }
            Head(candidate, rows, vocab, valid); Head(reference, rows, vocab, valid);
        }
        internal void FailureVector(string name, float[] values, int elements)
        {
            string file = Put(MemoryMarshal.AsBytes(values.AsSpan(0, elements))); Rows.Add(new { failure = name, file, elements });
        }
        internal void VerifyFiles() { foreach (string hash in _files) Need(FilePin(Path.Combine(_output, "vectors", hash + ".f32")).Sha256 == hash, "A retained vector changed before finalization."); }
    }
    private sealed class GraphCoverage
    {
        internal readonly bool Enabled;
        internal readonly List<object> Operations = new();
        internal int EligibleCalls, OrdinaryCountedCalls;
        internal GraphCoverage(bool enabled) { Enabled = enabled; }
        internal static (long Capture, long Replay, long Eviction) Mark(Q35CudaEngine engine)
            => (engine.NinferDecodeGraphCaptureCount, engine.NinferDecodeGraphReplayCount, engine.NinferDecodeGraphEvictionCount);
        internal void Check(Q35CudaEngine engine, (long Capture, long Replay, long Eviction) before, bool eligible, string operation)
        {
            var after = Mark(engine); long captures = after.Capture - before.Capture, replays = after.Replay - before.Replay;
            Need(replays == (Enabled && eligible ? 1 : 0), "Required graph replay coverage differs for " + operation);
            Need(captures >= 0 && captures <= (Enabled && eligible ? 1 : 0), "Unexpected graph capture for " + operation);
            if (eligible) EligibleCalls++; else OrdinaryCountedCalls++;
            Operations.Add(new { operation, eligible, graphEnabled = Enabled, captures, replays, evictions = after.Eviction - before.Eviction });
        }
        internal object Final(Q35CudaEngine engine)
        {
            var totals = Mark(engine);
            Need(Enabled ? totals.Capture > 0 && totals.Replay > totals.Capture : totals == (0L, 0L, 0L), "Missing actual graph capture/later replay or graph-disabled path used graphs.");
            return new { enabled = Enabled, eligibleCalls = EligibleCalls, ordinaryCountedCalls = OrdinaryCountedCalls,
                captureCount = totals.Capture, replayCount = totals.Replay, evictionCount = totals.Eviction,
                cachedGraphsBeforeDisposal = engine.CapturedNinferDecodeGraphCount };
        }
    }
    private static float[] Decode(Q35CudaEngine engine, Q35CudaEngine.Slot[] slots, int[] tokens, int pass, int valid, Store store, GraphCoverage graphs, string label)
    {
        int[] input = tokens.ToArray(), before = slots.Select(x => x.NPast).ToArray(), drawn = Enumerable.Repeat(IntGuard, slots.Length + 16).ToArray();
        float[] logits = Poison(slots.Length * engine.VocabSize, pass); var graphBefore = GraphCoverage.Mark(engine);
        engine.DecodeBatch(slots, tokens, logits, drawn); graphs.Check(engine, graphBefore, true, label + "/DecodeBatch1");
        try
        {
            Need(tokens.AsSpan().SequenceEqual(input) && slots.Select((s, i) => s.NPast == before[i] + 1).All(x => x), "Decode1 input/position changed unexpectedly.");
            Head(logits, slots.Length, engine.VocabSize, valid, drawn);
        }
        catch { store.FailureVector(label, logits, slots.Length * engine.VocabSize); throw; }
        return logits;
    }
    private static Q35CudaEngine.Slot[] Clone(Q35CudaEngine engine, Q35CudaEngine.Slot[] source)
    {
        var result = new List<Q35CudaEngine.Slot>();
        try { foreach (var slot in source) { var clone = engine.CreateSlot(0); result.Add(clone); engine.CopySlot(slot, clone); Need(clone.NPast == slot.NPast && clone.RopeDelta == slot.RopeDelta, "Snapshot clone metadata differs."); } return result.ToArray(); }
        catch { foreach (var slot in result) engine.FreeSlot(slot); throw; }
    }
    private static void Release(Q35CudaEngine engine, Q35CudaEngine.Slot[] slots, Q35CudaEngine.Slot restore) { engine.Active = restore; foreach (var slot in slots) engine.FreeSlot(slot); }
    private static void SeedPrefill(Q35CudaEngine engine, Q35CudaEngine.Slot[] slots, Seed seed)
    {
        foreach (var slot in slots) engine.ResetSlot(slot);
        for (int start = 0; start < seed.Prompts.Max(x => x.Length); start += engine.UBatch)
        {
            int at = start; int[] indices = Enumerable.Range(0, slots.Length).Where(i => seed.Prompts[i].Length > at).ToArray();
            int[][] chunks = indices.Select(i => seed.Prompts[i].AsSpan(at, Math.Min(engine.UBatch, seed.Prompts[i].Length - at)).ToArray()).ToArray();
            engine.PrefillBatch(indices.Select(i => slots[i]).ToArray(), chunks, new float[indices.Length][]);
        }
        Need(slots.Select((s, i) => s.NPast == seed.Prompts[i].Length && s.RopeDelta == 0).All(x => x), "Seed reset/prefill metadata differs.");
    }
    private static unsafe string DeviceDigest(IntPtr pointer, long bytes, byte[] buffer)
    {
        Need(pointer != IntPtr.Zero && bytes > 0, "Readable device allocation required."); using var digest = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        for (long offset = 0; offset < bytes;)
        {
            int count = (int)Math.Min(buffer.Length, bytes - offset);
            fixed (byte* dst = buffer) CudaDriverApi.cuMemcpyDtoH((IntPtr)dst, pointer + (nint)offset, (UIntPtr)count).ThrowOnError();
            digest.AppendData(buffer, 0, count); offset += count;
        }
        return Convert.ToHexString(digest.GetHashAndReset()).ToLowerInvariant();
    }
    private static object[] Snapshot(Q35CudaEngine engine, Q35CudaEngine.Slot[] slots, bool kv)
    {
        engine.Synchronize(); Need(!engine.HasDrafter, "Target-only state digest required."); var buffer = new byte[1 << 20]; var result = new List<object>();
        foreach (var slot in slots)
        {
            // Actual original target: 48 recurrent layers, DConv4, ConvDim10240,
            // 48 value heads of128x128 F32 state. Both regions are256-byte aligned.
            Need(slot.Conv.Length == 48 && slot.State.Length == 48 && engine.SlotStateBytes == 48L * (3 * 10240 * 4 + 48 * 128 * 128 * 4), "Original target recurrent layout differs.");
            string recurrent = DeviceDigest(slot.Arena, engine.SlotStateBytes, buffer); var blocks = new List<string>();
            if (kv)
            {
                Need(slot.Blocks.Count > 0 && engine.SlotKvBytes(slot) % slot.Blocks.Count == 0, "Complete KV block geometry required.");
                long bytes = engine.SlotKvBytes(slot) / slot.Blocks.Count;
                foreach (IntPtr block in slot.Blocks) blocks.Add(DeviceDigest(block, bytes, buffer));
            }
            result.Add(new { slot.NPast, slot.RopeDelta, version = kv ? slot.Version : (int?)null,
                recurrentBytes = engine.SlotStateBytes, recurrentSha256 = recurrent, kvBlockSha256 = blocks });
        }
        return result.ToArray();
    }
    private static string Encoded(object value) => JsonSerializer.Serialize(value, Json);
    private static float[] FullWindowReference(Q35CudaEngine engine, Q35CudaEngine.Slot[] seedSlots, Seed seed,
        int window, int[] continuation, int pass, int valid, Store store, GraphCoverage graphs)
    {
        var ordinary = Clone(engine, seedSlots);
        try
        {
            int rows = ordinary.Length * window; float[] head = Poison(rows * engine.VocabSize, pass);
            int[] tokens = Tokens(seed, window, continuation), unchanged = tokens.ToArray();
            int[] positions = ordinary.Select(s => s.NPast).ToArray();
            int[] counts = Enumerable.Repeat(window, ordinary.Length).ToArray();
            var graphBefore = GraphCoverage.Mark(engine);
            engine.NinferValidationPrefillPadded(ordinary, window, tokens, counts, head);
            graphs.Check(engine, graphBefore, false, seed.Id + $"/window{window}/ordinary-all-row-head");
            Need(tokens.AsSpan().SequenceEqual(unchanged) && counts.All(x => x == window)
                && ordinary.Select((s, i) => s.NPast == positions[i] + window).All(x => x), "Full-window reference changed input/counts or advanced incorrect positions.");
            try { Head(head, rows, engine.VocabSize, valid); }
            catch { store.FailureVector(seed.Id + $"/window{window}/ordinary-all-row-head", head, rows * engine.VocabSize); throw; }
            return head;
        }
        finally { Release(engine, ordinary, seedSlots[0]); }
    }
    private static void Case(Q35CudaEngine engine, Q35CudaEngine.Slot[] seedSlots, Seed seed, int window, int[] counts, int[] continuation,
        int pass, int valid, Store store, GraphCoverage graphs, List<object> records, float[] fullWindowHeadReference)
    {
        string id = seed.Id + $"/window{window}/keep" + string.Join('-', counts); int[] tokens = Tokens(seed, window, continuation);
        var candidate = Clone(engine, seedSlots); var control = Clone(engine, seedSlots); Q35CudaEngine.Slot[]? extra = null;
        var record = new Dictionary<string, object?> { ["id"] = id, ["pass"] = pass, ["window"] = window, ["lanes"] = seedSlots.Length,
            ["kept"] = counts.ToArray(), ["beforePositions"] = seedSlots.Select(x => x.NPast).ToArray(), ["passed"] = false };
        records.Add(record);
        try
        {
            bool zero = counts.All(x => x == 0), full = counts.All(x => x == window);
            object[]? zeroBefore = zero ? Snapshot(engine, candidate, false) : null;
            int[] positions = candidate.Select(x => x.NPast).ToArray(); int n = candidate.Length * window;
            int[] drawn = Enumerable.Repeat(IntGuard, n + 16).ToArray(), unchanged = tokens.ToArray(); float[] verifyHead = Poison(n * engine.VocabSize, pass);
            var timer = Stopwatch.StartNew(); var graphBeforeVerify = GraphCoverage.Mark(engine);
            engine.Verify(candidate, window, tokens, drawn, verifyHead);
            graphs.Check(engine, graphBeforeVerify, true, id + "/Verify");
            record["verifyWallMillisecondsDiagnosticOnly"] = timer.Elapsed.TotalMilliseconds;
            Need(candidate.Select((s, i) => s.NPast == positions[i]).All(x => x), "Verify advanced committed positions.");
            try { Head(verifyHead, n, engine.VocabSize, valid, drawn); }
            catch { store.FailureVector(id + "/verify-head", verifyHead, n * engine.VocabSize); throw; }
            record["verifyFullHeadSha256"] = Hash(MemoryMarshal.AsBytes(verifyHead.AsSpan(0, n * engine.VocabSize)));
            record["verifyHeadPhysicalRowsChecked"] = n; record["verifyDrawn"] = drawn.Take(n).ToArray();
            Need(tokens.AsSpan().SequenceEqual(unchanged), "Verify changed input token array.");
            // CPU-only byte comparison/file capture cannot invalidate the pending
            // Verify checkpoint. The reference head has exactly this n-row shape.
            store.Pair(id, pass, "verify-window-same-shape-all-row-head", verifyHead, fullWindowHeadReference,
                n, engine.VocabSize, valid);
            record["verifyWindowAllRowHeadExact"] = true;
            int[] kept = counts.ToArray(); engine.Commit(candidate, window, kept);
            Need(kept.AsSpan().SequenceEqual(counts) && candidate.Select((s, i) => s.NPast == positions[i] + counts[i]).All(x => x), "Commit input or positions differ.");
            bool rejected = false; try { engine.Commit(candidate, window, kept); } catch (InvalidOperationException) { rejected = true; }
            Need(rejected, "Consumed checkpoint accepted a duplicate commit."); record["duplicateCommitRejected"] = true;
            if (zero)
            {
                object[] zeroAfter = Snapshot(engine, candidate, false); record["zeroRecurrentBefore"] = zeroBefore; record["zeroRecurrentAfter"] = zeroAfter;
                Need(Encoded(zeroBefore!) == Encoded(zeroAfter), "All-zero commit changed recurrent or convolution bytes.");
            }
            // This ordinary path preserves all physical rows and original future
            // tokens while advancing only counts. It never calls Verify/Commit.
            int[] controlTokens = tokens.ToArray(), controlCounts = counts.ToArray();
            var graphBeforeOrdinary = GraphCoverage.Mark(engine);
            engine.NinferValidationPrefillPadded(control, window, controlTokens, controlCounts);
            graphs.Check(engine, graphBeforeOrdinary, false, id + "/ordinary-padded-count-control");
            Need(controlTokens.AsSpan().SequenceEqual(tokens) && controlCounts.AsSpan().SequenceEqual(counts)
                && control.Select((s, i) => s.NPast == positions[i] + counts[i]).All(x => x), "Ordinary padded reference input/positions differ.");
            float[] a = Decode(engine, candidate, seed.NextTokens, pass, valid, store, graphs, id + "/candidate-next"),
                b = Decode(engine, control, seed.NextTokens, pass, valid, store, graphs, id + "/padded-control-next");
            store.Pair(id, pass, "padded-count-control-next-decode1", a, b, candidate.Length, engine.VocabSize, valid);
            if (zero || full)
            {
                extra = Clone(engine, seedSlots);
                if (full)
                {
                    if (extra.Length == 1) { engine.Active = extra[0]; engine.Forward(tokens, null); }
                    else engine.PrefillBatch(extra, Enumerable.Range(0, extra.Length).Select(i => tokens.AsSpan(i * window, window).ToArray()).ToArray(), new float[extra.Length][]);
                }
                float[] independent = Decode(engine, extra, seed.NextTokens, pass, valid, store, graphs, id + "/ordinary-control-next");
                store.Pair(id, pass, zero ? "no-operation-control-next-decode1" : "ordinary-full-window-control-next-decode1", a, independent, candidate.Length, engine.VocabSize, valid);
            }
            record["subsequentDecodePhysicalRows"] = candidate.Length; record["afterDecodePositions"] = candidate.Select(x => x.NPast).ToArray();
            record["passed"] = true;
        }
        finally
        {
            if (extra != null) Release(engine, extra, seedSlots[0]); Release(engine, control, seedSlots[0]); Release(engine, candidate, seedSlots[0]);
        }
    }
    private static Dictionary<string, string?> EnvironmentSnapshot() => Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>()
        .Where(x => ((string)x.Key).StartsWith("TS_Q35_", StringComparison.Ordinal) || (string)x.Key is "CUDA_VISIBLE_DEVICES" or "CUDA_CACHE_DISABLE" or "MAX_CONTEXT")
        .OrderBy(x => (string)x.Key, StringComparer.Ordinal).ToDictionary(x => (string)x.Key, x => (string?)x.Value, StringComparer.Ordinal);
    private static Pin[] NativePins(Dictionary<string, string?> environment)
    {
        var files = new HashSet<string>(StringComparer.Ordinal);
        foreach (var p in environment.Where(p => p.Key.EndsWith("_PTX", StringComparison.Ordinal) && p.Value != null))
        {
            Need(Path.IsPathFullyQualified(p.Value!) && File.Exists(p.Value), "Configured native file unavailable."); files.Add(p.Value!);
            if (p.Value!.EndsWith(".ptx", StringComparison.OrdinalIgnoreCase)) foreach (string sibling in new[] { Path.ChangeExtension(p.Value, ".fatbin"), Path.ChangeExtension(p.Value, ".fatbin") + ".sha256" }) if (File.Exists(sibling)) files.Add(sibling);
        }
        foreach (string directory in new[] { AppContext.BaseDirectory, Path.Combine(AppContext.BaseDirectory, "cuda_kernels") })
            if (Directory.Exists(directory)) foreach (string file in Directory.EnumerateFiles(directory, "tensorsharp_q35*").Concat(Directory.EnumerateFiles(directory, "tensorsharp_w8*")))
                if (file.EndsWith(".ptx", StringComparison.OrdinalIgnoreCase) || file.EndsWith(".fatbin", StringComparison.OrdinalIgnoreCase) || file.EndsWith(".sha256", StringComparison.OrdinalIgnoreCase)) files.Add(file);
        return files.Order(StringComparer.Ordinal).Select(FilePin).ToArray();
    }
    private static void VerifyPins(IEnumerable<Pin> pins) { foreach (var pin in pins) Need(FilePin(pin.Path) == pin, "Pinned source/runtime/input/native file changed."); }
    private static object SelfTest()
    {
        var seeds = Seeds(Enumerable.Range(0, 4096).Select(x => x % 257).ToArray()); int checks = 0;
        Need(seeds.Length == 9 && seeds.Select(s => s.Prompts.Length).Distinct().SequenceEqual(LaneCounts), "Seed lane coverage."); checks++;
        int cases = 0; long rows = 0;
        foreach (int lanes in LaneCounts) foreach (int w in Windows)
        {
            var plans = Counts(lanes, w); Need(plans.All(x => x.Length == lanes && x.All(k => k >= 0 && k <= w)), "Invalid kept plan."); checks++;
            Need(Enumerable.Range(0, w + 1).All(k => plans.Any(x => x.All(c => c == k))), "Missing keep0/full/partial."); checks++;
            Need(plans.Select(x => string.Join(',', x)).Distinct().Count() == plans.Length, "Duplicate kept vector."); checks++;
            cases += plans.Length * 3; rows += (long)plans.Length * lanes * 3;
        }
        var poison = Poison(8, 0); Need(Guarded(poison, 8), "Guard construction."); checks++; poison[^1] = 0; Need(!Guarded(poison, 8), "Guard corruption detection."); checks++;
        Need(!MemoryMarshal.AsBytes(new[] { 0f }.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(new[] { BitConverter.Int32BitsToSingle(unchecked((int)0x80000000)) }.AsSpan())), "Signed zero must differ under exact byte gate."); checks++;
        long referenceRows = seeds.Sum(s => (long)s.Prompts.Length * Windows.Sum());
        Need(Windows.Contains(8) && Enumerable.Range(0, 9).All(k => Counts(1, 8).Any(x => x[0] == k)), "Full DFlash seven-draft-plus-anchor coverage missing."); checks++;
        Need(rows + referenceRows == 1716 && (rows + referenceRows) * 248320 * 4 < Budget, "Complete raw head capture plan exceeds its declared bound."); checks++;
        return new { CPU = "PASS", contracts = checks, seeds = seeds.Length, windowSizes = Windows, lanes = LaneCounts,
            uniqueCommitCases = cases, resetRepeats = 2, commitCalls = cases * 2, subsequentPrimaryRowsPerPass = rows,
            fullWindowReferenceRowsPerPass = referenceRows,
            exactRowPairsIncludingReset = 2L * seeds.Sum(s => Windows.Sum(w => (long)Counts(s.Prompts.Length, w).Length * s.Prompts.Length * (w + 1) + 2L * s.Prompts.Length)),
            potentialUniqueRawRows = rows + referenceRows, potentialSuccessfulRawBytes = (rows + referenceRows) * 248320 * 4,
            GPU = "NOT_RUN", verificationCommit = "NOT_RUN", modelQuality = "NOT_RUN", performance = "NOT_RUN" };
    }
    public static int Main(string[] args)
    {
        string? output = null; bool owned = false; Store? store = null; var records = new List<object>(); var campaigns = new List<object>();
        var state = new Dictionary<string, object?> { ["schema"] = "ninfer-target-verify-independent-state-probe/1", ["complete"] = false,
            ["GPU"] = "NOT_RUN", ["modelQuality"] = "NOT_RUN", ["independentEngineOracle"] = "NOT_RUN", ["endToEndPerformance"] = "NOT_QUALIFIED", ["cases"] = records, ["campaigns"] = campaigns };
        void Persist() { if (owned && output != null) Save(Path.Combine(output, "result.json"), state); }
        void Finish() { Persist(); if (owned && output != null) File.WriteAllText(Path.Combine(output, "result.json.sha256"), FilePin(Path.Combine(output, "result.json")).Sha256 + "  result.json\n"); }
        try
        {
            Need(BitConverter.IsLittleEndian, "Little-endian raw F32 captures required."); var options = Options(args);
            if (options.ContainsKey("--self-test")) { Console.WriteLine(JsonSerializer.Serialize(SelfTest(), Json)); return 0; }
            if (!options.ContainsKey("--execute")) { Console.WriteLine("PLAN: set TS_Q35_NINFER_VERIFY=1, then --execute --artifact FILE --expect-directory-sha SHA --inputs ORIGINAL_BODY_INPUTS --expect-inputs-sha SHA --out FRESH_LOCAL_IGNORED_DIRECTORY [--context8192] [--ubatch512]. Windows1/2/3/4/7/8, lanes1/2/4, positions127/255/1023; everykeep0..full andmixedcounts, resetrepeat, clone/COW, exact subsequentDecode1 under independent same-physical-window ordinary control. --self-test performs no model/CUDA work."); return 0; }
            int context = Positive(options, "--context", 8192), ubatch = Positive(options, "--ubatch", 512);
            Need(context is >= 2048 and <= 262144 && ubatch is >= 128 and <= 2048, "Bounded context/ubatch required.");
            string artifact = Path.GetFullPath(Required(options, "--artifact")), inputsPath = Path.GetFullPath(Required(options, "--inputs"));
            string directorySha = ValidationSha256.Normalize(Required(options, "--expect-directory-sha")), inputSha = ValidationSha256.Normalize(Required(options, "--expect-inputs-sha"));
            output = Required(options, "--out"); Fresh(output); var before = Stat(artifact);
            Need(!Link(inputsPath), "Regular original input file required."); byte[] original = File.ReadAllBytes(inputsPath); Need(Hash(original) == inputSha, "Original input SHA differs.");
            var environment = EnvironmentSnapshot(); Need(Environment.GetEnvironmentVariable("TS_Q35_NINFER_VERIFY") == "1", "Explicit candidate TS_Q35_NINFER_VERIFY=1 required.");
            Need(!environment.ContainsKey("TS_Q35_NINFER_GDN_CHUNK_PTX"), "Unqualified chunk option must remain outside this check.");
            string? graphOption = Environment.GetEnvironmentVariable("TS_Q35_NINFER_DECODE_GRAPHS");
            Need(graphOption is null or "0" or "1", "Graph option must be absent, 0 or 1."); var graphs = new GraphCoverage(graphOption == "1");
            var native = NativePins(environment); var assemblies = new[] { typeof(Program).Assembly, typeof(Qwen35CudaExecutor).Assembly, typeof(Q35CudaEngine).Assembly, typeof(NinferArtifactReader).Assembly }.Select(a => FilePin(a.Location)).ToArray();
            using var reader = new NinferArtifactReader(artifact); Need(reader.DirectorySha256 == directorySha, "Artifact directory differs.");
            var frontend = NinferQwen38Frontend.FromReader(reader); int valid = frontend.Tokenizer.VocabSize; Need(valid == 248077 && frontend.Tokenizer.HeadVocabSize == 248320, "Original frontend geometry required.");
            var pool = Inputs(original, valid); var seeds = Seeds(pool.Prompt);
            Directory.CreateDirectory(output); owned = true; store = new(output); state["outputPairs"] = store.Rows;
            File.WriteAllBytes(Path.Combine(output, "source-inputs.json"), original);
            Save(Path.Combine(output, "derived-inputs.json"), new { schema = "ninfer-target-verify-fixed-inputs/1", originalInputSha256 = inputSha, seeds,
                scenarios = seeds.Select(s => new { s.Id, windows = Windows.Select(w => new { window = w, fullTokens = Tokens(s, w, pool.Continuation), keptVectors = Counts(s.Prompts.Length, w) }) }) });
            var inputPins = new[] { FilePin(inputsPath), FilePin(Path.Combine(output, "source-inputs.json")), FilePin(Path.Combine(output, "derived-inputs.json")) };
            state["graphOperations"] = graphs.Operations; state["graphCoverageRequested"] = graphs.Enabled;
            state["inputs"] = inputPins; state["nativeFiles"] = native; state["assemblies"] = assemblies; state["artifactBefore"] = before;
            state["artifactDirectorySha256"] = directorySha; state["fullArtifactSha256Recomputed"] = false; state["frontendResourceHashes"] = frontend.Resources.ToDictionary(x => x.Key, x => x.Value.Sha256);
            state["environment"] = environment; state["context"] = context; state["ubatch"] = ubatch; state["rawOutputBudgetBytes"] = Budget;
            state["control"] = "Public Verify/Commit versus internal probe-only ordinary padded-count RunRows(verify:false), preserving full physical window/projection/attention shape; every Verify head matches an independent ordinary all-row head at identical nRows, and committed state is compared through subsequent DecodeBatch1 at identical lanes. No-op and public Forward/PrefillBatch additionally cover keep0/full.";
            state["diagnosticReads"] = "Read-only bounded cuMemcpyDtoH SHA of original recurrent arena and seed KV blocks; no native/device mutation. Public methods perform all model/state evolution; only independent padded-count control is a named internal friend seam.";
            state["limitations"] = new[] { "State/clone/head exactness for supplied tokens is not an independent engine oracle or broad model/sampling/task quality.", "Every Verify-window head is byte-compared with an independent ordinary all-row head at identical nRows; no comparison to different one-row W8 arithmetic is used. This does not prove equality of bulk versus one-row sampling distributions.", "No skipped/unsupported/failed keep shape can pass. Whole original artifact bytes and physical device identity need external operator provenance.", "No DFlash proposer, acceptance, API serving or end-to-end speed is qualified; timers are diagnostic only." };
            Persist(); state["GPU"] = "ATTEMPTED";
            using var executor = new Qwen35CudaExecutor(artifact, context, ubatch); var engine = executor.Engine; state["GPU"] = "EXECUTED";
            Need(executor.Draft == null && !engine.HasDrafter && engine.FeatureWidth == 0 && engine.IsNinferModel
                && engine.VocabSize == frontend.Tokenizer.HeadVocabSize && engine.ContextSize == context && engine.UBatch == ubatch,
                "Target-only original NInfer engine and requested full-head/context/ubatch geometry required.");
            state["numericalStateNamespace"] = engine.KVStateNamespaceSuffix; Persist(); var first = engine.Active; var extras = new List<Q35CudaEngine.Slot>();
            try
            {
                var all = new Q35CudaEngine.Slot[4]; all[0] = first; for (int i = 1; i < all.Length; i++) { all[i] = engine.CreateSlot(0); extras.Add(all[i]); }
                foreach (var seed in seeds) for (int pass = 0; pass < 2; pass++)
                {
                    var slots = all.Take(seed.Prompts.Length).ToArray(); SeedPrefill(engine, slots, seed);
                    object[] checkpoint = Snapshot(engine, slots, true); int expected = Windows.Sum(w => Counts(slots.Length, w).Length), start = records.Count;
                    foreach (int window in Windows)
                    {
                        float[] fullWindowHead = FullWindowReference(engine, slots, seed, window, pool.Continuation, pass, valid, store, graphs);
                        foreach (int[] counts in Counts(slots.Length, window))
                        { Case(engine, slots, seed, window, counts, pool.Continuation, pass, valid, store, graphs, records, fullWindowHead); Persist(); }
                    }
                    object[] after = Snapshot(engine, slots, true); Need(Encoded(checkpoint) == Encoded(after), "Parent checkpoint bytes/metadata changed after clone/COW verification campaign.");
                    Need(records.Count - start == expected, "Incomplete kept-count matrix.");
                    campaigns.Add(new { seed.Id, pass, cases = expected, checkpointBefore = checkpoint, checkpointAfter = after, parentBytesUnchanged = true, resetReplay = pass == 1 }); Persist();
                }
            }
            finally { engine.Active = first; foreach (var slot in extras) engine.FreeSlot(slot); }
            state["actualGraphCoverage"] = graphs.Final(engine);
            Need(graphs.EligibleCalls == records.Count * 3 + seeds.Length * Windows.Length * 4, "Incomplete eligible Verify/Decode graph coverage matrix.");
            Need(graphs.OrdinaryCountedCalls == records.Count + seeds.Length * Windows.Length * 2, "Incomplete graph-refused ordinary-control matrix.");
            executor.Dispose(); // Finish model teardown before authoritative final evidence.
            int planned = seeds.Sum(s => Windows.Sum(w => Counts(s.Prompts.Length, w).Length)) * 2;
            Need(records.Count == planned && campaigns.Count == seeds.Length * 2, "Incomplete verification/repeat campaign.");
            long expectedPairsPerPass = seeds.Sum(s => Windows.Sum(w => (long)Counts(s.Prompts.Length, w).Length * s.Prompts.Length * (w + 1) + 2L * s.Prompts.Length));
            Need(store.Pairs == expectedPairsPerPass * 2 && store.Repeats == expectedPairsPerPass, "Incomplete all-row head/subsequent decode/reset comparison matrix.");
            VerifyPins(inputPins.Concat(native).Concat(assemblies)); store.VerifyFiles();
            Need(Encoded(environment) == Encoded(EnvironmentSnapshot()) && Encoded(before) == Encoded(Stat(artifact)), "Environment/artifact stat changed.");
            using var afterReader = new NinferArtifactReader(artifact); Need(afterReader.DirectorySha256 == directorySha, "Artifact directory changed.");
            state["complete"] = true; state["passed"] = true; state["actualCommitCasesIncludingResetRepeat"] = records.Count;
            state["exactOutputRowPairs"] = store.Pairs; state["exactResetRepeatRowPairs"] = store.Repeats; state["storedRawBytes"] = store.StoredBytes;
            state["headScalarsCompared"] = store.ScalarsChecked; state["artifactAfter"] = Stat(artifact); Finish();
            SaveCompleted(output, state);
            Console.WriteLine(JsonSerializer.Serialize(new { complete = true, commitCases = records.Count, exactOutputRowPairs = store.Pairs,
                resetRepeatRowPairs = store.Repeats, storedRawBytes = store.StoredBytes, modelQuality = "NOT_RUN", independentEngineOracle = "NOT_RUN", endToEndPerformance = "NOT_QUALIFIED" })); return 0;
        }
        catch (Exception error) { state["passed"] = false; state["error"] = error.ToString(); if (store != null) state["storedRawBytes"] = store.StoredBytes; Finish(); Console.Error.WriteLine(error); return 1; }
    }
}
