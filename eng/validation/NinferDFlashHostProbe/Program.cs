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

internal static class Program
{
    private const int Guard = 64, GuardBits = 0x4e57a193, IntGuard = 0x6327194e, K = 7, Vocab = 248320, Valid = 248077;
    private const long Budget = 3L << 30;
    private static readonly int[] LaneCounts = { 1, 2, 3, 4 }, Bases = { 127, 2047 };
    private static readonly string[] Phases = { "seed", "zero8", "mixed8", "full8", "pending1", "pending-then-decode1", "pending-then-mixed8" };
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
    private sealed record Pin(string Path, long Bytes, string Sha256);
    private sealed record Seed(string Id, int[][] Prompts, int[] Anchors, int[] NextTokens, int[] NextAnchors, int[] Window, int Ordinal);
    private sealed record Meta(int TargetPosition, int Frontier, int FirstPosition, int PendingBytes, string PendingSha256, string Namespace);
    private sealed record Proposal(float[] Head, int[][] Tokens, float[][] Confidence);
    private sealed record Outcome(Proposal Before, float[] Target, Proposal After, Meta[] BeforeMeta, Meta[] AfterMeta);
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
            string key = args[i]; Need(key is "--execute" or "--self-test" or "--artifact" or "--expect-directory-sha" or "--inputs" or "--expect-inputs-sha" or "--out" or "--context" or "--ubatch" or "--draft-module", "Unknown option.");
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
    private static Seed[] Seeds(int[] pool, int[] continuation)
    {
        var result = new List<Seed>();
        foreach (int lanes in LaneCounts) foreach (int basis in Bases)
        {
            int ordinal = result.Count;
            int[] Fixed(int at) => Enumerable.Range(0, lanes).Select(i => pool[(at + ordinal * 13 + i * 29) % pool.Length]).ToArray();
            int[][] prompts = Enumerable.Range(0, lanes).Select(i => Enumerable.Range(0, basis + i)
                .Select(j => pool[(j + ordinal * 17 + i * 31) % pool.Length]).ToArray()).ToArray();
            int[] window = Enumerable.Range(0, lanes).SelectMany(i => Enumerable.Range(0, 8).Select(j => continuation[(ordinal + i + j) % continuation.Length])).ToArray();
            result.Add(new($"lanes{lanes}-base{basis}", prompts, Fixed(111), Fixed(211), Fixed(311), window, ordinal));
        }
        return result.ToArray();
    }
    private static int[] Mixed(int lanes) => new[] { 1, 0, 3, 7 }.Take(lanes).ToArray();
    private static float[] Poison(int count, int pass)
    {
        var a = Enumerable.Repeat(BitConverter.Int32BitsToSingle(pass == 0 ? unchecked((int)0x7fc1f17e) : unchecked((int)0x7fc2f17e)), count + Guard).ToArray();
        for (int i = count; i < a.Length; i++) a[i] = BitConverter.Int32BitsToSingle(GuardBits); return a;
    }
    private static void Head(float[] values, int rows, int[]? drawn = null)
    {
        Need(values.Length == rows * Vocab + Guard && values.AsSpan(rows * Vocab).ToArray().All(x => BitConverter.SingleToInt32Bits(x) == GuardBits), "Head tail guard changed.");
        for (int r = 0; r < rows; r++)
        {
            int best = 0;
            for (int c = 0; c < Vocab; c++) { float x = values[r * Vocab + c]; Need(float.IsFinite(x), "Nonfinite physical head."); if (x > values[r * Vocab + best]) best = c; }
            Need(best < Valid, "Padded token wins the full physical head.");
            if (drawn != null) Need(drawn[r] == best, "Target argmax differs from full vector.");
        }
        if (drawn != null) Need(drawn.AsSpan(rows).ToArray().All(x => x == IntGuard), "Target argmax guard changed.");
    }
    private sealed class Store
    {
        private readonly string _out;
        private readonly HashSet<string> _files = new(StringComparer.Ordinal);
        private readonly Dictionary<string, string> _first = new(StringComparer.Ordinal);
        internal readonly List<object> Vectors = new();
        internal long Bytes; internal int ExactPairs, RepeatRows;
        internal Store(string output) { _out = output; Directory.CreateDirectory(Path.Combine(output, "vectors")); }
        private string Put(ReadOnlySpan<byte> bytes)
        {
            string h = Hash(bytes), p = Path.Combine(_out, "vectors", h + ".f32");
            if (_files.Add(h))
            {
                Need(Bytes + bytes.Length <= Budget, "Evidence budget exhausted; run cannot pass.");
                using var f = new FileStream(p, FileMode.CreateNew, FileAccess.Write, FileShare.None); f.Write(bytes); Bytes += bytes.Length;
            }
            else Need(FilePin(p).Sha256 == h && new FileInfo(p).Length == bytes.Length, "Retained vector changed.");
            return h;
        }
        internal void Record(string id, int pass, float[] a, int rows)
        {
            Head(a, rows);
            for (int r = 0; r < rows; r++)
            {
                string h = Put(MemoryMarshal.AsBytes(a.AsSpan(r * Vocab, Vocab))), key = id + '/' + r;
                if (pass == 0) Need(_first.TryAdd(key, h), "Duplicate reset reference.");
                else { Need(_first.TryGetValue(key, out var expected) && expected == h, "Reset-repeat complete head bytes differ."); RepeatRows++; }
                Vectors.Add(new { id, pass, row = r, file = "vectors/" + h + ".f32", sha256 = h, resetRepeatEqual = pass == 1 ? true : (bool?)null });
            }
        }
        internal void Pair(string id, float[] a, float[] b, int rows)
        {
            Head(a, rows); Head(b, rows);
            for (int r = 0; r < rows; r++)
            {
                var aa = MemoryMarshal.AsBytes(a.AsSpan(r * Vocab, Vocab)); var bb = MemoryMarshal.AsBytes(b.AsSpan(r * Vocab, Vocab));
                string ah = Put(aa), bh = Put(bb); bool same = aa.SequenceEqual(bb);
                Vectors.Add(new { id, row = r, candidateFile = "vectors/" + ah + ".f32", referenceFile = "vectors/" + bh + ".f32", candidateSha256 = ah, referenceSha256 = bh, exactBytes = same });
                Need(same, "Complete full-head bytes differ; both captures retained: " + id); ExactPairs++;
            }
        }
        internal void VerifyFiles() { foreach (string h in _files) Need(FilePin(Path.Combine(_out, "vectors", h + ".f32")).Sha256 == h, "Retained evidence changed before completion."); }
    }
    private static Q35CudaEngine.Slot[] Clone(Q35CudaEngine e, Q35CudaEngine.Slot[] slots)
    {
        var result = new List<Q35CudaEngine.Slot>();
        try { foreach (var s in slots) { var c = e.CreateSlot(0); result.Add(c); e.CopySlot(s, c); Need(c.NPast == s.NPast && c.RopeDelta == s.RopeDelta, "Clone metadata differs."); } return result.ToArray(); }
        catch { foreach (var s in result) e.FreeSlot(s); throw; }
    }
    private static void Release(Q35CudaEngine e, Q35CudaEngine.Slot[] slots, Q35CudaEngine.Slot restore)
    { e.Active = restore; foreach (var s in slots) e.FreeSlot(s); }
    private static Meta Metadata(Q35CudaEngine.HostSlot host)
    {
        var m = host.NinferDraft ?? throw new InvalidDataException("Native draft capsule missing."); Need(m.TargetPosition == host.NPast, "Native draft capsule missing or wrong target position.");
        Need(m.Pending.IsEmpty ? m.Frontier == host.NPast : m.Frontier == host.NPast - 1 && m.Pending.Length == 51200, "Frontier/pending invariant differs.");
        Need(m.FirstPosition >= Math.Max(0, m.Frontier - 2048) && m.FirstPosition <= m.Frontier, "Invalid retained ring frontier.");
        return new(m.TargetPosition, m.Frontier, m.FirstPosition, m.Pending.Length, Hash(m.Pending.Span), m.Namespace);
    }
    private static Meta[] Metadata(Q35CudaEngine e, Q35CudaEngine.Slot[] slots)
    {
        var result = new List<Meta>();
        foreach (var s in slots) { var h = e.CopyToHost(s); try { result.Add(Metadata(h)); } finally { e.FreeHostSlot(h); } }
        return result.ToArray();
    }
    private static string Encoded(object x) => JsonSerializer.Serialize(x, Json);
    private static unsafe string DeviceDigest(IntPtr ptr, long count, byte[] buffer)
    {
        Need(ptr != IntPtr.Zero && count > 0, "Invalid read-only device range."); using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        for (long at = 0; at < count;) { int n = (int)Math.Min(buffer.Length, count - at); fixed (byte* p = buffer) CudaDriverApi.cuMemcpyDtoH((IntPtr)p, ptr + (nint)at, (UIntPtr)n).ThrowOnError(); hash.AppendData(buffer, 0, n); at += n; }
        return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
    }
    private static object[] Digest(Q35CudaEngine e, Q35CudaEngine.Slot[] slots, bool rings, bool kv = true, bool liveKv = false)
    {
        e.Synchronize(); var result = new List<object>(); byte[] buffer = new byte[1 << 20];
        foreach (var s in slots)
        {
            Need(s.Conv.Length == 48 && s.State.Length == 48, "Original target recurrent layout required.");
            var parts = new List<string>();
            for (int i = 0; i < 48; i++) { parts.Add(DeviceDigest(s.Conv[i], 3L * 10240 * 4, buffer)); parts.Add(DeviceDigest(s.State[i], 48L * 128 * 128 * 4, buffer)); }
            if (kv) foreach (IntPtr block in s.Blocks.Take(liveKv ? (s.NPast + Q35Kernels.KvBlockTokens - 1) / Q35Kernels.KvBlockTokens : s.Blocks.Count)) parts.Add(DeviceDigest(block, e.SlotKvBytes(s) / s.Blocks.Count, buffer));
            if (rings)
            {
                Need(s.RingK.Length == 5 && s.RingV.Length == 5 && s.RingPos != IntPtr.Zero, "Original five-layer draft ring required.");
                for (int i = 0; i < 5; i++) { parts.Add(DeviceDigest(s.RingK[i], 2048L * 1024 * 2, buffer)); parts.Add(DeviceDigest(s.RingV[i], 2048L * 1024 * 2, buffer)); }
                parts.Add(DeviceDigest(s.RingPos, 2048L * 4, buffer));
            }
            result.Add(new { s.NPast, s.RopeDelta, hashes = parts });
        }
        return result.ToArray();
    }
    private static void SeedPrefill(Q35CudaEngine e, Q35CudaEngine.Slot[] slots, Seed seed)
    {
        foreach (var s in slots) e.ResetSlot(s);
        for (int start = 0; start < seed.Prompts.Max(x => x.Length); start += e.UBatch)
        {
            int at = start; int[] indices = Enumerable.Range(0, slots.Length).Where(i => seed.Prompts[i].Length > at).ToArray();
            int[][] tokens = indices.Select(i => seed.Prompts[i].AsSpan(at, Math.Min(e.UBatch, seed.Prompts[i].Length - at)).ToArray()).ToArray();
            e.PrefillBatch(indices.Select(i => slots[i]).ToArray(), tokens, new float[indices.Length][]);
        }
        Need(slots.Select((s, i) => s.NPast == seed.Prompts[i].Length).All(x => x), "Seed prefill positions differ.");
        Need(Metadata(e, slots).Select((m, i) => m.TargetPosition == slots[i].NPast && m.PendingBytes == 0).All(x => x), "Seed ring was not injected to target frontier.");
    }
    private static float[] Decode(Q35CudaEngine e, Q35CudaEngine.Slot[] slots, int[] tokens, int pass)
    {
        int[] unchanged = tokens.ToArray(), positions = slots.Select(x => x.NPast).ToArray(), drawn = Enumerable.Repeat(IntGuard, slots.Length + Guard).ToArray();
        float[] head = Poison(slots.Length * Vocab, pass); e.DecodeBatch(slots, tokens, head, drawn);
        Need(tokens.AsSpan().SequenceEqual(unchanged) && slots.Select((s, i) => s.NPast == positions[i] + 1).All(x => x), "Decode inputs/positions differ.");
        Head(head, slots.Length, drawn); return head;
    }
    private static float[] VerifyCommit(Q35CudaEngine e, Q35CudaEngine.Slot[] slots, int window, int[] tokens, int[] kept, int pass)
    {
        int rows = window * slots.Length; int[] positions = slots.Select(x => x.NPast).ToArray(), unchanged = tokens.ToArray(), k = kept.ToArray(), drawn = Enumerable.Repeat(IntGuard, rows + Guard).ToArray();
        float[] head = Poison(rows * Vocab, pass); e.Verify(slots, window, tokens, drawn, head);
        Need(slots.Select((s, i) => s.NPast == positions[i]).All(x => x) && tokens.AsSpan().SequenceEqual(unchanged), "Verify changed committed positions/input.");
        Head(head, rows, drawn); e.Commit(slots, window, k);
        Need(k.AsSpan().SequenceEqual(kept) && slots.Select((s, i) => s.NPast == positions[i] + kept[i]).All(x => x), "Commit positions/counts differ."); return head;
    }
    private static Dictionary<string, float[]> PreparePhase(Q35CudaEngine e, Q35CudaEngine.Slot[] slots, Seed seed, string phase, int pass, Store store, int[] proposalWindow, bool ordinaryAnchorControl = false)
    {
        string id = seed.Id + '/' + phase; int n = slots.Length; var transitionHeads = new Dictionary<string, float[]>(StringComparer.Ordinal);
        void Record(string label, float[] head, int rows) { transitionHeads.Add(label, head); if (!ordinaryAnchorControl) store.Record(id + '/' + label, pass, head, rows); }
        if (phase.StartsWith("pending", StringComparison.Ordinal))
        {
            float[] anchorHead = ordinaryAnchorControl ? Decode(e, slots, seed.Anchors, pass)
                : VerifyCommit(e, slots, 1, seed.Anchors, Enumerable.Repeat(1, n).ToArray(), pass);
            Record("anchor-target", anchorHead, n);
            Meta[] before = Metadata(e, slots);
            Need(before.All(m => m.PendingBytes == (ordinaryAnchorControl ? 0 : 51200)), "Window1 pending-anchor capsule not observed before any proposal/flush.");
            if (phase == "pending-then-decode1") Record("flush-target", Decode(e, slots, seed.NextTokens, pass), n);
        }
        if (phase is "zero8" or "mixed8" or "full8" or "pending-then-mixed8")
        {
            int[] counts = phase == "zero8" ? new int[n] : phase == "full8" ? Enumerable.Repeat(8, n).ToArray() : Mixed(n);
            object[]? before = phase == "zero8" ? Digest(e, slots, true, kv: false) : null; Meta[]? mb = phase == "zero8" ? Metadata(e, slots) : null;
            Record("verify8-target", VerifyCommit(e, slots, 8, phase == "pending-then-mixed8" ? seed.Window : proposalWindow, counts, pass), n * 8);
            if (before != null) Need(Encoded(before) == Encoded(Digest(e, slots, true, kv: false)) && Encoded(mb!) == Encoded(Metadata(e, slots)), "Commit0 changed target/ring/frontier bytes.");
        }
        bool pending = !ordinaryAnchorControl && phase == "pending1";
        Need(Metadata(e, slots).Select((m, q) => m.PendingBytes == (pending || !ordinaryAnchorControl && phase == "pending-then-mixed8" && Mixed(n)[q] == 0 ? 51200 : 0)).All(x => x), "Unexpected final pending-anchor phase state."); return transitionHeads;
    }
    private static Proposal Propose(Q35CudaEngine e, Q35CudaEngine.Slot[] slots, int[] anchors, int pass, bool direct = false)
    {
        int n = slots.Length; int[] unchanged = anchors.ToArray(), positions = slots.Select(x => x.NPast).ToArray();
        int[][] tokens = Enumerable.Range(0, n).Select(_ => Enumerable.Repeat(IntGuard, K + Guard).ToArray()).ToArray();
        float[][] confidence = Enumerable.Range(0, n).Select(_ => Enumerable.Repeat(BitConverter.Int32BitsToSingle(GuardBits), K + Guard).ToArray()).ToArray();
        float[] head = Poison(n * K * Vocab, pass);
        if (direct) e.ProposeNinfer(slots, anchors, tokens, confidence, head); else e.Draft(slots, anchors, tokens, confidence, head);
        Need(anchors.AsSpan().SequenceEqual(unchanged) && slots.Select((s, i) => s.NPast == positions[i]).All(x => x), "Proposal changed anchors or target positions."); Head(head, n * K);
        for (int q = 0; q < n; q++)
        {
            Need(tokens[q].Take(K).All(t => t >= 0 && t < Valid) && tokens[q].Skip(K).All(t => t == IntGuard), "Draft token invalid or tail guard changed.");
            Need(confidence[q].Take(K).All(x => BitConverter.SingleToInt32Bits(x) == 0x3f800000) && confidence[q].Skip(K).All(x => BitConverter.SingleToInt32Bits(x) == GuardBits), "Draft admission marker/guard differs; confidence1 is not a probability.");
        }
        return new(head, tokens.Select(x => x.Take(K).ToArray()).ToArray(), confidence.Select(x => x.Take(K).ToArray()).ToArray());
    }
    private static void PairProposal(Store store, string id, Proposal a, Proposal b, int lanes)
    { store.Pair(id, a.Head, b.Head, lanes * K); Need(Encoded(a.Tokens) == Encoded(b.Tokens) && Encoded(a.Confidence) == Encoded(b.Confidence), "Selector tokens/admission markers differ despite equivalent state."); }
    private static Outcome Run(Q35CudaEngine e, Q35CudaEngine.Slot[] slots, Seed seed, int pass, Store store, string id)
    {
        Meta[] before = Metadata(e, slots); object[] target = Digest(e, slots, false);
        Proposal first = Propose(e, slots, seed.Anchors, pass); Meta[] flushed = Metadata(e, slots);
        Need(flushed.All(m => m.PendingBytes == 0 && m.Frontier == m.TargetPosition), "Proposal did not flush pending anchor to current target frontier.");
        object[] afterFirst = Digest(e, slots, true);
        PairProposal(store, id + "/proposal-repeat", first, Propose(e, slots, seed.Anchors, pass), slots.Length);
        PairProposal(store, id + "/public-versus-direct-route", first, Propose(e, slots, seed.Anchors, pass, direct: true), slots.Length);
        Need(Encoded(target) == Encoded(Digest(e, slots, false)) && Encoded(afterFirst) == Encoded(Digest(e, slots, true)), "Proposal repeat mutated target/ring bytes.");
        float[] next = Decode(e, slots, seed.NextTokens, pass); Proposal after = Propose(e, slots, seed.NextAnchors, pass);
        Meta[] final = Metadata(e, slots); Need(final.All(m => m.PendingBytes == 0 && m.Frontier == m.TargetPosition), "Ordinary decode did not inject complete draft context.");
        return new(first, next, after, before, final);
    }
    private static void Pair(Store store, string id, Outcome a, Outcome b, int lanes, bool requireInitialMetadata = true)
    {
        PairProposal(store, id + "/proposal", a.Before, b.Before, lanes); store.Pair(id + "/target-next-decode1", a.Target, b.Target, lanes); PairProposal(store, id + "/proposal-after-target-decode1", a.After, b.After, lanes);
        Need((!requireInitialMetadata || Encoded(a.BeforeMeta) == Encoded(b.BeforeMeta)) && Encoded(a.AfterMeta) == Encoded(b.AfterMeta), "Restored native frontier/pending metadata differs.");
    }
    private static Q35CudaEngine.Slot[] HostHandoff(Q35CudaEngine e, Q35CudaEngine.Slot[] source, bool async, List<object> events)
    {
        var donors = Clone(e, source); var hosts = new List<Q35CudaEngine.HostSlot>(); var completions = new List<IntPtr>(); var result = new List<Q35CudaEngine.Slot>(); bool donorsFreed = false;
        try
        {
            foreach (var s in donors)
            {
                if (async) { var h = e.BeginCopyToHost(s, out IntPtr done); hosts.Add(h); completions.Add(done); }
                else hosts.Add(e.CopyToHost(s));
            }
            foreach (IntPtr done in completions) e.WaitCopy(done); completions.Clear();
            Meta[] owned = hosts.Select(Metadata).ToArray();
            // The copy event is complete before source reset/free. Host metadata and
            // ring bytes must remain owned after the original device slot is gone.
            foreach (var s in donors) e.ResetSlot(s); Release(e, donors, source[0]); donorsFreed = true;
            foreach (var h in hosts) { var dst = e.CreateSlot(0); result.Add(dst); e.CopyFromHost(h, dst); }
            Need(Encoded(owned) == Encoded(Metadata(e, result.ToArray())), "Restored host capsule differs after completed copy and source free.");
            events.Add(new { asynchronous = async, completedCopies = hosts.Count, sourcesResetAndFreedBeforeRestore = true, metadata = owned,
                kvUploadForced = false, limitation = "Live seed/source checkpoints may retain shared attention blocks; raw recurrent/ring bytes are restored, physical KV eviction is not forced." });
            return result.ToArray();
        }
        catch { foreach (IntPtr done in completions) e.WaitCopy(done); foreach (var s in result) e.FreeSlot(s); throw; }
        finally { if (!donorsFreed) Release(e, donors, source[0]); foreach (var h in hosts) e.FreeHostSlot(h); }
    }
    private static void NegativeAdmission(Q35CudaEngine e, Q35CudaEngine.Slot[] seedSlots, Seed seed, int pass, Store store, List<object> records)
    {
        var a = Clone(e, seedSlots); var b = Clone(e, seedSlots); int checks = 0;
        try
        {
            string start = Encoded(new { state = Digest(e, a, true), meta = Metadata(e, a) });
            void Unchanged() => Need(start == Encoded(new { state = Digest(e, a, true), meta = Metadata(e, a) }), "Rejected target request mutated state/positions/metadata.");
            void ArgumentRefused(Action action)
            {
                bool rejected = false; try { action(); } catch (ArgumentException) { rejected = true; }
                Need(rejected, "Malformed target geometry/self-copy was not rejected before mutation."); Unchanged(); checks++;
            }
            ArgumentRefused(() => e.CopySlot(a[0], a[0]));
            int[] drawn = Enumerable.Repeat(IntGuard, a.Length * 9 + Guard).ToArray();
            ArgumentRefused(() => e.Verify(a, 9, Enumerable.Range(0, a.Length).SelectMany(q => seed.Window.AsSpan(q * 8, 8).ToArray().Concat(new[] { seed.Anchors[q] })).ToArray(), drawn));
            float[] shortHead = Poison(a.Length * Vocab - 1, pass);
            // Use an actually short full-head buffer: active poison plus no tail
            // guard is passed here so the caller guard cannot fill the missing row.
            float[] actualShort = shortHead.Take(a.Length * Vocab - 1).ToArray(); byte[] actualBefore = MemoryMarshal.AsBytes(actualShort.AsSpan()).ToArray();
            ArgumentRefused(() => e.Verify(a, 1, seed.Anchors, drawn, actualShort));
            Need(actualBefore.AsSpan().SequenceEqual(MemoryMarshal.AsBytes(actualShort.AsSpan())), "Rejected short head was overwritten.");
            int[] positions = a.Select(x => x.NPast).ToArray(), validDrawn = Enumerable.Repeat(IntGuard, a.Length + Guard).ToArray();
            float[] validHead = Poison(a.Length * Vocab, pass); e.Verify(a, 1, seed.Anchors, validDrawn, validHead); Head(validHead, a.Length, validDrawn);
            store.Record(seed.Id + "/negative-admission-valid-verify1", pass, validHead, a.Length);
            string verified = Encoded(new { state = Digest(e, a, true), meta = Metadata(e, a) });
            string other = Encoded(new { state = Digest(e, b, true), meta = Metadata(e, b) });
            float[] rejectedHead = Poison(b.Length * Vocab, pass); byte[] rejectedBefore = MemoryMarshal.AsBytes(rejectedHead.AsSpan()).ToArray();
            bool pendingRefused = false;
            try { e.DecodeBatch(b, seed.NextTokens, rejectedHead, Enumerable.Repeat(IntGuard, b.Length + Guard).ToArray()); }
            catch (InvalidOperationException) { pendingRefused = true; }
            Need(pendingRefused && rejectedBefore.AsSpan().SequenceEqual(MemoryMarshal.AsBytes(rejectedHead.AsSpan())), "Pending Verify allowed another target pass or overwrote caller output.");
            Need(verified == Encoded(new { state = Digest(e, a, true), meta = Metadata(e, a) }) && other == Encoded(new { state = Digest(e, b, true), meta = Metadata(e, b) }), "Refused other-slot decode changed verified checkpoint/other slot."); checks++;
            e.Commit(a, 1, Enumerable.Repeat(1, a.Length).ToArray());
            Need(a.Select((x, i) => x.NPast == positions[i] + 1).All(x => x) && Metadata(e, a).All(x => x.PendingBytes == 51200), "Valid original checkpoint could not commit after rejected target request."); checks++;
            records.Add(new { seed.Id, pass, checks, selfCopyRejected = true, window9Rejected = true, shortHeadRejected = true,
                pendingOtherSlotDecodeRejected = true, rejectedPathsPreserveStateAndOutputs = true, originalValidCheckpointCommitted = true });
        }
        finally { Release(e, b, seedSlots[0]); Release(e, a, seedSlots[0]); }
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
        int contracts = 0; var seeds = Seeds(Enumerable.Range(0, 4096).Select(i => i % 257).ToArray(), Enumerable.Range(17, 16).ToArray());
        Need(seeds.Length == 8 && seeds.Select(x => x.Prompts.Length).Distinct().SequenceEqual(LaneCounts), "All lanes1..4 and both context classes required."); contracts++;
        foreach (var seed in seeds)
        {
            int n = seed.Prompts.Length; Need(seed.Prompts.Select((p, i) => p.Length == (seed.Id.EndsWith("127", StringComparison.Ordinal) ? 127 : 2047) + i).All(x => x), "Divergent ring-boundary positions missing."); contracts++;
            Need(seed.Window.Length == n * 8 && seed.Anchors.Length == n && seed.NextTokens.Length == n && seed.NextAnchors.Length == n, "Fixed inputs shape differs."); contracts++;
            Need(Mixed(n).Length == n && Mixed(n).All(k => k >= 0 && k <= 8), "Mixed keeps invalid."); contracts++;
        }
        Need(seeds.Any(s => s.Prompts.Any(p => p.Length < 2048)) && seeds.Any(s => s.Prompts.Any(p => p.Length > 2048)), "Ring wrap coverage missing."); contracts++;
        Need(Phases.Length == 7 && Phases.Distinct().Count() == 7 && Phases.Count(p => p.StartsWith("pending", StringComparison.Ordinal)) == 3, "Phase matrix differs."); contracts++;
        long rows = seeds.Sum(s => s.Prompts.Length * (8L + Phases.Length * 15L + 36L));
        Need(rows == 2980 && rows * Vocab * 4 < Budget, "Successful complete raw-output bound exceeds budget."); contracts++;
        Need(BitConverter.SingleToInt32Bits(0f) != BitConverter.SingleToInt32Bits(BitConverter.Int32BitsToSingle(unchecked((int)0x80000000))), "Exact signed-zero gate must distinguish signs."); contracts++;
        float[] poison = Poison(2, 0); Need(poison.Skip(2).All(x => BitConverter.SingleToInt32Bits(x) == GuardBits) && !float.IsFinite(poison[0]), "Caller poison/guard construction differs."); contracts++;
        return new { CPU = "PASS", contracts, seeds = seeds.Length, phases = Phases, laneCounts = LaneCounts, resetPasses = 2,
            actualPhaseCasesPlanned = seeds.Length * Phases.Length * 2, lifecycleVariantsPerCase = 3, runsPlanned = 496,
            actualProposalBatchCallsPlanned = 2000, rawHeadRowEquivalentBound = rows, rawVectorBudgetBytes = Budget,
            rawSuccessfulByteBound = rows * Vocab * 4, GPU = "NOT_RUN", independentOriginalDraftOracle = "NOT_RUN", broadQuality = "NOT_RUN", endToEndPerformance = "NOT_QUALIFIED" };
    }
    public static int Main(string[] args)
    {
        string? output = null; bool owned = false; Store? store = null; var cases = new List<object>(); var handoffs = new List<object>(); var negative = new List<object>(); var campaigns = new List<object>();
        var state = new Dictionary<string, object?> { ["schema"] = "ninfer-dflash-original-host-integration-probe/1", ["complete"] = false,
            ["GPU"] = "NOT_RUN", ["independentOriginalDraftOracle"] = "NOT_RUN", ["broadQuality"] = "NOT_RUN", ["endToEndPerformance"] = "NOT_QUALIFIED",
            ["cases"] = cases, ["handoffs"] = handoffs, ["campaigns"] = campaigns, ["negativeAdmission"] = negative };
        void Persist() { if (owned && output != null) Save(Path.Combine(output, "result.json"), state); }
        void Finish() { Persist(); if (owned && output != null) File.WriteAllText(Path.Combine(output, "result.json.sha256"), FilePin(Path.Combine(output, "result.json")).Sha256 + "  result.json\n"); }
        try
        {
            Need(BitConverter.IsLittleEndian, "Little-endian exact physical-head storage required."); var options = Options(args);
            if (options.ContainsKey("--self-test")) { Console.WriteLine(JsonSerializer.Serialize(SelfTest(), Json)); return 0; }
            if (!options.ContainsKey("--execute")) { Console.WriteLine("PLAN: --execute --artifact ORIGINAL --expect-directory-sha SHA --inputs ORIGINAL_BODY_INPUTS --expect-inputs-sha SHA --draft-module ABS_MODULE --out FRESH_LOCAL_IGNORED_DIRECTORY [--context8192] [--ubatch512]; explicit TS_Q35_NINFER_VERIFY=1. FixedK7, lanes1/2/3/4, bases127/2047, seven mandatory phases, exact complete proposal/target heads under reset/copy/sync+completedasync host paths. No model/CUDA work without --execute."); return 0; }
            int context = Positive(options, "--context", 8192), ubatch = Positive(options, "--ubatch", 512);
            Need(context is >= 4096 and <= 262144 && ubatch is >= 128 and <= 2048, "Bounded context/ubatch required.");
            string artifact = Path.GetFullPath(Required(options, "--artifact")), input = Path.GetFullPath(Required(options, "--inputs")), module = Path.GetFullPath(Required(options, "--draft-module"));
            string directorySha = ValidationSha256.Normalize(Required(options, "--expect-directory-sha")), inputSha = ValidationSha256.Normalize(Required(options, "--expect-inputs-sha"));
            output = Required(options, "--out"); Fresh(output); var artifactBefore = Stat(artifact);
            Need(!Link(input) && !Link(module), "Regular pinned input/native files required."); byte[] original = File.ReadAllBytes(input); Need(Hash(original) == inputSha, "Original fixed input SHA differs.");
            var env = EnvironmentSnapshot(); Need(Environment.GetEnvironmentVariable("TS_Q35_NINFER_VERIFY") == "1", "Explicit target verification candidate flag required.");
            Need(!env.ContainsKey("TS_Q35_NINFER_GDN_CHUNK_PTX") && (Environment.GetEnvironmentVariable("TS_Q35_NINFER_HOST_EMBEDDING") is null or "0"), "Unqualified chunk/host-embedding combinations are outside this probe.");
            var draftPins = new[] { module, Path.ChangeExtension(module, ".fatbin"), Path.ChangeExtension(module, ".fatbin") + ".sha256" }.Distinct().Where(File.Exists).Select(FilePin).ToArray();
            var native = NativePins(env).Concat(draftPins).DistinctBy(p => p.Path).ToArray();
            var assemblies = new[] { typeof(Program).Assembly, typeof(Qwen35CudaExecutor).Assembly, typeof(Q35CudaEngine).Assembly, typeof(NinferArtifactReader).Assembly }.Select(a => FilePin(a.Location)).ToArray();
            var pool = Inputs(original, Valid); var seeds = Seeds(pool.Prompt, pool.Continuation);
            Directory.CreateDirectory(output); owned = true; store = new(output); state["vectors"] = store.Vectors;
            File.WriteAllBytes(Path.Combine(output, "source-inputs.json"), original); Save(Path.Combine(output, "derived-inputs.json"), new { seeds, phases = Phases, mixedKeptByLanes = LaneCounts.ToDictionary(n => n, Mixed), originalInputSha256 = inputSha });
            var inputs = new[] { FilePin(input), FilePin(Path.Combine(output, "source-inputs.json")), FilePin(Path.Combine(output, "derived-inputs.json")) };
            state["artifactBefore"] = artifactBefore; state["artifactDirectorySha256"] = directorySha; state["fullArtifactSha256Recomputed"] = false;
            state["nativeFiles"] = native; state["assemblies"] = assemblies; state["inputs"] = inputs; state["environment"] = env; state["context"] = context; state["ubatch"] = ubatch; state["rawOutputBudgetBytes"] = Budget;
            state["limits"] = new[] { "Deterministic exact same-history integration check, not an independent original Ignis draft oracle, task quality, acceptance distribution, broad speculative sampler or speed qualification.", "All seven proposals and every248320-value head are retained/compared, including physical padding. Confidence1 is a fixed-window admission marker, not a probability.", "Partial window8 draft states compare identical physical operation histories only. Shortened ordinary prefixes may use different original A16 FC/QKV arithmetic and are not exact proposal oracles.", "Pending-anchor control alone compares Verify1+Commit1 to ordinary DecodeBatch1 at the same physical target shape and per-lane T1 context encoding.", "Live source/seed slots can retain shared KV blocks during host restore. Raw recurrent/ring restore and owned pending metadata are exercised; physical KV eviction/upload is not forced.", "Full-state hashes read individual Conv/State/RingK/RingV/RingPos allocations. Active slot ring may be separate from its recurrent arena after attachment.", "Original complete artifact SHA/device identity require external operator provenance; no skipped/refused case is PASS, timers are diagnostics only." };
            Persist(); state["GPU"] = "ATTEMPTED";
            using var executor = new Qwen35CudaExecutor(artifact, context, ubatch); var e = executor.Engine;
            Need(e.IsNinferModel && !e.HasDrafter && executor.Draft == null && e.VocabSize == Vocab, "Fresh target-only original body factory required.");
            // The production draft loading bridge, not a validation-built descriptor.
            // Sources/readers are closed immediately after synchronous original upload.
            using (var reader = new NinferArtifactReader(artifact))
            {
                Need(reader.DirectorySha256 == directorySha, "Original artifact directory differs."); var front = NinferQwen38Frontend.FromReader(reader);
                Need(front.Tokenizer.VocabSize == Valid && front.Tokenizer.HeadVocabSize == Vocab, "Original frontend geometry differs.");
                var plan = NinferQwen38DraftPlan.Create(reader); using var source = new NinferQwen38DraftSource(plan, K, CudaNinferDraftHeadPolicy.FullTargetVocabulary);
                state["draftOriginalDescriptors"] = plan.Tensors.Select(t => new { t.Descriptor.Name, t.Role, t.Descriptor.Shape, t.Descriptor.Format, t.Descriptor.AbsoluteOffset, t.Descriptor.ByteLength, t.Descriptor.WeightDivisorBits }).ToArray();
                state["draftDescriptorProvenance"] = source.Descriptor.Provenance; e.AttachNinferDrafter(source.Descriptor, module);
            }
            Need(e.HasNinferDrafter && e.HasDrafter && e.DraftBlockSize == 8 && e.DraftRingRows == 2048 && e.NinferDraftMaximumLanes == 4, "Native public draft integration/maximum head shape differs.");
            Need(draftPins.Where(p => !p.Path.EndsWith(".sha256", StringComparison.OrdinalIgnoreCase)).Any(p => ValidationSha256.Matches(e.NinferDraftModuleSha256, p.Sha256)), "Loaded original draft module image differs.");
            state["GPU"] = "EXECUTED"; state["draftPayloadSha256"] = e.NinferDraftPayloadSha256; state["draftModuleImageSha256"] = e.NinferDraftModuleSha256; state["numericalStateNamespace"] = e.KVStateNamespaceSuffix;
            e.ConfigureHostPool(4L << 30, 0); var first = e.Active; var extras = new List<Q35CudaEngine.Slot>(); var semantics = new Dictionary<string, string>(StringComparer.Ordinal);
            try
            {
                var all = new Q35CudaEngine.Slot[4]; all[0] = first; for (int i = 1; i < 4; i++) { all[i] = e.CreateSlot(0); extras.Add(all[i]); }
                foreach (var seed in seeds) for (int pass = 0; pass < 2; pass++)
                {
                    var slots = all.Take(seed.Prompts.Length).ToArray(); SeedPrefill(e, slots, seed);
                    object[] parentBefore = Digest(e, slots, true); Meta[] parentMeta = Metadata(e, slots);
                    NegativeAdmission(e, slots, seed, pass, store, negative);
                    Proposal seedProposal = Propose(e, slots, seed.Anchors, pass); store.Record(seed.Id + "/initial-proposal", pass, seedProposal.Head, slots.Length * K);
                    int[] proposalWindow = Enumerable.Range(0, slots.Length).SelectMany(q => new[] { seed.Anchors[q] }.Concat(seedProposal.Tokens[q])).ToArray();
                    int start = cases.Count;
                    foreach (string phase in Phases)
                    {
                        string id = seed.Id + '/' + phase; var sourceSlots = Clone(e, slots);
                        try
                        {
                            var transition = PreparePhase(e, sourceSlots, seed, phase, pass, store, proposalWindow);
                            Meta[] sourceMetadata = Metadata(e, sourceSlots); object[] sourceBefore = Digest(e, sourceSlots, true); object[] sourceLive = Digest(e, sourceSlots, true, liveKv: true);
                            Outcome baseline; var baselineSlots = Clone(e, sourceSlots);
                            try { Need(Encoded(sourceLive) == Encoded(Digest(e, baselineSlots, true, liveKv: true)), "Baseline clone did not preserve full recurrent/ring/liveKV bytes."); baseline = Run(e, baselineSlots, seed, pass, store, id + "/baseline"); }
                            finally { Release(e, baselineSlots, slots[0]); }
                            if (phase is "seed" or "zero8") PairProposal(store, id + "/initial-actual-proposal", seedProposal, baseline.Before, slots.Length);
                            store.Record(id + "/proposal", pass, baseline.Before.Head, slots.Length * K); store.Record(id + "/target-next", pass, baseline.Target, slots.Length); store.Record(id + "/proposal-after-target", pass, baseline.After.Head, slots.Length * K);
                            string semantic = Encoded(new { baseline.Before.Tokens, beforeConfidence = baseline.Before.Confidence, afterTokens = baseline.After.Tokens, afterConfidence = baseline.After.Confidence, baseline.BeforeMeta, baseline.AfterMeta });
                            if (pass == 0) Need(semantics.TryAdd(id, semantic), "Duplicate semantic reset reference."); else Need(semantics.TryGetValue(id, out var old) && old == semantic, "Selector tokens/confidence/frontier reset-repeat differs.");
                            foreach (string route in new[] { "clone", "host-sync", "host-async" })
                            {
                                var variant = route == "clone" ? Clone(e, sourceSlots) : HostHandoff(e, sourceSlots, route == "host-async", handoffs);
                                try { Need(Encoded(sourceLive) == Encoded(Digest(e, variant, true, liveKv: true)), "Lifecycle route did not preserve full recurrent/ring/liveKV bytes: " + route); Pair(store, id + '/' + route, baseline, Run(e, variant, seed, pass, store, id + '/' + route), slots.Length); }
                                finally { Release(e, variant, slots[0]); }
                            }
                            if (phase.StartsWith("pending", StringComparison.Ordinal))
                            {
                                var ordinary = Clone(e, slots);
                                try
                                {
                                    var ordinaryTransition = PreparePhase(e, ordinary, seed, phase, pass, store, proposalWindow, ordinaryAnchorControl: true);
                                    Need(ordinaryTransition.Keys.SequenceEqual(transition.Keys), "Ordinary pending control operation history differs.");
                                    foreach (var pair in transition) store.Pair(id + "/ordinary-anchor-control/" + pair.Key, pair.Value, ordinaryTransition[pair.Key], pair.Value.Length / Vocab);
                                    Pair(store, id + "/ordinary-T1-anchor-control", baseline, Run(e, ordinary, seed, pass, store, id + "/ordinary-T1-anchor-control"), slots.Length, requireInitialMetadata: false);
                                }
                                finally { Release(e, ordinary, slots[0]); }
                            }
                            Need(Encoded(sourceBefore) == Encoded(Digest(e, sourceSlots, true)) && Encoded(sourceMetadata) == Encoded(Metadata(e, sourceSlots)), "Lifecycle clones/host operations mutated the source checkpoint.");
                            cases.Add(new { id, pass, lanes = slots.Length, phase, originalPositions = slots.Select(s => s.NPast).ToArray(), sourceMetadata, lifecycleRoutes = new[] { "clone", "host-sync", "host-async" }, pendingOrdinaryT1Control = phase.StartsWith("pending", StringComparison.Ordinal), passed = true }); Persist();
                        }
                        finally { Release(e, sourceSlots, slots[0]); }
                    }
                    Need(cases.Count - start == Phases.Length && Encoded(parentBefore) == Encoded(Digest(e, slots, true)) && Encoded(parentMeta) == Encoded(Metadata(e, slots)), "Parent reset checkpoint mutated or phase matrix incomplete.");
                    campaigns.Add(new { seed.Id, pass, phases = Phases.Length, parentBytesUnchanged = true, parentMetadata = parentMeta }); Persist();
                }
            }
            finally { e.Active = first; foreach (var slot in extras) e.FreeSlot(slot); }
            Need(cases.Count == 112 && handoffs.Count == 224 && campaigns.Count == 16 && negative.Count == 16, "Incomplete host/lifecycle matrix.");
            Need(store.RepeatRows == 2980 && store.ExactPairs == 32800, "Incomplete complete-head exact/reset comparison coverage.");
            Need(e.HostBytesInUse == 0 && e.HostDraftMetadataBytesInUse == 0, "Retained host snapshot/pending metadata charge leaked.");
            state["graphCountsDiagnosticOnly"] = new { e.NinferDecodeGraphCaptureCount, e.NinferDecodeGraphReplayCount, e.NinferDecodeGraphEvictionCount }; state["hostBytesInUseBeforeDisposal"] = e.HostBytesInUse; state["hostDraftMetadataBytesInUseBeforeDisposal"] = e.HostDraftMetadataBytesInUse;
            executor.Dispose(); VerifyPins(inputs.Concat(native).Concat(assemblies)); store.VerifyFiles();
            Need(Encoded(env) == Encoded(EnvironmentSnapshot()) && Encoded(artifactBefore) == Encoded(Stat(artifact)), "Environment/artifact stat changed.");
            using var afterReader = new NinferArtifactReader(artifact); Need(afterReader.DirectorySha256 == directorySha, "Original directory changed.");
            state["complete"] = true; state["passed"] = true; state["exactHeadRowPairs"] = store.ExactPairs; state["resetRepeatHeadRows"] = store.RepeatRows; state["rawBytes"] = store.Bytes; state["artifactAfter"] = Stat(artifact);
            Finish(); SaveCompleted(output, state); Console.WriteLine(JsonSerializer.Serialize(new { complete = true, passed = true, phaseCases = cases.Count, hostHandoffs = handoffs.Count, exactHeadRowPairs = store.ExactPairs, resetRepeatRows = store.RepeatRows, rawBytes = store.Bytes, independentOriginalDraftOracle = "NOT_RUN", broadQuality = "NOT_RUN", endToEndPerformance = "NOT_QUALIFIED" })); return 0;
        }
        catch (Exception error) { state["passed"] = false; state["error"] = error.ToString(); if (store != null) state["rawBytes"] = store.Bytes; Finish(); Console.Error.WriteLine(error); return 1; }
    }
}
