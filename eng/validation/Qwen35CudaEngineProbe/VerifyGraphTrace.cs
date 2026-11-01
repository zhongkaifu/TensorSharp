// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Runtime;

internal static class VerifyGraphTrace
{
    // Separate processes, same tokens/configuration: graph=0 and graph=1 must produce
    // byte-identical trace.json. This is a submission-equivalence probe, not an answer-
    // quality or performance benchmark. No drafter is loaded, so atomic draft reductions
    // cannot mask a target numerical mismatch.
    public static int Run(string modelPath, string output, string? lengths, bool snapshots)
    {
        int[] keyLengths = (lengths ?? "127,128,129,255,256,257,8192,8193,16384,16385")
            .Split(',').Select(int.Parse).Distinct().Order().ToArray();
        if (keyLengths.Length == 0 || keyLengths[0] <= 8)
            throw new ArgumentException("Verify attention boundary lengths must exceed the 8-row window.");
        Directory.CreateDirectory(output);
        File.WriteAllText(Path.Combine(output, "run.json"), "{\"complete\":false}");
        using var metadata = new GgufFile(modelPath);
        var tokenizer = ModelBase.CreateTokenizerFromGguf(metadata);
        int[] teacher = tokenizer.Encode(
            "A library stores books on labelled shelves. Each book has an author, a title, and a publication year. " +
            "A reader borrows two books on Monday and returns them on Thursday. The catalogue preserves the original order. " +
            "In Python, a function can iterate over a list and return the sum of its integer elements. " +
            "The capital of Japan is Tokyo. The capital of France is Paris. The numbers one through five sum to fifteen.\n",
            addSpecial: false).ToArray();
        using var exec = new Qwen35CudaExecutor(modelPath, checked(keyLengths[^1] + 64), 512);
        var engine = exec.Engine;
        var primary = engine.Active;
        var trace = new List<object>();
        int record = 0;
        string tracePath = Path.Combine(output, "trace.json");

        string Hash(float[] values)
        {
            if (values.Any(v => !float.IsFinite(v)))
                throw new InvalidOperationException("Non-finite Verify logits.");
            return Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan()))).ToLowerInvariant();
        }
        void Save(string name, float[] values) => File.WriteAllBytes(Path.Combine(output, name), MemoryMarshal.AsBytes(values.AsSpan()));
        int[] Tokens(int start, int count) => Enumerable.Range(0, count).Select(i => teacher[(start + i) % teacher.Length]).ToArray();
        void FillTo(int tokens)
        {
            engine.Active = primary;
            if (primary.NPast > tokens)
                engine.Reset();
            if (primary.NPast < tokens)
                engine.Forward(Tokens(primary.NPast, tokens - primary.NPast), null);
            engine.Synchronize();
        }
        Q35CudaEngine.Slot[] Clone(int lanes)
        {
            var slots = new List<Q35CudaEngine.Slot>();
            try
            {
                for (int i = 0; i < lanes; i++)
                {
                    var slot = engine.CreateSlot(0);
                    slots.Add(slot);
                    engine.CopySlot(primary, slot);
                }
                return slots.ToArray();
            }
            catch
            {
                foreach (var slot in slots) engine.FreeSlot(slot);
                throw;
            }
        }
        void Free(Q35CudaEngine.Slot[] slots)
        {
            engine.Active = primary;
            foreach (var slot in slots) engine.FreeSlot(slot);
        }
        int[] WindowTokens(Q35CudaEngine.Slot[] slots, int window) => slots
            .SelectMany((slot, lane) => Tokens(slot.NPast + 13 * lane, window)).ToArray();
        (string Hash, int[] Drawn) Check(string label, Q35CudaEngine.Slot[] slots, int window, int[] tokens)
        {
            int id = record++, rows = slots.Length * window;
            int[] positions = slots.Select(s => s.NPast).ToArray();
            var first = new float[rows * exec.VocabSize];
            var second = new float[first.Length];
            var drawn = new int[rows];
            var again = new int[rows];
            engine.Verify(slots, window, tokens, drawn, first);
            int featureWidth = engine.FeatureWidth;
            var firstFeatures = new float[rows * featureWidth];
            var secondFeatures = new float[firstFeatures.Length];
            engine.ReadFeatureRows(firstFeatures, rows);
            if (snapshots) Save($"verify-{id:D3}.f32", first);
            if (snapshots && featureWidth > 0) Save($"features-{id:D3}.f32", firstFeatures);
            string hash = Hash(first);
            string featureHash = Hash(firstFeatures);
            engine.Verify(slots, window, tokens, again, second);
            engine.ReadFeatureRows(secondFeatures, rows);
            if (Hash(second) != hash || Hash(secondFeatures) != featureHash ||
                !drawn.SequenceEqual(again) || !positions.SequenceEqual(slots.Select(s => s.NPast)))
            {
                Save($"verify-{id:D3}.first.f32", first);
                Save($"verify-{id:D3}.repeat.f32", second);
                Save($"features-{id:D3}.first.f32", firstFeatures);
                Save($"features-{id:D3}.repeat.f32", secondFeatures);
                File.WriteAllText(Path.Combine(output, "repeat-failure.json"), JsonSerializer.Serialize(new {
                    label, id, positions, positions_after = slots.Select(s => s.NPast).ToArray(),
                    logits_sha256 = hash, repeated_logits_sha256 = Hash(second), drawn, repeated_drawn = again,
                    feature_width = featureWidth, features_sha256 = featureHash, repeated_features_sha256 = Hash(secondFeatures),
                    complete = false
                }, new JsonSerializerOptions { WriteIndented = true }));
                throw new InvalidOperationException($"Verify changed logits or advanced host state on repeat: {label}");
            }
            trace.Add(new { id, label, window, positions, tokens, logits_sha256 = hash, drawn,
                feature_width = featureWidth, features_sha256 = featureHash });
            File.WriteAllText(tracePath, JsonSerializer.Serialize(trace, new JsonSerializerOptions { WriteIndented = true }));
            return (hash, drawn);
        }

        foreach (int maxKeys in keyLengths)
        {
            FillTo(maxKeys - 8);
            foreach (int lanes in new[] { 1, 2, 4 })
            {
                var slots = Clone(lanes);
                try
                {
                    // The clones share their prefix blocks; Verify's first write must copy a
                    // shared tail block before the captured table reader can touch it.
                    Check($"boundary-{maxKeys}-lanes-{lanes}", slots, 8, WindowTokens(slots, 8));
                    engine.Commit(slots, 8, Enumerable.Range(0, lanes).Select(i => i + 1).ToArray());
                    Array.Reverse(slots);
                    Check($"committed-reordered-{maxKeys}-lanes-{lanes}", slots, 8, WindowTokens(slots, 8));
                }
                finally { Free(slots); }
            }
        }
        // 64 distinct lane/window combinations force the bounded cache to evict and recapture.
        // They straddle the GEMV/FP4 row threshold without altering either arithmetic path.
        for (int lanes = 1; lanes <= 8; lanes++)
        {
            var slots = Clone(lanes);
            try
            {
                for (int window = 1; window <= 8; window++)
                    Check($"lru-lanes-{lanes}-window-{window}", slots, window, WindowTokens(slots, window));
            }
            finally { Free(slots); }
        }
        {
            var slots = Clone(1);
            try
            {
                long captures = engine.VerifyGraphCaptureCount;
                Check("lru-revisit-evicted-first-shape", slots, 1, WindowTokens(slots, 1));
                if (Environment.GetEnvironmentVariable("TS_Q35_VERIFY_GRAPHS") == "1" && engine.VerifyGraphCaptureCount != captures + 1)
                    throw new InvalidOperationException("The oldest evicted Verify shape was not recaptured on revisit.");
            }
            finally { Free(slots); }
        }
        var original = Clone(1)[0];
        Q35CudaEngine.HostSlot? host = null;
        Q35CudaEngine.Slot? restored = null;
        Q35CudaEngine.Slot? fresh = null;
        try
        {
            host = engine.CopyToHost(original);
            int[] tokens = WindowTokens(new[] { original }, 8);
            var reference = Check("host-source", new[] { original }, 8, tokens);
            engine.FreeSlot(original);
            original = null!;
            restored = engine.CreateSlot(0);
            engine.CopyFromHost(host, restored);
            var replay = Check("host-restored-new-slot", new[] { restored }, 8, tokens);
            if (reference.Hash != replay.Hash || !reference.Drawn.SequenceEqual(replay.Drawn))
                throw new InvalidOperationException("Host restore changed Verify logits.");

            engine.ResetSlot(restored);
            engine.Active = restored;
            engine.Forward(Tokens(0, 121), null);
            fresh = engine.CreateSlot(0);
            engine.Active = fresh;
            engine.Forward(Tokens(0, 121), null);
            int[] resetTokens = WindowTokens(new[] { restored }, 8);
            var reset = Check("reset-refilled-slot", new[] { restored }, 8, resetTokens);
            var clean = Check("fresh-matching-slot", new[] { fresh }, 8, resetTokens);
            if (reset.Hash != clean.Hash || !reset.Drawn.SequenceEqual(clean.Drawn))
                throw new InvalidOperationException("Reset/refill differs from a fresh slot.");

            // Layout invalidation plus growth after Verify buffers already exist. These taps
            // copy residuals only, so target logits must remain exactly those of the no-tap run.
            foreach (int[]? taps in new int[]?[] { new[] { 0 }, new[] { 0, 1 }, null })
            {
                engine.SetFeatureTaps(taps);
                if (engine.CapturedVerifyGraphCount != 0)
                    throw new InvalidOperationException("Feature layout change retained stale Verify graphs.");
                var selected = Check($"feature-layout-{taps?.Length ?? 0}", new[] { fresh }, 8, resetTokens);
                if (selected.Hash != clean.Hash || !selected.Drawn.SequenceEqual(clean.Drawn))
                    throw new InvalidOperationException("Feature layout changed target Verify logits.");
            }
        }
        finally
        {
            engine.Active = primary;
            if (host != null) engine.FreeHostSlot(host);
            if (original != null) engine.FreeSlot(original);
            if (restored != null) engine.FreeSlot(restored);
            if (fresh != null) engine.FreeSlot(fresh);
        }
        bool requested = Environment.GetEnvironmentVariable("TS_Q35_VERIFY_GRAPHS") == "1";
        if (requested && (engine.VerifyGraphCaptureCount < 33 || engine.VerifyGraphEvictionCount < 1 ||
            engine.VerifyGraphReplayCount <= engine.VerifyGraphCaptureCount || engine.CapturedVerifyGraphCount > 32))
            throw new InvalidOperationException("Requested Verify graph capture/replay/eviction coverage is incomplete.");
        trace.Add(new { complete = true, records = record });
        File.WriteAllText(tracePath, JsonSerializer.Serialize(trace, new JsonSerializerOptions { WriteIndented = true }));
        File.WriteAllText(Path.Combine(output, "run.json"), JsonSerializer.Serialize(new {
            complete = true, records = record, no_drafter = !engine.HasDrafter, key_lengths = keyLengths,
            verify_graph_setting = Environment.GetEnvironmentVariable("TS_Q35_VERIFY_GRAPHS"),
            live_graphs = engine.CapturedVerifyGraphCount, captures = engine.VerifyGraphCaptureCount,
            replays = engine.VerifyGraphReplayCount, evictions = engine.VerifyGraphEvictionCount, snapshots
        }, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"[q35-probe] Verify graph trace complete: {record} fixed-input records; captures={engine.VerifyGraphCaptureCount}, replays={engine.VerifyGraphReplayCount}, evictions={engine.VerifyGraphEvictionCount}");
        return 0;
    }
}
