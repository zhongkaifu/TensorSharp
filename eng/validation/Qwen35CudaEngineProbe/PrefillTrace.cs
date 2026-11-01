// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Runtime;

internal static class PrefillTrace
{
    // Compare separate production-precision processes with prepared GDN disabled/enabled.
    // Every complete vocabulary logit after prefill and a teacher-forced continuation is
    // hashed and saved; no greedy agreement or error tolerance substitutes for bit equality.
    public static int Run(string modelPath, string output, string? lengths, int ubatch)
    {
        int[] sizes = (lengths ?? "31,32,33,127,128,129,511,512,513,1024,2049,4096")
            .Split(',').Select(int.Parse).Distinct().Order().ToArray();
        if (sizes.Length == 0 || sizes[0] < 1 || ubatch < 1)
            throw new ArgumentException("Positive prefill lengths and ubatch required.");
        Directory.CreateDirectory(output);
        File.WriteAllText(Path.Combine(output, "run.json"), "{\"complete\":false}");
        using var metadata = new GgufFile(modelPath);
        var tokenizer = ModelBase.CreateTokenizerFromGguf(metadata);
        int[] teacher = tokenizer.Encode(
            "A programmer maintains a library of small functions. The function sum_numbers accepts a list of integers and returns their total. " +
            "The test data contains the values 2, 3, and 5, whose total is 10. A second function sorts the values while preserving repeated entries. " +
            "Every change is reviewed and compared with the previous result. The city of Tokyo is in Japan and Paris is in France.\n",
            addSpecial: false).ToArray();
        int[] Tokens(int start, int count) => Enumerable.Range(0, count).Select(i => teacher[(start + i) % teacher.Length]).ToArray();
        using var exec = new Qwen35CudaExecutor(modelPath, Math.Max(512, sizes[^1] + 64), ubatch);
        var engine = exec.Engine;
        var primary = engine.Active;
        var trace = new List<object>();
        var reference = new Dictionary<string, string>();
        int record = 0;
        string Hash(float[] values)
        {
            if (values.Any(v => !float.IsFinite(v)))
                throw new InvalidOperationException("Non-finite prefill/decode logits.");
            return Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan()))).ToLowerInvariant();
        }
        void Observe(string key, float[] logits, int repeat)
        {
            string hash = Hash(logits);
            if (repeat == 0)
            {
                File.WriteAllBytes(Path.Combine(output, $"logits-{record:D3}.f32"), MemoryMarshal.AsBytes(logits.AsSpan()));
                reference.Add(key, hash);
                trace.Add(new { record, key, logits_sha256 = hash, argmax = Argmax(logits) });
                record++;
                File.WriteAllText(Path.Combine(output, "trace.json"), JsonSerializer.Serialize(trace, new JsonSerializerOptions { WriteIndented = true }));
            }
            else if (reference[key] != hash)
            {
                File.WriteAllBytes(Path.Combine(output, "repeat-failure.f32"), MemoryMarshal.AsBytes(logits.AsSpan()));
                File.WriteAllText(Path.Combine(output, "repeat-failure.json"), JsonSerializer.Serialize(new {
                    key, reference_sha256 = reference[key], repeated_sha256 = hash, complete = false
                }));
                throw new InvalidOperationException($"Prefill/reset repeat changed full logits: {key}");
            }
        }
        void Decode(string key, Q35CudaEngine.Slot slot, float[] logits, int repeat)
        {
            engine.Active = slot;
            for (int step = 0; step < 8; step++)
            {
                engine.Forward(Tokens(slot.NPast, 1), logits);
                Observe(key + $"-decode-{step}", logits, repeat);
            }
        }
        foreach (int size in sizes)
        {
            var slot = engine.CreateSlot(0);
            try
            {
                for (int repeat = 0; repeat < 2; repeat++)
                {
                    engine.Active = slot;
                    engine.Reset();
                    var logits = new float[exec.VocabSize];
                    engine.Forward(Tokens(0, size), logits);
                    Observe($"prefill-{size}", logits, repeat);
                    Decode($"prefill-{size}", slot, logits, repeat);
                }
            }
            finally { engine.Active = primary; engine.FreeSlot(slot); }
        }
        // Variable-length padded groups exercise prepared scratch reuse and zero-g/beta tails.
        int[] batchSizes = { 129, 127, 65 };
        var slots = new List<Q35CudaEngine.Slot>();
        try
        {
            for (int i = 0; i < batchSizes.Length; i++) slots.Add(engine.CreateSlot(0));
            for (int repeat = 0; repeat < 2; repeat++)
            {
                foreach (var slot in slots) engine.ResetSlot(slot);
                var logits = new float[slots.Count][];
                engine.PrefillBatch(slots.ToArray(), batchSizes.Select((size, lane) => Tokens(lane * 17, size)).ToArray(), logits);
                for (int lane = 0; lane < slots.Count; lane++)
                {
                    Observe($"batch-lane-{lane}", logits[lane], repeat);
                    Decode($"batch-lane-{lane}", slots[lane], logits[lane], repeat);
                }
            }
        }
        finally { engine.Active = primary; foreach (var slot in slots) engine.FreeSlot(slot); }
        if (Environment.GetEnvironmentVariable("TS_Q35_GDN_PREPARED_PREFILL") == "1" && engine.PreparedGdnPrefillLaunchCount == 0)
            throw new InvalidOperationException("Prepared GDN was requested but never executed.");
        trace.Add(new { complete = true, records = record });
        File.WriteAllText(Path.Combine(output, "trace.json"), JsonSerializer.Serialize(trace, new JsonSerializerOptions { WriteIndented = true }));
        File.WriteAllText(Path.Combine(output, "run.json"), JsonSerializer.Serialize(new {
            complete = true, records = record, sizes, ubatch,
            prepared_gdn_setting = Environment.GetEnvironmentVariable("TS_Q35_GDN_PREPARED_PREFILL"),
            prepared_gdn_launches = engine.PreparedGdnPrefillLaunchCount,
            prepared_gdn_projection_tail_reused = engine.PreparedGdnScratchReusesProjection,
            no_drafter = !engine.HasDrafter
        }, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"[q35-probe] prefill trace complete: {record} full-logit records, prepared GDN launches={engine.PreparedGdnPrefillLaunchCount}");
        return 0;
    }
    private static int Argmax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++) if (values[i] > values[best]) best = i;
        return best;
    }
}
