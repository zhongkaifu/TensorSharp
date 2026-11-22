// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The legacy model's real-weight KV tests (MuseGlimmerKvSnapshotTests, removed with it in S9: decode across a cache grow, a restored prefix below and
// past the ring) re-pointed at the unified-runtime family (stage S3 of docs/design/muse-glimmer-unified.md). The runtime keeps
// no position-addressed block snapshots where layers ring; a prefix is restored from a checkpoint (a deep copy of a request's
// caches, the rings whole), a clone of it, or a checkpoint file. Sizes come from the model's seams
// (StandardDecoderModel.InitialCacheLengthForTests), not from MAX_CONTEXT or TS_KV_INITIAL_TOKENS.
//
// Opt-in, as MuseGlimmerParityTests: TS_TEST_MODEL_DIR holds Muse-Glimmer-30B-UD-IQ2_XXS.gguf; TS_MUSE_GLIMMER_BACKEND names
// the backend (default: the pinned GGML backend; mlx runs MLX).
using System.Security.Cryptography;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class MuseGlimmerRuntimeKvTests(ITestOutputHelper output)
{
    /// <summary>
    /// Decode across the full layers' growths (the primary from 256 cells: 256 -> 512 -> 1024 at positions 256 and 512) and
    /// across a request slot's (reserved for its prompt and first token: 512 cells, grown at 512), with every step compared
    /// with a cache big enough from the start. Bit for bit: a decode row's arithmetic does not depend on its cache's capacity,
    /// so any difference is a lost or moved row (the legacy loop at position 2049 on ggml_metal).
    /// </summary>
    [MuseGlimmerModelFact]
    public void DecodingAcrossCacheGrowthsMatchesAPresizedCache()
    {
        BackendType backend = MuseGlimmerParity.Backend();
        string path = MuseGlimmerParity.FindModel();
        const int promptLength = 200, steps = 360;   // positions 200..559
        int[] prompt = null;

        List<(string Digest, int ArgMax)> Run(Func<StandardDecoderModel, int[], (Func<int, float[]> Step, Func<int> Capacity)> start,
            int initial, int[] forced, out int[] chosen, string label)
        {
            StandardDecoderModel.InitialCacheLengthForTests.Value = initial;
            try
            {
                using var model = (StandardDecoderModel)MuseGlimmerArchitecture.CreateRuntimeModel(path, backend);
                prompt ??= BuildPrompt(model, promptLength);
                var (step, capacity) = start(model, prompt);
                int first = capacity();
                var digests = new List<(string, int)>(steps);
                chosen = new int[steps];
                float[] logits = step(-1);
                for (int i = 0; i < steps; i++)
                {
                    digests.Add((Digest(logits), ArgMax(logits)));
                    chosen[i] = forced?[i] ?? ArgMax(logits);
                    if (i + 1 < steps) logits = step(chosen[i]);
                }
                output.WriteLine($"{label}: {first} -> {capacity()} cells over positions {promptLength}..{promptLength + steps - 1}");
                return digests;
            }
            finally { StandardDecoderModel.InitialCacheLengthForTests.Value = null; }
        }

        // The primary cache: the first call forwards the prompt, later ones a token.
        (Func<int, float[]>, Func<int>) Primary(StandardDecoderModel model, int[] p)
            => (t => t < 0 ? model.Forward(p) : model.Forward(new[] { t }), () => model.ActiveCacheCapacityForTests);
        // A request's slot through the unified step, reserved for `reserve` tokens.
        Func<StandardDecoderModel, int[], (Func<int, float[]>, Func<int>)> Slot(int reserve) => (model, p) =>
        {
            var unified = (IUnifiedBatchModel)model;
            unified.EnsureSequenceSlot("r", reserve);
            var logits = new float[1][];
            int at = 0;
            return (t =>
            {
                int[] rows = t < 0 ? p : new[] { t };
                unified.EnsureSequenceSlot("r", reserve, at + rows.Length);
                Assert.True(unified.TryForwardUnified(new[] { "r" }, new[] { rows }, new[] { at }, logits));
                at += rows.Length;
                return logits[0];
            }, () => model.SlotCapacityForTests("r") ?? 0);
        };

        var presized = Run(Primary, 8192, null, out int[] reference, $"{backend} primary presized");
        Compare(presized, Run(Primary, 256, reference, out _, $"{backend} primary from 256"), "the primary grown from 256 cells");
        Compare(presized, Run(Slot(8192), 8192, reference, out _, $"{backend} slot reserved for 8192"), "a slot reserved for 8192");
        Compare(presized, Run(Slot(promptLength + 1), 8192, reference, out _, $"{backend} slot reserved for {promptLength + 1}"),
            $"a slot reserved for {promptLength + 1} tokens and grown");

        void Compare(List<(string Digest, int ArgMax)> want, List<(string Digest, int ArgMax)> got, string what)
        {
            for (int i = 0; i < steps; i++)
                Assert.True(want[i].Digest == got[i].Digest,
                    $"{what}: step {i} (position {promptLength + i}) differs from the presized primary (argmax {want[i].ArgMax} vs {got[i].ArgMax})");
            output.WriteLine($"{what}: {steps} decode steps identical to the presized primary");
        }
    }

    /// <summary>
    /// A checkpoint taken past the rings' first lap (3,500 positions; the rings are 2,560 cells, 3,072 on MLX and direct CUDA,
    /// whose prompt steps carry 1,024 rows), cloned into a new request's slot, and written to
    /// a checkpoint file another model instance reads back and clones: both continue as the live cache does, logit for logit,
    /// for 8 tokens. The legacy test restored 256-token blocks, below the ring and past it.
    /// </summary>
    [MuseGlimmerModelFact]
    public void ACheckpointOfWrappedRingsContinuesAsTheLiveCacheDoes()
    {
        BackendType backend = MuseGlimmerParity.Backend();
        string path = MuseGlimmerParity.FindModel();
        const int prefixLength = 3500;
        int[] tokens, tail;
        var live = new List<float[]>();
        byte[] file;
        using (var model = (StandardDecoderModel)MuseGlimmerArchitecture.CreateRuntimeModel(path, backend))
        {
            Assert.True(model.RingCellsForTests(0) is > 0 and < prefixLength, "the sliding-window rings must have wrapped at the checkpoint");
            tokens = BuildPrompt(model, prefixLength + 8);
            tail = tokens[prefixLength..];
            var slots = (IBatchedPagedModel)model;
            model.Forward(tokens[..prefixLength]);
            Assert.True(slots.TryCheckpointActiveCache("shared"));
            foreach (int t in tail) live.Add((float[])model.Forward(new[] { t }).Clone());

            Assert.True(slots.TryCloneRetainedCache("shared", "clone"));
            Assert.False(slots.BindSequenceCache("clone"));
            Assert.Equal(prefixLength, model.CacheSeqLen);
            AssertContinues(model, tail, live, $"{backend} clone");
            slots.RestorePrimaryCache();

            using var stream = new MemoryStream();
            Assert.True(slots.TryExportRetainedCache("shared", stream));
            file = stream.ToArray();
            output.WriteLine($"{backend}: checkpoint file of {prefixLength} positions, {file.Length:N0} bytes (rings of {model.RingCellsForTests(0)} cells)");
        }
        using (var other = (StandardDecoderModel)MuseGlimmerArchitecture.CreateRuntimeModel(path, backend))
        {
            var slots = (IBatchedPagedModel)other;
            Assert.True(slots.TryImportRetainedCache("restored", new MemoryStream(file)));
            Assert.True(slots.TryCloneRetainedCache("restored", "c"));
            slots.BindSequenceCache("c");
            AssertContinues(other, tail, live, $"{backend} checkpoint file in a new instance");
        }
    }

    private void AssertContinues(ModelBase model, int[] tail, List<float[]> live, string label)
    {
        double worst = 0;
        for (int i = 0; i < tail.Length; i++)
        {
            float[] got = model.Forward(new[] { tail[i] });
            Assert.Equal(ArgMax(live[i]), ArgMax(got));
            for (int v = 0; v < got.Length; v++) worst = Math.Max(worst, Math.Abs(got[v] - live[i][v]));
        }
        output.WriteLine($"{label}: {tail.Length} tokens, max |delta| {worst:E3} against the live cache");
        Assert.True(worst == 0, $"{label}: the restored cache's logits differ from the live cache's by up to {worst:E3}");
    }

    private static string Digest(float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return Convert.ToHexString(SHA256.HashData(bytes));
    }

    /// <summary>A real tokenization tiled to <paramref name="length"/> tokens (the positions are real; the content does not matter).</summary>
    private static int[] BuildPrompt(ModelBase model, int length)
    {
        int[] seed = model.Tokenizer.Encode("The quick brown fox jumps over the lazy dog. 请详细介绍最终幻想7。", addSpecial: true).ToArray();
        var tokens = new List<int>(length);
        while (tokens.Count < length) tokens.AddRange(seed);
        return tokens.Take(length).ToArray();
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }
}
