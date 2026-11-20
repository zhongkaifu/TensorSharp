// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// End-to-end cover for Muse-Glimmer's KV block snapshot, on the real weights.
//
// Two bugs lived here, both only reachable once a single sequence ran long:
//
//  1. CRASH. The 39 sliding-window layers keep a RING of _kvSwaRows rows
//     (pad256(window + prefill chunk + 1) = 4352 by default) while the 13 full
//     layers get the whole context. KvBlockTransfer addressed both as
//     row == absolute position, so the block starting at 4352 read past the end of
//     every ring tensor and the server died in Buffer.MemoryCopy with
//     AccessViolationException, in BatchExecutor.CaptureNewlyFullBlocks. A prompt
//     plus 4608 generated tokens was enough.
//
//  2. STALE BYTES. The fused kernel writes K/V on-device only; the snapshot reads
//     host pointers. TryExtractKVBlock never pulled the device rows back, so on
//     ggml_cuda every captured block was the zero-filled host mirror. Nothing
//     crashed - restores just resurrected a cache full of zeros.
//
// Opt-in, same convention as MuseGlimmerParityTests:
//   TS_TEST_MODEL_DIR        directory holding Muse-Glimmer-*.gguf
//   TS_MUSE_GLIMMER_BACKEND  backend name (default: the GGML backend the process pins, TS_TEST_GGML_BACKEND; an unknown name
//                            fails the tests rather than silently picking one)
// These cover the legacy model. The unified-runtime family's equivalents (it keeps no block snapshots: a prefix restores from
// a checkpoint, a clone or a checkpoint file) are MuseGlimmerRuntimeKvTests.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// [ModelFact] for the Muse-Glimmer weights that also skips when the backend these
/// tests construct (TS_MUSE_GLIMMER_BACKEND, default the pinned GGML backend) is not the
/// GGML backend this process pinned: the second backend could only fail to initialize.
/// </summary>
[Xunit.Sdk.TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
public sealed class MuseGlimmerKvFactAttribute : FactAttribute, Xunit.Sdk.ITraitAttribute
{
    public string RequiresValue => "Models";

    public MuseGlimmerKvFactAttribute()
        => Skip = TestGates.ModelSkip("TS_TEST_MODEL_DIR", "muse-glimmer") ?? PinSkip();

    // An unknown backend name does not skip: the test runs and fails on it (MuseGlimmerParity.Skip's rule).
    private static string PinSkip()
    {
        try { return TestGates.GgmlPinSkip(MuseGlimmerKvSnapshotTests.ResolveBackend()); }
        catch (ArgumentException) { return null; }
    }
}

public class MuseGlimmerKvSnapshotTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";

    /// <summary>Matches SchedulerConfig.BlockSize, the size CaptureNewlyFullBlocks uses.</summary>
    private const int BlockSize = 256;

    private readonly ITestOutputHelper _output;
    public MuseGlimmerKvSnapshotTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// Walk a sequence past the sliding-window ring, capturing every block the way
    /// the batch executor does. Before the fix this faulted the process the first
    /// time a block started at or beyond _kvSwaRows.
    /// </summary>
    [MuseGlimmerKvFact]
    public void MuseGlimmer_CapturingBlocksPastTheSwaRing_DoesNotFault()
    {
        string modelPath = TryFindModel();
        if (modelPath == null) return;

        using var env = new EnvScope();
        env.Set("MAX_CONTEXT", "8192");
        using var model = ModelBase.Create(modelPath, ResolveBackend());

        int cap = model.MaxReusablePrefixTokens;
        _output.WriteLine($"MaxReusablePrefixTokens = {cap}");
        Assert.True(cap != int.MaxValue,
            "The sliding-window ring must publish a finite pooled-reuse cap; an unbounded cap would let " +
            "the executor re-inject positions the ring physically dropped.");

        // Comfortably past cap + BlockSize, which is the first block the old code
        // read out of bounds.
        int total = cap + 3 * BlockSize;
        int[] tokens = BuildPrompt(model, total);
        _output.WriteLine($"prompt tokens: {tokens.Length} (ring cap {cap}, block {BlockSize})");

        model.ResetKVCache();

        long blockBytes = model.ComputeKVBlockByteSize(BlockSize);
        Assert.True(blockBytes > 0);
        var scratch = new byte[blockBytes];

        int captured = 0, refused = 0, nonZero = 0;
        int computed = 0;
        double firstCaptureMs = 0, laterCaptureMs = 0;
        int laterCaptures = 0;
        // Feed it the way the executor does: prefill chunks, capturing after each.
        while (computed < tokens.Length)
        {
            int take = Math.Min(2048, tokens.Length - computed);
            model.Forward(tokens.Skip(computed).Take(take).ToArray());
            computed += take;

            for (int b = 0; b < computed / BlockSize; b++)
            {
                Array.Clear(scratch);
                var sw = System.Diagnostics.Stopwatch.StartNew();
                bool ok = model.TryExtractKVBlock(b * BlockSize, BlockSize, scratch);
                sw.Stop();
                // Only the first capture after a forward pays the device->host pull;
                // the rest of the round reads the mirror it just refreshed.
                if (b == 0) firstCaptureMs += sw.Elapsed.TotalMilliseconds;
                else { laterCaptureMs += sw.Elapsed.TotalMilliseconds; laterCaptures++; }
                if (!ok) { refused++; continue; }
                captured++;
                if (scratch.Any(x => x != 0)) nonZero++;
            }
        }

        _output.WriteLine($"capture attempts: {captured} ok, {refused} refused, {nonZero} non-zero");
        _output.WriteLine($"first capture after a forward: {firstCaptureMs:F1} ms total over 3 forwards " +
            $"(includes the KV device->host sync); remaining {laterCaptures} captures: " +
            $"{laterCaptureMs:F1} ms total ({laterCaptureMs / Math.Max(1, laterCaptures):F2} ms each)");

        // Getting here at all is the crash regression. The rest pins bug 2: a
        // captured block that is all zeros means we snapshotted a stale host mirror.
        Assert.True(captured > 0, "No block was capturable at all.");
        Assert.True(nonZero == captured,
            $"{captured - nonZero} of {captured} captured blocks were entirely zero - the device-side K/V " +
            "was not synchronized back to the host before the snapshot.");
    }

    /// <summary>
    /// Capture a prefix, throw the cache away, replay the blocks, and the model must
    /// pick up exactly where it left off. Bit-identical logits are the bar: the
    /// restore writes the same rows the prefill did, so nothing should shift.
    /// </summary>
    [MuseGlimmerKvFact]
    public void MuseGlimmer_RestoringACapturedPrefix_ReproducesTheLivePrefillLogits()
    {
        string modelPath = TryFindModel();
        if (modelPath == null) return;

        using var env = new EnvScope();
        env.Set("MAX_CONTEXT", "8192");
        using var model = ModelBase.Create(modelPath, ResolveBackend());

        // Stay inside the pooled cap: that is exactly the range the executor is
        // allowed to restore, and inside it no ring layer has wrapped.
        int prefixLen = Math.Min(1024, model.MaxReusablePrefixTokens);
        prefixLen -= prefixLen % BlockSize;
        Assert.True(prefixLen >= 2 * BlockSize);

        int[] tokens = BuildPrompt(model, prefixLen + 1);
        int[] prefix = tokens.Take(prefixLen).ToArray();
        int nextToken = tokens[prefixLen];

        // 1. Live prefill, capture each block, then one more token for the baseline.
        model.ResetKVCache();
        model.Forward(prefix);

        long blockBytes = model.ComputeKVBlockByteSize(BlockSize);
        int numBlocks = prefixLen / BlockSize;
        var blocks = new byte[numBlocks][];
        for (int b = 0; b < numBlocks; b++)
        {
            blocks[b] = new byte[blockBytes];
            Assert.True(model.TryExtractKVBlock(b * BlockSize, BlockSize, blocks[b]),
                $"capture of block {b} failed");
        }
        float[] baseline = (float[])model.Forward(new[] { nextToken }).Clone();

        // 2. Fresh cache, replay the captured blocks, same next token. Scrambled
        //    first: a reset keeps the device copies on the GPU backends, so a replay
        //    that restored nothing would otherwise read the live prefill's rows back.
        ScrambleCache(model, prefixLen + 1);
        for (int b = 0; b < numBlocks; b++)
        {
            Assert.True(model.TryInjectKVBlock(b * BlockSize, BlockSize, blocks[b]),
                $"restore of block {b} failed");
        }
        float[] restored = (float[])model.Forward(new[] { nextToken }).Clone();

        Assert.Equal(baseline.Length, restored.Length);
        int diffs = 0;
        float maxAbs = 0;
        for (int i = 0; i < baseline.Length; i++)
        {
            float d = Math.Abs(baseline[i] - restored[i]);
            if (d > maxAbs) maxAbs = d;
            if (d != 0) diffs++;
        }
        _output.WriteLine($"restored {numBlocks} blocks ({prefixLen} tokens); " +
            $"logit diffs {diffs}/{baseline.Length}, max |delta| {maxAbs:E3}");
        _output.WriteLine($"argmax live={ArgMax(baseline)} restored={ArgMax(restored)}");

        Assert.Equal(ArgMax(baseline), ArgMax(restored));
        Assert.True(maxAbs == 0f,
            $"Restored logits differ from the live prefill (max |delta| {maxAbs:E3}); the snapshot is lossy.");
    }

    /// <summary>
    /// The restore above again, at ggml_metal's default: the full layers START at
    /// 2048 rows, below the 4352-row sliding-window ring, and the prefix is the
    /// 1536 tokens the server's radix cache reused when the bug showed up. The
    /// kernel used to address the ring as a [2048]-row flat cache there, so the
    /// captured blocks held zeros for all 39 sliding-window layers and the
    /// restored model answered from half its context. (The tests above pin
    /// MAX_CONTEXT=8192, which sizes the cache past the ring and hid this.)
    /// </summary>
    [MuseGlimmerKvFact]
    public void MuseGlimmer_RestoringAPrefixBelowTheRingCapacity_ReproducesTheLivePrefillLogits()
    {
        string modelPath = TryFindModel();
        Assert.True(modelPath != null, "The model gate admitted this test but no Muse-Glimmer*.gguf was found.");

        using var env = new EnvScope();
        env.Set("TS_KV_INITIAL_TOKENS", "2048");
        using var model = ModelBase.Create(modelPath, ResolveBackend());

        const int prefixLen = 6 * BlockSize;
        int[] tokens = BuildPrompt(model, prefixLen + 1);
        int nextToken = tokens[prefixLen];

        model.ResetKVCache();
        model.Forward(tokens.Take(prefixLen).ToArray());
        long blockBytes = model.ComputeKVBlockByteSize(BlockSize);
        var blocks = new byte[prefixLen / BlockSize][];
        for (int b = 0; b < blocks.Length; b++)
        {
            blocks[b] = new byte[blockBytes];
            Assert.True(model.TryExtractKVBlock(b * BlockSize, BlockSize, blocks[b]), $"capture of block {b} failed");
        }
        float[] baseline = (float[])model.Forward(new[] { nextToken }).Clone();

        ScrambleCache(model, prefixLen + 1);
        for (int b = 0; b < blocks.Length; b++)
            Assert.True(model.TryInjectKVBlock(b * BlockSize, BlockSize, blocks[b]), $"restore of block {b} failed");
        float[] restored = (float[])model.Forward(new[] { nextToken }).Clone();

        float maxAbs = MaxAbsDiff(baseline, restored);
        _output.WriteLine($"restored {prefixLen} tokens at the default capacity: max |delta| {maxAbs:E3}, " +
            $"argmax live={ArgMax(baseline)} restored={ArgMax(restored)}");
        Assert.True(maxAbs == 0f,
            $"Restored logits differ from the live prefill (max |delta| {maxAbs:E3}); the snapshot is lossy.");
    }

    /// <summary>
    /// Decode across the full layers' first grows (256 -> 512 -> 1024) with the
    /// real weights and compare every step with a cache that was big enough from
    /// the start. Before the fix the first grow dropped every row the fused kernel
    /// had written on the device, and greedy decode fell into a one-token loop -
    /// on ggml_metal at position 2049, where its default 2048-row cache first grows.
    /// </summary>
    [MuseGlimmerKvFact]
    public void MuseGlimmer_DecodingAcrossACacheGrow_MatchesAPresizedCache()
    {
        string modelPath = TryFindModel();
        Assert.True(modelPath != null, "The model gate admitted this test but no Muse-Glimmer*.gguf was found.");

        const int promptLen = 200, steps = 360;   // positions 200..559: grows at 256 and 512
        int[] prompt = null;
        // A bit-exact comparison, so a digest per step stands in for 202K logits.
        List<(string Digest, int ArgMax)> Run(string initialTokens, int[] forced, out int[] chosen)
        {
            using var env = new EnvScope();
            env.Set("TS_KV_INITIAL_TOKENS", initialTokens);
            using var model = ModelBase.Create(modelPath, ResolveBackend());
            prompt ??= BuildPrompt(model, promptLen);
            model.ResetKVCache();
            var digests = new List<(string, int)>(steps);
            chosen = new int[steps];
            float[] current = model.Forward(prompt);
            for (int i = 0; i < steps; i++)
            {
                var bytes = new byte[current.Length * sizeof(float)];
                Buffer.BlockCopy(current, 0, bytes, 0, bytes.Length);
                digests.Add((Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(bytes)), ArgMax(current)));
                chosen[i] = forced != null ? forced[i] : ArgMax(current);
                if (i + 1 < steps)
                    current = model.Forward(new[] { chosen[i] });
            }
            return digests;
        }

        var presized = Run("8192", null, out int[] reference);
        var grown = Run("256", reference, out _);

        for (int i = 0; i < steps; i++)
        {
            Assert.True(presized[i].Digest == grown[i].Digest,
                $"Step {i} (position {promptLen + i}) differs from the presized cache (argmax {presized[i].ArgMax} " +
                $"vs {grown[i].ArgMax}): the grow lost or moved K/V rows.");
        }
        _output.WriteLine($"{steps} decode steps across two grows matched the presized cache exactly.");
    }

    // ------------------------------------------------------------------

    /// <summary>
    /// Overwrite the first <paramref name="rows"/> positions - host rows AND their
    /// device copies - with unrelated K/V, then empty the cache again, so a restore
    /// has to put every row back itself to reproduce the live logits.
    /// </summary>
    private static void ScrambleCache(ModelBase model, int rows)
    {
        model.ResetKVCache();
        int[] other = BuildPrompt(model, rows);
        Array.Reverse(other);
        model.Forward(other);
        model.ResetKVCache();
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float max = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            float d = Math.Abs(a[i] - b[i]);
            if (float.IsNaN(d)) return float.NaN;
            if (d > max) max = d;
        }
        return max;
    }

    /// <summary>Tile a real tokenization up to <paramref name="length"/> tokens.
    /// Content does not matter here - only that the positions are real.</summary>
    private static int[] BuildPrompt(ModelBase model, int length)
    {
        var seed = model.Tokenizer
            .Encode("The quick brown fox jumps over the lazy dog. 请详细介绍最终幻想7。", addSpecial: true)
            .ToArray();
        var tokens = new List<int>(length);
        while (tokens.Count < length)
            tokens.AddRange(seed);
        return tokens.Take(length).ToArray();
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }

    /// <summary>The backend the tests run on, as every Muse-Glimmer real-weight test reads it (<see cref="MuseGlimmerParity.Backend"/>):
    /// TS_MUSE_GLIMMER_BACKEND, else the pinned GGML backend. It used to default to ggml_cuda and to fall back to it on a name it
    /// could not parse, so on the Metal and CPU lanes every test here skipped as "another GGML backend".</summary>
    internal static BackendType ResolveBackend() => MuseGlimmerParity.Backend();

    private string TryFindModel()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrWhiteSpace(dir) || !Directory.Exists(dir))
        {
            _output.WriteLine($"{EnvModelDir} not set or missing - skipping.");
            return null;
        }

        string path = Directory.EnumerateFiles(dir, "*.gguf", SearchOption.TopDirectoryOnly)
            .Where(p => Path.GetFileName(p).StartsWith("Muse-Glimmer", StringComparison.OrdinalIgnoreCase))
            .Where(p => !Path.GetFileName(p).StartsWith("mmproj", StringComparison.OrdinalIgnoreCase))
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault();

        if (path == null)
            _output.WriteLine($"No Muse-Glimmer*.gguf under {dir} - skipping.");
        return path;
    }
}
