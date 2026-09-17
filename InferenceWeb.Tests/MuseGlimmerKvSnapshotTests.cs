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
//   TS_MUSE_GLIMMER_BACKEND  backend name (default: ggml_cuda)
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
/// tests construct (TS_MUSE_GLIMMER_BACKEND, default ggml_cuda) is not the GGML
/// backend this process pinned: the second backend could only fail to initialize.
/// </summary>
[Xunit.Sdk.TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
public sealed class MuseGlimmerKvFactAttribute : FactAttribute, Xunit.Sdk.ITraitAttribute
{
    public string RequiresValue => "Models";

    public MuseGlimmerKvFactAttribute()
        => Skip = TestGates.ModelSkip("TS_TEST_MODEL_DIR", "muse-glimmer")
            ?? TestGates.GgmlPinSkip(MuseGlimmerKvSnapshotTests.ResolveBackend());
}

public class MuseGlimmerKvSnapshotTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string EnvBackend = "TS_MUSE_GLIMMER_BACKEND";

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

        // 2. Fresh cache, replay the captured blocks, same next token.
        model.ResetKVCache();
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

    // ------------------------------------------------------------------

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

    internal static BackendType ResolveBackend()
    {
        string name = Environment.GetEnvironmentVariable(EnvBackend);
        if (string.IsNullOrWhiteSpace(name))
            return BackendType.GgmlCuda;
        return Enum.TryParse<BackendType>(name, ignoreCase: true, out var parsed)
            ? parsed
            : BackendType.GgmlCuda;
    }

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
