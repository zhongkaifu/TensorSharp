// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// [ModelFact] for <see cref="Qwen3BatchedForwardTests"/>: the gate and the loader
/// share <see cref="Qwen3BatchedForwardTests.FindModel"/>, so a model directory
/// without a base Qwen3 / Bonsai-8B GGUF is a visible skip rather than a failure.
/// </summary>
[Xunit.Sdk.TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
public sealed class Qwen3BatchedModelFactAttribute : FactAttribute, Xunit.Sdk.ITraitAttribute
{
    public string RequiresValue => "Models";

    public Qwen3BatchedModelFactAttribute()
        => Skip = TestGates.ModelSkip(Qwen3BatchedForwardTests.EnvModelDir)
            ?? (Qwen3BatchedForwardTests.FindModel() == null
                ? $"Requires a base Qwen3 or Bonsai-8B GGUF under {Qwen3BatchedForwardTests.EnvModelDir}."
                : null);
}

/// <summary>
/// Opt-in end-to-end checks for Qwen3/Bonsai-8B's real paged batching path.
/// Set TS_TEST_MODEL_DIR to a directory containing the model to run them.
/// </summary>
[Trait("Requires", "Models")]
public sealed class Qwen3BatchedForwardTests
{
    internal const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private readonly ITestOutputHelper _output;

    public Qwen3BatchedForwardTests(ITestOutputHelper output) => _output = output;

    [Qwen3BatchedModelFact]
    public Task BatchSize1_MatchesSingleSequenceTop1() => RunSingleSequenceAsync();

    [Qwen3BatchedModelFact]
    public Task BatchSize2_KeepsSequencesIndependent() => RunTwoSequencesAsync();

    [Qwen3BatchedModelFact]
    public Task RetainedDecodeGraph_SurvivesTruncateAndResidencyRelease() =>
        RunDecodeGraphLifecycleAsync();

    [Qwen3BatchedModelFact]
    public Task PerSequenceCache_GrowsReleasesAndReturnsToPrimary() =>
        RunPerSequenceGrowthLifecycleAsync();

    private async Task RunSingleSequenceAsync()
    {
        var model = await TryLoadModelAsync();
        try
        {
            int[] prompt = [1, 100, 200, 300, 400, 500];
            model.ResetKVCache();
            int expected = ArgMax(model.Forward(prompt));

            var pool = new BlockPool(8, 16, model.ComputeKVBlockByteSize(16));
            var sequence = CreateSequence(pool, "r0", prompt);
            var actual = ((IBatchedPagedModel)model).ForwardBatch(
                BuildContext([(sequence, prompt)], 16));

            Assert.Single(actual);
            _output.WriteLine($"[bonsai-8b] scalar top-1={expected}, batch top-1={ArgMax(actual[0])}");
            Assert.Equal(expected, ArgMax(actual[0]));
        }
        finally
        {
            model.Dispose();
        }
    }

    private async Task RunTwoSequencesAsync()
    {
        var model = await TryLoadModelAsync();
        try
        {
            int[] promptA = [1, 100, 200, 300];
            int[] promptB = [1, 555, 666, 777, 888];

            model.ResetKVCache();
            int expectedA = ArgMax(model.Forward(promptA));
            model.ResetKVCache();
            int expectedB = ArgMax(model.Forward(promptB));
            model.ResetKVCache();

            var pool = new BlockPool(16, 16, model.ComputeKVBlockByteSize(16));
            var sequenceA = CreateSequence(pool, "rA", promptA);
            var sequenceB = CreateSequence(pool, "rB", promptB);

            var logits = ((IBatchedPagedModel)model).ForwardBatch(
                BuildContext([(sequenceA, promptA), (sequenceB, promptB)], 16));

            Assert.Equal(2, logits.Count);
            int topA = ArgMax(logits[0]);
            int topB = ArgMax(logits[1]);
            _output.WriteLine(
                $"[bonsai-8b batch=2] A={topA} (serial {expectedA}), " +
                $"B={topB} (serial {expectedB})");
            Assert.Equal(expectedA, topA);
            Assert.Equal(expectedB, topB);
        }
        finally
        {
            model.Dispose();
        }
    }

    private async Task RunDecodeGraphLifecycleAsync()
    {
        var model = await TryLoadModelAsync();
        try
        {
            int[] prompt = [1, 100, 200, 300, 400, 500];

            model.ResetKVCache();
            int token = ArgMax(model.Forward(prompt));
            float[] expectedAfterToken = (float[])model.Forward([token]).Clone();

            // Truncation invalidates the cache's device buffers. A retained native
            // graph must be retired before that invalidation and rebuilt here.
            model.TruncateKVCache(prompt.Length);
            float[] afterTruncate = (float[])model.Forward([token]).Clone();
            AssertLogitsClose(expectedAfterToken, afterTruncate, "truncate");

            int nextToken = ArgMax(afterTruncate);
            float[] expectedAfterNext = (float[])model.Forward([nextToken]).Clone();

            // Warm the retained decode graph again, then evict model weights. The
            // next decode must lazily rebuild rather than replay stale bindings.
            model.ResetKVCache();
            model.Forward(prompt);
            model.Forward([token]);
            model.ReleaseGgmlDeviceResidency();
            float[] afterRelease = (float[])model.Forward([nextToken]).Clone();
            AssertLogitsClose(expectedAfterNext, afterRelease, "residency release");
        }
        finally
        {
            model.Dispose();
        }
    }

    private async Task RunPerSequenceGrowthLifecycleAsync()
    {
        var model = await TryLoadModelAsync();
        const string requestId = "bonsai-growth";
        try
        {
            Assert.True(model.BindSequenceCache(requestId));
            int[] longPrompt = Enumerable.Range(0, 2050)
                .Select(i => i == 0 ? 1 : 100 + (i * 37) % 900)
                .ToArray();

            float[] promptLogits = model.Forward(longPrompt);
            int token = ArgMax(promptLogits);
            float[] decodeLogits = model.Forward([token]);
            Assert.True(float.IsFinite(decodeLogits[ArgMax(decodeLogits)]));

            model.OnSequenceReleased(requestId);
            Assert.False(model.HasFusedSequenceCache(requestId));

            // Active release restores the untouched primary holder. It must be
            // usable immediately after the grown request and native graph die.
            model.ResetKVCache();
            float[] primaryLogits = model.Forward([1, 100, 200, 300]);
            Assert.True(float.IsFinite(primaryLogits[ArgMax(primaryLogits)]));
            _output.WriteLine(
                $"[bonsai-8b cache growth] request top-1={token}, " +
                $"restored-primary top-1={ArgMax(primaryLogits)}");
        }
        finally
        {
            if (model.HasFusedSequenceCache(requestId))
                model.OnSequenceReleased(requestId);
            model.Dispose();
        }
    }

    private static SequenceState CreateSequence(BlockPool pool, string id, int[] tokens)
    {
        var sequence = new SequenceState(id, tokens, 1, 16, SamplingConfig.Default);
        int blocks = (tokens.Length + 15) / 16;
        foreach (var block in pool.AllocateNew(blocks))
            sequence.BlockTable.AppendBlock(block);
        return sequence;
    }

    private static BatchedForwardContext BuildContext(
        (SequenceState Sequence, int[] Tokens)[] items, int blockSize)
    {
        var queryStart = new List<int> { 0 };
        var positions = new List<int>();
        var slots = new List<int>();
        var tables = new int[items.Length][];
        int total = 0;

        for (int i = 0; i < items.Length; i++)
        {
            var (sequence, tokens) = items[i];
            total += tokens.Length;
            queryStart.Add(total);
            positions.AddRange(Enumerable.Range(0, tokens.Length));

            int[] blockIds = sequence.BlockTable.Blocks.Select(block => block.Id).ToArray();
            tables[i] = blockIds;
            for (int token = 0; token < tokens.Length; token++)
                slots.Add(blockIds[token / blockSize] * blockSize + token % blockSize);
        }

        return new BatchedForwardContext
        {
            Sequences = items.Select(item => item.Sequence).ToList(),
            NumScheduledTokens = items.Select(item => item.Tokens.Length).ToList(),
            QueryStartLoc = queryStart,
            Positions = positions,
            SlotMapping = slots,
            BlockTables = tables,
            MaxQueryLen = items.Max(item => item.Tokens.Length),
            MaxSeqLen = items.Max(item => item.Tokens.Length),
        };
    }

    /// <summary>The GGUF these tests load: TS_TEST_MODEL_DIR itself when it names a
    /// file, else the first base Qwen3 / Bonsai-8B GGUF in that directory; null when
    /// there is none. Shared with <see cref="Qwen3BatchedModelFactAttribute"/>.</summary>
    internal static string FindModel()
    {
        string directory = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrWhiteSpace(directory))
            return null;
        if (File.Exists(directory))
            return directory;
        if (!Directory.Exists(directory))
            return null;
        return Directory.GetFiles(directory, "*.gguf")
            .OrderBy(candidate => candidate, StringComparer.Ordinal)
            .FirstOrDefault(candidate =>
            {
                string name = Path.GetFileName(candidate).ToLowerInvariant();
                bool baseQwen3 = name.Contains("qwen3") &&
                    !name.Contains("qwen3.5") && !name.Contains("qwen35") &&
                    !name.Contains("qwen3.6");
                return !name.Contains("mmproj") &&
                    (baseQwen3 || name.Contains("bonsai-8b"));
            });
    }

    private async Task<Qwen3Model> TryLoadModelAsync()
    {
        string path = FindModel();
        Assert.False(path is null, $"{EnvModelDir} names no base Qwen3 or Bonsai-8B GGUF (the gate should have skipped).");

        _output.WriteLine($"[bonsai-8b] loading {Path.GetFileName(path)}");
        // The pinned GGML backend: one backend per process (GgmlBackendTestInitializer).
        var model = ModelBase.Create(path, TestGates.PinnedGgmlBackend) as Qwen3Model;
        Assert.NotNull(model);
        string native = TestGates.MappedNativeGgmlOpsPath();
        string nativeSha = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(File.ReadAllBytes(native))).ToLowerInvariant();
        _output.WriteLine($"[bonsai-8b] native={native} sha256={nativeSha}");
        string expectedNative = Environment.GetEnvironmentVariable("TS_TEST_QWEN3_EXPECTED_NATIVE_SHA256");
        if (!string.IsNullOrEmpty(expectedNative)) Assert.Equal(expectedNative, nativeSha);
        await Task.Yield();
        return model;
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best])
                best = i;
        return best;
    }

    private void AssertLogitsClose(float[] expected, float[] actual, string operation)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maxAbs = 0f;
        for (int i = 0; i < expected.Length; i++)
            maxAbs = Math.Max(maxAbs, Math.Abs(expected[i] - actual[i]));

        _output.WriteLine(
            $"[bonsai-8b {operation}] top-1={ArgMax(actual)}, max |delta|={maxAbs:E3}");
        Assert.Equal(ArgMax(expected), ArgMax(actual));
        Assert.True(maxAbs <= 1e-4f,
            $"{operation} changed logits by {maxAbs:E3} (limit 1.000E-004).");
    }
}
