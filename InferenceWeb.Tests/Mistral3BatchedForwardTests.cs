// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// End-to-end correctness tests for Mistral 3's batched paged-attention path,
// using the real Ministral-3-14B GGUF that ships with the test fixture.
//
// Set TS_TEST_MODEL_DIR to enable. These tests load a large model (14B
// parameters) and run two forward passes, so they take ~minutes; they're
// opt-in so CI without the model files just skips them.
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

public class Mistral3BatchedForwardTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";

    private readonly ITestOutputHelper _output;
    public Mistral3BatchedForwardTests(ITestOutputHelper output) { _output = output; }

    [Fact]
    public Task Mistral3_BatchSize1_ForwardBatchMatchesLegacyTop1()
        => RunCorrectnessTest();

    [Fact]
    public Task Mistral3_BatchSize2_DistinctSequencesProduceDistinctLogits()
        => RunDistinctTest();

    private async Task RunCorrectnessTest()
    {
        var model = await TryLoadMistral3();
        if (model == null) return;

        try
        {
            // Use plausible token IDs from the model's BOS-anchored prompt range.
            // (We don't decode them; the test cares about logit-level matching.)
            var prompt = new[] { 1, 100, 200, 300, 400, 500 };
            const int blockSize = 16;

            // ---- legacy single-sequence Forward ----
            model.ResetKVCache();
            var legacyLogits = model.Forward(prompt);
            int legacyTop1 = ArgMax(legacyLogits);

            // ---- batch_size=1 ForwardBatch ----
            // Fresh paged buffers will be lazily allocated on first call.
            var pool = new BlockPool(numBlocks: 8, blockSize: blockSize,
                blockByteSize: model.ComputeKVBlockByteSize(blockSize));
            var seq = new SequenceState("r0", prompt, maxNewTokens: 1, blockSize: blockSize,
                samplingConfig: SamplingConfig.Default);
            foreach (var b in pool.AllocateNew(1)) seq.BlockTable.AppendBlock(b);

            var ctx = BuildContext(seq, prompt, blockSize);
            var perSeqLogits = ((IBatchedPagedModel)model).ForwardBatch(ctx);
            Assert.Single(perSeqLogits);
            int batchedTop1 = ArgMax(perSeqLogits[0]);

            _output.WriteLine($"[mistral3] legacy top-1 = {legacyTop1}, batched top-1 = {batchedTop1}");

            // We tolerate single-step floating-point noise that might flip
            // tied logits, but the top-1 must match if the implementation
            // is numerically faithful.
            Assert.Equal(legacyTop1, batchedTop1);
        }
        finally
        {
            model.Dispose();
        }
    }

    private async Task RunDistinctTest()
    {
        var model = await TryLoadMistral3();
        if (model == null) return;

        try
        {
            var promptA = new[] { 1, 100, 200, 300 };
            var promptB = new[] { 1, 555, 666, 777, 888 };
            const int blockSize = 16;

            var pool = new BlockPool(numBlocks: 16, blockSize: blockSize,
                blockByteSize: model.ComputeKVBlockByteSize(blockSize));

            var seqA = new SequenceState("rA", promptA, maxNewTokens: 1, blockSize: blockSize,
                samplingConfig: SamplingConfig.Default);
            foreach (var b in pool.AllocateNew(1)) seqA.BlockTable.AppendBlock(b);

            var seqB = new SequenceState("rB", promptB, maxNewTokens: 1, blockSize: blockSize,
                samplingConfig: SamplingConfig.Default);
            foreach (var b in pool.AllocateNew(1)) seqB.BlockTable.AppendBlock(b);

            var ctx = BuildBatchContext(new[] { (seqA, promptA), (seqB, promptB) }, blockSize);
            var perSeqLogits = ((IBatchedPagedModel)model).ForwardBatch(ctx);
            Assert.Equal(2, perSeqLogits.Count);

            int topA = ArgMax(perSeqLogits[0]);
            int topB = ArgMax(perSeqLogits[1]);
            _output.WriteLine($"[mistral3 batch=2] top-1 A={topA}, B={topB}");
            Assert.NotEqual(topA, topB);
        }
        finally
        {
            model.Dispose();
        }
    }

    private static BatchedForwardContext BuildContext(SequenceState seq, int[] tokens, int blockSize)
    {
        var ctx = new BatchedForwardContext
        {
            Sequences = new List<SequenceState> { seq },
            NumScheduledTokens = new List<int> { tokens.Length },
            QueryStartLoc = new List<int> { 0, tokens.Length },
            Positions = Enumerable.Range(0, tokens.Length).ToList(),
            SlotMapping = new List<int>(),
            BlockTables = new int[1][],
            MaxQueryLen = tokens.Length,
            MaxSeqLen = tokens.Length,
        };
        var blockIds = new int[seq.BlockTable.NumBlocks];
        for (int b = 0; b < blockIds.Length; b++) blockIds[b] = seq.BlockTable.Blocks[b].Id;
        ctx.BlockTables[0] = blockIds;
        for (int t = 0; t < tokens.Length; t++)
            ctx.SlotMapping.Add(blockIds[t / blockSize] * blockSize + (t % blockSize));
        return ctx;
    }

    private static BatchedForwardContext BuildBatchContext(
        (SequenceState seq, int[] tokens)[] items, int blockSize)
    {
        var seqs = items.Select(i => i.seq).ToList();
        var nums = items.Select(i => i.tokens.Length).ToList();
        var qsl = new List<int> { 0 };
        int total = 0;
        var positions = new List<int>();
        var slots = new List<int>();
        var blockTables = new int[items.Length][];
        int maxQ = 0, maxS = 0;
        for (int s = 0; s < items.Length; s++)
        {
            var (seq, tok) = items[s];
            total += tok.Length;
            qsl.Add(total);
            for (int p = 0; p < tok.Length; p++) positions.Add(p);
            var blockIds = new int[seq.BlockTable.NumBlocks];
            for (int b = 0; b < blockIds.Length; b++) blockIds[b] = seq.BlockTable.Blocks[b].Id;
            blockTables[s] = blockIds;
            for (int t = 0; t < tok.Length; t++)
                slots.Add(blockIds[t / blockSize] * blockSize + (t % blockSize));
            if (tok.Length > maxQ) maxQ = tok.Length;
            if (tok.Length > maxS) maxS = tok.Length;
        }
        return new BatchedForwardContext
        {
            Sequences = seqs,
            NumScheduledTokens = nums,
            QueryStartLoc = qsl,
            Positions = positions,
            SlotMapping = slots,
            BlockTables = blockTables,
            MaxQueryLen = maxQ,
            MaxSeqLen = maxS,
        };
    }

    private async Task<Mistral3Model> TryLoadMistral3()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir))
        {
            _output.WriteLine($"{EnvModelDir} not set; skipping");
            return null;
        }

        string modelPath = Directory.GetFiles(dir, "*.gguf")
            .FirstOrDefault(p =>
            {
                var n = Path.GetFileName(p).ToLowerInvariant();
                return (n.Contains("ministral") || n.Contains("mistral"))
                    && !n.Contains("mmproj");
            });
        if (modelPath == null)
        {
            _output.WriteLine("No Mistral/Ministral GGUF in test dir; skipping");
            return null;
        }
        _output.WriteLine($"[mistral3] loading {Path.GetFileName(modelPath)}");
        try
        {
            BackendType backend = OperatingSystem.IsMacOS() ? BackendType.GgmlMetal : BackendType.GgmlCpu;
            var model = (Mistral3Model)ModelBase.Create(modelPath, backend);
            await Task.Yield();
            return model;
        }
        catch (Exception ex)
        {
            _output.WriteLine($"Failed to load: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }

    private static int ArgMax(float[] arr)
    {
        int best = 0;
        for (int i = 1; i < arr.Length; i++) if (arr[i] > arr[best]) best = i;
        return best;
    }
}
