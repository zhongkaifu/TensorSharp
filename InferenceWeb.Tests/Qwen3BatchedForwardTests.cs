// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// End-to-end correctness tests for the real Qwen3 batched paged-attention
// forward path. These are opt-in (set TS_TEST_MODEL_DIR pointing at a
// directory that contains a Qwen3-family GGUF) because loading a real model
// is expensive. The tests verify that:
//   * a single-sequence ForwardBatch produces the same TOP-1 prediction as
//     the legacy single-sequence Forward (i.e. the batched plumbing is
//     numerically faithful);
//   * a batch_size=2 ForwardBatch produces distinct, sensible per-sequence
//     logits (i.e. K/V state is correctly partitioned by sequence).
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class Qwen3BatchedForwardTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";

    private readonly ITestOutputHelper _output;
    public Qwen3BatchedForwardTests(ITestOutputHelper output) { _output = output; }

    [Fact]
    public Task Qwen3_BatchSize1_ForwardBatchMatchesLegacyTop1()
        => RunCorrectnessTest();

    [Fact]
    public Task Qwen3_BatchSize2_DistinctSequencesProduceDistinctLogits()
        => RunBatchedDistinctTest();

    private async Task RunCorrectnessTest()
    {
        var (model, _) = await TryLoadQwen3();
        if (model == null) return;

        try
        {
            var prompt = new[] { 1, 100, 200, 300, 400, 500 };

            // ---- Legacy Forward ----
            model.ResetKVCache();
            var legacyLogits = model.Forward(prompt);
            int legacyTop1 = ArgMax(legacyLogits);

            // ---- ForwardBatch(batch_size=1) ----
            // Build a minimal BatchedForwardContext for one sequence at
            // positions [0..promptLen).
            var pool = new BlockPool(numBlocks: 8, blockSize: 16,
                blockByteSize: model.ComputeKVBlockByteSize(16));
            var seq = new SequenceState("r0", prompt, maxNewTokens: 1, blockSize: 16,
                samplingConfig: SamplingConfig.Default);
            var blocks = pool.AllocateNew(1);
            foreach (var b in blocks) seq.BlockTable.AppendBlock(b);

            var ctx = BuildContext(new[] { seq }, prompt, blockSize: 16);
            var perSeqLogits = ((IBatchedPagedModel)model).ForwardBatch(ctx);
            Assert.Single(perSeqLogits);
            int batchedTop1 = ArgMax(perSeqLogits[0]);

            _output.WriteLine(
                $"[qwen3] legacy top-1 = {legacyTop1}, batched top-1 = {batchedTop1}");
            // Strong assertion: top-1 predictions must match. (Floating-point
            // noise can flip ties but the dominant token should be the same.)
            Assert.Equal(legacyTop1, batchedTop1);
        }
        finally
        {
            model.Dispose();
        }
    }

    private async Task RunBatchedDistinctTest()
    {
        var (model, _) = await TryLoadQwen3();
        if (model == null) return;

        try
        {
            var promptA = new[] { 1, 100, 200, 300 };
            var promptB = new[] { 1, 555, 666, 777, 888 };

            var pool = new BlockPool(numBlocks: 16, blockSize: 16,
                blockByteSize: model.ComputeKVBlockByteSize(16));

            var seqA = new SequenceState("rA", promptA, maxNewTokens: 1, blockSize: 16,
                samplingConfig: SamplingConfig.Default);
            foreach (var b in pool.AllocateNew(1)) seqA.BlockTable.AppendBlock(b);

            var seqB = new SequenceState("rB", promptB, maxNewTokens: 1, blockSize: 16,
                samplingConfig: SamplingConfig.Default);
            foreach (var b in pool.AllocateNew(1)) seqB.BlockTable.AppendBlock(b);

            // Build a ctx with both sequences (each one prefill chunk).
            var ctx = BuildBatchContext(new[] { (seqA, promptA), (seqB, promptB) }, blockSize: 16);
            var perSeqLogits = ((IBatchedPagedModel)model).ForwardBatch(ctx);
            Assert.Equal(2, perSeqLogits.Count);

            int topA = ArgMax(perSeqLogits[0]);
            int topB = ArgMax(perSeqLogits[1]);
            _output.WriteLine($"[qwen3 batch=2] top-1 A={topA}, top-1 B={topB}");

            // The two prompts diverge from token 1 onwards, so their
            // top-1 predictions should differ. (If they are the same the
            // batched K/V state was leaking across sequences.)
            Assert.NotEqual(topA, topB);
        }
        finally
        {
            model.Dispose();
        }
    }

    private static BatchedForwardContext BuildContext(SequenceState[] seqs, int[] tokens, int blockSize)
    {
        // Single-sequence helper: tokens is the prompt for the only sequence.
        var ctx = new BatchedForwardContext
        {
            Sequences = new List<SequenceState>(seqs),
            NumScheduledTokens = new List<int> { tokens.Length },
            QueryStartLoc = new List<int> { 0, tokens.Length },
            Positions = Enumerable.Range(0, tokens.Length).ToList(),
            SlotMapping = new List<int>(),
            BlockTables = new int[1][],
            MaxQueryLen = tokens.Length,
            MaxSeqLen = tokens.Length,
        };
        var blockIds = new int[seqs[0].BlockTable.NumBlocks];
        for (int b = 0; b < blockIds.Length; b++) blockIds[b] = seqs[0].BlockTable.Blocks[b].Id;
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

    private async Task<(Qwen3Model model, string modelPath)> TryLoadQwen3()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir))
        {
            _output.WriteLine($"{EnvModelDir} not set; skipping");
            return (null, null);
        }

        // We need a base-Qwen3 GGUF (not Qwen3.5 / Qwen3.6 which are different
        // architectures in TensorSharp). If the test dir doesn't have one,
        // skip - this test is informational on dev machines that do.
        string modelPath = Directory.GetFiles(dir, "*.gguf")
            .FirstOrDefault(p =>
            {
                var name = Path.GetFileName(p).ToLowerInvariant();
                return name.Contains("qwen3") &&
                    !name.Contains("qwen3.5") &&
                    !name.Contains("qwen35") &&
                    !name.Contains("qwen3.6") &&
                    !name.Contains("mmproj");
            });
        if (modelPath == null)
        {
            _output.WriteLine("No base Qwen3 GGUF available; skipping");
            return (null, null);
        }

        _output.WriteLine($"[qwen3] loading {Path.GetFileName(modelPath)}");
        try
        {
            BackendType backend = OperatingSystem.IsMacOS() ? BackendType.GgmlMetal : BackendType.GgmlCpu;
            var model = (Qwen3Model)ModelBase.Create(modelPath, backend);
            await Task.Yield();
            return (model, modelPath);
        }
        catch (Exception ex)
        {
            _output.WriteLine($"Failed to load: {ex.GetType().Name}: {ex.Message}");
            return (null, null);
        }
    }

    private static int ArgMax(float[] arr)
    {
        int best = 0;
        for (int i = 1; i < arr.Length; i++) if (arr[i] > arr[best]) best = i;
        return best;
    }
}
