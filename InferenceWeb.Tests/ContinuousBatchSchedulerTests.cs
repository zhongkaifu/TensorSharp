// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>
/// Unit tests for the new paged KV pool, the continuous-batching scheduler,
/// the executor and the engine glue. They use a deterministic
/// <see cref="StubModel"/> that produces predictable logits and supports the
/// KV-snapshot contract, so we can drive the engine without loading a real
/// LLM.
/// </summary>
public class ContinuousBatchSchedulerTests
{
    private const int BlockSize = 8;
    private const int VocabSize = 16;
    private const int NumLayers = 2;
    private const int NumKVHeads = 2;
    private const int HeadDim = 4;

    [Fact]
    public void BlockPool_AllocateAndFree_RoundTrips()
    {
        var pool = NewPool(numBlocks: 4);
        var blocks = pool.AllocateNew(2);
        Assert.NotNull(blocks);
        Assert.Equal(2, blocks.Length);
        Assert.Equal(2, pool.NumFreeBlocks);

        foreach (var b in blocks)
            Assert.Equal(1, b.RefCount);

        pool.Free(blocks);
        Assert.Equal(4, pool.NumFreeBlocks);
        foreach (var b in blocks)
            Assert.Equal(0, b.RefCount);
    }

    [Fact]
    public void BlockPool_RefCountedShare_IsThreadSafeInSerial()
    {
        var pool = NewPool(numBlocks: 2);
        var b = pool.AllocateNew(1)[0];
        Assert.Equal(1, b.RefCount);
        pool.Touch(b);
        Assert.Equal(2, b.RefCount);
        pool.Free(b);
        Assert.Equal(1, b.RefCount);
        Assert.Equal(1, pool.NumFreeBlocks);
        pool.Free(b);
        Assert.Equal(0, b.RefCount);
        Assert.Equal(2, pool.NumFreeBlocks);
    }

    [Fact]
    public void BlockPool_RegistersHash_EnablesPrefixHit()
    {
        var pool = NewPool(numBlocks: 4);
        var b = pool.AllocateNew(1)[0];
        var hash = KvBlockHasher.ComputeBlockHashes(
            Enumerable.Range(0, BlockSize).ToList(), BlockSize, "fp")[0];
        pool.RegisterFullBlock(b, hash, BlockSize);

        Assert.True(pool.TryFindByHash(hash, out var found));
        Assert.Same(b, found);
    }

    [Fact]
    public void BlockPool_RestorableDuplicateWins_AndRecyclingStaleDuplicateKeepsMapping()
    {
        var pool = NewPool(numBlocks: 2);
        var blocks = pool.AllocateNew(2);
        var stale = blocks[0];
        var checkpoint = blocks[1];
        var hash = KvBlockHasher.ComputeBlockHashes(
            Enumerable.Range(0, BlockSize).ToList(), BlockSize, "fp-duplicate")[0];

        pool.RegisterFullBlock(
            stale, hash, BlockSize, isRestorablePrefixEnd: false);
        pool.RegisterFullBlock(
            checkpoint, hash, BlockSize, isRestorablePrefixEnd: true);

        Assert.True(pool.TryFindByHash(hash, out var preferred));
        Assert.Same(checkpoint, preferred);

        // Recycling the stale duplicate evicts its own metadata.  It must not
        // unregister the newer checkpoint that now owns the hash-index entry.
        pool.Free(stale);
        var recycled = pool.AllocateNew(1)[0];
        Assert.Same(stale, recycled);
        Assert.Null(recycled.ContentHash);
        Assert.Equal(0, recycled.Used);
        Assert.True(recycled.IsRestorablePrefixEnd);
        Assert.True(pool.TryFindByHash(hash, out preferred));
        Assert.Same(checkpoint, preferred);
    }

    [Fact]
    public void BlockTable_AdvanceTokens_TracksUsedBlocks()
    {
        var pool = NewPool(numBlocks: 4);
        var seq = new SequenceState("r0", new[] { 1, 2, 3, 4, 5 }, maxNewTokens: 5, BlockSize, SamplingConfig.Default);
        var blocks = pool.AllocateNew(1);
        seq.BlockTable.AppendBlock(blocks[0]);
        seq.AdvanceComputedTokens(5);
        Assert.Equal(5, seq.NumComputedTokens);
        Assert.Equal(BlockSize, seq.BlockTable.CapacityTokens);
        Assert.Equal(BlockSize - 5, seq.BlockTable.FreeSlotsInCurrentBlocks);
    }

    [Fact]
    public void Scheduler_AdmitsWaitingSequence_AllocatesBlocks()
    {
        var pool = NewPool(numBlocks: 4);
        var sched = NewScheduler(pool, "fp");
        var seq = NewSequence("r0", promptLen: 12, maxNew: 4);
        sched.Submit(seq);

        var step = sched.Schedule();
        Assert.Single(step.ScheduledWork);
        Assert.Equal(seq, step.ScheduledWork[0].Sequence);
        Assert.True(step.ScheduledWork[0].IsPrefill);
        Assert.True(step.ScheduledWork[0].NumScheduledTokens > 0);
        Assert.Equal(SequenceStatus.Running, seq.Status);
        Assert.True(seq.BlockTable.NumBlocks >= 2);
    }

    [Fact]
    public void Scheduler_MultipleSequences_RunConcurrently()
    {
        var pool = NewPool(numBlocks: 16);
        var sched = NewScheduler(pool, "fp");

        var seqs = new List<SequenceState>();
        for (int i = 0; i < 3; i++)
        {
            var s = NewSequence($"r{i}", promptLen: 5 + i, maxNew: 4);
            sched.Submit(s);
            seqs.Add(s);
        }

        var step = sched.Schedule();
        Assert.Equal(3, step.ScheduledWork.Count);
        Assert.All(seqs, s => Assert.Equal(SequenceStatus.Running, s.Status));
    }

    [Fact]
    public void Scheduler_RecurrentPrefix_BacktracksToLastRestorableEndpoint()
    {
        const string fingerprint = "fp-recurrent-prefix";
        var pool = NewPool(numBlocks: 12);
        var sched = NewScheduler(pool, fingerprint, requiresPerBlockCapture: true);
        int[] prompt = Enumerable.Range(1, 4 * BlockSize + 1).ToArray();
        var hashes = KvBlockHasher.ComputeBlockHashes(prompt, BlockSize, fingerprint);
        var cached = pool.AllocateNew(4);

        // Model one 3-block fused prefill followed by a later interior block:
        // the first two blocks have usable attention slices but their recurrent
        // payload belongs to a later boundary; block 2 is the last real
        // checkpoint, while block 3 must not be the adopted endpoint.
        bool[] restorable = { false, false, true, false };
        for (int i = 0; i < cached.Length; i++)
        {
            pool.RegisterFullBlock(
                cached[i], hashes[i], BlockSize,
                isRestorablePrefixEnd: restorable[i]);
        }

        var seq = NewSequenceFromTokens("reuse", prompt, maxNew: 1);
        sched.Submit(seq);
        var step = sched.Schedule();

        Assert.Single(step.ScheduledWork);
        Assert.Equal(3 * BlockSize, seq.PrefixCacheReusedTokens);
        Assert.Equal(2, cached[0].RefCount);
        Assert.Equal(2, cached[1].RefCount);
        Assert.Equal(2, cached[2].RefCount);
        Assert.Equal(1, cached[3].RefCount); // scan-only: no leaked Touch past endpoint
        Assert.Same(cached[0], seq.BlockTable.Blocks[0]);
        Assert.Same(cached[1], seq.BlockTable.Blocks[1]);
        Assert.Same(cached[2], seq.BlockTable.Blocks[2]);
        Assert.DoesNotContain(cached[3], seq.BlockTable.Blocks);
    }

    [Fact]
    public void Scheduler_RecurrentPrefix_WithNoRestorableEndpoint_ReusesNothing()
    {
        const string fingerprint = "fp-no-recurrent-checkpoint";
        var pool = NewPool(numBlocks: 8);
        var sched = NewScheduler(pool, fingerprint, requiresPerBlockCapture: true);
        int[] prompt = Enumerable.Range(1, 2 * BlockSize + 1).ToArray();
        var hashes = KvBlockHasher.ComputeBlockHashes(prompt, BlockSize, fingerprint);
        var cached = pool.AllocateNew(2);
        for (int i = 0; i < cached.Length; i++)
        {
            pool.RegisterFullBlock(
                cached[i], hashes[i], BlockSize,
                isRestorablePrefixEnd: false);
        }

        var seq = NewSequenceFromTokens("reuse-none", prompt, maxNew: 1);
        sched.Submit(seq);
        sched.Schedule();

        Assert.Equal(0, seq.PrefixCacheReusedTokens);
        Assert.All(cached, block => Assert.Equal(1, block.RefCount));
        Assert.DoesNotContain(cached[0], seq.BlockTable.Blocks);
        Assert.DoesNotContain(cached[1], seq.BlockTable.Blocks);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Scheduler_ExplicitCacheNone_AdoptsNoPrefixBlocks(bool useZeroBreakpoint)
    {
        const string fingerprint = "fp-explicit-none-adoption";
        var pool = NewPool(numBlocks: 8);
        var sched = NewScheduler(pool, fingerprint);
        int[] prompt = Enumerable.Range(1, 2 * BlockSize + 1).ToArray();
        var hashes = KvBlockHasher.ComputeBlockHashes(prompt, BlockSize, fingerprint);
        var cached = pool.AllocateNew(2);
        for (int i = 0; i < cached.Length; i++)
            pool.RegisterFullBlock(cached[i], hashes[i], BlockSize);

        IReadOnlyList<int> explicitNone = useZeroBreakpoint
            ? new[] { 0 }
            : Array.Empty<int>();
        var seq = new SequenceState(
            $"reuse-none-{useZeroBreakpoint}", prompt, maxNewTokens: 1,
            BlockSize, SamplingConfig.Default, cacheBreakpoints: explicitNone);

        sched.Submit(seq);
        var step = sched.Schedule();

        Assert.Single(step.ScheduledWork);
        Assert.NotNull(seq.CacheBreakpoints);
        Assert.Equal(0, seq.CacheBreakpointLimit);
        Assert.Equal(0, seq.PrefixCacheReusedTokens);
        Assert.All(cached, block => Assert.Equal(1, block.RefCount));
        Assert.DoesNotContain(cached[0], seq.BlockTable.Blocks);
        Assert.DoesNotContain(cached[1], seq.BlockTable.Blocks);
    }

    [Fact]
    public void Scheduler_ExplicitBreakpoint_CapsPrefixBlockAdoption()
    {
        const string fingerprint = "fp-capped-adoption";
        var pool = NewPool(numBlocks: 12);
        var sched = NewScheduler(pool, fingerprint);
        int[] prompt = Enumerable.Range(1, 4 * BlockSize + 1).ToArray();
        var hashes = KvBlockHasher.ComputeBlockHashes(prompt, BlockSize, fingerprint);
        var cached = pool.AllocateNew(4);
        for (int i = 0; i < cached.Length; i++)
            pool.RegisterFullBlock(cached[i], hashes[i], BlockSize);

        var seq = new SequenceState(
            "reuse-capped", prompt, maxNewTokens: 1, BlockSize, SamplingConfig.Default,
            cacheBreakpoints: new[] { 2 * BlockSize });

        sched.Submit(seq);
        var step = sched.Schedule();

        Assert.Single(step.ScheduledWork);
        Assert.Equal(2 * BlockSize, seq.PrefixCacheReusedTokens);
        Assert.Equal(2, cached[0].RefCount);
        Assert.Equal(2, cached[1].RefCount);
        Assert.Equal(1, cached[2].RefCount);
        Assert.Equal(1, cached[3].RefCount);
        Assert.Same(cached[0], seq.BlockTable.Blocks[0]);
        Assert.Same(cached[1], seq.BlockTable.Blocks[1]);
        Assert.DoesNotContain(cached[2], seq.BlockTable.Blocks);
        Assert.DoesNotContain(cached[3], seq.BlockTable.Blocks);
    }

    [Fact]
    public void Engine_DriveOneSequence_ProducesTokens()
    {
        var model = new StubModel("fp-x", peakToken: 7);
        using var engine = new InferenceEngine(model, SmallConfig(), NullLogger.Instance);

        var seq = NewSequence("r0", promptLen: 10, maxNew: 5);
        var handle = engine.SubmitRequest(seq);

        var tokens = ReadAll(handle, timeoutMs: 5000);
        Assert.True(tokens.Count > 0);
        // The stub model always picks token 7, so the output should contain 7s
        // (until EOS or max).
        Assert.Contains(7, tokens);

        var completion = handle.Completion.GetAwaiter().GetResult();
        Assert.True(completion.OutputTokenCount > 0);
    }

    [Fact]
    public void Engine_ParallelSequences_AllComplete()
    {
        var model = new StubModel("fp-par", peakToken: 7);
        using var engine = new InferenceEngine(model, SmallConfig(), NullLogger.Instance);

        var handles = new List<InferenceRequestHandle>();
        for (int i = 0; i < 4; i++)
        {
            var seq = NewSequence($"r{i}", promptLen: 6 + i, maxNew: 4);
            handles.Add(engine.SubmitRequest(seq));
        }

        foreach (var h in handles)
        {
            var completion = h.Completion.GetAwaiter().GetResult();
            Assert.True(completion.OutputTokenCount > 0,
                $"Sequence {h.RequestId} produced no tokens.");
        }
    }

    [Fact]
    public void Engine_RespectsMaxTokens()
    {
        var model = new StubModel("fp-cap", peakToken: 5);
        using var engine = new InferenceEngine(model, SmallConfig(), NullLogger.Instance);
        var seq = NewSequence("r0", promptLen: 4, maxNew: 3);
        var handle = engine.SubmitRequest(seq);

        var completion = handle.Completion.GetAwaiter().GetResult();
        Assert.Equal(3, completion.OutputTokenCount);
        Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
    }

    [Fact]
    public void Engine_StopsOnEos()
    {
        var model = new StubModel("fp-eos", peakToken: 0); // 0 is EOS
        model.SetEos(0);
        using var engine = new InferenceEngine(model, SmallConfig(), NullLogger.Instance);
        var seq = NewSequence("r0", promptLen: 4, maxNew: 10);
        var handle = engine.SubmitRequest(seq);

        var completion = handle.Completion.GetAwaiter().GetResult();
        Assert.Equal(SequenceStatus.FinishedStopped, completion.Status);
        Assert.Equal("eos", completion.FinishReason);
    }

    [Fact]
    public void Engine_PrefixCacheHit_ReducesUncomputedTokens()
    {
        var model = new StubModel("fp-prefix", peakToken: 3);
        var cfg = SmallConfig();
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);

        // Prime the cache: run a request that produces full blocks.
        var promptA = Enumerable.Range(10, BlockSize * 2).ToArray();
        var seqA = NewSequenceFromTokens("rA", promptA, maxNew: 2);
        engine.SubmitRequest(seqA).Completion.GetAwaiter().GetResult();

        // Same prompt comes in again on a NEW sequence.
        var seqB = NewSequenceFromTokens("rB", promptA, maxNew: 2);
        var handleB = engine.SubmitRequest(seqB);
        handleB.Completion.GetAwaiter().GetResult();
        // We expect at least one block worth of prefix-cache reuse.
        Assert.True(seqB.PrefixCacheReusedTokens >= BlockSize,
            $"Expected >= {BlockSize} prefix-cache tokens, got {seqB.PrefixCacheReusedTokens}");
    }

    [Fact]
    public void Engine_RecurrentLargePrefill_CachesOnlyExactChunkEndpoints()
    {
        var model = new RecurrentStubModel("fp-recurrent-engine", peakToken: 3);
        var cfg = RecurrentConfig();
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
        int[] prompt = Enumerable.Range(1, 5 * BlockSize + 5).ToArray();

        var first = NewSequenceFromTokens("recurrent-a", prompt, maxNew: 1);
        engine.SubmitRequest(first).Completion.GetAwaiter().GetResult();

        // Keep the high-throughput 32-token fused prefill.  Only the final
        // prompt chunk is split so position 40 becomes an exact checkpoint.
        Assert.Equal(
            new[] { 4 * BlockSize, BlockSize, 5 },
            model.ForwardChunkSizes.Take(3).ToArray());

        var second = NewSequenceFromTokens("recurrent-b", prompt, maxNew: 1);
        engine.SubmitRequest(second).Completion.GetAwaiter().GetResult();

        Assert.Equal(5 * BlockSize, second.PrefixCacheReusedTokens);
        Assert.Equal(new[] { 5, 1 }, model.ForwardChunkSizes.Skip(4).Take(2).ToArray());
    }

    [Fact]
    public void Engine_RecurrentExplicitBreakpoint_MakesTheMarkedPrefixRestorable()
    {
        // A client cache breakpoint lands mid-prompt, capping registration and
        // adoption at its block boundary. The prefill round must snap there so
        // the capture records a real recurrent checkpoint; otherwise the marked
        // blocks stay in the index as non-restorable interior slices and every
        // follow-up request re-prefills them (the "matched N / restorable 0"
        // warning instead of a warm cache).
        var model = new RecurrentStubModel("fp-recurrent-breakpoint", peakToken: 3);
        var cfg = RecurrentConfig();
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
        int[] prompt = Enumerable.Range(1, 5 * BlockSize + 5).ToArray();

        var first = new SequenceState(
            "bp-a", prompt, maxNewTokens: 1, BlockSize, SamplingConfig.Default,
            cacheBreakpoints: new[] { 3 * BlockSize });
        engine.SubmitRequest(first).Completion.GetAwaiter().GetResult();

        // The fused chunk is split AT the breakpoint boundary (3 blocks) so the
        // capture round ends on a genuine checkpoint, then the tail runs with
        // the usual final-chunk alignment.
        Assert.Equal(
            new[] { 3 * BlockSize, 2 * BlockSize, 5 },
            model.ForwardChunkSizes.Take(3).ToArray());

        var second = new SequenceState(
            "bp-b", prompt, maxNewTokens: 1, BlockSize, SamplingConfig.Default,
            cacheBreakpoints: new[] { 3 * BlockSize });
        engine.SubmitRequest(second).Completion.GetAwaiter().GetResult();

        // The marked prefix (everything up to the breakpoint) is fully restorable.
        Assert.Equal(3 * BlockSize, second.PrefixCacheReusedTokens);
        Assert.Equal(new[] { 2 * BlockSize, 5 }, model.ForwardChunkSizes.Skip(4).Take(2).ToArray());
    }

    [Fact]
    public void Engine_RecurrentShorterSibling_NoCheckpointInside_RefillsThenSelfHeals()
    {
        // A sibling prompt that shares only the interior blocks of a longer
        // fused-round chain matches them in the hash index but has no recurrent
        // checkpoint inside its span - those blocks carry the round-end state, so
        // adopting them would corrupt output. The safe answer is 0 reuse and a
        // re-prefill of the whole prefix; after that re-prefill the sibling's own
        // chain ends on a checkpoint and the identical follow-up reuses it.
        var model = new RecurrentStubModel("fp-recurrent-sibling", peakToken: 3);
        var cfg = RecurrentConfig();
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
        int[] longPrompt = Enumerable.Range(1, 5 * BlockSize + 5).ToArray();
        int[] shortPrompt = Enumerable.Range(1, 3 * BlockSize + 5).ToArray();

        var first = NewSequenceFromTokens("sibling-long", longPrompt, maxNew: 1);
        engine.SubmitRequest(first).Completion.GetAwaiter().GetResult();
        // One 4-block fused round + the aligned final block: blocks 0..2 are
        // interior (non-restorable), block 3 and block 4 are the checkpoints.
        Assert.Equal(new[] { 4 * BlockSize, BlockSize, 5 }, model.ForwardChunkSizes.Take(3).ToArray());

        // Shares blocks 0..2 of that chain: matched but nothing restorable.
        var sibling = NewSequenceFromTokens("sibling-short", shortPrompt, maxNew: 1);
        engine.SubmitRequest(sibling).Completion.GetAwaiter().GetResult();
        Assert.Equal(0, sibling.PrefixCacheReusedTokens);
        Assert.Equal(new[] { 3 * BlockSize, 5 }, model.ForwardChunkSizes.Skip(4).Take(2).ToArray());

        // Its re-prefill registered a checkpoint at its own end, so the
        // identical follow-up now reuses the whole prefix.
        var sibling2 = NewSequenceFromTokens("sibling-short-2", shortPrompt, maxNew: 1);
        engine.SubmitRequest(sibling2).Completion.GetAwaiter().GetResult();
        Assert.Equal(3 * BlockSize, sibling2.PrefixCacheReusedTokens);
    }

    [Fact]
    public void Executor_RecurrentOwnerSwap_PreservesFullCheckpoints_AndRefreshesPartialTail()
    {
        var model = new RecurrentStubModel("fp-recurrent-swap", peakToken: 3);
        var cfg = RecurrentConfig();
        var pool = NewPool(numBlocks: cfg.NumBlocks);
        var sched = new ContinuousBatchScheduler(
            cfg, pool, model.KVStateFingerprint, NullLogger.Instance,
            requiresPerBlockCapture: true);
        var executor = new BatchExecutor(model, pool, sched, NullLogger.Instance);

        var owner = NewSequence("owner", promptLen: 5 * BlockSize + 5, maxNew: 8);
        sched.Submit(owner);
        for (int i = 0; i < 3; i++)
        {
            var step = sched.Schedule();
            var result = executor.ExecuteStep(step);
            Assert.Single(result);
            Assert.Same(owner, result[0].Sequence);
        }
        Assert.Equal(new[] { 4 * BlockSize, BlockSize, 5 }, model.ForwardChunkSizes);
        Assert.Equal(5, model.ExtractCalls.Count); // one call per newly-full block

        var newcomer = NewSequence("newcomer", promptLen: BlockSize, maxNew: 1);
        sched.Submit(newcomer);
        var swapStep = sched.Schedule();
        var swapResult = executor.ExecuteStep(swapStep);

        Assert.Single(swapResult);
        Assert.Same(newcomer, swapResult[0].Sequence);
        // Owner swap extracts only the live partial tail (40..45); the five
        // previously-captured full checkpoints are not overwritten.  The final
        // call is newcomer's own newly-full block capture.
        Assert.Equal(
            new[] { (5 * BlockSize, 5), (0, BlockSize) },
            model.ExtractCalls.Skip(5).ToArray());
    }

    [Fact]
    public void TrimComputedToTotalTokens_RestoresInvariantAfterTailTruncation()
    {
        // Mirrors what the engine does after a mid-batch speculative stop:
        // the step advanced the committed count over the whole forwarded
        // batch, then the output tail was truncated, leaving the committed
        // count ahead of the token list.
        var seq = NewSequence("r0", promptLen: BlockSize, maxNew: 64);
        var pool = NewPool(numBlocks: 4);
        var blocks = pool.AllocateNew(3);
        foreach (var b in blocks) seq.BlockTable.AppendBlock(b);

        seq.AdvanceComputedTokens(BlockSize);          // prompt committed (block 0 full)
        for (int i = 0; i < 4; i++) seq.AppendOutputToken(200 + i);
        seq.AdvanceComputedTokens(4);                  // speculative batch: computed=12, total=12

        // Engine truncates the output tail to a single surviving token.
        seq.OutputTokens.RemoveRange(seq.OutputTokens.Count - 3, 3); // total=9, computed=12
        Assert.True(seq.NumComputedTokens > seq.NumTotalTokens);

        seq.TrimComputedToTotalTokens();
        Assert.Equal(seq.NumTotalTokens, seq.NumComputedTokens);
    }

    [Fact]
    public void NotifyStop_AfterMidBatchSpeculativeStop_DoesNotThrow()
    {
        // Regression for the --mtp-spec crash: a speculative step advances
        // NumComputedTokens past a generated block boundary, then a mid-batch
        // EOS truncates the output tail so NumComputedTokens > NumTotalTokens.
        // FinishSequence -> CacheFullBlocksForSequence used to index TokenAt()
        // past the end of the (shortened) token list and throw
        // ArgumentOutOfRangeException(pos). This drives the scheduler's
        // defense-in-depth clamp WITHOUT the engine's trim to prove the
        // finish path is crash-safe on its own.
        var pool = NewPool(numBlocks: 8);
        var sched = NewScheduler(pool, "fp");

        var seq = NewSequence("r0", promptLen: BlockSize, maxNew: 64); // prompt fills block 0
        var blocks = pool.AllocateNew(3);                              // 24-token capacity
        foreach (var b in blocks) seq.BlockTable.AppendBlock(b);

        seq.AdvanceComputedTokens(BlockSize);          // prompt committed: computed=8, total=8
        for (int i = 0; i < 6; i++) seq.AppendOutputToken(100 + i);
        seq.AdvanceComputedTokens(6);                  // decode to position 14

        // Speculative step forwards [sampled, d1, d2, d3] (all 3 drafts
        // accepted): 4 output tokens appended, committed count advanced 4 ->
        // crosses the block-1 boundary at 16.
        for (int i = 0; i < 4; i++) seq.AppendOutputToken(200 + i);
        seq.AdvanceComputedTokens(4);                  // computed=18, total=18

        // The sampled (first) token was EOS: keep only it, drop the 3 drafts.
        seq.OutputTokens.RemoveRange(seq.OutputTokens.Count - 3, 3); // total=15, computed=18
        Assert.True(seq.NumComputedTokens > seq.NumTotalTokens);

        var output = new SchedulerOutput();
        var ex = Record.Exception(
            () => sched.NotifyStop(seq, SequenceStatus.FinishedStopped, "eos", output));
        Assert.Null(ex);
        Assert.Contains("r0", output.FinishedRequestIds);
        Assert.True(seq.Status.IsFinished());
    }

    [Fact]
    public void Scheduler_ExplicitBreakpoint_RegistersOnlyBlocksInsideTheMarkedPrefix()
    {
        const string fingerprint = "fp-explicit-breakpoint";
        var pool = NewPool(numBlocks: 8);
        var sched = NewScheduler(pool, fingerprint);

        // 4 full blocks of prompt, with the client marking the end of block 1.
        int[] prompt = Enumerable.Range(1, 4 * BlockSize).ToArray();
        var seq = new SequenceState(
            "bp", prompt.ToList(), maxNewTokens: 1, BlockSize, SamplingConfig.Default,
            cacheBreakpoints: new List<int> { 2 * BlockSize });

        var blocks = pool.AllocateNew(4);
        foreach (var b in blocks)
            seq.BlockTable.AppendBlock(b);
        seq.AdvanceComputedTokens(prompt.Length);

        sched.OnBlocksCommitted(seq, previousTokens: 0);

        var hashes = KvBlockHasher.ComputeBlockHashes(prompt.ToList(), BlockSize, fingerprint);

        // Blocks 0 and 1 end at or before the breakpoint and are cacheable.
        Assert.True(pool.TryFindByHash(hashes[0], out _), "Block 0 should be registered.");
        Assert.True(pool.TryFindByHash(hashes[1], out _), "Block 1 should be registered.");

        // Blocks 2 and 3 lie past it and must stay out of the index.
        Assert.False(pool.TryFindByHash(hashes[2], out _), "Block 2 is past the breakpoint.");
        Assert.False(pool.TryFindByHash(hashes[3], out _), "Block 3 is past the breakpoint.");
    }

    [Fact]
    public void Scheduler_ExplicitBreakpointMidBlock_DropsTheStraddlingBlock()
    {
        const string fingerprint = "fp-breakpoint-midblock";
        var pool = NewPool(numBlocks: 8);
        var sched = NewScheduler(pool, fingerprint);

        int[] prompt = Enumerable.Range(1, 3 * BlockSize).ToArray();
        // A breakpoint halfway through block 1: the index is block-granular, so
        // block 1 holds tokens from both sides of the mark and is not cacheable.
        var seq = new SequenceState(
            "bp-mid", prompt.ToList(), maxNewTokens: 1, BlockSize, SamplingConfig.Default,
            cacheBreakpoints: new List<int> { BlockSize + (BlockSize / 2) });

        var blocks = pool.AllocateNew(3);
        foreach (var b in blocks)
            seq.BlockTable.AppendBlock(b);
        seq.AdvanceComputedTokens(prompt.Length);

        sched.OnBlocksCommitted(seq, previousTokens: 0);

        var hashes = KvBlockHasher.ComputeBlockHashes(prompt.ToList(), BlockSize, fingerprint);
        Assert.True(pool.TryFindByHash(hashes[0], out _), "Block 0 ends before the breakpoint.");
        Assert.False(pool.TryFindByHash(hashes[1], out _), "Block 1 straddles the breakpoint.");
        Assert.False(pool.TryFindByHash(hashes[2], out _), "Block 2 is past the breakpoint.");
    }

    [Fact]
    public void Scheduler_NoExplicitBreakpoints_RegistersEveryFullBlock()
    {
        const string fingerprint = "fp-no-breakpoint";
        var pool = NewPool(numBlocks: 8);
        var sched = NewScheduler(pool, fingerprint);

        int[] prompt = Enumerable.Range(1, 3 * BlockSize).ToArray();
        var seq = NewSequenceFromTokens("no-bp", prompt, maxNew: 1);

        var blocks = pool.AllocateNew(3);
        foreach (var b in blocks)
            seq.BlockTable.AppendBlock(b);
        seq.AdvanceComputedTokens(prompt.Length);

        sched.OnBlocksCommitted(seq, previousTokens: 0);

        var hashes = KvBlockHasher.ComputeBlockHashes(prompt.ToList(), BlockSize, fingerprint);
        for (int i = 0; i < 3; i++)
            Assert.True(pool.TryFindByHash(hashes[i], out _), $"Block {i} should be registered.");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Scheduler_ExplicitCacheNone_RegistersNoBlocks(bool useZeroBreakpoint)
    {
        const string fingerprint = "fp-zero-breakpoint";
        var pool = NewPool(numBlocks: 8);
        var sched = NewScheduler(pool, fingerprint);

        int[] prompt = Enumerable.Range(1, 2 * BlockSize).ToArray();
        IReadOnlyList<int> explicitNone = useZeroBreakpoint
            ? new[] { 0 }
            : Array.Empty<int>();
        var seq = new SequenceState(
            "zero-bp", prompt.ToList(), maxNewTokens: 1, BlockSize, SamplingConfig.Default,
            cacheBreakpoints: explicitNone);
        foreach (var block in pool.AllocateNew(2))
            seq.BlockTable.AppendBlock(block);
        seq.AdvanceComputedTokens(prompt.Length);

        sched.OnBlocksCommitted(seq, previousTokens: 0);

        var hashes = KvBlockHasher.ComputeBlockHashes(prompt.ToList(), BlockSize, fingerprint);
        Assert.NotNull(seq.CacheBreakpoints);
        Assert.Equal(0, seq.CacheBreakpointLimit);
        Assert.False(pool.TryFindByHash(hashes[0], out _));
        Assert.False(pool.TryFindByHash(hashes[1], out _));
    }

    [Fact]
    public void SequenceState_CopiesCacheBreakpoints_SoLaterCallerEditsCannotDrift()
    {
        var supplied = new List<int> { 2 * BlockSize };
        var seq = new SequenceState(
            "copy", Enumerable.Range(1, 4 * BlockSize).ToList(), maxNewTokens: 1,
            BlockSize, SamplingConfig.Default, cacheBreakpoints: supplied);

        // The pipeline hands over the same List it edited while truncating; a
        // later edit must not move the limit the scheduler already read.
        supplied.Clear();
        supplied.Add(99 * BlockSize);

        Assert.Equal(2 * BlockSize, seq.CacheBreakpointLimit);
        Assert.Equal(new[] { 2 * BlockSize }, seq.CacheBreakpoints);
    }

    // ----------------------- helpers -----------------------

    private static BlockPool NewPool(int numBlocks)
    {
        long blockBytes = 2L * NumLayers * NumKVHeads * BlockSize * HeadDim * sizeof(float);
        return new BlockPool(numBlocks, BlockSize, blockBytes);
    }

    private static ContinuousBatchScheduler NewScheduler(
        BlockPool pool,
        string fp,
        bool requiresPerBlockCapture = false)
    {
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 1024,
            MaxNumRunningSequences = 16,
            MaxPrefillChunkSize = 64,
            NumBlocks = pool.NumBlocks,
            BlockSize = BlockSize,
            EnablePrefixCaching = true,
            DecodeQuantumTokens = BlockSize,
        };
        return new ContinuousBatchScheduler(
            cfg, pool, fp, NullLogger.Instance,
            requiresPerBlockCapture: requiresPerBlockCapture);
    }

    private static SchedulerConfig SmallConfig() => new()
    {
        MaxNumBatchedTokens = 256,
        MaxNumRunningSequences = 8,
        MaxPrefillChunkSize = 64,
        NumBlocks = 16,
        BlockSize = BlockSize,
        EnablePrefixCaching = true,
        DecodeQuantumTokens = BlockSize,
    };

    private static SchedulerConfig RecurrentConfig() => new()
    {
        MaxNumBatchedTokens = 4 * BlockSize,
        MaxNumRunningSequences = 8,
        MaxPrefillChunkSize = 4 * BlockSize,
        SoloPrefillChunkSize = 4 * BlockSize,
        NumBlocks = 32,
        BlockSize = BlockSize,
        EnablePrefixCaching = true,
        DecodeQuantumTokens = BlockSize,
    };

    private static SequenceState NewSequence(string id, int promptLen, int maxNew)
    {
        var tokens = Enumerable.Range(1, promptLen).ToList();
        return new SequenceState(id, tokens, maxNew, BlockSize, SamplingConfig.Default);
    }

    private static SequenceState NewSequenceFromTokens(string id, int[] tokens, int maxNew)
        => new(id, tokens.ToList(), maxNew, BlockSize, SamplingConfig.Default);

    private static List<int> ReadAll(InferenceRequestHandle handle, int timeoutMs)
    {
        var list = new List<int>();
        var deadline = DateTime.UtcNow.AddMilliseconds(timeoutMs);
        while (DateTime.UtcNow < deadline)
        {
            if (handle.Tokens.TryRead(out var t))
            {
                list.Add(t);
                continue;
            }
            if (handle.Completion.IsCompleted) break;
            handle.Tokens.WaitToReadAsync().AsTask().Wait(50);
        }
        return list;
    }

    /// <summary>
    /// Deterministic stub model used in the engine tests above. Returns logits
    /// that always peak at a configurable token id, supports the full
    /// KV-snapshot contract (so the executor can swap state between
    /// sequences), and exposes a no-op tokenizer that recognises a single EOS.
    /// </summary>
    private sealed class StubModel : IModelArchitecture
    {
        private readonly string _fp;
        private readonly int _peak;
        private byte[] _state = Array.Empty<byte>();
        private int _cacheSeqLen;
        private int _eos = -1;

        public StubModel(string fingerprint, int peakToken)
        {
            _fp = fingerprint;
            _peak = peakToken;
            Tokenizer = new StubTokenizer(VocabSize, this);
        }

        public ModelConfig Config { get; } = new ModelConfig { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; }
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;

        public void SetEos(int eosId) => _eos = eosId;
        public int CurrentSeqLen => _cacheSeqLen;

        public float[] Forward(int[] tokens)
        {
            // Pretend to write KV state for each input token.
            int newCount = _cacheSeqLen + tokens.Length;
            long needed = ComputeKVBlockByteSize(newCount);
            if (_state.Length < needed) Array.Resize(ref _state, (int)needed);
            for (int t = 0; t < tokens.Length; t++)
            {
                long start = ComputeKVBlockByteSize(_cacheSeqLen + t);
                long end = ComputeKVBlockByteSize(_cacheSeqLen + t + 1);
                byte v = (byte)(tokens[t] & 0xFF);
                for (long i = start; i < end; i++)
                    _state[i] = v;
            }
            _cacheSeqLen += tokens.Length;

            var logits = new float[VocabSize];
            logits[_peak] = 10.0f;
            return logits;
        }

        public void ResetKVCache() => _cacheSeqLen = 0;
        public void TruncateKVCache(int tokenCount) => _cacheSeqLen = Math.Min(_cacheSeqLen, tokenCount);
        public void Dispose() { }

        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint => _fp;

        public long ComputeKVBlockByteSize(int tokenCount)
            => 2L * NumLayers * NumKVHeads * tokenCount * HeadDim * sizeof(float);

        public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            long expected = ComputeKVBlockByteSize(tokenCount);
            if (destination.Length != expected) return false;
            if (startToken < 0 || startToken + tokenCount > _cacheSeqLen) return false;
            long startByte = ComputeKVBlockByteSize(startToken);
            new ReadOnlySpan<byte>(_state, (int)startByte, (int)expected).CopyTo(destination);
            return true;
        }

        public bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (destToken != _cacheSeqLen) return false;
            long expected = ComputeKVBlockByteSize(tokenCount);
            if (source.Length != expected) return false;
            long needed = ComputeKVBlockByteSize(destToken + tokenCount);
            if (_state.Length < needed) Array.Resize(ref _state, (int)needed);
            long startByte = ComputeKVBlockByteSize(destToken);
            source.CopyTo(new Span<byte>(_state, (int)startByte, (int)expected));
            _cacheSeqLen = destToken + tokenCount;
            return true;
        }

        private sealed class StubTokenizer : ITokenizer
        {
            private readonly StubModel _owner;
            public StubTokenizer(int vocab, StubModel owner)
            {
                _owner = owner;
                Vocab = new string[vocab];
                for (int i = 0; i < vocab; i++) Vocab[i] = i.ToString();
            }
            public string[] Vocab { get; }
            public int BosTokenId => -1;
            public int[] EosTokenIds => _owner._eos >= 0 ? new[] { _owner._eos } : Array.Empty<int>();
            public int VocabSize => Vocab.Length;
            public List<int> Encode(string text, bool addSpecial = true) => new();
            public string Decode(List<int> ids) => string.Join(",", ids);
            public void AppendTokenBytes(int tokenId, List<byte> buffer)
            {
                foreach (var b in System.Text.Encoding.UTF8.GetBytes(tokenId.ToString()))
                    buffer.Add(b);
            }
            public bool IsEos(int tokenId) => _owner._eos == tokenId;
            public int LookupToken(string tokenStr) => -1;
        }
    }

    /// <summary>
    /// Snapshot model whose attention bytes are per-token but whose recurrent
    /// payload is one running-state position.  A large Forward therefore writes
    /// the same final recurrent position into every block extracted afterwards,
    /// reproducing the hybrid Qwen/Nemotron checkpoint constraint cheaply.
    /// </summary>
    private sealed class RecurrentStubModel : IModelArchitecture
    {
        private readonly string _fp;
        private readonly int _peak;
        private byte[] _tokenState = Array.Empty<byte>();
        private int _cacheSeqLen;
        private int _recurrentPosition;

        public RecurrentStubModel(string fingerprint, int peakToken)
        {
            _fp = fingerprint;
            _peak = peakToken;
            Tokenizer = new RecurrentStubTokenizer(VocabSize);
        }

        public List<int> ForwardChunkSizes { get; } = new();
        public List<(int StartToken, int TokenCount)> ExtractCalls { get; } = new();
        public ModelConfig Config { get; } = new() { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; }
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => false;
        public bool SupportsKVStateSnapshot => true;
        public bool RequiresPerBlockCapture => true;
        public string KVStateFingerprint => _fp;

        public float[] Forward(int[] tokens)
        {
            if (_recurrentPosition != _cacheSeqLen)
            {
                throw new InvalidOperationException(
                    $"Recurrent state is at {_recurrentPosition}, attention cache is at {_cacheSeqLen}.");
            }

            ForwardChunkSizes.Add(tokens.Length);
            int needed = _cacheSeqLen + tokens.Length;
            if (_tokenState.Length < needed)
                Array.Resize(ref _tokenState, needed);
            for (int i = 0; i < tokens.Length; i++)
                _tokenState[_cacheSeqLen + i] = (byte)(tokens[i] & 0xFF);
            _cacheSeqLen = needed;
            _recurrentPosition = needed;

            var logits = new float[VocabSize];
            logits[_peak] = 10.0f;
            return logits;
        }

        public void ResetKVCache()
        {
            _cacheSeqLen = 0;
            _recurrentPosition = 0;
        }

        public void TruncateKVCache(int tokenCount) => throw new NotSupportedException();
        public void Dispose() { }

        public long ComputeKVBlockByteSize(int tokenCount)
            => tokenCount + sizeof(int);

        public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            ExtractCalls.Add((startToken, tokenCount));
            if (destination.Length != ComputeKVBlockByteSize(tokenCount)) return false;
            if (startToken < 0 || startToken + tokenCount > _cacheSeqLen) return false;

            _tokenState.AsSpan(startToken, tokenCount).CopyTo(destination);
            BitConverter.TryWriteBytes(destination[tokenCount..], _recurrentPosition);
            return true;
        }

        public bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (destToken != _cacheSeqLen) return false;
            if (source.Length != ComputeKVBlockByteSize(tokenCount)) return false;

            int needed = destToken + tokenCount;
            if (_tokenState.Length < needed)
                Array.Resize(ref _tokenState, needed);
            source[..tokenCount].CopyTo(_tokenState.AsSpan(destToken, tokenCount));
            _cacheSeqLen = needed;
            _recurrentPosition = BitConverter.ToInt32(source[tokenCount..]);
            return true;
        }

        private sealed class RecurrentStubTokenizer : ITokenizer
        {
            public RecurrentStubTokenizer(int vocab)
            {
                Vocab = Enumerable.Range(0, vocab).Select(i => i.ToString()).ToArray();
            }

            public string[] Vocab { get; }
            public int BosTokenId => -1;
            public int[] EosTokenIds => Array.Empty<int>();
            public int VocabSize => Vocab.Length;
            public List<int> Encode(string text, bool addSpecial = true) => new();
            public string Decode(List<int> ids) => string.Join(",", ids);
            public void AppendTokenBytes(int tokenId, List<byte> buffer) { }
            public bool IsEos(int tokenId) => false;
            public int LookupToken(string tokenStr) => -1;
        }
    }
}
